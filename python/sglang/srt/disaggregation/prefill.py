"""
Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request

中译：PD（Prefill-Decode）分离部署下，prefill 节点上一个请求的完整生命周期，横跨三个队列：

1. Bootstrap 队列（PrefillBootstrapQueue）
    a. 为每个请求初始化一个 KV sender（负责后续把 KV 缓存发往 decode 节点）；
    b. 队列中暂存那些 bootstrap（与 decode 节点握手 + 预分配元数据 buffer）尚未完成的请求；
    c. 轮询各 sender 检查 bootstrap 状态；
    d. 一旦 bootstrap 完成，把请求转入 Waiting 队列（等待调度前向）。

2. Waiting 队列（调度器的 waiting_queue）
    a. 由 PrefillAdder 依据显存等约束从队列取出请求组批；
    b. 运行前向（即 prefill 计算，产出首个 next token）；
    c. 把请求加入 Inflight（在途传输）队列。

3. Inflight 队列（disagg_prefill_inflight_queue）
    a. 非阻塞地轮询请求 sender 的 KV 传输状态；
    b. 一旦传输完成，把请求返回（回收资源、向客户端流式返回结果）。
"""

from __future__ import annotations

import hashlib
import logging
from array import array
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.staging_buffer import (
    compute_grid_segments,
    staging_grid_tokens,
)
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    build_kv_layer_ids,
    build_staging_slot_metadata,
    get_dsv4_c128_state_indices,
    get_kv_class,
    is_aborted,
    is_dsv4_c128_online_enabled,
    is_mla_backend,
    poll_and_all_reduce_attn_cp_tp_group,
    poll_and_all_reduce_pp,
    prepare_abort,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import (
    kv_to_page_indices,
    kv_to_page_num,
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.runtime_context import (
    get_disagg,
    get_parallel,
    get_schedule,
)
from sglang.srt.utils import is_npu
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)

_is_npu = is_npu()


def should_force_retry(req: Req) -> bool:
    """Test hook to force a request into optimistic prefill retry.

    中译：仅用于测试的钩子，按概率强制将请求推入「乐观 prefill 重试」路径，
          用于验证重试逻辑的正确性。仅在该环境变量概率 > 0、请求尚未重试过、
          且未被回撤时生效；基于 rid 的哈希值作确定性抽样。
    """
    retry_prob = envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
    # Force only before/during the first attempt (count is 1 while it runs).
    if retry_prob <= 0 or req.prefill_attempt_count > 1 or req.is_retracted:
        return False

    # 中译：取 rid 的 SHA256 前 8 字节作为无符号整数，与阈值比较实现「按概率命中」。
    digest = hashlib.sha256(str(req.rid).encode()).digest()
    return int.from_bytes(digest[:8], "big") < retry_prob * 2**64


def _transfer_start_layer(*, pool, hf_text_config) -> int:
    # Hybrid pools count all layers in start_layer, but peer KV lists contain only
    # full-attention layers, so translate to a full-attention-relative offset.
    if not isinstance(pool, HybridLinearKVPool):
        return pool.start_layer
    return sum(
        1 for lid in hf_text_config.full_attention_layer_ids if lid < pool.start_layer
    )


def maybe_release_metadata_buffer(
    req: Req, allocator: ReqToMetadataIdxAllocator
) -> None:
    """
    Release the metadata buffer index allocated for a request in prefill disaggregation mode.

    This function safely releases the metadata buffer index if it was allocated.

    Args:
        req: The request object that may have a metadata_buffer_index allocated
        allocator: The ReqToMetadataIdxAllocator instance to free the index

    中译：在 prefill 分离模式下，释放为某请求分配的 metadata buffer 索引。
          若该索引已分配则安全释放。
    参数：
        req: 可能持有 metadata_buffer_index 的请求对象
        allocator: 用于释放该索引的 ReqToMetadataIdxAllocator 实例
    """
    # 中译：仅当请求已分配 metadata buffer（index >= 0）时才释放，并把 index 置 -1 防止重复释放。
    if req.metadata_buffer_index >= 0:
        allocator.free(req.metadata_buffer_index)
        req.metadata_buffer_index = -1


class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping

    中译：Bootstrap 阶段的请求队列。负责管理 KV 传输所需的底层资源（KVManager、
          KV sender、metadata buffer），并驱动请求从「握手/预分配中」推进到「已就绪」，
          再交给调度器的 waiting_queue。
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        scheduler: Scheduler,
        pp_rank: int,
        pp_size: int,
        transfer_backend: TransferBackend,
    ):
        # 中译：主模型的 KV cache 池（token -> KV 存储），KV 传输的数据源，其显存地址/
        #       布局会注册给 KVManager。
        self.token_to_kv_pool = token_to_kv_pool
        # 中译：draft 模型的 KV cache 池（EAGLE 等投机解码时存在），需与主模型 KV 一并传输；
        #       无投机解码时为 None。
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        # 中译：是否为 MLA 后端（据 KV 池类型判定）。MLA 的 KV 组织方式不同，影响传输参数与 staging 开关。
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        # 中译：元数据 buffer（承载随 KV 一同传输的 aux 数据，如首 token、hidden states 等）。
        self.metadata_buffers = metadata_buffers
        # 中译：metadata buffer 槽位分配器——为每个请求分配/回收一个 buffer 索引。
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        # 中译：本进程在张量并行（TP）维度下的 rank 序号与总规模。
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        # 中译：本进程在流水线并行（PP）维度下的 rank 序号与总规模。
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        # 中译：本进程绑定的 GPU 设备号。
        self.gpu_id = gpu_id
        # 中译：bootstrap server 的端口（与 decode 侧建立关联/握手的控制面端口）。
        self.bootstrap_port = bootstrap_port
        # 中译：bootstrap 队列本体——暂存所有握手/预分配尚未完成的请求，按 poll 状态逐轮推进。
        self.queue: List[Req] = []
        # 中译：gloo 进程组，用于跨 rank 的 CPU 侧集合通信/同步。
        self.gloo_group = gloo_group
        # 中译：所属调度器实例，用于访问 tp_worker、tree_cache、失败处理、输出流等公共设施。
        self.scheduler = scheduler
        # 中译：KV 池可容纳的最大 token 数，作为单请求输入长度的容量上限（超限则中止请求）。
        self.max_total_num_tokens = (
            self.scheduler.tp_worker.model_runner.effective_max_total_num_tokens
        )
        # 中译：KV 传输后端类型（Mooncake/NIXL/Mori/Ascend 等），决定实际使用的 KVManager/Sender 实现。
        self.transfer_backend = transfer_backend
        # 中译：staging buffer 是面向非 MLA 模型（GQA/MHA）的中转发送优化，MLA 模型不应开启。
        # 中译：初始化底层 KV 传输管理器（负责注册显存 buffer、维护握手/传输通道）。
        if envs.SGLANG_DISAGG_STAGING_BUFFER.get():
            if self.is_mla_backend:
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                    "(e.g. GQA, MHA). MLA models should not set this flag."
                )
            page_size = self.scheduler.token_to_kv_pool_allocator.page_size
            # Same source as send_kv_chunk's staging grid below, so validation
            # and the grid cannot disagree after a post-publish override.
            chunked_prefill_size = get_schedule().chunked_prefill_size
            cps = chunked_prefill_size or 8192
            # Staging slices each send into a fixed page-aligned grid, so an
            # unbounded (-1) or non-page-aligned chunk size has no valid grid.
            if cps <= 0 or cps % page_size != 0:
                raise RuntimeError(
                    f"SGLANG_DISAGG_STAGING_BUFFER requires a positive "
                    f"chunked_prefill_size that is a multiple of page_size "
                    f"({page_size}); got {chunked_prefill_size}."
                )
            if self.pp_size > 1 and self.transfer_backend != TransferBackend.MOONCAKE:
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER with pp_size > 1 is only "
                    "supported by Mooncake."
                )
            if get_parallel().enable_prefill_context_parallel:
                # CP rewrites index_slice per rank, breaking the chunk grid.
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER does not support "
                    "prefill context parallelism."
                )
        self.kv_manager = self._init_kv_manager()

    def _init_kv_manager(self) -> CommonKVManager:
        # 中译：根据传输后端选择对应的 KVArgs 类，并逐项填充 KV 传输所需的元信息：
        #       rank 信息、KV pool 的显存地址/长度、head 数、page size、辅助 metadata
        #       buffer 地址等，最终据此构造并返回 KVManager。
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.ps.dp_rank
        kv_args.kv_cache_dtype_str = (
            self.scheduler.tp_worker.model_runner.kv_cache_dtype_str
        )
        layer_shard_enabled = getattr(
            self.token_to_kv_pool, "layer_shard_enabled", False
        )
        layer_shard_rank = getattr(self.token_to_kv_pool, "layer_shard_rank", None)
        layer_shard_size = getattr(self.token_to_kv_pool, "layer_shard_size", 1)
        transfer_draft_cache = (
            not layer_shard_enabled or layer_shard_rank == layer_shard_size - 1
        )
        kv_args.prefill_start_layer = (
            getattr(
                self.token_to_kv_pool,
                "layer_shard_start",
                self.token_to_kv_pool.start_layer,
            )
            if layer_shard_enabled
            else _transfer_start_layer(
                pool=self.token_to_kv_pool,
                hf_text_config=self.scheduler.model_config.hf_text_config,
            )
        )
        kv_args.mla_compression_ratios = None
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )
        kv_args.prefill_end_layer = (
            kv_args.prefill_start_layer + len(kv_data_ptrs)
            if layer_shard_enabled
            else getattr(self.token_to_kv_pool, "end_layer", None)
        )

            # 中译：也需传输 draft 模型的 KV cache；其索引总是与目标（主）模型共享。
        draft_kv_pool = self.draft_token_to_kv_pool if transfer_draft_cache else None
        num_draft_entries = 0
        if draft_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                draft_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens
            num_draft_entries = len(draft_kv_data_ptrs)

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.kv_layer_ids = build_kv_layer_ids(
            token_to_kv_pool=self.token_to_kv_pool,
            draft_token_to_kv_pool=draft_kv_pool,
            num_draft_entries=num_draft_entries,
            num_hidden_layers=self.scheduler.model_config.num_hidden_layers,
        )
        if not self.is_mla_backend:
            kv_args.kv_head_num = self.token_to_kv_pool.head_num
            kv_args.total_kv_head_num = (
                self.scheduler.model_config.get_total_num_kv_heads()
            )
        kv_args.page_size = self.token_to_kv_pool.page_size

        # 中译：aux（辅助）buffer 用于承载随 KV 一同传输的元数据（如首 token、hidden states 等）。
        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = get_disagg().disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.ps.gpu_id

        req_to_token_pool = getattr(self.scheduler, "req_to_token_pool", None)
        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool if transfer_draft_cache else None,
            self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=req_to_token_pool,
        )

        if isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
            # 中译：V4 的 KVCache 按压缩率桶（compression-ratio buckets）而非按层组织。
            kv_args.mla_compression_ratios = list(
                self.token_to_kv_pool.compression_ratios
            )

        # 中译：以 PREFILL 角色实例化 KVManager（内部会启动握手/传输所需的服务）。
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # 中译：（staging 模式）把 KV 池的张量引用传给 manager，供 GPU 侧 gather 使用。
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and hasattr(kv_manager, "set_kv_buffer_tensors")
            and not self.is_mla_backend
        ):
            kv_pool = self.token_to_kv_pool
            if hasattr(kv_pool, "full_kv_pool"):
                kv_pool = kv_pool.full_kv_pool
            staging_slots = build_staging_slot_metadata(
                kv_layer_ids=kv_args.kv_layer_ids,
                num_draft_entries=num_draft_entries,
                kv_pool=kv_pool,
                draft_kv_pool=draft_kv_pool,
            )
            if staging_slots is not None:
                k_buffers, v_buffers, slot_layer_ids = staging_slots
                kv_manager.set_kv_buffer_tensors(
                    k_buffers,
                    v_buffers,
                    kv_pool.page_size,
                    slot_layer_ids=slot_layer_ids,
                )
        return kv_manager

    def create_sender(self, req: Req, num_kv_heads: int) -> bool:
        """Create a KV sender for the request without enqueuing it.
        Returns False if the request exceeds KV capacity.

        中译：为请求创建 KV sender（但不入队）。若请求超过 KV 容量则返回 False。
        """
        if self._check_if_req_exceed_kv_capacity(req):
            return False

        # 中译：若为测试用的假 bootstrap host，则选用 FAKE 后端（不真正发送），否则用真实后端。
        backend = (
            TransferBackend.FAKE
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST
            else self.transfer_backend
        )
        kv_sender_class = get_kv_class(backend, KVClassType.SENDER)

        dest_tp_ranks = [self.tp_rank]

        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=self.pp_rank,
            req_has_disagg_prefill_dp_rank=req.disagg_prefill_dp_rank is not None,
        )
        # 中译：把 max_new_tokens 强制为 1（prefill 节点只产首 token），并标记请求进入 pending_bootstrap。
        self._process_req(req)
        req.pending_bootstrap = True
        return True

    def ensure_metadata_buffer(self, req: Req) -> bool:
        # 中译：确保请求拿到一个 metadata buffer 槽位（用于承载随 KV 传输的元数据）。
        #       已分配则直接返回 True；无空闲槽位则返回 False（调用方稍后重试）。
        if req.metadata_buffer_index >= 0:
            return True

        if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
            return False
        req.metadata_buffer_index = self.req_to_metadata_buffer_idx_allocator.alloc()
        assert req.metadata_buffer_index is not None
        return True

    def finalize_bootstrap(self, req: Req) -> bool:
        """Initialize the sender after bootstrap completes.
        Returns False if no metadata buffer is available (non-terminal)."""
        # 中译：断言确保本方法非幂等——只应在 pending_bootstrap 为真时调用一次。
        assert req.pending_bootstrap, "finalize_bootstrap is not idempotent"
        if not self.ensure_metadata_buffer(req):
            return False

        # 中译：记录 bootstrap 完成时间。
        req.time_stats.set_bootstrap_done_time()
        # 中译：decode 节点可能已缓存了本请求的一段前缀（decode_prefix_len），这部分 KV 无需
        decode_prefix_len = req.disagg_kv_sender.pop_decode_prefix_len()
        num_kv_indices = len(req.origin_input_ids)
        req.start_send_idx = decode_prefix_len
        # Base of the staging chunk grid (suffix-relative send coordinates).
        req.disagg_decode_prefix_len = decode_prefix_len
        num_kv_indices_to_send = num_kv_indices - decode_prefix_len
        # 中译：按 page size 把待发送 token 数换算成页数，并以此初始化 sender。
        num_pages = kv_to_page_num(
            num_kv_indices_to_send,
            self.scheduler.token_to_kv_pool_allocator.page_size,
        )
        req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
        req.pending_bootstrap = False
        return True

    def add(self, req: Req, num_kv_heads: int) -> None:
        # 中译：先创建 sender（失败/超容则不入队），成功后把请求放入 bootstrap 队列。
        if not self.create_sender(req, num_kv_heads):
            return
        self.queue.append(req)

    def extend(self, reqs: List[Req], num_kv_heads: int) -> None:
        # 中译：批量版 add。
        for req in reqs:
            self.add(req, num_kv_heads)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        # 中译：检查请求输入长度是否超过 KV pool 总容量；超限则直接中止并流式返回错误，返回 True。
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            req.time_stats.trace_ctx.abort(abort_info={"reason": message})
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.output_streamer.stream_output([req], req.return_logprob)
            return True
        return False

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate

        中译：把 max_new_tokens 设为 1，使 PrefillAdder 的显存估算准确（prefill 节点只产首 token）。
        """
        req.sampling_params.max_new_tokens = 1

    def pop_bootstrapped(
        self,
        return_failed_reqs: bool = False,
        pp_good_rids: Optional[List[str]] = None,
        pp_bad_rids: Optional[List[str]] = None,
    ) -> List[Req] | tuple[List[Req], List[Req]]:
        """
        pop the reqs which has finished bootstrapping

        return_failed_reqs: For PP, on rank 0, also return the failed reqs to notify the next rank
        pp_good_rids: RIDs that PP consensus determined as WaitingForInput.
        pp_bad_rids: RIDs that PP consensus determined as Failed.
        """

        # 中译：bootstrapped_reqs 收集本轮已就绪、可转入 waiting_queue 的请求；
        #       failed_reqs 收集 bootstrap 失败的请求；indices_to_remove 记录需从队列剔除的下标。
        bootstrapped_reqs = []
        failed_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0:
            if return_failed_reqs is False:
                return []
            else:
                return [], []

        # 中译：在 attn-CP / attn-TP 组内一致地轮询队列中所有 sender 的 bootstrap 状态，
        if self.pp_size > 1:
            polls = poll_and_all_reduce_pp(
                (req.rid for req in self.queue),
                KVPoll.WaitingForInput,
                pp_good_rids,
                pp_bad_rids,
            )
            uncovered = [i for i, poll in enumerate(polls) if poll is None]
            if uncovered:
                local_polls = poll_and_all_reduce_attn_cp_tp_group(
                    [self.queue[i].disagg_kv_sender for i in uncovered],
                    self.scheduler.attn_cp_cpu_group,
                    self.scheduler.attn_tp_cpu_group,
                )
                for i, local_poll in zip(uncovered, local_polls):
                    if local_poll == KVPoll.Failed:
                        polls[i] = KVPoll.Failed
        else:
            polls = poll_and_all_reduce_attn_cp_tp_group(
                [req.disagg_kv_sender for req in self.queue],
                self.scheduler.attn_cp_cpu_group,
                self.scheduler.attn_tp_cpu_group,
            )

        # 中译：逐个请求依据其 poll 状态分流处理。
        for i, (req, poll) in enumerate(zip(self.queue, polls)):
                # 中译：PP 模式下，bootstrap 成功仍需跨 rank 达成共识；而本地失败是终态，
            if poll is None:
                continue

            if poll == KVPoll.Failed:
                # 中译：bootstrap 失败——交给调度器统一处理（中止、释放资源、上报），并从队列剔除。
                self.scheduler.handle_bootstrap_failure(req)
                indices_to_remove.add(i)
                failed_reqs.append(req)
            elif poll == KVPoll.Bootstrapping:
                # 中译：仍在 bootstrapping。若开启了「乐观 prefill」且当前请求还有重试额度、
                #       未被引擎暂停回撤，则乐观地先让它进入 waiting_queue 抢跑前向
                #       （前提是能拿到 metadata buffer），bootstrap 结果稍后再校验。
                if (
                    req.prefill_attempt_count < get_disagg().optimistic_prefill_attempts
                    and not req.is_retracted  # engine paused
                ):
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.prefill_attempt_count += 1
                    bootstrapped_reqs.append(req)
                    indices_to_remove.add(i)
                    req.time_stats.set_wait_queue_entry_time()
            elif poll == KVPoll.WaitingForInput:
                # 中译：bootstrap 已完成、等待输入——正式 finalize（分配 buffer、初始化 sender）。
                if should_force_retry(req):  # skip checking for testing
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.prefill_attempt_count += 1
                elif not self.finalize_bootstrap(req):
                    continue
                bootstrapped_reqs.append(req)
                indices_to_remove.add(i)
                req.time_stats.set_wait_queue_entry_time()
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} in pop_bootstrapped"
                )

        # 中译：从队列中剔除所有已处理（就绪/失败）的请求，仅保留仍在 bootstrapping 的请求。
        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if return_failed_reqs is False:
            return bootstrapped_reqs
        else:
            return bootstrapped_reqs, failed_reqs

    def release_memory_occupation(self):
        # 中译：释放显存占用（如引擎暂停时）：清空队列，并从传输引擎注销已注册的 buffer。
        self.queue.clear()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        # 中译：恢复显存占用：把 buffer 重新注册回传输引擎。
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill

    中译：混入 Scheduler 的 Mixin，集中承载 PD 分离下 prefill 节点特有的调度逻辑
          （事件循环、批结果处理、在途传输队列轮询、KV 分块发送等）。
    """

    def maybe_prefetch_staging_for_batch(self: Scheduler, batch: ScheduleBatch) -> None:
        """Pre-send STAGING_REQ so decode allocates staging during GPU forward.

        中译：在 GPU 前向计算期间提前向 decode 节点发送 STAGING_REQ，使 decode 端能
              并行地预分配 staging 空间，从而与前向计算重叠、降低端到端时延。
        """
        kv_mgr = self.disagg_prefill_bootstrap_queue.kv_manager
        prefetch = getattr(kv_mgr, "_prefetch_staging_reqs", None)
        if prefetch is None:
            return
        for req in batch.reqs:
            room = getattr(req, "bootstrap_room", None)
            if room is not None and room in kv_mgr.transfer_infos:
                prefetch(room)

    def resolve_waiting_queue_bootstrap(self: Scheduler) -> None:
        """Resolve bootstrap status for waiting prefill requests before admission.

        Covers the window between leaving the bootstrap queue and being admitted
        into a running batch: aborts requests whose decode peer died, and
        finalizes optimistic requests whose bootstrap completed so they skip
        the post-forward bootstrap check.

        中译：在请求被纳入 prefill 批之前，补查 waiting_queue 中请求的握手状态。
              覆盖「已离开 bootstrap 队列、尚未进入运行批」这段窗口：
              - decode 对端已失效（握手失败）的请求直接中止，避免白算 prefill；
              - 乐观抢跑的请求若握手已完成，立即 finalize，从而跳过前向后的握手复查。
        """
        # 中译：只检查尚未中止的请求；已中止的请求会在后续流程中被清理，无需再轮询。
        candidates = [req for req in self.waiting_queue if not is_aborted(req)]
        # 中译：没有待检查的请求，直接返回，省去一次跨 rank 通信。
        if not candidates:
            return
        # 中译：轮询各请求 KV sender 的握手状态，并在 attention CP/TP 组内汇总，
        #       保证同组各 rank 看到一致的结果，避免不同 rank 对同一请求做出不同决定。
        polls = poll_and_all_reduce_attn_cp_tp_group(
            # 中译：每个请求对应一个 KV sender。
            [req.disagg_kv_sender for req in candidates],
            # 中译：attention 上下文并行（CP）组的 CPU 通信组。
            self.attn_cp_cpu_group,
            # 中译：attention 张量并行（TP）组的 CPU 通信组。
            self.attn_tp_cpu_group,
        )
        # 中译：记录本轮握手失败的请求，稍后统一从 waiting_queue 中移除。
        failed = set()
        # 中译：按顺序把每个请求与其汇总后的轮询状态配对处理。
        for req, poll in zip(candidates, polls):
            # 中译：握手失败（如 decode 对端已退出），不能再发送 KV。
            if poll == KVPoll.Failed:
                # 中译：中止请求、释放资源并向客户端返回错误。
                self.handle_bootstrap_failure(req)
                # 中译：标记待移除，避免遍历 candidates 时直接修改 waiting_queue。
                failed.add(req)
            # 中译：握手已完成、正在等待 KV 输入，且该请求仍处于乐观（pending）状态。
            elif (
                poll == KVPoll.WaitingForInput
                and req.pending_bootstrap
                # 中译：测试钩子强制走重试路径时跳过 finalize，留给前向后的检查处理。
                and not should_force_retry(req)
            ):
                # Optimistic requests reserved a metadata buffer when popped, so
                # finalize cannot fail here; if it ever does, the request stays
                # pending and the post-forward check resolves it.
                # 中译：乐观请求出队时已预留元数据槽位，finalize 通常不会失败；
                #       万一失败，请求保持 pending，由前向后的握手检查兜底处理。
                # 中译：初始化 sender（计算待发送页数、绑定元数据槽位），并清除 pending 标记。
                self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
        # 中译：存在失败请求时，重建 waiting_queue，把它们移出，防止被纳入后续 prefill 批。
        if failed:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in failed
            ]

    def has_bootstrapped_waiting_req(self: Scheduler) -> bool:
        return any(
            not req.pending_bootstrap and not is_aborted(req)
            for req in self.waiting_queue
        )

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def get_next_disagg_prefill_batch_to_run(
        self: Scheduler,
        # 中译：当前运行态批次；P 端不做 decode，这里主要承载分块请求等状态。
        running_batch: ScheduleBatch,
        # 中译：上一轮执行的批次，用于衔接分块 prefill 的收尾。
        last_batch: Optional[ScheduleBatch],
    ) -> NextBatchPlan:
        # 中译：组织 prefill 节点下一个要跑的批次；与聚合模式不同，这里不会转入 decode 选批。
        # 中译：先处理分块请求上被延迟到安全时机执行的中止，避免带着失效状态继续组批。
        self.process_pending_chunked_abort()

        # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
        # Otherwise, it hangs under high concurrency
        # 中译：P 端从不进入 update_running_batch（该标志通常在那里重置），需手动清零，否则高并发下会卡死。
        running_batch.batch_is_full = False

        # 中译：补查 waiting_queue 中乐观请求的握手状态：已完成的补做 finalize，失败的中止。
        self.resolve_waiting_queue_bootstrap()

        # 中译：处理上一轮遗留的分块（chunked）prefill：推进进度、按需发送已完成部分的 KV、校验握手。
        self.process_prefill_chunk(last_batch=last_batch, running_batch=running_batch)

        # 中译：复用通用 prefill 组批器，从 waiting_queue 按显存/token 预算攒一个新的 prefill 批。
        prefill_plan = self.get_new_batch_prefill(running_batch)
        # 中译：取出本轮要执行的 prefill 批（可能为 None）。
        batch = prefill_plan.batch_to_run
        # 中译：回写组批过程中可能更新的运行态批次。
        running_batch = prefill_plan.running_batch
        # 中译：DP attention 下各 rank 需同步参与 MLP collective，必要时补 idle 批以免部分 rank 缺席。
        batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)

        # 中译：有批可跑时记录调度时间点，用于请求生命周期统计。
        if batch:
            set_schedule_time_batch(batch)

        # 中译：返回本轮执行批与更新后的运行态批次。
        return NextBatchPlan(batch_to_run=batch, running_batch=running_batch)

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode."""
        while True:
            # 中译：接收请求。接收新到达的请求并做输入处理（其中会为新请求创建 sender 并放入 bootstrap 队列）。
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            # 中译：把 bootstrap 已完成的请求追加进 waiting_queue，等待组批前向。
            if self._engine_paused:
                continue
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )

            # 中译：组取下一个要跑的批次。
            # Get the next batch to run
            plan = self.get_next_disagg_prefill_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            # 中译：启动当前批次。
            if batch:
                # 中译：若开启 staging，提前触发 decode 端 staging 预分配以重叠时延。
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                # 中译：运行前向并处理结果（产出首 token、缓存 KV、发起 KV 传输、入在途队列）。
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # 中译：无批可跑时执行空闲自检/状态重初始化。
                self.on_idle()

            # 中译：轮询在途传输队列，完成传输的请求会被回收并向客户端返回。
            self.process_disagg_prefill_inflight_queue()

            # 中译：更新 last_batch——记录本轮批次，供下轮调度参考。
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        # 中译：PD 分离下 prefill 节点的「重叠」调度主循环——通过 result_queue 把「前向发起」与
        #       「上一批结果处理」错开一拍，使当前批的 GPU 前向与上一批的 CPU 侧结果处理重叠，
        #       从而提升吞吐。
        # 中译：初始化结果队列，元素为（批次快照, 前向结果）；稳态下轮末只积压 1 批。
        self.result_queue = deque()

        # 中译：调度主循环，常驻运行直到进程退出。
        while True:
            # 中译：从 TokenizerManager / RPC 非阻塞拉取本轮新到的请求。
            recv_reqs = self.request_receiver.recv_requests()
            # 中译：分发处理请求；生成请求会创建 KV sender 并进入 bootstrap 队列，控制请求直接回包。
            self.process_input_requests(recv_reqs)
            # 中译：引擎暂停（如权重更新）时只收请求，不推进组批与前向。
            if self._engine_paused:
                # 中译：跳过本轮剩余步骤，进入下一轮。
                continue
            # 中译：把握手已完成（或允许乐观抢跑）的请求从 bootstrap 队列移入 waiting_queue。
            self.waiting_queue.extend(
                # 中译：轮询各请求 sender 的握手状态，返回可参与 prefill 的请求。
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )

            # Get the next batch to run
            # 中译：组取下一个要跑的 prefill 批：补查乐观握手、处理遗留 chunk、攒新 prefill 批。
            plan = self.get_next_disagg_prefill_batch_to_run(
                # 中译：传入当前运行态与上一轮批次，用于分块 prefill 的收尾衔接。
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            # 中译：回写可能被更新的运行态批次。
            self.running_batch = plan.running_batch
            # 中译：取出本轮真正要执行的批（可能为 None）。
            batch = plan.batch_to_run
            # 中译：若启用 ngram embedding，为本批准备额外的前向输入；未启用时原样返回。
            batch = self.ngram_embedding_manager.prepare_for_forward(
                # 中译：传入当前分块请求，便于处理分块场景下的输入。
                batch, chunked_req=self.chunked_req
            )
            # 中译：记录当前批次，供调试 / 看门狗在异常时打印现场。
            self.cur_batch_for_debug = batch

            # 中译：有批可跑时，发起当前批次的前向。
            if batch:
                # 中译：若启用 staging，提前通知 D 端为本批预分配暂存空间。
                if self.enable_staging:
                    # 中译：让 D 端预分配与 P 端前向计算并行，降低端到端时延。
                    self.maybe_prefetch_staging_for_batch(batch)
                # 中译：在 forward_stream 上提交本批前向并返回结果对象；此时 GPU 可能尚未算完。
                batch_result = self.run_batch(batch)
                # 中译：WAR 屏障：让后续 schedule_stream 写入等待本批对共享缓冲的读取完成，
                #       防止下一轮调度改写仍在被前向读取的数据；这是 stream 依赖，不阻塞 CPU。
                self._apply_war_barrier()
                # 中译：保存（批次快照, 结果）入队，本轮不处理，留到下一轮与下一批前向重叠。
                self.result_queue.append((batch.copy(), batch_result))
            # 中译：本轮无批可跑。
            else:
                # 中译：置空结果，供后面的延迟采样判断跳过。
                batch_result = None

            # 中译：若上一轮提交过批次，则此处处理它的结果（与本批 GPU 前向重叠）。
            if self.last_batch:
                # 中译：按 FIFO 取出最早入队的上一批快照及结果。
                tmp_batch, tmp_result = self.result_queue.popleft()
                # 中译：处理上一批结果：追加首 token、缓存 KV、入在途队列并按需发送 KV。
                self.process_batch_result(tmp_batch, tmp_result)
            # 中译：上一批与本批都不存在，说明服务空闲。
            elif batch is None:
                # 中译：服务空闲时做自检并重新初始化部分状态。
                self.on_idle()

            # 中译：轮询在途传输队列：传输成功则释放 KV 并回包，失败则中止；未完成的继续保留。
            self.process_disagg_prefill_inflight_queue()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            # 中译：对当前批执行延迟采样；因其依赖上一批结果（如 grammar 状态），放在上一批处理完之后。
            self.launch_batch_sample_if_needed(batch_result, batch)

            # 中译：本批成为下一轮待回收的「上一批」，维持一拍错位。
            self.last_batch = batch

    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        Adapted from process_batch_result_prefill

        中译：PD 分离模式下 prefill 节点专用的"前向结果处理"方法（改编自普通的
              process_batch_result_prefill）。核心职责：对本批前向（prefill）已完成的请求，
              把首个 next token 落到请求上、缓存其 KV、然后通过 KV 传输把这些请求的 KV 缓存
              发往 decode 节点，并将请求挪进 disagg_prefill_inflight_queue（在途传输队列）
              等待传输完成。对仍在分块（chunked）中、尚未完成 prefill 的请求，则只递减
              其剩余分块计数、按需发送中间 chunk 的 KV，并不产出 token。

              与普通 prefill 的关键差异：
              1) 不在本节点做 decode，产出首 token 后立刻把 KV 传给 decode 节点；
              2) 引入"乐观（optimistic）bootstrap"机制——请求可能在握手未完成时就乐观地
                 进入前向，这里在产生副作用前再次轮询 bootstrap 状态，失败则回退/重排队。
        """
        # 中译：从前向结果对象里解包本方法需要的字段：
        #   logits_output            —— 本批的 logits / logprob 输出；
        #   next_token_ids           —— 采样出的下一 token（GPU 张量，后面会转成 list）；
        #   extend_input_len_per_req —— 每个请求本次 extend(prefill) 实际处理的输入长度；
        #   extend_logprob_start_len_per_req —— 每个请求开始计算 input logprob 的起点；
        #   copy_done                —— D2H 异步拷贝完成事件（重叠模式下用于同步等待）。
        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
            copy_done,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
            result.copy_done,
        )

        # 中译：重叠模式下结果是异步拷回 CPU 的，这里先阻塞等待拷贝完成，确保后续读到的
        #       CPU 张量数据有效。
        if copy_done is not None:
            copy_done.synchronize()
        # 中译：finalize 并释放 MoE 路由专家输出 / indexer top-k 输出等附带产物，回收其占用。
        auxiliary_output_starts = (
            self.batch_result_processor.snapshot_auxiliary_output_starts(batch, result)
        )
        auxiliary_output = result.auxiliary_host_output
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None

        # 中译：logprob_pt 是遍历各请求时在扁平化 logprob 张量里的游标（逐请求向后推进）。
        logprob_pt = 0
        # 中译：把采样出的 next token 从 GPU 张量转成 Python list，便于逐请求处理。
        assert batch.spec_info is result.next_draft_input
        draft_input = result.next_draft_input
        draft_hidden_states_cpu = None
        draft_dsa_topk_indices_cpu = None
        if self.spec_algorithm.is_eagle() and draft_input is not None:
            draft_hidden_states_cpu = draft_input.hidden_states.to(
                "cpu", non_blocking=False
            )
            if batch.spec_info.dsa_topk_indices is not None:
                draft_dsa_topk_indices_cpu = batch.spec_info.dsa_topk_indices.to(
                    "cpu", non_blocking=False
                )
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        next_token_ids = result.next_token_ids.tolist()
        # 中译：把 logprob 相关张量搬到 CPU，供后续按请求切片、组装返回值。
        self.batch_result_processor.move_logprobs_to_cpu(
            batch=batch,
            logits_output=logits_output,
        )

        # 中译：辅助函数——当某请求被提前跳过（如 bootstrap 失败、被中止）而未走正常的 logprob
        #       累加路径时，仍需把 logprob_pt 游标按该请求应占的 input logprob 数量手动前移，
        #       否则后续请求会读错切片位置。仅在该请求要返回 logprob 时才推进。
        def advance_logprob_pt(i: int, req: Req) -> None:
            nonlocal logprob_pt
            if not req.return_logprob or extend_input_len_per_req is None:
                return
            extend_logprob_start_len = extend_logprob_start_len_per_req[i]
            extend_input_len = extend_input_len_per_req[i]
            if extend_logprob_start_len < extend_input_len:
                logprob_pt += extend_input_len - extend_logprob_start_len

        # 中译：轮询本批中的"乐观 prefill"请求的 bootstrap 状态。
        # 中译：筛出"仍在等待 bootstrap 且已是最后一个 chunk"的乐观请求（带原始下标 i）。
            # 中译：在 attn-CP / attn-TP 组内一致地轮询这些请求的 KV sender 状态，
            # 中译：建立"请求下标 -> poll 结果"的映射，供下面主循环按 i 查询。
        # 中译：逐请求处理本批结果。strict=True 确保 reqs 与 next_token_ids 长度严格一致。
        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            # 中译：inflight_middle_chunks <= 0 表示该请求的最后一个 chunk 也已完成，
            #       即整个 prefill 真正结束，进入"产出首 token + 传输 KV"的主路径。
            if req.inflight_middle_chunks <= 0:
                # 中译：记录该请求 prefill 完成的时间点（用于请求生命周期统计）。
                req.time_stats.set_prefill_finished_time()

                # 中译：对乐观请求，在产生任何副作用（追加 token、缓存、入队）之前先确认
                # Test hook: exercise the release/requeue retry path.
                if req.pending_bootstrap and should_force_retry(req):
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    continue

                # 中译：把采样出的首个 next token 追加到请求的输出序列。
                req.output_ids.append(next_token_id)
                # 中译：把这个（尚未结束的）请求的 KV 写入 radix tree cache，以便前缀复用。
                maybe_cache_unfinished_req(req, self.tree_cache)
                # 中译：把请求加入"在途传输队列"，后续 process_disagg_prefill_inflight_queue
                #       会轮询其 KV 传输是否完成。
                self.disagg_prefill_inflight_queue.append(req)
                # 中译：EAGLE 投机解码下，需要把 draft 所需的 top-k 概率/索引与 hidden states
                # 中译：若该请求需要返回 logprob，按其 input 区间组装 input/output logprob 返回值，
                if self.spec_algorithm.is_eagle() and draft_input is not None:
                    req.output_topk_p = draft_input.topk_p[i]
                    req.output_topk_index = draft_input.topk_index[i]
                    req.hidden_states_tensor = draft_hidden_states_cpu[i].clone()
                    if draft_dsa_topk_indices_cpu is not None:
                        req.output_dsa_topk_indices = draft_dsa_topk_indices_cpu[
                            i
                        ].clone()
                    else:
                        req.output_dsa_topk_indices = None
                else:
                    req.hidden_states_tensor = None
                    req.output_dsa_topk_indices = None
                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    self.batch_result_processor.logprob_result_processor.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                    # 中译：last_chunk=True，发送该请求最后一块（也即全部）KV 到 decode 节点。
                # 中译：记录请求进入"传输队列"的时间点（生命周期统计）。
                if req.return_sampling_mask:
                    self.batch_result_processor.add_sampling_mask_return_values(
                        i, req, logits_output
                    )
                if not req.pending_bootstrap:
                    self.send_kv_chunk(req, last_chunk=True)
                req.time_stats.set_prefill_transfer_queue_entry_time()

                # 中译：若启用了语法约束（grammar），让语法状态机吃掉首个 token；
                #       accept 失败说明该 token 违反语法，释放其 KV 并标记请求中止。
                if req.grammar is not None:
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        prepare_abort(
                            req,
                            error_message,
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    req.grammar.finished = req.finished()
            else:
                # 中译：进入此分支说明该请求还有后续 chunk 未跑完（prefill 尚未结束），
                #       本轮不产出 token，仅递减剩余中间 chunk 计数。
                req.inflight_middle_chunks -= 1

                # 中译：重叠场景下，某乐观请求在 process_prefill_chunk 阶段被叫停后，其资源释放
                # Still chunking iff its next chunk was launched: either it is
                # still self.chunked_req, or its final chunk (extend_range
                # reaching the end of the input) is in flight. A yielded req
                # is neither, so do its deferred release here.
                still_chunking = self.chunked_req is req or (
                    req.extend_range is not None
                    and req.extend_range.end >= len(req.origin_input_ids)
                )
                if req.pending_bootstrap and not still_chunking:
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                # 中译：乐观 bootstrap 可能在这个重叠 chunk 已经在跑时才失败；此时直接丢弃该
                #       （已中止的）chunk，不再发送 KV。
                if is_aborted(req):
                    self.clear_pending_chunk_send(req)
                    advance_logprob_pt(i, req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                # 中译：中间 chunk 也可能需要累计 input logprob（仅当起点落在本 chunk 输入范围内）。
                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.batch_result_processor.logprob_result_processor.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

                # 中译：重叠模式下，中间 chunk 跑完即可把这一块的 KV 先发出去（last_chunk=False，
                # In non-overlap-mode, KV is sent in process_prefill_chunk
                # Only send when req's sender is initialized
                if self.enable_overlap and not req.pending_bootstrap:
                    assert (
                        req.metadata_buffer_index >= 0
                    ), f"Req {req.rid} does not have metadata buffer allocated"
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
                # 中译：记录本（非最后）分块 prefill 完成的时间点。
                req.time_stats.set_last_chunked_prefill_finish_time()

        # 中译：上报本批 prefill 的统计指标（是否走 CUDA graph、DP 协同信息等）。
        if auxiliary_output is not None:
            self.batch_result_processor.consume_auxiliary_output(
                batch,
                auxiliary_output,
                auxiliary_output_starts,
            )

        can_run_cuda_graph = result.can_run_cuda_graph
        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def process_disagg_prefill_inflight_queue(
        self: Scheduler, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.

        中译：轮询在途传输队列（正在把 KV 发往 decode 节点）中的请求：传输完成则回收资源
              并向客户端返回，失败则中止。
              rids_to_check：仅 PP 模式 rank>0 使用，用于与上一 rank 对“哪些 rid 已传输完成”达成共识。
        """
        if len(self.disagg_prefill_inflight_queue) == 0:
            return []

        # 中译：done_reqs 收集本轮达到终态（成功/失败）、可从队列移除的请求。
        done_reqs = []

        # 中译：组内一致轮询所有在途请求的传输状态（all-reduce 保证各 rank 结果一致）。
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        # 中译：undone_reqs 收集仍在传输中的请求，循环结束后会回写回队列继续等待。
        undone_reqs: List[Req] = []
        # 中译：逐个检查在途队列中请求的 poll() 状态；若成功则向客户端返回并从队列移除。
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if rids_to_check is not None:
                # 中译：PP 共识校验：不在上一 rank 已完成名单里的请求，本 rank 先保留为未完成。
                if req.rid not in rids_to_check:
                    undone_reqs.append(req)
                    continue

                # 中译：PP 模式下，上一 rank 可能已到达终态（Success/Failed），而本 rank 的本地
                #       poll 因时钟偏差或传播延迟仍处于瞬态。把非终态视为未完成而非崩溃。
                if poll not in (
                    KVPoll.Success,
                    KVPoll.Failed,
                ):
                    logger.warning_once(
                        f"PP rank {self.ps.pp_rank}: unexpected poll state {poll} for rid {req.rid} "
                        f"from consensus; treating as undone",
                    )
                    undone_reqs.append(req)
                    continue

            if req.pending_bootstrap:
                # Parked: prefill finished before bootstrap completed.
                if self.handle_pending_bootstrap(req, poll):
                    self.send_kv_chunk(req, last_chunk=True)
                    undone_reqs.append(req)
                elif poll != KVPoll.Failed:
                    undone_reqs.append(req)
                continue

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                # 中译：仍在等待输入 / 传输中，本轮不处理，保留到下轮。
                # 中译：传输成功——释放（解锁）该请求在 radix tree 的 KV，标记完成（长度 0，
                # FIXME(中译)：应清理该请求在传输引擎中的数据。
                # 中译：传输失败——构造错误信息（若异常来自另一 rank 的传播，降级为 debug 日志
                # 中译：对于传播而来的异常，静默错误信息（降为 debug）以避免重复日志。
                # todo: set Transferring correctly in backend
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                if not isinstance(req.finished_reason, FINISH_ABORT):
                    req.finished_reason = FINISH_LENGTH(length=0)
                release_kv_cache(req, self.tree_cache)  # unlock the tree
                # FIXME: clean up req's data in transfer engine
                req.disagg_kv_sender.clear()
                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()
            elif poll == KVPoll.Failed:
                self.handle_inflight_transfer_failure(req)
                done_reqs.append(req)
            else:
                # 中译：未预期的 poll 状态，保守处理为未完成（而非崩溃）。
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} in inflight queue"
                )

        # 中译：为已完成请求记录整体完成时间。
        for req in done_reqs:
            req.time_stats.set_completion_time()

        # 中译：为成功完成的请求计算并上报 KV 传输指标（时延/速度）；跳过中止请求、
        #       假 bootstrap host，以及 CP dummy rank（无实际传输）。
        for req in done_reqs:
            if isinstance(req.finished_reason, FINISH_ABORT):
                continue
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
                continue
            kv_mgr = getattr(req.disagg_kv_sender, "kv_mgr", None)
            if kv_mgr and getattr(kv_mgr, "is_dummy_cp_rank", False):
                continue
            metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
                req.disagg_kv_sender.get_transfer_metric()
            )
            if metrics:
                # 中译：更新供 REST API 读取的“最新值”。
                if "latency_ms" in metrics:
                    self.metrics_reporter.kv_transfer_latency_ms = metrics["latency_ms"]
                if "speed_gb_s" in metrics:
                    self.metrics_reporter.kv_transfer_speed_gb_s = metrics["speed_gb_s"]

        # 中译：把已完成传输的请求流式返回给客户端（prefill 节点至此完成其职责）。
        self.output_streamer.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        # 中译：释放已完成请求占用的 metadata buffer 槽位。
        for req in done_reqs:
            req: Req

            maybe_release_metadata_buffer(
                req, self.req_to_metadata_buffer_idx_allocator
            )

        # 中译：用未完成集合替换在途队列（即剔除已完成项），并返回本轮已完成的请求。
        self.disagg_prefill_inflight_queue = undone_reqs

        return done_reqs

    def handle_inflight_transfer_failure(
        self: Scheduler, req: Req
    ) -> Optional[Exception]:
        """Conclude an inflight request whose KV transfer failed."""
        error_message = (
            f"Prefill transfer failed for request rank={self.ps.tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        exc: Optional[Exception] = None
        try:
            req.disagg_kv_sender.failure_exception()
        except Exception as e:
            exc = e
            error_message += f" with exception {e}"
        # Mute error message for propagated exceptions to avoid duplicate logging
        if getattr(exc, "is_from_another_rank", False):
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        release_kv_cache(req, self.tree_cache)  # unlock the tree
        if not isinstance(req.finished_reason, FINISH_ABORT):
            prepare_abort(
                req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
            )
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_transfer_failed_reqs()
        return exc

    def clear_pending_chunk_send(self: Scheduler, req: Req) -> None:
        """Drop `req` from the sent-but-unconcluded chunk set.

        # 中译：收集所有达到终态的请求 rid（传输已完成或已失败）。
        # 中译：统一处理 bootstrap 失败：构造错误信息（来自他 rank 传播的异常降级为 debug），
        Every path that retires a request without a `last_chunk=True` send must
        call this: a stale entry holds the unified-memory compaction gate closed
        for the process lifetime.
        """
        self.disagg_prefill_pending_chunk_rids.discard(req.rid)

    def handle_bootstrap_failure(self: Scheduler, req: Req) -> None:
        self.clear_pending_chunk_send(req)
        error_message = (
            f"Prefill bootstrap failed for request rank={self.ps.tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        is_propagated = False
        try:
            req.disagg_kv_sender.failure_exception()
        except Exception as e:
            error_message += f" with exception {e}"
            is_propagated = getattr(e, "is_from_another_rank", False)
        # 中译：对于传播而来的异常，静默错误信息（降为 debug）以避免重复日志。
        if is_propagated:
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        if req.kv.is_held or req.mamba_pool_idx is not None:
            release_kv_cache(req, self.tree_cache)
        maybe_release_metadata_buffer(req, self.req_to_metadata_buffer_idx_allocator)
        req.pending_bootstrap = False
        prepare_abort(req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        self.output_streamer.stream_output([req], req.return_logprob)
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_bootstrap_failed_reqs()
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)

    def handle_pending_bootstrap(self: Scheduler, req: Req, poll: KVPoll) -> bool:
        """Return True when bootstrap is finalized and KV transfer can proceed."""
        if poll == KVPoll.Failed:
            # 中译：bootstrap 失败，统一处理并返回 False。
            self.handle_bootstrap_failure(req)
            return False
        elif poll == KVPoll.Bootstrapping:
            # 中译：仍在握手——乐观前向跑快了；非延迟释放则立即释放 KV 并重新入队重试。
            # 中译：bootstrap 已完成。先看测试钩子是否强制重试；否则 finalize 后即可传输。
            return False
        elif poll == KVPoll.WaitingForInput:
            if should_force_retry(req):  # test hook
                return False
            # 中译：metadata buffer 已在 pop_bootstrapped（请求入 waiting_queue 之前）分配，
            #       故 finalize 应不会失败（以 assert 保障）。
            assert self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
            return True
        else:
            raise RuntimeError(
                f"Unexpected poll state {poll} for req {req.rid} in handle_pending_bootstrap"
            )

    def check_bootstrap(self: Scheduler, req: Req) -> bool:
        """Check bootstrap status for an optimistic prefilled request.
        Returns True if bootstrap is finished.

        中译：检查一个乐观 prefill 请求的 bootstrap 状态，完成返回 True。
              若请求本就不在 pending_bootstrap（非乐观路径）则直接视为已完成。
        """
        if not req.pending_bootstrap:
            return True
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        return self.handle_pending_bootstrap(req, polls[0])

    def process_prefill_chunk(
        self: Scheduler,
        # 中译：上一轮执行的批次；若是 extend 批，需从中剔除仍在分块中的请求。
        last_batch: Optional[ScheduleBatch],
        # 中译：当前运行态批次；这里只用来复位 batch_is_full 标志。
        running_batch: ScheduleBatch,
    ) -> None:
        # 中译：处理上一轮遗留的分块（chunked）prefill。分块 prefill 把超长输入拆成多段，
        #       跨多轮逐段计算；每轮开始时需推进该请求、按需发送已算完部分的 KV，并校验握手。
        # 中译：收集需要从上一批中剔除的分块请求——它们的 prefill 尚未结束，不能被当作已完成处理。
        chunked_req_to_exclude = set()
        # 中译：存在跨轮进行中的分块请求时才需处理。
        if (req := self.chunked_req) is not None:
            # 中译：该请求仍在分块中，稍后从上一批中剔除。
            chunked_req_to_exclude.add(req)
            # 中译：把已算完的部分登记进 radix tree（chunked=True），既能前缀复用，也保证 KV 在后续 chunk 中被引用。
            maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)

            # 中译：乐观请求需先确认握手状态；非乐观请求直接视为已完成。
            if not self.check_bootstrap(req):
                # 中译：握手失败导致请求已被中止。
                if is_aborted(req):
                    # bootstrap failed
                    # 中译：停止继续分块计算该请求。
                    self.chunked_req = None
                # 中译：握手仍未完成，但 waiting_queue 里已有握手完成的请求在等。
                elif self.has_bootstrapped_waiting_req():
                    # optimistic request yields to waiting requests
                    # 中译：乐观请求让位给已就绪请求，停止分块计算。
                    self.chunked_req = None
                    # 中译：非重叠模式下立即释放已占用的 KV 并重新排队；
                    #       重叠模式下上一批结果尚未处理，释放推迟到结果处理阶段。
                    if not self.enable_overlap:
                        self.optimistic_release_and_requeue(req)
                # else: still bootstrapping, keep computing without sending
                # 中译：否则仍在握手中且没有其他就绪请求，继续计算下一块，但暂不发送 KV。
            # 中译：握手已完成，且处于重叠模式。
            elif self.enable_overlap:
                # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                # 中译：重叠模式下上一块的前向结果尚未确认，KV 发送推迟到 process_batch_result_disagg_prefill；
                #       这里只记录本块的结束位置，作为发送区间的右边界。
                req.tmp_end_idx = min(
                    # 中译：本块 extend 覆盖到的位置。
                    req.extend_range.end,
                    # 中译：不超过原始输入长度，避免越界。
                    len(req.origin_input_ids),
                )
            # 中译：握手已完成，且处于非重叠模式：上一块结果已处理完，可直接发送。
            else:
                # 中译：把 [start_send_idx, 当前已算位置) 的 KV 页发往 decode 节点。
                self.send_kv_chunk(req)

            # 中译：分块请求仍需继续计算时，复位满批标志，确保本轮还能组批跑它的下一块。
            if self.chunked_req is not None:
                running_batch.batch_is_full = False

        # 中译：上一批是 extend（prefill）批时，需把仍在分块中的请求从中剔除。
        if last_batch and last_batch.forward_mode.is_extend():
            # 中译：上一批记录着自己的分块请求。
            if last_batch.chunked_req:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                # 中译：context PP 下，最后一块跑完后当前 micro-batch 仍可能挂着过时的 chunked_req，一并剔除。
                chunked_req_to_exclude.add(last_batch.chunked_req)

            # 中译：记录过滤前的批大小。
            last_bs = last_batch.batch_size()
            # 中译：剔除已完成/中止的请求以及上述分块请求。
            last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
            # 中译：批大小变小说明腾出了容量，复位满批标志，允许继续接纳新请求。
            if last_batch.batch_size() < last_bs:
                running_batch.batch_is_full = False

    def maybe_send_cached_prefix_chunk(self: Scheduler, req: Req) -> None:
        if not envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get():
            return

        # Staging sends into positional grid slots, so the early-send boundary
        # must stay stable across the request's batches: snapshot the at-rest
        # prefix on the first batch. Non-staging reads the live prefix.
        if self.enable_staging and req.early_send_prefix_end is None:
            req.early_send_prefix_end = max(
                0, len(req.prefix_indices) - req.host_hit_length
            )

        if req.pending_bootstrap:
            return

        # Device-resident prefix only; page-aligned so start_send_idx stays exact.
        cached_end = (
            req.early_send_prefix_end
            if self.enable_staging
            else len(req.prefix_indices) - req.host_hit_length
        )
        if cached_end <= req.start_send_idx:
            return
        if cached_end % self.token_to_kv_pool_allocator.page_size != 0:
            # DCP radix hits can end on a logical cache-page boundary that is
            # not a complete physical DCP page. The regular final send covers
            # the full range; only skip this optional early-send optimization.
            return
        # Early-send issues the KV read before this step's forward is enqueued,
        # but under overlap scheduling the PRIOR step's prefill forward may still
        # be writing these prefix pages on forward_stream. Record a completion
        # event now so the transfer worker can wait on those writes before the
        # RDMA read, instead of racing them.
        if self.enable_overlap:
            ev = torch.cuda.Event()
            ev.record(self.forward_stream)
            req.disagg_kv_sender._early_send_wait_event = ev
        self.send_kv_chunk(req, last_chunk=False, end_idx=cached_end)

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
    ) -> None:
        """
        Send a prefilled chunk to the decode server

        中译：把一块已 prefill 的 KV（[start_send_idx, end_idx) 区间对应的页）发送给 decode 节点。
              last_chunk=True 表示这是最后一块（同时会写入元数据 buffer、处理各种状态页）。
        """
        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        # 中译：未显式指定 end_idx 时，默认取已填充长度与原始输入长度的较小值。
        transfer_input_len = len(req.origin_input_ids)
        end_idx = (
            end_idx
            if end_idx is not None
            else min(req.extend_range.end, transfer_input_len)
        )

        if not last_chunk:
            # 中译：非最后一块时，若末尾不满一页，先不发这个部分页，推迟到下一次发送。
            end_idx = end_idx - end_idx % page_size
            if self.enable_staging:
                # Staging identifies chunks positionally against a uniform
                # prefetched grid, so non-last sends must end on a grid
                # boundary; the remainder rides with the next send.
                grid_tokens = staging_grid_tokens(
                    get_schedule().chunked_prefill_size, page_size
                )
                base = req.disagg_decode_prefix_len
                end_idx = base + ((end_idx - base) // grid_tokens) * grid_tokens

        if end_idx < start_idx:
            logger.debug(
                "send_kv_chunk skip: rid=%s start_send_idx=%s end_idx=%s",
                req.rid,
                start_idx,
                end_idx,
            )
            return

        # 中译：取出本区间对应的 KV cache 索引（从 req_to_token 映射表拿 token->KV 位置）。
        # 中译：state_indices 用于携带特殊状态（Mamba/SWA/DSA/SWA_RING）的页索引，仅最后一块需要。
        state_indices: Optional[List] = None
        if last_chunk:
            # 中译：最后一块：写入随 KV 传输的元数据（如首 token、hidden states 等）到 buffer。
            self.disagg_metadata_buffers.set_buf(req)

            # 中译：fill_ids 包含 prefill 期间采样出的 token，但 decode 侧是按 origin_input_ids
            # Most state payloads read token-pool rows and should match the KV
            # range actually materialized on prefill. C128 state is request
            # scoped, so its transfer index must use the logical input length
            # that decode used to register the destination row.
            seq_len = min(req.extend_range.end, transfer_input_len)
            c128_seq_len = transfer_input_len

            def _mamba_payload():
                return [
                    self.req_to_token_pool.translate_mamba_indices(
                        self.req_to_token_pool.req_index_to_mamba_index_mapping[
                            req.kv.req_pool_idx
                        ]
                    )
                    .cpu()
                    .numpy()
                ]

            def _swa_payload():
                window_size = self.sliding_window_size
                window_start = max(req.disagg_decode_prefix_len, seq_len - window_size)
                window_start = (window_start // page_size) * page_size
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, window_start:seq_len
                ]
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                return kv_to_page_indices(window_kv_indices_swa, page_size)

            def _full_kv_pages_payload():
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, :seq_len
                ]
                return kv_to_page_indices(kv_indices_full, page_size)

            def _swa_ring_payload():
                # 中译：Unified_kv SWA 环行（req_pool_idx*ring_stride + pos%ring_stride），
                # 取最后 `window` 个位置，按位置升序排列，以便 decode（用其自己的
                # req_pool_idx）在位置上对应得上。
                _pool = self.token_to_kv_pool_allocator.get_kvcache()
                ring_stride = _pool.unified_swa_ring_size
                window_size = _pool.unified_swa_window
                window_start = max(0, seq_len - window_size)
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                state_slot = int(req.kv.req_pool_idx)
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                return ring_rows.astype(np.int32)

            def _c128_state_payload():
                online = is_dsv4_c128_online_enabled()
                ring_size = (
                    1
                    if online
                    else self.token_to_kv_pool_allocator.get_kvcache().get_ring_size(
                        128
                    )
                )
                return get_dsv4_c128_state_indices(
                    int(req.kv.req_pool_idx),
                    c128_seq_len,
                    online=online,
                    ring_size=ring_size,
                )

            state_types = (
                self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.state_types
            )
            payloads = {
                StateType.MAMBA: _mamba_payload,
                StateType.SWA: _swa_payload,
                StateType.DSA: _full_kv_pages_payload,
                StateType.MINIMAX_INDEX_K: _full_kv_pages_payload,
                StateType.SWA_RING: _swa_ring_payload,
                StateType.C128_STATE: _c128_state_payload,
                StateType.BLOCK_SCALE: _full_kv_pages_payload,
                StateType.BLOCK_SCALE_SWA: _swa_payload,
            }
            if _is_npu and isinstance(
                self.token_to_kv_pool_allocator.get_kvcache(),
                DeepSeekV4TokenToKVPool,
            ):
                from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                    dsv4_state_payloads,
                )

        # 中译：把 token 索引换算为页索引；若 sender 判定无需发送（如本块为空）则直接返回。
        # 中译：发送本块 KV（及状态页）到 decode 节点，并把发送起点前移到 end_idx，供下一块续发。
                payloads.update(
                    dsv4_state_payloads(
                        self.req_to_token_pool,
                        req.kv.req_pool_idx,
                        seq_len,
                        page_size,
                        prefix_len=req.disagg_decode_prefix_len,
                    )
                )
            state_indices = [
                payloads[st]() if st in payloads else None for st in state_types
            ]

        if self.enable_staging:
            # One sender.send per grid slot; the sender's cumulative page
            # counter marks only the final sub-send of the final chunk as
            # is_last, routing aux/state correctly.
            segments = compute_grid_segments(
                start_idx,
                end_idx,
                req.disagg_decode_prefix_len,
                staging_grid_tokens(get_schedule().chunked_prefill_size, page_size),
            )
        else:
            segments = [(start_idx, end_idx)]

        for seg_start, seg_end in segments:
            is_final_segment = seg_end == end_idx
            kv_indices = self.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, seg_start:seg_end
            ]
            # Unified memory: req_to_token holds VIRTUAL ids; the transfer needs
            # physical ones. Per segment, since each is its own gather.
            kv_indices = (
                self.token_to_kv_pool_allocator.translate_kv_indices_for_transfer(
                    kv_indices
                )
            )
            page_indices = kv_to_page_indices(kv_indices, page_size)
            segment_is_last = last_chunk and is_final_segment
            if not req.disagg_kv_sender.should_send_kv_chunk(
                len(page_indices), segment_is_last
            ):
                continue
            req.disagg_kv_sender.send(
                page_indices,
                state_indices if segment_is_last else None,
                num_kv_tokens=seg_end - seg_start,
            )
        req.start_send_idx = end_idx
        # A last chunk needs no entry: every `last_chunk=True` call site has
        # already put the request on `disagg_prefill_inflight_queue`.
        if last_chunk:
            self.disagg_prefill_pending_chunk_rids.discard(req.rid)
        else:
            self.disagg_prefill_pending_chunk_rids.add(req.rid)

    def optimistic_release_and_requeue(self: Scheduler, req: Req) -> None:
        # 中译：先缓存已完成部分（保留前缀复用价值），再释放 KV。
        """Release KV cache and requeue an optimistic prefill request."""
        max_attempts = get_disagg().optimistic_prefill_attempts
        maybe_cache_unfinished_req(req, self.tree_cache)
        release_kv_cache(req, self.tree_cache)
        # 中译：按回撤重置请求状态，清空输出、发送游标、临时结束位置与 hidden states。
        req.reset_for_retract()
        req.output_ids = array("q")
        req.start_send_idx = 0
        self.clear_pending_chunk_send(req)  # re-sends from scratch
        req.tmp_end_idx = -1
        req.disagg_decode_prefix_len = 0
        req.early_send_prefix_end = None
        req.hidden_states_tensor = None
        req.output_dsa_topk_indices = None
        req.pending_bootstrap = True
        req.time_stats.reset_prefill_retry_time()
            # 中译：乐观重试次数已耗尽——回退到 bootstrap 队列，走一次完整的真实握手。
        if req.prefill_attempt_count >= max_attempts:
            logger.info(
                f"Req {req.rid} exhausted optimistic prefill attempts "
                "falling back to bootstrap queue"
            )
            # 中译：重置 bootstrap 完成时间，以便记录下一次真实 bootstrap 的完成时刻。
            req.time_stats.bootstrap_done_time = 0.0
            self.disagg_prefill_bootstrap_queue.queue.append(req)
        else:
            # 中译：仍有重试额度——重试计数 +1，按需上报重试指标，并插回 waiting_queue 头部尽快重跑。
            req.prefill_attempt_count += 1
            logger.info(
                f"Req {req.rid} optimistic prefill yielded "
                f"({req.prefill_attempt_count}/{max_attempts} attempts used)"
            )
            if self.metrics_reporter.enable_metrics:
                self.metrics_collector.increment_prefill_retries(1)
            req.time_stats.set_wait_queue_entry_time()
            self.waiting_queue.insert(0, req)
