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
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_aborted,
    is_mla_backend,
    poll_and_all_reduce_attn_cp_tp_group,
    prepare_abort,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
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
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


def should_force_retry(req: Req) -> bool:
    """Test hook to force a request into optimistic prefill retry.

    中译：仅用于测试的钩子，按概率强制将请求推入「乐观 prefill 重试」路径，
          用于验证重试逻辑的正确性。仅在该环境变量概率 > 0、请求尚未重试过、
          且未被回撤时生效；基于 rid 的哈希值作确定性抽样。
    """
    retry_prob = envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
    if retry_prob <= 0 or req.time_stats.prefill_retry_count > 0 or req.is_retracted:
        return False

    # 中译：取 rid 的 SHA256 前 8 字节作为无符号整数，与阈值比较实现「按概率命中」。
    digest = hashlib.sha256(str(req.rid).encode()).digest()
    return int.from_bytes(digest[:8], "big") < retry_prob * 2**64


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
            self.scheduler.tp_worker.model_runner.max_token_pool_size
        )
        # 中译：KV 传输后端类型（Mooncake/NIXL/Mori/Ascend 等），决定实际使用的 KVManager/Sender 实现。
        self.transfer_backend = transfer_backend
        # 中译：staging buffer 是面向非 MLA 模型（GQA/MHA）的中转发送优化，MLA 模型不应开启。
        if envs.SGLANG_DISAGG_STAGING_BUFFER.get() and self.is_mla_backend:
            raise RuntimeError(
                "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                "(e.g. GQA, MHA). MLA models should not set this flag."
            )
        # 中译：初始化底层 KV 传输管理器（负责注册显存 buffer、维护握手/传输通道）。
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
        kv_args.prefill_start_layer = self.token_to_kv_pool.start_layer
        kv_args.prefill_end_layer = getattr(self.token_to_kv_pool, "end_layer", None)
        kv_args.mla_compression_ratios = None
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )

        if self.draft_token_to_kv_pool is not None:
            # 中译：也需传输 draft 模型的 KV cache；其索引总是与目标（主）模型共享。
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
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
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.ps.gpu_id

        req_to_token_pool = getattr(self.scheduler, "req_to_token_pool", None)
        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool,
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
            if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                kv_manager.set_kv_buffer_tensors(
                    kv_pool.k_buffer,
                    kv_pool.v_buffer,
                    kv_pool.page_size,
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
        assert req.pending_bootstrap, f"finalize_bootstrap is not idempotent"
        if not self.ensure_metadata_buffer(req):
            return False

        # 中译：记录 bootstrap 完成时间。
        req.time_stats.set_bootstrap_done_time()
        num_kv_indices = len(req.origin_input_ids)

        # 中译：decode 节点可能已缓存了本请求的一段前缀（decode_prefix_len），这部分 KV 无需
        #       重复传输；因此发送起点从该前缀之后开始，只发送剩余部分。
        decode_prefix_len = req.disagg_kv_sender.pop_decode_prefix_len()
        req.start_send_idx = decode_prefix_len
        num_kv_indices_to_send = num_kv_indices - decode_prefix_len
        # 中译：按 page size 把待发送 token 数换算成页数，并以此初始化 sender。
        num_pages = kv_to_page_num(
            num_kv_indices_to_send, self.token_to_kv_pool.page_size
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
        rids_to_check: Optional[List[str]] = None,
    ) -> List[Req]:
        """
        pop the reqs which has finished bootstrapping

        return_failed_reqs: For PP, on rank 0, also return the failed reqs to notify the next rank
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
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
        #       并通过 all-reduce 保证同组各 rank 得到一致结果，避免不同 rank 做出分歧决策。
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.queue],
            self.scheduler.attn_cp_cpu_group,
            self.scheduler.attn_tp_cpu_group,
        )

        # 中译：逐个请求依据其 poll 状态分流处理。
        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if (
                rids_to_check is not None
                and req.rid not in rids_to_check
                and poll != KVPoll.Failed
            ):
                # 中译：PP 模式下，bootstrap 成功仍需跨 rank 达成共识；而本地失败是终态，
                #       即使前面的 PP rank 已移除该请求，也必须在本 rank 把它排掉。
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
                    req.time_stats.prefill_retry_count
                    < self.scheduler.server_args.optimistic_prefill_retries
                    and not req.is_retracted  # 中译：engine 已暂停
                ):
                    if not self.ensure_metadata_buffer(req):
                        continue  # 中译：无可用 metadata buffer
                    bootstrapped_reqs.append(req)
                    indices_to_remove.add(i)
                    req.time_stats.set_wait_queue_entry_time()
            elif poll == KVPoll.WaitingForInput:
                # 中译：bootstrap 已完成、等待输入——正式 finalize（分配 buffer、初始化 sender）。
                #       finalize 失败（无 metadata buffer）则本轮跳过，下轮重试。
                if not self.finalize_bootstrap(req):
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

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def get_next_disagg_prefill_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        # 中译：组织 prefill 节点下一个要跑的批次。
        # 中译 HACK (byronhsu)：prefill 节点不会进入 update_running_batch（那里会重置该标志），
        #       因此这里手动重置 batch_is_full，否则高并发下会挂起。
        self.running_batch.batch_is_full = False

        # 中译：先处理上一轮遗留的分块（chunked）prefill（推进/发送/校验其状态）。
        self.process_prefill_chunk()

        # 中译：从 waiting_queue 组一个新的 prefill 批次，并按需为 DP attention 准备 MLP 同步批。
        batch = self.get_new_batch_prefill()
        batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)

        if batch:
            set_schedule_time_batch(batch)

        return batch

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode.

        中译：PD 分离下 prefill 节点的「非重叠」调度主循环——每轮串行地：收请求→就绪请求入队
              →组批→前向→处理结果→轮询在途 KV 传输。
        """
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()

        while True:
            # 中译：接收请求。接收新到达的请求并做输入处理（其中会为新请求创建 sender 并放入 bootstrap 队列）。
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            # 中译：把 bootstrap 已完成的请求追加进 waiting_queue，等待组批前向。
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            if self._engine_paused:
                continue

            # 中译：组取下一个要跑的批次。
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

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
        self.result_queue = deque()
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()

        while True:
            # 中译：收请求 + 输入处理；把 bootstrap 完成的请求入 waiting_queue。
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            if self._engine_paused:
                continue

            # 中译：共享 GPU buffer（req_to_token_pool / SWA 映射）上的写后读（WAR）屏障：
            #       调度流需等前向流完成，避免读写冲突。
            if self._war_barrier_enabled:
                self.schedule_stream.wait_stream(self.forward_stream)

            # 中译：组取下一个要跑的批次。
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # 中译：启动当前批次。
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                # 中译：发起当前批前向，并把（批副本, 结果）入队 result_queue，本轮先不处理它的结果，
                #       以便与下面处理「上一批」结果重叠。
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # 中译：处理「上一批」的结果（与当前批的 GPU 前向重叠）；若无上一批且当前也空则空闲自检。
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # 中译：服务空闲时做自检并重新初始化部分状态。
                self.on_idle()

            # 中译：轮询在途传输队列，回收已完成传输的请求。
            self.process_disagg_prefill_inflight_queue()

            # 中译：对当前批执行采样。因其依赖上一批的结果（如 grammar 状态），故放在处理完
            #       上一批之后再执行。
            self.launch_batch_sample_if_needed(batch_result)

            # 中译：更新 last_batch。
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
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None

        # 中译：logprob_pt 是遍历各请求时在扁平化 logprob 张量里的游标（逐请求向后推进）。
        logprob_pt = 0
        # 中译：把采样出的 next token 从 GPU 张量转成 Python list，便于逐请求处理。
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
        #       注意：在重叠调度下，那些在 process_prefill_chunk 时仍处于 pending 的分块请求
        #       不会在此处再次检查；即使它们在间隙中变为就绪，我们仍会重试该请求，
        #       以保持分块 prefill 的状态管理足够简单。
        optimistic_polls = {}
        # 中译：筛出"仍在等待 bootstrap 且已是最后一个 chunk"的乐观请求（带原始下标 i）。
        optimistic_reqs = [
            (i, req)
            for i, req in enumerate(batch.reqs)
            if req.pending_bootstrap and req.inflight_middle_chunks <= 0
        ]
        if optimistic_reqs:
            # 中译：在 attn-CP / attn-TP 组内一致地轮询这些请求的 KV sender 状态，
            #       并做 all-reduce 保证同组各 rank 拿到一致的 poll 结果（避免决策分歧）。
            polls = poll_and_all_reduce_attn_cp_tp_group(
                [req.disagg_kv_sender for _, req in optimistic_reqs],
                self.attn_cp_cpu_group,
                self.attn_tp_cpu_group,
            )
            # 中译：建立"请求下标 -> poll 结果"的映射，供下面主循环按 i 查询。
            optimistic_polls = {
                idx: poll for (idx, _), poll in zip(optimistic_reqs, polls)
            }

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
                #       bootstrap 是否真正成功；失败则推进 logprob 游标并跳过本请求。
                if i in optimistic_polls:
                    if not self.handle_pending_bootstrap(
                        req, optimistic_polls[i], defer_release=False
                    ):
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
                #       一并随请求传给 decode 节点（hidden_states 拷回 CPU 并 clone 以脱离 GPU
                #       生命周期）；否则不携带 hidden states。
                if self.spec_algorithm.is_eagle() and batch.spec_info is not None:
                    req.output_topk_p = batch.spec_info.topk_p[i]
                    req.output_topk_index = batch.spec_info.topk_index[i]
                    req.hidden_states_tensor = (
                        batch.spec_info.hidden_states[i].cpu().clone()
                    )
                else:
                    req.hidden_states_tensor = None
                # 中译：若该请求需要返回 logprob，按其 input 区间组装 input/output logprob 返回值，
                #       并相应前移 logprob_pt 游标。
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
                    self.send_kv_chunk(req, last_chunk=True)
                # 中译：记录请求进入"传输队列"的时间点（生命周期统计）。
                req.time_stats.set_prefill_transfer_queue_entry_time()

                # 中译：若启用了语法约束（grammar），让语法状态机吃掉首个 token；
                #       accept 失败说明该 token 违反语法，释放其 KV 并标记请求中止。
                if req.grammar is not None:
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        release_kv_cache(req, self.tree_cache)
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
                #       被推迟到这里：执行延迟释放并重新入队，推进 logprob 游标后跳过。
                if req.pending_bootstrap:
                    advance_logprob_pt(i, req)
                    self.optimistic_release_and_requeue(req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                # 中译：乐观 bootstrap 可能在这个重叠 chunk 已经在跑时才失败；此时直接丢弃该
                #       （已中止的）chunk，不再发送 KV。
                if is_aborted(req):
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
                #       end_idx 指明本块结束位置），让 KV 传输与后续计算重叠。
                #       前提是该请求已分配 metadata buffer 槽位。
                if self.enable_overlap:
                    assert (
                        req.metadata_buffer_index >= 0
                    ), f"Req {req.rid} does not have metadata buffer allocated"
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
                # 中译：记录本（非最后）分块 prefill 完成的时间点。
                req.time_stats.set_last_chunked_prefill_finish_time()

        # 中译：上报本批 prefill 的统计指标（是否走 CUDA graph、DP 协同信息等）。
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

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                # 中译：仍在等待输入 / 传输中，本轮不处理，保留到下轮。
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # 中译：传输完成
                # 中译：传输成功——释放（解锁）该请求在 radix tree 的 KV，标记完成（长度 0，
                #       因为首 token 已随 KV 传给 decode，后续生成在 decode 节点进行），
                #       清理 sender 在传输引擎中的残留数据，并入 done_reqs。
                release_kv_cache(req, self.tree_cache)  # 中译：解锁 tree cache
                req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME(中译)：应清理该请求在传输引擎中的数据。
                if hasattr(req.disagg_kv_sender, "clear"):
                    req.disagg_kv_sender.clear()
                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()
            elif poll == KVPoll.Failed:
                # 中译：传输失败——构造错误信息（若异常来自另一 rank 的传播，降级为 debug 日志
                #       避免重复告警），释放 KV、中止请求并入 done_reqs，按需上报失败指标。
                error_message = f"Prefill transfer failed for request rank={self.ps.tp_rank} {req.rid=} {req.bootstrap_room=}"
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
                release_kv_cache(req, self.tree_cache)  # 中译：解锁 tree cache
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                done_reqs.append(req)
                if self.metrics_reporter.enable_metrics:
                    self.metrics_collector.increment_transfer_failed_reqs()
            else:
                # 中译：未预期的 poll 状态，保守处理为未完成（而非崩溃）。
                logger.warning_once(
                    f"Unexpected polling state {poll} for rid {req.rid} in inflight queue; "
                    f"treating as undone",
                )
                undone_reqs.append(req)

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

    def get_transferred_rids(self: Scheduler) -> List[str]:
        """
        Used by PP, get the transferred rids but **do not pop**

        中译：供 PP 使用：获取已达终态（成功/失败）的 rid 列表，但**不从队列弹出**，
              仅用于跨 rank 达成共识。
        """
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        transferred_rids: List[str] = []

        # 中译：收集所有达到终态的请求 rid（传输已完成或已失败）。
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if poll == KVPoll.Success or poll == KVPoll.Failed:
                transferred_rids.append(req.rid)

        return transferred_rids

    def handle_bootstrap_failure(self: Scheduler, req: Req) -> None:
        # 中译：统一处理 bootstrap 失败：构造错误信息（来自他 rank 传播的异常降级为 debug），
        #       释放已分配的 KV 与 metadata buffer，中止请求并流式返回错误，按需上报指标并
        #       在 hicache 下释放被中止请求的存储引用。
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
        if req.req_pool_idx is not None or self.tree_cache.supports_mamba():
            release_kv_cache(req, self.tree_cache)
        maybe_release_metadata_buffer(req, self.req_to_metadata_buffer_idx_allocator)
        req.pending_bootstrap = False
        prepare_abort(req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        self.output_streamer.stream_output([req], req.return_logprob)
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_bootstrap_failed_reqs()
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)

    def handle_pending_bootstrap(
        self: Scheduler, req: Req, poll: KVPoll, defer_release: bool
    ) -> bool:
        """Return True when bootstrap is finalized and KV transfer can proceed.

        中译：处理一个「乐观 prefill」请求的 bootstrap 轮询结果。当 bootstrap 已完成化
              （finalize）、可以继续 KV 传输时返回 True；否则返回 False。
              defer_release：重叠模式下为 True，表示本处不立即释放/重排，而是推迟到
              process_batch_result_disagg_prefill 中处理。
        """
        if poll == KVPoll.Failed:
            # 中译：bootstrap 失败，统一处理并返回 False。
            self.handle_bootstrap_failure(req)
            return False
        elif poll == KVPoll.Bootstrapping:
            # 中译：仍在握手——乐观前向跑快了；非延迟释放则立即释放 KV 并重新入队重试。
            if not defer_release:
                self.optimistic_release_and_requeue(req)
            return False
        elif poll == KVPoll.WaitingForInput:
            # 中译：bootstrap 已完成。先看测试钩子是否强制重试；否则 finalize 后即可传输。
            force_retry = should_force_retry(req)  # 中译：测试钩子
            if force_retry:
                if not defer_release:
                    self.optimistic_release_and_requeue(req)
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
        return self.handle_pending_bootstrap(
            req, polls[0], defer_release=self.enable_overlap
        )

    def process_prefill_chunk(self: Scheduler) -> None:
        # 中译：处理分块（chunked）prefill 的遗留状态。分块 prefill 指把超长输入拆成多个
        #       chunk 逐次前向；本方法在每轮组批前推进当前 chunked 请求（发送已完成
        #       chunk 的 KV、校验其 bootstrap），并从上一批中滤除已处理的 chunked 请求。
        # 中译：chunked_req_to_exclude 收集需从上一批过滤掉的 chunked 请求（避免重复处理）。
        chunked_req_to_exclude = set()
        if self.chunked_req:
            chunked_req_to_exclude.add(self.chunked_req)
            # 中译：把当前 chunked 请求已完成的部分缓存入 radix tree（以便前缀复用）。
            maybe_cache_unfinished_req(self.chunked_req, self.tree_cache, chunked=True)

            if not self.check_bootstrap(self.chunked_req):
                # 中译：bootstrap 未就绪（乐观失败），停掉当前分块 prefill。
                self.chunked_req = None  # 中译：停掉当前分块 prefill
            elif self.enable_overlap:
                # 中译：重叠模式下把本 chunk 的 KV 发送推迟到 process_batch_result_disagg_prefill，
                #       仅先记下本 chunk 的结束位置 tmp_end_idx（确保结果已解析后再发）。
                self.chunked_req.tmp_end_idx = min(
                    self.chunked_req.fill_len,
                    len(self.chunked_req.origin_input_ids),
                )
            else:
                # 中译：非重叠模式直接发送当前已完成 chunk 的 KV。
                self.send_kv_chunk(self.chunked_req)

            if self.chunked_req is not None:
                self.running_batch.batch_is_full = False

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req:
                # 中译：在 context PP 下，最后一个 chunk 之后当前微批仍追踪着过时的 chunked_req，需丢弃它。
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            # 中译：从上一批中过滤掉上述 chunked 请求；若批大小因此变小，说明还有容量，重置 batch_is_full。
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

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
        end_idx = (
            end_idx
            if end_idx is not None
            else min(req.fill_len, len(req.origin_input_ids))
        )

        if not last_chunk:
            # 中译：非最后一块时，若末尾不满一页，先不发这个部分页，推迟到下一次发送。
            end_idx = end_idx - end_idx % page_size

        if end_idx < start_idx:
            logger.debug(
                "send_kv_chunk skip: rid=%s start_send_idx=%s end_idx=%s",
                req.rid,
                start_idx,
                end_idx,
            )
            return

        # 中译：取出本区间对应的 KV cache 索引（从 req_to_token 映射表拿 token->KV 位置）。
        kv_indices = (
            self.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
            .cpu()
            .numpy()
        )
        # 中译：state_indices 用于携带特殊状态（Mamba/SWA/DSA/SWA_RING）的页索引，仅最后一块需要。
        state_indices: Optional[List] = None
        if last_chunk:
            # 中译：最后一块：写入随 KV 传输的元数据（如首 token、hidden states 等）到 buffer。
            self.disagg_metadata_buffers.set_buf(req)

            # 中译：fill_ids 包含 prefill 期间采样出的 token，但 decode 侧是按 origin_input_ids
            # 注册状态页的（DecodePreallocQueue），且主池发送已在上面限制到 end_idx。
            # 这里匹配该长度，可避免当采样 token 跨页边界时多发一个状态页（那会导致
            # group_concurrent_contiguous 中 src/dst 长度不匹配）。
            seq_len = min(req.fill_len, len(req.origin_input_ids))

            def _mamba_payload():
                return [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]

            def _swa_payload():
                window_size = self.sliding_window_size
                window_start = max(0, seq_len - window_size)
                window_start = (window_start // page_size) * page_size
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, window_start:seq_len
                ]
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                return kv_to_page_indices(
                    window_kv_indices_swa.cpu().numpy(), page_size
                )

            def _dsa_payload():
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :seq_len
                ]
                return kv_to_page_indices(kv_indices_full.cpu().numpy(), page_size)

            def _swa_ring_payload():
                # 中译：Unified_kv SWA 环行（req_pool_idx*ring_stride + pos%ring_stride），
                # 取最后 `window` 个位置，按位置升序排列，以便 decode（用其自己的
                # req_pool_idx）在位置上对应得上。
                _pool = self.token_to_kv_pool_allocator.get_kvcache()
                ring_stride = _pool.unified_swa_ring_size
                window_size = _pool.unified_swa_window
                window_start = max(0, seq_len - window_size)
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                state_slot = int(req.req_pool_idx)
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                return ring_rows.astype(np.int32)

            state_types = (
                self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.state_types
            )
            state_indices = []
            for st in state_types:
                if st == StateType.MAMBA:
                    state_indices.append(_mamba_payload())
                elif st == StateType.SWA:
                    state_indices.append(_swa_payload())
                elif st == StateType.DSA:
                    state_indices.append(_dsa_payload())
                elif st == StateType.SWA_RING:
                    state_indices.append(_swa_ring_payload())
                else:
                    state_indices.append(None)

        # 中译：把 token 索引换算为页索引；若 sender 判定无需发送（如本块为空）则直接返回。
        page_indices = kv_to_page_indices(kv_indices, page_size)
        if not req.disagg_kv_sender.should_send_kv_chunk(len(page_indices), last_chunk):
            return
        # 中译：发送本块 KV（及状态页）到 decode 节点，并把发送起点前移到 end_idx，供下一块续发。
        req.disagg_kv_sender.send(page_indices, state_indices)
        req.start_send_idx = end_idx

    def optimistic_release_and_requeue(self: Scheduler, req: Req) -> None:
        """Release KV cache and requeue an optimistic prefill request.

        中译：当乐观 prefill 提前跑了前向但 bootstrap 尚未就绪时，回滚并重新排队该请求：
              释放其 KV、重置与本次前向相关的临时状态，重置 pending_bootstrap。
              若重试次数耗尽则回退到 bootstrap 队列（重走完整握手）；否则插回 waiting_queue 头部重试。
        """
        max_retries = self.server_args.optimistic_prefill_retries
        # 中译：先缓存已完成部分（保留前缀复用价值），再释放 KV。
        maybe_cache_unfinished_req(req, self.tree_cache)
        release_kv_cache(req, self.tree_cache)
        # 中译：按回撤重置请求状态，清空输出、发送游标、临时结束位置与 hidden states。
        req.reset_for_retract()
        req.output_ids = array("q")
        req.start_send_idx = 0
        req.tmp_end_idx = -1
        req.hidden_states_tensor = None
        req.pending_bootstrap = True
        req.time_stats.reset_prefill_retry_time()
        if req.time_stats.prefill_retry_count >= max_retries:
            # 中译：乐观重试次数已耗尽——回退到 bootstrap 队列，走一次完整的真实握手。
            logger.info(
                f"Req {req.rid} exhausted optimistic prefill retries "
                "falling back to bootstrap queue"
            )
            # 中译：重置 bootstrap 完成时间，以便记录下一次真实 bootstrap 的完成时刻。
            req.time_stats.bootstrap_done_time = 0.0
            self.disagg_prefill_bootstrap_queue.queue.append(req)
        else:
            # 中译：仍有重试额度——重试计数 +1，按需上报重试指标，并插回 waiting_queue 头部尽快重跑。
            req.time_stats.prefill_retry_count += 1
            logger.info(
                f"Req {req.rid} optimistic prefill retry "
                f"{req.time_stats.prefill_retry_count}/{max_retries}"
            )
            if self.metrics_reporter.enable_metrics:
                self.metrics_collector.increment_prefill_retries(1)
            req.time_stats.set_wait_queue_entry_time()
            self.waiting_queue.insert(0, req)
