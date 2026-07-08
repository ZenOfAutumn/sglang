"""
Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding

中译：decode（解码）服务器中一个请求的生命周期。

在 PD（Prefill-Decode，预填充-解码）分离架构下，decode 节点不做 prefill 前向计算，
而是从 prefill 节点接收已算好的 KV Cache，然后只做逐 token 的解码生成。请求依次流经
以下四个阶段（对应四个队列/批次）：

1. PreallocQueue（预分配队列）：
    a. 为每个请求初始化一个 KV 接收器（kv_receiver）；
    b. 请求先与 prefill 节点握手（handshake），一旦本地有空闲 KV 空间就预分配 KV；
    c. 预分配成功后把请求移入 TransferQueue（传输队列）。

2. TransferQueue（传输队列）：
    a. 轮询（poll）接收器以检查 KV 传输状态；
    b. 若传输完成，则把请求移入 WaitingQueue（等待队列）。

3. WaitingQueue（等待队列）：
    a. 用队列中的请求构造 PrebuiltExtendBatch（预构建的 extend 批次）；
    b. 跳过 prefill 前向计算，只填充所需的元数据（因为 KV 已由 prefill 节点算好并传来）。

4. RunningBatch（运行批次）：
    a. 把已就绪的 PrebuiltExtendBatch 合并进运行批次，开始执行解码。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributed import ProcessGroup

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    _is_fake_transfer,
    get_kv_class,
    is_mla_backend,
    poll_and_all_reduce,
    poll_and_all_reduce_with_staging,
    prepare_abort,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
)
from sglang.srt.mem_cache.common import (
    kv_to_page_indices,
    page_align_floor,
    release_kv_cache,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    KVCache,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.observability.req_time_stats import (
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.utils import get_num_new_pages
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

CLIP_MAX_NEW_TOKEN = envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get()


def _bootstrap_addr(req: Req) -> str:
    # FIXME: make a property of a req
    # 中译：把请求的 bootstrap 主机与端口拼成 "host:port" 字符串，作为标识 prefill 节点的地址。
    #       FIXME（原注）：这本应做成 Req 的一个属性（property）。
    return NetworkAddress(req.bootstrap_host, req.bootstrap_port).to_host_port_str()


class DecodeReqToTokenPool:
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.

    中译：DecodeReqToTokenPool 与普通 ReqToTokenPool 的区别在于，前者额外为「预分配请求」
          预留（subscribe）一块内存。

          在 ReqToTokenPool 中，若 `--max-running-requests`（最大并发运行请求数）为 8，则约束为：
              预分配数 + 传输中数 + 运行中数 <= 8；
          但实际上还有更多内存可以承载预分配请求，这块能力被浪费了。

          在 DecodeReqToTokenPool 中，同样 `--max-running-requests` 为 8 时，约束变为：
              运行中数 <= 8，且 预分配数 + 传输中数 <= pre_alloc_size；
          这样就能用空闲内存去提前预分配请求，从而不阻塞（unblock）上游 prefill 节点的推进。
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        # +1 padding row at index 0; see ReqToTokenPool for rationale.
        # 中译：额外 +1 是在索引 0 处放一行 padding（占位），原因参见 ReqToTokenPool。
        #       因此实际分配大小 = 运行槽位 size + 预分配槽位 pre_alloc_size + 1 行占位。
        self._alloc_size = size + pre_alloc_size + 1
        self.max_context_len = max_context_len
        self.device = device
        self.pre_alloc_size = pre_alloc_size
        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )

        self.free_slots = list(range(1, self._alloc_size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: List[Req]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        # 中译：reusing 收集那些「已经拥有 req_pool_idx、会复用现有槽位」的请求下标
        #       （例如分块 prefill 跨 chunk 继续时，同一请求需沿用同一个槽位）。
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        assert (
            len(reusing) <= 1
        ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
            for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))


class HybridMambaDecodeReqToTokenPool(HybridReqToTokenPool):
    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: Mamba2CacheParams,
        mamba_layer_ids: List[int],
        speculative_num_draft_tokens: int,
        enable_mamba_extra_buffer: bool,
        pre_alloc_size: int,
        enable_overlap_schedule: bool,
        mamba_size: int = None,
        start_layer: int = None,
    ):
        DecodeReqToTokenPool.__init__(
            self,
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        # Each request needs 1 main mamba slot + ping-pong slots when extra_buffer is enabled.
        # Cap the pool at max concurrent requests * slots_per_req to avoid allocating failed.
        # 中译：每个请求需要 1 个主 mamba 槽位；当启用 extra_buffer 时还需额外的 ping-pong 槽位。
        #       池大小上限设为「最大并发请求数 * 每请求槽位数」，以避免运行期分配失败。
        slots_per_req = 1 + (
            self.mamba_ping_pong_track_buffer_size if enable_mamba_extra_buffer else 0
        )
        max_slots_needed = (size + pre_alloc_size) * slots_per_req
        if mamba_size is not None:
            effective_mamba_size = max(mamba_size, max_slots_needed)
            if mamba_size < max_slots_needed:
                logger.warning(
                    "mamba_size (%d) is less than decode side's max_slots_needed (%d = %d reqs * %d slots/req), "
                    "raising effective_mamba_size to %d",
                    mamba_size,
                    max_slots_needed,
                    size + pre_alloc_size,
                    slots_per_req,
                    effective_mamba_size,
                )
        else:
            effective_mamba_size = max_slots_needed
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            mamba_size=effective_mamba_size,
            mamba_spec_state_size=size + pre_alloc_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=self.enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))
        self.mamba_allocator.clear()


@dataclass
class DecodeRequest:
    """decode 侧对一个请求的包装，贯穿「预分配 → 传输」两个队列的生命周期。

    中译：DecodeRequest 在原始 Req 之外，额外挂载了 PD 分离 decode 侧所需的运行时状态——
          KV 接收器、握手进度、元数据 buffer 落点，以及 HiCache（分层缓存）相关的命中/回载状态。
          它随请求在 DecodePreallocQueue 与 DecodeTransferQueue 之间流转；一旦 KV 传输完成、
          请求进入等待队列开始解码，kv_receiver 会被清理（置 None），本包装的使命即结束。
    """

    # 中译：被包装的原始请求对象（含 input_ids、采样参数、bootstrap_room、logprob 状态等）。
    req: Req
    # 中译：本请求专属的 KV 接收器，负责与 prefill 节点握手、告知 KV 落点并接收 KV 传输；
    #       其 poll() 返回的状态（Bootstrapping/WaitingForInput/Transferring/Success/Failed）
    #       驱动请求在各队列间的推进。传输完成后会被 clear() 并置为 None。
    kv_receiver: CommonKVReceiver
    # 中译：握手是否已完成、进入「等待输入（KV）」状态。由 _update_handshake_waiters 在轮询到
    #       KVPoll.WaitingForInput 时置 True；只有该标志为 True 的请求才会被预分配 KV。
    waiting_for_input: bool = False
    # 中译：本请求在 MetadataBuffers 中占用的槽位下标（-1 表示尚未分配）。prefill 侧会把首 token、
    #       logprob、hidden_states、bootstrap_room 等元数据写入该槽位，decode 侧据此提交传输结果。
    metadata_buffer_index: int = -1

    # HiCache Status
    # 中译：以下为 HiCache（分层缓存）相关状态字段，用于记录 decode 侧命中/回载的前缀 KV 情况。
    # 中译：前缀匹配结果——记录本请求在 decode radix cache 中命中的前缀（L1 设备命中长度、
    #       L1+L2+L3 的总前缀长度、命中节点、需回载 token 数等），用于跳过重复传输并计入复用。
    prefix_match: Optional[DecodePrefixMatch] = None
    # 中译：从 L2/L3 回载（loadback）到设备（L1）的 KV 索引张量——即 [prefix_len, total) 缺口
    #       被 HiCache 填补后所占用的显存 KV 位置。
    hicache_restored_kv_indices: Optional[torch.Tensor] = None
    # 中译：回载完成后在 radix 树中对应的缓存节点，用于后续引用计数与缓存挂接。
    hicache_restored_node: Any = None
    # 中译：HiCache 异步回载操作的消费者索引（consumer index），用于向 cache controller
    #       轮询/领取该请求回载操作的完成结果（-1 表示尚未发起或不适用）。
    hicache_load_consumer_index: int = -1
    # 中译：本请求的 HiCache 本地恢复状态机：PENDING（回载进行中）/ READY（回载完成、可提交）/
    #       FAILED（回载失败）。TransferQueue 会用它对 KVPoll.Success 做门控——只有回载也就绪
    #       才真正提交传输、放行进入解码。
    hicache_restore_status: HiCacheRestoreResult = HiCacheRestoreResult.PENDING

    @property
    def seqlen(self) -> int:
        # 中译：代理到底层 Req 的序列长度（origin_input_ids + output_ids 的当前总长）。
        return self.req.seqlen

    @property
    def priority(self) -> Optional[int]:
        # 中译：代理到底层 Req 的调度优先级（启用优先级调度时用于队列排序）。
        return self.req.priority


class DecodePreallocQueue(DecodeHiCachePreallocMixin):
    """
    Store the requests that are preallocating.

    中译：预分配队列。存放正处于「预分配 KV」阶段的请求——即已发起握手、正在等待或
          已获得本地 KV 空间的请求。这是 decode 请求生命周期的第一个队列。
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: BasePrefixCache,
        gloo_group: ProcessGroup,
        tp_rank: int,
        tp_size: int,
        dp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        pp_rank: int,
        num_reserved_decode_tokens: int,
        transfer_backend: TransferBackend,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache
        self.gloo_group = gloo_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens
        self.pp_rank = pp_rank
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.transfer_backend = transfer_backend
        # Queue for requests pending pre-allocation
        # 中译：queue——已创建接收器、等待预分配 KV 的请求主队列。
        self.queue: List[DecodeRequest] = []
        # 中译：retracted_queue——被回退（retract，因显存不足被换出）的请求，待后续重新分配。
        self.retracted_queue: List[Req] = []
        # 中译：pending_reqs——尚未解析出 prefill 侧 dp rank、需走慢路径查询的请求。
        self.pending_reqs: List[DecodeRequest] = []
        # 中译：以下四个字段用于控制「解析 prefill 信息（_ensure_prefill_info）」的重试节奏：
        #       _ensure_retry_count：每个 bootstrap 地址已重试的次数（以调度周期计）。
        self._ensure_retry_count: Dict[str, int] = {}
        self._max_ensure_retries: int = 15  # scheduling cycles  # 中译：最大重试周期数
        # 中译：_ensure_last_attempt_time：每个地址上次尝试解析的时间戳，用于控制重试间隔。
        self._ensure_last_attempt_time: Dict[str, float] = {}
        self._ensure_retry_interval: float = 1.0  # seconds  # 中译：重试间隔（秒）
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        if self.enable_staging and self.is_mla_backend:
            raise RuntimeError(
                "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                "(e.g. GQA, MHA). MLA models should not set this flag."
            )
        self.kv_manager = self._init_kv_manager()
        if self.enable_staging:
            self.transfer_queue._init_staging_handler(self.kv_manager)

        if (
            self.scheduler.tp_worker.is_hybrid_swa
            and not self._uses_swa_tail_prealloc()
        ):
            # Fallback for SWA allocators that still allocate the SWA pool at
            # full prompt length.
            # 中译：对于仍按「完整 prompt 长度」分配 SWA 池的分配器，这里做一个降级处理：
            #       把 max_total_num_tokens 限制为模型的 swa_max_total_num_tokens，避免预估过高。
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

    def _uses_swa_tail_prealloc(self) -> bool:
        return (
            isinstance(self.token_to_kv_pool, (SWAKVPool, DeepSeekV4TokenToKVPool))
            and self.token_to_kv_pool_allocator.page_size > 1
            and hasattr(self.token_to_kv_pool_allocator, "alloc_extend_swa_tail")
        )

    def _swa_tail_len(self, seq_len: int) -> int:
        if not self._uses_swa_tail_prealloc() or seq_len <= 0:
            return max(seq_len, 0)

        window_size = self.scheduler.sliding_window_size
        if window_size is None or window_size <= 0:
            return seq_len

        page_size = self.token_to_kv_pool_allocator.page_size
        window_start = max(0, seq_len - window_size)
        window_start = (window_start // page_size) * page_size
        return seq_len - window_start

    def _swa_retractable_len(self, req: Req) -> int:
        if not self._uses_swa_tail_prealloc():
            return len(req.origin_input_ids) + len(req.output_ids)
        return self._swa_tail_len(len(req.origin_input_ids)) + len(req.output_ids)

    def _prealloc_kv_lens(self, req: Req) -> Tuple[int, int]:
        allocated_kv_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        if self._uses_swa_tail_prealloc():
            return allocated_kv_len, self._swa_tail_len(allocated_kv_len)
        return allocated_kv_len, allocated_kv_len

    def _prealloc_required_tokens(self, req: Req) -> Tuple[int, int]:
        full_len, swa_len = self._prealloc_kv_lens(req)
        return (
            full_len + self.num_reserved_decode_tokens,
            swa_len + self.num_reserved_decode_tokens,
        )

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()

        attn_tp_size = get_attention_tp_size()
        kv_args.engine_rank = self.tp_rank % (attn_tp_size)

        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.ps.dp_rank
        transfer_kv_pool = (
            self.scheduler.hisparse_coordinator.mem_pool_host
            if self.scheduler.enable_hisparse
            else self.token_to_kv_pool
        )
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            transfer_kv_pool.get_contiguous_buf_infos()
        )
        if self.scheduler.enable_hisparse and isinstance(
            self.token_to_kv_pool, DeepSeekV4TokenToKVPool
        ):
            device_kv_data_ptrs, device_kv_data_lens, device_kv_item_lens = (
                self.token_to_kv_pool.get_contiguous_buf_infos()
            )
            c4_layer_num = self.scheduler.hisparse_coordinator.mem_pool_host.layer_num
            kv_data_ptrs += device_kv_data_ptrs[c4_layer_num:]
            kv_data_lens += device_kv_data_lens[c4_layer_num:]
            kv_item_lens += device_kv_item_lens[c4_layer_num:]
        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            # 中译：（推测解码时）需一并传输 draft 草稿模型的 KV Cache；其索引总是与 target
            #       目标模型共享（因此无需单独的一套索引）。
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )

        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool,
            total_kv_layers=self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=getattr(self, "req_to_token_pool", None),
        )

        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.ps.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.DECODE,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Staging buffer setup (only when heterogeneous TP staging is enabled)
        # 中译：配置 staging（暂存）缓冲区，仅在开启异构 TP staging 时生效（MLA 不支持）。
        #       用于 prefill 与 decode 两侧 TP 大小不一致时，先落到中转缓冲区再重排到目标布局。
        if self.enable_staging and not self.is_mla_backend:
            kv_pool_for_heads = self.token_to_kv_pool
            if hasattr(kv_pool_for_heads, "full_kv_pool"):
                kv_pool_for_heads = kv_pool_for_heads.full_kv_pool
            per_rank_kv_heads = getattr(kv_pool_for_heads, "head_num", 0)
            if per_rank_kv_heads > 0:
                kv_args.kv_head_num = per_rank_kv_heads
                kv_args.total_kv_head_num = per_rank_kv_heads * attn_tp_size
            if hasattr(kv_manager, "set_kv_buffer_tensors"):
                kv_pool = kv_pool_for_heads
                if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                    kv_manager.set_kv_buffer_tensors(
                        kv_pool.k_buffer, kv_pool.v_buffer, kv_pool.page_size
                    )
        return kv_manager

    def add(self, req: Req, is_retracted: bool = False) -> None:
        """Add a request to the pending queue.

        中译：把一个请求加入预分配（pending）队列。
              若是被回退的请求，直接放回 retracted_queue；否则创建 KV 接收器，
              并尝试走快路径（本地缓存）解析 prefill 侧 dp rank 后直接 init；
              若无法解析则放入 pending_reqs，后续走慢路径查询。
        """
        # 中译：若请求长度超过 KV 容量上限，直接在内部报错/中止并返回，不入队。
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if is_retracted:
            # 中译：被回退的请求清空 retraction_mb_id，直接放入回退队列等待重新调度。
            req.retraction_mb_id = None
            self.retracted_queue.append(req)
        else:
            # 中译：为新请求创建 KV 接收器并封装为 DecodeRequest。
            decode_req = self._create_receiver_and_enqueue(req)

            # NOTE: fake transfer does not need to resolve prefill dp rank in the pending queue
            # 中译：fake transfer（伪传输，仅测试用）无需在 pending 队列中解析 prefill dp rank，
            #       直接用 rank 0 初始化接收器即可。
            if _is_fake_transfer(req, self.scheduler.server_args):
                decode_req.kv_receiver.init(0)
                return

            # Fast path: cache-only lookup, no network calls
            # 中译：快路径——仅查本地缓存，不发网络请求；若能直接解析出 dp rank 则立即 init。
            prefill_dp_rank = self._resolve_prefill_dp_rank(req)
            logger.debug(f"prefill_dp_rank: {prefill_dp_rank}")
            if prefill_dp_rank is not None:
                decode_req.kv_receiver.init(prefill_dp_rank)
                return

            # 中译：快路径未命中（本地无缓存），放入 pending_reqs 等待慢路径解析。
            self.pending_reqs.append(decode_req)

    def _match_prefix_and_lock(self, req: Req) -> DecodePrefixMatch:
        """
        Match a request against the decode-side radix cache, lock the matched
        node to prevent eviction, and return the matched prefix information.

        中译：在 decode 侧的 radix cache（前缀树）中对请求做前缀匹配，锁住命中节点
              以防止被驱逐（eviction），并返回匹配到的前缀信息。
        """
        result = match_prefix_for_req(
            self.tree_cache,
            req,
            req.origin_input_ids,
            cow_mamba=self.tree_cache.supports_mamba(),
            include_req=True,
        )
        # Always lock to match aggregated scheduling behavior
        # 中译：总是加锁，以与聚合式（非分离）调度的行为保持一致。
        self.tree_cache.inc_lock_ref(result.last_device_node)
        return self._build_decode_prefix_match(req, result)

    def _resolve_prefill_dp_rank(self, req: Req) -> Optional[int]:
        prefill_info = self.kv_manager.prefill_info_table.get(_bootstrap_addr(req))
        # If None, it will go to the slow path and resolve prefill_info by _ensure_prefill_info then cache it
        # 中译：若本地缓存中无 prefill_info（返回 None），会走慢路径通过 _ensure_prefill_info
        #       去解析，并将结果缓存下来。
        if prefill_info is None:
            return None

        if req.disagg_prefill_dp_rank is not None:
            return req.disagg_prefill_dp_rank

        if prefill_info.dp_size == 1:
            return 0

        if (
            prefill_info.follow_bootstrap_room
            and not envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.get()
        ):
            return req.bootstrap_room % prefill_info.dp_size

        return None

    def _create_receiver_and_enqueue(self, req: Req) -> DecodeRequest:
        backend = (
            TransferBackend.FAKE
            if _is_fake_transfer(req, self.scheduler.server_args)
            else self.transfer_backend
        )
        kv_receiver_class = get_kv_class(backend, KVClassType.RECEIVER)

        kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=_bootstrap_addr(req),
            bootstrap_room=req.bootstrap_room,
        )

        decode_req = DecodeRequest(req=req, kv_receiver=kv_receiver)
        self.queue.append(decode_req)
        return decode_req

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.output_streamer.stream_output([req], req.return_logprob)
            return True
        if self._uses_swa_tail_prealloc():
            _, swa_required = self._prealloc_required_tokens(req)
            swa_capacity = self.token_to_kv_pool_allocator.size_swa
            if swa_required > swa_capacity:
                message = (
                    f"Request {req.rid} requires too many SWA KV tokens for "
                    f"decode preallocation: {swa_required} > {swa_capacity}"
                )
                logger.error(message)
                prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
                self.scheduler.output_streamer.stream_output([req], req.return_logprob)
                return True
        return False

    def extend(self, reqs: List[Req], is_retracted: bool = False) -> None:
        """Add a request to the pending queue.

        中译：批量版的 add——把一组请求逐个加入预分配（pending）队列。
        """
        for req in reqs:
            self.add(req, is_retracted=is_retracted)

    def release_memory_occupation(self):
        self.queue.clear()
        self.retracted_queue.clear()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()

    def resume_retracted_reqs(
        self, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        # TODO refactor the scheduling part, reuse with the unified engine logic as much as possible
        # 中译：恢复被回退（retracted）的请求——在显存回升后，尽量把回退队列中的请求
        #       重新预分配并回载其 KV。
        #       TODO（原注）：重构调度部分，尽可能复用统一引擎（unified engine）的逻辑。

        # allocate memory
        # 中译：先估算可分配的 token 预算（区分是否使用 SWA 尾部预分配）。
        resumed_reqs = []
        indices_to_remove = set()
        uses_swa_tail_prealloc = self._uses_swa_tail_prealloc()
        if uses_swa_tail_prealloc:
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(count_retracted=False)
            )
        else:
            full_allocatable_tokens = self._allocatable_token_budgets(
                count_retracted=False
            )

        for i, req in enumerate(self.retracted_queue):
            if rids_to_check is not None and req.rid not in rids_to_check:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            full_required, swa_required = self._prealloc_required_tokens(req)
            if full_required > full_allocatable_tokens:
                break
            if uses_swa_tail_prealloc and swa_required > swa_allocatable_tokens:
                break

            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            self._pre_alloc(req)
            full_allocatable_tokens -= full_required
            if uses_swa_tail_prealloc:
                swa_allocatable_tokens -= swa_required

            # load from cpu, release the cpu copy
            # 中译：从 CPU 回载 KV Cache 到显存，并释放 CPU 上的副本。
            req.load_kv_cache(self.req_to_token_pool, self.token_to_kv_pool_allocator)

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]

        return resumed_reqs

    def _update_handshake_waiters(
        self, rids_to_check: Optional[List[str]] = None
    ) -> None:
        if not self.queue:
            return

        # Still poll if any receiver was aborted, otherwise it stays stuck.
        # 中译：即使所有请求都已「等待输入」，只要有接收器被 abort（Failed）也仍需轮询，
        #       否则那个失败请求会一直卡住无法推进。
        if all(decode_req.waiting_for_input for decode_req in self.queue) and not any(
            decode_req.kv_receiver.conclude_state == KVPoll.Failed
            for decode_req in self.queue
        ):
            return

        polls = poll_and_all_reduce(
            [decode_req.kv_receiver for decode_req in self.queue], self.gloo_group
        )

        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
                decode_req.req.time_stats.set_bootstrap_done_time()
            elif poll == KVPoll.Failed:
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                is_propagated = False
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                    is_propagated = getattr(e, "is_from_another_rank", False)
                # Mute error message for propagated exceptions to avoid duplicate logging
                # 中译：对于从其他 rank 传播而来的异常，静默错误信息（降为 debug）以避免重复日志。
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                if self.scheduler.metrics_reporter.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def _ensure_prefill_info(
        self, addr_to_reqs: Dict[str, List[DecodeRequest]]
    ) -> Tuple[Dict[str, List[DecodeRequest]], List[DecodeRequest]]:
        """Non-blocking ensure parallel info for each addr.
        Returns (ready_addrs, remaining_reqs).

        中译：非阻塞地为每个（bootstrap）地址确保已获取 prefill 侧的并行信息。
              受重试间隔与最大重试次数控制；达到上限仍失败则 abort 对应请求。
              返回（已就绪的地址 -> 请求列表，尚需等待重试的剩余请求）。
        """
        ready: Dict[str, List[DecodeRequest]] = {}
        remaining: List[DecodeRequest] = []

        now = time.monotonic()
        for bootstrap_addr, reqs in addr_to_reqs.items():
            last_attempt = self._ensure_last_attempt_time.get(bootstrap_addr)
            if last_attempt is not None and (
                now - last_attempt < self._ensure_retry_interval
            ):
                remaining.extend(reqs)
                continue

            self._ensure_last_attempt_time[bootstrap_addr] = now

            if self.kv_manager.try_ensure_parallel_info(bootstrap_addr):
                if bootstrap_addr in self._ensure_retry_count:
                    del self._ensure_retry_count[bootstrap_addr]
                if bootstrap_addr in self._ensure_last_attempt_time:
                    del self._ensure_last_attempt_time[bootstrap_addr]
                ready[bootstrap_addr] = reqs
                continue

            count = self._ensure_retry_count.get(bootstrap_addr, 0) + 1
            self._ensure_retry_count[bootstrap_addr] = count

            if count >= self._max_ensure_retries:
                error_msg = f"Could not fetch prefill parallel info from {bootstrap_addr} after {count} attempts"
                logger.error(error_msg)
                for decode_req in reqs:
                    decode_req.kv_receiver.abort()
                del self._ensure_retry_count[bootstrap_addr]
                del self._ensure_last_attempt_time[bootstrap_addr]
            else:
                remaining.extend(reqs)

        return ready, remaining

    def _resolve_pending_reqs(self) -> None:
        """Batch-resolve prefill_dp_ranks for pending requests and initialize receivers.

        中译：批量为 pending 请求解析 prefill 侧 dp rank，并初始化其 KV 接收器。
              处理流程分两趟：先确保拿到并行信息（Pass 1），再对已就绪地址解析 dp rank（Pass 2）。
        """
        if not self.pending_reqs:
            return

        # Group pending requests by bootstrap_addr
        # 中译：按 bootstrap 地址对 pending 请求分组（同一 prefill 节点的请求归为一组）。
        addr_to_reqs: Dict[str, List[DecodeRequest]] = {}
        for decode_req in self.pending_reqs:
            addr = _bootstrap_addr(decode_req.req)
            addr_to_reqs.setdefault(addr, []).append(decode_req)

        # Pass 1: ensure parallel info for each addr
        # 中译：第一趟——为每个地址确保已拿到 prefill 侧并行信息。
        ready_addrs, remaining = self._ensure_prefill_info(addr_to_reqs)

        resolved: List[Tuple[DecodeRequest, int]] = []
        for bootstrap_addr, decode_reqs in ready_addrs.items():
            need_query: List[DecodeRequest] = []
            for decode_req in decode_reqs:
                prefill_dp_rank = self._resolve_prefill_dp_rank(decode_req.req)
                if prefill_dp_rank is not None:
                    resolved.append((decode_req, prefill_dp_rank))
                else:
                    need_query.append(decode_req)

            # Pass 2: resolve dp rank for addrs whose info is available
            # 中译：第二趟——对那些已拿到信息但本地仍无法直接推导的请求，
            #       向 prefill 侧批量查询 dp rank（按 bootstrap_room）。
            if need_query:
                rooms = [decode_req.req.bootstrap_room for decode_req in need_query]
                room_to_rank = CommonKVReceiver.query_prefill_dp_ranks(
                    bootstrap_addr, rooms
                )
                for decode_req in need_query:
                    prefill_dp_rank = room_to_rank.get(
                        str(decode_req.req.bootstrap_room)
                    )
                    if prefill_dp_rank is not None:
                        resolved.append((decode_req, int(prefill_dp_rank)))
                    else:
                        remaining.append(decode_req)

        self.pending_reqs = remaining

        for decode_req, prefill_dp_rank in resolved:
            decode_req.kv_receiver.init(prefill_dp_rank)

    def pop_preallocated(
        self, rids_to_check: Optional[List[str]] = None
    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:
        """Pop the preallocated requests from the pending queue (FIFO).

        中译：从预分配队列中按 FIFO（先入先出）弹出已完成预分配的请求。
              流程：先解析 pending 请求、更新握手状态；将失败请求出队；然后在显存/
              元数据 buffer 等预算允许的前提下，尽量为已握手完成的请求预分配 KV。
              返回（已预分配请求列表，失败请求列表）。
        """
        # 中译：先解析 pending 请求（拿 dp rank、init 接收器），再更新握手等待者状态。
        self._resolve_pending_reqs()
        self._update_handshake_waiters(rids_to_check)

        failed_reqs = []
        preallocated_reqs = []
        indices_to_remove = set()

        # We need to make sure that the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request
        # Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.
        # 中译：必须保证「在途 token 数 + 可分配 token 数」大于每个在途请求的最大输入+输出长度；
        #       否则可能出现：某个正在 decode 的请求显存耗尽（OOM），而其他请求都卡在无法
        #       回退（retract）的传输队列里，造成死锁。这里先算出可回退的 token 总量。
        retractable_tokens = sum(
            len(r.origin_input_ids) + len(r.output_ids)
            for r in self.scheduler.running_batch.reqs
        )

        uses_swa_tail_prealloc = self._uses_swa_tail_prealloc()
        swa_allocatable_tokens = 0
        if uses_swa_tail_prealloc:
            retractable_swa_tokens = sum(
                self._swa_retractable_len(r) for r in self.scheduler.running_batch.reqs
            )
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    retractable_swa_tokens=retractable_swa_tokens,
                    count_retracted=True,
                )
            )
        else:
            retractable_swa_tokens = 0
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens, count_retracted=True
            )
        reserved_restore_tokens = self._hicache_pending_restore_tokens()
        full_allocatable_tokens -= reserved_restore_tokens
        # Sort by priority before any index-based bookkeeping so that both the
        # abort-scan loop and the preallocation loop operate on the same order.
        # 中译：在任何基于下标的记账之前先按优先级排序，确保「扫描 abort 循环」与
        #       「预分配循环」作用于同一个顺序（否则下标会错位）。
        if self.scheduler.enable_priority_scheduling:
            priority_sign = (
                1 if self.scheduler.schedule_low_priority_values_first else -1
            )
            self.queue.sort(key=lambda r: r.req.priority * priority_sign)

        # First, remove all failed requests from the queue
        # 中译：第一步：先把队列中所有已失败（FINISH_ABORT）的请求清理出队，
        #       回流给客户端并释放其接收器。
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                self.scheduler.output_streamer.stream_output(
                    [decode_req.req],
                    decode_req.req.return_logprob,
                )
                decode_req.kv_receiver.clear()
                decode_req.kv_receiver = None
                failed_reqs.append(decode_req)
                indices_to_remove.add(i)

        # HiSparse physical constraint: max requests by device buffer capacity.
        # Each admitted req needs padded_buffer_size from hisparse device pool.
        # waiting_queue reqs already have device buffers (allocated in admit_request_direct),
        # only transfer_queue reqs are pending device buffer allocation.
        # 中译：HiSparse 物理约束：可接纳请求数受设备 buffer 容量限制。
        #       每个被接纳请求需从 hisparse 设备池占用 padded_buffer_size；
        #       waiting_queue 中的请求已拥有设备 buffer（在 admit_request_direct 中分配），
        #       只有 transfer_queue 中的请求尚待分配设备 buffer。
        hisparse_req_budget = float("inf")
        if self.scheduler.enable_hisparse:
            hisparse_avail = (
                self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
            )
            hisparse_req_budget = max(
                0,
                hisparse_avail // self.scheduler.hisparse_coordinator.padded_buffer_size
                - len(self.transfer_queue.queue),
            )

        # Then, preallocate the remaining requests if possible
        # 中译：第二步：在各项预算（req_pool、元数据 buffer、HiSparse、显存 token）允许的情况下，
        #       为剩余已握手请求逐个预分配 KV（任一预算不够就 break，保证 FIFO）。
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if i in indices_to_remove:
                continue

            if not decode_req.waiting_for_input:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                break

            if hisparse_req_budget <= 0:
                break

            # Memory estimation: don't add if the projected memory cannot be met
            # TODO: add new_token ratio
            # 中译：显存估算：若预估显存无法满足则不接纳该请求。TODO（原注）：加入 new_token 比率。
            origin_input_len = len(decode_req.req.origin_input_ids)
            prefix_match: Optional[DecodePrefixMatch] = None
            if self.scheduler.server_args.disaggregation_decode_enable_radix_cache:
                # Match prefix against decode's radix cache.
                # 中译：在 decode 侧 radix cache 中做前缀匹配（并锁住命中节点）。
                prefix_match = self._match_prefix_and_lock(decode_req.req)
                prefix_indices = prefix_match.prefix_indices
                # prefix_len: tokens already on device (L1 hit).
                # total_prefix_len: full prefix promised to prefill
                # (L1 + L2 host hit + L3 storage hit), sent as PD
                # protocol's `decode_prefix_len`. The [prefix_len, total)
                # gap is filled by HiCache loadback later.
                # 中译：prefix_len：已在设备（显存）上的 token 数（L1 命中）。
                #       total_prefix_len：承诺给 prefill 的完整前缀长度
                #       （L1 + L2 主机命中 + L3 存储命中），作为 PD 协议的 `decode_prefix_len` 发送。
                #       [prefix_len, total) 之间的缺口稍后由 HiCache 回载（loadback）填补。
                prefix_len = prefix_match.l1_prefix_len
                total_prefix_len = prefix_match.decode_prefix_len

                fill_len = origin_input_len + max(len(decode_req.req.output_ids) - 1, 0)
                required_alloc_tokens = self._required_alloc_tokens(
                    fill_len=fill_len, prefix_len=prefix_len
                )
                # Matching may lock previously-evictable radix pages, so refresh
                # the admission budget against the post-lock pool state before we
                # decide whether this request still fits.
                # 中译：前缀匹配可能会锁住之前可驱逐的 radix 页，因此在判断该请求是否仍能装下
                #       之前，先基于「加锁后的池状态」重新刷新接纳预算。
                full_allocatable_tokens = self._allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    count_retracted=True,
                    extra_reserved_reqs=len(preallocated_reqs),
                    hicache_reserved_tokens=reserved_restore_tokens,
                )
            else:
                prefix_indices = None
                prefix_len = 0
                total_prefix_len = 0
                required_alloc_tokens = origin_input_len

            required_tokens_for_request = (
                required_alloc_tokens + self.num_reserved_decode_tokens
            )

            if (
                max(
                    required_tokens_for_request,
                    origin_input_len
                    - prefix_len
                    + min(
                        decode_req.req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKEN,
                    )
                    - retractable_tokens,
                )
                > full_allocatable_tokens
            ):
                if prefix_len > 0:
                    self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                break
            if required_tokens_for_request > full_allocatable_tokens:
                if prefix_len > 0:
                    self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                break

            if uses_swa_tail_prealloc:
                _, swa_required = self._prealloc_required_tokens(decode_req.req)
                _, swa_len = self._prealloc_kv_lens(decode_req.req)
                max_new_tokens = min(
                    decode_req.req.sampling_params.max_new_tokens,
                    CLIP_MAX_NEW_TOKEN,
                )
                if (
                    max(
                        swa_required,
                        swa_len + max_new_tokens - retractable_swa_tokens,
                    )
                    > swa_allocatable_tokens
                ):
                    if prefix_len > 0:
                        self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                    break

            dst_kv_indices = self._pre_alloc(
                decode_req.req,
                prefix_indices,
                prefix_len,
                total_prefix_len,
            )
            decode_req.prefix_match = prefix_match
            if self.scheduler.enable_decode_hicache:
                self._start_hicache_prefetch(decode_req.req, prefix_match)
            hisparse_req_budget -= 1
            # Recompute from actual pool state for the next queue entry.
            # This accounts for page rounding and newly locked evictable cache.
            # 中译：为下一个队列项重新从实际池状态计算预算，以计入页对齐开销及新锁住的可驱逐缓存。
            if prefix_match is not None:
                reserved_restore_tokens += prefix_match.restore_token_count
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens,
                count_retracted=True,
                extra_reserved_reqs=len(preallocated_reqs) + 1,
                hicache_reserved_tokens=reserved_restore_tokens,
            )
            if uses_swa_tail_prealloc:
                # SWA budget uses simple decrement (no radix cache eviction in
                # the SWA pool, so page-rounding drift is negligible).
                # 中译：SWA 预算直接递减即可（SWA 池不做 radix 缓存驱逐，页对齐误差可忽略）。
                swa_allocatable_tokens -= swa_required
            decode_req.req.cache_protected_len = total_prefix_len

            page_size = self.token_to_kv_pool_allocator.page_size
            kv_transfer_page_size = page_size
            if self.scheduler.enable_hisparse:
                # Direct-to-host sends host/C4 rows; keep allocator.page_size
                # logical and use the compressed page size only for these indices.
                # 中译：直达主机（Direct-to-host）方式发送的是 host/C4 行；保持 allocator.page_size
                #       为逻辑页大小，仅对这些索引使用压缩后的页大小。
                kv_transfer_page_size = getattr(
                    self.token_to_kv_pool_allocator,
                    "hisparse_page_size",
                    page_size,
                )
                # Must cast to int32 for ZMQ serialization -- from_zmq reads np.int32.
                # 中译：必须转为 int32 以便 ZMQ 序列化（from_zmq 按 np.int32 读取）。
                kv_indices = (
                    dst_kv_indices[: origin_input_len - prefix_len]
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                )
            else:
                # Only send delta indices (beyond prefix) to prefill.
                # 中译：仅向 prefill 发送「前缀之外的增量（delta）索引」（前缀部分已命中，无需重传）。
                kv_indices = (
                    self.req_to_token_pool.req_to_token[decode_req.req.req_pool_idx][
                        total_prefix_len:origin_input_len
                    ]
                    .cpu()
                    .numpy()
                )

            seq_len = len(decode_req.req.origin_input_ids)

            def _mamba_payload():
                return [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        decode_req.req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]

            def _swa_payload():
                window_size = self.scheduler.sliding_window_size
                window_start = max(0, seq_len - window_size)
                window_start = page_align_floor(window_start, page_size)
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, window_start:seq_len
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
                    decode_req.req.req_pool_idx, :seq_len
                ]
                # Indexer lives on device pool; always use device page_size
                # 中译：indexer 位于设备池上，因此总是使用设备端的 page_size。
                device_page_size = self.token_to_kv_pool.page_size
                return kv_to_page_indices(
                    kv_indices_full.cpu().numpy(), device_page_size
                )

            def _swa_ring_payload():
                # Mirror of prefill _swa_ring_payload using this side's req_pool_idx.
                # Same window positions and order -> positional match with prefill.
                # 中译：与 prefill 侧 _swa_ring_payload 镜像对应，但使用本侧自己的 req_pool_idx。
                #       采用相同的窗口位置与顺序，从而在位置上与 prefill 对应得上。
                ring_stride = self.token_to_kv_pool.unified_swa_ring_size
                window_size = self.token_to_kv_pool.unified_swa_window
                window_start = max(0, seq_len - window_size)
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                state_slot = int(decode_req.req.req_pool_idx)
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                return ring_rows.astype(np.int32)

            state_types = self.kv_manager.kv_args.state_types
            state_indices: Optional[List] = []
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

            decode_req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert decode_req.metadata_buffer_index is not None
            page_indices = kv_to_page_indices(kv_indices, kv_transfer_page_size)
            decode_req.kv_receiver.send_metadata(
                page_indices,
                decode_req.metadata_buffer_index,
                state_indices,
                decode_prefix_len=total_prefix_len,
            )
            if (
                self.transfer_queue.enable_staging
                and hasattr(decode_req.kv_receiver, "require_staging")
                and decode_req.kv_receiver.require_staging
            ):
                self.transfer_queue.staging_handler.register_decode_req(
                    decode_req.req.bootstrap_room, decode_req
                )
            preallocated_reqs.append(decode_req)
            indices_to_remove.add(i)
            decode_req.req.time_stats.set_decode_transfer_queue_entry_time()

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs, failed_reqs

    @property
    def num_tokens_pre_allocated(self):
        return sum(decode_req.req.fill_len for decode_req in self.transfer_queue.queue)

    def _need_space_for_single_req(
        self, retractable_tokens: Optional[int] = None
    ) -> int:
        need_space_for_single_req = (
            max(
                [
                    min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                    + len(x.origin_input_ids)
                    - retractable_tokens
                    for x in self.scheduler.running_batch.reqs
                ]
            )
            if retractable_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
            else 0
        )
        return need_space_for_single_req

    def _active_req_count(self, extra_reserved_reqs: int = 0) -> int:
        return (
            len(self.scheduler.running_batch.reqs)
            + len(self.transfer_queue.queue)
            + len(self.scheduler.waiting_queue)
            + extra_reserved_reqs
        )

    def _active_reserved_tokens(
        self, n_active: Optional[int] = None, extra_reserved_reqs: int = 0
    ) -> int:
        if n_active is None:
            n_active = self._active_req_count(extra_reserved_reqs)
        return self.num_reserved_decode_tokens * n_active

    def _swa_aware_allocatable_token_budgets(
        self,
        retractable_tokens: Optional[int] = None,
        retractable_swa_tokens: Optional[int] = None,
        count_retracted: bool = True,
    ) -> Tuple[int, int]:
        n_active = self._active_req_count()
        reserved_tokens = self._active_reserved_tokens(n_active)

        full_allocatable_tokens = self._allocatable_token_budgets(
            retractable_tokens=retractable_tokens,
            count_retracted=count_retracted,
            reserved_tokens=reserved_tokens,
        )

        return full_allocatable_tokens, self._swa_tail_allocatable_token_budget(
            retractable_tokens=retractable_tokens,
            retractable_swa_tokens=retractable_swa_tokens,
            count_retracted=count_retracted,
            n_active=n_active,
            reserved_tokens=reserved_tokens,
        )

    def _allocatable_token_budgets(
        self,
        retractable_tokens: Optional[int] = None,
        count_retracted: bool = True,
        extra_reserved_reqs: int = 0,
        reserved_tokens: Optional[int] = None,
        hicache_reserved_tokens: int = 0,
    ) -> int:
        need_space_for_single_req = self._need_space_for_single_req(retractable_tokens)
        if reserved_tokens is None:
            reserved_tokens = self._active_reserved_tokens(
                extra_reserved_reqs=extra_reserved_reqs
            )

        if self.scheduler.enable_hisparse:
            # HiSparse pre-alloc only allocates logical indices (alloc_logical_only),
            # so the logical pool is the binding constraint for admission control.
            # 中译：HiSparse 预分配仅分配逻辑索引（alloc_logical_only），因此逻辑池
            #       才是接纳控制（admission control）的真正约束。
            available_size = (
                self.token_to_kv_pool_allocator.logical_attn_allocator.available_size()
            )
        elif self._uses_swa_tail_prealloc():
            available_size = self.token_to_kv_pool_allocator.full_available_size()
            if self.scheduler.server_args.disaggregation_decode_enable_radix_cache:
                available_size += self.tree_cache.evictable_size()
        else:
            available_size = self.token_to_kv_pool_allocator.available_size()
            # Include evictable decode-radix cache entries in the budget -- they
            # can be freed on demand before allocation.
            # 中译：把可驱逐的 decode-radix 缓存项也计入预算——它们可在分配前按需释放。
            if self.scheduler.server_args.disaggregation_decode_enable_radix_cache:
                available_size += self.tree_cache.evictable_size()
        allocatable_tokens = available_size - max(
            reserved_tokens, need_space_for_single_req
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        # 中译：注意——若上一个预构建（prebuilt）extend 批刚刚完成，而下一轮立即进入
        #       `pop_preallocated`，此时该 extend 批不在任何队列中，因此需要在这里显式扣除
        #       其预留的 token 槽位。
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            allocatable_tokens -= self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )

        if count_retracted:
            for req in self.retracted_queue:
                full_required, _ = self._prealloc_required_tokens(req)
                allocatable_tokens -= full_required

        allocatable_tokens -= hicache_reserved_tokens
        return allocatable_tokens

    def _swa_tail_allocatable_token_budget(
        self,
        retractable_tokens: Optional[int] = None,
        retractable_swa_tokens: Optional[int] = None,
        count_retracted: bool = True,
        n_active: Optional[int] = None,
        reserved_tokens: Optional[int] = None,
    ) -> int:
        need_swa_space_for_single_req = self._need_space_for_single_req(
            retractable_tokens
        )
        if (
            retractable_swa_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
        ):
            need_swa_space_for_single_req = max(
                self._swa_tail_len(len(x.origin_input_ids))
                + min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                - retractable_swa_tokens
                for x in self.scheduler.running_batch.reqs
            )

        if n_active is None:
            n_active = self._active_req_count()
        if reserved_tokens is None:
            reserved_tokens = self._active_reserved_tokens(n_active)

        # SWA growth is bounded by the sliding window: once a req's SWA
        # footprint reaches `sliding_window_size`, further decode tokens
        # evict old ones and net growth is zero. The linear reservation
        # `num_reserved_decode_tokens * n_active` (correct for the full
        # pool) over-reserves SWA in steady state. Cap by the actual
        # remaining headroom up to per-req window cap.
        # 中译：SWA 的增长受滑动窗口限制：一旦某请求的 SWA 占用达到 `sliding_window_size`，
        #       后续解码 token 会驱逐旧 token，净增长为零。线性预留
        #       `num_reserved_decode_tokens * n_active`（对完整池而言是正确的）在稳态下会
        #       对 SWA 过度预留。因此按「实际剩余余量」且不超过「每请求窗口上限」来封顶。
        window_size = self.scheduler.sliding_window_size or 0
        swa_total = self.token_to_kv_pool_allocator.size_swa
        swa_used = swa_total - self.token_to_kv_pool_allocator.swa_available_size()
        swa_growth_potential = max(0, n_active * window_size - swa_used)
        swa_reserved_tokens = min(reserved_tokens, swa_growth_potential)
        swa_allocatable_tokens = (
            self.token_to_kv_pool_allocator.swa_available_size()
            - max(swa_reserved_tokens, need_swa_space_for_single_req)
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        # 中译：同上——若上一个 prebuilt extend 批刚完成且本轮立即进入 pop_preallocated，
        #       它不在任何队列中，故需在此显式扣除其 SWA 预留 token。
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            prebuilt_reserved_tokens = self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )
            prebuilt_n = len(self.scheduler.last_batch.reqs)
            prebuilt_swa_growth = max(0, prebuilt_n * window_size - swa_used)
            swa_allocatable_tokens -= min(prebuilt_reserved_tokens, prebuilt_swa_growth)

        if count_retracted:
            for req in self.retracted_queue:
                _, swa_required = self._prealloc_required_tokens(req)
                swa_allocatable_tokens -= swa_required

        return swa_allocatable_tokens

    def _required_alloc_tokens(self, *, fill_len: int, prefix_len: int) -> int:
        """Compute the number of KV-pool tokens that must be *newly* allocated
        to grow a sequence from ``prefix_len`` to ``fill_len``, accounting for
        page-size alignment.

        中译：计算把一条序列从 ``prefix_len`` 增长到 ``fill_len`` 时，KV 池需要
              **新分配**的 token 数量，并考虑分页（page）对齐。

              - ``fill_len``：目标总长度（该请求最终要占用的 token 数）。
              - ``prefix_len``：已分配 / 已复用的前缀长度（无需再分配的部分）。

              分两种情况：
              1) ``page_size == 1``：逐 token 分页，直接返回新增 token 数
                 ``fill_len - prefix_len``。
              2) ``page_size > 1``：KV 池以“页”为最小分配粒度，需先算出从
                 ``prefix_len`` 增长到 ``fill_len`` 会跨越几个尚未分配的新页
                 （``get_num_new_pages``），再乘以 ``page_size`` 得到按页对齐后
                 实际要占用的 token 数。由于页可能未填满，返回值通常 ≥ 新增
                 token 数（即存在页内空洞）。
        """
        page_size = self.token_to_kv_pool_allocator.page_size
        if page_size == 1:
            return fill_len - prefix_len

        num_new_pages = get_num_new_pages(
            seq_lens=torch.tensor([fill_len], dtype=torch.int64),
            prefix_lens=torch.tensor([prefix_len], dtype=torch.int64),
            page_size=page_size,
        )
        return num_new_pages * page_size

    def _pre_alloc(
        self,
        req: Req,
        prefix_indices: Optional[torch.Tensor] = None,
        prefix_len: Optional[int] = None,
        total_prefix_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Pre-allocate the memory for req_to_token and token_kv_pool.

        ``prefix_len`` is the L1 device-resident prefix length (already
        backed by ``prefix_indices``). ``total_prefix_len`` is the full
        prefix committed to prefill as ``decode_prefix_len`` (L1 + L2 + L3);
        the ``[prefix_len, total_prefix_len)`` gap is filled later by HiCache
        loadback.

        中译：为 req_to_token 与 token_kv_pool 预分配内存。
              ``prefix_len``：L1 层已驻留于设备（显存）的前缀长度（已由 ``prefix_indices`` 支撑）。
              ``total_prefix_len``：承诺给 prefill 的完整前缀长度（即 ``decode_prefix_len``，
              含 L1 + L2 + L3）；区间 ``[prefix_len, total_prefix_len)`` 的缺口稍后由 HiCache 回载填补。
        """
        if prefix_len is None:
            prefix_len = 0
        if total_prefix_len is None:
            total_prefix_len = prefix_len

        req_pool_indices = self.req_to_token_pool.alloc([req])

        assert (
            req_pool_indices is not None
        ), "req_pool_indices is full! There is a bug in memory estimation."

        fill_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        req.kv_allocated_len = fill_len
        req.kv_committed_len = fill_len

        if prefix_len > 0:
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(0, prefix_len)), prefix_indices
            )

        # TODO(retraction): when retraction is implemented with radix cache
        # awareness, a retracted request should re-match the tree here
        # instead of re-allocating from scratch. See resume_retracted_reqs.
        # 中译：TODO（回退）：当回退机制实现为感知 radix cache 后，被回退的请求应在此处
        #       重新匹配前缀树，而非从头重新分配。参见 resume_retracted_reqs。
        delta_len = fill_len - total_prefix_len
        required_alloc_tokens = self._required_alloc_tokens(
            fill_len=fill_len, prefix_len=prefix_len
        )

        # Evict cached entries if the pool doesn't have enough free pages.
        # 中译：若池中空闲页不够，先驱逐（evict）一些可驱逐的缓存项以腾出空间。
        if (
            self.scheduler.server_args.disaggregation_decode_enable_radix_cache
            and self.token_to_kv_pool_allocator.available_size() < required_alloc_tokens
        ):
            num_to_evict = (
                required_alloc_tokens - self.token_to_kv_pool_allocator.available_size()
            )
            result = self.tree_cache.evict(EvictParams(num_tokens=num_to_evict))
            if self.token_to_kv_pool_allocator.available_size() < required_alloc_tokens:
                logger.warning(
                    f"Eviction insufficient: needed {required_alloc_tokens} tokens, "
                    f"available {self.token_to_kv_pool_allocator.available_size()} "
                    f"after evicting {result.num_tokens_evicted}/{num_to_evict} tokens. "
                    f"evictable_size={self.tree_cache.evictable_size()}, "
                    f"protected_size={self.tree_cache.protected_size()}, "
                    f"fill_len={fill_len}, prefix_len={prefix_len}, "
                    f"total_prefix_len={total_prefix_len}, delta_len={delta_len}, "
                    f"page_size={self.token_to_kv_pool_allocator.page_size}, "
                    f"req={req.rid}"
                )

        if self.scheduler.enable_hisparse:
            # HiSparse is incompatible with decode-side L1 radix cache. Keep
            # this path on the upstream full-allocation semantics.
            # 中译：HiSparse 与 decode 侧 L1 radix cache 不兼容，故此路径保持上游的「全量分配」语义。
            assert prefix_len == 0

            # Direct-to-host path: only allocate logical indices (no hisparse
            # device indices) and allocate host indices for RDMA destination.
            # 中译：直达主机路径：仅分配逻辑索引（不分配 hisparse 设备索引），
            #       并为 RDMA 目的地分配主机端索引。
            coordinator = self.scheduler.hisparse_coordinator
            device = self.token_to_kv_pool_allocator.device
            kv_loc = self.token_to_kv_pool_allocator.alloc_logical_only(
                prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
                extend_num_tokens=fill_len,
            )

            # Allocate host indices for the RDMA transfer target.
            host_indices = coordinator.mem_pool_host.alloc_paged_token_slots(
                coordinator.req_to_host_pool,
                coordinator.req_to_host_pool_allocated_len,
                req.req_pool_idx,
                0,
                coordinator.host_token_len(fill_len),
            )
        elif self.token_to_kv_pool_allocator.page_size == 1:
            kv_loc = self.token_to_kv_pool_allocator.alloc(delta_len)
        else:
            device = self.token_to_kv_pool_allocator.device
            last_loc = (
                prefix_indices[-1:].to(dtype=torch.int64, device=device)
                if prefix_len > 0
                else torch.tensor([-1], dtype=torch.int64, device=device)
            )
            if self._uses_swa_tail_prealloc() and prefix_len == 0:
                # Tail-only SWA allocation: only valid when prefix_len == 0.
                # When prefix_len > 0 (radix cache hit), we fall back to
                # alloc_extend which allocates SWA at full page count; the
                # SWA budget in that case may slightly under-estimate.
                # 中译：仅分配 SWA 尾部：仅当 prefix_len == 0 时有效。
                #       当 prefix_len > 0（radix 缓存命中）时回退到 alloc_extend，它按完整页数分配 SWA；
                #       那种情况下 SWA 预算可能会略微低估。
                kv_loc = self.token_to_kv_pool_allocator.alloc_extend_swa_tail(
                    prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                    prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                    seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                    seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                    last_loc=last_loc,
                    extend_num_tokens=fill_len,
                    swa_tail_len=self._swa_tail_len(fill_len),
                )
            else:
                kv_loc = self.token_to_kv_pool_allocator.alloc_extend(
                    prefix_lens=torch.tensor(
                        [total_prefix_len], dtype=torch.int64, device=device
                    ),
                    prefix_lens_cpu=torch.tensor([total_prefix_len], dtype=torch.int64),
                    seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                    seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                    last_loc=last_loc,
                    extend_num_tokens=delta_len,
                )

        assert kv_loc is not None, (
            f"KV cache is full! Bug in memory estimation. "
            f"available={self.token_to_kv_pool_allocator.available_size()}, "
            f"evictable={self.tree_cache.evictable_size()}, "
            f"protected={self.tree_cache.protected_size()}, "
            f"required_alloc={required_alloc_tokens}, delta={delta_len}, "
            f"fill={fill_len}, prefix={prefix_len}, total_prefix={total_prefix_len}, "
            f"page_size={self.token_to_kv_pool_allocator.page_size}, "
            f"req={req.rid}"
        )

        self.req_to_token_pool.write(
            (
                req.req_pool_idx,
                slice(total_prefix_len, total_prefix_len + len(kv_loc)),
            ),
            kv_loc,
        )

        # Truncate fill_len to kv_committed_len so cache_unfinished_req only
        # inserts committed KV into the radix tree. The last output token
        # hasn't had KV committed yet (output_ids is 1 ahead).
        # 中译：把 fill_len 截断到 kv_committed_len，以保证 cache_unfinished_req 只把已提交的 KV
        #       插入 radix 树。最后一个输出 token 的 KV 尚未提交（output_ids 比 KV 领先 1 个）。
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
        req.fill_len = req.kv_committed_len
        # Set prefix_indices so downstream consumers (init_next_round_input,
        # prepare_for_extend) see the correct prefix length. In the agg path
        # this is done inside init_next_round_input, but decode-disagg needs
        # allocation info before batch assembly so we set it here.
        # 中译：设置 prefix_indices，以便下游消费者（init_next_round_input、prepare_for_extend）
        #       能看到正确的前缀长度。在聚合（agg）路径中这是在 init_next_round_input 内完成的，
        #       但 decode-disagg 在组批前就需要分配信息，故在此处设置。
        req.prefix_indices = (
            prefix_indices if prefix_len > 0 else torch.empty((0,), dtype=torch.int64)
        )
        req.set_extend_input_len(req.fill_len - total_prefix_len)

        # Return the transfer destination indices:
        # 中译：返回传输目的地索引（供 prefill 向此处写入 KV）：
        #       HiSparse 返回主机端索引，否则返回显存 KV 位置 kv_loc。
        if self.scheduler.enable_hisparse:
            return host_indices
        return kv_loc


class DecodeTransferQueue(DecodeHiCacheTransferMixin):
    """
    Store the requests that is polling kv

    中译：传输队列。存放「正在轮询 KV 传输状态」的请求——即已预分配好目标显存、
          等待 prefill 节点把 KV 传过来的请求。传输完成后会移入等待队列。
    """

    def __init__(
        self,
        gloo_group: ProcessGroup,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        tp_rank: int,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[DecodeRequest] = []
        self.gloo_group = gloo_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tree_cache = tree_cache
        self.spec_algorithm = scheduler.spec_algorithm
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.staging_handler = None

    def add(self, decode_req: DecodeRequest) -> None:
        self.queue.append(decode_req)

    def extend(self, decode_reqs: List[DecodeRequest]) -> None:
        self.queue.extend(decode_reqs)
        if self.enable_staging:
            for dr in decode_reqs:
                if (
                    hasattr(dr.kv_receiver, "require_staging")
                    and dr.kv_receiver.require_staging
                ):
                    self.staging_handler.register_decode_req(dr.req.bootstrap_room, dr)

    def _commit_transfer_to_req(self, decode_req: DecodeRequest):
        idx = decode_req.metadata_buffer_index
        (
            output_id,
            cached_tokens,
            output_token_logprobs_val,
            output_token_logprobs_idx,
            output_top_logprobs_val,
            output_top_logprobs_idx,
            output_topk_p,
            output_topk_index,
            output_hidden_states,
            output_bootstrap_room,
        ) = self.metadata_buffers.get_buf(idx)

        # Validate bootstrap_room to detect context corruption
        # 中译：校验 bootstrap_room，以检测上下文错乱（元数据 buffer 索引碰撞）。
        actual_room = output_bootstrap_room[0].item()
        expected_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )

        if _is_fake_transfer(decode_req.req, self.scheduler.server_args):
            pass
        elif actual_room == 0:
            # Should never happen: _poll_with_metadata_gate already confirmed
            # readiness on all TP ranks. Abort deterministically to avoid
            # cross-rank queue divergence.
            # 中译：理论上不应发生：_poll_with_metadata_gate 已在所有 TP rank 上确认就绪。
            #       这里确定性地 abort，以避免跨 rank 的队列状态发生分歧。
            logger.error(
                f"Metadata unexpectedly not ready after readiness gate: "
                f"request {decode_req.req.rid}, bootstrap_room={expected_room}, "
                f"metadata_buffer_index={idx}"
            )
            prepare_abort(
                decode_req.req,
                "Metadata unexpectedly not ready after readiness gate "
                "(bootstrap_room=0)",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return
        elif actual_room != expected_room:
            # Real corruption detected (mismatch)
            # Abort the request and remove from the queue
            # 中译：检测到真实的上下文错乱（room 不匹配）：abort 该请求并从队列中移除。
            error_msg = (
                f"Context corruption detected: Request {decode_req.req.rid} "
                f"(bootstrap_room={expected_room}) received metadata from "
                f"bootstrap_room={actual_room}. "
                f"Metadata buffer index: {idx}. "
                f"This indicates metadata buffer index collision."
            )
            logger.error(error_msg)
            prepare_abort(
                decode_req.req,
                "Metadata corruption detected - bootstrap_room mismatch",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return

        self._commit_hicache_local_restore_to_req(decode_req)

        # Case 3: Success - commit the transfer
        # 中译：情况 3：传输成功——提交本次传输结果。
        decode_req.req.output_ids.append(output_id[0].item())
        decode_req.req.cached_tokens = cached_tokens[0].item()
        # The prefill node already reported its prefix-cache hit in
        # cached_tokens[0]. Seed already_computed with it so that
        # prepare_for_prebuilt's `cached_tokens += pre_len - already_computed`
        # only adds decode-side reuse *beyond* what prefill counted, instead of
        # double-counting the shared prompt prefix (which would make
        # cached_tokens exceed prompt_tokens when decode radix cache is on).
        # 中译：prefill 节点已将其前缀缓存命中数写入 cached_tokens[0]。
        #       用它初始化 already_computed，这样 prepare_for_prebuilt 中的
        #       `cached_tokens += pre_len - already_computed` 只累加 decode 侧
        #       超出 prefill 已计入部分的复用量，而非重复计算共享的 prompt 前缀
        #       （否则在开启 decode radix cache 时 cached_tokens 会超过 prompt_tokens）。
        decode_req.req.already_computed = decode_req.req.cached_tokens
        decode_req.req.cached_tokens_device = cached_tokens[1].item()
        decode_req.req.cached_tokens_host = cached_tokens[2].item()
        decode_req.req.cached_tokens_storage = cached_tokens[3].item()
        if not self.spec_algorithm.is_none():
            decode_req.req.output_topk_p = output_topk_p
            decode_req.req.output_topk_index = output_topk_index
            decode_req.req.hidden_states_tensor = output_hidden_states

        if decode_req.req.return_logprob:
            decode_req.req.logprob.output_token_logprobs_val.append(
                output_token_logprobs_val[0].item()
            )
            decode_req.req.logprob.output_token_logprobs_idx.append(
                output_token_logprobs_idx[0].item()
            )
            decode_req.req.logprob.output_top_logprobs_val.append(
                output_top_logprobs_val[
                    : decode_req.req.logprob.top_logprobs_num
                ].tolist()
            )
            decode_req.req.logprob.output_top_logprobs_idx.append(
                output_top_logprobs_idx[
                    : decode_req.req.logprob.top_logprobs_num
                ].tolist()
            )

        decode_req.kv_receiver.clear()
        decode_req.kv_receiver = None
        decode_req.req.time_stats.set_wait_queue_entry_time()
        return

    def _poll_with_metadata_gate(self) -> List[int]:
        pollers = (
            [HiCacheRestoreGatedKVReceiver(dr) for dr in self.queue]
            if self.scheduler.enable_decode_hicache
            else [dr.kv_receiver for dr in self.queue]
        )
        return poll_and_all_reduce(
            pollers,
            self.gloo_group,
            decode_reqs=self.queue,
            metadata_buffers=self.metadata_buffers,
            server_args=self.scheduler.server_args,
        )

    def _poll_with_staging(self) -> list:
        return poll_and_all_reduce_with_staging(
            self.queue,
            self.staging_handler,
            self.gloo_group,
            metadata_buffers=self.metadata_buffers,
            server_args=self.scheduler.server_args,
        )

    def _init_staging_handler(self, kv_manager):
        """Create staging handler from kv_manager. Must be called exactly once.

        中译：从 kv_manager 创建 staging（暂存）处理器。必须且只能调用一次。
        """
        from sglang.srt.disaggregation.common.staging_handler import (
            DecodeStagingHandler,
        )

        self.staging_handler = DecodeStagingHandler.create(
            kv_manager, self.scheduler, self.tp_rank
        )
        kv_manager._staging_handler = self.staging_handler

    def pop_transferred(self, rids_to_check: Optional[List[str]] = None) -> List[Req]:
        if not self.queue:
            return []

        if self.scheduler.enable_decode_hicache:
            self._process_hicache_local_restores(
                [
                    decode_req
                    for decode_req in self.queue
                    if rids_to_check is None or decode_req.req.rid in rids_to_check
                ]
            )

        if self.enable_staging:
            polls = self._poll_with_staging()
        else:
            polls = self._poll_with_metadata_gate()

        transferred_reqs = []
        indices_to_remove = set()
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            hicache_restore_status = decode_req.hicache_restore_status
            if (
                poll == KVPoll.Failed
                or hicache_restore_status == HiCacheRestoreResult.FAILED
            ):
                error_message = (
                    f"Decode transfer failed for request rank={self.tp_rank} "
                    f"{decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                )
                is_propagated = False
                if poll == KVPoll.Failed:
                    try:
                        decode_req.kv_receiver.failure_exception()
                    except Exception as e:
                        error_message += f" with exception {e}"
                        is_propagated = getattr(e, "is_from_another_rank", False)
                self._clean_hicache_prefetch_resources(decode_req)
                # Mute error message for propagated exceptions to avoid duplicate logging
                # 中译：对于从其他 rank 传播而来的异常，静默错误信息以避免重复日志。
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.output_streamer.stream_output(
                    [decode_req.req],
                    decode_req.req.return_logprob,
                )
                if self.scheduler.enable_hisparse:
                    self.scheduler.hisparse_coordinator.request_finished(decode_req.req)
                # release pre-allocated kv cache, but don't insert into the tree since it's failed
                # 中译：释放已预分配的 KV Cache，但不插入前缀树（因为请求失败了，KV 无效）。
                release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                decode_req.kv_receiver.clear()
                decode_req.kv_receiver = None
                indices_to_remove.add(i)
                if self.scheduler.metrics_reporter.enable_metrics:
                    self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                continue
            elif poll == KVPoll.Success:
                if (
                    self.scheduler.enable_decode_hicache
                    and hicache_restore_status == HiCacheRestoreResult.PENDING
                ):
                    continue
                self._commit_transfer_to_req(decode_req)
                indices_to_remove.add(i)
                # Check if request was aborted due to corruption
                # 中译：检查请求是否因上下文错乱而被 abort。
                if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                    self.scheduler.output_streamer.stream_output(
                        [decode_req.req],
                        decode_req.req.return_logprob,
                    )
                    if self.scheduler.enable_hisparse:
                        self.scheduler.hisparse_coordinator.request_finished(
                            decode_req.req
                        )
                    self._clean_hicache_prefetch_resources(decode_req)
                    release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                    if self.scheduler.metrics_reporter.enable_metrics:
                        self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                else:
                    transferred_reqs.append(decode_req.req)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            if self.enable_staging and self.staging_handler.is_staging_room(
                self.queue[i].req.bootstrap_room
            ):
                self.staging_handler.unregister_decode_req(
                    self.queue[i].req.bootstrap_room
                )
            idx = self.queue[i].metadata_buffer_index
            assert idx != -1
            # Reset so the next owner sees actual_room == 0 ("not yet written")
            # instead of the stale value, avoiding a false-positive mismatch.
            # 中译：重置 bootstrap_room 为 0，使下一个拥有者读到 "尚未写入" 的状态，
            #       而非旧的残留值，以避免假阳性不匹配。
            self.metadata_buffers.bootstrap_room[idx] = 0
            self.req_to_metadata_buffer_idx_allocator.free(idx)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs

    def release_memory_occupation(self):
        """Clean up in-flight transfers before releasing GPU memory.

        中译：在释放 GPU 显存前清理在途传输。
        """
        self.queue.clear()

    def resume_memory_occupation(self):
        """Queues are already cleared on release; new transfers can be accepted.

        中译：队列在 release 时已清空；可接收新的传输。
        """
        pass


class SchedulerDisaggregationDecodeMixin:
    # 中译：Decode 侧调度器的分离架构混入（Mixin），提供事件循环与批次管理逻辑。

    @torch.no_grad()
    def event_loop_normal_disagg_decode(self: Scheduler):
        """A normal scheduler loop for decode worker in disaggregation mode.

        中译：分离架构下 decode 工作线程的普通（非重叠）调度循环。
              每轮循环：接收请求 → 处理 decode 队列 → 取下一批次 → 运行 → 处理结果。
        """

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            self.process_decode_queue()
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                # 中译：服务器空闲时执行自检并重新初始化一些状态。
                self.on_idle()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        # 中译：分离架构下 decode 工作线程的重叠（overlap）调度循环。
        #       与普通循环不同，调度与前向计算重叠：当前轮调度的批次在下一轮才处理结果。
        self.result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None

        def pop_and_process():
            # 中译：从结果队列中弹出一对（batch, result）并处理。
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            self.process_decode_queue()
            if self._engine_paused:
                continue

            # WAR barrier: this iter's schedule writes to shared GPU buffers wait for prev forward's reads.
            # 中译：WAR（Write-After-Read）屏障：本轮调度要写共享 GPU buffer，
            #       需等待上一轮前向计算的读完成。
            if self._war_barrier_enabled:
                self.schedule_stream.wait_stream(self.forward_stream)

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch
            # overlap + spec + grammar is unsupported (would desync DP ranks).
            # 中译：overlap + spec（推测解码） + grammar 不受支持（会导致 DP rank 间失步）。
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

            if disable_overlap_for_batch and self.last_batch:
                pop_and_process()

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                self.on_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            # 中译：运行当前批次的采样。它依赖上一批次的结果（如 grammar），
            #       所以放在上一批次处理完毕之后执行。
            self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch

    def _run_batch_prebuilt(
        self: Scheduler, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        if batch.inner_idle_batch is not None:
            idle_batch = batch.inner_idle_batch
            # Reset the inner idle batch to avoid reusing it.
            # 中译：重置内部 idle 批次引用，避免后续误用。
            batch.inner_idle_batch = None
            return self.run_batch(idle_batch)

        return GenerationBatchResult()

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def get_next_disagg_decode_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        """Process prebuilt batch and schedule the next decode batch.

        中译：处理预构建（prebuilt）批次并调度下一个解码批次。
        """
        # Process pending prebuilt batch: output processing + filter + merge
        # 中译：处理待处理的预构建批次：输出处理 + 过滤 + 合并。
        new_prebuilt_batch = self.get_new_prebuilt_batch()
        if new_prebuilt_batch:
            assert self.chunked_req is None
            self.batch_result_processor.process_batch_result_prebuilt(
                new_prebuilt_batch
            )
            new_prebuilt_batch.filter_batch()
            if not new_prebuilt_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = new_prebuilt_batch
                    if self.enable_hisparse:
                        self.running_batch.hisparse_coordinator = (
                            self.hisparse_coordinator
                        )
                else:
                    self.running_batch.merge_batch(new_prebuilt_batch)

        # Schedule decode batch
        # 中译：调度解码批次。
        if self.running_batch.is_empty():
            ret = None
        else:
            self.running_batch = self.update_running_batch(self.running_batch)
            ret = self.running_batch if not self.running_batch.is_empty() else None

        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret)
        if ret:
            set_schedule_time_batch(ret)
        return ret

    def get_new_prebuilt_batch(self: Scheduler) -> Optional[ScheduleBatch]:
        """Create a schedulebatch for fake completed prefill

        中译：为「伪完成 prefill」创建 ScheduleBatch——即从等待队列中取出请求，
              构造一个跳过 prefill 前向计算、仅填充元数据的 extend 批次。
        """
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if len(self.waiting_queue) == 0:
            return None

        if self.enable_priority_scheduling:
            self.policy.calc_priority(self.waiting_queue, self.running_batch)

        curr_batch_size = self.running_batch.batch_size()

        batch_size = min(self.req_to_token_pool.size, self.max_running_requests)

        num_not_used_batch = batch_size - curr_batch_size

        # pop req from waiting queue
        # 中译：从等待队列中取出可运行的请求。
        can_run_list: List[Req] = []
        waiting_queue: List[Req] = []

        for i in range(len(self.waiting_queue)):
            req = self.waiting_queue[i]
            # we can only add at least `num_not_used_batch` new batch to the running queue
            if i < num_not_used_batch:
                can_run_list.append(req)
                # Decode-radix path: new requests already matched in
                # `pop_preallocated`. Retracted requests reset `last_node`,
                # so re-match only when that state is missing.
                # 中译：Decode-radix 路径：新请求已在 `pop_preallocated` 中匹配过前缀。
                #       被回退的请求重置了 `last_node`，因此仅在该状态缺失时才重新匹配。
                if self.server_args.disaggregation_decode_enable_radix_cache:
                    tree_cache = self.tree_cache if req.last_node is None else None
                else:
                    tree_cache = self.tree_cache
                req.init_next_round_input(tree_cache)
                # Truncate fill_len to kv_committed_len so cache_unfinished_req
                # only sees committed KV (full array includes one uncommitted
                # token because init_next_round_input rebuilt it as full).
                # 中译：将 fill_len 截断为 kv_committed_len，使 cache_unfinished_req
                #       只看到已提交的 KV（完整数组因 init_next_round_input 重建包含一个未提交 token）。
                if req.kv_committed_len is not None:
                    req.fill_len = req.kv_committed_len
                    req.set_extend_input_len(req.fill_len - len(req.prefix_indices))
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        set_time_batch(can_run_list, "set_forward_entry_time")

        # construct a schedule batch with those requests and mark as decode
        # 中译：用这些请求构造一个 ScheduleBatch 并标记为解码模式。
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )

        # construct fake completed prefill
        # 中译：构造「伪完成 prefill」——跳过真实前向计算，仅填充元数据。
        new_batch.prepare_for_prebuilt()
        new_batch.process_prebuilt(self.server_args, self.future_map)

        return new_batch

    def process_decode_queue(self: Scheduler):
        """推进 decode 端 PD 分离请求在各生命周期队列间的流转。

        中译：这是 decode 端每轮事件循环都会调用的核心驱动函数，负责把请求沿着
              「预分配队列 → 传输队列 → 等待队列」逐级推进。它并不做真实的解码前向，
              只负责队列间的状态搬运与准入控制。整体分为四步：

              1. 处理 HiCache 异步事件与 KV 卸载进度（若启用）；
              2. 优先恢复被回退（retracted）的请求——只有回退队列清空后才接纳新请求，
                 避免新请求与待恢复请求争抢显存导致的饥饿/死锁；
              3. 按轮询间隔（polling_interval）节流，避免每轮都做昂贵的跨 rank 轮询；
              4. 在轮询周期到达时，把已预分配好的请求送入传输队列，并把「KV 已到达」
                 的请求移入等待队列，交给后续 get_next_disagg_decode_batch_to_run 构批解码。
        """
        # 中译：若启用 decode 端 HiCache，先检查并处理分层缓存的异步事件
        #       （如 L2/L3 → L1 的回载完成通知），推进本地恢复状态机。
        if self.enable_decode_hicache:
            self.tree_cache.check_hicache_events()

        # 中译：若启用 KV Cache 卸载（offload），检查卸载操作的进度，回收已完成卸载的资源。
        if self.server_args.disaggregation_decode_enable_offload_kvcache:
            self.decode_offload_manager.check_offload_progress()

        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        # 中译：尝试恢复被回退的请求——仅当显存足以再支撑 `num_reserved_decode_tokens` 步解码时才恢复。
        #       被回退请求是此前因显存不足被换出（KV 存到 CPU）的请求，此处在显存回升后
        #       重新为它们预分配 KV 并回载，恢复成功的请求直接进入等待队列。
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            # 中译：若回退队列仍未清空，则本轮不接纳新请求——优先保证已被回退的请求恢复，
            #       避免新请求继续抢占显存，导致回退请求长期无法恢复。
            return

        # 中译：惰性初始化轮询计数器与轮询间隔。polling_interval 控制多少轮事件循环
        #       才真正做一次预分配/传输轮询（跨 rank 的 poll 开销较大，需节流）。
        if not hasattr(self, "polling_count"):
            self.polling_count = 0
            self.polling_interval = (
                self.server_args.disaggregation_decode_polling_interval
            )

        # 中译：计数器在 [0, polling_interval) 间循环递增。
        self.polling_count = (self.polling_count + 1) % self.polling_interval

        # 中译：仅在计数归零（即每隔 polling_interval 轮）时执行一次实际的队列推进。
        if self.polling_count % self.polling_interval == 0:
            # 中译：从预分配队列弹出已完成 KV 预分配的请求（req_conns），送入传输队列，
            #       此时会向 prefill 端告知 KV 落点、开始等待 KV 到达。
            req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
            self.disagg_decode_transfer_queue.extend(req_conns)
            transferred_reqs = (
                self.disagg_decode_transfer_queue.pop_transferred()
            )  # the requests which kv has arrived  # 中译：KV 已传输到达的请求
            # 中译：HiSparse 直达主机（direct-to-host）路径——KV 数据已在主机池中，
            #       无需再经暂存（staging），直接接纳这些请求。
            if self.enable_hisparse:
                for req in transferred_reqs:
                    # Direct-to-host: KV data already in host pool, skip staging
                    # 中译：直达主机：KV 数据已在主机池中，跳过暂存步骤。
                    self.hisparse_coordinator.admit_request_direct(req)
            # 中译：把 KV 已到达的请求加入等待队列，后续由构批逻辑取出、预构建 extend 批并解码。
            self.waiting_queue.extend(transferred_reqs)
