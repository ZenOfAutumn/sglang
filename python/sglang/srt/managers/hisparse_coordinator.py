# to be combined with the sparse coordinator class and sparse algorithm family
# 中译：本文件实现 HiSparse（层次化稀疏 KV 缓存）的协调器（coordinator）。
#       未来计划与「稀疏协调器基类」及一整套稀疏算法家族合并统一。
#
# 模块职责概述：
#   HiSparse 的核心思想是把巨大的 KV 缓存分层存放——把全量（压缩后）的 KV 放在
#   容量更大的「主机内存池（host pool）」，而 GPU 上只保留一个小的「设备缓冲区
#   （device buffer / hot buffer）」。解码（decode）时通过 top-k 选择真正需要参与
#   注意力计算的少量 token，把它们从主机换入（swap-in）到设备缓冲区，从而在有限的
#   显存下支持超长上下文。本协调器负责：
#     - 准入（admit）：把新请求的 prefill KV 从设备搬到主机（staging），或直连主机（direct）。
#     - 备份（backup）：解码过程中把新产生的压缩 token 异步备份到主机。
#     - 换入（swap-in）：每层注意力前，把 top-k 选中的 token 加载进设备缓冲区。
#     - 资源回收（finish / retract / abort）：请求结束或被抢占时释放设备与主机资源。

import logging
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch

from sglang.kernels.ops.kvcache.hisparse import (
    copy_cache_planned_mla,
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
)
from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import get_device_module, is_hip

device_module = get_device_module()

_is_hip = is_hip()

logger = logging.getLogger(__name__)


class HiSparseAct(NamedTuple):
    """staging（暂存）阶段一次异步搬运的记录单元。

    中译：当一个请求被准入到 staging 时，会启动一次「设备 → 主机」的异步 DMA 拷贝，
          用一对 CUDA event 跟踪它的开始与完成，并记录关联的请求。后续靠轮询
          finish_event 判断该请求的 KV 是否已全部备份到主机、可以进入解码。
    字段：
        start_event：拷贝开始事件（用于让 staging 流等待调度流就绪）。
        finish_event：拷贝完成事件（query() 为 True 表示已备份完毕）。
        req：关联的请求对象。
    """

    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseTokenStats(NamedTuple):
    """HiSparse 设备/主机两级 KV 池的 token 用量统计快照。

    中译：用于观测/上报当前 GPU 设备池与主机池各自的 token 占用量与占用率。
    字段：
        device_tokens：设备池已用 token 数。
        device_token_usage：设备池占用率（0~1）。
        host_tokens：主机池已用 token 数。
        host_token_usage：主机池占用率（0~1）。
    """

    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


def resolve_shared_index_layers(
    *,
    hf_text_config,
    pp_size: int,
    is_speculative: bool,
) -> Optional[List[bool]]:
    """Per-layer "reuses the previous layer's top-k index" pattern, or None.

    Mirrors DeepseekV2AttentionMLA's skip_topk derivation (index_topk_pattern /
    index_topk_freq / cli_factor); None when the model has no sharing or the
    prefetch cannot run (PP, speculative decoding, kill-switch).
    """
    if not is_deepseek_dsa(hf_text_config):
        return None
    num_layers = hf_text_config.num_hidden_layers
    cli_factor = getattr(hf_text_config, "cli_factor", 1) or 1
    if cli_factor > 1:
        pattern = [i % cli_factor != 0 for i in range(num_layers)]
    else:
        pattern = [dsa_layer_skips_topk(hf_text_config, i) for i in range(num_layers)]
    if not any(pattern):
        return None
    if pp_size != 1 or is_speculative:
        logger.warning(
            "HiSparse shared-index prefetch is unsupported under pipeline "
            "parallelism / speculative decoding; falling back to synchronous "
            "swap-in."
        )
        return None
    if envs.SGLANG_DISABLE_HISPARSE_PREFETCH.get():
        logger.info(
            "HiSparse shared-index prefetch disabled via "
            "SGLANG_DISABLE_HISPARSE_PREFETCH; using synchronous swap-in."
        )
        return None
    return pattern


def _build_prefetch_groups(
    is_shared_index_layer: List[bool],
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Group consecutive shared-index (skip) layers under their anchor layer.

    Returns (groups, slot): anchor layer_id -> ordered skip layers, and each
    skip layer's position in its group (indexes the per-slot prefetch events).
    """
    groups: Dict[int, List[int]] = {}
    slot = [0] * len(is_shared_index_layer)
    anchor = None
    for i, is_shared in enumerate(is_shared_index_layer):
        if not is_shared:
            anchor = i  # compute layer; anchors the skip layers after it
            continue
        assert anchor is not None, (
            f"shared-index (skip) layer {i} has no preceding compute layer; "
            "the model's index-topk pattern is invalid"
        )
        group = groups.setdefault(anchor, [])
        slot[i] = len(group)
        group.append(i)
    return groups, slot


class HiSparseCoordinator:
    """HiSparse 层次化稀疏 KV 缓存的协调器。

    中译：负责协调「GPU 设备缓冲区」与「主机内存池」之间的 KV 数据流转，是 HiSparse
          方案的中枢。它既要管理每个请求在设备缓冲区中的占位与映射，又要在解码过程中
          完成 token 的备份（设备→主机）与换入（主机→设备），并在请求结束/抢占时回收资源。

    在系统中的角色：
        被调度器（Scheduler）/模型执行层调用，配合注意力后端在每层 forward 前换入 top-k
        选中的 token。支持两条数据通路：普通 MLA（HiSparseTokenToKVPoolAllocator）和
        DeepSeek-V4 专用通路（DeepSeekV4HiSparseTokenToKVPoolAllocator，带 compress_ratio 压缩）。

    关键协作对象：
        - req_to_token_pool：请求 → token 槽位映射池（ReqToTokenPool）。
        - token_to_kv_pool_allocator：设备侧 KV 池分配器（决定走普通还是 dsv4 通路）。
        - mem_pool_device / mem_pool_host：设备侧、主机侧 KV 内存池。
        - JIT 换入核函数：load_cache_to_device_buffer_(dsv4_)mla。
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
        swap_in_block_size: int = 960,
        shared_index_layers: Optional[List[bool]] = None,
    ):
        # 中译：构造函数。保存依赖对象，并根据分配器类型选择普通 MLA 或 dsv4 两条通路，
        #       预分配后续 staging/备份/换入所需的全部张量缓冲与流（stream）/事件（event）。
        #       关键参数：
        #         top_k——每步注意力换入的 token 数；
        #         device_buffer_size——每个请求在设备上的热缓冲区容量（token 数）；
        #         host_to_device_ratio——主机池相对设备池的容量倍数（普通通路用）；
        #         tp_group——张量并行通信组，用于跨 worker 同步 staging 完成进度。
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        # 中译：compress_ratio——每多少个原始 token 压缩为一个 KV 槽（dsv4 通路 > 1）。
        self.swap_in_block_size = swap_in_block_size
        # Timing probe: skip the host->device KV bytes to measure the "IO is
        # free" floor. Produces garbage output; benchmarking only.
        self.skip_io = envs.SGLANG_DEBUG_HISPARSE_SKIP_IO.get()
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        # 中译：根据分配器类型判定是否为 DeepSeek-V4 专用 HiSparse 通路（带分页主机池+压缩）。
        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        if self.is_dsv4_hisparse:
            # 中译：dsv4 通路——设备池来自分配器，主机池用分页（paged）实现，按页对齐管理。
            #       num_host_pages 按「全量压缩后长度 / 页大小」向上取整估算主机所需页数。
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            page_size = self.mem_pool_device.page_size
            num_host_pages = (
                self.token_to_kv_pool_allocator.size_full // self.compress_ratio
                + page_size
                - 1
            ) // page_size
            self.mem_pool_host = DeepSeekV4PagedHostPool(
                pool_name="dsv4_hisparse_c4",
                device_buffers=self.mem_pool_device.kv_buffer,
                item_bytes=self.mem_pool_device.bytes_per_page_padded,
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout="layer_first",
            )
            self.item_size_bytes = (
                self.mem_pool_device.kv_cache_total_dim
                * self.mem_pool_device.store_dtype.itemsize
            )
        else:
            # 中译：普通 MLA 通路——设备池由分配器返回，主机池用 MLATokenToKVPoolHost，
            #       容量由 host_to_device_ratio 决定（host_size=0 表示按倍数自动推算）。
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            self.mem_pool_device: HiSparseDSATokenToKVPool = (
                self.token_to_kv_pool_allocator.get_kvcache()
            )
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=self.mem_pool_device.page_size,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size
        self.page_size = self.mem_pool_device.page_size

        # 中译：以「最大并发请求槽数 × 最大上下文长度」为上界，预分配所有逐请求的映射表。
        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        # 中译：压缩后的最大上下文长度（向上取整），主机池映射表按此尺寸开。
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # to have an extra page for new tokens
        # 中译：padded_buffer_size 在 device_buffer_size 之上额外多预留一页，
        #       用来安放「当前新生成的 token」（保留槽位，见 device_buffer_size 处的 reserved slot）。
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        # 中译：req_to_device_buffer[req, i] = 该请求第 i 个（压缩）token 在设备 KV 池中的物理槽位。
        self.req_to_device_buffer = torch.zeros(
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        # 中译：每个请求当前已分配的设备缓冲区长度（CPU 上维护，便于按需 grow）。
        self.req_device_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        # 中译：req_to_host_pool[req, i] = 该请求第 i 个（压缩）token 在主机池中的物理槽位（-1 表示未分配）。
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        # 中译：每个请求已在主机池分配的 token 数（CPU 维护，供分页分配器续分配用）。
        self.req_to_host_pool_allocated_len = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )

        # 中译：两条独立 CUDA 流——staging 写入流（准入时设备→主机）与解码备份流，
        #       与主调度流并行，避免阻塞解码热路径。
        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        # 中译：尚未确认完成的 staging 任务队列（FIFO，按准入顺序）。
        self.ack_staging_queue: List[HiSparseAct] = []
        # 中译：解码生产者流（外部注入，用于让备份流等待解码计算完成后再读 KV）。
        self.decode_producer_stream = None
        # 中译：解码备份完成事件 + 是否有未决备份的标志，用于在换入前确保备份已落盘到主机。
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        # 中译：张量并行（TP）组与其 world size，用于跨 TP worker 同步 staging 完成进度（取最小值）。
        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        # 中译：初始化换入核函数所需的逐层数据结构（每层各一份映射）。
        layer_num = self.mem_pool_device.layer_num
        # 中译：req_device_buffer_tokens[layer, req, slot] = 该设备缓冲槽当前缓存的是哪个 token 位置
        #       （-1 表示空槽）。换入核函数据此判断 top-k 命中（hit）还是缺失（miss→需从主机加载）。
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        # 中译：req_device_buffer_token_locs[layer, req, slot] = 该槽对应的设备 KV 物理槽位。
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        # 中译：LRU 槽位初始顺序 [0,1,...,buf-1]，作为每个(层,请求)LRU 状态的初值模板。
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        # 中译：lru_slots——换入核函数用于在缓冲区满时做 LRU 淘汰的逐(层,请求)状态。
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        # 中译：常量 arange [0..buf-1]，用于在分配设备缓冲时一次性填充 token 位置标记。
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        # 中译：换入结果的预分配输出缓冲（固定地址，对 CUDA Graph 安全——可被反复 replay）。
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        # 中译：标量张量——本批次中「真实（非填充）」请求数。CUDA Graph 下批大小固定，
        #       超出真实数的填充块会据此提前返回（early-return），每次 replay 前更新它。
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        # 中译：逐请求 CPU 标志——True 表示「下一步解码跳过备份」，因为 staging 阶段
        #       已把所有 prefill token 备份过了；用过一次后即清除。
        self._skip_first_backup = [False] * max_num_req_slots

        self._init_shared_index_prefetch(
            shared_index_layers=shared_index_layers,
            layer_num=layer_num,
            max_num_req_slots=max_num_req_slots,
        )

    def _init_shared_index_prefetch(
        self,
        shared_index_layers: Optional[List[bool]],
        layer_num: int,
        max_num_req_slots: int,
    ) -> None:
        """Set up the plan-then-IO prefetch for shared-index (IndexShare) models:
        the anchor's kernel records its miss plan and skip layers replay it on
        `prefetch_stream`, overlapping their IO with the intervening compute."""
        if shared_index_layers is not None and len(shared_index_layers) != layer_num:
            # Attention-layer count differs from num_hidden_layers (e.g. Longcat
            # doubles it): pattern would be misindexed, fall back to synchronous.
            logger.warning(
                "HiSparse shared-index prefetch disabled: pattern length %d != "
                "KV pool layer_num %d; using synchronous swap-in.",
                len(shared_index_layers),
                layer_num,
            )
            shared_index_layers = None
        self._is_shared_index_layer = list(shared_index_layers or [False] * layer_num)
        self.enable_prefetch = any(self._is_shared_index_layer)
        self._prefetch_groups, self._prefetch_slot = _build_prefetch_groups(
            self._is_shared_index_layer
        )
        if not self.enable_prefetch:
            return

        # Small fixed grid for the copy-only kernel: low SM footprint so the
        # copies overlap compute with little contention.
        self._prefetch_copy_blocks = 4
        max_group_size = max(len(g) for g in self._prefetch_groups.values())
        self.prefetch_stream = device_module.Stream()
        self._prefetch_events = [device_module.Event() for _ in range(max_group_size)]
        # Plan recorded by the current anchor, replayed by its skip layers. One
        # buffer set suffices: the last skip layer's event wait orders the next
        # anchor's writes after this group's copies.
        self._miss_src = torch.zeros(
            (max_num_req_slots, self.top_k), dtype=torch.int64, device=self.device
        )
        self._miss_dst = torch.zeros(
            (max_num_req_slots, self.top_k), dtype=torch.int32, device=self.device
        )
        self._miss_count = torch.zeros(
            (max_num_req_slots,), dtype=torch.int32, device=self.device
        )
        logger.info(
            "HiSparse: shared-index prefetch (plan-then-IO) enabled; %d anchor "
            "group(s), %d skip layer(s) of %d total.",
            len(self._prefetch_groups),
            sum(self._is_shared_index_layer),
            layer_num,
        )

    def set_decode_producer_stream(self, stream) -> None:
        # 中译：注入「解码生产者流」。备份/回收时会让相关流等待它，确保解码计算产出的
        #       KV 已写完再被读取或释放，避免读到未完成的数据。
        self.decode_producer_stream = stream

    def destroy(self) -> None:
        # Drain in-flight transfers so the buffer is idle, then unregister it.
        # See HostKVCache.destroy for why the explicit unregister matters.
        self.write_staging_stream.synchronize()
        self.decode_backup_stream.synchronize()
        if self.enable_prefetch:
            # Skip-layer copies read the pinned host pool on the prefetch stream.
            self.prefetch_stream.synchronize()
        self.mem_pool_host.destroy()

    def get_token_stats(self) -> HiSparseTokenStats:
        # 中译：采集设备池与主机池的 token 占用量/占用率，返回统计快照（供监控上报）。
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def admit_request_into_staging(self, req: Req) -> None:
        # 中译：把一个新请求准入到「staging（暂存）」通路。
        #       做法：在 staging 流上异步把该请求 prefill 阶段的全部 KV 从设备搬运（备份）到主机池，
        #       并把这次搬运记入 ack_staging_queue。搬运完成（finish_event）后才会真正进入解码。
        #       副作用：标记 req.hisparse_staging=True；在主机池分配槽位；入队一条 HiSparseAct。
        req.hisparse_staging = True

        # 中译：取该请求 prefill 部分的全量 KV 槽位，并翻译为 hisparse 设备索引空间。
        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : req.extend_range.end
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        # 中译：在主机池为这 prefill_len 个 token 分配槽位（写入 req_to_host_pool 映射）。
        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.kv.req_pool_idx,
            0,
            prefill_len,
        )

        # 中译：在 staging 流上发起跨层（all_layer）的设备→主机异步备份；
        #       start_event 让 staging 流等待调度流就绪，finish_event 标记备份完成。
        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        # 中译：把本次 staging 任务（含两事件与请求）入队，待 collect_ready_reqs 轮询完成。
        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.

        中译：「直连主机」准入通路——KV 数据已经通过 RDMA 直接落在主机池里。
              因此完全跳过 staging 的 DMA 搬运，只为解码期的换入分配一小块设备缓冲，
              然后把请求标记为就绪。主机侧索引此前已写入 req_to_host_pool。
        关于 alloc_device_buffer() 后的元数据修正：
              alloc_device_buffer() 会把 device_buffer_tokens 置成 [0,1,...,buf-1]，
              意为「这些 token 已缓存在设备缓冲里」。在 staging 通路这是对的（prefill 填满了缓冲），
              但在直连通路缓冲其实是空的，故需按下面两种情况修正。
        """
        self.alloc_device_buffer(req)

        host_len = self.host_token_len(req.kv.kv_allocated_len)
        if host_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            # 中译：短序列（长度 <= 设备缓冲容量）——换入核函数走快路径，直接返回设备槽位、
            #       不做任何主机加载；因此这里必须先把主机池里的全部 token 预加载进设备缓冲。
            self._preload_to_device_buffer(req)
        else:
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            # 中译：长序列——把 device_buffer_tokens 全置 -1，让核函数看到所有槽为空，
            #       从而每个 top-k 查找都视为 miss → 触发从主机加载。
            self.req_device_buffer_tokens[
                :, req.kv.req_pool_idx, : self.device_buffer_size
            ] = -1

        req.hisparse_staging = False
        # 中译：直连通路下 KV 已在主机，故首步解码跳过备份。
        self._skip_first_backup[req.kv.req_pool_idx] = True
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def host_token_len(self, kv_allocated_len: int) -> int:
        # 中译：把「已分配的全量 token 长度」换算为「主机池中的 token 数」。
        #       dsv4 通路因压缩需除以 compress_ratio；普通通路一一对应。
        if self.is_dsv4_hisparse:
            return kv_allocated_len // self.compress_ratio
        return kv_allocated_len

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
        n = self.host_token_len(req.kv.kv_allocated_len)
        host_indices = self.req_to_host_pool[req.kv.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.kv.req_pool_idx, :n]

        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:
        # 中译：为请求分配设备缓冲区，并建立「逻辑（压缩）位置 → 设备物理槽位」的映射。
        #       dsv4 通路直接按 padded_buffer_size 整块分配；普通通路按当前 token 数页对齐
        #       分配（最多到 device_buffer_size，填满时再含预留页）。
        #       副作用：写入 req_to_device_buffer / req_device_buffer_size /
        #       req_device_buffer_tokens / req_device_buffer_token_locs。分配失败抛 RuntimeError。
        if self.is_dsv4_hisparse:
            allocated_len = req.extend_range.end
            alloc_size = self.padded_buffer_size
        else:
            allocated_len = req.kv.kv_allocated_len
            page_size = self.mem_pool_device.page_size
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            # 中译：仅按当前 token 数做页对齐分配；若已填满 device_buffer_size，则改用含预留页的尺寸。
            alloc_size = min(
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            if alloc_size == self.device_buffer_size:
                alloc_size = self.padded_buffer_size

        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)

        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size
        )
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.kv.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.kv.req_pool_idx] = alloc_size

        # 中译：把缓冲槽标记为「缓存了 token 位置 0..buf-1」，并写入各层的物理槽位映射，
        #       供换入核函数判断命中/缺失。（直连通路会在 admit_request_direct 中按需修正此标记。）
        self.req_device_buffer_tokens[
            :, req.kv.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32
        self.req_device_buffer_token_locs[:, req.kv.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Grow device buffers for requests whose sequence length exceeds current capacity.

        中译：为序列长度超出当前已分配缓冲容量的请求「扩容」设备缓冲区。
              仅处理仍属「短序列」（长度 <= device_buffer_size）且已超出当前容量的请求；
              先在 CPU 上算出各自需要扩多少，再合并成一次性批量分配（减少分配开销），
              然后把新槽位填进映射表并更新容量。
        返回：每个请求「最新 token」对应的预留缓冲槽位（reserved slot）张量，
              供调用方写入 device_buffer_size 处的保留槽。
        """
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        # 中译：只对短序列扩容（长序列走主机加载路径，无需扩设备缓冲）。
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            # 中译：先在 CPU 上算清每个请求新旧容量与扩增量，累加成 total_grow 后一次性批量分配。
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
            total_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])

                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size

            if total_grow > 0:
                # 中译：一次性向设备分配器申请 total_grow 个槽位，再按各请求切片（chunk）分发。
                all_new_indices = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        total_grow
                    )
                )
                if all_new_indices is None:
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )

                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        # 中译：返回每个请求「最新 token（seq_len-1）」所在的缓冲槽位（超出则钳到预留槽 device_buffer_size）。
        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def has_ongoing_staging(self) -> bool:
        # 中译：是否还有未完成的 staging 任务（队列非空）。
        return len(self.ack_staging_queue) > 0

    def collect_ready_reqs(self) -> List[Req]:
        # 中译：轮询 staging 队列头部，收集「设备→主机备份已完成」的请求，转入解码就绪态。
        #       因队列 FIFO，遇到第一个未完成的就停止（只取连续完成的前缀）。
        #       TP 多 worker 时用 all_reduce(MIN) 取各 worker 共同完成的最小数量，保证调度一致。
        #       副作用：为就绪请求分配设备缓冲、标记跳过首步备份、清除 staging 标志。
        ready_reqs: List[Req] = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            # 中译：跨 TP worker 同步，确保所有 worker 对调度器做出相同数量的状态更新。
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            # 中译：为就绪请求分配设备缓冲，并把它标记为非 staging、首步跳过备份，加入返回列表。
            self.alloc_device_buffer(req)
            self._skip_first_backup[req.kv.req_pool_idx] = True
            req.hisparse_staging = False
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        # 中译：把「当前最新 token」的设备物理槽位登记进设备缓冲映射，使注意力计算能写入正确位置。
        #       流程：先备份上一个压缩 token 到主机，再按通路（普通/dsv4）解析最新 token 的预留槽，
        #       并更新 req_device_buffer_token_locs 与设备池的 full→hisparse 索引映射。
        #       dsv4 通路仅在 seq_len 恰好是 compress_ratio 倍数（产生新压缩 token）时才更新。
        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not self.is_dsv4_hisparse:
            # Grow device buffers if needed and resolve the latest-token slot.
            # 中译：普通通路——按需扩容设备缓冲，并解析出最新 token 的槽位。
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                out_cache_loc
            )
            # ROCm: the decode remap creates a temporary hisparse device slot per
            # new token (via the page_size==1 allocator path). Free the stale
            # slot before pointing the mapping at the reserved device-buffer slot,
            # otherwise the temporary slots leak and corrupt later swap-in lookups.
            # CUDA keeps the original behavior: the swap-in kernel consumes only
            # top_k_device_locs, so stale mapping entries are harmless there.
            if _is_hip:
                previous_locs = self.mem_pool_device._translate_loc_to_hisparse_device(
                    compressed_locs
                )
                stale_locs = previous_locs[
                    (previous_locs > 0) & (previous_locs != reserved_buffer_loc)
                ]
                if stale_locs.numel() > 0:
                    self.token_to_kv_pool_allocator.free_hisparse_indices(stale_locs)

            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc
            return

        # 中译：dsv4 通路——只有 seq_len 是 compress_ratio 整数倍的请求才刚产生新的压缩 token，
        #       仅对这些「活跃」请求更新映射；否则本步无新压缩 token，直接返回。
        active_reqs = seq_lens % self.compress_ratio == 0
        if not torch.any(active_reqs):
            return

        active_seq_lens = seq_lens[active_reqs]
        active_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        compressed_seq_lens = active_seq_lens // self.compress_ratio
        reserved_positions = (compressed_seq_lens - 1).clamp(
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, reserved_positions
        ]

        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
        )

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous compressed token to host memory.

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.

        Two cases are skipped:
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.

        中译：把「上一个压缩 token」备份到主机内存。
              每产生一个新的压缩 token（每 compress_ratio 个解码步产生一个）都必须备份到主机，
              这样换入核函数日后才能从主机恢复它。
        两种情况会跳过备份：
              - staging 后的第一个解码步：所有 prefill token 在 staging 时已备份过，没有新数据。
              - (seq_len - 1) % compress_ratio != 0 的步：本步未产生新的压缩 token。
        在解码备份流（decode_backup_stream）上异步执行，并通过 _backup_done_event 标记完成。
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        # 中译：构造本批次中需要备份的位置列表：跳过 staging 后首步、跳过未对齐（无新压缩 token）的步。
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:
                backup_indices.append(i)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        backup_req_indices = req_pool_indices[backup_indices_gpu]

        # The previous compressed token's position and its device buffer slot:
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - long:  slot = device_buffer_size      (the reserved slot)
        # 中译：上一个压缩 token 的逻辑位置及其设备缓冲槽：
        #       compressed_pos = (seq_len-1)//compress_ratio - 1；
        #       短序列时槽 = compressed_pos（普通缓冲区内），长序列时槽 = device_buffer_size（预留槽）。
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        # 中译：为每个待备份请求在主机池分配 1 个槽位（位于 start_pos 处），再拼接成一个张量。
        host_locs_list = []
        for i in backup_indices:
            req_idx = int(req_pool_indices_cpu[i])
            start_pos = (int(seq_lens_cpu[i]) - 1) // self.compress_ratio - 1
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                1,
            )
            host_locs_list.append(host_locs)
        host_locs = torch.cat(host_locs_list)

        # 中译：先等上一轮备份完成，再在备份流上发起本轮跨层备份；备份流需等待调度流与
        #       解码生产者流，以保证读取到的是已算完的 KV。完成后记录 _backup_done_event。
        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_compressed_pos.is_cuda:
                actual_compressed_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def wait_for_pending_backup(self) -> None:
        # 中译：若存在未决的备份，让当前流等待 _backup_done_event，确保主机数据已写完后再继续
        #       （如换入/资源回收前的同步），随后清除未决标志。
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)

        中译：把 top-k 选中的 token 加载进设备内存，并返回它们的设备索引。
              这是一个朴素的「逐请求循环」实现，仅用于调试/验证；生产代码改用
              swap_in_selected_pages（JIT CUDA 核函数）。
              注意：不支持 dsv4 hisparse（其主机池无 load_to_device_per_layer 且索引在压缩空间）。
              目前仅在 test_hisparse_unit.py 中作为核函数的「参考实现（oracle）」使用（非 dsv4 路径）。
        参数：
              req_pool_indices——各请求的池索引，形状 (num_reqs,)；
              seq_lens——各请求序列长度，形状 (num_reqs,)；
              top_k_tokens——各请求选中的 token 位置，形状 (num_reqs, top_k)；
              layer_id——要加载 KV 的层号。
        返回：选中 token 的设备 KV 索引，形状 (num_reqs, top_k)。
        """
        assert (
            not self.is_dsv4_hisparse
        ), "naive_load_topk is not implemented for dsv4 hisparse"
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            assert torch.all(
                selected_tokens >= 0
            ), f"Req {req_idx}: selected tokens contain negative positions"
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                # 中译：短序列——所有 token 都在设备缓冲里，直接按位置取槽位即可。
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                # 中译：长序列——最新 token 在预留槽（device_buffer_size 处），其余需从主机加载。
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    # 中译：主机槽位为负说明该 token 未被备份过，属于异常，直接报错并指出位置。
                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def abort_staging_request(self, req: Req) -> None:
        """Remove a request from the staging queue and free its host + device resources.

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).

        中译：把一个请求从 staging 队列移除，并释放其占用的主机 + 设备资源。
              当中止一个「已准入 staging 但尚未完成」（req.hisparse_staging 为 True）的请求时必须调用。
              副作用：清队列项、释放设备 KV 与主机槽位、清零相关映射、复位标志。
        """
        # Remove from staging queue
        # 中译：从 staging 队列里剔除该请求对应的项。
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        # Wait for any in-flight staging DMA to complete before freeing
        # 中译：释放前先同步 staging 流，等待仍在途的 DMA 拷贝完成，避免释放后被写。
        self.write_staging_stream.synchronize()

        prefill_len = req.extend_range.end
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)

        # Free host memory that was allocated during admit_request_into_staging
        # 中译：释放 admit_request_into_staging 时在主机池分配的内存，并把映射复位为 -1/0。
        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.kv.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.kv.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[req.kv.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.kv.req_pool_idx] = 0
        self._skip_first_backup[req.kv.req_pool_idx] = False
        req.hisparse_staging = False

    def retract_req(self, req: Req) -> None:
        # 中译：抢占/撤回一个请求。若它仍在 staging 中走 abort_staging_request，
        #       否则按正常结束流程 request_finished 释放资源。
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        # 中译：请求正常结束时释放其全部 HiSparse 资源（设备缓冲 + 主机池 + 各映射复位）。
        # release resources only after the execution of a potential overlapped batch
        # 中译：若启用了重叠（overlap）调度，需先等解码生产者流与未决备份完成，避免释放仍在用的内存。
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        # 中译：用 kv_allocated_len 而非 seqlen——投机解码下分配器可能超额分配，
        # 中译：只释放真正分配过的设备缓冲槽位（去重且 >0），归还给分配器。
        allocated_len = req.kv.kv_allocated_len

        # release memory -- only free actually-allocated buffer indices
        current_cap = int(self.req_device_buffer_size[req.kv.req_pool_idx])
        if current_cap > 0:
            side_buf_hi = self.req_to_device_buffer[req.kv.req_pool_idx, :current_cap]
            all_hi = torch.unique(side_buf_hi[side_buf_hi > 0])
            if all_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :allocated_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            allocated_locs
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = 0

        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.kv.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.kv.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        # 中译：把该请求槽位的所有逐请求状态复位（token 标记/槽位映射/容量/LRU/备份标志）。
        self.req_device_buffer_tokens[:, req.kv.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.kv.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.kv.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.kv.req_pool_idx] = 0
        self.req_to_host_pool[req.kv.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.kv.req_pool_idx] = 0
        self.lru_slots[:, req.kv.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.kv.req_pool_idx] = False

    def _run_swap_in_kernel(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        record_plan: bool = False,
    ) -> torch.Tensor:
        # 中译：从预分配缓冲切出本批所需部分并清为 -1（CUDA Graph 复用同一地址）。
        """Run the full plan+IO swap-in kernel for one layer; return its slot table.

        record_plan (set on the anchor of a shared-index group) also records the
        miss plan into self._miss_{src,dst,count} for the skip layers to replay.
        """
        num_reqs = req_pool_indices.size(0)
        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]

        # 中译：核函数分块大小（可调以优化性能）；按通路选择 dsv4 或普通 MLA 的换入核函数。
        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        plan = (
            dict(
                miss_src=self._miss_src[:num_reqs],
                miss_dst=self._miss_dst[:num_reqs],
                miss_count=self._miss_count[:num_reqs],
            )
            if record_plan
            else {}
        )
        swap_in_fn(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            skip_io=self.skip_io,
            **plan,
        )
        return top_k_indices

    def _run_copy_only_kernel(self, num_reqs: int, skip_layer: int) -> None:
        """Replay the anchor's recorded miss plan into a skip layer's buffers
        (IO-only; the anchor's slot table stays valid -- lockstep layout)."""
        copy_cache_planned_mla(
            miss_src=self._miss_src[:num_reqs],
            miss_dst=self._miss_dst[:num_reqs],
            miss_count=self._miss_count[:num_reqs],
            num_real_reqs=self.num_real_reqs,
            host_cache=self.mem_pool_host.kv_buffer[skip_layer],
            device_buffer=self.mem_pool_device.kv_buffer[skip_layer],
            item_size_bytes=self.item_size_bytes,
            num_blocks=self._prefetch_copy_blocks,
            is_dsv4_layout=self.is_dsv4_hisparse,
            skip_io=self.skip_io,
        )

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices.

        With prefetch enabled, anchors swap in synchronously (recording the miss
        plan) and prefetch their skip layers' copies; skip layers just wait.
        """
        if not self.enable_prefetch:
            return self._run_swap_in_kernel(
                req_pool_indices, compressed_seq_lens, top_k_result, layer_id
            )

        num_reqs = req_pool_indices.size(0)
        if self._is_shared_index_layer[layer_id]:
            # Skip layer: wait for its prefetched copy; the anchor's slot table
            # applies (shared index + lockstep buffers).
            slot = self._prefetch_slot[layer_id]
            self._prefetch_events[slot].wait(device_module.current_stream())
            return self.top_k_device_locs_buffer[:num_reqs]

        # Anchor: swap in synchronously (recording the plan), then prefetch the
        # skip layers' copies on the side stream.
        group = self._prefetch_groups.get(layer_id)
        anchor_locs = self._run_swap_in_kernel(
            req_pool_indices,
            compressed_seq_lens,
            top_k_result,
            layer_id,
            record_plan=group is not None,
        )
        if group:
            # Fork: the prefetch stream must observe the anchor's plan (produced
            # on the current stream) before replaying it.
            self.prefetch_stream.wait_stream(device_module.current_stream())
            with device_module.stream(self.prefetch_stream):
                for skip_layer in group:
                    self._run_copy_only_kernel(num_reqs, skip_layer)
                    self._prefetch_events[self._prefetch_slot[skip_layer]].record(
                        self.prefetch_stream
                    )
        return anchor_locs
