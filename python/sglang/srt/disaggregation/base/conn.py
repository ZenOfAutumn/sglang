from __future__ import annotations

import dataclasses
import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import DisaggregationMode


class StateType(str, enum.Enum):
    MAMBA = "mamba"
    SWA = "swa"
    DSA = "dsa"
    MINIMAX_INDEX_K = "minimax_index_k"
    # DeepSeek-V4 unified_kv SWA ring: addressed per-row by ring slot
    # (req_pool_idx * ring_stride + pos % ring_stride), needs its own component.
    SWA_RING = "swa_ring"
    # DeepSeek-V4 online C128 request-scoped state.
    C128_STATE = "c128_state"
    # A block-scaled KV dtype keeps its per-block scales in buffers parallel to
    # K/V, one component per sub-pool so each carries the index payload of the
    # KV it describes (whole sequence for full attention, window for SWA).
    BLOCK_SCALE = "block_scale"
    BLOCK_SCALE_SWA = "block_scale_swa"


@dataclasses.dataclass
class KVTransferMetric:
    """单次 KV 传输的指标；无法拆分某项的后端可将其留为 None。"""

    # 传输耗时（秒）；无法单独统计的后端留 None。
    transfer_latency_s: Optional[float] = None
    # 分配等待耗时（秒）；无法单独统计的后端留 None。
    alloc_latency_s: Optional[float] = None
    transfer_total_bytes: Optional[int] = None  # 本次传输的总字节数


class KVArgs:
    engine_rank: int
    kv_data_ptrs: List[int]
    kv_data_lens: List[int]
    kv_item_lens: List[int]
    kv_layer_ids: List[int]
    kv_cache_dtype_str: str
    aux_data_ptrs: List[int]
    aux_data_lens: List[int]
    aux_item_lens: List[int]
    state_types: List[StateType]
    state_data_ptrs: List[List[int]]
    state_data_lens: List[List[int]]
    state_item_lens: List[List[int]]
    state_layer_ids: List[List[int]]
    # Per-tensor TP slice dim, used when prefill/decode attn_tp_size differ.
    state_dim_per_tensor: List[List[int]]
    # Number of rows before the slice axis in each per-slot state tensor.
    state_slice_outer_counts: List[List[int]]
    is_hybrid_mla_backend: bool
    # Per-tensor conv sub-block dims (GDN: [key_dim, key_dim, value_dim]) so the
    # scatter transfer can slice each independently head-sharded sub-block; None
    # per tensor when the single contiguous slice already matches the layout.
    state_conv_shard_groups: List[List[Optional[List[int]]]]
    ib_device: str
    gpu_id: int
    kv_head_num: int
    total_kv_head_num: int
    page_size: int
    # for system dp
    system_dp_rank: int
    # 用于 PP（流水线并行）下的 prefill
    pp_rank: int
    prefill_start_layer: int  # 本 PP 阶段负责的起始层
    # 本 prefill PP 阶段的绝对结束层（不含）。当 kv_data_ptrs 不采用扁平的按层索引布局时
    # （如 DeepSeek V4 按缓冲区类型组织的扁平列表），需要它来重建 PP 子区间。
    prefill_end_layer: Optional[int]
    # 仅用于 DeepSeek V4（及其他压缩 MLA）内存池。
    # 每层的全模型压缩比（取值 0/4/128）。连接层据此以 PP 感知的方式
    # 切分「按缓冲区类型组织的扁平列表」。
    mla_compression_ratios: Optional[List[int]]
    # 仅 NPU 使用，KV 缓冲分组数
    kv_buf_groups: int
    # 仅 NPU 使用，decode 侧的 KV 总层数
    total_kv_layers: int


class KVPoll:
    """KV 传输状态机的状态枚举（poll() 的返回值）。"""

    Failed = 0  # 传输失败
    Bootstrapping = 1  # 正在与 bootstrap server 握手
    WaitingForInput = 2  # 等待对端提供传输元数据
    Transferring = 3  # 数据传输中
    Success = 4  # 传输完成


class BaseKVManager(ABC):
    """KV 传输状态管理的基类。"""

    @abstractmethod
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ): ...

    @abstractmethod
    def register_to_bootstrap(self):
        """把 prefill server 的信息注册到 bootstrap server。"""
        ...


class BaseKVSender(ABC):
    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ): ...

    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """在本地登记请求的索引元数据，或把 kv 索引长度与 aux index 通知 decode 端。"""
        ...

    @abstractmethod
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
    ):
        """把给定 kv 索引处的 KV cache、以及给定索引处的额外 cache/state 发送到 decode 端。"""
        ...

    def pop_decode_prefix_len(self) -> int:
        """取出 decode 侧前缀长度（默认 0；支持 decode 侧 HiCache 的后端会覆盖）。"""
        return 0

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        """判断是否需要发送该 KV 分块（默认：只要有页就发）。"""
        return num_pages > 0

    @abstractmethod
    def get_transfer_metric(self) -> KVTransferMetric:
        """返回本 sender 的后端相关传输指标。"""
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """查询 KV cache 传输的当前状态。"""
        ...

    @abstractmethod
    def failure_exception(self):
        """当 KV cache 传输失败时抛出异常。"""
        ...

    def clear(self):
        """
        Clear any internal states.
        """
        pass

    def abort(self):
        """
        Abort the current transfer.
        """
        pass


class BaseKVReceiver(ABC):
    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ): ...

    @abstractmethod
    def init(
        self,
        prefill_dp_rank: int,
    ):
        """解析 bootstrap 元数据，并标记接收端已就绪、可接收传输元数据。"""
        ...

    @abstractmethod
    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        """把 kv 索引、aux index 与 state_indices 通知给 prefill 端（告知数据该写到哪）。"""
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """查询 KV cache 传输的当前状态。"""
        ...

    @abstractmethod
    def failure_exception(self):
        """当 KV cache 传输失败时抛出异常。"""
        ...

    def clear(self):
        """清理内部状态（默认空实现）。"""
        pass

    def abort(self):
        """中止当前传输（默认空实现）。"""
        pass


class BaseKVBootstrapServer(ABC):
    """bootstrap server 基类：供 prefill/decode 两侧交换连接元数据。"""

    @abstractmethod
    def __init__(self, host: str, port: int): ...
