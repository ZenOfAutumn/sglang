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
    """PD 分离传输中，KV 之外需额外搬运的「状态」类型（各后端据此拆分组件）。"""

    MAMBA = "mamba"  # Mamba/线性注意力的循环状态（temporal + conv）
    SWA = "swa"  # 滑动窗口注意力（Sliding Window Attention）的侧缓存
    DSA = "dsa"  # DeepSeek Sparse Attention 的 indexer 状态
    # DeepSeek-V4 的 unified_kv SWA 环形缓冲：按环形槽位逐行寻址
    # （req_pool_idx * ring_stride + pos % ring_stride），需作为独立组件处理。
    SWA_RING = "swa_ring"


@dataclasses.dataclass
class KVTransferMetric:
    """单次 KV 传输的指标；无法拆分某项的后端可将其留为 None。"""

    # 传输耗时（秒）；无法单独统计的后端留 None。
    transfer_latency_s: Optional[float] = None
    # 分配等待耗时（秒）；无法单独统计的后端留 None。
    alloc_latency_s: Optional[float] = None
    transfer_total_bytes: Optional[int] = None  # 本次传输的总字节数


class KVArgs:
    """KV 传输后端初始化所需的全部参数（内存布局、并行维度、设备信息等）。"""

    engine_rank: int  # 当前 engine 的全局 rank
    kv_data_ptrs: List[int]  # 各 KV 缓冲区的起始指针
    kv_data_lens: List[int]  # 各 KV 缓冲区的总字节长度
    kv_item_lens: List[int]  # 各 KV 缓冲区中单个 item（每 token/page）的字节数
    aux_data_ptrs: List[int]  # 辅助数据缓冲区指针（如 aux/metadata）
    aux_data_lens: List[int]  # 辅助数据缓冲区总长度
    aux_item_lens: List[int]  # 辅助数据单 item 字节数
    state_types: List[StateType]  # 需额外传输的状态类型列表（见 StateType）
    state_data_ptrs: List[List[int]]  # 各状态类型对应的缓冲区指针（按类型分组）
    state_data_lens: List[List[int]]  # 各状态缓冲区总长度
    state_item_lens: List[List[int]]  # 各状态缓冲区单 item 字节数
    # 每个 state 张量的 TP 切分维度；当 prefill/decode 两侧 attn_tp_size 不同时用于对齐切分。
    state_dim_per_tensor: List[List[int]]
    ib_device: str  # InfiniBand 设备名
    ib_traffic_class: str  # IB 流量分类（QoS）
    gpu_id: int  # 本地 GPU 编号
    kv_head_num: int  # 本 rank 的 KV head 数
    total_kv_head_num: int  # 全模型的 KV head 总数
    page_size: int  # 每 page 的 token 数
    # 用于 system dp（数据并行）
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
    """KV 发送端基类（prefill 侧）：负责把 KV cache 传给 decode 侧。"""

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
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


class BaseKVReceiver(ABC):
    """KV 接收端基类（decode 侧）：负责从 prefill 侧接收 KV cache。"""

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
