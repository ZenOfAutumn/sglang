"""DP Attention（数据并行注意力）的全局状态管理与集合通信工具。

背景：
在 MoE 类模型（如 DeepSeek-V3）中，Attention 部分若采用纯 TP 切分，会导致
KV Cache 在每个 TP rank 上重复存储，显存利用率低。DP Attention 的思路是：
  - Attention 阶段：不同 DP rank 处理各自独立的一批 token（数据并行），
    每个 rank 只保存自己那部分请求的 KV Cache，显存不再冗余；
  - MoE / Dense FFN 阶段：需要全局所有 token，因此在进入 FFN 前做一次
    `dp_gather`（把各 DP rank 的 token 汇聚成全局张量），FFN 结束后再做
    `dp_scatter`（把全局张量切回本 rank 的那一段）。

本文件提供的能力：
  1. DP/TP/CP 各维度 rank 与 world size 的计算和全局缓存；
  2. gather 缓冲区（global/local buffer）的元信息管理与分配；
  3. 两种 padding 模式（MAX_LEN / SUM_LEN）的选择策略；
  4. 基于 Triton 的 memcpy 以及 gather/scatter 的具体集合通信实现。
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.distributed import (
    GroupCoordinator,
    get_attn_context_model_parallel_rank,
    get_attn_context_model_parallel_world_size,
    get_attn_cp_group,
    get_attn_tensor_model_parallel_rank,
    get_attn_tensor_model_parallel_world_size,
    get_attn_tp_group,
)
from sglang.srt.distributed import get_moe_dp_group as _get_moe_dp_group
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.utils import get_bool_env_var, is_hip

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# ---------------------------------------------------------------------------
# 全局状态（进程级单例）。这些变量在 `initialize_dp_attention` 中被赋值，
# 之后由各处的 getter 读取，避免层层透传参数。
# ---------------------------------------------------------------------------
# 当前进程在 attention 数据并行组中的 rank
_ATTN_DP_RANK: Optional[int] = None
# attention 数据并行组的大小（未开启 DP attention 时为 1）
_ATTN_DP_SIZE: Optional[int] = None
# "local" 系列用于 moe_dense_tp_size 场景：dense 层可能使用比全局 TP 更小的
# TP 度，于是在每个 dense-TP 子组内部又形成了一套局部的 DP 划分。
_LOCAL_ATTN_DP_SIZE: Optional[int] = None
_LOCAL_ATTN_DP_RANK: Optional[int] = None
# 是否开启了 DP attention
_ENABLE_DP_ATTENTION_FLAG: bool = False
# 针对混合 SSM（Hybrid-Mamba）模型的开关：允许在存在空闲 rank（token 数为 0）时
# 仍然走 MAX_LEN 模式，以便通过"伪造行"的方式让空闲 rank 参与计算。
_DP_MAX_LEN_WITH_IDLE = False

_is_hip = is_hip()
# ROCm 7.0.0 alpha 上 RCCL 的临时规避开关
_USE_ROCM700A_WA = _is_hip and get_bool_env_var("SGLANG_USE_ROCM700A")


class DpPaddingMode(IntEnum):
    """DP gather 时的 padding（对齐）模式。

    两种模式对应两种不同的通信原语，通信量不同：
      - MAX_LEN：把每个 rank 的 token 数都 pad 到全局最大值 max_len，
        全局缓冲区大小为 ``max_len * dp_size``，用 ``all_gather_into_tensor`` 汇聚。
      - SUM_LEN：全局缓冲区大小恰为 ``sum_len = sum(global_num_tokens)``，
        各 rank 把自己的数据写到对应偏移、其余位置填 0，再用 ``all_reduce`` 汇聚。
    """

    # 将 token 数 pad 到最大长度，然后用 `all_gather_into_tensor` 聚合
    MAX_LEN = auto()
    # 将 token 数 pad 到总长度，然后用 `all_reduce` 聚合
    SUM_LEN = auto()

    def is_max_len(self):
        return self == DpPaddingMode.MAX_LEN

    def is_sum_len(self):
        return self == DpPaddingMode.SUM_LEN

    @classmethod
    def get_dp_padding_mode(
        cls, is_extend_in_batch, global_num_tokens: List[int]
    ) -> DpPaddingMode:
        """根据当前 batch 的 token 分布，选择通信开销更小的 padding 模式。

        Args:
            is_extend_in_batch: 本次 batch 中是否包含 extend（prefill）请求。
            global_num_tokens: 各 DP rank 的 token 数列表。
        """
        dp_size = get_attention_dp_size()

        # 当 batch 中含有 extend 请求且 dp_size > 1 时，各 rank 的 token 数差异
        # 通常很大（prefill 长度不一），此时用 SUM_LEN 可避免按最大长度 padding
        # 带来的巨大冗余开销。
        # 而 dp_size == 1 时 max_len == sum_len，优先选 MAX_LEN 以便启用
        # symmetric memory 优化（DSA CP 等特性需要）。
        if is_extend_in_batch and dp_size > 1:
            # 混合 SSM 模型需要通过 MAX_LEN 的"伪造行"机制让空闲 rank 也有输入；
            # 其他模型仍走主线的 SUM_LEN。
            if _DP_MAX_LEN_WITH_IDLE and min(global_num_tokens) == 0:
                return DpPaddingMode.MAX_LEN
            return DpPaddingMode.SUM_LEN

        # 选择通信量最小的模式：
        #   MAX_LEN 的通信量 ~ max_len * dp_size
        #   SUM_LEN 的通信量 ~ sum_len * 2（all_reduce 约为 all_gather 的 2 倍）
        # 两者相等时优先 MAX_LEN，以启用 symmetric memory。
        max_len = max(global_num_tokens)
        sum_len = sum(global_num_tokens)
        if sum_len * 2 >= max_len * dp_size:
            return cls.MAX_LEN
        else:
            return cls.SUM_LEN

    @classmethod
    def get_default_mode_in_cuda_graph(cls) -> DpPaddingMode:
        """CUDA Graph 捕获时使用的默认模式。

        CUDA Graph 要求形状固定，因此不能按 batch 动态选择模式，统一使用 MAX_LEN。
        """
        # TODO(kkhuang-amd): noqa, 这是针对 rocm 7.0.0 alpha 的临时规避方案，
        # 待 RCCL 修复后即可安全移除。
        if _USE_ROCM700A_WA:
            return cls.SUM_LEN
        else:
            return cls.MAX_LEN


class _DpGatheredBufferWrapper:
    """DP gather 缓冲区的元信息容器（全部为类变量，进程级单例）。

    这里只保存"如何创建 buffer"所需的元信息（hidden_size / dtype / device /
    长度等），buffer 本身在每次调用 getter 时按需 `torch.empty` 分配，
    以便复用 PyTorch 的 caching allocator 或 symmetric memory 池。
    """

    # 模型 hidden size，即 buffer 的第二维
    _hidden_size: int
    # buffer 的数据类型
    _dtype: torch.dtype
    # buffer 所在设备
    _device: torch.device
    # 全局 buffer 的行数（gather 之后的总 token 数）
    _global_dp_buffer_len: int
    # 本 rank 的 buffer 行数（本地 token 数，可能已为 cuda graph padding）
    _local_dp_buffer_len: int
    # 是否采用"按最大长度 padding"（即 MAX_LEN 模式），决定能否用 symmetric memory
    _dp_max_padding: bool
    # 各 DP rank 的 token 数（CPU 侧列表）
    _global_num_tokens: Optional[List[int]]
    # 本次 batch 是否包含 extend 请求
    _is_extend_in_batch: bool

    @classmethod
    def set_metadata(cls, hidden_size: int, dtype: torch.dtype, device: torch.device):
        """设置 buffer 的静态元信息，仅在初始化时调用一次。"""
        cls._hidden_size = hidden_size
        cls._dtype = dtype
        cls._device = device

    @classmethod
    def set_dp_buffer_len(
        cls,
        global_dp_buffer_len: int,
        local_dp_buffer_len: int,
        dp_max_padding: bool,
        global_num_tokens: Optional[List[int]] = None,
    ):
        """设置本次 forward 的 buffer 长度信息，每个 batch 前调用一次。"""
        cls._global_dp_buffer_len = global_dp_buffer_len
        cls._local_dp_buffer_len = local_dp_buffer_len
        cls._dp_max_padding = dp_max_padding
        cls._global_num_tokens = global_num_tokens

    @classmethod
    def get_global_dp_buffer(cls, group: GroupCoordinator) -> torch.Tensor:
        """分配用于存放"全局所有 token"的 buffer，形状为 (global_len, hidden_size)。

        只有在 MAX_LEN 模式（各 rank 长度一致）下才能使用 symmetric memory，
        因为对称内存要求所有 rank 分配完全相同的大小。
        """
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (cls._global_dp_buffer_len, cls._hidden_size),
                dtype=cls._dtype,
                device=cls._device,
            )
        return buffer

    @classmethod
    def get_local_dp_buffer(cls, group: GroupCoordinator) -> torch.Tensor:
        """分配用于存放"本 rank token"的 buffer，形状为 (local_len, hidden_size)。"""
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (cls._local_dp_buffer_len, cls._hidden_size),
                dtype=cls._dtype,
                device=cls._device,
            )
        return buffer

    @classmethod
    def get_global_dp_buffer_len(cls) -> int:
        return cls._global_dp_buffer_len

    @classmethod
    def get_local_dp_buffer_len(cls) -> int:
        return cls._local_dp_buffer_len

    @classmethod
    def get_dp_global_num_tokens(cls) -> List[int]:
        return cls._global_num_tokens

    @classmethod
    def get_dp_hidden_size(cls) -> int:
        return cls._hidden_size

    @classmethod
    def get_dp_dtype(cls) -> torch.dtype:
        return cls._dtype

    @classmethod
    def get_dp_device(cls) -> torch.device:
        return cls._device

    @classmethod
    def set_is_extend_in_batch(cls, is_extend_in_batch: bool):
        cls._is_extend_in_batch = is_extend_in_batch

    @classmethod
    def get_is_extend_in_batch(cls) -> bool:
        return cls._is_extend_in_batch

    @classmethod
    def is_dp_max_padding(cls) -> bool:
        return cls._dp_max_padding


# ---------------------------------------------------------------------------
# 下面一组模块级函数是 `_DpGatheredBufferWrapper` 的薄封装，
# 便于外部以函数形式访问，无需感知内部类。
# ---------------------------------------------------------------------------


def set_dp_buffer_len(
    global_dp_buffer_len: int,
    local_dp_buffer_len: int,
    dp_max_padding: bool,
    global_num_tokens: Optional[List[int]] = None,
):
    """设置本次 forward 的 DP buffer 长度信息。"""
    _DpGatheredBufferWrapper.set_dp_buffer_len(
        global_dp_buffer_len, local_dp_buffer_len, dp_max_padding, global_num_tokens
    )


def get_global_dp_buffer(group: GroupCoordinator) -> torch.Tensor:
    """获取（分配）全局 gather buffer。"""
    return _DpGatheredBufferWrapper.get_global_dp_buffer(group=group)


def get_local_dp_buffer(group: GroupCoordinator) -> torch.Tensor:
    """获取（分配）本 rank 的局部 buffer。"""
    return _DpGatheredBufferWrapper.get_local_dp_buffer(group=group)


def get_global_dp_buffer_len() -> int:
    """返回全局 buffer 的行数（token 总数）。"""
    return _DpGatheredBufferWrapper.get_global_dp_buffer_len()


def get_local_dp_buffer_len() -> int:
    """返回本 rank buffer 的行数。"""
    return _DpGatheredBufferWrapper.get_local_dp_buffer_len()


def get_dp_global_num_tokens() -> List[int]:
    """返回各 DP rank 的 token 数列表（CPU 侧）。"""
    return _DpGatheredBufferWrapper.get_dp_global_num_tokens()


def get_dp_hidden_size() -> int:
    """返回 buffer 的 hidden size。"""
    return _DpGatheredBufferWrapper.get_dp_hidden_size()


def get_dp_dtype() -> torch.dtype:
    """返回 buffer 的 dtype。"""
    return _DpGatheredBufferWrapper.get_dp_dtype()


def get_dp_device() -> torch.device:
    """返回 buffer 所在的设备。"""
    return _DpGatheredBufferWrapper.get_dp_device()


def set_is_extend_in_batch(is_extend_in_batch: bool):
    """记录本次 batch 是否包含 extend（prefill）请求。"""
    _DpGatheredBufferWrapper.set_is_extend_in_batch(is_extend_in_batch)


def get_is_extend_in_batch() -> bool:
    """查询本次 batch 是否包含 extend（prefill）请求。"""
    return _DpGatheredBufferWrapper.get_is_extend_in_batch()


def is_dp_max_padding() -> bool:
    """当前是否采用"按最大长度 padding"（MAX_LEN 模式）。"""
    return _DpGatheredBufferWrapper.is_dp_max_padding()


def compute_dp_attention_world_info(
    enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size: int = 1
):
    """由全局 TP rank 反推 attention 侧的 (tp, dp) 坐标。

    全局 ``tp_size`` 个 rank 被划分为三个维度：DP × CP × TP，满足
    ``tp_size = attn_dp_size * attn_cp_size * attn_tp_size``。

    Args:
        enable_dp_attention: 是否开启 DP attention。
        tp_rank: 当前进程的全局 TP rank。
        tp_size: 全局 TP world size。
        dp_size: DP attention 的并行度。
        attn_cp_size: attention 的 context parallel 并行度。

    Returns:
        (attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size) 四元组。
    """
    # 未开启 DP attention 时退化为 dp_size == 1，即纯 TP
    attn_dp_size = dp_size if enable_dp_attention else 1
    # 剩下的 rank 分给 attention 内部的 TP
    attn_tp_size = tp_size // attn_dp_size // attn_cp_size
    attn_tp_rank = tp_rank % attn_tp_size

    if not enable_dp_attention:
        attn_dp_rank = 0
    else:
        # rank 布局为 (dp, cp, tp)，其中 tp 是变化最快的维度：
        # tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size + attn_tp_rank
        # 因此除以 (attn_tp_size * attn_cp_size) 即可得到 dp 维坐标。
        attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)

    return attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size


def compute_dp_attention_local_info(
    enable_dp_attention, tp_rank, tp_size, dp_size, moe_dense_tp_size
):
    """计算 dense 层子组内部的"局部" attention (tp, dp) 坐标。

    当通过 ``moe_dense_tp_size`` 为 MoE 模型中的 dense（共享专家 / 稠密 FFN）层
    指定了一个更小的 TP 度时，全局 rank 会被切成若干个 ``moe_dense_tp_size``
    大小的子组；在每个子组内部又形成一套独立的 DP/TP 划分。本函数返回的就是
    这套"局部"坐标。未开启 DP attention 时直接退化为全局 TP 坐标。

    Returns:
        (local_attn_tp_rank, local_attn_tp_size, local_attn_dp_rank) 三元组。
    """
    if not enable_dp_attention:
        return tp_rank, tp_size, 0

    # 子组大小；未指定 moe_dense_tp_size 时就是全局 TP
    local_tp_size = moe_dense_tp_size if moe_dense_tp_size else tp_size
    # 当前 rank 在子组内的偏移
    local_tp_rank = tp_rank % local_tp_size
    # 子组内部的 DP 度 = 全局 DP 度 / 子组个数
    local_dp_size = max(1, dp_size // (tp_size // local_tp_size))

    # 在子组内部再按 (dp, tp) 二维展开
    local_attn_tp_size = local_tp_size // local_dp_size
    local_attn_dp_rank = local_tp_rank // local_attn_tp_size
    local_attn_tp_rank = local_tp_rank % local_attn_tp_size

    return local_attn_tp_rank, local_attn_tp_size, local_attn_dp_rank


def initialize_dp_attention(
    server_args: ServerArgs,
    model_config: ModelConfig,
):
    """初始化 DP attention 的全局状态，需在分布式进程组建立之后调用一次。"""
    global _ATTN_DP_RANK, _ATTN_DP_SIZE
    global _LOCAL_ATTN_DP_SIZE, _LOCAL_ATTN_DP_RANK, _ENABLE_DP_ATTENTION_FLAG
    global _DP_MAX_LEN_WITH_IDLE
    # 混合 SSM（Mamba/Attention 混合）模型的 HF config 中含有
    # `hybrid_override_pattern` 字段，以此识别并启用带空闲 rank 的 MAX_LEN 路径
    _DP_MAX_LEN_WITH_IDLE = (
        getattr(model_config.hf_config, "hybrid_override_pattern", None) is not None
    )
    enable_dp_attention = server_args.enable_dp_attention
    dp_size = server_args.dp_size
    moe_dense_tp_size = server_args.moe_dense_tp_size
    attn_cp_size = server_args.attn_cp_size

    _ENABLE_DP_ATTENTION_FLAG = enable_dp_attention

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    # 计算全局 DP rank
    _, _, _ATTN_DP_RANK, _ = compute_dp_attention_world_info(
        enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size
    )
    # 计算 dense 子组内的局部 DP rank
    _, _, _LOCAL_ATTN_DP_RANK = compute_dp_attention_local_info(
        enable_dp_attention, tp_rank, tp_size, dp_size, moe_dense_tp_size
    )

    if enable_dp_attention:
        _ATTN_DP_SIZE = dp_size
        if moe_dense_tp_size is None:
            # 未单独指定 dense TP 度时，局部与全局一致
            _LOCAL_ATTN_DP_SIZE = _ATTN_DP_SIZE
        else:
            # 否则局部 DP 度 = 全局 DP 度 / dense 子组个数
            _LOCAL_ATTN_DP_SIZE = max(1, dp_size // (tp_size // moe_dense_tp_size))
    else:
        _ATTN_DP_SIZE = 1
        _LOCAL_ATTN_DP_SIZE = 1

    # 记录 gather buffer 的静态元信息
    _DpGatheredBufferWrapper.set_metadata(
        hidden_size=model_config.hidden_size,
        dtype=model_config.dtype,
        device=torch.device(server_args.device),
    )


def is_dp_attention_enabled() -> bool:
    """是否开启了 DP attention。"""
    return _ENABLE_DP_ATTENTION_FLAG


def is_allocation_symmetric() -> bool:
    """当前的显存分配是否在各 rank 间对称（可用 symmetric memory）。

    未开启 DP attention 时天然对称；开启后只有 MAX_LEN 模式各 rank 长度一致。
    """
    return not is_dp_attention_enabled() or is_dp_max_padding()


def get_attention_tp_group() -> GroupCoordinator:
    """返回 attention 内部的 TP 通信组。"""
    return get_attn_tp_group()


def get_attention_tp_rank() -> int:
    """返回当前进程在 attention TP 组中的 rank。"""
    return get_attn_tensor_model_parallel_rank()


def get_attention_tp_size() -> int:
    """返回 attention TP 组的大小。"""
    return get_attn_tensor_model_parallel_world_size()


def get_attention_cp_group() -> GroupCoordinator:
    """返回 attention 的 context parallel（序列切分）通信组。"""
    return get_attn_cp_group()


def get_attention_cp_rank() -> int:
    """返回当前进程在 attention CP 组中的 rank。"""
    return get_attn_context_model_parallel_rank()


def get_attention_cp_size() -> int:
    """返回 attention CP 组的大小。"""
    return get_attn_context_model_parallel_world_size()


def get_attention_dp_rank() -> int:
    """返回当前进程的全局 attention DP rank。"""
    assert _ATTN_DP_RANK is not None, "dp attention not initialized!"
    return _ATTN_DP_RANK


def get_attention_dp_size() -> int:
    """返回全局 attention DP 组的大小。"""
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"
    return _ATTN_DP_SIZE


def get_local_attention_dp_rank() -> int:
    """返回 dense 子组内部的局部 attention DP rank。"""
    assert _LOCAL_ATTN_DP_RANK is not None, "dp attention not initialized!"
    return _LOCAL_ATTN_DP_RANK


def get_local_attention_dp_size() -> int:
    """返回 dense 子组内部的局部 attention DP 组大小。"""
    assert _LOCAL_ATTN_DP_SIZE is not None, "dp attention not initialized!"
    return _LOCAL_ATTN_DP_SIZE


@contextmanager
def disable_dp_size():
    """临时把 DP size 置为 1，退出上下文后自动还原。

    该方法用于投机采样（speculative decoding）的 draft worker：draft 模型可能
    使用与 target 模型不同的并行度，运行 draft 模型期间需要屏蔽 DP 划分。

    Args:
        tp_group (GroupCoordinator): tp 组协调器
    """
    global _ATTN_DP_SIZE
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"

    old_dp_size = _ATTN_DP_SIZE
    _ATTN_DP_SIZE = 1
    try:
        yield
    finally:
        # 无论上下文内是否抛异常，都要还原全局状态
        _ATTN_DP_SIZE = old_dp_size


def get_dp_local_info(forward_batch: ForwardBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算本 rank 数据在全局 buffer 中的 (起始偏移, 长度)，结果为 GPU 张量。

    偏移由各 rank token 数的前缀和给出：``start = sum(global_num_tokens[:dp_rank])``。
    结果会缓存在 ``forward_batch`` 上，避免同一次 forward 中重复计算。
    使用 GPU 张量而非 Python int，是为了避免 device→host 同步、并兼容 CUDA Graph。
    """
    # `get_dp_local_info` 只在全局 DP 的 gather / scatter 中调用，这里使用全局 DP rank
    dp_rank = get_attention_dp_rank()

    if forward_batch.dp_local_start_pos is None:
        # 前缀和：cumtokens[i] = global_num_tokens[0..i] 之和
        cumtokens = torch.cumsum(forward_batch.global_num_tokens_gpu, dim=0)
        if dp_rank == 0:
            # rank 0 从 0 开始；用 zeros_like 保持 dtype/device 一致
            local_start_pos = torch.zeros_like(cumtokens[0])
        else:
            # 其他 rank 的起点是前面所有 rank 的 token 总数
            local_start_pos = cumtokens[dp_rank - 1]
        local_num_tokens = forward_batch.global_num_tokens_gpu[dp_rank]

        # 缓存到 forward_batch，供后续多次 gather/scatter 复用
        forward_batch.dp_local_start_pos = local_start_pos
        forward_batch.dp_local_num_tokens = local_num_tokens

    return forward_batch.dp_local_start_pos, forward_batch.dp_local_num_tokens


def get_dp_local_slice_cpu(
    forward_batch: ForwardBatch,
    can_run_graph: bool,
    cuda_graph_batch: Optional[int],
) -> Tuple[int, int]:
    """CPU 版本的本地切片计算，返回 Python int 的 (起始偏移, 长度)。

    与 `get_dp_local_info` 的区别：
      - 直接用 CPU 侧的 ``global_num_tokens_cpu``，不产生 D2H 同步；
      - 额外处理 CUDA Graph 的等长 padding 布局。

    Args:
        can_run_graph: 本次是否走 CUDA Graph 回放。走 graph 时每个 rank 在 buffer 中
            占据固定的 ``cuda_graph_batch`` 行，因此偏移是简单的 ``rank * batch``；
            否则按各 rank 真实 token 数做前缀和。
        cuda_graph_batch: CUDA Graph 捕获时每个 rank 的固定行数。
    """
    # 在按 rank padding 的 buffer 中，DP 本地数据的 (start, length) 切片
    global_num_tokens = forward_batch.global_num_tokens_cpu
    dp_rank = get_attention_dp_rank()
    local_num_tokens = global_num_tokens[dp_rank]
    if can_run_graph:
        # CUDA Graph 下各 rank 等长 padding，偏移可直接相乘得到
        local_start_pos = dp_rank * cuda_graph_batch
    else:
        # 非 graph 路径按真实 token 数做前缀和
        local_start_pos = sum(global_num_tokens[:dp_rank])
    return local_start_pos, local_num_tokens


@triton.jit
def memcpy_triton_kernel(
    dst_ptr,
    src_ptr,
    offset_ptr,
    sz_ptr,
    offset_src: tl.constexpr,
    chunk_size,  # 对 offset 和 sz 的放大倍数（即每行的元素个数）
    BLOCK_SIZE: tl.constexpr,
):
    """带偏移的显存拷贝 kernel。

    之所以不用 `torch.Tensor.copy_` / 切片赋值，是因为这里的 ``offset`` 和 ``sz``
    都是**GPU 张量**（而非 Python int）。用切片会强制 D2H 同步，且无法被 CUDA Graph
    捕获；用自定义 kernel 则可以在 device 端直接读取这两个标量。

    Args:
        offset_ptr: 指向偏移量标量的指针（单位：行）。
        sz_ptr: 指向拷贝行数标量的指针（单位：行）。
        offset_src: 编译期常量。True 表示偏移作用在 src 上（scatter：从全局取本地），
            False 表示偏移作用在 dst 上（gather：把本地写入全局）。
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    # 行偏移 / 行数 → 元素偏移 / 元素数；转 int64 防止大张量溢出
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size

    start_index = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    # 尾块可能不足 BLOCK_SIZE，用 mask 屏蔽越界元素
    mask = start_index + offs < sz

    if offset_src:
        # scatter 方向：src[offset : offset+sz] -> dst[0 : sz]
        data = tl.load(src_ptr + offset + start_index + offs, mask=mask)
        tl.store(dst_ptr + start_index + offs, data, mask=mask)
    else:
        # gather 方向：src[0 : sz] -> dst[offset : offset+sz]
        data = tl.load(src_ptr + start_index + offs, mask=mask)
        tl.store(dst_ptr + offset + start_index + offs, data, mask=mask)


def prod(x):
    """求可迭代对象所有元素的乘积（空序列返回 1）。"""
    return functools.reduce(lambda a, b: a * b, x, 1)


def memcpy_triton(dst, src, dim, offset, sz, offset_src):
    """`memcpy_triton_kernel` 的 Python 封装，负责计算 grid 并做形状校验。

    Args:
        dst: 目标张量。
        src: 源张量。
        dim: 偏移所在的维度，目前只支持 0（第 0 维为 token 维）。
        offset: GPU 标量张量，行偏移。
        sz: GPU 标量张量，拷贝的行数。
        offset_src: 偏移是否作用在 src 上，参见 kernel 说明。
    """
    # 实际拷贝量不会超过两者中较小的元素数，以此上界估算 grid
    max_size = min(src.numel(), dst.numel())
    assert dim == 0, "dim != 0 unsupported"
    # 除第 0 维外形状必须一致，这样每行的元素个数才相同
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same shape"
    chunk_size = prod(src.shape[1:])
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(max_size, BLOCK_SIZE),)

    memcpy_triton_kernel[grid](dst, src, offset, sz, offset_src, chunk_size, BLOCK_SIZE)


def _dp_gather_via_all_reduce(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    """SUM_LEN 模式的 gather 实现：先把本地数据写到全局 buffer 的对应偏移，再 all_reduce。

    由于全局 buffer 先被清零、且每个 rank 只写自己的那一段（互不重叠），
    all_reduce（求和）的结果就等价于一次拼接（concat）。

    Args:
        is_partial: 本地数据是否是"部分和"。
            True 表示 local_tokens 在 attn TP 组内是分片的部分结果，所有 TP rank
            都要参与写入（相加后才是完整值）；
            False 表示各 TP rank 持有的是完整副本，只需 rank 0 写入以免重复累加。

    示例（``tp_size=2``、``attn_dp_size=2``、``attn_tp_size=1``，各 rank token 数不等）：

    设 rank0 有 1 个 token ``a0``，rank1 有 2 个 token ``b0, b1``。
    SUM_LEN 模式下全局 buffer 长度 = ``1 + 2 = 3``（不做 padding），
    各 rank 的偏移由 `get_dp_local_info` 的前缀和给出：rank0→0，rank1→1。

    ::

        输入 local_tokens:
            rank0: [a0]              local_start_pos=0, local_num_tokens=1
            rank1: [b0, b1]          local_start_pos=1, local_num_tokens=2

        写入后（fill_(0) + memcpy_triton，各写各段，互不重叠）:
            rank0: [a0,  0,  0]
            rank1: [ 0, b0, b1]
                    ↑   ↑   ↑
                  只有一个 rank 在该位置写了非零值

        all_reduce(SUM) 之后，两个 rank 的 global_tokens 都是:
            [a0, b0, b1]             ← 等价于按 dp_rank 顺序 concat

    ``is_partial`` 的影响（改设 ``attn_tp_size=2``，即 rank0/rank1 同属一个 DP 组）：

    ::

        is_partial=True   两卡各持部分和 [x] 与 [y]，都写入 → all_reduce 得 [x+y]（正确）
        is_partial=False  两卡各持完整副本 [v]，若都写入会得到 [2v]（错误）；
                          故只让 attn_tp_rank==0 写入 → all_reduce 得 [v]（正确）
    """
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    # 必须先清零，否则 all_reduce 会把残留数据一起累加进来
    global_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()

    # is_partial=False 时只让 attn TP rank 0 写入，避免副本被重复累加
    if local_tokens.shape[0] > 0 and (is_partial or get_attention_tp_rank() == 0):
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between global_tokens and local_tokens not allowed"

        # 把 local_tokens 写入 global_tokens[local_start_pos : +local_num_tokens]
        memcpy_triton(
            global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False
        )

    # input_ids 是 int32 类型。单机场景下需要使用 inplace_all_reduce，
    # 因为自定义 all-reduce（custom all reduce）走的是原地路径。
    NUM_GPUS_PER_NODE = 8
    if (
        not local_tokens.dtype.is_floating_point
        and get_tensor_model_parallel_world_size() <= NUM_GPUS_PER_NODE
    ):
        from sglang.srt.distributed.parallel_state import inplace_all_reduce

        inplace_all_reduce(global_tokens, group_name=get_tp_group().unique_name)

    else:
        global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)


def _dp_gather_via_all_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    """MAX_LEN 模式的 gather 实现：直接用 all_gather 汇聚等长的分片。

    当 attn TP > 1 时不能直接 all_gather（同一 DP 组内的多个 TP rank 会重复贡献），
    因此先在 attn TP 组内做 reduce_scatter 把数据规约并切成不重叠的小片，
    再在全局 TP 组上 all_gather —— 总通信量与直接 all_gather 相当，且结果正确。

    与 SUM_LEN 的关键区别：MAX_LEN 要求**各 rank 的 local_tokens 等长**
    （调用方已 pad 到 ``max(global_num_tokens)``），因此可以直接用 all_gather，
    无需偏移量；代价是 padding 行会占用带宽。

    示例 1 —— ``attn_tp_size == 1`` 的快路径（``tp_size=2``、``attn_dp_size=2``）:

    每个 rank 的数据本身就是一份独立分片，直接 all_gather 沿第 0 维拼接即可。

    ::

        输入 local_tokens（已 pad 到等长 2）:
            rank0: [a0, a1]
            rank1: [b0, b1]

        all_gather_into_tensor 后（两个 rank 结果相同，按 rank 序拼接）:
            global_tokens: [a0, a1, b0, b1]

    示例 2 —— ``attn_tp_size == 2`` 的通用路径（``tp_size=4``、``attn_dp_size=2``）:

    rank0/1 同属 DP 组 0，rank2/3 同属 DP 组 1。此时若直接 all_gather，同组两卡
    会各贡献一份，全局张量长度会翻倍且内容重复。故先在组内 reduce_scatter：
    把长度 2 的 local_tokens 规约后切成 2 段，每卡只留 1 段（互不重叠），
    再 all_gather 这些小段，正好拼回长度 4 的全局张量。

    ::

        输入 local_tokens（is_partial=True，组内两卡各持部分和）:
            rank0: [a0', a1']        rank2: [b0', b1']
            rank1: [a0", a1"]        rank3: [b0", b1"]
            （满足 a0'+a0" = a0，a1'+a1" = a1，b 同理）

        组内 reduce_scatter_tensor 后，每卡持有 1 段完整结果:
            rank0: [a0]              rank2: [b0]
            rank1: [a1]              rank3: [b1]

        全局 all_gather_into_tensor 后（4 个 rank 结果相同）:
            global_tokens: [a0, a1, b0, b1]

    ``is_partial=False`` 时组内两卡持有的是**完整副本**而非部分和，
    直接 reduce_scatter 求和会得到 2 倍值；因此先把 ``attn_tp_rank != 0``
    的 local_tokens 清零，使求和结果仍等于 rank 0 的原值。
    """
    if get_attention_tp_size() == 1:
        # attn TP = 1：每个 rank 的数据本身就是一份独立分片，直接 all_gather
        get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
        return

    if not is_partial:
        # 非部分和场景下各 TP rank 是完整副本，只保留 rank 0 的值，
        # 其余置零，这样后续 reduce_scatter（求和）不会把副本重复累加
        if get_attention_tp_rank() != 0:
            local_tokens.fill_(0)
    # 取出本 TP rank 在 reduce_scatter 后应负责的那一片
    scattered_local_tokens = local_tokens.tensor_split(get_attention_tp_size())[
        get_attention_tp_rank()
    ]
    # 在 attn TP 组内规约并切分：每个 rank 拿到互不重叠的一小段完整结果
    get_attention_tp_group().reduce_scatter_tensor(scattered_local_tokens, local_tokens)
    # 全局 TP 组上把所有小段拼接成完整的全局张量
    get_tp_group().all_gather_into_tensor(global_tokens, scattered_local_tokens)


def _dp_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    """gather 的统一入口，按 padding 模式分派到具体实现。"""
    if forward_batch.dp_padding_mode.is_max_len():
        _dp_gather_via_all_gather(
            global_tokens, local_tokens, forward_batch, is_partial
        )
    else:
        _dp_gather_via_all_reduce(
            global_tokens, local_tokens, forward_batch, is_partial
        )


def dp_gather_partial(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    """汇聚"部分和"张量：local_tokens 在 attn TP 组内是分片的，需要各 rank 相加。

    典型场景：attention 输出在 TP 维度上是部分结果，尚未做 all-reduce。
    """
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=True)


def dp_gather_replicate(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    """汇聚"复制"张量：local_tokens 在 attn TP 组内每个 rank 都是完整副本。

    典型场景：hidden_states 已在 TP 组内 all-reduce 过，或 input_ids 这类天然复制的数据。
    """
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=False)


def dp_scatter(
    local_tokens: torch.Tensor,  # 输出
    global_tokens: torch.Tensor,  # 输入
    forward_batch: ForwardBatch,
):
    """gather 的逆操作：从全局张量中切出本 DP rank 负责的那一段。

    通常用在 MoE / dense FFN 计算完成后，把全局结果切回各 DP rank 继续做
    后续的 attention 层。
    """
    # local_num_tokens 不一定等于 local_tokens.shape[0]，
    # 因为 local_tokens 可能为了 cuda graph 而做了 padding
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    # 先清零，保证 padding 出来的多余行是确定值（0）而非脏数据
    local_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0:
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between local_tokens and global_tokens not allowed"

        # 把 global_tokens[local_start_pos : +local_num_tokens] 拷贝到 local_tokens
        memcpy_triton(
            local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True
        )


def dp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    """在 DP 维度上做 reduce-scatter：规约全局张量并把结果切回各 DP rank。

    当 tp_size == dp_size（即 attn TP 为 1）时，一次 reduce_scatter 即可。
    否则需要两步：先在全局 TP 组上 reduce_scatter 得到更细的分片，
    再在 attn TP 组内 all_gather 还原成每个 DP rank 完整的那一段。
    """
    if get_tensor_model_parallel_world_size() == get_attention_dp_size():
        get_tp_group().reduce_scatter_tensor(output, input)
    else:
        # 取出本全局 TP rank 应负责的细分片
        scattered_local_tokens = input.tensor_split(
            get_tensor_model_parallel_world_size()
        )[get_tensor_model_parallel_rank()]
        # 第一步：全局规约并细分
        get_tp_group().reduce_scatter_tensor(scattered_local_tokens, input)
        # 第二步：在 attn TP 组内拼回本 DP rank 的完整分片
        get_attention_tp_group().all_gather_into_tensor(output, scattered_local_tokens)


def attn_tp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    """在 attention TP 组内执行 reduce-scatter。"""
    return get_attention_tp_group().reduce_scatter_tensor(output, input)


def attn_cp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    """在 attention CP（context parallel）组内执行 reduce-scatter。"""
    return get_attention_cp_group().reduce_scatter_tensor(output, input)


def attn_tp_all_reduce(input: torch.Tensor):
    """在 attention TP 组内执行 all-reduce。"""
    return get_attention_tp_group().all_reduce(input)


def attn_tp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    """在 attention TP 组内执行 all-gather，结果写入预分配的 output 张量。"""
    return get_attention_tp_group().all_gather_into_tensor(output, input)


def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    """在 attention CP 组内执行 all-gather，结果写入预分配的 output 张量。"""
    return get_attention_cp_group().all_gather_into_tensor(output, input)


def get_moe_cp_group() -> GroupCoordinator:
    """返回 MOE_DP 组；当 attn_cp_size > moe_dp_size 时该组会包含 CP 伙伴 rank。"""
    return _get_moe_dp_group()


def get_moe_cp_rank() -> int:
    """返回当前进程在 MOE_DP 组中的 rank。"""
    return _get_moe_dp_group().rank_in_group


def get_moe_cp_size() -> int:
    """返回 MOE_DP 组的大小。"""
    return _get_moe_dp_group().world_size


def is_enable_moe_cp_allgather() -> bool:
    """当 moe_dp_size < attn_cp_size 时返回 True，此时进入 MoE 前需跨 CP rank 做 all-gather。

    原因：attention 阶段序列被 CP 切分到多个 rank，而 MoE 的 DP 度更小，
    因此必须先把被 CP 切散的 token 汇总回来，MoE 才能看到完整序列。
    """
    from sglang.srt.server_args import get_global_server_args

    sa = get_global_server_args()
    return sa.attn_cp_size > sa.moe_dp_size


def moe_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    """在 MOE_DP（含 CP 伙伴）组内执行 all-gather。"""
    return _get_moe_dp_group().all_gather_into_tensor(output, input)


def attn_tp_all_gather(output_list: List[torch.Tensor], input: torch.Tensor):
    """在 attention TP 组内执行 all-gather，结果以张量列表形式返回（每个 rank 一个）。"""
    return get_attention_tp_group().all_gather(input, output_tensor_list=output_list)
