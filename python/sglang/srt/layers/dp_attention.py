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
    get_attn_cp_group,
    get_attn_cp_overlap_group,
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
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    derive_attention_widths,
    get_device,
    get_exec,
    get_flags,
    get_forward,
    get_parallel,
    get_resources,
    get_stream,
)
from sglang.srt.utils import get_bool_env_var, is_cpu, is_hip

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


def world_dp_gather_enabled() -> bool:
    """Whether DP gathers should use expanded WORLD after joiner admission."""
    dp = get_flags().dp
    return dp.use_world_group_for_gather and not dp.joiner_skip_all_gather


def enable_joiner_all_gather():
    get_flags().dp.joiner_skip_all_gather = False


def update_dp_attention_post_scale(new_dp_size: int, new_dp_rank: int):
    global _ATTN_DP_SIZE, _ATTN_DP_RANK
    _ATTN_DP_SIZE = new_dp_size
    _ATTN_DP_RANK = new_dp_rank
    get_parallel().stamp_derived_widths(attn_dp_size=new_dp_size)
    get_flags().dp.use_world_group_for_gather = True
    logger.debug(
        "[Elastic EP] dp_attention switched to WORLD: dp_size=%d dp_rank=%d",
        new_dp_size,
        new_dp_rank,
    )


_is_hip = is_hip()
# ROCm 7.0.0 alpha 上 RCCL 的临时规避开关
_USE_ROCM700A_WA = _is_hip and get_bool_env_var("SGLANG_USE_ROCM700A")
_is_cpu = is_cpu()


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

        # (trangdough) pplx-kernels a2a is a symmetric collective: every EP rank
        # must dispatch the same number of tokens or the device-side handshake
        # deadlocks (idle DP ranks with 0 tokens never signal their peers).
        # Force MAX_LEN so all ranks are padded to equal token counts.
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        moe_a2a_backend = get_moe_a2a_backend()
        if moe_a2a_backend.is_pplx():
            return DpPaddingMode.MAX_LEN

        if moe_a2a_backend.is_deepep_v2() and envs.SGLANG_DEEPEP_V2_FORCE_MAX_LEN.get():
            return DpPaddingMode.MAX_LEN

        # When is_extend_in_batch and dp_size > 1, use SUM_LEN to avoid padding
        # overhead from uneven token distribution.
        # For dp_size=1, max_len equals sum_len, so prefer MAX_LEN mode
        # to enable symmetric memory optimization (needed for DSA CP, etc.).
        if is_extend_in_batch and dp_size > 1:
            # Hybrid-SSM models materialize idle ranks via the MAX_LEN
            # fabricated-row conversion; other models keep mainline SUM_LEN.
            if get_flags().dp.max_len_with_idle and min(global_num_tokens) == 0:
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
    """Facade for the DP gathered-buffer state: allocation metadata lives on
    ``flags.dp`` (set once at initialize_dp_attention). The per-forward
    sizing quartet stays as class attributes: the values are read inside
    torch.compile-traced model code, and attribute-source ints get dynamo's
    automatic-dynamic treatment, while contextvars are untraceable and dict
    slots value-guard into the recompile limit (one recompile per distinct
    size)."""

    # Real defaults (not bare annotations): the sizing quartet is overwritten
    # per-forward by set_dp_buffer_len, but callers that run before the first
    # forward — notably the load-time mhc_pre prewarm, which has no ForwardBatch
    # yet — read _dp_max_padding via is_allocation_symmetric(). A bare
    # annotation creates no class attribute, so those reads raised
    # AttributeError. Defaulting _dp_max_padding to False (non-symmetric) is
    # safe for prewarm: it only JIT-compiles kernels and never enters a real
    # all-reduce, so the symmetric pool is not needed there.
    _global_dp_buffer_len: int = 0
    _local_dp_buffer_len: int = 0
    _dp_max_padding: bool = False
    _global_num_tokens: Optional[List[int]] = None
    _global_num_tokens_gpu: Optional[torch.Tensor] = None

    @classmethod
    def set_metadata(cls, hidden_size: int, dtype: torch.dtype, device: torch.device):

        dp = get_flags().dp
        dp.buffer_hidden_size = hidden_size
        dp.buffer_dtype = dtype
        dp.buffer_device = device

    @classmethod
    def set_dp_buffer_len(
        cls,
        global_dp_buffer_len: int,
        local_dp_buffer_len: int,
        dp_max_padding: bool,
        global_num_tokens: Optional[List[int]] = None,
        global_num_tokens_gpu: Optional[torch.Tensor] = None,
    ):
        """设置本次 forward 的 buffer 长度信息，每个 batch 前调用一次。"""
        cls._global_dp_buffer_len = global_dp_buffer_len
        cls._local_dp_buffer_len = local_dp_buffer_len
        cls._dp_max_padding = dp_max_padding
        cls._global_num_tokens = global_num_tokens
        cls._global_num_tokens_gpu = global_num_tokens_gpu

    @classmethod
    def get_global_dp_buffer(cls, group: GroupCoordinator) -> torch.Tensor:

        dp = get_flags().dp
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (cls._global_dp_buffer_len, dp.buffer_hidden_size),
                dtype=dp.buffer_dtype,
                device=dp.buffer_device,
            )
        return buffer

    @classmethod
    def get_local_dp_buffer(cls, group: GroupCoordinator) -> torch.Tensor:

        dp = get_flags().dp
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (cls._local_dp_buffer_len, dp.buffer_hidden_size),
                dtype=dp.buffer_dtype,
                device=dp.buffer_device,
            )
        return buffer

    @classmethod
    def get_global_dp_buffer_len(cls) -> int:
        return cls._global_dp_buffer_len

    @classmethod
    def get_local_dp_buffer_len(cls) -> int:
        return cls._local_dp_buffer_len

    @classmethod
    def set_local_dp_buffer_len(cls, local_dp_buffer_len: int) -> None:
        cls._local_dp_buffer_len = local_dp_buffer_len

    @classmethod
    def get_dp_global_num_tokens(cls) -> List[int]:
        return cls._global_num_tokens

    @classmethod
    def get_dp_global_num_tokens_gpu(cls) -> Optional[torch.Tensor]:
        return cls._global_num_tokens_gpu

    @classmethod
    def get_dp_hidden_size(cls) -> int:

        return get_flags().dp.buffer_hidden_size

    @classmethod
    def get_dp_dtype(cls) -> torch.dtype:

        return get_flags().dp.buffer_dtype

    @classmethod
    def get_dp_device(cls) -> torch.device:

        return get_flags().dp.buffer_device

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
    global_num_tokens_gpu: Optional[torch.Tensor] = None,
):
    """设置本次 forward 的 DP buffer 长度信息。"""
    _DpGatheredBufferWrapper.set_dp_buffer_len(
        global_dp_buffer_len,
        local_dp_buffer_len,
        dp_max_padding,
        global_num_tokens,
        global_num_tokens_gpu,
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


def set_local_dp_buffer_len(local_dp_buffer_len: int) -> None:
    _DpGatheredBufferWrapper.set_local_dp_buffer_len(local_dp_buffer_len)


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
    # Sticky within the thread: every ForwardBatch construction writes it,
    # graph runners force False around capture; readers are the EP
    # dispatchers on the same (single) forward thread.

    get_forward().set("is_extend_in_batch", is_extend_in_batch)


def get_is_extend_in_batch() -> bool:

    return get_forward().is_extend_in_batch


def is_dp_max_padding() -> bool:
    """当前是否采用"按最大长度 padding"（MAX_LEN 模式）。"""
    return _DpGatheredBufferWrapper.is_dp_max_padding()


def compute_dp_attention_world_info(
    enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size: int = 1
):
    """This rank's place in the attention topology, plus the widths it sits in.

    The widths come from `derive_attention_widths`; what this adds is the two
    ranks, which are per-process and so are not part of the stamped set.
    """
    attn_dp_size, attn_tp_size = derive_attention_widths(
        tp_size=tp_size,
        attn_cp_size=attn_cp_size,
        dp_size=dp_size,
        enable_dp_attention=enable_dp_attention,
    )
    attn_tp_rank = tp_rank % attn_tp_size

    if not enable_dp_attention:
        attn_dp_rank = 0
    else:
        # rank 布局为 (dp, cp, tp)，其中 tp 是变化最快的维度：
        # tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size + attn_tp_rank
        # 因此除以 (attn_tp_size * attn_cp_size) 即可得到 dp 维坐标。
        attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)

    return attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size


def initialize_dp_attention(
    server_args: ServerArgs,
    model_config: ModelConfig,
):
    """初始化 DP attention 的全局状态，需在分布式进程组建立之后调用一次。"""
    global _ATTN_DP_RANK, _ATTN_DP_SIZE
    dp = get_flags().dp
    dp.max_len_with_idle = (
        getattr(model_config.hf_config, "hybrid_override_pattern", None) is not None
    )
    enable_dp_attention = get_parallel().enable_dp_attention
    dp_size = get_parallel().dp_size
    attn_cp_size = get_parallel().attn_cp_size

    dp.enabled = enable_dp_attention

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    _, _, _ATTN_DP_RANK, _ATTN_DP_SIZE = compute_dp_attention_world_info(
        enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size
    )
    get_parallel().stamp_derived_widths(attn_dp_size=_ATTN_DP_SIZE)

    if get_exec().moe.elastic_ep_backend is not None and get_parallel().max_ep_size:
        _ATTN_DP_RANK = tp_rank + get_parallel().ep_join_rank_offset
        if server_args.is_ep_scale_joiner:
            dp.joiner_skip_all_gather = True

    # 记录 gather buffer 的静态元信息
    _DpGatheredBufferWrapper.set_metadata(
        hidden_size=model_config.hidden_size,
        dtype=model_config.dtype,
        device=torch.device(get_device().device),
    )


def is_dp_attention_enabled() -> bool:
    return get_flags().dp.enabled


def is_allocation_symmetric() -> bool:
    """当前的显存分配是否在各 rank 间对称（可用 symmetric memory）。

    未开启 DP attention 时天然对称；开启后只有 MAX_LEN 模式各 rank 长度一致。
    """
    return not is_dp_attention_enabled() or is_dp_max_padding()


def get_attention_dp_rank() -> int:
    """返回当前进程的全局 attention DP rank。"""
    assert _ATTN_DP_RANK is not None, "dp attention not initialized!"
    return _ATTN_DP_RANK


def get_attention_dp_size() -> int:
    """返回全局 attention DP 组的大小。"""
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"
    return _ATTN_DP_SIZE


@contextmanager
def disable_dp_size():
    """Run without DP attention until this scope ends.

    This is for draft workers of speculative decoding, which run the draft model
    at a different width from the target model's workers.

    The scope replaces both the module global that ``get_attention_dp_size()``
    reads and the derived width the runtime context answers with, so the two
    spellings of the name cannot disagree inside it.
    """
    global _ATTN_DP_SIZE
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"

    old_dp_size = _ATTN_DP_SIZE
    _ATTN_DP_SIZE = 1
    try:
        with get_parallel().override(attn_dp_size=1):
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


from sglang.kernels.ops.memory.memcpy_triton import memcpy_triton


# TODO: write c++ kernel for cpu
def memcpy_cpu(dst, src, dim, offset, sz, offset_src):
    assert dim == 0, "Only dim=0 supported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same trailing shape"

    total_rows_dst, total_rows_src = dst.shape[0], src.shape[0]
    dst_start, src_start = 0, 0

    if offset_src:
        # src[offset:] → dst[0:]
        src_start = offset
        dst_start = 0
    else:
        # src[0:] → dst[offset:]
        src_start = 0
        dst_start = offset

    dst_end = min(dst_start + sz, total_rows_dst)
    src_end = min(src_start + sz, total_rows_src)
    actual_sz = min(dst_end - dst_start, src_end - src_start)

    if actual_sz <= 0:
        return

    dst[dst_start : dst_start + actual_sz].copy_(src[src_start : src_start + actual_sz])


memcpy_func = memcpy_cpu if _is_cpu else memcpy_triton


def memcpy(dst, src, dim, offset, sz, offset_src):
    memcpy_func(dst, src, dim, offset, sz, offset_src)


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

    if local_tokens.shape[0] > 0 and (
        is_partial or get_attn_tensor_model_parallel_rank() == 0
    ):
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between global_tokens and local_tokens not allowed"

        memcpy(global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False)

    # Input IDs are in int 32. We should use inplace_all_reduce for local case because of custom all reduce.
    if world_dp_gather_enabled():
        torch.distributed.all_reduce(
            global_tokens,
            op=torch.distributed.ReduceOp.SUM,
            group=torch.distributed.group.WORLD,
        )
    else:
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
    use_world = world_dp_gather_enabled()

    if get_attn_tensor_model_parallel_world_size() == 1:
        if use_world:
            torch.distributed.all_gather_into_tensor(
                global_tokens,
                local_tokens,
                group=torch.distributed.group.WORLD,
            )
        else:
            get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
        return

    if not is_partial:
        if get_attn_tensor_model_parallel_rank() != 0:
            local_tokens.fill_(0)
    scattered_local_tokens = local_tokens.tensor_split(
        get_attn_tensor_model_parallel_world_size()
    )[get_attn_tensor_model_parallel_rank()]
    get_attn_tp_group().reduce_scatter_tensor(scattered_local_tokens, local_tokens)
    if use_world:
        torch.distributed.all_gather_into_tensor(
            global_tokens,
            scattered_local_tokens,
            group=torch.distributed.group.WORLD,
        )
    else:
        get_tp_group().all_gather_into_tensor(global_tokens, scattered_local_tokens)


# Variable-length DP-MoE gather (reference https://github.com/ROCm/ATOM/pull/930): instead of padding every
# rank to max_len (all_gather) or all-reducing a sum_len zero-buffer (all_reduce),
# gather exactly sum(per-rank tokens) via all_gatherv. Env-gated; only the simple
# tp_size==dp_size (attn_tp_size==1) case is supported for now (e.g. tp8dp8).
_USE_DP_GATHERV = get_bool_env_var("SGLANG_DP_USE_GATHERV")

_DP_GATHER_FP8_GROUP = 128
# Grow-only gathered fp8 payload / scales buffers, keyed by device.
_dp_gather_fp8_bufs: dict = {}


@functools.lru_cache(maxsize=1)
def _use_dp_gather_fp8() -> bool:
    return envs.SGLANG_ENABLE_DP_GATHER_FP8.get()


def _get_dp_gather_fp8_bufs(rows: int, hidden: int, device: torch.device):
    key = str(device)
    bufs = _dp_gather_fp8_bufs.get(key)
    if bufs is None or bufs[0].shape[0] < rows:
        bufs = (
            torch.empty((rows, hidden), dtype=torch.uint8, device=device),
            torch.empty(
                (rows, hidden // _DP_GATHER_FP8_GROUP),
                dtype=torch.float32,
                device=device,
            ),
        )
        _dp_gather_fp8_bufs[key] = bufs
    return bufs[0][:rows], bufs[1][:rows]


@triton.jit
def _dequant_per_token_group_fp8_kernel(
    q_ptr,
    s_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    NGROUPS: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    # HIDDEN may not be a multiple of BLOCK (e.g. DeepSeek 7168 vs BLOCK
    # 2048): the tail iteration must be masked or it reads/writes up to
    # BLOCK-1 elements past the row (cross-row corruption + OOB on the last
    # row).  HIDDEN is constexpr, so the mask folds away when it divides.
    for start in tl.static_range(0, HIDDEN, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HIDDEN
        qv = tl.load(q_ptr + row * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
        sv = tl.load(s_ptr + row * NGROUPS + offs // GROUP, mask=mask, other=0.0)
        tl.store(out_ptr + row * HIDDEN + offs, (qv * sv).to(tl.bfloat16), mask=mask)


@triton.jit
def _mask_dp_pad_topk_ids_kernel(
    topk_ids_ptr,
    counts_ptr,
    max_len,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    rank = row // max_len
    pos = row % max_len
    valid = pos < tl.load(counts_ptr + rank)
    if valid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(topk_ids_ptr + row * TOPK + offs, -1, mask=offs < TOPK)


def mask_dp_pad_moe_topk_ids(topk_ids: torch.Tensor) -> None:
    """Set MAX_LEN pad rows' (post-translation, local) topk_ids to -1 in place.

    Under dp-attention MAX_LEN padding the gathered MoE buffer is
    [dp_size * max_len, hidden] with rank r's real rows at
    [r*max_len, r*max_len + global_num_tokens[r]); the pad rows carry stale
    hidden values, run the router, and get dispatched into experts whose
    outputs are then discarded by the post-reorder scatter — pure wasted
    compute, and a masked-grouped-GEMM workspace blow-up when they collide
    on the same top-k.  -1 is the drop sentinel both the triton fused_moe
    (filter_expert) and the DeepGEMM EP preprocess honor; it must be applied
    AFTER the local_expert_mapping gather (a pre-translation -1 aliases to
    the mapping table's last entry).  Capture-safe: per-batch state is read
    only from the replay-updated global_num_tokens_gpu tensor.
    """
    counts = _DpGatheredBufferWrapper.get_dp_global_num_tokens_gpu()
    if counts is None:
        return
    max_len = _DpGatheredBufferWrapper.get_local_dp_buffer_len()
    rows, topk = topk_ids.shape
    if max_len <= 0 or rows != counts.shape[0] * max_len:
        # Layout mismatch (e.g. non-DP or logits-path caller): do nothing.
        return
    _mask_dp_pad_topk_ids_kernel[(rows,)](
        topk_ids,
        counts,
        max_len,
        TOPK=topk,
        BLOCK=triton.next_power_of_2(topk),
    )


def _dp_gather_via_all_gatherv_fp8(
    global_tokens: torch.Tensor,
    local_real: torch.Tensor,
    sizes: List[int],
):
    """fp8 wire format for the variable-length DP gather: quantize the local
    rows per-token-group (the SAME group-128 quantization the MoE expert GEMMs
    apply to their input downstream), gather payload (as uint8 — NCCL has no
    fp8 dtype; the gatherv leg is broadcast-only so a byte view is safe) and
    scales in two output-buffered gatherv calls, then dequantize into the
    bf16 global buffer.  Zero pad rows quantize to (q=0, s=eps) and so
    dequantize back to exact zeros — the MoE-tail invariant is preserved.
    The combine leg (reduce_scatterv) stays bf16: NCCL SUM cannot run on fp8."""
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    rows = global_tokens.shape[0]
    hidden = global_tokens.shape[-1]
    q, s = sglang_per_token_group_quant_fp8(
        local_real.contiguous(), _DP_GATHER_FP8_GROUP
    )
    gq, gs = _get_dp_gather_fp8_bufs(rows, hidden, global_tokens.device)
    tp_group = get_tp_group()
    tp_group.all_gatherv(q.view(torch.uint8), sizes=sizes, output=gq)
    tp_group.all_gatherv(s, sizes=sizes, output=gs)
    _dequant_per_token_group_fp8_kernel[(rows,)](
        gq.view(torch.float8_e4m3fn),
        gs,
        global_tokens,
        HIDDEN=hidden,
        NGROUPS=hidden // _DP_GATHER_FP8_GROUP,
        GROUP=_DP_GATHER_FP8_GROUP,
        BLOCK=2048,
    )


def is_dp_gatherv_active() -> bool:
    """Variable-length DP-MoE gather/scatter (all_gatherv + reduce_scatterv) is
    enabled and applicable to the CURRENT forward. Requires:
      - env SGLANG_DP_USE_GATHERV (default off),
      - supported layout (attn_tp_size==1, tp_size==dp_size),
      - SUM_LEN padding mode. The gatherv pair (all_gatherv + reduce_scatterv) is
        only valid under SUM_LEN; under MAX_LEN the buffer is equal-padded and the
        gather/combine use all_gather / (aiter) reduce_scatter instead. Reading the
        per-forward padding via _DpGatheredBufferWrapper.is_dp_max_padding() (set by
        set_dp_buffer_len) keeps callers that lack a ForwardBatch (e.g.
        dp_reduce_scatter_tensor) consistent."""
    return (
        _USE_DP_GATHERV
        and not world_dp_gather_enabled()
        and get_attn_tensor_model_parallel_world_size() == 1
        and get_tensor_model_parallel_world_size() == get_attention_dp_size()
        and not _DpGatheredBufferWrapper.is_dp_max_padding()
    )


def _dp_gatherv_sizes(forward_batch) -> Optional[List[int]]:
    """Per-rank CPU token counts for the buffer being gathered. The MoE gather
    passes a ForwardBatch (global_num_tokens_cpu); the logits gather passes a
    LogitsMetadata (global_num_tokens_for_logprob_cpu). Return the sizes that
    match the LOCAL tensor for this context, or None to fall back."""
    sizes = getattr(forward_batch, "global_num_tokens_for_logprob_cpu", None)
    if sizes is None:
        sizes = getattr(forward_batch, "global_num_tokens_cpu", None)
    if sizes is None:
        return None
    try:
        return [int(x) for x in sizes]
    except (TypeError, ValueError):
        return None


def _dp_gather_via_all_gatherv(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
    sizes: List[int],
):
    # attn_tp_size == 1: each DP rank contributes exactly `sizes[rank]` rows.
    # CRITICAL: the MoE downstream runs on the WHOLE `global_tokens` buffer
    # (M = global_tokens.shape[0]), so the gather MUST fill every row. We pad
    # each rank's local tensor up to sizes[rank] with zeros (matching the
    # buffer's reserved per-rank slot) so sum(sizes) == buffer rows and there
    # is no uninitialized tail for the MoE to read.
    rank = get_attention_dp_rank()
    local_rows = sizes[rank]
    if local_tokens.shape[0] == local_rows:
        local_real = local_tokens
    elif local_tokens.shape[0] > local_rows:
        local_real = local_tokens[:local_rows]
    else:
        local_real = local_tokens.new_zeros((local_rows, *local_tokens.shape[1:]))
        local_real[: local_tokens.shape[0]].copy_(local_tokens)
    # sum(sizes) == global_tokens.shape[0] is guaranteed by the caller (else it
    # falls back to all_reduce). Pass global_tokens as the NCCL output buffer so
    # the gather writes directly into it -- avoids the previous extra full-buffer
    # torch.cat + copy_ (two ~sum(sizes)*hidden DtoD copies, ~700us/layer at c512).
    # NOTE: the fp8 branch condition must be identical on EVERY DP rank (all
    # ranks must issue the same NCCL op sequence) — env/dtype/hidden are
    # rank-uniform; never gate on per-rank state like forward_mode (ranks can
    # be extend/idle-mixed within one global forward).  Prefill-only is
    # already structural: the gatherv path runs only under SUM_LEN padding,
    # which decode-only steps and CUDA-graph capture never select.
    if (
        _use_dp_gather_fp8()
        and global_tokens.dtype == torch.bfloat16
        and global_tokens.shape[-1] % _DP_GATHER_FP8_GROUP == 0
    ):
        _dp_gather_via_all_gatherv_fp8(global_tokens, local_real, sizes)
        return
    get_tp_group().all_gatherv(local_real, sizes=sizes, output=global_tokens)


def _dp_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    if (
        is_dp_gatherv_active()
        and forward_batch.dp_padding_mode is not None
        and not forward_batch.dp_padding_mode.is_max_len()
    ):
        # The gatherv per-rank sizes MUST sum to the pre-allocated global buffer
        # (the MoE runs on the whole buffer, so any unfilled tail = garbage).
        # The buffer was sized from the ceil_align'd global_num_tokens stored via
        # set_dp_buffer_len (forward_batch_info), so the authoritative sizes are
        # get_dp_global_num_tokens() — the SAME source the reduce_scatterv combine
        # uses (symmetric). _dp_gatherv_sizes() reads the raw (un-aligned, and for
        # the MoE-gather context the logprob-token) counts, which do NOT match the
        # buffer for prefill steps -> would force an all_reduce fallback.
        # Prefer the buffer-aligned sizes; fall back to the per-batch sizes only
        # if they happen to match (e.g. the logits gather path).
        _gatherv_sizes = get_dp_global_num_tokens()
        if _gatherv_sizes is None or sum(_gatherv_sizes) != global_tokens.shape[0]:
            _gatherv_sizes = _dp_gatherv_sizes(forward_batch)
        if _gatherv_sizes is not None and sum(_gatherv_sizes) == global_tokens.shape[0]:
            _dp_gather_via_all_gatherv(
                global_tokens, local_tokens, forward_batch, is_partial, _gatherv_sizes
            )
            return
    if (
        forward_batch.dp_padding_mode is not None
        and forward_batch.dp_padding_mode.is_max_len()
    ):
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

        memcpy(local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True)


def dp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    if is_dp_gatherv_active():
        # Variable-length combine matching all_gatherv dispatch: scatter the
        # global (sum_len) tensor back to per-rank token counts. Fall through to
        # the default reduce-scatter path if per-rank sizes are unavailable.
        sizes = get_dp_global_num_tokens()
        if sizes is not None:
            get_tp_group().reduce_scatterv(input, output=output, sizes=sizes)
            return
    if get_tensor_model_parallel_world_size() == get_attention_dp_size():
        get_tp_group().reduce_scatter_tensor(output, input)
    else:
        # 取出本全局 TP rank 应负责的细分片
        scattered_local_tokens = input.tensor_split(
            get_tensor_model_parallel_world_size()
        )[get_tensor_model_parallel_rank()]
        # 第一步：全局规约并细分
        get_tp_group().reduce_scatter_tensor(scattered_local_tokens, input)
        get_attn_tp_group().all_gather_into_tensor(output, scattered_local_tokens)


# ---------------------------------------------------------------------------
# Two-batch-overlap (non-EP / DP TP-MoE) async gather + combine.
#
# The DP TP-MoE path (deepseek_v4) gathers local hidden -> a global buffer
# before the experts and reduce-scatters back after. For TBO we run those two
# collectives on a single shared comm stream (mirroring the mori dispatcher's
# _comm_stream) and return a CUDA event, so the op engine can yield and let the
# OTHER ubatch's attn+MoE compute run on the compute stream while this ubatch's
# gather/combine proceeds on the comm stream. Both ubatches share ONE comm
# stream -> their collectives serialize in-order (no concurrent-collective
# deadlock on the RCCL communicator), each overlapping the other's compute.
# ---------------------------------------------------------------------------
def get_dp_tbo_comm_stream() -> torch.cuda.Stream:

    return get_stream("dp_tbo_comm")


# Persistent reusable CUDA events for non-EP DP TBO, keyed by (kind, subbatch).
# CRITICAL: do NOT create a fresh event per gather/combine -- that is ~244 new
# torch.cuda.Event per forward (61 layers x 2 ubatches x 2), and the HSA signal
# pool is exhausted after a few hundred forwards -> HSA_STATUS_ERROR_OUT_OF_RESOURCES
# ("...create internal OS-specific events"). Reuse one event per (kind, subbatch)
# and just re-record it (mirrors the mori CommStreamPool event reuse).
def _tbo_event(key) -> torch.cuda.Event:

    pool = get_resources().tbo_event_pool
    ev = pool.get(key)
    if ev is None:
        ev = torch.cuda.Event()
        pool[key] = ev
    return ev


def dp_gather_partial_async(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    event_key=("gather", 0),
) -> torch.cuda.Event:
    """Launch `dp_gather_partial` (all_gatherv) on the shared DP TBO comm stream;
    re-record + return a PERSISTENT event (keyed by `event_key`) that fires when
    the gather completes. Caller yields, then `compute_stream.wait_event(ev)`
    before reading `global_tokens`."""
    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()
    # Keep buffers alive across streams (caching allocator).
    local_tokens.record_stream(comm)
    global_tokens.record_stream(comm)
    ev = _tbo_event(event_key)
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)  # inputs were produced on the compute stream
        dp_gather_partial(global_tokens, local_tokens, forward_batch)
        ev.record(comm)
    return ev


# Persistent grow-only buffers for non-EP DP TBO, keyed by (kind, tbo_subbatch).
# Reused across ALL layers (and forwards) so the caching allocator does not churn
# a fresh per-layer `torch.empty` for the 8x DP-gather / combine buffers. That
# churn (different sizes per forward x 2 ubatches x 61 layers, kept alive by the
# comm-stream record_stream) ballooned `reserved` to ~270GB and tripped
# HSA_STATUS_ERROR_OUT_OF_RESOURCES at large prefill chunks, even though the live
# (allocated) working set was only ~10GB.
_TBO_PERSIST_BUF: dict = {}


def get_tbo_persistent_buffer(
    key, rows: int, hidden: int, dtype: torch.dtype, device
) -> torch.Tensor:
    """Return a [rows, hidden] view of a grow-only persistent buffer for `key`.
    Reallocates only when the request exceeds the cached capacity / changes
    dtype|hidden. Caller must treat the returned view as scratch (overwritten)."""
    buf = _TBO_PERSIST_BUF.get(key)
    cap = 0 if buf is None else buf.shape[0]
    if buf is None or rows > cap or buf.shape[1] != hidden or buf.dtype != dtype:
        new_rows = max(rows, cap)
        buf = torch.empty((new_rows, hidden), dtype=dtype, device=device)
        _TBO_PERSIST_BUF[key] = buf
    return buf[:rows]


def dp_reduce_scatterv_async(
    output_local: torch.Tensor,
    global_tokens: torch.Tensor,
    sizes: List[int],
    event_key=("combine", 0),
) -> torch.cuda.Event:
    """Launch the variable-length reduce_scatterv (combine) on the shared DP TBO
    comm stream; re-record + return a PERSISTENT event (keyed by `event_key`).
    Matches the gatherv (SUM_LEN) path."""
    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()
    ev = _tbo_event(event_key)
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)
        get_tp_group().reduce_scatterv(global_tokens, output=output_local, sizes=sizes)
        ev.record(comm)
    return ev


def attn_tp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_tp_group().reduce_scatter_tensor(output, input)


def attn_cp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_cp_group().reduce_scatter_tensor(output, input)


def attn_tp_all_reduce(input: torch.Tensor):
    return get_attn_tp_group().all_reduce(input)


def attn_tp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_tp_group().all_gather_into_tensor(output, input)


def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_cp_group().all_gather_into_tensor(output, input)


def attn_cp_overlap_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_cp_overlap_group().all_gather_into_tensor(output, input)


def attn_cp_overlap_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_cp_overlap_group().reduce_scatter_tensor(output, input)


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
    """True when moe_dp_size < attn_cp_size, requiring allgather across CP ranks before MoE.

    In that configuration ``initialize_model_parallel`` aliases ``_MOE_DP`` to
    ``_ATTN_CP``, so the two groups report equal widths.
    """
    return get_parallel().attn_cp_size > get_parallel().moe_dp_size


def moe_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    """在 MOE_DP（含 CP 伙伴）组内执行 all-gather。"""
    return _get_moe_dp_group().all_gather_into_tensor(output, input)


def attn_tp_all_gather(output_list: List[torch.Tensor], input: torch.Tensor):
    return get_attn_tp_group().all_gather(input, output_tensor_list=output_list)
