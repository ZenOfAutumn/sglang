from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module

DEFAULT_BLOCK_QUOTA = 2


@cache_once
def _jit_hicache_module(*, element_size: int, unroll: int, block_quota: int) -> Module:
    args = make_cpp_args(
        element_size,
        unroll,
        block_quota,
        1024,  # num_threads, can be tuned for performance
    )
    return load_jit(
        "hicache",
        *args,
        cuda_files=["hicache.cuh"],
        cuda_wrappers=[
            ("launch_one", f"&HiCacheKernel<{args}>::run_one"),
            ("launch_all", f"&HiCacheKernel<{args}>::run_all"),
            ("launch_one_mla", f"&HiCacheKernel<{args}>::run_one_mla"),
            ("launch_all_mla", f"&HiCacheKernel<{args}>::run_all_mla"),
        ],
    )


def can_use_hicache_jit_kernel(
    *,
    element_size: int,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> bool:
    logger = logging.getLogger(__name__)
    if element_size % 128 != 0:
        logger.warning(f"Unsupported {element_size = } for JIT HiCache kernel")
        return False
    try:
        unroll = unroll or _default_unroll(element_size)
        block_quota = block_quota or DEFAULT_BLOCK_QUOTA
        _jit_hicache_module(
            element_size=element_size,
            unroll=unroll,
            block_quota=block_quota,
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT HiCache kernel: {e}")
        return False


def _default_unroll(element_size: int) -> int:
    if element_size <= 512:
        return 4

    if element_size <= 1024:
        return 2

    # fallback: no unroll
    return 1


@debug_kernel_api
def transfer_hicache_one_layer(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    element_dim: int | None = None,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    element_dim = element_dim or k_cache_dst.size(-1)
    k_cache_src = k_cache_src.view(-1, element_dim)
    v_cache_src = v_cache_src.view(-1, element_dim)
    k_cache_dst = k_cache_dst.view(-1, element_dim)
    v_cache_dst = v_cache_dst.view(-1, element_dim)
    element_size = element_dim * k_cache_dst.element_size()
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_one(
        k_cache_dst,
        v_cache_dst,
        indices_dst,
        k_cache_src,
        v_cache_src,
        indices_src,
    )


@debug_kernel_api
def transfer_hicache_all_layer(
    k_ptr_dst: torch.Tensor,
    v_ptr_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_ptr_src: torch.Tensor,
    v_ptr_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    kv_cache_src_stride_bytes: int,
    kv_cache_dst_stride_bytes: int,
    element_size: int | None = None,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    """一次 kernel 调用完成「所有层」KV 缓存的 gather/scatter 式搬运（HiCache L1<->L2）。

    与逐层版 transfer_hicache_one_layer 不同，本函数把「全部层」的搬运合并到单次 kernel
    启动里，从而摊薄 kernel launch 开销，是分层缓存换入/换出（H2D / D2H）的高性能路径。

    多层是如何寻址的：
      这里的 k_ptr_dst / v_ptr_dst / k_ptr_src / v_ptr_src 不是普通张量数据，而是「每层
      KV 缓存基地址的指针数组」（长度 = 层数）。kernel 内部先按层取出基址，再用
      *_stride_bytes 计算某个槽位在该层缓冲中的字节偏移，据此在源/目的间拷贝。

    按索引搬运（离散槽位）：
      indices_src / indices_dst 分别是源、目的一侧的槽位（页/ token 槽）索引，二者一一对应；
      每一层都按同一组索引做 gather（从 src[indices_src]）/ scatter（到 dst[indices_dst]），
      因此只搬这些离散槽位，而非整块缓冲。

    参数说明：
      k_ptr_dst / v_ptr_dst      : 目的侧各层 K / V 缓存基地址指针数组
      indices_dst                : 目的侧槽位索引
      k_ptr_src / v_ptr_src      : 源侧各层 K / V 缓存基地址指针数组
      indices_src                : 源侧槽位索引
      kv_cache_src_stride_bytes  : 源侧「每个槽位」的字节跨距（用于定位偏移）
      kv_cache_dst_stride_bytes  : 目的侧「每个槽位」的字节跨距
      element_size               : 单次拷贝的元素字节数；None 时要求源/目的跨距相等并取该跨距
      unroll                     : 循环展开度（性能调优；None 时按 element_size 自动选择）
      block_quota                : 每个 SM 上的 block 配额（降低对其他 kernel 的干扰）
    """
    if element_size is None:  # assume both contiguous
        # 未显式给定元素字节数时，假定源/目的都是连续布局：两侧跨距必须相等，
        # 直接用该跨距作为单次拷贝的字节数。
        assert kv_cache_dst_stride_bytes == kv_cache_src_stride_bytes
        element_size = kv_cache_dst_stride_bytes

    # 未指定时用默认 block 配额；unroll 按 element_size 自动选择（见 _default_unroll）。
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    # 按 (element_size, unroll, block_quota) 编译/取回缓存的 JIT kernel 模块（cache_once 去重编译）。
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    # 启动「全层」搬运 kernel：传入各层基址指针数组、两侧索引以及两侧的槽位字节跨距。
    module.launch_all(
        k_ptr_dst,
        v_ptr_dst,
        indices_dst,
        k_ptr_src,
        v_ptr_src,
        indices_src,
        kv_cache_src_stride_bytes,
        kv_cache_dst_stride_bytes,
    )


def transfer_hicache_one_layer_mla(
    cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    element_dim: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    element_dim = element_dim or cache_dst.size(-1)
    cache_src = cache_src.view(-1, element_dim)
    cache_dst = cache_dst.view(-1, element_dim)
    element_size = element_dim * cache_dst.element_size()
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_one_mla(
        cache_dst,
        indices_dst,
        cache_src,
        indices_src,
    )


def transfer_hicache_all_layer_mla(
    ptr_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    ptr_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    cache_src_stride_bytes: int,
    cache_dst_stride_bytes: int,
    element_size: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    if element_size is None:
        assert cache_dst_stride_bytes == cache_src_stride_bytes
        element_size = cache_dst_stride_bytes

    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_all_mla(
        ptr_dst,
        indices_dst,
        ptr_src,
        indices_src,
        cache_src_stride_bytes,
        cache_dst_stride_bytes,
    )
