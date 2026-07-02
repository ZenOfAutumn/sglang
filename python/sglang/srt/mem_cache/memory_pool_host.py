# ============================================================================
# HiCache 主机侧（Host / CPU 内存）KV 缓存池实现。
#
# SGLang 的 HiCache（分层缓存）把 KV 缓存分成两层：
#   - device 侧：显存（GPU）中的 KV 池，容量小、速度快，直接参与注意力计算；
#   - host 侧：主机内存（CPU RAM）中的 KV 池，容量大、速度慢，作为显存的“二级缓存”。
# 当显存放不下时，把暂时用不到的 KV 页从显存换出（backup）到主机内存；需要时再换入
# （load）回显存。本文件定义的就是各种「主机侧 KV 池」，负责在 CPU 内存里分配缓冲区、
# 记录空闲槽位，并实现与显存之间双向搬运（transfer）的具体逻辑。
#
# 主要类层次：
#   HostKVCache（抽象基类，定义分配/释放/搬运接口）
#     ├─ MHATokenToKVPoolHost           标准多头注意力（MHA）的主机池，K/V 分开存
#     │    └─ AsymmetricMHATokenToKVPoolHost  K、V 头数/维度不对称的变体
#     ├─ MLATokenToKVPoolHost           DeepSeek MLA（低秩压缩 KV）的主机池
#     ├─ MambaPoolHost                  Mamba/线性注意力循环状态的主机池
#     ├─ DeepSeekV4PagedHostPool        DeepSeek V4 稀疏注意力的分页主机池
#     ├─ DeepSeekV4StateHostPool        DeepSeek V4 状态主机池
#     └─ DSAIndexerPoolHost             DSA（DeepSeek 稀疏注意力）索引器主机池
# 另有 HostPoolGroup（把多个主机池聚合成一组统一管理）、LogicalHostPool（逻辑视图）等辅助类。
# ============================================================================
from __future__ import annotations

import abc
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import PoolName

import numpy as np
import psutil
import torch

from sglang.jit_kernel.hicache import (
    can_use_hicache_jit_kernel,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer as jit_transfer_hicache_all_layer,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer_mla as jit_transfer_hicache_all_layer_mla,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_one_layer as jit_transfer_hicache_one_layer,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_one_layer_mla as jit_transfer_hicache_one_layer_mla,
)
from sglang.jit_kernel.hisparse import transfer_cache_dsv4_mla
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    KVCache,
    MambaPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.mmap_allocator import alloc_mmap
from sglang.srt.utils import is_cuda, is_hip, is_mps, is_npu, is_xpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_mps = is_mps()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer,
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_lf_pf,
        transfer_kv_all_layer_lf_ph,
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
        transfer_kv_per_layer_pf_lf,
        transfer_kv_per_layer_ph_lf,
    )
if _is_npu:
    from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_kv_dim_exchange

logger = logging.getLogger(__name__)

# 在按可用内存自动计算 HiCache 主机池大小时，需要预留出来、不占用的主机内存
# （留给操作系统和其他进程使用），默认 10 GiB。
HICACHE_HOST_MEMORY_RESERVE_BYTES: int = 10 * (1024**3)


def synchronized(func):
    """方法级同步装饰器：进入方法前先获取 self.lock，退出后自动释放。

    主机池的分配/释放会被多个线程（如后台的换入换出线程与主调度线程）并发调用，
    这里用一把可重入/互斥锁保护空闲列表等共享状态，避免竞争。被装饰的方法所在的类
    必须提供 self.lock 属性。
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class HostTensorAllocator:
    """主机张量分配器：用 mmap（内存映射）在 CPU 内存上分配缓冲区。

    相比普通的 torch.empty，mmap 分配的匿名内存可以按需缺页调页，且便于配合
    大页/共享内存等机制，适合 HiCache 这种大容量的主机缓冲区场景。
    """

    def __init__(self):
        """初始化分配器；dtype/dims 在首次 allocate 时记录，便于后续复用元信息。"""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        # 该分配器只服务于 CPU 主机内存；传入非 cpu 设备属于调用方错误，直接断言拦截。
        assert (
            device == "cpu"
        ), f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        return alloc_mmap(dims, dtype)


class HiSparseHostPoolMixin:
    """稀疏注意力（HiSparse）主机池共用的「按页分配」混入类。

    稀疏注意力场景下，一个请求的 KV 在主机内存里是按页（page）连续申请、
    但按 token 粒度使用的。该 Mixin 提供页对齐、按页扩容、以及从
    「请求 -> 主机槽位」映射表中取出已分配槽位的通用方法，供多个稀疏主机池复用。
    使用方需自身具备 self.page_size 属性及 self.alloc() 方法。
    """

    def _round_up_to_page_size(self, size: int) -> int:
        # 把 size 向上取整到 page_size 的整数倍（不足一页也占满一页）。
        return (size + self.page_size - 1) // self.page_size * self.page_size

    def alloc_page(self, num_pages: int) -> Optional[torch.Tensor]:
        # 申请 num_pages 页，返回 token 粒度的槽位索引；容量不足时返回 None。
        return self.alloc(num_pages * self.page_size)

    def alloc_paged_token_slots(
        self,
        req_to_host_pool: torch.Tensor,
        req_to_host_pool_allocated_len: torch.Tensor,
        req_pool_idx: int,
        start_pos: int,
        num_tokens: int,
    ) -> torch.Tensor:
        """按页为某个请求分配主机槽位，并返回 token 粒度的槽位索引。

        参数：
            req_to_host_pool: 二维映射表 [请求槽位, 序列位置] -> 主机池槽位索引。
            req_to_host_pool_allocated_len: 每个请求已分配到的长度（按页对齐）。
            req_pool_idx: 当前请求在映射表中的行号。
            start_pos: 本次要写入的序列起始位置。
            num_tokens: 本次要写入的 token 数。
        逻辑：若 [start_pos, start_pos+num_tokens) 超出已分配长度，则按页扩容并把
        新申请的主机槽位回填进映射表，最后切片返回本段对应的槽位索引。
        """
        device = req_to_host_pool.device
        if num_tokens <= 0:
            # 没有要写入的 token，返回空张量。
            return torch.empty((0,), dtype=torch.int64, device=device)

        allocated_len = int(req_to_host_pool_allocated_len[req_pool_idx])
        end_pos = start_pos + num_tokens
        page_end = self._round_up_to_page_size(end_pos)  # 本次结束位置对齐到整页
        assert start_pos <= allocated_len  # 起始位置必须落在已分配范围内，否则出现空洞

        if page_end > allocated_len:
            # 需要新增的页数 = （对齐后的结束位置 - 已分配长度）/ 页大小。
            num_new_pages = (page_end - allocated_len) // self.page_size
            host_locs = self.alloc_page(num_new_pages)
            if host_locs is None:
                logger.error(
                    "HiSparse: host mem pool alloc failed for %d host pages "
                    "(req_pool_idx=%d, start_pos=%d, num_tokens=%d)",
                    num_new_pages,
                    req_pool_idx,
                    start_pos,
                    num_tokens,
                )
                raise RuntimeError(
                    f"HiSparse host mem pool alloc failed for {num_new_pages} pages"
                )

            # 把新申请的槽位索引写回映射表的 [allocated_len, page_end) 区间，
            # 并推进该请求的已分配长度。
            req_to_host_pool[req_pool_idx, allocated_len:page_end] = host_locs.to(
                device=device, non_blocking=True
            )
            req_to_host_pool_allocated_len[req_pool_idx] = page_end

        # 返回本次 [start_pos, end_pos) 对应的 token 粒度主机槽位索引。
        return req_to_host_pool[req_pool_idx, start_pos:end_pos]

    def allocated_host_indices(
        self,
        req_to_host_pool: torch.Tensor,
        req_pool_idx: int,
        allocated_len: int,
    ) -> torch.Tensor:
        """取出某请求当前已分配的全部有效主机槽位索引（过滤掉未分配占位的 -1）。"""
        allocated_len = int(allocated_len)
        # 取已分配长度对齐到整页后的长度，但不超过映射表实际列数。
        host_len = min(
            self._round_up_to_page_size(allocated_len),
            req_to_host_pool.shape[1],
        )
        host_indices = req_to_host_pool[req_pool_idx, :host_len]
        # 映射表中未分配的位置以负值占位，这里只保留 >= 0 的有效索引。
        return host_indices[host_indices >= 0]


def get_allocator_from_storage(allocator_type):
    """根据存储后端类型返回对应的主机张量分配器。

    当使用 Mooncake 作为分布式 KV 存储后端时，需要它专用的分配器
    （分配的内存可被 Mooncake 传输引擎注册/RDMA 访问）；其余情况回退到默认分配器。
    Mooncake 版本过低而导入失败时，打印告警并降级为默认分配器。
    """
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    else:
        return HostTensorAllocator()


def _cuda_host_register(buffer: torch.Tensor) -> None:
    """把一块主机内存注册为 CUDA 页锁定（pinned）内存。

    页锁定内存不会被操作系统换页，能让 GPU 用 DMA 做异步、高带宽的主机<->显存拷贝。
    调用 cudaHostRegister 把已有缓冲区就地锁定；失败时抛异常——因为若不锁定，
    后续的设备间异步传输可能悄无声息地读到过期数据（而非报错），必须硬性拦截。
    """
    cudart = torch.cuda.cudart()
    n_bytes = buffer.numel() * buffer.element_size()  # 缓冲区总字节数
    rc = cudart.cudaHostRegister(buffer.data_ptr(), n_bytes, 0)
    if int(rc) != 0:
        raise RuntimeError(
            f"cudaHostRegister failed (rc={int(rc)}, "
            f"{cudart.cudaGetErrorString(rc)}) for ptr={buffer.data_ptr():#x} "
            f"size={n_bytes}; host buffer is not pinned and device transfers "
            f"may silently return stale data."
        )


def alloc_with_host_register(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
) -> torch.Tensor:
    """分配张量，并用 cudaHostRegister 显式注册为页锁定内存。

    仅当 pin_memory=True 时才执行注册。这条路径用于自定义分配器（如 mmap / Mooncake）
    分配出的内存——它们不走 PyTorch 的 pin_memory 标志，需要手动注册。
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        _cuda_host_register(buffer)
    return buffer


def alloc_with_pin_memory(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """直接用 PyTorch 内建的 pin_memory 标志分配页锁定张量。

    适用于 NPU / MUSA 等平台：由框架自身管理页锁定，无需手动 cudaHostRegister。
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


# 按设备类型选择分配函数：默认走 cudaHostRegister 路径（CUDA/HIP），
# NPU、MUSA 则改用 PyTorch 内建 pin_memory。defaultdict 保证未列出的设备也有默认实现。
ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)


class HostKVCache(abc.ABC):
    """所有主机侧 KV 缓存池的抽象基类。

    职责：
      1. 在主机内存里分配一大块 KV 缓冲区（大小按 host_size 或相对 device 池的比例确定）；
      2. 用一个「空闲槽位列表」以页对齐的粒度管理分配/释放；
      3. 定义与 device 池之间双向搬运 KV 的抽象接口（换入 load、换出 backup）
         以及以「数据页」为单位读写的接口（供落盘/远端存储等 storage 后端使用）。
    具体的张量布局、K/V 是否分开、搬运用哪个 kernel，由各子类实现。
    """

    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
        allocator_type: str = "default",
    ):
        """
        参数：
            device_pool: 对应的 device（显存）KV 池，主机池是它的二级缓存。
            host_to_device_ratio: 主机池相对显存池的容量倍数（host_size<=0 时按此推算）。
            host_size: 主机池目标大小（单位 GB）；>0 时优先使用它换算槽位数。
            page_size: 页大小（每页 token 数），分配/释放以页为最小粒度。
            layout: 张量布局（如 layer_first / page_first 等），影响搬运 kernel 选择。
            pin_memory: 是否把主机缓冲区锁页，以支持异步高带宽 H2D/D2H 拷贝。
            device: 主机缓冲区所在设备，通常为 "cpu"。
            allocator_type: 分配器类型（"default" 或 "mooncake" 等）。
        """
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)

        self.dtype = device_pool.store_dtype  # 与显存池保持一致的存储 dtype
        self.size_per_token = self.get_size_per_token()  # 每个 token 占用的字节数
        if host_size > 0:
            # 指定了目标 GB 数：容量（token 数）= 目标字节数 / 每 token 字节数。
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            # 否则按相对显存池的比例推算。
            self.size = int(device_pool.size * host_to_device_ratio)
        # 把主机池容量向上对齐到页大小（多留一页，保证是整页的整数倍）。
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer  # 本进程负责的起始层（PP 切分）
        self.end_layer = device_pool.end_layer  # 本进程负责的结束层

        # 现有协议要求主机池必须大于显存池（否则无法充当二级缓存容纳换出的数据）。
        assert (
            self.size > device_pool.size
        ), "The host memory should be larger than the device memory with the current protocol"

        # 校验主机可用内存是否足够，避免分配时 OOM 拖垮整个进程。
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        self.kv_buffer = self.init_kv_buffer()  # 真正分配主机 KV 缓冲区（子类实现布局）

        # 保护内存分配与状态迁移的可重入锁；配合 @synchronized 装饰器使用。
        self.lock = threading.RLock()
        self.clear()

    @abc.abstractmethod
    def get_size_per_token(self):
        """返回每个 token 在本池中占用的字节数（用于换算容量）。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def init_kv_buffer(self):
        """按子类的张量布局分配并返回主机 KV 缓冲区。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        """把指定层的 KV 数据从主机池「换入」到显存池（H2D）。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """把所有层的 KV 数据从显存池「换出/备份」到主机池（D2H）。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """从主机池取出一个（默认扁平化的）数据页，供 storage 后端落盘/上传。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """返回一个全零的占位数据页，用于预取占位或初始化空页。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """把一个扁平数据页写回主机池的指定位置（从 storage 后端加载时使用）。"""
        raise NotImplementedError()

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        """判断每页的步长（stride）是否为 page_size_bytes 的整数倍。

        对齐后才能对基于文件的 NIXL 后端启用 O_DIRECT 直接 I/O。子类应按自身布局
        重写该方法给出精确判断；基类默认打印告警并返回 False（安全的保守值），
        此时会退化为普通拷贝模式。
        """
        logger.warning(
            "%s does not implement is_stride_page_aligned(); assuming not aligned. "
            "O_DIRECT with a file-based NIXL backend will fall back to copy mode for this pool.",
            type(self).__name__,
        )
        return False

    @synchronized
    def clear(self):
        # 重置内存状态与空闲槽位：mem_state 记录每个槽位的状态，free_slots 为可分配槽位列表。
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        # 初始时所有槽位（0 .. size-1）均空闲。
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        """当前可分配的槽位数量。"""
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """分配 need_size 个连续槽位（必须是页大小整数倍）；不足则返回 None。"""
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size > self.available_size():
            return None

        # 从空闲列表头部取出 need_size 个槽位，并把它们从空闲列表中移除。
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        """归还槽位：把 indices 重新拼回空闲列表，返回归还的数量。"""
        self.free_slots = torch.cat([self.free_slots, indices.cpu()])
        return len(indices)


class MHATokenToKVPoolHost(HostKVCache):
    """标准多头注意力（MHA）的主机侧 KV 池。

    KV 缓冲区形状约定：第一维为 2，分别存放 K 和 V（即 kv_buffer[0] 为 K，[1] 为 V）。
    支持多种张量布局（layout）以适配不同的搬运 kernel 与落盘方式：
      - layer_first:        [2, layer, token, head, head_dim]，按层优先。
      - page_first:         [2, token, layer, head, head_dim]，按页/ token 优先。
      - page_first_direct:  以 page 为最外层，便于对齐的直接 I/O。
      - page_head:          head 维前置的分头布局。
    换入/换出既可以走定制 kernel（io_backend="kernel"），也可以走直接拷贝（"direct"）。
    """

    device_pool: MHATokenToKVPool

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
        )
        # 单个 token 的元素维度 = 头数 * 每头维度，用于判断能否使用 JIT 搬运 kernel。
        self.element_dim = self.device_pool.head_num * self.device_pool.head_dim
        self.can_use_jit = _is_cuda and can_use_hicache_jit_kernel(
            element_size=self.element_dim * self.dtype.itemsize
        )

        # 预先为每一层构造 K/V 缓冲区的「按层视图」，并缓存各视图的数据指针，
        # 供 kernel 直接按指针访问，避免每次搬运都重新计算偏移。
        if self.layout == "page_first":
            # page_first 布局下缓冲区是 [token/page, layer, ...]，
            # 转置成 [layer, token/page, ...] 得到按层视图——transpose 只换步长不拷贝数据。
            k_transposed = self.k_buffer.transpose(0, 1)
            v_transposed = self.v_buffer.transpose(0, 1)
            self.k_data_refs = [k_transposed[i] for i in range(self.layer_num)]
            self.v_data_refs = [v_transposed[i] for i in range(self.layer_num)]
        else:
            # 其余布局第 0 维即为层维，直接索引即可。
            self.k_data_refs = [self.k_buffer[i] for i in range(self.layer_num)]
            self.v_data_refs = [self.v_buffer[i] for i in range(self.layer_num)]
        # 把各层视图的首地址收集成张量（放到 device 上），供 kernel 按层索引指针。
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def get_size_per_token(self):
        # 顺带从 device 池抄下头数/头维/层数等元信息缓存到 self 上。
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num
        # 每 token 字节数 = head_dim * head_num * layer_num * dtype字节 * 2（K 和 V 各一份）。
        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def get_ksize_per_token(self):
        # 单独 K 的每 token 字节数：总量的一半（K、V 对称）。
        return self.get_size_per_token() // 2

    def init_kv_buffer(self):
        # 根据布局确定缓冲区的形状 dims，第 0 维恒为 2（K/V）。
        if self.layout == "layer_first":
            dims = (2, self.layer_num, self.size, self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (2, self.size, self.layer_num, self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (
                2,
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        elif self.layout == "page_head":
            dims = (
                2,
                self.page_num,
                self.head_num,
                self.page_size,
                self.layer_num,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        # 单 token 在单层内的字节步长，以及跨全部层的步长，供对齐/搬运计算复用。
        self.token_stride_size = self.head_num * self.head_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        # 依据设备类型选择分配函数（cudaHostRegister 或 pin_memory），分配主机缓冲区。
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        return buffer

    @property
    def k_buffer(self):
        # K 缓冲区视图：kv_buffer 的第 0 个分量。
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        # V 缓冲区视图：kv_buffer 的第 1 个分量。
        return self.kv_buffer[1]

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        """把单层的 KV 从主机池换入显存池（H2D）。

        搬运按 (io_backend, layout) 两级分派到不同实现：
          - io_backend="kernel":       用 sgl_kernel 的散列搬运 kernel；CUDA 上若满足条件
                                        再优先走更快的 JIT kernel（self.can_use_jit）。
          - io_backend="direct":       直接按页拷贝，不经专用 kernel。
          - io_backend="kernel_ascend": 昇腾 NPU 专用路径。
        host_indices / device_indices 分别是源（主机）与目的（显存）的槽位索引，
        搬运只在这些离散槽位间进行（gather/scatter 式）。
        """
        if io_backend == "kernel":
            if self.layout == "layer_first":
                # layer_first：源缓冲区第 0 维即层，按 layer_id 取出单层视图直接搬运。
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=device_pool.k_buffer[layer_id],
                        v_cache_dst=device_pool.v_buffer[layer_id],
                        k_cache_src=self.k_buffer[layer_id],
                        v_cache_src=self.v_buffer[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer(
                        src_k=self.k_buffer[layer_id],
                        dst_k=device_pool.k_buffer[layer_id],
                        src_v=self.v_buffer[layer_id],
                        dst_v=device_pool.v_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                # page_first：主机侧是 page 优先布局，需要按层取带步长的视图。
                if self.can_use_jit:
                    # 用构造期缓存好的按层视图 k/v_data_refs（已转置为 [layer, page, ...]），
                    # kernel 会自动处理源/目的不同的步长（strided layout）。
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=device_pool.k_buffer[layer_id],
                        v_cache_dst=device_pool.v_buffer[layer_id],
                        k_cache_src=self.k_data_refs[layer_id],
                        v_cache_src=self.v_data_refs[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    # 非 JIT：由 pf_lf（page-first -> layer-first）kernel 完成布局转换搬运。
                    transfer_kv_per_layer_pf_lf(
                        src_k=self.k_buffer,
                        dst_k=device_pool.k_buffer[layer_id],
                        src_v=self.v_buffer,
                        dst_v=device_pool.v_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                    )
            elif self.layout == "page_head":
                transfer_kv_per_layer_ph_lf(
                    src_k=self.k_buffer,
                    dst_k=device_pool.k_buffer[layer_id],
                    src_v=self.v_buffer,
                    dst_v=device_pool.v_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    item_size=self.token_stride_size,
                    src_layout_dim=self.layout_dim,
                    page_size=self.page_size,
                    head_num=self.head_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            # 直接拷贝路径：不走专用散列 kernel，按页在源/目的槽位间搬运。
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.k_buffer[layer_id], self.v_buffer[layer_id]],
                    dst_layers=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.k_buffer, self.v_buffer],
                    dst_ptrs=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_direct":
                # 昇腾专用：该 kernel 一次搬运所有层，因此只在 layer_id==0 时触发一次。
                if layer_id == 0:
                    transfer_kv_dim_exchange(
                        device_indices=device_indices,
                        host_indices=host_indices,
                        device_k=device_pool.k_buffer,
                        host_k=self.k_buffer,
                        device_v=device_pool.v_buffer,
                        host_v=self.v_buffer,
                        page_size=self.page_size,
                        direction=TransferDirection.H2D,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        """把所有层的 KV 从显存池「换出/备份」到主机池（D2H），是 load 的反向操作。

        与 load 逐层不同，backup 一次性搬运全部层（用各层首地址指针数组 k/v_data_ptrs），
        同样按 (io_backend, layout) 分派实现。
        """
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer(
                        k_ptr_dst=self.k_data_ptrs,
                        v_ptr_dst=self.v_data_ptrs,
                        indices_dst=host_indices,
                        k_ptr_src=device_pool.k_data_ptrs,
                        v_ptr_src=device_pool.v_data_ptrs,
                        indices_src=device_indices,
                        kv_cache_dst_stride_bytes=self.token_stride_size,
                        kv_cache_src_stride_bytes=self.token_stride_size,
                        element_size=self.element_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer(
                        src_k_layers=device_pool.k_data_ptrs,
                        dst_k_layers=self.k_data_ptrs,
                        src_v_layers=device_pool.v_data_ptrs,
                        dst_v_layers=self.v_data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    # 用转置后的按层指针，使 kernel 按 [layer, page, item] 视图写入，
                    # 每个 token 的目的步长为 layout_dim（跨全部层）。
                    jit_transfer_hicache_all_layer(
                        k_ptr_dst=self.k_data_ptrs,
                        v_ptr_dst=self.v_data_ptrs,
                        indices_dst=host_indices,
                        k_ptr_src=device_pool.k_data_ptrs,
                        v_ptr_src=device_pool.v_data_ptrs,
                        indices_src=device_indices,
                        kv_cache_src_stride_bytes=self.token_stride_size,
                        kv_cache_dst_stride_bytes=self.layout_dim,
                        element_size=self.element_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer_lf_pf(
                        src_k_layers=device_pool.k_data_ptrs,
                        dst_k=self.k_buffer,
                        src_v_layers=device_pool.v_data_ptrs,
                        dst_v=self.v_buffer,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        dst_layout_dim=self.layout_dim,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_head":
                transfer_kv_all_layer_lf_ph(
                    src_k_layers=device_pool.k_data_ptrs,
                    dst_k=self.k_buffer,
                    src_v_layers=device_pool.v_data_ptrs,
                    dst_v=self.v_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    dst_layout_dim=self.layout_dim,
                    num_layers=self.layer_num,
                    page_size=self.page_size,
                    head_num=self.head_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.k_buffer + device_pool.v_buffer,
                    dst_layers=self.k_data_refs + self.v_data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.k_buffer + device_pool.v_buffer,
                    dst_ptrs=[self.k_buffer, self.v_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_direct":
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_pool.k_buffer,
                    host_k=self.k_buffer,
                    device_v=device_pool.v_buffer,
                    host_v=self.v_buffer,
                    page_size=self.page_size,
                    direction=TransferDirection.D2H,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """取出从 index 起、一页（page_size 个 token）的 KV 数据，供落盘/上传到 storage。

        不同布局下「一页」在缓冲区里的切片方式不同；flat=True 时展平成一维，
        便于按连续字节写入外部存储。
        """
        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, :, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :, :]
        elif self.layout in ["page_first_direct", "page_head"]:
            # 这两种布局第 1 维是 page 而非 token，需把 token 索引换算成页索引。
            real_index = index // self.page_size
            data_page = self.kv_buffer[:, real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """返回一页大小的全零占位数据（已展平），用于预取占位或初始化空页。"""
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """把一个扁平数据页写回缓冲区 index 处的一页；按当前布局 reshape 后回填。"""
        if self.layout == "layer_first":
            self.kv_buffer[:, :, index : index + self.page_size, :, :] = (
                data_page.reshape(
                    2,
                    self.layer_num,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                )
            )
        elif self.layout == "page_first":
            self.kv_buffer[:, index : index + self.page_size, :, :, :] = (
                data_page.reshape(
                    2, self.page_size, self.layer_num, self.head_num, self.head_dim
                )
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = (
                data_page.reshape(
                    2, 1, self.layer_num, self.page_size, self.head_num, self.head_dim
                )
            )
        elif self.layout == "page_head":
            real_index = index // self.page_size
            self.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = (
                data_page.reshape(
                    2, 1, self.head_num, self.page_size, self.layer_num, self.head_dim
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        """按 split_factor 拆分注意力头，返回各分块的 (指针列表, 元素字节数列表)。

        用于「异构 rank」间零拷贝（zero copy）传输 KV：当不同 rank 承载的头数不同时，
        把每页的 K/V 缓冲区按 head 维切成 split_factor 份，给出每一份的首地址与大小，
        对端可据此直接 RDMA 读写而无需中转拷贝。仅支持 page_head 布局。
        """
        assert self.layout == "page_head"
        assert len(indices) % self.page_size == 0
        assert self.head_num % split_factor == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        for index in range(0, len(indices), self.page_size):
            for head_id in range(0, self.head_num, self.head_num // split_factor):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.head_num
                    * self.head_dim
                    * self.dtype.itemsize
                    + head_id
                    * self.page_size
                    * self.layer_num
                    * self.head_dim
                    * self.dtype.itemsize
                )
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
        element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
            // split_factor
        )
        element_size_list = [element_size] * len(ptr_list)
        return ptr_list, element_size_list

    def get_page_buffer_meta(self, indices):
        """返回 indices 各页的 (K/V 首地址列表, 每项字节数列表)，供零拷贝传输。

        遍历给定槽位（每 page_size 个为一页），按当前布局计算每页 K、V 的内存首地址；
        v_offset 是 V 相对 K 的固定字节偏移（因为 kv_buffer[0] 是全部 K、[1] 是全部 V）。
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        # V 相对 K 的字节偏移：一整份 K 缓冲区的大小。
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        if self.layout == "layer_first":
            # layer_first：每页每层各一个 K/V 指针（token 优先在层内连续）。
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index]
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                        + layer_id
                        * self.size
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                    )
                    v_ptr = k_ptr + v_offset
                    ptr_list.append(k_ptr)
                    ptr_list.append(v_ptr)
            element_size = (
                self.dtype.itemsize * self.page_size * self.head_num * self.head_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct", "page_head"]:
            # page 优先：整页（含所有层）在内存中连续，每页只需一个 K 和一个 V 指针。
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.head_num
                    * self.head_dim
                    * self.dtype.itemsize
                )
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
            element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * self.head_num
                * self.head_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        """判断每页步长是否为 page_size_bytes（默认 4KiB，即 OS 页）的整数倍。

        对基于文件的 NIXL 后端启用 O_DIRECT 时，传给 kernel 的每个数据指针都必须页对齐。
        零拷贝模式下，第 p 页的指针为：

            base_ptr + p * page_size * layer_num * head_num * head_dim * itemsize

        因此（在 base_ptr 已页对齐的前提下）只要每页步长本身是 OS 页的整数倍即满足对齐。
        仅 page 优先布局可能满足；对齐失败会退化为拷贝模式。
        """
        if self.layout not in ("page_first", "page_first_direct", "page_head"):
            return False
        stride = (
            self.page_size
            * self.layer_num
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        base_aligned = self.kv_buffer.data_ptr() % page_size_bytes == 0
        return base_aligned and stride % page_size_bytes == 0


class AsymmetricMHATokenToKVPoolHost(MHATokenToKVPoolHost):
    """K、V 头维不同（head_dim != v_head_dim）的 MHA 模型的主机 KV 池，例如 MiMo-V2。

    与对称 MHA 不同，这里把 K、V 存在两块独立的主机缓冲区（self.k_buffer、self.v_buffer），
    而不是单个 (2, ...) 张量——因为两者步长不一致，各自保留自己的原生步长。
    kernel 搬运路径把 K、V 当作两次独立的单缓冲区拷贝，各用自己的 item_size。
    而「direct 直接搬运」和「扁平页 L3 存储接口」都假设 K/V 共用同一个 item_size，
    这在非对称场景下不安全，因此这些路径直接抛异常，而不是悄悄写坏 V 的数据。
    """

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num
        self.v_head_dim = self.device_pool.v_head_dim  # V 的头维，可能与 K 不同
        # 每 token 字节数：K 用 head_dim、V 用 v_head_dim，二者相加再乘头数/层数/dtype。
        return (
            (self.head_dim + self.v_head_dim)
            * self.head_num
            * self.layer_num
            * self.dtype.itemsize
        )

    def get_ksize_per_token(self):
        # 单独 K 的每 token 字节数（因 K/V 不对称，不能简单取总量一半）。
        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize

    def init_kv_buffer(self):
        # 非对称 K/V 仅支持 page_first 布局；K、V 分别按各自的头维分配。
        if self.layout == "page_first":
            k_dims = (self.size, self.layer_num, self.head_num, self.head_dim)
            v_dims = (self.size, self.layer_num, self.head_num, self.v_head_dim)
        else:
            raise ValueError(
                f"Unsupported layout for models with head_dim != v_head_dim: "
                f"{self.layout}; expected 'page_first'."
            )

        # 刻意不设置 token_stride_size / layout_dim：K、V 步长不同，任何试图取用
        # 单一共享步长的调用都是 bug。此时会因 AttributeError 显式报错，
        # 而不是误用 K 的步长去拷贝 V（悄悄写坏数据）。

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        k_buffer = alloc_func(
            k_dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        v_buffer = alloc_func(
            v_dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        return (k_buffer, v_buffer)

    def _k_token_stride_size(self) -> int:
        # K 侧单 token 单层的字节步长。
        return self.head_num * self.head_dim * self.dtype.itemsize

    def _v_token_stride_size(self) -> int:
        # V 侧单 token 单层的字节步长（用 v_head_dim）。
        return self.head_num * self.v_head_dim * self.dtype.itemsize

    def _k_layout_dim(self) -> int:
        # K 侧跨全部层的字节步长。
        return self._k_token_stride_size() * self.layer_num

    def _v_layout_dim(self) -> int:
        # V 侧跨全部层的字节步长。
        return self._v_token_stride_size() * self.layer_num

    def _flat_page_unsupported(self) -> NotImplementedError:
        # 非对称 K/V 不支持扁平页接口，统一返回带说明的异常给上层抛出。
        return NotImplementedError(
            "Models with head_dim != v_head_dim do not support the flat-page "
            "interface used by HiCache L3 storage backends {hf3fs, eic, nixl}. "
            "Use a backend that does not use this interface (e.g. mooncake, simm)."
        )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        # 换入：K、V 各做一次独立搬运，各用自己的 item_size / layout_dim；仅支持 kernel。
        if io_backend == "kernel":
            if self.layout != "page_first":
                raise ValueError(
                    f"Unsupported layout for models with head_dim != v_head_dim "
                    f"and io_backend='kernel': {self.layout}; expected 'page_first'."
                )
            transfer_kv_per_layer_mla_pf_lf(
                src=self.k_buffer,
                dst=device_pool.k_buffer[layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=layer_id,
                item_size=self._k_token_stride_size(),
                src_layout_dim=self._k_layout_dim(),
            )
            transfer_kv_per_layer_mla_pf_lf(
                src=self.v_buffer,
                dst=device_pool.v_buffer[layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=layer_id,
                item_size=self._v_token_stride_size(),
                src_layout_dim=self._v_layout_dim(),
            )
        else:
            raise ValueError(
                f"Unsupported IO backend for models with head_dim != v_head_dim: "
                f"{io_backend}; expected 'kernel'."
            )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        # 换出：同样把 K、V 分成两次独立的全层搬运；仅支持 kernel。
        if io_backend == "kernel":
            if self.layout != "page_first":
                raise ValueError(
                    f"Unsupported layout for models with head_dim != v_head_dim "
                    f"and io_backend='kernel': {self.layout}; expected 'page_first'."
                )
            transfer_kv_all_layer_mla_lf_pf(
                src_layers=device_pool.k_data_ptrs,
                dst=self.k_buffer,
                src_indices=device_indices,
                dst_indices=host_indices,
                item_size=self._k_token_stride_size(),
                dst_layout_dim=self._k_layout_dim(),
                num_layers=self.layer_num,
            )
            transfer_kv_all_layer_mla_lf_pf(
                src_layers=device_pool.v_data_ptrs,
                dst=self.v_buffer,
                src_indices=device_indices,
                dst_indices=host_indices,
                item_size=self._v_token_stride_size(),
                dst_layout_dim=self._v_layout_dim(),
                num_layers=self.layer_num,
            )
        else:
            raise ValueError(
                f"Unsupported IO backend for models with head_dim != v_head_dim: "
                f"{io_backend}; expected 'kernel'."
            )

    # 以下扁平页接口对非对称 K/V 不安全，统一抛异常（见 _flat_page_unsupported 说明）。
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raise self._flat_page_unsupported()

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise self._flat_page_unsupported()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise self._flat_page_unsupported()

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        # 分头元数据依赖 page_head 布局，非对称 K/V 不支持，直接报错。
        raise NotImplementedError(
            "get_split_heads_page_buffer_meta requires layout='page_head', "
            "which is not supported for models with head_dim != v_head_dim."
        )

    def get_page_buffer_meta(self, indices):
        # 零拷贝元数据：K、V 各自独立计算首地址与字节数（各用自己的头维）。
        assert len(indices) % self.page_size == 0
        if self.layout != "page_first":
            raise ValueError(
                f"Unsupported layout for models with head_dim != v_head_dim: "
                f"{self.layout}"
            )
        indices = indices.tolist()
        k_base_ptr = self.k_buffer.data_ptr()
        v_base_ptr = self.v_buffer.data_ptr()
        k_element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
        )
        v_element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.v_head_dim
        )
        ptr_list = []
        element_size_list = []
        for index in range(0, len(indices), self.page_size):
            k_ptr = (
                k_base_ptr
                + indices[index]
                * self.layer_num
                * self.head_num
                * self.head_dim
                * self.dtype.itemsize
            )
            v_ptr = (
                v_base_ptr
                + indices[index]
                * self.layer_num
                * self.head_num
                * self.v_head_dim
                * self.dtype.itemsize
            )
            ptr_list.extend([k_ptr, v_ptr])
            element_size_list.extend([k_element_size, v_element_size])
        return ptr_list, element_size_list

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        # K、V 两侧的每页步长与基址都必须页对齐，才允许 O_DIRECT。
        if self.layout != "page_first":
            return False
        k_stride = (
            self.page_size
            * self.layer_num
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        v_stride = (
            self.page_size
            * self.layer_num
            * self.head_num
            * self.v_head_dim
            * self.dtype.itemsize
        )
        base_aligned = (
            self.k_buffer.data_ptr() % page_size_bytes == 0
            and self.v_buffer.data_ptr() % page_size_bytes == 0
        )
        return (
            base_aligned
            and k_stride % page_size_bytes == 0
            and v_stride % page_size_bytes == 0
        )


def get_mha_host_pool_cls(device_pool: MHATokenToKVPool) -> type:
    """根据 device 池的 K/V 维度选择合适的 MHA 主机池类。

    当 head_dim != v_head_dim（如 MiMo-V2）时返回非对称版
    AsymmetricMHATokenToKVPoolHost，否则返回默认的 MHATokenToKVPoolHost。
    """
    if device_pool.head_dim != device_pool.v_head_dim:
        return AsymmetricMHATokenToKVPoolHost
    return MHATokenToKVPoolHost


class MLATokenToKVPoolHost(HiSparseHostPoolMixin, HostKVCache):
    """DeepSeek MLA（多头潜在注意力）的主机 KV 池。

    MLA 把 K/V 压缩成一份低秩潜在向量（维度 = kv_lora_rank + qk_rope_head_dim），
    因此不像 MHA 那样区分 K、V 两块，而是每 token 只存一份 kv_cache_dim 的向量，
    显著降低 KV 缓存体积。混入 HiSparseHostPoolMixin 以支持按页分配。
    """

    device_pool: MLATokenToKVPool

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        override_kv_cache_dim: Optional[int] = None,
    ):
        # override_kv_cache_dim：外部强制指定潜在向量维度（否则由 lora_rank+rope 推算）。
        self.override_kv_cache_dim = override_kv_cache_dim
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
        )
        self.can_use_jit = _is_cuda and can_use_hicache_jit_kernel(
            element_size=self.kv_cache_dim * self.dtype.itemsize
        )

        # 预构造按层视图与其数据指针（同 MHA 思路，但只有一份缓冲区而非 K/V 两份）。
        if self.layout == "page_first" and self.can_use_jit:
            # page_first：转置 [page, layer, ...] -> [layer, page, ...] 得到按层视图，
            # 只换步长不拷贝数据。
            transposed = self.kv_buffer.transpose(0, 1)
            self.data_refs = [transposed[i] for i in range(self.layer_num)]
        else:
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def get_contiguous_buf_infos(self):
        """返回与 device 池同格式的 (data_ptrs, data_lens, item_lens)。

        供 PD 分离（disaggregation）的传输引擎注册这块主机内存，以便跨进程/跨机
        直接搬运 KV。三元组分别是各层首地址、各层字节数、每项（一页）的字节数。
        """
        data_ptrs = [int(self.data_ptrs[i].item()) for i in range(self.layer_num)]
        data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.token_stride_size * self.page_size] * self.layer_num
        return data_ptrs, data_lens, item_lens

    def get_size_per_token(self):
        self.kv_lora_rank = self.device_pool.kv_lora_rank  # 潜在向量的低秩压缩维度
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim  # RoPE 位置编码维度
        self.layer_num = self.device_pool.layer_num
        # MLA 每 token 的潜在向量维度 = 低秩维 + RoPE 维（可被外部覆盖）。
        self.kv_cache_dim = self.override_kv_cache_dim or (
            self.kv_lora_rank + self.qk_rope_head_dim
        )
        # 每 token 字节数 = 维度 * dtype字节 * 层数（只有一份，无需 *2）。
        return self.kv_cache_dim * self.dtype.itemsize * self.layer_num

    def get_ksize_per_token(self):
        # MLA 只有一份潜在向量，K 的大小即全部大小。
        return self.get_size_per_token()

    def init_kv_buffer(self):
        # 单份潜在向量缓冲区，无 K/V 第 0 维；倒数第二维恒为 1（占位单头）。
        if self.layout == "layer_first":
            dims = (
                self.layer_num,
                self.size,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first":
            dims = (
                self.size,
                self.layer_num,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first_direct":
            dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        # 昇腾专用：对齐 NPUMLATokenToKVPool 的布局，把潜在向量拆成
        # k_buffer（低秩部分）与 v_buffer（RoPE 部分）分开存，便于数据搬运。
        elif self.layout == "page_first_kv_split":
            base_dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                1,
            )
            alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
            self.k_buffer = alloc_func(
                (*base_dims, self.kv_lora_rank),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.v_buffer = alloc_func(
                (*base_dims, self.qk_rope_head_dim),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.index_k_buffer = None
            if self.device_pool.index_head_dim is not None:
                self.index_k_buffer = alloc_func(
                    (*base_dims, self.device_pool.index_head_dim),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
            # 返回 k_buffer 只是为了复用基类里 kv_buffer / data_refs 的初始化逻辑，
            # 昇腾路径实际并不使用这些派生视图。
            return self.k_buffer
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        # 单 token 单层的字节步长，以及跨全部层的步长。
        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        return buffer

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        """把单层的 MLA 潜在向量从主机换入显存（H2D）；结构同 MHA 但只搬一份缓冲区。"""
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.kv_buffer[layer_id],
                        cache_src=self.kv_buffer[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    transfer_kv_per_layer_mla(
                        src=self.kv_buffer[layer_id],
                        dst=device_pool.kv_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.kv_buffer[layer_id],
                        cache_src=self.data_refs[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    transfer_kv_per_layer_mla_pf_lf(
                        src=self.kv_buffer,
                        dst=device_pool.kv_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.kv_buffer[layer_id]],
                    dst_layers=[device_pool.kv_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.kv_buffer],
                    dst_ptrs=[device_pool.kv_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_kv_split":
                # 昇腾专用：该 kernel 一次搬运所有层，只在 layer_id==0 触发一次；
                # 分别搬运低秩(k)、RoPE(v) 及可选的索引(index_k)三块缓冲区。
                if layer_id == 0:
                    transfer_kv_dim_exchange(
                        device_indices=device_indices,
                        host_indices=host_indices,
                        device_k=device_pool.k_buffer,
                        host_k=self.k_buffer,
                        device_v=device_pool.v_buffer,
                        host_v=self.v_buffer,
                        device_index_k=device_pool.index_k_buffer,
                        host_index_k=self.index_k_buffer,
                        page_size=self.page_size,
                        direction=TransferDirection.H2D,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        """把所有层的 MLA 潜在向量从显存换出到主机（D2H）。"""
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer_mla(
                        ptr_dst=self.data_ptrs,
                        indices_dst=host_indices,
                        ptr_src=device_pool.data_ptrs,
                        indices_src=device_indices,
                        cache_dst_stride_bytes=self.token_stride_size,
                        cache_src_stride_bytes=self.token_stride_size,
                        element_size=self.kv_cache_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer_mla(
                        src_layers=device_pool.data_ptrs,
                        dst_layers=self.data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer_mla(
                        ptr_dst=self.data_ptrs,
                        indices_dst=host_indices,
                        ptr_src=device_pool.data_ptrs,
                        indices_src=device_indices,
                        cache_src_stride_bytes=self.token_stride_size,
                        cache_dst_stride_bytes=self.layout_dim,
                        element_size=self.kv_cache_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer_mla_lf_pf(
                        src_layers=device_pool.data_ptrs,
                        dst=self.kv_buffer,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        dst_layout_dim=self.layout_dim,
                        num_layers=self.layer_num,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.kv_buffer,
                    dst_layers=self.data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.kv_buffer,
                    dst_ptrs=[self.kv_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_kv_split":
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_pool.k_buffer,
                    host_k=self.k_buffer,
                    device_v=device_pool.v_buffer,
                    host_v=self.v_buffer,
                    device_index_k=device_pool.index_k_buffer,
                    host_index_k=self.index_k_buffer,
                    page_size=self.page_size,
                    direction=TransferDirection.D2H,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """取出从 index 起一页的 MLA 潜在向量数据（无 K/V 维），可选展平。"""
        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[index : index + self.page_size, :, :, :]
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            data_page = self.kv_buffer[real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """返回一页大小的全零占位潜在向量（已展平）。"""
        return torch.zeros(
            (
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            ),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """把一个扁平数据页 reshape 后写回主机缓冲区 index 处的一页。"""
        if self.layout == "layer_first":
            self.kv_buffer[:, index : index + self.page_size, :, :] = data_page.reshape(
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first":
            self.kv_buffer[index : index + self.page_size, :, :, :] = data_page.reshape(
                self.page_size,
                self.layer_num,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[real_index : real_index + 1, :, :, :, :] = data_page.reshape(
                1,
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """返回各页潜在向量的 (首地址列表, 每项字节数列表)，供零拷贝传输。

        MLA 只有一份缓冲区（无 K/V 之分），故每项只记录一个指针。
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        if self.layout == "layer_first":
            # layer_first：每页每层各一个指针。
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index] * self.kv_cache_dim * self.dtype.itemsize
                        + layer_id * self.size * self.kv_cache_dim * self.dtype.itemsize
                    )
                    ptr_list.append(k_ptr)
            element_size = self.dtype.itemsize * self.page_size * self.kv_cache_dim
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct"]:
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.kv_cache_dim
                    * self.dtype.itemsize
                )
                ptr_list.append(k_ptr)
            element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * self.kv_cache_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        """判断每页步长是否为 page_size_bytes（默认 OS 页 4KiB）的整数倍。

        对基于文件的 NIXL 后端启用 O_DIRECT 时，每个数据指针都必须页对齐。
        零拷贝下第 p 页指针为：

            base_ptr + p * page_size * layer_num * kv_cache_dim * itemsize

        故在 base_ptr 已对齐前提下，只要每页步长是 OS 页整数倍即满足。
        """
        if self.layout not in ("page_first", "page_first_direct"):
            return False
        stride = (
            self.page_size * self.layer_num * self.kv_cache_dim * self.dtype.itemsize
        )
        base_aligned = self.kv_buffer.data_ptr() % page_size_bytes == 0
        return base_aligned and stride % page_size_bytes == 0


class MambaPoolHost(HostKVCache):
    """Mamba / 线性注意力等模型「循环状态」的主机池。

    与注意力 KV 不同，Mamba 每层维护的是固定大小的循环状态：
    卷积状态（conv_state）与 SSM 状态（temporal/ssm_state）。这些状态按层组织、
    页大小固定为 1（每个 token 位置一份状态），主机池负责把它们在显存与主机内存间换入换出。
    """

    def __init__(
        self,
        device_pool: MambaPool,
        host_to_device_ratio: float,
        host_size: int,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        layout: str = "layer_first",
    ):
        self.device_pool = device_pool
        self.page_size = 1
        assert layout in [
            "page_first",
            "page_first_direct",
            "layer_first",
        ], f"Unsupported layout: {layout}"

        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.num_mamba_layers = device_pool.num_mamba_layers

        # 从 device 池抄下每层卷积状态与时序（SSM）状态的形状/元素数/dtype。
        # shape[2:] 去掉前两维（层维与槽位维），只保留单份状态的形状。
        self.conv_state_shapes = [
            conv_state.shape[2:] for conv_state in device_pool.mamba_cache.conv
        ]
        self.temporal_state_shape = device_pool.mamba_cache.temporal.shape[2:]
        self.temporal_state_elem_size = int(np.prod(self.temporal_state_shape))
        self.conv_state_elem_sizes = [
            int(np.prod(conv_shape)) for conv_shape in self.conv_state_shapes
        ]
        self.conv_dtype = device_pool.mamba_cache.conv[0].dtype
        self.temporal_dtype = device_pool.mamba_cache.temporal.dtype
        # conv 与 temporal 的 dtype 可能不同；self.dtype 取 conv 的 dtype 作为代表。
        self.dtype = self.conv_dtype
        self.size_per_token = self.get_size_per_token()

        # 注意：Mamba 状态由 conv + temporal 两类缓冲区组成，结构与注意力 KV 差异较大，
        # 因此该类没有调用父类 __init__，而是在此重复了容量推算与内存校验逻辑。

        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)

        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size

        assert (
            self.size > device_pool.size
        ), "The host memory should be larger than the device memory with the current protocol"

        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        logger.info(
            "Allocating %.2f GB host memory for hierarchical Mamba cache (layout=%s).",
            requested_bytes / 1e9,
            self.layout,
        )

        self.init_kv_buffer()
        self.lock = threading.RLock()
        self.clear()

    def init_kv_buffer(self):
        # 分别为「时序状态」和每一类「卷积状态」分配主机缓冲区；两种布局形状不同。
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]

        if self.layout in ["page_first", "page_first_direct"]:
            # page 优先：(size, num_layers, 1, *shape)，同一页的数据在内存中连续。
            temporal_dims = (
                self.size,
                self.num_mamba_layers,
                1,
            ) + self.temporal_state_shape
            self.temporal_buffer = alloc_func(
                temporal_dims,
                dtype=self.temporal_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.conv_buffer = []
            for conv_shape in self.conv_state_shapes:
                conv_dims = (self.size, self.num_mamba_layers, 1) + conv_shape
                self.conv_buffer.append(
                    alloc_func(
                        conv_dims,
                        dtype=self.conv_dtype,
                        device=self.device,
                        pin_memory=self.pin_memory,
                        allocator=self.allocator,
                    )
                )
        else:
            # layer 优先：(num_layers, size, *shape)，同一层的所有槽位在内存中连续。
            temporal_dims = (
                self.num_mamba_layers,
                self.size,
            ) + self.temporal_state_shape
            self.temporal_buffer = alloc_func(
                temporal_dims,
                dtype=self.temporal_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.conv_buffer = []
            for conv_shape in self.conv_state_shapes:
                conv_dims = (self.num_mamba_layers, self.size) + conv_shape
                self.conv_buffer.append(
                    alloc_func(
                        conv_dims,
                        dtype=self.conv_dtype,
                        device=self.device,
                        pin_memory=self.pin_memory,
                        allocator=self.allocator,
                    )
                )

    def get_hybrid_pool_buffer(self):
        # 暴露所有需要向 Mooncake 注册的 Mamba 主机张量（时序 + 各卷积缓冲区）。
        return [self.temporal_buffer, *self.conv_buffer]

    def _iter_page_tensors(self, index: int):
        # 依次产出某页（index 处）的时序状态与各卷积状态的张量视图，
        # 供按页读写/展平时统一遍历。两种布局的切片方式不同。
        if self.layout in ["page_first", "page_first_direct"]:
            yield self.temporal_buffer[index]
            for conv_buf in self.conv_buffer:
                yield conv_buf[index]
        else:
            yield self.temporal_buffer[:, index : index + self.page_size]
            for conv_buf in self.conv_buffer:
                yield conv_buf[:, index : index + self.page_size]

    @staticmethod
    def _flatten_tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
        # 把张量按原始字节展平成一维 uint8，便于跨 dtype（conv/temporal 不同）拼接搬运。
        return tensor.contiguous().view(torch.uint8).reshape(-1)

    @synchronized
    def clear(self):
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size > self.available_size():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)

    def get_size_per_token(self):
        # 每 token 字节数 = (所有卷积状态字节 + 时序状态字节) * 层数。
        conv_total_size = sum(
            conv_elem_size * self.conv_dtype.itemsize
            for conv_elem_size in self.conv_state_elem_sizes
        )
        temporal_size = self.temporal_state_elem_size * self.temporal_dtype.itemsize
        return (conv_total_size + temporal_size) * self.num_mamba_layers

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    @staticmethod
    def _item_size_per_index(tensor: torch.Tensor) -> int:
        # 单个槽位（index）对应的字节数 = 去掉第 0 维后的元素数 * 元素字节。空张量返回 0。
        if tensor.shape[0] == 0:
            return 0
        return int(tensor[0].numel() * tensor.element_size())

    @staticmethod
    def _copy_tensor(
        src: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        io_backend: str,
    ) -> None:
        # 在单层（layer_first）布局下，按索引搬运单个状态张量（时序或某个卷积）。
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            # TODO: 为清晰起见应重命名该接口。
            # 这里复用 transfer_kv_per_layer_mla 来搬运 Mamba 状态，与 MLA 无关——
            # 仅仅因为该接口恰好能搬运「单个池」而被复用。
            transfer_kv_per_layer_mla(
                src=src,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                item_size=MambaPoolHost._item_size_per_index(src),
            )
        elif io_backend == "direct":
            transfer_kv_direct(
                src_layers=[src],
                dst_layers=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    @staticmethod
    def _copy_tensor_pf_lf(
        src: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        layer_id: int,
        num_layers: int,
        io_backend: str,
    ) -> None:
        # page-first 源 -> layer-first 目的（换入单层时用）：src 为整块 page 优先缓冲区，
        # 按 layer_id 取出对应层写入 dst。
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            item_size = MambaPoolHost._item_size_per_index(dst)
            transfer_kv_per_layer_mla_pf_lf(
                src=src,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                layer_id=layer_id,
                item_size=item_size,
                src_layout_dim=item_size * num_layers,
            )
        elif io_backend == "direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[src],
                dst_ptrs=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                layer_id=layer_id,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    @staticmethod
    def _copy_tensor_all_layers_lf_pf(
        src_layers: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        num_layers: int,
        device: str,
        io_backend: str,
    ) -> None:
        # layer-first 源 -> page-first 目的（换出全部层时用）：把各层状态搬进
        # page 优先的主机缓冲区。kernel 路径需先收集各层首地址指针数组。
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            item_size = MambaPoolHost._item_size_per_index(src_layers[0])
            src_ptrs = torch.tensor(
                [src_layers[i].data_ptr() for i in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            transfer_kv_all_layer_mla_lf_pf(
                src_layers=src_ptrs,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                item_size=item_size,
                dst_layout_dim=item_size * num_layers,
                num_layers=num_layers,
            )
        elif io_backend == "direct":
            src_ptrs = [src_layers[i] for i in range(num_layers)]
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=src_ptrs,
                dst_ptrs=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend="kernel",
    ):
        # 按层将 Mamba 状态从 host 载入 device（H2D swap-in）。
        # Mamba 层的状态由两部分组成：temporal（时序/ssm 状态）与 conv（卷积状态，
        # 可能有多个分量）。这里对两类缓冲区分别执行拷贝。
        if self.layout in ["page_first", "page_first_direct"]:
            # page-first 布局：host 端按 page 组织，需要用 pf->lf 的转换拷贝把
            # 单层数据从 page-first 还原到 device 的 layer-first 布局。
            self._copy_tensor_pf_lf(
                src=self.temporal_buffer,
                dst=device_pool.mamba_cache.temporal[layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=layer_id,
                num_layers=self.num_mamba_layers,
                io_backend=io_backend,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor_pf_lf(
                    src=self.conv_buffer[conv_idx],
                    dst=device_pool.mamba_cache.conv[conv_idx][layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    num_layers=self.num_mamba_layers,
                    io_backend=io_backend,
                )
        else:
            # layer_first 布局：host 端每层独立存放，直接按索引做整块拷贝即可。
            self._copy_tensor(
                self.temporal_buffer[layer_id],
                device_pool.mamba_cache.temporal[layer_id],
                host_indices,
                device_indices,
                io_backend,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor(
                    self.conv_buffer[conv_idx][layer_id],
                    device_pool.mamba_cache.conv[conv_idx][layer_id],
                    host_indices,
                    device_indices,
                    io_backend,
                )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend="kernel"
    ):
        # 一次性将所有 Mamba 层的状态从 device 备份到 host（D2H swap-out）。
        # 与 load 相反，这里对 temporal 与 conv 缓冲区分别执行 device->host 拷贝。
        if self.layout in ["page_first", "page_first_direct"]:
            # page-first 布局：用 lf->pf 的全层转换拷贝，把 device 上 layer-first
            # 的各层数据聚合写入 host 的 page-first 缓冲区。
            self._copy_tensor_all_layers_lf_pf(
                src_layers=device_pool.mamba_cache.temporal,
                dst=self.temporal_buffer,
                src_indices=device_indices,
                dst_indices=host_indices,
                num_layers=self.num_mamba_layers,
                device=self.device_pool.device,
                io_backend=io_backend,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor_all_layers_lf_pf(
                    src_layers=device_pool.mamba_cache.conv[conv_idx],
                    dst=self.conv_buffer[conv_idx],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    num_layers=self.num_mamba_layers,
                    device=self.device_pool.device,
                    io_backend=io_backend,
                )
        else:
            # layer_first 布局：逐层、逐 conv 分量分别做整块拷贝。
            for layer_id in range(self.num_mamba_layers):
                self._copy_tensor(
                    device_pool.mamba_cache.temporal[layer_id],
                    self.temporal_buffer[layer_id],
                    device_indices,
                    host_indices,
                    io_backend,
                )
                for conv_idx in range(len(self.conv_state_shapes)):
                    self._copy_tensor(
                        device_pool.mamba_cache.conv[conv_idx][layer_id],
                        self.conv_buffer[conv_idx][layer_id],
                        device_indices,
                        host_indices,
                        io_backend,
                    )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        # 将某个 page 内的 temporal 与所有 conv 分量拉平为字节序列并拼接，
        # 得到一个连续的 data page（可用于落盘/网络传输的序列化表示）。
        data_page = torch.cat(
            [
                self._flatten_tensor_bytes(tensor)
                for tensor in self._iter_page_tensors(index)
            ]
        )
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        # 构造一个全零的占位 data page（大小为 page_size * size_per_token 字节），
        # 用于预分配缓冲区或占位场景。
        return torch.zeros(
            self.page_size * self.size_per_token,
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def set_from_flat_data_page(
        self,
        index: int,
        data_page: torch.Tensor,
    ) -> None:
        # get_data_page 的逆操作：把一段扁平字节序列按顺序切分，
        # 依次还原写回该 page 的 temporal 与各 conv 分量张量。
        flat_bytes = data_page.contiguous().view(torch.uint8).reshape(-1)
        start = 0
        for tensor in self._iter_page_tensors(index):
            # 按每个分量占用的字节数从 flat_bytes 中切出对应片段。
            num_bytes = tensor.numel() * tensor.element_size()
            tensor_bytes = flat_bytes[start : start + num_bytes]
            start += num_bytes
            # 将字节片段按原 dtype/shape 复原后写回目标张量。
            restored = tensor_bytes.view(dtype=tensor.dtype).reshape(tensor.shape)
            tensor.copy_(restored)

    def get_page_buffer_meta(self, indices):
        """零拷贝存储 I/O 所需的元数据（数据指针 + 元素字节数）。

        Mamba 的零拷贝存储仅支持 page-first 布局：因为只有在 page-first 下，
        temporal/conv 缓冲区中每个 page slot 才是可直接寻址的连续内存，
        从而能给传输引擎（如 RDMA/传输引擎）提供裸指针进行零拷贝。
        """
        assert len(indices) % self.page_size == 0
        if self.layout not in ["page_first", "page_first_direct"]:
            raise ValueError(
                f"Mamba storage zero-copy requires page_first layout, got {self.layout}"
            )
        indices = indices.tolist()
        ptr_list = []
        element_size_list = []

        # 基址只需计算一次；后续每个 page 的指针都是基址加上偏移。
        temporal_base_ptr = self.temporal_buffer.data_ptr()
        conv_base_ptrs = [buf.data_ptr() for buf in self.conv_buffer]
        # 各分量每个 page 的字节大小在所有 page 间是恒定的，因此同样预计算一次。
        temporal_element_size = (
            self.page_size
            * self.num_mamba_layers
            * self.temporal_dtype.itemsize
            * self.temporal_state_elem_size
        )
        conv_element_sizes = [
            (
                self.page_size
                * self.num_mamba_layers
                * self.conv_dtype.itemsize
                * self.conv_state_elem_sizes[i]
            )
            for i in range(len(self.conv_state_shapes))
        ]

        for i in range(0, len(indices), self.page_size):
            # 以稳定顺序输出各分量指针：先 temporal，再依次 conv_0..conv_n。
            # 顺序必须与 get_data_page 中 _iter_page_tensors 的遍历顺序一致，
            # 否则读回时字节切分会错位。
            temporal_ptr = (
                temporal_base_ptr
                + indices[i]
                * self.num_mamba_layers
                * self.temporal_state_elem_size
                * self.temporal_dtype.itemsize
            )
            ptr_list.append(temporal_ptr)
            element_size_list.append(temporal_element_size)
            for j in range(len(self.conv_buffer)):
                conv_ptr = (
                    conv_base_ptrs[j]
                    + indices[i]
                    * self.num_mamba_layers
                    * self.conv_state_elem_sizes[j]
                    * self.conv_dtype.itemsize
                )
                ptr_list.append(conv_ptr)
                element_size_list.append(conv_element_sizes[j])
        return ptr_list, element_size_list


# ---- V4 Compressed KV Host Pools ----


class LogicalHostPool:
    """V4 HiCache 使用的纯逻辑锚点池（不持有任何 KV 张量）。

    该池只负责管理按 page 对齐的 token slot 分配/回收，本身不存储 KV 数据。
    V4 的压缩侧子池（compressed side pools）借用这里分配出的逻辑 FULL 索引
    作为稳定的 page 锚点（page anchor），从而在多个子池之间共享一致的寻址口径。

    因为不持有真实缓冲区，所以下面所有涉及数据搬运/序列化的接口
    （backup/load/get_data_page 等）都是空实现或返回空张量，仅保留接口契约。
    """

    def __init__(self, size: int, page_size: int):
        # size 必须按 page_size 对齐，否则无法保证逻辑索引以 page 为粒度分配。
        if size % page_size != 0:
            raise ValueError(
                "LogicalHostPool size must be page-aligned, "
                f"got size={size}, page_size={page_size}"
            )
        self.size = size
        self.page_size = page_size
        self.device = "cpu"
        self.layout = "layer_first"
        self.dtype = torch.uint8
        # 逻辑池不持有真实数据，因此层数、缓冲区、每 token 字节数均为 0/None。
        self.layer_num = 0
        self.start_layer = 0
        self.end_layer = 0
        self.kv_buffer = None
        self.size_per_token = 0
        self.allocator = None
        self.lock = threading.RLock()
        self.clear()

    @synchronized
    def clear(self):
        # 初始化时所有 slot 均空闲：free_slots 为 [0, size) 的连续索引。
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        # 分配必须按 page 对齐；从空闲列表头部取出 need_size 个索引。
        if need_size % self.page_size != 0:
            raise ValueError(
                "LogicalHostPool allocation must be page-aligned, "
                f"got need_size={need_size}, page_size={self.page_size}"
            )
        # 空闲不足时返回 None，交由上层决定驱逐/等待。
        if need_size > self.available_size():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        # 回收同样要求按 page 对齐；把释放的索引拼回空闲列表尾部。
        if len(indices) % self.page_size != 0:
            raise ValueError(
                "LogicalHostPool free must be page-aligned, "
                f"got len(indices)={len(indices)}, page_size={self.page_size}"
            )
        self.free_slots = torch.cat(
            [self.free_slots, indices.to(dtype=torch.int64, device="cpu").flatten()]
        )
        return len(indices)

    # 以下均为空实现：逻辑池没有真实 KV 缓冲区，不参与实际数据搬运与序列化，
    # 仅为满足 host pool 的统一接口契约而存在。
    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        pass

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        pass

    def get_data_page(self, index, flat=True):
        return torch.empty(0, dtype=torch.uint8)

    def get_dummy_flat_data_page(self):
        return torch.empty(0, dtype=torch.uint8)

    def set_from_flat_data_page(self, index, data_page):
        pass

    def get_page_buffer_meta(self, indices):
        return None

    def get_ksize_per_token(self):
        return 0


class DeepSeekV4PagedHostPool(HiSparseHostPoolMixin, HostKVCache):
    """DeepSeek V4 分页 KV / indexer 子池在 host 端的镜像池。

    V4 采用稀疏/分页的 KV 组织：device 侧有若干层的 page 缓冲区（device_buffers），
    这里在 host 内存中开辟一份等容量的镜像，用作 L2 缓存承接 D2H 换出的 page，
    并支持后续 H2D 换入。以「page × item_bytes」为最小寻址单位，item_bytes 表示
    单个 page slot 的字节大小。
    """

    def __init__(
        self,
        pool_name: str,
        device_buffers: list[torch.Tensor],
        item_bytes: int,
        num_host_pages: int,
        slot_page_size: int,
        layout: str = "layer_first",
        device: str = "cpu",
        pin_memory: bool = True,
        allocator_type: str = "default",
    ):
        self.pool_name = pool_name
        # 层数直接由 device 缓冲区个数推断（每层一个缓冲区）。
        self.layer_num = len(device_buffers)
        self.item_bytes = item_bytes
        self.num_host_pages = num_host_pages
        self.slot_page_size = slot_page_size
        # 该池以字节（uint8）为存储单位，统一按裸字节管理各类压缩 KV。
        self.dtype = torch.uint8
        self.device = device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        self.page_size = slot_page_size
        # 总容量以 token slot 计：page 数 × 每 page 的 slot 数。
        self.size = num_host_pages * slot_page_size
        self.layout = layout
        self.size_per_token = item_bytes
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = threading.RLock()

        self.device_buffers = device_buffers
        # 记录 device 缓冲区所在设备（GPU/NPU），用于后续拷贝与指针张量构造。
        self.gpu_device = device_buffers[0].device if device_buffers else device

        # 预估所需 host 内存并做可用性检查：预留 HICACHE_HOST_MEMORY_RESERVE_BYTES
        # 给系统/其他组件，避免占满物理内存导致 OOM。
        requested_bytes = self.layer_num * num_host_pages * self.item_bytes
        host_mem = psutil.virtual_memory()
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for V4 paged pool {pool_name}. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        # 根据目标设备选择对应的内存分配函数（如 CPU pinned / mmap 等）。
        alloc_func = ALLOC_MEMORY_FUNCS[self.gpu_device]
        self.data_refs = []
        if self.layout == "layer_first":
            # layer_first：每层单独一个 (page 数, item_bytes) 的缓冲区。
            self.kv_buffer = [
                alloc_func(
                    (num_host_pages, self.item_bytes),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                for _ in range(self.layer_num)
            ]
            # data_refs 保存各层缓冲区引用，便于统一取裸指针。
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        elif self.layout == "page_first":
            # page_first：单个大缓冲区，page 维在最外层，(page, layer, item_bytes)。
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, self.item_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        elif self.layout == "page_first_direct":
            # page_first_direct：在 page_first 基础上多插入一维（=1），
            # 使每个 page slot 单独可寻址，便于 direct/零拷贝传输对齐。
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, 1, self.item_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

        logger.info(
            "Allocating %.2f GB host memory for V4 paged pool '%s' "
            "(layers=%d, pages=%d, item_bytes=%d, layout=%s).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            self.item_bytes,
            self.layout,
        )

        # 把 device 各层缓冲区的裸指针打包成 uint64 张量，供拷贝 kernel 使用。
        self.device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_buffers],
            dtype=torch.uint64,
            device=self.gpu_device,
        )
        # 同样打包 host 侧各层缓冲区指针；仅 layer_first 布局填充了 data_refs，
        # 其他布局为单一大缓冲区，此处为 None。
        self.data_ptrs = (
            torch.tensor(
                [x.data_ptr() for x in self.data_refs],
                dtype=torch.uint64,
                device=self.gpu_device,
            )
            if self.data_refs
            else None
        )
        self.clear()

    def get_contiguous_buf_infos(self):
        """返回逐层的 page-row 缓冲区信息，供 PD 分离场景下 direct-to-host 传输使用。"""
        # 分别返回：各层数据指针、各层总字节数、每个 item（page slot）的字节数。
        data_ptrs = [int(self.data_ptrs[i].item()) for i in range(self.layer_num)]
        data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.item_bytes * self.dtype.itemsize] * self.layer_num
        return data_ptrs, data_lens, item_lens

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        # 将「token slot 索引」折算为「page 索引」：每 slot_page_size 个 slot 归一
        # 到一个 page，取该组首元素并整除页大小得到 page 编号。
        return indices.reshape(-1, self.slot_page_size)[:, 0] // self.slot_page_size

    def _has_transfer_indices(
        self, host_indices: torch.Tensor | None, device_indices: torch.Tensor | None
    ) -> bool:
        # 判断本次是否有实际需要传输的索引：任一侧为空则无需传输；
        # 两侧数量必须一致，否则说明调用方传入了不匹配的索引对。
        if host_indices is None or device_indices is None:
            return False
        if host_indices.numel() != device_indices.numel():
            raise ValueError(
                f"{self.pool_name} transfer index size mismatch: "
                f"host={host_indices.numel()}, device={device_indices.numel()}"
            )
        return host_indices.numel() > 0

    def get_size_per_token(self):
        return self.item_bytes

    def get_ksize_per_token(self):
        return self.item_bytes

    def init_kv_buffer(self):
        return self.kv_buffer

    def get_hybrid_pool_buffer(self):
        # 统一返回缓冲区列表：layer_first 本就是 list，其他布局包一层再返回。
        return self.kv_buffer if isinstance(self.kv_buffer, list) else [self.kv_buffer]

    def clear(self):
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        # 将请求向上取整到 page 边界（不足一页也占满一页），保证按 page 粒度分配。
        need_size = (
            (need_size + self.slot_page_size - 1) // self.slot_page_size
        ) * self.slot_page_size
        if need_size > self.available_size():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat(
            [self.free_slots, indices.to(dtype=torch.int64, device="cpu").flatten()]
        )
        return len(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        # 将所有层的 page 从 device 备份到 host（D2H swap-out）。
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        if (
            host_indices.numel() % self.slot_page_size != 0
            or device_indices.numel() % self.slot_page_size != 0
        ):
            # 整页对齐的 C4 数据可走下面常规的 HiCache page-row 拷贝；
            # 但当索引不是整页对齐（token 粒度）时，必须走这个专用 helper：
            # 因为 DSV4 的 C4 量化布局里，一个 token 在 page-row 内并不是一段连续
            # 字节，而是被拆成 [value0..value63][scale0..scale63] 两段，
            # 需要按该布局分别搬运。
            transfer_cache_dsv4_mla(
                src_ptrs=self.device_ptrs,
                dst_ptrs=self.data_ptrs,
                src_indices=device_indices.to(dtype=torch.int64),
                dst_indices=host_indices.to(dtype=torch.int64),
            )
            return
        # 整页对齐：先把 token 索引折算为 page 行号，再按 backend/layout 分派拷贝。
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_all_layer_mla(
                src_layers=self.device_ptrs,
                dst_layers=self.data_ptrs,
                src_indices=device_rows,
                dst_indices=host_rows,
                item_size=self.item_bytes,
                num_layers=self.layer_num,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            transfer_kv_all_layer_mla_lf_pf(
                src_layers=self.device_ptrs,
                dst=self.kv_buffer,
                src_indices=device_rows,
                dst_indices=host_rows,
                item_size=self.item_bytes,
                dst_layout_dim=self.layer_num * self.item_bytes,
                num_layers=self.layer_num,
            )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=self.device_buffers,
                dst_layers=self.data_refs,
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=self.device_buffers,
                dst_ptrs=[self.kv_buffer],
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        else:
            raise ValueError(
                f"Unsupported V4 paged host layout/backend: {self.layout}/{io_backend}"
            )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        # 按层将 page 从 host 载入 device（H2D swap-in），是 backup 的逆操作。
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        if (
            host_indices.numel() % self.slot_page_size != 0
            or device_indices.numel() % self.slot_page_size != 0
        ):
            # 与 backup 相同的 DSV4 C4 布局问题：token 粒度预取无法走常规
            # page-row 拷贝，这里只针对当前 layer_id 的指针切片做专用搬运。
            transfer_cache_dsv4_mla(
                src_ptrs=self.data_ptrs[layer_id : layer_id + 1],
                dst_ptrs=self.device_ptrs[layer_id : layer_id + 1],
                src_indices=host_indices.to(dtype=torch.int64),
                dst_indices=device_indices.to(dtype=torch.int64),
            )
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)

        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_per_layer_mla(
                src=self.data_refs[layer_id],
                dst=self.device_buffers[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                item_size=self.item_bytes,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            transfer_kv_per_layer_mla_pf_lf(
                src=self.kv_buffer,
                dst=self.device_buffers[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                item_size=self.item_bytes,
                src_layout_dim=self.layer_num * self.item_bytes,
            )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=[self.data_refs[layer_id]],
                dst_layers=[self.device_buffers[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[self.kv_buffer],
                dst_ptrs=[self.device_buffers[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                page_size=1,
            )
        else:
            raise ValueError(
                f"Unsupported V4 paged host layout/backend: {self.layout}/{io_backend}"
            )

    def get_data_page(self, index, flat=True):
        # 取出某个 page（跨所有层）的数据，用于序列化落盘/传输。
        # 入参 index 是 token slot 索引，先折算成 page 行号。
        index = int(index) // self.slot_page_size
        if self.layout == "layer_first":
            # layer_first：从每层缓冲区各取该 page 行，再沿层维堆叠。
            data_page = torch.stack(
                [self.kv_buffer[i][index] for i in range(self.layer_num)]
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            # page_first(_direct)：该 page 的所有层数据本就连续存放，直接索引即可。
            data_page = self.kv_buffer[index]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self):
        # 构造一个全零占位 page（含所有层），形状 (layer_num, item_bytes) 后拉平。
        return torch.zeros(
            (self.layer_num, self.item_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        # get_data_page 的逆操作：把扁平 page 按 layout 还原写回对应 page 行。
        index = int(index) // self.slot_page_size
        if self.layout == "layer_first":
            # 还原为 (layer_num, item_bytes) 后逐层写回各自缓冲区。
            data = data_page.view(self.dtype).reshape(self.layer_num, self.item_bytes)
            for i in range(self.layer_num):
                self.kv_buffer[i][index].copy_(data[i])
        elif self.layout == "page_first":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(self.layer_num, self.item_bytes)
            )
        elif self.layout == "page_first_direct":
            # page_first_direct 多一个大小为 1 的维度，reshape 时需对应带上。
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(self.layer_num, 1, self.item_bytes)
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        # 为零拷贝存储 I/O 生成 (指针列表, 每段字节数列表)。
        ptr_list = []
        rows = self._to_page_indices(indices).tolist()
        if self.layout == "layer_first":
            # layer_first：每个 page 需给出各层的独立指针（基址 + page 行偏移）。
            for row in rows:
                page_index = int(row)
                for layer_id in range(self.layer_num):
                    ptr = (
                        self.kv_buffer[layer_id].data_ptr()
                        + page_index * self.item_bytes * self.dtype.itemsize
                    )
                    ptr_list.append(ptr)
            # 每段大小为单层单 page 的字节数。
            element_size = self.item_bytes * self.dtype.itemsize
            return ptr_list, [element_size] * len(ptr_list)
        if self.layout in ["page_first", "page_first_direct"]:
            # page_first(_direct)：整页各层连续，一个 page 只需一个指针，
            # 段大小为「层数 × 单层字节数」。
            page_bytes = self.layer_num * self.item_bytes * self.dtype.itemsize
            for row in rows:
                ptr_list.append(self.kv_buffer[int(row)].data_ptr())
            return ptr_list, [page_bytes] * len(ptr_list)
        raise ValueError(f"Unsupported layout: {self.layout}")


class DeepSeekV4StateHostPool(HostKVCache):
    """V4 CompressStatePool 状态 page 行在 host 端的镜像池。

    与 DeepSeekV4PagedHostPool 类似，但镜像的对象是 V4 的「压缩状态」子池
    （CompressStatePool，如带 ring buffer 的 kv_score 状态），以 SWA（滑动窗口
    注意力）的 page 为单位组织。该池不自带分配器：它复用 SWA 侧下发的传输索引，
    因此 alloc/free/available_size 均不支持（见下方 NotImplementedError）。
    """

    def __init__(
        self,
        pool_name: str,
        state_pools: list,
        num_host_pages: int,
        swa_page_size: int,
        layout: str = "layer_first",
        device: str = "cpu",
        pin_memory: bool = True,
        allocator_type: str = "default",
    ):
        # 每层对应一个 device 侧状态子池，不允许有 None。
        if any(pool is None for pool in state_pools):
            raise ValueError(f"{pool_name} state_pools must not contain None")

        self.pool_name = pool_name
        self.state_pools = state_pools
        self.layer_num = len(state_pools)
        self.num_host_pages = num_host_pages
        self.swa_page_size = swa_page_size
        self.dtype = torch.uint8
        self.device = device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        self.page_size = swa_page_size
        self.size = num_host_pages * swa_page_size
        self.layout = layout
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = threading.RLock()

        # ring_size / state_page_bytes / device_page_views 由 _init_device_page_views
        # 从各 device 状态子池推断得出（此处先置零/空占位）。
        self.ring_size = 0
        self.state_page_bytes = 0
        self.device_page_views = []
        self.gpu_device = device
        self._init_device_page_views()
        self.size_per_token = self.state_page_bytes

        # 预估所需 host 内存并做可用性检查，预留一部分给系统避免 OOM。
        requested_bytes = self.layer_num * num_host_pages * self.state_page_bytes
        host_mem = psutil.virtual_memory()
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for V4 state pool {pool_name}. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        # 布局与 DeepSeekV4PagedHostPool 一致，只是每 page 的字节数为 state_page_bytes。
        alloc_func = ALLOC_MEMORY_FUNCS[self.gpu_device]
        self.data_refs = []
        if self.layout == "layer_first":
            # layer_first：每层一个 (page 数, state_page_bytes) 缓冲区。
            self.kv_buffer = [
                alloc_func(
                    (num_host_pages, self.state_page_bytes),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                for _ in range(self.layer_num)
            ]
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        elif self.layout == "page_first":
            # page_first：单个大缓冲区，(page, layer, state_page_bytes)。
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, self.state_page_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        elif self.layout == "page_first_direct":
            # page_first_direct：多插入大小为 1 的维度，使每 page slot 可直接寻址。
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, 1, self.state_page_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        logger.info(
            "Allocating %.2f GB host memory for V4 state pool '%s' "
            "(layers=%d, pages=%d, state_page_bytes=%d, layout=%s).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            self.state_page_bytes,
            self.layout,
        )
        # device 侧指针取自 _init_device_page_views 构造的 page 视图。
        self.device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_page_views],
            dtype=torch.uint64,
            device=self.gpu_device,
        )
        self.data_ptrs = (
            torch.tensor(
                [x.data_ptr() for x in self.data_refs],
                dtype=torch.uint64,
                device=self.gpu_device,
            )
            if self.data_refs
            else None
        )

    def _init_device_page_views(self) -> None:
        # 把各 device 状态子池的状态张量重解释为「按 page 组织的字节视图」，
        # 同时校验所有层共享一致的 ring_size 与每 page 字节数。
        expected_ring_size = None
        expected_state_page_bytes = None
        for pool in self.state_pools:
            # 取该层的状态张量（kv_score），要求内存连续以便安全地按字节重排。
            state_tensor = pool.kv_score_buffer.kv_score
            if not state_tensor.is_contiguous():
                raise ValueError(f"{self.pool_name} state tensor must be contiguous")
            # 一个 page 覆盖 ring_size 个 slot，每 slot 占 slot_bytes 字节。
            ring_size = pool.ring_size
            slot_bytes = state_tensor[0].nbytes
            state_page_bytes = ring_size * slot_bytes
            if expected_ring_size is None:
                # 以第一层为基准，记录期望的 ring_size / page 字节数与设备。
                expected_ring_size = ring_size
                expected_state_page_bytes = state_page_bytes
                self.gpu_device = state_tensor.device
            elif (
                expected_ring_size != ring_size
                or expected_state_page_bytes != state_page_bytes
            ):
                # 各层必须一致，否则无法用统一的 page 布局做批量搬运。
                raise ValueError(
                    f"{self.pool_name} state pools must share ring size and slot bytes"
                )

            # 将状态张量按 uint8 展平到 (slot 数, 每 slot 字节)。
            state_bytes = state_tensor.view(torch.uint8).reshape(
                state_tensor.shape[0], -1
            )
            # 只取能被 ring_size 整除的整数个 page 的部分（丢弃尾部不足一页的 slot）。
            usable_slots = (state_tensor.shape[0] // ring_size) * ring_size
            self.device_page_views.append(
                state_bytes[:usable_slots].reshape(-1, state_page_bytes)
            )

        self.ring_size = expected_ring_size or 0
        self.state_page_bytes = expected_state_page_bytes or 0

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        # 将 token slot 索引折算为 page 行号；要求索引数量按 SWA page 对齐。
        if indices.numel() % self.swa_page_size != 0:
            raise ValueError(
                f"{self.pool_name} transfer indices must be SWA-page-aligned, "
                f"got numel={indices.numel()}, swa_page_size={self.swa_page_size}"
            )
        return indices.reshape(-1, self.swa_page_size)[:, 0] // self.swa_page_size

    def get_size_per_token(self):
        return self.state_page_bytes

    def get_ksize_per_token(self):
        return self.state_page_bytes

    def init_kv_buffer(self):
        return self.kv_buffer

    def get_hybrid_pool_buffer(self):
        return self.kv_buffer if isinstance(self.kv_buffer, list) else [self.kv_buffer]

    def clear(self):
        # 该池不维护自己的空闲列表，无需清理（索引由 SWA 侧统一管理）。
        pass

    # 本池复用 SWA 下发的传输索引，自身没有分配器/空闲列表，
    # 因此以下三个接口直接抛出 NotImplementedError，防止被误用。
    def available_size(self):
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no allocator"
        )

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no allocator"
        )

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no free list"
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        # 将所有层的状态 page 从 device 备份到 host（D2H）。
        if host_indices is None or device_indices is None:
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        if io_backend == "kernel" and self.layout == "layer_first":
            assert self.data_ptrs is not None
            transfer_kv_all_layer_mla(
                src_layers=self.device_ptrs,
                dst_layers=self.data_ptrs,
                src_indices=device_rows,
                dst_indices=host_rows,
                item_size=self.state_page_bytes,
                num_layers=self.layer_num,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            transfer_kv_all_layer_mla_lf_pf(
                src_layers=self.device_ptrs,
                dst=self.kv_buffer,
                src_indices=device_rows,
                dst_indices=host_rows,
                item_size=self.state_page_bytes,
                dst_layout_dim=self.layer_num * self.state_page_bytes,
                num_layers=self.layer_num,
            )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=self.device_page_views,
                dst_layers=self.data_refs,
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=self.device_page_views,
                dst_ptrs=[self.kv_buffer],
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        else:
            raise ValueError(
                f"Unsupported V4 state host layout/backend: {self.layout}/{io_backend}"
            )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        # 按层将状态 page 从 host 载入 device（H2D），是 backup 的逆操作。
        if host_indices is None or device_indices is None:
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_per_layer_mla(
                src=self.data_refs[layer_id],
                dst=self.device_page_views[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                item_size=self.state_page_bytes,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            transfer_kv_per_layer_mla_pf_lf(
                src=self.kv_buffer,
                dst=self.device_page_views[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                item_size=self.state_page_bytes,
                src_layout_dim=self.layer_num * self.state_page_bytes,
            )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=[self.data_refs[layer_id]],
                dst_layers=[self.device_page_views[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[self.kv_buffer],
                dst_ptrs=[self.device_page_views[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                page_size=1,
            )
        else:
            raise ValueError(
                f"Unsupported V4 state host layout/backend: {self.layout}/{io_backend}"
            )

    def get_data_page(self, index, flat=True):
        # 取出某 page（跨所有层）的状态数据用于序列化；入参先按 SWA page 折算。
        # 逻辑与 DeepSeekV4PagedHostPool.get_data_page 相同，仅每 page 字节数不同。
        index = int(index) // self.swa_page_size
        if self.layout == "layer_first":
            data_page = torch.stack(
                [self.kv_buffer[i][index] for i in range(self.layer_num)]
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            data_page = self.kv_buffer[index]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self):
        # 全零占位 page（含所有层），形状 (layer_num, state_page_bytes) 后拉平。
        return torch.zeros(
            (self.layer_num, self.state_page_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        # get_data_page 的逆操作：把扁平 page 按 layout 还原写回对应 page 行。
        index = int(index) // self.swa_page_size
        if self.layout == "layer_first":
            data = data_page.view(self.dtype).reshape(
                self.layer_num, self.state_page_bytes
            )
            for i in range(self.layer_num):
                self.kv_buffer[i][index].copy_(data[i])
        elif self.layout == "page_first":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(
                    self.layer_num, self.state_page_bytes
                )
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(
                    self.layer_num, 1, self.state_page_bytes
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        # 为零拷贝存储 I/O 生成 (指针列表, 每段字节数列表)，规则同 paged 池：
        # layer_first 逐层给指针、每段=单层单 page；page_first(_direct) 每 page 一个
        # 指针、每段=层数×单层字节数。
        ptr_list = []
        rows = self._to_page_indices(indices).tolist()
        if self.layout == "layer_first":
            for row in rows:
                page_index = int(row)
                for layer_id in range(self.layer_num):
                    ptr = (
                        self.kv_buffer[layer_id].data_ptr()
                        + page_index * self.state_page_bytes * self.dtype.itemsize
                    )
                    ptr_list.append(ptr)
            element_size = self.state_page_bytes * self.dtype.itemsize
            return ptr_list, [element_size] * len(ptr_list)
        if self.layout in ["page_first", "page_first_direct"]:
            page_bytes = self.layer_num * self.state_page_bytes * self.dtype.itemsize
            for row in rows:
                ptr_list.append(self.kv_buffer[int(row)].data_ptr())
            return ptr_list, [page_bytes] * len(ptr_list)
        raise ValueError(f"Unsupported layout: {self.layout}")


@dataclass
class PoolEntry:
    # HostPoolGroup 中的一个成员条目：把一个 host 侧镜像池与其对应的 device 池
    # 绑定在一起，并携带该池所需的层映射与可选的驱逐/分配回调。
    name: PoolName
    host_pool: Any
    device_pool: Any
    # 将「全局 layer_id」映射到「该 device 池内的局部 layer_id」（不属于则返回 None），
    # 用于在多池混合场景下按层分派 load/backup。
    layer_mapper: Callable[[int], Optional[int]]
    # 是否作为主索引锚点：组内以锚点池的 layout/page_size/分配器等作为对外口径。
    is_primary_index_anchor: bool = False
    # Optional eviction callbacks for auto-alloc in HybridCacheController.
    # host_evict_fn(n): evict n slots from the host pool (used by write()).
    # device_evict_fn(n): evict n slots from the device pool (used by load()).
    host_evict_fn: Optional[Callable] = None
    device_evict_fn: Optional[Callable] = None
    # Optional alloc/free overrides for the device side, used by
    # _resolve_pool_transfers_allocation. Set when entry.device_pool is the
    # raw KV/state pool (layout) rather than an allocator (e.g. SWA/Mamba,
    # where alloc lives on a separate allocator object).
    # When None, fall back to entry.device_pool.alloc/free.
    device_alloc_fn: Optional[Callable] = None
    device_free_fn: Optional[Callable] = None


class HostPoolGroup:
    """把多个 host 侧镜像池（PoolEntry）聚合为一个「池组」统一对外暴露。

    混合模型（如带 Mamba 状态、SWA 窗口、索引器的模型）会同时用到多个缓存池。
    该组选定其中一个作为「主索引锚点」(anchor)：对外的 layout/page_size/分配器/
    size_per_token 等属性、以及 alloc/free/available_size 等索引操作，都直接委托给
    锚点池——因为所有池共享同一套 page 索引空间，只需由锚点池统一分配即可。
    而 clear 等需要作用到全部池的操作，则遍历所有 entry 逐个执行。
    """

    def __init__(self, entries: list[PoolEntry]):
        if not entries:
            raise ValueError("HostPoolGroup requires at least one pool entry.")
        self.entries = entries
        # 按池名建立索引，便于通过 PoolName 直接取到对应 entry / host_pool。
        self.entry_map = {entry.name: entry for entry in entries}
        # 选出主索引锚点：优先取标记了 is_primary_index_anchor 的池，
        # 否则退化为第一个 entry。
        self.anchor_entry = next(
            (entry for entry in entries if entry.is_primary_index_anchor),
            entries[0],
        )

        # 对外的布局/页大小/设备/容量口径均以锚点池为准。
        self.layout = self.anchor_entry.host_pool.layout
        self.page_size = self.anchor_entry.host_pool.page_size
        self.device = self.anchor_entry.host_pool.device
        self.size = self.anchor_entry.host_pool.size

    # 以下属性/方法均委托给锚点池，保证池组对外表现得像单个池。
    @property
    def kv_buffer(self):
        return self.anchor_entry.host_pool.kv_buffer

    @property
    def size_per_token(self):
        return self.anchor_entry.host_pool.size_per_token

    @property
    def allocator(self):
        return self.anchor_entry.host_pool.allocator

    @property
    def dtype(self):
        return self.anchor_entry.host_pool.dtype

    @property
    def start_layer(self):
        return self.anchor_entry.host_pool.start_layer

    @property
    def end_layer(self):
        return self.anchor_entry.host_pool.end_layer

    def get_ksize_per_token(self):
        return self.anchor_entry.host_pool.get_ksize_per_token()

    def get_pool(self, name: PoolName):
        # 按池名取出组内某个具体的 host 池（如单独访问 Mamba/SWA 池）。
        return self.entry_map[name].host_pool

    def get_page_buffer_meta(self, indices):
        return self.anchor_entry.host_pool.get_page_buffer_meta(indices)

    def clear(self) -> None:
        # clear 需作用到组内每个池，因此遍历全部 entry 逐个清理。
        for entry in self.entries:
            entry.host_pool.clear()

    def available_size(self):
        return self.anchor_entry.host_pool.available_size()

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        # 索引分配统一由锚点池负责：组内各池共享同一套 page 索引空间。
        return self.anchor_entry.host_pool.alloc(need_size)

    def free(self, indices: torch.Tensor) -> int:
        return self.anchor_entry.host_pool.free(indices)

    def get_data_page(self, index, flat: bool = True):
        return self.anchor_entry.host_pool.get_data_page(index, flat)

    def get_dummy_flat_data_page(self):
        return self.anchor_entry.host_pool.get_dummy_flat_data_page()

    def set_from_flat_data_page(self, index: int, data_page) -> None:
        return self.anchor_entry.host_pool.set_from_flat_data_page(index, data_page)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        pool_transfers: Optional[list] = None,
    ) -> None:
        # 按层将该池组的数据从 host 载入 device（H2D）。分两步：
        # 先搬锚点（KV）池，再按 pool_transfers 搬各附属池（Mamba/SWA/索引器等）。
        # 1. 锚点（KV）池传输
        anchor = self.anchor_entry
        # 把全局 layer_id 映射为锚点池内的局部层号；不属于本池则跳过。
        local_layer_id = anchor.layer_mapper(layer_id)
        if local_layer_id is not None and host_indices.numel() > 0:
            anchor.host_pool.load_to_device_per_layer(
                anchor.device_pool,
                host_indices,
                device_indices,
                local_layer_id,
                io_backend,
            )

        # 2. 附属池传输：每个 transfer 描述一个附属池本次要搬的索引。
        for transfer in pool_transfers or []:
            entry = self.entry_map.get(transfer.name)
            # 组内不存在该池、或本次无索引需搬，则跳过。
            if entry is None or transfer.host_indices is None:
                continue
            local_layer_id = entry.layer_mapper(layer_id)
            if local_layer_id is None:
                continue
            entry.host_pool.load_to_device_per_layer(
                entry.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                local_layer_id,
                io_backend,
            )

    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
        pool_transfers: Optional[list] = None,
    ) -> None:
        # 将该池组所有层的数据从 device 备份到 host（D2H），是 load 的逆操作。
        # 1. 锚点（KV）池备份
        self.anchor_entry.host_pool.backup_from_device_all_layer(
            self.anchor_entry.device_pool,
            host_indices,
            device_indices,
            io_backend,
        )
        # 2. 附属池备份：按各 transfer 描述逐池搬回。
        for transfer in pool_transfers or []:
            entry = self.entry_map.get(transfer.name)
            if entry is None or transfer.host_indices is None:
                continue
            entry.host_pool.backup_from_device_all_layer(
                entry.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                io_backend,
            )


class DSAIndexerPoolHost(HostKVCache):
    """仅承载 DSA（DeepSeek Sparse Attention）索引缓冲区的 host 池。

    DSA 在标准 MLA KV 之外，额外维护一份「索引器」缓冲（index_k 及其量化 scale），
    用于稀疏注意力的候选选取。该池只镜像这份索引数据，其 slot 索引空间与作为锚点的
    MLA host 池（anchor_host）完全对齐——即用同一套 page 索引，从而 KV 与索引能一一对应。
    """

    device_pool: DSATokenToKVPool

    def __init__(
        self,
        device_pool: DSATokenToKVPool,
        anchor_host: MLATokenToKVPoolHost,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        # 复用锚点 MLA host 池的 page_size 与容量，保证索引与 KV 的 slot 对齐。
        self.page_size = anchor_host.page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.dtype = device_pool.store_dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.layer_num = device_pool.layer_num

        self.index_head_dim = device_pool.index_head_dim
        self.indexer_quant_block_size = device_pool.quant_block_size
        self.indexer_dtype = DSATokenToKVPool.index_k_with_scale_buffer_dtype
        # 每 token 的索引元素数 = index_k 本体 + 量化 scale：
        # 每 quant_block_size 个元素配 1 个 scale，scale 占 4 字节（故除以块大小再乘 4）。
        self.indexer_size_per_token = (
            self.index_head_dim
            + self.index_head_dim // self.indexer_quant_block_size * 4
        )
        self.size = anchor_host.size
        self.page_num = anchor_host.page_num

        # 单层单 page 的索引字节数（步长）。
        self.indexer_page_stride_size = (
            self.indexer_size_per_token * self.page_size * self.indexer_dtype.itemsize
        )
        # page-first 布局下，一整页（跨所有层）的字节维度。
        self.indexer_layout_dim = self.indexer_page_stride_size * self.layer_num
        self.indexer_page_num = (self.size + self.page_size + 1) // self.page_size
        # 每 token 跨所有层的索引字节数。
        self.size_per_token = (
            self.indexer_size_per_token * self.layer_num * self.indexer_dtype.itemsize
        )

        # 预估索引缓冲区总内存并做可用性检查，预留部分内存避免 OOM。
        buf_elem_size = self.page_num * self.layer_num * self.indexer_page_stride_size
        requested_bytes = buf_elem_size * self.indexer_dtype.itemsize
        host_mem = psutil.virtual_memory()
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for DSA indexer hierarchical cache. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )
        logger.info(
            "Allocating %.2f GB host memory for DSA indexer (layout=%s).",
            requested_bytes / 1e9,
            layout,
        )
        self.init_kv_buffer()
        self.lock = threading.RLock()
        self.clear()

    def get_size_per_token(self):
        return (
            self.indexer_size_per_token * self.layer_num * self.indexer_dtype.itemsize
        )

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def init_kv_buffer(self):
        # 分配 host 侧索引缓冲，并把 device 侧索引缓冲的裸指针打包成 uint64 张量。
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_pool.index_k_with_scale_buffer],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        if self.layout == "layer_first":
            # layer_first：外层为层维，(layer, page 数, 单层单页字节)。
            self.index_k_with_scale_buffer = alloc_func(
                (self.layer_num, self.indexer_page_num, self.indexer_page_stride_size),
                dtype=self.indexer_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            # 保存各层视图与其指针张量，供拷贝 kernel 使用。
            self.index_k_data_refs = [
                self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
            ]
            self.index_k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in self.index_k_data_refs],
                dtype=torch.uint64,
                device=self.device_pool.device,
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            # page_first(_direct)：外层为 page 维，(page 数, layer, 1, 单层单页字节)，
            # 中间大小为 1 的维度使每个 page slot 可直接寻址。
            self.index_k_with_scale_buffer = alloc_func(
                (
                    self.indexer_page_num,
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                ),
                dtype=self.indexer_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_hybrid_pool_buffer(self):
        return [self.index_k_with_scale_buffer]

    def _get_indexer_page_indices(self, host_indices, device_indices):
        # 把 token slot 索引折算为 page 行号（host / device 两侧同时处理）。
        if host_indices.numel() == 0:
            return host_indices, device_indices
        # 索引器传输要求按 page 对齐。
        if host_indices.numel() % self.page_size != 0:
            raise ValueError(
                "Index buffer transfer expects page-aligned indices for DSA."
            )
        host_page_indices = (
            host_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        device_page_indices = (
            device_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        return host_page_indices, device_page_indices

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        # 按层将索引缓冲从 host 载入 device（H2D）。
        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        # kernel 路径要求每 page 步长按 8 字节对齐；否则退回 direct 拷贝。
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.index_k_with_scale_buffer[layer_id],
                    dst=device_pool.index_k_with_scale_buffer[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_mla_pf_lf(
                    src=self.index_k_with_scale_buffer,
                    dst=device_pool.index_k_with_scale_buffer[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=layer_id,
                    item_size=self.indexer_page_stride_size,
                    src_layout_dim=self.indexer_layout_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.index_k_with_scale_buffer[layer_id]],
                    dst_layers=[device_pool.index_k_with_scale_buffer[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.index_k_with_scale_buffer],
                    dst_ptrs=[device_pool.index_k_with_scale_buffer[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=layer_id,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        # 一次性将所有层的索引缓冲从 device 备份到 host（D2H），是 load 的逆操作。
        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_all_layer_mla(
                    src_layers=self.index_k_device_ptrs,
                    dst_layers=self.index_k_data_ptrs,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                    num_layers=self.layer_num,
                )
            elif self.layout == "page_first":
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=self.index_k_device_ptrs,
                    dst=self.index_k_with_scale_buffer,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                    dst_layout_dim=self.indexer_layout_dim,
                    num_layers=self.layer_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.index_k_with_scale_buffer,
                    dst_layers=self.index_k_data_refs,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.index_k_with_scale_buffer,
                    dst_ptrs=[self.index_k_with_scale_buffer],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        # 取出某 page（跨所有层）的索引数据用于序列化；入参先折算为 page 行号。
        page_idx = int(index) // self.page_size
        if self.layout == "layer_first":
            # layer_first：层维在前，取所有层该 page 的切片。
            data_page = self.index_k_with_scale_buffer[:, page_idx : page_idx + 1, :]
        elif self.layout in ["page_first", "page_first_direct"]:
            # page_first(_direct)：page 维在前，直接取该 page 的整块。
            data_page = self.index_k_with_scale_buffer[page_idx : page_idx + 1, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        # 全零占位 page（含所有层），形状 (layer_num, 单层单页字节) 后拉平。
        return torch.zeros(
            (self.layer_num, self.indexer_page_stride_size),
            dtype=self.indexer_dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        # get_data_page 的逆操作：把扁平 page 按 layout reshape 后写回对应 page 行。
        page_idx = int(index) // self.page_size
        if self.layout == "layer_first":
            self.index_k_with_scale_buffer[:, page_idx : page_idx + 1, :] = (
                data_page.reshape(
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                )
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            self.index_k_with_scale_buffer[page_idx : page_idx + 1, :, :, :] = (
                data_page.reshape(
                    1,
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """零拷贝存储 I/O 所需的元数据（仅支持 page-first 布局）。"""
        assert len(indices) % self.page_size == 0
        # 只有 page-first 布局下每个 page 才是连续可寻址的，才能提供零拷贝裸指针。
        if self.layout not in ["page_first", "page_first_direct"]:
            raise ValueError(f"Unsupported layout: {self.layout}")
        ptr_list = []
        indices = indices.tolist()
        # 一整页（跨所有层）的字节步长。
        page_stride_bytes = (
            self.layer_num * self.indexer_page_stride_size * self.indexer_dtype.itemsize
        )
        base_ptr = self.index_k_with_scale_buffer.data_ptr()
        # 每个 page 的指针 = 基址 + page 行号 × 页步长；每段大小即页步长。
        for i in range(0, len(indices), self.page_size):
            page_index = int(indices[i]) // self.page_size
            ptr_list.append(base_ptr + page_index * page_stride_bytes)
        return ptr_list, [page_stride_bytes] * len(ptr_list)
