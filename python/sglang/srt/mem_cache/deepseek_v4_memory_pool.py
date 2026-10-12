from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Literal, NamedTuple, Optional, Tuple

import torch

from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.kernels.ops.attention.dsv4 import (
    clear_unaccepted_c128_draft_states,
    fused_k_norm_rope_flashmla,
    fused_store_cache,
)
from sglang.kernels.ops.attention.dsv4 import (
    index_buf_accessor as dsv4_index_buf_accessor,
)
from sglang.kernels.ops.attention.dsv4.index_buf_accessor import NopeFp8RopeBf16Pack
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.runtime_context import get_exec, get_spec
from sglang.srt.utils import ceil_div, is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

ONLINE_C128 = not _is_hip and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


def get_compress_state_ring_size(
    compress_ratio: int, is_speculative: bool = False
) -> int:
    assert compress_ratio in [4, 128], f"Unsupported {compress_ratio = }"
    # 在线（online）C128 每个下标只保留一份 (max, sum, kv) 汇总状态，
    # 而不是 128 槽的原始 token ring，因此 ring_size 收缩为 1。
    # 在线模式目前与投机解码（MTP）不兼容。
    if compress_ratio == 128 and ONLINE_C128:
        if is_speculative and not envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            raise AssertionError("online c128 does not support MTP")
        return 1
    if is_speculative:
        return 16 if compress_ratio == 4 else 256
    else:
        return 8 if compress_ratio == 4 else 128


def get_compress_state_write_pad(compress_ratio: int, ring_size: int) -> int:
    """返回该 ring 能容纳的最大草稿（draft）token 数。

    语义与 `c_plan.cuh` 中的 `mtp_pad` 一致（该上界在那里推导），供投机解码
    预留写入空间。非投机场景下 ring 宽度恰好等于一个窗口，此时返回 0。

    参数：
        compress_ratio: 压缩率，只能是 4 或 128。
        ring_size: 该压缩率对应的 ring 长度（见 `get_compress_state_ring_size`）。

    返回：
        可容纳的草稿 token 数（int）；ring 不足一个窗口时返回 0。

    示例：
        >>> get_compress_state_write_pad(4, 8)      # 非投机 C4：窗口 4×2 = 8
        0
        >>> get_compress_state_write_pad(4, 16)     # 投机 C4：16 - 8 + 2
        10
        >>> get_compress_state_write_pad(128, 256)  # 投机 C128：256 - 128 + 2
        130
    """
    window_size = compress_ratio * (2 if compress_ratio == 4 else 1)
    return ring_size - window_size + 2 if ring_size > window_size else 0


class DeepSeekV4SingleKVPool(KVCache):
    """DeepSeek-V4 的单一 KV 物理池：SWA / C4 / C128 三类子池共用本实现。

    【定位】
    父池 `DeepSeekV4TokenToKVPool` 按压缩率给每类层各建一个本池实例；本类只负责
    「一块按字节紧凑打包的 K 缓冲」，不决定容量、也不负责层号换算：

    - SWA 子池：页大小 `swa_page_size`（默认 256），覆盖本 stage 内的全部层；
    - C4 子池：页大小 `page_size // 4`（默认 64），只覆盖 `compress_ratio=4` 的层；
    - C128 子池：页大小 `page_size // 128`（默认 2），只覆盖 `compress_ratio=128` 的层。

    启用 HiSparse 时，C4 子池由子类 `HiSparseC4DevicePool` 承担（额外地做
    full 槽位 → device 热池槽位的下标翻译）。

    【每 token 584 字节】缓冲以 `uint8` 承载，逐段拼接：

    - nope：`qk_nope_head_dim` = 448 字节（FP8，每元素 1 字节）；
    - rope：`qk_rope_head_dim` = 64，按 BF16 存 → 128 字节；
    - nope FP8 scale：`448 // 64` = 7 字节（UE8M0，每 64 通道 1 个）；
    - 对齐填充：`scale_pad` = 1 字节。

    合计 448 + 128 + 7 + 1 = 584 字节；每页字节再向上取整到 `576` 的整数倍
    （576 是 **页**级对齐单位，不是每 token 的字节数）。

    【为什么没有 V 缓冲】
    该模型的 K 与 V 是同一个「头间共享」向量，因此这里只存一份 K：
    `get_value_buffer` / `get_kv_buffer` / `set_kv_buffer` 一律不支持，
    注意力侧只调用 `get_key_buffer`。

    【存储 dtype】
    `dtype` 恒为 FP8，基类据此把 `store_dtype` 设为 `torch.uint8`（FP8 不支持
    直接 `index_put`），所以缓冲实际是「字节张量」，读取时再 `view(dtype)`
    重解释为 FP8。

    参数：
        size: 本子池槽位数（SWA 为原始 token 数；C4 / C128 为压缩后条目数）。
        page_size: 本子池页大小（每页槽位数）。
        dtype: KV 计算 dtype（恒为 FP8）。
        qk_nope_head_dim: nope 维（448），FP8 存储。
        qk_rope_head_dim: RoPE 维（64），BF16 存储。
        layer_num: 本子池覆盖的层数（**本 stage 内**）。
        device: 缓冲所在设备。
        enable_memory_saver: 是否接入显存节省适配器。
        start_layer / end_layer: 全局层区间；父池构池时不传，恒为 0 / layer_num - 1。

    示例：
        >>> pool = DeepSeekV4SingleKVPool(
        ...     size=8192, page_size=64, dtype=torch.float8_e4m3fn,
        ...     qk_nope_head_dim=448, qk_rope_head_dim=64,
        ...     layer_num=4, device="cuda:0", enable_memory_saver=False,
        ... )
        >>> len(pool.kv_buffer)         # 每层一份缓冲
        4
        >>> pool.get_bytes_per_token()  # 每 token 584 字节
        584
        >>> pool.kv_buffer[0].shape     # [页数, 按 576 对齐后的页字节]
        torch.Size([129, 37440])
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        """构造单个 KV 子池：先让基类确定 size / page_size / store_dtype 等，再记录
        打包布局常量并分配按字节的缓冲。

        参数：见类文档字符串；其中 `layer_num` 是**本 stage 内**的层数。

        返回：
            无（返回实例本身）。

        示例：
            >>> p = DeepSeekV4SingleKVPool(
            ...     1024, 2, torch.float8_e4m3fn, 448, 64, 1, "cuda", False
            ... )
            >>> p.quantize_block_size, p.rope_storage_dtype, p.scale_pad
            (64, torch.bfloat16, 1)
            >>> p.bytes_per_page_padded  # 2 × 584 = 1168 → 向上取整到 576 的倍数
            1728
        """
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim

        # ---- 打包布局常量（必须与 get_bytes_per_token / create_buffer 保持一致）----
        self.scale_pad = 1  # nope scale 之后补 1 字节，把每 token 行宽对齐到 8 的倍数
        self.quantize_block_size = 64  # nope 每 64 通道共享 1 个 FP8 scale（448/64 = 7）
        self.rope_storage_dtype = torch.bfloat16  # rope 恒按 BF16 存（64 → 128 字节）
        self.k_with_scale_buffer_dtype = torch.int8  # 预留字段，本类内部未再读取
        self._create_buffers()

    def _create_buffers(self):
        """按层分配 KV 字节缓冲（由 `__init__` 调用）。

        每层一张 `[页数, bytes_per_page_padded]` 的 `uint8` 张量，页数取
        `ceil((size + page_size + 1) / page_size)`：`+ page_size` 为尾部跨页写入留
        一整页余量，末尾再 `+ 1` 保证紧跟最后一个有效槽位之后的写入不越界。

        分配包在 `GPU_MEMORY_TYPE_KV_CACHE` 的 memory-saver 区域内，并在可用时
        复用 PD + NVLink 的自定义显存池。

        参数：无。

        返回：
            None（结果就地写入 `self.kv_buffer`）。

        示例：
            >>> pool.kv_buffer[0].dtype
            torch.uint8
            >>> len(pool.kv_buffer) == pool.layer_num
            True
            >>> pool.kv_buffer[0].shape
            torch.Size([129, 37440])
        """
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.kv_buffer = [
                    self.create_buffer(
                        num_pages=(self.size + self.page_size + 1) // self.page_size,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_bytes_per_token(self) -> int:
        """返回「每 token 的字节数」，即本池打包布局的一行宽度（恒为 584）。

        逐项相加（单位均为字节）：

        - `qk_nope_head_dim` = 448：nope，FP8 每元素 1 字节；
        - `qk_rope_head_dim × 2` = 128：rope，BF16 每元素 2 字节；
        - `qk_nope_head_dim // quantize_block_size` = 7：nope FP8 scale；
        - `scale_pad` = 1：对齐填充。

        注意这里的 `quantize_block_size = 64` 与索引器池的 `quant_block_size = 128`
        不同，二者不要混用。

        参数：无。

        返回：
            每 token 字节数（int），当前实现恒为 584。

        示例：
            >>> pool.get_bytes_per_token()
            584
        """
        dim_per_token = (
            self.qk_nope_head_dim
            + self.qk_rope_head_dim * self.rope_storage_dtype.itemsize
            + self.qk_nope_head_dim // self.quantize_block_size
            + self.scale_pad
        )
        return dim_per_token

    def create_buffer(self, *, num_pages: int):
        """新建一层的 KV 字节缓冲，并顺带推导本池的字节 / 页常量。

        副作用（重要）：会把 `kv_cache_total_dim`（= 每 token 字节数）与
        `bytes_per_page_padded` 写回 `self`；HiSparse 协调器用
        `item_size_bytes = kv_cache_total_dim × store_dtype.itemsize` 估算
        host ↔ device 的搬运粒度。

        参数：
            num_pages: 要分配的页数（由 `_create_buffers` 依据 size / page_size 推出）。

        返回：
            `[num_pages, bytes_per_page_padded]` 的 `uint8` 全零张量。

        异常：
            AssertionError: 布局不是 584 字节 / token，或存储 dtype 不是 `uint8`
                （即 `dtype` 不是 FP8）。

        示例：
            >>> buf = pool.create_buffer(num_pages=3)
            >>> buf.shape, buf.dtype
            (torch.Size([3, 37440]), torch.uint8)
            >>> pool.kv_cache_total_dim  # 顺带写入：每 token 字节数
            584
        """
        bytes_per_token = self.get_bytes_per_token()
        self.kv_cache_total_dim = bytes_per_token
        bytes_per_page_non_padded = self.page_size * bytes_per_token
        self.bytes_per_page_padded = ceil_div(bytes_per_page_non_padded, 576) * 576

        # 布局自检：448(FP8 nope) + 64×2(BF16 rope) + 7(FP8 scale) + 1(pad) = 584 B/token。
        assert bytes_per_token == 448 + 64 * 2 + 8, (
            "DSV4 KV layout: qk_nope_head_dim FP8 (448) + qk_rope_head_dim BF16 "
            "(64*2) + nope FP8 scales + scale_pad = 584 bytes/token"
            "（DSV4 KV 布局：448 字节 FP8 nope + 128 字节 BF16 rope + 7 字节 FP8 scale "
            "+ 1 字节对齐填充 = 每 token 584 字节）"
        )
        # FP8 不支持直接 index_put，基类会把 store_dtype 置为 uint8；此处确认这一点。
        assert self.store_dtype == torch.uint8

        return torch.zeros(
            num_pages,
            self.bytes_per_page_padded,
            dtype=self.store_dtype,
            device=self.device,
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ):
        """把**已量化好**的三段 K 数据写入指定槽位（走 `SetKAndS` kernel）。

        输入：
            layer_id: 本 stage 内的局部层号（父池已把全局层号换算好）。
            loc: `[T]` 的一维下标张量（int32 / int64），本批要写的槽位。
            cache_nope_fp8_rope_bf16_pack: `NopeFp8RopeBf16Pack`，含
                `k_nope_fp8` `[T, 448]`（FP8）、`k_rope_bf16` `[T, 64]`（BF16）、
                `scale_k_nope_ue8m0` `[T, 7]`（uint8）。

        返回：
            None（就地写入 `self.kv_buffer[layer_id]`）。

        示例：
            >>> loc = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
            >>> pack = NopeFp8RopeBf16Pack(
            ...     k_nope_fp8=torch.zeros(3, 448, dtype=torch.float8_e4m3fn, device="cuda"),
            ...     k_rope_bf16=torch.zeros(3, 64, dtype=torch.bfloat16, device="cuda"),
            ...     scale_k_nope_ue8m0=torch.zeros(3, 7, dtype=torch.uint8, device="cuda"),
            ... )
            >>> pool.set_key_buffer(0, loc, pack) is None  # 槽位 0/1/2 被覆盖
            True
        """
        dsv4_index_buf_accessor.SetKAndS.execute(
            pool=self,
            buf=self.kv_buffer[layer_id],
            loc=loc,
            nope_fp8_rope_bf16_pack=cache_nope_fp8_rope_bf16_pack,
        )

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        """把**未量化**的 BF16 K 一次性量化 + 打包写入（融合 kernel）。

        与 `set_key_buffer` 的分工：这里输入是 BF16 原值，FP8 量化、UE8M0 scale
        计算、rope 的 BF16 拷贝与分页落位都在 kernel 内完成；`page_size` 由池自身
        提供，用于换算页内偏移。

        输入：
            layer_id: 本 stage 内的局部层号。
            loc: `[T]` 的槽位下标张量。
            cache_k: BF16 张量 `[T, qk_nope_head_dim + qk_rope_head_dim]`
                （典型 `[T, 512]`：前 448 列 nope、后 64 列 rope），
                必须是「已归一化、已加 RoPE、尚未量化」的 K。

        返回：
            None。

        示例：
            >>> cache_k = torch.randn(5, 512, dtype=torch.bfloat16, device="cuda")
            >>> loc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device="cuda")
            >>> pool.set_key_buffer_fused(0, loc, cache_k) is None
            True
        """
        return fused_store_cache(
            input=cache_k,
            cache=self.kv_buffer[layer_id],
            indices=loc,
            page_size=self.page_size,
            type="flashmla",
        )

    def get_key_buffer(self, layer_id: int):
        """取出指定层的 K 缓冲（FP8 视图），供注意力 kernel 按页读取。

        这里同时承担「取 K」与「取 KV」两种用途：本池只存一份共享向量，
        基类的 `get_value_buffer` 已改为抛异常。

        参数：
            layer_id: 本 stage 内的局部层号（**不是**全局层号）。

        返回：
            `[页数, bytes_per_page_padded]` 的 FP8 张量（由 `uint8` 缓冲
            `view(dtype)` 重解释；HUZ 平台上为 `float8_e4m3fnuz`）。

        说明：
            当前 `dtype` 恒为 FP8、`store_dtype = uint8`，两者不等，因此实际总是
            走第一个分支；`start_layer` 缺省为 0（父池构池时未传），所以
            `layer_id - start_layer` 与 `layer_id` 等价。两个分支的差别只在
            子类显式传入 `start_layer` 时才体现。

        示例：
            >>> buf = pool.get_key_buffer(0)
            >>> buf.dtype
            torch.float8_e4m3fn
            >>> buf.shape
            torch.Size([129, 37440])
        """
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id]

    def set_kv_buffer(self, *args, **kwargs) -> None:
        """本池不支持写入 V：K 与 V 共用同一份数据，请改用 `set_key_buffer` /
        `set_key_buffer_fused`。

        参数：
            *args / **kwargs: 仅为兼容基类签名，任意传入都会失败。

        返回：
            无返回值——直接抛 `NotImplementedError`。
        """
        raise NotImplementedError(
            "DeepSeekV4SingleKVPool 不存 V（K/V 共用同一份数据），"
            "请改用 set_key_buffer / set_key_buffer_fused。"
        )

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """本池只存 K，取 V 请改用 `get_key_buffer`。

        参数：
            layer_id: 本 stage 内局部层号（此处仅用于保持签名兼容）。

        返回：
            无返回值——直接抛 `NotImplementedError`。
        """
        raise NotImplementedError(
            "本池只存 K（K/V 共用同一份数据），请改用 get_key_buffer。"
        )

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """本池没有独立的 (K, V) 两个缓冲，请改用 `get_key_buffer`。

        参数：
            layer_id: 本 stage 内局部层号（此处仅用于保持签名兼容）。

        返回：
            无返回值——直接抛 `NotImplementedError`。
        """
        raise NotImplementedError(
            "本池没有独立的 (K, V) 缓冲（K/V 共用同一份数据），"
            "请改用 get_key_buffer。"
        )


class HiSparseC4DevicePool(DeepSeekV4SingleKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.compress_ratio = 4

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        mask = (full_indices + 1) % self.compress_ratio == 0
        compressed_indices = full_indices[mask] // self.compress_ratio
        return compressed_indices

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(
            self.translate_loc_from_full_to_compressed(full_indices)
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().set_key_buffer_fused(layer_id, loc, cache_k)

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseC4DevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseC4DevicePool does not support load_cpu_copy")


class DeepSeekV4IndexerPool(KVCache):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.index_head_dim = index_head_dim
        self.use_fp4_indexer = get_exec().kernel.enable_deepseek_v4_fp4_indexer

        self._create_buffer()

    def get_bytes_per_token(self) -> int:
        if self.use_fp4_indexer:
            return self.index_head_dim // 2 + 4
        return self.index_head_dim + 4

    def _create_buffer(self):
        page_bytes = self.page_size * self.get_bytes_per_token()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.index_k_with_scale_buffer = [
                    torch.zeros(
                        (self.size + self.page_size + 1) // self.page_size,
                        page_bytes,
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_k_with_scale_buffer[layer_id]

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        buf = self.index_k_with_scale_buffer[layer_id]
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def set_index_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            indices=loc,
            page_size=self.page_size,
            type="indexer",
        )

    def set_index_fp4(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            store_fp4_index_k_cache,
        )

        return store_fp4_index_k_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            loc=loc,
            page_size=self.page_size,
        )


class DeepSeekV4LayerItem(NamedTuple):
    compress_ratio: Literal[0, 4, 128]
    compress_layer_id: int
    compress_kv_pool: Optional[DeepSeekV4SingleKVPool] = None


# The following kv pool follows ATOM's unified_kv kernel layout.
class DeepSeekV4UnifiedKVPool:
    """
    Layout:
    unified_kv[L]: ``[swa_pages + padded_compress_rows, head_dim]`` bf16
    - rows ``[0, swa_pages)``   = SWA ring (``req_pool_indices * swa_window + pos % swa_window``)
    - rows ``[swa_pages, ...)`` = compressed (``swa_pages + page_index``)
    """

    K_PER_BLOCK = {0: 0, 4: 32, 128: 1}

    def __init__(
        self,
        *,
        stage_ratios: List[int],
        num_slots: int,
        num_blocks: int,
        page_size: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        device: str,
        memory_saver_adapter,
        custom_mem_pool,
        swa_ring_size: int,
    ):
        self.swa_ring_size = swa_ring_size
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.num_slots = num_slots
        self.swa_pages = num_slots * self.swa_ring_size
        self.num_blocks = num_blocks
        self.page_size = page_size
        self.k_per_block = dict(self.K_PER_BLOCK)

        bufs = []
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(custom_mem_pool)
                if custom_mem_pool
                else nullcontext()
            ):
                for ratio in stage_ratios:
                    # Pad by one extra page. The KV pool reserves a null slot
                    # (token indices run 1..size).
                    compress_rows = self.num_blocks * self.k_per_block[ratio]
                    rows_per_page = self.page_size // ratio if ratio else 0
                    padded_compress_rows = compress_rows + rows_per_page
                    bufs.append(
                        torch.zeros(
                            self.swa_pages + padded_compress_rows,
                            self.head_dim,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                    )
        self.kv_buffer = bufs

    def get_unified_kv(self, local_layer_id: int) -> torch.Tensor:
        return self.kv_buffer[local_layer_id]

    def get_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs = [b.data_ptr() for b in self.kv_buffer]
        data_lens = [b.nbytes for b in self.kv_buffer]
        item_lens = [b[0].nbytes for b in self.kv_buffer]
        return data_ptrs, data_lens, item_lens


class DeepSeekV4TokenToKVPool(BaseSWAKVPool):

    def __init__(
        self,
        max_num_reqs: int,
        swa_size: int,
        c4_size: int,
        c128_size: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        page_size: int,
        swa_page_size: int,
        dtype: torch.dtype,
        c4_state_dtype: torch.dtype,
        c128_state_dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        indexer_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        compression_ratios: List[int],
        sliding_window: int = 128,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_hisparse: bool = False,
        online_mtp_max_draft_tokens: int = 0,
        num_req_slots: Optional[int] = None,
    ):
        super().__init__(
            swa_size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        c4_logical_size = c128_size * 32

        logger.info(
            "Initialize DeepSeekV4TokenToKVPool with "
            f"{max_num_reqs=} {swa_size=} {c4_size=} "
            f"{c4_logical_size=} {c128_size=} "
            f"{c4_state_pool_size=} {c128_state_pool_size=}"
        )

        self.max_num_reqs = max_num_reqs
        # SWA ring needs one slot per addressable req_pool_idx. PD decode inflates
        # req_to_token past max_num_reqs (pre-alloc), so the caller passes the real
        # capacity; sizing as max_num_reqs+1 overflows ("length out of range").
        self.num_req_slots = (
            num_req_slots if num_req_slots is not None else max_num_reqs + 1
        )
        self.c4_size = c4_size
        self.c4_logical_size = c4_logical_size
        self.c128_size = c128_size
        self.c4_state_pool_size = c4_state_pool_size
        c128_ring_size = self.get_ring_size(128)
        if ONLINE_C128:
            # Request-scoped online C128 state is indexed by req_pool_idx.
            # PD decode can allocate pre-transfer slots beyond
            # max_num_reqs, so size to the actual req_to_token row count.
            c128_state_pool_size = max(c128_state_pool_size, self.num_req_slots)
        else:
            # Offline C128 keeps a per-request raw state ring.
            c128_state_pool_size = max(
                c128_state_pool_size, self.num_req_slots * c128_ring_size
            )
        self.c128_state_pool_size = c128_state_pool_size
        self.c4_state_dtype = c4_state_dtype
        self.c128_state_dtype = c128_state_dtype
        self.compression_ratios = compression_ratios
        self.online_mtp_max_draft_tokens = online_mtp_max_draft_tokens
        self.online_c128_state_num_req_slots = c128_state_pool_size
        self.online_c128_mtp_pending_seq_lens: Optional[torch.Tensor] = None
        if ONLINE_C128 and envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            self.online_c128_mtp_pending_seq_lens = torch.empty(
                self.online_c128_state_num_req_slots, dtype=torch.int64, device=device
            )

        # Determine this PP stage's absolute layer range
        if (
            start_layer is not None
            and end_layer is not None
            and len(compression_ratios) >= end_layer
        ):
            self._stage_start = start_layer
            self._stage_end = end_layer
        else:
            self._stage_start = 0
            self._stage_end = len(compression_ratios)
        stage_ratios = compression_ratios[self._stage_start : self._stage_end]

        assert page_size % swa_page_size == 0
        self.sliding_window = sliding_window

        self.swa_size = swa_size
        self.swa_window_size = swa_page_size
        self.swa_page_size = swa_page_size
        self.scale_pad = 1

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.indexer_head_dim = indexer_head_dim

        stage_layer_num = len(stage_ratios)
        c4_layer_num = sum(1 for r in stage_ratios if r == 4)
        c128_layer_num = sum(1 for r in stage_ratios if r == 128)
        c4_page_size = page_size // 4
        c128_page_size = page_size // 128

        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        self._unified_kv = is_unified_kv_triton()

        if self._unified_kv:
            self.swa_kv_pool = None
            self.c4_kv_pool = None
            self.c128_kv_pool = None
            spec_extra = (
                (get_spec().speculative_num_draft_tokens - 1)
                if get_spec().speculative_algorithm is not None
                else 0
            )
            self.unified_kv_pool = DeepSeekV4UnifiedKVPool(
                stage_ratios=stage_ratios,
                num_slots=self.num_req_slots,
                num_blocks=self.c128_size,
                page_size=page_size,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                device=device,
                memory_saver_adapter=self.memory_saver_adapter,
                custom_mem_pool=self.custom_mem_pool,
                swa_ring_size=self.sliding_window + spec_extra,
            )

            self.unified_swa_window = self.sliding_window
            self.unified_swa_ring_size = self.sliding_window + spec_extra
            self.unified_swa_pages = self.unified_kv_pool.swa_pages
        else:
            self.unified_kv_pool = None
            self.swa_kv_pool = self._make_kv_pool(
                size=swa_size,
                page_size=swa_page_size,
                dtype=dtype,
                layer_num=stage_layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                global_page_size=swa_page_size,
            )

            c4_kv_pool_type = DeepSeekV4SingleKVPool
            if enable_hisparse:
                c4_kv_pool_type = HiSparseC4DevicePool
            self.c4_kv_pool = self._make_kv_pool(
                size=c4_size,
                page_size=c4_page_size,
                dtype=dtype,
                layer_num=c4_layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                global_page_size=page_size,
                cls=c4_kv_pool_type,
            )

            self.c128_kv_pool = self._make_kv_pool(
                size=c128_size,
                page_size=c128_page_size,
                dtype=dtype,
                layer_num=c128_layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                global_page_size=page_size,
            )

        indexer_size = self.c4_logical_size
        self.c4_indexer_kv_pool = self._make_indexer_pool(
            indexer_size,
            c4_page_size,
            dtype,
            indexer_head_dim,
            c4_layer_num,
            device,
            enable_memory_saver,
        )

        self._init_compressed_layer_mapping()

        self._init_paged_compress_states(enable_memory_saver)

    def get_unified_kv(self, layer_id: int) -> torch.Tensor:
        # Under HiCache the compressed region is loaded H->D per layer; wait for this
        # layer's transfer before attention reads it. No-op when HiCache is off.
        self.wait_layer_transfer(layer_id)
        return self.unified_kv_pool.get_unified_kv(layer_id - self._stage_start)

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def get_ring_size(self, compress_ratio: int) -> int:
        is_speculative = get_spec().speculative_algorithm is not None
        return get_compress_state_ring_size(compress_ratio, is_speculative)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices]

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        if self._unified_kv:
            # Unified buffer per layer: [swa_pages + padded_compress_rows, head_dim].
            # Compressed region [swa_pages:] is page-contiguous (row swa_pages +
            # loc//ratio), so reuse the page-block PD transfer by offsetting the ptr
            # past the SWA ring and setting item_len = one page of rows. The SWA ring
            # ships separately as StateType.SWA_RING. Order [c4, c4_indexer, c128]
            # mirrors the non-unified kv_data layout (keeps PP ptr-slicing valid).
            stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]
            swa_pages = self.unified_kv_pool.swa_pages

            def _append_compressed_entry(local_layer_id: int, ratio: int) -> None:
                buf = self.unified_kv_pool.kv_buffer[local_layer_id]
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                row_bytes = buf[0].nbytes
                rows_per_page = self.page_size // ratio
                compress_rows = buf.shape[0] - swa_pages
                data_ptrs.append(buf.data_ptr() + swa_pages * row_bytes)
                data_lens.append(compress_rows * row_bytes)
                item_lens.append(rows_per_page * row_bytes)

            c4_locals = [i for i, r in enumerate(stage_ratios) if r == 4]
            c128_locals = [i for i, r in enumerate(stage_ratios) if r == 128]

            for i in c4_locals:
                _append_compressed_entry(i, 4)
            for buf in self.c4_indexer_kv_pool.index_k_with_scale_buffer:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)
            for i in c128_locals:
                _append_compressed_entry(i, 128)

            return data_ptrs, data_lens, item_lens

        buf_groups = [
            self.c4_kv_pool.kv_buffer,
            self.c4_indexer_kv_pool.index_k_with_scale_buffer,
            self.c128_kv_pool.kv_buffer,
        ]

        for bufs in buf_groups:
            for buf in bufs:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)

        return data_ptrs, data_lens, item_lens

    def get_unified_swa_ring_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """SWA-ring region [0, swa_pages) of every unified_kv layer, addressed
        per-row by ring slot. Shipped as the StateType.SWA_RING PD component."""
        # TODO(billishyahao): validate PP layer-slicing for SWA_RING.
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        if not self._unified_kv:
            return data_ptrs, data_lens, item_lens
        swa_pages = self.unified_kv_pool.swa_pages
        for buf in self.unified_kv_pool.kv_buffer:
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
            row_bytes = buf[0].nbytes
            data_ptrs.append(buf.data_ptr())
            data_lens.append(swa_pages * row_bytes)
            item_lens.append(row_bytes)
        return data_ptrs, data_lens, item_lens

    def unified_region_buffers(self, ratio: int) -> Tuple[List[torch.Tensor], int]:
        """
        In unified_kv, swa/c4/c128 share one buffer with one slot per row. But the
        HiCache host pool transfers a whole page per indexed row, so we reshape the
        compressed region into the layout it expects: skip the SWA segment, reshape to
        one row per page, then cast to uint8.
        """
        assert self._unified_kv, "unified_region_buffers requires unified_kv layout"
        assert ratio in (4, 128), f"unsupported compression ratio: {ratio}"

        swa_pages = self.unified_kv_pool.swa_pages
        head_dim = self.unified_kv_pool.head_dim
        rows_per_page = self.page_size // ratio
        stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]
        local_layer_ids = [i for i, r in enumerate(stage_ratios) if r == ratio]

        views: List[torch.Tensor] = []
        for local_layer_id in local_layer_ids:
            buf = self.unified_kv_pool.kv_buffer[local_layer_id]
            compress_rows = buf.shape[0] - swa_pages
            assert compress_rows % rows_per_page == 0, (
                f"compressed rows {compress_rows} not a multiple of "
                f"rows_per_page {rows_per_page} for ratio {ratio}"
            )
            num_pages = compress_rows // rows_per_page
            page_view = (
                buf.narrow(0, swa_pages, compress_rows)
                .reshape(num_pages, rows_per_page * head_dim)
                .view(torch.uint8)
            )
            views.append(page_view)

        item_bytes = (
            rows_per_page * head_dim * self.unified_kv_pool.kv_buffer[0].element_size()
        )
        return views, item_bytes

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        if not self._unified_kv:
            for buf in self.swa_kv_pool.kv_buffer:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)

        for pools in [
            self.compress_state_pools,
            self.indexer_compress_state_pools,
        ]:
            for pool in pools:
                if pool is None:
                    continue
                if pool.ratio == 128:
                    continue
                t = pool.kv_score_buffer.kv_score
                assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
                data_ptrs.append(t.data_ptr())
                data_lens.append(t.nbytes)
                item_lens.append(t[0].nbytes * pool.ring_size)

        return data_ptrs, data_lens, item_lens

    def get_c128_state_buf_infos(
        self,
    ) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio != 128:
                continue
            t = pool.kv_score_buffer.kv_score
            assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
            data_ptrs.append(t.data_ptr())
            data_lens.append(t.nbytes)
            item_lens.append(t[0].nbytes if ONLINE_C128 else t[0].nbytes * 128)
        return data_ptrs, data_lens, item_lens

    def _make_kv_pool(
        self,
        *,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        global_page_size: int,
        cls: type = DeepSeekV4SingleKVPool,
    ) -> DeepSeekV4SingleKVPool:
        """Build a full / SWA / c4 / c128 single-KV pool. ``global_page_size``
        is the model-wide page_size (== ``page_size`` for the SWA pool, larger
        for the per-ratio c4/c128 pools); the default CUDA pool ignores it.
        Overridden by :class:`DSV4NPUTokenToKVPool` to swap in the NPU bf16
        PA_ND variant, which needs ``global_page_size`` for its kernel view."""
        del global_page_size  # CUDA pools key only off their own page_size
        return cls(
            size,
            page_size,
            dtype,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
        )

    def _make_indexer_pool(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
    ) -> DeepSeekV4IndexerPool:
        """Build the c4 lightning-indexer K pool (packed CUDA layout).
        Overridden by :class:`DSV4NPUTokenToKVPool` to swap in the
        dedicated-buffer NPU variant (int8 K + fp16 scale)."""
        return DeepSeekV4IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
        )

    def _state_pool_size(self, ratio: int) -> int:
        return self.c4_state_pool_size if ratio == 4 else self.c128_state_pool_size

    def _make_attn_state_pool(
        self, ratio: int, enable_memory_saver: bool
    ) -> CompressStatePool:
        """Build the per-layer attention compress-state pool for ``ratio``
        (4 or 128). Overridden by :class:`DSV4NPUTokenToKVPool` to swap the
        ring-buffered pool for the NPU paged one."""
        return CompressStatePool(
            size=self._state_pool_size(ratio),
            ring_size=self.get_ring_size(ratio),
            overlap=ratio == 4,
            head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            dtype=self.c4_state_dtype if ratio == 4 else self.c128_state_dtype,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            online=(ratio == 128 and ONLINE_C128),
            swa_page_size=self.swa_page_size,
            online_mtp_max_draft_tokens=(
                self.online_mtp_max_draft_tokens if ratio == 128 else 0
            ),
        )

    def _make_indexer_state_pool(
        self, ratio: int, enable_memory_saver: bool
    ) -> CompressStatePool:
        """Build the per-layer indexer compress-state pool (c4 only)."""
        return CompressStatePool(
            size=self._state_pool_size(ratio),
            ring_size=self.get_ring_size(ratio),
            overlap=ratio == 4,
            head_dim=self.indexer_head_dim,
            device=self.device,
            dtype=self.c4_state_dtype,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            swa_page_size=self.swa_page_size,
        )

    def _init_paged_compress_states(self, enable_memory_saver: bool):
        c4_state_pool_size = self.c4_state_pool_size
        c128_state_pool_size = self.c128_state_pool_size
        total_L = len(self.compression_ratios)
        self.compress_state_pools: List[Optional[CompressStatePool]] = [None] * total_L
        self.indexer_compress_state_pools: List[Optional[CompressStatePool]] = [
            None
        ] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                continue

            self.compress_state_pools[idx] = self._make_attn_state_pool(
                ratio, enable_memory_saver
            )

            if ratio == 4:
                self.indexer_compress_state_pools[idx] = self._make_indexer_state_pool(
                    ratio, enable_memory_saver
                )

    def _init_compressed_layer_mapping(self):
        c1_cnt = c4_cnt = c128_cnt = 0
        total_L = len(self.compression_ratios)
        self.layer_mapping: List[Optional[DeepSeekV4LayerItem]] = [None] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=0,
                    compress_layer_id=c1_cnt,
                )
                c1_cnt += 1
            elif ratio == 4:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=4,
                    compress_layer_id=c4_cnt,
                    compress_kv_pool=self.c4_kv_pool,
                )
                c4_cnt += 1
            elif ratio == 128:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=128,
                    compress_layer_id=c128_cnt,
                    compress_kv_pool=self.c128_kv_pool,
                )
                c128_cnt += 1
            else:
                raise ValueError(f"Unsupported compression ratio: {ratio}")

    def wait_layer_transfer(self, layer_id: int) -> None:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_attention_compress_states(self, layer_id: int) -> CompressStatePool:
        self.wait_layer_transfer(layer_id)
        compress_state_pool = self.compress_state_pools[layer_id]
        assert (
            compress_state_pool is not None
        ), "Only c4/c128 layers have attention states."
        return compress_state_pool

    def get_online_c128_mtp_state_slot_offset(self) -> int:
        for pool in self.compress_state_pools:
            if pool is not None and pool.ratio == 128:
                return int(pool.online_mtp_state_slot_offset)
        return 0

    def get_online_c128_mtp_max_draft_tokens(self) -> int:
        for pool in self.compress_state_pools:
            if pool is not None and pool.ratio == 128:
                return int(pool.online_mtp_max_draft_tokens)
        return 0

    def get_online_c128_state_num_req_slots(self) -> int:
        return self.online_c128_state_num_req_slots

    def get_online_c128_mtp_pending_seq_lens(self) -> torch.Tensor:
        assert self.online_c128_mtp_pending_seq_lens is not None
        return self.online_c128_mtp_pending_seq_lens

    def clear_c128_req_state(self, req_pool_idx: int) -> None:
        """Reset request-scoped C128 state for one req slot."""
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio != 128:
                continue

            state = pool.kv_score_buffer.kv_score
            if ONLINE_C128:
                row = state[req_pool_idx]
                head_dim = row.shape[-1] // 3
                row[:head_dim].fill_(float("-inf"))
                row[head_dim:].zero_()
            else:
                start = req_pool_idx * pool.ring_size
                rows = state[start : start + pool.ring_size]
                half = rows.shape[-1] // 2
                rows[:, :half].zero_()
                rows[:, half:].fill_(float("-inf"))

    def clear_unaccepted_c128_draft_states(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        accept_lens: torch.Tensor,
        num_draft_tokens: int,
    ) -> None:
        """Clear offline C128 ring slots written for rejected speculative tokens."""
        if ONLINE_C128 or num_draft_tokens <= 1 or req_pool_indices.numel() == 0:
            return

        bs = req_pool_indices.numel()
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio != 128:
                continue

            clear_unaccepted_c128_draft_states(
                pool.kv_score_buffer.kv_score,
                req_pool_indices,
                seq_lens,
                accept_lens,
                ring_size=pool.ring_size,
                num_draft_tokens=num_draft_tokens,
            )

    def get_indexer_compress_states(self, layer_id: int) -> CompressStatePool:
        self.wait_layer_transfer(layer_id)
        indexer_compress_state_pool = self.indexer_compress_state_pools[layer_id]
        assert (
            indexer_compress_state_pool is not None
        ), "Only c4 layers have indexer states."
        return indexer_compress_state_pool

    def _swa_local_layer_id(self, layer_id: int) -> int:
        """Convert absolute model layer_id to SWA-pool-local (PP-stage-local) index."""
        return layer_id - self._stage_start

    def get_swa_raw_buffer(self, layer_id: int) -> torch.Tensor:
        return self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)]

    def get_swa_key_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def set_swa_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        self.swa_kv_pool.set_key_buffer(
            self._swa_local_layer_id(layer_id), loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_extra_key_page_size(self, layer_id: int) -> int:
        _, _, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.page_size

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:
        self.wait_layer_transfer(layer_id)
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.get_key_buffer(compress_layer_id)

    def set_extra_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        compress_kv_pool.set_key_buffer(
            compress_layer_id, loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_index_k_page_size(self) -> int:
        return self.c4_indexer_kv_pool.page_size

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_with_scale_buffer(compress_layer_id)

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_scale_buffer(
            compress_layer_id,
            seq_len_tensor,
            page_indices,
            seq_len_sum,
            max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        self.c4_indexer_kv_pool.set_index_k_scale_buffer(
            compress_layer_id, loc, index_k, index_k_scale
        )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def set_swa_key_buffer_radix(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        self.swa_kv_pool.set_key_buffer(
            self._swa_local_layer_id(layer_id), swa_loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return self.swa_kv_pool.set_key_buffer_fused(
            self._swa_local_layer_id(layer_id), swa_loc, cache_k
        )

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=swa_loc,
            kvcache=self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)],
            page_size=self.swa_kv_pool.page_size,
        )

    def set_unified_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """unified_kv counterpart of set_swa_key_buffer_radix_fused_norm_rope.

        Under unified_kv the (fp8, paged) swa_kv_pool is None -- SWA K lives in
        the shared bf16 unified_kv ring instead. Norm+RoPE the draft KV in place
        (the same freqs_cis path the main model uses via _compute_kv_bf16) and
        scatter it into ``unified_kv[swa_loc]``. Rows with swa_loc < 0
        (uncommitted verify tokens) are skipped by the scatter.
        """
        from sglang.kernels.ops.attention.dsv4 import fused_norm_rope_inplace
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime

        fused_norm_rope_inplace(kv, kv_weight, eps, freqs_cis, positions)
        runtime.scatter_bf16_into_unified(
            kv=kv,
            loc=swa_loc,
            unified_kv=self.get_unified_kv(layer_id),
        )

    def set_extra_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.set_key_buffer_fused(compress_layer_id, loc, cache_k)

    def set_index_k_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.set_index_fused(compress_layer_id, loc, cache_k)

    def set_index_k_fp4(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.set_index_fp4(compress_layer_id, loc, cache_k)
