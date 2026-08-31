"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class BaseTokenToKVPoolAllocator(abc.ABC):
    """KV 槽位分配器的抽象基类（两级内存池中的「第二级」）。

    职责：在底层物理 KV 存储（``KVCache``）之上，管理「哪些 KV 槽位下标空闲、
    哪些已被占用」，对外提供 ``alloc``（申请槽位下标）/ ``free``（归还槽位下标）。
    它本身**不持有显存**，只维护一份「空闲下标」的账本——真正的显存在 ``KVCache``。

    两种典型子类：
    * ``TokenToKVPoolAllocator``：``page_size == 1``，按单 token 粒度分配，最简单；
    * ``PagedTokenToKVPoolAllocator``：``page_size > 1``，按「页」分配，对应 paged
      attention，减少显存碎片。

    空闲账本被拆成两份，以配合「延迟排序」优化（见 ``need_sort`` 与
    ``merge_and_sort_free``）：
    * ``free_pages``：可立即分配的空闲页（下标）；
    * ``release_pages``：刚被释放、尚未并入 ``free_pages`` 的页（延迟合并/排序）。
    """

    @abc.abstractmethod
    def __init__(
        self,
        size: int,  # 可分配的 KV 槽位总数（token 数，不含 padding）
        page_size: int,  # 分页粒度：1 表示按 token，>1 表示按页（paged attention）
        dtype: torch.dtype,  # KV 存储的数据类型（透传给底层 KVCache）
        device: str,  # 所在设备，如 "cuda"
        kvcache: KVCache,  # 底层真正持有显存的物理 KV 存储
        need_sort: bool,  # 是否启用「延迟排序」：释放先入 release_pages，需要时再合并排序
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        # 空闲页账本：free_pages 可直接分配；release_pages 暂存刚释放、待合并的页。
        # 由各子类在 clear() 中初始化为具体张量。
        self.free_pages = None
        self.release_pages = None
        # None: free right away. A list: hold frees until free_group_end().
        self.free_group: list[torch.Tensor] | None = None

    @property
    def size_full(self):
        # 分配器的「完整容量」。基类等于 size；某些子类（如带 SWA 的）可能重写。
        return self.size

    def debug_print(self) -> str:
        # 调试用：返回分配器内部状态的可读描述，基类默认返回空串。
        return ""

    def available_size(self):
        # 当前还能分配多少个 token 槽位 = (可分配页 + 待合并页) × 每页 token 数。
        # 注意把 release_pages 也算进来，因为它们随时可被 merge_and_sort_free 复用。
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        # 返回底层物理 KV 存储对象。
        return self._kvcache

    def free_group_begin(self):
        self.free_group = []

    def free_group_end(self):
        pending, self.free_group = self.free_group, None
        if pending:
            self.free(torch.cat(pending))

    @staticmethod
    def _copy_for_free_group(free_index: torch.Tensor) -> torch.Tensor:
        """Take ownership before a caller can mutate a deferred tensor view."""
        return free_index.clone()

    def merge_and_sort_free(self):
        # 把延迟释放的 release_pages 合并回 free_pages 并排序。
        # 排序的目的：让分配尽量取到「连续」的下标，利于 paged attention 的局部性。
        # 仅在 need_sort 为真、且确实有待合并页时由子类按需调用（如 alloc 不够时）。
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def translate_kv_indices_for_transfer(
        self, kv_indices: torch.Tensor
    ) -> torch.Tensor:
        """Token ids as the PD-disaggregation transfer engine addresses them.

        Identity here: a static pool's token ids index its registered buffers
        directly. Virtual-id pools must override.
        """
        return kv_indices

    def get_cpu_copy(self, indices, mamba_indices=None):
        # 把指定下标的 KV 拷贝到 CPU（用于卸载/换出）。
        # FIXME: 待分页分配器实现后复用统一的 get_cpu_copy。
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # 把 CPU 上的 KV 拷回设备指定下标（用于换入）。
        # FIXME: 待分页分配器实现后复用统一的 load_cpu_copy。
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        # 扩展分配（prefill / 续写时按非对齐长度扩展）：仅分页分配器支持。
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        # 解码分配（每步追加少量 token）：仅分页分配器支持。
        raise NotImplementedError("alloc_decode is only for paged allocator")

    def resize(self, config) -> None:
        self.size = config.max_total_num_tokens
        if self.page_size > 1:
            self.num_pages = config.max_total_num_tokens // self.page_size
        self.clear()

    @abc.abstractmethod
    def clear(self):
        # 重置分配器：把所有槽位重新标记为空闲（子类必须实现）。
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        # 申请 need_size 个 KV 槽位下标；空间不足返回 None（子类必须实现）。
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        # 归还 free_index 指向的 KV 槽位下标（子类必须实现）。
        raise NotImplementedError()

    def free_full(self, free_index: torch.Tensor):
        """Free slots whose SWA peers the caller already released.

        A hybrid SWA allocator pairs each full-attention slot with an SWA slot
        that can die first; this releases the full side alone. A single pool has
        no peer, so it is a plain free()."""
        self.free(free_index)

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        """Free ``kv_row[start_pos : start_pos + n]`` of one request (or a
        page-aligned copy); subclasses may use ``start_pos`` to skip the
        data-dependent dedup. Default: plain free()."""
        self.free(free_index)

    def free_segments(self, segments):
        """Free disjoint ascending ``(free_index, start_pos)`` segments of one
        request's kv row; a boundary page shared by consecutive segments is
        emitted once (the later segment's head is trimmed)."""
        ps = self.page_size
        prev_end = None
        for free_index, start_pos in segments:
            n = free_index.numel()
            if n == 0:
                continue
            seg_end = start_pos + n
            if prev_end is not None and start_pos // ps == (prev_end - 1) // ps:
                boundary = (start_pos // ps + 1) * ps
                free_index = free_index[boundary - start_pos :]
                start_pos = boundary
            prev_end = seg_end
            self.free_segment(free_index, start_pos=start_pos)
