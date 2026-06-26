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
        # 是否「不在批量释放分组中」。为 True 时 free 立即生效；
        # 在 free_group_begin/free_group_end 之间为 False，释放会先攒进 free_group。
        self.is_not_in_free_group = True
        # 批量释放期间暂存待释放下标的列表，结束时一次性合并释放。
        self.free_group = []

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

    def restore_state(self, state):
        # 从 backup_state 保存的快照恢复空闲账本（用于回滚等场景）。
        self.free_pages, self.release_pages = state

    def backup_state(self):
        # 备份当前空闲账本，便于之后 restore_state 回滚。
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        # 开启「批量释放」模式：此后的 free 不立即生效，而是先攒进 free_group，
        # 待 free_group_end 时一次性合并释放，减少多次 torch.cat 的开销。
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        # 结束「批量释放」模式：把期间攒下的所有待释放下标拼接后一次性 free。
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

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
