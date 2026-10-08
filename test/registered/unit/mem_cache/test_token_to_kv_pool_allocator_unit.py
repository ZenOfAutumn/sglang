"""TokenToKVPoolAllocator / PagedTokenToKVPoolAllocator 的单测（两级内存池的第二级）。

职责回顾：分配器只维护「哪些 KV 槽位空闲」的账本（``free_pages`` / ``release_pages``），
自身不持有显存；真正的显存在 ``KVCache`` 里（通过 ``_kvcache`` 引用）。

覆盖点：

- 构造 / ``clear``：``free_pages`` 从 1 开始（0 号 slot 是 padding，永不分配）；
- ``alloc``：从头部取、``need_size == 0``、耗尽返回 None、分页版输出按页对齐；
- ``free``：归还位置（Token 版追加到尾部 / Paged 版按页去重后插到头部）、空张量为 no-op；
- ``need_sort``：释放先进 ``release_pages``，不够分时才 ``merge_and_sort_free`` 排序合并；
- ``free_group``：批量释放先挂起，``free_group_end`` 一次性归还，并对张量取所有权；
- 基类契约：``alloc_extend`` / ``alloc_decode`` 只有分页版支持、``get_cpu_copy`` 转发给
  KVCache、``resize`` / ``translate_kv_indices_for_transfer`` / ``free_full`` /
  ``free_segment`` 的默认行为。

运行方式（仓库根目录 /home/sankuai/sglang）::

    # 标准方式（CI 同款）
    python -m pytest test/registered/unit/mem_cache/test_token_to_kv_pool_allocator_unit.py -v

    # 直接跑文件（文件末尾有 unittest.main()）
    cd python && python3 ../test/registered/unit/mem_cache/test_token_to_kv_pool_allocator_unit.py -v

本机开发容器的额外说明（site-packages 的 deep_gemm 过旧，且设 PYTHONPATH 会让 numpy
找不到）::

    cd /home/sankuai/sglang/python && python3 -c "import sys; sys.path.insert(0,'/home/sankuai/tb4.0/dsv41-pd/DeepGEMM/build'); import pytest; sys.exit(pytest.main(['/home/sankuai/sglang/test/registered/unit/mem_cache/test_token_to_kv_pool_allocator_unit.py','-v']))"
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PAGE_SIZE = 4
NUM_PAGES = 4


class _FakeKVCache:
    """假的底层 KV 存储：只记录 ``get_cpu_copy`` / ``load_cpu_copy`` 的转发。"""

    def __init__(self):
        self.calls = []

    def get_cpu_copy(self, indices, mamba_indices=None):
        """记录一次换出调用，并返回一个哨兵值供断言。"""
        self.calls.append(("get_cpu_copy", indices.tolist(), mamba_indices))
        return "cpu_copy"

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        """记录一次换入调用，并返回一个哨兵值供断言。"""
        self.calls.append(("load_cpu_copy", indices.tolist(), mamba_indices))
        return "loaded"


def _make_token_allocator(size=8, need_sort=False, kvcache=None):
    """构造 page_size == 1 的 ``TokenToKVPoolAllocator``（CPU）。"""
    return TokenToKVPoolAllocator(
        size=size,
        dtype=torch.float16,
        device="cpu",
        kvcache=kvcache,
        need_sort=need_sort,
    )


def _make_paged_allocator(num_pages=NUM_PAGES, page_size=PAGE_SIZE, need_sort=False):
    """构造 page_size > 1 的 ``PagedTokenToKVPoolAllocator``（CPU）。"""
    return PagedTokenToKVPoolAllocator(
        size=num_pages * page_size,
        page_size=page_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


def _slots(values, dtype=torch.int64):
    """把 python list 转成分配器要求的 int64 张量（free 的入参必须是张量）。"""
    return torch.tensor(values, dtype=dtype)


class TestTokenAllocatorConstruction(CustomTestCase):
    """构造 / clear：空闲表初值与 0 号 padding slot。"""

    def test_clear_starts_free_pages_at_one(self):
        """``clear`` 后空闲 slot 是 1..size，0 号 padding slot 不在其中。"""
        kv = _FakeKVCache()
        alloc = _make_token_allocator(size=8, kvcache=kv)

        self.assertEqual(alloc.size, 8)
        self.assertEqual(alloc.page_size, 1)
        self.assertEqual(alloc.free_pages.tolist(), [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(len(alloc.release_pages), 0)
        self.assertEqual(alloc.available_size(), 8)
        self.assertEqual(alloc.size_full, 8)
        self.assertIs(alloc.get_kvcache(), kv)

    def test_padding_slot_zero_is_never_allocated(self):
        """把全部 slot 分完也拿不到 0 号 slot。"""
        alloc = _make_token_allocator(size=4)

        all_slots = alloc.alloc(4)

        self.assertEqual(all_slots.tolist(), [1, 2, 3, 4])
        self.assertNotIn(0, all_slots.tolist())
        self.assertEqual(alloc.available_size(), 0)

    def test_clear_resets_a_used_allocator(self):
        """``clear`` 把已分配 / 已释放的状态全部抹掉，回到初始空闲表。"""
        alloc = _make_token_allocator(size=4)
        alloc.alloc(3)
        alloc.free(_slots([2]))

        alloc.clear()

        self.assertEqual(alloc.free_pages.tolist(), [1, 2, 3, 4])
        self.assertEqual(alloc.available_size(), 4)


class TestTokenAllocatorAlloc(CustomTestCase):
    """``alloc``：头部取、need_size == 0、耗尽返回 None。"""

    def test_alloc_takes_from_head_and_shrinks_free_pages(self):
        """从空闲表头部取，返回值的类型/设备与分配器一致。"""
        alloc = _make_token_allocator(size=8)

        out = alloc.alloc(3)

        self.assertEqual(out.tolist(), [1, 2, 3])
        self.assertEqual(out.dtype, torch.int64)
        self.assertEqual(alloc.free_pages.tolist(), [4, 5, 6, 7, 8])
        self.assertEqual(alloc.available_size(), 5)

    def test_alloc_zero_returns_empty_and_keeps_free_pages(self):
        """``need_size == 0`` 返回空张量，空闲表不变（free_pages[:0] 与 [0:] 都是合法切片）。"""
        alloc = _make_token_allocator(size=4)

        out = alloc.alloc(0)

        self.assertEqual(out.numel(), 0)
        self.assertEqual(alloc.free_pages.tolist(), [1, 2, 3, 4])

    def test_alloc_returns_none_when_exhausted(self):
        """需求超过空闲数时整体返回 None，空闲表保持原样（不做部分分配）。"""
        alloc = _make_token_allocator(size=2)

        self.assertIsNone(alloc.alloc(3))
        self.assertEqual(alloc.free_pages.tolist(), [1, 2])

    def test_alloc_extend_and_decode_are_paged_only(self):
        """Token 版不实现按长度扩展 / decode 追加分配，基类统一抛 NotImplementedError。"""
        alloc = _make_token_allocator(size=4)

        with self.assertRaisesRegex(NotImplementedError, "only for paged allocator"):
            alloc.alloc_extend()
        with self.assertRaisesRegex(NotImplementedError, "only for paged allocator"):
            alloc.alloc_decode()


class TestTokenAllocatorFree(CustomTestCase):
    """``free``：追加到尾部（因此释放的 slot 最后才被复用）、空张量为 no-op。"""

    def test_free_appends_to_tail(self):
        """释放的 slot 追加到空闲表尾部，而 alloc 从头部取 —— 新 slot 优先，旧的复用靠后。"""
        alloc = _make_token_allocator(size=4)
        alloc.alloc(2)

        alloc.free(_slots([1, 2]))

        self.assertEqual(alloc.free_pages.tolist(), [3, 4, 1, 2])
        self.assertEqual(alloc.available_size(), 4)
        # 下次 alloc 仍然先拿没用过的 3、4。
        self.assertEqual(alloc.alloc(2).tolist(), [3, 4])

    def test_free_empty_is_noop_and_ignores_dtype(self):
        """空张量直接返回：既不校验 dtype 也不改账本（释放路径常拿到空切片）。"""
        alloc = _make_token_allocator(size=4)

        self.assertIsNone(alloc.free(torch.empty(0, dtype=torch.int32)))

        self.assertEqual(alloc.free_pages.tolist(), [1, 2, 3, 4])

    def test_free_full_defaults_to_free(self):
        """单池没有 SWA 兄弟 slot，``free_full`` 就是普通 ``free``。"""
        alloc = _make_token_allocator(size=4)
        alloc.alloc(4)

        alloc.free_full(_slots([1, 2]))

        self.assertEqual(sorted(alloc.free_pages.tolist()), [1, 2])

    def test_free_segment_defaults_to_free(self):
        """基类 ``free_segment`` 默认忽略 ``start_pos``，等价整体释放。"""
        alloc = _make_token_allocator(size=4)
        alloc.alloc(4)

        alloc.free_segment(_slots([1, 2, 3]), start_pos=1)

        self.assertEqual(sorted(alloc.free_pages.tolist()), [1, 2, 3])


class TestTokenAllocatorNeedSort(CustomTestCase):
    """``need_sort``：释放先进 ``release_pages``，不够分时才排序合并回 ``free_pages``。"""

    def test_free_routes_to_release_pages(self):
        """need_sort 时释放不再直接进 free_pages，而是先进 release_pages。"""
        alloc = _make_token_allocator(size=4, need_sort=True)
        alloc.alloc(4)

        alloc.free(_slots([1, 2]))

        self.assertEqual(alloc.free_pages.tolist(), [])
        self.assertEqual(alloc.release_pages.tolist(), [1, 2])
        # available_size 把待合并页也算进来，否则会被误判成没内存了。
        self.assertEqual(alloc.available_size(), 2)

    def test_merge_on_demand_sorts_and_recycles_low_slots(self):
        """空闲表不够分时才触发 merge：释放的（更低）slot 被排序后重新可分配。"""
        alloc = _make_token_allocator(size=4, need_sort=True)
        self.assertEqual(alloc.alloc(2).tolist(), [1, 2])
        alloc.free(_slots([1, 2]))

        # free_pages 还有 [3,4]，够分，所以不会触发 merge。
        self.assertEqual(alloc.alloc(2).tolist(), [3, 4])
        self.assertEqual(alloc.release_pages.tolist(), [1, 2])

        # 再要 2 个：free_pages 空了，触发 merge_and_sort_free → [1,2] + [] 排序后可用。
        self.assertEqual(alloc.alloc(2).tolist(), [1, 2])
        self.assertEqual(len(alloc.release_pages), 0)
        self.assertEqual(alloc.free_pages.tolist(), [])

    def test_merge_and_sort_free_without_pending_is_noop(self):
        """没有待合并页时 ``merge_and_sort_free`` 什么都不做。"""
        alloc = _make_token_allocator(size=4, need_sort=True)

        alloc.merge_and_sort_free()

        self.assertEqual(alloc.free_pages.tolist(), [1, 2, 3, 4])


class TestTokenAllocatorFreeGroup(CustomTestCase):
    """``free_group_*``：批量释放先挂起，结束时一次性归还。"""

    def test_group_defers_until_group_end(self):
        """free_group 期间释放不生效，free_group_end 才一次性进入空闲表。"""
        alloc = _make_token_allocator(size=4)
        alloc.alloc(4)

        alloc.free_group_begin()
        alloc.free(_slots([1]))
        alloc.free(_slots([2]))
        self.assertEqual(alloc.free_pages.tolist(), [])

        alloc.free_group_end()

        self.assertEqual(alloc.free_pages.tolist(), [1, 2])

    def test_group_takes_ownership_of_the_tensor(self):
        """挂起时会 clone：调用方随后改写自己的张量不会污染账本。"""
        alloc = _make_token_allocator(size=4)
        alloc.alloc(4)
        pending = _slots([1, 2])

        alloc.free_group_begin()
        alloc.free(pending)
        pending.fill_(99)  # 调用方复用了缓冲区
        alloc.free_group_end()

        self.assertEqual(alloc.free_pages.tolist(), [1, 2])

    def test_group_end_without_frees_is_noop(self):
        """只在 free_group_begin/end 之间没有释放时，结束调用不改变账本。"""
        alloc = _make_token_allocator(size=4)

        alloc.free_group_begin()
        alloc.free_group_end()

        self.assertEqual(alloc.free_pages.tolist(), [1, 2, 3, 4])


class TestTokenAllocatorMisc(CustomTestCase):
    """``resize``、transfer 地址翻译、CPU 拷贝转发等外围契约。"""

    def test_resize_rebuilds_free_pages(self):
        """``resize`` 会改 size 并 ``clear``，空闲表按新容量重建。"""
        alloc = _make_token_allocator(size=4)

        alloc.resize(SimpleNamespace(max_total_num_tokens=6))

        self.assertEqual(alloc.size, 6)
        self.assertEqual(alloc.free_pages.tolist(), [1, 2, 3, 4, 5, 6])

    def test_translate_kv_indices_is_identity(self):
        """静态池的 token id 直接索引已注册 buffer，因此翻译是恒等映射。"""
        alloc = _make_token_allocator(size=4)
        indices = _slots([1, 2])

        self.assertIs(alloc.translate_kv_indices_for_transfer(indices), indices)

    def test_cpu_copy_hooks_delegate_to_kvcache(self):
        """换入 / 换出都由底层 KVCache 实现，分配器只做转发（含 mamba_indices）。"""
        kv = _FakeKVCache()
        alloc = _make_token_allocator(size=4, kvcache=kv)

        self.assertEqual(alloc.get_cpu_copy(_slots([1, 2])), "cpu_copy")
        self.assertEqual(
            alloc.load_cpu_copy("cpu_cache", _slots([3])), "loaded"
        )

        self.assertEqual(
            kv.calls,
            [
                ("get_cpu_copy", [1, 2], None),
                ("load_cpu_copy", [3], None),
            ],
        )


class TestPagedAllocator(CustomTestCase):
    """分页版：按页分配（输出保证按页对齐）与按页去重释放。"""

    def test_clear_reserves_page_zero(self):
        """空闲页是 1..num_pages，第 0 页留给 padding token。"""
        alloc = _make_paged_allocator(num_pages=4, page_size=4)

        self.assertEqual(alloc.num_pages, 4)
        self.assertEqual(alloc.free_pages.tolist(), [1, 2, 3, 4])
        # 每页 4 个 token，共 16 个可分配 token slot。
        self.assertEqual(alloc.available_size(), 16)

    def test_alloc_returns_page_aligned_indices(self):
        """一次拿 2 页：取 page 1、2，输出展开成 8 个连续 token slot。"""
        alloc = _make_paged_allocator(num_pages=4, page_size=4)

        out = alloc.alloc(8)

        self.assertEqual(out.tolist(), [4, 5, 6, 7, 8, 9, 10, 11])
        self.assertEqual(alloc.free_pages.tolist(), [3, 4])
        self.assertEqual(alloc.available_size(), 8)

    def test_alloc_returns_none_when_pages_exhausted(self):
        """页不够（这里要 4 页只剩 2 页）时返回 None。"""
        alloc = _make_paged_allocator(num_pages=4, page_size=4)
        alloc.alloc(8)

        self.assertIsNone(alloc.alloc(16))

    def test_non_page_aligned_alloc_floors_the_page_count(self):
        """非页对齐需求会按页向下取整（``need_size // page_size``），只返回整页。"""
        alloc = _make_paged_allocator(num_pages=4, page_size=4)
        alloc.debug_mode = False  # 关掉 DEBUG 断言，专测取整行为

        out = alloc.alloc(5)

        self.assertEqual(out.tolist(), [4, 5, 6, 7])
        self.assertEqual(alloc.free_pages.tolist(), [2, 3, 4])

    def test_free_dedups_pages_and_prepends(self):
        """释放按页去重（同一页多次出现只归还一次），且插到空闲表头部。"""
        alloc = _make_paged_allocator(num_pages=4, page_size=4)
        alloc.alloc(8)  # 拿走 page 1、2，free_pages == [3, 4]

        alloc.free(_slots([4, 5, 5, 4]))  # 全是 page 1 的 token

        self.assertEqual(alloc.free_pages.tolist(), [1, 3, 4])
        self.assertEqual(alloc.available_size(), 12)

    def test_free_empty_is_noop(self):
        """空张量不归还任何页。"""
        alloc = _make_paged_allocator(num_pages=4, page_size=4)

        alloc.free(torch.empty(0, dtype=torch.int64))

        self.assertEqual(alloc.free_pages.tolist(), [1, 2, 3, 4])

    def test_free_routes_to_release_pages_when_need_sort(self):
        """need_sort 时释放的页先进 release_pages，页号插在其头部。"""
        alloc = _make_paged_allocator(num_pages=4, page_size=4, need_sort=True)
        alloc.alloc(8)

        alloc.free(_slots([4, 5, 6, 7]))  # page 1

        self.assertEqual(alloc.release_pages.tolist(), [1])
        self.assertEqual(alloc.free_pages.tolist(), [3, 4])

    def test_group_defers_until_group_end(self):
        """分页版同样支持 free_group 挂起，结束时统一按页归还。"""
        alloc = _make_paged_allocator(num_pages=4, page_size=4)
        alloc.alloc(8)

        alloc.free_group_begin()
        alloc.free(_slots([4, 5, 6, 7]))  # page 1
        alloc.free(_slots([8, 9]))  # page 2
        self.assertEqual(alloc.free_pages.tolist(), [3, 4])

        alloc.free_group_end()

        self.assertEqual(alloc.free_pages.tolist(), [1, 2, 3, 4])

    def test_resize_uses_page_size_for_num_pages(self):
        """``resize`` 在分页版会按 page_size 重算 num_pages 再重建空闲页。"""
        alloc = _make_paged_allocator(num_pages=4, page_size=4)

        alloc.resize(SimpleNamespace(max_total_num_tokens=8))

        self.assertEqual(alloc.size, 8)
        self.assertEqual(alloc.num_pages, 2)
        self.assertEqual(alloc.free_pages.tolist(), [1, 2])
        self.assertEqual(alloc.available_size(), 8)


if __name__ == "__main__":
    unittest.main()
