"""
Unit tests for the RadixCache implementation.

This module tests the core functionality of RadixCache, RadixKey, and TreeNode
following SGLang testing patterns.

Test Coverage:
- RadixKey: token ID management, slicing, iteration, representation
- TreeNode: node properties, reference counting, hash values
- RadixCache: insert/match operations, eviction, page alignment, error handling
- Cache events and request handling
- Boundary conditions with parameterized testing

Usage:
    python test_radix_cache_unit.py
    python -m pytest test_radix_cache_unit.py -v
    python -m pytest test_radix_cache_unit.py::TestRadixCache::test_insert_basic

测试用例总览（按类）：
- TestKVCacheEventQueue：KVCacheEventRecorder 的事件合并规则（哪些能合并、哪些必须断开）
- TestRadixKey：键的长度/迭代/切片、match() 的前缀长度与页对齐、bigram 视图、命名空间携带
- TestTreeNode：字段默认值、evicted/backuped 派生属性、host 引用计数、哈希前缀拼接
- TestRadixCache：插入/匹配、淘汰与锁计数账本、页对齐边界、节点分裂、extra_key/cache_salt
  隔离、KV 事件（store/remove/哈希切分）、大规模插入的内存占用量级

约定：RadixCache 用例统一用 RadixCache.create_simulated(...) 构造不接真实显存的仿真树
（device 断言为 cpu）；需要验证 slot 归还时注入 mock allocator；涉及加速器显存的只有
TestRadixCache.test_memory_allocated，因此整份文件挂在单卡 GPU CI 下。

运行方式（在仓库根目录 /home/sankuai/sglang 下）::

    # 标准方式（CI 同款，推荐）
    python -m pytest test/registered/unit/mem_cache/test_radix_cache_unit.py -v

    # 直接跑文件也可以（文件末尾有 unittest.main()）
    cd python && python3 ../test/registered/unit/mem_cache/test_radix_cache_unit.py -v

本机开发容器的额外说明（与 test_req_to_token_pool_unit.py 相同）：site-packages 里的
``deep_gemm`` 版本过旧（缺少新代码需要的 ``deep_gemm.utils.layout``），且设置
``PYTHONPATH`` 会导致 ``numpy`` 找不到（``/usr/local/lib/packages`` 不在路径里）。
因此需要先 ``cd`` 到 ``python/`` 让 ``import sglang`` 命中工作区源码，再把匹配的
DeepGEMM 插到 ``sys.path`` 前面::

    cd /home/sankuai/sglang/python && python3 -c "import sys; sys.path.insert(0,'/home/sankuai/tb4.0/dsv41-pd/DeepGEMM/build'); import pytest; sys.exit(pytest.main(['/home/sankuai/sglang/test/registered/unit/mem_cache/test_radix_cache_unit.py','-q']))"

本机实测：39 passed（含 test_memory_allocated，需要可用 GPU）。
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-based unit test, runs quickly on any GPU runner
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

import random
import unittest
import unittest.mock
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    BlockStoredMetadata,
    BlockStoredWithMetadata,
    StorageMedium,
)
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.srt.mem_cache.mamba_radix_cache import TreeNode as MambaTreeNode
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.utils import get_device

# Test constants
DEFAULT_PAGE_SIZE = 4


class TestKVCacheEventQueue(unittest.TestCase):
    """KVCacheEventRecorder 的入队合并（coalesce）行为。

    事件会被 router 用来重建缓存索引，因此“相邻且兼容的事件合并成一条”
    可以显著压低事件量；但合并条件必须严格：medium / lora_id / block_size /
    parent 链 / cache_salt 任一不同，都不能合并，否则外部索引会串味。
    """

    @staticmethod
    def _store(
        block_hash: int,
        parent_block_hash: int | None,
        *,
        block_size: int = 2,
        medium: StorageMedium = StorageMedium.GPU,
        lora_id: int | None = None,
        cache_salt: str | None = None,
    ) -> BlockStored:
        """构造单块 BlockStored 的测试小工具；带 cache_salt 时改用 BlockStoredWithMetadata。"""
        event_args = dict(
            block_hashes=[block_hash],
            parent_block_hash=parent_block_hash,
            token_ids=[block_hash, block_hash + 1][:block_size],
            block_size=block_size,
            lora_id=lora_id,
            medium=medium,
        )
        if cache_salt is None:
            return BlockStored(**event_args)
        return BlockStoredWithMetadata(
            **event_args,
            metadata=BlockStoredMetadata(cache_salt=cache_salt),
        )

    def test_enqueue_coalesces_compatible_stores(self):
        """相邻且兼容的 BlockStored 应合并为一条事件。

        构造：先入队父块（hash=1, parent=None），再入队子块（hash=2, parent=1）；
        两者 medium/block_size/lora_id 一致，且后者的 parent 正好是前者的 hash。
        断言：只产出 1 条事件；block_hashes 顺序拼接为 [1, 2]；
        合并后 parent 取第一块的 parent（None），不会变成中间块的哈希；
        token_ids 一并拼接（每块 2 个 token）→ [1, 2, 2, 3]。
        """
        queue = KVCacheEventRecorder(enabled=True, page_size=DEFAULT_PAGE_SIZE)
        queue.enqueue(self._store(1, None))
        queue.enqueue(self._store(2, 1))

        events = queue.take()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].block_hashes, [1, 2])
        self.assertEqual(events[0].parent_block_hash, None)
        self.assertEqual(events[0].token_ids, [1, 2, 2, 3])

    def test_enqueue_coalesces_compatible_removes(self):
        """相邻的 BlockRemoved 应合并为一条（批量上报淘汰）。

        构造：入队 [1]，再入队 [2, 3]，medium 相同。
        断言：合并成 1 条 BlockRemoved，block_hashes = [1, 2, 3]。
        """
        queue = KVCacheEventRecorder(enabled=True, page_size=DEFAULT_PAGE_SIZE)
        queue.enqueue(BlockRemoved(block_hashes=[1], medium=StorageMedium.GPU))
        queue.enqueue(BlockRemoved(block_hashes=[2, 3], medium=StorageMedium.GPU))

        events = queue.take()
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], BlockRemoved)
        self.assertEqual(events[0].block_hashes, [1, 2, 3])

    def test_enqueue_preserves_fusion_boundaries(self):
        """不兼容的事件必须保持为独立事件（合并边界不能被越过）。

        四组“相邻但不兼容”的 store，每组都应产出 2 条事件：
        medium 不同（GPU/CPU）、lora_id 不同（None/1）、block_size 不同（2/1）、
        父哈希不连续（第二块 parent=None，无法与第一块串成链）。
        另外三组边界：store/remove/AllBlocksCleared/store 交替 → 4 条
        （清空事件必须打断合并）；两条 remove 但 medium 不同 → 2 条；
        两条 store 但 cache_salt 不同（tenant-a/tenant-b）→ 2 条。
        """
        incompatible_stores = [
            self._store(2, 1, medium=StorageMedium.CPU),
            self._store(3, 1, lora_id=1),
            self._store(4, 1, block_size=1),
            self._store(5, None),
        ]
        for incoming in incompatible_stores:
            queue = KVCacheEventRecorder(enabled=True, page_size=DEFAULT_PAGE_SIZE)
            queue.enqueue(self._store(1, None))
            queue.enqueue(incoming)
            self.assertEqual(len(queue.take()), 2)

        queue = KVCacheEventRecorder(enabled=True, page_size=DEFAULT_PAGE_SIZE)
        queue.enqueue(self._store(1, None))
        queue.enqueue(BlockRemoved(block_hashes=[1], medium=StorageMedium.GPU))
        queue.enqueue(AllBlocksCleared())
        queue.enqueue(self._store(2, None))
        self.assertEqual(len(queue.take()), 4)

        queue = KVCacheEventRecorder(enabled=True, page_size=DEFAULT_PAGE_SIZE)
        queue.enqueue(BlockRemoved(block_hashes=[1], medium=StorageMedium.GPU))
        queue.enqueue(BlockRemoved(block_hashes=[2], medium=StorageMedium.CPU))
        self.assertEqual(len(queue.take()), 2)

        queue = KVCacheEventRecorder(enabled=True, page_size=DEFAULT_PAGE_SIZE)
        queue.enqueue(self._store(1, None, cache_salt="tenant-a"))
        queue.enqueue(self._store(2, 1, cache_salt="tenant-b"))
        self.assertEqual(len(queue.take()), 2)


class TestRadixKey(unittest.TestCase):
    """RadixKey 的键语义：长度/迭代/切片、match() 的公共前缀长度、bigram 视图与命名空间。

    纯 CPU、纯数据结构测试，不依赖 RadixCache；`_assert_match` 是通用的
    “给定两段 token + page_size → 断言公共前缀长度” 小工具。
    """

    def test_init_with_extra_key(self):
        """构造时 token_ids 与 extra_key 被原样保存。

        断言 token_ids 转回 list 后等于输入，extra_key 可原样读回。
        """
        token_ids = [1, 2, 3]
        extra_key = "test_key"
        key = RadixKey(array("q", token_ids), extra_key)
        self.assertEqual(list(key.token_ids), token_ids)
        self.assertEqual(key.extra_key, extra_key)

    def test_len_and_iter(self):
        """__len__ 返回 token 个数，__iter__ 逐个产出 token。

        参数化覆盖：普通序列 [1,2,3]→3、空序列→0、单元素→1。
        """
        test_cases = [
            ([1, 2, 3], 3),
            ([], 0),
            ([42], 1),
        ]

        for tokens, expected in test_cases:
            with self.subTest(tokens=tokens):
                key = RadixKey(array("q", tokens))
                self.assertEqual(len(key), expected)
                self.assertEqual(list(key), tokens)

    def test_getitem_int(self):
        """整数下标返回长度为 1 的 RadixKey（而不是单个 int）。

        覆盖 0、负索引 -1（应换算成最后一个）、末尾索引 2，
        并断言返回类型仍是 RadixKey（保证后续能继续做切片/匹配）。
        """
        test_cases = [
            ([10, 20, 30], 0, [10]),
            ([10, 20, 30], -1, [30]),
            ([10, 20, 30], 2, [30]),
        ]

        for tokens, index, expected in test_cases:
            with self.subTest(tokens=tokens, index=index):
                key = RadixKey(array("q", tokens))
                result = key[index]
                self.assertIsInstance(result, RadixKey)
                self.assertEqual(list(result.token_ids), expected)

    def test_getitem_slice(self):
        """切片返回新的 RadixKey，继承 extra_key；边界切片不报错。

        key[1:4] → [2,3,4] 且 extra_key 仍为 "extra"；
        key[2:2] → 空切片；key[:] → 完整拷贝。
        """
        key = RadixKey(array("q", [1, 2, 3, 4, 5]), "extra")

        # Basic slice
        sliced = key[1:4]
        self.assertIsInstance(sliced, RadixKey)
        self.assertEqual(list(sliced.token_ids), [2, 3, 4])
        self.assertEqual(sliced.extra_key, "extra")

        # Edge cases
        self.assertEqual(list(key[2:2].token_ids), [])  # Empty slice
        self.assertEqual(list(key[:].token_ids), [1, 2, 3, 4, 5])  # Full slice

    def test_cache_salt_is_preserved_by_slicing(self):
        """回归保护：切片必须同时保留 extra_key 与 cache_salt。

        树在分裂/匹配时会对 key 做切片；若切片丢掉命名空间字段，
        不同 salt 的缓存就可能被错误共享，因此这里显式断言两者都被带下去。
        """
        key = RadixKey(
            array("q", [1, 2, 3, 4]),
            extra_key="classification",
            cache_salt="tenant-a",
        )
        sliced = key[1:3]
        self.assertEqual(sliced.extra_key, "classification")
        self.assertEqual(sliced.cache_salt, "tenant-a")

    def test_getitem_invalid_index(self):
        """越界下标必须抛 IndexError（而不是静默返回空）。

        调用方依赖“越界即报错”来发现长度计算错误，静默返回空会被
        误当成“正常但无命中”。
        """
        key = RadixKey(array("q", [1, 2, 3]))
        with self.assertRaises(IndexError):
            _ = key[10]  # Out of bounds

    def _assert_match(self, a, b, page_size, expected, is_bigram=False):
        """测试小工具：构造两个 RadixKey 并断言 match() 返回的公共前缀长度。"""
        key_a = RadixKey(array("q", a), is_bigram=is_bigram)
        key_b = RadixKey(array("q", b), is_bigram=is_bigram)
        self.assertEqual(key_a.match(key_b, page_size=page_size), expected)

    def test_match_page_size_1(self):
        """page_size=1 时 match() 返回精确的公共前缀长度（不做页对齐）。

        覆盖 7 种形态：完全相同(4)、第 2 个 token 分叉(2)、首位就分叉(0)、
        对方是自己的前缀(3)、自身为空(0)、对方为空(0)、双方为空(0)。
        """
        self._assert_match([1, 2, 3, 4], [1, 2, 3, 4], 1, 4)  # identical
        self._assert_match([1, 2, 3, 4], [1, 2, 9, 9], 1, 2)  # diverge at index 2
        self._assert_match([9, 2, 3], [1, 2, 3], 1, 0)  # diverge at index 0
        self._assert_match([1, 2, 3, 4], [1, 2, 3], 1, 3)  # other is a prefix
        self._assert_match([], [1, 2], 1, 0)  # empty self
        self._assert_match([1, 2], [], 1, 0)  # empty other
        self._assert_match([], [], 1, 0)  # both empty

    def test_match_page_size_gt_1_rounds_down(self):
        """page_size>1 时公共前缀长度向下取整到页的整数倍。

        覆盖：第 7 个 token 才分叉(→4，不足一页丢掉)、首页内部就分叉(→0)、
        第 5 个 token 分叉(→4)、完全相同(→8)、序列总长不足一页(→0)。
        """
        self._assert_match([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 9, 8], 4, 4)
        self._assert_match(
            [1, 2, 3, 4], [1, 9, 3, 4], 4, 0
        )  # diverge inside first page
        self._assert_match([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 9, 6, 7, 8], 4, 4)
        self._assert_match([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], 4, 8)
        self._assert_match([1, 2, 3], [1, 2, 3], 4, 0)  # shorter than one page

    def test_match_long_keys_exponential_search(self):
        """长 key 上的匹配正确性，专门覆盖“倍增窗口 + 二分”那条快路径。

        base = [0..1999] 各位置取值互不相同，把 b 在 div 处置成 -1 后，
        两者必然恰好在 div 处首次分叉，期望公共长度 = div 向下取整到页。
        div 取值刻意压在倍增窗口的边界上：1、2、63/64/65、127/128、
        511/512/513，以及靠末尾的 1234、1999；page_size 取 1/4/64。
        最后补一个“完整匹配”用例，确保 gallop 能一直推进到序列末尾。
        """
        base = list(range(2000))
        for div in (1, 2, 63, 64, 65, 127, 128, 511, 512, 513, 1234, 1999):
            b = base[:]
            b[div] = -1
            for page_size in (1, 4, 64):
                with self.subTest(div=div, page_size=page_size):
                    self._assert_match(
                        base, b, page_size, (div // page_size) * page_size
                    )
        # Full match of a long key: the gallop must reach the end.
        self._assert_match(base, base[:], 64, (2000 // 64) * 64)

    def test_match_bigram(self):
        """bigram 视图下：L 个原始 token 相同 ⇒ L-1 个二元组相同。

        覆盖：第 3 个 token 分叉(→2)、完全相同(5 token → 4)、
        仅 2 个 token(→1)、长序列第 70 个 token 分叉(69 个 bigram 按页 64 取整 → 64)。
        """
        self._assert_match([1, 2, 3, 4, 5], [1, 2, 3, 9, 5], 1, 2, is_bigram=True)
        self._assert_match([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1, 4, is_bigram=True)
        self._assert_match([1, 2], [1, 2], 1, 1, is_bigram=True)
        # Raw diverge at token 70 -> 69 matching bigrams -> rounded down to 64.
        long_a = list(range(130))
        long_b = list(range(130))
        long_b[70] = -1
        self._assert_match(long_a, long_b, 64, 64, is_bigram=True)


class TestTreeNode(unittest.TestCase):
    """TreeNode 的字段默认值、evicted/backuped 派生状态、host 引用计数与哈希拼接。

    setUp 里重置 TreeNode.counter，使 id 相关断言与用例执行顺序无关
    （counter 是跨用例的全局自增计数器）。
    """

    def setUp(self):
        """Reset the counter before each test."""
        TreeNode.counter = 0

    def test_init_basic(self):
        """新节点的字段默认值（setUp 已把 counter 归零，故 id 从 0 开始）。

        断言：children 为空、parent/key/value 为 None、
        lock_ref/hit_count/host_ref_counter 为 0、host_value/hash_value 为 None。
        """
        node = TreeNode()
        self.assertEqual(node.id, 0)
        self.assertEqual(len(node.children), 0)
        self.assertIsNone(node.parent)
        self.assertIsNone(node.key)
        self.assertIsNone(node.value)
        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(node.hit_count, 0)
        self.assertEqual(node.host_ref_counter, 0)
        self.assertIsNone(node.host_value)
        self.assertIsNone(node.hash_value)

    def test_init_with_id(self):
        """显式传入 id 时只用指定值，不额外消耗计数器语义上的“期望值”。

        传 id=42 后节点 id 即 42；紧接着新建的节点拿到 1，
        说明显式 id 只覆盖本节点，全局计数器仍按自己的节奏自增。
        """
        node = TreeNode(id=42)
        self.assertEqual(node.id, 42)
        node2 = TreeNode()
        self.assertEqual(node2.id, 1)  # Counter was incremented

    def test_evicted_backuped_properties(self):
        """evicted / backuped 只由 value / host_value 是否存在决定。

        参数化四组组合（有 device value × 有 host value）：
        evicted = (value is None)、backuped = (host_value is not None)。
        其中“已淘汰但 host 有备份”（value 缺失 + host_value 存在）
        对应 HiCache 可回载而无需重算的场景。
        """
        test_cases = [
            (False, False, True, False),
            (True, False, False, False),
            (True, True, False, True),
            (False, True, True, True),
        ]

        for (
            has_value,
            has_host_value,
            expected_evicted,
            expected_backuped,
        ) in test_cases:
            with self.subTest(has_value=has_value, has_host_value=has_host_value):
                node = TreeNode()

                if has_value:
                    node.value = torch.tensor([1, 2, 3])
                if has_host_value:
                    node.host_value = torch.tensor([4, 5, 6])

                self.assertEqual(node.evicted, expected_evicted)
                self.assertEqual(node.backuped, expected_backuped)

    def test_protect_release_host(self):
        """protect_host / release_host 维护 host_ref_counter。

        加一次 → 1，释放一次 → 0；计数已为 0 时再释放必须抛 RuntimeError，
        用来暴露 host 侧加解锁不配对的逻辑错误。
        """
        node = TreeNode()
        self.assertEqual(node.host_ref_counter, 0)

        node.protect_host()
        self.assertEqual(node.host_ref_counter, 1)

        node.release_host()
        self.assertEqual(node.host_ref_counter, 0)

        # Test error case
        with self.assertRaises(RuntimeError):
            node.release_host()

    def test_get_last_hash_value(self):
        """get_last_hash_value 取哈希列表的最后一项；尚未计算哈希时返回 None。"""
        node = TreeNode()
        self.assertIsNone(node.get_last_hash_value())

        node.hash_value = ["hash1", "hash2", "hash3"]
        self.assertEqual(node.get_last_hash_value(), "hash3")

    def test_get_prefix_hash_values_not_shared_across_calls(self):
        """回归保护：get_prefix_hash_values 返回的列表不得跨调用共享同一对象。

        背景：下游存储代码会在处理每一页时 **就地** extend 这个返回的列表；
        若实现把结果缓存并直接返回同一个 list，下一次调用就会读到被污染的数据。

        构造 root→n1→n2→n3 链，对 (TreeNode, MambaTreeNode) 两种实现分别验证：
        第一次调用返回 ["h1","h2"]；就地追加 ["h3"] 后，第二次调用仍返回
        ["h1","h2"] 且不是同一个对象；更深的节点能拼出完整前缀 ["h1","h2","h3"]。
        """
        for node_cls in (TreeNode, MambaTreeNode):
            with self.subTest(node_cls=node_cls.__module__):
                root = node_cls()
                n1 = node_cls()
                n1.parent = root
                n1.hash_value = ["h1"]
                n2 = node_cls()
                n2.parent = n1
                n2.hash_value = ["h2"]
                n3 = node_cls()
                n3.parent = n2
                n3.hash_value = ["h3"]

                first = n3.get_prefix_hash_values(n2)
                self.assertEqual(first, ["h1", "h2"])

                # Downstream storage code extends prefix_keys in place while
                # processing pages. A cached list must not be observable by a
                # later call.
                first += ["h3"]

                second = n3.get_prefix_hash_values(n2)
                self.assertEqual(second, ["h1", "h2"])
                self.assertIsNot(second, first)

                n4 = node_cls()
                n4.parent = n3
                n4.hash_value = ["h4"]
                self.assertEqual(n4.get_prefix_hash_values(n3), ["h1", "h2", "h3"])


class TestRadixCache(unittest.TestCase):
    """RadixCache 的核心行为：插入/匹配、淘汰、锁计数账本、页对齐、分裂、隔离与事件。

    统一用 RadixCache.create_simulated(...) 构造不接真实显存的仿真树
    （用例断言 cache.device == cpu）；需要检查 slot 归还/释放时再注入
    mock allocator。setUp 重置 TreeNode.counter。
    """

    def setUp(self):
        """Set up test fixtures."""
        TreeNode.counter = 0

    def test_init_variations(self):
        """三种构造参数组合下字段是否按入参落位。

        组合 (page_size, disable, enable_events) = (1,F,F) / (4,F,T) / (1,T,F)。
        断言 page_size / disable / kv_events.enabled 与入参一致、device 为 cpu、
        根节点存在且根 key 长度为 0（根是空 key 的哨兵节点）。
        """
        test_cases = [
            (1, False, False),
            (4, False, True),
            (1, True, False),
        ]

        for page_size, disable, enable_events in test_cases:
            with self.subTest(
                page_size=page_size, disable=disable, enable_events=enable_events
            ):
                cache = RadixCache.create_simulated(
                    disable=disable,
                    page_size=page_size,
                    enable_kv_cache_events=enable_events,
                )

                self.assertEqual(cache.page_size, page_size)
                self.assertEqual(cache.disable, disable)
                self.assertEqual(cache.kv_events.enabled, enable_events)
                self.assertEqual(cache.device, torch.device("cpu"))
                self.assertIsNotNone(cache.root_node)
                self.assertEqual(len(cache.root_node.key), 0)

    def test_reset(self):
        """reset() 清空整棵树与两个账本。

        先插入 3 个 token 让 total_size > 0，再 reset；
        断言 total_size / evictable_size / protected_size 全部归零。
        """
        cache = RadixCache.create_simulated()

        # Insert some data
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3])),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )
        self.assertGreater(cache.total_size(), 0)

        # Reset
        cache.reset()
        self.assertEqual(cache.total_size(), 0)
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 0)

    def test_insert_and_match_basic(self):
        """最基本的 insert → match_prefix 流程，并覆盖 disable 分支。

        对 disable_cache ∈ {False, True} 各跑一遍：
        - disable=True：insert 返回 prefix_len=0 且不建树（total_size 仍为 0）；
        - disable=False：首次插入无可复用前缀（prefix_len=0），
          total_size 与 evictable_size 都等于 3；
          用等长 key 匹配取回全部 3 个 slot，且与写入值一致；
          用更短的 key [1,2] 匹配只返回前 2 个 slot（前缀匹配语义）。
        """
        for disable_cache in [False, True]:
            with self.subTest(disable_cache=disable_cache):
                cache = RadixCache.create_simulated(disable=disable_cache)

                key = RadixKey(array("q", [1, 2, 3]))
                value = torch.tensor([10, 20, 30], dtype=torch.int64)
                result = cache.insert(InsertParams(key=key, value=value))
                prefix_len = result.prefix_len

                if disable_cache:
                    self.assertEqual(prefix_len, 0)
                    self.assertEqual(cache.total_size(), 0)
                    continue

                self.assertEqual(prefix_len, 0)  # No existing prefix
                self.assertEqual(cache.total_size(), 3)
                self.assertEqual(cache.evictable_size(), 3)

                # Test match_prefix
                result = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3])))
                )
                self.assertEqual(len(result.device_indices), 3)
                torch.testing.assert_close(result.device_indices, value)

                # Test partial match
                result = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", [1, 2])))
                )
                self.assertEqual(len(result.device_indices), 2)
                torch.testing.assert_close(
                    result.device_indices, torch.tensor([10, 20], dtype=torch.int64)
                )

    def test_insert_with_none_value(self):
        """value=None 时的兜底路径：直接用 token_ids 充当 slot 索引。

        只关心“不崩且计入树大小”，故断言 prefix_len=0、total_size=3。
        该分支主要服务调试/测试（见 radix_cache.insert 里的兜底实现）。
        """
        cache = RadixCache.create_simulated()

        key = RadixKey(array("q", [1, 2, 3]))
        result = cache.insert(InsertParams(key=key, value=None))
        prefix_len = result.prefix_len

        # When None is passed, it should create value from token_ids
        self.assertEqual(prefix_len, 0)
        self.assertEqual(cache.total_size(), 3)

    def test_total_size(self):
        """total_size 累计树中所有在册 token 数。

        空树为 0；插入 3 token 后为 3；再插入一条与之**无公共前缀**的
        2 token 序列后为 5（两条分支的 token 数相加）。
        """
        cache = RadixCache.create_simulated()

        self.assertEqual(cache.total_size(), 0)

        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3])),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )
        self.assertEqual(cache.total_size(), 3)

        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [4, 5])),
                value=torch.tensor([40, 50], dtype=torch.int64),
            )
        )
        self.assertEqual(cache.total_size(), 5)

    def test_cache_unfinished_req_deferred_free_owns_original_indices(self):
        """回归保护：cache_unfinished_req 的“重复段延迟释放 + 索引回写”语义。

        场景：KV 池里同时存在两份内容相同的 slot——
        tree_indices 已被树持有，request_indices 是当前请求自己分配的。

        步骤：先以 tree_indices 把该段 token 插入树；再让缓存重新匹配同一 key
        （命中树）→ 请求应改用树持有的 tree_indices，并在 free_group 内
        延迟释放自己那份 request_indices。

        断言：allocator 可用量恰好增加 request_indices 的数量；free_pages 末尾
        就是被归还的 request_indices（顺序一致）；req_to_token[0] 被改写为
        tree_indices，即请求后续与树共用同一批 slot（不重复占显存）。
        """

        class ReqToTokenPool:
            def __init__(self, row):
                self.req_to_token = row.unsqueeze(0)

            def write(self, indices, values):
                self.req_to_token[indices] = values

        allocator = TokenToKVPoolAllocator(
            size=16,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        cache = RadixCache.create_simulated(mock_allocator=allocator)
        token_ids = array("q", [1, 2, 3])
        tree_indices = allocator.alloc(3)
        request_indices = allocator.alloc(3)
        assert tree_indices is not None
        assert request_indices is not None
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", token_ids)),
                value=tree_indices,
            )
        )
        cache.req_to_token_pool = ReqToTokenPool(request_indices.clone())
        req = unittest.mock.Mock(
            kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=0),
            extra_key=None,
            cache_salt=None,
            priority=0,
            last_node=cache.root_node,
        )
        req.get_fill_ids.return_value = token_ids

        available_before_free = allocator.available_size()
        allocator.free_group_begin()
        cache.cache_unfinished_req(req)
        allocator.free_group_end()

        self.assertEqual(
            allocator.available_size(),
            available_before_free + request_indices.numel(),
        )
        torch.testing.assert_close(allocator.free_pages[-3:], request_indices)
        torch.testing.assert_close(
            cache.req_to_token_pool.req_to_token[0], tree_indices
        )

    def test_kv_cache_events(self):
        """KV 事件开关与事件字段自洽性。

        对 (page_size, enable_events) = (1,T) / (2,T) / (1,F) 各跑一遍：
        - 开启时插入后能取到 ≥1 条事件，且其中包含 BlockStored；
        - 每条 BlockStored 满足 block_size ≤ page_size、
          len(token_ids) == block_size × len(block_hashes)（字段自洽，router 可据此还原块）；
        - 关闭时 take_events() 返回空列表。
        """
        test_cases = [
            (1, True),
            (2, True),
            (1, False),
        ]

        for page_size, enable_events in test_cases:
            with self.subTest(page_size=page_size, enable_events=enable_events):
                cache = RadixCache.create_simulated(
                    page_size=page_size, enable_kv_cache_events=enable_events
                )

                # Insert data
                cache.insert(
                    InsertParams(key=RadixKey(array("q", [1, 2, 3, 4, 5])), value=None)
                )

                # Take events
                events = cache.take_events()

                if enable_events:
                    self.assertGreater(len(events), 0)
                    # Verify events include BlockStored events (there might be other event types)
                    block_stored_events = [
                        e for e in events if isinstance(e, BlockStored)
                    ]
                    self.assertGreater(len(block_stored_events), 0)
                    for event in block_stored_events:
                        self.assertLessEqual(event.block_size, page_size)
                        self.assertEqual(
                            len(event.token_ids),
                            event.block_size * len(event.block_hashes),
                        )
                else:
                    self.assertEqual(len(events), 0)

    def test_kv_cache_events_with_eviction(self):
        """淘汰会产生 BlockRemoved，且其哈希与先前的 BlockStored 对应。

        page_size=2、4 个 token → 2 个块。先插入再 evict 全部，断言：
        淘汰量 ≥ 序列长度；事件里既有 BlockStored 也有 BlockRemoved；
        stored 的块哈希恰好 2 个，且 remove 的 block_hashes 与它们完全一致
        ——说明上报“删掉的块”就是先前“存过的块”，router 侧可按哈希抵消。
        """
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache.create_simulated(
            mock_allocator=mock_allocator,
            page_size=2,
            enable_kv_cache_events=True,
        )

        # Insert and then evict data
        seq = [1, 2, 3, 4]
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", seq)),
                value=torch.tensor([10, 20, 30, 40], dtype=torch.int64),
            )
        )
        result = cache.evict(EvictParams(num_tokens=len(seq)))
        self.assertIsInstance(result, EvictResult)
        self.assertGreaterEqual(
            result.num_tokens_evicted,
            len(seq),
            f"evicted {result.num_tokens_evicted} tokens, expected at least {len(seq)}",
        )

        # Take events - should include both store and remove events
        events = cache.take_events()
        self.assertGreater(len(events), 0)

        # Check event types
        event_types = [type(event).__name__ for event in events]
        self.assertIn("BlockStored", event_types)

        stored_hashes = [
            block_hash
            for event in events
            if isinstance(event, BlockStored)
            for block_hash in event.block_hashes
        ]
        self.assertEqual(len(stored_hashes), 2)

        # Verify BlockRemoved event content
        remove_events = [e for e in events if isinstance(e, BlockRemoved)]
        self.assertEqual(len(remove_events), 1)
        self.assertEqual(remove_events[0].block_hashes, stored_hashes)

    def test_extra_key_isolation(self):
        """不同 extra_key 下的相同 token 序列必须互相隔离，绝不共享节点。

        同一段 [1,2,3] 分别以 extra_key = "key1" / "key2" / None 插入（值不同）。
        断言：三次 match 各自只命中自己的那组 slot；用不存在的 extra_key
        （"nonexistent"）匹配返回 0 长度，不会“串门”命中别人的缓存。
        """
        cache = RadixCache.create_simulated()

        # Insert same token sequence with different extra keys
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3]), "key1"),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3]), "key2"),
                value=torch.tensor([40, 50, 60], dtype=torch.int64),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3]), None),
                value=torch.tensor([70, 80, 90], dtype=torch.int64),
            )
        )

        # Keys with different extra_key should not match each other
        result1 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), "key1"))
        )
        result2 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), "key2"))
        )
        result3 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), None))
        )
        result4 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), "nonexistent"))
        )

        # Each should match only its own data
        self.assertEqual(len(result1.device_indices), 3)
        torch.testing.assert_close(
            result1.device_indices, torch.tensor([10, 20, 30], dtype=torch.int64)
        )

        self.assertEqual(len(result2.device_indices), 3)
        torch.testing.assert_close(
            result2.device_indices, torch.tensor([40, 50, 60], dtype=torch.int64)
        )

        self.assertEqual(len(result3.device_indices), 3)
        torch.testing.assert_close(
            result3.device_indices, torch.tensor([70, 80, 90], dtype=torch.int64)
        )

        # Non-existent extra_key should not match
        self.assertEqual(len(result4.device_indices), 0)

    def test_cache_salt_isolation_is_independent_of_extra_key(self):
        """extra_key 与 cache_salt 是两维独立命名空间，不能拼接后互相错配。

        刻意构造极易混淆的一对：
        (extra_key="bc", cache_salt="a") 与 (extra_key="c", cache_salt="ab")。
        若实现把两者简单拼成一个字符串，这两组会塌缩成同一命名空间；
        正确实现下各自命中自己的值（[10,20,30] 与 [40,50,60]）。
        """
        cache = RadixCache.create_simulated()
        tokens = array("q", [1, 2, 3])

        cache.insert(
            InsertParams(
                key=RadixKey(tokens, extra_key="bc", cache_salt="a"),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(tokens, extra_key="c", cache_salt="ab"),
                value=torch.tensor([40, 50, 60], dtype=torch.int64),
            )
        )

        first = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(tokens, extra_key="bc", cache_salt="a"))
        )
        second = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(tokens, extra_key="c", cache_salt="ab"))
        )
        torch.testing.assert_close(
            first.device_indices, torch.tensor([10, 20, 30], dtype=torch.int64)
        )
        torch.testing.assert_close(
            second.device_indices, torch.tensor([40, 50, 60], dtype=torch.int64)
        )

    def test_cache_salt_is_included_in_store_and_remove_events(self):
        """cache_salt 参与事件哈希；同一 salt 下 store/remove 的哈希必须一致。

        带 salt="tenant-a" 插入 4 个 token（page_size=2 → 2 块）后全部淘汰：
        - 只有 1 条 BlockStored，其 metadata.cache_salt="tenant-a"、
          parent_block_hash=None、含 2 个块哈希；
        - BlockRemoved 的块哈希与 BlockStored 完全一致（同 salt 可对应抵消）；
        对照：不带 salt 插入同一段 token，得到的块哈希与带 salt 的不同，
        说明 salt 确实进入了哈希计算（外部 KV 存储层面同样隔离）。
        """
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")
        cache = RadixCache.create_simulated(
            mock_allocator=mock_allocator,
            page_size=2,
            enable_kv_cache_events=True,
        )
        tokens = array("q", [1, 2, 3, 4])
        cache.insert(
            InsertParams(
                key=RadixKey(tokens, cache_salt="tenant-a"),
                value=torch.tensor([10, 20, 30, 40], dtype=torch.int64),
            )
        )
        cache.evict(EvictParams(num_tokens=len(tokens)))
        events = cache.take_events()
        stored = [event for event in events if isinstance(event, BlockStored)]
        removed = [event for event in events if isinstance(event, BlockRemoved)]

        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].metadata.cache_salt, "tenant-a")
        self.assertEqual(stored[0].parent_block_hash, None)
        self.assertEqual(len(stored[0].block_hashes), 2)
        self.assertEqual(removed[0].block_hashes, stored[0].block_hashes)

        unsalted = RadixCache.create_simulated(page_size=2, enable_kv_cache_events=True)
        unsalted.insert(InsertParams(key=RadixKey(tokens), value=None))
        unsalted_hashes = [
            block_hash
            for event in unsalted.take_events()
            if isinstance(event, BlockStored)
            for block_hash in event.block_hashes
        ]
        self.assertNotEqual(unsalted_hashes, stored[0].block_hashes)

    def test_cache_salt_event_hashes_are_preserved_across_node_split(self):
        """节点分裂前后，事件哈希按页切分后拼接必须与原来完全一致。

        先插入 [1,2,3,4]（salt=tenant-a）并记录末节点的 event_hash_value；
        再插入 [1,2,9,10] 触发在 [1,2] 之后分裂。断言：
        parent.event_hash_value + child.event_hash_value == 原来的哈希序列。
        意义：分裂只改变树的形状，不能让外部 KV 事件的“块身份”发生变化。
        """
        cache = RadixCache.create_simulated(page_size=2, enable_kv_cache_events=True)
        original = RadixKey(array("q", [1, 2, 3, 4]), cache_salt="tenant-a")
        cache.insert(
            InsertParams(
                key=original,
                value=torch.tensor([10, 20, 30, 40], dtype=torch.int64),
            )
        )
        original_node = cache.match_prefix(
            MatchPrefixParams(key=original)
        ).last_device_node
        original_hashes = list(original_node.event_hash_value)

        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 9, 10]), cache_salt="tenant-a"),
                value=torch.tensor([10, 20, 90, 100], dtype=torch.int64),
            )
        )
        split_child = cache.match_prefix(
            MatchPrefixParams(key=original)
        ).last_device_node
        split_parent = split_child.parent

        self.assertEqual(
            split_parent.event_hash_value + split_child.event_hash_value,
            original_hashes,
        )

    def test_lock_ref_operations(self):
        """inc_lock_ref / dec_lock_ref 在 evictable 与 protected 两个账本之间搬运 token。

        插入 3 个 token 后取末节点：加锁 → protected +3、evictable -3
        （锁的是“根到该节点”的整条路径）；解锁 → 两个账本回到原值。
        """
        cache = RadixCache.create_simulated()

        # Insert sequence
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3])),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )

        # Get node
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3])))
        )
        node = result.last_device_node

        initial_evictable = cache.evictable_size()
        initial_protected = cache.protected_size()

        # Lock the node
        cache.inc_lock_ref(node)
        self.assertEqual(cache.protected_size(), initial_protected + 3)
        self.assertEqual(cache.evictable_size(), initial_evictable - 3)

        # Unlock the node
        cache.dec_lock_ref(node)
        self.assertEqual(cache.protected_size(), initial_protected)
        self.assertEqual(cache.evictable_size(), initial_evictable)

    def test_evict_functionality(self):
        """evict 会真正调用 allocator 释放 slot，并缩小树。

        用 mock allocator 插入两条互不共享前缀的 2-token 序列，请求淘汰 2 个 token：
        断言返回 EvictResult 且 num_tokens_evicted ≥ 2；free_segment 被调用过；
        total_size 变小。
        """
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache.create_simulated(mock_allocator=mock_allocator)

        # Insert sequences
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2])),
                value=torch.tensor([10, 20], dtype=torch.int64),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [3, 4])),
                value=torch.tensor([30, 40], dtype=torch.int64),
            )
        )

        initial_size = cache.total_size()

        # Evict some tokens
        result = cache.evict(EvictParams(num_tokens=2))
        self.assertIsInstance(result, EvictResult)
        self.assertGreaterEqual(
            result.num_tokens_evicted,
            2,
            f"evicted {result.num_tokens_evicted} tokens, expected at least 2",
        )

        # Should have called free_segment and reduced size
        mock_allocator.free_segment.assert_called()
        self.assertLess(cache.total_size(), initial_size)

    def test_page_alignment_boundary(self):
        """命中长度必须按页对齐，不足一页的尾部不能进树。

        三组 (page_size, seq_len) = (1,5) / (2,5) / (4,6)，长度都刻意不是页的整数倍。
        断言：插入后匹配能命中（长度 > 0），且 match_len % page_size == 0。
        """
        test_cases = [
            (1, 5),
            (2, 5),
            (4, 6),
        ]

        for page_size, sequence_length in test_cases:
            with self.subTest(page_size=page_size, sequence_length=sequence_length):
                cache = RadixCache.create_simulated(page_size=page_size)

                tokens = list(range(sequence_length))
                key = RadixKey(array("q", tokens))
                cache.insert(
                    InsertParams(
                        key=key,
                        value=torch.tensor(tokens, dtype=torch.int64)[: len(key)],
                    )
                )

                result = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", tokens)))
                )
                self.assertGreater(len(result.device_indices), 0)

                # Match length should be page-aligned
                match_len = len(result.device_indices)
                self.assertEqual(match_len % page_size, 0)

    def test_advanced_prefix_match_with_node_splits(self):
        """匹配过程中触发节点分裂时的正确性，以及“分裂只切索引、不增数据”不变式。

        page_size ∈ {1, 2} 各跑一遍：
        1. 先插入长序列 seq1（8 token，值 = token×10），再插入分叉序列 seq2=[1,2,9,10]，
           树中形成内部节点，此时 total_size 应为 10（8 + 2）；
        2. 用 [1,2,3,4,999,1000] 匹配 → 在原节点内部第 4 个 token 处分裂，
           断言返回 val1[:4]，且 total_size 不变（不搬 KV、不新增数据）；
        3. 再完整匹配 seq1，仍能取回全部 8 个 slot；
        4. 用 [1,2,3,4,5,6,777,888] 触发一次更深的分裂，返回 val1[:6]，total_size 仍不变；
        5. 匹配短分支 seq2 返回它自己的 val2。
        """
        for page_size in [1, 2]:
            with self.subTest(page_size=page_size):
                cache = RadixCache.create_simulated(page_size=page_size)

                # Insert a long sequence that will be split later.
                seq1 = [1, 2, 3, 4, 5, 6, 7, 8]
                val1 = torch.tensor([x * 10 for x in seq1], dtype=torch.int64)
                cache.insert(InsertParams(key=RadixKey(array("q", seq1)), value=val1))

                # Insert a diverging branch to create an internal node on the path.
                seq2 = [1, 2, 9, 10]
                val2 = torch.tensor([x * 10 for x in seq2], dtype=torch.int64)
                cache.insert(InsertParams(key=RadixKey(array("q", seq2)), value=val2))
                print(cache.pretty_print())

                baseline_total = cache.total_size()
                expected_total = 10  # 8 + 2
                self.assertEqual(baseline_total, expected_total)

                # Match that causes a split inside an existing node:
                # take first 4 tokens of seq1, then diverge.
                query1 = [1, 2, 3, 4, 999, 1000]
                result1 = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", query1)))
                )
                torch.testing.assert_close(result1.device_indices, val1[:4])
                # No data change after structural split during matching.
                self.assertEqual(cache.total_size(), baseline_total)

                # Full match of the long sequence still returns the full indices.
                result_full = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", seq1)))
                )
                torch.testing.assert_close(result_full.device_indices, val1)

                # Another split deeper on the path (after matching 6 tokens, then diverge).
                query2 = [1, 2, 3, 4, 5, 6, 777, 888]
                result2 = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", query2)))
                )
                torch.testing.assert_close(result2.device_indices, val1[:6])
                self.assertEqual(cache.total_size(), baseline_total)

                # Matching the short diverging branch should return exactly its indices.
                result_branch = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", seq2)))
                )
                torch.testing.assert_close(result_branch.device_indices, val2)

    def test_hash_value_storage(self):
        """hash_value 惰性计算：插入时不算，事件发射时才生成，并按页切分。

        page_size=4、8 个 token → 2 页（2 个哈希）。插入后调用 take_events()
        触发事件（哈希在此惰性计算），再沿 children 找到该节点，
        断言 hash_value 非 None 且长度为 2。
        """
        cache = RadixCache.create_simulated(
            page_size=4,
            enable_kv_cache_events=True,
        )

        # Insert a sequence
        cache.insert(
            InsertParams(key=RadixKey(array("q", [1, 2, 3, 4, 5, 6, 7, 8])), value=None)
        )

        # Trigger event emission to compute hash_value lazily
        cache.take_events()

        # Find the inserted node (traverse from root)
        node = cache.root_node
        for i in range(0, 8, 4):  # page_size=4, so 2 pages
            child_key = tuple([1, 2, 3, 4][:4]) if i == 0 else tuple([5, 6, 7, 8][:4])
            if child_key in node.children:
                node = node.children[child_key]
                break

        # Verify hash_value is set (computed lazily during event emission)
        self.assertIsNotNone(node.hash_value)
        # Should have 2 pages (8 tokens / 4 page_size)
        self.assertEqual(len(node.hash_value), 2)

    def test_hash_value_repeating_tokens(self):
        """重复 token 模式必须得到不同的“位置感知”哈希。

        插入 [1,2,3,4,1,2,3,4]（page_size=4 → 两页内容完全相同）：
        - 两个块合并为 1 条 BlockStored，含 2 个块哈希；
        - 但两个块哈希必须不同——哈希带序列位置（前缀链），否则不同位置的
          同内容块会被外部 KV 存储错误复用；
        - 合并后仍保留最初的 parent_block_hash（None）。
        """
        cache = RadixCache.create_simulated(
            page_size=4,
            enable_kv_cache_events=True,
        )

        # Insert a sequence with repeating token pattern: [1,2,3,4, 1,2,3,4]
        cache.insert(
            InsertParams(key=RadixKey(array("q", [1, 2, 3, 4, 1, 2, 3, 4])), value=None)
        )

        events = cache.take_events()
        block_stored_events = [e for e in events if isinstance(e, BlockStored)]

        # The two pages should be represented by one parent-linked store event.
        self.assertEqual(len(block_stored_events), 1)
        self.assertEqual(len(block_stored_events[0].block_hashes), 2)

        # Extract block hashes
        block_hash_1, block_hash_2 = block_stored_events[0].block_hashes

        # The two blocks should have DIFFERENT hashes despite same content
        # because they are at different positions (sequence-aware hashing)
        self.assertNotEqual(
            block_hash_1,
            block_hash_2,
            "Repeating token patterns should get different sequence-aware hashes",
        )

        # The coalesced event keeps the original root parent and ordered hashes.
        self.assertIsNone(block_stored_events[0].parent_block_hash)

    def test_hash_value_split(self):
        """节点分裂后 hash_value 按页正确切分。

        page_size=2：先插入 [1,2,3,4] 并 take_events() 让哈希先算出来，
        再插入 [1,2,5,6] 在 [1,2] 之后分裂。断言 [1,2] 那个节点的
        hash_value 非 None 且只有 1 页（长度 1）。
        （若分裂前哈希尚未计算，事件阶段会惰性补齐，两条路径都应满足该断言。）
        """
        cache = RadixCache.create_simulated(
            page_size=2,
            enable_kv_cache_events=True,
        )

        # Insert a sequence that will cause a split
        cache.insert(InsertParams(key=RadixKey(array("q", [1, 2, 3, 4])), value=None))
        cache.take_events()  # Clear events and compute hash_value for first node

        # Insert a diverging sequence that will cause a split at page boundary
        cache.insert(InsertParams(key=RadixKey(array("q", [1, 2, 5, 6])), value=None))
        cache.take_events()  # Trigger event emission to compute hash_value

        # Find the split node
        node = cache.root_node
        child_key = tuple([1, 2])
        if child_key in node.children:
            node = node.children[child_key]
            # After split and event emission, hash_value should be computed
            # Note: If hash_value wasn't set before split, it will be computed lazily
            # during event emission. If it was set, it will be split.
            # Either way, after events are emitted, it should be set.
            self.assertIsNotNone(node.hash_value)
            # Should have 1 page (split at page_size=2)
            self.assertEqual(len(node.hash_value), 1)

    def test_memory_allocated(self):
        """大规模插入下的内存占用量级回归（防意外拷贝/泄漏）。

        数据：10000 条序列 = 10000 token 公共前缀 + 100 token 各不相同后缀，
        值张量放在 get_device() 上；插入前记录 memory_allocated 基线。
        断言：插入导致的显存增量 < total_size() × 4 字节 × 2
        （4 字节 = int32 索引元素大小）；若树实现出现整段拷贝或泄漏，
        增量会远超该上界。
        注：本用例依赖加速器的 memory_allocated()，所以整份文件挂在单卡 GPU CI。
        """
        keys, values = [], []

        num_seqs = 10000
        vocab_size = 1000
        base_prefix_len = 10000
        suffix_len = 100

        torch_allocated_before = torch.get_device_module().memory_allocated()

        # build dataset with common prefix
        common_prefix = [
            random.randint(1, vocab_size - 1) for _ in range(base_prefix_len)
        ]
        for _ in range(num_seqs):
            suffix = [random.randint(1, vocab_size - 1) for _ in range(suffix_len)]
            seq = common_prefix + suffix
            keys.append(seq)
            values.append(torch.zeros(len(seq), device=get_device(), dtype=torch.int32))

        cache: RadixCache = RadixCache.create_simulated()

        for key, value in zip(keys, values):
            cache.insert(InsertParams(key=RadixKey(array("q", key)), value=value))

        del values

        torch_allocated = (
            torch.get_device_module().memory_allocated() - torch_allocated_before
        )
        cache_size_bytes = cache.total_size() * 4
        print(f"\nCache size (MB): {cache_size_bytes / (1024 * 1024)}")
        print(f"Torch allocated (MB): {torch_allocated / (1024 * 1024)}")

        # The cache size should be within reasonable bounds of the actual allocated memory.
        self.assertLess(torch_allocated, cache_size_bytes * 2)


if __name__ == "__main__":
    unittest.main()
