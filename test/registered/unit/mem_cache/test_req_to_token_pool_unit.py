"""Unit tests for the base-class ``ReqToTokenPool`` (level-1 request -> KV slot map).

Other tests only use this pool as scaffolding or fake it; this file pins the
contract of the real class:

- construction: padding row 0, ``free_slots`` starting at 1, dtypes/shapes of
  ``req_to_token`` and ``req_generation``;
- ``alloc`` / ``alloc_rows``: tail-pop order, ``None`` on exhaustion, generation
  bump, row reuse for chunked / committed requests, ``need_size == 0``;
- ``free`` / ``free_rows``: return-to-tail order, ``req_pool_idx`` reset;
- ``write``: advanced-index stores (token slice, per-row column, python list);
- ``clear``;
- the aux-cache contract (``attach_aux_cache``, ``schedulable_token_capacity``,
  ``alloc_aux_to_lengths``, ``reset_aux_cache_allocator``).

Row 0 is the CUDA-graph padding row and must never be handed out.

中文说明：本文件只测基类 ``ReqToTokenPool``（两级映射的第一级）。其它测试要么把它
当脚手架，要么用假类替换，所以这里把真实类的契约固定下来：构造（padding 行 0、空闲表
从 1 开始）、alloc/alloc_rows（尾部取行、耗尽返回 None、generation 递增、行复用）、
free/free_rows（归还到尾部、req 字段清理）、write（写 KV 槽位映射）、clear，以及
aux cache 的转发契约。0 号行是 CUDA Graph 的 padding 行，永远不能被分配出去。

运行方式（在仓库根目录 /home/sankuai/sglang 下）::

    # 标准方式（CI 同款，推荐）
    python -m pytest test/registered/unit/mem_cache/test_req_to_token_pool_unit.py -v

    # 直接跑文件也可以（文件末尾有 unittest.main()）
    cd python && python3 ../test/registered/unit/mem_cache/test_req_to_token_pool_unit.py -v

本机开发容器的额外说明：site-packages 里的 ``deep_gemm`` 版本过旧（只有
``utils.py``，缺少新代码需要的 ``deep_gemm.utils.layout``），直接 ``import sglang``
会失败；且设置 ``PYTHONPATH`` 会导致 ``numpy`` 找不到（``/usr/local/lib/packages``
不在路径里）。因此需要先 ``cd`` 到 ``python/`` 让 ``import sglang`` 命中工作区源码，
再把匹配的 DeepGEMM 插到 ``sys.path`` 前面::

    cd /home/sankuai/sglang/python && python3 -c "import sys; sys.path.insert(0,'/home/sankuai/tb4.0/dsv41-pd/DeepGEMM/build'); import pytest; sys.exit(pytest.main(['/home/sankuai/sglang/test/registered/unit/mem_cache/test_req_to_token_pool_unit.py','-v']))"

测试本身不依赖上述路径，CI 里按标准方式执行即可。
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_req(req_pool_idx=None, *, kv_committed_len=0, inflight_middle_chunks=0):
    """构造最小化的 Req 替身（鸭子类型）。

    池子只会访问 ``inflight_middle_chunks``、``kv.req_pool_idx`` 和
    ``kv.kv_committed_len`` 这三个字段，所以不需要真正的 ``Req``。
    """
    return SimpleNamespace(
        inflight_middle_chunks=inflight_middle_chunks,
        kv=SimpleNamespace(
            req_pool_idx=req_pool_idx,
            kv_committed_len=kv_committed_len,
        ),
    )


def _make_pool(size=4, max_context_len=8):
    """在 CPU 上构造一个真实的 ``ReqToTokenPool``（关闭 memory saver，便于单测）。"""
    return ReqToTokenPool(
        size=size,
        max_context_len=max_context_len,
        device="cpu",
        enable_memory_saver=False,
    )


class _RecordingAuxCache:
    """记录收到的每一次调用，供断言核对 aux cache 契约。"""

    dense_capacity = 42  # 供 schedulable_token_capacity 读取的逻辑容量

    def __init__(self):
        """初始化各调用记录：释放的行号、clear / reset 次数、alloc_to_lengths 参数。"""
        self.freed = []
        self.clear_calls = 0
        self.reset_calls = 0
        self.alloc_calls = []

    def free(self, index):
        """记录一次按行释放（free_rows 会逐行回调这里）。"""
        self.freed.append(index)

    def clear(self):
        """记录一次 clear 调用。"""
        self.clear_calls += 1

    def reset_allocator(self):
        """记录一次 reset_allocator 调用。"""
        self.reset_calls += 1

    def alloc_to_lengths(self, *, req_pool_indices_cpu, target_seq_lens_cpu):
        """记录一次「按长度分配」的入参（行号与目标序列长度，转成 list 便于比较）。"""
        self.alloc_calls.append(
            (req_pool_indices_cpu.tolist(), target_seq_lens_cpu.tolist())
        )


class TestReqToTokenPoolConstruction(CustomTestCase):
    """构造阶段：padding 行 0、空闲表初值、张量元信息。"""

    def test_padding_row_and_initial_free_slots(self):
        """构造后表有 size + 1 行（0 号为 padding），空闲表从 1 开始。"""
        pool = _make_pool(size=4, max_context_len=8)

        # 第 0 行是 CUDA Graph 的 padding 行：表有 size + 1 行，空闲表从 1 开始。
        self.assertEqual(pool._alloc_size, 5)
        self.assertEqual(pool.size, 4)
        self.assertEqual(pool.max_context_len, 8)
        self.assertEqual(pool.free_slots, [1, 2, 3, 4])
        self.assertEqual(pool.available_size(), 4)
        self.assertTrue(torch.all(pool.req_to_token[0] == 0))

    def test_tensor_dtypes_and_shapes(self):
        """``req_to_token`` 是 int32，``req_generation`` 是 int64，形状与 _alloc_size 对齐。"""
        pool = _make_pool(size=4, max_context_len=8)

        self.assertEqual(pool.req_to_token.shape, (5, 8))
        self.assertEqual(pool.req_to_token.dtype, torch.int32)
        self.assertEqual(pool.req_generation.shape, (5,))
        self.assertEqual(pool.req_generation.dtype, torch.int64)
        self.assertTrue(torch.all(pool.req_generation == 0))


class TestReqToTokenPoolAlloc(CustomTestCase):
    """alloc / alloc_rows：取行顺序、耗尽行为、generation 递增、行复用。"""

    def test_alloc_takes_rows_from_the_tail(self):
        """alloc 从空闲表尾部取行（[1,2,3,4] 取 2 个得到 [3,4]），并回填 req.kv.req_pool_idx。"""
        pool = _make_pool(size=4)
        r1, r2 = _make_req(), _make_req()

        indices = pool.alloc([r1, r2])

        self.assertEqual(indices, [3, 4])
        self.assertEqual(r1.kv.req_pool_idx, 3)
        self.assertEqual(r2.kv.req_pool_idx, 4)
        self.assertEqual(pool.free_slots, [1, 2])
        self.assertEqual(pool.available_size(), 2)

    def test_alloc_returns_none_when_exhausted_and_leaves_reqs_unbound(self):
        """空闲行不够时整体返回 None，且不给任何请求绑定行号（表状态不变）。"""
        pool = _make_pool(size=1)
        r1, r2 = _make_req(), _make_req()

        self.assertIsNone(pool.alloc([r1, r2]))

        self.assertIsNone(r1.kv.req_pool_idx)
        self.assertIsNone(r2.kv.req_pool_idx)
        self.assertEqual(pool.free_slots, [1])

    def test_alloc_rows_bumps_generation_without_binding_a_req(self):
        """alloc_rows 只取行并递增这些行的 generation，不绑定 Req（beam 成员行走这条路）。"""
        pool = _make_pool(size=4)

        rows = pool.alloc_rows(2)

        self.assertEqual(rows, [3, 4])
        self.assertEqual(pool.req_generation[3].item(), 1)
        self.assertEqual(pool.req_generation[4].item(), 1)
        self.assertEqual(pool.req_generation[1].item(), 0)

    def test_alloc_rows_none_when_need_exceeds_free(self):
        """need_size 超过空闲行数时直接返回 None，什么都不改。"""
        pool = _make_pool(size=1)

        self.assertIsNone(pool.alloc_rows(2))
        self.assertEqual(pool.free_slots, [1])

    def test_alloc_rows_zero_returns_empty_list(self):
        """need_size == 0 需单独处理：free_slots[-0:] 会取到整个列表，所以显式返回 []。"""
        # free_slots[-0:] 会取到整个列表，所以 need_size == 0 要单独处理。
        pool = _make_pool(size=4)

        self.assertEqual(pool.alloc_rows(0), [])
        self.assertEqual(pool.free_slots, [1, 2, 3, 4])

    def test_padding_row_is_never_allocated(self):
        """即使把 size 个行全部分完，也绝不会把 0 号 padding 行发出去。"""
        pool = _make_pool(size=4)

        rows = pool.alloc_rows(4)

        self.assertEqual(sorted(rows), [1, 2, 3, 4])
        self.assertNotIn(0, rows)
        self.assertEqual(pool.free_slots, [])

    def test_alloc_reuses_row_of_chunked_request(self):
        """分块 prefill 的后续块复用原来的行，只有新请求才从尾部取新行。"""
        pool = _make_pool(size=4)
        r1, r2 = _make_req(), _make_req()
        self.assertEqual(pool.alloc([r1, r2]), [3, 4])

        r1.inflight_middle_chunks = 1  # 分块 prefill 的后续块
        r3 = _make_req()

        # r1 继续用第 3 行，只有新来的 r3 取一个新行（从尾部取）。
        self.assertEqual(pool.alloc([r1, r3]), [3, 2])
        self.assertEqual(pool.free_slots, [1])

    def test_alloc_reuses_row_of_request_with_committed_kv(self):
        """已有已提交 KV 的请求复用原行；此时 need_size == 0，不取行也不递增 generation。"""
        pool = _make_pool(size=4)
        req = _make_req(req_pool_idx=3, kv_committed_len=5)

        # need_size == 0：不取行，也不递增 generation。
        self.assertEqual(pool.alloc([req]), [3])
        self.assertEqual(pool.free_slots, [1, 2, 3, 4])
        self.assertEqual(pool.req_generation[3].item(), 0)

    def test_alloc_rejects_reuse_without_chunk_or_committed_kv(self):
        """既非分块、又没有已提交 KV 的请求带着 req_pool_idx 进来会触发断言。"""
        pool = _make_pool(size=4)
        req = _make_req(req_pool_idx=3)  # 既非分块、也没有已提交 KV

        with self.assertRaises(AssertionError):
            pool.alloc([req])

    def test_rows_are_unique_while_held(self):
        """在途请求占用的行号互不相同；释放后该行立即可被下一个请求复用。"""
        pool = _make_pool(size=4)
        reqs = [_make_req() for _ in range(4)]

        self.assertEqual(sorted(pool.alloc(reqs)), [1, 2, 3, 4])
        self.assertIsNone(pool.alloc([_make_req()]))

        pool.free(reqs[0])
        self.assertEqual(pool.alloc([_make_req()]), [1])


class TestReqToTokenPoolFree(CustomTestCase):
    """free / free_rows：归还顺序、req 字段清理、行复用与 generation。"""

    def test_free_returns_row_to_tail_and_clears_req(self):
        """free 把行号追加回空闲表尾部，并把 req.kv.req_pool_idx 置为 None。"""
        pool = _make_pool(size=4)
        r1, r2 = _make_req(), _make_req()
        pool.alloc([r1, r2])

        pool.free(r1)

        self.assertEqual(pool.free_slots, [1, 2, 3])
        self.assertIsNone(r1.kv.req_pool_idx)
        self.assertEqual(r2.kv.req_pool_idx, 4)

    def test_freed_row_is_reused_next_and_generation_bumps(self):
        """刚释放的行下一次分配就会被复用，且 generation 再 +1（用于识别行被复用）。"""
        pool = _make_pool(size=4)
        r1 = _make_req()
        (row,) = pool.alloc([r1])
        self.assertEqual(row, 4)
        self.assertEqual(pool.req_generation[4].item(), 1)

        pool.free(r1)
        r2 = _make_req()
        self.assertEqual(pool.alloc([r2]), [4])
        self.assertEqual(pool.req_generation[4].item(), 2)

    def test_free_requires_a_bound_row(self):
        """free 要求请求确实持有行号（req_pool_idx 不为 None），否则断言失败。"""
        pool = _make_pool(size=4)

        with self.assertRaises(AssertionError):
            pool.free(_make_req())

    def test_free_rows_appends_in_given_order(self):
        """free_rows 按传入顺序追加，不做排序；传入的行必须是本池已分配出去的。"""
        pool = _make_pool(size=4)
        self.assertEqual(pool.alloc_rows(2), [3, 4])

        pool.free_rows([4, 3])  # 顺序原样保留，不重新排序

        self.assertEqual(pool.free_slots, [1, 2, 4, 3])
        self.assertEqual(pool.available_size(), 4)

    def test_free_rows_empty_is_noop(self):
        """free_rows([]) 是空操作，空闲表保持不变。"""
        pool = _make_pool(size=4)

        pool.free_rows([])

        self.assertEqual(pool.free_slots, [1, 2, 3, 4])


class TestReqToTokenPoolWrite(CustomTestCase):
    """write：把 KV 槽位写进映射表的各种索引形式。"""

    def test_write_token_slice_into_row(self):
        """把某请求各 token 的 KV 槽位按切片写进对应行，其它行（含 padding 行）不受影响。"""
        pool = _make_pool(size=4, max_context_len=8)

        pool.write((3, slice(0, 3)), torch.tensor([11, 12, 13], dtype=torch.int32))

        self.assertEqual(pool.req_to_token[3, :4].tolist(), [11, 12, 13, 0])
        # 其它行（含 padding 行）保持不变。
        self.assertTrue(torch.all(pool.req_to_token[0] == 0))
        self.assertTrue(torch.all(pool.req_to_token[4] == 0))

    def test_write_per_row_column_matches_decode_path(self):
        """对齐 decode 路径的写法：req_to_token[req_pool_indices, seq_lens] = out_cache_loc。"""
        # decode 路径的写法：req_to_token[req_pool_indices, seq_lens] = out_cache_loc。
        pool = _make_pool(size=4, max_context_len=8)

        pool.write(
            (torch.tensor([3, 4]), torch.tensor([4, 7])),
            torch.tensor([100, 200], dtype=torch.int32),
        )

        self.assertEqual(pool.req_to_token[3, 4].item(), 100)
        self.assertEqual(pool.req_to_token[4, 7].item(), 200)

    def test_write_accepts_python_list_indices(self):
        """下标可以是 python list，但 values 必须已经是 tensor（write 是裸赋值）且返回 None。"""
        pool = _make_pool(size=4, max_context_len=8)

        # 下标可以是 python list，但 values 必须已经是 tensor。
        returned = pool.write((4, [1, 2]), torch.tensor([7, 8], dtype=torch.int32))

        self.assertIsNone(returned)
        self.assertEqual(pool.req_to_token[4, 1].item(), 7)
        self.assertEqual(pool.req_to_token[4, 2].item(), 8)

    def test_alloc_write_read_back_round_trip(self):
        """alloc → write → 读回：验证「行号 → 槽位」映射的端到端闭环。"""
        pool = _make_pool(size=4, max_context_len=8)
        req = _make_req()
        (row,) = pool.alloc([req])

        slots = torch.tensor([101, 102, 103], dtype=torch.int32)
        pool.write((row, slice(0, 3)), slots)

        self.assertEqual(pool.req_to_token[row, :3].tolist(), [101, 102, 103])


class TestReqToTokenPoolClearAndAuxCache(CustomTestCase):
    """clear 的重置语义，以及 aux cache 的转发契约。"""

    def test_clear_resets_free_slots_and_generations(self):
        """clear 重置空闲表并把所有 generation 归零（等价于回到初始状态）。"""
        pool = _make_pool(size=4)
        pool.alloc([_make_req()])
        self.assertEqual(pool.req_generation[4].item(), 1)

        pool.clear()

        self.assertEqual(pool.free_slots, [1, 2, 3, 4])
        self.assertTrue(torch.all(pool.req_generation == 0))
        self.assertEqual(pool.available_size(), 4)

    def test_aux_hooks_are_noops_without_aux_cache(self):
        """未挂 aux cache 时，容量查询原样返回物理容量，各 aux hook 均为空操作。"""
        pool = _make_pool(size=4)

        self.assertEqual(pool.schedulable_token_capacity(17), 17)
        pool.reset_aux_cache_allocator()
        pool.alloc_aux_to_lengths(
            req_pool_indices_cpu=torch.tensor([1]),
            target_seq_lens_cpu=torch.tensor([3]),
        )
        pool.clear()

    def test_aux_cache_hooks_are_wired(self):
        """挂上 aux cache 后：容量查询改用 dense_capacity，free/clear/reset/alloc_to_lengths 全部转发。"""
        pool = _make_pool(size=4)
        aux = _RecordingAuxCache()
        pool.attach_aux_cache(aux)

        self.assertEqual(pool.schedulable_token_capacity(1000), 42)
        pool.reset_aux_cache_allocator()
        pool.alloc_aux_to_lengths(
            req_pool_indices_cpu=torch.tensor([1, 2]),
            target_seq_lens_cpu=torch.tensor([3, 4]),
        )
        pool.free_rows([1, 2])
        pool.clear()

        self.assertEqual(aux.reset_calls, 1)
        self.assertEqual(aux.alloc_calls, [([1, 2], [3, 4])])
        self.assertEqual(aux.freed, [1, 2])
        self.assertEqual(aux.clear_calls, 1)
        self.assertEqual(pool.free_slots, [1, 2, 3, 4])

    def test_attach_aux_cache_rejects_second_attach(self):
        """aux cache 只能挂载一次，重复挂载触发断言。"""
        pool = _make_pool(size=4)
        pool.attach_aux_cache(_RecordingAuxCache())

        with self.assertRaises(AssertionError):
            pool.attach_aux_cache(_RecordingAuxCache())


if __name__ == "__main__":
    unittest.main()
