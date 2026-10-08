"""MHATokenToKVPool 单测（两级映射背后的物理 KV 存储）。

职责回顾：``TokenToKVPoolAllocator`` 只维护「哪些槽位空闲」的账本，真正的显存
在 ``KVCache``（本文件测的 ``MHATokenToKVPool``）里。本池为每层维护
``k_buffer[layer]`` / ``v_buffer[layer]``，默认 NHD 布局形状
``[size + page_size, head_num, head_dim]``。额外容量用于 padding 页；分页分配器
保留的是第 0 页，而不是最后一页，池自身不负责分配槽位。

覆盖点：

- 构造：每层 K/V 缓冲的形状与 dtype、``row_dim`` / ``v_row_dim``、SWA 参数覆盖、
  fp8 按 uint8 存储、NHD 默认布局、显存统计、``data_ptrs`` / ``data_strides``、
  ``_kv_buffer_descs``、``start_layer`` 的本地索引偏移；
- 取缓冲：``get_key_buffer`` / ``get_value_buffer`` / ``get_kv_buffer`` /
  ``get_v_head_dim`` / ``get_kv_buffer_shape``、逐层传输计数器同步；
- 写缓冲：``set_kv_buffer`` 的槽位回填、层选择、``layer_id_override``、dtype 转换
  与 ``k_scale`` / ``v_scale``、``KVWriteLoc`` 解包、最高合法槽位、fp8 字节重解释；
- 前缀有效写入：``set_kv_buffer_prefix_valid`` 的形状校验、只写已提交前缀、
  空提交 no-op，以及非 CUDA 平台上非对称 K/V 走掩码回退；
- 搬运：``move_kv_cache`` 的原生路径、空索引 no-op、零层池短路，以及未开启
  ``enable_kv_cache_copy`` 时缺少拷贝配置会断言失败、warmup 配置和分块调用；
- CPU offload：``get_cpu_copy`` / ``load_cpu_copy`` 往返、按 ``cpu_offloading_chunk_size``
  分块、行数不匹配断言、HND 池拒绝 offload；
- HND 布局：``SGLANG_USE_HND_KVCACHE=1`` 时的 4-D 缓冲、按 (page, off) 散写与搬运、
  描述符 tokens_per_row、拒绝 disagg / offload、``(size+page_size) % page_size`` 断言；
- 子类 ``NoOpMHATokenToKVPool``：占位缓冲形状、``size`` 保持逻辑容量、零显存统计、
  ``set_kv_buffer`` 抛错、读取占位与 move no-op、忽略 HND 环境变量；
- 外围：量化策略委托及 workspace 校验、post-capture 生命周期、缓冲清理。

本文件在 CPU 上执行真实张量读写，隔离平台探测、runner、同步与硬件内核边界。
GPU 融合内核的数值正确性仍由 ``test_asymmetric_mha_pool.py`` 等设备测试验证；
量化算法、CUDA VMM 和 AITER 内核不在本文件中模拟其数值实现。
已知缺陷：CPU prefix-valid 的 FP8 数据被重复转换，对应回归用例标记 expectedFailure；
修复生产实现后该用例会 unexpected success，届时应移除此标记。

运行方式（在仓库根目录执行即可，支持只跑单个 case）::

    # 全部用例
    python3 -m pytest test/registered/unit/mem_cache/test_mha_token_to_kv_pool_unit.py -v

    # 单个用例（pytest nodeid）
    python3 -m pytest "test/registered/unit/mem_cache/test_mha_token_to_kv_pool_unit.py::TestMHAConstruction::test_default_layout_is_nhd" -v

    # 单个用例（不依赖 pytest，直接跑文件 + unittest 选择器）
    python3 test/registered/unit/mem_cache/test_mha_token_to_kv_pool_unit.py TestMHAConstruction.test_default_layout_is_nhd

    # 单个测试类
    python3 test/registered/unit/mem_cache/test_mha_token_to_kv_pool_unit.py TestMHAConstruction

    # 按关键字筛选
    python3 -m pytest test/registered/unit/mem_cache/test_mha_token_to_kv_pool_unit.py -k "hnd or prefix"

注意用 ``python3``：部分开发容器里 ``python`` 指向系统自带的 3.9，没有装 pytest；
``python3`` 才是有 pytest 的解释器（可用 ``python3 -c "import pytest"`` 自查）。

模块导入时会先执行 ``_bootstrap_repo_imports()``：把本仓库的 ``python/`` 前置到
``sys.path``，避免开发镜像里 editable 安装的另一份（可能过旧的）sglang 抢先命中；
同时把环境变量 ``SGLANG_TEST_EXTRA_PYTHONPATH``（冒号分隔）与
``<home>/tb4.0/*/DeepGEMM/build`` 中真实存在的目录一并前置，用于补齐部分镜像缺失的
deep_gemm。上述目录在 CI 上不存在，因此该引导在 CI 中是 no-op。
"""

import glob
import os
import sys
import types
import unittest
import unittest.mock
from contextlib import ExitStack


def _bootstrap_repo_imports() -> None:
    """开发容器引导：确保直接执行本文件时 import 到的是本仓库代码。

    镜像里常预先 editable 安装过另一份 sglang，它挂在 ``sys.meta_path`` 上；
    而 ``PathFinder`` 位于其之前，因此把本仓库的 ``python/`` 前置到 ``sys.path``
    就能让仓库代码优先命中（对 pytest 与脚本直跑都生效）。

    额外目录来源（按需、均需真实存在）：

    - ``SGLANG_TEST_EXTRA_PYTHONPATH``：冒号分隔的自定义路径；
    - ``<home>/tb4.0/*/DeepGEMM/build``：本机开发容器的 deep_gemm 构建产物，
      其中 ``<home>`` 同时尝试 ``~`` 与仓库根目录的上一级目录（工作区常位于
      ``/home/<user>/sglang``，而容器 ``HOME`` 可能指向别处）。

    CI 上这些目录不存在，整个引导退化为 no-op，行为与改造前一致。
    """
    root = os.path.dirname(os.path.abspath(__file__))
    while root != os.path.dirname(root) and not os.path.isdir(
        os.path.join(root, "python", "sglang")
    ):
        root = os.path.dirname(root)

    candidates = [os.path.join(root, "python")]
    candidates += [
        path
        for path in os.environ.get("SGLANG_TEST_EXTRA_PYTHONPATH", "").split(os.pathsep)
        if path
    ]
    for home in dict.fromkeys([os.path.expanduser("~"), os.path.dirname(root)]):
        candidates += glob.glob(os.path.join(home, "tb4.0", "*", "DeepGEMM", "build"))

    for path in reversed(candidates):  # 靠前的候选优先级更高
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_repo_imports()

import torch

from sglang.srt.mem_cache import memory_pool as _mp
from sglang.srt.mem_cache.memory_pool import (
    KVWriteLoc,
    MHATokenToKVPool,
    NoOpMHATokenToKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

DEVICE = "cpu"
DTYPE = torch.float16
HEAD_NUM = 2
HEAD_DIM = 8
V_HEAD_DIM = 4
LAYER_NUM = 2
PAGE_SIZE = 2
SIZE = 8
NUM_SLOTS = SIZE + PAGE_SIZE  # 可分配容量 + 第 0 页 padding 所需容量


def _build_pool(**overrides) -> MHATokenToKVPool:
    """构造一个小尺寸的 CPU 池；``overrides`` 可覆盖任意构造参数。"""
    kwargs = dict(
        size=SIZE,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        v_head_dim=V_HEAD_DIM,  # 非对称，CPU 上走朴素写路径
        layer_num=LAYER_NUM,
        device=DEVICE,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    kwargs.update(overrides)
    return MHATokenToKVPool(**kwargs)


def _layer(layer_id: int = 0):
    """注意力 backend 传给 set_kv_buffer 的 layer 对象只用到 layer_id。"""
    return types.SimpleNamespace(layer_id=layer_id)


def _rand_k(n: int, dtype=DTYPE, head_dim=HEAD_DIM):
    return torch.randn(
        n, HEAD_NUM, head_dim, dtype=dtype, generator=torch.Generator().manual_seed(7)
    )


def _rand_v(n: int, dtype=DTYPE):
    return torch.randn(
        n,
        HEAD_NUM,
        V_HEAD_DIM,
        dtype=dtype,
        generator=torch.Generator().manual_seed(13),
    )


class _MHACPUTestCase(CustomTestCase):
    """只隔离外部依赖，保留 pool 本身及张量操作；每个用例结束自动恢复。"""

    def setUp(self):
        """安装每个用例都要的隔离：平台开关、环境变量、依赖替身，结束后自动恢复。"""
        super().setUp()
        # 使用 ExitStack 兼容 Python 3.10；setUp 中途失败也会执行恢复。
        stack = ExitStack()
        self.addCleanup(stack.close)
        # 平台标志来自宿主而不是张量设备，显式选择 CPU 朴素路径（不依赖 AMX）。
        stack.enter_context(
            unittest.mock.patch.multiple(
                _mp,
                _is_cuda=False,
                _is_hip=False,
                _is_cpu=True,
                _cpu_has_amx_support=False,
                _use_aiter=False,
            )
        )
        stack.enter_context(
            unittest.mock.patch.dict(
                os.environ,
                {"SGLANG_USE_HND_KVCACHE": "0", "SGLANG_NATIVE_MOVE_KV_CACHE": "0"},
            )
        )
        stack.enter_context(
            unittest.mock.patch.object(
                _mp, "maybe_init_custom_mem_pool", return_value=(False, None, None)
            )
        )
        # CPU 张量无需设备同步，但保留 mock 以验证 offload 的同步契约。
        self.synchronize = stack.enter_context(
            unittest.mock.patch.object(_mp.current_platform, "synchronize")
        )
        # 朴素写路径只依赖 capture 状态，不应为了单测导入整个模型 runner。
        # 仅恢复这一项，避免 patch.dict(sys.modules) 回滚其他模块的正常导入。
        name = "sglang.srt.model_executor.runner"
        if name in sys.modules:
            stack.enter_context(
                unittest.mock.patch.object(
                    sys.modules[name], "get_is_capture_mode", return_value=False
                )
            )
        else:
            runner = types.ModuleType(name)
            runner.get_is_capture_mode = unittest.mock.Mock(return_value=False)
            sys.modules[name] = runner
            stack.callback(sys.modules.pop, name, None)


class _FakeTransferCounter:
    """记录 get_key/value_buffer 请求等待了哪些本地层号。"""

    def __init__(self):
        self.waits = []

    def wait_until(self, local_layer_id: int):
        self.waits.append(local_layer_id)


class TestMHAConstruction(_MHACPUTestCase):
    """构造：缓冲形状 / dtype、几何推导、布局、显存统计与元数据表。"""

    def test_buffers_allocated_per_layer_with_expected_shapes(self):
        """每层各有 K/V 两个缓冲，形状为 [size+page_size, head_num, head_dim]。"""
        pool = _build_pool()

        self.assertEqual(len(pool.k_buffer), LAYER_NUM)
        self.assertEqual(len(pool.v_buffer), LAYER_NUM)
        for buf in pool.k_buffer:
            self.assertEqual(tuple(buf.shape), (NUM_SLOTS, HEAD_NUM, HEAD_DIM))
            self.assertEqual(buf.dtype, DTYPE)
            self.assertEqual(buf.device.type, DEVICE)
        for buf in pool.v_buffer:
            self.assertEqual(tuple(buf.shape), (NUM_SLOTS, HEAD_NUM, V_HEAD_DIM))
            self.assertEqual(buf.dtype, DTYPE)

    def test_row_dims_follow_head_dims(self):
        """``row_dim`` 是写入内核每行的元素数，非对称时 V 行更窄。"""
        pool = _build_pool()

        self.assertEqual(pool.row_dim, HEAD_NUM * HEAD_DIM)
        self.assertEqual(pool.v_row_dim, HEAD_NUM * V_HEAD_DIM)
        self.assertNotEqual(pool.row_dim, pool.v_row_dim)

    def test_v_head_dim_defaults_to_head_dim(self):
        """未给 ``v_head_dim`` 时 V 退化为与 K 同宽。"""
        pool = _build_pool(v_head_dim=None)

        self.assertEqual(pool.v_head_dim, HEAD_DIM)
        self.assertEqual(pool.v_row_dim, pool.row_dim)

    def test_swa_params_override_head_geometry(self):
        """SWA 子池通过 swa_* 参数整体覆盖 K/V 几何（含缓冲形状）。"""
        pool = _build_pool(swa_head_num=1, swa_head_dim=6, swa_v_head_dim=3)

        self.assertEqual(pool.head_num, 1)
        self.assertEqual(pool.head_dim, 6)
        self.assertEqual(pool.v_head_dim, 3)
        self.assertEqual(pool.row_dim, 6)
        self.assertEqual(pool.v_row_dim, 3)
        self.assertEqual(tuple(pool.k_buffer[0].shape), (NUM_SLOTS, 1, 6))
        self.assertEqual(tuple(pool.v_buffer[0].shape), (NUM_SLOTS, 1, 3))

    def test_swa_head_dim_without_swa_v_head_dim_falls_back(self):
        """只给 swa_head_dim 时，v_head_dim 仍取自顶层 ``v_head_dim``。"""
        pool = _build_pool(swa_head_num=1, swa_head_dim=6)

        self.assertEqual(pool.head_dim, 6)
        self.assertEqual(pool.v_head_dim, V_HEAD_DIM)
        self.assertEqual(pool.v_row_dim, 1 * V_HEAD_DIM)

    def test_fp8_dtype_stored_as_uint8(self):
        """fp8 一律以 uint8 存储（Tensor.index_put 未支持 fp8），但计算 dtype 保留。"""
        pool = _build_pool(dtype=torch.float8_e4m3fn)

        self.assertEqual(pool.dtype, torch.float8_e4m3fn)
        self.assertEqual(pool.store_dtype, torch.uint8)
        self.assertEqual(pool.k_buffer[0].dtype, torch.uint8)
        self.assertEqual(pool.v_buffer[0].dtype, torch.uint8)

    def test_non_fp8_store_dtype_matches_compute_dtype(self):
        """非 fp8 时存储 dtype 与计算 dtype 一致（不做字节重解释）。"""
        pool = _build_pool()

        self.assertEqual(pool.store_dtype, DTYPE)

    def test_default_layout_is_nhd(self):
        """未显式指定布局且未开 HND 环境变量时，默认使用 NHD。"""
        pool = _build_pool()

        self.assertEqual(pool.kv_cache_layout, "nhd")
        self.assertFalse(pool.use_hnd)

    def test_kv_size_bytes_and_mem_usage(self):
        """``get_kv_size_bytes`` 统计 K/V 各自字节数，``mem_usage`` 为 GB 之和。"""
        pool = _build_pool()

        k_bytes, v_bytes = pool.get_kv_size_bytes()
        expected_k = NUM_SLOTS * HEAD_NUM * HEAD_DIM * DTYPE.itemsize * LAYER_NUM
        expected_v = NUM_SLOTS * HEAD_NUM * V_HEAD_DIM * DTYPE.itemsize * LAYER_NUM
        self.assertEqual(k_bytes, expected_k)
        self.assertEqual(v_bytes, expected_v)
        print(pool.mem_usage)
        print(k_bytes)
        print(v_bytes)
        self.assertAlmostEqual(
            pool.mem_usage, (expected_k + expected_v) / (1024**3), places=10
        )

    def test_data_ptrs_and_strides_cover_k_then_v(self):
        """指针/跨度表按 k0..k(L-1), v0..v(L-1) 排列，跨度等于每 token 行字节。"""
        pool = _build_pool()

        self.assertEqual(pool.k_data_ptrs.numel(), LAYER_NUM)
        self.assertEqual(pool.v_data_ptrs.numel(), LAYER_NUM)
        self.assertEqual(pool.data_ptrs.numel(), 2 * LAYER_NUM)
        self.assertEqual(pool.k_data_ptrs.dtype, torch.uint64)
        self.assertEqual(pool.v_data_ptrs.dtype, torch.uint64)
        self.assertEqual(pool.data_ptrs.dtype, torch.uint64)
        self.assertEqual(
            pool.k_data_ptrs.tolist(), [b.data_ptr() for b in pool.k_buffer]
        )
        self.assertEqual(
            pool.v_data_ptrs.tolist(), [b.data_ptr() for b in pool.v_buffer]
        )
        self.assertEqual(
            pool.data_ptrs.tolist(),
            [b.data_ptr() for b in pool.k_buffer + pool.v_buffer],
        )
        expected_strides = [HEAD_NUM * HEAD_DIM * DTYPE.itemsize] * LAYER_NUM + [
            HEAD_NUM * V_HEAD_DIM * DTYPE.itemsize
        ] * LAYER_NUM
        self.assertEqual(pool.data_strides.tolist(), expected_strides)

    def test_kv_buffer_descs_order_and_row_bytes(self):
        """描述符按 k0..k(L-1), v0..v(L-1) 生成，NHD 下每行一个 token。"""
        pool = _build_pool()

        descs = pool._kv_buffer_descs
        self.assertEqual([d.name for d in descs], ["k0", "k1", "v0", "v1"])
        self.assertEqual(descs[0].shape, (NUM_SLOTS, HEAD_NUM, HEAD_DIM))
        self.assertEqual(descs[2].shape, (NUM_SLOTS, HEAD_NUM, V_HEAD_DIM))
        self.assertEqual(descs[0].row_bytes, HEAD_NUM * HEAD_DIM * DTYPE.itemsize)
        self.assertEqual(descs[2].row_bytes, HEAD_NUM * V_HEAD_DIM * DTYPE.itemsize)
        self.assertEqual(descs[0].tokens_per_row, 1)
        self.assertEqual(descs[2].tokens_per_row, 1)

    def test_get_contiguous_buf_infos_spans_full_buffers(self):
        """PD 传输注册用 (ptrs, lens, item_lens)：lens 覆盖整个缓冲。"""
        pool = _build_pool()

        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        print(f"ptrs[0] = {ptrs[0]} (0x{ptrs[0]:x})")
        self.assertEqual(len(ptrs), 2 * LAYER_NUM)
        self.assertEqual(ptrs[0], pool.k_buffer[0].data_ptr())
        self.assertEqual(ptrs[2], pool.v_buffer[0].data_ptr())
        self.assertEqual(
            lens[0], pool.k_buffer[0].numel() * pool.k_buffer[0].element_size()
        )
        self.assertEqual(
            lens[2], pool.v_buffer[0].numel() * pool.v_buffer[0].element_size()
        )
        # item_len 是「一页」的字节数（NHD 下 = page_size * 行字节）。
        self.assertEqual(item_lens[0], PAGE_SIZE * pool._kv_buffer_descs[0].row_bytes)

    def test_start_layer_offsets_local_indexing(self):
        """start_layer 非 0 时，全局层号要减去 start_layer 才能索引本地缓冲。"""
        pool = _build_pool(start_layer=5, end_layer=6, layer_num=2)

        self.assertEqual(pool.start_layer, 5)
        self.assertIs(pool._get_key_buffer(5), pool.k_buffer[0])
        self.assertIs(pool._get_key_buffer(6), pool.k_buffer[1])
        self.assertIs(pool._get_value_buffer(6), pool.v_buffer[1])

    def test_default_quant_method_is_wired(self):
        """未显式传 quant_method 时使用 unquantized 实现，池报告为非量化。"""
        pool = _build_pool()

        self.assertFalse(pool.is_quantized_kv_cache)
        self.assertIsNotNone(pool.quant_method)
        self.assertIs(pool.get_kv_cache_quant_method(), pool.quant_method)


class TestMHAGetBuffers(_MHACPUTestCase):
    """取缓冲：本地层索引、v_head_dim、缓冲形状与逐层同步。"""

    def test_get_key_value_and_kv_buffer_use_local_layer_index(self):
        """三个读取接口都按「本地层号」直接返回对应列表元素（不做拷贝）。"""
        pool = _build_pool()

        for i in range(LAYER_NUM):
            self.assertIs(pool.get_key_buffer(i), pool.k_buffer[i])
            self.assertIs(pool.get_value_buffer(i), pool.v_buffer[i])
            k, v = pool.get_kv_buffer(i)
            self.assertIs(k, pool.k_buffer[i])
            self.assertIs(v, pool.v_buffer[i])

    def test_get_v_head_dim(self):
        """get_v_head_dim 返回构造期确定的 V 头维度（供 mambaish 分支使用）。"""
        self.assertEqual(_build_pool().get_v_head_dim(), V_HEAD_DIM)

    def test_get_kv_buffer_shape(self):
        """get_kv_buffer_shape 返回首层 (K, V) 形状，K/V 最后一维可不同。"""
        pool = _build_pool()

        k_shape, v_shape = pool.get_kv_buffer_shape()
        self.assertEqual(k_shape, torch.Size([NUM_SLOTS, HEAD_NUM, HEAD_DIM]))
        self.assertEqual(v_shape, torch.Size([NUM_SLOTS, HEAD_NUM, V_HEAD_DIM]))

    def test_layer_transfer_counter_is_awaited(self):
        """注册了逐层传输计数器后，取该层缓冲前必须等待其传输完成。"""
        pool = _build_pool(start_layer=5, end_layer=6)
        counter = _FakeTransferCounter()
        pool.register_layer_transfer_counter(counter)

        pool.get_key_buffer(5)
        pool.get_value_buffer(6)
        pool.get_kv_buffer(6)
        self.assertEqual(counter.waits, [0, 1, 1, 1])
        pool._get_key_buffer(5)
        pool._get_value_buffer(6)
        self.assertEqual(counter.waits, [0, 1, 1, 1])  # 内部读不触发等待


class TestMHASetKVBuffer(_MHACPUTestCase):
    """写缓冲：槽位回填、层选择、dtype/scale、KVWriteLoc、padding、fp8。"""

    def test_writes_only_target_slots_and_reads_back(self):
        """只写 loc 指定的槽位，其余槽位保持初始零值（不污染邻近行）。"""
        pool = _build_pool()
        loc = torch.tensor([1, 4])
        cache_k = _rand_k(2)
        cache_v = _rand_v(2)

        pool.set_kv_buffer(_layer(0), loc, cache_k, cache_v)

        self.assertTrue(torch.equal(pool.k_buffer[0][loc], cache_k))
        self.assertTrue(torch.equal(pool.v_buffer[0][loc], cache_v))
        untouched = torch.ones(NUM_SLOTS, dtype=torch.bool)
        untouched[loc] = False
        self.assertEqual(pool.k_buffer[0][untouched].abs().sum().item(), 0.0)
        self.assertEqual(pool.v_buffer[0][untouched].abs().sum().item(), 0.0)

    def test_layer_id_selects_the_right_buffer(self):
        """layer_id 决定写入哪一层的缓冲，其他层不受影响。"""
        pool = _build_pool()
        loc = torch.tensor([0])
        cache_k = torch.ones(1, HEAD_NUM, HEAD_DIM, dtype=DTYPE)
        cache_v = torch.ones(1, HEAD_NUM, V_HEAD_DIM, dtype=DTYPE)

        pool.set_kv_buffer(_layer(1), loc, cache_k, cache_v)

        self.assertEqual(pool.k_buffer[1][0].sum().item(), HEAD_NUM * HEAD_DIM)
        self.assertEqual(pool.k_buffer[0].abs().sum().item(), 0.0)

    def test_layer_id_override_wins(self):
        """覆盖非零 start_layer、override=0 和 layer=None 的合法调用。"""
        for start, layer, override in (
            (5, _layer(5), 6),
            (0, _layer(1), 0),
            (5, None, 6),
        ):
            with self.subTest(start=start, override=override):
                pool = _build_pool(start_layer=start, end_layer=start + 1)
                loc = torch.tensor([2])
                cache_k, cache_v = _rand_k(1), _rand_v(1)
                pool.set_kv_buffer(
                    layer, loc, cache_k, cache_v, layer_id_override=override
                )
                for i, (kb, vb) in enumerate(zip(pool.k_buffer, pool.v_buffer)):
                    expected_k, expected_v = torch.zeros_like(kb), torch.zeros_like(vb)
                    if i == override - start:
                        expected_k[loc], expected_v[loc] = cache_k, cache_v
                    torch.testing.assert_close(kb, expected_k)
                    torch.testing.assert_close(vb, expected_v)

    def test_dtype_cast_and_scales(self):
        """输入 dtype 不同时先原地除以各自 scale，再转换成池 dtype。"""
        pool = _build_pool()
        loc = torch.tensor([1])
        cache_k = torch.full((1, HEAD_NUM, HEAD_DIM), 4.0, dtype=torch.float32)
        cache_v = torch.full((1, HEAD_NUM, V_HEAD_DIM), 9.0, dtype=torch.float32)

        pool.set_kv_buffer(_layer(0), loc, cache_k, cache_v, k_scale=2.0, v_scale=3.0)

        self.assertTrue(
            torch.equal(
                pool.k_buffer[0][loc],
                torch.full((1, HEAD_NUM, HEAD_DIM), 2.0, dtype=DTYPE),
            )
        )
        self.assertTrue(
            torch.equal(
                pool.v_buffer[0][loc],
                torch.full((1, HEAD_NUM, V_HEAD_DIM), 3.0, dtype=DTYPE),
            )
        )

    def test_cast_without_scales_and_same_dtype_ignores_scales(self):
        """dtype 不同且无 scale 时只做类型转换；dtype 相同时即使传了 scale 也不缩放。"""
        for dtype, scale in ((torch.float32, None), (DTYPE, 3.0)):
            with self.subTest(dtype=dtype):
                pool = _build_pool(v_head_dim=HEAD_DIM)
                loc = torch.tensor([2, 5])
                k, v = _rand_k(2, dtype=dtype), _rand_k(2, dtype=dtype) + 1
                k_before, v_before = k.clone(), v.clone()
                pool.set_kv_buffer(_layer(), loc, k, v, k_scale=scale, v_scale=scale)
                torch.testing.assert_close(pool.k_buffer[0][loc], k_before.to(DTYPE))
                torch.testing.assert_close(pool.v_buffer[0][loc], v_before.to(DTYPE))
                torch.testing.assert_close(k, k_before)
                torch.testing.assert_close(v, v_before)

    def test_empty_write_and_noncontiguous_input(self):
        """空写入是 no-op；非连续输入也能正确落盘（回退路径不要求输入连续）。"""
        pool = _build_pool(start_layer=5, end_layer=6)
        for n in (0, 2):
            with self.subTest(rows=n):
                # transpose 后每个 head 的数据不连续，CPU 回退不应依赖连续输入。
                k = _rand_k(n).transpose(1, 2).contiguous().transpose(1, 2)
                v = _rand_v(n).transpose(1, 2).contiguous().transpose(1, 2)
                loc = torch.tensor([2, 5], dtype=torch.int64)[:n]
                pool.set_kv_buffer(_layer(6), loc, k, v)
                torch.testing.assert_close(pool.get_key_buffer(6)[loc], k)
                torch.testing.assert_close(pool.get_value_buffer(6)[loc], v)
                self.assertEqual(pool.k_buffer[0].count_nonzero().item(), 0)
                self.assertEqual(pool.v_buffer[0].count_nonzero().item(), 0)

    def test_accepts_kv_write_loc_dataclass(self):
        """loc 参数既可以是裸张量，也可以是 KVWriteLoc（内部统一 unwrap）。"""
        pool = _build_pool()
        loc = torch.tensor([3])
        cache_k = torch.ones(1, HEAD_NUM, HEAD_DIM, dtype=DTYPE)
        cache_v = torch.ones(1, HEAD_NUM, V_HEAD_DIM, dtype=DTYPE)

        pool.set_kv_buffer(_layer(0), KVWriteLoc(loc=loc), cache_k, cache_v)

        self.assertEqual(pool.k_buffer[0][3].sum().item(), HEAD_NUM * HEAD_DIM)

    def test_last_valid_slot_is_writable(self):
        """最高合法槽位为 size + page_size - 1，不应误当成越界或 padding。"""
        pool = _build_pool()
        loc = torch.tensor([NUM_SLOTS - 1])
        cache_k = _rand_k(1)
        cache_v = _rand_v(1)

        pool.set_kv_buffer(_layer(0), loc, cache_k, cache_v)

        self.assertTrue(torch.equal(pool.k_buffer[0][NUM_SLOTS - 1], cache_k[0]))
        self.assertTrue(torch.equal(pool.v_buffer[0][NUM_SLOTS - 1], cache_v[0]))

    def test_fp8_store_reinterprets_bytes(self):
        """按 uint8 比较存储位模式，并验证公开 getter 返回共享存储的 fp8 view。"""
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz):
            with self.subTest(dtype=dtype):
                pool = _build_pool(dtype=dtype)
                loc = torch.tensor([2, 4])
                cache_k = _rand_k(2).to(dtype)
                cache_v = _rand_v(2).to(dtype)
                pool.set_kv_buffer(_layer(0), loc, cache_k, cache_v)

                for raw, actual, expected in zip(
                    (pool.k_buffer[0], pool.v_buffer[0]),
                    pool.get_kv_buffer(0),
                    (cache_k, cache_v),
                ):
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(actual.shape, raw.shape)
                    self.assertEqual(actual.data_ptr(), raw.data_ptr())
                    torch.testing.assert_close(raw[loc], expected.view(torch.uint8))
                    torch.testing.assert_close(
                        actual.view(torch.uint8)[loc], expected.view(torch.uint8)
                    )


class TestMHAPrefixValid(_MHACPUTestCase):
    """前缀有效写入：形状校验 + 仅写已提交前缀（非 CUDA 走掩码回退）。"""

    def _kv_rows(self, num_rows: int):
        """构造每行数值 = 行号（+1）的 K/V，便于按行断言落点。"""
        cache_k = torch.zeros(num_rows, HEAD_NUM, HEAD_DIM, dtype=DTYPE)
        cache_v = torch.zeros(num_rows, HEAD_NUM, V_HEAD_DIM, dtype=DTYPE)
        for r in range(num_rows):
            cache_k[r].fill_(float(r + 1))
            cache_v[r].fill_(float(r + 1) * 10.0)
        return cache_k, cache_v

    def test_requires_rank_2_loc(self):
        """loc_2d 必须二维，否则报错（拦住把一维槽位表误传进来）。"""
        pool = _build_pool()
        cache_k, cache_v = self._kv_rows(2)

        with self.assertRaisesRegex(ValueError, "rank-2"):
            pool.set_kv_buffer_prefix_valid(
                _layer(0),
                torch.tensor([1, 2]),  # 1 维
                torch.tensor([1]),
                cache_k,
                cache_v,
            )

    def test_commit_lens_must_match_batch(self):
        """commit_lens 长度必须等于 loc_2d 的批大小。"""
        pool = _build_pool()
        cache_k, cache_v = self._kv_rows(4)

        with self.assertRaisesRegex(ValueError, "commit_lens must match"):
            pool.set_kv_buffer_prefix_valid(
                _layer(0),
                torch.tensor([[1, 2], [3, 4]]),
                torch.tensor([1]),  # batch 应为 2
                cache_k,
                cache_v,
            )

    def test_dense_rows_must_match_loc_size(self):
        """K/V 行数必须等于 loc_2d 的元素总数（每个槽位对应一行）。"""
        pool = _build_pool()
        cache_k, cache_v = self._kv_rows(3)  # 应为 4 行

        with self.assertRaisesRegex(ValueError, "dense KV rows must match"):
            pool.set_kv_buffer_prefix_valid(
                _layer(0),
                torch.tensor([[1, 2], [3, 4]]),
                torch.tensor([1, 2]),
                cache_k,
                cache_v,
            )

    def test_writes_only_committed_prefix(self):
        """每行只写前 commit_lens[i] 个槽位，未提交的槽位保持为空。

        本池默认非对称（head_dim != v_head_dim），CPU 上没有 tiled 内核，因此
        走的是「掩码筛出有效槽位后退化为 set_kv_buffer」的回退分支。
        """
        pool = _build_pool(start_layer=5, end_layer=6)
        loc_2d = torch.tensor([[2, -999], [4, 5]], dtype=torch.int32)
        commit_lens = torch.tensor([1, 2], dtype=torch.int64)
        cache_k, cache_v = self._kv_rows(4)
        for kb, vb in zip(pool.k_buffer, pool.v_buffer):
            kb.fill_(-7)
            vb.fill_(-9)
        expected_k = [b.clone() for b in pool.k_buffer]
        expected_v = [b.clone() for b in pool.v_buffer]
        expected_k[1][[2, 4, 5]] = cache_k[[0, 2, 3]]
        expected_v[1][[2, 4, 5]] = cache_v[[0, 2, 3]]

        pool.set_kv_buffer_prefix_valid(
            _layer(5), loc_2d, commit_lens, cache_k, cache_v, layer_id_override=6
        )

        # 无效位置即使越界也不应访问，未写槽位及其他层必须保留原值。
        for actual, expected in zip(
            pool.k_buffer + pool.v_buffer, expected_k + expected_v
        ):
            torch.testing.assert_close(actual, expected)

    def test_zero_commit_and_empty_batch_are_noop(self):
        """commit_lens 全为 0 或批大小为 0 时不访问任何槽位，已有数据不变。"""
        for batch in (0, 1):
            with self.subTest(batch=batch):
                pool = _build_pool()
                for b in pool.k_buffer + pool.v_buffer:
                    b.fill_(-5)
                before = [b.clone() for b in pool.k_buffer + pool.v_buffer]
                k, v = self._kv_rows(batch * 2)
                pool.set_kv_buffer_prefix_valid(
                    _layer(),
                    torch.full((batch, 2), -999),
                    torch.zeros(batch, dtype=torch.int32),
                    k,
                    v,
                )
                for actual, expected in zip(pool.k_buffer + pool.v_buffer, before):
                    torch.testing.assert_close(actual, expected)

    @unittest.expectedFailure  # 已复现：uint8 位模式在 set_kv_buffer 中被再次数值转换。
    def test_prefix_valid_fp8_preserves_storage_bytes(self):
        """回归：CPU 回退不得把 FP8 的 uint8 位模式当作数值再次量化。"""
        pool = _build_pool(dtype=torch.float8_e4m3fn)
        k = torch.ones(2, HEAD_NUM, HEAD_DIM).to(pool.dtype)
        v = torch.full((2, HEAD_NUM, V_HEAD_DIM), 2.0).to(pool.dtype)
        pool.set_kv_buffer_prefix_valid(
            _layer(),
            torch.tensor([[2, 3]]),
            torch.tensor([1]),
            k,
            v,
        )
        torch.testing.assert_close(pool.k_buffer[0][2], k[0].view(torch.uint8))
        torch.testing.assert_close(pool.v_buffer[0][2], v[0].view(torch.uint8))
        self.assertEqual(pool.k_buffer[0][3].count_nonzero().item(), 0)
        self.assertEqual(pool.v_buffer[0][3].count_nonzero().item(), 0)


class TestMHAMoveKV(_MHACPUTestCase):
    """搬运：原生路径、空索引、零层池短路、缺少拷贝配置的契约。"""

    def test_native_move_copies_src_to_tgt(self):
        """原生搬运按「先快照再写」处理源/目标重叠，且覆盖所有层。"""
        pool = _build_pool()
        pool.use_native_move_kv_cache = True  # 无需 Triton，源/目标重叠时仍按快照复制。
        src, tgt = torch.tensor([2, 3]), torch.tensor([3, 5])
        for layer in range(LAYER_NUM):
            pool.set_kv_buffer(
                _layer(layer), src, _rand_k(2) + layer, _rand_v(2) - layer
            )
        expected = [b.clone() for b in pool.k_buffer + pool.v_buffer]
        for b in expected:
            b[tgt] = b[src].clone()

        pool.move_kv_cache(tgt, src)

        for actual, wanted in zip(pool.k_buffer + pool.v_buffer, expected):
            torch.testing.assert_close(actual, wanted)

    def test_empty_loc_is_noop(self):
        """空索引在原生与拷贝两种路径下都不改写数据。"""
        for native in (False, True):
            with self.subTest(native=native):
                pool = _build_pool()
                pool.use_native_move_kv_cache = native
                for b in pool.k_buffer + pool.v_buffer:
                    b.fill_(7)
                before = [b.clone() for b in pool.k_buffer + pool.v_buffer]
                pool.move_kv_cache(
                    torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
                )
                for actual, expected in zip(pool.k_buffer + pool.v_buffer, before):
                    torch.testing.assert_close(actual, expected)

    def test_zero_layer_pool_short_circuits(self):
        """零层池（如 all-SWA 模型的 full 子池）没有缓冲，搬运直接返回。"""
        pool = _build_pool(layer_num=0)

        pool.move_kv_cache(torch.tensor([1]), torch.tensor([2]))  # 不应抛错

        self.assertEqual(pool.layer_num, 0)

    def test_missing_copy_config_asserts(self):
        """未开启 enable_kv_cache_copy 且未走原生搬运时，缺少拷贝配置会断言失败。"""
        pool = _build_pool()

        self.assertFalse(pool.use_native_move_kv_cache)
        self.assertIsNone(pool._kv_copy_config)
        with self.assertRaises(AssertionError):
            pool.move_kv_cache(torch.tensor([1]), torch.tensor([2]))


class TestMHACopyDispatch(_MHACPUTestCase):
    """只 mock GPU 拷贝函数，真实执行配置推导和分块调度，不冒充内核数值测试。"""

    def test_copy_config_thresholds_and_warmup(self):
        """按行字节数选择 tile/warps 档位，并校验构造期 warmup 的 dummy 调用参数。"""
        # 每个条目为 (行字节数, tile 字节, tile 数, warps, 最大槽位数)。
        cases = (
            (32, 128, 1, 4, 256),
            (4094, 128, 32, 4, 256),
            (4096, 256, 16, 4, 256),
            (8190, 256, 32, 4, 256),
            (8192, 512, 16, 8, 128),
        )
        for row_bytes, tile, tiles, warps, cap in cases:
            with self.subTest(row_bytes=row_bytes), unittest.mock.patch.object(
                _mp, "copy_all_layer_kv_cache_func"
            ) as copy:
                pool = _build_pool(
                    size=2,
                    page_size=1,
                    head_num=1,
                    head_dim=row_bytes // 2,
                    v_head_dim=None,
                    layer_num=1,
                    enable_kv_cache_copy=True,
                )
                self.assertEqual(
                    pool._kv_copy_config,
                    {
                        "bytes_per_tile": tile,
                        "byte_tiles": tiles,
                        "num_warps": warps,
                        "num_locs_upper": cap,
                    },
                )
                copy.assert_called_once()
                ptrs, strides, tgt, src, count, upper, cfg = copy.call_args.args
                self.assertIs(ptrs, pool.data_ptrs)
                self.assertIs(strides, pool.data_strides)
                self.assertIs(src, tgt)
                self.assertEqual(tgt.dtype, torch.int64)
                self.assertEqual(tgt.shape, (cap,))
                self.assertEqual(tgt.count_nonzero().item(), 0)
                self.assertEqual((count, upper), (1, cap))
                self.assertIs(cfg, pool._kv_copy_config)

    def test_copy_dispatch_slices_at_cap_and_rounds_upper(self):
        """超过单次上限时按 cap 分块转发，并把每块 upper 上取整到 2 的幂。"""
        with unittest.mock.patch.object(_mp, "copy_all_layer_kv_cache_func") as copy:
            pool = _build_pool(size=520, enable_kv_cache_copy=True)
            for count in (0, 1, 3, 256, 257, 515):
                with self.subTest(count=count):
                    copy.reset_mock()  # 不把构造时的 warmup 计入实际搬运。
                    src = torch.arange(count, dtype=torch.int64)
                    tgt = src + 1
                    pool.move_kv_cache(tgt, src)
                    cap = 256
                    self.assertEqual(copy.call_count, (count + cap - 1) // cap)
                    for index, call in enumerate(copy.call_args_list):
                        ptrs, strides, out, inp, n, upper, cfg = call.args
                        start, end = index * cap, min((index + 1) * cap, count)
                        self.assertIs(ptrs, pool.data_ptrs)
                        self.assertIs(strides, pool.data_strides)
                        torch.testing.assert_close(out, tgt[start:end])
                        torch.testing.assert_close(inp, src[start:end])
                        self.assertEqual(n, end - start)
                        self.assertEqual(upper, 1 << (n - 1).bit_length())
                        self.assertIs(cfg, pool._kv_copy_config)

    def test_zero_layer_and_hnd_skip_copy_warmup(self):
        """零层池与 HND 布局都不进入拷贝 warmup：配置为 None 且不调用拷贝内核。"""
        with unittest.mock.patch.object(_mp, "copy_all_layer_kv_cache_func") as copy:
            empty = _build_pool(layer_num=0, enable_kv_cache_copy=True)
            with unittest.mock.patch.dict(os.environ, {"SGLANG_USE_HND_KVCACHE": "1"}):
                hnd = _build_pool(enable_kv_cache_copy=True)
            self.assertIsNone(empty._kv_copy_config)
            self.assertIsNone(hnd._kv_copy_config)
            self.assertEqual(empty.get_kv_size_bytes(), (0, 0))
            self.assertEqual(empty.get_contiguous_buf_infos(), ([], [], []))
            copy.assert_not_called()


class TestMHACPUOffload(_MHACPUTestCase):
    """CPU offload：换出/换入往返、分块、行数校验、HND 拒绝。"""

    def test_get_and_load_cpu_copy_roundtrip(self):
        """换出→清空→换入应逐层完全恢复，且换出/换入各同步一次设备。"""
        pool = _build_pool()
        loc = torch.tensor([1, 3, 5])
        k0 = _rand_k(3)
        v0 = _rand_v(3)
        pool.set_kv_buffer(_layer(0), loc, k0, v0)
        pool.set_kv_buffer(_layer(1), loc, k0 * 2, v0 * 2)

        cpu = pool.get_cpu_copy(loc)

        self.assertEqual(len(cpu), LAYER_NUM)
        self.assertEqual(len(cpu[0]), 1)  # 未超 chunk_size，单块
        self.assertTrue(torch.equal(cpu[0][0][0], pool.k_buffer[0][loc].cpu()))
        self.assertTrue(torch.equal(cpu[0][0][1], pool.v_buffer[0][loc].cpu()))

        # 清空后回灌，应完全恢复且逐层正确。
        for layer in range(LAYER_NUM):
            pool.k_buffer[layer].zero_()
            pool.v_buffer[layer].zero_()
        pool.load_cpu_copy(cpu, loc)

        self.assertTrue(torch.equal(pool.k_buffer[0][loc], k0))
        self.assertTrue(torch.equal(pool.k_buffer[1][loc], k0 * 2))
        self.assertTrue(torch.equal(pool.v_buffer[0][loc], v0))
        self.assertTrue(torch.equal(pool.v_buffer[1][loc], v0 * 2))
        self.assertEqual(self.synchronize.call_count, 4)

    def test_cpu_offloading_chunk_size_splits_indices(self):
        """按 cpu_offloading_chunk_size 切块（2+2+1），且换入后与原值一致。"""
        pool = _build_pool()
        pool.cpu_offloading_chunk_size = 2
        loc = torch.arange(1, 6)  # 5 个槽位 → 2 + 2 + 1
        k0 = _rand_k(5)
        v0 = _rand_v(5)
        pool.set_kv_buffer(_layer(0), loc, k0, v0)

        cpu = pool.get_cpu_copy(loc)

        self.assertEqual(len(cpu[0]), 3)
        self.assertEqual([chunk[0].shape[0] for chunk in cpu[0]], [2, 2, 1])
        # 回灌路径同样按 chunk_size 切分，往返应一致。
        pool.k_buffer[0].zero_()
        pool.v_buffer[0].zero_()
        pool.load_cpu_copy(cpu, loc)
        self.assertTrue(torch.equal(pool.k_buffer[0][loc], k0))
        self.assertTrue(torch.equal(pool.v_buffer[0][loc], v0))

    def test_load_cpu_copy_row_count_must_match(self):
        """换入时每块的 K/V 行数必须与目标槽位数一致，否则断言失败。"""
        pool = _build_pool()
        loc = torch.tensor([1, 2])
        cpu = pool.get_cpu_copy(loc)  # 每块 2 行

        with self.assertRaises(AssertionError):
            pool.load_cpu_copy(cpu, torch.tensor([1, 2, 3]))

    def test_empty_offload_is_noop_and_returns_per_layer_lists(self):
        """空索引换出返回「每层一个空列表」，换入不改写任何数据。"""
        pool = _build_pool()
        for b in pool.k_buffer + pool.v_buffer:
            b.fill_(3)
        before = [b.clone() for b in pool.k_buffer + pool.v_buffer]
        indices = torch.empty(0, dtype=torch.int64)
        cpu = pool.get_cpu_copy(indices)
        self.assertEqual(cpu, [[], []])
        pool.load_cpu_copy(cpu, indices)
        for actual, expected in zip(pool.k_buffer + pool.v_buffer, before):
            torch.testing.assert_close(actual, expected)

    def test_offload_snapshot_can_load_into_different_slots(self):
        """换出快照独立于原池（可先改池再换回），且允许换入到不同槽位。"""
        pool = _build_pool()
        pool.cpu_offloading_chunk_size = 2
        src, tgt = torch.tensor([5, 2, 4]), torch.tensor([6, 9, 7])
        for layer in range(LAYER_NUM):
            pool.set_kv_buffer(
                _layer(layer), src, _rand_k(3) + layer, _rand_v(3) - layer
            )
        before = [b.clone() for b in pool.k_buffer + pool.v_buffer]
        cpu = pool.get_cpu_copy(src)
        for layer in cpu:
            for k, v in layer:
                self.assertEqual(k.device.type, "cpu")
                self.assertEqual(v.device.type, "cpu")
        for b in pool.k_buffer + pool.v_buffer:
            b.fill_(-1)  # CPU 快照必须独立，不能随原池变化。
        pool.load_cpu_copy(cpu, tgt)
        for actual, original in zip(pool.k_buffer + pool.v_buffer, before):
            expected = torch.full_like(original, -1)
            expected[tgt] = original[src]
            torch.testing.assert_close(actual, expected)

    def test_hnd_pool_rejects_cpu_offload(self):
        """HND 池按 (page, head) 组织，CPU offload 按槽位索引，二者不兼容。"""
        with unittest.mock.patch.dict(os.environ, {"SGLANG_USE_HND_KVCACHE": "1"}):
            pool = _build_pool()

        with self.assertRaisesRegex(AssertionError, "HND"):
            pool.get_cpu_copy(torch.tensor([1]))
        with self.assertRaises(AssertionError):
            pool.load_cpu_copy([], torch.tensor([1]))


@unittest.mock.patch.dict(os.environ, {"SGLANG_USE_HND_KVCACHE": "1"})
class TestMHAHNDLayout(_MHACPUTestCase):
    """HND 布局：4-D 缓冲、按 (page, off) 散写/搬运、描述符与拒绝路径。"""

    def test_hnd_buffer_shapes_and_layout(self):
        """HND 下缓冲是 4-D [页, head, 页内 token, dim]，页数含 padding 页。"""
        pool = _build_pool()

        self.assertTrue(pool.use_hnd)
        self.assertEqual(pool.kv_cache_layout, "hnd")
        self.assertEqual(pool.num_pages, NUM_SLOTS // PAGE_SIZE)
        self.assertEqual(
            tuple(pool.k_buffer[0].shape),
            (pool.num_pages, HEAD_NUM, PAGE_SIZE, HEAD_DIM),
        )
        self.assertEqual(
            tuple(pool.v_buffer[0].shape),
            (pool.num_pages, HEAD_NUM, PAGE_SIZE, V_HEAD_DIM),
        )

    def test_hnd_desc_tokens_per_row_is_page(self):
        """HND 的每「行」是整页，故 tokens_per_row == page_size。"""
        pool = _build_pool()

        descs = pool._kv_buffer_descs
        self.assertEqual(descs[0].tokens_per_row, PAGE_SIZE)
        self.assertEqual(
            descs[0].row_bytes, HEAD_NUM * PAGE_SIZE * HEAD_DIM * DTYPE.itemsize
        )

    def test_hnd_set_kv_buffer_scatters_by_page_and_offset(self):
        """HND 写入按 (slot // page, slot % page) 定位，散写到 4-D 缓冲。"""
        pool = _build_pool()
        loc = torch.tensor([1, 3])  # page 0/1, off 均为 1
        cache_k = _rand_k(2)
        cache_v = _rand_v(2)

        pool.set_kv_buffer(_layer(0), loc, cache_k, cache_v)

        self.assertTrue(torch.equal(pool.k_buffer[0][0, :, 1, :], cache_k[0]))
        self.assertTrue(torch.equal(pool.k_buffer[0][1, :, 1, :], cache_k[1]))
        self.assertTrue(torch.equal(pool.v_buffer[0][1, :, 1, :], cache_v[1]))

    def test_hnd_move_kv_cache_uses_page_offsets(self):
        """HND 搬运同样按 (page, off) 处理，源/目标 offset 不同也能正确定位。"""
        pool = _build_pool()
        src, tgt = torch.tensor([2, 5]), torch.tensor([7, 8])
        for layer in range(LAYER_NUM):
            pool.set_kv_buffer(
                _layer(layer), src, _rand_k(2) + layer, _rand_v(2) - layer
            )
        expected = [b.clone() for b in pool.k_buffer + pool.v_buffer]
        for b in expected:
            b[3, :, 1, :] = b[1, :, 0, :].clone()
            b[4, :, 0, :] = b[2, :, 1, :].clone()

        pool.move_kv_cache(tgt, src)

        for actual, wanted in zip(pool.k_buffer + pool.v_buffer, expected):
            torch.testing.assert_close(actual, wanted)

    def test_hnd_rejects_disagg_registration(self):
        """HND 不支持 PD 传输注册，get_contiguous_buf_infos 直接断言拒绝。"""
        pool = _build_pool()

        with self.assertRaises(AssertionError):
            pool.get_contiguous_buf_infos()

    def test_explicit_nhd_layout_overrides_hnd_env(self):
        """显式传 kv_cache_layout='nhd' 时优先于 HND 环境变量。"""
        pool = _build_pool(kv_cache_layout="nhd")
        self.assertFalse(pool.use_hnd)
        self.assertEqual(pool.kv_cache_layout, "nhd")
        self.assertEqual(pool.k_buffer[0].shape, (NUM_SLOTS, HEAD_NUM, HEAD_DIM))

    def test_hnd_requires_page_divisible_size(self):
        """HND 要求 (size + page_size) 能被 page_size 整除，否则构造即断言失败。"""
        with self.assertRaises(AssertionError):
            _build_pool(size=7)  # (7 + 2) % 2 != 0


class TestNoOpMHAPool(_MHACPUTestCase):
    """NoOpMHATokenToKVPool：只分配占位缓冲的 embedding/prefill-only 变体。"""

    def _build(self, **overrides):
        """构造 NoOpMHATokenToKVPool（默认参数与 _build_pool 对齐，可覆盖）。"""
        kwargs = dict(
            size=SIZE,
            page_size=PAGE_SIZE,
            dtype=DTYPE,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            layer_num=LAYER_NUM,
            device=DEVICE,
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        kwargs.update(overrides)
        return NoOpMHATokenToKVPool(**kwargs)

    def test_placeholder_buffers_keep_logical_size(self):
        """占位缓冲只有一页大小，但 size 仍是逻辑容量（供调度准入）。"""
        pool = self._build()

        self.assertEqual(pool.size, SIZE)  # 逻辑容量仍参与调度准入
        self.assertEqual(tuple(pool.k_buffer[0].shape), (PAGE_SIZE, HEAD_NUM, HEAD_DIM))
        self.assertEqual(
            tuple(pool.v_buffer[0].shape), (PAGE_SIZE, HEAD_NUM, V_HEAD_DIM)
        )

    def test_reports_zero_memory(self):
        """no-op 池不占真实 KV 显存，显存统计返回 0。"""
        pool = self._build()

        self.assertEqual(pool.get_kv_size_bytes(), (0, 0))
        self.assertEqual(pool.mem_usage, 0.0)

    def test_set_kv_buffer_raises(self):
        """误用写入路径时显式抛错，而不是静默丢数据。"""
        pool = self._build()

        with self.assertRaisesRegex(RuntimeError, "NoOpMHATokenToKVPool"):
            pool.set_kv_buffer(_layer(0), torch.tensor([0]), _rand_k(1), _rand_v(1))

    def test_placeholder_reads_and_move_noop(self):
        """读取返回占位缓冲；move_kv_cache 是 no-op 且不抛错。"""
        pool = self._build()

        self.assertIs(pool.get_key_buffer(1), pool.k_buffer[1])
        self.assertIs(pool.get_value_buffer(1), pool.v_buffer[1])
        k, v = pool.get_kv_buffer(0)
        self.assertIs(k, pool.k_buffer[0])
        self.assertIs(v, pool.v_buffer[0])
        pool.move_kv_cache(torch.tensor([1]), torch.tensor([2]))  # no-op，不抛错

    @unittest.mock.patch.dict(os.environ, {"SGLANG_USE_HND_KVCACHE": "1"})
    def test_ignores_hnd_env(self):
        """即使开了 HND 环境变量，no-op 池仍用占位 NHD 缓冲。"""
        pool = self._build()

        self.assertFalse(pool.use_hnd)
        self.assertEqual(pool.kv_cache_layout, "nhd")
        self.assertEqual(tuple(pool.k_buffer[0].shape), (PAGE_SIZE, HEAD_NUM, HEAD_DIM))


class TestMHAQuantizedContract(_MHACPUTestCase):
    """量化策略是外部依赖：测试 buffer 所有权、参数转发、保护条件，而非量化算法。"""

    def _make_quantized(self, **overrides):
        """用 mock 量化策略构造池，返回 (pool, method, buffers)，便于断言委托关系。"""
        method = unittest.mock.Mock()
        method.name = "test_quantized"
        method.dequant_workspace_dtype.return_value = torch.float32
        method.scale_buffer_view_dtype.return_value = None
        method.needs_plain_kv_dequant_read.return_value = True
        shape = (NUM_SLOTS, HEAD_NUM, HEAD_DIM)
        buffers = {
            "k_buffer": [
                torch.zeros(shape, dtype=torch.uint8) for _ in range(LAYER_NUM)
            ],
            "v_buffer": [
                torch.zeros(shape, dtype=torch.uint8) for _ in range(LAYER_NUM)
            ],
            "k_scale_buffer": [
                torch.ones(NUM_SLOTS, HEAD_NUM, 1, dtype=torch.uint8)
                for _ in range(LAYER_NUM)
            ],
            "v_scale_buffer": [
                torch.ones(NUM_SLOTS, HEAD_NUM, 1, dtype=torch.uint8)
                for _ in range(LAYER_NUM)
            ],
            "dq_k_buffer": torch.zeros(shape, dtype=torch.float32),
            "dq_v_buffer": torch.zeros(shape, dtype=torch.float32),
            "store_dtype": torch.uint8,
        }
        method.create_buffers.return_value = buffers
        method.k_scales_gpu = torch.arange(1, 9, dtype=torch.float32)
        method.v_scales_gpu = torch.arange(11, 19, dtype=torch.float32)
        pool = _build_pool(v_head_dim=HEAD_DIM, quant_method=method, **overrides)
        return pool, method, buffers

    def test_strategy_owns_buffers_pointers_and_memory_accounting(self):
        """缓冲所有权归策略；指针/跨度表与显存统计都要把它纳入计算。"""
        pool, method, buffers = self._make_quantized()
        method.create_buffers.assert_called_once_with(
            NUM_SLOTS, HEAD_NUM, HEAD_DIM, LAYER_NUM, DEVICE
        )
        self.assertTrue(pool.is_quantized_kv_cache)
        self.assertIs(pool.get_kv_cache_quant_method(), method)
        for name, value in buffers.items():
            self.assertIs(getattr(pool, name), value)
        all_buffers = (
            pool.k_buffer + pool.v_buffer + pool.k_scale_buffer + pool.v_scale_buffer
        )
        self.assertEqual(pool.data_ptrs.tolist(), [b.data_ptr() for b in all_buffers])
        self.assertEqual(
            pool.data_strides.tolist(),
            [b[0].numel() * b.element_size() for b in all_buffers],
        )
        expected = []
        for prefix in ("k", "v"):
            tensors = (
                buffers[prefix + "_buffer"]
                + buffers[prefix + "_scale_buffer"]
                + [buffers["dq_" + prefix + "_buffer"]]
            )
            expected.append(sum(b.numel() * b.element_size() for b in tensors))
        self.assertEqual(pool.get_kv_size_bytes(), tuple(expected))

    def test_quantized_write_uses_local_buffer_and_global_scales(self):
        """量化写入用本地层缓冲；未显式给 scale 时按全局层号取策略的 scale。"""
        pool, method, _ = self._make_quantized(start_layer=5, end_layer=6)
        loc = torch.tensor([2])
        k, v = _rand_k(1), _rand_k(1) + 1
        for k_scale, v_scale in ((None, None), (2.0, 3.0)):
            with self.subTest(explicit=k_scale is not None):
                method.quantize_and_store.reset_mock()
                pool.set_kv_buffer(
                    _layer(5),
                    KVWriteLoc(loc),
                    k,
                    v,
                    k_scale,
                    v_scale,
                    layer_id_override=6,
                )
                method.quantize_and_store.assert_called_once()
                args = method.quantize_and_store.call_args.args
                for actual, expected in zip(
                    args[:7],
                    (
                        pool.k_buffer[1],
                        pool.v_buffer[1],
                        pool.k_scale_buffer[1],
                        pool.v_scale_buffer[1],
                        loc,
                        k,
                        v,
                    ),
                ):
                    self.assertIs(actual, expected)
                if k_scale is None:
                    torch.testing.assert_close(args[7], method.k_scales_gpu[5:6])
                    torch.testing.assert_close(args[8], method.v_scales_gpu[5:6])
                else:
                    self.assertEqual(args[7:], (k_scale, v_scale))

    def test_quantized_read_delegates_and_exposes_raw_buffers(self):
        """读取走策略反量化并透传原始数据与 scale；raw/workspace 接口返回同一批缓冲。"""
        pool, method, _ = self._make_quantized(start_layer=5, end_layer=6)
        k, v = torch.full_like(pool.dq_k_buffer, 3), torch.full_like(
            pool.dq_v_buffer, 7
        )
        method.dequantize_kv_tensor.side_effect = [k, v]
        actual_k, actual_v = pool.get_kv_buffer(6)
        self.assertIs(actual_k, k)
        self.assertIs(actual_v, v)
        for call, data, scale in zip(
            method.dequantize_kv_tensor.call_args_list,
            (pool.k_buffer[1], pool.v_buffer[1]),
            (pool.k_scale_buffer[1], pool.v_scale_buffer[1]),
        ):
            self.assertIs(call.args[0], data)
            self.assertIs(call.args[1], scale)
            self.assertEqual(call.args[2], 6)
        for actual, expected in zip(
            pool.get_raw_kv_buffer(6),
            (
                pool.k_buffer[1],
                pool.v_buffer[1],
                pool.k_scale_buffer[1],
                pool.v_scale_buffer[1],
            ),
        ):
            self.assertIs(actual, expected)
        self.assertIs(pool.get_dequant_workspace()[0], pool.dq_k_buffer)
        self.assertIs(pool.get_dequant_workspace()[1], pool.dq_v_buffer)

    def test_quantized_workspace_validation(self):
        """workspace 与策略声明的四类矛盾（只建一个、都没建、未声明、dtype 不符）都要报错。"""
        for case, message in (
            ("one", "only one"),
            ("absent", "did not create"),
            ("undeclared", "does not declare"),
            ("dtype", "workspace dtype"),
        ):
            with self.subTest(case=case):
                pool, method, _ = self._make_quantized()
                if case == "one":
                    pool.dq_v_buffer = None
                elif case == "absent":
                    pool.dq_k_buffer = pool.dq_v_buffer = None
                elif case == "undeclared":
                    method.dequant_workspace_dtype.return_value = None
                else:
                    pool.dq_v_buffer = pool.dq_v_buffer.to(torch.float16)
                with self.assertRaisesRegex(RuntimeError, message):
                    pool._check_quantized_buffer_access_requirements()

    def test_quantized_move_includes_scales_and_cleanup(self):
        """搬运要连同 scale 缓冲一起搬；清理时六类缓冲全部销毁。"""
        pool, _, _ = self._make_quantized()
        pool.use_native_move_kv_cache = True
        buffers = (
            pool.k_buffer + pool.v_buffer + pool.k_scale_buffer + pool.v_scale_buffer
        )
        for i, b in enumerate(buffers):
            b[2].fill_(i + 2)
        expected = [b.clone() for b in buffers]
        for b in expected:
            b[5] = b[2]
        pool.move_kv_cache(torch.tensor([5]), torch.tensor([2]))
        for actual, wanted in zip(buffers, expected):
            torch.testing.assert_close(actual, wanted)
        pool._clear_buffers()
        for name in (
            "k_buffer",
            "v_buffer",
            "k_scale_buffer",
            "v_scale_buffer",
            "dq_k_buffer",
            "dq_v_buffer",
        ):
            self.assertFalse(hasattr(pool, name))

    def test_quantized_rejects_mask_and_post_capture(self):
        """量化池拒绝 dcp_kv_mask，且不支持 post-capture 延迟 backing。"""
        pool, method, _ = self._make_quantized()
        with self.assertRaisesRegex(RuntimeError, "dcp_kv_mask"):
            pool.set_kv_buffer(
                _layer(),
                torch.tensor([2]),
                _rand_k(1),
                _rand_k(1),
                dcp_kv_mask=torch.tensor([True]),
            )
        method.quantize_and_store.assert_not_called()
        with self.assertRaisesRegex(NotImplementedError, "Post-capture"):
            _build_pool(quant_method=method, post_capture_active=True)


class TestMHAPostCapture(_MHACPUTestCase):
    """用真实 CPU 张量替代 VMM owner，验证池与 owner 的生命周期协议。"""

    def test_post_capture_rounds_capacity_and_finalizes_backing(self):
        """预留容量按页上取整；finalize 只改容量与 backing，不改虚拟地址。"""
        tensors = [
            torch.zeros(NUM_SLOTS, HEAD_NUM, dim, dtype=DTYPE)
            for dim in (HEAD_DIM, HEAD_DIM, V_HEAD_DIM, V_HEAD_DIM)
        ]
        owner = unittest.mock.Mock(tensors=tensors, backed_bytes=0)
        with unittest.mock.patch.object(
            _mp, "KvVmmBufferOwner", return_value=owner
        ) as factory, unittest.mock.patch.object(
            torch.cuda, "current_device", return_value=0
        ):
            pool = _build_pool(size=7, post_capture_active=True)
        self.assertEqual(pool.size, 8)  # 预留容量向上按页对齐。
        self.assertIs(pool.k_buffer[0], tensors[0])
        self.assertIs(pool.v_buffer[1], tensors[3])
        self.assertEqual(factory.call_args.kwargs["reserved_num_tokens"], 8)
        self.assertEqual(pool.post_capture_backed_bytes, 0)
        pointers = pool.data_ptrs.clone()
        pool.finalize_backing(types.SimpleNamespace(max_total_num_tokens=4))
        owner.finalize.assert_called_once_with(4)
        owner.backed_bytes = 1234
        self.assertEqual(pool.size, 4)
        self.assertEqual(pool.post_capture_backed_bytes, 1234)
        torch.testing.assert_close(
            pool.data_ptrs, pointers
        )  # finalize 不改变虚拟地址。
        _, lens, _ = pool.get_contiguous_buf_infos()
        self.assertEqual(
            lens,
            [
                (4 + PAGE_SIZE) * HEAD_NUM * dim * DTYPE.itemsize
                for dim in (HEAD_DIM, HEAD_DIM, V_HEAD_DIM, V_HEAD_DIM)
            ],
        )
        pool._clear_buffers()
        owner.close.assert_called_once()
        self.assertIsNone(pool._post_capture_owner)
        self.assertEqual(pool.post_capture_backed_bytes, 0)


class TestMHAMisc(_MHACPUTestCase):
    """外围契约：FP4 专用接口守卫、缓冲释放。"""

    def test_fp4_only_interfaces_raise_on_plain_pool(self):
        """普通池访问 FP4 专用接口应显式报错，避免拿到 None 静默出错。"""
        pool = _build_pool()

        with self.assertRaisesRegex(RuntimeError, "non-FP4"):
            pool.get_raw_kv_buffer(0)
        with self.assertRaisesRegex(RuntimeError, "dequant buffers"):
            pool.get_dequant_workspace()

    def test_clear_buffers_releases_storage(self):
        """_clear_buffers 真正释放 K/V 缓冲（属性被删除，而不是置空）。"""
        pool = _build_pool()

        pool._clear_buffers()

        self.assertFalse(hasattr(pool, "k_buffer"))
        self.assertFalse(hasattr(pool, "v_buffer"))


if __name__ == "__main__":
    # 支持选择单个 case / 单个类：
    #   python <file> TestMHAConstruction.test_default_layout_is_nhd
    #   python <file> TestMHAConstruction
    # 不传参数则运行本文件全部用例。
    unittest.main(verbosity=2)
