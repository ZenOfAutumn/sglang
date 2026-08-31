"""DFLASH spec-v2 overlap 调度数据结构模块。

本模块定义 DFlashDraftInputV2，用于 DFLASH V2 算法草稿侧的 overlap 调度。
DFLASH V2 采用「预分配」策略：提前预留未来 KV 槽位，使 worker 可以直接
从 req_to_token 获取 out_cache_loc，无需分配器 backup/restore。

关键概念：
    - committed_len:  目标模型已确认的 KV 长度（滞后一轮）
    - planning_len:   committed_len + block_size，本步草稿规划长度
    - reserved_len:   max(cur_alloc, committed + 2*block_size)，预留上界
    - CPU 元数据滞后一轮，与 GPU 预留上界分开维护
"""

import contextlib
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils.common import is_pin_memory_available

# 每设备缓存一个 plan stream，用于 overlap 调度的 prep kernel
_OVERLAP_PLAN_STREAMS: dict[str, torch.cuda.Stream] = {}


def _get_overlap_plan_stream(
    device: torch.device | str,
) -> tuple[Optional[torch.cuda.Stream], contextlib.AbstractContextManager]:
    """返回 overlap 调度 prep kernel 的可选 plan stream 和上下文。

    当 SGLANG_ENABLE_OVERLAP_PLAN_STREAM 启用时，返回一个专用 stream
    用于执行 prepare_for_decode 中的预分配 kernel，使其与前向计算重叠。
    未启用时返回 (None, nullcontext)。

    参数：
        device: 目标设备

    返回：(stream, context_manager)
        - stream: CUDA stream 或 None
        - context_manager: 用于切换 stream 的上下文

    示例：
        >>> stream, ctx = _get_overlap_plan_stream("cuda:0")
        >>> with ctx:
        ...     # 在 plan stream 上执行 prep kernel
        ...     prepare_kernels()
    """
    if not envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        return None, contextlib.nullcontext()

    device_str = str(device)
    stream = _OVERLAP_PLAN_STREAMS.get(device_str)
    if stream is None:
        stream = torch.get_device_module(device_str).Stream()
        _OVERLAP_PLAN_STREAMS[device_str] = stream
    return stream, torch.get_device_module(device_str).stream(stream)


@dataclass
class DFlashDraftInputV2(SpecInput):
    """DFLASH V2 草稿侧状态，在 overlap 迭代间传递（spec-v2）。

    保留了 Eagle 兼容的字段（topk_p / topk_index），但 DFLASH overlap
    在常见无形状变化路径中直接携带 new_seq_lens / verified_id；
    FutureMap 作为 filter/merge 的回退方案。

    核心属性：
        topk_p / topk_index:  Eagle 兼容字段（DFLASH 中实际未使用树形选择）
        verified_id:          上轮验证接受的 token id
        new_seq_lens:         新序列长度
        hidden_states:        隐藏状态
        verify_done:          验证完成事件（用于同步）
        max_top_k:            最大 top_k 值
        uniform_top_k_value:  统一 top_k 值（所有请求相同时）
        cur_allocated_seq_lens_cpu: 当前已分配 KV 长度（CPU）
        planning_seq_lens_cpu: 规划 KV 长度（CPU）
        reserved_seq_lens_cpu: 预留 KV 长度上界（CPU）
        direct_carry_valid:   是否可直接携带状态（filter/merge 后置 False）
        future_indices:       调度器分派后填充的索引

    示例：
        >>> # 正常创建
        >>> draft_input = DFlashDraftInputV2(
        ...     topk_p=torch.tensor([[0.5]]),
        ...     topk_index=torch.tensor([[42]]),
        ...     verified_id=torch.tensor([42]),
        ...     new_seq_lens=torch.tensor([10]),
        ...     hidden_states=torch.zeros(1, 4096),
        ... )
        >>> # idle 创建
        >>> idle_input = DFlashDraftInputV2.create_idle_device(torch.device("cuda"))
    """

    # Eagle 兼容字段，仅为了 dataclass 兼容性保留。
    # DFLASH overlap 在常见无形状变化路径中直接携带 new_seq_lens / verified_id；
    # FutureMap 作为 filter/merge 的回退方案。
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    verified_id: torch.Tensor
    new_seq_lens: torch.Tensor
    hidden_states: torch.Tensor
    verify_done: Optional[torch.cuda.Event] = None
    max_top_k: int = 1
    uniform_top_k_value: Optional[int] = None
    cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None
    planning_seq_lens_cpu: Optional[torch.Tensor] = None
    planning_seq_lens_sum: Optional[int] = None
    reserved_seq_lens_cpu: Optional[torch.Tensor] = None
    reserved_seq_lens_sum: Optional[int] = None
    direct_carry_valid: bool = True
    # prepare_for_decode 使用的 CPU/GPU 缓冲区（按需增长）
    _prepare_committed_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_planning_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_batch_seq_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_cur_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_nxt_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_cur_kv_lens_gpu_buf: Optional[torch.Tensor] = None
    _prepare_nxt_kv_lens_gpu_buf: Optional[torch.Tensor] = None

    # 调度器分派后填充的索引。
    future_indices: Optional[torch.Tensor] = None

    def __post_init__(self):
        """初始化后处理：设置 spec_input_type 为 DFLASH_DRAFT。"""
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        """返回投机 token 调整系数。

        Spec v2 草稿状态本身不改变 token 计数。

        返回：(1, 1)
        """
        # Spec v2 草稿状态本身不改变 token 计数。
        return (1, 1)

    def _ensure_prepare_length_buffers(
        self, bs: int, device: torch.device | str
    ) -> None:
        """确保 prepare_for_decode 使用的 CPU/GPU 长度缓冲区足够大。

        按需增长策略：新容量 = max(bs, 32, current * 2)。
        CPU 缓冲区考虑 pin_memory，GPU 缓冲区确保在正确设备上。

        参数：
            bs:     batch size
            device: 目标设备
        """
        pin_memory = is_pin_memory_available(device)

        def needs_cpu_alloc(buf: Optional[torch.Tensor]) -> bool:
            return buf is None or buf.numel() < bs or buf.is_pinned() != pin_memory

        def needs_gpu_alloc(buf: Optional[torch.Tensor]) -> bool:
            return buf is None or buf.numel() < bs or str(buf.device) != str(device)

        def grown_capacity(buf: Optional[torch.Tensor]) -> int:
            current = 0 if buf is None else int(buf.numel())
            return max(bs, 32, current * 2 if current > 0 else 0)

        if needs_cpu_alloc(self._prepare_committed_kv_lens_cpu_buf):
            capacity = grown_capacity(self._prepare_committed_kv_lens_cpu_buf)
            self._prepare_committed_kv_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )
            self._prepare_planning_kv_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )
            self._prepare_batch_seq_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int64, device="cpu"
            )
            self._prepare_cur_kv_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )
            self._prepare_nxt_kv_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )

        if needs_gpu_alloc(self._prepare_cur_kv_lens_gpu_buf):
            capacity = grown_capacity(self._prepare_cur_kv_lens_gpu_buf)
            self._prepare_cur_kv_lens_gpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device=device
            )
            self._prepare_nxt_kv_lens_gpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device=device
            )

    @classmethod
    def create_idle_input(cls, device: torch.device) -> "DFlashDraftInputV2":
        """创建一个空的 idle 输入（用于无请求时的占位）。

        参数：
            device: 目标设备

        返回：空的 DFlashDraftInputV2

        示例：
            >>> idle = DFlashDraftInputV2.create_idle_input(torch.device("cuda"))
            >>> # idle.verified_id.numel() == 0
        """
        return cls(
            topk_p=torch.empty((0, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, 0), device=device, dtype=torch.int64),
            verified_id=torch.empty((0,), device=device, dtype=torch.int32),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
            hidden_states=torch.empty((0, 0), device=device, dtype=torch.float16),
            verify_done=None,
        )

    def prepare_for_decode(self, batch: ScheduleBatch):
        """为下一个 DFLASH 步骤在共享 req_to_token 池中预分配 KV 槽位。

        DFLASH spec-v2 使用 overlap 调度的「预分配」策略：提前预留
        未来 KV 槽位，使 worker 可以直接从 req_to_token 获取 out_cache_loc，
        无需分配器 backup/restore。CPU 元数据有意滞后一轮，与支持
        预分配映射的预留上界分开维护。

        核心流程：
            1. 计算每个请求的 committed/planning/reserved 长度
            2. 在 plan stream 或同步模式下分配 KV 槽位
            3. 更新 req_to_token 映射表
            4. 更新请求的 kv_allocated_len 高水位标记

        参数：
            batch: 当前调度 batch

        示例：
            >>> # batch 有 2 个请求，block_size=4
            >>> draft_input.prepare_for_decode(batch)
            >>> # batch.req_to_token 中已预填充了未来 KV 槽位
            >>> # draft_input.reserved_seq_lens_cpu = tensor([14, 18])
        """
        plan_stream, plan_stream_ctx = _get_overlap_plan_stream(batch.device)
        if plan_stream is None and self.verify_done is not None:
            # 确保前一个前向完成后再修改共享缓冲区。
            self.verify_done.synchronize()

        bs = batch.batch_size()
        if bs == 0:
            return
        self._ensure_prepare_length_buffers(bs, batch.device)
        assert self._prepare_committed_kv_lens_cpu_buf is not None
        assert self._prepare_planning_kv_lens_cpu_buf is not None
        assert self._prepare_batch_seq_lens_cpu_buf is not None
        assert self._prepare_cur_kv_lens_cpu_buf is not None
        assert self._prepare_nxt_kv_lens_cpu_buf is not None
        assert self._prepare_cur_kv_lens_gpu_buf is not None
        assert self._prepare_nxt_kv_lens_gpu_buf is not None
        committed_kv_lens_cpu_t = self._prepare_committed_kv_lens_cpu_buf[:bs]
        planning_kv_lens_cpu_t = self._prepare_planning_kv_lens_cpu_buf[:bs]
        batch_seq_lens_cpu_t = self._prepare_batch_seq_lens_cpu_buf[:bs]
        cur_kv_lens_cpu_t = self._prepare_cur_kv_lens_cpu_buf[:bs]
        cur_allocated_seq_lens_cpu = self.cur_allocated_seq_lens_cpu

        # DFLASH 每步需要一个固定大小的验证 block。
        block_size = int(get_global_server_args().speculative_num_draft_tokens)
        if block_size <= 0:
            raise ValueError(
                f"DFLASH invalid speculative_num_draft_tokens={block_size}."
            )
        page_size = batch.token_to_kv_pool_allocator.page_size
        nxt_kv_lens_cpu_t = self._prepare_nxt_kv_lens_cpu_buf[:bs]
        committed_seq_lens_sum = 0
        planning_seq_lens_sum = 0
        reserved_seq_lens_sum = 0
        num_needed_tokens = 0
        max_top_k = 1
        uniform_top_k_value = None
        uniform_top_k = True
        for i, req in enumerate(batch.reqs):
            committed_len = int(req.kv_committed_len)
            if cur_allocated_seq_lens_cpu is not None and i < len(
                cur_allocated_seq_lens_cpu
            ):
                cur_alloc_len = int(cur_allocated_seq_lens_cpu[i])
            else:
                cur_alloc_len = int(req.kv_allocated_len)
            planning_len = committed_len + block_size
            reserved_len = max(cur_alloc_len, committed_len + 2 * block_size)
            top_k = int(req.sampling_params.top_k)

            committed_kv_lens_cpu_t[i] = committed_len
            batch_seq_lens_cpu_t[i] = committed_len
            cur_kv_lens_cpu_t[i] = cur_alloc_len
            planning_kv_lens_cpu_t[i] = planning_len
            nxt_kv_lens_cpu_t[i] = reserved_len

            committed_seq_lens_sum += committed_len
            planning_seq_lens_sum += planning_len
            reserved_seq_lens_sum += reserved_len
            num_needed_tokens += reserved_len - cur_alloc_len

            if top_k > max_top_k:
                max_top_k = top_k
            if i == 0:
                uniform_top_k_value = top_k
            elif uniform_top_k and top_k != uniform_top_k_value:
                uniform_top_k = False

        self.max_top_k = max(max_top_k, 1)
        self.uniform_top_k_value = uniform_top_k_value if uniform_top_k else None

        caller_stream = None
        if plan_stream is not None:
            caller_stream = torch.get_device_module(batch.device).current_stream()

        with plan_stream_ctx:
            if plan_stream is not None and caller_stream is not None:
                # `batch.seq_lens`、`batch.req_pool_indices` 等张量可能刚在
                # 调度器 stream 上被 filter/merge 重建。plan stream 必须等待
                # 这些写入完成后才能读取。
                plan_stream.wait_stream(caller_stream)

            if plan_stream is not None and self.verify_done is not None:
                plan_stream.wait_event(self.verify_done)

            cur_kv_lens = self._prepare_cur_kv_lens_gpu_buf[:bs]
            nxt_kv_lens = self._prepare_nxt_kv_lens_gpu_buf[:bs]
            cur_kv_lens.copy_(cur_kv_lens_cpu_t, non_blocking=True)
            nxt_kv_lens.copy_(nxt_kv_lens_cpu_t, non_blocking=True)

            if num_needed_tokens > 0:
                if page_size == 1:
                    out_cache_loc = alloc_token_slots(
                        batch.tree_cache, num_needed_tokens
                    )
                else:
                    last_loc = get_last_loc(
                        batch.req_to_token_pool.req_to_token,
                        batch.req_pool_indices,
                        cur_kv_lens,
                    )
                    out_cache_loc = alloc_paged_token_slots_extend(
                        batch.tree_cache,
                        cur_kv_lens,
                        cur_kv_lens_cpu_t,
                        nxt_kv_lens,
                        nxt_kv_lens_cpu_t,
                        last_loc,
                        num_needed_tokens,
                    )

                # 更新 req_to_token 是对共享张量的写入：不能与前一个 batch
                # 的前向（同样读取 req_to_token）重叠。
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    cur_kv_lens,
                    nxt_kv_lens,
                    out_cache_loc,
                    bs,
                )
        if caller_stream is not None:
            # 在调用方 stream 上排队依赖，而不是在 plan-stream 上下文内，
            # 这样前向工作不会观察到部分准备的 req_to_token / KV 分配状态。
            caller_stream.wait_stream(plan_stream)

        # 请求侧的高水位标记用于 release_kv_cache() 在请求结束后
        # 回收 DFLASH 预分配的空间。
        for i, req in enumerate(batch.reqs):
            req.kv_allocated_len = max(req.kv_allocated_len, int(nxt_kv_lens_cpu_t[i]))

        # 保留滞后的 committed CPU 视图，并单独维护更紧凑的 planning 上界，
        # 与完整的预留分配器上界分开。Overlap 调度在 committed 前缀长度上
        # 最多漂移一个 DFlash block。
        batch.seq_lens_cpu = batch_seq_lens_cpu_t
        batch.seq_lens_sum = committed_seq_lens_sum
        self.planning_seq_lens_cpu = planning_kv_lens_cpu_t
        self.planning_seq_lens_sum = planning_seq_lens_sum
        self.reserved_seq_lens_cpu = nxt_kv_lens_cpu_t
        self.reserved_seq_lens_sum = reserved_seq_lens_sum

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        """过滤 batch：根据 new_indices 索引保留对应请求的状态。

        参数：
            new_indices:         要保留的请求索引
            has_been_filtered:   是否已被过滤过
        """
        if self.cur_allocated_seq_lens_cpu is not None:
            self.cur_allocated_seq_lens_cpu = self.cur_allocated_seq_lens_cpu[
                new_indices.cpu()
            ]
        if self.planning_seq_lens_cpu is not None:
            self.planning_seq_lens_cpu = self.planning_seq_lens_cpu[new_indices.cpu()]
            self.planning_seq_lens_sum = int(self.planning_seq_lens_cpu.sum().item())
        if self.reserved_seq_lens_cpu is not None:
            self.reserved_seq_lens_cpu = self.reserved_seq_lens_cpu[new_indices.cpu()]
            self.reserved_seq_lens_sum = int(self.reserved_seq_lens_cpu.sum().item())

        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            self.direct_carry_valid = False
            return

        self.topk_p = self.topk_p[new_indices]
        self.topk_index = self.topk_index[new_indices]
        self.verified_id = self.verified_id[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]
        self.hidden_states = self.hidden_states[new_indices]

    def merge_batch(self, spec_info: "DFlashDraftInputV2"):
        """合并两个 batch 的草稿状态。

        将 spec_info 中的所有状态张量拼接到 self 中。

        参数：
            spec_info: 要合并的另一个 DFlashDraftInputV2
        """
        if self.cur_allocated_seq_lens_cpu is not None:
            assert spec_info.cur_allocated_seq_lens_cpu is not None
            self.cur_allocated_seq_lens_cpu = torch.cat(
                [self.cur_allocated_seq_lens_cpu, spec_info.cur_allocated_seq_lens_cpu]
            )
        elif spec_info.cur_allocated_seq_lens_cpu is not None:
            self.cur_allocated_seq_lens_cpu = spec_info.cur_allocated_seq_lens_cpu

        if self.planning_seq_lens_cpu is not None:
            assert spec_info.planning_seq_lens_cpu is not None
            self.planning_seq_lens_cpu = torch.cat(
                [self.planning_seq_lens_cpu, spec_info.planning_seq_lens_cpu]
            )
            self.planning_seq_lens_sum = int(self.planning_seq_lens_cpu.sum().item())
        elif spec_info.planning_seq_lens_cpu is not None:
            self.planning_seq_lens_cpu = spec_info.planning_seq_lens_cpu
            self.planning_seq_lens_sum = spec_info.planning_seq_lens_sum

        if self.reserved_seq_lens_cpu is not None:
            assert spec_info.reserved_seq_lens_cpu is not None
            self.reserved_seq_lens_cpu = torch.cat(
                [self.reserved_seq_lens_cpu, spec_info.reserved_seq_lens_cpu]
            )
            self.reserved_seq_lens_sum = int(self.reserved_seq_lens_cpu.sum().item())
        elif spec_info.reserved_seq_lens_cpu is not None:
            self.reserved_seq_lens_cpu = spec_info.reserved_seq_lens_cpu
            self.reserved_seq_lens_sum = spec_info.reserved_seq_lens_sum

        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = torch.cat(
                [self.future_indices, spec_info.future_indices]
            )
            self.direct_carry_valid = False
            return

        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p], dim=0)
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index], dim=0)
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        self.new_seq_lens = torch.cat(
            [self.new_seq_lens, spec_info.new_seq_lens], dim=0
        )
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], dim=0
        )
