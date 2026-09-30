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
from typing import List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocation import alloc_for_spec_decode
from sglang.srt.mem_cache.allocation_sizing import page_aligned_decode_alloc_lens
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
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

    Draft-side state carried across overlap iterations (spec-v2).
    """

    # Legacy Eagle-shaped fields; DFLASH relays via FutureMap so these are unused.
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    bonus_tokens: torch.Tensor
    new_seq_lens: torch.Tensor
    hidden_states: torch.Tensor
    max_top_k: int = 1
    uniform_top_k_value: Optional[int] = None
    nxt_kv_lens_cpu: Optional[torch.Tensor] = None
    nxt_kv_lens_sum: Optional[int] = None
    _prepare_batch_seq_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_cur_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_nxt_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_cur_kv_lens_gpu_buf: Optional[torch.Tensor] = None
    _prepare_nxt_kv_lens_gpu_buf: Optional[torch.Tensor] = None

    # 调度器分派后填充的索引。
    future_indices: Optional[torch.Tensor] = None

    verify_token_budget: Optional[int] = None

    def __post_init__(self):
        """初始化后处理：设置 spec_input_type 为 DFLASH_DRAFT。"""
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)
        # Spec v2 draft state itself does not change token accounting.
        self.num_tokens_per_req = 1
        self.num_tokens_for_logprob_per_req = 1

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
            return buf is None or buf.numel() < bs

        def needs_gpu_alloc(buf: Optional[torch.Tensor]) -> bool:
            return buf is None or buf.numel() < bs or str(buf.device) != str(device)

        def grown_capacity(buf: Optional[torch.Tensor]) -> int:
            current = 0 if buf is None else int(buf.numel())
            return max(bs, 32, current * 2 if current > 0 else 0)

        # The three CPU scratch buffers grow together; capacity is the only
        # invariant (batch is int64 non-pinned, cur/nxt are int32 pinned).
        if needs_cpu_alloc(self._prepare_batch_seq_lens_cpu_buf):
            capacity = grown_capacity(self._prepare_batch_seq_lens_cpu_buf)
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
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
            hidden_states=torch.empty((0, 0), device=device, dtype=torch.float16),
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

        bs = batch.batch_size()
        if bs == 0:
            return

        batch.maybe_evict_swa()

        self._ensure_prepare_length_buffers(bs, batch.device)
        assert self._prepare_batch_seq_lens_cpu_buf is not None
        assert self._prepare_cur_kv_lens_cpu_buf is not None
        assert self._prepare_nxt_kv_lens_cpu_buf is not None
        assert self._prepare_cur_kv_lens_gpu_buf is not None
        assert self._prepare_nxt_kv_lens_gpu_buf is not None
        batch_seq_lens_cpu_t = self._prepare_batch_seq_lens_cpu_buf[:bs]
        cur_kv_lens_cpu_t = self._prepare_cur_kv_lens_cpu_buf[:bs]
        nxt_kv_lens_cpu_t = self._prepare_nxt_kv_lens_cpu_buf[:bs]

        # For DFLASH, each decode step needs a fixed-size verify block.
        block_size = int(get_spec().speculative_num_draft_tokens)
        if block_size <= 0:
            raise ValueError(
                f"DFLASH invalid speculative_num_draft_tokens={block_size}."
            )
        reserve = 2 * block_size
        page_size = batch.token_to_kv_pool_allocator.page_size

        cur_kv_lens, nxt_kv_lens, num_needed_tokens = page_aligned_decode_alloc_lens(
            batch.reqs,
            reserve=reserve,
            page_size=page_size,
        )

        max_top_k = 1
        uniform_top_k_value = None
        uniform_top_k = True
        nxt_kv_lens_sum = 0
        committed_seq_lens_sum = 0
        for i, (req, cur, nxt) in enumerate(zip(batch.reqs, cur_kv_lens, nxt_kv_lens)):
            committed_len = int(req.kv.kv_committed_len)
            committed_seq_lens_sum += committed_len
            top_k = int(req.sampling_params.top_k)

            batch_seq_lens_cpu_t[i] = committed_len
            cur_kv_lens_cpu_t[i] = cur
            nxt_kv_lens_cpu_t[i] = nxt

            nxt_kv_lens_sum += nxt
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

            cur_kv_lens = self._prepare_cur_kv_lens_gpu_buf[:bs]
            nxt_kv_lens = self._prepare_nxt_kv_lens_gpu_buf[:bs]
            cur_kv_lens.copy_(cur_kv_lens_cpu_t, non_blocking=True)
            nxt_kv_lens.copy_(nxt_kv_lens_cpu_t, non_blocking=True)

            alloc_for_spec_decode(
                batch.tree_cache,
                batch.req_to_token_pool,
                reqs=batch.reqs,
                req_pool_indices=batch.req_pool_indices,
                cur_kv_lens=cur_kv_lens,
                cur_kv_lens_cpu=cur_kv_lens_cpu_t,
                nxt_kv_lens=nxt_kv_lens,
                nxt_kv_lens_cpu=nxt_kv_lens_cpu_t,
                num_needed_tokens=num_needed_tokens,
                batch=batch,
            )
        if caller_stream is not None:
            # 在调用方 stream 上排队依赖，而不是在 plan-stream 上下文内，
            # 这样前向工作不会观察到部分准备的 req_to_token / KV 分配状态。
            caller_stream.wait_stream(plan_stream)
        for req in batch.reqs:
            req.decode_batch_idx += 1
        # Seed committed; overlap's resolve overwrites it with the published value.
        batch.seq_lens_cpu = batch_seq_lens_cpu_t
        batch.seq_lens_sum = committed_seq_lens_sum
        self.nxt_kv_lens_cpu = nxt_kv_lens_cpu_t
        self.nxt_kv_lens_sum = nxt_kv_lens_sum

    def filter_batch(
        self,
        new_indices: torch.Tensor,
        new_indices_cpu: Optional[List[int]] = None,
    ):
        if self.nxt_kv_lens_cpu is not None:
            if new_indices_cpu is not None:
                self.nxt_kv_lens_cpu = self.nxt_kv_lens_cpu[new_indices_cpu]
            else:
                self.nxt_kv_lens_cpu = self.nxt_kv_lens_cpu[new_indices.cpu()]
            self.nxt_kv_lens_sum = int(self.nxt_kv_lens_cpu.sum().item())

        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            return

        self.topk_p = self.topk_p[new_indices]
        self.topk_index = self.topk_index[new_indices]
        self.bonus_tokens = self.bonus_tokens[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]
        self.hidden_states = self.hidden_states[new_indices]

    def merge_batch(self, spec_info: "DFlashDraftInputV2"):
        if self.nxt_kv_lens_cpu is not None:
            assert spec_info.nxt_kv_lens_cpu is not None
            self.nxt_kv_lens_cpu = torch.cat(
                [self.nxt_kv_lens_cpu, spec_info.nxt_kv_lens_cpu]
            )
            self.nxt_kv_lens_sum = int(self.nxt_kv_lens_cpu.sum().item())
        elif spec_info.nxt_kv_lens_cpu is not None:
            self.nxt_kv_lens_cpu = spec_info.nxt_kv_lens_cpu
            self.nxt_kv_lens_sum = spec_info.nxt_kv_lens_sum

        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = torch.cat(
                [self.future_indices, spec_info.future_indices]
            )
            return

        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p], dim=0)
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index], dim=0)
        self.bonus_tokens = torch.cat(
            [self.bonus_tokens, spec_info.bonus_tokens], dim=0
        )
        self.new_seq_lens = torch.cat(
            [self.new_seq_lens, spec_info.new_seq_lens], dim=0
        )
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], dim=0
        )
