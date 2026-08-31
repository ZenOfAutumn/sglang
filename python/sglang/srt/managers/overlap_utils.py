"""Overlap-scheduling helpers for relaying cross-iteration values.

中译：overlap（计算/CPU 处理重叠）调度的辅助工具。
      overlap 调度的核心思想是：在第 N 轮 forward 还在 GPU 上跑时，调度器就提前为第 N+1 轮
      做 CPU 端准备（取下一批请求、组 batch 等）。但「下一批要喂给模型的 input_ids」此刻
      还没算出来——它正是第 N 轮采样的结果。于是用占位 + 延迟解析的方式：
        - 第 N 轮 forward 结束后，把采样得到的 token（及投机解码的额外信息）写入 FutureMap
          中「按 req_pool_index 索引」的常驻缓冲区（publish / stash）。
        - 第 N+1 轮 forward 真正进入时，再从 FutureMap 把这些值取出、填回 batch
          （resolve_forward_inputs / resolve_seq_lens_cpu）。
      FutureMap 因此扮演「跨迭代值的中继站」角色，使 CPU 准备与 GPU 计算得以重叠。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import msgspec
import torch

from sglang.kernels.ops.speculative.gather_spec_extras import gather_spec_extras
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_exec,
    get_spec,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.ngram_info import NgramVerifyInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def decide_needs_cpu_seq_lens(
    attn_backends: Sequence[AttentionBackend],
) -> bool:
    """Whether FutureMap must publish seq_lens_cpu / sum.

    OR over per-backend needs_cpu_seq_lens; force True under TBO (it reads the
    CPU mirror outside the backend layer to split the batch) or ngram (its
    USE_FULL_MASK verify path reads the host mirror regardless of backend).
    """
    # Local import: keep overlap_utils' module-level deps leaf-only so it stays
    # importable everywhere; spec_info pulls in the spec/schedule_batch graph.
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if get_exec().overlap.enable_two_batch_overlap:
        # FIXME: support TBO without seq lens cpu value
        # 中译：TBO 目前必须有 CPU 端 seq_lens，故强制返回 True（待支持无 CPU 值的 TBO）。
        return True
        # 中译：piecewise CUDA Graph（PCG）同理，暂时强制需要 CPU 端 seq_lens。
    algo = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
    if algo.is_ngram():
        # ngram's USE_FULL_MASK verify path reads seq_lens_cpu per req to size
        # the tree mask, regardless of the attn backend (e.g. Triton opts out).
        return True
    # Skip unset slots (e.g. draft_extend_attn_backend on some spec configs);
    # missing flag -> True so undeclared backends stay on the legacy path.
    # 中译：跳过未设置的后端槽位（如某些投机配置下的 draft_extend_attn_backend）；
    #       缺失标志默认按 True 处理，使「未声明」的后端保持在旧（带 CPU 镜像）路径上。
    return any(
        getattr(b, "needs_cpu_seq_lens", True) for b in attn_backends if b is not None
    )


def decide_needs_confidence_relay() -> bool:
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
    if not algo.is_dspark():
        return False
    return read_ragged_verify_mode() is not RaggedVerifyMode.STATIC


_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

# Token-buf consume tracking: init to -1, assert non-negative on gather,
# write -1 back. Catches "gather without intermediate stash" bugs. CI enables
# via the existing SGLANG_IS_IN_CI; off in production.
# 中译：token 缓冲区的「消费」追踪开关（仅 CI 调试用，生产关闭）。
#       缓冲区初始填 -1，每次 gather（取值）时断言取出的值非负，取完写回 -1。
#       这样若某处「未先 stash 就 gather」（即读到了已失效/未写入的槽位），就会读到 -1 被断言捕获。
_DEBUG_ASSERT = envs.SGLANG_IS_IN_CI.get()


@torch.compile(dynamic=True, disable=_is_npu)
def _assert_nonneg_and_invalidate(
    values: torch.Tensor, buf: torch.Tensor, indices: torch.Tensor
) -> None:
    """Fused: assert all `values >= 0` and scatter -1 into `buf[indices]`.
    Compiled so the reduction + assert + scatter run as one kernel launch.

    中译：融合算子——断言取出的 values 全部 >= 0，并把 -1 散写回 buf[indices] 使其失效。
          用 torch.compile 编译，让「归约 + 断言 + 散写」合并为一次 kernel 启动以省开销。
    """
    torch._assert_async((values >= 0).all())
    buf[indices] = -1


def resolve_forward_inputs(batch: ScheduleBatch, future_map: FutureMap) -> None:
    """Materialize input_ids at forward entry. Two sources:

    - Prefill: H2D copy from pinned CPU staging (prefill_input_ids_cpu).
    - Decode/spec_v2: gather from FutureMap (last iter's sampled token).

    中译：在 forward 入口「兑现」本批的 input_ids。来源有两种：
      - Prefill：从 pinned（锁页）CPU 暂存区 prefill_input_ids_cpu 做 H2D（主机到设备）拷贝；
        prefill 的 token 在 CPU 准备阶段就已知，无需等上一轮采样。
      - Decode / spec_v2：从 FutureMap 取上一轮采样出的 token（此前由 publish/stash 写入）。
      混合批（mix）时两段拼接：prefill 段来自 CPU 暂存，decode 段来自 FutureMap 缓冲。
    """
    if batch.prefill_input_ids_cpu is not None:
        # 中译：prefill 段——把 CPU 暂存的 input_ids 异步拷到设备（non_blocking 配合 pinned 内存）。
        prefill_gpu = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        if batch.mix_running_indices is not None:
            # 中译：混合批——decode 段的 token 从 FutureMap 取（上一轮采样结果，按 req_pool_index 索引）。
            decode_gpu = future_map.output_tokens_buf[batch.mix_running_indices]
            if _DEBUG_ASSERT:
                _assert_nonneg_and_invalidate(
                    decode_gpu,
                    future_map.output_tokens_buf,
                    batch.mix_running_indices,
                )
            # 中译：prefill 段在前、decode 段在后拼成本批完整 input_ids。
            batch.input_ids = torch.cat([prefill_gpu, decode_gpu])
        else:
            batch.input_ids = prefill_gpu
        # 中译：用完即清空，避免下一轮误用上一轮的暂存。
        batch.prefill_input_ids_cpu = None
        batch.mix_running_indices = None
    elif batch.input_ids is None and future_map.spec_algo.is_none():
        # 中译：纯 decode 且非投机解码——直接按 req_pool_indices 从 FutureMap 取上一轮采样 token。
        batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                batch.input_ids, future_map.output_tokens_buf, batch.req_pool_indices
            )

    # Only the overlap path relays spec extras through the future_map; the
    # synchronous (non-overlap) V2 path installs next_draft_input directly.
    # 中译：只有 overlap 路径才通过 future_map 中继投机解码的额外信息（topk、隐藏态、bonus token 等）；
    #       同步（非 overlap）的 V2 路径会直接安装 next_draft_input，无需中继。
    if batch.enable_overlap and not batch.spec_algorithm.is_none():
        future_map._resolve_spec_extras(batch)


CONFIDENCE_RELAY_RING_LAG: int = 2
CONFIDENCE_RELAY_RING_DEPTH: int = CONFIDENCE_RELAY_RING_LAG + 1


class ResolvedConfidence(msgspec.Struct):

    confidence: torch.Tensor
    generation: torch.Tensor


@dataclass
class RelayPayload:
    """Per-iteration stash payload for the FutureMap bufs. Non-spec fills only
    `bonus_tokens`; which spec extras get relayed is decided by
    `FutureMap.spec_algo`, not by this payload's shape."""

    bonus_tokens: Optional[torch.Tensor]
    topk_p: Optional[torch.Tensor] = None
    topk_index: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    draft_probs: Optional[torch.Tensor] = None
    dsa_topk_indices: Optional[torch.Tensor] = None
    # ngram delays the draft extend (ngram update)
    accept_tokens: Optional[torch.Tensor] = None
    accept_lens: Optional[torch.Tensor] = None

    @classmethod
    def from_ngram(cls, draft_input: NgramVerifyInput) -> RelayPayload:
        return cls(
            bonus_tokens=None,
            accept_tokens=draft_input.accept_tokens.reshape(
                -1, draft_input.draft_token_num
            ),
            accept_lens=draft_input.accept_lens,
        )

    @classmethod
    def from_draft_input(cls, draft_input: EagleDraftInput) -> RelayPayload:
        return cls(
            bonus_tokens=draft_input.bonus_tokens,
            topk_p=draft_input.topk_p,
            topk_index=draft_input.topk_index,
            hidden_states=draft_input.hidden_states,
            draft_probs=getattr(draft_input, "draft_probs", None),
            dsa_topk_indices=draft_input.dsa_topk_indices,
        )


class ConfidenceRelay(msgspec.Struct):

    device: torch.device
    req_pool_size: int
    pool: Any
    confidence_buf: Optional[torch.Tensor] = None
    conf_ring: Optional[torch.Tensor] = None
    gen_ring: Optional[torch.Tensor] = None
    copy_done: Optional[list] = None
    ring_pos: int = 0
    initialized: bool = False

    def _lazy_init(self, confidence: torch.Tensor) -> None:
        self.initialized = True
        gamma = confidence.shape[-1]
        self.confidence_buf = torch.empty(
            (self.req_pool_size, gamma), dtype=torch.float32, device=self.device
        )
        if _is_cuda:
            depth = CONFIDENCE_RELAY_RING_DEPTH
            self.conf_ring = torch.empty(
                (depth, self.req_pool_size, gamma),
                dtype=torch.float32,
                pin_memory=True,
            )
            self.gen_ring = torch.zeros((depth, self.req_pool_size), dtype=torch.int64)
            self.copy_done = [
                torch.get_device_module(self.device).Event() for _ in range(depth)
            ]

    def scatter(self, indices: torch.Tensor, confidence: torch.Tensor) -> None:
        if not self.initialized:
            self._lazy_init(confidence)
        self.confidence_buf[indices] = confidence.to(self.confidence_buf.dtype)

    def issue_ring_copy(self, *, stream, publish_ready) -> None:
        if not self.initialized or stream is None or publish_ready is None:
            return
        slot = self.ring_pos % CONFIDENCE_RELAY_RING_DEPTH
        stream.wait_event(publish_ready)
        with torch.get_device_module(self.device).stream(stream):
            self.conf_ring[slot].copy_(self.confidence_buf, non_blocking=True)
            self.copy_done[slot].record()
        self.gen_ring[slot].copy_(self.pool.req_generation)
        self.ring_pos += 1

    def resolve(
        self, batch: ScheduleBatch, *, stream, publish_ready
    ) -> Optional[ResolvedConfidence]:
        if not self.initialized:
            return None
        draft_input = batch.spec_info
        if draft_input is None:
            return None
        fi = draft_input.future_indices
        if fi is None or fi.shape[0] == 0:
            return None

        if stream is None or publish_ready is None:
            idx = batch.req_pool_indices
            idx_cpu = batch.req_pool_indices_cpu
            return ResolvedConfidence(
                confidence=self.confidence_buf[idx].cpu(),
                generation=self.pool.req_generation[idx_cpu].clone(),
            )

        if self.ring_pos < CONFIDENCE_RELAY_RING_LAG:
            return None
        slot = (self.ring_pos - CONFIDENCE_RELAY_RING_LAG) % CONFIDENCE_RELAY_RING_DEPTH
        if not self.copy_done[slot].query():
            return None

        idx_cpu = batch.req_pool_indices_cpu
        return ResolvedConfidence(
            confidence=self.conf_ring[slot][idx_cpu],
            generation=self.gen_ring[slot][idx_cpu],
        )


class FutureMap:
    """Always-on pool-indexed relay for cross-iter values. Forward writes via
    publish/stash; next iter reads via resolve_forward_inputs / resolve_seq_lens_cpu.

    中译：跨迭代值的「常驻中继站」，所有缓冲区都以 req_pool_index 为下标（请求在 req-to-token
          池中的槽位），故称 pool-indexed。
          写入侧：本轮 forward 结束后通过 publish（写 seq_lens）/ stash（写采样 token 及投机额外信息）。
          读取侧：下一轮 forward 进入时通过 resolve_forward_inputs / resolve_seq_lens_cpu 取回。
          槽位 0 对应 KV 的 padding 行，因此 CUDA Graph 的填充批（req_pool_idx == 0）读到的是无害值。
    """

    def __init__(
        self,
        device: torch.device,
        spec_algo: SpeculativeAlgorithm,
        req_to_token_pool: ReqToTokenPool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ):
        # Bufs indexed by req_pool_idx; slot 0 mirrors KV padding row so
        # CUDA-graph padded batches (req_pool_idx == 0) are harmless.
        # 中译：所有缓冲区均以 req_pool_idx 为下标；槽位 0 对应 KV padding 行，
        #       使 CUDA Graph 填充批（req_pool_idx == 0）的读写都落在无害位置。
        self.device = device
        self.spec_algo = spec_algo
        # Computed by decide_needs_cpu_seq_lens(); see that helper for the
        # full decision (per-backend flag + TBO / piecewise CG overrides).
        # 中译：是否需要维护 seq_lens 的 CPU 镜像，由 decide_needs_cpu_seq_lens() 决定（见上）。
        self.needs_cpu_seq_lens = needs_cpu_seq_lens
        # 中译：缓冲区长度 = req-to-token 池的容量（最大并发请求数）。
        # 中译：output_tokens_buf 存放每个请求上一轮采样出的 token id（decode 下一轮的 input）。
        # 中译：new_seq_lens_buf 存放每个请求更新后的序列长度（GPU 端）。
        self.needs_confidence_relay = needs_confidence_relay
        self.req_pool_size = req_to_token_pool.req_to_token.shape[0]

        if _DEBUG_ASSERT:
            # Poisoned init: every row must be written before its first gather.
            self.output_tokens_buf = torch.full(
                (self.req_pool_size,), -1, dtype=torch.int64, device=self.device
            )
            self.new_seq_lens_buf = torch.full(
        else:
            self.output_tokens_buf = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            self.new_seq_lens_buf = torch.empty(
        # Pinned host copy of new_seq_lens_buf + private stream for fwd-prepare
        # D2H pulls (gated only on publish, off the schedule stream). CUDA-only:
        # recovers occupancy lost to the WAR barrier (also CUDA-only); other
        # platforms have no barrier and use the plain .cpu() bootstrap path.
        # 中译：CUDA 上额外建一份 pinned（锁页）主机镜像 + 一条专用 stream，用于 forward 准备阶段
        #       的 D2H（设备到主机）拷贝。该拷贝只以 publish 事件为门控、不挂在调度 stream 上，
        #       从而恢复因 WAR（write-after-read）屏障损失的并发度（屏障也仅 CUDA 才有）。
        #       其他平台没有该屏障，走普通 .cpu() 的 bootstrap 路径即可。
        if _is_cuda:
            self.new_seq_lens_cpu_pinned = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, pin_memory=True
            )
            self.fwd_prepare_d2h_stream = torch.get_device_module(self.device).Stream()
        else:
            self.new_seq_lens_cpu_pinned = None
            self.fwd_prepare_d2h_stream = None
            # 中译：投机解码相关缓冲区延迟初始化（首次 stash 时按 draft_input 的真实形状创建）。
        # 中译：延迟创建的设备事件，标记 publish 已完成；仅 spec_v2 需要它来给 D2H 拷贝做门控。
        # 中译：首次拿到真实 draft_input 时，按其各字段的 dtype/shape 延迟创建投机解码缓冲区。
        # 中译：need_verified_id —— 是否需要中继 verified_id（即目标模型额外吐出的 bonus token）。
        # 中译：need_bonus_tokens —— 是否需要中继 bonus_tokens（验证之外恒定多吐的那一个 token）。
        # 中译：need_topk —— 是否需要中继 draft 的 topk 概率/索引（EAGLE 等草稿采样所需）。
        # 中译：need_hidden_states —— 是否需要中继隐藏态（部分算法用上一轮隐藏态作为下一轮草稿输入）。
        self.need_topk = False
        self.need_hidden_states = False
        self.topk_p_buf = None
        self.topk_index_buf = None
        self.hidden_states_buf = None
        self.draft_probs_buf = None
        self.dsa_topk_indices_buf = None

        # ngram-only relay bufs
        self.accept_tokens_buf: Optional[torch.Tensor] = None
        self.accept_lens_buf: Optional[torch.Tensor] = None
        self.publish_ready = None  # lazy device.Event(); only spec_v2 needs it
        # Debug consume-once state: armed by a recording publish, consumed by
        # resolve; arm/consume strictly alternate across all batch interleavings.
        self._publish_fresh = False
        self.confidence_relay = ConfidenceRelay(
            device=self.device,
            req_pool_size=self.req_pool_size,
            pool=req_to_token_pool,
        )

    def _maybe_init_forward_bufs(self, payload: RelayPayload) -> None:
        # Local import (see decide_needs_cpu_seq_lens): keep module-level deps leaf.
        from sglang.srt.speculative.spec_utils import spec_need_hidden_states

        # Prefill can omit spec extras; initialize each buffer when decode first
        # carries it instead of fixing the layout from the first payload.
        if not self.need_topk and (
            self.spec_algo.is_some()
            and self.spec_algo.need_topk()
            and payload.topk_p is not None
        ):
            self.need_topk = True
            topk_p0 = payload.topk_p[0]
            topk_index0 = payload.topk_index[0]
            self.topk_p_buf = torch.empty(
                (self.req_pool_size, *topk_p0.shape),
                dtype=topk_p0.dtype,
                device=self.device,
            )
            self.topk_index_buf = torch.empty(
                (self.req_pool_size, *topk_index0.shape),
                dtype=topk_index0.dtype,
                device=self.device,
            )

        if not self.need_hidden_states and (
            self.spec_algo.is_some()
            and spec_need_hidden_states()
            and payload.hidden_states is not None
        ):
            self.need_hidden_states = True
            hidden_states0 = payload.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.req_pool_size, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

        if self.draft_probs_buf is None and payload.draft_probs is not None:
            draft_probs0 = payload.draft_probs[0]
            self.draft_probs_buf = torch.empty(
                (self.req_pool_size, *draft_probs0.shape),
                dtype=draft_probs0.dtype,
                device=self.device,
            )

    def _maybe_init_dsa_topk_indices_buf(self, payload: RelayPayload) -> None:
        if self.dsa_topk_indices_buf is not None or payload.dsa_topk_indices is None:
            return
        seed0 = payload.dsa_topk_indices[0]
        self.dsa_topk_indices_buf = torch.empty(
            (self.req_pool_size, *seed0.shape),
            dtype=payload.dsa_topk_indices.dtype,
            device=self.device,
        )

    def _maybe_init_ngram_bufs(self, payload: RelayPayload) -> None:
        if self.accept_tokens_buf is not None:
            return
        # zeros, not empty: an unstashed row resolves to accept_len 0 (empty
        # splice at draft prep) instead of a garbage length.
        self.accept_tokens_buf = torch.zeros(
            (self.req_pool_size, payload.accept_tokens.shape[1]),
            dtype=payload.accept_tokens.dtype,
            device=self.device,
        )
        self.accept_lens_buf = torch.zeros(
            (self.req_pool_size,),
            dtype=payload.accept_lens.dtype,
            device=self.device,
        )

    def resolve_confidence_cpu(
        self, batch: ScheduleBatch
    ) -> Optional[ResolvedConfidence]:
        if not self.needs_confidence_relay:
            return None
        return self.confidence_relay.resolve(
            batch,
            stream=self.fwd_prepare_d2h_stream,
            publish_ready=self.publish_ready,
        )

    def _resolve_spec_extras(self, batch: ScheduleBatch) -> None:
        # 中译：读取侧——从各投机缓冲区按 future_indices 取回上一轮 stash 的投机额外信息，
        #       填回本批的 draft_input（verified_id / bonus_tokens / topk / hidden_states）。
        if self.spec_algo.is_ngram():
            # 中译：ngram 算法的草稿是预先算好的，无需中继，直接返回（待支持预计算草稿后移除）。
            draft_input = batch.spec_info
            if draft_input is None or draft_input.future_indices is None:
                return
            indices = draft_input.future_indices
            if indices.shape[0] == 0:
            draft_input.accept_tokens = self.accept_tokens_buf[indices].flatten()
            draft_input.accept_lens = self.accept_lens_buf[indices]
            return
        draft_input: EagleDraftInput = batch.spec_info
        if draft_input is None:
            # FIXME(lsyin): only prefill; not compatible with mixed mode
            # 中译：draft_input 为空通常意味着纯 prefill；当前与混合模式不兼容，直接返回。
            return
            # 中译：DFLASH 在「直接携带」有效时，额外信息已随批传递，无需再从缓冲区取。
        indices = draft_input.future_indices
        if indices.shape[0] == 0:
            return
        # FIXME: indices = batch.req_pool_indices, pinned 2 iters via
        # record_batch_in_overlap; record_stream here is redundant.
        # 中译：indices 实际等于 batch.req_pool_indices，已通过 record_batch_in_overlap 钉住两轮，
        #       此处的 record_stream 是冗余的（待统一后移除）。
        indices.record_stream(torch.get_device_module(self.device).current_stream())
            # 中译：取回 bonus token（代码中字段名为 verified_id）。
        if self.need_topk:
            # 中译：需要 topk 时一次性 gather 出 topk_p/topk_index/bonus_tokens/hidden_states
            #       （融合在 gather_spec_extras kernel 中以减少启动开销）。
            hidden_states_buf = (
                self.hidden_states_buf if self.need_hidden_states else None
            )
            (
                draft_input.topk_p,
                draft_input.topk_index,
                bonus_tokens,
                hidden_states,
            ) = gather_spec_extras(
                indices,
                self.topk_p_buf,
                self.topk_index_buf,
                self.output_tokens_buf,
                hidden_states_buf,
            )
            draft_input.bonus_tokens = bonus_tokens
            if hidden_states is not None:
                draft_input.hidden_states = hidden_states
            if self.draft_probs_buf is not None and draft_input.draft_probs is not None:
                draft_input.draft_probs = self.draft_probs_buf[indices]
        else:
            draft_input.bonus_tokens = self.output_tokens_buf[indices]
        if self.need_hidden_states and not self.need_topk:
            draft_input.hidden_states = self.hidden_states_buf[indices]
        if draft_input.future_dsa_topk_indices_available:
            assert self.dsa_topk_indices_buf is not None
            draft_input.dsa_topk_indices = self.dsa_topk_indices_buf[indices]
        else:
            draft_input.dsa_topk_indices = None
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                draft_input.bonus_tokens, self.output_tokens_buf, indices
            )

    def resolve_seq_lens_cpu(self, batch: ScheduleBatch) -> None:
        # Lazy pull from new_seq_lens_buf for spec_v2 (accept_lens not known to
        # 中译：读取侧——为 spec_v2 从 new_seq_lens_buf 延迟取回更新后的序列长度（接受长度
        # schedule). The CPU mirror is gated by needs_cpu_seq_lens; backends that
        # opt out take the GPU-only path below. A private D2H stream overlaps the copy.
        draft_input = batch.spec_info
        if draft_input is None:
            return

        fi = draft_input.future_indices
        if fi is None:
            return
        if self.publish_ready is not None:
            # 中译：等 publish（写 new_seq_lens_buf）完成后再 gather，确保读到的是本轮的新长度。
            if _DEBUG_ASSERT:
                # Consume-once: every event wait must be re-armed by a fresh
                # forward publish; a stale consume means a publish went missing.
                assert self._publish_fresh, "resolve without a fresh forward publish"
                self._publish_fresh = False
            if _is_hip:
                # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
                # 中译：临时绕过——在 AMD MI355 上 Event.wait() 会拖慢 TPOT，改用 synchronize()。
                self.publish_ready.synchronize()
            else:
                self.publish_ready.wait()
        batch.seq_lens = self.new_seq_lens_buf[fi]

            # 中译：DFLASH 把 seq_lens_cpu 当作「滞后的已提交主机视图」，
        if not self.needs_cpu_seq_lens:
            # GPU gather above is kept (SB.seq_lens must advance each verify);
            # skip the .cpu() D2H. Downstream takes the GPU-only path.
            # 中译：无需 CPU 镜像时——上面的 GPU gather 保留（每次 verify 后 seq_lens 必须推进），
            #       但跳过 .cpu() 的 D2H 拷贝；下游走纯 GPU 路径。
            batch.seq_lens_cpu = None
            batch.seq_lens_sum = None
            if _DEBUG_ASSERT:
                # Poison consumed rows: each row must be re-published/seeded
                # before the next resolve gathers it (safe here: the forward's
                # re-publish is fenced behind this stream via wait_stream).
                _assert_nonneg_and_invalidate(batch.seq_lens, self.new_seq_lens_buf, fi)
            return

        if self.fwd_prepare_d2h_stream is None or self.publish_ready is None:
            batch.seq_lens_cpu = batch.seq_lens.cpu()  # bootstrap / non-CUDA
            # 中译：bootstrap 阶段或非 CUDA 平台——直接同步 .cpu()，并求和得到 seq_lens_sum。
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            if _DEBUG_ASSERT:
                _assert_nonneg_and_invalidate(batch.seq_lens, self.new_seq_lens_buf, fi)
            return

        # Mechanism: don't sync the schedule stream; gate a private stream on the
        # publish event and copy into the static pinned buffer.
        # 中译：机制——不阻塞调度 stream；让专用 stream 等 publish 事件后，把数据拷进固定的 pinned 缓冲。
        self.fwd_prepare_d2h_stream.wait_event(self.publish_ready)
        with torch.get_device_module(self.device).stream(self.fwd_prepare_d2h_stream):
            self.new_seq_lens_cpu_pinned.copy_(self.new_seq_lens_buf, non_blocking=True)
        self.fwd_prepare_d2h_stream.synchronize()

        # FIXME: fi == batch.req_pool_indices; unify future_indices and req_pool_indices.
        batch.seq_lens_cpu = self.new_seq_lens_cpu_pinned[batch.req_pool_indices_cpu]
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        if _DEBUG_ASSERT:
            # After the D2H copy completed (synchronize above), so the pinned
            # mirror is not poisoned.
            _assert_nonneg_and_invalidate(batch.seq_lens, self.new_seq_lens_buf, fi)

        # 中译：写入侧（之一）——把本轮更新后的序列长度写入 new_seq_lens_buf 对应槽位。
    def publish(
        self,
        future_indices: torch.Tensor,
        new_seq_lens: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> None:
        indices = future_indices
        if indices.shape[0] == 0:
            return  # DP idle
            # 中译：DP（数据并行）空闲：本 rank 这一轮没有请求，无需写入。
        self.new_seq_lens_buf[indices] = new_seq_lens.to(self.new_seq_lens_buf.dtype)
        publish_confidence = self.needs_confidence_relay and confidence is not None
        if publish_confidence:
            self.confidence_relay.scatter(indices, confidence)
        # Only spec_v2 needs the event; it gates the seq_lens D2H on the private stream.
        # 中译：只有 spec_v2 需要记录事件，用于给专用 stream 上的 seq_lens D2H 拷贝做门控。
        if self.spec_algo.is_some():
            if self.publish_ready is None:
                self.publish_ready = torch.get_device_module(self.device).Event()
            self.publish_ready.record()
            self._publish_fresh = True
        if publish_confidence:
            self.confidence_relay.issue_ring_copy(
                stream=self.fwd_prepare_d2h_stream,
                publish_ready=self.publish_ready,
            )

        # 中译：写入侧（之二）——把本轮采样结果暂存到缓冲区，供下一轮读取。
            # 中译：ngram 草稿预计算，无需暂存。
    def stash(self, future_indices: torch.Tensor, payload: RelayPayload) -> None:
        indices = future_indices
        if indices.shape[0] == 0:
            # DP idle: payload is empty stub; lazy-init shape peek would IndexError.
            # 中译：DP 空闲时 payload 是空占位，若继续做延迟初始化的形状探测会 IndexError，故提前返回。
            return
        # 中译：按 payload 的「类型」分派，而非按 spec_algo：非投机 decode 传入的是 token 张量。
            # 中译：纯 token 张量——直接写入 output_tokens_buf 作为下一轮的 input_ids。
            # 中译：首次进入投机分支时，按 draft_input 的真实形状延迟初始化各缓冲区。
        if self.spec_algo.is_ngram():
            self._maybe_init_ngram_bufs(payload)
            self.accept_tokens_buf[indices] = payload.accept_tokens
            self.accept_lens_buf[indices] = payload.accept_lens
            return
        self._maybe_init_forward_bufs(payload)
        self._maybe_init_dsa_topk_indices_buf(payload)
        self.output_tokens_buf[indices] = payload.bonus_tokens.to(
            self.output_tokens_buf.dtype
        )

        if self.need_topk:
            self.topk_p_buf[indices] = payload.topk_p.to(self.topk_p_buf.dtype)
            self.topk_index_buf[indices] = payload.topk_index.to(
                self.topk_index_buf.dtype
            )
        if self.need_hidden_states:
            self.hidden_states_buf[indices] = payload.hidden_states.to(
                self.hidden_states_buf.dtype
            )
        if self.draft_probs_buf is not None and payload.draft_probs is not None:
            self.draft_probs_buf[indices] = payload.draft_probs
        if (
            self.dsa_topk_indices_buf is not None
            and payload.dsa_topk_indices is not None
        ):
            self.dsa_topk_indices_buf[indices] = payload.dsa_topk_indices.to(
                self.dsa_topk_indices_buf.dtype
            )
