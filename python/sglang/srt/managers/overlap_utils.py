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

from typing import TYPE_CHECKING, Sequence, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.srt.speculative.triton_ops.gather_spec_extras import gather_spec_extras
from sglang.srt.utils import is_cuda, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def decide_needs_cpu_seq_lens(
    server_args: ServerArgs,
    attn_backends: Sequence[AttentionBackend],
) -> bool:
    """Whether FutureMap must publish seq_lens_cpu / sum.

    中译：判断 FutureMap 是否必须额外维护 seq_lens 的 CPU 镜像（seq_lens_cpu 及其求和）。
          决策方式：对各注意力后端的 needs_cpu_seq_lens 标志取「或」；并在
          TBO（two-batch-overlap）/ piecewise CUDA Graph 下强制 True——因为它们会在后端层之外
          读取这份 CPU 镜像。
    """
    if server_args.enable_two_batch_overlap:
        # FIXME: support TBO without seq lens cpu value
        # 中译：TBO 目前必须有 CPU 端 seq_lens，故强制返回 True（待支持无 CPU 值的 TBO）。
        return True
    cuda_graph_config = server_args.cuda_graph_config
    if (
        cuda_graph_config is not None
        and cuda_graph_config.prefill.backend == Backend.TC_PIECEWISE
    ):
        # FIXME: support PCG without seq lens cpu value
        # 中译：piecewise CUDA Graph（PCG）同理，暂时强制需要 CPU 端 seq_lens。
        return True
    # Skip unset slots (e.g. draft_extend_attn_backend on some spec configs);
    # missing flag -> True so undeclared backends stay on the legacy path.
    # 中译：跳过未设置的后端槽位（如某些投机配置下的 draft_extend_attn_backend）；
    #       缺失标志默认按 True 处理，使「未声明」的后端保持在旧（带 CPU 镜像）路径上。
    return any(
        getattr(b, "needs_cpu_seq_lens", True) for b in attn_backends if b is not None
    )


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
        self.req_pool_size = req_to_token_pool.req_to_token.shape[0]

        # 中译：output_tokens_buf 存放每个请求上一轮采样出的 token id（decode 下一轮的 input）。
        #       调试模式下初始填 -1 以便捕获脏读，生产用 empty（不初始化省开销）。
        self.output_tokens_buf = (
            torch.full((self.req_pool_size,), -1, dtype=torch.int64, device=self.device)
            if _DEBUG_ASSERT
            else torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            )
        )
        # 中译：new_seq_lens_buf 存放每个请求更新后的序列长度（GPU 端）。
        self.new_seq_lens_buf = torch.empty(
            (self.req_pool_size,), dtype=torch.int64, device=self.device
        )
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
        if self.spec_algo.is_some():
            # 中译：投机解码相关缓冲区延迟初始化（首次 stash 时按 draft_input 的真实形状创建）。
            self._forward_buf_initialized = False

        self.publish_ready = None  # lazy device.Event(); only spec_v2 needs it
        # 中译：延迟创建的设备事件，标记 publish 已完成；仅 spec_v2 需要它来给 D2H 拷贝做门控。

    def _lazy_init_forward_buf(self, draft_input: EagleDraftInput):
        # 中译：首次拿到真实 draft_input 时，按其各字段的 dtype/shape 延迟创建投机解码缓冲区。
        #       哪些字段需要中继由下面几个 need_* 标志决定（不同投机算法所需信息不同）。
        self._forward_buf_initialized = True

        # 中译：need_verified_id —— 是否需要中继 verified_id（即目标模型额外吐出的 bonus token）。
        self.need_verified_id = getattr(draft_input, "verified_id", None) is not None
        # 中译：need_bonus_tokens —— 是否需要中继 bonus_tokens（验证之外恒定多吐的那一个 token）。
        self.need_bonus_tokens = getattr(draft_input, "bonus_tokens", None) is not None
        # 中译：need_topk —— 是否需要中继 draft 的 topk 概率/索引（EAGLE 等草稿采样所需）。
        self.need_topk = self.spec_algo.need_topk()
        # 中译：need_hidden_states —— 是否需要中继隐藏态（部分算法用上一轮隐藏态作为下一轮草稿输入）。
        self.need_hidden_states = (
            spec_need_hidden_states()
            and getattr(draft_input, "hidden_states", None) is not None
        )

        if self.need_verified_id:
            verified_id0 = draft_input.verified_id[0]
            self.verified_id_buf = (
                torch.full(
                    (self.req_pool_size, *verified_id0.shape),
                    -1,
                    dtype=verified_id0.dtype,
                    device=self.device,
                )
                if _DEBUG_ASSERT
                else torch.empty(
                    (self.req_pool_size, *verified_id0.shape),
                    dtype=verified_id0.dtype,
                    device=self.device,
                )
            )
        if self.need_topk:
            topk_p0 = draft_input.topk_p[0]
            topk_index0 = draft_input.topk_index[0]
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
        if self.need_hidden_states:
            hidden_states0 = draft_input.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.req_pool_size, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

    def _resolve_spec_extras(self, batch: ScheduleBatch) -> None:
        # 中译：读取侧——从各投机缓冲区按 future_indices 取回上一轮 stash 的投机额外信息，
        #       填回本批的 draft_input（verified_id / bonus_tokens / topk / hidden_states）。
        if self.spec_algo.is_ngram():
            # FIXME: remove once precomputed draft is supported.
            # 中译：ngram 算法的草稿是预先算好的，无需中继，直接返回（待支持预计算草稿后移除）。
            return
        draft_input: EagleDraftInput = batch.spec_info
        if draft_input is None:
            # FIXME(lsyin): only prefill; not compatible with mixed mode
            # 中译：draft_input 为空通常意味着纯 prefill；当前与混合模式不兼容，直接返回。
            return
        if self.spec_algo.is_dflash() and getattr(
            draft_input, "direct_carry_valid", False
        ):
            # 中译：DFLASH 在「直接携带」有效时，额外信息已随批传递，无需再从缓冲区取。
            return
        indices = draft_input.future_indices
        if indices.shape[0] == 0:
            return
        # FIXME: indices = batch.req_pool_indices, pinned 2 iters via
        # record_batch_in_overlap; record_stream here is redundant.
        # 中译：indices 实际等于 batch.req_pool_indices，已通过 record_batch_in_overlap 钉住两轮，
        #       此处的 record_stream 是冗余的（待统一后移除）。
        indices.record_stream(torch.get_device_module(self.device).current_stream())
        if self.need_verified_id:
            # 中译：取回 bonus token（代码中字段名为 verified_id）。
            draft_input.verified_id = self.verified_id_buf[indices]
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
            if self.need_bonus_tokens:
                draft_input.bonus_tokens = bonus_tokens
            if hidden_states is not None:
                draft_input.hidden_states = hidden_states
        elif self.need_bonus_tokens:
            draft_input.bonus_tokens = self.output_tokens_buf[indices]
        if self.need_hidden_states and not self.need_topk:
            draft_input.hidden_states = self.hidden_states_buf[indices]
        if _DEBUG_ASSERT:
            if self.need_verified_id:
                _assert_nonneg_and_invalidate(
                    draft_input.verified_id, self.verified_id_buf, indices
                )
            if self.need_bonus_tokens:
                _assert_nonneg_and_invalidate(
                    draft_input.bonus_tokens, self.output_tokens_buf, indices
                )

    def resolve_seq_lens_cpu(self, batch: ScheduleBatch) -> None:
        # Lazy pull from new_seq_lens_buf for spec_v2 (accept_lens not known to
        # schedule). DFLASH intentionally keeps host-side lengths lagging and
        # uses its carried KV allocation watermark for planning, so only the GPU
        # seq_lens is resolved there. Other spec-v2 algorithms still need the CPU
        # mirror for host planning; use a private D2H stream for those copies.
        # 中译：读取侧——为 spec_v2 从 new_seq_lens_buf 延迟取回更新后的序列长度（接受长度
        #       accept_lens 在 CPU 调度时还不知道，故必须延迟到这里）。
        #       DFLASH 故意让主机端长度滞后、用其携带的 KV 分配水位线来做规划，所以只解析 GPU 端 seq_lens；
        #       其他 spec_v2 算法仍需 CPU 镜像做主机侧规划，那些拷贝走专用 D2H stream。
        draft_input = batch.spec_info
        if draft_input is None:
            return
        if self.spec_algo.is_dflash() and getattr(
            draft_input, "direct_carry_valid", False
        ):
            batch.seq_lens = draft_input.new_seq_lens
            return

        fi = draft_input.future_indices
        if fi is None:
            return
        if self.publish_ready is not None:
            # 中译：等 publish（写 new_seq_lens_buf）完成后再 gather，确保读到的是本轮的新长度。
            if _is_hip:
                # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
                # 中译：临时绕过——在 AMD MI355 上 Event.wait() 会拖慢 TPOT，改用 synchronize()。
                self.publish_ready.synchronize()
            else:
                self.publish_ready.wait()
        batch.seq_lens = self.new_seq_lens_buf[fi]

        if self.spec_algo.is_dflash():
            # DFLASH keeps seq_lens_cpu as the lagging committed host view;
            # planning/reserved host lengths live on DFlashDraftInputV2.
            # 中译：DFLASH 把 seq_lens_cpu 当作「滞后的已提交主机视图」，
            #       真正用于规划/预留的主机长度存在 DFlashDraftInputV2 上，这里不更新 CPU 端。
            return

        if not self.needs_cpu_seq_lens:
            # GPU gather above is kept (SB.seq_lens must advance each verify);
            # skip the .cpu() D2H. Downstream takes the GPU-only path.
            # 中译：无需 CPU 镜像时——上面的 GPU gather 保留（每次 verify 后 seq_lens 必须推进），
            #       但跳过 .cpu() 的 D2H 拷贝；下游走纯 GPU 路径。
            batch.seq_lens_cpu = None
            batch.seq_lens_sum = None
            return

        if self.fwd_prepare_d2h_stream is None or self.publish_ready is None:
            batch.seq_lens_cpu = batch.seq_lens.cpu()  # bootstrap / non-CUDA
            # 中译：bootstrap 阶段或非 CUDA 平台——直接同步 .cpu()，并求和得到 seq_lens_sum。
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
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

    def publish(self, future_indices: torch.Tensor, new_seq_lens: torch.Tensor) -> None:
        # 中译：写入侧（之一）——把本轮更新后的序列长度写入 new_seq_lens_buf 对应槽位。
        indices = future_indices
        if indices.shape[0] == 0:
            return  # DP idle
            # 中译：DP（数据并行）空闲：本 rank 这一轮没有请求，无需写入。
        self.new_seq_lens_buf[indices] = new_seq_lens.to(self.new_seq_lens_buf.dtype)
        # Only spec_v2 needs the event; it gates the seq_lens D2H on the private stream.
        # 中译：只有 spec_v2 需要记录事件，用于给专用 stream 上的 seq_lens D2H 拷贝做门控。
        if self.spec_algo.is_some():
            if self.publish_ready is None:
                self.publish_ready = torch.get_device_module(self.device).Event()
            self.publish_ready.record()

    def stash(
        self,
        future_indices: torch.Tensor,
        payload: Union[torch.Tensor, EagleDraftInput],
    ) -> None:
        # 中译：写入侧（之二）——把本轮采样结果暂存到缓冲区，供下一轮读取。
        #       payload 可能是纯 token 张量（非投机 decode），也可能是 EagleDraftInput（投机解码）。
        if self.spec_algo.is_ngram():
            # FIXME: remove once precomputed draft is supported.
            # 中译：ngram 草稿预计算，无需暂存。
            return
        indices = future_indices
        if indices.shape[0] == 0:
            # DP idle: payload is empty stub; lazy-init shape peek would IndexError.
            # 中译：DP 空闲时 payload 是空占位，若继续做延迟初始化的形状探测会 IndexError，故提前返回。
            return
        # Dispatch by payload type, not spec_algo: non-spec decode passes a
        # token Tensor here.
        # FIXME(lsyin): unify this relay path with a dataclass instead of the
        # Tensor / EagleDraftInput type switch.
        # 中译：按 payload 的「类型」分派，而非按 spec_algo：非投机 decode 传入的是 token 张量。
        #       （待用统一的 dataclass 替代「张量 / EagleDraftInput」的类型分支）
        if isinstance(payload, torch.Tensor):
            # 中译：纯 token 张量——直接写入 output_tokens_buf 作为下一轮的 input_ids。
            self.output_tokens_buf[indices] = payload.to(torch.int64)
            return

        draft_input: EagleDraftInput = payload
        if not self._forward_buf_initialized:
            # 中译：首次进入投机分支时，按 draft_input 的真实形状延迟初始化各缓冲区。
            self._lazy_init_forward_buf(draft_input)
        if self.need_verified_id:
            self.verified_id_buf[indices] = draft_input.verified_id.to(
                self.verified_id_buf.dtype
            )
        if self.need_bonus_tokens:
            self.output_tokens_buf[indices] = draft_input.bonus_tokens.to(
                self.output_tokens_buf.dtype
            )

        if self.need_topk:
            self.topk_p_buf[indices] = draft_input.topk_p.to(self.topk_p_buf.dtype)
            self.topk_index_buf[indices] = draft_input.topk_index.to(
                self.topk_index_buf.dtype
            )
        if self.need_hidden_states:
            self.hidden_states_buf[indices] = draft_input.hidden_states.to(
                self.hidden_states_buf.dtype
            )
