"""DFLASH 投机解码验证输入数据结构模块。

本模块定义 DFlashVerifyInput，用于 DFLASH 算法的目标模型验证阶段。
DFLASH 采用线性（非树形）草稿序列，验证时目标模型对整个 block 一次性前向，
返回所有 token 的 logits 以计算接受长度。

关键概念：
    - DFLASH verify 是线性的，topk 固定为 1（非树形）
    - 支持自定义 attention mask（custom_mask）用于约束验证时的注意力
    - 支持 CUDA graph replay 和 eager 两种执行路径
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

_is_npu = is_npu()


@dataclass
class DFlashVerifyInput(SpecInput):
    """DFLASH 目标模型验证前向的输入数据结构。

    验证前向以 `ForwardMode.TARGET_VERIFY` 模式运行，使目标模型
    返回 block 中所有 token 的 logits，从而计算接受长度。

    与 EAGLE 的树形验证不同，DFLASH 验证是线性的（非树形），
    topk 固定为 1。

    属性：
        draft_token:           草稿 token id [total_draft_tokens]
        positions:             每个 token 的位置 [total_draft_tokens]
        draft_token_num:       每请求的草稿 token 数
        topk:                  固定为 1（DFLASH 为线性验证）
        custom_mask:           自定义注意力允许掩码，True 表示允许 (q, k) 对
        capture_hidden_mode:   隐藏状态捕获模式
        num_tokens_per_batch:  每 batch token 数（用于 padding）

    示例：
        >>> # 3 个请求，每请求 4 个草稿 token
        >>> draft_token = torch.tensor([10, 20, 30, 40, 11, 21, 31, 41, 12, 22, 32, 42])
        >>> positions = torch.tensor([5, 6, 7, 8, 3, 4, 5, 6, 10, 11, 12, 13])
        >>> verify_input = DFlashVerifyInput(
        ...     draft_token=draft_token,
        ...     positions=positions,
        ...     draft_token_num=4,
        ... )
        >>> # verify_input.topk == 1
        >>> # verify_input.num_tokens_per_batch == 12
    """

    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    # 保留 topk 字段以兼容按 `topk > 1` 控制树元数据的注意力后端。
    # DFLASH 验证是线性的（非树形），所以始终为 1。
    topk: int = 1
    # TARGET_VERIFY 的自定义注意力「允许掩码」，供需要它的注意力后端使用。
    # 语义遵循 SGLang 投机解码约定：True 表示允许该 (q, k) 对。
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Shape info for padding (e.g., DP attention / CUDA graph).
    num_tokens_per_req: int = -1

    ragged_verify_layout: Optional[RaggedVerifyLayout] = None
    # Committed/live lengths before the verify caller temporarily expands
    # batch.seq_lens_cpu to the target-attention KV lengths.
    live_seq_lens_cpu: Optional[torch.Tensor] = None

    def __post_init__(self):
        """初始化后处理：设置 spec_input_type 并默认 num_tokens_per_batch。

        示例：
            >>> input = DFlashVerifyInput(draft_token=t, positions=p, draft_token_num=4)
            >>> # input.spec_input_type == SpecInputType.DFLASH_VERIFY
            >>> # input.num_tokens_per_batch == 4
        """
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        if self.num_tokens_per_req == -1:
            self.num_tokens_per_req = int(self.draft_token_num)
        self.num_tokens_for_logprob_per_req = int(self.draft_token_num)

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        target_worker: TpModelWorker,
    ) -> tuple[ForwardBatch, bool]:
        """为 DFLASH 验证前向准备 ScheduleBatch 和 ForwardBatch。

        The caller computes and stores `batch.out_cache_loc` before this
        method is called. GPU keeps the original pre-planning path. NPU leaves
        attention/graph metadata initialization to ModelRunner because DP/EP
        padding can still change the compressor's runtime shapes.
        """
        from sglang.srt.speculative.spec_utils import prepare_mamba_track_for_verify

        batch.input_ids = self.draft_token
        batch.spec_info = self
        if _is_npu and not batch.forward_mode.is_idle():
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                maybe_build_dsv4_verify_bundle,
            )

            batch.out_cache_loc_dsv4 = maybe_build_dsv4_verify_bundle(
                batch,
                self.draft_token_num,
                live_seq_lens_cpu=self.live_seq_lens_cpu,
            )

        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        if not batch.forward_mode.is_idle():
            # Rebuild mamba track indices (lazy: gather the positions planned
            # by mamba_lazy_spec_prepare) and clear the stale extend-time mask
            # before init_new snapshots them into the verify ForwardBatch.
            # Same hook eagle/ngram/dspark run before TARGET_VERIFY.
            prepare_mamba_track_for_verify(batch)
        verify_forward_batch = ForwardBatch.init_new(
            batch,
            target_worker.model_runner,
            capture_hidden_mode=self.capture_hidden_mode,
            return_hidden_states_before_norm=False,
        )

        can_run_cuda_graph = bool(
            target_worker.model_runner.decode_cuda_graph_runner
            and target_worker.model_runner.decode_cuda_graph_runner.can_run_graph(
                verify_forward_batch
            )
        )
        if _is_npu:
            # Do not pre-plan target verify on NPU. DP/EP padding can change
            # the compressor's logical batch without changing ForwardBatch's
            # stale-plan shape fields. Let ModelRunner select graph/eager and
            # initialize metadata after final batch preparation.
            return verify_forward_batch, can_run_cuda_graph
        elif can_run_cuda_graph:
            target_worker.model_runner.decode_cuda_graph_runner.load_batch(
                verify_forward_batch
            )
        elif not batch.forward_mode.is_idle():
            target_worker.model_runner.attn_backend.init_forward_metadata(
                verify_forward_batch
            )

        return verify_forward_batch, can_run_cuda_graph

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
        kv_start_idx: Optional[torch.Tensor] = None,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """为 prefill 阶段的注意力计算生成参数（kv_indices、indptr、mask 等）。

        参数：
            req_pool_indices:    请求池索引 [bs]
            paged_kernel_lens:   每 request 的 KV cache 长度 [bs]
            paged_kernel_lens_sum: KV 长度总和
            req_to_token:        req_to_token 映射表 [max_reqs, max_ctx_len]
            kv_start_idx:        KV 起始索引（可选）

        返回：(kv_indices, cum_kv_seq_len, qo_indptr, mask)
            - kv_indices:     KV cache 索引数组
            - cum_kv_seq_len: 累积 KV 序列长度 [bs+1]
            - qo_indptr:      query 偏移指针 [bs+1]
            - mask:           自定义注意力掩码（或 None）

        示例：
            >>> # bs=2, draft_token_num=4, paged_kernel_lens=[10, 8]
            >>> kv_indices, cum_len, qo_indptr, mask = verify_input.generate_attn_arg_prefill(
            ...     req_pool_indices, paged_kernel_lens, 18, req_to_token)
            >>> # qo_indptr = [0, 4, 8]
            >>> # cum_kv_seq_len = [0, 14, 26]  (10+4, 8+4)
        """
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        layout = self.ragged_verify_layout
        if layout is not None and layout.bs != bs:
            # Graph replay pads the batch to the captured slots; match it.
            layout = layout.padded_to_bucket(padded_bs=bs)

        if layout is None:
            qo_indptr = torch.arange(
                0,
                (bs + 1) * self.draft_token_num,
                step=self.draft_token_num,
                dtype=torch.int32,
                device=device,
            )
            verify_lens = self.draft_token_num
            kv_indices_extra = self.draft_token_num * bs
        else:
            qo_indptr = layout.qo_indptr_device
            verify_lens = layout.verify_lens
            kv_indices_extra = layout.total_verify_tokens

        # 累积 KV 序列长度：原始长度 + 草稿 token 数
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + verify_lens
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        if kv_indices_buf is not None:
            # Sync-free fast-plan path: write straight into the attention
            # backend's cuda-graph kv_indices buffer (the captured kernels read
            # it), skipping both the fresh allocation and the wrapper plan()'s
            # device-to-device refresh copy.
            kv_indices = kv_indices_buf
        else:
            kv_indices = torch.empty(
                paged_kernel_lens_sum + kv_indices_extra,
                dtype=torch.int32,
                device=device,
            )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            kv_start_idx,
            kv_indices,
            req_to_token.size(1),
        )
        # 处理自定义 mask 的 padding（CUDA graph 场景）
        mask = self.custom_mask
        if mask is not None:
            mask_numel = (
                paged_kernel_lens_sum * self.draft_token_num
                + (self.draft_token_num**2) * bs
            )
            if mask.numel() < mask_numel:
                # FIXME(attn): CUDA graph 下 custom mask padding 的临时修复
                mask = torch.cat(
                    [
                        mask,
                        torch.full(
                            (mask_numel - mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                self.custom_mask = mask
        return kv_indices, cum_kv_seq_len, qo_indptr, mask
