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
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker


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

    # 形状信息，用于 padding（如 DP attention / CUDA graph）。
    num_tokens_per_batch: int = -1

    def __post_init__(self):
        """初始化后处理：设置 spec_input_type 并默认 num_tokens_per_batch。

        示例：
            >>> input = DFlashVerifyInput(draft_token=t, positions=p, draft_token_num=4)
            >>> # input.spec_input_type == SpecInputType.DFLASH_VERIFY
            >>> # input.num_tokens_per_batch == 4
        """
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.draft_token_num)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        """返回投机 token 调整系数 (分子, 分母)。

        DFLASH 中两者都等于 draft_token_num，表示不做额外调整。

        返回：(draft_token_num, draft_token_num)
        """
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        target_worker: TpModelWorker,
    ) -> tuple[ForwardBatch, bool]:
        """为 DFLASH 验证前向准备 ScheduleBatch 和 ForwardBatch。

        调用方在此之前已计算并存储 `batch.out_cache_loc`。
        本方法仅打包验证前向数据，并预初始化 CUDA graph replay
        元数据或 eager attention 元数据，使实际前向可以以
        `skip_attn_backend_init=True` 运行。

        参数：
            batch:         当前调度 batch
            target_worker: 目标模型 worker

        返回：(verify_forward_batch, can_run_cuda_graph)
            - verify_forward_batch: 构建好的 ForwardBatch
            - can_run_cuda_graph:   是否可使用 CUDA graph replay

        示例：
            >>> fwd_batch, use_cg = verify_input.prepare_for_verify(batch, target_worker)
            >>> if use_cg:
            ...     target_worker.model_runner.decode_cuda_graph_runner.replay(fwd_batch)
            ... else:
            ...     target_worker.model_runner.forward(fwd_batch)
        """
        batch.input_ids = self.draft_token
        batch.spec_info = self
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = self.capture_hidden_mode
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        can_run_cuda_graph = bool(
            target_worker.model_runner.decode_cuda_graph_runner
            and target_worker.model_runner.decode_cuda_graph_runner.can_run(
                verify_forward_batch
            )
        )
        if can_run_cuda_graph:
            target_worker.model_runner.decode_cuda_graph_runner.replay_prepare(
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

        # 构建 query 偏移指针：每请求 draft_token_num 个 query
        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )

        # 累积 KV 序列长度：原始长度 + 草稿 token 数
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        # 使用 Triton kernel 生成 KV cache 索引
        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * bs,
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
