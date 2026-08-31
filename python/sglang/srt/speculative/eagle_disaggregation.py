"""EAGLE 分布式解码（Disaggregation）模块。

本模块提供在分布式（PD 分离）场景下构建 EAGLE 草稿输入的工具函数。
在 PD 分离架构中，prefill 和 decode 在不同节点执行，
需要通过 future_map 传递隐藏状态和 topk 信息。

主要函数：
    - build_eagle_disagg_draft_input: 构建分布式 EAGLE 草稿输入
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.eagle_info import EagleDraftInput

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


def build_eagle_disagg_draft_input(
    batch: ScheduleBatch,
    server_args: ServerArgs,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> EagleDraftInput:
    """在分布式（PD 分离）场景下构建 EAGLE 草稿模型输入。

    从各请求的 output_topk_p / output_topk_index / hidden_states_tensor
    中收集数据，组装为 EagleDraftInput。当启用 overlap 时，
    通过 future_map 发布和暂存 spec_info 以供异步消费。

    参数：
        batch:               调度批次
        server_args:         服务器参数（含 eagle_topk / num_steps 配置）
        last_tokens_tensor:  最后 token 张量 [bs]，作为 bonus_tokens
        future_map:          异步 future 映射（用于 overlap 传递 spec_info）

    返回：EagleDraftInput 实例，含 topk_p / topk_index / hidden_states / bonus_tokens

    示例：
        >>> spec_info = build_eagle_disagg_draft_input(
        ...     batch, server_args, last_tokens, future_map)
        >>> spec_info.topk_p.shape       # [bs, num_states]
        >>> spec_info.topk_index.shape   # [bs, num_states]
        >>> spec_info.hidden_states.shape # [bs, hidden_dim]
        >>> spec_info.bonus_tokens.shape  # [bs]
    """
    num_states = server_args.speculative_eagle_topk
    if server_args.enable_multi_layer_eagle:
        num_states *= server_args.speculative_num_steps

    topk_p = torch.stack(
        [
            torch.as_tensor(
                req.output_topk_p[:num_states],
                device=batch.device,
                dtype=torch.float32,
            )
            for req in batch.reqs
        ],
        dim=0,
    )
    topk_index = torch.stack(
        [
            torch.as_tensor(
                req.output_topk_index[:num_states],
                device=batch.device,
                dtype=torch.int64,
            )
            for req in batch.reqs
        ],
        dim=0,
    )

    hidden_states = torch.stack(
        [req.hidden_states_tensor for req in batch.reqs], dim=0
    ).to(batch.device)

    spec_info = EagleDraftInput(
        topk_p=topk_p,
        topk_index=topk_index,
        hidden_states=hidden_states,
        bonus_tokens=last_tokens_tensor,
    )
    spec_info.capture_hidden_mode = CaptureHiddenMode.LAST

    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(spec_info.future_indices, spec_info)

    return spec_info
