# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Frozen-KV MTP 投机解码工具函数模块。

本模块提供 Frozen-KV MTP 算法的上下文管理、位置设置、
topk 扩展等工具函数，被 frozen_kv_mtp_worker_v2.py 调用。

主要功能：
    - KV 池切换：frozen_kv_target_view / target_kv_pool_view
    - 位置设置：set_frozen_kv_positions / position_for_batch
    - topk 扩展：expand_for_topk_draft
    - 隐藏状态选取：select_last_extend_hidden
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPContext

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


@contextmanager
def frozen_kv_target_view(
    forward_batch: ForwardBatch,
    kv_context: FrozenKVMTPContext,
    draft_attn_backend: AttentionBackend,
):
    """构建基于已提交目标前缀几何的注意力元数据（上下文管理器）。

    将 ``draft_attn_backend.token_to_kv_pool`` 临时替换为冻结的目标 KV 池，
    使元数据初始化期间任何读取 ``get_token_to_kv_pool()`` 的辅助函数
    都能看到冻结目标池。通过单个 backend 属性交换，
    ``get_token_to_kv_pool()`` 和 backend 自身的 ``self.token_to_kv_pool``
    均能看到交换后的池。

    参数：
        forward_batch:       前向批次
        kv_context:          Frozen-KV MTP 上下文
        draft_attn_backend:  草稿模型注意力后端

    Yields：无（上下文管理器，在 with 块内执行目标池视图）

    异常：RuntimeError - 当 kv_context 为 None（未绑定模型）时

    示例：
        >>> with frozen_kv_target_view(batch, ctx, attn_backend):
        ...     # 在此块内，注意力后端使用目标 KV 池
        ...     attn_backend.forward(...)
    """
    if kv_context is None:
        raise RuntimeError(
            "Frozen-KV MTP target view called before the model was bound; "
            "bind the frozen KV context first."
        )
    saved_spec_info = forward_batch.spec_info
    forward_batch.spec_info = None
    saved_backend_pool = draft_attn_backend.token_to_kv_pool
    draft_attn_backend.token_to_kv_pool = kv_context.target_token_to_kv_pool
    try:
        yield
    finally:
        forward_batch.spec_info = saved_spec_info
        draft_attn_backend.token_to_kv_pool = saved_backend_pool


@contextmanager
def target_kv_pool_view(
    forward_batch: ForwardBatch,
    kv_context: FrozenKVMTPContext,
    draft_attn_backend: AttentionBackend,
):
    """使用目标冻结 KV 池运行草稿模型前向（上下文管理器）。

    将 ``draft_attn_backend.token_to_kv_pool`` 临时替换为冻结的目标 KV 池。
    通过单个 backend 属性交换，``get_token_to_kv_pool()``
    和 backend 自身的 ``self.token_to_kv_pool`` 读取均能看到交换后的池。

    参数：
        forward_batch:       前向批次
        kv_context:          Frozen-KV MTP 上下文
        draft_attn_backend:  草稿模型注意力后端

    Yields：无（上下文管理器）

    异常：RuntimeError - 当 kv_context 为 None 时

    示例：
        >>> with target_kv_pool_view(batch, ctx, attn_backend):
        ...     # 草稿模型使用目标 KV 池进行前向
        ...     draft_model.forward(...)
    """
    if kv_context is None:
        raise RuntimeError(
            "Frozen-KV MTP target KV pool view called before the model was bound; "
            "bind the frozen KV context first."
        )
    saved_backend_pool = draft_attn_backend.token_to_kv_pool
    draft_attn_backend.token_to_kv_pool = kv_context.target_token_to_kv_pool
    try:
        yield
    finally:
        draft_attn_backend.token_to_kv_pool = saved_backend_pool


def set_frozen_kv_positions(forward_batch: ForwardBatch, topk: int) -> None:
    """设置 Frozen-KV MTP 的 RoPE 位置。

    RoPE 阶段使用最后写入的目标槽位（seq_len - 1），
    而非随草稿步推进。当 topk > 1 时，将位置 repeat_interleave 到 B*topk。

    参数：
        forward_batch: 前向批次
        topk:          每请求候选数

    示例：
        >>> # seq_lens = [5, 10], topk = 2
        >>> set_frozen_kv_positions(batch, topk=2)
        >>> # batch.positions = tensor([4, 4, 9, 9])
    """
    seq_lens = forward_batch.seq_lens
    positions = torch.clamp(seq_lens - 1, min=0).to(torch.int64)
    if (
        topk > 1
        and forward_batch.positions is not None
        and forward_batch.positions.numel() == positions.numel() * topk
    ):
        positions = positions.repeat_interleave(topk, dim=0)
    if forward_batch.positions is None:
        forward_batch.positions = positions
    else:
        if forward_batch.positions.shape == positions.shape:
            forward_batch.positions.copy_(positions)
        else:
            forward_batch.positions = positions


def expand_for_topk_draft(forward_batch: ForwardBatch, topk: int) -> None:
    """将已提交前缀的元数据扩展到活跃的 B * topk 前沿。

    当 topk > 1 时，将 batch_size、req_pool_indices、seq_lens 等
    repeat_interleave topk 倍，使每个请求产生 topk 个候选分支。

    参数：
        forward_batch: 前向批次（必须未扩展，即 batch_size == len(seq_lens)）
        topk:          每请求候选数

    异常：RuntimeError - 当 batch 已扩展（batch_size != len(seq_lens)）时

    示例：
        >>> # batch_size=2, topk=3
        >>> expand_for_topk_draft(batch, topk=3)
        >>> # batch.batch_size = 6, seq_lens repeat_interleave(3)
    """
    if topk == 1 or forward_batch.batch_size == 0:
        return

    if forward_batch.batch_size != forward_batch.seq_lens.shape[0]:
        raise RuntimeError(
            "Frozen-KV MTP topk expansion expects an unexpanded forward "
            "batch where batch_size == len(seq_lens)."
        )

    forward_batch.batch_size *= topk
    forward_batch.req_pool_indices = forward_batch.req_pool_indices.repeat_interleave(
        topk, dim=0
    )
    forward_batch.seq_lens = forward_batch.seq_lens.repeat_interleave(topk, dim=0)
    if forward_batch.seq_lens_cpu is not None:
        forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu.repeat_interleave(
            topk, dim=0
        )
        forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
    else:
        forward_batch.seq_lens_sum = torch.sum(forward_batch.seq_lens).item()

    positions = torch.clamp(forward_batch.seq_lens - 1, min=0).to(torch.int64)
    forward_batch.positions = positions
    forward_batch.num_token_non_padded_cpu = positions.numel()
    if forward_batch.num_token_non_padded is not None:
        forward_batch.num_token_non_padded.fill_(positions.numel())
    if (
        forward_batch.mrope_positions is not None
        and forward_batch.mrope_positions.shape[-1] * topk == positions.numel()
    ):
        forward_batch.mrope_positions = forward_batch.mrope_positions.repeat_interleave(
            topk, dim=-1
        )


def position_for_batch(batch: ScheduleBatch) -> torch.Tensor:
    """计算批次的 RoPE 位置（seq_len - 1，clamp 到 0）。

    参数：
        batch: 调度批次

    返回：int64 位置张量 [bs]

    示例：
        >>> # batch.seq_lens = [5, 10, 1]
        >>> position_for_batch(batch)
        tensor([4, 9, 0])
    """
    return torch.clamp(batch.seq_lens - 1, min=0).to(torch.int64)


def select_last_extend_hidden(
    batch: ScheduleBatch, hidden_states: torch.Tensor
) -> torch.Tensor:
    """从 extend 隐藏状态中选取每个请求的最后一个 token 的隐藏状态。

    参数：
        batch:         调度批次
        hidden_states: extend 前向的隐藏状态 [total_extend_tokens, hidden_dim]

    返回：每个请求最后 token 的隐藏状态 [bs, hidden_dim]

    示例：
        >>> # extend_lens = [3, 2], hidden_states shape = [5, 4096]
        >>> # last_indices = cumsum([3,2]) - 1 = [2, 4]
        >>> select_last_extend_hidden(batch, hidden_states).shape
        torch.Size([2, 4096])
    """
    if hidden_states.shape[0] == batch.batch_size():
        return hidden_states
    lens = torch.tensor(batch.extend_lens, device=hidden_states.device)
    last_indices = torch.cumsum(lens, dim=0) - 1
    return hidden_states[last_indices.to(torch.long)]
