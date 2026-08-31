# Copyright 2023-2025 SGLang Team
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

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.runtime_context import get_exec


@dataclass
class ExpertLocationDispatchInfo:
    ep_dispatch_algorithm: Literal["static", "dynamic", "fake", "lp"]
    # (num_logical_experts,)
    partial_logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
    # 形状：(逻辑专家数量, 最大副本数量)
    partial_logical_to_all_physical_map: torch.Tensor
    # 形状：(逻辑专家数量,)
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    num_physical_experts: int
    # Whether every rank must pick the same physical expert for a token. True
    # without an a2a backend: all EP ranks then run the MoE over the same tokens
    # and sum their partial outputs, so a rank-dependent pick counts a replicated
    # logical expert several times. With one, each rank dispatches only its own
    # tokens and is free to disagree.
    rank_invariant: bool = False

    @classmethod
    def init_new(cls, layer_id: int):
        ep_dispatch_algorithm = get_exec().moe.ep_dispatch_algorithm
        expert_location_metadata = get_global_expert_location_metadata()
        assert expert_location_metadata is not None

        # 未配置分发算法时不需要执行逻辑专家到物理专家的转换。
        if ep_dispatch_algorithm is None:
            return None

        # 全局映射的第 0 维是层，这里只保留当前层，避免调用方重复传 layer_id。
        return cls(
            ep_dispatch_algorithm=ep_dispatch_algorithm,
            rank_invariant=get_exec().moe.moe_a2a_backend == "none",
            partial_logical_to_rank_dispatch_physical_map=(
                expert_location_metadata.logical_to_rank_dispatch_physical_map[
                    layer_id, :
                ]
                if expert_location_metadata.logical_to_rank_dispatch_physical_map
                is not None
                else None
            ),
            partial_logical_to_all_physical_map=expert_location_metadata.logical_to_all_physical_map[
                layer_id, :
            ],
            partial_logical_to_all_physical_map_num_valid=expert_location_metadata.logical_to_all_physical_map_num_valid[
                layer_id, :
            ],
            num_physical_experts=expert_location_metadata.num_physical_experts,
        )


def transform_select_experts_inputs(
    router_logits: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    info: Optional[ExpertLocationDispatchInfo],
):
    """按专家分发算法调整 Top-K 选择阶段的输入。

    正常算法保持输入不变；``fake`` 算法用区间 ``[5, 10)`` 内的均匀随机数
    原地覆盖路由 logits，并清零校正偏置，从而构造与模型路由结果无关的随机
    专家流量。该模式主要用于验证和压测专家分发流程。

    Args:
        router_logits: Router 为每个 token 计算的专家 logits，通常形状为
            ``(num_tokens, num_logical_experts)``。``fake`` 模式会原地修改它。
        correction_bias: 可选的专家校正偏置，通常形状为
            ``(num_logical_experts,)``。
        info: 当前层的专家位置分发信息；为 ``None`` 时不做任何转换。

    Returns:
        二元组 ``(router_logits, correction_bias)``。正常模式返回原输入；
        ``fake`` 模式返回随机 logits 和全零校正偏置。

    输入输出示例：
        输入 ``router_logits.shape == (2, 4)``、``correction_bias.shape == (4,)``
        且算法为 ``fake``，输出 logits 的形状仍为 ``(2, 4)``、所有值位于
        ``[5, 10)``，输出 bias 为 ``tensor([0., 0., 0., 0.])``。
    """
    if (info is not None) and (info.ep_dispatch_algorithm == "fake"):
        # 原地随机化 logits，避免额外分配一个与 token 数量相关的大张量。
        router_logits.uniform_(5, 10)
        if correction_bias is not None:
            # fake 模式不应再受到模型校正偏置影响。
            correction_bias = torch.zeros_like(correction_bias)
    return router_logits, correction_bias


def topk_ids_logical_to_physical(
    topk_ids: torch.Tensor,
    info: Optional[ExpertLocationDispatchInfo],
    log2phy_prob: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """将 Router 选出的逻辑专家 ID 转换成实际执行的物理专家 ID。

    该函数是不同分发算法的统一入口。它不改变 Top-K 分数，也不重新选择
    逻辑专家，只决定每个已选逻辑专家最终落到哪个物理专家槽位。

    Args:
        topk_ids: Top-K 选择产生的逻辑专家 ID，通常形状为
            ``(num_tokens, top_k)``。
        info: 当前层的专家位置分发信息。为 ``None`` 表示逻辑布局与物理布局
            一致，无需转换。

    Returns:
        与 ``topk_ids`` 形状相同的物理专家 ID 张量。未启用 EPLB 分发时直接
        返回输入张量本身。

    输入输出示例：
        若 ``topk_ids = tensor([[0, 2], [1, 0]])``，静态映射为
        ``tensor([5, 3, 7])``，则输出为
        ``tensor([[5, 7], [3, 5]])``。
    """
    # 没有 EPLB 分发信息时，逻辑专家 ID 可以直接作为物理专家 ID 使用。
    if info is None:
        return topk_ids

    # static：同一个逻辑专家始终映射到预先指定的物理专家。
    if info.ep_dispatch_algorithm == "static":
        return _topk_ids_logical_to_physical_static(topk_ids, info)
    # dynamic/fake：同一个逻辑专家可在其多个物理副本之间随机分流。
    if info.ep_dispatch_algorithm in ["dynamic", "fake"]:
        return _topk_ids_logical_to_physical_dynamic(topk_ids, info)
    if info.ep_dispatch_algorithm == "lp":
        if log2phy_prob is None:
            raise RuntimeError(
                "ep_dispatch_algorithm='lp' but log2phy_prob is None at dispatch "
                f"time (topk_ids.shape={tuple(topk_ids.shape)})."
            )
        return _topk_ids_logical_to_physical_probability(topk_ids, info, log2phy_prob)
    raise NotImplementedError(f"Unknown algorithm {info.ep_dispatch_algorithm}")


def _topk_ids_logical_to_physical_static(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    """使用固定查找表把逻辑专家 ID 映射为物理专家 ID。

    Args:
        topk_ids: 任意形状的逻辑专家 ID 张量。
        info: 包含当前层静态映射表的专家分发信息。

    Returns:
        与 ``topk_ids`` 形状相同的物理专家 ID 张量。

    输入输出示例：
        输入 ``topk_ids = tensor([2, 0])``，映射表为 ``tensor([4, 8, 6])``，
        输出 ``tensor([6, 4])``。
    """
    # PyTorch 高级索引会逐元素查表，并自然保留 topk_ids 的原始形状。
    physical_topk_ids = info.partial_logical_to_rank_dispatch_physical_map[topk_ids]
    if physical_topk_ids.dtype != topk_ids.dtype:
        physical_topk_ids = physical_topk_ids.to(topk_ids.dtype)
    return physical_topk_ids


def _topk_ids_logical_to_physical_dynamic(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    """Spread each (token, logical expert) over that logical expert's replicas.

    Under ``rank_invariant`` the replica is picked by token row rather than at
    random: ``torch.randint`` reads the default CUDA generator, so agreement
    across ranks would rest on their philox offsets staying aligned, which
    nothing asserts, and it would make greedy requests non-reproducible. Both
    pick evenly, which is the load split EPLB's placement solver assumes.

    Row indexing leaves replicas unused for a single-row batch, where there is no
    imbalance to fix anyway.
    """
    topk_ids_original_shape = topk_ids.shape
    original_dtype = topk_ids.dtype
    device = topk_ids.device
    topk_ids = topk_ids.flatten()

    num_valid = info.partial_logical_to_all_physical_map_num_valid[topk_ids]
    if info.rank_invariant:
        slots_per_token = (
            topk_ids_original_shape[-1] if len(topk_ids_original_shape) > 1 else 1
        )
        row_index = (
            torch.arange(topk_ids.shape[0], dtype=num_valid.dtype, device=device)
            // slots_per_token
        )
        chosen_dispatch_index = row_index % num_valid
    else:
        chosen_dispatch_index = (
            torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
            % num_valid
        )
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]
    if topk_ids.dtype != original_dtype:
        topk_ids = topk_ids.to(original_dtype)

    # 恢复调用方传入的 Top-K ID 布局。
    topk_ids = topk_ids.view(topk_ids_original_shape)
    return topk_ids


def _topk_ids_logical_to_physical_probability(
    topk_ids: torch.Tensor,
    info: ExpertLocationDispatchInfo,
    log2phy_prob: torch.Tensor,
) -> torch.Tensor:
    """Select physical experts via the JIT-compiled CUDA dispatch kernel.

    Raises if ``topk_ids`` isn't on CUDA — the LP path requires the fused
    kernel and there is no torch reference fallback at runtime.
    """
    if not topk_ids.is_cuda:
        raise RuntimeError(
            "LP dispatch requires CUDA tensors; got topk_ids on " f"{topk_ids.device}."
        )
    from sglang.kernels.ops.lplb import cuda_solver

    return cuda_solver.dispatch_probability(
        topk_ids, log2phy_prob, info.partial_logical_to_all_physical_map
    )
