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
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_hip

_is_hip = is_hip()


@dataclass
class ExpertLocationDispatchInfo:
    """单个 MoE 层执行“逻辑专家 ID → 物理专家 ID”转换所需的信息。

    Router 选择的是模型定义中的逻辑专家；EPLB 可能移动或复制这些专家，
    因而执行 MoE kernel 前还需要把逻辑 ID 转换为当前布局中的物理 ID。

    Attributes:
        ep_dispatch_algorithm: 专家副本分发算法。``static`` 使用预先确定的映射；
            ``dynamic``/``fake`` 在某个逻辑专家的全部物理副本中随机选择。
        partial_logical_to_rank_dispatch_physical_map: 当前层的静态映射，形状为
            ``(num_logical_experts,)``，每个元素是该逻辑专家应使用的物理专家 ID。
        partial_logical_to_all_physical_map: 当前层的全部副本映射，形状为
            ``(num_logical_experts, max_num_replicas)``。每行保存一个逻辑专家
            可用的所有物理专家 ID；无效的尾部槽位不会被访问。
        partial_logical_to_all_physical_map_num_valid: 每个逻辑专家实际拥有的
            物理副本数量，形状为 ``(num_logical_experts,)``。
        num_physical_experts: 当前 EPLB 布局中的物理专家总数。
    """

    ep_dispatch_algorithm: Literal["static", "random"]
    # 形状：(逻辑专家数量,)
    partial_logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
    # 形状：(逻辑专家数量, 最大副本数量)
    partial_logical_to_all_physical_map: torch.Tensor
    # 形状：(逻辑专家数量,)
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    num_physical_experts: int

    @classmethod
    def init_new(cls, layer_id: int):
        """从全局 EPLB 元数据中提取指定 MoE 层的分发信息。

        Args:
            layer_id: MoE 层在全局专家布局元数据中的层索引。

        Returns:
            若未启用专家分发算法，则返回 ``None``；否则返回仅包含指定层
            映射切片的 :class:`ExpertLocationDispatchInfo`。

        输入输出示例：
            假设 ``layer_id=2``，全局静态映射的形状为 ``(8, 64)``，则输入
            ``2`` 后，返回对象中的静态映射形状为 ``(64,)``，表示第 2 层
            64 个逻辑专家各自对应的物理专家 ID。
        """
        # 分发算法由服务启动参数统一配置，布局映射由 EPLB 全局元数据维护。
        ep_dispatch_algorithm = get_global_server_args().ep_dispatch_algorithm
        expert_location_metadata = get_global_expert_location_metadata()
        assert expert_location_metadata is not None

        # 未配置分发算法时不需要执行逻辑专家到物理专家的转换。
        if ep_dispatch_algorithm is None:
            return None

        # 全局映射的第 0 维是层，这里只保留当前层，避免调用方重复传 layer_id。
        return cls(
            ep_dispatch_algorithm=ep_dispatch_algorithm,
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
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
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
    if _is_hip:
        # HIP 后端的索引结果类型可能与调用方期望不同，显式对齐输入 ID 类型。
        physical_topk_ids = physical_topk_ids.to(topk_ids.dtype)
    return physical_topk_ids


def _topk_ids_logical_to_physical_dynamic(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    """为每个逻辑专家随机选择一个有效的物理副本。

    每个 Top-K 槽位都会独立采样，因此即使多个 token 选中了同一个逻辑专家，
    它们也可能被分发到不同的物理副本。采样只在该逻辑专家的有效副本范围内
    进行，不会访问映射表每行末尾的无效填充槽位。

    Args:
        topk_ids: 逻辑专家 ID 张量，通常形状为 ``(num_tokens, top_k)``。
        info: 包含“逻辑专家 → 全部物理副本”映射及有效副本数的信息。

    Returns:
        与 ``topk_ids`` 形状相同的物理专家 ID 张量。

    输入输出示例：
        若 ``topk_ids = tensor([[0, 1]])``，逻辑专家 0 的物理副本是
        ``[2, 5]``，逻辑专家 1 的物理副本是 ``[3, 7]``，则一次可能输出
        ``tensor([[5, 3]])``；另一次也可能输出 ``tensor([[2, 7]])``。
    """
    # 先记录形状，展平后可用一对一的二维高级索引完成所有副本查找。
    topk_ids_original_shape = topk_ids.shape
    device = topk_ids.device
    topk_ids = topk_ids.flatten()

    # 先为每个逻辑专家生成随机整数，再对其有效副本数取模，将副本下标约束在
    # [0, num_valid_replicas) 内。65536 提供固定且足够大的随机取值范围。
    chosen_dispatch_index = (
        torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
        % info.partial_logical_to_all_physical_map_num_valid[topk_ids]
    )
    # 第一维按逻辑专家 ID 选行，第二维按随机副本下标选取物理专家 ID。
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]
    if _is_hip:
        # HIP 后端的索引结果类型可能与调用方期望不同，显式保持 ID 类型一致。
        topk_ids = topk_ids.to(topk_ids.dtype)

    # 恢复调用方传入的 Top-K ID 布局。
    topk_ids = topk_ids.view(topk_ids_original_shape)
    return topk_ids
