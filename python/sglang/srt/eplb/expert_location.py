# Copyright 2023-2024 SGLang Team
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

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

import torch
import torch.distributed
import torch.nn.functional as F

from sglang.srt.runtime_context import (
    get_device,
    get_exec,
    get_parallel,
    get_resources,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


def _prefer_same_node_experts() -> bool:
    from sglang.srt.elastic_ep.elastic_ep import elastic_expanded_world_enabled

    return (
        get_exec().moe.ep_join_mode != "scale" and not elastic_expanded_world_enabled()
    )


def _compute_elastic_expert_layout(
    base_num_physical_experts: int,
    initial_ep_size: int,
    effective_ep_size: int,
) -> tuple[int, int]:
    assert base_num_physical_experts % initial_ep_size == 0
    num_local_physical_experts = base_num_physical_experts // initial_ep_size
    return (
        num_local_physical_experts * effective_ep_size,
        num_local_physical_experts,
    )


@dataclass
class ExpertLocationMetadata:
    """专家位置元数据：维护「物理专家 <-> 逻辑专家」之间的双向映射。

    术语约定：
    - 逻辑专家（logical expert）：模型定义中的专家，数量固定为 num_logical_experts。
    - 物理专家（physical expert）：实际部署在各 GPU 上的副本，
      num_physical_experts = num_logical_experts + ep_num_redundant_experts。
      热点逻辑专家可以拥有多个物理副本以分担负载。

    各字段的形状说明（layers 表示 MoE 层数）：
    - physical_to_logical_map: (layers, num_physical_experts)，每个物理专家对应的逻辑专家 id（多对一）。
    - logical_to_all_physical_map: (layers, num_logical_experts, X)，每个逻辑专家对应的所有物理副本 id，
      不足处用 -1 padding；X 会被 pad 到 num_physical_experts。
    - logical_to_all_physical_map_num_valid: (layers, num_logical_experts)，上表每个逻辑专家的有效副本个数。
    - logical_to_rank_dispatch_physical_map: (layers, num_logical_experts)，当前 rank 上每个逻辑专家
      应分发到的具体物理专家 id（仅在 static 分发算法下生成，否则为 None）。

    数值示例（贯穿所有字段）：
      设 num_layers=1、num_logical_experts=4、num_physical_experts=6（即 2 个冗余专家），
      ep_size=2（每卡 3 个物理专家：GPU0 持有物理 0/1/2，GPU1 持有物理 3/4/5）。
      假设逻辑专家 0、1 为热点，各自拥有 2 个物理副本：

      - physical_to_logical_map:            (1, 6)
          [[0, 1, 2, 3, 0, 1]]
          # 物理 0->逻辑0, 1->逻辑1, 2->逻辑2, 3->逻辑3, 4->逻辑0, 5->逻辑1

      - logical_to_all_physical_map:        (1, 4, X)，X pad 到 6，空位填 -1
          [[[0, 4, -1, -1, -1, -1],   # 逻辑0 的物理副本: 0, 4
            [1, 5, -1, -1, -1, -1],   # 逻辑1 的物理副本: 1, 5
            [2, -1, -1, -1, -1, -1],  # 逻辑2 的物理副本: 2
            [3, -1, -1, -1, -1, -1]]] # 逻辑3 的物理副本: 3

      - logical_to_all_physical_map_num_valid: (1, 4)
          [[2, 2, 1, 1]]              # 逻辑0/1 各 2 个副本，逻辑2/3 各 1 个

      - logical_to_rank_dispatch_physical_map（假设当前为 GPU0，ep_rank=0）: (1, 4)
          [[0, 1, 2, 3]]
          # 逻辑0 就近选同卡的物理0（而非物理4）、逻辑1 选物理1；逻辑2/3 唯一副本 2/3
          # 若在 GPU1（ep_rank=1）上，则为 [[4, 5, 2, 3]]（逻辑0/1 就近选物理4/5）

      - physical_to_logical_map_cpu / logical_to_all_physical_map_cpu:
          分别是上面两张映射的 CPU 张量副本，数值相同。
    """

    # (layers, num_physical_experts)：物理专家 -> 逻辑专家 的映射
    physical_to_logical_map: torch.Tensor
    # 上者的 CPU 副本（避免每次调用触发 GPU->CPU 同步）
    physical_to_logical_map_cpu: torch.Tensor
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical_experts, X)
    logical_to_all_physical_map_cpu: torch.Tensor  # CPU copy for performance
    logical_to_all_physical_map_num_valid: torch.Tensor  # (layers, num_logical_experts)
    ep_size: int
    # (layers, num_logical_experts)
    logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]

    # -------------------------------- properties ------------------------------------

    @property
    def num_layers(self) -> int:
        return self.physical_to_logical_map.shape[0]

    @property
    def num_physical_experts(self) -> int:
        return self.physical_to_logical_map.shape[1]

    @property
    def num_local_physical_experts(self) -> int:
        # 单个 EP rank（单张 GPU）上分到的物理专家数量，要求可被 ep_size 整除
        ans, remainder = divmod(self.num_physical_experts, self.ep_size)
        assert remainder == 0
        return ans

    @property
    def num_logical_experts(self) -> int:
        return self.logical_to_all_physical_map.shape[1]

    def __post_init__(self):
        # 一致性校验：三张映射表在 layers / num_logical_experts / num_physical_experts 维度上必须彼此对齐
        num_layers_0, num_physical_experts_0 = self.physical_to_logical_map.shape
        num_layers_1, num_logical_experts_0, num_physical_experts_1 = (
            self.logical_to_all_physical_map.shape
        )
        num_layers_2, num_logical_experts_1 = (
            self.logical_to_all_physical_map_num_valid.shape
        )
        assert num_layers_0 == num_layers_1 == num_layers_2
        assert num_logical_experts_0 == num_logical_experts_1
        assert num_physical_experts_0 == num_physical_experts_1

    # -------------------------------- construction ------------------------------------

    @staticmethod
    def init_trivial(model_config: ModelConfig, moe_ep_rank: int):
        """Trivial location - logical expert i corresponds to physical expert i"""
        common = ExpertLocationMetadata._init_common(model_config)

        if common is None:
            return None

        num_physical_experts = common["num_physical_experts"]
        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_layers = model_config_for_expert_location.num_layers
        num_logical_experts = model_config_for_expert_location.num_logical_experts

        base_num_physical_experts = common["base_num_physical_experts"]
        physical_to_logical_map = (
            torch.arange(0, base_num_physical_experts).repeat(num_layers, 1)
            % num_logical_experts
        )
        physical_to_logical_map = append_trivial_expert_slots(
            physical_to_logical_map,
            num_physical_experts - base_num_physical_experts,
            num_logical_experts,
        )

        return ExpertLocationMetadata.init_by_mapping(
            model_config,
            physical_to_logical_map=physical_to_logical_map,
            moe_ep_rank=moe_ep_rank,
        )

    @staticmethod
    def init_by_mapping(
        model_config: ModelConfig,
        physical_to_logical_map,
        moe_ep_rank: int = None,
    ):
        """根据给定的「物理 -> 逻辑」映射直接构造元数据。

        输入：
        - physical_to_logical_map: (layers, num_physical_experts) 张量或可转为张量的嵌套列表。
        - moe_ep_rank: 当前 EP rank；非 None 时会尽量把逻辑专家映射到本地/本节点的物理副本。
        返回：ExpertLocationMetadata（模型无专家位置信息时返回 None）。

        示例：
            physical_to_logical_map = [[0, 1, 2, 3, 0, 1]]  # 1 层、6 个物理专家、4 个逻辑专家
            -> 内部据此反推出 logical_to_all_physical_map：
               逻辑 0 -> [0, 4]，逻辑 1 -> [1, 5]，逻辑 2 -> [2]，逻辑 3 -> [3]
        """
        # 统一转换为张量，并搬到目标设备
        if not isinstance(physical_to_logical_map, torch.Tensor):
            physical_to_logical_map = torch.tensor(physical_to_logical_map)
        physical_to_logical_map = physical_to_logical_map.to(get_device().device)

        common = ExpertLocationMetadata._init_common(model_config)

        if common is None:
            return None

        model_config_for_expert_location = common["model_config_for_expert_location"]
        if common["num_physical_experts"] > common["base_num_physical_experts"]:
            if physical_to_logical_map.shape[-1] == common["base_num_physical_experts"]:
                physical_to_logical_map = append_trivial_expert_slots(
                    physical_to_logical_map,
                    common["num_physical_experts"]
                    - common["base_num_physical_experts"],
                    model_config_for_expert_location.num_logical_experts,
                )
            assert physical_to_logical_map.shape[-1] == common["num_physical_experts"]
        logical_to_all_physical_map = _compute_logical_to_all_physical_map(
            physical_to_logical_map=physical_to_logical_map,
            num_logical_experts=model_config_for_expert_location.num_logical_experts,
            ep_size=common["ep_size"],
            moe_ep_rank=moe_ep_rank,
        )

        return ExpertLocationMetadata._init_raw(
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
            moe_ep_rank=moe_ep_rank,
        )

    @staticmethod
    def init_by_eplb(
        model_config: ModelConfig,
        logical_count: torch.Tensor,
        *,
        use_flat_topology: bool = False,
    ):
        """依据实测的逻辑专家负载计数，调用 EPLB 算法计算新的专家放置。

        这是运行时重平衡的核心入口（由 EPLBManager.rebalance 调用）。

        输入：
        - logical_count: 逻辑专家的负载计数，形状 (layers, num_logical_experts) 或
          (num_windows, layers, num_logical_experts)；二维时会自动补一个维度。
        返回：ExpertLocationMetadata，其中的映射由 eplb 算法根据负载重新分配（热点专家获得更多物理副本）。

        示例：
            logical_count = [[100, 5, 5, 5]]  # 逻辑专家 0 是热点
            经算法重平衡后，逻辑专家 0 可能获得多个物理副本以分摊负载。
        """
        # 统一为张量，二维输入补齐 window 维度，并搬到目标设备
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        if len(logical_count.shape) == 2:
            logical_count = logical_count.unsqueeze(0)
        logical_count = logical_count.to(get_device().device)

        common = ExpertLocationMetadata._init_common(model_config)

        if common is None:
            return None

        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_physical_experts = common["num_physical_experts"]
        num_groups = model_config_for_expert_location.num_groups
        num_nodes = 1 if use_flat_topology else get_parallel().nnodes

        from sglang.srt.eplb import eplb_algorithms

        # 调用 eplb 算法：输入各逻辑专家的负载，输出新的物理/逻辑映射
        physical_to_logical_map, logical_to_all_physical_map, expert_count = (
            eplb_algorithms.rebalance_experts(
                tokens_per_expert=logical_count,
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_physical_experts // common["ep_size"],
                num_groups=num_groups,
                num_nodes=num_nodes,
                algorithm=eplb_algorithms.compute_algorithm(
                    raw_algorithm=get_exec().moe.eplb_algorithm,
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                ),
            )
        )

        return ExpertLocationMetadata._init_raw(
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map.to(get_device().device),
            logical_to_all_physical_map=logical_to_all_physical_map.to(
                get_device().device
            ),
        )

    @staticmethod
    def _init_common(model_config: ModelConfig):

        model_config_for_expert_location = (
            ModelConfigForExpertLocation.from_model_config(model_config)
        )

        # 模型未提供专家位置信息（非 MoE 或未实现该接口），直接返回 None
        if model_config_for_expert_location is None:
            return None

        base_num_physical_experts = (
            model_config_for_expert_location.num_logical_experts
            + get_exec().moe.ep_num_redundant_experts
        )
        # elastic-EP scale-up rewrites ep_size on the published config
        ep_size = get_parallel().ep_size
        num_physical_experts = base_num_physical_experts
        initial_ep_size = get_parallel().elastic_ep_initial_size
        if initial_ep_size is not None:
            if get_exec().moe.ep_join_mode == "scale":
                ep_size = max(
                    ep_size,
                    get_parallel().ep_join_rank_offset + get_parallel().tp_size,
                )
            num_physical_experts, num_local_physical_experts = (
                _compute_elastic_expert_layout(
                    base_num_physical_experts,
                    initial_ep_size,
                    ep_size,
                )
            )
        else:
            assert num_physical_experts % ep_size == 0
            num_local_physical_experts = num_physical_experts // ep_size

        return dict(
            model_config_for_expert_location=model_config_for_expert_location,
            base_num_physical_experts=base_num_physical_experts,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            ep_size=ep_size,
        )

    @staticmethod
    def _init_raw(
        ep_size: int,
        physical_to_logical_map: torch.Tensor,
        logical_to_all_physical_map: torch.Tensor,
        moe_ep_rank: Optional[int] = None,
    ):

        _, num_physical_experts = physical_to_logical_map.shape

        # 把变长的副本维度 pad 成统一长度（num_physical_experts），空位填 -1
        logical_to_all_physical_map_padded = F.pad(
            logical_to_all_physical_map,
            (0, num_physical_experts - logical_to_all_physical_map.shape[-1]),
            value=-1,
        )

        # 统计每个逻辑专家非 -1（即有效）的物理副本个数
        logical_to_all_physical_map_num_valid = torch.count_nonzero(
            logical_to_all_physical_map != -1, dim=-1
        )

        return ExpertLocationMetadata(
            physical_to_logical_map=physical_to_logical_map,
            physical_to_logical_map_cpu=physical_to_logical_map.cpu(),
            logical_to_all_physical_map=logical_to_all_physical_map_padded,
            logical_to_all_physical_map_cpu=logical_to_all_physical_map_padded.cpu(),
            logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
            ep_size=ep_size,
            logical_to_rank_dispatch_physical_map=(
                compute_logical_to_rank_dispatch_physical_map(
                    logical_to_all_physical_map=logical_to_all_physical_map,
                    ep_size=ep_size,
                    num_physical_experts=num_physical_experts,
                    ep_rank=(
                        moe_ep_rank
                        if moe_ep_rank is not None
                        else torch.distributed.get_rank() % ep_size
                    ),
                )
                if get_exec().moe.ep_dispatch_algorithm == "static"
                else None
            ),
        )

    # -------------------------------- mutation ------------------------------------

    def update(
        self,
        other: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        """用 other 的映射局部更新本对象（仅更新 update_layer_ids 指定的层）。

        用于重平衡后切换到新的专家位置；支持按 chunk 分批更新以降低单次停顿。

        输入：
        - other: 新的元数据（形状必须与本对象一致，且 ep_size 相同）。
        - update_layer_ids: 需要更新的层 id 列表，例如 [0, 2]。
        效果：仅这些层的各映射张量被 other 对应层覆盖，其余层保持不变。
        """
        # ep_size 必须一致，否则两个元数据不可混用
        for field in [
            "ep_size",
        ]:
            assert getattr(self, field) == getattr(other, field)

        for field in [
            "physical_to_logical_map",
            "physical_to_logical_map_cpu",
            "logical_to_all_physical_map",
            "logical_to_all_physical_map_cpu",
            "logical_to_all_physical_map_num_valid",
            "logical_to_rank_dispatch_physical_map",
        ]:
            other_field = getattr(other, field)
            self_field = getattr(self, field)
            assert (other_field is not None) == (self_field is not None)
            if self_field is not None:
                # 构造按层的布尔掩码，并 reshape 为可广播形状（首维为层，其余维为 1）
                mask_update = torch.tensor(
                    [i in update_layer_ids for i in range(self.num_layers)]
                )
                mask_update = mask_update.view(*([-1] + [1] * (self_field.dim() - 1)))
                mask_update = mask_update.to(self_field.device, non_blocking=True)
                # 掩码为 True 的层取 other，否则保留自身
                self_field[...] = torch.where(mask_update, other_field, self_field)

    # -------------------------------- usage ------------------------------------

    def logical_to_all_physical(
        self,
        layer_id: int,
        logical_expert_id: int,
        require_global_experts: bool = False,
    ) -> List[int]:
        # Use CPU copy to avoid GPU→CPU sync on every call, which is expensive in update weights scenario
        cpu_map = self.logical_to_all_physical_map_cpu
        # Draft workers can query MoE layers whose layer_id lies beyond the
        # target-sized expert map; fall back to the identity mapping (no EPLB
        # rebalancing for those layers) instead of indexing out of range.
        if layer_id >= cpu_map.shape[0]:
            if require_global_experts:
                num_physical_experts = cpu_map.shape[-1]
                return list(
                    range(
                        logical_expert_id,
                        num_physical_experts,
                        self.num_logical_experts,
                    )
                )
            return [logical_expert_id]
        if require_global_experts:
            num_physical_experts = cpu_map[layer_id].shape[-1]
            return list(
                range(logical_expert_id, num_physical_experts, self.num_logical_experts)
            )
        return [
            physical_expert_id
            for physical_expert_id in cpu_map[layer_id, logical_expert_id].tolist()
            if physical_expert_id != -1
        ]


def format_expert_location_layout(
    metadata: Optional[ExpertLocationMetadata],
    layer_ids: Optional[Iterable[int]] = None,
) -> str:
    if metadata is None:
        return "<none>"

    return format_physical_to_logical_map(
        metadata.physical_to_logical_map_cpu,
        ep_size=metadata.ep_size,
        layer_ids=layer_ids,
    )


def format_expert_location_layout_diff(
    old_metadata: Optional[ExpertLocationMetadata],
    new_metadata: Optional[ExpertLocationMetadata],
    layer_ids: Optional[Iterable[int]] = None,
) -> str:
    if old_metadata is None or new_metadata is None:
        return "<none>"

    old_map = old_metadata.physical_to_logical_map_cpu
    new_map = new_metadata.physical_to_logical_map_cpu
    if old_map.shape != new_map.shape:
        return f"shape_changed old_shape={tuple(old_map.shape)} new_shape={tuple(new_map.shape)}"

    layer_ids = _normalize_layer_ids(layer_ids, num_layers=old_map.shape[0])
    num_physical_experts = old_map.shape[1]

    changed_by_layer = []
    for layer_id in layer_ids:
        num_changed = torch.count_nonzero(old_map[layer_id] != new_map[layer_id]).item()
        if num_changed > 0:
            changed_by_layer.append((layer_id, num_changed))

    total_changed = sum(num_changed for _, num_changed in changed_by_layer)
    total_slots = len(layer_ids) * num_physical_experts
    lines = [f"changed_physical_slots={total_changed}/{total_slots}"]
    if not changed_by_layer:
        lines.append("changed_layers=[]")
        return "\n".join(lines)

    for layer_id, num_changed in changed_by_layer:
        lines.append(f"layer={layer_id}: changed={num_changed}/{num_physical_experts}")
    return "\n".join(lines)


def format_physical_to_logical_map(
    physical_to_logical_map: torch.Tensor,
    ep_size: int,
    layer_ids: Optional[Iterable[int]] = None,
) -> str:
    physical_to_logical_map = physical_to_logical_map.cpu()
    if physical_to_logical_map.numel() == 0:
        return "<empty>"

    layer_ids = _normalize_layer_ids(
        layer_ids, num_layers=physical_to_logical_map.shape[0]
    )
    num_physical_experts = physical_to_logical_map.shape[1]
    num_local_physical_experts, remainder = divmod(num_physical_experts, ep_size)

    lines = [
        "physical_to_logical_map "
        f"num_layers={physical_to_logical_map.shape[0]} "
        f"num_physical_experts={num_physical_experts} "
        f"ep_size={ep_size}"
    ]
    for layer_id in layer_ids:
        row = physical_to_logical_map[layer_id].tolist()
        if remainder != 0:
            lines.append(
                f"layer={layer_id}: "
                f"physical={json.dumps(row, separators=(',', ':'))}"
            )
            continue

        rank_chunks = []
        for ep_rank in range(ep_size):
            start = ep_rank * num_local_physical_experts
            end = start + num_local_physical_experts
            rank_chunks.append(
                f"ep{ep_rank}={json.dumps(row[start:end], separators=(',', ':'))}"
            )
        lines.append(f"layer={layer_id}: " + " ".join(rank_chunks))

    return "\n".join(lines)


def _normalize_layer_ids(
    layer_ids: Optional[Iterable[int]],
    num_layers: int,
) -> List[int]:
    if layer_ids is None:
        return list(range(num_layers))

    normalized_layer_ids = [int(layer_id) for layer_id in layer_ids]
    for layer_id in normalized_layer_ids:
        assert 0 <= layer_id < num_layers, f"{layer_id=} {num_layers=}"
    return normalized_layer_ids


def get_global_expert_location_metadata():

    return get_resources().expert_location_metadata


def set_global_expert_location_metadata(value, allow_overwrite=False):

    resources = get_resources()
    if not allow_overwrite:
        assert resources.expert_location_metadata is None
    resources.expert_location_metadata = value


def append_trivial_expert_slots(
    physical_to_logical_map: torch.Tensor,
    count: int,
    num_logical_experts: int,
    start: int = 0,
) -> torch.Tensor:
    if count <= 0:
        return physical_to_logical_map
    new_slots = torch.arange(
        start,
        start + count,
        dtype=physical_to_logical_map.dtype,
        device=physical_to_logical_map.device,
    ).unsqueeze(0)
    new_slots = new_slots.expand(physical_to_logical_map.shape[0], -1)
    return torch.cat([physical_to_logical_map, new_slots % num_logical_experts], dim=1)


def broadcast_global_expert_location_metadata(
    model_config: ModelConfig,
    moe_ep_rank: int,
    src_rank: int = 0,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> ExpertLocationMetadata:
    metadata = get_global_expert_location_metadata()
    assert metadata is not None

    metadata.physical_to_logical_map = metadata.physical_to_logical_map.contiguous()
    torch.distributed.broadcast(
        metadata.physical_to_logical_map, src=src_rank, group=group
    )
    metadata = ExpertLocationMetadata.init_by_mapping(
        model_config,
        metadata.physical_to_logical_map,
        moe_ep_rank=moe_ep_rank,
    )
    set_global_expert_location_metadata(metadata, allow_overwrite=True)
    return metadata


def _compute_logical_to_all_physical_map(
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
    ep_size: int,
    moe_ep_rank: int,
):

    # This is rarely called, so we use for loops for maximum clarity

    num_layers, num_physical_experts = physical_to_logical_map.shape

    logical_to_all_physical_map = [
        [[] for _ in range(num_logical_experts)] for _ in range(num_layers)
    ]

    # 找出每层、每个逻辑专家对应的候选物理专家
    for layer_id in range(num_layers):
        for physical_expert_id in range(num_physical_experts):
            logical_expert_id = physical_to_logical_map[
                layer_id, physical_expert_id
            ].item()
            logical_to_all_physical_map[layer_id][logical_expert_id].append(
                physical_expert_id
            )

    # Replace by the physical expert on local GPU or node if possible. Skipped
    # without an a2a backend, where all EP ranks must agree on the pick: this
    # collapse is per-rank, and the full candidate list is what lets the dispatch
    # spread a hot expert over its replicas. See ExpertLocationDispatchInfo.
    if moe_ep_rank is not None and get_exec().moe.moe_a2a_backend != "none":
        num_local_gpu_physical_experts = num_physical_experts // ep_size
        prefer_same_node = _prefer_same_node_experts()
        num_gpus_per_node = (
            get_parallel().ep_size // get_parallel().nnodes
            if prefer_same_node
            else None
        )
        num_local_node_physical_experts = (
            num_local_gpu_physical_experts * num_gpus_per_node
            if num_gpus_per_node is not None
            else None
        )
        for layer_id in range(num_layers):
            for logical_expert_id in range(num_logical_experts):
                # 尝试找到“最近”的物理专家（同 GPU > 同节点）
                nearest_expert = _find_nearest_expert(
                    candidate_physical_expert_ids=logical_to_all_physical_map[layer_id][
                        logical_expert_id
                    ],
                    num_local_gpu_physical_experts=num_local_gpu_physical_experts,
                    moe_ep_rank=moe_ep_rank,
                    num_gpus_per_node=num_gpus_per_node,
                    num_local_node_physical_experts=num_local_node_physical_experts,
                )

                # 若找到则只保留这个最近的物理专家
                if nearest_expert != -1:
                    logical_to_all_physical_map[layer_id][logical_expert_id] = [
                        nearest_expert
                    ]

    logical_to_all_physical_map = _pad_nested_array(
        logical_to_all_physical_map, pad_value=-1
    )

    return torch.tensor(
        logical_to_all_physical_map, device=physical_to_logical_map.device
    )


def _pad_nested_array(arr, pad_value):
    # 把不等长的内层列表用 pad_value 补齐到相同长度（最大内层长度）
    max_len = max(len(inner) for outer in arr for inner in outer)
    padded = [
        [inner + [pad_value] * (max_len - len(inner)) for inner in outer]
        for outer in arr
    ]
    return padded


# TODO 优化性能（重写，和/或在独立进程中并行执行以重叠开销）
def compute_logical_to_rank_dispatch_physical_map(
    logical_to_all_physical_map: torch.Tensor,
    ep_size: int,
    num_physical_experts: int,
    ep_rank: int,
    seed: int = 42,
):

    r = random.Random(seed)

    device = logical_to_all_physical_map.device
    logical_to_all_physical_map = logical_to_all_physical_map.cpu()

    num_local_gpu_physical_experts = num_physical_experts // ep_size
    prefer_same_node = _prefer_same_node_experts()
    num_gpus_per_node = (
        get_parallel().ep_size // get_parallel().nnodes if prefer_same_node else None
    )
    num_local_node_physical_experts = (
        num_local_gpu_physical_experts * num_gpus_per_node
        if num_gpus_per_node is not None
        else None
    )
    num_layers, num_logical_experts, _ = logical_to_all_physical_map.shape
    dtype = logical_to_all_physical_map.dtype

    result_list = [
        [[-1] * num_logical_experts for _ in range(num_layers)] for _ in range(ep_size)
    ]

    for layer_id in range(num_layers):
        for logical_expert_id in range(num_logical_experts):
            candidate_physical_expert_ids = _logical_to_all_physical_raw(
                logical_to_all_physical_map, layer_id, logical_expert_id
            )

            # 阶段1：为每个 rank 先尝试分配最近（同 GPU/同节点）的物理专家；未命中记入 remaining_ranks
            remaining_ranks = []
            for moe_ep_rank in range(ep_size):
                val = _find_nearest_expert(
                    candidate_physical_expert_ids=candidate_physical_expert_ids,
                    num_local_gpu_physical_experts=num_local_gpu_physical_experts,
                    moe_ep_rank=moe_ep_rank,
                    num_gpus_per_node=num_gpus_per_node,
                    num_local_node_physical_experts=num_local_node_physical_experts,
                )

                result_list[moe_ep_rank][layer_id][logical_expert_id] = val
                if val == -1:
                    remaining_ranks.append(moe_ep_rank)

            # 阶段2：对未命中的 rank，在候选副本中公平随机分配，确保无遗漏（最终断言无 -1）
            if remaining_ranks:
                choices = _fair_choices(
                    candidate_physical_expert_ids, k=len(remaining_ranks), r=r
                )
                for moe_ep_rank, choice in zip(remaining_ranks, choices, strict=True):
                    result_list[moe_ep_rank][layer_id][logical_expert_id] = choice

    logical_to_rank_dispatch_physical_map = torch.tensor(result_list, dtype=dtype)
    assert torch.all(logical_to_rank_dispatch_physical_map != -1)

    return logical_to_rank_dispatch_physical_map[ep_rank, :, :].to(device)


def _logical_to_all_physical_raw(
    logical_to_all_physical_map, layer_id: int, logical_expert_id: int
) -> List[int]:
    return [
        physical_expert_id
        for physical_expert_id in logical_to_all_physical_map[
            layer_id, logical_expert_id
        ].tolist()
        if physical_expert_id != -1
    ]


def _compute_gpu_id_of_physical_expert(
    physical_expert_id: int, num_local_gpu_physical_experts: int
) -> int:
    return physical_expert_id // num_local_gpu_physical_experts


def _compute_node_id_of_physical_expert(
    physical_expert_id: int, num_local_host_physical_experts: int
) -> int:
    return physical_expert_id // num_local_host_physical_experts


def _find_nearest_expert(
    candidate_physical_expert_ids: List[int],
    num_local_gpu_physical_experts: int,
    moe_ep_rank: int,
    num_gpus_per_node: Optional[int],
    num_local_node_physical_experts: Optional[int],
) -> int:
    """在候选物理专家中，按“同 GPU > 同节点”的优先级为指定 rank 选一个最近的物理专家。

    返回：选中的物理专家 id；若既无同 GPU 也无同节点副本，返回 -1（表示未找到）。

    示例：
        candidate=[0, 4], moe_ep_rank=1, num_local_gpu_physical_experts=3
        -> GPU(0)=0、GPU(4)=1，同 GPU 命中 4，返回 4。
    """
    # 1. 若只有一个候选，直接返回
    if len(candidate_physical_expert_ids) == 1:
        return candidate_physical_expert_ids[0]

    # 2. 优先选择同一 GPU 上的专家
    same_gpu_physical_expert_ids = [
        physical_expert_id
        for physical_expert_id in candidate_physical_expert_ids
        if _compute_gpu_id_of_physical_expert(
            physical_expert_id, num_local_gpu_physical_experts
        )
        == moe_ep_rank
    ]
    if len(same_gpu_physical_expert_ids) > 0:
        return same_gpu_physical_expert_ids[0]

    # Prefer same-node experts only when it narrows the candidate set.
    if num_gpus_per_node is not None and num_local_node_physical_experts is not None:
        node_rank = moe_ep_rank // num_gpus_per_node
        same_node_physical_expert_ids = [
            physical_expert_id
            for physical_expert_id in candidate_physical_expert_ids
            if _compute_node_id_of_physical_expert(
                physical_expert_id, num_local_node_physical_experts
            )
            == node_rank
        ]
        if 0 < len(same_node_physical_expert_ids) < len(candidate_physical_expert_ids):
            return same_node_physical_expert_ids[0]

    # 4. 最后，保留 -1 表示未找到。
    return -1


def _fair_choices(arr: List, k: int, r: random.Random) -> List:
    """从 arr 中公平地选出 k 个元素（允许重复）并打乱顺序。

    每个元素至少被选 floor(k/len(arr)) 次，剩余名额随机不重复抽取，从而尽量均衡。

    示例：
        _fair_choices([0, 1], k=3) 可能返回 [1, 0, 0] 或 [0, 1, 1]（顺序已打乱）。
    """
    quotient, remainder = divmod(k, len(arr))
    ans = arr * quotient + r.sample(arr, k=remainder)
    r.shuffle(ans)
    return ans


@dataclass
class ModelConfigForExpertLocation:
    num_layers: int
    num_logical_experts: int
    num_groups: Optional[int] = None

    @staticmethod
    def from_model_config(model_config: ModelConfig):
        """从模型配置提取专家位置所需的配置（层数、逻辑专家数、分组数）。

        若模型类未实现 get_model_config_for_expert_location（非 MoE 或不支持 EPLB），则返回 None。
        """
        from sglang.srt.model_loader import get_model_architecture

        model_class, _ = get_model_architecture(model_config)
        if hasattr(model_class, "get_model_config_for_expert_location"):
            return model_class.get_model_config_for_expert_location(
                model_config.hf_config
            )
        else:
            return None


def compute_initial_expert_location_metadata(
    model_config: ModelConfig,
    moe_ep_rank: int,
) -> Optional[ExpertLocationMetadata]:
    data = get_exec().moe.init_expert_location
    if data == "trivial":
        return ExpertLocationMetadata.init_trivial(model_config, moe_ep_rank)

    # TODO unify with the utils function
    if data.endswith(".pt"):
        data_dict = torch.load(data, weights_only=True, map_location="cpu")
    elif data.endswith(".json"):
        data_dict = json.loads(Path(data).read_text())
    else:
        data_dict = json.loads(data)

    if "physical_to_logical_map" in data_dict:
        logger.info(
            "init_expert_location from init_by_mapping using ServerArgs.init_expert_location"
        )
        return ExpertLocationMetadata.init_by_mapping(
            model_config,
            **data_dict,
            moe_ep_rank=moe_ep_rank,
        )
    elif "logical_count" in data_dict:
        logger.info(
            "init_expert_location from init_by_eplb using ServerArgs.init_expert_location"
        )
        return ExpertLocationMetadata.init_by_eplb(
            model_config, logical_count=data_dict["logical_count"]
        )
    else:
        raise NotImplementedError(
            f"Unknown init_expert_location format ({list(data_dict.keys())=})"
        )
