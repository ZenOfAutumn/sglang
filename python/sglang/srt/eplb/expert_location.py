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
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.distributed
import torch.nn.functional as F

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


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
    # (layers, num_logical_experts, X)：逻辑专家 -> 其全部物理副本 的映射，空位用 -1 填充
    logical_to_all_physical_map: torch.Tensor
    # 上者的 CPU 副本，用于提升性能
    logical_to_all_physical_map_cpu: torch.Tensor
    # (layers, num_logical_experts)：每个逻辑专家有效物理副本的数量
    logical_to_all_physical_map_num_valid: torch.Tensor
    # (layers, num_logical_experts)：本 rank 上逻辑专家 -> 实际分发目标物理专家；仅 static 分发时有值
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

    @property
    def ep_size(self):
        # TODO 当 EP size != world size 时需要修改此处
        # 目前假设 EP 组大小等于分布式 world size
        return torch.distributed.get_world_size()

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
    def init_trivial(
        server_args: ServerArgs, model_config: ModelConfig, moe_ep_rank: int
    ):
        """平凡（默认）布局：逻辑专家 i 对应物理专家 i。

        当存在冗余物理专家时，靠 `% num_logical_experts` 循环回绕分配。

        输入：
        - server_args: 服务端参数（含 ep_size、ep_num_redundant_experts 等）。
        - model_config: 模型配置。
        - moe_ep_rank: 当前 MoE EP rank，用于就近选择本地物理专家。
        返回：ExpertLocationMetadata（若模型不含专家位置信息则返回 None）。

        示例：
            假设 num_layers=1, num_logical_experts=4, num_physical_experts=6，则
            physical_to_logical_map = [[0, 1, 2, 3, 0, 1]]
            （物理专家 4、5 分别是逻辑专家 0、1 的冗余副本）。
        """
        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        num_physical_experts = common["num_physical_experts"]
        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_layers = model_config_for_expert_location.num_layers
        num_logical_experts = model_config_for_expert_location.num_logical_experts

        physical_to_logical_map = (
            torch.arange(0, num_physical_experts).repeat(num_layers, 1)
            % num_logical_experts
        )

        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            physical_to_logical_map=physical_to_logical_map,
            moe_ep_rank=moe_ep_rank,
        )

    @staticmethod
    def init_by_mapping(
        server_args: ServerArgs,
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
        physical_to_logical_map = physical_to_logical_map.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        model_config_for_expert_location = common["model_config_for_expert_location"]
        logical_to_all_physical_map = _compute_logical_to_all_physical_map(
            server_args=server_args,
            physical_to_logical_map=physical_to_logical_map,
            num_logical_experts=model_config_for_expert_location.num_logical_experts,
            ep_size=common["ep_size"],
            moe_ep_rank=moe_ep_rank,
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
        )

    @staticmethod
    def init_by_eplb(
        server_args: ServerArgs, model_config: ModelConfig, logical_count: torch.Tensor
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
        logical_count = logical_count.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_physical_experts = common["num_physical_experts"]
        num_groups = model_config_for_expert_location.num_groups
        num_nodes = server_args.nnodes

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
                    raw_algorithm=server_args.eplb_algorithm,
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                ),
            )
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map.to(server_args.device),
            logical_to_all_physical_map=logical_to_all_physical_map.to(
                server_args.device
            ),
        )

    @staticmethod
    def _init_common(server_args: ServerArgs, model_config: ModelConfig):
        """提取各构造入口共用的基础量（专家总数、EP 尺寸、每卡专家数）。

        返回：包含 model_config_for_expert_location / num_physical_experts /
        num_local_physical_experts / ep_size 的 dict；若模型不支持专家位置则返回 None。

        示例：
            num_logical_experts=4, ep_num_redundant_experts=2, ep_size=2
            -> num_physical_experts=6, num_local_physical_experts=3
        """
        model_config_for_expert_location = (
            ModelConfigForExpertLocation.from_model_config(model_config)
        )

        # 模型未提供专家位置信息（非 MoE 或未实现该接口），直接返回 None
        if model_config_for_expert_location is None:
            return None

        # 物理专家数 = 逻辑专家数 + 冗余专家数
        num_physical_experts = (
            model_config_for_expert_location.num_logical_experts
            + server_args.ep_num_redundant_experts
        )
        ep_size = server_args.ep_size
        # 物理专家必须能被 EP 尺寸均匀分配到各 rank
        assert num_physical_experts % ep_size == 0
        num_local_physical_experts = num_physical_experts // ep_size

        return dict(
            model_config_for_expert_location=model_config_for_expert_location,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            ep_size=ep_size,
        )

    @staticmethod
    def _init_raw(
        server_args: ServerArgs,
        ep_size: int,
        physical_to_logical_map: torch.Tensor,
        logical_to_all_physical_map: torch.Tensor,
    ):
        """由两张基础映射表组装出完整的 ExpertLocationMetadata。

        内部完成：将 logical_to_all_physical_map 的最后一维 pad 到 num_physical_experts（用 -1）、
        统计每个逻辑专家的有效副本数、生成 CPU 副本，以及（仅 static 分发时）计算
        logical_to_rank_dispatch_physical_map。
        """
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
            logical_to_rank_dispatch_physical_map=(
                compute_logical_to_rank_dispatch_physical_map(
                    server_args=server_args,
                    logical_to_all_physical_map=logical_to_all_physical_map,
                    ep_size=ep_size,
                    num_physical_experts=num_physical_experts,
                    # TODO improve when we have real EP rank
                    ep_rank=torch.distributed.get_rank() % ep_size,
                )
                if server_args.ep_dispatch_algorithm == "static"
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
        """查询某层、某个逻辑专家对应的全部物理副本 id 列表。

        输入：
        - layer_id: 层 id；logical_expert_id: 逻辑专家 id。
        - require_global_experts: 若为 True，按全局步长 num_logical_experts 枚举副本（忽略 -1）。
        返回：物理专家 id 列表（已过滤 -1 的空位）。

        示例：
            若 logical_to_all_physical_map[layer_id, 0] = [0, 4, -1, -1]，
            则 logical_to_all_physical(layer_id, 0) -> [0, 4]。
        """
        # 使用 CPU 副本，避免每次调用都触发 GPU->CPU 同步（在权重更新场景下开销很大）
        if require_global_experts:
            num_physical_experts = self.logical_to_all_physical_map_cpu[layer_id].shape[
                -1
            ]
            return list(
                range(logical_expert_id, num_physical_experts, self.num_logical_experts)
            )
        return [
            physical_expert_id
            for physical_expert_id in self.logical_to_all_physical_map_cpu[
                layer_id, logical_expert_id
            ].tolist()
            if physical_expert_id != -1
        ]


_global_expert_location_metadata: Optional[ExpertLocationMetadata] = None


def get_global_expert_location_metadata():
    return _global_expert_location_metadata


def set_global_expert_location_metadata(value):
    # 全局单例只能设置一次，防止重复初始化
    global _global_expert_location_metadata
    assert _global_expert_location_metadata is None
    _global_expert_location_metadata = value


def broadcast_global_expert_location_metadata(
    src_rank: int = 0, group: Optional[torch.distributed.ProcessGroup] = None
):
    """将全局 ExpertLocationMetadata 从 src_rank 广播到所有 rank。

    用于弹性 EP（Elastic EP）的 rank 恢复场景，确保所有 rank（包括新恢复的）
    共享完全相同的专家位置元数据。

    注意：调用方必须保证 src_rank 是健康的 rank。在恢复场景中，本函数在
    try_recover_ranks 成功后调用，此时所有 rank（含 src_rank=0）都已恢复就绪。
    """
    metadata = get_global_expert_location_metadata()
    assert metadata is not None

    # 原地广播前确保设备张量是连续内存的
    metadata.physical_to_logical_map = metadata.physical_to_logical_map.contiguous()
    metadata.logical_to_all_physical_map = (
        metadata.logical_to_all_physical_map.contiguous()
    )
    metadata.logical_to_all_physical_map_num_valid = (
        metadata.logical_to_all_physical_map_num_valid.contiguous()
    )
    if metadata.logical_to_rank_dispatch_physical_map is not None:
        metadata.logical_to_rank_dispatch_physical_map = (
            metadata.logical_to_rank_dispatch_physical_map.contiguous()
        )

    device_tensors = [
        metadata.physical_to_logical_map,
        metadata.logical_to_all_physical_map,
        metadata.logical_to_all_physical_map_num_valid,
    ]
    if metadata.logical_to_rank_dispatch_physical_map is not None:
        device_tensors.append(metadata.logical_to_rank_dispatch_physical_map)

    for tensor in device_tensors:
        torch.distributed.broadcast(tensor, src=src_rank, group=group)

    # 广播完设备张量后，刷新对应的 CPU 副本
    metadata.physical_to_logical_map_cpu = metadata.physical_to_logical_map.cpu()
    metadata.logical_to_all_physical_map_cpu = (
        metadata.logical_to_all_physical_map.cpu()
    )


def _compute_logical_to_all_physical_map(
    server_args: ServerArgs,
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
    ep_size: int,
    moe_ep_rank: int,
):
    """由「物理 -> 逻辑」映射反推出「逻辑 -> 全部物理副本」映射。

    若传入 moe_ep_rank，则会尽量把每个逻辑专家收敛到其在本地 GPU / 本节点上的物理副本，
    以减少跨卡/跨节点通信。

    示例：
        physical_to_logical_map = [[0, 1, 2, 3, 0, 1]]
        -> 逻辑 0 的候选物理副本为 [0, 4]，逻辑 1 为 [1, 5]，逻辑 2 为 [2]，逻辑 3 为 [3]；
        pad 后返回形状 (layers, num_logical_experts, max_replicas) 的张量。
    """
    # 该函数调用频率很低，故使用 for 循环以获得最大可读性

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

    # 如可能，替换为本地 GPU 或本节点上的物理专家
    if moe_ep_rank is not None:
        num_gpus_per_node = server_args.ep_size // server_args.nnodes
        num_local_gpu_physical_experts = num_physical_experts // ep_size
        num_local_node_physical_experts = (
            num_local_gpu_physical_experts * num_gpus_per_node
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
    server_args: ServerArgs,
    logical_to_all_physical_map: torch.Tensor,
    ep_size: int,
    num_physical_experts: int,
    ep_rank: int,
    seed: int = 42,
):
    """为 static 分发算法预计算「每个 rank 上逻辑专家 -> 具体物理专家」的固定映射。

    策略：优先选本 rank 同 GPU / 同节点的副本；若都无则在候选副本中用固定随机种子
    公平地随机分配（_fair_choices），保证各 rank 结果确定且负载均衡。

    输入：
    - logical_to_all_physical_map: (layers, num_logical_experts, X)。
    - ep_rank: 当前 rank，最终只返回该 rank 对应的切片。
    - seed: 随机种子，保证各 rank 生成一致的全局分配。
    返回：(layers, num_logical_experts) 张量，元素为本 rank 上每个逻辑专家实际分发到的物理专家 id。
    """
    r = random.Random(seed)

    device = logical_to_all_physical_map.device
    logical_to_all_physical_map = logical_to_all_physical_map.cpu()

    num_local_gpu_physical_experts = num_physical_experts // ep_size
    num_gpus_per_node = server_args.ep_size // server_args.nnodes
    num_local_node_physical_experts = num_local_gpu_physical_experts * num_gpus_per_node
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
    num_gpus_per_node: int,
    num_local_node_physical_experts: int,
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

    # 3. 否则，优先选择同一节点上的专家
    node_rank = moe_ep_rank // num_gpus_per_node
    same_node_physical_expert_ids = [
        physical_expert_id
        for physical_expert_id in candidate_physical_expert_ids
        if _compute_node_id_of_physical_expert(
            physical_expert_id, num_local_node_physical_experts
        )
        == node_rank
    ]
    if len(same_node_physical_expert_ids) > 0:
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
    server_args: ServerArgs,
    model_config: ModelConfig,
    moe_ep_rank: int,
) -> Optional[ExpertLocationMetadata]:
    """根据 ServerArgs.init_expert_location 计算初始的专家位置元数据。

    init_expert_location 可以是：
    - "trivial"：采用平凡布局（init_trivial）。
    - ".pt" / ".json" 文件路径或 JSON 字符串：
        含 "physical_to_logical_map" -> 走 init_by_mapping；
        含 "logical_count"          -> 走 init_by_eplb。
    返回：ExpertLocationMetadata（模型不支持时为 None）；格式未知时抛 NotImplementedError。
    """
    data = server_args.init_expert_location
    if data == "trivial":
        return ExpertLocationMetadata.init_trivial(
            server_args, model_config, moe_ep_rank
        )

    # TODO unify with the utils function
    if data.endswith(".pt"):
        data_dict = torch.load(data, weights_only=True)
    elif data.endswith(".json"):
        data_dict = json.loads(Path(data).read_text())
    else:
        data_dict = json.loads(data)

    if "physical_to_logical_map" in data_dict:
        logger.info(
            "init_expert_location from init_by_mapping using ServerArgs.init_expert_location"
        )
        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            **data_dict,
            moe_ep_rank=moe_ep_rank,
        )
    elif "logical_count" in data_dict:
        logger.info(
            "init_expert_location from init_by_eplb using ServerArgs.init_expert_location"
        )
        return ExpertLocationMetadata.init_by_eplb(
            server_args, model_config, logical_count=data_dict["logical_count"]
        )
    else:
        raise NotImplementedError(
            f"Unknown init_expert_location format ({list(data_dict.keys())=})"
        )
