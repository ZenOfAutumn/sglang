"""EPLB（Expert-Parallelism Load Balancer，专家并行负载均衡器）算法入口。

本模块统一定义可选的负载均衡算法，并根据调用方选择的算法，将专家负载
统计分发给对应实现。各实现最终都返回物理专家与逻辑专家之间的新映射关系。
"""

from enum import Enum, auto
from typing import Optional

import torch

from sglang.srt.eplb.eplb_algorithms import deepseek, deepseek_vec, elasticity_aware


class EplbAlgorithm(Enum):
    """EPLB 支持的专家负载均衡算法。

    名称带 ``hierarchical`` 的算法会感知专家组、节点和 GPU 的层级结构，
    优先在高速互联范围内放置专家；其余算法执行全局负载均衡。
    """

    # DeepSeek 基于聚合负载的全局均衡算法。
    deepseek = auto()
    # DeepSeek 基于聚合负载的分层均衡算法。
    deepseek_hierarchical = auto()
    # 保留负载时间步/向量信息的全局均衡算法。
    deepseek_vec = auto()
    # 保留负载时间步/向量信息的分层均衡算法。
    deepseek_vec_hierarchical = auto()
    # 感知弹性 EP 活跃 rank 状态的全局均衡算法。
    elasticity_aware = auto()
    # 感知弹性 EP 活跃 rank 状态的分层均衡算法。
    elasticity_aware_hierarchical = auto()
    # TODO：后续可能会增加更多负载均衡算法。


def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: Optional[int],
    num_nodes: int,
    algorithm: EplbAlgorithm,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """根据指定算法重新分配物理专家副本。

    Args:
        tokens_per_expert: 各统计步、MoE 层和逻辑专家处理的 token 数，形状为
            ``[num_steps, num_layers, num_logical_experts]``。
        num_physical_experts: 所有 GPU 上可放置的物理专家副本总数。
        num_local_physical_experts: 每个 GPU 承载的物理专家副本数。物理专家
            总数必须能被该值整除，两者之商即参与均衡的 GPU 数。
        num_groups: 逻辑专家组数。仅分层算法需要有效值；全局算法可传
            ``None``。
        num_nodes: 参与均衡的计算节点数。
        algorithm: 要使用的 :class:`EplbAlgorithm` 枚举值。

    Returns:
        三个整型张量组成的元组：

        - ``physical_to_logical_map``：每个物理专家对应的逻辑专家编号，
          形状为 ``[num_layers, num_physical_experts]``。
        - ``logical_to_physical_map``：每个逻辑专家的物理副本编号，形状为
          ``[num_layers, num_logical_experts, max_num_replicas]``；没有对应
          副本的位置通常以 ``-1`` 填充。
        - ``expert_count``：每个逻辑专家拥有的物理副本数，形状为
          ``[num_layers, num_logical_experts]``。

    Raises:
        NotImplementedError: ``algorithm`` 不属于当前支持的算法集合。

    Example:
        输入 2 个统计步、1 个 MoE 层、2 个逻辑专家的负载，为其分配 4 个
        物理专家；输出分别描述物理到逻辑映射、逻辑到物理映射和副本数：

        >>> loads = torch.tensor([[[10, 2]], [[6, 2]]])
        >>> phy2log, log2phy, counts = rebalance_experts(
        ...     tokens_per_expert=loads,
        ...     num_physical_experts=4,
        ...     num_local_physical_experts=2,
        ...     num_groups=None,
        ...     num_nodes=1,
        ...     algorithm=EplbAlgorithm.deepseek,
        ... )
        >>> phy2log.shape
        torch.Size([1, 4])
        >>> log2phy.shape[:2]
        torch.Size([1, 2])
        >>> counts.shape
        torch.Size([1, 2])
    """
    if algorithm in [EplbAlgorithm.deepseek, EplbAlgorithm.deepseek_hierarchical]:
        # 标量版 DeepSeek 算法只使用每层每个专家的累计负载，因此先沿
        # 统计步维度求和，将三维输入压缩为 [层数, 逻辑专家数]。
        return deepseek.rebalance_experts(
            weight=tokens_per_expert.sum(dim=0),
            num_replicas=num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            # 每张 GPU 承载相同数量的本地物理专家，由此反推出 GPU 总数。
            num_gpus=num_physical_experts // num_local_physical_experts,
            enable_hierarchical=algorithm == EplbAlgorithm.deepseek_hierarchical,
        )

    if algorithm in [
        EplbAlgorithm.deepseek_vec,
        EplbAlgorithm.deepseek_vec_hierarchical,
    ]:
        # 向量版算法需要观察不同统计步之间的负载变化，因此保留原始三维
        # token 计数，不像标量版那样预先聚合统计步。
        return deepseek_vec.rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            enable_hierarchical=algorithm == EplbAlgorithm.deepseek_vec_hierarchical,
        )

    if algorithm in [
        EplbAlgorithm.elasticity_aware,
        EplbAlgorithm.elasticity_aware_hierarchical,
    ]:
        # 仅在选择弹性感知算法时导入状态管理器，避免普通 EPLB 路径加载
        # Elastic EP 相关依赖，也避免潜在的循环导入。
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        return elasticity_aware.rebalance_experts(
            weight=tokens_per_expert.sum(dim=0),
            num_replicas=num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_physical_experts // num_local_physical_experts,
            enable_hierarchical=(
                algorithm == EplbAlgorithm.elasticity_aware_hierarchical
            ),
            # 已初始化管理器时使用实时活跃 rank；尚未初始化时，按所有 rank
            # 健康的默认状态执行均衡，使非弹性启动流程也可安全调用。
            active_ranks=(
                ElasticEPStateManager.instance().active_ranks
                if ElasticEPStateManager.instance() is not None
                else ElasticEPStateManager.healthy_rank_state()
            ),
        )

    # 该分支通常只会在新增枚举值却未同步增加分发逻辑时触发。
    raise NotImplementedError(f"Unsupported EPLB algorithm: {algorithm}")


def compute_algorithm(
    raw_algorithm: str,
    num_groups: Optional[int],
    num_nodes: int,
) -> EplbAlgorithm:
    """将配置字符串解析为最终使用的 EPLB 算法。

    当 ``raw_algorithm`` 不是 ``"auto"`` 时，按枚举成员名称直接解析；
    自动模式仅在专家组可以被节点数整除时选择分层 DeepSeek 算法，否则
    回退到不依赖分组拓扑的全局 DeepSeek 算法。

    Args:
        raw_algorithm: 算法配置值，可为 :class:`EplbAlgorithm` 的成员名称
            或特殊值 ``"auto"``。
        num_groups: 逻辑专家组数；未知或未启用分组时为 ``None``。
        num_nodes: 计算节点数，用于判断专家组能否均匀分配到节点。

    Returns:
        解析或自动选择得到的 :class:`EplbAlgorithm`。

    Raises:
        KeyError: 显式指定的 ``raw_algorithm`` 不是有效的枚举成员名称。

    Example:
        显式配置会原样解析；自动模式会根据分组和节点的整除关系选择算法：

        >>> compute_algorithm("deepseek_vec", num_groups=None, num_nodes=1)
        <EplbAlgorithm.deepseek_vec: 3>
        >>> compute_algorithm("auto", num_groups=8, num_nodes=2)
        <EplbAlgorithm.deepseek_hierarchical: 2>
        >>> compute_algorithm("auto", num_groups=7, num_nodes=2)
        <EplbAlgorithm.deepseek: 1>
    """
    if raw_algorithm != "auto":
        # 枚举下标访问会严格校验配置名称，非法名称由 KeyError 明确暴露。
        return EplbAlgorithm[raw_algorithm]

    # TODO：需要在真实业务场景中对比各算法性能，再完善自动选择策略。
    # 分层算法要求每个节点分得相同数量的专家组，否则退回全局均衡算法。
    if (num_groups is not None) and (num_groups % num_nodes == 0):
        return EplbAlgorithm.deepseek_hierarchical
    else:
        return EplbAlgorithm.deepseek
