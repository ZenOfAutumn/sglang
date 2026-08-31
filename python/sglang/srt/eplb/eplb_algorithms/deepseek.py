"""DeepSeek EPLB 专家负载均衡算法的参考实现。

本文件复制自 https://github.com/deepseek-ai/EPLB/blob/main/eplb.py，
原因是上游项目没有发布为可直接依赖的 PyPI 软件包。

算法分为三个阶段：将专家组均衡分配到节点、为热点逻辑专家创建冗余
物理副本，以及将物理专家均衡放置到 GPU。非分层模式可视为只有一个
专家组和一个节点的特例。
"""

from typing import Tuple

import torch


def balanced_packing(
    weight: torch.Tensor, num_packs: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将带权对象等量装入多个容器，并尽量均衡各容器的总权重。

    每行负载独立执行一次贪心装箱。算法按权重从大到小处理对象，每次将
    当前对象放入“尚未装满且累计权重最小”的容器。每个容器最终必须恰好
    包含 ``n / num_packs`` 个对象。

    Args:
        weight: 对象权重，形状为 ``[X, n]``；``X`` 通常表示 MoE 层数，
            ``n`` 表示每层待分配的对象数。
        num_packs: 容器数量，要求 ``n`` 能被该值整除。

    Returns:
        两个形状均为 ``[X, n]`` 的整型张量：

        - ``pack_index``：每个对象被分配到的容器编号。
        - ``rank_in_pack``：每个对象在所属容器中的槽位编号。

    Example:
        将四个对象等量放入两个容器。算法会让两个容器的总权重都等于 11：

        >>> weight = torch.tensor([[10.0, 1.0, 8.0, 3.0]])
        >>> pack_index, rank_in_pack = balanced_packing(weight, num_packs=2)
        >>> pack_index
        tensor([[0, 0, 1, 1]])
        >>> rank_in_pack
        tensor([[0, 1, 0, 1]])
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        # 每个容器只有一个对象时，无需执行贪心搜索，直接按原顺序分配。
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # 将每层对象按权重降序排列。先放置重对象能降低贪心装箱最终产生
    # 极端重载容器的概率；转换成 Python 列表是为了执行逐层的状态更新。
    indices_list = weight.float().sort(-1, descending=True).indices.tolist()
    weight_list = weight.tolist()
    pack_index_list = [[-1] * num_groups for _ in range(num_layers)]
    rank_in_pack_list = [[-1] * num_groups for _ in range(num_layers)]
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices_list[i]:
            # 只从未满的容器中选择当前累计权重最小者。
            pack = min(
                (j for j in range(num_packs) if pack_items[j] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < groups_per_pack
            pack_index_list[i][group] = pack
            rank_in_pack_list[i][group] = pack_items[pack]
            pack_weights[pack] += weight_list[i][group]
            pack_items[pack] += 1
    pack_index = torch.tensor(pack_index_list, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.tensor(rank_in_pack_list, dtype=torch.int64, device="cpu")
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把逻辑专家扩展为指定数量的物理副本，并尽量降低最大副本负载。

    每个逻辑专家初始拥有一个物理副本。之后逐个分配冗余名额，每次选择
    当前平均副本负载 ``weight / logcnt`` 最大的逻辑专家，为其增加一个
    副本。这里假设同一逻辑专家的负载能在其所有副本间均匀分摊。

    Args:
        weight: 各逻辑专家的负载，形状为 ``[X, num_log]``。
        num_phy: 复制完成后的物理专家总数，不能小于逻辑专家数
            ``num_log``。

    Returns:
        三个整型张量组成的元组：

        - ``phy2log``：每个物理专家对应的逻辑专家编号，形状为
          ``[X, num_phy]``。
        - ``rank``：物理专家在同一逻辑专家全部副本中的序号，形状为
          ``[X, num_phy]``，首个副本的序号为 0。
        - ``logcnt``：每个逻辑专家最终拥有的副本数，形状为
          ``[X, num_log]``。

    Example:
        两个冗余名额都会分配给负载远高于其他专家的专家 0：

        >>> weight = torch.tensor([[100.0, 5.0, 5.0, 5.0]])
        >>> phy2log, rank, logcnt = replicate_experts(weight, num_phy=6)
        >>> phy2log
        tensor([[0, 1, 2, 3, 0, 0]])
        >>> rank
        tensor([[0, 0, 0, 0, 1, 2]])
        >>> logcnt
        tensor([[3, 1, 1, 1]])
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        # 选择增加副本前平均负载最高的专家。每层独立选择，因此
        # redundant_indices 中每个元素对应一层的专家编号。
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """按“专家组→节点→GPU”的层级执行专家复制与负载均衡。

    算法先以专家组为不可拆分单位，将组负载均衡到各节点；然后在每个
    节点内部为热点专家分配冗余副本；最后将节点内的物理专家等量、均衡
    地装入 GPU。这样既控制计算负载，又尽量保留专家组的节点局部性。

    Args:
        weight: 各 MoE 层中逻辑专家的负载，形状为
            ``[num_moe_layers, num_logical_experts]``。
        num_physical_experts: 创建冗余副本后的物理专家总数。
        num_groups: 专家组数量，要求逻辑专家数能被其整除，并且该值能
            被 ``num_nodes`` 整除。
        num_nodes: 服务器节点数；通常节点内网络（如 NVLink）快于节点间
            网络。
        num_gpus: GPU 总数，要求能被 ``num_nodes`` 整除，且物理专家总数
            能被该值整除。

    Returns:
        三个整型张量组成的元组：

        - ``physical_to_logical_map``：物理专家对应的逻辑专家编号，形状为
          ``[num_moe_layers, num_physical_experts]``。
        - ``physical_rank``：物理专家在对应逻辑专家副本中的序号，形状为
          ``[num_moe_layers, num_physical_experts]``。
        - ``logical_count``：每个逻辑专家拥有的副本数，形状为
          ``[num_moe_layers, num_logical_experts]``。

    Example:
        在没有冗余专家的最小示例中，两个逻辑专家分别映射到自身：

        >>> weight = torch.tensor([[8.0, 2.0]])
        >>> phy2log, physical_rank, logcnt = rebalance_experts_hierarchical(
        ...     weight,
        ...     num_physical_experts=2,
        ...     num_groups=1,
        ...     num_nodes=1,
        ...     num_gpus=1,
        ... )
        >>> phy2log
        tensor([[0, 1]])
        >>> physical_rank
        tensor([[0, 0]])
        >>> logcnt
        tensor([[1, 1]])
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        """逐行计算排列的逆映射。

        例如输入 ``[[2, 0, 1]]`` 时返回 ``[[1, 2, 0]]``，因为原排列中
        数值 0、1、2 分别位于位置 1、2、0。
        """
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(
                perm.shape
            ),
        )
        return inv

    # 第一步：计算每个专家组的总负载，再把完整专家组均衡分配到节点。
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
        ).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)
    ).flatten(-2)
    mlog2log = inverse(log2mlog)

    # 第二步：按第一步得到的排列重排逻辑专家，并在每个节点内部独立
    # 创建冗余副本。展平层和节点维度后，张量形状为
    # [num_layers * num_nodes, num_logical_experts // num_nodes]。
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # 第三步：用“逻辑专家负载 / 副本数”估算每个物理副本的负载，再将
    # 节点内物理专家等量装入节点内各 GPU。此时张量形状为
    # [num_layers * num_nodes, num_physical_experts // num_nodes]。
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    # 按 GPU 槽位顺序取得节点内逻辑专家编号，形状为
    # [num_layers * num_nodes, num_logical_experts // num_nodes]。
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            device=group_pack_index.device,
        ).view(1, -1, 1)
    ).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSeek 专家并行负载均衡算法的统一入口。

    Args:
        weight: 所有逻辑专家的累计负载，形状为
            ``[layers, num_logical_experts]``。
        num_replicas: 物理专家副本总数，要求是 ``num_gpus`` 的整数倍。
        num_groups: 专家组数量；分层模式下必须满足相关整除约束。
        num_nodes: 服务器节点数；通常节点内网络（如 NVLink）快于节点间
            网络。
        num_gpus: GPU 总数，要求是 ``num_nodes`` 的整数倍。
        enable_hierarchical: 是否启用节点感知的分层均衡。为 ``False`` 时，
            将所有专家和设备视为一个全局资源池。

    Returns:
        三个整型张量组成的元组：

        - ``physical_to_logical_map``：每个物理副本对应的逻辑专家编号，
          形状为 ``[layers, num_replicas]``。
        - ``logical_to_physical_map``：每个逻辑专家对应的物理副本编号列表，
          形状为 ``[layers, num_logical_experts, max_replica_count]``，空槽
          使用 ``-1`` 填充。
        - ``expert_count``：每个逻辑专家的物理副本数，形状为
          ``[layers, num_logical_experts]``。

    Example:
        输入一层、两个专家的负载，并使用两个物理槽位：

        >>> weight = torch.tensor([[8.0, 2.0]])
        >>> phy2log, log2phy, count = rebalance_experts(
        ...     weight=weight,
        ...     num_replicas=2,
        ...     num_groups=1,
        ...     num_nodes=1,
        ...     num_gpus=1,
        ...     enable_hierarchical=False,
        ... )
        >>> phy2log
        tensor([[0, 1]])
        >>> log2phy
        tensor([[[0], [1]]])
        >>> count
        tensor([[1, 1]])
    """

    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if enable_hierarchical:
        # 分层策略保留专家组和节点边界，以减少代价较高的跨节点通信。
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        # 全局策略将全部专家视为一个组、全部 GPU 视为位于一个节点，
        # 从而复用同一套分层实现并获得最大的放置自由度。
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )
    # logical_to_physical_map 的最后一维按最大副本数统一补齐；不存在的
    # 副本槽位保留为 -1，便于下游结合 logcnt 判断有效范围。
    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    # 将“物理专家→逻辑专家 + 副本序号”反向散射为
    # “逻辑专家→全部物理副本”的查找表。
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(
            num_layers, -1
        ),
    )
    return phy2log, log2phy, logcnt


__all__ = ["rebalance_experts"]
