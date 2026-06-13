# srt/eplb/eplb_algorithms

## 目录用途
该目录汇集专家并行负载均衡的具体重平衡算法。给定各专家的 token 负载统计，这些算法计算逻辑专家到物理副本（含冗余副本）的放置方案，使各 GPU/节点的负载尽可能均衡。`__init__.py` 提供统一入口，按 `EplbAlgorithm` 枚举分派到对应实现，并支持层次化（hierarchical）变体。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 定义 `EplbAlgorithm` 枚举与统一入口 `rebalance_experts`，按算法类型分派到 deepseek / deepseek_vec / elasticity_aware。 |
| `deepseek.py` | DeepSeek 官方 EPLB 算法（自其仓库拷贝），含均衡装箱 `balanced_packing`、专家复制与层次化重平衡 `rebalance_experts_hierarchical`。 |
| `deepseek_vec.py` | DeepSeek 算法的向量化实现，按分块（chunkwise）批量生成冗余专家与物理↔逻辑映射，提升大规模计算效率。 |
| `elasticity_aware.py` | 弹性感知重平衡算法，依据 `active_ranks` 在部分 rank 失效/变动时回退到全局负载均衡策略并复用 deepseek 层次化算法。 |
