# srt/eplb

## 目录用途
该目录实现专家并行负载均衡（Expert Parallelism Load Balancing, EPLB）。在 MoE 模型的专家并行部署中，它负责统计各专家的 token 负载分布、计算逻辑专家到物理副本的映射、周期性地重平衡专家在各 GPU/节点上的放置，并在运行时动态更新专家位置与分发路由，从而缓解专家负载不均带来的性能瓶颈。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化文件（空）。 |
| `eplb_manager.py` | `EPLBManager` 调度入口，按配置的迭代周期触发专家负载统计与重平衡流程，驱动专家位置更新。 |
| `expert_distribution.py` | 专家负载分布记录器，采集并聚合各 forward pass/rank 的物理专家计数，支持 per_pass 等记录模式并提供全局单例。 |
| `expert_location.py` | `ExpertLocationMetadata` 专家位置元数据，维护物理↔逻辑专家映射，结合 eplb 算法计算专家放置。 |
| `expert_location_dispatch.py` | `ExpertLocationDispatchInfo` 专家分发信息，按 static/random 算法生成逻辑专家到物理副本的运行时分发映射。 |
| `expert_location_updater.py` | `ExpertLocationUpdater`，在重平衡后通过点对点通信（P2POp）在各 rank 间迁移专家权重并更新位置元数据，支持弹性 EP。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `eplb_algorithms` | 专家重平衡算法实现（DeepSeek 原版、向量化版、弹性感知版）。详见其 `README_zh.md`。 |
| `eplb_simulator` | EPLB 离线仿真/分析工具，从记录的专家分布数据回放并评估均衡策略。详见其 `README_zh.md`。 |

### `eplb_algorithms/` 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 定义 `EplbAlgorithm` 枚举与统一入口 `rebalance_experts`/`compute_algorithm`，按算法类型分派到 deepseek / deepseek_vec / elasticity_aware，并支持层次化（hierarchical）变体。 |
| `deepseek.py` | DeepSeek 官方 EPLB 算法（自其仓库拷贝），含均衡装箱 `balanced_packing`、专家复制与层次化重平衡 `rebalance_experts_hierarchical`。 |
| `deepseek_vec.py` | DeepSeek 算法的向量化实现，按分块（chunkwise）批量生成冗余专家与物理↔逻辑映射，提升大规模计算效率。 |
| `elasticity_aware.py` | 弹性感知重平衡算法，依据 `active_ranks` 在部分 rank 失效/变动时回退到全局负载均衡策略并复用 deepseek 层次化算法。 |
| `README_zh.md` | 该子目录的中文说明文档。 |

### `eplb_simulator/` 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 导入并暴露 `reader` 模块。 |
| `reader.py` | 读取 `per_pass` 模式记录的专家分布数据（`.pt`），聚合各 rank 的全局物理专家计数，并复用 `_convert_global_physical_count_to_logical_count` 转换为逻辑计数，用于离线评估重平衡策略。 |
| `README_zh.md` | 该子目录的中文说明文档。 |
