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
| `eplb_algorithms` | 专家重平衡算法实现（DeepSeek 原版、向量化版、弹性感知版）。 |
| `eplb_simulator` | EPLB 离线仿真/分析工具，从记录的专家分布数据回放并评估均衡策略。 |
