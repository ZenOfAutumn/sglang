# srt/eplb/eplb_simulator

## 目录用途
该目录提供 EPLB 的离线仿真与分析工具。它读取运行时由 `ExpertDistributionRecorder` 记录的专家负载数据（`.pt` 文件），按 forward pass 与 rank 聚合成全局物理专家计数，进而可将物理计数转换为逻辑计数，用于离线评估与调试不同的专家重平衡策略。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 导入并暴露 `reader` 模块。 |
| `reader.py` | 读取 `per_pass` 模式记录的专家分布数据，聚合各 rank 的全局物理专家计数，并复用 `_convert_global_physical_count_to_logical_count` 转换为逻辑计数。 |
