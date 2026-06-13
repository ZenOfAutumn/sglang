# srt/batch_overlap

## 目录用途
本目录实现计算与通信的重叠（overlap）调度，用于 MoE/DeepEP 等场景下提升 GPU 利用率。它将一次前向拆分为可分阶段执行的"操作"序列，支持单批次重叠（SBO）和双批次重叠（TBO，Two-Batch Overlap），通过交错执行两个微批的不同阶段来隐藏 all-to-all 通信延迟。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `operations.py` | 操作执行框架：定义 `YieldOperation`/`ExecutionOperation`、`_StageExecutor` 阶段执行器，以及 `execute_operations`/`execute_overlapped_operations` 与基于 yield 分隔符的阶段切分逻辑。 |
| `operations_strategy.py` | 定义 `OperationsStrategy` 并为 DeepSeek、Qwen3、MiMo-V2 等 MoE 模型的 prefill/decode 计算 TBO 操作序列与 DeepGEMM SM 数等策略。 |
| `single_batch_overlap.py` | 单批次重叠（SBO）：`SboFlags` 开关、`CombineOverlapArgs`/`DownGemmOverlapArgs` 及 `compute_overlap_args`，用于将 combine/下投影 GEMM 与 dispatch 重叠。 |
| `two_batch_overlap.py` | 双批次重叠（TBO）核心：序列拆分索引计算、投机信息拆分、CUDA Graph 重放支持，以及 `TboCudaGraphRunnerPlugin`、`TboDPAttentionPreparer`、`TboForwardBatchPreparer` 等准备器。 |
