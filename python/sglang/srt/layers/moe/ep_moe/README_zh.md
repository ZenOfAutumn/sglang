# srt/layers/moe/ep_moe

## 目录用途
专家并行（Expert Parallelism, EP）MoE 实现。在多卡间切分专家，结合 token 分发后端（DeepEP、Mori 等）完成 permute/scatter/gather 与分组专家计算，并提供 NPU FuseEP、Mori EP 等专门变体。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 空的包初始化文件。 |
| `kernels.py` | 专家并行所需的 Triton kernel 集合：DeepEP permute/反排序、src2dst 映射、CUTLASS MoE 预处理、silu_and_mul 后量化、ep scatter 等。 |
| `layer.py` | 专家并行 MoE 层实现：`DeepEPMoE`（继承 `FusedMoE`）及 `NpuFuseEPMoE`、`MoriEPMoE` 变体，并提供 `get_moe_impl_class` 选择具体实现。 |
