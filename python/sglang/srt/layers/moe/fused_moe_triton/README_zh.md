# srt/layers/moe/fused_moe_triton

## 目录用途
基于 Triton 的 fused MoE 核心实现。包含融合专家 GEMM kernel、block-size 对齐、配置加载/调优、Marlin 量化 MoE，以及对外暴露的 `FusedMoE` 层模块；configs 子目录存放按 GPU 与形状预调优的 kernel JSON 配置。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 导出 `fused_experts`、`FusedMoE`、`moe_align_block_size`、配置工具及 `override_config` 上下文管理器。 |
| `fused_marlin_moe.py` | 基于 Marlin 的量化（int4/fp4）fused MoE kernel 封装，含标量类型推导与 `fused_marlin_moe` 入口。 |
| `fused_moe.py` | fused MoE kernel 顶层调度：`fused_experts` 及 inplace/outplace 实现、SwiGLU 激活、moe_sum_reduce 等。 |
| `fused_moe_triton_config.py` | MoE kernel 调优配置管理：生成配置文件名、加载预调优 JSON、给出默认配置与最优配置选择。 |
| `fused_moe_triton_kernels.py` | 核心 Triton kernel 实现：`fused_moe_kernel`、GPTQ/AWQ 变体、激活与 moe_sum_reduce kernel、TMA 描述符等。 |
| `layer.py` | `FusedMoE` 层模块及 `FusedMoeWeightScaleSupported` 枚举、dispatcher 创建与 piecewise CUDA graph 前向实现。 |
| `moe_align_block_size.py` | 将各专家的 token 分布对齐到 block 边界，封装 sgl_kernel 的 `moe_align_block_size`。 |
| `triton_kernels_moe.py` | 基于外部 `triton_kernels` 库（matmul_ogs）的 MoE 前向实现，含带 bias 变体与量化辅助。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `configs` | 按 Triton 版本、GPU 型号与 MoE 形状预调优的 fused MoE kernel JSON 配置。 |
