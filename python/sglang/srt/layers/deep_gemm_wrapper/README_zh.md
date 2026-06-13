# srt/layers/deep_gemm_wrapper

## 目录用途
本目录封装 DeepSeek 的 DeepGEMM 库，为 SGLang 提供 fp8/bf16 的 GEMM 与 grouped GEMM 算子入口。它负责在运行时探测是否启用 DeepGEMM（按 SM 架构与依赖），管理 JIT 编译与预热，并对外暴露统一的调用接口。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包入口，重导出 `entrypoint` 中的公共符号。 |
| `configurer.py` | 运行时配置探测：根据 GPU SM 版本与依赖计算 `ENABLE_JIT_DEEPGEMM`、`DEEPGEMM_BLACKWELL`、`DEEPGEMM_SCALE_UE8M0` 等开关。 |
| `compile_utils.py` | JIT 编译与预热工具：`DeepGemmKernelType` 枚举、各类 warmup executor、按 kernel 类型预编译及执行 hook。 |
| `entrypoint.py` | 对外算子入口：`grouped_gemm_nt_f8f8bf16_masked/contig`、`gemm_nt_f8f8bf16`、`gemm_nt_bf16bf16f32`，及配置/SMs 设置函数。 |

## 子目录
无子目录。
