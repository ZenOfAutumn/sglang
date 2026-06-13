# srt/layers/moe/moe_runner

## 目录用途
MoE 运行器（MoeRunner）抽象层。定义统一的 RunnerInput/Output、量化信息与 permute 方法注册机制，并为各计算后端（Triton、DeepGEMM、Marlin、FlashInfer TRTLLM、triton_kernels）提供具体 Core 实现，将不同 dispatch 格式与 fused experts kernel 解耦衔接。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 导出 `MoeRunnerConfig` 与 `MoeRunner`。 |
| `base.py` | 运行器抽象基类与数据结构：`MoeRunnerConfig`、`RunnerInput/Output`、`MoeQuantInfo`、`MoeRunnerCore`，以及 fused 函数与 pre/post permute 的注册池。 |
| `deep_gemm.py` | DeepGEMM 后端核心 `DeepGemmRunnerCore` 及其输入/输出/量化信息，含 standard 与 deepep（normal/ll）的 permute 转换。 |
| `flashinfer_trtllm.py` | FlashInfer TRTLLM 后端：FP8/FP4/BF16 量化信息、权重对齐工具与 `fused_experts_none_to_flashinfer_trtllm` 系列入口。 |
| `marlin.py` | Marlin 后端核心：`MarlinRunnerInput/Output`、`MarlinMoeQuantInfo` 与 `fused_experts_none_to_marlin`。 |
| `runner.py` | `MoeRunner` 调度类，按后端组装对应 Core 并串联 permute 与 fused 计算。 |
| `triton.py` | Triton 后端核心 `TritonRunnerCore` 及输入/输出/量化信息与 standard↔triton 的 permute 实现。 |
| `triton_kernels.py` | 基于外部 triton_kernels 库的后端核心骨架 `TritonKernelsRunnerCore` 及其 permute 转换。 |
