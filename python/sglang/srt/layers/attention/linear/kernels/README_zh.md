# srt/layers/attention/linear/kernels

## 目录用途
本目录提供 GDN 与 KDA 线性注意力的可插拔算子后端实现。所有 kernel 继承统一基类 `LinearAttnKernelBase`，由上层 dispatcher 根据硬件与配置选择 Triton、CuteDSL 或 FlashInfer 实现。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| __init__.py | 包初始化文件（空）。 |
| kernel_backend.py | 线性注意力算子统一抽象基类 `LinearAttnKernelBase`（ABC）。 |
| gdn_cutedsl.py | GDN 的 CuteDSL 算子实现 `CuteDSLGDNKernel`。 |
| gdn_flashinfer.py | GDN 的 FlashInfer 算子实现 `FlashInferGDNKernel`。 |
| gdn_triton.py | GDN 的 Triton 算子实现 `TritonGDNKernel`。 |
| kda_cutedsl.py | KDA 的 CuteDSL 算子实现 `CuteDSLKDAKernel`。 |
| kda_triton.py | KDA 的 Triton 算子实现 `TritonKDAKernel`。 |
