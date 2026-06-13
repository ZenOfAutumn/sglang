# srt/layers/attention/linear

## 目录用途
本目录实现各类线性注意力后端及其调度逻辑，覆盖 GDN（Gated Delta Net）、KDA（Kimi Delta Attention）、Lightning Attention 与 SegLA 等机制。各后端基于 `MambaAttnBackendBase`，通过 kernel dispatcher 在 kernels/ 子目录下选择具体算子实现（Triton/CuteDSL/FlashInfer）。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| __init__.py | 包初始化文件（空）。 |
| gdn_backend.py | GDN 线性注意力后端（`GDNAttnBackend`）及 kernel 调度器 `GDNKernelDispatcher`。 |
| kda_backend.py | KDA 线性注意力后端（`KDAAttnBackend`）及 kernel 调度器 `KDAKernelDispatcher`。 |
| lightning_attn.py | Lightning Attention 的 Triton 算子（对角/非对角/KV 并行/解码）及 `BailingLinearKernel`。 |
| lightning_backend.py | Lightning Attention 后端（`LightningAttentionBackend`）。 |
| linear_metadata.py | 线性注意力前向元数据 `BailingLinearMetadata`。 |
| seg_la.py | SegLA（分段线性注意力）Triton 算子集合（前向/MTP/求和等 kernel）及 `SegLaMeta`。 |
| utils.py | 线性注意力工具：`LinearAttnKernelBackend` 枚举与按 ServerArgs 选择 prefill/decode kernel 后端。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| kernels | GDN/KDA 的具体算子后端实现（Triton/CuteDSL/FlashInfer）及统一基类。 |
