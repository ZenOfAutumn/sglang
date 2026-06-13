# srt/hardware_backend/npu/attention

## 目录用途
本目录实现昇腾 NPU 的注意力后端，将 SGLang 注意力抽象映射到 `torch_npu` 与 `sgl_kernel_npu` 算子，覆盖标准注意力、attention sinks、MLA 等场景，并提供基于 Torch 原生 SDPA 的回退路径及 MLA 预处理优化。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `ascend_backend.py` | NPU 主注意力后端（继承 `AttentionBackend`），整合 NPU 算子、attention sinks 与 MLA 预处理，处理 prefill/decode 各前向模式。 |
| `ascend_torch_native_backend.py` | `AscendTorchNativeAttnBackend`，基于 `scaled_dot_product_attention` 的原生注意力实现，支持 softcapping/logit cap 等作为回退路径。 |
| `mla_preprocess.py` | MLA 预处理工具与 `NPUFusedMLAPreprocess`，提供 `is_mla_preprocess_enabled`/`is_fia_nz` 开关、对齐 round_up 及 NZ 格式相关预处理。 |
