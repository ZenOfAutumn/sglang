# srt/lora/backend

## 目录用途
该目录定义 LoRA 计算后端的统一抽象与各硬件/算法后端实现。后端封装 LoRA 各类核(embedding、shrink/expand SGEMM、qkv/gate_up 投影等)的具体计算方式，使上层 `LoRAManager` 可在不同硬件与算法间切换。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `base_backend.py` | `BaseLoRABackend`：所有后端的基类，定义 LoRA 各算子接口及通用参数(max_loras_per_batch、device)。 |
| `lmhead_mixing.py` | `LoRABackendLmHeadMixing`：为后端提供 LMHead LoRA(含分块 logprobs 多 pass)的 batch_info 管理 mixin。 |
| `triton_backend.py` | `TritonLoRABackend`：基于 Triton 核(sgemm a/b、qkv_b、gate_up_b、embedding)的默认后端。 |
| `chunked_backend.py` | `ChunkedSgmvLoRABackend`：基于 Punica 分块 SGMV 算法的后端，将序列切分为固定块以减少核启动开销。 |
| `torch_backend.py` | `TorchNativeLoRABackend` 及 `TorchNativeLoRABatchInfo`：纯 PyTorch 实现的后端，含 CPU 侧张量字段。 |
| `ascend_backend.py` | `AscendLoRABackend`：基于昇腾 NPU(torch_npu / sgl_kernel_npu)的后端实现。 |
| `lora_registry.py` | 后端注册表：`register_lora_backend` 装饰器与 `get_backend_from_name`，按名称(triton/csgmv/ascend/torch_native/flashinfer)创建后端。 |
