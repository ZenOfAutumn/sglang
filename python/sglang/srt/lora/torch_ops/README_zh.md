# srt/lora/torch_ops

## 目录用途
该目录提供基于纯 PyTorch 的 LoRA 计算算子实现，供 `TorchNativeLoRABackend` 在不依赖 Triton 的环境(如部分 CPU/通用设备)下完成分段 LoRA SGEMM。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 导出 `sgemm_lora_a_fwd`、`sgemm_lora_b_fwd` 两个算子。 |
| `lora_ops.py` | `sgemm_lora_a_fwd` / `sgemm_lora_b_fwd`：按段(weight_indices/seg_len)对各适配器执行 LoRA A、B 矩阵乘的 PyTorch 实现。 |
