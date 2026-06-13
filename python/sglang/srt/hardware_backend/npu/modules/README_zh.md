# srt/hardware_backend/npu/modules

## 目录用途
本目录存放针对昇腾 NPU 的模型模块级适配，将特定模型结构的前向计算替换为 NPU 优化实现或对第三方处理器打补丁，目前覆盖 DeepSeek-V2 的 MLA 注意力与 Qwen-VL 的图像预处理。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `deepseek_v2_attention_mla_npu.py` | DeepSeek-V2 MLA 注意力的 NPU 前向实现，提供 MHA/MLA/DSA 的 prepare 与 core 系列函数及 `npu_mla_preprocess`，结合融合 QK norm 与 MLA 预处理优化。 |
| `qwen_vl_processor.py` | Qwen-VL 图像处理器的 NPU 补丁，封装 `npu_wrapper_preprocess` 并通过 `npu_apply_qwen_image_preprocess_patch` 替换 transformers 的快速预处理流程。 |
