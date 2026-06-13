# srt/hardware_backend/npu/quantization

## 目录用途
本目录实现昇腾 NPU 的量化计算方法，覆盖线性层与融合 MoE 的多种量化格式（W8A8、W4A4、W4A8、W4A16 等），基于 SGLang 的 `LinearMethodBase`/`FusedMoEMethodBase` 抽象，并结合 NPU 格式转换（NZ）与权重后处理优化。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `linear_method_npu.py` | NPU 线性层量化方法，基类 `_NPULinearMethodBase` 及 `NPUW8A8Int8LinearMethod`、`NPUW8A8Int8DynamicLinearMethod`、`NPU_W4A4DynamicLinearMethod`，含加载后权重转置/格式处理。 |
| `fused_moe_method_npu.py` | NPU 融合 MoE 量化方法，提供 `npu_fused_experts*`/`fused_moe_npu` 等专家计算函数与 `NPUW4A4Int4`/`W8A8Int8`/`W4A8Int8`/`W4A16Int4` 等动态 MoE 方法类。 |
