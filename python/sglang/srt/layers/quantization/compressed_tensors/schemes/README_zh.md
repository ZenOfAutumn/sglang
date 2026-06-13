# srt/layers/quantization/compressed_tensors/schemes

## 目录用途
本目录实现 compressed-tensors 框架支持的各类量化 scheme。每个 scheme 继承自 `CompressedTensorsLinearScheme` 或 `CompressedTensorsMoEScheme`，封装某种具体量化格式（按权重位宽/激活位宽与张量布局区分）的权重创建、加载后处理与前向计算逻辑，由上层 `CompressedTensorsConfig` 按层匹配后调用。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 汇总导出本目录所有 scheme 类 |
| `compressed_tensors_scheme.py` | scheme 抽象基类 `CompressedTensorsLinearScheme`/`CompressedTensorsMoEScheme` |
| `compressed_tensors_w4a4_mxint4_moe.py` | W4A4 MXINT4 MoE scheme（`CompressedTensorsMxInt4MoE`） |
| `compressed_tensors_w4a4_nvfp4.py` | W4A4 NVFP4 线性 scheme（`CompressedTensorsW4A4Fp4`） |
| `compressed_tensors_w4a4_nvfp4_moe.py` | W4A4 NVFP4 MoE scheme（`CompressedTensorsW4A4Nvfp4MoE`，cutlass MoE） |
| `compressed_tensors_w4a8_int8_moe.py` | W4A8 INT8 动态 MoE scheme（NPU 实现） |
| `compressed_tensors_w8a16_fp8.py` | W8A16 FP8 线性 scheme（`CompressedTensorsW8A16Fp8`） |
| `compressed_tensors_w8a8_fp8.py` | W8A8 FP8 线性 scheme（`CompressedTensorsW8A8Fp8`，支持 per-tensor/channel/block） |
| `compressed_tensors_w8a8_fp8_moe.py` | W8A8 FP8 MoE scheme（`CompressedTensorsW8A8Fp8MoE`） |
| `compressed_tensors_w8a8_int8.py` | W8A8 INT8 线性 scheme（`CompressedTensorsW8A8Int8` 及 NPU 变体） |
| `compressed_tensors_w8a8_int8_moe.py` | W8A8 INT8 动态 MoE scheme（NPU 实现） |
| `compressed_tensors_wNa16.py` | WNA16（权重 N bit、激活 16bit）线性 scheme（`CompressedTensorsWNA16`，Marlin） |
| `compressed_tensors_wNa16_moe.py` | WNA16 MoE scheme（Marlin/Triton 及 NPU W4A16 变体） |
