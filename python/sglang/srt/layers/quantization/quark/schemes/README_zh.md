# srt/layers/quantization/quark/schemes

## 目录用途
本目录实现 AMD Quark 框架支持的各类量化 scheme。每个 scheme 继承自 `QuarkLinearScheme` 或 `QuarkMoEScheme`，封装 FP8 或 MXFP4 量化格式的权重创建、加载后处理与前向计算，主要面向 ROCm/AITER 内核。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 汇总导出本目录所有 scheme 类 |
| `quark_scheme.py` | scheme 抽象基类 `QuarkLinearScheme`/`QuarkMoEScheme` |
| `quark_w4a4_mxfp4.py` | W4A4 MXFP4 线性 scheme（`QuarkW4A4MXFP4`，HIP/AITER） |
| `quark_w4a4_mxfp4_moe.py` | W4A4 MXFP4 MoE scheme（`QuarkW4A4MXFp4MoE`） |
| `quark_w8a8_fp8.py` | W8A8 FP8 线性 scheme（`QuarkW8A8Fp8`，支持 per-tensor/channel） |
| `quark_w8a8_fp8_moe.py` | W8A8 FP8 MoE scheme（`QuarkW8A8FP8MoE`） |
