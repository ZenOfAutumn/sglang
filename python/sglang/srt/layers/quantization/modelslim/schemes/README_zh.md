# srt/layers/quantization/modelslim/schemes

## 目录用途
本目录实现 ModelSlim 框架支持的各类量化 scheme。每个 scheme 继承自 `ModelSlimLinearScheme` 或 `ModelSlimMoEScheme`，封装某种 INT4/INT8 量化格式的权重创建与前向计算，底层调用昇腾 NPU 的量化线性/MoE 方法。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 汇总导出本目录所有 scheme 类 |
| `modelslim_scheme.py` | scheme 抽象基类 `ModelSlimLinearScheme`/`ModelSlimMoEScheme` |
| `modelslim_w4a4_int4.py` | W4A4 INT4 动态线性 scheme（`ModelSlimW4A4Int4`，NPU） |
| `modelslim_w4a4_int4_moe.py` | W4A4 INT4 动态 MoE scheme（`ModelSlimW4A4Int4MoE`，NPU） |
| `modelslim_w4a8_int8_moe.py` | W4A8 INT8 动态 MoE scheme（`ModelSlimW4A8Int8MoE`，NPU） |
| `modelslim_w8a8_int8.py` | W8A8 INT8 线性 scheme（`ModelSlimW8A8Int8`，支持静态/动态，NPU） |
| `modelslim_w8a8_int8_moe.py` | W8A8 INT8 动态 MoE scheme（`ModelSlimW8A8Int8MoE`，NPU） |
