# srt/layers/quantization/modelslim

## 目录用途
本目录适配华为 ModelSlim 量化框架（主要面向昇腾 NPU）。`ModelSlimConfig` 解析量化配置并为各层选择 INT4/INT8 的线性或 MoE scheme，权重创建与前向计算委托给底层 NPU 量化方法实现。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `modelslim.py` | `ModelSlimConfig` 配置解析及 `ModelSlimLinearMethod`/`ModelSlimFusedMoEMethod`（含 NPU RMSNorm 适配封装） |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `schemes` | 各 ModelSlim 量化 scheme（W4A4/W4A8/W8A8 INT 的线性与 MoE）实现 |
