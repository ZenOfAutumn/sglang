# srt/layers/quantization/compressed_tensors

## 目录用途
本目录适配 compressed-tensors 量化框架，解析其 config_groups 与 ignore 规则，并据此为各层分发到合适的量化 scheme。`CompressedTensorsConfig` 负责整体配置解析与 Linear/MoE 方法构造，具体的权重创建与前向计算委托给 `schemes` 子目录下的各 scheme 类。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `compressed_tensors.py` | `CompressedTensorsConfig` 配置解析及 `CompressedTensorsLinearMethod`/`CompressedTensorsFusedMoEMethod`，按层选择对应 scheme |
| `utils.py` | 辅助工具：激活量化格式判断、层忽略规则、按正则匹配目标层名等 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `schemes` | 各 compressed-tensors 量化 scheme（W8A8/W4A4/WNA16 等）的具体实现 |
