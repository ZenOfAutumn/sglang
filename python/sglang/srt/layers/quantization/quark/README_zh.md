# srt/layers/quantization/quark

## 目录用途
本目录适配 AMD Quark 量化框架。`QuarkConfig` 解析 Quark 导出的量化配置，按 fnmatch/正则匹配层名并为各层选择对应 scheme，提供 Linear/MoE/KVCache 量化方法，主要面向 ROCm/AITER 后端的 FP8 与 MXFP4 量化。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化（空） |
| `quark.py` | `QuarkConfig` 配置解析及 `QuarkLinearMethod`/`QuarkFusedMoEMethod`/`QuarkKVCacheMethod` |
| `utils.py` | 辅助工具：配置深比较、层忽略/正则匹配、MXFP4 动态量化与反量化等 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `schemes` | 各 Quark 量化 scheme（W8A8 FP8、W4A4 MXFP4 的线性与 MoE）实现 |
