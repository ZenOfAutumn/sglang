# srt/models/deepseek_common

## 目录用途
本目录存放 DeepSeek 系列模型（DeepSeek V2/V3/V3.2、NextN/MTP 等）共享的公共组件，包括注意力后端分发、权重加载与设备/量化能力检测等。这些工具被 `models/deepseek_v2.py`、`models/deepseek_nextn.py` 等模型实现复用，以避免重复代码。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包标识文件（空），将本目录声明为 Python 包。 |
| `attention_backend_handler.py` | 注意力后端注册与分发逻辑（`AttentionBackendRegistry`、MLA/MHA 子类型分发），根据后端名称与运行模式选择对应的 `AttnForwardMethod`。 |
| `deepseek_weight_loader.py` | DeepSeek 模型的权重加载器，处理 MoE 专家权重、FP8 量化权重等的并发加载与映射。 |
| `utils.py` | 设备与量化能力检测的公共工具（`_is_hip`、`_is_cuda`、`_is_fp8_fnuz` 等标志位及相关辅助函数）。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `attention_forward_methods` | DeepSeek 注意力前向计算的实现集合，含 MHA/MLA 各 Mixin 与前向方法枚举。 |
