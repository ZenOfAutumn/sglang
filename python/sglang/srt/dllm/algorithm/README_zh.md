# srt/dllm/algorithm

## 目录用途
本目录存放扩散式 LLM（dLLM）的去噪/揭示算法。每种算法定义在一步前向后如何根据 logits 决定哪些被 mask 的 token 被确定下来（解码），并支持通过插件式注册按名称选择具体策略。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 算法注册与发现：`import_algorithms` 自动扫描本包内含 `Algorithm` 的模块并建立映射，`get_algorithm` 按 `DllmConfig` 返回对应算法实例。 |
| `base.py` | `DllmAlgorithm` 基类，读取块大小与 mask id，并提供 `from_server_args` 工厂以委托给 `get_algorithm`。 |
| `joint_threshold.py` | `JointThreshold` 算法，基于联合阈值确定揭示 token，并支持带 `edit_threshold` 的后编辑步骤。 |
| `low_confidence.py` | `LowConfidence` 算法，基于置信度阈值揭示高置信 token 的去噪策略。 |
