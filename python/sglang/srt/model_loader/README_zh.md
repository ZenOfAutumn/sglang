# srt/model_loader

## 目录用途
该目录负责模型权重的下载、选择与加载，是 SGLang 从磁盘/远端到 GPU 显存的权重装载入口。它根据加载配置选择合适的加载器(默认、分片、量化、GGUF、远端实例、流式等)，识别模型架构，并提供权重迭代器与各类权重加载函数。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 导出 `get_model`、`get_model_loader` 等入口，根据配置创建加载器并加载模型。 |
| `loader.py` | 各类模型加载器实现：`BaseModelLoader` 及 Default、Layered、QuantizedRL、Dummy、ShardedState、BitsAndBytes、GGUF、RemoteInstance、Remote、ModelOpt、RunaiModelStreamer 等，及 `get_model_loader` 选择逻辑。 |
| `utils.py` | 模型架构选择与加载辅助：`get_model_architecture`、`resolve_transformers_arch`、`set_default_torch_dtype`、MoE/序列分类判定、加载后处理等。 |
| `weight_utils.py` | 权重下载与初始化工具：HF 下载、safetensors/pt/gguf 权重迭代器、`default_weight_loader`、KV cache scale 加载、量化配置获取等。 |
| `ci_weight_validation.py` | 仅 CI 环境使用的权重校验与缓存清理:safetensors 校验、缺失分片检查、损坏文件清理、下载重试、离线模式启用等。 |
| `remote_instance_weight_loader_utils.py` | 远端实例权重加载工具：发送/接收权重通信组初始化、传输引擎信息获取、内存区域注册(`RemoteInstanceWeightLoaderBackend`)。 |
