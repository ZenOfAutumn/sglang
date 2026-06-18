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

## 各加载器功能说明

所有加载器继承自 `BaseModelLoader`（定义 `download_model` / `load_model` 抽象接口）。`get_model_loader` 会依据 `LoadConfig.load_format` 选择对应实现。

| 加载器 | 基类 | 功能 | 适用场景 |
| --- | --- | --- | --- |
| `BaseModelLoader` | `ABC` | 加载器抽象基类，定义统一的下载与加载接口。 | 不直接使用，供子类继承。 |
| `DefaultModelLoader` | `BaseModelLoader` | 默认加载器，从本地磁盘/HF/ModelScope 下载并加载 `safetensors`/`pt`/`bin` 等格式权重；支持多线程权重加载、MTP 权重过滤。 | 绝大多数常规模型加载（默认路径）。 |
| `LayeredModelLoader` | `DefaultModelLoader` | 逐层加载权重，加载完一层即可先量化再加载下一层，从而压低峰值显存。 | 显存紧张、需要边加载边量化的场景。 |
| `QuantizedRLModelLoader` | `DefaultModelLoader` | 面向 RL 训练的免 profile 原生 FP8 量化加载：首次加载基座 → 记录状态 → 应用 FP8 量化，便于后续权重热更新。 | RL 训练中的 FP8 量化与权重在线更新。 |
| `DummyModelLoader` | `BaseModelLoader` | 不读取真实权重，把模型参数填充为随机值。 | 性能基准测试、调试、跑通流程而无需真实权重。 |
| `ShardedStateLoader` | `BaseModelLoader` | 直接加载每个 worker 自己的分片 state dict，无需读取完整 checkpoint。 | 大规模张量并行(TP)模型的快速加载。 |
| `BitsAndBytesModelLoader` | `BaseModelLoader` | 以 BitsAndBytes 进行权重量化加载（读取 `adapter_config.json`）。 | 4bit/8bit BitsAndBytes 量化模型。 |
| `GGUFModelLoader` | `BaseModelLoader` | 加载 GGUF 格式权重，支持完整模型与分片模型。 | 加载以 GGUF 量化/保存的模型。 |
| `RemoteInstanceModelLoader` | `BaseModelLoader` | 通过通信组从另一个运行中的 SGLang 实例直接拉取张量权重。 | 实例间权重直传，免落盘冷启动。 |
| `RemoteModelLoader` | `BaseModelLoader` | 从远端数据库/对象存储(如 S3 connector)加载张量权重。 | 权重集中存储于远端 KV/对象存储的场景。 |
| `ModelOptModelLoader` | `DefaultModelLoader` | 在默认加载基础上应用 NVIDIA Model Optimizer(ModelOpt)量化。 | 使用 NVIDIA ModelOpt 量化的模型。 |
| `RunaiModelStreamerLoader` | `BaseModelLoader` | 使用 Run:ai Model Streamer 边流式传输边加载权重。 | 从 SSD/共享文件系统/对象存储(S3、GCS、Azure)高速流式加载。 |
