# srt/debug_utils

## 目录用途
本目录是 SGLang 的调试工具集合，提供推理过程中张量转储（dump）、跨实现/跨框架数值比对、日志解析、调度模拟、CUDA 崩溃取证以及运行时源码热补丁等能力，主要服务于精度对齐、性能分析与故障排查。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 空的包初始化文件，标识 `debug_utils` 为 Python 包。 |
| `cuda_coredump.py` | 在 `SGLANG_CUDA_COREDUMP=1` 时注入 CUDA coredump 环境变量，使 GPU 异常产生轻量级 coredump 供 cuda-gdb 事后分析。 |
| `dump_comparator.py` | 单文件简化版转储比对脚本，逐张量比较两个 dump 目录；高级功能见 `comparator/` 包。 |
| `dump_loader.py` | 转储数据加载器，从文件名解析元数据、加载 `.pt` 文件为 `ValueWithMeta`，并提供按条件过滤行的工具。 |
| `dumper.py` | 张量转储核心实现，定义配置基类与 Dumper，将前向过程中的张量及元信息（含并行信息）落盘。 |
| `log_parser.py` | 用正则解析推理日志中的 Decode batch 行，提取吞吐、token 用量等指标为 polars DataFrame。 |
| `model_truncator.py` | 将 HuggingFace 模型按层/维度截断为小模型（含 safetensors 与 config 改写），便于调试。 |
| `tensor_dump_forward_hook.py` | 为模型每个算子注册前向 hook，逐次前向把所有中间张量按 rank 落盘为 `.pt` 文件。 |
| `text_comparator.py` | 比对基准测试文本输出（如 lm_eval / gsm8k / mmlu 结果），查找差异。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `comparator/` | 功能完整的张量 bundle 比对包，支持 unshard、token 对齐、按维度标注与可视化。 |
| `schedule_simulator/` | 请求调度模拟器，分析多 GPU 上不同路由/调度策略的负载均衡性。 |
| `source_patcher/` | 运行时源码热补丁工具，按 YAML 配置对函数源码做匹配替换以注入调试逻辑。 |
