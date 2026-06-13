# srt/debug_utils/comparator

## 目录用途
本目录是功能完整的张量 bundle 比对包（可通过 `python -m sglang.srt.debug_utils.comparator` 运行），负责把两个转储目录中的张量按 key 配对、对齐（unshard、reorder、token 对齐、轴对齐）后逐张量做数值比对，并以文本/可视化形式输出报告。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化，重建 pydantic 模型（`ComparisonTensorRecord`）以解析前向引用。 |
| `__main__.py` | 模块入口，调用 `entrypoint.main()`。 |
| `bundle_comparator.py` | 比对一对张量 bundle：先规划并执行对齐计划，再逐张量比较并产出比对记录。 |
| `bundle_matcher.py` | 按元数据 key（排除 skip_keys）将基准侧与目标侧的张量文件配对成多组 bundle。 |
| `display.py` | 收集并发出展示性记录（rank 信息、input_ids/positions 等），渲染为文本。 |
| `dp_utils.py` | DP 过滤工具：当 dp_size>1 时仅保留非空 dp_rank 的张量项。 |
| `entrypoint.py` | 命令行入口，解析参数、加载元数据、匹配 bundle、驱动整体比对流程。 |
| `log_sink.py` | 日志收集器 `LogSink`，以上下文栈方式聚合比对过程中的 info/error 日志。 |
| `meta_overrider.py` | 按 YAML 规则（正则匹配张量名）覆盖元数据字段（当前主要是 `dims`），无需重跑转储。 |
| `output_formatter.py` | 比对输出记录的格式化逻辑（文本与 rich 渲染），从 `output_types` 中分离。 |
| `output_types.py` | 定义各类比对输出记录的数据结构（张量/非张量/跳过/错误/汇总/配置/日志等）。 |
| `per_token_visualizer.py` | 生成 per-token 相对误差热力图 PNG（行=张量名，列=token 位置）。 |
| `preset.py` | 定义命令行预设（raw / sglang_dev / sglang_megatron）并展开 `--preset` 参数。 |
| `report_sink.py` | 统一的报告输出端 `ReportSink`，控制输出格式、详尽度与落盘文件。 |
| `utils.py` | 通用工具：`Pair` 泛型容器、pydantic 基类、相对误差计算、目录自动下钻等。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `aligner/` | 对齐子系统：unshard、reorder、token 对齐、轴对齐的规划与执行。 |
| `dims_spec/` | dims 字符串的解析与张量维度命名工具（含并行修饰符语法）。 |
| `tensor_comparator/` | 单张量对的数值比对与差异统计、结果格式化。 |
| `visualizer/` | 张量对比的多面板可视化图（热力图、直方图、散点等）。 |
