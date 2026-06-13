# srt/observability

## 目录用途
本目录实现 SGLang 的可观测性能力，包括 Prometheus 指标采集与导出、请求各阶段耗时统计、分布式链路追踪（OpenTelemetry/OTLP）、函数与启动阶段的耗时计时，以及 CPU 监控等。它为调度器、分词器、API Server 等组件提供统一的指标与追踪基础设施。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `metrics_collector.py` | 核心指标采集器：`SchedulerStats`、`SchedulerMetricsCollector`、`TokenizerMetricsCollector`、存储/专家分发/Radix 缓存等多种 Prometheus 指标收集器及直方图配置 |
| `scheduler_metrics_mixin.py` | `SchedulerMetricsMixin` 及 `PrefillStats`/`KvMetrics`，为调度器混入指标统计与 KV 事件发布能力 |
| `req_time_stats.py` | 请求阶段耗时统计：阶段定义、跨线程时间换算，以及 API Server / DPController / Scheduler 各自的 `ReqTimeStats` 实现 |
| `request_metrics_exporter.py` | 请求级指标导出器：抽象基类、文件导出器 `FileRequestMetricsExporter`、管理器及工厂函数 |
| `trace.py` | 基于 OpenTelemetry 的分布式链路追踪：trace header 提取、OTLP exporter 初始化、线程/切片上下文与自定义 ID 生成 |
| `func_timer.py` | 函数延迟计时装饰器 `time_func_latency` 及开关，用于统计被装饰函数的执行耗时 |
| `startup_func_log_and_timer.py` | 启动阶段计时工具：`startup_timer` 上下文管理器、指标记录与最大耗时跟踪 |
| `cpu_monitor.py` | `start_cpu_monitor_thread`，启动后台线程周期性采集进程 CPU 占用 |
| `label_transform.py` | 指标标签转换工具，如 `transform_priority` 将优先级数值映射为标签字符串 |
| `utils.py` | 指标直方图分桶辅助函数：双侧指数分桶、`exponential_buckets` 等 |
