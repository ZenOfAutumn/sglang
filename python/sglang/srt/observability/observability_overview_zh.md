# srt/observability 模块总览

## 目录用途
本目录实现 SGLang 的可观测性能力，包括 Prometheus 指标采集与导出、请求各阶段耗时统计、分布式链路追踪（OpenTelemetry/OTLP）、函数与启动阶段的耗时计时，以及 CPU 监控等。它为调度器、分词器、API Server 等组件提供统一的指标与追踪基础设施。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `metrics_collector.py` | 核心指标采集器：`SchedulerStats`、`SchedulerMetricsCollector`、`TokenizerMetricsCollector`、存储/专家分发/Radix 缓存等多种 Prometheus 指标收集器及直方图配置 |
| `forward_pass_metrics.py` | 每次调度迭代的实时指标：`ForwardPassMetrics`（msgspec 结构体，含已调度/排队请求统计），经 `_FpmPublisherThread` 后台线程通过 ZMQ PUB 推送给外部消费者（如 Dynamo），并附 Welford 在线方差累加器与空闲心跳机制 |
| `req_time_stats.py` | 请求阶段耗时统计：阶段定义、跨线程时间换算，以及 API Server / DPController / Scheduler 各自的 `ReqTimeStats` 实现 |
| `request_metrics_exporter.py` | 请求级指标导出器：抽象基类、文件导出器 `FileRequestMetricsExporter`、管理器及工厂函数 |
| `trace.py` | 基于 OpenTelemetry 的分布式链路追踪：trace header 提取、OTLP exporter 初始化、线程/切片上下文与自定义 ID 生成 |
| `mooncake_trace.py` | Mooncake KV 传输链路追踪：定义 `MooncakeRequestStage` 各传输阶段，并提供 `mooncake_trace_slice` / `mooncake_trace_func` 装饰器将 send/recv 阶段记录到 trace 切片 |
| `func_timer.py` | 函数延迟计时装饰器 `time_func_latency` 及开关，用于统计被装饰函数的执行耗时 |
| `startup_func_log_and_timer.py` | 启动阶段计时工具：`startup_timer` 上下文管理器、指标记录与最大耗时跟踪 |
| `cpu_monitor.py` | `start_cpu_monitor_thread`，启动后台线程周期性采集进程 CPU 占用 |
| `label_transform.py` | 指标标签转换工具，如 `transform_priority` 将优先级数值映射为标签字符串 |
| `utils.py` | 指标直方图分桶辅助函数：双侧指数分桶、`exponential_buckets` 等 |

## 可观测内容详解

### 1. 调度器 Prometheus 指标（`SchedulerMetricsCollector`）
由 Scheduler 进程周期性上报，指标前缀均为 `sglang:`。类型说明：Gauge=瞬时值，Counter=单调累计值（适合 `rate()`），Histogram=分布直方图。

#### 1.1 运行状态与吞吐
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `num_running_reqs` | Gauge | 当前正在运行（已进入 batch 参与 forward）的请求数 |
| `num_queue_reqs` | Gauge | 等待队列中的请求数，持续增长说明供给能力不足 |
| `num_grammar_queue_reqs` | Gauge | 等待 grammar（结构化输出约束）编译完成的请求数 |
| `gen_throughput` | Gauge | 生成吞吐（token/s），最近一个统计窗口内的解码速度 |
| `cache_hit_rate` | Gauge | 前缀缓存（RadixCache）命中率，命中的 prompt token 无需重新计算 |
| `decode_sum_seq_lens` | Gauge | decode 中所有序列长度之和，反映 decode 批次的 KV 读取量 |
| `utilization` | Gauge | 综合利用率，由 token 占用与请求并发推导，衡量引擎繁忙程度 |
| `fwd_occupancy` | Gauge | forward pass 的 GPU 占用百分比（GPU 实际执行 forward 的时间占比） |
| `new_token_ratio` | Gauge | 调度器为 decode 预留新 token 空间的估算比例，用于防止 KV 池耗尽导致回退（retract） |

#### 1.2 KV Cache / 显存用量
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `token_usage` | Gauge | KV 池整体 token 占用率（0~1），接近 1 时会触发请求回退 |
| `full_token_usage` | Gauge | 全注意力（full attention）层的 token 占用率（混合注意力模型下与 SWA 分开统计） |
| `swa_token_usage` | Gauge | SWA（滑动窗口注意力）层的 token 占用率 |
| `mamba_usage` | Gauge | Mamba（SSM 状态空间）层的状态槽占用率 |
| `num_used_tokens` | Gauge | 已占用的 KV token 槽位绝对数量 |
| `kv_available_tokens` | Gauge | KV 池中空闲 token 槽位数 |
| `kv_evictable_tokens` | Gauge | 可驱逐的 token 槽位数（被 RadixCache 缓存但无请求引用，需要时可回收） |
| `kv_used_tokens` | Gauge | 被活跃请求实际占用的 token 槽位数 |
| `swa_available/evictable/used_tokens` | Gauge | 同上三项，针对 SWA 池（仅混合 SWA 模型） |
| `mamba_available/evictable/used_tokens` | Gauge | 同上三项，针对 Mamba SSM 状态池（仅混合 SSM 模型） |

三者关系：`available + evictable + used ≈ 池总容量`，`evictable` 高说明缓存复用潜力大。

#### 1.3 投机解码
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `spec_accept_length` | Gauge | 平均接受长度 = 每次 forward 被接受的草稿 token 数 + 1 个 bonus token，是投机解码加速比的直接指标 |
| `spec_accept_rate` | Gauge | 接受率 = 批内被接受草稿数 / 提出草稿数，反映 draft 模型质量 |
| `spec_num_steps` | Gauge | 当前生效的投机步数（`speculative_num_steps`，可能被动态调整） |
| `spec_num_draft_tokens` | Gauge | 当前生效的草稿 token 数（topk>1 时与步数解耦） |
| `spec_verify_calls_total`（Tokenizer 侧） | Counter | 投机解码验证（verify）调用总次数 |

#### 1.4 请求回退与暂停
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `num_retracted_reqs` | Gauge | 当前统计窗口内被回退的请求数。回退指 KV 池不足时将运行中请求踢回等待队列并释放其 KV |
| `num_retracted_requests_total` | Counter | 累计回退请求总数，非零且持续增长说明显存压力大 |
| `num_retracted_input_tokens_total` / `num_retracted_output_tokens_total` | Counter | 回退时被丢弃的输入/已生成 token 累计数，代表被浪费的计算量 |
| `num_paused_reqs` | Gauge | 因异步权重同步（RLHF 场景 update_weights）而暂停的请求数 |

#### 1.5 PD 分离（Prefill/Decode Disaggregation）
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `num_prefill_bootstrap_queue_reqs` | Gauge | Prefill 侧等待与 decode 侧建立连接（bootstrap 握手）的请求数 |
| `num_prefill_inflight_queue_reqs` | Gauge | Prefill 已完成、KV 正在向 decode 侧传输中的请求数 |
| `num_decode_prealloc_queue_reqs` | Gauge | Decode 侧等待预分配 KV 空间的请求数 |
| `num_decode_transfer_queue_reqs` | Gauge | Decode 侧等待 KV 传输完成的请求数 |
| `pending_prealloc_token_usage` | Gauge | 待预分配（尚未真正分配）token 的占用率 |
| `kv_transfer_speed_gb_s` | Histogram | 单次 KV 传输速度分布（GB/s），衡量 RDMA/网络带宽利用 |
| `kv_transfer_latency_ms` | Histogram | 单次 KV 传输延迟分布（ms），直接影响 PD 分离下的 TTFT |
| `kv_transfer_bootstrap_ms` | Histogram | Bootstrap 握手耗时分布（ms） |
| `kv_transfer_alloc_ms` | Histogram | 等待 decode 侧分配 KV 空间的耗时分布（ms） |
| `kv_transfer_total_mb` | Histogram | 单次 KV 传输数据量分布（MB），与序列长度成正比 |
| `num_bootstrap_failed_reqs_total` | Counter | Bootstrap 握手失败的请求累计数 |
| `num_transfer_failed_reqs_total` | Counter | KV 传输失败的请求累计数 |
| `num_prefill_retries_total` | Counter | Prefill 重试累计次数 |

#### 1.6 CUDA Graph
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `is_cuda_graph` | Gauge | 当前批次是否命中 CUDA Graph 回放（1/0）。命中可消除 kernel launch 开销 |
| `cuda_graph_passes_total` | Counter | 按 `mode` 标签（命中/未命中）分类的 forward pass 累计数，可算出 CUDA Graph 命中率 |

#### 1.7 LoRA
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `lora_pool_slots_used` | Gauge | GPU 显存中已被 LoRA adapter 占用的槽位数 |
| `lora_pool_slots_total` | Gauge | LoRA 槽位总数（`max_loras_per_batch`） |
| `lora_pool_utilization` | Gauge | LoRA 池利用率（used/total），1.0 表示池满，新 adapter 需换入换出 |

#### 1.8 HiCache 与流式会话
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `hicache_host_used_tokens` | Gauge | Host（CPU）内存 KV 缓存已使用的 token 数 |
| `hicache_host_total_tokens` | Gauge | Host KV 缓存总容量（token 数） |
| `num_streaming_sessions` | Gauge | 活跃的流式会话数 |
| `streaming_session_held_tokens` | Gauge | 流式会话槽位持有的 KV token 数（会话保持期间 KV 不释放） |

#### 1.9 路由键（cache-aware 路由）
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `num_unique_running_routing_keys` | Gauge | 运行批次中不同 routing key 的数量，反映负载多样性 |
| `routing_key_running_req_count` | 分桶分布 | 按运行中请求数对 routing key 分桶的分布（gt < count <= le） |
| `routing_key_all_req_count` | 分桶分布 | 按运行+等待请求数对 routing key 分桶的分布 |

#### 1.10 请求延迟
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `queue_time_seconds` | Histogram | 请求在等待队列中的排队耗时分布 |
| `per_stage_req_latency_seconds` | Histogram | 按 `stage` 标签区分的各阶段延迟分布（阶段定义见第 4 节），指数分桶覆盖 1ms~约 1191s |

#### 1.11 结构化输出 / Grammar
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `grammar_compilation_time_seconds` | Histogram | grammar（JSON schema/EBNF/正则）编译耗时分布，编译在后台线程进行 |
| `num_grammar_cache_hit_total` | Counter | grammar 编译缓存命中累计数（相同 schema 复用编译结果） |
| `num_grammar_aborted_total` | Counter | grammar 编译被中止的请求累计数 |
| `num_grammar_timeout_total` | Counter | grammar 编译超时累计数 |
| `num_grammar_total` | Counter | 使用 grammar 的请求总数 |
| `grammar_schema_count` | Histogram | 单个请求中 schema 数量的分布 |
| `grammar_ebnf_size` | Histogram | EBNF 文法文本大小的分布 |
| `grammar_tree_traversal_time_avg` / `_max` | Histogram | 解码期间 grammar 状态机树遍历（约束下一 token 掩码计算）的平均/最大耗时分布，过大会拖慢 decode |

#### 1.12 算力与带宽估算（MFU 计算）
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `realtime_tokens_total` | Counter | 处理的 token 累计数，`mode` 标签区分 prefill_compute（实际计算）/ prefill_cache（缓存命中）/ decode |
| `forward_execution_seconds_total` | Counter | GPU 执行 forward 的累计繁忙时间，按 ForwardMode 分类 |
| `estimated_flops_per_gpu_total` | Counter | 每 GPU 估算浮点运算累计量，与理论峰值相除可得 MFU（Model FLOPs Utilization） |
| `estimated_read_bytes_per_gpu_total` / `estimated_write_bytes_per_gpu_total` | Counter | 每 GPU 估算显存读/写字节累计量，用于估算显存带宽利用率（MBU） |
| `dp_cooperation_realtime_tokens_total` / `dp_cooperation_forward_execution_seconds_total` | Counter | 上述 token/时间指标的 DP 协同版本，标签额外标注 DP rank 间协作状态（如本 rank 是否空转陪跑） |

#### 1.13 Prefill Delayer
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `prefill_delayer_wait_forward_passes` | Histogram | prefill 被延迟器推迟的 forward pass 次数分布 |
| `prefill_delayer_wait_seconds` | Histogram | prefill 被推迟的等待时长分布 |
| `prefill_delayer_outcomes_total` | Counter | 延迟器决策结果计数（按结果标签分类，如放行/超时放行等） |

#### 1.14 EPLB
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `eplb_balancedness` | Gauge | MoE 专家并行的负载均衡度（按 `forward_mode` 分标签），1 表示各 GPU 专家负载完全均衡，越低说明热点专家越集中 |

#### 1.15 引擎静态信息（启动时上报一次）
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `max_total_num_tokens` | Gauge | KV 池最大 token 容量，由显存大小与模型结构决定 |
| `max_running_requests_under_SLO` | Gauge | SLO 约束下允许的最大并发请求数 |
| `engine_startup_time` | Gauge | 引擎启动总耗时（秒） |
| `engine_load_weights_time` | Gauge | 权重加载耗时（秒） |
| `weight_load_duration_seconds` | Gauge | 最近一次 `update_weights_from_*` 调用的耗时，`source` 标签为 disk/tensor/distributed 等 |
| `page_size` | Gauge | KV 缓存页大小（token 数/页） |
| `num_pages` | Gauge | KV 缓存页总数 |
| `context_len` | Gauge | 最大上下文长度 |
| `startup_available_gpu_memory_gb` | Gauge | 启动时可用 GPU 显存（GB） |

### 2. Tokenizer / API 层指标（`TokenizerMetricsCollector`）
面向请求生命周期的用户侧指标，直方图分桶可通过环境变量自定义（`get_histogram_conf_from_env`）。

| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `prompt_tokens_total` | Counter | 处理的 prompt（prefill）token 累计数，`rate()` 即输入吞吐 |
| `generation_tokens_total` | Counter | 生成 token 累计数，`rate()` 即输出吞吐 |
| `cached_tokens_total` | Counter | 命中缓存的 prompt token 累计数，`cache_source` 标签区分来源：device（GPU RadixCache）/ host（CPU HiCache）/ storage（远端存储） |
| `num_requests_total` | Counter | 处理完成的请求总数 |
| `num_so_requests_total` | Counter | 结构化输出（structured output）请求总数 |
| `num_aborted_requests_total` | Counter | 被中止（客户端断连、超时、主动 abort）的请求总数 |
| `spec_verify_calls_total` | Counter | 投机解码验证调用总次数 |
| `prompt_tokens_histogram` | Histogram | 单请求 prompt 长度分布 |
| `uncached_prompt_tokens_histogram` | Histogram | 单请求未命中缓存（需实际计算）的 prompt 长度分布，与上一项对比可见缓存收益 |
| `generation_tokens_histogram` | Histogram | 单请求生成长度分布 |
| `time_to_first_token_seconds` | Histogram | TTFT：从请求到达至首 token 返回的耗时分布，主要由排队+prefill 决定 |
| `inter_token_latency_seconds` | Histogram | ITL：相邻输出 token 之间的间隔分布，反映 decode 每步耗时 |
| `e2e_request_latency_seconds` | Histogram | 端到端请求延迟分布（从到达至完成） |
| `get_loads_duration_seconds` | Histogram | 处理 `/v1/loads` 负载查询接口的耗时分布 |

### 3. 其他专项收集器

#### `StorageMetricsCollector`（HiCache 三级缓存的存储层）
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `prefetched_tokens_total` | Counter | 从存储层预取（storage→host/device）的 prompt token 累计数 |
| `backuped_tokens_total` | Counter | 备份到存储层（host→storage）的 token 累计数 |
| `prefetch_pgs` / `backup_pgs` | Histogram | 每批次预取/备份的页数分布 |
| `prefetch_bandwidth` / `backup_bandwidth` | Histogram | 预取/备份带宽分布（GB/s），衡量存储 I/O 性能 |

#### `ExpertDispatchCollector`（MoE 专家分发）
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `eplb_gpu_physical_count` | Histogram | 各层、各 GPU rank 上被选中的物理专家数量分布（`layer` 标签），观察专家激活的层间差异 |

#### `RadixCacheMetricsCollector`（Radix 前缀缓存换入换出）
| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `eviction_duration_seconds` | Histogram | KV 从 GPU 驱逐到 CPU 的耗时分布 |
| `evicted_tokens_total` | Counter | 从 GPU 驱逐到 CPU 的 token 累计数 |
| `load_back_duration_seconds` | Histogram | KV 从 CPU 加载回 GPU 的耗时分布 |
| `load_back_tokens_total` | Counter | 从 CPU 加载回 GPU 的 token 累计数 |

### 4. 请求阶段耗时统计（`req_time_stats.py`）
`RequestStage` 定义了请求在各组件中经历的阶段，用于 `per_stage_req_latency_seconds` 直方图与链路追踪切片：

- **入口层**：`TOKENIZE`、`API_SERVER_DISPATCH`、`DPC_DISPATCH`（DP Controller 分发）、`REQUEST_PROCESS`。
- **常规推理**：`PREFILL_WAITING`、`PREFILL_FORWARD`、`PREFILL_CHUNKED_FORWARD`、`DECODE_FORWARD`、`DECODE_LOOP`、`MM_ENCODE`（多模态编码）、`RUN_BATCH_CPU`。
- **PD 分离专属**：prefill 侧 `PREFILL_PREPARE` / `PREFILL_BOOTSTRAP` / `PREFILL_TRANSFER_KV_CACHE`；decode 侧 `DECODE_PREPARE` / `DECODE_BOOTSTRAP` / `DECODE_WAITING` / `DECODE_TRANSFERRED` / `DECODE_FAKE_OUTPUT` / `DECODE_QUICK_FINISH`。
- **投机解码**：`SPEC_DRAFT`、`SPEC_VERIFY`。

按组件分别实现 `APIServerReqTimeStats`、`DPControllerReqTimeStats`、`SchedulerReqTimeStats`、`EncoderReqTimeStats`，并提供 `convert_time_to_realtime_ns` 做跨线程/跨进程的单调时钟到真实时间换算。

### 5. 分布式链路追踪（`trace.py`）
基于 OpenTelemetry，通过 OTLP exporter 将 span 上报到后端（如 Jaeger）：

- `extract_trace_headers` 从 HTTP 请求头提取上游 trace 上下文，实现跨服务串联。
- `process_tracing_init` 初始化每进程的 tracer 与 OTLP exporter；`TraceCustomIdGenerator` 生成稳定的 trace/span ID。
- `TraceReqContext` 为单个请求维护追踪上下文，提供 `trace_slice_start/end` 记录阶段切片；`TraceThreadContext` 记录线程信息；`TraceNullContext` 作为追踪关闭时的空实现。
- `set_global_trace_level` 控制追踪粒度级别（level 越大越细）。
- `mooncake_trace.py` 在此之上补充 Mooncake KV 传输的 send/recv/worker 阶段切片。

### 6. 实时迭代级指标（`forward_pass_metrics.py`）
Prometheus 是拉模式、有采样间隔，无法观察每次调度迭代；本模块通过 ZMQ PUB 主动推送每个 forward pass 的 `ForwardPassMetrics`（msgpack 序列化），包含：已调度请求的 prefill/decode 数量、token 总量与长度方差（Welford 在线算法）、排队请求的同类统计，以及 `wall_time` 迭代耗时。空闲时发送全零心跳。外部消费者（如 Dynamo）经 ZMQ SUB 订阅实现实时观测。

### 7. 请求级指标导出（`request_metrics_exporter.py`）
将单个请求完成后的完整指标记录（而非聚合值）导出到外部：抽象基类 `RequestMetricsExporter` 定义接口，`FileRequestMetricsExporter` 写入文件，`RequestMetricsExporterManager` 统一管理多个导出器，`create_request_metrics_exporters` 按配置构建。适合离线分析每个请求的耗时明细。

### 8. 辅助计时与监控
- **`func_timer.py`**：`time_func_latency` 装饰器统计任意函数（含 async）执行耗时并上报 Prometheus，`enable_func_timer` 全局开关。
- **`startup_func_log_and_timer.py`**：`startup_timer` 上下文管理器与 `time_startup_latency` 装饰器，度量启动各阶段（加载权重、CUDA Graph 捕获等）耗时，`set_startup_metric` 记录并跟踪最大值。
- **`cpu_monitor.py`**：`start_cpu_monitor_thread` 启动后台线程周期性采样进程 CPU 占用率并输出日志，用于发现调度器 CPU 瓶颈。
- **`label_transform.py` / `utils.py`**：标签规范化（如优先级数值转标签）与直方图分桶生成（指数分桶、双侧指数分桶）。
