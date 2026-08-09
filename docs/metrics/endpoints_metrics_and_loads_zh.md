# `/metrics` 与 `/v1/loads` 接口详解：请求与返回值

> 本文逐字段说明 SGLang HTTP Server 两个观测接口的**请求参数**与**返回结构**，所有内容均可溯源到代码：
>
> | 主题 | 代码位置 |
> | --- | --- |
> | 路由注册 | `python/sglang/srt/entrypoints/http_server.py` |
> | `/metrics` 挂载 | `python/sglang/srt/utils/common.py::add_prometheus_middleware` |
> | 指标定义 | `python/sglang/srt/observability/metrics_collector.py` |
> | 指标填充 | `python/sglang/srt/managers/scheduler_components/metrics_reporter.py` |
> | `/v1/loads` 处理 | `python/sglang/srt/entrypoints/v1_loads.py` |
> | 负载数据生产 | `python/sglang/srt/managers/scheduler_components/load_inquirer.py` |
> | 负载数据传输 | `python/sglang/srt/managers/load_snapshot.py` |
> | 请求/响应数据类 | `python/sglang/srt/managers/io_struct.py` |
>
> 配套阅读：`docs/metrics/overload_metrics_zh.md`（如何用这些指标判断过载）

---

## 0. 两个接口的定位对比

| 维度 | `/metrics` | `/v1/loads` |
| --- | --- | --- |
| **用途** | 监控大盘、告警、容量分析 | 路由器/网关实时选点、DP 均衡 |
| **格式** | Prometheus 文本格式 | JSON（默认）/ Prometheus 文本（可选） |
| **数据来源** | 各进程写入 `PROMETHEUS_MULTIPROC_DIR` 的多进程共享文件 | `/dev/shm` 上的 `LoadSnapshot` mmap 文件 |
| **是否跨进程通信** | 否（读多进程指标目录） | 否（直接读共享内存，**不往返调度器**） |
| **开销** | 较重（全量指标序列化，几百到几千行） | 极轻（一次 mmap 读 + msgpack 解码） |
| **推荐拉取频率** | 15s ~ 30s | 可秒级甚至更高频 |
| **开关** | 必须 `--enable-metrics` | **始终可用**，无需任何开关 |
| **指标数量** | 100+ | 12 个核心 + 5 个可选分区 |
| **粒度** | 按 `tp_rank`/`dp_rank`/`pp_rank` 等标签细分 | 按 `dp_rank` 一行 |

**一句话区分**：`/metrics` 是「全量、给人和监控系统看的」，`/v1/loads` 是「精简、给机器做路由决策的」。

---

# 第一部分：`/metrics`

## 1. 请求

### 1.1 基本形式

```
GET /metrics
```

无任何查询参数、无请求体、无认证（不受 `--api-key` 保护）。

```bash
curl -s http://localhost:30000/metrics
```

### 1.2 前置条件：必须显式开启

`/metrics` 路由**不是默认注册的**。只有 `--enable-metrics` 时才在 lifespan 阶段挂载：

```text
# http_server.py :: lifespan
if server_args.enable_metrics:
    add_prometheus_middleware(app)
    enable_func_timer()
```

未开启时请求 `/metrics` 返回 **404**。

### 1.3 挂载机制（决定了多进程语义）

```text
# utils/common.py :: add_prometheus_middleware
from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)
metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

# Workaround for 307 Redirect for /metrics
metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
app.routes.append(metrics_route)
```

三个关键点：

1. **多进程模式**：SGLang 的 scheduler / tokenizer / detokenizer 是独立进程，各自写指标到 `PROMETHEUS_MULTIPROC_DIR`（由 `set_prometheus_multiproc_dir()` 创建的临时目录）。`/metrics` 由 HTTP 进程统一聚合读出。
2. **`path_regex` 改写**：使 `/metrics/`、`/metrics/xxx` 都能命中，规避 Starlette 的 307 重定向。
3. **`multiprocess_mode`**：绝大多数 Gauge 用 `mostrecent`（取各进程最新值），少数用 `livesum`（活跃进程求和，如 `sglang:http_requests_active`）。**这直接影响你该用 `max` 还是 `sum` 聚合。**

### 1.4 相关启动参数

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--enable-metrics` | `False` | **总开关**，不开则无 `/metrics` 路由 |
| `--enable-metrics-for-all-schedulers` | `False` | 让所有 TP rank 都上报（默认仅 `attn_tp_rank == 0`） |
| `--enable-mfu-metrics` | `False` | 额外上报 FLOPs/带宽估算指标 |
| `--extra-metric-labels` | `None` | 给所有指标追加自定义标签 |
| `--tokenizer-metrics-allowed-custom-labels` | `None` | 允许请求头传入的动态标签名白名单 |
| `--tokenizer-metrics-custom-labels-header` | `x-custom-labels` | 携带动态标签的 HTTP 头名 |
| `--bucket-time-to-first-token` | `None` | 覆盖 TTFT 直方图分桶 |
| `--bucket-inter-token-latency` | `None` | 覆盖 ITL 直方图分桶 |
| `--bucket-e2e-request-latency` | `None` | 覆盖 e2e 延迟直方图分桶 |
| `--collect-tokens-histogram` | `False` | 开启 prompt/generation token 长度直方图 |
| `--decode-log-interval` | `40` | **决定 decode 侧指标刷新周期** |

---

## 2. 返回值

### 2.1 响应头与格式

```
HTTP/1.1 200 OK
content-type: text/plain; version=0.0.4; charset=utf-8
```

标准 Prometheus 文本格式，每个指标三行结构：

```text
# HELP sglang:num_running_reqs The number of running requests.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="Qwen/Qwen3-8B",engine_type="unified",tp_rank="0",pp_rank="0",moe_ep_rank="0"} 12.0
```

### 2.2 标签体系

**Scheduler 侧指标标签**（`metrics_collector.py::SchedulerMetricsCollector.init_new`）：

```text
labels = {
    "model_name":  server_args.served_model_name,
    "engine_type": DisaggregationMode.to_engine_type(...),  # unified | prefill | decode
    "tp_rank":     tp_rank,
    "pp_rank":     pp_rank,
    "moe_ep_rank": ps.moe_ep_rank,
}
if enable_priority_scheduling: labels["priority"] = ""
if dp_rank is not None:        labels["dp_rank"] = dp_rank
if server_args.extra_metric_labels: labels.update(...)
```

**Tokenizer 侧指标标签**（`tokenizer_manager.py::init_metric_collector_watchdog`）——注意**没有 rank 维度**：

```text
labels = {
    "model_name":  server_args.served_model_name,
    "engine_type": engine_type,
}
# + priority（若开启优先级调度）
# + tokenizer_metrics_allowed_custom_labels 中声明的动态标签
# + extra_metric_labels
```

| `engine_type` 取值 | 对应部署 |
| --- | --- |
| `unified` | 非 PD 分离（`--disaggregation-mode null`，默认） |
| `prefill` | PD 分离的 P 节点 |
| `decode` | PD 分离的 D 节点 |

> **聚合陷阱**：Scheduler 指标带 rank 标签，多 TP/DP 会产生多条时间序列。水位类指标（`token_usage` 等）必须用 `max` 或 `avg by (dp_rank)`，**不能 `sum`**。

### 2.3 指标全景（按来源分组）

#### A. Scheduler — 基础运行状态

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:num_running_reqs` | gauge | 正在运行（running batch 中）的请求数 |
| `sglang:num_queue_reqs` | gauge | 等待队列长度 |
| `sglang:num_grammar_queue_reqs` | gauge | 等待语法编译的请求数 |
| `sglang:gen_throughput` | gauge | 生成吞吐（token/s） |
| `sglang:cache_hit_rate` | gauge | 前缀缓存命中率（**仅 prefill 上报时有效，decode 阶段固定写 0**） |
| `sglang:decode_sum_seq_lens` | gauge | decode 批次内所有序列长度之和 |

#### B. Scheduler — 内存池使用率

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:token_usage` | gauge | **各池使用率的最大值**（瓶颈值），非仅 KV |
| `sglang:full_token_usage` | gauge | 全注意力 KV 池使用率 |
| `sglang:swa_token_usage` | gauge | 滑窗注意力池使用率（hybrid-SWA 模型） |
| `sglang:mamba_usage` | gauge | Mamba SSM 状态池使用率（hybrid-SSM 模型） |

口径（`pool_stats_observer.py`）：

$$
\text{usage} = \frac{\text{total} - (\text{available} + \text{evictable})}{\text{total}},
\qquad
\text{token\_usage} = \max(\text{full},\ \text{swa},\ \text{mamba})
$$

> 源码里有 FIXME：`token_usage` 命名有误导性（实际是跨池最大值），改名需走 API 弃用流程。

#### C. Scheduler — 内存池绝对 token 数

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:num_used_tokens` | gauge | 已用 token 数 |
| `sglang:kv_available_tokens` | gauge | KV 池空闲槽位 |
| `sglang:kv_evictable_tokens` | gauge | KV 池可淘汰（radix 缓存）槽位 |
| `sglang:kv_used_tokens` | gauge | KV 池活跃占用槽位 |
| `sglang:swa_available_tokens` / `swa_evictable_tokens` / `swa_used_tokens` | gauge | SWA 池三分解（hybrid-SWA） |
| `sglang:mamba_available_tokens` / `mamba_evictable_tokens` / `mamba_used_tokens` | gauge | Mamba 池三分解（hybrid-SSM） |

不变量（差额是 session 持有等受保护部分）：

$$
\text{kv\_available} + \text{kv\_evictable} + \text{kv\_used} \;\le\; \text{max\_total\_num\_tokens}
$$

#### D. Scheduler — 投机解码

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:spec_accept_length` | gauge | 平均接受长度 |
| `sglang:spec_accept_rate` | gauge | 草稿接受率 |
| `sglang:spec_num_steps` | gauge | 当前生效的 `speculative_num_steps` |
| `sglang:spec_num_draft_tokens` | gauge | 当前生效的草稿 token 数 |

口径：

$$
\text{accept\_length} = \frac{\text{accepted\_draft\_tokens} + \text{bonus\_tokens}}{\text{num\_forward\_passes}},
\qquad
\text{accept\_rate} = \frac{\text{accepted\_draft\_tokens}}{\text{proposed\_draft\_tokens}}
$$

#### E. Scheduler — 抢占与回撤（过载确证）

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:num_retracted_reqs` | gauge | 本上报周期回撤数 |
| `sglang:num_retracted_requests_total` | **counter** | 累计回撤请求数 |
| `sglang:num_retracted_input_tokens_total` | counter | 回撤浪费的输入 token |
| `sglang:num_retracted_output_tokens_total` | counter | 回撤丢弃的输出 token |
| `sglang:num_paused_reqs` | gauge | 因**异步权重同步**暂停的请求数（RL 场景，非过载） |

#### F. Scheduler — PD 分离专用（**非 PD 部署下恒为 0**）

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:num_prefill_bootstrap_queue_reqs` | gauge | P 端握手队列 |
| `sglang:num_prefill_inflight_queue_reqs` | gauge | P 端 KV 传输中队列 |
| `sglang:num_decode_prealloc_queue_reqs` | gauge | D 端预分配队列 |
| `sglang:num_decode_transfer_queue_reqs` | gauge | D 端传输队列 |
| `sglang:pending_prealloc_token_usage` | gauge | 待预分配 token 占比 |
| `sglang:kv_transfer_speed_gb_s` | histogram | KV 传输速度 |
| `sglang:kv_transfer_latency_ms` | histogram | KV 传输延迟 |
| `sglang:kv_transfer_bootstrap_ms` | histogram | 握手耗时 |
| `sglang:kv_transfer_alloc_ms` | histogram | 分配等待耗时 |
| `sglang:kv_transfer_total_mb` | histogram | 单次传输大小 |
| `sglang:num_bootstrap_failed_reqs_total` | counter | 握手失败数 |
| `sglang:num_transfer_failed_reqs_total` | counter | 传输失败数 |
| `sglang:num_prefill_retries_total` | counter | prefill 重试数 |

#### G. Scheduler — 利用率与调度策略

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:utilization` | gauge | ⚠️ **当前恒为 0**，见 §2.5 |
| `sglang:fwd_occupancy` | gauge | 前向 GPU 占用率 %（需 `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1`，否则 NaN） |
| `sglang:new_token_ratio` | gauge | 调度保守度系数（长期不衰减 = 反复回撤） |
| `sglang:is_cuda_graph` | gauge | 本批次是否走 CUDA Graph（1/0） |
| `sglang:cuda_graph_passes_total` | counter | 按 `mode` 标签分类的前向次数 |

#### H. Scheduler — 请求延迟与阶段耗时

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:queue_time_seconds` | histogram | **排队耗时**（调度器视角） |
| `sglang:per_stage_req_latency_seconds` | histogram | 各阶段耗时，带 `stage` 标签 |

`stage` 取值来自 `req_time_stats.py::RequestStage`，非 PD 场景常见：`request_process`、`prefill_forward`、`chunked_prefill`；PD 场景另有 `prefill_bootstrap`、`decode_waiting`、`decode_transferred` 等。

#### I. Scheduler — 结构化输出（Grammar）

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:grammar_compilation_time_seconds` | histogram | 语法编译耗时 |
| `sglang:num_grammar_cache_hit_total` | counter | 语法缓存命中 |
| `sglang:num_grammar_aborted_total` | counter | 语法中止 |
| `sglang:num_grammar_timeout_total` | counter | 语法超时 |
| `sglang:num_grammar_total` | counter | 语法请求总数 |
| `sglang:grammar_schema_count` | histogram | schema 数量分布 |
| `sglang:grammar_ebnf_size` | histogram | EBNF 大小分布 |
| `sglang:grammar_tree_traversal_time_avg` / `_max` | histogram | 语法树遍历耗时 |

#### J. Scheduler — 执行与 MFU（需 `--enable-mfu-metrics`）

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:realtime_tokens_total` | counter | 按 `mode`（prefill_compute / prefill_cache / decode）分类的 token 数 |
| `sglang:forward_execution_seconds_total` | counter | GPU 前向忙碌时间，带 `category` 标签 |
| `sglang:estimated_flops_per_gpu_total` | counter | 估算 FLOPs |
| `sglang:estimated_read_bytes_per_gpu_total` | counter | 估算读字节 |
| `sglang:estimated_write_bytes_per_gpu_total` | counter | 估算写字节 |
| `sglang:dp_cooperation_realtime_tokens_total` | counter | 带 `num_prefill_ranks` 标签的 DP 协同 token |
| `sglang:dp_cooperation_forward_execution_seconds_total` | counter | 带 DP 协同标签的前向时间 |
| `sglang:eplb_balancedness` | summary | MoE 专家负载均衡度（需 `SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC`） |

#### K. Scheduler — Prefill Delayer（**仅非 PD + overlap 调度**）

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:prefill_delayer_wait_forward_passes` | histogram | 推迟的前向轮数 |
| `sglang:prefill_delayer_wait_seconds` | histogram | 推迟的墙钟时间 |
| `sglang:prefill_delayer_outcomes_total` | counter | 带 `input_estimation`/`output_allow`/`output_reason`/`actual_execution` 标签的决策计数 |

#### L. Scheduler — LoRA / HiCache / Session / Routing Key（条件注册）

| 指标 | 注册条件 | 说明 |
| --- | --- | --- |
| `sglang:lora_pool_slots_used` / `_total` / `sglang:lora_pool_utilization` | `enable_lora` | LoRA 槽位 |
| `sglang:hicache_host_used_tokens` / `_total_tokens` | `enable_hierarchical_cache` | 主机层 KV |
| `sglang:num_streaming_sessions`、`sglang:streaming_session_held_tokens` | `enable_streaming_session` | 流式会话 |
| `sglang:num_unique_running_routing_keys` | always | 运行中去重路由键数 |
| `sglang:routing_key_running_req_count` | always | 路由键请求数分布（GaugeHistogram） |
| `sglang:routing_key_all_req_count` | always | 含排队的路由键分布 |

#### M. Scheduler — 启动常量（`emit_constants`，启动时写一次）

| 指标 | 说明 |
| --- | --- |
| `sglang:max_total_num_tokens` | KV 池容量上限 |
| `sglang:max_running_requests_under_SLO` | ⚠️ 无 setter，通常不出现，见 §2.5 |
| `sglang:engine_startup_time` | 引擎启动耗时 |
| `sglang:engine_load_weights_time` | 权重加载耗时 |
| `sglang:page_size` | KV page 大小 |
| `sglang:num_pages` | KV page 数量 |
| `sglang:context_len` | 最大上下文长度 |
| `sglang:startup_available_gpu_memory_gb` | 启动时可用显存 |
| `sglang:weight_load_duration_seconds` | 权重加载耗时，带 `source` 标签（disk/distributed/tensor/ipc） |

#### N. Tokenizer — 请求级指标（**无 rank 标签**）

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:prompt_tokens_total` | counter | 累计 prefill token |
| `sglang:generation_tokens_total` | counter | 累计生成 token |
| `sglang:cached_tokens_total` | counter | 命中缓存 token，带 `cache_source` 标签（device/host/storage_xxx） |
| `sglang:spec_verify_calls_total` | counter | 投机验证调用次数 |
| `sglang:num_requests_total` | counter | 请求总数 |
| `sglang:num_so_requests_total` | counter | 结构化输出请求数 |
| `sglang:num_aborted_requests_total` | counter | **被中止请求数**（队列满拒绝时会增长） |
| `sglang:time_to_first_token_seconds` | histogram | TTFT |
| `sglang:inter_token_latency_seconds` | histogram | ITL |
| `sglang:e2e_request_latency_seconds` | histogram | 端到端延迟 |
| `sglang:prompt_tokens_histogram` | histogram | prompt 长度分布 |
| `sglang:uncached_prompt_tokens_histogram` | histogram | 未命中缓存的 prompt 长度分布 |
| `sglang:generation_tokens_histogram` | histogram | 生成长度分布 |
| `sglang:get_loads_duration_seconds` | histogram | **`/v1/loads` 自身的处理耗时** |

#### O. HTTP 层（`add_prometheus_track_response_middleware`）

| 指标 | 类型 | 标签 | 说明 |
| --- | --- | --- | --- |
| `sglang:http_requests_total` | counter | `endpoint`, `method` | HTTP 请求数 |
| `sglang:http_responses_total` | counter | `endpoint`, `status_code`, `method` | HTTP 响应数 |
| `sglang:http_requests_active` | gauge (`livesum`) | `endpoint`, `method` | 活跃请求数 |
| `sglang:routing_keys_active` | gauge (`livesum`) | — | 活跃路由键数（引用计数） |
| `sglang:func_latency_seconds` | histogram | `name` | 被 `@time_func_latency` 装饰的函数耗时 |

#### P. 存储与 Radix Cache（HiCache L3 场景）

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:prefetched_tokens_total` | counter | L3 预取 token |
| `sglang:backuped_tokens_total` | counter | 备份到 L3 的 token |
| `sglang:prefetch_pgs` / `sglang:backup_pgs` | histogram | 预取/备份页数 |
| `sglang:prefetch_bandwidth` / `sglang:backup_bandwidth` | histogram | 预取/备份带宽 GB/s |
| `sglang:evicted_tokens_total` | counter | GPU→CPU 淘汰 token |
| `sglang:load_back_tokens_total` | counter | CPU→GPU 回载 token |
| `sglang:eviction_duration_seconds` | histogram | 淘汰耗时（可用 `SGLANG_BUCKET_EVICTION_DURATION` 覆盖分桶） |
| `sglang:load_back_duration_seconds` | histogram | 回载耗时（可用 `SGLANG_BUCKET_LOAD_BACK_DURATION` 覆盖分桶） |
| `sglang:eplb_gpu_physical_count` | histogram | 各层各 GPU 的物理专家选中次数 |

### 2.4 返回样例（截断）

```text
# HELP sglang:num_running_reqs The number of running requests.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{engine_type="unified",model_name="Qwen/Qwen3-8B",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 24.0

# HELP sglang:token_usage The token usage.
# TYPE sglang:token_usage gauge
sglang:token_usage{engine_type="unified",model_name="Qwen/Qwen3-8B",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 0.73

# HELP sglang:num_queue_reqs The number of requests in the waiting queue.
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{engine_type="unified",model_name="Qwen/Qwen3-8B",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 3.0

# HELP sglang:gen_throughput The generation throughput (token/s).
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{engine_type="unified",model_name="Qwen/Qwen3-8B",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 1842.55

# HELP sglang:num_retracted_requests_total Total number of retracted requests.
# TYPE sglang:num_retracted_requests_total counter
sglang:num_retracted_requests_total{engine_type="unified",model_name="Qwen/Qwen3-8B",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 0.0

# HELP sglang:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_bucket{engine_type="unified",le="0.1",model_name="Qwen/Qwen3-8B"} 120.0
sglang:time_to_first_token_seconds_bucket{engine_type="unified",le="0.2",model_name="Qwen/Qwen3-8B"} 480.0
...
sglang:time_to_first_token_seconds_sum{engine_type="unified",model_name="Qwen/Qwen3-8B"} 312.44
sglang:time_to_first_token_seconds_count{engine_type="unified",model_name="Qwen/Qwen3-8B"} 1024.0

# HELP sglang:max_total_num_tokens Maximum total number of tokens in the KV cache pool.
# TYPE sglang:max_total_num_tokens gauge
sglang:max_total_num_tokens{engine_type="unified",model_name="Qwen/Qwen3-8B",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 524288.0
```

### 2.5 使用 `/metrics` 的关键注意事项

**① 刷新节奏不是实时的**

- decode 侧重指标每 `--decode-log-interval`（默认 40）个 step 刷新一次；
- prefill 侧每个 prefill batch 刷新一次；
- 完全空闲时兜底每 30s 刷一次（`_maybe_log_itle_metrics`）。

因此告警的 `for` 至少给 1~2 分钟，避免抓到两次刷新之间的陈旧值。

**② `sglang:utilization` 当前不可用**

设计口径本应取「请求并发维度」与「KV token 维度」中更紧张的一个：

$$
\text{utilization} =
\max\!\left(
  \frac{\text{num\_running\_reqs}}{\text{max\_running\_requests\_under\_SLO}},\;
  \frac{\text{token\_usage}}{0.9}
\right)
$$

但实际代码中该分支进不去：

```text
# metrics_reporter.py :: _calculate_utilization
if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
    self.stats.utilization = -1
else:
    # TODO: max_running_requests_under_SLO has no setter
    #       -> sglang:utilization stuck at 0 (regressed #22713)
    max_under_slo = getattr(self.scheduler, "max_running_requests_under_SLO", None)
    if max_under_slo is not None and max_under_slo > 0:
        self.stats.utilization = max(...)
```

`max_running_requests_under_SLO` 无 setter，实际为 `None`，整个分支不执行，指标恒为 0。PD 分离的 prefill 节点则显式置 `-1`（N/A）。

**③ 谁在上报**

默认只有 `attn_tp_rank == 0` 的 scheduler 上报；`--enable-metrics-for-all-schedulers` 后所有 TP rank 都报。后者在开启 DP attention 时很有用（否则所有指标看起来都来自 TP 0）。

**④ 部分指标条件注册**

LoRA / HiCache / Streaming Session / EPLB 相关指标只在对应功能开启时才创建。查不到指标先确认功能是否启用，而不是怀疑采集故障。

---

# 第二部分：`/v1/loads`

## 3. 请求

### 3.1 基本形式

```
GET /v1/loads?dp_rank={int}&include={csv}&format={json|prometheus}
```

路由由独立的 `APIRouter` 提供并挂载到主 app：

```text
# http_server.py
from sglang.srt.entrypoints.v1_loads import router as v1_loads_router
app.include_router(v1_loads_router)
```

**无需 `--enable-metrics`，始终可用**（这是它与 `/metrics` 最重要的差别之一）。

### 3.2 查询参数

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `dp_rank` | `int?` | `None` | 只返回指定 DP rank。越界（`<0` 或 `>= dp_size`）时返回**空列表**而非报错 |
| `include` | `str?` | `None`（等价 `all`） | 逗号分隔的分区名，控制返回哪些可选段 |
| `format` | `str?` | `json` | `json` 或 `prometheus` |

**`include` 合法取值**（`GetLoadsReqInput.VALID_SECTIONS` / `LoadSnapshot.VALID_SECTIONS`）：

| 值 | 含义 |
| --- | --- |
| `core` | 仅核心 12 字段（**最轻量，路由器推荐**） |
| `memory` | 显存细分 |
| `spec` | 投机解码 |
| `lora` | LoRA 槽位 |
| `disagg` | PD 分离队列与传输 |
| `queues` | 队列细分 |
| `all` | 全部（默认） |

传入非法分区名返回 **400**：

```json
{"detail": "Invalid include sections: {'foo'}. Valid options: ['all', 'core', 'disagg', 'lora', 'memory', 'queues', 'spec']"}
```

### 3.3 请求示例

```bash
# 全部分区（默认）
curl -s http://localhost:30000/v1/loads

# 仅核心指标（路由器高频轮询用，最轻）
curl -s 'http://localhost:30000/v1/loads?include=core'

# 多个分区
curl -s 'http://localhost:30000/v1/loads?include=core,memory,queues'

# 指定 DP rank
curl -s 'http://localhost:30000/v1/loads?dp_rank=2&include=core'

# Prometheus 文本格式
curl -s 'http://localhost:30000/v1/loads?format=prometheus'
```

### 3.4 数据链路（为什么这么快）

`/v1/loads` **不向调度器发 ZMQ 请求**，而是直接读共享内存：

```text
Scheduler(dp_rank=i)
    └─ SchedulerLoadInquirer.get_loads()      # 组装 GetLoadsReqOutput
         └─ LoadSnapshot.from_get_loads_output()   # 拍平为扁平结构
              └─ ShmLoadSnapshotWriter.write()     # 写 /dev/shm 的第 i 个槽位
                                                        │
HTTP Server                                             │
    └─ TokenizerManager.get_loads()                     │
         └─ ShmLoadSnapshotReader.read_all()  ◀─────────┘
```

SHM 文件布局：`[Header(12B)][slot_0(16KB)][slot_1]...[slot_{dp_size-1}]`，每个 DP rank 独占一个槽位，读写用 `flock` 共享锁/独占锁协调，写入顺序为「先清零长度 → 写 payload → 最后写回真实长度」，保证读者不会读到撕裂数据。

**多节点 DP attention**（`enable_dp_attention and nnodes > 1`）时共享内存跨不了节点，自动切换为 ZMQ PUSH/PULL：各节点 scheduler PUSH，node 0 的 owner 进程 PULL 后写入本地 SHM，其余读者仍读 SHM。可用 `SGLANG_LOAD_SNAPSHOT_USE_ZMQ=1` 强制。

---

## 4. 返回值

### 4.1 JSON 格式（默认）

顶层结构：

```json
{
  "timestamp": "2026-08-06T09:15:23.481920+00:00",
  "version": "0.5.x",
  "loads": []
}
```

其中 `loads` 数组的每个元素对应一个 DP rank。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `timestamp` | string | **响应生成时刻**的 UTC ISO8601（注意：不是快照采样时刻） |
| `version` | string | SGLang 版本号 |
| `loads` | array | 每个 DP rank 一个负载对象；`dp_size=1` 时长度为 1 |

> ⚠️ 顶层**没有** `aggregate` 或 `dp_rank_count` 字段。历史上曾有服务端聚合，现已移除（见 `test/registered/unit/entrypoints/test_v1_loads_aggregate.py` 的断言）。跨 rank 聚合请由调用方自行完成。

### 4.2 `loads[]` 核心字段（始终返回）

来自 `LoadSnapshot.to_dict()`，共 12 个：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `dp_rank` | int | DP rank 编号 |
| `num_running_reqs` | int | 运行中请求数 |
| `num_waiting_reqs` | int | 等待请求数（PD 模式下含各专用队列） |
| `num_waiting_uncached_tokens` | int | **等待 prefill 且未命中缓存**的 token 数 |
| `num_used_tokens` | int | 已占用 KV 的 token 数 |
| `num_total_tokens` | int | **`num_used_tokens` + 等待队列 seqlen 之和** |
| `max_total_num_tokens` | int | KV 容量上限 |
| `max_running_requests` | int | 并发上限 |
| `token_usage` | float | 池使用率（跨池最大值，保留 4 位小数） |
| `gen_throughput` | float | 生成吞吐 token/s（保留 2 位） |
| `cache_hit_rate` | float | 前缀缓存命中率（保留 4 位） |
| `utilization` | float | 综合利用率（⚠️ 恒为 0，见 §2.5） |

**两个最重要的派生量**：

`num_total_tokens` —— 跨节点比较负载的**统一标量**：

$$
\text{num\_total\_tokens} = \underbrace{\text{num\_used\_tokens}}_{\text{GPU 上已占的 KV}} \;+\; \sum_{\text{req} \in \text{等待队列}} \text{req.seqlen}
$$

它同时包含「在跑的」和「排队的」两部分负债，量纲统一为 token，不受请求长度分布影响。SGLang 的 `--load-balance-method total_tokens` 与网关 Power-of-Two 策略都用它。

> ⚠️ 排队部分**未扣除前缀缓存命中**，直接累加完整 `seqlen`，高前缀复用场景下会**高估**负载。这是为避免在负载查询热路径做前缀匹配而做的保守近似。

`num_waiting_uncached_tokens` —— 真正欠的 prefill 算力：

$$
\text{num\_waiting\_uncached\_tokens} =
\sum_{\text{req} \in \text{等待队列}} \max\bigl(0,\ \text{req.seqlen} - \text{req.num\_matched\_prefix\_tokens}\bigr)
$$

纯 DECODE 节点无 prefill，该值直接返回 0。

### 4.3 可选分区字段

#### `memory`（`include=memory`）

```json
{
  "memory": {
    "weight_gb": 15.234,
    "kv_cache_gb": 42.117,
    "graph_gb": 1.882,
    "token_capacity": 524288
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `weight_gb` | 模型权重显存 GB |
| `kv_cache_gb` | KV cache 显存 GB |
| `graph_gb` | CUDA Graph 显存 GB |
| `token_capacity` | KV 可容纳 token 上限 |

> 该段用 `try/except AttributeError` 保护；属性缺失时整段降级为 `null`（不会报错）。

#### `speculative`（`include=spec`）

```json
{
  "speculative": { "accept_length": 2.87, "accept_rate": 0.62 }
}
```

**仅当启用投机算法且已有前向计数时才填充**，否则为 `null`。`accept_length` 为现算值：

$$
\text{accept\_length} = \frac{\text{cumulative\_accepted\_tokens}}{\text{cumulative\_forward\_passes}}
$$

#### `lora`（`include=lora`）

```json
{
  "lora": { "slots_used": 3, "slots_total": 8, "utilization": 0.375 }
}
```

仅 `--enable-lora` 时填充，否则 `null`。其中：

$$
\text{utilization} = \frac{\text{slots\_used}}{\text{slots\_total}}
$$

#### `disaggregation`（`include=disagg`）

```json
{
  "disaggregation": {
    "mode": "null",
    "prefill_bootstrap_queue_reqs": 0,
    "prefill_inflight_queue_reqs": 0,
    "decode_prealloc_queue_reqs": 0,
    "decode_transfer_queue_reqs": 0,
    "decode_retracted_queue_reqs": 0,
    "kv_transfer_speed_gb_s": 0.0,
    "kv_transfer_latency_ms": 0.0
  }
}
```

| `mode` | 含义 |
| --- | --- |
| `"null"` | 非 PD 分离 —— **此时后续所有队列字段恒为 0** |
| `"prefill"` | 只有 `prefill_*` 有值 |
| `"decode"` | 只有 `decode_*` 有值 |

传输时 `mode` 用整数编码（`null=0, prefill=1, decode=2`），读出时反查回字符串。

#### `queues`（`include=queues`）

```json
{
  "queues": { "waiting": 3, "grammar": 0, "paused": 0, "retracted": 0 }
}
```

| 字段 | 说明 |
| --- | --- |
| `waiting` | 主等待队列长度（**不含** PD 专用队列，与核心字段 `num_waiting_reqs` 口径不同） |
| `grammar` | 等待语法编译数 |
| `paused` | 被权重同步暂停数 |
| `retracted` | 被回撤数 |

> 注意 `queues.waiting` 与 `num_waiting_reqs` 的差异：前者只统计主 `waiting_queue`，后者在 PD 模式下还会加上 bootstrap/prealloc/transfer/retracted 队列。非 PD 场景下两者相等。

### 4.4 完整 JSON 返回样例

`GET /v1/loads`（默认 `include=all`，非 PD 分离，`dp_size=2`）：

```json
{
  "timestamp": "2026-08-06T09:15:23.481920+00:00",
  "version": "0.5.4",
  "loads": [
    {
      "dp_rank": 0,
      "num_running_reqs": 24,
      "num_waiting_reqs": 3,
      "num_waiting_uncached_tokens": 6144,
      "num_used_tokens": 382910,
      "num_total_tokens": 401294,
      "max_total_num_tokens": 524288,
      "max_running_requests": 256,
      "token_usage": 0.7303,
      "gen_throughput": 1842.55,
      "cache_hit_rate": 0.6412,
      "utilization": 0.0,
      "memory": {
        "weight_gb": 15.234,
        "kv_cache_gb": 42.117,
        "graph_gb": 1.882,
        "token_capacity": 524288
      },
      "disaggregation": {
        "mode": "null",
        "prefill_bootstrap_queue_reqs": 0,
        "prefill_inflight_queue_reqs": 0,
        "decode_prealloc_queue_reqs": 0,
        "decode_transfer_queue_reqs": 0,
        "decode_retracted_queue_reqs": 0,
        "kv_transfer_speed_gb_s": 0.0,
        "kv_transfer_latency_ms": 0.0
      },
      "queues": { "waiting": 3, "grammar": 0, "paused": 0, "retracted": 0 }
    },
    {
      "dp_rank": 1,
      "num_running_reqs": 19,
      "num_waiting_reqs": 0,
      "num_waiting_uncached_tokens": 0,
      "num_used_tokens": 291044,
      "num_total_tokens": 291044,
      "max_total_num_tokens": 524288,
      "max_running_requests": 256,
      "token_usage": 0.5551,
      "gen_throughput": 1521.08,
      "cache_hit_rate": 0.7130,
      "utilization": 0.0,
      "memory": {
        "weight_gb": 15.234,
        "kv_cache_gb": 42.117,
        "graph_gb": 1.882,
        "token_capacity": 524288
      },
      "disaggregation": {
        "mode": "null",
        "prefill_bootstrap_queue_reqs": 0,
        "prefill_inflight_queue_reqs": 0,
        "decode_prealloc_queue_reqs": 0,
        "decode_transfer_queue_reqs": 0,
        "decode_retracted_queue_reqs": 0,
        "kv_transfer_speed_gb_s": 0.0,
        "kv_transfer_latency_ms": 0.0
      },
      "queues": { "waiting": 0, "grammar": 0, "paused": 0, "retracted": 0 }
    }
  ]
}
```

`GET /v1/loads?include=core`（只有 12 个核心字段，无任何可选段）：

```json
{
  "timestamp": "2026-08-06T09:15:23.481920+00:00",
  "version": "0.5.4",
  "loads": [
    {
      "dp_rank": 0,
      "num_running_reqs": 24,
      "num_waiting_reqs": 3,
      "num_waiting_uncached_tokens": 6144,
      "num_used_tokens": 382910,
      "num_total_tokens": 401294,
      "max_total_num_tokens": 524288,
      "max_running_requests": 256,
      "token_usage": 0.7303,
      "gen_throughput": 1842.55,
      "cache_hit_rate": 0.6412,
      "utilization": 0.0
    }
  ]
}
```

### 4.5 Prometheus 格式（`format=prometheus`）

```
content-type: text/plain; version=0.0.4; charset=utf-8
```

转换规则（`v1_loads.py::_format_loads_prometheus`）：

- 标量字段 → `sglang_{key}{dp_rank="N"} value`
- 嵌套段字段 → `sglang_{prefix}_{sub_key}{dp_rank="N"} value`
- 段名前缀映射：`speculative` → `spec`，`disaggregation` → `disagg`，其余保持原名
- 全部声明为 `gauge`；非数值字段（如 `disaggregation.mode`）自动跳过

```text
# TYPE sglang_num_running_reqs gauge
sglang_num_running_reqs{dp_rank="0"} 24
sglang_num_running_reqs{dp_rank="1"} 19
# TYPE sglang_token_usage gauge
sglang_token_usage{dp_rank="0"} 0.7303
sglang_token_usage{dp_rank="1"} 0.5551
# TYPE sglang_num_total_tokens gauge
sglang_num_total_tokens{dp_rank="0"} 401294
sglang_num_total_tokens{dp_rank="1"} 291044
# TYPE sglang_memory_weight_gb gauge
sglang_memory_weight_gb{dp_rank="0"} 15.234
# TYPE sglang_queues_waiting gauge
sglang_queues_waiting{dp_rank="0"} 3
```

> **注意指标名分隔符不同**：`/metrics` 用冒号（`sglang:num_running_reqs`），`/v1/loads?format=prometheus` 用下划线（`sglang_num_running_reqs`）。两者是**不同的时间序列**，抓取时不要混淆。

### 4.6 边界与错误行为

| 场景 | 行为 |
| --- | --- |
| `include` 含非法分区 | **400**，`detail` 说明合法取值 |
| `dp_rank` 越界 | **200**，`loads` 为空数组（不报错） |
| SHM 文件尚未创建（启动初期） | **200**，`loads` 为空数组 |
| 某 rank 快照解码失败 | 该 rank 被静默跳过，其余正常返回 |
| 某可选段属性缺失 | 该段为 `null`，其余字段正常 |
| 请求了某段但功能未启用（如 `lora`） | 该段为 `null` |

**自监控**：每次调用都会记录 `sglang:get_loads_duration_seconds` 直方图（在 `finally` 中，即使抛异常也记）。若该指标 P99 异常升高，说明 SHM 读取遇到锁竞争。

---

## 5. 附：已废弃的 `/get_load`

```
GET /get_load
```

保留的兼容 shim，内部调用 `/v1/loads?include=core` 后投影为旧字段名，**会打 WARNING 日志**：

```text
# http_server.py :: get_load
load_results = await tokenizer_manager.get_loads(include=["core"])
ts = time.perf_counter()
return [
    {
        "dp_rank":            r.dp_rank,
        "num_reqs":           r.num_running_reqs + r.num_waiting_reqs,
        "num_waiting_reqs":   r.num_waiting_reqs,
        "num_tokens":         r.num_total_tokens,
        "num_pending_tokens": r.num_total_tokens - r.num_used_tokens,
        "ts_tic":             ts,
    }
    for r in load_results
]
```

其中两个字段是派生量：

$$
\text{num\_reqs} = \text{num\_running\_reqs} + \text{num\_waiting\_reqs},
\qquad
\text{num\_pending\_tokens} = \text{num\_total\_tokens} - \text{num\_used\_tokens}
$$

返回**裸数组**（无 `loads` 包裹）：

```json
[
  {
    "dp_rank": 0,
    "num_reqs": 27,
    "num_waiting_reqs": 3,
    "num_tokens": 401294,
    "num_pending_tokens": 18384,
    "ts_tic": 128374.55
  }
]
```

注意 `ts_tic` 是 `time.perf_counter()`（单调时钟，**非 Unix 时间戳**），只能用于计算相对间隔。新代码请改用 `/v1/loads`。

---

## 6. 选型建议

| 场景 | 用哪个 | 参数 |
| --- | --- | --- |
| Grafana 监控大盘 | `/metrics` | — |
| 告警规则 | `/metrics` | — |
| 网关/路由器选点（高频） | `/v1/loads` | `?include=core` |
| DP rank 负载倾斜检测 | `/v1/loads` | `?include=core` |
| 显存容量分析 | `/v1/loads` | `?include=memory` |
| 投机解码调参 | `/metrics` 或 `/v1/loads?include=spec` | — |
| 排查单个 DP rank | `/v1/loads` | `?dp_rank=N&include=all` |
| 已有 Prometheus 但未开 `--enable-metrics` | `/v1/loads` | `?format=prometheus` |
| 容器存活探针 | `/health` | — |
| 容器就绪探针 | `/health_generate` | — |

### 6.1 网关接入的已知坑

`sgl-model-gateway` 的 `parse_load_response` 解析的是：

```text
# sgl-model-gateway/src/core/worker_manager.rs
let load_url = format!("{}/v1/loads?include=core", url);
json["aggregate"]["total_tokens"]   // 解析为 isize
// 失败 / 字段缺失 → 返回 -1
```

而引擎原生返回的是 `{"loads": [{"num_total_tokens": N, ...}]}`（**无 `aggregate` 层，字段名也不同**）。若直连引擎且中间无格式转换层，网关会拿到 `-1`，导致 Power-of-Two 策略**静默退化为按本地请求计数比较**。接入前务必用 `curl` 验证一次实际响应结构。

---

## 7. 速查表

```text
/metrics
  请求   GET /metrics                      无参数、无认证
  前提   --enable-metrics（否则 404）
  返回   text/plain; version=0.0.4          Prometheus 文本
  命名   sglang:xxx                         冒号分隔
  标签   Scheduler: model_name, engine_type, tp_rank, pp_rank, moe_ep_rank[, dp_rank]
         Tokenizer: model_name, engine_type（无 rank）
  刷新   decode 每 --decode-log-interval(40) 步；prefill 每批；空闲 30s 兜底
  聚合   水位类用 max / avg by(dp_rank)，切勿 sum
  失效   sglang:utilization 恒为 0（#22713）

/v1/loads
  请求   GET /v1/loads?dp_rank=&include=&format=
  前提   无（始终可用）
  参数   dp_rank  int?     越界返回空数组
         include  csv      core|memory|spec|lora|disagg|queues|all，默认 all
         format   str      json（默认）| prometheus
  返回   {timestamp, version, loads:[...]}   顶层无 aggregate
  核心   dp_rank, num_running_reqs, num_waiting_reqs, num_waiting_uncached_tokens,
         num_used_tokens, num_total_tokens, max_total_num_tokens, max_running_requests,
         token_usage, gen_throughput, cache_hit_rate, utilization
  可选   memory / speculative / lora / disaggregation / queues（未启用则为 null）
  命名   format=prometheus 时用 sglang_xxx（下划线，与 /metrics 的冒号不同）
  链路   读 /dev/shm 的 LoadSnapshot，不往返调度器；多节点 DP 走 ZMQ 中转
  自监控 sglang:get_loads_duration_seconds

/get_load（已废弃）
  返回裸数组，字段：dp_rank, num_reqs, num_waiting_reqs, num_tokens,
                    num_pending_tokens, ts_tic(perf_counter，非 Unix 时间)
```

跨节点比较负载时使用的统一标量：

$$
\text{num\_total\_tokens} = \text{num\_used\_tokens} + \sum_{\text{req} \in \text{等待队列}} \text{req.seqlen}
$$

