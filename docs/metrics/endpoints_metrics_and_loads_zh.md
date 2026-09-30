# `/metrics` 接口详解：请求与返回值

> 本文逐字段说明 SGLang HTTP Server `/metrics` 观测接口的**请求参数**与**返回结构**，所有内容均可溯源到代码：
>
> | 主题 | 代码位置 |
> | --- | --- |
> | 路由注册 | `python/sglang/srt/entrypoints/http_server.py` |
> | `/metrics` 挂载 | `python/sglang/srt/utils/common.py::add_prometheus_middleware` |
> | 指标定义 | `python/sglang/srt/observability/metrics_collector.py` |
> | 指标填充 | `python/sglang/srt/managers/scheduler_components/metrics_reporter.py` |
> | 请求/响应数据类 | `python/sglang/srt/managers/io_struct.py` |
>
> 配套阅读：`docs/metrics/overload_metrics_zh.md`（如何用这些指标判断过载）、`docs/metrics/v1_loads_zh.md`（`/v1/loads` 接口）

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

> `/v1/loads` 接口的请求参数、返回结构、分区字段、Prometheus 格式与已废弃的 `/get_load`，见 `docs/metrics/v1_loads_zh.md`。

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

#### B.1 `token_usage` 完整计算过程

以下口径来自 `PoolStats` / `SchedulerPoolStatsObserver`（`pool_stats_observer.py`），各池算法一致：

$$
\text{num\_used}_{\text{pool}} = \text{capacity} - (\text{available} + \text{evictable}),
\qquad
\text{usage}_{\text{pool}} = \frac{\text{num\_used}_{\text{pool}}}{\text{capacity}}
$$

各符号来源：

| 符号 | 含义 | 取值来源 |
| --- | --- | --- |
| `available` | 池中空闲、可直接分配的槽位 | plain/SWA：`token_to_kv_pool_allocator.available_size()`；mamba：`req_to_token_pool.mamba_allocator.available_size()` |
| `evictable` | 前缀缓存（radix tree）中可被淘汰复用的槽位 | `tree_cache.evictable_size()`，SWA/mamba 用各自的 `full_evictable_size()` / `swa_evictable_size()` / `mamba_evictable_size()` |
| `capacity` | 池总容量 | plain：`max_total_num_tokens`；SWA：`full_tokens_per_layer` / `swa_tokens_per_layer`；mamba：`req_to_token_pool.mamba_pool.size` |

最终上报值（`update_scheduler_stats`）：

$$
\text{token\_usage} = \text{round}\Big(\max\big(\text{full\_token\_usage},\ [\text{swa\_token\_usage}],\ [\text{mamba\_usage}]\big),\ 2\Big)
$$

- 普通（非 SWA/非 SSM）模型：只有 `full` 一项 → `token_usage = full_token_usage`；
- hybrid-SWA 模型：并入 `swa` 参与取 max；
- hybrid-SSM（mamba）模型：再并入 `mamba`（SWA 与 SSM 可同时存在）。
- **上报值已 `round(…, 2)`，即保留 2 位小数。**

配套的绝对 token 数（`get_kv_token_stats`）：

$$
\text{num\_used\_tokens} =
\begin{cases}
\max(\text{full\_num\_used},\ \text{swa\_num\_used}) & \text{hybrid-SWA} \\
\text{full\_num\_used} & \text{其它}
\end{cases}
$$

即 `sglang:num_used_tokens` / `sglang:kv_used_tokens` = `full_num_used`；`kv_available_tokens = full_available_size`；`kv_evictable_tokens = full_evictable_size`。

> 特殊情况：HiSparse 开启时 `full_num_used` / `swa_num_used` 会被 `max(0, ·)` 截断（host-backup 瞬时过释放的临时规避）；某池 `capacity == 0` 时其 usage 记 `0.0`。

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

## 3. 速查表

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
```

> `/v1/loads` 接口（请求参数、返回结构、分区字段、Prometheus 格式、已废弃的 `/get_load` 等）以及两个接口的选型建议，见 `docs/metrics/v1_loads_zh.md`。
