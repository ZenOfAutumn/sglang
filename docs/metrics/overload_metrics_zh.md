# 非 PD 分离场景下的节点过载与集群过载判定指标

> 适用范围：`--disaggregation-mode null`（默认，即 **unified** 部署，prefill 与 decode 在同一实例内混跑）。
>
> 本文只使用**本仓库真实存在**的指标与字段，均可溯源到代码：
> - 调度器指标定义：`python/sglang/srt/observability/metrics_collector.py`
> - 指标填充与上报：`python/sglang/srt/managers/scheduler_components/metrics_reporter.py`
> - 内存池口径：`python/sglang/srt/managers/scheduler_components/pool_stats_observer.py`
> - 负载查询（`/v1/loads`）：`python/sglang/srt/managers/scheduler_components/load_inquirer.py`
> - 负载快照与 DP 均衡：`python/sglang/srt/managers/load_snapshot.py`、`python/sglang/srt/managers/data_parallel_controller.py`
> - 网关侧指标：`sgl-model-gateway/src/observability/metrics.rs`、`sgl-model-gateway/src/core/worker_manager.rs`

---

## 0. 一句话结论

**节点过载 = 「KV 显存吃紧」「并发槽位打满」「排队堆积」三者中任一越线，且已经反映到 SLO（TTFT/ITL）上。**

推荐的**最小充分指标集**（4 个核心 + 2 个确证）：

| 角色 | 指标 | 含义 | 过载判据（经验起点） |
| --- | --- | --- | --- |
| 核心① KV 水位 | `sglang:token_usage` | KV 池真实占用率（已扣可淘汰） | 持续 $> 0.9$ |
| 核心② 并发水位 | `sglang:num_running_reqs / sglang:max_running_requests_under_SLO`（或启动日志中的 `max_running_requests`） | 运行槽位占用率 | 持续 $\ge 0.95$ |
| 核心③ 排队深度 | `sglang:num_queue_reqs` | 等待队列长度 | 持续 $> 0$ 且单调增长 |
| 核心④ 排队时延 | `sglang:queue_time_seconds` P95 | 请求排队时间 | 超过 TTFT SLO 的 50% |
| 确证① 抢占 | `sglang:num_retracted_requests_total` 增量 | 显存不足触发回撤 | 只要 $>0$ 即为**硬过载** |
| 确证② SLO | `sglang:time_to_first_token_seconds` / `sglang:inter_token_latency_seconds` | 用户可感知时延 | 超过业务 SLO |

**集群过载 = 所有节点的核心水位都越线**；若只有部分节点越线，那是**负载倾斜**（路由问题），不是容量问题。二者的处置动作完全不同。

---

## 1. 先搞清指标从哪来、什么时候更新

判错阈值大多源于误解采样语义，先明确三条链路。

### 1.1 三个数据面

| 出口 | 提供者 | 内容 | 典型用途 |
| --- | --- | --- | --- |
| `GET /metrics` | `SchedulerMetricsCollector` + `TokenizerMetricsCollector` | Prometheus 全量指标（`sglang:*`） | 监控大盘、告警 |
| `GET /v1/loads` | `SchedulerLoadInquirer` → `LoadSnapshot`（共享内存/ZMQ） | 精简负载快照，低开销 | 路由器/网关实时选点 |
| `GET /health`、`/health_generate` | `http_server.py` | 发一个 1-token 的探测请求 | 存活探测 |

`/v1/loads` 不走 ZMQ 往返调度器，而是直接读 `/dev/shm` 上的负载快照，所以可以高频拉取；`/metrics` 是重接口，按抓取周期（15s/30s）拉即可。

### 1.2 指标标签与「谁在上报」

调度器指标的固定标签（`metrics_collector.py::init_new`）：

```
model_name, engine_type, tp_rank, pp_rank, moe_ep_rank [, dp_rank] [, priority]
```

- **非 PD 分离下 `engine_type="unified"`**（`DisaggregationMode.to_engine_type` 对 `null` 返回 `unified`）。做面板时用 `engine_type="unified"` 过滤即可排除 P/D 节点。
- 默认**只有 `attn_tp_rank == 0` 的 scheduler 上报**；开了 `--enable-metrics-for-all-schedulers` 才每个 TP rank 各报一份。**聚合时务必用 `max`/`avg by (dp_rank)`，不要 `sum`**，否则 TP/DP 会重复计数。

### 1.3 更新节奏（决定了告警的 `for` 时长下限）

- decode 侧的重指标（水位、队列、吞吐）**每 `--decode-log-interval`（默认 40）个 decode step 才刷一次**；
- prefill 侧每次 prefill batch 刷一次；
- 完全空闲时兜底每 30s 刷一次（`_maybe_log_idle_metrics`）。

所以：**告警 `for` 至少给 1~2 分钟**，避免抓到「刚好在两次刷新之间」的陈旧值。

---

## 2. 节点过载：五个维度的指标

### 2.1 维度一：KV 显存水位（最关键）

**首选 `sglang:token_usage`。** 它的定义在 `pool_stats_observer.py`：

```text
# pool_stats_observer.py :: SchedulerPoolStatsObserver._get_token_info
available_size = self.token_to_kv_pool_allocator.available_size()
evictable_size = self.tree_cache.evictable_size()
num_used = self.max_total_num_tokens - (available_size + evictable_size)
token_usage = num_used / self.max_total_num_tokens
```

即：

$$
\text{token\_usage} \;=\; \frac{\text{max\_total\_num\_tokens} - (\text{available} + \text{evictable})}{\text{max\_total\_num\_tokens}}
$$

**关键点：分子已经扣掉了 `evictable`（可被淘汰的 radix 前缀缓存）**，所以 `token_usage` 衡量的是**不可释放的硬占用**。这也是它能直接作为过载判据的原因——它高，就意味着真的没地方放新请求的 KV 了。

混合架构下 `token_usage` 取各池的**瓶颈值**（`get_max_pool_usage`）：

$$
\text{token\_usage} = \max(\text{full\_token\_usage},\ \text{swa\_token\_usage},\ \text{mamba\_usage})
$$

配套细分指标（排查用，不作告警主判据）：

| 指标 | 用途 |
| --- | --- |
| `sglang:full_token_usage` / `sglang:swa_token_usage` / `sglang:mamba_usage` | 定位是哪个池先满（SWA 模型、Mamba 混合模型） |
| `sglang:kv_used_tokens` / `sglang:kv_evictable_tokens` / `sglang:kv_available_tokens` | 绝对值三分解，判断「缓存挤占」还是「真占用」 |
| `sglang:num_used_tokens` | 已用 token 绝对数 |
| `sglang:max_total_num_tokens` | 容量常量（启动时 `emit_constants` 一次性写入） |

**一个高价值的派生判据——缓存被挤压率**：

$$
\text{cache\_squeeze} = \frac{\text{kv\_evictable\_tokens}}{\text{max\_total\_num\_tokens}}
$$

它从高位快速跌到近 0，说明前缀缓存正在被新请求疯狂驱逐——通常紧跟着就是 `cache_hit_rate` 崩塌和吞吐掉坡。这是**过载的先行指标**，比 `token_usage` 触顶更早。

### 2.2 维度二：抢占/回撤（硬过载的确证）

KV 真的不够时，调度器会把正在 decode 的请求踢回等待队列（retract），`scheduler.py::update_running_batch`：

```text
# scheduler.py :: Scheduler.update_running_batch
if (kv_full_retract_flag := not batch.check_decode_mem()) or (
    TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
):
    retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(...)
    # 日志："KV cache pool is full. Retract requests."
```

对应指标：

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:num_retracted_requests_total` | counter | 累计被回撤请求数 |
| `sglang:num_retracted_input_tokens_total` | counter | 回撤浪费的输入 token（需重算） |
| `sglang:num_retracted_output_tokens_total` | counter | 回撤丢弃的输出 token |
| `sglang:num_retracted_reqs` | gauge | 本上报周期回撤数 |
| `sglang:new_token_ratio` | gauge | 调度保守度系数 |

**判据：`rate(sglang:num_retracted_requests_total[5m]) > 0` 即判定节点硬过载。** 回撤意味着算过的 token 被丢弃重来，是纯粹的算力浪费，且会直接把这批请求的 TTFT 打成两倍以上。

`sglang:new_token_ratio` 是很好的**辅助信号**：它在没发生回撤时会持续衰减到 `min`，一旦回撤就被重置回 `init`（`new_token_ratio_tracker.py`）：

```text
# new_token_ratio_tracker.py :: NewTokenRatioTracker
def decay_step(self) -> None:
    # 没发生回撤时逐步衰减：越跑越乐观
    self.current = max(self.current - self.decay, self.min)

def reset(self) -> None:
    # 发生回撤后重置回最保守的 init
    self.current = self.init
```

所以 **`new_token_ratio` 长期贴在高位不衰减 = 系统在反复触发回撤**，即便 counter 采样间隔漏掉了尖峰，这个 gauge 也能暴露出来。

> 注意：`sglang:num_paused_reqs` 是**异步权重同步**造成的暂停（RL 场景），不是过载信号，不要混用。

### 2.3 维度三：并发槽位水位

| 指标 | 说明 |
| --- | --- |
| `sglang:num_running_reqs` | 当前 running batch 中的请求数 |
| `sglang:max_running_requests_under_SLO` | 容量常量（见下方坑） |
| `sglang:decode_sum_seq_lens` | decode 批次内所有序列长度之和 |

$$
\text{concurrency\_ratio} = \frac{\text{num\_running\_reqs}}{\text{max\_running\_requests}}
$$

`max_running_requests` 来自 `--max-running-requests` 或由显存自动推算，会打印在启动日志：

```
max_total_num_tokens=..., chunked_prefill_size=..., max_prefill_tokens=..., max_running_requests=..., context_len=...
```

**若 Prometheus 里 `sglang:max_running_requests_under_SLO` 缺失或为 0**，就用启动日志/部署模板里的静态值做分母（见 §3 的坑）。

`sglang:decode_sum_seq_lens` 是一个被低估的指标：同样的 `num_running_reqs`，序列长度和翻倍意味着 attention 的读带宽翻倍。**长上下文场景下应该用它、而不是请求数来衡量 decode 侧压力。**

### 2.4 维度四：排队深度与排队时延

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `sglang:num_queue_reqs` | gauge | 等待队列长度（不含 grammar 队列） |
| `sglang:num_grammar_queue_reqs` | gauge | 等待语法编译的请求数 |
| `sglang:queue_time_seconds` | histogram | 请求排队耗时分布 |
| `num_waiting_uncached_tokens`（`/v1/loads`） | — | 等待 prefill 且**未命中缓存**的 token 数 |

**判据组合**：

- `num_queue_reqs > 0` **持续存在**（不是瞬时尖峰）→ 接纳能力已被打满；
- `histogram_quantile(0.95, queue_time_seconds)` 逼近或超过 TTFT SLO → 排队已经吃掉 SLO 预算；
- `num_grammar_queue_reqs` 单独高 → **不是算力过载**，是 grammar 编译成了瓶颈（结构化输出场景），应查 `sglang:grammar_compilation_time_seconds`，扩容 GPU 没用。

`num_waiting_uncached_tokens` 比 `num_queue_reqs` 更精准地描述「还欠多少 prefill 算力」——它按 `req.seqlen - req.num_matched_prefix_tokens` 累加，扣除了前缀缓存命中部分（`load_inquirer.py::get_num_waiting_uncached_tokens`）。100 个短请求和 1 个 128k 长请求在 `num_queue_reqs` 上都是数字，在这个指标上差两个数量级。

**队列上限的硬边界**：`--max-queued-requests` 设置后，超限请求会被直接 abort（`scheduler.py::_abort_on_queued_limit`，返回 "The request queue is full."）。此时应监控 `sglang:num_aborted_requests_total`：

```text
# scheduler.py :: Scheduler._abort_on_queued_limit
if (
    self.max_queued_requests is None
    or len(self.waiting_queue) + 1 <= self.max_queued_requests
):
    return False
# 否则：abort 请求，message = "The request queue is full."
```

$$
\text{rate}(\text{sglang:num\_aborted\_requests\_total}[5m]) > 0 \;\Rightarrow\; \text{已在丢请求（最严重级别）}
$$

### 2.5 维度五：SLO 与算力效率（判定「是否真的疼」）

前四个维度描述**系统状态**，这一维度描述**用户感受**。水位高但 SLO 达标，那是资源用得好，不是过载。

| 指标 | 说明 | 非 PD 场景的特殊含义 |
| --- | --- | --- |
| `sglang:time_to_first_token_seconds` | TTFT 分布 | 受排队 + prefill 排队共同影响 |
| `sglang:inter_token_latency_seconds` | ITL（token 间隔）分布 | **非 PD 的核心痛点指标**，见下 |
| `sglang:e2e_request_latency_seconds` | 端到端时延 | 综合 |
| `sglang:gen_throughput` | 生成吞吐 token/s | 单看无意义，需与 `num_running_reqs` 一起看 |
| `sglang:fwd_occupancy` | GPU 前向占用率 % | 需 `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1` |
| `sglang:cache_hit_rate` | 前缀缓存命中率 | 掉坡 = 缓存被过载挤爆 |

**非 PD 分离特有的过载表征——ITL 抖动。** 在 unified 部署下 prefill 与 decode 在同一张卡上混跑，一个大 prefill chunk 插入 decode 批次会直接拉长这一步的 ITL。所以：

$$
\text{ITL}_{P99} / \text{ITL}_{P50} \gg 1 \;\Rightarrow\; \text{prefill 正在抢占 decode（非 PD 特有）}
$$

这个比值在 PD 分离下不会出现（decode 节点不做 prefill），是 unified 部署最值得单独盯的一条曲线。缓解手段是调 `--chunked-prefill-size` 变小，或启用 prefill delayer（`--prefill-delayer-max-delay-passes` 等，**仅在非 PD + overlap 调度下可用**），对应观测指标 `sglang:prefill_delayer_wait_forward_passes`、`sglang:prefill_delayer_wait_seconds`、`sglang:prefill_delayer_outcomes_total`。

**吞吐塌陷的判定**（区分「满负荷高效运行」与「过载抖动」）：

$$
\text{efficiency} = \frac{\text{gen\_throughput}}{\text{num\_running\_reqs}}
$$

健康时该值随并发上升缓慢下降；**过载时它会陡降**（因为算力被回撤重算、缓存 miss 重新 prefill 吃掉了）。

`sglang:fwd_occupancy` 给出更直接的答案：

- 高水位 + 高 `fwd_occupancy`（>90%）→ **GPU 真的算不过来**，需要扩容；
- 高水位 + 低 `fwd_occupancy`（<60%）→ **不是算力瓶颈**，是调度/内存/CPU 侧问题，扩容浪费钱。

### 2.6 节点过载分级判定表

| 级别 | 判据（同时满足） | 语义 | 动作 |
| --- | --- | --- | --- |
| **健康** | `token_usage < 0.8` 且 `num_queue_reqs == 0` 且 SLO 达标 | 有余量 | — |
| **饱和（期望态）** | `token_usage ∈ [0.8, 0.9]`，`num_queue_reqs` 小幅波动，SLO 达标 | 资源用满但不疼 | 保持 |
| **预警** | `token_usage > 0.9` 持续 2min，或 `queue_time` P95 > 0.5×TTFT SLO，或 `cache_hit_rate` 环比下降 30% | 开始排队 | 准备扩容 / 限流 |
| **过载** | 预警条件 + `rate(num_retracted_requests_total) > 0` 或 SLO 已破 | 抢占已发生，算力在浪费 | 立即限流 + 扩容 |
| **崩边缘** | `rate(num_aborted_requests_total) > 0`，或 `/health_generate` 超时，或 `sglang:num_queue_reqs` 单调增长不回落 | 在丢请求 | 摘流 + 扩容 |

---

## 3. 不要用（或要小心用）的指标

这一节很重要，能省掉大量误判。

### 3.1 `sglang:utilization` —— 当前实现有缺陷，不要作为判据

```text
# metrics_reporter.py :: SchedulerMetricsReporter._calculate_utilization
if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
    self.stats.utilization = -1
else:
    # TODO: max_running_requests_under_SLO has no setter
    #       -> sglang:utilization stuck at 0 (regressed #22713)
    max_under_slo = getattr(self.scheduler, "max_running_requests_under_SLO", None)
    if max_under_slo is not None and max_under_slo > 0:
        self.stats.utilization = max(
            self.stats.num_running_reqs.total / max_under_slo,
            self.stats.token_usage / 0.9,
        )
```

设计意图是很好的复合指标：

$$
\text{utilization} = \max\!\left(\frac{\text{num\_running\_reqs}}{\text{max\_running\_requests\_under\_SLO}},\ \frac{\text{token\_usage}}{0.9}\right)
$$

但 `max_running_requests_under_SLO` **没有 setter**（代码内 TODO 明确标注，issue #22713），实际部署中该值为 `None`，整个分支不执行，**`sglang:utilization` 恒为 0**。

**替代方案**：在 Prometheus 侧自己算这个复合水位，用启动时已知的 `max_running_requests` 常量作分母：

```promql
max(
  sglang:num_running_reqs{engine_type="unified"} / <MAX_RUNNING_REQUESTS>,
  sglang:token_usage{engine_type="unified"} / 0.9
)
```

### 3.2 其他易误用项

| 指标 | 坑 |
| --- | --- |
| `sglang:num_prefill_bootstrap_queue_reqs` 等 4 个 PD 队列指标 | **非 PD 分离下恒为 0**，别放在面板上占位置 |
| `sglang:kv_transfer_*` | 同上，PD 专用 |
| `sglang:gen_throughput` 单独看 | 低吞吐可能只是没流量，必须与 `num_running_reqs` 联合判断 |
| `sglang:num_paused_reqs` | 是权重同步暂停，不是过载 |
| `sglang:cache_hit_rate`（decode 上报时） | decode 阶段该值被固定写 0，只在 prefill 上报时有意义 |
| 对 `sglang:token_usage` 做 `sum()` | TP/DP 多 rank 会重复，必须 `max` 或 `avg by (dp_rank)` |
| `sglang:fwd_occupancy` 默认值 | 未开 `SGLANG_ENABLE_METRICS_DEVICE_TIMER` 时为 `NaN` |

---

## 4. 集群过载判定

### 4.1 先分清「容量不足」还是「负载倾斜」

设集群有 $N$ 个 unified 节点，节点 $i$ 的水位为 $u_i$（取 §3.1 的复合水位或直接用 `token_usage`）：

$$
\bar{u} = \frac{1}{N}\sum_{i=1}^{N} u_i,
\qquad
u_{\max} = \max_i u_i,
\qquad
\mathrm{CV} = \frac{\sigma_u}{\bar{u}}
$$

| 情形 | 判据 | 结论 | 动作 |
| --- | --- | --- | --- |
| **集群过载** | $\bar{u} > 0.9$ 且 $\mathrm{CV} < 0.15$ | 均匀地满了，真的没容量 | 扩容 |
| **负载倾斜** | $u_{\max} > 0.9$ 但 $\bar{u} < 0.7$ 或 $\mathrm{CV} > 0.3$ | 路由把流量压到少数节点 | 改路由策略，扩容无效 |
| **容量临界** | $\bar{u} \in [0.8, 0.9]$ 且集群级排队 $>0$ | 一个节点故障就会雪崩 | 补冗余 |

**倾斜的常见根因**（非 PD 场景）：
- 使用了 `cache_aware` / `prefix_hash` / `consistent_hashing` 策略，热点前缀集中到少数节点；
- 使用 `round_robin` 而请求长度分布长尾（请求数均衡 ≠ token 均衡）；
- 部分节点刚重启（缓存冷），`cache_hit_rate` 低导致实际算力打折。

### 4.2 集群级排队总量

$$
Q_{\text{cluster}} = \sum_{i=1}^{N} \text{num\_queue\_reqs}_i,
\qquad
T_{\text{cluster}} = \sum_{i=1}^{N} \text{num\_waiting\_uncached\_tokens}_i
$$

$Q_{\text{cluster}}$ **单调增长不回落**是集群过载最直白的证据——说明到达率持续大于服务率：

$$
\lambda > \mu_{\text{cluster}} \;\Longrightarrow\; \frac{dQ}{dt} > 0
$$

此时无论怎么调路由都无解，只能扩容或限流。

### 4.3 用 `/v1/loads` 做实时倾斜检测

`/v1/loads?include=core` 是低开销接口（直接读共享内存），适合秒级轮询。核心字段：

```text
# load_inquirer.py :: SchedulerLoadInquirer.get_loads
GetLoadsReqOutput(
    dp_rank=...,                     timestamp=...,
    num_running_reqs=...,            num_waiting_reqs=...,
    num_waiting_uncached_tokens=..., num_used_tokens=...,
    num_total_tokens=...,            max_total_num_tokens=...,
    token_usage=...,                 gen_throughput=...,
    cache_hit_rate=...,              utilization=...,
    max_running_requests=...,
)
```

其中 `num_total_tokens` 是**跨节点比较负载最合适的单一标量**：

$$
\text{num\_total\_tokens} = \underbrace{\text{num\_used\_tokens}}_{\text{GPU 上已占的 KV}} + \sum_{\text{req} \in \text{等待队列}} \text{req.seqlen}
$$

它同时包含「在跑的」和「排队的」两部分负债，量纲统一为 token，不受请求长度分布影响。这也正是 SGLang DP 均衡 `--load-balance-method total_tokens` 和网关 Power-of-Two 策略所用的信号。

> ⚠️ 两个已知偏差，用它做判据时要心里有数：
> 1. **排队部分未扣前缀缓存命中**：直接累加完整 `seqlen`，在高前缀复用场景会**高估**负载（偏保守，换取零热路径开销）；
> 2. **网关解析字段不一致**：`sgl-model-gateway` 的 `parse_load_response` 读的是 `json["aggregate"]["total_tokens"]`，而引擎原生返回的是 `{"loads":[{"num_total_tokens":...}]}`（无 `aggregate` 层）。若中间没有格式转换层，网关会拿到 `-1` 并**静默退化为按请求计数比较**。接 P2C 策略前务必验一次。

### 4.4 DP 内部的均衡（单实例多 DP rank）

开启 `--dp-size > 1` 时，`DataParallelController` 用 `DPBudget` 在各 DP rank 间分发：

```text
# data_parallel_controller.py :: DPBudget.update_budget
for load in loads:
    if load.timestamp == self.last_timestamp[load.dp_rank]:
        continue  # 快照未更新，跳过以免回退到旧值
    self.last_timestamp[load.dp_rank] = load.timestamp
    self.total_requests[load.dp_rank] = load.num_running_reqs + load.num_waiting_reqs
    self.total_tokens[load.dp_rank] = load.num_total_tokens
```

监控要点：**按 `dp_rank` 分组看 `sglang:token_usage` 和 `sglang:num_running_reqs` 的离散度**。若 DP rank 间差异大：
- `--load-balance-method round_robin` → 改成 `total_tokens`；
- 已经是 `total_tokens` 仍不均 → 检查快照刷新（`refresh_load_budget` 有 20ms 节流，突发流量下靠推测式 +1 打散）。

### 4.5 网关侧的集群健康指标

`sgl-model-gateway` 暴露的 `smg_*` 指标是判断「集群是否已经在对外劣化」的最外层视角：

| 指标 | 过载语义 |
| --- | --- |
| `smg_http_rate_limit_total{result="rejected"}` | 网关已在限流拒绝，$>0$ 即入口过载 |
| `smg_worker_cb_state` | 熔断器状态（0=closed / 1=open / 2=half_open），出现 1 即有节点被摘 |
| `smg_worker_retries_exhausted_total` | 重试耗尽，后端普遍不可用 |
| `smg_http_inflight_request_age_count` | 在途请求年龄分桶，长尾桶（>300s）堆积 = 请求卡住 |
| `smg_worker_requests_active` | **按 worker 的活跃请求数，直接算集群倾斜度** |
| `smg_worker_pool_size` vs 健康节点数 | 可用容量是否已缩水 |
| `smg_router_ttft_seconds` / `smg_router_tpot_seconds` | 网关视角的 SLO（gRPC 模式） |

**判据组合**：

$$
\text{集群对外过载} \iff
\begin{cases}
\text{rate}(\texttt{smg\_http\_rate\_limit\_total\{result="rejected"\}}) > 0 \\
\text{或}\quad \max(\texttt{smg\_worker\_cb\_state}) = 1 \\
\text{或}\quad \texttt{smg\_router\_ttft\_seconds}_{P99} > \text{SLO}
\end{cases}
$$

注意区分：网关侧 `smg_worker_requests_active` 是**网关记账的在途请求数**，与引擎侧 `sglang:num_running_reqs` 口径不同（前者含网络在途、后者只算已进 running batch）。两者背离过大说明请求卡在排队或网络上。

---

## 5. 可直接落地的告警规则

以下 PromQL 假定单模型、按 `dp_rank` 区分实例；多模型再加 `model_name` 分组。

```yaml
groups:
- name: sglang-unified-overload
  rules:

  # ---------- 节点级 ----------
  - alert: SGLangKVPressureHigh
    expr: max by (instance, dp_rank) (sglang:token_usage{engine_type="unified"}) > 0.9
    for: 2m
    labels: {severity: warning}
    annotations:
      summary: "KV 池水位 > 90%，节点接近饱和"

  - alert: SGLangRetractionDetected      # 硬过载确证
    expr: rate(sglang:num_retracted_requests_total{engine_type="unified"}[5m]) > 0
    for: 1m
    labels: {severity: critical}
    annotations:
      summary: "发生请求回撤（KV 不足抢占），算力正在被浪费"

  - alert: SGLangQueueBuildup
    expr: |
      max by (instance, dp_rank) (sglang:num_queue_reqs{engine_type="unified"}) > 0
      and
      deriv(sglang:num_queue_reqs{engine_type="unified"}[10m]) > 0
    for: 5m
    labels: {severity: warning}
    annotations:
      summary: "等待队列持续增长，到达率 > 服务率"

  - alert: SGLangQueueTimeSLOBreach
    expr: |
      histogram_quantile(0.95,
        sum by (le, instance) (rate(sglang:queue_time_seconds_bucket{engine_type="unified"}[5m]))
      ) > 1.0                              # 按 TTFT SLO 的 50% 设定
    for: 3m
    labels: {severity: warning}

  - alert: SGLangRequestsDropped         # 最严重
    expr: rate(sglang:num_aborted_requests_total[5m]) > 0
    for: 1m
    labels: {severity: critical}
    annotations:
      summary: "队列已满，正在拒绝请求"

  # 非 PD 特有：prefill 抢占 decode
  - alert: SGLangITLJitterHigh
    expr: |
      histogram_quantile(0.99, sum by (le) (rate(sglang:inter_token_latency_seconds_bucket[5m])))
      /
      histogram_quantile(0.50, sum by (le) (rate(sglang:inter_token_latency_seconds_bucket[5m])))
      > 5
    for: 5m
    labels: {severity: warning}
    annotations:
      summary: "ITL P99/P50 > 5，prefill 正在抢占 decode，考虑调小 chunked-prefill-size"

  # 前缀缓存被挤爆（过载先行指标）
  - alert: SGLangCacheSqueezed
    expr: |
      (sglang:kv_evictable_tokens / sglang:max_total_num_tokens) < 0.05
      and sglang:token_usage > 0.85
    for: 3m
    labels: {severity: warning}

  # ---------- 集群级 ----------
  - alert: SGLangClusterSaturated        # 均匀地满 = 真缺容量
    expr: |
      avg(sglang:token_usage{engine_type="unified"}) > 0.9
      and
      stddev(sglang:token_usage{engine_type="unified"})
        / avg(sglang:token_usage{engine_type="unified"}) < 0.15
    for: 5m
    labels: {severity: critical}
    annotations:
      summary: "集群整体饱和且分布均匀，需要扩容"

  - alert: SGLangLoadSkew                # 倾斜 = 路由问题
    expr: |
      stddev(sglang:token_usage{engine_type="unified"})
        / avg(sglang:token_usage{engine_type="unified"}) > 0.3
      and max(sglang:token_usage{engine_type="unified"}) > 0.85
    for: 5m
    labels: {severity: warning}
    annotations:
      summary: "负载倾斜，扩容无效，请检查路由策略"

  # ---------- 网关级 ----------
  - alert: SGLangGatewayRejecting
    expr: rate(smg_http_rate_limit_total{result="rejected"}[5m]) > 0
    for: 1m
    labels: {severity: critical}

  - alert: SGLangWorkerCircuitOpen
    expr: max(smg_worker_cb_state) == 1
    for: 1m
    labels: {severity: critical}
```

---

## 6. 排查决策树

```
告警：SLO 破了 / 排队堆积
  │
  ├─ rate(num_aborted_requests_total) > 0 ?
  │    └─ 是 → 【崩边缘】立即摘流+扩容；复查 --max-queued-requests 是否过小
  │
  ├─ rate(num_retracted_requests_total) > 0 ?
  │    └─ 是 → 【KV 不足】
  │           ├─ 调大 --mem-fraction-static / 换更大显存
  │           ├─ 调小 --max-running-requests（少接一点，别抢崩）
  │           └─ 检查是否有超长 context 请求（decode_sum_seq_lens 尖峰）
  │
  ├─ token_usage > 0.9 且 fwd_occupancy > 90% ?
  │    └─ 是 → 【真算不过来】扩容
  │
  ├─ token_usage > 0.9 但 fwd_occupancy < 60% ?
  │    └─ 是 → 【非算力瓶颈】查 CPU 侧 / 调度 / detokenizer / 网络
  │
  ├─ ITL_P99/ITL_P50 > 5 ?
  │    └─ 是 → 【prefill 抢 decode，非 PD 特有】
  │           ├─ 调小 --chunked-prefill-size
  │           ├─ 启用 prefill delayer 并观察 prefill_delayer_* 指标
  │           └─ 长期方案：改 PD 分离部署
  │
  ├─ num_grammar_queue_reqs 高但 token_usage 低 ?
  │    └─ 是 → 【grammar 编译瓶颈】查 grammar_compilation_time_seconds，
  │             不是 GPU 问题
  │
  ├─ cache_hit_rate 骤降 + kv_evictable_tokens 趋 0 ?
  │    └─ 是 → 【缓存被挤爆】路由亲和性失效 / 流量特征变化，
  │             查是否刚做过扩缩容导致哈希环重排
  │
  └─ 单节点高、集群均值低（CV > 0.3）?
       └─ 是 → 【倾斜】改路由策略（round_robin → total_tokens / cache_aware 调阈值），
                扩容无效
```

---

## 7. 面板建议（最小可用大盘）

一屏放 6 张图，按「状态 → 压力 → 后果」排列：

| # | 图 | 指标 | 判读 |
| --- | --- | --- | --- |
| 1 | KV 水位 | `max by(dp_rank) (sglang:token_usage)` + 0.9 阈值线 | 主水位 |
| 2 | 并发与队列 | `sglang:num_running_reqs`、`sglang:num_queue_reqs` 双 Y 轴 | 接纳能力 |
| 3 | 抢占 | `rate(sglang:num_retracted_requests_total[5m])` + `sglang:new_token_ratio` | 硬过载确证 |
| 4 | SLO | TTFT / ITL / e2e 的 P50·P95·P99 | 用户感受 |
| 5 | 效率 | `sglang:gen_throughput`、`sglang:fwd_occupancy`、`sglang:cache_hit_rate` | 区分「满」与「废」 |
| 6 | 倾斜 | `stddev/avg (sglang:token_usage)` + 各节点 `smg_worker_requests_active` | 容量 vs 路由 |

---

## 8. 速查表

**节点过载（非 PD / unified）**

```
主判据    sglang:token_usage                        > 0.9 持续
          sglang:num_running_reqs / max_running_requests  ≥ 0.95 持续
          sglang:num_queue_reqs                     > 0 且增长
          histogram_quantile(0.95, sglang:queue_time_seconds)  > 0.5 × TTFT SLO

确证      rate(sglang:num_retracted_requests_total[5m]) > 0     ← 硬过载
          rate(sglang:num_aborted_requests_total[5m])   > 0     ← 已丢请求

先行      kv_evictable_tokens / max_total_num_tokens → 0        ← 缓存被挤
          sglang:new_token_ratio 长期不衰减                      ← 反复回撤
          sglang:cache_hit_rate 骤降

非 PD 特有 ITL_P99 / ITL_P50 > 5                                ← prefill 抢 decode

辅助      sglang:fwd_occupancy      区分算力瓶颈 vs 调度瓶颈
          sglang:decode_sum_seq_lens 长上下文压力
          sglang:num_grammar_queue_reqs  grammar 瓶颈（非 GPU）

不要用    sglang:utilization        恒为 0（#22713）
          sglang:num_*_queue_reqs（PD 四件套）  非 PD 下恒 0
```

**集群过载**

```
容量不足  avg(token_usage) > 0.9  且  CV < 0.15
负载倾斜  CV > 0.3  且  max(token_usage) > 0.85     → 改路由，别扩容
入口劣化  smg_http_rate_limit_total{rejected} > 0
          max(smg_worker_cb_state) == 1
          smg_http_inflight_request_age_count 长尾桶堆积
排队总量  Σ num_queue_reqs 单调增长                   → λ > μ，只能扩容/限流
跨节点比较统一标量：/v1/loads 的 num_total_tokens

