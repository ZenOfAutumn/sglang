# SGLang KV Cache 调度与 PD 分离全流程

> 本文串联「请求参数 → 调度预算 → KV 分配 → 回退 → PD 分离 → 重算」的完整链路，并给出每个环节对应的观测指标。所有结论可溯源到代码。

| 主题 | 代码位置 |
| --- | --- |
| 请求 / 采样参数结构 | `python/sglang/srt/managers/io_struct.py`、`managers/schedule_batch.py` |
| OpenAI 参数映射 | `python/sglang/srt/entrypoints/openai/protocol.py` |
| 调度预算（PrefillAdder） | `python/sglang/srt/managers/schedule_policy.py` |
| 调度主循环 | `python/sglang/srt/managers/scheduler.py` |
| new_token_ratio | `python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py` |
| PD 分离 — Prefill 端 | `python/sglang/srt/disaggregation/prefill.py` |
| PD 分离 — Decode 端 | `python/sglang/srt/disaggregation/decode.py` |
| PD 分离 — 通用连接/重算投递 | `python/sglang/srt/disaggregation/common/conn.py` |
| 指标定义 | `python/sglang/srt/observability/metrics_collector.py` |

> 配套阅读：`docs/metrics/endpoints_metrics_and_loads_zh.md`（`/metrics` 与 `/v1/loads` 接口详解）。

---

## 0. 全局关系（一图看懂）

```
                       ┌─────────────────────── 统一部署 / PD 的 D 节点 ───────────────────────┐
请求参数                │                                                                        │
max_tokens ──▶ max_new_tokens ──▶ 调度预算估计（new_token_ratio × min(max_new, CLIP)）            │
                       │                        │                                               │
                       │                        ▼                                               │
                       │   waiting_queue ──▶ PrefillAdder 准入 ──▶ can_run_list                  │
                       │                        │                        │                      │
                       │              不满足预算则留在 waiting           ▼                      │
                       │                                     prepare_for_extend                  │
                       │                                          │                             │
                       │                                          ▼                             │
                       │                                   alloc_for_extend（真正分配 KV）         │
                       │                                          │                             │
                       │                                          ▼                             │
                       │                                   前向 / 生成 decode                    │
                       │                                          │                             │
                       │                      KV 不够 ──▶ check_decode_mem 失败 ──▶ retract_decode │
                       └────────────────────────────────────────────────────────────────────────┘

                       ┌────────────────────── PD 分离 ──────────────────────┐
   P 节点：max_new_tokens 强制=1，只产首 token，做握手/传输              │
   D 节点：prealloc 队列（显存估算）──▶ transfer 队列（收 KV）──▶ decode  │
   回退时：D 通知 P 重算（rebootstrap / HTTP /generate）                  │
                       └─────────────────────────────────────────────────────┘
```

三个核心概念贯穿全文：

1. **预算估计**：调度器不知道请求最终生成多少 token，用 `new_token_ratio` 做乐观折扣来预留 KV。
2. **真正分配**：KV slot 只在请求进入批次、执行 `prepare_for_extend` 时才分配，排队中的请求不占 KV。
3. **PD 分离**：P 端只做 prefill 与 KV 传输，D 端负责 decode；两端各有一套队列与显存判据。

---

## 1. 术语与角色

| 术语 | 含义 |
| --- | --- |
| `engine_type` | `unified`（非 PD）/ `prefill`（PD 的 P 节点）/ `decode`（PD 的 D 节点） |
| `req.sampling_params.max_new_tokens` | 请求最多生成的新 token 数（内部字段，见 §2） |
| KV pool / `token_to_kv_pool_allocator` | KV Cache 物理池，按 token（页对齐）分配 |
| `req_to_token_pool` | 每个请求一行，记录其 token 到 KV 槽位的映射 |
| `new_token_ratio` | 对「未来还要生成多少 token」的乐观折扣系数（§4） |
| `CLIP_MAX_NEW_TOKENS` | 预算估计时对 `max_new_tokens` 的上限（§4） |
| retract（回退） | 显存不足时把部分在跑请求踢回等待队列、释放 KV（§5） |
| rebootstrap | PD 分离下回退请求由 P 端重算前缀 KV（§7） |

---

## 2. 请求参数：`max_tokens` → `max_new_tokens`

内部统一使用 `sampling_params.max_new_tokens`，不同接口的入口名不同：

- **OpenAI 兼容接口**：`max_tokens`（`max_completion_tokens` 优先）直接映射为 `max_new_tokens`：

```python
# entrypoints/openai/protocol.py
sampling_params = {
    "temperature": get_param("temperature"),
    "max_new_tokens": self.max_completion_tokens or self.max_tokens,
    "min_new_tokens": self.min_tokens,
    ...
}
```

- **原生 `/generate` 接口**：无 `max_tokens` 别名，直接在 `sampling_params` 中传 `max_new_tokens`。

因此 `req.sampling_params.max_new_tokens` 语义上就是「最多生成的新 token 数」。它会进一步受两处约束：

1. 剩余上下文长度；
2. 调度预算里的 `CLIP_MAX_NEW_TOKENS`（仅影响估算，见 §4）。

---

## 3. KV Cache 的分配时机

**关键结论：排队中、尚未被调度进批次的请求不占用任何 KV slot。**

分配发生在「请求被调度器选中、加入本批」后的前向前一刻：

```python
# managers/schedule_batch.py
def prepare_for_extend(self):
    # 1) 取去掉已命中前缀后要新算的 input token
    # 2) 计算 seq_lens / prefix_lens / extend_lens 等
    # 3) alloc_for_extend 分配 KV 槽位与 req_pool 行号
    ...
```

- `PrefillAdder.add_one_req()` 只做**准入预算估算**（决定"能不能加"），不做真正的 KV 分配；
- 真正的 `alloc_for_extend` 在 `prepare_for_extend` 内执行；
- **chunked prefill** 下按当前 chunk 的 `extend_range` **增量分配**，不会为整条序列预留。

> 唯一相关的是 PD 的 P 节点：处于 `disagg_prefill_bootstrap_queue`（握手）或 `disagg_prefill_inflight_queue`（传输中）的请求会占用 `req_to_token_pool` 行，从而减少可分配名额（`scheduler.py` 中在 PREFILL 模式额外检查 `req_to_token_pool.available_size()`），但这些属于**已调度**请求，不属于 waiting 队列。

---

## 4. 调度预算估计

### 4.1 `CLIP_MAX_NEW_TOKENS`：只裁估算，不改停止条件

```python
# managers/schedule_policy.py
# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
CLIP_MAX_NEW_TOKENS = int(
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")
)
```

- 作用：请求声明 `max_new_tokens=100000` 时，调度器按 `min(max_new_tokens, 4096)` 估算预留显存，避免为一个请求预留过多而压低并发。
- 只影响**估计值**，请求仍可一直生成到其原始 `max_new_tokens`。
- ⚠️ 与 `SGLANG_MAX_NEW_TOKENS_LIMIT` 不同：后者是**真正强制截断**请求的 `max_new_tokens`（`scheduler.py` 中改写请求），会改变生成上限。

### 4.2 `new_token_ratio`：对预留量的自适应折扣

`new_token_ratio` 介于 `(min, init]`，用于把「声明的 `max_new_tokens`」折算成「实际可能生成的 token 数」——系数越大越保守（预留多、更不易回退）。实现见 `new_token_ratio_tracker.py`。

**① 初始值（`from_config`）**

$$
\text{init}=\min(\underbrace{0.7}_{\text{SGLANG\_INIT\_NEW\_TOKEN\_RATIO}}\times \text{schedule\_conservativeness},\ 1.0)
$$

$$
\text{min}=\min(\text{init}\times \underbrace{0.14}_{\text{SGLANG\_MIN\_NEW\_TOKEN\_RATIO\_FACTOR}},\ 1.0)
$$

$$
\text{decay}=\frac{\text{init}-\text{min}}{\underbrace{600}_{\text{SGLANG\_NEW\_TOKEN\_RATIO\_DECAY\_STEPS}}},\qquad \text{current}=\text{init}
$$

`schedule_conservativeness` 由 `--schedule-conservativeness` 控制（默认 1.0，调大更保守）。

**② 运行时演进**

- **无回退**：每个 decode step 调 `decay_step()`，`current` 线性下降一个 `decay`、不低于 `min`（越跑越乐观，逐步释放余量换吞吐）。
- **发生回退**：现场重估，用剩余请求的实际进度：

$$
\text{ratio}=\min\left(1.0,\ \frac{\sum_{\text{req}}\text{len(output\_ids)} + \underbrace{20}_{\text{SGLANG\_RETRACT\_DECODE\_STEPS}}\times N}{\sum_{\text{req}}\text{max\_new\_tokens}+1}\right)
$$

结果写回 `tracker.current`。

**③ 用途**

作为 `PrefillAdder` 的 `new_token_ratio`，估算单请求未来需预留的 KV：

$$
\min(\text{max\_new\_tokens}-\text{已生成},\ \text{CLIP\_MAX\_NEW\_TOKENS})\times \text{new\_token\_ratio}
$$

### 4.3 并发上限：`num_running_reqs` 与 `max_running_requests`

一般有 `num_running_reqs ≤ max_running_requests`，但存在**跳过该检查**的例外：

- **chunked prefill**（尤其 PP 场景）：分块请求可跨微批跳跃，每微批的 `max_running_requests` 不作严格限制（否则会造成显存泄漏）；
- **优先级抢占**（`enable_priority_preemption`）：满足条件时允许先加后抢。

此时 `num_running_reqs` 可能瞬时 `> max_running_requests`。

---

## 5. Retract（回退 / 抢占）

### 5.1 触发条件

回退的直接判据是「**下一步 decode 整批是否装得下**」，而非单个请求的估算：

```python
# managers/schedule_batch.py
def check_decode_mem(self, selected_indices=None):
    """先按需触发 tree cache 淘汰腾空间，再判断下一步 decode 是否装得下。"""
    num_tokens = self.new_tokens_required_next_decode(selected_indices)
    evict_from_tree_cache(self.tree_cache, num_tokens)
    return self.token_to_kv_pool_allocator.available_size() >= num_tokens
```

### 5.2 根因

调度器为在跑请求预留的 KV 是**乐观折扣**：

$$
\min(\text{max\_new\_tokens},\ \text{CLIP\_MAX\_NEW\_TOKENS})\times \text{new\_token\_ratio},\quad \text{new\_token\_ratio}<1
$$

这是为提升吞吐而**故意**做的乐观估计（赌请求不会跑满 `max_new_tokens`）。当实际生成长度超过该折扣预留、且池里又无可驱逐/空闲空间时，就触发回退。其它触发源还包括：并发过高、前缀缓存被锁满无法驱逐、其他请求抢占显存等。

### 5.3 处理

```python
# managers/schedule_batch.py
def retract_decode(self):
    """Retract the decoding requests when there is not enough memory."""
    sorted_indices = self._get_decode_retraction_order(self.reqs)
    ...
    while first_iter or (not self.check_decode_mem(selected_indices=sorted_indices)):
        ...
```

- 按排序把部分请求踢回 `waiting_queue`、**立即释放其 KV**（后续需重新调度）；
- 目的：让**剩余请求的下一步 decode 能执行**；
- 同时用实际进度重估 `new_token_ratio`，使下次更保守；
- 被回退请求留下 `retraction_stain` 标记与 `retraction_count` 计数（用于统计与调度惩罚）。

---

## 6. PD 分离：Prefill 端行为

P 节点只做 prefill 并产出**首 token**，不负责后续生成，因此：

```python
# disaggregation/prefill.py
def _process_req(self, req: Req) -> None:
    """
    Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
    中译：把 max_new_tokens 设为 1，使 PrefillAdder 的显存估算准确（prefill 节点只产首 token）。
    """
    req.sampling_params.max_new_tokens = 1
```

请求在进入 `waiting_queue` 之前（bootstrap 阶段）就被改写。由此：

$$
\text{max\_new}=\min(1,\ \text{CLIP\_MAX\_NEW\_TOKENS})=1
$$

`new_token_ratio` 只是对这 1 个 token 做折扣，影响可忽略；且 P 端没有 decode 的 `running_batch`。**结论：`new_token_ratio` 对 P 端调度基本不起作用。**

P 端请求链路：`bootstrap_queue`（握手）→ `waiting_queue` → 调度进批 → `inflight_queue`（KV 传输中）。

---

## 7. PD 分离：Decode 端 KV 生命周期与卡点

### 7.1 队列与阶段

D 端请求链路：`disagg_decode_prealloc_queue`（预分配 KV）→ `disagg_decode_transfer_queue`（接收 KV）→ decode 运行。

### 7.2 显存估算过大时的卡点

D 端每轮在 `pop_preallocated()` 中对队列请求做显存估算，不满足就直接 `break`（不给它预分配 KV）：

```python
# disaggregation/decode.py
required_tokens_for_request = required_alloc_tokens + self.num_reserved_decode_tokens

if (
    max(
        required_tokens_for_request,
        origin_input_len - prefix_len
        + min(decode_req.req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
        - retractable_tokens,
    )
    > full_allocatable_tokens
):
    ...
    break
if required_tokens_for_request > full_allocatable_tokens:
    ...
    break
```

即需满足：

$$
\max\Big(\text{req\_alloc}+\text{reserved\_decode},\ \ \text{origin\_len}-\text{prefix\_len}+\min(\text{max\_new},\text{CLIP})-\text{retractable}\Big)\le \text{full\_allocatable}
$$

否则请求**滞留在 prealloc 队列**，不进入 transfer 队列，也不会去接收 P 传来的 KV。表现：

- `sglang:num_decode_prealloc_queue_reqs` 持续 > 0、`sglang:pending_prealloc_token_usage` 累积升高，TTFT 拉长；
- P 端对应请求可能停在 inflight 等接收。

> 注意区分：若请求长度**超过 KV 池总容量**，会在入队前被 `_check_if_req_exceed_kv_capacity` 直接 abort（属于失败，不是「卡」）。
> 只有等 D 端 running 请求完成释放 KV、或触发 retract 腾出空间后，该请求才会被预分配并进入传输阶段。

---

## 8. PD 分离：回退后的重算（true-retraction rebootstrap）

当 D 端回退请求时，KV 不做 device→host 备份，而是让原 P 节点**重算前缀 KV**。链路如下。

### 8.1 D 端回退时打标记

```python
# managers/scheduler.py
if self.disaggregation_mode == DisaggregationMode.DECODE:
    if req.output_ids:
        req.pd_rebootstrap_forced_output_id = req.output_ids.pop()
    req.pd_rebootstrap_in_progress = True
    req.time_stats.set_retract_time()
    self.disagg_decode_prealloc_queue.hold_rebootstrap(req)
```

注意 `retract_all(..., offload_kv=False)`：PD decode 回退**不做 device→host 备份**，因为要改为让 P 重算。

### 8.2 恢复时以 `is_rebootstrap=True` 重新入队，并在预分配成功后触发重算

```python
# disaggregation/decode.py（pop_preallocated 内）
if decode_req.is_rebootstrap:
    self.kv_manager.submit_prefill_recompute(
        decode_req.kv_receiver,
        decode_req.req.build_rebootstrap_payload(),
    )
```

### 8.3 构造「重算请求」payload

`input_ids = origin_input_ids + output_ids`（原 prompt + 已生成的全部 token，即要重算的完整前缀），`max_new_tokens=1`，并带上原 `bootstrap_host/port/room`、`rid`、`disagg_prefill_dp_rank`：

```python
# managers/schedule_batch.py
def build_rebootstrap_payload(self) -> dict:
    ...
    return {
        "input_ids": [int(x) for x in self.origin_input_ids]
        + [int(x) for x in self.output_ids],
        "sampling_params": {"max_new_tokens": 1, ...},
        "rid": self.rid,
        "bootstrap_host": self.bootstrap_host,
        "bootstrap_port": self.bootstrap_port,
        "bootstrap_room": self.bootstrap_room,
        ...
        "disagg_prefill_dp_rank": self.disagg_prefill_dp_rank,
    }
```

### 8.4 目标 URL 与投递

D 端凭 bootstrap 信息拼出 `http://{bootstrap_host}:{prefill_http_port}`（P 在 bootstrap 注册时把 HTTP 端口写进 `PrefillServerInfo`），在线程池里异步 POST，**只有 leader rank 提交**，避免重复重算：

```python
# disaggregation/common/conn.py
if self.attn_tp_rank != 0 or self.attn_cp_rank != 0 or self.pp_rank != 0:
    return
prefill_url = self._resolve_rebootstrap_prefill_url(kv_receiver)
```

### 8.5 P 端响应

P 端收到该 `/generate` 后，按当前权重重新 prefill 这段完整前缀、只产 1 个 handoff token，再把 KV 通过**已有的 bootstrap 通道**传回 D；D 侧在 transfer commit 时用 `pd_rebootstrap_forced_output_id` 覆盖那 1 个 token，保证与已输出内容一致。

**信号本质**：一次指向原 P 节点 `/generate` 的「把 origin + output token 重算一遍」的请求，靠相同的 `rid`/`bootstrap_room` 与已建好的传输通道完成关联与回传——而不是发送一个请求列表。

---

## 9. 观测指标

### 9.1 Prefill 算力负载

**`/v1/loads`（首选，给路由选点）**

- `num_waiting_uncached_tokens` —— **真正欠的 prefill 算力**：

$$
\text{num\_waiting\_uncached\_tokens}=\sum_{\text{req}\in\text{等待队列}}\max\bigl(0,\ \text{req.seqlen}-\text{req.num\_matched\_prefix\_tokens}\bigr)
$$

（纯 DECODE 节点无 prefill，返回 0。）
- `num_total_tokens`、`num_running_reqs`、`num_waiting_reqs` —— 排队/在跑负债，跨节点比较的统一标量。

**`/metrics`（需按 `engine_type="prefill"` 过滤）**

- 排队压力：`sglang:num_queue_reqs`
- P 端专用队列：`sglang:num_prefill_bootstrap_queue_reqs`、`sglang:num_prefill_inflight_queue_reqs`、`sglang:num_prefill_retries_total`
- 待预分配占比：`sglang:pending_prealloc_token_usage`
- 前缀缓存命中率（仅 prefill 上报有效）：`sglang:cache_hit_rate`
- 实际算力消耗/利用率（需 `--enable-mfu-metrics`）：`sglang:realtime_tokens_total{mode="prefill_compute"}`、`sglang:forward_execution_seconds_total`、`sglang:estimated_flops_per_gpu_total`
- GPU 占用率/前向次数：`sglang:fwd_occupancy`、`sglang:cuda_graph_passes_total`
- 累计/分布：`sglang:prompt_tokens_total`、`sglang:uncached_prompt_tokens_histogram`

### 9.2 Prefill 显存负载

**`/metrics`**

- 池使用率（水位）：

$$
\text{token\_usage}=\max(\text{full},\text{swa},\text{mamba}),\qquad
\text{usage}=\frac{\text{total}-(\text{available}+\text{evictable})}{\text{total}}
$$

  对应 `sglang:token_usage`、`sglang:full_token_usage`、`sglang:swa_token_usage`、`sglang:mamba_usage`
- 绝对 token 数：`sglang:num_used_tokens`、`sglang:kv_used_tokens`、`sglang:kv_available_tokens`、`sglang:kv_evictable_tokens`
- 容量上限：`sglang:max_total_num_tokens`；启动时可用显存：`sglang:startup_available_gpu_memory_gb`

**`/v1/loads`**

- 核心：`num_used_tokens`、`token_usage`、`max_total_num_tokens`
- `include=memory` 分区：`kv_cache_gb`、`weight_gb`、`graph_gb`、`token_capacity`

### 9.3 调度与回退

| 指标 | 含义 |
| --- | --- |
| `sglang:new_token_ratio` | 当前自适应折扣系数 |
| `sglang:num_decode_prealloc_queue_reqs` | D 端预分配队列长度（§7 卡点） |
| `sglang:num_decode_transfer_queue_reqs` | D 端传输队列长度 |
| `sglang:num_retracted_requests_total` | 累计回退请求数 |
| `sglang:num_retracted_input_tokens_total` / `_output_tokens_total` | 回退浪费的输入 / 输出 token |

> 聚合提示：`/metrics` 中这些指标带 `engine_type` 标签，务必用 `{engine_type="prefill"}` 过滤；水位类指标用 `max`/`avg by (dp_rank)`，**不能 `sum`**。

---

## 10. 全链路小结

1. **参数**：OpenAI 的 `max_tokens`/`max_completion_tokens` → 内部 `max_new_tokens`。
2. **估算**：调度用 `min(max_new_tokens, CLIP_MAX_NEW_TOKENS) × new_token_ratio` 估算需预留的 KV；`new_token_ratio` 随运行自适应（decay / 回退重估）。
3. **分配**：KV slot 仅在 `prepare_for_extend → alloc_for_extend` 时才真正分配；排队请求不占 KV；chunked prefill 按 chunk 增量分配。
4. **回退**：实际生成超过乐观预留、池子无空间时，`check_decode_mem` 失败 → `retract_decode` 踢请求、释放 KV、保住其余请求。
5. **PD 分离**：P 端 `max_new_tokens` 强制为 1（`new_token_ratio` 无意义）；D 端在 prealloc 队列做显存准入，估算过大即滞留。
6. **重算**：PD decode 回退不落盘，而是由 D 通过 HTTP `/generate` 通知原 P 节点重算前缀 KV 并回传。
