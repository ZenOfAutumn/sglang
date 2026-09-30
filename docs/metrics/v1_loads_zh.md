# `/v1/loads` 接口详解：请求与返回值

> 本文逐字段说明 SGLang HTTP Server `/v1/loads` 观测接口的**请求参数**与**返回结构**，所有内容均可溯源到代码：
>
> | 主题 | 代码位置 |
> | --- | --- |
> | 路由注册 | `python/sglang/srt/entrypoints/http_server.py` |
> | `/v1/loads` 处理 | `python/sglang/srt/entrypoints/v1_loads.py` |
> | 负载数据生产 | `python/sglang/srt/managers/scheduler_components/load_inquirer.py` |
> | 负载数据传输 | `python/sglang/srt/managers/load_snapshot.py` |
> | 请求/响应数据类 | `python/sglang/srt/managers/io_struct.py` |
>
> 配套阅读：`docs/metrics/endpoints_metrics_and_loads_zh.md`（`/metrics` 接口）、`docs/metrics/overload_metrics_zh.md`（如何用这些指标判断过载）

---

## 0. 与 `/metrics` 的定位对比

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

## 1. 请求

### 1.1 基本形式

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

### 1.2 查询参数

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

### 1.3 请求示例

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

### 1.4 数据链路（为什么这么快）

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

## 2. 返回值

### 2.1 JSON 格式（默认）

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

### 2.2 `loads[]` 核心字段（始终返回）

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
| `utilization` | float | 综合利用率（⚠️ 恒为 0，见 `endpoints_metrics_and_loads_zh.md` §2.5） |

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

#### `token_usage` 的含义与计算

`token_usage` 是 **KV 池的实时占用率**，衡量「不可立即释放的硬占用」占容量的比例；分子已扣除 `evictable`（可被淘汰的 radix 前缀缓存），因此它高就意味着真的没有空间放新请求的 KV。口径来自 `PoolStats` / `SchedulerPoolStatsObserver`（`pool_stats_observer.py`），各池算法一致：

$$
\text{num\_used}_{\text{pool}} = \text{capacity} - (\text{available} + \text{evictable}),
\qquad
\text{usage}_{\text{pool}} = \frac{\text{num\_used}_{\text{pool}}}{\text{capacity}}
$$

各符号来源：

| 符号 | 含义 | 来源 |
| --- | --- | --- |
| `available` | 池中空闲、可直接分配的槽位 | plain/SWA：`token_to_kv_pool_allocator.available_size()`；mamba：`req_to_token_pool.mamba_allocator.available_size()` |
| `evictable` | 前缀缓存（radix tree）中可被淘汰复用的槽位 | `tree_cache.evictable_size()`；SWA/mamba 用各自的 `full_evictable_size()` / `swa_evictable_size()` / `mamba_evictable_size()` |
| `capacity` | 池总容量 | plain：`max_total_num_tokens`；SWA：`full_tokens_per_layer` / `swa_tokens_per_layer`；mamba：`req_to_token_pool.mamba_pool.size` |

`token_usage` 取各池的**瓶颈值**（`update_scheduler_stats`）；`/v1/loads` 上报时保留 **4 位小数**（`/metrics` 的 `sglang:token_usage` 保留 2 位）：

$$
\text{token\_usage} = \operatorname{round}\Big(\max\big(\text{full\_token\_usage},\ [\text{swa\_token\_usage}],\ [\text{mamba\_usage}]\big),\ 4\Big)
$$

- 普通（非 SWA / 非 SSM）模型：只有 `full` 一项 → `token_usage = full_token_usage`；
- hybrid-SWA 模型：并入 `swa` 参与取 max；
- hybrid-SSM（mamba）模型：再并入 `mamba`（SWA 与 SSM 可同时存在）。

> ⚠️ `token_usage` 命名有误导性：它实际是**跨池最大值**，并不单指 KV（源码中有 FIXME，改名需走 API 弃用流程）。
>
> 聚合提示：多 TP/DP 会产生多条时间序列，水位类指标必须用 `max` 或 `avg by (dp_rank)`，**不能 `sum`**。

#### 附：`evictable` token 的真实含义（PD 分离 + 分块 prefill）

> 背景关系（见 `endpoints_metrics_and_loads_zh.md` §B.1）：

$$
\text{num\_used\_tokens} = \text{capacity} - (\text{available} + \text{evictable})
$$

> 其中 `evictable` = radix 前缀缓存树中 `lock_ref == 0` 的 token 数（未被任何请求/会话锁定、可随时淘汰的部分）。

**单个 seq 在 P 节点上的状态流转**：

| 步骤 | 动作 | 该 seq KV 的状态 |
| --- | --- | --- |
| 1 | 到达 P，`match_prefix` 命中已有前缀 → `inc_lock_ref` | protected |
| 2 | 每个 chunk forward 完 → `cache_unfinished_req(chunked=True)`：insert 本 chunk + `dec_lock_ref(旧末端)` → `inc_lock_ref(新末端)`（锁前移） | **protected** |
| 3 | 最后一个 chunk（prefill 结束）→ `cache_unfinished_req` 写完整前缀 → 入 inflight 队列 → `send_kv_chunk(last_chunk=True)` | protected |
| 4 | **`KVPoll.Success`**（KV transfer 完成）→ `release_kv_cache` → `cache_finished_req`：insert 补齐 page 对齐尾段 + `dec_lock_ref` | **evictable**（唯一转化点） |
| 5 | 显存紧张 → LRU 从 `evictable_leaves` 淘汰；或被新请求命中 → 重新 `inc_lock_ref` | 释放 / 变回 protected |

**核心结论（PD + chunked prefill）**：

$$
\text{evictable token} \approx \text{“已传完 KV 走人、但前缀仍留在 P 上的缓存”token 数}
$$

- 一个 seq 在整个活跃期（进入 → transfer 成功 → 解锁）写进树的 KV **全是 protected**，一个都不算 evictable；evictable 只在第 4 步解锁那一刻才产生。
- 它是**可回收量的上界**：`evictable_size_` 计所有 `lock_ref == 0` 的在册节点（含内部节点），真正能立即淘汰的只有 `evictable_leaves`（无在册子节点）。
- 粒度 page 对齐；不足一页的尾部不入树，等后续 chunk/finish 补齐才可能计入。
- 它**不代表正在服务的 KV**，也不含任何 prefill / 传输中的请求，反映的是「缓存水位 / 可释放余量」，而非实时负载。

### 2.3 可选分区字段

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

### 2.4 完整 JSON 返回样例

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

### 2.5 Prometheus 格式（`format=prometheus`）

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

### 2.6 边界与错误行为

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

## 3. 附：已废弃的 `/get_load`

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

## 4. 选型建议

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

### 4.1 网关接入的已知坑

`sgl-model-gateway` 的 `parse_load_response` 解析的是：

```text
# sgl-model-gateway/src/core/worker_manager.rs
let load_url = format!("{}/v1/loads?include=core", url);
json["aggregate"]["total_tokens"]   // 解析为 isize
// 失败 / 字段缺失 → 返回 -1
```

而引擎原生返回的是 `{"loads": [{"num_total_tokens": N, ...}]}`（**无 `aggregate` 层，字段名也不同**）。若直连引擎且中间无格式转换层，网关会拿到 `-1`，导致 Power-of-Two 策略**静默退化为按本地请求计数比较**。接入前务必用 `curl` 验证一次实际响应结构。

---

## 5. 速查表

```text
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

`/metrics` 的速查表见 `docs/metrics/endpoints_metrics_and_loads_zh.md`。

跨节点比较负载时使用的统一标量：

$$
\text{num\_total\_tokens} = \text{num\_used\_tokens} + \sum_{\text{req} \in \text{等待队列}} \text{req.seqlen}
$$
