<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
SPDX-License-Identifier: Apache-2.0
-->

# sgl-router（实验性）监控

用于实验性 router 的 Prometheus 指标的 Grafana 仪表盘。这些指标通过 router 服务端口
（默认 `30000`）上的 `/metrics` 端点暴露（text/plain，版本 0.0.4）。

## 文件

- `grafana-dashboard.json` —— 可导入的 Grafana 仪表盘，名为 **SGLang Router
  (experimental)**（uid 为 `sgl-router-experimental`）。

## 覆盖的指标

该仪表盘绘制了 router 发出的每一类指标：

| 指标 | 类型 | 含义 |
|---|---|---|
| `sgl_router_requests_total` | Counter | 按 `worker_url`、`model_id`、`mode`、`outcome` 统计的分发次数 |
| `sgl_router_request_duration_seconds` | Histogram | 按 `model_id` 统计的端到端请求延迟 |
| `sgl_router_ttft_seconds` | Histogram | 按 `model_id` 统计的首 token 时间（流式） |
| `sgl_router_responses_total` | Counter | 客户端可见的 HTTP `status_code` |
| `sgl_router_overlap_blocks` | Histogram | 按 `model_id` 统计的 cache-aware-zmq 重叠块数 |
| `sgl_router_active_load` | Gauge | 单 worker 的 prefill-token / decode-block 负载 |
| `sgl_router_workers` | Gauge | 按 `mode` 统计的已注册 worker 数量 |
| `sgl_router_worker_health` | Gauge | 单 worker 健康状态（1=熔断器放行，0=熔断打开） |
| `sgl_router_worker_cb_state` | Gauge | 单 worker 熔断器状态（0=closed，1=open，2=half_open） |
| `sgl_router_worker_inflight_requests` | Gauge | 每个 worker 正在处理的请求数 |
| `sgl_router_stale_requests_total` | Counter | 过期请求的取消次数 |
| `sgl_router_decode_affinity_total` | Counter | PD decode 亲和性结果 |
| `sgl_router_sticky_total` | Counter | 粘性会话（sticky-session）选择结果 |

`sgl_router_workers` / `sgl_router_worker_*` 这些 gauge 在每次抓取时都会从存活的 worker
注册表中采样，因此被移除的 worker 会立即停止发出对应的时间序列，而不会遗留一个陈旧的值。

## 仪表盘面板一览

仪表盘共约 20 个面板，按 6 个分组（row）组织。以下列出每个面板具体展示的数值：

### Overview（概览）

| 面板 | 单位 | 具体展示的数值 |
|---|---|---|
| Request rate | reqps | 全局请求速率 `sum(rate(sgl_router_requests_total[…]))`（请求/秒） |
| Error ratio | percent | 错误占比 `100 * error / 总请求`（0–100，分母做了 `clamp_min` 兜底避免除零） |
| P99 latency | s | 请求端到端延迟的 P99 分位（秒） |
| Healthy workers | short | 健康 worker 总数 `sum(sgl_router_worker_health)`（个） |

### Traffic & Errors（流量与错误）

| 面板 | 单位 | 具体展示的数值 |
|---|---|---|
| Request rate by outcome | reqps | 按 `outcome`（如 success / error）分组的请求速率 |
| Request rate by mode | reqps | 按 `mode`（如 regular / pd）分组的请求速率 |
| Responses by HTTP status | reqps | 按 `status_code`（200 / 4xx / 5xx 等）分组的响应速率 |
| Request rate by worker | reqps | 按 `worker_url` 分组的请求速率（各 worker 的分发量） |

### Latency（延迟）

| 面板 | 单位 | 具体展示的数值 |
|---|---|---|
| Request latency quantiles | s | 端到端延迟的 p50 / p90 / p99 三条分位曲线（秒） |
| TTFT quantiles (streaming) | s | 流式首 token 时间的 p50 / p90 / p99 三条分位曲线（秒） |
| Request latency distribution | s | 按直方图桶 `le` 展开的延迟分布（各桶速率） |

### Workers & Health（Worker 与健康）

| 面板 | 单位 | 具体展示的数值 |
|---|---|---|
| Workers by mode | short | 按 `mode` 分组的已注册 worker 数量（个） |
| Worker health | short | 每个 worker 的健康值（1=熔断器放行，0=熔断打开） |
| Circuit breaker state | short | 每个 worker 的熔断器状态（0=closed，1=open，2=half_open） |
| In-flight requests per worker | short | 每个 worker 正在处理的请求数（个） |

### Cache & Load（缓存与负载）

| 面板 | 单位 | 具体展示的数值 |
|---|---|---|
| Active load — prefill tokens | short | 每个 worker 的 prefill-token 负载（`kind="prefill_tokens"`） |
| Active load — decode blocks | short | 每个 worker 的 decode-block 负载（`kind="decode_blocks"`） |
| Overlap blocks quantiles | short | cache-aware-zmq 重叠块数的 p50 / p99 分位（块） |

### Routing Policy（路由策略）

| 面板 | 单位 | 具体展示的数值 |
|---|---|---|
| Sticky-session outcomes | reqps | 按 `outcome` 分组的粘性会话选择结果速率 |
| Decode-affinity outcomes | reqps | 按 `outcome` 分组的 PD decode 亲和性结果速率 |
| Stale-request cancellations | reqps | 过期请求取消速率（`expired`，请求/秒） |

> 所有基于 `rate(...)` 的面板使用 `$__rate_interval` 作为窗口，并受顶部 `model_id` /
> `worker_url` 模板变量约束（`sgl_router_responses_total` 等无这两个标签的指标除外）。

## Prometheus 抓取配置

让 Prometheus 指向 router 的 `/metrics` 端点：

```yaml
scrape_configs:
  - job_name: sgl-router
    metrics_path: /metrics
    static_configs:
      - targets:
          - '127.0.0.1:30000'   # router 的 host:port
```

## 导入 Grafana

1. **Dashboards → New → Import**。
2. 上传 `grafana-dashboard.json`（或粘贴其内容）。
3. 出现提示时，为 `Datasource` 变量选择你的 Prometheus 数据源。该仪表盘使用模板化的
   数据源，因此无需修改 JSON 即可导入到任意 Grafana 中。

顶部栏暴露了 `model_id` 和 `worker_url` 两个模板变量（都默认为 *All*），用于限定各面板的范围。

## 重新生成

该 JSON 是以编程方式生成的，以保持约 20 个面板的一致性。如果指标集发生变化，请更新生成器
并覆盖 JSON，而不要手动编辑 —— 手动编辑会偏离面板的既定约定。
