# sgl-router

Slim, KV-aware, OpenAI-compatible router for SGLang workers.

> 中译：面向 SGLang worker 的**轻量、KV 感知、OpenAI 兼容**路由器。

Serves a single model and routes across its workers. Exposes
`/v1/tokenize`, `/v1/detokenize`, `/v1/models`, `/v1/chat/completions`
(buffered and SSE), plus `/healthz` / `/readyz` and `/metrics`. Worker
pools come from either a static URL list or Kubernetes EndpointSlice
discovery.

> 中译：它只服务**单个模型**，并在该模型的多个 worker 之间做路由。对外暴露以下接口：
> `/v1/tokenize`（分词）、`/v1/detokenize`（反分词）、`/v1/models`（模型列表）、
> `/v1/chat/completions`（对话补全，支持缓冲返回与 SSE 流式两种），以及
> `/healthz`（存活探针）/ `/readyz`（就绪探针）和 `/metrics`（Prometheus 指标）。
> worker 池的来源有两种：**静态 URL 列表**，或 **Kubernetes EndpointSlice 服务发现**。

## Building（编译）

```bash
cd experimental/sgl-router
cargo build --release
```

## Running（运行）

The router is configured entirely through CLI flags (run
`sgl-router --help` for the full list). It serves exactly one model, so
`--model-id` is required, along with exactly one discovery backend.
`--tokenizer-path` is optional: give it a local `tokenizer.json` path or a
HuggingFace repo id, and when omitted the router downloads the tokenizer
for `--model-id` from HuggingFace (honoring `HF_TOKEN` / `HF_HOME`).

> 中译：路由器**完全通过命令行参数（CLI flags）配置**（运行 `sgl-router --help` 查看完整列表）。
> 由于它只服务一个模型，因此 `--model-id` 是必填项，并且必须**恰好指定一个服务发现后端**。
> `--tokenizer-path` 是可选的：可传入本地 `tokenizer.json` 路径或 HuggingFace 仓库 id；
> 若省略，路由器会从 HuggingFace 下载 `--model-id` 对应的分词器（会遵循 `HF_TOKEN` / `HF_HOME` 环境变量）。

Static worker list（静态 worker 列表）:

```bash
sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000
```

Kubernetes EndpointSlice discovery（Kubernetes EndpointSlice 服务发现）:

```bash
sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --service-discovery \
  --service-discovery-namespace prod \
  --selector app=engines-qwen3
```

Omit `--service-discovery-namespace` to watch all namespaces (requires
cluster-wide RBAC). For prefill/decode disaggregation, replace `--selector`
with `--prefill-selector` and `--decode-selector`.

> 中译：省略 `--service-discovery-namespace` 可监听**所有命名空间**（需要集群级 RBAC 权限）。
> 若使用 **prefill/decode 分离（PD 分离）** 部署，则把 `--selector` 替换为
> `--prefill-selector` 与 `--decode-selector`，分别发现预填充与解码两类 worker。

## 架构与集群拓扑

### Router 是软状态（soft-state），而非持久有状态

Router **不持有任何需要持久化的权威状态**，但为了性能会在内存中维护一份**可从 worker 事件流重建**的软状态：

- **KV 事件哈希前缀树（`HashTree`）**：cache-aware 策略用它跟踪「哪个 worker 缓存了哪些前缀」。它是
  **每进程一份**，且策略侧**只读**——由后台 indexer 订阅各 worker 通过 ZMQ 推送的 KV 事件来填充
  （见 `src/policies/cache_aware_zmq.rs` 与 `src/policies/kv_events/`）。
- **worker 注册表 / 健康状态 / 活跃负载**：均为运行时观测值（见 `src/workers/`、`src/health/`）。
- **block_size_oracle**：从 worker 上报的 `page_size` 学习到的块大小。

这些状态都能在 router 重启后通过重新订阅事件恢复，**不落盘、不需要 worker 反向依赖某个特定 router**。
即使前缀树尚未建好或分词器缺失，cache-aware 也会**降级为按负载路由**而非报错：

> The implementation never returns `None` for a non-empty `workers` slice; a misconfigured tree or
> tokenizer degrades to round-robin-with-load tiebreak, not a routing failure.
>
> 中译：只要候选 worker 非空，选择逻辑永不返回 `None`；前缀树或分词器异常时会退化为
> 「带负载 tiebreak 的轮询」，而不会造成路由失败。

因此 Router 本质上是**软状态、shared-nothing 的**，可任意水平扩展、无主从、无一致性协议。

### 集群节点关系

```
                    ┌─────────────┐
   Client ────────▶ │   Router    │ (可多副本, 各自维护软状态, 互不通信)
                    └──────┬──────┘
              路由决策 ┌───┴────────────┐ 订阅 KV 事件(ZMQ)
                     ▼                 ▲
        ┌────────────────────────────────────────┐
        │        SGLang Worker 池（单模型）         │
        │  ┌──────────┐        ┌──────────┐        │
        │  │ Prefill  │──KV──▶ │ Decode   │        │  (PD 分离时)
        │  │ worker   │ RDMA   │ worker   │        │
        │  └──────────┘        └──────────┘        │
        └────────────────────────────────────────┘
                     ▲
              服务发现(静态列表 / K8s EndpointSlice)
```

1. **Router ↔ Worker：单向请求 + 事件反馈**
   - Router → Worker：转发请求（HTTP），主调用方向。
   - Worker → Router：通过 ZMQ 推送 KV 事件（前缀块缓存/驱逐），供 router 构建前缀树。
     Worker **不感知也不依赖具体哪个 router**。

2. **Router ↔ Router：无关系（shared-nothing）**
   - 多个 router 副本之间**不通信、不共享状态**，各自独立订阅同一批 worker 的事件、各自维护前缀树，
     并各自收敛到一致的路由决策（见 `tests/e2e/chat_completions/test_two_router_convergence.py`）。
   - 因此 router 可自由水平扩展、无主从、无一致性协议——这也是它能在 K8s 中随意扩缩容的原因。

3. **Worker ↔ Worker：仅 PD 分离时有关系**
   - 普通部署下 worker 之间无关系。
   - PD 分离时，prefill worker 通过 RDMA/Mooncake 把 KV 直接传给 decode worker，凭 router 注入的
     `bootstrap_room` 配对（见 `tests/proxy/pd_bootstrap_injection.rs`、`tests/proxy/pd_pool_isolation.rs`）。

**一句话**：Router 软状态、副本零耦合、可水平扩展；集群中 Router 单向转发并反向订阅 Worker 的 KV 事件，
Worker 之间仅在 PD 分离时通过 RDMA 传 KV。

## License（许可证）

Apache-2.0.

---

## 学习计划

面向想吃透 `sgl-router`（Rust 实现的高性能路由器）的读者，按「概念 → 配置入口 → 服务器骨架 →
worker 管理 → 路由策略 → 服务发现 → 分词 → 健康与可观测性 → 端到端」的顺序推进。每个阶段给出目标、
建议阅读的源码文件与自测问题，可按需跳过已熟悉的部分。

### 前置知识
- **路由器定位**：router 不做推理，只负责把 OpenAI 兼容请求分发到后端 SGLang worker；相比 Python 版
  `mini_lb`，Rust 版追求高吞吐、低延迟与丰富的负载均衡策略。
- **KV 感知路由（KV-aware）**：为什么「把相同前缀的请求发往已缓存该前缀的 worker」能提升 radix cache
  命中率，从而降低 TTFT。
- **PD 分离**：prefill（计算密集、一次性）与 decode（访存密集、自回归）拆到不同 worker，router 需为每个
  请求选一对 (prefill, decode) 后端并注入 `bootstrap_room` 配对。
- **Rust/async 基础**：`tokio` 异步运行时、`axum`/`hyper` 的 HTTP 服务模型、SSE 流式响应。
- **Kubernetes 基础**：EndpointSlice、Service、Selector、RBAC 与服务发现机制。

### 阶段一：整体架构与请求生命周期（约 0.5 天）
- **目标**：建立「一个请求从进入 router 到分发给 worker、再流式返回」的全局心智模型。
- **阅读顺序**：本 README → `src/main.rs`（进程入口）→ `src/lib.rs`（模块导出）→ `src/server/mod.rs`
  与 `src/server/app.rs`（HTTP 服务装配）。
- **自测**：router 启动后经历哪几步初始化？一个 `/v1/chat/completions` 请求在内部依次经过哪些模块？

### 阶段二：配置与 CLI 入口（约 0.5 天）
- **目标**：掌握所有可配置项与「单模型 + 单发现后端」的约束由何处校验。
- **阅读顺序**：`src/config/cli.rs`（CLI flag 定义）→ `src/config/types.rs`（配置数据结构）→
  `src/config/mod.rs`（校验与组装）。
- **自测**：`--model-id`、`--tokenizer-path`、`--worker-urls` 与服务发现相关 flag 如何互斥/互补？
  省略 `--tokenizer-path` 时下载分词器的逻辑在哪里触发？

### 阶段三：服务器骨架与路由表（约 0.5 天）
- **目标**：理解各 HTTP 端点如何注册与分发。
- **阅读顺序**：`src/server/routes/mod.rs`（路由注册）→ 逐个看 `chat.rs`、`tokenize.rs`、`models.rs`、
  `health.rs`、`metrics.rs`、`cache.rs` → `src/server/app_context.rs`（共享上下文）、
  `src/server/error.rs`、`src/server/header_utils.rs`。
- **自测**：`/v1/chat/completions` 的 buffered 与 SSE 两种返回分别由哪段代码处理？请求头是如何透传给 worker 的？

### 阶段四：worker 管理（约 1 天）
- **目标**：掌握 worker 的注册、健康状态维护与生命周期。
- **阅读顺序**：`src/workers/worker.rs`（单 worker 抽象）→ `src/workers/registry.rs`（注册表）→
  `src/workers/manager.rs`（增删与状态机）→ `src/workers/introspect.rs`（探测 worker 能力/DP size 等）。
- **自测**：worker 从「被发现」到「可路由」经历哪些状态？故障 worker 如何被摘除与恢复？

### 阶段五：路由策略（约 1.5 天，重点）
- **目标**：吃透各负载均衡策略的取舍，尤其是 KV 感知（cache-aware）与 KV 事件订阅。
- **阅读顺序**：
  1. `src/policies/mod.rs` + `src/policies/factory.rs` + `src/policies/registry.rs`（策略抽象与工厂）。
  2. 简单策略：`random.rs`、`round_robin.rs`、`power_of_two.rs`、`load_based.rs`、`active_load.rs`、`sticky.rs`。
  3. KV 感知主线：`src/policies/cache_aware_zmq.rs`，再深入 `src/policies/kv_events/`
     （`tree.rs` 近似前缀树、`hash.rs` 哈希、`index.rs` 索引、`subscriber.rs`/`discovery.rs` 事件订阅、
     `wire.rs` 线格式、`block_size_oracle.rs` 块大小推断）。
- **自测**：cache-aware 如何用前缀树跟踪各 worker 的缓存？命中率不足或负载失衡时如何回退？
  power_of_two（P2C）为什么能以低开销逼近最小负载？

### 阶段六：服务发现（约 1 天）
- **目标**：理解静态列表与 K8s EndpointSlice 两种发现方式，及 PD 分离下的 prefill/decode 分组。
- **阅读顺序**：`src/discovery/mod.rs` + `src/discovery/types.rs` → `src/discovery/static_urls.rs`
  → `src/discovery/k8s.rs`（EndpointSlice watch、命名空间与 selector）。
- **自测**：`--prefill-selector` / `--decode-selector` 如何把 worker 分成两个池？watch 所有命名空间需要什么 RBAC？

### 阶段七：分词与代理层（约 1 天）
- **目标**：掌握 router 侧本地分词、chat 模板渲染与到 worker 的代理转发。
- **阅读顺序**：`src/tokenizer/mod.rs` + `adapter.rs` + `chat_template.rs` + `dsv4.rs`（DeepSeek 特化）→
  `src/proxy/mod.rs`（转发）+ `src/proxy/sse.rs`（SSE 流式透传）。
- **自测**：为什么 router 需要自带分词器？`/v1/tokenize` 与转发前的分词分别用于什么？SSE chunk 如何被逐块透传？

### 阶段八：健康检查与可观测性（约 0.5 天）
- **目标**：理解熔断、健康探针与指标暴露。
- **阅读顺序**：`src/health/mod.rs` + `src/health/circuit_breaker.rs` → `src/server/metrics.rs`
  → `monitoring/README.md` 与 `monitoring/grafana-dashboard.json`。
- **自测**：熔断器（circuit breaker）在什么条件下打开/半开/关闭？`/healthz` 与 `/readyz` 的语义差异是什么？

### 阶段九：测试与基准（按需选读）
- **目标**：学会本地验证与性能评估。
- **阅读顺序**：`tests/proxy/`（转发、故障切换、PD 池隔离、bootstrap 注入等 Rust 集成测试）→
  `tests/component/`（组件级测试）→ `tests/e2e/`（含 K8s 集成、chat/tokenize 冒烟）→
  `benches/tree_lookup.rs`、`benches/policy_select.rs` 与 `BENCHMARKS.md`。
- **自测**：`pd_pool_isolation.rs` 与 `pd_bootstrap_injection.rs` 分别验证 PD 分离的哪个关键行为？

### 学习建议
- **带着请求流读代码**：始终追问「这个请求现在在哪个模块、下一步交给谁、依据什么策略选后端」。
- **先跑静态列表，再上 K8s**：用 `--worker-urls` 跑通最简链路，再研究 EndpointSlice 服务发现。
- **策略先看简单再看 KV 感知**：先理解 random/round_robin 的骨架，再攻 cache-aware + kv_events。
- **对照 Python mini_lb**：与 `python/sglang/srt/.../mini_lb.py` 对比，理解生产版在策略上的增强。
