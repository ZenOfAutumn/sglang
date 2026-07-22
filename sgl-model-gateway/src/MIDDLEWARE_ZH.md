# Middleware 中间件层原理与作用（`src/middleware.rs`）

> 本文档基于 `src/middleware.rs` 源码，说明网关 HTTP 中间件层的职责、实现原理与执行顺序。

---

## 0. 概览

`middleware.rs` 是模型网关的 **HTTP 请求横切层（cross-cutting layer）**，位于 axum/tower 的 `Service` 管线中，在请求真正进入业务路由（router → worker）之前和响应返回客户端的过程中，统一处理与业务无关的通用能力：

| 能力 | 关键类型 / 函数 | 形态 |
|---|---|---|
| API Key 鉴权 | `auth_middleware` | 函数式中间件 |
| 请求 ID 注入与回填 | `RequestIdLayer` / `RequestIdMiddleware` | Tower Layer/Service |
| 访问日志与链路 Span | `create_logging_layer`（`RequestSpan`/`RequestLogger`/`ResponseLogger`） | TraceLayer |
| HTTP 指标采集 & 在途请求追踪 | `HttpMetricsLayer` / `HttpMetricsMiddleware` | Tower Layer/Service |
| 并发限流 + 排队 | `concurrency_limit_middleware` + `TokenBucket` + `QueueProcessor` | 函数式中间件 |
| 令牌随流释放 | `TokenGuardBody` | Body 包装器 |
| WASM 插件扩展（请求/响应改写） | `wasm_middleware` | 函数式中间件 |
| 指标路径归一化（防高基数） | `normalize_path_for_metrics` / `is_dynamic_id` | 辅助函数 |

它们组合起来实现了「**鉴权 → 限流 → 可观测 → 可扩展**」的统一入口治理。

---

## 1. 两种中间件形态

代码里存在两种 tower 中间件写法，理解它们的区别有助于看懂执行顺序：

- **函数式中间件**（`async fn(State, Request, Next) -> Response`）：通过 `axum::middleware::from_fn_with_state` 挂载，用 `.route_layer(...)` 绑定到具体路由组。如 `auth_middleware`、`concurrency_limit_middleware`、`wasm_middleware`。
- **Layer/Service 中间件**（实现 `tower::Layer` + `tower::Service`）：通过 `.layer(...)` 全局挂载。如 `RequestIdLayer`、`HttpMetricsLayer`、`create_logging_layer` 返回的 `TraceLayer`。

---

## 2. 执行顺序（关键）

在 `src/server.rs` 中的挂载方式决定了实际执行顺序。tower 的 `.layer()` 是**后挂载的先执行（最外层）**，因此全局层的进入顺序为：

```
（请求进入）
CORS
  → RequestIdLayer          // 生成/透传 request_id，写入 extensions
    → HttpMetricsLayer      // 活跃连接计数 + 在途追踪 + 时延指标
      → create_logging_layer(TraceLayer)  // 建 span、记录 on_request/on_response
        → RequestBodyLimit / DefaultBodyLimit  // 请求体大小限制
          → [route_layer] wasm_middleware        // WASM OnRequest 改写
            → [route_layer] auth_middleware      // API Key 校验
              → [route_layer] concurrency_limit_middleware  // 限流/排队
                → 业务 handler（router → worker）
（响应按相反顺序回溯，附带 x-request-id、指标、日志、WASM OnResponse 改写）
```

> 注意：`route_layer` 同样遵循「后写先执行」。在 `server.rs` 中依次写了 `concurrency_limit → auth → wasm`，所以进入时的顺序是 **wasm → auth → concurrency_limit**。即 WASM 先于鉴权执行，鉴权先于限流。

---

## 3. 各中间件详解

### 3.1 `auth_middleware` — API Key 鉴权

- 仅当配置了 `AuthConfig.api_key` 时才生效；未配置直接放行。
- 校验 `Authorization: Bearer <token>` 头。
- **安全要点**：使用 `subtle::ConstantTimeEq` 做**常量时间比较**，避免通过响应耗时差异推断密钥的**时序攻击（timing attack）**。长度不等时先返回 401（长度检查本身非常量时间，但必要）。

```112:148:src/middleware.rs
pub async fn auth_middleware(
    State(auth_config): State<AuthConfig>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    ...
                if token_bytes.ct_eq(expected_bytes).unwrap_u8() != 1 {
                    return Err(StatusCode::UNAUTHORIZED);
                }
```

### 3.2 `RequestIdLayer` — 请求 ID 注入

- **入站**：按配置的候选头名（`headers`）依次查找已有的请求 ID；找不到则用 `generate_request_id(path)` 生成。
- ID 格式**兼容 OpenAI**：按路径加前缀 —— `chatcmpl-`（chat）、`cmpl-`（completions）、`gnt-`（generate）、`resp-`（responses）、否则 `req-`，后接 24 位随机字母数字。
- 把 `RequestId` 存入 `request.extensions()`，供后续中间件/handler 使用。
- **出站**：把请求 ID 写回响应头 `x-request-id`，实现端到端链路关联。
- 性能细节：随机串用**字节数组 O(1) 索引**生成，避免 `chars().nth()` 的 O(n)。

### 3.3 日志与链路层 — `create_logging_layer`

基于 `tower_http::TraceLayer`，由三个自定义组件构成：

- `RequestSpan`（`MakeSpan`）：为每个请求创建 `http_request` span，预置 `method/uri/version` 与占位字段（`request_id`、`status_code`、`latency`、`error`），`target = smg::otel-trace`，接入 OpenTelemetry。
  - 注意：span 创建时 request_id **尚不可得**（TraceLayer 早于 RequestIdLayer 处理该字段），故先置 `Empty`，稍后回填。
- `RequestLogger`（`OnRequest`）：进入时从 extensions 补记 `request_id`，上报 `Metrics::record_http_request(method, path)`，打印 "started processing request"。
- `ResponseLogger`（`OnResponse`）：记录 `status_code`、`latency`（微秒整数，避免格式化分配），上报 `record_http_response`，并按 5xx/4xx/2xx 分级打 `error!/warn!/info!` 日志。`extract_error_code_from_response` 提取业务错误码。

### 3.4 `HttpMetricsLayer` — 指标与在途追踪

- 维护全局 `ACTIVE_HTTP_CONNECTIONS`（原子计数）反映当前正在处理的连接数。
- **健壮性设计**：计数**在 async 块内**自增，保证 future 被提前 drop 时不会泄漏计数；用 `InFlightRequestTracker::track()` 返回的 guard 追踪在途请求，`inner.call` 结果先捕获再 `drop(guard)` 并自减，确保**成功/失败都会正确减少计数**。
- 请求结束后上报 `record_http_duration(method, path, duration)`。

### 3.5 并发限流 — `concurrency_limit_middleware` + 令牌桶 + 队列

这是最复杂的一环，实现「**令牌桶限流 + 有界排队 + 超时**」：

1. **Mesh 全局限流优先**：若启用 mesh，先查 `check_global_rate_limit()`，超限直接返回 429（带 `current_count/limit` JSON）。
2. **本地令牌桶**：`rate_limiter` 未配置则直接放行。
3. **快速路径**：`try_acquire(1.0)` 立即拿到令牌 → 放行，指标记 `allowed`。
4. **排队路径**：拿不到令牌且启用队列（`concurrency_queue_tx`）时，构造 `QueuedRequest`（含入队时间 + oneshot 回执通道）`try_send` 入队：
   - 队满 → 429（`rejected`）。
   - 入队成功 → 等 `QueueProcessor` 通过 oneshot 回执授予令牌或返回超时（`REQUEST_TIMEOUT`）。
   - 针对 `/v1/embeddings` 单独维护 `EMBEDDINGS_QUEUE_SIZE` 计数。
5. **无队列**：拿不到令牌直接 429。

`QueueProcessor::run` 是**单任务队列消费者**：
- 出队时先判断是否**已在队列中超时**（`queued_at.elapsed() >= queue_timeout`），是则回 `REQUEST_TIMEOUT`。
- 否则以**剩余超时**尝试获取令牌；能立即拿到就直接回执，否则**才** `tokio::spawn` 一个等待任务（`acquire_timeout`），减少无谓的任务开销。

```418:461:src/middleware.rs
    pub async fn run(mut self) {
        ...
        while let Some(queued) = self.queue_rx.recv().await {
            let elapsed = queued.queued_at.elapsed();
            if elapsed >= self.queue_timeout {
                let _ = queued.permit_tx.send(Err(StatusCode::REQUEST_TIMEOUT));
                continue;
            }
            ...
```

#### 3.5.1 `check_global_rate_limit` — Mesh 分布式全局限流原理

`check_global_rate_limit` 不在本仓库，而是 `smg-mesh` crate（`MeshSyncManager`）提供的方法。它解决的问题是：**多个网关实例组成 mesh 集群时，如何对"全集群每秒总请求数"做统一限流**（本地令牌桶只能限单实例）。

其核心是一个 **基于 CRDT PNCounter + 一致性哈希分片 + 时间窗口重置** 的分布式计数器。返回 `(is_exceeded, current_count, limit)`：

```288:306:smg-mesh-1.0.0/src/sync.rs
    pub fn check_global_rate_limit(&self) -> (bool, i64, u64) {
        let config = self.get_global_rate_limit_config().unwrap_or_default();
        if config.limit_per_second == 0 {
            return (false, 0, 0);   // 未配置 → 关闭全局限流
        }
        // 若本节点是该 key 的 owner，则 +1
        self.sync_rate_limit_inc(GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string(), 1);
        // 读取（CRDT 合并后的）全局计数
        let current_count = self
            .get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY)
            .unwrap_or(0);
        let is_exceeded = current_count > config.limit_per_second as i64;
        (is_exceeded, current_count, config.limit_per_second)
    }
```

**关键机制拆解**：

1. **配置来源**：`limit_per_second` 存放在 mesh 的 `AppStore`（key = `GLOBAL_RATE_LIMIT_KEY`），是集群共享配置；为 0 表示关闭全局限流，直接放行。

2. **计数（写）—— 一致性哈希决定谁来记账**：
   计数器 key 为 `GLOBAL_RATE_LIMIT_COUNTER_KEY`，落在 `RateLimitStore` 里。`sync_rate_limit_inc` / `inc` **只有当本节点是该 key 的 owner 时才 +1**：

   ```403:413:smg-mesh-1.0.0/src/stores.rs
       pub fn inc(&self, key: String, actor: String, delta: i64) {
           if !self.is_owner(&key) {
               return;   // 非 owner 不记账
           }
           ...
           counter.inc(actor, delta);
       }
   ```

   owner 由**一致性哈希环**（`ConsistentHashRing::is_owner`）判定——把计数 key 哈希到环上某个节点。这样避免每个节点各记各的、重复放大计数。

3. **计数类型是 CRDT PNCounter（无冲突可合并）**：
   底层用 `SyncPNCounter`（`crdts` 库的 PN-Counter，正负分离计数）。每个 owner 用**自己的节点名作为 actor** 累加，节点间通过 gossip 交换并 `merge_counter` 合并。PNCounter 的数学性质保证**多节点并发累加、乱序合并后仍收敛到一致的总和**，无需分布式锁。

4. **读取—— 聚合全局值**：`get_rate_limit_value` → `RateLimitStore::value` 读取的是**合并后的 CRDT 值**，即全集群累计请求数。与 `limit_per_second` 比较判断是否超限。

5. **时间窗口重置**：PNCounter 本身只增不减、无法直接"清零"。由独立的 `RateLimitWindow` 后台任务按 `window_seconds` 周期性调用 `reset_global_rate_limit_counter`，通过**反向递减当前值**（`inc(-current_count)`）近似实现"每窗口重置"，从而把计数器语义变成"最近一个时间窗口内的请求数"：

   ```29:42:smg-mesh-1.0.0/src/rate_limit_window.rs
       pub async fn start_reset_task(self) {
           let mut interval_timer = interval(Duration::from_secs(self.window_seconds));
           loop {
               interval_timer.tick().await;
               self.sync_manager.reset_global_rate_limit_counter();
           }
       }
   ```

**小结**：`check_global_rate_limit` = 「一致性哈希选 owner 记账 → CRDT PNCounter 跨节点无锁合并出全局计数 → 与共享配置阈值比较 → 后台窗口任务周期性递减重置」。它在中间件里**先于本地令牌桶执行**，形成"集群级配额"与"实例级并发"的两级限流。

> 注意：这是**最终一致**的近似限流——gossip 同步和窗口重置都有延迟，短时可能略微过冲，换取的是无锁、高可用、可水平扩展。

### 3.6 `TokenGuardBody` — 令牌随流释放

限流的关键正确性保证：令牌**不能在 handler 返回时就归还**，否则**流式响应**尚未发送完毕就会被放行新请求，导致并发数虚高。

`TokenGuardBody` 包装响应 `Body`，在**整个响应流被消费完或被 drop 时**（`Drop` impl）才通过 `return_tokens_sync` 归还令牌：

```71:82:src/middleware.rs
impl Drop for TokenGuardBody {
    fn drop(&mut self) {
        if let Some(bucket) = self.token_bucket.take() {
            ...
            bucket.return_tokens_sync(self.tokens);
        }
    }
}
```

它实现 `http_body::Body`，`poll_frame` 透传给内层 body（`get_mut` 安全，因为 Body 是 Unpin）。这样把「并发额度占用时长」精确对齐到「客户端真正收完响应」的时刻——**尤其对 SSE 流式接口至关重要**。

### 3.7 `wasm_middleware` — WASM 插件扩展

提供**可编程的请求/响应改写**能力（沙箱化插件）：

- 未启用 WASM 或无 manager → 直接放行。
- **OnRequest 阶段**：按 attach point 取出所有 OnRequest 模块；若有，则**一次性读取请求体**（受 `max_body_size` 限制），逐个执行模块，模块返回三种 `Action`：
  - `Continue`：继续；
  - `Reject(status)`：立即以该状态码拒绝；
  - `Modify`：改写 headers / 替换 body（改写会累积传给下一个模块）。
  处理后用改写结果**重建请求**再交给下游。
- **OnResponse 阶段**：对响应做同样的模块链处理（读响应体 → 逐模块执行 → Continue/Reject/Modify），最后重建响应。
- **容错**：读体失败、取模块失败、单模块执行返回 None 时，均**降级为放行/跳过**，保证插件异常不影响主链路。

### 3.8 `normalize_path_for_metrics` — 指标路径归一化

为避免 URL 中的动态 ID（如 `resp_xxx`、UUID、数字 ID）导致 Prometheus **标签高基数爆炸**，把路径中第 3 段之后、被 `is_dynamic_id` 判定为动态的段替换为 `{id}`。

`is_dynamic_id` 判定规则：
- 带下划线且长度 > 10 的前缀 ID（`resp_...`、`chatcmpl_...`）；
- 长度 ≥ 32 的十六进制/带 `-` 串（UUID）；
- 纯数字。

实现上**仅在需要归一化时才分配 String**，单遍扫描，兼顾正确性与性能。相关单测见文件末尾 `tests` 模块。

---

## 4. 设计要点小结

- **安全**：常量时间比较防时序攻击；WASM 插件沙箱化且失败降级。
- **正确性**：限流令牌随「响应流结束」释放（`TokenGuardBody`），在途计数在 async 块内增减、异常路径也保证归还。
- **可观测**：请求 ID 端到端贯通（extensions → `x-request-id` → span），分层指标（HTTP 层连接/时延/限流），路径归一化防高基数。
- **性能**：字节数组 O(1) 生成 ID、按需分配的路径归一化、队列消费者按需 spawn。
- **可扩展**：WASM OnRequest/OnResponse 双阶段钩子，支持无侵入地改写请求与响应。

---

## 5. 相关文件

- `src/server.rs`：中间件挂载与执行顺序。
- `src/core/token_bucket.rs`：令牌桶实现（`TokenBucket`）。
- `src/observability/inflight_tracker.rs`：在途请求追踪。
- `src/observability/metrics.rs`：指标上报。
- `src/wasm/`：WASM 模块管理与执行。

