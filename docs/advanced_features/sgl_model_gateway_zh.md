# SGLang Model Gateway

SGLang Model Gateway 是面向大规模 LLM 部署的高性能模型路由 gateway。它统一管理 worker 生命周期,在异构协议(HTTP、gRPC、OpenAI 兼容)之间均衡流量,并对历史存储、MCP 工具调用以及涉及隐私的工作流提供企业级的管控能力。该 gateway 针对 SGLang 服务运行时进行了深度优化,但也可以路由到任意 OpenAI 兼容的后端。

---

## 目录

1. [概述](#overview)
2. [架构](#architecture)
   - [控制平面](#control-plane)
   - [数据平面](#data-plane)
   - [存储与隐私](#storage-and-privacy)
3. [安装](#installation)
4. [快速开始](#quick-start)
5. [部署模式](#deployment-modes)
   - [联合启动 Router 与 Worker](#co-launch-router-and-workers)
   - [分离启动(HTTP)](#separate-launch-http)
   - [gRPC 启动](#grpc-launch)
   - [Prefill-Decode 分离](#prefill-decode-disaggregation)
   - [OpenAI 后端代理](#openai-backend-proxy)
   - [多模型推理 Gateway](#multi-model-inference-gateway)
6. [API 参考](#api-reference)
   - [推理端点](#inference-endpoints)
   - [Tokenization 端点](#tokenization-endpoints)
   - [Parser 端点](#parser-endpoints)
   - [分类 API](#classification-api)
   - [会话与响应 API](#conversation-and-response-apis)
   - [Worker 管理 API](#worker-management-apis)
   - [管理与健康端点](#admin-and-health-endpoints)
7. [负载均衡策略](#load-balancing-policies)
8. [可靠性与流量控制](#reliability-and-flow-control)
   - [重试](#retries)
   - [熔断器](#circuit-breaker)
   - [限流与排队](#rate-limiting-and-queuing)
   - [健康检查](#health-checks)
9. [Reasoning Parser 集成](#reasoning-parser-integration)
10. [工具调用解析](#tool-call-parsing)
11. [Tokenizer 管理](#tokenizer-management)
12. [MCP 集成](#mcp-integration)
13. [服务发现(Kubernetes)](#service-discovery-kubernetes)
14. [历史与数据连接器](#history-and-data-connectors)
15. [WASM 中间件](#wasm-middleware)
16. [语言绑定](#language-bindings)
17. [安全与认证](#security-and-authentication)
    - [Gateway 服务器的 TLS(HTTPS)](#tls-https-for-gateway-server)
    - [Worker 通信的 mTLS](#mtls-for-worker-communication)
18. [可观测性](#observability)
    - [Prometheus 指标](#prometheus-metrics)
    - [OpenTelemetry 追踪](#opentelemetry-tracing)
    - [日志](#logging)
19. [生产环境建议](#production-recommendations)
    - [安全最佳实践](#security-best-practices)
    - [高可用](#high-availability)
    - [性能](#performance)
    - [Kubernetes 部署](#kubernetes-deployment)
    - [使用 PromQL 监控](#monitoring-with-promql)
20. [配置参考](#configuration-reference)
21. [故障排查](#troubleshooting)

---

## Overview

- **统一的控制平面**,用于在异构模型集群中注册、监控并编排常规(regular)、prefill 和 decode worker。
- **多协议数据平面**,在 HTTP、PD(prefill/decode)、gRPC 以及 OpenAI 兼容后端之间路由流量,并共享统一的可靠性原语。
- **业界首创的 gRPC 流水线**,具备原生 Rust tokenization、reasoning parser 和工具调用执行能力,实现高 throughput、OpenAI 兼容的服务;同时支持单阶段(single-stage)和 PD 拓扑。
- **Inference Gateway 模式(`--enable-igw`)** 可动态实例化多套 router 栈(HTTP regular/PD、gRPC),并为多租户部署应用按模型(per-model)的策略。
- **会话与响应连接器** 将聊天历史集中保存在 router 内部,使得同一上下文可以在多个模型与 MCP 循环之间复用,而不会将数据泄露给上游厂商(memory、none、Oracle ATP、PostgreSQL)。
- **企业级隐私**:agentic 多轮 `/v1/responses`、原生 MCP 客户端(STDIO/HTTP/SSE/Streamable)以及历史存储全部在 router 边界内运行。
- **可靠性内核**:带抖动(jitter)的重试、按 worker 作用域的熔断器、带排队的 token-bucket 限流、后台健康检查,以及缓存感知(cache-aware)的负载监控。
- **全面的可观测性**:40+ 个 Prometheus 指标、OpenTelemetry 分布式追踪、结构化日志以及请求 ID 透传。

---

## Architecture

### Control Plane

- **Worker Manager** 发现各 worker 的能力(`/server_info`、`/get_model_info`),跟踪负载,并在共享注册表中注册/移除 worker。
- **Job Queue** 对添加/移除请求进行串行化,并暴露状态(`/workers/{worker_id}`),以便客户端跟踪接入进度。
- **Load Monitor** 通过实时 worker 负载统计为 cache-aware 和 power-of-two 策略提供数据。
- **Health Checker** 持续探测 worker,并更新就绪状态、熔断器状态以及 router 指标。
- **Tokenizer Registry** 管理动态注册的 tokenizer,支持从 HuggingFace 或本地路径异步加载。

### Data Plane

- **HTTP router**(regular 与 PD)实现了 `/generate`、`/v1/chat/completions`、`/v1/completions`、`/v1/responses`、`/v1/embeddings`、`/v1/rerank`、`/v1/classify`、`/v1/tokenize`、`/v1/detokenize` 及相关管理端点。
- **gRPC router** 将 tokenize 后的请求直接流式发送到 SRT gRPC worker,完全运行在 Rust 中——tokenizer、reasoning parser 和 tool parser 均驻留在进程内。支持单阶段和 PD 路由,包括 embeddings 和分类。
- **OpenAI router** 将 OpenAI 兼容端点代理到外部厂商(OpenAI、xAI 等),同时将聊天历史与多轮编排保留在本地。

### Storage and Privacy

- 会话与响应历史存储在 router 层(memory、none、Oracle ATP 或 PostgreSQL)。同一份历史可以为多个模型或 MCP 循环提供支撑,而无需将数据发送给上游厂商。
- `/v1/responses` agentic 流程、MCP 会话以及会话 API 共享同一存储层,从而为受监管的工作负载实现合规。

---

## Installation

### Docker

预构建的 Docker 镜像已发布在 Docker Hub,支持多架构(x86_64 和 ARM64):

```bash
docker pull lmsysorg/sgl-model-gateway:latest
```

### Prerequisites

- **Rust 与 Cargo**
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  rustc --version
  cargo --version
  ```
- **Python**,并安装可用的 `pip` 与 virtualenv 工具。

### Rust Binary

```bash
cd sgl-model-gateway
cargo build --release
```

### Python Package

```bash
pip install maturin

# Fast development mode
cd sgl-model-gateway/bindings/python
maturin develop

# Production build
maturin build --release --out dist --features vendored-openssl
pip install --force-reinstall dist/*.whl
```

---

## Quick Start

### Regular HTTP Routing

```bash
# Rust binary
./target/release/sgl-model-gateway \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware

# Python launcher
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware
```

### gRPC Routing

```bash
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser json \
  --host 0.0.0.0 --port 8080
```

---

## Deployment Modes

### Co-launch Router and Workers

在单个进程中同时启动 router 与一组 SGLang worker:

```bash
python -m sglang_router.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4 \
  --host 0.0.0.0 \
  --port 30000
```

包含 router 参数(以 `--router-` 为前缀)的完整示例:

```bash
python -m sglang_router.launch_server \
  --host 0.0.0.0 \
  --port 8080 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --dp-size 8 \
  --grpc-mode \
  --log-level debug \
  --router-prometheus-port 10001 \
  --router-tool-call-parser llama \
  --router-model-path meta-llama/Llama-3.1-8B-Instruct \
  --router-policy round_robin \
  --router-log-level debug
```

### Separate Launch (HTTP)

独立运行 worker,并将 router 指向它们的 HTTP 端点:

```bash
# Worker nodes
python -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000
python -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8001

# Router node
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --policy cache_aware \
  --host 0.0.0.0 --port 30000
```

### gRPC Launch

使用 SRT gRPC worker 以解锁最高 throughput,并访问原生的 reasoning/tool 流水线:

```bash
# Workers expose gRPC endpoints
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --grpc-mode \
  --port 20000

# Router
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser json \
  --host 0.0.0.0 --port 8080
```

gRPC router 同时支持等价于常规 HTTP 的服务和 PD(prefill/decode)服务。只要连接模式解析为 gRPC,就需要提供 `--tokenizer-path` 或 `--model-path`(HuggingFace ID 或本地目录)。

### Prefill-Decode Disaggregation

拆分 prefill 与 decode worker,以实现 PD 感知的缓存与均衡:

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://prefill1:30001 9001 \
  --decode http://decode1:30011 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two
```

Prefill 条目可接受一个可选的 bootstrap port。PD 模式将 prefill 元数据与 decode 输出合并,并将结果流式返回给客户端。

### OpenAI Backend Proxy

代理 OpenAI 兼容端点,同时将历史与 MCP 会话保留在本地:

```bash
python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend memory
```

OpenAI 后端模式要求每个 router 实例恰好有一个 `--worker-urls` 条目。

### Multi-Model Inference Gateway

启用 IGW 模式,通过单个 router 路由多个模型:

```bash
./target/release/sgl-model-gateway \
  --enable-igw \
  --policy cache_aware \
  --max-concurrent-requests 512

# Register workers dynamically
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
        "url": "http://worker-a:8000",
        "model_id": "mistral",
        "priority": 10,
        "labels": {"tier": "gold"}
      }'
```

---

## API Reference

### Inference Endpoints

| Method | Path | 说明 |
|--------|------|-------------|
| `POST` | `/generate` | SGLang generate API |
| `POST` | `/v1/chat/completions` | OpenAI 兼容的 chat completions(流式/工具调用) |
| `POST` | `/v1/completions` | OpenAI 兼容的文本 completions |
| `POST` | `/v1/embeddings` | Embedding 生成(HTTP 与 gRPC) |
| `POST` | `/v1/rerank`, `/rerank` | Reranking 请求 |
| `POST` | `/v1/classify` | 文本分类 |

### Tokenization Endpoints

gateway 提供用于文本 tokenization 的 HTTP 端点,支持批处理,设计上与 SGLang 的 Python tokenization API 保持一致。

| Method | Path | 说明 |
|--------|------|-------------|
| `POST` | `/v1/tokenize` | 将文本 tokenize 为 token ID(单条或批量) |
| `POST` | `/v1/detokenize` | 将 token ID 还原为文本(单条或批量) |
| `POST` | `/v1/tokenizers` | 注册新的 tokenizer(异步,返回任务状态) |
| `GET` | `/v1/tokenizers` | 列出所有已注册的 tokenizer |
| `GET` | `/v1/tokenizers/{id}` | 通过 UUID 获取 tokenizer 信息 |
| `GET` | `/v1/tokenizers/{id}/status` | 检查 tokenizer 的异步加载状态 |
| `DELETE` | `/v1/tokenizers/{id}` | 从注册表中移除 tokenizer |

#### Tokenize Request

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "Hello, world!"
}
```

#### Batch Tokenize Request

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": ["Hello", "World", "How are you?"]
}
```

#### Tokenize Response

```json
{
  "tokens": [15339, 11, 1917, 0],
  "count": 4,
  "char_count": 13
}
```

#### Detokenize Request

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "tokens": [15339, 11, 1917, 0],
  "skip_special_tokens": true
}
```

#### Detokenize Response

```json
{
  "text": "Hello, world!"
}
```

#### Add Tokenizer (Async)

```bash
curl -X POST http://localhost:30000/v1/tokenizers \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3", "source": "meta-llama/Llama-3.1-8B-Instruct"}'
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Tokenizer registration queued"
}
```

Check status:
```bash
curl http://localhost:30000/v1/tokenizers/550e8400-e29b-41d4-a716-446655440000/status
```

### Parser Endpoints

gateway 提供管理端点,用于从 LLM 输出中解析 reasoning 内容和函数调用。

| Method | Path | 说明 |
|--------|------|-------------|
| `POST` | `/parse/reasoning` | 将 reasoning(`<think>`)从正常文本中分离 |
| `POST` | `/parse/function_call` | 从文本中解析函数/工具调用 |

#### Separate Reasoning Request

```json
{
  "text": "<think>Let me analyze this step by step...</think>The answer is 42.",
  "parser": "deepseek-r1"
}
```

#### Response

```json
{
  "normal_text": "The answer is 42.",
  "reasoning_text": "Let me analyze this step by step..."
}
```

#### Function Call Parsing

```json
{
  "text": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}",
  "parser": "json"
}
```

### Classification API

`/v1/classify` 端点使用序列分类模型(例如 `Qwen2ForSequenceClassification`、`BertForSequenceClassification`)提供文本分类。

#### Request

```bash
curl http://localhost:30000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": "I love this product!"
  }'
```

#### Response

```json
{
  "id": "classify-a1b2c3d4-5678-90ab-cdef-1234567890ab",
  "object": "list",
  "created": 1767034308,
  "model": "jason9693/Qwen2.5-1.5B-apeach",
  "data": [
    {
      "index": 0,
      "label": "positive",
      "probs": [0.12, 0.88],
      "num_classes": 2
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 0,
    "total_tokens": 6
  }
}
```

#### Response Fields

| Field | 说明 |
|-------|-------------|
| `label` | 预测的类别标签(来自模型的 `id2label` 配置,或回退为 `LABEL_N`) |
| `probs` | 所有类别上的概率分布(对 logits 做 softmax) |
| `num_classes` | 分类类别数量 |

#### Notes

- 分类复用了 embedding 后端——scheduler 返回 logits,再通过 softmax 转换为概率
- 标签来自模型的 HuggingFace 配置(`id2label` 字段);没有该映射的模型使用通用标签(`LABEL_0`、`LABEL_1` 等)
- HTTP 与 gRPC router 均支持分类

### Conversation and Response APIs

| Method | Path | 说明 |
|--------|------|-------------|
| `POST` | `/v1/responses` | 创建后台响应(agentic 循环) |
| `GET` | `/v1/responses/{id}` | 检索已存储的响应 |
| `POST` | `/v1/responses/{id}/cancel` | 取消后台响应 |
| `DELETE` | `/v1/responses/{id}` | 删除响应 |
| `GET` | `/v1/responses/{id}/input_items` | 列出响应的输入项 |
| `POST` | `/v1/conversations` | 创建会话 |
| `GET` | `/v1/conversations/{id}` | 获取会话 |
| `POST` | `/v1/conversations/{id}` | 更新会话 |
| `DELETE` | `/v1/conversations/{id}` | 删除会话 |
| `GET` | `/v1/conversations/{id}/items` | 列出会话项 |
| `POST` | `/v1/conversations/{id}/items` | 向会话添加项 |
| `GET` | `/v1/conversations/{id}/items/{item_id}` | 获取会话项 |
| `DELETE` | `/v1/conversations/{id}/items/{item_id}` | 删除会话项 |

### Worker Management APIs

| Method | Path | 说明 |
|--------|------|-------------|
| `POST` | `/workers` | 将 worker 注册请求入队(返回 202 Accepted) |
| `GET` | `/workers` | 列出 worker 及其健康、负载和策略元数据 |
| `GET` | `/workers/{worker_id}` | 查看特定 worker 或任务队列条目 |
| `PUT` | `/workers/{worker_id}` | 将 worker 更新请求入队 |
| `DELETE` | `/workers/{worker_id}` | 将 worker 移除请求入队 |

#### Add Worker

```bash
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{"url":"grpc://0.0.0.0:31000","worker_type":"regular"}'
```

#### List Workers

```bash
curl http://localhost:30000/workers
```

Response:
```json
{
  "workers": [
    {
      "id": "2f3a0c3e-3a7b-4c3f-8c70-1b7d4c3a6e1f",
      "url": "http://0.0.0.0:31378",
      "model_id": "mistral",
      "priority": 50,
      "cost": 1.0,
      "worker_type": "regular",
      "is_healthy": true,
      "load": 0,
      "connection_mode": "Http"
    }
  ],
  "total": 1,
  "stats": {
    "prefill_count": 0,
    "decode_count": 0,
    "regular_count": 1
  }
}
```

### Admin and Health Endpoints

| Method | Path | 说明 |
|--------|------|-------------|
| `GET` | `/liveness` | 健康检查(始终返回 OK) |
| `GET` | `/readiness` | 就绪检查(检查是否有健康的 worker 可用) |
| `GET` | `/health` | liveness 的别名 |
| `GET` | `/health_generate` | health generate 测试 |
| `GET` | `/engine_metrics` | 来自 worker 的引擎级指标 |
| `GET` | `/v1/models` | 列出可用模型 |
| `GET` | `/get_model_info` | 获取模型信息 |
| `GET` | `/server_info` | 获取服务器信息 |
| `POST` | `/flush_cache` | 清空所有缓存 |
| `GET` | `/get_loads` | 获取所有 worker 的负载 |
| `POST` | `/wasm` | 上传 WASM 模块 |
| `GET` | `/wasm` | 列出 WASM 模块 |
| `DELETE` | `/wasm/{module_uuid}` | 移除 WASM 模块 |

---

## Load Balancing Policies

| Policy | 说明 | 用法 |
|--------|-------------|-------|
| `random` | 均匀随机选择 | `--policy random` |
| `round_robin` | 按顺序轮询各 worker | `--policy round_robin` |
| `power_of_two` | 采样两个 worker 并选择较轻的那个 | `--policy power_of_two` |
| `cache_aware` | 结合缓存局部性与负载均衡(默认) | `--policy cache_aware` |
| `bucket` | 将 worker 划分到负载桶,并动态调整边界 | `--policy bucket` |

### Cache-Aware Policy Tuning

```bash
--cache-threshold 0.5 \
--balance-abs-threshold 32 \
--balance-rel-threshold 1.5 \
--eviction-interval-secs 120 \
--max-tree-size 67108864
```

| Parameter | 默认值 | 说明 |
|-----------|---------|-------------|
| `--cache-threshold` | 0.3 | 判定缓存命中的最小前缀匹配比例 |
| `--balance-abs-threshold` | 64 | 触发再均衡前的绝对负载差 |
| `--balance-rel-threshold` | 1.5 | 触发再均衡前的相对负载比 |
| `--eviction-interval-secs` | 120 | 缓存淘汰的周期(秒) |
| `--max-tree-size` | 67108864 | 缓存树中的最大节点数 |

---

## Reliability and Flow Control

### Retries

配置指数退避重试:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --retry-max-retries 5 \
  --retry-initial-backoff-ms 50 \
  --retry-max-backoff-ms 30000 \
  --retry-backoff-multiplier 1.5 \
  --retry-jitter-factor 0.2
```

| Parameter | 默认值 | 说明 |
|-----------|---------|-------------|
| `--retry-max-retries` | 5 | 最大重试次数 |
| `--retry-initial-backoff-ms` | 50 | 初始退避时长(ms) |
| `--retry-max-backoff-ms` | 5000 | 最大退避时长(ms) |
| `--retry-backoff-multiplier` | 2.0 | 指数退避乘数 |
| `--retry-jitter-factor` | 0.1 | 随机抖动因子(0.0-1.0) |
| `--disable-retries` | false | 完全禁用重试 |

**可重试的状态码:** 408、429、500、502、503、504

### Circuit Breaker

按 worker 的熔断器可防止级联故障:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --cb-failure-threshold 5 \
  --cb-success-threshold 2 \
  --cb-timeout-duration-secs 30 \
  --cb-window-duration-secs 60
```

| Parameter | 默认值 | 说明 |
|-----------|---------|-------------|
| `--cb-failure-threshold` | 5 | 触发熔断打开的连续失败次数 |
| `--cb-success-threshold` | 2 | 从半开状态恢复关闭所需的成功次数 |
| `--cb-timeout-duration-secs` | 30 | 进入半开尝试前的等待时间 |
| `--cb-window-duration-secs` | 60 | 失败计数窗口 |
| `--disable-circuit-breaker` | false | 禁用熔断器 |

**熔断器状态:**
- **Closed**:正常运行,允许请求
- **Open**:故障状态,请求立即被拒绝
- **Half-Open**:测试恢复,允许有限请求

### Rate Limiting and Queuing

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --max-concurrent-requests 256 \
  --rate-limit-tokens-per-second 512 \
  --queue-size 128 \
  --queue-timeout-secs 30
```

超出并发上限的请求会在 FIFO 队列中等待。返回:
- 队列已满时返回 `429 Too Many Requests`
- 队列超时到期时返回 `408 Request Timeout`

### Health Checks

```bash
--health-check-interval-secs 30 \
--health-check-timeout-secs 10 \
--health-success-threshold 2 \
--health-failure-threshold 3 \
--health-check-endpoint /health
```

---

## Reasoning Parser Integration

gateway 为使用思维链(Chain-of-Thought,CoT)推理并带有显式 thinking 块的模型内置了 reasoning parser。

### Supported Parsers

| Parser ID | 模型家族 | Think Tokens |
|-----------|--------------|--------------|
| `deepseek-r1` | DeepSeek-R1 | `<think>...</think>`(初始 reasoning) |
| `qwen3` | Qwen-3 | `<think>...</think>` |
| `qwen3-thinking` | Qwen-3 Thinking | `<think>...</think>`(初始 reasoning) |
| `kimi` | Kimi K2 | Unicode think tokens |
| `glm45` | GLM-4.5/4.6/4.7 | `<think>...</think>` |
| `step3` | Step-3 | `<think>...</think>` |
| `minimax` | MiniMax | `<think>...</think>` |

### Usage

```bash
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path deepseek-ai/DeepSeek-R1 \
  --reasoning-parser deepseek-r1
```

gRPC router 会自动:
1. 在流式输出中检测 reasoning 块
2. 将 reasoning 内容从正常文本中分离
3. 应用带缓冲管理的增量流式解析
4. 处理部分 token 检测,以保证正确的流式行为

---

## Tool Call Parsing

gateway 支持以多种格式解析 LLM 输出中的函数/工具调用。

### Supported Formats

| Parser | 格式 | 说明 |
|--------|--------|-------------|
| `json` | JSON | 标准 JSON 工具调用 |
| `python` | Pythonic | Python 函数调用语法 |
| `xml` | XML | XML 格式的工具调用 |

### Usage

```bash
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tool-call-parser json
```

---

## Tokenizer Management

### Tokenizer Sources

gateway 支持多种 tokenizer 后端:
- **HuggingFace**:通过模型 ID 从 HuggingFace Hub 加载
- **Local**:从本地的 `tokenizer.json` 或目录加载
- **Tiktoken**:自动检测 OpenAI GPT 模型(gpt-4、davinci 等)

### Configuration

```bash
# HuggingFace model
--model-path meta-llama/Llama-3.1-8B-Instruct

# Local tokenizer
--tokenizer-path /path/to/tokenizer.json

# With chat template override
--chat-template /path/to/template.jinja
```

### Tokenizer Caching

两级缓存以实现最佳性能:

| Cache | 类型 | 说明 |
|-------|------|-------------|
| L0 | 精确匹配 | 针对重复 prompt 的整串缓存 |
| L1 | 前缀匹配 | 针对增量 prompt 的前缀边界匹配 |

```bash
--enable-l0-cache \
--l0-max-entries 10000 \
--enable-l1-cache \
--l1-max-memory 52428800  # 50MB
```

---

## MCP Integration

gateway 提供原生的 Model Context Protocol(MCP)客户端集成,用于工具执行。

### Supported Transports

| Transport | 说明 |
|-----------|-------------|
| STDIO | 本地进程执行 |
| SSE | Server-Sent Events(HTTP) |
| Streamable | 双向流式 |

### Configuration

```bash
python -m sglang_router.launch_router \
  --mcp-config-path /path/to/mcp-config.yaml \
  --worker-urls http://worker1:8000
```

### MCP Configuration File

```yaml
servers:
  - name: "filesystem"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    protocol: "stdio"
    required: false

  - name: "github"
    url: "https://api.github.com/mcp"
    token: "ghp_xxxxx"
    protocol: "sse"
    required: false

  - name: "custom-tools"
    url: "https://tools.example.com/mcp"
    protocol: "streamable"
    required: true

pool:
  max_connections: 100
  idle_timeout: 300

proxy:
  http: "http://proxy.internal:8080"
  https: "https://proxy.internal:8443"
  no_proxy: "localhost,127.0.0.1,*.internal"

inventory:
  enable_refresh: true
  tool_ttl: 300
  refresh_interval: 300
```

---

## Service Discovery (Kubernetes)

通过 Kubernetes pod 选择器启用自动 worker 发现:

```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker role=inference \
  --service-discovery-namespace production \
  --service-discovery-port 8000
```

### PD Mode Discovery

```bash
--pd-disaggregation \
--prefill-selector app=sglang component=prefill \
--decode-selector app=sglang component=decode \
--service-discovery
```

Prefill pod 可通过 `sglang.ai/bootstrap-port` annotation 暴露 bootstrap port。RBAC 必须允许对 pod 执行 `get`、`list` 和 `watch`。

---

## History and Data Connectors

| Backend | 说明 | 用法 |
|---------|-------------|-------|
| `memory` | 内存存储(默认) | `--history-backend memory` |
| `none` | 不持久化 | `--history-backend none` |
| `oracle` | Oracle Autonomous Database | `--history-backend oracle` |
| `postgres` | PostgreSQL Database | `--history-backend postgres` |
| `redis` | Redis | `--history-backend redis` |

### Oracle Configuration

```bash
# Connection descriptor
export ATP_DSN="(description=(address=(protocol=tcps)(port=1522)(host=adb.region.oraclecloud.com))(connect_data=(service_name=service_name)))"

# Or TNS alias (requires wallet)
export ATP_TNS_ALIAS="sglroutertestatp_high"
export ATP_WALLET_PATH="/path/to/wallet"

# Credentials
export ATP_USER="admin"
export ATP_PASSWORD="secret"
export ATP_POOL_MIN=4
export ATP_POOL_MAX=32

python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend oracle
```

### PostgreSQL Configuration

```bash
export POSTGRES_DB_URL="postgres://user:password@host:5432/dbname"

python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend postgres
```

### Redis Configuration

```bash
export REDIS_URL="redis://localhost:6379"
export REDIS_POOL_MAX=16
export REDIS_RETENTION_DAYS=30

python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend redis \
  --redis-retention-days 30
```

使用 `--redis-retention-days -1` 实现持久化存储(默认为 30 天)。

---

## WASM Middleware

gateway 支持 WebAssembly(WASM)中间件模块,用于自定义请求/响应处理。这使得组织能够实现针对认证、限流、计费、日志等的特定逻辑——无需修改或重新编译 gateway。

### Overview

WASM 中间件运行在沙箱环境中,具备内存隔离、无网络/文件系统访问,以及可配置的资源限制。

| Attach Point | 执行时机 | 用例 |
|--------------|---------------|-----------|
| `OnRequest` | 转发到 worker 之前 | 认证、限流、请求修改 |
| `OnResponse` | 收到 worker 响应之后 | 日志、响应修改、错误处理 |

| Action | 说明 |
|--------|-------------|
| `Continue` | 不做修改,继续处理 |
| `Reject(status)` | 以 HTTP 状态码拒绝请求 |
| `Modify(...)` | 修改 header、body 或 status |

### Examples

完整可运行的示例位于 `examples/wasm/`:

| Example | 说明 |
|---------|-------------|
| `auth/` | 针对受保护路由的 API key 认证 |
| `rate_limit/` | 按客户端的限流(请求/分钟) |
| `logging/` | 请求跟踪 header 与响应修改 |

接口定义位于 `src/wasm/interface`。

### Building Modules

```bash
# Prerequisites
rustup target add wasm32-wasip2
cargo install wasm-tools

# Build
cargo build --target wasm32-wasip2 --release

# Convert to component format
wasm-tools component new \
  target/wasm32-wasip2/release/my_middleware.wasm \
  -o my_middleware.component.wasm
```

### Deploying Modules

```bash
# Enable WASM support
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --enable-wasm

# Upload module
curl -X POST http://localhost:30000/wasm \
  -H "Content-Type: application/json" \
  -d '{
    "modules": [{
      "name": "auth-middleware",
      "file_path": "/absolute/path/to/auth.component.wasm",
      "module_type": "Middleware",
      "attach_points": [{"Middleware": "OnRequest"}]
    }]
  }'

# List modules
curl http://localhost:30000/wasm

# Remove module
curl -X DELETE http://localhost:30000/wasm/{module_uuid}
```

### Runtime Configuration

| Parameter | 默认值 | 说明 |
|-----------|---------|-------------|
| `max_memory_pages` | 1024 (64MB) | WASM 最大内存 |
| `max_execution_time_ms` | 1000 | 执行超时 |
| `max_stack_size` | 1MB | 栈大小上限 |
| `module_cache_size` | 10 | 每个 worker 缓存的模块数 |

**注意:** 限流状态是按 worker 线程维护的,不在多个 gateway 副本之间共享。在生产环境中,建议在共享层(例如 Redis)实现限流。

---

## Language Bindings

SGLang Model Gateway 为 Python 和 Go 提供官方语言绑定,以便与不同的技术栈和组织需求集成。

### Python Bindings

Python 绑定提供了一个基于 PyO3 的 Rust gateway 库封装。这是一个直接的绑定,从 Python 调用 gateway 服务器的启动逻辑。

#### Installation

```bash
# From PyPI
pip install sglang-router

# Development build
cd sgl-model-gateway/bindings/python
pip install maturin && maturin develop --features vendored-openssl
```

#### Usage

本文档通篇都在使用 Python 绑定。详细示例见 [Quick Start](#quick-start) 和 [Deployment Modes](#deployment-modes) 章节。

关键组件:
- `RouterArgs` dataclass,提供 50+ 个配置项
- `Router.from_args()`,用于以编程方式启动
- CLI 命令:`smg launch`、`smg server`、`python -m sglang_router.launch_router`

### Go Bindings

Go 绑定为采用 Go 基础设施的组织提供高性能的 gRPC 客户端库。它非常适合:

- 与内部 Go 服务和工具集成
- 高性能客户端应用
- 构建自定义的 OpenAI 兼容代理服务器

#### Architecture

```
┌─────────────────────────────────────────┐
│         High-Level Go API               │
│   (client.go - OpenAI-style interface)  │
├─────────────────────────────────────────┤
│         gRPC Layer                      │
├─────────────────────────────────────────┤
│         Rust FFI Layer                  │
│   (Tokenization, Parsing, Conversion)   │
└─────────────────────────────────────────┘
```

**关键特性:**
- 通过 FFI 实现的原生 Rust tokenization(线程安全、无锁)
- 完整的流式支持,带 context 取消
- 可配置的 channel 缓冲区大小,适应高并发
- 内置工具调用解析与 chat template 应用

#### Installation

```bash
# Build the FFI library first
cd sgl-model-gateway/bindings/golang
make build && make lib

# Then use in your Go project
go get github.com/sgl-project/sgl-go-sdk
```

**要求:** Go 1.24+,Rust 工具链

#### Examples

完整可运行的示例位于 `bindings/golang/examples/`:

| Example | 说明 |
|---------|-------------|
| `simple/` | 非流式 chat completion |
| `streaming/` | 带 SSE 的流式 chat completion |
| `oai_server/` | 完整的 OpenAI 兼容 HTTP 服务器 |

```bash
# Run examples
cd sgl-model-gateway/bindings/golang/examples/simple && ./run.sh
cd sgl-model-gateway/bindings/golang/examples/streaming && ./run.sh
cd sgl-model-gateway/bindings/golang/examples/oai_server && ./run.sh
```

#### Testing

```bash
cd sgl-model-gateway/bindings/golang

# Unit tests
go test -v ./...

# Integration tests (requires running SGLang server)
export SGL_GRPC_ENDPOINT=grpc://localhost:20000
export SGL_TOKENIZER_PATH=/path/to/tokenizer
go test -tags=integration -v ./...
```

### Comparison

| Feature | Python | Go |
|---------|--------|-----|
| **主要用途** | Gateway 服务器启动器 | gRPC 客户端库 |
| **CLI 支持** | 完整 CLI(smg、sglang-router) | 仅库 |
| **K8s 发现** | 原生支持 | N/A(客户端库) |
| **PD 模式** | 内置 | N/A(客户端库) |

**何时使用 Python:** 启动并管理 gateway 服务器、服务发现、PD 分离。

**何时使用 Go:** 构建自定义客户端应用、与 Go 微服务集成、OpenAI 兼容代理服务器

---

## Security and Authentication

### Router API Key

```bash
python -m sglang_router.launch_router \
  --api-key "your-router-api-key" \
  --worker-urls http://worker1:8000
```

客户端访问受保护端点时必须提供 `Authorization: Bearer <key>`。

### Worker API Keys

```bash
# Add worker with explicit key
curl -H "Authorization: Bearer router-key" \
  -X POST http://localhost:8080/workers \
  -H "Content-Type: application/json" \
  -d '{"url":"http://worker:8000","api_key":"worker-key"}'
```

### Security Configurations

1. **无认证**(默认):仅在受信任的环境中使用
2. **仅 Router 认证**:客户端向 router 认证
3. **仅 Worker 认证**:router 开放,worker 需要 key
4. **完整认证**:router 与 worker 均受保护

### TLS (HTTPS) for Gateway Server

启用 TLS 以通过 HTTPS 提供 gateway 服务:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --tls-cert-path /path/to/server.crt \
  --tls-key-path /path/to/server.key
```

| Parameter | 说明 |
|-----------|-------------|
| `--tls-cert-path` | 服务器证书路径(PEM 格式) |
| `--tls-key-path` | 服务器私钥路径(PEM 格式) |

这两个参数必须同时提供。gateway 使用 rustls 配合 ring crypto provider 进行 TLS 终止。如果未配置 TLS,gateway 会回退到普通 HTTP。

### mTLS for Worker Communication

在 HTTP 模式下为与 worker 的安全通信启用双向 TLS(mTLS):

```bash
python -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --client-cert-path /path/to/client.crt \
  --client-key-path /path/to/client.key \
  --ca-cert-path /path/to/ca.crt
```

| Parameter | 说明 |
|-----------|-------------|
| `--client-cert-path` | 用于 mTLS 的客户端证书路径(PEM 格式) |
| `--client-key-path` | 用于 mTLS 的客户端私钥路径(PEM 格式) |
| `--ca-cert-path` | 用于验证 worker TLS 的 CA 证书路径(PEM 格式,可重复) |

**要点:**
- 客户端证书与私钥必须同时提供
- 可通过多个 `--ca-cert-path` flag 添加多个 CA 证书
- 配置 TLS 时使用 rustls 后端
- 为所有 worker 创建单个 HTTP 客户端(假定处于单一安全域)
- 为长连接启用了 TCP keepalive(30 秒)

### Full TLS Configuration Example

Gateway HTTPS + Worker mTLS + API Key 认证:

```bash
python -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --tls-cert-path /etc/certs/server.crt \
  --tls-key-path /etc/certs/server.key \
  --client-cert-path /etc/certs/client.crt \
  --client-key-path /etc/certs/client.key \
  --ca-cert-path /etc/certs/ca.crt \
  --api-key "secure-api-key" \
  --policy cache_aware
```

---

## Observability

### Prometheus Metrics

通过 `--prometheus-host`/`--prometheus-port` 启用(默认为 `0.0.0.0:29000`)。

#### Metric Categories (40+ metrics)

| Layer | Prefix | Metrics |
|-------|--------|---------|
| HTTP | `smg_http_*` | `requests_total`, `request_duration_seconds`, `responses_total`, `connections_active`, `rate_limit_total` |
| Router | `smg_router_*` | `requests_total`, `request_duration_seconds`, `request_errors_total`, `stage_duration_seconds`, `upstream_responses_total` |
| Inference | `smg_router_*` | `ttft_seconds`, `tpot_seconds`, `tokens_total`, `generation_duration_seconds` |
| Worker | `smg_worker_*` | `pool_size`, `connections_active`, `requests_active`, `health_checks_total`, `selection_total`, `errors_total` |
| Circuit Breaker | `smg_worker_cb_*` | `state`, `transitions_total`, `outcomes_total`, `consecutive_failures`, `consecutive_successes` |
| Retry | `smg_worker_*` | `retries_total`, `retries_exhausted_total`, `retry_backoff_seconds` |
| Discovery | `smg_discovery_*` | `registrations_total`, `deregistrations_total`, `sync_duration_seconds`, `workers_discovered` |
| MCP | `smg_mcp_*` | `tool_calls_total`, `tool_duration_seconds`, `servers_active`, `tool_iterations_total` |
| Database | `smg_db_*` | `operations_total`, `operation_duration_seconds`, `connections_active`, `items_stored` |

#### Key Inference Metrics (gRPC mode)

| Metric | 类型 | 说明 |
|--------|------|-------------|
| `smg_router_ttft_seconds` | Histogram | 首 token 时间(Time to first token) |
| `smg_router_tpot_seconds` | Histogram | 每个输出 token 的时间(Time per output token) |
| `smg_router_tokens_total` | Counter | token 总数(输入/输出) |
| `smg_router_generation_duration_seconds` | Histogram | 端到端生成时间 |

#### Duration Buckets

1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 15s, 30s, 45s, 60s, 90s, 120s, 180s, 240s

### OpenTelemetry Tracing

通过 OTLP 导出启用分布式追踪:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --enable-trace \
  --otlp-traces-endpoint localhost:4317
```

#### Features

- OTLP/gRPC exporter(默认端口 4317)
- 针对 HTTP 和 gRPC 的 W3C Trace Context 透传
- 批量 span 处理(500ms 延迟,64 个 span 的批大小)
- 自定义过滤以减少噪声
- 向上游 worker 请求中注入 trace context
- Service name:`sgl-router`

### Logging

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --log-level debug \
  --log-dir ./router_logs
```

带可选文件落地(file sink)的结构化追踪。日志级别:`debug`、`info`、`warn`、`error`。

### Request ID Propagation

```bash
--request-id-headers x-request-id x-trace-id x-correlation-id
```

响应会包含 `x-request-id` header 以便关联。

---

## Production Recommendations

本节为在生产环境中部署 SGLang Model Gateway 提供指导。

### Security Best Practices

**在生产中始终启用 TLS:**

```bash
python -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --tls-cert-path /etc/certs/server.crt \
  --tls-key-path /etc/certs/server.key \
  --client-cert-path /etc/certs/client.crt \
  --client-key-path /etc/certs/client.key \
  --ca-cert-path /etc/certs/ca.crt \
  --api-key "${ROUTER_API_KEY}"
```

**安全检查清单:**
- 为 gateway HTTPS 终止启用 TLS
- 当 worker 处于不受信任的网络时,为 worker 通信启用 mTLS
- 设置 `--api-key` 以保护 router 端点
- 使用 Kubernetes Secrets 或密钥管理工具存放凭据
- 定期轮换证书和 API key
- 使用防火墙或网络策略限制网络访问

### High Availability

**扩展策略:**

gateway 支持在负载均衡器后面运行多个副本以实现高可用。不过需要注意以下重要事项:

| Component | 跨副本共享 | 影响 |
|-----------|----------------------|--------|
| Worker Registry | 否(各自独立) | 每个副本独立发现 worker |
| Radix Cache Tree | 否(各自独立) | 缓存命中可能下降 10-20% |
| Circuit Breaker State | 否(各自独立) | 每个副本独立跟踪失败 |
| Rate Limiting | 否(各自独立) | 限流按副本生效,而非全局 |

**建议:**

1. **优先水平扩展而非垂直扩展**:部署多个较小的 gateway 副本,而不是单个 CPU 与内存过大的实例。这带来:
   - 更好的容错性(单个副本故障不会导致整个 gateway 宕机)
   - 更可预测的资源使用
   - 更易于容量规划

2. **使用 Kubernetes 服务发现**:让 gateway 自动发现并管理 worker:
   ```bash
   python -m sglang_router.launch_router \
     --service-discovery \
     --selector app=sglang-worker \
     --service-discovery-namespace production
   ```

3. **接受缓存效率上的权衡**:在多副本情况下,cache-aware 路由策略的 radix 树不会在副本之间同步。这意味着:
   - 每个副本构建自己的缓存树
   - 来自同一用户的请求可能命中不同副本
   - 预期缓存命中率下降:**10-20%**
   - 考虑到 HA 的收益,这通常是可接受的

4. **配置会话亲和性(可选)**:如果缓存效率至关重要,可基于请求的一致性哈希(例如用户 ID 或 API key)为负载均衡器配置会话亲和性。

**HA 架构示例:**
```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │   (L4/L7)       │
                    └────────┬────────┘
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │  Gateway  │  │  Gateway  │  │  Gateway  │
        │ Replica 1 │  │ Replica 2 │  │ Replica 3 │
        └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │  Worker   │  │  Worker   │  │  Worker   │
        │  Pod 1    │  │  Pod 2    │  │  Pod N    │
        └───────────┘  └───────────┘  └───────────┘
```

### Performance

**使用 gRPC 模式以获得高 throughput:**

对于 SGLang worker,gRPC 模式提供最高性能:

```bash
# Start workers in gRPC mode
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --grpc-mode \
  --port 20000

# Configure gateway for gRPC
python -m sglang_router.launch_router \
  --worker-urls grpc://worker1:20000 grpc://worker2:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --policy cache_aware
```

**gRPC 的性能优势:**
- 原生 Rust tokenization(无 Python 开销)
- 更低延迟的流式
- 内置 reasoning parser 执行
- 在 gateway 中进行工具调用解析
- 减少序列化开销

**调优建议:**

| Parameter | 建议 | 原因 |
|-----------|---------------|--------|
| `--policy` | `cache_aware` | 最适合重复 prompt,延迟降低约 30% |
| `--max-concurrent-requests` | worker 数量的 2-4 倍 | 在最大化 throughput 的同时防止过载 |
| `--queue-size` | max-concurrent 的 2 倍 | 为突发流量提供缓冲 |
| `--request-timeout-secs` | 基于最大生成长度 | 防止请求卡死 |

### Kubernetes Deployment

**用于服务发现的 Pod 标签:**

为了让 gateway 自动发现 worker,请为 worker pod 一致地打标签:

```yaml
# Worker Deployment (Regular Mode)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-worker
  namespace: production
spec:
  replicas: 4
  selector:
    matchLabels:
      app: sglang-worker
      component: inference
  template:
    metadata:
      labels:
        app: sglang-worker
        component: inference
        model: llama-3-8b
    spec:
      containers:
      - name: worker
        image: lmsysorg/sglang:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 20000
          name: grpc
```

**用于发现的 Gateway 配置:**
```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker component=inference \
  --service-discovery-namespace production \
  --service-discovery-port 8000
```

**PD(Prefill/Decode)模式标签:**

```yaml
# Prefill Worker
metadata:
  labels:
    app: sglang-worker
    component: prefill
  annotations:
    sglang.ai/bootstrap-port: "9001"

# Decode Worker
metadata:
  labels:
    app: sglang-worker
    component: decode
```

**用于 PD 发现的 Gateway 配置:**
```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --pd-disaggregation \
  --prefill-selector app=sglang-worker component=prefill \
  --decode-selector app=sglang-worker component=decode \
  --service-discovery-namespace production
```

**RBAC 要求:**

gateway 需要 watch pod 的权限:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: sglang-gateway
  namespace: production
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: sglang-gateway
  namespace: production
subjects:
- kind: ServiceAccount
  name: sglang-gateway
  namespace: production
roleRef:
  kind: Role
  name: sglang-gateway
  apiGroup: rbac.authorization.k8s.io
```

### Monitoring with PromQL

配置 Prometheus 抓取 gateway 的指标端点(默认:`:29000/metrics`)。

**核心仪表盘:**

**1. 请求速率与延迟:**
```promql
# Request rate by endpoint
sum(rate(smg_http_requests_total[5m])) by (path, method)

# P50 latency
histogram_quantile(0.50, sum(rate(smg_http_request_duration_seconds_bucket[5m])) by (le))

# P99 latency
histogram_quantile(0.99, sum(rate(smg_http_request_duration_seconds_bucket[5m])) by (le))

# Error rate
sum(rate(smg_http_responses_total{status=~"5.."}[5m])) / sum(rate(smg_http_responses_total[5m]))
```

**2. Worker 健康:**
```promql
# Healthy workers
sum(smg_worker_pool_size)

# Active connections per worker
smg_worker_connections_active

# Worker health check failures
sum(rate(smg_worker_health_checks_total{result="failure"}[5m])) by (worker_id)
```

**3. 熔断器状态:**
```promql
# Circuit breaker states (0=closed, 1=open, 2=half-open)
smg_worker_cb_state

# Circuit breaker transitions
sum(rate(smg_worker_cb_transitions_total[5m])) by (worker_id, from_state, to_state)

# Workers with open circuits
count(smg_worker_cb_state == 1)
```

**4. 推理性能(gRPC 模式):**
```promql
# Time to first token (P50)
histogram_quantile(0.50, sum(rate(smg_router_ttft_seconds_bucket[5m])) by (le, model))

# Time per output token (P99)
histogram_quantile(0.99, sum(rate(smg_router_tpot_seconds_bucket[5m])) by (le, model))

# Token throughput
sum(rate(smg_router_tokens_total[5m])) by (model, direction)

# Generation duration P95
histogram_quantile(0.95, sum(rate(smg_router_generation_duration_seconds_bucket[5m])) by (le))
```

**5. 限流与排队:**
```promql
# Rate limit rejections
sum(rate(smg_http_rate_limit_total{decision="rejected"}[5m]))

# Queue depth (if using concurrency limiting)
smg_worker_requests_active

# Retry attempts
sum(rate(smg_worker_retries_total[5m])) by (worker_id)

# Exhausted retries (failures after all retries)
sum(rate(smg_worker_retries_exhausted_total[5m]))
```

**6. MCP 工具执行:**
```promql
# Tool call rate
sum(rate(smg_mcp_tool_calls_total[5m])) by (server, tool)

# Tool latency P95
histogram_quantile(0.95, sum(rate(smg_mcp_tool_duration_seconds_bucket[5m])) by (le, tool))

# Active MCP server connections
smg_mcp_servers_active
```

**告警规则示例:**

```yaml
groups:
- name: sglang-gateway
  rules:
  - alert: HighErrorRate
    expr: |
      sum(rate(smg_http_responses_total{status=~"5.."}[5m]))
      / sum(rate(smg_http_responses_total[5m])) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate on SGLang Gateway"

  - alert: CircuitBreakerOpen
    expr: count(smg_worker_cb_state == 1) > 0
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Worker circuit breaker is open"

  - alert: HighLatency
    expr: |
      histogram_quantile(0.99, sum(rate(smg_http_request_duration_seconds_bucket[5m])) by (le)) > 30
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "P99 latency exceeds 30 seconds"

  - alert: NoHealthyWorkers
    expr: sum(smg_worker_pool_size) == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "No healthy workers available"
```

---

## Configuration Reference

### Core Settings

| Parameter | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `--host` | str | 127.0.0.1 | Router host |
| `--port` | int | 30000 | Router port |
| `--worker-urls` | list | [] | Worker URL(HTTP 或 gRPC) |
| `--policy` | str | cache_aware | 路由策略 |
| `--max-concurrent-requests` | int | -1 | 并发上限(-1 表示禁用) |
| `--request-timeout-secs` | int | 600 | 请求超时 |
| `--max-payload-size` | int | 256MB | 最大请求负载 |

### Prefill/Decode

| Parameter | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `--pd-disaggregation` | flag | false | 启用 PD 模式 |
| `--prefill` | list | [] | Prefill URL + 可选 bootstrap port |
| `--decode` | list | [] | Decode URL |
| `--prefill-policy` | str | None | 覆盖 prefill 节点的策略 |
| `--decode-policy` | str | None | 覆盖 decode 节点的策略 |
| `--worker-startup-timeout-secs` | int | 600 | Worker 初始化超时 |

### Kubernetes Discovery

| Parameter | 类型 | 说明 |
|-----------|------|-------------|
| `--service-discovery` | flag | 启用发现 |
| `--selector` | list | 标签选择器(key=value) |
| `--prefill-selector` / `--decode-selector` | list | PD 模式选择器 |
| `--service-discovery-namespace` | str | 要监视的 namespace |
| `--service-discovery-port` | int | Worker 端口(默认 80) |
| `--bootstrap-port-annotation` | str | bootstrap port 的 annotation |

### TLS Configuration

| Parameter | 类型 | 说明 |
|-----------|------|-------------|
| `--tls-cert-path` | str | gateway HTTPS 的服务器证书(PEM) |
| `--tls-key-path` | str | gateway HTTPS 的服务器私钥(PEM) |
| `--client-cert-path` | str | worker mTLS 的客户端证书(PEM) |
| `--client-key-path` | str | worker mTLS 的客户端私钥(PEM) |
| `--ca-cert-path` | str | 用于验证 worker 的 CA 证书(PEM,可重复) |

---

## Troubleshooting

### Workers Never Ready

增大 `--worker-startup-timeout-secs`,或确保在 router 启动前健康探针已能响应。

### Load Imbalance / Hot Workers

按 worker 检查 `smg_router_requests_total`,并调优 cache-aware 阈值(`--balance-*`、`--cache-threshold`)。

### Circuit Breaker Flapping

增大 `--cb-failure-threshold`,或延长 timeout/window 时长。可考虑临时禁用重试。

### Queue Overflow (429)

增大 `--queue-size`,或降低客户端并发。确保 `--max-concurrent-requests` 与下游容量相匹配。

### Memory Growth

减小 `--max-tree-size`,或降低 `--eviction-interval-secs` 以进行更激进的缓存裁剪。

### Debugging

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --log-level debug \
  --log-dir ./router_logs
```

### gRPC Connection Issues

确保 worker 以 `--grpc-mode` 启动,并验证已向 router 提供 `--model-path` 或 `--tokenizer-path`。

### Tokenizer Loading Failures

对于私有模型,检查 HuggingFace Hub 凭据(`HF_TOKEN` 环境变量)。验证本地路径是否可访问。

---

SGLang Model Gateway 会随 SGLang 运行时持续演进。在采用新特性或贡献改进时,请保持 CLI flag、集成和文档的一致。
