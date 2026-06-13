# SGLang Model Gateway

面向大规模 LLM 部署的高性能模型路由控制面与数据面。该网关编排成组的 worker，在 HTTP 与 gRPC 后端之间均衡流量，并对外暴露兼容 OpenAI 的 API，支持可插拔的历史存储与工具集成——同时针对 SGLang 服务运行时进行了深度优化。

## 概述
- 统一的控制面，用于在异构模型集群中注册、监控和编排 prefill、decode 及常规 worker。
- 数据面，在 HTTP、PD（prefill/decode）、gRPC 以及兼容 OpenAI 的后端之间路由请求，并共享可靠性特性。
- 业界首个 gRPC 流水线，原生采用 Rust 实现的分词、推理（reasoning）与工具调用执行，面向高吞吐的 OpenAI 兼容服务。
- 多模型推理网关模式（`--enable-igw`），可同时运行多个路由器并应用按模型的策略。
- 会话（conversation）、响应（response）与聊天历史连接器，将状态集中在路由器侧，从而在模型/MCP 循环之间实现合规共享，提供内存、空操作（no-op）或 Oracle ATP 存储选项。
- 内置可靠性原语：带指数退避的重试、熔断器、令牌桶限流与排队。
- 一流的可观测性，支持结构化日志、OpenTelemetry 链路追踪与 Prometheus 指标。

### 架构速览
**控制面**
- Worker Manager 校验 worker、发现其能力，并保持注册表同步。
- Job Queue 串行化后台操作（添加/移除），并通过 `/workers/{worker_id}` 暴露状态。
- 后台健康检查器与负载监控器持续向熔断器和策略提供信息。
- 可选的 Kubernetes 服务发现，使注册表与 Pod 保持一致。

**数据面**
- 用于常规与 PD（prefill/decode）流量的 SGLang HTTP 路由器，具备策略感知的选择能力。
- SGLang gRPC 路由器与流水线，将分词后的请求流式传输至 SRT gRPC worker，并以完全 Rust 实现的分词器、推理解析器和工具解析器实现极致的 OpenAI API 性能，同时支持单阶段与 PD 服务拓扑。
- OpenAI 路由器，将 OpenAI 风格的请求、响应和会话代理到远端供应商（OpenAI、xAI、Gemini 及其他 OpenAI 兼容提供商），同时保留流式/SSE 语义。
- Router Manager 在启用 IGW 时协调多个路由器实现。
- 弹性层提供令牌桶限流、请求排队、重试执行器以及按 worker 的熔断器，以保证流量在故障时仍能通行。
- 高级负载均衡，包含缓存感知的请求复用、负载感知（power-of-two）选择，以及按模型的策略覆盖。

## 特性亮点
- 多种负载均衡策略（`random`、`round_robin`、`cache_aware`、`power_of_two`、`bucket`），并支持 DP 感知调度。
- 多模型 HTTP 服务与推理网关路由，支持按模型的特定策略。
- Prefill/decode 分离，包括 bootstrap 端口处理与缓存感知的合并。
- gRPC 路由，完全采用 Rust 实现的分词器加载、推理解析器选择与工具解析器集成，面向 OpenAI 兼容端点——在流式与非流式模式下支持 DeepSeek、Llama、Kimi K2、Qwen、GPT-OSS、Mistral、Step-3、GLM4、GLM4.7 及其他具备推理能力的模型。
- 兼容 OpenAI 的 `/v1/chat/completions`、`/v1/responses`、`/v1/conversations`、`/v1/embeddings`、`/v1/rerank`、`/v1/classify` 端点。
- **分词 API**：提供 tokenize（`/v1/tokenize`）和 detokenize（`/v1/detokenize`）的 HTTP 端点并支持批量；以及用于动态注册的分词器管理 API。
- **解析器端点**：推理解析器（`/parse/reasoning`）和函数调用解析器（`/parse/function_call`），用于分离推理内容并提取工具调用。
- 原生 MCP 客户端集成，支持所有 MCP 传输协议（STDIO、HTTP、SSE 和 Streamable）以实现工具执行循环。
- 可插拔的历史连接器：内存、禁用、Oracle ATP 或 PostgreSQL（支持连接池与凭据）。
- 可靠性控制：带抖动的重试、按 worker 范围的熔断器、带可选队列的令牌桶限流器，以及缓存刷新 API。
- 面向常规与 PD 工作负载的服务发现，支持独立的选择器（selector）。
- **全面的可观测性**：覆盖 HTTP、路由器、worker、熔断器、重试、发现、MCP 与数据库各层的 40+ 项 Prometheus 指标；支持带 OTLP 导出的 OpenTelemetry 追踪；以及带请求 ID 透传的结构化日志。

## 文档
- **用户指南**：[docs.sglang.io/advanced_features/sgl_model_gateway.html](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)
- 更多指南、API 参考与部署模式将随 SGLang 版本持续更新。

## 安装

### Docker
Docker Hub 上提供预构建的 Docker 镜像，支持多架构（x86_64 和 ARM64）：
```bash
docker pull lmsysorg/sgl-model-gateway:latest
```

### 前置条件
- **Rust 和 Cargo**
  ```bash
  # 安装 rustup（Rust 安装器与版本管理器）
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

  # 重新加载 shell 环境
  source "$HOME/.cargo/env"

  # 验证安装
  rustc --version
  cargo --version
  ```
- **Python**，并具备可用的 `pip` 和 virtualenv 工具。

### Rust 二进制
```bash
# 构建 release 二进制
cargo build --release
```

### Python 包
```bash
pip install maturin

# 快速开发模式（debug 构建，不生成 wheel，即时）
# 使用系统 OpenSSL（需要 libssl-dev/openssl-devel）
cd bindings/python
maturin develop

# 生产构建（已优化，生成 wheel）
# 使用内置（vendored）OpenSSL（跨平台兼容）
cd bindings/python
maturin build --release --out dist --features vendored-openssl
pip install --force-reinstall dist/*.whl

# 使用系统 OpenSSL 的开发构建（更快）
# 需要：apt install libssl-dev pkg-config（Ubuntu/Debian）
#   或：yum install openssl-devel（RHEL/CentOS）
cd bindings/python
maturin build --release --out dist
pip install --force-reinstall dist/*.whl
```
> **注意：** Python 绑定位于 `bindings/python/`，拥有独立的 Cargo.toml。开发过程中使用 `maturin develop` 进行快速迭代（以 debug 模式构建并直接安装）。生产环境的 wheel 使用 `maturin build --release --features vendored-openssl`，可获得完整优化（opt-level="z"、lto="fat"）和跨平台兼容性。该包使用 abi3 支持以兼容 Python 3.8+。

## 检查版本

安装后，验证安装并查看版本信息：

```bash
# 简单版本（Rust 二进制）
./target/release/sgl-model-gateway --version
# 或使用别名
./target/release/smg --version
./target/release/amg --version

# 包含构建详情的完整版本信息
./target/release/sgl-model-gateway --version-verbose

# Python CLI
amg --version
amg --version-verbose
python3 -m sglang_router --version
```

`--version`（或 `-V`）标志显示版本字符串。使用 `--version-verbose` 可获取包含 Git 提交、构建时间、编译器版本与平台详情的完整构建信息。

## 快速开始
### 常规 HTTP 路由
- **Rust 二进制**
  ```bash
  ./target/release/sgl-model-gateway \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware
  ```
  开发期间 `cargo run --release -- …` 提供相同的行为。
- **Python 启动器**
  ```bash
  python3 -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware
  ```

### Prefill/Decode 分离（PD）
- **Rust 二进制**
  ```bash
  ./target/release/sgl-model-gateway \
    --pd-disaggregation \
    --prefill http://prefill1:30001 9001 \
    --prefill http://prefill2:30002 \
    --decode http://decode1:30011 \
    --decode http://decode2:30012 \
    --policy cache_aware \
    --prefill-policy cache_aware \
    --decode-policy power_of_two
  ```
- **Python 启动器**
  ```bash
  python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://prefill1:30001 9001 \
    --prefill http://prefill2:30002 \
    --decode http://decode1:30011 \
    --decode http://decode2:30012 \
    --policy cache_aware
  ```
Prefill 条目可接受一个可选的 bootstrap 端口。PD 模式将 prefill 元数据与 decode 输出合并，并将结果流式回传给客户端。

### 多模型推理网关
启用 IGW 模式，通过单个路由器路由多个模型，同时应用按模型的策略：
```bash
./target/release/sgl-model-gateway \
  --enable-igw \
  --policy cache_aware \
  --max-concurrent-requests 512

# 动态注册 worker
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
        "url": "http://worker-a:8000",
        "model_id": "mistral",
        "priority": 10,
        "labels": {"tier": "gold"}
      }'

# 添加另一个使用不同模型/策略提示的 worker
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
        "url": "http://worker-b:8000",
        "model_id": "llama3",
        "priority": 20,
        "labels": {"policy": "power_of_two", "tier": "silver"}
      }'

# 查看已注册的 worker
curl http://localhost:30000/workers
```
示例响应（http worker）：
```json
{
  "workers": [
    {"id":"2f3a0c3e-3a7b-4c3f-8c70-1b7d4c3a6e1f","url":"http://0.0.0.0:31378","model_id":"mistral","priority":50,"cost":1.0,"worker_type":"regular","is_healthy":true,"load":0,"connection_mode":"Http"},
    {"id":"9b0f6c2a-1c4f-4c2a-9f4a-1f2a6c0b9d3e","url":"http://0.0.0.0:34881","model_id":"llama3","priority":50,"cost":1.0,"worker_type":"regular","is_healthy":true,"load":0,"connection_mode":"Http"}
  ],
  "total": 2,
  "stats": {
    "prefill_count": 0,
    "decode_count": 0,
    "regular_count": 2
  }
}
```
使用相同的 API 可添加更多 worker；按需包含可选的 `labels`（用于按模型策略）或 `tokenizer_path` / `reasoning_parser` / `tool_parser` 字段。`/workers/{worker_id}` 在后台作业完成注册期间暴露排队作业的状态。

### gRPC 路由
- **Rust 二进制**
  ```bash
  ./target/release/sgl-model-gateway \
    --worker-urls grpc://worker-grpc-0:31001 grpc://worker-grpc-1:31002 \
    --tokenizer-path /path/to/tokenizer.json \
    --reasoning-parser deepseek-r1 \
    --tool-call-parser json
  ```
- **Python 路由器**
  ```bash
  python3 -m sglang_router.launch_router \
    --worker-urls grpc://127.0.0.1:20000 \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8080
  ```
gRPC 路由器在本地完成输入分词，支持工具调用解析，并以流式方式返回响应。当 worker 注册表中包含 PD worker 时，它同时支持等价于常规 HTTP 的服务以及 PD（prefill/decode）服务。每当连接模式解析为 gRPC 时，请提供 `--model-path` 或 `--tokenizer-path`（HuggingFace ID 或本地目录）。
使用 `--reasoning-parser` 选择内置的推理流水线（DeepSeek-R1、Qwen3、Step-3、GLM4、GLM4.7 等），并使用 `--tool-call-parser` 在流式或非流式模式下选择 JSON/Pythonic/XML 工具契约。

### OpenAI 后端模式
将请求路由到 OpenAI 或 OpenAI 兼容的端点：

```bash
# 路由到 OpenAI API
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \

# 路由到自定义的 OpenAI 兼容端点（Gemini、xAI 等）
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls http://my-openai-compatible-service:8000 \
```

**注意事项**
- OpenAI 后端模式作为代理，指向单一远端端点；不应用负载均衡。
- 每个路由器实例应只提供一个 `--worker-urls` 条目。
- Rust 二进制支持相同的标志（`./target/release/sgl-model-gateway --backend openai ...`）。

### MCP 集成
SGL Model Gateway 提供原生的 Model Context Protocol（MCP）客户端集成，支持跨 STDIO、SSE 和 Streamable 传输的工具调用。MCP 服务器通过 YAML 配置文件进行配置，并在启动时通过工作流引擎注册。

#### 基本用法
```bash
# Rust 二进制
./target/release/sgl-model-gateway \
  --mcp-config-path /path/to/mcp-config.yaml \
  --worker-urls http://worker1:8000

# Python 启动器
python3 -m sglang_router.launch_router \
  --mcp-config-path /path/to/mcp-config.yaml \
  --worker-urls http://worker1:8000
```

#### MCP 配置文件
创建 MCP 配置文件以定义服务器、传输方式和连接设置：

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
  idle_timeout: 300  # 秒

proxy:
  http: "http://proxy.internal:8080"
  https: "https://proxy.internal:8443"
  no_proxy: "localhost,127.0.0.1,*.internal"

inventory:
  enable_refresh: true
  tool_ttl: 300  # 秒 - 工具被视为“新鲜”的时长
  refresh_interval: 300  # 秒 - 后台刷新间隔
```

#### 配置选项

**服务器配置**（`servers` 数组）：
- `name`：MCP 服务器的唯一标识符
- `command` + `args`：用于 STDIO 传输（本地进程执行）
- `url`：用于 SSE 或 Streamable 传输（HTTP/HTTPS 端点）
- `token`：基于 HTTP 的传输的可选认证令牌
- `protocol`：协议类型（`"sse"`、`"streamable"` 或 `"stdio"`）
- `required`：若为 `true`，当服务器不可达时路由器将启动失败（默认：`false`）
- `envs`：STDIO 进程的环境变量（可选）
- `proxy`：按服务器的代理覆盖（设为 `null` 可绕过全局代理）

**连接池**（`pool`）：
- `max_connections`：动态服务器的最大池化连接数（默认：100）
- `idle_timeout`：空闲连接清理前的超时时间（秒，默认：300）

**代理配置**（`proxy`）：
- `http`/`https`：MCP 服务器连接的代理 URL（非 LLM 流量）
- `no_proxy`：以逗号分隔、需排除代理的主机（支持通配符）
- **注意**：目前 `streamable` 传输会忽略代理设置。如需代理支持，请使用 STDIO 或 SSE 传输。

**清单（Inventory）设置**（`inventory`）：
- `enable_refresh`：启用工具清单的自动后台刷新（默认：true）
- `tool_ttl`：工具缓存 TTL（秒）——工具被视为“新鲜”的时长（默认：300）
- `refresh_interval`：后台刷新间隔（秒）——主动刷新清单（默认：300）

#### 传输类型

**STDIO**（本地进程）：
```yaml
name: "local-tools"
command: "python"
args: ["-m", "my_mcp_server"]
envs:
  API_KEY: "secret"
  DEBUG: "true"
```

**SSE**（Server-Sent Events）：
```yaml
name: "remote-sse"
url: "https://mcp.example.com/events"
token: "bearer-token"
protocol: "sse"
```

**Streamable**（双向流式）：
```yaml
name: "streaming-tools"
url: "https://mcp.example.com/stream"
protocol: "streamable"
required: true
```

#### 服务器生命周期
- MCP 服务器通过工作流引擎注册，并带有重试逻辑（100 次尝试，STDIO 服务器有 2 小时超时）
- 发现阶段识别工具、提示词（prompts）和资源
- 工具清单以可配置的 TTL 缓存并定期刷新
- 失败的可选服务器记录警告；必需服务器则中止启动
- 静态服务器（来自配置）是永久的；动态服务器（按请求）使用连接池

通过 Prometheus 指标查看 MCP 活动（`mcp_*` 指标），并通过 admin API 查看工作流作业状态。

### Python 启动器（路由器 + Worker）
同时启动路由器与 SGLang worker 进程；`launch_server` 一次性拉起 worker（HTTP 或 gRPC）和路由器。
```bash
python3 -m sglang_router.launch_server --host 0.0.0.0
```
按生产部署需要添加标志：
```bash
python3 -m sglang_router.launch_server \
  --host 0.0.0.0 \
  --port 8080 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --dp-size 8 \
  --grpc-mode
```
省略 `--grpc-mode` 即可启动 HTTP worker；路由器会自动配置 worker URL，并根据提供的 DP 大小进行调度。

### Mini 负载均衡器（调试）
```bash
python3 -m sglang_router.launch_router \
  --mini-lb \
  --pd-disaggregation \
  --prefill http://localhost:30001 \
  --decode http://localhost:30011
```
MiniLB 使用简单的随机路由转发 PD 请求，仅用于本地调试。

### 运行 Worker 服务器
使用上游 SGLang 二进制启动专用的 worker 进程。
- **Prefill worker 服务器（gRPC 模式）**：
  ```bash
  python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 20000 \
    --tp-size 1 \
    --grpc-mode
  ```
  对于 HTTP worker，移除 `--grpc-mode`。结合上文的路由器命令，通过 CLI 标志或控制面 API 注册该 worker。

## 控制面

### Worker 生命周期与作业队列
- `JobQueue` 处理异步的添加/移除操作，以避免阻塞客户端。
- `WorkerManager` 检查 worker 元数据（`/server_info`、`/get_model_info`），跟踪负载，并暴露 `flush_cache` 和 `get_loads`。
- 按 worker 的熔断器和健康探测保持注册表健康；负载监控器为缓存感知与 power-of-two 策略提供指标。

### 管理与 Worker API
| 方法     | 路径             | 说明                                                                                                                                                       |
|----------|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `POST`   | `/workers`       | 排队注册 worker（prefill/decode/regular）。请求体匹配 `WorkerConfigRequest`。在作业队列处理请求期间返回 `202 Accepted`。                                     |
| `GET`    | `/workers`       | 列出 worker，包含健康状态、负载、策略元数据以及排队作业状态。                                                                                               |
| `GET`    | `/workers/{worker_id}` | 检查特定 worker 或作业队列条目（UUID）。                                                                                                               |
| `PUT`    | `/workers/{worker_id}` | 按 UUID 排队更新 worker。                                                                                                                              |
| `DELETE` | `/workers/{worker_id}` | 按 UUID 排队移除 worker。                                                                                                                              |
| `POST`   | `/flush_cache`   | 触发跨 HTTP worker 的缓存刷新，并给出成功/失败明细。                                                                                                         |
| `GET`    | `/get_loads`     | 采样每个 worker 上报的当前负载。                                                                                                                            |

当提供 `--api-key` 时，所有管理路由都继承路由器的 API-key 保护。作业状态包含带时间戳的 `pending`、`processing` 和 `failed` 阶段。

### 服务发现
启用 Kubernetes 发现以自动协调 worker：
```bash
./target/release/sgl-model-gateway \
  --service-discovery \
  --selector app=sglang-worker role=inference \
  --service-discovery-namespace sglang-system \
  --service-discovery-port 8000
```
PD 模式接受专用的选择器：
```bash
--pd-disaggregation \
--prefill-selector app=sglang component=prefill \
--decode-selector app=sglang component=decode \
--service-discovery
```
Prefill Pod 可通过 `sglang.ai/bootstrap-port` 注解暴露 bootstrap 端口。RBAC 必须允许对 Pod 执行 `get`、`list` 和 `watch`。

## 数据面

### 路由器能力（HTTP 与 gRPC）
两套路由器堆栈均：
- 共享负载均衡策略（random、round-robin、cache-aware、power-of-two），并支持 DP 感知调度、重试、熔断器与限流。
- 按请求记录指标，跟踪运行中负载，并与路由器范围的策略注册表集成。

HTTP 路由器暴露完整的 OpenAI 兼容接口（`/generate`、`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`、`/v1/responses`、`/v1/rerank` 等）。gRPC 路由器目前提供极速的 `/generate` 和 `/v1/chat/completions`，其余端点在其流水线完成前返回 `501 Not Implemented`。

#### HTTP 路由器细节
- **常规路由器** 处理经典的单阶段 worker，并支持按模型的策略覆盖。
- **Prefill/Decode 路由器** 协调分离的 prefill 与 decode worker，合并元数据，并管理流式聚合（fan-in）。

#### gRPC 路由器细节
- 业界首个完全采用 Rust 实现的 OpenAI 兼容 gRPC 推理网关，包含进程内的分词器、推理解析器与工具解析器执行，以获得最大吞吐。
- 同时支持单阶段与 PD（prefill/decode）worker 拓扑；路由器会按模型自动选择合适的流水线。
- 提供与 HTTP 路由器相同的 `/v1/*` API，同时将分词后的请求/响应直接流式传输给 SRT gRPC worker。
- 内置面向 DeepSeek、Qwen、Llama、Mistral、GPT-OSS、Step-3、GLM4、GLM4.7、Kimi K2 及其他结构化思维模型的推理解析器。
- 面向 JSON、Pythonic、XML 及自定义模式的工具调用解析器，支持流式与非流式执行循环。
- 分词器工厂支持 HuggingFace 模型、本地 tokenizer.json 文件以及聊天模板覆盖（参见 `src/tokenizer`）。
- 可在 `src/reasoning_parser`、`src/tool_parser` 和 `src/tokenizer` 中探索为 gRPC 模式提供动力的端到端 Rust 实现的代码路径。

### OpenAI 路由器
- 代理 OpenAI 兼容的 chat completions 与 responses API，端到端保留请求头与 SSE 流。
- 支持 `/v1/responses` 后台作业，可取消、删除并列出输入项——从而在不于远端供应商端点持久化数据的情况下实现智能体式、多轮编排。
- 会话 API（`/v1/conversations` 和 `/v1/conversations/{id}/items`）与所配置的会话存储后端交互，以实现合规的聊天历史管理。会话状态保存在路由器层，因此同一份历史可驱动不同的模型或 MCP 循环，而不会向上游供应商泄露数据。
- 聊天历史、智能体式多轮 `/v1/responses` 以及原生 MCP 客户端（STDIO/HTTP/SSE/Streamable 传输）的设计旨在通过将敏感状态保留在路由器内，满足企业数据隐私要求。

### 请求端点
| 端点                                                                             | 备注                                                       |
|----------------------------------------------------------------------------------|------------------------------------------------------------|
| `POST /generate`                                                                 | SGLang generate API。                                      |
| `POST /v1/chat/completions`                                                      | OpenAI 兼容的 chat。支持流式与工具调用。                   |
| `POST /v1/completions`                                                           | OpenAI 兼容的文本补全。                                    |
| `POST /v1/responses`                                                             | 创建后台 responses，返回 response ID。                     |
| `GET /v1/responses/{id}`                                                         | 检索已存储的 responses。                                   |
| 会话端点（`/v1/conversations`、`/v1/conversations/{id}`、`/v1/conversations/{id}/items`） | 管理聊天历史。                                             |
| `POST /v1/embeddings`                                                            | 转发 embedding 请求（HTTP 与 gRPC）。                      |
| `POST /v1/rerank`、`POST /rerank`                                                | 排序（Ranking）API。                                       |
| `POST /v1/classify`                                                              | 文本分类端点。                                             |

### 分类（Classification）API

`/v1/classify` 端点使用序列分类模型（例如 `Qwen2ForSequenceClassification`、`BertForSequenceClassification`）提供文本分类。

**请求：**
```bash
curl http://localhost:30000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": "I love this product!"
  }'
```

**响应：**
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

**字段：**
- `label`：预测的类别标签（来自模型的 `id2label` 配置，或回退为 `LABEL_N`）
- `probs`：所有类别上的概率分布（对 logits 做 softmax）
- `num_classes`：分类类别数

**注意事项：**
- 分类复用 embedding 后端——调度器返回 logits，再通过 softmax 转换为概率
- 标签来自模型的 HuggingFace 配置（`id2label` 字段）；没有该映射的模型使用通用标签（`LABEL_0`、`LABEL_1` 等）
- HTTP 与 gRPC 路由器都支持分类

公共健康端点（`/liveness`、`/readiness`、`/health`、`/health_generate`）反映注册表状态；readiness 会确保 PD worker 已配对，且 IGW 至少有一条健康路由。

### 分词（Tokenization）端点

网关提供文本分词的 HTTP 端点，设计上对齐 SGLang Python 分词 API，并支持批量操作。

| 端点                          | 方法     | 说明                                                  |
|-------------------------------|----------|-------------------------------------------------------|
| `POST /v1/tokenize`           | `POST`   | 将文本分词为 token ID（单条或批量）。                 |
| `POST /v1/detokenize`         | `POST`   | 将 token ID 还原为文本（单条或批量）。                |
| `POST /v1/tokenizers`         | `POST`   | 注册新的分词器（异步，返回作业状态）。                |
| `GET /v1/tokenizers`          | `GET`    | 列出所有已注册的分词器。                              |
| `GET /v1/tokenizers/{id}`     | `GET`    | 按 UUID 获取分词器信息。                              |
| `GET /v1/tokenizers/{id}/status` | `GET` | 检查异步分词器的加载状态。                            |
| `DELETE /v1/tokenizers/{id}`  | `DELETE` | 从注册表中移除分词器。                                |

**Tokenize 请求：**
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "Hello, world!"
}
```

**批量 Tokenize 请求：**
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": ["Hello", "World", "How are you?"]
}
```

**Tokenize 响应：**
```json
{
  "tokens": [15339, 11, 1917, 0],
  "count": 4,
  "char_count": 13
}
```

**Detokenize 请求：**
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "tokens": [15339, 11, 1917, 0],
  "skip_special_tokens": true
}
```

**添加分词器（异步注册）：**
```bash
# 从 HuggingFace 注册
curl -X POST http://localhost:30000/v1/tokenizers \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3", "source": "meta-llama/Llama-3.1-8B-Instruct"}'

# 检查状态
curl http://localhost:30000/v1/tokenizers/{tokenizer_id}/status
```

### 解析器（Parser）端点

网关提供管理端点，用于从 LLM 输出中解析推理内容与函数调用。

| 端点                     | 方法   | 说明                                                   |
|--------------------------|--------|--------------------------------------------------------|
| `POST /parse/reasoning`  | `POST` | 将推理内容（`<think>`）与普通文本分离。                |
| `POST /parse/function_call` | `POST` | 从文本中解析函数/工具调用。                          |

**分离推理请求：**
```json
{
  "text": "<think>Let me analyze this step by step...</think>The answer is 42.",
  "parser": "deepseek-r1"
}
```

**响应：**
```json
{
  "normal_text": "The answer is 42.",
  "reasoning_text": "Let me analyze this step by step..."
}
```

**支持的推理解析器：**
- `deepseek-r1` - DeepSeek-R1（初始推理模式）
- `qwen3` - Qwen-3 模型
- `qwen3-thinking` / `qwen-thinking` - Qwen thinking 变体
- `kimi` - 带 Unicode token 的 Kimi K2
- `glm45` / `glm47` - GLM-4.5/4.6/4.7 模型
- `step3` - Step-3 模型
- `minimax` - MiniMax 模型

**函数调用解析：**
```json
{
  "text": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}",
  "parser": "json"
}
```

支持的工具解析器：`json`、`python`、`xml`。

## 会话、响应与数据连接器
- `--history-backend memory`（默认）将响应和会话存储在进程内。
- `--history-backend none` 在保留 API 的同时禁用持久化。
- `--history-backend oracle` 使用 Oracle Autonomous Database；通过标志或环境变量提供凭据。
- `--history-backend postgres` 使用 PostgreSQL 数据库。
- `--history-backend redis` 使用 Redis。
- 会话项存储与历史后端保持一致（Oracle 或内存）。同一存储同时为 OpenAI `/responses` 和会话 API 提供支持。

### 历史后端（OpenAI 路由器模式）
存储会话与响应数据，用于跟踪、调试或分析。

> **注意：** 目前仅在以 `--backend openai` 运行时支持历史后端。gRPC 模式对 `/v1/responses` API 的支持正在规划中。

#### 可用存储选项
- **Memory**（默认）：内存存储，快速但易失。
- **None**：无存储，开销最小。
- **Oracle**：由 Oracle Autonomous Database 支持的持久化存储。
- **Postgres**：由 PostgreSQL 数据库支持的持久化存储。
- **Redis**：由 Redis 支持的持久化存储。

```bash
# Memory 后端（默认）
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend memory

# 无存储以获得最高性能
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend none

# Oracle ATP 后端（参见下方配置）
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend oracle

# PostgreSQL 后端
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend postgres

# Redis 后端
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend redis
```

#### Oracle 配置
安装 Oracle Instant Client 并相应设置 `LD_LIBRARY_PATH`。请选择**一种**连接方式：
```bash
# 方式 1：完整连接描述符
export ATP_DSN="(description=(address=(protocol=tcps)(port=1522)(host=adb.region.oraclecloud.com))(connect_data=(service_name=service_name)))"

# 方式 2：TNS 别名（需要 wallet）
export ATP_TNS_ALIAS="sglroutertestatp_high"
export ATP_WALLET_PATH="/path/to/wallet"
```
提供数据库凭据和可选的连接池配置：
```bash
export ATP_USER="admin"
export ATP_PASSWORD="YourPassword123"
export ATP_POOL_MIN=4
export ATP_POOL_MAX=32
```

路由器标志映射到这些值：
- `--oracle-dsn`（环境变量：`ATP_DSN`）或 `--oracle-tns-alias` 配合 `--oracle-wallet-path`。
- `--oracle-user` / `--oracle-password`（`ATP_USER` / `ATP_PASSWORD`）。
- 使用 TNS 别名时的 `--oracle-wallet-path`（`ATP_WALLET_PATH`）。
- `--oracle-pool-min`、`--oracle-pool-max`、`--oracle-pool-timeout-secs`。

`--oracle-dsn` 与 `--oracle-tns-alias` 只应提供其一。

#### Redis 配置
提供 Redis 连接 URL 和可选的连接池配置：
```bash
export REDIS_URL="redis://localhost:6379"
export REDIS_POOL_MAX=16
export REDIS_RETENTION_DAYS=30
```

路由器标志映射到这些值：
- `--redis-url`（环境变量：`REDIS_URL`）
- `--redis-pool-max`（环境变量：`REDIS_POOL_MAX`）
- `--redis-retention-days`（环境变量：`REDIS_RETENTION_DAYS`）。设为 `-1` 表示持久化存储（默认：30 天）。

## 可靠性与流量控制
- **重试**：默认最大重试次数 = 5，采用指数退避（`--retry-max-retries`、`--retry-initial-backoff-ms`、`--retry-max-backoff-ms`、`--retry-backoff-multiplier`、`--retry-jitter-factor`）。重试在 408/429/500/502/503/504 时触发。
- **熔断器**：按 worker 的阈值（`--cb-failure-threshold`、`--cb-success-threshold`、`--cb-timeout-duration-secs`、`--cb-window-duration-secs`）。通过 `--disable-circuit-breaker` 禁用。
- **限流**：由 `--max-concurrent-requests` 驱动的令牌桶。设置 `--rate-limit-tokens-per-second` 可覆盖补充速率。通过 `--queue-size` 和 `--queue-timeout-secs` 配置请求队列；排队请求遵循 FIFO 顺序并尊重取消。
- **健康检查**：通过 `--health-check-interval-secs`、`--health-check-timeout-secs`、失败/成功阈值以及 `--health-check-endpoint` 进行运行时探测。使用 `--disable-health-check` 可完全跳过健康检查。
- **缓存管理**：`/flush_cache` 在重新部署 PD worker 时确保 LRU 淘汰。

## 负载均衡策略
- `random`：均匀随机选择 worker。
- `round_robin`：使用原子计数器的顺序轮询。
- `cache_aware`：维护提示词的前缀树，以路由重复流量，并通过可配置阈值（`--cache-threshold`、`--balance-abs-threshold`、`--balance-rel-threshold`、`--eviction-interval`、`--max-tree-size`）均衡负载。
- `power_of_two`：在两个随机候选中选择较轻的 worker；与 `LoadMonitor` 集成。
  在 PD 模式（`--prefill-policy`、`--decode-policy`）和 IGW 模式（通过 worker 注册表）下可进行按模型的覆盖。

## 可观测性

### 日志
通过 `tracing` 进行结构化追踪，支持可选的文件输出（`--log-dir`）和 `--log-level`（`debug`、`info`、`warn`、`error`）。

### Prometheus 指标
通过 `--prometheus-host`/`--prometheus-port`（默认为 `0.0.0.0:29000`）启用。

**指标类别（40+ 项指标）：**

| 层 | 指标前缀 | 说明 |
|-------|---------------|-------------|
| HTTP | `smg_http_*` | 请求计数、耗时、活跃连接、限流 |
| 路由器 | `smg_router_*` | 按模型/端点的请求、延迟、错误、上游响应 |
| 推理 | `smg_router_ttft/tpot/tokens_*` | 首 token 时间、每个输出 token 的时间、token 计数（gRPC） |
| Worker | `smg_worker_*` | 池大小、活跃连接、健康检查、选择事件 |
| 熔断器 | `smg_worker_cb_*` | 状态（closed/open/half-open）、转换、结果 |
| 重试 | `smg_worker_retries_*` | 重试次数、重试耗尽、退避时长 |
| 发现 | `smg_discovery_*` | K8s 注册、同步耗时、发现的 worker |
| MCP | `smg_mcp_*` | 工具调用、耗时、活跃服务器、迭代 |
| 数据库 | `smg_db_*` | 操作、耗时、连接、已存储项 |

**关键指标：**
- `smg_router_ttft_seconds` - 首 token 时间直方图（gRPC 模式）
- `smg_router_tpot_seconds` - 每个输出 token 时间直方图（gRPC 模式）
- `smg_router_tokens_total` - 按模型的输入/输出 token 总数
- `smg_router_generation_duration_seconds` - 端到端生成时间
- `smg_worker_cb_state` - 熔断器状态仪表（0=closed，1=open，2=half-open）

**耗时分桶（Buckets）：**
1ms、5ms、10ms、25ms、50ms、100ms、250ms、500ms、1s、2.5s、5s、10s、15s、30s、45s、60s、90s、120s、180s、240s

### OpenTelemetry 追踪
通过 OTLP 导出启用分布式追踪：

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --enable-trace \
  --otlp-traces-endpoint localhost:4317
```

**特性：**
- OTLP/gRPC 导出器（默认端口 4317）
- 针对 HTTP 与 gRPC 的 W3C Trace Context 透传
- 批量 span 处理（500ms 延迟，64 个 span 的批大小）
- 自定义过滤以降低噪声（仅导出相关 span）
- 向上游 worker 请求注入追踪上下文

**配置：**
- `--enable-trace` - 启用 OpenTelemetry 追踪
- `--otlp-traces-endpoint <host:port>` - OTLP 收集器端点

### 请求 ID 透传
配置用于提取请求 ID 的请求头：
```bash
--request-id-headers x-request-id x-trace-id x-correlation-id
```
响应中包含用于关联的 `x-request-id` 请求头。

### CORS
设置 `--cors-allowed-origins` 以允许浏览器访问。

## 安全

### 路由器与 Worker 的 API Key
- **路由器 API key（`--api-key`）** 保护客户端对路由器端点的访问；所有受保护路由都需要 `Authorization: Bearer <key>`。
- `--worker-urls` 中列出的 worker 会自动继承路由器 API key。
- 动态添加 worker 时，需通过 payload 或查询字符串显式提供 API key；它们**不会**自动继承。

```bash
# 路由器与初始 worker 共享同一 key
python3 -m sglang_router.launch_router \
  --api-key "shared-api-key" \
  --worker-urls http://worker1:8000 http://worker2:8000

# 在路由器已设置 key 的情况下添加无 key 的 worker，会触发警告并使该 worker 不受保护
curl -X POST http://localhost:8080/add_worker?url=http://worker3:8000

# 添加带显式 key 的 worker
curl -X POST "http://localhost:8080/add_worker?url=http://worker3:8000&api_key=worker3-specific-key"
```

### 安全配置
1. **无认证**（默认）：路由器与 worker 在无 key 的情况下接受请求——仅在可信环境中使用。
2. **仅路由器认证**：提供 `--api-key`；客户端必须提供 key，路由器在无凭据的情况下访问 worker。
3. **仅 Worker 认证**：路由器对客户端开放；每个 worker 需要自己的 key。在调用 `/workers` 或 `/add_worker` 时提供 key。
4. **完整认证**：设置路由器 API key 并为每个 worker 提供 key。示例：
   ```bash
   python3 -m sglang_router.launch_router --api-key "router-key"
   curl -H "Authorization: Bearer router-key" \
     -X POST http://localhost:8080/add_worker?url=http://worker:8000&api_key=worker-key
   ```

### 重要说明
- 通过 CLI 声明的初始 worker 继承路由器 key；动态 worker 必须显式提供 key。
- 当路由器期望认证而某个 worker 在注册时没有提供 key 时，路由器会记录警告。
- 即使路由器与 worker 共享同一 key，在调用动态注册 API 时仍需带上该 key。

### 网关服务器的 TLS（HTTPS）

启用 TLS 以通过 HTTPS 提供网关服务：

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --tls-cert-path /path/to/server.crt \
  --tls-key-path /path/to/server.key
```

| 参数 | 说明 |
|-----------|-------------|
| `--tls-cert-path` | 服务器证书路径（PEM 格式） |
| `--tls-key-path` | 服务器私钥路径（PEM 格式） |

两个参数必须同时提供。网关使用 rustls 配合 ring 加密提供者进行 TLS 终止。如果未配置 TLS，网关将回退到纯 HTTP。

### Worker 通信的 mTLS

在 HTTP 模式下启用双向 TLS（mTLS）以与 worker 安全通信：

```bash
python3 -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --client-cert-path /path/to/client.crt \
  --client-key-path /path/to/client.key \
  --ca-cert-path /path/to/ca.crt
```

| 参数 | 说明 |
|-----------|-------------|
| `--client-cert-path` | 用于 mTLS 的客户端证书路径（PEM 格式） |
| `--client-key-path` | 用于 mTLS 的客户端私钥路径（PEM 格式） |
| `--ca-cert-path` | 用于验证 worker TLS 的 CA 证书路径（PEM 格式） |

**关键要点：**
- 客户端证书和私钥必须同时提供
- 可通过多个 `--ca-cert-path` 标志添加多个 CA 证书
- 配置 TLS 时使用 rustls 后端
- 为所有 worker 创建单个 HTTP 客户端（假定为单一安全域）
- 为长连接启用 TCP keepalive（30 秒）

**完整 TLS 示例（网关 HTTPS + Worker mTLS）：**
```bash
python3 -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --tls-cert-path /etc/certs/server.crt \
  --tls-key-path /etc/certs/server.key \
  --client-cert-path /etc/certs/client.crt \
  --client-key-path /etc/certs/client.key \
  --ca-cert-path /etc/certs/ca.crt \
  --api-key "secure-api-key"
```

### 控制面认证

网关为控制面 API（worker 管理、分词器注册、缓存操作）支持基于角色的访问控制（RBAC）。提供两种认证方式：

#### 认证方式

| 方式 | 适用场景 | 配置 |
|--------|----------|---------------|
| **API Keys** | 服务账户、内部服务 | `--control-plane-api-keys` |
| **JWT/OIDC** | 通过身份提供商进行用户认证 | `--jwt-issuer`、`--jwt-audience` |

两种方式可同时使用。请求按以下顺序认证：API key → JWT token。

#### 角色

| 角色 | 访问权限 |
|------|--------|
| `admin` | 对所有控制面 API（worker、分词器、缓存等）的完全访问 |
| `user` | 仅推理/数据面 API（chat completions、embeddings 等） |

#### API Key 认证

用于服务账户与自动化的静态 API key：

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --control-plane-api-keys 'svc1:CI Pipeline:admin:secret-key-123' \
                           'svc2:Monitoring:user:readonly-key-456' \
  --control-plane-audit-enabled
```

**格式：** `id:name:role:key`
- `id` - key 的唯一标识符
- `name` - 人类可读的描述
- `role` - `admin` 或 `user`
- `key` - 密钥（内部以 SHA-256 哈希存储）

**用法：**
```bash
curl -H "Authorization: Bearer secret-key-123" \
  http://localhost:30000/workers
```

#### JWT/OIDC 认证

通过外部身份提供商（Azure AD、Okta、Auth0、Keycloak 等）认证用户：

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --jwt-issuer "https://login.microsoftonline.com/{tenant-id}/v2.0" \
  --jwt-audience "api://my-gateway-client-id" \
  --jwt-jwks-uri "https://login.microsoftonline.com/{tenant-id}/discovery/v2.0/keys" \
  --jwt-role-mapping 'Gateway.Admins=admin' 'Gateway.Users=user' \
  --control-plane-audit-enabled
```

| 参数 | 说明 |
|-----------|-------------|
| `--jwt-issuer` | OIDC 颁发者（issuer）URL。用于校验 `iss` 声明，并通过 `.well-known/openid-configuration` 发现 JWKS 端点。 |
| `--jwt-audience` | 期望的受众（`aud` 声明）。通常是你应用的 client ID 或 API 标识（例如 `api://client-id`）。 |
| `--jwt-jwks-uri` | （可选）显式的 JWKS URI。若省略，则从颁发者的 OIDC 配置中自动发现。 |
| `--jwt-role-mapping` | 将 IDP 的组/角色名映射为网关角色。格式：`idp_role=gateway_role`。 |

**工作原理：**
1. 用户通过身份提供商认证（OAuth2/OIDC 流程）
2. IDP 颁发 JWT token
3. 用户将 token 发送给网关：`Authorization: Bearer <jwt-token>`
4. 网关校验该 JWT：
   - 根据 JWKS 验证签名
   - 检查 `iss` 是否匹配 `--jwt-issuer`
   - 检查 `aud` 是否匹配 `--jwt-audience`
   - 校验过期时间及其他标准声明
   - 从 `roles` 声明（或回退到 `groups`）提取角色
   - 通过 `--jwt-role-mapping` 将 IDP 角色映射为网关角色

**Azure AD 配置示例：**
```bash
# Azure AD 颁发的 token 包含：
#   iss: https://login.microsoftonline.com/{tenant}/v2.0
#   aud: api://your-client-id（或 client ID 本身）
#   roles: ["Gateway.Admins"] 或 groups: ["group-id"]

python3 -m sglang_router.launch_router \
  --jwt-issuer "https://login.microsoftonline.com/your-tenant-id/v2.0" \
  --jwt-audience "api://your-client-id" \
  --jwt-role-mapping 'Gateway.Admins=admin' 'Gateway.Users=user'
```

#### 审计日志

启用 `--control-plane-audit-enabled` 可记录所有控制面操作，包含：
- 时间戳
- 主体（API key ID 或 JWT subject）
- 角色
- 执行的操作
- 成功/失败状态

#### 组合认证示例

针对不同用例同时使用 API key 与 JWT：

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  # 用于服务账户的 API key
  --control-plane-api-keys 'ci:CI/CD Pipeline:admin:ci-secret' \
  # 通过 Azure AD 面向真人用户的 JWT
  --jwt-issuer "https://login.microsoftonline.com/{tenant}/v2.0" \
  --jwt-audience "api://gateway" \
  --jwt-role-mapping 'Platform.Admins=admin' 'Platform.Users=user' \
  # 启用审计日志
  --control-plane-audit-enabled
```

## 开发与测试
```bash
# 构建 Rust 组件（debug 模式，快速）
cargo build

# 运行 Rust 测试
cargo test

# 快速 Python 开发（以 debug 模式重新构建并安装）
cd bindings/python && maturin develop

# 运行 Python 测试
cd ../..  # 回到 sgl-model-gateway 根目录
pytest e2e_test/
```
对于生产构建，请在 `bindings/python/` 目录下使用 `maturin build --release --out dist` 创建优化后的 wheel。开发期间，`maturin develop` 会即时重新构建并安装，而不生成 wheel 文件。使用 `python -m sglang_router.launch_server` 在小型集群中联合启动路由器与 SGLang worker，以进行本地验证。

### 构建缓存

**本地开发** 默认使用增量编译（在 `.cargo/config.toml` 中配置），对于编辑-编译-测试循环最为合适。

**对于 release 构建或 CI**，你可以选择使用 [sccache](https://github.com/mozilla/sccache) 来缓存编译产物：

```bash
# 安装 sccache
cargo install sccache

# 方式 1：设置环境变量（按会话）
export RUSTC_WRAPPER=sccache
cargo build --release

# 方式 2：添加到全局 cargo 配置（~/.cargo/config.toml）
# [build]
# rustc-wrapper = "sccache"
```

> **注意：** sccache 与增量编译互斥——sccache 无法缓存增量编译的 crate。本项目默认使用增量编译以加快本地迭代。在缓存跨构建更重要的 clean/release 构建中使用 sccache。CI 工作流使用 sccache 配合 GitHub Actions 缓存后端，以实现跨作业的编译缓存。

---

## 发布管理

### 创建网关发布

为 Gateway/Router 组件创建发布，并对提交进行过滤：

```bash
# 使用 make
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0

# 保存到文件
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 OUTPUT=RELEASE_NOTES.md

# 创建草稿发布（需要 gh CLI，默认行为）
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1

# 立即发布（需要 gh CLI）
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1 DRAFT=0
```

**标签命名**：使用 `gateway-*` 或 `router-*` 前缀，以避免触发无关的 CI 工作流。

### 发布工作流

1. **创建并推送标签**：
   ```bash
   git tag -a gateway-v1.0.0 <commit-hash> -m "Gateway release v1.0.0"
   git push origin gateway-v1.0.0
   ```

2. **生成发布说明**（自动过滤与网关相关的提交）：
   ```bash
   make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0
   ```

3. **创建 GitHub 发布**：
   ```bash
   # 创建草稿（默认 - 发布前先审阅）
   make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1

   # 或立即发布（跳过草稿）
   make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1 DRAFT=0
   ```

### 过滤的路径

发布说明仅包含涉及以下路径的提交：
- `sgl-model-gateway/` - 路由器代码库
- `python/sglang/srt/grpc/` - gRPC 协议
- `python/sglang/srt/entrypoints/grpc_server.py` - gRPC 服务器

该脚本会自动提取作者归属、PR 链接，并识别新贡献者。

---

SGLang Model Gateway 将持续与核心 SGLang 运行时一同演进。贡献时应保持 CLI 标志、文档与 Python 绑定同 Rust 实现一致。

