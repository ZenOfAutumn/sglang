# DP、DPA 和 SGLang DP Router

本指南解释 Data Parallelism(DP,数据并行)与 Data Parallelism Attention(DPA,数据并行注意力)之间的区别,如何正确启用每种模式,以及如何使用 SGLang Model Gateway(SMG)进行生产级的 DP 部署。

## Data Parallelism(DP,数据并行)

**Data Parallelism(DP,数据并行)** 是最常见的并行策略,它在多个 GPU 集合上复制整个模型,并并行处理不同批次的请求。每个 GPU 集合处理独立的请求。借助专用的路由策略(我们将在后文介绍),在 SGLang Model Gateway 中配合那些恰当的路由算法,你的服务系统吞吐量几乎可以线性倍增。

### 关键特征

- 每个副本拥有模型的完整副本
- 请求被分发/分散到各副本
- 在单个请求的推理期间副本间没有通信(对于简单 DP)

## Data Parallelism Attention(DPA,数据并行注意力)

**Data Parallelism Attention(DPA,数据并行注意力)**,也称为 DP Attention,是一种高级并行策略。虽然 DPA 对 **Multi-Head Latent Attention(MLA)** 模型(如 DeepSeek、MiniMax、Kimi-K2)带来的收益最为显著,但它也支持像 Qwen 这样的 **标准注意力模型**。

### MLA 模型使用张量并行的问题

最常见的推理并行策略是 **Tensor Parallelism(TP,张量并行)**。然而,对于某些模型,TP 可能并非最高效的策略。例如,DeepSeek 模型使用 MLA,只有 **一个 KV head**。如果我们在 8 个 GPU 上使用张量并行,将导致:

- 在所有 GPU 上 **复制 KV cache**
- **不必要的内存占用**,从而限制 batch size
- 由于内存约束导致 **吞吐量降低**

### DPA 的工作原理

DPA 通过 **专门对注意力组件应用数据并行** 来解决这些限制。

<table>
<tr>
<td width="50%">
<img src="../_static/image/dpa.png" alt="DPA + EP Architecture" width="100%">
</td>
<td width="50%" valign="top">

**每个 DP 副本:**

- 独立处理不同的批次(可处于不同的 forward 模式:prefill、decode 或 idle)
- 维护自己的 KV cache(无重复)
- 由于节省内存,可支持显著更大的 batch size

**DPA + EP 中的通信模式:**
-
-  **All2All (Dispatch)**:根据 gating 决策将 token 路由到 expert 子组
- **All2All (Combine)**:将 expert 计算出的结果收集回原始 token 位置

</td>
</tr>
</table>

### DPA 的关键收益

1. **显著降低 KV cache 内存**:每个 DP 副本只存储自己批次的 KV cache
2. **更大的 batch size**:内存节省可支持更大的 batch size
3. **改善 decoding 吞吐量**:为基于 MLA 的模型带来显著的吞吐量提升
4. **独立的 forward 模式**:每个 DP 副本可处于不同的 forward 模式(prefill、decode 或 idle),并在注意力计算期间独立处理分配给它的批次

### 用于 MoE 的 DPA 与 Expert Parallelism

对于像 DeepSeek 这样的 MoE 模型,DPA **通常** 与 Expert Parallelism(EP)配对,以在大规模下获得最佳吞吐量。然而,**DPA 不要求 EP**:如果你的部署不需要 expert 分片,你可以在不使用 EP 的情况下启用 DPA。

- 将 256+ 个 expert 权重分布到多个 GPU 上(无法装入单个 GPU)
- 通过 DeepEP 实现高效的 all-to-all token 路由
- 扩展到大型集群(相比原生 TP 可达 5 倍吞吐量提升)

### DeepSeek 的推荐设置

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --dp-size 8 \
    --ep 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm
```

> **注意**:使用 `--enable-dp-attention` 时必须显式设置 `--dp-size`。如果 `dp_size` 为 1(默认),DPA 将被禁用。

关于详细的 EP 配置(DeepEP、Two-Batch Overlap、EPLB),请参阅 [Expert Parallelism](expert_parallelism.md)。

### 目标模型

DPA 支持以下模型架构:

- **MLA(Multi-Head Latent Attention)模型** —— DPA 在此带来最显著的收益:
  - DeepSeek 系列(DeepSeek-V2、DeepSeek-V3、DeepSeek-R1)
  - MiniMax 模型
  - Kimi-K2
  - 其他使用 MLA 架构的模型

- **标准注意力模型** —— 也受支持:
  - Qwen 模型(参见 [PR #6121](https://github.com/sgl-project/sglang/pull/6121))

对于像 Llama 这样使用标准 GQA 的模型,通常推荐标准 DP 或 TP。

要启用 DPA,请在服务器启动命令中添加 `--enable-dp-attention`。

### 激活逻辑

DPA 通过服务器参数(CLI 或配置)显式启用。你必须同时设置 `--dp-size` 和 `--enable-dp-attention`:

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --dp-size 8 \
    --enable-dp-attention
```

**重要**:`--dp-size` 必须大于 1,DPA 才能工作。当 `dp_size == 1`(默认)时,`--enable-dp-attention` 会被自动禁用。约束条件 `tp_size % dp_size == 0` 也必须满足。

### MLA 模型的标准 DP

请注意,MLA 模型当然也支持 DP。假设你想为 MLA 模型启用标准 DP。首先,独立启动每个 MLA 模型的副本。你可以逐个启动这些启用了 DPA 的副本。在启动每个 MLA 模型的副本之后,启动一个 SMG 并将所有副本连接到该 SMG。关于 SMG 的详细解释如下。

## 现代数据并行 SGLang Model Gateway(SMG)

### 原生 DP 模式

SGLang 中的原生 DP(内置数据并行)在单个 SGLang 实例内创建多个 worker 进程,由 `DataParallelController` 控制,启动参数为 `dp-size`。


```bash
# Native DP mode
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4
```

**局限性:**

- 仅内置进程内负载均衡(例如 `round_robin`、`total_requests`、`total_tokens`)
- 没有 cache-aware 路由
- 可观测性和指标有限
- 没有容错或熔断器
- 不适合生产工作负载

⚠️ 目前 **强烈不推荐使用原生 DP**。它仅用于一些古老/过时的 RL 框架。你可以使用 SGLang Model Gateway(SMG)在任何使用场景下增强你的数据并行能力。

### 基于 SMG 的 DP(推荐)

从 2024 年 9 月开始,SGLang Model Gateway(即 SMG,前称 SGLang DP Router)专门作为一个用 Rust 构建的生产就绪 DP 路由系统而打造。它从 DP 路由起步,但后来我们进一步扩展了其范围,以协调 RL、PD Disaggregation 和其他场景。本文档仅讨论 SMG 在 DP 路由中的用法。其他用法请参阅 [SGLang Model Gateway 文档](sgl_model_gateway.md)。

> 为了实现最佳的生产级路由性能并将开销降到极致,我们使用 Rust 来构建 SMG,而非 Python,因为 Python 永远不够 FAST。

**我们强烈建议在生产级数据并行中使用 SGLang Model Gateway(SMG)。** 相比原生 DP 模式,SMG 提供了显著的优势。

```bash
# SMG-based DP mode (Recommended)
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4
```

⚠️ 请注意,**SMG 和 Naive DP 共享相同的启动参数 `--dp-size`**。但 Naive DP 的入口是 `python -m sglang.launch_server`,而 SMG 的入口是 `python -m sglang_router.launch_server`。

**基于 SMG 的 DP 的优势:**

| 特性 | 原生 DP | 基于 SMG 的 DP |
|---------|-----------|--------------|
| **负载均衡** | 内置进程内方法 | 高级策略(cache-aware、power-of-two 等) |
| **缓存感知** | ❌ 否 | ✅ 是 —— 显著更高的缓存命中率 |
| **吞吐量** | 基准 | 显著提升 |
| **多节点支持** | 有限 | ✅ 完全支持 |
| **Worker 健康监控** | 基础 | ✅ 熔断器、健康检查 |
| **可靠性** | 基础 | ✅ 重试、限流、排队 |
| **可观测性** | 基础指标 | ✅ 40+ Prometheus 指标、OpenTelemetry |
| **热添加/移除 Worker** | ❌ 否 | ✅ 是 |

###  SMG 的性能

SMG 中的 cache-aware 路由策略对于具有共享前缀的工作负载显著提升性能:

| 指标 | 不带 Cache-Aware | 带 Cache-Aware SMG |
|--------|---------------------|----------------------|
| 吞吐量(token/s) | 82,665 | 158,596 (+92%) |
| 缓存命中率 | 20% | 75% (+275%) |

*基准数据来自 [SGLang v0.4 博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/),工作负载包含多个长前缀组,8x A100 80GB GPU,dp-size=8*

### 各自的适用时机

**在以下情况使用原生 DP:**

- ~永远不要使用 原生/Naive DP~
- DP 路由的学习材料

**在以下情况使用基于 SMG 的 DP:**

- 任何情况下,当你认为需要 DP 时
- 生产部署
- 多节点分布式设置
- 具有共享前缀的工作负载(高缓存复用潜力)
- 你需要高可用性和可靠性特性
- 你需要详细的可观测性和指标
- 你想拥有高效的 RL rollout 系统

请注意,对于 RL rollout 系统,**基于 SMG 的 DP 远胜于 naive DP 路由有四个关键原因**。详情可参阅 [Load Balancing Router in RL](./sglang_for_rl.md#load-balancing-router)。

### SMG 快速上手

**安装**

```bash
pip install sglang-router
# or
pip install "sglang[all]"
```

**选项 A:同时启动 Workers 和 SMG(最简单)**

这是最简单的入门方式 —— SMG 和 workers 一起启动:

```bash
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 30000
```

**选项 B:分开启动(多节点)**

适用于跨多台机器的分布式部署:

1. 在每个节点上启动 workers

```bash
# Node 1
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000

# Node 2
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000
```

2. 启动指向 workers 的 SMG

```bash
python -m sglang_router.launch_router \
    --worker-urls http://node1:8000 http://node2:8000 \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port 30000
```

**选项 C:动态 Worker 注册**

适用于可动态添加/移除 worker 的弹性部署:

```bash
# Launch SMG first
python -m sglang_router.launch_router \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port 30000

# Register workers dynamically
curl -X POST http://localhost:30000/workers \
    -H "Content-Type: application/json" \
    -d '{"url": "http://worker1:8000"}'

curl -X POST http://localhost:30000/workers \
    -H "Content-Type: application/json" \
    -d '{"url": "http://worker2:8000"}'
```

### 负载均衡策略

SMG 支持多种负载均衡策略:

| 策略 | 描述 | 最适合 |
|--------|-------------|----------|
| `cache_aware` | 将缓存局部性与负载均衡相结合 | **大多数工作负载推荐** |
| `round_robin` | 按顺序轮流分配 worker | 简单、可预测的分发 |
| `random` | 随机选择 worker | 基准、测试 |
| `power_of_two` | 采样两个 worker,选择负载较轻的那个 | 低延迟需求 |

**Cache-Aware 策略(默认,推荐)**

cache-aware 策略为大多数工作负载提供最佳性能:

```bash
python -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware \
    --cache-threshold 0.5 \
    --balance-abs-threshold 32 \
    --balance-rel-threshold 1.5 \
    --eviction-interval-secs 120 \
    --max-tree-size 67108864
```

**工作原理:**

1. 基于请求历史,为每个 worker 维护一个近似的 radix tree
2. 将请求路由到具有最高前缀匹配(缓存命中)的 worker
3. 当负载不均衡时回退到最短队列路由
4. 自动驱逐旧条目以防止内存溢出

### 最佳实践

1. **从 `cache_aware` 策略开始** —— 它在缓存局部性和负载分布之间为大多数工作负载提供了最佳平衡
2. **生产环境使用 SMG** —— 优先选择 `sglang_router.launch_server` 而非 `sglang.launch_server`,以获得更好的可靠性和可观测性
3. **启用健康检查** —— 配置 `--router-health-check-interval-secs` 以自动检测和移除不健康的 worker

**应用了最佳实践的推荐命令:**

```bash
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4 \
    --router-policy cache_aware \
    --router-health-check-interval-secs 30 \
    --router-prometheus-port 10001 \
    --host 0.0.0.0 \
    --port 30000
```

关于高级配置(熔断器、重试、Prometheus 指标、K8s 集成),请参阅 [SGLang Model Gateway 文档](sgl_model_gateway.md)。

### 验证流量分布

启动 SMG 后,验证流量是否被正确分发:

**1. 检查 worker 状态:**

```bash
curl http://localhost:30000/workers
```

**2. 检查负载分布:**

```bash
curl http://localhost:30000/get_loads
```

**3. 监控指标(如果启用了 Prometheus):**

```bash
# Key metrics to check
smg_router_requests_total{model="..."}
smg_worker_requests_active{worker="..."}
sglang_cache_hit_rate{source="..."}
```

关于详细的指标和监控设置,请参阅 [SGLang Model Gateway 文档](sgl_model_gateway.md)。

## 参考

| 策略 | 使用场景 | 关键收益 |
|----------|----------|-------------|
| **原生 DP**(`--dp-size`) | 永不 | 易于理解,非基于 rust |
| **基于 SMG 的 DP** | **生产(推荐)** | Cache-aware 路由、高可用性 |
| **DPA**(`--dp-size N --enable-dp-attention`) | DeepSeek/MLA 模型 | 消除 KV cache 重复,提升吞吐量 |
| **DPA + EP** | DeepSeek MoE 模型 | 相比原生 TP 显著提升吞吐量 |

**DeepSeek 的推荐生产设置:**
1. 为注意力层启用 **DPA**(`--dp-size 8 --enable-dp-attention`)
2. 为 MoE 层启用 **EP**(`--ep 8 --moe-a2a-backend deepep`)
3. 使用带 **cache_aware** 策略的 **SMG**

**相关文档:**
- [Expert Parallelism](expert_parallelism.md) - DeepEP、Two-Batch Overlap、EPLB
- [SGLang Model Gateway 文档](sgl_model_gateway.md) - SMG 配置与故障排查
- [Large-Scale EP Blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/) - 96 GPU 部署指南
