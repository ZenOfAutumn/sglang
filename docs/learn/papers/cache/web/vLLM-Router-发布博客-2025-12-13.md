> 来源: <https://vllm-project.github.io/2025/12/13/vllm-router-release.html>
> 抓取时间: 2026-09-01 ｜ 格式: 网页转 Markdown
> 所属项目: <https://github.com/vllm-project/router>

# vLLM Router：面向大规模服务的高性能、预填充/解码感知负载均衡器

2025 年 12 月 13 日 • vLLM 团队

在大规模生产级 vLLM 部署中，高效管理模型副本集群间的请求分发至关重要。标准负载均衡器通常无法满足要求，因为它们不了解 LLM 推理的有状态特性（例如 KV 缓存），也无法管理预填充/解码分离等复杂的服务模式。

为了解决这一问题，我们推出了 **vLLM Router**（[GitHub 仓库](https://github.com/vllm-project/router)）。它是一款专为 vLLM 设计的高性能轻量级负载均衡器。Router 使用 Rust 构建，将额外开销降至最低；它位于客户端与 vLLM 工作节点集群之间，充当智能且具备状态感知能力的负载均衡器，既可部署在 K8s 集群中，也可部署在裸金属 GPU 集群中。

vLLM Router 衍生自 [SGLang Model Gateway](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway) 的一个分支，并经过修改和简化以适配 vLLM。随着我们探索将该 Router 合并进 vLLM 主仓库，预计两者未来会进一步分化。另一方面，我们也可能与 SGLang Model Gateway 的开发者合作，统一面向大规模部署的网关功能。

## 核心架构与能力

vLLM Router 旨在解决大规模服务中的两个主要挑战：智能负载均衡，以及对预填充/解码分离的支持。

### 1. 智能负载均衡策略

与简单的轮询不同，vLLM Router 提供多种先进的负载均衡算法，以优化性能并保持有状态亲和性。对于对话类工作负载，将同一用户的后续请求路由到保存其 KV 缓存的同一工作节点，对于降低延迟至关重要。

为此，Router 支持以下几种策略：

* **一致性哈希：** 这是实现性能最大化的关键策略。它确保具有相同路由键（例如会话 ID 或用户 ID）的请求具备“粘性”，始终被路由到同一工作节点副本，从而最大限度复用 KV 缓存。
* **二选一（Power of Two，PoT）：** 一种开销较低的随机选择策略，可实现出色的负载分布。
* **轮询与随机：** 用于无状态负载分发的标准策略。

### 2. 原生支持预填充/解码分离

Router 被设计为 vLLM 最先进服务架构——预填充/解码（P/D）分离——的编排层。

在这一架构中，计算密集型的预填充阶段和内存密集型的解码阶段分别由独立的专用工作节点组负责。vLLM Router 管理这一复杂工作流：

1. 将新请求智能路由到预填充工作节点组。
2. 预填充完成后，将请求状态传递给适当的解码工作节点以生成 Token。
3. 支持发现和路由 **NIXL** 与**基于 NCCL（使用 ZMQ 发现）**的分离后端。

## 企业级韧性与可观测性

vLLM Router 具备用于生产环境的功能，可在大规模环境中维持高可用性。

* **Kubernetes 服务发现：** Router 可以 Kubernetes 原生模式运行，通过标签选择器自动发现和监控 vLLM 工作节点 Pod，并将请求路由至这些 Pod。
* **容错能力：** 内置可配置的**重试逻辑**（支持指数退避和抖动）与**熔断器**。如果某个工作节点未通过健康检查，Router 会立即将其从路由池中移除并重试请求，从而防止级联故障。
* **可观测性：** 内置 Prometheus 端点（`/metrics`），可导出请求量、延迟、错误率及各工作节点健康状况等详细指标，为整个服务集群提供完整的可观测能力。

## 基准测试分析：大规模场景下性能最佳的选择

我们将新的 vLLM Router 与以下两种广泛使用的方案进行了基准测试对比：

* **[llm-d](https://github.com/llm-d/llm-d)：** 一个 Kubernetes 原生路由框架，默认采用队列感知负载均衡。
* **vLLM 原生方案：** 标准的 [K8s 原生负载均衡器](https://kubernetes.io/docs/concepts/services-networking/)，采用基本的轮询策略。关键在于，该方案无法感知预填充/解码状态，而是将所有 Pod 都视为相同的 vLLM 副本。

**关于排除项的说明：** 我们未将 vLLM 内置的 DP/EP 协调器纳入基准测试。该协调器是 vLLM 集群推荐的[外部负载均衡](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment.html#external-load-balancing)方案，但由于一个已知的[性能问题](https://github.com/vllm-project/vllm/issues/24461)，其吞吐量仅为其他方案的八分之一。

### Llama 3.1 8B：8 个预填充 Pod 和 8 个解码 Pod

* vLLM Router（蓝线）的每秒请求数（Req/S）吞吐量比 llm-d（紫线）高 25%，比 K8s 原生负载均衡器（橙线）高 100%。
* vLLM Router 的首 Token 延迟（TTFT）接近 K8s 原生负载均衡器，比 llm-d 快 1200 ms。

### DeepSeek V3：1 个预填充 Pod（TP8）和 1 个解码 Pod（TP8）

* vLLM Router（蓝线）的每秒请求数（Req/S）吞吐量接近 llm-d（紫线），比 K8s 原生负载均衡器（橙线）高 100%。
* vLLM Router 的首 Token 延迟（TTFT）比 llm-d 和 K8s 原生方案快 2000 ms。

## 总结

vLLM Router 是在生产规模下运行 vLLM 的关键组件。它将服务架构从一组彼此独立的实例，转变为一个统一且具备韧性的集群。通过提供智能负载均衡并原生支持预填充/解码分离，它实现了更高水平的性能和运维效率。

## 致谢

* 感谢 Phi 和 AWS 团队提供技术支持与测试集群。
* 特别感谢 Naman Lalit 推动全面的性能与正确性基准测试工作。
* 同时感谢 SGLang Model Gateway 团队。通过复用其成熟的 API 实现和服务框架，我们得以在保持与开放标准一致的同时，显著加快设计与实现进程。
* 最后，感谢 Tyler Michael Smith 和 Robert Shaw 分享 llm-d 方面的专业经验及实测资料，为性能优化和基准测试扫清了障碍。

