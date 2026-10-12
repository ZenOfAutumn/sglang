> 来源: <https://www.kubeai.org/blog/2025/02/26/llm-load-balancing-at-scale-chwbl/>
> 抓取时间: 2026-09-01 ｜ 格式: 网页转 Markdown

# 大规模 LLM 负载均衡：带负载上界的一致性哈希¶

本研究由 KubeAI 的创建者 [Substratus.AI](https://www.substratus.ai/) 开展。

## 摘要：¶

在真实场景中大规模运行大语言模型（LLM）需要多个后端副本。副本间的负载均衡算法会影响现代推理引擎固有的缓存动态，从而极大地影响整体系统性能。在本研究中，我们探索了带负载上界的一致性哈希（Consistent Hashing with Bounded Loads，CHWBL）算法在 LLM 推理领域的应用。我们在业界应用最广泛的部署平台 Kubernetes 上模拟了真实世界的负载模式。结果表明，与基线策略相比，首 Token 时间（TTFT）_降低了 95%_，整体吞吐量_提升了 127%_。

## 1\. 引言¶

在 vLLM 等推理引擎开始生成输出 token 之前，必须先在"预填充阶段（prefill phase）"处理输入提示词。所产生的中间表示会存储在键值（KV）缓存中。这个缓存可以在共享相同前缀的多个请求之间复用——这一特性被称为"前缀缓存（prefix caching）"。在多个真实场景中，前缀缓存的影响可能非常显著：

**多轮对话：** 在聊天机器人（如 ChatGPT）或自主 AI 智能体等应用中，每一轮新对话都会在不断演进的共享上下文上追加内容。

**共享上下文的多线程请求：** 针对同一篇长文档发起多次查询的场景对前缀缓存的效率尤其敏感。当请求并发执行时，即使缓存命中率的微小提升也能显著降低端到端延迟。

Kubernetes 提供的传统随机路由策略往往导致缓存利用率不佳，造成频繁的缓存淘汰和性能下降。

## 2\. 问题陈述¶

一个有效的 LLM 服务负载均衡策略应满足以下标准：

**最大化缓存利用率：** 将具有共同前缀的请求路由到缓存处于热状态的 vLLM 副本。

**适应副本变化：** 在副本增减时尽量减少缓存重排。

**避免热点：** 防止任何单个副本过载。

**考虑 LoRA 适配器：** 不同适配器的前缀缓存应被视为彼此独立。

**避免客户端侧要求：** 常见的客户端库应能开箱即用。

## 3\. 粘性会话？¶

跨有状态后端的负载均衡并不是一个新问题。我们考虑过粘性会话（sticky sessions）——有状态负载均衡的首选方案。在类似 ChatGPT 的场景中，可以用 Cookie 将用户可靠地映射到活跃会话线程。然而许多智能体系统并不基于浏览器，这使得客户端 IP 成为开箱即用的主要客户端标识来源。而客户端 IP 通常不足以可靠地确定终端客户端（NAT、动态 IP）。更糟的是，单个智能体程序通常会模拟 N 个逻辑智能体——即多个并发的多轮对话。

## 4\. 算法¶

带负载上界的一致性哈希（CHWBL）算法天然地解决了上述挑战，使其成为 LLM 负载均衡的一个有吸引力的选择。它在视频分发等缓存敏感场景的大规模应用中有着成熟的实践记录。

CHWBL 算法在传统一致性哈希的基础上引入了负载上界，确保任何单个副本收到的请求不会超过其公平份额。这种方法不仅保持了缓存亲和性，还防止任何单台服务器过载，从而优化整体系统性能。

## 5\. 实现¶

我们将 CHWBL 路由策略以 PrefixHash 配置的形式集成到了 [KubeAI 项目](https://github.com/kubeai-project/kubeai)中。该策略的工作流程如下：

  1. **请求检查：** 分析传入的 HTTP 请求体（OpenAI 格式的 API 请求）。
  2. **前缀提取：** 从请求中提取一个前缀（例如，在 chat completions 中使用第一条用户消息），并提取所请求的 LoRA 适配器（如适用）。
  3. **哈希：** 使用 `xxHash` 算法对 `(extracted_prefix + lora_adapter_name)` 进行哈希。
  4. **副本查找：** 使用 CHWBL 算法根据哈希值选择合适的 vLLM 副本。该配置通过以下 YAML 片段按模型逐一指定：

[code]
    kind: Model
    spec:
      # ...
      loadBalancing:
        strategy: PrefixHash
        # 可选参数：
        prefixHash:
          meanLoadFactor: 125
          prefixCharLength: 100
          replication: 256

[/code]

## 6\. 评估¶

### 6.1. 场景¶

我们使用三种不同的负载均衡策略进行了实验：

  1. **Kubernetes Service（随机）**
     * 使用默认的 Kubernetes Service，绕过 KubeAI 代理。
     * 依赖 kube-proxy 通过 `iptables` 进行代理。
  2. **KubeAI（LeastLoad）**
     * 使用 KubeAI 负载均衡器将流量路由到活跃请求数最少的副本。
  3. **KubeAI（PrefixHash）**
     * 应用由 CHWBL 驱动的策略，通过 KubeAI 代理路由请求。

### 6.2. 环境搭建¶

搭建了一个运行 KubeAI 的 Kubernetes 集群，规格如下：

  * **硬件：** 8x L4 GPU
  * **软件：** 8x vLLM 实例
  * **模型：** Llama 3.1 8B
  * **数据集：** 源自 ShareGPT 的消息会话线程
  * **工作负载：** 自定义负载生成器，模拟并行的 chat completion 会话线程，通过循环追加 LLM 响应来保持对话状态。

### 6.3. 指标¶

我们的评估聚焦于两个关键性能指标：

  * **首 Token 时间（TTFT）：** 生成第一个 token 之前的延迟。
  * **每秒 Token 数（TPS）：** token 生成的整体吞吐量。

### 6.4. 结果¶

基准测试表明，与默认 Kubernetes Service 和 LeastLoad 策略相比，PrefixHash 策略在 TTFT 和吞吐量方面都带来了显著改善。

**TTFT：** 当系统运行在 1200 个并发线程（每副本 150 个线程）时，PrefixHash 策略相比 Kubernetes 内置策略使_首 Token 时间降低了 95%_。

**吞吐量：** 当系统运行在 1200 个并发线程（每副本 150 个线程）时，PrefixHash 策略相比 Kubernetes 内置策略使_吞吐量提升了 127%_。

## 7\. 结论¶

在这项工作中，我们证明了通过 KubeAI 中的 PrefixHash 策略将 CHWBL 算法引入 LLM 负载均衡，可以在真实场景中显著提升性能。值得注意的是，缓存命中率的提升带来了更低的延迟和更高的吞吐量。展望未来，我们计划扩展基准测试以覆盖更广泛的工作负载。我们还计划在新兴策略出现时将其纳入基准测试，最终目标是确保 KubeAI 具备最有效的负载均衡技术。

## 参考文献¶

  * [vLLM 自动前缀缓存文档](https://docs.vllm.ai/en/latest/design/prefix_caching/)
  * [Google Research：带负载上界的一致性哈希](https://research.google/blog/consistent-hashing-with-bounded-loads/)
  * [KubeAI 负载均衡概念](https://www.kubeai.org/concepts/load-balancing/)
