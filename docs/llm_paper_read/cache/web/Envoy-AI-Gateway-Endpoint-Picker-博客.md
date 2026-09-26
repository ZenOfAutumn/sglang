> 来源: <https://aigateway.envoyproxy.io/blog/endpoint-picker-for-inference-routing/>
> 抓取时间: 2026-09-01 ｜ 格式: 网页转 Markdown

# Envoy AI Gateway 推出 Endpoint Picker 支持

2025 年 7 月 30 日 · 阅读约 7 分钟

[](/blog/authors/missberg)

[Erica Hughberg](/blog/authors/missberg)

Envoy AI Gateway 维护者 - Tetrate

[](https://github.com/missBerg "GitHub")[](https://www.linkedin.com/in/ericahughberg/ "LinkedIn")

[](/blog/authors/xunzhuo)

[Xunzhuo (Bit) Liu](/blog/authors/xunzhuo)

Envoy AI Gateway 维护者 - AMD

[](https://github.com/Xunzhuo "GitHub")[](https://www.linkedin.com/in/bitliu/ "LinkedIn")

## 引言​

**Envoy AI Gateway 现已按照** [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/) **支持 Endpoint Picker Provider（EPP）集成。**

该特性使你能够为 AI 推理工作负载利用智能、动态的路由，基于实时指标（包括 KV-cache 使用率、排队请求数以及 LoRA 适配器信息）进行智能端点选择。

在大规模运行 AI 推理时，这意味着你的系统可以为每个请求自动选择最优的推理端点，从而优化资源利用率。

## 问题：传统负载均衡难以胜任 AI 工作负载​

由于 LLM 推理工作负载相比传统 API 服务器具有一组全新的特征，要让负载均衡器知道哪个上游目标最适合处理该请求，就需要一组新的信息来做出这些决策。

AI 推理工作负载的一些独有特征：

  * **处理时间可变**，取决于模型复杂度和输入大小
  * **资源需求各异**，不同模型和配置的需求不同
  * **实时性能指标**持续变化（KV-cache 使用率、队列深度等）
  * **专用硬件需求**（GPU、TPU、不同的模型变体）

如果在路由时不考虑这些因素，你最终会遇到：

  * 一些端点过载而另一些闲置
  * 因端点选择不佳导致延迟增加
  * 负载分配不合理造成资源浪费
  * 需要人工干预才能获得最佳性能

## 解决方案：Endpoint Picker Provider 集成​

Envoy AI Gateway 新的 EPP 支持通过为用户提供集成 Endpoint Picker 的选项来应对这些挑战。Endpoint Picker 会将这些信息纳入考量，并告知网关应将请求路由到哪个最佳上游目标。

### 1\. 智能端点选择​

Endpoint Picker（EPP）通过分析来自推理端点的实时指标，自动将请求路由到最合适的后端，这些指标例如：

  * 当前 KV-cache 使用率
  * 排队中的推理请求数量
  * LoRA 适配器信息
  * 端点健康状况与性能指标

### 2\. 动态负载均衡​

与静态负载均衡器不同，EPP 持续接收端点状态，从而实时影响路由决策。这种动态决策确保了整个推理基础设施的资源利用率达到最优。

### 3\. 自动故障转移​

当某个端点不可用或性能下降时，EPP 会自动将流量路由到健康的备选端点，为你的 AI 服务确保高可用性。

### 4\. 可扩展架构​

EPP 架构支持自定义的 endpoint picker provider，允许你实现针对自身独特需求量身定制的领域特定路由逻辑。

## 工作原理：两种集成方式​

Envoy AI Gateway 通过两种强大的方式支持 EPP 集成：

### HTTPRoute + InferencePool​

对于简单的推理工作负载，你可以使用标准的 Gateway API HTTPRoute 搭配 InferencePool。这种方式提供基础的智能路由、自动负载均衡和简明的配置，非常适合简单的使用场景。
[code]
    apiVersion: gateway.networking.k8s.io/v1


    kind: HTTPRoute


    metadata:


      name: inference-pool-with-httproute


      namespace: default


    spec:


      parentRefs:


        - group: gateway.networking.k8s.io


          kind: Gateway


          name: inference-pool-with-httproute


          namespace: default


      rules:


        - backendRefs:


            - group: inference.networking.k8s.io


              kind: InferencePool


              name: vllm-llama3-8b-instruct


              namespace: default


              weight: 1


          matches:


            - path:


                type: PathPrefix


                value: /


          timeouts:


            request: 60s

[/code]

这种方式提供：

  * 基础的智能路由
  * 自动负载均衡
  * 简单的配置

### AIGatewayRoute + InferencePool​

若需要高级的 AI 专属特性，请使用 Envoy AI Gateway 的自定义 AIGatewayRoute。
[code]
    apiVersion: aigateway.envoyproxy.io/v1alpha1


    kind: AIGatewayRoute


    metadata:


      name: inference-pool-with-aigwroute


      namespace: default


    spec:


      targetRefs:


        - name: inference-pool-with-aigwroute


          kind: Gateway


          group: gateway.networking.k8s.io


      rules:


        - matches:


            - headers:


                - type: Exact


                  name: x-ai-eg-model


                  value: meta-llama/Llama-3.1-8B-Instruct


          backendRefs:


            - group: inference.networking.k8s.io


              kind: InferencePool


              name: vllm-llama3-8b-instruct


        - matches:


            - headers:


                - type: Exact


                  name: x-ai-eg-model


                  value: mistral:latest


          backendRefs:


            - group: inference.networking.k8s.io


              kind: InferencePool


              name: mistral


        - matches:


            - headers:


                - type: Exact


                  name: x-ai-eg-model


                  value: some-cool-self-hosted-model


          backendRefs:


            - name: envoy-ai-gateway-basic-testupstream

[/code]

这种增强方式额外提供：

  * 基于请求体 modelName 的**多模型路由**
  * 面向自托管模型成本控制的**基于 token 的限流**
  * 覆盖各 LLM 模型的**高级 LLM 可观测性**

## 实际收益​

### 对 AI/ML 工程师​

  * **更低的延迟**：请求被自动路由到最优端点
  * **更高的吞吐量**：智能分配防止出现瓶颈
  * **成本优化**：高效的资源使用降低基础设施成本
  * **增强的可观测性**：实时指标与性能洞察

### 对平台团队​

  * **符合标准**：构建于 Gateway API Inference Extension 之上
  * **供应商灵活性**：支持多种 EPP 实现
  * **面向未来的架构**：可扩展设计以应对不断演进的需求
  * **Kubernetes 原生**：与现有基础设施无缝集成

### 对 DevOps 团队​

  * **降低运维开销**：不再需要手动管理端点
  * **提升可靠性**：自动故障转移与健康监控
  * **更好的资源利用率**：动态负载均衡最大化效率
  * **简化扩容**：新增端点无需变更配置

## 快速上手​

在 Envoy AI Gateway 中启用 EPP 支持非常简单：

### 1\. 安装 InferencePool CRD​
[code]
    # 安装 Gateway API Inference Extension


    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v0.5.1/manifests.yaml

[/code]

### 2\. 部署你的推理后端​
[code]
    # 部署示例 vLLM 后端


    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/v0.5.1/config/manifests/vllm/sim-deployment.yaml

[/code]

### 3\. 配置 InferenceObjective 和 InferencePool​
[code]
    apiVersion: inference.networking.k8s.io/v1


    kind: InferenceObjective


    metadata:


      name: base-model


    spec:


      modelName: meta-llama/Llama-3.1-8B-Instruct


      criticality: Critical


      poolRef:


        name: vllm-llama3-8b-instruct


    ---


    apiVersion: inference.networking.k8s.io/v1


    kind: InferencePool


    metadata:


      name: vllm-llama3-8b-instruct


    spec:


      targetPortNumber: 8000


      selector:


        app: vllm-llama3-8b-instruct


      extensionRef:


        name: vllm-llama3-8b-instruct-epp

[/code]

### 4\. 创建你的路由​

根据你的需求在 HTTPRoute 和 AIGatewayRoute 之间选择其一，然后就可以开始使用了！

## 下一步计划​

EPP 支持的引入是 Envoy AI Gateway 演进过程中的一个重要里程碑。该特性为 AI 工作负载启用了智能路由，使部署和管理生产级 AI 推理服务变得前所未有的简单。

### 即将推出的增强特性​

  * **上游一致性测试**：与 **Gateway API Inference Extension** 一致性测试集成。
  * **回退支持**：该特性基于 Envoy 的 HostOverride LbPolicy，在 Endpoint Picker 不可用时提供端点选择回退支持。即使 Endpoint Picker 发生故障，也能确保服务持续可用。
  * **内部托管 EPP**：这一即将推出的特性将提供由 Envoy AI Gateway 托管的 Endpoint Picker，简化你对 AI 推理端点的管理。
  * **增强端到端（E2E）测试**：为 InferencePool 支持添加更多 E2E 测试。

### 社区贡献​

我们非常期待看到社区如何利用这一能力。无论你是在构建自定义的 EPP 实现、为 Gateway API Inference Extension 做贡献，还是分享你的部署经验，我们都很乐意听到你的声音。

## 结语​

Endpoint Picker Provider 支持使 Envoy AI Gateway 不仅可以作为面向出口（Egress）的 AI Gateway，还可以作为面向 AI 推理工作负载的入口（Ingress）AI Gateway。通过基于实时指标自动选择最优端点，该特性为自托管推理系统提升了性能并最大化了资源利用率。

无论你运行的是一个小型 AI 服务还是大规模推理平台，EPP 支持都能提供你所需的智能路由能力，为你的用户交付可靠、高性能的 AI 服务。

准备好开始使用 Envoy AI Gateway 了吗？查看我们的[文档](https://aigateway.envoyproxy.io/docs/capabilities/inference/)获取指南和示例，加入我们的[社区讨论](https://github.com/envoyproxy/ai-gateway/discussions)来分享你的经验、提出问题、请求新特性并向他人学习。

* * *

_Envoy AI Gateway 正持续演进，致力于成为路由和管理 AI 工作负载的首选解决方案。我们将不断简化、增强并提升 AI 部署的可靠性与效率，敬请期待更多令人兴奋的特性和能力。_

## 资源​

  * [推理优化文档](/docs/capabilities/inference/)
  * [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)
  * [AIGatewayRoute + InferencePool 指南](/docs/capabilities/inference/aigatewayroute-inferencepool/)
  * [HTTPRoute + InferencePool 指南](/docs/capabilities/inference/httproute-inferencepool/)
  * [GitHub 仓库](https://github.com/envoyproxy/ai-gateway)

**标签：**

  * [新闻](/blog/tags/news "项目新闻")
  * [特性](/blog/tags/features "关于 Envoy AI Gateway 特性的文章")
