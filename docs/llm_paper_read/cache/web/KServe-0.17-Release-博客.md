> 来源: <https://kserve.github.io/website/blog/kserve-0.17-release>
> 抓取时间: 2026-09-01 ｜ 格式: 网页转 Markdown
> 综述原链接 kserve-0-17-release(连字符)为笔误,正确为 kserve-0.17-release

# KServe v0.17 发布公告 —— 借助 LLMInferenceService 实现生产级 LLM 服务

2026 年 3 月 13 日 · 阅读时长 14 分钟

[](https://github.com/yuzisun)

[Dan Sun](https://github.com/yuzisun)

KServe 联合创始人

[](https://github.com/yuzisun "GitHub")

 _发布于 2026 年 3 月 13 日_

我们非常高兴地宣布 **KServe v0.17** 正式发布，这是一个里程碑式的版本，它将 **LLMInferenceService** 推向生产就绪状态，并带来了基于 [llm-d](https://github.com/llm-d/llm-d) 框架构建的 GenAI 优先架构。本版本引入了 KV-cache 感知的智能路由、Prefill-Decode 分离、支持张量/数据/专家并行的分布式推理、集成基于 Token 限流的 Envoy AI Gateway，以及完全重构的模块化 Helm chart 架构。

## 🤖 LLMInferenceService：GenAI 优先架构​

KServe v0.17 将 **LLMInferenceService** 从实验特性提升为生产就绪的 CRD，专为生成式 AI 工作负载而构建。LLMInferenceService 基于 [llm-d](https://github.com/llm-d/llm-d) 框架构建，提供了一种 GenAI 优先的架构，超越传统的 InferenceService，以应对大规模服务大语言模型所面临的独特挑战。

与面向预测式 AI 工作负载设计的 InferenceService 不同，LLMInferenceService 原生支持：

  * **分布式推理**：跨多节点和多 GPU
  * **KV-cache 感知调度**：实现智能请求路由
  * **Prefill-Decode 分离**：实现最优资源利用
  * **Gateway Inference Extension (GIE) 集成**：提供高级流量管理
  * **基于 Token 的限流**：通过 Envoy AI Gateway 实现

特性| InferenceService| LLMInferenceService
---|---|---
**主要使用场景**|  预测式 AI| 生成式 AI
**路由**|  标准 Gateway| 基于 EPP 的 KV-cache 感知路由
**并行方式**|  Worker Spec| 原生支持 TP、DP、EP
**Prefill-Decode**|  不适用| 分离式架构
**扩缩容**|  HPA/KPA| WVA + KEDA
[code]
    apiVersion: serving.kserve.io/v1alpha2
    kind: LLMInferenceService
    metadata:
      name: llama3-serving
    spec:
      model:
        uri: hf://meta-llama/Llama-3.1-8B-Instruct
        name: meta-llama--Llama-3.1-8B-Instruct
      replicas: 3
      template:
        spec:
          containers:
            - name: vllm
              resources:
                limits:
                  nvidia.com/gpu: "1"
      router:
        gateway:
          managed: {}
        route:
          httpRoute: {}
        scheduler:
          pool: {}

[/code]

这将创建一套完整的服务栈，包括 Deployment、Service、Gateway、HTTPRoute、**InferencePool**、**InferenceModel** 以及 **EPP（Endpoint Picker Pod）** —— 全部由 LLMInferenceService 控制器统一管理。

## 🚀 v0.17 中 LLMInferenceService 的关键特性​

### 🧠 基于 Gateway Inference Extension 的 KV-Cache 感知调度​

LLMInferenceService 集成了 [**Gateway Inference Extension (GIE) v1.3.0**](https://gateway-api-inference-extension.sigs.k8s.io/)，这是一个 Kubernetes SIG 项目，为 Gateway API 扩展了面向 AI 的路由能力。此集成的核心是来自 [llm-d inference scheduler](https://github.com/llm-d/llm-d-inference-scheduler) 的 **Endpoint Picker Pod（EPP）**，这是一个智能调度器，它基于实时 KV-cache 状态而非简单的轮询或随机负载均衡来路由请求。

传统负载均衡对所有 LLM 推理请求一视同仁，但在实践中，具有相似提示词的请求如果被路由到同一个 Pod，将会带来巨大收益 —— 因为该 Pod 已经加载了相关的 KV cache 块。EPP 通过 ZMQ 事件（`BlockStored`、`BlockRemoved`）跟踪所有 vLLM 实例的实时 KV cache 状态，并构建从 `{ModelName, BlockHash}` 到 `{PodID, DeviceTier}` 的索引映射，从而解决了这一问题。

调度行为通过 `EndpointPickerConfig` 进行配置，它定义了一个带加权评分器的插件流水线：
[code]
    apiVersion: inference.networking.x-k8s.io/v1alpha1
    kind: EndpointPickerConfig
    plugins:
      - type: single-profile-handler
      - type: prefix-cache-scorer
      - type: load-aware-scorer
        parameters:
          threshold: 100
      - type: max-score-picker
    schedulingProfiles:
      - name: default
        plugins:
          - pluginRef: prefix-cache-scorer
            weight: 2.0
          - pluginRef: load-aware-scorer
            weight: 1.0
          - pluginRef: max-score-picker

[/code]

该流水线使用三类插件（详情参见 [llm-d 调度器架构](https://github.com/llm-d/llm-d-inference-scheduler/blob/main/docs/architecture.md)）：

  * **prefix-cache-scorer**（权重：2.0）：跟踪所有 vLLM 实例中的实际 KV cache 内容，并根据缓存的前缀块与传入请求提示词的匹配数量对 Pod 进行评分。这通过避免对重复或相似提示词进行冗余 prefill 计算来降低首 Token 时间（TTFT）—— 对多轮对话和 RAG 工作负载尤其有益。
  * **load-aware-scorer**（权重：1.0）：根据候选 Pod 当前的队列深度进行评分。队列为空的 Pod 得分为 0.5，而队列不断增长的 Pod 得分则逐步向 0 递减。`threshold` 参数控制敏感度 —— 当队列深度超过阈值时，该 Pod 得分接近于零。
  * **max-score-picker**：在所有评分器运行完毕后，选择加权聚合得分最高的 Pod。

`EndpointPickerConfig` 既可以内联写在 LLMInferenceService spec 中，也可以引用某个 ConfigMap，这使平台团队能够灵活地在多个部署之间标准化调度行为：
[code]
    apiVersion: serving.kserve.io/v1alpha2
    kind: LLMInferenceService
    metadata:
      name: llama3-with-scheduler
    spec:
      model:
        uri: hf://meta-llama/Llama-3.1-8B-Instruct
        name: meta-llama--Llama-3.1-8B-Instruct
      replicas: 4
      template:
        spec:
          containers:
            - name: vllm
              resources:
                limits:
                  nvidia.com/gpu: "1"
      router:
        gateway:
          managed: {}
        route:
          httpRoute: {}
        scheduler:
          config:
            ref:
              name: custom-endpoint-picker-config
              key: endpoint-picker-config.yaml
          pool: {}

[/code]

GIE 的 CRD（**InferencePool** 和 **InferenceModel**）现已作为 KServe 安装包的一部分捆绑提供，简化了安装配置。

### 🔀 Prefill-Decode 分离​

LLMInferenceService 原生支持 **Prefill-Decode 分离**，即把计算密集型的 prefill 阶段和内存密集型的 decode 阶段拆分为相互独立的工作负载。这使得每个阶段都可以独立地扩缩容和优化。
[code]
    apiVersion: serving.kserve.io/v1alpha2
    kind: LLMInferenceService
    metadata:
      name: llama3-prefill-decode
    spec:
      model:
        uri: hf://meta-llama/Llama-3.1-8B-Instruct
        name: meta-llama--Llama-3.1-8B-Instruct
      replicas: 2
      template:
        spec:
          containers:
            - name: vllm
              resources:
                limits:
                  nvidia.com/gpu: "1"
      prefill:
        replicas: 2
        template:
          spec:
            containers:
              - name: vllm
                resources:
                  limits:
                    nvidia.com/gpu: "1"
      router:
        gateway:
          managed: {}
        route:
          httpRoute: {}
        scheduler:
          pool: {}

[/code]

KV cache 数据通过 **NixlConnector** 在 prefill 与 decode Pod 之间传输，采用基于 RDMA 的 RoCE 实现高吞吐、低延迟的块传输。

### 📐 分布式推理：张量、数据与专家并行​

LLMInferenceService 引入了一套完善的**并行规范（parallelism specification）**，借助 [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) 实现跨多节点、多 GPU 的分布式推理：

  * **张量并行（Tensor Parallelism, TP）**：将模型层切分到节点内的多个 GPU 上
  * **数据并行（Data Parallelism, DP）**：运行多个模型副本以提升吞吐量
  * **数据本地并行（Data-Local Parallelism）**：控制每个节点的 GPU 数量，以获得最优的 NUMA 亲和性
  * **专家并行（Expert Parallelism, EP）**：将混合专家（MoE）模型的专家分布到多个 GPU 上

[code]
    apiVersion: serving.kserve.io/v1alpha2
    kind: LLMInferenceService
    metadata:
      name: llama3-multi-node
    spec:
      model:
        uri: hf://meta-llama/Llama-3.1-70B-Instruct
        name: meta-llama--Llama-3.1-70B-Instruct
      replicas: 8
      parallelism:
        tensor: 4
        data: 8
        dataLocal: 4
      template:
        spec:
          containers:
            - name: vllm
              resources:
                limits:
                  nvidia.com/gpu: "4"
      worker:
        spec:
          containers:
            - name: vllm
              resources:
                limits:
                  nvidia.com/gpu: "4"
      router:
        gateway:
          managed: {}
        route:
          httpRoute: {}
        scheduler:
          pool: {}

[/code]

### 🌐 集成 Envoy AI Gateway 实现基于 Token 的限流​

LLMInferenceService 集成了 [**Envoy AI Gateway**](https://aigateway.envoyproxy.io)，提供面向 AI 的原生流量管理。这使得**基于 Token 的限流**成为可能 —— 对于 LLM 服务而言，这项能力至关重要，因为请求成本会随输入和输出 Token 数量的不同而剧烈变化，而不是简单地取决于请求次数。
[code]
    apiVersion: aigateway.envoyproxy.io/v1alpha1
    kind: AIGatewayRoute
    metadata:
      name: llm-route
    spec:
      targetRefs:
        - group: gateway.networking.k8s.io
          kind: HTTPRoute
          name: llama3-serving
      llmRequestCosts:
        - metadataKey: llm_input_token
          type: InputToken
        - metadataKey: llm_output_token
          type: OutputToken
        - metadataKey: llm_total_token
          type: TotalToken
    ---
    apiVersion: gateway.envoyproxy.io/v1alpha1
    kind: BackendTrafficPolicy
    metadata:
      name: llm-rate-limit
    spec:
      targetRefs:
        - group: aigateway.envoyproxy.io
          kind: AIGatewayRoute
          name: llm-route
      rateLimit:
        type: Global
        global:
          rules:
            - clientSelectors:
                - headers:
                    - name: x-user-id
                      type: Distinct
              limit:
                requests: 1000
                unit: Hour
              cost:
                request:
                  from: Number
                  number: 0
                response:
                  from: Metadata
                  key: llm_total_token

[/code]

### ⚡ 支持 WVA 的自动扩缩容 API​

LLMInferenceService 新增了一个 **autoscaling API**，支持 [**Workload Variant Autoscaler（WVA）**](https://github.com/llm-d/llm-d-workload-variant-autoscaler)，这是一个专为 LLM 推理工作负载设计的基于 Kubernetes 的全局自动扩缩容器。传统的基于 CPU/内存的自动扩缩容对 LLM 来说并不适用，因为推理成本是由 Token 吞吐量、KV cache 利用率和队列深度驱动的，而非 CPU 或内存使用率。

WVA 通过 Prometheus 持续监控推理服务器的指标 —— 具体而言是 **KV cache 利用率**和**队列深度** —— 以判断服务器是否正接近饱和。随后它会计算出 `wva_desired_replicas` 指标并写入 Prometheus，由执行后端（**HPA** 或 **KEDA**）读取该指标来驱动实际的扩缩容：

  * **WVA + KEDA**：直接从 Prometheus 查询 `wva_desired_replicas` 指标。不需要 Prometheus Adapter。通过 `idleReplicaCount` 支持空闲时缩容到零。
  * **WVA + HPA**：通过 Kubernetes Metrics API 读取 `wva_desired_replicas` 指标。需要 Prometheus Adapter。支持标准的 HPA 扩缩容行为。

WVA 中的一个关键概念是 **variant（变体）** —— 即服务某个模型的一种特定部署配置（硬件、运行时、并行策略）。同一个基础模型可能由多个变体来提供服务：例如，在 A100 GPU 上以 TP=4 运行的 Llama-3 是一个变体，而在 H100 GPU 上以 TP=2 运行的 Llama-3 则是另一个变体。`variantCost` 字段指定了每个变体单副本的相对成本，使 WVA 能够跨变体做出**成本感知的扩缩容决策** —— 需求增加时优先扩容更便宜的变体，需求下降时优先缩容最昂贵的变体。
[code]
    apiVersion: serving.kserve.io/v1alpha2
    kind: LLMInferenceService
    metadata:
      name: llama3-wva-autoscaling
    spec:
      model:
        uri: hf://meta-llama/Llama-3.1-8B-Instruct
        name: meta-llama--Llama-3.1-8B-Instruct
      scaling:
        minReplicas: 1
        maxReplicas: 10
        wva:
          variantCost: "15.0"
          keda:
            pollingInterval: 30
            cooldownPeriod: 300
            initialCooldownPeriod: 120
            idleReplicaCount: 0
            fallback:
              failureThreshold: 3
              replicas: 2
      template:
        spec:
          containers:
            - name: vllm
              resources:
                limits:
                  nvidia.com/gpu: "1"
      router:
        gateway:
          managed: {}
        route:
          httpRoute: {}
        scheduler:
          pool: {}

[/code]

在上面的示例中，`variantCost: "15.0"` 表示运行该变体每个副本的相对成本。如果同一模型的另一个变体的 `variantCost: "5.0"`，那么 WVA 会优先在那个更便宜的变体上增加容量，然后再扩容这个变体。若未指定，默认值为 `"10.0"`。当使用 KEDA 后端时，`fallback` 字段可确保即使指标管道发生故障，部署也能维持最小副本数（此处为 2 个副本）—— 这是生产环境 LLM 部署至关重要的安全保障。

### 🔧 调度器高可用​

LLMInferenceService 调度器（EPP）现已支持**扩缩容和高可用**，允许在需要容错和更高路由吞吐量的生产部署中运行多个 EPP 副本。

### 🛡️ CRD Webhook 校验​

LLMInferenceService 现已包含带有完整 E2E 测试的 **CRD webhook 校验**，可在无效配置到达控制器之前提供早期反馈。它能在准入（admission）阶段捕获并行设置、工作负载规范和路由配置中的错误。

### 📋 通过 LLMInferenceServiceConfig 实现配置组合​

LLMInferenceService 通过 LLMInferenceServiceConfig 支持**配置组合模型**，可以创建能在多个 LLMInferenceService 资源之间共享的可复用模板。合并顺序如下：

  1. **Well-Known Configs（内置常用配置）** → 2. **显式 BaseRefs** → 3. **LLMInferenceService Spec**

这使得平台团队可以定义标准化的 vLLM worker 模板、路由/调度器配置和资源默认值，同时允许应用团队覆盖特定设置。

### 📦 LLMInferenceService 其他改进​

  * **标签和注解传播**到下游工作负载资源（[#5009](https://github.com/kserve/kserve/pull/5009)）
  * **Prometheus 注解传播**到工作负载，用于指标采集（[#5086](https://github.com/kserve/kserve/pull/5086)）
  * **证书管理**，支持 DNS/IP SAN 及自签名证书的自动续期（[#5099](https://github.com/kserve/kserve/pull/5099)）
  * **改进的 CA bundle 管理**，实现安全通信（[#4803](https://github.com/kserve/kserve/pull/4803)）
  * **可选的 storageInitializer** —— 使用预加载模型时可跳过模型下载（[#4970](https://github.com/kserve/kserve/pull/4970)）
  * **InferencePool 自动迁移**，实现无缝升级（[#5007](https://github.com/kserve/kserve/pull/5007)）
  * **仅将 completions 请求经由 InferencePool 路由**，用于 chat/completion 端点（[#5087](https://github.com/kserve/kserve/pull/5087)）
  * **为 vLLM 容器添加启动探针**，实现更可靠的健康监测（[#5063](https://github.com/kserve/kserve/pull/5063)）
  * **vLLM 参数迁移到 command 字段**，使配置更加清晰（[#5049](https://github.com/kserve/kserve/pull/5049)）
  * **版本化的 well-known 配置解析**，实现稳定的配置管理（[#5096](https://github.com/kserve/kserve/pull/5096)）
  * **调度器配置支持 ConfigMap 或内联方式**，配置更加灵活（[#4856](https://github.com/kserve/kserve/pull/4856)）
  * **Pod init 容器故障监控**，提升可观测性（[#5034](https://github.com/kserve/kserve/pull/5034)）
  * **在调谐（reconciliation）期间保留外部管理的副本数**（[#4996](https://github.com/kserve/kserve/pull/4996)）
  * **允许优雅地停止 LLMInferenceService**（[#4839](https://github.com/kserve/kserve/pull/4839)）
  * **增强的 Gateway API URL 发现**，支持监听器主机名回退（[#5104](https://github.com/kserve/kserve/pull/5104)、[#5079](https://github.com/kserve/kserve/pull/5079)）

## 🏗️ 模块化组件架构​

KServe v0.17 引入了向**模块化、组件化部署**的根本性架构转变。KServe 现在由三个独立组件构成：

  * **kserve**（核心）：管理 InferenceService、ServingRuntime、ClusterServingRuntime、InferenceGraph 和 TrainedModel CRD。
  * **llmisvc**：面向生成式 AI 工作负载的 LLMInferenceService 控制器，管理 LLMInferenceService 和 LLMInferenceServiceConfig CRD。
  * **localmodel**（可选）：用于高效模型缓存的 LocalModel 控制器，管理 LocalModelCache、LocalModelNode 和 LocalModelNodeGroup CRD。

组合方式| 使用场景| 组件
---|---|---
**仅 KServe**|  预测式 AI| kserve
**KServe + LLMIsvc**|  预测式 AI + 生成式 AI| kserve + llmisvc
**全栈**|  预测式 AI + 生成式 AI + 模型缓存| kserve + llmisvc + localmodel

### Helm Chart 重构​

为了支持新的组件架构，**Helm chart 已被彻底重构**，从单一 chart 拆分为 **10 个独立的 Helm chart**：

**CRD Chart**（6 个 chart，提供完整版和精简版两种变体）：

  * `kserve-crd` / `kserve-crd-minimal`
  * `kserve-llmisvc-crd` / `kserve-llmisvc-crd-minimal`
  * `kserve-localmodel-crd` / `kserve-localmodel-crd-minimal`

**资源 Chart**（4 个 chart）：

  * `kserve-resources`（由 `kserve` 更名而来）
  * `kserve-llmisvc-resources`（新增）
  * `kserve-localmodel-resources`（新增）
  * `kserve-runtime-configs`（新增 —— 管理 ClusterServingRuntime 和 LLMIsvcConfig）

警告

这是一项**破坏性变更（breaking change）**。从 v0.16 升级的用户**不能**直接使用简单的 `helm upgrade` 命令。请遵循详细的[升级指南](https://kserve.github.io/website/docs/install/upgrade-guide)按步骤完成迁移。我们强烈建议先在非生产环境中测试升级。

对于全新安装，新的基于 Kustomize 组件的架构还通过独立 overlay、附加组件 overlay 和一体化 overlay 提供了可组合的部署选项。详情参见[安装概念](https://kserve.github.io/website/docs/install/overview)。

## 🔧 InferenceService 与平台改进​

### 存储性能​

  * **从 Azure 和 S3 并行下载 blob**，加快模型加载（[#4709](https://github.com/kserve/kserve/pull/4709)、[#4714](https://github.com/kserve/kserve/pull/4714)）
  * **更快的 S3 并行下载**，支持可配置的文件筛选（[#5102](https://github.com/kserve/kserve/pull/5102)、[#5119](https://github.com/kserve/kserve/pull/5119)）
  * **支持 Git 仓库**，可通过 HTTPS 直接从 Git 仓库下载模型（[#4966](https://github.com/kserve/kserve/pull/4966)）

### 新增 Serving Runtime​

  * **OpenVINO Model Server** —— Intel 优化的推理运行时，在 Intel 硬件上提供高性能服务（[#4592](https://github.com/kserve/kserve/pull/4592)）
  * **PredictiveServer** 运行时，配备完整的构建/发布基础设施和 E2E 测试（[#4954](https://github.com/kserve/kserve/pull/4954)）

### Gateway 与路由​

  * **Gateway API 升级至 v1.4.0**（[#5038](https://github.com/kserve/kserve/pull/5038)）
  * **PathTemplate 配置**，实现灵活的推理服务路由（[#4817](https://github.com/kserve/kserve/pull/4817)）

### vLLM 后端​

  * **升级到 vLLM v0.15.1**，带来性能提升（[#5098](https://github.com/kserve/kserve/pull/5098)）
  * **移除对 Python 3.9 的支持**（[#4851](https://github.com/kserve/kserve/pull/4851)）

### 其他增强​

  * **CSV 和 Parquet 序列化器（marshaller）**，扩展数据格式支持（[#5115](https://github.com/kserve/kserve/pull/5115)）
  * **事件循环配置**，新增 `--event_loop` 标志，支持 `auto`、`asyncio` 和 `uvloop`（[#4971](https://github.com/kserve/kserve/pull/4971)）
  * **基于注解的运行时默认值**，用于 MLServer（[#5064](https://github.com/kserve/kserve/pull/5064)）
  * **向服务容器暴露 `INFERENCE_SERVICE_NAME` 环境变量**（[#5013](https://github.com/kserve/kserve/pull/5013)）
  * **在 InferenceService 状态中呈现故障条件**（[#5114](https://github.com/kserve/kserve/pull/5114)）
  * **推理日志批处理**，支持外部序列化（[#5061](https://github.com/kserve/kserve/pull/5061)）

### 基础设施更新​

  * Kubernetes 相关包升级至 **v0.34.0**
  * Knative Serving 更新至 **v1.21.1**
  * Go 更新至 **1.25**
  * Kubebuilder 更新至 **1.9.0**
  * KEDA 从 2.16.1 升级至 **2.17.3**
  * 测试基础设施中的 MinIO 替换为 **SeaweedFS**

## 🔒 安全修复​

修复了多个安全漏洞：

  * CVE-2025-62727（Starlette）
  * CVE-2025-22872、CVE-2025-47914、CVE-2025-58181
  * CVE-2024-43598（LightGBM 更新至 4.6.0）
  * CVE-2025-43859（h11 HTTP 解析）
  * CVE-2025-66418（解压链）
  * CVE-2025-68156（expr-lang/expr）
  * CVE-2026-26007（cryptography 子群攻击）
  * CVE-2026-24486（python-multipart 任意文件写入）
  * https.go 及 tar 解压中的路径遍历漏洞

## 🔍 发布说明​

要查看全部 167 个已合并 Pull Request、缺陷修复和已知问题的完整列表，请访问 GitHub 发布页面：

  * [v0.17.0](https://github.com/kserve/kserve/releases/tag/v0.17.0)
  * [v0.17.0-rc1](https://github.com/kserve/kserve/releases/tag/v0.17.0-rc1)
  * [v0.17.0-rc0](https://github.com/kserve/kserve/releases/tag/v0.17.0-rc0)

## 🙏 致谢​

我们衷心感谢促成此次发布的 **38 位以上贡献者**，其中包括 **21 位首次贡献者**。你们的努力持续推动 KServe 作为领先的机器学习模型服务平台不断进步。

  * **核心贡献者**：KServe 维护者和长期贡献者
  * **社区**：所有报告问题、提供反馈和测试功能的人
  * **新贡献者**：欢迎所有帮助塑造此次发布的首次贡献者

## 🤝 加入社区​

我们邀请你探索 KServe v0.17 的新特性，并参与项目的持续开发：

  * 访问我们的[官网](https://kserve.github.io/website/)或 [GitHub](https://github.com/kserve)
  * 加入 Slack（[#kserve](https://github.com/kserve/community?tab=readme-ov-file#questions-and-issues)）
  * 订阅 [KServe 日历](https://zoom-lfx.platform.linuxfoundation.org/meetings/kserve?view=month)，参加我们的社区会议。
  * 查看我们的[社区 GitHub 仓库](https://github.com/kserve/community)，了解如何做出贡献。我们期待与你一起让 KServe 变得更好，并推动它的广泛采用！

**祝服务顺利（Happy serving）！**

* * *

_KServe 团队致力于让机器学习模型服务变得简单、可扩展且标准化。感谢你成为我们社区的一员！_

**标签：**

  * [Releases](/website/blog/tags/releases "项目发布")

[编辑此页](https://github.com/kserve/website/tree/main/blog/2026-03-13-KServe-0.17-release.md)
