# 粘性直到饱和（Sticky Until Saturated）：llm-d 中的 Token 感知路由

> **文档信息**
>
> - **来源**：[llm-d 官方博客](https://llm-d.ai/blog/sticky-until-saturated-token-aware-routing)
> - **发布日期**：2026-08-17
> - **阅读时长**：39 分钟
> - **抓取时间**：2026-09-01
> - **格式**：网页转 Markdown

> *llm-d 路由器如何在 KV-cache 亲和性与 token 负载之间取得平衡。*

## 作者

| 作者 | 职位 |
| --- | --- |
| [Kaushik Mitra](https://github.com/kaushikmitr) | 软件工程师，Google |
| [Abdullah Gharaibeh](https://github.com/ahg-g) | 高级主任软件工程师，Google |
| [Cheng-Hsiang Chiu](https://github.com/cheng-hsiang-chiu) | 软件工程师，Google |
| [Brent Stephens](https://github.com/brent-srg) | 软件工程师，Google |

llm-d 路由器的默认配置已经改为运维人员可以推理理解的方案，它建立在一个单一的方法论之上：**token 感知路由（token-aware routing）**。调度器优先考虑 KV-cache 亲和性，让每个请求保持在已经持有其前缀的端点上，除非该端点上的负载超过了一个经过标定的上限；超过上限后，调度器会忽略亲和性，仅根据负载来选择端点。负载以匹配工作负载瓶颈的 token 来度量，由此得到两种配置：对于 prefill 受限的流量（长提示词），使用**前缀缓存亲和 + token 负载**（`prefix-cache-affinity-filter + token-load-scorer`），基于在途未缓存 prefill token 进行路由；对于 decode 受限的流量（长输出），使用**前缀缓存亲和 + 活跃请求**（`prefix-cache-affinity-filter + active-request-scorer`），基于活跃流数量进行路由。我们把亲和过滤器与某个负载信号的每一种配对称为一个 _瓶颈匹配配置_（bottleneck-matched configuration，简称 _匹配配置_）。

促成这次改变的是两种失败模式。第一种就是之前的默认配置本身：一个由四个信号加权混合的方案（前缀缓存匹配、队列深度、KV 利用率、LRU），其涌现行为难以预测，更难以调优。第二种是大多数路由器因过度依赖 KV-cache 亲和性而遭受的热点问题：亲和性会把流量集中到缓存已预热的端点上，而如果没有显式的饱和释放机制，预热端点会持续吸收负载，直到超过冷端点本可以更快服务请求的临界点。token 感知路由将一个信号与一个瓶颈和一个标定上限配对，因此调度器是可解读的：运维人员能够预测它会做什么、为什么这样做，以及在负载下它会如何退化。

结果支持了这次改变。在下面的基准测试中，匹配配置在 prefill 受限的工作负载上可以持续达到 Kubernetes Service 轮询（round-robin）2–3 倍的吞吐量，同时在整个工作区间内保持 TTFT，而在其他所有场景中也能持平或更优。该方案现在已成为 llm-d 的 [optimized-baseline](https://github.com/llm-d/llm-d/tree/main/guides/optimized-baseline)、[agentic-serving](https://github.com/llm-d/llm-d/tree/main/guides/agentic-serving)、[multimodal-serving](https://github.com/llm-d/llm-d/tree/main/guides/multimodal-serving) 和 [P/D-disaggregation](https://github.com/llm-d/llm-d/tree/main/guides/pd-disaggregation) 指南中的默认配置，并已运行在生产推理服务集群上，包括 Google Cloud Vertex AI、Red Hat 和 Mistral。

本文是这次转变背后的分析。它在覆盖两种瓶颈类型（prefill 计算、decode 槽位）的三个工作负载上构建了该框架，从一次硬件标定出发以封闭形式推导出该配置中唯一的阈值（见"从硬件推导 τ"），并评估了延迟预测器（latency-predictor）流水线，作为面向高方差流量或瓶颈未知场景的运维人员的工作负载无关替代方案。这些实验是在亲和过滤器较早的参数化接口上运行的；随后发布的接口已围绕本文推导出的标定量重新设计（`peakPrefillThroughput`，见"饱和覆盖"），标定流程也随路由器一起以[共享配方](https://github.com/llm-d/llm-d/tree/main/guides/recipes/router/calibration)的形式发布，并附有按（模型、加速器）划分的[配置矩阵](https://github.com/llm-d/llm-d/blob/main/guides/recipes/router/calibration/configuration-matrix.md)。

## 实验设置

**硬件与推理服务栈。** 10 台（b2b-saas 为 8 台）运行 vLLM 的 Qwen3-32B 模型服务器，每台采用张量并行 TP=2，跨 2× H100 GPU，部署在专用 GKE 节点上。vLLM 配置为 `max-num-batched-tokens=8192` 并启用连续批处理（continuous batching）。每台模型服务器以 Kubernetes pod 的形式运行，前端由 Gateway API 的 `InferencePool` 资源通过路由器转发推理流量。全文中我们把一台模型服务器称为一个端点（endpoint）；在本部署中每个端点即一个 pod，但在多节点部署中一个端点可以跨多个 pod。

**路由器流水线与插件配置。** 路由器通过由过滤器（filter）和评分器（scorer）插件组成的流水线来构建调度决策。过滤器收缩候选端点集合；评分器分配加权得分；选择器（picker）插件从已评分的候选中选出最终目标。我们评估了五种调度器配置：

- **k8**：Kubernetes Service 轮询，完全绕过路由器。作为无调度器基线。
- **token-load-aware**（token 负载感知）：`prefix-cache-affinity-filter`（`affinityThreshold=0.8`、`maxTokensInFlightPenalty=286720`；早期接口中基于预测器的 TTFT 阈值在该实验组中未启用，未部署预测器），其后接 `token-load-scorer` 和 `max-score-picker`。在已发布的接口下，此配置对应 `peakPrefillThroughput=20480` 与 `maxTTFTPenaltyMs=14000`（两种参数化之间的映射见"饱和覆盖"）。这是 prefill 受限工作负载的匹配配置。本文中我们将 `maxTokensInFlightPenalty` 统一记为 τ。token 负载评分器采用前瞻（lookahead）公式：`score(endpoint) = current_uncached_in_flight(endpoint) + prompt_tokens × (1 - hit_rate(endpoint))`，其中第二项是该请求在各候选端点上将新增的未缓存边际工作量。所有工作负载都启用了流式传输；在在途负载生产者（in-flight load producer）的默认配置下（`addEstimatedOutputTokens=false`），请求的 token 在第一个流式分块到达时即被释放，因此在途计数器只跟踪尚未 prefill 的 token。也可以将生产者配置为（`addEstimatedOutputTokens=true`）额外持有一个输出估计量 `round(inputTokens × outputRatio)`（默认比率 1.5，受客户端输出上限和可选的运维上限约束），直到响应完成，以近似 KV cache 中持有的 token 数。
- **active-request-aware**（活跃请求感知）：`prefix-cache-affinity-filter`（与 token-load-aware 配置相同的 `affinityThreshold=0.8` 和 `maxTokensInFlightPenalty=286720` 饱和覆盖），其后接 `active-request-scorer` 和 `max-score-picker`。这是 decode 受限工作负载的匹配配置。亲和过滤器在结构上包含在内，但当共享前缀内容太少、无法达到 0.8 阈值时它是一个空操作（no-op）；在 reasoning 工作负载的 250-token 共享系统提示词下，过滤器会放行所有候选端点，下游评分器在完整集合上运作。
- **legacy blend**（遗留混合方案）：在这些实验运行时作为路由器默认配置发布的启发式多信号加权混合方案（此后已被 prefix+token 匹配配置取代其默认地位）：`prefix-cache-scorer (weight=3) + queue-scorer (weight=2) + kv-cache-utilization-scorer (weight=2) + no-hit-lru-scorer (weight=2)`，配合 `max-score-picker`。无亲和过滤器，前缀局部性通过 prefix-cache-scorer 在混合中的权重来体现。
- **latency-predictor**（延迟预测器）：`prefix-cache-affinity-filter (maxTTFTPenaltyMs=5000, maxTokensInFlightPenalty=0) + latency-scorer`，配合 `weighted-random-picker`。工作负载无关的流水线，预测器的训练与预测 sidecar 与每台模型服务器一起部署。

所有使用亲和过滤器的配置都采用相同的 `affinityThreshold=0.8` 和 `explorationProbability=0`。完整的 configmap 见附录。

**工作负载。** 使用三个工作负载：

- **code-generation**（代码生成，prefill 受限）：来自 [`inference-perf` 工作负载目录](https://github.com/kubernetes-sigs/inference-perf/tree/main/workload-catalog/code-generation)的会话回放（conversation-replay）工作负载。每个会话的系统提示词为 3000–100000 token（平均约 55000），每个会话 2–20 轮（平均 6 轮），每轮输入 100–10000 token（平均 1500），每轮输出 50–4000 token（平均 800）。并发从 10 扫到 100。
- **reasoning**（推理，decode 受限）：来自 [`inference-perf` 工作负载目录](https://github.com/kubernetes-sigs/inference-perf/tree/main/workload-catalog/reasoning)的推理工作负载。250-token 共享系统提示词，1000-token 用户提示词，输出均值 8000 token（最大 32000）。每次运行生成 `num_conversations = concurrency_level` 个单轮会话（每个会话由一个唯一用户提示词与同一共享系统提示词配对），并发出 `3 × concurrency_level` 个请求，因此每个会话在整个运行中大约被发出三次。这种重复确保工作负载考察的是稳态负载均衡，而不是被初始到达突发所主导。并发从 50 扫到 450。
- **b2b-saas**（prefill 受限，病态场景）：[llm-d optimized-baseline 基准指南](https://github.com/llm-d/llm-d/blob/main/guides/optimized-baseline/benchmark-templates/guide.yaml)使用、并在 [llm-d "KV-cache wins you can see" 博客文章](https://llm-d.ai/blog/kvcache-wins-you-can-see#a1-benchmark-setup-details)中分析过的共享前缀工作负载。150 个不同的 6000-token 系统提示词 × 5 个问题变体，1200-token 问题，1000-token 输出。在泊松到达下进行共享前缀采样，QPS 从 3 逐步升到 60。同一个 750 条提示词的语料库在集群中反复循环，因此这是对匹配配置缓存管理行为的压力测试，而非代表性的流量形态。相关定位见"b2b-saas 注意事项"。

完整的工作负载 YAML 规格见附录。

**负载生成与指标。** 负载由 [`inference-perf`](https://github.com/kubernetes-sigs/inference-perf) 生成。分阶段指标（TTFT、TPOT、端到端延迟、吞吐量、成功率、队列深度、KV 利用率、前缀缓存命中率）以 JSON 形式导出到 Google Cloud Storage 并汇总分析。每个（工作负载、配置、并发级别）组合各运行一次；我们报告实际达到的值而非置信区间。

## 主要结论

识别工作负载的瓶颈，部署对应的匹配配置：

| 瓶颈 | 工作负载特征 | 配置 | 说明 |
| --- | --- | --- | --- |
| **Prefill 计算** | 提示词较长（相对输出而言），且流量规模足以使聚合提示词处理压力逼近单端点 prefill 吞吐上限 | `prefix-cache-affinity-filter`（标定的 `peakPrefillThroughput`，`maxTTFTPenaltyMs` \(= T_{\max}\)）+ `token-load-scorer` + `max-score-picker` | 同一匹配配置既覆盖典型的 prefill 受限工作负载（code-generation），也覆盖同一小语料库反复出现的病态场景（b2b-saas）。该门控编码了 \(\tau = \texttt{peakPrefillThroughput} \times \texttt{maxTTFTPenaltyMs}\)（见推导）；我们的运行对应 \(20{,}480\ \text{tok/s} \times 14\ \text{s} = 286{,}720\)。`maxTTFTPenaltyMs` 必须非零，它是饱和释放机制；为 0 时粘性将永远不会被打破。 |
| **Decode 槽位** | 提示词短、输出长，decode 批处理容量是约束条件 | `prefix-cache-affinity-filter`（标定的 `peakPrefillThroughput`，`maxTTFTPenaltyMs` \(= T_{\max}\)）+ `active-request-scorer` + `max-score-picker` | 亲和过滤器在结构上包含在内，但当共享前缀内容太少、无法达到 `affinityThreshold=0.8` 时它是一个空操作。 |
| **混合 / 未知 / 高方差** | 流量形态异构；瓶颈随时间迁移的生产工作负载 | `prefix-cache-affinity-filter`（`ttftSource=latencyPredictor`、`maxTTFTPenaltyMs=5000`）+ `latency-scorer` + `weighted-random-picker` | 在本文评估的所有工作负载上都优于 k8。但会增加部署开销（预测器训练、sidecar 部署），当工作负载已被充分理解时应优先采用匹配配置。对于瓶颈动态迁移的高方差生产工作负载，它很可能是最佳选择，但这一点在本文中未经测试。 |

亲和过滤器中的阈值 τ 应从每个（模型、加速器）组合的一次硬件标定测量中推导得出，见推导部分。在 Qwen3-32B / H100 上，配置 `max-num-batched-tokens=8192`、TTFT 退化容忍度为 14s 时，得到 τ = 286720（= 35 × 8192，即 35 个 max-num-batched-tokens 大小的待处理未缓存 prefill 工作块）。

prefix+token 配置是 prefill 受限工作负载的瓶颈匹配选择，同时也是跨场景的强力默认方案；它现在就是 optimized-baseline 指南作为 llm-d 默认发布的配置。特别是对于 decode 受限的工作负载（活跃流数量是关键信号的场景），`active-request-scorer` 配置在工作区间拐点处能提供小幅额外改进。对于瓶颈动态迁移的混合工作负载，延迟预测器流水线是更稳妥的默认选择，代价是额外的部署开销。

关于启发式加权混合方案（legacy blend）的一点说明：一套精心调优的权重在其针对性调优的特定工作负载上会优于通用配置。该混合方案的权重（`prefix=3, queue=2, kv=2, lru=2`）体现了对重复语料、chat-completion 形态流量的优化，并在 b2b-saas 上取得了最低的 TTFT-p90（见下文结果）。我们并不声称匹配配置能在混合方案专门设计针对的工作负载上击败一套定制权重；我们声称的是匹配配置跨工作负载可预测、可由一次标定测量推导得出，并且在每种瓶颈类型的整条负载曲线上都具有竞争力。这里的权衡是：在单一工作负载上的最佳性能 vs. 跨多种工作负载的可解读性与可移植性。

"结果"一节给出证据。"前缀缓存亲和过滤器"一节展开亲和过滤器的架构论证。"从硬件推导 τ"一节从硬件标定推导 τ。

## 三个工作负载，两种瓶颈

本工作的核心主张是：LLM 推理调度受益于一种不同于对多信号评分器进行权重调优的组织原则：先识别在目标工作负载上限制吞吐的因素，再部署信号与该限制相匹配的调度器配置。本节建立这一分类体系。

### 两种瓶颈类型

一个经过 vLLM 的请求由两个资源特征不同的阶段组成。**Prefill**（预填充）以 `max-num-batched-tokens`（我们的设置中为 8192）为单位分块处理提示词，受 GPU 计算吞吐量和单块注意力开销约束。**Decode**（解码）在一批并发请求中逐 token 生成输出，受 KV-cache 显存和可用 decode 槽位数量约束。哪个阶段先饱和取决于工作负载的形态，具体来说是提示词 token 与输出 token 的比例。

我们识别出两种瓶颈类型：

**Prefill 受限（计算受限）。** 瓶颈是单端点的 prefill 吞吐量。提示词足够长，以至于填满 `max-num-batched-tokens` 在每次迭代中消耗可观时间，且若请求分配不当，聚合提示词量会超出集群的处理能力。decode 批很快填满，但大部分迭代时间花在 prefill 块而非 decode 步骤上。正确的调度信号是**每个端点当前在途的未缓存 prefill token 数**：把新提示词路由到在途未缓存 token 计数最低的端点，可以最大化有效 prefill 吞吐。code-generation（提示词均值约 70k token，输出均值 800 token）和 b2b-saas（7200-token 提示词、1000-token 输出、固定 750 条提示词语料库）都是 prefill 受限。二者在缓存管理动态上有所不同，见"b2b-saas 注意事项"。

**Decode 受限（槽位受限）。** 瓶颈是一个端点能维持的并发 decode 流数量。提示词短、输出长，prefill 很快完成，集群大部分时间都在生成输出 token。KV-cache 压力随并发请求数增长，而不是随单个请求的大小增长。正确的调度信号是**每个端点的运行中请求数**：把新工作发送到活跃流最少的端点，可以让 decode 批保持均匀负载。另一种以 token 计量的信号是处理中的 token 数：在途负载生产者跟踪尚未 prefill 的 token，也可以配置为（`addEstimatedOutputTokens=true`）额外持有一个基于比率的输出估计量直到响应完成，以近似 KV-cache 占用。一旦请求越过 prefill 阶段，其在 KV cache 中的实际 token 原则上也可以作为 decode 负载信号来跟踪——通过路由器基于事件驱动的 KV 块索引，而不是引擎侧轮询式的 KV 使用率指标（后者滞后于分发）。实践中简单的活跃请求计数效果良好，这也是我们在本文中基准测试的形式。reasoning（提示词均值 1000 token，输出均值 8000 token）是这一类型的代表。

这两种类型并非总是互斥，真实工作负载可能介于两者之间，或随负载变化在两者之间迁移，但它们覆盖了路由器在实践中遇到的主要情形。这个分类体系之所以有用，恰恰在于两者的匹配配置各不相同。

### B2B-SaaS 注意事项

在展示结果之前，需要对 b2b-saas 做明确定位。与近似真实流量形态的 code-generation 和 reasoning 不同，**b2b-saas 是在一个固定的小语料库上的压力测试**：150 个不同的系统提示词 × 5 个问题变体，随机采样。同一个 750 条提示词的集合在集群中反复循环。由于聚合可缓存内容（约 90 万 token）超过集群的单端点 KV 容量（每端点约 20–30 万 token × 8 个端点，若前缀被复制则存在大量重叠），这个工作负载被刻意设计成让缓存分布成为 prefill 计算之上的主导次级因素。

这产生了两个特征，使 b2b-saas 尽管与 code-generation 同为 prefill 受限却有所区别：

- **集群会依 τ 的取值稳定到两种截然不同的稳态之一。** 要么收敛到稳定分区（每个端点拥有前缀的一个子集，命中率保持高位），要么级联陷入淘汰震荡（每个端点都试图缓存所有前缀，命中率崩溃）。不存在平滑的中间态，τ 的微小变化就能让集群从一种状态翻转到另一种。
- **稳态行为取决于爬坡阶段提示词在各端点上的初始播种方式。** 由于同样的 150 个前缀反复出现，最初几个爬坡阶段形成的分区会在余下的运行中持续存在。同一配置的两次运行可能收敛到不同的分区并产生不同的绝对数值，尽管各配置的_相对_排名是稳定的。

这些特性使 b2b-saas 成为检验缓存管理机制的优秀基准，却是超参数扫描的糟糕基准。我们专门用它来证明：(a) token 负载感知路由在前缀重用密集的工作负载上以 2–2.5× 的优势击败轮询；(b) `prefix-cache-affinity-filter` 中 τ=286720 的阈值（= 35 × 8192）起到了饱和阀的作用，这一点通过更小 τ 值的失败模式得到了实证验证。我们**不**用它来声称一个精确的最优 τ；双稳态意味着这样的声称即使在 b2b-saas 内部也无法泛化，更遑论跨工作负载。prefill 受限场景的匹配配置是结构性的；调参则来自推导部分的 τ_sat 公式，而非 b2b-saas 的参数扫描。

### 主要指标

在下文中，对 prefill 受限的工作负载（code-gen 和 b2b-saas）我们关注 **TTFT-p90 和每秒输入 token 数**，对 decode 受限的工作负载（reasoning）我们关注 **TPOT-p90 和每秒输出 token 数**。这些指标反映了各工作负载的瓶颈是否得到了良好管理；其他指标在附录中报告，但在分析中不起决定作用。

## 结果

### Code-Generation（Prefill 受限）

长提示词（均值 70k token，随会话轮次增长），中等输出（均值 800），无跨请求前缀重叠。单端点 prefill 计算是约束条件。匹配配置：`prefix-cache-affinity-filter`（τ=286720，等价于 35 × B = 35 个块）+ `token-load-scorer` + `max-score-picker`。前瞻评分：`score(endpoint) = current_uncached_in_flight + prompt_tokens × (1 - hit_rate(endpoint))`。

#### TTFT-p90 与每秒输入 token 数

#### 前缀缓存命中率

每个会话的提示词是唯一的，因此缓存命中只来自会话内轮次的连续性。每个并发点都是独立运行（集群在运行之间不预热），因此 conc=10 这一列噪声较大——该次运行总共只有约 60 个请求（10 个会话 × 约 6 轮），在这个量级上命中率的绝对值主要取决于各个会话恰好落在哪些端点上。在 conc=20 时，prefix+token 达到了最高的持续命中率（48%），这与亲和过滤器保留轮次间局部性的机制一致，而软亲和（latency-predictor）和无亲和（legacy blend，它依赖其混合中 prefix-cache-scorer 的权重来替代）配置则做不到。在高并发下，所有配置的缓存命中率都会崩溃，因为活跃会话的多样性耗尽了单端点 KV 容量，但上文的 prefill 吞吐优势依然存在，表明即使稳态命中率指标偏低，过滤器仍在发挥有效作用。

#### 小结

在整条负载曲线上，prefix+token 拥有最低的 TTFT-p90 和最高的每秒输入 token 数。在 conc=60 以内 TTFT-p90 保持在 40s 以下；吞吐在 conc=60 时到达约 47k in_t/s 的平台并保持。latency-predictor 在整条曲线上与 prefix+token 的 TTFT 差距保持在约 10s 以内。k8 和 legacy blend 在多个并发点出现悬崖式失效，不适合持续运行。

### Reasoning（Decode 受限）

短提示词（均值 1000 token），长输出（均值 8000，最大 32k），250-token 共享系统提示词。每次运行有 `num_conversations = conc` 个单轮会话，每个大约发出三次（共 `3 × conc` 个请求）。在约 150 以上的所有并发级别，decode 槽位压力是约束条件。匹配配置：`prefix-cache-affinity-filter + active-request-scorer + max-score-picker`（即 "run-req" 配置）。亲和过滤器虽然包含在内，但实际上是一个空操作：一个会话的首次发出最多只能匹配 250-token 共享提示词（相对约 1000-token 的用户提示词最多 0.25 的亲和度），且观测到的命中率始终低于 0.8 阈值（见下文命中率数据），因此过滤器放行所有端点。我们也纳入了与工作负载无关的 prefix+token 配置作对比；它在此有竞争力，但在 decode 槽位信号上并未超过 run-req。

#### TPOT-p90 与每秒输出 token 数

#### 前缀缓存命中率

命中率从未达到 0.8 的亲和过滤阈值：一个会话的首次发出最多只能命中 250-token 共享提示词（最多 0.25 的亲和度），而且尽管同一会话的重复发出原则上完全可缓存，观测到的命中率从未越过该阈值。run-req 和 prefix+token 在工作区间内都取得了高命中率，因为它们的调度器不会主动打散请求（run-req 最小化每端点的流数量，作为副作用保留了局部性；prefix+token 的过滤器不生效，但下游的 token-load-scorer 同样通过低负载偏好保留了局部性）。legacy blend 造成更多分散，而 latency-predictor 的加权随机采样会主动打散请求。

#### 小结

总体上 run-req 拥有最低的 TPOT-p90 和最高的每秒输出 token 数。prefix+token 具有竞争力，但没有超过 run-req。latency-predictor 在饱和点上与 run-req 持平。legacy blend 以前缀为主导的加权评分制造了热点，在 50 以上的所有并发级别都造成损害。

### B2B-SaaS（病态的 Prefill 受限）

合成压力工作负载：150 个不同的系统提示词 × 5 个问题变体，在泊松到达下随机采样，QPS 从 3 → 60 逐步爬升。7200-token 提示词（6000-token 系统提示词 + 1200-token 问题），1000-token 输出，流式传输。若前缀被复制而非分区，聚合可缓存内容（约 90 万 token）会超过聚合单端点 KV 容量。匹配配置：`prefix-cache-affinity-filter`（τ=286720，硬性）+ `token-load-scorer` + `max-score-picker`。关于 b2b-saas 作为压力测试的定位见"b2b-saas 注意事项"。

#### TTFT-p90 与每秒输入 token 数

#### 前缀缓存命中率

prefix+token 和 legacy blend 在负载爬升过 QPS 约 20 之后都维持了很高的命中率（约 90–95%）。这表明集群的前缀分区维持住了：每个端点拥有 150 个前缀中的一个子集，并在本地服务对该子集的重复请求。两种配置都在工作区间内保住了分区，这正是二者都取得上文 prefill 吞吐优势的原因。不过它们保住分区的机制不同：prefix+token 用 `prefix-cache-affinity-filter` 把候选限制到已预热端点，再用 `token-load-scorer + max-score-picker` 在其中挑选。legacy blend 没有亲和过滤器（见实验设置）；其前缀局部性靠给 `prefix-cache-scorer` 在混合中赋以最大权重（3）来实现，这使 `max-score-picker` 几乎总是选中前缀匹配度最高的端点。二者机制上不同，但在这个工作负载中产生了相似的稳态行为，因为最终它们大部分时间都路由到缓存已预热的端点。在非常高的 QPS（50+）下，prefix+token 的命中率降至 70–73%，而 legacy blend 维持在 96–97%；这反映的是 τ=286720 的饱和阀在 prefix+token 上被触发——当预热端点累积了足够多的在途未缓存 token 而触发覆盖机制时，流量会被短暂地从预热端点重新分配出去。legacy blend 没有这样的阀门。

#### 小结

总体上 prefix+token（τ=286720）和 legacy blend 都维持了可用的 TTFT 和高 prefill 吞吐。k8 无法达到这个区间，latency-predictor 也不能。在工作区间之外，legacy blend 的定制调优权重使其 TTFT 略占优势。下文的推导从一次标定测量和一个 SLO 上限得出 τ=286720，免去了为每个工作负载凭经验调 τ 的需要。

### 跨工作负载总结

三个工作负载、两种瓶颈、两个匹配配置。三个工作负载呈现出一致的模式：信号与工作负载瓶颈匹配的配置在稳定工作区间内胜出，而其他替代方案要么更早出现拐点，要么表现出悬崖式失效，或者两者兼有。没有单一配置能在全部三个工作负载上都胜出，但选择部署哪个配置完全由瓶颈识别所决定。

| 工作负载 | 瓶颈 | 匹配配置 | 拐点位置（匹配配置） | 拐点位置（k8） | 拐点处吞吐（匹配配置 vs. k8） |
| --- | --- | --- | ---: | ---: | --- |
| code-generation | prefill 计算 | prefix-cache-affinity（\(\tau = 286{,}720\)）+ token-load + max-score | conc=60 | conc=30 | 46k vs. 16k in_t/s（2.9×） |
| reasoning | decode 槽位 | prefix-cache-affinity（\(\tau = 286{,}720\)）+ active-request-scorer + max-score | conc=350 | conc=250 | 7.0k vs. 6.8k out_t/s（约持平） |
| b2b-saas | prefill 计算（病态缓存动态） | prefix-cache-affinity（\(\tau = 286{,}720\)）+ token-load + max-score | QPS=38 | QPS=18 | 90k vs. 45k in_t/s（2.0×） |

## 前缀缓存亲和过滤器

该过滤器收缩传递给下游评分器的候选集合。每个端点被赋予一个亲和度得分（传入提示词 token 中已在该端点上被缓存的比例）；亲和度 ≥ `affinityThreshold`（我们的设置中为 0.8）的端点构成候选集。如果没有端点达到阈值，则所有端点放行通过，下游评分器在完整集合上运作。

### 跨瓶颈类型的可组合性

在全部三个匹配配置中，该过滤器都以相同的 `affinityThreshold=0.8` 存在。变化的是下游评分器，过滤器本身不变。

- **Prefill 受限，典型场景**（code-gen）：过滤器把候选限制到缓存已预热的端点（当存在轮次间缓存连续性时亲和度 > 0.8）→ `token-load-scorer` 在预热端点中选出在途未缓存 token 最少的那个
- **Prefill 受限，病态场景**（b2b-saas）：过滤器把候选限制到缓存已预热的端点（提示词中的系统提示词部分带来亲和度 > 0.8）→ `token-load-scorer` 在预热端点中选出在途未缓存 token 最少的那个
- **Decode 受限**（reasoning）：亲和度得分始终低于 0.8（250-token 共享提示词相对 1000-token 用户提示词在会话首次发出时最多给出 0.25 的亲和度，重复发出时观测命中率也低于阈值）→ 过滤器放行所有端点 → `active-request-scorer` 选出活跃流最少的端点，等同于过滤器不存在

一个过滤器、一个阈值、三个匹配配置，无需针对每个工作负载重新配置。

### 饱和覆盖机制

当继续坚守预热端点集合会带来危害时，过滤器允许非预热端点重新进入候选集。在我们实验所依据的接口中，这一覆盖机制表现为两个可选阈值：

- **`maxTokensInFlightPenalty`**（τ）：若某个非预热端点的在途未缓存 token 计数比预热端点低至少 τ，则将其加回候选集。这是匹配配置使用的饱和保护（τ=286720），只在预热端点确实计算饱和时才会触发。早期接口中基于预测器的阈值（下一条）需要 latency-predictor sidecar，在这些实验组中不生效。
- **`maxTTFTPenaltyMs`**：若某个非预热端点的预测 TTFT 与最优预热端点的预测 TTFT 之差在预算（毫秒）以内，则将其加回候选集。需要 latency-predictor 提供 TTFT 估计。latency-predictor 流水线使用此项（`maxTokensInFlightPenalty=0`、`maxTTFTPenaltyMs=5000`）；见下文。

llm-d main 分支上的过滤器把这两者统一为单个时域门控。每个端点的 TTFT 都会被估计——要么由 `peakPrefillThroughput` 得出（`in_flight_tokens / R_peak`，默认方式），要么由延迟预测器得出，由 `ttftSource` 选择——当最优预热端点的估计 TTFT 超出最优非预热端点的估计 TTFT 且差值大于 `maxTTFTPenaltyMs` 时，粘性被打破。在使用吞吐来源时，这正是上文的 token 差值条件：门控在在途差值超过内部计算的 τ = R_peak × T_max 时触发。发布的默认值（`peakPrefillThroughput=15928`、`maxTTFTPenaltyMs=18000`）复现了我们的阈值：15928 × 18s = 286,704 ≈ 286720。下一节从硬件标定推导这一数值；在当前接口下，标定测量值_就是_配置参数。

### Latency-Predictor 的阈值配置

latency-predictor 流水线设置 `maxTTFTPenaltyMs=5000`，并在过滤器产生的候选中使用加权随机采样。在适合使用预测器的工作负载上（code-gen 和 reasoning），这个组合经验上表现良好。如果观测到的 TTFT 回退需要更紧的覆盖机制，5s 这个值可以上调；我们尚未系统性地刻画其敏感度。在当前的过滤器接口中，该配置对应 `ttftSource: latencyPredictor` 加 `maxTTFTPenaltyMs=5000`。

## 从硬件推导 τ

上文中出现的 286720 阈值（= 35 × 8192，即 35 个 max-num-batched-tokens 大小的块）没有给出理由。本节从一次标定测量和一个运维人员选择的 SLO 上限推导它，并证明该公式可跨（模型、加速器）组合泛化。

这个推导已不再只是运维人员的操作配方：llm-d main 分支上的过滤器在内部实现了它，将 `peakPrefillThroughput`（R_peak）作为标定参数、`maxTTFTPenaltyMs`（T_max）作为 SLO 上限（见"饱和覆盖机制"）。标定测量本身以 [`guides/recipes/router/calibration`](https://github.com/llm-d/llm-d/tree/main/guides/recipes/router/calibration) 的形式发布（`calibrate.sh` 报告 `peakPrefillThroughput = CHUNK_SIZE / median(TTFT)`，正是下文的 `B / T(B)`），并在[配置矩阵](https://github.com/llm-d/llm-d/blob/main/guides/recipes/router/calibration/configuration-matrix.md)中附有按（模型、加速器、引擎）实测的参考值。

### τ 代表什么，以及它应在何时触发

τ 是在途未缓存 token 的差值，达到该差值时我们允许请求绕过预热端点转向较冷的端点。由于 vLLM 以每次迭代 `max-num-batched-tokens` 大小的块批处理 prefill，而非逐 token 串行处理，坚守粘性的代价受限于按块的吞吐量，而非按 token 的吞吐量。因此 τ 最好以块为单位表达：允许预热端点相对最冷端点累积多少个 `max-num-batched-tokens` 大小的待处理未缓存 prefill 工作作为领先量，之后覆盖机制才触发。阈值应当只在预热端点确实计算饱和时触发，即当把新到达的请求留在该端点上会导致其 TTFT 超过运维人员选择的 SLO 上限时。低于饱和点时，粘性基本上是免费的（预热端点在处理既有工作的同时也处理新的块）；到达饱和点时，粘性开始有害，因为新到达的请求要排在冷端点本可以更快完成的工作之后。

### 推导 τ_sat

每个端点的峰值 prefill 吞吐由 vLLM 处理一个完全填满的块的速度决定：

\[
R_{\text{peak}} = \frac{B}{T(B)}
\]

其中 `B = max-num-batched-tokens`（你的 vLLM 部署中配置的值，我们的设置中为 8192（见实验设置），但一般来说可由运维人员调整），`T(B)` 是在目标硬件上 prefill 一个 `B` token 大小的块所测得的墙钟时间。这就是一次标定测量。

在饱和点，在途未缓存 token 以速率 \(R_{\text{peak}}\) 排空。新到达请求的排队时间约为 \(\tau / R_{\text{peak}}\)，其中 \(\tau\) 是预热端点相对最冷端点在在途未缓存 token 上的领先量（即亲和过滤器覆盖机制配置为触发的那个量，见实验设置）。要让该排队时间不超过运维人员选择的 TTFT 退化上限 \(T_{\max}\)：

\[
\tau_{\text{sat}} = R_{\text{peak}} \times T_{\max} = \frac{B \times T_{\max}}{T(B)}
\]

该阈值只依赖一个测量量（`T(B)`）和一个运维人员选择的量（`T_max`）。别无其他。

### Qwen3-32B / H100 的标定

我们在提示词大小 `P = k × 8192`（`k = 1..19`）下测量了单请求的 TTFT：

`T(B = 8192) = 0.40` 秒，得出每端点 `R_peak = 20480 tokens/sec`。

（发布的标定配方对同一名义路径（Qwen3-32B / H100 / TP=2 / B=8192）报告 `peakPrefillThroughput = 15928`，低于我们的 20480，原因是方法学差异：配方取重复随机 token 请求经完整请求路径的 TTFT 中位数，而我们的值来自单请求二次拟合。两种分解在真正重要的量——覆盖阈值上是一致的：15928 × 18s 与 20480 × 14s 都给出 τ ≈ 286.7k。）

不同 T_max 取值：

| \(T_{\max}\) | \(\tau_{\text{sat}}\) |
| ---: | ---: |
| 10 s | 205k |
| **14 s** | **286,720** |
| 18 s | 369k |
| 25 s | 512k |

286720 对应 T_max = 14s，即"当保持粘性请求会导致新到达请求排队约 14 秒的 prefill 工作量时，阈值触发"。等价地，由于对某个整数 K 有 `τ = K × B` 且 `T_max = K × T(B)`，该阈值可以用块数表达：**τ = 35 × B = 35 个 max-num-batched-tokens 大小的待处理未缓存 prefill 工作块**，相当于在峰值块吞吐下约 14 秒的排队时间（35 × 0.40s）。块数表述在硬件之间是不变的：任何采用相同 `K=35` 的（模型、加速器）组合，其阈值都对应单块墙钟时间的同一倍数。这与 b2b-saas 结果中的经验观察相符：覆盖机制在 QPS 50+ 左右开始触发，此时 TTFT-p90 正逼近 30s 区间，集群接近计算饱和。

### 跨（模型、加速器、引擎）的可移植性

该公式要求每个（模型、加速器、引擎）组合进行一次标定运行。`R_peak` 以单个数字捕获了推理路径的一切特定因素——计算吞吐、注意力效率、内核质量、框架开销。

下表每一行都是实测值，没有估计值。带锚定标记的行是我们上文标定中的单请求拟合结果（20,480 tok/s）。GLM-5.2 一行在生产 Vertex AI 推理服务集群上标定（SGLang 加 EAGLE 投机解码，每 pod 8× B200，TP=8）：十次约 32k-token 的缓存未命中 prefill 经完整网关路径测量，TTFT 中位数 1.369s，得出每 pod 24,027 tok/s。其余各行是随路由器一同发布在[配置矩阵](https://github.com/llm-d/llm-d/blob/main/guides/recipes/router/calibration/configuration-matrix.md)中的实测 `peakPrefillThroughput` 值，外加 [agentic-serving](https://github.com/llm-d/llm-d/tree/main/guides/agentic-serving) 指南的 TPU v7x 标定。配方测得的值低于同一路径的单请求拟合（参考 H100 路径上 15,928 vs 我们的 20,480，见上文的调和说明），因此由它们推导出的 τ 偏保守。每个值只在其部署所运行的块大小、TP 和量化条件下成立（矩阵中的路径使用 `max-num-batched-tokens=8192`；GLM-5.2 集群运行 32k 的块）：改变其中任何一项后请重新测量。

这个分布让本节的观点比任何估计都更鲜明：`R_peak` 是完整推理路径的属性，而非加速器的属性。gpt-oss-120B 尽管是表中最大的模型，却在 TP=1 下取得了表中最高值，因为它是稀疏 MoE（约 5B 激活参数，MXFP4），一次 prefill 步骤只触及少量权重；在完全相同的 H100 硬件上，它达到了稠密 Qwen3-32B 路径约 2.5 倍的水平。架构、引擎、稀疏性和量化都会改变这个数字，这就是为什么矩阵以（模型、加速器、引擎）为行键，也是为什么只要硬件可用，一次标定 Job 就胜过任何估计。（SGLang / H100 一行修正了矩阵中当前发布的值 30,720，该值高了 2 倍。）（对于尚无法测量的硬件，可以用基于 FLOPs 的估计作为初始值；`portability_script/portability.py` 包含了这些测量存在之前我们使用的模型。）

| 路径 | 引擎 | TP | `peakPrefillThroughput`（tok/s） | \(T_{\max}=14\ \text{s}\) 时的 \(\tau_{\text{sat}}\) |
| --- | --- | ---: | ---: | ---: |
| gpt-oss-120B / H100 | vLLM | 1 | 39,065 | ≈547,000 |
| Qwen3-32B / TPU v7x | vLLM | 8 | 27,336 | ≈383,000 |
| Qwen3-32B / TPU v6e | vLLM | 8 | 26,290 | ≈368,000 |
| GLM-5.2-NVFP4 / 8× B200 | SGLang | 8 | 24,027 | ≈336,000 |
| Qwen3-32B / H100（锚定） | vLLM | 2 | 20,480 | **286,720** |
| Qwen3-Coder-480B-FP8 / TPU v7x | vLLM | 8 | 16,444 | ≈230,000 |
| Qwen3-VL-32B / H200 | vLLM | 2 | 15,751 | ≈221,000 |
| Qwen3-32B / H100 | SGLang | 2 | 15,360 | ≈215,000 |

### 为什么 τ 不能直接移植

上表解释了为什么单个 τ 值不能跨部署移植，即使这些部署共享同样的硬件。在同一 H100 上，把我们的 286,720 移到 gpt-oss-120B 路径会使阈值低了约 1.9 倍：该路径在同样 14s 预算内能排空约 547,000 个 token，因此阀门会在预热端点仍有余量时就触发，过早放弃缓存亲和性。把它移到 Qwen3-VL-32B / H200 路径会使阈值高了约 1.3 倍：该端点在约 221,000 个在途 token 附近就饱和，因此阀门会过晚介入，新到达的请求会排队超过 SLO 上限。同样的反转在同一加速器家族内部也存在：TPU v7x 在服务 Qwen3-32B 时支持 τ = 383,000，但服务 Qwen3-Coder-480B 时只有 230,000。

### 建议

- 按（模型、加速器、引擎、`max-num-batched-tokens`）组合逐一标定，测量 `T(B)`。先查阅随发布的[配置矩阵](https://github.com/llm-d/llm-d/blob/main/guides/recipes/router/calibration/configuration-matrix.md)，你的组合可能已经被测量过；否则针对你部署的服务栈运行 [`calibrate.sh`](https://github.com/llm-d/llm-d/tree/main/guides/recipes/router/calibration)，并把报告的值设置为过滤器的 `peakPrefillThroughput`。
- 根据运维方对 TTFT SLO 退化的容忍度选择 `T_max`（`maxTTFTPenaltyMs`）。我们使用 14s，对应我们的 b2b-saas 工作区间在饱和时逼近的 TTFT 退化容忍度；过滤器发布的默认值是 18s。SLO 更严格的工作负载应使用成比例更小的 T_max。
- 不要在未重新测量的情况下把 τ 值（或 `peakPrefillThroughput` 值）跨部署移植。

## 讨论

该框架把一个多维调优问题（选哪些评分器、哪些权重、哪些阈值）简化为一步决策（是哪种瓶颈），所有其他参数要么由匹配配置固定，要么可从一次标定测量推导得出。这在本文评估的三个工作负载上是有效的。自然的后续工作是在更广泛的流量形态上验证该框架。

[`inference-perf` 工作负载目录](https://github.com/kubernetes-sigs/inference-perf)提供了标准化的工作负载生成器，覆盖我们所评估的三个工作负载之外的多种提示词/输出分布、多轮结构和前缀重用模式。我们会优先考虑：

- **工具调用 / 智能体（agentic）工作负载**：大量短轮次、每轮提示词规模中等、上下文不断增长。测试当每轮提示词的增长使瓶颈在单次会话内在两种类型之间迁移时，prefill 受限的匹配配置是否仍然正确。这一点在这些实验之后已得到部分解决：[agentic-serving 指南](https://github.com/llm-d/llm-d/tree/main/guides/agentic-serving)现已为 H200 和 TPU v7 上的智能体代码生成工作负载发布了 prefill 受限的匹配配置（token 负载路由，配合可感知卸载的缓存核算和标定的 `peakPrefillThroughput`），包括 P/D 分离变体，并发布了基准结果；对单次会话内瓶颈类型迁移的系统性研究仍是待解问题。
- **多模态推理服务**：图像加文本的流量引入了一个双瓶颈分类未覆盖的阶段：一个先于 prefill 运行、可以独立饱和的视觉编码器。这一点在这些实验之后已经解决：[multimodal-serving 指南](https://github.com/llm-d/llm-d/tree/main/guides/multimodal-serving)现已按调度画像（scheduling profile）分别应用这些匹配配置：encode 按队列深度路由，每个执行 prefill 的画像（包括 prefill+decode 共置）都使用亲和过滤器 + token 负载方案，纯 decode 则只按活跃请求评分。与轮询基线在共享前缀多模态工作负载上的基准对比（8 台模型服务器，各 2×H200，TP=2）已随指南发布。多模态还对 token 负载信号背后的一个假设构成了考验：图像输入没有文本长度，因此每个请求的 token 计数来自多模态 `token-producer`，它根据图像的分辨率和数量估计每张图像的贡献，并针对所服务的模型进行了计数标定。为适配多模态，本文的两个常量不得不调整：`affinityThreshold` 从 0.8 降到 0.6，因为多模态提示词的可缓存比例上限约为 0.7（图像和问题部分永远不会缓存命中），所以 0.8 的默认值永远无法生效；`maxTTFTPenaltyMs` 从默认的 18s 加倍到 36s，因为产生 `peakPrefillThroughput` 的标定（Qwen3-VL-32B 在 H200/TP=2 上测得 15751）只覆盖文本 prefill：视觉编码器的耗时使图像密集的 prefill 按估计 token 计算比测得吞吐所暗示的更慢，因此过滤器的预测 TTFT 会低于实际墙钟时间。以本文的术语说，对图像密集流量而言有效 τ 加倍了；这个加倍后的门控已经基准验证，与此前发布的加权混合方案打平。
- **长输出摘要**：中等提示词、极长输出的工作负载。测试当输出 token 的多样性（相对 reasoning 那种集中式思维链风格）改变 decode 批处理动态时，decode 受限的匹配配置是否依然成立。
- **混合流量场景**：同一集群上同时存在 code-gen 风格与聊天风格的流量。测试是否需要按路由选择配置，或者单一配置是否足以服务异构流量。
- **高前缀重用的生产轨迹**：真实的 chat-completion 流量，前缀重用率高但语料库无界（不同于 b2b-saas 固定的 150 条提示词语料库）。验证匹配配置在真实缓存管理动态下是否成立，或者 b2b-saas 式的病态现象在实践中是否根本不会出现。

Prefill/decode 分离（P/D-disaggregated）部署值得单独说明：它们是该框架的极限情形。当 prefill 和 decode 在独立的资源池中运行时，两种瓶颈类型不再是需要运维人员识别的工作负载属性，而变成了每个池的结构属性：每个请求都会依次经历两种瓶颈，一次一个池。llm-d 的 P/D 指南（[pd-disaggregation](https://github.com/llm-d/llm-d/tree/main/guides/pd-disaggregation) 和 [agentic-serving](https://github.com/llm-d/llm-d/tree/main/guides/agentic-serving) 的 P/D 变体）因此同时发布了两种匹配配置：prefill 画像用亲和过滤器 + token 负载方案路由（带有自己标定的 `peakPrefillThroughput`），纯 decode 画像只按活跃请求评分。多模态的 e-p-d 配置把同一模式扩展到三个池，encode 按队列深度路由。在这些部署中，框架要求运维人员做的那一个决策（是哪种瓶颈）已由架构本身回答；剩下的只是按池标定。

一个次级后续工作：在瓶颈动态迁移的高方差生产流量上刻画 latency-predictor 流水线的表现。我们的三个基准工作负载各自处于单一瓶颈类型；生产流量往往不是这样。latency-predictor 流水线的按请求 TTFT 估计可能在这类场景中赋予它结构性优势，胜过任何"在一种类型上最优、在另一种上次优"的单一匹配配置，但我们尚未测量这一点。降低该流水线的部署开销（预测器训练、sidecar 部署、模型服务）也会降低把它部署为工作负载无关默认方案的成本；当前过滤器的 `ttftSource: latencyPredictor` 模式已经消除了其中配置层面的开销，不过预测器 sidecar 仍需部署和训练。

一个三级后续工作在这些实验之后已经发布：随路由器分发的[配置矩阵](https://github.com/llm-d/llm-d/blob/main/guides/recipes/router/calibration/configuration-matrix.md)记录了 llm-d 支持的各（模型、加速器、引擎）组合的实测 `peakPrefillThroughput` 值，因此在这些路径上的运维人员无需自己运行标定步骤。公式是可移植的；常量则不然，而这个矩阵正是常量如今的传播方式。

## 附录

### Configmaps

全部五种配置的完整调度器 configmap：[llm-d-benchmarking/configmaps](https://github.com/kaushikmitr/llm-d/tree/feature/vllm-optimized-benchmarking/llm-d-benchmarking/configmaps)

### 工作负载 YAML

完整工作负载规格：[llm-d-benchmarking/workloads](https://github.com/kaushikmitr/llm-d/tree/feature/vllm-optimized-benchmarking/llm-d-benchmarking/workloads)

### 完整结果表

各工作负载的结果表和分析脚本位于各工作负载的 `analysis/` 子目录中：[llm-d-benchmarking/workloads](https://github.com/kaushikmitr/llm-d/tree/feature/vllm-optimized-benchmarking/llm-d-benchmarking/workloads)

---

**标签**：[博客文章](https://llm-d.ai/blog/tags/blog "日常博客文章") · [调度](https://llm-d.ai/blog/tags/scheduling "面向 LLM 推理的请求调度与负载均衡") · [推理](https://llm-d.ai/blog/tags/inference "LLM 推理服务与优化") · [SIG-Benchmarking](https://llm-d.ai/blog/tags/sig-benchmarking "基准测试与性能分析特别兴趣小组")

[在 GitHub 上编辑此页](https://github.com/llm-d/llm-d/edit/main/website/blog/2026-08-17_sticky-until-saturated-token-aware-routing.mdx)
