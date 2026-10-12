> 来源: <https://cloud.google.com/blog/products/containers-kubernetes/gke-inference-gateway-prefix-caching-accelerates-ai-inference>
> 抓取时间: 2026-09-01 ｜ 格式: 网页转 Markdown

容器与 Kubernetes

#

报告：GKE Inference Gateway 使 AI 响应速度最高提升 92%

2026 年 6 月 10 日

  * [](https://x.com/intent/tweet?text=Report:%20GKE%20Inference%20Gateway%20delivers%20up%20to%2092%%20faster%20AI%20responses%20@googlecloud&url=https://cloud.google.com/blog/products/containers-kubernetes/gke-inference-gateway-prefix-caching-accelerates-ai-inference)
  * [](https://www.linkedin.com/shareArticle?mini=true&url=https://cloud.google.com/blog/products/containers-kubernetes/gke-inference-gateway-prefix-caching-accelerates-ai-inference&title=Report:%20GKE%20Inference%20Gateway%20delivers%20up%20to%2092%%20faster%20AI%20responses)
  * [](https://www.facebook.com/sharer/sharer.php?caption=Report:%20GKE%20Inference%20Gateway%20delivers%20up%20to%2092%%20faster%20AI%20responses&u=https://cloud.google.com/blog/products/containers-kubernetes/gke-inference-gateway-prefix-caching-accelerates-ai-inference)
  * [](mailto:?subject=Report:%20GKE%20Inference%20Gateway%20delivers%20up%20to%2092%%20faster%20AI%20responses&body=Check%20out%20this%20article%20on%20the%20Cloud%20Blog:%0A%0AReport:%20GKE%20Inference%20Gateway%20delivers%20up%20to%2092%%20faster%20AI%20responses%0A%0AGKE%20Inference%20Gateway%20uses%20prefix%20caching%20and%20other%20routing%20algorithms%20to%20land%20requests%20on%20the%20best%20accelerator%20for%20the%20job.%0A%0Ahttps://cloud.google.com/blog/products/containers-kubernetes/gke-inference-gateway-prefix-caching-accelerates-ai-inference)

##### Bob Tian

软件工程师

##### Susan Wu

对外产品经理

##### 立即试用 Gemini Enterprise

工作场所 AI 的前门

[立即试用 ](https://business.gemini.google/?utm_source=cloud.google.com/blog&utm_medium=et&utm_campaign=FY26-Q2-GLOBAL-GLO27877-physicalevent-er-next26-mc-105752)

随着生成式 AI 从实验性试点走向大规模生产环境，基础设施的效率成为终极差异化因素。要最大化基础设施效益并尽量减少昂贵的加速器空闲时间，一种方式是利用 [Google Kubernetes Engine (GKE) Inference Gateway](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/about-gke-inference-gateway)，它基于模型服务器的实时指标智能地路由生成式 AI 工作负载。

传统的、朴素的轮询（round-robin）负载均衡经常触发昂贵的加速器重复计算并使用户延迟飙升；与之不同，这个 [GKE Gateway](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/gateway-api) 的原生扩展利用了[前缀缓存](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/about-gke-inference-gateway)和[模型感知路由](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/about-gke-inference-gateway)等高级能力。通过确保请求准确落在已经准备好立即处理它的加速器上，GKE 彻底改变了你服务大语言模型（LLM）的方式，实现出色的硬件利用率和极快的响应时间。

事实上，根据一份[独立基准测试报告](https://www.principledtechnologies.com/Google/GKE-Inference-Gateway-study-0526.pdf)，**GKE Inference Gateway 相比排名第二的托管 Kubernetes 服务，吞吐量高出 15.7%，等待时间缩短 92.8%，token 间延迟降低 62.6%**。这样的性能让基于 LLM 的应用从迟缓昂贵变为快速且达到生产级水准。

这一性能表现与 [Snap](https://www.snap.com/) 使用 GKE Inference Gateway 的实际体验相符。

"在 Snap，我们正在将 llm-d 集成到我们的生产 AI 基础设施中，以支撑大规模的高性能推理。通过采用前缀缓存感知路由，我们实现了高达 75-80% 的前缀缓存命中率。我们非常欣赏 llm-d 的开源属性，因为它能够与我们基于 Envoy 的 Service Mesh 无缝集成。" \- Vinay Kola，Snap Inc. 软件工程高级经理

在本篇博客中，我们将结合示例深入介绍 GKE Inference Gateway 的前缀缓存，并提供关于其基准测试结果的更多细节。让我们开始吧。

### **低延迟 AI 的秘诀：前缀缓存**

**前缀缓存**通过存储长且重复的提示词前缀的 KV 缓存（激活状态）来优化 LLM 性能。当连续的用户请求共享相同的系统指令、上下文或文档时，模型可以完全跳过对这些 token 的重复处理。GKE Inference Gateway 读取传入请求的前缀，并将其匹配到内存中已经持有该数据的特定 pod。这消除了 GPU 和 TPU 上的"思考"开销，将繁重的推理循环转变为近乎即时的响应。

**用例 1：基于检索增强生成（RAG）的文档与代码库问答**

在查询海量企业资料库时，你可以使用 RAG 将整套文档固定为静态缓存前缀，从而在不增加任何延迟的情况下为 LLM 的回答提供依据。

GKE Inference Gateway 不会强迫 LLM 为每一个用户问题重新阅读数千行 API 参考或企业 wiki，而是将查询路由到已经在 KV 缓存中预热了该特定上下文的 pod。LLM 只需计算用户简短的动态问题，完全绕过昂贵的文档重复评估。

加载中...

[STATIC PREFIX - STAYS IN CACHE] You are an expert AI assistant specializing in technical documentation. Below is the complete API documentation for our software platform. Use this context to answer the user's questions accurately. If the answer cannot be found in the documentation, say "I cannot find that in the provided context." <documentation> [10,000+ words of API reference documentation, endpoints, error codes, etc.] </documentation> [DYNAMIC SUFFIX - CHANGES PER REQUEST] User Question: How do I handle a 429 rate limit error using the Python SDK?

图 1：软件排障场景的提示词结构拆解，展示了被缓存的静态前缀和每次请求都会变化的动态后缀

**用例 2：多轮对话**

你还可以使用前缀缓存，在数千个并发会话中维持客服交互而不叠加计算成本。做法是将固定不变的系统人设和核心业务规则直接缓存在 LLM 服务器上。

在企业级聊天架构中，基础系统提示词和参考表格在数百万次客户交互中保持完全一致。GKE Inference Gateway 使用上下文感知路由来处理这些多轮对话，绕过重复的 token 处理，让你的聊天机器人即使在流量峰值下也能保持极高的响应速度。

加载中...

[STATIC PREFIX - STAYS IN CACHE] -System Persona: You are "FinBot", a helpful, empathetic, and compliant virtual assistant for ABC Banking Solutions. You must strictly adhere to the following rules: 1. Never provide concrete investment advice. 2. Always verify if the user is asking about checking or savings. 3. Keep your answers under 3 sentences. 4. If a user is angry, offer to connect them to a human manager. Here is the current interest rate table for May 2026: \- Savings: 4.2% APR \- Checking: 0.5% APR \- CD (12-month): 5.1% APR [DYNAMIC SUFFIX - CHANGES PER REQUEST] User: Hi, I'm trying to figure out how much I'd make if I locked away $10,000 for a year?

图 2：以上提示词展示了银行聊天机器人交互中的静态前缀和随每次请求变化的动态部分

### **GKE 优于其他托管 Kubernetes 方案**

为验证这些架构优势，Principled Technologies 近期发布了一份独立的[基准测试报告](https://www.principledtechnologies.com/Google/GKE-Inference-Gateway-study-0526.pdf)，将（配备 GKE Inference Gateway 的）GKE 与使用传统轮询 HTTP 负载均衡的标准第三方托管 Kubernetes 服务进行了对比。

测试在 Llama 3.1 8B Instruct 共享前缀工作负载上进行，硬件完全相同（8 块 NVIDIA A100 40GB GPU），结果显示两种 Kubernetes 服务之间存在巨大的性能差距。**GKE 不仅仅是获胜，它在三个关键指标上彻底重新定义了推理效率：**

  * **更高的吞吐量：** 每秒处理的 token 数多 15.7%，可以支撑更高的请求容量，或在相同工作负载下减少硬件需求

  * **首 token 时间（TTFT）大幅缩短：** 等待时间缩短 92.8%，在交互式场景中带来明显更快的感知响应启动

  * **更低的 token 间延迟（ITL）：** 降低 62.6%，在首个 token 之后带来更流畅、更快的 token 流式输出

图 3：在 Llama 3.1-8B Instruct LLM 的共享前缀用例中，配备 GKE Inference Gateway 的 GKE 与第三方托管 Kubernetes 服务的平均延迟（按每输出 token 归一化的时间）。两种方案使用相同硬件。来源：Principled Technologies

|  **GKE** |  **第三方托管** **Kubernetes 服务** |  **GKE 优势**
---|---|---|---
**平均输出** **token 吞吐量** |  每秒 7,169.21 个输出 token |  每秒 6,042.05 个输出 token |  输出 token 吞吐量高 15.7%
**平均首 token** **时间（TTFT）** |  188.36 ms |  2624.73 ms |  TTFT 低 92.8%
**平均 token 间** **延迟（ITL）** |  30.20 ms |  81.03 ms |  ITL 低 62.6%

图 4：与使用标准 HTTP 负载均衡的第三方托管 Kubernetes 服务相比，配备 GKE Inference Gateway 的 GKE 提供了更卓越的 AI 推理性能。

### 准备好加速你的生成式 AI 推理工作负载了吗？

无论你部署的是实时客服智能体、动态编码助手，还是亚秒级欺诈检测模型等推理工作负载，基础设施延迟都决定着用户体验。通过确保共享的提示词前缀几乎 100% 命中活跃缓存，GKE Inference Gateway 将你的 LLM 从迟缓、昂贵的推理引擎转变为快速、资本高效、生产级的强大引擎。

准备好探索 GKE Inference Gateway 能为你的生成式 AI 工作负载带来的性能优势了吗？在[这里](https://www.principledtechnologies.com/Google/GKE-Inference-Gateway-study-0526.pdf)获取完整的基准测试报告，并观看这个讲解[视频](https://youtu.be/RXX-LouimPY?si=dPGbP91TakSonOq9)了解更多。

* * *

_特别感谢 Principled Technologies 高级性能架构师 Dan Sullivan。_

发布于

  * [容器与 Kubernetes](https://cloud.google.com/blog/products/containers-kubernetes)
  * [网络](https://cloud.google.com/blog/products/networking)
  * [AI 与机器学习](https://cloud.google.com/blog/products/ai-machine-learning)
  * [AI 基础设施](https://cloud.google.com/blog/topics/ai-infrastructure)
  * [GKE](https://cloud.google.com/blog/products/kubernetes-engine)

##### 相关文章

[AI 基础设施 · 八月 AI 基础设施与编排新动态 · 作者 Alex Barrett • 13 分钟阅读](https://cloud.google.com/blog/topics/ai-infrastructure/whats-new-in-ai-infrastructure-this-month)

[容器与 Kubernetes · 将 gVisor 沙箱引入分布式 Ray 集群 · 作者 Andrew Sy Kim • 3 分钟阅读](https://cloud.google.com/blog/products/containers-kubernetes/gvisor-sandboxes-for-ray-clusters-on-gke)

[网络 · GKE 中的 ClusterNetworkPolicy：在微服务的控制与自治之间取得平衡 · 作者 Srini Jasti • 4 分钟阅读](https://cloud.google.com/blog/products/networking/new-clusternetworkpolicy-in-gke)

[容器与 Kubernetes · 少花钱多办事：GKE 如何将你的单智能体成本降低 75% · 作者 Drake Williams • 6 分钟阅读](https://cloud.google.com/blog/products/containers-kubernetes/reduce-your-agents-costs-with-gke-agent-sandbox)
