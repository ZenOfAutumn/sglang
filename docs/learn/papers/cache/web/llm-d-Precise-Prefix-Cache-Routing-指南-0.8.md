> 来源: <https://llm-d.ai/docs/0.8/well-lit-paths/foundations/precise-prefix-cache-routing>
> 抓取时间: 2026-09-01 ｜ 格式: 网页转 Markdown

* [](/)
  * [Well-Lit Paths](/docs/0.8/well-lit-paths)
  * [基础](/docs/0.8/well-lit-paths/foundations)
  * 启用精确前缀缓存感知路由

版本: v0.8

本页目录

# 精确前缀缓存路由

模型服务器是关于其自身 GPU 和各内存层级上缓存内容的最准确的事实来源。vLLM、SGLang 和 NVIDIA TensorRT-LLM 会将每一次缓存变化作为事件发布出来；llm-d 订阅这个事件流，构建整个集群中驻留缓存块的近实时视图，并据此对请求进行打分。前缀亲和性分数会与标准的负载感知打分器组合使用，方式与 [Optimized Baseline](/docs/0.8/well-lit-paths/foundations/optimized-baseline) 路径类似。

KV 事件已经成为生态标准的底层机制，用于暴露准确的缓存状态——可复用的推理状态位于何处，以及它如何随时间变化。随着 KV 缓存编排日益复杂、智能体工作负载让前缀越来越长，缓存状态成为控制平面需要观测并据此行动的对象。同样的视图可以自然地扩展到：

  * 跨 GPU HBM、CPU DRAM、本地 NVMe 和共享存储的分层缓存跟踪；
  * 考虑显式提示词缓存放置和动态 KV 卸载的策略；
  * 面向集群级 KV 复用的缓存迁移与预取工作流；
  * 面向智能体模式的高级 KV 保留与淘汰策略；
  * 混合注意力模型，其中各层组（全量、滑动窗口、线性）独立进行淘汰。

## 部署​

有关 manifest 和分步部署说明，参见[精确前缀缓存路由指南](https://github.com/llm-d/llm-d/tree/main/guides/precise-prefix-cache-routing)。

## 架构​

职责划分非常直接：**模型服务器**在每次缓存变化时产生 KV 事件；**llm-d Router** 消费这些事件，为 pod 打分以做出更好的路由决策。两侧彼此解耦——模型服务器和 llm-d Router 的副本可以独立伸缩。

在 llm-d Router 内部：

  * 一个 **indexer** 消费事件流，为集群中驻留的每个块维护一份 `block key → pods` 的映射。
  * 一个 **scorer** 从输入确定性地推导出块键并查询索引。它返回每个候选 pod 已缓存的最长连续前缀，并按存储层级加权。

事件通过 **pod 发现**机制经 ZMQ 从模型服务器 pod 流向 llm-d Router：每个模型服务器 pod 绑定自己的 ZMQ socket，每个 llm-d Router 副本独立订阅每个 pod。所有 Router 副本会收敛到相同的索引，从而开箱即用地支持双活（active-active）高可用。参考指南默认在一个 Service 后部署两个 Router 副本，小规模集群可缩减到一个。

## 延伸阅读​

完整架构参见 [KV-Cache Indexer](/docs/0.8/architecture/advanced/kv-management/kv-indexer)
