> 来源: <https://llm-d.ai/docs/0.8/architecture/advanced/kv-management/prefix-cache-aware-routing>
> 抓取时间: 2026-09-01 ｜ 格式: 网页转 Markdown
> 综述原链接 docs/guide/Installation/precise-prefix-cache-aware 已 404,现为版本化路径

* [](/)
  * [概念（架构）](/docs/0.8/architecture)
  * 能力
  * [KV 缓存管理](/docs/0.8/architecture/advanced/kv-management)
  * 前缀缓存感知路由

版本: v0.8

本页目录

# 前缀缓存感知路由

前缀缓存感知路由是由 **llm-d Router**（具体通过其 **Endpoint Picker (EPP)** 组件）管理的一项核心技术，用于降低尾部延迟并提升吞吐量。通过将请求路由到已经持有该提示词前缀对应键值（KV）缓存的模型服务器副本，系统避免了冗余的"预填充（prefill）"计算，同时节省时间和加速器（GPU/TPU）资源。该技术要求底层模型服务器支持跨请求的 KV 缓存，例如 vLLM 的[自动前缀缓存](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)特性。

llm-d 为这一能力提供了两种不同的实现，以满足不同的运维需求和精度要求。

* * *

## 1\. 近似实现​

近似实现的设计目标是轻量，除标准的 EPP 部署外不需要任何外部依赖。

### 组件​

  * [**`approx-prefix-cache-producer`**](https://github.com/llm-d/llm-d-router/tree/main/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix)（DataProducer 插件）
  * [**`prefix-cache-scorer`**](https://github.com/llm-d/llm-d-router/tree/main/pkg/epp/framework/plugins/scheduling/scorer/prefix)（Scorer 插件）

### 工作原理​

  1. **近似估算**：由于 EPP 本身不包含 tokenizer，它使用字符与 token 的比例来近似估算 token。
  2. **哈希**：`approx-prefix-cache-producer` 将传入的提示词切分为固定大小的块（例如以字符近似的 16 个 token），并构建滚动哈希链。
  3. **本地索引**：EPP 在内存中维护一个 LRU 索引，记录哪些前缀哈希最近被发送到了哪些 Pod。
  4. **打分**：`prefix-cache-scorer` 读取匹配信息，并根据匹配块数与提示词总块数之比给出分数。
  5. **学习**：在做出路由决策后，EPP 更新其本地索引，假定被选中的 Pod 现在将持有该前缀。

### 优缺点​

  * **优点**：极其轻量；不需要 tokenizer sidecar；不需要与模型服务器建立网络连接（ZMQ）；由于不要求模型服务器上报 KV 缓存事件，因此无需显式的模型服务器集成。
  * **缺点**：可能与模型服务器的实际状态产生偏差（例如某个 Pod 因内存压力淘汰了某个前缀）；精度低于基于 token 的匹配。

* * *

## 2\. 精确实现​

精确实现通过利用真实的 token 数据和来自模型服务器的实时状态更新，提供 100% 的精度。

### 组件​

  * [**`token-producer`**](https://github.com/llm-d/llm-d-router/tree/main/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer)（DataProducer 插件）
  * [**`precise-prefix-cache-producer`**](https://github.com/llm-d/llm-d-router/tree/main/pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache)（DataProducer 插件）— 拥有 KV 块索引
  * [**`prefix-cache-scorer`**](https://github.com/llm-d/llm-d-router/tree/main/pkg/epp/framework/plugins/scheduling/scorer/prefix)（Scorer 插件）— 通过 `prefixMatchInfoProducerName: precise-prefix-cache-producer` 使用该 producer 的匹配信息进行打分
  * **KV-Cache Indexer**（EPP 数据层组件）

### 工作原理​

  1. **精确分词**：`token-producer` 插件将提示词发送到 vLLM 的 HTTP render 端点（`/v1/completions/render`）——通常是 EPP pod 中的一个 `vllm launch render` sidecar（回环访问）或一个共享的 render Service——以获取精确的 Token ID。（旧版基于 gRPC-over-UDS 的 tokenizer 后端已废弃。）
  2. **实时事件**：模型服务器（如 vLLM）被配置为在其内部 KV 缓存发生变化（块被添加或淘汰）时通过 ZeroMQ（ZMQ）发出 `KVEvents`。
  3. **全局索引**：**KV-Cache Indexer** 订阅这些事件，维护一个精确且全局一致的视图，准确记录哪些 token 块驻留在哪些 Pod 上。
  4. **精确匹配**：`prefix-cache-scorer` 读取 `precise-prefix-cache-producer` 的匹配信息，根据每个候选 pod 在该全局索引中驻留的精确 Token-ID 前缀长度对其打分。
  5. **推测性索引**：为了弥合路由决策与后续 `KVEvent` 到达之间的"盲区"，producer 可以在路由完成后立即主动向索引中添加"推测性"条目。

### 优缺点​

  * **优点**：100% 精度；能应对复杂的缓存淘汰策略；原生支持 Prefill/Decode 分离（通过识别需要传输的具体块）。
  * **缺点**：需要额外的基础设施（vLLM render 端点、ZMQ 连接）；资源开销略高；要求模型服务器支持发出 KV 缓存事件。

* * *

## 对比总结​

特性| 近似实现| 精确实现
---|---|---
**精度**|  启发式（基于字符）| 100%（基于 token）
**状态来源**|  EPP 本地假设| 来自模型服务器的实时 `KVEvents`
**依赖**|  无| vLLM render 端点、ZMQ
**适用场景**|  简单、同构的工作负载| 复杂、大规模的生产服务
**P/D 分离支持**|  基础| 高级/原生

### 与 KV 缓存管理的组合​

两种实现都是 llm-d 中更广泛的 **KV 缓存管理**生态的一部分。近似实现是自包含的，而精确实现依赖于 [KV-Cache Indexer](/docs/0.8/architecture/advanced/kv-management/kv-indexer)，并可以与 [KV Offloading](/docs/0.8/architecture/advanced/kv-management/kv-offloader) 协同工作，跨加速器与主机内存边界管理缓存状态。
