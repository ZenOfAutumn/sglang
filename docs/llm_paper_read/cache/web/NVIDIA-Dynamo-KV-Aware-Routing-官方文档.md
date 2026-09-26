> 来源: <https://docs.nvidia.com/dynamo/kubernetes/kv-aware-routing/overview>
> 说明: 综述原链接 docs.nvidia.com/dynamo/latest/router/README.html 与仓库 docs/architecture/kv_cache_routing.md 均已迁移至此(现为 Fern 文档)
> 抓取时间: 2026-09-01 ｜ 格式: 官方原始 Markdown

> 若需本页的干净 Markdown 内容，请在此 URL 后追加 .md。完整文档索引参见 https://docs.nvidia.com/dynamo/llms.txt。包含 API 参考和 SDK 示例的完整内容参见 https://docs.nvidia.com/dynamo/llms-full.txt。

# Kubernetes 上的 KV 感知路由

Dynamo 在 Kubernetes 上支持两种 KV 感知请求路由拓扑。它们使用相同的路由概念，但将 worker 选择放在不同的组件中。请根据你的平台如何接入和治理流量来选择，而不是因为某种拓扑使用了不同的 KV 路由算法。

## 推荐

除非你的平台本身就需要 Kubernetes Gateway API，否则请使用 **Dynamo Frontend 路由**。它的组件更少，不需要安装 Gateway API，并且可以配合标准的 `DynamoGraphDeployment` 使用。

当 Gateway 必须掌控外部请求路径时——例如平台团队通过 Gateway API 统一管理入口、认证、限流、流量策略或网关级遥测——使用 **Gateway API 路由**。

| 选择 | 适用场景 | 额外基础设施 | 路由配置位置 |
|---|---|---|---|
| 使用 Dynamo Frontend | 客户端可以直接调用单个 Dynamo Frontend Service，并且你希望部署尽可能简单。 | 除常规 Dynamo 平台外无需其他。 | 带 `--router-mode kv` 的 Frontend。 |
| 将 GAIE 与 Dynamo 配合使用 | 流量必须经由 Kubernetes Gateway 进入，或需要共享平台管理的网关策略。 | Gateway API、GAIE、一个兼容的 Gateway 实现，以及一个 Dynamo EPP。 | EPP；worker 的 Frontend sidecar 使用 `--router-mode direct`。 |
| 将 GAIE 与原生 vLLM 配合使用 | 你已经在运行上游 vLLM pod，希望在不将整个集群迁移到 Dynamo worker 的情况下引入 Dynamo 的高级 KV 感知路由。 | Gateway API、GAIE、一个兼容的 Gateway 实现，以及独立的 Dynamo EPP 支持。 | EPP，配合直接的 vLLM worker 发现和 vLLM token 渲染。 |

Frontend 拓扑与 GAIE 拓扑是同一条请求路径上的两种备选方案。不要在 EPP 已经选定 worker 之后再让 Frontend 去选择 worker。

## Dynamo Frontend 路由

Dynamo Frontend 接收客户端请求，对其进行分词，为符合条件的 worker 打分，然后将请求转发到被选中的 worker。Router 相关的标志和环境变量应放在 Frontend 组件上。

在以下情况下选择此拓扑：

- 每个部署都可以暴露自己的 Frontend Service。
- 请求路径中不需要 Gateway API 策略。
- 你希望安装和运维的路由组件尽可能少。
- 由 Dynamo 同时负责 HTTP 请求处理和 worker 选择。

此拓扑在常规 Dynamo 平台上开箱即用，不需要任何 Gateway API 组件。参见[使用 Dynamo Frontend](/dynamo/kubernetes/kv-aware-routing/using-the-dynamo-frontend)，在 DGD 中启用 KV 感知的 worker 选择。

## GAIE 路由

Gateway API 路由将流量入口与 worker 选择分离。由 Kubernetes `Gateway` 接收请求，并在将请求转发到 worker pod 之前，请求 Dynamo Endpoint Picker Plugin（EPP）选择一个端点。

在以下情况下选择此拓扑：

- 你所在的组织已经将 Gateway API 作为标准的入口层。
- 认证、鉴权、限流、流量策略或遥测应当放在 Gateway 层。
- 多条模型路由需要共享一个由平台管理的地址和 listener。
- 平台团队管理 Gateway，应用团队管理 DGD 和 `HTTPRoute`。

Gateway API 路由需要单独安装。参见[安装 Gateway API Inference Extension](/dynamo/kubernetes/installation/gateway-api-inference-extension)，然后参见 [GAIE 与 Dynamo 配合使用](/dynamo/kubernetes/kv-aware-routing/using-gaie-with-dynamo)或 [GAIE 与原生 vLLM 配合使用](/dynamo/kubernetes/kv-aware-routing/vanilla-vllm-onramp)。

## 什么是 Dynamo EPP？

Dynamo **Endpoint Picker Plugin（EPP）**是一个实现 GAIE 端点选择协议的服务。Gateway 在转发推理请求之前会调用它。EPP 对请求进行分词，读取 Dynamo worker 和 KV 缓存的状态，为符合条件的端点打分，然后将选中的 worker 返回给 Gateway。

EPP 不是公开的推理端点，也不运行模型。当 DGD 中包含 `type: epp` 的组件时，Dynamo operator 会部署它。在这种拓扑中，worker 的 Frontend sidecar 使用 `--router-mode direct` 运行，因为 EPP 已经做出了路由决策。

完整的请求与状态流转参见
[Gateway API 路由架构](/dynamo/knowledge-base/kubernetes/kubernetes-operator/gateway-api-routing)。

## 共同的路由行为

两种拓扑都可以利用 worker 发布的 KV 缓存事件来考虑前缀重叠和活跃负载。在精确的 KV 事件不可用时，两者也都可以基于预测状态运行。选择算法和调优概念是共享的，只有承载它们的组件不同。

关于成本模型，参见[路由概念](/dynamo/knowledge-base/modular-components/router/routing-concepts)。关于共享的调优指导，参见[配置与调优](/dynamo/knowledge-base/modular-components/router/configuration-and-tuning)。
