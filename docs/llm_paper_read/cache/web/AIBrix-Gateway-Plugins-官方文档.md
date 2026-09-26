> 来源: <https://aibrix.readthedocs.io/latest/features/gateway-plugins.html>
> 抓取时间: 2026-09-01 ｜ 格式: 网页转 Markdown

# 网关路由（Gateway Routing）#

AIBrix 网关基于 Envoy Gateway 构建，是所有 LLM 推理请求的统一入口。它负责动态模型发现、请求路由、限流以及响应流式传输。若想深入了解内部设计，请参阅 [AIBrix Router](../designs/aibrix-router.html) 架构指南。

## 动态路由#

当部署一个模型或 LoRA 适配器时，相应的控制器会自动创建一个 `HTTPRoute` 对象。网关会动态发现这些路由并转发传入请求，无需任何手动配置。

首先，获取 Envoy 代理的外部 IP 和端口：
[code]
    NAME                                     TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)                                   AGE
    envoy-aibrix-system-aibrix-eg-903790dc   LoadBalancer   10.96.239.246   101.18.0.4    80:32079/TCP                              10d
    envoy-gateway                            ClusterIP      10.96.166.226   <none>        18000/TCP,18001/TCP,18002/TCP,19001/TCP   10d

[/code]

在大多数 Kubernetes 环境中，`LoadBalancer` 默认即受支持。获取外部 IP：
[code]
    LB_IP=$(kubectl get svc/envoy-aibrix-system-aibrix-eg-903790dc -n envoy-gateway-system -o=jsonpath='{.status.loadBalancer.ingress[0].ip}')
    ENDPOINT="${LB_IP}:80"

[/code]

在发送流量之前，请确认 `HTTPRoute` 的状态为 `Accepted`：
[code]
    $ kubectl get httproute -A
    NAMESPACE       NAME                                  HOSTNAMES   AGE
    aibrix-system   aibrix-reserved-router                            17m # 保留路由
    aibrix-system   deepseek-r1-distill-llama-8b-router               14m # 为每个模型部署创建
    ....

[/code]
[code]
    $ kubectl describe httproute deepseek-r1-distill-llama-8b-router -n aibrix-system
    Name:         deepseek-r1-distill-llama-8b-router
    Namespace:    aibrix-system
    Labels:       <none>
    Annotations:  <none>
    API Version:  gateway.networking.k8s.io/v1
    Kind:         HTTPRoute
    Metadata:
      Creation Timestamp:  2025-02-16T17:56:03Z
      Generation:          1
      Resource Version:    2641
      UID:                 2f3f9620-bf7c-487a-967e-2436c3809178
    Spec:
      Parent Refs:
        Group:      gateway.networking.k8s.io
        Kind:       Gateway
        Name:       aibrix-eg
        Namespace:  aibrix-system
      Rules:
        Backend Refs:
          Group:
          Kind:       Service
          Name:       deepseek-r1-distill-llama-8b
          Namespace:  default
          Port:       8000
          Weight:     1
        Matches:
          Headers:
            Name:   model
            Type:   Exact
            Value:  deepseek-r1-distill-llama-8b
          Path:
            Type:   PathPrefix
            Value:  /
        Timeouts:
          Request:  120s
    Status:
      Parents:
        Conditions:
          Last Transition Time:  2025-02-16T17:56:03Z
          Message:               Route is accepted
          Observed Generation:   1
          Reason:                Accepted
          Status:                True
          Type:                  Accepted
          Last Transition Time:  2025-02-16T17:56:03Z
          Message:               Resolved all the Object references for the Route
          Observed Generation:   1
          Reason:                ResolvedRefs
          Status:                True
          Type:                  ResolvedRefs
        Controller Name:         gateway.envoyproxy.io/gatewayclass-controller
        Parent Ref:
          Group:      gateway.networking.k8s.io
          Kind:       Gateway
          Name:       aibrix-eg
          Namespace:  aibrix-system
    Events:           <none>

[/code]

从 v0.5.0 开始，每个模型的 `HTTPRoute` 在创建时会带有一组默认路径前缀（例如 `/v1/chat/completions`、`/v1/completions`）。若要暴露额外的自定义路径，请在 `Deployment`、`ModelAdapter` 或 `RayClusterFleet` 清单中添加 `model.aibrix.ai/model-router-custom-paths` 注解：
[code]
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      labels:
        model.aibrix.ai/name: deepseek-r1-distill-llama-8b
        model.aibrix.ai/port: "8000"
      annotations:
        model.aibrix.ai/model-router-custom-paths: /version,/score # 逗号分隔；空格和空项会被忽略
      name: deepseek-r1-distill-llama-8b
      namespace: default

[/code]

请求体中的模型名称必须与部署中的 `model.aibrix.ai/name` 标签一致：
[code]
    curl -v http://${ENDPOINT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-r1-distill-llama-8b",
        "messages": [{"role": "user", "content": "Say this is a test!"}],
        "temperature": 0.7
    }'

[/code]

注意（Attention）

AIBrix 会向互联网暴露一个公共端点。请启用身份认证来保护它。对于 vLLM 后端，可以传入 `--api-key` 参数或设置 `VLLM_API_KEY` 环境变量，从而要求在 `Authorization` 头中携带 API key。详情参见 [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server)。

启用身份认证后，每个请求都需要携带 `Authorization` 头：
[code]
    curl -v http://${ENDPOINT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer any_key" \
    -d '{
        "model": "deepseek-r1-distill-llama-8b",
        "messages": [{"role": "user", "content": "Say this is a test!"}],
        "temperature": 0.7
    }'

[/code]

## 路由策略#

路由策略控制网关如何为每个请求选择目标 Pod。它可以通过 `routing-strategy` 请求头按请求设置，也可以通过 `ROUTING_ALGORITHM` 环境变量全局设置。

部分策略依赖于从 Prometheus HTTP API（PromQL）查询的指标。配置方法参见 [Prometheus API Access](../production/observability.html#prometheus-api-access)。

### 通用负载均衡#

  * `random`：路由到随机选择的 Pod。适合作为基线，或在所有 Pod 等价时使用。

  * `least-request`：路由到在途（in-flight）请求数最少的 Pod。

  * `least-busy-time`：路由到累计繁忙处理时间最少的 Pod。

  * `least-latency`：路由到平均处理延迟最低的 Pod。

  * `least-kv-cache`：路由到 KV cache 占用最小（VRAM 使用最少）的 Pod。

  * `least-gpu-cache`：路由到 GPU 缓存利用率最低的 Pod。

  * `least-utilization`：路由到总体利用率得分最低的 Pod。

  * `load-balance`：容量感知的加权最少请求路由。将每个 Pod 的得分计算为 `running_requests / drain_rate`（排空时间/pending time），其中 `drain_rate` 是观测到的请求完成速率。选择 pending time 最低的 Pod，并使用 GPU+CPU KV-cache 合计使用量最少来打破平局（若平局 Pod 的缓存指标不可用则回退为随机选择）。当指标不可用时回退到统一容量（`drain_rate = 1`）。它的负载失衡门控（load-imbalance gate）——在负载严重倾斜时将候选限制为负载最低的 Pod——由网关在实际路由每个请求的策略之前集中应用，而不仅仅是在选择了 `load-balance` 本身时（参见 `pkg/plugins/gateway/ENV_VARS.md`）。

  * `throughput`：路由到累计处理加权 token 总数最少的 Pod，偏向负载不足的 Pod。

  * `power-of-two`：应用二选一（power-of-two-choices）算法——随机采样两个 Pod 并选择其中较优者。

### KV-cache 感知#

  * `prefix-cache`：路由到已持有与请求提示词前缀相匹配的 KV cache 的 Pod，在可配置的标准差阈值内选择前缀匹配最佳的 Pod。它纯粹关注前缀缓存——本身并不基于集群级负载失衡进行门控（网关会在它之前集中应用该门控；参见上文的 `load-balance`）。支持两种模式：标准模式（本地哈希表）和 KV 同步模式（通过 `AIBRIX_PREFIX_CACHE_KV_EVENT_SYNC_ENABLED=true` 启用的实时分布式索引）。

  * `prefix-cache-preble`：同时考虑前缀缓存命中和 Pod 负载进行路由，基于 [Preble: Efficient Distributed Prompt Scheduling for LLM Serving](https://arxiv.org/abs/2407.00023)。

### 公平性#

  * `vtc-basic`：使用一个在按用户 token 公平性与 Pod 利用率之间取得平衡的混合得分进行路由。是 Virtual Token Counter（VTC，虚拟令牌计数器）算法的简化变体。背景资料参见 [VTC-artifact](https://github.com/Ying1123/VTC-artifact)。

### SLO 感知#

  * `slo`：在感知每请求服务级别目标（SLO）的前提下进行路由。

  * `slo-pack-load`：SLO 感知路由，将负载集中到更少的 Pod 上。

  * `slo-least-load`：SLO 感知路由，将负载分散到负载最低的 Pod。

  * `slo-least-load-pulling`：`slo-least-load` 的变体，直接拉取指标而非使用缓存快照。

### 专用策略#

  * `pd`：prefill-decode 分离（disaggregation）路由。将处理拆分到专用的 prefill Pod 与 decode Pod，以优化端到端延迟。
[code] curl -v http://${ENDPOINT}/v1/chat/completions \
        -H "routing-strategy: pd" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "your-model-name",
            "messages": [{"role": "user", "content": "Say this is a test!"}],
            "temperature": 0.7
        }'

[/code]

  * `session-affinity`：会话保持（粘性会话）路由。将目标 Pod 的地址（`IP:Port`）以 base64 编码写入 `x-session-id` 响应头。后续携带该头的请求会被路由到同一个 Pod。如果该 Pod 已不可用，网关会透明地故障转移到新 Pod 并签发新的会话 ID。

工作原理：

    * 首次请求（不带 `x-session-id`）时，网关选择一个就绪的 Pod，并返回 `x-session-id` 响应头。

    * 客户端存储该头，并在后续请求中重新发送（例如在多轮对话中）。

    * 网关解码会话 ID 以恢复原始 Pod 地址，并路由到该 Pod。

    * 如果该 Pod 不可用（被缩容或驱逐），则故障转移到新 Pod 并签发新的会话 ID。

注意（Note）

`x-session-id` 只编码网络位置。它不是安全令牌，绝不能用于认证或授权。

调用方也可以改为发送一个稳定的、不透明的 `x-aibrix-session-key` 值。网关会将该值一致性地映射到一个就绪的 Pod，而不暴露后端地址。该 key 仅在 `routing-strategy` 为 `session-affinity` 时使用，且绝不能用于认证或授权。
[code] curl -v http://${ENDPOINT}/v1/chat/completions \
    -H "routing-strategy: session-affinity" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "your-model-name",
        "messages": [{"role": "user", "content": "Say this is a test!"}],
        "temperature": 0.7
    }'

[/code]

### 自动混合的容量感知#

这种自动混合（auto-blend）**默认启用**——无需主动开启。上述所有策略，除专属策略（`pd`、`slo`/`slo-*`）和显式单独选择的 `load-balance` 之外，都会在幕后静默混入 `load-balance` 的容量感知评分——当所选策略本身不按请求数路由时，还会混入 `least-request`，以保证多端口/数据并行 Pod 的路由在混合模式下正常工作。调用方对此完全无感：`ctx.Algorithm`、响应头以及 `Validate()` 仍然只反映被请求的那个策略。这可以防止任何单一策略在上文所述负载失衡门控之外仍将流量导向已经过热的 Pod。两个混合权重均默认为 `1`（参见 `pkg/plugins/gateway/ENV_VARS.md`）；设置 `AIBRIX_ROUTING_AUTO_BLEND_LOAD_BALANCE_WEIGHT=0` 可将其禁用。

要针对单个请求覆盖策略，可通过 `routing-strategy` 请求头传入上述任意值：
[code]
    curl -v http://${ENDPOINT}/v1/chat/completions \
    -H "routing-strategy: least-request" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "your-model-name",
        "messages": [{"role": "user", "content": "Say this is a test!"}],
        "temperature": 0.7
    }'

[/code]

## 限流（Rate Limiting）#

网关支持基于每分钟请求数（RPM）和每分钟 token 数（TPM）的按用户限流。传入带有唯一标识符的 `user` 请求头，即可对该客户端启用限流。

关于用户管理的详情，参见 [User Management 文档](https://github.com/vllm-project/aibrix/blob/main/pkg/metadata/README.md)。
[code]
    curl -v http://${ENDPOINT}/v1/chat/completions \
    -H "user: your-user-id" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer any_key" \
    -d '{
        "model": "your-model-name",
        "messages": [{"role": "user", "content": "Say this is a test!"}],
        "temperature": 0.7
    }'

[/code]

注意（Note）

如果不需要限流，可以省略 `user` 请求头。

## 外部过滤器（External Filter）#

`external-filter` 请求头使用 Kubernetes 标签选择器（label selector）语法，在路由策略做出最终选择之前收窄候选 Pod 池。它在路由之前求值，且仅当同时设置了 `routing-strategy` 时才生效。

支持的选择器形式：

  * `key=value`

  * `key in (a, b)`

  * `key!=value`

  * 逗号分隔的选择器列表

完整语法参见 [Kubernetes 标签选择器参考](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/)。
[code]
    curl -v http://${ENDPOINT}/v1/completions \
       -H "Content-Type: application/json" \
       -H "routing-strategy: random" \
       -H "external-filter: environment=production,tier=frontend" \
       -d '{
             "model": "deepseek-r1-distill-llama-8b",
             "prompt": "San Francisco is a",
             "max_tokens": 128,
             "temperature": 0
           }'

[/code]

注意（Note）

  1. 过滤发生在路由策略**之前**，并不会改变策略认为最优的 Pod。

  2. `external-filter` 仅在设置了 `routing-strategy` 时才生效。

  3. 它在 `model.aibrix.ai/name` 已筛选出的 Pod 基础上进一步收窄范围。

  4. 如果过滤器排除了所有 Pod，请求将以 `no ready pods for routing` 失败。

  5. `external-filter` 是可选的。省略时不会应用额外过滤。

## 请求头参考（Headers Reference）#

本节介绍请求处理中用于路由、调试和限流的自定义请求头。

### 目标与通用请求头#

请求头名称 | 方向 | 说明
---|---|---
`request-id` | 响应 | 与客户端请求关联的唯一请求 ID。可用于关联日志。
`x-went-into-req-headers` | 后端请求、响应 | 指示请求头是否被正确处理。用于调试请求头解析问题。
`target-pod` | 后端请求、响应 | 标识路由算法选中的目标：后端请求中携带其 `IP:port` 地址，响应中则报告其 Pod 名称。
`target-pod-ip` | 响应 | 路由算法选中的目标 Pod 的 `IP:port` 地址。
`routing-strategy` | 请求、后端请求、响应 | 为请求选择路由策略，将实际采用的策略转发给后端，并在响应中报告。
`external-filter` | 请求 | 用于在路由前进一步过滤候选 Pod 的标签选择器表达式。
`model` | 后端请求 | 在未显式选择路由策略时注入后端请求的模型名称。
`config-profile` | 请求 | 为请求选择模型配置 profile。
`x-aibrix-config-profile` | 响应 | 当 `config-profile` 为 `auto` 时报告实际选中的具体 profile。
`x-session-id` | 请求、响应 | `session-affinity` 路由使用的会话标识符。响应中的值应在后续请求中回传以保持会话保持。
`x-aibrix-session-key` | 请求 | 由调用方持有的不透明会话 key，供 `session-affinity` 路由使用。
`traceparent` | 请求、后端请求 | 在网关发起的请求（包括 prefill-decode 路由）中传播的 W3C Trace Context。
`prefill-target-pod` | 响应 | `pd` 路由选中的 prefill Pod 的名称。
`prefill-target-pod-ip` | 响应 | `pd` 路由选中的 prefill Pod 的 IP 地址。

### 路由与错误调试请求头#

请求头名称 | 说明
---|---
`x-error-user` | 标识由错误的用户输入导致的错误。
`x-error-routing` | 指示路由失败，例如无法选出目标 Pod。
`x-error-response-unmarshal` | 表示响应体无法解析，通常由内部错误导致。
`x-error-response-unknown` | 未能识别具体原因时的通用错误头。
`x-error-request-body-processing` | 标记请求体解析失败，例如无效的 JSON。
`x-error-no-model-in-request` | 指示请求中未指定模型。
`x-error-no-model-backends` | 指示所请求的模型存在，但没有活跃的后端 Pod。
`x-error-invalid-routing-strategy` | 指示指定了不受支持的路由策略。

### 流式传输请求头#

请求头名称 | 说明
---|---
`x-error-streaming` | 表示响应流式传输期间发生错误。
`x-error-stream` | 指示请求体中的 `stream` 值不正确。
`x-error-no-stream-options-include-usage` | 指示流式响应中是否包含用量统计信息。

### 限流请求头#

请求头名称 | 说明
---|---
`x-update-rpm` | 表示 RPM（每分钟请求数）计数器已成功更新。
`x-update-tpm` | 表示 TPM（每分钟 token 数）计数器已成功更新。
`x-error-rpm-exceeded` | 表示请求超出了允许的 RPM 限制。
`x-error-tpm-exceeded` | 表示请求超出了允许的 TPM 限制。
`x-error-incr-rpm` | 递增 RPM 计数器时遇到错误。
`x-error-incr-tpm` | 递增 TPM 计数器时遇到错误。

### 调试指南#

  1. **识别错误头** —— 检查 `x-error-routing`、`x-error-user`、`x-error-response-unmarshal` 和 `x-error-response-unknown` 以定位根本原因。对于请求体问题，检查 `x-error-request-body-processing` 和 `x-error-no-model-in-request`。

  2. **验证路由与模型分配** —— 确认 `target-pod` 已被设置。如果出现 `x-error-no-model-in-request` 或 `x-error-no-model-backends`，请确认请求体包含有效的模型名称，且该模型有活跃的后端 Pod。如果出现 `x-error-invalid-routing-strategy`，请确认策略名称受支持。

  3. **诊断流式传输问题** —— 检查 `x-error-streaming` 和 `x-error-stream`。如果流式响应缺少用量统计信息，请检查 `x-error-no-stream-options-include-usage`。

  4. **排查限流问题** —— 如果请求被拒绝，检查 `x-error-rpm-exceeded` 或 `x-error-tpm-exceeded`。如果计数器更新失败，查看 `x-error-incr-rpm` 或 `x-error-incr-tpm`。更新成功由 `x-update-rpm` 和 `x-update-tpm` 指示。

确认所有网关对象的 `status.conditions == Accepted`：
[code]
    kubectl describe gatewayclass -n aibrix-system

    kubectl describe gateway -n aibrix-system

    kubectl describe envoypatchpolicy -n aibrix-system

    # 检查所有对象
    kubectl describe envoyextensionpolicy -n aibrix-system

    # 检查所有对象
    kubectl describe httproute -n aibrix-system

[/code]

收集 Envoy 代理和网关插件的日志以做进一步排查：
[code]
    kubectl get pods -n envoy-gateway-system

    NAME                                                      READY   STATUS    RESTARTS   AGE
    envoy-aibrix-system-aibrix-eg-903790dc-84ccfcbc6b-hw2lq   2/2     Running   0          13m
    envoy-gateway-7c7659ffc9-rvm5s                            1/1     Running   0          16m

    kubectl logs envoy-aibrix-system-aibrix-eg-903790dc-84ccfcbc6b-hw2lq -n envoy-gateway-system

[/code]
[code]
    kubectl get pods -n aibrix-system

    NAME                                        READY   STATUS             RESTARTS   AGE
    aibrix-controller-manager-fb4495448-j9k6g   1/1     Running            0          22m
    aibrix-gateway-plugins-6bd9fcd5b9-2bwpr     1/1     Running            0          22m
    aibrix-gpu-optimizer-df9db96c8-2fctd        1/1     Running            0          22m
    aibrix-kuberay-operator-5bf4985d86-7g4tz    1/1     Running            0          22m
    aibrix-metadata-service-9d4cd7f77-mq7tr     1/1     Running            0          22m
    aibrix-redis-master-7d6b77c794-bcqxc        1/1     Running            0          22m

    kubectl logs aibrix-gateway-plugins-6bd9fcd5b9-2bwpr -n aibrix-system

[/code]

## 可观测性与遥测（Observability & Telemetry）#

网关插件内置了基于 OpenTelemetry（OTel）的分布式追踪支持。这使你能够在外部处理（external processing）流水线中端到端地追踪请求的完整生命周期。

为节省系统资源，追踪默认严格采用主动开启（opt-in）模式。要启用遥测组件，需要完成以下两个步骤：

### 第 1 步：在 EnvoyProxy 中启用 OpenTelemetry#

在你的 helm chart 中启用 `openTelemetry.enable`，或直接在 `EnvoyProxy` 资源中添加遥测配置。
[code]
    apiVersion: gateway.envoyproxy.io/v1alpha1
    kind: EnvoyProxy
    metadata:
      name: aibrix-custom-proxy-config
    spec:
      telemetry:
        tracing:
          samplingRate: 100
          provider:
            type: OpenTelemetry
            openTelemetry:
              # -- OpenTelemetry Collector/OTLP 接收器的主机名或 IP 地址。
              # 此处不要包含协议方案（例如 http://）或端口。
              host: "otel-collector.monitoring.svc.cluster.local"

              # -- OTLP gRPC 接收器的端口（Envoy 在此严格使用 gRPC）。
              port: 4317

[/code]

### 第 2 步：配置 OTLP Exporter 端点#

接下来，必须在环境中显式配置一个 OTLP exporter 端点。可以通过设置全局端点或 traces 专用端点来启用。

例如，如果你在集群内运行了本地 OTel Collector，注入以下环境变量：
[code]
    env:
      # 方案 1：全局 OTLP 端点（推荐）
      # 注意：如果协议为 HTTP，SDK 会自动在该 URL 后追加 '/v1/traces'。
      # 如果协议为 gRPC，则按原样使用基础 URL。
      - name: OTEL_EXPORTER_OTLP_ENDPOINT
        value: "http://otel-collector.monitoring.svc.cluster.local:4317"

      # --- 或者 ---

      # 方案 2：traces 专用端点（取消注释以替代方案 1 使用）
      # SDK 将完全按原样使用该 URL，不追加任何路径。
      # （以下示例使用 4318 端口进行 HTTP/Protobuf 导出）
      # - name: OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
      #   value: "http://otel-collector.monitoring.svc.cluster.local:4318/v1/traces"

      # 指定传输协议。有效选项：grpc、http 或 http/protobuf
      - name: OTEL_EXPORTER_OTLP_PROTOCOL
        value: "grpc"

      # 保持 TLS 启用，但跳过服务器证书校验（适用于自签名证书）
      - name: OTEL_EXPORTER_OTLP_INSECURE_SKIP_VERIFY
        value: "true"

      # 如果你的 OTel Collector 或 APM 后端（例如 Datadog、Honeycomb）需要认证，请设置 Headers
      - name: OTEL_EXPORTER_OTLP_HEADERS
        value: "Authorization=Bearer {{ token }}"

[/code]

**高级配置**

关于支持的遥测配置的完整列表——包括采样率、TLS 选项和自定义请求头——请参阅完整配置指南：pkg/plugins/gateway/ENV_VARS.md

## 配置 Profile（Config Profiles）#

**config profile（配置档案）** 允许你将路由设置直接内嵌到模型的 Pod 注解中，使发往该模型的每个请求都能自动获取正确的配置。你可以在一个注解中定义多个命名 profile，并通过单个请求头按请求切换。

**设置 profile**

在 `Deployment`、`StormService` 或 `RayClusterFleet` 的 Pod 模板中添加 `model.aibrix.ai/config` 注解：
[code]
    annotations:
      model.aibrix.ai/config: |
        {
          "defaultProfile": "default",
          "profiles": {
            "default": {
              "routingStrategy": "least-request",
              "requestsPerSecond": 200
            },
            "large-input": {
              "routingStrategy": "pd",
              "routingConfig": {
                "promptLenBucketMinLength": 8192,
                "promptTokensGte": 8192,
                "prefillScorePolicy": "prefix_cache",
                "decodeScorePolicy": "load_balancing"
              },
              "requestsPerSecond": 50
            },
            "cache-friendly": {
              "routingStrategy": "least-kv-cache",
              "requestsPerSecond": 150
            },
            "offline-generation": {
              "routingStrategy": "throughput",
              "routingConfig": {
                "maxTokensGte": 2048
              }
            }
          }
        }

[/code]

也可以选择在模型级别锁定单一路由策略，使其无法在请求时被覆盖（参见下文的路由策略优先级）。`lockedRoutingStrategy` 只锁定策略：其余按 profile 的配置项（`requestsPerSecond`、`routingConfig`）仍然遵循由 `config-profile` 选中的 profile：
[code]
    annotations:
      model.aibrix.ai/config: |
        {
          "lockedRoutingStrategy": "pd",
          "defaultProfile": "default",
          "profiles": {
            "default": {
              "routingStrategy": "pd"
            },
            "batch": {
              "routingStrategy": "random",
              "requestsPerSecond": 50
            }
          }
        }

[/code]

**在请求时选择 profile**

两个请求头在请求时驱动配置：

  * `config-profile` 选择一个命名 profile；缺省时使用 `defaultProfile`（或 `"default"`）。`config-profile: auto` 要求网关根据各 profile 的 `routingConfig` 中与请求相关的提示条件（hints）选出一个具体 profile。

  * `routing-strategy` 覆盖所选 profile 的 `routingStrategy`，除非设置了 `lockedRoutingStrategy`（参见下文的路由策略优先级）。

[code]
    # 本次请求使用 "batch" profile
    curl http://${ENDPOINT}/v1/chat/completions \
      -H "config-profile: batch" \
      -H "Content-Type: application/json" \
      -d '{"model": "my-model", "messages": [{"role": "user", "content": "Summarize..."}]}'

[/code]
[code]
    # 让网关根据 routingConfig 中的提示条件解析出具体 profile
    curl http://${ENDPOINT}/v1/chat/completions \
      -H "config-profile: auto" \
      -H "Content-Type: application/json" \
      -d '{"model": "my-model", "messages": [{"role": "user", "content": "Summarize..."}], "max_tokens": 4096}'

[/code]

**顶层字段**

字段 | 说明
---|---
`lockedRoutingStrategy` | 在模型级别锁定单一路由策略。设置后，其优先级高于 `routing-strategy` 请求头、按 profile 的 `routingStrategy` 以及 `ROUTING_ALGORITHM` 环境变量。其余按 profile 的配置项（`requestsPerSecond`、`routingConfig`）仍正常生效。
`defaultProfile` | 未发送 `config-profile` 请求头时使用的 profile 名称。省略时回退为 `"default"`。
`profiles` | 命名 profile 的映射，每个 profile 携带下文所列的字段。

**Profile 字段**

字段 | 说明
---|---
`routingStrategy` | 该 profile 的路由算法（例如 `least-latency`、`prefix-cache`、`pd`）。完整列表参见上文的“路由策略”一节。
`requestsPerSecond` | 该 profile 的模型级 RPS 上限。超出限制的请求会以 HTTP 429 被拒绝。省略或设为 `0` 表示不限制。详情参见 [Production Model Deployments](../production/model-deployment.html)。
`routingConfig` | 算法特定的设置，以嵌套 JSON 对象形式给出。`config-profile: auto` 也会从该对象读取与请求相关的选择提示条件。当前支持的自动选择提示条件为 `promptTokensGte`、`promptTokensLt`、`maxTokensGte` 和 `maxTokensLt`。现有的策略特定字段（如 `pd` 的 `promptLenBucketMinLength`）仍然可用。详情参见 [Prefill-Decode Disaggregation](pd-disaggregation.html)。

使用 `config-profile: auto` 时，网关会对每个 profile 的 `routingConfig` 中受支持的请求相关提示条件进行求值。只有当某个 profile 的所有受支持提示条件都匹配请求时，该 profile 才算匹配。如果没有任何 profile 匹配，网关使用 `defaultProfile`（或 `"default"`）。`promptTokens*` 使用网关现有的提示词文本提取和本地 token 估算。`maxTokens*` 优先读取 `max_tokens`，`max_tokens` 缺失时再读取 `max_completion_tokens`。

**路由策略优先级**（从高到低）：

  1. 配置中在模型级别锁定的 `lockedRoutingStrategy` —— 一旦设置总是胜出，甚至优先于 `routing-strategy` 请求头。

  2. `routing-strategy` 请求头。

  3. 解析出的 profile 中的 `routingStrategy`。解析出的 profile 来自具体的 `config-profile` 请求头、`config-profile: auto` 的 routingConfig 提示条件，或 `defaultProfile`。

  4. 网关插件上的 `ROUTING_ALGORITHM` 环境变量。

**向后兼容性**：如果 Pod 没有 `model.aibrix.ai/config` 注解，网关会回退到 `routing-strategy` 请求头，然后是 `ROUTING_ALGORITHM` 环境变量（即上文的第 2 步和第 4 步）。现有部署无需任何迁移。

## Prometheus API 访问#

部分路由策略依赖于从 Prometheus HTTP API（PromQL）查询的指标。请使用以下环境变量配置端点及可选的 Basic Auth 凭据。

环境变量 | 默认值 | 说明
---|---|---
`PROMETHEUS_ENDPOINT` | （空） | Prometheus HTTP API 基础 URL（例如 `http://prometheus-operated.prometheus.svc:9090`）。为空时跳过基于 PromQL 的指标。
`PROMETHEUS_BASIC_AUTH_SECRET_NAME` | （空） | 包含 Basic Auth 凭据的 Kubernetes Secret 名称。设置时优先于下方的明文环境变量。
`PROMETHEUS_BASIC_AUTH_SECRET_NAMESPACE` | `aibrix-system` | `PROMETHEUS_BASIC_AUTH_SECRET_NAME` 指定的 Secret 所在的命名空间。
`PROMETHEUS_BASIC_AUTH_USERNAME_KEY` | `username` | `Secret.data` 中用作 Basic Auth 用户名的键。
`PROMETHEUS_BASIC_AUTH_PASSWORD_KEY` | `password` | `Secret.data` 中用作 Basic Auth 密码的键。
`PROMETHEUS_BASIC_AUTH_USERNAME` | （空） | Basic Auth 用户名。仅在未设置 `PROMETHEUS_BASIC_AUTH_SECRET_NAME` 时使用。
`PROMETHEUS_BASIC_AUTH_PASSWORD` | （空） | Basic Auth 密码。仅在未设置 `PROMETHEUS_BASIC_AUTH_SECRET_NAME` 时使用。

### 示例（明文环境变量）#
[code]
    export PROMETHEUS_ENDPOINT="http://prometheus-operated.prometheus.svc:9090"
    export PROMETHEUS_BASIC_AUTH_USERNAME="prom_user"
    export PROMETHEUS_BASIC_AUTH_PASSWORD="prom_pass"

[/code]

### 示例（Kubernetes Secret）#
[code]
    apiVersion: v1
    kind: Secret
    metadata:
      name: prometheus-basic-auth
      namespace: aibrix-system
    type: Opaque
    stringData:
      username: prom_user
      password: prom_pass

[/code]
[code]
    export PROMETHEUS_ENDPOINT="http://prometheus-operated.prometheus.svc:9090"
    export PROMETHEUS_BASIC_AUTH_SECRET_NAME="prometheus-basic-auth"
    export PROMETHEUS_BASIC_AUTH_SECRET_NAMESPACE="aibrix-system"

[/code]

## 消息大小配置#

流经网关的请求受两个相互独立的大小限制约束。两者的默认值均为 **4 MiB**，在调高或调低限制时必须保持同步。

### 网关插件 gRPC 缓冲区（`AIBRIX_GRPC_MAX_MESSAGE_SIZE_BYTES`）#

Envoy 通过外部处理（ext_proc）gRPC 连接与 AIBrix 网关插件通信。该环境变量控制该连接上单个 gRPC 消息的最大大小。超过此值的请求或响应会在到达路由逻辑之前被拒绝。

在 `values.yaml` 的 `gatewayPlugin.envs` 下配置：
[code]
    gatewayPlugin:
      envs:
        AIBRIX_GRPC_MAX_MESSAGE_SIZE_BYTES: "4194304"  # 4 MiB

[/code]

### Envoy 连接缓冲区（`clientTrafficPolicy.connection.bufferLimit`）#

Envoy 自身会为每条传入的 HTTP 连接缓冲最多 `bufferLimit` 字节后才开始处理。如果请求体超过该限制，Envoy 会在 ext_proc 过滤器被调用之前直接返回 `413`。

在 `values.yaml` 的 `envoyGateway.clientTrafficPolicy` 下配置：
[code]
    envoyGateway:
      clientTrafficPolicy:
        connection:
          bufferLimit: 4194304  # 4 MiB

[/code]

### 保持两个值同步#

由于当 `bufferLimit` 成为实际约束时，Envoy 会在网关插件看到请求之前就断开连接，因此这两个值应始终设置为相同大小。当需要支持更大的负载（例如大型 KV-cache 状态同步消息）时，请**同时**更新两个字段：
[code]
    gatewayPlugin:
      envs:
        AIBRIX_GRPC_MAX_MESSAGE_SIZE_BYTES: "16777216"  # 16 MiB

    envoyGateway:
      clientTrafficPolicy:
        connection:
          bufferLimit: 16777216  # 16 MiB

[/code]
