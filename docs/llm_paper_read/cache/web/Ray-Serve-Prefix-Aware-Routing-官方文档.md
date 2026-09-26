# 前缀感知路由

> **文档信息**
>
> - **来源**：[Ray Serve 官方文档](https://docs.ray.io/en/latest/serve/llm/user-guides/prefix-aware-routing.html)
> - **抓取时间**：2026-09-01
> - **格式**：官方原始 Markdown

利用前缀感知请求路由，通过缓存局部性优化 LLM 推理。

> [!WARNING]
> 此 API 目前处于 **Alpha 阶段**，在正式稳定之前可能发生变化。

LLM 推理可以从缓存局部性优化中显著获益。当一个副本处理多个共享前缀的提示词时，推理引擎可以复用先前计算好的 KV 缓存条目，降低计算开销并改善响应时间。这一技术在 vLLM 中被称为[自动前缀缓存（Automatic Prefix Caching，APC）](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching.html)。

`PrefixCacheAffinityRouter` 将具有相似前缀的请求路由到相同的副本，以最大化 KV 缓存命中率。

## 何时使用前缀感知路由

在以下情况下使用前缀感知路由：

- 你的工作负载中有大量共享前缀的请求（例如相同的系统提示词或 few-shot 示例）
- 你在使用启用了自动前缀缓存的 vLLM
- 在负载均衡的场景下，缓存命中率比完美的负载均衡更重要

## 工作原理

`PrefixCacheAffinityRouter` 实现了一种多层路由策略，在缓存局部性与负载分布之间取得平衡：

### 1. 负载均衡检查

首先，它通过比较队列长度来评估当前各副本间的负载是否均衡。如果最高与最低队列长度之差低于 `imbalanced_threshold`，则继续执行前缀缓存感知路由。

### 2. 前缀匹配策略

当负载均衡时，router 使用前缀树查找先前处理过相似输入文本的副本：

- **高匹配率（≥10%）**：路由到前缀匹配率最高的副本，以获得更好的缓存命中率
- **低匹配率（<10%）**：回退到前缀缓存利用率最低的副本，以提升其利用率
- **无前缀数据**：使用默认的 Power of Two Choices（二选一）选择算法

### 3. 负载不均衡时的回退

当负载不均衡（队列长度之差超过阈值）时，router 优先考虑负载均衡而非缓存局部性，回退到标准的 Power of Two Choices 算法。

### 前缀树管理

router 维护一个分布式前缀树 actor，它：

- 跟踪每个副本处理过的输入文本前缀
- 支持自动淘汰旧条目以管理内存使用
- 借助 Ray 的 detached actor 模式在 router 实例之间持久存在

## 部署前缀感知路由

以下示例展示如何部署一个使用前缀感知路由的 LLM：

```{literalinclude} ../../../llm/doc_code/serve/prefix_aware_router/prefix_aware_example.py
:start-after: __prefix_aware_example_start__
:end-before: __prefix_aware_example_end__
:language: python
```

## 配置参数

`PrefixCacheAffinityRouter` 提供了若干配置参数用于调优其行为：

### 核心路由参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `imbalanced_threshold` | \(\infty\) | 判定负载均衡的队列长度差阈值。值越低，越优先考虑负载均衡而非缓存局部性。 |
| `match_rate_threshold` | `0.1` | 启用前缀缓存感知路由所需的最小前缀匹配率，取值范围为 \([0.0, 1.0]\)。值越高，要求前缀匹配程度越强。 |

### 内存管理参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `do_eviction` | `False` | 是否自动淘汰前缀树中的旧条目，以近似模拟 LLM 推理引擎的淘汰策略。 |
| `eviction_threshold_chars` | `400,000` | LLM 推理引擎触发淘汰前，前缀树允许容纳的最大字符数。 |
| `eviction_target_chars` | `360,000` | 执行淘汰时，前缀树需要缩减到的目标字符数。 |
| `eviction_interval_secs` | `10` | 启用淘汰后，两次淘汰检查之间的间隔秒数。 |

## 最佳实践

- **启用 vLLM APC**：确保在 `engine_kwargs` 中设置 `enable_prefix_caching=True`，否则 router 不会产生任何效果。
- **调优阈值**：根据工作负载特征调整 `imbalanced_threshold` 和 `match_rate_threshold`。
- **监控缓存命中率**：跟踪 vLLM 的缓存命中指标，验证 router 确实在提升性能。
- **从保守配置开始**：先使用默认设置，再根据观察到的行为逐步调优。

## 另请参阅

- {doc}`架构：请求路由 <../architecture/routing-policies>`
- [vLLM 自动前缀缓存](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching.html)
