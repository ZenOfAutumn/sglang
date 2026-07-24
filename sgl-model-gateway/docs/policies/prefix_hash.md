# Prefix Hash（前缀哈希）算法详解

> 本文档是 [ROUTING_POLICIES_ZH.md](./ROUTING_POLICIES_ZH.md) 中「Prefix Hash」策略的深入拆解，
> 包含 `load_factor` 计算方式与选路分支等细节。主文档仅保留概览。

**实现**：`src/policies/prefix_hash.rs` — 取前 $N$ 个 token 做 xxhash，落到一致性哈希环，再做有界负载检查。

## 第一性原理降维

这是第一个显式优化 $H_i$（缓存复用）的策略，但用的是**近似手段**：

$$
\text{worker} = \text{Ring.lookup}\big(\text{xxh3}(\text{tokens}[0{:}N])\big), \quad N = \texttt{prefix\_token\_count}\ (\text{默认 }256)
$$

核心假设：**相同前缀 → 相同哈希 → 相同 Worker → KV Cache 命中**。但纯亲和会造成热点，因此加了**有界负载均衡**。

### `load_factor` 的详细计算方式

对应实现见 `PrefixHashPolicy::load_ok`（`src/policies/prefix_hash.rs`）。每次选路时，先在**健康 Worker**集合上计算全局指标，再对候选 Worker 做「负载是否可接受」判定：

```text
// 1) 仅统计健康 Worker
//    注意：这里的 worker.load() 是“活跃请求数”（在途/并发请求数），
//    而非 token 数——它由请求进入时 +1、完成时 -1 的原子计数器维护。
total_load  = Σ worker.load()   // 所有健康 Worker 的当前活跃请求数之和
num_workers = 健康 Worker 数量

// 2) 计算“含本次请求”的人均负载（+1 用于模拟即将进入的这一个请求）
avg_load    = (total_load + 1) / num_workers

// 3) 由 load_factor 放大得到可接受负载阈值
threshold   = avg_load * load_factor        // load_factor 默认 1.25

// 4) 判定：候选 Worker 负载 ≤ 阈值 即视为“负载 OK”
load_ok     = worker.load() <= threshold
```

用公式表示，即候选 Worker $w$ 被判定为可接受当且仅当：

$$
\text{load}(w) \;\le\; \underbrace{\frac{\left(\sum_{i} \text{load}(w_i)\right) + 1}{\text{num\_workers}}}_{\text{avg\_load}} \times \texttt{load\_factor}
$$

**关键细节：**
- **`+1` 的含义**：把「即将到来的这次请求」计入人均，避免在低负载时阈值被算得过低而误判过载。
- **`load_factor` 的语义**：允许单个 Worker 的负载最多达到人均的 `load_factor` 倍。默认 `1.25` 表示「最多超出平均 25%」。
  - 取值越大 → 越偏向**缓存亲和**（更容忍热点，命中率高，但负载更不均）。
  - 取值越接近 `1.0` → 越偏向**均衡**（更早触发迁移，命中率下降）。
- **边界短路**：当 `total_load == 0` 或 `num_workers == 0` 时，`load_ok` 直接返回 `true`（无负载可比，直接走缓存亲和）。

### 判定后的选路分支（结合阈值）

1. **RingHit**：哈希环命中的初始 Worker 满足 `load_ok` → 直接选它（最佳情况，缓存亲和成立）。
2. **LoadBalanceWalk**：初始 Worker 过载（不满足 `load_ok`）→ 在**同样满足 `load_ok`** 的健康 Worker 中挑负载最小者；若**所有** Worker 都过载，则退回使用初始 Worker（放弃迁移，避免无意义抖动）。
3. **FallbackLeastLoad**：无哈希环或环查找失败 → 忽略亲和，直接选负载最小的健康 Worker。

即在「缓存亲和」与「负载上限」之间由 `load_factor` 设定了一个可调的安全阀。

## 高阶结构化类比

等价于 **NUMA 调度的缓存亲和 + 负载封顶**：优先让线程回到其缓存所在的 NUMA 节点，但当该节点过载时允许迁移，避免亲和性演变成拥塞。

## 边界与反模式

| 维度 | prefix_hash | cache_aware（基数树） |
|---|---|---|
| 查找 | $O(\log n)$ | $O(\text{prefix\_len})$ |
| 内存 | $O(\text{workers} \times v_n)$ | $O(\text{total\_tokens})$ |
| 精度 | 前缀分组（粗） | 精确最长前缀匹配 |

- **边界**：与 cache_aware 的分界线是**精度 vs 可预测性**——prefix_hash 用固定长度前缀分组换取稳定的 $O(\log n)$，放弃了精确匹配。
- **反模式**：`prefix_token_count` 设得过短会把不同请求错误聚合（假共享），过长则失去分组意义。

## 演进动机

它是 cache_aware「太重」时的轻量替代：用哈希近似代替显式前缀树，把更新成本从 $O(\text{prefix\_len})$ 降到 $O(1)$。

