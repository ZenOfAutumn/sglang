# Cache Aware（缓存感知）算法详解

> 本文档是 [ROUTING_POLICIES_ZH.md](./ROUTING_POLICIES_ZH.md) 中「Cache Aware」策略的深入拆解，
> 包含双模式动态切换、基数树与 NUMA 类比等细节。主文档仅保留概览。

**实现**：`src/policies/cache_aware.rs` + `src/policies/tree.rs` — 每个 Worker 维护近似基数树，在「缓存亲和」与「最短队列」间动态切换。

## 第一性原理降维

Cache Aware 是唯一**精确**估计 $H_i$ 的策略，也是最能体现「路由=在线调度」本质的策略。它对每个 Worker 用**近似基数树**（存原始字符而非 token，省去分词开销）记录历史请求前缀，从而精确计算最长前缀匹配率。

其核心是一个**双模式动态切换**：

$$
\text{imbalanced} \iff (\text{max} - \text{min}) > \tau_\text{abs} \;\wedge\; \text{max} > \tau_\text{rel} \cdot \text{min}
$$

- **系统均衡** → 走**缓存亲和**：
  - 若最高匹配率 > `cache_threshold`（默认 0.5）→ 路由到匹配最高的 Worker（复用 KV）；
  - 否则 → 路由到**树最小**的 Worker（缓存容量最富余）。
- **系统失衡** → 走**最短队列**：直接路由到 pending 最少的 Worker，牺牲缓存换取负载再平衡。

背景任务周期性 LRU 淘汰树叶节点（`eviction_interval_secs`、`max_tree_size`）以约束内存。特别地，树以 `pool::model` 为 key 隔离 prefill/decode/regular 池，防止交替调用互相驱逐。

## 高阶结构化类比

最贴切的类比是 **NUMA 感知操作系统调度器**：

| OS 调度器 | Cache Aware |
|---|---|
| CPU 本地缓存亲和 | KV/Prefix Cache 亲和 |
| 工作集迁移成本 | 重新 Prefill 成本 |
| 负载失衡时抢占迁移 | 失衡时切最短队列 |

它精确回答了那个反直觉命题：**最空闲的实例未必最优**——只要 $Q_A - H_A < Q_B - H_B$，选更忙但有缓存的 $A$ 反而完成更快。

## 边界与反模式

- **边界**：与 prefix_hash 的分界线是**精确 vs 近似**（基数树最长匹配 vs 固定前缀哈希）。
- **极限**：请求完全无共享前缀时，基数树退化为纯开销，此时应回退到更轻的策略。
- **反模式**：**缓存亲和绝对优先**。若只追 $\arg\max_i H_i$，会形成「缓存越热→流量越集中→缓存越热」的正反馈灾难。代码用「双阈值失衡检测 + 切最短队列」正是为破除此正反馈。

## 演进动机

它推翻了「所有健康实例等价」的假设，把路由从「无状态副本选择」进化为「**状态位置选择**」——这是模型网关区别于普通 L4/L7 负载均衡器的分水岭。

