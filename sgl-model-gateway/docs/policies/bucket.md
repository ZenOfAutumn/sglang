# Bucket（分桶）算法详解

> 本文档是 [ROUTING_POLICIES_ZH.md](./ROUTING_POLICIES_ZH.md) 中「Bucket」策略的深入拆解，
> 包含双层机制、边界重划算法与关键数据结构等细节。主文档仅保留概览。

**实现**：`src/policies/bucket.rs` — 按 **请求长度（字符数）** 把 Prefill Worker 划分为若干「桶」，每个 Worker 负责一段长度区间；后台线程按滑动窗口内的真实负载**动态重划边界**，主要服务 PD（Prefill/Decode 分离）场景的 **prefill_policy**。

## 第一性原理降维

Bucket 的核心洞察是：**Prefill 阶段的计算成本近似正比于输入长度**。因此，与其在请求到达时估计每个 Worker 的排队/缓存状态，不如**按输入长度做静态分片**——让"长请求"和"短请求"分别落到固定的 Worker，从而把「重活」隔离开，避免长短请求混在同一实例上互相拖尾。

它把连续的长度轴 $[0, \infty)$ 切成 $N$ 段（$N$ = prefill worker 数），每个 Worker 认领一段区间 `[min, max]`：

$$
\text{route}(x) = \text{Worker}_k \quad\text{s.t.}\quad x \in [\text{min}_k,\ \text{max}_k]
$$

其中 $x$ 是本次请求的**字符数**（`request_text.chars().count()`，故 `needs_request_text() = true`）。选桶用**二分查找** `find_boundary`（边界有序），复杂度 $O(\log N)$。

**初始边界**是等分的：`gap = l_max / worker_cnt`（`l_max` 初值 4096），最后一个桶的上界扩到 `usize::MAX` 兜底任意超长请求。

## 双层机制：静态分桶 + 动态再平衡

**① 请求期（select_worker）——分桶为主、失衡时降级为最小负载**

每次选 Worker 时，先读取滑动窗口内各 Worker 的累计字符负载 `chars_per_url`，用**双阈值**判断是否失衡（与 Cache Aware 同构）：

$$
\text{imbalanced} \iff (\text{max} - \text{min}) > \tau_\text{abs} \;\wedge\; \text{max} > \tau_\text{rel} \cdot \text{min}
$$

- **未失衡** → 走 **Bucket 分桶**：按请求长度二分命中所属区间的 Worker（长度决定归属，可预测、稳定）。
- **已失衡** → 临时降级为**最小负载优先**：直接选当前累计字符最少的 Worker，牺牲长度亲和换取快速再平衡。

选定后 `post_process_request` 会：把本次 `char_cnt` 累加到该 URL 的负载、生成一条带时间戳的 `SequencerRequest` 入队，并**淘汰超出滑动窗口**（`period = bucket_adjust_interval_secs × 1000` ms）的历史请求、回滚其负载。即负载统计是一个**时间滑动窗口**，只反映最近一段时间的流量。

**② 后台期（adjust_boundary）——按负载分位重划边界**

后台线程每隔 `bucket_adjust_interval_secs`（默认 5s）对每个模型的桶做一次边界重算，目标是让**每个桶承担的总负载尽量均等**：

1. 计算目标单桶负载 `new_single_bucket_load = 总负载 / worker_cnt`；
2. **迟滞（hysteresis）保护**：若新旧单桶负载相差不到 2 倍（且旧值非 0），认为无需调整，直接跳过——避免边界频繁抖动；
3. 否则把窗口内所有请求长度**排序**，按"累积负载达到单桶目标"为切点，依次给每个 Worker 划定新的 `[min, max]` 区间（本质是**按负载做等分位切分**，而非按长度等分）。

这样，如果短请求特别多，短请求区间会被切得更细（多个 Worker 分摊）；长请求稀疏，则由少数 Worker 覆盖大长度区间——**边界随真实长度分布自适应**。

## 关键数据结构

| 字段 | 作用 |
|---|---|
| `boundary: Vec<Boundary>` | 有序的 `{url, [min,max]}` 列表，二分选桶的依据 |
| `chars_per_url` | 各 Worker 在滑动窗口内的累计字符负载（失衡判断 + 重划分位） |
| `request_list: VecDeque<SequencerRequest>` | 按时间排序的请求队列，用于滑动窗口过期淘汰 |
| `period` | 滑动窗口长度（ms），等于调整间隔 |

桶以 `normalize_model_key(model_id)` 为 key 隔离，不同模型各自维护独立分桶与负载窗口。Worker 增删（`add_prefill_url` / `remove_prefill_url`）会重置边界并同步 `chars_per_url`。

## 高阶结构化类比

类似磁盘的**分区/分级存储**或数据库的**范围分片（range sharding）**——按 key（这里是"请求长度"）的区间把负载路由到固定分片，再用后台任务按实际数据分布**动态调整分片边界**（类比 HBase Region Split / 自动 rebalance）。也可类比 CPU 调度里把长短任务分到不同队列的**多级队列**思想。

## 边界与反模式

- **边界**：适用范围窄，主要作为 PD 分离的 **prefill_policy**（长度 ≈ prefill 计算量时最有效），不建议作为通用 Regular 流量默认策略。
- **与 Cache Aware 的区别**：两者都用"双阈值失衡检测 + 降级"，但 Bucket 的一等公民是**请求长度分片**（无状态、可预测），Cache Aware 的一等公民是**前缀缓存亲和**（有状态）。Bucket 不感知 KV 缓存。
- **反模式**：
  - 当请求长度与真实计算成本**弱相关**时（如长度相近但难度差异大），长度分桶失去意义。
  - 迟滞阈值（2×）设置过松会导致边界长期不更新、分桶僵化；过紧则边界抖动、路由不稳定。
  - 长度分布极度长尾时，个别覆盖超长区间的 Worker 可能持续偏载，依赖失衡降级兜底。

## 演进动机

在 PD 分离下，prefill 是"算力密集、成本随长度线性增长"的阶段。把"按长度分片 + 按负载自适应重划边界"结合，既保留了**同长度请求路由稳定**的可预测性，又通过后台再平衡与请求期失衡降级，避免了静态分片在流量倾斜时的僵化——这是为"长度即成本"这一 prefill 特性量身定制的均衡器。

