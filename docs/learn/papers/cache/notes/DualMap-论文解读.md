# DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving —— 论文解读

> 论文：*DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving*
> 作者：Ying Yuan（华中科技大学，华为实习）、Pengfei Zuo†（华为）、Bo Wang、Zhangyu Chen（华为）、Zhipeng Tan†（华中科技大学）、Zhou Yu（华为）
> 会议/版本：ICLR 2026，arXiv:2602.06502v1（2026-02-06）
> 项目代码：[ASISys/DualMap](https://github.com/ASISys/DualMap)（官方 GitHub 仓库）

---

## 1. 一句话概括

DualMap 认为「缓存亲和 vs 负载均衡」之所以长期只能二选一，根因是**已有调度器都在单一映射空间里做决策**——一个 prompt 前缀只对应唯一一个"正确"实例，于是要么牺牲亲和去均衡、要么牺牲均衡去亲和。DualMap 用**两个独立哈希函数**把同一前缀映射到**两个**候选实例，再在这两个候选之间用 TTFT 预测做二选一（Power of Two Choices），把两个目标同时拿到手：缓存亲和的理论保证从 $\max(0, 1-1/m)$ 只降到 $\max(0, 1-2/m)$，而负载不均衡度从 $\Theta(\sqrt{m\log n/n})$ 骤降到 $\log\log n + O(1)$。在昇腾 NPU 集群上，有效请求容量最高提升 **2.25×**。

---

## 2. 它要解决的问题

分布式 LLM 服务的路由器面对每个请求时要回答一个问题：**发给哪个实例**。两个目标天然对抗：

- **缓存亲和（Cache Affinity）**：把共享同一 prompt 前缀的请求发到同一实例，复用 KV cache，跳过 prefill。前缀命中率高 → TTFT 低。
- **负载均衡（Load Balancing）**：把请求摊平到所有实例，避免热点实例排队。队列短 → TTFT 低。

问题在于真实负载的前缀分布**极度倾斜**：少数热前缀（系统提示词、工具描述、长文档、agent 轨迹前缀）承载了绝大多数请求。纯亲和策略会把这些请求全压到同一实例上，造成排队爆炸；纯均衡策略则完全放弃了 prefill 复用。

论文的关键论断是：**已有方案（Preble、Mooncake、Dynamo）之所以只能"折中"，是因为它们都在单映射空间内工作**——前缀 $p$ 只有唯一的"归属实例"，调度器唯一能做的事就是"要不要背叛这个归属"。于是所有方案本质上都是一个开关：


| 系统                | 决策形式                                                        | 退化行为                                               |
| ------------------- | --------------------------------------------------------------- | ------------------------------------------------------ |
| Preble              | 前缀命中率 < 50% 时切换到负载优先                               | 高负载下退化为 Least Loaded                            |
| Dynamo              | $\max_i(\text{KVMatch}_i - \text{Load}_i)$                      | 权重固定，两端都不最优                                 |
| Mooncake / Min-TTFT | $\max_i(\ell_{\mathrm{pending}}(i) - \ell_{\mathrm{recompute}}(r,i))$ | 重载时退化为 Least Loaded，轻载时退化为 Cache Affinity |

论文附录 B 专门刻画了这些 baseline 的退化点。DualMap 的做法是换掉底层几何：**给每个前缀两个家**。

---

## 3. 工作负载画像

论文在两类真实负载上评测（NPU 集群，Qwen2.5）：


| 负载                            | 平均输入长度  | 平均输出长度 | 前缀共享比例 | 请求数 |
| ------------------------------- | ------------- | ------------ | ------------ | ------ |
| Conversation（多轮对话）        | 12,035 tokens | 343 tokens   | 40%          | 4,000  |
| Tool & Agent（工具调用/智能体） | 8,596 tokens  | 182 tokens   | 59%          | 8,000  |

输入长度上限分别截断到 20,480 / 10,240 tokens，前 500 条作为 warm-up 不计入统计。

共享前缀的 CDF 显示：**48% 的 Conversation 请求和 76% 的 Tool & Agent 请求，其 prompt 至少 50% 是与其他请求共享的前缀**。这与 Preble 观察到的结论一致——prompt 远长于 decode，且共享比例很高——因此 prefill 复用的收益空间巨大，值得为它设计专门的路由。

---

## 4. 核心设计

### 4.1 双哈希映射（Dual Hashing）

对请求 prompt 的前缀 $p$，用两个**相互独立**的哈希函数计算两个候选实例：

```text
instance_id1 = f1(prefix(p)) mod num_instances
instance_id2 = f2(prefix(p)) mod num_instances

# Eq5: 冲突时的兜底，保证两个候选不重合
if instance_id2 == instance_id1:
    instance_id2 = (instance_id1 + 1) mod num_instances
```

这一步不看负载、不看命中率，纯粹由**前缀内容**决定。关键性质：

- **确定性**：任何时刻、任何路由器实例，同一前缀算出的候选对完全相同（无需共享状态、无需分布式一致性协议）。
- **前缀绑定（prefix-bound）**：候选对是前缀的函数，不是请求的函数。因此同一前缀的所有请求只会落在这两台机器上，KV cache 最多被复制两份。

**缓存亲和的理论保证**：设集群有 $m$ 个实例，纯 Cache Affinity 下某前缀的 KV cache 在集群中的"分散度"为 $\max(0, 1-1/m)$（即只有 $1/m$ 的实例持有它）；DualMap 变为 $\max(0, 1-2/m)$。也就是说**亲和性只损失了一份副本的代价**，而不是像负载均衡策略那样把前缀打散到全部 $m$ 台。

### 4.2 为什么是 2 而不是 $d$？

论文引用 Power of Two Choices（Mitzenmacher 2002）的经典结论。把 $m$ 个请求投到 $n$ 个实例：

$$\max_i L(I_i) \le \frac{m}{n} + \frac{\log\log n}{\log d} + O(1) \qquad (\text{Eq2})$$

- $d = 1$（随机 / 单映射）：$\dfrac{m}{n} + \Theta\left(\sqrt{\dfrac{m\log n}{n}}\right)$ —— 不均衡度随 $m$ 增长 (Eq3)
- $d = 2$：$\dfrac{m}{n} + \log\log n + O(1)$ —— 不均衡度**与 $m$ 无关**，只随 $\log\log n$ 增长 (Eq4)

从 1 到 2 是**指数级**的改进；从 2 到 3 只是把 $\log\log n$ 除以 $\log 3/\log 2$，收益急剧递减。而代价是线性的：$d$ 增大意味着热前缀的 KV cache 要在 $d$ 台机器上各存一份，**locality 被稀释**。$d = n$ 时就退化成"全局选最优"，即纯负载均衡。

> 这是全文最漂亮的一处论证：$d=2$ 不是拍脑袋选的超参，而是"均衡收益的指数拐点"与"亲和成本的线性增长"的交点。

### 4.3 候选间的选择：TTFT 预测

拿到 $\{I_1, I_2\}$ 后，用 TTFT 估计选一个：

$$\mathrm{TTFT}(r, i) = T_q(r, i) + T_c(r, i) \qquad (\text{Eq7})$$

- $T_q(r,i)$：请求 $r$ 在实例 $i$ 上的**排队时间**（由 $i$ 当前 pending prefill tokens 推算）
- $T_c(r,i)$：请求 $r$ 在实例 $i$ 上的**计算时间**（由未命中的 prompt 长度推算，命中的前缀不计入）

命中率高的实例 $T_c$ 小，负载轻的实例 $T_q$ 小——两个目标被统一到同一个可比的量纲（毫秒）上，不再需要人工加权系数。这比 Dynamo 的 $\max_i(\text{KVMatch}_i - \text{Load}_i)$（两个不同量纲相减）在原理上更干净。

#### Min-TTFT baseline 如何预估 TTFT

Min-TTFT 是论文对 Mooncake 请求调度策略的简化实现。对请求 $r$ 和候选实例 $i$，它首先查询该实例可复用的最长 Prefix，然后计算：

$$\ell_{\mathrm{recompute}}(r,i) = \ell_{\mathrm{prompt}}(r) - \ell_{\mathrm{cached}}(r,i)$$

再分别估算排队与计算开销：

$$\widehat{T}_q(r,i) = f_q\left(\ell_{\mathrm{pending}}(i)\right), \qquad \widehat{T}_c(r,i) = f_c\left(\ell_{\mathrm{recompute}}(r,i)\right)$$

其中，$\ell_{\mathrm{pending}}(i)$ 是实例 $i$ 队列中等待 Prefill 的 token 数，$\ell_{\mathrm{recompute}}(r,i)$ 是请求在该实例上的未命中 token 数；$f_q$ 和 $f_c$ 将 token 工作量换算为时间。最终选择预测 TTFT 最低的实例：

$$i^* = \arg\min_i \left[ \widehat{T}_q(r,i) + \widehat{T}_c(r,i) \right]$$

可将其直观理解为：**预测 TTFT ≈ 前面排队请求的 Prefill 时间 + 当前请求未命中部分的 Prefill 时间**

例如，实例 $I_1$ 已缓存请求 Prefix，但队列较长；$I_2$ 没有缓存，但队列为空。只要满足

$$\widehat{T}_q(r,I_1)+\widehat{T}_c(r,I_1) > \widehat{T}_q(r,I_2)+\widehat{T}_c(r,I_2)$$

Min-TTFT 就会牺牲缓存命中，将请求发往 $I_2$。

这里需要区分两种评测设置：

- **全局 Min-TTFT baseline**：在所有实例中计算并选择最小预测 TTFT。
- **DualMap-min-TTFT 消融项**：只在双哈希得到的两个候选实例之间执行同样的选择。

它的问题是逐请求贪心：实例负载轻微变化就可能令最优实例来回切换，导致相同 Prefix 被反复打散、额外重算，并进一步改变后续队列，形成 cache-aware 与 load-aware 决策振荡。DualMap 的完整策略因此不总选瞬时最小 TTFT，而是**优先保持 Cache Affinity，仅当缓存命中实例的预测 TTFT 将超过 SLO 时才切换到较轻载候选**。

> 论文只给出了上述估计结构及基于 pending/recompute tokens 的简化表达，没有披露 $f_q$、$f_c$ 的具体拟合系数或 profiling 方法，因此不宜进一步假设其精确时间模型。

### 4.4 自适应哈希前缀长度

哈希输入取 prompt 的前多少 token？取短了，不同请求集合会碰撞到同一 key，负载被人为堆到少数实例；取长了，hash key 超出了真实的共享前缀，本该复用的请求被拆散到不同实例。调度器事先并不知道共享前缀有多长，且不同请求的共享长度各不相同——所以这个长度必须在线自适应。

DualMap 维护一棵**请求前缀热度树（request prefix hotness tree）**动态调整。

#### 热度指标 ρ

$$\rho(p)=\frac{\left|\{ r \in W : \operatorname{prefix}(r)=p \}\right|}{|W|}$$

#### 参数逐项解释

- $p$：热度树中当前被统计的**前缀节点**，即某个 block 级请求前缀。
- $r$：滑动窗口内的一条请求，其 prompt 是一个 token 序列。
- $W$：当前**滑动窗口**内的全部请求集合；论文没有披露窗口大小及按请求数还是按时间滑动。
- $\operatorname{prefix}(r)=p$：请求 $r$ 在当前树深度上的前缀等于 $p$，并不是说 $r$ 的完整 prompt 只包含 $p$。
- $\{r\in W:\operatorname{prefix}(r)=p\}$：窗口内命中前缀 $p$ 的请求集合。
- $|\cdot|$：集合基数，即集合中的请求条数。因此分子是命中 $p$ 的请求数，分母 $|W|$ 是窗口总请求数，而非 token 数或全历史累计量。
- $\rho(p)$：前缀 $p$ 的流量占比（热度），取值范围为 $[0,1]$。例如窗口内有 100 条请求，其中 30 条命中 $p$，则 $\rho(p)=0.3$。
- $n$：**集群中的推理实例总数**，不是哈希前缀长度。假设实例等容量，则单台实例约占集群容量的 $1/n$，DualMap 为一个前缀提供两个候选实例，因此其候选对的容量上界约为 $2/n$。
- $1/n$：一台实例对应的集群容量份额，用作 hot 状态的示例退出阈值。
- $2/n$：两个候选实例对应的容量上界，用作进入 hot 状态的阈值。

这里真正的**哈希前缀长度**没有用 $n$ 表示，而是由热度树的深度决定。若当前 root→leaf 路径包含 $d$ 个 block，且每个 block 为 512 tokens，则哈希 key 最多覆盖

$$L_{\mathrm{hash}} = 512\,d\ \text{tokens}$$

其中 $d$ 是路径上的 block 数，$L_{\mathrm{hash}}$ 才是哈希前缀长度。判热时增加子节点，使 $d$ 增大；降温时删除子节点，使 $d$ 减小。

#### 判据

- 进入 hot（延长 hash key）：$\rho(p) > \dfrac{2}{n}$
- 退出 hot（缩短 hash key）：窗口间 $\rho(p)$ 从 $> \dfrac{2}{n}$ 显著下降到 $< \dfrac{1}{n}$

**$2/n$ 不是经验阈值，而是容量守恒的推论**（原文措辞：*reflecting the dual-mapping strategy's upper bound*）：双映射下前缀 $p$ 只绑定 $\{I_1,I_2\}$ 两台实例，它能合法占用的集群处理能力上限恰好是 $2/n$。一旦 $\rho(p)>2/n$，即使这两台把 100% 算力全给 $p$ 也吃不下，排队必然无界增长——此时候选间怎么选、怎么迁移都无解，唯一出路是把 $p$ 拆开，让它绑定更多候选对。

对称地，$1/n$ 是**单映射的容量份额**：$\rho<1/n$ 意味着一台实例就能独吞这个前缀，无需再占两个候选，合并回去可恢复 locality。$[1/n,\ 2/n]$ 因此构成迟滞带（hysteresis），进热区要 $>2/n$、退出要 $<1/n$，中间不动作，避免前缀在边界上反复分裂/合并把 KV cache 反复冲刷掉。

> 需要精确的一点：论文里 $\rho<1/n$ **不是一个独立的"冷前缀"判定**，而是 hot 状态的**退出条件**——原文是 "If a *hot* prefix's traffic drops significantly between windows (e.g., $\rho>2/n$ to $\rho<1/n$), its hotness is updated and child nodes removed"。

#### 调整粒度：block，不是 token

附录 A.1.1 明确 hash key 长度以 **block** 计，**1 block = 512 tokens**。树的结构是：

- 每个节点 = 一个 block 级的前缀片段
- hash key = **root → leaf 的完整路径**
- 判热 → **增加子节点**，hash key 加长一个 block，原本同属一个前缀的请求按第 $k{+}1$ 个 block 的内容分裂成多个子前缀，各自算出独立的候选对，热点被摊到更多实例
- 降温 → **删除子节点**，hash key 缩短，请求重新聚拢，亲和恢复

实测的 hash key 长度分布可以反证机制确实在动作：

| 数据集 | hash key 长度分布 | 解读 |
|---|---|---|
| Conversation | 95% 的请求 = 2 blocks | 1 个 system block + 1 个 user 输入 block 已足够区分；前缀不倾斜，无需分裂 |
| Tool & Agent | 45.2% = 2 blocks；14.9% = 6 blocks；37.8% = 13 blocks | 长尾来自两个"异常流行的前缀"被识别后主动延长 hash key，避免它们全压到同一实例 |

#### 论文没说清的部分

1. **滑动窗口长度**（按请求数还是时间？多少？）完全没给。它直接决定机制的反应速度，在大集群倾斜负载下是关键路径参数。
2. **"drops significantly" 的判定**：原文用的是 "e.g."，说明 $2/n\to1/n$ 只是举例，实现里真实的退出条件不确定。
3. **一次延长一层还是可跳级**：13 blocks 说明发生过多层扩展，但没说是逐窗口迭代加深还是单次到位。
4. **热度树的规模上限与淘汰策略**未提，长尾前缀多时树可能无界增长。
5. **无敏感性分析**：$2/n$ 虽有容量上界依据，但窗口长度、block 大小（512）的选择都没做消融。

### 4.5 重平衡：Cuckoo 启发的迁移

即使有双候选，负载仍可能倾斜（比如某个候选对的两台机器同时被别的前缀压住）。DualMap 借鉴 **Cuckoo hashing** 的思想做迁移：把实例 $i$ 上排队的请求 $r$ 推给它的**另一个候选** $j$。

迁移收益：

$$B_r^{(i \to j)} = \mathrm{TTFT}_{r,i} - \mathrm{TTFT}_{r,j} \qquad (\text{Eq6})$$

迁移条件是**两个都要满足**：

1. $B_r^{(i\to j)} > 0$（确实变快）
2. $\mathrm{TTFT}_{r,j} < \mathrm{SLO}$（迁过去之后确实达标——避免把违约请求从一台机器搬到另一台机器，只是转移痛苦）

与 Cuckoo hashing 的关键区别：**不递归、不级联**。迁移只在该前缀绑定的那一对实例之间进行，单轮批量完成。这牺牲了一点最优性，换来了有界的、可预测的调度开销（后面会看到是 2.2–2.5ms）。

不均衡度用变异系数衡量：

$$CV = \frac{\sqrt{\frac{1}{n}\sum_i (x_i - \mu)^2}}{\mu} \qquad (\text{Eq1})$$

其中 $x_i$ 是实例 $i$ 的 pending prefill tokens。

触发重平衡的开关是 `ttft_slo_threshold`——即在 SLO 内能处理完的最大 pending prefill token 数。超过它就说明这台机器已经无法按时交付，启动迁移。

### 4.6 弹性伸缩：双哈希环

实例数变化时，`mod num_instances` 会让所有映射失效。DualMap 用**一致性哈希环**（Karger 1997）承载两个哈希函数，各自成环。扩缩容时只有相邻区间被重映射：

```text
环上顺序: A → B → C → D
插入 E（落在 B 与 C 之间）
仅 (B, E] 区间的前缀被重映射到 E
其余前缀的 KV cache 完全不受影响
```

实测：QPS=4 时在 $t=74\ \mathrm{s}$ 从 4 实例扩到 8 实例，SLO 达标率恢复到 90%；QPS=2 时在 $t=1809\ \mathrm{s}$ 从 8 缩到 4，达标率仍保持 >90%。

### 4.7 Decode 瓶颈的检测与修正（附录 A.7）

一个容易被忽略的失效模式：当显存被 KV cache 耗尽时，实例会长时间卡在 decode 上，根本不启动新的 prefill。此时基于 pending prefill tokens 的 TTFT 估计**严重低估**——队列看起来是空的，但请求根本进不去。

检测方式：

```text
prefill_interval = t_current - t_last_prefill
if prefill_interval > T:      # T = 3s，经验值
    进入 decode-bottleneck 模式
```

修正后的估计：

$$\mathrm{TTFT} = D_{\mathrm{estimated}} + T_q + T_c$$

其中 $D_{\mathrm{estimated}}$ 取观测到的 `prefill_interval`。这一项把"要等多久才能轮到 prefill"显式建模了进去。

> 这是全文最有工程价值的一段。任何基于队列长度做 TTFT 预测的路由器，都会在显存打满时遇到这个陷阱。

---

## 5. 实验方法

**硬件**：每节点 8× 昇腾 NPU

- Qwen2.5-7B → 910B4（32GB）
- Qwen2.5-14B → 910B3（64GB）
- 1.5TB DRAM 作为 context cache 后端

**软件**：vLLM 推理引擎，fp16
**Context cache 容量**：7B 场景 1M tokens（约覆盖 30% 工作集）、14B 场景 0.5M tokens（约 15%）

**指标**：

- **Effective Request Capacity**：在满足 TTFT SLO 前提下能承载的最大请求速率
- **Goodput**：达标请求的吞吐
- **P50 / P90 TTFT**
- **Cache hit rate**（并与理论上限对比）
- **CV**（负载不均衡度）

**Baseline**：Cache Affinity、Least Loaded、Min-TTFT（Mooncake 请求调度策略的简化实现）、Preble，共 4 个。论文没有实际评测 Dynamo，因为作者认为其 cost-based 设计与 Mooncake/Min-TTFT 相似。

---

## 6. 主要结果

### 6.1 端到端


| 指标                     | 提升                         |
| ------------------------ | ---------------------------- |
| 有效请求容量（最佳情况） | **2.25×（+125%）**          |
| Tool & Agent goodput     | +16.7% ~ +48%                |
| Conversation 有效容量    | +40.6% ~ +80%                |
| Conversation goodput     | +14.3% ~ +40%                |
| P50 TTFT                 | −55.4% ~ −97.4%            |
| P90 TTFT                 | −82.3% ~ −97%              |
| Cache hit rate           | 达到理论上限的 62.5% ~ 96.4% |

#### 6.1.1 与 baseline 的具体对比

论文在 8 个实例上，将 DualMap 与 **Cache Affinity、Least Loaded、Min-TTFT、Preble** 比较。下面的提升均是相对每项实验中的**最佳 baseline**，而不是固定相对某一个算法：

| 工作负载 | 模型 | Effective Request Capacity | Goodput |
|---|---|---:|---:|
| Conversation | Qwen2.5-7B | **+80%** | **+40%** |
| Tool & Agent | Qwen2.5-7B | **+125%（2.25×）** | **+48%** |
| Conversation | Qwen2.5-14B | **+40.6%** | **+14.3%** |
| Tool & Agent | Qwen2.5-14B | **+55%** | **+16.7%** |

在高 QPS 下，相对最佳 baseline，DualMap 的 **P50 TTFT 降低 55.4%–97.4%**，**P90 TTFT 降低 82.3%–97%**；P50/P90 端到端延迟也呈现相同趋势。论文图中没有逐一报告 DualMap 相对每个 baseline 的精确延迟降幅，因此不能把该范围归因于某个固定 baseline。

Qwen2.5-7B 实验进一步给出了各策略的 Cache Hit Rate：

| 策略 | Conversation | Tool & Agent | 结果解读 |
|---|---:|---:|---|
| Cache Affinity | 29.6% | 64.1% | 缓存命中高，但热点前缀导致严重负载倾斜 |
| Least Loaded | 13.2% | 56.5% | 负载最均匀，但请求被打散，缓存复用最低 |
| Min-TTFT | 24.4% | 61.8% | 优于纯负载策略，但在 cache-aware/load-aware 间切换会破坏局部性 |
| Preble | 14.3% | 55.9% | 本实验中接近 Least Loaded；作者归因于其仅在 Prefix 命中率超过 50% 时采用 cache-aware 调度 |
| **DualMap** | **29.5%** | **65.4%** | Conversation 几乎追平 Cache Affinity，Tool & Agent 反而略高于 Cache Affinity |

综合来看：

- 对 **Cache Affinity**：DualMap 保留了几乎相同的缓存命中率，但显著改善热点和尾延迟。
- 对 **Least Loaded**：DualMap 同样保持较均衡的负载，同时减少重复 Prefill，使总 pending tokens 更少。
- 对 **Min-TTFT**：DualMap 避免逐请求追求最小 TTFT 所导致的策略振荡，缓存命中率更稳定。
- 对 **Preble**：DualMap 在该论文的两个工作负载中同时获得更高 Cache Hit、更低负载和更好的 SLO 表现；此外，双候选路由开销为 $O(1)$，而 Preble 需要扫描实例计算代价，随实例数增长。

> 注意：以上 Preble 是 DualMap 作者在其 vLLM/NPU 测试床上的实现与评测结果，不应直接与 Preble 原论文的 SGLang/GPU 数字横向比较。

### 6.2 消融（Conversation，Qwen2.5-14B）


| 策略                  | P50 TTFT  | P90 TTFT  | Cache hit | CV       |
| --------------------- | --------- | --------- | --------- | -------- |
| DualMap-cache-affinity | 2.8 s     | 20 s      | 29.5%     | 1.24     |
| DualMap-least-loaded   | 1.8 s     | 6.8 s     | 22.2%     | 0.05     |
| DualMap-min-TTFT       | 1.7 s     | 6.5 s     | 25.7%     | 0.09     |
| DualMap w/o rebalance  | 1.3 s     | 5.3 s     | 26.2%     | 0.00     |
| **DualMap（完整）**   | **1.2 s** | **4.7 s** | **26.5%** | **0.00** |

读这张表要注意几件事：

1. **DualMap-cache-affinity 的 CV = 1.24**，P90 TTFT 高达 20s——在双候选内只按亲和选择，仍会造成尾延迟灾难。
2. **DualMap w/o rebalance 已经把 CV 降到 0.00**，且相比 DualMap-min-TTFT，P50 / P90 分别降低 23.5% / 18.5%。这说明 SLO-aware 路由已贡献大部分收益，重平衡进一步降低 P90 TTFT 11.3%。
3. 完整 DualMap 的 Cache Hit 为 26.5%，低于 DualMap-cache-affinity 的 29.5%——它并非追求最高命中率，而是以约 3 个百分点的命中损失，把 CV 从 1.24 降至 0。注意这张表是**双候选内部选路策略的消融**，不是上一节四个全局 baseline 的结果。

### 6.3 开销

**元数据**：每 128-token block 16B。

- 7B / 64GB cache = 9,632 blocks → **146.2 KB/实例**，32 实例共 4.57 MB
- 14B → 44 KB/实例，32 实例共 1.38 MB

**运行时延迟**：


| 操作               | 耗时                           |
| ------------------ | ------------------------------ |
| KV 查询 / 保存     | ~0.2 ms                        |
| 路由决策           | ~0.6 ms（其中双哈希 ~0.05 ms） |
| 重平衡（每次调用） | 2.2 ~ 2.5 ms                   |

关键性质：**全部与集群规模无关**。因为路由不需要遍历全部实例——双哈希直接算出两个候选，只需比较这两个。这是相比 Preble（需要遍历所有 GPU 计算 load cost）在**可扩展性**上的结构性优势。

---

## 7. 负面结果与运行边界

论文本身比较克制地列了几条，我再补充几条读下来的判断：

1. **Cache hit rate 反而低于纯亲和策略**（26.5% vs 29.5%）。DualMap 的价值主张不是"命中率最高"，而是"用最小的命中率损失换最大的均衡收益"。如果某个场景 prefill 复用收益极高而队列几乎不会积压（例如实例数很少、负载很轻），纯亲和仍然更优。
2. **$d=2$ 的均衡保证是渐进的**。$\log\log n + O(1)$ 里的 $O(1)$ 常数在小集群（$n = 4, 8$）里未必可忽略，论文的实验规模也就在这个量级。
3. **重平衡不递归**。Cuckoo hashing 的强力来自级联踢出，DualMap 为了控开销砍掉了它。在双候选都饱和的极端倾斜下（单个超热前缀吃掉集群大半流量），DualMap 只有靠自适应前缀长度延长来救，而那需要热度树先观测到、再反应——存在滞后。
4. **自适应前缀长度机制的关键参数缺失**。阈值本身有依据（$2/n$ 是双映射的容量上界，$1/n$ 是单映射份额，见 4.4），但**滑动窗口长度、降温的"显著下降"判定、block 大小 512 的选择**都没有说明，也没有敏感性分析。窗口长度尤其要紧——它决定机制对流量突变的反应延迟。
5. **decode 瓶颈阈值 $T = 3\ \mathrm{s}$ 是经验值**，且强依赖模型/硬件组合。换到 H100 + 70B 上这个值几乎肯定要重调。
6. **评测全部在昇腾 NPU 上**，Qwen2.5-7B/14B，单节点 8 卡量级。跨节点网络、跨 region、异构硬件均未涉及。
7. **厂商自评**。作者主体来自华为，硬件是自家 NPU，baseline 是自己复现的。这不必然影响结论，但值得对照第三方数据看。

### 7.1 一个值得对照的第三方数据点

Meta 的 CacheRoute 论文（*Planned Prefix-Affinity Routing*）里复现了一个 "DualMap 风格双候选" 的 baseline，在 60×H100 / Llama-3.3-70B fp8 上测得：

- KV 命中率 **88.7 ± 1.9%**（很高）
- 但有效容量只有 **58 ± 22 QPS**，p99 达 **5.3 s**

对照 CacheRoute 自身的 176 ± 11 QPS @ p99 ≤ 3.5s。也就是说：**在更大规模、更大模型、更强硬件的环境下，双候选拿到了很高的命中率，尾延迟却不好**，而且方差极大（±22）。

这与 DualMap 自报的 2.25× 并不直接矛盾（规模、模型、硬件、负载、SLO 定义全都不同，且 Meta 复现的未必是完整 DualMap——很可能没有热度树和重平衡），但它提示了一个真实风险：**双候选在大集群里可能不足以吸收倾斜**。$n$ 从 8 涨到 60，$\log\log n$ 变化很小，但热前缀的绝对流量和候选对的绝对承载能力之比会显著恶化。届时自适应前缀长度（4.4 节）就从"优化项"变成"生死项"。

---

## 8. 与相关工作的定位


| 系统                | 映射空间      | 决策依据                                                         | 亲和保证     | 均衡保证            | 调度开销随规模         |
| ------------------- | ------------- | ---------------------------------------------------------------- | ------------ | ------------------- | ---------------------- |
| Cache Affinity      | 单映射        | 前缀                                                             | $1-1/m$ 集中 | 无                  | $O(1)$                 |
| Least Loaded        | 无映射        | 负载                                                             | 无           | 好                  | $O(n)$                 |
| Preble ($E^2$)      | 单映射 + 开关 | $\ell_{\text{missed}}$ vs $\ell_{\text{cached}}$，再算 load cost | 中           | 中                  | $O(n)$                 |
| Mooncake / Min-TTFT | 单映射        | $\max_i(\text{pending} - \text{recompute})$                      | 两端退化     | 两端退化            | $O(n)$                 |
| Dynamo              | 单映射        | $\max_i(\text{KVMatch}_i - \text{Load}_i)$                       | 权重固定     | 权重固定            | $O(n)$                 |
| CacheRoute (Meta)   | 规划式        | 离线 LPT 放置 + 周期重规划                                       | 强           | 强                  | 重规划 $O(\text{keys})$ |
| **DualMap**         | **双映射**    | 双候选 + TTFT 预测 + Cuckoo 迁移                                 | $1-2/m$      | $\log\log n + O(1)$ | **$O(1)$**             |

**与 Chord / Dynamo 的区别（附录 A.6）**：DualMap 的两个候选是**前缀绑定**的——$\text{Prefix}(p) \Rightarrow \{I_1, I_2\}$，两者地位对等、都由内容决定。而 Chord 的 successor 是单一的，Dynamo 的 primary + replicas 是有主从层级的（replica 用于容错，不是用于负载选择）。

**与 CacheRoute 的哲学分歧**很值得玩味：

- **DualMap = 在线、无状态、$O(1)$**：不需要全局视图，两个哈希算完就走。代价是决策视野只有两个候选。
- **CacheRoute = 离线、有状态、规划式**：周期性用 LPT 算出全局最优放置表。代价是重规划时 94.5% 的 cache churn，以及 128,824 keys 需要 345ms 建表。

前者适合规模大、变化快、无法承受全局同步的场景；后者适合负载模式稳定、可以摊销规划成本的场景。**这两条路线目前还没有被公平地在同一 testbed 上比较过**，是这个方向明显的空白。

---

## 9. 工程借鉴清单

1. **双候选 + 二选一，是一个可以低成本嫁接到现有路由器上的改动**。不需要全局状态、不需要一致性协议、$O(1)$ 决策。如果你的路由器现在是"哈希到一台"，改成"哈希到两台再比 TTFT"是投入产出比极高的一步。
2. **TTFT 统一量纲**优于加权打分。把亲和收益和排队代价都换算成毫秒，避免了 $\alpha \cdot \text{hit} - \beta \cdot \text{load}$ 这种需要人工调参且跨负载不可迁移的形式。
3. **迁移必须同时满足"变快"和"迁过去能达标"**。只判断"变快"会导致违约请求在实例间来回搬运，白白消耗调度开销。
4. **decode 瓶颈检测**（`prefill_interval > T`）。任何基于队列长度预测 TTFT 的系统都必须处理这个情况，否则显存打满时预测会系统性乐观。
5. **用 CV 而不是 max/min 衡量不均衡**。CV 对全局分布敏感，max/min 只反映两个极端点。
6. **一致性哈希环承载哈希函数**，让扩缩容只影响相邻区间。别用 `mod n`。
7. **阈值要从容量守恒推、并留迟滞带**。$2/n$ 不是拍出来的：双候选能承载的算力份额上限就是 $2/n$，超了必然积压；$1/n$ 是单候选的份额，退回去就该合并。$2/n$ 进、$1/n$ 出形成迟滞带，防止边界抖动把缓存反复冲刷。这套"用系统容量反推阈值 + 留迟滞"的做法适用于所有基于阈值的自适应机制。
8. **元数据开销要按 block 粒度算给自己看**：16B/128-token-block、32 实例 4.57MB——这个量级说明"路由器持有全集群 KV 索引"在几十实例规模下是完全可行的，不必过早引入分层。

---

## 10. 个人评述

### 亮点

- **问题诊断比解法更值钱**。"已有方案都被困在单一映射空间"这个观察，把一堆看似不同的启发式（Preble 的开关、Dynamo 的加权、Mooncake 的减法）归约成了同一个结构性缺陷。有了这个视角，解法几乎是自然推出来的。
- **$d=2$ 的论证是教科书级的**。均衡收益从 $d{=}1$ 到 $d{=}2$ 是指数改进、之后递减；亲和成本随 $d$ 线性增长。两条曲线一画，$d=2$ 就是唯一答案。这不是调参，是推导。
- **消融表诚实**。作者明确展示了 DualMap 的 cache hit（26.5%）低于纯亲和（29.5%），没有藏。并且承认 w/o rebalance 已经拿到大部分收益——等于主动削弱了自己一个组件的重要性。这种诚实在系统论文里不常见。
- **开销全部与集群规模无关**，这是相对 Preble 最有说服力的结构性优势。Preble 的 load cost 计算要遍历所有 GPU，DualMap 只看两个。

### 需要留意的

- **规模验证不足**。实验在 8 卡量级，而理论保证是渐进的、目标场景是大集群。Meta 在 60×H100 上复现的双候选 baseline 表现不佳（58±22 QPS，p99 5.3s），虽然不能直接等同于 DualMap，但足以让人对"$n$ 变大后双候选是否够用"保持怀疑。
- **自适应前缀长度是隐藏的关键路径**。论文把它当作一个辅助机制轻描淡写地带过（没有消融、没有敏感性分析），但按上一条的分析，在大集群倾斜负载下它才是决定成败的部件。这块的实验缺失是全文最大的遗憾。
- **未交代的调参项**。decode 瓶颈阈值 $T=3\ \mathrm{s}$ 是纯经验值，换硬件必重调；热度阈值 $2/n$ / $1/n$ 本身有推导依据，但它依赖的滑动窗口长度论文只字未提，等于把一个关键旋钮藏了起来。
- **重平衡砍掉递归**在论文的负载下没暴露问题，但那是因为 CV 已经被双哈希打到 0.00 了——重平衡几乎没被真正压测过。真正倾斜的负载下，非递归迁移是否够用是未知的。
- **厂商自评**。华为团队 + 昇腾 NPU + 自复现 baseline。数字大概率是真的，但选择性（选哪些负载、哪些 SLO 定义、哪个 baseline 配置）无法排除。

### 一句话定位

**DualMap 是"缓存亲和 vs 负载均衡"这条线上第一个给出结构性答案而非启发式折中的工作**——它没有发明新的打分函数，而是换了底层的映射几何，让两个目标从"此消彼长"变成"各让一小步"。理论优雅、开销 $O(1)$、可嫁接性极强；但它的规模验证停在 8 卡，而它最想解决的问题恰恰出现在大集群。如果要在生产里用，先补上大规模倾斜负载下自适应前缀长度的实验。

---

## 附：与本仓库其他解读的交叉引用

- [`Preble-论文解读.md`](Preble-论文解读.md) —— $E^2$ 算法、prompt-aware load cost、Key Portion；DualMap 附录 B 对 Preble 的退化分析可与之对读
- [`CacheRoute-论文解读.md`](CacheRoute-论文解读.md) —— Meta 的规划式路由；其 §6.1 包含对 DualMap 风格双候选的第三方复现数据
- [`LLM服务负载均衡与调度-文献综述.md`](LLM服务负载均衡与调度-文献综述.md) —— 整体脉络定位
