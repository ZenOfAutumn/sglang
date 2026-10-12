# CacheRoute:面向大规模 LLM 服务的「有计划的前缀亲和路由」

> 原文:*CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM Serving*,Huang Cheng(Meta),arXiv:2608.19677,2026-08

---

## 1. 一句话概括

Prefix caching 只有在重复请求回到「仍持有该前缀 KV」的那台机器时才能省掉 prefill。**CacheRoute 把「路由决策」从每请求的反应式判断,改成周期性离线计算的一张全局路由表**:按 key 速率做准入(admission)、按负载算副本数、用 LPT 做放置,表在一个 control interval 内保持不变。在 60 张 H100 上服务 Llama-3.3-70B(fp8)时,$p99 \le 3.5\,\mathrm{s}$ 的 SLO 容量达到 **176±11 QPS**,是最强 baseline(Preble,76±11)的 **2.3×**;served KV 命中率从 64.1% 提升到 **93.2%**。

同样重要的是论文给出的**反例**:两个 32B 工作负载上亲和性反而让容量降到 0.5–0.67×。因此作者明确建议——**不要凭工作负载统计就开启亲和路由,必须先做 shadow replay 门禁**。

---

## 2. 问题:cache-blind 均衡与 sticky 亲和的两难

### 2.1 为什么均衡会让缓存变冷

一个业务 key 速率为 $\lambda_b$,若被均匀打散到 $R$ 个 destination,则**单台机器上该前缀的重访间隔**为:

$$
\mathbb{E}[\Delta t_{\text{revisit}}] \;\approx\; \frac{R}{\lambda_b}
$$

这意味着:

> 扩大集群($R$ 变大)虽然总缓存容量增加,却会让单个前缀**变得更冷**。

设有效驱逐时间为 $t_{\text{evict}}$,则只要

$$
\frac{R}{\lambda_b} \;>\; t_{\text{evict}}
$$

每个请求都要重做 prefill。

### 2.2 为什么纯 sticky 也不行

一致性哈希能保住 locality,但把 key 的**倾斜度直接映射成队列倾斜度**——最热的哈希桶决定了整个集群的 p99。Bounded hashing(CHWBL)和反应式 cache/load 策略缓解了这一点,但**一旦 cap 触发,溢出的那个请求就落到冷机器上,前缀白白浪费**。

### 2.3 真正的判据

论文把部署决策归结为一个**可测量的权衡**——设亲和带来的命中率增益为 $\Delta h$,残留的负载不均衡度为 $\rho = L_{\max}/\bar{L}$,则值得启用当且仅当:

$$
\underbrace{\Delta h \cdot c_{\text{prefill}}}_{\text{省下的计算}} \;>\; \underbrace{(\rho - 1)\cdot \bar{L}}_{\text{热点带来的排队代价}}
$$

> 换句话说:**可恢复(recoverable)的命中率增益,必须大过残留的负载倾斜代价。**

模型大小、精度、前缀长度、offered load、active set 宽度、cache 容量都会移动这个平衡点。第 5.3 节甚至给出「命中率更高但容量更低」的实测案例。

---

## 3. 工作负载画像(关键:可路由的只有 15%)

这是全文最容易被忽略、但决定成败的一段。工作负载是**半合成(semi-synthetic)**的:从脱敏运营遥测中提取 per-key 速率、到达间隔、流行度分布,再合成请求流,不含任何原始文本/用户数据/业务标识。

| 属性 | 实测值 |
|---|---|
| 不同业务 key 数 | 128,824 |
| 流量 Gini 系数 | 0.756 |
| 单 key 最大占比 | < 0.3% |
| Top 4% key 流量占比 | ≈ 47% |
| 多轮请求占比 | 80.8% |
| per-key 到达间隔 CV 中位数 | 1.93 |
| 平均输入长度 | ≈ 1.2K tokens |
| 当前轮之前的 token 占比 | ≈ 90% |
| **稳定的 per-business 上下文** | **≈ 180 tokens(仅占 15%)** |
| 相邻请求对中因 RAG 导致精确前缀断裂的比例 | 67% |

Prompt 的四段结构:**全局模板 + 业务稳定上下文 + 检索到的 few-shot 块 + 当前轮**。

- 全局模板对所有业务通用,**每台机器都会自然预热**,不构成路由机会;
- 检索块变化频繁,67% 的情况下会打断精确前缀匹配;
- **真正的路由杠杆只有那 180 tokens 的业务上下文**。

> 这解释了为什么 32B 的两个负载会失败:那里可恢复的 KV 工作量太小(flat 命中 1.1%,亲和后也只有 11.8%),收益盖不过倾斜代价。

---

## 4. CacheRoute 设计

整体是**「周期性离线规划 + 快速在线派发」**两层分离。

### 4.1 规划目标

设业务集合 $\mathcal{B}$,估计速率 $\lambda_b$,集群有 $R$ 个 destination(一个 destination = 一个可独立路由的模型服务组;70B 部署中是一对 TP-2)。规划器产出一张在一个 control interval 内固定的映射:

$$
T:\; \mathcal{B} \longrightarrow 2^{\{1,\dots,R\}}
$$

**(1) 基于负载的副本数**

从单个 destination 的延迟/负载拐点标定 $q_{\text{cap}}$,则:

$$
k_b \;=\; \max\!\left(1,\; \left\lceil \frac{\lambda_b}{q_{\text{cap}}} \right\rceil \right) \tag{1}
$$

使每个 assignment 的期望负载满足 $\lambda_b / k_b \le q_{\text{cap}}$。**注意:这是负载控制规则,不是驱逐阈值,也不提供缓存驻留保证。**

**(2) 暖集准入(warm-set admission)**

假设稳定前缀大小大致均匀,把缓存建模为 $C = R \cdot W$ 个前缀 slot(每台 $W$ 个)。按 $\lambda_b$ 降序考察 key,在下式约束下准入:

$$
\sum_{b \in \mathcal{A}} k_b \;\le\; C, \qquad \mathcal{A} = \{\text{已准入的 key 集合}\}
$$

> 局限:等 slot 模型无法覆盖前缀大小异构的场景,那需要 byte-aware 的度量与准入。

**(3) 放置与派发**

每个准入 key 贡献 $k_b$ 个大小为 $\lambda_b / k_b$ 的 job。先用「预期冷尾负载」初始化各 destination 的负载 $L[r]$,再按速率降序把每个 job 分配给**当前负载最小的合法 destination**:

$$
T(b) \;=\; \underset{S \subseteq \{1,\dots,R\},\, |S| = k_b}{\arg\min} \sum_{r \in S} L[r],
\qquad
L[r] \;\mathrel{+}=\; \frac{\lambda_b}{k_b}\;\; (\forall r \in T(b))
$$

即经典 LPT list scheduling(Graham 1969)。

> 论文诚实指出:由于存在 split job 与 distinct-destination 约束,**LPT 的标准近似界 $\frac{4}{3} - \frac{1}{3m}$ 并不适用**。

运行期:准入流量在其固定的 $T(b)$ 内选负载最小的成员;未准入的冷尾流量走全集群的 **power-of-two choices**。

### 4.2 算法与开销

```
Algorithm 1 周期性路由表构建
输入: {λ_b}, R, q_cap, slot 配额 C
1: for each key b:  k_b ← max(1, ceil(λ_b / q_cap))
2: 按 λ_b 降序排序; 在 Σ k_b ≤ C 时准入
3: 用预期冷尾负载初始化 L[r], r ∈ [1, R]
4: for each 准入的 b (按 λ_b 降序):
5:     T(b) ← L 最小的 k_b 个不同 destination
6:     for r ∈ T(b): L[r] ← L[r] + λ_b / k_b
7: 准入流量在 T(b) 内派发, 其余走 flat-LB
```

- 复杂度:

$$
O\!\left(|\mathcal{B}| \log |\mathcal{B}| \;+\; \sum_{b} k_b \log R \right)
$$

- **实测构建耗时:$R=30$、128,824 个业务时仅 345 ms**,且完全在请求路径之外
- 在线路径只是「查表 + 在小集合内比较负载」

### 4.3 运行语义(几个重要的「不保证」)

- 表在 interval 内固定:换来缓存稳定性,代价是可能过期;换表本身又会让部分 assignment 变冷(§5.4 两者都有测量)
- **CacheRoute 既不预留也不迁移 KV block**,准入前缀与冷尾前缀共享引擎原生缓存与驱逐策略
- **不提供任何解析式驻留保证**;启用前必须 shadow replay 实测 served token-weighted KV 命中与延迟

### 4.4 关键澄清:主结果里 $k_b$ 恒为 1

在主分布上

$$
\max_{b \in \mathcal{B}} \lambda_b \;<\; q_{\text{cap}}
\quad\Longrightarrow\quad
k_b = 1,\;\; \forall b
$$

因此**每个业务只有一个 destination**。也就是说 70B 的结果衡量的是:

> **top-rate 准入 + 稳定的单副本亲和 + 均衡放置**

副本机制(replication)只在合成 whale 的机制实验里才被触发,**对主结果没有贡献**。这一点作者反复强调,值得称道。

---

## 5. 实验方法

- **实现**:客户端路由 harness 绕过线上聚合 LB,按策略直发;请求元数据带 experiment id,以便把 served token-weighted KV 命中归因到正确的 run
- **旗舰测试床**:Llama-3.3-70B fp8,30 个 TP-2 destination(60×H100),每 destination 实测 40,071 个 KV block(≈641K tokens)
- **机制测试床**:Llama-3.1-8B-Instruct bf16,30 个单卡 H100
- **对比策略(6 个)**:Flat-LB(P2C)、Sticky 一致性哈希、CHWBL(ε=0.25)、DualMap 风格双候选、Preble 风格前缀历史+实时负载(后三者为同一 harness 下的重实现,**不等同于原系统复现**)
- **SLO 容量定义**:在 offered-QPS 阶梯 $\Lambda$ 上取

$$
\mathrm{Cap} \;=\; \max\Big\{\, \lambda \in \Lambda \;\Big|\; p_{99}(\lambda) \le \tau \;\wedge\; \mathrm{fail}(\lambda) \le 5\% \,\Big\}
$$

  逐 run 计算再取均值,报告 95% Student-$t$ 区间($n=5$ 时 $t_{0.975,4}=2.776$;$n=3$ 时 $t_{0.975,2}=4.303$,区间必然很宽)

---

## 6. 主要结果

### 6.1 70B / 60×H100 主结果(top-K 128,5 组配对种子)

| 策略 | KV 命中 | Cap.@3.5s | Cap.@5s | p99@100QPS |
|---|---|---|---|---|
| Flat-LB | 64.1±1.3% | 42±20 | 80 | 5.7 s |
| Sticky | 87.3±2.4% | 30 | 42±20 | 8.5 s |
| CHWBL | 75.6±0.8% | 64±11 | 128±14 | 3.8 s |
| DualMap | 88.7±1.9% | 58±22 | 84±32 | 5.3 s |
| Preble | 72.0±0.7% | 76±11 | 140 | 3.8 s |
| **CacheRoute** | **93.2±0.5%** | **176±11** | **180** | **1.8 s** |

读法:

- Sticky / DualMap **命中率也很高(87–89%)但 p99 极差**(8.5s / 5.3s)——**光有 locality,一旦队列成为瓶颈就没用**
- Preble / CHWBL **队列均衡但命中率掉到 72%/75.6%**
- CacheRoute 占据「高复用 + 无热点」的中间地带
- 在严格的 2s SLO 下,**只有 CacheRoute 通过任何测试负载(120 QPS)**,其余 5 个策略全部在 30 QPS 下限被左删失

### 6.2 更宽 active set 与第二个分布(优势会收窄)

- top-K 256(8 seed 专项确认):3.5s 拐点 80 QPS vs DualMap 60 QPS,**仅 1.33×**
- 第二个独立采集的分布:top-K 128 时三个 cache-aware 策略**并列 100 QPS(打平)**;扩到 top-K 256 才分开——CacheRoute 160 vs 最佳 baseline 100(1.6×)

> 作者主动指出:优势**只在 active set 超出暖 slot 配额之后才显现**。这个「不复现每一格」的对比反而更有说服力。

### 6.3 收益到底来自哪里?(8B 组件消融)

合成 whale 负载,top-K 128,逐个叠加组件:

| 配置 | KV 命中 | 不均衡度 | 容量 |
|---|---|---|---|
| Flat-LB | 56±1.9% | 1.00× | 240 |
| + 亲和 | 88±1.5% | **3.46×** | 240(无变化) |
| + 副本 | 88±1.6% | 2.60× | 240(仍无变化) |
| **+ LPT 放置(完整)** | 90±1.0% | **1.24×** | **500(右删失)** |

**结论极其清晰:亲和负责「把缓存工作捞回来」,LPT 放置负责「让这份收益在饱和时可用」。缺任何一个,容量都纹丝不动。**

---

## 7. 负面结果与运行边界(本文最有价值的部分之一)

| 场景 | flat 命中 | 亲和命中 | 容量倍率 | 结果 |
|---|---|---|---|---|
| 8B 合成 whale | 9.3% | 77.0% | 2–6× | **赢** |
| Aggregate A – 32B | 1.1% | 11.8% | 0.50–0.67× | **输** |
| Aggregate B – 32B | 0.8% | 8.5% | 1.0× | **打平** |

32B 的两个负载(Qwen3-32B bf16,30 个 MP2 destination)中,亲和只带来 8–11 个百分点的命中提升,**不足以抵消残留倾斜**。作者明确说:**模型大小本身并不能区分这些结果**。

### 部署门禁:shadow replay

- 保留候选 assignment 但不把响应返回给用户
- 在「当前拐点以下一档」和「拐点附近」两个负载点,对比 served KV 命中、每 destination 负载、p99 与失败率
- **仅命中率提升不算通过——必须 p99 或容量改善才启用**
- 模型、精度、上下文长度、batching、cache 配额、active-set 分布发生变化后,**重跑门禁**

### 为什么不做解析式驻留预测(负面结果)

作者尝试了 single-characteristic-time 占用模型,组件级插桩通过了 7 项隔离检查,每个 TP2 destination 暴露 40,071 个实测 KV block——**预测 served 命中率仍在中位数偏差 14.3 个百分点、p90 偏差 44.7 个百分点**。实测驻留曲线「先陡降后平台」,形状落在所测模型族之外,且作者拒绝在无直接证据时把这个形状归因于 batching 或 block 量化。

> **这正是 shadow replay 必须留在部署流程里的原因。**

---

## 8. 突发性与重规划

### 到达过程敏感性(70B fp8 / top-K 128)

| 到达过程 | Flat-LB 容量/命中 | CacheRoute 容量/命中 |
|---|---|---|
| Poisson | 30 / 64% | 180 / 94% |
| Gamma(CV=1.9) | 30 / 63% | 160 / 90% |
| Trace blocks(CV=2.73) | 左删失(<30)/ 67% | 180 / 93% |

主要改善**不依赖平滑到达**。(注:这是两策略、单 active-set 的敏感性测试,不外推到其他 baseline。)

### 速率漂移与换表(lognormal 随机游走 σ=0.3,6 个 interval)

其中 cache churn 定义为相邻两次 assignment 集合的 Jaccard 补:

$$
\mathrm{churn} \;=\; 1 - \frac{1}{|\mathcal{A}|}\sum_{b \in \mathcal{A}} \frac{|T_{i}(b) \cap T_{i+1}(b)|}{|T_{i}(b) \cup T_{i+1}(b)|}
$$

| 指标 | 均值 | 最差 |
|---|---|---|
| Cache churn($1-\mathrm{Jaccard}$) | 94.5% | 98.0% |
| Count churn($k_b$ 变化比例) | 1.1% | 1.6% |
| 过期表的 KV 命中损失 | 1.0 pt | 3.0 pt |
| 过期表的 p99 损失 | +192 ms | +858 ms |
| **装新表的预热瞬时命中跌幅** | **13.6 pt** | 15.6 pt |

**从头重算 LPT 会改动 94.5% 的 key→destination 映射**(即使 $k_b$ 没变,key 也可能被挪到另一台负载相同的机器)。因此:

> 应当**保持既有放置,直到实测的「过期代价」超过「预热代价」**。作者坦承尚未实现 churn-aware(带滞回)的重规划器。

---

## 9. 泛化性与局限

**可以泛化的**:只需要三样东西——稳定的路由 key、可复用的 key 专属前缀、在至少一个预热 interval 内有效的速率估计。路由 key 可以换成 tenant / 文档集合 / agent / 应用。由于只改放置不改模型执行,**不依赖特定模型架构或缓存实现**。

**尚不能泛化的**:

- 等 slot 准入假设稳定前缀大小大致均匀;异构长上下文需要 byte-aware 价值函数(**未做,不宣称 byte-knapsack 或最优准入**)
- 冷尾流量共享原生缓存,**可能驱逐已准入的前缀**
- 聚合 replay 保留 key 流行度和(突发实验中的)短程 gap 相关性,**不保留精确的 per-key 时间戳**
- Preble/DualMap/CHWBL 是同 harness 重实现,**不是原系统端到端复现**
- 多个 8B 拐点在 500 QPS 右删失;三 seed 研究的 t 区间很宽

---

## 10. 与相关工作的定位

| 方向 | 代表工作 | 与 CacheRoute 的关系 |
|---|---|---|
| 引擎内前缀复用 | vLLM/PagedAttention、SGLang | 它们让复用高效,**CacheRoute 决定请求去哪台机器** |
| 反应式 cache-aware 路由 | Preble、DualMap | 按实时 cache/load 打分;CacheRoute 构建**一个 interval 内的全局速率表** |
| 有界负载亲和 | 一致性哈希、CHWBL | 提供稳定亲和 + 显式负载控制,但 cap 触发即溢出到冷机 |
| KV 迁移/池化 | Mooncake、MemServe | **搬移** KV 状态,而非通过入口亲和**保住**它 |
| 请求迁移 / P-D 分离 | Llumnix、DistServe、Splitwise | 正交,这些系统仍可叠加入口规划来避免重复 prefill |

作者明确表态:**top-rate 准入与 LPT 都是成熟技术,本文不宣称新的调度原语**;贡献在于路由设计本身、大规模实测,以及「什么条件下运维应当保留普通 LB」的边界证据。

---

## 11. 工程落地清单

1. 收集聚合 per-key 速率 $\{(b, \lambda_b)\}$(key 可加盐或临时化,无需请求内容)
2. 在单个 destination 上实测延迟/负载拐点,标定 $q_{\text{cap}}$
3. 选一个**保守的**暖 slot 配额 $C = R \cdot W$
4. 对候选表与 Flat-LB 在匹配负载下做 **shadow replay**,对比 served KV 命中、不均衡度、p99、失败率
5. **仅当 p99 或容量改善时**才放金丝雀流量
6. 当 active set 或实测 p99 超出既定包络时,**回退到 Flat-LB**
7. 模型/精度/上下文长度/batching/缓存配额/分布变更后**重跑门禁**

---

## 12. 个人评述

**亮点:**

- **诚实度罕见**:主动声明主结果里 $k_b \equiv 1$(副本机制没起作用)、LPT 近似界不适用、baseline 是重实现、多处结果右删失、第二个分布上 top-K 128 打平、解析模型失败——这些本可以藏起来
- **消融把因果讲透了**:亲和把命中从 56% 拉到 88% 但容量纹丝不动,只有 LPT 把不均衡从 3.46× 压到 1.24× 后容量才动——「locality 必须配上 balance 才可兑现」这个结论比 2.3× 的数字更有工程价值
- **负面结果被当成一等公民**,并直接转化为部署门禁

**需要留意的:**

- 2.3× 高度依赖于「稳定业务前缀 180 tokens / 占 prompt 15%」这一特定结构。**换个 prompt 组织方式(比如检索块前置),可复用段可能归零**
- 94.5% 的 cache churn 说明**重规划目前是个未解问题**,论文自己也承认没实现 churn-aware 算法。生产中「什么时候换表」目前只能靠经验对比两个惩罚
- 等 slot 准入对长上下文(异构前缀大小)场景基本不适用,而这恰恰是当下 agent / 长文档场景的主流形态
- 只有两个真实衍生分布,且都来自同一类多租户对话助手场景

**如果要复现/借鉴**:先做的第一件事不是实现路由器,而是**测量自己负载的「可恢复前缀占比」**——如果像 32B 那两个负载一样只有 1% 左右的 flat 命中率,那么整套方案会让你更慢。
