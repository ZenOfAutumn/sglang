# TOPAS: Workflow-Aware Prefix-State Scheduling for Multi-Agent LLM Serving —— 论文解读

> 论文：*TOPAS: Workflow-Aware Prefix-State Scheduling for Multi-Agent LLM Serving*
> 作者：Hongqiu Ni、Han Tian、Guopeng Li、Haisheng Tan（中国科学技术大学）、Chi Zhang（合肥工业大学）
> 版本：**Preprint**，arXiv:2608.25523v1（2026-08-26，cs.CL），未经同行评审，引用数 0
> 代码与数据：**论文中未给出任何开源链接**（未获取）。正文多次引用 "supplementary material"，但 arXiv v1 的 PDF **不含附录**——aging 更新规则、first-task completion / throughput 完整结果、全部超参取值都在这份拿不到的补充材料里
>
> ⚠️ **阅读提示**：这是一篇**只有 8 页（含参考文献）的短会议格式论文**，全部实验跑在**单张 A100 80GB** 上，模型 Qwen2.5-32B-Instruct。值得读的是它的**问题提法**——把「哪些 agent 前缀该常驻显存」从 cache 淘汰策略的副产品提升为**显式的 workflow 级调度变量**，与请求准入**联合**决策（§III 的 Eq.2 那个联合容量约束是全文最精炼的一句话）。以及 §II 的两个 diagnostic，尤其是 Table I：**同样的请求集合、同样的 KV 预算，只改顺序，平均 batch 从 4.7 变成 9.3，makespan 差 1.8–1.9×**。这个微基准比后面所有的 JCT 百分比都更有说服力。
>
> **需要提前设防的**：（1）问题形式化的第一句话就是 "All tasks follow the same known DAG"——**整篇论文建立在 DAG 已知且所有任务同构这个强假设上**；（2）实验的到达率低到离谱（0.005–0.35 task/s），JCT 量级 500–2000 秒，是一个**极度过载、任务级排队主导**的工作点；（3）消融里 "相对 TOPAS-base 降 60.5%" 这个数字大得反常，反而说明基础效用函数本身可能是坏的；（4）所有超参数（$\beta$、$\rho$、$H_{\text{age}}$、$M$、$K$）**在正文中一个取值都没有给**。

---

## 1. 一句话概括

在多 Agent LLM workflow 服务里，**一个 agent 的长系统提示 KV cache 是否常驻显存，本身就是一个应该被调度的决策变量**，而不是「请求排序 + LRU 淘汰」自然产生的副产品。TOPAS 把「保留哪些 agent 前缀（resident-prefix set $R'$）」和「准入哪些就绪请求（$Q'$）」放进**同一个联合容量约束** $\sum_{a \in R'} \ell_a + \sum_{r \in Q'} d_r(t) \le B_t$ 下同时决策，用一个以**任务级 JCT** 为目标的效用函数给候选 post-decision 状态打分：收益是「任务最长剩余服务路径的期望缩短量」，代价是前缀搬运时间、抢占重算、以及被撤销的准入信用，再加一项**一跳前瞻**的下游前缀复用价值。在单张 A100 上、SGLang v0.5.3 内实现，相对**逐工作负载逐指标的最强 baseline**：三个合成 DAG 的 mean/p99 JCT 最多降 **39.8% / 49.4%**；MetaGPT-SOP mean JCT 降 **9.8%**；MetaGPT-TL mean/p99 降 **22.0% / 26.6%**。

**它相对 Preble / CacheRoute 这类「请求级前缀亲和」多做的一件事**：请求级亲和只能看到「现在队列里谁能复用现有前缀」，是**反应式**的；而 workflow 的 DAG 告诉你「A 阶段跑完之后，B 阶段的请求一定会到」——**未来的到达是结构上可预测的**，所以可以**主动**把 B 的前缀提前留住、或者反过来主动把 A 的前缀踢掉换成 B 的。这是从「猜未来」到「读 DAG」的信息升级。

---

## 2. 它要解决的问题

### 2.1 多 Agent 负载的前缀结构，和多轮对话不一样

先把背景摆清楚（这部分论文自己没展开，用本仓库其他解读的数据补）：

| 负载类型 | 前缀复用的形状 | 复用是谁跟谁之间的 |
| --- | --- | --- |
| 多轮对话 | 单调增长的**会话历史**，每轮在上一轮末尾追加 | **同一个会话内**，跨轮次；会话之间几乎不共享 |
| RAG | 系统提示 + 若干可复用 passage 拼装 | **跨请求**，但复用哪几个 passage 是检索结果决定的、事前不可知 |
| **多 Agent workflow** | 每个 agent 一个**固定的长 static prefix**（role 定义、tool schema、policy 文本），后接请求特有的 dynamic suffix | **同一 agent 跨不同 task 的请求之间**——复用集合是**有限且事前完全已知的**（就是 agent 的个数） |

DualMap 的负载统计给了一个量化对照：**48% 的对话请求**共享 ≥50% 前缀，而**工具/Agent 类请求这个比例是 76%**。多 Agent 负载的前缀共享程度远高于对话，而且更重要的是**结构不同**：

- 对话负载的前缀空间是**开放的**——每个会话一条独立的前缀链，数量随并发会话数线性增长，谈不上「哪几个前缀该常驻」，只能做 LRU。
- Agent 负载的前缀空间是**闭合且小的**——一个 MetaGPT SOP workflow 只有 5 个角色，TOPAS 的 DAG-10-Wide 也只有 10 个 prefix namespace。**「保留哪几个」变成了一个真实可枚举的组合优化问题**，这正是 TOPAS 敢做 $\mathcal{P}(\mathcal{A}_t)$ 全枚举的前提。

这个区别是 TOPAS 全部立论的基础，但论文只用了一句话带过（"Requests produced by agent $a$ share a static prefix of length $\ell_a$"），没有做任何负载统计来支撑「agent 前缀确实又长又少」。**这是一个可以补而没补的实证缺口。**

### 2.2 Diagnostic 1：异构前缀共存吃掉 batch 容量（Table I）

两个 agent A / B，各 $n$ 个独立请求，前缀不同且都很长，dynamic suffix 和 decode 长度全部相同。ALTERNATING 按 A,B,A,B,… 排，GROUPED 按 agent 分组排。**请求集合完全相同，KV 预算完全相同，只有顺序不同。**

| Load | Makespan (s) GROUPED/ALTERNATING | Mean batch G/A | Peak batch G/A |
| --- | --- | --- | --- |
| $n=100$ | **1145.5 / 2086.0** | 9.3 / 4.7 | 15 / 8 |
| $n=150$ | **1639.3 / 3104.4** | 10.0 / 4.8 | 16 / 8 |
| $n=200$ | **2143.5 / 4079.1** | 11.5 / 5.0 | 16 / 7 |

ALTERNATING 耗时是 GROUPED 的 **1.8–1.9×**，平均 batch 恰好差**约一倍**。

> **这是全文最扎实的一个实验**，也是我读完之后唯一觉得「必须记住」的结果。它揭示的机制非常朴素：长前缀的 KV 占用是**乘在 batch 容量上的固定成本**，两个长前缀同时常驻就等于把留给 dynamic suffix 和 decode reservation 的空间砍掉一大块，而 batch size 直接决定 GPU 的有效服务率。
>
> **但要注意它的实验条件是被刻意放大的**：两个 agent 前缀"distinct long prefixes"（具体多长没说），dynamic suffix 完全相同。在 $\ell_a$ 相对 $B_t$ 很小的情况下（比如 2k token 前缀 vs 80GB A100 的几十万 token KV 预算），这个效应会显著弱化。**论文既没有给 $\ell_a$ 的具体数值，也没有做 $\ell_a / B_t$ 的敏感性扫描**——而这恰恰是判断这个机制在你自己的负载上值不值钱的唯一参数。这是一个不该省的数字。

### 2.3 Diagnostic 2：局部性优先与进度优先各错一半（Fig.1 / Fig.2）

Chain-3 workflow（$A \to B \to C$），0.15 task/s：

- **LPM（locality-first）**：优先选能复用当前常驻前缀的请求。问题在于它是**反应式、到达驱动**的——它自己不选目标前缀状态，是「请求顺序 + cache 淘汰」决定了驻留前缀，驻留前缀又反过来成为下一次决策的局部性信号。在 A 端持续到来的负载下，这个正反馈会**把服务牢牢钉在 workflow 入口**，B、C 阶段饿死。
- **SRPT（progress-first）**：优先剩余处理时间短的任务，能把任务推向出口。但它**完全没有前缀驻留的概念**——一路追着最短剩余工作走就会不断在 agent 之间切换，每次切换都要付前缀搬运或重算的代价。
- **Strawman（LPM + 下游门控）**：作者加了一个粗糙的干预——一旦 C 的 backlog 积累就优先 C。Fig.1 显示这个**极其粗糙的主动控制在 mean JCT 上同时打败了 LPM 和 SRPT**。

> **Strawman 这个对照做得很聪明**：它证明了「哪怕是最笨拙的主动前缀状态控制也胜过两个纯粹的单目标策略」，从而把论证从"我们的方法好"降级到"主动控制这件事本身有价值"——后者是一个更弱但更可信的主张。**可惜 Fig.1 是纯柱状图，论文正文没有给任何数值**，只能从图上读出 mean JCT 大致是 LPM ≈ 265s、SRPT ≈ 205s、Strawman ≈ 180s 的量级。**这三个数字论文一个都没写进正文。**

Fig.2 的 Gantt 图把机制画得很清楚（假设显存一次只能放一个 agent 前缀，每请求 4 时间单位，每次前缀 setup 1 时间单位）：

- SRPT：`A1 → B1 → C1 → A2 → B2 → C2`，每次跨阶段都换前缀付 setup，完成时刻 14 和 29，mean JCT = **21.5**
- LPM：`A1 A2 → B1 B2 → C1 C2`，setup 只付 3 次，完成时刻 22 和 26，mean JCT = **24**

这个玩具例子里 SRPT 反而赢——因为它让第一个任务早早完成。**论文用它说明「两种策略在不同结构下各有胜负，不存在一个总是对的顺序」**，这个用法是恰当的，没有过度引申。

### 2.4 问题形式化（§III）

所有任务遵循**同一个已知的 DAG** $G = (V, E)$，任务 $J_i$ 在 $\tau_i$ 到达。节点 $v$ 是一个由某 agent 执行的 stage，前驱全部完成后 ready。一个 stage 可能发出多个 LLM 请求。

$$C(J_i) = \max_{v \in V} C_{i,v}, \qquad \text{JCT}(J_i) = C(J_i) - \tau_i \qquad (\text{Eq1})$$

调度点 $t$ 的 GPU 状态 $S_t = (R_t, Q_t)$，$R_t$ 是常驻 agent 前缀集合，$Q_t$ 是运行中请求集合。动作选一个 post-decision 状态 $S' = (R', Q')$：$Q' \setminus Q_t$ 是新准入，$Q_t \setminus Q'$ 是被抢占。**强制 $R' = \{a(r) : r \in Q'\}$**（不允许空转的常驻前缀）。核心约束：

$$\sum_{a \in R'} \ell_a + \sum_{r \in Q'} d_r(t) \le B_t \qquad (\text{Eq2})$$

$d_r(t)$ 含 dynamic suffix KV、已生成 token、decode 预留；**同一 agent 的多个请求只付一次 static prefix 成本**——这一项正是 GROUPED 优于 ALTERNATING 的数学表达。

**调度器能观测**：就绪请求、当前 GPU 状态、任务到达时间、stage 进度、**workflow 拓扑与 agent 映射**。
**不能观测**：未来任务到达、未来 ready 时刻、ReAct 迭代次数、未展开的 tool 输出。

> **"All tasks follow the same known DAG" 是全文最大的一处假设，而论文对它没有任何讨论。** 它同时要求三件事：(a) DAG 事前已知；(b) 所有任务**共享同一个** DAG（不是每个任务一个自己的 DAG）；(c) DAG 是静态的，没有数据依赖的条件分支。
>
> 现实中的 agent 系统里，(a) 对 MetaGPT / ChatDev 这类 SOP 驱动的框架成立，对 AutoGPT / 开放式 ReAct agent 完全不成立；(b) 意味着**多租户混合 workflow 的场景直接出局**；(c) 意味着 `if 检查失败 then 回到 coder 重写` 这种极常见的条件回边不被支持。作者只把 ReAct 迭代次数和 tool 延迟放进了「运行时未知」——**这是把不确定性限制在了 DAG 节点内部，而拓扑本身仍然是刚性的**。
>
> 公平地说，这个假设换来了 TOPAS 的核心能力：**因为知道 DAG，所以知道 B 的请求一定会来，所以可以主动保留 B 的前缀**。整篇论文的价值就建立在这个交换上。但读者必须清楚这个交换的代价有多大——**它把适用范围收缩到了「静态 SOP 型多 Agent 流水线」这一类**，而这恰好也是作者选 MetaGPT 做评测的原因。

---

## 3. 核心设计

TOPAS = 一个 **post-decision state 的评分 + 搜索** 框架。每个调度事件上：枚举/贪心生成候选 prefix set $R$ → 对每个 $R$ 用 GREEDYPACK 构造兼容的请求分配 → 给每个候选状态打分 → 贪心提交最高分。

### 3.1 效用函数总览（Eq.3 / Eq.13）

$$U_{\text{base}}(S' \mid S_t) = J_{\text{admit}} - J_{\text{move}} - J_{\text{redo}} - J_{\text{revoke}} \qquad (\text{Eq3})$$

$$\text{Score}(S' \mid S_t) = J^{\text{age}}_{\text{admit}} - J_{\text{move}} - J_{\text{redo}} - J_{\text{revoke}} + V_{\text{reuse}} \qquad (\text{Eq13})$$

**所有项的量纲统一为秒**——这是这个效用函数最值得称道的设计选择：它不是「加权求和一堆异质分数」，而是**把收益和代价都折算成时间**，权重问题因此在很大程度上被消解了（只剩 $\beta$ 和 $\rho$ 两个真正的旋钮）。对照 PRISM 用 $10^6/10^5/1$ 模拟字典序的做法，TOPAS 这里干净得多。

| 项 | 含义 | 公式 |
| --- | --- | --- |
| $J_{\text{admit}}$ | 准入请求带来的**任务最长剩余路径期望缩短量**之和 | Eq.7 |
| $J_{\text{move}}$ | 前缀在 GPU↔CPU 之间搬运的时间 × 受影响任务数 | Eq.8 |
| $J_{\text{redo}}$ | 抢占导致的已生成 token 重新解码时间 | Eq.9 |
| $J_{\text{revoke}}$ | 撤销被抢占请求当初被授予的准入信用 | Eq.10 |
| $V_{\text{reuse}}$ | 一跳前瞻的下游前缀复用价值 | Eq.11 |
| $g_i(t)$ | 任务级 aging 乘子（只乘在 admit 项上） | Eq.12 |

### 3.2 $J_{\text{admit}}$：为什么是「最长剩余路径」而不是「服务时间」

论文说了一句很关键的话：**"A request's service time is not itself task progress."** 在 DAG 里推进一个非瓶颈分支，任务完成时间可能纹丝不动。所以一个请求的价值应该是「它完成之后，任务的**最长剩余 LLM 服务路径**缩短了多少」。

残差建模（这部分是全文技术上最细致的地方）：

设 stage $v$ 在任务 $J_i$ 的总 LLM 服务量 $X_{i,v} \sim F_{a(v),v}$。$F$ 的来源分两段：**有历史完成记录时用经验分布（含所有 ReAct 轮次、排除 tool 等待）；没有时退化为 profiled 值上的点质量**。设 $e_{i,v}(t)$ 是已记录的累计服务，已开始未完成的 stage 残差为

$$Z_{i,v}(t) \overset{d}{=} X_{i,v} - e_{i,v}(t) \ \big|\ X_{i,v} > e_{i,v}(t) \qquad (\text{Eq4})$$

> **Eq.4 处理 ReAct 的方式很巧妙**：它把「重复的 ReAct 轮次」条件化在**已观测到的服务量**上，从而**不需要假设固定的轮数**。已经跑了 3 轮的 stage，其剩余服务的条件分布自然会反映出「这个 stage 倾向于跑很多轮」。而 tool 等待被明确排除在外——tool 影响的是 readiness 而不是可控服务量。这个切分是对的。

对请求 $r$，$\widehat{D}_r(t)$ 是它的期望 LLM 服务贡献（无历史时回退到 $T_{\text{prefill}}(|\text{input}_r|) + T_{\text{decode}} \cdot \text{max\_new}_r$），假想完成后把对应残差更新为 $[Z_{i,v}(t) - \widehat{D}_r(t)]_+$。然后

$$\widehat{L}_i(\mathcal{H}, t) = \mathbb{E}_{\mathbf{Z}_i(\mathcal{H},t)}\big[\text{LP}(G, \mathbf{Z}_i(\mathcal{H},t))\big] \qquad (\text{Eq5})$$

$$\Delta_{\text{task}}(r \mid \mathcal{H}, t) = \widehat{L}_i(\mathcal{H}, t) - \widehat{L}_i(\mathcal{H} \cup \{r\}, t) \qquad (\text{Eq6})$$

$$J_{\text{admit}} = \sum_{r \in A} \Delta_{\text{task}}(r \mid A^{<r}_i, t) \qquad (\text{Eq7})$$

**期望被放在 LP 外面**（$\mathbb{E}[\text{LP}(\cdot)]$ 而非 $\text{LP}(\mathbb{E}[\cdot])$）——这样关键路径可以随残差实现而改变，而不是被平均时长钉死在一条路径上。论文明确点出了这个设计意图，是对的（Jensen 不等式方向：$\mathbb{E}[\max] \ge \max[\mathbb{E}]$，用后者会系统性低估剩余路径）。

> **但论文完全没说这个期望是怎么算的。** $\text{LP}$ 要在反向拓扑序上对一个随机残差向量求最长路径的期望——这在一般 DAG 上是 **#P-hard** 的经典问题（stochastic PERT）。实际实现只可能是蒙特卡洛采样，那么**采样多少条？**论文一个字没提。这直接关系到 §5 报告的 1.9 ms/决策 这个开销数字可不可信。**这是全文最大的一处实现细节缺失。**（一个可能的解释是：在 Chain-3 这种链式 DAG 上 LP 退化成求和，期望可以闭式算；在 DAG-4 / DAG-10-Wide 上就不行了。而论文恰恰没有分工作负载报告调度开销。）

同一候选内的请求按 §IV-E 的贪心序处理，$A^{<r}_i$ 是同任务中更早的假想完成——**序贯残差更新使得这些项加起来正好等于该任务的联合期望剩余路径缩短量**，避免了「同一任务的两个请求各自宣称缩短了同一段关键路径」的重复计数。这个细节做得干净。

### 3.3 三项代价

**前缀搬运**（Eq.8）：$R_{\text{in}} = R' \setminus R_t$，$R_{\text{out}} = R_t \setminus R'$，$\mu$ 为每前缀 token 的 KV 字节数，$B_{\text{swap}}$ 为交换带宽，**搬运在一次 transition 内串行**：

$$T_{\text{move}}(S' \mid S_t) = \frac{\mu}{B_{\text{swap}}}\left(\sum_{a \in R_{\text{out}}} \ell_a + \sum_{a \in R_{\text{in}}} \ell_a\right), \qquad J_{\text{move}} = T_{\text{move}} \cdot |\mathcal{T}_{\text{in}}|$$

$\mathcal{T}_{\text{in}}$ 是「在等一个 incoming 前缀」的不同任务集合。乘以任务数是因为**每个受影响的任务都要等这段搬运时间**——量纲上仍是「秒 × 任务数」，与 $J_{\text{admit}}$（也是跨任务求和的秒数）一致。常驻前缀的准入与搬运重叠，不计入。

**抢占重算**（Eq.9）：SGLang 通过自回归解码重建被抢占请求的已生成 token KV，$y_r(t)$ 为待重建 token 数：

$$J_{\text{redo}} = \sum_{r \in P} T_{\text{decode}} \cdot y_r(t)$$

**信用撤销**（Eq.10）：准入时刻 $t_a$ 存下条件信用 $\lambda_r = \Delta_{\text{task}}(r \mid A^{<r}_i, t_a)$，抢占时撤销：

$$J_{\text{revoke}} = \sum_{r \in P} \lambda_r$$

> **$J_{\text{revoke}}$ 是这套设计里最容易被略过、但机制上最必要的一项。** 没有它，一个请求可以被反复「准入 → 抢占 → 再准入」，每次准入都从 $J_{\text{admit}}$ 里领一份进度分，而实际上一点活都没干成。它本质上是在防止效用函数被**自我循环刷分**。论文的处理是：信用在兄弟分支推进时保持不变、完成时清除、重新准入时重建。
>
> 这与 GORGO 那篇里的 reward hacking 是同一类问题的不同表现——**任何基于「预期收益」的调度评分函数都必须防止收益被重复计入**，否则最优策略会退化成不断制造「即将获得收益」的状态而永不兑现。

### 3.4 $V_{\text{reuse}}$：一跳前瞻，也是「workflow 感知」的全部兑现处

当前的复用已经被容量约束 Eq.2 反映了（同 agent 请求共享一次 static prefix 成本）。要给**近未来**的复用定价，TOPAS 统计「即将 ready 的下游 stage」：$n_a(t)$ = 分配给 agent $a$ 的、**唯一未完成前驱正在运行中**的 stage 数量。fork 的每个子节点单独计数；join 只有在其他前驱都完成时才计数。

$$V_{\text{reuse}} = \beta \sum_{a \in R'} n_a(t) \frac{\mu \ell_a}{B_{\text{swap}}} \qquad (\text{Eq11})$$

用 $\mu\ell_a/B_{\text{swap}}$（重新加载时间）作为「保住一个昂贵前缀」的价值单位——量纲上仍是秒，与其他项一致。$\beta$ 控制前瞻压力强度。

**为什么只前瞻一跳**：作者的理由是"farther demand depends on unresolved branches, ReAct rounds, and tool outcomes"。这是诚实的。

> **这一项是"workflow-aware"这四个字唯一真正兑现的地方，而它比标题暗示的要弱得多。** 注意 Eq.11 的最后一句限定：**"Idle prefixes remain infeasible, so this term ranks active states rather than prefetching inactive agents."** 也就是说——
>
> **TOPAS 并不能真正"提前把 B 的前缀预取进来"。** 因为 Eq.2 强制 $R' = \{a(r) : r \in Q'\}$（不允许空转的常驻前缀），一个还没有 ready 请求的 agent 的前缀根本进不了 $R'$。$V_{\text{reuse}}$ 能做的只是：**在多个都已有 ready 请求的候选状态之间，偏向那个下游马上要用到的**。这是「延迟淘汰」而不是「主动预取」。
>
> 对比 KVFlow（NeurIPS 2025，论文自己在 §VI 引了）——KVFlow 的 Agent Step Graph 明确做 **prefetching**。TOPAS 在这一点上**反而比它引用的相关工作保守**。所以回到最开始那个问题「workflow 级前缀驻留相比请求级前缀亲和多解决了什么」，诚实的答案是：
>
> 1. ✅ **多解决了「该不该现在切走」**：请求级亲和只会不断加强当前驻留前缀（LPM 的正反馈），TOPAS 通过 $J_{\text{admit}}$ 的最长路径视角知道「继续留在 A 已经不划算了」。**这是真实且重要的增量。**
> 2. ✅ **多解决了「同时留几个」**：请求级亲和从不显式决定驻留集合的**基数**，TOPAS 的 Eq.2 把「多留一个前缀 = 少几个 batch 槽位」明码标价。**这是 Table I 那个 2× batch 差距的直接对策。**
> 3. ⚠️ **没有解决「提前准备」**：一跳前瞻 + 禁止 idle 常驻，使它只能延迟淘汰不能预取。

### 3.5 任务级 aging（Eq.12）

$$g_i(t) = 1 + \rho \log\left(1 + \frac{t - \tau_i}{H_{\text{age}}}\right), \qquad J^{\text{age}}_{\text{admit}} = \sum_{r \in A} g_{\text{task}(r)}(t)\,\Delta_{\text{task}}(r \mid A^{<r}_i, t)$$

$H_{\text{age}}$ 是观测到的服务时间尺度（用于归一化），$\rho$ 控制强度。**只乘在 admit 项上**，不乘代价项——这是对的，否则老任务的抢占代价也会被放大，反而更难被抢占。存储的信用用准入时刻的权重，所以撤销的正好是当初授予的值。

**更新规则在 supplementary material 里，拿不到。**

### 3.6 分层状态搜索（§IV-E / Algorithm 1）

联合状态搜索是组合爆炸的，且任务级效用**在请求之间不可分**（$\Delta_{\text{task}}$ 依赖同任务中更早的假想完成）。TOPAS 把「前缀集合生成」和「请求分配」分开：

**GREEDYPACK($R$)**：给定前缀集合 $R$
1. 保留被 $R$ 覆盖的运行中请求，其余推导为抢占
2. 反复准入「**每单位额外请求特有 KV 预留**所带来的 age-weighted 条件进度最大」的可行请求——即一个**贪心背包的性价比准则**
3. 同任务的边际值在每次选择后更新
4. 为 $R$ 中每个尚未被覆盖的前缀**预留一个准入槽位和容量**，覆盖不了的集合直接拒绝
5. **发出每一个可行的中间分配**（不只是最终的满配状态）
6. **穿过零增益的选择继续打包**，以便暴露互补的 fork 分支，并偏好已被代表的任务

$\Phi(R)$ = 该 $R$ 发出的最佳完整得分，不可行则 $-\infty$。

**前缀集合枚举**：活跃 agent 池 $\mathcal{A}_t = \{a(r) : r \in W_t \cup Q_t\}$
- $|\mathcal{A}_t| \le M$：**全枚举幂集** $\mathcal{P}(\mathcal{A}_t)$
- 否则：从 $R_g = R_t$ 出发做贪心单 agent 添加；**第一次停滞时试一遍所有剩余 agent 对**（pair），若有改进则加入并恢复单添加；另外从 $R_t$ 和最终 $R_g$ 各做一次 REPAIR（单 agent 的 drop 与 swap 修复），保留 incumbent。复杂度 $O(|\mathcal{A}_t|^2)$ 次前缀集合评估

**上限 $K$** 分别作用于准入与抢占：$|A_R| \le K$、$|P_R| \le K$；$K=0$ 表示取消上限。自然完成的请求在搜索前提交，不进入 $P_R$。

$$S^{\text{sel}}_t = \arg\max_{S' \in \widehat{\mathcal{C}}_t} \text{Score}(S' \mid S_t) \qquad (\text{Eq14})$$

**这是一个事件级贪心策略**：每个调度事件重新优化，不做多步规划。

> **搜索设计有两处值得学的细节**：
> - **"穿过零增益继续打包"**：因为 $\Delta_{\text{task}}$ 是最长路径缩短量，**非瓶颈分支上的请求边际收益恰好是 0**。如果贪心在零增益处停下，fork 的两个分支就永远只推进一个——而 DAG 的 join 需要两个都推进。这个看似反常的规则是对「最长路径」这个价值函数的必要补丁。
> - **"为未覆盖前缀预留槽位"**：防止选了一个前缀集合 $R$ 却因为容量被别的 agent 请求吃光而使某个前缀空转（违反 $R' = \{a(r)\}$）。
>
> **但 $M$ 和 $K$ 的取值论文一个都没给。** $M$ 直接决定「什么时候从全枚举退化成贪心」，$K$ 直接决定抢占的激进程度——这两个是超参空间里最重要的，全部缺失。DAG-10-Wide 有 10 个 prefix namespace，$2^{10} = 1024$ 个子集，如果 $M \ge 10$ 就是全枚举，每个子集还要跑一次 GREEDYPACK 和若干次随机 LP 期望——**这与 1.9 ms/决策 的开销数字是否相容，无法核对**（那个开销数字是在 MetaGPT-SOP 上测的，SOP 只有 5 个角色）。

---

## 4. 实验方法

### 4.1 硬件与实现

| 项 | 配置 |
| --- | --- |
| GPU | **单张 NVIDIA A100 80GB** |
| 引擎 | **SGLang v0.5.3**，TOPAS 作为一个 scheduling module |
| 模型 | **Qwen2.5-32B-Instruct**（全部工作负载） |
| KV 预算 | 加载模型权重后 A100 剩余显存全部给 KV cache，**不施加工作负载特定的 KV 限制** |
| 调度开销 | MetaGPT-SOP @ 0.15 task/s 单独测：**平均 1.9 ms / 决策**，累计调度器时间占实验墙钟 **0.31%** |

> **单卡 A100 + 32B 模型这个组合值得算一笔账**：Qwen2.5-32B 的 FP16 权重约 64 GB，A100 只有 80 GB——**留给 KV cache 的大约只有 10–15 GB**。这解释了为什么 Table I 里 batch size 只有 4.7–11.5 这么小的数量级。
>
> 这个配置**极大地放大了论文要研究的效应**：KV 预算越紧，长前缀的驻留成本相对越高，Eq.2 那个约束越吃紧。换一台 8×A100 跑 TP、或者换个 7B 模型，同样的 workflow 下 $\ell_a / B_t$ 会小一个数量级以上，TOPAS 的收益会大幅缩水。**论文没有做任何 KV 预算的敏感性扫描，也没讨论这一点**——这是我认为最影响结论可迁移性的一处方法论缺陷。作者说"we impose no workload-specific KV limit"，语气上像是在强调实验的自然性，但实际效果是**把一个关键的实验变量交给了硬件巧合**。

### 4.2 工作负载

| 名称 | 类型 | 结构 | 到达率扫描 |
| --- | --- | --- | --- |
| **Chain-3** | 合成 | 线性 $A \to B \to C$ | 0.025 – 0.15 task/s |
| **DAG-4** | 合成 | fork/join | 0.025 – 0.15 task/s |
| **DAG-10-Wide** | 合成 | 宽执行结构，10 个 prefix namespace | 0.10 – 0.35 task/s |
| **MetaGPT-SOP** | 真实 prompt | 早期 MetaGPT 的静态五角色流水线 | 0.01 – 0.125 task/s |
| **MetaGPT-TL** | 真实 prompt | 新版星型组织展开成九阶段路径，TeamLeader 与四个专家交替 | 0.005 – 0.0175 task/s |

三个合成工作负载用**固定长度的用户查询和固定的每阶段生成量**，作者明确说这是为了"isolating topology and prefix-state pressure from variations in user content"。两个 MetaGPT 工作负载从 **MetaGPT SoftwareDev 数据集**采样真实 prompt。

任务到达服从 **Poisson**，**采样出的到达 trace 在所有策略间固定共享**——这一点做得规范。

> **关于 trace 真实性的判定**：这不是「真实生产 trace」。MetaGPT-SOP / TL 用的是**真实的 prompt 内容**，但**到达过程是人工 Poisson**，workflow 结构是作者按 MetaGPT 的 SOP 定义手工展开的。所以准确的说法是「真实 prompt + 合成到达 + 合成拓扑」。三个 DAG 工作负载则是**完全合成**的。
>
> **MetaGPT-TL 是被刻意构造来对 TOPAS 有利的**——论文自己说得很直白："alternating a TeamLeader with four specialists **to preserve recurrent reuse of the central prefix**"。而 TOPAS 在 TL 上的收益（22.0%/26.6%）恰好远大于在 SOP 上的（9.8%/4.5%），作者也主动解释了这个差异："SOP advances through five roles in a largely one-pass pipeline, whereas TL repeatedly returns to a central agent, creating more opportunities for explicit residency control."
>
> 这个自我解释是诚实的、机制上也讲得通。但反过来读就是：**在最接近原始 MetaGPT 的那个工作负载（SOP）上，TOPAS 的 p99 JCT 只赢了 4.5%**，而 22.0% 这个被写进摘要的数字来自一个作者自己设计的、"为保持中心前缀的重复复用"而构造的变体。**摘要里"lowering mean/p99 JCT by 22.0%/26.6% on MetaGPT-TL"这句话在没有 SOP 的 4.5% 作为陪衬时是有误导性的。**

### 4.3 Baseline

| Baseline | 出处 | 定位 | 强度评估 |
| --- | --- | --- | --- |
| **FCFS** | — | 按请求到达时间 | 弱，作为下界 |
| **LPM** | — | 最长可复用前缀优先 | **locality-first 的代表**，SGLang 原生策略 |
| **Parrot-FCFS** | Parrot, OSDI'24 | 按**任务**到达时间排序，请求到达时间破平 | 中等，application-aware 但不 prefix-aware |
| **Autellix-LAS** | Autellix, arXiv 2025 | 累计 LLM 服务最少的任务优先 | 中等，program-progress-aware |
| **SPF / SRPT** | — | 更深的 workflow 阶段优先，同深度内 FCFS；Chain-3 上等价于 SRPT | **progress-first 的代表** |

> **Baseline 覆盖了「两个极端 + 两个 application-aware」这四个象限，选得算合理**，特别是 Parrot 和 Autellix 都是这条线上的重要工作。
>
> **但缺了最关键的那一个：KVFlow**（NeurIPS 2025）。KVFlow 是论文 §VI 里唯一被单独拿出来对照的工作——"a workflow-aware KV-cache management system that uses an Agent Step Graph to guide KV-node eviction and prefetching"，**它是与 TOPAS 最直接可比的方法**，而且是同样做 workflow 感知的前缀管理。**它没有出现在 baseline 列表里，论文也没有解释为什么。** 这是本文实验部分最大的一处缺口——所有的 baseline 都是「不知道前缀驻留」或「不知道 workflow」的，**没有一个 baseline 同时知道两者**，于是 TOPAS 的联合决策相对「已有的 workflow 感知前缀管理」到底赢多少，完全无从判断。
>
> 另外，Parrot-FCFS 和 Autellix-LAS 都是作者自己在 SGLang 上**重新实现的策略近似**（"Parrot-FCFS orders ready requests by task arrival time"——这显然只是 Parrot 调度策略的一个简化提取，Parrot 本身还有 DAG 分析、共享提示识别等一整套机制）。这类「把原论文的核心排序规则抽出来在自己的框架里复现」的做法在系统论文里很常见，但确实会系统性地弱化 baseline。

### 4.4 指标

- **主指标**：mean task JCT、p99 task JCT
- **辅助**：first-task completion time、request throughput（**完整结果在 supplementary material 里，拿不到**）
- **工作负载级比较方式**：把每个指标在所报告的各个到达率上**平均**；对每个 JCT 指标，比较对象是**平均值最低的那个 baseline**

> **"per workload and metric, the comparator is the baseline with the lowest average"** 这个比较方式要留意。它意味着 mean JCT 和 p99 JCT 可能在跟**不同的** baseline 比（MetaGPT-TL 就是这样：mean 比 SPF，p99 比 Parrot-FCFS）。这个做法本身是保守的（总是跟最强的比，对 TOPAS 不利），**方向上是加分的**。
>
> 但「在各到达率上平均」这一步就有问题了，和 PRISM 那篇犯的是同一个错：**把轻载点和过载点等权平均**。看 Fig.4(c) MetaGPT-TL，作者自己说"At light load, the policies remain close; as contention grows, TOPAS pulls ahead"——**收益全部来自最高的那一两个到达率**，而这几个点上的 JCT 已经是 1500–2000 秒。等权平均之后得到 22.0%，掩盖了「在轻载区几乎没有收益」这个事实。

---

## 5. 主要结果

### 5.1 端到端 JCT（Fig.3 / Fig.4）

**全部结果都是折线图，论文正文没有给任何一个绝对数值表。** 能提取的只有百分比降幅（相对逐工作负载逐指标的最强 baseline）：

| 工作负载 | mean JCT 降幅 | p99 JCT 降幅 | 比较对象 |
| --- | --- | --- | --- |
| **Chain-3** | **−27.5%** | **−31.7%** | 未点名（图上看是 Parrot-FCFS 或 SPF） |
| **DAG-4** | **−39.8%** | **−49.4%** | 未点名 |
| **DAG-10-Wide** | **−27.7%** | **−30.8%** | 未点名 |
| **MetaGPT-SOP** | **−9.8%** | **−4.5%**（另 throughput **+6.7%**） | SPF（最强 mean-JCT baseline） |
| **MetaGPT-TL** | **−22.0%** | **−26.6%** | SPF（mean）/ Parrot-FCFS（p99） |

从 Fig.3 / Fig.4 的坐标轴可以读出量级：合成工作负载的 mean JCT 在 100–650 秒之间，p99 在 100–1000 秒之间；MetaGPT-SOP 的 mean JCT 在 600–1200 秒，p99 在 750–2000 秒；MetaGPT-TL 的 mean 在 500–1750 秒，p99 在 500–2000+ 秒。

一个论文自己披露的细节：**MetaGPT-SOP 上 LPM 拿到最低 p99 JCT，Autellix-LAS 拿到最高 request throughput**——即在这个最接近原版 MetaGPT 的工作负载上，**TOPAS 在两个指标上都不是最优**。作者选择跟 SPF（最强 mean-JCT baseline）比，才得到了"improves all three metrics"这句话。这个措辞是精确的但也是有选择性的。

> **关于实验工作点：这些到达率低得需要单独讨论。**
>
> MetaGPT-TL 的扫描范围是 **0.005 – 0.0175 task/s**——即每 57 秒到 200 秒才来一个任务。而 mean JCT 是 500–1750 秒。用 Little 定律粗估：$L = \lambda W$，在 0.0175 task/s × 1200 s ≈ **21 个任务同时在系统里**。九阶段 workflow × 21 个任务 = 系统里有上百个 stage 在竞争一张 A100 的十几 GB KV cache。**这是深度过载。**
>
> 深度过载下所有百分位数本质上都在测「队列积压的斜率」，而不是服务质量。这与 PRISM 那篇的问题完全一样，但 TOPAS 更极端——**它没有任何一个工作点处于"系统能跟上负载"的状态**。Fig.3 / Fig.4 里所有曲线都是单调上升且不收敛的，说明整个扫描范围都在饱和区之上。
>
> 这里有一个替 TOPAS 说话的角度：**多 Agent workflow 服务本来就常常运行在这个区间**（一个软件开发任务花 20 分钟是可以接受的），JCT 以千秒计并不一定意味着"崩溃"。**任务级 JCT 而不是 TTFT 才是这类负载的正确指标**，这一点论文选对了。但即便如此，「所有实验点都在饱和区」仍然意味着**这些百分比降幅无法外推到轻载场景**——而作者自己在 TL 的描述里也承认了 "At light load, the policies remain close"。

### 5.2 消融（Fig.5，MetaGPT-TL @ 0.0175 task/s）

四个变体：**TOPAS-base**（保留基础效用 Eq.3 + 联合状态搜索，去掉 $V_{\text{reuse}}$ 和 aging）、**w/o reuse**、**w/o aging**、**TOPAS**（完整）。

| 对照 | mean JCT 降幅 | p99 JCT 降幅 |
| --- | --- | --- |
| 完整 vs **TOPAS-base** | **−60.5%** | **−53.6%** |
| 完整 vs 两个**单组件**变体 | −44.9% ~ −51.0% | −44.2% ~ −48.5% |

从 Fig.5 的坐标轴读出绝对量级：mean JCT 约 TOPAS-base 2.1×10³ s、w/o reuse 1.5×10³ s、w/o aging 1.7×10³ s、TOPAS 0.85×10³ s；p99 约 2.9 / 2.4 / 2.7 / 1.2 ×10³ s。

> **这个消融的数字大得不正常，而且方向是对论文不利的。**
>
> 按论文的叙事，核心贡献是 §IV-B 的基础效用函数（把前缀驻留和请求准入联合起来、用最长剩余路径估值），$V_{\text{reuse}}$ 和 aging 是两个"additionally incorporated"的辅助机制。但消融显示：**去掉这两个辅助机制，性能就退化 60.5%**。
>
> 更关键的是把它和 §5.1 的主结果放在一起看：**TOPAS 相对最强 baseline 只赢 22.0%，但相对自己的 base 版本赢 60.5%**。反推可知 **TOPAS-base 比所有 baseline 都差得多**——粗算 $2.1 \times 10^3 \times (1 - 0.605)^{-1}$ 那个方向：完整版 mean JCT ≈ 0.85×10³ s，最强 baseline ≈ 0.85/(1−0.22) ≈ 1.09×10³ s，而 TOPAS-base ≈ 2.1×10³ s——**TOPAS-base 的 mean JCT 差不多是最强 baseline 的两倍**。
>
> 这意味着：**论文宣称的核心机制（联合 prefix-request 状态搜索 + 最长剩余路径效用）单独使用时是一个糟糕的调度器**，全部的净收益都来自那两个"辅助"机制。这个结论论文完全没有提，但它就写在 Fig.5 里。
>
> 可能的解释（论文没给）：没有 aging 时，$J_{\text{admit}}$ 会系统性偏向**剩余路径长的任务**（缩短空间大），也就是**新到达的任务**——这与 SRPT 的方向相反，会导致老任务被无限期推后，p99 直接爆炸。也就是说 aging 不是"防饿死的补丁"，而是**在修复基础效用函数本身的一个结构性偏差**。如果这个解释成立，那么正确的写法应该是把 aging 放进核心设计而不是放在 §IV-D 一个小节里。
>
> **消融的另外两个问题**：只在**一个工作负载的一个到达率点**（MetaGPT-TL @ 0.0175，恰好是最高负载点）上做；**没有对联合状态搜索本身做消融**（比如"最长剩余路径效用 + 普通 LRU 淘汰"vs"完整联合搜索"）——而这才是标题里"prefix-state scheduling"那部分的核心。**读者无法知道联合决策相对分离决策值多少。**

### 5.3 调度开销

唯一的开销数字：MetaGPT-SOP @ 0.15 task/s 的单独一次运行中，**平均 1.9 ms / 调度决策**，**累计调度器时间占实验墙钟 0.31%**。

> 0.31% 这个数字看起来很漂亮，但要注意分母：**实验墙钟是以千秒计的**（mean JCT 就有 600–1200 s）。在一个每个请求要跑几十秒的 32B 模型上，1.9 ms 的调度开销当然可以忽略。
>
> **真正的问题是这个数字的代表性**：MetaGPT-SOP 只有 **5 个 agent**，$|\mathcal{A}_t| \le 5$ 意味着幂集只有 32 个子集。而 **DAG-10-Wide 有 10 个 prefix namespace**，幂集 1024 个，每个还要跑 GREEDYPACK 并对每个候选请求算一次 $\mathbb{E}[\text{LP}]$。**论文没有在 DAG-10-Wide 上报告任何开销数字**，也没说 $M$ 取多少（即什么时候切换到 $O(|\mathcal{A}_t|^2)$ 贪心）。考虑到 §3.2 提到的 $\mathbb{E}[\text{LP}]$ 在一般 DAG 上需要蒙特卡洛，**这个 1.9 ms 极可能是最有利的那个配置下测出来的**。

### 5.4 没有的东西

作为对实验完整性的清点，以下内容**论文中完全没有**：

- 任何绝对数值表（除了 Table I 的微基准）
- 任何误差棒、重复实验、随机种子说明
- KV cache 命中率 / 前缀复用率的直接测量（**这对一篇关于前缀驻留的论文是个奇怪的缺失**）
- 前缀搬运次数、抢占次数的统计
- $\ell_a$（agent 前缀长度）的具体数值
- $\beta$、$\rho$、$H_{\text{age}}$、$M$、$K$ 的取值
- KV 预算 / 前缀长度比的敏感性扫描
- first-task completion、request throughput 的完整结果（在拿不到的 supplementary 里）
- 与 KVFlow 的对比

---

## 6. 局限（作者自述 + 我的补充）

**作者自述**：**这篇论文没有 Limitations 章节。** Conclusion（§VII）只有五句话，全部是结果复述，**一条局限都没写**。这在 8 页会议格式下可以理解，但仍然是一个减分项——尤其是 §III 那个 "All tasks follow the same known DAG" 的假设，本应至少有一句话交代。

**我补充的**：

1. **⚠️ DAG 已知且同构是硬前提，且未被讨论**。"All tasks follow the same known DAG"同时要求：DAG 事前已知、所有任务共享同一个 DAG、拓扑静态无条件分支。这直接排除了：开放式 ReAct agent（AutoGPT 式）、多租户混合 workflow、带条件回边的流程（`if 测试失败 then 回到 coder`）。**论文把不确定性限制在了 DAG 节点内部（ReAct 轮数、tool 延迟），拓扑本身是刚性的。** 这是 TOPAS 与 Preble / CacheRoute 这类无需先验知识的方法在部署门槛上的根本差异。

2. **⚠️ DAG 未知 / 动态分支怎么办：论文没有任何降级路径**。一个自然的猜想是 $J_{\text{admit}}$ 会退化（$\mathbb{E}[\text{LP}]$ 无法计算），从而整个效用函数只剩代价项和 $V_{\text{reuse}}$，TOPAS 退化成一个「不搬前缀、不抢占」的惰性策略。但论文没有验证过这个退化行为。**实际系统里 DAG 的一部分（比如 stage 数）通常可以从 orchestration 框架拿到，但每个 stage 的服务量分布需要 warm-up 才有经验分布**——论文提到了这个 fallback（用 profiled 点质量），但**没有测量冷启动期的性能**。

3. **⚠️ workflow 提前终止 / 失败时，被保留的 KV 谁释放：论文完全没有涉及**。这是一个真实的工程问题。TOPAS 的 $V_{\text{reuse}}$ 基于"下游 stage 即将 ready"这个预期给前缀加分，如果上游 stage 抛异常导致整个任务被取消，这个预期就作废了。论文的模型里没有任务失败这个事件，$\mathcal{T}_{\text{in}}$、$\lambda_r$、$n_a(t)$ 这些状态在任务异常终止时如何清理没有定义。**不过要公平地说，Eq.2 的 $R' = \{a(r) : r \in Q'\}$ 约束提供了一个天然的保险**：既然不允许空转的常驻前缀，一旦某 agent 的所有请求消失，它的前缀就自动失去驻留资格。所以这个问题在 TOPAS 里比在 KVFlow 那种真做 prefetch 的系统里要轻。

4. **⚠️ 长尾 workflow 会不会永久占住 KV**：aging（Eq.12）只作用于**准入优先级**，不作用于**前缀驻留**。一个跑得极慢的老任务会因为 $g_i(t)$ 越来越大而不断被准入，它的 agent 前缀就会被反复拉进 $R'$。在有 $K$ 上限的情况下这会挤占其他任务的准入槽位。$\rho$ 和 $H_{\text{age}}$ 控制这个强度，但**这两个参数的取值和敏感性论文都没给**。

5. **重规划开销未被完整测量**（§5.3 已展开）。1.9 ms 是在 5 个 agent 的工作负载上测的，10 个 namespace 的 DAG-10-Wide 上没有数字。$\mathbb{E}[\text{LP}]$ 的计算方法（采样数？闭式？）完全未披露。

6. **单卡 A100 + 32B 模型是一个被硬件巧合决定的关键实验变量**。权重占 64 GB，留给 KV 的只有十几 GB，这把 $\ell_a / B_t$ 推到了最有利于论文主张的区间。**没有 KV 预算敏感性扫描**是最影响可迁移性的一处缺陷。

7. **缺少最相关的 baseline（KVFlow）**。论文在 §VI 里单独把 KVFlow 拿出来对照定位，却不把它放进实验。所有 baseline 都缺失了「workflow 感知的前缀管理」这个维度。

8. **消融揭示的事实与论文叙事不符**（§5.2 已展开）：TOPAS-base 的 mean JCT 约为最强 baseline 的两倍，全部净收益来自被描述为"additionally incorporated"的 aging 和 $V_{\text{reuse}}$。**核心机制的独立价值从未被证明为正。**

9. **没有对联合状态搜索本身做消融**。标题叫"Prefix-State Scheduling"，但没有一个实验回答「联合决策 vs 分离决策（同样的效用函数 + 普通 LRU 淘汰）值多少」。

10. **实验点全部在深度过载区**，且工作负载级比较把所有到达率等权平均。作者自己承认轻载下"policies remain close"。

11. **MetaGPT-TL 是为论文构造的变体**，论文明说是"to preserve recurrent reuse of the central prefix"。摘要里的 22.0%/26.6% 来自这个变体，而更接近原版的 MetaGPT-SOP 只有 9.8%/4.5%，且 p99 和 throughput 上 TOPAS 都不是最优。

12. **全部超参未披露**（$\beta$、$\rho$、$H_{\text{age}}$、$M$、$K$、$\ell_a$），aging 更新规则在拿不到的 supplementary 里。**这篇论文目前状态下无法复现。**

13. **无误差棒、无重复实验、无种子说明**。全部降幅都是点估计。

14. **没有测量任何前缀复用的直接指标**——一篇核心机制是"prefix residency"的论文，却没有报告 KV 命中率、前缀搬运次数、驻留集合大小分布中的任何一个。所有的证据都是端到端 JCT 这个高度间接的指标。**这使得"收益确实来自前缀驻留控制"这个因果链条只有 Table I 的微基准和作者的机制叙述在支撑。**

15. **无代码、无数据、0 引用、未经同行评审**。

---

## 7. 与相关工作的定位

论文自己的 §VI 只有两段，且**没有做任何维度化的对比**——它把相关工作按「引擎优化 / 前缀感知 / agentic 应用」三堆平铺引用，只对 KVFlow 单独说了一句区分。下面这张表是我按对这类工作真正有区分度的六个维度重排的。

### 7.1 六维对比表

| 系统 | 调度单元 | 决策层次 | 规划式 / 反应式 | 准入控制 | 未来需求视野 | 实验规模 |
| --- | --- | --- | --- | --- | --- | --- |
| **TOPAS**（本文） | **workflow / task**（名义上）<br>实际决策粒度是 (前缀集合, 请求集合) 对 | **引擎内**（SGLang scheduler 模块） | **反应式**（事件级贪心，每个调度事件重新优化，无多步规划） | **有**：Eq.2 是准入与驻留的**联合**容量约束 | **一跳**（$V_{\text{reuse}}$ 只数「唯一未完成前驱正在运行」的下游 stage） | **单张 A100 80GB**，Qwen2.5-32B |
| **CacheRoute**（arXiv:2608.19677） | **请求**，但按前缀桶 $b$ 聚合决策 | **入口路由层**（网关，引擎外） | **规划式**：周期性离线算放置表，interval 内**冻结**不再改 | **有**：per-key 准入 $\sum_k k_b \le C$ | **一个 interval**（用前缀到达率 $\lambda_b$ 的**统计外推**） | **60×H100**，176±11 QPS vs Preble 76±11 |
| **PRISM**（同批次） | **batch** | **引擎内**（调度-内存协同） | **反应式**，但**带一个 batch 的前瞻**：QAS 把「下一批要发什么」的 segment 需求分数直接作为 radix 树淘汰键（DART） | 有（车道 + 配额） | **一个 batch** | 单 A800，P99 TTFT −23.3% / −37.1% |
| **Preble**（ICLR'25） | **请求** | **入口路由层** | **反应式**（E2 算法在线打分 + 热前缀自动复制） | 无显式准入 | **无**（只看当前队列与当前缓存态） | 多卡集群 |
| **DualMap**（华为） | **请求** | **入口路由层**（双哈希） | 反应式 | 无 | 无 | 2.25× 有效容量 |
| **KVFlow**（NeurIPS'25，**本文引了但没比**） | **workflow**（Agent Step Graph） | **引擎内**（KV 淘汰 + **预取**） | 反应式 + **真预取** | 无（只管 KV，不管请求准入） | **多跳**（Step Graph 上的 steps-to-execution） | — |
| **Mooncake**（FAST'25 最佳论文） | **请求** | **全局调度器**（KVCache-centric） | 反应式 | **有**，但阈值**手工调** | 无 | 大规模生产 |
| **BanaServe**（同批次，**反对派**） | 请求 | 全局 KV store + 迁移 | — | — | — | 论据薄弱（核心命题只有一张手画示意图，baseline 里**没有任何缓存亲和路由系统**） |
| **Parrot / Autellix**（本文 baseline） | **task / program** | 引擎内排序 | 反应式 | 无 | 无 | — |

### 7.2 三轴定位：TOPAS 落在哪一档

按「**规划式 vs 反应式** × **入口路由 vs 引擎内** × **有无准入**」这三个轴：

**TOPAS = 反应式 + 引擎内 + 有准入。**

- **反应式**：依据是 §3.6 的最后一句——"这是一个事件级贪心策略：每个调度事件重新优化，不做多步规划"。TOPAS 没有任何离线/周期性的规划阶段，没有对未来到达率建模，Eq.14 是一个纯 argmax。它与 CacheRoute 的根本区别就在这里：**CacheRoute 把「未来」编码成前缀桶的到达率 $\lambda_b$ 并据此提前算好一张表，TOPAS 把「未来」编码成 DAG 上的一跳可达性 $n_a(t)$ 并在当下打分。**
- **引擎内**：TOPAS 是 SGLang v0.5.3 里的一个 scheduling module，单卡，**完全不涉及跨实例路由**。它和 CacheRoute / Preble / DualMap **根本不在同一层，不构成竞争关系**——理论上一个集群可以在网关跑 CacheRoute、在每张卡里跑 TOPAS。论文没有说这一点，但这是它最合理的部署形态。
- **有准入**：这是 TOPAS 最不该被忽略的一点。$Q'$ 就是准入集合，Eq.2 $\sum_{a \in R'} \ell_a + \sum_{r \in Q'} d_r(t) \le B_t$ 把「多留一个前缀」和「多准入一个请求」放在**同一个预算里明码标价**。相比之下 KVFlow 只管 KV 淘汰/预取、不管谁进 batch，Preble/DualMap 只管路由不管准入。**「联合」这两个字是 TOPAS 相对全部相关工作最实的差异化点**——可惜 §5.2 的消融恰恰**没有对这个联合性单独做消融**（局限 9）。

### 7.3 与 CacheRoute 的对照：两种「未来」

两篇论文都在做同一件事的不同版本——**用「未来需求」而不是「过去访问」来指导缓存**（LRU 是纯过去，两者都想摆脱它）。区别在于未来从哪来：

| | CacheRoute | TOPAS |
| --- | --- | --- |
| 未来的来源 | **统计规划**：前缀桶的历史到达率 $\lambda_b$ 外推 | **workflow 结构**：DAG 上「B 是 A 的后继」这个硬事实 |
| 未来的可靠性 | 依赖负载平稳；负载漂移则整个 interval 的表都错 | 结构上确定（B 一定会来），但**要求 DAG 已知且同构**（局限 1） |
| 视野长度 | 一个 interval（分钟级，覆盖很多请求） | **一跳**（下一个 stage） |
| 兑现方式 | 冻结的放置表 + per-key 准入 | 打分项 $V_{\text{reuse}}$，且**不能预取**（§3.4） |

> **一个有意思的反讽**：TOPAS 拥有的未来信息在**质量上**远优于 CacheRoute（结构确定 vs 统计外推），但它**用得远比 CacheRoute 浅**——CacheRoute 敢把整个 interval 的放置冻结下来，TOPAS 只敢在打分里加一个一跳的软偏好。信息更好，利用得更保守。原因大概是 TOPAS 没有把「未来到达」纳入优化目标，$\mathbb{E}[\text{LP}]$ 算的是**已在系统内**任务的剩余路径，对**尚未到达**的任务完全没有建模。

### 7.4 与 PRISM 的对照：最像的一篇

PRISM 和 TOPAS 是这批工作里**机制最接近的一对**——都在引擎内、都用「未来需求信息」去改写 KV 保留决策、都是反应式：

- PRISM 的 QAS 把「下一批要发什么」的 segment 需求分数**直接塞进 radix 树的淘汰键**（DART）——未来信息进入的是**淘汰机制**。
- TOPAS 的 $V_{\text{reuse}}$ 把「下一跳要用什么」加进**候选状态的打分**——未来信息进入的是**状态选择**。
- 视野：PRISM 是**一个 batch**（诚实且明确）；TOPAS 名义上是 workflow，**实则也是一跳**。**两者的实际前瞻深度基本相当**，但 PRISM 没有在标题里写"workflow-aware"。

> **PRISM 那篇还有一个发现可以直接用来批评 TOPAS**：PRISM 的冷车道消融显示，**缓存命中率更高的配置 P99 TTFT 反而恶化 16.5%**——命中率和端到端延迟可以反向。这条结论有两面：
>
> - **对 TOPAS 有利的一面**：它说明只报端到端 JCT、不报命中率，方向上是对的——命中率本来就是个错误的优化目标。
> - **对 TOPAS 不利、且我认为更重要的一面**：正因为命中率和端到端指标**可以解耦**，TOPAS 完全不报告任何前缀复用直接指标（局限 14）就变成了一个**因果链条上的硬缺口**。既然两者不必同向，那么「JCT 降了 22%」就**不能反推出「是前缀驻留控制起的作用」**——收益完全可能来自 aging（一个纯排队策略，与前缀无关），而 §5.2 的消融数字恰恰强烈暗示了这一点（去掉 aging + reuse 就退化 60.5%）。**PRISM 做了机制层面的指标（命中率、淘汰行为）并因此能诚实地讨论「命中率涨了但延迟坏了」；TOPAS 一个机制指标都没有，于是连自我证伪的机会都没给自己留。**

---

## 8. 工程借鉴清单

1. **先做 Table I 那个微基准，再决定要不要管前缀驻留。** 固定请求集合、固定 KV 预算，只改顺序（GROUPED vs ALTERNATING），量 mean batch size 和 makespan。这是全文最扎实、成本最低、也最容易在自己负载上复现的实验。如果你的 mean batch 在两种顺序下差不到 20%，后面所有机制都不值得做。

2. **关键判据是 $\ell_a / B_t$（单前缀长度 / KV 预算），不是「有没有长前缀」。** TOPAS 的全部收益都建立在这个比值很大上（单卡 A100 跑 32B，权重吃掉 64 GB，KV 只剩十几 GB）。**换成 8 卡 TP 或 7B 模型，这个比值小一个数量级以上，收益会大幅缩水。** 论文没扫这个参数（局限 6），你必须自己扫。

3. **效用函数的所有项统一折算成「秒」。** 这是 TOPAS 最值得直接抄的工程决策：收益（路径缩短量）、搬运、重算、撤销、复用价值全部是时间量纲，于是「怎么配权重」这个最难调的问题被消解掉了，只剩 $\beta$、$\rho$ 两个真旋钮。对照 PRISM 用 $10^6/10^5/1$ 模拟字典序的做法，TOPAS 干净得多。

4. **请求的价值 = 任务最长剩余路径的缩短量，不是它自己的服务时间。** "A request's service time is not itself task progress." 在 DAG 负载里，推进非瓶颈分支对完成时间毫无贡献。任何 agent 平台的调度器只要拿得到 DAG，这条就应该落地。

5. **期望要放在 max 外面**：用 $\mathbb{E}[\text{LP}(\mathbf{Z})]$ 而不是 $\text{LP}(\mathbb{E}[\mathbf{Z}])$。Jensen 方向是 $\mathbb{E}[\max] \ge \max[\mathbb{E}]$，用后者会**系统性低估剩余路径**，并把关键路径钉死在一条上。

6. **⚠️ 但 $\mathbb{E}[\text{LP}]$ 在一般 DAG 上是 #P-hard（stochastic PERT）。** 论文完全没说怎么算的（局限 5）。工程上的现实选择：链式 DAG 退化成求和可闭式算；一般 DAG 只能蒙特卡洛，**采样数直接决定调度开销**。上线前必须把这个成本量出来，别信论文那个 1.9 ms——它是在只有 5 个 agent 的 MetaGPT-SOP 上测的。

7. **必须有 $J_{\text{revoke}}$ 这类「信用撤销」。** 任何基于「预期收益」打分的调度器，都必须防止一个请求通过「准入 → 抢占 → 再准入」的循环**反复领取同一份进度分**。做法：准入时刻把授予的信用 $\lambda_r$ 存下来，抢占时原额扣回，完成时清除。**这是防止评分函数被自我循环刷分的必要机制，不是可选优化。**

8. **同一任务内多个请求的边际收益必须序贯更新。** 否则同任务的两个请求会各自宣称缩短了**同一段**关键路径，重复计数。TOPAS 用 $A^{<r}_i$（同任务中更早的假想完成）处理，这个细节很干净。

9. **贪心打包要「穿过零增益继续」。** 因为在「最长剩余路径缩短量」这个价值函数下，**非瓶颈分支上的请求边际收益恰好是 0**。停在零增益处，fork 的两个分支就永远只推进一个，而 join 要求两个都推进。这是最长路径价值函数的必要补丁——**如果你抄了第 4 条，就必须一起抄这条**。

10. **aging 只乘在收益项上，不乘代价项。** 否则老任务的抢占代价也被放大，反而更难被抢占，与防饿死的初衷相反。存储的信用用**准入时刻**的权重，撤销时才刚好扣回当初授予的值。

11. **⚠️ aging 很可能不是「防饿死补丁」，而是在修基础效用函数的结构性偏差。** 消融显示去掉 aging + reuse 就退化 60.5%，反推 TOPAS-base 的 mean JCT 约为最强 baseline 的**两倍**（§5.2）。一个合理解释：$J_{\text{admit}}$ 天然偏向**剩余路径长**的任务，也就是**新到达的任务**，方向与 SRPT 相反，老任务被无限期推后、p99 爆炸。**如果你实现了「最长剩余路径」价值函数，请把 aging 当成核心组件而不是补丁，并优先验证它。**

12. **⚠️ 别指望「workflow-aware」能带来预取。** TOPAS 因为强制 $R' = \{a(r) : r \in Q'\}$（禁止空转常驻前缀），**结构上不可能预取还没有 ready 请求的 agent 前缀**，$V_{\text{reuse}}$ 只能做「延迟淘汰」。要真预取得看 KVFlow 的 Agent Step Graph 路线。**先想清楚你要的是延迟淘汰还是预取——这两件事的实现复杂度和风险完全不同。**

13. **「禁止空转常驻前缀」这条约束是个免费的正确性保险。** 它自动解决了「workflow 提前终止/失败时，被保留的 KV 谁释放」这个工程问题：某 agent 的请求全消失，它的前缀立刻失去驻留资格。**做 prefetch 的系统没有这个保险，必须显式设计任务取消时的 KV 回收路径。**

14. **枚举驻留集合是可行的，前提是 agent 数少。** Agent 负载的前缀空间是**闭合且小的**（MetaGPT SOP 只有 5 个角色），这才是 TOPAS 敢做幂集全枚举的前提。**对话负载的前缀空间是开放的（每会话一条链），只能 LRU。** 判断你的负载属于哪一类，决定了整条技术路线。

15. **⚠️ 别用缓存命中率当优化目标，但也别一个机制指标都不报。** PRISM 的冷车道消融证明命中率可以涨而 P99 恶化；但 TOPAS 走到另一个极端，完全不报命中率、搬运次数、驻留集合大小分布，导致「收益来自前缀驻留控制」这个因果链条无法验证。**正确做法是：端到端指标做决策，机制指标做归因。**

16. **⚠️ 部署门槛是这条路线的真实成本。** TOPAS 要求「所有任务遵循同一个已知的静态 DAG」——排除了开放式 ReAct agent、多租户混合 workflow、带条件回边的流程（`if 测试失败 then 回 coder`）。**如果你的 agent 平台是 MetaGPT / ChatDev 式 SOP 驱动的，这条路线成立；如果是 AutoGPT 式开放规划的，直接放弃，回到 Preble / CacheRoute 那种无需先验知识的请求级方案。**

17. **DAG 未知时的降级路径要自己设计。** 论文没有给（局限 2）。自然的退化是 $\mathbb{E}[\text{LP}]$ 算不出来 → 效用只剩代价项 → TOPAS 变成「不搬前缀、不抢占」的惰性策略。另外**冷启动期没有经验分布**，只能用 profiled 点质量，论文也没测过这段的性能。

18. **这套机制的收益窗口在过载区。** 论文全部实验点都在深度过载（Little 定律粗估同时有 ~21 个九阶段任务在抢一张卡的十几 GB KV），作者自己承认「轻载下各策略接近」。**如果你的系统大部分时间不排队，别做这个。**

---

## 9. 个人评述

### 亮点

1. **问题提法是对的，而且是新的。** 把「哪些 agent 前缀该常驻显存」从 cache 淘汰策略的**副产品**提升为**显式的调度变量**，并与请求准入放进同一个容量约束——Eq.2 那一行是全文最精炼的贡献。在此之前，前缀驻留要么由 LRU 决定（引擎内），要么由路由亲和间接决定（Preble / DualMap / CacheRoute），**没有人把它和「谁进 batch」当成一个联合决策**。KVFlow 管淘汰不管准入，PRISM 管准入也管淘汰但视野只有一个 batch 且不以任务完成时间为目标。**这个联合性是真实的空白。**

2. **Table I 的微基准值得单独记住。** 同样的请求集合、同样的 KV 预算、只改顺序，mean batch 从 4.7 变 9.3、makespan 差 1.8–1.9×。这个实验比论文后面所有 JCT 百分比加起来都更有说服力，而且**任何人在自己负载上半天就能复现**。它揭示的机制朴素得几乎是常识——长前缀的 KV 占用是乘在 batch 容量上的固定成本——但把它量化出来是有价值的。

3. **效用函数的量纲统一（全部折算成秒）是漂亮的工程决策。** 它把「多目标加权」这个最容易被审稿人和线上事故一起打脸的问题，降级成了两个真旋钮。对照 PRISM 用 $10^6/10^5/1$ 模拟字典序，高下立判。

4. **$J_{\text{admit}}$ 用「最长剩余路径缩短量」而不是服务时间**，且**期望放在 LP 外面**（$\mathbb{E}[\max] \ne \max[\mathbb{E}]$，用后者会系统性低估）——这两处论文都明确点出了设计意图，说明作者是想清楚了的，不是套公式。

5. **Eq.4 处理 ReAct 的方式很巧妙**：把剩余服务量条件化在**已观测服务量**上，从而不需要假设固定轮数；同时把 tool 等待明确排除（它影响 readiness 而非可控服务量）。这个切分是对的。

6. **$J_{\text{revoke}}$ 和「穿过零增益继续打包」这两个细节体现了实现深度。** 前者防止评分函数被「准入-抢占-再准入」循环刷分；后者是最长路径价值函数在 fork/join 上的必要补丁。**这两条都是只有真写过代码才会遇到的问题**，它们的存在让我相信这套东西确实跑起来了。

7. **§II 的 Strawman 对照做得聪明。** 用一个粗糙到近乎可笑的干预（C 的 backlog 积累就优先 C）同时打赢 LPM 和 SRPT，从而把论证从「我们的方法好」降级到「主动前缀状态控制这件事本身有价值」——**后者是一个更弱但更可信、也更有普适性的主张**。

8. **指标选对了。** 多 Agent workflow 服务的正确指标是**任务级 JCT** 而不是 TTFT/TPOT。这一点看似平凡，但一批同期工作还在用请求级指标衡量 agent 负载。

### 需要留意的

1. **⚠️「workflow-aware」这四个字被兑现得远比标题浅——这是全文最大的落差，也是我对这篇论文最主要的保留。**

   标题和摘要暗示的是**DAG 级的规划**：知道整条流水线的结构，因此可以对前缀驻留做全局安排。而实际兑现「workflow 感知」的只有一处—— $V_{\text{reuse}}$（Eq.11），它只统计「**唯一未完成前驱正在运行中**」的下游 stage，即**一跳前瞻**。更要命的是 Eq.11 后面那句限定：*"Idle prefixes remain infeasible, so this term ranks active states rather than prefetching inactive agents."* 配合 Eq.2 强制的 $R' = \{a(r) : r \in Q'\}$（禁止空转常驻前缀），结论是：

   **TOPAS 结构上不可能提前把下游 agent 的前缀准备好。它能做的只是「在多个都已有 ready 请求的候选状态里，偏向下游马上要用到的那个」——这是延迟淘汰，不是预取。**

   这个落差为什么重要？回到 §1 提的那个核心问题：**workflow 级调度相对请求级前缀亲和，多解决了什么？** 最有力的答案本应是「**workflow 内后续步骤的到达是结构可预测的，所以可以主动保留 KV 而不是被动 LRU**」——这是从「猜未来」到「读 DAG」的信息升级，也是唯一能证明「必须上升到 workflow 粒度」的理由。而 TOPAS 只兑现了一跳的软偏好，**它拥有的未来信息（DAG 结构，确定性的）在质量上远好于 CacheRoute 的统计外推，却用得比 CacheRoute 保守得多**——CacheRoute 敢冻结整个 interval 的放置表，TOPAS 只敢加一个打分项。

   更尴尬的是：论文自己在 §VI 引用的 **KVFlow 明确做 prefetching**。**TOPAS 在「workflow 感知」这个它写进标题的维度上，反而比它引用的相关工作更保守，而且没有把 KVFlow 放进 baseline。**

   公允地说，TOPAS 确实有两项真实增量：(a) 知道「什么时候该切走」（请求级亲和的 LPM 只会正反馈地加强当前驻留，TOPAS 通过最长路径视角知道继续留在 A 不划算）；(b) 知道「同时该留几个」（Eq.2 把「多留一个前缀 = 少几个 batch 槽位」明码标价，这是 Table I 那个 2× 差距的直接对策）。**但这两项都不需要"workflow 级规划"，只需要"任务级进度感知 + 显式容量记账"。真正需要 DAG 的那部分（预测未来到达并提前准备），恰恰没做。**

2. **⚠️ 消融的数字与论文叙事矛盾，而论文回避了这个矛盾。** 完整版 vs TOPAS-base 差 60.5%，而完整版 vs 最强 baseline 只差 22.0%——反推 **TOPAS-base 的 mean JCT 约为最强 baseline 的两倍**。也就是说：**被论文当作核心贡献的「联合 prefix-request 状态搜索 + 最长剩余路径效用」，单独使用时是一个明显差于 SPF 的调度器**，全部净收益来自两个被描述为 "additionally incorporated" 的辅助机制。这个结论就写在 Fig.5 里，论文一个字没提。

3. **⚠️ 没有对「联合性」本身做消融——而这是标题里 "Prefix-State Scheduling" 的核心。** 缺的那个实验很明确：**同样的效用函数 + 普通 LRU 淘汰（分离决策）vs 完整联合搜索**。没有它，读者无法知道 TOPAS 最独特的主张值多少钱。

4. **⚠️ 完全没有前缀复用的直接指标。** 一篇核心机制是 prefix residency 的论文，不报 KV 命中率、不报搬运次数、不报驻留集合大小分布。结合 PRISM 那条发现（命中率与 P99 可以反向），这意味着**「JCT 降了 22% 是因为前缀驻留控制」这个因果推断完全没有证据支撑**——收益完全可能主要来自 aging，一个与前缀无关的纯排队策略。而消融数字恰恰指向这个方向。

5. **⚠️ "All tasks follow the same known DAG" 这个强假设未被讨论。** 它同时要求 DAG 事前已知、所有任务同构、拓扑静态无条件分支。排除了开放式 ReAct、多租户混合 workflow、条件回边。**论文没有 Limitations 章节，Conclusion 只有五句结果复述，这个假设一句话都没有交代。** DAG 未知/动态分支时的降级路径、workflow 提前终止时的状态清理（$\lambda_r$、$\mathcal{T}_{\text{in}}$、$n_a(t)$ 怎么回收）、长尾任务被 aging 不断抬升导致其前缀永久占住 $R'$——**这三个真实的工程问题论文全部没有涉及**。（其中第二个问题因为「禁止空转常驻前缀」的约束而天然减轻，算是运气好而非设计使然。）

6. **⚠️ 实验规模与工作点都在最有利的角落。** 单张 A100 跑 32B（权重 64 GB，KV 只剩十几 GB，$\ell_a/B_t$ 被推到最大）、所有到达率都在深度过载区、工作负载级比较把轻载点和过载点等权平均（掩盖了「轻载几乎无收益」）、**MetaGPT-TL 是作者为论文构造的变体**（论文明说是 "to preserve recurrent reuse of the central prefix"，而摘要里的 22.0%/26.6% 正来自它；更接近原版的 MetaGPT-SOP 只有 9.8%/4.5%，且 p99 和 throughput 上 TOPAS 都不是最优）。

7. **⚠️ 无法复现。** $\beta$、$\rho$、$H_{\text{age}}$、$M$、$K$、$\ell_a$ **一个取值都没给**；aging 更新规则、完整的 throughput / first-task 结果全在 arXiv v1 **不包含**的 supplementary material 里；$\mathbb{E}[\text{LP}]$ 的计算方法（采样数？闭式？）完全未披露，直接使 1.9 ms/决策 这个开销数字不可核对；无代码、无误差棒、无重复实验、无种子。**这不是「细节不足」，是「论文当前状态下第三方无法验证任何一个数字」。**

8. **它和 CacheRoute / Preble / DualMap 不构成竞争关系，论文却没说清楚。** TOPAS 是引擎内单卡调度，那三者是入口路由。合理的部署形态是**网关跑 CacheRoute、每张卡里跑 TOPAS**。论文没有讨论这个分层，也没有讨论多实例场景下自己的机制会怎么变（比如驻留集合的选择是否应该跨实例协调）。

### 一句话定位

**TOPAS 提出了一个正确且新的问题——「agent 前缀是否常驻显存」应当与「哪些请求进 batch」在同一个显存预算下联合决策——并给出了一个量纲干净、细节扎实的效用框架；但它把「workflow-aware」兑现成了一跳前瞻的软偏好而非 DAG 级规划，真正需要 DAG 的预取能力因为「禁止空转常驻前缀」的约束被结构性排除，加上消融显示净收益主要来自被当作补丁的 aging、全部超参未披露、无任何前缀复用直接指标，因此它目前的价值在于**问题提法和 Table I 那个微基准**，而不是它报告的任何一个百分比。**

---

## 附：与本仓库其他解读的交叉引用

| 解读 | 关系 |
| --- | --- |
| [CacheRoute-论文解读.md](./CacheRoute-论文解读.md) | **最有价值的对照组**。两者都用「未来需求」而非「过去访问」指导缓存，但未来的来源不同：CacheRoute 来自**统计规划**（前缀桶到达率 $\lambda_b$，周期离线算表、interval 内冻结 + per-key 准入 $\sum_b k_b \le C$），TOPAS 来自 **workflow 结构**（DAG 后继关系）。**反讽在于 TOPAS 的信息质量更高却用得更浅**（见 §7.3）。层次上也互补：CacheRoute 在入口路由层（60×H100，176±11 QPS vs Preble 76±11），TOPAS 在引擎内单卡，**理论上可以叠着用**。 |
| [PRISM-论文解读.md](./PRISM-论文解读.md) | **机制上最像的一篇**，同批次工作。都在引擎内做调度-内存协同、都用未来需求指导 KV 保留：PRISM 的 QAS 把「下一批要发什么」的 segment 需求分数直接作为 radix 树淘汰键（DART），单 A800 上 P99 TTFT 降 23.3%/37.1%。**视野对比：PRISM 明确是一个 batch，TOPAS 名义 workflow 实则一跳——实际前瞻深度相当，但 PRISM 没在标题里写 workflow-aware。** PRISM 的冷车道消融（命中率更高但 P99 反而恶化 16.5%）是批评 TOPAS 完全不报命中率的关键弹药，见 §7.4。 |
| [DualMap-论文解读.md](./DualMap-论文解读.md) | 华为双哈希路由，2.25× 有效容量。**它的负载统计是 TOPAS「面向 Agent 负载」这个动机的最好外部佐证**：48% 的对话请求 vs **76% 的工具/Agent 请求**共享 ≥50% 前缀。TOPAS 自己**没有做任何负载统计**来支撑「agent 前缀确实又长又少」（§2.1 的实证缺口），这个数字正好补上。 |
| [Preble-论文解读.md](./Preble-论文解读.md) | ICLR 2025，**请求级前缀亲和的代表作**（E2 算法 + 热前缀自动复制）。TOPAS 的全部立论都是相对它这一类方法的：请求级亲和只能看到「现在队列里谁能复用现有前缀」，是反应式的、会在 LPM 式正反馈下把服务钉在 workflow 入口。**读 TOPAS 前应先读它，才能理解 §II 的两个 diagnostic 在打谁。** |
| [BanaServe-论文解读.md](./BanaServe-论文解读.md) | 同批次的**反对派**：主张放弃前缀亲和路由、改用全局 KV store + 迁移。与 TOPAS 方向完全相反（TOPAS 是把前缀亲和推到更精细的联合决策）。**但该解读已核实其论据很弱**——核心命题只有一张手画示意图，baseline 里没有任何缓存亲和路由系统。**作为「这条路线是否走错了」的反方意见可以参考，但证据强度不足以动摇 TOPAS 的前提。** |
| [Mooncake-论文解读.md](./Mooncake-论文解读.md) | FAST'25 最佳论文，KVCache-centric 全局调度，**有准入但阈值手工调**。与 TOPAS 的对照点在准入设计：Mooncake 的准入是一个手调阈值，TOPAS 的准入是效用最大化的产物（Eq.2 联合约束下的 $Q'$）。**TOPAS 在这一点上更有原则性，代价是引入了一堆同样没披露取值的超参。** |
| [LLM服务负载均衡与调度-文献综述.md](./LLM服务负载均衡与调度-文献综述.md) | 整体脉络。TOPAS 在其中的坐标：**反应式 + 引擎内 + 有准入**，调度单元名义上是 workflow（实际是「前缀集合 × 请求集合」的联合状态），未来视野一跳。 |

---
