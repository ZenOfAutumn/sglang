# PrefixShield: Preserving Admission Responsibility in Multi-Tenant LLM Prefix Caches —— 论文解读

> 论文：*Preserving Admission Responsibility in Multi-Tenant Large Language Model Prefix Caches*
> 作者：**Zhiyu Wang、Rajkumar Buyya**（University of Melbourne，School of Computing and Information Systems，Quantum Cloud Computing and Distributed Systems / qCLOUDS Lab）
> 版本：**Preprint**，arXiv:2608.01657**v1** [cs.DC]（2026-08-03），未经同行评审，0 引用
> 代码与数据：**全文未给出任何仓库链接、artifact 声明或数据发布承诺**——一篇声称改造了 vLLM V1 prefix-caching 路径的系统论文，所有数字均不可独立验证
>
> ✅ **系统名核实结论（本次任务的首要核实项）**：**综述的命名是对的**。论文摘要第 4 句即写 "we propose **PrefixShield**"，正文 §I、§III 标题、Table IV、结论段均以 PrefixShield 自称，不存在"综述自行造名"的问题。可以把综述里 `PrefixShield` 条目的「⚠️ 作者与发表场所未逐一核实」中的**作者部分**去掉：作者是 Zhiyu Wang 与 Rajkumar Buyya（墨尔本大学，Buyya 是云计算领域的高产作者）。
>
> ⚠️ **但发表场所仍需保留警告**：论文**没有任何 venue 标注**（无 camera-ready 页眉、无 copyright block、无 "to appear in"）。排版是 IEEE 双栏 + `Index Terms`，说明目标是 IEEE 系会议/期刊，但截至本笔记撰写时**仅有 arXiv 预印本状态**。综述若曾标注具体发表场所，应改回「预印本」。（与同批次 BanaServe 的「已发表于 SPE」缺乏实证是同一类问题，此处至少论文自身没做任何过度声称。）
>
> ⚠️ **阅读提示**：这篇论文的**问题陈述价值远高于其实验价值**。它提出的 "admission-responsibility gap"（准入责任缺口）是本仓库其他论文都没有正面处理的维度，值得单独立一节。但它的实验配置极端到接近人造：512 个 KV block（= 8192 token）的缓存池、受害者热集占用 75% 可用容量、唯一的"公平"baseline 被作者自己配置成必然失败。**摘要里最醒目的两个数字（84.87% vs 4.92%、35.16pp）一个来自最有利的容量点，一个来自与自身消融变体的对比而非任何 baseline。** 另外必须先说清楚一件事：论文标题里的 "multi-tenant"**不包含跨租户内容共享**——原型把前缀查找按租户做了 namespace 隔离，所以它不涉及任何跨租户 KV 泄漏/侧信道问题。详见 §2.4 与 §6。

---

## 1. 一句话概括

共享前缀缓存把 GPU 显存变成了**跨请求存活的持久状态**，于是产生了一个既有机制都没覆盖的错配：**制造缓存压力的租户，和被淘汰掉状态的租户，不是同一个人**——请求级调度器（VTC 一类）只核算"请求在跑的时候"的服务量，替换策略（LRU / S3-FIFO）只排序"对象值不值得留"，静态配额则会把空闲容量锁死。作者把这个错配命名为 **admission-responsibility gap**，并提出 PrefixShield：**对每个租户新物化的完整 KV block 计量、记入一个 token bucket 的透支（admission debt），债务未清时禁止其状态被提升到高保留等级，淘汰时用「投影债务」先选出该谁掏血、再用保留等级选具体块**。在 vLLM 0.10.2 上实现，one-touch 污染下受害者命中率比 LRU 高 9.39pp、比 S3-FIFO 高 8.64pp。

用一句更朴素、也更准确的话重述它的机制：**PrefixShield 就是给每个租户的「新块创建速率」加了一个令牌桶限流器，但透支时不拒绝服务，而是把透支额度拿去做两件事——(a) 当淘汰受害者组的选择键，(b) 当保留等级晋升的门禁。** 这个朴素描述比论文的"责任"话术更有助于判断它的边界（见 §6.2）。

---

## 2. 它要解决的问题

### 2.1 持久状态改变了隔离问题的性质

论文 §II-B 的论证是全篇最扎实的部分，值得原样搬过来：

> Unlike bandwidth or request-time compute, cached KV state **remains after the creating request completes** and continues to shape later replacement decisions.

带宽和算力是**瞬时资源**——请求结束，占用就归还了，所以 VTC（Sheng et al., OSDI'24，论文引用 [8]）那种"核算已处理的客户端工作量"的公平机制在原理上是自洽的。但 KV cache 不是：请求 A 结束之后，它塞进去的 block 还在，并且**继续参与后续的替换竞争**。这是一个跨请求的外部性。

作者进一步给了一个非常重要的区分（这一段是本文对整个领域的贡献）：

> A request that hits a long cached prefix may generate **almost no new KV state**, whereas a similarly sized request containing an unseen suffix may materialize **many new blocks**.

也就是说：**请求速率不是缓存压力的可靠代理变量，请求 token 长度也不是**。两个 token 数完全相同的请求，一个命中 1500 token 前缀（$n_r = 0$，零新增压力），另一个全是未见前缀（$n_r = 96$ 个块）。任何按请求数或按 token 数计费的机制都会把这两者混为一谈。这个观察直接否定了"用 QPS 配额做缓存隔离"的朴素方案。

### 2.2 三个已有设计点，中间有个空洞

| 机制 | 代表 | 做的是什么 | 缺什么 |
| --- | --- | --- | --- |
| 全局 LRU | vLLM 原生 [7] | work-conserving，空闲容量随便借 | 状态进了缓存之后，替换完全不看是谁造的压力 |
| 复用感知替换 | S3-FIFO [9]、TinyLFU [12] | quick demotion + 复用证据估计**对象价值** | 只回答"哪个对象值得留"，不回答"哪个组该掏血" |
| 静态每组配额 | FairRide [10]、NyxCache [11] | 硬隔离 | 有组空闲时容量被锁死（strand idle capacity） |
| 准入策略 | TinyLFU [12] | 判断对象是否**值得**进缓存 | 进了之后没人对压力负责 |

作者要的操作点是：**全局共享、可自由借用，但保留"状态创建"与"后续回收"之间的因果关系**。

> 这个 taxonomy 我认为是成立的，而且论文 Table IV 把它压缩成了一张"每条工作线回答什么问题"的表，是全文最好的一张图：
> - KV reuse and placement（Online Context Caching / Mooncake / MELL）→ 可复用状态该放哪、怎么迁、怎么供给？
> - Replacement and admission（S3-FIFO / TinyLFU）→ 哪些对象值得留/值得准入？
> - Service and cache sharing（VTC / FairRide / NyxCache）→ 服务与缓存资源该怎么在客户端间划分？
> - **PrefixShield → 准入之后，谁还对这份持久共享状态的压力负责？**

### 2.3 威胁模型（自洽，但范围收得很窄）

| 维度 | 论文的设定 |
| --- | --- |
| 攻击者身份 | **合法客户端**（不是外部攻击者），能控制自己请求的 token 序列与时序 |
| 能力 | 生成未见前缀、随时间轮换前缀、重放已准入的前缀 |
| 不能做 | 修改他组请求 / 缓存元数据 / KV 内容；不攻破模型、runtime、OS、认证层 |
| 明确排除 | 网络洪泛、纯算力 DoS、内存安全漏洞、**无限量创建独立计费组** |
| 攻击载荷 | ① one-touch pollution [7]（持续注入未见前缀）② low-and-slow rotation [15]（同等压力摊到时间上）③ replay-based pollution [12,16]（重访已准入前缀，伪造复用证据） |

这个威胁模型内部是自洽的，三个攻击载荷分别对应三个设计目标（直接创建 / 时序规避 / 复用洗白），这一点做得干净。

**但有一处必须点出来**：排除项里的「unbounded creation of independently entitled accounting groups」是一个很大的口子。PrefixShield 的全部保护都建立在"计费组是 operator 定义的稳定实体"之上。在真实的公有云 LLM API 场景里，注册新 API key / 新 project 的成本接近零。作者把 Sybil 攻击整体划到范围外，等于说：**这套机制只在计费组与真实经济主体一一对应的托管环境（企业内部平台、有实名与配额审批的云租户）里成立**，对开放注册的公共 endpoint 无效。论文没有讨论这个边界条件，我认为这是威胁模型陈述中最应该展开而没有展开的地方。

### 2.4 ⚠️ 这篇论文**不**处理跨租户 KV 泄漏——原型按租户做了 namespace 隔离

这是我在读的时候最想确认、结论也最重要的一点。论文 §II-C 第二段写得很清楚：

> In the prototype and evaluation, native prefix lookup is **namespace-scoped by accounting group** even though all groups compete for the same physical block pool. **Cross-group interference therefore arises through shared capacity and replacement rather than cross-group content deduplication.**

以及未来工作：

> Future work includes extending responsibility accounting to **cross-group content deduplication**, which requires multi-principal ownership and admission rules.

含义：

1. **A 租户的前缀，B 租户永远命中不了。** 所以本文语境下的"多租户共享"只是**容量共享**，不是**内容共享**。
2. 因此，学界最关注的那类前缀缓存安全问题——**跨租户 prompt cache 的 TTFT 时序侧信道 / 成员推断（"某个 prompt 是否被别人问过"可以通过首 token 延迟推断）**——在这篇论文里**根本不会发生**，因为它被架构性地排除了。论文全文**没有引用任何一篇 prompt-cache 侧信道 / 成员推断文献**，安全叙事完全是 **DoS / cache pollution 方向**，不是 confidentiality 方向。
3. 这同时意味着：**它把跨租户共享的收益也一并放弃了**。真实生产中最大的一块前缀复用来自**公共系统提示词模板**（参见 DualMap 与 WildChat 的观察：跨用户共享主要来自共享模板），namespace 隔离之后这块收益归零。论文没有量化这个代价。

> **对提问角度 ②「多租户共享前缀缓存的安全侧面」的直接回答：这篇论文没有覆盖这个角度。** 它的独特价值不在机密性，而在**可用性/公平性侧的一个具体失效模式（缓存污染的外部性）**，以及一个到目前为止本仓库确实没有的机制类别（把"状态创建责任"做成一个跨请求存活的一等信号）。如果综述里把它归到"安全/侧信道"分支，那是错位的，应该归到"多租户公平与隔离"分支。
>
> 另一个必须记下的推论：**跨租户内容去重与准入责任在设计上是冲突的。** 一旦允许 B 命中 A 载入的前缀，那个 block 的 owner 该记谁？记 A 则 A 为 B 的收益持续背债（真正的"白嫖"），记 B 则 A 可以通过预热来给 B 挖坑，按引用计数分摊则需要多主体所有权规则——这正是作者说的 "multi-principal ownership and admission rules"。**这个问题至今无人解决，是一个明确的开放方向。**

### 2.5 关于"准入责任缺口"的一处常见误读（值得校正）

直觉上很容易把 "admission-responsibility gap" 理解成"**A 把前缀载入缓存，B 白嫖收益，成本记在 A 头上**"。**论文说的不是这个。** 由于 §2.4 的 namespace 隔离，白嫖在原型里不可能发生。

论文说的 gap 是**反方向的外部性**：

> control over shared-state creation is decoupled from responsibility for the **replacement cost** that creation later induces.

即：**A 持续注入一次性前缀 → 撑爆容量 → 触发回收 → 被淘汰的却是 B 的热前缀。** A 制造了**淘汰成本**并把它甩给了共享池。是「污染的外部性」，不是「收益的搭便车」。

两者是不同的问题，而且需要不同的机制：污染外部性可以用创建计量解决（本文），搭便车外部性需要收益分摊/多主体所有权（本文明确列为未来工作）。**综述在描述这篇论文时应当用前者的表述。**

---

## 3. 核心设计

PrefixShield 的原则一句话：**制造了新的共享缓存压力的计费组，应当持续对这份压力负责，直到其 admission debt 恢复为止。责任不阻止使用空闲容量，只在容量被争抢时才变得有后果。**

三个阶段 + 两个跨请求持久信号（论文 Fig. 1）：

| 阶段 | 做什么 | 信号 |
| --- | --- | --- |
| ① 请求路径 | 原生前缀查找区分「复用已驻留状态」与「创建新持久状态」；只对后者计费 | $n_r$ = 新物化的完整块数 |
| ② 跨请求持久信号 | **admission debt $D_g$**（责任 / 状态创建）与 **reuse evidence**（对象价值）并行维护；债务作为晋升门禁 | $D_g$、retention class |
| ③ 争抢下的回收 | Step 1「谁掏血」：投影债务最高的组；Step 2「掏哪块」：该组内保留等级最弱的块 | $\widetilde{D}_g$ → $\kappa(b)$ |

### 3.1 数据结构

| 对象 | 字段 | 说明 |
| --- | --- | --- |
| 每个受管 block | **immutable accounting owner**、service contract、logical prefix id、retention class $\kappa(b)$、prefix depth、insertion order、recent access order、access count | owner 在 block 生命周期内**不迁移**；requester 身份单独跟踪，与 owner 分离 |
| retention class $\kappa(b)$ | $\{\textsc{Probationary},\textsc{Normal},\textsc{Reuse}\}$ | 淘汰序：$\textsc{Prob} \prec \textsc{Norm} \prec \textsc{Reuse}$ |
| 每 (组, 合约) 计费状态 | 块计价余额 $B_{g,c}$、上次 refill 时间 | 惰性 refill，**无后台定时器** |
| 历史 | prefix-reference count、successful-admission history | 用于区分"首次准入"与"已建立的复用" |

元数据维护在 KV payload 之外，**不改变原生 KV 内容或 block hash**——这一点对工程落地很重要，意味着它不破坏 vLLM 原有的 content-hash 查找语义。

论文特意强调：**logical prefix identifier 是策略元数据，与 vLLM 原生的 content-hash 查找键是两套东西**；namespace 隔离走原生 cache namespace 那条路。这个分离设计是干净的。

### 3.2 完全计量的准入责任（Fully Metered Admission Responsibility）

令 $n_r$ 为请求 $r$ 在原生前缀查找**之后**新物化的完整 KV block 数。惰性 refill 的有效余额：

$$\widehat{B}_{g,c}(t) = \min\{\beta_c,\; B_{g,c}(t_0) + \rho_c (t - t_0)\} \qquad (1)$$

扣费：

$$B_{g,c}(t) = \widehat{B}_{g,c}(t) - n_r \qquad (2)$$

组 $g$ 的**原始债务**（跨该组所有服务合约求和）：

$$D_g(t) = \sum_c \max\{0,\; -\widehat{B}_{g,c}(t)\} \qquad (3)$$

$\epsilon \ge 0$ 是"视为清偿"的实现容差。

几个关键语义，逐条点评：

| 语义 | 论文原文含义 | 我的点评 |
| --- | --- | --- |
| 纯缓存命中 $n_r = 0$，不计费 | 计量的是**状态创建**而非请求活动 | ✅ 这是全文最正确的设计决定，直接落实 §2.1 的观察 |
| 负余额**不抑制准入** | 只表示未清偿责任，保持 work conservation | ✅ 与 CacheRoute 的拒绝式准入形成对照 |
| **淘汰不偿还令牌桶**（"eviction itself does not repay the token bucket"） | 债务只随**时间**恢复 | ⚠️ 这一条决定了 "debt" 这个名字有误导性——它不是欠了容量要还，而是**限流器的透支额度随时间自愈**。见 §6.2 |
| $\rho_c$ / $\beta_c$ 不是配额 | "Neither parameter imposes a cache-occupancy quota or an upper bound on request-driven state creation" | ⚠️ 字面成立，但**实质上 $\rho_c$ 就是一条新块创建速率线**：低于它完全免罚（实验里 $\lambda_{new}/\rho = 0.54$ 时增益恰为 0.00pp） |

**Proposition 1（静默期债务清偿）**：设所有活跃合约 $\rho_c > 0$，$d_{g,c}(t_0) = \max\{0, -\widehat{B}_{g,c}(t_0)\}$，$\tau_g(t_0) = \max_{c: d_{g,c}(t_0)>0} d_{g,c}(t_0)/\rho_c$。若 $g$ 在 $t_0$ 后不再物化新块，则对所有 $t \ge t_0 + \tau_g(t_0)$ 有 $D_g(t) = 0$。

> **这个"命题"是三行代数的令牌桶恒等式**：以速率 $\rho$ 填补深度 $d$ 的坑需要 $d/\rho$ 时间，取所有合约的最大值。它不含任何关于 PrefixShield 机制的信息，写成 Proposition + 完整证明是包装。（与 PRISM 的 Theorem 1 是同一种做法。）
>
> 更重要的是：**这是全文唯一的形式化结果，而且它不是公平性保证**。它只保证"惩罚会过期"。论文**没有任何**关于隔离度、公平份额、受害者最坏情况保留量的定理——**回答提问角度 ③ 的最后一问：没有公平性理论保证，一条都没有。**

**计费主体 vs 服务合约**的分离设计值得一记：**raw debt $D_g$ 在组级聚合用于复用门禁，所以换合约标签洗不掉未清责任**；而合约级 refill 速率与归一化权重只影响**回收优先级**。这是一个显式的防绕过设计，考虑得比较周到。

**初始保留等级**由扣费后余额 + 前缀准入历史决定：

- 首次成功准入 → $\textsc{Probationary}$
- 若 $B_{g,c}(t) < -\epsilon$（扣完还欠着）→ $\textsc{Probationary}$，**无论历史如何**
- 否则，同组已准入过的前缀再次**有余额地**准入 → 可进 $\textsc{Normal}$

即："PrefixShield 改变的是**保留敞口**，而不是拒绝新状态"。

### 3.3 债务门禁的复用晋升（Debt-Gated Reuse Promotion）

问题：攻击者注入大量状态后，**重放这些前缀来伪造复用证据**，试图把自产的污染洗成高保留等级条目。

规则：

$$\kappa(b) \rightarrow \textsc{Reuse} \implies \text{ProvenReuse}(b,r) \;\wedge\; D_g(t) \le \epsilon \qquad (4)$$

当 $D_g(t) > \epsilon$ 时，PrefixShield **仍然记录复用、仍然更新访问元数据**，但块保持当前等级。

两条不变式：

- **I1 完全压力归因**：每个新物化的完整 KV block 恰好被计费一次给引入它的组；纯复用不产生任何新的准入责任。
- **I2 复用不可洗钱**：重访前缀可以建立复用证据，但未清偿的准入债务阻止该证据立即把状态提升到 $\textsc{Reuse}$。

> I2 是本文最有原创性的机制。它在概念上等价于说：**"复用证据"在自证场景下是不可信的输入，必须先验证发言者的信用**。这是一个可以推广的思路——任何用"观测到的对象行为"做保护决策的缓存策略，都存在被对象所有者自我制造证据来攻击的可能。S3-FIFO 的 quick demotion 恰恰是这类可被自证攻击的机制。
>
> 但它的强度完全由 $\tau_g = d/\rho_c$ 决定：**攻击者只要等够 $\tau_g$ 秒再重放，门禁就完全失效**。论文的 Fig. 3(b) 自己把这个证明了——见 §5.4。

### 3.4 责任感知 + 复用分段的淘汰（两阶段）

**Step 1：谁掏血。** 归一化：$\rho_{\min} = \min_{c'} \rho_{c'}$，$w_c = \rho_c/\rho_{\min}$，

$$\widetilde{D}_g(t) = \sum_c \frac{\max\{0,\; -\widehat{B}_{g,c}(t)\}}{w_c} \qquad (5)$$

含义：**相对于合约授权额度的超额压力**。同样的原始欠款，$\rho_c$ 更大的高等级合约贡献更少的归一化债务——债务仍然完全计量，只是回收优先级反映"相对于配置的压力"。

候选集 $C$ 中欠债的候选持有组：

$$\mathcal{G}^+(C,t) = \{g:\ \exists b \in C,\ \mathrm{owner}(b) = g,\ \widetilde{D}_g(t) > \epsilon\} \qquad (6)$$

多块分配时维护**临时投影责任分数** $P_g^{(j)}$，初始 $P_g^{(0)} = \widetilde{D}_g(t)$：

$$g_j^* = \arg\max_{g \in \mathcal{G}_j^+} P_g^{(j)} \qquad (7)$$

$$P_{g_j^*}^{(j+1)} = \max\left\{0,\ P_{g_j^*}^{(j)} - \frac{1}{W_{g_j^*}(t)}\right\} \qquad (8)$$

其中 $W_g(t)$ 是组 $g$ 当前欠债合约中的**最大**权重，因此 $1/W_g(t)$ 是该组单块归一化减量的**最小值**——这是一个**保守的、合约无关的**递减，避免假设被回收的块该冲抵哪个合约。平局用保留段 + 原生队列序打破。

> ⚠️ **注意 Eq.(8) 的投影只影响当前分配批次内的受害者选择**：持久令牌桶余额只通过新状态计量与 refill 演化。这个分离是对的（否则淘汰就成了还债，会让 §3.2 的"淘汰不偿还"语义自相矛盾），但它也意味着**同一批次内一个组被连续掏血的次数上界是 $\widetilde{D}_g \cdot W_g$**，超过之后就退回全局分段序——这是一个隐含的、未被讨论的保护上限。

**Step 2：掏哪块。**

$$\textsc{Probationary} \prec \textsc{Normal} \prec \textsc{Reuse} \qquad (9)$$

段内用原生 free-queue 序做 recency tie-break。若没有任何候选持有组有正投影债务，则退回**全局**分段序。

**Algorithm 2（责任感知批量回收）**的顺序值得记：

```
1. S ← TakeTrueFree(C, k)        // 真正空闲的块先分配 —— work conservation 的实现基础
2. m ← k - |S|; if m == 0 return  // 有空闲就完全不触发责任逻辑
5. Q ← SegmentQueuesByOwner(C\S)
7. while m > 0:
8.     G+ ← {g : Q[g,·] ≠ ∅ ∧ P_g > ε}
9-12.  if G+ ≠ ∅: g* ← ArgMaxDebt; b ← FirstSegmentQueue(Q[g*,·]); P_g* -= 1/W_g*
13-14. else:      b ← GlobalFirstSegmentQueue(Q)   // 无人欠债 → 退化为分段 LRU
```

### 3.5 Work-Conserving Sharing

明确的设计承诺：

- admission debt **不定义最大占用**
- **不因为组耗尽了压力预算就拒绝任何 block**
- 有空闲容量时任何组都可以吃掉；**truly free blocks 先于缓存状态被分配**
- 责任感知替换**只在容量被争抢时激活**

净效果：只修改了**保留分类、晋升资格、受害者选择**三件事，原生 cache admission 语义完全保留。

### 3.6 它在哪一层？（回答提问角度 ④）

**引擎层内部的一个计量/替换层。** 具体地：

| 维度 | 定位 |
| --- | --- |
| 实现位置 | vLLM **V1 prefix-caching path** 的 cache-management 路径（block pool / KVCacheManager 一侧） |
| 挂钩点 | 请求分配、前缀匹配后的块物化、cache-touch 事件、cached-block 回收 |
| **不碰** | 模型权重、attention kernel、KV tensor 内容、请求可见的生成语义、**请求调度顺序**、**路由** |
| 作用域 | **单实例**。全文没有任何多副本/分布式内容 |

这是一个重要的定位判断：**PrefixShield 既不是路由层，也不是请求调度器，而是"引擎内的缓存会计层"**。它甚至不改变任何请求的执行顺序——它只改变谁的块先被淘汰。这是本仓库目前唯一一篇把干预点完全放在**淘汰受害者选择**上的工作（PRISM 改的是淘汰**键**，是相邻但不同的位置）。

单实例作用域带来一个论文完全没讨论的漏洞：**在 N 副本部署下，每个实例独立记账，路由器对 debt 一无所知。攻击者把污染流量摊到 N 个副本上，每个副本看到的 $\lambda_{new}$ 只有 $1/N$，可能全部落在 $\rho_c$ 之下从而完全免罚，而受害者的热前缀（被前缀亲和路由钉在某个副本上）仍然承受全部压力。** 要修这个洞需要跨副本的 debt 聚合，也就是需要一个路由层的对等物——论文没有提。

---

## 4. 实验方法

### 4.1 平台与配置

| 项 | 取值 |
| --- | --- |
| 引擎 | vLLM **0.10.2**，开启 Automatic Prefix Caching |
| 默认平台 | **Qwen2.5-1.5B-Instruct @ 单张 NVIDIA L4** |
| 可移植性 | NVIDIA A100 × {Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct} |
| 缓存容量 | **512 个配置 block，511 可用**（vLLM 保留 1 个）；16 token/block ⇒ **总容量 8,192 token** |
| 共享前缀长度 | 1,536 token = **96 block/前缀** |
| 受害者热集 | 4 个热前缀 = **384 block = 511 可用块的 75.1%** |
| 名义负载 | 默认 60% profile；$C_{\text{prefill}}$ 由 32 个 distinct-prefix 请求标定的 LRU 唯一 prefill 吞吐；基准到达率 $\alpha C_{\text{prefill}}$；one-touch 压力相 $\min\{0.75C_{\text{prefill}},\ 1.15\alpha C_{\text{prefill}}\}$ |
| PrefixShield 策略参数 | refill $\rho_c$ = **64 / 40 / 16 blocks/s**，burst $\beta_c$ = **256 / 192 / 128 blocks**（critical / standard / best-effort 三档合约） |
| 统计 | **10 组配对运行**，组内各策略收到完全相同的 trace 与参数；95% bootstrap CI over paired run-level differences |
| 指标 | $h_r = \texttt{cached\_tokens}_r / \texttt{prompt\_tokens}_r$（vLLM 在原生前缀查找后上报）；攻击 trace 取受害者请求均值，benign trace 取全部请求均值；P95 TTFT 同人群 |

作者对容量设置有一句自辩：

> The replacement domain is **deliberately capped** to create controlled, repeatable contention within finite traces; the configured block count is an experimental cache-capacity parameter rather than an estimate of the GPU's maximum KV capacity.

而且策略参数**跨所有报告的负载固定，不逐 trace 调参**——这一点必须给分，比很多逐场景调参的论文诚实。

> ⚠️ 但自辩不能消解后果。**8,192 token 的 KV 缓存池**在任何真实部署里都不存在：一张 L4（24GB）跑 Qwen2.5-1.5B，实际可用 KV block 数是数万级。把池子缩到 512 块并把受害者热集设成 75%，等于**人为制造了一个"任何一次准入都必然触发淘汰"的饱和态**。所有绝对数字（LRU 5.54% 受害者命中率）都是这个配置的产物，不是负载性质的产物。

### 4.2 Baseline

| Baseline | 配置 | 代表的设计点 | 强度评估 |
| --- | --- | --- | --- |
| **LRU** [7] | vLLM 原生全局 work-conserving 替换 | 原生全局共享 | 弱但必要（counterfactual） |
| **S3-FIFO** [9] | 等大 KV 对象、10% small FIFO、90% main FIFO、等大 ghost FIFO、2-bit 计数器上限；**不接收任何租户或责任元数据** | 复用感知的对象价值替换 | ✅ **真正强的 baseline**（SOSP'23），是全文比较的主战场 |
| **StaticQuota** [10,11] | 固定等分每组分区，**不允许跨组借用** | 硬隔离 | ❌ **作者自建的稻草人**，见下 |

**关于 StaticQuota 必须说清楚**：它的受害者分区是 255 块，而受害者热集是 384 块——**分区比工作集还小**。作者自己在正文里承认了两次：

> StaticQuota reaches 1.94% because its 255-block victim partition is **smaller than the 384-block hot set**: fixed isolation does not help when the entitled share is itself smaller than the working set.
> StaticQuota falls to **0.00%** because its 255-block victim partition is smaller than the reuse distance across the four 96-block hot prefixes.

这个坦率是好的，但它没有改变一个事实：**唯一代表"公平/隔离"这条线的 baseline，被实验配置注定为必然失败**。论文引用了 VTC [8]、FairRide [10]、NyxCache [11] 三篇多租户公平/隔离的真正代表作，**一篇都没有实现为 baseline**。这是本文最大的方法论问题，详见 §6.2 与 §7.2。

另一个不对称：S3-FIFO "receives no tenant or responsibility metadata"——**用一个租户感知的策略去打两个租户盲的策略，在一个专门构造的租户级攻击负载上**。这个比较的结论几乎是先验的。真正有信息量的比较应该是 **per-tenant S3-FIFO** 或 **VTC + LRU**。

### 4.3 负载（6 条主 trace，3 类）

| 类别 | Trace | 出处 | 作用 |
| --- | --- | --- | --- |
| 直接与自适应污染 | **one-touch** | [7] KVCache in the Wild (ATC'25) | 持续注入未见前缀 |
| | **low-and-slow** | [15] ServeGen (NSDI'26) | 同等压力时间摊薄（时序规避） |
| | **two-pass replay** | [12,16] TinyLFU / IC-Cache | 重访已准入前缀（复用洗白） |
| 良性流量 | **ShareGPT** | [2] | 对话式前缀复用 |
| | **bursty ShareGPT** | [2,15] | 良性到达集中成短突发 |
| work conservation | **borrowing** | [10,11] | 一组闲置，测活跃对等方能占多少共享缓存 |

三条攻击 trace 分别对应 §2.3 的三种攻击面，映射是干净的。良性 trace 用 ShareGPT 而非更新的负载（WildChat / 生产 trace）是个小缺憾，但可接受。

### 4.4 敏感性扫描的自由度

作者扫了 5 个维度（Fig. 2）：缓存容量（512–4096，**8×**）、$\lambda_{new}/\rho$ 创建压力（0.54–4.23）、名义负载（40/60/80%）、硬件（L4 / A100）、模型（1.5B / 7B）。**每个维度都是单变量扫描，其余固定在默认点**，且**全部以 vLLM 原生 LRU 为共同 counterfactual**（不是 S3-FIFO）。

> 用 LRU 做扫描的 counterfactual 是一个可以理解但会放大数字的选择：主表里 PrefixShield 相对 S3-FIFO 只有 8.64pp，而扫描图里报的是相对 LRU 的 9.39→79.95pp。**Fig. 2 的所有数字都不能与"PrefixShield 比 SOTA 强多少"划等号。** 作者在正文里说明了这个选择的理由（"共同 counterfactual"），但没有在图注里重复警示。

---

## 5. 主要结果

### 5.1 三种污染下的受害者保护（Table I）

| 负载 | 策略 | 受害者命中率 (%) ↑ | P95 TTFT (ms) ↓ |
| --- | --- | --- | --- |
| **One-touch** | LRU | 5.54 | 1440.0 |
| | S3-FIFO | 6.29 | **1814.8** ⚠️ |
| | StaticQuota | 1.94 | 1466.1 |
| | **PrefixShield** | **14.93** | **1428.2** |
| **Low-and-slow** | LRU | 92.64 | 66.3 |
| | S3-FIFO | 98.64 | **34.5** |
| | StaticQuota | 0.00 | 155.3 |
| | **PrefixShield** | **98.90** | 34.7 |
| **Two-pass replay** | LRU | 92.87 | 132.9 |
| | S3-FIFO | 93.96 | 141.3 |
| | StaticQuota | 26.31 | 316.4 |
| | **PrefixShield** | **95.97** | **104.8** |

无攻击时的 LRU 对照：**99.19%**。

配对效应量（10 组配对运行，95% bootstrap CI）：

| 对比 | 负载 | 增益 | 95% CI | 我的判断 |
| --- | --- | --- | --- | --- |
| PS − LRU | one-touch | **+9.39 pp** | [7.94, 10.96] | 显著，但对照物很弱 |
| PS − S3-FIFO | one-touch | **+8.64 pp** | [7.23, 9.43] | **全文唯一有分量的效应量** |
| PS − S3-FIFO | low-and-slow | **+0.25 pp** | [0.06, 0.51] | 统计显著、**实际意义近乎为零** |
| PS − LRU | two-pass replay | +3.10 pp | — | — |
| PS − S3-FIFO | two-pass replay | **+2.00 pp** | [0.93, 3.23] | 小但真实 |

诚实点评：

1. **除了 one-touch，PrefixShield ≈ S3-FIFO。** low-and-slow 的 0.25pp 作者自己都写成 "a statistically positive 0.25-pp advantage"——这是一个措辞非常谨慎的说法，等于承认没有实际意义。two-pass replay 的 2.00pp 稍好。**换句话说，8.64pp 这个 headline 只在"持续、高速率、直接注入未见前缀"这一种攻击形态下成立**，而这恰恰是最容易被入口层速率限制拦住的一种。
2. **绝对值都很低（one-touch 下最好也只有 14.93%）**，作者归因于 384/511 = 75% 的极端配置。这个自辩是成立的，但同时也说明 one-touch 这一列的三个数（5.54 / 6.29 / 14.93）都处在曲线的坍塌段，**在坍塌段做倍数比较（14.93/6.29 = 2.37×）意义有限**。
3. **S3-FIFO 在 one-touch 下的 P95 TTFT 1814.8ms 比 LRU 的 1440.0ms 还差 26%**——即 S3-FIFO 在这个场景里是负优化。这个数字论文没有解释，但它是有信息的：quick demotion 在极端污染下会把有用状态和污染状态一起快速降级。
4. **PrefixShield 在 two-pass replay 下把 P95 TTFT 从 S3-FIFO 的 141.3ms 降到 104.8ms（−25.8%）**，这是延迟侧最好的一个数，比命中率的 2.00pp 更有说服力。

### 5.2 良性流量（Table II）——安全性检查，不是优势展示

| 负载 | 策略 | 命中率 (%) | P95 TTFT (ms) |
| --- | --- | --- | --- |
| **ShareGPT** | LRU | 35.83 | 106.6 |
| | S3-FIFO | 38.85 | **96.5** |
| | StaticQuota | 35.24 | 110.2 |
| | **PrefixShield** | **38.92** | 97.4 |
| **Bursty ShareGPT** | LRU | 37.17 | 190.9 |
| | S3-FIFO | **38.72** | 185.5 |
| | StaticQuota | 36.46 | 205.3 |
| | **PrefixShield** | 38.10 | **168.1** |

作者的表述值得引用，因为它罕见地克制：

> statistical indistinguishability here is **the intended safety result, not evidence that PrefixShield should dominate every benign trace**.

- ShareGPT：38.92 vs 38.85，无差别。
- **Bursty：PrefixShield 38.10 < S3-FIFO 38.72，PrefixShield 实际上更低**——但 P95 TTFT 168.1 vs 185.5 更好。这个"命中率略输、尾延迟更好"的组合与 PRISM 的 $q_{\text{cold}}$ 消融给出的教训一致：**命中率与尾延迟不是同向的**。
- 突发工作负载**没有导致租户被永久归类为有害**：责任随 refill 衰减，债务清偿后合法复用恢复正常保护。这是 Proposition 1 的实验对应，是个合理的安全性论证。

### 5.3 work conservation（Table III）

| 策略 | 命中率 (%) | P95 TTFT (ms) | Borrower Blocks |
| --- | --- | --- | --- |
| LRU | **99.33** | **38.1** | –（无 per-group 计数器） |
| S3-FIFO | 92.78 | 93.6 | 509–510 |
| StaticQuota | 18.18 | 277.1 | 254–255 |
| **PrefixShield** | **99.33** | 38.5 | **509–510** |

**这是全文我认为最干净的一个结果。** 一组闲置时，活跃组占到 511 个可用块中的 **509–510** 个，且命中率与 LRU 完全一致（99.33%）。它证明了 PrefixShield 在无争抢时**确实退化成原生全局 LRU**——G3 达成。

顺带一个有意思的对照：**S3-FIFO 虽然也占到 509–510 块，命中率却只有 92.78%、P95 TTFT 93.6ms（比 LRU 差 2.5×）**。作者由此得出一个正确的区分：**"capacity work conservation" 与 "retention efficiency" 是两件事**——占满了容量不等于留对了东西。S3-FIFO 的 quick demotion 在这条 trace 上把有用状态误降级了。

### 5.4 扫描与消融（Fig. 2 / Fig. 3）

**(a) 缓存容量（vs LRU 配对增益）：**

| 块数 | 8,192 → token 容量 | PS − LRU 增益 |
| --- | --- | --- |
| 512 | 8,192 tok | **9.39 pp** |
| 1024 | 16,384 tok | 12.01 pp |
| 2048 | 32,768 tok | 39.71 pp |
| 4096 | 65,536 tok | **79.95 pp**（LRU 4.92% → PS **84.87%**） |

受害者工作集随容量缩放，在 1024–4096 时占用约 **76.6%** 可用块，one-touch 攻击窗口固定。4096 块时仍发生 **5.64 次全缓存周转**。

> ⚠️ **这个趋势必须谨慎解读。** 增益随容量单调放大到 79.95pp，看起来是"越大越好"，但根因是**攻击窗口固定而容量增长 ⇒ 相对 churn 下降 ⇒ 保护的边际收益进入陡峭段**。它不是"在真实的大缓存上会有 80pp 收益"，而是"在这个特定的攻击强度/容量比上会有"。注意 **LRU 在 4096 块时（4.92%）反而比 512 块时（5.54%）更差**——这本身就说明这组扫描里有不止一个变量在动。摘要把 84.87% vs 4.92% 放在最显眼位置，是选取了最有利的一点。

**(b) 创建压力 $\lambda_{new}/\rho$（本文最有信息量的一张图）：**

| 实测 $\lambda_{new}/\rho$ | 平均债务 (blocks) | PS − LRU 增益 |
| --- | --- | --- |
| 0.54 | **0** | **0.00 pp** |
| 1.02 | 5.69 | 5.11 pp |
| 2.14 | 260.90 | 24.38 pp |
| 4.23 | 901.91 | **59.07 pp** |

作者说明：因为债务门禁与回收优先级都依赖"实测创建 / 配置 refill"，**这条曲线等价于对运营者选择的 $\rho_c$ 的敏感性分析**（refill 减半 ⇒ 实测比值翻倍）。令牌桶通过 $\beta_c$ 吸收短突发，所以 trace 平均比值接近 1 不构成一个不连续阈值。

> **$\lambda_{new}/\rho = 0.54$ 时增益恰为 0.00pp，平均债务恰为 0** —— 这行数字是全文对自己边界最诚实的披露，也是最要命的一行。它明确说明：**创建速率低于配置额度的攻击者完全不受任何惩罚**。作者把它写成 "creation within the configured entitlement accrues no responsibility **by design**"，即这是特性不是缺陷。这在"责任"的语义下自洽，但在"防护"的语义下意味着：**PrefixShield 的保护强度完全由运营者对 $\rho_c$ 的取值决定，机制本身不提供任何与攻击者行为无关的下界。** 见 §6.2。

**(c) 名义负载：** 40% → **20.06 pp**，60% → 9.39 pp，80% → 8.47 pp。**增益随负载单调下降**——这与"越拥挤越需要保护"的直觉相反，论文没有解释。我的猜测是高负载下受害者本身的请求间隔缩短、复用距离变小，LRU 的基线水位被抬高。但这也意味着**在最需要保护的高压场景，PrefixShield 的相对价值最小**，这一点应该被明说而没有被明说。

**(d)(e) 部署：** L4 9.39 pp / A100 4.99 pp；A100 上 1.5B 4.99 pp / 7B 8.28 pp。**A100 上增益比 L4 低约一半**。可移植性成立（都为正），但效应量对硬件敏感。

**开销：**

| 项 | 512 blocks / 8 groups | 4096 blocks / 32 groups |
| --- | --- | --- |
| 受害者选择耗时（mean / P95） | **0.483 / 0.511 ms** | **3.080 / 3.149 ms** |
| Python 元数据占用 | 218.2 KiB | 1.57 MiB |
| 吞吐（60% 负载，LRU vs PS） | **5.955 vs 5.956 req/s** | — |

作者坦白原型选择器是**软件线性扫描**（为了可审计性），并主动指出 4096/32 时 3ms 已"becomes measurable"，需要索引化的 per-group / per-segment 队列。**主动测量而非假设可忽略**，这是加分项。但也要注意：**0.5–3ms 的 CPU 选择器开销发生在 KV 分配的关键路径上**，在 TTFT 只有几十毫秒的低负载场景（如 Table I 的 low-and-slow，34.7ms）这已经是个位数百分比的量级，而吞吐测量是在 5.96 req/s 这种极低速率下做的——**这个开销测量没有覆盖高 QPS 下的关键路径压力**。

**消融 1：reuse-segmented eviction（Fig. 3a，vs 无分段变体）**

| 负载 | 增益 | 95% CI |
| --- | --- | --- |
| One-touch | 3.38 pp | [2.99, 3.76] |
| Low-and-slow | 1.84 pp | [1.21, 2.34] |
| ShareGPT | 2.31 pp | [0.43, 4.54] |
| Borrowing | **0.00 pp** | — |

borrowing 为 0 是符合预期的（有空闲容量时不需要淘汰区分）。one-touch 的 8.64pp 里有 3.38pp 来自第二阶段的分段，剩下约 5pp 来自第一阶段的责任选组。

**消融 2：debt-gated promotion（Fig. 3b，延迟重放）—— 摘要里 35.16pp 的来源**

设置：193 个 replay-eligible block。

| 重放等待 / 设计清偿时间 | 完整 PrefixShield | 无债务门禁变体 |
| --- | --- | --- |
| 0 与 0.5 | 晋升 **0 / 193**，门禁 193 个 | 晋升 **193 / 193** |
| 后续中性压力下的淘汰 | 淘汰 **74 个攻击者块、0 个受害者块** | 淘汰 **2 个攻击者块、72 个受害者块** |
| 受害者命中率 | **99.3%** | **64.1%** ⇒ 差距 **35.16 pp** |
| **1.0（设计清偿时间点）** | 晋升 193/193 | 晋升 193/193 |
| | **两者收敛，命中率相同** | |
| 2.0 | 同上，收敛 | 同上 |

> **必须说清楚两件事：**
>
> 1. **35.16pp 是与自身消融变体的对比，不是与任何 baseline 的对比。** 摘要把它写成 "Delayed replay yields a 35.16-point advantage while debt remains"，紧接在 8.64 / 2.00 这两个 baseline 数字之后。读者很容易误读成"比 S3-FIFO 高 35pp"。这是本文最明显的一处表述框架问题。
> 2. **这张图同时是债务门禁失效的证明。** 在 wait = 1.0（设计清偿时间）时，"measured replay debt has reached zero, **both variants promote all 193 eligible blocks, and their victim hit ratios converge**"。也就是说：**攻击者只要把重放推迟到 $\tau_g = d/\rho_c$ 之后，I2 不变式提供的保护完全归零。** 作者把这写成正面性质（"it does not permanently penalize a group after quiescent recovery"，避免误伤良性突发），这个权衡取向是合理的——但**这个"合理的取向"在有耐心的攻击者面前就是一个可编程的等待时间**。攻击者的成本只是延迟，而延迟对缓存污染攻击几乎是零成本。

---

## 6. 局限

### 6.1 作者自述

| 局限 | 论文位置 | 是否充分 |
| --- | --- | --- |
| 跨组内容去重未支持，需要多主体所有权与准入规则 | §II-C、§VIII | ✅ 明确，且指出了技术困难所在 |
| 原型选择器是软件线性扫描，大候选集下开销可测量 | §IV、§V-D | ✅ 主动测量而非回避 |
| 512 块是实验参数而非 GPU 真实容量 | §V-A | ⚠️ 承认了，但没做真实容量验证 |
| StaticQuota 因分区小于工作集而失败 | §V-B | ⚠️ 承认了，但没换配置重测 |
| 绝对命中率低是因为 75% 极端配置 | §V-B | ✅ 明确 |
| 良性场景只是"统计不可区分"，不是优势 | §V-C | ✅ 罕见的克制 |
| Sybil（无限创建计费组）超出范围 | §II-C | ⚠️ 只做了范围声明，没讨论后果 |

### 6.2 我的补充

**(1) "debt" 是限流器透支，不是可偿还的债务——这决定了机制的强度上界。**

论文明确写了 "eviction itself does not repay the token bucket"，即债务**只随时间自愈**。加上 $n_r$ 只按**创建事件**计费、**不含驻留时长项**，得出一个重要推论：

> **PrefixShield 计费的是"创建速率"，不是"占用量 × 时间"。**

后果是一个未被讨论的攻击面——**蹲位攻击（squatting）**：攻击者以 $\lambda_{new} \le \rho_c$ 的速率**慢慢**灌入大量前缀，全程零债务（Fig. 2b 的 0.54 那一列已经证明），然后重放这些前缀获得 $\textsc{Reuse}$ 等级（门禁不触发，因为从未欠债），从此**永久占据缓存的高保留段**。此时 PrefixShield 的两阶段淘汰会把它排在最后一位被淘汰——**保护机制反而成了攻击者的护身符**。论文的三条攻击 trace 都不覆盖这个模式：one-touch 与 two-pass replay 都是高速率的，low-and-slow 虽然摊薄了时间但显然仍配置在 $\rho$ 之上（否则不会有 0.25pp 增益）。

**要堵这个洞需要一个占用时长项**（例如按 block·second 计费），而这会与 work conservation 直接冲突——因为持有空闲容量本身就会开始计费。这是一个真实的设计张力，论文没有触及。

**(2) 唯一的公平性 baseline 是自建的、被配置成必然失败的。**

论文引用了 VTC（OSDI'24）、FairRide（NSDI'16）、NyxCache（FAST'22），Table IV 还专门给它们列了一行"Service and cache sharing"，**但实现的是自己写的 StaticQuota**，且配置成分区（255）< 工作集（384）。这不是"baseline 弱"，这是**代表性 baseline 缺席**。最起码应该做的对照有三个，一个都没有：

- **per-tenant S3-FIFO**（每租户独立的 S3-FIFO 实例或加权分段）
- **VTC + LRU**（用请求级公平计数器 + 原生淘汰）
- **StaticQuota 但分区 ≥ 工作集**（例如 384/127 的非等分配置）

第三个尤其关键：作者已经证明了 StaticQuota 在 255 < 384 时失败，但没有证明**在合理配置下静态配额也不行**——而"静态配额会 strand idle capacity"这个论点已经被 Table III 单独证明了（254–255 vs 509–510）。所以真正需要的是一张同时展示"配额够大时保护很好、但代价是借不到空闲容量"的表，从而把 PrefixShield 定位成帕累托改进。**这张表不存在。**

**(3) 没有任何公平性理论保证。** 唯一的形式化结果 Proposition 1 是令牌桶的自恢复恒等式，它保证的是"惩罚会过期"，不是"受害者会被保护到什么程度"。相比之下 VTC 有明确的服务公平性论证、FairRide 有近最优性证明。本文的所有保护主张都是纯经验的，且只在一个（512 块 / 75% 热集 / 固定 $\rho_c$）的配置点上被观察到。

**(4) 实验规模是本仓库同批次里最小的。** 单张 L4 + 1.5B 模型 + 8,192 token 缓存池。作为对照：CacheRoute 用 60×H100，Preble 用多副本 A100 集群。PrefixShield 的机制是引擎内单实例的，所以不需要大集群——**但它需要真实容量**。在真实的数万 block 池上，one-touch 攻击者需要多得多的流量才能撑爆容量，而那个流量水平早就被入口 QPS 限流拦住了。**论文从未论证"缓存污染在真实容量下仍是一个可达成的攻击"**，这是威胁模型与实验之间的一个缺口。

**(5) 无代码、无 artifact、0 引用、未经同行评审。** 一篇声称修改了 vLLM V1 缓存管理路径的系统论文不开源，意味着 Algorithm 1/2 的实现细节（$\epsilon$ 的实际取值全文未给、`ProvenReuse` 的判定阈值 $q_r \ge 1$、logical prefix id 的具体构造）都无法独立验证。**$\epsilon$ 是一个直接决定门禁行为的参数，全文只以符号出现，从未给出数值**——这与 PRISM 未披露 DART 保护预算 $f$ 是同一类问题。

**(6) 单实例作用域下的分布式漏洞**（已在 §3.6 展开）：N 副本各自独立记账 + 路由器不感知 debt ⇒ 污染流量摊薄到 $\lambda_{new}/N$ 即可全面免罚。

**(7) 参数自由度：** $\rho_c$、$\beta_c$ 各 3 档共 6 个数（64/40/16、256/192/128）、$\epsilon$、`ProvenReuse` 阈值、三个保留等级的转移条件——全部手工设定，只有 $\rho_c$ 通过 $\lambda_{new}/\rho$ 扫描被间接探测过。$\beta_c$ 完全没有扫描，而它直接决定突发容忍度（也就是良性突发流量会不会被误伤）——Table II 的 bursty 结果依赖于 $\beta_c$ 恰好够大，换一组 $\beta_c$ 这个"安全性结果"可能就不成立了。

---

## 7. 与相关工作的定位

### 7.1 两种「准入」：性能导向 vs 责任导向

这是本篇解读的主轴。Meta CacheRoute 与 PrefixShield 都做 per-key 准入，但它们**几乎在所有维度上相反**。

| 维度 | **CacheRoute 的准入**（性能导向） | **PrefixShield 的准入**（责任导向） |
| --- | --- | --- |
| 层次 | **入口路由层**（全局规划器） | **引擎内缓存管理层**（vLLM V1 block pool） |
| 准入的对象 | **前缀 key** —— 决定哪些前缀值得配副本 | **计费组** —— 决定哪个组该承担回收压力 |
| 决策变量 | 每前缀副本数 $k_b$，约束 $\sum_b k_b \le C$；按 top-rate 选前缀 | 每组债务 $\widetilde{D}_g$；无任何容量约束 |
| 目标函数 | **最大化整体命中率 / 吞吐**（60×H100 上 176±11 QPS） | **保持"谁造压力谁担责"的因果关系** |
| 对"不值得的前缀" | **拒绝配额** —— 低频前缀拿不到副本 | **从不拒绝** —— 只降低其保留等级敞口 |
| 是否 work-conserving | 否（预算 $C$ 是硬约束） | **是**（明确的设计目标 G3，Table III 验证） |
| 需要的输入 | 全局前缀访问速率统计（跨副本） | 每组新块创建量 $n_r$（单实例本地可观测） |
| 失效模式 | 前缀访问速率**预测偏差**（CacheRoute 自己测出解析式缓存驻留预测中位偏差 14.3pp、p90 44.7pp，最终弃用解析预测改用 shadow replay） | **额度内污染完全免罚**（$\lambda_{new}/\rho = 0.54$ ⇒ 增益 0.00pp）；**等待 $\tau_g$ 后门禁失效** |
| 对抗性假设 | **无**——假设负载是给定的、非对抗的 | **有**——攻击者主动构造 token 序列与时序 |
| 被攻击时会怎样 | 攻击者只要把一次性前缀刷成"高速率"就能骗到副本配额；top-rate 排序**本身就是可操纵的** | 攻击者只要把速率压到 $\rho_c$ 之下就完全隐身 |

**它们能共存在同一个系统里吗？答案是能，而且互补性很强，但存在一个具体的接口冲突。**

- **互补**：CacheRoute 决定"某个前缀在哪些副本上、有几份"，PrefixShield 决定"某个副本内存不下了先淘汰谁的"。前者是**放置**，后者是**回收受害者选择**——正交的两个决策。事实上 CacheRoute 完全没有租户概念（它把所有流量当成一个池子），把 PrefixShield 塞进它的每个 vLLM 实例，能直接补上它缺失的多租户维度。
- **冲突**：CacheRoute 的规划器**假设自己控制哪些前缀驻留**，并据此分配副本预算。PrefixShield 的责任感知淘汰会因为**与前缀价值无关的会计原因**把规划器精心放置的 block 淘汰掉。规划器随后观察到命中率下降，会**加大**该前缀的副本配额——而这会进一步增加该组的 $n_r$、加深其债务、招致更多淘汰。**这是一个正反馈失控回路。** 要避免它，规划器必须能读到引擎的 debt 状态，也就是需要一个**跨层接口**（这与我在 PRISM 笔记里指出的空白是同一个：路由器与引擎淘汰器之间至今没有标准接口）。
- **一个更深的观察**：CacheRoute 的 top-rate 准入**在对抗性负载下是危险的**。它按访问速率选前缀，而访问速率正是攻击者最容易伪造的量（重放即可）。一个 CacheRoute-only 的系统面对 two-pass replay 攻击会主动给攻击者分配副本配额。**PrefixShield 的 I2 不变式（复用不可洗钱）恰好是修这个洞的正确机制**——这可能是两者组合最有价值的地方，而且没人做过。

### 7.2 与 Preble 的 VTC 式公平路由对比（单列一节）

Preble（ICLR 2025）是本仓库里与 PrefixShield 公平视角最接近的工作，值得逐维度拆。

| 维度 | **Preble**（E2 算法 + 缓存感知公平） | **PrefixShield** |
| --- | --- | --- |
| 层次 | **副本间调度器/路由层** | **引擎内缓存管理层** |
| 公平性的计价单位 | **服务量**（VTC 式虚拟计数器，用 cache 命中率调整每请求代价——命中多则计费少） | **新物化的 KV block 数** $n_r$ |
| 干预手段 | **调度优先级** —— 决定请求何时被服务、发往哪个副本 | **淘汰受害者选择 + 保留等级晋升** —— 不改变任何请求的服务顺序 |
| 对攻击者的效果 | 攻击者请求全是 miss ⇒ 代价高 ⇒ **被降低服务优先级 / 被节流** | 攻击者债务高 ⇒ **其自己的块先被淘汰**，服务不受影响 |
| 受害者被保护的方式 | **间接**：攻击者少发请求 ⇒ 少产生新块 ⇒ 少污染 | **直接**：淘汰指针被引导到攻击者的块上 |
| work conservation | 有（工作量守恒的虚拟时钟） | 有（真空闲块优先，无占用上限） |
| 理论保证 | VTC 谱系有服务公平性论证 | **无** |
| 持久状态感知 | ❌ **这正是 PrefixShield 的批评点**：VTC "account for processed client work **while requests are being served**"，请求结束后创建的状态不再计入 | ✅ 责任跨请求存活 |
| 失效模式 | 攻击者**在自己的公平服务份额内**照样能持续污染——因为 VTC 只保证服务份额，不保证缓存份额 | 攻击者**在自己的 $\rho_c$ 额度内**照样能持续污染 |

**最关键的一句总结**：两者其实是**同一个"额度内漏洞"的两个不同计价方式**。VTC 的额度以**服务时间**计价，PrefixShield 的额度以**创建块数**计价。PrefixShield 的论点是后者是更正确的计价单位——**这个论点我认为是对的**，因为 §2.1 已经证明请求量与状态创建量可以差出两个数量级。

但由此也可以看出**两者是叠加而非替代**：

- 只有 VTC：攻击者压低 QPS、每个请求塞满未见长前缀 ⇒ 服务份额内、缓存污染最大化 ⇒ **VTC 完全无感**。
- 只有 PrefixShield：攻击者把创建速率压到 $\rho_c$ 以下，但用海量高频命中请求挤占 GPU 算力 ⇒ **PrefixShield 完全无感**（它根本不改变服务顺序）。
- **两者叠加**：服务侧由 VTC 管、状态创建侧由 PrefixShield 管，两个漏洞互相封堵。**这个组合论文完全没有提，但它是最自然、最有价值的后续工作。** 论文只是把 VTC 引为"请求时公平机制"并说它不覆盖持久状态，没有讨论叠加。

另外必须指出：**Preble 恰恰是 PrefixShield 应该实现而没有实现的 baseline**。Preble 用 cache 命中率调整公平代价这一点，意味着它**对 one-touch 攻击天然有部分免疫力**（攻击者全 miss ⇒ 代价高 ⇒ 被压制）。所以 "PrefixShield 比 S3-FIFO 高 8.64pp" 这个结论**不能推广成"PrefixShield 比现有公平机制强"**——后者根本没被测过。

### 7.3 与本仓库其他工作的层次定位

| 系统 | 层 | 干预点 | 是否有租户/公平概念 | 是否有准入 |
| --- | --- | --- | --- | --- |
| CacheRoute | 路由层（全局规划） | 前缀→副本放置 + 副本数预算 | ❌ | ✅ 性能导向、拒绝式 |
| Preble | 路由层 + 副本调度 | 请求优先级 + 热前缀复制 | ✅ VTC 式服务公平 | ❌ |
| Mooncake | 分离式架构 + 全局调度 | 请求早期拒绝 | ❌ | ✅ 拒绝式（decoding slot 预测，阈值手工调） |
| PRISM | 引擎层 | radix 树**淘汰键**（下批 segment 需求分数） | ❌ | ⚠️ 重排式（热/冷双车道） |
| DualMap | 路由层 | 双哈希（亲和 + 均衡） | ❌ | ❌ |
| **PrefixShield** | **引擎层（缓存会计）** | **淘汰的受害者组选择 + 保留等级晋升** | ✅ **状态创建责任** | ✅ **责任导向、非拒绝式** |

三种准入的谱系（这个划分对综述可能有用）：

1. **拒绝式**（Mooncake）：判断服务不了就早退，保护的是**服务容量**。
2. **配额式**（CacheRoute）：判断值不值得占资源，保护的是**整体效率**。
3. **责任式**（PrefixShield）：从不拒绝，只重定向压力，保护的是**租户间的因果公平**。

PrefixShield 是本仓库里第三类的唯一实例。

### 7.4 与 PRISM 的一处直接对照

PRISM 与 PrefixShield 都在引擎层动淘汰逻辑，但改的是不同的东西：

- **PRISM 改的是淘汰键**（EvictKey = 下一批 batch 的 segment 需求分数）——它把"未来的调度意图"注入淘汰排序。
- **PrefixShield 改的是淘汰的受害者组**——它把"过去的创建责任"注入淘汰排序。

**一个未来一个过去，一个效率一个公平，二者可以同时存在于同一个两阶段淘汰里**：Step 1 用债务选组（PrefixShield），Step 2 在组内用 PRISM 的 segment 需求分数替代静态的三段保留等级。这个组合是自然的，且比论文当前 Step 2 的 $\textsc{Prob} \prec \textsc{Norm} \prec \textsc{Reuse}$ 三段更有信息量。

还有一个反向的警示：**PRISM 的冷车道消融证明了"命中率是错误的优化目标"（命中率更高但 P99 TTFT 恶化 16.5%）。** PrefixShield 的主指标恰恰是命中率，且 Table II 的 bursty 行已经出现了同样的分叉（PrefixShield 命中率 38.10 < S3-FIFO 38.72，但 P95 TTFT 168.1 < 185.5）。**论文没有把这个分叉当回事，只在正文里陈述了数字。** 如果换成以 P95 TTFT 为主指标重新评估，PrefixShield 在良性负载上的表现反而更好，而在攻击负载上的结论（one-touch P95 1428.2 vs S3-FIFO 1814.8）也更强——**作者选错了 headline 指标，对自己不利。**

---

## 8. 工程借鉴清单

1. **计量状态创建，而不是请求活动。** $n_r = \max(0, n_{\text{full}} - n_{\text{cached}})$ —— 这一行是全文最可直接照抄的东西。任何做多租户 LLM 平台计费/配额的系统都应该有这个计数器，因为它是唯一能区分"命中长前缀的大请求"和"全新长前缀的大请求"的量。按 token 计费会把这两者算成一样。

2. **纯命中 $n_r = 0$，零计费。** 这个语义能直接对齐商业定价（缓存命中的 token 通常打折），是一个会计与技术一致的少见例子。

3. **"复用证据在自证场景下不可信"是可推广的安全原则。** 任何用观测到的对象行为（访问频率、复用次数、recency）来做保护决策的缓存策略，都可以被对象所有者通过自我制造证据攻击。S3-FIFO 的 quick demotion、TinyLFU 的频率草图、LFU 的计数器全都在此列。**门禁的形式（先验证发言者信用，再采信其证据）值得抄。**

4. **两阶段淘汰：先选主体、再选对象。** 把"谁掏血"和"掏哪块"拆开是一个干净的抽象。全局替换策略把这两个决策塌缩成一个，导致所有租户级信息在淘汰时刻丢失。这个拆分本身不依赖 PrefixShield 的具体公式——你可以用任何公平性度量做 Step 1、任何价值度量做 Step 2。

5. **投影分数（Eq.7/8）解决批量分配的贪心问题。** 一次要淘汰 $k$ 块时，不能对每一块都重新用同一个静态债务选组（会把一个组掏干）。用一个只在本批次内递减的临时分数 $P_g$，且递减量取**该组最保守的单块归一化减量** $1/W_g$，是一个正确的小设计。这个模式在任何"按权重批量选择"的场景都适用。

6. **组级聚合防标签绕过。** raw debt 在**组**级聚合用于门禁，合约级权重只影响**排序**。这样"换个合约标签"洗不掉责任。做多租户配额时，要显式区分「责任归属的主体」与「决定额度的配置」，并让前者不可被后者重标签。

7. **惰性 refill 不需要后台定时器。** $\widehat{B}(t) = \min\{\beta, B(t_0) + \rho(t-t_0)\}$ 在读取时求值。这是令牌桶的标准做法但值得重申——在 KV 分配这种高频路径上加定时器是不可接受的。

8. **`TakeTrueFree` 必须在责任逻辑之前。** Algorithm 2 第 1–4 行：先分配真正空闲的块，如果够了就直接返回、完全不进入责任逻辑。这是 work conservation 的实现基础，也是"零争抢时零开销"的保证。做任何隔离机制都应该有这个短路。

9. **主动测量选择器开销而不是宣称可忽略。** 作者测出 512/8 时 0.483ms、4096/32 时 3.080ms，并据此指出需要索引化队列。这个做法应该成为惯例——**软件线性扫描在原型阶段是对的（可审计），但必须测量它什么时候崩**。

10. **元数据放在 KV payload 之外，不改 block hash。** 这保证了策略层可以被完整摘除而不影响引擎正确性，也保证了与原生 content-hash 查找的兼容。做引擎侵入式改造时的一条基本纪律。

11. **⚠️ 反面借鉴：不要只按创建速率计费。** 缺少驻留时长项导致"慢速灌入 + 延迟重放"的蹲位攻击完全免疫（§6.2）。如果你要抄这套机制，至少要考虑加一个 block·second 的占用项，或者对 $\textsc{Reuse}$ 等级设置一个 TTL / 需要周期性重新证明。

12. **⚠️ 反面借鉴：单实例记账在多副本下无效。** 债务必须能跨副本聚合，否则攻击者摊薄流量即可免罚。这需要路由层参与——如果你的部署是多副本的，直接照抄这篇的实现不会给你任何保护。

---

## 9. 个人评述

### 亮点

- **问题命名是真贡献。** "admission-responsibility gap" 这个概念以及支撑它的观察——**KV 状态在创建它的请求结束后仍然存活并继续参与竞争，因此请求时公平机制在原理上无法覆盖它**——是清晰、正确、且此前没有被明确表述过的。仅凭这一点，这篇论文在综述里就应该有位置。
- **"请求速率不是缓存压力的代理变量"这个论证是干净的。** 两个 token 数相同的请求可以产生 0 和 96 个新块。这一句话就否定了"用 QPS 配额做缓存隔离"的整类方案，并且给出了正确的替代计价单位。
- **I2 不变式（复用不可洗钱）是有原创性的机制。** 它识别出"用对象自身的行为证据做保护决策"这一整类缓存策略共有的自证攻击面，并给出了一个具体的修法（先验证发言者的信用）。这个思路比它的具体实现更有价值。
- **两阶段淘汰的抽象拆分（谁掏血 / 掏哪块）是干净且可复用的。** 它与具体的公平度量和价值度量都解耦。
- **work conservation 的证据是全文最扎实的一块。** Table III 显示活跃组占 509–510/511 且命中率与 LRU 完全相同（99.33%），这不是一个"差不多"的结果，是一个精确退化。它证明了 G3 不是口号。
- **多处罕见的自我克制**：明说良性结果"是预期的安全性结果、不是 PrefixShield 应该在每条良性 trace 上占优的证据"；明说 StaticQuota 为什么失败；主动测量选择器开销；策略参数跨所有负载固定不逐 trace 调；$\lambda_{new}/\rho=0.54$ 时如实报告增益为 0.00pp。**这些如实披露的水准明显高于同批次的一些论文。**

### 需要留意的

- **摘要的数字编排有框架问题。** "84.87% vs 4.92%" 取自 4 个容量点里最有利的一个；"35.16-point advantage" 是与**自身消融变体**的对比却被排在 8.64 / 2.00 这两个 baseline 数字之后。诚实的摘要应该写 "**8.64 pp over S3-FIFO under one-touch, 2.00 pp under replay, and statistically indistinguishable on benign traffic**"——这三个数才是它真正的战绩。
- **除 one-touch 外，PrefixShield ≈ S3-FIFO。** low-and-slow 0.25pp（作者自己写成 "a statistically positive advantage"），benign 无差异且 bursty 命中率反而更低。**8.64pp 只在一种攻击形态下成立，而那恰好是最容易被入口速率限制拦住的一种。**
- **$\lambda_{new}/\rho = 0.54 \Rightarrow$ 增益 0.00pp 是机制的硬边界。** 论文把它写成 "by design"，语义上自洽，但它意味着**保护强度完全由运营者对 $\rho_c$ 的取值决定，机制本身不提供任何与攻击者行为无关的下界**。$\rho_c$ 定高了没保护，定低了误伤良性突发。这个参数的运营含义（怎么定、错定的代价是什么）全文没有讨论。
- **Fig. 3(b) 同时是债务门禁失效的证明。** 等待 $\tau_g$ 之后两个变体完全收敛。攻击者对抗 I2 的成本只是**延迟**，而延迟对缓存污染攻击接近零成本。作者把这写成正面性质（避免永久惩罚），权衡取向合理，但**没有给出任何针对"有耐心的攻击者"的讨论**。
- **蹲位攻击（慢速灌入 + 延迟重放 ⇒ 拿到 $\textsc{Reuse}$ 等级永久占位）是三条 trace 都没覆盖的、且机制原理上无法防的模式**（§6.2(1)）。根因是只计量创建速率、不计量占用时长。
- **唯一的公平性 baseline 是自建的稻草人。** 引用了 VTC / FairRide / NyxCache 三篇，实现了零篇；StaticQuota 被配置成分区（255）小于工作集（384）而必然失败，作者自己承认了两次却没换配置重测。**这是本文最大的方法论疑点：论文声称填补了多租户公平这条线的空白，却从未与该条线上的任何真实系统比较过。**
- **实验规模是本仓库同批次最小的**：单张 L4、1.5B 模型、**8,192 token 的 KV 池**、受害者热集占 75%。这个配置人为制造了"任何准入必然触发淘汰"的饱和态。论文从未论证**在真实容量（数万 block）下缓存污染仍是一个可达成的攻击**——威胁模型与实验之间存在这个缺口。
- **单实例作用域，无分布式故事。** N 副本下各自记账、路由器不感知 debt ⇒ 摊薄流量即可全面免罚。论文一个字都没提多副本。
- **增益随负载单调下降（40%: 20.06 → 80%: 8.47 pp），未被解释。** 意味着在最需要保护的高压场景相对价值最小。
- **无形式化公平保证。** 唯一的 Proposition 1 是令牌桶自恢复的三行代数，保证的是"惩罚会过期"，不是"受害者被保护到什么程度"。
- **无代码、无 artifact、0 引用、无 venue、未经同行评审**；$\epsilon$ 这个直接决定门禁行为的参数**全文只以符号出现，从未给出数值**。

### 一句话定位

**PrefixShield 提出了一个真实且此前未被正面命名的问题（持久 KV 状态使得"制造缓存压力的人"与"承担淘汰代价的人"脱钩），并给出了一个机制上干净、语义上自洽的答案（按新块创建计量的令牌桶透支 → 淘汰受害者组选择 + 保留等级晋升门禁）；但它的实验配置（8,192 token 缓存池、75% 热集、被配置成必然失败的唯一公平 baseline、缺席的 VTC/FairRide 对照）决定了它的所有数字都只能当作"机制方向可行"的存在性证据，而不能当作可迁移的效应量；机制本身还有两个原理性缺口——额度内污染完全免罚、门禁在 $\tau_g$ 后自动失效——使它更像一个"给缓存创建加了一层可审计的会计"的工程贡献，而非一个防护机制。** 对本仓库的价值在于：它是**唯一一篇把干预点放在"淘汰的受害者组选择"上的工作**，也是唯一一篇把多租户责任做成跨请求持久信号的工作；它与 Preble 的 VTC 式服务公平**在计价单位上互补而非竞争**，两者叠加才能同时封住服务侧与状态创建侧的漏洞——**而这个叠加至今无人做过**。

---

## 附：与本仓库其他解读的交叉引用

- [`CacheRoute-论文解读.md`](CacheRoute-论文解读.md) —— **两种「准入」的核心对照**（详见 §7.1）。Meta CacheRoute 在**入口路由层**做**性能导向**的 per-key 准入（top-rate 选前缀、副本数预算 $\sum_b k_b \le C$、60×H100 上 176±11 QPS）；PrefixShield 在**引擎缓存层**做**责任导向**的准入（从不拒绝，只重定向淘汰压力）。两者正交可叠加，但存在一个具体的正反馈风险：CacheRoute 的规划器假设自己控制驻留，PrefixShield 会因会计原因淘汰它放置的块，规划器观察到命中率下降后加大配额 ⇒ $n_r$ 上升 ⇒ 债务加深 ⇒ 更多淘汰。**更值得注意的是反向互补：CacheRoute 的 top-rate 排序在对抗性负载下会主动给重放攻击者分配副本配额，而 PrefixShield 的 I2 不变式恰好是修这个洞的正确机制。**
- [`Preble-论文解读.md`](Preble-论文解读.md) —— **公平视角上最接近的工作，也是 PrefixShield 应该实现而没有实现的 baseline**（详见 §7.2）。Preble 用 cache 命中率调整 VTC 式公平代价，干预的是**请求调度优先级**；PrefixShield 干预的是**淘汰受害者选择**，两者计价单位不同（服务量 vs 创建块数）、漏洞位置不同（服务份额内污染 vs 创建额度内污染），**是叠加关系而非替代关系**。关键警示：Preble 对 one-touch 攻击天然有部分免疫（攻击者全 miss ⇒ 代价高 ⇒ 被压制），所以"比 S3-FIFO 高 8.64pp"**不能推广成"比现有公平机制强"**。
- [`PRISM-论文解读.md`](PRISM-论文解读.md) —— **同为引擎层、改的却是淘汰逻辑的不同部分**（详见 §7.4）。PRISM 改**淘汰键**（下一批 batch 的 segment 需求分数 ⇒ 注入"未来的调度意图"），PrefixShield 改**淘汰的受害者组**（注入"过去的创建责任"）。二者可以组合进同一个两阶段淘汰：Step 1 用债务选组，Step 2 用 PRISM 的需求分数替代静态三段等级。另一条直接相关的教训：PRISM 的冷车道消融证明"缓存命中率是错误的优化目标"（命中率更高但 P99 TTFT 恶化 16.5%），而 PrefixShield 的 bursty 行已经出现同样分叉（38.10 vs 38.72 命中率更低，168.1 vs 185.5 ms 尾延迟更好）——**PrefixShield 选错了 headline 指标，对自己不利**。
- [`Mooncake-论文解读.md`](Mooncake-论文解读.md) —— 准入谱系的第三个点。Mooncake 是**拒绝式**准入（基于 decoding slot 预测的早期拒绝，阈值手工调），保护的是服务容量；CacheRoute 是**配额式**（保护整体效率）；PrefixShield 是**责任式**（从不拒绝、只重定向压力，保护租户间因果公平）。三者构成一个可用于综述的划分。另外 PrefixShield 引用了 Mooncake [6] 但只把它归入 "KV reuse and placement" 一类，没有讨论其准入控制——这是一处引用不充分。
- [`DualMap-论文解读.md`](DualMap-论文解读.md) —— 提供负载侧的现实性佐证：48% 对话请求 / 76% Agent 请求共享 ≥50% 前缀。这个数字对 PrefixShield 有一层**反向含义**：既然真实负载中跨请求前缀共享如此普遍，那么 PrefixShield 原型采用的**按租户 namespace 隔离前缀查找**（§2.4）放弃的收益就相当可观——尤其是共享系统提示词模板这一块。论文对这个代价一字未量化。
- [`LLM服务负载均衡与调度-文献综述.md`](LLM服务负载均衡与调度-文献综述.md) —— 定位建议：应归入**多租户公平与隔离**分支（而非安全/侧信道分支——原型的 namespace 隔离使其不涉及任何跨租户内容共享，因此与 prompt-cache 时序侧信道 / 成员推断完全无关），并在层次上标注为**引擎层缓存会计**（单实例，非路由层，无分布式内容）。系统名 `PrefixShield` 核实无误，作者为 Zhiyu Wang 与 Rajkumar Buyya（墨尔本大学 qCLOUDS Lab），但**发表场所应标注为预印本**（arXiv:2608.01657v1，2026-08-03，IEEE 双栏排版但无 venue 标注）。
