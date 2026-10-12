# Towards Efficient LLM Serving: A Survey on System-Aware KV Cache Optimization —— 论文解读

> 论文：*Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization*
> 作者：Jiantong Jiang、Peiyu Yang†（墨尔本大学）、Rui Zhang（华中科技大学）、Feng Liu（墨尔本大学）
> 版本：arXiv:2607.08057v1（2026-07-09），cs.LG，正文 9 页 + 附录，共 27 页
> 配套仓库：[jjiantong/Awesome-KV-Cache-Optimization](https://github.com/jjiantong/Awesome-KV-Cache-Optimization)
> 本地 PDF：[`Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization.pdf`](../papers/Towards%20Efficient%20Large%20Language%20Model%20Serving:%20A%20Survey%20on%20System-Aware%20KV%20Cache%20Optimization.pdf)

---

## 1. 一句话概括

这是一篇 KV Cache 优化综述，但它的真正贡献不在"又列了 150 篇论文"，而在于**换了一套坐标系**：不按生命周期（训练/部署/后处理）也不按抽象层（token/model/system）来分类，而是按**系统行为**把所有工作投影到「时间（temporal）—空间（spatial）—结构（structural）」三个正交维度上。作者把这个范围命名为 **sKis**（system-aware KV infrastructure for serving LLMs），然后在这个坐标系上做了两件前人没做的事：一张**行为 × 目标矩阵**（哪类行为真正改善哪个指标），一张**行为 × 行为协同亲和网络**（哪些技术在文献中总是一起出现）。由此定量地暴露出几个空白区域——最刺眼的一个是：**KV 压缩最热门，却几乎不与任何其他行为协同设计**。

---

## 2. 为什么还需要一篇 KV Cache 综述

作者在 Table 1 / 附录 C 里把自己和 10 篇前作逐一对齐，筛选条件是四个「✓」：

| 综述 | KV 为核心 | 仅服务期 | 不需重训 | 系统指标 | 组织原则 |
| --- | :---: | :---: | :---: | :---: | --- |
| Miao et al. 2023 | | ✓ | | ✓ | 算法层 / 系统层 |
| Yuan et al. 2024 | | ✓ | | ✓ | 优化层（参数/算法/系统/硬件） |
| Li et al. 2024a | | ✓ | ✓ | ✓ | 系统组件 |
| Zhou et al. 2024b | | ✓ | | ✓ | 优化层（数据/模型/系统） |
| Zhen et al. 2025 | | ✓ | ✓ | ✓ | 服务规模（实例级/集群级） |
| Shi et al. 2024 | ✓ | | ✓ | | 生命周期（训练/部署/后训练） |
| Li et al. 2024b | ✓ | | ✓ | ✓ | 优化层（token/model/system） |
| Liu et al. 2025c | ✓ | ✓ | ✓ | | 压缩类型 |
| **本文（sKis）** | **✓** | **✓** | **✓** | **✓** | **系统行为（时间/空间/结构）** |

两类前作各有各的缺口：**通用推理综述**里 KV cache 只是一小节；**KV 专项综述**又按生命周期或抽象层切分，导致"服务期到底发生了什么行为"这件事从未被单独审视。

### sKis 的边界定义

一个方法属于 sKis，当且仅当同时满足三条：

1. 运行在**服务期（推理阶段）**；
2. 以 **KV cache 为首要优化对象**；
3. 目标是改善**系统指标**，且**不重训基座权重、不改 Transformer 结构**。

明确排除在外的方向：重新设计注意力/KV 机制（MLA、NSA 等）、改动骨干网络结构（SliceGPT 等）、长上下文记忆增强、纯 kernel fusion、训练期 KV 压缩。理由很实在——这些做法要么改模型要么要适配，而 sKis 面向的是社区最常见的部署场景：**拿现成模型直接上线**。

这个边界值得注意：**MLA（DeepSeek-V2）不在范围内**。所以这篇综述覆盖的是"给定模型，系统还能怎么榨"，而不是"模型层面怎么少产生 KV"。

---

## 3. 三维行为分类法

这是全文的骨架。作者的类比是操作系统：一个现代 OS 有调度、内存、I/O 三大组件，LLM 服务系统同样如此。

```
sKis
├── 时间行为：KV 执行与调度（§3）—— 何时访问/计算/调度 KV
│   ├── KVS  KV-centric scheduling      KV 感知调度
│   ├── OVLP Pipelining & overlapping   流水与重叠
│   └── HAE  Hardware-aware execution   硬件感知执行
│         ├── 分离式推理（PD 分离）
│         └── 计算卸载（CPU / PIM / CSD）
├── 空间行为：KV 放置与迁移（§4）—— KV 放在哪、怎么搬
│   ├── MHO  Memory hierarchy orchestration  内存层级编排
│   │     ├── 跨设备层级（HBM ↔ DRAM ↔ SSD ↔ CXL）
│   │     └── GPU 内层级（L1/L2 ↔ HBM）
│   └── CDO  Compute device orchestration    计算设备间编排
└── 结构行为：KV 表示与保留（§5）—— KV 长什么样、留多少
    ├── KVCC KV cache compression       压缩
    │     ├── 量化（含 VQ）
    │     ├── 低秩近似
    │     └── 结构化压缩（剪枝 / 合并）
    └── KVRM KV retention management    保留管理
          ├── 分配与复用（PagedAttention / RadixAttention）
          └── 驱逐（H2O / StreamingLLM / SnapKV ...）
```

作者坦承一篇论文常常横跨多个类别（例如 Mooncake 同时是 KVS + HAE），因此只标注 1–2 个**主类别**，次要关联统一收进附录 D 的 Tab. 9。这个「主/次」标注后面成了计算协同亲和网络的数据基础。

---

## 4. 时间维度：执行与调度

### 4.1 KV 感知调度（KVS）—— 详解

> 本小节把论文中散落在正文 §3、§3.1、Fig. 4、Tab. 7、附录 D/E.1/F 里所有与 KVS 有关的内容集中整理。凡标注「论文原文」的为综述明确陈述，标注「推导」的为我基于论文公开数据的反推。

#### 4.1.1 定义与在分类法中的位置

KVS 属于**时间维度（temporal behavior）**，与 OVLP、HAE 并列。论文对时间维度的界定是：*"KV cache 使用的时间行为，包括缓存条目在运行时如何被调度和高效执行"*。

对 KVS 本身的界定只有一句，但这句是全节的分界线：

> 调度本身是长期研究的系统问题，KVS 方法的区别在于**显式地把 KV 特征整合进运行时决策**（explicitly integrate KV characteristics into runtime decisions）。

换句话说，判定一个调度器是否算 KVS，不看它调度什么，而看**它的决策变量里有没有 KV**。FCFS、SJF 不算；用队列长度做负载均衡不算；用「这个请求会命中多少前缀 KV」或「这个请求会吃掉多少 KV 内存」做决策，才算。

#### 4.1.2 KVS 完整方法清单（Fig. 4 主类别，15 篇）

论文在 Fig. 4 的 KVS 叶子节点下列出了 15 个方法。补上附录 D Tab. 9 的发表venue：

| # | 方法 | Venue | 年份 | 在 §3.1 中的归属粒度 |
| --- | --- | --- | --- | --- |
| 1 | TetriInfer | arXiv | 2024 | 请求级 · KV 用量感知 |
| 2 | Preble | ICLR | 2024 | 请求级 · 复用感知 |
| 3 | MuxServe | ICML | 2024 | 请求级 · KV 用量感知 |
| 4 | Quest | ICML | 2024 | Token 级 |
| 5 | SparQ Attention | ICML | 2024 | Token 级 |
| 6 | LayerKV | arXiv | 2024 | 请求级 · KV 用量感知 |
| 7 | LAMPS | arXiv | 2024 | 请求级 · KV 用量感知 |
| 8 | LoongServe | SOSP | 2024 | 请求级 · KV 用量感知 |
| 9 | RadixAttention (SGLang) | NeurIPS | 2024 | 请求级 · 复用感知 |
| 10 | Loki | NeurIPS | 2024 | Token 级 |
| 11 | Mooncake | FAST | 2024 | 请求级 · 复用感知 |
| 12 | FlashInfer | MLSys | 2025 | **Kernel 级** |
| 13 | RocketKV | ICML | 2025 | Token 级 |
| 14 | RefreshKV | ACL | 2025 | Token 级 |
| 15 | TokenSelect | EMNLP | 2025 | Token 级 |

发表场所横跨 ICLR/ICML/NeurIPS（ML）、ACL/EMNLP（NLP）、SOSP/FAST/MLSys（系统），印证了附录 D 的「venue 多样性」论述——KV 优化本质上是跨社区议题。

#### 4.1.3 请求级（request level）

论文把请求级又分成两条支线。

**支线 A：KV 用量感知（KV usage-aware）** —— 目标是平衡资源负载、减少争用。

论文点名 5 篇：TetriInfer、MuxServe、LayerKV、LAMPS、LoongServe。只对 TetriInfer 展开了一句：

> TetriInfer **用预测出的 KV 用量给请求排优先级**，以缓解 prefill-decode 干扰（mitigate prefill-decode interference）。

这条支线的共性是：调度器需要一个**预测器**来估计请求未来的 KV 占用（本质上要预测输出长度），再据此决定排队顺序、批次组合或实例分配。

**支线 B：复用感知（reuse-aware）** —— 目标是最大化 KV 命中率。

论文进一步区分了两种用法，措辞上的差别值得注意：

- **RadixAttention**：*"优先调度高复用请求以最大化 KV 缓存命中率"*（prioritizing high-reuse requests to maximize KV cache hit rate）——命中率本身就是优化目标。
- **Preble、Mooncake**：*"把 KV 复用潜力作为决策中的一个关键信号"*（using KV reuse potential as a key signal in decisions）——复用是**信号之一**，要和负载等其他信号一起权衡。

这个区分很精准：前者是单目标最大化，后者是多目标权衡。本仓库关注的前缀感知路由这条线（Preble → Mooncake → DualMap/CacheRoute）全都属于后者，其核心矛盾正是「复用信号 vs 负载信号」如何加权。

#### 4.1.4 Token 级（token level）

> 在更细的粒度上，token 级方法**基于估计贡献来决定哪些 KV 条目参与注意力计算**。

论文点名 6 篇：Quest、SparQ Attention、Loki、RocketKV、RefreshKV、TokenSelect。唯一展开的是 RefreshKV：

> ……例如通过**周期性刷新，在全上下文注意力与子集注意力之间交替**（periodic refresh that alternates full-context and subset attention）。

这里有一个容易混淆的边界，值得单独说明：**token 级 KVS 与 KVRM 的驱逐（§5.2.2）看起来都在"选 token"，但不是一回事**。

| | Token 级 KVS（§3.1） | KV 驱逐（§5.2.2） |
| --- | --- | --- |
| 行为维度 | 时间 | 结构 |
| 做什么 | 决定哪些 KV **参与本次注意力计算** | 决定哪些 KV **继续留在内存里** |
| KV 是否还在 | 在，只是本步不算 | 不在了（或降级到别处） |
| 可逆性 | 完全可逆，下一步可以再选回来 | 通常不可逆（除非配合召回/合并） |
| 省什么 | 计算与带宽 | 内存 |

RefreshKV 的「周期性刷回全上下文」恰恰只有在 KV 没被真正丢掉的前提下才可能——这就是它被归入 KVS 而非驱逐的原因。

#### 4.1.5 Kernel 级（kernel level）

只有 FlashInfer 一个代表：

> FlashInfer 这类方法**根据 query 与 KV 的长度，把注意力负载调度到不同的 CUDA thread block 上**。

这是 KVS 里粒度最细的一层——调度对象不再是请求或 token，而是 GPU 线程块。它同时也是 KVS 与 KVRM 交叉的例子：FlashInfer 在 §5.2.1 里还作为「标准化 KV 布局给 kernel 用」的代表（块稀疏、可组合格式）。

#### 4.1.6 附录 E.1 的完整 takeaway

论文正文对 KVS 只给了三段话，真正的经验结论都在附录 E.1。E.1 开篇先给了一个统合判断：

> KVS 与 OVLP 都直接针对运行时停顿。**KVS 把有限资源优先分配给最可复用、最延迟敏感的工作；OVLP 本身也是一种调度**——它对齐计算与数据传输以填充流水气泡。

四条 takeaway（前两条直接关于 KVS）：

1. **KVS 是一个多目标优化问题。** 现代调度器**优先考虑 KV 随时间的用量，而不是 FLOPS**（prioritize KV usage over time rather than FLOPS），且 **KV 复用驱动的调度已成为默认范式**（the default paradigm）。
2. **KVS 靠预测增强。** *"轻量预测器 + 稳健策略的组合，优于传统 FCFS 或 SJF 方案"*——点名 TetriInfer、Mooncake、LAMPS 三篇。
3.（关于 OVLP）关键是在真正的瓶颈处做非对称流水。
4.（关于 OVLP）宁可重算不传输；在集合通信期间预取 KV 到 L2。

第 1 条里「**优先考虑 KV 用量而非 FLOPS**」是整篇综述关于调度最凝练的一句话——它说明 LLM 服务调度的核心资源已经从"算力"迁移到"KV 内存与其复用结构"。第 2 条则明确了方法论：**不要指望无预测的启发式**，但也不要指望重量级预测器，轻量预测器配稳健策略是当前的最优实践。

#### 4.1.7 KVS 在「行为 × 目标」矩阵中的位置（Tab. 7）

KVS 那一行的标注：

| 目标 | 平均延迟 | 尾延迟 | 吞吐 | GPU 内存 | 互连 I/O | 能耗 | 质量损失↓ |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KVS** | **●⋆** | **●** | **●** | ○ | ○ | ○ | ○ |

读法：**●** = 直接影响，**○** = 间接影响，**⋆** = ≥70% 的相关论文报告了该项收益。

三个信息：

1. KVS 是**唯一一个在「平均延迟」上拿到 ●⋆ 而在「GPU 内存」上只有 ○ 的行为**（KVS/OVLP 两行同此，但 OVLP 的平均延迟与吞吐双 ⋆）。这说明 **KVS 是纯粹的延迟/吞吐手段，不直接省内存**——它省内存是通过提高复用间接实现的。
2. KVS 在「尾延迟」上标 ● 而无 ⋆，意味着**虽然理论上直接相关，但报告尾延迟收益的论文不足 70%**。这正是 O2 所说的"尾延迟上报非常稀少"，也是 C1 挑战的直接来源。
3. KVS 在「质量损失」上标 ○——**调度类方法不直接损害质量**，但论文在 O5 中提醒：时间类方法有"请求处理不一致"（inconsistent request handling）的风险。

#### 4.1.8 KVS 在协同亲和网络中的位置（Tab. 11 / Tab. 12）

KVS 与其余六个行为的原始共现权重 $C$ 和 Tanimoto 归一化亲和度 $S$：

| 配对 | $C_{ij}$（原始共现） | $S_{ij}$（归一化亲和） | 是否连边（$\theta=0.14$） |
| --- | :---: | :---: | :---: |
| KVS – **KVRM** | **6.5**（KVS 行最大） | 0.14 | ✅ 恰好达阈值 |
| KVS – **HAE** | 4.75 | **0.16**（KVS 行最强） | ✅ |
| KVS – MHO | 4.0 | 0.11 | ❌ |
| KVS – OVLP | 2.5 | 0.09 | ❌ |
| KVS – CDO | 1.75 | 0.07 | ❌ |
| KVS – **KVCC** | **0.00** | **0.00** | ❌ |

几个结论：

**(a) KVS–KVCC = 0 是一个严格结论。** 权重规则是主类别记 1、次类别记 0.5，$C_{ij}=\sum_p \omega(\ell_{p,i})\,\omega(\ell_{p,j})$。$C=0$ 意味着**在全部被调研的论文中，没有任何一篇同时被标注为 KVS 与 KVCC——哪怕是次类别**。整张 Tab. 11 里只有这一个零。这是 O7（"KVCC 仍然孤立"）最硬的证据：最热门的 KV 压缩方向，与调度的交集是空集。

**(b) KVS 的最强搭档是 HAE 和 KVRM，但都很弱。** $S_{\text{KVS,HAE}}=0.16$ 和 $S_{\text{KVS,KVRM}}=0.14$ 刚过阈值，对比 HAE–CDO 的 0.53、OVLP–HAE 的 0.42，KVS 在整张网络中是**边缘节点**。Fig. 6 中 KVS 只连出两条细边。

- KVS–HAE（0.16）：Mooncake、TetriInfer、MuxServe 这类既做 PD 分离又做 KV 调度的系统贡献了主要权重。
- KVS–KVRM（0.14，原始共现 6.5 为 KVS 行最高）：RadixAttention 是典型——基数树既是复用感知调度的依据（KVS），又是 KV 分配复用的机制（KVRM）。**调度与保留管理天然共享同一个数据结构**，这解释了为何原始共现最高。

**(c) KVS 的研究密度偏低。**〔推导〕由 $S_{ij}=C_{ij}/(Q_i+Q_j-C_{ij})$ 可得 $Q_i+Q_j=C_{ij}/S_{ij}+C_{ij}$，21 个配对构成超定方程组，加权最小二乘解出：

| 行为 | $Q_i$（反推） | Fig. 4 主类别方法数 | 隐含的次类别标注数 $n$ |
| --- | :---: | :---: | :---: |
| KVRM | 37.8 | 33 | ~19 |
| KVCC | 37.4 | 37 | **~1.6** |
| MHO | 23.9 | 23 | ~4 |
| HAE | 18.3 | 15 | ~13 |
| **KVS** | **16.1** | **15** | **~4.4** |
| OVLP | 13.0 | 8 | **~20** |
| CDO | 10.5 | 9 | ~6 |

（$Q_i=\sum_p w_{p,i}^2 = \#\text{主} + 0.25\,\#\text{次}$，故 $n=(Q_i-\#\text{主})/0.25$。KVS 相关方程的拟合残差均 < 1.1，可信度较高。）

这张表本身就是几个关键结论的定量佐证：

- **$Q_{\text{KVS}}=16.1$ vs Fig. 4 的 15 个主类别方法**——两者高度吻合，反向验证了推导的正确性，同时说明**只有约 4 篇论文把 KVS 当作次要贡献**。KVS 既不是热门主战场，也很少被别的工作顺带做掉。
- **$Q_{\text{KVCC}}=37.4$ ≈ 主类别数 37**——KVCC 几乎从不作为次类别出现，定量印证 O7 的"孤立"。
- **$Q_{\text{OVLP}}=13.0$ vs 8 个主类别**——OVLP 有约 20 次次类别标注，是全表里"被顺带做掉"最多的行为，印证 E.3 所说"多数 MHO/CDO 方案把 OVLP 作为次要类别"。

#### 4.1.9 KVS 的局限（§3.3 后的 Takeaways & Limitations 框）

论文对整个时间维度给出的局限，第一条直指 KVS：

> KVS 与 OVLP 直接针对 KV 复用与停顿隐藏，轻量成本模型或预测器常能增强它们。**然而，它们通常只在受控负载上评测，对突发流量或多租户场景下的鲁棒性分析非常有限**（typically evaluated on controlled workloads, with limited analysis of robustness under bursty traffic or multi-tenant settings）。

这一条几乎是对所有前缀感知调度工作的通用批评：论文里跑的是固定 trace、固定并发，而生产环境是突发的、多租户的。

#### 4.1.10 KVS 关联的开放挑战：C1

C1（SLO 驱动的尾延迟控制）由 O2 导出，而 O2 正是关于时间行为的观察。C1 的核心论证：

> SLO 对 LLM 服务至关重要，尾延迟主导用户体验，**但多数系统压根不报告尾延迟指标**。在长上下文与突发流量下，KV 的生成、迁移、压缩都可能触发 SLO 违约，而这些违约**往往源于跨行为干扰，而非单一操作**（arise from cross-behavior interference rather than a single operation）。真正的挑战是**把 SLO 违约归因到具体的 KV 行为和路径**，这促使未来的 sKis 研究去建立标准化的抢占与降级语义（standardized preemption and degradation semantics），让尾部结果变得可控。

对做调度的人，这段话给出三个可落地的方向：

1. **归因**：需要能把一次 P99 违约拆解到"是排队等 KV 传输、还是 prefill 抢占、还是压缩解压开销"。当前缺的是可观测性基础设施——这与附录 G.3.2 建议上报的「KV 相关停顿占比」指标直接对应。
2. **标准化抢占语义**：调度器抢占一个正在 decode 的请求时，它的 KV 该丢、该换出、还是该压缩后保留？目前每个系统各有一套，没有公共语义。
3. **降级语义**：SLO 快要违约时，允许调度器主动降低服务质量（更激进的驱逐/更低的位宽）以保住延迟——这正是 C5 里「把 bitwidth、rank、sparsity 暴露给运行时」那条建议在调度侧的镜像。

#### 4.1.11 补充观察：综述在 KVS 上的盲区

〔以下为我的判断，非论文内容〕

论文的 KVS 请求级支线只收录了 8 篇（TetriInfer、Preble、MuxServe、LayerKV、LAMPS、LoongServe、RadixAttention、Mooncake），且**全部是 2024 年的工作**。2025–2026 年围绕「跨实例前缀感知路由」形成的一整条产品化路线完全未被覆盖：

- 开源网关/路由器：llm-d（Precise Prefix Cache Routing）、AIBrix Gateway Plugins、SGLang Model Gateway、NVIDIA Dynamo KV Router、Envoy AI Gateway EPP、GKE Inference Gateway、KServe、KubeAI（CHWBL）、Ray Serve Prefix-Aware Routing、vLLM Production Stack
- 学术工作：DualMap、CacheRoute、SkyWalker、GORGO、TOPAS、PRISM、BanaServe、Preserving Admission Responsibility 等

这个盲区的成因大概有二：一是这些工作多以工程博客/开源实现形态存在，未进入传统会议检索范围；二是它们的决策单元是**跨实例的请求路由**，介于"单实例调度器"和"集群编排"之间，在 sKis 分类法里没有专门的槽位——只能笼统塞进 KVS 请求级。

这恰恰暴露了分类法的一个粒度缺口：**KVS 的"请求级"实际混合了两个不同层次**——

| | 实例内调度（intra-instance） | 跨实例路由（inter-instance） |
| --- | --- | --- |
| 决策 | 队列里先跑哪个请求、怎么组批 | 请求发给哪个实例 |
| 代表 | TetriInfer、LoongServe、LAMPS | Preble、Mooncake、DualMap、llm-d |
| 关键约束 | KV 内存容量、PD 干扰 | 前缀命中率 vs 实例负载 |
| 理论工具 | 排队论、抢占调度 | 一致性哈希、Power of Two Choices、有界负载 |
| 失败模式 | OOM、抢占抖动 | 热点实例、缓存颠簸 |

两者的理论工具几乎不重叠（本仓库收录的 Consistent Hashing、Consistent Hashing with Bounded Loads、Power of Two Random Choices、Graham 1969 这几篇经典，全都只服务于右列）。综述把它们并为一类，导致右列这个正在快速工程化的方向在分类法里"不可见"——而这，可能是当前 sKis 版图里最值得补的一块。

### 4.2 流水与重叠（OVLP）

这是全文里最实用的一张表（Tab. 2），按重叠模式归类：

| 模式 | 方法 | 重叠的两个操作（含传输路径） | 粒度 |
| --- | --- | --- | --- |
| Comp–Comp | FastDecode | CPU R-part 计算 ↔ GPU S-part 计算 | token 级 |
| Comp–Comp | Neo | CPU 注意力计算 ↔ GPU 线性算子 | sub-batch 级 |
| Comp–I/O | CComp | CPU MHSA 计算 ↔ FFN 数据传输（CPU→GPU） | 切分点 |
| Comp–I/O | CachedAttention | GPU 计算 ↔ KV 加载/落盘（CPU↔GPU） | 层级 |
| Comp–I/O | AsyncKV | GPU 注意力计算 ↔ GPU KV 预取（HBM→L2） | KV block 级 |
| Comp–I/O | KVPR | GPU KV 重算 ↔ KV 传输（CPU↔GPU） | 切分点 |
| Comp–I/O | CXL-SpecKV | GPU 计算 ↔ KV 预取（CXL 内存→L2） | KV page 级 |
| I/O–Comm | PRESERVE | GPU KV 预取（HBM→L2） ↔ GPU 集合通信 | 算子级 |

> **要点**：OVLP 的关键是**在真正的瓶颈处做非对称流水**——计算密集的 prefill 留在 GPU，内存密集的 decode 注意力与 KV 去和 I/O 或集合通信重叠。两个反直觉但有效的结论：**宁可重算也不传输**（KVPR），以及**在集合通信期间把 KV 预取进 L2**（AsyncKV、PRESERVE）——带宽是瓶颈时收益尤其明显。

### 4.3 硬件感知执行（HAE）

**分离式推理**：Infinite-LLM 在算子级把注意力拆到分布式实例；Splitwise / DistServe / TetriInfer / DéjàVu 做 PD 分离，把计算密集的 prefill 与访存密集的 decode 放进不同计算池；Mooncake 进一步把 PD 分离与 KV 中心调度器 + 分布式缓存池耦合；MuxServe 走另一条路——用 SM 分区在**同一张 GPU 内**共置 PD 任务。

**计算卸载**：CPU 卸载遵循 **compute-near-cache** 原则（FastDecode 用成本感知硬件选择，Neo 用负载感知调度器，两者都同时卸载注意力和 KV）；此外还有卸载到计算存储盘 CSD（InstInfer）和存内计算 PIM（PAPI、AttAcc）的路线。

> **局限**：HAE 依赖大量底层原语，**可移植性差**——换一套硬件拓扑，kernel、tiling、内存布局往往要重做。

---

## 5. 空间维度：放置与迁移

### 5.1 内存层级编排（MHO）

**跨设备层级**分两条思路：

- **重要性感知**（主流）：设计打分策略，GPU 上只保留关键 KV。典型手法是把**完整 KV 全部卸载到 CPU，GPU 只留一个轻量代理信号**，用代理信号估计重要性来指导下一步预取——ArkVale、OmniKV、ClusterKV、PQCache、SpeCache 都是这个模式。
- **系统成本视角**：FlexGen 用成本模型在 GPU/CPU/磁盘之间做带宽和延迟约束下的吞吐最大化放置。近期工作把层级扩展到 **CXL 共享/池化内存**（CXL-SpecKV、Beluga、TraCT）。

运行时的卸载/回载决策则由排队状态、内存压力、计算与 I/O 成本、未来复用信号等系统级信号驱动（FastServe、CachedAttention、LayerKV、FastSwitch、RAGCache、LMCache、KVFlow）。

**GPU 内层级**是一条较冷门但有意思的线：在片上 L1/L2 与片外 HBM 之间搬 KV。AsyncKV 异步预取 KV block 到 L2 让后续注意力尽量命中 L2；PRESERVE 通过图级优化**有选择地**插入预取算子，避免污染缓存。

### 5.2 计算设备间编排（CDO）

- **集群内编排**，通常与 PD 分离绑定。DistServe 按节点亲和度高/低把 prefill 和 decode 放到不同 GPU 集群，用 **pull 模式**让 decode GPU 按需从 prefill GPU 拉取 KV。TraCT 则把 CXL 共享内存当作 PD worker 间的 KV 传输介质。
- **跨网络传输**：CacheGen 把 KV 张量编码成比特流，按运行时带宽自适应流式传输，以降低网络延迟。
- **异构加速器**：把注意力卸载到 PIM、CSD。

> **要点与缺口**（附录 E.3）：MHO / CDO 直接针对互连瓶颈，且**绝大多数系统都靠"传输与计算重叠"来藏延迟**——这是它们生效的中心因素。但两个明显缺口：（1）带宽处理普遍**没有对并发 KV 传输的争用做显式建模**，导致尾部行为难以分析；（2）**预取-驱逐协同设计极其罕见**，缺一个同时考虑「预取截止时间」与「驱逐风险」的统一目标函数。

---

## 6. 结构维度：表示与保留

### 6.1 量化（Tab. 3，23 个方法）

两个反复出现的模式：

- **非对称 KV 量化**：Key 和 Value 的离群点分布与量化敏感度不同，常见做法是 **Key 按 channel 量化、Value 按 token 量化**（KIVI、GEAR、KVQuant、OTT、ChanMix 都是这个组合）。
- **离群点是命门**：多数方法给离群点更高位宽或专门处理（稀疏矩阵存 FP16、平滑缩放、channel-wise 归一化、bit 重分配）。位宽越低，离群点和异质敏感度越难保住，压缩-质量权衡在超低比特区间尤其吃紧。

近期趋势是**向量量化（VQ）**：CQ 耦合 channel 学习质心做到 1-bit；VQ-LLM 与 CommVQ 分别用融合 VQ kernel 和 RoPE-可交换码本降低开销；NSNQuant 改进校准鲁棒性。平均位宽已能压到 **1.2–2 bit** 且质量损失可控。

> **关键提醒**：**更低位宽不等于端到端收益**。实际收益还取决于 (de)quantization 的运行时开销、额外的 kernel 边界、以及这些开销能否被融合或重叠。只有在负载本身是 memory-bound / I/O-bound、且量化被有效集成进系统栈时，内存节省才会转化为延迟或吞吐改善。

### 6.2 低秩近似（Tab. 4）

按作用对象分三类：(i) 已缓存的 KV 张量（xKV 做 layer-group SVD，ShadowKV 只对 K 做，ReCalKV 对 K 用 head-group、对 V 用 layer-wise）；(ii) KV 投影权重 $W_K, W_V$（Palu 分解并搜索秩分配，LoRC 用规则）；(iii) QKV 注意力子空间（EigenAttention）。

权衡很清楚：**作用于 KV 张量最即插即用，但增加投影开销；作用于权重和 QKV 则加剧 kernel 耦合与工程开销**。需要学习额外参数的低秩方法因为要训练，被排除在 sKis 范围外。

### 6.3 结构化压缩（Tab. 5）

- **合并**：KVMerger（token 级，注意力+相似度信号）、CaM（token，注意力）、KVSharer / MiniCache（layer 级，相似度）、D2O（token，相似度）
- **剪枝**：CHAI（head 级，相似度）、ThinK（channel 级，query norm）、ClusterAttn（token 级，注意力）

### 6.4 保留管理（KVRM）

**分配与复用**分三条线：

1. **虚拟化分配**——PagedAttention 用固定大小页 + 逻辑到物理映射消除碎片（vLLM、vTensor、vAttention、FastSwitch）；
2. **结构化索引做 prompt 共享**——RadixAttention 的基数树是代表（还有 PromptCache、ChunkAttention）；
3. **标准化 KV 布局给 kernel 用**——FlashInfer 提出块稀疏且可组合的格式。

语义引导的方向则更激进：LazyLLM 只为关键 token 计算 KV；MemServe 用弹性 MemPool 把复用扩展到分离式服务。

### 6.5 驱逐（Tab. 6，23 个方法按时间排序）

三个洞察：

1. **静态 vs 动态**：静态方法在 prefill 期间/之后驱逐一次，decode 阶段保留集固定（FastGen、PyramidKV、SnapKV、DynamicKV）；动态方法在 decode 期间在线更新以追踪重要性漂移（其余多数）。
2. **保留模式趋同**：几乎都保留一个近期窗口（R）+ 一小撮 **attention sink**（S，即初始 token），再用轻量信号选额外 token。信号从累积注意力（H2O）演进到注意力均值方差（RoCo、CAKE）、Key 的 $L_2$ 范数（L2KV）、Value 的 $L_1$ 范数（VATP）、分隔符 token（SepLLM）等。
3. **预算策略从均匀走向自适应**：早期统一上限，近期按层甚至按 head 分配——预设金字塔（PyramidKV、PyramidInfer）、阶梯（LaCache）、自适应（CAKE 按层偏好、D2O 按注意力密度、DiffKV/Ada-KV 按 head 稀疏度、EvolKV 用进化搜索）。Ada-KV 和 EvolKV 甚至把预算策略做成**可插拔组件**，套在已有驱逐规则上。

> **要点**（附录 E.5）：**token 重要性不应只看注意力分数**——KV 范数是强且低开销的信号。异构预算优于统一上限：**浅层通常值得更大保留量，深层强调全局语义、能容忍更高稀疏度**。此外，**驱逐配合相似度召回或合并，比硬删除更强**（D2O）。

---

## 7. 全文的真正价值：两张交叉分析图

### 7.1 行为 × 目标矩阵（Tab. 7）

作者对每个行为在七个服务目标上标注**直接影响（●）/ 间接影响（○）**，并用星号（⋆）统计标记「≥70% 的论文报告了该项收益」：

| 行为 | 平均延迟 | 尾延迟 | 吞吐 | GPU 内存 | 互连 I/O | 能耗 | 质量损失↓ |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| KVS（调度） | ●⋆ | ● | ● | ○ | ○ | ○ | ○ |
| OVLP（重叠） | ●⋆ | ○ | ●⋆ | ○ | ● | ○ | ○ |
| HAE（硬件感知） | ●⋆ | ● | ●⋆ | ● | ● | ● | ○ |
| MHO（内存层级） | ○ | ○ | ○ | ●⋆ | ●⋆ | ○ | ● |
| CDO（设备编排） | ○ | ○ | ● | ● | ●⋆ | ● | ● |
| KVCC（压缩） | ○ | ○ | ○ | ●⋆ | ● | ● | ● |
| KVRM（保留） | ○ | ○ | ○ | ●⋆ | ● | ● | ● |

由此得出 O1–O5：

- **O1**：结构类工作最多，且主导内存节省——反映社区**偏向内存效率**的取向。
- **O2**：时间行为最直接作用于**延迟与吞吐**（KVS/OVLP/HAE 分别对应调度停顿、流水气泡、设备利用率不足）。但**尾延迟的上报非常稀少**。
- **O3**：空间方法主要针对**互连 I/O**，且几乎总是与 OVLP 配对（靠重叠藏传输延迟）。
- **O4**：**能耗几乎无人探索**，尽管很多方法降低了内存或计算强度，理应带来能耗收益。
- **O5**：**质量损失普遍存在**：时间方法有请求处理不一致的风险，空间方法有 KV 数据缺失的风险，结构方法直接降低 KV 精度。实际问题不是"有没有损失"，而是**损失是否可控**。

### 7.2 行为 × 行为协同亲和网络（Fig. 6 / 附录 F）

这是方法论上最讲究的一块。作者把附录 D 的主/次类别标注量化：主类别权重 1，次类别权重 $\alpha=0.5$，即

$$\omega(\ell_{p,i}) = \mathbb{1}[\ell_{p,i}=P] + \alpha\,\mathbb{1}[\ell_{p,i}=S]$$

原始共现矩阵 $C_{ij} = \sum_{p\in\mathcal{P}} \omega(\ell_{p,i})\,\omega(\ell_{p,j})$，但它会被各行为的**流行度**（边际频次）带偏，因此用 **Tanimoto 系数**归一化：

$$Q_i = \sum_{p\in\mathcal{P}} w_{p,i}^2, \qquad S_{ij} = \frac{C_{ij}}{Q_i + Q_j - C_{ij}}$$

阈值 $\theta = 0.14$ 以上才连边，边粗正比于 $S_{ij}$。归一化后的亲和矩阵（Tab. 12）：

| | KVS | OVLP | HAE | MHO | CDO | KVCC | KVRM |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KVS** | – | 0.09 | **0.16** | 0.11 | 0.07 | 0.00 | **0.14** |
| **OVLP** | 0.09 | – | **0.42** | **0.30** | **0.38** | 0.06 | 0.05 |
| **HAE** | **0.16** | **0.42** | – | 0.10 | **0.53** | 0.02 | 0.06 |
| **MHO** | 0.11 | **0.30** | 0.10 | – | 0.10 | 0.06 | 0.10 |
| **CDO** | 0.07 | **0.38** | **0.53** | 0.10 | – | 0.03 | 0.06 |
| **KVCC** | 0.00 | 0.06 | 0.02 | 0.06 | 0.03 | – | 0.02 |
| **KVRM** | **0.14** | 0.05 | 0.06 | 0.10 | 0.06 | 0.02 | – |

两个结论跳出来：

- **O6：HAE–CDO 是最强协同模式（0.53）**。利用设备异构性的计算布局，天然要和 KV 共置/传输一起设计，在利用率和 I/O 上同时得益。OVLP–HAE（0.42）和 OVLP–CDO（0.38）紧随其后，印证了 O3。
- **O7：KVCC 几乎完全孤立**。整行最大值只有 0.06，**与 KVS 的亲和度是 0.00**——最热门的研究方向，与系统其他部分零协同。这是全文最有信息量的一个数字。

---

## 8. 六个开放挑战

| 挑战 | 源自 | 核心问题 |
| --- | --- | --- |
| **C1** SLO 驱动的尾延迟控制 | O2 | 尾延迟主导用户体验，但多数系统压根不报。长上下文和突发流量下，KV 生成/迁移/压缩都可能触发 SLO 违约，且**违约常源于跨行为干扰而非单一操作**。真正的难点是把 SLO 违约**归因到具体 KV 行为和路径**，并建立标准化的抢占与降级语义。 |
| **C2** 能耗感知的 sKis | O4 | 把功耗剖析纳入运行时决策，建立服务期能耗模型，在功率约束下联合优化能耗-延迟-质量。不止看总功率，还要在真实 trace 下测**每请求/每 token 能耗**；另一方向是研究"能耗友好"的 KV 粒度与布局。 |
| **C3** 可信且高效的 sKis | O5 | 结构方法可能驱逐或压缩掉"低显著度但关键"的上下文——合同里的例外条款、财务上限里的一个高值——**平均准确率纹丝不动，但在小部分输入上灾难性出错**。Agent 负载下这类风险被分布漂移放大：统计上稀疏的 token 恰恰逻辑上重要，被剪掉后 agent 会"幻觉成功"。有意思的是 sKis 技术**双刃**：RobustKV 把 KV 驱逐反过来做成越狱攻击的防御。 |
| **C4** 可泛化的 HAE–CDO | O6 | 现有策略常为特定互连或单租户场景定制。虽然当前 sKis 实践以 GPU 为中心，但**时间-空间-结构这个透镜本身是硬件无关的**——跨硬件变化的只是具体的层级、互连和 kernel 支持。需要让协同模式可移植（NVLink/NVSwitch/PCIe/CXL）并适配多租户。 |
| **C5** 协同优化与中间语义 | O7 | 多数工作孤立优化单一行为。可探索共享预算下的协同优化，例如**在给定预测复用率、成功概率和 I/O 争用下，联合决定驱逐、卸载与预取**。边缘场景下 KVCC 应与 OVLP 耦合——KVCC 降内存，OVLP 防止 (de)compression 开销吃掉收益。 |
| **C6** 统一基准 | — | 各工具的指标定义与测量方式不一致，导致跨论文无法苹果对苹果比较。 |

### C5 的"中间语义"是个不错的想法

作者提出：不要只有「保留 vs 驱逐」这种二元决策，而应引入中间状态——`GPU 上可回收`、`GPU 上已压缩`、`CPU 上已压缩`、`CPU/SSD 上已摘要`……协同优化策略就形式化为**这些状态之间的转移**：

- **compress-then-offload**：`keep` → `GPU 上已压缩` → `CPU 上已压缩`，创造一个低保真的常驻态，用精度换 I/O 带宽。
- **lazy eviction**：先转到 `GPU 上可回收` 并给一个宽限期，允许廉价恢复，宽限期过后才真正转到 `evict`。

### C6 的基准设计清单（附录 G.3.2）

作者不打算定义一套完整基准，只给设计原则：

**指标**——除标准指标外补充两类：
1. **可信性指标**：尾延迟（P90/P95/P99）、SLO 违约率、goodput（满足 SLO 的吞吐）、recall SLO（特定语义片段的成功率）、语义违约率。
2. **KV 相关资源指标**：KV 内存占比（占 GPU 总内存 %）、平均有效 KV 位宽、KV 相关互连 I/O 量、各内存层 KV 命中率、**KV 相关停顿占比**（等待 KV 传输的时间 %）、有效带宽利用率（有用 KV 传输占比）、能效（焦耳/token）。

**负载**——三类，分别压测三个维度：
1. **多租户/突发在线负载** → 压测时间维度的调度稳定性；
2. **长上下文负载** → 压测空间维度的放置与迁移；
3. **异构负载**（agent 负载、结构化编辑、领域漂移）→ 压测结构维度对分布漂移的鲁棒性。

**上报标准**：分档上下文长度下的性能曲线（验证可扩展性）、结构方法的准确率-内存权衡曲线、详细的硬件与拓扑配置（对时间和空间方法尤其重要）。

---

## 9. 与本仓库其他论文的关联

这篇综述可以当作本仓库已读论文的一张地图。按 sKis 分类法归位：

| 本仓库论文 | sKis 主类别 | 在综述中的位置 |
| --- | --- | --- |
| [Mooncake](Mooncake-论文解读.md) | KVS + HAE | 综述明确点名：PD 分离 + KV 中心调度器 + 分布式缓存池的耦合 |
| [Preble](Preble-论文解读.md) | KVS | 请求级复用感知调度的代表 |
| [Pensieve](../papers/Stateful%20Large%20Language%20Model%20Serving%20with%20Pensieve.pdf) | MHO | 跨设备层级、重要性感知 |
| [PRISM](PRISM-论文解读.md) | KVS + KVRM | 调度-内存协同设计 |
| [DualMap](DualMap-论文解读.md) / [CacheRoute](CacheRoute-论文解读.md) / [SkyWalker](SkyWalker-论文解读.md) / [GORGO](GORGO-论文解读.md) | **KVS（请求级、复用感知）** | 综述覆盖较薄的一块 |
| [TOPAS](TOPAS-论文解读.md) | KVS | agent 负载的前缀状态调度，对应 C3 提到的 agent 分布漂移风险 |
| [BanaServe](BanaServe-论文解读.md) | CDO + HAE | 对应 O6 最强协同模式 |

一个观察：**本仓库关注的"前缀感知路由/负载均衡"这条线，在综述里被归入 KVS 的"请求级复用感知"子类，而这恰是全文着墨最少的部分之一**——KVS 的研究密度反推值只有 16.1，在七个行为里排倒数第三。详细分析见 [§4.1.11](#4111-补充观察综述在-kvs-上的盲区)。

---

## 10. 评价

**优点**：

1. **分类法是真的正交且稳定**。时间/空间/结构这个切法与模型细节、kernel 细节解耦，因此新技术出现时能被稳定归位——这在一个半年就换一波 SOTA 的领域里是有价值的。
2. **两张交叉分析图有量化依据**，不是拍脑袋。Tanimoto 归一化处理边际流行度偏差这个细节做得很扎实，否则 KVCC 因为论文多会虚高。
3. **"低位宽 ≠ 端到端收益"这类提醒非常务实**，直指工程落地时最常踩的坑。
4. **表格密度极高**：量化 23 个方法、驱逐 23 个方法、低秩 6 个、结构化 8 个、OVLP 8 个，每个都按核心算法轴对齐——单是 Tab. 3 和 Tab. 6 就值得单独存档当查询手册。

**局限**（作者自陈 + 我的补充）：

1. **没有任何新实验**。所有结论都是对公开论文和开源实现的综合。作者明确拒绝跨论文聚合加速比或内存数字，理由是"报告的收益与模型、硬件、负载、baseline 选择强耦合"——这个克制是对的，但也意味着这篇综述**无法回答"哪个方法更好"**，只能回答"有哪些方法、它们在设计空间的哪里"。
2. **协同亲和网络衡量的是"文献共现"，不是"验证过的性能收益"**。作者自己标注了这一点。所以 O6（HAE–CDO 最强）严格讲只说明"研究者习惯把它们一起做"，不等于"一起做真的更好"。
3. **KVS 的请求级/跨实例路由覆盖偏薄**（见 §9）。
4. **MLA 类模型层改动被排除**，但现实中 DeepSeek-V2/V3 的 MLA 对 KV 占用的影响量级远超多数 sKis 方法。范围定义虽然自洽，却让综述回避了当前最有效的一条路径。
5. **时效性**：截止 2026 年 7 月，且承认可能遗漏同等价值的贡献。

**一句话评价**：这不是一篇"读完就能上手调优"的综述，而是一篇**帮你在设计空间里定位自己工作、并找到空白区域**的综述。如果你要写 KV cache 相关的论文，Tab. 12 的那个 KVCC 全零行就是最直接的选题提示；如果你在做系统落地，附录 E 的五组 takeaway 和 G.3.2 的指标清单最有用。
