# BanaServe: Unified KV Cache and Dynamic Module Migration for Balancing Disaggregated LLM Serving in AI Infrastructure —— 论文解读

> 论文：*BanaServe: Unified KV Cache and Dynamic Module Migration for Balancing Disaggregated LLM Serving in AI Infrastructure*
> 作者：Yiyuan He（南方科技大学 / 中科院深圳先进院 / 阿里巴巴 AIOS Team）、Minxian Xu（通讯，SIAT）、Jingfeng Wu、Jianmin Hu、Chong Ma、Min Shen、Le Chen、Chengzhong Xu（澳门大学）、Lin Qu、Kejiang Ye
> 版本：**Preprint**，arXiv:2510.13223**v1**（2025-10-15），cs.DC
> 期刊状态：**PDF 使用的是 Wiley《Software: Practice and Experience》(SPE) 的官方 LaTeX 模板**——首页有 `Received: Added at production / Revised: Added at production / Accepted: Added at production`、`DOI: xxx/xxxx`、`ARTICLE TYPE` 等**未填充的占位符**，参考文献里也刻意堆了 3 篇 SPE 论文（ref 4/5/6）。这说明它是**投稿到 SPE 的版本**，但 arXiv v1 里**没有任何卷号、期号、页码或已接收日期**，无法证实"已发表"。综述中"已发表于 SPE"的说法应降级为"SPE 在投/格式化，接收状态未在 PDF 中体现"。
> 代码：**未提供**（全文无任何仓库链接）
>
> ⚠️ **阅读提示**：这篇论文在本综述中的价值**不在于它的系统性能数字**（那部分问题很大，见第 5、6 节），而在于它是整个「前缀亲和路由 vs 全局 KV 池」路线之争里，**唯一一篇把"反对前缀亲和路由"直接写进 Abstract 和 Contribution 列表的论文**。读它的正确姿势是：把它当成一份**立场声明书**来审，逐条追问「你的论据在哪」。答案会让人失望——它对核心立场的论证是**一张手画的示意图 + 反复重述的断言**，而实验里**没有任何一个前缀亲和 baseline**。

---

## 1. 一句话概括

BanaServe 主张：PD 分离架构下的三大顽疾（静态资源分配、prefill/decode 天然负载不对称、**前缀缓存亲和路由造成负载倾斜**）有一个共同病根——**资源分配与状态（缓存/权重）放置被绑死了**。它的解法是把状态从计算里解耦出来：用**层级权重迁移**（粗粒度）+ **注意力头级 KV 迁移**（细粒度）动态重分配 prefill/decode 之间的算力与显存，用一个 **CPU/SSD 支撑的 Global KV Cache Store** 让所有 prefill 实例共享前缀缓存，从而使路由器**彻底不看 cache hit rate、只看负载**。实测相对 vLLM 提升吞吐 1.2×–3.9×、相对 DistServe 1.1×–2.8×。

但（这是全文最关键的一句）：**它用来支撑"前缀亲和必然倾斜"这一核心立场的，只有一张示意图；它的两个 baseline（vLLM、DistServe）都不是前缀亲和路由器；它的核心机制（迁移）没有任何一处开销实测。**

---

## 2. 它要解决的问题

### 2.1 作者列出的三条"limitation"

| # | 问题 | 作者给的证据 | 证据类型 |
| --- | --- | --- | --- |
| 1 | 静态配置导致资源闲置：RPS ≤ 10 时 HFT / vLLM 有 20%–40% GPU 资源闲置 | Fig.1，单张 A100 + LLaMA-13B，5 次重复 | ✅ **实测**（但只有单实例，跟"静态 PD 配置"其实没直接关系） |
| 2 | **前缀缓存亲和路由造成严重负载倾斜** | Fig.2a | ❌ **纯示意图**，见 §2.2 |
| 3 | PD 分离天然负载不平衡：prefill 算力 >95% / 显存 ~35%，decode 算力 ~35% / 显存高 | Fig.2b，DistServe + LLaMA-13B + Alpaca 实测 | ✅ **实测**（这条是扎实的，也是本文最站得住的动机） |

### 2.2 ⚠️ 关键审查：Fig.2a 到底是什么

这是全篇最需要盯住的地方。Figure 2 的**总标题**写的是：

> "Two empirically validated limitations in current LLM serving architectures."（两个**经过实证验证**的局限）

但 Fig.2a 的正文（第 2 页）是这么写的：

> "**a representative scenario illustrates** how a prefix cache aware router allocates incoming requests across three serving instances... In **this example**, Instance 1 achieves the highest cache hit rate and receives the majority of incoming queries. It operates at full compute capacity (100%)... Instance 2 receives fewer queries and maintains low compute utilization (40%)... Instance 3 sustains moderate compute utilization (60%)..."

三个实例，利用率恰好是 **100% / 40% / 60%** 这样的整数；缓存前缀恰好是 **Q1–Q5 / Q6–Q7 / Q8–Q10**。这是**手工构造的示意例子**，不是任何一次测量的输出。文中**没有说明**这个"prefix cache aware router"具体是哪个系统（SGLang router？vLLM 的 prefix cache？自研？），没有 trace、没有数据集、没有硬件、没有 CV / Gini / p99 等任何倾斜度量。

而 Fig.2b 明确写了 "derived from **empirical measurements conducted in our experimental environment**... we deployed the DistServe with a LLaMA-13B model, using the publicly available Alpaca dataset"。

**两幅子图的证据等级完全不同，但被 Figure 总标题统一标成了 "empirically validated"。** 这是一处实质性的表述夸大。

### 2.3 "正反馈飞轮"论证

作者对倾斜机制的口头论证是（第 3 页）：

> 高命中率实例吸引更多请求 → 缓存内容进一步膨胀 → 调度优先级进一步提高 → 排队延迟上升、缓存效率下降；低命中率实例既存冗余数据又反复重算。

这个飞轮**在纯亲和策略下是真实的**——DualMap 与 GORGO 都用实测证实了（见 §7）。但作者随即把它推广到**所有**考虑 cache 的路由器上，包括他自己在 Fig.2a 里明确描述为"based on **both** cache hit rate **and current load**"的混合策略。**一个已经把负载写进目标函数的路由器为什么还会 100%/40%/60% 地倾斜，作者没有给出任何解释。** 这是论证链上的断裂点，详见 §7.3 的稻草人分析。

### 2.4 作者自己总结的病根

> "These limitations share a common root cause: **the tight coupling between resource allocation and state (cache) placement**."

这句话本身是好的。把"计算在哪"和"状态在哪"解耦，确实是 Mooncake / MemServe / LMCache 这一路线的共同哲学。BanaServe 的独特之处只是把它推到了极端：**连模型权重都可以搬**。

---

## 3. 核心设计

BanaServe 由三个机制组成，加两个运行时算法。

### 3.1 层级权重迁移（Layer-level migration，粗粒度）

把一段连续的 Transformer 层从一个 GPU 实例整体搬到另一个，权重 $W_\ell$ 和对应 KV Cache $\mathcal{K}_\ell$ 一起搬，实现**动态模型并行**。

迁移决策被写成一个最小化峰值利用率的问题：

$$\min_{\mathcal{M}} \; \max_{g \in \mathcal{G}} U_g(\mathcal{M}) \qquad \text{s.t.}\quad T_{\text{mig}}(\mathcal{M}) \le T_{\text{budget}} \tag{1,2}$$

数据量与延迟模型：

$$S^{\text{total}}_\ell = S^{w}_\ell + S^{kv}_\ell, \qquad T_{\text{layer}} \approx \frac{S^{\text{total}}_\ell}{B_{\text{net}}} + T_{\text{sync}} \tag{3,4}$$

正确性靠"搬完照算"来保证：$\mathbf{y}_\ell = f_\ell(\mathbf{x}_\ell; W_\ell, \mathcal{K}_\ell)$。

> 🔍 **这里有个被一笔带过的大数字**。作者只说 "$S^w_\ell \gg S^{kv}_\ell$，延迟由权重传输主导，但在 NVLink / Infiniband 下仍然 practical"，**没有算一次**。我们替它算：LLaMA-13B 有 40 层，单层权重约 $13\text{B}/40 \times 2\,\text{bytes} \approx 650\,\text{MB}$（FP16）。在作者自己后面用的 200 Gbps（25 GB/s）下，**单层迁移 ≈ 26 ms**；要挪动足以改变负载格局的分量（比如 8 层），就是 **~210 ms + 同步开销**，而且这段时间里流水线是被切开的。全文**没有任何一处测量过 $T_{\text{layer}}$ 或 $T_{\text{sync}}$**，$T_{\text{budget}}$ 和效率阈值 $\rho$ 的取值也从未给出。

### 3.2 注意力头级 KV 迁移（Attention-level migration，细粒度）

沿 attention head 维度把 KV Cache 切成两个不相交子集：$K^{(1)}, V^{(1)}$ 留在热 GPU，$K^{(2)}, V^{(2)}$ 卸载到冷 GPU，两边并行算局部 attention，再合并。

$$S^{(j)} = Q (K^{(j)})^\top,\qquad A^{(j)} = \exp(S^{(j)}) \tag{6,7}$$

$$\ell^{(j)} = \begin{cases} \sum_i A^{(1)}_i, & j = 1\\[2pt] \ell^{(1)} + \sum_i A^{(2)}_i, & j = 2\end{cases} \tag{8}$$

$$O^{(j)} = \frac{A^{(j)}}{\ell^{(j)}} \cdot V^{(j)}, \qquad O = O^{(1)} + O^{(2)} \tag{9,10}$$

$$T_{\text{attn}} \approx \frac{S_{kv}}{B_{\text{net}}}, \qquad T_{\text{attn}} \ll T_{\text{layer}} \tag{11}$$

> 🐛 **公式 (8)–(10) 在数学上是错的**。$O^{(1)}$ 用**局部**分母 $\ell^{(1)}$ 归一化，$O^{(2)}$ 用**全局**分母 $\ell^{(2)} = \ell^{(1)} + \sum_i A^{(2)}_i$ 归一化，然后直接相加。正确的分块 softmax（FlashAttention / Ring Attention 的标准做法）要求**两个分支最终都用同一个全局分母**，即 $O = (A^{(1)} V^{(1)} + A^{(2)} V^{(2)}) / \ell^{\text{global}}$，$O^{(1)}$ 必须被 rescale 一次。按论文写法，$O^{(1)}$ 的权重被系统性放大。这大概率是写作疏漏而非实现错误（否则模型输出会崩），但它说明**公式部分没有被认真核对过**。
>
> 🔍 另外，公式 (8) 里 $\ell^{(2)}$ **依赖** $\ell^{(1)}$，也就是冷 GPU 必须等热 GPU 的部分和到达才能归一化——这与正文"$S^{(1)}$ 和 $S^{(2)}$ can be computed **entirely in parallel**... without intermediate data dependencies"直接矛盾。真实情况是：算 score 可以并行，**归一化不能**。
>
> 🔍 **最要命的是 (11) 的量纲**。$T_{\text{attn}}$ 被建模成一次性的 KV 搬迁成本。但头级切分不是"搬一次就完事"——切分之后，**decode 的每一步、每一层都要跨设备交换 $\ell$ 和 $O$**。这是一个**持续存在的 per-token per-layer 同步开销**，量级由网络往返延迟（微秒级但乘以 40 层 × 数百 token）决定，而不是由 $S_{kv}/B_{\text{net}}$ 决定。作者把一个稳态通信开销当成一次性迁移成本来算了。这是全文对"迁移代价"最严重的低估。

### 3.3 Global KV Cache Store + 层级重叠传输

所有 prefill 节点共享一个 CPU/SSD 支撑的 KV 存储，前缀 KV 与增量 KV 都写进去；decode 实例从同一个存储里取完整 KV。**路由器因此不再需要考虑缓存位置**——这是全文的立论核心。

为论证"共享存储的取数延迟可以被隐藏"，作者设计了 prefetch / compute / store 三级层间流水，并给了一组数字：

$$T_{F,\text{layer}} = \frac{T_F \cdot r}{N}, \qquad T_{KV} = \frac{S_{kv}\cdot L \cdot r}{B} \tag{12,13}$$

以 llama-3.1-8B 为例（$d_{\text{model}}=4096$, $h_{\text{total}}=32$, GQA $h_{kv}=8$, BF16）：

| 量 | 推导 | 值 |
| --- | --- | --- |
| $d_{\text{head}}$ | $4096/32$ | 128 |
| 每层每 token KV | $8 \times 128 \times 2 \times 2$ B | **4 KB** |
| 全模型每 token KV | $32 \times 4$ KB | **128 KB** |
| $T_{F,\text{layer}}$ | $270\,\text{ms} \times 0.5 / 32$ | **4.22 ms** |
| $T_{KV}$ | $4\text{KB}\times 1000 \times 0.5 / 200\,\text{Gbps}$ | **0.082 ms** |

结论：$T_{KV} \ll T_{F,\text{layer}}$（差 51×），所以传输可被完全隐藏。

> 🔍 **这个"51× 余量"是全文唯一为 Global KV Store 的代价所做的定量论证，而它建立在至少 6 个乐观假设上**：
>
> 1. **$B = 200$ Gbps 被称为 "effective PCIe bandwidth"**。PCIe Gen4 x16 理论上限约 256 Gbps，取 200 Gbps 作为"有效"值几乎等于理论峰值。而且论文明说存储是 **CPU/SSD-backed**——一旦落到 SSD 层，带宽要低 1–2 个数量级，这时 51× 的余量瞬间蒸发。
> 2. **单请求独占带宽**。真实服务是 continuous batching，几十个并发请求共享同一条 PCIe/网络链路，每请求可用带宽要除以并发度。
> 3. **只算了取（HtoD），没算存（DtoH）**。图 6 的时间线里三条通道同时跑，总流量至少翻倍。
> 4. **只算了本机 PCIe，没算跨节点网络**。"Global"意味着缓存可能在别的机器上，这一跳完全没有出现在模型里。
> 5. **没有任何元数据开销**：全局索引查找、前缀匹配、一致性协议、淘汰策略——论文对 Store 的**内部设计只字未提**（没有 hash 方案、没有分块粒度、没有 eviction、没有副本策略、没有并发控制）。对比 Mooncake，人家用了大半篇幅讲块级 hash 去重、热块复制、冷块换出。
> 6. **公式 (12) 本身很奇怪**：$T_{F,\text{layer}} = T_F \cdot r / N$ 把每层前向时间**乘以命中率**——命中率越高，"每层前向时间"越小，于是分子分母同乘 $r$，51× 这个比值对 $r$ 完全不敏感。这更像是为了让两边都出现 $r$ 而凑出来的式子，而非有物理意义的建模。
>
> **更根本的问题是，这个论证回避了最关键的对比。** 前缀亲和路由的价值不是"有缓存"，而是**缓存在 HBM 里，命中即零成本**。BanaServe 的方案是把命中成本从 0 换成一次 PCIe/网络取数。它真正需要论证的命题是：**「100% 逻辑命中率 × PCIe 取数成本」是否优于「70% HBM 命中率 × 30% 重算成本」**。这个 trade-off 全文从头到尾**没有被测量过一次**——论文甚至没有报告过任何一个实测的 cache hit rate 数字。

### 3.4 性能建模

联合目标：

$$\max_{\mathcal{M}} \;\; \alpha \cdot U_{\text{avg}}(\mathcal{M}) - \beta \cdot T_{\text{avg-latency}}(\mathcal{M}) + \gamma \cdot \Theta(\mathcal{M}) \tag{18,31}$$

延迟分解：

$$T_{\text{TTFT}} = T_p + T_x + T_q, \quad T_x = T_{\text{load}} + T_{\text{fetch}} \tag{20,21}$$
$$T_{\text{TPOT}} = T_d + T_c + T_m \tag{22}$$

迁移成本：$\text{Cost}_{\text{mig}} = k(T_{\text{x\_lat}} + T_{\text{sync}} + T_{\text{mem\_realloc}})$ (28)

吞吐：$\Theta = \dfrac{N \cdot L_{\text{out}}}{T_{\text{TTFT}} + L_{\text{out}} \cdot T_{\text{TPOT}}}$ (30)

> 🔍 这一整节（4.3）是**装饰性建模**：$\alpha, \beta, \gamma$ 的取值从未给出，$U_{\text{avg}}$/$T$/$\Theta$ 量纲完全不同却直接加权相加，公式 (18)/(31) 是同一个式子写了两遍，而且**后面的算法 1 和算法 2 都不求解这个目标函数**——它们用的是简单的阈值启发式。这些式子对系统的实际行为没有任何约束力，也没有做过任何模型-实测的校验。

### 3.5 两个运行时算法

**算法 1：Adaptive Module Migration**

设备归一化负载定义为算力与显存利用率**直接相加**（取值域 0–2）：

$$U_d = \frac{C_d}{C_d^{\max}} + \frac{M_d}{M_d^{\max}} \tag{32}$$

| 步骤 | 行为 |
| --- | --- |
| 1. 负载测量 | 对每个 $d \in D$ 调 `MeasureUtilization`（compute + memory） |
| 2. 负载分类 | $\textit{overload} = \{d : load[d] - \min(load) > \delta\}$，$\textit{underload} = \{d : \max(load) - load[d] > \delta\}$ |
| 3. 迁移决策 | 若 $\Delta \ge \delta$ 且支持层迁移 → `MigrateLayer`；否则若支持头迁移 → `MigrateKVHeads`。执行前校验 $\textit{Benefit}(m)/\textit{Cost}(m) \ge \rho$，其中 $\textit{Benefit} = \Delta_{\text{before}} - \Delta_{\text{after}}$，$\textit{Cost} = T_{\text{transfer}} + T_{\text{sync}}$ |
| 4. 更新 | 更新 load 与配置，循环至无设备同时满足两个条件 |

复杂度声称 $O(|D| + N_m)$，用双阈值 $(\delta^\uparrow, \delta^\downarrow)$ 做迟滞控制防抖。

> 🐛 **步骤 2 的集合定义有 bug**：`overload` 相对 $\min$ 定义，`underload` 相对 $\max$ 定义，两者是**独立**的判据。三设备负载为 (1.8, 1.0, 0.2)、$\delta = 0.5$ 时，中间那台 1.0 **同时属于** overload（1.0−0.2 > 0.5）和 underload（1.8−1.0 > 0.5）。它既是迁出方也是迁入方，while 循环的终止性没有保证。
>
> 🐛 **复杂度 $O(|D| + N_m)$ 也不对**：$O(N_m)$ 只覆盖了**一次**模块选择，而 while 循环会执行多次迁移，真实复杂度至少是 $O(|D| + k \cdot N_m)$。
>
> 🔍 **公式 (32) 把算力利用率和显存利用率等权相加**，没有任何论证。一台 100% 算力 / 0% 显存的机器与一台 50%/50% 的机器在这个度量下完全等价——而这两种状态对"该不该再给它一个请求"的含义截然相反。讽刺的是，论文的第 3 条 motivation（Fig.2b）恰恰是在说 prefill 与 decode 的**算力/显存画像完全相反**，然后它的负载度量把这两维压成了一个标量。

**算法 2：Load-aware Request Scheduling**

| 步骤 | 行为 |
| --- | --- |
| 1 | 测每个 prefill 实例的 $U_{p_i}$（同 (37)，仍是 compute+memory 相加）与队列长度 |
| 2 | 按 (load, queue_len) 升序排序 |
| 3 | 逐请求选最轻载实例；若 $U_{p_{\text{target}}} < \delta_L$ 则分配，**否则**改选队列最短的实例分配；分配后用 `EstimateLoad(req)` 递增负载 |
| 4 | 返回 dispatch map |

复杂度 $O(|P|\log|P| + |Q|)$。

> 🔍 这就是全文"纯负载感知路由"的全部内容：**一个带次级排序键的 least-loaded**。它没有比 vLLM / Envoy 的 least-request 多出任何东西，也没有 Power-of-Two-Choices、没有有界一致性哈希、没有准入控制、没有过载丢弃。第 3 步的 fallback 逻辑尤其奇怪：最轻载实例超过 $\delta_L$ 时"改选队列最短的"——但列表本来就是先按 load 再按 queue 排的，且**所有**实例都可能超阈值，此时算法仍然照单全收，没有排队或拒绝路径。
>
> 也就是说：**BanaServe 用一个非常弱的路由算法，替换掉了它所批判的（据称很复杂的）缓存亲和路由。** 它的赌注全押在"Global KV Store 让缓存位置不再重要"这个前提上——而这个前提正是没被实验检验的那一个。

---

## 4. 实验方法

### 4.1 配置

| 项 | 设置 |
| --- | --- |
| 模型 | **LLaMA-13B**（族内评测）、**OPT-13B**（跨架构验证），均为 decoder-only |
| 数据集 | **Alpaca**（短上下文，输入 4–50 token）、**LongBench**（长上下文，~2K–85K token，21 任务 6 类别） |
| 输出长度 | 统一截断至 **512 token** |
| 负载 | **1–20 RPS**（LongBench 只到 10 RPS），**泊松到达** |
| 预热 | 60 秒 |
| 重复 | 每配置 **5 次**不同随机种子，报告均值 + 95% 置信区间 |
| 指标 | Throughput、Total time、Average latency（拆 TTFT / inter-token latency） |
| **硬件** | ❌ **全文未给出**——主实验的 GPU 型号、数量、prefill/decode 实例数、互连方式（NVLink？IB？）**一个字都没写** |

### 4.2 Baseline

只有两个：

| Baseline | 类型 | 是否前缀亲和路由 |
| --- | --- | --- |
| **vLLM** | 单体（co-located）+ PagedAttention + continuous batching | ❌ |
| **DistServe** | PD 分离（OSDI'24） | ❌ |

> 🚨 **这是全篇最致命的方法论问题，必须单独拎出来说。**
>
> 论文的**核心主张**是"前缀缓存亲和路由会造成负载倾斜，应该被全局 KV 池取代"。要检验这个主张，唯一有意义的对照组是**一个真正的前缀亲和路由器**。而实验里：
>
> - ❌ 没有 **SGLang Router**（cache-aware scheduling，论文正文引了 SGLang[14] 并明确点名它"routes requests to maximize KV Cache hit rates"，却不拿它做 baseline）
> - ❌ 没有 **Preble**（ICLR'25，前缀亲和路由的代表作）——**全文连引用都没有**
> - ❌ 没有 **NVIDIA Dynamo KV Router**（引了 Dynamo[26]，只引了产品主页 URL，无 baseline）
> - ❌ 没有 **Mooncake / MemServe**（引了[30][31]，无 baseline）——这两个还是**同路线**的直接竞争者
> - ❌ 没有 **round-robin / least-load 消融**，无法回答"提升到底来自 Global Store 还是来自迁移"
> - ❌ 没有 **vLLM/SGLang 开启 prefix caching** 的配置说明
>
> vLLM 是单体架构、DistServe 是无前缀缓存的 PD 分离——**两个 baseline 都不做缓存亲和路由**。BanaServe 声称推翻的那条路线，**根本没有出现在它的实验里**。这不是"baseline 不够强"，而是**对照组缺失**：论文测的是"BanaServe 整体 vs 两个通用系统"，而不是"负载感知路由 vs 缓存亲和路由"。
>
> 更进一步：**全文没有任何 ablation study**。三个机制（层迁移、头级迁移、Global Store）+ 负载感知路由共四个组件，任何一个的独立贡献都无法从结果中分离出来。所以即使数字全是真的，也**推不出**"应该放弃缓存亲和路由"这个结论——提升完全可能来自 PD 之间的动态资源再平衡（motivation 3，那条有实测支撑的），与路由策略无关。

### 4.3 未测量的东西清单

对一篇以"负载均衡"为题的论文，下面这些的缺席是反常的：

| 应该测但没测 | 为什么重要 |
| --- | --- |
| **任何负载倾斜度量**（CV / Gini / max-min / p99 实例利用率） | 这是论文的**核心命题**。DualMap 报 CV=1.24，GORGO 报 p95 TTFT 6784ms，BanaServe 一个倾斜数字都没有 |
| **cache hit rate** | 论文的建模里 $r$ 是核心参数，实验里从未测量 |
| **迁移开销 $T_{\text{layer}}$ / $T_{\text{attn}}$ / $T_{\text{sync}}$** | 核心机制的代价，零测量 |
| **迁移频率 / 触发次数 / 抖动** | 无法判断迁移是否真的发生过 |
| **$T_{\text{load}}$ / $T_{\text{fetch}}$ 分解** | Global Store 的实际代价，公式 (21) 定义了却从未测 |
| **TTFT / TPOT 分位数** | 只报 average latency，尾延迟完全不可见 |
| **$\delta, \delta_L, \rho, \alpha, \beta, \gamma, T_{\text{budget}}$ 的取值** | 全部超参数无一给出 |
| **Global Store 的容量与淘汰行为** | 长上下文 85K token 场景下存储压力如何 |

---

## 5. 主要结果

原文只给了 4 组折线图（Fig. 8–11），**没有一张数值表格**，所有数字都以"1.x×"/"y%"的区间形式散落在正文里。整理如下：

| 模型 | 上下文 | vs DistServe 吞吐 | vs vLLM 吞吐 | 延迟下降 vs vLLM | 延迟下降 vs DistServe |
| --- | --- | --- | --- | --- | --- |
| LLaMA-13B | 短（Alpaca） | 1.1×–1.2× | 1.1×–1.2× | — | — |
| **OPT-13B** | **短（Alpaca）** | **2.8×–3.9×** | **up to 3.9×** | **3.9%–78.4%** | **1.4%–70.1%** |
| LLaMA-13B | 长（LongBench） | 1.3×–1.5× | up to 1.5× | 20.6%–65.3% | 1.4%–20.6% |
| OPT-13B | 长（LongBench） | 1.1×–1.3× | up to 1.3× | — | — |

摘要里那组亮眼的数字（**3.9× 吞吐、78.4% 延迟下降**）来自**唯一一个格子**：OPT-13B + 短上下文。其余三格都在 1.1×–1.5× 区间。

> 🔍 **效应量点评**：真实的、跨配置一致的收益大约是 **1.1×–1.5×**。这对一个引入了「跨 GPU 搬权重 + 跨 GPU 切 attention head + 全局 KV 存储」这么重的机制的系统来说，**不算大**。而 3.9× 这个数字既是单点、又出现在 OPT-13B 上（LLaMA-13B 同配置只有 1.1×–1.2×）——**同参数量、同架构族、同数据集，收益差 3 倍**，论文对此的解释只有一句"high-frequency context-switch scenarios"，没有任何机理分析或佐证数据。一个 3× 的跨模型方差没有被解释，通常意味着某个配置里 baseline 被配坏了。

### 5.1 ⚠️ 图表里的三处内部矛盾

这几处是判断结果可信度的关键：

1. **吞吐的单位与量级对不上**。§5.1.2 明确写 "Throughput, **measured in tokens per second**"，但 Fig. 8–11 的 y 轴全都是 "**Throughput (requests/s)**"，且**峰值只有 0.4–1.0 req/s**。而 x 轴的输入负载是 **1–20 RPS**。也就是说：在 20 RPS 的输入下，最好的系统只输出 0.4 req/s——**99.98% 的请求没有完成**。三个系统在整个 RPS 区间都远远处于崩溃状态，那么"吞吐提升 3.9×"比较的是三个都已经过载的系统之间的相对崩溃程度。要么 y 轴标错了，要么这组实验的负载设置有严重问题。

2. **"Average Latency (ms)" 的数值不可能**。Fig. 8–11 的延迟轴范围是 **1–17 ms**。一个 13B 模型生成 **512 个 token**，即便 TPOT 只有 10 ms，端到端也要 **5 秒以上**。1–17 ms 比物理下限低了约 **3 个数量级**。同一批图里的 "Total Time" 是 **10–80 秒**，量级才是对的。所以 "Average Latency" 的单位（或定义）是错的，而摘要里"延迟下降 78.4%"正是从这张图上读出来的。

3. **95% 置信区间没有出现在图上**。§5.1.3 承诺"repeated five times... report mean values along with 95% confidence intervals"，但四组图的描述里没有任何误差带/误差棒。5 次重复在 LongBench 这种任务高度异质、输入长度跨 2K–85K 的数据集上，方差会很大。

> 单独看任何一条都可以辩解为排版失误；三条同时出现，且**恰好都发生在支撑摘要核心数字的那几张图上**，说明结果部分没有经过基本的自洽性检查。**这些数字不适合直接引用到综述里。**

### 5.2 硬件与"大规模"的落差

致谢与作者贡献声明里写：阿里巴巴 AIOS Team "provided Alibaba's **large-scale experimental platform** for performance validation"。而实际报告的实验是：**13B 模型、单数据集、1–20 RPS、硬件配置未披露**。标题里的 "AI Infrastructure" 与实际的实验规模之间落差很大。对比同路线的 Mooncake（生产集群、真实 trace）、CacheRoute（60× H100），BanaServe 的**实验规模属于本综述所收论文中偏小的一档，且是唯一一篇连硬件都没写的**。

---

## 6. 局限

### 6.1 作者自述（Conclusion 节）

1. **异构硬件未优化**：调度与迁移策略没有针对混合精度加速器、专用通信 fabric 做优化，可能导致集群资源闲置。
2. **纯反应式，无预测**：只依赖瞬时负载指标，没有基于历史模式的预测模型，在负载快速波动下效果打折。
3. **单区域**：未处理跨区域部署，尤其是 WAN 上的 KV 缓存共享。

未来工作：硬件感知调度、基于 RL 的预测式编排、WAN 优化的多区域 KV 同步、与 Kubernetes 集成。

> 这三条自述**都是"未来可以做得更好"型的**，没有一条触及"我的核心主张可能不成立"。特别是第 3 条——单区域是 BanaServe 立场的**最有利环境**（带宽最高、延迟最低），一旦跨区，Global KV Store 的代价会指数级放大，而前缀亲和路由的相对优势正是在带宽稀缺时最大。作者把这个最危险的边界条件列为"future work"。

### 6.2 我的补充

| # | 局限 | 严重度 |
| --- | --- | --- |
| L1 | **核心命题的对照组缺失**：批判前缀亲和路由，却没有一个前缀亲和 baseline，也没引 Preble | 🔴 致命 |
| L2 | **Fig.2a 是示意图不是测量**，却被 Figure 总标题标为 "empirically validated" | 🔴 致命 |
| L3 | **无 ablation**，四个组件的贡献无法分离，提升可能全部来自 PD 动态再平衡而与路由无关 | 🔴 致命 |
| L4 | **迁移开销零测量**，$T_{\text{layer}}$、$T_{\text{sync}}$、迁移频率、抖动全部缺席 | 🔴 高 |
| L5 | **结果图存在单位/量级矛盾**（吞吐 0.4 req/s @ 20 RPS；延迟 1–17 ms @ 512 token 输出） | 🔴 高 |
| L6 | **头级迁移的通信成本被建模成一次性**，实际是 per-token per-layer 的稳态同步开销 | 🟠 高 |
| L7 | **公式 (8)–(10) 分块 softmax 归一化错误**；"entirely parallel" 与 $\ell^{(2)}$ 依赖 $\ell^{(1)}$ 自相矛盾 | 🟠 中 |
| L8 | **Global Store 内部设计完全空白**：无分块、无索引、无一致性、无淘汰、无副本策略 | 🟠 高 |
| L9 | **算法 1 的 overload/underload 集合可重叠**，循环终止性无保证；复杂度分析漏了外层循环 | 🟡 中 |
| L10 | **负载度量 $U_d$ 把算力与显存等权相加**，与论文自己的 motivation 3（两者画像相反）冲突 | 🟡 中 |
| L11 | **§4.3 的多目标建模是装饰性的**：权重未给、量纲不齐、算法不求解它、未做模型校验 | 🟡 中 |
| L12 | **未开源**，无硬件描述，超参全部缺失 → 不可复现 | 🟠 高 |
| L13 | **实验自由度**：模型 2 个但都是 13B、数据集 2 个、baseline 2 个、无 trace 回放、无生产负载。3.9× 出现在 2×2=4 个格子中的 1 个 | 🟠 高 |

---

## 7. 与相关工作的定位：「反对派」的位置

### 7.1 路线之争的全景

领域里存在一条清晰的分野，两边回答的是同一个问题：**KV Cache 在集群里是"钉在某台机器上"还是"可以流动"？**

| 路线 | 核心主张 | 代表系统 | 路由器要不要看 cache |
| --- | --- | --- | --- |
| **A. 让请求跟着缓存走** | KV 留在 HBM 里最值钱，把请求送到有缓存的机器上 | Preble、DualMap、SGLang Router、Meta CacheRoute、Dynamo KV Router | ✅ 要 |
| **B. 让缓存跟着请求走** | 缓存位置不该约束调度，把 KV 做成可搬运的全局资源 | Mooncake、MemServe、LMCache、**BanaServe** | ❌ 不要（BanaServe 最极端） |

BanaServe 是 B 路线里**立场最激进的一个**：Mooncake 虽然有全局 KV 池，但它的 Conductor **仍然做缓存亲和调度**（按块 hash 定位副本、热块复制、冷块换出），只是用 `kvcache_balancing_threshold` 这个手工旋钮在亲和与均衡之间取舍。**BanaServe 是唯一主张把 cache 信号从路由器里彻底删掉的。**

### 7.2 对比表

| 系统 | 亲和 vs 负载的处理 | 搬 KV？ | 搬权重？ | Baseline 里有强亲和路由吗 | 实验规模 |
| --- | --- | --- | --- | --- | --- |
| **Preble**（ICLR'25） | E2 算法混合亲和与负载；热前缀**自动复制**到多实例 | 复制而非迁移 | ❌ | — | 多 GPU |
| **DualMap** | 双映射：亲和 + 负载**同时**约束；实测纯亲和 CV=1.24 / P90 TTFT 20s | ❌ | ❌ | ✅（含纯亲和消融） | 中 |
| **GORGO** | 三信号加权（网络 + 队列 + 前缀）在线调参 | ❌ | ❌ | ✅（prefix-cache / session-affinity） | 小（3 副本） |
| **CacheRoute**（Meta） | **规划式**前缀亲和 + per-key 准入；纯入口决策，**不搬 KV** | ❌ | ❌ | ✅（**对比 Preble**：176±11 vs 76±11 QPS） | **60× H100** |
| **Mooncake**（FAST'25 最佳论文） | KVCache-centric 全局池 + **仍然做亲和调度**，阈值调和 | ✅ | ❌ | ✅ | 生产集群 |
| **BanaServe** | **删除亲和信号**，路由器只看负载 | ✅（Global Store） | ✅（层迁移） | ❌ **没有** | 未披露（13B） |

**最后一列是这张表的重点**：唯一一篇主张"废除亲和"的论文，是唯一一篇没有把亲和路由放进 baseline 的论文。

### 7.3 ⚠️ 它攻击的是「纯亲和」还是「一切亲和」？——稻草人分析

这是理解 BanaServe 在综述中位置的核心问题。

**第一步：它的前提部分成立。** "纯粹按缓存命中率路由会造成负载倾斜"——这个命题是**真的**，而且有两篇独立论文的实测支撑：

- **DualMap**：纯 Cache Affinity 策略下实例负载 **CV = 1.24**，**P90 TTFT 达 20 秒**
- **GORGO**：纯 prefix-cache 策略在饱和时崩溃，**p95 TTFT 6784 ms**

讽刺的是，这两组数据**比 BanaServe 自己的 Fig.2a 强得多**——BanaServe 想论证的东西，被它的"对立面"用实测证明了。

**第二步：但没有人在主张「纯亲和」。** A 路线的每一个严肃系统都已经内建了负载均衡机制：

| 系统 | 它怎么防倾斜 |
| --- | --- |
| Preble | E2 算法把负载写进代价函数；**热前缀自动复制**到多个实例，从根上消除"只有一台有缓存"的单点 |
| DualMap | 双映射结构，亲和与负载是**同时**的硬约束，不是先亲和后修正 |
| CacheRoute | **规划式**分配 + **per-key 准入控制**，主动限制单 key 能吸引的流量 |
| SGLang Router | cache-aware 分数里带 balance 项与阈值 |
| Mooncake | `kvcache_balancing_threshold` 显式在亲和与均衡间取舍 |
| Dynamo KV Router | overlap score 与负载 cost 加权 |

**第三步：BanaServe 攻击的对象是自己构造的。** Fig.2a 的正文明说这个路由器是 "based on **both** cache hit rate **and current load**"——即一个**混合**路由器——然后不加解释地断言它会退化成 100%/40%/60% 的倾斜。**一个已经把负载写进目标的路由器，只有在负载权重设得极低时才会这样。** 论文没有说明这个假想路由器的具体策略，也没有实现它、测量它，更没有拿任何真实的混合路由器来验证。

**结论**：BanaServe **在措辞上攻击的是「一切亲和路由」**（Abstract、Contribution 1、§3.4 Summary 都是无限定的 "prefix cache aware routing **inherently** skews load distribution"），**而它实际能成立的只有「纯亲和会倾斜」**——一个 A 路线自己早就承认并已解决的问题。**这是一个稻草人。**

**第四步：真正的分歧被绕过了。** A 路线和 B 路线的**真实**争论点不是"要不要防倾斜"（双方都要），而是：

> **当前缀缓存必须跨机器共享时，是该「复制缓存」（Preble：热前缀多副本）、「规划分配」（CacheRoute：入口侧限流），还是「集中存储 + 每次取数」（BanaServe：Global Store）？**

三者代价结构完全不同：复制耗 HBM，规划耗调度复杂度与命中率，集中存储耗 PCIe/网络带宽与每请求取数延迟。**这是一个可以做实验回答的问题，而 BanaServe 没有做这个实验。**

### 7.4 那么，什么时候 B 路线（BanaServe）真的更优？

论文没回答，我们推一下这条路线的成立边界：

**✅ 有利条件**
1. **前缀复用高度分散**：大量中等热度的前缀、没有明显热点 → 亲和路由本来也难以命中，复制成本高，集中存储的均摊代价更低。
2. **单前缀热度极端集中**：一个超热前缀被所有请求共享 → 亲和路由必然把流量压到少数几台（除非复制），此时共享存储 + 均衡路由确实更好。（注意 1 和 2 是两个**极端**都有利，中间地带才是亲和路由的主场。）
3. **互连带宽极其充裕**：NVLink / 400G IB 全互联，且 Store 完全驻留 DRAM 不落 SSD。
4. **长 prompt + 短 output**：prefill 计算时间长，足够掩盖 KV 预取；反之短 prompt（Alpaca 那种 4–50 token）根本没有可供重叠的计算窗口——**这恰恰是论文报出 3.9× 的那个配置**，很值得怀疑。
5. **PD 资源比例波动剧烈**：这是 BanaServe 迁移机制的真实价值所在，与路由争论无关。

**❌ 不利条件**
1. **跨区/跨机房**（作者自己承认）——带宽稀缺时 HBM 亲和的价值最大化。
2. **高并发**：Store 带宽被并发度稀释，51× 余量迅速归零。
3. **Store 落到 SSD 层**：带宽掉 1–2 个数量级。
4. **高度倾斜但可复制的前缀分布**：Preble 式热前缀复制的边际成本极低，收益极高，B 路线毫无优势。
5. **小集群**：全局存储的固定开销分摊不掉。

**综合判断**：BanaServe 的立场在**单机房、DRAM 全驻留、长 prompt、PD 比例剧烈波动**的窄带里可能成立；论文没有刻画这条边界，也没有在边界外测试过。

---

## 8. 工程借鉴清单

抛开立场争论，这篇论文里**确实可以拿走**的东西：

| # | 可借鉴点 | 说明 |
| --- | --- | --- |
| 1 | **PD 资源画像的实测**（Fig.2b） | prefill 算力 >95% / 显存 ~35%，decode 算力 ~35% / 显存高，通信是单向 prefill→decode。这是本文最扎实的一条数据，可直接用于容量规划与 PD 配比设计 |
| 2 | **层级 KV 预取三级流水** | prefetch(L+1) / compute(L) / store(L−1) 三通道并行，是任何"KV 从慢存储取"的系统都该做的标准手法（LMCache 也是这个思路）。落地时要**按并发度重新算带宽**，不要照抄 51× |
| 3 | **每 token KV 大小的估算模板** | $S_{kv} = h_{kv} \cdot d_{head} \cdot 2 \cdot \text{bytes}$，llama-3.1-8B = 4 KB/层，128 KB/token。做 KV 池容量规划时可直接套 |
| 4 | **注意力头级卸载的思路** | 沿 head 维切分 KV 到冷 GPU 是真正的细粒度均衡手段（尽管本文公式写错了）。正确实现应参考 Ring Attention / FlashAttention 的分块 softmax rescale |
| 5 | **迁移的 Benefit/Cost 门限** | 只在 $\textit{Benefit}/\textit{Cost} \ge \rho$ 时执行迁移 + 双阈值迟滞 $(\delta^\uparrow, \delta^\downarrow)$ 防抖——任何带迁移的均衡器都需要这两个保护 |
| 6 | **反面教训：负载标量化** | 不要把算力利用率与显存利用率等权相加成一个标量（公式 32）。PD 分离下这两维的语义正相反，应保留为向量或按阶段分别设阈值 |
| 7 | **反面教训：把稳态开销当一次性开销** | 头级切分之后的 per-token per-layer 跨设备同步，不能用 $S_{kv}/B_{net}$ 建模。任何"切开算子跨设备"的设计都要先算**每步**的通信，不是**一次**的搬迁 |

---

## 9. 个人评述

### 亮点

- **它提出了一个正确的问题**：把资源分配与状态放置解耦，是 PD 分离架构真正的结构性痛点。Fig.2b 的资源画像实测是扎实的。
- **层级权重迁移在这条赛道上是相对少见的动作**。多数系统只搬 KV（Mooncake、MemServe、Llumnix），BanaServe 连模型权重都搬，等于把"动态模型并行"做成了运行时原语。这个想法本身有价值（BlitzScale 是同方向），只是本文没有把它的代价算清楚。
- **它把路线之争摆到了台面上**。综述需要一篇明确站在"废除亲和信号"一侧的论文来定义争论的边界，BanaServe 提供了这个坐标——即使它自己没有守住这个坐标。

### 需要留意的

- **立场与证据严重不匹配**。核心主张（前缀亲和必然倾斜）的全部证据是一张手画示意图，且这张图被 Figure 总标题错误地标为 "empirically validated"。实验里没有任何前缀亲和 baseline，没有引用 Preble，没有测量任何倾斜指标。**这个反对立场从未被检验过。**
- **它反对的是稻草人**。措辞攻击"一切亲和路由"，实际只能证伪"纯亲和路由"——而后者的问题，DualMap 和 GORGO 已经用更好的实测证明过，且 A 路线的解决方案是"亲和 + 负载混合"而非"放弃亲和"。
- **迁移代价被系统性低估**：层迁移的权重传输（13B 单层约 650 MB）零测量；头级迁移的 per-token 稳态通信被误建模为一次性搬迁；Global Store 的 51× 带宽余量建立在单请求独占 200 Gbps 且不落 SSD 的假设上。
- **结果图存在无法忽视的内部矛盾**：20 RPS 输入下吞吐 0.4 req/s、512 token 输出的"平均延迟"1–17 ms、承诺的 95% CI 未出现。这些数字**不应直接引入综述**。
- **不可复现**：无代码、无硬件配置、无超参、无 ablation。而致谢里提到阿里提供了"large-scale experimental platform"，报告的却是 13B 模型的小规模实验。
- **期刊状态需谨慎表述**：PDF 是 SPE 模板但所有元数据都是占位符，arXiv v1 无接收信息。宜写作"SPE 格式投稿，接收状态未证实"。

### 一句话定位

**BanaServe 是本综述里唯一明确主张「路由器应彻底不看缓存」的论文，因此在结构上不可替代；但它对这一主张的论证是一张示意图，实验里没有任何前缀亲和对照组，攻击的又是一个已经被 A 路线解决了的「纯亲和」稻草人——所以它的真实贡献不是「反对前缀亲和路由」，而是「PD 分离下的细粒度模块迁移」，而后者的代价它自己也没算清。引用它时应引其立场与 Fig.2b 的资源画像，不宜引其性能数字。**

---

## 附：交叉引用

- [Mooncake-论文解读.md](./Mooncake-论文解读.md) —— **同路线但立场温和得多**。同为 KVCache-centric 全局池（FAST'25 最佳论文），但 Conductor **仍然做缓存亲和调度**：按块 hash 去重复用、热块复制、冷块换出，用 `kvcache_balancing_threshold`（手工调）在亲和与均衡之间取舍。Mooncake 证明了"有全局 KV 池"**不蕴含**"要放弃亲和信号"——这正是 BanaServe 多迈的那一步，也是它唯一没有证据的那一步。BanaServe 引用了 Mooncake（ref 30）但未将其作为 baseline。

- [CacheRoute-论文解读.md](./CacheRoute-论文解读.md) —— **BanaServe 的直接对立面**。Meta 的 CacheRoute 是纯入口侧决策、**完全不搬移 KV**，靠规划式前缀亲和 + per-key 准入控制来同时拿到命中率与均衡，在 **60× H100** 上做到 **176±11 QPS**（Preble 为 76±11）。它用生产规模的实验证明了"亲和路由 + 主动准入"这条路可以扩展——恰好是 BanaServe 断言不可行的那条路。两篇的实验规模差距（60 卡 vs 未披露的 13B 实验）也决定了可信度差距。

- [DualMap-论文解读.md](./DualMap-论文解读.md) —— **同时支持又反驳 BanaServe**。DualMap 的实测（纯 Cache Affinity 策略负载 **CV=1.24**、**P90 TTFT 20 秒**）是"纯亲和会倾斜"这一命题最强的证据，比 BanaServe 自己的 Fig.2a 强得多。但 DualMap 的应对是**双映射把亲和与负载做成同时约束**，而不是删掉亲和信号。它的存在直接说明：BanaServe 的前提对、结论跳步。

- [GORGO-论文解读.md](./GORGO-论文解读.md) —— 另一份独立实测：纯 prefix-cache 策略在饱和时崩溃（**p95 TTFT 6784 ms**）。GORGO 的路线是把网络、队列、前缀三个信号加权并**在线调权重**——同样是"混合"而非"废除"。它还提供了一个 BanaServe 缺失的东西：明确刻画了自己结论成立的负载边界（长上下文 + 高用户内复用）。

- [Preble-论文解读.md](./Preble-论文解读.md) —— **BanaServe 应该对比却完全没有引用的论文**。ICLR 2025，前缀亲和路由的代表作，E2 算法把负载写进代价函数，并用**热前缀自动复制**从根上消除"只有一台机器有缓存"的单点。BanaServe 的 Fig.2a 所描绘的倾斜场景，在 Preble 的设计里是被显式处理过的。缺失这个引用与这个 baseline，是本文最大的方法论漏洞。

- [LLM服务负载均衡与调度-文献综述.md](./LLM服务负载均衡与调度-文献综述.md) —— 整体脉络。BanaServe 在其中的位置：「让缓存跟着请求走」路线的最激进端点，也是全综述**唯一一篇公开反对前缀亲和路由**的论文。建议在综述中以"立场坐标"而非"性能证据"的角色引用。
