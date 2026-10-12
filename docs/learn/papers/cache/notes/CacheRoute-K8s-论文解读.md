# CacheRoute (K8s 版): KV-Cache-Aware Routing for LLM Backend Services on Kubernetes —— 论文解读

> ⚠️ **阅读前的重要提示（三条，请先读完再看正文）**
>
> 1. **同名不同文**。本仓库里已有一篇 `CacheRoute-论文解读.md`，解读的是 Meta 的 *CacheRoute: Planned Prefix-Affinity Routing*（arXiv:2608.19677，60×H100 规模）。**本文是完全不同的另一篇论文**，只是恰好重名。两者作者、机构、方法、规模都没有关系。
> 2. **这是一篇 6 页的短会议论文**（IEEE 双栏格式，无明确 venue 标注），作者为 CMU / Northeastern / UCSD 的学生。测试床是 **4 节点 A100 集群、LLaMA-2-7B**，负载 2–16 req/s。与 Meta 版的 60 卡 H100 / Llama-3.3-70B 不在一个量级。
> 3. **论文的 §II "Methodology Foundation" 存在明显的引用注水**。这一节整整 1.5 页，引用了 [16]–[34] 共 19 篇文献，其中绝大多数与 LLM 路由**没有实质关系**：包括会计科目异常检测的图建模 [27]、多语言 RAG 的 LLM-as-a-Judge 校准 [29]、金融场景的联邦学习 [32]、拜占庭容错去中心化联邦学习 [33]、细粒度情感分类 [26]。这些引用以极其抽象的措辞（"provide the conceptual basis for…"、"Under this methodological view…"）串联，是典型的**引文互引环（citation ring）**特征。**这一节可以直接跳过，它不影响论文的技术内容，但它是判断这篇论文可信度时必须知道的事实。**
>
> 我仍然写这篇解读，因为它的**技术设计本身是干净且有参考价值的**：它是我读到的唯一一篇把 KV-cache 感知路由**明确落到 Kubernetes Gateway API Inference Extension 标准接口上**的论文，工程落地路径最清晰。但它的实验数字应当被当作"小规模 sanity check"而非"可外推的性能结论"。

---

## 1. 一句话概括

把 prefix 相似度、GPU 显存利用率、预测 TTFT 三个信号做**线性加权打分**，实现成一个符合 Kubernetes Gateway API Inference Extension 规范的 **Endpoint Picker（EPP）**，在 4 节点 A100 上让 p95 TTFT 相比 Round Robin 降低 43%（8 req/s）到 54%（16 req/s），KV cache 命中率从 13.6% 提到 64.5%。

论文自己对贡献的定位其实很克制（§V "Relation to Existing Work"）：

> "The Gateway API **already supports** prefix-cache-aware and latency-based routing. CacheRoute's contribution is a **controlled ablation quantifying the marginal value of combining all three signals**."

也就是说：**作者承认三个信号本身都不是新的，新的是"把三个一起用"，以及把每个信号的边际贡献量化出来**。这个自我定位是诚实的，也是这篇论文真正有价值的地方——它回答了一个工程师真实会问的问题：*我已经开了 prefix-aware 插件了，再加 latency 预测还有多少收益？* 答案是 **中负载 8–11%，高负载 18–20% 的 p95 TTFT**。

---

## 2. 问题：为什么标准 K8s 负载均衡对 LLM 不够用

论文在 §I 给了三条理由，这三条基本是这个研究方向的公共前提，但组织得很清楚：

| # | 性质 | 对负载均衡的影响 |
|---|------|------------------|
| 1 | 请求是 **prefill（算力密集）+ decode（显存带宽密集）** 两阶段流水线 [Orca, OSDI'22] | 一个"请求"不是一个原子的等价工作单元，RR/LR 的"请求数"计数没有物理意义 |
| 2 | **KV cache 随序列长度线性增长**，主导 GPU 显存 [PagedAttention, SOSP'23] | 显存是硬约束；被驱逐的 cache 必须**整段重算** |
| 3 | **多轮对话与 RAG 频繁复用 context 前缀** [SGLang] | 存在跨请求的复用机会，但标准 LB 对此完全无感知 |

然后指出既有工作的盲区：Sarathi-Serve（chunked prefill）、FlashAttention/-2、CacheGen（KV 压缩）、CacheBlend（跨请求 KV 融合）**都在单实例内部优化**，不解决**实例间的路由决策**。

这个"单机优化 vs. 实例间路由"的切分是干净的，也解释了为什么这条研究线（Preble / DualMap / SkyWalker / Meta CacheRoute）是独立于引擎优化的一层。

### 2.1 与本仓库其他几篇的关系

这是本仓库唯一一篇**站在 Kubernetes 生态标准之上**的论文，值得单独说明这个语境：

- **Gateway API Inference Extension**（Kubernetes SIG Network，2025）定义了两个抽象：`InferencePool`（一组同模型的推理 Pod）和 **Endpoint Picker（EPP）**（决定把请求发给池中哪个 Pod 的可插拔组件）。
- Gateway 通过 **ext-proc**（Envoy External Processing gRPC 协议）把端点选择**外包**给 EPP。
- KServe 和 llm-d 都构建在这个框架上。
- 官方已提供 **prefix-cache-aware 插件** [14] 和 **latency-based routing** [15]。

所以 CacheRoute(K8s) 实质上是在说：**官方那两个插件应该合起来用，而且再加一个 TTFT 预测，我来量化一下值不值。**

这与 Preble / DualMap / SkyWalker 的路子不同——那三篇都是**自建调度器**，而这篇是**填标准接口的槽**。对于"我明天就要在生产 K8s 上落地"的读者，这篇的路径最短。

---

## 3. 设计

### 3.1 系统架构

```
                  ┌─────────────────────────┐
   请求 ─────────▶│  K8s Inference Gateway  │  (Envoy)
                  │       (InferencePool)   │
                  └───────────┬─────────────┘
                              │ ext-proc (gRPC)
                              ▼
                  ┌─────────────────────────────────────┐
                  │   CacheRoute Endpoint Picker (Go)   │
                  │  ┌───────────────────────────────┐  │
                  │  │ Prefix Similarity Scorer  S() │  │
                  │  │ GPU Load Signal           U() │  │
                  │  │ TTFT Predictor            L() │  │
                  │  └──────────────┬────────────────┘  │
                  │      Composite Scoring  Φ(r, Iⱼ)    │
                  └──────────────┬──────────────────────┘
                                 │ argmax
        ┌────────────────┬───────┴────────┬────────────────┐
        ▼                ▼                ▼                ▼
   ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
   │ vLLM #1 │      │ vLLM #2 │      │ vLLM #3 │      │ vLLM #4 │
   │ +sidecar│      │ +sidecar│      │ +sidecar│      │ +sidecar│
   └────┬────┘      └────┬────┘      └────┬────┘      └────┬────┘
        └────────────────┴────────────────┴────────────────┘
                     每 500ms 上报（Prometheus 兼容端点）:
                     cached prefix hashes / KV 利用率 / queue depth
```

关键点：**sidecar 只读不写**。论文明确说明（§III-C）：

> "This hook does not alter scheduling, KV-cache allocation, or model execution kernels."

sidecar 周期性（500ms）读取 vLLM **block manager 的 hash table**，导出 cached block hashes 和 KV-cache 利用率。**不改 vLLM 任何一行调度/分配/kernel 代码**。这是这篇论文在工程上最讨喜的一点——不 fork 引擎，升级 vLLM 不会打架。

对比一下：Preble 需要深度改 SGLang 的调度器；DualMap 需要 vLLM 侧配合做 prefix 上报和迁移；Meta 的 CacheRoute 需要全局重规划。CacheRoute(K8s) 的侵入性是这几篇里最低的。

### 3.2 复合打分函数（核心，Eq.1）

设实例集合 $\mathcal{I} = \{I_1, I_2, \dots, I_n\}$，请求 $r$ 的 tokenized prefix 为 $p_r$：

$$
\Phi(r, I_j) = \alpha \cdot S(r, I_j) + \beta \cdot \bigl(1 - U(I_j)\bigr) + \gamma \cdot L(r, I_j)
$$

其中：

- $S(r, I_j) \in [0,1]$：归一化的 **prefix cache 相似度**
- $U(I_j) \in [0,1]$：归一化的 **GPU 显存利用率**（注意取 $1-U$，即"越空越好"）
- $L(r, I_j) \in [0,1]$：归一化的 **预测 TTFT 的倒数**（越快越好）
- $\alpha, \beta, \gamma \ge 0$，且 $\alpha + \beta + \gamma = 1$

选中的实例 $I^* = \arg\max_{I_j \in \mathcal{I}} \Phi(r, I_j)$。

论文用 grid search（在 $\{0.1, 0.2, \dots, 0.6\}$ 上）定出 **$\alpha = 0.5,\ \beta = 0.3,\ \gamma = 0.2$**。

> **和 GORGO 对照着看**：GORGO 用 (1+1)-ES 在线学权重，结果学出 $w_{\text{queue}} = 0$ 并 reward hack 了目标函数。这里用离线 grid search 定死权重，虽然笨，但**不会被 hack**——因为人在环里看了整体指标。这是个有意思的对照：**自动调参在单指标目标下有风险，手工 grid search 在多指标人工判断下反而稳**。

### 3.3 三个信号各自怎么算

#### (a) Prefix Cache 相似度（Eq.2）

$$
S(r, I_j) = \frac{\mathrm{LongestPrefixMatch}\bigl(p_r,\ \mathrm{Cache}(I_j)\bigr)}{|p_r|}
$$

实现细节（这几条是真正可复制的部分）：

| 项 | 取值 | 备注 |
|----|------|------|
| 匹配粒度 | **16 token / block** | 对齐 vLLM 的 `block_size=16` |
| 哈希方式 | **rolling hash over token-ID 序列** | 不是文本哈希，是 token ID 哈希 |
| 哈希冲突后果 | **冗余 prefill，不是错误结果** | 关键的正确性论证：路由错了只是白算，不会算错 |
| 驱逐同步延迟 | **≤ 1 个轮询周期（500 ms）** | sidecar 在下一轮删掉对应 hash |

"哈希冲突只导致冗余 prefill，不导致错误结果"这句话值得单独强调——它说明**这个组件是 fail-soft 的**。路由器的 cache 视图可以过期、可以错，最坏后果是性能退化，不会有正确性问题。这在生产环境是非常重要的性质，也是为什么 500ms 的陈旧视图可以接受。

#### (b) GPU 负载信号

$U(I_j)$ = **已分配 KV cache blocks / 总 blocks**。

论文对这个信号的设计意图讲得很明确：

> "Since $\alpha > \beta$, the cache reward **dominates** the load penalty for moderately loaded instances ($U < 0.85$). The penalty becomes significant only near memory saturation ($U > 0.85$), where further routing would trigger evictions."

翻译成人话：**$\beta$ 项不是用来做负载均衡的，是用来做"悬崖保护"的**。在 $U < 0.85$ 的正常区间，cache 亲和性说了算；只有逼近显存饱和、再路由进来就会触发驱逐时，负载项才开始起决定作用。

这是一个和 Preble / DualMap 都不同的哲学：
- **Preble**：负载和 cache 是**同一个 cost 函数里的可比项**，持续权衡
- **DualMap**：用双哈希在**结构上**限制 cache 亲和的自由度，再用 TTFT 触发迁移
- **CacheRoute(K8s)**：默认**完全偏向 cache**，负载只在**逼近悬崖时**踩刹车

第三种在低到中负载下是最优的（cache 命中最大化），但在**高负载下理论上更脆弱**——因为 $U$ 是显存占用而不是排队长度，一个实例可能显存没满但队列已经很长。这正是他们需要第三个信号（TTFT 预测）的原因。

#### (c) TTFT 预测（Eq.3, Eq.4）

线性模型：

$$
\hat{T}(r, I_j) = a \cdot |p_r| \cdot \bigl(1 - S(r, I_j)\bigr) + b \cdot Q(I_j) + c
$$

- $|p_r|$：输入 token 数
- $\bigl(1 - S(r, I_j)\bigr)$：**未命中比例** → 所以 $|p_r| \cdot (1-S)$ 就是**实际需要 prefill 的 token 数**
- $Q(I_j)$：队列深度
- $a, b, c$：最小二乘回归标定，标定窗口见 §4.1

归一化：

$$
L(r, I_j) = \mathrm{clip}\left(1 - \frac{\hat{T}(r, I_j)}{T_{\max}},\ 0,\ 1\right)
$$

$T_{\max}$ 取标定窗口的 **p99 TTFT**。

**这个模型的物理形式其实和 GORGO 的 Eq.1 是同构的**：

| | CacheRoute(K8s) Eq.3 | GORGO Eq.1 |
|---|---|---|
| prefill 项 | $a \cdot \lvert p_r \rvert \cdot (1-S)$ | $T_{\text{prefill}}(x_r \setminus c_i)$ |
| 排队项 | $b \cdot Q(I_j)$ | $T_{\text{queue}}(i)$ |
| 网络项 | **无** | $T_{\text{network}}(i)$ |
| 系数来源 | 离线最小二乘 | 在线 (1+1)-ES |

差别只有两处：CacheRoute(K8s) 是**单区域部署所以没有网络项**，且系数是**离线拟合**的。这个对照说明这两篇独立的工作收敛到了同一个 TTFT 分解形式——这大概率不是巧合，而是这个分解确实是对的。

**打分开销：每请求 < 0.3 ms**（实测均值 0.27 ms，p99 0.41 ms）。

### 3.4 实现

| 项 | 值 |
|----|-----|
| EPP 实现 | **Go，gRPC ext-proc server** |
| sidecar 轮询周期 | 500 ms |
| sidecar 导出方式 | Prometheus 兼容端点 |
| sidecar 资源占用 | **< 50 MB 内存，0.02 CPU core / pod** |
| 决策延迟 | 均值 0.27 ms，p99 0.41 ms |
| 复杂度 | $O(n)$ per request（$n$ = 实例数） |

sidecar 的资源占用（50MB / 0.02 core）低到可以忽略，这个数字对于说服 SRE 接受"每个推理 Pod 加一个 sidecar"很有用。

---

## 4. 实验设置

### 4.1 测试床

| 项 | 配置 |
|----|------|
| 集群 | **4 节点，每节点 1× NVIDIA A100 (40 GB)** |
| 引擎 | vLLM v0.6，**automatic prefix caching 开启** |
| 模型 | LLaMA-2-7B-Chat (`meta-llama/Llama-2-7b-chat-hf`)，BF16 |
| Gateway | Envoy-based Inference Gateway |
| `gpu_memory_utilization` | 0.90 |
| `max_num_batched_tokens` | 4096 |
| `max_num_seqs` | 64 |
| `max_model_len` | 4096 |
| `block_size` | 16 |
| 权重 | $\alpha{=}0.5,\ \beta{=}0.3,\ \gamma{=}0.2$ |

**实验流程**（这一段的严谨性是这篇论文的加分项）：

1. **5 分钟预热**
2. 预热的**最后 2 分钟**作为 **TTFT 标定窗口**（拟合 $a,b,c$ 和 $T_{\max}$）—— **明确排除在评测之外**
3. 接下来 **10 分钟**为评测窗口
4. **每组实验重复 3 次**，报告均值，**run 间差异 < 5%**

标定窗口和评测窗口分离、重复 3 次并报告方差范围，这两点做得比 GORGO（单次 30 分钟窗口）规范。

### 4.2 负载

三种 profile，全部从 **LMSYS-Chat-1M** 构造：

| Workload | 描述 | 输入 tokens | 输出 tokens | 混合占比 |
|----------|------|-------------|-------------|----------|
| **Short Q&A** | 单轮 | 50–200 | 50–150 | 40% |
| **Multi-turn Chat** | 3–8 轮，追加历史 | 200–2048 | — | 35% |
| **RAG-like Long-context** | LMSYS query 前置 2–4 个 512-token 合成块 | 1024–4096 | — | 25% |

到达过程：**Poisson，2–16 req/s**。采样参数 temperature 0.7 / top-p 0.9。

> ⚠️ **RAG 的合成块是这个实验设计里最弱的一环，作者自己也在 §V 承认了**："The RAG-like workload uses **synthetic** chunks, so absolute hit rates may differ from production RAG services." 合成的 512-token 块之间的复用模式（是否共享、共享比例）完全由构造方式决定，而这直接决定了 RAG 工况的 cache 命中率上限。

### 4.3 五个 baseline

| 缩写 | 全称 | 说明 |
|------|------|------|
| **RR** | Round Robin | 轮转 |
| **LR** | Least Request | 活跃请求数最少 |
| **SA** | Session Affinity | 同 session 固定实例 |
| **CA** | Cache Affinity | 最长 prefix 匹配，**无负载/TTFT 信号** |
| **PL** | Prefix + Load | 双信号（$\alpha_{PL}{=}0.6,\ \beta_{PL}{=}0.4$），**无 TTFT 预测**。作者说明这**近似于既有的 prefix-aware EPP 插件** |

**PL 这个 baseline 是整篇论文的关键**。因为 K8s Gateway API 官方已经有 prefix-aware 插件了，读者真正想知道的不是"比 RR 好多少"（那是必然的），而是**"比我现在已经能用的东西好多少"**。PL 就是这个"现在已经能用的东西"的代理。

---

## 5. 主要结果

### 5.1 8 req/s 混合负载全指标（Table I）

| 指标 | RR | LR | SA | CA | PL | **CR** |
|------|-----|-----|-----|-----|-----|--------|
| **p95 TTFT (ms)** | 420 | 380 | 340 | 295 | 270 | **240** |
| **p99 TTFT (ms)** | 645 | 570 | 495 | 405 | 365 | **315** |
| TPOT (ms/tok) | 28.5 | 27.8 | 28.1 | 27.4 | 27.2 | **26.9** |
| Throughput (tok/s) | 385 | 405 | 420 | 465 | 478 | **498** |
| **Cache Hit (%)** | 13.6 | 16.7 | 39.2 | 52.0 | 57.9 | **64.5** |
| Reused Prefix (%) | 5.1 | 6.8 | 24.5 | 37.2 | 43.6 | **51.8** |
| **GPU Mem (%)** | 82.4 | 79.6 | 76.3 | 74.1 | 72.5 | **69.5** |

这张表读下来最值得注意的是**指标之间的一致性排序**：RR < LR < SA < CA < PL < CR，**七个指标全部同序，没有一次交叉**。

这既是好消息也是需要警惕的信号：
- 好消息：说明"加信号确实单调改善"，方法论上自洽
- 需要警惕：**真实系统的多指标之间几乎总有 trade-off**。GORGO 的 reward hacking 案例就是典型——TTFT 全线最优但 E2E p95 差 4.8×。这里七个指标完美同序、无一例外，要么说明这个负载还远未到饱和点（trade-off 尚未显现），要么说明数据经过了某种平滑。结合 4 节点 A100 + 8 req/s 的规模看，**前者更可能**：这个负载点系统压力不大，各策略都在舒适区，所以只体现"cache 命中越高越好"这一个单调关系。

### 5.2 尾延迟随负载的变化

| 负载 | CR p95 TTFT | RR p95 TTFT | 相对 RR | 相对 CA | 相对 PL |
|------|-------------|-------------|---------|---------|---------|
| **8 req/s** | 240 ms | 420 ms | **−43%** | −19% | **−11%** |
| **16 req/s** | 1120 ms | 2450 ms | **−54%** | — | **−20%** |

**关键观察：负载越高，CacheRoute 相对 PL 的优势越大（11% → 20%）**。作者的解释：

> "TTFT prediction provides value beyond prefix-plus-load balancing by **avoiding routing long-context requests to instances with deep queues even when those instances have favorable cache state**."

这正好印证了 §3.3(b) 我提到的那个结构性弱点：$U$ 是显存占用，不是队列长度。一个实例可以显存宽裕但队列很深——PL 看不到这一点，TTFT 预测里的 $b \cdot Q(I_j)$ 项能看到。**所以第三个信号真正补的洞是"显存 ≠ 排队"。** 这是本文最实质的技术发现，也是它相对官方 prefix-aware 插件唯一站得住的增量。

### 5.3 分工况 cache 命中率

| Workload | CacheRoute 命中率 |
|----------|-------------------|
| Short Q&A | 56% |
| **Multi-turn Chat** | **76%** |
| RAG-like Long-context | 62% |
| **加权平均** | **64.5%** |

相比 CA 提升 **12.5 个百分点**，作者归因为"**避开了 cache 丰富但显存饱和的实例**"（cache-rich but memory-saturated）。

这个归因是自洽的：CA 只看 cache，会把请求都堆到"最热"的那个实例上，结果那个实例显存饱和→驱逐→cache 反而变差。**纯 cache 亲和会自我毁灭 cache**——这是 Preble / DualMap / SkyWalker 共同发现的现象，这里又一次被独立验证。

多轮对话 76% 明显高于其他两类，符合直觉（多轮天然共享全部历史前缀）。

### 5.4 吞吐-延迟权衡与显存平滑度

- **500 ms SLO 下**：CR 498 tok/s，PL 478 tok/s，RR 385 tok/s
- **10 req/s 显存曲线**：RR 和 LR **逼近 90% 显存预算**（提高驱逐概率）；CR **稳定在 60% 左右**
- **显存利用率标准差：CR 3.8% vs. RR 10.2%**

显存标准差从 10.2% 降到 3.8% 这个数字比平均值更有意义——它说明 CacheRoute 不只是降低了平均显存，更重要的是**消除了显存尖峰**。而显存尖峰正是驱逐的触发器，驱逐又会毁掉 cache。所以这是一个正反馈：稳定显存 → 少驱逐 → 高命中 → 少 prefill → 更稳定显存。

### 5.5 消融与灵敏度（§IV-E）

论文给了一组很干净的**退化关系**，这是理解这个打分函数的最好方式：

| 参数设置 | 退化为 |
|----------|--------|
| $\alpha = 1$ | **Cache Affinity (CA)** |
| $\beta = 1$ | **Least Request (LR)** |
| $\gamma = 0$，$\alpha + \beta = 1$ | **Prefix + Load (PL)** |

也就是说 **CA、LR、PL 全都是 CacheRoute 打分函数的特例**。这个设计很优雅——所有 baseline 都在同一个参数空间里，消融就是在这个空间里移动，避免了"不同 baseline 实现质量不同"导致的不公平比较。

灵敏度结论：最佳区间 $\alpha \in [0.4, 0.6]$，$\beta \in [0.2, 0.4]$。

**TTFT 项的边际贡献**（本文最核心的量化结论）：

| 负载 | 加入 $\gamma = 0.2$ 后 p95 TTFT 降低 |
|------|--------------------------------------|
| 中负载 | **8–11%** |
| 高负载 | **18–20%** |

**开销**：ext-proc 决策均值 0.27 ms（p99 0.41 ms）；sidecar < 50 MB + 0.02 core / pod。

---

## 6. 局限（作者自陈 + 我的补充）

### 6.1 作者自陈的四条（§V Limitations）

1. **只能做 token 级哈希匹配，无法感知语义相似**。16-token block 粒度的哈希比较，语义相同但 token 不同的两个 prompt 完全不共享。
2. **线性 TTFT 预测器（Eq.3）无法刻画极端负载下的非线性排队效应**。排队论上 $T_{\text{queue}}$ 在逼近饱和时是超线性发散的，线性模型必然低估。
3. **单模型（LLaMA-2-7B）+ 同构 A100**，异构部署需要扩展打分函数。
4. **RAG 工况用合成 chunk**，绝对命中率与生产 RAG 服务可能有差异。

### 6.2 我补充的几条

5. **规模极小，外推性存疑**。4 节点 / 7B 模型 / 8 req/s。$O(n)$ 打分在 $n{=}4$ 时当然不是问题（论文自己在 §V Scalability 里承认了，并建议"两阶段：先按 cache 亲和过滤，再对 top candidates 打分"），但**真正的问题不是打分开销，而是结论是否成立**。Meta 的 CacheRoute 论文在 60 卡规模上发现的很多现象（如 94.5% cache churn on replan）在 4 卡上根本不会出现。

6. **七个指标完美同序，缺少 trade-off 的证据**。见 §5.1。一个方法在所有指标上全面胜出、且负载扫描中没有任何交叉点，通常意味着实验点选在了系统的舒适区。**建议读者关注：这篇论文没有报告任何一个 CacheRoute 输掉的场景。** 对比 SkyWalker（§6.3 承认均匀 ToT 下 CH 赢 2%）和 GORGO（§Appendix D 自曝 reward hacking），本文缺少这种自我否证。

7. **权重是离线 grid search 定死的，没有讨论负载漂移**。$\alpha{=}0.5,\beta{=}0.3,\gamma{=}0.2$ 是在特定混合比（40/35/25）下搜出来的。生产环境的工况混合比会随时间漂移，论文没有讨论权重是否需要重新标定、多久标定一次。（作者在 Future Work 里提到了 "adaptive weight tuning"，即承认这是未解决的。）

8. **TTFT 标定窗口只有 2 分钟**，而且是在预热期。$T_{\max}$ 取自这 2 分钟的 p99，如果评测期负载分布偏移，归一化的分母就不对了。这一点作者没有讨论。

9. **§II 的引用注水严重损害论文可信度**（见开头提示）。虽然它不影响 §III–§V 的技术内容，但它是一个很强的负面信号，会让人有理由怀疑其他部分（比如实验数据）是否同样经过了修饰。**我的建议是：把 §III 的设计当作有价值的工程参考，把 §IV 的数字当作"存在但需要独立复现"的弱证据。**

10. **没有与本领域主要工作做实验对比**。既没有对比 Preble、也没有对比 SGLang 的 cache-aware router、更没有对比它明确宣称要超越的官方 EPP 插件的**实际实现**（PL 只是"近似"）。§V 承认了这一点："Future work includes… comparison with latency-based EPP configurations."——**即论文实际上并未与官方 latency-based routing 做过对比**。

---

## 7. 与本仓库其他论文的定位

| | 部署形态 | 核心机制 | 规模 | 侵入性 | 是否自曝负面结果 |
|---|---|---|---|---|---|
| **Preble** | 自建调度器 | $E^2$ 算法 + 三部分负载代价 | 8–32 GPU | 高（改 SGLang 调度器） | 部分 |
| **DualMap** | 自建路由 | 双哈希 + TTFT 触发迁移 | 8 NPU | 中（需引擎配合上报） | 否 |
| **SkyWalker** | 两层跨区路由 | 区域快照前缀树 + 选择性推送 | 12× L4 跨三洲 | 中（SkyServe 之上） | **是**（§6.3 CH 赢 2%） |
| **GORGO** | 代理层在线调参 | (1+1)-ES 学三信号权重 | **3 副本** | 低 | **是**（Appendix D reward hacking） |
| **Meta CacheRoute** | 全局重规划 | 计划式 prefix 亲和 + LPT | **60× H100** | 高 | **是**（94.5% churn） |
| **CacheRoute (K8s)** | **K8s 标准 EPP** | 三信号线性加权 | **4× A100** | **最低（sidecar 只读）** | **否** |

**这篇的独特位置**：它是唯一一篇**不需要改引擎、直接填 K8s 标准接口槽**的方案。如果你的场景是"生产 K8s 集群，已经在用 Gateway API Inference Extension，想加 cache 感知"，那么：

- **不要从 Preble / DualMap / Meta CacheRoute 起步**（都要改引擎或重建调度层）
- **直接用官方 prefix-aware 插件**，然后**参考本文的 Eq.3 加一个 TTFT 预测项**
- 预期收益：相比纯 prefix+load，中负载 8–11% / 高负载 18–20% 的 p95 TTFT

### 7.1 与 GORGO 的直接对照（很有意思）

两篇论文的 TTFT 模型形式几乎一致（见 §3.3(c) 的对照表），但**权重的确定方式截然相反**：

| | CacheRoute (K8s) | GORGO |
|---|---|---|
| 权重确定 | 离线 grid search，人工看多指标 | 在线 (1+1)-ES，单指标（TTFT）优化 |
| 结果 | $\alpha{=}0.5,\beta{=}0.3,\gamma{=}0.2$，无异常 | $w_{\text{queue}}{=}0$，**100% 流量打到最近副本** |
| 后果 | 七指标一致改善 | TTFT 全线最优，**E2E p95 恶化 4.8×** |

**这是一个几乎教科书式的对照实验**（虽然是两篇独立论文无意造成的）：**同一个模型形式，自动单指标优化会被 hack，人工多指标 grid search 不会。** 根本原因不是"grid search 更聪明"，而是**人在环里看的是多个指标，目标函数隐式是多维的**。

工程结论：如果你要自动调这三个权重，**目标函数必须包含 E2E 延迟或 ITL，不能只有 TTFT**。

---

## 8. 工程落地清单

如果要在生产 K8s 上实现这套东西，按优先级：

1. **先确认你的 Gateway 支持 ext-proc**（Envoy-based Inference Gateway 支持）。这是整个方案的前提。
2. **sidecar 只读 vLLM block manager 的 hash table，绝不改引擎**。这是这个方案能长期维护的关键——vLLM 升级不会破坏你的路由层。
3. **prefix 匹配粒度对齐 `block_size`（vLLM 默认 16）**。不对齐会导致匹配到的 block 实际不可复用。
4. **用 rolling hash over token-ID，不要用文本哈希**。tokenizer 之后才是引擎真正 cache 的单位。
5. **接受哈希冲突**。冲突的后果是冗余 prefill 而非错误结果，为此加复杂的冲突检测不值得。
6. **轮询周期 500ms 是个合理起点**。cache 视图陈旧 500ms 造成的损失，远小于更高频轮询对引擎的干扰。
7. **负载信号用"已分配 KV blocks / 总 blocks"，并把它当作悬崖保护而非均衡器**（$U > 0.85$ 才真正起作用）。如果你希望它做真正的负载均衡，需要另外引入队列长度信号。
8. **TTFT 预测的 prefill 项必须乘 $(1 - S)$**，即只算未命中部分。这是把 cache 状态耦合进延迟预测的关键一步，也是这个线性模型能 work 的原因。
9. **标定窗口与评测/生产窗口分离**，并周期性重新标定 $a, b, c, T_{\max}$。论文用的是 2 分钟窗口，生产建议更长且滚动更新。
10. **权重不要用单指标自动优化**（见 §7.1）。要么人工 grid search 看多指标，要么目标函数里必须包含 E2E/ITL。
11. **实例数大时改两阶段**：先按 cache 亲和过滤出 top-k，再对 k 个候选算完整打分。$O(n)$ 在 $n$ 上百时会成为问题。
12. **监控 GPU 显存利用率的标准差，不只是均值**。标准差是驱逐风险的先行指标（本文：3.8% vs RR 的 10.2%）。

---

## 9. 个人评述

### 亮点

- **唯一一篇讲清楚"怎么在 K8s 标准接口上落地"的论文**。ext-proc / EPP / InferencePool / sidecar 的组合是真实可部署的，侵入性最低。这个价值不来自算法新颖性，来自**工程路径的具体性**。
- **消融设计很优雅**：CA、LR、PL 都是打分函数的参数特例，所有比较在同一个参数空间内进行，避免了实现质量差异带来的不公平。
- **诚实的自我定位**：明说"Gateway API already supports…，我们的贡献是量化三信号组合的边际价值"。没有夸大新颖性。
- **找到了一个真实的技术洞见**：$\beta$ 项用的是显存占用，看不到队列深度；高负载下"cache 好但队列深"的实例会被 PL 错选。这是 TTFT 预测项真正在补的洞，也解释了为什么它的收益随负载增长（11% → 20%）。

### 需要留意的

- **§II 的引用注水是硬伤**（19 篇引用中大部分与主题无关，涉及会计异常检测、联邦学习、情感分类等）。这不影响技术内容的可读性，但严重影响可信度判断。
- **规模太小**（4× A100 / 7B / 8 req/s），几乎所有大规模系统才会暴露的问题（cache churn、调度器吞吐瓶颈、异构、故障）都没触及。
- **没有任何负面结果**。七个指标全面且单调胜出，缺少 trade-off 的证据。对比同仓库的 SkyWalker 和 GORGO 都主动报告了自己输的场景，本文的实验呈现显得过于干净。
- **未与官方 EPP 实际实现对比**（PL 只是"近似"），而这恰恰是它最该对比的东西。作者在 Future Work 里承认了。

### 一句话定位

**技术设计干净、工程路径最清晰的一篇小论文，但实验规模和学术规范性都远不足以支撑它的数字——把它当作一份"K8s 上做 cache 感知路由的实现指南"来读，而不是一份性能结论。**

---

## 附录：与本仓库其他解读的交叉引用

- **`CacheRoute-论文解读.md`**（Meta 版，同名不同文）—— 60× H100 规模、计划式 prefix 亲和、94.5% cache churn 现象。**读完本文后强烈建议对照读**，可以直观看到规模差两个数量级后问题形态如何变化。
- **`Preble-论文解读.md`** —— $E^2$ 算法。本文的 $\Phi$ 打分与 Preble 的 $\text{Cost}_i = L_i + M_i + P_i$ 是同一类"多信号加权"思路，但 Preble 的三项都是**时间量纲**可直接相加，本文的三项是**归一化到 [0,1] 后加权**，物理意义更弱。
- **`DualMap-论文解读.md`** —— 双哈希。DualMap 在**结构上**限制 cache 亲和的自由度（只给两个候选），本文在**数值上**通过 $\beta$ 项做悬崖保护。两种思路对"纯 cache 亲和会自我毁灭"这个共同发现给出了不同的解法。
- **`SkyWalker-论文解读.md`** —— 跨区域。本文是单区域单集群，两者场景正交，可组合。
- **`GORGO-论文解读.md`** —— 见 §7.1 的重点对照。**这两篇一起读的价值大于分开读**。
- **`LLM服务负载均衡与调度-文献综述.md`** —— §4.1 有本文的二手概述。
