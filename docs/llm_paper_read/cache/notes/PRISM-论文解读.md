# PRISM: Fast Online LLM Serving via Scheduling-Memory Co-design —— 论文解读

> 论文：*PRISM: Fast Online LLM Serving via Scheduling-Memory Co-design*
> 作者：Xingyu Qu、Tianhao Lin、Yiqi Li、Sheng Wang（武汉大学计算机学院）、Zhiyu Chen（Amazon）
> 版本：**Preprint**，arXiv:2605.08581v1（2026-05-09），未经同行评审，引用数 0
> 代码与数据：**论文中未给出任何开源链接**（未获取）
>
> ⚠️ **阅读提示**：全部实验跑在**单张 A800**上，作者自己在 Limitations 里承认了这一点。因此它不是一篇"分布式调度"论文，而是一篇**引擎内部**的论文。值得读的是它的核心机制主张——**让调度器把"下一批要发什么"这个信息直接喂给 KV cache 的淘汰键**——以及 §4.5 那个把 DART 单独摘出来的消融（PRISM 49.1% vs LRU+Active Counter 30.4% 命中率）。第 2 节的排队论推导（Theorem 1）看起来唬人，实际上是一行代数，不必花时间。

---

## 1. 一句话概括

在 RAG / Agent 这类"提示词由可复用 segment 拼装 + 热点严重倾斜"的负载下，**调度侧的重排序和 cache 侧的淘汰策略是两个互相不知情的模块**：调度器辛苦把共享前缀的请求排到一起，cache 却可能在它们被执行前就把那个热前缀淘汰了。PRISM 把这两件事接起来——**QAS**（Query-Aware Scheduler）按可复用 segment 给请求分桶排序，并把"本轮 dispatch batch 里每个 segment 的需求分数"作为提示传给 **DART**（Demand-Aware Radix Tree），DART 用这个分数直接做 radix 树的**淘汰键**，保护即将被用到的 anchor。单 A800 上相对最强 baseline，平均 per-QPS P99 TTFT 降 **23.3%（Qwen3-4B）/ 37.1%（Llama2-13B）**，精确前缀 KV 命中率提升 **5.9 / 12.2 个百分点**。

> **综述里给 PRISM 的那句话经核实成立**：23.3% / 37.1% 是"averaged over the QPS sweep, relative to the strongest baseline"的 P99 TTFT 降幅（论文 §4.3）。但要补两句限定：**这是单 GPU 上的数字**，且 baseline 里最强的是 ContextPilot（MLSys 2026），相对 native SGLang 的降幅要大得多（63.5% / 80.8%）——后者才是这个领域论文最爱引的那个数。

---

## 2. 它要解决的问题

### 2.1 两个各干各的优化流派

| 流派 | 代表 | 干什么 | 看不到什么 |
| --- | --- | --- | --- |
| KV cache 管理 | vLLM、SGLang、RAGCache、CacheBlend、TurboRAG、HotPrefix、KVFlow | 利用提示词结构提高命中率 | 不知道调度器**下一批要发什么** |
| 复用感知调度 | Orca、Sarathi-Serve、k-LPM、ContextPilot | 重排 admission，让共享前缀的请求在时间上靠近 | 无法告诉 backend **哪个 radix anchor 应该活下来** |

作者的核心论点（Figure 1 那张图）很简洁：调度器可以制造前缀局部性，**但在 KV 内存压力下，一个热前缀仍可能在下一个匹配请求到达之前被淘汰**。此时调度器所有的重排序努力全部作废——请求沿着正确的路径往下走，却在被淘汰的 anchor A 处断链，退化为 MISS。

由此推出三条设计要求：
1. **Segment 级管理**：跟踪可复用的上下文片段而非整条请求
2. **Query-aware 调度**：用可复用前缀信息重排，并**把这个信息传给 cache 后端**
3. **Demand-aware 保留**：内存紧张时保护"活跃的或即将被复用的"前缀

### 2.2 请求抽象

$$x_i = \left[\, x^{\text{sys}} \,\|\, x^{r_{i,1}} \,\|\, x^{r_{i,2}} \,\|\, \cdots \,\|\, x^{r_{i,m_i}} \,\|\, x^{\text{priv}}_i \,\right] \qquad (\text{Eq1})$$

系统提示 + 若干可复用 segment + 请求私有后缀。这个抽象覆盖 RAG（检索到的 passage）与 Agent（tool schema、policy 文本、确定性 tool 输出）。

### 2.3 TTFT 分解

$$\text{TTFT}_i = W^{\text{admit}}_i + T^{\text{prefill}}_i + T^{\text{1tok}}_i \qquad (\text{Eq2})$$

$$L^{\text{hit}}_i(t) = \max_{c \in \mathcal{C}_t} \text{LCP}(z_i, c), \qquad T^{\text{prefill}}_i \approx \frac{\ell(z_i) - L^{\text{hit}}_i(t)}{R_{\text{pf}}(B_t)} \qquad (\text{Eq3, Eq4})$$

有一句话值得划出来：

> *"Segment identities are only control-plane hints; KV hits still require exact serialized-prefix equality."*

**segment 身份只是控制面提示，真正的命中仍然要求 token 级精确前缀相等。** 这是 PRISM 与 KVShare（语义近似共享）、TurboRAG（离线预计算 KV chunk）路线的分界线——PRISM 刻意只做精确前缀，不引入语义假阳性。这个选择是对的，也是它能在 SGLang 原生 radix tree 上落地的原因。

### 2.4 内存侧的量级

论文给了一个有用的数量级：k=5 × 128-token chunk ≈ 640 token 的证据载荷，在 **Llama2-13B 几何 + FP16 KV** 下**仅这部分就约 0.5 GiB KV 内存**——还没算系统提示、用户后缀和并发 decode 状态。所以"把所有重复前缀都留在显存里"在单卡上根本不成立，必须做取舍。这是 DART 存在的物理理由。

### 2.5 排队论包装（Theorem 1）

论文定义 wave（一起发起 prefill 的请求集合），得到请求级服务率：

$$\mathbb{E}_\pi[S_{\text{wave}}] = \frac{L_\pi(1-h_\pi)}{R^{\text{eff}}_{\text{pf},\pi}}, \qquad \mu_\pi = \frac{\bar{M}_\pi R^{\text{eff}}_{\text{pf},\pi}}{L_\pi(1-h_\pi)} \qquad (\text{Eq5})$$

**Theorem 1**：在 $L$、$\bar{M}$、$R_{\text{pf}}$ 全部固定的受控比较下，若命中率差 $\Delta h := h_{\text{PRISM}} - h_{\text{LRU}} \geq 0$，则

$$\mu_{\text{PRISM}} - \mu_{\text{LRU}} = \mu_{\text{LRU}} \cdot \frac{\Delta h}{1 - h_{\text{PRISM}}} \geq 0 \qquad (\text{Eq7})$$

$$\Delta\lambda^\star = \frac{\bar{M} R_{\text{pf}}}{L} \cdot \frac{\Delta h}{(1 - h_{\text{LRU}} - \Delta h)(1 - h_{\text{LRU}})} \qquad (\text{Eq8})$$

> **我的评价：这个"定理"就是 $\frac{1}{1-a-\Delta} - \frac{1}{1-a}$ 的通分**（附录 E.1 的证明是三行）。它说的是"命中率越高服务率越高"，这是一个把所有其他因素都固定住之后的恒等式，**不含任何关于 PRISM 为什么能提高命中率的信息**。真正有信息量的是分母 $(1-h)^2$ 这个形式：**在高命中率区，同样的 $\Delta h$ 带来的服务率增益被放大**。在 $h \approx 0.49$ 时，1pp 的命中率提升约等于 3.8% 的服务率提升——这解释了为什么 5.9pp 的命中率差能变成 23.3% 的 P99 TTFT 差（在拐点附近排队因子再放大一次）。这个机制解释是站得住的，但把它包装成 Theorem 有点用力过猛。

附录 E 里更有用的其实是两处**作者主动交代的坑**：
- **wave-to-request 转换的量纲错误**（E.1 末）：把 $1/\mathbb{E}[S_{\text{wave}}]$ 当作请求级服务率会**低估可持续 $\lambda^\star$ 达 $\bar{M}$ 倍**（4B trace 在 60 QPS 时 $\bar{M} \approx 33.1$，读成 waves/s 就只有 1.5 而不是 49.8）。这是一个很实在的建模陷阱。
- **common-attenuation 何时失效**（E.4）：列了三种会让 decode 竞争变成 policy-dependent 的情形——输出长度受策略影响、chunked-prefill chunk size 与 decode 协同调参（Sarathi 式）、decode 受带宽限而 prefill 受算力限。作者明确说自己的短输出 RAG 负载下第三种可忽略。

---

## 3. 核心设计

### 3.1 QAS：Query-Aware Scheduler

QAS 在长度为 $\Delta t_{\text{win}}$ 的挂起窗口上工作，维护三个 segment 计数器：

| 计数器 | 定义 | 含义 |
| --- | --- | --- |
| $g_t(r)$ | $\|\{q \in \mathcal{Q}^{\text{queue}} : r \in \mathcal{R}(q)\}\|$ | **global**：排队集合里含该 segment 的请求数 |
| $a_t(r)$ | $\|\{q \in \mathcal{Q}^{\text{active}}_t : r \in \mathcal{R}(q)\}\|$ | **active**：后端在跑的请求里含该 segment 的数量 |
| $n_t(r;\mathcal{B})$ | $\|\{q \in \mathcal{B} : r \in \mathcal{R}(q)\}\|$ | **next**：某个候选 batch 内的复用数 |

统一优先级分数：

$$P_t(r; \mathcal{B}) = w_g\, g_t(r) + w_a\, a_t(r) + w_n\, n_t(r;\mathcal{B}) \qquad (\text{Eq9})$$

**权重取 $w_a = 10^6$、$w_n = 10^5$、$w_g = 1$**，全部实验固定不变。作者说得很直白：这组权重的作用是**强制一个严格的序数层级**——活跃需求 > 未来批次复用 > 排队普遍度。

> 这其实不是"加权求和"，而是用数值幅度模拟的**字典序比较**。这样写比写三级排序简单，但也意味着 $10^6$ 这个数隐含假设了 $g_t(r) < 10^5$（排队里含同一 segment 的请求不超过 10 万条）。在实验规模（N=2048 replay）下当然安全，但这是个**没有被讨论的隐式上界**，工程化时需要显式化。

**桶签名生成**：把优先级 top-$f_{\text{front}}$ 的 segment 移到骨架前部得到 $\Pi^\star_i$，然后

$$\sigma_i = \text{Top}^{\rightarrow}_\kappa(\Pi^\star_i; P_t(\cdot;\emptyset)), \qquad b_i = \text{hash}(\sigma_i) \qquad (\text{Eq10})$$

$\text{Top}^{\rightarrow}_\kappa$ 挑高优先级 segment 但**按服务时序写回**——所以两个请求只有在其主导 segment 以**相同序列化顺序**出现时才落到同一个桶。全部实验用 $f_{\text{front}} = 3$、$\kappa = 1$。

**桶打分**：

$$U(b) = \frac{1}{\max(1, |\mathcal{R}_b|)} \sum_{r \in \mathcal{R}_b} P_t(r; \mathcal{B}^{\text{hot}}_b), \qquad \text{Score}(b) = \alpha_{\text{size}} |\mathcal{T}[b]| + \beta_{\text{util}} U(b) \qquad (\text{Eq11, Eq12})$$

$\alpha_{\text{size}} = 1$、$\beta_{\text{util}} = 0.5$，同样全局固定。

**热/冷双车道**：每个 batch 分成 hot lane（按 $\text{Score}(b)$ 降序扫桶填充，最多 $M - q_{\text{cold}}$ 个槽）和 cold lane（$q_{\text{cold}}$ 个槽，FIFO 收最老的未选中请求，防饿死）。默认 $q_{\text{cold}} = 2$。

### 3.2 DART：Demand-Aware Radix Tree

DART **不改 radix 树本身**，只在 token 边界（系统提示后、每个可复用 segment 后、私有后缀后）加一层 anchor 元数据。若导出的 offset 落在某条 radix 边内部，DART 会**分裂节点**使 anchor 落在精确 token 边界上。

QAS 敲定 $\mathcal{B}^{\text{dispatch}}_t$ 后，DART 实例化 dispatch-batch 优先级 $\widehat{P}_t(r) := P_t(r; \mathcal{B}^{\text{dispatch}}_t)$，并取 top-$f$ 的常驻 anchor 组成保护集 $\mathcal{F}_t$。淘汰键是一个**字典序三元组**：

$$\text{EvictKey}(u) = \begin{cases}
\big(0,\ \text{last}(u)\big) & u \text{ 是私有后缀节点} \\
\big(1,\ \widehat{P}_t(\text{seg}(u)),\ \text{last}(u)\big) & u \text{ 是未保护的可复用节点} \\
\big(2,\ \text{last}(u)\big) & u \in \mathcal{F}_t\text{，受保护的可复用 anchor}
\end{cases} \qquad (\text{Eq13})$$

小键先淘汰。所以顺序是：**冷私有后缀 → 未保护可复用节点（按 dispatch 优先级升序）→ 受保护 anchor**。

这是全文最核心、也是最容易被低估的一行：**它把"调度器下一批要发什么"直接编码进了 cache 的淘汰序**。相比之下 LRU 只看 `last(u)`、LFU 只看历史频次，两者都无法回答"这个前缀 50ms 后会不会被用到"。

**实现细节**（附录 B.1，值得注意）：
- 计数器在挂起窗口**冻结之后**才计算，用当前调度状态而非到达时的陈旧快照
- $n_b(r)$ 是"已物化的挂起请求上的一次性反事实"，**明确不是对未来到达的预测**，且在扫描多个桶时不重算
- DART 因 radix 分裂/摘除会产生陈旧堆项，采取**惰性校验**：pop 时重算 key，不一致就重新压回

### 3.3 复杂度

设 $S = \sum_{q_i \in \mathcal{Q}^{\text{queue}}} m_i$ 为一轮内排队 segment 出现次数，$V$ 为可淘汰 radix 节点数：

| 组件 | 开销 |
| --- | --- |
| QAS 计数/分桶/候选打分 | 期望 $O(S)$ |
| QAS 前置对齐 | $O(S \log f_{\text{front}})$ |
| QAS 有序签名 | $O(S \log \kappa)$ |
| DART 每次候选淘汰 | $O(\log V)$ |

论文强调 PRISM **避免了全对语义搜索、离线 KV 预计算、per-token 控制逻辑**。这个论证是可信的——控制面只在 segment ID 和 anchor 元数据上工作。**但论文没有给任何实测的控制面开销数字**（没有 CPU 占用、没有调度延迟分解），只有渐进复杂度。这是一个缺口。

---

## 4. 实验方法

### 4.1 硬件与实现

| 项 | 配置 |
| --- | --- |
| GPU | **单张 NVIDIA A800 80GB PCIe** |
| CPU | 14 vCPU Intel Xeon Gold 6348 @ 2.60GHz |
| CUDA | 13.0 |
| 引擎 | **SGLang**，保留原生 continuous batching 与 chunked prefill |
| 模型 | Qwen3-4B-Instruct-2507、Llama2-13B |

作者原话："With limited compute resources, we focus on the single-GPU setting."

### 4.2 Baseline

| Baseline | 出处 | 定位 |
| --- | --- | --- |
| **native SGLang** | NeurIPS 2024 | 引擎级基线：continuous batching + chunked prefill + radix 前缀复用 |
| **k-LPM** | Dexter et al., arXiv:2502.04677 (2025) | 前缀复用调度，与 QAS 最接近的调度侧对照。取 $k = 2$（"best performance"） |
| **ContextPilot** | Jiang et al., MLSys 2026 | 上下文复用服务系统，作者称之为 SOTA |

所有方法共享同一 engine binary、模型、到达过程、并发上限、内存配置，**每次运行从空 KV cache 开始**。这一点做得规范。

### 4.3 负载构造

**RAG 主 trace**（`zipf_hotspot`）：

| 参数 | 值 |
| --- | --- |
| 来源 | MultiHopRAG (Tang & Yang, 2024) |
| 唯一证据 passage 数 | 981 |
| 热点集大小 $\|\mathcal{R}_{\text{hot}}\|$ | 20 |
| 热请求率 $r_{\text{hot}}$ | 0.7 |
| Zipf 指数 $\alpha$ | 1.2 |
| 回放长度 $N$ | **2,048** 请求 |
| 检索深度 | $k = 5$，标称 128-token chunk |
| 到达过程 | Poisson（目标 QPS） |

**关键的设计分离**：到达时间戳来自 Poisson 过程，热点文档来自 Zipf 分布——**offered load 与 segment 流行度被解耦**。这是一个干净的实验设计。

实际物化的提示长度（附录 D.1，很重要的披露）：

| 配置 | mean / p50 / p90 / p99 / max（prompt tokens） |
| --- | --- |
| 4B, k=5 | 768.1 / 770 / 791 / 818 / 879 |
| 4B, k=7 | 1029.5 / 1036 / 1059 / 1083 / 1147 |
| 4B, k=10 | 1429.6 / 1438 / 1461 / 1486 / 1548 |
| 13B validation | 371.5 / 370 / 417.5 / 457 / 485（可复用前缀 295.3，私有后缀 76.1） |

**Agent trace**（AGENTPREFIX）：从 τ-bench (Yao et al., ICLR 2025) 构造，保留序列化的 agent 提示（系统/policy 文本、tool 定义、任务记录、历史轮次、确定性 tool 输出）。2,048 请求，Poisson 到达 30/40/50 QPS，**32-token decode 预算**，最大 prefill 长度 40,960 token。作者明确说明："should be read as a serving benchmark for agent prompts rather than as an end-to-end measurement of task success"。

### 4.4 指标

- **TTFT** P50/P90/P95/P99，重点看 P99
- **精确前缀 KV cache 命中率**（backend 侧统计）
- **答案 F1**——作为正确性护栏（RAG 场景）

**测量语义的坦白交代**（附录 D.2）：4B 的 phase trace 里，TTFT = 前端 pre-admission + **admission-to-first-token 包络**。后者包含后端入队、GPU 服务等待、chunked-prefill 执行、首 token 返回路径，因此它是 $T^{\text{prefill}}_i + T^{\text{1tok}}_i$ 的**上包络而非纯 prefill 计时器**。13B 的老 request-level trace 才是更细的验证源。这个披露是加分项，但也意味着**两个模型的 TTFT 数字不完全同源**。

---

## 5. 主要结果

### 5.1 命中率与答案质量（Table 1）

| 模型 | QPS | PRISM 命中率 / F1 | ContextPilot | k-LPM | native SGLang |
| --- | --- | --- | --- | --- | --- |
| Qwen3-4B | 40 | **48.98** / 40.61 | 43.64 / 40.33 | 19.94 / 39.81 | 15.17 / 39.76 |
| | 50 | **48.90** / 40.33 | 43.51 / 39.85 | 16.93 / 39.81 | 15.07 / 39.81 |
| | 60 | **49.25** / 40.29 | 43.76 / 39.36 | 20.26 / 39.76 | 15.14 / 39.81 |
| | 70 | **49.10** / 40.44 | 42.04 / 39.48 | 20.39 / 39.76 | 14.98 / 39.53 |
| | 80 | **49.27** / 40.28 | 43.24 / 40.56 | 19.86 / 39.81 | 16.06 / 39.48 |
| Llama2-13B | 10 | **39.79** / 21.69 | 26.51 / 20.97 | 20.10 / 21.01 | 10.17 / 20.78 |
| | 15 | **39.02** / 21.58 | 27.01 / 20.98 | 20.14 / 20.45 | 10.07 / 20.88 |
| | 20 | **39.15** / 21.36 | 27.06 / 20.48 | 20.34 / 20.56 | 10.14 / 20.74 |
| | 25 | **39.22** / 21.79 | 28.14 / 20.67 | 19.89 / 20.68 | 9.98 / 20.96 |
| | 30 | **40.15** / 21.88 | 27.50 / 20.58 | 20.86 / 20.66 | 11.06 / 20.98 |

汇总：4B 约 **49.1% vs 43.2%**（+5.9pp），13B 约 **39.5% vs 27.2%**（+12.3pp）。F1 全程稳定，PRISM 甚至略高——说明 TTFT 收益来自服务侧复用而非生成质量下降。

**几个值得注意的点**：
- PRISM 与 ContextPilot 的命中率在 5 个 QPS 点上**几乎是常数**（48.90–49.27，方差 0.37pp）。这说明这两个系统在这个负载上已经逼近了**该负载结构决定的命中率天花板**——热点集只有 20 个 passage，Zipf α=1.2，可复用载荷占提示的比例是固定的。**命中率上限约 50%**（对照 $q_{\text{cold}}=0$ 时的 50.06%）。
- **native SGLang 的 15% 命中率非常低**。k=5 个 passage 的**排列顺序**决定了精确前缀能匹配多长；不做任何重排时，两个共享 3 个 passage 但顺序不同的请求几乎无法互相命中。PRISM 靠 $\text{Top}^{\rightarrow}_\kappa$ 强制统一顺序把这部分捞了回来——**这才是命中率提升的主要来源，而不是 DART 的保护**（见 §5.3 的辨析）。
- 13B 的提升（12.3pp）比 4B（5.9pp）大得多。原因是 13B 的 KV 内存压力大得多（同样 token 数占用约 3 倍显存），保留策略的边际价值更高。这一点论文没有明说，但从 §2.3 的 0.5 GiB 估算可以推出来。

### 5.2 P99 TTFT

论文只给了 Figure 4 的折线图，**主表没有 TTFT 的具体数值**。可提取的汇总：

| 对照 | Qwen3-4B | Llama2-13B |
| --- | --- | --- |
| vs 最强 baseline（ContextPilot） | **−23.3%** | **−37.1%** |
| vs native SGLang | −63.5% | −80.8% |

单独可读到的绝对数字只有 PRISM 自己的（附录 D.3 Table 7，$q_{\text{cold}}=2$ 主配置）：

| Target QPS | 吞吐 (req/s) | P99 TTFT (s) | Doc. 命中率 (%) | $\bar{M}$ | Extend/wave (tokens) | Admit→1tok mean (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 40 | 38.72 | 0.86 | 48.98 | 6.42 | 2.63k | 0.29 |
| 50 | 48.28 | 2.00 | 48.90 | 23.87 | 9.80k | 0.93 |
| 60 | 49.79 | **6.50** | 49.25 | 33.11 | 13.51k | 3.48 |
| 70 | 50.44 | 10.99 | 49.10 | 34.25 | 14.00k | 6.18 |
| 80 | 50.06 | 14.14 | 49.27 | 33.74 | 13.73k | 8.10 |

**服务拐点在 50–60 QPS 之间**：40/50 QPS 时吞吐紧跟 offered load、P99 TTFT < 2.01s；60–80 QPS 吞吐平台在 50 req/s，P99 TTFT 从 6.50s 涨到 14.14s。

> **这张表其实削弱了主结果的说服力**。5 个 QPS 采样点里有 3 个（60/70/80）处在**深度饱和区**——P99 TTFT 分别是 6.5s / 11.0s / 14.1s。在一个已经饱和到 P99 TTFT 超过 10 秒的系统上比较"谁的 P99 更低"，比的其实是队列积压的斜率，而不是服务质量。而"平均 per-QPS 降 23.3%"这个数把这三个饱和点和两个健康点**等权平均**了。如果只看 40 QPS 这个唯一真正 pre-knee 的点（P99 = 0.86s），从附录 C.1 的描述可以推知**差距会小很多**——那里明确写了 "at $r_{\text{hot}}$=0.5 and 40 QPS, ContextPilot is only 0.01 s lower in P99 TTFT"，**即在轻载下 ContextPilot 反而赢**。
>
> 换句话说：**PRISM 的收益几乎全部来自饱和区**。这在论文自己的机制解释里是自洽的（"near the service knee, the queueing factor amplifies each point of $\Delta h$"），但它意味着**在 SLO 内运行的系统上，PRISM 的价值远小于 23.3% 这个数字给人的印象**。

### 5.3 消融：DART 的贡献（§4.5，全文最有说服力的一节）

固定 QAS、负载、engine binary、内存配置，**只换后端保留规则**：LRU / LRU+Active Counter / LFU / DART。扫 λ ∈ {60, 70, 80}（KV 压力使保留策略真正起作用的区间）。

| | 命中率 | P99 TTFT 降幅（vs 最佳非 PRISM） |
| --- | --- | --- |
| PRISM (DART) | **49.10–49.27%** | — |
| LRU + Active Counter（最强非 PRISM 变体） | 30.41–30.56% | 60 QPS: **−37.6%** |
| LRU / LFU | 更低 | 70 QPS: −15.9% |
| | | 80 QPS: −25.4% |

命中率差 **18.54–18.86 个百分点**。

> **这是全文最干净的一个实验**，也是我认为 PRISM 的主要证据。它证明了一件具体的事：**在同样的调度决策下，仅仅把"下一批要发什么"喂给淘汰键，就能把命中率从 30% 拉到 49%**。注意 LRU+Active Counter 这个对照设计得很讲究——它已经用了 QAS 的 active 计数器，即它知道"现在正在跑什么"，但**不知道"下一批要发什么"**。18.5pp 的差距全部归因于这个 $n_t(r; \mathcal{B}^{\text{dispatch}}_t)$ 项。
>
> **但也要注意作者说的一句话**："We do not isolate QAS because without QAS hints, DART reduces to LRU." 这意味着**QAS 的独立贡献从未被测量**。上面 §5.1 我怀疑命中率提升的主因是 $\text{Top}^{\rightarrow}_\kappa$ 的顺序统一（QAS 侧），而这个消融**恰好无法回答这个问题**——因为所有四个变体都开着 QAS。要区分二者，正确的对照是"QAS + LRU"vs"原生调度 + DART-like 保留"，但后者按作者的说法不存在。**结果是：我们知道 QAS→DART 这条信息通路值 18.5pp，但不知道 QAS 本身相对 native SGLang 值多少。**

### 5.4 冷车道配额消融（Table 2 / Table 3）

这是论文里第二个值得看的实验，因为它展示了一个**反直觉的权衡**：

| QPS | $q_{\text{cold}}=0$ | $q_{\text{cold}}=2$ | $q_{\text{cold}}=4$ | $q_{\text{cold}}=8$ |
| --- | --- | --- | --- | --- |
| **P50/P90/P95/P99 TTFT (s)** | | | | |
| 40 | 0.37/0.79/0.94/1.47 | 0.32/0.55/0.64/0.86 | 0.35/0.59/0.67/**0.85** | 0.34/0.56/0.66/0.89 |
| 50 | 1.01/1.57/1.73/2.68 | 0.96/1.47/1.63/2.00 | 0.93/1.46/1.63/**1.97** | 1.03/1.63/1.78/2.03 |
| 60 | 3.09/6.16/6.43/7.60 | 2.94/5.96/6.24/**6.50** | 2.94/6.12/6.46/6.85 | 3.11/6.34/6.62/7.07 |
| 70 | 5.82/10.24/10.70/12.23 | 5.88/10.01/10.48/**10.99** | 6.12/10.39/10.78/11.29 | 5.95/10.29/10.72/11.26 |
| 80 | 7.57/12.86/14.53/16.21 | 7.84/13.04/13.62/**14.14** | 7.77/13.20/13.79/14.35 | 7.39/13.08/13.59/14.81 |
| **KV 命中率 (%)** | | | | |
| 40 | **49.87** | 48.98 | 49.01 | 48.89 |
| 60 | **50.07** | 49.25 | 49.12 | 49.05 |
| 80 | **50.30** | 49.27 | 49.04 | 49.30 |

**关掉冷车道（$q_{\text{cold}}=0$）命中率最高（平均 50.06% vs 49.10%），但平均 P99 TTFT 从 6.90s 涨到 8.04s**——因为稀有前缀的请求会一直排在热前缀桶后面挨饿。$q_{\text{cold}}=4$ 和 $8$ 相对 $q_{\text{cold}}=2$ 又分别把平均 P99 抬高 2.4% 和 4.6%。

> **"最大化复用不等于最优服务"这个结论比论文的主结果更有普适价值。** 1pp 的命中率换来 16.5% 的 P99 TTFT 恶化——这是一个非常陡的兑换率，而且方向和直觉相反。任何做前缀亲和路由/调度的系统都会遇到这条曲线：**缓存命中率是一个诱人但错误的优化目标**。这一点与 GORGO 那篇里 p95 TTFT 被 reward hack 的教训是同一类问题的两个侧面——**单指标目标函数在系统机制作为中介时会骗你**。

### 5.5 Agent 负载泛化（Table 4）

| QPS | 方法 | P50 | P90 | P95 | P99 | Hit | NG hit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | **PRISM** | 21.41 | **36.41** | **38.95** | **41.27** | **0.764** | **0.753** |
| 30 | ContextPilot | **21.39** | 39.76 | 42.50 | 58.88 | 0.742 | 0.679 |
| 30 | k-LPM | 21.60 | 41.79 | 43.90 | 54.96 | 0.740 | 0.677 |
| 30 | native SGLang | 57.53 | 105.06 | 111.37 | 131.66 | 0.730 | 0.642 |
| 40 | **PRISM** | 30.40 | **52.40** | **55.10** | **57.86** | **0.764** | **0.751** |
| 40 | ContextPilot | 30.28 | 55.42 | 59.56 | 65.70 | 0.741 | 0.678 |
| 40 | k-LPM | **30.27** | 55.76 | 59.73 | 94.09 | 0.741 | 0.678 |
| 40 | native SGLang | 65.53 | 120.19 | 127.47 | 140.71 | 0.731 | 0.645 |
| 50 | **PRISM** | **35.42** | **62.07** | **65.36** | **68.68** | **0.764** | **0.751** |
| 50 | ContextPilot | 36.98 | 65.19 | 69.68 | 75.07 | 0.741 | 0.676 |
| 50 | k-LPM | 36.36 | 64.64 | 69.12 | 91.01 | 0.742 | 0.678 |
| 50 | native SGLang | 72.05 | 130.46 | 138.21 | 150.86 | 0.731 | 0.646 |

P99 TTFT 相对最强 baseline 降 **24.9% / 11.9% / 8.5%**（30/40/50 QPS），相对 native SGLang 平均降 60.7%。NG hit（non-global，排除全局共享部分后的命中率）**0.75 vs 0.68** 是这里最有信息量的指标——它说明增益确实来自 agent 特有区域（tool schema、任务记录）而非公共系统提示。

> **但请看绝对数字：P50 TTFT 就是 21–35 秒，P99 高达 41–69 秒。** 这个"agent 服务基准"运行在一个**彻底崩溃**的工作点上——没有任何生产系统会在 P50 TTFT 21 秒的状态下运行。在这种深度过载下，所有百分位数本质上都是队列积压的读数。而且注意增益随 QPS **单调下降**（24.9% → 11.9% → 8.5%），即**越过载 PRISM 优势越小**。这与 RAG 主实验的叙事（"越接近拐点增益越大"）正好相反，论文没有解释这个矛盾。

### 5.6 敏感性（附录 C、F）

| 扫描维度 | 设置 | PRISM 相对最强非 PRISM baseline |
| --- | --- | --- |
| **热请求率** $r_{\text{hot}}$ | 0.5 / 0.3 / 0.1（50–80 QPS 区间） | P50 TTFT −10.4~16.9% / −5.7~20.6% / −4.2~8.2%；P99 TTFT 平均 −13.8% / −14.3% / −10.2%；命中率优势 4.72–8.45pp |
| **Zipf 指数** $\alpha$ | 0.6→2.0（固定 60 QPS，$r_{\text{hot}}$=0.5，**单种子**） | P50 −10.9~34.2%，P99 −9.8~30.0%；PRISM P99 从 8.82s（α=0.6）降到 5.12s（α=2.0）；命中率 47.1%→51.1%（ContextPilot 41.4%→46.9%），优势 4.20–6.51pp |
| **检索深度** top-$k$ | 7 / 10 / 15（固定 60 QPS） | 命中率 23.46% / 24.87% / 16.76%（ContextPilot 18.51% / 18.64% / 10.96%），优势 4.95–6.23pp；P99 TTFT 33.32s / 62.03s / **122.79s**，降幅 −13.8% / −9.3% / −8.1% |

三组敏感性实验的方向一致，说明结论不依赖某一个特定超参。**但三组的效应量都比主结果的 23.3% 小**（10.2%–14.3%、9.8%–30.0%、8.1%–13.8%），而且 top-$k$ 那组的绝对 P99 TTFT 已经到了 122 秒——这已经不是"服务"而是"批处理"。

**$\alpha$ 扫描明确标注了 "single-seed"**，其余实验**没有任何关于重复次数或方差的说明**。这是本文方法论上最大的一处空白。

---

## 6. 局限（作者自述 + 我的补充）

**作者自述（Conclusion 的 Limitations 段，只有两条）**：
1. PRISM 面向固定硬件与负载设定，把在线服务分析扩展为更一般的理论是 future work
2. **全部实验跑在单张 A800 上**，多 GPU KV 放置、互联影响、异构 GPU 池留待未来研究

附录 E 里额外交代的理论边界（这部分作者做得比正文诚实）：
3. Theorem 1 只在 $L$、$\bar{M}$、$R_{\text{pf}}$ 全固定的受控比较下成立
4. 不依赖任何形式化的方差收缩定理；关于 $c^2_{s,\pi}$ 的精确论断需要从 trace 计算 wave 级服务时间矩，未做
5. common-attenuation 假设在三种情形下失效（输出长度策略相关、chunk size 与 decode 协同调参、decode 受带宽限）

**我补充的几条**：

6. **QAS 的独立贡献从未被测量**（§5.3 已展开）。"没有 QAS 提示 DART 就退化为 LRU"这个说法解释了为什么不做，但不解释为什么不能做"QAS + LRU"这个对照——那是完全可行的，而且是判断两个组件相对重要性的必要实验。**我倾向于认为命中率提升的主因是 QAS 的 segment 顺序统一，而 DART 的保护是二阶效应**，但论文的实验设计恰好让这一点无法验证。

7. **主结果的 QPS 采样点严重偏向饱和区**（§5.2 已展开）。5 个点里 3 个 P99 TTFT > 6 秒，等权平均后得到 23.3%。而唯一的轻载对照（附录 C.1，$r_{\text{hot}}$=0.5 / 40 QPS）显示 **ContextPilot 反而领先 0.01s**。

8. **没有实测的控制面开销**。QAS 每个窗口要重算所有 segment 计数器、做前置对齐、分桶、给每个桶算候选 batch 的 $n_b(r)$；DART 每次淘汰要维护一个 lazy-validated min-heap。论文只给了渐进复杂度和一句"much cheaper than model prefill and decode execution"。在 4B 模型 + 50 QPS 这个吞吐下，调度器的 CPU 时间**完全可能成为瓶颈**（只有 14 vCPU），但没有任何数据。

9. **超参全部靠手工固定，没有调参过程的披露**。$w_a{=}10^6$、$w_n{=}10^5$、$w_g{=}1$、$f_{\text{front}}{=}3$、$\kappa{=}1$、$\alpha_{\text{size}}{=}1$、$\beta_{\text{util}}{=}0.5$、$q_{\text{cold}}{=}2$、保护预算 $f$（**从未给出取值**！）。其中只有 $q_{\text{cold}}$ 做了消融。$\kappa{=}1$ 尤其可疑——签名只取一个 segment，意味着分桶粒度极粗；论文给的理由是"larger κ fragments buckets and weakens locality"，但没有数据支持。**$f$ 的取值缺失是一个实打实的可复现性缺陷**，它直接决定了 DART 保护多少 anchor。

10. **k-LPM 这个 baseline 的调参不对称**。作者说 k-LPM 取 $k=2$"with the best performance"——即给 baseline 做了扫描并取最优；而 PRISM 自己的超参有 8 个以上且大多没扫。这在方向上是对 baseline 有利的（好），但同时也意味着**PRISM 的超参可能是在同一批实验上选出来的**，存在过拟合到这个 trace 的风险。没有留出集。

11. **负载是完全合成的**。MultiHopRAG 的 981 个 passage + 人工 Zipf(1.2) + Poisson 到达 + 20 个热点。作者自己说 "The Zipfian model therefore serves as a controlled stress test... rather than as a claim that production traces follow one exact parametric law"——这个自我限定是诚实的。但对照 RAGPulse（论文引用了，Wang et al. 2025，一个真实 RAG 负载 trace），**作者本可以在真实 trace 上验证却没有做**。

12. **回放长度只有 2,048 请求**。在 60 QPS 下这相当于**34 秒的流量**。Figure 3 的横轴确实就是 0–35 秒。在一个 34 秒的窗口里、从空 cache 开始、测 P99（即最差的 20 条请求），统计稳定性存疑。论文在正文里提到 "finite-replay stress regimes"——即高 QPS 下 trace 会提前放完，这本身就是一个会影响结果的实验假象。

13. **没有种子重复、没有误差棒**（除 $\alpha$ 扫描标了 single-seed 之外全文未提）。所有百分比降幅都是点估计。在 2,048 请求 / 34 秒的 trace 上，P99 是由约 20 条请求决定的，**23.3% 这个数字的不确定区间可能相当大**。

14. **无代码、无数据发布**。作为一篇声称在 SGLang 上做了非平凡引擎修改（radix 节点分裂、anchor 元数据层、lazy heap）的论文，不开源使得所有结果都无法独立验证。

15. **预印本，0 引用，未经同行评审**。ContextPilot（MLSys 2026）作为最强 baseline 本身也是一篇很新的工作，PRISM 对它的复现质量无法核查。

---

## 7. 与相关工作的定位：同一个耦合思想的两个层次

这是本文对本仓库主线最有价值的地方。**「准入控制 × 缓存保留」这个耦合思想，PRISM 和 Meta CacheRoute 分别在两个不同的层次上实现了它**。

### 7.1 层次对照

| 维度 | **入口路由层**（Meta CacheRoute） | **引擎层**（PRISM） |
| --- | --- | --- |
| 决策粒度 | 请求 → **副本** | 请求 → **本轮 dispatch batch** + radix 节点 → 淘汰序 |
| 耦合形式 | 前缀 $b$ 的副本数预算 $k_b$（$\sum_b k_b \le C$）+ per-key 准入 | segment 优先级 $P_t(r;\mathcal{B}^{\text{dispatch}}_t)$ 同时驱动分桶排序与 EvictKey |
| 时间尺度 | 周期重规划（分钟级） | 每个挂起窗口 $\Delta t_{\text{win}}$（毫秒–百毫秒级） |
| 缓存视图 | **间接、需要建模** —— 路由器不在引擎内，必须预测"这个前缀还在不在" | **直接、精确** —— DART 就是那棵 radix 树本身 |
| 规模 | **60×H100**，176±11 QPS vs Preble 76±11 | **1×A800**，约 50 req/s 平台 |
| 看不到什么 | 引擎内部的 batch 组成、淘汰时机、chunked prefill 状态 | 副本间负载分布、跨副本前缀放置、网络延迟、扩缩容 |

### 7.2 最关键的差异：缓存驻留是被预测还是被观测

CacheRoute 论文里最有价值的负面结果是：**它的解析式缓存驻留模型预测偏差中位 14.3pp、p90 44.7pp，最后放弃了解析预测，改用 shadow replay。** 原因很朴素——路由器在引擎外面，它不知道引擎内部什么时候会淘汰什么。

**PRISM 结构上不存在这个问题**：DART 就是引擎里的那棵树，$\text{last}(u)$、$\text{seg}(u)$、常驻 anchor 集合全部是**观测量而非预测量**。它甚至可以在淘汰发生的那一刻按当前 dispatch 优先级重算 EvictKey（附录 B.1 的 lazy validation 正是这么做的）。

代价是**视野只有一张卡**。PRISM 的 $P_t(r;\mathcal{B})$ 三个计数器全部来自本地队列和本地 active 集合，它对"隔壁那张卡上是不是已经有这个前缀"一无所知。

> **一句话概括这个对照**：CacheRoute 有全局视野但缓存状态是猜的；PRISM 缓存状态是准的但只看得见一张卡。**两者的失效模式因此完全互补**——CacheRoute 的失效是"路由到了一个前缀已被淘汰的副本"（预测偏差），PRISM 的失效是"保护了一个本地热但全局冷的前缀，同时隔壁卡在重复 prefill 同一个 segment"（视野盲区）。
>
> 更进一步：**这两层的耦合是可以叠加的，而且叠加时需要小心。** 如果 CacheRoute 式的路由层在上面做前缀亲和，PRISM 式的引擎层在下面做 demand-aware 保留，那么路由层的"这个副本应该缓存前缀 $b$"这个规划，与引擎层的"本轮 dispatch batch 里没有 $b$，淘汰它"这个决策**会直接冲突**。PRISM 的 $n_t$ 计数器只看已经物化的挂起请求（作者明确说"not a prediction of future arrivals"），它没有任何机制接收来自上层路由器的"我马上要往你这里发一批 $b$"的信号。**这是一个真实的、未被任何一篇论文处理的跨层接口空白。**

### 7.3 与其他系统的位置

| 系统 | 准入控制 | 缓存保留决策 | 二者耦合 | 层次 | 规模 |
| --- | --- | --- | --- | --- | --- |
| **CacheRoute (Meta)** | ✅ per-key 准入 | ⚠️ 通过副本数预算间接影响 | ✅ 规划式 | 路由层 | 60×H100 |
| **Mooncake** (FAST'25) | ✅ 基于 decoding slot 预测的早期拒绝（阈值手工） | ✅ KVCache-centric 全局池 | ⚠️ Conductor 内部隐式 | 全局调度器 | 生产级 |
| **Preble** (ICLR'25) | ❌ | ⚠️ 热前缀自动复制 | ❌ | 路由层 | 多副本 |
| **DualMap** | ❌ | ❌（双哈希影响放置） | ❌ | 路由层 | 8 卡 |
| **GORGO** | ❌ | ❌ | ❌ | 路由层 | 3 副本 |
| **k-LPM / ContextPilot** | ✅ 重排 admission | ❌ **不告诉 backend 保留谁** | ❌ | 引擎调度 | 单卡 |
| **HotPrefix / KVFlow / RAGCache** | ❌ | ✅ | ❌ **不知道调度器下一步** | 引擎 cache | 单卡 |
| **PRISM** | ✅ 分桶 + 热/冷双车道 | ✅ EvictKey 三元组 | ✅ **同一个 $P_t$ 分数** | **引擎层** | **1×A800** |

论文自己对 k-LPM / ContextPilot 的刻画很准确，值得引用：

> *"These methods primarily act on admission order: they can increase the chance that nearby requests share a prefix, but they do not explicitly tell the backend which radix-tree anchors should survive under KV-cache pressure. PRISM differs by making scheduler-side segment counters part of the cache-control plane."*

"把调度器的 segment 计数器变成 cache 控制面的一部分"——这句话是 PRISM 全文最好的一句自我定位。

### 7.4 与 Mooncake 的对照

Mooncake 是本仓库里另一个具备准入控制的系统，但它的准入是**基于 decoding slot 预测的早期拒绝，阈值手工调**。PRISM 的准入不是"拒绝"而是"排序"——它从不丢请求，只决定谁进 hot lane 谁进 cold lane。**两者的准入语义完全不同**：Mooncake 的准入是保护系统不过载（拒绝服务），PRISM 的准入是制造缓存局部性（重排服务）。混用"准入控制"这个词时需要注意这个区别。

有意思的是，$q_{\text{cold}}$ 消融揭示的"最大化命中率反而恶化 P99"这个现象，恰恰说明**纯重排式准入自带一个公平性债务**，必须显式还（FIFO 冷车道）。而 Mooncake 式的拒绝型准入没有这个问题——它不会让任何请求无限等待，它直接告诉你"不服务了"。

---

## 8. 工程借鉴清单

1. **把"下一批要发什么"作为淘汰键的第一维**。这是 PRISM 最可直接落地的一条。任何维护 radix / trie 前缀缓存的系统，只要调度器能给出下一批 dispatch 的组成，就可以把 EvictKey 从 `last(u)` 升级成 `(protection_class, dispatch_score, last(u))` 的字典序三元组。**消融显示这一项值 18.5pp 命中率**（在单卡、KV 压力大的前提下）。

2. **保护类别用字典序而非加权求和**。Eq.(13) 的三类（私有后缀 / 未保护可复用 / 受保护 anchor）是硬分层的，不是打分相加。这避免了"一个特别老的受保护 anchor 被一堆刚访问的私有后缀挤掉"这种数值意外。同理，QAS 用 $10^6 / 10^5 / 1$ 模拟字典序——**但更好的做法是直接写多级排序**，避免隐式的数量级上界假设（见 §6 第 9 条）。

3. **私有后缀应该被无条件优先淘汰**。这是一个免费的优化：请求特有的后缀几乎不可能被复用，却占据了与共享前缀同等的 KV 空间。任何前缀缓存都应该区分"共享段"与"私有段"并优先回收后者。PRISM 通过 anchor 的 `segment type` 字段做到这一点。

4. **⚠️ 缓存命中率是一个诱人但错误的优化目标**。$q_{\text{cold}}=0$ 拿到最高命中率（50.06%）却让平均 P99 TTFT 从 6.90s 恶化到 8.04s。**1pp 命中率 = 16.5% 的 P99 恶化。** 部署任何前缀亲和策略时，必须有一个显式的公平性/饥饿保护旋钮，且这个旋钮的最优值不在端点上（$q_{\text{cold}}=2$ 优于 0、4、8）。

5. **segment 顺序统一比 segment 选择更重要**。native SGLang 只有 15% 命中率，而同样的负载 PRISM 有 49%——差距的很大一部分来自 $\text{Top}^{\rightarrow}_\kappa$ 强制把高优先级 segment 按统一顺序排到前面。**在 RAG 场景下，检索结果的排列顺序是一个可以自由支配的自由度**（检索器返回的是集合，顺序对答案质量影响有限——论文的 F1 数据支持这一点），把它用于最大化前缀共享是几乎零成本的收益。这条对 Agent 场景同样适用（tool schema 的排列顺序）。

6. **控制面只在 segment ID 上工作，不碰 token**。QAS $O(S)$、DART 每次淘汰 $O(\log V)$。避免全对语义搜索、离线 KV 预计算、per-token 控制逻辑。这个设计原则让 PRISM 能作为 SGLang 的一个 patch 而非一套新系统存在。

7. **只做精确前缀，不做语义近似**。"Segment identities are only control-plane hints; KV hits still require exact serialized-prefix equality." 用 segment ID 做**调度决策**，用 token 级精确匹配做**正确性保证**——两者分离，避免了 KVShare 一类语义共享方案的假阳性风险。

8. **wave-to-request 的量纲转换**（附录 E.1）。把批量服务系统建模成 M/G/1 时，$1/\mathbb{E}[S_{\text{wave}}]$ 的单位是 wave/s 不是 req/s，两者差一个平均批大小 $\bar{M}$。本文 4B trace 在 60 QPS 时 $\bar{M} \approx 33$——搞错会把可持续吞吐低估 33 倍。这是一个容易犯的错。

9. **实验先扫服务拐点，报告结果时区分拐点前后**。本文 Table 7 给了一个规范的拐点标定（吞吐平台 + P99 爆炸的双指标交叉验证）。但它随后**把拐点前后的点等权平均**——这是应该避免的做法。正确做法是分别报告 pre-knee 和 post-knee 的效应量。

10. **披露实际物化的 token 长度而非标称预算**（附录 D.1）。"k=5 × 128-token chunk"是标称的 640 token，实际物化是 768.1 token（含模板与私有后缀）。用标称值代入排队模型会系统性低估服务需求。

---

## 9. 个人评述

### 亮点

- **"让调度器的 dispatch 意图进入 cache 淘汰键"这个机制主张是正确、具体、可直接落地的**。它不是一个模糊的"协同设计"口号，而是一行明确的 EvictKey 定义（Eq.13）。§4.5 的消融把这一项单独摘出来并给出了 18.5pp 的命中率差，是全文最扎实的证据。
- **$q_{\text{cold}}$ 消融的结论比主结果更有价值**。"最大化复用不是最优服务"——1pp 命中率换 16.5% P99 恶化——这是一个可以直接推广到整个前缀亲和领域的教训，而且方向反直觉。作者本可以只报 $q_{\text{cold}}=2$ 的数字，他们做了完整扫描并如实报告了 $q_{\text{cold}}=0$ 命中率更高这个对自己叙事不利的事实。
- **附录写得比正文诚实**。TTFT 是上包络而非纯 prefill 计时器（D.2）、标称 vs 物化 token 长度（D.1）、common-attenuation 何时失效（E.4）、不依赖形式化方差收缩定理（E.5）、wave-to-request 量纲陷阱（E.1）——这些自我限定在正文里一个都没出现，但在附录里交代得清清楚楚。**读这篇论文只读正文会高估它，只读附录会低估它。**
- **只做精确前缀这个克制的选择是对的**。用 segment ID 做控制决策、token 精确匹配做正确性保证，这个分离让 PRISM 与 SGLang 原生 radix tree 天然兼容，也避免了语义近似路线的正确性风险。
- **baseline 选得不算弱**。ContextPilot（MLSys 2026）和 k-LPM 都是同赛道的近期工作，且 k-LPM 做了 $k$ 的扫描取最优。相比很多只跟 vLLM/SGLang 原生比的论文，这是加分。

### 需要留意的

- **单张 A800 是根本性的规模限制**。这不是"实验做小了"的问题，而是**论文所处的层次决定的**——PRISM 的所有机制（segment 计数器、dispatch 优先级、radix anchor 保护）都是单引擎内的。它的结论对分布式 LLM 服务**没有直接推论**：多副本下的核心问题（前缀在哪张卡上、要不要复制、跨副本负载偏斜）PRISM 一个都没碰。综述里给它的定位必须明确写成"引擎层"。
- **主结果的 23.3% 主要来自饱和区**。5 个 QPS 点里 3 个 P99 TTFT 超过 6 秒，等权平均后得到 23.3%。而唯一的轻载对照（附录 C.1，40 QPS / $r_{\text{hot}}$=0.5）显示 **ContextPilot 反而领先 0.01s**。Agent 实验更极端——P50 TTFT 21–35 秒，且增益随负载单调下降（24.9%→11.9%→8.5%），与 RAG 实验的叙事直接矛盾且未被解释。**在 SLO 内运行的系统上，PRISM 的实际价值远小于 23.3% 这个数字的暗示。**
- **QAS 的独立贡献从未被测量**，理由（"没有 QAS 提示 DART 就退化为 LRU"）解释了为什么不做反向消融，但不解释为什么不做 "QAS + LRU"。我的判断是**命中率提升的主因很可能是 QAS 的 segment 顺序统一而非 DART 的保护**（native SGLang 只有 15% 命中率，这个数字太低了，几乎只能用"顺序不一致导致前缀断链"来解释），而论文的实验设计恰好让这个假设无法证伪。
- **Theorem 1 是包装**。三行代数的通分被写成定理 + 独立附录 + 完整证明。它不含任何关于 PRISM 机制的信息，只是"命中率高则服务率高"的恒等式。真正的洞见（$(1-h)^2$ 分母意味着高命中率区的边际收益放大）被埋在里面没有被强调。
- **统计严谨性缺失**：2,048 请求 / 约 34 秒的合成 trace、从空 cache 起、只有 $\alpha$ 扫描标了 single-seed、全文无误差棒、无重复实验。P99 由约 20 条请求决定。**23.3% 这个点估计的不确定区间可能很大**。
- **超参自由度大且大多未扫**：8 个以上超参手工固定，其中 **DART 的保护预算 $f$ 从未给出取值**——这是一个直接影响核心机制的参数，其缺失使得实验无法复现。$\kappa=1$（签名只取一个 segment）也只有定性理由没有数据。
- **无代码、无数据、0 引用、未经同行评审**。作为一篇声称做了非平凡引擎修改的系统论文，不开源意味着所有数字都不可独立验证。
- **负载完全合成**，而论文自己引用了 RAGPulse（真实 RAG 负载 trace）却没有在上面验证。这个选择很难解释——除非在真实 trace 上效果不好。

### 一句话定位

**PRISM 的机制主张（把调度器的 dispatch 意图编码进 KV cache 的淘汰键）是正确且值得抄的，但它的实验规模（单张 A800）、采样偏向（5 个 QPS 点里 3 个深度饱和）、统计严谨性（2048 请求、无重复、无误差棒、关键超参未披露）和预印本状态，共同决定了 23.3% / 37.1% 这两个数字不应被当作可迁移的效应量引用。** 对本仓库的价值在于它提供了「准入 × 缓存」耦合思想在**引擎层**的一个完整实例——与 Meta CacheRoute 在**路由层**的实例构成互补的一对，二者的视野盲区和失效模式恰好相反，而**跨层接口（路由器如何把"我马上要发一批 $b$ 过来"告诉引擎的淘汰器）至今是空白**。

---

## 附：与本仓库其他解读的交叉引用

- [`CacheRoute-论文解读.md`](CacheRoute-论文解读.md) —— **最重要的对照**。同为「准入 × 缓存」耦合，但在入口路由层：规划式前缀亲和 + per-key 准入（top-rate 选前缀，副本数预算 $\sum_b k_b \le C$），60×H100 上 176±11 QPS vs Preble 76±11。关键互补：CacheRoute 的解析式缓存驻留预测偏差中位 14.3pp / p90 44.7pp，最终放弃解析预测改用 shadow replay——**PRISM 结构上免疫这个问题（它就在引擎里，缓存状态是观测量），代价是视野只有一张卡**。
- [`Mooncake-论文解读.md`](Mooncake-论文解读.md) —— 另一个具备准入控制的系统。但 Mooncake 的准入是**拒绝型**（基于 decoding slot 预测的早期拒绝，阈值手工调），PRISM 的准入是**重排型**（热/冷双车道）。重排型自带公平性债务，必须用 FIFO 冷车道显式偿还——这正是 $q_{\text{cold}}$ 消融揭示的权衡。
- [`Preble-论文解读.md`](Preble-论文解读.md) —— PRISM 未引用 Preble，但两者都在做"调度制造前缀局部性"。Preble 的 E2 算法在**副本间**做 exploitation-exploration 与热前缀复制，PRISM 在**单引擎内**做分桶排序——同一思想的分布式版与单机版。
- [`GORGO-论文解读.md`](GORGO-论文解读.md) —— PRISM 的 $q_{\text{cold}}$ 消融（最大化命中率反而恶化 P99 TTFT）与 GORGO 的 reward hacking 案例（优化 p95 TTFT 导致 E2E 尾延迟崩溃）是**同一类问题的两个侧面**：单指标目标函数在有复杂系统机制作为中介时会误导。另外 GORGO 的 `prefix-cache` baseline 在饱和时崩溃（TTFT p95 6784ms）与 PRISM 观察到的 native SGLang 低命中率（15%）指向同一个根因——**没有顺序统一的前缀亲和是脆弱的**。
- [`DualMap-论文解读.md`](DualMap-论文解读.md) —— 提供了负载侧的经验依据：48% 对话请求 / 76% 工具-Agent 请求共享 ≥50% 前缀。这为 PRISM 的 AGENTPREFIX 实验（NG hit 0.75）提供了独立的负载真实性佐证——**agent 负载的前缀共享确实比对话高得多**。
- [`LLM服务负载均衡与调度-文献综述.md`](LLM服务负载均衡与调度-文献综述.md) —— 整体脉络定位。PRISM 应被归入**引擎层调度-内存协同**这一支，与路由层工作明确区分。
