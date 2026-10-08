# 分布式通信代价模型详解（Cost Models）

> 本文系统介绍用于**预测和分析分布式通信开销**的各类理论模型：从最经典的
> $\alpha$-$\beta$（Hockney）模型出发，逐步引入 LogP 家族、带宽层次模型、
> Roofline、对分带宽模型、BSP 与 $\alpha$-$\beta$-$\gamma$ 等，
> 说明**每个模型解释什么、忽略什么、什么时候会失效**，
> 并对应到 SGLang 中 TP/EP/PD 分离等真实场景的选型判断。
>
> 前置阅读：`collective_communication.md`（各通信原语的语义与通信量）。
> 本文是它 §10「$\alpha$-$\beta$ 代价模型」一节的展开与推广。

## 目录

0. [模型谱系：提出时间、提出人与历史背景](#0-模型谱系提出时间提出人与历史背景)
1. [为什么需要代价模型](#1-为什么需要代价模型)
2. [$\alpha$-$\beta$（Hockney）模型](#2-alpha-beta-hockney模型)
3. [用 $\alpha$-$\beta$ 分析集合通信](#3-用-alpha-beta-分析集合通信) · [算法交叉点](#交叉点分析什么时候该换算法)
4. [模型的失效场景：它忽略了什么](#4-模型的失效场景它忽略了什么)
5. [LogP 家族：把"处理器占用"算进来](#5-logp-家族把处理器占用算进来)
6. [$\alpha$-$\beta$-$\gamma$：把归约计算算进来](#6-alpha-beta-gamma把归约计算算进来)
7. [带宽层次与拓扑模型](#7-带宽层次与拓扑模型)
8. [对分带宽模型：All-to-All 的正确标尺](#8-对分带宽模型all-to-all-的正确标尺)
9. [Roofline：通信与计算的统一视角](#9-roofline通信与计算的统一视角)
10. [BSP 与同步开销模型](#10-bsp-与同步开销模型)
11. [模型选型速查](#11-模型选型速查)
12. [在 SGLang 中的应用](#12-在-sglang-中的应用)

---

## 0. 模型谱系：提出时间、提出人与历史背景

本文涉及的模型跨越了近 40 年。先摆出完整谱系，再进入细节，会更容易理解
**每个模型都是在回应当时硬件的某个新矛盾**，而不是凭空变复杂。

| 模型 | 年份 | 提出人 / 机构 | 出处 | 回应的时代矛盾 |
| --- | --- | --- | --- | --- |
| **BSP** | 1990 | Leslie Valiant（哈佛，2010 图灵奖得主） | CACM, *A Bridging Model for Parallel Computation* | 并行机器百花齐放但无通用抽象，需要一个类比冯·诺依曼模型的“桥梁模型” |
| **LogP** | 1993 | David Culler, Richard Karp, David Patterson et al.（UC Berkeley） | PPoPP'93 | 向量机衰落、MPP/工作站机群兴起，通信开销中 **CPU 占用** 开始与线上延迟同量级 |
| **Hockney $\alpha$-$\beta$** | 1994（思想更早） | Roger Hockney（英国 Reading 大学） | Parallel Computing, *The communication challenge for MPP* | 需要一个能用 ping-pong 实测直接拟合、工程上能用的极简模型 |
| **LogGP** | 1995 | Albert Alexandrov, Mihai Ionescu, Klaus Schauser, Chris Scheiman（UCSB） | SPAA'95 | LogP 假设定长小消息，但实际应用大量传**长消息** |
| **LogGPS** | 2001 | Fumihiko Ino, Noriyuki Fujimoto, Kenichi Hagihara（大阪大学） | ICS'01 | MPI 实现在 eager/rendezvous 两种协议间切换，实测曲线出现**台阶** |
| **Roofline** | 2009 | Samuel Williams, Andrew Waterman, David Patterson（UC Berkeley） | CACM, *Roofline: An Insightful Visual Performance Model* | 多核时代算力增长远快于内存带宽，**memory wall** 成为主矛盾 |
| **对分带宽** | 1980s– | 源自 VLSI 布线理论（Thompson 1980），经 Leiserson 的 fat-tree（1985）普及 | — | 大规模互连网络下，“每链路独立”假设失效，全局流量模式才是瓶颈 |
| **$\alpha$-$\beta$-$\gamma$** | 2000s | 非单一提出人；Chan, Heimlich, Purkayastha & van de Geijn（UT Austin，2007）系统化 | CCPE 2007 | 集合通信算法系统化分析时，归约的**算术开销**必须入账 |

### 一条主线：每一次扩展都在“把一个被忽略的资源重新计入”

```text
Hockney α-β  ──加归约计算──►  α-β-γ
     │
     │──加处理器占用──►  LogP ──加长消息──►  LogGP ──加协议切换──►  LogGPS
     │
     │──加带宽层次──►    分层模型
     │
     │──加链路争抢──►    对分带宽模型
     │
     └──加同步等待──►    BSP（实为独立分支：1990 年即发表，并非从 α-β 衍生）

Roofline 是另一条线：不分析通信算法，而是把**计算与数据移动**放到同一张图上
```

### 为什么一个 1994 年的模型今天还在用

值得玩味的是：最古老、最粗糙的 $\alpha$-$\beta$ 反而是今天工程上用得最多的。原因在于：

1. **参数可测**——LogP 的 $o$ 和 $g$ 很难单独标定，而 $\alpha$、$\beta$ 一次 ping-pong 就拟合出来了；
2. **硬件演进反而帮了它**——RDMA / GPUDirect / NVLink 把 LogP 最在意的 $o$（处理器占用）压到极低，
   LogP 在这些链路上几乎退化回 $\alpha$-$\beta$；
3. **它回答的是结构性问题**——“延迟受限还是带宽受限”这个二分法，40 年没变。

但在 **MoE / EP** 这类场景下，$\alpha$-$\beta$ 会错得很离谱（§8）——
这恰恰说明了为什么后面那些“更复杂”的模型仍有存在必要。

---

## 1. 为什么需要代价模型

面对"TP 开 4 还是 8？"、"EP 能不能跨机？"、"这个 all-reduce 该用 ring 还是 one-shot？"
这类问题，有三种回答方式：

| 方式 | 成本 | 问题 |
| --- | --- | --- |
| 直接跑 benchmark | 高（要真机、真模型） | 只能告诉你"哪个快"，不能告诉你"为什么"、"换个配置还成不成立" |
| 拍脑袋 | 零 | 经常错 |
| **建代价模型** | 低 | 能给出**趋势和量级**，指导搜索方向 |

代价模型的价值不在于精确预测毫秒数（做不到），而在于回答**结构性问题**：

- 这个开销随卡数 $p$ 是**常数、对数、线性还是平方**增长？
- 瓶颈是**延迟**还是**带宽**？（决定优化方向完全不同）
- 换个算法/拓扑，量级会变吗？

> **核心心态**：所有模型都是错的，但有些是有用的。
> 用模型定位**瓶颈类型**和**scaling 趋势**，用 benchmark 定常数。

---

## 2. $\alpha$-$\beta$（Hockney）模型

> **出处**：Roger Hockney（1929–1999，英国 Reading 大学），
> *The communication challenge for MPP: Intel Paragon and Meiko CS-2*（Parallel Computing, 1994）。
> **背景**：1990 年代初是 MPP（大规模并行处理机）的黄金期，Intel Paragon、Meiko CS-2、CM-5 等
> 机器拓扑各异，用户需要一个能**跨机器横向对比**的共同语言。Hockney 的做法是把一台机器的通信能力
> 压缩成两个可实测的数字。他同时推广了 $r_\infty$（渐进带宽）与 $n_{1/2}$（半带宽点）这对指标——
> 这套词汇其实是他从更早的**向量机性能分析**（§2.2 的 $n_{1/2}$ 最初是描述向量流水线启动开销的）平移过来的。
>
> 需要注意：“线性延迟 + 带宽”这个形式在通信领域已非正式流传多年（甚至可追到 1970s 的排队与网络分析），
> Hockney 的贡献在于**把它标准化为并行机器的基准测量方法**，而非“发明”了这个公式。

### 2.1 定义

最经典、最常用的通信模型。传输 $n$ 字节的时间为：

$$
T(n) = \alpha + n\beta
$$

| 符号 | 含义 | 单位 | 典型量级（单机 NVLink） |
| --- | --- | --- | --- |
| $\alpha$ | **启动延迟**（latency / startup cost），与消息大小无关 | s | 1~10 μs |
| $\beta$ | **每字节传输代价** $=1/B$，$B$ 为带宽 | s/byte | NVLink 4.0 单向 ~450 GB/s → $\beta \approx 2.2\times10^{-3}$ μs/KB |
| $n$ | 消息字节数 | byte | — |

$\alpha$ 里包含什么：kernel launch、协议握手、对端就绪的 flag 轮询/信号量同步、
链路固定开销、（跨节点时）网卡中断与协议栈处理。

### 2.2 关键推论：半带宽点

令延迟项与带宽项相等，$\alpha = n_{1/2}\beta$，得到**半带宽消息长度**：

$$
n_{1/2} = \frac{\alpha}{\beta} = \alpha B
$$

这个量非常有用——它是**"小消息"与"大消息"的分界线**：

- $n \ll n_{1/2}$：**延迟受限（latency-bound）**，时间几乎与 $n$ 无关，优化方向是**减少轮次/合并消息**；
- $n \gg n_{1/2}$：**带宽受限（bandwidth-bound）**，时间正比于 $n$，优化方向是**减少字节数/压缩**。

代入 NVLink（$\alpha\approx5\ \mu s$、$B\approx450$ GB/s）：

$$
n_{1/2} \approx 5\times10^{-6}\,\text{s} \times 450\times10^9\,\text{B/s} \approx 2.25\ \text{MB}
$$

**结论很反直觉**：即便在 NVLink 上，小于约 2 MB 的消息都还处在延迟受限区。
LLM decode 阶段单次 all-reduce 的张量通常只有几十 KB ~ 几百 KB
（$\text{batch} \times d_\text{model} \times 2\text{B}$），**远在延迟区**——
这正是 `collective_communication.md` §10 里 one-shot all-reduce 能赢 ring 的根本原因。

跨节点 IB（$\alpha\approx2\ \mu s$、$B\approx25$ GB/s）则 $n_{1/2}\approx 50$ KB，分界线低得多。

### 2.3 为什么这个模型这么流行

1. **只有两个参数**，可以用 ping-pong benchmark 直接测出来（拟合 $T$-$n$ 直线的截距和斜率）；
2. **可加性好**：多步算法直接把每步的 $\alpha+n\beta$ 相加，便于比较算法；
3. **抓住了主要矛盾**：延迟 vs 带宽这个二分法，覆盖了绝大多数工程决策。

---

## 3. 用 $\alpha$-$\beta$ 分析集合通信

把 §2 的模型套到各原语上（$p$ 个 rank，每 rank 数据量 $N$），
这也是 `collective_communication.md` 各节通信量结论的来源：

| 原语 | 算法 | 代价 |
| --- | --- | --- |
| Broadcast | 扁平（root 直发） | $(p-1)(\alpha + N\beta)$ |
| Broadcast | 二叉树 | $\lceil\log_2 p\rceil(\alpha + N\beta)$ |
| Broadcast | scatter + all-gather | $(\log_2 p + p-1)\alpha + 2\frac{p-1}{p}N\beta$ |
| All-Gather | Ring | $(p-1)\alpha + \frac{p-1}{p}N\beta$ |
| All-Gather | Direct（一轮全发） | $\alpha + \frac{p-1}{p}N\beta$ |
| Reduce-Scatter | Ring | $(p-1)\alpha + \frac{p-1}{p}N\beta$（+ 归约计算，见 §6） |
| **All-Reduce** | **Ring** | $2(p-1)\alpha + 2\frac{p-1}{p}N\beta$ |
| **All-Reduce** | **Recursive Halving-Doubling** | $2\log_2 p\,\alpha + 2\frac{p-1}{p}N\beta$ |
| **All-Reduce** | **One-shot（直写）** | $\alpha + (p-1)N\beta$ |

### 交叉点分析：什么时候该换算法

以 Ring vs One-shot all-reduce 为例，令两者相等：

$$
2(p-1)\alpha + 2\tfrac{p-1}{p}N\beta \;=\; \alpha + (p-1)N\beta
$$

解出交叉点（精确解）：

$$
N^{*} \;=\; \frac{\alpha}{\beta}\cdot\frac{p(2p-3)}{(p-1)(p-2)}
\qquad\xrightarrow{\ p\to\infty\ }\qquad
2\,\frac{\alpha}{\beta} \;=\; 2\,n_{1/2}
$$

这个系数收敛得很快：


| $p$ | 3 | 4 | 6 | 8 | 16 | 64 |
| --- | --- | --- | --- | --- | --- | --- |
| $N^{*}/n_{1/2}$ | 4.50 | 3.33 | 2.70 | **2.48** | 2.21 | 2.05 |

即**交叉点大致就在半带宽点的 2~3 倍**。代入 NVLink 的 $n_{1/2}\approx 2.25$ MB（§2.2）、$p=8$：

$$
N^{*} \approx 2.48 \times 2.25\ \text{MB} \approx 5.6\ \text{MB}
$$

与 SGLang `custom_all_reduce` 的 `_MAX_CAR_SIZE = 8 MB`（CUDA）**落在同一量级**。
剩余差异来自模型未涵盖的因素：two-shot 变体、NVLS 硬件归约、显存拷贝开销，以及实测调优。

> **这就是代价模型的正确用法**：它给出了「阈值应该在 MB 量级而非 KB 或 GB 量级」这个结论，
> 具体是 4 MB 还是 8 MB 要靠 benchmark 定。

---

## 4. 模型的失效场景：它忽略了什么

$\alpha$-$\beta$ 模型的简洁来自一系列强假设。**知道它何时失效，比会用它更重要**：

| 被忽略的因素 | 后果 | 何时必须考虑 |
| --- | --- | --- |
| **CPU/GPU 占用**（发消息也要算力） | 低估了通信对计算的干扰 | 想做通信-计算 overlap 时 → §5 LogP |
| **归约的算术开销** | 低估 all-reduce | 大张量 reduce、低算力设备 → §6 |
| **带宽非均匀**（NVLink vs PCIe vs IB） | 单一 $\beta$ 无意义 | 跨节点、混合拓扑 → §7 |
| **链路争抢 / 对分带宽** | 严重低估 all-to-all | EP、大规模 → §8 |
| **网络拥塞、incast** | 尾延迟被低估 | 多对一流量模式 |
| **负载不均**（各 rank 到达时间不同） | 忽略了同步等待 | MoE 专家不均、变长序列 → §10 |
| **流水线/分块** | 高估了多步算法 | NCCL chunk 流水线（见 `collective_communication.md` §10） |

一个具体的例子：§3 表中 direct all-gather 只要 $\alpha + \frac{p-1}{p}N\beta$，
看起来永远优于 ring。但模型假设"每 rank 一条独立链路"，
实际上 direct 让**一张卡的出口同时服务 $p-1$ 个对端**，
在非全互联拓扑上会严重劣化——这正是
`collective_communication.md` §8.1 讨论的内容，也是需要 §7、§8 模型的原因。

---

## 5. LogP 家族：把"处理器占用"算进来

### 5.1 LogP（Culler et al., 1993）

> **出处**：David Culler, Richard Karp, David Patterson, Abhijit Sahay, Klaus Schauser,
> Eunice Santos, Ramesh Subramonian, Thorsten von Eicken（UC Berkeley），
> *LogP: Towards a Realistic Model of Parallel Computation*（PPoPP 1993）。
> 作者名单里有两位图灵奖得主（Karp 1985、Patterson 2017）。
>
> **背景**：这篇论文是对当时主流的 **PRAM 模型的公开反叛**。PRAM 假设共享内存、零延迟访问，
> 用它设计出的算法在真机上普遍跑不出性能。作者们观察到：随着 CM-5、nCUBE 这些
> **分布式内存机群**取代共享内存机，且微处理器性能飞速提升（网络相对变慢），
> “发一条消息要花多少 CPU 周期”已经**超过了消息在线上飞行的时间**。
> 于是他们把“处理器占用”$o$ 和“注入速率”$g$ 提升为一等公民。
> 模型名字本身就是四个参数首字母的拼写：$L, o, g, P$。

$\alpha$-$\beta$ 把通信看作“纯粹的等待”，但实际上**发送和接收都要占用处理器**。
LogP 用四个参数刻画：

| 参数 | 含义 |
| --- | --- |
| $L$ (**L**atency) | 网络传输延迟（线上飞行时间） |
| $o$ (**o**verhead) | **处理器**为收/发一条消息付出的**占用**时间（这段时间 CPU/GPU 不能干别的） |
| $g$ (**g**ap) | 连续两次消息注入的**最小间隔**，$1/g$ 即每处理器的消息注入率上限 |
| $P$ | 处理器数 |

单条小消息的端到端时间 $\approx o_{\text{send}} + L + o_{\text{recv}}$。

**$o$ 与 $g$ 的区别是精髓所在**：

- $o$ 是**占用**（occupancy）：这段时间处理器被"扣住"，无法参与计算；
- $g$ 是**节流**（throttling）：由网卡/链路注入率决定，$g > o$ 时处理器有空闲，可以插入计算。

这直接解释了**为什么通信-计算 overlap 有时无效**：如果 $o$ 很大（每条消息都要 CPU 深度参与，
如无 GPUDirect 的场景），那"异步通信"其实还是在占用处理器，overlap 不出来。
反之 RDMA/NVLink 的 $o$ 极小，overlap 才真正有效。

> **SGLang 的关联**：DeepEP 用 NVSHMEM 做 low-latency dispatch，本质就是把 $o$ 压到极低
> （GPU 直接发起通信，不经过 CPU），从而让 MoE 的 all-to-all 能与专家计算 overlap。
> 参见 `EP.md` 与 `python/sglang/srt/layers/moe/token_dispatcher/`。

### 5.2 LogGP：补上大消息

> **出处**：Albert Alexandrov, Mihai Ionescu, Klaus Schauser, Chris Scheiman（UC Santa Barbara），
> *LogGP: Incorporating Long Messages into the LogP Model*（SPAA 1995）。
> Schauser 本人就是 LogP 原作者之一，可以看作官方修订。
> **背景**：两年实践下来发现 LogP 对**长消息预测得很差**——当时 Meiko CS-2、IBM SP-2 等机器
> 已能将长消息流水线化传输，而 LogP 只能把它拆成 $n/w$ 条小消息逐条计费，严重高估。

LogP 假设消息都很小（定长）。**LogGP** 加了参数 $G$（每字节的 gap），
使长消息的传输时间为 $(n-1)G$：

$$
T_{\text{LogGP}} \approx o_{\text{send}} + (n-1)G + L + o_{\text{recv}}
$$

可以看出 **LogGP 退化到 $\alpha$-$\beta$**：令 $\alpha = o_s + L + o_r$、$\beta = G$ 即得。
所以 $\alpha$-$\beta$ 可视为 LogGP 的"忽略处理器占用"版本。

### 5.3 其他变体

- **LogGPS**（Ino, Fujimoto & Hagihara，大阪大学，ICS 2001）：再加入 $S$（同步阈值），
  刻画 MPI 中小消息用 eager 协议、大消息用 rendezvous 协议（需先握手）的**协议切换**。
  背景是 1990s 末 MPI 成为事实标准后，模型需要解释的不再是裸硬件，而是**带协议栈的实现**——
  这解释了实测曲线上在某个 size 处出现的**台阶**。
- **LoGPC**（Moritz & Frank，MIT，1998）：加入网络竞争（**C**ontention）建模，
  回应的是多级网络上多流量叠加时 LogP 系列全部失准的问题。

---

## 6. $\alpha$-$\beta$-$\gamma$：把归约计算算进来

> **出处**：没有单一“提出人”，是集合通信算法社区在 1990s–2000s 逐步形成的惯例记法。
> 系统化整理的代表作是 Ernie Chan, Marcel Heimlich, Avi Purkayastha & Robert van de Geijn（UT Austin），
> *Collective communication: theory, practice, and experience*（CCPE 2007）；
> Rabenseifner（斯图加特 HLRS）在 2004 年提出的 reduce-scatter + all-gather 型 all-reduce（即今天的 Ring）
> 也是在这套记法下分析的。
> **背景**：当人们开始为 **MPI 库选择最优集合通信算法**时，发现不同算法的归约次数并不相同
> （比如 recursive doubling 的归约量是 $\log p \cdot N$，而 ring 只有 $\frac{p-1}{p}N$），
> 不把 $\gamma$ 写出来就无法公平对比。在当时的 CPU 上这一项确实不小。

All-Reduce / Reduce-Scatter 除了搬数据还要**做加法**。引入 $\gamma$ = 每字节（或每元素）的归约计算代价：

$$
T = \#\text{rounds}\cdot\alpha \;+\; (\text{bytes})\cdot\beta \;+\; (\text{reduced elements})\cdot\gamma
$$

Ring All-Reduce 的完整形式：

$$
T_{\text{ring-AR}} = 2(p-1)\alpha \;+\; 2\frac{p-1}{p}N\beta \;+\; \frac{p-1}{p}N\gamma
$$

注意 $\gamma$ 项**只有一份**（$\frac{p-1}{p}N$ 而非 $2\times$）——因为只有 Reduce-Scatter 阶段做加法，
All-Gather 阶段是纯搬运。这与 `collective_communication.md` §10 的
"归约段可压缩 / 复制段不可压缩"是同一件事的两个侧面。

**什么时候 $\gamma$ 不可忽略**：

- GPU 上做 fp32 加法极快，$\gamma \lll \beta$，通常可忽略；
- 但在**低精度累加需要升位**（fp8 → fp32 累加再量化回去）、
  或**归约算子复杂**（如 MoE combine 的加权求和）时，$\gamma$ 会显现；
- CPU all-reduce（如 SGLang 的 gloo 路径、参数同步）上 $\gamma$ 常常不可忽略。

---

## 7. 带宽层次与拓扑模型

> **出处**：同样无单一提出人。它的思想源头是 1980s–1990s 的 **NUMA / 存储层次结构**研究，
> 在集合通信上的典型落地是“hierarchical collectives”：
> Kandalla et al.（OSU，2009）的多核感知 MPI 集合通信、
> 以及后来 NCCL 的 tree / CollNet 算法。
> **背景**：2000s 中后期多核 + 多路 CPU 普及，“同一节点内的两个进程”与
> “跨节点的两个进程”带宽差一两个数量级，单一 $\beta$ 彻底失去意义。
> GPU 时代 NVLink 的出现（NVLink 1.0, 2016）又把这个差距拉得更大。

现实系统的带宽是**分层**的，用单一 $\beta$ 无法描述。典型层次（数量级）：

| 层次 | 单向带宽 | $\beta$ 相对值 | 延迟 $\alpha$ |
| --- | --- | --- | --- |
| GPU HBM | 3~8 TB/s | 1× | ns |
| NVLink / NVSwitch（机内） | 400~900 GB/s | ~10× | ~1-5 μs |
| PCIe Gen5 x16 | ~60 GB/s | ~100× | ~5-10 μs |
| IB / RoCE（跨机） | 25~50 GB/s | ~200× | ~2-10 μs |
| 以太网（跨机房） | 1~10 GB/s | ~1000× | ms |

### 分层代价模型

把通信按所处层次分开计费：

$$
T = \sum_{\ell \in \text{levels}} \left( r_\ell\,\alpha_\ell + n_\ell\,\beta_\ell \right)
$$

**这是"TP 不跨机、PP 才跨机"的定量依据**：

- TP 每层 2 次 all-reduce，若跨机则 $\beta$ 涨 ~20 倍，且频率极高 → 实践上强烈不建议
  （SGLang 对 CP 更是直接断言 `tp_size <= 8`，见 `CP.md` §9）；
- PP 只在 stage 边界传一次激活，频率低、量小 → 跨机可接受；
- EP 的 all-to-all 跨机则要看对分带宽（§8）。

### 分层算法（hierarchical collectives）

带宽层次也直接指导算法设计：**先在机内归约，再跨机通信，最后机内广播**。

以 2 机 × 8 卡的 all-reduce 为例：

1. 机内 reduce-scatter（NVLink，快）→ 每卡持 $N/8$；
2. 跨机 all-reduce 这 $N/8$（IB，只有 $1/8$ 的量）；
3. 机内 all-gather（NVLink）。

**跨机流量从 $N$ 降到 $N/8$**，这就是 NCCL 的 tree/CollNet 算法和
各类 hierarchical all-reduce 的核心思想。

---

## 8. 对分带宽模型：All-to-All 的正确标尺

> **概念源头**：“bisection width / bandwidth”来自 **VLSI 布线复杂度理论**——
> C. D. Thompson 的博士论文 *A Complexity Theory for VLSI*（CMU, 1980）
> 用切割芯片的“割面宽度”推导面积下界。
> **引入并行计算的关键人物**：Charles Leiserson（MIT），
> *Fat-Trees: Universal Networks for Hardware-Efficient Supercomputing*（IEEE ToC, 1985）——
> fat-tree 的整个设计动机就是“让越靠近根的链路越粗，以保证对分带宽不随规模退化”。
> 今天 IB / RoCE 组网的 “full bisection” 宣传词，直接继承自这篇 1985 年的论文。
>
> **背景**：当并行机从几十节点扩到数千节点，人们发现“点到点带宽”这个指标彻底不够用了：
> 单流跑得快不代表全局流量模式下不崩。FFT、矩阵转置、排序这类
> **all-to-all 型**负载是当时暴露问题的主力——而今天 MoE 的 dispatch/combine
> 恰好是同一类流量模式，所以这个 40 年前的标尺又重新变得关键。

对 All-to-All，$\alpha$-$\beta$ 会给出 $\frac{p-1}{p}N\beta$——
与 all-gather 相同，**严重低估**。因为它假设链路独立，而 all-to-all 会打满整个网络。

正确的标尺是**对分带宽（bisection bandwidth）** $B_{\text{bisec}}$：
把 $p$ 个节点任意均分成两半，跨越切面的总带宽。

All-to-All 时跨切面的流量约为（见 `collective_communication.md` §11.2）：

$$
\text{跨切面流量} \approx \frac{p}{2}\times\frac{p/2}{p}N = \frac{p}{4}N
\qquad\Longrightarrow\qquad
T_{\text{a2a}} \gtrsim \frac{pN/4}{B_{\text{bisec}}}
$$

关键在于 $B_{\text{bisec}}$ 随拓扑的 scaling：

| 拓扑 | 对分带宽 | All-to-All 表现 |
| --- | --- | --- |
| Fat-tree（full bisection） | $\propto p$ | $T$ 约为常数，可扩展 |
| Fat-tree（oversubscribed 1:4） | $\propto p/4$ | 退化 4 倍 |
| 2D Torus | $\propto \sqrt{p}$ | $T \propto \sqrt{p}$，勉强 |
| Ring / 1D | $O(1)$ | $T \propto p$，不可用 |

**这解释了为什么大规模 EP 必须上 full-bisection 的 IB/RoCE 组网**，
而 TP 靠机内 NVSwitch 就够——all-reduce 的每卡通信量趋于常数 $2N$，
all-to-all 的对分压力却随 $p$ 线性上升。

---

## 9. Roofline：通信与计算的统一视角

> **出处**：Samuel Williams, Andrew Waterman, David Patterson（UC Berkeley），
> *Roofline: An Insightful Visual Performance Model for Multicore Architectures*（CACM, 2009）。
> 前身是 Williams 的博士论文（2008）。
> **背景**：2005 年前后主频撞墙，产业转向**多核**，但内存带宽的增长远跟不上核数——
> 即“memory wall”。开发者面对“我优化到底还有多少空间”时无从判断。
> Roofline 的设计目标非常明确：不追求精确，而是用**一张双对数图**让工程师一眼看出
> 自己的 kernel 是撞到了算力天花板还是带宽斜坠。这种“可视化优先”的哲学是它能普及的主因。
>
> Patterson 同时是 LogP（§5.1）的作者之一，相隔 16 年。

前面的模型只看通信。**Roofline** 把计算和数据移动放在同一张图上，
判断一个 kernel 到底受限于**算力**还是**访存/通信**。

定义**算术强度**（arithmetic intensity）：

$$
I = \frac{\text{FLOPs}}{\text{Bytes moved}}
\qquad
P_{\text{attainable}} = \min\left(P_{\text{peak}},\; I \times B\right)
$$

拐点 $I^* = P_{\text{peak}}/B$：$I < I^*$ 为**访存/通信受限**，$I > I^*$ 为**算力受限**。

### 对 LLM 推理的直接结论

| 阶段 | 算术强度 | 受限于 |
| --- | --- | --- |
| **Prefill**（GEMM，大 $M$） | 高（$\propto$ token 数） | **算力**（compute-bound） |
| **Decode**（GEMV，$M=1$） | 极低（每个权重字节只做 2 FLOP） | **HBM 带宽**（memory-bound） |

这解释了两件事：

1. **Decode 阶段权重读取是瓶颈** → 量化（fp8/int4）直接提升 decode 吞吐，
   因为它减少的是 bytes 而非 FLOPs；
2. **Decode 的通信也在延迟区**（§2.2）→ 所以 TP 在 decode 时的扩展性远差于 prefill，
   这也是 PD 分离（`disaggregation/`）能带来收益的理论依据之一：
   **prefill 和 decode 的瓶颈类型根本不同，应该分开配置并行度**。

> **把 Roofline 推广到通信**：把"Bytes moved"换成"跨卡通信字节"，
> 就得到一个判断"该不该增大 TP"的粗略标尺——
> TP 增大会同时降低单卡计算量（好）和增加通信量（坏），
> 当通信项超过计算项时继续增大 TP 就是负收益。

---

## 10. BSP 与同步开销模型

### BSP（Bulk Synchronous Parallel）

> **出处**：Leslie Valiant（哈佛大学），*A Bridging Model for Parallel Computation*（CACM, 1990）。
> Valiant 因在计算理论（含 PAC 学习理论）的贡献获 2010 年图灵奖。
> **背景**：论文标题里的 “bridging model” 是理解它的钥匙。Valiant 的论点是：
> 串行计算之所以繁荣，是因为**冯·诺依曼模型**在硬件与软件之间架了一座桥，
> 让两边可以独立演进；而并行计算当时缺的正是这座桥。
> 他提出用 superstep + 屏障作为约定，硬件只需保证 $g$ 和 $\ell$ 两个指标。
>
> **它的影响远超代价模型本身**：Google Pregel、Apache Hama、Spark 的 stage 划分，
> 以及深度学习里的**同步 SGD**，本质上都是 BSP 结构。
> 本文引用它，主要是因为那两个 $\max$ 项精准刻画了“负载不均 → 通信等待”这个现象。

把并行程序看作一串 **superstep**，每步 = 本地计算 + 通信 + **全局屏障**：

$$
T_{\text{superstep}} = \max_i w_i \;+\; \max_i h_i \cdot g \;+\; \ell
$$

其中 $w_i$ 是 rank $i$ 的计算量，$h_i$ 是其收发消息数，$\ell$ 是屏障代价。

**关键是那两个 $\max$**：整体速度由**最慢的 rank** 决定。这引出集合通信中常被忽略的一项：

$$
T_{\text{实际}} = T_{\text{通信}} + \underbrace{T_{\text{等待}}}_{\text{负载不均导致}}
$$

### 为什么这对 SGLang 很重要

集合通信是**同步**的——所有 rank 必须都到达才能开始。因此**任何负载不均都会转化为通信等待**，
而 profiler 上会误显示为"通信慢"：

| 场景 | 不均来源 | SGLang 的应对 |
| --- | --- | --- |
| **MoE / EP** | 热点 expert 收到的 token 远多于其他 | EPLB 专家负载均衡（`EP.md` §6） |
| **DP attention** | 各 DP rank 的 batch token 数不同 | DP rank 间同步 token 数、padding |
| **CP + causal** | 后半段序列的 attention 计算量大得多 | zigzag 切分（`CP.md` §4） |
| **TBO** | 两个 micro-batch 大小不均 | `tbo_token_distribution_threshold`（默认 0.48）判断是否值得开 |

> **诊断提示**：如果 profile 显示 all-reduce 耗时远超 $\alpha$-$\beta$ 预测值，
> **先怀疑负载不均（等待），而不是网络慢**。
> 验证方法：在集合通信前插一个 barrier，如果 barrier 很慢而 all-reduce 变快，那就是不均问题。

---

## 11. 模型选型速查

| 想回答的问题 | 用哪个模型 | 关键量 |
| --- | --- | --- |
| 小消息还是大消息？该优化延迟还是带宽？ | **$\alpha$-$\beta$** | $n_{1/2}=\alpha/\beta$ |
| ring / tree / one-shot 怎么选？ | **$\alpha$-$\beta$** | 轮次 vs 字节的交叉点 |
| 通信能不能和计算 overlap？ | **LogP** | $o$（处理器占用） |
| 大张量 all-reduce，加法开销要紧吗？ | **$\alpha$-$\beta$-$\gamma$** | $\gamma$ 相对 $\beta$ |
| TP 能不能跨机？ | **带宽层次模型** | $\beta_{\text{IB}}/\beta_{\text{NVLink}}$ |
| EP 能扩到多少卡？ | **对分带宽模型** | $B_{\text{bisec}}$ 的 scaling |
| decode 为什么这么慢？ | **Roofline** | 算术强度 $I$ |
| 为什么实测比预测慢很多？ | **BSP** | 负载不均导致的 $\max$ |

**组合使用是常态**。例如判断"DeepSeek-V3 的 EP 该怎么配"：
用对分带宽模型看 all-to-all 的 scaling，用 LogP 看能否与专家计算 overlap，
用 BSP 看 EPLB 是否必要，最后用 $\alpha$-$\beta$ 定 dispatch 的 chunk 大小。

---

## 12. 在 SGLang 中的应用

### 12.1 参数怎么测

- **$\alpha$、$\beta$**：`nccl-tests` 的 `all_reduce_perf -b 1K -e 1G -f 2`，
  对 size-时间曲线做线性拟合：截距是 $\alpha \times$ 轮次，斜率是 $\beta \times$ 系数。
- **$n_{1/2}$**：曲线上"时间开始明显随 size 增长"的拐点。
- **$B_{\text{bisec}}$**：`all_to_all_perf`，或用一半节点对另一半打流。

### 12.2 模型解释的设计决策

| SGLang 设计 | 模型依据 |
| --- | --- |
| `custom_all_reduce` 按 size 分流（`_MAX_CAR_SIZE`） | $\alpha$-$\beta$ 交叉点（§3） |
| 只支持 world size ∈ {2,4,6,8} | direct 的 $\alpha(p)$ 退化（§4、`collective_communication.md` §8.1） |
| TP 实践上尽量不跨机；CP 更是硬性要求单机（`server_args.py` 断言 `tp_size <= 8`） | 带宽层次（§7） |
| PP 用于跨机 | 分层模型：低频 P2P 容忍高 $\beta$（§7） |
| DeepEP / NVSHMEM low-latency 模式 | LogP 的 $o$ 最小化，换取 overlap（§5.1） |
| EPLB 专家均衡 | BSP 的 $\max$ 项（§10） |
| Two-Batch Overlap（`enable_two_batch_overlap`） | 用计算填充通信的 $\alpha$ 与传输窗口（§5、§10） |
| PD 分离 | Roofline：prefill 算力受限、decode 访存受限（§9） |
| FP8 dispatch / BF16 combine | 带宽项 $n\beta$ 与精度的权衡（§6） |

### 12.3 相关代码

- `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`：size 阈值分流；
- `python/sglang/srt/layers/moe/token_dispatcher/`：EP all-to-all 后端；
- `python/sglang/srt/eplb/`：专家负载均衡；
- `python/sglang/srt/batch_overlap/`、`server_args.py` 的 `enable_two_batch_overlap`：通信-计算重叠；
- `python/sglang/srt/disaggregation/`：PD 分离。

---

## 参考与延伸

- 同目录：`collective_communication.md`（各原语的通信量与 Ring 下界证明，本文的直接前置）、
  `TP.md`、`EP.md`、`PP.md`、`CP.md`、`SP.md`、`DP.md`、`DP_attention.md`。
- **$\alpha$-$\beta$ / 集合通信算法**：
  Hockney（Reading 大学）, *The communication challenge for MPP: Intel Paragon and Meiko CS-2*
  （Parallel Computing, 1994）——$\alpha$-$\beta$ 模型的标准出处；
  Rabenseifner（HLRS Stuttgart）, *Optimization of Collective Reduction Operations*（ICCS 2004）——
  reduce-scatter + all-gather 型 all-reduce（即今天的 Ring）；
  Thakur, Rabenseifner & Gropp（Argonne / HLRS）, *Optimization of Collective Communication Operations in MPICH*
  （IJHPCA 2005）——按消息大小切换算法的经典依据；
  Chan, Heimlich, Purkayastha & van de Geijn（UT Austin）,
  *Collective communication: theory, practice, and experience*（CCPE 2007）——
  各原语的下界、recursive halving-doubling 与 $\alpha$-$\beta$-$\gamma$ 记法的系统化。
- **LogP 家族**：
  Culler, Karp, Patterson, Sahay, Schauser, Santos, Subramonian & von Eicken（UC Berkeley）,
  *LogP: Towards a Realistic Model of Parallel Computation*（PPoPP 1993）——对 PRAM 模型的反叛；
  Alexandrov, Ionescu, Schauser & Scheiman（UCSB）,
  *LogGP: Incorporating Long Messages into the LogP Model*（SPAA 1995）；
  Moritz & Frank（MIT）, *LoGPC*（1998）——加入网络竞争；
  Ino, Fujimoto & Hagihara（大阪大学）, *LogGPS: A Parallel Computational Model for Synchronization Analysis*
  （ICS 2001）——eager/rendezvous 协议切换建模。
- **BSP**：Valiant（哈佛大学，2010 图灵奖）, *A Bridging Model for Parallel Computation*（CACM 1990）。
- **Roofline**：Williams, Waterman & Patterson（UC Berkeley）,
  *Roofline: An Insightful Visual Performance Model for Multicore Architectures*（CACM 2009）。
- **对分带宽 / 拓扑**：
  Thompson（CMU）, *A Complexity Theory for VLSI*（博士论文，1980）——bisection width 概念源头；
  Leiserson（MIT）, *Fat-Trees: Universal Networks for Hardware-Efficient Supercomputing*
  （IEEE Trans. Computers, 1985）——full-bisection 组网的理论基础。
- **实践工具**：NVIDIA `nccl-tests`、NCCL 的 `NCCL_ALGO` / `NCCL_PROTO` 环境变量
  （可强制指定 ring/tree/CollNet 与 LL/LL128/Simple 协议，用于验证模型预测）。

