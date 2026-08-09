# MLP / FFN 结构演进：从两层线性到 GLU 与 MoE

> 本文用**第一性原理拆解** Transformer 中 FFN（Feed-Forward Network，也叫 MLP）的各种结构：
> 为什么必须存在这一层、为什么它要「先升维再降维」、
> 为什么 SwiGLU 用三个矩阵取代了两个矩阵、
> 以及 MoE 如何把「参数量」与「计算量」解耦。
> 最后对应到 SGLang 中 `python/sglang/srt/layers/activation.py` 与
> `python/sglang/srt/models/*.py` 的真实实现与推理侧工程考量。

## 目录

1. [第一性原理：FFN 存在的不可降维理由](#1-第一性原理ffn-存在的不可降维理由)
2. [结构谱系总览](#2-结构谱系总览)
3. [Vanilla FFN：两层线性 + 逐点非线性](#3-vanilla-ffn两层线性--逐点非线性)
4. [GLU 家族：把激活从「标量函数」升级为「乘法门」](#4-glu-家族把激活从标量函数升级为乘法门)
5. [高阶结构化类比](#5-高阶结构化类比)
6. [MoE：把参数量与计算量解耦](#6-moe把参数量与计算量解耦)
7. [边界与反模式](#7-边界与反模式)
8. [张量并行下的切分语义](#8-张量并行下的切分语义)
9. [SGLang 中的实现](#9-sglang-中的实现)
10. [语义压缩：一句话本质](#10-语义压缩一句话本质)
11. [常见问题速答](#11-常见问题速答)

---

## 1. 第一性原理：FFN 存在的不可降维理由

### 1.1 核心矛盾：Attention 在通道维上是线性的

剥离所有「特征提取」「知识存储」之类的应用层叙事，FFN 存在的理由可以压缩成一个代数事实：

**自注意力对每个 token 的特征向量只做了线性变换与凸组合，它在通道（channel）维度上不引入任何非线性。**

设注意力权重矩阵 $A$（已经过 softmax，与 $V$ 的取值无关地看作系数），单头输出为：

$$\text{Attn}(X) = A\,X\,W_V W_O$$

关键观察：固定 $A$ 后，$X \mapsto A X W_V W_O$ 是**关于 $X$ 的线性映射**。
非线性只藏在 $A = \text{softmax}(QK^\top/\sqrt{d})$ 对 $X$ 的依赖里，
而那个非线性作用在 **token 之间的关系**上，不作用在**单个 token 的特征组合**上。

于是若把若干注意力层直接堆叠，在 $A$ 被冻结的意义上整个网络退化为一串矩阵乘的复合 ——
即一个线性模型。**必须有一个逐 token、跨通道的非线性算子来打破这种退化**，这就是 FFN 的全部理由。

> 一个更强的表述：Attention 负责 **token 间混合（mixing across positions）**，
> FFN 负责 **通道间混合（mixing across features）**。
> 两者正交，缺一不可。MetaFormer 系列工作正是把这个观察抽象成了
> 「token mixer + channel MLP」的统一模板。

### 1.2 第二个矛盾：为什么要先升维

若 FFN 写成 $W_2\,\sigma(W_1 x)$ 且 $W_1 \in \mathbb{R}^{H\times H}$（不升维），
则非线性 $\sigma$ 只能在 $H$ 个方向上做切分。**升维的本质是增加分段线性函数的「铰链数」。**

以 ReLU 为例：$\text{ReLU}(W_1 x)$ 中每一行 $w_i^\top x = 0$ 定义了输入空间中的一张超平面。
$I$ 个中间神经元 = $I$ 张超平面 = 把 $\mathbb{R}^H$ 切成指数多个线性区域。
**中间维 $I$ 直接决定了该层能表达的分段线性复杂度**，这是升维不可替代的原因。

$$\text{线性区域数上界} \sim \sum_{k=0}^{H}\binom{I}{k} \quad (\text{Zaslavsky 定理})$$

经验上 $I = 4H$ 成为惯例（原始 Transformer 论文取 $d_{\text{ff}} = 2048$，$d_{\text{model}} = 512$）。

### 1.3 第三个事实：FFN 占据了大部分参数与算力

| 组件 | 参数量 | 占比（$I=4H$，MHA） |
| --- | --- | --- |
| Attention（$W_Q,W_K,W_V,W_O$） | $4H^2$ | $\approx 33\%$ |
| FFN（$W_1, W_2$） | $2 \cdot H \cdot 4H = 8H^2$ | $\approx 67\%$ |

**FFN 是参数与 FLOPs 的主战场**。这解释了后续所有演进的动机：
GLU 在**同等参数下提升表达力**，MoE 在**同等 FLOPs 下提升参数量**。
两条路线都在攻击同一个约束 —— 参数量与计算量的强耦合。

---

## 2. 结构谱系总览

| 结构 | 公式 | 矩阵数 | 中间维惯例 | 代表模型 |
| --- | --- | --- | --- | --- |
| **Vanilla FFN (ReLU)** | $W_2\,\text{ReLU}(W_1x)$ | 2 | $4H$ | 原始 Transformer、BERT |
| **GELU FFN** | $W_2\,\text{GELU}(W_1x)$ | 2 | $4H$ | GPT-2/3、ViT |
| **GeGLU** | $W_d(\text{GELU}(W_gx)\odot W_ux)$ | 3 | $\frac{8}{3}H$ | T5 v1.1、Gemma |
| **SwiGLU** | $W_d(\text{SiLU}(W_gx)\odot W_ux)$ | 3 | $\frac{8}{3}H$ | LLaMA、Qwen、Mistral、DeepSeek |
| **ReGLU / Bilinear** | $W_d(\text{ReLU}(W_gx)\odot W_ux)$ / 无激活 | 3 | $\frac{8}{3}H$ | 消融实验对照组 |
| **MoE FFN** | $\sum_{i\in\text{TopK}} g_i \cdot \text{FFN}_i(x)$ | $3E{+}1$ | 每专家更小 | Mixtral、DeepSeek-V3、Qwen-MoE |
| **Shared + Routed MoE** | $\text{FFN}_{\text{shared}}(x) + \sum g_i\text{FFN}_i(x)$ | — | — | DeepSeek-V2/V3 |

$H$ = hidden_size，$I$ = intermediate_size，$E$ = 专家数。

---

## 3. Vanilla FFN：两层线性 + 逐点非线性

### 3.1 结构

$$\text{FFN}(x) = W_2\,\sigma(W_1 x + b_1) + b_2, \qquad W_1\in\mathbb{R}^{I\times H},\ W_2\in\mathbb{R}^{H\times I}$$

命名上 $W_1$ 常称 `up_proj` / `fc1` / `dense_h_to_4h`，$W_2$ 称 `down_proj` / `fc2`。

### 3.2 激活函数的演进链条

| 激活 | 公式 | 引入动机 | 缺陷 |
| --- | --- | --- | --- |
| ReLU | $\max(0,x)$ | 解决 sigmoid 梯度消失、计算极廉价 | 负半轴梯度恒 0（dying ReLU）；在 0 处不可导 |
| GELU | $x\,\Phi(x)$ | 平滑化：用「按概率保留」替代「硬阈值」 | 含 erf，需 tanh 近似 |
| SiLU/Swish | $x\,\sigma(x)$ | 同样平滑且非单调，形式比 GELU 更简单 | 需要 exp |

GELU 的概率解释值得展开：$\text{GELU}(x) = x\cdot P(Z\le x),\ Z\sim\mathcal{N}(0,1)$，
即「以输入自身的分位数为概率保留该输入」。这是**随机正则化（dropout）的确定性期望版本**。

**关键限制**：以上全部是 $\mathbb{R}\to\mathbb{R}$ 的**逐点标量函数**。
第 $j$ 个中间通道的输出只依赖第 $j$ 个中间通道的输入。
这是 vanilla FFN 表达力的天花板 —— 也正是 GLU 要突破的那条线。

---

## 4. GLU 家族：把激活从「标量函数」升级为「乘法门」

### 4.1 结构

$$\text{SwiGLU}(x) = W_{\text{down}}\Big(\underbrace{\text{SiLU}(W_{\text{gate}}\,x)}_{\text{门控信号}}\ \odot\ \underbrace{W_{\text{up}}\,x}_{\text{被门控的值}}\Big)$$

| 名称 | 形状 | 作用 |
| --- | --- | --- |
| `gate_proj` | $[I, H]$ | 过激活后作为**门控权重**，决定每个中间通道放多少信息通过 |
| `up_proj` | $[I, H]$ | 升维，提供**被门控的内容** |
| `down_proj` | $[H, I]$ | 降维回隐藏维度 |

`gate_proj` 与 `up_proj` **形状相同、输入相同**，仅训练所得权重与用途不同。这是它们可被合并的前提。

### 4.2 本质差异：引入了「输入依赖的二次项」

Vanilla FFN 的中间表示对 $x$ 是「线性 + 逐点非线性」；
GLU 的中间表示含 $x$ 的**乘性交互**：

$$h_j = \text{SiLU}(w_{g,j}^\top x)\cdot (w_{u,j}^\top x)$$

展开可见这是两个投影的乘积 —— 一个**双线性形式（bilinear form）**被激活函数调制。
双线性项 $x^\top(w_{g,j}w_{u,j}^\top)x$ 提供了逐点激活无法表达的**特征间乘性交互**。

这就是「门控」的真正含义：**用数据自身决定通路的开合**，
而非用一个固定的标量曲线压缩每个通道。当 $\text{SiLU}(\cdot)\approx 0$ 时该通道被关闭，
$\approx 1$ 时放行 —— 这是**数据相关的动态特征选择**。

### 4.3 为什么中间维取 $\frac{2}{3}$

三个矩阵而非两个，为保持参数量与 FLOPs 可比：

$$\underbrace{2 \cdot H \cdot 4H}_{\text{vanilla}} = \underbrace{3 \cdot H \cdot I}_{\text{GLU}} \implies I = \frac{8}{3}H \approx 2.67H$$

LLaMA-7B：$H=4096$，$I=11008 \approx 2.69H$（另有对齐到 256 倍数的取整）。
**所以 SwiGLU 并非「更大所以更好」，而是同等预算下的结构性增益** ——
这是 Noam Shazeer 在 *GLU Variants Improve Transformer* (2020) 中控制变量后的结论。

### 4.4 GLU 变体的消融

Shazeer 的实验中各变体差距很小（困惑度差异在 0.01~0.03 量级），
其中 SwiGLU / GeGLU 略优。论文结尾那句话极为诚实：

> "We offer no explanation as to why these architectures seem to work;
> we attribute their success, as all else, to divine benevolence."

即：**GLU 的优势是经验性的，缺乏理论解释**。这一点在阅读相关材料时值得保持清醒 ——
上文 4.2 的双线性分析是一种**事后合理化（post-hoc rationalization）**，不是被证明的因果机制。

---

## 5. 高阶结构化类比

### 5.1 SwiGLU ≙ 数字电路中的「传输门 / 三态缓冲器」

| 数字电路 | SwiGLU |
| --- | --- |
| 数据总线 `D` | $W_{\text{up}}x$（待传输的值） |
| 使能信号 `OE`（output enable） | $\text{SiLU}(W_{\text{gate}}x)$（门控） |
| 三态缓冲器输出 `D & OE` | $\odot$ 逐元素乘 |
| 汇流至下游总线 | $W_{\text{down}}$ 的降维求和 |

拓扑同构点：**控制通路与数据通路物理分离，且控制信号本身由数据产生**。
Vanilla FFN 相当于把数据直接接到一个固定的非线性衰减器上 —— 没有独立的控制通路。

这个类比还解释了为什么 GLU 的门控是**软的**（$[0,1]$ 连续）而非三态门的硬开关：
硬开关不可导，无法反向传播。SiLU 是三态使能信号的**可微松弛（differentiable relaxation）**。

### 5.2 MoE ≙ 数据库的「分区表 + 分区裁剪」

| 数据库 | MoE |
| --- | --- |
| 分区表（partitioned table） | $E$ 个专家 FFN |
| 分区键（partition key） | Router 的 $\arg\text{TopK}$ |
| 分区裁剪（partition pruning） | 只激活 TopK 个专家 |
| 全表大小 ≫ 单次扫描量 | 总参数量 ≫ 激活参数量 |
| 数据倾斜（data skew） | 专家负载不均衡 |
| 分区键选择不当 → 全表扫描 | Router 坍塌 → 退化为稠密模型 |

同构核心：**用一个廉价的路由决策，把「存储总量」与「单次访问量」解耦**。
数据库的分区裁剪让 TB 级表的点查只扫描 MB 级数据；
MoE 让 671B 参数的模型每 token 只激活 37B（DeepSeek-V3）。

「数据倾斜」这个映射尤其精确：数据库需要 rebalance，MoE 需要
auxiliary load-balancing loss 或 DeepSeek-V3 的 bias 调整策略。

---

## 6. MoE：把参数量与计算量解耦

### 6.1 结构

$$\text{MoE}(x) = \sum_{i \in \text{TopK}(x)} g_i(x)\cdot \text{FFN}_i(x),
\qquad g(x) = \text{softmax}\big(\text{TopK}(W_r x)\big)$$

其中每个 $\text{FFN}_i$ 通常就是一个独立的 SwiGLU（三矩阵结构）。
Router $W_r \in \mathbb{R}^{E\times H}$ 是唯一的稠密部分，参数量可忽略。

### 6.2 打破的假设

Vanilla / GLU FFN 隐含假设：**所有 token 走同一条计算通路，参数量 $\propto$ 计算量**。
MoE 推翻的正是这条：

$$\text{参数量} = E \cdot 3HI \qquad\text{而}\qquad \text{FLOPs} \propto K \cdot 3HI,\quad K \ll E$$

以 DeepSeek-V3 为例：$E=256$ 路由专家 + 1 共享专家，$K=8$，
总参 671B 但每 token 仅激活 37B —— **参数量放大 18 倍，FLOPs 几乎不变**。

### 6.3 Shared + Routed 混合设计

DeepSeek-V2/V3 的关键改良：保留一个**始终激活的 shared expert**：

$$\text{MoE}(x) = \text{FFN}_{\text{shared}}(x) + \sum_{i\in\text{TopK}} g_i\,\text{FFN}_i(x)$$

动机是**职责分离**：通用知识（语法、常识）由 shared expert 稳定承载，
routed experts 专注领域特化。这缓解了纯 routed MoE 中「每个专家都不得不重复学习通用模式」
造成的参数冗余。

> 工程注记：SGLang 中可通过 `--disable-shared-experts-fusion` 控制是否把
> shared expert 融合进 routed experts 的 grouped GEMM（见 `deepseek_v2.py` 中的
> `num_fused_shared_experts`）。融合后 shared expert 被重映射为
> `mlp.experts.256`，从而与 routed experts 共享同一个 MoE kernel，减少 kernel 启动与显存往返。

### 6.4 MoE 的真实代价

| 代价 | 说明 |
| --- | --- |
| **显存墙** | 全部专家权重必须常驻（或动态换入），显存需求按总参数量而非激活参数量计 |
| **通信墙** | EP（专家并行）下需 All-to-All 分发 token 与回收结果，通信量随 $K$ 与序列长度增长 |
| **负载不均衡** | 热门专家成为长尾瓶颈，需 capacity factor / drop token / aux loss 干预 |
| **batch 效率下降** | 每个专家实际拿到的 token 数 $\approx \frac{BK}{E}$，GEMM 变小、Tensor Core 利用率下降 |

**MoE 是「显存换算力」而非「免费提升」** —— 这是最容易被忽略的等价交换。

---

## 7. 边界与反模式

### 7.1 概念边界：与最邻近方案的分界线

| 对比 | 分界线划在哪里 |
| --- | --- |
| **GLU vs 逐点激活** | 中间表示是否含 $x$ 的**乘性交互项**。逐点激活下 $h_j = \sigma(w_j^\top x)$；GLU 下 $h_j$ 是两个投影的乘积。这是表达力的质变，不是宽度的量变 |
| **GLU vs Attention 门控** | GLU 的门控**逐 token 独立**（不跨位置）；Attention 的权重跨位置。GLU 永远不改变 token 间的信息流 |
| **MoE vs Ensemble** | Ensemble 全部成员都执行后平均；MoE **条件执行**（TopK 稀疏）。前者 FLOPs $\propto E$，后者 $\propto K$ |
| **MoE vs Dropout** | 两者都是「随机/稀疏地关闭子网络」，但 Dropout 的掩码**与输入无关且仅训练时启用**；MoE 的路由**由输入决定且训练推理一致** |
| **FFN vs 1×1 Conv** | 在数学上**完全等价**（都是逐位置的通道混合）。差别仅在张量布局与命名传统 —— 这不是边界，是同一个东西的两个名字 |

### 7.2 极限边界（Edge Cases）

| 极端情况 | 后果 |
| --- | --- |
| $I \to H$（不升维） | 分段线性区域数塌缩，FFN 退化为「线性 + 轻微弯曲」，深度无法换取表达力 |
| $I \to \infty$ | 单层即万能逼近器，但参数与显存爆炸；且实践上远不如「更深 + 适中宽度」 |
| GLU 中 $W_{\text{gate}} \equiv 0$ | $\text{SiLU}(0)=0$，**整层输出恒为 0**，梯度完全断流 |
| GLU 不加激活（Bilinear） | 仍保留乘性交互，实测困惑度仅略差于 SwiGLU —— 说明**乘法门本身比激活函数的具体形状更重要** |
| MoE 中 $K = E$ | 退化为「加权稠密 ensemble」，丧失全部稀疏收益，FLOPs 反而高于稠密模型 |
| MoE 中 $K = 1$ | Switch Transformer 设定；路由决策不可微性最强，训练最不稳定 |
| MoE 中 Router 输出趋于均匀 | 无有效专业化，等价于参数量被浪费的稠密模型 |

### 7.3 灾难性反模式

**反模式 1：迁移 SwiGLU 时不缩放中间维**

把 vanilla FFN 的 $I=4H$ 直接套给 SwiGLU，参数量凭空上涨 50%
（$3\times4H^2$ vs $2\times4H^2$）。此时若报告「SwiGLU 更强」，
结论完全无效 —— 增益来自参数量而非结构。**这是 GLU 相关消融中最常见的实验污染。**

**反模式 2：TP 下把 gate 与 up 切到不同 rank**

因 $\text{SiLU}(\text{gate}_j)\odot\text{up}_j$ 是**逐元素**运算，
第 $j$ 个中间通道的 gate 与 up 必须在**同一张卡**上。
若按「先 gate 全部、再 up 全部」的朴素方式沿合并后的 $2I$ 维等分切分：

```
错误切法（tp_size=2）：
  rank0: [ gate[0:I] ]              ← 只有 gate，没有对应的 up
  rank1: [ up[0:I]   ]              ← 只有 up，没有对应的 gate
→ 逐元素乘需要跨卡通信，或直接算出错误结果
```

正确切法必须让每个 rank 各持两段的对应一半（见第 8 节）。
**这就是 `MergedColumnParallelLinear.weight_loader` 需要两层偏移的根本原因。**

**反模式 3：把 shared expert 当成「又一个 routed expert」**

shared expert 的语义是**无条件激活**。若把它塞进 TopK 候选池，
它可能在某些 token 上未被选中，通用能力随即失稳。融合实现（如 SGLang 的
`num_fused_shared_experts`）必须保证它**恒定被选中**，融合只是 kernel 层面的复用。

**反模式 4：MoE 显存预算按激活参数量估算**

「37B 激活参数」不等于「37B 显存」。全部 671B 权重必须可访问。
按激活量做容量规划会在加载阶段直接 OOM。

---

## 8. 张量并行下的切分语义

### 8.1 SwiGLU MLP 的标准 TP 方案

```
x  ────► gate_up_proj (ColumnParallel, 合并两段)  ────► SiLU⊙Mul ────► down_proj (RowParallel) ────► all-reduce ──► out
         输出沿列切分，无通信                              逐元素，无通信      输入沿行切分，无通信         唯一一次通信
```

关键性质：**列并行的切分输出可直接作为行并行的切分输入**，
中间零通信，整个 MLP 只需末尾一次 all-reduce。

### 8.2 合并权重的正确切分布局

`MergedColumnParallelLinear(input_size=H, output_sizes=[I, I])`，$tp\_size=2$：

```
全局列布局:  [ gate(0 .. I) | up(I .. 2I) ]

rank0 持有:  [ gate[0   : I/2] , up[0   : I/2] ]
rank1 持有:  [ gate[I/2 : I  ] , up[I/2 : I  ] ]
```

**每个 rank 同时抽取每一段的一部分**，而非整段归属某个 rank。
这保证了逐元素乘所需的 gate/up 通道配对始终在同卡。

由此产生权重加载中的两层偏移：

| 偏移 | 含义 | 计算 |
| --- | --- | --- |
| `shard_offset` | 本段在**本 rank 合并参数**中的起始位置 | $\text{prefix\_sum}(\text{output\_sizes}) / tp\_size$ |
| `start_idx` | 在**全局的本段权重**中本 rank 该取哪一截 | $tp\_rank \times shard\_size$ |

对应实现见 `python/sglang/srt/layers/linear.py` 中
`MergedColumnParallelLinear.weight_loader`。

### 8.3 通信量对比

| 结构 | 每层 all-reduce 次数 | 单次通信量 |
| --- | --- | --- |
| Vanilla FFN（Col→Row） | 1 | $B\cdot S\cdot H$ |
| SwiGLU（Col合并→Row） | 1 | $B\cdot S\cdot H$ |
| 若误用 Col→Col（需 gather） | 2 | 中间态 $B\cdot S\cdot I$，更大 |

**SwiGLU 相比 vanilla 不增加通信次数** —— 三个矩阵但仍只有一次归约，
因为 gate/up 共享同一次列并行、down 承担唯一的行并行归约。

---

## 9. SGLang 中的实现

### 9.1 融合激活算子

`python/sglang/srt/layers/activation.py` 中的 `SiluAndMul` 与 `GeluAndMul`
把「劈半 + 激活 + 逐元素乘」融合为单个 kernel：

输入 $[T, 2I]$ → 输出 $[T, I]$，无需显式 `chunk`。

收益：
- 省掉一次 $[T,2I]$ 的中间张量读写（HBM 带宽是 decode 阶段的主要瓶颈）
- 减少 kernel 启动开销
- `chunk` 产生的非连续视图不利于后续 GEMM，融合后直接输出连续张量

### 9.2 典型模型的组装

以 `python/sglang/srt/models/llama.py` 的 `LlamaMLP` 为例，结构为：

- `gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False)`
- `down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)`
- `act_fn = SiluAndMul()`

前向即 `gate_up → act_fn → down_proj`，与第 8.1 节的通信拓扑完全对应。

### 9.3 checkpoint 权重名映射

HuggingFace checkpoint 里 `gate_proj` 与 `up_proj` 是**分开存储**的，
需在加载时映射到合并参数的对应段。`llama.py` 中的映射表：

| checkpoint 权重名 | 合并后参数 | shard_id |
| --- | --- | --- |
| `.gate_proj` | `.gate_up_proj` | 0 |
| `.up_proj` | `.gate_up_proj` | 1 |
| `.q_proj` | `.qkv_proj` | 0 (`"q"`) |
| `.k_proj` | `.qkv_proj` | 1 (`"k"`) |
| `.v_proj` | `.qkv_proj` | 2 (`"v"`) |

`shard_id` 正是传给 `weight_loader(param, weight, shard_id)` 的第三个参数，
用于定位第 8.2 节中的 `shard_offset`。

### 9.4 Gemma 的差异点

`gemma2.py` 同样使用 `MergedColumnParallelLinear`，但激活是
`gelu_pytorch_tanh`（GeGLU 而非 SwiGLU），且代码显式校验
`hidden_act == hidden_activation == "gelu_pytorch_tanh"` ——
因为 HF config 中曾存在这两个字段不一致导致静默用错激活的历史问题。
**激活函数选错不会报错，只会让输出质量缓慢劣化**，故用断言前置暴露。

---

## 10. 语义压缩：一句话本质

> **FFN 是给每个 token 单独配备的一台「先展开再压回」的非线性混音台；
> GLU 给它加了一路由信号自己控制的音量推子；
> MoE 则把一台混音台换成一排，每次只推开其中几台。**

去掉全部类比的最硬版本：

> **注意力只在位置之间做线性混合，所以必须有一个逐位置、跨通道的非线性算子来阻止网络退化成线性模型；
> 这个算子的宽度决定表达力上限，它的门控形式决定表达效率，它的稀疏化程度决定参数量能否脱离算力增长。**

---

## 11. 常见问题速答

**Q：为什么 SwiGLU 里 `gate_proj` 和 `up_proj` 谁是谁？换一下有区别吗？**

有区别，但**不对称性完全由训练建立**。$\text{SiLU}$ 只作用在 `gate` 分支上，
两者在初始化时统计同分布，训练会自然分化出门控者与被门控者的角色。
但推理时若把加载的两段权重张冠李戴，等价于把 SiLU 作用到了错误的分支，输出会明显劣化 ——
这类 bug 不会抛异常，只表现为生成质量下降，需靠权重名映射表严格保证。

**Q：SwiGLU 的 3 个矩阵能否合并成 1 个大矩阵？**

`gate` 与 `up` 可以（形状相同、输入相同、都在激活之前），这正是 `gate_up_proj`。
`down` 不能 —— 它在逐元素乘**之后**，输入是 $[T,I]$ 而非 $[T,H]$，且承担行并行归约。

**Q：为什么不干脆用两层 GLU 堆叠？**

FFN 的深度收益远不如「注意力 + FFN」交替。堆两层 GLU 会破坏
「token mixing / channel mixing 交替」的节奏，且中间又需要一次归一化与残差，
参数效率不如直接加一整个 Transformer block。

**Q：MoE 里每个专家一定是 SwiGLU 吗？**

不一定，但现代 MoE 模型（Mixtral、DeepSeek、Qwen-MoE）的专家几乎都是 SwiGLU。
MoE 与 GLU 是**正交的两个维度**：GLU 改进单个 FFN 的结构，MoE 决定有多少个 FFN 以及如何选择。

**Q：$I = \frac{8}{3}H$ 为什么实际值常不是整数倍？**

需对齐到硬件友好的倍数（常见 128 或 256），以及 TP 下必须被 $tp\_size$ 整除。
LLaMA-7B 的 $11008 = 43\times256$，既接近 $\frac{8}{3}\times4096\approx10923$，又满足对齐与整除。
`MergedColumnParallelLinear.__init__` 中的
`assert all(output_size % tp_size == 0 ...)` 正是在守这条约束。

**Q：bias 为什么大多设为 `False`？**

LLaMA 系列去掉了 FFN 与 Attention 投影的 bias。经验上对表达力影响可忽略，
但省去参数、减少一次 kernel、且在 TP 下避免 all-reduce 时 bias 被重复累加的处理成本
（行并行中需仅在 rank 0 加 bias，见 `RowParallelLinear.forward`）。

---

## 参考

- Vaswani et al., *Attention Is All You Need* (2017) —— vanilla FFN
- Dauphin et al., *Language Modeling with Gated Convolutional Networks* (2017) —— GLU 原型
- Shazeer, *GLU Variants Improve Transformer* (2020) —— SwiGLU/GeGLU 消融
- Hendrycks & Gimpel, *Gaussian Error Linear Units* (2016) —— GELU
- Shazeer et al., *Outrageously Large Neural Networks* (2017) —— Sparse MoE
- Fedus et al., *Switch Transformers* (2021) —— $K=1$ 路由
- DeepSeek-AI, *DeepSeek-V2 / V3 Technical Report* —— shared + routed MoE
- Yu et al., *MetaFormer Is Actually What You Need for Vision* (2022) —— token/channel mixing 抽象

相关代码：
- `python/sglang/srt/layers/activation.py` —— `SiluAndMul` / `GeluAndMul`
- `python/sglang/srt/layers/linear.py` —— `MergedColumnParallelLinear` / `RowParallelLinear`
- `python/sglang/srt/models/llama.py` —— `LlamaMLP` 与权重映射
- `python/sglang/srt/models/deepseek_v2.py` —— MoE 与 shared expert 融合

