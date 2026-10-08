# MoE 门控与负载均衡：路由机制的数学原理与演进

> 姊妹篇 [`MoE.md`](MoE.md) 从第一性原理与系统工程视角回答了「MoE 为什么存在、代价从哪里来」。
> 本文聚焦另一条线索：**门控（gating）函数本身的数学结构**，以及它如何在保持可微、可训练的同时，
> 逐步演化出对硬件友好、对负载均衡友好的形态。
>
> 换句话说，`MoE.md` 关心「稀疏激活这条约束」，本文关心「实现这条约束的那个可学习函数 $g(\cdot)$ 到底长什么样、怎么求导、怎么被正则」。
> 阅读顺序建议：先读 `MoE.md` 建立系统直觉，再读本文补齐门控与均衡的数学细节。

## 目录

1. [门控函数的通用形式](#1-门控函数的通用形式)
2. [Top-K 的不可微性与梯度如何穿过它](#2-top-k-的不可微性与梯度如何穿过它)
3. [Noisy Top-K：用噪声换探索与均衡](#3-noisy-top-k用噪声换探索与均衡)
4. [负载均衡损失的推导](#4-负载均衡损失的推导)
5. [Token-Choice 与 Expert-Choice：两种对偶的分配视角](#5-token-choice-与-expert-choice两种对偶的分配视角)
6. [从 Softmax 到 Sigmoid：无辅助损失均衡](#6-从-softmax-到-sigmoid无辅助损失均衡)
7. [共享专家与细粒度专家](#7-共享专家与细粒度专家)
8. [代表性模型的门控设计对照](#8-代表性模型的门控设计对照)

---

## 1. 门控函数的通用形式

设 token 表示为 $x \in \mathbb{R}^{d_{\text{model}}}$，共 $E$ 个专家。所有 MoE 门控都可以写成三段式：

$$
\underbrace{s = f_{\text{score}}(x)}_{\text{打分}}
\;\longrightarrow\;
\underbrace{S = f_{\text{select}}(s)}_{\text{选择}}
\;\longrightarrow\;
\underbrace{g = f_{\text{weight}}(s, S)}_{\text{加权}}
$$

其中 $s \in \mathbb{R}^{E}$ 是专家分数，$S \subseteq \{1,\dots,E\}$ 是被选中的专家集合，$g \in \mathbb{R}^{E}$ 是稀疏的门控权重（未选中的分量为 $0$）。最终层输出为：

$$
y = \sum_{e \in S} g_e \operatorname{Expert}_e(x)
$$

不同 MoE 方案的差异，几乎都可以归结为这三个函数的选择：

| 组件 | 经典选项 | 后续演进 |
|---|---|---|
| $f_{\text{score}}$ | $s = \operatorname{softmax}(W_r x)$ | 加噪声、改用 $\operatorname{sigmoid}$、分组打分 |
| $f_{\text{select}}$ | $\operatorname{TopK}(s, K)$ | Expert-Choice、Hash、可学习阈值 |
| $f_{\text{weight}}$ | 保留 softmax 原值 | Top-K 内重归一化、乘可学习缩放 |

本文的核心论点：**MoE 的大部分算法创新，不是发明新的专家结构，而是在改造这三个函数，使其同时满足「可微、可均衡、可高效执行」这三个彼此拉扯的目标。**

### 1.1 最朴素的门控：softmax over experts

最早（Shazeer 等，2017）的门控直接对专家维做 softmax：

$$
g_e = \frac{\exp\big((W_r x)_e\big)}{\sum_{j=1}^{E}\exp\big((W_r x)_j\big)}
$$

若不加选择（$S = \{1,\dots,E\}$），这就是一个稠密加权混合，$\sum_e g_e = 1$，完全可微，但没有任何稀疏收益——它等价于把 $E$ 个 FFN 全算一遍再加权。稀疏性只有在引入 $\operatorname{TopK}$ 后才出现，而 $\operatorname{TopK}$ 恰恰是不可微的。这就引出第 2 节的核心张力。

---

## 2. Top-K 的不可微性与梯度如何穿过它

### 2.1 为什么 Top-K 不可微

$\operatorname{TopK}$ 输出的是一个离散的索引集合。考虑两个专家分数 $s_1, s_2$，选择结果为「取较大者」。当 $s_1$ 从略小于 $s_2$ 连续增大到略大于 $s_2$ 时，被选集合会在某一点**瞬间跳变**：

$$
S =
\begin{cases}
\{2\}, & s_1 < s_2 \\
\{1\}, & s_1 > s_2
\end{cases}
$$

这个跳变点处不存在导数，且集合 $S$ 对 $s$ 的雅可比在几乎所有点都为零（局部微小扰动不改变谁被选中）。因此不能直接对「选择动作」求梯度。

### 2.2 权重通路承载梯度

MoE 的关键技巧：**梯度不流经「选谁」，只流经「选中者的权重 $g_e$」。** 对被选中的专家 $e \in S$，输出对分数的偏导为：

$$
\frac{\partial y}{\partial s_e}
=
\frac{\partial g_e}{\partial s_e}\operatorname{Expert}_e(x)
$$

由于 $g_e$ 是 $s_e$ 的可微函数（softmax 或其重归一化），这条路径是可微的。于是 Router 权重 $W_r$ 能收到梯度信号：

$$
\frac{\partial \mathcal{L}}{\partial W_r}
=
\sum_{e \in S}
\frac{\partial \mathcal{L}}{\partial y}
\frac{\partial g_e}{\partial s_e}
\frac{\partial s_e}{\partial W_r}
\operatorname{Expert}_e(x)
$$

这带来一个深刻后果：**未被选中的专家收不到任何梯度。** 一个从未被选中的专家永远不会更新，也就永远不会变得「更该被选」——这是专家塌缩（expert collapse）的数学根源，也是负载均衡损失（第 4 节）存在的根本原因。

### 2.3 直觉总结

$$
\boxed{\text{Top-K 提供稀疏结构，softmax 权重提供可微梯度；两者分工，缺一不可}}
$$

如果去掉 softmax 权重、直接用硬 one-hot，梯度会彻底断掉；如果去掉 Top-K，稀疏性消失。现代门控都在这条钢丝上做文章。

---

## 3. Noisy Top-K：用噪声换探索与均衡

### 3.1 动机

第 2.2 节的塌缩问题需要一种机制，让「暂时分数略低」的专家也有机会被选中，从而获得梯度、参与竞争。Shazeer 等提出在打分时注入可学习的高斯噪声：

$$
s_e = (W_r x)_e + \varepsilon_e \cdot \operatorname{Softplus}\big((W_{\text{noise}} x)_e\big),
\qquad
\varepsilon_e \sim \mathcal{N}(0, 1)
$$

其中 $W_{\text{noise}}$ 是可学习的、决定每个专家噪声幅度的参数。含义是：

- 训练早期噪声较大，路由更「随机」，让所有专家都有机会被激活并接收梯度；
- 随训练推进，网络可以学到把噪声幅度调小，让路由趋于确定。

再对加噪后的分数取 Top-K 并 softmax：

$$
g = \operatorname{softmax}\big(\operatorname{KeepTopK}(s, K)\big),
\qquad
\operatorname{KeepTopK}(s,K)_e =
\begin{cases}
s_e, & e \in \operatorname{TopK}(s,K) \\
-\infty, & \text{otherwise}
\end{cases}
$$

把未选中位置置为 $-\infty$，softmax 后自然归零，且梯度只在选中集合内流动。

### 3.2 噪声作为一种隐式正则

噪声等价于在离散选择前对分数做随机平滑：分数接近的专家，被选中的概率也接近，而非硬性由第 $K$ 名和第 $K{+}1$ 名的微小差距一刀切。这在统计上缓解了「赢者通吃」，是**架构层面**的均衡手段（相对于第 4 节的**损失层面**手段）。

---

## 4. 负载均衡损失的推导

即使有噪声，梯度反馈仍会自我强化热门专家。工程上通常再加一个显式的**辅助损失**（auxiliary loss）$\mathcal{L}_{\text{balance}}$，与任务损失联合优化：

$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \, \mathcal{L}_{\text{balance}}
$$

### 4.1 两个需要区分的量：importance 与 load

对一个批次 $B$（含 $T$ 个 token），定义两个统计量：

**Importance（重要性）**——专家 $e$ 在批次内收到的门控权重之和，是连续、可微的：

$$
\operatorname{Imp}_e = \sum_{t \in B} g_{t,e}
$$

**Load（负载）**——专家 $e$ 实际被选中的 token 数，是离散、不可微的：

$$
\operatorname{Load}_e = \sum_{t \in B} \mathbf{1}[e \in S_t]
$$

理想均衡希望二者在专家间都尽量均匀。但 $\operatorname{Load}_e$ 不可微，不能直接进损失，所以早期做法是约束 $\operatorname{Imp}_e$。

### 4.2 用变异系数约束 importance

Shazeer 等用重要性的**变异系数平方**（$\mathrm{CV}^2$，方差除以均值平方）作为损失，鼓励各专家重要性相等：

$$
\mathcal{L}_{\text{imp}}
=
\left(
\frac{\operatorname{std}_e(\operatorname{Imp}_e)}{\operatorname{mean}_e(\operatorname{Imp}_e)}
\right)^2
=
\frac{\operatorname{Var}_e(\operatorname{Imp}_e)}{\big(\operatorname{mean}_e \operatorname{Imp}_e\big)^2}
$$

$\mathrm{CV}^2$ 是无量纲的：它只惩罚「不均匀」而不惩罚「整体大小」，因此不会干扰任务损失对 $g$ 绝对尺度的需求。

### 4.3 Switch Transformer 的简化损失

Switch Transformer（$K=1$）给出一个被广泛沿用的可微近似。定义批次内：

- $f_e$：被路由到专家 $e$ 的 token 占比（源自离散 load）

$$
f_e = \frac{1}{T}\sum_{t \in B}\mathbf{1}[\,\arg\max_j g_{t,j} = e\,]
$$

- $P_e$：路由到专家 $e$ 的平均门控概率（连续、可微，作为 load 的可微代理）

$$
P_e = \frac{1}{T}\sum_{t \in B} g_{t,e}
$$

辅助损失取二者内积并乘以专家数：

$$
\mathcal{L}_{\text{balance}} = E \cdot \sum_{e=1}^{E} f_e \, P_e
$$

**为什么这个式子有效：** $f_e$ 提供不可微的实际负载信息（作为常数系数，不回传梯度），$P_e$ 提供可微的概率信号（承载梯度）。当某专家实际负载 $f_e$ 偏高时，损失会通过 $P_e$ 把该专家的门控概率往下压。在完全均匀时 $f_e = P_e = 1/E$，损失达到最小值 $1$：

$$
\mathcal{L}_{\text{balance}}^{\min} = E \cdot \sum_{e=1}^{E}\frac{1}{E}\cdot\frac{1}{E} = 1
$$

系数 $E$ 使最小值与专家数无关，便于跨配置比较与设定 $\lambda$。

### 4.4 均衡损失的本质：把「离散公平」松弛为「连续公平」

$$
\boxed{\underbrace{\operatorname{Load}_e \text{ 均匀}}_{\text{真正想要、不可微}}
\;\;\xrightarrow{\text{松弛}}\;\;
\underbrace{P_e \text{ 均匀}}_{\text{可微代理}}}
$$

所有辅助损失的共同套路，都是找一个可微的概率量去逼近不可微的计数量。代价是：它是**代理目标**，过强会牺牲任务性能（对应 `MoE.md` 反模式 3）。这就是第 6 节「无辅助损失」路线的动机。

---

## 5. Token-Choice 与 Expert-Choice：两种对偶的分配视角

前面所有方案都是 **token-choice**：每个 token 选自己的 Top-K 专家。它有一个结构性缺陷——**无法保证专家侧负载均衡**，因为选择权在 token 手里，热门专家可能被大量 token 同时选中而溢出。

### 5.1 视角对偶

把路由看成 token 与专家之间的二部图分配问题。两种方向：

| 视角 | 谁做选择 | 保证什么 | 代价 |
|---|---|---|---|
| Token-Choice | 每个 token 选 $K$ 个专家 | 每 token 恰好 $K$ 个专家 | 专家负载可能倾斜，需容量因子/辅助损失 |
| Expert-Choice | 每个专家选 $C$ 个 token | 每专家恰好 $C$ 个 token（**天然均衡**） | 某些 token 可能被 $0$ 个专家选中，另一些被多选 |

Expert-Choice（Zhou 等，2022）把分配矩阵转置来看：不再是每行（token）取 Top-K，而是每列（专家）取 Top-$C$：

$$
\text{Token-Choice: } S_t = \operatorname{TopK}_e(s_{t,e}, K) \quad \text{(沿专家维)}
$$

$$
\text{Expert-Choice: } S_e = \operatorname{TopK}_t(s_{t,e}, C) \quad \text{(沿 token 维)}
$$

其中容量 $C = \gamma \cdot T K / E$。

### 5.2 权衡

Expert-Choice **不需要辅助负载均衡损失**，因为每个专家精确处理 $C$ 个 token，负载天然完美均衡，GEMM 形状也固定规则。但它放弃了「每个 token 保证被处理 $K$ 次」的性质：一个 token 可能被很多专家争抢，也可能一个都没被选中（表示为该层近似残差直通）。这在自回归解码中较难使用，因为解码时是逐 token 生成、缺乏批内的 token 维竞争。

$$
\boxed{\text{Token-Choice 保证 token 覆盖但需治理专家倾斜；Expert-Choice 保证专家均衡但放弃 token 覆盖}}
$$

---

## 6. 从 Softmax 到 Sigmoid：无辅助损失均衡

### 6.1 Softmax 门控的两个副作用

跨专家的 softmax 有内在耦合：$\sum_e g_e = 1$ 意味着提高一个专家分数必然压低其余专家。在专家数很大（细粒度 MoE，$E$ 可达数百）时，单个专家分到的概率被稀释，梯度信号变弱。此外辅助损失 $\lambda \mathcal{L}_{\text{balance}}$ 会与任务损失竞争，$\lambda$ 过大伤性能、过小不均衡，调参困难。

### 6.2 Sigmoid 门控 + 独立打分

DeepSeek-V3 等采用**逐专家独立的 sigmoid** 打分，去掉跨专家归一化：

$$
s_e = \sigma\big((W_r x)_e\big) = \frac{1}{1 + \exp(-(W_r x)_e)}
$$

每个专家的分数不再彼此挤占，然后对选中的 Top-K 分数做归一化得到权重：

$$
g_e = \frac{s_e}{\sum_{j \in S}s_j}, \qquad e \in S = \operatorname{TopK}(s, K)
$$

### 6.3 无辅助损失负载均衡（aux-loss-free）

DeepSeek-V3 进一步提出用一个**可动态调整的偏置项** $b_e$ 只影响「选择」、不影响「权重」，从而在不引入梯度竞争的前提下均衡负载。选择依据带偏置的分数：

$$
S = \operatorname{TopK}\big(s_e + b_e,\; K\big)
$$

但最终权重仍用原始 $s_e$（偏置只用于挑选，不进入 $g_e$）。偏置 $b_e$ 不由梯度更新，而由一条简单的**反馈控制规则**在每步后调整：

$$
b_e \leftarrow b_e + u \cdot \operatorname{sign}\big(\bar{N} - N_e\big)
$$

其中 $N_e$ 是该步专家 $e$ 的实际负载，$\bar{N} = TK/E$ 是目标平均负载，$u$ 是更新步长。含义直白：

- 专家过载（$N_e > \bar{N}$）→ 降低其偏置 → 下一步更难被选中；
- 专家欠载（$N_e < \bar{N}$）→ 提高其偏置 → 下一步更容易被选中。

$$
\boxed{\text{把负载均衡从「梯度里的软惩罚」改为「选择前的确定性配额调节」，避免污染任务梯度}}
$$

这是控制论式的均衡：偏置像一个 PID 控制器的比例项，把系统推向均衡点，而完全不占用反向传播的梯度预算。

---

## 7. 共享专家与细粒度专家

DeepSeekMoE 提出两项与门控紧密相关的结构改进，目标是提高专家专门化程度、减少知识冗余。

### 7.1 细粒度专家切分（fine-grained expert segmentation）

把每个专家的中间维 $d_{\text{ff}}$ 切成 $m$ 份，专家数变为 $mE$，同时把每 token 激活数从 $K$ 提到 $mK$，**保持总激活参数量不变**：

$$
\text{原始: } E \text{ 专家, 激活 } K
\quad\Longrightarrow\quad
\text{细粒度: } mE \text{ 专家, 激活 } mK
$$

激活 FLOPs 近似不变，但可选组合数从 $\binom{E}{K}$ 暴增到 $\binom{mE}{mK}$：

$$
\binom{mE}{mK} \gg \binom{E}{K}
$$

更多的组合意味着更精细的知识分工，每个专家可以更专门化。这解释了为什么现代 MoE 倾向「很多小专家」而非「少数大专家」。

### 7.2 共享专家（shared expert isolation）

保留 $K_s$ 个**始终被激活**的共享专家，处理所有 token 共有的通用知识；其余为需要路由的专属专家：

$$
y = \sum_{i=1}^{K_s}\operatorname{Expert}^{\text{shared}}_i(x)
\;+\;
\sum_{e \in S}g_e \operatorname{Expert}^{\text{routed}}_e(x)
$$

直觉：不同专家之间往往重复学习「所有 token 都需要的公共变换」，浪费容量。把公共部分抽出到共享专家后，路由专家可以专注于差异化知识，减少冗余、提升专门化。共享专家无需路由决策，也降低了 Router 需要承载的公共负担。

---

## 8. 代表性模型的门控设计对照

把前面各节的机制落到具体模型上，可以看到一条清晰的演进主线：**打分函数从 softmax 走向 sigmoid，均衡手段从辅助损失走向确定性配额，专家从少而大走向多而细。**

| 模型 | 打分 $f_{\text{score}}$ | 选择 $f_{\text{select}}$ | 均衡手段 | 专家结构特征 |
|---|---|---|---|---|
| Shazeer 2017 (Sparsely-Gated) | softmax + 可学习高斯噪声 | Noisy Top-K | importance/load 的 $\mathrm{CV}^2$ 损失 | LSTM 之间的大 MoE 层 |
| GShard | softmax | Top-2 + 容量因子 | 辅助负载损失 + 随机第二选择 | 每层替换 FFN，跨设备分片 |
| Switch Transformer | softmax | Top-1 | $E\sum f_e P_e$ 辅助损失 | 极简 Top-1，强调工程可扩展 |
| Expert-Choice | softmax | 每专家 Top-$C$（沿 token 维） | 结构性均衡，**无需辅助损失** | 天然固定 GEMM 形状 |
| DeepSeekMoE / V2 | softmax | Top-K | 辅助损失 + 设备级均衡 | 细粒度专家 + 共享专家 |
| DeepSeek-V3 | **sigmoid**（逐专家独立） | Top-K（含偏置 $b_e$） | **无辅助损失**（偏置反馈调节） | 细粒度 + 共享专家，数百专家 |

### 8.1 演进主线的一句话压缩

> **早期 MoE 把「可微」放在第一位（softmax 门控 + 辅助损失软约束），代价是均衡靠正则、易与任务损失冲突；
> 现代 MoE 把「可执行、可均衡」提到同等位置——用独立 sigmoid 打分解耦专家竞争、用确定性偏置把均衡挪出梯度、用细粒度专家换取更强专门化，
> 从而让一个可学习的门控函数同时满足可微、可均衡、可高效执行这三个原本互相拉扯的目标。**

### 8.2 与 `MoE.md` 的呼应

- 本文第 4 节的辅助损失，正是 `MoE.md` 反模式 3「强迫绝对均匀破坏专门化」在数学上的来源与治理对象；
- 本文第 6 节的无辅助损失偏置，回应了 `MoE.md` 中「Router 同时优化 $\mathcal{L}_{\text{task}}$ 与 $\mathcal{L}_{\text{balance}}$ 可能冲突」的张力；
- 本文第 5 节 Expert-Choice 的「token 可能被 0 个专家选中」，对应 `MoE.md` 情形 D/容量三角约束中的信息损失取舍；
- 本文第 2.2 节的「未选中专家收不到梯度」，是 `MoE.md` 情形 F「Router 塌缩」的微观机制解释。

两篇合起来：`MoE.md` 给出「稀疏激活为什么必须存在、系统代价在哪里」，本文给出「实现稀疏激活的那个门控函数如何被一步步设计成今天的样子」。
