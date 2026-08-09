# LayerNorm 与 RMSNorm：背景、原理与计算过程

> 本文系统讲解 Transformer 中两种主流归一化层：**LayerNorm**（层归一化）与 **RMSNorm**（均方根归一化）。
> 从「为什么需要归一化」出发，推导两者的前向/反向公式，手算一个数值示例，
> 说明 RMSNorm 为什么能取代 LayerNorm 成为现代 LLM 的默认选择，
> 最后对应到 SGLang 中 `python/sglang/srt/layers/layernorm.py` 的真实实现与推理侧的工程考量。

## 目录

1. [背景：为什么需要归一化](#1-背景为什么需要归一化)
2. [归一化家族的定位：BN / LN / RMSNorm](#2-归一化家族的定位bn--ln--rmsnorm)
3. [LayerNorm 原理与计算过程](#3-layernorm-原理与计算过程)
4. [RMSNorm 原理与计算过程](#4-rmsnorm-原理与计算过程)
5. [手算数值示例](#5-手算数值示例)
6. [两者对比：为什么现代 LLM 都选 RMSNorm](#6-两者对比为什么现代-llm-都选-rmsnorm)
7. [Pre-Norm vs Post-Norm](#7-pre-norm-vs-post-norm)
8. [反向传播与梯度](#8-反向传播与梯度)
9. [数值精度问题（推理工程中最容易踩的坑）](#9-数值精度问题推理工程中最容易踩的坑)
10. [SGLang 中的实现](#10-sglang-中的实现)
11. [变体：GemmaRMSNorm、QK-Norm 等](#11-变体gemmarmsnormqk-norm-等)
12. [常见问题速答](#12-常见问题速答)

---

## 1. 背景：为什么需要归一化

### 1.1 原始问题：内部协变量偏移（Internal Covariate Shift）

深层网络训练时，每一层的输入分布会随着前面层参数的更新而不断变化。
Sergey Ioffe 与 Christian Szegedy 在 2015 年提出 **BatchNorm** 时，把这个现象命名为
**internal covariate shift（ICS）**：后面的层需要不断适应前面层输出分布的漂移，
导致学习率必须调小、收敛变慢。

> **注意（这是一个重要的历史修正）**：ICS 这个解释在 2018 年被
> Santurkar et al. 的论文 *How Does Batch Normalization Help Optimization?*（NeurIPS 2018）**证伪**了。
> 他们做了一个关键实验：在 BN 之后**人为注入**分布噪声（故意制造 ICS），
> 网络依然训练得很好。真正的作用机制是：
> **归一化让损失曲面（loss landscape）显著变平滑，使梯度更可预测、允许更大的学习率**。
>
> 所以今天说「归一化是为了解决 ICS」并不准确，但这个词已经流传开了。
> 更准确的说法是：**归一化改善了优化的条件数（conditioning）**。

### 1.2 归一化到底做了什么

抛开历史争论，归一化实际带来三个可验证的好处：

| 好处 | 机制 |
| --- | --- |
| **稳定激活尺度** | 把每层输出拉回到可控的量级，避免深层网络中激活值指数级放大或衰减 |
| **平滑损失曲面** | 使梯度的 Lipschitz 常数变小，梯度方向更稳定，允许更大学习率 |
| **对权重缩放不变** | 若 $W \to cW$，归一化后输出不变——这解耦了「权重大小」与「函数行为」，使优化更容易 |

第三点值得展开：归一化具有**尺度不变性（scale invariance）**。
对 LayerNorm，$\mathrm{LN}(cx) = \mathrm{LN}(x)$（$c > 0$）。
这意味着优化器不需要精细调整权重的绝对尺度，只需关心方向。

### 1.3 为什么 Transformer 不用 BatchNorm

BatchNorm 在 CNN 上非常成功，但在 Transformer/NLP 上几乎不可用：

| 问题 | 说明 |
| --- | --- |
| **依赖 batch 维** | BN 沿 batch 维统计均值方差，batch 小时统计量噪声大 |
| **变长序列** | NLP 的序列长度不一，padding 位置会污染统计量 |
| **训练/推理不一致** | BN 训练时用 batch 统计量、推理时用滑动平均，两者存在 gap |
| **自回归推理致命** | **decode 时 batch 内各请求相互独立，用 batch 统计量会造成请求间信息泄漏**，且结果依赖 batch 组成——同一请求在不同 batch 里输出不同，这在 serving 场景完全不可接受 |

最后一条是根本性的。这直接导向 LayerNorm：**只在单个样本的特征维内做归一化，与 batch 完全解耦**。

---

## 2. 归一化家族的定位：BN / LN / RMSNorm

设输入张量形状为 $(B, S, H)$（batch、序列长、隐藏维）。**关键区别在于沿哪个维度求统计量**：

```text
张量 (B, S, H)

BatchNorm  : 沿 B（和 S）求统计量，每个特征通道一组 (μ, σ)
             ─────────────► 跨样本，推理需滑动平均

LayerNorm  : 沿 H 求统计量，每个 token 一组 (μ, σ)
             ─────────────► 样本内，训练=推理

RMSNorm    : 沿 H 求 RMS，每个 token 一个 RMS（不减均值）
             ─────────────► LayerNorm 的简化版
```

用一句话概括三者：

$$
\begin{aligned}
\text{BatchNorm} &: \text{跨样本、同特征} \\
\text{LayerNorm} &: \text{同样本、跨特征，中心化 + 缩放} \\
\text{RMSNorm}   &: \text{同样本、跨特征，仅缩放}
\end{aligned}
$$

**对 LLM 而言，归一化都是逐 token 独立的**——这一点非常重要，它意味着：

- 归一化**不需要任何跨卡通信**（这解释了为什么 TP 中 LayerNorm/RMSNorm 在每张卡上冗余计算，
  见 `docs/theory/distributed/TP.md`）；
- 归一化对 batch 组成不敏感，天然满足 serving 的确定性要求。

---

## 3. LayerNorm 原理与计算过程

### 3.1 出处

> **论文**：Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey Hinton，*Layer Normalization*（arXiv, 2016）。
> Hinton 是 2018 年图灵奖得主。
> **背景**：作者的直接动机是**把 BatchNorm 的成功迁移到 RNN 上**。
> RNN 的时间步长度可变，BN 需要为每个时间步维护独立统计量，非常笨拙。
> LayerNorm 通过「在单个样本的特征维内归一化」绕开了对 batch 和时间步的依赖。
> 2017 年 *Attention Is All You Need* 采用它之后，LayerNorm 才随 Transformer 真正成为主流。

### 3.2 前向公式

对单个 token 的隐藏向量 $x \in \mathbb{R}^{H}$：

**第一步：求均值**

$$
\mu = \frac{1}{H}\sum_{i=1}^{H} x_i
$$

**第二步：求方差**（注意是有偏估计，除以 $H$ 而非 $H-1$）

$$
\sigma^2 = \frac{1}{H}\sum_{i=1}^{H} (x_i - \mu)^2
$$

**第三步：归一化**

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

**第四步：仿射变换**（可学习参数 $\gamma, \beta \in \mathbb{R}^{H}$）

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

合并写成一个式子：

$$
\boxed{\;\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta\;}
$$

其中 $\odot$ 是逐元素乘。

### 3.3 各部分的作用

| 组件 | 作用 | 去掉会怎样 |
| --- | --- | --- |
| **减均值 $\mu$** | 中心化（re-centering），消除输入的平移分量 | RMSNorm 就是去掉它 → 见 §4 |
| **除以标准差** | 缩放（re-scaling），把向量拉到单位尺度 | 归一化的核心，不能去 |
| **$\epsilon$** | 数值保护，防止 $\sigma^2 \approx 0$ 时除零 | 全零输入会产生 NaN |
| **$\gamma$（weight）** | 恢复表达能力，让网络可以学出「不需要归一化」 | 表达能力受限 |
| **$\beta$（bias）** | 恢复偏移能力 | RMSNorm 通常也省略它 |

> **为什么需要 $\gamma, \beta$**：归一化强行把分布拉到均值 0、方差 1，这是一个**很强的约束**，
> 可能损害网络表达力。加上仿射变换后，网络可以通过学习 $\gamma = \sqrt{\sigma^2+\epsilon}$、$\beta = \mu$
> 来**还原**归一化前的分布——即归一化层至少不会让模型变差。这与 BatchNorm 的设计思路一致。

### 3.4 计算过程（逐步）

对 $x \in \mathbb{R}^{H}$，一次 LayerNorm 的完整数据流：

```text
x (H,)
  │
  ├──► sum(x)/H ──────────────► μ  (标量)
  │
  ├──► sum((x-μ)²)/H ─────────► σ² (标量)      [需要第二遍遍历 x]
  │
  ├──► (x - μ) * rsqrt(σ²+ε) ─► x̂ (H,)
  │
  └──► γ ⊙ x̂ + β ─────────────► y  (H,)
```

**计算量统计**（每个 token）：

| 项目 | 次数 |
| --- | --- |
| 归约（reduction） | **2 次**（一次求 $\mu$，一次求 $\sigma^2$） |
| 逐元素运算 | 约 $5H$ 次浮点运算（减、平方、乘、加） |
| 内存访问 | 读 $x$（若不缓存则读 2 遍）、读 $\gamma,\beta$、写 $y$ |

> **两次归约是 LayerNorm 的关键开销**。归约在 GPU 上需要 block 内同步，
> 两次归约意味着两次同步屏障。工程上可以用 **Welford 算法** 或
> **$\mathbb{E}[x^2] - (\mathbb{E}[x])^2$** 技巧把它压成一次归约
> （同时累加 $\sum x$ 和 $\sum x^2$），但后者数值稳定性较差。

---

## 4. RMSNorm 原理与计算过程

### 4.1 出处

> **论文**：Biao Zhang（爱丁堡大学）, Rico Sennrich（爱丁堡 / 苏黎世大学），
> *Root Mean Square Layer Normalization*（NeurIPS 2019）。
> **背景与核心洞察**：作者做了一个消融实验，把 LayerNorm 的两个作用拆开：
> **re-centering（减均值）** 与 **re-scaling（除标准差）**。
> 结果发现——**re-scaling 才是让训练稳定的关键，re-centering 几乎没有贡献**。
> 既然如此，减均值这一步（以及它带来的一次额外归约）就是纯粹的浪费。
>
> 论文报告 RMSNorm 相比 LayerNorm **减少 7%–64% 的运行时间**，而模型质量基本持平。
> 这个「去掉一半计算但效果不变」的结论，使它被 LLaMA、GPT-NeoX、T5、PaLM、
> Gemma、Qwen、DeepSeek 等几乎所有现代开源 LLM 采用。

### 4.2 前向公式

$$
\mathrm{RMS}(x) = \sqrt{\frac{1}{H}\sum_{i=1}^{H} x_i^2}
$$

$$
\boxed{\;\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\dfrac{1}{H}\sum_{i=1}^{H} x_i^2 + \epsilon}}\;}
$$

对比 LayerNorm，差异只有两处：

1. **不减均值** $\mu$；
2. **通常不加偏置** $\beta$。

### 4.3 与 LayerNorm 的形式关系

如果输入恰好已经零均值（$\mu = 0$），那么

$$
\sigma^2 = \frac{1}{H}\sum (x_i - 0)^2 = \frac{1}{H}\sum x_i^2 = \mathrm{RMS}(x)^2
$$

即 **RMSNorm 与 LayerNorm 完全等价**。所以可以这样理解：

> RMSNorm = 「假设输入已经中心化」的 LayerNorm。
> 它把 $\sigma$ 换成了 $\mathrm{RMS}$，两者的关系是
> $\mathrm{RMS}(x)^2 = \sigma^2 + \mu^2$。

由此可见，**当 $|\mu| \ll \sigma$ 时两者几乎相同**——而高维随机向量的均值天然接近 0
（$\mu$ 是 $H$ 个数的平均，其标准差约为 $\sigma/\sqrt{H}$，$H$ 越大越接近 0），
这从理论上解释了为什么去掉 re-centering 影响很小。

### 4.4 计算过程（逐步）

```text
x (H,)
  │
  ├──► sum(x²)/H ─────────────► ms (标量)     [只需一次归约]
  │
  ├──► x * rsqrt(ms+ε) ───────► x̂ (H,)
  │
  └──► γ ⊙ x̂ ─────────────────► y  (H,)
```

**计算量对比**：

| 项目 | LayerNorm | RMSNorm | 节省 |
| --- | --- | --- | --- |
| 归约次数 | 2 | **1** | 50% |
| 逐元素 FLOPs | ~$5H$ | ~$3H$ | ~40% |
| 参数量 | $2H$（$\gamma,\beta$） | $H$（$\gamma$） | 50% |
| block 内同步 | 2 次 | 1 次 | 50% |

> **为什么归约次数比 FLOPs 更重要**：归一化是**访存密集型（memory-bound）**算子，
> 不是算力密集型。用 §9 的 Roofline 视角看（参见 `docs/theory/distributed/cost_model.md`），
> 它的算术强度极低，瓶颈在 HBM 带宽与同步开销，而不是浮点运算。
> 因此「少一次全维归约、少一次同步」带来的收益远大于 FLOPs 数字的减少。

---

## 5. 手算数值示例

取 $H = 4$，$x = [1,\ 2,\ 3,\ 4]$，$\epsilon = 0$（为简化手算），$\gamma = \mathbf{1}$，$\beta = \mathbf{0}$。

### 5.1 LayerNorm

**均值**：

$$
\mu = \frac{1+2+3+4}{4} = \frac{10}{4} = 2.5
$$

**中心化后的向量**：

$$
x - \mu = [-1.5,\ -0.5,\ 0.5,\ 1.5]
$$

**方差**：

$$
\sigma^2 = \frac{(-1.5)^2 + (-0.5)^2 + 0.5^2 + 1.5^2}{4}
= \frac{2.25 + 0.25 + 0.25 + 2.25}{4} = \frac{5}{4} = 1.25
$$

$$
\sigma = \sqrt{1.25} \approx 1.1180
$$

**归一化结果**：

$$
\mathrm{LN}(x) = \frac{[-1.5,\ -0.5,\ 0.5,\ 1.5]}{1.1180}
\approx [-1.3416,\ -0.4472,\ 0.4472,\ 1.3416]
$$

**验证**：输出均值 $= 0$ ✓，输出方差 $= \frac{1.3416^2 \times 2 + 0.4472^2 \times 2}{4} = \frac{3.6 + 0.4}{4} = 1$ ✓

### 5.2 RMSNorm

**均方**：

$$
\mathrm{ms} = \frac{1^2 + 2^2 + 3^2 + 4^2}{4} = \frac{1+4+9+16}{4} = \frac{30}{4} = 7.5
$$

$$
\mathrm{RMS}(x) = \sqrt{7.5} \approx 2.7386
$$

**归一化结果**：

$$
\mathrm{RMSNorm}(x) = \frac{[1,\ 2,\ 3,\ 4]}{2.7386}
\approx [0.3651,\ 0.7303,\ 1.0954,\ 1.4606]
$$

**验证恒等式** $\mathrm{RMS}^2 = \sigma^2 + \mu^2$：

$$
\sigma^2 + \mu^2 = 1.25 + 2.5^2 = 1.25 + 6.25 = 7.5 = \mathrm{RMS}^2 \quad\checkmark
$$

### 5.3 对比与观察

| | LayerNorm | RMSNorm |
| --- | --- | --- |
| 输出 | $[-1.342, -0.447, 0.447, 1.342]$ | $[0.365, 0.730, 1.095, 1.461]$ |
| 输出均值 | 0 | 0.9129（$= \mu/\mathrm{RMS}$） |
| 输出 RMS | 1 | 1 |

**关键观察**：这个例子里两者差异**很大**，因为 $\mu = 2.5$ 相对 $\sigma = 1.118$ 并不小
（$|\mu|/\sigma = 2.24$）。但在真实网络中，隐藏维 $H$ 通常是 4096~8192，
激活的均值统计上接近 0，此时两者结果非常接近——这正是 RMSNorm 能替代 LayerNorm 的实证基础。

**一个有用的直觉**：RMSNorm 保留了输入的「方向 + 均值分量」，只把**长度**归一化到 $\sqrt{H}$；
LayerNorm 则额外把均值分量（即沿全 1 向量 $\mathbf{1}$ 的分量）也投影掉了。
用几何语言说：

$$
\mathrm{RMSNorm}(x) = \sqrt{H}\cdot\frac{x}{\|x\|_2},
\qquad
\mathrm{LN}(x) = \sqrt{H}\cdot\frac{x - (\mathbf{1}^\top x/H)\mathbf{1}}{\|x - (\mathbf{1}^\top x/H)\mathbf{1}\|_2}
$$

即 **RMSNorm 把向量投影到半径 $\sqrt{H}$ 的球面上；LayerNorm 先把向量投影到与 $\mathbf{1}$ 正交的超平面，再投到球面上。**

---

## 6. 两者对比：为什么现代 LLM 都选 RMSNorm

| 维度 | LayerNorm | RMSNorm |
| --- | --- | --- |
| 提出年份 | 2016 | 2019 |
| 归约次数 | 2 | 1 |
| 参数量 | $2H$ | $H$ |
| 是否中心化 | 是 | 否 |
| 尺度不变性 | ✓ | ✓ |
| **平移不变性** | ✓（$\mathrm{LN}(x+c\mathbf{1}) = \mathrm{LN}(x)$） | ✗ |
| 典型模型 | 原始 Transformer、BERT、GPT-2 | LLaMA 系、Qwen、DeepSeek、Gemma、T5、PaLM |
| 相对速度 | 基准 | 快 7%–64%（原论文数据） |

### 6.1 RMSNorm 胜出的三个理由

1. **更快**：少一次全维归约、少一次同步屏障。对访存密集算子，这是实打实的收益。
2. **质量不降**：大量实证表明 re-centering 对 Transformer 收敛几乎无贡献。
3. **参数更少**：省掉 $\beta$，每层少 $H$ 个参数（虽然占比很小，但也省掉了对应的显存与通信）。

### 6.2 什么时候 LayerNorm 仍然有意义

- **需要平移不变性**的场景。RMSNorm 对输入整体加常数敏感：
  $\mathrm{RMSNorm}(x + c\mathbf{1}) \neq \mathrm{RMSNorm}(x)$。
- **复现已有模型**。BERT、GPT-2、ViT 等用的是 LayerNorm，权重不可互换。
- **某些多模态/视觉编码器**仍沿用 LayerNorm（SGLang 中 `LayerNorm` 类主要服务这类模块）。

---

## 7. Pre-Norm vs Post-Norm

归一化**放在哪里**和用哪种归一化同样重要。

### 7.1 两种放法

```text
Post-Norm（原始 Transformer, 2017）:
    x ──► Sublayer ──► (+) ──► Norm ──► out
    └────────────────► ┘

Pre-Norm（现代 LLM 默认）:
    x ──► Norm ──► Sublayer ──► (+) ──► out
    └───────────────────────────► ┘
```

公式表达：

$$
\begin{aligned}
\text{Post-Norm}&: \quad x_{l+1} = \mathrm{Norm}\big(x_l + F(x_l)\big) \\
\text{Pre-Norm} &: \quad x_{l+1} = x_l + F\big(\mathrm{Norm}(x_l)\big)
\end{aligned}
$$

### 7.2 为什么现代 LLM 几乎都用 Pre-Norm

关键在于**残差路径是否「干净」**。Pre-Norm 下展开递推：

$$
x_L = x_0 + \sum_{l=0}^{L-1} F_l\big(\mathrm{Norm}(x_l)\big)
$$

存在一条**从输入直达输出、不经过任何归一化的恒等路径**，梯度可以无衰减地回传。
而 Post-Norm 每层都有归一化挡在残差路径上，深层时梯度容易衰减。

| | Post-Norm | Pre-Norm |
| --- | --- | --- |
| 训练稳定性 | 差，深层需要 warmup | 好，可去掉/缩短 warmup |
| 是否需要 learning rate warmup | 强依赖 | 弱依赖 |
| 最终效果 | 略好（充分调参时） | 略差但差距很小 |
| 深层可扩展性 | 差 | **好**（关键） |

> **相关论文**：Xiong et al., *On Layer Normalization in the Transformer Architecture*（ICML 2020）——
> 从理论上分析了 Post-Norm 的梯度在初始化时随深度爆炸/消失，而 Pre-Norm 的梯度良好，
> 这是 Post-Norm 需要 warmup 的根本原因。

### 7.3 对推理的直接影响：残差融合

Pre-Norm 的结构 $x_{l+1} = x_l + F(\mathrm{Norm}(x_l))$ 意味着
**归一化的输入总是「上一层输出 + 残差」**。因此可以把「残差加法」和「归一化」
**融合进同一个 kernel**，省掉一次中间结果的显存往返——这就是 SGLang 里
`fused_add_rmsnorm` 的由来（见 §10.3）。

---

## 8. 反向传播与梯度

虽然推理不需要反向，但理解梯度有助于解释「为什么归一化能稳定训练」。

### 8.1 LayerNorm 的梯度

设 $\hat{x} = \dfrac{x-\mu}{\sqrt{\sigma^2+\epsilon}}$，$y = \gamma\odot\hat{x}+\beta$，上游梯度为 $g = \partial L/\partial y$。

对参数的梯度很简单（逐元素）：

$$
\frac{\partial L}{\partial \gamma_i} = g_i \hat{x}_i,
\qquad
\frac{\partial L}{\partial \beta_i} = g_i
$$

对输入的梯度（记 $\tilde{g}_i = g_i\gamma_i$，$s = \sqrt{\sigma^2+\epsilon}$）：

$$
\frac{\partial L}{\partial x_i} = \frac{1}{s}\left(
\tilde{g}_i
- \underbrace{\frac{1}{H}\sum_{j}\tilde{g}_j}_{\text{均值项}}
- \hat{x}_i\cdot\underbrace{\frac{1}{H}\sum_{j}\tilde{g}_j\hat{x}_j}_{\text{方差项}}
\right)
$$

### 8.2 RMSNorm 的梯度

设 $r = \sqrt{\frac{1}{H}\sum x_j^2 + \epsilon}$，$\hat{x} = x/r$：

$$
\frac{\partial L}{\partial \gamma_i} = g_i \hat{x}_i
$$

$$
\frac{\partial L}{\partial x_i} = \frac{1}{r}\left(
\tilde{g}_i - \hat{x}_i\cdot\frac{1}{H}\sum_{j}\tilde{g}_j\hat{x}_j
\right)
$$

**对比**：RMSNorm 的反向式子里**少了「均值项」**——与前向少一次归约完全对应。
反向也从 2 次归约降到 1 次。

### 8.3 梯度的两个重要性质

1. **梯度与输入尺度成反比**（$1/s$ 或 $1/r$ 因子）：输入越大，梯度越小。
   这是一个**自动的梯度裁剪机制**，防止激活爆炸导致梯度爆炸。
2. **梯度被投影到与 $\hat{x}$ 正交的方向**（那个 $-\hat{x}\cdot(\cdots)$ 项）：
   沿 $\hat{x}$ 方向的分量被减掉了。几何上讲，因为归一化输出被约束在球面上，
   **沿半径方向的移动不改变输出**，所以那个方向的梯度自然为零。

---

## 9. 数值精度问题（推理工程中最容易踩的坑）

这是理论文档常忽略、但工程上极其重要的一节。

### 9.1 为什么必须用 FP32 累加

模型权重与激活通常是 **bf16** 或 **fp16**，但归一化的**归约必须在 fp32 中做**。原因：

- **bf16 只有 8 位尾数**（约 2–3 位十进制有效数字）。累加 $H = 8192$ 个平方项时，
  当累加器已经很大而新增项很小，会发生**吞位（swamping）**——小数被直接舍弃。
- **fp16 的动态范围只到 65504**。若激活值约为 100，则 $x^2 = 10^4$，
  累加 8192 项就会 **溢出到 inf**，随后产生 NaN。

因此标准做法是：

```text
输入 bf16 ──► 转 fp32 ──► 求 RMS（fp32 累加）──► 归一化（fp32）──► 转回 bf16
```

SGLang 的 `forward_native` 严格遵循这个模式：

```464:465:python/sglang/srt/layers/layernorm.py
        orig_dtype = self.override_orig_dtype or x.dtype
        x = x.to(torch.float32)
```

```493:494:python/sglang/srt/layers/layernorm.py
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
```

### 9.2 一个微妙的坑：乘 $\gamma$ 之前还是之后转换 dtype

看似等价的两种写法，数值结果并不同：

```text
写法 A：先乘权重（在 fp32 下做乘法），再转回窄类型
    x = (x * weight).to(orig_dtype)

写法 B：先转回窄类型，再乘权重（乘法在窄类型下做）
    x = weight * x.to(orig_dtype)
```

**HuggingFace 的 `LlamaRMSNorm` 用的是写法 B**。若推理框架用写法 A，
就会与 HF 产生微小但可累积的数值差异，在长序列生成上可能导致 token 分歧。

SGLang 用 `cast_x_before_out_mul` 这个开关精确控制这个行为：

```496:499:python/sglang/srt/layers/layernorm.py
        if self.cast_x_before_out_mul:
            x = self.weight * x.to(orig_dtype)
        else:
            x = (x * self.weight).to(orig_dtype)
```

代码注释里也明确说明了它的用途——为了与 HF 逐位对齐：

```112:116:python/sglang/srt/layers/layernorm.py
if _is_cuda:
    # HF-semantics RMSNorm kernel (JIT-compiled).  Used when `cast_x_before_out_mul=True`
    # (the transformers backend path) to produce outputs that are numerically identical
    # to HuggingFace `LlamaRMSNorm`: the cast from fp32 to the activation dtype happens
    # BEFORE the weight multiply, so the multiply is done in the narrow dtype.
```

> **这是一个很好的例子**：说明「数学上等价」不等于「浮点上等价」。
> 做模型对齐（accuracy alignment）排查时，这类 cast 顺序问题是常见根因。

### 9.3 $\epsilon$ 放在根号内还是根号外

两种常见写法：

$$
\text{(a)}\;\; \frac{x}{\sqrt{\mathrm{ms} + \epsilon}}
\qquad\qquad
\text{(b)}\;\; \frac{x}{\sqrt{\mathrm{ms}} + \epsilon}
$$

**主流实现（含 PyTorch、HF、SGLang）都用 (a)**。不同模型的 $\epsilon$ 取值也不同
（LLaMA 系常用 $10^{-5}$ 或 $10^{-6}$，Gemma 用 $10^{-6}$），
**移植权重时必须对齐 $\epsilon$**，否则会有精度偏差。

### 9.4 批不变性（batch invariance）

一个容易被忽视的问题：归一化虽然逐 token 独立，但 **kernel 的归约顺序可能随 batch 大小改变**
（例如不同 batch 触发不同的 tile 划分或不同的归约树形状）。
由于浮点加法**不满足结合律**，这会导致同一请求在不同 batch 下输出有微小差异。

SGLang 为此提供了批不变模式：

```272:275:python/sglang/srt/layers/layernorm.py
        if is_batch_invariant_mode_enabled():
            if (
                residual is not None
```

对应的实现在 `sglang/srt/batch_invariant_ops` 的 `rms_norm_batch_invariant`。
这在需要**可复现输出**的场景（如评测、调试、A/B 对比）中很关键。

---

## 10. SGLang 中的实现

核心文件：`python/sglang/srt/layers/layernorm.py`（约 960 行）。

### 10.1 类结构总览

| 类 | 用途 |
| --- | --- |
| `RMSNorm` | 标准 RMSNorm，LLM 主干使用 |
| `LayerNorm` | 标准 LayerNorm，主要给多模态/视觉编码器等模块用 |
| `GemmaRMSNorm` | Gemma 系变体，权重语义为 $\gamma + 1$ |
| `Gemma3RMSNorm` / `Gemma4RMSNorm` | Gemma 3/4 的进一步变体 |
| `RMSNormWithoutScale` | 无可学习权重的 RMSNorm |

所有类都继承 `MultiPlatformOp`，按硬件后端分发到不同实现：
`forward_cuda` / `forward_hip` / `forward_cpu` / `forward_npu` / `forward_xpu` / `forward_native`。
`forward_native` 是**纯 PyTorch 参考实现，定义了语义标准**，其他路径必须与它数值对齐。

### 10.2 参考实现（语义标准）

`RMSNorm.forward_native` 完整体现了 §4 与 §9 的所有要点：

前半段——升 fp32、融合残差：

```462:473:python/sglang/srt/layers/layernorm.py
        if not x.is_contiguous():
            x = x.contiguous()
        orig_dtype = self.override_orig_dtype or x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            if post_residual_addition is not None:
                x = x + post_residual_addition.to(torch.float32)
            if self.fp32_residual:
                residual = x.clone()
            else:
                residual = x.to(orig_dtype)
```

后半段——求均方、缩放、乘权重、转回原 dtype：

```493:504:python/sglang/srt/layers/layernorm.py
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)

        if self.cast_x_before_out_mul:
            x = self.weight * x.to(orig_dtype)
        else:
            x = (x * self.weight).to(orig_dtype)

        if residual is None:
            return x
        else:
            return x, residual
```

逐行对应到公式：

| 代码 | 公式 |
| --- | --- |
| `x.to(torch.float32)` | fp32 累加保护（§9.1） |
| `x = x + residual` | Pre-Norm 的残差融合（§7.3） |
| `x.pow(2).mean(dim=-1)` | $\mathrm{ms} = \frac{1}{H}\sum x_i^2$ |
| `torch.rsqrt(variance + eps)` | $1/\sqrt{\mathrm{ms}+\epsilon}$，用 `rsqrt` 而非 `1/sqrt` 更快 |
| `x * self.weight` | $\gamma \odot \hat{x}$ |

注意 `RMSNorm` 里变量名叫 `variance`，但算的其实是**均方（mean square）而非方差**——
因为没有减均值。这是沿用 HF 的命名，容易误导。

### 10.3 残差融合：`fused_add_rmsnorm`

这是推理侧最重要的优化。CUDA 路径导入了融合 kernel：

```88:93:python/sglang/srt/layers/layernorm.py
    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )
```

**为什么要融合**：Pre-Norm 结构下每层都要做「加残差 → 归一化」，不融合的话：

```text
未融合：读 x、读 residual → 写 tmp → 读 tmp → 写 y     (4 次 HBM 往返)
已融合：读 x、读 residual → 写 y、写 residual          (2 次 HBM 往返)
```

由于归一化是**访存密集型**算子，减少一半的 HBM 往返几乎等于翻倍的吞吐。

### 10.4 模型侧的调用模式

以 `llama.py` 的 decoder layer 为例，可以看到典型的 Pre-Norm + 残差传递写法：

```322:334:python/sglang/srt/models/llama.py
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
```

**关键设计**：`residual` 在层与层之间**以未归一化的形式传递**，
每层的归一化顺带完成上一层的残差加法。第一层因为还没有残差，走单参数分支。
这样整个网络只在最后做一次残差合并，中间全部融合。

### 10.5 后端分发

同一个 `RMSNorm` 在不同硬件上走不同 kernel：

| 后端 | 实现 |
| --- | --- |
| CUDA | `sgl_kernel` 的 `rmsnorm` / `fused_add_rmsnorm`，或 JIT kernel |
| HIP (ROCm) | aiter 的 `rmsnorm2d_fwd`，或 vllm 的 `rms_norm` |
| CPU | `torch.ops.sgl_kernel.rmsnorm_cpu`（需 AMX 支持），否则回退 native |
| NPU / XPU / MUSA | 各自的厂商 kernel |
| 兜底 | `forward_native`（纯 PyTorch） |

---

## 11. 变体：GemmaRMSNorm、QK-Norm 等

### 11.1 GemmaRMSNorm：权重的 $+1$ 偏移

Gemma 系模型的 RMSNorm 权重语义与标准版不同：

$$
y = (1 + \gamma) \odot \hat{x}
$$

即**存储的权重是「相对 1 的偏移量」**，初始化为 0（而非标准 RMSNorm 的初始化为 1）。
这样做的好处是权重衰减（weight decay）会把 $\gamma$ 拉向 0，即拉向恒等变换，
而不是像标准写法那样拉向 0 倍缩放（会摧毁信号）。

SGLang 的实现预先算好 $\gamma+1$ 并缓存，避免每次前向都做加法：

```657:671:python/sglang/srt/layers/layernorm.py
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.register_buffer(
            "gemma_weight", torch.ones_like(self.weight), persistent=False
        )
        # (Chen-0210) Gemma weight = standard_weight + 1. Precompute once.
        # If TRTLLM allreduce fusion ever provides gemma-style norm
        # natively, this can be removed.
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)
        # Keep storage stable for CUDA graphs or fused paths that capture this buffer.
        torch.add(param.data, 1.0, out=self.gemma_weight)
```

注释里提到「保持存储地址稳定」，这是为了 **CUDA Graph 捕获**——
被捕获的 kernel 会记住显存地址，若权重张量被重新分配，graph 就会读到错误数据。

> **迁移警告**：把 Gemma 权重直接塞进标准 RMSNorm（或反之）会得到完全错误的输出，
> 因为差了一个 $+1$。这是模型移植时的经典 bug。

### 11.2 QK-Norm

近年不少模型（Gemma 2/3、Qwen3 等）在 attention 内部对 **Q 和 K 分别做 RMSNorm**：

$$
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\left(\frac{\mathrm{Norm}(Q)\,\mathrm{Norm}(K)^\top}{\sqrt{d}}\right)V
$$

**动机**：训练大模型时 attention logits 容易爆炸（$QK^\top$ 数值过大导致 softmax 饱和、梯度消失）。
对 Q/K 归一化后，logits 的量级被强制约束住，训练更稳定。

注意这里归一化是**沿 head_dim 维**做的，不是沿 hidden_size。

### 11.3 变体速查

| 变体 | 公式差异 | 代表模型 |
| --- | --- | --- |
| 标准 RMSNorm | $\gamma \odot \hat{x}$ | LLaMA, Qwen, DeepSeek |
| GemmaRMSNorm | $(1+\gamma) \odot \hat{x}$ | Gemma 系 |
| RMSNormWithoutScale | $\hat{x}$（无权重） | 部分模块 |
| QK-Norm | 对 Q/K 沿 head_dim 归一化 | Gemma 2/3, Qwen3 |
| `var_hidden_size` | 只用前 $k$ 维算统计量，但缩放全部维度 | 部分特殊结构 |

最后一个变体在 SGLang 中通过 `variance_size_override` 支持：

```482:491:python/sglang/srt/layers/layernorm.py
        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[..., : self.variance_size_override]
```

---

## 12. 常见问题速答

**Q1：RMSNorm 为什么不需要减均值？**
A：Zhang & Sennrich (2019) 的消融实验证明 re-scaling 是稳定训练的关键，re-centering 贡献极小。
且高维向量的均值统计上接近 0（标准差约 $\sigma/\sqrt{H}$），减不减差别不大。

**Q2：为什么 TP（张量并行）不切分归一化层？**
A：归一化需要沿 **hidden 维**做全维归约。若按 hidden 维切分，每张卡只有部分维度，
求 RMS 就需要一次 all-reduce。而归一化本身的计算量极小（相比 GEMM 可忽略），
**为省这点计算而增加一次通信完全不划算**。所以每张卡冗余计算完整的归一化。
详见 `docs/theory/distributed/TP.md`。

**Q3：$\epsilon$ 取多少？**
A：常见 $10^{-5}$（LLaMA）或 $10^{-6}$（Gemma、多数新模型）。
**必须与原模型配置一致**，否则数值不对齐。SGLang 从 `config.rms_norm_eps` 读取。

**Q4：为什么 `variance` 变量名算的是均方？**
A：沿用 HuggingFace 的命名。RMSNorm 不减均值，所以 `x.pow(2).mean()` 算的是
$\mathbb{E}[x^2]$（均方），只有在 $\mu=0$ 时才等于方差。命名有误导性，但已成惯例。

**Q5：归一化是 compute-bound 还是 memory-bound？**
A：**memory-bound**。每个元素只做几次浮点运算，但要完整读写一遍张量，算术强度极低。
所以优化方向是**减少 HBM 往返**（kernel 融合），而不是减少 FLOPs。
参见 `docs/theory/distributed/cost_model.md` §9 的 Roofline 分析。

**Q6：为什么要把残差加法融进归一化 kernel？**
A：Pre-Norm 下每层都是「加残差 → 归一化」，融合后 HBM 往返从 4 次降到 2 次。
对 memory-bound 算子，这几乎等于翻倍吞吐。

**Q7：同一个请求在不同 batch 下输出会变吗？**
A：数学上不会（归一化逐 token 独立），但**浮点上可能会**——kernel 的归约顺序可能随
batch 大小改变，而浮点加法不满足结合律。需要严格可复现时，用 SGLang 的批不变模式。

---

## 参考与延伸

- **原始论文**：
  Ba, Kiros & Hinton（多伦多大学）, *Layer Normalization*（arXiv:1607.06450, 2016）；
  Zhang & Sennrich（爱丁堡大学 / 苏黎世大学）, *Root Mean Square Layer Normalization*（NeurIPS 2019）；
  Ioffe & Szegedy（Google）, *Batch Normalization*（ICML 2015）——归一化的起点。
- **机制解释（重要）**：
  Santurkar, Tsipras, Ilyas & Madry（MIT）,
  *How Does Batch Normalization Help Optimization?*（NeurIPS 2018）——推翻 ICS 解释，
  提出「平滑损失曲面」才是真正机制。
- **Pre-Norm vs Post-Norm**：
  Xiong et al., *On Layer Normalization in the Transformer Architecture*（ICML 2020）；
  Wang et al., *DeepNet: Scaling Transformers to 1,000 Layers*（2022）。
- **本仓库相关文档**：
  `docs/theory/distributed/TP.md`（为什么归一化层不切分）；
  `docs/theory/distributed/cost_model.md` §9（Roofline，理解 memory-bound）；
  `docs/theory/op/`（其他算子原理文档）。
- **相关代码**：
  `python/sglang/srt/layers/layernorm.py`（全部归一化类）；
  `python/sglang/srt/batch_invariant_ops/`（批不变实现）；
  `python/sglang/jit_kernel/norm.py`、`python/sglang/jit_kernel/rmsnorm_hf.py`（JIT kernel）；
  `python/sglang/srt/models/llama.py`（Pre-Norm + 残差传递的调用范式）。

