# Decoder-Only Transformer 网络架构详解

> 本文自底向上拆解一个现代 **decoder-only**（仅解码器，自回归）Transformer 的完整结构：
> 每一层是什么模块、数据如何流动、**每一步张量的输入输出维度**。
> 归一化统一采用 **RMSNorm**（Pre-Norm 放置），注意力采用**最普通的多头注意力（MHA）**，
> 结构整体对标 LLaMA 系列（去掉 GQA，退化为标准 MHA）。
>
> 相关阅读：`../MHA_MQA_GQA.md`（注意力家族）、`../Glossary.md`（术语表）、
> `../distributed/TP.md`（这些层如何切到多卡）。

## 目录

1. [符号约定](#1-符号约定)
2. [整体架构一览](#2-整体架构一览)
3. [模块逐层拆解与维度](#3-模块逐层拆解与维度)
   - [3.1 Token Embedding](#31-token-embedding)
   - [3.2 Decoder Block（重复 L 次）](#32-decoder-blockl-次)
   - [3.3 RMSNorm](#33-rmsnorm)
   - [3.4 Multi-Head Attention（MHA）](#34-multi-head-attentionmha)
   - [3.5 前馈网络 FFN](#35-前馈网络-ffn)
   - [3.6 残差连接与 Pre-Norm](#36-残差连接与-pre-norm)
   - [3.7 Final Norm + LM Head](#37-final-norm--lm-head)
4. [完整前向的维度追踪表](#4-完整前向的维度追踪表)
5. [具体数值示例（LLaMA-7B 配置）](#5-具体数值示例llama-7b-配置)
6. [Prefill 与 Decode：推理时的维度差异](#6-prefill-与-decode推理时的维度差异)
7. [logits 之后：约束解码、采样与输出解析](#7-logits-之后约束解码采样与输出解析)
   - [7.1 约束解码（Grammar / xgrammar）](#71-约束解码constrained-decoding--grammar)
   - [7.2 采样（Sampling）](#72-采样sampling)
   - [7.3 推理内容解析（Reasoning Parser）](#73-推理内容解析reasoning-parser)
   - [7.4 工具调用解析（Tool-Call Parser）](#74-工具调用解析tool-call-parser)
8. [参考与延伸](#8-参考与延伸)

---

## 1. 符号约定


| 符号                     | 含义                            | LLaMA-7B 示例 |
| ------------------------ | ------------------------------- | ------------- |
| $V$                      | 词表大小（vocab size）          | 32000         |
| $d_\text{model}$         | 隐藏维度（hidden size）         | 4096          |
| $s$                      | 序列长度（当前处理的 token 数） | 变量          |
| $h$                      | 注意力头数（num heads）         | 32            |
| $d_k = d_\text{model}/h$ | 每个头的维度（head dim）        | 128           |
| $d_\text{ff}$            | FFN 中间维（intermediate size） | 11008         |
| $L$                      | decoder block 层数              | 32            |
| $\varepsilon$            | RMSNorm 数值稳定项              | 1e-6          |

> **关于 batch 维**：为让维度追踪清爽，正文一律省略 batch 维，用 $(s, d_\text{model})$ 表示**单条序列**。
> 实际张量带 batch，形如 $(b, s, d_\text{model})$，batch 维在所有逐 token 操作中原样透传。

---

## 2. 整体架构一览

decoder-only 的骨架是：**词嵌入 → 堆叠 $L$ 个结构相同的 Decoder Block → 最终归一化 → 输出投影**。
全程隐藏维度保持 $d_\text{model}$ 不变，只在首尾的 embedding / LM head 处与词表维 $V$ 相接。

```
     input_ids  (s,)  ── 整数 token id
         │
         ▼
 ┌───────────────────┐
 │  Token Embedding  │   (s,) ──► (s, d_model)      查表
 └─────────┬─────────┘
           │  x₀ = (s, d_model)
           ▼
 ╔═══════════════════════════════════════════╗
 ║          Decoder Block × L                 ║   每层输入/输出都是 (s, d_model)
 ║                                            ║
 ║   x ──┬──────────────────────────┐         ║
 ║       ▼                          │         ║
 ║   ┌────────┐   ┌──────────┐      │(残差)    ║
 ║   │RMSNorm │──►│   MHA    │      │         ║   ← 注意力子层（Pre-Norm）
 ║   └────────┘   └────┬─────┘      │         ║
 ║                     ▼            │         ║
 ║                    (＋)◄─────────┘         ║   x = x + MHA(RMSNorm(x))
 ║                     │                      ║
 ║       ┌─────────────┴────────────┐         ║
 ║       ▼                          │(残差)    ║
 ║   ┌────────┐   ┌──────────┐      │         ║
 ║   │RMSNorm │──►│   FFN    │      │         ║   ← 前馈子层（Pre-Norm）
 ║   └────────┘   └────┬─────┘      │         ║
 ║                     ▼            │         ║
 ║                    (＋)◄─────────┘         ║   x = x + FFN(RMSNorm(x))
 ║                     │                      ║
 ╚═════════════════════╪══════════════════════╝
                       │  x_L = (s, d_model)
                       ▼
              ┌─────────────────┐
              │  Final RMSNorm  │   (s, d_model) ──► (s, d_model)
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │    LM Head      │   (s, d_model) ──► (s, V)     logits
              └────────┬────────┘
                       ▼
                logits (s, V)  ──► softmax / 采样得到下一个 token
```

三个要点：

- **Pre-Norm**：RMSNorm 放在每个子层**之前**（`x + Sublayer(Norm(x))`），残差路径上是「干净」的 $x$，
  梯度可无损直达底层——这是现代大模型能稳定堆到几十上百层的关键。
- **两个残差子层**：每个 Decoder Block = 「注意力子层」+「前馈子层」，各自带一条残差。
- **维度守恒**：从 $x_0$ 到 $x_L$，隐藏维度始终是 $d_\text{model}$，模块可无限堆叠。

---

## 3. 模块逐层拆解与维度

### 3.1 Token Embedding

一张 $(V, d_\text{model})$ 的查找表，把每个整数 token id 映射成一个 $d_\text{model}$ 维向量。


| 项              | 维度                  |
| --------------- | --------------------- |
| 输入`input_ids` | $(s,)$，整数          |
| 权重$W_E$       | $(V, d_\text{model})$ |
| 输出$x_0$       | $(s, d_\text{model})$ |

```
  input_ids    [ 101,  2054,   ...,   102 ]        (s,)   整数 id
                  │      │              │
                  ▼      ▼              ▼            查表 W_E (V, d_model)
               ┌──────┬──────┬─────┬──────┐
   x0          │ 行101 │行2054│ ... │ 行102 │         每个 id 取出对应一行
               └──────┴──────┴─────┴──────┘         (s, d_model)
```

> **位置信息**：本文采用 **RoPE（旋转位置编码）**——它不作用在这里，而是在每层 MHA 内部
> 施加到 Q、K 上（见 §3.4），**不改变任何张量维度**，也无需额外的位置 embedding 表。

### 3.2 Decoder Block（重复 $L$ 次）

一个 Block 的伪代码（Pre-Norm，RMSNorm）：

```python
def decoder_block(x):            # x: (s, d_model)
    x = x + mha(rms_norm(x))     # 注意力子层，输出 (s, d_model)
    x = x + ffn(rms_norm(x))     # 前馈子层，输出 (s, d_model)
    return x                     # (s, d_model)
```

把每个模块单独框出来、标上输入输出维度，一个 Block 的数据流如下（左侧竖线是残差主干，始终携带未归一化的 $x$）：

```
          x  (s, d_model)
          │
    ┌─────┴─────┐ 残差主干（原样透传 x）
    │           ▼
    │   ┌───────────────┐   in :  (s, d_model)
    │   │   RMSNorm      │
    │   └───────┬───────┘   out:  (s, d_model)
    │           ▼
    │   ┌───────────────┐   in :  (s, d_model)
    │   │   MHA          │   （内部临时展开 (h,s,d_k)、(h,s,s)，见 §3.4）
    │   └───────┬───────┘   out:  (s, d_model)
    │           ▼
    └─────────►(＋)  残差相加         (s, d_model) + (s, d_model)
                │
                ▼
          x  (s, d_model)   ← 注意力子层输出
          │
    ┌─────┴─────┐ 残差主干（原样透传 x）
    │           ▼
    │   ┌───────────────┐   in :  (s, d_model)
    │   │   RMSNorm      │
    │   └───────┬───────┘   out:  (s, d_model)
    │           ▼
    │   ┌───────────────┐   in :  (s, d_model)
    │   │   FFN          │   （内部临时升维到 (s, d_ff)，见 §3.5）
    │   └───────┬───────┘   out:  (s, d_model)
    │           ▼
    └─────────►(＋)  残差相加         (s, d_model) + (s, d_model)
                │
                ▼
          x  (s, d_model)   ← Block 输出
```

逐模块速览（详细维度见对应小节）：


| 模块               | 输入                      | 输出                  | 说明                                    |
| ------------------ | ------------------------- | --------------------- | --------------------------------------- |
| RMSNorm（attn 前） | $(s, d_\text{model})$     | $(s, d_\text{model})$ | 逐 token 归一化（§3.3）                |
| MHA                | $(s, d_\text{model})$     | $(s, d_\text{model})$ | 内部展开$(h,s,d_k)$、$(h,s,s)$（§3.4） |
| 残差相加 ①        | 两个$(s, d_\text{model})$ | $(s, d_\text{model})$ | $x + \text{MHA}(\dots)$                 |
| RMSNorm（ffn 前）  | $(s, d_\text{model})$     | $(s, d_\text{model})$ | 同上（§3.3）                           |
| FFN                | $(s, d_\text{model})$     | $(s, d_\text{model})$ | 内部升维到$(s, d_\text{ff})$（§3.5）   |
| 残差相加 ②        | 两个$(s, d_\text{model})$ | $(s, d_\text{model})$ | $x + \text{FFN}(\dots)$                 |

$L$ 个 Block 结构完全相同、权重各自独立。**每个模块以及整个 Block 的输入输出维度都恒为 $(s, d_\text{model})$**——
只有 MHA、FFN 的**内部**会临时变形，出口都收回到 $d_\text{model}$。

### 3.3 RMSNorm

RMSNorm（Root Mean Square Normalization）相比 LayerNorm **去掉了减均值（re-centering）和 bias**，
只按均方根做缩放，再乘一个可学习的逐通道增益 $g$：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\dfrac{1}{d_\text{model}}\displaystyle\sum_{i=1}^{d_\text{model}} x_i^2 \;+\; \varepsilon}} \odot g
$$


| 项      | 维度                  | 说明                                   |
| ------- | --------------------- | -------------------------------------- |
| 输入$x$ | $(s, d_\text{model})$ | 对最后一维（$d_\text{model}$）做归一化 |
| 增益$g$ | $(d_\text{model},)$   | 可学习，逐通道缩放；**无 bias**        |
| 输出    | $(s, d_\text{model})$ | 形状不变                               |

```
   x (s, d_model)                       每个 token 的 d_model 维向量各自独立处理
   ┌───────────────┐
   │ token 0  ───► rms=√(mean(x²)+ε) ───► x / rms ───► ⊙ g ───► out0 │
   │ token 1  ───► rms=√(mean(x²)+ε) ───► x / rms ───► ⊙ g ───► out1 │
   │  ...                                                     ...    │   (s, d_model)
   │ token s-1───► rms=√(mean(x²)+ε) ───► x / rms ───► ⊙ g ───► out_{s-1} │
   └───────────────┘
            └── 只沿 d_model 维统计，不跨 token；g 是 (d_model,) 逐通道增益 ──┘
```

要点：**逐 token 独立**（每个 token 的 $d_\text{model}$ 维向量各自归一），不改变形状；
比 LayerNorm 少一次求均值，更快且实践中同样稳定。

### 3.4 Multi-Head Attention（MHA）

标准多头注意力：$h$ 个头各有独立的 Q/K/V 投影（即 $h_{kv} = h$，无 GQA/MQA 共享）。
设输入 $x' = \text{RMSNorm}(x)$，形状 $(s, d_\text{model})$。

**逐步维度展开**：


| 步骤        | 运算                                                                             | 输出维度                |
| ----------- | -------------------------------------------------------------------------------- | ----------------------- |
| ① QKV 投影 | $Q=x'W_Q,\ K=x'W_K,\ V=x'W_V$，各 $W_\bullet\in(d_\text{model}, d_\text{model})$ | 各$(s, d_\text{model})$ |
| ② 拆头     | reshape$(s, d_\text{model}) \to (s, h, d_k)$，转置 $\to (h, s, d_k)$             | $(h, s, d_k)$           |
| ③ RoPE     | 对 Q、K 施加旋转位置编码                                                         | $(h, s, d_k)$（不变）   |
| ④ 打分     | $\text{scores} = QK^\top / \sqrt{d_k}$                                           | $(h, s, s)$             |
| ⑤ 因果掩码 | 上三角（未来位置）置$-\infty$                                                    | $(h, s, s)$             |
| ⑥ softmax  | 沿最后一维归一化                                                                 | $(h, s, s)$             |
| ⑦ 加权求和 | $\text{scores} \cdot V$                                                          | $(h, s, d_k)$           |
| ⑧ 合头     | 转置回$(s, h, d_k)$，reshape $\to (s, d_\text{model})$                           | $(s, d_\text{model})$   |
| ⑨ 输出投影 | $\text{out}\cdot W_O$，$W_O\in(d_\text{model}, d_\text{model})$                  | $(s, d_\text{model})$   |

数据流（标注每一步的维度变化）：

```
   x' (s, d_model)
     │
     ├──► W_Q ──► Q (s,d_model) ─拆头─► (h,s,d_k) ─RoPE─┐
     ├──► W_K ──► K (s,d_model) ─拆头─► (h,s,d_k) ─RoPE─┤
     └──► W_V ──► V (s,d_model) ─拆头─► (h,s,d_k) ───────┤
                                                        │
                            ┌───────────────────────────┘
                            ▼
                 scores = Q Kᵀ / √d_k          (h, s, s)     ④
                            │
                    + 因果掩码 M（未来位置 -∞）  (h, s, s)     ⑤
                            │
                       softmax（沿最后一维）     (h, s, s)     ⑥
                            │
                          · V                   (h, s, d_k)   ⑦
                            │
                    合头 reshape                 (s, d_model)  ⑧
                            │
                          W_O                    (s, d_model)  ⑨
                            ▼
                        MHA 输出 (s, d_model)
```

> 因果掩码使 scores 变成下三角有效：位置 $i$ 只能看 $\le i$ 的 token。

其中每个头独立计算：

$$
\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}} + M\right) V_i,
\qquad
\text{MHA}(x') = [\text{head}_1;\dots;\text{head}_h]\,W_O
$$

$M$ 是**因果掩码**（causal mask）：$M_{ij} = -\infty$ 当 $j > i$，否则 $0$。它保证位置 $i$ 只能看到
$\le i$ 的 token——这正是 **decoder-only 自回归**的本质，也是它区别于 encoder（双向可见）的关键。

> **参数量**：MHA 的 4 个投影 $W_Q,W_K,W_V,W_O$ 各 $d_\text{model}^2$，合计 $4\,d_\text{model}^2$。
> 计算量主线是 ④⑦ 两个 $(h,s,d_k)\times(h,s,s)$ 量级的批矩阵乘，随序列长度呈 $O(s^2)$。

### 3.5 前馈网络 FFN

对每个 token 独立地做一次「升维 → 非线性 → 降维」。设输入 $x'' = \text{RMSNorm}(x)$，$(s, d_\text{model})$。

**经典两层 MLP**（最直观的版本）：


| 步骤    | 运算                                                 | 输出维度              |
| ------- | ---------------------------------------------------- | --------------------- |
| ① 升维 | $u = x'' W_1$，$W_1\in(d_\text{model}, d_\text{ff})$ | $(s, d_\text{ff})$    |
| ② 激活 | $a = \text{GELU}(u)$（逐元素）                       | $(s, d_\text{ff})$    |
| ③ 降维 | $y = a W_2$，$W_2\in(d_\text{ff}, d_\text{model})$   | $(s, d_\text{model})$ |

```
   x'' (s, d_model)
        │
        │  W_1 (d_model, d_ff)          升维
        ▼
    u  (s, d_ff)   ◄── 宽度膨胀到 d_ff (≈4·d_model)
        │
        │  GELU（逐元素，形状不变）
        ▼
    a  (s, d_ff)
        │
        │  W_2 (d_ff, d_model)          降维
        ▼
    y  (s, d_model)  ◄── 收回 d_model
```

$d_\text{ff}$ 通常取 $\approx 4\,d_\text{model}$，在 token 维度上逐个独立处理，不跨 token 交互
（跨 token 交互全部由 MHA 完成）。

> **现代变体 SwiGLU**：LLaMA 实际用 SwiGLU，用两个升维投影 `gate`、`up` 加门控：
> $y = \big(\text{SiLU}(x''W_\text{gate}) \odot (x''W_\text{up})\big)\,W_\text{down}$，
> 其中 $W_\text{gate},W_\text{up}\in(d_\text{model}, d_\text{ff})$、$W_\text{down}\in(d_\text{ff}, d_\text{model})$。
> 维度流与经典版一致（$d_\text{model}\to d_\text{ff}\to d_\text{model}$），只是把「一次升维 + 激活」换成
> 「两次升维 + 门控乘法」。为保持「最普通」，本文主线用经典两层 MLP。

### 3.6 残差连接与 Pre-Norm

每个子层的输出都**加回**其输入（残差）：

$$
x \leftarrow x + \text{MHA}(\text{RMSNorm}(x)), \qquad
x \leftarrow x + \text{FFN}(\text{RMSNorm}(x))
$$

- 加法两侧都是 $(s, d_\text{model})$，形状严格一致才能相加；
- **Pre-Norm**（Norm 在子层内、残差在外）让残差主干始终是未经归一化的 $x$，
  梯度可沿残差路径无衰减回传，是深层稳定训练的核心。

本文的 **Pre-Norm** 与原始 Transformer 的 **Post-Norm** 对比（关键差别：Norm 放在残差相加之前还是之后）：

```
      Pre-Norm（本文，LLaMA 风格）            Post-Norm（原始 Transformer）

      x ─┬───────────────┐                  x ─┬───────────────┐
         │               │                     │               │
         ▼               │残差                  ▼               │残差
      RMSNorm            │                   Sublayer          │
         │               │                     │               │
         ▼               │                     ▼               │
      Sublayer           │                    (＋)◄────────────┘
         │               │                     │
         ▼               │                     ▼
        (＋)◄────────────┘                  RMSNorm
         │                                     │
         ▼                                     ▼
   残差主干是干净的 x                      归一化夹在残差主干上
   （梯度无衰减直达底层，易堆深层）        （深层时梯度易受 Norm 影响，需 warmup）
```

### 3.7 Final Norm + LM Head

$L$ 层之后，先做一次 RMSNorm，再用 LM Head 投影到词表维得到 logits：


| 项            | 运算                                                                           | 维度                  |
| ------------- | ------------------------------------------------------------------------------ | --------------------- |
| Final RMSNorm | 归一化$x_L$                                                                    | $(s, d_\text{model})$ |
| LM Head       | $\text{logits} = x_L W_O^\text{head}$，$W_O^\text{head}\in(d_\text{model}, V)$ | $(s, V)$              |

```
   x_L (s, d_model)   ← 最后一个 Block 的输出
        │
   Final RMSNorm
        ▼
      (s, d_model)
        │
   LM Head  W_head (d_model, V)      投影到词表维
        ▼
   logits (s, V) ──► softmax / 采样 ──► 下一个 token
        │
        └─ 推理时通常只取最后一个位置: logits[-1] (1, V)
```

> **权重共享（weight tying）**：许多模型让 LM Head 复用输入 Embedding 的转置
> $W_O^\text{head} = W_E^\top$，省下一份 $V\times d_\text{model}$ 参数。
> 自回归推理时通常只需**最后一个 token** 的 logits $(1, V)$ 来采样下一个词。

---

## 4. 完整前向的维度追踪表

把一次前向的每一步张量形状串起来（省略 batch 维）：


| 阶段                   | 张量        | 维度                    |
| ---------------------- | ----------- | ----------------------- |
| 输入                   | `input_ids` | $(s,)$                  |
| Embedding 后           | $x_0$       | $(s, d_\text{model})$   |
| — 进入 Block$\ell$ — | $x$         | $(s, d_\text{model})$   |
| RMSNorm（attn）        |             | $(s, d_\text{model})$   |
| QKV 投影               | $Q,K,V$     | 各$(s, d_\text{model})$ |
| 拆头                   | $Q,K,V$     | 各$(h, s, d_k)$         |
| 注意力打分             | scores      | $(h, s, s)$             |
| 加权求和 + 合头        |             | $(s, d_\text{model})$   |
| $W_O$ 投影             |             | $(s, d_\text{model})$   |
| 残差相加               | $x$         | $(s, d_\text{model})$   |
| RMSNorm（ffn）         |             | $(s, d_\text{model})$   |
| FFN 升维               |             | $(s, d_\text{ff})$      |
| FFN 降维               |             | $(s, d_\text{model})$   |
| 残差相加               | $x$         | $(s, d_\text{model})$   |
| — Block$\ell$ 输出 — |             | $(s, d_\text{model})$   |
| （重复$L$ 次）         | $x_L$       | $(s, d_\text{model})$   |
| Final RMSNorm          |             | $(s, d_\text{model})$   |
| LM Head                | logits      | $(s, V)$                |

**一句话记忆**：除了 embedding 查表把 $V\to d_\text{model}$、FFN 内部临时升到 $d_\text{ff}$、
注意力内部临时展开成 $(h,s,d_k)$/$(h,s,s)$、以及末尾 LM Head 把 $d_\text{model}\to V$ 之外，
**主干残差流始终是 $(s, d_\text{model})$**。

---

## 5. 具体数值示例（LLaMA-7B 配置）

代入 $d_\text{model}=4096,\ h=32,\ d_k=128,\ d_\text{ff}=11008,\ L=32,\ V=32000$，
处理一条 $s=1024$ 的序列：


| 张量                      | 维度                  | 元素数（近似） |
| ------------------------- | --------------------- | -------------- |
| Embedding 权重$W_E$       | $(32000, 4096)$       | 1.31 亿        |
| 每层 QKV+O 投影           | $4\times(4096,4096)$  | 6710 万 / 层   |
| 每层 FFN（SwiGLU 三投影） | $3\times(4096,11008)$ | 1.35 亿 / 层   |
| 单序列隐藏态$x$           | $(1024, 4096)$        | 419 万         |
| 单头注意力打分 scores     | $(32, 1024, 1024)$    | 3355 万        |
| logits                    | $(1024, 32000)$       | 3277 万        |

单层参数 $\approx 6710\text{万} + 1.35\text{亿} \approx 2.02$ 亿，$\times 32$ 层 $\approx 6.5$ 亿…… 加上
embedding 与其余，总计约 **67 亿参数**（7B）。注意力打分张量随 $s^2$ 膨胀——$s$ 翻倍，
scores 显存/算力翻 4 倍，这是长上下文的主要开销来源。

---

## 6. Prefill 与 Decode：推理时的维度差异

自回归推理分两个阶段，唯一变化的就是序列维 $s$：


| 阶段                       | 一次前向处理的 token              | 隐藏态维度            | KV Cache                     |
| -------------------------- | --------------------------------- | --------------------- | ---------------------------- |
| **Prefill**（处理 prompt） | 一次性喂入全部$s$ 个 prompt token | $(s, d_\text{model})$ | 写入$s$ 个位置的 K/V         |
| **Decode**（逐字生成）     | 每步只喂**1 个**新 token          | $(1, d_\text{model})$ | 复用历史 KV，仅追加 1 个位置 |

Decode 阶段每步的注意力打分是 $(h, 1, s_\text{ctx})$：新 token（1 个 query）对历史全部
$s_\text{ctx}$ 个 K/V 做注意力。**历史 K/V 缓存在显存里避免重算**，这就是 KV Cache。
每 token 每层缓存 $2 \times h \times d_k = 2\,d_\text{model}$ 个元素（K 和 V），
其显存随 $s_\text{ctx}\times L$ 线性增长——正是它撑起了并发上限，也催生了 GQA/MLA 等 KV 压缩方案
（见 `../MHA_MQA_GQA.md`、`../MLA.md`）。

---

## 7. logits 之后：约束解码、采样与输出解析

前面 §1–§6 讲的是「input_ids → logits」的网络前向。但一次真正的推理请求，在拿到每一步的
$(1, V)$ logits 之后还有一整条**后处理链路**：先（可选）用**语法约束**把非法 token 的 logits 打成
$-\infty$，再**采样**出 token；生成完整文本后，还要把这段纯文本**结构化拆分**成「思维链（reasoning）+ 正文 + 工具调用（tool call）」返回给上层。本节把这条链路补全。

### 7.0 全景：一步解码 + 一次输出解析

```
每步解码循环（逐 token）
┌───────────────────────────────────────────────────────────┐
│                                                           │
│   x_L (1, d_model) ─► Final Norm ─► LM Head ─► logits      │ ← §6
│                                                 │         │
│                    ┌────────────────────────────┤         │
│                    ▼                             │         │
│         ① grammar mask（约束解码，禁止项置 -inf）  │         │ ← §7.1
│                    │                             │         │
│                    ▼                             │         │
│         ② 温度 / top-k / top-p 截断              │         │ ← §7.2
│                    │                             │         │
│                    ▼                             │         │
│         ③ 重复 / 频率 / 存在惩罚                  │         │ ← §7.2
│                    │                             │         │
│                    ▼                             │         │
│              softmax ─► 采样 ─► next token        │         │ ← §7.2
│                    │                             │         │
│                    ├─► accept_token（推进语法状态机）        │ ← §7.1
│                    ├─► 追加到 KV Cache                      │
│                    └─────────────────────────────┘         │
│                          （回到下一步，直到 EOS）           │
└───────────────────────────────────────────────────────────┘
                              │
                              ▼  生成结束（EOS / max_new_tokens）
     完整输出文本  "……<think>推理…</think>正文…<tool▁call>…"
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                       ▼
  ④ reasoning 解析        ⑤ 正文                 ⑥ tool-call 解析     ← §7.3 / §7.4
  reasoning_content       content                tool_calls[]
```

- **① 约束 mask、③ 惩罚、② 采样**发生在**每一步解码**、在 logits 上原地操作（GPU 侧）。
- **④⑤⑥ 输出解析**发生在**文本生成之后**（或流式地边生成边增量解析），在 CPU 侧对**字符串**做状态机拆分。

### 7.1 约束解码（Constrained Decoding / Grammar）

当请求要求输出严格符合某种结构（JSON Schema、正则、EBNF、结构化标签，或"必须调用某工具"）时，
SGLang 在**采样前**对 logits 施加**语法掩码**：把当前语法状态下**不允许出现**的 token 的 logits 置为
$-\infty$，从而保证采样结果一定合法。

统一抽象（所有后端实现同一组接口，见 `../../python/sglang/srt/constrained/`）：


| 步骤     | 接口                  | 作用                                        | 维度                                            |
| -------- | --------------------- | ------------------------------------------- | ----------------------------------------------- |
| 分配掩码 | `allocate_vocab_mask` | 按 batch 分配词表掩码                       | 位掩码$(b, \lceil V/32\rceil)$ 或 bool $(b, V)$ |
| 填充掩码 | `fill_vocab_mask`     | 由引擎根据**当前语法状态**算出禁止 token    | 同上                                            |
| 施加掩码 | `apply_vocab_mask`    | 原地把禁止项置$-\infty$（Triton/CUDA 算子） | logits$(b, V)$ 不变                             |
| 推进状态 | `accept_token`        | 采样出 token 后推进语法状态机               | —                                              |
| 回退     | `rollback(k)`         | 回退 k 个 token（投机解码 verify 失败时用） | —                                              |
| 跳跃前进 | `try_jump_forward`    | 语法唯一确定的片段直接"跳过"、省若干次前向  | —                                              |

三个后端差异在「语法如何编译成状态机」与「掩码如何表示」：


| 后端                 | 状态机         | 掩码表示                                | 表达力                      | 特点                                |
| -------------------- | -------------- | --------------------------------------- | --------------------------- | ----------------------------------- |
| **xgrammar**（默认） | 下推自动机 PDA | **位掩码**（1 bit/token，省 32× 显存） | CFG（可处理 JSON 嵌套括号） | 最快最全，支持 rollback、结构化标签 |
| outlines             | 有限状态机 FSM | bool 张量                               | 仅正则（无法表达任意嵌套）  | 不支持 rollback，与投机解码兼容弱   |
| llguidance           | LLMatcher      | 位掩码                                  | JSON/正则/结构化标签        | rollback 友好                       |

> **与推理的衔接**：`GrammarManager` 用线程池**异步编译**语法（编译较慢），编译好后每个请求持有一个语法对象，
> 解码循环里 `fill_vocab_mask → apply_vocab_mask → 采样 → accept_token` 循环推进。
>
> **reasoner 包装**：`ReasonerGrammarBackend` 对带思考段的模型（如 DeepSeek-R1），在
> `<think>...</think>` 内**不施加约束**（让模型自由推理），只在思考结束后才对最终答案施加结构约束。

### 7.2 采样（Sampling）

约束 mask 之后，对 logits 做温度缩放、top-k/top-p 截断、重复/频率惩罚等，再 softmax 采样出下一个 token：


| 处理               | 作用                                                  | 对 logits/概率的影响 |
| ------------------ | ----------------------------------------------------- | -------------------- |
| 温度$T$            | $\text{logits}/T$；$T\to 0$ 趋近 greedy，$T$ 大更随机 | 缩放                 |
| top-k              | 只保留概率最高的$k$ 个 token                          | 其余置$-\infty$      |
| top-p（nucleus）   | 保留累计概率达$p$ 的最小集合                          | 其余置$-\infty$      |
| 重复/频率/存在惩罚 | 降低已出现 token 的分数                               | 相应位置减一个惩罚值 |

采样出的 token 会：① `accept_token` 推进语法状态机（若有约束）；② 追加进 KV Cache；③ 作为下一步的输入 token。
如此循环直到 **EOS** 或达到 **max_new_tokens**。

### 7.3 推理内容解析（Reasoning Parser）

生成结束后拿到的是一段**纯文本**，形如 `<think>先想一想……</think>这是给用户的正文`。
`ReasoningParser`（`../../python/sglang/srt/parser/reasoning_parser.py`）按模型的 think 起止标记把它拆成两段：

```
  raw:  "<think> 让我算一下 3+5=8 </think> 答案是 8。"
                 └──── reasoning_text ────┘        └ normal_text ┘
   parser.parse_non_stream(raw)  ─►  ("让我算一下 3+5=8", "答案是 8。")
```

- 起止标记因模型而异：DeepSeek-R1/Qwen3 用 `<think>…</think>`，GPT-OSS 用 Harmony 频道，
  DeepSeek-V4 用 `</think>` + DSML（见 §7.4）。
- 支持**流式**：`parse_streaming_increment(new_text)` 维护 `_buffer`，边收边判断当前在 reasoning 段还是正文段，
  分别产出 `reasoning_text` / `normal_text`；只有 `separate_reasoning=True` 时才拆分，否则整段作为正文返回。
- 只有 API 层（`serving_chat.py` 等）想把「思考」与「答案」分成两个字段（`reasoning_content` / `content`）时才启用。

### 7.4 工具调用解析（Tool-Call Parser）

若正文里还夹带**工具调用**标记，`FunctionCallParser`
（`../../python/sglang/srt/function_call/function_call_parser.py`）再把它拆出来。它本身不含解析规则，
而是按 `--tool-call-parser` 名称选出对应模型的 **detector**，委托其解析：

```
  content:  "好的，我来查天气。<tool▁call>get_weather\n{\"city\":\"北京\"}</tool▁call>"
                └ 留在 content 的正文 ┘  └──────── 被 detector 拆成 tool_calls[] ────────┘
   parser.parse_non_stream(content)
     ─►  normal_text = "好的，我来查天气。"
         tool_calls  = [ ToolCallItem(name="get_weather", parameters='{"city":"北京"}') ]
```

- **一模型一 detector**：`ToolCallParserEnum` 把 `deepseekv4`/`qwen25`/`hermes`/… 映射到各自的 detector 类。
  不同模型标记不同：Qwen 系用 `<tool_call>…</tool_call>`（JSON），DeepSeek-V4 用 **DSML**（类 XML 的
  `<｜tool▁calls▁begin｜>…`），pythonic 模型直接输出函数调用语法。
- **流式**：`parse_streaming_increment` 用状态机做增量解析，靠**稳定前缀 diff** 把已确定的参数片段逐步吐给 SSE。
- **约束协同**：detector 可通过 `structural_tag` / JSON Schema 生成 §7.1 的语法约束，
  在解码阶段就**强制**模型按工具的参数 schema 输出，从源头保证 tool call 合法（受 `ToolStrictLevel` 控制）。

### 7.5 一句话串起来

**logits $(1,V)$ →（可选）语法 mask 打 $-\infty$ → 温度/截断/惩罚 → softmax 采样 → token → 推进语法状态机 + 写 KV →
循环到 EOS →** 得到整段文本 **→ reasoning 解析（思考/正文）→ tool-call 解析（正文/工具调用）→** 返回结构化结果。
前半段（约束+采样）在 GPU 上逐 token 做，后半段（两个解析器）在 CPU 上对字符串做状态机拆分。

---

## 8. 参考与延伸

- 同目录理论文档：`../MHA_MQA_GQA.md`（本文用的是其中 $h_{kv}=h$ 的 MHA 特例）、
  `../MLA.md`（KV Cache 压缩）、`../Glossary.md`（术语表）。
- 并行化：`../distributed/TP.md`（这些层如何按 head / 张量维 / 词表维切到多卡）。
- 经典论文：*Attention Is All You Need*（Vaswani et al., 2017，原始 Transformer）、
  LLaMA（Touvron et al., 2023，Pre-Norm + RMSNorm + RoPE + SwiGLU 的现代 decoder-only 范式）。
