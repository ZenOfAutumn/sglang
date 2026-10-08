# 投机解码（Speculative Decoding）原理与 SGLang 实现

本文详细介绍投机解码的动机、数学原理（拒绝采样的正确性证明）、树形草稿与验证机制，以及它在 SGLang 中的实现结构与关键参数。

## 1. 它解决什么问题

自回归解码每生成 1 个 token 就要做一次完整的目标模型 forward。decode 阶段是典型的 **memory-bound** 场景：每一步都要把全部权重从 HBM 搬进计算单元，却只产出 1 个 token，算力利用率极低。

投机解码的核心思路是：**用一个便宜的"草稿机制"一次提出多个候选 token，再让目标模型在一次 forward 里并行验证它们**。验证一批 token 的成本与生成 1 个 token 的成本接近（同样是读一遍权重，只是 batch 维度上多了几个位置），因此若平均每次验证能接受 $\tau$ 个 token，解码步数就近似缩小为原来的 $1/\tau$。

关键约束：**输出分布必须与目标模型单独解码完全一致**（无损加速）。这由下文的拒绝采样机制保证。

## 2. 基本流程

一轮投机解码（一个 draft-verify 周期）包含三步：

1. **Draft（起草）**：草稿机制（小模型 / EAGLE 头 / n-gram 匹配等）从当前上下文出发，自回归地生成 $k$ 个草稿 token（或一棵草稿树）。
2. **Verify（验证）**：目标模型对「上下文 + 草稿序列」做一次 forward，得到每个草稿位置上目标模型自己的下一 token 分布 $p(\cdot)$。
3. **Accept/Reject（接受判定）**：从第一个草稿开始逐个判定。一旦某个草稿被拒绝，丢弃它及其后所有草稿；无论接受多少个草稿，目标模型都会在最后一个接受位置**额外产出 1 个 token**——即 **bonus token**。

因此一轮至少前进 1 个 token（全部草稿被拒时只剩 bonus token），至多前进 $k+1$ 个。

### 术语与计数约定（与 SGLang 代码一致）

| 名称 | 含义 | 是否含 bonus |
| --- | --- | --- |
| `correct_drafts` / `num_correct_drafts` | 通过验证的草稿 token | 否 |
| `accept_tokens` / `num_accept_tokens` | 本轮实际前进的 token 序列 | 是（= correct_drafts + 1 个 bonus） |
| `bonus_token` | 目标模型验证时"白送"的下一个 token | — |
| `accept_rate`（论文中的 $\alpha$） | 单个草稿被接受的概率 = correct_drafts / proposed_drafts | 否 |
| `accept_length`（论文中的 $\tau$） | 每次 verify 平均前进的 token 数 | 是 |

## 3. 数学原理：投机采样（Speculative Sampling）的正确性

设草稿模型在某位置的分布为 $q(x)$，目标模型分布为 $p(x)$，草稿 token 为 $\tilde{x} \sim q$。本节公式涉及的符号统一说明如下（各公式下方再标注该公式专属的含义）：

| 符号 | 含义 |
| --- | --- |
| $x$ | 输出词表中的任意一个候选 token；出现在 $\sum_x$、$\max$、$p(\cdot)$ 里时表示「遍历整个词表逐个取到」 |
| $\tilde{x}$ | 草稿模型在该位置**实际采样出**的那个草稿 token，服从草稿分布 $q$，即 $\tilde{x}\sim q$ |
| $p(\cdot)$ | 目标模型在该位置给出的下一 token 分布（我们要无损复现的「真值」分布）；$p(x)$ 是其中 token $x$ 的概率 |
| $q(\cdot)$ | 草稿模型在同一位置给出的下一 token 分布；$q(x)$ 是其中 token $x$ 的概率 |
| $p(\tilde{x})/q(\tilde{x})$ | 草稿 token 在目标分布与草稿分布下的概率之比，作为接受比例（草稿不可能采到 $q(\tilde{x})=0$ 的 token，该情形视为 $p/q=+\infty$，无条件接受） |
| $x'$ | 求和哑变量，含义与 $x$ 相同，仅用于把「被遍历求和的 token」与「当前讨论的 token $x$」区分开 |
| $p'(x)$ | 拒绝草稿后、用于重采样修正 token 的**残差分布** |
| $X$ | 该位置最终输出的随机变量 |
| $\alpha$ | 单 token 接受概率（接受率），也是衡量草稿质量的核心指标 |
| $\tau$ | 每轮 verify 平均前进的 token 数（接受长度），含 bonus token |
| $k$ | 本轮草稿步数，即链式草稿的链长 / 树形草稿的深度 |

判定规则（Leviathan et al. 2023）：

1. 以概率 $\min\!\left(1, \dfrac{p(\tilde{x})}{q(\tilde{x})}\right)$ **接受** $\tilde{x}$；
2. 若拒绝，则从**残差分布**中重采样：

$$
p'(x) = \frac{\max\big(0,\; p(x) - q(x)\big)}{\sum_{x'} \max\big(0,\; p(x') - q(x')\big)}
$$

逐项含义：分子 $\max(0,\,p(x)-q(x))$ 是 token $x$ 上「目标模型想说、草稿模型却提名不足」的**概率缺口**（$q(x)\ge p(x)$ 的 token 被 $\max$ 截为 0，不再补）；分母 $\sum_{x'}\max(0,p(x')-q(x'))$ 是全部缺口之和，充当**归一化常数**，其数值恰好等于总拒绝概率 $1-\alpha$（推导见下），除以它才使 $p'$ 成为合法的概率分布。

**正确性证明**：最终输出 token 为 $x$ 的概率是「草稿恰好是 $x$ 且被接受」加上「草稿被拒绝后重采样到 $x$」两部分之和：

$$
P(X = x) = q(x)\min\!\left(1, \frac{p(x)}{q(x)}\right) + \Big(1 - \sum_{x'} \min\big(p(x'), q(x')\big)\Big)\, p'(x)
$$

两项的来历：
- **第一项** $q(x)\min\!\left(1,\frac{p(x)}{q(x)}\right)$——「草稿恰好抽到 $x$ 且被接受」的概率：$q(x)$ 是草稿抽到 $x$ 的概率，$\min(1,p(x)/q(x))$ 是它被接受的概率，二者相乘即 $\min(p(x),q(x))$。
- **第二项** $\left(1-\sum_{x'}\min(p(x'),q(x'))\right)p'(x)$——「草稿被拒绝、再从残差分布重采样到 $x$」的概率：括号内是**总拒绝概率**（因为 $\sum_{x'}\min(p(x'),q(x'))=\alpha$ 是总接受概率），$p'(x)$ 是拒绝后落到 $x$ 的条件概率。

第一项等于 $\min(p(x), q(x))$；第二项中拒绝概率 $1 - \sum \min(p, q) = \sum \max(0, p - q)$ 恰好是残差分布的归一化常数，于是第二项等于 $\max(0, p(x) - q(x))$。两项相加：

$$
P(X = x) = \min\big(p(x), q(x)\big) + \max\big(0,\; p(x) - q(x)\big) = p(x)
$$

即无论草稿分布 $q$ 好坏，**输出分布严格等于目标分布 $p$**。$q$ 的质量只影响速度（接受率），不影响正确性。

单 token 接受率的期望：

$$
\alpha = \sum_x \min\big(p(x), q(x)\big) = 1 - \tfrac{1}{2}\lVert p - q \rVert_1
$$

**逐 token 展开**：这个求和形式来自对「草稿会抽到哪个 token」做边缘化——把「抽到 $x$」与「给定 $x$ 后被接受」两步概率相乘再对词表求和：

$$
\alpha = \sum_x \underbrace{q(x)}_{\text{抽到 } x}\cdot\underbrace{\min\!\left(1, \tfrac{p(x)}{q(x)}\right)}_{\text{给定 } x \text{ 的接受概率}} = \sum_x \min\big(p(x), q(x)\big)
$$

与之互补的**总拒绝概率**（即上文 $p'(x)$ 的系数 $1-\alpha$）同样可以逐 token 展开：

$$
1-\alpha = \sum_x q(x)\Big[1 - \min\!\left(1, \tfrac{p(x)}{q(x)}\right)\Big] = \sum_x \max\big(0,\; q(x) - p(x)\big)
$$

注意这里是对**全部** token 求和（**包含 $x$ 自身**）：「拒绝」不是"抽到非 $x$ 的 token"才可能发生，它是个与具体输出 $x$ 无关的常数。两个求和的对偶关系：

| 量 | 逐 token 形式 | 非零质量来自哪些 token |
| --- | --- | --- |
| 总接受概率 $\alpha$ | $\sum_x\min(p,q)$ | 两个分布的重叠部分 |
| 总拒绝概率 $1-\alpha$ | $\sum_x\max(0,\; q-p)$ | 只有草稿**过度提名**的 token（$q>p$） |
| 残差分子 $\max(0,\; p-q)$ | 逐项 | 只有目标**提名不足**的 token（$p>q$） |

三者由同一个恒等式串联：$\min(p,q)+\max(0,p-q)=p$ 对 $x$ 求和即得 $\alpha+\sum_x\max(0,p-q)=1$，于是

$$
1-\alpha = \sum_x \max\big(0,\; p(x)-q(x)\big) = \sum_x \max\big(0,\; q(x)-p(x)\big) = \tfrac{1}{2}\lVert p-q\rVert_1
$$

含义：**拒绝的质量从「草稿多说的 token」上收走，残差再把它补给「目标想说而草稿少说的 token」，收走量 $=$ 补上量**。若草稿恰好抽到 $x$ 又被拒，则必有 $p(x)<q(x)$，而残差在该 token 上恰为 $0$——两条通道天然互补，不会重复计数。

其中 $\lVert p-q\rVert_1=\sum_x|p(x)-q(x)|$ 是 $p$ 与 $q$ 的 $L_1$ 距离，$\tfrac12\lVert p-q\rVert_1$ 即两分布的**总变差距离（TV distance）**，衡量草稿分布离目标分布有多远。$\alpha$ 的三种等价读法：草稿被接受的概率、两个分布逐 token 取小后的「重叠面积」、以及 $1$ 减去 TV 距离。$p=q$ 时 $\alpha=1$（草稿永远被接受），两分布完全错开时 $\alpha=0$。

若各位置接受近似独立且接受率为 $\alpha$，链长为 $k$ 的草稿的期望前进长度：

$$
\mathbb{E}[\tau] = \frac{1 - \alpha^{k+1}}{1 - \alpha} = \sum_{i=0}^{k}\alpha^i
$$

其中 $\alpha^i$ 是「前 $i$ 个草稿全部通过」的概率（独立假设下即连乘）。求和的含义：本轮恰好前进 $i+1$ 个 token（$i$ 个草稿 + 1 个 bonus/修正 token）的概率贡献，$i$ 从 $0$（第一个草稿就被拒，只前进 $1$ 个）累加到 $k$（全部 $k$ 个草稿通过，前进 $k+1$ 个）。$\alpha\to1$ 时 $\mathbb{E}[\tau]\to k+1$（逼近上限），$\alpha\to0$ 时 $\mathbb{E}[\tau]\to1$（退化为普通解码）。

$\mathbb{E}[\tau]$ 就是端到端**加速比上界的主项**，与实测加速比的完整换算（还要扣掉草稿与验证的成本）见 §7.1。

**Greedy 解码退化情形**：温度为 0 时验证退化为精确匹配——草稿 token 等于目标模型 argmax 则接受，否则拒绝并以 argmax 作为修正 token。SGLang 的 `verify_tree_greedy_func`（`eagle_utils.py`）即此路径。

## 4. 链式草稿（Chain Drafting）的详细过程

链式草稿是最原始的形态（Leviathan 2023 / Chen 2023 的设定）：草稿是一条单链，每步只保留 1 个候选。完整走一遍一轮的流程：

设当前已确认的上下文为 $x_{1..n}$，草稿步数为 $k$。

### 4.1 Draft 阶段：草稿模型自回归 $k$ 步

草稿模型 $q$ 从 $x_{1..n}$ 出发，逐 token 生成：

$$
\tilde{x}_1 \sim q(\cdot \mid x_{1..n}),\quad
\tilde{x}_2 \sim q(\cdot \mid x_{1..n}, \tilde{x}_1),\quad
\dots,\quad
\tilde{x}_k \sim q(\cdot \mid x_{1..n}, \tilde{x}_{1..k-1})
$$

这是 $k$ 次串行的草稿模型 forward（草稿模型小，单步便宜），同时要**保留每一步的完整分布** $q_i(\cdot)$——后面接受判定要用，不能只存采样出的 token。

### 4.2 Verify 阶段：目标模型一次 forward

把 $[x_{1..n}, \tilde{x}_1, \dots, \tilde{x}_k]$ 拼成一段送入目标模型 $p$，一次 forward 得到 $k+1$ 个位置的分布：

$$
p_1(\cdot \mid x_{1..n}),\ p_2(\cdot \mid x_{1..n}, \tilde{x}_1),\ \dots,\ p_{k+1}(\cdot \mid x_{1..n}, \tilde{x}_{1..k})
$$

注意 $p_{k+1}$ 是「假设全部草稿都对」时下一个 token 的分布——bonus token 就从这里来。这一步在推理引擎里就是一次长度为 $k$ 的 **extend/prefill 式 forward**（复用已有 KV cache，只算新 token），成本与 decode 一步同量级。

### 4.3 Accept 阶段：从左到右逐个判定

对 $i = 1, 2, \dots, k$ 依次执行：

1. 采样 $u \sim \mathrm{Uniform}(0,1)$；
2. 若 $u < \dfrac{p_i(\tilde{x}_i)}{q_i(\tilde{x}_i)}$ → **接受** $\tilde{x}_i$，继续判定下一个；
3. 否则 → **拒绝**，从残差分布 $p'_i(x) \propto \max(0,\, p_i(x) - q_i(x))$ 重采样出修正 token $x^{\ast}$，**本轮到此结束**，$\tilde{x}_{i..k}$ 全部作废。

**$u < p/q$ 的真实目的**：它不是在判断草稿 token「好不好」，而是在做**频率校正**——把该 token 的实际输出频率从草稿分布的 $q(x)$ 精确压到目标分布的 $p(x)$：

- $p(\tilde{x}) \ge q(\tilde{x})$：草稿模型对它提名得不够多，来一个收一个（无条件接受）；
- $p(\tilde{x}) < q(\tilde{x})$：草稿模型过度提名，按比例 $p/q$ 接受，使「被提名且被接受」的联合频率恰为 $q(x)\cdot\frac{p(x)}{q(x)} = p(x)$。

于是每个 token 经接受通道的输出频率是 $\min(p(x), q(x))$，不足 $p(x)$ 的缺口 $\max(0, p-q)$ 正好由拒绝后的残差重采样补齐。这套规则是 $p$ 与 $q$ 之间的**最大耦合（maximal coupling）**：在输出严格服从 $p$ 的前提下，接受率 $\alpha = \sum\min(p,q) = 1 - \tfrac{1}{2}\lVert p - q\rVert_1$ 已是理论上限——任何无损方案都不可能接受得更多。

**残差分布为什么是 $\max(0, p-q)$ 再归一化，而不是直接用 $p - q$**：

- **为什么要 $\max(0, \cdot)$**：对于 $q(x) \ge p(x)$ 的 token（草稿模型过度提名的），它们经「提名 + 按 $p/q$ 接受」这条通道的输出频率已经拉满到 $p(x)$，重采样时必须给它们**零概率**，否则总频率会超过 $p(x)$。$p - q$ 在这些 token 上恰好 $\le 0$，被 $\max$ 截掉。
- **为什么要归一化**：$\max(0, p-q)$ 逐项求和不等于 1——其和恰好等于本次的拒绝概率 $1 - \sum\min(p,q)$，除以它才是合法分布。

所以重采样只会落在「目标模型想说但草稿模型提名不足」的 token 上，补的正是接受通道没补够的那部分缺口。两条通道相加：$\min(p,q) + \max(0, p-q) = p$，逐 token 严格无损。

结束时有两种情形：

| 情形 | 本轮前进的 token | 数量 |
| --- | --- | --- |
| 在第 $i$ 个草稿处被拒 | $\tilde{x}_{1..i-1}$ + 修正 token $x^{\ast}$ | $i$ 个（$1 \le i \le k$） |
| 全部 $k$ 个草稿通过 | $\tilde{x}_{1..k}$ + 从 $p_{k+1}$ 采样的 bonus token | $k+1$ 个 |

两种情形都恰好多得 1 个非草稿 token（修正 token 或 bonus token），所以**每轮最少前进 1 个、最多 $k+1$ 个**。修正 token / bonus token 之后的 KV cache 尚未计算，作为下一轮的起点。

### 4.4 一个具体例子

设 $k=4$，上下文为 "The capital of France"，草稿链为 `[" is", " Paris", ",", " which"]`：

```
位置      草稿 token   p/q 判定        结果
i=1       " is"        u < p/q  ✓     接受
i=2       " Paris"     u < p/q  ✓     接受
i=3       ","          u ≥ p/q  ✗     拒绝 → 从 max(0, p−q) 重采样得 "."
i=4       " which"     （不再判定，作废）
```

本轮前进 3 个 token：`" is" " Paris" "."`。对应计数：`num_correct_drafts = 2`（通过验证的草稿），`num_accept_tokens = 3`（含修正 token）。

### 4.5 数值例：把一轮的账算一遍

§4.4 只看了 token 的流向，看不出概率是怎么对上的。下面给一组真实数字，把「接受通道 + 残差通道」的账目算清楚。

#### (a) 单个位置：一个微型词表上的完整分布

设某个位置上草稿分布 $q$ 与目标分布 $p$ 如下（词表只有 5 个 token）：

| token | $q$（草稿） | $p$（目标） | $p/q$ | 接受通道 $\min(p,q)$ | 残差通道 $\max(0,p-q)$ |
| --- | --- | --- | --- | --- | --- |
| Paris | 0.40 | 0.45 | 1.125 | 0.40 | 0.05 |
| London | 0.35 | 0.10 | 0.286 | 0.10 | 0.00 |
| Rome | 0.15 | 0.25 | 1.667 | 0.15 | 0.10 |
| Tokyo | 0.10 | 0.10 | 1.000 | 0.10 | 0.00 |
| Lyon | 0.00 | 0.10 | — | 0.00 | 0.10 |
| **合计** | **1.00** | **1.00** | | **0.75** | **0.25** |

几行读数：

- 单步接受率 $\alpha=\sum_x\min(p,q)=0.75$，拒绝概率 $1-\alpha=0.25$；残差通道的未归一化和 $\sum_x\max(0,p-q)=0.05+0.10+0.10=0.25$ **恰好等于它**——这不是巧合，而是 §4.3 那条恒等式的数值形态。
- $p'$ 归一化：$\max(0,p-q)$ 除以 0.25，得 Paris 0.20、London 0、Rome 0.40、Tokyo 0、Lyon 0.40。
- 逐 token 校验 $\min(p,q)+\max(0,p-q)=p$：Paris $0.40+0.05=0.45$ ✓、Rome $0.15+0.10=0.25$ ✓、Lyon $0+0.10=0.10$ ✓。

三种抽签结果（"草稿抽到"由 $q$ 抽出，$u\sim\mathrm{Uniform}(0,1)$）：

| 情形 | 草稿抽到 | $p/q$ | 抽到的 $u$ | 判定 | 最终输出 |
| --- | --- | --- | --- | --- | --- |
| ① | London（$q{=}0.35$） | 0.286 | 0.62 | $u \ge p/q$ → 拒绝 | 从 $p'$ 重采样（Paris .20 / Rome .40 / Lyon .40），设为 Rome |
| ② | London（$q{=}0.35$） | 0.286 | 0.20 | $u < p/q$ → 接受 | London |
| ③ | Paris（$q{=}0.40$） | 1.125 | 任意 | $p \ge q$ → 无条件接受 | Paris |

①②是**同一个草稿 token、同一组分布**，只因 $u$ 不同就走了不同的路：$p<q$ 时既可能接受也可能拒绝——这正是 §4.3 说的"按 $p/q$ 的比例稀释过度提名"。③则不需要抽签，因为 $p/q\ge 1$。

特别注意 **Lyon：草稿从未提名（$q=0$），却有 0.10 的最终概率，全部由残差通道补上**——残差只落在"目标想说、但草稿提名不足"的 token 上。

#### (b) 一整轮 $k=3$

把上面的账在三个位置上串起来。为紧凑起见，每层词表只保留 3 个 token $A,B,C$：

| 位置 | $q_i$ (A/B/C) | $p_i$ (A/B/C) | $p_i/q_i$ | $\alpha_i$ | 草稿抽到 | $u$ | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | .50/.30/.20 | .60/.20/.20 | 1.20/0.667/1.00 | 0.90 | A | — | 接受（$p \ge q$，$u$ 无关） |
| 2 | .20/.50/.30 | .30/.30/.40 | 1.50/0.60/1.333 | 0.80 | B | 0.40 | 接受（$0.40 < 0.60$，注意此处 $p_2(B)=0.30 < q_2(B)=0.50$） |
| 3 | .60/.10/.30 | .20/.30/.50 | 0.333/3.00/1.667 | 0.60 | A | 0.50 | **拒绝**（$0.50 \ge 0.333$） |

第 3 个位置被拒后按规则重采样：

$$
\max(0,\,p_3-q_3) = (0,\ 0.20,\ 0.20),\qquad 0.20+0.20 = 0.40 = 1-\alpha_3
$$

未归一化和恰为拒绝概率（与 (a) 同一恒等式），归一化得 $p_3' = (0,\ 0.50,\ 0.50)$，抽签落到 **C** —— 即修正 token。

本轮结果：

| 项 | 值 |
| --- | --- |
| 前进的 token | $A_1,\ B_2,\ C^{\ast}$（修正 token） |
| `num_correct_drafts` | 2 |
| `num_accept_tokens` | 3（$=2+1$） |
| 作废 | 被拒草稿 $A_3$ 及其后（本例如无） |

再对第 3 个位置做逐 token 的频率校验，确认"两条通道相加 $=p$"：

$$
P(\text{输出}=A) = \min(0.2,0.6) + 0 = 0.2 = p_3(A)
$$

$$
P(\text{输出}=B) = \min(0.3,0.1) + 0.20 = 0.3 = p_3(B)
$$

$$
P(\text{输出}=C) = \min(0.5,0.3) + 0.20 = 0.5 = p_3(C)
$$

注意 $A$：草稿以 0.60 提名它，最终频率被压到 $0.60 \times (0.2/0.6) = 0.20$ —— 这就是 §4.3 说的"过度提名被按比例抽掉"。

#### (c) 顺带校验期望前进长度

取 (a) 的 $\alpha=0.75$，代入 §3 的 $\mathbb{E}[\tau]=\frac{1-\alpha^{k+1}}{1-\alpha}$：

| $k$（草稿步数） | $\mathbb{E}[\tau]$ |
| --- | --- |
| 3 | $\frac{1-0.75^4}{0.25} \approx 2.73$ |
| 5 | $\frac{1-0.75^6}{0.25} \approx 3.29$ |
| 8 | $\frac{1-0.75^9}{0.25} \approx 3.70$ |

$k$ 从 3 加到 8（深度近 3 倍），平均前进长度只从 2.73 涨到 3.70 —— 这就是 §4.6 说"单纯加深链收益递减"的数值形态。

### 4.6 被拒后缀为什么必须作废

第 $i$ 个位置被拒后，实际输出变成了 $x^{\ast} \ne \tilde{x}_i$，而 $\tilde{x}_{i+1..k}$ 是以 $\tilde{x}_i$ 为条件生成的——条件已经不成立，它们对应的目标分布 $p(\cdot \mid \dots, x^{\ast})$ 也从未被计算过。保留它们会破坏无损性，所以只能丢弃，同时回收这些位置已写入的 KV cache。

这正是链式草稿的结构性弱点：**前缀出错，后缀全部陪葬**。期望前进长度 $\mathbb{E}[\tau] = \frac{1-\alpha^{k+1}}{1-\alpha}$ 随 $k$ 增长很快饱和（$\alpha=0.7$ 时 $k$ 从 4 加到 8，$\tau$ 只从 2.75 涨到 3.19），单纯加深链收益递减——这直接引出树形草稿。

## 5. 树形草稿（Tree Drafting）：从链到树

> 树形草稿的完整原理详见同目录 [tree_draft.md](./tree_draft.md)，EAGLE 详见 [eagle.md](./eagle.md)，本节仅概述。

链式草稿的弱点：第 $i$ 个草稿一旦错，其后全部作废。EAGLE / Medusa 等方法改为在每一步保留 top-k 个候选，展开成一棵**草稿树**，让目标模型一次验证整棵树，接受其中最长的正确路径。

树形验证的关键实现点：

- **树形注意力掩码（tree mask）**：树上的多个 token 摊平进同一个序列送入目标模型，但每个节点只能 attend 到它的祖先路径，不能看到兄弟分支。这需要构造自定义 attention mask（`build_tree_kernel_efficient`，`eagle_utils.py`），并要求注意力后端支持自定义 mask 的 target verify 模式。
- **路径选择**：验证后从根开始逐层走：在当前节点的子节点里找目标模型接受的那个，走到没有可接受子节点为止，得到最长接受路径；路径末端追加 bonus token。
- **位置编码与 KV cache**：树上同层节点共享 position，被拒绝分支占用的 KV cache 槽位需要回收，接受路径的 KV 要正确落位（`spec_utils.py` 中的缓存槽分配 / KV 索引生成）。

### EAGLE 的草稿方式

EAGLE 不用独立小模型，而是在目标模型上外挂一个轻量自回归头：输入是目标模型**倒数第二层的 hidden states** 与已生成 token 的 embedding，逐步预测后续 token 的特征并解码出候选。由于复用目标模型的语义特征，接受率显著高于同尺寸的独立草稿模型。EAGLE3 进一步融合多层特征训练草稿头。

## 6. SGLang 中的实现

代码位于 `python/sglang/srt/speculative/`（目录内有 `README_zh.md` 文件清单）。

### 6.1 支持的算法（`spec_info.py` 中的 `SpeculativeAlgorithm`）

| 算法 | 草稿来源 | 说明 |
| --- | --- | --- |
| `EAGLE` / `EAGLE3` | 目标模型外挂 EAGLE 头 | 主力路径，树形草稿 + 树形验证 |
| `STANDALONE` | 独立草稿模型 | 不与目标模型共享 embedding/lm_head |
| `NGRAM` | n-gram 语料匹配 | 无需神经网络草稿，从历史/外部语料检索续写候选（`cpp_ngram/`，见 [ngram.md](./ngram.md)） |
| `FROZEN_KV_MTP` | MTP（multi-token prediction）头 | 冻结 KV 的 MTP 变体 |
| `DFLASH` | DFlash 草稿机制 | 块级草稿 |
| 插件注册 | `SpeculativeAlgorithm.register` | 允许外部注册自定义算法 |

### 6.2 核心组件分工

- **Worker 层**（`eagle_worker_v2.py`、`ngram_worker.py`、`standalone_worker_v2.py` 等）：继承 `base_spec_worker.py` 的 `BaseSpecWorker`/`BaseDraftWorker`，串联 draft → verify → accept 全流程，接入 scheduler 的 overlap 调度。
- **Info 结构**（`eagle_info.py` 等）：`EagleDraftInput` / `EagleVerifyInput` / `EagleVerifyOutput`，承载草稿树张量、tree mask、KV 索引等。
- **工具与内核**（`eagle_utils.py`、`spec_utils.py`、`triton_ops/`）：树构建、greedy 树验证、缓存槽分配、top-k 选择等 Triton/CUDA 内核。
- **CUDA Graph**（`eagle_draft_cuda_graph_runner.py` 等）：草稿模型的 decode/extend 阶段用 CUDA Graph 消除小 kernel 的启动开销——草稿模型很小，launch 开销占比高，CUDA Graph 收益显著。
- **自适应步数**（`adaptive_spec_params.py`、`adaptive_runtime_state.py`）：前者按批次大小路由档位、按接受长度在档内阶梯上调节；后者持有每个候选步数的整套运行时资源并负责原子切换。原理见 §8。

### 6.3 关键启动参数

```bash
python -m sglang.launch_server \
  --model-path <target-model> \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path <eagle-head> \
  --speculative-num-steps 5 \          # 草稿自回归步数（树深）
  --speculative-eagle-topk 4 \         # 每步保留的分支数（树宽）
  --speculative-num-draft-tokens 8     # 整棵树送验的 token 总数
```

其他常用参数：`--speculative-accept-threshold-single/-acc`（接受阈值调节，牺牲无损性换接受率）、`--speculative-attention-mode`、`--speculative-token-map`（草稿词表裁剪）、n-gram 系列参数（`--speculative-ngram-*`）。`--speculative-adaptive` 可根据当前批次大小与运行时接受率自适应调整草稿步数，原理见 §8。

### 6.4 观测指标

- `accept_rate`（$\alpha$）：草稿质量指标，不含 bonus。低于 ~0.5 时投机解码可能得不偿失。
- `accept_length`（$\tau$）：每次 verify 平均前进 token 数（= `completion_tokens / spec_verify_ct`），含 bonus，直接对应加速比上界（换算见 §7.1）。
- 请求级 `meta_info` 与 Prometheus 指标中均有导出。

## 7. 性能权衡

投机解码不是免费的加速。要谈权衡，先把「加速比」这个量定义清楚——它是所有收益与成本拉扯之后的落点。

### 7.1 加速比（Speedup）的定义

**定义**：加速比是**同一段输出、同一套硬件**上，开启投机与不开投机（baseline 普通自回归）的墙钟时间之比：

$$
\text{Speedup} = \frac{T_{\text{baseline}}}{T_{\text{spec}}}
$$

要产出 $N$ 个 token，两条路径的串行结构完全不同：

- **baseline**：$N$ 次目标模型 decode forward，严格串行、一次一个 token，**没有可压缩的余地**；
- **spec**：约 $N/\tau$ 轮 draft-verify，每轮 $=\ k$ 次草稿 forward（串行，草稿模型小）$+\ 1$ 次目标模型 verify forward（一次并行验证整条链 / 整棵树）。

记三个单步耗时：$t_p$ = 目标模型生成 1 个 token 的 decode 步，$t_v$ = 一次 verify forward，$t_d$ = 草稿模型单步。则

$$
T_{\text{baseline}} = N\,t_p, \qquad
T_{\text{spec}} = \frac{N}{\tau}\left(t_v + k\,t_d\right)
$$

代入定义，$N$ 约掉：

$$
\text{Speedup} = \frac{\tau\, t_p}{t_v + k\, t_d}
= \frac{\tau}{\underbrace{t_v/t_p}_{\rho_v} \;+\; k\underbrace{t_d/t_p}_{\rho_d}}
$$

分子分母逐项读：

| 项 | 含义 | 量级直觉 |
| --- | --- | --- |
| $\tau$（分子） | 每轮真正前进的 token 数（含 bonus），即 §3 的 $\mathbb{E}[\tau]$ | 1~8，草稿越好越大 |
| $\rho_v = t_v/t_p$ | 「一次 verify 相当于几次普通 decode」 | memory-bound 时 $\approx 1$；大 batch 下 $>1$ |
| $\rho_d = t_d/t_p$ | 草稿单步相对目标模型单步的便宜程度 | EAGLE 头约 0.05~0.10 |

由此得到三条结论：

1. **$\tau$ 是加速比上界的主项**。若草稿零成本（$\rho_d=0$）且 verify 与 decode 同价（$\rho_v=1$），则 $\text{Speedup}=\tau$。这就是 §6.4 把 `accept_length` 当作首要观测指标、也是各家论文直接报「接受长度 6+」的原因。
2. **分母是「一轮的串行步数成本」**：1 次 verify $+$ $k$ 次草稿。草稿足够便宜时 $k\rho_d$ 远小于 $\rho_v$，加速比几乎只由 $\tau$ 决定。EAGLE 的草稿头只有一层（$\rho_d\approx0.08$），$k=5$ 时 $k\rho_d\approx0.4$，分母仅膨胀四成。
3. **实测加速比必然低于 $\tau$**。差额来自 $\rho_v>1$（verify 的那 $\tau$ 个位置在 batch 变大时变成真实算力开销，见 7.2 第 2 条）以及树构建、KV 搬运、采样等固定开销。工程上常把「实测加速比 / $\tau$」视作系统效率。

**数值例**：取 §4.5(c) 的 $\alpha=0.75$、$k=3$，得 $\mathbb{E}[\tau]\approx2.73$；再设 $\rho_v=1.1$、$\rho_d=0.08$，则

$$
\text{Speedup} \approx \frac{2.73}{1.1 + 3\times0.08} \approx 2.04\times
$$

即「理论上界 $2.73\times$，实际约 $2.0\times$」。

**两种口径不要混**：单请求**延迟加速比**（TPOT、端到端时延）直接套上式；多并发**吞吐加速比**则要把 $t_v,t_p$ 换成「每轮的系统总成本 / 每轮产出的 token 数」，batch 越大分母里的 $\rho_v$ 越大，加速比衰减得越快。同一份配置在两种口径下可以一个 $>1$、一个 $<1$。

最后，因为拒绝采样保证输出分布严格不变（§3），加速比是**纯墙钟比值、不掺任何质量折扣**——这是它与量化、蒸馏那类「以精度换速度」手段的本质区别（只有 `--speculative-accept-threshold-*` 这类故意放宽的阈值参数才打破无损性）。

### 7.2 收益与成本的拉锯

1. **草稿成本**：每轮多跑 $k$ 步草稿模型 forward；草稿模型太大或接受率太低时净收益为负（对应 §7.1 分母里的 $k\rho_d$）。
2. **验证成本随树增大**：`num_draft_tokens` 越大，verify 这次 forward 越接近 compute-bound；大 batch 下目标模型 decode 本身已不那么 memory-bound，$\rho_v$ 随之上涨，投机解码收益随 batch size 增大而衰减。
3. **显存**：草稿模型权重 + 草稿树的临时 KV 占用，会压缩可用 KV cache，间接降低最大并发。
4. **适用场景**：小 batch、低延迟场景（如交互式对话）收益最大；高吞吐大 batch 场景需实测。

### 7.3 经验法则

先用默认 EAGLE 参数压测，观察 `accept_length`（即 $\tau$，加速比上界）；$\tau > 2$ 时通常有明显收益，再在 (num_steps, topk, num_draft_tokens) 空间里调优。

## 8. 自适应步数：让草稿深度跟着负载走

§7.2 的权衡说明「最优草稿深度」并不是一个常数：它同时取决于当前有多少并发请求、以及当前这批请求的草稿好不好猜。固定 `--speculative-num-steps` 只能取一个折中值——小 batch 时太浅，白白浪费了空闲的算力；大 batch 时太深，验证开销反过来吃掉收益。自适应机制的做法是**在线上运行时不断重选步数**。

### 8.1 两个正交的决策信号

自适应由两个互不干扰的信号共同决定，理解它们的分工是理解整套机制的关键：

- **批次大小 → 选哪一档**。每一轮 decode 开始起草之前，先看当前 batch 有多大，据此路由到一个预置的「档位」。这是外层选择，随负载起伏而秒级切换。
- **接受长度 → 在档内上下移动**。每个档位内部维护一个候选步数的阶梯，并根据实际观测到的接受长度在阶梯上爬升或回落。这是内层调节，是对草稿质量的慢速跟踪。

之所以把批次大小放在外层，是因为它决定的是**问题的性质**而不只是参数取值：小 batch 时目标模型 decode 严重 memory-bound，算力大量闲置，深草稿几乎是白送的；batch 变大后 verify 逐渐 compute-bound，每多验一个草稿 token 都要付真实代价。这个转折是硬件特性决定的，不该等接受率统计慢慢发现。默认配置里 batch 大到一定程度后候选阶梯直接塌缩成「只允许 1 步」，本质上就是在承认「这个负载区间里投机解码只剩下最浅的一层还有正收益」。

路由本身要和 CUDA graph 对齐：先把实际 batch size 向上取整到最近的 CUDA graph 捕获尺寸，再用这个尺寸去挑档位。否则同一张 CUDA graph 会在相邻两轮里被路由到不同步数，切换代价反而超过收益。档位表是稀疏的（只在少数几个 batch size 上定义），落在中间的尺寸取「不超过它的最大档位」。

### 8.2 档内的阶梯调节

档位内部的规则可以概括成一句话：**探比看到的再深一步**。观测到的平均接受长度经指数滑动平均平滑后，目标步数取「四舍五入后加一」，再截断到该档允许的候选范围内。加一是刻意的乐观：如果当前平均能接受 3 个 token，说明 3 步深度已经被吃满，值得再往下探一步看看能不能接受更多——不主动探测就永远停在当前深度。

三个抑制抖动的设计：

1. **预热期**：刚启动的若干批不参与调节，避免用还没稳定的统计量做决策。
2. **降频重算**：每隔固定批数才重新计算一次目标步数，EMA 本身仍然每批更新。
3. **非对称迟滞**：升级和降级用不同的阈值。默认配置下降级阈值被额外压低，意味着**降级比升级更难触发**——接受长度短暂下滑不会立刻放弃深草稿，因为一次误降级会连带丢掉一整段本可以接受的 token；而误升级的代价只是多起草一步。同理，即便配置了动态上限，这个上限也只用来限制降级、不阻止升级，好让系统保有向上探索、等 EMA 追上来的空间。

### 8.3 为什么切换是「整体状态交换」而非改一个数字

改变草稿步数会改变草稿树的 token 总数，而这个数字被烘焙进了三个阶段的执行资源里：草稿自回归、目标模型验证、以及 prefill 之后的 draft-extend。每个阶段都各自持有 attention backend 的元数据缓冲和捕获好的 CUDA graph，形状都依赖草稿 token 数。

因此实现上并不在运行时修改参数，而是**在启动阶段就为每个候选步数各自建好一整套资源**，运行时切换退化成一次指针交换：原子地把当前生效的那一套换掉。代价是显存占用随候选步数的个数增长——候选阶梯配得越长，启动时要捕获的 CUDA graph 越多。这也解释了为什么默认阶梯很短（最多三档）。

### 8.4 适用边界

自适应目前只在链式草稿这一条最简单的路径上启用：要求算法是 EAGLE/EAGLE3、且草稿树宽为 1（topk = 1）。宽度大于 1 时树形结构本身也要跟着变，状态交换的复杂度陡增。此外与若干把 batch 切分或重排的特性互斥（DP attention、多层 EAGLE、双批次重叠、PD 复用），因为这些特性下「当前 batch size」不再是一个能唯一决定执行形状的量。

## 9. 工业界现状：开源旗舰模型的投机方案（2026-08 调研）

截至 2026 年 8 月，最新一批开源 SOTA 模型的投机解码方案对比：

| 模型 | 主力投机方案 | 关键数字 |
| --- | --- | --- |
| GLM-5.3（沿用 5.2 架构） | **原生 MTP 头**；训练时 **3 个 MTP 层共享参数**（显存与单层 MTP 持平，多步接受率更高） | 4 步草稿接受长度 2.76（vs DeepSeek-V3.2 的 2.55）；SGLang 低延迟配置（steps 5 / topk 1 / draft 6）接受长度可达 4~6 |
| Kimi-K3（2.8T MoE，KDA 混合注意力） | 预训练 MTP 层 **fine-tune 成 EAGLE-3 式草稿头**；旗舰方案 **DSpark**（块扩散骨干一次并行出 7 个草稿 token） | vLLM + DSpark：118 → 370 tok/s（3.14×）；chat 负载接受长度约 2.7 |
| DeepSeek-V4-Pro | 原生 MTP-1 为生产基线；新开源 **DSpark**（半自回归块草稿 + 置信度调度验证，随负载自动收缩验证预算） | 离线接受长度比 EAGLE-3 高 26~31%、比 DFlash 高 16~18%；线上比 MTP-1 快 60~85%（同吞吐） |
| Qwen3.8-Max（2.4T-A95B 开源权重） | **原生 MTP 头**，vLLM / SGLang / llama.cpp 均可直接启用，无需外挂草稿模型 | 社区实测约 2× 吞吐、1.88× 延迟改善；外挂 EAGLE-3 头在 Qwen3 系接受长度 1.74~2.2 |

五个趋势（与前文原理的对应关系）：

1. **原生 MTP 头成为标配**：四家全部在预训练时内置 MTP，用户零训练成本开箱即用（详见 [mtp.md](./mtp.md)、[mtp_vs_eagle1.md](./mtp_vs_eagle1.md) §5 的决策图）。
2. **MTP → EAGLE-3 蒸馏成为标准升级路径**：Kimi-K3 官方直接把 MTP 层微调成 EAGLE-3 式草稿头（SpecForge 在线蒸馏，目标模型由 SGLang 引擎在线提供 hidden states）——即 [mtp_vs_eagle1.md](./mtp_vs_eagle1.md) §7「MTP 头是最好的 EAGLE 初始化」的工业实践。
3. **训练时对齐推理成为共识**：GLM-5 训练时 3 个 MTP 层共享参数，治的正是「训练只学 1 步、推理自递归多步」的暴露偏差，与 EAGLE-3 的 training-time test（[eagle.md](./eagle.md) §2.4 ③）思路同源。
4. **前沿从「自回归草稿」转向「并行块草稿」**：DeepSeek 与 Kimi 的 DSpark 用块扩散/半自回归骨干**一次并行出整块草稿**，起草成本不随深度线性增长；置信度调度按负载动态收缩验证预算，直击 §7.2 第 2 条「大 batch 下验证开销吃掉收益」的痛点。原理详见 [dspark.md](./dspark.md) 与 [dflash.md](./dflash.md)。
5. **生产接受长度 ≪ 论文接受长度**：生产 MoE 巨模型的链式 MTP 接受长度普遍在 2.5~2.8（GLM-5 2.76、K3 约 2.7、DS-V3.2 2.55），远低于 EAGLE-3 论文的 6+——高并发下只敢用浅草稿（验证预算 4~6 token 且随负载收缩），与 §7 的性能权衡一致。

> 调研主要来源：[GLM-5 技术报告](https://arxiv.org/pdf/2602.15763)、[Kimi K3 Tech Blog](https://www.kimi.ai/blog/kimi-k3)、[SGLang Kimi-K3 Day-0 支持](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support/)、[DSpark 论文](https://arxiv.org/html/2607.05147v1)、[Qwen3.8-Max 开源权重报道](https://llm-stats.com/blog/research/qwen3-8-max-open-weights)。数字随版本与硬件浮动，以各家最新文档为准。

## 参考文献

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023
- Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, ICML 2024
- Li et al., *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*, 2025
