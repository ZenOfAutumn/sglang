# 并行方式总览：背景、问题、优缺点与适用范围

> 本文横向对比大模型训练 / 推理中六种主流并行方式——数据并行（Data Parallelism, DP）、
> 张量并行（Tensor Parallelism, TP）、流水线并行（Pipeline Parallelism, PP）、
> 序列并行（Sequence Parallelism, SP）、上下文并行（Context Parallelism, CP）、
> 专家并行（Expert Parallelism, EP）——分别说明它们
> **出现的历史背景**、**要解决的核心问题**、**优缺点**、**适用范围**与**所受限制**，
> 并给出选型建议与组合方式。每种并行的完整数学原理与 SGLang 实现，见同目录下的专文
> （`DP.md` / `TP.md` / `PP.md` / `SP.md` / `CP.md` / `EP.md`）。

## 目录

1. [一张图看懂：并行是在切什么](#1-一张图看懂并行是在切什么)
2. [为什么会出现这么多种并行](#2-为什么会出现这么多种并行)
3. [数据并行（Data Parallelism, DP）](#3-数据并行data-parallelism-dp)
4. [张量并行（Tensor Parallelism, TP）](#4-张量并行tensor-parallelism-tp)
5. [流水线并行（Pipeline Parallelism, PP）](#5-流水线并行pipeline-parallelism-pp)
6. [序列并行（Sequence Parallelism, SP）](#6-序列并行sequence-parallelism-sp)
7. [上下文并行（Context Parallelism, CP）](#7-上下文并行context-parallelism-cp)
8. [专家并行（Expert Parallelism, EP）](#8-专家并行expert-parallelism-ep)
9. [横向对比速查表](#9-横向对比速查表)
10. [并行如何组合：3D / 4D 并行](#10-并行如何组合3d--4d-并行)
11. [选型决策路径](#11-选型决策路径)

---

## 1. 一张图看懂：并行是在切什么

所有并行方式，本质都是在回答同一个问题：**一个庞大的「模型 × 数据」计算，沿哪个维度切开分给多张卡？**

一次 Transformer 前向的计算，可以看成一个多维张量在流动，主要维度有：

- **权重维度**：每层的大矩阵 $A \in \mathbb{R}^{d_\text{in} \times d_\text{out}}$；
- **层维度（深度）**：模型有 $L$ 层，串行叠起来；
- **批维度**：一批里有 $B$ 条请求 / 样本；
- **序列维度**：每条请求有 $s$ 个 token；
- **专家维度**（仅 MoE）：一层里有 $E$ 个 expert。

不同并行方式，就是沿不同维度下刀：

| 并行方式 | 切的维度 | 一句话概括 |
|---|---|---|
| **Data Parallelism（DP）** | 批维度 | 复制整个模型，喂不同的数据 |
| **Tensor Parallelism（TP）** | 权重维度（层内张量） | 把一层的大矩阵拆成几块并行算 |
| **Pipeline Parallelism（PP）** | 层维度（深度） | 不同层放到不同卡，像流水线串起来 |
| **Sequence Parallelism（SP）** | 序列维度（针对非矩阵乘算子） | 把 LayerNorm / Dropout 等按序列切，配合 TP 省激活显存 |
| **Context Parallelism（CP）** | 序列维度（针对 attention） | 把超长序列按 token 切，解决长上下文 |
| **Expert Parallelism（EP）** | 专家维度（仅 MoE） | 不同 expert 放到不同卡 |

> 关键直觉：这些维度**彼此正交**，所以并行方式可以**任意组合**——
> 一个真实的大规模部署往往是 `DP × TP × PP × EP` 的乘积（见 [§10](#10-并行如何组合3d--4d-并行)）。

---

## 2. 为什么会出现这么多种并行

并行方式不是一次性设计出来的，而是随着模型规模增长，**一个瓶颈解决后暴露出下一个瓶颈**，
逐步演化出来的。理解这条演化线，就能理解每种并行「为什么会出现」：

```
模型还小            → 单卡够用，不需要并行
吞吐不够            → DP：复制模型，多喂数据           （最早，源自训练）
单层装不下 / 算不动  → TP：把层内矩阵切开              （Megatron-LM, 2019）
整个模型装不下       → PP：把层切到不同卡              （GPipe/PipeDream, 2018-19）
激活显存爆了         → SP：把 TP 覆盖不到的算子按序列切  （Megatron-SP, 2022）
上下文越来越长       → CP：把序列本身切开              （Ring Attention 等, 2023-24）
MoE 专家总量爆炸     → EP：把专家分散到不同卡          （GShard/Switch, 2020-21）
```

一条主线：**每一种新并行，都是在「上一种并行解决不了的那个资源」上重新下刀。**

- DP 解决**吞吐**，但不解决单卡装不下；
- TP 解决**单层**装不下 / 算不动，但受限于单机高速互联，且无法无限切；
- PP 解决**整个模型**跨机装不下，但引入了流水线气泡；
- SP 解决 TP 之外**激活显存**的浪费；
- CP 解决**序列长度**带来的 attention 显存 / 算力爆炸；
- EP 解决 **MoE 专家总量**带来的权重爆炸。

---

## 3. 数据并行（Data Parallelism, DP）

> 详见 `DP.md`。

**出现背景**：最早的并行方式，源自深度学习训练。单卡训练一个 epoch 太慢，于是复制模型到多卡，
每卡喂不同的 mini-batch，梯度做 all-reduce 同步。推理时代则演化为「模型副本 + 请求分发」。

**要解决的问题**：**吞吐不够**。单个模型副本的 QPS 有上限，请求一多就排队。

**核心做法**：复制**整个模型**成多份，把**不同的请求 / 数据**分给不同副本并行处理。

**优点**：
- 实现简单、扩展性好，吞吐近似线性提升；
- 推理时副本间**几乎零通信**（训练时才需要梯度 all-reduce），**可跨机**横向扩展（scale-out）；
- 与所有其他并行正交，最容易叠加。

**缺点 / 限制**：
- **不解决单卡装不下**——每张卡仍需放下一个完整副本（或副本的一份分片）；
- 训练场景下梯度同步的 all-reduce 通信量随模型增大而增大；
- 副本越多，KV cache、显存冗余越多（每个副本各存一份权重）。

**适用范围**：模型（或其 TP/PP 组合后的单副本）已能在若干卡上跑起来，需要靠增加硬件**线性提升吞吐**的场景。线上高 QPS 服务几乎必用。

---

## 4. 张量并行（Tensor Parallelism, TP）

> 详见 `TP.md`。

**出现背景**：2019 年 Megatron-LM 提出。当模型单层（如超大 FFN、超多 head 的 attention）
已经大到单卡装不下、算不动时，DP/PP 都无能为力——它们最小粒度是「整个副本」或「整层」。

**要解决的问题**：**单层**权重装不下、单步矩阵乘算不动。

**核心做法**：利用分块矩阵乘恒等式，把**同一层内**的大矩阵沿列 / 行切成 $p$ 块，
分给 $p$ 张卡同时算，再用一次集合通信（all-reduce / all-gather）拼回结果。

**优点**：
- 同时摊薄**单层的权重和计算**，降低单步延迟到 $\approx 1/p$，突破单卡显存；
- 是让超大稠密模型「跑得起来」的关键手段；
- 与 DP / PP 正交，可组合。

**缺点 / 限制**：
- **每层都要跨卡集合通信**，对互联带宽极其敏感；
- 因此通常**只能在单机多卡、NVLink 高带宽互联内使用**，跨机 TP 会被网络拖垮；
- TP 路数 $p$ 受限于单机卡数与 head 数 / 隐藏维的整除关系，**无法无限切**。

**适用范围**：单机内、需要装下 / 加速超大单层的场景。是大模型推理的基础手段，TP=8 是常见配置。

---

## 5. 流水线并行（Pipeline Parallelism, PP）

> 详见 `PP.md`。

**出现背景**：2018-2019 年 GPipe / PipeDream 提出。当整个模型（几十上百层）连 TP 都装不下
一台机器时，需要跨机切分——而 TP 跨机通信太贵，于是按**层的深度**切。

**要解决的问题**：**整个模型跨机装不下**。

**核心做法**：把不同的 layer 分到不同卡 / 不同机器（stage），数据像流水线一样在 stage 间逐段前向。
配合 micro-batch 让各 stage 尽量同时忙碌。

**优点**：
- **stage 间只需点对点（P2P）传一份激活**，通信量小，**天然适合跨机**扩展深度；
- 能把百层大模型摊到多机，突破单机显存总量上限。

**缺点 / 限制**：
- **流水线气泡（bubble）**：填充 / 排空阶段部分 stage 空闲，气泡率 $\approx \frac{p-1}{m+p-1}$（$p$ 为 stage 数，$m$ 为 micro-batch 数），需要足够多 micro-batch 才能摊薄；
- 负载需在 stage 间均衡切分，否则最慢的 stage 成为瓶颈；
- 调度复杂（1F1B、interleaved 等），实现难度高。

**适用范围**：模型层数多、单机（哪怕 TP=8）也装不下，需要跨机扩展**深度**的超大模型。

---

## 6. 序列并行（Sequence Parallelism, SP）

> 详见 `SP.md`。

**出现背景**：2022 年 Megatron-SP 提出。TP 只切了矩阵乘（Linear / Attention），
但 LayerNorm、Dropout、residual 这些**非矩阵乘算子**在每个 TP rank 上是**完整复制**的，
它们的**激活显存**没有被 TP 摊薄，成了新的瓶颈。

**要解决的问题**：TP 覆盖不到的算子的**激活显存冗余**。

**核心做法**：在 TP 的基础上，把这些逐元素 / 归一化算子沿**序列维**切分，
每个 rank 只处理序列的一段。TP 区与 SP 区之间用 all-gather / reduce-scatter 衔接
（合并后通信量与纯 TP 的 all-reduce 相当，不额外增加）。

**优点**：
- 显著降低**激活显存**（尤其长序列 / 大 batch 训练），几乎不增加通信量；
- 与 TP 无缝配合，是 TP 的自然补充。

**缺点 / 限制**：
- 依附于 TP 存在，**不单独使用**；
- 主要收益在**训练**（激活显存是训练痛点），推理侧收益有限；
- 增加了 all-gather / reduce-scatter 的实现与调度复杂度。

**适用范围**：大规模**训练**、启用 TP 且激活显存吃紧的场景。常与 TP 捆绑出现。

---

## 7. 上下文并行（Context Parallelism, CP）

> 详见 `CP.md`。

**出现背景**：2023-2024 年，随着长上下文（32K、128K、1M token）需求兴起。
Attention 的显存与算力随序列长度 $s$ 呈 $O(s^2)$（或 KV cache 呈 $O(s)$）增长，
单卡放不下超长序列的激活 / KV，SP 那种「切非矩阵乘算子」的做法不够。

**要解决的问题**：**超长序列**下 attention 的显存 / 算力爆炸。

**核心做法**：把**序列本身**按 token 切成若干段，分到不同卡。attention 需要跨段交互，
通过 Ring Attention（环形传递 KV 分块）或 all-gather KV 等方式完成全局 attention。

**优点**：
- 直接解决**单卡放不下超长序列**的问题，把序列维摊到多卡；
- 与 TP / DP 正交，可组合出超长上下文能力。

**缺点 / 限制**：
- attention 跨段需要**传递 KV / 分块**，通信随序列段数增加；
- 实现复杂（环形调度、因果 mask 的负载均衡等）；
- 只对 **attention 主导**、序列极长的场景收益明显；短序列反而得不偿失。

**适用范围**：长 / 超长上下文（长文档、长对话、代码库级输入）的训练与推理。

---

## 8. 专家并行（Expert Parallelism, EP）

> 详见 `EP.md`。

**出现背景**：2020-2021 年 GShard / Switch Transformer 提出。MoE 模型用**大量 expert**
替代单个稠密 FFN（一层可能几百上千个 expert），专家总量极大——单卡 HBM 根本放不下所有 expert。

**要解决的问题**：**MoE 专家总量**带来的权重爆炸。

**核心做法**：把**不同的 expert 分到不同 rank**，每 rank 只持有一部分完整 expert。
token 经 router 选出 top-k 后，通过 **all-to-all** 路由（dispatch）到其 expert 所在 rank，
算完再 all-to-all 收回（combine）。

**优点**：
- 利用 expert 间**天然独立**的性质，让不同 rank 各算各的 expert，专家权重降至 $\approx 1/p$；
- 相比对每个 expert 做 TP，EP 的每个 expert 计算无需 rank 间协作，粒度更自然；
- 是超大 MoE（DeepSeek、Mixtral 等）落地的关键。

**缺点 / 限制**：
- 依赖 **all-to-all** 通信，对网络带宽 / 拓扑敏感；
- **负载不均**：热门 expert 会让个别 rank 过载，需要 EPLB（Expert Parallel Load Balancing）动态均衡；
- **仅适用于 MoE 层**，稠密模型用不上。

**适用范围**：MoE 架构模型的训练与推理。常与 DP（attention 走 DP）、TP 组合。

---

## 9. 横向对比速查表

| 维度 | Data Parallelism（DP） | Tensor Parallelism（TP） | Pipeline Parallelism（PP） | Sequence Parallelism（SP） | Context Parallelism（CP） | Expert Parallelism（EP） |
|---|---|---|---|---|---|---|
| **切什么** | 批（复制模型） | 层内张量（权重） | 层（深度） | 序列（非矩阵乘算子） | 序列（attention） | 专家（MoE） |
| **主要解决** | 吞吐不足 | 单层装不下/算不动 | 整模型跨机装不下 | 激活显存冗余 | 超长序列爆炸 | MoE 专家权重爆炸 |
| **通信原语** | all-reduce（训练梯度）/ 无（推理） | all-reduce / all-gather | P2P（点对点） | all-gather / reduce-scatter | Ring / all-gather KV | all-to-all |
| **通信频率** | 低（推理近零） | 高（每层） | 低（stage 边界） | 中（与 TP 合并） | 中-高（随序列段） | 高（每 MoE 层两次） |
| **可否跨机** | 可，友好 | 不建议（带宽敏感） | 可，友好 | 随 TP（不建议跨机） | 可 | 可（需高速网络） |
| **降低单卡显存** | 否 | 是（权重+激活） | 是（权重，按层） | 是（激活） | 是（激活/KV） | 是（专家权重） |
| **降低单步延迟** | 否 | 是（$\approx 1/p$） | 否（反而有气泡） | 否 | 否 | 部分 |
| **主要收益场景** | 推理高 QPS | 单机大模型 | 跨机超大模型 | 大规模训练 | 长上下文 | MoE 模型 |
| **典型限制** | 每卡放完整副本 | 单机卡数/整除关系 | 流水线气泡 | 依附 TP、偏训练 | 实现复杂、需长序列 | 负载不均、仅 MoE |
| **正交/依附** | 正交，最易叠加 | 正交 | 正交 | 依附 TP | 正交 | 正交（仅 MoE 层） |

---

## 10. 并行如何组合：3D / 4D 并行

由于各维度正交，大规模部署几乎都是**多种并行的乘积**：

$$
\text{world\_size} = \text{DP} \times \text{TP} \times \text{PP} \quad (\times\ \text{EP}\ /\ \text{CP})
$$

经验组合原则（本质是「让通信量大的并行留在带宽高的层级」）：

1. **TP 放最内层（单机内 NVLink）**：TP 每层都通信，必须吃满机内高带宽；
2. **PP 放机间**：PP 只在 stage 边界传激活，通信少，适合跨机；
3. **DP 放最外层**：推理近零通信，最适合横向 scale-out；
4. **EP 与 attention 的 DP/TP 配合**：MoE 层走 EP，attention 层常走 DP（即 DP Attention，见 `DP_attention.md`）；
5. **CP / SP 按需叠加**：长上下文加 CP，训练省激活加 SP。

一个典型的 DeepSeek 大规模推理布局可能是：
`DP=4`（4 个副本横向扩吞吐）× 每副本内 `TP=8`（机内摊薄稠密层）+ MoE 层走 `EP`（专家分散到全部卡），
attention 层走 DP Attention。

> 组合的黄金法则：**通信越频繁的并行，越要放在带宽越高、延迟越低的互联层级上。**

---

## 11. 选型决策路径

按下面的问题逐步判断，即可快速定位需要哪些并行：

```
① 单卡能装下整个模型吗？
   能  → 只需 DP（复制多份扩吞吐）即可
   不能 → 往下

② 单机（多卡 NVLink）能装下吗？
   能  → 用 TP 把模型摊到机内多卡；再叠 DP 扩吞吐
   不能 → 往下

③ 模型层数多、跨机才装得下？
   是  → 加 PP 跨机切深度（机内仍用 TP）

④ 是 MoE 模型吗？
   是  → MoE 层加 EP（专家分散），attention 常配 DP Attention

⑤ 上下文非常长（几十 K 以上）吗？
   是  → 加 CP，把序列切到多卡

⑥ 在做大规模训练、激活显存吃紧吗？
   是  → 在 TP 基础上加 SP，摊薄激活显存
```

一句话总结：

> **DP 扩吞吐，TP 摊单层，PP 跨机切深度，EP 拆 MoE 专家，CP 撑长上下文，SP 省训练激活。**
> 它们互相正交，按「资源瓶颈」逐层叠加，按「通信带宽层级」合理排布，即得到一个高效的大规模并行方案。

---

## 参考与延伸

- 各并行的完整数学原理与 SGLang 实现：同目录 `DP.md` / `TP.md` / `PP.md` / `SP.md` / `CP.md` / `EP.md`；
- MLA 模型的 DP Attention：`DP_attention.md`；
- rank 布局与拓扑：`rank_topology.md`；
- 通信代价模型（何时该换算法 / 换互联层级）：`cost_model.md`；
- 集合通信原语：`collective_communication.md`。

