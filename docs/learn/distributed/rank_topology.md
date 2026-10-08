# 并行组合的 Rank 拓扑图解（world_size=16）

> 本文用统一的 16 卡（`g0` ~ `g15`）示例，图解 SGLang 中不同并行方式组合后的 **rank 拓扑**，
> 并**分别画出 Attention 层与 MoE 层**的分组，因为二者复用同一批 rank，但切分维度不同。
>
> 所有分组严格对应 `python/sglang/srt/distributed/parallel_state.py::initialize_model_parallel`
> 的建组逻辑。阅读前建议先看 `TP.md`、`EP.md`、`DP_attention.md`、`CP.md`、`PP.md`。

## 目录

1. [统一约定与分组规律](#1-统一约定与分组规律)
2. [组合 A：纯 TP=16](#2-组合-a纯-tp16)
3. [组合 B：TP=8 + PP=2](#3-组合-btp8--pp2)
4. [组合 C：DP Attention + 纯 EP](#4-组合-cdp-attention--纯-ep)
5. [组合 D：Attention CP + MoE(DP+EP)](#5-组合-dattention-cp--moedpep)
6. [组合 E：EP + expert TP](#6-组合-eep--expert-tp)
7. [组合 F：MoE(DP+EP+TP) 三维组合](#7-组合-fmoedpeptp-三维组合)
8. [组合速查表](#8-组合速查表)

---

## 1. 统一约定与分组规律

- 一个模型实例的 rank 域大小满足：

$$
world\_size = tp\_size \times pp\_size .
$$

- Attention 侧与 MoE 侧各自复用同一批 rank，分别满足：

$$
attn\_tp\_size = \frac{tp\_size}{attn\_cp\_size \times attn\_dp\_size},
\qquad
moe\_tp\_size = \frac{tp\_size}{moe\_ep\_size \times moe\_dp\_size}.
$$

各类组在一个 TP 组（连续 `tp_size` 个 rank）内的构建规律：

| 组 | 构建方式 | 直观理解 |
| --- | --- | --- |
| TP | 连续 `tp_size` 个 rank | 一个模型实例（除 PP）的 rank 域 |
| Attention TP | 连续 `attn_tp_size` 个 rank | 组内切 head/张量维 |
| Attention CP | 步长 `attn_tp_size` 收集 | 组内切序列（token） |
| MoE TP | 连续 `moe_tp_size` 个 rank | 组内切单个专家 |
| MoE EP | 步长 `moe_tp_size` 收集 | 组内分不同专家 |
| MoE DP | 步长 `moe_tp_size × moe_ep_size` 收集 | 组内不同 token 副本 |
| PP | 步长 `world_size / pp_size` 跨全局 | 组内分不同流水线阶段 |

> 记忆口诀：**“连续切张量，跳步分数据/专家”**。切张量维（TP / Attention TP / MoE TP）用连续 rank；
> 分不同数据或专家（CP / EP / DP）用跨步 rank。

---

## 2. 组合 A：纯 TP=16

**参数**：`tp_size=16, pp_size=1`；attention/moe 全部退化为 `attn_tp=16` / `moe_tp=16`。

一个 16 路 TP 组，16 张卡协同算同一批 token，所有权重按张量维切成 16 份。

```text
world_size = 16，单个 TP 组

┌─────────────────────────────────────────────────────────────┐
│  TP group (16)                                                │
│  g0 g1 g2 g3 g4 g5 g6 g7 g8 g9 g10 g11 g12 g13 g14 g15        │
└─────────────────────────────────────────────────────────────┘

Attention 层：ATTN_TP = 整个 TP 组（attn_tp_size=16）
MoE 层：     MoE_TP  = 整个 TP 组（moe_tp_size=16），无 EP
```

- Attention 层：所有卡按 head 维切分，持有相同 token。
- MoE 层：每个专家的 FFN 按张量维切成 16 份，无跨卡 token 路由（无 EP）。

---

## 3. 组合 B：TP=8 + PP=2

**参数**：`tp_size=8, pp_size=2`。注意此时**一个模型实例的 rank 域是 8**（不是 16），
16 张卡被 PP 切成两个流水线阶段，每阶段一个 8 路 TP 组。

```text
world_size = 16 = tp_size(8) × pp_size(2)

TP 组（连续 8 个）:
┌───────────────────────────┐   ┌───────────────────────────┐
│ TP group 0 (stage 0)      │   │ TP group 1 (stage 1)      │
│ g0 g1 g2 g3 g4 g5 g6 g7   │   │ g8 g9 g10 ... g15         │
└───────────────────────────┘   └───────────────────────────┘

PP 组（步长 world_size/pp_size = 8）:
  PP group k = [gk, gk+8]，k = 0..7
  [g0,g8] [g1,g9] [g2,g10] [g3,g11] [g4,g12] [g5,g13] [g6,g14] [g7,g15]
```

拓扑网格（行=PP 阶段，列=TP rank）：

```text
            TP0  TP1  TP2  TP3  TP4  TP5  TP6  TP7
stage 0 →   g0   g1   g2   g3   g4   g5   g6   g7
stage 1 →   g8   g9   g10  g11  g12  g13  g14  g15
             │                                  │
             └────────── 每列是一个 PP 组 ───────┘
```

- Attention 层 / MoE 层：在**每个阶段的 8 路 TP 组内**再按各自并行度细分（本例 attn/moe 均为纯 TP=8）。
- 阶段之间只通过 PP 组传递 `hidden_states`，不参与层内 all-reduce / all-to-all。

> 关键：`tp` 与 `pp` 是**相乘**关系（占用不同 GPU）；而 `ep`/`dp` 等是在同一 TP 域内**重组**。

---

## 4. 组合 C：DP Attention + 纯 EP

**参数**：`tp_size=16, pp_size=1`；`attn_dp_size=8, attn_cp_size=1 → attn_tp_size=2`；
MoE 侧 `moe_ep_size=16, moe_dp_size=1 → moe_tp_size=1`（纯 EP）。

这是典型的 **DP Attention + 大 EP** 部署：Attention 用 8 份数据并行（每份 2 路 TP），
MoE 让 16 张卡组成一个 EP group，各持不同专家。

### Attention 层

```text
attn_dp_size=8，每个 DP 副本内 attn_tp_size=2
ATTN_TP 组（连续 2 个）：
  [g0,g1] [g2,g3] [g4,g5] [g6,g7] [g8,g9] [g10,g11] [g12,g13] [g14,g15]
    DP0     DP1     DP2     DP3     DP4      DP5        DP6       DP7

每个 DP 副本处理不同的 token 分片，副本内 2 卡按 head 切分。
```

### MoE 层

```text
moe_ep_size=16（moe_tp_size=1）：整个 world 就是一个 EP 组
┌─────────────────────────────────────────────────────────────┐
│  MoE_EP group (16)：每卡持有不同专家                          │
│  g0 g1 g2 ... g15                                             │
└─────────────────────────────────────────────────────────────┘
进入 MoE 前，各 DP 副本的 token 通过 all-to-all dispatch 路由到专家所在卡。
```

对比图（同一批 rank，两层视角不同）：

```text
              g0 g1 | g2 g3 | g4 g5 | g6 g7 | g8 g9 | g10 g11 | g12 g13 | g14 g15
Attention →  [DP0 ] [DP1 ] [DP2 ] [DP3 ] [DP4 ] [ DP5  ] [ DP6  ] [ DP7  ]  (每对内 attn_tp=2)
MoE       →  └─────────── 一个 EP 组，16 卡各持不同专家（all-to-all） ───────────┘
```

---

## 5. 组合 D：Attention CP + MoE(DP+EP)

**参数**：`tp_size=16, pp_size=1`；`attn_cp_size=2, attn_dp_size=1 → attn_tp_size=8`；
MoE 侧 `moe_dp_size=2, moe_ep_size=8 → moe_tp_size=1`。

长序列场景：Attention 用 2 路上下文并行（CP）切序列，每份 8 路 TP；
MoE 用 2 份数据并行、每份 8 路 EP。

### Attention 层

```text
attn_tp_size=8（连续 8 个）：
  ATTN_TP: [g0..g7]  [g8..g15]

attn_cp_size=2（步长 attn_tp_size=8 收集）：
  ATTN_CP: [g0,g8] [g1,g9] [g2,g10] [g3,g11] [g4,g12] [g5,g13] [g6,g14] [g7,g15]
```

网格（行=CP 分片，列=Attention TP rank）：

```text
             aTP0 aTP1 aTP2 aTP3 aTP4 aTP5 aTP6 aTP7
CP shard 0 → g0   g1   g2   g3   g4   g5   g6   g7
CP shard 1 → g8   g9   g10  g11  g12  g13  g14  g15
              │                                   │
              └──────── 每列是一个 CP 组 ──────────┘
（同一序列被切成 2 段，分别放在 CP shard 0 / 1；每段内部 8 卡按 head 切分）
```

### MoE 层

```text
moe_ep_size=8, moe_dp_size=2, moe_tp_size=1

MoE_EP 组（moe_dp_idx 固定，步长 moe_tp_size=1 连续收集 8 个）：
  EP group 0: [g0..g7]      (moe_dp 副本 0)
  EP group 1: [g8..g15]     (moe_dp 副本 1)

MoE_DP 组（因 attn_cp_size(2) == moe_dp_size(2)，走 else 分支，步长 moe_ep*moe_tp=8）:
  [g0,g8] [g1,g9] [g2,g10] [g3,g11] [g4,g12] [g5,g13] [g6,g14] [g7,g15]
```

网格（行=MoE DP 副本，列=EP 内专家序号）：

```text
              EP0  EP1  EP2  EP3  EP4  EP5  EP6  EP7
DP replica0 → g0   g1   g2   g3   g4   g5   g6   g7
DP replica1 → g8   g9   g10  g11  g12  g13  g14  g15
               │                                   │
               └──────── 每列是一个 MoE DP 组 ──────┘
```

> 注意：本例 Attention 的 CP 网格与 MoE 的 DP 网格**形状一致、rank 对齐**，
> 这正是代码中 `attn_cp_size == moe_dp_size` 时让 MoE DP 复用 CP 通信路径的原因。

---

## 6. 组合 E：EP + expert TP

**参数**：`tp_size=16, pp_size=1`；attention 用 `attn_tp_size=16`（纯 TP）；
MoE 侧 `moe_ep_size=8, moe_dp_size=1 → moe_tp_size=2`。

MoE 先按专家分成 8 组（EP），每个专家再由 2 卡做张量并行（expert TP）。

### Attention 层

```text
ATTN_TP = 整个 world（16）：所有卡持相同 token，按 head 切分。
```

### MoE 层

```text
moe_ep_size=8, moe_tp_size=2

MoE_TP 组（连续 moe_tp_size=2 个）：每个专家由 2 卡协同
  [g0,g1] [g2,g3] [g4,g5] [g6,g7] [g8,g9] [g10,g11] [g12,g13] [g14,g15]

MoE_EP 组（步长 moe_tp_size=2 收集 8 个）：跨越不同专家
  EP group 0: [g0,g2,g4,g6,g8,g10,g12,g14]   (各专家的 tp 分片 0)
  EP group 1: [g1,g3,g5,g7,g9,g11,g13,g15]   (各专家的 tp 分片 1)
```

网格（行=expert TP 分片，列=专家 EP 序号）：

```text
              EP0  EP1  EP2  EP3  EP4  EP5  EP6  EP7
mTP shard0 →  g0   g2   g4   g6   g8   g10  g12  g14
mTP shard1 →  g1   g3   g5   g7   g9   g11  g13  g15
               │    │
               │    └ 专家1 由 [g2,g3] 两卡张量并行
               └───── 专家0 由 [g0,g1] 两卡张量并行
（每列是一个 MoE_TP 组=一个专家；每行是一个 MoE_EP 组）
```

---

## 7. 组合 F：MoE(DP+EP+TP) 三维组合

**参数**：`tp_size=16, pp_size=1`；attention `attn_tp_size=16`；
MoE 侧 `moe_dp_size=2, moe_ep_size=4, moe_tp_size=2`（满足 $2\times4\times2=16$）。

MoE 侧同时启用数据并行、专家并行、专家张量并行，是最完整的三维组合。

### MoE 层三维坐标

每个 rank 对应一个坐标 `(moe_dp, moe_ep, moe_tp)`，rank 编号规律：

$$
rank = moe\_dp \times (moe\_ep \times moe\_tp) + moe\_ep \times moe\_tp\_idx_{\text{block}} + moe\_tp .
$$

更直观地，按 `moe_dp` 分成两大块，每块 8 卡：

```text
DP 副本 0：g0 ~ g7        DP 副本 1：g8 ~ g15

块内再按 (EP, TP) 排布（连续 2 个是 MoE_TP，步长 2 收集是 MoE_EP）：

           EP0     EP1     EP2     EP3
DP0  TP:  [g0,g1] [g2,g3] [g4,g5] [g6,g7]
DP1  TP:  [g8,g9] [g10,11][g12,13][g14,15]
```

三类 MoE 组：

```text
MoE_TP 组（连续 2）：
  [g0,g1] [g2,g3] [g4,g5] [g6,g7] [g8,g9] [g10,g11] [g12,g13] [g14,g15]

MoE_EP 组（moe_dp 固定，步长 moe_tp=2 收集 moe_ep=4 个）：
  DP0: [g0,g2,g4,g6]   [g1,g3,g5,g7]
  DP1: [g8,g10,g12,g14] [g9,g11,g13,g15]

MoE_DP 组（步长 moe_ep×moe_tp=8）：
  [g0,g8] [g1,g9] [g2,g10] [g3,g11] [g4,g12] [g5,g13] [g6,g14] [g7,g15]
```

立体示意（两层 DP，各含 4 专家 × 2 TP 分片）：

```text
        DP replica 0                     DP replica 1
   EP0  EP1  EP2  EP3               EP0  EP1  EP2  EP3
TP0 g0   g2   g4   g6            TP0 g8   g10  g12  g14
TP1 g1   g3   g5   g7            TP1 g9   g11  g13  g15
     └ 每列=1个专家(MoE_TP)          （两副本处理不同 token）
     └ 每行=1个 EP 组
  纵向跨两块(g_i, g_{i+8})=1个 MoE_DP 组
```

---

## 8. 组合速查表

| 组合 | tp | pp | attn_cp | attn_dp | attn_tp | moe_dp | moe_ep | moe_tp | 特点 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A 纯 TP | 16 | 1 | 1 | 1 | 16 | 1 | 1 | 16 | 全部张量并行 |
| B TP+PP | 8 | 2 | 1 | 1 | 8 | 1 | 1 | 8 | tp×pp 相乘占卡 |
| C DP-Attn+纯EP | 16 | 1 | 1 | 8 | 2 | 1 | 16 | 1 | DP 注意力 + 大 EP |
| D CP+MoE(DP+EP) | 16 | 1 | 2 | 1 | 8 | 2 | 8 | 1 | 长序列 CP，CP↔MoE-DP 对齐 |
| E EP+expert TP | 16 | 1 | 1 | 1 | 16 | 1 | 8 | 2 | 每专家再 2 路 TP |
| F MoE 三维 | 16 | 1 | 1 | 1 | 16 | 2 | 4 | 2 | DP×EP×TP 全组合 |

### 通用结论

1. **`tp × pp = world_size`**：PP 与 TP 相乘占用不同 GPU；其余并行度在同一 TP 域内**重组**，不额外占卡。
2. **Attention 与 MoE 复用同一批 rank**，但切分维度不同，因此需要分别画拓扑。
3. **连续 rank 切张量，跳步 rank 分数据/专家**：
   - 连续：TP、Attention TP、MoE TP；
   - 跳步：Attention CP、MoE EP、MoE DP、PP。
4. 当 `attn_cp_size == moe_dp_size` 时，MoE DP 组直接复用 Attention CP 组，减少通信组重复创建。
5. 校验约束：$attn\_cp\times attn\_dp\times attn\_tp = tp$，$moe\_dp\times moe\_ep\times moe\_tp = tp$，
   且 physical expert 总数需能被 `moe_ep_size` 整除。

---

## 参考

- 代码：`python/sglang/srt/distributed/parallel_state.py::initialize_model_parallel`
- 同目录：`TP.md`、`EP.md`、`DP_attention.md`、`CP.md`、`PP.md`、`DP.md`

