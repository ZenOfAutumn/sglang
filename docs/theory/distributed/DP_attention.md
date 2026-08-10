# 数据并行注意力（DP Attention）原理详解

> 本文介绍 SGLang 针对 MLA 类模型的 **DP Attention**（数据并行注意力）：它要解决什么问题、
> 与普通 TP 的区别、rank 布局如何计算、attention 与 MoE/FFN 两段之间如何用
> gather / scatter 衔接，并给出数值直觉，最后对应到 `layers/dp_attention.py` 的真实实现。
>
> 前置阅读：`TP.md`（张量并行）。DP Attention 是在 TP 基础上对 attention 子层做的并行方式改造。

## 目录

1. [为什么需要 DP Attention](#1-为什么需要-dp-attention)
2. [核心思想：attention 走 DP，MoE 走 TP/EP](#2-核心思想attention-走-dpmoe-走-tpep)
3. [rank 布局与关键公式](#3-rank-布局与关键公式)
   - 3.1 [第三个维度：attention CP 是什么](#31-第三个维度attention-cp-是什么)
   - 3.2 [rank 布局公式的推导](#32-rank-布局公式的推导)
   - 3.3 [三个维度对应的进程组](#33-三个维度对应的进程组)
   - 3.4 [完整示例](#34-完整示例tp_size8-attn_dp_size2-attn_cp_size2)
   - 3.5 [典型配置速查](#35-典型配置速查)
   - 3.6 [CP 与 MoE 的衔接](#36-cp-与-moe-的衔接)
4. [一次前向的数据流：gather → MoE → scatter](#4-一次前向的数据流gather--moe--scatter)
5. [两种 padding 模式：MAX_LEN 与 SUM_LEN](#5-两种-padding-模式max_len-与-sum_len)
6. [与普通 TP / 副本级 DP 的区别](#6-与普通-tp--副本级-dp-的区别)
7. [SGLang 中的实现](#7-sglang-中的实现)
8. [局限与权衡](#8-局限与权衡)

---

## 1. 为什么需要 DP Attention

普通 TP 对 attention 按 **head** 切分（见 `TP.md` §6）。这在标准 MHA 上很好用，但对 **MLA（Multi-head Latent Attention，如 DeepSeek / LongCat）** 有两个痛点：

1. **KV Cache 无法有效切分而被迫复制**。MLA 把 KV 压缩成一个低秩的 latent 向量（`kv_lora_rank`，如 512），它**不是按 head 组织**的。TP 按 query head 切分 attention，但每张卡做 attention 都需要**完整的 KV latent**，所以 latent 只能在每张卡上各存一份——KV Cache 在 TP 组内 $p$ 份冗余，严重浪费 HBM。

   **数值例子（DeepSeek-V3，TP=8）**：MLA 每 token 每层缓存的 latent 维度为 `kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576`，模型共 61 层，bf16 下每 token 的 KV Cache ≈ $576 \times 61 \times 2\,\text{B} \approx 68.6\,\text{KB}$。假设某时刻整机需要缓存 8192 个 token：

   - **TP=8 复制方案**：latent 切不动，8 张卡各存一整份 ≈ $8192 \times 68.6\,\text{KB} \approx 549\,\text{MB}$；8 卡合计约 **4.3 GB**，其中 $7/8$（≈ 3.75 GB）是纯冗余。
   - **DP attention 方案**：8 个 rank 各处理 1024 个 token，每卡只存 $1024 \times 68.6\,\text{KB} \approx 68.6\,\text{MB}$；8 卡合计仍是 549 MB，**零冗余**——单卡 KV Cache 占用直接降到 $1/8$，等价于同样显存能容纳 8 倍的并发 token。
2. **attention 的 all-reduce 通信频繁且规模小**。TP 每层 attention 末尾要 all-reduce，MLA 下这部分通信收益低。

**DP Attention 的思路**：既然 MLA 的 KV Cache 切不动，那就干脆**不切 attention，改成数据并行**——把不同的请求（token）分给不同的 rank，每个 rank 只处理**自己那批 token 的完整 attention**，各自只存自己请求的 KV Cache。这样：

- KV Cache **不再跨卡复制**，显存利用率大幅提升；
- attention 内部**不需要（或只需很小的）TP 通信**。

代价是：MoE/FFN 仍然需要全局 TP/EP，所以要在 attention 与 MoE 之间插入 gather/scatter 通信（见 §4）。

---

## 2. 核心思想：attention 走 DP，MoE 走 TP/EP

一个 Transformer block 被拆成两段，用**不同的并行维度**：

| 子层 | 并行方式 | 每个 rank 处理什么 |
| --- | --- | --- |
| **Attention** | 数据并行（DP）+ 可选小 TP | 只处理**本 DP rank 的那批 token**，只存这批 token 的 KV Cache |
| **MoE / 共享 FFN** | 专家并行（EP）/ 张量并行（TP） | 处理**全局所有 token**（需要先把各 rank 的 token 汇聚起来） |

```
  各 rank 各自的 token
        │  （attention：各算各的，无跨 rank 通信）
        ▼
   ┌─────────────┐
   │  DP gather  │  把所有 rank 的 token 汇聚成 global batch
   └─────────────┘
        │
        ▼
   MoE / FFN（EP/TP，在 global batch 上算）
        │
        ▼
   ┌─────────────┐
   │  DP scatter │  把结果切回各 rank 各自的 token
   └─────────────┘
        │
        ▼
   各 rank 继续下一层 attention
```

关键点：**同一个 TP 进程组，在 attention 阶段被"看作"若干个 DP 组，在 MoE 阶段又被"看作"完整的 TP/EP 组**——并行维度是在层内动态切换的，而不是起两套进程。

### 2.1 数据布局：`ScatterMode`

要讲清楚 block 怎么执行，先要有描述「此刻数据在各 rank 上如何分布」的语言。SGLang 用 `ScatterMode`（`communicator.py:198`）刻画，其 docstring 给了最直观的定义：

```python
"""
Suppose we have TP=4, DP=2, enable-dp-attention, and the system handles seq a,b,c,d
Model input/output: [ab, ab, cd, cd] for four ranks respectively
SCATTERED:    [a, b, c, d]
TP_ATTN_FULL: [ab, ab, cd, cd], i.e. all ranks inside a TP attn group have full data of the group
FULL:         [abcd, abcd, abcd, abcd]
MOE_FULL: full within the MoE group (cp_per_moe CP chunks), used when moe_dp_size < attn_cp_size
"""
```

以 `tp_size=4, attn_dp_size=2`（故 `attn_tp_size=2`）、4 条序列 `a,b,c,d` 为例：

| 模式 | rank0 | rank1 | rank2 | rank3 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `SCATTERED` | `a` | `b` | `c` | `d` | 每卡一份，最细粒度 |
| `TP_ATTN_FULL` | `ab` | `ab` | `cd` | `cd` | **DP 组内**完整（attention 工作态） |
| `FULL` | `abcd` | `abcd` | `abcd` | `abcd` | 全局完整（MoE/dense 工作态） |

每个 rank 的显存里只有对应那份数据。DP attention 的本质就是：**attention 阶段停在 `TP_ATTN_FULL`，MoE 阶段升到 `FULL`，算完再降回来**。

每种模式对应的通信组大小（`communicator.py:811`）：

```python
ScatterMode.SCATTERED: 1,
ScatterMode.TP_ATTN_FULL: attn_tp_size,
...
ScatterMode.FULL: tp_size // attn_cp_size,
```

### 2.2 一个 block 的三段式执行

真正驱动模式切换的是 `LayerCommunicator`。每个 block 的 forward 被切成**三个通信阶段**，中间夹着两段计算（`deepseek_v2.py:2118`, `:2140`, `:2185`）：

```python
# ① attention 前：确保数据处于 attn_mode
hidden_states, residual = self.layer_communicator.prepare_attn(
    hidden_states, residual, forward_batch, ...
)

#    —— 计算 self_attn（各 DP 组独立，无跨组通信）——

# ② attention 后 / MLP 前：all-reduce + layernorm + 升到 mlp_mode（gather）
hidden_states, residual = self.layer_communicator.prepare_mlp(
    hidden_states, residual, forward_batch
)

#    —— 计算 MoE / MLP（在 global batch 上）——

# ③ MLP 后：降回 layer_output_mode（scatter）
hidden_states, residual = self.layer_communicator.postprocess_layer(
    hidden_states, residual, forward_batch
)
```

各阶段的目标模式在建层时就静态算好了（`LayerScatterModes.init_new`，`communicator.py:366`）：

```python
layer_input_mode  = 上一层的 layer_output_mode（第 0 层为 model_input_output()）
attn_mode         = ScatterMode.TP_ATTN_FULL      # ← 恒定，attention 永远在这个模式下算
mlp_mode          = 稀疏层→FULL / MOE_FULL / SCATTERED；dense 层→FULL 或 SCATTERED
middle_residual_mode = TP_ATTN_FULL（当 mlp_mode 为 FULL 类）
layer_output_mode = 末层→model_input_output()；否则随 mlp_mode 推导
```

注意 `attn_mode` 是**硬编码常量** `TP_ATTN_FULL`（`:371`）——这正是「attention 走 DP」的代码体现：无论外层怎么配，attention 一定在「DP 组内完整、组间独立」的布局下计算。

### 2.3 完整数据流（`tp_size=4, attn_dp_size=2, attn_tp_size=2`）

```
              ┌──────────── DP 组 0 ────────────┐ ┌──────────── DP 组 1 ────────────┐
              │   rank0            rank1        │ │   rank2            rank3        │
              │  (attn_tp0)      (attn_tp1)     │ │  (attn_tp0)      (attn_tp1)     │
              └─────────────────────────────────┘ └─────────────────────────────────┘

layer 输入        [ab]              [ab]              [cd]              [cd]        ← TP_ATTN_FULL
                    │                 │                 │                 │
  ┌─────────────────┴─────────────────┴─────────────────┴─────────────────┴──────┐
  │ ① prepare_attn：input_layernorm + _communicate_simple_fn（本例已是目标模式，无通信）│
  └─────────────────┬─────────────────┬─────────────────┬─────────────────┬──────┘
                    │                 │                 │                 │
              ╔═════▼═════════════════▼═════╗     ╔═════▼═════════════════▼═════╗
              ║   self_attn（DP 组 0）        ║     ║   self_attn（DP 组 1）        ║
              ║  · 只算 token a,b            ║     ║  · 只算 token c,d            ║
              ║  · KV Cache 只存 a,b ✅       ║     ║  · KV Cache 只存 c,d ✅       ║
              ║  · 组内 2 卡按 head 切 (TP)   ║     ║  · 组内 2 卡按 head 切 (TP)   ║
              ║  · o_proj 后组内 all-reduce  ║     ║  · o_proj 后组内 all-reduce  ║
              ║    （仅 2 卡，不跨组）✅       ║     ║    （仅 2 卡，不跨组）✅       ║
              ║  ↓ 盒内细节见 §2.4           ║     ║  ↓ 盒内细节见 §2.4           ║
              ╚═════╤═════════════════╤═════╝     ╚═════╤═════════════════╤═════╝
                    │                 │                 │                 │
  ┌─────────────────┴─────────────────┴─────────────────┴─────────────────┴──────┐
  │ ② prepare_mlp：post_attention_layernorm + DP gather（TP_ATTN_FULL → FULL）      │
  │    ← 唯一的跨 DP 组通信，4 卡全参与                                              │
  └─────────────────┬─────────────────┬─────────────────┬─────────────────┬──────┘
                    │                 │                 │                 │
                 [abcd]            [abcd]            [abcd]            [abcd]     ← FULL
                    │                 │                 │                 │
              ╔═════▼═════════════════▼═════════════════▼═════════════════▼═════╗
              ║          MoE / dense FFN（4 卡组成完整 TP/EP 组）                 ║
              ║          在全局 4 个 token 上计算，专家分布在 4 卡                  ║
              ╚═════╤═════════════════╤═════════════════╤═════════════════╤═════╝
                    │                 │                 │                 │
  ┌─────────────────┴─────────────────┴─────────────────┴─────────────────┴──────┐
  │ ③ postprocess_layer：DP scatter（FULL → TP_ATTN_FULL），可用 reduce_scatter 优化 │
  └─────────────────┬─────────────────┬─────────────────┬─────────────────┬──────┘
                    │                 │                 │                 │
layer 输出        [ab]              [ab]              [cd]              [cd]      ← TP_ATTN_FULL
                    │                 │                 │                 │
                    ▼                 ▼                 ▼                 ▼
                              进入下一层 block（重复 ①②③）
```

**逐阶段拆解**：

| 阶段 | 代码入口 | 做什么 | 通信范围 |
| --- | --- | --- | --- |
| ① `prepare_attn` | `communicator.py:520` | `input_layernorm` → 调整到 `attn_mode`(`TP_ATTN_FULL`) | 通常无（已是目标模式） |
| **attention 计算** | 模型层 | 各 DP 组算各自 token；组内按 head 做小 TP | **仅 DP 组内**（`attn_tp_size` 卡） |
| ② `prepare_mlp` | `communicator.py:688` | attention 输出 all-reduce + `post_attention_layernorm` + **DP gather** | **跨全部 rank**（`tp_size` 卡） |
| **MoE/MLP 计算** | 模型层 | 全局 batch 上算，专家按 EP 分布 | EP 的 all-to-all（另一套机制） |
| ③ `postprocess_layer` | `communicator.py:706` | **DP scatter** 切回本 rank 的 token | 跨全部 rank |

### 2.4 展开 attention 内部：MLA 的上/下投影怎么切

上图把 `self_attn` 当成黑盒。但对 MLA 模型，盒子里的**下投影与上投影切分方式完全不同**，这正是 DP attention 能省 KV Cache 的关键。

MLA 把原本一个 $h \cdot d_k$ 的大投影拆成「先降维、再升维」两步（见 `../transformer/MLA.md` §2）。SGLang 对这两步用了**不同的 Linear 类型**：

| 权重 | 层类型 | 切分 | 原因 |
| --- | --- | --- | --- |
| **下投影** `fused_qkv_a_proj_with_mqa` / `kv_a_proj_with_mqa` | `ReplicatedLinear`（`deepseek_v2.py:1523`, `:1550`） | ❌ **每卡完整复制** | 输出是 latent，**不按 head 组织**，切了就得通信 |
| **上投影** `q_b_proj` | `ColumnParallelLinear`（`:1531`） | ✅ 按 `attn_tp_size` | 输出按 head 组织，可切 |
| **上投影** `kv_b_proj` | `ColumnParallelLinear`（`:1622`） | ✅ 按 `attn_tp_size` | 同上 |
| 输出投影 `o_proj` | `RowParallelLinear`（`:1632`） | ✅ 按 `attn_tp_size` | 与列切配对，行切 |

注意下投影**没有传** `tp_rank`/`tp_size`，而上投影三者都显式传入 `attn_tp_rank` / `attn_tp_size`——切分粒度由 attention 的小 TP 组决定，而非全局 `tp_size`：

```python
# deepseek_v2.py:1497
attn_tp_rank = get_attention_tp_rank()
attn_tp_size = get_attention_tp_size()
...
assert num_heads % attn_tp_size == 0
self.num_local_heads = num_heads // attn_tp_size     # ← 每卡负责几个 head
```

**放大 DP 组 0（rank0 + rank1，`attn_tp_size=2`）内部**：

```
       rank0 (attn_tp_rank=0)                    rank1 (attn_tp_rank=1)
   ┌───────────────────────────────┐      ┌───────────────────────────────┐
   │  hidden [ab]  (D=5120)        │      │  hidden [ab]  (D=5120)        │  ← TP_ATTN_FULL：
   └───────────────┬───────────────┘      └───────────────┬───────────────┘    组内两卡数据相同
                   │                                      │
       ╔═══════════▼═══════════╗              ╔═══════════▼═══════════╗
       ║ 下投影 W^DKV / W^DQ    ║              ║ 下投影 W^DKV / W^DQ    ║
       ║ ReplicatedLinear      ║              ║ ReplicatedLinear      ║
       ║ 【完整副本，不切】🔁    ║              ║ 【完整副本，不切】🔁    ║
       ╚═══════════╤═══════════╝              ╚═══════════╤═══════════╝
                   │                                      │
          c^KV(512)+k_pe(64)                     c^KV(512)+k_pe(64)
          c^Q(1536)                              c^Q(1536)
                   │      ← 两卡算出完全相同的 latent（冗余计算，但换来无通信）
                   │                                      │
       ┌───────────▼───────────┐              ┌───────────▼───────────┐
       │ KV Cache: 只存 [ab]    │              │ KV Cache: 只存 [ab]    │  ← 组内复制
       │ 的 576 维 latent       │              │ 的 576 维 latent       │    组间不同 ✅
       └───────────┬───────────┘              └───────────┬───────────┘
                   │                                      │
       ╔═══════════▼═══════════╗              ╔═══════════▼═══════════╗
       ║ 上投影 W^UK/W^UV/W^UQ  ║              ║ 上投影 W^UK/W^UV/W^UQ  ║
       ║ ColumnParallelLinear  ║              ║ ColumnParallelLinear  ║
       ║ 【按 head 切】✂        ║              ║ 【按 head 切】✂        ║
       ║  head 0 ~ 63          ║              ║  head 64 ~ 127        ║
       ╚═══════════╤═══════════╝              ╚═══════════╤═══════════╝
                   │                                      │
            attention(head 0~63)                   attention(head 64~127)
                   │                                      │
       ╔═══════════▼═══════════╗              ╔═══════════▼═══════════╗
       ║ o_proj RowParallel    ║              ║ o_proj RowParallel    ║
       ║ 【按输入维切】✂         ║              ║ 【按输入维切】✂         ║
       ╚═══════════╤═══════════╝              ╚═══════════╤═══════════╝
                   │  部分和                                │  部分和
                   └──────────► all-reduce ◄───────────────┘
                          （仅组内 2 卡，不跨 DP 组）✅
                                     │
                                     ▼
                            attention 输出 [ab]
```

**三个要点**：

1. **下投影复制是「必须」而非「浪费」**。$c^{KV}$ 是要写进 KV Cache 的 latent，若按列切，每卡只有分片，写 cache 前得先 all-gather 拼回 512 维——而 DP attention 的**全部意义**就是让每 rank 独立持有自己 token 的完整 KV。况且 $W^{DKV}$ 形状仅 $5120 \times 576$，远小于上投影的 $512 \times 32768$，复制小矩阵比为它通信划算得多。

2. **组内冗余、组间独立**。DP 组 0 的两张卡算出**完全相同**的 latent 并各存一份 KV Cache（组内复制），但与 DP 组 1 的 `[cd]` 完全不同（组间独立）。所以每卡 KV Cache 是全局的 $1/\text{attn\_dp\_size}$，而不是 $1/\text{tp\_size}$。

3. **`attn_tp_size=1` 时上投影也退化为复制**。`dp_size == tp_size` 的典型配置下 `num_local_heads = num_heads`，`ColumnParallelLinear` 不再切分——**这才是 DP attention 下 attention 权重冗余的真正来源**：层的声明没变，变的是传进去的 `tp_size` 参数。

> decode 路径下 `kv_b_proj` 会被拆成 `w_kc` / `w_vc` 并「吸收」进 Q 侧与输出侧（矩阵吸收，见 `../transformer/MLA.md` §4）。此时 $W^{UK}$ 不再是独立 GEMM，但**分片方式不变**，仍按 `num_local_heads` 组织。

### 2.5 为什么这样切能省 KV Cache

对比同样 4 卡、同样 4 条序列的**纯 TP** 方案：

| | 纯 TP（tp=4） | DP attention（tp=4, dp=2） |
| --- | --- | --- |
| attention 每卡处理的 token | 全部 4 条 `abcd` | 只有 2 条（`ab` 或 `cd`） |
| **每卡 KV Cache** | `abcd` 全量（MLA latent 切不动，4 卡各存一份） | 只存 `ab` 或 `cd` → **减半** |
| attention all-reduce 范围 | 4 卡 | **2 卡**（组内） |
| MoE 前后额外通信 | 无 | gather + scatter |

推广到 `attn_dp_size = p`：每卡 KV Cache 降到 $1/p$，attention all-reduce 从 `tp_size` 卡缩到 `tp_size/p` 卡；代价是每层多两次跨全组的 gather/scatter。

### 2.6 三个关键设计点

1. **模式是静态预计算的**：`LayerScatterModes` 在建层时一次算好，`_post_init_communicate`（`:470`）据此绑定具体的通信函数（`_communicate_simple_fn` 等）。运行时不做分支判断，避免每步开销。

2. **相邻层会「串联」**：`layer_input_mode` 直接取自上一层的 `layer_output_mode`（`:381`）。若某层 `mlp_mode == SCATTERED`（如启用 DeepEP a2a 或 `moe_dense_fully_dp`），输出就停在 `SCATTERED`，下一层的 `prepare_attn` 再负责升回 `TP_ATTN_FULL`——**通信被摊到层边界上，能省则省**。

3. **可用 reduce_scatter 替代 all-reduce**：③ 阶段若满足条件（`should_use_reduce_scatter`，`deepseek_v2.py:2151`），直接用 `reduce_scatter` 一步完成「求和 + 切分」，省掉一次完整 all-reduce。此外 `should_fuse_mlp_allreduce_with_next_layer`（`:2145`）还能把本层 MLP 的 all-reduce 融进下一层，进一步减少 kernel 启动。

---

## 3. rank 布局与关键公式

DP Attention 复用同一个 TP group（大小 `tp_size`），在其内部划分出 DP 维度。核心公式（`dp_attention.py:245-259` `compute_dp_attention_world_info`）：

```python
attn_dp_size = dp_size if enable_dp_attention else 1
attn_tp_size = tp_size // attn_dp_size // attn_cp_size
attn_tp_rank = tp_rank % attn_tp_size
attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)   # 开启 DP attention 时
```

### 3.1 第三个维度：attention CP 是什么

前面两节只讲了 DP 和 TP，公式里却出现了 `attn_cp_size`。**CP（Context Parallelism，上下文/序列并行）是 attention 的第三个正交切分维度**——它切的是**同一条请求内部的序列（token 维）**。

三者切的东西完全不同，不要混淆：

| 维度 | 切什么 | 每卡的 KV Cache | 解决的问题 |
| --- | --- | --- | --- |
| **TP** | head（权重的列/行） | 部分 head × 完整序列 | 单卡放不下**权重** |
| **DP** | **不同请求**之间 | 本 rank 那批请求的完整 KV | KV Cache **跨卡冗余** |
| **CP** | **同一请求内部**的序列 | 该请求的**一段 token** 的 KV | 单条序列**太长**，一张卡放不下它的 KV |

举例：一条 1M token 的请求，即使已经用 DP 把它单独分给了某个 rank，这一条的 KV Cache 仍可能撑爆单卡。此时只能沿序列维再切 —— 这就是 CP。

**CP 的难点在于 attention 不是逐 token 独立的**：query $q_i$ 需要看到 $[0, i]$ 的全部 key/value。序列切开后每个 rank 只有局部 KV，因此要么在 rank 间轮转传 KV（Ring Attention），要么各自算局部结果后用 **log-sum-exp 校正**合并：

$$
\text{out} = \frac{\sum_{r} e^{m_r - m}\, l_r \cdot \text{out}_r}{\sum_{r} e^{m_r - m}\, l_r},
\qquad m = \max_r m_r
$$

其中 $m_r$、$l_r$ 分别是 rank $r$ 上局部 softmax 的最大值与分母和。这正是 `dp_attention.py` 里 `attn_cp_all_gather_into_tensor` / `attn_cp_reduce_scatter_tensor` 这组 CP 专用通信原语存在的原因。CP 的完整机制见 `CP.md`，本节只关心它**如何参与 rank 编号**。

> 未开启 CP 时 `attn_cp_size = 1`，下面所有公式中含 `attn_cp_size` 的因子都退化为 1，可以直接忽略。

### 3.2 rank 布局公式的推导

一个 TP group 内的 `tp_size` 张卡，要同时承载 DP、CP、TP 三个维度，必须满足**容量守恒**：

$$
\texttt{tp\_size} = \texttt{attn\_dp\_size} \times \texttt{attn\_cp\_size} \times \texttt{attn\_tp\_size}
$$

这就是代码里 `attn_tp_size = tp_size // attn_dp_size // attn_cp_size` 的由来——前两者由用户通过 `--dp-size` / `--attn-cp-size` 指定，TP 度是**被动算出来的余数**。这也解释了 §8 的约束：`tp_size` 必须能被 `attn_dp_size * attn_cp_size` 整除。

把三个维度理解为一个**三维坐标系**，每张卡是坐标 $(\text{dp}, \text{cp}, \text{tp})$ 上的一个点。要把三维坐标压成一维的 `tp_rank`，就是常见的**行主序（row-major）展开**，SGLang 选择的顺序是 `(dp, cp, tp)`，**tp 变化最快、dp 变化最慢**：

$$
\texttt{tp\_rank} = \underbrace{\big(\texttt{attn\_dp\_rank} \times \texttt{attn\_cp\_size} + \texttt{attn\_cp\_rank}\big)}_{\text{前两维压成一个 "组编号"}} \times \texttt{attn\_tp\_size} + \texttt{attn\_tp\_rank}
$$

这与多维数组 `arr[dp][cp][tp]` 在内存中的线性下标计算完全同构：**每个维度的"步长（stride）"等于它右边所有维度大小的乘积**。

| 维度 | 步长（stride） | 取值范围 |
| --- | --- | --- |
| `attn_dp_rank` | `attn_cp_size * attn_tp_size` | `[0, attn_dp_size)` |
| `attn_cp_rank` | `attn_tp_size` | `[0, attn_cp_size)` |
| `attn_tp_rank` | `1`（最快变化） | `[0, attn_tp_size)` |

**反解**（已知 `tp_rank` 求三个坐标）就是逐层做整除与取余，即代码中的写法：

```python
attn_tp_rank = tp_rank % attn_tp_size                      # 取最低位
attn_cp_rank = (tp_rank // attn_tp_size) % attn_cp_size    # 去掉最低位后取次低位
attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)    # 去掉低两位剩下的
```

注意 `compute_dp_attention_world_info` 只返回了 `attn_tp_rank` 和 `attn_dp_rank`，**没有返回 `attn_cp_rank`**——CP rank 由 `get_attention_cp_rank()` 直接向 CP 进程组查询（`_ATTN_CP.rank_in_group`），无需在这里重复推导。

**为什么把 tp 放在最快变化维？** 因为 attention TP 的通信最频繁（每层 `o_proj` 后都要 all-reduce）。让同一 TP 组的卡拿到**连续的 rank 编号**，它们就更可能落在同一台机器的同一 NVLink 域内，把最密集的通信留在机内。DP 组变化最慢，则保证同一 DP 组的 `attn_cp_size * attn_tp_size` 张卡也是一段连续区间。

### 3.3 三个维度对应的进程组

三个坐标各自对应一个通信组，**组内成员 = 固定另外两维、只让本维变化的那些卡**（`parallel_state.py:2126-2196`）：

| 进程组 | 成员构成 | 组内 rank 是否连续 |
| --- | --- | --- |
| `_ATTN_TP` | 固定 `(dp, cp)`，遍历 `tp` | ✅ 连续（步长 1） |
| `_ATTN_CP` | 固定 `(dp, tp)`，遍历 `cp` | ❌ 跨步（步长 `attn_tp_size`） |
| DP 组（隐式） | 固定 `dp`，遍历 `(cp, tp)` | ✅ 连续的一整段 |

CP 组的构造代码正是这个"跨步取样"：

```python
# parallel_state.py:2141 —— 固定 dp_idx 与 attn_tp_idx，沿 cp 维以 attn_tp_size 为步长取样
st = dp_idx * attn_tp_size * attn_cp_size + attn_tp_idx
en = (dp_idx + 1) * attn_tp_size * attn_cp_size + attn_tp_idx
ranks = list(range(st, en, attn_tp_size))
```

### 3.4 完整示例：`tp_size=8, attn_dp_size=2, attn_cp_size=2`

由容量守恒得 `attn_tp_size = 8 / 2 / 2 = 2`。三个维度的步长分别是 dp→4、cp→2、tp→1，展开后：

| 全局 rank | dp | cp | tp | 反算校验 $(dp \cdot 2 + cp) \cdot 2 + tp$ |
| --- | --- | --- | --- | --- |
| g0 | 0 | 0 | 0 | $(0{\cdot}2{+}0){\cdot}2{+}0 = 0$ ✅ |
| g1 | 0 | 0 | 1 | $(0{\cdot}2{+}0){\cdot}2{+}1 = 1$ ✅ |
| g2 | 0 | 1 | 0 | $(0{\cdot}2{+}1){\cdot}2{+}0 = 2$ ✅ |
| g3 | 0 | 1 | 1 | $(0{\cdot}2{+}1){\cdot}2{+}1 = 3$ ✅ |
| g4 | 1 | 0 | 0 | $(1{\cdot}2{+}0){\cdot}2{+}0 = 4$ ✅ |
| g5 | 1 | 0 | 1 | $(1{\cdot}2{+}0){\cdot}2{+}1 = 5$ ✅ |
| g6 | 1 | 1 | 0 | $(1{\cdot}2{+}1){\cdot}2{+}0 = 6$ ✅ |
| g7 | 1 | 1 | 1 | $(1{\cdot}2{+}1){\cdot}2{+}1 = 7$ ✅ |

由此得到的三套进程组（与 `parallel_state.py` 的构造结果完全一致）：

```
DP 组（2 个，各 4 卡）:  [g0 g1 g2 g3]  [g4 g5 g6 g7]     ← 不同请求
ATTN_TP 组（4 个，各 2 卡）: [g0 g1] [g2 g3] [g4 g5] [g6 g7]  ← 切 head，连续
ATTN_CP 组（4 个，各 2 卡）: [g0 g2] [g1 g3] [g4 g6] [g5 g7]  ← 切序列，步长 2
```

图示（左半边是 DP 组 0，右半边是 DP 组 1）：

```
        ┌──────────── DP 组 0（请求 ab）────────────┐ ┌──────── DP 组 1（请求 cd）────────┐
        │   cp=0            │   cp=1               │ │   cp=0          │   cp=1         │
        │ ┌───────┬───────┐ │ ┌───────┬───────┐    │ │ ┌────┬────┐     │ ┌────┬────┐    │
        │ │  g0   │  g1   │ │ │  g2   │  g3   │    │ │ │ g4 │ g5 │     │ │ g6 │ g7 │    │
        │ │ tp=0  │ tp=1  │ │ │ tp=0  │ tp=1  │    │ │ │tp=0│tp=1│     │ │tp=0│tp=1│    │
        │ └───┬───┴───┬───┘ │ └───┬───┴───┬───┘    │ │ └────┴────┘     │ └────┴────┘    │
        │     └── TP ───┘   │     └── TP ───┘      │ │                 │                │
        │         └──────── CP ─────────┘          │ │      └──── CP ──────┘            │
        └──────────────────────────────────────────┘ └──────────────────────────────────┘
             g0↔g2, g1↔g3 组成 CP 组（序列前半/后半）
```

解读：g0 和 g2 处理**同一批请求 `ab` 的不同序列段**（CP 伙伴），而 g0 和 g1 处理**同一段序列的不同 head**（TP 伙伴）；g0 与 g4 则处理**完全不同的请求**（DP 伙伴）。

### 3.5 典型配置速查

固定 `tp_size = 8`，改变另外两个自由参数，`attn_tp_size` 随之被动确定：

| tp_size | attn_dp_size | attn_cp_size | attn_tp_size | 含义 |
| --- | --- | --- | --- | --- |
| 8 | 1 | 1 | 8 | 退化为普通 TP（未开 DP attention） |
| 8 | 2 | 1 | 4 | 2 个 DP 组，每组 4 卡做 attention TP → **DP + 小 TP** |
| 8 | 8 | 1 | 1 | 8 个 DP 组，每组 1 卡 → attention **纯 DP，完全不走 TP** |
| 8 | 2 | 2 | 2 | 三维混合（即 §3.4 的例子） |
| 8 | 1 | 8 | 1 | 纯 CP：8 卡切同一条超长序列 |

> 两个容易踩的点：
> 1. "attention 不走 TP"只有在 `attn_tp_size == 1` 时才成立；否则是"TP 度变小 + 叠加 DP"。
> 2. `attn_cp_size` 会**挤占** `attn_tp_size` 的份额。开了 CP 之后 attention 的 TP 度会同比例下降，`num_heads` 必须仍能被新的 `attn_tp_size` 整除（`deepseek_v2.py:1497` 处有 `assert num_heads % attn_tp_size == 0`）。

### 3.6 CP 与 MoE 的衔接

CP 只是 **attention 的**切分方式；MoE 是**逐 token 独立**的，不需要序列切分。当 MoE 侧的并行度小于 CP 度时，进入 MoE 前必须先把被 CP 切散的 token 汇聚回来：

```python
# dp_attention.py:625
def is_enable_moe_cp_allgather() -> bool:
    """当 moe_dp_size < attn_cp_size 时返回 True，此时进入 MoE 前需跨 CP rank 做 all-gather。"""
    return sa.attn_cp_size > sa.moe_dp_size
```

SGLang 的实现很巧妙：直接让 `_MOE_DP = _ATTN_CP`（`parallel_state.py:2204-2208`），**复用 CP 组当作 MoE 的 DP 组**，这样 §4 里现成的 gather/scatter 机制就自动完成了 token 共享，不需要额外写一套通信。这也是 §2.1 `ScatterMode` 里 `MOE_FULL` 这个模式、以及 `ScatterMode.FULL` 的组大小要写成 `tp_size // attn_cp_size` 的原因。

---

## 4. 一次前向的数据流：gather → MoE → scatter

由于 attention 阶段各 rank 只有自己那批 token，而 MoE 需要在全局 token 上计算，中间必须做 **DP gather**；MoE 算完再 **DP scatter** 切回去。

### 4.1 gather 的两种实现（`dp_attention.py:472-540`）

`_dp_gather` 根据 padding 模式选择：

- **`_dp_gather_via_all_gather`**（MAX_LEN 模式）：各 rank 先 `reduce_scatter` 再 `all_gather_into_tensor`，把各 rank 的 token 拼进 global buffer。
- **`_dp_gather_via_all_reduce`**（SUM_LEN 模式）：先把本 rank 的 token 用 `memcpy_triton` 写到 global buffer 里属于自己的 `[local_start_pos, +local_num_tokens)` 区间（其余位置填 0），再对整个 global buffer 做 **all-reduce 求和**——因为每个位置只有一个 rank 写了非零值，求和即等于拼接。

`get_dp_local_info`（`:396-411`）负责算出本 DP rank 在 global buffer 里的偏移：对 `global_num_tokens` 做前缀和，`dp_rank` 之前所有 rank 的 token 数之和即本 rank 的 `local_start_pos`。

### 4.2 数值直觉（SUM_LEN / all-reduce 版）

设 2 个 DP rank，hidden=2：
- rank 0 有 1 个 token `[a0]`，rank 1 有 2 个 token `[b0, b1]`；
- global buffer 长度 = 1 + 2 = 3。

| 位置 | rank 0 写入 | rank 1 写入 | all-reduce 求和 |
| --- | --- | --- | --- |
| 0 | `a0` | `0` | `a0` |
| 1 | `0` | `b0` | `b0` |
| 2 | `0` | `b1` | `b1` |

all-reduce 后 global buffer = `[a0, b0, b1]`，即所有 rank 的 token 拼接结果。MoE 在这 3 个 token 上计算，算完 scatter 把位置 0 还给 rank 0、位置 1~2 还给 rank 1。

---

## 5. 两种 padding 模式：MAX_LEN 与 SUM_LEN

各 DP rank 的 token 数往往不等（`global_num_tokens` 参差），gather 前需要统一长度。模式由 `DpPaddingMode.get_dp_padding_mode`（`dp_attention.py:67-91`）选择：

| 模式 | 缓冲区长度 | gather 方式 | 适用 |
| --- | --- | --- | --- |
| **MAX_LEN** | 各 rank padding 到 `max(global_num_tokens)` | all-gather | 各 rank token 数接近时通信更省；可启用对称内存优化 |
| **SUM_LEN** | 长度 = `sum(global_num_tokens)` | all-reduce | token 分布不均时避免 padding 浪费 |

选择逻辑（简化）：

- `is_extend_in_batch and dp_size > 1`（prefill/extend 阶段）→ 优先 **SUM_LEN**，避免长短不齐的 padding 开销；
- 否则比较 `sum_len` 与 `max_len * dp_size` 的通信代价，取更小者；代价相等时偏向 **MAX_LEN**（可启用对称内存）。

---

## 6. 与普通 TP / 副本级 DP 的区别

有三个容易混淆的概念，务必区分：

| 概念 | 切什么 | 进程组 | 谁管理 |
| --- | --- | --- | --- |
| **普通 TP** | attention 按 head、FFN 按张量维 | 单个 TP group | 层内切分 |
| **DP Attention** | attention 按 **token（请求）** 走 DP，MoE 仍全局 TP/EP | **复用同一个 TP group**，层内切换维度 | `initialize_dp_attention` + gather/scatter |
| **副本级 DP**（`DataParallelController`） | 整个模型副本，请求分发到不同副本 | **多个独立 TP group**，每副本一套 | `DataParallelController` |

- DP Attention 与普通 TP 是**同一组进程**，只是 attention 阶段把这组进程逻辑上分成 `attn_dp_size` 个 DP 子组；
- 副本级 DP 是**多套完全独立的进程组**（每个 `dp_rank` 各有 scheduler/TP group），由 DPC 在更外层分发请求。

> 例：LongCat decode 集群 `dp_size=16, attn_tp_size=8`——这是 **16 个副本级 DP 单元**（由 DPC 管理），**每个副本内部 8 卡做 attention TP**。这里的 `dp_size=16` 是副本级 DP，不是层内 DP attention 的 `attn_dp_size`；两者是不同层级。

---

## 7. SGLang 中的实现

### 7.1 初始化：`python/sglang/srt/layers/dp_attention.py`

- `initialize_dp_attention`（`:279`）：从 `server_args` 读 `enable_dp_attention` / `dp_size` / `attn_cp_size` / `moe_dense_tp_size`，计算并缓存全局的 `_ATTN_DP_RANK` / `_ATTN_DP_SIZE`、本地的 `_LOCAL_ATTN_DP_*`。
- `compute_dp_attention_world_info`（`:245`）：从 `tp_rank` 反推 `(attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size)`。
- `compute_dp_attention_local_info`（`:262`）：`moe_dense_tp_size` 场景下算本节点内的局部 DP 布局。

### 7.2 gather / scatter

- `_dp_gather` / `dp_gather_partial`（`:527`, `:543`）：attention → MoE 前把 token 汇聚成 global batch。
- `_dp_gather_via_all_gather`（`:507`）/ `_dp_gather_via_all_reduce`（`:472`）：两种 gather 实现。
- `get_dp_local_info`（`:396`）/ `get_dp_local_slice_cpu`（`:414`）：计算本 rank 在 global buffer 中的偏移与长度。
- `_DpGatheredBufferWrapper`（`:103`）+ `set_dp_buffer_len`（`:190`）：管理 gather 用的全局/本地缓冲区。

### 7.3 进程组查询

- `get_attention_dp_rank` / `get_attention_dp_size`（`:355`, `:360`）：查询 attention DP 维度；
- `get_attention_tp_group` / `get_attention_tp_rank` / `get_attention_tp_size`（`:331`, `:335`, `:339`）：查询 attention 内的小 TP 组。

### 7.4 与 DataParallelController 的衔接

`managers/data_parallel_controller.py` 中 `enable_dp_attention` 为真时走 `launch_dp_attention_schedulers`（所有 DP rank 复用同一 TP group、共用 nccl 端口），否则走 `launch_dp_schedulers`（每副本独立）。详见 `python/sglang/srt/managers/README_data_parallel_controller_zh.md`。

### 7.5 启动参数

```bash
python -m sglang.launch_server --model <MLA-model> --tp-size 8 --dp-size 8 --enable-dp-attention
```

- `--enable-dp-attention`：开启层内 DP attention；
- `--dp-size`：此处即 `attn_dp_size`（在同一 TP group 内切几个 attention DP 组）；
- 可选 `--moe-dense-tp-size`、`--attn-cp-size` 调整 MoE 侧局部 TP 与 context 并行。

---

## 8. 局限与权衡

1. **多一次 gather/scatter 通信**：attention 与 MoE 之间必须汇聚/切回 token，是 DP attention 相对纯 TP 的额外开销；收益（省 KV Cache 显存、免 attention TP）通常远大于此。
2. **负载不均导致 padding 浪费**：各 DP rank 的 token 数不等时，MAX_LEN 模式会 padding；SUM_LEN 缓解但 all-reduce 缓冲更大。框架按代价自动选模式。
3. **主要面向 MLA / 大 MoE 模型**：标准 MHA 且无大 MoE 时，普通 TP 已足够，DP attention 收益有限。
4. **切分约束**：`tp_size` 必须能被 `attn_dp_size * attn_cp_size` 整除。
5. **单条长 seq 的 attention 延迟比 TP 高**：一句话总结——单条长 seq 的 attention，DP attention 因整条序列压在一张卡、没有 head 分摊而比 TP 慢；但它换来的是高并发吞吐、省 KV Cache 显存、免 all-reduce，适合多请求而非单条长 seq。真要给单条超长 seq 提速，正确做法是叠加 **CP**（按 token 维切分该序列，见 `CP.md`），而不是指望 DP attention——这也是 `enable_dp_attention` 常作为 zigzag CP 前置条件的原因。

---

## 参考与延伸

- 同目录：`TP.md`（张量并行，DP attention 的基础）、`SP.md`（序列并行，与 DP-attention 共用同一套
  `ScatterMode` gather/scatter 机制，对比见其 §7.2）、`CP.md`（上下文并行）、`../MLA.md`（MLA 与 KV Cache 压缩）。
- SGLang 代码：`python/sglang/srt/layers/dp_attention.py`、`python/sglang/srt/managers/data_parallel_controller.py`。
- 相关文档：`python/sglang/srt/managers/README_data_parallel_controller_zh.md`。

