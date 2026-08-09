# 序列并行（Sequence Parallelism, SP）原理详解

> 本文介绍序列并行的动机（TP 无法消除的**激活冗余**）、"按 token 维切分 + all-reduce 拆成
> reduce-scatter/all-gather"的核心思想、为什么它**通信量零增加**、与 CP / DP-attention 的区别，
> 最后对应到 SGLang 中 `layers/communicator.py` 的 `ScatterMode` 状态机与真实实现。
>
> 前置阅读：`TP.md`（张量并行，SP 的宿主）、`collective_communication.md`（reduce-scatter / all-gather 语义）。
> 延伸阅读：`CP.md`（同样切序列，但目标完全不同）、`DP_attention.md`。

## 目录

1. [为什么需要序列并行：TP 的激活冗余](#1-为什么需要序列并行tp-的激活冗余)
2. [核心思想：把冗余区间按 token 维切开](#2-核心思想把冗余区间按-token-维切开)
3. [为什么切 token 维是安全的](#3-为什么切-token-维是安全的)
4. [关键恒等式：all-reduce 拆成两个半场](#4-关键恒等式all-reduce-拆成两个半场)
5. [完整数据流：一个 Transformer block](#5-完整数据流一个-transformer-block)
6. [数值示例：手算通信量与显存](#6-数值示例手算通信量与显存)
7. [SP 与 TP / CP / DP-attention 的关系](#7-sp-与-tp--cp--dp-attention-的关系)
8. [SGLang 中的实现](#8-sglang-中的实现)
9. [局限与权衡](#9-局限与权衡)

---

## 1. 为什么需要序列并行：TP 的激活冗余

TP 把权重按 hidden 维切开，每卡只存 $1/p$ 的参数。但**激活（activation）并没有全切**——标准 TP 的一个 Transformer block 里，有一段区间是 $p$ 张卡**各存一份完全相同的完整激活**：

```
       [完整 hidden: s × d]        ← 每张卡都存一份（冗余!）
            ↓ LayerNorm            ← 每张卡重复算同样的东西
       [完整 hidden: s × d]
            ↓ ColumnParallel QKV   ← 从这里才开始切（切 head）
       [切片: s × d/p]
            ↓ Attention
            ↓ RowParallel o_proj
            ↓ all-reduce           ← 回到完整
       [完整 hidden: s × d]        ← 又冗余了
            ↓ 残差 + LayerNorm
```

问题拆开看：

- **显存冗余**：LayerNorm 输入/输出、残差分支的 $s \times d$ 张量，$p$ 张卡存了 $p$ 份相同数据，有效利用率只有 $1/p$；
- **算力冗余**：LayerNorm、残差加、Dropout 这些逐元素算子，每张卡都把**同一份完整数据**算了一遍，纯粹浪费。

TP 自己解决不了这个问题——因为这段区间的算子（LayerNorm）需要**完整的 hidden 维**做归约，不能沿 $d$ 切。

**序列并行（SP）的思路：既然不能沿 hidden 维切，那就沿 token（序列）维切。** 在 TP 覆盖不到的区间，把 $s$ 切成 $p$ 段，每卡只处理 $s/p$ 个 token 的 LayerNorm 与残差。

---

## 2. 核心思想：把冗余区间按 token 维切开

SP **不是独立的并行维度**，而是 TP 的**互补切分**：同一组 GPU，在一个 block 内部**沿不同维度来回切换**。

| 区间 | 切的维度 | 每卡持有 | 算子 |
| --- | --- | --- | --- |
| **SP 区** | **`s`（token 维）** | $s/p \times d$（完整 hidden，部分 token） | LayerNorm、残差、Dropout |
| **TP 区** | `d`（hidden / head 维） | $s \times d/p$（完整 token，部分 hidden） | QKV、Attention、MLP |

```
   SP 区                TP 区                 SP 区
[s/p × d]  --all-gather-->  [s × d/p]  --reduce-scatter-->  [s/p × d]
 切 token                    切 hidden              切 token
```

两个区间的**总元素数都是 $\frac{s \cdot d}{p}$**——每卡的激活占用在整个 block 内**始终是 $1/p$**，冗余被彻底消除。

关键在于：**切换维度的通信是免费的**（§4）。

---

## 3. 为什么切 token 维是安全的

SP 区的算子都是 **per-token 独立**的，切 token 天然不需要任何跨卡通信。

以 LayerNorm 为例：

$$\text{LN}(x_t) = \gamma \odot \frac{x_t - \mu_t}{\sigma_t} + \beta,\quad
\mu_t = \frac{1}{d}\sum_{k=1}^{d} x_{t,k},\quad
\sigma_t^2 = \frac{1}{d}\sum_{k=1}^{d}(x_{t,k}-\mu_t)^2$$

注意求和下标是 $k$（**hidden 维**），不是 $t$：

- **沿 $d$ 切（TP 方向）** → $\mu_t$ 的求和被切断，必须 all-reduce 统计量 → ❌ 需通信、需改算子；
- **沿 $s$ 切（SP 方向）** → 每个 token 的 $\mu_t, \sigma_t$ 只依赖自己那一行，本地就能算完 → ✅ **零通信、零改算子**。

RMSNorm 同理（$\text{RMS}(x_t)=\sqrt{\frac1d\sum_k x_{t,k}^2}$），残差加、Dropout、激活函数更是纯逐元素。

> 这正是 `collective_communication.md` 里"选择切哪个维度，使尽可能多的算子落在无需改写的类别"这一设计原则的最佳范例：**SP 精确地挑了那个让归约维度不被切断的维度**。

---

## 4. 关键恒等式：all-reduce 拆成两个半场

SP 最精妙的地方在于——**它不增加任何通信量**。

回顾 `collective_communication.md` §11 的恒等式：

$$\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}$$

对应通信量（每 rank）：

$$\underbrace{2\frac{p-1}{p}N}_{\text{all-reduce}} = \underbrace{\frac{p-1}{p}N}_{\text{reduce-scatter}} + \underbrace{\frac{p-1}{p}N}_{\text{all-gather}}$$

**标准 TP** 在 `o_proj` / `down_proj` 之后各做一次完整 all-reduce。
**SP + TP** 把这次 all-reduce 从中间劈开：

```
标准 TP：   [部分和] ──────── all-reduce ────────► [完整 s × d]
                                                       ↓ LayerNorm（冗余计算）

SP + TP：   [部分和] ─── reduce-scatter ───► [s/p × d]
                                                ↓ LayerNorm（各算各的 1/p）
                     ─── all-gather ──────► [完整 s × d]
```

**通信总量完全相同**，但**中间夹了一段 SP 计算**。这就是 SP 的"免费午餐"：

- 通信量：**不变**（$2\frac{p-1}{p}N$）；
- 激活显存：LayerNorm/残差区从 $s \times d$ 降到 $s/p \times d$，即 **$1/p$**；
- 逐元素算力：从"每卡算全部"降到"每卡算 $1/p$"。

> ⚠️ 注意通信**次数**变多了（一次 all-reduce → 两次集合通信），小 batch 下 kernel 启动开销和延迟略有增加。见 §9。

---

## 5. 完整数据流：一个 Transformer block

设 TP=SP=$p$，序列长 $s$，hidden 维 $d$：

```
┌─ SP 区 ────────────────────────────────────────────┐
│  hidden: [s/p, d]      residual: [s/p, d]           │
│      ↓ LayerNorm（本地，无通信）                     │
└────────────────────────────────────────────────────┘
       ↓ all-gather（沿 token 维拼回）
┌─ TP 区 ────────────────────────────────────────────┐
│  hidden: [s, d]                                     │
│      ↓ ColumnParallel QKV      → [s, d/p]           │
│      ↓ Attention（本卡负责部分 head）                │
│      ↓ RowParallel o_proj      → [s, d] 部分和       │
└────────────────────────────────────────────────────┘
       ↓ reduce-scatter（求和 + 沿 token 维切开）
┌─ SP 区 ────────────────────────────────────────────┐
│  hidden: [s/p, d]                                   │
│      ↓ 残差加 + LayerNorm（本地，无通信）             │
└────────────────────────────────────────────────────┘
       ↓ all-gather
┌─ TP 区 ─── MLP ────────────────────────────────────┐
│      ↓ ColumnParallel gate/up  → [s, d_ff/p]        │
│      ↓ SiLU（逐元素）                                │
│      ↓ RowParallel down_proj   → [s, d] 部分和       │
└────────────────────────────────────────────────────┘
       ↓ reduce-scatter
┌─ SP 区 ── 进入下一层 ───────────────────────────────┐
│  hidden: [s/p, d]                                   │
└────────────────────────────────────────────────────┘
```

每个 block 的通信：**2 × (all-gather + reduce-scatter)**，与标准 TP 的 **2 × all-reduce** 通信量相等。

**残差分支要跟着一起切**：残差是逐元素加，所以它一直保持在 SP 布局（`[s/p, d]`）即可，不需要 gather。SGLang 里 hidden_states 与 residual 的布局是**分别追踪**的（见 §8 的 `LayerScatterModes`）。

---

## 6. 数值示例：手算通信量与显存

设 $p=8$（TP=SP=8）、$s=4096$、$d=8192$、bf16（2 字节）：

单个 $s \times d$ 激活张量大小：

$$N = 4096 \times 8192 \times 2\ \text{B} = 64\ \text{MB}$$

### 通信量对比（每 block，每 rank）

| 方案 | 通信操作 | 每 rank 通信量 |
| --- | --- | --- |
| 标准 TP | 2 × all-reduce | $2 \times 2\cdot\frac{7}{8}\cdot 64 = 224$ MB |
| TP + SP | 2 × (reduce-scatter + all-gather) | $2 \times (\frac{7}{8}\cdot64 + \frac{7}{8}\cdot64) = 224$ MB |

**完全相等** ✓

### 激活显存对比（LayerNorm / 残差区，每 rank）

| 方案 | 每卡该区间激活 |
| --- | --- |
| 标准 TP | $64$ MB（完整，8 卡共存了 512 MB 的重复数据） |
| TP + SP | $64 / 8 = 8$ MB |

**省下 56 MB/层/张量**。对 60 层模型、每层 2 处 SP 区间：

$$60 \times 2 \times 56\ \text{MB} \approx 6.7\ \text{GB}\ \text{每卡节省}$$

这部分显存可以直接换成更大的 KV Cache 池 → 更高并发。

---

## 7. SP 与 TP / CP / DP-attention 的关系

### 7.1 SP vs CP：同样切序列，目标完全不同

这是最容易混淆的一对。**判据只有一个：attention 计算时，本卡手上是完整序列还是分片？**

| | **SP（序列并行）** | **CP（上下文并行）** |
| --- | --- | --- |
| 切什么 | 序列维（token） | 序列维（token） |
| **切在哪些层** | **只在 TP 覆盖不到的层**（LayerNorm/残差） | **全程，包括 attention 内部** |
| **attention 时** | **完整序列**（进 attention 前已 all-gather） | **分片序列**（靠 KV all-gather / ring 通信） |
| attention 算子 | ✅ 完全不用改 | ❌ 必须改（ring softmax / KV gather / zigzag） |
| 负载均衡 | ✅ 天然均衡 | ❌ causal 导致不均，需 zigzag（`CP.md` §4） |
| 省 KV Cache | ❌ **不省**（KV 是完整序列的） | ✅ 每卡只存 $1/\text{CP}$ |
| 省激活 | ✅ norm/残差区降到 $1/p$ | ✅ 全程 $1/\text{CP}$ |
| 通信增量 | **0**（all-reduce 换形式） | 每 attention 层额外 KV 通信 |
| 通信组 | **复用 TP group** | **独立的 `_ATTN_CP` group** |
| 解决的问题 | 激活内存冗余 | 单序列过长装不下 |

一句话：

> **SP 是"TP 的省显存补丁"——切序列但躲开 attention；**
> **CP 是"长序列的独立并行"——切序列并硬啃 attention。**

### 7.2 SP vs DP-attention

在 SGLang 里两者的实现机制高度重合（都靠 token 维的 gather/scatter），但语义不同：

| | SP | DP-attention |
| --- | --- | --- |
| 切分依据 | 同一批数据的 token 均分 | 按**请求**归属分给不同 attn DP rank |
| 目的 | 消除激活冗余 | 消除 **KV Cache 冗余**（MLA 下每卡存全量 KV 太贵） |
| MoE 部分 | 仍是 TP/EP | all-gather 后走全局 TP/EP |

详见 `DP_attention.md`。SGLang 的 `communicator.py` 用**同一套 `ScatterMode` 状态机**统一描述了 SP、DP-attention、CP、MoE 的所有布局切换。

### 7.3 组合关系

SP 与 TP 是**同一组 GPU 的两个视角**，不是相乘关系：

$$\text{world} = \underbrace{TP}_{\text{内含 SP}} \times CP \times DP \times PP$$

所以通常 $\text{SP size} = \text{TP size}$（或 `attn_tp_size`），不单独配置。

---

## 8. SGLang 中的实现

SGLang **没有名为 "sequence parallel" 的独立开关**——SP 的思想被吸收进了统一的**布局（layout）状态机**：`python/sglang/srt/layers/communicator.py`。

### 8.1 `ScatterMode`：描述"数据当前切在哪个维度"

```python
class ScatterMode(Enum):
    SCATTERED = auto()      # [a, b, c, d]     —— 沿 token 维散开（SP 布局）
    TP_ATTN_FULL = auto()   # [ab, ab, cd, cd] —— attn TP 组内持有完整数据
    FULL = auto()           # [abcd, ...]      —— 全局完整
    MOE_FULL = auto()       # MoE 组内完整
```

`SCATTERED` 就是 **SP 布局**：hidden 沿 token 维切开，每 rank 只持有 $1/p$ 的 token。

### 8.2 `LayerScatterModes`：每层的布局规划

`LayerScatterModes.init_new(...)` 为每一层静态推导出 5 个布局点：

| 字段 | 含义 |
| --- | --- |
| `layer_input_mode` | 进入本层时的布局 |
| `attn_mode` | attention 计算时的布局（**恒为 `TP_ATTN_FULL`**） |
| `mlp_mode` | MLP/MoE 计算时的布局 |
| `middle_residual_mode` | attention 与 MLP 之间残差的布局 |
| `layer_output_mode` | 离开本层时的布局 |

**`attn_mode` 恒为 `TP_ATTN_FULL`——这正是 §7.1 中"SP 在 attention 前必须 gather 回完整序列"的代码证据。**

而 `_compute_mlp_mode` 在 `enable_moe_dense_fully_dp()`（即 `--moe-dense-tp-size 1`）时返回 `SCATTERED`——这就是稠密 MLP 走 SP 布局的开关。

### 8.3 布局切换：reduce-scatter / all-gather 的落点

`CommunicateWithAllReduceAndLayerNormFn` 里两个对称的方法，就是 §4 拆开的两个半场：

- **`_scatter_hidden_states_and_residual`**（`TP_ATTN_FULL → SCATTERED`）——**进入 SP 区**：

```python
hidden_states = hidden_states.tensor_split(context.attn_tp_size)[context.attn_tp_rank]
attn_tp_reduce_scatter_tensor(hidden_states, input_hidden_states)   # ← all-reduce 的前半
if residual_input_mode == ScatterMode.TP_ATTN_FULL:
    residual = residual.tensor_split(context.attn_tp_size)[context.attn_tp_rank]  # 残差跟着切
if hidden_states.shape[0] != 0:
    hidden_states, residual = layernorm(hidden_states, residual)    # ← 在 1/p 数据上做 LayerNorm
```

- **`_gather_hidden_states_and_residual`**（`SCATTERED → TP_ATTN_FULL`）——**离开 SP 区**：

```python
if residual_input_mode == ScatterMode.SCATTERED and context.attn_tp_size > 1:
    residual, local_residual = get_local_dp_buffer(get_attention_tp_group()), residual
    attn_tp_all_gather_into_tensor(residual, local_residual)        # ← all-reduce 的后半
```

### 8.4 通信原语：`python/sglang/srt/layers/dp_attention.py`

| 函数 | 原语 | 作用 |
| --- | --- | --- |
| `attn_tp_reduce_scatter_tensor` | Reduce-Scatter | 进入 SP 区（求和 + 切 token） |
| `attn_tp_all_gather_into_tensor` | All-Gather | 离开 SP 区（拼回完整 token） |
| `attn_tp_all_reduce` | All-Reduce | 不走 SP 时的传统路径 |
| `dp_gather_partial` / `dp_gather_replicate` | All-Gather | DP-attention 的跨 DP 收集 |
| `dp_scatter` | Scatter | DP-attention 的回切 |
| `dp_reduce_scatter_tensor` | Reduce-Scatter | TP≠DP 时的组合路径 |

`attn_cp_all_gather_into_tensor` / `attn_cp_reduce_scatter_tensor` 是 CP group 的对应物——**同样的原语，不同的 group**，再次印证 SP 与 CP 的结构相似性。

### 8.5 相关开关

| 开关 | 位置 | 作用 |
| --- | --- | --- |
| `--moe-dense-tp-size 1` | `server_args.py: moe_dense_tp_size` | 稠密 MLP 走 `SCATTERED`（SP）布局，不做 TP |
| `--enable-attn-tp-input-scattered` | `server_args.py: enable_attn_tp_input_scattered` | 让 attention 的**输入**保持 scattered，延后 all-gather（`AttnTpContext`） |
| `--disable-attn-tp-gather` | `server_args.py: disable_attn_tp_gather` | 配套的 gather 抑制 |
| `--enable-dp-attention` | — | 开启 DP-attention，与 SP 共用 gather/scatter 机制 |

`LayerCommunicator.should_use_reduce_scatter(forward_batch)` 在运行时决定本层输出是否走 reduce-scatter（而非 all-reduce），条件包括 `dp_padding_mode.is_max_len()`、CP 开启、`input_scattered` 等。

### 8.6 `enable_attn_tp_input_scattered`：更激进的 SP

`AttnTpContext` 实现了一个进阶优化——让**进入 attention 的输入也保持 scattered**，把 all-gather 推迟到 QKV 投影之后（gather 压缩后的 `qkv_latent` 而非完整 hidden，通信量更小）：

```python
def fetch_qkv_latent(self):
    self.qkv_latent_ = self.qkv_latent_func(self.hidden_states_local, self.forward_batch)
    if get_attn_tp_context().input_scattered:
        self.qkv_latent_ = self.tp_all_gather_hidden_states(self.qkv_latent_, self.forward_batch)
    return self.qkv_latent_
```

生效条件严格（`init_context`）：需 CUDA/NPU、`q_lora_rank is not None`（即 MLA 类模型）、非 DSA、`tp_size > 1`、**未开 DP-attention**、MoE a2a 后端为 none、非 EAGLE3 等。

---

## 9. 局限与权衡

1. **通信次数翻倍**：一次 all-reduce 变成 reduce-scatter + all-gather 两次集合通信。虽然**总字节数不变**，但 kernel 启动开销和固定延迟增加了一次。小 batch / decode 阶段（$s$ 很小）时，延迟敏感，收益可能为负——这是 SGLang 用 `should_use_reduce_scatter` 做**运行时判断**而非静态开启的原因。

2. **不省 KV Cache**：SP 只压激活。KV Cache 是推理显存的大头，要省它得靠 **MLA 压缩**、**DP-attention** 或 **CP**。

3. **收益随 batch/序列增长**：激活显存正比于 $s \times \text{bs}$。decode 阶段每步只有 1 个 token/请求，激活本来就小，SP 几乎没收益；**prefill 长序列**才是 SP 的主场。

4. **与 CUDA Graph / 融合优化的冲突**：`should_fuse_mlp_allreduce_with_next_layer` 在 `mlp_mode == SCATTERED` 时直接返回 `False`——**SP 布局下没有 all-reduce 可供融合**，因此与 FlashInfer/AITER 的 allreduce-fusion 互斥。`enable_attn_tp_input_scattered` 也对 piecewise CUDA graph 和 EAGLE3 做了排除。

5. **切分要求整除**：token 数需能被 $p$ 整除，否则要 padding（SGLang 用 `DpPaddingMode` + `get_local_dp_buffer` 统一处理），padding 本身带来少量浪费。

6. **不是独立扩展维度**：SP 复用 TP group，**不能靠加 SP 来突破 TP 的规模上限**。要扩展并行规模仍需 DP/PP/EP/CP。

---

## 参考与延伸

- 同目录：`TP.md`（张量并行，SP 的宿主）、`CP.md`（上下文并行，切序列但含 attention）、`DP.md`、`DP_attention.md`、`EP.md`、`PP.md`、`collective_communication.md`（reduce-scatter / all-gather 语义与恒等式）。
- SGLang 代码：`python/sglang/srt/layers/communicator.py`（`ScatterMode` / `LayerScatterModes` / `LayerCommunicator`）、`python/sglang/srt/layers/dp_attention.py`（`attn_tp_reduce_scatter_tensor` / `attn_tp_all_gather_into_tensor`）、`python/sglang/srt/server_args.py`。
- 相关论文：Megatron-LM 的 *Reducing Activation Recomputation in Large Transformer Models*（提出 TP + SP 组合，SP 的原始出处）。

