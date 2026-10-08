# 上下文并行（Context Parallelism, CP）原理详解

> 本文介绍上下文并行的动机、按**序列维**切分的核心思想、causal attention 带来的负载不均与
> **zigzag** 解决方案、CP 下 attention 如何通过 all-gather 拿到完整 KV，以及 CP 与 TP/DP/EP 的关系，
> 最后对应到 SGLang 中 `layers/utils/cp_utils.py` 与 `enable_prefill_cp` 的真实实现。
>
> 前置阅读：`TP.md`（张量并行）、`DP_attention.md`。CP 在 SGLang 中**主要用于 prefill 阶段**长序列。
>
> ⚠️ **不要混淆 CP 与 SP**：两者都切序列维，但 SP（`SP.md`）在进 attention 前会 all-gather 回完整序列、
> attention 算子无需修改；而 CP **带着分片做 attention**，必须改算子并处理 causal 负载不均。对比见 `SP.md` §7.1。

## 目录

1. [为什么需要上下文并行](#1-为什么需要上下文并行)
2. [核心思想：按序列维切分](#2-核心思想按序列维切分)
3. [难点：causal attention 的负载不均](#3-难点causal-attention-的负载不均)
4. [zigzag 切分：均衡负载](#4-zigzag-切分均衡负载)
5. [CP 下的 attention：all-gather 完整 KV](#5-cp-下的-attentionall-gather-完整-kv)
6. [数值示例：手算 zigzag 切分](#6-数值示例手算-zigzag-切分)
7. [CP 与 TP / DP / EP 的关系](#7-cp-与-tp--dp--ep-的关系)
8. [完整示例：tp=8, dp=2, cp=2 的 attention 执行流程](#8-完整示例tp8-dp2-cp2-的-attention-执行流程)
9. [SGLang 中的实现](#9-sglang-中的实现)
10. [局限与权衡](#10-局限与权衡)

---

## 1. 为什么需要上下文并行

TP 切模型权重、DP 切 batch、EP 切 expert，但都没解决一个问题：**单条序列极长时怎么办？**

当上下文长度达到 128K、1M token 时：

- **激活显存爆炸**：attention 的中间激活、KV 都随序列长度线性（甚至平方）增长，单卡放不下；
- **单卡算力不足**：一条序列的 prefill 计算量随 $s^2$ 增长，单卡 prefill 延迟过高。

TP 无法解决——它切的是 hidden 维（`d_model`），序列维 $s$ 完全复制在每张卡上。

**上下文并行（CP）的思路：把同一条序列沿 token（序列）维切成若干段，分到不同卡上并行处理。** 每卡只负责序列的一部分 token，激活显存和计算量都降到 $1/\text{CP}$。

---

## 2. 核心思想：按序列维切分

回顾各并行切的维度（设 hidden 为 `[s, d_model]`）：

| 并行 | 切的维度 | 每卡持有 |
| --- | --- | --- |
| **TP** | `d_model`（hidden 维） | 完整序列、部分 hidden |
| **DP** | batch 维 | 不同请求 |
| **EP** | expert 维 | 部分 expert |
| **CP** | **`s`（序列 / token 维）** | **同一序列的一段 token** |

CP 把 `[s, d]` 沿 $s$ 切成 CP 份：rank 0 拿 token 段 0，rank 1 拿段 1……每卡本地只有 $s/\text{CP}$ 个 token。

```
完整序列 (s 个 token)：t0 t1 t2 t3 t4 t5 t6 t7
                        └─── 沿序列维切成 CP=2 份 ───┘
   rank 0:  t0 t1 t2 t3          rank 1:  t4 t5 t6 t7
```

对于 **FFN、LayerNorm** 这类**逐 token 独立**的算子，CP 天然并行——每卡各算各的 token，无需通信。

真正的难点在 **attention**：每个 token 要 attend 到它**之前所有** token 的 KV，而这些 KV 可能在别的卡上（见 §5）。

---

## 3. 难点：causal attention 的负载不均

decoder 是 **causal（因果）** attention：token $i$ 只能看到 token $0..i$。所以**越靠后的 token，需要 attend 的 KV 越多，计算量越大**。

如果简单地把序列**连续等分**（rank 0 拿前半、rank 1 拿后半）：

```
连续切分（naive）：
  rank 0:  t0 t1 t2 t3   ← 每个 token 平均 attend ~2 个 KV，计算轻
  rank 1:  t4 t5 t6 t7   ← 每个 token 平均 attend ~6 个 KV，计算重
```

结果：**rank 1 的计算量远大于 rank 0**，木桶效应下整体被慢的那张卡拖累，并行效率低。这就是 causal attention 下 CP 的核心痛点。

---

## 4. zigzag 切分：均衡负载

SGLang 用 **zigzag（之字形）切分**解决负载不均（`cp_strategy="zigzag"`，即原 `in-seq-split` 模式）。

思路：把序列切成 **$2\times\text{CP}$** 个块，然后让**每个 rank 各拿一个"靠前的块"和一个"靠后的块"**，从而每卡计算量均衡。

以 CP=2 为例，切成 $2\times 2 = 4$ 块 [B0, B1, B2, B3]：

```
zigzag 分配：
  rank 0:  B0（最前，最轻） + B3（最后，最重）
  rank 1:  B1（较前）       + B2（较后）
```

- rank 0 = 最轻 + 最重，rank 1 = 次轻 + 次重 → **两卡计算量近似相等**；
- 这正是代码里 `seq_len // (cp_size * 2)` 要求序列能切成 `2 * cp_size` 块的原因（`cp_utils.py: can_cp_split`）。

另一种策略是 **interleave（交错 / round-robin）**（`cp_strategy="interleave"`），按 token 轮流分配到各 rank，也能均衡负载。

> `cp_utils.py` 中的 `ContextParallelMetadata` 记录了切分方案：`split_list`（各块长度）、`zigzag_index`（重排索引）、`cp_reverse_index`（还原索引）等，`cp_split_and_rebuild_data` 据此对 hidden/position 做 split + zigzag 重排。

---

## 5. CP 下的 attention：all-gather 完整 KV

CP 的关键通信在 attention：token $i$ 要 attend 到它之前**所有** token 的 KV，但那些 KV 分散在各卡。解决方式是**用 all-gather 把各卡的 KV 收集齐**再做 attention：

```
每卡本地算出自己 token 段的 K、V
        │  all-gather（沿 CP group）
        ▼
每卡拿到完整序列的 K、V（拼接各 rank 的段）
        │
        ▼
本卡的 Q（只有本段 token）× 完整 K、V → attention 输出
```

- 每卡的 **Q 只有本段 token**（省了 Q 侧的算力）；
- 但 **K、V 需要 all-gather 成完整序列**，这样本段 token 才能 attend 到前面所有 token；
- MLA / DSA 模型下，all-gather 的是压缩后的 KV latent（`cp_all_gather_reorganized_into_tensor`），因为 zigzag 各 rank token 数可能不等，还需 padding 对齐后再 all-gather（`cp_utils.py`）。

因此 CP 的通信量与 KV 大小成正比——序列越长、KV 越大，all-gather 成本越高。这也是 CP 目前**限制在单机（`tp_size <= 8`）** 的原因之一（跨机 all-gather 带宽不足且有精度问题，见 `server_args.py`）。

---

## 6. 数值示例：手算 zigzag 切分

设序列长度 $s=8$、CP=2、zigzag 策略：

1. **切块**：切成 $2\times 2=4$ 块，每块 2 个 token：
   - B0 = [t0,t1]，B1 = [t2,t3]，B2 = [t4,t5]，B3 = [t6,t7]
2. **zigzag 分配**：
   - rank 0 = B0 + B3 = [t0,t1,t6,t7]
   - rank 1 = B1 + B2 = [t2,t3,t4,t5]
3. **计算量估算**（causal，token $i$ attend $i{+}1$ 个 KV）：
   - rank 0：t0,t1 各 1,2 个；t6,t7 各 7,8 个 → 合计 18
   - rank 1：t2,t3 各 3,4 个；t4,t5 各 5,6 个 → 合计 18
   - **两卡都是 18，完全均衡** ✓（对比连续切分：前半 1+2+3+4=10，后半 5+6+7+8=26，严重失衡）
4. **attention 时**：两卡各自 all-gather 出完整 KV（t0..t7），再用本卡 Q 段做 attention；
5. **输出还原**：按 `cp_reverse_index` 把结果重排回原始 token 顺序。

---

## 7. CP 与 TP / DP / EP 的关系

CP 与其他并行**正交**，可组合使用。SGLang 中 CP 复用 attention 的 rank 布局，与 DP-attention 协同：

| 并行 | 切什么 | attention 通信 |
| --- | --- | --- |
| **TP** | hidden 维 | all-reduce |
| **DP-attention** | batch（不同请求） | 无（各算各的） |
| **EP** | expert | all-to-all |
| **CP** | **同一序列的 token 段** | **KV all-gather** |

SGLang 里的关键量：

- `attn_cp_size`：attention 的 CP 路数（`cp_utils.py: get_attention_cp_size`）；启用 prefill CP 时通常 `attn_cp_size = tp_size // dp_size`（`server_args.py`）。
- CP group（`_ATTN_CP`，`distributed/parallel_state.py`）：从全局 rank 布局中按 `attn_tp` / `attn_dp` 的排布切出 CP 通信组；当 `attn_cp_size == tp_size` 时直接复用 TP group。

启用 zigzag CP 时，SGLang 还会强制一组约束（`server_args.py`）：`enable_dp_attention=True`、`moe_dense_tp_size=1`、`moe_a2a_backend=deepep`、`ep_size=tp_size`——即 **CP 与 DP-attention + EP 联合工作**处理长序列 MoE 模型。

---

## 8. 完整示例：tp=8, dp=2, cp=2 的 attention 执行流程

下面用一个具体配置串起前面各节，展示一条序列在 attention 模块里如何被 CP + head-TP **双重切分**、通信、合并。与 `attn_tp=1` 的简单情形不同，本例组内还存在 head 维 TP，因此会多出 `o_proj` 的 attention-TP all-reduce。

### 8.1 配置与 rank 结构

```
tp_size = 8
dp_size = 2
attn_cp_size = 2
```

关键关系：

$$
TP_{\text{attn}} = \frac{tp\_size}{dp\_size} = \frac{8}{2} = 4,\qquad
attn\_tp_{\text{有效}} = \frac{TP_{\text{attn}}}{attn\_cp\_size} = \frac{4}{2} = 2
$$

即每个 DP 组内有 4 个 rank，被组织成 **2(CP) × 2(head-TP)** 的二维结构：CP 切 token，head-TP 切注意力头。

8 卡布局：

```
DP 组 0（请求 A）：rank0, rank1, rank2, rank3
DP 组 1（请求 B）：rank4, rank5, rank6, rank7

DP 组 0 内部（2×2）：
             head 前半 (H0)      head 后半 (H1)
  token 段 P0   rank0              rank1        ← CP 分片 0
  token 段 P1   rank2              rank3        ← CP 分片 1
   同一行 = 同一段 token 的不同 head（head-TP 组）
   同一列 = 同一 head 子集的不同 token 段（CP 组）
```

- **组间是 DP**：组 0/组 1 处理不同请求，互不通信；
- **组内是 CP + head-TP**：CP 组沿列方向（`{rank0,rank2}`、`{rank1,rank3}`），head-TP 组沿行方向（`{rank0,rank1}`、`{rank2,rank3}`）。

下面只跟踪 DP 组 0（序列 A），组 1 完全对称。

### 8.2 序列切分（zigzag）

设序列 A 长度 $s=8$，token 记为 $t_0 \dots t_7$，按 §4 的 zigzag 切成 $2\times\text{CP}=4$ 块，得到两个 CP token 段：

```
token 段 P0: t0, t1, t6, t7   （最前块 + 最后块）
token 段 P1: t2, t3, t4, t5   （次前块 + 次后块）
```

同一 token 段的 head 再按 head-TP 切成前半 H0 / 后半 H1，落到该行的两个 rank：

```
rank0 = (P0, H0)   rank1 = (P0, H1)
rank2 = (P1, H0)   rank3 = (P1, H1)
```

causal 下两个 token 段计算量均衡（各约 18，见 §6）。

### 8.3 QKV 投影（本地，无通信）

QKV 是**列并行**：每个 rank 用切好的权重列块 $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}$（只含本 rank 负责的 head），对自己持有的 token 段做投影：

$$
Q_r = X_r W_Q^{(h)},\quad K_r = X_r W_K^{(h)},\quad V_r = X_r W_V^{(h)}
$$

- rank0：$\{t_0,t_1,t_6,t_7\}$ 的 **H0 部分** head 的 Q/K/V；
- rank1：$\{t_0,t_1,t_6,t_7\}$ 的 **H1 部分** head 的 Q/K/V；
- rank2 / rank3：$\{t_2,t_3,t_4,t_5\}$ 的 H0 / H1 部分。

注意：同一个 token 的不同 head 分给不同 rank，各算各的，**投影本身仍无需通信**。

### 8.4 KV all-gather 与注意力计算

因为 head 之间相互独立，attention 只需在**同一 head 子集内**跨 CP 段拿到完整序列 KV。所以 KV all-gather 沿 **CP 组（列方向）** 进行，而不是整个组内 4 卡：

```
CP 组 H0 = {rank0(P0,H0), rank2(P1,H0)}：
   rank0 本地 KV(t0,t1,t6,t7 | H0) ─┐
                                    ├─ all-gather → H0 的完整 t0..t7 KV
   rank2 本地 KV(t2,t3,t4,t5 | H0) ─┘

CP 组 H1 = {rank1(P0,H1), rank3(P1,H1)}：同理 all-gather 出 H1 的完整 KV
```

causal 注意力中，$Q_i$ 需要与所有 $j \le i$ 的 KV 交互：

$$
O_i = \text{softmax}\!\left(\frac{Q_i K_{\le i}^\top}{\sqrt{d}}\right) V_{\le i}
$$

- 每个 rank 的 **Q 只有本段 token、本 head 子集**；
- 沿 CP 组 all-gather 后，本段 token 即可 attend 到该 head 子集下前面所有 token；
- MLA / DSA 下 all-gather 的是压缩 KV latent，且因 zigzag 各 rank token 数可能不等，需 padding 对齐（`cp_all_gather_reorganized_into_tensor`）。

各 rank 算出自己 (token 段, head 子集) 的注意力输出：

```
rank0: (t0,t1,t6,t7 | H0) 的 attn 输出   rank1: (t0,t1,t6,t7 | H1)
rank2: (t2,t3,t4,t5 | H0) 的 attn 输出   rank3: (t2,t3,t4,t5 | H1)
```

### 8.5 输出投影 o_proj（需 attention-TP all-reduce）

`o_proj` 是**行并行**：每个 rank 用只含本 head 子集的权重行块 $W_O^{(h)}$，得到一份 **partial** 的 hidden 输出：

$$
Y_r^{\text{partial}} = O_r W_O^{(h)}
$$

因为 $Y = \sum_h O^{(h)} W_O^{(h)}$，必须在 **head-TP 组（行方向）** 内做 all-reduce 把各 head 的贡献加起来：

```
head-TP 组 {rank0, rank1}：all-reduce → 得到 P0 段 token 的完整 hidden
head-TP 组 {rank2, rank3}：all-reduce → 得到 P1 段 token 的完整 hidden
```

> ⚠️ 这一步就是与 `attn_tp=1` 版本的关键区别：`attn_tp=2` 时 `o_proj` 产生 partial 输出，必须经 attention-TP all-reduce 才能进入后续 dense/FFN。这也正是 DSA CP 强制 `moe_dense_tp_size=1`（dense 部分不再叠加额外 TP，避免与该规约路径冲突）的背景。all-reduce 后 token 仍沿 CP 分片（P0 在 {rank0,rank1}，P1 在 {rank2,rank3}）。

### 8.6 衔接 MoE

后续 MoE 阶段（本配置 `ep_size = tp_size = 8`、`deepep` 后端）：

- 每个 rank 手里分片的 token 按 router 结果，通过 all-to-all 发给对应专家所在 rank；
- 专家计算完再 combine 回来；
- dense / 共享专家部分（`moe_dense_tp_size=1`）按 token 复制计算，直接消费上一步 all-reduce 后的 hidden。

### 8.7 全景图

```
DP 组 0：序列 A (t0..t7)                       DP 组 1：序列 B（对称，略）
   │ zigzag 切 token 段 + head-TP 切 head
   │      P0={t0,t1,t6,t7}   P1={t2,t3,t4,t5}
   ├── rank0=(P0,H0)  rank1=(P0,H1)
   └── rank2=(P1,H0)  rank3=(P1,H1)
        │
        │ ① QKV 投影（列并行，本地无通信）
        │ ② KV all-gather —— 沿 CP 组（列）：{rank0,rank2}、{rank1,rank3}
        │ ③ 本段 Q × 该 head 的完整 KV → partial attn 输出
        │ ④ o_proj（行并行）→ head-TP all-reduce —— 沿行：{rank0,rank1}、{rank2,rank3}
        ▼
   token 仍按 CP 分片（P0 在 {rank0,rank1}，P1 在 {rank2,rank3}）
        │
        ▼ MoE: all-to-all(EP=8, deepep) + 共享专家(dense_tp=1)
        ▼
   按 cp_reverse_index 还原原始 token 顺序
```

> 注：具体切分与通信细节以 SGLang 实际 DSA CP 实现为准（`cp_utils.py`、`parallel_state.py` 的 `_ATTN_CP`）；这里给出的是符合其约束的执行模型。CP 组与 head-TP 组的具体 rank 编号排布取决于 `attn_tp`/`attn_dp`/`attn_cp` 的实际布局约定。

---

## 9. SGLang 中的实现

### 9.1 CP 切分与元数据：`python/sglang/srt/layers/utils/cp_utils.py`

- `ContextParallelMetadata`：记录 `split_list`、`zigzag_index`、`cp_reverse_index`、各 rank 实际 token 数等切分方案。
- `can_cp_split(seq_len, cp_size, forward_batch)`：判断能否做 CP 切分（要求 `seq_len // (cp_size*2) != 0`、处于 prefill/extend、非 MIXED 等）。
- `prepare_context_parallel_metadata(...)`：构建上述元数据（含 radix-cache 前缀偏移）。
- `cp_split_and_rebuild_data` / `cp_split_and_rebuild_position`：按 zigzag 重排 hidden / position。
- `cp_all_gather_reorganized_into_tensor(...)`：attention 阶段对 KV / hidden 做 padding + all-gather + 去 padding 重组。

### 9.2 CP 通信组：`python/sglang/srt/distributed/parallel_state.py`

`_ATTN_CP` group 的初始化（按 `attn_tp` / `attn_dp` 布局切分 CP group）。相关访问器在 `layers/dp_attention.py`：`get_attention_cp_group / rank / size`、`attn_cp_all_gather_into_tensor`。

### 9.3 attention 后端支持

`flashattention_backend.py` 等在 `forward_batch.forward_mode.is_context_parallel_extend()` 且 `attn_cp_size > 1` 时走 CP 路径（分块 attention + KV all-gather）。MLA / DSA 有专门实现（`layers/attention/dsa/`、`deepseek_nextn.py` 的 prefill CP 分支）。

### 9.4 启动参数：`python/sglang/srt/server_args.py`

```bash
python -m sglang.launch_server --model <long-context-MLA-model> \
    --tp-size 8 --enable-prefill-cp --cp-strategy zigzag
```

- `--enable-prefill-cp`：开启 prefill 阶段上下文并行；
- `--cp-strategy {zigzag, interleave}`：切分策略（`zigzag` 即原 `in-seq-split`，`interleave` 即原 `round-robin-split`）；
- `--enable-dsa-prefill-context-parallel` / `--enable-nsa-prefill-context-parallel`：已废弃别名，统一用 `--enable-prefill-cp`。

---

## 10. 局限与权衡

1. **主要用于 prefill**：SGLang 当前 CP 面向 prefill 长序列（`enable_prefill_cp`），decode 阶段每步只生成 1 个 token，序列维切分收益有限。
2. **单机限制**：`tp_size <= 8`，跨机 CP 有精度问题且 all-gather 带宽不足（`server_args.py` 断言）。
3. **KV all-gather 开销**：通信量随 KV 大小增长，长序列时成为主要成本；MLA 的压缩 KV 能缓解。
4. **负载均衡依赖切分策略**：连续切分会因 causal 严重失衡，必须用 zigzag / interleave；序列需能切成 `2*cp_size` 块，太短的请求会回退到非 CP。
5. **实验特性**：CP 仍在实验阶段，主要在 Hopper 平台验证（`server_args.py` 的 warning）。

---

## 参考与延伸

- 同目录：`TP.md`（张量并行）、`SP.md`（序列并行，与 CP 的对比见其 §7.1）、`DP.md`（数据并行）、`PP.md`（流水线并行）、`EP.md`（专家并行）、`DP_attention.md`。
- SGLang 代码：`python/sglang/srt/layers/utils/cp_utils.py`、`python/sglang/srt/distributed/parallel_state.py`（`_ATTN_CP`）、`python/sglang/srt/layers/dp_attention.py`。
- 相关方法：Ring Attention / zigzag context parallelism（负载均衡的 causal CP）。

