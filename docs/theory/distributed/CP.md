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
8. [SGLang 中的实现](#8-sglang-中的实现)
9. [局限与权衡](#9-局限与权衡)

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

## 8. SGLang 中的实现

### 8.1 CP 切分与元数据：`python/sglang/srt/layers/utils/cp_utils.py`

- `ContextParallelMetadata`：记录 `split_list`、`zigzag_index`、`cp_reverse_index`、各 rank 实际 token 数等切分方案。
- `can_cp_split(seq_len, cp_size, forward_batch)`：判断能否做 CP 切分（要求 `seq_len // (cp_size*2) != 0`、处于 prefill/extend、非 MIXED 等）。
- `prepare_context_parallel_metadata(...)`：构建上述元数据（含 radix-cache 前缀偏移）。
- `cp_split_and_rebuild_data` / `cp_split_and_rebuild_position`：按 zigzag 重排 hidden / position。
- `cp_all_gather_reorganized_into_tensor(...)`：attention 阶段对 KV / hidden 做 padding + all-gather + 去 padding 重组。

### 8.2 CP 通信组：`python/sglang/srt/distributed/parallel_state.py`

`_ATTN_CP` group 的初始化（按 `attn_tp` / `attn_dp` 布局切分 CP group）。相关访问器在 `layers/dp_attention.py`：`get_attention_cp_group / rank / size`、`attn_cp_all_gather_into_tensor`。

### 8.3 attention 后端支持

`flashattention_backend.py` 等在 `forward_batch.forward_mode.is_context_parallel_extend()` 且 `attn_cp_size > 1` 时走 CP 路径（分块 attention + KV all-gather）。MLA / DSA 有专门实现（`layers/attention/dsa/`、`deepseek_nextn.py` 的 prefill CP 分支）。

### 8.4 启动参数：`python/sglang/srt/server_args.py`

```bash
python -m sglang.launch_server --model <long-context-MLA-model> \
    --tp-size 8 --enable-prefill-cp --cp-strategy zigzag
```

- `--enable-prefill-cp`：开启 prefill 阶段上下文并行；
- `--cp-strategy {zigzag, interleave}`：切分策略（`zigzag` 即原 `in-seq-split`，`interleave` 即原 `round-robin-split`）；
- `--enable-dsa-prefill-context-parallel` / `--enable-nsa-prefill-context-parallel`：已废弃别名，统一用 `--enable-prefill-cp`。

---

## 9. 局限与权衡

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

