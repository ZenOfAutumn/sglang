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

---

## 3. rank 布局与关键公式

DP Attention 复用同一个 TP group（大小 `tp_size`），在其内部划分出 DP 维度。核心公式（`dp_attention.py:245-259` `compute_dp_attention_world_info`）：

```python
attn_dp_size = dp_size if enable_dp_attention else 1
attn_tp_size = tp_size // attn_dp_size // attn_cp_size
attn_tp_rank = tp_rank % attn_tp_size
attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)   # 开启 DP attention 时
```

含义：

- `attn_dp_size`：attention 被切成几个 DP 组；
- `attn_tp_size` = `tp_size / attn_dp_size / attn_cp_size`：**每个 DP 组内有几张卡做 attention TP**（即"几卡组成一个 attention DP 单元"）；
- rank 布局为 `(dp, cp, tp)`，tp 是最快变化维：
  ```
  tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size + attn_tp_rank
  ```

### 三种典型配置（设 `attn_cp_size=1`）

| tp_size | attn_dp_size | attn_tp_size | 含义 |
| --- | --- | --- | --- |
| 8 | 8 | 1 | 8 个 DP 组，每组 1 卡 → attention **纯 DP，完全不走 TP** |
| 8 | 2 | 4 | 2 个 DP 组，每组 4 卡做 attention TP → **DP + 小 TP** |
| 8 | 1 | 8 | 退化为普通 TP（未开 DP attention） |

> 所以"attention 不走 TP"只有在 `attn_tp_size == 1`（即 `dp_size == tp_size`）时才成立；否则是"TP 度变小 + 叠加 DP"。

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

- 同目录：`TP.md`（张量并行，DP attention 的基础）、`../MLA.md`（MLA 与 KV Cache 压缩）。
- SGLang 代码：`python/sglang/srt/layers/dp_attention.py`、`python/sglang/srt/managers/data_parallel_controller.py`。
- 相关文档：`python/sglang/srt/managers/README_data_parallel_controller_zh.md`。

