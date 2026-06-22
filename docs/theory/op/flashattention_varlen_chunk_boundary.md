# FlashAttention Varlen 模式的分块边界计算原理与实现

本文说明 FlashAttention 在 **变长序列（variable-length，简称 varlen）模式** 下，如何用「累积序列长度（cumulative sequence lengths，`cu_seqlens`）」来表达每个序列在拼接大张量里的**分块边界**，以及 SGLang 在 `flashattention_backend.py` 中是如何计算这些边界的。文末给出可手算的数值示例。

## 1. 它解决什么问题

一个推理 batch 里，不同请求的序列长度往往**互不相同**（如请求 A 有 3 个 query token、请求 B 有 5 个）。如果硬要把它们 pad 到同一长度再堆成规整的 `(batch, max_len, ...)` 张量，会产生两类浪费：

- **计算浪费**：attention 要对 pad 出来的无效位置做无用计算（或额外维护 mask）。
- **显存浪费**：短序列被 pad 到 `max_len`，显存按最长序列计费。

varlen 模式的做法是：把一个 batch 内所有序列的 token **首尾相接拼成一根扁平的一维张量**（packed / ragged 布局），形状为 `(total_tokens, num_heads, head_dim)`，其中 `total_tokens = Σ seqlen_i`。这样没有任何 pad，但带来一个新问题——**kernel 怎么知道每个序列从哪里开始、到哪里结束？** 这正是 `cu_seqlens`（分块边界）要回答的。

## 2. 核心原理：用前缀和表达分块边界

### 2.1 `cu_seqlens` 的定义

设 batch 内各序列长度为 $[s_0, s_1, \dots, s_{B-1}]$（$B$ 为 batch size）。则累积序列长度数组定义为：

$$
\texttt{cu\_seqlens}[i] = \sum_{j=0}^{i-1} s_j, \quad i = 0, 1, \dots, B
$$

即它是序列长度的**前缀和（prefix sum）**，并在最前面补一个 0。长度为 $B+1$。

- `cu_seqlens[0] = 0`（第一个序列从偏移 0 开始）；
- `cu_seqlens[i]` = 第 $i$ 个序列在扁平张量里的**起始偏移**；
- `cu_seqlens[i+1]` = 第 $i$ 个序列的**结束偏移（不含）**；
- 第 $i$ 个序列占据扁平张量的左闭右开区间 `[cu_seqlens[i], cu_seqlens[i+1])`，长度恰为 $s_i$；
- `cu_seqlens[B] = total_tokens` = 全部 token 总数。

**关键洞察**：相邻两个边界之差就是序列长度，`cu_seqlens[i+1] - cu_seqlens[i] = s_i`。kernel 内部为每个序列启动一个（或一组）thread block，用 `cu_seqlens` 这对边界**切出**自己负责的那段连续内存，无需 pad、无需显式 mask。

### 2.2 为什么要有 Q 和 K 两套边界

attention 的 query 和 key/value 序列长度在很多场景下**并不相等**，所以 FlashAttention varlen 需要**两套**独立的边界：

- `cu_seqlens_q`：query 侧分块边界，由每个序列**本次要计算的 query token 数**累加而来；
- `cu_seqlens_k`：key/value 侧分块边界，由每个序列**能被注意到的 KV 总长度**累加而来。

两者分别配套 `max_seqlen_q` / `max_seqlen_k`（各序列长度的最大值，kernel 用于确定 tile 循环上界）。

典型场景对应关系：

| 场景 | query 长度 $s^q_i$ | KV 长度 $s^k_i$ | 说明 |
| --- | --- | --- | --- |
| **首次 Prefill** | 整个 prompt 长度 | = query 长度 | Q/K 等长，`cu_seqlens_q == cu_seqlens_k` |
| **Chunked Prefill / 带前缀复用** | 本 chunk 的新 token 数 `extend_seq_len` | 前缀 + 本 chunk = 完整 `seq_len` | Q < K，两套边界不同 |
| **Decode** | 每序列 1 个 token | 历史全部 KV | `cu_seqlens_q` 是 `0,1,2,...,B` 的等差数列 |
| **投机解码 Draft Decode** | 每序列 `topk` 个 | 历史 KV（+若干步） | `cu_seqlens_q` 步长为 `topk` |

这就是"分块边界计算方案"的核心：**针对不同 forward mode，分别构造 query 与 KV 的累积长度数组**。

## 3. 在 SGLang 中的实现

SGLang 在 `python/sglang/srt/layers/attention/flashattention_backend.py` 的 `init_forward_metadata` 中，按 forward mode 计算这两套边界。核心手法只有两招：**`torch.cumsum`（前缀和）+ `torch.nn.functional.pad((1,0))`（在最前面补 0）**。

### 3.1 Extend / Chunked Prefill：Q 与 K 边界分开算

```648:659:python/sglang/srt/layers/attention/flashattention_backend.py
            if (
                any(forward_batch.extend_prefix_lens_cpu)
                or forward_batch.forward_mode.is_draft_extend_v2()
            ):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k
```

- **有前缀复用时**（`extend_prefix_lens` 非全 0，即 chunked prefill 或 RadixCache 命中前缀）：`cu_seqlens_q` 按 **本次新增 token 数 `extend_seq_lens`** 做前缀和——这就是 query 侧的分块边界；而 `cu_seqlens_k` 早先已按**完整 KV 长度**算好（见下）。此时 Q < K。
- **无前缀时**（纯首次 prefill）：Q 与 K 等长，直接令 `cu_seqlens_q = cu_seqlens_k`，省一次计算。

KV 侧边界（`cu_seqlens_k`）由完整序列长度 `cache_seqlens` 累加而来，同样是 `cumsum + pad`：

```394:399:python/sglang/srt/layers/attention/flashattention_backend.py
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
```

### 3.2 Decode：Q 边界是等差数列

decode 时每个序列只算 1 个 query token，所以 query 边界退化成 `0,1,2,...,B` 的等差数列，用 `torch.arange` 直接生成（比 cumsum 更省）：

```391:399:python/sglang/srt/layers/attention/flashattention_backend.py
                    metadata.cu_seqlens_q = torch.arange(
                        0, batch_size + 1, dtype=torch.int32, device=device
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
```

### 3.3 投机解码 Draft Decode：Q 边界步长为 topk

投机解码的 draft decode 每个序列要并行验证 `topk` 个候选，query 边界步长变成 `topk`：

```421:427:python/sglang/srt/layers/attention/flashattention_backend.py
                    metadata.cu_seqlens_q = torch.arange(
                        0,
                        batch_size * self.topk + 1,
                        step=self.topk,
                        dtype=torch.int32,
                        device=device,
                    )
```

### 3.4 `pad((1, 0))` 的作用

`torch.cumsum` 给出的是「含当前项」的前缀和，例如 `cumsum([3,5,2]) = [3,8,10]`，缺少开头的起始偏移 0。`torch.nn.functional.pad(x, (1, 0))` 在一维张量**最前面补一个 0**，得到 `[0,3,8,10]`，正好满足 §2.1 中「长度 $B+1$、首元素为 0」的定义。这是把"前缀和"变成"分块边界"的关键一步。

## 4. 数值示例（手算）

设一个 batch 有 3 个请求，处于 **chunked prefill 带前缀复用** 场景：

| 请求 | 已缓存前缀 `prefix_len` | 本次新 token `extend_seq_len` | 完整 KV 长度 `seq_len` |
| --- | --- | --- | --- |
| req0 | 4 | 3 | 7 |
| req1 | 0 | 5 | 5 |
| req2 | 6 | 2 | 8 |

### 4.1 计算 query 侧边界 `cu_seqlens_q`

按 `extend_seq_lens = [3, 5, 2]` 计算：

```
cumsum([3, 5, 2])      = [3, 8, 10]
pad((1,0)) → 前面补 0   = [0, 3, 8, 10]
cu_seqlens_q           = [0, 3, 8, 10]
max_seqlen_q           = max(3, 5, 2) = 5
```

含义：扁平的 query 张量共 `total_q = 10` 个 token；

- req0 的 query 占 `[0, 3)`（3 个）；
- req1 的 query 占 `[3, 8)`（5 个）；
- req2 的 query 占 `[8, 10)`（2 个）。

### 4.2 计算 KV 侧边界 `cu_seqlens_k`

按完整 `seq_lens = [7, 5, 8]`（= prefix + extend）计算：

```
cumsum([7, 5, 8])      = [7, 12, 20]
pad((1,0)) → 前面补 0   = [0, 7, 12, 20]
cu_seqlens_k           = [0, 7, 12, 20]
max_seqlen_k           = max(7, 5, 8) = 8
```

含义：扁平的 KV 张量共 `total_k = 20` 个 token；

- req0 的 KV 占 `[0, 7)`（7 个 = 4 前缀 + 3 新）；
- req1 的 KV 占 `[7, 12)`（5 个）；
- req2 的 KV 占 `[12, 20)`（8 个 = 6 前缀 + 2 新）。

### 4.3 kernel 如何使用这两套边界

对第 $i$ 个请求，kernel 取出：

- query 段：`Q[cu_seqlens_q[i] : cu_seqlens_q[i+1]]`，长度 $s^q_i$；
- KV 段：`K/V[cu_seqlens_k[i] : cu_seqlens_k[i+1]]`，长度 $s^k_i$；
- 用 $s^q_i \times s^k_i$ 的注意力计算，配合 causal mask（query 的第 $t$ 个位置只能看到 KV 的前 `prefix_len + t + 1` 个）。

以 req0 为例：3 个新 query token 对 7 个 KV（4 个前缀 + 3 个本次）做带因果掩码的注意力——这正是 chunked prefill「新 token 注意到全部历史 + 自身」的语义。

### 4.4 对比 Decode 场景

若同样 3 个请求处于 decode（各出 1 个 token，历史 KV 分别为 7、5、8）：

```
cu_seqlens_q = arange(0, 3+1) = [0, 1, 2, 3]      （等差，每序列 1 个 query）
cu_seqlens_k = pad(cumsum([7,5,8])) = [0, 7, 12, 20]
max_seqlen_q = 1,  max_seqlen_k = 8
```

query 总数仅 3，每个请求 1 个 query token 注意到自己的全部历史 KV。

## 5. 小结

FlashAttention varlen 模式把变长 batch **无 pad 地拼成扁平张量**，用一对**累积序列长度数组** `cu_seqlens_q` / `cu_seqlens_k` 标记每个序列的分块边界：

- **本质**：边界 = 序列长度的**前缀和**，相邻边界之差即序列长度，区间 `[cu_seqlens[i], cu_seqlens[i+1])` 切出第 $i$ 个序列。
- **实现**：SGLang 用 `torch.cumsum + pad((1,0))` 两步构造；decode/draft 等长场景退化为 `torch.arange` 直接生成。
- **Q/K 分离**：query 边界按「本次计算的 token 数」算，KV 边界按「完整可见 KV 长度」算；首次 prefill 两者相等，chunked prefill / 前缀复用时 Q < K。
- **收益**：免去 pad 的计算与显存浪费，让一个融合 kernel 高效处理整个变长 batch。

