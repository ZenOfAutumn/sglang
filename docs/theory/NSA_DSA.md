# 原生稀疏注意力（Native Sparse Attention, NSA / DSA）原理详解

> 本文介绍可训练稀疏注意力的数学原理：从 DeepSeek 原始 NSA 论文的「压缩 + 选择 + 滑窗」
> 三分支设计，到 DeepSeek-V3.2 实际采用的 DSA（DeepSeek Sparse Attention）「闪电索引器 +
> top-k 选择」单分支方案。并详解 SGLang 中 DSA 的索引器、top-k 选择、page table 集成与
> attention backend 实现。
>
> **重要澄清**：SGLang 代码库中 `nsa/` 目录实现的是 **DeepSeek-V3.2 的 DSA**，
> 即只有「选择性 top-k」一条稀疏分支，**不包含**原始 NSA 论文的压缩分支与滑窗分支。
> 下文先讲原始 NSA 概念，再聚焦 SGLang 真正实现的 DSA。

## 目录

1. [为什么需要可训练稀疏注意力](#1-为什么需要可训练稀疏注意力)
2. [原始 NSA 的三分支设计](#2-原始-nsa-的三分支设计)
3. [DeepSeek-V3.2 DSA：闪电索引器 + top-k](#3-deepseek-v32-dsa闪电索引器--top-k)
4. [复杂度与显存收益分析](#4-复杂度与显存收益分析)
5. [SGLang 中的 DSA 实现](#5-sglang-中的-dsa-实现)
6. [支持的模型](#6-支持的模型)
7. [局限与权衡](#7-局限与权衡)

---

## 1. 为什么需要可训练稀疏注意力

标准注意力是 $O(n^2)$ 的：每个 query 都要对全部历史 key 打分。长上下文（128K+）下，
prefill 计算量与 decode 访存都急剧上升。

朴素的稀疏注意力（如固定模式的局部窗口、strided）虽能降复杂度，但有两个缺陷：

1. **模式是预先固定的**，无法学习「哪些历史 token 真正重要」。
2. **训练与推理脱节**：很多稀疏方案只在推理期近似，模型并未针对稀疏模式训练，质量受损。

「原生 / 可训练稀疏注意力」的目标是：**让稀疏选择本身可微、可学习，训练和推理用同一套
稀疏机制**，从而在大幅降低计算/访存的同时保持质量。NSA 与 DSA 都属于这一思路。

---

## 2. 原始 NSA 的三分支设计

DeepSeek 的 NSA 论文提出把注意力拆成三条互补分支，再用一个**可学习的门控（gate）**
加权融合：

1. **压缩分支（compressed / coarse）**：把历史 KV 按块聚合成粗粒度的「压缩 token」，
   query 先对这些压缩块做注意力，获得全局概览。
2. **选择分支（selected / fine-grained）**：基于压缩分支的打分，挑出 top-k 个最重要的
   块，对这些块内的原始 token 做细粒度注意力。
3. **滑窗分支（sliding window）**：固定关注最近的若干 token，保证局部连续性
   （类似 `sliding_window_attention_zh.md` 中的 SWA）。

三条分支输出经门控加权求和：$o = g_c \cdot o_{cmp} + g_s \cdot o_{sel} + g_w \cdot o_{win}$。

> **注意**：这是论文设计。SGLang 当前**并未**实现这三分支——它实现的是 DeepSeek-V3.2
> 的 DSA 变体（见下节）。在 `nsa/` 目录中检索 `compress`/`coarse`/`window` 均无命中，
> 唯一的稀疏分支是 top-k 选择。

---

## 3. DeepSeek-V3.2 DSA：闪电索引器 + top-k

DeepSeek-V3.2 简化为单分支的 **DSA（DeepSeek Sparse Attention）**，核心是一个轻量的
**闪电索引器（lightning indexer）**：

1. **索引打分**：用一组低维（`index_head_dim`，约 128）的 Query/Key 投影，对全部历史
   位置算一个廉价的 FP8 打分 $s_{ij}$，衡量「第 $j$ 个历史 token 对当前 query 的重要度」。
   这个打分头数少、维度低、用 FP8，故称「闪电」。
2. **top-k 选择**：对每个 query，取打分最高的 `index_topk` 个历史位置。
3. **稀疏注意力**：仅在这 top-k 个位置上做完整精度的注意力（基于 MLA 的潜在向量）。

与原始 NSA 的差异：DSA 去掉了压缩分支与滑窗分支，只保留「可学习的 top-k 选择」，
并把它与 MLA 的低秩 KV 缓存深度耦合。索引器中的 per-head 门控用于打分（而非三分支融合）：

```python
# python/sglang/srt/layers/attention/nsa/nsa_indexer.py:267
def _get_logits_head_gate(self, x: torch.Tensor, q_scale: torch.Tensor):
    weights = self._weights_proj_bf16_in_fp32_out(x)
    weights = weights * self.n_heads**-0.5
    weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
```

---

## 4. 复杂度与显存收益分析

设序列长 $n$、选择数 $k = \text{index\_topk}$（$k \ll n$）。

| 阶段 | 标准（MLA）注意力 | DSA |
|------|-----------------|-----|
| 索引打分 | — | $O(n)$ 廉价 FP8 打分 |
| 注意力计算 | $O(n)$ 个 key/query | $O(k)$ 个 key/query |

decode 时每个 query 原本要对全部 $n$ 个历史潜在向量做注意力，DSA 把它降到固定的 $k$ 个。
当上下文极长（$n \gg k$）时，注意力部分近似变为**常数访存**，是长上下文 decode 的关键加速。
代价是多了一个 $O(n)$ 的索引打分，但因其低维 + FP8，远比全精度注意力廉价。

显存上，DSA 复用 MLA 的潜在 KV cache，并额外维护一份 FP8 的索引器 KV cache
（用于打分），整体仍远小于稠密 MHA。

---

## 5. SGLang 中的 DSA 实现

`python/sglang/srt/layers/attention/nsa/` 目录包含索引器与多套 kernel：
`nsa_indexer.py`（核心打分/选择）、`quant_k_cache.py` / `dequant_k_cache.py`
（FP8 索引器 KV）、`transform_index.py`、`triton_kernel.py`、`tilelang_kernel.py` 等；
上一层的 `nsa_backend.py` 是 backend 总入口。

### 5.1 模型识别：何时启用 DSA

`is_deepseek_nsa` 判定：模型架构在白名单内 **且** 配置里设置了 `index_topk`：

```python
# python/sglang/srt/configs/model_config.py:55
def is_deepseek_nsa(config) -> bool:
    ...
    return (
        architectures is not None
        and architectures[0]
        in [
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "DeepseekV3ForCausalLMNextN",
            "MistralLarge3ForCausalLM",
            "PixtralForConditionalGeneration",
            "GlmMoeDsaForCausalLM",
        ]
        and index_topk is not None
    )
```

相关超参由 `get_nsa_index_topk`（`:86`）、`get_nsa_index_head_dim`（`:81`）、
`get_nsa_index_n_heads`（`:91`）读取。

### 5.2 索引器：打分与 top-k 选择

`Indexer`（`nsa_indexer.py:149`）用低维投影 + FP8 MQA 打分 kernel 对历史位置打分，
再取 top-k。ragged（变长 prefill）路径的核心：

```python
# python/sglang/srt/layers/attention/nsa/nsa_indexer.py:944
index_score = fp8_index(q_fp8_partial, weights_partial, k_fp8, k_scale)
end_pos = seq_len
topk_indices = index_score.topk(min(topk, end_pos), dim=-1)[1].squeeze(0)
```

paged（decode）路径 `_get_topk_paged`（`:384`）调用 `deep_gemm.fp8_paged_mqa_logits`
打分，再经 `metadata.topk_transform(logits, self.index_topk)`（`:484`）选择。
PAGED / RAGGED 两种 `topk_transform` 在 `nsa_backend.py:212` 处分派。

### 5.3 page table 集成：把 top-k 当成 page_size=1 的页表

索引器读取请求的 page table，按页步进取 KV：

```python
# python/sglang/srt/layers/attention/nsa/nsa_indexer.py:905
block_tables = forward_batch.req_to_token_pool.req_to_token[
    forward_batch.req_pool_indices, :
]
strided_indices = torch.arange(0, block_tables.shape[-1], page_size, device="cuda")
block_tables = block_tables[:, strided_indices] // page_size
```

选出的 `topk_indices` 随后被当作**页大小为 1** 的页表交给稀疏注意力 kernel
（`nsa_backend.py:1353` 注释「here, we use page size = 1」，`page_table_1 = topk_indices`
在 `:1358`/`:1414`）。索引器的 KV 单独以 FP8 缓存（`quant_k_cache.py`）。

### 5.4 backend 前向流程

`NativeSparseAttnBackend`（`nsa_backend.py:284`）在 `init_forward_metadata`（`:386`）
构建页表与按 topk 裁剪的 `nsa_cache_seqlens_int32`。`forward_extend`（`:1255`）与
`forward_decode`（`:1465`）接收上游算好的 `topk_indices`，pad 后构建 page_size=1 页表，
再分派到具体 kernel 实现：

| 实现 | 入口 |
|------|------|
| FA3 | `_forward_fa3`（`nsa_backend.py:1613`） |
| FlashMLA 稀疏 | `_forward_flashmla_sparse`（`:1651`） |
| FlashMLA KV | `_forward_flashmla_kv`（`:1700`） |
| 稠密 MHA 回退 | `_forward_standard_mha`（`:1745`） |
| TileLang | `_forward_tilelang`（`:1810`） |
| aiter / trtllm | `_forward_aiter`（`:1828`）/ `_forward_trtllm`（`:1917`） |

可选 backend 由启动参数 `nsa_prefill_backend` / `nsa_decode_backend`
（`server_args.py:482`）控制，候选见 `NSA_CHOICES`（`server_args.py:169`）：
`flashmla_sparse / flashmla_kv / flashmla_auto / fa3 / tilelang / aiter / trtllm`。
DSA 模型会被自动设为 `attention_backend = "nsa"`（`server_args.py:1533`）。

### 5.5 数据流总览

```
请求到达（基于 MLA 的潜在 KV cache + FP8 索引器 KV cache）
  │
  ▼
闪电索引器 Indexer.forward          # 低维 FP8 打分全部历史位置
  │   └─ fp8_index / fp8_paged_mqa_logits
  ▼
top-k 选择                          # 每 query 取 index_topk 个最重要位置
  │   └─ topk_transform → topk_indices
  ▼
把 topk_indices 作为 page_size=1 页表
  │
  ▼
稀疏注意力 kernel（FA3 / FlashMLA / TileLang ...）
  │   └─ 仅在 k 个选中位置上做 MLA 注意力
  ▼
输出
```

---

## 6. 支持的模型

由 `is_deepseek_nsa` 白名单可知，当前支持 DSA 的模型为：

- **DeepSeek-V3 / V3.2**（及其 NextN/MTP 变体）。
- **GLM-MoE-DSA**（`GlmMoeDsaForCausalLM`）。
- 白名单还含 `MistralLarge3ForCausalLM`、`PixtralForConditionalGeneration`
  （需配置中带 `index_topk` 才实际启用）。

这些模型本身都是 MLA 模型——DSA 是叠加在 MLA 之上的稀疏选择层，二者强耦合。

---

## 7. 局限与权衡

- **与论文 NSA 不同**：SGLang 实现的是 DeepSeek-V3.2 DSA 单分支，**没有**压缩分支和
  滑窗分支。若按原始 NSA 论文理解三分支门控融合，会与代码不符。
- **强依赖 MLA**：DSA 的稀疏注意力建立在 MLA 潜在向量之上，无法独立用于普通 MHA/GQA 模型。
- **索引打分开销**：top-k 选择需要 $O(n)$ 的索引打分与排序；上下文不够长时（$n$ 与 $k$
  接近），稀疏收益被索引开销抵消，未必快于稠密注意力（故保留 `_forward_standard_mha`
  稠密回退）。
- **页大小约束**：索引器假设 `page_size == 64`（`nsa_indexer.py:894`），而稀疏注意力侧
  又把选择结果当作 page_size=1 处理，两套页语义并存，实现复杂、对元数据正确性要求高。
- **必须为 DSA 训练**：索引器是可学习模块，需在训练期联合优化，不能免训练加到现成模型上。

---

## 参考实现位置

| 模块 | 文件路径 |
|------|---------|
| 模型识别 / 超参 | `python/sglang/srt/configs/model_config.py` |
| 闪电索引器（打分 + top-k） | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| backend 总入口与前向 | `python/sglang/srt/layers/attention/nsa_backend.py` |
| FP8 索引器 KV 量化 | `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` |
| top-k 索引变换 | `python/sglang/srt/layers/attention/nsa/transform_index.py` |
| NSA KV 池 | `python/sglang/srt/mem_cache/memory_pool.py`（`NSATokenToKVPool`） |
| 启动参数 | `python/sglang/srt/server_args.py` |

---

## 附录 A：DSA 数值计算示例

为直观理解 §3 的「闪电索引器 → top-k 选择 → 稀疏注意力」三步，下面用一组**极小维度**的
具体数字，演示一个 query 如何只在 top-k 个历史位置上做注意力，并与稠密注意力对比。

### A.0 设定

- 历史序列长度 $n = 6$（位置 $0 \dots 5$），当前 decode 一个新 query
- top-k 选择数 $k = \text{index\_topk} = 2$
- 索引器低维 $d_{idx} = 2$；注意力头维 $d = 2$，单头

### A.1 第一步：闪电索引器打分（§3）

索引器用**低维廉价**投影得到 query 的索引向量 $q^{idx}$ 与各历史位置的索引向量 $k^{idx}_j$
（实际为 FP8，这里用普通小数演示）：

$$
q^{idx} = [1,\ 1]
$$

| 位置 $j$ | $k^{idx}_j$ | 索引打分 $s_j = q^{idx}\cdot k^{idx}_j$ |
|---------|-------------|------|
| 0 | $[0.1,\ 0.0]$ | $0.1$ |
| 1 | $[0.9,\ 0.8]$ | $1.7$ |
| 2 | $[0.2,\ 0.1]$ | $0.3$ |
| 3 | $[1.0,\ 1.0]$ | $2.0$ |
| 4 | $[0.0,\ 0.3]$ | $0.3$ |
| 5 | $[0.4,\ 0.2]$ | $0.6$ |

这一步是 $O(n)$ 的：对全部 6 个位置各算一次低维内积，廉价。

### A.2 第二步：top-k 选择（§3）

取打分最高的 $k=2$ 个位置：

$$
\text{排序后} : s_3=2.0 > s_1=1.7 > s_5=0.6 > \dots
$$

$$
\Rightarrow \text{topk\_indices} = \{3,\ 1\}
$$

其余位置 $\{0,2,4,5\}$ 被**丢弃**，不参与后续注意力。这组索引随后被当作
**page_size=1 的页表**交给稀疏注意力 kernel（见 §5.3）。

### A.3 第三步：仅在 top-k 位置做完整注意力

只对选中的位置 $\{1, 3\}$ 取出**全精度**的 Q/K/V（基于 MLA 潜在向量解出），做标准注意力。
设 query $q = [1,\ 0]$，选中位置的 K/V：

| 位置 $j$ | $k_j$ | $v_j$ |
|---------|-------|-------|
| 1 | $[1,\ 0]$ | $[10,\ 0]$ |
| 3 | $[0,\ 1]$ | $[0,\ 20]$ |

打分（$\sqrt{d}=\sqrt2$）：

$$
s'_1 = \frac{[1,0]\cdot[1,0]}{\sqrt2} = \frac{1}{\sqrt2}\approx0.707,\qquad
s'_3 = \frac{[1,0]\cdot[0,1]}{\sqrt2} = 0
$$

softmax：

$$
\text{权重} = \text{softmax}([0.707,\ 0]) \approx [0.670,\ 0.330]
$$

输出：

$$
\text{out} \approx 0.670\cdot[10,0] + 0.330\cdot[0,20] = [6.70,\ 6.60]
$$

注意 softmax 的归一化**只在选中的 2 个位置上做**，这是稀疏注意力与稠密的关键区别。

### A.4 对比稠密注意力

稠密注意力要对全部 6 个位置算 K/V 内积、softmax、加权——既要访存 6 份 KV，
也要算 6 个打分。DSA 把它降到固定的 $k=2$ 份：

| 指标 | 稠密（MLA） | DSA（$k=2$） |
|------|-----------|--------------|
| 索引打分次数 | 0 | 6（低维 FP8，廉价） |
| 全精度注意力的 key 数 | 6 | 2 |

当上下文极长（如 $n=128\text{K}$、$k=2048$）时：

- 全精度注意力的 key 数从 $n$ 降到固定的 $k$，decode 访存近似变为**常数**；
- 多出的索引打分是 $O(n)$ 但低维 + FP8，远比全精度注意力廉价。

这正是 §4 复杂度表的数值体现：DSA 用一次廉价的 $O(n)$ 打分，换取把昂贵的注意力
从 $O(n)$ 压到 $O(k)$。

> **注意**：本例的索引打分（A.1）与最终注意力打分（A.3）用的是**两套不同的 Q/K**——
> 前者是索引器的低维 FP8 投影（只为「选谁」），后者是 MLA 的全精度潜在向量（真正算注意力）。
> 这与 §3「索引器轻量打分、选中后才用完整精度」的描述一致。
