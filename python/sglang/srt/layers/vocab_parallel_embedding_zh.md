# VocabParallelEmbedding / ParallelLMHead 原理详解

> 代码位置：`python/sglang/srt/layers/vocab_parallel_embedding.py`
> 改编自 vLLM `v0.6.3.post1` 的同名模块。

本模块实现 Transformer 的**输入嵌入**与**输出词表投影（LM head）**，两者都沿**词表维**做张量并行切分。它要解决三个问题：

1. **词表切分**：词表动辄十万量级，$V \times H$ 的嵌入表是模型中最大的单个张量之一，必须切分。
2. **Padding 对齐**：$V$ 通常不能被 $p$ 整除，需要补齐；且 LoRA 新增词必须固定放在分片末尾。
3. **越界屏蔽**：切分后每个 rank 只持有部分词，输入 token id 落在别人分片里时要屏蔽掉，再靠 all-reduce 汇总。

### 符号约定

| 符号 | 含义 | 代码字段 |
| --- | --- | --- |
| $V_{\text{org}}$ | 原始词表大小（不含 LoRA） | `org_vocab_size` |
| $V$ | 总词表大小（含 LoRA 新增词） | `num_embeddings` |
| $A$ | LoRA 新增词数，$A = V - V_{\text{org}}$ | `num_added_embeddings` |
| $\tilde{V}_{\text{org}}$ | padding 后的原始词表大小 | `org_vocab_size_padded` |
| $\tilde{V}$ | padding 后的总词表大小 | `num_embeddings_padded` |
| $H$ | 隐藏维度 | `embedding_dim` |
| $T$ | 本次 forward 的 token 总数（SGLang 已将 batch 与 seq 展平为一维，故 $T = \sum_b S_b$，而非 $B \times S$） | `input_ids.numel()` |
| $P$ | padding 粒度（默认 64） | `padding_size` |
| $p$ | 张量并行度 | `tp_size` |
| $i$ | 当前 rank 编号 | `tp_rank` |

---

## 1. 为什么词表维要切分

嵌入表与 LM head 权重的形状都是 $V \times H$。以 Llama-3 为例：$V = 128256,\ H = 4096$，单个张量即：

$$
128256 \times 4096 \times 2\ \text{B} \approx 1.05\ \text{GB}
$$

输入嵌入和 LM head 各一份（若不共享权重），仅这两层就占 2 GB。切分是必须的。

**切哪一维**：$V$ 和 $H$ 都可以切，但切 $H$ 会破坏后续所有层的 hidden 维一致性，因此统一切 $V$：

$$
W = \begin{bmatrix} W^{(0)} \\ W^{(1)} \\ \vdots \\ W^{(p-1)} \end{bmatrix},
\qquad W^{(i)} \in \mathbb{R}^{\frac{\tilde{V}}{p} \times H}
$$

两条使用路径的通信模式**恰好相反**：

| 层 | 输入 | 输出 | 通信 |
| --- | --- | --- | --- |
| `VocabParallelEmbedding` | token id（全局，各 rank 相同） | $[T, H]$ 完整形状但**部分为 0** | **all-reduce** |
| `ParallelLMHead` | $[T, H]$ 完整 | $[T, \tilde{V}/p]$ 词表切片 | **all-gather** |

> 嵌入是「查表后求和」，形状完整但数值不完整 → all-reduce；
> LM head 是「投影出部分词表」，数值完整但形状不完整 → all-gather。这与行并行/列并行的二元性是一致的。

---

## 2. Padding 规则：两次对齐

`num_embeddings_padded` 的计算是**两步 padding**，不是一步：

```276:282:python/sglang/srt/layers/vocab_parallel_embedding.py
        self.org_vocab_size_padded = pad_vocab_size(
            self.org_vocab_size, self.padding_size
        )
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings, self.padding_size
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded
```

记向上对齐算子为：

$$
\mathrm{pad}_P(x) = \left\lceil \frac{x}{P} \right\rceil \cdot P
$$

则两步 padding 为：

$$
\tilde{V}_{\text{org}} = \mathrm{pad}_P\!\left(V_{\text{org}}\right),
\qquad
\tilde{V} = \mathrm{pad}_P\!\left(\tilde{V}_{\text{org}} + A\right)
$$

**关键在于先对 $V_{\text{org}}$ 单独 padding，再加 $A$**，即：

$$
\mathrm{pad}_P\!\left(\mathrm{pad}_P(V_{\text{org}}) + A\right)
\neq
\mathrm{pad}_P\!\left(V_{\text{org}} + A\right)
\quad \text{(一般情况下)}
$$

两者的本质差别在于 **base 段的结束位置是否对齐**。只有前者才能保证：

$$
\tilde{V}_{\text{org}} \equiv 0 \pmod{P}
\qquad\Longrightarrow\qquad
\frac{\tilde{V}_{\text{org}}}{p} \in \mathbb{Z}
$$

否则 base 词与 LoRA 词的边界不在 $P$ 的倍数上，两者会在同一个分片内混杂，无法各自独立切分。

### 类 docstring 的例子

$V_{\text{org}} = 1010,\ A = 16,\ P = 64$：

$$
\tilde{V}_{\text{org}}
= \mathrm{pad}_{64}(1010)
= \left\lceil \frac{1010}{64} \right\rceil \cdot 64
= \left\lceil 15.78 \right\rceil \cdot 64
= 16 \cdot 64
= 1024
$$

$$
\tilde{V}
= \mathrm{pad}_{64}(1024 + 16)
= \left\lceil \frac{1040}{64} \right\rceil \cdot 64
= \left\lceil 16.25 \right\rceil \cdot 64
= 17 \cdot 64
= 1088
$$

于是单卡（TP1）的布局为四段，各段的 index 区间为：

$$
\underbrace{[0,\ 1010)}_{\text{BASE}}
\cup
\underbrace{[1010,\ 1024)}_{\text{BASE PAD}}
\cup
\underbrace{[1024,\ 1040)}_{\text{LORA}}
\cup
\underbrace{[1040,\ 1088)}_{\text{LORA PAD}}
$$

对应到 token id（$-1$ 表示无效槽位）：

```
|<---- BASE ---->|<- BASE PAD ->|<-- LORA -->|<- LORA PAD ->|
 index 0..1009     1010..1023     1024..1039   1040..1087
 token 0..1009        -1          1010..1015       -1
```

**为什么 padding 到 64**：一是保证能被常见的 $p$（2/4/8/16/32/64）整除；二是对齐 GEMM 的 tile 尺寸，避免 LM head 矩阵乘出现零碎尾块。

### CPU 的特殊处理

```262:267:python/sglang/srt/layers/vocab_parallel_embedding.py
        if (
            _is_cpu
            and pad_vocab_size(self.org_vocab_size, padding_size) % self.tp_size != 0
        ):
            padding_size *= self.tp_size
```

即当下式不成立时：

$$
\mathrm{pad}_P\!\left(V_{\text{org}}\right) \equiv 0 \pmod{p}
$$

就把对齐粒度放大为 $P' = P \cdot p$。此时整除性自动成立：

$$
\mathrm{pad}_{P'}\!\left(V_{\text{org}}\right) = k \cdot P \cdot p
\quad\Longrightarrow\quad
\frac{\mathrm{pad}_{P'}\!\left(V_{\text{org}}\right)}{p} = k \cdot P \in \mathbb{Z}
$$

代价是浪费更多 padding 槽位（最多 $P \cdot p - 1$ 个），但避开了 $p$ 为非 2 的幂（如 $p = 48$）时无法均分的问题。

---

## 3. 四段布局与分片索引

每个 rank 的本地张量都严格保持 **BASE → BASE PAD → LORA → LORA PAD** 四段顺序。之所以要这样固定，是因为 LoRA 权重可以在运行时热加载/卸载，把它放在末尾才能在不动 base 部分的前提下增删。

`VocabParallelEmbeddingShardIndices` 用 8 个索引描述这个布局（4 个 padded 边界 + 4 个真实边界）：

```83:91:python/sglang/srt/layers/vocab_parallel_embedding.py
    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int

    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int
```

`__post_init__` 里有一整组断言校验这 8 个索引的偏序关系（真实边界必须被 padded 边界包住），是排查布局问题的第一道防线。

### 索引计算

base 段与 added 段**各自独立**按 $p$ 均分：

$$
\left[\text{padded\_org\_start}_i,\ \text{padded\_org\_end}_i\right)
= \left[\ i \cdot \frac{\tilde{V}_{\text{org}}}{p},\ (i+1) \cdot \frac{\tilde{V}_{\text{org}}}{p}\ \right)
$$

$$
\left[\text{padded\_added\_start}_i,\ \text{padded\_added\_end}_i\right)
= \left[\ V_{\text{org}} + i \cdot \frac{\tilde{A}}{p},\ V_{\text{org}} + (i+1) \cdot \frac{\tilde{A}}{p}\ \right),
\qquad \tilde{A} \triangleq \tilde{V} - \tilde{V}_{\text{org}}
$$

注意 added 段带了 `offset=org_vocab_size`——它是以**真实的** $V_{\text{org}}$（而非 $\tilde{V}_{\text{org}}$）为起点编号的，因为 LoRA token 的实际 id 从 $V_{\text{org}}$ 开始。

然后用 `min` 裁掉 padding 部分，得到真实边界：

```366:370:python/sglang/srt/layers/vocab_parallel_embedding.py
        # remove padding
        org_vocab_start_index = min(padded_org_vocab_start_index, org_vocab_size)
        org_vocab_end_index = min(padded_org_vocab_end_index, org_vocab_size)
        added_vocab_start_index = min(padded_added_vocab_start_index, vocab_size)
        added_vocab_end_index = min(padded_added_vocab_end_index, vocab_size)
```

即对 base 段用 $V_{\text{org}}$ 截断、对 added 段用 $V$ 截断：

$$
\text{org\_start}_i = \min\!\left(\text{padded\_org\_start}_i,\ V_{\text{org}}\right),
\qquad
\text{org\_end}_i = \min\!\left(\text{padded\_org\_end}_i,\ V_{\text{org}}\right)
$$

$$
\text{added\_start}_i = \min\!\left(\text{padded\_added\_start}_i,\ V\right),
\qquad
\text{added\_end}_i = \min\!\left(\text{padded\_added\_end}_i,\ V\right)
$$

于是本 rank 的**真实**词数为：

$$
n^{\text{org}}_i = \text{org\_end}_i - \text{org\_start}_i
= \max\!\left(0,\ \min\!\left(\text{padded\_org\_end}_i,\ V_{\text{org}}\right) - \text{padded\_org\_start}_i\right)
$$

当某个 rank 的 padded 区间**完全落在 padding 里**时（即 $\text{padded\_org\_start}_i \ge V_{\text{org}}$），有 $\text{start} = \text{end} = V_{\text{org}}$，于是 $n^{\text{org}}_i = 0$——这个 rank 在 base 段没有任何真实词。

### TP2 的例子

沿用 $V_{\text{org}} = 1010,\ A = 16$，此时 $\tilde{V}_{\text{org}} = 1024,\ \tilde{V} = 1088$，每 rank 持有 $1088/2 = 544$ 个 slot：

| rank | base 真实范围 | base padding | LoRA 真实范围 | LoRA padding |
| --- | --- | --- | --- | --- |
| 0 | token 0..511（512 个） | 0 个 | token 1010..1015（6 个）| 26 个 |
| 1 | token 512..1009（498 个） | 14 个 | 无 | 32 个 |

可见 rank 1 的 base 段只有 $1010 - 512 = 498$ 个真实词，剩下 14 个是 padding。

**各 rank 的真实词数可以不等，但 padded 后的 slot 数必然相等**，即不变量：

$$
\forall i:\quad
n^{\text{org}}_i + \text{pad}^{\text{org}}_i + n^{\text{added}}_i + \text{pad}^{\text{added}}_i
= \frac{\tilde{V}}{p}
= \text{const}
$$

同时真实词数求和恰好覆盖全词表：

$$
\sum_{i=0}^{p-1} n^{\text{org}}_i = V_{\text{org}},
\qquad
\sum_{i=0}^{p-1} n^{\text{added}}_i = A
$$

前一个不变量由构造函数断言保证：

```322:324:python/sglang/srt/layers/vocab_parallel_embedding.py
        assert (
            self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        )
```

---

## 4. 前向：Mask + 局部查表 + All-Reduce

这是本模块最核心的算法。所有 rank 拿到的 `input_` 是**完全相同的全局 token id**，但每个 rank 只持有部分词表。

### 三步走

设 rank $i$ 持有的 token id 集合为 $\mathcal{S}_i$，则：

$$
\text{Embed}(t) = \sum_{i=0}^{p-1} \underbrace{\mathbf{1}\!\left[t \in \mathcal{S}_i\right] \cdot W^{(i)}\!\left[\phi_i(t)\right]}_{\text{rank } i \text{ 的贡献}}
$$

其中 $\phi_i$ 是「全局 token id → 本地行号」的映射。因为 $\{\mathcal{S}_i\}$ 两两不交且并集覆盖全部词，上式恰好有**唯一一项非零**，所以求和（all-reduce）就能还原正确结果。

```504:532:python/sglang/srt/layers/vocab_parallel_embedding.py
        if self.tp_size > 1:
            # Build the mask.
            masked_input, input_mask = get_masked_input_and_mask(...)
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.quant_method.embedding(self, masked_input.long())
        if self.tp_size > 1:
            # Mask the output embedding.
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            ...
                output_parallel = tensor_model_parallel_all_reduce(output_parallel)
```

1. **Mask**：算出哪些 token 属于本 rank，并把全局 id 换算成本地行号；
2. **查表**：不属于本 rank 的 token 被映射到行号 0（安全值），先查了再说；
3. **清零 + All-Reduce**：把不属于本 rank 的行强制置 0，再求和。

> 注意第 2 步是「先查再清零」而不是「先判断再查」——因为 GPU 上分支发散代价高，无条件 gather 再 mask 反而更快。

### 地址换算

`get_masked_input_and_mask` 把两段（base / added）的全局 id 映射到连续的本地行号：

$$
\phi_i(t) =
\begin{cases}
t - \text{org\_start}_i, & t \in \left[\text{org\_start}_i,\ \text{org\_end}_i\right) \\[2ex]
t - \text{added\_start}_i + n^{\text{org}}_i + \text{pad}^{\text{org}}_i, & t \in \left[\text{added\_start}_i,\ \text{added\_end}_i\right) \\[2ex]
0 \quad (\text{无效，将被清零}), & \text{otherwise}
\end{cases}
$$

其中 $n^{\text{org}}_i$ 是本 rank base 段的真实词数、$\text{pad}^{\text{org}}_i$ 是 base padding 长度。第二行的加项正是**跳过 BASE + BASE PAD 两段**，落到 LORA 段起点。

代码里用无分支的算术实现（便于 `torch.compile` 融合成单 kernel）：

```153:163:python/sglang/srt/layers/vocab_parallel_embedding.py
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask
```

`valid_offset` 用两个布尔 mask 相乘做选择（互斥，最多一个为真），最后再乘 `vocab_mask` 把无效 token 归零。整个函数被 `@torch.compile` 装饰，所有 pointwise 操作融合成一个 kernel。

### tp_size == 1 时不做 mask

单卡下 `masked_input = input_` 直接透传，**不做任何越界检查**。因此在入口处有一道显式的 OOB 探测：

```501:503:python/sglang/srt/layers/vocab_parallel_embedding.py
        maybe_detect_oob(
            input_, 0, self.num_embeddings, "VocabParallelEmbedding input id"
        )
```

否则非法 token id 会静默读到越界显存，产生难以定位的错误结果而非报错。

---

## 5. 通信组的选择：全局 TP vs Attention TP

```526:531:python/sglang/srt/layers/vocab_parallel_embedding.py
            if not get_attn_tp_context().input_scattered:
                if self.use_attn_tp_group:
                    output_parallel = attn_tp_all_reduce(output_parallel)
                else:
                    # Reduce across all the model parallel GPUs.
                    output_parallel = tensor_model_parallel_all_reduce(output_parallel)
```

三种情形：

| 条件 | 行为 | 原因 |
| --- | --- | --- |
| `input_scattered` | **不通信** | 下游本来就按 rank 消费局部 token，无需汇总 |
| `use_attn_tp_group` | 在 **attention TP 子组**内 reduce | DP attention 下每个 rank 只负责自己的 token，全局 reduce 是浪费 |
| 默认 | 全局 TP 组 all-reduce | 常规 TP |

### 嵌入复制模式

```178:185:python/sglang/srt/layers/vocab_parallel_embedding.py
    if envs.SGLANG_ENABLE_EMBED_REPLICATION.get():
        # Replicate the full table on every rank: skips the embed all-reduce
        # at the cost of duplicated embedding weights.
        return {"enable_tp": False}
    return {"enable_tp": True, "use_attn_tp_group": is_dp_attention_enabled()}
```

`SGLANG_ENABLE_EMBED_REPLICATION=1` 时每张卡存**完整**嵌入表，用显存换掉那次 all-reduce：

$$
\text{显存代价} = (p - 1) \cdot V \cdot H \cdot \text{sizeof(dtype)}
\qquad\text{换取}\qquad
\text{每层 0 次 embed all-reduce}
$$

> `get_embedding_tp_kwargs()` 存在的意义是**强制一致性**：EAGLE / NextN 投机解码的 draft 模型会通过 `set_embed` 直接共享 target 的 `embed_tokens.weight`。若两者布局不同，draft 的 mask/索引计算会作用在错误布局的张量上，导致 `accept_len` 静默下降（不报错，只是投机接受率变差）。所以 target 和所有 draft 必须走同一个 helper。

---

## 6. 权重加载

```469:496:python/sglang/srt/layers/vocab_parallel_embedding.py
        # Shard indexes for loading the weight
        start_idx = self.shard_indices.org_vocab_start_index
        shard_size = self.shard_indices.org_vocab_end_index - start_idx
        ...
        if not self.use_presharded_weights:
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        param[: loaded_weight.shape[0]].data.copy_(loaded_weight)
        param[loaded_weight.shape[0] :].data.fill_(0)
```

注意两点：

1. **只加载 base 段**：`start_idx` / `shard_size` 取自 `org_vocab_*`，checkpoint 里没有 LoRA 词。
2. **尾部显式填 0**：`param[shard_size:].fill_(0)`。padding 行必须是 0 而不是未初始化显存——否则 LM head 会在 padding 位置产生随机 logits，可能被采样命中，输出非法 token。

打包量化（如 4bit）时下标需按 `packed_factor` 折算：

$$
\text{start\_idx}' = \frac{\text{start\_idx}}{\text{packed\_factor}},
\qquad
\text{shard\_size}' = \frac{\text{shard\_size}}{\text{packed\_factor}}
$$

---

## 7. ParallelLMHead

`ParallelLMHead` 继承自 `VocabParallelEmbedding`，**复用全部切分逻辑**，但用法完全不同。

### 它的 forward 是禁用的

```618:620:python/sglang/srt/layers/vocab_parallel_embedding.py
    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")
```

`ParallelLMHead` 只是个**权重容器**。真正的计算在 `logits_processor.py` 的 `_compute_lm_head` 里，直接取 `lm_head.weight` 做矩阵乘：

$$
\text{logits}^{(i)} = X \left(W^{(i)}\right)^{\!\top} \in \mathbb{R}^{T \times \tilde{V}/p}
$$

然后由 `LogitsProcessor` 做 all-gather 拼回完整词表。这样设计是为了让 logits 计算能与 fp32 提升、logit_scale、softcap、DP attention 的 gather/scatter 等逻辑统一编排。

### 权重绑定

```609:616:python/sglang/srt/layers/vocab_parallel_embedding.py
    def tie_weights(self, embed_tokens: VocabParallelEmbedding):
        """Tie the weights with word embeddings."""
        # GGUF quantized embed_tokens.
        if self.quant_config and self.quant_config.get_name() == "gguf":
            return embed_tokens
        else:
            self.weight = embed_tokens.weight
            return self
```

许多模型（Gemma、部分 Llama 变体）令 $W_{\text{lm\_head}} = W_{\text{embed}}$，直接省下一份 $V \times H$。因为两者切分布局完全一致，这里只需把 `weight` 指过去即可。GGUF 是例外——它的嵌入权重带特殊量化布局，只能反过来返回 `embed_tokens` 本身。

---

## 8. get_sharded_to_full_mapping：采样前的重排

all-gather 之后，logits 的下标顺序是「按 rank 拼接的四段布局」，**不等于 token id**：

```
rank0: [base_0 | pad_0 | lora_0 | pad_0'] rank1: [base_1 | pad_1 | lora_1 | pad_1'] ...
```

而采样需要 `index == token_id`。`get_sharded_to_full_mapping()` 生成一个重排索引，把同类段收拢到一起：

$$
\text{ret} = \underbrace{\left[\ \text{base}_0,\ \text{base}_1,\ \ldots\ \right]}_{\text{token } 0 \ldots V_{\text{org}} - 1}
\Vert
\underbrace{\left[\ \text{lora}_0,\ \text{lora}_1,\ \ldots\ \right]}_{\text{token } V_{\text{org}} \ldots V - 1}
\Vert
\underbrace{\left[\ \text{所有 padding}\ \right]}_{\text{尾部，采样时截断}}
$$

```443:445:python/sglang/srt/layers/vocab_parallel_embedding.py
        ret = base_embeddings + added_embeddings + padding
        assert len(ret) == self.num_embeddings_padded
        return ret
```

所有 padding 被推到**最末尾**，这样采样时只要取前 $V$ 个就自动排除了 padding。$p < 2$ 时返回 `None`（无需重排）。

---

## 9. 与其他并行层的对比

| 层 | 切分维 | 输入 | 输出 | 通信 |
| --- | --- | --- | --- | --- |
| `VocabParallelEmbedding` | 词表 $V$ | 全局 token id | $[T,H]$，部分为 0 | all-reduce |
| `ParallelLMHead` | 词表 $V$ | $[T,H]$ 完整 | $[T,\tilde{V}/p]$ | all-gather（在 LogitsProcessor） |
| `ColumnParallelLinear` | 输出维 | 完整 | 切分 | 可选 all-gather |
| `RowParallelLinear` | 输入维 | 切分 | 部分和 | all-reduce |

结构上：`VocabParallelEmbedding` ≈ **行并行**（输出形状完整、数值需累加），`ParallelLMHead` ≈ **列并行**（输出形状切分、数值完整）。二者共享同一份切分布局，正好构成模型首尾的对称。

---

## 10. 自检问题

1. $V_{\text{org}} = 32000,\ A = 0,\ P = 64,\ p = 8$ 时，$\tilde{V}$ 是多少？每 rank 几个 slot？有 padding 吗？
2. 为什么 padding 要分两步（先对齐 $V_{\text{org}}$ 再加 $A$），而不是直接对齐 $V_{\text{org}} + A$？
3. 前向里为什么可以「先无条件查表、再把无效行清零」而不会读到越界地址？
4. `tp_size == 1` 时不走 mask 分支，那非法 token id 靠什么发现？
5. 权重加载末尾的 `param[shard_size:].fill_(0)` 如果去掉，会出现什么现象？
6. `tie_weights` 为什么对 GGUF 要特殊处理，返回的对象有什么不同？
7. all-gather 后的 logits 下标为什么不等于 token id？`get_sharded_to_full_mapping` 如何修正？

