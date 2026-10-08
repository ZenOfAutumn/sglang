# 多头 / 多查询 / 分组查询注意力（MHA / MQA / GQA）原理详解

> 本文系统介绍标准注意力家族——多头注意力（MHA）、多查询注意力（MQA）、
> 分组查询注意力（GQA）——的数学原理、复杂度与显存收益，
> 以及 SGLang 中通过 `num_kv_heads` 统一三者的实现：注意力层标记、张量并行分片、
> KV cache 池按 KV 头数分配、attention backend 的 KV 头广播。

## 目录

1. [为什么需要 MQA / GQA](#1-为什么需要-mqa--gqa)
2. [数学定义：从 MHA 到 GQA](#2-数学定义从-mha-到-gqa)
3. [三者的统一视角](#3-三者的统一视角)
4. [复杂度与显存收益分析](#4-复杂度与显存收益分析)
5. [SGLang 中的实现](#5-sglang-中的实现)
6. [支持的模型](#6-支持的模型)
7. [局限与权衡](#7-局限与权衡)

---

## 1. 为什么需要 MQA / GQA

标准的多头注意力（Multi-Head Attention, MHA）为每个注意力头都配备独立的
Query、Key、Value 投影。设头数为 $h$、每头维度为 $d_k$，则每个 token 在每一层都要
缓存 $h$ 组 Key 和 Value。自回归推理时这些 KV 常驻显存（KV cache），其显存占用为：

$$
\text{KV cache} \propto n \cdot L \cdot h \cdot d_k \cdot 2
$$

其中 $n$ 为序列长度、$L$ 为层数。当 batch size 与上下文长度增大，KV cache 会迅速
吞掉显存，成为限制并发吞吐的主要瓶颈。

关键观察是：**注意力的「表达能力」主要来自多个 Query 头各自不同的关注模式，而 Key/Value
头的数量未必需要和 Query 头一样多。** 由此衍生出两种压缩思路：

- **MQA（Multi-Query Attention）**：所有 Query 头共享**同一组** K/V（$h$ 个 Q 头、1 组 KV）。
  KV cache 直接缩小 $h$ 倍，但表达能力损失较大、训练易不稳定。
- **GQA（Grouped-Query Attention）**：把 Query 头分成 $g$ 组，**每组共享一组 K/V**
  （$h$ 个 Q 头、$g$ 组 KV，$1 < g < h$）。它在 MHA（$g=h$）与 MQA（$g=1$）之间提供了
  一个可调的折中点，是当前主流大模型（Llama 系列等）的默认选择。

---

## 2. 数学定义：从 MHA 到 GQA

设隐藏维度 $D$、Query 头数 $h$、每头维度 $d_k = D/h$。

### MHA

每个头 $i \in \{1,\dots,h\}$ 拥有独立的投影矩阵 $W_i^Q, W_i^K, W_i^V$：

$$
\text{head}_i = \text{softmax}\!\left(\frac{(XW_i^Q)(XW_i^K)^\top}{\sqrt{d_k}}\right)(XW_i^V)
$$

输出为各头拼接后再投影：$\text{MHA}(X) = [\text{head}_1;\dots;\text{head}_h]\,W^O$。
此时 K/V 头数 $h_{kv} = h$。

### MQA

所有 Query 头共用一组 $W^K, W^V$（即 $h_{kv}=1$）：

$$
\text{head}_i = \text{softmax}\!\left(\frac{(XW_i^Q)(XW^K)^\top}{\sqrt{d_k}}\right)(XW^V)
$$

### GQA

把 $h$ 个 Query 头划分为 $h_{kv}$ 组，组大小 $G = h / h_{kv}$。第 $i$ 个 Query 头
使用第 $\lfloor i / G \rfloor$ 组的 K/V：

$$
\text{head}_i = \text{softmax}\!\left(\frac{(XW_i^Q)(XW_{\lfloor i/G\rfloor}^K)^\top}{\sqrt{d_k}}\right)
(XW_{\lfloor i/G\rfloor}^V)
$$

计算时通常把每组 K/V「广播（repeat）」到组内的 $G$ 个 Query 头上，再做标准注意力。
注意：**广播只发生在计算时，缓存里只存 $h_{kv}$ 组 K/V**——这正是显存收益的来源。

---

## 3. 三者的统一视角

MHA、MQA、GQA 并非三套独立机制，而是同一公式在「KV 头数 $h_{kv}$」这一维度上的
三个取值：

```
            h_kv = h          1 < h_kv < h          h_kv = 1
            ┌────────┐        ┌────────┐            ┌────────┐
Q 头:  ████████ (h)     ████████ (h)         ████████ (h)
KV 头: ████████ (h)     ██  ██  (h_kv)       █        (1)
            MHA              GQA                  MQA
       每个 Q 头独占       每组 Q 头共享        所有 Q 头共享
       一组 KV             一组 KV              一组 KV
```

SGLang 正是利用这个统一视角：代码里**没有** MHA/MQA/GQA 三条分支，只有一个
`num_kv_heads` 参数。三者的区别完全由模型配置中 `num_attention_heads` 与
`num_key_value_heads` 的取值关系决定。

---

## 4. 复杂度与显存收益分析

### 计算复杂度

三者的注意力计算量都是 $O(n^2 \cdot D)$ 量级——GQA/MQA 并不减少注意力分数矩阵的规模
（仍是 $h$ 个 Query 头各算一份），主要节省的是 **KV cache 显存**与**访存带宽**。

### KV cache 显存

| 类型 | KV 头数 | KV cache 显存 | 相对 MHA |
|------|---------|--------------|---------|
| MHA  | $h$      | $\propto n L h\, d_k$         | $1\times$ |
| GQA  | $h_{kv}$ | $\propto n L h_{kv} d_k$      | $h_{kv}/h$ |
| MQA  | $1$      | $\propto n L d_k$            | $1/h$ |

例如 Llama-2-70B 使用 $h=64$、$h_{kv}=8$ 的 GQA，KV cache 缩小为 MHA 的 $1/8$，
在长上下文、大 batch 场景下显著提升可容纳的并发数。访存上，decode 阶段每步要从显存
读取整段 KV，KV 越小则越接近算力受限而非带宽受限，吞吐更高。

---

## 5. SGLang 中的实现

SGLang 对三者的处理是「**一套代码、参数区分**」，贯穿四个层面：
**注意力层标记 → 张量并行分片 → KV 池按 KV 头分配 → backend 广播**。

### 5.1 注意力层只记录 Q / K / V 三个头数

`RadixAttention` 构造时把传入的 `num_heads`（Query 头）和 `num_kv_heads`
（Key/Value 头）分别记录，K 头与 V 头数相同：

```python
# python/sglang/srt/layers/radix_attention.py:71
self.tp_q_head_num = num_heads
self.tp_k_head_num = num_kv_heads
self.tp_v_head_num = num_kv_heads
```

下游所有逻辑都只看这三个数。`tp_` 前缀表示这是**张量并行切分后**本 GPU 持有的头数。

### 5.2 模型侧的头数与张量并行分片

以 Llama 为例，Query 头与 KV 头各自按张量并行度 `tp_size` 切分。KV 头数较少时，
用 `max(1, ...)` 保证至少 1 个，必要时跨 GPU 复制：

```python
# python/sglang/srt/models/llama.py:140
self.num_heads = self.total_num_heads // tp_size
self.total_num_kv_heads = num_kv_heads
...
# python/sglang/srt/models/llama.py:150
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

QKV 三个投影融合在一个 `QKVParallelLinear` 中，分别传入 Q 头总数与 KV 头总数，
由它负责按头切分输出通道：

```python
# python/sglang/srt/models/llama.py:163
self.qkv_proj = QKVParallelLinear(
    ...
    self.total_num_heads,
    self.total_num_kv_heads,
    ...
)
```

每 GPU 的 KV 头数由 `model_config` 统一计算，与 KV 池、backend 共用同一来源：

```python
# python/sglang/srt/configs/model_config.py:677
return max(1, total_num_kv_heads // tensor_parallel_size)
```

### 5.3 KV cache 池按「KV 头数」分配

GQA/MQA 的显存收益在 `MHATokenToKVPool` 落地：K/V 缓冲区的头维度用的是
`head_num`（即 KV 头数），而**不是** Query 头数。Query 头多、KV 头少时，缓冲区随之变小：

```python
# python/sglang/srt/mem_cache/memory_pool.py:853
(self.size + self.page_size, self.head_num, self.head_dim),
...
# python/sglang/srt/mem_cache/memory_pool.py:861
(self.size + self.page_size, self.head_num, self.v_head_dim),
```

`head_num` 的取值正来自 `model_config.get_num_kv_heads(...)`，与 §5.2 同源。
这意味着 MQA（KV 头=1）和 GQA（KV 头=$h_{kv}$）天然地只占用对应比例的显存，
无需任何特判分支。

### 5.4 attention backend 把 KV 头广播到 Query 头

计算注意力时，需要把每组 K/V 复用到组内的多个 Query 头上。组大小
$G = h_q / h_{kv}$ 在 kernel 内动态推导。以 Triton decode kernel 为例：

```python
# python/sglang/srt/layers/attention/triton_ops/decode_attention.py:207
kv_group_num = q.shape[1] // k_buffer.shape[1]
...
# python/sglang/srt/layers/attention/triton_ops/decode_attention.py:78
cur_kv_head = cur_head // kv_group_num
```

即「当前 Query 头索引 ÷ 组大小 = 它该读的 KV 头索引」。Triton backend 在分组存在时
（`num_kv_group > 1`）会选用专门的分组 decode kernel：

```python
# python/sglang/srt/layers/attention/triton_backend.py:1299
block_h, num_kv_group = 16, num_head // num_kv_head
if num_kv_group == 1:
    ...
```

FlashAttention backend 则直接把两套头数原样传给 FA3，由 kernel 内部隐式完成
GQA 广播，无需手动 repeat：

```python
# python/sglang/srt/layers/attention/flashattention_backend.py:1007
q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
```

### 5.5 数据流总览

```
模型配置 (num_attention_heads, num_key_value_heads)
  │   h_kv == h → MHA ；1 < h_kv < h → GQA ；h_kv == 1 → MQA
  ▼
QKVParallelLinear 分片         # 按 tp_size 切 Q 头与 KV 头
  ▼
RadixAttention(tp_q_head_num, tp_k/v_head_num)
  ▼
MHATokenToKVPool                # K/V 缓冲按 head_num(=KV 头数) 分配，省显存
  ▼
attention backend
  └─ kv_group_num = q_heads / kv_heads
     cur_kv_head = cur_head // kv_group_num   # 广播到 Query 头
```

---

## 6. 支持的模型

由于三者共用同一套 `num_kv_heads` 机制，几乎所有 Transformer 模型都自动支持，
区别仅在配置：

- **GQA**：Llama 2/3、ChatGLM、StableLM、MiniCPM3、Step3-VL 等绝大多数现代模型。
  Llama 在 `models/llama.py:279` 处把 `num_kv_heads=config.num_key_value_heads` 传入。
- **MQA**：GPTBigCode / StarCoder 通过 `multi_query` 标志强制 KV 头为 1：

  ```python
  # python/sglang/srt/models/gpt_bigcode.py:59
  self.multi_query = config.multi_query
  if self.multi_query:
      total_num_kv_heads = 1
      self.num_kv_heads = 1
  ```

- **MHA**：早期模型（如原始 GPT-2、BERT 类）或显式设置 `num_key_value_heads ==
  num_attention_heads` 的模型。

---

## 7. 局限与权衡

- **质量与显存的折中**：MQA 显存最省但质量损失最明显、训练易不稳；GQA 通过保留若干
  KV 组在质量与显存间取得平衡，已成为事实标准。组数 $h_{kv}$ 是关键超参，太小伤质量、
  太大省不了显存。
- **计算量并未下降**：GQA/MQA 节省的是 KV cache 显存与访存带宽，注意力的浮点计算量
  与 MHA 同阶。在算力受限（而非带宽受限）场景下，收益相对有限。
- **张量并行下的 KV 头复制**：当 `tp_size > h_kv` 时，KV 头需跨 GPU 复制
  （见 `max(1, total_num_kv_heads // tp_size)`），会带来一定冗余，限制了可用的并行度配置。
- **与其他机制的叠加**：GQA 是 MLA、SWA 等更激进压缩方案的基线对照。后者在 GQA
  之上进一步压缩（潜在向量、滑动窗口），但实现复杂度也随之上升。

---

## 参考实现位置

| 模块 | 文件路径 |
|------|---------|
| 注意力层头数标记 | `python/sglang/srt/layers/radix_attention.py` |
| 模型侧分片与 QKV 投影 | `python/sglang/srt/models/llama.py` |
| 每 GPU KV 头数计算 | `python/sglang/srt/configs/model_config.py` |
| KV 池按 KV 头分配 | `python/sglang/srt/mem_cache/memory_pool.py` |
| Triton 分组 decode kernel | `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` |
| Triton backend 调度 | `python/sglang/srt/layers/attention/triton_backend.py` |
| FlashAttention backend | `python/sglang/srt/layers/attention/flashattention_backend.py` |
| MQA 模型示例 | `python/sglang/srt/models/gpt_bigcode.py` |
