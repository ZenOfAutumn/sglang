# 多头潜在注意力（Multi-head Latent Attention, MLA）原理详解

> 本文系统介绍 DeepSeek 系列提出的多头潜在注意力（MLA）的数学原理、低秩 KV 压缩、
> 解耦 RoPE 与矩阵吸收（matrix absorption）技巧，及其复杂度与显存收益，
> 以及 SGLang 中针对 MLA 的双前向路径、专用 KV cache 池与多种 MLA attention backend 的实现。

## 目录

1. [为什么需要 MLA](#1-为什么需要-mla)
2. [MLA 的数学定义](#2-mla-的数学定义)
3. [解耦 RoPE](#3-解耦-rope)
4. [矩阵吸收与两条前向路径](#4-矩阵吸收与两条前向路径)
5. [复杂度与显存收益分析](#5-复杂度与显存收益分析)
6. [SGLang 中的 MLA 实现](#6-sglang-中的-mla-实现)
7. [支持 MLA 的模型](#7-支持-mla-的模型)
8. [局限与权衡](#8-局限与权衡)

---

## 1. 为什么需要 MLA

GQA/MQA 通过减少 KV 头数来压缩 KV cache（见 `mha_mqa_gqa_zh.md`），但它是「粗粒度」的：
要么共享整组 K/V，要么不共享，质量损失与压缩率难以解耦。

MLA 的核心思想更激进：**不直接缓存每个头的 Key/Value，而是把它们压缩成一个低维的
「潜在向量」（latent vector）$c^{KV}$，推理时只缓存这个潜在向量，用到时再即时解压。**

这带来两个好处：

1. **KV cache 极致压缩**：每个 token 每层只需缓存一个 $d_c$ 维的潜在向量
   （DeepSeek-V2 中 $d_c = 512$），而非 $h_{kv}$ 组完整的 $d_k$ 维 K/V。
2. **质量几乎无损**：低秩压缩是「可学习」的连续压缩，相比 GQA 的「离散丢头」，
   在同等甚至更小的 KV cache 下能保持接近 MHA 的质量。

但天真的「缓存潜在向量、用时解压」会与 RoPE 位置编码冲突，也会在 decode 时引入额外
矩阵乘。MLA 用**解耦 RoPE**和**矩阵吸收**两个技巧解决这两个问题。

---

## 2. MLA 的数学定义

### 2.1 低秩联合压缩

设输入隐藏向量为 $h_t \in \mathbb{R}^{D}$。MLA 先把它降维到一个潜在向量：

$$
c^{KV}_t = W^{DKV}\, h_t \in \mathbb{R}^{d_c}
$$

其中 $W^{DKV}$ 是降维（down-projection）矩阵，$d_c \ll h \cdot d_k$。**KV cache 只存
$c^{KV}_t$。** 需要时再用升维（up-projection）矩阵 $W^{UK}, W^{UV}$ 还原出每个头的
Key 与 Value：

$$
k^{C}_t = W^{UK} c^{KV}_t, \qquad v^{C}_t = W^{UV} c^{KV}_t
$$

Query 端同样做低秩压缩（缓解训练显存、提升稳定性），但 Query 不需缓存：

$$
c^{Q}_t = W^{DQ} h_t, \qquad q^{C}_t = W^{UQ} c^{Q}_t
$$

DeepSeek-V2/V3 的典型维度：`q_lora_rank=1536`、`kv_lora_rank=512`、
`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`。

### 2.2 维度变化全流程

下面以 DeepSeek-V2 的真实配置，跟踪**一个 token** 的隐藏向量从输入到输出的维度变化
（暂不考虑张量并行切分，即看单卡全量）。配置：

| 超参 | 值 |
|------|----|
| `hidden_size` $D$ | 5120 |
| `num_heads` $h$ | 128 |
| `q_lora_rank` | 1536 |
| `kv_lora_rank` $d_c$ | 512 |
| `qk_nope_head_dim` | 128 |
| `qk_rope_head_dim` | 64 |
| `qk_head_dim`（= nope + rope） | 192 |
| `v_head_dim` | 128 |

#### Query 侧（不缓存，每步重算）

| 步骤 | 投影 | 维度变化 |
|------|------|---------|
| 输入隐藏向量 | — | $D = 5120$ |
| Query 降维 | $W^{DQ}$ | $5120 \to 1536$（$c^Q$） |
| Query 升维 | $W^{UQ}$ | $1536 \to h\cdot 192 = 128\times192 = 24576$ |
| 拆分到每头 | reshape | $24576 \to (128\text{ 头},\ 192)$ |
| 每头再拆 nope/rope | split | $192 \to 128(\text{nope}) + 64(\text{rope})$ |

#### Key/Value 侧（只缓存潜在向量）

| 步骤 | 投影 | 维度变化 |
|------|------|---------|
| 输入隐藏向量 | — | $D = 5120$ |
| KV 降维（含共享 rope） | $W^{DKV}$ | $5120 \to d_c + 64 = 512 + 64 = 576$ |
| **缓存** | — | 只存这 **576** 维：`[c^KV(512) \| k_pe(64)]` |
| K/V 升维 | $W^{UK},W^{UV}$ | $512 \to h\cdot(128_{\text{nope}}+128_v)=128\times256=32768$ |
| 拆分到每头 | reshape/split | 每头 $k^{nope}=128$、$v=128$；$k^{rope}=64$ 为所有头共享 |

#### 注意力与输出

| 步骤 | 操作 | 维度变化 |
|------|------|---------|
| 每头 Q/K 内积 | $q\cdot k$（nope 128 + rope 64） | 得标量打分（每头、每个历史 token 一个） |
| 加权求和 | $\sum \text{softmax}\cdot v$ | 每头输出 $= v\_head\_dim = 128$ |
| 多头拼接 | concat | $128\text{ 头}\times128 = 16384$ |
| 输出投影 | $W^O$（`o_proj`） | $16384 \to D = 5120$ |

可见**首尾都是 $D=5120$**：MLA 只改变了中间「如何产生并缓存 Q/K/V」，
对外仍是「输入 $D$ 维 → 输出 $D$ 维」的标准注意力块。

#### 关键对比：缓存维度

- **MLA 每 token 每层缓存**：$d_c + d_{rope} = 512 + 64 = 576$ 维。
- **等价 MHA 每 token 每层缓存**：$2\cdot h\cdot d_k$。若按 $h=128$、$d_k=192$
  计为 $2\times128\times192 = 49152$ 维。

二者相差约 **85 倍**——这正是 §5 显存收益的来源。注意 `o_proj` 的输入维是
$h\cdot v\_head\_dim$ 而非 $h\cdot qk\_head\_dim$，因为输出走的是 value 头维。

> SGLang 对应实现：`fused_qkv_a_proj_with_mqa` 产出 `q_lora + kv_lora + qk_rope`
> （`deepseek_v2.py:1131`），`kv_b_proj` 做 K/V 升维（`:1187`），
> `o_proj` 输入维为 `num_heads * v_head_dim`（`:1197`）。

---

## 3. 解耦 RoPE

问题在于：RoPE（旋转位置编码）是**位置相关**的旋转，作用在 Q、K 上。如果把 RoPE
施加在「解压后的 $k^C$」上，那么 $W^{UK}$ 就不能再被吸收进 Query 侧（见 §4），
矩阵吸收技巧失效，缓存潜在向量的意义大打折扣。

MLA 的解法是**把每个头的维度拆成两部分**。这两段的命名直接来自「是否施加 RoPE」：

- **nope 部分**（$d_{nope}=128$）：**nope = No-RoPE，即「不加 RoPE」**。这一段位置无关，
  走低秩压缩/解压通路（decode 时走矩阵吸收）。正因为它不带位置旋转，升维矩阵 $W^{UK}$
  才能被预乘吸收进 Query 侧（见 §4），KV 压缩才有意义。
- **rope 部分**（$d_{rope}=64$）：**rope = RoPE（旋转位置编码）**。这一段单独走一条携带
  位置信息的通路并施加 RoPE，在所有头间共享（类似 MQA 的一组），与潜在向量一起缓存
  （布局 `[c^KV | k_pe]`）。

换句话说，拆分的根本动因是 **RoPE 与矩阵吸收冲突**：RoPE 是位置相关的旋转，若施加在
「解压后的 $k^C$」上，$W^{UK}$ 就无法再被吸收（§4 技巧失效）。于是 MLA 把位置信息隔离到
rope 这一小段单独处理，让 nope 段保持位置无关、可压缩可吸收。两段在打分时**分别计算再相加**
（详见附录 A.5）。

最终 Query/Key 由两部分拼接而成：$q_t = [q^{nope}_t; q^{rope}_t]$，
$k_t = [k^{nope}_t; k^{rope}_t]$。其中 $k^{rope}$ 在所有头间共享（类似 MQA 的一组），
与潜在向量一起缓存。SGLang 中这一拆分清晰可见：

```python
# python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:191
_, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
...
k_pe = latent_cache[:, :, self.kv_lora_rank :]
if self.rotary_emb is not None:
    q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
```

注意缓存的 `latent_cache` 布局就是 `[c^{KV} (kv_lora_rank) | k_pe (qk_rope_head_dim)]`，
前段是待解压的潜在向量、后段是共享的带位置 Key。

---

## 4. 矩阵吸收与两条前向路径

### 4.1 矩阵吸收的动机

decode 阶段（一次只算一个新 token，但要对全部历史 KV 做注意力）是访存瓶颈所在。
若每步都把整段历史的潜在向量解压成完整 K/V，反而增加了计算与中间显存。

**矩阵吸收（matrix absorption）** 利用矩阵乘结合律，把升维矩阵预先「吸收」进相邻的
权重，从而让注意力**直接在压缩的潜在向量上进行**，全程不解压。

注意力打分 $q^{nope\top} k^{nope} = (W^{UQ}c^Q)^\top (W^{UK}c^{KV})
= (c^Q)^\top (W^{UQ\top} W^{UK})\, c^{KV}$。括号里的 $W^{UQ\top}W^{UK}$ 可在
权重加载时预乘成一个矩阵 `w_kc`，于是 Query 被「吸收」到潜在空间，直接与缓存的
$c^{KV}$ 做内积。同理输出端用 `w_vc` 把潜在空间的注意力结果吸收回值空间。

SGLang 在权重加载时由 `kv_b_proj` 的权重拆分出 `w_kc` / `w_vc`：

```python
# python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:555
w_kc, w_vc = w.unflatten(
    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
```

decode 路径中把 Query 吸收进潜在空间（与 `w_kc` 做 batched matmul）：

```python
# python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:291
q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
```

注意力在潜在维度上完成后，再用 `w_vc` 把结果吸收回 `v_head_dim` 输出空间：

```python
# python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:448
attn_bmm_output = torch.bmm(
    attn_output.transpose(0, 1),
    self.w_vc.to(torch.bfloat16) * self.w_scale,
    ...
)
```

### 4.2 两条路径：prefill 走 MHA、decode 走 absorb

MLA 在不同阶段用不同策略，由一个枚举调度（`forward_methods.py`，含 `MHA`、`MLA`、
`MHA_CHUNKED_KV` 等模式）：

- **MHA 路径（prefill）**：序列较长、本就要算全 $n\times n$ 注意力，此时把潜在向量
  **解压成完整 per-head K/V** 反而能直接复用高效的 FlashAttention kernel。
  `forward_mha.py` 通过 `kv_b_proj` 还原 `k_nope`/`v` 再拼上 `k_pe`。
- **MLA absorb 路径（decode）**：一次一个 query token、要对长历史做注意力，
  访存敏感。此时走 §4 的吸收路径，**直接在压缩潜在向量上注意力**，省下解压与显存。

为支撑两条路径，DeepSeek 模型侧构造了**两个** `RadixAttention` 实例：

- `attn_mqa`：维度为潜在空间（`v_head_dim = kv_lora_rank`），服务 absorb 路径
  （`deepseek_v2.py:1229`）。
- `attn_mha`：维度为 `qk_nope + qk_rope`，服务 prefill 解压路径
  （`deepseek_v2.py:1240`）。

---

## 5. 复杂度与显存收益分析

### 5.1 KV cache 显存

| 类型 | 每 token 每层缓存量 | 相对 MHA（DeepSeek-V2 配置） |
|------|------------------|----------------|
| MHA  | $2 \cdot h \cdot d_k$               | $1\times$ |
| GQA  | $2 \cdot h_{kv} \cdot d_k$          | $h_{kv}/h$ |
| MLA  | $d_c + d_{rope}$（潜在 + 共享 rope） | 约 $1/h_{kv}$ 量级，且 $\ll$ GQA |

MLA 每 token 每层只缓存 `kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576` 个元素，
而等价 MHA 需缓存 $2 \times 128 \times 128 \approx 3.3$ 万个元素。压缩比可达数十倍，
同时质量接近 MHA——这是 MLA 相比 GQA 的核心优势。

### 5.2 计算复杂度

prefill 与标准注意力同阶 $O(n^2 D)$。decode 阶段经矩阵吸收后，注意力在 $d_c$ 维潜在
空间进行，避免了逐 token 解压，把访存换成了少量额外 matmul，整体更适配带宽受限的 decode。

---

## 6. SGLang 中的 MLA 实现

### 6.1 低秩投影的构造

DeepSeek 模型在 `deepseek_v2.py` 中构造降维/升维投影。当 `q_lora_rank` 存在时，
Query 降维与 KV 降维融合在一个投影里：

```python
# python/sglang/srt/models/deepseek_v2.py:1131
self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
    self.hidden_size,
    self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
    ...
)
```

KV 升维投影 `kv_b_proj`（即 §4 中拆出 `w_kc`/`w_vc` 的来源）：

```python
# python/sglang/srt/models/deepseek_v2.py:1187
self.kv_b_proj = ColumnParallelLinear(
    self.kv_lora_rank,
    self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
    ...
)
```

`w_kc` / `w_vc` 初始为 `None`（`deepseek_v2.py:1252`），在权重加载阶段由
`kv_b_proj` 权重拆分填充（见 §4）。

### 6.2 专用 KV 池：`MLATokenToKVPool`

与 MHA 池按「头数 × 头维」分配不同，MLA 池每 token 每层只存**一个**潜在向量
（含共享的 rope 段）：

```python
# python/sglang/srt/mem_cache/memory_pool.py:1471
self.kv_cache_dim = (
    override_kv_cache_dim
    if self.nsa_kv_cache_store_fp8
    else (kv_lora_rank + qk_rope_head_dim)
)
```

缓冲区形状是 `(size, 1, kv_cache_dim)`——头维退化为 1，正体现「不分头、只存潜在向量」：

```python
# python/sglang/srt/mem_cache/memory_pool.py:1496
self.kv_buffer = [
    torch.zeros(
        (self.size + self.page_size, 1, self.kv_cache_dim),
        dtype=self.store_dtype,
        device=self.device,
    )
    for _ in range(self.layer_num)
]
```

`get_value_buffer` 取潜在向量的前 `kv_lora_rank` 段（`memory_pool.py:1539`），
`set_mla_kv_buffer`（`:1566`）负责写入。该池还派生出 FP4 量化版 `MLATokenToKVPoolFP4`
与 NSA 版 `NSATokenToKVPool`。

### 6.3 多种 MLA 专用 attention backend

由于 MLA 的潜在空间注意力与标准注意力 kernel 不同，SGLang 提供了一组专用 backend：

| backend | 文件 |
|---------|------|
| FlashInfer MLA | `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` |
| FlashMLA | `python/sglang/srt/layers/attention/flashmla_backend.py` |
| CUTLASS MLA | `python/sglang/srt/layers/attention/cutlass_mla_backend.py` |
| TRT-LLM MLA | `python/sglang/srt/layers/attention/trtllm_mla_backend.py` |

### 6.4 数据流总览

```
hidden h_t
  │
  ├─ W^DKV → c^KV (kv_lora_rank)  ┐
  └─ (rope 段) k_pe (qk_rope_dim) ┘→ 拼接后存入 MLATokenToKVPool: [c^KV | k_pe]
  ▼
prefill?
  ├─ 是 → MHA 路径：kv_b_proj 解压成完整 K/V → FlashAttention（attn_mha）
  └─ 否(decode) → absorb 路径：
        q_nope @ w_kc 吸收进潜在空间
        在 c^KV 上做注意力（attn_mqa）
        结果 @ w_vc 吸收回 v 空间
  ▼
解耦 RoPE：q_pe / k_pe 单独加旋转位置编码后拼接
```

---

## 7. 支持 MLA 的模型

MLA 由 DeepSeek 提出，SGLang 中通过共享 `DeepseekV2AttentionMLA` 与
`deepseek_common` 复用逻辑的模型包括：

- **DeepSeek-V2 / V3 / V3.2**（`deepseek_v2.py`）及其 NextN/MTP 变体（`deepseek_nextn.py`）。
- **DeepSeek-VL2 / DeepSeek-OCR**（多模态，`deepseek_vl2.py`、`deepseek_ocr.py`）。
- **GLM-4-MoE-Lite**（`glm4_moe_lite.py`）、**LongCat-Flash**（`longcat_flash.py`）。
- **Kimi-Linear / Kimi-VL**（`kimi_linear.py`、`kimi_vl.py`，与线性注意力混合）。
- **Bailing-MoE-linear、Sarvam-MoE、MiniCPM3** 等。

---

## 8. 局限与权衡

- **实现复杂度高**：双前向路径、矩阵吸收、解耦 RoPE、两个 RadixAttention 实例、
  专用 KV 池与多套 backend，远比 GQA 的「一个 `num_kv_heads`」复杂，调试与维护成本高。
- **prefill / decode 行为不一致**：两条路径数值上需保持一致，量化（FP8/FP4）下尤其要
  小心 `w_scale` 等缩放因子的对齐（见 `forward_mla.py` 中大量 dtype 分支）。
- **与前缀缓存/稀疏注意力的叠加**：MLA 进一步与 NSA（DeepSeek-V3.2 稀疏注意力）组合时，
  KV 池需存 FP8 量化潜在向量（`nsa_kv_cache_store_fp8`），复杂度再升一层。
- **模型必须为 MLA 训练**：MLA 是一种训练期就确定的架构，不能把已训练好的 MHA/GQA
  模型「免训练」转成 MLA。

---

## 参考实现位置

| 模块 | 文件路径 |
|------|---------|
| MLA 投影与双 attention 实例 | `python/sglang/srt/models/deepseek_v2.py` |
| prefill 解压路径 | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` |
| decode 吸收路径 | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 路径调度枚举 | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py` |
| w_kc/w_vc 权重拆分 | `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` |
| MLA KV 池 | `python/sglang/srt/mem_cache/memory_pool.py` |
| MLA attention backend | `python/sglang/srt/layers/attention/{flashinfer_mla,flashmla,cutlass_mla,trtllm_mla}_backend.py` |

---

## 附录 A：MLA 数值计算示例

为直观理解 §2~§4 的低秩压缩、解耦 RoPE 与矩阵吸收，下面用一组**极小维度**的具体数字
完整走一遍 MLA 的计算流程，并验证「矩阵吸收」与「先解压再注意力」在数值上等价。

### A.0 维度约定（玩具配置）

| 符号 | 含义 | 取值 |
|------|------|------|
| $D$ | 隐藏维 | 4 |
| $d_c$ | 潜在向量维（kv_lora_rank） | 2 |
| $d_{nope}$ | nope 部分头维 | 2 |
| $d_{rope}$ | rope 部分头维 | 2 |
| $h$ | 头数 | 1（单头便于手算） |

注意力总头维 $d_k = d_{nope} + d_{rope} = 4$。下面只演示 **1 个 query token 对 1 个历史 token**
做注意力（decode 场景），足以体现矩阵吸收。

### A.1 权重与输入（设定）

输入隐藏向量（当前 query token 与 1 个历史 token）：

$$
h_q = [1,\ 0,\ 1,\ 0],\qquad h_1 = [0,\ 1,\ 0,\ 1]
$$

降维矩阵（KV 与 Query 各一个，取简单 0/1 形式）：

$$
W^{DKV} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix},\qquad
W^{DQ}  = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

升维矩阵（潜在 $d_c=2$ → nope 头维 $d_{nope}=2$）：

$$
W^{UK} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix},\qquad
W^{UQ} = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix},\qquad
W^{UV} = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}
$$

为聚焦核心，下面 **rope 部分先记为零向量**（$q^{rope}=k^{rope}=[0,0]$），
即位置贡献为 0，A.4 再单独说明 RoPE 如何并入。

### A.2 低秩压缩（§2）

潜在向量（只缓存历史 token 的 $c^{KV}$）：

$$
c^{KV}_1 = W^{DKV} h_1 = [0,\ 1]
$$

Query 压缩：

$$
c^{Q} = W^{DQ} h_q = [1,\ 0]
$$

**KV cache 里只存 $c^{KV}_1 = [0,1]$（2 个数）**，而非完整的 per-head K/V。

### A.3 路径一：先解压再注意力（prefill / MHA 路径）

解压历史 token 的 nope Key、以及 Value：

$$
k^{nope}_1 = W^{UK} c^{KV}_1 = \begin{bmatrix}1&1\\0&1\end{bmatrix}\begin{bmatrix}0\\1\end{bmatrix} = [1,\ 1]
$$

$$
v_1 = W^{UV} c^{KV}_1 = \begin{bmatrix}2&0\\0&1\end{bmatrix}\begin{bmatrix}0\\1\end{bmatrix} = [0,\ 1]
$$

解压 Query 的 nope 部分：

$$
q^{nope} = W^{UQ} c^{Q} = \begin{bmatrix}1&0\\1&1\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix} = [1,\ 1]
$$

注意力打分（nope 内积 + rope 内积，rope 段为 0）：

$$
\text{score} = q^{nope}\cdot k^{nope}_1 + q^{rope}\cdot k^{rope}_1 = (1\cdot1 + 1\cdot1) + 0 = 2
$$

只有 1 个历史 token，softmax 后权重为 1，故注意力输出 = $v_1 = [0,\ 1]$。

### A.4 路径二：矩阵吸收（decode / absorb 路径，§4）

不解压，先把升维矩阵预乘成吸收矩阵：

$$
W^{abs} = (W^{UQ})^\top W^{UK}
= \begin{bmatrix}1&1\\0&1\end{bmatrix}\begin{bmatrix}1&1\\0&1\end{bmatrix}
= \begin{bmatrix}1&2\\0&1\end{bmatrix}
$$

> 说明：打分 $q^{nope}\!\cdot k^{nope} = (W^{UQ}c^{Q})^\top(W^{UK}c^{KV})
> = (c^{Q})^\top\big((W^{UQ})^\top W^{UK}\big) c^{KV} = (c^{Q})^\top W^{abs}\, c^{KV}$。

直接在**潜在空间**用 $c^Q$、$c^{KV}$ 计算打分，全程不解压：

$$
\text{score} = (c^{Q})^\top W^{abs}\, c^{KV}_1
= [1,\ 0]\begin{bmatrix}1&2\\0&1\end{bmatrix}\begin{bmatrix}0\\1\end{bmatrix}
= [1,\ 2]\begin{bmatrix}0\\1\end{bmatrix} = 2
$$

与 A.3 的 score = 2 **完全一致**，验证了矩阵吸收的等价性。

注意力在潜在空间得到的加权结果就是 $c^{KV}_1=[0,1]$（单 token，权重为 1），
再用 $W^{UV}$ 吸收回值空间：

$$
\text{out} = W^{UV} c^{KV}_1 = [0,\ 1]
$$

同样与 A.3 的输出 $[0,1]$ 一致。

### A.5 解耦 RoPE 如何并入（§3）

上面把 rope 段设为 0 以聚焦吸收。实际中 rope 段**不参与吸收**，而是单独计算后**加到打分上**：

$$
\text{score} = \underbrace{(c^{Q})^\top W^{abs}\, c^{KV}}_{\text{nope，潜在空间吸收}}
\;+\; \underbrace{q^{rope}\cdot k^{rope}}_{\text{rope，带位置，直接内积}}
$$

下面把 $q^{rope}$、$k^{rope}$ 的值**真正用 RoPE 算出来**，而非凭空假设。

#### A.5.1 RoPE 旋转的定义

RoPE 把 rope 维度两两分组，对处在位置 $p$ 的向量，第 $g$ 组（角频率 $\theta_g$）施加一个旋转角
$\alpha = p\,\theta_g$ 的二维旋转矩阵：

$$
R(p\,\theta_g) =
\begin{bmatrix} \cos(p\theta_g) & -\sin(p\theta_g) \\ \sin(p\theta_g) & \cos(p\theta_g) \end{bmatrix}
$$

本例 $d_{rope}=2$，恰好只有 1 组，取角频率 $\theta_0 = \tfrac{\pi}{2}$（玩具取值，便于手算）。

#### A.5.2 旋转前的 rope 分量与位置

设旋转前（投影出但尚未加位置）的 rope 分量均为单位向量 $[1,\ 0]$：

$$
\tilde{q}^{rope} = [1,\ 0],\qquad \tilde{k}^{rope}_1 = [1,\ 0]
$$

设当前 query token 在位置 $m = 2$，历史 key（$t_1$）在位置 $n = 1$。

#### A.5.3 施加旋转

query 在位置 $m=2$，旋转角 $\alpha_q = m\theta_0 = 2\cdot\tfrac{\pi}{2} = \pi$：

$$
q^{rope} = R(\pi)\,\tilde{q}^{rope}
= \begin{bmatrix}\cos\pi & -\sin\pi \\ \sin\pi & \cos\pi\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix}
= \begin{bmatrix}-1 & 0 \\ 0 & -1\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix}
= [-1,\ 0]
$$

key 在位置 $n=1$，旋转角 $\alpha_k = n\theta_0 = 1\cdot\tfrac{\pi}{2} = \tfrac{\pi}{2}$：

$$
k^{rope}_1 = R(\tfrac{\pi}{2})\,\tilde{k}^{rope}_1
= \begin{bmatrix}\cos\tfrac{\pi}{2} & -\sin\tfrac{\pi}{2} \\ \sin\tfrac{\pi}{2} & \cos\tfrac{\pi}{2}\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix}
= \begin{bmatrix}0 & -1 \\ 1 & 0\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix}
= [0,\ 1]
$$

#### A.5.4 rope 项打分与「相对位置」性质

rope 项即旋转后两向量的内积：

$$
q^{rope}\cdot k^{rope}_1 = [-1,\ 0]\cdot[0,\ 1] = 0
$$

RoPE 的关键性质是：旋转后内积只依赖**相对位置** $m-n$。验证——把两次旋转合并，
$q^{rope}\cdot k^{rope} = (R(\alpha_q)\tilde q)^\top (R(\alpha_k)\tilde k)
= \tilde q^\top R(\alpha_k-\alpha_q)\,\tilde k$，这里
$\alpha_k - \alpha_q = (n-m)\theta_0 = -\tfrac{\pi}{2}$：

$$
\tilde q^{\top} R(-\tfrac{\pi}{2})\,\tilde k
= [1,0]\begin{bmatrix}0 & 1 \\ -1 & 0\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix}
= [0,\ 1]\begin{bmatrix}1\\0\end{bmatrix} = 0
$$

与直接内积结果一致，说明打分只与相对距离 $m-n=1$ 有关。

#### A.5.5 合并到总打分

把 A.4 的 nope 项（=2）与此处 rope 项（=0）相加：

$$
\text{score} = \underbrace{2}_{\text{nope}} + \underbrace{0}_{\text{rope}} = 2
$$

> 若改设 query 也在位置 $m=1$（与 key 同位，相对距离 0），则 $\alpha_q=\alpha_k=\tfrac{\pi}{2}$，
> 旋转后 $q^{rope}=k^{rope}_1=[0,1]$，rope 项 $=1$，总打分变为 $2+1=3$。
> 可见 rope 项随相对位置变化，正是位置信息的来源。

这正是 §3 中「nope 走吸收通路、rope 走独立带位置通路、二者打分相加」的数值体现：
rope 段经 RoPE 旋转后保留了相对位置信息，且**不能被吸收进 $W^{abs}$**（旋转是位置相关的），
因此必须把 $k^{rope}$（即 `k_pe`）与潜在向量**一起缓存**（布局 `[c^KV | k_pe]`，见 §3）。

### A.6 显存直观对比

本例中 MLA 每个历史 token 只缓存 $c^{KV}=[0,1]$ 共 **2 个数**（外加共享的 rope 段）；
而等价 MHA 需缓存完整的 $k=[k^{nope};k^{rope}]$ 与 $v$，即 $4+2=6$ 个数。
随着真实维度（$d_c=512$ vs $2\times h\times d_k$）放大，这一压缩比可达数十倍，
即 §5 所述的核心显存收益。
