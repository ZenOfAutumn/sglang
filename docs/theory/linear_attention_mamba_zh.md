# 线性注意力与状态空间混合（Linear Attention / Mamba Hybrid）原理详解

> 本文介绍线性注意力与状态空间模型（SSM / Mamba2）如何用「固定大小的循环状态」取代
> 随序列增长的 KV cache，及其 $O(n)$ 时间、$O(1)$ 状态显存的复杂度优势；并详解 SGLang 中
> Mamba2 mixer、FLA（GatedDeltaNet / KDA）、Lightning Attention、逐层混合调度，
> 以及为「状态快照」设计的 Mamba 前缀缓存。

## 目录

1. [为什么需要线性注意力 / SSM](#1-为什么需要线性注意力--ssm)
2. [线性注意力与状态空间模型的数学定义](#2-线性注意力与状态空间模型的数学定义)
3. [混合架构：全注意力层 + 线性层](#3-混合架构全注意力层--线性层)
4. [复杂度与显存收益分析](#4-复杂度与显存收益分析)
5. [SGLang 中的实现](#5-sglang-中的实现)
6. [支持的模型](#6-支持的模型)
7. [局限与权衡](#7-局限与权衡)

---

## 1. 为什么需要线性注意力 / SSM

标准 softmax 注意力的代价来自两点：prefill 的 $O(n^2)$ 计算，以及 decode 时随序列
线性增长、必须常驻显存的 KV cache（$O(n)$）。即便有 GQA、MLA、SWA 等压缩手段，
KV cache 仍随上下文增长。

线性注意力与状态空间模型（SSM）走的是一条根本不同的路：

> **把「对全部历史 token 做注意力」改写成「维护一个固定大小的循环状态」。**

每来一个新 token，就用它更新这个固定大小的状态，而不再回看所有历史 KV。于是：

- **时间复杂度**降为 $O(n)$（每 token 常数代价）。
- **状态显存**为 $O(1)$——不随序列长度增长，这是与所有 KV cache 方案的本质区别。

代价是状态容量有限，对超长程精确检索能力弱于 softmax 注意力（见 §7）。现代模型因此
普遍采用**混合架构**：大部分层用线性/SSM，少数层保留全注意力。

---

## 2. 线性注意力与状态空间模型的数学定义

### 线性注意力

标准注意力为 $o_i = \sum_{j\le i}\frac{\exp(q_i k_j^\top)}{\sum \exp(\cdot)} v_j$。
线性注意力去掉 softmax，用特征映射 $\phi$ 近似，利用结合律改写：

$$
o_i = \frac{\phi(q_i)\sum_{j\le i}\phi(k_j)^\top v_j}{\phi(q_i)\sum_{j\le i}\phi(k_j)^\top}
$$

令状态矩阵 $S_i = \sum_{j\le i}\phi(k_j)^\top v_j$，它满足**递推** $S_i = S_{i-1} + \phi(k_i)^\top v_i$。
于是 decode 时只需维护固定大小的 $S$，每步 $o_i = \phi(q_i) S_i$，无需回看历史——这正是
「固定状态取代 KV cache」。门控/衰减变体（GatedDeltaNet、KDA、Lightning）在此基础上
给状态加可学习的遗忘/更新项。

### 状态空间模型（SSM / Mamba2）

SSM 把序列建模为一个连续系统的离散化递推：

$$
h_t = A\, h_{t-1} + B\, x_t, \qquad y_t = C\, h_t + D\, x_t
$$

其中 $h_t$ 是固定维度的隐状态。Mamba 的关键是让 $A,B,C$ 与步长 $\Delta t$（`dt`）
**依赖输入**（selective SSM），从而具备内容感知的选择能力。推理时同样只维护：

- **conv state**：一个短因果卷积（`conv1d`）的滑动窗口状态。
- **ssm state（temporal）**：SSM 递推的隐状态 $h_t$。

两者都是**固定大小**，与序列长度无关。

---

## 3. 混合架构：全注意力层 + 线性层

与 SWA 的混合思路类似（见 `sliding_window_attention_zh.md`），现代线性模型让一部分层
保持全注意力、其余层用线性/SSM：

```
层索引:   0       1       2       3       4       5    ...
类型:    Linear  Linear  Full   Linear  Linear  Full
状态:    固定态  固定态  全 KV  固定态  固定态  全 KV
```

- **全注意力层**：少量，负责精确长程检索，仍需完整 KV cache。
- **线性/SSM 层**：多数，$O(1)$ 状态，承担高效序列建模。

SGLang 用 `HybridLinearAttnBackend` 按层 id 把前向分派到「全注意力 backend」或
「线性 backend」，二者各自管理自己的缓存。

---

## 4. 复杂度与显存收益分析

| 机制 | prefill 时间 | decode 每步 | 每层缓存随 $n$ |
|------|------------|------------|---------------|
| 全注意力（MHA） | $O(n^2)$ | $O(n)$ 访存 | 线性增长 KV |
| 线性 / SSM | $O(n)$ | $O(1)$ | **固定**（conv + ssm state） |
| 混合 | 介于之间 | 介于之间 | 全注意力层 KV + 线性层固定态 |

线性层的状态显存与上下文长度**无关**，这使混合模型在超长上下文、大 batch 下的显存占用
远低于纯注意力模型。SGLang 中该状态由 `MambaPool` 以「每请求一个槽位」的方式管理，
而非「每 token 一个槽位」。

---

## 5. SGLang 中的实现

实现贯穿四个层面：**固定状态池 → SSM/线性 mixer → 逐层混合调度 → 状态快照式前缀缓存**。

### 5.1 固定状态池：`MambaPool`

`MambaPool` 为每层维护两类固定大小的状态张量——卷积状态与时序（SSM）状态，
**按请求槽位**索引，而非按 token：

```python
# python/sglang/srt/mem_cache/memory_pool.py:187
class MambaPool:
    @dataclass(frozen=True, kw_only=True)
    class State:
        conv: List[torch.Tensor]
        temporal: torch.Tensor
```

其形状（`memory_pool.py:421` 附近注释）形如
`conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]`——只与状态维度、并行度
有关，**与序列长度无关**，故显存是 $O(1)$。

### 5.2 Mamba2 mixer：选择性 SSM

mixer 为 `MambaMixer2`（`mamba/mamba.py:155`）。SSM 参数 $A$、$D$、`dt_bias` 是可学习参数：

```python
# python/sglang/srt/layers/attention/mamba/mamba.py:358
self.A = nn.Parameter(...)
self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))
```

前向时从缓存取出本请求的 conv / ssm 状态：

```python
# python/sglang/srt/layers/attention/mamba/mamba.py:405
conv_state = layer_cache.conv[0]
ssm_state = layer_cache.temporal
```

- **prefill**：用分块扫描 `mamba_chunk_scan_combined`（`mamba.py:533`，传入 A/B/C/D/dt_bias），
  扫完把末态写回 `ssm_state[state_indices_tensor_p] = varlen_state`（`mamba.py:562`）。
- **decode**：用 `causal_conv1d_update` + `selective_state_update` 单步更新状态
  （`mamba.py:619`–`636`）。

底层 SSD 扫描 kernel 位于 `mamba/ops/`（`ssd_chunk_scan.py`、`ssd_combined.py`、
`ssd_state_passing.py`）。

### 5.3 FLA 与 Lightning：多种线性算子

`fla/` 目录从 flash-linear-attention 移植了门控线性注意力算子：

- **GatedDeltaNet**：`fla/chunk.py:121` 的 `chunk_gated_delta_rule`（prefill 分块）、
  `fla/fused_recurrent.py:16` 的 `fused_recurrent_gated_delta_rule_fwd_kernel`（decode 单步）。
- **KDA（KimiDeltaAttention）**：`fla/kda.py:39` 的 `fused_recurrent_kda_fwd`。
- **Lightning Attention（MiniMax）**：独立于 fla，位于 `linear/lightning_attn.py` 与
  `linear/lightning_backend.py:22` 的 `LightningAttentionBackend`。

各架构的线性 backend（`linear/gdn_backend.py:242` `GDNAttnBackend`、
`linear/kda_backend.py:129` `KDAAttnBackend` 等）都继承统一基类 `MambaAttnBackendBase`。

### 5.4 逐层混合调度：`HybridLinearAttnBackend`

```python
# python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:721
class HybridLinearAttnBackend(AttentionBackend):
    def __init__(self, full_attn_backend, linear_attn_backend, full_attn_layers):
```

前向时按层 id 判断走哪条 backend（decode 路径 `:832`）：

```python
# python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:832
if layer_id in self.full_attn_layers:
    return self.full_attn_backend.forward_decode(...)
return self.linear_attn_backend.forward_decode(...)
```

`full_attn_layers` 即「哪些层是全注意力层」的集合，由模型配置决定。

### 5.5 状态快照式前缀缓存：`MambaRadixCache`

普通 RadixCache 共享的是「逐 token 的 KV 序列」，而 Mamba 状态是一个**整体快照**
（走到某前缀末尾时的状态），无法像 token 那样拼接复用。`mamba_radix_cache.py` 为此
单独设计：每个树节点存一个 `mamba_value` 槽位索引（`:76`），并有独立的锁与 LRU
（`mamba_lock_ref` `:84`、`LRUList(mamba=True)` `:169`）。

插入时存的是**克隆的单槽快照**，而非一段序列：

```python
# python/sglang/srt/mem_cache/mamba_radix_cache.py:576
mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()
```

前缀复用通过 `mamba_pool.fork_from(mamba_value)`（`:663`）**复制状态槽**，而非拼接 token。
淘汰逻辑 `evict_mamba`（`:775`）断言每节点恰好一个状态槽（`:786`
`assert len(x.mamba_value) == 1`）后释放。`match_prefix`（`:475`）、`insert`（`:501`）
也都为状态语义改写。host 卸载变体为 `HiMambaRadixCache`（`hi_mamba_radix_cache.py:93`），
配 `MambaPoolHost`（`:131`）把状态卸载到 CPU。

### 5.6 数据流总览

```
请求到达
  │
  ▼
HybridLinearAttnBackend 按 layer_id 分派
  ├─ 全注意力层 → 普通 KV cache + RadixCache（逐 token 复用）
  └─ 线性/SSM 层 → MambaPool 固定状态（每请求一槽）
        ├─ prefill：mamba_chunk_scan_combined → 写回末态
        └─ decode：causal_conv1d_update + selective_state_update（单步）
  │
  ▼
前缀复用
  ├─ 全注意力层：RadixCache 逐 token 共享
  └─ 线性层：MambaRadixCache fork_from 复制状态快照
```

---

## 6. 支持的模型

接入 `HybridLinearAttnBackend` 的混合模型包括：

- **Qwen3-Next**（`models/qwen3_next.py`）。
- **MiniMax-M2**（`models/minimax_m2.py`，Lightning Attention）。
- **Kimi-Linear**（`models/kimi_linear.py`，`KimiDeltaAttention` 在 `:167`，与 MLA 混合）。
- **Bailing-MoE-linear**（`models/bailing_moe_linear.py`）。
- **Nemotron-H、Falcon-H1、GraniteMoE-Hybrid、Jet-Nemotron**
  （`nemotron_h.py`、`falcon_h1.py`、`granitemoehybrid.py`、`jet_nemotron.py`），
  这些模型均断言其 attention backend 为 `HybridLinearAttnBackend`。

---

## 7. 局限与权衡

- **长程精确检索弱**：固定大小状态是有损压缩，对「精确召回很久以前的某个具体 token」
  弱于 softmax 注意力。混合架构靠少量全注意力层弥补，是当前主流折中。
- **前缀缓存粒度粗**：Mamba 状态只能整段快照式复用（`fork_from`），无法像逐 token KV
  那样部分命中、灵活拼接，前缀复用收益不如纯注意力模型。
- **实现复杂度高**：双状态（conv + ssm）、分块扫描 vs 单步更新两套 kernel、独立的状态池
  与 radix cache、逐层混合调度，复杂度明显高于统一 KV cache。
- **算子生态分散**：GatedDeltaNet / KDA / Lightning / Mamba2 各有专用 kernel
  与 backend，新架构常需新增对应实现，维护面广。
- **状态并行受限**：状态张量按 `tp_size` 切分（`conv_dim/tp`、`num_heads/tp_size`），
  并行配置受状态维度约束。

---

## 参考实现位置

| 模块 | 文件路径 |
|------|---------|
| 固定状态池 | `python/sglang/srt/mem_cache/memory_pool.py`（`MambaPool`） |
| Mamba2 mixer | `python/sglang/srt/layers/attention/mamba/mamba.py` |
| SSD 扫描 kernel | `python/sglang/srt/layers/attention/mamba/ops/` |
| FLA 线性算子（GDN/KDA） | `python/sglang/srt/layers/attention/fla/` |
| Lightning Attention | `python/sglang/srt/layers/attention/linear/lightning_attn.py` |
| 各架构线性 backend | `python/sglang/srt/layers/attention/linear/{gdn,kda,lightning}_backend.py` |
| 逐层混合调度 | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 状态快照前缀缓存 | `python/sglang/srt/mem_cache/mamba_radix_cache.py` |
| host 卸载变体 | `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py` |
