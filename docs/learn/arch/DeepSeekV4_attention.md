# DeepSeek-V4-Pro 注意力变体（DSv4：滑窗 + CSA + HCA）原理详解

> 本文承接 [`NSA_DSA.md`](NSA_DSA.md)，讲清 SGLang 中 **DeepSeek-V4-Pro（代码中称 DSv4）**
> 注意力的**原理**。核心结论：DSv4 在 V3.2 DSA「闪电索引器 + top-k 单分支」之上，重新引入了
> **压缩**与**滑窗**，做成三路互补的注意力——
>
> - **SWA**（滑动窗口）：看最近的、最精确的局部；
> - **CSA**（Compressed Sparse Attention，压缩稀疏注意力）：**适度压缩 4× + 稀疏选 top-k**，精挑历史重点；
> - **HCA**（Heavy Compression Attention，重度压缩注意力）：**重度压缩 128× + 全部都看**，提供全局兜底。
>
> 三路再加一个 sink（注意力汇聚点）在同一个 softmax 里融合。CSA / HCA 是 DSv4 代码里的**官方术语**
> （`dsv4/compressor.py:86`：*"DSV4 supports CSA(4x) and HCA(128x) only"*），分别对应代码中以
> `c4_` / `c128_` 前缀出现的两套缓存。
>
> 本文先讲原理（§1–§5），把代码细节集中到「实现备注」（§6）与参考表（§10）。所有涉及代码的
> 描述都附 `文件路径:行号`，凡属论文层面的对照会明确标注。

> **实现口径校正（适用于全文）**：当前代码按层选择压缩比，单层实际是 **SWA + CSA + sink**
> 或 **SWA + HCA + sink**，并非三路同时进入同一层（`deepseek_v4_backend.py:1696–1704`）。
> 下文 §5 的表述、§5.5 的结构图与附录 A 只能作为三类信息的合并示意，不能当作真实单层计算图；
> 新增的 §3.1（生成机制）、§3.2 / §4.1（具体例子）与 §3.3 / §4.2（结构图）分别按实际 CSA / HCA 层展开。

## 目录

1. [演进：从 V3.2 单分支回到多路](#1-演进从-v32-单分支回到多路)
2. [为什么要分级压缩：单一策略的两难](#2-为什么要分级压缩单一策略的两难)
3. [CSA：压缩稀疏注意力的原理](#3-csa压缩稀疏注意力的原理)
4. [HCA：重度压缩注意力的原理](#4-hca重度压缩注意力的原理)
5. [三路 + sink 如何在一个 softmax 里融合](#5-三路--sink-如何在一个-softmax-里融合)
6. [实现备注（代码层面）](#6-实现备注代码层面)
7. [与 V3.2 DSA / 原始 NSA 的对照](#7-与-v32-dsa--原始-nsa-的对照)
8. [注意力之外的 DSv4 架构差异](#8-注意力之外的-dsv4-架构差异)
9. [局限与权衡](#9-局限与权衡)
10. [参考实现位置](#10-参考实现位置)
11. [附录 A：DSv4 多路注意力数值示例](#附录-adsv4-多路注意力数值示例)

---

## 1. 演进：从 V3.2 单分支回到多路

回顾 `NSA_DSA.md`：原始 NSA 论文设计了「压缩 + 选择 + 滑窗」三分支；DeepSeek-V3.2 的 DSA 把它
砍成单分支——只保留「闪电索引器打分 + top-k 选择」，去掉压缩与滑窗。

DSv4 又把压缩与滑窗加了回来，形成 **SWA + CSA + HCA** 三类信息。与原始 NSA 的相似在于「信息构成」
重新覆盖了局部 / 重点 / 全局三类；不同在于 DSv4 不做三分支门控加权，而是把参与该层的几路 key 拼进
**同一个 softmax** 一次算完（§5）。注意这三类信息**不共存于同一层**：每层按 `compress_ratios` 选择
启用 CSA 还是 HCA（`model_config.py:154`），单层实为 **SWA + CSA** 或 **SWA + HCA** 两路。

---

## 2. 为什么要分级压缩：单一策略的两难

长上下文 decode 的根本矛盾是：**历史 token 太多，全看太贵，但又不能漏掉重要的。** 若只用一种
压缩策略，必然陷入两难：

- **压得轻**（如 4:1）：细节保留得好，但压完 token 数仍多，全看依然贵；要降本就得 top-k 截断，
  而一旦截断，**没被选中的区域就彻底看不见了**。
- **压得重**（如 128:1）：token 数少到可以全看，但 128 个糊成 1 个，**细节全丢**，只剩大概轮廓。

DSv4 的解法是不二选一，而是让两级压缩各司其职，再用滑窗补最近的精确信息：**CSA 负责"重点看清"，
HCA 负责"全局不漏"，SWA 负责"最近最准"。**

---

## 3. CSA：压缩稀疏注意力的原理

> 关键词：**适度压缩（4×）+ 精挑细选（top-k）**

1. **适度压缩**：每新增 4 个 token，生成 1 个「压缩 token」。聚合不是简单平均，而是按学习到的
   分数做 softmax 加权池化。CSA 的压缩窗口是**重叠滑动**的：步长为 4，除序列起点外，每个输出
   覆盖前后两组、共 8 个 token（`dsv4/compress_hip.py:78–86`）。所以 **4:1 指输出数量缩减比例，
   不是每个输出只看 4 个输入**；相邻窗口交叠，避免硬切边界丢信息。

2. **稀疏选择**：用轻量的「闪电索引器」给这些 4× 压缩 token 快速打分，只挑出最重要的
   top-512（或 1024）个，**只在被选中的 token 上做全精度注意力**。

3. **角色**：CSA 是**「重点检索」**——在保留细节的粒度上，精准地把历史里真正相关的几段拎出来
   看清楚，对应原始 NSA 的「选择分支」。

代价：被 top-k 截断的区域看不到。这正是需要 HCA 兜底的原因。

### 3.1 压缩 token 是怎么产生的（CSA / HCA 通用）

CSA 与 HCA 共用同一个压缩器（`Compressor`，`dsv4/compressor.py:335`），差别只在窗口长度与是否重叠。
生成一个压缩 token 分四步：

**第 1 步：每个 token 投影出「待聚合的 kv」与「池化分数」。** 隐藏状态 $x_i$ 过一个线性层
`wkv_gate`（`dsv4/compressor.py:366–373`），得到 $2\,\text{coff}\,d$ 维输出，并按通道拆成两半
（$d$ 为 head dim，$\text{coff}=2$ 仅在窗口重叠时取，否则为 $1$）：

- $z_i$：待聚合的 **kv 分量**（缓冲区布局 `kv_score[..., :coff·d]`）；
- $g_i$：**池化分数**（布局 `kv_score[..., coff·d:]`）。

二者都保留在同一块缓存行里，由 `KVAndScore` 的 `kv` / `score` 属性切分
（`mem_cache/deepseek_v4_compress_state.py:26–31`）。

再给分数加一个**可学习的位置偏置** $\mathrm{ape}$：

$$
g_i \leftarrow g_i + a_{\mathrm{pos}(i)}
$$

它随 token 在窗口内的位置变化，使同一窗口内不同位置的先验重要性不同（`dsv4/compressor.py:361–364`）。

**第 2 步：在窗口内 softmax 归一化并加权求和。** 取窗口 $W$（长度 $=\text{ratio}$）：

$$
p_i(d)=\frac{e^{g_i(d)}}{\sum_{j\in W}e^{g_j(d)}},\qquad
\mathrm{comp}_W(d)=\sum_{i\in W} p_i(d)\, z_i(d)
$$

每个通道各自归一化、各自加权，所以同一窗口不同通道的 $p_i$ 并不相同。实现即
`fused_softmax_pool_triton`（`python/sglang/kernels/ops/attention/deepseek_v4_rope.py:507`）。
这就是它和「组内平均」的本质区别：**权重是可学的、逐通道的**。

**第 3 步：后处理。** 池化结果依次经过 **RMSNorm → 独立 RoPE**（压缩 token 用
`compress_rope_theta=40000`，位置取窗口对应位置）**→ 可选 Hadamard 旋转**
（`rotate_activation`，为 FP4 / 量化做准备），得到最终可写入缓存的压缩 token
（`dsv4/compress_hip.py:229–237`、`dsv4/compressor.py:154–161`）。

**第 4 步：窗口怎么滑（CSA 与 HCA 的唯一差别）。**

| | CSA（`ratio=4`） | HCA（`ratio=128`） |
|---|---|---|
| 窗口 | **重叠滑动**，步长 4，每个输出覆盖前后两组共 8 个 token | **不重叠分块**，每块严格 128 个 |
| $\text{coff}$ | $2$（前后两组各占一套通道） | $1$ |
| 产出时机 | 每凑满 4 个新 token 产一个 | 每凑满 128 个才产一个；不足一块的尾部留在状态池等后续 token |

**一个容易混淆的点：池化分数 ≠ 索引分数。** 同一层里压缩器会出现两次
（`dsv4/compressor.py:192–219`）：

- **注意力压缩器**：产物写入 `extra_k_cache`（`c4_` / `c128_`），供**主注意力**读取；
- **索引器压缩器**（仅 CSA 层，`ratio=4`）：产物写入**索引器 key**，供**闪电索引器**打分。

两者用同一套池化公式，但**各自独立投影、独立缓存**。所以索引器「看到的摘要」与主注意力
「读到的摘要」不是同一份张量——这也是 §3.2 第 3 步的索引分数无法直接拿来当注意力权重的原因。

### 3.2 具体例子：16 个 token 如何压缩、打分、选中

设当前 query 的因果可见前缀为 $x_0,\ldots,x_{15}$，共 $n=16$ 个 token。
这里演示一个 CSA 层（`compress_ratio=4`）：**保留真实的压缩步长与重叠范围**，但为了便于展示，
把 top-k 缩成 $k=2$、SWA 窗口缩成 $w=4$；这两个小参数仅用于教学，不是实际运行配置。

**第 1 步：每前进 4 个位置，形成一个压缩 token。**

| 压缩 token | 覆盖的原始位置 | 何时生成 |
|-------------|----------------|----------|
| $a_0$ | $0\!:\!3$ | 处理完位置 $3$；前一组不存在，其分数置 $-\infty$ 屏蔽 |
| $a_1$ | $0\!:\!7$ | 处理完位置 $7$ |
| $a_2$ | $4\!:\!11$ | 处理完位置 $11$ |
| $a_3$ | $8\!:\!15$ | 处理完位置 $15$ |

表中区间均包含两端。“覆盖位置”指参与池化的位置，不是主注意力会去读的原始位置。最终是
$16/4=4$ 个压缩 token；例如 $a_1$ 与 $a_2$ 共享位置 $4\!:\!7$，这就是 overlap，
而不是把序列切成互不相交的四段。

**第 2 步：动手生成一个压缩 token（按 §3.1 的四步走）。** 以 $a_1$（覆盖位置 $0\!:\!7$）的某一个通道为例。

*2a 投影。* 该层 `wkv_gate` 对这 8 个 token 各输出一份「待聚合 kv 分量」$z_i$ 与一份「池化分数」$g_i$；
为便于手算，下面只看其中一个通道、且先把位置偏置设为 $0$（真实中重叠窗口的前后两组各有自己的投影
通道，见 §3.1）：

| 位置 $i$ | $0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ |
|---|---|---|---|---|---|---|---|---|
| $z_i$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ | $8$ |
| $g_i$ | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ | $\ln 9$ |

*2b softmax 归一化（逐通道）。*

$$
p_i=\frac{e^{g_i}}{\sum_{j=0}^{7}e^{g_j}},\qquad
p=\left[\frac1{16},\frac1{16},\frac1{16},\frac1{16},\frac1{16},\frac1{16},\frac1{16},\frac9{16}\right].
$$

*2c 加权求和。*

$$
\widetilde a_1^{(d)}=\sum_{i=0}^{7}p_i z_i
=\frac{1+2+3+4+5+6+7+9\times8}{16}=6.25.
$$

*2d 后处理。* RMSNorm → 独立 RoPE → （可选）Hadamard 旋转，得到真正写进 `c4_` 缓存的 $a_1$。

对比一下：普通平均只有 $4.5$。上面算出的 $6.25$ 只是 **RMSNorm、RoPE 之前的一维池化值**，不是最终
压缩 KV；这里为简便取 $\mathrm{ape}=0$，真实中它会抬高靠近当前的 token 的分数（§3.1 第 1 步）。
$a_0,a_2,a_3$ 同理各自产出一个压缩 token。

**第 3 步：当前 query 用索引器在 4 个压缩位置中选 2 个。** 假设索引器给出的分数如下：

| 压缩位置 | $a_0$ | $a_1$ | $a_2$ | $a_3$ |
|----------|---------|---------|---------|---------|
| 索引分数 | $0.2$ | $2.8$ | $0.6$ | $1.7$ |
| top-2 结果 | 不选 | **选中** | 不选 | **选中** |

这些是演示用的索引分数，**不是上一步的池化分数，也不是最终注意力权重**。索引器有自己独立的压缩
表示；它返回位置 $\{1,3\}$，主注意力再读取同位置的压缩 KV（`dsv4/compressor.py:163–218`）。
换一个 query，选中的位置可能改变，但已有压缩表示不需要因此重新生成。

**第 4 步：在选中的压缩 KV 与滑窗 KV 上计算主注意力。** 本例提供的 KV 槽为：

$$
\underbrace{\{x_{12},x_{13},x_{14},x_{15}\}}_{\text{SWA：4 个原始位置}}
\quad\cup\quad
\underbrace{\{a_1,a_3\}}_{\text{CSA：2 个压缩位置}}.
$$

主注意力用自己的 query/key **重新计算分数**，将这 6 个 KV 槽与 sink 的 logit 共同归一化。
选中 $a_1$ 是读取 **1 个压缩表示**，不是再展开、读取位置 $0\!:\!7$ 的 8 份原始 KV；
$a_3$ 与滑窗覆盖范围重叠也是允许的。当前代码在 CSA 层组合的是 **SWA + CSA + sink**
（`deepseek_v4_backend.py:1696–1704`）。

**一句话记住：先把 16 个位置压成 4 个摘要，再按当前 query 选 2 个摘要；不是选中摘要后回读整段原文。**

### 3.3 CSA 结构图

**（a）数据流**

```mermaid
flowchart TB
    H["hidden_states&nbsp;(B, T, hidden)"]:::io
    Q["当前 query q"]:::io
    QIDX["索引 query q_idx（FP4、低维）"]:::io

    WG["wkv_gate 线性投影<br/>输出 2·coff·d 维"]:::core

    subgraph CMP["压缩器 Compressor（ratio = 4、overlap）"]
        direction TB
        Z["kv 分量 z_i"]:::kv
        G["池化分数 g_i（+ 可学习位置偏置 ape）"]:::sc
        POOL["逐通道 softmax 加权池化<br/>窗口 = 8（前一组 + 后一组，重叠滑动）"]:::cmp
        POST["RMSNorm → 独立 RoPE →（可选）Hadamard"]:::cmp
    end

    AC["注意力压缩器产物<br/>写 c4_（每 4 个位置 1 个压缩 token）"]:::cache
    IC["索引器压缩器产物<br/>另一套投影、独立缓存"]:::cache
    IDX["闪电索引器打分 + top-k（512 / 1024）"]:::sel
    SWA["SWA 路：最近 128 个原始 KV（不压缩）"]:::swa
    SINK["attn_sink（只占 logit 槽，不贡献 value）"]:::attn
    ATTN["统一 softmax（单个 FlashMLA kernel）<br/>logits = [ SWA | 选中的 c4 | attn_sink ]"]:::attn
    OUT["输出投影 + Output LoRA"]:::core

    H --> WG
    WG -- "kv 分量" --> Z
    WG -- "池化分数" --> G
    Z --> POOL
    G --> POOL --> POST
    POST --> AC
    POST -. "同一套池化公式" .-> IC
    IC --> IDX
    QIDX --> IDX
    AC -. "按选中位置读取" .-> ATTN
    IDX -- "选中位置（比如 1 与 3）" --> ATTN
    Q --> ATTN
    SWA --> ATTN
    SINK --> ATTN
    ATTN --> OUT

    classDef io fill:#eef2ff,stroke:#6366f1,color:#1e1b4b;
    classDef core fill:#fff7ed,stroke:#f97316,color:#7c2d12;
    classDef kv fill:#e0f2fe,stroke:#0284c7,color:#075985;
    classDef sc fill:#fae8ff,stroke:#c026d3,color:#701a75;
    classDef cmp fill:#fef9c3,stroke:#ca8a04,color:#713f12;
    classDef cache fill:#f1f5f9,stroke:#94a3b8,color:#334155;
    classDef sel fill:#ecfdf5,stroke:#10b981,color:#064e3b;
    classDef swa fill:#e0f2fe,stroke:#0284c7,color:#075985;
    classDef attn fill:#ecfdf5,stroke:#10b981,color:#064e3b;
```

**（b）重叠窗口的覆盖**（16 个位置 → 4 个压缩 token，● 表示参与池化）

| 压缩 token | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 长度 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $a_0$ | ● | ● | ● | ● | | | | | | | | | | | | | 4 |
| $a_1$ | ● | ● | ● | ● | ● | ● | ● | ● | | | | | | | | | 8 |
| $a_2$ | | | | | ● | ● | ● | ● | ● | ● | ● | ● | | | | | 8 |
| $a_3$ | | | | | | | | | ● | ● | ● | ● | ● | ● | ● | ● | 8 |

读图：**每行错开 4 格（步长 = 4），每行跨度 8 格（窗口 = 8）**，所以相邻两行有 4 格重叠——这就是
overlap。$a_0$ 只有 4 格是因为它前面不存在“前一组”（置 $-\infty$ 屏蔽）。

---

## 4. HCA：重度压缩注意力的原理

> 关键词：**重度压缩（128×）+ 全部都看**

1. **重度压缩**：每 128 个 token 聚合成 1 个，压缩比极高。与 CSA 不同，HCA 的窗口是**不重叠的
   分块**（non-overlap，`dsv4/compressor.py:356–359` 中仅 CSA 启用 overlap），
   干净地把整段历史切成少量大块。

2. **不选择、全稠密**：因为 128:1 压完后 token 数已极少（128K 上下文只剩约 1K 个），**不需要
   top-k，直接全看**，开销依然很低。

3. **角色**：HCA 是**「全局概览」**——用极少量重度压缩 token 覆盖**整个**历史，保证「哪里都看
   得到一点」，不会因 CSA 的 top-k 截断而漏掉某个区域，对应原始 NSA 的「压缩分支」。

代价：粒度粗、细节糊；但它的任务本就是兜底全局，不负责看细节。

### 4.1 具体例子：400 个 token 如何压成 3 个完整块并全部关注

设当前 query 的因果可见前缀为 $x_0,\ldots,x_{399}$，共 $n=400$ 个 token。
这里演示一个 HCA 层（`compress_ratio=128`），保留真实的压缩比与 $w=128$ 的滑窗大小；压缩 token 的
生成机制与 §3.1 完全相同，只是窗口不重叠、更宽。

**第 1 步：按 128 个 token 不重叠分块，只提交已经完成的块。**

| 压缩 token | 覆盖的原始位置 | 当前状态 |
|-------------|----------------|----------|
| $c_0$ | $0\!:\!127$ | 已完成，参与注意力 |
| $c_1$ | $128\!:\!255$ | 已完成，参与注意力 |
| $c_2$ | $256\!:\!383$ | 已完成，参与注意力 |
| 下一块 $c_3$ | 目前只有 $384\!:\!399$ | 仅积累了 16 个 token，尚未生成完整压缩 token |

因此当前有效的 HCA token 数是：

$$
N_{\mathrm{HCA}}=\left\lfloor\frac{400}{128}\right\rfloor=3.
$$

这里必须**向下取整**，不是凑出第 4 个块，也不能提前使用未来位置。尾部 $384\!:\!399$ 保留在
压缩状态中等待后续 token，并由 SWA 提供精确信息（`dsv4/compress_hip.py:95–102,163–186`；
`python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py:52–80`）。

**第 2 步：每块仍做 §3.1 的加权池化，不是把 128 个位置简单平均。** 以 $c_0$ 的一个维度为例，设该
窗口内 `wkv_gate` 输出的「待聚合 kv 分量」与「池化分数」为：前 $127$ 个位置 $z_i=2$、$g_i=0$，
最后一个位置 $z_{127}=10$、$g_{127}=\ln127$（同样先不计 $\mathrm{ape}$）。softmax 后，
最后一个位置与其余 $127$ 个位置合计各占一半：

$$
p_{127}=\frac{127}{127+127}=\frac12,\qquad
\sum_{i=0}^{126}p_i=\frac12,
$$

$$
\widetilde c_0^{(d)}=\frac12\times2+\frac12\times10=6.
$$

若普通平均，结果只是 $(127\times2+10)/128=2.0625$。这说明重度压缩仍能通过学习到的分数突出某些
信息，但**不能保证保留所有细节**。同样，这只是 RMSNorm、RoPE 之前的一维池化演示
（生成机制见 §3.1）。

**第 3 步：不做 top-k，三个完整压缩 token 全部进入主注意力。** 本例提供的 KV 槽为：

$$
\underbrace{\{x_{272},\ldots,x_{399}\}}_{\text{SWA：最近 128 个原始位置}}
\quad\cup\quad
\underbrace{\{c_0,c_1,c_2\}}_{\text{HCA：全部 3 个完整压缩位置}}.
$$

共 $128+3=131$ 个 KV 槽，再加一个不贡献 value 的 sink logit，共同做 softmax；**没有索引器筛选**。
“全部看”不等于“平均看”：当前 query 仍会通过主注意力分数，为 $c_0,c_1,c_2$ 分配不同权重。
当前代码在 HCA 层组合的是 **SWA + HCA + sink**（`deepseek_v4_backend.py:1696–1704`）。

**一句话记住：CSA 是“很多较细摘要中挑重点”，HCA 是“少量较粗摘要全部看”；未满 128 的最新尾部交给滑窗。**

### 4.2 HCA 结构图

**（a）数据流**

```mermaid
flowchart TB
    H["hidden_states&nbsp;(B, T, hidden)"]:::io
    Q["当前 query q"]:::io

    WG["wkv_gate 线性投影<br/>输出 2·d 维（coff = 1）"]:::core

    subgraph CMP["压缩器 Compressor（ratio = 128、non-overlap）"]
        direction TB
        Z["kv 分量 z_i"]:::kv
        G["池化分数 g_i（+ 可学习位置偏置 ape）"]:::sc
        POOL["逐通道 softmax 加权池化<br/>窗口 = 128（不重叠分块）"]:::cmp
        POST["RMSNorm → 独立 RoPE →（可选）Hadamard"]:::cmp
    end

    FULL{"凑满 128 个？"}:::gate
    CACHE["写 c128_ 缓存<br/>（c_0, c_1, c_2 …）"]:::cache
    TAIL["不足一块的尾部<br/>留在压缩状态池，等后续 token"]:::tail
    SWA["SWA 路：最近 128 个原始 KV<br/>（尾部由它提供精确信息）"]:::swa
    SINK["attn_sink"]:::attn
    ATTN["统一 softmax（单个 FlashMLA kernel）<br/>logits = [ SWA | 全部已完成的 c128 | attn_sink ]<br/>无索引器、无 top-k"]:::attn
    OUT["输出投影 + Output LoRA"]:::core

    H --> WG
    WG -- "kv 分量" --> Z
    WG -- "池化分数" --> G
    Z --> POOL
    G --> POOL --> POST --> FULL
    FULL -- "是（每 128 个产 1 个）" --> CACHE
    FULL -- "否" --> TAIL
    CACHE --> ATTN
    Q --> ATTN
    SWA --> ATTN
    SINK --> ATTN
    ATTN --> OUT
    TAIL -. "后续补齐后再产出一块" .-> CACHE

    classDef io fill:#eef2ff,stroke:#6366f1,color:#1e1b4b;
    classDef core fill:#fff7ed,stroke:#f97316,color:#7c2d12;
    classDef kv fill:#e0f2fe,stroke:#0284c7,color:#075985;
    classDef sc fill:#fae8ff,stroke:#c026d3,color:#701a75;
    classDef cmp fill:#fef9c3,stroke:#ca8a04,color:#713f12;
    classDef gate fill:#fff7ed,stroke:#f97316,color:#7c2d12;
    classDef cache fill:#f1f5f9,stroke:#94a3b8,color:#334155;
    classDef tail fill:#fee2e2,stroke:#ef4444,color:#7f1d1d;
    classDef swa fill:#e0f2fe,stroke:#0284c7,color:#075985;
    classDef attn fill:#ecfdf5,stroke:#10b981,color:#064e3b;
```

**（b）不重叠分块与“只提交完整块”**

| 来源 | 覆盖位置（含两端） | 长度 | 是否参与注意力 |
|---|---|---|---|
| $c_0$ | $0$–$127$ | 128 | 是 |
| $c_1$ | $128$–$255$ | 128 | 是 |
| $c_2$ | $256$–$383$ | 128 | 是 |
| 下一块（未产出） | $384$–$399$（当前） | 16 | 否，留在压缩状态池 |
| SWA 路 | $272$–$399$ | 128 | 是（原始 KV，不压缩） |

与 §3.3 的 CSA 覆盖表对比：**HCA 每行的起点、终点都严格对齐 128 的整数倍，行与行零重叠；CSA 每行
错开 4 格、跨度 8 格。** 这导致两个后果：

- HCA 的压缩 token 数就是 $\lfloor n/128\rfloor$，尾部不满一块**永远不产出**；
- CSA 的压缩 token 数是每 4 个位置一个，每个都只“看”8 个位置，**粒度细得多**。

---

## 5. 三路 + sink 如何在一个 softmax 里融合

把三路放在一起，分工就清楚了。类比读一本很厚的书：

| 路 | 压缩 | 选择 | 类比 | 负责 |
|----|------|------|------|------|
| **SWA 滑窗** | 不压 | 最近 128 | 正在读的这几页 | 局部，最清晰 |
| **CSA** | 4×（重叠） | top-512/1024 | 翻回去精读几个关键章节 | 重点，较清晰 |
| **HCA** | 128×（不重叠） | 全看 | 扫一遍全书目录 | 全局，模糊但不漏 |

- **SWA** 保证最近内容看得最清；
- **CSA** 保证重要历史片段被精准放大；
- **HCA** 保证整个历史都有粗略印象、不留盲区。

三者覆盖「局部精确 / 重点细看 / 全局兜底」三个层次，又都很便宜。**关键：DSv4 把参与该层的几路 key
拼进同一个 softmax**（外加一个 sink logit 稳定数值），由一个 FlashMLA kernel 一次算完——而不是
像原始 NSA 那样三分支各自 softmax 再门控加权。数值演示见[附录 A](#附录-adsv4-多路注意力数值示例)。
（再次提醒：真实单层只有 **SWA + CSA** 或 **SWA + HCA** 两路，并非三路齐上，见开头口径校正。）

**复杂度直觉**：一次 decode 的全精度 key 数约为 $\underbrace{128}_{\text{SWA}} +
\underbrace{512}_{\text{CSA top-k}} + \underbrace{n/128}_{\text{HCA}}$。前两项是常数，第三项随
上下文 $n$ 线性但被 128 倍压缩。代价是多一次闪电索引器的廉价打分与压缩器的池化——这就是 DSv4
在保留三类信息的同时，把长上下文 decode 的注意力访存压到近似常数的原理。

---

## 5.5 一个 DSv4 attention block 的完整结构图

把前面 §3–§5 的所有部件串起来，一个完整的 DSv4 注意力块（代码中的 `MQALayer`，
`models/deepseek_v4.py:262`）的数据流如下。整体仍是「MLA 下投影 → 三路取 KV → 统一 softmax →
输出投影」，三路并行准备 key/value，最终拼进同一个 FlashMLA kernel。

```mermaid
flowchart TB
    H["hidden_states&nbsp;(B, T, hidden_size)"]:::io

    subgraph DEC["DecoderLayer 外层（MQALayer 之外）"]
        direction TB
        RN_IN["input RMSNorm"]:::norm
    end

    MLA["MLA 下投影<br/>q = W_q · h&nbsp;&nbsp;|&nbsp;&nbsp;kv_latent = W_kv · h<br/><i>（低秩潜在 KV，三路共享）</i>"]:::core
    ROPE["RoPE 旋转编码<br/>(fused rope)"]:::core

    subgraph KV["按 compress_ratio 解出三路 KV（并行）"]
        direction TB
        SWA["<b>SWA 路</b>&nbsp;ratio=0<br/>最近 window=128 原始 token<br/>不压缩 · 最精确（局部）"]:::swa
        CSA["<b>CSA 路</b>&nbsp;ratio=4<br/>Compressor 4× 重叠压缩<br/>→ C4Indexer(FP4) 打分 → top-k(512/1024)<br/>仅选中 token 取全精度 KV（重点）"]:::csa
        HCA["<b>HCA 路</b>&nbsp;ratio=128<br/>Compressor 128× 不重叠分块<br/>全部都看 · 不做 top-k（全局兜底）"]:::hca
    end

    ATTN["统一 FlashMLA 注意力（一个 kernel、一个 softmax）<br/>logits = [ SWA | CSA | HCA | attn_sink ]<br/><i>sink 只占 logit 槽、不贡献 value</i>"]:::attn
    OUT["输出投影 + Output LoRA<br/>(o_lora_rank=1024, o_groups=8)"]:::core
    Y["attention block 输出<br/><i>→ 回 DecoderLayer：残差相加 → post RMSNorm → MoE</i>"]:::io

    H --> RN_IN --> MLA
    MLA -- "q" --> ROPE
    MLA -- "kv_latent" --> KV
    ROPE -- "q_rope" --> ATTN
    SWA --> ATTN
    CSA --> ATTN
    HCA --> ATTN
    ATTN -- "attn_out" --> OUT --> Y

    classDef io fill:#eef2ff,stroke:#6366f1,color:#1e1b4b;
    classDef core fill:#fff7ed,stroke:#f97316,color:#7c2d12;
    classDef norm fill:#f1f5f9,stroke:#94a3b8,color:#334155;
    classDef attn fill:#ecfdf5,stroke:#10b981,color:#064e3b;
    classDef swa fill:#e0f2fe,stroke:#0284c7,color:#075985;
    classDef csa fill:#fef9c3,stroke:#ca8a04,color:#713f12;
    classDef hca fill:#fae8ff,stroke:#c026d3,color:#701a75;
```

读图要点：

- **共享潜在 KV**：三路并不各自存一份完整 KV，而是共享 MLA 的低秩潜在向量 `kv_latent`，
  各路只是用不同的 `compress_ratio`（0 / 4 / 128）从中解出对应粒度的 key/value（§6 术语映射表）。
- **三路并行、softmax 统一**：虚线框内的 SWA / CSA / HCA 是**并行准备 key/value**，但**不各自
  做 softmax**——它们的 logits 连同 sink 一起进同一个 FlashMLA kernel 归一化（§5、附录 A）。
- **CSA 是唯一带"选择"的路**：只有它经过 `C4Indexer` 打分 + top-k 截断；SWA 固定取最近窗口、
  HCA 压完直接全看，都没有选择步骤。
- **block 边界**：本图聚焦注意力块本身（`MQALayer`）；其外层的 input/post RMSNorm、残差相加与
  MoE 属于 `DeepseekV4DecoderLayer`（`models/deepseek_v4.py:1016`），图中以灰字标注衔接关系。

---

## 6. 实现备注（代码层面）

> 本节是给读代码的人看的索引，原理已在 §2–§5 讲完，可跳过。

**术语映射**：注意力前向 `forward`（`deepseek_v4_backend.py:1165`）接收
`compress_ratio ∈ {0, 4, 128}`，对应三套缓存（`metadata.py:44` 定义 `c4_`/`c128_` 前缀）：

| 本文术语 | `compress_ratio` | 代码前缀 | 页大小（`page_size=256` 下） |
|---------|-----------------|---------|------|
| SWA | 0（C1，原始 token） | `swa_` / `c1_` | 256 |
| CSA | 4 | `c4_` | 64 |
| HCA | 128 | `c128_` | 2 |

- **元数据**：`DSV4AttnMetadata`（`:106`）按级别维护 `c1_/c4_/c128_flashmla_metadata`（`:133`），
  由 `get_flashmla_metadata(compress_ratio)`（`:141`）分派。`page_size==256` 是硬约束
  （`metadata.py:142`）；CSA top-k 限定 `c4_sparse_topk in (512, 1024)`（`:309`）。
- **压缩器**：`dsv4/compressor.py`、`compressor_v2.py`、`fused_compress_triton.py`、
  `compress_hip.py`。online softmax-pool 见 `compressor_v2.py:50`；压缩 token 用独立 RoPE
  （`compress_rope_theta=40000`，`configs/deepseek_v4.py:104`），RoPE 位置
  `= seq_len - compress_ratio`（`compressor_v2.py:317`）。**CSA indexer 只挂在
  `compress_ratio==4` 的层**（`model_config.py:154`），逐层压缩比由 `compress_ratios` 配置决定。
- **闪电索引器（FP4）**：DSv4 把索引器 KV 从 V3.2 的 FP8 进一步压到 **FP4（E2M1）+ UE8M0 微缩放**
  （`dsv4/fp4_indexer.py`，128 维→64 字节 + 4 字节 scale；开关 `--enable-deepseek-v4-fp4-indexer`，
  `server_args.py:819`）。打分走 FP8 paged MQA logits + `topk_transform_512`（`indexer.py`）。
- **统一前向**：`forward`（`:1165`）一次把 `swa_k_cache` + `extra_k_cache`（CSA 稀疏 / HCA 全部）
  + `attn_sink`（`:1246` 断言必需）喂给 `flash_mla_with_kvcache`（`:1292`）；大 batch 变长 prefill
  走 `_forward_prefill_sparse`（`:1314`）+ `flash_mla_sparse_fwd`。索引最后一维需对齐 64（`:1251`）。
- **decode 索引打包**：`dsv4/unified_kv_kernels/paged_decode_indices.py` 把三路统一打包为
  `swa_/csa_/hca_indices`（ragged，每路长度分别为 SWA 窗口、`min(committed, index_topk)`、HCA 全部）。
- **识别与注册**：`is_deepseek_v4`（`model_config.py:118`，架构白名单，不依赖 `index_topk`）；
  backend 注册名 `"dsv4"`（`attention_registry.py:127`，CUDA→`DeepseekV4AttnBackend`，
  HIP→`DeepseekV4HipRadixBackend`），废弃别名 `"compressed"`（`server_args.py:188`）。

---

## 7. 与 V3.2 DSA / 原始 NSA 的对照

| 维度 | 原始 NSA（论文） | V3.2 DSA（SGLang `nsa/`） | **DSv4（SGLang `dsv4/`）** |
|------|-----------------|--------------------------|---------------------------|
| 滑窗（局部） | 有 | **无** | **有，SWA window=128** |
| 选择（重点） | 有（top-k 块） | 有（top-k 位置，page_size=1） | **有，CSA = 4× 压缩 + top-512/1024** |
| 压缩（全局） | 有（粗粒度块） | **无** | **有，HCA = 128× 重度压缩，稠密** |
| 融合方式 | 三分支**门控加权求和** | 单分支（无融合） | **单 FlashMLA kernel 内统一 softmax**（含 sink） |
| 索引器精度 | — | FP8 | **FP4（E2M1）+ UE8M0 微缩放** |
| 压缩粒度 | 单一块大小 | — | **两级：CSA 4× 与 HCA 128×** |
| 与 MLA 关系 | 独立设计 | 叠加在 MLA 之上 | 叠加在 MLA 之上（沿用潜在 KV） |

要点：DSv4 在「信息构成」上回归 NSA 三类（局部/重点/全局），但**融合方式仍是 DSA 式的单 kernel
稀疏注意力**，而非论文的门控加权——所以它既不是纯 NSA，也不是纯 V3.2 DSA。

---

## 8. 注意力之外的 DSv4 架构差异

为完整理解 DSv4-Pro，以下是与注意力强耦合但严格说不属于「注意力分支」的几处改动（简述）：

- **Output LoRA**：配置含 `o_lora_rank=1024`、`o_groups=8`（`configs/deepseek_v4.py:100`），
  即输出投影也做低秩分组（V3 系列只有 `q_lora_rank`/`kv_lora_rank`）；头维也变大
  （`qk_nope_head_dim=448`、`v_head_dim=512`）。
- **MHC / hc_head（LM-head mixer）**：`layers/mhc_head.py` 把 `(T, hc_mult, hidden_size)` 的
  `hc_mult` 个候选经 RMSNorm + Linear + sigmoid 门控加权求和成 `(T, hidden_size)`，每次前向在最后
  一个 PP rank 触发一次。超参 `n_hash_layers=3`、`hc_mult=4`、`hc_sinkhorn_iters=20`。
- **MoE 路由差异**：`scoring_func="sqrtsoftplus"`（`configs/deepseek_v4.py:90`），训练时**不做
  group limiting**；`model_config.py:274`-`283` 强制 `topk_group == n_group` 走未分组的
  sqrtsoftplus 路径（注释警告：分组实现只支持 sigmoid 打分，误入会静默损坏专家权重）。
- **FP4/MXFP4 专家**：`is_fp4_experts`（`model_config.py:260`）经 `try_detect_fp4_experts`
  探测权重 dtype，自动判定路由专家是 MXFP4 打包还是转换后的 FP8。

---

## 9. 局限与权衡

- **强依赖 MLA + FlashMLA**：统一前向建立在 `flash_mla_with_kvcache` / `flash_mla_sparse_fwd`
  之上，SM120 另有专门路径（`:1272`）。脱离 MLA 潜在 KV 无法使用。
- **多套页语义并存**：`page_size==256` 硬编码，索引最后一维必须对齐 64；SWA=256、CSA=64、
  HCA=2、稀疏 page_size=1 多套语义共存，对元数据正确性要求极高。
- **元数据复杂度高**：相比 V3.2 单分支，DSv4 要同时维护三路 FlashMLA 调度元数据、稀疏页索引、
  压缩计划（`DSV4AttnMetadata` 字段众多，`:106`-`:328`），CUDA graph 切换逻辑也更繁重。
- **CSA 非全层**：只有 `compress_ratio==4` 的层做稀疏选择（`model_config.py:154`），其余层靠
  SWA + HCA 覆盖，层间行为不一致，调试需对照 `compress_ratios`。
- **必须为 DSv4 训练**：索引器、压缩器、MHC head、sqrtsoftplus 路由都是可学习/训练耦合模块，
  不能免训练加到现成模型上。
- **代码新、形状假设多**：大量 `assert`（如 `c4_sparse_topk in (512,1024)`、`page_size==256`）
  表明实现对配置组合敏感，超出验证过的组合可能直接断言失败。

---

## 10. 参考实现位置

| 模块 | 文件路径 |
|------|---------|
| 模型识别 / 超参 / FP4 专家 | `python/sglang/srt/configs/model_config.py`（`is_deepseek_v4` `:118`） |
| DSv4 配置类 / FP4 探测 | `python/sglang/srt/configs/deepseek_v4.py` |
| backend 总入口与统一前向（CUDA） | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| backend（HIP / ROCm） | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` |
| 压缩器（CSA/HCA，online softmax-pool） | `dsv4/compressor.py`、`compressor_v2.py`、`fused_compress_triton.py`、`compress_hip.py` |
| FP4 闪电索引器 | `dsv4/fp4_indexer.py`、`dsv4/indexer.py` |
| 三路索引统一打包 | `dsv4/unified_kv_kernels/paged_decode_indices.py` |
| 元数据 / 术语定义 | `dsv4/metadata.py`、`dsv4/metadata_kernel.py` |
| 索引器 KV 量化 / 反量化 | `dsv4/quant_k_cache.py`、`dsv4/dequant_k_cache.py`、`dsv4/index_buf_accessor.py` |
| MHC / hc_head（LM-head mixer） | `python/sglang/srt/layers/mhc_head.py` |
| backend 注册 | `python/sglang/srt/layers/attention/attention_registry.py`（`:127`） |
| 启动参数 / 默认值 hook | `python/sglang/srt/server_args.py`、`python/sglang/srt/arg_groups/deepseek_v4_hook.py` |

---

## 附录 A：DSv4 多路注意力数值示例

为直观理解 §5 的「SWA + CSA + HCA 三类信息如何汇进**同一个 softmax**」，下面用一个**可以从头算到尾**的
极小规模例子完整演示。设当前 decode 一个新 query，历史序列长度 $n = 8$（位置 $0\dots7$），单头，
头维 $d = 2$。为方便手算，本附录所有数都取得很整。

> 提醒：按开头「实现口径校正」，真实单层只启用 **SWA + CSA** 或 **SWA + HCA**；本附录把三路放在一起
> 只为一次看全三类信息。压缩 token 的真实生成规则见 §3.1，单独看 CSA / HCA 的例子见 §3.2 / §4.1。

### A.0 原始输入数据

历史 8 个 token 的（全精度）key / value 向量如下——这是后续一切计算的**唯一原始数据**，三路都从它派生：

| 位置 $i$ | $0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|
| key $k_i$ | $[0.2,0.0]$ | $[0.4,0.2]$ | $[0.1,0.5]$ | $[0.1,0.7]$ | $[0.3,0.6]$ | $[0.2,0.9]$ | $[1.0,0.0]$ | $[0.7,0.7]$ |
| value $v_i$ | $[2,1]$ | $[4,2]$ | $[1,6]$ | $[0,8]$ | $[3,7]$ | $[2,9]$ | $[10,0]$ | $[8,8]$ |
| 池化分数 $g_i$ | $0.1$ | $0.2$ | $0.3$ | $0.4$ | $0.5$ | $1.5$ | — | — |

其中 $g_i$ 是 §3.1 第 1 步里 `wkv_gate` 输出的**池化分数**（已含位置偏置 $\mathrm{ape}$），
**只用于生成压缩 token，不参与主注意力打分**；它和 A.3 的索引分数是两回事。位置 $6,7$ 不做压缩
（它们由 SWA 直接贡献原始 KV），所以没有池化分数。

当前 query 取 $q = [1,\ 0]$，缩放因子 $\sqrt d = \sqrt 2 \approx 1.414$。三路分工（与 §5 一致）：

| 路 | 压缩比 | 提供的 key | 说明 |
|----|-------|-----------|------|
| SWA 滑窗 | 0 | 位置 $\{6, 7\}$ 的原始 KV | 最近窗口（真实 window=128，此处缩成 2） |
| CSA 稀疏 | 4 | $0\dots7$ 压成 2 个 token，索引器选 top-1 | 适度压缩 + 精挑 |
| HCA 压缩 | 128 | $0\dots5$ 重度压成 1 个 token $c$ | 重度压缩，全看不选择 |

> 说明：为让三路 key 互不重复、便于区分贡献，本例让 SWA 取最近的 $\{6,7\}$，而 CSA/HCA 只压缩较早的
> $0\dots5$。另外，**本例把压缩比与窗口也都缩到教学尺度**（真实 CSA 步长为 4 且重叠、HCA 必须凑满
> 128 个才产出，§3.1 第 4 步）；真实实现中各路覆盖范围按页对齐、会有重叠。这里只保留“加权池化 +
> 稀疏选择 + 单 softmax 融合”这三个要演示的机制。

### A.1 SWA 路：直接取原始 KV，无需计算

滑窗不压缩，直接把最近窗口内位置 $6,7$ 的原始 key/value 原样拿来：

$$
k_6=[1.0,\ 0.0],\ v_6=[10,\ 0]; \qquad k_7=[0.7,\ 0.7],\ v_7=[8,\ 8].
$$

### A.2 HCA 路：重度压缩（online softmax-pool）的逐步计算

HCA 把位置 $0\dots5$ 这 6 个 token **加权聚合成 1 个**压缩 token $c$，走的就是 §3.1 的流程：`wkv_gate`
给出待聚合 kv 分量与池化分数 $g_i$（即上表），对 $g_i$ 做 **softmax 加权池化**；聚合不是普通平均。

**第 1 步：对池化分数 $g_i$ 做 softmax，得到每个 token 的聚合占比。** 先取指数（保留 3 位小数）：

$$
e^{0.1}=1.105,\ e^{0.2}=1.221,\ e^{0.3}=1.350,\ e^{0.4}=1.492,\ e^{0.5}=1.649,\ e^{1.5}=4.482.
$$

求和：$Z = 1.105+1.221+1.350+1.492+1.649+4.482 = 11.299$。归一化得权重 $p_i = e^{g_i}/Z$：

$$
p_0=0.098,\ p_1=0.108,\ p_2=0.119,\ p_3=0.132,\ p_4=0.146,\ p_5=0.397.
$$

（注意位置 5 权重最大，聚合后它的贡献占比最高，约 $40\%$。）

**第 2 步：用 $p_i$ 加权求和 key/value，得到压缩 token $c$。** 以 value 第 1 维为例：

$$
v_c^{(1)} = 0.098\cdot2 + 0.108\cdot4 + 0.119\cdot1 + 0.132\cdot0 + 0.146\cdot3 + 0.397\cdot2 = 2.07.
$$

同理逐维算完得：

$$
k_c \approx [0.30,\ 0.62], \qquad v_c \approx [2.07,\ 6.50].
$$

这一个 $c$ 就代表了「位置 $0\dots5$ 的全局概貌」——粒度粗，但一个都没漏。注意这里只算了池化值，
真实还要再经 RMSNorm → RoPE →（可选）Hadamard 才写入 `c128_` 缓存（§3.1 第 3 步）。

### A.3 CSA 路：先压缩成 2 个，再用索引器选 top-1

**第 1 步：压缩。** CSA 压缩比为 4，把 $0\dots5$（演示里取 6 个，凑成 2 组）每 3 个压成 1 个压缩 token
（这里把步长与重叠都缩到教学尺度，真实规则见 §3.1 第 4 步、§3.2）：

- $a_0$ 由位置 $\{0,1,2\}$ 聚合，$a_1$ 由位置 $\{3,4,5\}$ 聚合。简单起见这里用**组内均值**近似（真实仍是
  §3.1 的 softmax 加权池化）：

$$
k_{a_0}=\tfrac{1}{3}([0.2,0]+[0.4,0.2]+[0.1,0.5])=[0.233,\ 0.233],\quad v_{a_0}=\tfrac13([2,1]+[4,2]+[1,6])=[2.33,\ 3.00];
$$
$$
k_{a_1}=\tfrac{1}{3}([0.1,0.7]+[0.3,0.6]+[0.2,0.9])=[0.200,\ 0.733],\quad v_{a_1}=\tfrac13([0,8]+[3,7]+[2,9])=[1.67,\ 8.00].
$$

**第 2 步：索引器打分（轻量、低维）。** 闪电索引器用一套**独立的**索引 query $q_{idx}=[2,\ 1]$ 对两个压缩
token 的 key 打分（点积，不缩放）：

$$
s(a_0)=q_{idx}\!\cdot\! k_{a_0}=2\times0.233+1\times0.233=0.699,
$$
$$
s(a_1)=q_{idx}\!\cdot\! k_{a_1}=2\times0.200+1\times0.733=1.133.
$$

**第 3 步：取 top-1。** $s(a_1)=1.133 > s(a_0)=0.699 \Rightarrow$ 选中 $a_1$，丢弃 $a_0$。
（真实 DSv4 中 `index_topk=512`，这里缩成 1。）CSA 最终只贡献一个 key：$k_{a_1}=[0.200,\ 0.733]$，
$v_{a_1}=[1.67,\ 8.00]$。

> 关键：索引打分用的是 $q_{idx}$（索引器的低维 FP4 投影，**只为「在压缩 token 里选谁」**），
> 与下面 A.4 真正算注意力用的 $q$ 是**两套不同的 Q**——这与 V3.2 DSA「索引器轻量打分、选中后才用全精度」
> 一脉相承（见 `NSA_DSA.md` 附录 A）。

### A.4 汇总三路 key/value，并计算注意力打分

现在把三路最终提供的 key/value 收齐，用**注意力 query** $q=[1,0]$ 算打分 $q\!\cdot\!k/\sqrt2$：

| 来源 | key $k$ | value $v$ | 点积 $q\!\cdot\!k$ | 打分 $q\!\cdot\!k/\sqrt2$ |
|------|---------|-----------|--------|------|
| SWA 位 6 | $[1.0,\ 0.0]$ | $[10,\ 0]$ | $1.000$ | $0.707$ |
| SWA 位 7 | $[0.7,\ 0.7]$ | $[8,\ 8]$ | $0.700$ | $0.495$ |
| HCA 压缩 $c$ | $[0.30,\ 0.62]$ | $[2.07,\ 6.50]$ | $0.300$ | $0.212$ |
| CSA 选中 $a_1$ | $[0.20,\ 0.73]$ | $[1.67,\ 8.00]$ | $0.200$ | $0.141$ |
| sink | （仅占一个 logit 槽，无 value） | — | — | $0$（设其 logit=0） |

打分只用了 key 的第 1 维（因为 $q=[1,0]$），例如 HCA：$q\!\cdot\!k_c = 1\times0.30+0\times0.62 = 0.30$，
除以 $\sqrt2$ 得 $0.212$。

### A.5 统一 softmax（关键：三路 + sink 一起归一化）

**第 1 步：对 5 个 logit 取指数。**

$$
e^{0.707}=2.028,\ e^{0.495}=1.640,\ e^{0.212}=1.236,\ e^{0.141}=1.151,\ e^{0}=1.000.
$$

**第 2 步：求和。** $Z = 2.028+1.640+1.236+1.151+1.000 = 7.055$。

**第 3 步：归一化得注意力权重 $\alpha$。**

$$
\alpha = \Big[\tfrac{2.028}{7.055},\ \tfrac{1.640}{7.055},\ \tfrac{1.236}{7.055},\ \tfrac{1.151}{7.055},\ \tfrac{1.000}{7.055}\Big]
= [0.287,\ 0.232,\ 0.175,\ 0.163,\ 0.142].
$$

最后一个 $0.142$ 是 sink 的权重——它**只参与归一化、稀释其他权重以稳定数值**，但**不贡献 value**。

### A.6 加权求和得到输出

用前 4 个权重对各自 value 加权求和（sink 跳过）：

$$
\text{out} = 0.287[10,0] + 0.232[8,8] + 0.175[2.07,6.50] + 0.163[1.67,8.00].
$$

逐维计算：

$$
\text{out}^{(1)} = 0.287\cdot10 + 0.232\cdot8 + 0.175\cdot2.07 + 0.163\cdot1.67 = 2.87+1.86+0.36+0.27 = 5.36,
$$
$$
\text{out}^{(2)} = 0.287\cdot0 + 0.232\cdot8 + 0.175\cdot6.50 + 0.163\cdot8.00 = 0+1.86+1.14+1.30 = 4.30.
$$

$$
\boxed{\ \text{out} \approx [5.36,\ 4.30]\ }
$$

可以看到结果由「最近、打分最高的 SWA 位 6」主导（贡献了 $2.87$ 的第 1 维），同时 CSA/HCA 也把历史信息
按权重掺了进来——这正是三路互补、单 softmax 融合的效果。

### A.7 与原始 NSA 的对比

差异一目了然：NSA 让三分支**各自做 softmax 再门控加权**

$$
o = g_w\, o_{win} + g_c\, o_{cmp} + g_s\, o_{sel},
$$

需要算 3 次 softmax 再学 3 个门控系数 $g$；而 DSv4 把三路 key 拼进**同一个 softmax**（外加 sink logit
稳定数值），由 FlashMLA **一次算完**（即上面 A.5–A.6 的单次归一化），既省了门控参数，也省了多次 softmax。
