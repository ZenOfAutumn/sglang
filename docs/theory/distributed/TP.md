# 张量并行（Tensor Parallelism, TP）原理详解

> 本文系统介绍张量并行的数学原理、行并行 / 列并行两种分块矩阵乘方式、
> Transformer 各层（MLP、Attention、Embedding、LM Head）如何被切分到多张 GPU 上、
> 前向 / 反向中的集合通信（all-reduce / all-gather），并给出可手算验证的数值示例，
> 最后对应到 SGLang 中 `linear.py`、`vocab_parallel_embedding.py` 与 `distributed/` 的真实实现。

## 目录

1. [为什么需要张量并行](#1-为什么需要张量并行)
2. [核心思想：把一个大矩阵乘拆到多张卡](#2-核心思想把一个大矩阵乘拆到多张卡)
3. [列并行与行并行](#3-列并行与行并行)
4. [数值示例：手算一遍 MLP 的 TP](#4-数值示例手算一遍-mlp-的-tp)
5. [Transformer 各层的切分方案](#5-transformer-各层的切分方案)
6. [Attention 的 head 切分与数值示例](#6-attention-的-head-切分与数值示例)
7. [Embedding 与 LM Head 的词表切分](#7-embedding-与-lm-head-的词表切分)
8. [通信量与显存分析](#8-通信量与显存分析)
9. [SGLang 中的实现](#9-sglang-中的实现)
10. [TP 与其他并行的关系](#10-tp-与其他并行的关系)
11. [局限与权衡](#11-局限与权衡)

---

## 1. 为什么需要张量并行

一个 LLM 的单层权重可能就有数十亿参数，单张 GPU 既装不下完整模型权重，也算不动单步前向。
解决「装不下、算不动」有几条互补的路线：

- **数据并行（DP）**：每张卡放一份完整模型，喂不同的数据。解决吞吐，**不解决单卡装不下**。
- **流水线并行（PP）**：把不同的 layer 放到不同卡上，像流水线一样串起来。按**层**切。
- **张量并行（TP）**：把**同一层内部**的大矩阵乘，沿某个维度切成几块，分给几张卡同时算，
  再用一次集合通信把结果拼回来。按**层内的张量维度**切。

TP 的独特价值：它能把**单层的权重和计算**都摊薄到多张卡，因此

1. 让原本单卡放不下的层（如超大 FFN、超多 head 的 attention）能跑起来；
2. 让单步矩阵乘的算力需求降到 $1/p$（$p$ 为 TP 路数），降低单步延迟。

代价是每层要引入**跨卡集合通信**，所以 TP 通常只在**单机多卡、NVLink 高带宽互联**内使用，
跨机则交给 PP / DP（见 [§10](#10-tp-与其他并行的关系)）。

---

## 2. 核心思想：把一个大矩阵乘拆到多张卡

Transformer 里绝大部分计算量是线性层 $Y = XA$（这里省略 bias，且采用「行向量 × 矩阵」约定）：

- $X \in \mathbb{R}^{s \times d_\text{in}}$：输入，$s$ 个 token，每个 $d_\text{in}$ 维。
- $A \in \mathbb{R}^{d_\text{in} \times d_\text{out}}$：权重矩阵。
- $Y \in \mathbb{R}^{s \times d_\text{out}}$：输出。

张量并行的全部技巧，就是利用**分块矩阵乘**的两个恒等式，把 $A$ 切成 $p$ 块分到 $p$ 张卡：

> **沿列切（输出维）**：$A = [A_1 \mid A_2 \mid \dots \mid A_p]$，则
> $$XA = [XA_1 \mid XA_2 \mid \dots \mid XA_p]$$
> 每张卡用完整的 $X$ 乘自己那块 $A_i$，得到输出的一个**列分片**，**无需通信即可各算各的**。

> **沿行切（输入维）**：$A = \begin{bmatrix} A_1 \\ A_2 \\ \vdots \\ A_p \end{bmatrix}$，
> 同时把 $X$ 沿列切 $X = [X_1 \mid \dots \mid X_p]$，则
> $$XA = \sum_{i=1}^{p} X_i A_i$$
> 每张卡算一个**部分和**，最后必须做一次 **all-reduce 求和**才能得到完整结果。

整个 TP 的设计，就是把这两种切法巧妙地**串起来**：让前一层用列切（输出是分片的），
正好作为后一层行切所需的分片输入，这样**两层之间不需要通信**，只在行并行层的末尾做**一次** all-reduce。

---

## 3. 列并行与行并行

### 3.1 列并行 ColumnParallelLinear

- 权重 $A$ 按**输出维** $d_\text{out}$ 切：每张卡持有 $A_i \in \mathbb{R}^{d_\text{in} \times (d_\text{out}/p)}$。
- 输入 $X$ 是**完整**的（每张卡都有一份）。
- 输出 $Y_i = X A_i$ 是**列分片**（每张卡只有 $d_\text{out}/p$ 列）。
- **前向**：默认不通信，直接把分片输出 $Y_i$ 交给下一层；
  若设置 `gather_output=True`，才做一次 all-gather 拼成完整 $Y$。

### 3.2 行并行 RowParallelLinear

- 权重 $A$ 按**输入维** $d_\text{in}$ 切：每张卡持有 $A_i \in \mathbb{R}^{(d_\text{in}/p) \times d_\text{out}}$。
- 输入 $X_i$ 必须是**列分片**的（`input_is_parallel=True`，正好接列并行层的输出）。
- 每张卡算部分和 $X_i A_i \in \mathbb{R}^{s \times d_\text{out}}$。
- **前向**：必须做一次 **all-reduce 求和** $Y = \sum_i X_i A_i$（`reduce_results=True`）。

### 3.3 黄金组合：列并行 → 行并行

把两者串联，中间的分片输出直接对接，**全程只在行并行层末尾通信一次**：

```
            列并行 (无通信)          行并行 (1次 all-reduce)
  X(完整) ──────────────► [Y_1|Y_2] ──────────────► Y(完整)
  每卡有完整 X            每卡持有自己的列分片        all-reduce 求和
```

这正是 SGLang 中 MLP 与 Attention 的实现方式（见 [§5](#5-transformer-各层的切分方案)）。

---

## 4. 数值示例：手算一遍 MLP 的 TP

设 $p=2$（两张卡），用一个 hidden=2、中间维=4 的玩具 MLP（省略激活，只看两层线性的并行）。

**单卡基准**。输入 $x = [1,\ 2]$（一个 token）。

第一层权重（$2 \times 4$，列并行）：
$$
A = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{bmatrix},
\qquad
Y = xA = [\,1,\ 2,\ 1,\ 2\,]
$$

第二层权重（$4 \times 2$，行并行）：
$$
B = \begin{bmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix},
\qquad
Z = YB = [\,5,\ 4\,]
$$

（验证：$Z_0 = 1{\cdot}1+2{\cdot}1+1{\cdot}0+2{\cdot}1 = 5$，$Z_1 = 1{\cdot}1+2{\cdot}0+1{\cdot}1+2{\cdot}1 = 4$。）

### 4.1 列并行切第一层

$A$ 沿列切成两块，每卡一半输出维：
$$
A_0 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\ (\text{rank 0}),
\qquad
A_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\ (\text{rank 1})
$$

两张卡都拿到**完整的** $x=[1,2]$，各算各的（**无通信**）：
$$
Y_0 = x A_0 = [1,\ 2]\ (\text{rank 0}),
\qquad
Y_1 = x A_1 = [1,\ 2]\ (\text{rank 1})
$$

此时输出 $[Y_0 \mid Y_1] = [1,2,1,2]$ 与单卡一致，但**物理上分散在两张卡**，无需拼回。

### 4.2 行并行切第二层

$B$ 沿行切成两块（输入维），正好吃上一步的分片输出：
$$
B_0 = \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix}\ (\text{rank 0}),
\qquad
B_1 = \begin{bmatrix} 0 & 1 \\ 1 & 1 \end{bmatrix}\ (\text{rank 1})
$$

各卡用自己的分片输入算**部分和**：
$$
Y_0 B_0 = [1,2]\begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix} = [\,3,\ 1\,],
\qquad
Y_1 B_1 = [1,2]\begin{bmatrix} 0 & 1 \\ 1 & 1 \end{bmatrix} = [\,2,\ 3\,]
$$

最后 **all-reduce 求和**：
$$
Z = [3,1] + [2,3] = [\,5,\ 4\,]
$$

与单卡基准 $[5,4]$ **完全一致**。整个两层 MLP 只在最后做了 **1 次 all-reduce**，第一层完全无通信——
这就是「列并行 → 行并行」组合的精髓。

---

## 5. Transformer 各层的切分方案

一个标准 Transformer block 由 **Attention 子层**和 **MLP 子层**组成，TP 对两者都用「列并行 → 行并行」。

### 5.1 MLP 子层

以 LLaMA 的 `SwiGLU` MLP 为例（`gate_up_proj` → `act` → `down_proj`）：

| 层 | 并行方式 | 切分维度 | 通信 |
|----|---------|---------|------|
| `gate_up_proj`（`MergedColumnParallelLinear`） | 列并行 | 中间维 $d_\text{ff}$ | 无 |
| `SiluAndMul` 激活 | 逐元素 | — | 无（分片上逐元素计算天然可并行） |
| `down_proj`（`RowParallelLinear`） | 行并行 | 中间维 $d_\text{ff}$ | **1 次 all-reduce** |

`gate` 与 `up` 被合并成一个 `MergedColumnParallelLinear`（一次 GEMM 出两份），列并行后每卡持有
$d_\text{ff}/p$ 个中间通道；激活在分片上逐元素算；`down_proj` 行并行把分片收束回 hidden 维并 all-reduce。

### 5.2 Attention 子层

| 层 | 并行方式 | 切分维度 | 通信 |
|----|---------|---------|------|
| `qkv_proj`（`QKVParallelLinear`） | 列并行 | 注意力 **head** | 无 |
| 各 head 的注意力计算（`RadixAttention`） | head 独立 | — | 无（每 head 自洽） |
| `o_proj`（`RowParallelLinear`） | 行并行 | head | **1 次 all-reduce** |

关键点：注意力是**按 head 天然可分**的——每个 head 的 $QK^\top$、softmax、$\cdot V$ 都只在该 head 内部进行，
head 之间互不依赖。所以把 head 分到不同卡上，每卡独立算自己负责的那几个 head，
最后 `o_proj` 行并行 all-reduce 合并（详见 [§6](#6-attention-的-head-切分与数值示例)）。

### 5.3 一个 block 的通信开销

```
hidden(完整) ─► [Attention: QKV列并行→o_proj行并行] ─► all-reduce#1 ─►
            ─► [MLP: gate_up列并行→down_proj行并行] ─► all-reduce#2 ─► hidden(完整)
```

**每个 Transformer block 前向恰好 2 次 all-reduce**（Attention 一次、MLP 一次）。
LayerNorm/RMSNorm 在 all-reduce 之后的**完整 hidden** 上做，因此**权重在每张卡上复制**、不参与切分。

---

## 6. Attention 的 head 切分与数值示例

设总共 $H=4$ 个 head、每个 head 维度 $d_h=2$、$p=2$ 张卡，则每卡负责 $H/p = 2$ 个 head。

`QKVParallelLinear` 按 head 把 Q/K/V 投影的输出维切开：

- **rank 0** 持有 head 0、1 的 Q/K/V 投影权重；
- **rank 1** 持有 head 2、3 的 Q/K/V 投影权重。

每张卡从**完整的** hidden 出发，算出自己那 2 个 head 的 $q,k,v$，
独立完成 $\text{softmax}(qk^\top/\sqrt{d_h})\,v$，得到 2 个 head 的输出（各 $d_h=2$ 维）：

- rank 0 输出 $O_0 \in \mathbb{R}^{s \times 4}$（head 0,1 拼接）；
- rank 1 输出 $O_1 \in \mathbb{R}^{s \times 4}$（head 2,3 拼接）。

`o_proj`（$d_\text{model} \times d_\text{model} = 8 \times 4$）行并行：权重按输入维（head）切成两块，
rank 0 用前 4 行、rank 1 用后 4 行，各算部分和后 **all-reduce** 得到最终 attention 输出。

### GQA / MQA 下的 KV head 复制

当 KV head 数 $H_{kv}$ 小于 TP 路数 $p$ 时（如 MQA 只有 1 个 KV head，却要分到 8 卡），
无法把 KV head 平均切开。SGLang 的处理是**复制 KV head**：

```python
# QKVParallelLinear.__init__
if tp_size >= total_num_kv_heads:
    num_kv_heads = 1
    num_kv_head_replicas = tp_size // total_num_kv_heads   # KV head 在多卡间复制
else:
    num_kv_heads = total_num_kv_heads // tp_size
    num_kv_head_replicas = 1
```

即：Q head 始终均分，KV head 不够分时就在若干卡上各放一份相同的拷贝。
这会让 KV cache 在这些卡上冗余，但保证每卡都能独立完成自己 Q head 的注意力。

---

## 7. Embedding 与 LM Head 的词表切分

输入词嵌入和输出 LM Head 的权重都是 $\mathbb{R}^{V \times d}$（$V$ 是词表大小，可达十几万），
TP 沿**词表维 $V$** 切分。

### 7.1 VocabParallelEmbedding（输入嵌入）

每张卡只持有词表的一段 $[\text{start}_i, \text{end}_i)$ 对应的 embedding 行。前向时：

1. 对每个输入 token id，判断是否落在本卡负责的词表区间；
2. **不在本卡区间**的 token，先把 id 掩成 0 做查表，再把对应输出**置零**（`masked_fill_`）；
3. 各卡输出相加 → **all-reduce**，得到完整的嵌入向量。

因为每个 token 只会命中**唯一一张**卡的词表区间，其余卡输出 0，all-reduce 求和后正好还原。

### 7.2 ParallelLMHead（输出投影）

LM Head 把 hidden 映射到 $V$ 维 logits，按词表维**列并行**：每卡算出词表一段的 logits 分片。
随后 `LogitsProcessor` 视采样需求决定是否 **all-gather** 拼成完整 $V$ 维 logits
（例如需要在全词表上做 argmax / softmax 采样时）。

---

## 8. 通信量与显存分析

### 8.1 通信量

- **每个 Transformer block 前向 2 次 all-reduce**，每次 all-reduce 的张量大小为 $s \times d_\text{model}$
  （$s$ = batch 内 token 总数）。
- Ring all-reduce 单次的实际跨卡传输量约为 $2\frac{p-1}{p} \cdot (s \cdot d_\text{model})$ 个元素，
  **几乎与 $p$ 无关**——这是 all-reduce 相比 all-gather 的优势，也是 TP 能扩到 8 卡的关键。
- 所以 TP 对**互联带宽**极其敏感：NVLink（数百 GB/s）下高效，跨 PCIe 或跨机（IB）则通信易成瓶颈。
  这是 TP 通常限制在**单机内**的根本原因。

### 8.2 显存

| 部分 | 是否随 TP 切分 | 单卡占用 |
|------|--------------|---------|
| Attention QKV/O、MLP gate_up/down 权重 | 是 | $\approx 1/p$ |
| Embedding / LM Head 权重 | 是（按词表切） | $\approx 1/p$ |
| LayerNorm / RMSNorm 权重 | 否（复制） | $\times 1$（极小，可忽略） |
| KV cache | 按 KV head 切；GQA 不够分时部分复制 | $\approx 1/p$（复制时偏大） |

主体权重和 KV cache 都约降到 $1/p$，这正是 TP「让单卡装得下」的来源。

---

## 9. SGLang 中的实现

### 9.1 并行线性层：`python/sglang/srt/layers/linear.py`

- `ColumnParallelLinear`（`linear.py:293`）：`output_size_per_partition = divide(output_size, tp_size)`
  按输出维切；`forward` 中 `gather_output` 为真才 `tensor_model_parallel_all_gather`，否则直接返回分片。
- `RowParallelLinear`（`linear.py:1341`）：`input_size_per_partition = divide(input_size, tp_size)`
  按输入维切；`forward` 末尾在 `reduce_results and tp_size > 1` 时做 `tensor_model_parallel_all_reduce`。
  bias 只在 `tp_rank == 0` 上加，避免重复累加。
- `MergedColumnParallelLinear`（`linear.py:491`）：把 gate/up 等多个输出打包进一次列并行 GEMM。
- `QKVParallelLinear`（`linear.py:895`）：按 head 切，`num_heads = divide(total_num_heads, tp_size)`，
  KV head 不够分时用 `num_kv_head_replicas` 复制。

### 9.2 词表并行：`python/sglang/srt/layers/vocab_parallel_embedding.py`

- `VocabParallelEmbedding`（`:188`）：`forward`（`:498`）对越界 token `masked_fill_(..., 0)` 后
  `tensor_model_parallel_all_reduce`。
- `ParallelLMHead`（`:544`）：继承词表并行，输出 logits 分片，由 `LogitsProcessor` 决定是否 all-gather。

### 9.3 通信原语与进程组：`python/sglang/srt/distributed/`

- `communication_op.py`：`tensor_model_parallel_all_reduce`（`:18`）、`tensor_model_parallel_all_gather`（`:43`）。
- `parallel_state.py`：`initialize_model_parallel`（`:1848`）建立 TP 进程组；
  `get_tensor_model_parallel_world_size`（`:2261`）/ `get_tensor_model_parallel_rank`（`:2266`）查询 $p$ 与本卡 rank。

### 9.4 模型侧用法：`python/sglang/srt/models/llama.py`

```python
# LlamaMLP（llama.py:85）
self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, ...)  # 列并行
self.down_proj   = RowParallelLinear(intermediate_size, hidden_size, ...)                  # 行并行 → all-reduce

# LlamaAttention（llama.py:147）
tp_size = get_tensor_model_parallel_world_size()
self.num_heads    = self.total_num_heads // tp_size      # head 按 TP 均分
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
self.qkv_proj = QKVParallelLinear(hidden_size, head_dim, total_num_heads, total_num_kv_heads, ...)  # 列并行
self.o_proj   = RowParallelLinear(total_num_heads * head_dim, hidden_size, ...)                     # 行并行 → all-reduce
```

### 9.5 启动参数：`python/sglang/srt/server_args.py`

```bash
python -m sglang.launch_server --model <model> --tp-size 4   # 等价 --tensor-parallel-size 4
```

`tp_size`（`server_args.py:534`，CLI 定义 `server_args.py:5635`）即 TP 路数 $p$；
MoE 场景另有 `--moe-dense-tp-size` 等专用旋钮。

---

## 10. TP 与其他并行的关系

实际部署常把多种并行组合成 **N 维并行网格**：

| 并行 | 切什么 | 通信粒度 | 适用范围 |
|------|-------|---------|---------|
| **TP（张量并行）** | 层内矩阵的张量维 | 每层 all-reduce（频繁、小） | 单机内、高带宽 NVLink |
| **PP（流水线并行）** | 不同 layer 分到不同卡 | 层间激活传递（稀疏） | 跨机，配合 micro-batch |
| **DP（数据并行）** | batch 数据 | 反向梯度 all-reduce（训练） | 任意，提吞吐 |
| **EP（专家并行）** | MoE 的 expert | all-to-all | MoE 模型 |

典型组合：`world_size = TP × PP × DP`。例如 8 机 × 8 卡共 64 卡跑超大模型，
可设 TP=8（机内切层）、PP=8（跨机切层），DP 再叠在外层提吞吐。
**经验法则：TP 优先吃满单机 NVLink，超出单机后再上 PP/DP。**

SGLang 中 attention 还支持 **DP attention**（`use_dp_attention_reduce`），
在 attention 子层用更小的 attention-TP 组做 all-reduce，详见相关代码与术语表。

---

## 11. 局限与权衡

1. **通信开销**：每层 2 次 all-reduce，对带宽极敏感，跨机 TP 几乎不可行。
2. **切分约束**：`head 数`、`中间维`、`词表大小`都必须能被 $p$ 整除（代码里大量 `divide(...)` / `assert ... % tp_size == 0`）；
   GQA 的 KV head 不够分时只能复制，带来 KV cache 冗余。
3. **小层不划算**：当 $d_\text{ff}/p$ 太小，GEMM 利用率下降，可能比单卡更慢——这也是 `--moe-dense-tp-size` 等参数存在的原因。
4. **与量化/算子的耦合**：分片后的权重加载（`weight_loader`）、量化 scale 切分都需要专门处理，
   增加了实现复杂度。

---

## 参考与延伸

- 经典论文：Megatron-LM（Shoeybi et al., 2019），本文的列并行 / 行并行组合即源自此。
- 同目录其他理论文档：`../MLA.md`（MLA 与 TP 下的 KV cache 切分）、`../Glossary.md`（术语表）。
- SGLang 代码：`python/sglang/srt/layers/linear.py`、`vocab_parallel_embedding.py`、`distributed/parallel_state.py`。
