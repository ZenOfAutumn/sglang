# 分布式推理集合通信原语详解（Collective Communication）

> 本文系统介绍分布式推理中用到的所有通信原语（communication primitive）：point-to-point、
> broadcast、reduce、all-reduce、gather、all-gather、scatter、reduce-scatter、all-to-all，
> 逐个讲清**语义、示意图、通信量、Ring 实现、在 SGLang 各并行方式中的用途**，
> 并给出与 TP/DP/EP/PP/CP 各文档的对应关系。
>
> 前置阅读：`TP.md`（all-reduce/all-gather）、`EP.md`（all-to-all）、`PP.md`（P2P）、`CP.md`（all-gather）。

## 目录

1. [总览：一张表看懂所有原语](#1-总览一张表看懂所有原语)
2. [基础概念：rank / group / world](#2-基础概念-rank--group--world)
3. [Point-to-Point（P2P：send / recv）](#3-point-to-pointp2psend--recv)
4. [Broadcast（广播）](#4-broadcast广播)
5. [Reduce（归约）](#5-reduce归约)
6. [All-Reduce（全归约）](#6-all-reduce全归约)
7. [Gather / All-Gather（收集 / 全收集）](#7-gather--all-gather收集--全收集)
8. [Scatter（分发）](#8-scatter分发)
9. [Reduce-Scatter（归约分发）](#9-reduce-scatter归约分发)
10. [All-to-All（全交换）](#10-all-to-all全交换)
11. [组合关系与通信量对比](#11-组合关系与通信量对比)
12. [SGLang 中的实现](#12-sglang-中的实现)

---

## 1. 总览：一张表看懂所有原语

设通信组内有 $p$ 个 rank，每个 rank 初始持有大小为 $N$ 的数据（除特别说明）。

| 原语 | 一句话语义 | 输出在谁手上 | 是否做算术归约 | 典型通信量/rank | SGLang 用途 |
| --- | --- | --- | --- | --- | --- |
| **P2P (send/recv)** | 一个 rank 发给另一个 rank | 指定的目标 rank | 否 | $N$ | PP 层间传激活 |
| **Broadcast** | 一个 rank 把数据发给所有 rank | 所有 rank | 否 | $N$ | 权重/元数据同步 |
| **Reduce** | 所有 rank 数据求和到一个 rank | 单个 root rank | 是 | $N$ | 少用 |
| **All-Reduce** | 所有 rank 数据求和且都拿到结果 | 所有 rank | 是 | $2\frac{p-1}{p}N$ | **TP 层内求和**、DP 同步 |
| **Gather** | 各 rank 的分片汇总到一个 rank | 单个 root rank | 否 | $N$ | logits 汇总 |
| **All-Gather** | 各 rank 的分片汇总且都拿到全部 | 所有 rank | 否 | $\frac{p-1}{p}N_{\text{total}}$ | **CP KV 收集**、SP、TP 输出拼接 |
| **Scatter** | 一个 rank 把不同分片发给各 rank | 所有 rank（各拿一片） | 否 | — | 少用 |
| **Reduce-Scatter** | 求和后每 rank 只拿结果的一片 | 所有 rank（各拿一片） | 是 | $\frac{p-1}{p}N$ | SP、all-reduce 的前半 |
| **All-to-All** | 每 rank 把不同分片发给不同 rank | 所有 rank（重排后） | 否 | $\frac{p-1}{p}N$ | **EP dispatch/combine** |

> 记忆线索：
> - 名字带 **All-** → 结果落在**所有** rank；不带 → 落在**单个** root。
> - 带 **Reduce** → 有算术求和；带 **Gather/Scatter/All-to-All** → 只搬运不求和。
> - **All-Reduce = Reduce-Scatter + All-Gather**（见 §6、§11）。

---

## 2. 基础概念：rank / group / world

- **rank**：通信组内每个进程/GPU 的编号（0 ~ $p-1$）。
- **group（通信组）**：一组参与同一次集合通信的 rank。SGLang 为每种并行建独立 group：TP group、DP group、PP group、EP group、`_ATTN_CP` group 等（`distributed/parallel_state.py`）。
- **world**：所有进程的全集（world_size = 全局 GPU 数）。

同一张 GPU 可能同时属于多个 group（例如既在 TP group 又在 PP group），不同 group 做不同的集合通信。下面所有原语都是**在某个 group 内部**执行的。

---

## 3. Point-to-Point（P2P：send / recv）

**语义**：最基础的通信——rank A 调用 `send` 把张量发给 rank B，rank B 调用 `recv` 接收。点对点，不涉及第三方。

```
rank 0            rank 1
 [X] ──send──────► [X]   (recv)
```

- **同步 send/recv**：阻塞直到完成；
- **异步 isend/irecv**：立即返回一个 handle，之后 `wait()`；用于计算与通信重叠。

**通信量**：一次传 $N$，只占用两个 rank 之间的一条链路。

**SGLang 用途**：
- **PP（流水线并行）** 的核心通信——stage $i$ 算完后把激活（hidden states）P2P 发给 stage $i+1$（`PP.md` §5）。PP 通信稀疏、只在相邻 stage 间，所以能跨机。
- PD 分离下 KV Cache 的跨节点传输在更高层（Mooncake/RDMA），但底层也是点对点语义。

对应代码：`parallel_state.py` 的 `send` / `recv`、`send_tensor_dict` / `recv_tensor_dict`、`send_object` / `recv_object`。

---

## 4. Broadcast（广播）

**语义**：一个 root rank 把它的数据**原样复制**给组内所有其他 rank。结束后所有 rank 持有相同数据。

```
        rank 0 (root)
          [X]
     ┌─────┼─────┐
     ▼     ▼     ▼
  rank0  rank1  rank2
  [X]    [X]    [X]
```

**通信量**：每个接收 rank 收到 $N$；Ring/tree 实现下 root 出口带宽是瓶颈，总量约 $(p-1)N$。

**SGLang 用途**：
- 采样参数、批次元数据、`tensor_dict` 从 TP rank 0 广播到其余 rank（保证 TP 组内一致）；
- 权重加载、随机种子同步等一次性同步。

对应代码：`broadcast`、`broadcast_object` / `broadcast_object_list`、`broadcast_tensor_dict`；进程间用共享内存的 `shm_broadcast.py`。

---

## 5. Reduce（归约）

**语义**：把所有 rank 的数据**按元素求和**（或 max/min/avg），结果只放到**一个 root rank**。

```
 rank0[a]  rank1[b]  rank2[c]
     └────────┼────────┘
              ▼  求和
        rank0 [a+b+c]   （只有 root 拿到）
```

**与 All-Reduce 的区别**：Reduce 结果只在 root；All-Reduce 所有 rank 都有。推理中很少单独用 Reduce（通常需要所有 rank 都拿到结果，故用 All-Reduce）。

---

## 6. All-Reduce（全归约）

**语义**：所有 rank 的数据按元素求和，且**每个 rank 都拿到最终求和结果**。这是 TP 最核心的原语。

```
 rank0[a]  rank1[b]  rank2[c]
     └────────┼────────┘  求和并分发回所有 rank
     ┌────────┼────────┐
 rank0     rank1     rank2
[a+b+c]   [a+b+c]   [a+b+c]
```

### Ring All-Reduce 通信量

高效实现是 **Ring All-Reduce = Reduce-Scatter + All-Gather** 两阶段，每阶段 $p-1$ 步，每步传 $N/p$：

$$
\text{单 rank 通信量} = 2 \cdot \frac{p-1}{p} \cdot N
$$

$N$ 是被 all-reduce 的张量大小（TP 中约 $s \times d_\text{model}$）。当 $p$ 增大，$\frac{p-1}{p} \to 1$，**每卡通信量趋于常数 $2N$，与 $p$ 无关**——这解释了为什么 TP 尚可扩展，但因为要走高带宽 NVLink，通常限制在单机 8 卡内（详见 `TP.md` §8）。

**SGLang 用途**：
- **TP**：每个 Transformer block 2 次 all-reduce（attention 的 `o_proj` 行并行后、MLP 的 `down_proj` 行并行后），`VocabParallelEmbedding` 也用 all-reduce（`TP.md` §7）；
- **DP**：副本间的梯度/统计同步（推理侧主要是 DP-attention 相关的 token 数同步）。

对应代码：`parallel_state.py: all_reduce`；高性能后端 `device_communicators/`：`pynccl.py`（NCCL）、`custom_all_reduce.py` / `quick_all_reduce.py`（自定义低延迟 all-reduce）、`pymscclpp.py`。

---

## 7. Gather / All-Gather（收集 / 全收集）

### Gather

**语义**：各 rank 持有一个分片，全部**拼接**（不求和）到一个 root rank。

```
 rank0[A] rank1[B] rank2[C]
     └───────┼───────┘
             ▼ 拼接
      rank0 [A|B|C]   （只有 root 拿到）
```

### All-Gather

**语义**：各 rank 的分片拼接，且**每个 rank 都拿到完整拼接结果**。

```
 rank0[A] rank1[B] rank2[C]
     └───────┼───────┘ 拼接并分发回所有 rank
     ┌───────┼───────┐
[A|B|C]  [A|B|C]  [A|B|C]
```

**通信量**：All-Gather 每 rank 约 $\frac{p-1}{p} N_\text{total}$（$N_\text{total}$ 为拼接后总大小）。**注意 All-Gather 不做求和，只搬运**，所以没有 all-reduce 的 2× 系数。

**SGLang 用途**：
- **CP（上下文并行）**：attention 阶段各卡把本段 KV all-gather 成完整序列 KV，本段 Q 才能 attend 到前面所有 token（`CP.md` §5，`cp_all_gather_reorganized_into_tensor`）；
- **Sequence Parallel / DP-attention**：把分散的 token 激活收集齐（`attn_tp_all_gather`，`DP_attention.md`）；
- **TP**：`gather_output=True` 时把列并行输出 all-gather 拼回完整维；logits 按需 all-gather 拼成完整词表维（`TP.md` §7）。

对应代码：`all_gather`、`all_gather_into_tensor`、`all_gatherv`（变长）、`gather`、`all_gather_object`。

---

## 8. Scatter（分发）

**语义**：Broadcast 的"分片版"——一个 root rank 把一个大张量**切成 $p$ 片，每个 rank 拿到不同的一片**（对比 broadcast 是每个 rank 拿到相同的完整数据）。

```
      rank0 (root) [A|B|C]
     ┌──────┼──────┐  各发一片
     ▼      ▼      ▼
  rank0   rank1   rank2
   [A]     [B]     [C]
```

推理中单独使用较少，更多是作为 reduce-scatter / all-to-all 的组成语义出现。

---

## 9. Reduce-Scatter（归约分发）

**语义**：先把所有 rank 的数据**按元素求和**，再把求和结果**切片**，每个 rank 只拿到结果的**一个分片**（相当于 Reduce + Scatter）。

```
 rank0[a0 a1 a2]  rank1[b0 b1 b2]  rank2[c0 c1 c2]
        └──────── 按位求和 ────────┘
        s0=a0+b0+c0  s1=...  s2=...
        └──── 各 rank 只拿一片 ────┘
   rank0[s0]      rank1[s1]      rank2[s2]
```

**通信量**：每 rank 约 $\frac{p-1}{p} N$，是 All-Reduce 的**前一半**。

**SGLang 用途**：
- **Sequence Parallel**：把 all-reduce 拆成 `reduce_scatter`（各 rank 只保留自己负责的 token 分片）+ 后续 `all_gather`，在序列维分摊 LayerNorm 等计算；
- 是 Ring All-Reduce 的组成阶段（§6）。

对应代码：`reduce_scatter`、`reduce_scatter_tensor`、`reduce_scatterv`（变长）。

---

## 10. All-to-All（全交换）

**语义**：最"全"的通信——**每个 rank 都把自己的数据切成 $p$ 份，第 $j$ 份发给 rank $j$**；同时从每个 rank 收一份。相当于一次分布式的"矩阵转置"。

```
发送前（行=持有者，列=目标）      接收后
 rank0: [→0 →1 →2]              rank0: 收到各 rank 的第0片
 rank1: [→0 →1 →2]      ==>     rank1: 收到各 rank 的第1片
 rank2: [→0 →1 →2]              rank2: 收到各 rank 的第2片
```

**通信量**：每 rank 发出/收到约 $\frac{p-1}{p} N$，但它是**全连接式**通信（每对 rank 都有流量），对网络对分带宽（bisection bandwidth）要求最高。

**SGLang 用途**：
- **EP（专家并行）** 的核心——**dispatch**（all-to-all #1：把 token 按其选中 expert 发到对应卡）和 **combine**（all-to-all #2：把专家输出送回 token 原卡）。跨节点 EP（如 `ep_size=128`）对 RDMA/IB 带宽要求极高（`EP.md` §4、§8）。

对应代码：`parallel_state.py: all_to_all_single`；EP 专用高性能后端在 `layers/moe/token_dispatcher/`（DeepEP、Mooncake、NIXL、Mori、FlashInfer）。

---

## 11. 组合关系与通信量对比

### 关键恒等式

```
All-Reduce  =  Reduce-Scatter  +  All-Gather
（求和分片）      （拼回完整）

Reduce      =  Reduce-Scatter  +  Gather(到root)
Broadcast   ≈  Scatter         +  All-Gather（概念上）
```

这解释了为什么 All-Reduce 通信量恰好是 Reduce-Scatter（$\frac{p-1}{p}N$）+ All-Gather（$\frac{p-1}{p}N$）= $2\frac{p-1}{p}N$。

### 各并行方式主导的通信原语

| 并行方式 | 主导通信原语 | 频率 | 拓扑要求 |
| --- | --- | --- | --- |
| **TP**（`TP.md`） | All-Reduce | 每 block 2 次 | 单机 NVLink 高带宽 |
| **DP**（`DP.md`） | 请求分发 + 少量 All-Reduce | 低 | 松散 |
| **DP-attention**（`DP_attention.md`） | All-Gather / Reduce-Scatter | 层内切换维度时 | 单机 |
| **EP**（`EP.md`） | All-to-All（dispatch+combine） | 每 MoE 层 2 次 | 高对分带宽（IB/RoCE） |
| **PP**（`PP.md`） | P2P（send/recv） | 每 stage 边界 | 稀疏，可跨机 |
| **CP**（`CP.md`） | All-Gather（KV） | 每 attention 层 | 单机（跨机有精度问题） |

**通信成本直觉**（从轻到重）：P2P（点对点）< Broadcast/Gather < All-Gather/Reduce-Scatter < All-Reduce < All-to-All（全连接，最重）。这也是为什么 TP/EP 尽量放单机、PP 才用于跨机。

---

## 12. SGLang 中的实现

### 12.1 通信原语入口：`python/sglang/srt/distributed/parallel_state.py`

`GroupCoordinator` 封装了一个通信组的所有原语：

| 方法 | 原语 |
| --- | --- |
| `all_reduce` | All-Reduce |
| `all_gather` / `all_gather_into_tensor` / `all_gatherv` | All-Gather |
| `reduce_scatter` / `reduce_scatter_tensor` / `reduce_scatterv` | Reduce-Scatter |
| `all_to_all_single` | All-to-All |
| `broadcast` / `broadcast_tensor_dict` / `broadcast_object` | Broadcast |
| `gather` | Gather |
| `send` / `recv` / `send_tensor_dict` / `recv_tensor_dict` | P2P |

各并行 group（`_TP` / `_PP` / `_DP` / `_ATTN_CP` / EP group 等）都是 `GroupCoordinator` 实例。

### 12.2 设备通信后端：`python/sglang/srt/distributed/device_communicators/`

- `pynccl.py` / `pynccl_wrapper.py`：NCCL（NVIDIA 集合通信库），CUDA 默认后端；
- `custom_all_reduce.py` / `custom_all_reduce_v2.py` / `quick_all_reduce.py`：自定义低延迟 all-reduce（小消息优于 NCCL）；
- `pymscclpp.py`：MSCCL++ 后端；
- `shm_broadcast.py`：进程间共享内存广播；
- `npu_communicator.py` / `xpu_communicator.py` / `hpu_communicator.py`：昇腾 / XPU / HPU 平台后端；
- `torch_symm_mem.py`：对称内存（symmetric memory）支持。

### 12.3 EP 专用 all-to-all：`python/sglang/srt/layers/moe/token_dispatcher/`

EP 的 dispatch/combine 不走通用 `all_to_all_single`，而是用专门优化的 DeepEP / Mooncake / NIXL / Mori / FlashInfer 后端（低延迟、支持 FP8 dispatch），见 `EP.md` §8。

### 12.4 目录说明

`python/sglang/srt/distributed/README_zh.md`、`device_communicators/README_zh.md`。

---

## 参考与延伸

- 同目录：`TP.md`（All-Reduce/All-Gather）、`DP.md`、`PP.md`（P2P）、`EP.md`（All-to-All）、`CP.md`（All-Gather）、`DP_attention.md`。
- SGLang 代码：`python/sglang/srt/distributed/parallel_state.py`、`python/sglang/srt/distributed/device_communicators/`。
- 相关背景：NCCL 集合通信语义、Ring All-Reduce 算法（Baidu ring-allreduce）、DeepEP（EP all-to-all 通信库）。

