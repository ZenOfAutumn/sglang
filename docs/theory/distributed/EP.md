# 专家并行（Expert Parallelism, EP）原理详解

> 本文介绍专家并行的动机与数学结构、MoE 层的 token 路由与 all-to-all dispatch/combine、
> EP 与 TP/DP 的关系、负载不均与 EPLB，并给出数值直觉，最后对应到 SGLang 中
> `layers/moe/`（`ep_moe/`、`token_dispatcher/`）的真实实现。
>
> 前置阅读：`TP.md`（张量并行）。EP 是**专门针对 MoE（Mixture-of-Experts）层**的并行方式。

## 目录

1. [为什么需要专家并行](#1-为什么需要专家并行)
2. [MoE 层回顾：router + experts](#2-moe-层回顾router--experts)
3. [EP 的核心思想：按 expert 切分](#3-ep-的核心思想按-expert-切分)
4. [数据流：dispatch → 专家计算 → combine](#4-数据流dispatch--专家计算--combine)
5. [数值示例：手算一次 EP 路由](#5-数值示例手算一次-ep-路由)
6. [负载不均与 EPLB](#6-负载不均与-eplb)
7. [EP 与 TP / DP 的关系](#7-ep-与-tp--dp-的关系)
8. [SGLang 中的实现](#8-sglang-中的实现)
9. [局限与权衡](#9-局限与权衡)

---

## 1. 为什么需要专家并行

MoE 模型（如 DeepSeek、LongCat、Mixtral）用**大量专家（expert）** 替代单个稠密 FFN：每层可能有几百到上千个 expert（如 256、768 个），但每个 token 只激活其中 top-k 个（如 top-8）。

问题在于：**专家总量极大**。以 768 个 expert、每个 50M 参数为例，单层 MoE 权重就有约 38B 参数——单卡 HBM 根本放不下。

常见的切分办法有两种：

- **TP**：把每个 expert 的 FFN 矩阵按张量维切分（`TP.md` §5.1 的做法）。使用 $p$ 路 TP 时，每卡持有所有 expert 的 $1/p$ 分片，权重约降至 $1/p$；但每个 expert 的计算需要多个 TP rank 协作，并产生 collective 通信。
- **EP（专家并行）**：把**不同的 expert 分到不同 rank**。纯 EP 下，768 个 expert 分到 8 个 rank，每 rank 持有 96 个完整 expert；分到 128 个 rank，每 rank 持有 6 个完整 expert，专家权重约降至 $1/p$。若 EP 同时组合 expert TP，则 local expert 仍会继续按张量维切分。

EP 的独特价值不是「只有 EP 才能减少每卡权重」，而是利用 expert 之间天然独立的性质，让不同 rank 独立计算不同 expert。代价是 token 需要跨 rank 路由到其 expert 所在位置，通常产生 all-to-all 通信。

---

## 2. MoE 层回顾：router + experts

一个 MoE 层的前向：

```
hidden (s × d)
   │
   ▼
┌─────────┐  router：为每个 token 打分，选出 top-k 个 expert + 权重
│ Router  │  topk_ids  [s, k]，topk_weights [s, k]
└─────────┘
   │
   ▼  每个 token 送到它选中的 k 个 expert
┌──────────────┐
│  Experts     │  每个 expert 是一个独立的 FFN（SwiGLU）
│ E0 E1 ... EN │
└──────────────┘
   │
   ▼  按 topk_weights 加权求和各 expert 的输出
output (s × d)
```

- **Router**（`layers/moe/router.py`、`topk.py`）：算出每个 token 选哪 k 个 expert（`topk_ids`）及权重（`topk_weights`）；
- **Experts**：每个 expert 是独立 FFN，只处理路由到它的 token；
- **合并**：每个 token 的输出 = 其 k 个 expert 输出的加权和。

关键特性：**每个 expert 天然独立**——它只处理分配给它的 token，与其他 expert 无数据依赖。这正是 EP 能按 expert 切分的基础。

---

## 3. EP 的核心思想：按 expert 切分

设总共 $E$ 个 physical expert、EP 路数为 $p$（即 `ep_size`），并假设 $E$ 能被 $p$ 整除，则每个 EP rank 负责 $E/p$ 个 local experts。**只有纯 EP（MoE TP 为 1）时，local expert 才是完整权重**；若同时使用 expert TP，每个 local expert 仍只是一个 TP 分片。为便于说明，本节图示先采用纯 EP。

```
                所有 token（经 router 得到 topk_ids）
                          │
        ┌─────────────────┼─────────────────┐
        │ all-to-all dispatch：按 expert 归属把 token 发到对应卡
        ▼                 ▼                 ▼
   rank 0            rank 1            rank p-1
   E0 ~ E(E/p-1)     E(E/p) ~ ...      ... ~ E(E-1)
   本地专家计算       本地专家计算        本地专家计算
        │                 │                 │
        └─────────────────┼─────────────────┘
        │ all-to-all combine：把结果送回 token 原来的卡
        ▼
                各 token 收齐自己 k 个 expert 的输出 → 加权求和
```

与 TP 切 expert 的对比：

| 方式 | 切什么 | 每 rank 权重 | 通信 |
| --- | --- | --- | --- |
| **TP 切 expert** | 每个 expert 的 FFN 矩阵按张量维切 | 所有 expert 的 $1/p$ 分片（约降到 $1/p$） | expert 计算前后通常需要 collective |
| **纯 EP** | 不同 expert 分到不同 rank | $E/p$ 个完整 expert（约降到 $1/p$） | dispatch + combine，通常基于 all-to-all |
| **EP + expert TP** | 先按 expert 切，再切 local expert 的张量维 | $E/\text{EP}$ 个 expert 的 TP 分片 | all-to-all + TP collective |

EP 和 TP 都能降低每 rank 的专家权重占用，但并行对象和通信模式不同。EP 用 token 路由通信换取 expert 级独立计算，是部署大规模 MoE 的关键手段之一。

---

## 4. 数据流：dispatch → 专家计算 → combine

从概念上看，EP 的一次 MoE 前向分三步，包含 dispatch 和 combine 两个跨 rank 交换阶段；具体后端可能对通信、复制和加权归并进行融合：

### 4.1 Dispatch（分发）

每个 token 经 router 选出 top-k 个 expert，这些 experts 可能分散在不同 rank 上。**dispatch 把 token payload 和路由元数据发送到目标 expert 所在的 rank**；不同后端可能使用 all-to-all 或等价的专用通信原语：

- 输入：本卡的 token 及其 `topk_ids`；
- 输出：本卡收到的、路由到**本卡 local experts** 的所有 token（可能来自任意卡）。

### 4.2 专家计算

每个 rank 在收到的 token 上运行自己的 local experts（通常使用 grouped GEMM）。权重容量约随 EP 路数降至 $1/p$；计算量则由实际收到的 token-expert assignments 决定。只有在路由负载理想均衡时，每个 rank 才大约承担全局 expert 计算量的 $1/p$。

### 4.3 Combine（合并）

**combine 将 expert 输出送回 token 的来源 rank**，最终语义是按 `topk_weights` 对该 token 的 $k$ 个 expert 输出加权求和。加权归并可能发生在 combine 之前、之后或通信算子内部，具体顺序取决于后端实现。

```
Dispatch (all-to-all #1)          Combine (all-to-all #2)
 token → 其 expert 所在卡    →    expert 输出 → token 原所在卡
```

> SGLang 用 `BaseDispatcher`（`token_dispatcher/base.py`）统一 dispatch/combine 接口，并提供 DeepEP、Mooncake、NIXL、Mori、FlashInfer 等实现。`StandardDispatcher` 主要处理标准数据格式、local expert 映射和部分 gather/scatter 路径；Ascend FuseEP 则绕过这里的普通 dispatcher 流程（见 §8）。

### 4.4 通信量分析

设 $T$ 是整个 EP group 的 token 总数，$k$ 是每个 token 激活的 expert 数，$d$ 是 `hidden_size`。若把每个 token-expert assignment 都看作一份独立的 $d$ 维 payload，则 dispatch 和 combine 的**逻辑 payload 上限**分别为：

$$
V_{\text{dispatch, logical}} \le Tkd,
\qquad
V_{\text{combine, logical}} \le Tkd
$$

因此两阶段的逻辑 payload 上限为：

$$
V_{\text{logical}} \le 2Tkd
$$

这里的 $2Tkd$ **不是实际网络字节数**。实际通信还受到以下因素影响：

1. **本地命中不走网络**：若 expert 均匀放置且路由近似均匀，assignment 落在 token 来源 rank 的概率约为 $1/p$，粗略期望为

$$
V_{\text{network}} \approx 2Tkd\left(1-\frac{1}{p}\right)
$$

2. **同一目标 rank 可合并传输**：若一个 token 选中的多个 experts 位于同一目标 rank，后端可以只发送一份 hidden state，并用路由元数据在接收端展开。因此 dispatch 数据量更准确地与 token 命中的不同远端 rank 数有关，而不一定严格等于 $Tkd$。
3. **dtype 和元数据不同**：dispatch 可配置为 BF16、FP8、INT8 或 NVFP4，取决于后端、硬件和量化配置；还需传输 `topk_ids`、`topk_weights`、scales 等元数据。combine 常见为 BF16 payload，但不是所有后端的统一约束。
4. **padding 和固定容量 buffer**：low-latency 或硬件专用后端可能按固定容量、对齐粒度传输或预留 workspace，实际链路字节数可能高于有效 payload。

因此性能分析应分别说明：逻辑 assignment payload、有效跨 rank payload、协议实际传输字节和 workspace 容量。

---

## 5. 数值示例：手算一次 EP 路由

设 $E=4$ 个 expert、EP $p=2$ 卡（每卡 2 个 expert）、top-k=1、共 4 个 token：

- **rank 0** 持有 E0, E1；**rank 1** 持有 E2, E3；
- token 初始分布：rank 0 有 tokenA、tokenB；rank 1 有 tokenC、tokenD；
- router 结果（每 token 选 1 个 expert）：
  - tokenA → E2，tokenB → E0，tokenC → E3，tokenD → E1。

**Dispatch**（按 expert 归属发送）：

| token | 选中 expert | expert 所在卡 | 发往 |
| --- | --- | --- | --- |
| A | E2 | rank 1 | rank 0 → rank 1 |
| B | E0 | rank 0 | 留本地 |
| C | E3 | rank 1 | 留本地 |
| D | E1 | rank 0 | rank 1 → rank 0 |

dispatch 后：**rank 0** 处理 {B(E0), D(E1)}；**rank 1** 处理 {A(E2), C(E3)}。

**专家计算**：各卡在本地 expert 上算出输出 $O_B, O_D$（rank 0）、$O_A, O_C$（rank 1）。

**Combine**（送回原卡）：$O_A$ 回 rank 0、$O_D$ 回 rank 1……最终每个 token 在自己原来的卡上拿到输出（top-1 时直接就是该 expert 输出；top-k>1 时加权求和）。

可见：token 被"发出去算、再收回来"，每卡只算自己 expert 的那部分——这就是 EP 的 all-to-all 本质。

---

## 6. 负载不均与 EPLB

EP 的最大实践问题是**专家负载不均**：某些"热门 expert"被大量 token 选中，其所在卡成为瓶颈；而冷门 expert 所在卡空闲。

SGLang 用 **EPLB（Expert Parallelism Load Balancing）** 缓解：

- 在整个 EP group 中额外分配 `ep_num_redundant_experts` 个 physical expert 槽位，把热点 logical expert 复制到多个位置；该参数是**每层全局冗余槽位总数**，不是每个 rank 的数量；
- 周期性（`eplb_rebalance_num_iterations`）统计各 expert 的访问频率；启用 EPLB 且未显式指定时，`expert_distribution_recorder_mode` 会自动设为 `stat`；
- 根据统计结果重新计算 physical-to-logical expert 映射；`ep_dispatch_algorithm=static` 时使用预计算的 logical-expert 到 rank 映射进行分发。

设 logical expert 数为 $E$、全局冗余槽位数为 $R$，则：

$$
E_{\text{physical}}=E+R,
\qquad
E_{\text{local}}=\frac{E+R}{p}
$$

并要求 $E+R$ 能被 `ep_size` 整除。例如 $E=768$、$R=128$、$p=128$ 时，共有 896 个 physical expert 槽位，每个 EP rank 持有 7 个槽位。

相关代码在 `python/sglang/srt/eplb/`，核心映射在 `expert_location.py`。

---

## 7. EP 与 TP / DP 的关系

EP 作用于 **MoE expert 层**；attention、稠密 FFN 和完整模型副本可以使用不同层级的 TP / DP。实际部署常组合：

| 并行 | 作用对象 | 切什么 | 典型通信 |
| --- | --- | --- | --- |
| **Attention TP** | attention | head 或张量维 | all-reduce / all-gather 等 |
| **EP** | MoE experts | 不同 expert 分到不同 rank | all-to-all dispatch/combine |
| **MoE TP** | 单个 local expert | expert FFN 张量维 | MoE TP collective |
| **DP Attention** | attention | token/batch | 与非 attention 层之间 gather/scatter |
| **MoE DP** | MoE 层 | token/batch | 与 EP/TP 共同组成 rank mesh |
| **副本级 DP** | 整个模型 | 请求/batch | 入口请求分发，副本间通常无层内通信 |

SGLang 通常复用同一个外层 `tp_size` rank 集合，并将 MoE 侧分解为：

$$
moe\_tp\_size
=
\frac{tp\_size}{ep\_size\times moe\_dp\_size}
$$

因此这些并行方式在计算语义上作用于不同维度，但在进程拓扑和资源上并非完全独立。相关参数包括：

- `dp_size` 与 `attn_cp_size`：决定 DP Attention、CP 和 Attention TP 的分解；
- `ep_size`：MoE 的 EP 路数；
- `moe_dp_size`：MoE 层的数据并行度；
- `moe_dense_tp_size`：MoE 模型中 dense 部分使用的 TP 规模。

对 DeepEP、Mooncake、NIXL、Mori、FlashInfer 等专用 A2A 后端，当前参数处理会把 `ep_size` 调整为 `tp_size`，此时通常有 `moe_tp_size=1`。

---

## 8. SGLang 中的实现

### 8.1 MoE 层与 EP 实现：`python/sglang/srt/layers/moe/`

- `router.py` / `topk.py`：router 打分与 top-k 选择，产出 `topk_ids` / `topk_weights`。
- `ep_moe/layer.py`：定义 `DeepEPMoE`（继承 `FusedMoE`）；`get_moe_impl_class` 根据 A2A backend 选择 `DeepEPMoE` 或 `FusedMoE`。DeepEP、Mooncake、NIXL、Mori 当前共用 `DeepEPMoE` 入口，Ascend FuseEP 则走 `FusedMoE` 外部的融合路径。
- `ep_moe/kernels.py`：EP 所需 Triton kernel（permute/反排序、src2dst 映射、EP scatter、加权归并、量化等）。

### 8.2 token 分发后端：`python/sglang/srt/layers/moe/token_dispatcher/`

- `base.py`：`BaseDispatcher` 抽象、dispatch/combine 协议以及 `DispatchOutput`/`CombineInput` 格式。
- `standard.py`：标准格式 dispatcher，负责 local expert 映射，并为部分 runner 提供 all-gather/reduce-scatter 路径；它不等同于通用 all-to-all 实现。
- `deepep.py`：DeepEP 后端，支持 normal / low-latency 模式。
- `mooncake.py` / `nixl.py` / `moriep.py` / `flashinfer.py`：Mooncake / NIXL / Mori / FlashInfer 后端。
- Ascend FuseEP 不在该目录中提供 `fuseep.py` dispatcher，而是在 NPU MoE 路径中绕过普通 dispatch/combine 流程。

### 8.3 负载均衡：`python/sglang/srt/eplb/`

EPLB 的专家分布记录、再平衡算法与 expert-location 元数据管理。

### 8.4 启动参数：`python/sglang/srt/server_args.py`

```bash
python -m sglang.launch_server --model <MoE-model> \
    --tp-size 8 --moe-a2a-backend deepep --enable-eplb
```

- `--ep-size`：EP 路数 $p$；
- `--moe-a2a-backend`：all-to-all 通信后端，如 `deepep`、`mooncake`、`nixl`、`mori`、`flashinfer` 或 `none`；
- `--moe-dp-size`：MoE 层的数据并行度；
- `--moe-dense-tp-size`：MoE 模型中 dense 部分的 TP；
- EPLB 相关：`--enable-eplb`、`--ep-num-redundant-experts`、`--eplb-rebalance-num-iterations` 等。

当前使用 DeepEP 等专用 A2A 后端时，`_handle_a2a_moe()` 会把 `ep_size` 调整为 `tp_size`，所以上例无需重复指定 `--ep-size 8`。参数约束以 `ServerArgs._handle_a2a_moe()`、`ServerArgs._handle_eplb_and_dispatch()` 和并行组初始化代码为准，避免依赖易漂移的源码行号。

---

## 9. 局限与权衡

1. **all-to-all 通信开销大**：dispatch + combine 通常包含两阶段跨 rank 交换；跨节点 EP 对 RoCE/IB 带宽和尾延迟要求很高，是 EP 的主要成本。
2. **负载不均**：热点 expert 会拖慢整个 EP group，EPLB 可以缓解但会增加统计、映射和权重重排的复杂度。
3. **切分约束**：physical expert 总数必须能被 `ep_size` 整除，即

$$
(E_{\text{logical}}+E_{\text{redundant}})\bmod ep\_size=0
$$

此外，`tp_size`、`ep_size`、`moe_dp_size` 和 `moe_tp_size` 之间还需满足 rank mesh 的整除关系，不同 A2A 和 MoE runner 后端可能有额外限制。
4. **与 batch 大小耦合**：token 少时 all-to-all 的小消息效率低，DeepEP low-latency 模式主要针对 decode 小 batch；token 多时则更关注吞吐、负载均衡和通信计算重叠。

---

## 参考与延伸

- 同目录：`TP.md`（张量并行）、`DP.md`（数据并行）、`PP.md`（流水线并行）、`DP_attention.md`。
- SGLang 代码：`python/sglang/srt/layers/moe/`（`ep_moe/`、`token_dispatcher/`）、`python/sglang/srt/eplb/`。
- 目录说明：`python/sglang/srt/layers/moe/README_zh.md`、`token_dispatcher/README_zh.md`、`ep_moe/README_zh.md`。
- 相关论文：DeepSeek-V3 / DeepEP（专家并行通信库）。

