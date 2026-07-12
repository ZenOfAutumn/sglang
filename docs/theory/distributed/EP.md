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

解决办法有两条：

- **TP**：把每个 expert 的 FFN 矩阵按张量维切分（`TP.md` §5.1 的做法）。但 expert 数量多时，每卡仍要存所有 expert 的分片，权重总量不降。
- **EP（专家并行）**：把**不同的 expert 整体分到不同卡**。768 个 expert 分到 8 卡，每卡只放 96 个；分到 128 卡，每卡只放 6 个。**每卡权重直接降到 $1/\text{EP}$**。

EP 的独特价值：**让每卡只加载一部分 expert，从而把巨大的 MoE 权重摊薄到多卡**，释放 HBM 给 KV Cache。代价是 token 需要跨卡路由到其 expert 所在的卡（all-to-all 通信）。

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

设总共 $E$ 个 expert、EP 路数为 $p$（即 `ep_size`），则**每卡持有 $E/p$ 个 expert 的完整权重**（称为 local experts）。

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

| 方式 | 切什么 | 每卡权重 | 通信 |
| --- | --- | --- | --- |
| **TP 切 expert** | 每个 expert 的 FFN 矩阵按张量维切 | 所有 expert 的分片（总量不降） | 每层 all-reduce |
| **EP** | 不同 expert 整体分到不同卡 | 只有 $E/p$ 个 expert（**降到 $1/p$**） | all-to-all dispatch + combine |

EP 用 all-to-all 通信换来了**每卡权重降到 $1/p$**——这是 MoE 大模型能部署的关键。

---

## 4. 数据流：dispatch → 专家计算 → combine

EP 的一次 MoE 前向分三步，中间两次 all-to-all：

### 4.1 Dispatch（分发）

每个 token 经 router 选出 top-k 个 expert，这些 expert 可能分散在不同卡上。**dispatch 用 all-to-all 把每个 token 发送到它所选 expert 所在的卡**：

- 输入：本卡的 token 及其 `topk_ids`；
- 输出：本卡收到的、路由到**本卡 local experts** 的所有 token（可能来自任意卡）。

### 4.2 专家计算

每卡在收到的 token 上跑自己的 local experts（分组 GEMM，grouped GEMM）。因为每卡只有 $E/p$ 个 expert，计算量和权重都只有 $1/p$。

### 4.3 Combine（合并）

**combine 用第二次 all-to-all 把专家输出送回 token 原来所在的卡**，每个 token 收齐它 k 个 expert 的输出后，按 `topk_weights` 加权求和得到最终输出。

```
Dispatch (all-to-all #1)          Combine (all-to-all #2)
 token → 其 expert 所在卡    →    expert 输出 → token 原所在卡
```

> SGLang 把 dispatch/combine 抽象成 `BaseDispatcher`（`token_dispatcher/base.py`），并为多种通信后端提供实现：标准、DeepEP、Mooncake、NIXL、Mori、FlashInfer、NPU FuseEP（见 §8）。

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

- 预分配**冗余专家槽位**（`ep_num_redundant_experts`），把热点 expert 复制到多张卡上；
- 周期性（`eplb_rebalance_num_iterations`）统计各 expert 访问频率（`expert_distribution_recorder_mode=stat`），据此**重新分配 expert 到卡的映射**；
- 分发算法如 `static_with_zero_expert`：静态分配保底 + 动态零专家（冗余副本）填充热点。

> 例：LongCat decode 集群 `ep_size=128, enable_eplb=true, ep_num_redundant_experts=128`——768 个路由 expert 分到 128 卡（每卡约 6 个），并用 128 个冗余槽位动态均衡热点。

相关代码在 `python/sglang/srt/eplb/`。

---

## 7. EP 与 TP / DP 的关系

EP 只作用于 **MoE 层**；模型的 attention 和稠密层仍用 TP / DP。实际部署常组合：

| 并行 | 作用对象 | 切什么 | 通信 |
| --- | --- | --- | --- |
| **TP** | attention、稠密 FFN | 层内张量维 | all-reduce |
| **EP** | **MoE 层** | 不同 expert 分到不同卡 | all-to-all（dispatch/combine） |
| **DP** | 整个模型副本 | batch 数据 | 请求分发 |

典型组合：一个 MoE 模型里 **attention 走 TP/DP-attention，MoE 走 EP**。SGLang 中：

- `attn_tp_size`：attention 的 TP；
- `ep_size`：MoE 的 EP 路数；
- `moe_dense_tp_size`：MoE 中稠密部分（如共享专家）的 TP。

> 例：LongCat prefill `ep_size=8`（节点内 EP），decode `ep_size=128`（全局跨节点 EP）——同一模型 attention 用 TP=8、MoE 用不同的 EP，二者正交。

---

## 8. SGLang 中的实现

### 8.1 MoE 层与 EP 实现：`python/sglang/srt/layers/moe/`

- `router.py` / `topk.py`：router 打分与 top-k 选择，产出 `topk_ids` / `topk_weights`。
- `ep_moe/layer.py`：专家并行 MoE 层 `DeepEPMoE`（继承 `FusedMoE`）及 `NpuFuseEPMoE`、`MoriEPMoE` 变体；`get_moe_impl_class` 选择实现。
- `ep_moe/kernels.py`：EP 所需 Triton kernel（permute/反排序、src2dst 映射、ep scatter、silu_and_mul 后量化等）。

### 8.2 token 分发后端：`python/sglang/srt/layers/moe/token_dispatcher/`

- `base.py`：`BaseDispatcher` 抽象 + dispatch/combine 协议、`DispatchOutput`/`CombineInput` 格式。
- `standard.py`：标准（非专用 a2a）分发器。
- `deepep.py`：DeepEP 后端（normal / low-latency 两种模式）。
- `mooncake.py` / `nixl.py` / `moriep.py` / `flashinfer.py` / `fuseep.py`：Mooncake / NIXL / Mori / FlashInfer / NPU FuseEP 各后端。

### 8.3 负载均衡：`python/sglang/srt/eplb/`

EPLB 的专家分布记录、再平衡算法与 expert-location 元数据管理。

### 8.4 启动参数：`python/sglang/srt/server_args.py`

```bash
python -m sglang.launch_server --model <MoE-model> \
    --tp-size 8 --ep-size 8 --moe-a2a-backend deepep --enable-eplb
```

- `--ep-size`（`server_args.py:852`）：EP 路数 $p$；
- `--moe-a2a-backend`（`:854`，如 `deepep`/`none`）：all-to-all 通信后端；
- `--moe-dense-tp-size`（`:907`）：MoE 稠密部分的 TP；
- EPLB 相关：`--enable-eplb`、`--ep-num-redundant-experts`、`--eplb-rebalance-num-iterations` 等。

---

## 9. 局限与权衡

1. **all-to-all 通信开销大**：dispatch + combine 是两次全交换通信，跨节点 EP（如 `ep_size=128`）对网络带宽要求极高（RoCE/IB），是 EP 的主要成本。
2. **负载不均**：热点 expert 拖累整体，必须靠 EPLB 缓解，增加系统复杂度。
3. **切分约束**：`ep_size` 需能整除 expert 数；冗余专家、量化 scale 的切分都需专门处理。
4. **与 batch 大小耦合**：token 少时 all-to-all 的小消息效率低，low-latency dispatch 模式（DeepEP LL）就是为 decode 小 batch 优化的。

---

## 参考与延伸

- 同目录：`TP.md`（张量并行）、`DP.md`（数据并行）、`PP.md`（流水线并行）、`DP_attention.md`。
- SGLang 代码：`python/sglang/srt/layers/moe/`（`ep_moe/`、`token_dispatcher/`）、`python/sglang/srt/eplb/`。
- 目录说明：`python/sglang/srt/layers/moe/README_zh.md`、`token_dispatcher/README_zh.md`、`ep_moe/README_zh.md`。
- 相关论文：DeepSeek-V3 / DeepEP（专家并行通信库）。

