# SGLang 并行学习计划（TP / DP / EP / PP / CP）

> 本文是 `docs/sglang_learning_plan_zh.md` 阶段 6「张量/专家并行」的**展开版**，专注推理侧并行。
> 沿用同一套「阅读 → 动手 → 自测题 → ✅ 通关标准」方法论。
> 所有路径相对仓库根目录，核心代码在 `python/sglang/srt/`。

---

## 零、先回答：为什么先学并行，再学 PD 分离

| 理由 | 说明 |
| --- | --- |
| **依赖单向** | PD 分离的 P 节点、D 节点内部各自都是 TP/DP/EP 组合，且两端并行度常常不同。KV 传输必须对齐两端实际的 KV Cache 布局与并行 rank 映射；MHA/GQA 可能按 KV head 分片，而 MLA latent 在 attention-TP 子组内可能复制、在 attention-DP 维度上按请求分布 |
| **公共底座** | 不管上不上 PD，都要先能回答「这个模型 8 卡怎么切」。unified 单实例就能跑通全部并行维度 |
| **可调试性** | 并行问题单机可复现；PD 问题是跨节点时序问题，链路长、日志分散在两组进程 |

**结论**：本计划（约 11 天，每天 4 小时）完成后，再进入 PD 分离，成本最低。

---

## 一、五种并行的定位（先建立全局观）

在读任何代码前，先记住这张表——**每种并行切的是什么、代价是什么**：

| 并行 | 切分对象 | 解决的问题 | 主导通信原语 | 每层通信次数 | SGLang 参数 |
| --- | --- | --- | --- | --- | --- |
| **TP** 张量并行 | 权重矩阵（层内） | 单卡放不下权重 | All-Reduce | 2 次/层（attn + MLP 各一次） | `--tp-size` |
| **DP** 数据并行 | 请求（副本级） | 提高吞吐、扩展并发 | 正常前向无副本间模型计算通信 | 0 | `--dp-size` |
| **DP Attention** | attention 的请求/token 维 | MLA 模型下 KV 被 TP 冗余复制 | All-Reduce 或 Reduce-Scatter + All-Gather；部分 scatter 为本地拷贝 | 依数据布局与后端而定 | `--enable-dp-attention` |
| **EP** 专家并行 | MoE 的 expert | MoE 权重巨大且稀疏激活 | 典型 token-dispatch 后端为 All-to-All | 通常 2 个交换阶段（dispatch + combine），具体依后端而定 | `--ep-size` |
| **PP** 流水并行 | 层（层间） | 单机放不下、跨机扩展 | P2P (send/recv) | 1 次/stage 边界 | `--pp-size` |
| **CP** 上下文并行 | 序列长度维 | 超长上下文 | All-Gather / Ring | 视实现 | `--attn-cp-size` / `--enable-prefill-cp` |

**关键区分（最容易混淆的两点）**：

1. **DP ≠ DP Attention**。前者是「整个模型复制多份，请求分流」，正常前向时副本间不做模型计算通信；后者是「同一个模型内部，attention 层按请求/token 切、MoE 层按 TP/EP 切」，层内需要在不同数据布局之间转换。名字像，机制完全不同。
2. **TP 和 EP 可以叠加**。DeepSeek 类模型的典型配置就是 attention 走 DP、MoE 走 EP，三者在同一次前向里协同。

通信量的基本量纲（设隐藏维为 $h$、序列长度为 $s$、TP 组大小为 $N$）：

$$
V_{\mathrm{All\text{-}Reduce}}
= 2 \cdot \frac{N-1}{N} \cdot s \cdot h \cdot \operatorname{sizeof}(\mathrm{dtype}).
$$

这个式子解释了为什么通常应优先把 TP 放在机内 NVLink 域；SGLang 也支持多节点 TP，但需要承担更高的跨机通信代价。这是后面并行度选型的基础。

---

## 二、仓库已有资料清单（重要：不要重复造轮子）

本仓库 `docs/theory/distributed/` 下已有多篇中文原理文档，包含数值示例与手算推演。本计划的定位是**串联这些文档 + 补充源码落点 + 给出动手实验**，而不是重写原理。文档持续演进，以下行数仅作大致参考。

| 文档 | 行数 | 内容要点 |
| --- | --- | --- |
| `docs/theory/distributed/TP.md` | 428 | 列/行并行、黄金组合、MLP 手算、head 切分、GQA/MQA、词表并行、通信量分析 |
| `docs/theory/distributed/DP.md` | 479 | 训练 DP vs 推理 DP、DataParallelController、4 种负载均衡策略、GPU/rank 布局数值示例、DPBudget |
| `docs/theory/distributed/DP_attention.md` | 约 585 | 动机、rank 布局公式、gather→MoE→scatter 数据流、MAX_LEN/SUM_LEN 两种 padding |
| `docs/theory/distributed/EP.md` | 266 | MoE 回顾、按 expert 切分、dispatch/combine、通信量、EPLB 负载均衡 |
| `docs/theory/distributed/PP.md` | 312 | 流水切分、气泡分析、micro-batch |
| `docs/theory/distributed/CP.md` | 208 | 上下文并行 |
| `docs/theory/distributed/collective_communication.md` | 918 | **9 种原语总表**、Ring All-Reduce 通信量与**通信下界证明**、关键恒等式、各并行主导原语 |
| `docs/theory/distributed/cost_model.md` | 473 | **代价模型专题**：$\alpha\text{-}\beta$、LogP/LogGP、$\alpha\text{-}\beta\text{-}\gamma$、带宽层次、对分带宽、Roofline、BSP |
| `docs/advanced_features/expert_parallelism_zh.md` | 199 | EP 部署实践 |
| `docs/advanced_features/pipeline_parallelism_zh.md` | 116 | PP 部署实践 |
| `python/sglang/srt/managers/README_data_parallel_controller_zh.md` | 约 418 | DP 控制器源码导读 |
| `python/sglang/srt/distributed/README_zh.md` | 18 | distributed 目录说明 |
| `python/sglang/srt/distributed/device_communicators/README_zh.md` | 25 | 通信后端说明 |

> **阅读顺序建议**：每个阶段先读对应的 `theory/` 文档建立原理，再按本计划的「源码落点」去读实现，最后做实验验证。

---

## 三、源码地图

```
python/sglang/srt/
├── distributed/                        ★ 通信与进程组基础设施
│   ├── parallel_state.py               ★★ 核心：GroupCoordinator、进程组初始化
│   │     ├── class GroupCoordinator            # 所有进程组的统一封装
│   │     ├── init_distributed_environment()    # 建立 world
│   │     ├── initialize_model_parallel()       # 切出 TP/PP/EP 各组
│   │     ├── get_tp_group() / get_pp_group()   # 取组
│   ├── communication_op.py             # 通信算子薄封装
│   ├── device_communicators/           # 后端实现
│   │   ├── pynccl.py                   # NCCL
│   │   ├── custom_all_reduce.py        # 定制 all-reduce（小消息低延迟）
│   │   ├── quick_all_reduce.py
│   │   ├── shm_broadcast.py            # 共享内存广播
│   │   └── torch_symm_mem.py / pymscclpp.py
│   └── naive_distributed.py
│
├── layers/
│   ├── linear.py                       ★★ TP 的落地：Column/RowParallelLinear
│   ├── vocab_parallel_embedding.py     ★ 词表切分
│   ├── dp_attention.py                 ★★ DP Attention 全部逻辑
│   │     ├── initialize_dp_attention()
│   │     ├── compute_dp_attention_world_info()
│   │     ├── _dp_gather_via_all_reduce() / _dp_gather_via_all_gather()
│   │     ├── dp_gather_partial() / dp_scatter()
│   │     └── DpPaddingMode (MAX_LEN / SUM_LEN)
│   └── moe/                            ★★ EP 的落地
│       ├── ep_moe/                     # EP MoE 层
│       ├── token_dispatcher/           # all-to-all 分发后端（deepep 等）
│       └── fused_moe_triton/
│
├── eplb/                               ★ 专家负载均衡
│   ├── eplb_algorithms/                # 重排算法
│   └── eplb_simulator/                 # 离线模拟
│
├── managers/
│   └── data_parallel_controller.py     ★★ 副本级 DP 的控制器
│
└── model_executor/
    └── model_runner.py                 # 并行初始化的调用方
```

---

## 四、分阶段计划（约 11 天）

> 节奏：每天 4 小时。**硬件建议**：至少 2 卡才能做 TP 实验；EP 实验需要 MoE 模型（如 Qwen3-30B-A3B）；无多卡时看「无卡替代方案」。

### 阶段 P0：集合通信原语（第 1 天上半天）— 地基

**目标**：能说清文档总表中各类通信原语的语义与通信量，这是理解所有并行的前置。

- **阅读**：`docs/theory/distributed/collective_communication.md` 全文（重点是 §1 总表、§12 关键恒等式；行有余力再看 §10 的通信下界证明、§8.1 Ring vs Direct）。
- **选读**：`docs/theory/distributed/cost_model.md`——搞清楚 $\alpha\text{-}\beta$ 模型与**半带宽点 $n_{1/2}=\alpha/\beta$**（§2），这是理解“为什么 decode 阶段小消息要用 one-shot allreduce”的关键。
- **源码落点**：`distributed/parallel_state.py` 的 `GroupCoordinator` 类，看 `all_reduce` / `all_gather` / `reduce_scatter` 三个方法的签名。
- **动手**：写一个 20 行的纯 PyTorch 脚本，用 `torch.distributed` 起 2 进程，分别跑一次 all-reduce 和 all-gather，打印各 rank 的输入输出。**这一步能让抽象概念立刻具象化。**
- **自测题**：
  1. All-Reduce = Reduce-Scatter + All-Gather，这个恒等式为什么成立？对通信量有什么意义？
  2. Ring All-Reduce 的通信量为什么是 $2\cdot\frac{N-1}{N}\cdot D$ 而不是 $N \cdot D$？
  3. All-to-All 与 All-Gather 的区别？为什么 EP 用前者？
- ✅ **通关标准**：能默画出常用通信原语的数据流示意图；能说出 TP/EP/PP 各自主导哪类原语。

---

### 阶段 P1：TP 张量并行（第 1 天下半天 – 第 2 天）★ 最重要

**目标**：理解权重如何切分、all-reduce 插在哪、为什么通常优先避免 TP 跨机。

- **阅读顺序**：
  1. `docs/theory/distributed/TP.md` §1–§5（原理 + MLP 手算，务必跟着算一遍 §4 的数值示例）。
  2. `srt/layers/linear.py`：`ColumnParallelLinear`、`RowParallelLinear`、`QKVParallelLinear`、`MergedColumnParallelLinear`。**重点看 `forward` 中 $\mathrm{tp\_size}>1$ 时的通信调用位置**。
  3. `srt/layers/vocab_parallel_embedding.py`：`VocabParallelEmbedding` / `ParallelLMHead` 的词表切分与 masked all-reduce。
  4. `srt/models/llama.py`：`LlamaMLP`、`LlamaAttention` —— 看真实模型如何组合这些并行层。
  5. `docs/theory/distributed/TP.md` §6–§9（head 切分、GQA/MQA、通信量、源码索引）。

- **核心心智模型（必须内化）**：

```text
MLP:   Column(gate/up) ──► 激活 ──► Row(down) ──► all-reduce
        无通信（列切）              行切后需要求和

Attn:  QKV(列切，按 head) ──► attention ──► O(行切) ──► all-reduce
```

「列并行 → 行并行」的黄金组合使**每个 Attention 或 MLP 子层只需在末尾做 1 次 all-reduce**；标准 Transformer block 通常仍有 2 次（Attention 的 `o_proj` 一次、MLP 的 `down_proj` 一次）。这是 Megatron-LM 的核心设计。

- **动手实验**：
  1. 同一模型分别用 `--tp-size 1` 和 `--tp-size 2` 启动，对比 `/get_server_info` 里的 `max_total_num_tokens`。权重显存被摊薄后容量通常会上升，但幅度取决于 KV head 切分、模型结构、通信缓冲区、CUDA Graph 与 attention backend，不保证线性翻倍。
  2. 用 `python -m sglang.bench_one_batch` 对比两者的 decode 延迟，观察 TP 的收益与通信开销的权衡。
  3. 在 `RowParallelLinear.forward` 加一行 `logger.debug`，确认 all-reduce 每层调用一次。

- **自测题**：
  1. 为什么 attention 按 head 切而不是按 hidden 维切？GQA 下 KV head 数少于 TP size 时怎么办？
  2. `ColumnParallelLinear` 的输出为什么不需要通信，而 `RowParallelLinear` 需要？
  3. LayerNorm / RMSNorm 为什么不切分（每卡冗余计算）？
  4. TP size 从 2 增到 8，通信量如何变化？为什么 TP 一般不跨机？

- **常见坑**：`linear.py` 里有多个 `*ParallelLinear` 变体，`MergedColumnParallelLinear` 是把 gate/up 合并成一次 GEMM 的优化，不要误以为是另一种并行方式。

- ✅ **通关标准**：能手画一个 transformer block 的 TP 切分图（标出 4 个 GEMM 的切法 + all-reduce 位置）；能手算 §4 的 MLP 数值示例；能解释为什么通常优先避免 TP 跨机。

---

### 阶段 P2：DP 副本级数据并行（第 3 天）

**目标**：理解推理 DP 与训练 DP 的本质区别，以及请求如何被分流。

- **阅读顺序**：
  1. `docs/theory/distributed/DP.md` §1–§3（重点是 §3「训练 DP vs 推理 DP」这个关键区分）。
  2. `python/sglang/srt/managers/README_data_parallel_controller_zh.md`（约 418 行源码导读，篇幅以后续版本为准）。
  3. `srt/managers/data_parallel_controller.py`：`event_loop`、`launch_dp_schedulers`、负载均衡策略。
  4. `docs/theory/distributed/DP.md` §5–§8（4 种负载均衡策略、GPU/rank 布局数值示例、DPBudget）。

- **关键认知**：推理 DP **没有梯度同步**，正常前向时副本之间不做模型计算通信；控制面仍负责请求分发、状态与控制消息协调。所以推理 DP 的主要难点不在梯度通信，而在**负载均衡**——这与训练 DP 截然不同。

- **与你已写文档的衔接**：`docs/metrics/v1_loads_zh.md` 里的 `num_total_tokens` 正是 `total_tokens` 均衡策略的决策依据：

$$
\mathrm{num\_total\_tokens}
= \mathrm{num\_used\_tokens}
+ \sum_{\mathrm{req} \in \text{等待队列}} \mathrm{req.seqlen}.
$$

- **动手实验**：
  1. `--dp-size 2` 启动，并发打压，`curl /v1/loads?include=core` 观察两个 rank 的 `num_total_tokens` 是否均衡。
  2. 对比 `--load-balance-method round_robin` 与 `total_tokens`，用**长短混合**的请求负载（这是两者差异最明显的场景）观察倾斜程度。

- **自测题**：
  1. 推理 DP 与训练 DP 最本质的区别是什么？
  2. `round_robin` 在什么负载下会退化？为什么 `total_tokens` 更鲁棒？
  3. $\mathrm{tp\_size}=4,\ \mathrm{dp\_size}=2$ 时共需几张卡？各进程的 `gpu_id` 如何计算？（对照 DP.md §6 的数值示例）
  4. 为什么 DPBudget 需要「推测式 +1」？

- ✅ **通关标准**：能说清 DP 控制器的请求分发链路；能对照数值示例算出任意 `tp/dp/pp` 组合下的 GPU 布局；能用 `/v1/loads` 数据判断负载是否倾斜。

---

### 阶段 P3：DP Attention（第 4–5 天）★ 难点

**目标**：理解为什么 MLA 模型需要它，以及 gather/scatter 的数据流。

> 这是**最容易混淆**的一块，务必在阶段 P1（TP）和 P2（DP）都通关后再来。

- **动机（一句话）**：MLA（DeepSeek）的 KV latent 不按 head 组织，普通 TP 难以有效切分，导致同一 attention-TP 子组内复制 KV。DP Attention 沿请求/token 维把 attention 切成多个 DP 组，每组只保存自己请求的 KV；组内仍可保留较小的 attention TP，MoE 则继续走 TP/EP。

- **阅读顺序**：
  1. `docs/theory/distributed/DP_attention.md` §1–§2（动机与核心思想）。
  2. `srt/layers/dp_attention.py`：
     - `compute_dp_attention_world_info()` / `initialize_dp_attention()` —— rank 布局
     - `_dp_gather_via_all_reduce()` vs `_dp_gather_via_all_gather()` —— 两种 gather 实现
     - `dp_gather_partial()` / `dp_scatter()`
     - `DpPaddingMode`（`MAX_LEN` / `SUM_LEN`）
  3. `docs/theory/distributed/DP_attention.md` §3–§7（rank 公式、数据流、padding 模式、与普通 TP 的区别）。

- **核心数据流**（下图用 $\mathrm{attn\_tp\_size}=1$ 的纯 attention-DP 情形简化展示；$\mathrm{attn\_tp\_size}>1$ 时每个 DP 组内还包含多个 attention-TP rank）：

```text
        ┌── rank0: attn(自己的 batch 分片, 自己的 KV) ──┐
输入 ───┤                                              ├── gather ──► MoE(TP/EP) ──► scatter ──► 输出
        └── rank1: attn(自己的 batch 分片, 自己的 KV) ──┘
         DP 组间 KV 独立；attn-TP 组内可能复制       全局 token 一起过 MoE
```

- **自测题**：
  1. 为什么 MLA latent 难以按普通 head TP 有效切分？GQA 的 KV 能否有效切分又如何受 KV head 数与 TP size 约束？
  2. `MAX_LEN` 与 `SUM_LEN` 两种 padding 各自的通信量与显存代价？何时选哪个？
  3. 为什么 `--enable-dp-attention` 要求 $\mathrm{tp\_size}$ 能被 $\mathrm{dp\_size}\times\mathrm{attn\_cp\_size}$ 整除？何时才会取 $\mathrm{dp\_size}=\mathrm{tp\_size}$？
  4. gather 的两种实现（all-reduce 版 / all-gather 版）分别在什么场景更优？

- **动手实验**：优先用 DeepSeek-V2/V3/R1、Kimi K2 等 MLA 模型，对比开关 `--enable-dp-attention` 时 `/get_server_info` 的 `max_total_num_tokens`。MLA 模型通常能明显减少 KV 复制；对 GQA/MQA 模型，收益取决于 KV head 数、并行配置与 backend，需实测确认。

- **无卡替代方案**：读 `DP_attention.md` §4.2 的数值直觉，用纸笔推演 2 卡场景下 token 的 gather/scatter 过程。

- ✅ **通关标准**：能画出 DP Attention 一层的数据流（含 gather/scatter 位置）；能解释它与「副本级 DP」「普通 TP」三者的区别；能说清 KV 显存的节省来源。

---

### 阶段 P4：EP 专家并行（第 6–7 天）★ 重点

**目标**：理解 MoE 如何按 expert 切分、all-to-all 通信、以及负载不均问题。

> 详细的按小时源码路线、手算题和验收项见 `docs/distribute/expert_parallelism_learning_plan_zh.md`。

- **阅读顺序**：
  1. `docs/theory/distributed/EP.md` §1–§5（含 §5 手算一次 EP 路由，务必跟着算）。
  2. `srt/layers/moe/ep_moe/`：EP MoE 层实现。
  3. `srt/layers/moe/token_dispatcher/`：all-to-all 分发后端（DeepEP 等），对照 `--moe-a2a-backend`。
  4. `docs/theory/distributed/EP.md` §6（EPLB 负载均衡）→ `srt/eplb/`。
  5. `docs/advanced_features/expert_parallelism_zh.md`（部署实践）。

- **核心数据流**：

```text
router 打分 ──► dispatch（典型为 all-to-all）──► 各卡算自己的 expert ──► combine（典型为 all-to-all）──► 输出
             按 expert 归属重排 token                         把结果送回原 token 所在卡
```

- **EP 的核心难点是负载不均**：token 路由到哪个 expert 由数据决定，热门 expert 所在的卡会成为瓶颈。EPLB 通过**统计专家负载 + 周期性重排/复制专家**来缓解。这是 EP 区别于 TP 的最大特点——**TP 的负载是静态均衡的，EP 是动态倾斜的**。

- **自测题**：
  1. EP 与 TP 切 MoE 分别有什么优劣？为什么大规模 MoE 倾向 EP？

     <details>
     <summary>参考答案</summary>

     设并行度为 $p$。EP 和 TP 都能把单卡专家权重显存降到约 $1/p$，但切分维度和通信方式完全不同。

     **EP：沿 expert 维切分。** 每个 rank 只保存约 $1/p$ 的完整专家：

     $$
     M_{\mathrm{EP}} \approx \frac{M_{\mathrm{expert}}}{p}.
     $$

     Router 为每个 token 选择 Top-K 专家后，系统先通过 All-to-All 将 token 分发到目标专家所在 rank；每个专家在本地独立执行 MLP；随后再通过 All-to-All 将结果送回 token 的来源 rank，并按路由权重合并。

     - **优点**：单个专家计算无需跨卡协作；能随专家数量增加扩展模型容量；通信量主要取决于激活和 token-expert 对数量，而非总专家参数量。
     - **缺点**：需要 dispatch/combine 两阶段 All-to-All；容易出现专家负载不均；跨节点时受网络拓扑影响明显；单个专家收到的 token 太少时，GEMM 的 $M$ 维较小，GPU 利用率会下降。

     **TP：沿每个专家权重的张量维切分。** 每个 rank 持有所有专家的约 $1/p$ 权重分片：

     $$
     M_{\mathrm{TP}} \approx \frac{M_{\mathrm{expert}}}{p}.
     $$

     对典型专家 MLP：

     $$
     h = \operatorname{SwiGLU}\!\left(xW_{\mathrm{gate}},\,xW_{\mathrm{up}}\right),
     \qquad
     y = hW_{\mathrm{down}},
     $$

     每个专家的计算都需要多个 TP rank 协作，并在行并行层等位置执行 All-Reduce 或 Reduce-Scatter。

     - **优点**：无需按专家归属跨卡发送 token；各 rank 通常静态均衡；单个专家大到一张卡放不下时，TP 是必要手段；适合 NVLink/NVSwitch 等节点内高速互联。
     - **缺点**：专家 MLP 的 collective 位于关键路径；TP 度增大后，每个权重分片和 GEMM 变小，通信占比上升；跨节点 TP 的 All-Reduce 代价较高。

     > 注意：不能简单说“每个 token 需要 $2K$ 次 All-Reduce”。实现通常将 token 按专家分组，批量执行 grouped GEMM 和 collective；Top-K 增大会增加 token-expert 对的计算与通信量，但 collective 调用次数不一定线性增长为 $2K$。

     大规模 MoE 通常满足：

     $$
     E \gg K,
     $$

     其中 $E$ 是专家总数，$K$ 是每个 token 激活的专家数。EP 能直接利用这种稀疏性：参数按专家分散，每个 token 只发送给命中的 $K$ 个专家，其通信量近似为

     $$
     V_{\mathrm{EP}} \propto T \times K \times H,
     $$

     其中 $T$ 是 token 数，$H$ 是隐藏维度。该通信量不直接随总专家数 $E$ 增长；增加专家数时，可通过增加 EP rank 扩展模型容量。相比之下，TP 会把每个专家都切到多个 rank，任意被命中的专家都需要 TP rank 协同，不能充分利用“专家很多、每个 token 只激活少数专家”的稀疏结构。

     因此，大规模 MoE 通常采用：

     $$
     \text{EP 为主} + \text{较小的节点内 TP} + \text{DP / DP Attention}.
     $$

     一句话总结：**EP 扩展专家容量，通信 token；TP 切分单个专家，通信中间激活。大规模 MoE 专家多但每个 token 只激活少数专家，因此更倾向 EP。**

     </details>

  2. 典型 token-dispatch EP 为什么使用 all-to-all 而不是 all-gather？不同 MoE backend 是否可能使用其他等价通信组合？
  3. 什么是专家负载不均？EPLB 用什么策略缓解？
  4. `--ep-size` 与 `--tp-size` 的关系？两者能否不等？

- **动手实验**：用 MoE 模型开 `--ep-size`，观察 `sglang:eplb_balancedness`（需设置 `SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC=1`）与 `sglang:eplb_gpu_physical_count`（还需设置 `SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL` 为大于 0 的采样间隔）指标，直观看到专家负载分布。

- ✅ **通关标准**：能手算 EP.md §5 的路由示例；能说清 dispatch→compute→combine 三阶段；能解释 EPLB 要解决的问题与基本思路。

---

### 阶段 P5：PP 流水并行 + CP 上下文并行（第 8 天）— 了解即可

> 这两者在推理侧使用频率低于 TP/DP/EP，**掌握动机与代价即可**，不必深挖实现。

- **PP**：
  - 阅读 `docs/theory/distributed/PP.md`（重点是气泡分析）+ `docs/advanced_features/pipeline_parallelism_zh.md`。
  - 源码：`distributed/parallel_state.py` 的 `get_pp_group()`，以及 scheduler 里的 micro-batch 处理。
  - 关键认知：PP 的代价是**流水气泡**，气泡率 $\displaystyle \approx \frac{P-1}{M+P-1}$（$P$ 为 stage 数，$M$ 为 micro-batch 数）。推理场景 batch 小，气泡相对更痛，所以 PP 优先级低于 TP。
- **CP**：阅读 `docs/theory/distributed/CP.md`，理解超长上下文下按序列维切分的动机即可。
  - 参数注意：**没有 `--cp-size`**。实际是 `--attn-cp-size`（等价 `--attention-context-parallel-size`）与 `--enable-prefill-cp`；旧的 `--enable-prefill-context-parallel` / `--enable-dsa-prefill-context-parallel` 已标记弃用。
  - 统一的 prefill CP 路径会按模型与 `cp_strategy` 自动调整 `attn_cp_size`，当前 DSA/MLA 路径通常令其为 $\mathrm{attn\_cp\_size}=\mathrm{tp\_size}/\mathrm{dp\_size}$。该特性仍处于实验阶段；单机限制、硬件、attention backend、batch size 与并行组合约束因模型和策略而异，应以 `server_args.py` 当前校验为准（例如 DSA CP 当前限制 $\mathrm{tp\_size}\leq 8$，MLA CP 主要在 Hopper + FA3 上验证）。

- ✅ **通关标准**：能说清 PP 的气泡来源与缓解手段；能说出 PP 相对 TP 的适用场景（跨机、单机放不下）。

---

### 阶段 P6：组合并行与选型（第 9–10 天）★ 实战收口

**目标**：面对一个真实模型 + 一批 GPU，能给出并行度方案并说明理由。

- **阅读**：
  1. `distributed/parallel_state.py` 的 `initialize_model_parallel()` —— **看清多个并行维度的进程组是如何同时切出来的**，这是把前面所有阶段串起来的关键函数。
  2. `srt/model_executor/model_runner.py` 中并行初始化的调用点。
  3. `srt/server_args.py` 里各并行参数的校验逻辑（搜 `tp_size` / `ep_size` 的一致性检查）。

- **典型配置对照表**：

| 场景 | 推荐配置 | 理由 |
| --- | --- | --- |
| 稠密模型单机 8 卡 | `--tp-size 8` | 简单直接，NVLink 内 all-reduce 快 |
| 稠密模型，吞吐优先 | `--tp-size 2 --dp-size 4` | 小 TP 减少通信，多副本提吞吐 |
| MoE（DeepSeek 类，8 卡纯 attention-DP 示例） | `--tp-size 8 --dp-size 8 --enable-dp-attention --ep-size 8` | attention 沿 8 个请求分片减少 MLA KV 复制，MoE 走 EP 分摊专家权重；实际配置需结合模型与 backend 校验 |
| 超大模型跨机 | `--tp-size 8 --pp-size 2` | 机内 TP、机间 PP（避免跨机 all-reduce） |

- **选型原则**（按优先级）：
  1. **先满足「放得下」**：权重 + KV + 激活 ≤ 显存。放不下就加 TP（机内优先），再不够上 PP（跨机）。
  2. **优先避免 TP 跨机**：SGLang 支持多节点 TP，但跨机 all-reduce 通常显著慢于机内 NVLink；只有容量、拓扑或部署约束需要时再采用，并以实测评估。
  3. **在模型放得下时优先比较小 TP + 大 DP**：标准 Transformer block 的 Attention 与 MLP 通常各有一次 TP all-reduce，而推理 DP 的正常前向无需副本间模型计算通信；但最终选择仍受单请求延迟、KV 切分、batch、模型结构与网络拓扑影响。
  4. **MoE 优先评估 EP**：EP 能让每卡只持有部分专家，但会引入 dispatch/combine 通信与动态负载不均；是否优于 TP 取决于专家规模、batch、网络和 MoE backend。

- **动手实验（本阶段核心产出）**：固定模型与卡数，跑 3 组配置的 `bench_serving`，记录吞吐/TTFT/ITL，输出一张对比表。例如 8 卡下对比 `tp8` / `tp4+dp2` / `tp2+dp4`。

- **自测题**：
  1. 给定 70B 模型 + 8×A100(80G)，如何切？如果换成 2 机 16 卡呢？
  2. 为什么在模型放得下时值得比较“小 TP + 大 DP”？什么情况下大 TP 仍然更合适？
  3. $\mathrm{tp\_size}=8,\ \mathrm{dp\_size}=2,\ \mathrm{pp\_size}=2$ 需要多少卡？各进程组如何划分？

- ✅ **通关标准**：能针对任意「模型 + 卡数」给出并行方案并讲清取舍；能用实测数据支撑选型结论；能读懂 `initialize_model_parallel()` 里各进程组的切分逻辑。

---

### 阶段 P7：并行问题排查（第 11 天）

**目标**：能定位多卡场景的 hang、OOM、性能不达标。

- **配套 skill**（本仓库 `.claude/skills/` 已有现成资产，遇到对应场景直接用）：
  - `debug-distributed-hang` —— 分布式 hang 排查（py-spy / watchdog / 二分定位发散点）
  - `llm-torch-profiler-analysis` —— profiler trace 分析通信与计算重叠
  - `generate-profile` —— 生成 e2e trace

- **排查套路**：

| 症状 | 首选手段 | 常见根因 |
| --- | --- | --- |
| 多卡 hang | `py-spy dump` 各 rank，比对栈 | 某 rank 走了不同分支，集合通信参与者不齐 |
| TP 后吞吐反降 | profiler 看 all-reduce 占比 | 通信开销 > 并行收益，TP 过大或跨机 |
| DP 负载倾斜 | `/v1/loads?include=core` 比对各 rank | 均衡策略不适配负载特征 |
| EP 某卡打满 | `sglang:eplb_*` 指标 | 专家负载不均，需 EPLB |
| 多卡 OOM | 对比各 rank 显存 | 可能是 KV 布局未有效切分、DP attention 配置不当、CUDA Graph/通信缓冲过大或负载倾斜 |

- **关键原则**：**集合通信要求所有 rank 参与**。绝大多数分布式 hang 的根因都是「某个 rank 因为数据依赖的分支走了不同路径，没进入同一个通信调用」。定位方法是二分查找第一个状态发散点。

- ✅ **通关标准**：能用 py-spy 抓取多 rank 栈并判断卡在哪个集合通信；能从 profiler trace 里量化通信占比。

---

## 五、进度追踪表

（以下用时均按**每天 4 小时**换算）

| 阶段 | 主题 | 建议用时 | 状态 | 产出物 |
| --- | --- | --- | --- | --- |
| P0 | 集合通信原语 | 0.5 天（2h） | ✅ | 常用原语手绘图 + 2 进程通信脚本 |
| P1 | TP 张量并行 ★ | 1.5 天（6h） | ✅ | transformer block TP 切分图 + MLP 手算 |
| P2 | DP 副本级并行 | 1 天（4h） | ✅ | GPU/rank 布局推算 + 负载均衡对比数据 |
| P3 | DP Attention ★ | 2 天（8h） | ☐ | gather/scatter 数据流图 + KV 容量对比 |
| P4 | EP 专家并行 ★ | 2 天（8h） | ☐ | EP 路由手算 + EPLB 指标观察记录 |
| P5 | PP + CP | 1 天（4h） | ☐ | 气泡率公式推导 + 适用场景笔记 |
| P6 | 组合并行与选型 ★ | 2 天（8h） | ☐ | **3 组配置的 bench 对比表** |
| P7 | 并行问题排查 | 1 天（4h） | ☐ | 一次 py-spy 或 profiler 实操记录 |

> **里程碑**：
> - ✅ 第 3 天末（P0–P2）：能说清 TP 与 DP 的区别，能算 GPU 布局。**已达成**
> - 第 8 天末（P3–P5）：能说清五种并行各自的切分对象与通信代价。
> - 第 11 天末（P6–P7）：能独立做并行选型并排查多卡问题 → **可以进入 PD 分离学习**。

---

## 六、学完之后：衔接 PD 分离

完成本计划后，按以下顺序进入 PD：

1. `python/sglang/srt/disaggregation/README_zh.md`（架构总览）
2. `docs/sglang_learning_plan_zh.md` §2.2.1 的 PD 时间轴（各阶段与队列的名词解释）
3. `docs/advanced_features/pd_disaggregation_zh.md`
4. `srt/disaggregation/prefill.py` / `decode.py`

**此时你会立刻理解**：

- P 节点与 D 节点为什么可以配不同的 TP/DP（因为并行度是实例内部的事）；
- KV 传输为什么必须对齐两端实际的 KV Cache 布局与并行 rank 映射：MHA/GQA 可能按 KV head 分片，而 MLA latent 在 attention-TP 子组内可能复制、在 attention-DP 维度上按请求分布；
- `/v1/loads` 的 `disaggregation` 段那几个队列字段（`prefill_bootstrap_queue_reqs`、`decode_prealloc_queue_reqs` 等）对应 PD 状态机的哪个阶段 —— 这部分你在 `docs/metrics/v1_loads_zh.md` 已经写过。

---

## 七、参考资源

- 本仓库：`docs/theory/distributed/`（原理，含数值示例）、`docs/advanced_features/`（部署实践）
- 官方文档：<https://docs.sglang.io/>
- 大规模 EP 博客（强烈推荐）：<https://lmsys.org/blog/>
- Megatron-LM 论文（TP 的理论源头）：*Efficient Large-Scale Language Model Training on GPU Clusters*
- DeepSeek-V3 技术报告（MLA + DP Attention + 大规模 EP 的工业实践）

