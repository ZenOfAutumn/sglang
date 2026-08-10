# SGLang 并行学习计划（TP / DP / EP / PP / CP）

> 本文是 `docs/sglang_learning_plan_zh.md` 阶段 6「张量/专家并行」的**展开版**，专注推理侧并行。
> 沿用同一套「阅读 → 动手 → 自测题 → ✅ 通关标准」方法论。
> 所有路径相对仓库根目录，核心代码在 `python/sglang/srt/`。

---

## 零、先回答：为什么先学并行，再学 PD 分离

| 理由 | 说明 |
| --- | --- |
| **依赖单向** | PD 分离的 P 节点、D 节点内部各自都是 TP/DP/EP 组合，且两端并行度常常不同（P 用大 TP 压 TTFT，D 用大 DP 提吞吐）。不懂 TP，就无法理解 KV 为什么要按 TP rank 切分传输 |
| **公共底座** | 不管上不上 PD，都要先能回答「这个模型 8 卡怎么切」。unified 单实例就能跑通全部并行维度 |
| **可调试性** | 并行问题单机可复现；PD 问题是跨节点时序问题，链路长、日志分散在两组进程 |

**结论**：本计划（约 11 天，每天 4 小时）完成后，再进入 PD 分离，成本最低。

---

## 一、五种并行的定位（先建立全局观）

在读任何代码前，先记住这张表——**每种并行切的是什么、代价是什么**：

| 并行 | 切分对象 | 解决的问题 | 主导通信原语 | 每层通信次数 | SGLang 参数 |
| --- | --- | --- | --- | --- | --- |
| **TP** 张量并行 | 权重矩阵（层内） | 单卡放不下权重 | All-Reduce | 2 次/层（attn + MLP 各一次） | `--tp-size` |
| **DP** 数据并行 | 请求（副本级） | 提高吞吐、扩展并发 | 无（副本间不通信） | 0 | `--dp-size` |
| **DP Attention** | attention 的 batch 维 | MLA 模型下 KV 被 TP 冗余复制 | All-Gather + Reduce-Scatter | 2 次/层 | `--enable-dp-attention` |
| **EP** 专家并行 | MoE 的 expert | MoE 权重巨大且稀疏激活 | All-to-All | 2 次/MoE 层（dispatch + combine） | `--ep-size` |
| **PP** 流水并行 | 层（层间） | 单机放不下、跨机扩展 | P2P (send/recv) | 1 次/stage 边界 | `--pp-size` |
| **CP** 上下文并行 | 序列长度维 | 超长上下文 | All-Gather / Ring | 视实现 | `--attn-cp-size` / `--enable-prefill-cp` |

**关键区分（最容易混淆的两点）**：

1. **DP ≠ DP Attention**。前者是「整个模型复制多份，请求分流」，副本间**完全不通信**；后者是「同一个模型内部，attention 层按 batch 切、MoE 层按 TP/EP 切」，层内**需要 gather/scatter**。名字像，机制完全不同。
2. **TP 和 EP 可以叠加**。DeepSeek 类模型的典型配置就是 attention 走 DP、MoE 走 EP，三者在同一次前向里协同。

通信量的基本量纲（设隐藏维 $h$、序列长 $s$、TP 组大小 $N$）：

$$
\text{All-Reduce 单次通信量} = 2 \cdot \frac{N-1}{N} \cdot s \cdot h \cdot \text{sizeof(dtype)}
$$

这个式子解释了为什么 TP 通常不跨机（NVLink 带宽 ≫ 网络带宽），是后面所有并行度选型的基础。

---

## 二、仓库已有资料清单（重要：不要重复造轮子）

本仓库 `docs/theory/distributed/` 下已有 **2000+ 行高质量中文原理文档**，包含数值示例与手算推演。本计划的定位是**串联这些文档 + 补充源码落点 + 给出动手实验**，而不是重写原理。

| 文档 | 行数 | 内容要点 |
| --- | --- | --- |
| `docs/theory/distributed/TP.md` | 428 | 列/行并行、黄金组合、MLP 手算、head 切分、GQA/MQA、词表并行、通信量分析 |
| `docs/theory/distributed/DP.md` | 479 | 训练 DP vs 推理 DP、DataParallelController、4 种负载均衡策略、GPU/rank 布局数值示例、DPBudget |
| `docs/theory/distributed/DP_attention.md` | 221 | 动机、rank 布局公式、gather→MoE→scatter 数据流、MAX_LEN/SUM_LEN 两种 padding |
| `docs/theory/distributed/EP.md` | 266 | MoE 回顾、按 expert 切分、dispatch/combine、通信量、EPLB 负载均衡 |
| `docs/theory/distributed/PP.md` | 312 | 流水切分、气泡分析、micro-batch |
| `docs/theory/distributed/CP.md` | 208 | 上下文并行 |
| `docs/theory/distributed/collective_communication.md` | 918 | **9 种原语总表**、Ring All-Reduce 通信量与**通信下界证明**、关键恒等式、各并行主导原语 |
| `docs/theory/distributed/cost_model.md` | 473 | **代价模型专题**：$\alpha$-$\beta$、LogP/LogGP、$\alpha$-$\beta$-$\gamma$、带宽层次、对分带宽、Roofline、BSP |
| `docs/advanced_features/expert_parallelism_zh.md` | 199 | EP 部署实践 |
| `docs/advanced_features/pipeline_parallelism_zh.md` | 116 | PP 部署实践 |
| `python/sglang/srt/managers/README_data_parallel_controller_zh.md` | 343 | DP 控制器源码导读 |
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

**目标**：能说清 7 种原语各自的语义与通信量，这是理解所有并行的前置。

- **阅读**：`docs/theory/distributed/collective_communication.md` 全文（重点是 §1 总表、§12 关键恒等式；行有余力再看 §10 的通信下界证明、§8.1 Ring vs Direct）。
- **选读**：`docs/theory/distributed/cost_model.md`——搞清楚 $\alpha$-$\beta$ 模型与**半带宽点 $n_{1/2}=\alpha/\beta$**（§2），这是理解“为什么 decode 阶段小消息要用 one-shot allreduce”的关键。
- **源码落点**：`distributed/parallel_state.py` 的 `GroupCoordinator` 类，看 `all_reduce` / `all_gather` / `reduce_scatter` 三个方法的签名。
- **动手**：写一个 20 行的纯 PyTorch 脚本，用 `torch.distributed` 起 2 进程，分别跑一次 all-reduce 和 all-gather，打印各 rank 的输入输出。**这一步能让抽象概念立刻具象化。**
- **自测题**：
  1. All-Reduce = Reduce-Scatter + All-Gather，这个恒等式为什么成立？对通信量有什么意义？
  2. Ring All-Reduce 的通信量为什么是 $2\cdot\frac{N-1}{N}\cdot D$ 而不是 $N \cdot D$？
  3. All-to-All 与 All-Gather 的区别？为什么 EP 用前者？
- ✅ **通关标准**：能默画出 7 种原语的数据流示意图；能说出 TP/EP/PP 各自主导哪种原语。

---

### 阶段 P1：TP 张量并行（第 1 天下半天 – 第 2 天）★ 最重要

**目标**：理解权重如何切分、all-reduce 插在哪、为什么 TP 不跨机。

- **阅读顺序**：
  1. `docs/theory/distributed/TP.md` §1–§5（原理 + MLP 手算，务必跟着算一遍 §4 的数值示例）。
  2. `srt/layers/linear.py`：`ColumnParallelLinear`、`RowParallelLinear`、`QKVParallelLinear`、`MergedColumnParallelLinear`。**重点看 `forward` 里 `tp_size > 1` 时的通信调用位置**。
  3. `srt/layers/vocab_parallel_embedding.py`：`VocabParallelEmbedding` / `ParallelLMHead` 的词表切分与 masked all-reduce。
  4. `srt/models/llama.py`：`LlamaMLP`、`LlamaAttention` —— 看真实模型如何组合这些并行层。
  5. `docs/theory/distributed/TP.md` §6–§9（head 切分、GQA/MQA、通信量、源码索引）。

- **核心心智模型（必须内化）**：

```text
MLP:   Column(gate/up) ──► 激活 ──► Row(down) ──► all-reduce
        无通信（列切）              行切后需要求和

Attn:  QKV(列切，按 head) ──► attention ──► O(行切) ──► all-reduce
```

「列并行 → 行并行」的黄金组合，**整个 block 只需 1 次 all-reduce**（而非 2 次），这是 Megatron-LM 的核心设计。

- **动手实验**：
  1. 同一模型分别用 `--tp-size 1` 和 `--tp-size 2` 启动，对比 `/get_server_info` 里的 `max_total_num_tokens`（KV 容量应接近翻倍，因为权重显存被摊薄）。
  2. 用 `python -m sglang.bench_one_batch` 对比两者的 decode 延迟，观察 TP 的收益与通信开销的权衡。
  3. 在 `RowParallelLinear.forward` 加一行 `logger.debug`，确认 all-reduce 每层调用一次。

- **自测题**：
  1. 为什么 attention 按 head 切而不是按 hidden 维切？GQA 下 KV head 数少于 TP size 时怎么办？
  2. `ColumnParallelLinear` 的输出为什么不需要通信，而 `RowParallelLinear` 需要？
  3. LayerNorm / RMSNorm 为什么不切分（每卡冗余计算）？
  4. TP size 从 2 增到 8，通信量如何变化？为什么 TP 一般不跨机？

- **常见坑**：`linear.py` 里有多个 `*ParallelLinear` 变体，`MergedColumnParallelLinear` 是把 gate/up 合并成一次 GEMM 的优化，不要误以为是另一种并行方式。

- ✅ **通关标准**：能手画一个 transformer block 的 TP 切分图（标出 4 个 GEMM 的切法 + all-reduce 位置）；能手算 §4 的 MLP 数值示例；能解释 TP 不跨机的带宽原因。

---

### 阶段 P2：DP 副本级数据并行（第 3 天）

**目标**：理解推理 DP 与训练 DP 的本质区别，以及请求如何被分流。

- **阅读顺序**：
  1. `docs/theory/distributed/DP.md` §1–§3（重点是 §3「训练 DP vs 推理 DP」这个关键区分）。
  2. `python/sglang/srt/managers/README_data_parallel_controller_zh.md`（343 行源码导读）。
  3. `srt/managers/data_parallel_controller.py`：`event_loop`、`launch_dp_schedulers`、负载均衡策略。
  4. `docs/theory/distributed/DP.md` §5–§8（4 种负载均衡策略、GPU/rank 布局数值示例、DPBudget）。

- **关键认知**：推理 DP **没有梯度同步**，副本之间完全不通信，唯一的协调点是「请求分发」。所以推理 DP 的难点不在通信，而在**负载均衡**——这与训练 DP 截然不同。

- **与你已写文档的衔接**：`docs/metrics/endpoints_metrics_and_loads_zh.md` 里的 `num_total_tokens` 正是 `total_tokens` 均衡策略的决策依据：

$$
\text{num\_total\_tokens} = \text{num\_used\_tokens} + \sum_{\text{req} \in \text{等待队列}} \text{req.seqlen}
$$

- **动手实验**：
  1. `--dp-size 2` 启动，并发打压，`curl /v1/loads?include=core` 观察两个 rank 的 `num_total_tokens` 是否均衡。
  2. 对比 `--load-balance-method round_robin` 与 `total_tokens`，用**长短混合**的请求负载（这是两者差异最明显的场景）观察倾斜程度。

- **自测题**：
  1. 推理 DP 与训练 DP 最本质的区别是什么？
  2. `round_robin` 在什么负载下会退化？为什么 `total_tokens` 更鲁棒？
  3. `tp_size=4, dp_size=2` 时共需几张卡？各进程的 `gpu_id` 如何计算？（对照 DP.md §6 的数值示例）
  4. 为什么 DPBudget 需要「推测式 +1」？

- ✅ **通关标准**：能说清 DP 控制器的请求分发链路；能对照数值示例算出任意 `tp/dp/pp` 组合下的 GPU 布局；能用 `/v1/loads` 数据判断负载是否倾斜。

---

### 阶段 P3：DP Attention（第 4–5 天）★ 难点

**目标**：理解为什么 MLA 模型需要它，以及 gather/scatter 的数据流。

> 这是**最容易混淆**的一块，务必在阶段 P1（TP）和 P2（DP）都通关后再来。

- **动机（一句话）**：MLA（DeepSeek）的 KV 只有一份 latent，TP 切不动 —— 强行 TP 会让每张卡都存**完整 KV 副本**，KV 显存随 TP size 线性浪费。于是让 attention 改走 DP（每卡处理不同请求的 batch 分片，各自持有自己那份 KV），MoE 仍走 TP/EP。

- **阅读顺序**：
  1. `docs/theory/distributed/DP_attention.md` §1–§2（动机与核心思想）。
  2. `srt/layers/dp_attention.py`：
     - `compute_dp_attention_world_info()` / `initialize_dp_attention()` —— rank 布局
     - `_dp_gather_via_all_reduce()` vs `_dp_gather_via_all_gather()` —— 两种 gather 实现
     - `dp_gather_partial()` / `dp_scatter()`
     - `DpPaddingMode`（`MAX_LEN` / `SUM_LEN`）
  3. `docs/theory/distributed/DP_attention.md` §3–§7（rank 公式、数据流、padding 模式、与普通 TP 的区别）。

- **核心数据流**：

```text
        ┌── rank0: attn(自己的 batch 分片, 自己的 KV) ──┐
输入 ───┤                                              ├── gather ──► MoE(TP/EP) ──► scatter ──► 输出
        └── rank1: attn(自己的 batch 分片, 自己的 KV) ──┘
             各卡 KV 独立，无冗余              全局 token 一起过 MoE
```

- **自测题**：
  1. 为什么 MLA 模型 TP 会导致 KV 冗余，而 GQA 模型不会？
  2. `MAX_LEN` 与 `SUM_LEN` 两种 padding 各自的通信量与显存代价？何时选哪个？
  3. 为什么 `--enable-dp-attention` 要求 dp_size == tp_size？
  4. gather 的两种实现（all-reduce 版 / all-gather 版）分别在什么场景更优？

- **动手实验**：用 DeepSeek 或 Qwen3-MoE，对比开关 `--enable-dp-attention` 时 `/get_server_info` 的 `max_total_num_tokens`（开启后 KV 容量应显著提升）。

- **无卡替代方案**：读 `DP_attention.md` §4.2 的数值直觉，用纸笔推演 2 卡场景下 token 的 gather/scatter 过程。

- ✅ **通关标准**：能画出 DP Attention 一层的数据流（含 gather/scatter 位置）；能解释它与「副本级 DP」「普通 TP」三者的区别；能说清 KV 显存的节省来源。

---

### 阶段 P4：EP 专家并行（第 6–7 天）★ 重点

**目标**：理解 MoE 如何按 expert 切分、all-to-all 通信、以及负载不均问题。

- **阅读顺序**：
  1. `docs/theory/distributed/EP.md` §1–§5（含 §5 手算一次 EP 路由，务必跟着算）。
  2. `srt/layers/moe/ep_moe/`：EP MoE 层实现。
  3. `srt/layers/moe/token_dispatcher/`：all-to-all 分发后端（DeepEP 等），对照 `--moe-a2a-backend`。
  4. `docs/theory/distributed/EP.md` §6（EPLB 负载均衡）→ `srt/eplb/`。
  5. `docs/advanced_features/expert_parallelism_zh.md`（部署实践）。

- **核心数据流**：

```text
router 打分 ──► dispatch (all-to-all) ──► 各卡算自己的 expert ──► combine (all-to-all) ──► 输出
             按 expert 归属重排 token                         把结果送回原 token 所在卡
```

- **EP 的核心难点是负载不均**：token 路由到哪个 expert 由数据决定，热门 expert 所在的卡会成为瓶颈。EPLB 通过**统计专家负载 + 周期性重排/复制专家**来缓解。这是 EP 区别于 TP 的最大特点——**TP 的负载是静态均衡的，EP 是动态倾斜的**。

- **自测题**：
  1. EP 与 TP 切 MoE 分别有什么优劣？为什么大规模 MoE 倾向 EP？
  2. dispatch/combine 为什么必须用 all-to-all 而不是 all-gather？
  3. 什么是专家负载不均？EPLB 用什么策略缓解？
  4. `--ep-size` 与 `--tp-size` 的关系？两者能否不等？

- **动手实验**：用 MoE 模型开 `--ep-size`，观察 `sglang:eplb_balancedness`（需 `SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC`）与 `sglang:eplb_gpu_physical_count` 指标，直观看到专家负载分布。

- ✅ **通关标准**：能手算 EP.md §5 的路由示例；能说清 dispatch→compute→combine 三阶段；能解释 EPLB 要解决的问题与基本思路。

---

### 阶段 P5：PP 流水并行 + CP 上下文并行（第 8 天）— 了解即可

> 这两者在推理侧使用频率低于 TP/DP/EP，**掌握动机与代价即可**，不必深挖实现。

- **PP**：
  - 阅读 `docs/theory/distributed/PP.md`（重点是气泡分析）+ `docs/advanced_features/pipeline_parallelism_zh.md`。
  - 源码：`distributed/parallel_state.py` 的 `get_pp_group()`，以及 scheduler 里的 micro-batch 处理。
  - 关键认知：PP 的代价是**流水气泡**，气泡率 $\approx \frac{P-1}{M+P-1}$（$P$ = stage 数，$M$ = micro-batch 数）。推理场景 batch 小，气泡相对更痛，所以 PP 优先级低于 TP。
- **CP**：阅读 `docs/theory/distributed/CP.md`，理解超长上下文下按序列维切分的动机即可。
  - 参数注意：**没有 `--cp-size`**。实际是 `--attn-cp-size`（等价 `--attention-context-parallel-size`）与 `--enable-prefill-cp`；旧的 `--enable-prefill-context-parallel` / `--enable-dsa-prefill-context-parallel` 已标记弃用。
  - 开启 prefill CP 时 `attn_cp_size` 由 `tp_size // dp_size` 推导（见 `server_args.py`）。该特性**仍处于实验阶段**，且限单机（跨机有精度问题）、主要在 Hopper 上验证过。

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
| MoE（DeepSeek 类） | `--tp-size 8 --enable-dp-attention --ep-size 8` | attention 走 DP 省 KV，MoE 走 EP 分摊权重 |
| 超大模型跨机 | `--tp-size 8 --pp-size 2` | 机内 TP、机间 PP（避免跨机 all-reduce） |

- **选型原则**（按优先级）：
  1. **先满足「放得下」**：权重 + KV + 激活 ≤ 显存。放不下就加 TP（机内优先），再不够上 PP（跨机）。
  2. **TP 不跨机**：跨机 all-reduce 走网络，延迟数量级劣于 NVLink。
  3. **能用 DP 就别加 TP**：TP 每层 2 次 all-reduce 是纯开销；DP 副本间零通信。在「放得下」的前提下，小 TP + 大 DP 通常吞吐更高。
  4. **MoE 优先 EP**：MoE 权重占比大且稀疏激活，EP 比 TP 更省。

- **动手实验（本阶段核心产出）**：固定模型与卡数，跑 3 组配置的 `bench_serving`，记录吞吐/TTFT/ITL，输出一张对比表。例如 8 卡下对比 `tp8` / `tp4+dp2` / `tp2+dp4`。

- **自测题**：
  1. 给定 70B 模型 + 8×A100(80G)，如何切？如果换成 2 机 16 卡呢？
  2. 为什么「能用 DP 就别加 TP」？什么情况下这条原则不成立？
  3. `tp_size=8, dp_size=2, pp_size=2` 需要多少卡？各进程组如何划分？

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
| 多卡 OOM | 对比各 rank 显存 | DP attention 未开导致 KV 冗余 |

- **关键原则**：**集合通信要求所有 rank 参与**。绝大多数分布式 hang 的根因都是「某个 rank 因为数据依赖的分支走了不同路径，没进入同一个通信调用」。定位方法是二分查找第一个状态发散点。

- ✅ **通关标准**：能用 py-spy 抓取多 rank 栈并判断卡在哪个集合通信；能从 profiler trace 里量化通信占比。

---

## 五、进度追踪表

（以下用时均按**每天 4 小时**换算）

| 阶段 | 主题 | 建议用时 | 状态 | 产出物 |
| --- | --- | --- | --- | --- |
| P0 | 集合通信原语 | 0.5 天（2h） | ✅ | 7 种原语手绘图 + 2 进程通信脚本 |
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
- KV 传输为什么要按 TP rank 切分对齐（因为 KV 本身就是按 TP 切开存的）；
- `/v1/loads` 的 `disaggregation` 段那几个队列字段（`prefill_bootstrap_queue_reqs`、`decode_prealloc_queue_reqs` 等）对应 PD 状态机的哪个阶段 —— 这部分你在 `docs/metrics/endpoints_metrics_and_loads_zh.md` 已经写过。

---

## 七、参考资源

- 本仓库：`docs/theory/distributed/`（原理，含数值示例）、`docs/advanced_features/`（部署实践）
- 官方文档：<https://docs.sglang.io/>
- 大规模 EP 博客（强烈推荐）：<https://lmsys.org/blog/>
- Megatron-LM 论文（TP 的理论源头）：*Efficient Large-Scale Language Model Training on GPU Clusters*
- DeepSeek-V3 技术报告（MLA + DP Attention + 大规模 EP 的工业实践）

