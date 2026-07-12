# 流水线并行（Pipeline Parallelism, PP）原理详解

> 本文系统介绍流水线并行的核心思想（按 layer 切模型）、micro-batch 如何填满流水线消除气泡、
> 各 stage 之间点对点（P2P）传递激活的通信模式、PP 与 TP/DP 的本质区别，
> 并给出可手算的调度时序示例，最后对应到 SGLang 中 `get_pp_indices`、`make_layers`、
> 模型 `forward` 的 `PPProxyTensors`、以及 `scheduler_pp_mixin.py` 里 `event_loop_pp` 的真实实现。

## 目录

1. [为什么需要流水线并行](#1-为什么需要流水线并行)
2. [核心思想：把不同的 layer 放到不同卡](#2-核心思想把不同的-layer-放到不同卡)
3. [朴素 PP 的致命问题：流水线气泡](#3-朴素-pp-的致命问题流水线气泡)
4. [micro-batch：用流水线填满气泡](#4-micro-batch用流水线填满气泡)
5. [推理场景的 PP：与训练的不同](#5-推理场景的-pp与训练的不同)
6. [stage 之间传什么：激活的 P2P 传递](#6-stage-之间传什么激活的-p2p-传递)
7. [层如何切分到各 stage](#7-层如何切分到各-stage)
8. [通信量与显存分析](#8-通信量与显存分析)
9. [SGLang 中的实现](#9-sglang-中的实现)
10. [PP 与 TP / DP 的关系](#10-pp-与-tp--dp-的关系)
11. [局限与权衡](#11-局限与权衡)

---

## 1. 为什么需要流水线并行

一个 LLM 可能有几十到上百个 Transformer layer，总权重远超单卡显存。解决「装不下、算不动」有几条互补的路线：

- **数据并行（DP）**：每张卡放一份完整模型，喂不同的数据。解决吞吐，**不解决单卡装不下**。
- **张量并行（TP）**：把**同一层内部**的大矩阵乘沿张量维切成几块分给几张卡。按**层内维度**切，每层都要集合通信，对带宽极敏感，**通常限制在单机内**。
- **流水线并行（PP）**：把**不同的 layer** 放到不同卡上，像工厂流水线一样串起来。按**层**切。

PP 的独特价值：

1. 让原本单卡放不下的**深层模型**能跑起来——每张卡只放 $1/p$ 的 layer（$p$ 为 PP stage 数）；
2. stage 之间**只在层的边界传一次激活**（点对点通信），通信量极小、且不随模型宽度暴涨；
3. 因此 **PP 能跨机扩展**：跨机带宽（IB）远低于机内 NVLink，但 PP 的稀疏 P2P 通信能容忍这种带宽。

代价是引入**流水线气泡**（bubble）——朴素做法下前后 stage 会互相空等（见 [§3](#3-朴素-pp-的致命问题流水线气泡)），
必须用 **micro-batch** 才能把流水线填满（见 [§4](#4-micro-batch用流水线填满气泡)）。

---

## 2. 核心思想：把不同的 layer 放到不同卡

设模型有 $L$ 层 Transformer block，PP 路数为 $p$，则把连续的 $L/p$ 层打包成一个 **stage**，
每个 stage 放在一个 **PP rank** 上。一条请求要**依次**流经 stage 0 → stage 1 → … → stage $p-1$ 才完成一次完整 forward：

```
  输入 tokens
      │
      ▼
┌───────────┐   激活    ┌───────────┐   激活    ┌───────────┐
│  stage 0  │ ────────► │  stage 1  │ ────────► │  stage 2  │ ──► logits
│ layer 0~k │  (P2P)    │layer k~2k │  (P2P)    │layer 2k~L │
│ + embed   │           │           │           │ + norm/head│
└───────────┘           └───────────┘           └───────────┘
   rank 0                  rank 1                  rank 2
```

- **首 stage（rank 0）**额外持有输入 embedding（`embed_tokens`）；
- **末 stage（rank p-1）**额外持有最后的 `norm` 与 LM Head，负责产出 logits / 采样出 token；
- **中间每个 stage** 只有一段连续的 decoder layer。

与 TP 的根本区别：**TP 切「一层的内部」，每层都通信；PP 切「层与层之间」，只在 stage 边界通信。**

---

## 3. 朴素 PP 的致命问题：流水线气泡

如果一次只送**一个** batch 进流水线，那么任一时刻只有一个 stage 在干活，其余 stage 全在空等：

```
时间 ─────────────────────────────────►
stage 0 │ F0 │    │    │              │
stage 1 │    │ F0 │    │              │
stage 2 │    │    │ F0 │              │
              ▲ 同一时刻只有 1 个 stage 在算，其余 2 个空转
```

$p$ 个 stage 的流水线，利用率只有约 $1/p$——PP 越深浪费越大。这段空转就是**流水线气泡（bubble）**。
朴素 PP 相当于把一次 forward 的延迟拉长了（多了 stage 间传递），却没有换来任何吞吐提升。

---

## 4. micro-batch：用流水线填满气泡

解决办法是把一个大 batch 拆成多个 **micro-batch（mb）**，像流水线上的多个工件一样**错峰**送入：
当 stage 0 算完 mb0、把它交给 stage 1 时，stage 0 立刻开始算 mb1，而不是干等。

```
时间 ───────────────────────────────────────────►
stage 0 │ F0 │ F1 │ F2 │ F3 │              │
stage 1 │    │ F0 │ F1 │ F2 │ F3 │         │
stage 2 │    │    │ F0 │ F1 │ F2 │ F3 │
          └──┬──┘              └──┬──┘
          填充期(bubble)       排空期(bubble)   中间：3 个 stage 全忙
```

- **稳态（中段）**：所有 $p$ 个 stage 同时在算不同的 micro-batch，利用率接近 100%；
- **填充期 / 排空期**：流水线头尾各有一段 bubble，长度约 $p-1$ 个 micro-batch 的时间。

若有 $m$ 个 micro-batch、$p$ 个 stage，气泡占比约为：

$$
\text{bubble ratio} \approx \frac{p-1}{m + p - 1}
$$

**结论：micro-batch 数 $m$ 越远大于 stage 数 $p$，气泡占比越小、流水线越满。** 这就是 PP 高效运行的前提。

---

## 5. 推理场景的 PP：与训练的不同

上面的经典图景来自**训练**（还有反向传播，故有 1F1B、interleaved 等复杂调度）。**LLM 推理只有前向**，没有反向，PP 的形态更简单，但也有推理特有的两个关键点：

### 5.1 prefill 与 decode 都要过完整流水线

无论是 prefill（处理 prompt）还是 decode（逐 token 生成），一次 forward 都要从 stage 0 走到 stage $p-1$。
decode 每步只产 1 个 token，单个 batch 的计算量很小，若不做 micro-batch 交错，气泡会非常严重——
所以 SGLang 在 PP 循环里始终按 micro-batch 槽位轮转（见 §9）。

### 5.2 只有末 stage 知道下一个 token

采样发生在末 stage（有 LM Head）。但**下一步 decode 的输入又要从 stage 0 开始**。
因此末 stage 采样出的 `next_token_ids` 必须**沿 PP 环绕回 rank 0**，rank 0 才能把它作为下一轮的输入喂进流水线。
SGLang 用「末位 stage → rank0」的额外一跳 P2P 完成这个回传（`_pp_send_output_to_next_stage` 中 `is_last_rank` 分支）。

---

## 6. stage 之间传什么：激活的 P2P 传递

stage 边界上传递的不是权重，而是**激活张量**——上一 stage 最后一层的输出，作为下一 stage 第一层的输入。
在 SGLang 里这被封装成 **`PPProxyTensors`**，对多数模型（如 LLaMA）它包含两个张量：

| 键             | 形状                        | 含义                                   |
| -------------- | --------------------------- | -------------------------------------- |
| `hidden_states`| $s \times d_\text{model}$   | 该 stage 算完后的隐藏态                |
| `residual`     | $s \times d_\text{model}$   | 尚未与 hidden 融合的残差流（延迟相加） |

之所以连 `residual` 一起传，是因为 SGLang 把 RMSNorm 与残差相加做了融合/延迟处理，跨 stage 时需要把「未合并的残差」也带过去（见 `llama.py` forward 中 `residual = pp_proxy_tensors["residual"]`）。

传递方式是**点对点通信（P2P send/recv）**，而非 TP 那样的集合通信（all-reduce）：

- 每个 stage 用**异步 send + 同步 recv**：发送不阻塞以减小开销，接收同步以避免收发错位；
- 为避免某些后端 `isend` 阻塞导致的**环形死锁**，SGLang 按 **PP rank 奇偶**决定收发顺序（偶数 rank 先发后收，奇数 rank 先收后发），保证每对相邻 stage 总有一发一收同时就绪。

除了激活，PP 环上还单向传递**请求元数据**（`recv_reqs` 沿环转发），保证各 stage 看到一致的请求顺序。

---

## 7. 层如何切分到各 stage

SGLang 用 `get_pp_indices` 计算「本 rank 负责哪一段连续 layer」，再用 `make_layers` 只实例化这一段、其余位置用占位层填充。

### 7.1 层区间划分：`get_pp_indices`

默认**尽量均分**；若 $L$ 不能被 $p$ 整除，则把多出来的 $L \bmod p$ 层分给**最后几个** stage：

```python
# distributed/utils.py: get_pp_indices
base_layers = num_hidden_layers // pp_size
remainder   = num_hidden_layers % pp_size
if pp_rank >= pp_size - remainder:   # 最后 remainder 个 stage 各多 1 层
    ...
    end_layer = start_layer + (base_layers + 1)
else:                                # 其余 stage 各 base_layers 层
    start_layer = pp_rank * base_layers
    end_layer   = start_layer + base_layers
```

也可用环境变量 `SGLANG_PP_LAYER_PARTITION="a,b,c,..."` **手动指定**每个 stage 的层数（长度须等于 $p$、之和须等于 $L$），
用于按各卡算力/显存不均衡时做非均匀切分。

### 7.2 占位层：`make_layers` + `PPMissingLayer`

每个 rank 上模型的 `layers` 列表长度仍是完整的 $L$，但**只有 `[start_layer, end_layer)` 是真正的 decoder layer**，
其余位置放 `PPMissingLayer`（一个 `torch.nn.Identity` 的变体，前向直接透传）。这样：

- 各 stage 的层编号 `layer_id` 在全局保持一致（KV cache、权重加载都靠它对齐）；
- 前向时只遍历本 stage 负责的区间：`for i in range(self.start_layer, self.end_layer)`。

### 7.3 首 / 末 stage 的特殊组件

```python
# models/llama.py: LlamaModel.__init__
if self.pp_group.is_first_rank:
    self.embed_tokens = VocabParallelEmbedding(...)   # 只有首 stage 有 embedding
...
if self.pp_group.is_last_rank:
    self.norm = RMSNorm(...)                          # 只有末 stage 有最终 norm + LM Head
```

forward 的分支也据此区分（见 `llama.py` 的 `LlamaModel.forward`）：

```python
if self.pp_group.is_first_rank:
    hidden_states = self.embed_tokens(input_ids)      # 首 stage：从 token 算起
    residual = None
else:
    hidden_states = pp_proxy_tensors["hidden_states"] # 中间/末 stage：从上一 stage 的激活算起
    residual      = pp_proxy_tensors["residual"]
...
if not self.pp_group.is_last_rank:
    return PPProxyTensors({"hidden_states": ..., "residual": ...})  # 非末 stage：把激活传下去
else:
    hidden_states, _ = self.norm(hidden_states, residual)          # 末 stage：出最终 hidden → logits
```

---

## 8. 通信量与显存分析

### 8.1 通信量

- **每条请求每次 forward，只在 $p-1$ 个 stage 边界各做 1 次 P2P 传递**，每次传输 `hidden_states` + `residual`，
  大小约 $2 \cdot s \cdot d_\text{model}$ 个元素（$s$ = 该 micro-batch 的 token 数）。
- 与 TP 相比：TP 每层 2 次 all-reduce（频繁、集合通信）；PP 整个模型只有 $p-1$ 次 P2P（稀疏、点对点）。
- 因此 PP 的通信量**只与 stage 数和激活大小有关，与模型深度/宽度基本无关**，且 P2P 对跨机带宽的容忍度远高于 all-reduce——**这是 PP 能跨机、TP 通常不能的根本原因**。

### 8.2 显存

| 部分                   | 是否随 PP 切分            | 单卡占用                       |
| ---------------------- | ------------------------- | ------------------------------ |
| 各 decoder layer 权重  | 是（按层切）              | $\approx 1/p$                  |
| embedding 权重         | 否（只在首 stage）        | 首 stage 独占，其余为 0        |
| norm + LM Head 权重    | 否（只在末 stage）        | 末 stage 独占，其余为 0        |
| KV cache               | 是（每 stage 只存自己层） | $\approx 1/p$                  |
| micro-batch 激活缓冲   | 随 micro-batch 数增加     | $\propto m$（填流水线的代价）  |

主体权重和 KV cache 都约降到 $1/p$，这正是 PP「让深层模型装得下」的来源；代价是要为 in-flight 的多个 micro-batch 保留激活缓冲。

---

## 9. SGLang 中的实现

### 9.1 层切分与占位：`distributed/utils.py`、`utils/common.py`

- `get_pp_indices`（`distributed/utils.py:101`）：计算本 rank 的 `[start_layer, end_layer)`，支持 `SGLANG_PP_LAYER_PARTITION` 手动分区。
- `make_layers`（`utils/common.py:695`）：按区间实例化真实 layer，区间外填 `PPMissingLayer`（`layers/utils/common.py:109`）。

### 9.2 模型 forward：`models/llama.py`

- `LlamaModel.__init__`（`llama.py:350` 附近）：`is_first_rank` 才建 `embed_tokens`，`is_last_rank` 才建 `norm`。
- `LlamaModel.forward`（`llama.py:386`）：非首 stage 从 `pp_proxy_tensors` 取输入，非末 stage 返回 `PPProxyTensors`（`hidden_states` + `residual`）。

### 9.3 PP 调度主循环：`managers/scheduler_pp_mixin.py`

`SchedulerPPMixin` 实现三套 PP 事件循环：

- `event_loop_pp`（`:88`）：普通 PP 调度。核心是按 `pp_loop_size = pp_size + pp_async_batch_depth` 个 micro-batch 槽位轮转，
  每个槽位：`recv 上一 stage 的 proxy` → `launch 本批前向` → `send proxy 给下一 stage` → 回收并后处理上一轮 micro-batch 的结果。
- `event_loop_pp_disagg_prefill`（`:222`）/ `event_loop_pp_disagg_decode`（`:417`）：PD 分离场景，额外在 PP rank 间对 bootstrap/release/retract/prealloc 等事件**沿环求共识**（取交/并集），保证最终一致性。

关键机制：

- **异步 send + 同步 recv**、按 **rank 奇偶**排序收发以避免环形死锁（`_pp_send_recv_and_preprocess_output_tensors`，`:1284`，`send_first = (not is_xpu()) or (pp_rank % 2 == 0)`）。
- **末 stage 输出绕回 rank0**：`_pp_send_output_to_next_stage`（`:1246`）的 `is_last_rank` 分支把 `next_token_ids` 发回首 stage。
- **消息按类型解复用**：proxy 与 output 在同一通道交错到达，用 `__msg_type__` 打标签，`_pp_recv_typed_dict`（`:1120`）按类型分桶暂存。
- **计算/通信重叠**：`pp_async_batch_depth > 0` 时在 launch 前提前收发上一轮输出（`_pp_launch_batch` 在独立 `forward_stream` 上发起前向）。
- **动态分块**：`ChunkSizePredictor`（`:1550`）用二次模型 $f(l)=al^2+bl+c$ 拟合 prefill 延迟，动态预测下一 chunk 大小以均衡各步耗时。

### 9.4 进程组：`distributed/parallel_state.py`

- `initialize_model_parallel`（`:1848`）按 `pipeline_model_parallel_size` 建立 PP 进程组；同一 PP 组内的 rank 相邻编号。
- `PPGroup.is_first_rank`（`:491`）/ `is_last_rank`（`:496`）判断首/末 stage；`send_tensor_dict`（`:1324`）/ `recv_tensor_dict`（`:1379`）做 P2P 张量传递。

### 9.5 启动参数：`server_args.py`

```bash
python -m sglang.launch_server --model <model> --pp-size 4   # 等价 --pipeline-parallel-size 4
```

- `pp_size`（`server_args.py:536`，CLI `:5656`）：PP stage 数 $p$。
- `pp_max_micro_batch_size`（`:538`，CLI `:5663`）：单个 micro-batch 的最大大小。
- `pp_async_batch_depth`（`:540`，CLI `:5669`）：异步批深度（额外 in-flight micro-batch 数），0 表示同步；`pp_loop_size = pp_size + pp_async_batch_depth`。
- 注意：`pp_size > 1` 时会强制 `disable_overlap_schedule`（`_handle_pipeline_parallelism`，`:4279`），且 PP 目前与 context parallelism、elastic EP 互斥（`:3905`、`:4256`）。

---

## 10. PP 与 TP / DP 的关系

| 并行                 | 切什么                | 通信粒度                       | 适用范围               |
| -------------------- | --------------------- | ------------------------------ | ---------------------- |
| **TP（张量并行）**   | 层内矩阵的张量维      | 每层 all-reduce（频繁、小）    | 单机内、高带宽 NVLink  |
| **PP（流水线并行）** | 不同 layer 分到不同卡 | 层间 P2P 传激活（稀疏）        | 跨机，配合 micro-batch |
| **DP（数据并行）**   | batch 数据            | 无（推理）/ 梯度 all-reduce（训练） | 任意，提吞吐       |
| **EP（专家并行）**   | MoE 的 expert         | all-to-all                     | MoE 模型               |

典型组合：`world_size = TP × PP × DP`。例如 8 机 × 8 卡共 64 卡跑超大模型，
可设 TP=8（机内切层内维度、吃满 NVLink）、PP=8（跨机切层、用稀疏 P2P 容忍 IB 带宽），DP 再叠在外层提吞吐。

**经验法则：TP 优先吃满单机 NVLink；单机装不下时用 PP 跨机扩展；DP 提升整体吞吐。**
PP 与 TP 正交互补——TP 让「单层」装得下、算得快，PP 让「整个深模型」跨卡跨机装得下。

---

## 11. 局限与权衡

1. **流水线气泡**：填充/排空期不可避免地空转，气泡占比约 $\frac{p-1}{m+p-1}$。$p$ 越大、$m$ 越小越浪费——必须保证足够多的 micro-batch。
2. **负载均衡**：各 stage 计算量须尽量相等，否则最慢的 stage 成为瓶颈（拖尾/straggler）；层不能整除时靠 `get_pp_indices` 或 `SGLANG_PP_LAYER_PARTITION` 调整。首/末 stage 还额外背负 embedding / LM Head，需要纳入均衡考量。
3. **延迟增加**：一次 forward 要串行经过 $p$ 个 stage，单请求延迟高于单卡——PP 换的是**吞吐与容量**，不是单请求延迟。
4. **实现复杂度**：micro-batch 调度、P2P 收发排序（防死锁）、末 stage 输出回传、PD 分离下的跨 rank 共识，都显著增加了调度器复杂度（见 `scheduler_pp_mixin.py`）。
5. **与其他特性的耦合**：`pp_size > 1` 需关闭 overlap schedule，且当前与 context parallelism、elastic EP 互斥。

---

## 参考与延伸

- 经典论文：GPipe（Huang et al., 2018，micro-batch 填流水线）、PipeDream / Megatron-LM 1F1B（训练侧调度）。
- 同目录其他理论文档：`./TP.md`（张量并行）、`./DP.md` / `./DP_attention.md`（数据并行）、`../Glossary.md`（术语表）。
- SGLang 代码：`python/sglang/srt/managers/scheduler_pp_mixin.py`、`distributed/utils.py`（`get_pp_indices`）、`utils/common.py`（`make_layers`）、`distributed/parallel_state.py`（PP 进程组与 P2P）、各模型 `forward` 中的 `PPProxyTensors` 分支。

