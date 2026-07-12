# 数据并行（Data Parallelism, DP）原理详解

> 本文系统介绍数据并行的基本思想、训练场景与推理场景的本质差异、
> SGLang 中「模型副本 + 请求分发」这一推理侧 DP 的实现方式（`DataParallelController`）、
> 四种负载均衡策略、GPU/进程/rank 布局的计算、PD 分离下的 `bootstrap_room` 配对，
> 并给出可手算验证的数值示例，最后对应到 `data_parallel_controller.py`、`server_args.py` 的真实实现。
>
> 前置阅读：`TP.md`（张量并行）。DP 与 TP 是正交的两条切分轴，实际部署常组合使用。
> 若你要找的是 **MLA 模型层内的「DP Attention」**（attention 走 DP、MoE 走 TP/EP），
> 那是另一回事，请看同目录 `DP_attention.md`——本文讲的是**副本级**数据并行。

## 目录

1. [为什么需要数据并行](#1-为什么需要数据并行)
2. [核心思想：复制模型，切分请求](#2-核心思想复制模型切分请求)
3. [训练 DP vs 推理 DP：一个关键区分](#3-训练-dp-vs-推理-dp一个关键区分)
4. [SGLang 的 DP 架构：DataParallelController](#4-sglang-的-dp-架构dataparallelcontroller)
5. [四种负载均衡策略](#5-四种负载均衡策略)
6. [GPU / 进程 / rank 布局与数值示例](#6-gpu--进程--rank-布局与数值示例)
7. [PD 分离下的 bootstrap_room 配对](#7-pd-分离下的-bootstrap_room-配对)
8. [负载预算 DPBudget 与推测式计数](#8-负载预算-dpbudget-与推测式计数)
9. [SGLang 中的实现](#9-sglang-中的实现)
10. [DP 与其他并行的关系](#10-dp-与其他并行的关系)
11. [局限与权衡](#11-局限与权衡)

---

## 1. 为什么需要数据并行

张量并行（`TP.md`）解决的是「**单卡装不下、算不动一层**」——把**同一份权重**切到多张卡。
但即便一个模型副本已经能在 $N$ 张卡上跑起来（比如 TP=8 装下了 DeepSeek），
仍然有一个问题：**单个副本的吞吐是有上限的**。请求一多，队列就排起来，延迟飙升。

数据并行解决的是另一个维度的问题——「**如何用更多硬件线性提升吞吐**」：

- **张量并行（TP）**：切**权重**。把一层摊到多卡，降低单步延迟、突破单卡显存。属**纵向**扩展（scale-up），受限于单机 NVLink。
- **数据并行（DP）**：复制**整个模型**成多个副本，把**不同的请求**分给不同副本并行处理。属**横向**扩展（scale-out），几乎不需要副本间通信，可跨机。

一句话对比：

> **TP 让一个副本更快/更大，DP 让副本更多。** 二者正交，常组合成 `world_size = DP × TP × PP`。

典型场景：一个 TP=8 的副本已能服务 DeepSeek，但要扛住线上 QPS，就再复制 4 份（DP=4），
共 32 卡，理论吞吐 ≈ 单副本的 4 倍——而这 4 份副本之间**推理时几乎零通信**。

---

## 2. 核心思想：复制模型，切分请求

数据并行的全部技巧只有一句话：**权重复制，数据切分**。

```
                      ┌──────────── 请求流 ────────────┐
                      │  req0  req1  req2  req3  req4 …  │
                      └───────────────┬─────────────────┘
                                      │  负载均衡分发
              ┌───────────────┬───────┴───────┬───────────────┐
              ▼               ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ 副本 0    │    │ 副本 1    │    │ 副本 2    │    │ 副本 3    │
        │ 完整模型  │    │ 完整模型  │     │ 完整模型  │     │ 完整模型  │
        │ (TP=8)   │    │ (TP=8)   │    │ (TP=8)   │    │ (TP=8)   │
        └──────────┘    └──────────┘    └──────────┘    └──────────┘
         处理 req0,4      处理 req1        处理 req2        处理 req3
                （副本间推理时互不通信，各算各的）
```

- **每个副本持有一份完整的模型权重**（副本内部可以再用 TP/PP 切，但对外是一个整体）；
- 不同请求被分发到不同副本；
- 副本之间在推理时**没有任何数据依赖**——不像 TP 每层都要 all-reduce。

### 吞吐的数值直觉

设单副本峰值吞吐为 $T$（tokens/s），$d$ 个副本：

$$
T_\text{total} \approx d \cdot T \quad (\text{理想线性})
$$

现实中略低于线性，损耗只来自：分发器（`DataParallelController`）的调度开销、以及副本间负载不均。
**不存在** TP 那种「每层 all-reduce」的通信税——这正是 DP 能轻松跨机扩展的根本原因。

> 对照 `TP.md` §8：TP 每个 block 前向要 2 次 all-reduce，对带宽极敏感，只能困在单机；
> DP 副本间零通信，可以跨机堆到任意规模。

---

## 3. 训练 DP vs 推理 DP：一个关键区分

「数据并行」这个词在**训练**和**推理**里指的是不同的东西，务必分清，否则会对 SGLang 的行为产生误解。


| 维度              | 训练 DP（如 PyTorch DDP）                 | 推理 DP（SGLang）                               |
| ----------------- | ----------------------------------------- | ----------------------------------------------- |
| 每卡/每副本放什么 | 一份完整模型副本                          | 一份完整模型副本（内部可再 TP/PP）              |
| 数据如何切        | 一个 batch 切成多份，各副本各算一份       | **不同请求**分发到不同副本                      |
| 副本间通信        | **反向传播后梯度 all-reduce**（核心开销） | **几乎没有**（仅请求分发；PD 分离另有 KV 传输） |
| 为什么要同步      | 保持所有副本权重一致，否则模型会发散      | 无需同步——权重是只读的，永不更新              |
| 主要收益          | 用更多卡加速训练（更大 global batch）     | 用更多副本提升在线服务吞吐                      |

关键结论：

> **SGLang 的推理 DP 里没有梯度、没有权重同步。** 每个副本是一个**完全独立的推理实例**，唯一需要协调的只是「哪个请求交给哪个副本」。所以 SGLang 的 DP 本质是
> **「多实例 + 请求路由 + 负载均衡」**，而不是训练里那种「梯度 all-reduce」。

这也解释了为什么 SGLang 的 DP 实现核心是一个**请求分发控制器**（`DataParallelController`），
而不是一套集合通信原语——它要解的是调度问题，不是通信问题。

---

## 4. SGLang 的 DP 架构：DataParallelController

SGLang 用 `DataParallelController`（下称 **DPC**，进程名 `sglang::data_parallel_controller`）
承载副本级 DP。它坐在 `TokenizerManager`（上游）和多个 DP worker 的 `Scheduler`（下游）之间：

```
                    ┌─────────────────────┐
   已分词请求         │  TokenizerManager   │
  ────────────────▶ │   (node_rank==0)    │
                    └──────────┬──────────┘
                               │ ZMQ PULL (scheduler_input_ipc_name)
                               ▼
                    ┌────────────────────────┐
                    │  DataParallelController │
                    │   · 负载均衡选 dp_rank    │
                    │   · 进程管理 / 看门狗      │
                    └──────────┬─────────────┘
                               │ ZMQ PUSH (self.workers[dp_rank])
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌───────────┐    ┌───────────┐    ┌───────────┐
       │ DP rank 0 │    │ DP rank 1 │ …  │ DP rank N │   每个 rank 内部是
       │ TP/PP组    │   │ TP/PP组    │   │ TP/PP组    │   一整套 Scheduler 进程
       └───────────┘    └───────────┘    └───────────┘
```

它做两件事：

1. **进程管理**：启动并看护所有 DP worker 的 `Scheduler` 子进程（每个 DP rank 是一整套 TP/PP 进程组）。
2. **请求分发**：把分词后的请求按负载均衡策略路由到某个 DP rank。

### 4.1 两套启动路径

由 `enable_dp_attention` 决定（这是「副本级 DP」与「层内 DP attention」在启动侧的分叉点）：


| 路径             | 函数                                       | 进程组                              | nccl 端口    | 适用                |
| ---------------- | ------------------------------------------ | ----------------------------------- | ------------ | ------------------- |
| **普通 DP**      | `launch_dp_schedulers`（`:354`）           | 每个 DP rank**各自独立**的 TP group | 每 rank 独立 | 本文主题：副本级 DP |
| **DP attention** | `launch_dp_attention_schedulers`（`:575`） | 所有 DP rank**复用同一** TP group   | 共用一个     | 见`DP_attention.md` |

普通 DP 模式下，每个 rank 分到独立的 GPU 区间和 nccl 端口，用独立线程并行启动以缩短总启动时间；
最终真正拉起 scheduler 子进程的是 `launch_tensor_parallel_group`（`:614`）。

> ⚠️ 启动线程 `launch_tensor_parallel_group_thread`（`:413`）完成后**必须常驻挂起**，
> 否则 `scheduler.py` 的 `kill_itself_when_parent_died` 会把它当成「父进程已死」而误杀对应 scheduler。

### 4.2 只有主节点分发

多机部署时，只有 `node_rank == 0` 的主节点建立从 TokenizerManager 拉请求的 PULL 套接字、
运行 `event_loop()` 分发；其余节点只做本地 worker 的进程管理，不接请求。

> 这解释了 PD 分离部署中，为什么 decode 集群 **Master 节点的 `data_parallel_controller` CPU 占用很高**
> （真正的调度中枢），而 Worker 节点的同名进程几乎空闲。

（DPC 的进程模型、构造流程、多节点端口广播等更细的机制，见
`python/sglang/srt/managers/README_data_parallel_controller_zh.md`。）

---

## 5. 四种负载均衡策略

`load_balance_method`（`server_args.py:697`，CLI `--load-balance-method`，`:6157`）
选定分发函数，全部定义在 `data_parallel_controller.py`：


| 方法                    | 分发函数                                    | 策略                                                                   |
| ----------------------- | ------------------------------------------- | ---------------------------------------------------------------------- |
| `round_robin`           | `round_robin_scheduler`（`:764`）           | 轮询，跳过非存活 worker，计数器取模回绕                                |
| `follow_bootstrap_room` | `follow_bootstrap_room_scheduler`（`:782`） | 按`bootstrap_room % len(workers)` 选 rank，**PD 分离配对**用（见 §7） |
| `total_requests`        | `total_requests_scheduler`（`:796`）        | 选当前累计**请求数**最少的 rank                                        |
| `total_tokens`          | `total_tokens_scheduler`（`:803`）          | 选当前累计**token 数**最少的 rank（token 相同时用请求数 tie-break）    |

### 默认值：`auto`

`load_balance_method` 默认是 `"auto"`，由 `_handle_load_balance_method`（`server_args.py:1493`）按部署形态解析：


| 部署形态                  | auto 解析为             |
| ------------------------- | ----------------------- |
| 非 PD 分离                | `round_robin`           |
| PD 分离 —— prefill 集群 | `follow_bootstrap_room` |
| PD 分离 —— decode 集群  | `round_robin`           |

### 数值示例：round_robin vs total_tokens

设 4 个 DP rank，依次到达 5 个请求，token 数分别为 `[1000, 50, 50, 50, 1000]`：

**round_robin**（只看顺序，不看负载）：


| 请求 | tokens | 分到 rank |
| ---- | ------ | --------- |
| r0   | 1000   | 0         |
| r1   | 50     | 1         |
| r2   | 50     | 2         |
| r3   | 50     | 3         |
| r4   | 1000   | 0         |

结果：rank 0 背了 `1000+1000=2000` token，rank 1/2/3 各 50——**严重倾斜**。

**total_tokens**（每次选当前 token 最少的 rank）：


| 请求 | tokens | 各 rank 当前负载`[r0,r1,r2,r3]` | 选中            |
| ---- | ------ | ------------------------------- | --------------- |
| r0   | 1000   | `[0,0,0,0]`                     | 0（并列取首个） |
| r1   | 50     | `[1000,0,0,0]`                  | 1               |
| r2   | 50     | `[1000,50,0,0]`                 | 2               |
| r3   | 50     | `[1000,50,50,0]`                | 3               |
| r4   | 1000   | `[1000,50,50,50]`               | 1（当前最小）   |

结果：`[1000, 1050, 50, 50]`——比 round_robin 均衡得多。这就是负载感知调度的价值
（代价是要维护每个 rank 的负载视图，见 §8）。

### 外部 router 直接路由优先

所有分发函数开头都会先调 `maybe_external_dp_rank_routing`（`:755`）：若请求已带 `routed_dp_rank`
（由 sgl-router / dynamo 等外部 router 注入），DPC 直接透传、跳过负载均衡。
即：**有外部 router 时 DPC 只做进程管理与透传，路由决策交给外部**。

---

## 6. GPU / 进程 / rank 布局与数值示例

普通 DP 模式下，每个 DP 副本占用 `tp_size × pp_size` 张卡的连续区间。
每个 scheduler 进程绑定哪张卡由 `gpu_id` 决定，分两步合成。

### 第一步：跨 DP rank 的起始偏移

```python
# launch_dp_schedulers（data_parallel_controller.py:354）
base_gpu_id = 0
for dp_rank in range(server_args.dp_size):
    ...
    base_gpu_id += server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
```

第 $k$ 个副本的起始偏移 = $k \cdot (\text{tp\_size} \times \text{pp\_size} \times \text{gpu\_id\_step})$。

### 第二步：合成每个进程的 gpu_id

```python
# launch_tensor_parallel_group（:614）
gpu_id = (
    server_args.base_gpu_id                                   # ① CLI --base-gpu-id 全局基准
    + base_gpu_id                                             # ② 本 DP 副本起始偏移
    + (pp_rank % pp_size_per_node) * tp_size_per_node         # ③ 节点内 PP 阶段偏移
    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step  # ④ 节点内 TP rank 偏移
)
```

### 数值示例：单机 `tp_size=4, dp_size=2, pp_size=1, gpu_id_step=1`

单副本占 4 卡，2 副本共 8 卡（正好铺满单机 8 卡）。因 `pp_size=1`，③ 项恒为 0，故
`gpu_id = base_gpu_id + tp_rank`：


| DP rank | base_gpu_id      | tp_rank | gpu_id  |
| ------- | ---------------- | ------- | ------- |
| 0       | 0                | 0,1,2,3 | 0,1,2,3 |
| 1       | `1×(4×1×1)=4` | 0,1,2,3 | 4,5,6,7 |

即副本 0 占 GPU `0~3`，副本 1 占 GPU `4~7`，互不重叠。副本内 4 卡做 TP，副本间做 DP。

### 数值示例：单机 16 卡 `tp_size=4, dp_size=2, pp_size=2, gpu_id_step=1`

单副本占 `tp×pp = 4×2 = 8` 卡，2 副本共 16 卡（铺满单机 16 卡）。单机 `nnodes=1`，节点内切分量：

- `pp_size_per_node = max(pp_size // nnodes, 1) = max(2//1,1) = 2`（本节点承载全部 2 个 PP 阶段）
- `nnodes_per_pp_rank = max(nnodes // pp_size, 1) = 1`
- `tp_size_per_node = tp_size // nnodes_per_pp_rank = 4`

此时 `pp_size=2`，③ 项 `(pp_rank % 2) * 4` **不再恒为 0**，故：

```
gpu_id = base_gpu_id + (pp_rank % 2) * 4 + tp_rank
```

第一步 `base_gpu_id`：DP0 → `0`；DP1 → `1×(4×2×1) = 8`。

| DP rank | base_gpu_id | pp_rank | tp_rank | 计算 | gpu_id |
| ------- | ----------- | ------- | ------- | ---- | ------ |
| 0 | 0 | 0 | 0,1,2,3 | `0 + 0 + tp_rank` | 0,1,2,3 |
| 0 | 0 | 1 | 0,1,2,3 | `0 + 4 + tp_rank` | 4,5,6,7 |
| 1 | 8 | 0 | 0,1,2,3 | `8 + 0 + tp_rank` | 8,9,10,11 |
| 1 | 8 | 1 | 0,1,2,3 | `8 + 4 + tp_rank` | 12,13,14,15 |

解读：

- **副本 0** 占 GPU `0~7`（PP 阶段 0 用 `0~3`、PP 阶段 1 用 `4~7`）；**副本 1** 占 GPU `8~15`（同理 `8~11` / `12~15`）。
- 每个副本内部：2 个 PP 阶段各占 4 卡、阶段内 4 卡做 TP；副本之间做 DP。两副本正好铺满 16 卡、互不重叠。
- 与前一个 8 卡示例的差别就在 ③ 项：`pp_size>1` 时它把同一副本的不同 PP 阶段错开到不同 GPU 区间。

> 多机 + PP 的更复杂布局（PP 跨节点、`pp_size_per_node` 变化时的逐行数值表），
> 见 `README_data_parallel_controller_zh.md` §1.5。

---

## 7. PD 分离下的 bootstrap_room 配对

PD（Prefill-Decode）分离部署中，prefill 和 decode 是**两个独立集群**，各有自己的 DPC。
一个请求先在某个 prefill rank 算出 KV Cache，再点对点传给某个 decode rank 继续解码。
问题来了：**如何保证第 r 号 prefill 副本产出的 KV，恰好被配对的 decode 副本消费？**

轮询做不到——两端各自独立轮询，编号对不上。解法是**确定性哈希**：

```
                 router 生成随机 bootstrap_room（每请求唯一，63-bit）
                              │
              ┌───────────────┴───────────────┐
              ▼（同一 room 同时下发两端）          ▼
   Prefill DPC                        Decode DPC
   room % len(workers) = r            room % len(workers) = r   ← 取模结果必然一致
              │                                │
              ▼                                ▼
   Prefill DP rank r  ──KV Cache 点对点传输──▶  Decode DP rank r
```

`bootstrap_room` 是 router 为每个请求生成的一次性配对 ID，prefill/decode 两端 DPC
用**同一个 room 做同样的取模** `bootstrap_room % len(workers)`，从而命中相互配对的 rank，
保证 KV Cache 传输不错位。这正是 `follow_bootstrap_room` 策略（§5）在 prefill 侧作为默认值的原因。

### 数值示例

设两端各 4 个 DP rank，某请求 `bootstrap_room = 8675309`：

$$
8675309 \bmod 4 = 1
$$

于是 prefill DP rank 1 算 KV，decode DP rank 1 接收——两端一致。
另一请求 `bootstrap_room = 8675310` → `% 4 = 2`，两端都命中 rank 2。

> 断言保护：若 `req.bootstrap_room` 为 `None`，说明请求被**直接发到 prefill/decode 实例而非经 router**
> （只有 router 才注入该字段），DPC 会直接报错提示「请发给 router」。
> 详见 `README_data_parallel_controller_zh.md` §2.3。

---

## 8. 负载预算 DPBudget 与推测式计数

`total_requests` / `total_tokens` 两种策略要知道每个 rank 的**当前负载**，
这由 `DPBudget`（`data_parallel_controller.py:109`）维护：

- **数据来源**：各 Scheduler 把自己的负载（`num_running_reqs` / `num_waiting_reqs` / `num_total_tokens`）
  写入共享内存的**负载快照**，DPC 读取刷新。
- **推测式 +1**：每分发一个请求给某 rank，就对该 rank 的负载估计**立即 +1**，不等下次快照。
- **20ms 节流**：两次快照刷新之间至少间隔 20ms。

### 为什么需要「推测式 +1 + 节流」

突发流量下，快照更新有延迟。若每次分发都用（同一份尚未更新的）快照覆盖负载估计，
就会把本波已累加的推测计数全抹掉——结果**整波请求全压到同一个「看起来最闲」的 rank**。

节流 + 推测式计数解决这个问题：一波请求先靠「每分发一个 +1」在本地打散，
20ms 后再用真实快照校准。这是一个「乐观估计 + 定期对账」的经典模式。

---

## 9. SGLang 中的实现

### 9.1 控制器：`python/sglang/srt/managers/data_parallel_controller.py`


| 组件                                                  | 位置            | 职责                                                   |
| ----------------------------------------------------- | --------------- | ------------------------------------------------------ |
| `LoadBalanceMethod`                                   | `:83`           | 负载均衡方法枚举                                       |
| `DPBudget`                                            | `:109`          | 负载预算表（total_requests/total_tokens 用，§8）      |
| `DataParallelController`                              | `:167`          | 控制器主体                                             |
| `launch_dp_schedulers`                                | `:354`          | 普通 DP：逐副本启动独立 TP group                       |
| `launch_dp_attention_schedulers`                      | `:575`          | DP attention：复用同一 TP group（见`DP_attention.md`） |
| `launch_tensor_parallel_group`                        | `:614`          | 真正拉起 scheduler 子进程、算 gpu_id（§6）            |
| `maybe_external_dp_rank_routing`                      | `:755`          | 外部 router 直接路由透传（§5）                        |
| `round_robin_scheduler`                               | `:764`          | 轮询分发                                               |
| `follow_bootstrap_room_scheduler`                     | `:782`          | PD 配对分发（§7）                                     |
| `total_requests_scheduler` / `total_tokens_scheduler` | `:796` / `:803` | 负载感知分发                                           |
| `event_loop`                                          | `:813`          | 主循环：非阻塞拉请求 → 交分发器                       |
| `run_data_parallel_controller_process`                | `:826`          | 进程入口                                               |

### 9.2 主事件循环

```python
# event_loop（:813）
while True:
    while True:
        self.soft_watchdog.feed()                                    # 喂看门狗表示存活
        try:
            recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)  # 非阻塞拉取
        except zmq.ZMQError:
            break                                                    # 队列暂空，跳出继续轮询
        self._request_dispatcher(recv_req)                           # 交类型分发器处理
```

### 9.3 启动参数：`python/sglang/srt/server_args.py`

```bash
# 4 个模型副本，每副本内部 TP=8，共 32 卡
python -m sglang.launch_server --model <model> --tp-size 8 --dp-size 4 \
    --load-balance-method total_tokens
```


| 参数                                              | 位置                | 含义                                                |
| ------------------------------------------------- | ------------------- | --------------------------------------------------- |
| `dp_size`（`--data-parallel-size` / `--dp-size`） | `:695`，CLI `:6151` | DP 副本数$d$                                        |
| `load_balance_method`（`--load-balance-method`）  | `:697`，CLI `:6157` | 负载均衡策略，默认`auto`（§5）                     |
| `moe_dp_size`（`--moe-data-parallel-size`）       | `:702`，CLI `:5649` | MoE 层的 DP 度（与副本级 DP 不同，属 MoE 内部并行） |

> 注意区分：`--dp-size` 在**未开** `--enable-dp-attention` 时是**副本级 DP**（本文）；
> **开启** `--enable-dp-attention` 时同一个 `--dp-size` 变成层内 `attn_dp_size`（见 `DP_attention.md` §3）。

---

## 10. DP 与其他并行的关系

实际部署常把多种并行组合成 **N 维并行网格**（对照 `TP.md` §10）：


| 并行                 | 切什么                 | 副本/进程间通信             | 适用范围                     |
| -------------------- | ---------------------- | --------------------------- | ---------------------------- |
| **DP（数据并行）**   | 复制模型、切分**请求** | 推理时几乎无（仅分发）      | 任意，**横向**提吞吐、可跨机 |
| **TP（张量并行）**   | 层内矩阵的张量维       | 每层 all-reduce（频繁、小） | 单机内、高带宽 NVLink        |
| **PP（流水线并行）** | 不同 layer 分到不同卡  | 层间激活传递（稀疏）        | 跨机，配合 micro-batch       |
| **EP（专家并行）**   | MoE 的 expert          | all-to-all                  | MoE 模型                     |

典型组合：`world_size = DP × TP × PP`。例如单副本 TP=8（吃满单机 NVLink），
复制 4 份 DP=4 提吞吐，共 32 卡。

**经验法则：先用 TP 吃满单机、让副本能跑起来，再用 DP 横向复制副本提吞吐。**

### 三个易混概念的辨析

SGLang 里带「DP」字样的东西有三个，层级完全不同，务必分清：


| 概念                        | 切什么                                     | 进程组                              | 谁管理                                     | 文档              |
| --------------------------- | ------------------------------------------ | ----------------------------------- | ------------------------------------------ | ----------------- |
| **副本级 DP**（本文）       | 整个模型副本，请求分发到不同副本           | **多个独立** TP group，每副本一套   | `DataParallelController`                   | 本文              |
| **DP Attention**            | attention 按 token 走 DP，MoE 仍全局 TP/EP | **复用同一** TP group，层内切换维度 | `initialize_dp_attention` + gather/scatter | `DP_attention.md` |
| **MoE DP**（`moe_dp_size`） | MoE 层内部的 DP 度                         | TP group 内部                       | MoE 层实现                                 | —                |

> 例：LongCat decode 集群 `dp_size=16, attn_tp_size=8`——若这里的 16 是**副本级 DP**，
> 就是 16 套独立实例由 DPC 管理，每套内部 8 卡做 attention TP；这与层内 `attn_dp_size` 是不同层级。
> （具体是哪种取决于是否 `--enable-dp-attention`，见 §9.3 的提示。）

---

## 11. 局限与权衡

1. **显存不摊薄**：DP 是**复制**模型，每个副本都要装下完整权重和自己的 KV Cache——
   $d$ 个副本就要 $d$ 份权重显存。它只提吞吐，**不解决单卡装不下**（那是 TP/PP 的活）。
2. **负载均衡是关键**：请求长短、到达时刻参差，简单轮询容易倾斜（见 §5 数值示例）。
   负载感知策略（`total_tokens`）更均衡，但要维护负载视图、承受快照延迟（见 §8）。
3. **分发器是单点**：只有 `node_rank==0` 跑 `event_loop` 分发，它是调度中枢，
   CPU 压力集中于此（见 §4.2）。
4. **PD 分离的配对约束**：跨集群时必须用 `follow_bootstrap_room` 做确定性配对（§7），
   不能随意换成轮询，否则 KV Cache 传输会错位。
5. **外部 router 会架空负载均衡**：一旦外部 router 注入 `routed_dp_rank`，DPC 的策略被绕过，
   路由质量取决于外部 router——但 DPC 的进程管理职责仍在。

---

## 参考与延伸

- 同目录：`TP.md`（张量并行，DP 的正交搭档）、`DP_attention.md`（MLA 层内 DP attention，与本文的副本级 DP 是不同层级）。
- 详细控制器机制：`python/sglang/srt/managers/README_data_parallel_controller_zh.md`
  （进程模型、构造流程、gpu_id 逐行数值表、多机端口广播、存活状态管理）。
- SGLang 代码：`python/sglang/srt/managers/data_parallel_controller.py`、`python/sglang/srt/server_args.py`。
