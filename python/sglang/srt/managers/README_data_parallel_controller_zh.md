# DataParallelController 详解

> 源码：`python/sglang/srt/managers/data_parallel_controller.py`
>
> 进程名：`sglang::data_parallel_controller`

`DataParallelController`（下文简称 **DPC**）是 SGLang 中**数据并行（Data Parallel, DP）的入口控制器**。它位于 `TokenizerManager`（上游）与多个 DP worker 的 `Scheduler`（下游）之间，主要做两件事：

1. **进程管理**：启动并管理所有 DP worker 的 `Scheduler` 子进程，维护它们的存活状态与生命周期。
2. **请求分发**：把分词后的请求按某种负载均衡策略分发到不同的 DP rank。

```
                    ┌─────────────────────┐
   已分词请求         │  TokenizerManager   │
  ────────────────▶ │   (node_rank==0)    │
                    └──────────┬──────────┘
                               │ ZMQ PULL (scheduler_input_ipc_name)
                               ▼
                    ┌─────────────────────┐
                    │ DataParallelController │
                    │  · 负载均衡选 rank      │
                    │  · 进程管理/看门狗       │
                    └──────────┬──────────┘
                               │ ZMQ PUSH (self.workers[dp_rank])
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌───────────┐    ┌───────────┐    ┌───────────┐
       │ DP rank 0 │    │ DP rank 1 │ …  │ DP rank N │   每个 rank 内部是
       │ TP/PP group│   │ TP/PP group│   │ TP/PP group│   一组 Scheduler 进程
       └───────────┘    └───────────┘    └───────────┘
```

---

## 一、进程模型与启动流程

### 1.1 进程入口

进程入口是 `run_data_parallel_controller_process()`（源码末尾）。它负责：

- 设置进程名 `sglang::data_parallel_controller`；
- `faulthandler.enable()`：崩溃时打印各线程 Python 调用栈；
- `kill_itself_when_parent_died()`：父进程死亡时自杀，避免孤儿进程；
- 配置日志、可选的链路追踪（按 PD 分离角色打标签 `Prefill DP Controller` / `Decode DP Controller`）；
- 创建 `DataParallelController`（**构造过程中会启动所有 scheduler 子进程并等待其就绪**）；
- 收集所有 scheduler 子进程 PID，连同容量上限（`max_total_num_tokens`、`max_req_input_len`）一起经管道回传给父进程，表示整体就绪；
- **仅 `node_rank==0` 的主节点**进入 `event_loop()` 分发请求；其余节点不接请求；
- 任一 scheduler 子进程退出即视为异常，记录日志；捕获异常时向父进程发 `SIGQUIT` 触发整体退出。

### 1.2 构造函数 `__init__` 做了什么

| 步骤 | 说明 |
| --- | --- |
| 解析负载均衡方法 | 把 `server_args.load_balance_method` 字符串解析为 `LoadBalanceMethod` 枚举 |
| 初始化 ZMQ | `zmq.Context(1 + dp_size)`；仅 `node_rank==0` 建立从 TokenizerManager 拉取请求的 `PULL` 套接字 |
| 选定分发函数 | 根据均衡方法从 `dispatch_lookup` 选出 `self.dispatching`；标记是否需在分发前刷新负载预算 |
| 初始化负载视图 | 创建 `DPBudget` 与负载快照读取器 `load_snapshot_reader` |
| 启动 worker | `enable_dp_attention` 走 `launch_dp_attention_schedulers`，否则走 `launch_dp_schedulers` |
| 初始化分发器 | `init_dispatcher()` 建立基于消息类型的路由表 |
| 软看门狗 | 创建 `soft_watchdog`（soft 模式只告警不杀进程） |
| CPU 监控 | 启用 metrics 时启动 CPU 监控线程 |

### 1.3 两套启动路径

DPC 有两条 worker 启动路径，由 `enable_dp_attention` 决定：

#### 普通 DP 模式 — `launch_dp_schedulers()`

- **每个 DP rank 各自独立的 TP group**，逐个启动；
- 为每个 rank 分配独立的 nccl 端口与 GPU 区间（`base_gpu_id` 按 `tp_size * pp_size * gpu_id_step` 递增）；
- 每个 rank 用一个独立线程调用 `launch_tensor_parallel_group_thread` 并行启动，缩短总启动时间；
- 主节点为每个 dp_rank 建立 `PUSH` 套接字（`self.workers[dp_rank]`）；
- 最后阻塞等待所有 worker 的 `ready_event`。

> ⚠️ 注意：`launch_tensor_parallel_group_thread` 启动完成后线程**必须常驻不能退出**（用超长 `sleep` 挂起），否则 `scheduler.py` 里的 `kill_itself_when_parent_died` 会把对应 scheduler 当作"父进程已死"而误杀。

#### DP attention 模式 — `launch_dp_attention_schedulers()`

- **所有 DP rank 复用同一个 TP group**，只调用一次 `launch_tensor_parallel_group`（`dp_rank` 传 `None`，由内部按 attention 分片逻辑推导各 rank）；
- 端口由 `node_rank==0` 预分配，并通过 `_broadcast_worker_ports` 广播给所有节点（避免多机端口冲突）；
- 所有 dp rank 共用同一个 nccl 端口。

### 1.4 `launch_tensor_parallel_group()` — 真正拉起 scheduler

按节点拓扑计算本节点应承载的 `pp_rank` / `tp_rank` 区间，为每个 `(pp_rank, tp_rank)` 组合：

1. 计算实际 GPU id；
2. 从 `tp_rank` 反推各并行维度的 rank（层级从外到内）：
   - **Attention**：Global(TP) → DP → ATTN_CP → ATTN_TP
   - **MoE**：Global(TP) → MOE_DP → EP → MOE_TP
3. 用 `env_lock` 保护 `CUDA_VISIBLE_DEVICES` 的设置，以子进程方式启动 scheduler（`run_scheduler_process_func`）；
4. 通过 `mp.Pipe` 阻塞等待每个 scheduler 回传"模型加载完成"信息；
5. 取首个进程上报的 `max_total_num_tokens`、`max_req_input_len` 作为全局上限。

### 1.5 多机端口广播（DP attention）

`_broadcast_worker_ports` / `_broadcast_ports_as_server` / `_receive_ports_as_client`：

- **node 0 作服务端**：用 `REP` 套接字按"收握手 → 回端口"应答其余 `nnodes-1` 个客户端节点；
- **其余节点作客户端**：用 `REQ` 套接字携带自身 `node_rank` 握手，接收端口列表（收发超时 10 分钟，超时抛 `RuntimeError`）；
- **弹性 EP 场景**（`elastic_ep_backend` 非 None）：广播完成后另起后台线程 `_reply_ports_as_server` 持续应答，供后续恢复的 EP rank 获取端口。

---

## 二、负载均衡策略

### 2.1 `LoadBalanceMethod` 枚举

| 方法 | 分发函数 | 说明 |
| --- | --- | --- |
| `ROUND_ROBIN` | `round_robin_scheduler` | 简单轮询，跳过非存活 worker，计数器取模回绕 |
| `FOLLOW_BOOTSTRAP_ROOM` | `follow_bootstrap_room_scheduler` | 按 `bootstrap_room % len(workers)` 选 rank，**PD 分离场景保证同一请求的 prefill 与 decode 命中相互配对的 rank**（详见 §2.3） |
| `TOTAL_REQUESTS` | `total_requests_scheduler` | 选当前累计请求数最少的 rank |
| `TOTAL_TOKENS` | `total_tokens_scheduler` | 选当前累计 token 数最少的 rank（token 数相同时用请求数做 tie-break） |

### 2.2 外部直接路由优先

所有分发函数开头都会调用 `maybe_external_dp_rank_routing(req)`：

- 若请求已显式指定 `req.routed_dp_rank`，直接路由到该 worker 并返回 `True`（跳过负载均衡）；
- 否则返回 `False`，交给对应策略选择目标 rank。

> 这是外部 router（如 sgl-router / dynamo）与 DPC 协作的关键点：外部 router 决定 rank 时，DPC 只做透传。

### 2.3 `bootstrap_room` 路由（PD 分离配对）

**一句话**：`bootstrap_room` 是 router 为**每个请求**（批请求/多采样时细到每个样本）生成的一次性 63-bit 随机配对 ID。prefill 集群和 decode 集群的 DPC 都用**同一个 `bootstrap_room` 做同样的取模** `bootstrap_room % len(workers)`，从而让同一请求的 prefill 输出与 decode 计算落到**相互配对的 DP rank** 上，保证 KV Cache 点对点传输不错位。

**流程**：

```
                 router 生成随机 bootstrap_room（每请求唯一）
                              │
              ┌───────────────┴───────────────┐
              ▼（同一 room 同时下发）             ▼
   Prefill DPC                        Decode DPC
   room % len(workers) = r            room % len(workers) = r   ← 取模结果一致
              │                                │
              ▼                                ▼
   Prefill DP rank r  ──KV Cache 点对点传输──▶  Decode DP rank r
```

**为什么不能用轮询**：`round_robin` / `total_tokens` 等是"就地负载均衡"，两端各自独立选 rank，无法保证 prefill 的第 r 号副本产出的 KV 恰好被 decode 的第 r 号副本消费。PD 分离要求 KV 从固定 prefill rank 精准传到固定 decode rank，**必须用确定性映射**——`bootstrap_room` 取模就是这个确定性哈希。

**特性**：请求级、一次性，不跨请求、不跨会话复用；请求结束后 room 即失效（disaggregation 侧 `_cleanup_room_tracking` 清理）。

**断言保护**：

```python
assert req.bootstrap_room is not None, (
    "req.bootstrap_room should not be None. Do not send requests directly to "
    "prefill or decode instances; send to the router instead."
)
```

`bootstrap_room` 为空说明请求被**直接发到 prefill/decode 实例而非经由 router**（只有 router 才会注入该字段），属误用，故直接报错提示"请发给 router"。

---

## 三、负载预算 `DPBudget`

为 `TOTAL_REQUESTS` / `TOTAL_TOKENS` 两种策略维护每个 DP rank 的负载估计：

- 数据来源：各 Scheduler 写入共享内存的**负载快照**（`num_running_reqs`、`num_waiting_reqs`、`num_total_tokens`）；
- `update_budget()`：用快照刷新预算表，**按时间戳跳过过期读取**；
- `dispatch()`：选出目标 rank，并对其做**推测式 +1 累加**。

### 3.1 为什么需要"推测式 +1"与 20ms 节流

`refresh_load_budget()` 做了 **20ms 节流**，注释解释了原因：

> 突发请求时若每次分发都刷新预算，会用（同一份尚未更新的）快照覆盖掉本波已累加的推测式 +1，导致整波请求全压到同一个 DP rank。20ms 节流让一波请求先靠推测式计数打散，下一批再用真实负载刷新。

因此两次快照刷新之间靠"每分发一个请求给目标 rank 计数 +1"来打散突发流量。

---

## 四、请求分发与事件循环

### 4.1 基于类型的分发器 `init_dispatcher()`

`TypeBasedDispatcher` 把不同消息路由到对应处理函数：

| 消息类型 | 处理函数 |
| --- | --- |
| `TokenizedGenerateReqInput` / `TokenizedEmbeddingReqInput` | `dispatching_with_trace`（带打点的单条分发） |
| `BatchTokenizedGenerateReqInput` | `dispatch_batch_generate`（批量分发） |
| `BatchTokenizedEmbeddingReqInput` | `dispatch_batch_embedding` |
| `BlockReqInput` / `ProfileReq` | `send_to_all_workers`（广播给所有存活 worker） |
| `ActiveRanksOutput` | `update_active_ranks`（更新存活状态） |
| 其他（兜底） | `send_control_message`（按 leader 广播控制消息） |

### 4.2 控制消息广播

- `send_to_all_workers(obj)`：广播给所有**存活**（`status[i]==True`）的 worker，用于 Block/Profile 等全局控制；
- `send_control_message(obj)`：按 `control_message_step` 步长发给各 TP group 的 leader，由 leader 在组内广播。
  - DP attention 且开启本地控制广播：步长 `1`（每个 DP 组 leader 都收到）；
  - 否则步长 `tp_size`（只发首个 leader，由它在整个 tp_group 广播）。

### 4.3 主事件循环 `event_loop()`

```python
while True:
    while True:
        self.soft_watchdog.feed()          # 喂看门狗表示存活
        try:
            recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)  # 非阻塞拉取
        except zmq.ZMQError:
            break                          # 队列暂空，跳出内层继续轮询
        self._request_dispatcher(recv_req) # 交类型分发器处理
```

持续非阻塞地从 TokenizerManager 拉取请求并交分发器处理，每轮喂软看门狗。

---

## 五、存活状态管理

- `self.status: List[bool]`：标记每个 DP rank 是否存活/可用；
- `update_active_ranks(ranks)`：由调度侧通过 `ActiveRanksOutput` 通知某 worker 故障/恢复后更新；
- 分发时 `round_robin_scheduler` 会跳过 `status==False` 的 worker；`send_to_all_workers` 也只发存活 worker。

---

## 六、多节点行为差异

| 行为 | `node_rank == 0`（主节点） | 其他节点 |
| --- | --- | --- |
| 从 TokenizerManager 拉请求 | ✅ 建立 PULL 套接字 | ❌ 不接请求 |
| 运行 `event_loop()` 分发 | ✅ | ❌ 不运行 |
| 建立 worker PUSH 套接字 | ✅ | ❌ |
| DP attention 端口广播 | 服务端（广播） | 客户端（接收） |
| 启动本节点 scheduler 进程 | ✅ | ✅ |

> 这解释了在 PD 分离部署中，为什么 decode 集群里 **Master 节点的 `data_parallel_controller` CPU 占用很高（真正的调度中枢）**，而 Worker 节点的同名进程只做本地 worker 的进程管理。

---

## 七、与 PD 分离 / 外部 router 的关系

- **原生 PD 调度**：`load_balance_method=follow_bootstrap_room` 时，DPC 依据 `bootstrap_room` 保证同一会话的 prefill/decode 命中同一 DP rank，是 PD 协同的核心。
- **外部 router 主导**：若外部 router（sgl-router / dynamo）在请求上设置了 `routed_dp_rank`，`maybe_external_dp_rank_routing` 直接透传，DPC 的负载均衡逻辑被架空——但其**进程管理职责仍然存在**（只要实例内 `dp_size > 1`）。
- **每副本独立实例（`dp_size=1`）**：不进入 DP 分发路径，DPC 相关逻辑不生效。

---

## 八、关键字段速查

| 字段 | 含义 |
| --- | --- |
| `self.workers: List[zmq.Socket]` | 发往各 DP worker 的 PUSH 套接字，下标即 `dp_rank` |
| `self.status: List[bool]` | 各 worker 存活状态 |
| `self.scheduler_procs` | 所有已启动的 scheduler 子进程 |
| `self.dispatching` | 当前选定的分发函数 |
| `self.dp_budget` | 负载预算表（TOTAL_REQUESTS / TOTAL_TOKENS 用） |
| `self.control_message_step` | 控制消息广播步长 |
| `self.max_total_num_tokens` | KV 缓存总 token 容量上限（首个 scheduler 上报） |
| `self.round_robin_counter` | 轮询计数器 |
| `SCHEDULER_PIDS_ARG` | 回传 scheduler PID 列表所用的键名 |

