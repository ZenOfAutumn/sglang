# srt/managers

## 目录用途

`managers` 是 SGLang 推理引擎的核心运行时管理与调度层，承担请求从入口（`entrypoints`）到模型执行（`model_executor`）之间的中枢角色。它将系统拆分为多个可独立运行的进程——前端的 `TokenizerManager`（分词与请求分发）、独立进程的 `Scheduler`（连续批处理调度、KV 缓存与显存池管理、驱动前向计算）、以及 `DetokenizerManager`（反分词、增量文本拼接）——进程间全部通过 ZMQ（PUSH/PULL）消息队列解耦通信。该目录还定义了跨进程传输的全部数据结构（`io_struct.py`）、批次的逐层数据流转（`ScheduleBatch` → `ModelWorkerBatch` → `ForwardBatch`）、调度策略、张量并行 worker 封装，并通过大量 Mixin 把投机解码、PD 分离、流水线并行、DP attention、分层缓存（HiCache）、权重热更新、多模态、会话管理等高级特性组合进来。

## 文件清单

| 文件 | 说明 |
| --- | --- |
| `async_dynamic_batch_tokenizer.py` | `AsyncDynamicbatchTokenizer`，对单条字符串 prompt 做异步动态批处理分词，将零散的 encode 请求聚合成批以提升分词吞吐。 |
| `cache_controller.py` | `HiCacheController` 及配套的层加载事件/计数器、缓存操作、传输缓冲、预取/备份操作，负责分层缓存（HiCache）在 device/host/storage 之间的异步搬运与预取调度。 |
| `configure_logging.py` | 命令行小工具，通过 HTTP 向运行中的 server 下发日志配置（`python -m sglang.srt.managers.configure_logging --url ...`）。 |
| `data_parallel_controller.py` | `DataParallelController`，数据并行（DP）入口控制器，按负载/轮询策略把请求分发到多个 DP worker（Scheduler 进程），并负责拉起这些调度进程。 |
| `detokenizer_manager.py` | `DetokenizerManager`，独立进程，从 Scheduler 接收 token id 批量输出，做反分词与增量可打印文本拼接，再把结果转发回 TokenizerManager；维护有上限的请求状态缓存。 |
| `disagg_service.py` | `start_disagg_service`，为 PD 分离部署启动 bootstrap / KV-store 相关的辅助服务。 |
| `hisparse_coordinator.py` | `HiSparseCoordinator`，稀疏注意力（HiSparse/NSA）协调器，在 host 与 device KV 池间预加载、回写、按 top-k 选择换入 token，并管理 staging 队列与 device 缓冲扩容。 |
| `io_struct.py` | 定义 TokenizerManager / Scheduler / DetokenizerManager 三进程之间传输的全部对象（请求/输出 dataclass），如 `GenerateReqInput`、`Tokenized*ReqInput`、`Batch*Output`、各类权重更新/控制请求等，是跨进程协议的核心。 |
| `mm_utils.py` | 多模态张量传输与 padding 工具：特征缓冲区管理、`TransportProxyTensor`（跨进程张量代理）、多种多模态数据 padding 模式（如成对特殊 token 包裹）。 |
| `multi_tokenizer_mixin.py` | 多 HTTP worker / 多 tokenizer 场景支持：`SocketMapping`、`MultiTokenizerRouter`（接收各 TokenizerWorker 请求并路由）、`TokenizerWorker`、Detokenizer 端的多 worker Mixin。 |
| `multimodal_processor.py` | 多模态处理器注册与查找：`import_processors` 扫描注册各模型的多模态 processor，`get_mm_processor` 按模型架构返回合适的 `BaseMultimodalProcessor`。 |
| `overlap_utils.py` | overlap（计算与 CPU 处理重叠）调度辅助：`FutureMap`/`FutureIndices` 用环形缓冲管理「未来 token id」占位，支持解码与 prefill chunk 的结果延迟解析。 |
| `prefill_delayer.py` | `PrefillDelayer` 及单趟执行器，依据 token 水位线、DP/TP 协商等条件决定是否延迟 prefill 批次，以平衡跨 DP rank 的负载并避免过度激进的预填充。 |
| `schedule_batch.py` | 定义请求与批次的核心数据结构（`Req`、`ScheduleBatch`、`ModelWorkerBatch` 等）及数据流 `ScheduleBatch → ModelWorkerBatch → ForwardBatch`，承载调度层（CPU）到模型执行层（GPU）的批次信息。 |
| `schedule_policy.py` | 请求调度策略：prefill 排队/优先级策略、基于前缀缓存命中（RadixCache）的排序、`max_new_tokens` 估算裁剪等，决定等待队列如何组成下一个批次。 |
| `scheduler.py` | `Scheduler`，运行在独立进程的核心调度器，管理 TP GPU worker；维护等待队列与运行批次实现连续批处理，组织 prefill/decode 调度、KV 缓存与显存池，驱动 `run_batch` 并处理输出；通过事件循环为主入口，并继承多个 Mixin 组合高级特性。 |
| `scheduler_dp_attn_mixin.py` | `SchedulerDPAttnMixin` 与 `MLPSyncBatchInfo`，为 DP attention 提供跨 DP rank 的 MLP 同步批次准备（如空闲批补齐、token 数对齐、cuda graph 可用性协商）。 |
| `scheduler_input_blocker.py` | `SchedulerInputBlocker`，可在全局屏障下临时阻塞/暂存 Scheduler 接收的请求（用于权重更新等需要全局静止的操作），随后统一放行。 |
| `scheduler_output_processor_mixin.py` | `SchedulerOutputProcessorMixin`，从 `scheduler.py` 拆出的输出处理逻辑，负责把前向结果整理为输出、计算缓存命中明细、组织流式/非流式回包。 |
| `scheduler_pp_mixin.py` | `SchedulerPPMixin`，流水线并行（PP）调度循环 `event_loop_pp` 及 `PPBatchMetadata`、`ChunkSizePredictor`，实现各 PP stage 的有序收发与计算/通信重叠。 |
| `scheduler_profiler_mixin.py` | `SchedulerProfilerMixin`，Scheduler 的性能分析能力，封装 torch profiler / ProfileManager 的初始化与按请求启停采集。 |
| `scheduler_recv_skipper.py` | `SchedulerRecvSkipper`，按 forward mode 加权计数控制 Scheduler 接收请求的频率（`scheduler_recv_interval`），减少高频 decode 下的接收开销。 |
| `scheduler_runtime_checker_mixin.py` | `SchedulerRuntimeCheckerMixin`，运行时一致性/资源核查，如统计会话占用的 token、显存与缓存状态的健全性检查。 |
| `scheduler_update_weights_mixin.py` | `SchedulerUpdateWeightsMixin`，Scheduler 侧权重热更新实现（从磁盘/分布式/IPC/tensor 更新、初始化/销毁更新进程组等），并在需要时刷新缓存。 |
| `session_controller.py` | 会话管理：`Session`、`SessionReqNode`、`SessionController`，维护多轮会话的请求树、容量/超时与缓存关联，支持有状态的会话式生成。 |
| `template_manager.py` | `TemplateManager`，集中管理 chat 模板与 completion 模板，负责检测/加载模板及其内容格式（string / openai）。 |
| `tokenizer_communicator_mixin.py` | `TokenizerCommunicatorMixin` 与 `_Communicator`，为 TokenizerManager 提供与 Scheduler 的控制面通信原语（权重更新组、远程实例权重收发等的请求-应答封装）。 |
| `tokenizer_manager.py` | `TokenizerManager`，运行在前端进程，是请求入口与结果出口：tokenize 预处理后经 ZMQ 发给 Scheduler，在异步事件循环 `handle_loop` 中按 rid 收集输出并以（流式/非流式）异步生成器返回；管理 `rid_to_state`、权重更新读写锁等。 |
| `tokenizer_manager_annotated_zh.py` | `tokenizer_manager.py` 的带详细中文注释学习副本，逻辑与原文件一致，仅供阅读理解，不应在生产中引用。 |
| `tokenizer_manager_multiitem_mixin.py` | `TokenizerManagerMultiItemMixin` 与 `ScoreResult`，为 TokenizerManager 提供多 item 打分（score）能力，对每个完整 prompt 后指定 token id 的概率打分。 |
| `tp_worker.py` | `TpModelWorker`（张量并行 worker），把调度器下发的 `ModelWorkerBatch` 转为 GPU 前向，对接 `ModelRunner`/KV 池/显存分配器，并执行各类权重与 LoRA 更新操作。 |
| `utils.py` | 调度层通用工具：`GenerationBatchResult`（前向结果容器，含 overlap 下 CPU 拷贝）、输入长度校验与截断、从结果中提取 logprob 等辅助函数。 |

> 提示：上表中的 `ModelWorkerBatch` 数据流描述为历史写法。当前源码中该中间结构已移除，实际数据流为 `ScheduleBatch → ForwardBatch`（在 `tp_worker.py:479` 由 `ForwardBatch.init_new(batch, model_runner)` 直接转换）。下方学习计划以最新代码为准。

---

## 模块深入学习计划

> 定位：本计划是仓库总学习计划 `docs/sglang_learning_plan_zh.md` 中**阶段 1（架构心智）**与**阶段 2（调度器）**的细化展开，专攻 `managers` 这一中枢层。建议在已能本地跑起一个 server（总计划阶段 0）之后开始。
>
> 节奏：约 **2 周**，每天 1.5–2 小时；分 6 个子阶段（A–F，F 选学）。每个子阶段统一按**阅读 → 动手打点 → 自检 → 产出物**四步推进。所有行号对应当前 `main` 代码，随版本演进可能漂移，以实际 `grep` 为准。

### 学习主线（一句话串起本目录）

```
HTTP 请求
  → TokenizerManager.generate_request（分词，前端进程，异步）
  → [ZMQ] io_struct.TokenizedGenerateReqInput
  → Scheduler.event_loop_*（独立进程，连续批处理）
       ├─ process_input_requests → handle_generate_request（入等待队列，建 Req）
       ├─ get_next_batch_to_run → get_new_batch_prefill / update_running_batch（组批）
       ├─ run_batch → TpModelWorker.forward_batch_generation
       │                 → ForwardBatch.init_new → ModelRunner.forward（GPU 前向）
       └─ process_batch_result（采样后处理、出队、流式回包）
  → [ZMQ] io_struct.BatchTokenIDOutput
  → DetokenizerManager.event_loop（反分词、增量拼接）
  → [ZMQ] io_struct.BatchStrOutput
  → TokenizerManager.handle_loop（按 rid 收集，异步生成器返回）
```

---

### 子阶段 A：三进程骨架与跨进程协议（第 1–2 天）

**目标**：在脑中固化「前端 / 调度 / 反分词」三进程的边界，以及它们之间传什么。

| 阅读 | 关键类/函数（行号） |
| --- | --- |
| `io_struct.py` | `GenerateReqInput`(`:138`)、`TokenizedGenerateReqInput`(`:752`)、`BatchTokenIDOutput`(`:1124`)、`BatchStrOutput`(`:1196`)、`BatchEmbeddingOutput`(`:1262`) |
| `scheduler.py` | `run_scheduler_process`(`:4069`) —— 调度进程入口，看它如何建 `Scheduler` 并进事件循环 |
| `detokenizer_manager.py` | `run_detokenizer_process`(`:478`)、`DetokenizerManager`(`:89`) |
| `tokenizer_manager.py` | `TokenizerManager`(`:237`) 类头与 `__init__`，看它持有哪些 ZMQ socket |

- **动手打点**：在 `TokenizedGenerateReqInput` 构造处、`Scheduler.process_input_requests`(`:1590`) 入口、`DetokenizerManager.event_loop`(`:159`) 入口各打一行日志，打印同一个 `rid`，串起一条请求穿越三进程的轨迹。
- **自检**：① 为什么要拆三进程而不是单进程多线程？（结合 GIL 与 CPU/GPU 重叠）② `GenerateReqInput` 和 `TokenizedGenerateReqInput` 的分界点在哪个进程、由谁完成转换？
- **产出物**：一张三进程 + ZMQ 队列方向的框图，标注每条边上流动的 `io_struct` 类型。

---

### 子阶段 B：TokenizerManager —— 请求入口（第 3–4 天）

**目标**：看清一条请求如何被分词、下发，以及结果如何按 `rid` 异步回收。

| 阅读 | 关键函数（行号） |
| --- | --- |
| `tokenizer_manager.py` | `generate_request`(`:575`)、`_tokenize_one_request`(`:770`)、`_handle_batch_request`(`:1523`)、`_wait_one_response`(`:1413`)、`handle_loop`(`:1812`) |
| `async_dynamic_batch_tokenizer.py` | `AsyncDynamicbatchTokenizer`（动态批分词如何聚合零散 encode） |
| `tokenizer_control_mixin.py` / `tokenizer_manager_score_mixin.py` | 控制面通信、score 能力（先了解入口即可） |

- **动手打点**：在 `generate_request`(`:575`) 与 `handle_loop`(`:1812`) 打点，并发送 1 个流式 + 1 个非流式请求，观察 `rid_to_state` 的建立与清理时机；对比流式（多次 yield）与非流式（一次返回）的回包路径差异。
- **自检**：① `generate_request` 是 `async` 生成器，它如何在等待 Scheduler 结果时不阻塞其他请求？② `handle_loop` 收到 `BatchStrOutput` 后，如何把一批输出分发回各自的 `rid` 协程？
- **产出物**：一段「单请求生命周期」时序笔记，覆盖 tokenize → 下发 → 等待 → 流式/非流式回收。

---

### 子阶段 C：Scheduler 事件循环与连续批处理（第 5–7 天）★ 本目录核心

**目标**：理解 SGLang 性能基石——零开销调度与 continuous batching。

| 阅读 | 关键函数（行号） |
| --- | --- |
| `scheduler.py` | `Scheduler`(`:291`)、`__init__`(`:301`)；先读 `event_loop_normal`(`:1471`)，再读 `event_loop_overlap`(`:1498`)，逐行对比 CPU/GPU 如何重叠 |
| `scheduler.py` | `process_input_requests`(`:1590`)、`handle_generate_request`(`:1961`)（请求入等待队列、建 `Req`） |
| `scheduler.py` | `get_next_batch_to_run`(`:2467`)、`get_new_batch_prefill`(`:2612`)、`update_running_batch`(`:2905`)（组批决策的三个核心） |
| `overlap_utils.py` | `FutureMap` / `FutureIndices`（overlap 模式下「未来 token id」占位的环形缓冲） |
| `scheduler_recv_skipper.py` | `SchedulerRecvSkipper`（高频 decode 下按 forward mode 加权降低收包频率） |

- **动手实验**：用 1 个长 prompt + 多个短请求并发，在 `get_new_batch_prefill`(`:2612`) 与 `update_running_batch`(`:2905`) 打点，打印每轮 batch 的 prefill/decode 请求数；调 `--chunked-prefill-size` 与 `--max-running-requests` 观察 batch 组成与吞吐变化。
- **自检**：① 用自己的话解释「为什么 overlap 模式能消除调度开销」，`FutureMap` 在其中扮演什么角色？② prefill 请求与 decode 请求是如何在「同一批」或「不同批」中被调度的？`get_next_batch_to_run` 的优先级是怎样的？
- **产出物**：`event_loop_normal` 与 `event_loop_overlap` 的对比时序图，标出两者在「采样结果可用时刻」的差异。

---

### 子阶段 D：批次数据流与组批预算（第 8–9 天）

**目标**：吃透 `Req` / `ScheduleBatch` 的状态机，以及 `PrefillAdder` 如何在显存/token 预算内组批。

| 阅读 | 关键类/函数（行号） |
| --- | --- |
| `schedule_batch.py` | `Req`(`:644`)、`init_next_round_input`(`:1096`)、`finished`(`:1068`)；`ScheduleBatch`(`:1634`)、`prepare_for_extend`(`:1967`)、`prepare_for_decode`(`:2544`)、`filter_batch`(`:2639`)、`merge_batch`(`:2715`) |
| `schedule_policy.py` | `SchedulePolicy`(`:149`)、`calc_priority`(`:170`)、`PrefillAdder`(`:425`)、`add_one_req`(`:858`)、`preempt_to_schedule`(`:1025`)；前缀缓存排序 `_sort_by_longest_prefix`(`:296`) |

- **动手打点**：在 `prepare_for_extend`(`:1967`) 与 `prepare_for_decode`(`:2544`) 打印 batch 的 `seqlen`/token 数；发送共享前缀的请求，在 `_compute_prefix_matches`(`:247`) 观察 RadixCache 命中如何改变排队顺序。
- **自检**：① `Req` 从「等待队列」到「运行批次」再到「完成出队」经历哪些方法？② `PrefillAdder` 的 token 预算（`rem_total_tokens` / `cur_rem_tokens`）如何决定一个请求能否加入本轮 prefill？preempt（抢占）在什么条件下触发？
- **产出物**：`Req` 生命周期状态机图 + 一份「组批预算」要点笔记。
- **衔接**：内存池 / RadixCache 的实现细节属于 `mem_cache` 目录，对应总计划阶段 3，此处只需理解调度侧如何「申请/释放」即可。

---

### 子阶段 E：TpModelWorker 前向 与 Detokenizer 出口（第 10–11 天）

**目标**：打通「调度结果 → GPU 前向 → 采样后处理 → 反分词回包」的下半程。

| 阅读 | 关键函数（行号） |
| --- | --- |
| `scheduler.py` | `run_batch`(`:3055`)、`process_batch_result`(`:3277`)（前向触发与后处理） |
| `tp_worker.py` | `TpModelWorker`(`:218`)、`__init__`(`:221`)、`forward_batch_generation`(`:466`)（注意 `:479` 的 `ForwardBatch.init_new` —— 当前真实的批次转换点） |
| `utils.py` | `GenerationBatchResult`（前向结果容器；overlap 下的 CPU 拷贝语义） |
| `detokenizer_manager.py` | `event_loop`(`:159`)、`trim_matched_stop`(`:169`)（增量反分词与停止串裁剪） |

- **动手打点**：在 `run_batch`(`:3055`) 前后与 `process_batch_result`(`:3277`) 打点统计单轮前向耗时；在 `trim_matched_stop`(`:169`) 观察 stop string 命中时如何裁剪输出。
- **自检**：① `forward_batch_generation` 接收 `ScheduleBatch`，内部为何还要转成 `ForwardBatch`？两者职责边界是什么？② overlap 模式下 `GenerationBatchResult` 为什么需要把结果从 GPU 拷回 CPU 才能交给后处理？
- **产出物**：补全子阶段 C 的时序图下半段（前向 → 采样 → detokenize → 回包），形成端到端闭环。

---

### 子阶段 F（选学）：高级特性 Mixin（第 12–14 天）

按兴趣/工作需要挑 1–2 个深入，其余了解入口即可。这些能力多以独立文件或 Mixin 形式组合进 Scheduler / TokenizerManager：

| 特性 | 入口文件 |
| --- | --- |
| 流水线并行（PP）调度循环 | `scheduler_pp_mixin.py`（`event_loop_pp`、`PPBatchMetadata`） |
| 数据并行（DP）请求分发 | `data_parallel_controller.py` |
| 权重热更新需要的全局静止 | `scheduler_input_blocker.py` |
| 分层缓存（HiCache）异步搬运 | `cache_controller.py`、`hisparse_coordinator.py` |
| 多模态处理 | `multimodal_processor.py`、`mm_utils.py` |
| 多 HTTP worker / 多 tokenizer | `multi_tokenizer_mixin.py` |
| PD 分离辅助服务 | `disagg_service.py` |
| 会话式多轮生成 | `session_controller.py` |
| chat/completion 模板 | `template_manager.py`、`template_detection.py` |

- **自检**：能说出所选特性「挂载到 Scheduler/TokenizerManager 的哪个扩展点、解决什么问题」。
- **产出物**：所选特性的一页式「入口 + 数据流 + 触发条件」速记。

---

### 阶段自检清单（学完本目录应能回答）

1. 一条请求从 HTTP 到 token 输出，依次经过哪三个进程、跨进程传了哪些 `io_struct` 对象？
2. `event_loop_normal` 与 `event_loop_overlap` 的本质区别是什么，后者靠什么消除调度开销？
3. `Req` 的完整生命周期与状态流转？`PrefillAdder` 用什么预算约束组批？
4. 当前真实的批次数据流是什么（注意不再有 `ModelWorkerBatch`）？转换发生在哪一行？
5. 你正在用的高级特性（如有）通过哪个 Mixin/文件挂载进调度链路？

### 学习方法提示（针对本目录）

- **按数据流读，别按文件读**：始终顺着上方「学习主线」那条链，遇到 Mixin 再按需跳转。
- **善用 `server_args.py`**：本目录大量分支由启动参数控制（如 `--chunked-prefill-size`、`--enable-overlap-schedule`、`--max-running-requests`），从参数反查处理逻辑能快速定位调度分支。
- **打点优于猜测**：调度路径状态多、跳转密，在 `event_loop_*` / `get_new_batch_prefill` / `process_batch_result` 三处打点，比纯读代码高效得多。
- **带注释副本**：`tokenizer_manager_annotated_zh.py` 是 `tokenizer_manager.py` 的逐行中文注释学习副本，读 B 阶段时可对照。
