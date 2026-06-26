# srt/managers

## 目录用途

`managers` 是 SGLang 推理引擎的核心运行时管理与调度层，承担请求从入口（`entrypoints`）到模型执行（`model_executor`）之间的中枢角色。它将系统拆分为多个可独立运行的进程——前端的 `TokenizerManager`（分词与请求分发）、独立进程的 `Scheduler`（连续批处理调度、KV 缓存与显存池管理、驱动前向计算）、以及 `DetokenizerManager`（反分词、增量文本拼接）——进程间全部通过 ZMQ（PUSH/PULL）消息队列解耦通信。该目录还定义了跨进程传输的全部数据结构（`io_struct.py`）、批次的逐层数据流转（`ScheduleBatch` → `ForwardBatch`）、调度策略、张量并行 worker 封装，并通过组件化（`scheduler_components/`）与 Mixin 把投机解码、PD 分离、流水线并行、DP attention、分层缓存（HiCache）、权重热更新、多模态、会话管理等高级特性组合进来。

## 核心能力总览

> 本节是对本目录「该层到底要做成哪几件事」的纲领性审计，按能力归类（而非按文件），并标注每项的**重要性**（对正确性/性能的影响）与**复杂性**（实现与维护难度）。重要性/复杂性均分为 高 / 中 / 低 三档。具体实现文件见后续「文件清单」。

| # | 核心能力 | 主要承载 | 重要性 | 复杂性 | 说明 |
| --- | --- | --- | --- | --- | --- |
| 1 | **三进程解耦与跨进程协议** | `tokenizer_manager.py`、`detokenizer_manager.py`、`io_struct.py`、`scheduler_components/ipc_channels.py` | 高 | 中 | 前端/调度/反分词三进程经 ZMQ 解耦，靠 GIL 隔离让 CPU 预处理与 GPU 计算互不阻塞；`io_struct` 是跨进程协议的唯一契约，改动需三端同步。 |
| 2 | **连续批处理调度（事件循环）** | `scheduler.py`（`event_loop_normal` / `event_loop_overlap`）、`overlap_utils.py` | 高 | 高 | SGLang 性能基石。overlap 把「启动前向即返回 + 结果延后一轮」做成 CPU/GPU 流水线，是 zero-overhead 调度的核心，也是最易出并发/正确性问题之处（见附录 A/B）。 |
| 3 | **组批决策与预算控制** | `schedule_policy.py`（`SchedulePolicy` / `PrefillAdder`）、`schedule_batch.py`、`prefill_delayer.py` | 高 | 高 | 在多重显存/token/条数预算下决定每轮收哪些请求、收多少、是否切块/抢占；直接决定吞吐与公平性（见子阶段 D 与附录 D）。 |
| 4 | **批次数据结构与状态机** | `schedule_batch.py`（`Req` / `ScheduleBatch`）、`utils.py`（`GenerationBatchResult`） | 高 | 中 | `Req` 的等待→运行→完成状态流转，以及 `ScheduleBatch → ForwardBatch` 的数据流，是调度层（CPU）到执行层（GPU）的承载体。 |
| 5 | **GPU 前向与采样后处理** | `tp_worker.py`、`scheduler_components/batch_result_processor.py`、`logprob_result_processor.py` | 高 | 中 | 把 `ScheduleBatch` 转 `ForwardBatch` 驱动 `ModelRunner`，并把前向结果整理为输出、计算 logprob、组织流式/非流式回包。 |
| 6 | **请求生命周期的资源安全** | `scheduler.py`（abort/timeout/retract 路径）、`session_controller.py` | 高 | 高 | 中止/超时/显存回撤等异常路径的资源安全收尾（`to_finish` 延迟标记等），是健壮性的关键，约定「等待中可直接丢、运行中须延迟收尾」（见附录 C）。 |
| 7 | **反分词与增量文本输出** | `detokenizer_manager.py`、`scheduler_components/output_streamer.py`、`output_sender.py` | 中 | 中 | 把 token id 流增量拼接为可打印文本并裁剪 stop 串，流式增量正确性依赖跨轮状态维护。 |
| 8 | **运行时可观测性与一致性核查** | `scheduler_components/`（`metrics_reporter.py`、`pool_stats_observer.py`、`invariant_checker.py`、`kv_events_publisher.py`、`profiler_manager.py`） | 中 | 中 | 指标上报、KV/池统计、watchdog 与不变量核查、profiler；对线上排障与防御性正确性至关重要，但不在请求主链路上。 |
| 9 | **权重热更新与全局静止** | `scheduler_components/weight_updater.py`、`scheduler_input_blocker.py`、`communicator.py`、`load_snapshot.py` | 中 | 高 | 从磁盘/分布式/IPC/tensor 更新权重，需全局屏障让系统静止、刷新缓存，复杂在于与在途请求和多进程的并发协调。 |
| 10 | **多模态处理** | `multimodal_processor.py`、`mm_utils.py`、`embed_types.py` | 中 | 中 | 多模态 processor 注册/查找、跨进程张量代理与 padding；难点在多模型差异与大张量跨进程传输。 |
| 11 | **分布式扩展：PP / DP / DP-attention** | `scheduler_pp_mixin.py`、`data_parallel_controller.py`、`scheduler_components/dp_attn.py` | 中 | 高 | PP 的有序收发与计算/通信重叠、DP 的请求分发、DP-attention 的跨 rank MLP 同步；都涉及多 rank 协商，调试成本高。 |
| 12 | **分层/稀疏缓存协调（HiCache / HiSparse）** | `cache_controller.py`、`hisparse_coordinator.py` | 中 | 高 | device/host/storage 间的异步搬运、预取与按 top-k 换入；与调度、KV 池强耦合，异步事件管理复杂。 |
| 13 | **PD 分离（Prefill/Decode 拆分）** | `disagg_service.py` 及 `scheduler.py` 内 PD 队列处理 | 中 | 高 | bootstrap/KV-store 辅助服务与各阶段队列管理；跨实例 KV 传输与握手是难点（见 abort 在 PD 各队列的处理）。 |
| 14 | **多 HTTP worker / 多 tokenizer** | `multi_tokenizer_mixin.py` | 中 | 中 | 多 TokenizerWorker 的路由与 socket 映射，提升前端分词吞吐。 |
| 15 | **会话与模板管理** | `session_controller.py`、`template_manager.py`、`template_detection.py` | 中 | 中 | 有状态多轮会话的请求树与缓存关联，以及 chat/completion 模板的检测与加载。 |
| 16 | **分词吞吐优化与收包节流** | `async_dynamic_batch_tokenizer.py`、`scheduler_recv_skipper.py`、`scheduler_components/idle_sleeper.py` | 低 | 低 | 动态批分词聚合、按 forward mode 降低收包频率、空闲休眠，属于边际性能优化。 |

**审计结论**：

- **完整性**：能力 1–7 构成请求主链路（必备核心），8–16 为高级/横切能力，整体覆盖完整；本目录的能力清单与实际代码基本对应。
- **准确性修正**：原「文件清单」存在与当前代码的偏差——① 大量原 `scheduler_*_mixin.py` 已重构进 `scheduler_components/` 子目录（如 `dp_attn.py`、`batch_result_processor.py`、`profiler_manager.py`、`invariant_checker.py`、`weight_updater.py`）；② 新增了 `communicator.py`、`embed_types.py`、`load_snapshot.py`、`template_detection.py` 等文件；③ score 能力实为 `tokenizer_manager_score_mixin.py`。下方文件清单中以 Mixin 文件名描述的若干项，应理解为「该能力现多由 `scheduler_components/` 承载」。

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

#### 参考框图（三进程 + ZMQ 队列方向 + 每条边的 `io_struct` 类型）

> 实线 `──▶` 为请求/输出的主数据流（沿请求生命周期单向流转）；
> 双向 `◀──▶` 为控制面的请求-应答（权重更新、缓存清理、暂停/恢复、负载查询等）。
> 三个进程之间均通过 ZMQ（PUSH/PULL，控制面为 REQ/REP 风格的请求-应答）解耦。

```
                          (HTTP / Engine 入口)
                                  │
                                  │  GenerateReqInput / EmbeddingReqInput
                                  │  （未分词的原始请求，可为单条或批次）
                                  ▼
                    ┌─────────────────────────────┐
                    │      TokenizerManager       │  前端进程（asyncio 事件循环）
                    │  分词 + 请求分发 + 结果回收   │
                    └─────────────────────────────┘
                       │  ▲                       ▲
   [ZMQ PUSH] 分词后下发 │  │ [ZMQ]                 │ [ZMQ] 控制面请求-应答
   TokenizedGenerateReq │  │ 控制面应答             │ ◀──▶ UpdateWeightsFrom*ReqInput/Output
   Input /              │  │ (RpcReqOutput 等)      │      FlushCacheReqInput/Output
   TokenizedEmbedding   │  │                       │      PauseGenerationReqInput …
   ReqInput /           │  │                       │      GetLoadsReqInput/Output
   Batch* 版本          │  │                       │      （权重更新/缓存/暂停/负载等）
                        ▼  │                       │
                    ┌─────────────────────────────┐
                    │         Scheduler           │  独立进程（连续批处理调度）
                    │  组批 → run_batch → 采样后处理 │  ← 控制面请求也在此进程处理
                    │  驱动 GPU 前向（TpModelWorker）│
                    └─────────────────────────────┘
                                  │
   [ZMQ PUSH] 批次 token id 输出   │  BatchTokenIDOutput        （生成：含 decode_ids 等）
                                  │  BatchEmbeddingOutput      （嵌入：原样透传，无需解码）
                                  ▼
                    ┌─────────────────────────────┐
                    │     DetokenizerManager      │  独立进程
                    │  反分词 + 增量可打印文本拼接   │  （嵌入输出仅透传）
                    └─────────────────────────────┘
                                  │
   [ZMQ PUSH] 解码后字符串输出      │  BatchStrOutput            （含 output_strs，流式为增量）
                                  │  BatchEmbeddingOutput      （透传回 TokenizerManager）
                                  ▼
                    ┌─────────────────────────────┐
                    │      TokenizerManager       │  handle_loop 按 rid 收集，
                    │  （回到前端，异步生成器返回）  │  以流式/非流式返回给 HTTP 调用方
                    └─────────────────────────────┘
```

**要点：**

- **前向主链单向流转**：`TokenizerManager ──▶ Scheduler ──▶ DetokenizerManager ──▶ TokenizerManager`，分别承载 `Tokenized*ReqInput` → `BatchTokenIDOutput` → `BatchStrOutput`。
- **嵌入请求不经过反分词逻辑**：`BatchEmbeddingOutput` 在 DetokenizerManager 处仅原样透传（无 token→文本解码）。
- **控制面单独成边**：权重更新、缓存清理、暂停/恢复、负载查询等走 `TokenizerManager ◀──▶ Scheduler` 的请求-应答，不与前向数据流混在同一条队列上。
- **分界点**：`GenerateReqInput`（未分词）→ `TokenizedGenerateReqInput`（已分词）的转换发生在 **TokenizerManager 进程**，之后 Scheduler 只处理已分词的请求形态。

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

> 📊 `event_loop_overlap` 的 CPU/GPU 重叠原理配图（核心思想、关键代码、时间轴对比、单轮内部步骤、正确性要点）已移至文末 **[附录 A：`event_loop_overlap` 的 CPU/GPU 重叠原理配图](#附录-aevent_loop_overlap-的-cpugpu-重叠原理配图)**，建议读完本小节后跳转查看。

> 📊 overlap 相比非 overlap 的效果提升（定量直觉、收益场景对比、代价与取舍、经验结论）已移至文末 **[附录 B：overlap 相比非 overlap 的效果提升](#附录-boverlap-相比非-overlap-的效果提升)**，建议读完本小节后跳转查看。

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

#### 参考流程图（批次数据流与组批预算 —— 各方法如何串联）

> 本图把子阶段 D 阅读清单里的方法按**一轮调度内的实际调用顺序**串起来：
> 左侧 **PREFILL 路径**（组新批）由 `get_new_batch_prefill` 驱动 `PrefillAdder` 在 token/显存预算内挑请求；
> 右侧 **DECODE 路径**（推进运行批）由 `update_running_batch` 驱动；
> 两者最终都产出 `ScheduleBatch`，再经 `ForwardBatch.init_new` 交给 GPU。

```
                    get_next_batch_to_run()  ── 每轮调度入口，决定本轮跑 prefill 还是 decode
                              │
            ┌─────────────────┴──────────────────┐
            ▼ (有可组的新请求)                     ▼ (无新批，推进运行批)
  ┌───────────────────────────┐        ┌───────────────────────────┐
  │   PREFILL 路径（组新批）    │        │   DECODE 路径（推进运行批）  │
  │   get_new_batch_prefill    │        │   update_running_batch      │
  └───────────────────────────┘        └───────────────────────────┘
            │                                        │
            ▼ ① 排序等待队列                          ▼ ① 预算不足则抢占
  SchedulePolicy.calc_priority()           preempt_to_schedule()
   ├─ CacheAware: _sort_by_longest_prefix   （回撤低优/后来的 req，
   │   （按 RadixCache 前缀命中长度排序）       release_kv_cache 腾显存）
   └─ Priority: 按 priority 排序                       │
            │                                         ▼ ② 为存活 req 续 1 token
            ▼ ② 逐个尝试加入，受预算约束        ScheduleBatch.prepare_for_decode()
  PrefillAdder(rem_total_tokens,            （seqlen+1、分配 KV slot、
              cur_rem_tokens)                  out_cache_loc、位置自增）
   ├─ add_chunked_req()  续跑上一轮被切块的 req         │
   ├─ add_one_req()      普通新 req                     │
   │    └─ Req.init_next_round_input()                  │
   │         （_refresh_fill_ids + tree_cache           │
   │          .match_prefix 前缀匹配 → prefix_indices    │
   │          → set_extend_input_len 算待 prefill 数）   │
   └─ 预算耗尽 / 命中 chunked → 停止收 req               │
            │                                           │
            ▼ ③ 把选中的 req 组成批                      │
  ScheduleBatch.prepare_for_extend()                    │
   （拼 input_ids、分配 req_pool/KV、                    │
     标记 chunked 中间块 inflight_middle_chunks）        │
            │                                           │
            └───────────────┬───────────────────────────┘
                            ▼
                    ScheduleBatch（本轮批次，含 forward_mode）
                            │
                            ▼  run_batch → tp_worker
                    ForwardBatch.init_new(batch, model_runner) ── 转 GPU 前向输入
                            │
                            ▼  GPU 前向 + 采样
                    process_batch_result（采样后处理）
                            │
                            ▼ 维护运行批集合
            ┌───────────────┴───────────────┐
            ▼ 剔除已完成 req                  ▼ prefill 批并入运行批
  ScheduleBatch.filter_batch()      ScheduleBatch.merge_batch()
   （req.finished() 为真者出队，       （新 prefill 批 ← 合并 → running_batch，
     释放其 KV / req_pool slot）         拼接各张量字段）
            │                               │
            └───────────────┬───────────────┘
                            ▼
                    running_batch（下一轮 decode 的输入）
                            └────────► 回到 get_next_batch_to_run()（继续下一轮）
```

**配合自检题理解：**

- **① `Req` 的方法链**（等待→运行→完成）：`init_next_round_input`（前缀匹配、算待 prefill 数）→ `prepare_for_extend`（进 prefill 批）→ `prepare_for_decode`（每轮续 1 token）→ `finished()` 为真后被 `filter_batch` 剔除并释放资源。
- **② 组批预算**：`PrefillAdder` 用 `rem_total_tokens`（含为运行请求预留未来生成空间的全局 KV 余量）与 `cur_rem_tokens`（当前这一步实际可占用的 KV 余量）双重约束，`add_one_req` 每加入一个 req 就扣减对应偏移；任一耗尽即停止收新 req（或把当前 req 切成 chunked 中间块）。详见下方 **PrefillAdder 专项说明**。
- **③ 抢占（preempt）触发时机**：仅在 **DECODE 路径**显存不足、无法为运行中 req 续 token 时，由 `preempt_to_schedule` 回撤低优先级/较晚的 req，`release_kv_cache` 腾出 KV 后再 `prepare_for_decode`。

#### PrefillAdder 专项说明（组批预算的核心）

> 源码：`schedule_policy.py` 的 `PrefillAdder`（类 `:536`、`add_one_req` `:1021`、`add_chunked_req` `:852`、`preempt_to_schedule` `:1206`）。
> 它是「**在多重显存/token 预算约束下，决定本轮 prefill 收哪些请求、收多少**」的执行体——`get_new_batch_prefill` 排好序后，逐个 `add_one_req` 喂给它，由它判定接纳 / 切块 / 停止。

**1）四类预算（构造期注入，逐请求扣减）**

| 预算 | 字段 | 含义 | 约束的是 |
| --- | --- | --- | --- |
| 总 KV 预算 | `rem_total_tokens`（属性）+ `rem_total_token_offset`（偏移） | 物理可用 + 可驱逐 − 偏移；偏移里**预留了运行中请求未来要生成的 token 空间**（按 `new_token_ratio` 折扣估算） | 全局显存「装不装得下」 |
| 当前步 KV 预算 | `cur_rem_tokens`（属性）+ `cur_rem_token_offset`（偏移） | 同口径但偏移只算**当前这一步实际占用**，不含未来预留 | 本步实际放得下放不下 |
| 输入 token 预算 | `rem_input_tokens` | 对应 `--max-prefill-tokens`，本轮 prefill 总输入上限 | 单轮喂进去的输入规模 |
| 分块 token 预算 | `rem_chunk_tokens` | 对应 `--chunked-prefill-size`；为 `None` 表示未启用分块 | 单个 chunk 的大小（触发切块） |
| SWA 池预算 | `rem_swa_tokens` | 仅混合 SWA 模型（如 Gemma2）有效，单独核算滑动窗口池 | SWA KV 池余量 |

**2）核心方法链**

```
add_one_req(req)                            ← 逐个请求入口（普通新请求）
  ├─ req.init_next_round_input()            前缀匹配，算出 extend_input_len（真正待 prefill 的 token 数）
  ├─ 计算本 req 占用 → 与四类预算比对
  │    ├─ 放得下且无需切块  → 计入 can_run_list，扣减各 offset，返回 CONTINUE
  │    ├─ 超过 rem_chunk_tokens → 切块：只收前 rem_chunk_tokens 个，
  │    │                          剩余记为 new_chunked_req（下一轮用 add_chunked_req 续跑），返回 OTHER
  │    └─ rem_total/cur/swa 任一 ≤ 0 → 返回 NO_TOKEN（停止收新 req）
  └─ budget_state()                          统一裁决返回 AddReqResult

add_chunked_req(req)                        ← 续跑上一轮被切块的中间块（优先于普通新请求）
add_one_req_ignore_eos(req)                 ← ignore_eos 请求的特殊预算估算（按剩余 token 预留）
preempt_to_schedule(req)                    ← 优先级调度下，回撤运行中低优 req 腾预算（→ preempt_list）
```

**3）三种产出（`AddReqResult`）与对应动作**

- `CONTINUE`：预算充足，继续取等待队列下一个请求。
- `NO_TOKEN`：KV/SWA token 预算耗尽，**本轮停止收新请求**。
- `OTHER`：因输入/分块/请求数上限等约束停止；常见于**当前请求被切成 chunked 中间块**（`new_chunked_req` 非空，下一轮续跑）。

**4）几个易混点**

- `rem_total_tokens` vs `cur_rem_tokens`：前者**含为运行请求预留的未来生成空间**（防止新 prefill 侵占 decode 的命脉），后者只看**当前步**的实际占用；两者都要 > 0 才可继续。
- 分块预算来自 `rem_chunk_tokens`（≈ `--chunked-prefill-size`），**不是** `cur_rem_tokens`——切块由前者触发。
- 抢占（`preempt_to_schedule`）**只在启用优先级调度时**于组批阶段发生，与 DECODE 路径里因显存不足的回撤是两条不同触发线。

> 📊 `prefill_max_requests`（配置硬上限）与 `max_prefill_bs`（运行时观测峰值）这对易混概念的辨析，以及「为什么要限制 prefill 请求条数」，见文末 **[附录 D：prefill 批的两个"上限"](#附录-dprefill-批的两个上限prefill_max_requests-vs-max_prefill_bs)**。
>
> 📊 正常路径之外的**异常路径**（请求中止 / 排队超时 / 运行超时 / 显存不足回撤）及其背后的资源安全约定，见文末 **[附录 C：请求的异常路径与资源安全](#附录-c请求的异常路径与资源安全中止--超时--抢占)**。

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
6. 等待中与运行中的请求被中止时，为何处理方式相反？`to_finish` 这个延迟标记解决了什么资源安全问题？（见附录 C）

### 学习方法提示（针对本目录）

- **按数据流读，别按文件读**：始终顺着上方「学习主线」那条链，遇到 Mixin 再按需跳转。
- **善用 `server_args.py`**：本目录大量分支由启动参数控制（如 `--chunked-prefill-size`、`--enable-overlap-schedule`、`--max-running-requests`），从参数反查处理逻辑能快速定位调度分支。
- **打点优于猜测**：调度路径状态多、跳转密，在 `event_loop_*` / `get_new_batch_prefill` / `process_batch_result` 三处打点，比纯读代码高效得多。
- **带注释副本**：`tokenizer_manager_annotated_zh.py` 是 `tokenizer_manager.py` 的逐行中文注释学习副本，读 B 阶段时可对照。

---

## 附录 A：`event_loop_overlap` 的 CPU/GPU 重叠原理配图

> 本附录配合 [子阶段 C：Scheduler 事件循环与连续批处理](#子阶段-cscheduler-事件循环与连续批处理第-57-天--本目录核心) 阅读。

**核心思想**：`event_loop_normal`（非重叠）每轮严格串行——CPU 调度 → 启动 GPU 前向 → **同步等 GPU 算完** → CPU 处理结果，两者互相阻塞。`event_loop_overlap` 则把「处理结果」**延后一轮**：`run_batch` 启动前向后不等结果，只把 `(batch.copy(), batch_result)` 压入 `result_queue`，立刻进入下一轮做 CPU 调度。于是**第 N 轮的 CPU 工作叠在第 N-1 轮的 GPU 前向之上**。

关键代码（`scheduler.py` 的 `event_loop_overlap`）：

```python
# Launch the current batch
if batch:
    batch_result = self.run_batch(batch)                    # 启动 GPU 前向，不等结果
    self.result_queue.append((batch.copy(), batch_result))  # 入队，延后处理
# Process the last batch
if self.last_batch:
    if not disable_overlap_for_batch:
        pop_and_process()                                   # 处理上一轮的结果
self.last_batch = batch
```

**时间轴对比：**

```
═════════ event_loop_normal（串行，CPU/GPU 互相等待）═════════
CPU: [调度B1]          [调度B2]          [调度B3]
GPU:         [前向B1]          [前向B2]          [前向B3]
         └─等待─┘└等待┘ └─等待─┘└等待┘
     ❌ CPU 调度时 GPU 空闲；GPU 前向时 CPU 空闲

═════════ event_loop_overlap（流水线，CPU 叠在 GPU 上）═════════
        轮1        轮2               轮3               轮4
CPU: [调度B1] │[调度B2 + 处理B1结果]│[调度B3 + 处理B2结果]│[调度B4 + 处理B3结果]
GPU:         │      [前向B1]       │      [前向B2]      │      [前向B3]
     时间 ────┴────────────────────┴───────────────────┴────────►
     ✅ 第N轮 CPU 工作 与 第N-1轮 GPU 前向 并行，CPU 开销被 GPU 时间隐藏
```

**单轮内部步骤（CPU 线程 vs GPU forward_stream）：**

```
┌────────────────────────── 第 N 轮 ──────────────────────────┐
│  CPU 线程                                GPU (forward_stream) │
│  1. recv_requests / process_input                            │
│  2. get_next_batch_to_run() → 本批 B_N                        │
│  3. run_batch(B_N) ───────────────────► [前向 B_N 在 GPU 运行] │
│     result_queue.append((B_N, result))         │ 同时进行 ↓   │
│  4. pop_and_process() → 处理上一轮 B_{N-1} 的结果 │            │
│     (取 token、判结束、流式输出、回收 KV)         │            │
│  5. launch_batch_sample_if_needed(B_N)          │            │
│  6. last_batch = B_N                            │            │
└──────────────────────────────────────────────────────────────┘
        ↓ 进入第 N+1 轮，再处理 B_N 的结果……
```

**保证正确性的关键点：**

- **`result_queue` 最多积压一批**：每轮压入一批、弹出一批，流水线深度固定为 1。
- **`batch.copy()`**：入队时快照本批状态，避免后续修改污染待处理结果。
- **`FutureMap` 中继**：弹出处理时 GPU 采样的 token 可能还没拷回 CPU，用 FutureMap 占位「未来 token id」，后续再解析真实值。
- **采样放在处理完上一批之后**（步骤 5）：采样可能依赖上一批结果（如 grammar 约束状态）。
- **WAR 屏障**：本轮调度若要写共享 GPU 缓冲，需 `schedule_stream.wait_stream(forward_stream)`，确保上一轮前向已读完，避免读写竞争。
- **必要时关闭重叠**（`is_disable_overlap_for_batch`，如连续两个 prefill、spec+grammar 组合）：先 `pop_and_process()` 处理完上一批再启动本批，退化为串行以保证正确性。

> 一句话总结：overlap 通过「**启动前向即返回 + 结果延后一轮处理**」，让第 N 轮的 CPU 调度与第 N-1 轮的 GPU 前向在时间上重叠，把 CPU 调度开销藏进 GPU 计算时间里——这就是 SGLang「zero-overhead 调度」的核心。

---

## 附录 B：overlap 相比非 overlap 的效果提升

> 本附录配合 [子阶段 C：Scheduler 事件循环与连续批处理](#子阶段-cscheduler-事件循环与连续批处理第-57-天--本目录核心) 阅读，是 [附录 A](#附录-aevent_loop_overlap-的-cpugpu-重叠原理配图) 的延伸。

**收益从哪来（定量直觉）**：设单轮 CPU 调度+后处理耗时为 `T_cpu`，单轮 GPU 前向耗时为 `T_gpu`。

```
非 overlap：每轮墙钟 ≈ T_cpu + T_gpu     （串行相加）
overlap：  每轮墙钟 ≈ max(T_cpu, T_gpu)   （两者重叠，取较大者）
```

- **理想加速比** ≈ `(T_cpu + T_gpu) / max(T_cpu, T_gpu)`。
- 当 `T_cpu ≈ T_gpu` 时，单轮墙钟最多可省去近一半 → **吞吐接近翻倍**。
- 当 `T_cpu << T_gpu`（GPU 远大于 CPU）时，`max ≈ T_gpu`，CPU 开销几乎被**完全隐藏**，提升仍可观但比例变小。

**哪种场景提升最明显**：

| 场景 | T_cpu 占比 | overlap 收益 |
| --- | --- | --- |
| **decode 为主、batch 大、序列多**（高并发解码） | 高（每轮要遍历大量 req 做采样后处理/出队/流式回包） | **最大**——CPU 后处理与 GPU 解码前向充分重叠 |
| 小模型 / 短序列 | 偏高（GPU 前向快，CPU 占比相对高） | 明显 |
| 超大模型 / 超长 prefill | 低（GPU 前向极重） | 较小（CPU 本就被淹没在 GPU 时间里） |
| 单请求、低并发 | 低 | 较小（没有足够批量摊薄调度开销） |

**为什么不是"无脑全开"——overlap 的代价与取舍**：

- **延迟一轮可见**：第 N 轮的结果在第 N+1 轮才处理，单请求**首 token 延迟（TTFT）会多等约一轮**。因此 `is_disable_overlap_for_batch` 会在「连续两个 prefill」时关闭重叠，优先压低首个 prefill 的 TTFT（牺牲少量吞吐换延迟）。
- **显存/引用开销**：`batch.copy()` 快照 + `batch_record_buf` 钉住张量 2 个迭代，需多保留约一轮的中间张量引用。
- **复杂度与兼容性**：需要 `FutureMap` 中继未来 token、WAR 屏障防读写竞争；个别组合（如 spec + grammar + decode）暂不支持重叠，必须退化为串行。

**经验结论**：在线服务的**高并发 decode** 工况下，overlap 通常带来**显著的吞吐提升（常见量级为百分之十几到接近翻倍，取决于 `T_cpu/T_gpu` 比值与 batch 规模）**；而对**单请求 TTFT 敏感**或 GPU 前向极重的场景，收益变小甚至需要按批关闭。SGLang 默认开启重叠（`--disable-overlap-schedule` 可关闭），并通过按批动态关闭兼顾延迟与吞吐。

---

## 附录 C：请求的异常路径与资源安全（中止 / 超时 / 抢占）

> 本附录配合 [子阶段 C](#子阶段-cscheduler-事件循环与连续批处理第-57-天--本目录核心) 与 [子阶段 D](#子阶段-d批次数据流与组批预算第-89-天) 阅读。
> 正文的学习主线只画了**正常路径**（happy path）；但调度器一大半的健壮性代码在处理**异常路径**——客户端主动断开、请求排队/运行超时、显存不足被迫回撤。这些路径的共同主题是**资源安全**：一个请求可能正持有 KV 缓存、可能正处在 GPU 前向中途，绝不能「说删就删」。

### C.1 两种超时中止：为什么等待中和运行中处理方式相反

两个方法都在每轮调度入口 `get_next_batch_to_run` 末尾被调用（`scheduler.py:2721-2722`），但动作截然不同：

| 方法 | 行号 | 针对的请求 | 是否持有 KV | 动作 |
| --- | --- | --- | --- | --- |
| `_abort_on_waiting_timeout` | `:2549` | **等待队列**中超时的 req | 否（还没组批） | **直接**发 `AbortReq` 给 tokenizer + 从 `waiting_queue` 出队 |
| `_abort_on_running_timeout` | `:1512` | **运行批次**中超时的 req | 是 | **延迟**：仅置 `req.to_finish = FINISH_ABORT(...)`（`:1536`），不直接删 |

**为什么相反？** 关键在于**有没有分配 KV、会不会正在 GPU 前向中途**：

- 等待中的请求**尚未占用任何 KV / req_pool slot**，也不可能正在前向里，所以可以原地丢弃——立即出队、立即回包，最省事。
- 运行中的请求**正持有 KV 缓存，且可能此刻正被 GPU 计算**（尤其 overlap 模式下结果还在 `result_queue` 里）。若在这里直接删它、释放它的 KV，正在跑的前向就会读到已释放的显存 → 崩溃或数据错乱。因此只能**打一个延迟标记** `to_finish`，把真正的「回包 + 释放 KV」交给正常的完成路径去做。

### C.2 `to_finish` 这个延迟标记在哪被消费

`to_finish` 不是立刻生效的中止，而是「**下一次该请求走到结果处理时，按中止收尾**」的约定：

```
_abort_on_running_timeout (:1512)
  └─ req.to_finish = FINISH_ABORT(...)      # 只打标记，不动 KV
              │
              ▼  （该 req 照常再跑/收尾一次）
process_batch_result (:3277) 内的中止收尾 (:3332-3340)
  └─ for req in reqs_to_abort:
       abort_reason = req.to_finish          # 读出标记
       send_to_tokenizer(AbortReq(...))      # 此刻才回包
     （随后走正常 filter_batch → 释放 KV / req_pool slot）
```

同样的「置 `to_finish` 而非直接删」手法也用在**客户端主动中止**里——`AbortReq` 处理逻辑对「已在 running batch 里 decode 的请求」采用 abort method 3（`:4252-4256`：`req.to_finish = FINISH_ABORT()`，让它再跑一步 decode 再收尾），而对还在等待队列的请求才直接移除。**判据始终一致：是否已占用 KV / 是否可能在前向中途。**

### C.3 第三条异常路径：decode 显存不足时的抢占回撤（retract）

除超时/主动中止外，还有一条由**显存压力**触发的异常路径，已在子阶段 D 正文提及，这里归并对照：

- 触发点：**DECODE 路径**为运行中 req 续 token 时 KV 池告罄（`update_running_batch` 内）。
- 动作：回撤（retract）部分运行中 req，`release_kv_cache` 腾出显存，被回撤的 req 重新入等待队列（其已生成的 token 不丢，靠前缀缓存复用）。
- 收尾：`process_batch_result` 里同样会统计 `retracted_reqs`（`:3325-3329`）并调整 `new_token_ratio`，让后续 prefill 预留更保守，减少再次回撤。

> 三条异常路径的统一心智：**等待中 → 可直接丢；运行中 → 必须延迟/经正常收尾路径释放资源**。读代码时凡看到 `to_finish`、`reqs_to_abort`、`retracted_reqs`，都属于「异常路径的资源安全收尾」这一族。

- **动手打点**：构造一个会超时的慢请求（或客户端中途断开），在 `_abort_on_running_timeout`(`:1512`)、`to_finish` 消费处(`:3332`) 打点，观察「打标记」与「真正回包+释放」之间隔了几轮。
- **自检**：① 为什么运行中的请求不能像等待中的那样被立即删除？② `to_finish` 从被设置到被消费，中间这个请求还会发生什么？

---

## 附录 D：prefill 批的两个"上限"——`prefill_max_requests` vs `max_prefill_bs`

> 本附录配合 [子阶段 D 的 PrefillAdder 专项说明](#prefilladder-专项说明组批预算的核心) 阅读。
> 这两个名字相近、都和「一批 prefill 收多少请求」有关，极易混淆，但**性质完全不同**：一个是**配置的硬上限**，一个是**运行时观测到的峰值统计**。

| | `prefill_max_requests` | `max_prefill_bs` |
| --- | --- | --- |
| 性质 | **配置项**（硬上限） | **运行时统计**（历史观测峰值） |
| 来源 | `server_args`，CLI `--prefill-max-requests`，默认 `None`（不限） | `Scheduler` 内部变量，初始 `0`（`scheduler.py:1100`） |
| 如何变化 | 启动后固定 | 每组完一批就 `max(self.max_prefill_bs, len(can_run_list))` 累积刷新（`:3184`） |
| 作用点 | `PrefillAdder.add_one_req`：`len(can_run_list) >= x` 时**停止收请求**（`schedule_policy.py:1064`，返回 `OTHER`） | 喂给 `prefill_delayer_single_pass.negotiate_should_allow_prefill`（`schedule_policy.py:1050`），参与**跨 DP rank 的 prefill 负载协商**与统计，**本身不卡准入** |
| 一句话 | 「这一批最多收几个，到了就不收了」 | 「历史上我最多一次收过几个」——一个被观测出来、再反过来指导延迟器的数字 |

**易错点**：看到 `max_prefill_bs=self.max_prefill_bs` 被传进 `PrefillAdder`（`scheduler.py:3013`）会以为它是准入上限——其实它只流向 prefill **延迟器**做负载协商；真正「数到几就停」的硬上限是 `prefill_max_requests`（`:1064`）那一行。

**为什么需要限制 prefill 的请求条数（而不是一路加到显存上限）？** 显存（token 预算）只约束「装不装得下」，但**请求条数**本身会带来与 token 数无关的开销与公平性问题：

1. **逐序列 / ragged-varlen 的固定开销**：prefill 的 attention 是变长拼接，请求条数越多，per-sequence 的元数据、kernel launch、边界处理开销越大——即使总 token 数不变。
2. **固定条数资源**：`max_running_requests`、各类按「最大请求数」预分配的元数据缓冲、CUDA graph 的 max batch size，都是按**条数**而非 token 数封顶的。
3. **队头公平性**：一批里塞进上千个极短请求，会让后到但更重要的请求等待整批 prefill 完成，恶化队头阻塞。

> ⚠️ 一个**常见误解**（本仓库讨论中曾出现并被纠正）：「放进上千个极短输入会让 decode 的 TPOT 炸裂」。**这是不准确的**——极短输入的 KV 很小、attention 很便宜，不会让 decode「雪崩」。真实机制是：**TPOT 随 batch 增大、越过 GEMM ridge point 后大致线性上升**，这是吞吐/延迟的权衡，不是崩溃。所以限制条数的真正理由在**上面三点（prefill 侧开销 + 固定条数资源 + 公平性）**，而非「decode 扛不住」。
