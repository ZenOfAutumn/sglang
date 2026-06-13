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
