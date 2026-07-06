# SGLang 工程总结与学习计划

> 本文面向想系统掌握 SGLang 源码与设计的工程师，提供一份"工程全景 + 分阶段学习路线"。
> 所有文件路径均相对于仓库根目录，核心代码位于 `python/sglang/srt`（srt = SGLang RunTime）。

---

## 一、工程总览

### 1.1 这是什么

SGLang 是一个**高性能大模型推理服务框架**（LLM/多模态/扩散模型 serving），目标是在从单卡到大规模分布式集群的各种部署下，提供**低延迟、高吞吐**的推理。它已成为业界事实标准之一，在 40 万+ GPU 上运行，并被大量 RL 后训练框架（verl、slime、AReaL 等）用作 rollout 后端。

### 1.2 核心能力（决定了学习的重点）

> 「重要性」= 对延迟/吞吐/成本或部署可行性的影响权重（★ 越多越关键）；「复杂性」= 上手与改动的难度（涉及的进程数、并发/同步、底层 kernel、跨节点协调等）。建议按「重要性高 × 复杂性中」优先攻克（如调度、RadixAttention）。

> 下表已按「重要性降序 → 同重要性再按复杂性降序」排列，越靠前越值得优先理解。

| 能力                     | 一句话说明                             | 对应源码区域                                           | 重要性 | 复杂性 | 为什么重要 / 难在哪 |
| ------------------------ | -------------------------------------- | ------------------------------------------------------ | ------ | ------ | ------------------- |
| **并行**                 | TP/PP/EP/DP（张量/流水/专家/数据并行） | `srt/distributed/`, `srt/layers/dp_attention.py`       | ★★★★★ | ★★★★★ | 决定能否跑超大模型与扩展上限；难在多种并行维度组合、集合通信（NCCL）正确性与负载均衡（尤其 EP/EPLB） |
| **Zero-overhead 调度**   | CPU 调度与 GPU 计算重叠，消除调度开销  | `srt/managers/scheduler.py` (`event_loop_overlap`)     | ★★★★★ | ★★★★☆ | 决定性能上限的核心，把 CPU 组批/采样后处理与 GPU 前向流水线重叠；难在主循环的多阶段状态管理、与 overlap 模式下结果延后一拍的处理 |
| **Continuous batching**  | 连续批处理，动态加入/退出请求          | `srt/managers/schedule_batch.py`, `schedule_policy.py` | ★★★★★ | ★★★★☆ | 高吞吐的基石，请求可逐步进出同一批而非整批等待；难在 prefill/decode 混合批、`Req`/`ScheduleBatch` 状态机与显存预算的动态裁剪 |
| **RadixAttention**       | 基于基数树的前缀缓存复用 KV cache      | `srt/mem_cache/radix_cache.py`                         | ★★★★★ | ★★★☆☆ | SGLang 的标志性能力，多请求共享前缀时直接省掉 prefill 计算，对 TTFT/吞吐影响极大；难在基数树的分裂/合并、LRU 驱逐与 KV 物理内存的引用计数一致性 |
| **Paged attention**      | 分页 KV 内存管理                       | `srt/mem_cache/memory_pool.py`, `mem_cache/allocator/` | ★★★★★ | ★★★☆☆ | 分页化是消除显存碎片、支撑大并发与前缀复用的前提；难在多种分配器（`paged.py`/`token.py`/`swa.py`/`mamba.py`）与上层 cache 的协同 |
| **PD 分离**              | Prefill/Decode 解耦部署                | `srt/disaggregation/`                                  | ★★★★☆ | ★★★★★ | 大规模部署下分别按 prefill/decode 特性独立扩缩、提高利用率；难在跨节点 KV 传输（Mooncake/NIXL/MORI）、bootstrap 建链与两端调度协调 |
| **Speculative decoding** | 投机解码（EAGLE 等）                   | `srt/speculative/`                                     | ★★★★☆ | ★★★★★ | 在内存受限的 decode 阶段成倍提速；难在 draft/verify 两阶段、专用 CUDA Graph runner、与连续批和 attention 后端的耦合 |
| **量化**                 | FP4/FP8/INT4/AWQ/GPTQ                  | `srt/layers/quantization/`                             | ★★★★☆ | ★★★★☆ | 直接降低显存与成本、提升吞吐；难在众多 scheme（AWQ/GPTQ/compressed-tensors/FP8/FP4）与硬件/kernel 适配 |
| **CUDA Graph**           | 捕获静态图降低 kernel launch 开销      | `srt/model_executor/runner/`（`base/decode/prefill_cuda_graph_runner.py`） | ★★★★☆ | ★★★★☆ | decode 阶段消除大量小 kernel 的 launch 开销，显著降延迟；难在静态形状捕获/重放、padding 策略与各特性（spec/PD/多模态）的专用 runner |
| **Chunked prefill**      | 长 prompt 分块预填充                   | `srt/managers/schedule_policy.py`                      | ★★★★☆ | ★★★☆☆ | 避免长 prompt 阻塞解码、平滑显存与延迟；难在分块大小与 decode 请求的调度权衡（`--chunked-prefill-size`）及跨 chunk 的状态衔接 |
| **多 LoRA**              | 批量 LoRA                              | `srt/lora/`                                            | ★★★☆☆ | ★★★★☆ | 一份基座服务多业务微调权重，省显存；难在同批内不同 LoRA 的批处理（SGMV 等 triton kernel）与动态加载 |
| **结构化输出**           | JSON/正则约束解码（压缩 FSM）          | `srt/constrained/`                                     | ★★★☆☆ | ★★★☆☆ | Agent/工具调用刚需，保证输出可被程序解析；难在多 grammar 后端（xgrammar/outlines/llguidance）与采样、压缩 FSM 的对接 |

### 1.3 技术栈

- 语言：Python（运行时主体）+ C++/CUDA（高性能 kernel，见 `sgl-kernel/`、`python/sglang/jit_kernel/`）
- 框架：PyTorch；注意力后端 FlashInfer / FlashAttention / FlashMLA 等
- 通信：基于 ZMQ 的多进程 IPC + NCCL 等集合通信
- 模型支持：168+ 模型实现（`srt/models/`），兼容 HuggingFace 与 OpenAI API

### 1.4 顶层目录地图

```
sglang/
├── python/sglang/          # Python 主包
│   ├── srt/                # ★ 核心运行时（学习重点）
│   ├── lang/               # 前端 DSL（编程模型，可选）
│   ├── cli/                # 命令行入口
│   ├── jit_kernel/         # 轻量 JIT CUDA kernel
│   ├── eval/ test/ bench/  # 评测/测试/基准
├── sgl-kernel/             # 重量级 AOT C++/CUDA kernel
├── sgl-model-gateway/      # 模型网关（路由/负载均衡，Rust）
├── docs/                   # 文档（本文所在）
├── examples/               # 使用示例（runtime / frontend_language / ...）
├── benchmark/              # 各类基准脚本
├── test/                   # CI 测试套件
└── docker/ scripts/ 3rdparty/
```

---

## 二、架构全景（必须先建立的心智模型）

### 2.1 多进程架构

SGLang 把一次推理拆成**三类进程**，通过 ZMQ 通信，实现 CPU 工作与 GPU 计算的流水线重叠：

```
HTTP 请求
   │
   ▼
┌─────────────────┐   tokenize    ┌───────────────┐   batch+forward   ┌───────────────────┐
│ TokenizerManager│ ───────────►  │   Scheduler   │ ────────────────► │ DetokenizerManager │
│  (前端/异步IO)   │ ◄───────────  │ (批处理+模型)  │ ◄──────────────── │   (解码文本)        │
└─────────────────┘   stream out  └───────────────┘                   └───────────────────┘
   src/entrypoints       managers/scheduler.py            managers/detokenizer_manager.py
   managers/tokenizer_manager.py
```

- **TokenizerManager** (`managers/tokenizer_manager.py:178`)：接收请求、分词、流式返回。`generate_request` (`:478`) 是入口。
- **Scheduler** (`managers/scheduler.py:273`)：核心大脑。组批、管理 KV cache、驱动模型前向。两个主循环：`event_loop_normal` (`:1300`) 与零开销的 `event_loop_overlap` (`:1328`)。进程入口 `run_scheduler_process` (`:3550`)。
- **DetokenizerManager**：把 token id 解码回文本流式输出。
- **TpWorker / ModelRunner**：Scheduler 内部调用，真正执行模型前向。

### 2.2 一次请求的完整链路

1. `entrypoints/http_server.py` 收到 HTTP（OpenAI 兼容 API 在 `entrypoints/openai/`）。
2. `entrypoints/engine.py` 的 `Engine` 封装；离线场景直接用 Engine。
3. → `TokenizerManager.generate_request`：分词 → 经 ZMQ 发给 Scheduler。
4. `Scheduler.recv_requests` (`:1420`) 收请求 → `schedule_policy.py` 组批 → `schedule_batch.py` 构造 `ScheduleBatch`。
5. → `ModelRunner.forward` (`model_executor/model_runner.py:2700`) 执行前向（可能走 CUDA Graph）。
6. 采样 (`layers/sampler.py`) → 结果经 `process_batch_result` (`scheduler.py:2808`) → DetokenizerManager → 流式回传。

### 2.2.1 一次请求的时间轴与监控

下面以「时间轴」方式串起一个请求从进入到完成经历的**每个阶段、对应的时间戳采集点、以及暴露的监控指标**。所有时间戳的真相源是 `srt/observability/req_time_stats.py`，阶段名定义在 `RequestStage`，时间戳用 `time.perf_counter()`（单调时钟）采集，跨进程传播时再用 `convert_time_to_realtime*` 校准回真实时间。

> 监控有两条互补通路：**Prometheus 指标**（`metrics_collector.observe_*`，用于聚合统计）与**分布式链路追踪 trace span**（`trace_slice`，用于单请求级火焰图，需开启 tracing）。下表「阶段名」即 trace span 名，「监控指标」标注是否会 `observe_per_stage_req_latency`（✅=该阶段单独上报 Prometheus 直方图）。

#### 统一模式（非 PD 分离，`disagg_mode = unified`）

```
t0 ──────────► t1 ──────────► t2 ──────────► t3 ──────────► t4 ──────────► t5 ──────────► t6
created    tokenize_     api_server_   scheduler_    wait_queue_   forward_     prefill_      completion
_time      finish        dispatch      recv_time     entry_time    entry_time   finished      _time
           _time         _finish_time                                           _time
│           │             │             │             │             │            │              │
│  Tokenizer 进程         │  跨 ZMQ      │        Scheduler 进程：排队 → 调度 → prefill → decode 循环  │
└── 分词 ────┴── 分发 ────┴── 传输 ──────┴── 排队等待 ─┴── (前向计算) ┴─ 首 token ─┴── 解码循环 ──┴── 结束
```

按时间顺序的阶段与监控点：

| # | 阶段（trace span / 含义） | 时间戳采集函数 | 进程 | 监控指标 |
| - | --- | --- | --- | --- |
| 1 | **请求创建** `created_time`：请求生命周期起点；同时开启 `tokenize` span | `APIServerReqTimeStats.set_created_time` | Tokenizer | 起点，`trace_req_start` |
| 2 | **`tokenize`**：分词完成 | `set_tokenize_finish_time` | Tokenizer | trace span |
| 3 | **`api_server_dispatch`**：API server 把请求经 ZMQ 分发给下游 | `set_api_server_dispatch_time` / `_finish_time` | Tokenizer | trace span |
| 4 | **`request_process`**：Scheduler 收到请求（`scheduler_recv_time`）到进入等待队列 | `set_scheduler_recv_time` → `set_wait_queue_entry_time` | Scheduler | ✅ `request_process` |
| 5 | **`prefill_waiting`**：在等待队列里排队，等待被组批调度（即排队时间 `queue_time`） | `set_forward_entry_time` | Scheduler | ✅ `observe_queue_time`（队列时间） |
| 6 | **`prefill_forward`**：prefill 前向计算（首 token 产出）；chunked prefill 会拆成多个 `chunked_prefill` 子片 | `set_prefill_finished_time` / `set_last_chunked_prefill_finish_time` | Scheduler | ✅ `prefill_forward`、✅ `chunked_prefill` |
| 7 | **`decode_loop`**：逐步解码循环，每步一个 `decode_forward`，`decode_ct` 记录步数 | `set_last_decode_finish_time` / `set_last_scheduled_time` | Scheduler | trace span（每步） |
| 8 | **请求完成** `completion_time`：生成结束 | `set_completion_time`（或 `set_quick_finish_time`） | Scheduler | 终点，`trace_req_finish` |
| 9 | **回传客户端** `response_sent_to_client_time`：结果经 Detokenizer 流式发回 | `set_response_sent_to_client_time` | Tokenizer | 出参 meta_info |

由这些时间戳派生的**端到端延迟指标**（`APIServerReqTimeStats`）：

- **TTFT（首 token 延迟）** = `first_token_time - created_time`（`get_first_token_latency`），trace 属性 `GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN`。
- **E2E（端到端延迟）** = `finished_time - created_time`（`get_e2e_latency`）。
- **解码延迟** = `finished_time - first_token_time`（`get_decode_latency`），并据此算 `decode_throughput`。
- **排队时间** = `forward_entry_time - wait_queue_entry_time`（`get_queueing_time`）。
- Scheduler 侧还会打印 `queue_duration` / `forward_duration`（`convert_to_duration`）。

#### PD 分离模式（Prefill / Decode disaggregation）

PD 分离把 prefill 与 decode 拆到不同节点，时间轴在中间多出 **KV cache 跨节点传输** 阶段（注释见 `SchedulerReqTimeStats` 文档串）：

- **Prefill 节点**：`prefill_prepare` → `prefill_bootstrap`（与对端握手建链）→ `prefill_waiting` → `prefill_forward` → `prefill_transfer_kv_cache`（把 KV 发往 decode 节点）。
  - 监控：✅ `prefill_bootstrap`、✅ `prefill_transfer_kv_cache`，以及 KV 传输速率 `transfer_speed_gb_s`、总量 `transfer_total_mb`（`compute_and_observe_kv_transfer_metrics`）。
- **Decode 节点**：`decode_prepare`（预分配队列）→ `decode_bootstrap` → `decode_waiting`（接收 KV）→ `decode_transferred` → `decode_forward` 解码循环 → `completion`。
  - 监控：✅ `decode_prepare`、✅ `decode_bootstrap`、✅ `decode_waiting`、✅ `decode_transferred`、✅ `fake_output`、✅ `quick_finish`。

```
[Prefill 节点]  bootstrap_queue ─► wait_queue ─► prefill_forward ─► transfer_queue ══(KV cache)══╗
                                                                                                 ▼
[Decode 节点]                         prealloc_queue ─► transfer_queue ─► wait_queue ─► decode_loop ─► completion
```

**名词解释（PD 分离）：** 整体架构见 `srt/disaggregation/README_zh.md`，prefill 端调度在 `disaggregation/prefill.py`、decode 端在 `disaggregation/decode.py`。核心思路是把请求的 **预填充（prefill）** 与 **解码（decode）** 拆到不同节点，prefill 算完后通过可插拔的 KV 传输后端（Mooncake / NIXL / MORI 等）把 KV cache 搬到 decode 节点继续解码。

Prefill 节点各阶段：

| 阶段 / 队列 | 名词解释 |
| --- | --- |
| **bootstrap（引导/建链）** | prefill 与对端 decode 节点**握手建链**的过程：交换元数据、确认 KV 传输通道就绪。`prefill_bootstrap_queue_entry_time` 标记进入 bootstrap 队列、`bootstrap_done_time` 标记建链完成。 |
| **bootstrap_queue（引导队列）** | 等待与 decode 节点完成 bootstrap 握手的排队队列，是 prefill 端请求的第一个落点。 |
| **wait_queue（等待队列）** | 握手完成后等待被组批做 prefill 前向计算的队列（与统一模式的等待队列同义）。 |
| **prefill_forward（预填充前向）** | 真正执行 prefill 计算、产出 KV cache 与首 token 的阶段。 |
| **transfer_queue（KV 传输队列）** | prefill 完成后，等待把 KV cache 通过传输后端发往 decode 节点的队列。`prefill_transfer_queue_entry_time` 进入、`prefill_kv_transfer_finish_time` 传输完成；伴随 `transfer_speed_gb_s`（速率）、`transfer_total_mb`（总量）指标。 |

Decode 节点各阶段：

| 阶段 / 队列 | 名词解释 |
| --- | --- |
| **prealloc_queue（预分配队列）** | decode 端请求的第一个落点：等待为「即将从 prefill 节点接收的 KV cache」**预分配显存**（KV 页 / token 槽位）。`decode_prealloc_queue_entry_time` 标记进入。 |
| **bootstrap（建链）** | 与 prefill 端对应的握手过程，`bootstrap_done_time` 标记完成；prealloc 阶段内部又细分为 bootstrap 子阶段与 alloc 等待子阶段。 |
| **transfer_queue（KV 传输队列）** | 显存预分配好后，等待**实际接收** prefill 节点发来的 KV cache 的队列。`decode_transfer_queue_entry_time` 标记进入。 |
| **wait_queue（等待队列）** | KV cache 接收落地后，等待被调度进入解码的队列。 |
| **prebuilt（预构建）** | decode 端**跳过 prefill 前向**、仅用接收到的 KV cache 与元数据直接构造出可解码状态（见 `decode_schedule_batch_mixin.py` 的「预构建 extend 批」）。`decode_prebuilt_finish_time` 标记完成，对应 trace span `fake_output`。 |
| **decode_loop（解码循环）** | 正式逐步解码，直到 `completion`。 |
| **quick_finish（快速结束）** | 特殊路径：某些请求（如刚接收完即满足结束条件）无需进入完整解码循环即可直接结束（`set_quick_finish_time`）。 |

> 一句话对照：**prefill 节点**多出 bootstrap（建链）与 transfer（发送 KV）两类额外阶段；**decode 节点**多出 prealloc（预分配显存）、transfer（接收 KV）、prebuilt（用 KV 直接建状态、跳过 prefill）三类额外阶段。二者通过 KV cache 跨节点传输衔接。

#### 投机解码（speculative decoding）

在 decode 阶段内，每步进一步细分为两个 span：

- **`spec_draft`**：草稿模型生成候选 token（`set_spec_draft_start_time` / `set_spec_draft_end_time`）。
- **`spec_verify`**：目标模型并行校验，trace 属性记录 `num_correct_drafts`（接受的草稿数）（`set_spec_verify_start_time` / `set_spec_verify_end_time`）。

#### 多模态（EPD encode）

带图像/音频的请求在最前面多一个 **`mm_encode`** 阶段（`EncoderReqTimeStats.set_mm_encode_start_time` / `_end_time`），位于 tokenize 之后、进入 Scheduler 之前。

> 小结：一个请求的时间轴 = `created → tokenize → dispatch → (scheduler) request_process → prefill_waiting → prefill_forward → decode_loop → completion → response_sent`；PD 分离在中间插入 KV 传输阶段，投机解码在 decode 内细分 draft/verify，多模态在最前插入 encode。每个阶段都有对应的 `set_*_time` 采集点、可选的 Prometheus 直方图（`metrics_is_observed=True` 的阶段）与 trace span，便于定位延迟瓶颈。

### 2.3 核心模块速查表


| 模块       | 路径                  | 关键文件/类                                                                                      |
| ---------- | --------------------- | ------------------------------------------------------------------------------------------------ |
| 服务入口   | `srt/entrypoints/`    | `http_server.py`, `engine.py`, `EngineBase.py`                                                   |
| 调度管理   | `srt/managers/`       | `scheduler.py`, `tokenizer_manager.py`, `schedule_batch.py`, `schedule_policy.py`                |
| 内存/缓存  | `srt/mem_cache/`      | `radix_cache.py` (`RadixCache:285`, `TreeNode:222`), `memory_pool.py`, `allocator/`              |
| 模型执行   | `srt/model_executor/` | `model_runner.py` (`ModelRunner:387`), `forward_batch_info.py`, `runner/*_cuda_graph_runner.py`  |
| 模型实现   | `srt/models/`         | 168+ 模型，如`llama.py`, `qwen2.py`, `deepseek_v2.py`                                            |
| 计算层     | `srt/layers/`         | `radix_attention.py`, `attention/`, `moe/`, `quantization/`, `sampler.py`, `logits_processor.py` |
| 配置       | `srt/`                | `server_args.py`（6700+ 行，所有启动参数的真相源）                                               |
| 投机解码   | `srt/speculative/`    | `eagle_worker.py`                                                                                |
| PD 分离    | `srt/disaggregation/` | `decode.py`, `prefill`/`encode` 相关                                                             |
| 并行       | `srt/distributed/`    | `parallel_state.py`；`layers/dp_attention.py`                                                    |
| 结构化输出 | `srt/constrained/`    | FSM/grammar 后端                                                                                 |
| LoRA       | `srt/lora/`           | 多 LoRA 批处理                                                                                   |
| 前端 DSL   | `python/sglang/lang/` | `interpreter.py`, `ir.py`, `api.py`                                                              |

---

## 三、分阶段学习计划

> 节奏建议：每天 1.5–2 小时，全程约 6 周。每个阶段包含**阅读 → 动手 → 自检**三步。先跑起来再读源码，永远比纯读代码高效。

### 3.0 如何使用本计划（学习方法论）

每个阶段都遵循同一套「四件套」，建议严格照做，避免陷入「只读不练」：

1. **阅读（Read）**：按给定顺序读源码，先读类/函数签名与 docstring，再读主流程，最后才抠细节。大文件（>1K 行）用编辑器的符号大纲或 `grep` 定位关键符号，不要逐行通读。
2. **动手（Do）**：跑给定实验，改一个参数、看一个指标变化。**先复现现象，再回头读实现**。
3. **自检（Check）**：合上代码，尝试口头/画图回答「自测题」。答不上来说明还没真懂，回到阅读。
4. **产出（Output）**：每阶段留一个可检索的产出物（一张图 / 一段笔记 / 一次 profiling trace），沉淀到 `docs/` 或个人笔记，形成可复习的知识资产。

> **验收标准（Definition of Done）**：每阶段末尾给出「✅ 通关标准」，全部达成才进入下一阶段。宁可慢，不要跳。

### 阶段 0：跑起来（第 0.5 周）

**目标**：在本地/容器里启动一个 SGLang server 并发请求成功。

- 阅读：`README.md`、`docs/get_started/install.md`、`docs/basic_usage/send_request.ipynb`。
- 动手：
  - 安装并启动：`python -m sglang.launch_server --model-path <小模型如 Qwen2.5-0.5B>`。
  - 用 OpenAI 客户端发一次 `/v1/chat/completions`。
  - 跑 `examples/runtime/` 下任意离线 Engine 示例。
  - 打开 `http://localhost:30000/health`、`/get_server_info`、`/metrics`（Prometheus）三个端点，看看服务暴露了什么。
- **常见坑**：
  - 显存不足：用小模型 + `--mem-fraction-static 0.7` 降低 KV 预留；或加 `--max-total-tokens` 限制。
  - 端口占用：`--port` 换端口；多卡时注意 `--tp-size` 与可见 GPU 数一致。
  - 首次启动慢：权重下载 + CUDA Graph 捕获耗时正常，看日志 `Capture cuda graph` 进度。
- 自检：能说清 launch_server 启动后起了哪几个进程（结合 2.1）。
- ✅ **通关标准**：能独立启动 server、发请求拿到回复；能用 `ps`/日志指认出 TokenizerManager / Scheduler / Detokenizer 三类进程；能读懂 `/get_server_info` 里的关键字段（模型、并行度、KV 容量）。

### 阶段 1：建立架构心智模型（第 1 周）

**目标**：能在脑中画出"请求 → token"的完整链路与进程划分。

- 阅读（按顺序）：
  1. `srt/entrypoints/engine.py`：Engine 如何拉起 TokenizerManager / Scheduler / Detokenizer。
  2. `srt/entrypoints/http_server.py`：路由与 OpenAI API 适配（`entrypoints/openai/`）。
  3. `srt/managers/tokenizer_manager.py` 的 `generate_request` (`:478`)：请求如何被分词并下发。
  4. `srt/managers/io_struct.py`：进程间传递的数据结构（先看请求/响应 dataclass）。
- 动手：在 `tokenizer_manager.py` 和 `scheduler.py:recv_requests` 加日志，打印一次请求经过的关键节点，串起链路。
- 自检：画一张时序图，标出每一步发生在哪个进程、走的什么通信。
- **自测题**：（1）为什么要拆成三类进程而不是一个？对延迟/吞吐有什么好处？（2）进程间为什么用 ZMQ 而不是直接函数调用？（3）一个 `rid`（请求 id）在哪里生成、如何贯穿全链路？
- **常见坑**：不要把 `sglang.lang`（前端 DSL）与 `sglang.srt`（运行时）混淆，两者同名但职责完全不同。
- ✅ **通关标准**：能脱稿画出「请求 → token → 输出」全链时序图（含进程边界与通信方式）；能在源码里指出请求从 HTTP 到 Scheduler 的每一道转发入口。

### 阶段 2：调度器与连续批处理（第 2 周）★ 重点

**目标**：理解 SGLang 性能的核心——零开销调度与 continuous batching。

- 阅读：
  1. `srt/managers/scheduler.py`：先读 `event_loop_normal` (`:1300`)，再读 `event_loop_overlap` (`:1328`)，对比两者如何重叠 CPU/GPU。
  2. `srt/managers/schedule_policy.py`：组批策略、chunked prefill、优先级。
  3. `srt/managers/schedule_batch.py`：`ScheduleBatch` / `Req` 的生命周期与状态机。
  4. `process_batch_result` (`scheduler.py:2808`)：一次前向后的后处理与请求出队。
- 动手：用一个长 prompt + 多个短请求并发，观察日志中 batch 的组成变化；尝试调 `--chunked-prefill-size` 看吞吐变化。
- **自测题**：（1）`waiting_queue` / `running_batch` / `chunked_req` 三者各自的职责？请求如何在它们之间流转？（2）`get_new_batch_prefill` 与 `PrefillAdder` 如何判定「还能不能再加一个请求」（token 预算/批大小/KV 显存）？（3）什么是 retract（回退）？什么情况会触发？（4）overlap 模式下为什么结果会「延后一拍」，如何处理？
- **常见坑**：`scheduler.py` 已拆分为大量 `*_mixin`/`scheduler_components`，读主类时遇到未定义方法要去对应 mixin 里找；不要对着一个 4000+ 行的文件硬读。
- 自检：解释"为什么 overlap 模式能消除调度开销"，以及 prefill 与 decode 请求如何在同一批/不同批中被调度。
- ✅ **通关标准**：能口述一次 `event_loop_overlap` 迭代里发生了什么（收请求 → 组批 → 前向 → 后处理）；能说清 chunked prefill 与 continuous batching 的关系；能用自己的话说清 CPU/GPU 重叠的原理。

### 阶段 3：内存管理与 RadixAttention（第 2.5 周）★ 重点

**目标**：理解前缀缓存复用与分页 KV 管理。

- 阅读：
  1. `srt/mem_cache/radix_cache.py`：`TreeNode` (`:121`)、`RadixCache` (`:285`)、`match_prefix` (`:374`)、`insert` (`:446`)、以及 LRU 驱逐。
  2. `srt/mem_cache/memory_pool.py` + `allocator.py`：KV cache 物理内存如何分页分配。
  3. `srt/mem_cache/base_prefix_cache.py`：抽象接口（理解可替换的 cache 策略，如 `chunk_cache.py`、`hiradix_cache.py`）。
- 动手：连续发送共享前缀的请求，对比开/关 radix cache（相关 server_args）时的 TTFT；阅读 `examples/monitoring/` 观察缓存命中。
- **自测题**：（1）`match_prefix` 如何在基数树上做最长前缀匹配？节点何时发生分裂（split）？（2）LRU 淘汰与引用计数（lock_ref）如何保证「正在使用的 KV 不被释放」？（3）page （页）粒度与 token 粒度分配的区别？（4）radix tree 的逻辑节点与 `memory_pool` 的物理 KV 如何建立映射？
- **实验建议**：先发请求 A（长 prompt），再发与 A 共享前缀的请求 B，对比两次 TTFT；然后 `curl /flush_cache` 后重发 B，看 TTFT 回升。
- 自检：画出基数树在多请求共享前缀时的结构变化；说明 cache 命中如何减少 prefill 计算。
- ✅ **通关标准**：能手画 radix tree 的 insert/split/match 三种操作；能解释 lock_ref 引用计数与 LRU 淘汰的一致性；能用实验数据说明前缀命中对 TTFT 的影响。

### 阶段 4：模型执行与 CUDA Graph（第 3 周）

**目标**：理解模型如何被加载、前向、用 CUDA Graph 加速。

- 阅读：
  1. `srt/model_executor/model_runner.py`：`ModelRunner` (`:285`)、`load_model` (`:1063`)、`forward` (`:2700`)。
  2. `srt/model_executor/forward_batch_info.py`：`ForwardBatch` / `ForwardMode`（prefill/decode/idle 等模式）。
  3. `srt/model_executor/cuda_graph_runner.py`：捕获与重放，padding 策略。
  4. 选一个模型 `srt/models/llama.py`：看 `forward` 如何串起 attention + MLP + sampler。
- 动手：开关 CUDA Graph（`--disable-cuda-graph`）对比 decode 延迟；在 `forward` 打点统计耗时。
- **自测题**：（1）为什么 CUDA Graph 主要用于 decode 而不是 prefill（形状是否静态）？（2）捕获时的 padding 策略解决了什么问题？（3）`ForwardBatch` 在 prefill/decode/idle 不同模式下张量形状有何不同？
- **常见坑**：`model_runner.py` 中行号引用可能随版本漂移，以类/方法名为准、用符号搜索定位。
- 自检：说清 prefill 与 decode 两种 ForwardMode 在内存访问与 CUDA Graph 适用性上的差异。
- ✅ **通关标准**：能说清 `load_model → forward → sample` 主链；能解释 CUDA Graph 捕获/重放机制与适用场景；能用实验数据说明开关 CUDA Graph 对 decode 延迟的影响。

### 阶段 5：计算层与注意力后端（第 3.5 周）

**目标**：理解可插拔的 attention backend、采样、量化。

- 阅读：
  1. `srt/layers/radix_attention.py`：算子层如何对接 cache。
  2. `srt/layers/attention/base_attn_backend.py` + 选一个实现（`flashinfer_backend.py` 或 `flashattention_backend.py`）。
  3. `srt/layers/sampler.py` + `logits_processor.py`：采样与 logits 处理。
  4. 概览 `srt/layers/quantization/`（先看 `base_config.py` 与 `awq.py`/`fp8`）。
  5. 概览 `srt/layers/moe/`（如涉及 DeepSeek/Mixtral）。
- 参考文档：`docs/advanced_features/attention_backend.md`、`quantization.md`。
- **自测题**：（1）attention backend 是如何可插拔的（基类接口 + 启动参数选择）？（2）采样（temperature/top-p/top-k）在 `sampler.py` 里如何实现？（3）量化 scheme（FP8/AWQ/GPTQ）对 GEMM 与显存各有什么影响？
- 自检：能说出切换 attention backend 的入口与各后端适用场景。
- ✅ **通关标准**：能指出算子层到 attention backend 的调用入口；能说清一次采样的数据流（logits → 处理 → 采样 → token）；能列举至少两种量化 scheme 的适用场景。

### 阶段 6：进阶特性（第 4–5 周，按需选学）

按兴趣/工作需要挑 2–3 个深入，其余了解入口即可：


| 特性           | 入口                                                          | 配套文档                                            |
| -------------- | ------------------------------------------------------------- | --------------------------------------------------- |
| 投机解码 EAGLE | `srt/speculative/eagle_worker.py`                             | `docs/advanced_features/speculative_decoding.ipynb` |
| PD 分离        | `srt/disaggregation/decode.py`                                | `pd_disaggregation.md`                              |
| 张量/专家并行  | `srt/distributed/parallel_state.py`, `layers/dp_attention.py` | `expert_parallelism.md`, `pipeline_parallelism.md`  |
| 结构化输出     | `srt/constrained/`                                            | `structured_outputs.ipynb`                          |
| 多 LoRA        | `srt/lora/`                                                   | `lora.ipynb`                                        |
| 分层 KV 缓存   | `mem_cache/hiradix_cache.py`                                  | `hicache.rst`                                       |
| 可观测性       | `srt/observability/`                                          | `observability.md`                                  |
| RL 集成       | `srt/weight_sync/`, `checkpoint_engine/`                      | `sglang_for_rl.md`                                  |

> **阶段 6 通关标准**：选定的 2–3 个特性，能各自说清「它解决什么问题 + 核心数据流 + 入口代码 + 开启参数」，并能在本地跑通一个最小示例。

#### 深化专题推荐（结合本仓库已有中文文档）

以下三个专题在本仓库已有较深入的中文资料，适合作为阶段 6 的深挖入口：

- **分层 KV 缓存 HiCache（L1/L2/L3）**：先读 `docs/theory/cache/hicache_transfer_zh.md`（五条传输路径 write/load-back/prefetch/backup/evict 与配套 drawio 图），再读 `mem_cache/hiradix_cache.py` 与 `managers/cache_controller.py`；关注「独立 CUDA stream + 后台线程异步搬运」与前向计算的 overlap。自测：能说清一条请求从 L3 预取到回载入 L1 的完整时序与异步重叠点。
- **PD 分离**：先读 `srt/disaggregation/README_zh.md` 与本文 2.2.1 的 PD 时间轴，再读 `disaggregation/prefill.py` / `decode.py`；关注 bootstrap 建链、prealloc 预分配、跨节点 KV 传输（Mooncake/NIXL/MORI）。自测：prefill 节点与 decode 节点各多出哪些阶段？
- **KV 容量与预算**：读 `docs/theory/cache/kv_cache_capacity_zh.md`，理解 `max_total_num_tokens` 如何推算、`PrefillAdder` 的准入预算与 `init_req_max_new_tokens` 的一致性约束。

### 阶段 7：贡献与扩展（第 6 周）

**目标**：能给 SGLang 加东西并通过 CI。

- 阅读：`docs/developer_guide/contribution_guide.md`、`benchmark_and_profiling.md`、`evaluating_new_models.md`。
- 动手（任选其一）：
  - 按 `docs/supported_models/extending/` 给 `srt/models/` 加一个新模型适配。
  - 用 `python/sglang/jit_kernel/` 加一个 JIT kernel（参考 `development_jit_kernel_guide.md`）。
  - 跑通 `test/` 套件：`python test/run_suite.py`（先读 `test/README.md`）。
- 自检：能本地复现一条 CI 测试并提交一个小 PR（修文档/加测试均可）。
- ✅ **通关标准**：能本地跑通至少一条 `test/` 用例；理解 CI 的分层与触发机制（参考 `test/README.md`）；能独立提一个小 PR 并通过本地自检。

---

## 三半、调试与性能分析工具链（贯穿全程）

> 工具不是独立阶段，而是从阶段 2 开始就该随手用起来。「打点优于猜测」是贯穿全程的原则。

| 场景 | 工具 / 入口 | 说明 |
| --- | --- | --- |
| 看实时指标 | `/metrics`（Prometheus）+ `examples/monitoring/`（Grafana 面板） | TTFT/TPOT/吞吐/队列长度/缓存命中率等 |
| 单请求链路 | 开启 tracing（见 `srt/observability/`）看 trace span | 定位单请求在哪个阶段慢（结合 2.2.1 时间轴） |
| 服务内部状态 | `curl /get_server_info`、`/get_internal_state` | 看 KV 容量、当前批、new_token_ratio 等运行时状态 |
| GPU 层 profiling | `docs/developer_guide/benchmark_and_profiling.md` + torch profiler / `examples/profiler/` | 抓 Chrome trace 分析 kernel 耗时与重叠机会 |
| 基准测试 | `python -m sglang.bench_serving` / `bench_one_batch` | 公平对比不同参数/版本的吞吐与延迟 |
| 进程卡死/hang | `py-spy dump`、watchdog 日志、CUDA coredump | 分布式 hang 时定位各 rank 的状态发散点 |
| 源码打点 | 在 `scheduler.py` / `model_runner.py` 关键路径加 `logger.debug` | 最直接的链路串联手段 |

**建议的性能分析四步法**：（1）用 `bench_serving` 定量现状 →（2）看 `/metrics` 定位瓶颈阶段（prefill? decode? 队列?）→（3）抓 profiler trace 看具体 kernel/重叠 →（4）改参数或代码后重新 `bench_serving` 验证。

---

## 四、可选支线：前端 DSL

`python/sglang/lang/` 是 SGLang 同名的**前端编程语言**（区别于 srt 运行时），用于多步骤、并行、受控的 LLM 程序编写。

- 阅读：`lang/ir.py`（中间表示）、`lang/interpreter.py`（执行）、`lang/api.py`（用户 API）。
- 示例：`examples/frontend_language/`。
- 文档：[Frontend Tutorial](https://docs.sglang.io/references/frontend/frontend_tutorial.html)。

> 注：若你的目标是推理性能/部署，可跳过；若做 agent/复杂提示流程，值得一看。

---

## 五、学习方法建议

1. **先跑后读**：每个阶段先复现现象（开关某特性看指标），再回去读实现。
2. **善用 server_args.py**：它是所有功能的"配置真相源"，看某参数的处理逻辑能快速定位相关模块。
3. **打点优于猜测**：在 scheduler / model_runner 关键路径加日志或用 `examples/profiler/` 抓 trace（profiling 见 `docs/developer_guide/benchmark_and_profiling.md`）。
4. **按数据流读代码**：始终顺着"请求 → batch → forward → sample → output"这条线，不要陷入单个文件。
5. **关注 Mixin 模式**：scheduler/tokenizer_manager 用大量 `*_mixin.py` 拆分职责，读主类时按需跳转对应 mixin。
6. **善用 skill 与文档**：本仓库 `.claude/skills/` 下有大量专题 skill（CI、性能、调试 hang、profiling 等），遇到对应场景先查有没有现成 skill；`docs/theory/` 下有中文原理文档可交叉参考。

---

## 五半、学习进度追踪表

> 建议把下表复制到个人笔记，按周打勾。全部「✅ 通关标准」达成才推进；「产出物」是每阶段留下的可复习资产。

| 阶段 | 主题 | 建议周次 | 状态 | 产出物 |
| --- | --- | --- | --- | --- |
| 0 | 跑起来 | 0.5 | ☐ | 启动命令 + `/get_server_info` 关键字段笔记 |
| 1 | 架构心智模型 | 1 | ☐ | 请求全链时序图 |
| 2 | 调度器与连续批处理 ★ | 2 | ☐ | `event_loop_overlap` 一次迭代的流程笔记 |
| 3 | 内存管理与 RadixAttention ★ | 2.5 | ☐ | radix tree insert/split/match 手绘图 + TTFT 对比数据 |
| 4 | 模型执行与 CUDA Graph | 3 | ☐ | CUDA Graph 开关的 decode 延迟对比 |
| 5 | 计算层与注意力后端 | 3.5 | ☐ | 采样数据流 + backend 切换入口笔记 |
| 6 | 进阶特性（选 2–3） | 4–5 | ☐ | 各特性「问题+数据流+入口」小结 + 一个最小示例 |
| 7 | 贡献与扩展 | 6 | ☐ | 跑通一条 CI 用例 + 一个小 PR |

> 里程碑检查点：第 2.5 周末（完成阶段 0–3）应能独立看懂「调度 + 缓存」核心链路；第 5 周末（完成阶段 0–6）应能就任一进阶特性讲清原理与入口；第 6 周末能提交并通过一个 PR。

---

## 六、关键参考资源

- 官方文档：[https://docs.sglang.io/](https://docs.sglang.io/)
- 本仓库文档：`docs/`（中文翻译见 `docs/README_zh.md`、`README_zh.md`）
- 设计博客（强烈推荐按时间线读 v0.2/v0.3/v0.4 与 RadixAttention/大规模 EP 博客）：[https://lmsys.org/blog/](https://lmsys.org/blog/)
- DeepWiki 代码问答：[https://deepwiki.com/sgl-project/sglang](https://deepwiki.com/sgl-project/sglang)
- 学习材料/Slides：[https://github.com/sgl-project/sgl-learning-materials](https://github.com/sgl-project/sgl-learning-materials)
