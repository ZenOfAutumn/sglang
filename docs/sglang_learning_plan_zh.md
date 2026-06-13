# SGLang 工程总结与学习计划

> 本文面向想系统掌握 SGLang 源码与设计的工程师，提供一份"工程全景 + 分阶段学习路线"。
> 所有文件路径均相对于仓库根目录，核心代码位于 `python/sglang/srt`（srt = SGLang RunTime）。

---

## 一、工程总览

### 1.1 这是什么

SGLang 是一个**高性能大模型推理服务框架**（LLM/多模态/扩散模型 serving），目标是在从单卡到大规模分布式集群的各种部署下，提供**低延迟、高吞吐**的推理。它已成为业界事实标准之一，在 40 万+ GPU 上运行，并被大量 RL 后训练框架（verl、slime、AReaL 等）用作 rollout 后端。

### 1.2 核心能力（决定了学习的重点）


| 能力                     | 一句话说明                             | 对应源码区域                                           |
| ------------------------ | -------------------------------------- | ------------------------------------------------------ |
| **RadixAttention**       | 基于基数树的前缀缓存复用 KV cache      | `srt/mem_cache/radix_cache.py`                         |
| **Zero-overhead 调度**   | CPU 调度与 GPU 计算重叠，消除调度开销  | `srt/managers/scheduler.py` (`event_loop_overlap`)     |
| **Continuous batching**  | 连续批处理，动态加入/退出请求          | `srt/managers/schedule_batch.py`, `schedule_policy.py` |
| **Chunked prefill**      | 长 prompt 分块预填充                   | `schedule_policy.py`                                   |
| **Paged attention**      | 分页 KV 内存管理                       | `srt/mem_cache/memory_pool.py`, `allocator.py`         |
| **Speculative decoding** | 投机解码（EAGLE 等）                   | `srt/speculative/`                                     |
| **PD 分离**              | Prefill/Decode 解耦部署                | `srt/disaggregation/`                                  |
| **并行**                 | TP/PP/EP/DP（张量/流水/专家/数据并行） | `srt/distributed/`, `srt/layers/dp_attention.py`       |
| **结构化输出**           | JSON/正则约束解码（压缩 FSM）          | `srt/constrained/`                                     |
| **量化**                 | FP4/FP8/INT4/AWQ/GPTQ                  | `srt/layers/quantization/`                             |
| **多 LoRA**              | 批量 LoRA                              | `srt/lora/`                                            |
| **CUDA Graph**           | 捕获静态图降低 kernel launch 开销      | `srt/model_executor/cuda_graph_runner.py`              |

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

### 2.3 核心模块速查表


| 模块       | 路径                  | 关键文件/类                                                                                      |
| ---------- | --------------------- | ------------------------------------------------------------------------------------------------ |
| 服务入口   | `srt/entrypoints/`    | `http_server.py`, `engine.py`, `EngineBase.py`                                                   |
| 调度管理   | `srt/managers/`       | `scheduler.py`, `tokenizer_manager.py`, `schedule_batch.py`, `schedule_policy.py`                |
| 内存/缓存  | `srt/mem_cache/`      | `radix_cache.py` (`RadixCache:285`, `TreeNode:121`), `memory_pool.py`, `allocator.py`            |
| 模型执行   | `srt/model_executor/` | `model_runner.py` (`ModelRunner:285`), `forward_batch_info.py`, `cuda_graph_runner.py`           |
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

### 阶段 0：跑起来（第 0.5 周）

**目标**：在本地/容器里启动一个 SGLang server 并发请求成功。

- 阅读：`README.md`、`docs/get_started/install.md`、`docs/basic_usage/send_request.ipynb`。
- 动手：
  - 安装并启动：`python -m sglang.launch_server --model-path <小模型如 Qwen2.5-0.5B>`。
  - 用 OpenAI 客户端发一次 `/v1/chat/completions`。
  - 跑 `examples/runtime/` 下任意离线 Engine 示例。
- 自检：能说清 launch_server 启动后起了哪几个进程（结合 2.1）。

### 阶段 1：建立架构心智模型（第 1 周）

**目标**：能在脑中画出"请求 → token"的完整链路与进程划分。

- 阅读（按顺序）：
  1. `srt/entrypoints/engine.py`：Engine 如何拉起 TokenizerManager / Scheduler / Detokenizer。
  2. `srt/entrypoints/http_server.py`：路由与 OpenAI API 适配（`entrypoints/openai/`）。
  3. `srt/managers/tokenizer_manager.py` 的 `generate_request` (`:478`)：请求如何被分词并下发。
  4. `srt/managers/io_struct.py`：进程间传递的数据结构（先看请求/响应 dataclass）。
- 动手：在 `tokenizer_manager.py` 和 `scheduler.py:recv_requests` 加日志，打印一次请求经过的关键节点，串起链路。
- 自检：画一张时序图，标出每一步发生在哪个进程、走的什么通信。

### 阶段 2：调度器与连续批处理（第 2 周）★ 重点

**目标**：理解 SGLang 性能的核心——零开销调度与 continuous batching。

- 阅读：
  1. `srt/managers/scheduler.py`：先读 `event_loop_normal` (`:1300`)，再读 `event_loop_overlap` (`:1328`)，对比两者如何重叠 CPU/GPU。
  2. `srt/managers/schedule_policy.py`：组批策略、chunked prefill、优先级。
  3. `srt/managers/schedule_batch.py`：`ScheduleBatch` / `Req` 的生命周期与状态机。
  4. `process_batch_result` (`scheduler.py:2808`)：一次前向后的后处理与请求出队。
- 动手：用一个长 prompt + 多个短请求并发，观察日志中 batch 的组成变化；尝试调 `--chunked-prefill-size` 看吞吐变化。
- 自检：解释"为什么 overlap 模式能消除调度开销"，以及 prefill 与 decode 请求如何在同一批/不同批中被调度。

### 阶段 3：内存管理与 RadixAttention（第 2.5 周）★ 重点

**目标**：理解前缀缓存复用与分页 KV 管理。

- 阅读：
  1. `srt/mem_cache/radix_cache.py`：`TreeNode` (`:121`)、`RadixCache` (`:285`)、`match_prefix` (`:374`)、`insert` (`:446`)、以及 LRU 驱逐。
  2. `srt/mem_cache/memory_pool.py` + `allocator.py`：KV cache 物理内存如何分页分配。
  3. `srt/mem_cache/base_prefix_cache.py`：抽象接口（理解可替换的 cache 策略，如 `chunk_cache.py`、`hiradix_cache.py`）。
- 动手：连续发送共享前缀的请求，对比开/关 radix cache（相关 server_args）时的 TTFT；阅读 `examples/monitoring/` 观察缓存命中。
- 自检：画出基数树在多请求共享前缀时的结构变化；说明 cache 命中如何减少 prefill 计算。

### 阶段 4：模型执行与 CUDA Graph（第 3 周）

**目标**：理解模型如何被加载、前向、用 CUDA Graph 加速。

- 阅读：
  1. `srt/model_executor/model_runner.py`：`ModelRunner` (`:285`)、`load_model` (`:1063`)、`forward` (`:2700`)。
  2. `srt/model_executor/forward_batch_info.py`：`ForwardBatch` / `ForwardMode`（prefill/decode/idle 等模式）。
  3. `srt/model_executor/cuda_graph_runner.py`：捕获与重放，padding 策略。
  4. 选一个模型 `srt/models/llama.py`：看 `forward` 如何串起 attention + MLP + sampler。
- 动手：开关 CUDA Graph（`--disable-cuda-graph`）对比 decode 延迟；在 `forward` 打点统计耗时。
- 自检：说清 prefill 与 decode 两种 ForwardMode 在内存访问与 CUDA Graph 适用性上的差异。

### 阶段 5：计算层与注意力后端（第 3.5 周）

**目标**：理解可插拔的 attention backend、采样、量化。

- 阅读：
  1. `srt/layers/radix_attention.py`：算子层如何对接 cache。
  2. `srt/layers/attention/base_attn_backend.py` + 选一个实现（`flashinfer_backend.py` 或 `flashattention_backend.py`）。
  3. `srt/layers/sampler.py` + `logits_processor.py`：采样与 logits 处理。
  4. 概览 `srt/layers/quantization/`（先看 `base_config.py` 与 `awq.py`/`fp8`）。
  5. 概览 `srt/layers/moe/`（如涉及 DeepSeek/Mixtral）。
- 参考文档：`docs/advanced_features/attention_backend.md`、`quantization.md`。
- 自检：能说出切换 attention backend 的入口与各后端适用场景。

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
| RL 集成        | `srt/weight_sync/`, `checkpoint_engine/`                      | `sglang_for_rl.md`                                  |

### 阶段 7：贡献与扩展（第 6 周）

**目标**：能给 SGLang 加东西并通过 CI。

- 阅读：`docs/developer_guide/contribution_guide.md`、`benchmark_and_profiling.md`、`evaluating_new_models.md`。
- 动手（任选其一）：
  - 按 `docs/supported_models/extending/` 给 `srt/models/` 加一个新模型适配。
  - 用 `python/sglang/jit_kernel/` 加一个 JIT kernel（参考 `development_jit_kernel_guide.md`）。
  - 跑通 `test/` 套件：`python test/run_suite.py`（先读 `test/README.md`）。
- 自检：能本地复现一条 CI 测试并提交一个小 PR（修文档/加测试均可）。

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

---

## 六、关键参考资源

- 官方文档：[https://docs.sglang.io/](https://docs.sglang.io/)
- 本仓库文档：`docs/`（中文翻译见 `docs/README_zh.md`、`README_zh.md`）
- 设计博客（强烈推荐按时间线读 v0.2/v0.3/v0.4 与 RadixAttention/大规模 EP 博客）：[https://lmsys.org/blog/](https://lmsys.org/blog/)
- DeepWiki 代码问答：[https://deepwiki.com/sgl-project/sglang](https://deepwiki.com/sgl-project/sglang)
- 学习材料/Slides：[https://github.com/sgl-project/sgl-learning-materials](https://github.com/sgl-project/sgl-learning-materials)
