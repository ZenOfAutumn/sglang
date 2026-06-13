# srt/model_executor

## 目录用途
该目录是 SGLang 的模型执行层，承载从调度批次到 GPU 前向计算的核心运行时。`ModelRunner` 负责模型加载、KV cache 初始化与前向调度；`ForwardBatch` 承载单次前向的底层张量元数据；各类 Graph Runner(CUDA Graph、分段 CUDA Graph、CPU torch.compile)用于捕获并重放计算图以降低开销。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `model_runner.py` | `ModelRunner`(继承 KV cache mixin):模型与权重加载、显存/KV cache 初始化、注意力后端选择、前向执行入口；含 `ModelRunnerOutput`、`LocalSerializedTensor`。 |
| `model_runner_kv_cache_mixin.py` | `ModelRunnerKVCacheMixin` 与 `MemoryPoolConfig`：为 ModelRunner 提供各类 KV cache 显存池(MHA/MLA/双稀疏/混合线性/FP4 等)的创建与配置。 |
| `forward_batch_info.py` | `ForwardBatch` 及 `ForwardMode`、`CaptureHiddenMode`、`PPProxyTensors`、位置计算等:前向批次的底层张量数据与元信息。 |
| `forward_batch_deepseek_mha_mixin.py` | `ForwardBatchDeepSeekMHAMixin`:DeepSeek MLA 分块前缀缓存(chunked prefill)的元数据管理与 KV 索引构建。 |
| `cuda_graph_runner.py` | `CudaGraphRunner` 及 `DecodeInputBuffers`、捕获模式工具:解码阶段的 CUDA Graph 捕获与重放。 |
| `piecewise_cuda_graph_runner.py` | `PiecewiseCudaGraphRunner` 及 `PrefillInputBuffers`:分段(piecewise)CUDA Graph 运行器,配合 torch.compile。 |
| `cpu_graph_runner.py` | `CPUGraphRunner`:基于 CPU torch.compile 的图运行器,接口对齐 CudaGraphRunner。 |
| `input_buffers.py` | `ForwardInputBuffers`:前向输入张量缓冲池,支持跨批次共享与复用。 |
| `hook_manager.py` | 前向 hook 管理:`register_forward_hooks` 按 server_args 配置为匹配模块挂载 forward hook,`resolve_callable` 解析工厂路径。 |
| `mindspore_runner.py` | MindSpore 分布式模块启动:HCCL 通信、并行环境设置与调度进程初始化。 |

## 学习计划

> 目标:理解一次前向请求如何从调度批次变成 GPU 上的张量计算,再到为降低开销而做的图捕获优化。建议按下面五个阶段顺序学习,每个阶段先读"为什么",再带着问题读代码。

### 阶段一:数据载体 —— 先搞清楚"传进来的是什么"(约 0.5 天)

一切执行都围绕 `ForwardBatch` 展开,必须先读它。

- 精读 `forward_batch_info.py`
  - `ForwardMode`(forward_batch_info.py:81):区分 `EXTEND`/`DECODE`/`IDLE`/`TARGET_VERIFY` 等前向模式,这是后续所有分支逻辑的总开关。
  - `ForwardBatch`:逐字段理解它承载的张量元数据(input_ids、positions、seq_lens、KV 索引、attention 元信息等)。
  - `CaptureHiddenMode`(forward_batch_info.py:196)、`PPProxyTensors`:理解隐藏态捕获与流水线并行的代理张量。
- 自测问题:`ModelWorkerBatch`(在 `managers/schedule_batch.py`)和 `ForwardBatch` 的边界在哪?谁负责把前者转成后者?

### 阶段二:执行主线 —— ModelRunner 的生命周期(1~2 天)

`model_runner.py` 是本目录的核心(3000 行),不要逐行读,按"初始化 → 加载 → 前向"三条主线抓主干。

- 初始化链路:`__init__`(model_runner.py:288)→ `initialize`(:460)→ `init_torch_distributed`(:881)→ `load_model`(:1063)→ `init_attention_backend`(:1918)。
- 前向入口(最终目标):`forward_decode`(:2597)、`forward_extend`(:2620)、`forward_idle`(:2660)。对照阶段一的 `ForwardMode` 看分发逻辑。
- `_dummy_run`(:2087):理解预热/捕获时如何构造假批次。
- 暂时跳过:权重热更新(`update_weights_from_*`)、LoRA、各类模型特化 config,用到再回看。
- 自测问题:`TpModelWorker`(`managers/tp_worker.py`)如何调用 `ModelRunner` 的 forward?(连接上一轮关于 worker 的认知)

### 阶段三:显存与 KV cache(1 天)

- 精读 `model_runner_kv_cache_mixin.py`:`ModelRunnerKVCacheMixin` 与 `MemoryPoolConfig`,理解 MHA/MLA/双稀疏/混合线性/FP4 等不同 KV cache 池的创建与差异。
- `configure_kv_cache_dtype`(model_runner.py:1861)、`max_token_pool_size`(:1838):显存预算如何决定可容纳的 token 数。
- 自测问题:为什么 MLA 模型的 KV cache 布局和标准 MHA 不同?

### 阶段四:性能优化 —— 图捕获与重放(1~2 天)

这是 SGLang 高吞吐的关键,读懂"为什么 decode 阶段适合 CUDA Graph"。

- `cuda_graph_runner.py`:`CudaGraphRunner` 与 `DecodeInputBuffers`,decode 阶段的捕获/重放。先理解"固定 shape 才能 replay"这一约束。
- `piecewise_cuda_graph_runner.py`:`PiecewiseCudaGraphRunner`,配合 torch.compile 处理变长 prefill。
- `input_buffers.py`:`ForwardInputBuffers`,跨批次复用的输入缓冲池(图捕获依赖固定地址)。
- `init_device_graphs`(model_runner.py:2393)、`init_piecewise_cuda_graphs`(:2439):捕获的触发入口。
- `cpu_graph_runner.py`:CPU torch.compile 版本,接口对齐 CUDA Graph,可对照理解抽象边界。

### 阶段五:特化与扩展(按需,0.5 天)

- `forward_batch_deepseek_mha_mixin.py`:DeepSeek MLA chunked prefill 的元数据与 KV 索引构建。
- `hook_manager.py`:按 server_args 给模块挂 forward hook 的机制(调试/观测用)。
- `mindspore_runner.py`:MindSpore 后端启动,与主流程解耦,了解即可。

### 推荐的整体阅读顺序(一句话版)

`ForwardBatch`(数据)→ `ModelRunner` 前向入口(主线)→ KV cache mixin(显存)→ Graph Runner(优化)→ 特化 mixin(扩展)。

### 贯穿全程的核心问题

1. 一条请求的张量从 `ModelWorkerBatch` 到 GPU kernel,中间经过哪些转换?
2. `EXTEND`(prefill)与 `DECODE` 在显存、attention、图捕获上的处理差异是什么?
3. CUDA Graph 为什么只在 decode 默认启用?prefill 用 piecewise graph 解决了什么问题?
