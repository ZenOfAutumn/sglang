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
  - `ForwardMode`(forward_batch_info.py:83):区分 `EXTEND`/`DECODE`/`IDLE`/`TARGET_VERIFY` 等前向模式,这是后续所有分支逻辑的总开关。
  - `ForwardBatch`(:304):逐字段理解它承载的张量元数据(input_ids、positions、seq_lens、KV 索引、attention 元信息等)。
  - `CaptureHiddenMode`(:220)、`PPProxyTensors`(:1430):理解隐藏态捕获与流水线并行的代理张量。
- 自测问题:`ModelWorkerBatch`(在 `managers/schedule_batch.py`)和 `ForwardBatch` 的边界在哪?谁负责把前者转成后者?

### 阶段二:执行主线 —— ModelRunner 的生命周期(1~2 天)

`model_runner.py` 是本目录的核心(3000+ 行),不要逐行读,按"初始化 → 加载 → 前向"三条主线抓主干。

- 初始化链路(按调用顺序):
  - `__init__`(model_runner.py:395):保存 server_args、设备/并行 rank 等基础字段,初始化各类占位属性,随后调用 `initialize`。
  - `initialize`(:741):核心初始化流程。依次完成内存节省器创建、专家位置/分布记录初始化、加载模型、计算有效层范围、应用量化/张量并行/LoRA,并推导 KV cache dtype。
  - `init_torch_distributed`(:1294):初始化 torch 分布式环境。绑定设备、选择通信后端、设置 all-reduce 策略、初始化各并行组(TP/PP/EP/DP)、预热 NCCL/RCCL,并返回模型加载前的可用显存。
  - `load_model`(:1504):加载模型权重。包含设备能力检查与 dtype 回退、准备模型配置、调用对应 loader 加载权重。
  - `init_backends`(:992):初始化注意力后端并捕获 CUDA Graph 的**统一入口**,按设备(cuda/cpu/npu/其他)分支,内部依次调用下面的 `init_attention_backend` 与 `init_decode_cuda_graph`/`init_prefill_cuda_graph`。
  - `init_attention_backend`(:2627):初始化注意力核后端本身,按是否启用 PDMux / 双 batch overlap 走不同分支。
- 前向入口(最终目标),对照阶段一的 `ForwardMode` 看分发逻辑:
  - `forward_decode`(:3563):执行一次解码(decode)前向的 eager 路径(未命中 CUDA Graph 时)。
  - `forward_extend`(:3619):执行一次 EXTEND(预填/拓展)前向,返回 `(输出, 是否命中分段 CUDA Graph)`。
  - `forward_idle`(:3725):执行空闲(IDLE)前向;DP attention 下用于 MLP 同步的(可能被 padding 的)空批次。
- `_dummy_run`(:2900):运行一次虚拟(dummy)前向,用于预热/profiling,可通过 `forward_mode_override` 强制 EXTEND/DECODE 模式。
- 暂时跳过:权重热更新(`update_weights_from_*`)、LoRA、各类模型特化 config,用到再回看。
- 自测问题:`TpModelWorker`(`managers/tp_worker.py`)如何调用 `ModelRunner` 的 forward?(连接上一轮关于 worker 的认知)

### 阶段三:显存与 KV cache(1 天)

- 精读 `model_runner_kv_cache_mixin.py`:`ModelRunnerKVCacheMixin` 与 `MemoryPoolConfig`,理解 MHA/MLA/双稀疏/混合线性/FP4 等不同 KV cache 池的创建与差异。
- `configure_kv_cache_dtype`(model_runner.py:2567):根据 `--kv-cache-dtype` 与模型量化配置确定 KV cache 的实际数据类型(auto/fp8_e5m2/fp8_e4m3/bf16/fp4_e2m1 等),HIP 与非 HIP 平台取用不同的 fp8 表示。
- `max_token_pool_size`(:2525):返回考虑了混合 SWA 设置后的最大 token 池大小,即显存预算最终决定可容纳的 token 数。
- 自测问题:为什么 MLA 模型的 KV cache 布局和标准 MHA 不同?

### 阶段四:性能优化 —— 图捕获与重放(1~2 天)

这是 SGLang 高吞吐的关键,读懂"为什么 decode 阶段适合 CUDA Graph"。

- `cuda_graph_runner.py`:`CudaGraphRunner` 与 `DecodeInputBuffers`,decode 阶段的捕获/重放。先理解"固定 shape 才能 replay"这一约束。
- `piecewise_cuda_graph_runner.py`:`PiecewiseCudaGraphRunner`,配合 torch.compile 处理变长 prefill。
- `input_buffers.py`:`ForwardInputBuffers`,跨批次复用的输入缓冲池(图捕获依赖固定地址)。
- `init_decode_cuda_graph`(model_runner.py:3200):decode 阶段设备图(CUDA/CPU/NPU graph)的捕获入口,仅对生成类模型生效。(旧名 `init_device_graphs`)
- `init_prefill_cuda_graph`(:3262):prefill 阶段分段(piecewise)CUDA Graph runner 的初始化入口,从模型中收集注意力层/MoE 层/indexer 后在满足条件时捕获,多种不支持场景下提前返回。(旧名 `init_piecewise_cuda_graphs`)
- 两者均由阶段二的 `init_backends`(:992) 统一触发。
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
