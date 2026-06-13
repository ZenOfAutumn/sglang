# 环境变量

SGLang 支持各种环境变量,可用于配置其运行时行为。本文档提供了一份全面的列表,并力求随时间保持更新。

*注意:SGLang 为环境变量使用了两个前缀:`SGL_` 和 `SGLANG_`。这可能是由于历史原因造成的。虽然目前两者都被用于不同的设置,但未来版本可能会将它们合并。*

## 通用配置

| 环境变量                      | 描述                                                                                                                      | 默认值                |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| `SGLANG_USE_MODELSCOPE`                   | 启用使用来自 ModelScope 的模型                                                                                              | `false`                      |
| `SGLANG_HOST_IP`                          | 服务器的主机 IP 地址                                                                                                   | `0.0.0.0`                    |
| `SGLANG_PORT`                             | 服务器的端口                                                                                                              | auto-detected                |
| `SGLANG_LOGGING_CONFIG_PATH`              | 自定义日志配置路径                                                                                                | Not set                      |
| `SGLANG_DISABLE_REQUEST_LOGGING`          | 禁用请求日志记录                                                                                                          | `false`                      |
| `SGLANG_LOG_REQUEST_HEADERS`              | 当启用 `--log-requests` 时要记录的额外 HTTP 头的逗号分隔列表。会追加到默认的 `x-smg-routing-key`。 | Not set                      |
| `SGLANG_HEALTH_CHECK_TIMEOUT`             | 健康检查的超时时间(以秒为单位)                                                                                              | `20`                         |
| `SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL` | 收集每层和每个 GPU rank 上所选物理专家计数指标的 pass 间隔。0 表示禁用。 | `0`                          |
| `SGLANG_FORWARD_UNKNOWN_TOOLS`            | 将未知的工具调用转发给客户端,而不是丢弃它们                                                                   | `false` (drop unknown tools) |
| `SGLANG_REQ_WAITING_TIMEOUT`              | 请求在队列中等待被调度前的超时时间(以秒为单位)                                                    | `-1`                         |
| `SGLANG_REQ_RUNNING_TIMEOUT`              | 请求在 decode batch 中运行的超时时间(以秒为单位)                                                    | `-1`                         |

## 性能调优

| 环境变量 | 描述 | 默认值 |
| --- | --- | --- |
| `SGLANG_ENABLE_TORCH_INFERENCE_MODE` | 控制是否使用 torch.inference_mode | `false` |
| `SGLANG_ENABLE_TORCH_COMPILE` | 启用 torch.compile | `false` |
| `SGLANG_SET_CPU_AFFINITY` | 启用 CPU 亲和性设置(在 Docker 构建中通常设为 `1`) | `false` |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | 允许调度器覆盖更长上下文长度的请求(在 Docker 构建中通常设为 `1`) | `false` |
| `SGLANG_IS_FLASHINFER_AVAILABLE` | 控制 FlashInfer 可用性检查 | `true` |
| `SGLANG_SKIP_P2P_CHECK` | 跳过 P2P(peer-to-peer)访问检查 | `false` |
| `SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD` | 设置启用分块前缀缓存的阈值 | `8192` |
| `SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION` | 在 Fused Multi-Layer Attention 中启用 RoPE 融合 | `1` |
| `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` | 为连续的 prefill batch 禁用 overlap 调度 | `false` |
| `SGLANG_SCHEDULER_MAX_RECV_PER_POLL` | 设置每次 poll 的最大请求数,负值表示无限制 | `-1` |
| `SGLANG_DISABLE_FA4_WARMUP` | 禁用 Flash Attention 4 预热 pass(设为 `1`、`true`、`yes` 或 `on` 以禁用) | `false` |
| `SGLANG_DATA_PARALLEL_BUDGET_INTERVAL` | DPBudget 更新的间隔 | `1` |
| `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT` | 调度器 recv skipper 计数器的默认权重值(当 forward mode 不匹配特定模式时使用)。仅在 `--scheduler-recv-interval > 1` 时生效。计数器累积权重,并在达到间隔阈值时触发请求轮询。 | `1000` |
| `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE` | 调度器 recv skipper 中 decode forward mode 的权重增量。与 `--scheduler-recv-interval` 配合使用,以控制 decode 阶段的轮询频率。 | `1` |
| `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY` | 调度器 recv skipper 中 target verify forward mode 的权重增量。与 `--scheduler-recv-interval` 配合使用,以控制验证阶段的轮询频率。 | `1` |
| `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE` | 调度器 recv skipper 中当 forward mode 为 None 时的权重增量。与 `--scheduler-recv-interval` 配合使用,以控制无特定 forward mode 处于活动状态时的轮询频率。 | `1` |
| `SGLANG_MM_BUFFER_SIZE_MB` | 用于多模态特征哈希优化的预分配 GPU 缓冲区大小(以 MB 为单位)。当设为正值时,临时将特征移至 GPU 以加快哈希计算,然后将其移回 CPU 以节省 GPU 内存。较大的特征从 GPU 哈希中受益更多。设为 `0` 以禁用。 | `0` |
| `SGLANG_MM_PRECOMPUTE_HASH` | 启用对 MultimodalDataItem 哈希值的预计算 | `false` |
| `SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH` | 在 overlap 调度器下准备 mlp sync batch 时启用 NCCL 进行 gather(不设此标志则使用 gloo 进行 gather) | `false` |
| `SGLANG_SYMM_MEM_PREALLOC_GB_SIZE` | 用于 NCCL 对称内存池的预分配 GPU 缓冲区大小(以 GB 为单位),用于限制内存碎片。仅在设置了服务器参数 `--enable-symm-mem` 时生效。 | `-1` |
| `SGLANG_CUSTOM_ALLREDUCE_ALGO` | 自定义 all-reduce 的算法。设为 `oneshot` 或 `1stage` 以强制使用 one-shot。设为 `twoshot` 或 `2stage` 以强制使用 two-shot。 | `` |
| `SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR` | flashinfer 中 TRT-LLM prefill attention 的 skip-softmax 阈值缩放因子。`None` 表示标准 attention。参见 https://arxiv.org/abs/2512.12087 | `None` |
| `SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR` | flashinfer 中 TRT-LLM decode attention 的 skip-softmax 阈值缩放因子。`None` 表示标准 attention。参见 https://arxiv.org/abs/2512.12087 | `None` |


## DeepGEMM 配置(高级优化)

| 环境变量 | 描述 | 默认值 |
| --- | --- | --- |
| `SGLANG_ENABLE_JIT_DEEPGEMM` | 启用 DeepGEMM 内核的即时(Just-In-Time)编译(在安装了 DeepGEMM 包的 NVIDIA Hopper (SM90) 和 Blackwell (SM100) GPU 上默认启用;设为 `"0"` 以禁用) | `"true"` |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE` | 启用 DeepGEMM 内核的预编译 | `"true"` |
| `SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS` | 用于并行 DeepGEMM 内核编译的 worker 数量 | `4` |
| `SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE` | 在 DeepGEMM 预编译脚本期间使用的指示标志 | `"false"` |
| `SGLANG_DG_CACHE_DIR` | 用于缓存已编译 DeepGEMM 内核的目录 | `~/.cache/deep_gemm` |
| `SGLANG_DG_USE_NVRTC` | 使用 NVRTC(而非 Triton)进行 JIT 编译(实验性) | `"false"` |
| `SGLANG_USE_DEEPGEMM_BMM` | 使用 DeepGEMM 进行批量矩阵乘法(BMM)操作 | `"false"` |
| `SGLANG_JIT_DEEPGEMM_FAST_WARMUP` | 在预热期间预编译更少的内核,将预热时间从 30 分钟减少到不到 3 分钟。可能会在运行时导致性能下降。 | `"false"` |

## DeepEP 配置

| 环境变量 | 描述 | 默认值 |
| --- | --- | --- |
| `SGLANG_DEEPEP_BF16_DISPATCH` | 使用 Bfloat16 进行 dispatch | `"false"` |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 每个 GPU 上 dispatch 的最大 token 数 | `"128"` |
| `SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 当 --moe-a2a-backend=flashinfer 时每个 GPU 上 dispatch 的最大 token 数 | `"1024"` |
| `SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS` | 启用 single batch overlap 时用于 DeepEP combine 的 SM 数量 | `"32"` |
| `SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO` | 在 GB200 上启用 single batch overlap 时,在备用流上运行 shared experts。当不设此标志时,shared experts 和 down gemm 将与 DeepEP combine 一起 overlap。 | `"false"` |

## MORI 配置

| 环境变量 | 描述 | 默认值 |
| --- | --- | --- |
| `SGLANG_MORI_DISPATCH_DTYPE` | 覆盖 MoRI-EP dispatch 量化类型。`auto` 使用从权重 dtype 自动检测;`bf16`/`fp8`/`fp4` 为所有层强制指定的类型 | `"auto"` |
| `SGLANG_MORI_FP8_COMB` | 使用 FP8 进行 combine | `"false"` |
| `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 用于 MORI-EP 缓冲区分配的每个 rank 的最大 dispatch token 数 | `4096` |
| `SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD` | 在 `InterNodeV1` 和 `InterNodeV1LL` 内核类型之间切换的阈值。如果 `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 小于或等于此阈值,则使用 `InterNodeV1LL`;否则使用 `InterNodeV1`。 | `256` |
| `SGLANG_MORI_QP_PER_TRANSFER` | 每次传输操作使用的 RDMA Queue Pair (QP) 数量 | `1` |
| `SGLANG_MORI_POST_BATCH_SIZE` | 在单个批次中向每个 QP 提交的 RDMA work request 数量 | `-1` |
| `SGLANG_MORI_NUM_WORKERS` | RDMA executor 线程池中的 worker 线程数量 | `1` |

## NSA 后端配置(用于 DeepSeek V3.2)

<!-- # Environment variable to control mtp precomputing of metadata for multi-step speculative decoding -->

| 环境变量 | 描述 | 默认值 |
| --- | --- | --- |
| `SGLANG_NSA_FUSE_TOPK` | 融合从 page table 中选取 topk logits 和选取 topk indices 的操作 | `true` |
| `SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA` | 当启用 MTP 时,预计算可在不同 draft step 之间共享的 metadata | `true` |
| `SGLANG_USE_FUSED_METADATA_COPY` | 控制是否为 cuda graph replay 使用 fused metadata copy 内核 | `true` |
| `SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` | 当当前 prefill batch 中的最大 kv len 超过此值时,将应用 sparse mla 内核,否则回退到 dense MHA 实现。默认为模型的 index topk(DeepSeek V3.2 为 2048) | `2048` |


## 内存管理

| 环境变量 | 描述 | 默认值 |
| --- | --- | --- |
| `SGLANG_DEBUG_MEMORY_POOL` | 启用内存池调试 | `false` |
| `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` | 为内存规划裁剪 max new tokens 估算值 | `4096` |
| `SGLANG_DETOKENIZER_MAX_STATES` | detokenizer 的最大状态数 | Default value based on system |
| `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK` | 启用对 Tensor Parallel rank 之间内存不平衡的检查 | `true` |
| `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` | 为 Mooncake 配置自定义内存池类型。支持 `NVLINK`、`BAREX`、`INTRA_NODE_NVLINK`。如果设为 `true`,则默认为 `NVLINK`。 | `None` |

## 模型特定选项

| 环境变量 | 描述 | 默认值 |
| --- | --- | --- |
| `SGLANG_USE_AITER` | 使用 AITER 优化实现 | `false` |
| `SGLANG_MOE_PADDING` | 启用 MoE padding(如果值为 `1` 则将 padding 大小设为 128,在 Docker 构建中通常设为 `1`) | `false` |
| `SGLANG_CUTLASS_MOE` (已弃用) | 在 Blackwell GPU 上使用 Cutlass FP8 MoE 内核(已弃用,请使用 --moe-runner-backend=cutlass) | `false` |

## 量化

| 环境变量 | 描述 | 默认值 |
| --- | --- | --- |
| `SGLANG_INT4_WEIGHT` | 启用 INT4 权重量化 | `false` |
| `SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2` | 应用带有融合 silu 和 mul 以及 masked m 的 per token group 量化内核 | `false` |
| `SGLANG_FORCE_FP8_MARLIN` | 即使有其他 FP8 内核可用,也强制使用 FP8 MARLIN 内核 | `false` |
| `SGLANG_FORCE_NVFP4_MARLIN` | 即使在具有原生 FP4 支持的 Blackwell GPU 上,也强制使用 NVFP4 Marlin 回退内核 | `false` |
| `SGLANG_FLASHINFER_FP4_GEMM_BACKEND` (已弃用) | 在 Blackwell GPU 上为 `mm_fp4` 选择后端。**已弃用**:请改用 `--fp4-gemm-backend`。 | `` |
| `SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN` | 在启动 DeepSeek NVFP4 checkpoint 时将 q_b_proj 从 BF16 量化为 FP8 | `false` |
| `SGLANG_MOE_NVFP4_DISPATCH` | 为 moe dispatch 使用 nvfp4(在 flashinfer_cutlass 或 flashinfer_cutedsl moe runner 后端上) | `"false"` |
| `SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE` | 在启动 DeepSeek NVFP4 checkpoint 时将 nextn 层的 moe 从 BF16 量化为 FP8 | `false` |
| `SGLANG_QUANT_ALLOW_DOWNCASTING` | 允许在加载期间进行权重 dtype 降级(例如 fp32 → fp16)。默认情况下,使用量化时 SGLang 拒绝此类降级。 | `false` |
| `SGLANG_FP8_IGNORED_LAYERS` | 在 FP8 量化期间要忽略的层名称的逗号分隔列表。例如:`model.layers.0,model.layers.1.,qkv_proj`。 | `""` |


## 分布式计算

| 环境变量 | 描述 | 默认值 |
| --- | --- | --- |
| `SGLANG_BLOCK_NONZERO_RANK_CHILDREN` | 控制对非零 rank 子进程的阻塞 | `1` |
| `SGLANG_IS_FIRST_RANK_ON_NODE` | 指示当前进程是否是其所在节点上的第一个 rank | `"true"` |
| `SGLANG_PP_LAYER_PARTITION` | Pipeline parallel 层划分规范 | Not set |
| `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS` | 为分布式计算设置每个进程一个可见设备 | `false` |

## 测试与调试(内部/CI)

*这些变量主要用于内部测试、持续集成或调试。*

| 环境变量 | 描述 | 默认值 |
| --- | --- | --- |
| `SGLANG_IS_IN_CI` | 指示是否在 CI 环境中运行 | `false` |
| `SGLANG_IS_IN_CI_AMD` | 指示是否在 AMD CI 环境中运行 | `false` |
| `SGLANG_TEST_RETRACT` | 启用 retract decode 测试 | `false` |
| `SGLANG_TEST_RETRACT_NO_PREFILL_BS` | 当启用 SGLANG_TEST_RETRACT 时,如果 batch size 超过 SGLANG_TEST_RETRACT_NO_PREFILL_BS,则不执行 prefill。 | `2 ** 31`     |
| `SGLANG_RECORD_STEP_TIME` | 记录 step 时间以用于性能分析 | `false` |
| `SGLANG_TEST_REQUEST_TIME_STATS` | 测试请求时间统计 | `false` |
| `SGLANG_KERNEL_API_LOGLEVEL` | 控制 crash-debug 内核 API 日志记录。`0` 禁用日志记录,`1` 记录 API 名称,`3` 记录 tensor metadata,`5` 添加 tensor 统计信息,`10` 还会写入调用前的转储快照。 | `0` |
| `SGLANG_KERNEL_API_LOGDEST` | crash-debug 内核 API 日志的目标。使用 `stdout`、`stderr` 或文件路径。`%i` 会被替换为进程 PID。 | `stdout` |
| `SGLANG_KERNEL_API_DUMP_DIR` | level-10 内核 API 输入/输出转储的输出目录。`%i` 会被替换为进程 PID。 | `sglang_kernel_api_dumps` |
| `SGLANG_KERNEL_API_DUMP_INCLUDE` | 要包含在 level-10 转储中的内核 API 名称的逗号分隔通配符模式。 | Not set |
| `SGLANG_KERNEL_API_DUMP_EXCLUDE` | 要从 level-10 转储中排除的内核 API 名称的逗号分隔通配符模式。 | Not set |

## 性能分析与基准测试

| 环境变量 | 描述 | 默认值 |
| --- | --- | --- |
| `SGLANG_TORCH_PROFILER_DIR` | PyTorch profiler 输出的目录 | `/tmp` |
| `SGLANG_PROFILE_WITH_STACK` | 为 PyTorch profiler 设置 `with_stack` 选项(bool)(捕获堆栈跟踪) | `true` |
| `SGLANG_PROFILE_RECORD_SHAPES` | 为 PyTorch profiler 设置 `record_shapes` 选项(bool)(记录 shapes) | `true` |
| `SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS` | 如果启用了 tracing,配置 BatchSpanProcessor.schedule_delay_millis | `500` |
| `SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE` | 如果启用了 tracing,配置 BatchSpanProcessor.max_export_batch_size | `64` |

## 存储与缓存

| 环境变量 | 描述 | 默认值 |
| --- | --- | --- |
| `SGLANG_WAIT_WEIGHTS_READY_TIMEOUT` | 等待权重的超时时间 | `120` |
| `SGLANG_DISABLE_OUTLINES_DISK_CACHE` | 禁用 Outlines 磁盘缓存 | `false` |
| `SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE` | 使用 SGLang 自定义的 Triton 内核缓存实现以降低开销(在 CUDA 上自动启用) | `false` |
| `SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE` | Decode 侧的增量 KV cache offload 步长。向下取整为 `--page-size` 的倍数(最小为 `--page-size`)。如果未设置/无效/<=0,则回退到 `--page-size`。 | Not set (uses `--page-size`) |


## 函数调用 / 工具使用

| 环境变量 | 描述 | 默认值 |
| --- | --- | --- |
| `SGLANG_TOOL_STRICT_LEVEL` | 控制工具调用解析和验证的严格级别。<br>**Level 0**:关闭 - 无严格验证 <br>**Level 1**:函数严格 - 为所有工具启用结构标签约束(即使没有任何工具设置了 `strict=True`) <br>**Level 2**:参数严格 - 为所有工具强制执行严格的参数验证,将它们视为全部设置了 `strict=True` | `0` |
