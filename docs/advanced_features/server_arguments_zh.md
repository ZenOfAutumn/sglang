# Server Arguments

本页提供了一份服务器参数列表,这些参数用于在命令行中配置语言模型服务器在部署期间的
行为和性能。借助这些参数,用户可以自定义服务器的关键方面,
包括模型选择、并行策略、
内存管理和优化技术。
你可以通过 `python3 -m sglang.launch_server --help` 找到所有参数。

## Common launch commands

- 要使用配置文件,创建一个包含你的服务器参数的 YAML 文件,并通过 `--config` 指定它。CLI 参数将覆盖配置文件中的值。

  ```bash
  # Create config.yaml
  cat > config.yaml << EOF
  model-path: meta-llama/Meta-Llama-3-8B-Instruct
  host: 0.0.0.0
  port: 30000
  tensor-parallel-size: 2
  enable-metrics: true
  log-requests: true
  EOF

  # Launch server with config file
  python -m sglang.launch_server --config config.yaml
  ```

- 要启用多 GPU tensor parallelism,添加 `--tp 2`。如果报错 "peer access is not supported between these two devices",请在服务器启动命令中添加 `--enable-p2p-check`。

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 2
  ```

- 要启用多 GPU data parallelism,添加 `--dp 2`。如果有足够的内存,data parallelism 对 throughput 更有利。它也可以与 tensor parallelism 一起使用。下面的命令总共使用 4 个 GPU。我们推荐使用 [SGLang Model Gateway (former Router)](../advanced_features/sgl_model_gateway.md) 来实现 data parallelism。

  ```bash
  python -m sglang_router.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --dp 2 --tp 2
  ```

- 如果你在服务期间遇到 out-of-memory 错误,尝试通过设置更小的 `--mem-fraction-static` 值来减少 KV cache 池的内存使用。默认值为 `0.9`。

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --mem-fraction-static 0.7
  ```

- 关于调优超参数以获得更好性能,请参见 [hyperparameter tuning](hyperparameter_tuning.md)。
- 对于 docker 和 Kubernetes 运行,你需要设置共享内存,它用于进程间通信。对于 docker 请参见 `--shm-size`,对于 Kubernetes manifest 请更新 `/dev/shm` 大小。
- 如果你在处理长 prompt 的 prefill 期间遇到 out-of-memory 错误,尝试设置更小的 chunked prefill 大小。

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --chunked-prefill-size 4096
  ```
- 要启用 fp8 权重 quantization,在 fp16 checkpoint 上添加 `--quantization fp8`,或者直接加载 fp8 checkpoint 而不指定任何参数。
- 要启用 fp8 kv cache quantization,添加 `--kv-cache-dtype fp8_e4m3` 或 `--kv-cache-dtype fp8_e5m2`。
- 要启用确定性推理和 batch invariant 操作,添加 `--enable-deterministic-inference`。更多细节可参见 [deterministic inference document](../advanced_features/deterministic_inference.md)。
- 如果模型在 Hugging Face tokenizer 中没有 chat template,你可以指定一个 [custom chat template](../references/custom_chat_template.md)。如果 tokenizer 有多个命名模板(例如 'default'、'tool_use'),你可以使用 `--hf-chat-template-name tool_use` 来选择其中一个。
- 要在多个节点上运行 tensor parallelism,添加 `--nnodes 2`。如果你有两个节点,每个节点上有两个 GPU,想要运行 TP=4,设 `sgl-dev-0` 为第一个节点的主机名,`50000` 为一个可用端口,你可以使用以下命令。如果遇到死锁,请尝试添加 `--disable-cuda-graph`
- (注意:此功能已停止维护,可能导致错误)要启用 `torch.compile` 加速,添加 `--enable-torch-compile`。它能加速小批量上的小模型。默认情况下,缓存路径位于 `/tmp/torchinductor_root`,你可以通过环境变量 `TORCHINDUCTOR_CACHE_DIR` 自定义它。更多细节请参考 [PyTorch official documentation](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html) 和 [Enabling cache for torch.compile](https://docs.sglang.io/references/torch_compile_cache.html)。
  ```bash
  # Node 0
  python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --tp 4 \
    --dist-init-addr sgl-dev-0:50000 \
    --nnodes 2 \
    --node-rank 0

  # Node 1
  python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --tp 4 \
    --dist-init-addr sgl-dev-0:50000 \
    --nnodes 2 \
    --node-rank 1
  ```

请查阅下方的文档以及 [server_args.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py),以了解更多关于启动服务器时可以提供的参数。

## Model and tokenizer
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--model-path`<br>`--model` | 模型权重的路径。可以是本地文件夹或 Hugging Face repo ID。 | `None` | Type: str |
| `--tokenizer-path` | tokenizer 的路径。 | `None` | Type: str |
| `--tokenizer-mode` | Tokenizer 模式。'auto' 会在可用时使用 fast tokenizer,'slow' 则始终使用 slow tokenizer。 | `auto` | `auto`, `slow` |
| `--tokenizer-worker-num` | tokenizer manager 的 worker 数量。 | `1` | Type: int |
| `--skip-tokenizer-init` | 如果设置,跳过 tokenizer 的初始化,并在 generate 请求中传入 input_ids。 | `False` | bool flag (set to enable) |
| `--load-format` | 要加载的模型权重格式。"auto" 会尝试以 safetensors 格式加载权重,如果 safetensors 格式不可用则回退到 pytorch bin 格式。"pt" 会以 pytorch bin 格式加载权重。"safetensors" 会以 safetensors 格式加载权重。"npcache" 会以 pytorch 格式加载权重并存储一个 numpy 缓存以加速加载。"dummy" 会用随机值初始化权重,主要用于 profiling。"gguf" 会以 gguf 格式加载权重。"bitsandbytes" 会使用 bitsandbytes quantization 加载权重。"layered" 会逐层加载权重,以便在加载下一层之前先量化某一层,从而使峰值内存占用更小。"flash_rl" 会以 flash_rl 格式加载权重。"fastsafetensors" 和 "private" 也受支持。"runai_streamer" 支持从对象存储和共享文件系统直接加载模型。| `auto` | `auto`, `pt`, `safetensors`, `npcache`, `dummy`, `sharded_state`, `gguf`, `bitsandbytes`, `layered`, `flash_rl`, `remote`, `remote_instance`, `fastsafetensors`, `private`, `runai_streamer` |
| `--model-loader-extra-config` | 模型 loader 的额外配置。这将被传递给与所选 load_format 对应的模型 loader。 | `{}` | Type: str |
| `--trust-remote-code` | 是否允许 Hub 上以自身 modeling 文件定义的自定义模型。 | `False` | bool flag (set to enable) |
| `--context-length` | 模型的最大上下文长度。默认为 None(将改为使用模型 config.json 中的值)。 | `None` | Type: int |
| `--is-embedding` | 是否将 CausalLM 用作 embedding 模型。 | `False` | bool flag (set to enable) |
| `--enable-multimodal` | 为所服务的模型启用 multimodal 功能。如果所服务的模型不是 multimodal,则不会有任何影响。 | `None` | bool flag (set to enable) |
| `--revision` | 要使用的特定模型版本。可以是分支名、标签名或 commit id。如果未指定,将使用默认版本。 | `None` | Type: str |
| `--model-impl` | 要使用的模型实现。* "auto" 会尝试使用 SGLang 实现(如果存在),并在没有 SGLang 实现时回退到 Transformers 实现。* "sglang" 会使用 SGLang 模型实现。* "transformers" 会使用 Transformers 模型实现。 | `auto` | Type: str |

## HTTP server
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--host` | HTTP 服务器的 host。 | `127.0.0.1` | Type: str |
| `--port` | HTTP 服务器的 port。 | `30000` | Type: int |
| `--fastapi-root-path` | 应用位于基于路径的路由代理之后。 | `""` | Type: str |
| `--grpc-mode` | 如果设置,使用 gRPC 服务器而非 HTTP 服务器。 | `False` | bool flag (set to enable) |
| `--skip-server-warmup` | 如果设置,跳过 warmup。 | `False` | bool flag (set to enable) |
| `--warmups` | 指定在服务器启动前运行的自定义 warmup 函数(csv),例如 --warmups=warmup_name1,warmup_name2 会在服务器开始监听请求前运行 warmup.py 中指定的函数 `warmup_name1` 和 `warmup_name2`。 | `None` | Type: str |
| `--nccl-port` | NCCL 分布式环境设置使用的 port。默认为随机端口。 | `None` | Type: int |
| `--checkpoint-engine-wait-weights-before-ready` | 如果设置,服务器将在通过 checkpoint-engine 或其他更新方法加载初始权重之后,才开始服务推理请求。 | `False` | bool flag (set to enable) |

## Quantization and data type
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--dtype` | 模型权重和激活的数据类型。* "auto" 会为 FP32 和 FP16 模型使用 FP16 精度,为 BF16 模型使用 BF16 精度。* "half" 表示 FP16。推荐用于 AWQ quantization。* "float16" 与 "half" 相同。* "bfloat16" 用于在精度和范围之间取得平衡。* "float" 是 FP32 精度的简写。* "float32" 表示 FP32 精度。 | `auto` | `auto`, `half`, `float16`, `bfloat16`, `float`, `float32` |
| `--quantization` | quantization 方法。 | `None` | `awq`, `fp8`, `gptq`, `marlin`, `gptq_marlin`, `awq_marlin`, `bitsandbytes`, `gguf`, `modelopt`, `modelopt_fp8`, `modelopt_fp4`, `petit_nvfp4`, `w8a8_int8`, `w8a8_fp8`, `moe_wna16`, `qoq`, `w4afp8`, `mxfp4`, `mxfp8`, `auto-round`, `compressed-tensors`, `modelslim`, `quark_int4fp8_moe` |
| `--quantization-param-path` | 包含 KV cache 缩放因子的 JSON 文件路径。当 KV cache dtype 为 FP8 时,通常应提供此项。否则,KV cache 缩放因子默认为 1.0,这可能导致精度问题。 | `None` | Type: Optional[str] |
| `--kv-cache-dtype` | kv cache 存储的数据类型。"auto" 会使用模型数据类型。"bf16" 或 "bfloat16" 表示 BF16 KV cache。"fp8_e5m2" 和 "fp8_e4m3" 在 CUDA 11.8+ 上受支持。"fp4_e2m1"(仅 mxfp4)在 CUDA 12.8+ 和 PyTorch 2.8.0+ 上受支持 | `auto` | `auto`, `fp8_e5m2`, `fp8_e4m3`, `bf16`, `bfloat16`, `fp4_e2m1` |
| `--enable-fp32-lm-head` | 如果设置,LM head 输出(logits)为 FP32。 | `False` | bool flag (set to enable) |
| `--modelopt-quant` | ModelOpt quantization 配置。支持的值:'fp8'、'int4_awq'、'w4a8_awq'、'nvfp4'、'nvfp4_awq'。这需要安装 NVIDIA Model Optimizer 库:pip install nvidia-modelopt | `None` | Type: str |
| `--modelopt-checkpoint-restore-path` | 用于恢复先前保存的 ModelOpt 量化 checkpoint 的路径。如果提供,将跳过量化过程,并从该 checkpoint 加载模型。 | `None` | Type: str |
| `--modelopt-checkpoint-save-path` | 量化后保存 ModelOpt 量化 checkpoint 的路径。这允许在未来的运行中复用量化后的模型。 | `None` | Type: str |
| `--modelopt-export-path` | 在 ModelOpt quantization 之后将量化模型导出为 HuggingFace 格式的路径。导出的模型随后可直接用于 SGLang 推理。如果未提供,模型将不会被导出。 | `None` | Type: str |
| `--quantize-and-serve` | 使用 ModelOpt 量化模型并立即服务,而不导出。这对开发和原型设计很有用。对于生产环境,建议使用独立的量化和部署步骤。 | `False` | bool flag (set to enable) |
| `--rl-quant-profile` | FlashRL quantization profile 的路径。使用 --load-format flash_rl 时必需。 | `None` | Type: str |

## Memory and scheduling
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--mem-fraction-static` | 用于静态分配(模型权重和 KV cache 内存池)的内存比例。如果你看到 out-of-memory 错误,请使用更小的值。 | `None` | Type: float |
| `--max-running-requests` | 正在运行的请求的最大数量。 | `None` | Type: int |
| `--max-queued-requests` | 排队请求的最大数量。使用 disaggregation-mode 时此选项被忽略。 | `None` | Type: int |
| `--max-total-tokens` | 内存池中的最大 token 数量。如果未指定,将根据内存使用比例自动计算。此选项通常用于开发和调试目的。 | `None` | Type: int |
| `--chunked-prefill-size` | chunked prefill 中每个 chunk 的最大 token 数量。设置为 -1 表示禁用 chunked prefill。 | `None` | Type: int |
| `--prefill-max-requests` | 一个 prefill batch 中请求的最大数量。如果未指定,则没有限制。 | `None` | Type: int |
| `--enable-dynamic-chunking` | 为 pipeline parallelism 启用动态 chunk 大小调整。启用后,chunk 大小会根据拟合函数动态计算,以保持各 chunk 间执行时间一致。 | `False` | bool flag (set to enable) |
| `--max-prefill-tokens` | 一个 prefill batch 中的最大 token 数量。实际的边界将是此值与模型最大上下文长度中的较大者。 | `16384` | Type: int |
| `--schedule-policy` | 请求的调度策略。 | `fcfs` | `lpm`, `random`, `fcfs`, `dfs-weight`, `lof`, `priority`, `routing-key` |
| `--enable-priority-scheduling` | 启用优先级调度。默认情况下,优先级整数值更高的请求会被优先调度。 | `False` | bool flag (set to enable) |
| `--abort-on-priority-when-disabled` | 如果设置,当优先级调度被禁用时,中止那些指定了优先级的请求。 | `False` | bool flag (set to enable) |
| `--schedule-low-priority-values-first` | 如果与 --enable-priority-scheduling 一起指定,调度器将优先调度优先级整数值更低的请求。 | `False` | bool flag (set to enable) |
| `--priority-scheduling-preemption-threshold` | 传入请求要抢占正在运行的请求所需的最小优先级差值。 | `10` | Type: int |
| `--schedule-conservativeness` | 调度策略的保守程度。值越大表示调度越保守。如果你看到请求被频繁回退(retracted),请使用更大的值。 | `1.0` | Type: float |
| `--page-size` | 一个 page 中的 token 数量。 | `1` | Type: int |
| `--swa-full-tokens-ratio` | SWA 层 KV tokens / full 层 KV tokens 的比例,与 swa:full 层的数量无关。它应在 0 和 1 之间。例如 0.5 表示如果每个 swa 层有 50 个 token,则每个 full 层有 100 个 token。 | `0.8` | Type: float |
| `--disable-hybrid-swa-memory` | 禁用 hybrid SWA 内存。 | `False` | bool flag (set to enable) |
| `--radix-eviction-policy` | radix tree 的淘汰策略。'lru' 表示 Least Recently Used(最近最少使用),'lfu' 表示 Least Frequently Used(最不经常使用)。 | `lru` | `lru`, `lfu` |
| `--enable-prefill-delayer` | 为 DP attention 启用 prefill delayer 以减少空闲时间。 | `False` | bool flag (set to enable) |
| `--prefill-delayer-max-delay-passes` | 延迟 prefill 的最大 forward pass 数。 | `30` | Type: int |
| `--prefill-delayer-token-usage-low-watermark` | prefill delayer 的 token 使用率低水位线。 | `None` | Type: float |
| `--prefill-delayer-forward-passes-buckets` | prefill delayer forward passes 直方图的自定义 bucket。0 和 max_delay_passes-1 会被自动添加。 | `None` | List[float] |
| `--prefill-delayer-wait-seconds-buckets` | prefill delayer 等待秒数直方图的自定义 bucket。0 会被自动添加。 | `None` | List[float] |

## Runtime options
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--device` | 要使用的设备('cuda'、'xpu'、'hpu'、'npu'、'cpu')。如果未指定,默认为自动检测。 | `None` | Type: str |
| `--tensor-parallel-size`<br>`--tp-size` | tensor parallelism 大小。 | `1` | Type: int |
| `--pipeline-parallel-size`<br>`--pp-size` | pipeline parallelism 大小。 | `1` | Type: int |
| `--attention-context-parallel-size`<br>`--attn-cp-size`| attention context parallelism 大小。 | `1` | Type: int|
| `--moe-data-parallel-size`<br>`--moe-dp-size`| moe data parallelism 大小。 | `1` | Type: int|
| `--pp-max-micro-batch-size` | pipeline parallelism 中的最大 micro batch 大小。 | `None` | Type: int |
| `--pp-async-batch-depth` | pipeline parallelism 的异步 batch 深度。 | `0` | Type: int |
| `--stream-interval` | 以 token 长度衡量的 streaming 间隔(或缓冲区大小)。较小的值使 streaming 更平滑,较大的值使 throughput 更高 | `1` | Type: int |
| `--incremental-streaming-output` | 是否以一系列不相交的片段进行输出。 | `False` | bool flag (set to enable) |
| `--random-seed` | 随机种子。 | `None` | Type: int |
| `--constrained-json-whitespace-pattern` | (仅 outlines 和 llguidance 后端)JSON 约束输出中允许的语法空白的正则模式。例如,要允许模型生成连续空白,将模式设置为 [\n\t ]* | `None` | Type: str |
| `--constrained-json-disable-any-whitespace` | (仅 xgrammar 和 llguidance 后端)在 JSON 约束输出中强制使用紧凑表示。 | `False` | bool flag (set to enable) |
| `--watchdog-timeout` | 设置 watchdog 超时(秒)。如果一个 forward batch 耗时超过此值,服务器将崩溃以防止挂起。 | `300` | Type: float |
| `--soft-watchdog-timeout` | 设置 soft watchdog 超时(秒)。如果一个 forward batch 耗时超过此值,服务器将转储信息用于调试。 | `None` | Type: float |
| `--dist-timeout` | 设置 torch.distributed 初始化的超时。 | `None` | Type: int |
| `--download-dir` | huggingface 的模型下载目录。 | `None` | Type: str |
| `--model-checksum` | 模型文件完整性校验。如果提供但无值,则使用 model-path 作为 HF repo ID。否则,提供 checksums JSON 文件路径或 HuggingFace repo ID。 | `None` | Type: str |
| `--base-gpu-id` | 开始分配 GPU 的基准 GPU ID。在同一台机器上运行多个实例时很有用。 | `0` | Type: int |
| `--gpu-id-step` | 所用的连续 GPU ID 之间的步长。例如,设置为 2 将使用 GPU 0,2,4,... | `1` | Type: int |
| `--sleep-on-idle` | 在 sglang 空闲时降低 CPU 使用率。 | `False` | bool flag (set to enable) |
| `--custom-sigquit-handler` | 注册一个自定义 sigquit handler,以便在服务器关闭后进行额外清理。这仅适用于 Engine,不适用于 CLI。 | `None` | Type: str |

## Logging
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--log-level` | 所有 logger 的日志级别。 | `info` | Type: str |
| `--log-level-http` | HTTP 服务器的日志级别。如果未设置,默认复用 --log-level。 | `None` | Type: str |
| `--log-requests` | 记录所有请求的元数据、输入和输出。详细程度由 --log-requests-level 决定 | `False` | bool flag (set to enable) |
| `--log-requests-level` | 0:记录元数据(无 sampling 参数)。1:记录元数据和 sampling 参数。2:记录元数据、sampling 参数和部分输入/输出。3:记录每个输入/输出。 | `2` | `0`, `1`, `2`, `3` |
| `--log-requests-format` | 请求日志格式:'text'(人类可读)或 'json'(结构化) | `text` | `text`, `json` |
| `--log-requests-target` | 请求日志的目标:'stdout' 和/或用于文件输出的目录路径。可以指定多个目标,例如 '--log-requests-target stdout /my/path'。 | `None` | List[str] |
| `--uvicorn-access-log-exclude-prefixes` | 排除请求路径以这些前缀中任意一个开头的 uvicorn access 日志。默认为空(禁用)。 | `[]` | List[str] |
| `--crash-dump-folder` | 用于转储崩溃前最后 5 分钟内请求(如有)的文件夹路径。如果未指定,崩溃转储将被禁用。 | `None` | Type: str |
| `--show-time-cost` | 显示自定义标记的耗时。 | `False` | bool flag (set to enable) |
| `--enable-metrics` | 启用记录 prometheus metrics。 | `False` | bool flag (set to enable) |
| `--enable-mfu-metrics` | 启用估算的 MFU 相关 prometheus metrics。 | `False` | bool flag (set to enable) |
| `--enable-metrics-for-all-schedulers` | 当你希望所有 TP rank(而不仅是 TP 0)上的调度器分别记录请求 metrics 时,启用 --enable-metrics-for-all-schedulers。这在启用 dp_attention 时尤其有用,否则所有 metrics 看起来都来自 TP 0。 | `False` | bool flag (set to enable) |
| `--tokenizer-metrics-custom-labels-header` | 指定用于传递 tokenizer metrics 自定义标签的 HTTP header。 | `x-custom-labels` | Type: str |
| `--tokenizer-metrics-allowed-custom-labels` | tokenizer metrics 允许的自定义标签。这些标签通过 HTTP 请求中 '--tokenizer-metrics-custom-labels-header' 字段里的 dict 指定,例如,若设置了 '--tokenizer-metrics-allowed-custom-labels label1 label2',则允许 {'label1': 'value1', 'label2': 'value2'}。 | `None` | List[str] |
| `--bucket-time-to-first-token` | time to first token 的 bucket,指定为浮点数列表。 | `None` | List[float] |
| `--bucket-inter-token-latency` | inter-token latency 的 bucket,指定为浮点数列表。 | `None` | List[float] |
| `--bucket-e2e-request-latency` | end-to-end 请求延迟的 bucket,指定为浮点数列表。 | `None` | List[float] |
| `--collect-tokens-histogram` | 收集 prompt/generation token 直方图。 | `False` | bool flag (set to enable) |
| `--prompt-tokens-buckets` | prompt token 的 bucket 规则。支持 3 种规则类型:'default' 使用预定义 bucket;'tse <middle> <base> <count>' 生成两侧指数分布的 bucket(例如,'tse 1000 2 8' 生成 bucket [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]);'custom <value1> <value2> ...' 使用自定义 bucket 值(例如,'custom 10 50 100 500')。 | `None` | List[str] |
| `--generation-tokens-buckets` | generation token 直方图的 bucket 规则。支持 3 种规则类型:'default' 使用预定义 bucket;'tse <middle> <base> <count>' 生成两侧指数分布的 bucket(例如,'tse 1000 2 8' 生成 bucket [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]);'custom <value1> <value2> ...' 使用自定义 bucket 值(例如,'custom 10 50 100 500')。 | `None` | List[str] |
| `--gc-warning-threshold-secs` | 长 GC 警告的阈值。如果一次 GC 耗时超过此值,将记录一条警告。设置为 0 以禁用。 | `0.0` | Type: float |
| `--decode-log-interval` | decode batch 的日志间隔。 | `40` | Type: int |
| `--enable-request-time-stats-logging` | 启用每请求的时间统计日志 | `False` | bool flag (set to enable) |
| `--kv-events-config` | NVIDIA dynamo KV 事件发布的 json 格式配置。如果使用此 flag,将启用发布。 | `None` | Type: str |
| `--enable-trace` | 启用 opentelemetry trace | `False` | bool flag (set to enable) |
| `--otlp-traces-endpoint` | 如果设置了 --enable-trace,配置 opentelemetry collector endpoint。格式:<ip>:<port> | `localhost:4317` | Type: str |

## RequestMetricsExporter configuration
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--export-metrics-to-file` | 将每个请求的性能 metrics 导出到本地文件(例如用于转发到外部系统)。 | `False` | bool flag (set to enable) |
| `--export-metrics-to-file-dir` | 写入性能 metrics 文件的目录路径(启用 --export-metrics-to-file 时必需)。 | `None` | Type: str |

## API related
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--api-key` | 设置服务器的 API key。它也用于 OpenAI API 兼容服务器。 | `None` | Type: str |
| `--admin-api-key` | 为管理/控制 endpoint(例如权重更新、缓存刷新、`/server_info`)设置 **admin API key**。当设置此项时,标记为仅管理员的 endpoint 需要 `Authorization: Bearer <admin_api_key>`。 | `None` | Type: str |
| `--served-model-name` | 覆盖 OpenAI API 服务器中 v1/models endpoint 返回的模型名。 | `None` | Type: str |
| `--weight-version` | 模型权重的版本标识符。如果未指定,默认为 'default'。 | `default` | Type: str |
| `--chat-template` | 内置 chat template 名称或 chat template 文件的路径。这仅用于 OpenAI 兼容 API 服务器。 | `None` | Type: str |
| `--hf-chat-template-name` | 当 HuggingFace tokenizer 有多个 chat template(例如 'default'、'tool_use'、'rag')时,指定要使用的命名模板。如果未设置,使用第一个可用模板。 | `None` | Type: str |
| `--completion-template` | 内置 completion template 名称或 completion template 文件的路径。这仅用于 OpenAI 兼容 API 服务器。目前仅用于代码补全。 | `None` | Type: str |
| `--file-storage-path` | 后端文件存储的路径。 | `sglang_storage` | Type: str |
| `--enable-cache-report` | 在每个 openai 请求的 usage.prompt_tokens_details 中返回缓存的 token 数量。 | `False` | bool flag (set to enable) |
| `--reasoning-parser` | 为 reasoning 模型指定 parser。支持的 parser:[deepseek-r1, deepseek-v3, glm45, gpt-oss, kimi, qwen3, qwen3-thinking, step3]。 | `None` | `deepseek-r1`, `deepseek-v3`, `glm45`, `gpt-oss`, `kimi`, `qwen3`, `qwen3-thinking`, `step3` |
| `--tool-call-parser` | 指定用于处理 tool-call 交互的 parser。支持的 parser:[deepseekv3, deepseekv31, glm, glm45, glm47, gpt-oss, kimi_k2, llama3, mistral, pythonic, qwen, qwen25, qwen3_coder, step3]。 | `None` | `deepseekv3`, `deepseekv31`, `glm`, `glm45`, `glm47`, `gpt-oss`, `kimi_k2`, `llama3`, `mistral`, `pythonic`, `qwen`, `qwen25`, `qwen3_coder`, `step3`, `gigachat3` |
| `--tool-server` | 为模型使用的工具服务器,可以是 'demo' 或逗号分隔的 tool server url 列表。如果未指定,将不使用 tool server。 | `None` | Type: str |
| `--sampling-defaults` | 从何处获取默认 sampling 参数。'openai' 使用 SGLang/OpenAI 默认值(temperature=1.0, top_p=1.0 等)。'model' 使用模型的 generation_config.json 来获取推荐的 sampling 参数(如果可用)。默认为 'model'。 | `model` | `openai`, `model` |

## Data parallelism
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--data-parallel-size`<br>`--dp-size` | data parallelism 大小。 | `1` | Type: int |
| `--load-balance-method` | data parallelism 的负载均衡策略。`total_tokens` 算法只能在应用了 DP attention 时使用。该算法基于 DP worker 的实时 token 负载进行负载均衡。 | `auto` | `auto`, `round_robin`, `follow_bootstrap_room`, `total_requests`, `total_tokens` |

## Multi-node distributed serving
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--dist-init-addr`<br>`--nccl-init-addr` | 用于初始化分布式后端的 host 地址(例如 `192.168.0.2:25000`)。 | `None` | Type: str |
| `--nnodes` | 节点数量。 | `1` | Type: int |
| `--node-rank` | 节点 rank。 | `0` | Type: int |

## Model override args
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--json-model-override-args` | 用于覆盖默认模型配置的 JSON 字符串格式字典。 | `{}` | Type: str |
| `--preferred-sampling-params` | 将在 /get_model_info 中返回的 json 格式 sampling 设置 | `None` | Type: str |

## LoRA
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-lora` | 为模型启用 LoRA 支持。如果提供了 `--lora-paths`,为了向后兼容,此参数会自动设置为 `True`。 | `False` | Bool flag (set to enable) |
| `--enable-lora-overlap-loading` | 启用异步 LoRA 权重加载,以便将 H2D 传输与 GPU 计算重叠。如果你发现 LoRA 工作负载受 adapter 权重加载瓶颈影响(例如频繁加载大型 LoRA adapter),应启用此项。 | `False` | Bool flag (set to enable)
| `--max-lora-rank` | 应支持的最大 LoRA rank。如果未指定,将根据 `--lora-paths` 中提供的 adapter 自动推断。当你预期在服务器启动后动态加载更大 LoRA rank 的 adapter 时,需要此参数。 | `None` | Type: int |
| `--lora-target-modules` | 应应用 LoRA 的所有 target module 的并集(例如 `q_proj`、`k_proj`、`gate_proj`)。如果未指定,将根据 `--lora-paths` 中提供的 adapter 自动推断。你也可以将其设置为 `all`,以为所有支持的 module 启用 LoRA;注意这可能带来轻微的性能开销。 | `None` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `qkv_proj`, `gate_up_proj`, `all` |
| `--lora-paths` | 要加载的 LoRA adapter 列表。每个 adapter 必须以下列格式之一指定:`<PATH>` \| `<NAME>=<PATH>` \| 符合 schema `{"lora_name": str, "lora_path": str, "pinned": bool}` 的 JSON。 | `None` | Type: List[str] / JSON objects |
| `--max-loras-per-batch` | 一个运行 batch 中 adapter 的最大数量,包括仅使用 base 的请求。 | `8` | Type: int |
| `--max-loaded-loras` | 如果指定,限制同时加载到 CPU 内存中的 LoRA adapter 的最大数量。必须 ≥ `--max-loras-per-batch`。 | `None` | Type: int |
| `--lora-eviction-policy` | 当 GPU 内存池满时的 LoRA adapter 淘汰策略。 | `lru` | `lru`, `fifo` |
| `--lora-backend` | 为 multi-LoRA 服务选择 kernel 后端。 | `csgmv` | `triton`, `csgmv`, `ascend`, `torch_native` |
| `--max-lora-chunk-size` | ChunkedSGMV LoRA 后端的最大 chunk 大小。仅当 `--lora-backend` 为 `csgmv` 时使用。较大的值可能提升性能。 | `16` | `16`, `32`, `64`, `128` |

## Kernel Backends (Attention, Sampling, Grammar, GEMM)
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--attention-backend` | 为 attention 层选择 kernel。 | `None` | `triton`, `torch_native`, `flex_attention`, `nsa`, `cutlass_mla`, `fa3`, `fa4`, `flashinfer`, `flashmla`, `trtllm_mla`, `trtllm_mha`, `dual_chunk_flash_attn`, `aiter`, `wave`, `intel_amx`, `ascend` |
| `--prefill-attention-backend` | 为 prefill attention 层选择 kernel(优先级高于 --attention-backend)。 | `None` | `triton`, `torch_native`, `flex_attention`, `nsa`, `cutlass_mla`, `fa3`, `fa4`, `flashinfer`, `flashmla`, `trtllm_mla`, `trtllm_mha`, `dual_chunk_flash_attn`, `aiter`, `wave`, `intel_amx`, `ascend` |
| `--decode-attention-backend` | 为 decode attention 层选择 kernel(优先级高于 --attention-backend)。 | `None` | `triton`, `torch_native`, `flex_attention`, `nsa`, `cutlass_mla`, `fa3`, `fa4`, `flashinfer`, `flashmla`, `trtllm_mla`, `trtllm_mha`, `dual_chunk_flash_attn`, `aiter`, `wave`, `intel_amx`, `ascend` |
| `--sampling-backend` | 为 sampling 层选择 kernel。 | `None` | `flashinfer`, `pytorch`, `ascend` |
| `--grammar-backend` | 为 grammar-guided 解码选择后端。 | `None` | `xgrammar`, `outlines`, `llguidance`, `none` |
| `--mm-attention-backend` | 设置 multimodal attention 后端。 | `None` | `sdpa`, `fa3`, `fa4`, `triton_attn`, `ascend_attn`, `aiter_attn` |
| `--nsa-prefill-backend` | 为 prefill 阶段选择 NSA 后端(在运行 DeepSeek NSA 风格 attention 时覆盖 `--attention-backend`)。 | `flashmla_sparse` | `flashmla_sparse`, `flashmla_kv`, `flashmla_auto`, `fa3`, `tilelang`, `aiter`, `trtllm` |
| `--nsa-decode-backend` | 在运行 DeepSeek NSA 风格 attention 时为 decode 阶段选择 NSA 后端。为解码覆盖 `--attention-backend`。 | `fa3` | `flashmla_sparse`, `flashmla_kv`, `fa3`, `tilelang`, `aiter`, `trtllm` |
| `--fp8-gemm-backend` | 为 Blockwise FP8 GEMM 操作选择 runner 后端。选项:'auto'(默认,根据硬件自动选择)、'deep_gemm'(JIT 编译;当安装了 DeepGEMM 时在 NVIDIA Hopper (SM90) 和 Blackwell (SM100) 上默认启用)、'flashinfer_trtllm'(FlashInfer TRTLLM 后端;仅 SM100/SM103)、'flashinfer_cutlass'(FlashInfer CUTLASS 后端,仅 SM120)、'flashinfer_deepgemm'(仅 Hopper SM90,在解码中针对较小的 M 维度使用 swapAB 优化)、'cutlass'(最适合 Hopper/Blackwell GPU 和高 throughput)、'triton'(回退方案,兼容性广)、'aiter'(仅 ROCm)。| `auto` | `auto`, `deep_gemm`, `flashinfer_trtllm`, `flashinfer_cutlass`, `flashinfer_deepgemm`, `cutlass`, `triton`, `aiter` |
| `--fp4-gemm-backend` | 为 NVFP4 GEMM 操作选择 runner 后端。选项:'flashinfer_cutlass'(默认)、'auto'(根据 CUDA/cuDNN 版本在 flashinfer_cudnn/flashinfer_cutlass 之间自动选择)、'flashinfer_cudnn'(FlashInfer cuDNN 后端,在 CUDA 13+ 且 cuDNN 9.15+ 上最优)、'flashinfer_trtllm'(FlashInfer TensorRT-LLM 后端,需要带 shuffling 的不同权重准备)。所有后端均来自 FlashInfer;当 FlashInfer 不可用时,会自动回退使用 sgl-kernel CUTLASS。| `flashinfer_cutlass` | `auto`, `flashinfer_cudnn`, `flashinfer_cutlass`, `flashinfer_trtllm` |
| `--disable-flashinfer-autotune` | Flashinfer autotune 默认启用。设置此 flag 以禁用 autotune。 | `False` | bool flag (set to enable) |

## Speculative decoding
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--speculative-algorithm` | speculative 算法。 | `None` | `EAGLE`, `EAGLE3`, `NEXTN`, `STANDALONE`, `NGRAM` |
| `--speculative-draft-model-path`<br>`--speculative-draft-model` | draft model 权重的路径。可以是本地文件夹或 Hugging Face repo ID。 | `None` | Type: str |
| `--speculative-draft-model-revision` | 要使用的特定 draft model 版本。可以是分支名、标签名或 commit id。如果未指定,将使用默认版本。 | `None` | Type: str |
| `--speculative-draft-load-format` | 要加载的 draft model 权重格式。如果未指定,将使用与 --load-format 相同的格式。使用 'dummy' 以随机值初始化 draft model 权重用于 profiling。 | `None` | Same as --load-format options |
| `--speculative-num-steps` | Speculative Decoding 中从 draft model 采样的步数。 | `None` | Type: int |
| `--speculative-eagle-topk` | eagle2 中每步从 draft model 采样的 token 数量。 | `None` | Type: int |
| `--speculative-num-draft-tokens` | Speculative Decoding 中从 draft model 采样的 token 数量。 | `None` | Type: int |
| `--speculative-accept-threshold-single` | 如果某个 draft token 在 target model 中的概率大于此阈值,则接受它。 | `1.0` | Type: float |
| `--speculative-accept-threshold-acc` | draft token 的接受概率从其 target 概率 p 提升到 min(1, p / threshold_acc)。 | `1.0` | Type: float |
| `--speculative-token-map` | draft model 的小词表的路径。 | `None` | Type: str |
| `--speculative-attention-mode` | speculative decoding 操作(target verify 和 draft extend)的 attention 后端。可以是 'prefill'(默认)或 'decode' 之一。 | `prefill` | `prefill`, `decode` |
| `--speculative-draft-attention-backend` | speculative decoding drafting 的 attention 后端。 | `None` | Same as attention backend options |
| `--speculative-moe-runner-backend` | EAGLE speculative decoding 的 MOE 后端,选项参见 --moe-runner-backend。如果未设置,与 moe runner backend 相同。 | `None` | Same as --moe-runner-backend options |
| `--speculative-moe-a2a-backend` | EAGLE speculative decoding 的 MOE A2A 后端,选项参见 --moe-a2a-backend。如果未设置,与 moe a2a backend 相同。 | `None` | Same as --moe-a2a-backend options |
| `--speculative-draft-model-quantization` | speculative model 的 quantization 方法。 | `None` | Same as --quantization options |

## Ngram speculative decoding
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--speculative-ngram-min-bfs-breadth` | ngram speculative decoding 中 BFS(Breadth-First Search,广度优先搜索)的最小宽度。 | `1` | Type: int |
| `--speculative-ngram-max-bfs-breadth` | ngram speculative decoding 中 BFS(Breadth-First Search,广度优先搜索)的最大宽度。 | `10` | Type: int |
| `--speculative-ngram-match-type` | Ngram 树构建模式。`BFS` 选择基于新近度(recency)的扩展,`PROB` 选择基于频率的扩展。此设置会转发给 ngram cache 实现。 | `BFS` | `BFS`, `PROB` |
| `--speculative-ngram-max-trie-depth` | ngram trie 存储和匹配的最大后缀长度。 | `18` | Type: int |
| `--speculative-ngram-capacity` | ngram speculative decoding 的缓存容量。 | `10000000` | Type: int |

## Multi-layer Eagle speculative decoding
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-multi-layer-eagle` | 启用 multi-layer Eagle speculative decoding。 | `False` | bool flag (set to enable) |

## MoE
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--expert-parallel-size`<br>`--ep-size`<br>`--ep` | expert parallelism 大小。 | `1` | Type: int |
| `--moe-a2a-backend` | 为 expert parallelism 选择 all-to-all 通信的后端。 | `none` | `none`, `deepep`, `mooncake`, `mori`, `nixl`, `ascend_fuseep`|
| `--moe-runner-backend` | 为 MoE 选择 runner 后端。 | `auto` | `auto`, `deep_gemm`, `triton`, `triton_kernel`, `flashinfer_trtllm`, `flashinfer_trtllm_routed`, `flashinfer_cutlass`, `flashinfer_mxfp4`, `flashinfer_cutedsl`, `cutlass` |
| `--flashinfer-mxfp4-moe-precision` | 选择 flashinfer mxfp4 moe 的计算精度 | `default` | `default`, `bf16` |
| `--enable-flashinfer-allreduce-fusion` | 启用 FlashInfer allreduce 与 Residual RMSNorm 的融合。 | `False` | bool flag (set to enable) |
| `--enable-aiter-allreduce-fusion` | 启用 aiter allreduce 与 Residual RMSNorm 的融合。 | `False` | bool flag (set to enable) |
| `--deepep-mode` | 启用 DeepEP MoE 时选择模式,可以是 `normal`、`low_latency` 或 `auto`。默认为 `auto`,表示对 decode batch 使用 `low_latency`,对 prefill batch 使用 `normal`。 | `auto` | `normal`, `low_latency`, `auto` |
| `--ep-num-redundant-experts` | 在 expert parallel 中分配这么多冗余 expert。 | `0` | Type: int |
| `--ep-dispatch-algorithm` | 在 expert parallel 中为冗余 expert 选择 rank 的算法。 | `None` | Type: str |
| `--init-expert-location` | EP expert 的初始位置。 | `trivial` | Type: str |
| `--enable-eplb` | 启用 EPLB 算法 | `False` | bool flag (set to enable) |
| `--eplb-algorithm` | 选择的 EPLB 算法 | `auto` | Type: str |
| `--eplb-rebalance-num-iterations` | 自动触发一次 EPLB 重新平衡的迭代次数。 | `1000` | Type: int |
| `--eplb-rebalance-layers-per-chunk` | 每个 forward pass 重新平衡的层数。 | `None` | Type: int |
| `--eplb-min-rebalancing-utilization-threshold` | 触发 EPLB 重新平衡的 GPU 平均利用率最小阈值。必须在 [0.0, 1.0] 范围内。 | `1.0` | Type: float |
| `--expert-distribution-recorder-mode` | expert distribution recorder 的模式。 | `None` | Type: str |
| `--expert-distribution-recorder-buffer-size` | expert distribution recorder 的环形缓冲区大小。设置为 -1 表示无限缓冲区。 | `None` | Type: int |
| `--enable-expert-distribution-metrics` | 启用记录 expert 均衡度的 metrics | `False` | bool flag (set to enable) |
| `--deepep-config` | 适合你自己集群的已调优 DeepEP 配置。可以是包含 JSON 内容的字符串或文件路径。 | `None` | Type: str |
| `--moe-dense-tp-size` | MoE dense MLP 层的 TP 大小。当 TP 大小较大、MLP 层权重维度小于 GEMM 支持的最小维度而导致错误时,此 flag 很有用。 | `None` | Type: int |
| `--elastic-ep-backend` | 为 elastic EP 指定集合通信后端。目前支持 'mooncake'。 | `none` | `none`, `mooncake` |
| `--enable-elastic-expert-backup` | 启用 elastic EP 后端在 DRAM 中备份 expert 权重的功能。目前支持 'mooncake'。| `False` | bool flag (set to enable) |
| `--mooncake-ib-device` | Mooncake Backend 传输使用的 InfiniBand 设备,接受多个逗号分隔的设备(例如,--mooncake-ib-device mlx5_0,mlx5_1)。默认为 None,当启用 Mooncake Backend 时会触发自动设备检测。 | `None` | Type: str |

## Mamba Cache
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--max-mamba-cache-size` | mamba cache 的最大大小。 | `None` | Type: int |
| `--mamba-ssm-dtype` | mamba cache 中 SSM 状态的数据类型。 | `float32` | `float32`, `bfloat16`, `float16` |
| `--mamba-full-memory-ratio` | mamba 状态内存与 full kv cache 内存的比例。 | `0.9` | Type: float |
| `--mamba-scheduler-strategy` | mamba 调度器使用的策略。`auto` 当前默认为 `no_buffer`。1. `no_buffer` 由于不分配额外的 mamba 状态缓冲区,不支持 overlap scheduler。分支点缓存支持是可行的但尚未实现。2. `extra_buffer` 通过分配额外的 mamba 状态缓冲区来跟踪 mamba 状态以用于缓存,从而支持 overlap schedule(每个运行 req 的 mamba 状态使用量对于非 spec 变为 `2x`;对于 spec dec 变为 `1+(1/(2+speculative_num_draft_tokens))x`(例如当 speculative_num_draft_tokens==4 时为 1.16x))。2a. 对于非 KV-cache-bound 的情况,`extra_buffer` 严格更优;对于 KV-cache-bound 的情况,权衡取决于启用 overlap 是否能抵消最大运行请求数的减少。2b. 在 radix cache 分支点进行 mamba 缓存严格优于不分支,但需要 kernel 支持(目前仅 FLA 后端),目前仅 extra_buffer 支持分支。 | `auto` | `auto`, `no_buffer`, `extra_buffer` |
| `--mamba-track-interval` | 在 decode 期间跟踪 mamba 状态的间隔(以 token 计)。仅当 `--mamba-scheduler-strategy` 为 `extra_buffer` 时使用。如果设置,必须能被 page_size 整除,且在使用 speculative decoding 时必须 >= speculative_num_draft_tokens。 | `256` | Type: int |

## Hierarchical cache
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-hierarchical-cache` | 启用 hierarchical cache | `False` | bool flag (set to enable) |
| `--hicache-ratio` | host KV cache 内存池大小与 device 池大小的比例。 | `2.0` | Type: float |
| `--hicache-size` | host KV cache 内存池的大小(GB),如果设置将覆盖 hicache_ratio。 | `0` | Type: int |
| `--hicache-write-policy` | hierarchical cache 的写策略。 | `write_through` | `write_back`, `write_through`, `write_through_selective` |
| `--hicache-io-backend` | CPU 与 GPU 之间 KV cache 传输的 IO 后端 | `kernel` | `direct`, `kernel`, `kernel_ascend` |
| `--hicache-mem-layout` | hierarchical cache 的 host 内存池布局。 | `layer_first` | `layer_first`, `page_first`, `page_first_direct`, `page_first_kv_split`, `page_head` |
| `--hicache-storage-backend` | hierarchical KV cache 的存储后端。内置后端:file、mooncake、hf3fs、nixl、aibrix。对于 dynamic 后端,使用 --hicache-storage-backend-extra-config 指定:backend_name(自定义名称)、module_path(Python 模块路径)、class_name(后端类名)。 | `None` | `file`, `mooncake`, `hf3fs`, `nixl`, `aibrix`, `dynamic`, `eic` |
| `--hicache-storage-prefetch-policy` | 控制何时停止从存储后端预取。 | `best_effort` | `best_effort`, `wait_complete`, `timeout` |
| `--hicache-storage-backend-extra-config` | JSON 字符串格式的字典,或以 `@` 开头后跟 JSON/YAML/TOML 格式配置文件的字符串,包含存储后端的额外配置。 | `None` | Type: str |

## Hierarchical sparse attention
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--hierarchical-sparse-attention-extra-config` | hierarchical sparse attention 配置的 JSON 字符串格式字典。必需字段:`algorithm` (str)、`backend` (str)。所有其他字段是算法特定的,会传递给算法构造函数。 | `None` | Type: str |

## LMCache
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-lmcache` | 使用 LMCache 作为替代的 hierarchical cache 解决方案 | `False` | bool flag (set to enable) |

## Ktransformers
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--kt-weight-path` | [ktransformers parameter] amx kernel 的量化 expert 权重路径。本地文件夹。 | `None` | Type: str |
| `--kt-method` | [ktransformers parameter] CPU 执行的 quantization 格式。 | `AMXINT4` | Type: str |
| `--kt-cpuinfer` | [ktransformers parameter] CPUInfer 线程数。 | `None` | Type: int |
| `--kt-threadpool-count` | [ktransformers parameter] 与 NUMA 节点数量一一对应(每个 NUMA 一个线程池)。 | `2` | Type: int |
| `--kt-num-gpu-experts` | [ktransformers parameter] GPU expert 的数量。 | `None` | Type: int |
| `--kt-max-deferred-experts-per-token` | [ktransformers parameter] 每个 token 推迟到 CPU 的 expert 最大数量。除最后一层外的所有 MoE 层都使用此值;最后一层始终使用 0。 | `None` | Type: int |

## Diffusion LLM

| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--dllm-algorithm` | diffusion LLM 算法,例如 LowConfidence。 | `None` | Type: str |
| `--dllm-algorithm-config` | diffusion LLM 算法配置。必须是 YAML 文件。 | `None` | Type: str |

## Double Sparsity
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-double-sparsity` | 启用 double sparsity attention | `False` | bool flag (set to enable) |
| `--ds-channel-config-path` | double sparsity channel 配置的路径 | `None` | Type: str |
| `--ds-heavy-channel-num` | double sparsity attention 中 heavy channel 的数量 | `32` | Type: int |
| `--ds-heavy-token-num` | double sparsity attention 中 heavy token 的数量 | `256` | Type: int |
| `--ds-heavy-channel-type` | double sparsity attention 中 heavy channel 的类型 | `qk` | Type: str |
| `--ds-sparse-decode-threshold` | double-sparsity 后端从 dense 回退切换到 sparse decode kernel 之前所需的最小 decode 序列长度。 | `4096` | Type: int |

## Offloading
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--cpu-offload-gb` | 为 CPU offloading 保留多少 GB 的 RAM。 | `0` | Type: int |
| `--offload-group-size` | offloading 中每组的层数。 | `-1` | Type: int |
| `--offload-num-in-group` | 一组内要卸载的层数。 | `1` | Type: int |
| `--offload-prefetch-step` | offloading 中预取的步数。 | `1` | Type: int |
| `--offload-mode` | offloading 的模式。 | `cpu` | Type: str |

## Args for multi-item scoring
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--multi-item-scoring-delimiter` | multi-item scoring 的分隔符 token ID。用于将 Query 和 Item 组合为单个序列:Query<delimiter>Item1<delimiter>Item2<delimiter>... 这能高效地对单个 query 批量处理多个 item。 | `None` | Type: int |

## Optimization/debug options
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--disable-radix-cache` | 禁用用于 prefix caching 的 RadixAttention。 | `False` | bool flag (set to enable) |
| `--cuda-graph-max-bs` | 设置 cuda graph 的最大 batch size。它会将 cuda graph capture 的 batch size 扩展到此值。 | `None` | Type: int |
| `--cuda-graph-bs` | 设置 cuda graph 的 batch size 列表。 | `None` | List[int] |
| `--disable-cuda-graph` | 禁用 cuda graph。 | `False` | bool flag (set to enable) |
| `--disable-cuda-graph-padding` | 在需要 padding 时禁用 cuda graph。在不需要 padding 时仍使用 cuda graph。 | `False` | bool flag (set to enable) |
| `--enable-profile-cuda-graph` | 启用 cuda graph capture 的 profiling。 | `False` | bool flag (set to enable) |
| `--enable-cudagraph-gc` | 在 CUDA graph capture 期间启用垃圾回收。如果禁用(默认),capture 期间会冻结 GC 以加速该过程。 | `False` | bool flag (set to enable) |
| `--enable-layerwise-nvtx-marker` | 为模型启用逐层 NVTX profiling 注释。这会为每一层添加 NVTX 标记,以便使用 Nsight Systems 进行详细的逐层性能分析。 | `False` | bool flag (set to enable) |
| `--enable-nccl-nvls` | 在可用时为 prefill 密集型请求启用 NCCL NVLS。 | `False` | bool flag (set to enable) |
| `--enable-symm-mem` | 启用 NCCL 对称内存以实现快速集合通信。 | `False` | bool flag (set to enable) |
| `--disable-flashinfer-cutlass-moe-fp4-allgather` | 为 flashinfer cutlass moe 禁用 all-gather 前的量化。 | `False` | bool flag (set to enable) |
| `--enable-tokenizer-batch-encode` | 启用批量 tokenization,以在处理多个文本输入时提升性能。不要与图像输入、预先 tokenized 的 input_ids 或 input_embeds 一起使用。 | `False` | bool flag (set to enable) |
| `--disable-tokenizer-batch-decode` | 在解码多个 completion 时禁用批量解码。 | `False` | bool flag (set to enable) |
| `--disable-outlines-disk-cache` | 禁用 outlines 的磁盘缓存,以避免与文件系统或高并发相关的可能崩溃。 | `False` | bool flag (set to enable) |
| `--disable-custom-all-reduce` | 禁用自定义 all-reduce kernel 并回退到 NCCL。 | `False` | bool flag (set to enable) |
| `--enable-mscclpp` | 启用为小消息使用 mscclpp 进行 all-reduce kernel,并回退到 NCCL。 | `False` | bool flag (set to enable) |
| `--enable-torch-symm-mem` | 启用为 all-reduce kernel 使用 torch symm mem,并回退到 NCCL。仅支持 CUDA 设备 SM90 及以上。SM90 支持 world size 4、6、8。SM10 支持 world size 6、8。 | `False` | bool flag (set to enable) |
| `--disable-overlap-schedule` | 禁用 overlap scheduler,它将 CPU 调度器与 GPU 模型 worker 重叠。 | `False` | bool flag (set to enable) |
| `--enable-mixed-chunk` | 在使用 chunked prefill 时,启用在一个 batch 中混合 prefill 和 decode。 | `False` | bool flag (set to enable) |
| `--enable-dp-attention` | 为 attention 启用 data parallelism,为 FFN 启用 tensor parallelism。dp 大小应等于 tp 大小。目前支持 DeepSeek-V2 和 Qwen 2/3 MoE 模型。 | `False` | bool flag (set to enable) |
| `--enable-dp-lm-head` | 在 attention TP 组内启用 vocabulary parallel,以避免跨 DP 组的 all-gather,从而在 DP attention 下优化性能。 | `False` | bool flag (set to enable) |
| `--enable-two-batch-overlap` | 启用两个 micro batch 重叠。 | `False` | bool flag (set to enable) |
| `--enable-single-batch-overlap` | 让计算与通信在一个 micro batch 内重叠。 | `False` | bool flag (set to enable) |
| `--tbo-token-distribution-threshold` | micro-batch-overlap 中两个 batch 之间 token 分布的阈值,决定是进行 two-batch-overlap 还是 two-chunk-overlap。设置为 0 表示禁用 two-chunk-overlap。 | `0.48` | Type: float |
| `--enable-torch-compile` | 使用 torch.compile 优化模型。实验性功能。 | `False` | bool flag (set to enable) |
| `--enable-torch-compile-debug-mode` | 为 torch compile 启用 debug 模式。 | `False` | bool flag (set to enable) |
| `--disable-piecewise-cuda-graph` | 为 extend/prefill 禁用 piecewise cuda graph。PCG 默认启用。 | `False` | bool flag (set to disable) |
| `--enforce-piecewise-cuda-graph` | 强制使用 piecewise cuda graph,跳过所有自动禁用条件。仅用于测试。 | `False` | bool flag (set to enable) |
| `--piecewise-cuda-graph-tokens` | 设置使用 piecewise cuda graph 时的 token 列表。 | `None` | Type: JSON list |
| `--piecewise-cuda-graph-compiler` | 设置 piecewise cuda graph 的编译器。可选:eager、inductor。 | `eager` | `eager`, `inductor` |
| `--torch-compile-max-bs` | 设置使用 torch compile 时的最大 batch size。 | `32` | Type: int |
| `--piecewise-cuda-graph-max-tokens` | 设置使用 piecewise cuda graph 时的最大 token 数。 | `4096` | Type: int |
| `--torchao-config` | 使用 torchao 优化模型。实验性功能。当前可选:int8dq、int8wo、int4wo-<group_size>、fp8wo、fp8dq-per_tensor、fp8dq-per_row | `` | Type: str |
| `--enable-nan-detection` | 启用用于调试目的的 NaN 检测。 | `False` | bool flag (set to enable) |
| `--enable-p2p-check` | 启用 GPU 访问的 P2P 检查,否则默认允许 p2p 访问。 | `False` | bool flag (set to enable) |
| `--triton-attention-reduce-in-fp32` | 将中间 attention 结果转换为 fp32,以避免与 fp16 相关的可能崩溃。这仅影响 Triton attention kernel。 | `False` | bool flag (set to enable) |
| `--triton-attention-num-kv-splits` | flash decoding Triton kernel 中的 KV split 数量。在更长上下文场景中更大的值更好。默认值为 8。 | `8` | Type: int |
| `--triton-attention-split-tile-size` | flash decoding Triton kernel 中 split KV tile 的大小。用于确定性推理。 | `None` | Type: int |
| `--num-continuous-decode-steps` | 运行多个连续解码步骤以减少调度开销。这可能提升 throughput,但也可能增加 time-to-first-token 延迟。默认值为 1,表示一次只运行一个解码步骤。 | `1` | Type: int |
| `--delete-ckpt-after-loading` | 在加载模型后删除模型 checkpoint。 | `False` | bool flag (set to enable) |
| `--enable-memory-saver` | 允许使用 release_memory_occupation 和 resume_memory_occupation 来节省内存 | `False` | bool flag (set to enable) |
| `--enable-weights-cpu-backup` | 在 release_weights_occupation 和 resume_weights_occupation 期间将模型权重保存到 CPU 内存 | `False` | bool flag (set to enable) |
| `--enable-draft-weights-cpu-backup` | 在 release_weights_occupation 和 resume_weights_occupation 期间将 draft model 权重保存到 CPU 内存 | `False` | bool flag (set to enable) |
| `--allow-auto-truncate` | 允许自动截断超过最大输入长度的请求,而不是返回错误。 | `False` | bool flag (set to enable) |
| `--enable-custom-logit-processor` | 允许用户向服务器传递自定义 logit processor(出于安全考虑默认禁用) | `False` | bool flag (set to enable) |
| `--flashinfer-mla-disable-ragged` | 运行 flashinfer mla 时不使用 ragged prefill wrapper | `False` | bool flag (set to enable) |
| `--disable-shared-experts-fusion` | 为 deepseek v3/r1 禁用 shared experts fusion 优化。 | `False` | bool flag (set to enable) |
| `--disable-chunked-prefix-cache` | 为 deepseek 禁用 chunked prefix cache 功能,这应能为短序列节省开销。 | `False` | bool flag (set to enable) |
| `--disable-fast-image-processor` | 采用基础 image processor 而非 fast image processor。 | `False` | bool flag (set to enable) |
| `--keep-mm-feature-on-device` | 在处理后将 multimodal feature tensor 保留在设备上,以节省 D2H 拷贝。 | `False` | bool flag (set to enable) |
| `--enable-return-hidden-states` | 启用在响应中返回 hidden states。 | `False` | bool flag (set to enable) |
| `--enable-return-routed-experts` | 启用在响应中返回每一层的 routed expert。 | `False` | bool flag (set to enable) |
| `--scheduler-recv-interval` | 调度器轮询请求的间隔。可设置为 >1 以减少此开销。 | `1` | Type: int |
| `--numa-node` | 为子进程设置 numa 节点。第 i 个元素对应第 i 个子进程。 | `None` | List[int] |
| `--enable-deterministic-inference` | 启用带 batch invariant 操作的确定性推理模式。 | `False` | bool flag (set to enable) |
| `--rl-on-policy-target` | SGLang 需要匹配的训练系统,以实现真正的 on-policy。 | `None` | `fsdp` |
| `--enable-attn-tp-input-scattered` | 允许在仅使用 tensor parallelism 时将 attention 的输入分散(scattered),以减少诸如 qkv latent 等操作的计算负载。 | `False` | bool flag (set to enable) |
| `--enable-nsa-prefill-context-parallel` | 在 DeepSeek v3.2 的长序列 prefill 阶段启用 context parallelism。 | `False` | bool flag (set to enable) |
| `--nsa-prefill-cp-mode` | 在 context parallelism 下 DeepSeek v3.2 prefill 阶段的 token 切分模式。可选值:`round-robin-split`(默认)、`in-seq-split`。`round-robin-split` 根据 `token_idx % cp_size` 在各 rank 间分配 token。它支持 multi-batch prefill、fused MoE 和 FP8 KV cache。 | `in-seq-split` | `in-seq-split`, `round-robin-split` |
| `--enable-fused-qk-norm-rope` | 启用融合的 qk normalization 和 rope rotary embedding。 | `False` | bool flag (set to enable) |
| `--enable-precise-embedding-interpolation` | 为 embeddings grid 的 resize 启用角点对齐(corner alignment),以确保对插值 embedding 值进行更准确(但更慢)的评估。 | `False` | bool flag (set to enable) |

## Dynamic batch tokenizer
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-dynamic-batch-tokenizer` | 启用异步动态批量 tokenizer,以在多个请求并发到达时提升性能。 | `False` | bool flag (set to enable) |
| `--dynamic-batch-tokenizer-batch-size` | [仅当设置 --enable-dynamic-batch-tokenizer 时使用] 动态批量 tokenizer 的最大 batch size。 | `32` | Type: int |
| `--dynamic-batch-tokenizer-batch-timeout` | [仅当设置 --enable-dynamic-batch-tokenizer 时使用] 批量 tokenization 请求的超时(秒)。 | `0.002` | Type: float |

## Debug tensor dumps
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--debug-tensor-dump-output-folder` | 转储 tensor 的输出文件夹。 | `None` | Type: str |
| `--debug-tensor-dump-layers` | 要转储的 layer id。如果未指定则转储所有层。 | `None` | Type: JSON list |
| `--debug-tensor-dump-input-file` | 转储 tensor 的输入文件名 | `None` | Type: str |
| `--debug-tensor-dump-inject` | 将来自 jax 的输出注入为每一层的输入。 | `False` | Type: str |

## PD disaggregation
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--disaggregation-mode` | 仅用于 PD disaggregation。"prefill" 表示仅 prefill 服务器,"decode" 表示仅 decode 服务器。如果未指定,则不是 PD disaggregated | `null` | `null`, `prefill`, `decode` |
| `--disaggregation-transfer-backend` | disaggregation 传输的后端。默认为 mooncake。 | `mooncake` | `mooncake`, `nixl`, `ascend`, `fake` |
| `--disaggregation-bootstrap-port` | prefill 服务器上的 bootstrap server 端口。默认为 8998。 | `8998` | Type: int |
| `--disaggregation-ib-device` | disaggregation 传输使用的 InfiniBand 设备,接受单个设备(例如,--disaggregation-ib-device mlx5_0)或多个逗号分隔的设备(例如,--disaggregation-ib-device mlx5_0,mlx5_1)。默认为 None,当启用 mooncake 后端时会触发自动设备检测。 | `None` | Type: str |
| `--disaggregation-decode-enable-offload-kvcache` | 在 decode 服务器上启用异步 KV cache offloading(PD 模式)。 | `False` | bool flag (set to enable) |
| `--num-reserved-decode-tokens` | 向运行 batch 添加新请求时将为其保留内存的 decode token 数量。 | `512` | Type: int |
| `--disaggregation-decode-polling-interval` | decode 服务器中轮询请求的间隔。可设置为 >1 以减少此开销。 | `1` | Type: int |

## Encode prefill disaggregation
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--encoder-only` | 对于带 encoder 的 MLLM,启动一个仅 encoder 服务器 | `False` | bool flag (set to enable) |
| `--language-only` | 对于 VLM,仅为 language model 加载权重。 | `False` | bool flag (set to enable) |
| `--encoder-transfer-backend` | encoder disaggregation 传输的后端。默认为 zmq_to_scheduler。 | `zmq_to_scheduler` | `zmq_to_scheduler`, `zmq_to_tokenizer`, `mooncake` |
| `--encoder-urls` | encoder 服务器 url 列表。 | `[]` | Type: JSON list |

## Custom weight loader
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--custom-weight-loader` | 用于更新模型的自定义 dataloader。应设置为有效的 import 路径,例如 my_package.weight_load_func | `None` | List[str] |
| `--weight-loader-disable-mmap` | 在使用 safetensors 加载权重时禁用 mmap。 | `False` | bool flag (set to enable) |
| `--remote-instance-weight-loader-seed-instance-ip` | 用于从远程实例加载权重的 seed 实例的 ip。 | `None` | Type: str |
| `--remote-instance-weight-loader-seed-instance-service-port` | 用于从远程实例加载权重的 seed 实例的服务端口。 | `None` | Type: int |
| `--remote-instance-weight-loader-send-weights-group-ports` | 用于从远程实例加载权重的通信组端口。 | `None` | Type: JSON list |
| `--remote-instance-weight-loader-backend` | 从远程实例加载权重的后端。可以是 'transfer_engine' 或 'nccl'。默认为 'nccl'。 | `nccl` | `transfer_engine`, `nccl` |
| `--remote-instance-weight-loader-start-seed-via-transfer-engine` | 为 remote instance weight loader 通过 transfer engine 后端启动 seed server。 | `False` | bool flag (set to enable) |

## For PD-Multiplexing
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-pdmux` | 启用 PD-Multiplexing,PD 在 greenctx stream 上运行。 | `False` | bool flag (set to enable) |
| `--pdmux-config-path` | PD-Multiplexing 配置文件的路径。 | `None` | Type: str |
| `--sm-group-num` | sm 分区组的数量。 | `8` | Type: int |

## Configuration file support
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--config` | 从配置文件读取 CLI 选项。必须是包含配置选项的 YAML 文件。 | `None` | Type: str |

## For Multi-Modal
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--mm-max-concurrent-calls` | 异步 mm 数据处理的最大并发调用数。 | `32` | Type: int |
| `--mm-per-request-timeout` | 每个 multi-modal 请求的超时(秒)。 | `10.0` | Type: int |
| `--enable-broadcast-mm-inputs-process` | 在调度器中启用 broadcast mm-inputs 处理。 | `False` | bool flag (set to enable) |
| `--mm-process-config` | Multimodal 预处理配置,一个包含键 `image`、`video`、`audio` 的 json 配置。 | `{}` | Type: JSON / Dict |
| `--mm-enable-dp-encoder` | 为 mm encoder 启用 data parallelism。dp 大小将自动设置为 tp 大小。 | `False` | bool flag (set to enable) |
| `--limit-mm-data-per-request` | 限制每个请求的 multimodal 输入数量。例如 '{"image": 1, "video": 1, "audio": 1}' | `None` | Type: JSON / Dict |
| `--enable-mm-global-cache` | 在 encoder 服务器上启用基于 Mooncake 的全局 multimodal embedding 缓存,使重复图像可以复用缓存的 ViT embedding 而非重新计算。 | `False` | bool flag (set to enable) |

## For checkpoint decryption
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--decrypted-config-file` | 解密后的配置文件的路径。 | `None` | Type: str |
| `--decrypted-draft-config-file` | 解密后的 draft 配置文件的路径。 | `None` | Type: str |
| `--enable-prefix-mm-cache` | 启用 prefix multimodal cache。目前仅支持 mm-only。 | `False` | bool flag (set to enable) |

## Forward hooks
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--forward-hooks` | JSON 格式的 forward hook 规格列表。每个元素必须包含 `target_modules`(与 `model.named_modules()` 名称匹配的 glob 模式列表)和 `hook_factory`(工厂的 Python import 路径,例如 `my_package.hooks:make_hook`)。可选的 `name` 字段用于日志记录,可选的 `config` 对象会作为 `dict` 传递给工厂。 | `None` | Type: JSON list |

## Deprecated arguments
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-ep-moe` | NOTE: --enable-ep-moe is deprecated. Please set `--ep-size` to the same value as `--tp-size` instead. | `None` | N/A |
| `--enable-deepep-moe` | NOTE: --enable-deepep-moe is deprecated. Please set `--moe-a2a-backend` to 'deepep' instead. | `None` | N/A |
| `--prefill-round-robin-balance` | Note: Note: --prefill-round-robin-balance is deprecated now. | `None` | N/A |
| `--enable-flashinfer-cutlass-moe` | NOTE: --enable-flashinfer-cutlass-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_cutlass' instead. | `None` | N/A |
| `--enable-flashinfer-cutedsl-moe` | NOTE: --enable-flashinfer-cutedsl-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_cutedsl' instead. | `None` | N/A |
| `--enable-flashinfer-trtllm-moe` | NOTE: --enable-flashinfer-trtllm-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_trtllm' instead. | `None` | N/A |
| `--enable-triton-kernel-moe` | NOTE: --enable-triton-kernel-moe is deprecated. Please set `--moe-runner-backend` to 'triton_kernel' instead. | `None` | N/A |
| `--enable-flashinfer-mxfp4-moe` | NOTE: --enable-flashinfer-mxfp4-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_mxfp4' instead. | `None` | N/A |
| `--crash-on-nan` | 在出现 nan logprobs 时崩溃服务器。 | `False` | Type: str |
| `--hybrid-kvcache-ratio` | 在 [0,1] 范围内的混合比例,介于 uniform 和 hybrid kv 缓冲区之间(0.0 = 纯 uniform:swa_size / full_size = 1)(1.0 = 纯 hybrid:swa_size / full_size = local_attention_size / context_length) | `None` | Optional[float] |
| `--load-watch-interval` | 负载监视的间隔(秒)。 | `0.1` | Type: float |
| `--nsa-prefill` | 为 prefill 阶段选择 NSA 后端(在运行 DeepSeek NSA 风格 attention 时覆盖 `--attention-backend`)。 | `flashmla_sparse` | `flashmla_sparse`, `flashmla_decode`, `fa3`, `tilelang`, `aiter` |
| `--nsa-decode` | 在运行 DeepSeek NSA 风格 attention 时为 decode 阶段选择 NSA 后端。为解码覆盖 `--attention-backend`。 | `flashmla_kv` | `flashmla_prefill`, `flashmla_kv`, `fa3`, `tilelang`, `aiter` |
