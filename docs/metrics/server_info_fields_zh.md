# `/server_info` 字段对照表（基于运行代码自动生成）

> 服务：`127.0.0.1:30000`（`DeepSeek-V4.1-Flash`，TP8/EP8）。数据 = 实时 `/server_info` 返回 + 运行代码定义。
> 运行代码：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/dsv41-pd-dspark-20260911-control/main-20260926/baseline/sglang`
> `ServerArgs` 是 msgspec Struct，字段由 `srt/arg_groups/*` 模块化注册；`类型/默认/说明` 取自 CLI 参数定义，`当前值` 取自本次实时响应。

## 1. 顶层结构：字段从哪来

`/server_info` = `{**server_args.resolved_dict(), **scheduler_info, "startup_time", "internal_states", "version", "kv_events"}`

| 顶层键 | 来源 | 说明 |
| --- | --- | --- |
| 全部 ServerArgs 字段（517 项） | `server_args.resolved_dict()` | 解析后的启动参数，见 §3 |
| `status` | 运行态 | 服务就绪状态（调度器握手 scheduler_info） |
| `max_total_num_tokens` | 运行态 | KV/内存池可容纳的总 token 数（scheduler_info） |
| `max_req_input_len` | 运行态 | 单请求允许的最大输入长度（scheduler_info） |
| `startup_time` | 运行态 | 启动各阶段耗时（load_weight / kv_cache_allocation / scheduler_e2e / cuda_graph / tokenizer_e2e） |
| `version` | 运行态 | sglang 包版本（本服务 0.0.0） |
| `frontend` | 运行态 | 前端类型（python） |
| `kv_events` | 运行态 | KV 事件发布器（zmq）描述，供 KV 感知路由订阅；未启用为 null |
| `launch_command` | 运行态 | 启动命令行原始字符串（复现用） |
| `internal_states` | 运行态 | 每个 DP rank 调度器的内部状态快照（§4） |

## 2. 运行态 / 衍生字段

| 字段 | 类型 | 当前值 | 说明 |
| --- | --- | --- | --- |
| `status` | str | `ready` | 服务就绪状态（调度器握手 scheduler_info） |
| `max_total_num_tokens` | int | `39050752` | KV/内存池可容纳的总 token 数（scheduler_info） |
| `max_req_input_len` | int | `1048570` | 单请求允许的最大输入长度（scheduler_info） |
| `startup_time` | dict | `{"load_weight": 373.500333301723, "kv_cache_allocation": …` | 启动各阶段耗时（load_weight / kv_cache_allocation / scheduler_e2e / cuda_graph / tokenizer_e2e） |
| `version` | str | `0.0.0` | sglang 包版本（本服务 0.0.0） |
| `frontend` | str | `python` | 前端类型（python） |
| `kv_events` | dict|null | `null` | KV 事件发布器（zmq）描述，供 KV 感知路由订阅；未启用为 null |
| `launch_command` | str | `--trust-remote-code --model-path /mnt/dolphin…` | 启动命令行原始字符串（复现用） |
| `internal_states` | list[dict] | `[{"model_path": "/mnt/dolphinfs/ssd_pool/docker/user/hado…` | 每个 DP rank 调度器的内部状态快照（§4） |

## 3. ServerArgs 参数对照表（共 517 项，按功能分组）

### 模型与分词器（17）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `model_path` | str | `null` | `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-fr…` | The path of the model weights. This can be a local folder or a Hugging Face repo ID. |
| `tokenizer_path` | str | `null` | `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-fr…` | The path of the tokenizer. |
| `tokenizer_mode` | str | `auto` | `auto` | Tokenizer mode. 'auto' will use the fast tokenizer if available, and 'slow' will always use the slow tokenizer. |
| `tokenizer_backend` | str | `huggingface` | `huggingface` | Tokenizer backend. 'huggingface' uses the default HuggingFace tokenizers library, and 'fastokens' uses the fastokens library for faster tokenization. Requires the fastokens package to be installed. |
| `tokenizer_worker_num` | int | `1` | `1` | The worker num of the tokenizer manager. |
| `detokenizer_worker_num` | int | `1` | `1` | The worker num of the detokenizer manager. |
| `skip_tokenizer_init` | bool | `false` | `false` | If set, skip init tokenizer and pass input_ids in generate request. |
| `load_format` | str | `auto` | `auto` | The format of the model weights to load. "auto" will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available. "pt" will load the weights in the pytorch bin format. "safetensors" will load the weights in the safetensors format. "npcache" will load the weights in pytorch format and store a numpy cache to speed up the loading. "dummy" will initialize the weights with random values, which is mainly for profiling."gguf" will load the weights in the gguf format. "expert_pack" is experimental and loads only the validated DeepSeek-V4-Flash-0731 MXFP4 or text-only Kimi-K3 Q2_K GGUF model with routed experts stored in an SSD expert pack. "bitsandbytes" will load the weights using bitsandbytes quantization."layered" loads weights layer by layer so that one can quantize a layer before loading another to make the peak memory envelope smaller."presharded" performs a normal first-time load (with quantization), then dumps a per-rank/per-tensor sharded checkpoint with content deduplication into <model_path>/presharded/<parallelism+quant subfolder>/. Subsequent runs with the same parallelism+quantization config load directly from this presharded checkpoint and skip re-quantization. The dump directory must be on a shared filesystem across all ranks/nodes. Optional model_loader_extra_config roots: presharded_path (target) and draft_presharded_path (speculative draft); each replaces <model_path>/presharded and still gets a config subfolder appended. Use a writable path when model_path is read-only (e.g. HF cache mounts). |
| `model_loader_extra_config` | str | `{}` | `{}` | Extra config for model loader. This will be passed to the model loader corresponding to the chosen load_format. For load_format=presharded, JSON may include presharded_path (target cache root), draft_presharded_path (draft cache root), max_file_bytes, hash_num_threads, and verify_on_load. |
| `trust_remote_code` | bool | `false` | `true` | Whether or not to allow for custom models defined on the Hub in their own modeling files. |
| `context_length` | human_readable_int | `null` | `null` | The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).  Supports standard SI suffixes (k, M, G, T) and IEC suffixes     (Ki, Mi, Gi, Ti). Suffixes are case-sensitive.      Decimals are allowed for SI suffixes only.      Examples:         '1k' -> 1000      '1M' -> 1000000    '25.6k' -> 25600         '1Ki' -> 1024     '1Mi' -> 1048576 |
| `is_embedding` | bool | `false` | `false` | Whether to use a CausalLM as an embedding model. |
| `enable_multimodal` | bool | `null` | `null` | Enable the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen |
| `revision` | str | `null` | `null` | The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. |
| `model_impl` | str | `auto` | `auto` | Which implementation of the model to use.  * "auto" will try to use the SGLang implementation if it exists and fall back to the Transformers implementation if no SGLang implementation is available. * "sglang" will use the SGLang model implementation. * "transformers" will use the Transformers model * "mindspore" will use the MindSpore model implementation. |
| `model_config_parser` | str | `auto` | `auto` | Which model-config parser to use. "auto" picks "mistral" via the is_mistral_model name heuristic, else "hf" (AutoConfig over config.json). Plugins can register additional parsers via @register_model_config_parser. |
| `json_model_override_args` | str | `{}` | `{}` | A dictionary in JSON string format used to override default model configurations. |

### 量化与数据类型（12）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `dtype` | str | `auto` | `auto` | Data type for model weights and activations.  * "auto" will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models. * "half" for FP16. Recommended for AWQ quantization. * "float16" is the same as "half". * "bfloat16" for a balance between precision and range. * "float" is shorthand for FP32 precision. * "float32" for FP32 precision. |
| `quantization` | str | `null` | `null` | The quantization method. |
| `quantization_param_path` | nullable_str | `null` | `null` | Path to the JSON file containing the KV cache scaling factors. This should generally be supplied, when KV cache dtype is FP8. Otherwise, KV cache scaling factors default to 1.0, which may cause accuracy issues. |
| `kv_cache_dtype` | str | `auto` | `fp8_e4m3` | Data type for kv cache storage. "auto" will use model data type. "bf16" or "bfloat16" for BF16 KV cache. "fp8_e5m2" and "fp8_e4m3" are supported for CUDA 11.8+. "mxfp8" is supported by the FA4 backend. "nvfp4" selects the NVFP4 FP4 E2M1 KV cache recipe; "fp4_mx_block16" selects the MX-style block-size-16 FP4 E2M1 KV cache recipe. Both require CUDA 12.8+ and PyTorch 2.8.0+ |
| `enable_fp32_lm_head` | bool | `false` | `false` | If set, the LM head outputs (logits) are in FP32. |
| `modelopt_quant` | str | `null` | `null` | The ModelOpt quantization configuration. Supported values: 'fp8', 'int4_awq', 'w4a8_awq', 'nvfp4', 'nvfp4_awq'. This requires the NVIDIA Model Optimizer library to be installed: pip install nvidia-modelopt |
| `modelopt_checkpoint_restore_path` | str | `null` | `null` | Path to restore a previously saved ModelOpt quantized checkpoint. If provided, the quantization process will be skipped and the model will be loaded from this checkpoint. |
| `modelopt_checkpoint_save_path` | str | `null` | `null` | Path to save the ModelOpt quantized checkpoint after quantization. This allows reusing the quantized model in future runs. |
| `modelopt_export_path` | str | `null` | `null` | Path to export the quantized model in HuggingFace format after ModelOpt quantization. The exported model can then be used directly with SGLang for inference. If not provided, the model will not be exported. |
| `quantize_and_serve` | bool | `false` | `false` | Quantize the model with ModelOpt and immediately serve it without exporting. This is useful for development and prototyping. For production, it's recommended to use separate quantization and deployment steps. |
| `rl_quant_profile` | str | `null` | `null` | Path to the FlashRL quantization profile. Required when using --load-format flash_rl. |
| `enable_tf32_matmul` | bool | `false` | `false` | Enable float32 matmuls to use TensorFloat32 precision for better performance (via torch.set_float32_matmul_precision). CUDA only. |

### 显存与调度（33）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `mem_fraction_static` | float | `null` | `0.8` | The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors. |
| `max_running_requests` | int | `null` | `256` | The maximum number of running requests. |
| `max_queued_requests` | int | `null` | `null` | The maximum number of queued requests. This option is ignored when using disaggregation-mode. |
| `max_total_tokens` | human_readable_int | `null` | `null` | The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. This option is typically used for development and debugging purposes.  Supports standard SI suffixes (k, M, G, T) and IEC suffixes     (Ki, Mi, Gi, Ti). Suffixes are case-sensitive.      Decimals are allowed for SI suffixes only.      Examples:         '1k' -> 1000      '1M' -> 1000000    '25.6k' -> 25600         '1Ki' -> 1024     '1Mi' -> 1048576 |
| `chunked_prefill_size` | int | `null` | `8192` | The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill. |
| `prefill_decode_interval` | int | `null` | `0` | The number of decode rounds to run after a prefill batch before scheduling the next prefill. By default, this is disabled except for profiled Qwen3-VL serving configurations on Hopper. In data-parallel attention mode, the interval is synchronized across all DP ranks. Set to 0 to disable. |
| `enable_dynamic_chunking` | bool | `false` | `false` | Enable dynamic chunk size adjustment for pipeline parallelism. When enabled, chunk sizes are dynamically calculated based on fitted function to maintain consistent execution time across chunks. |
| `max_prefill_tokens` | human_readable_int | `16384` | `16384` | The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length.  Supports standard SI suffixes (k, M, G, T) and IEC suffixes     (Ki, Mi, Gi, Ti). Suffixes are case-sensitive.      Decimals are allowed for SI suffixes only.      Examples:         '1k' -> 1000      '1M' -> 1000000    '25.6k' -> 25600         '1Ki' -> 1024     '1Mi' -> 1048576 |
| `prefill_max_requests` | int | `null` | `null` | The maximum number of requests in a prefill batch. If not specified, there is no limit. |
| `schedule_policy` | str | `fcfs` | `fcfs` | The scheduling policy of the requests. |
| `enable_priority_scheduling` | bool | `false` | `false` | Enable priority scheduling. Requests with higher priority integer values will be scheduled first by default. |
| `disable_priority_preemption` | bool | `false` | `false` | Disable priority scheduling preemption. |
| `default_priority_value` | int | `null` | `null` | Default priority for requests without explicit priority. |
| `abort_on_priority_when_disabled` | bool | `false` | `false` | If set, abort requests that specify a priority when priority scheduling is disabled. |
| `schedule_low_priority_values_first` | bool | `false` | `false` | If specified with --enable-priority-scheduling, the scheduler will schedule requests with lower priority integer values first. |
| `priority_scheduling_preemption_threshold` | int | `10` | `10` | Minimum difference in priorities for an incoming request to have to preempt running request(s). |
| `retraction_policy` | str | `length` | `length` | The decode retraction policy to use when the KV cache is full. 'length' preserves the existing behavior and retracts short-output, long-input requests first. 'priority' retracts lower-priority requests first, using the same priority direction as priority scheduling. |
| `schedule_conservativeness` | float | `1` | `1` | How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently. |
| `page_size` | int | `null` | `256` | The number of tokens in a page. |
| `c128_page_size` | int | `16` | `16` | The physical page size of the NPU DSV4 C128 KV cache. Must be a positive multiple of 16. |
| `swa_full_tokens_ratio` | float | `null` | `0.8` | The ratio of SWA layer KV tokens / full layer KV tokens, regardless of the number of swa:full layers. It should be between 0 and 1. E.g. 0.5 means if each swa layer has 50 tokens, then each full layer has 100 tokens. |
| `disable_hybrid_swa_memory` | bool | `false` | `false` | Disable the hybrid SWA memory pool. |
| `radix_eviction_policy` | str | `lru` | `lru` | The eviction policy of radix trees. 'lru' stands for Least Recently Used, 'lfu' stands for Least Frequently Used, 'slru' stands for Segmented Least Recently Used, 'priority' evicts lower-priority requests first, and 'tlru' stands for Tail-Optimized LRU (arXiv:2510.15152), which evicts the part of a conversation that cannot affect tail TTFT before falling back to LRU. See https://docs.sglang.io/docs/advanced_features/radix_eviction_policy for what each policy optimizes for. |
| `radix_eviction_policy_config` | loads | `null` | `null` | Tuning parameters for --radix-eviction-policy, as a json object passed to the policy as keyword arguments. 'slru' takes protected_threshold (int, default 2), e.g. '{"protected_threshold": 4}'; 'tlru' takes threshold and next_prompt_estimate (ints, tokens), e.g. '{"threshold": 4096, "next_prompt_estimate": 512}'. An unrecognized key fails at startup, naming the key and the policy. See https://docs.sglang.io/docs/advanced_features/radix_eviction_policy#policy-parameters for the full parameter list. |
| `prefill_only_disable_kv_cache` | bool | `false` | `false` | Skip the physical KV cache allocation for embedding-mode prefill-only workloads. Currently only valid with --is-embedding, --chunked-prefill-size=-1, --disable-radix-cache, an FA prefill backend, and non-FP4 KV cache so the fa_skip_kv_cache path is active (no layer reads or writes the cache). Other prefill-only workloads such as scoring/MIS may benefit from this later once their attention paths stop using paged KV. Scheduler admission accounting is unchanged; per-layer K/V tensors are sized to (page_size, head_num, head_dim) placeholders so GPU memory is not wasted. |
| `disable_radix_cache` | bool | `false` | `false` | Disable RadixAttention for prefix caching. |
| `enable_page_major_kv_layout` | bool | `false` | `false` | Enable the page-major KV layout: lay out the Mamba state and full/SWA KV caches in a page-granularity envelope (page is the outermost axis, layer-major within a page) instead of the default per-layer (layer-major) layout. Requires the Triton attention / linear-attn / Mamba backends. |
| `enable_unified_memory` | bool | `false` | `false` | Replace the statically-partitioned hybrid-model pools (full-attn KV + SWA/Mamba state) with one byte buffer split dynamically between sub-pools. Requires the Triton attention / linear-attn / Mamba backends. Supported PD-disaggregation and speculative-decoding configurations are validated at startup. |
| `disable_chunked_prefix_cache` | bool | `false` | `false` | Disable chunked prefix cache feature for deepseek, which should save overhead for short sequences. |
| `disable_overlap_schedule` | bool | `false` | `false` | Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker. |
| `num_continuous_decode_steps` | int | `1` | `1` | Run multiple continuous decoding steps to reduce scheduling overhead. This can potentially increase throughput but may also increase time-to-first-token latency. The default value is 1, meaning only run one decoding step at a time. |
| `scheduler_recv_interval` | int | `1` | `1` | The interval to poll requests in scheduler. Can be set to >1 to reduce the overhead of this. |
| `enable_mixed_chunk` | bool | `false` | `false` | Enabling mixing prefill and decode in a batch when using chunked prefill. |

### 分布式拓扑与并行 (TP/PP/DP/CP)（33）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `nccl_port` | int | `null` | `29500` | The port for NCCL distributed environment setup. Defaults to a random port. |
| `dist_timeout` | int | `null` | `null` | Set timeout for torch.distributed initialization. |
| `dist_init_addr` | str | `null` | `null` | The host address for initializing distributed backend (e.g., `192.168.0.2:25000`). |
| `gated_launch_port` | int | `null` | `null` | The port of the gated launch control server. When set, every rank blocks right after the distributed environment is initialized, before any sizable GPU allocation, until `POST /gate/activate` is sent to this port on the host of the first rank. This lets an external orchestrator defer the memory hungry part of startup to a safe window. Defaults to None, which disables the gate. |
| `nnodes` | int | `1` | `1` | The number of nodes. |
| `node_rank` | int | `0` | `0` | The node rank. |
| `tp_size` | int | `1` | `8` | The tensor parallelism size. |
| `dcp_size` | int | `1` | `1` | The decode context parallelism size. |
| `pp_size` | int | `1` | `1` | The pipeline parallelism size. |
| `pp_max_micro_batch_size` | int | `null` | `null` | The maximum micro batch size in pipeline parallelism. |
| `pp_async_batch_depth` | int | `0` | `0` | The async batch depth of pipeline parallelism. |
| `dp_size` | int | `1` | `1` | The data parallelism size. |
| `load_balance_method` | str | `auto` | `round_robin` | The load balancing strategy for data parallelism. |
| `attn_cp_size` | int | `1` | `1` | The attention context parallelism size. |
| `moe_dp_size` | int | `1` | `1` | The moe data parallelism size. |
| `dwdp_size` | int | `1` | `1` | DWDP (Distributed Weight Data Parallelism) group size. When > 1, MoE prefill uses weight prefetch instead of token all-to-all. Must equal tp_size. Only supported with --disaggregation-mode null or prefill. |
| `dcp_comm_backend` | str | `null` | `ag_rs` | Communication backend for the decode context-parallel (DCP) attention reduction: 'ag_rs' (AllGather + ReduceScatter), 'a2a' (fused NCCL All-to-All exchange of output+LSE + local Triton LSE combine), or 'fi_a2a' (FlashInfer MNNVL All-to-All kernel; requires Blackwell and a DCP group within one MNNVL domain). Unset resolves to 'fi_a2a' where supported, else 'a2a' on CUDA/ROCm, else 'ag_rs'. |
| `dcp_replicate_q_proj` | bool | `null` | `null` | For MLA decode context parallelism with the a2a/fi_a2a backend: replicate the Q projection so each DCP rank computes the full-head query locally (redundant projection compute), eliminating the per-layer head-dim all-gather of Q. Trades a small amount of extra GEMM for one fewer collective per layer. Use --no-dcp-replicate-q-proj to disable the model-specific default. |
| `enable_prefill_cp` | bool | `false` | `false` | Enable context parallelism for the prefill phase. Select the layout with --cp-strategy. |
| `cp_strategy` | str | `null` | `null` | Sharding strategy for prefill CP. 'zigzag' assigns each rank one early and one late sequence block; 'interleave' assigns token indices modulo the CP size. |
| `enable_dsa_cache_layer_split` | bool | `false` | `false` | Split DSA (DeepSeek Sparse Attention) GPU KV/indexer cache layers across context-parallel ranks to reduce per-rank KV memory. Currently only supported with the mooncake transfer backend (mooncake / mooncake_tcp); mori/nixl support will be added later by the community. |
| `enable_cp_decode_attn_tp` | bool | `false` | `false` | Enable attention tensor-parallel weight slicing during decode under context parallel (cp_size>1). Slices the replicated attention linears to the local CP partition, eliminating redundant decode GEMMs. |
| `enable_dp_attention` | bool | `false` | `false` | Enabling data parallelism for attention and tensor parallelism for FFN. The dp size should be equal to the tp size. Currently DeepSeek-V2 and Qwen 2/3 MoE models are supported. |
| `enable_dp_attention_local_control_broadcast` | bool | `false` | `false` | With DP-attention, send control messages to every DP group leader and broadcast within attn_tp_group instead of the full tp_group. Eliminates a costly all-ranks gloo sync on every scheduler iteration. |
| `enable_dp_lm_head` | bool | `false` | `false` | Enable vocabulary parallel across the attention TP group to avoid all-gather across DP groups, optimizing performance under DP attention. |
| `enable_tp_lm_head_all_to_all` | bool | `null` | `false` | Use all-to-all instead of TP all-gather followed by DP scatter for the TP-sharded LM head under DP attention. By default this is enabled only on decode-only PD nodes with pure DP attention (tp_size == dp_size > 1 and attn_cp_size == 1), and disabled on prefill-only and colocated nodes. Pass --no-enable-tp-lm-head-all-to-all to opt out. The path is incompatible with --enable-dp-lm-head; batches without an equal padded row count fall back to the existing all-gather path. |
| `enable_attn_tp_input_scattered` | bool | `false` | `false` | Allow input of attention to be scattered when only using tensor parallelism, to reduce the computational load of operations such as qkv latent. |
| `enable_shared_experts_attn_tp` | bool | `false` | `false` | Shard shared expert weights across the attention TP group when using an expert-parallel all-to-all backend. |
| `shared_experts_tp_size` | int | `null` | `null` | Shared-expert TP size for Kimi-K3 with an expert-parallel all-to-all backend. Must divide attention TP size. Overrides --enable-shared-experts-attn-tp when set; 1 replicates the weights. |
| `enable_dense_mlp_attn_tp` | bool | `false` | `false` | Shard dense MLP weights across the attention TP group under DP attention. |
| `enable_layernorm_sp` | bool | `false` | `false` | Enable Megatron-style sequence parallelism (arXiv:2205.05198) for the LayerNorm/residual regions under pure tensor parallelism: the row-parallel all-reduce becomes reduce-scatter + all-gather, so LayerNorm runs on sequence-sharded activations with no extra communication volume. Prefill only; Qwen3 dense; requires tp_size > 1 and NVLink/NVSwitch. |
| `disable_attn_tp_gather` | bool | `false` | `false` | Disable scheduler-side attn_tp_gather (the upstream SP path that pads num_tokens to attn_tp_size and pre-allocates a gathered buffer). Use for models that manage SP scatter/gather at the model level (e.g., perform their own all_gather/reduce_scatter inside attention) and do not consume the upstream gathered_buffer. Without this, the cuda graph runner pads num_tokens to attn_tp_size, which can cause kernel autotuners to select wrong-sized variants at small batches. |
| `enable_p2p_check` | bool | `false` | `false` | Enable P2P check for GPU access, otherwise the p2p access is allowed by default. |

### 设备信息与服务超时（12）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `device` | str | `null` | `cuda` | The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu', 'musa'). Defaults to auto-detection if not specified. |
| `base_gpu_id` | int | `0` | `0` | The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine. |
| `gpu_id_step` | int | `1` | `1` | The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,... |
| `random_seed` | int | `null` | `646881084` | The random seed. |
| `mlx_enable_sampling` | bool | `false` | `false` | MLX backend only: sample decode tokens (temperature / top-k / top-p / min-p) instead of greedy argmax. Sampling runs inside the lazy MLX graph, so it works with the overlap scheduler; first tokens from prefill/extend are sampled too. Greedy requests keep exact argmax behavior. Also enables on the MLX path: grammar vocab masks and custom logit processors (these break decode chaining per step; custom processors run on pure-decode steps only), logit_bias, output logprobs (sampled token / top-k / token_ids; prompt input logprobs are not computed), NaN sanitization (SGLANG_SANITIZE_NAN_LOGITS), and per-request sampling_seed under --enable-deterministic-inference (deterministic within MLX only). Penalties are not applied. |
| `watchdog_timeout` | float | `300` | `300` | Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging. |
| `soft_watchdog_timeout` | float | `null` | `null` | Set soft watchdog timeout in seconds. If a forward batch takes longer than this, the server will dump information for debugging. |
| `sleep_on_idle` | bool | `false` | `false` | Reduce CPU usage when sglang is idle. |
| `use_ray` | bool | `false` | `false` | Use Ray actors for scheduler process management. |
| `custom_sigquit_handler` | NoneType | — | `null` | （无 CLI 定义） |
| `numa_node` | int | `null` | `null` | Sets the numa node for the subprocesses. i-th element corresponds to i-th subprocess. If unset, will be automatically detected on NUMA systems. |
| `gc_threshold` | int | `null` | `null` | Set the garbage collection thresholds (the collection frequency). Accepts 1 to 3 integers. |

### HTTP 服务（14）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `host` | str | `127.0.0.1` | `127.0.0.1` | The host of the HTTP server. |
| `port` | int | `30000` | `30000` | The port of the HTTP server. |
| `fastapi_root_path` | str | `""` | `""` | App is behind a path based routing proxy. |
| `smg_grpc_mode` | bool | `false` | `false` | Use the legacy SMG gRPC server (smg-grpc-servicer) instead of the HTTP server. Replaces the deprecated --grpc-mode. |
| `grpc_mode` | bool | `false` | `false` | (Deprecated, use --smg-grpc-mode) Legacy SMG gRPC server selector. |
| `grpc_port` | int | `null` | `null` | Port for the native gRPC server, started alongside HTTP. Setting this (or SGLANG_GRPC_PORT) enables the native gRPC server; it is off by default. In legacy --smg-grpc-mode this is the SMG server port and defaults to --port + 10000. |
| `grpc_worker_threads` | int | — | `4` | （无 CLI 定义） |
| `sidecar` | str | `null` | `null` | Start a locally managed sidecar against the native gRPC server. The selected module must expose main(argv) and read the resolved native gRPC endpoint from SGLANG_GRPC_ENDPOINT. Requires --grpc-port or SGLANG_GRPC_PORT. |
| `sidecar_args` | json_list_type | `null` | `null` | JSON array passed to the selected sidecar module's main(argv) function. --sidecar-shutdown-timeout SECONDS is consumed by SGLang. |
| `skip_server_warmup` | bool | `false` | `true` | If set, skip warmup. |
| `warmups` | str | `null` | `null` | Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests |
| `enable_http2` | bool | `false` | `false` | Use Granian instead of Uvicorn as the ASGI server, enabling HTTP/1.1 and HTTP/2 auto-negotiation. Clients may use h2c (cleartext HTTP/2) or plain HTTP/1.1. Requires 'pip install sglang[http2]'. |
| `http2_max_concurrent_streams` | int | `200` | `200` | Maximum number of concurrent streams advertised on each HTTP/2 connection (1 to 2^32 - 1). Only applies with --enable-http2. |
| `http2_initial_connection_window_size` | int | `1048576` | `1048576` | Initial connection-level HTTP/2 receive window in bytes (1024 to 2^31 - 1). Only applies with --enable-http2. |

### SSL / TLS（5）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `ssl_keyfile` | str | `null` | `null` | The file path to the SSL key file. |
| `ssl_certfile` | str | `null` | `null` | The file path to the SSL certificate file. |
| `ssl_ca_certs` | str | `null` | `null` | The CA certificates file. |
| `ssl_keyfile_password` | str | `null` | `null` | The password to decrypt the SSL keyfile. |
| `enable_ssl_refresh` | bool | `false` | `false` | Enable automatic SSL certificate hot-reloading when cert/key files change on disk. Requires --ssl-certfile and --ssl-keyfile. |

### API、模板与工具调用（23）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `api_key` | str | `null` | `null` | Set API key of the server. It is also used in the OpenAI API compatible server. |
| `admin_api_key` | str | `null` | `null` | Set admin API key for sensitive management endpoints (e.g. /hicache/storage-backend/clear). When set, admin endpoints require this key and do NOT accept --api-key. |
| `served_model_name` | str | `null` | `deepseek-ai/DeepSeek-V4.1-Flash` | Override the model name returned by the v1/models endpoint in OpenAI API server. |
| `weight_version` | str | `default` | `default` | Version identifier for the model weights. Defaults to 'default' if not specified. |
| `chat_template` | str | `null` | `null` | The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server. |
| `trust_request_chat_template` | bool | `false` | `false` | Allow a request to override the server chat template via its chat_template_kwargs. Off by default for safety. |
| `hf_chat_template_name` | str | `null` | `null` | When the HuggingFace tokenizer has multiple chat templates (e.g., 'default', 'tool_use', 'rag'), specify which named template to use. If not set, the first available template is used. |
| `completion_template` | str | `null` | `null` | The buliltin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently. |
| `file_storage_path` | str | `sglang_storage` | `sglang_storage` | The path of the file storage in backend. |
| `enable_cache_report` | bool | `false` | `false` | Return number of cached tokens in usage.prompt_tokens_details for each openai request. |
| `return_input_ids` | bool | `false` | `false` | Return prompt (input) token ids on the response-level sglext extension for every chat completion request, as if return_input_ids_in_sglext were set on the request. |
| `return_output_ids` | bool | `false` | `false` | Return sampled output token ids on the response-level sglext extension for every chat completion request, as if return_output_ids_in_sglext were set on the request. |
| `reasoning_parser` | str | `null` | `deepseek-v41` | Specify the parser for reasoning models. Use 'auto' to detect from chat template. Options include: ['apertus2509', 'deepseek-r1', 'deepseek-v3', 'deepseek-v4', 'deepseek-v41', 'dots', 'glm45', 'ling3', 'hunyuan', 'gpt-oss', 'k2_horizon', 'kimi', 'kimi_k2', 'kimi_k3', 'mimo', 'muse', 'poolside_v1', 'qwen3', 'qwen3-thinking', 'minimax', 'minimax-append-think', 'minimax-m3', 'nanbeige', 'step3', 'step3p5', 'mistral', 'nemotron_3', 'granite_thinking_parser', 'interns1', 'gemma4', 'inkling', 'cohere_command4', 'gigachat35']. |
| `default_chat_template_kwargs` | loads | `null` | `null` | Default chat template kwargs applied to every request when not overridden per-request. Keys must match what the model's chat template expects (e.g. enable_thinking, thinking, reasoning_effort). Per-request chat_template_kwargs takes precedence. |
| `strip_thinking_cache` | bool | `false` | `false` | Skip caching reasoning-model output (thinking + answer) in the radix tree on finish; keep only the prompt prefix. Opt-in: changes cache contents. |
| `enable_strict_thinking` | bool | `false` | `false` | Enable strict token filtering during the thinking phase. Blocks model-specific excluded tokens (e.g., tool call markers) during reasoning. Requires a grammar backend that supports token filtering. |
| `tool_call_parser` | str | `null` | `deepseekv41` | Specify the parser for handling tool-call interactions. Use 'auto' to detect from chat template. Options include: ['apertus2509', 'cohere_command4', 'deepseekv3', 'deepseekv31', 'deepseekv32', 'deepseekv4', 'deepseekv41', 'dots', 'glm', 'glm45', 'glm47', 'gpt-oss', 'k2_horizon', 'kimi_k2', 'kimi_k3', 'lfm2', 'ling3', 'llama3', 'mimo', 'minicpm5', 'mistral', 'muse', 'poolside_v1', 'pythonic', 'qwen', 'qwen25', 'qwen3_coder', 'spark25', 'step3', 'step3p5', 'minimax-m2', 'minimax-m3', 'nanbeige', 'trinity', 'interns1', 'hermes', 'hunyuan', 'gigachat3', 'gigachat35', 'gemma4', 'inkling']. |
| `tool_server` | str | `null` | `null` | Either 'demo' or a comma-separated list of tool server urls to use for the model. If not specified, no tool server will be used. |
| `sampling_defaults` | str | `model` | `model` | Where to get default sampling parameters. 'openai' uses SGLang/OpenAI defaults (temperature=1.0, top_p=1.0, etc.). 'model' uses the model's generation_config.json to get the recommended sampling parameters if available. Default is 'model'. |
| `asr_max_buffer_seconds` | int | `60` | `60` | Maximum seconds of PCM audio the streaming ASR WebSocket handler will accumulate before closing the session with a buffer_overflow error. Guards against OOM when a client streams audio faster than inference can consume it. Default 60s. |
| `asr_max_concurrent_sessions` | int | `32` | `32` | Maximum number of concurrent realtime ASR WebSocket sessions served by /v1/realtime. New connections beyond this cap are accepted, sent an error{code:too_many_sessions} frame, and closed. Default 32. |
| `preferred_sampling_params` | loads | `null` | `null` | json-formatted sampling settings that will be returned in /get_model_info |
| `allow_auto_truncate` | bool | `false` | `false` | Allow automatically truncating requests that exceed the maximum input length instead of returning an error. |

### 流式输出（6）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `stream_interval` | int | `1` | `1` | The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher |
| `batch_notify_size` | int | `16` | `16` | Number of streaming notifications to batch before yielding to the event loop. Reduces asyncio wakeup overhead under high concurrency. |
| `stream_response_default_include_usage` | bool | `false` | `false` | Include usage in every streaming response (even when stream_options is not specified). |
| `incremental_streaming_output` | bool | `false` | `false` | Whether to output as a sequence of disjoint segments. |
| `enable_streaming_session` | bool | `false` | `false` | Enable streaming session mode and StreamingSession wrapper. |
| `enable_session_radix_cache` | bool | `false` | `false` | Track per-session references on UnifiedRadixCache KV: eviction consumes unreferenced entries before referenced ones, and closing a session only dereferences its KV. |

### 日志 / 指标 / 追踪（36）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `log_level` | str | `info` | `info` | The logging level of all loggers. |
| `log_level_http` | str | `null` | `null` | The logging level of HTTP server. If not set, reuse --log-level by default. |
| `log_requests` | bool | `false` | `false` | Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level |
| `log_requests_level` | int | `2` | `2` | 0: Log metadata (no sampling parameters). 1: Log metadata and sampling parameters. 2: Log metadata, sampling parameters and partial input/output. 3: Log every input/output. |
| `log_requests_format` | str | `text` | `text` | Format for request logging: 'text' (human-readable) or 'json' (structured) |
| `log_requests_target` | str | `null` | `null` | Target(s) for request logging: 'stdout' and/or directory path(s) for file output. Can specify multiple targets, e.g., '--log-requests-target stdout /my/path'. |
| `uvicorn_access_log_exclude_prefixes` | str | `[]` | `[]` | Exclude uvicorn access logs whose request path starts with any of these prefixes. Defaults to empty (disabled). Example: --uvicorn-access-log-exclude-prefixes /metrics /health |
| `crash_dump_folder` | str | `null` | `null` | Folder path to dump requests from the last 5 min before a crash (if any). If not specified, crash dumping is disabled. |
| `show_time_cost` | bool | `false` | `false` | Show time cost of custom marks. |
| `enable_metrics` | bool | `false` | `true` | Enable log prometheus metrics. |
| `smg_http_sidecar_port` | int | `null` | `null` | Port for the HTTP sidecar server in legacy SMG gRPC mode (--smg-grpc-mode). Serves Prometheus metrics and profiling endpoints. Defaults to --port + 1. Not used in HTTP mode. |
| `enable_mfu_metrics` | bool | `false` | `false` | Enable estimated MFU-related prometheus metrics. |
| `enable_metrics_for_all_schedulers` | bool | `false` | `false` | Enable --enable-metrics-for-all-schedulers when you want schedulers on all TP ranks (not just TP 0) to record request metrics separately. This is especially useful when dp_attention is enabled, as otherwise all metrics appear to come from TP 0. |
| `load_snapshot_publish_interval` | int | `15` | `15` | Publish load snapshot to shared memory every N decode iterations. Prefill and idle always publish immediately. |
| `tokenizer_metrics_custom_labels_header` | str | `x-custom-labels` | `x-custom-labels` | Specify the HTTP header for passing custom labels for tokenizer metrics. |
| `tokenizer_metrics_allowed_custom_labels` | str | `null` | `null` | The custom labels allowed for tokenizer metrics. The labels are specified via a dict in '--tokenizer-metrics-custom-labels-header' field in HTTP requests, e.g., {'label1': 'value1', 'label2': 'value2'} is allowed if '--tokenizer-metrics-allowed-custom-labels label1 label2' is set. |
| `extra_metric_labels` | loads | `null` | `null` | The custom labels for metrics. e.g. '{"label1": "value1", "label2": "value2"}' |
| `bucket_time_to_first_token` | float | `null` | `null` | The buckets of time to first token, specified as a list of floats. |
| `bucket_inter_token_latency` | float | `null` | `null` | The buckets of inter-token latency, specified as a list of floats. |
| `bucket_e2e_request_latency` | float | `null` | `null` | The buckets of end-to-end request latency, specified as a list of floats. |
| `prompt_tokens_buckets` | str | `null` | `null` | The buckets rule of prompt tokens. Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> <value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500'). |
| `generation_tokens_buckets` | str | `null` | `null` | The buckets rule for generation tokens histogram. Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> <value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500'). |
| `gc_warning_threshold_secs` | float | `0` | `0` | The threshold for long GC warning. If a GC takes longer than this, a warning will be logged. Set to 0 to disable. |
| `decode_log_interval` | int | `40` | `40` | The log and metrics reporting interval (in decode iterations) for decode batches. |
| `enable_request_time_stats_logging` | bool | `false` | `false` | Enable per request time stats logging |
| `kv_events_config` | str | `null` | `null` | Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used. Runtime-load publishing for load-aware routers is a separate opt-in; see --load-publish-endpoint. |
| `load_publish_endpoint` | str | `null` | `null` | Opt in to the runtime-load PUB socket that load-aware routers subscribe to. Off by default (unset or 'off'). Use 'auto' to reserve the dp_size ports packed after the --kv-events-config range, or a wildcard-host TCP address (e.g. tcp://*:6000) to place it explicitly; rank r binds port+r and /server_info advertises the base under the kv_events block. Requires --kv-events-config to describe a publisher (routers discover the base through /server_info); startup fails if this is set without one, is not bindable, or overlaps the KV range. Note: 'auto' reserves 2*dp_size ports from the KV base — space co-hosted engines accordingly. The router-facing update cadence follows --load-snapshot-publish-interval (shared to avoid double-collecting the snapshot), so a large value there also staleness-caps this feed. |
| `enable_forward_pass_metrics` | bool | `false` | `false` | Enable per-iteration forward pass metrics via ZMQ IPC. External consumers (e.g. Dynamo planner) subscribe to the IPC endpoint exposed in server_args.forward_pass_metrics_ipc_name. |
| `forward_pass_metrics_worker_id` | str | `""` | `""` | ==SUPPRESS== |
| `forward_pass_metrics_ipc_name` | str | `null` | `null` | ==SUPPRESS== |
| `enable_trace` | bool | `false` | `true` | Enable opentelemetry trace |
| `trace_modules` | str | `request` | `request` | Select the components to trace. Available options are 'request' and 'mooncake'. Format: <module1 name>,<module2 name>,... |
| `otlp_traces_endpoint` | str | `localhost:4317` | `127.0.0.1:4317` | Config opentelemetry collector endpoint if --enable-trace is set. format: <ip>:<port> |
| `export_metrics_to_file` | bool | `false` | `false` | Export performance metrics for each request to local file (e.g. for forwarding to external systems). |
| `export_metrics_to_file_dir` | str | `null` | `null` | Directory path for writing performance metrics files (required when --export-metrics-to-file is enabled). |
| `stat_loggers` | NoneType | — | `null` | （无 CLI 定义） |

### 约束解码（3）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `constrained_json_whitespace_pattern` | str | `null` | `null` | (outlines and llguidance backends only) Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model generate consecutive whitespaces, set the pattern to [ 	 ]* |
| `constrained_json_disable_any_whitespace` | bool | `false` | `false` | (xgrammar and llguidance backends only) Enforce compact representation in JSON constrained output. |
| `constrained_json_max_whitespace_cnt` | int | `null` | `null` | (xgrammar backend only) Max consecutive whitespace chars allowed in JSON constrained output. None means unbounded. |

### 算子后端 (Kernel backend)（19）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `attention_backend` | str | `null` | `dsv4` | Choose the kernels for attention layers. |
| `decode_attention_backend` | str | `null` | `null` | Choose the kernels for decode attention layers (have priority over --attention-backend). |
| `enable_lean_attention` | bool | `null` | `null` | Enable Lean (Work-Centric) Attention decode kernel for long-context serving. When None (default), uses auto-gate that activates Lean for long contexts and falls back to standard kernel for short contexts. Set to True to force enable, False to force disable. |
| `prefill_attention_backend` | str | `null` | `null` | Choose the kernels for prefill attention layers (have priority over --attention-backend). |
| `sampling_backend` | str | `null` | `flashinfer` | Choose the kernels for sampling layers. |
| `grammar_backend` | str | `null` | `xgrammar` | Choose the backend for grammar-guided decoding. |
| `radix_cache_backend` | str | `null` | `null` | Name of a radix-cache backend previously registered via register_radix_cache_backend. Omit this flag to use the built-in default cache selection chain. |
| `mm_attention_backend` | str | `null` | `null` | Set multimodal attention backend. |
| `fp8_gemm_runner_backend` | str | `auto` | `auto` | Choose the runner backend for Blockwise FP8 GEMM operations. Options: 'auto' (default, auto-selects based on hardware; MXFP8 dense picks flashinfer_cutedsl on SM100/SM103 and FlashInfer CUTLASS on other supported Blackwell GPUs), 'deep_gemm' (JIT-compiled; enabled by default on NVIDIA Hopper (SM90) and Blackwell (SM100) when DeepGEMM is installed), 'flashinfer_trtllm' (optimal for Blackwell and low-latency), 'flashinfer_cutlass' (FlashInfer CUTLASS groupwise FP8 GEMM), 'flashinfer_cutedsl' (FlashInfer CuTe DSL MXFP8 GEMM on SM100/SM103), 'flashinfer_deepgemm' (Hopper SM90 only; uses swapAB optimization for small M dimensions in decoding), 'cutlass' (optimal for SM120 GPUs), 'triton' (fallback, widely compatible), 'aiter' (ROCm only). |
| `fp4_gemm_runner_backend` | str | `auto` | `auto` | Choose the runner backend for NVFP4 GEMM operations. Options: 'auto' (default; selects flashinfer_cutedsl on SM100, marlin on SM80-SM90, flashinfer_cutlass otherwise (including SM120)), 'flashinfer_cutlass' (FlashInfer CUTLASS backend), 'flashinfer_cudnn' (FlashInfer cuDNN backend, optimal on CUDA 13+ with cuDNN 9.15+), 'flashinfer_cutedsl' (FlashInfer CuTe DSL backend), 'flashinfer_trtllm' (FlashInfer TensorRT-LLM backend, requires different weight preparation with shuffling), 'marlin' (weight-only W4A16 fallback for SM80+). |
| `bf16_gemm_backend` | str | `auto` | `auto` | Choose the backend for unquantized BF16 GEMM operations. Options: 'auto' (default; selects 'cutedsl' on SM10x GPUs, except deterministic inference selects 'torch'; otherwise uses cuBLAS via torch.nn.functional.linear), 'cutedsl' (SGLang JIT CuTe DSL TGV BF16 GEMM on SM10x; dispatches between the allowlisted low-M Split-K kernel, the CuTe DSL kernel, and cuBLAS; set SGLANG_ENABLE_BF16_SPLITK_GEMM=0 to disable Split-K), 'gemv', 'torch' (always uses cuBLAS via torch.nn.functional.linear). |
| `dsa_prefill_backend` | str | `null` | `null` | DSA (DeepSeek Sparse Attention) prefill backend. If not specified, auto-detects based on hardware and kv_cache_dtype. |
| `dsv4_prefill_backend` | str | `auto` | `auto` | DeepSeek-V4 sparse prefill backend. 'auto' and 'flashmla_sparse' use the existing BF16 sparse prefill path; 'flashmla_sparse_q8' enables the Q8KV8 sparse prefill path. |
| `dsa_decode_backend` | str | `null` | `null` | DSA (DeepSeek Sparse Attention) decode backend. If not specified, auto-detects based on hardware and kv_cache_dtype. |
| `dsa_paged_mqa_logits_backend` | str | `auto` | `auto` | DSA indexer paged MQA logits kernel backend. Options: 'auto' (default; DeepGEMM on CUDA, aiter on ROCm), 'deepgemm', 'cutedsl' (CuTe DSL kernel, SM 100 (Blackwell) only; wins at low batch size and long context), 'aiter' (ROCm only). |
| `dsa_topk_backend` | str | `sgl-kernel` | `sgl-kernel` | DSA indexer top-k backend for the target model. Options: 'sgl-kernel', 'torch', 'flashinfer'. The 'torch' backend currently requires SGLANG_DSA_FUSE_TOPK=false. |
| `disable_flashinfer_autotune` | bool | `false` | `false` | Disable FlashInfer autotuning. |
| `flashinfer_autotune_skip_ops` | str | `null` | `null` | FlashInfer custom-op identifiers to skip during autotuning. Skipped ops use FlashInfer's heuristic fallback. SGLang temporarily skips mxfp8_gemm by default due to an IMA. |
| `mamba_backend` | str | `triton` | `triton` | Choose the kernel backend for Mamba SSM operations. Default is 'triton'. Options: 'triton' (default), 'flashinfer' (requires FlashInfer with Mamba support). |

### CUDA Graph（16）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `cuda_graph_config` | parse_cuda_graph_config_arg | `null` | `{"decode": {"backend": "full", "max_bs": 64, "bs": [1, 2,…` | Per-phase CUDA graph settings as JSON, e.g. '{"decode":{"backend":"full","max_bs":256},"prefill":{"backend":"tc_piecewise","tc_compiler":"eager"}}'. Allowed backends per phase: full, breakable, tc_piecewise, disabled (full is decode-only). JSON wins over the per-phase --cuda-graph-* convenience flags and over legacy flags. |
| `cuda_graph_backend_decode` | str | `null` | `null` | Backend for the decode phase. Folds into cuda_graph_config[decode].backend. |
| `cuda_graph_backend_prefill` | str | `null` | `null` | Backend for the prefill phase. Folds into cuda_graph_config[prefill].backend. |
| `cuda_graph_max_bs_decode` | int | `null` | `64` | Maximum batch size captured for the decode cuda graph. |
| `cuda_graph_max_bs_prefill` | int | `null` | `null` | Maximum batch size captured for the prefill cuda graph. |
| `cuda_graph_bs_decode` | int | `null` | `null` | Explicit list of batch sizes to capture for the decode cuda graph. |
| `cuda_graph_bs_prefill` | int | `null` | `null` | Explicit list of batch sizes to capture for the prefill cuda graph. |
| `cuda_graph_prefill_max_context` | human_readable_int | `null` | `null` | Maximum context length supported by DeepSeek-V4 breakable/full prefill CUDA graphs. Context-shaped attention metadata and indexer logits are allocated at this fixed size instead of the model maximum. Larger live contexts fall back to eager.  Supports standard SI suffixes (k, M, G, T) and IEC suffixes     (Ki, Mi, Gi, Ti). Suffixes are case-sensitive.      Decimals are allowed for SI suffixes only.      Examples:         '1k' -> 1000      '1M' -> 1000000    '25.6k' -> 25600         '1Ki' -> 1024     '1Mi' -> 1048576 |
| `cuda_graph_tc_compiler` | str | `null` | `null` | Compiler used by the tc_piecewise backend (currently only the prefill phase consumes it). |
| `disable_prefill_cuda_graph` | bool | `false` | `false` | Disable the prefill-phase CUDA graph. Convenience for --cuda-graph-backend-prefill=disabled. |
| `disable_decode_cuda_graph` | bool | `false` | `false` | Disable the decode-phase CUDA graph. Convenience for --cuda-graph-backend-decode=disabled. |
| `disable_cuda_graph` | bool | `false` | `false` | Deprecated. Use --cuda-graph-backend-{decode,prefill}=disabled instead. |
| `disable_cuda_graph_padding` | bool | `false` | `false` | Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed. |
| `enable_profile_cuda_graph` | bool | `false` | `false` | Enable profiling of cuda graph capture. |
| `enable_cudagraph_gc` | bool | `false` | `false` | Enable garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process. |
| `debug_cuda_graph` | bool | `false` | `false` | Enable debug/eager mode for CUDA graph using breakable CUDA graph. When enabled, graph breaks are inserted so every operation runs eagerly while still going through the CUDA graph capture / replay path. Useful for debugging CUDA graph capture / replay issues. |

### 通信与算子开关（21）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `enable_layerwise_nvtx_marker` | bool | `false` | `false` | Enable layerwise NVTX profiling annotations for the model. |
| `enable_nccl_nvls` | bool | `false` | `false` | Enable NCCL NVLS for prefill heavy requests when available. |
| `enable_symm_mem` | bool | `false` | `false` | Enable NCCL symmetric memory for fast collectives. |
| `triton_attention_reduce_in_fp32` | bool | `false` | `false` | Cast the intermediate attention results to fp32 to avoid possible crashes related to fp16.This only affects Triton attention kernels. |
| `triton_attention_num_kv_splits` | int | `8` | `8` | The number of KV splits in flash decoding Triton kernel. Larger value is better in longer context scenarios. The default value is 8. |
| `triton_attention_split_tile_size` | int | `null` | `null` | The size of split KV tile in flash decoding Triton kernel. Used for deterministic inference. |
| `flashinfer_mla_disable_ragged` | bool | `false` | `false` | Not using ragged prefill wrapper when running flashinfer mla |
| `enable_fused_qk_norm_rope` | bool | `false` | `false` | Enable fused qk normalization and rope rotary embedding. |
| `enable_precise_embedding_interpolation` | bool | `false` | `false` | Enable corner alignment for resize of embeddings grid to ensure more accurate(but slower) evaluation of interpolated embedding values. |
| `enable_fused_moe_sum_all_reduce` | bool | `false` | `false` | Enable fused moe triton and sum all reduce. |
| `enable_deepseek_v4_fp4_indexer` | bool | `false` | `false` | Enable the experimental FP4 C4 indexer path for DeepSeek V4. Default keeps the existing indexer implementation. |
| `disable_custom_all_reduce` | bool | `false` | `false` | Disable the custom all-reduce kernel and fall back to NCCL. |
| `enable_mscclpp` | bool | `false` | `false` | Enable MSCCL++ for tuned AllReduce and AllGather messages, with NCCL fallback. |
| `enable_torch_symm_mem` | bool | `false` | `false` | Enable using torch symm mem for all-reduce kernel and fall back to NCCL. Only supports CUDA device SM90 and above. SM90 supports world size 4, 6, 8. SM100 supports world size 6, 8. |
| `enable_scattered_sconv` | bool | `false` | `false` | Inkling: replace the attention/MLP output all-reduce with a hidden-dimension reduce-scatter, run the channelwise output short convolution on the [T, H/P] shard, then all-gather before the residual add. This shards the convolution cache across tensor-parallel ranks without changing communication volume. |
| `pre_warm_nccl` | bool | `false` | `false` | Pre-warm NCCL/RCCL communicators during startup to reduce P99 TTFT cold-start latency. Default: enabled for AMD/HIP (RCCL), disabled for NVIDIA/CUDA (NCCL). |
| `enable_quant_communications` | bool | `false` | `false` | Enable INT8 quantization of TP communications (limited support). |
| `enable_flashinfer_allreduce_fusion` | bool | `false` | `false` | (Deprecated: use --flashinfer-allreduce-fusion-backend=auto) Enable FlashInfer allreduce fusion with Residual RMSNorm. |
| `enforce_disable_flashinfer_allreduce_fusion` | bool | `false` | `false` | Enforce disable FlashInfer allreduce fusion. |
| `flashinfer_allreduce_fusion_backend` | str | `null` | `auto` | Enable FlashInfer allreduce fusion and choose backend. Requires SM90 or SM10X NVIDIA GPUs. Defaults to auto. 'auto': choose mnnvl on Blackwell (SM100/SM103) systems (single- and multi-node) and trtllm on SM90 single-node systems. 'trtllm': available on single-node systems only. 'mnnvl': available on SM90 single-node systems and SM100/SM103 single-node or multi-node systems via MNNVL fabric. 'cutedsl': Blackwell-only bf16 MNNVL CuTe DSL backend; also fuses the MoE finalize and the shared-expert add into the collective when the MoE runner can defer them. Fuses allreduce with Residual + RMSNorm for supported MoE models. |
| `enable_aiter_allreduce_fusion` | bool | `false` | `false` | Enable Aiter AllReduce Fusion. |

### Torch Compile（3）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `enable_torch_compile` | bool | `false` | `false` | Optimize the model with torch.compile. Experimental feature. |
| `enable_torch_compile_debug_mode` | bool | `false` | `false` | Enable debug mode for torch compile |
| `torch_compile_max_bs` | int | `32` | `32` | Set the maximum batch size when using torch compile. |

### 投机解码（35）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `speculative_algorithm` | str | `null` | `null` | Speculative algorithm. Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK, UNO. Or any name registered via `SpeculativeAlgorithm.register`. |
| `uno_lora_path` | str | `null` | `null` | Path to the UNO draft LoRA checkpoint. |
| `speculative_draft_model_path` | str | `null` | `null` | The path of the draft model weights. This can be a local folder or a Hugging Face repo ID. |
| `speculative_draft_model_revision` | str | `null` | `null` | The specific draft model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. |
| `speculative_draft_load_format` | str | `null` | `null` | The format of the draft model weights to load. If not specified, will use the same format as --load-format. Use 'dummy' to initialize draft model weights with random values for profiling. |
| `speculative_num_steps` | int | `null` | `null` | The number of steps sampled from draft model in Speculative Decoding. |
| `speculative_eagle_topk` | int | `null` | `null` | The number of tokens sampled from the draft model in eagle2 each step. |
| `speculative_num_draft_tokens` | int | `null` | `null` | The number of tokens sampled from the draft model in Speculative Decoding. |
| `speculative_dflash_block_size` | int | `null` | `null` | DFLASH only. Block size (verify window length). Alias of --speculative-num-draft-tokens for DFLASH. |
| `speculative_dspark_block_size` | int | `null` | `null` | DSPARK only. Draft block size gamma (number of proposed draft tokens). The verify window is gamma + 1, so this sets --speculative-num-draft-tokens = gamma + 1. Omit to auto-infer gamma from the draft checkpoint block_size. |
| `speculative_dspark_sps_table_path` | str | `null` | `null` | DSPARK only. Path to a pre-profiled SPS cost table (JSON) built offline with sglang.benchmark.dspark_sps_profiler, consumed by the ragged-verify scheduler (cap-accept / compact). Omit for an uninitialized flat constant-SPS table: the budget degenerates to verify-all (zero throughput gain by itself). |
| `speculative_dspark_confidence_sts_path` | str | `null` | `null` | DSPARK only. Optional path to a per-position STS (sequential temperature scaling) calibration JSON, fit offline with sglang.benchmark.dspark_sts_fit. Calibrates the confidence-head survival probabilities the ragged-verify scheduler consumes. Omit to use identity (no calibration); losslessness is unaffected either way. |
| `speculative_dspark_align_verify_tokens_to_graph_tier` | bool | `false` | `false` | DSPARK compact ragged-verify only. Fill the per-request verify lengths so the total verify-token count reaches the cuda-graph tier the forward is already padded to: round the dp-max scheduled total up to the captured token bucket and let the top-k allocator admit that many real draft tokens (confidence-ordered). This recovers the padding the forward pays for anyway -- both the cuda-graph bucket round-up and the dp cross-rank max -- turning it into extra real verification at the same step time. Off by default; when off the schedule is byte-for-byte unchanged. |
| `speculative_accept_threshold_single` | float | `1` | `1` | Accept a draft token if its probability in the target model is greater than this threshold. |
| `speculative_accept_threshold_acc` | float | `1` | `1` | The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc). |
| `speculative_use_rejection_sampling` | bool | `false` | `false` | Use rejection sampling for speculative decoding (requires topk=1). |
| `speculative_token_map` | str | `null` | `null` | The path of the draft model's small vocab table. |
| `speculative_attention_mode` | str | `prefill` | `prefill` | Attention backend for speculative decoding operations (both target verify and draft extend). Can be one of 'prefill' (default) or 'decode'. |
| `speculative_draft_attention_backend` | str | `null` | `null` | Attention backend for speculative decoding drafting. |
| `speculative_dsa_topk_backend` | str | `sgl-kernel` | `sgl-kernel` | DSA indexer top-k backend for speculative draft workers. Options: 'sgl-kernel', 'torch', 'flashinfer'. The 'torch' backend currently requires SGLANG_DSA_FUSE_TOPK=false. |
| `speculative_draft_kv_cache_dtype` | str | `null` | `null` | KV cache dtype for the speculative draft model only. The draft pool is allocated with one slot per target token (draft and target share a slot index space), so for a small draft it can still rival the target pool: a 5-layer DFLASH draft costs 10240 bytes/token in bf16. Setting fp8_e4m3 halves the draft pool; the saving shows up as free device memory, so raise --mem-fraction-static to convert it into KV capacity. Default follows --kv-cache-dtype. |
| `speculative_draft_window_size` | int | `null` | `null` | Sliding window size for the draft model. Honored by Llama EAGLE-3 (`LlamaForCausalLMEagle3`), DFLASH, and the built-in EAGLE/MTP draft-decode path on the Triton and FlashInfer draft backends; other EAGLE-3 backends (e.g. MLA-based drafters) silently ignore it. For Llama EAGLE-3, the drafter only attends to the most recent N keys (verifier hidden states + its own outputs); the verifier is unaffected. For DFLASH, the draft worker keeps a recent target-token window in its local KV cache (paged backends may retain up to one extra page on the left for alignment). For the built-in EAGLE/MTP draft, each draft-decode step attends to a --speculative-draft-sink-size sink plus the most recent N tokens, leaving the target verify pass unchanged; it is ignored (with a warning) if the draft model has a native sliding window of its own. Default is full attention/context. |
| `speculative_moe_runner_backend` | str | `null` | `flashinfer_mxfp4` | Choose the runner backend for MoE in speculative decoding. |
| `speculative_moe_a2a_backend` | str | `null` | `null` | Choose the backend for MoE A2A in speculative decoding |
| `speculative_draft_model_quantization` | str | `null` | `null` | The quantization method for speculative model. |
| `_speculative_draft_quantization_explicitly_set` | bool | — | `false` | （无 CLI 定义） |
| `speculative_skip_dp_mlp_sync` | bool | `false` | `false` | Skip the extra MLP sync that the scheduler performs before merging a new batch when speculative decoding + DP attention are both enabled. |
| `enable_multi_layer_eagle` | bool | `false` | `false` | Enable multi-layer Eagle speculative decoding. |
| `speculative_adaptive` | bool | `false` | `false` | Enable adaptive speculative decoding that dynamically adjusts num_steps based on acceptance rate. |
| `speculative_adaptive_config` | str | `null` | `null` | Path to a JSON config file for adaptive speculative decoding tuning knobs. |
| `decoupled_spec_bind_endpoint` | str | `null` | `null` | ZMQ endpoint this engine binds for its inbound channel in decoupled speculative decoding (verifier: result PULL; drafter: control PULL). |
| `decoupled_spec_connect_endpoints` | json_list_type | `null` | `null` | Peer inbound (bind) endpoints to connect to, ordered by peer rank, for decoupled speculative decoding. |
| `decoupled_spec_rank` | int | `null` | `null` | This engine's rank within its own role space (verifier-rank or drafter-rank) for decoupled speculative decoding. |
| `decoupled_spec_role` | str | `null` | `null` | Role in decoupled speculative decoding: 'null' disables it, 'verifier' runs the target/verify half, 'drafter' runs the draft half. |
| `spec_trace_dir` | str | `null` | `null` | Directory to write decoupled speculative decoding trace files. |

### 投机解码 (n-gram)（8）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `speculative_ngram_min_bfs_breadth` | int | `1` | `1` | The minimum breadth for BFS (Breadth-First Search) in ngram speculative decoding. |
| `speculative_ngram_max_bfs_breadth` | int | `10` | `10` | The maximum breadth for BFS (Breadth-First Search) in ngram speculative decoding. |
| `speculative_ngram_match_type` | str | `BFS` | `BFS` | The match type for cache tree. |
| `speculative_ngram_max_trie_depth` | int | `18` | `18` | The max trie depth for ngram speculative decoding. |
| `speculative_ngram_capacity` | int | `10000000` | `10000000` | The cache capacity for ngram speculative decoding. |
| `speculative_ngram_external_corpus_path` | str | `null` | `null` | Path to an external JSONL corpus to pre-load into SAM at startup. Additional corpora can be added at runtime via POST /add_external_corpus. |
| `speculative_ngram_external_sam_budget` | int | `0` | `0` | Number of draft nodes reserved for the external SAM subtree in ngram speculative decoding. |
| `speculative_ngram_external_corpus_max_tokens` | int | `10000000` | `10000000` | Fail startup if the tokenized external ngram corpus exceeds this many tokens. Tune this based on your CPU memory budget. |

### 专家并行 (EP)（34）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `ep_size` | int | `1` | `8` | The expert parallelism size. |
| `moe_a2a_backend` | str | `none` | `none` | Choose the backend for MoE A2A. |
| `enable_w4a4_mxfp4_megamoe` | bool | `false` | `false` | Enable the W4A4 MXFP4 MegaMoE path with DeepGEMM's mxf4xmxf4 MMA type. Use with --moe-a2a-backend megamoe. |
| `deepep_v2_mode` | str | `direct` | `direct` | DeepEP v2 ElasticBuffer communication topology, fixed at server init: `direct` (single-node NVLink) or `hybrid` (multi-node scale-out). Layout/grouped-GEMM and the decode CUDA graph are chosen per batch by inference phase, independent of this knob; not equivalent to DeepEP v1 normal/low_latency. |
| `moe_runner_backend` | str | `auto` | `flashinfer_mxfp4` | Choose the runner backend for MoE. |
| `flashinfer_mxfp4_moe_precision` | str | `default` | `default` | Choose the computation precision of flashinfer mxfp4 moe. On SM90, `fp8` selects the Humming-style MXFP4-weight x FP8-activation path introduced by FlashInfer #3738 and requires FlashInfer >= 0.6.18. |
| `deepep_mode` | str | `auto` | `auto` | Select the mode when enable DeepEP or MoriEP MoE, could be `normal`, `low_latency` or `auto`. Default is `auto`, which means `low_latency` for decode batch and `normal` for prefill batch. |
| `fuseep_mode` | int | `2` | `2` | Select the mode when enable Ascend FuseEP MoE, 1 -> dispatch_gmm_combine_decode is executed；2 -> dispatch_ffn_combine is executed (support hybrid deployment when 2). |
| `deepep_dispatcher_output_dtype` | str | `auto` | `auto` | Select DeepEP dispatcher output dtype |
| `ep_num_redundant_experts` | int | `0` | `0` | Allocate this number of redundant experts in expert parallel. |
| `ep_dispatch_algorithm` | str | `null` | `null` | The algorithm to choose ranks for redundant experts in expert parallel. |
| `init_expert_location` | str | `trivial` | `trivial` | Initial location of EP experts. |
| `enable_eplb` | bool | `false` | `false` | Enable EPLB algorithm |
| `eplb_algorithm` | str | `auto` | `auto` | Chosen EPLB algorithm |
| `eplb_rebalance_num_iterations` | int | `1000` | `1000` | Number of iterations to automatically trigger a EPLB re-balance. |
| `eplb_rebalance_layers_per_chunk` | int | `null` | `null` | Number of layers to rebalance per forward pass. |
| `eplb_min_rebalancing_utilization_threshold` | float | `1` | `1` | Minimum threshold for GPU average utilization to trigger EPLB rebalancing. Must be in the range [0.0, 1.0]. |
| `expert_distribution_recorder_mode` | str | `null` | `null` | Mode of expert distribution recorder. |
| `expert_distribution_recorder_buffer_size` | int | `null` | `1000` | Circular buffer size of expert distribution recorder. Set to -1 to denote infinite buffer. |
| `expert_balancedness_report_mode` | str | `off` | `off` | Where to report expert balancedness. Options: off, server_log, prometheus, both. |
| `deepep_config` | str | `null` | `null` | Tuned DeepEP config suitable for your own cluster. It can be either a string with JSON content or a file path. |
| `moe_dense_tp_size` | int | `null` | `null` | TP size for MoE dense MLP layers. This flag is useful when, with large TP size, there are errors caused by weights in MLP layers having dimension smaller than the min dimension GEMM supports. |
| `elastic_ep_backend` | str | `null` | `null` | Specify the collective communication backend for elastic EP. Supports 'mooncake' and 'nixl'. |
| `enable_elastic_expert_backup` | bool | `false` | `false` | Enable elastic expert backup feature. |
| `mooncake_ib_device` | str | `null` | `null` | The InfiniBand devices for Mooncake Backend transfer, accepts multiple comma-separated devices (e.g., --mooncake-ib-device mlx5_0,mlx5_1). Default is None, which triggers automatic device detection when Mooncake Backend is enabled. |
| `enable_waterfill` | bool | `false` | `false` | Enable Waterfill: dispatch the fused shared expert as an extra routed expert slot to the least-loaded EP rank. Supports DeepEP and MegaMOE MoE A2A backends, implicitly enables shared-expert fusion, and supports --deepep-mode auto, normal, or low_latency when used with DeepEP. Use auto or low_latency for production DeepEP decode so CUDA graph remains enabled. Supported on DeepSeek-V3/R1 with EP >= 2. |
| `ep_join_mode` | str | `null` | `null` | Join mode for elastic EP. 'recover' rejoins an existing slot after a fault. 'scale' joins as a new rank beyond the original group size and requires --node-rank 1. |
| `ep_join_rank_offset` | int | `0` | `0` | Global rank offset of an elastic EP joining group. Scale joiners must set this to the current effective EP size. |
| `elastic_ep_initial_size` | int | `null` | `null` | EP size used to define the immutable per-rank expert storage layout. Scale joiners must use the primary deployment's launch-time EP size. |
| `max_ep_size` | int | `null` | `null` | Maximum EP size the server can scale to at runtime. Pre-allocates active-rank state and backend buffers to this size. Defaults to the launch-time world size. |
| `elastic_ep_scale_timeout` | float | `600` | `600` | Timeout in seconds for a pending elastic EP scale operation. |
| `disable_flashinfer_cutlass_moe_fp4_allgather` | bool | `false` | `false` | Disables quantize before all-gather for flashinfer cutlass moe. |
| `disable_shared_experts_fusion` | bool | `false` | `false` | Disable the built-in shared experts fusion optimization for DeepSeek V3/R1. Note: Waterfill (--enable-waterfill) routes the shared expert as an extra MoE slot, so the shared expert is not separated from the MoE path when Waterfill is enabled. |
| `enforce_shared_experts_fusion` | bool | `false` | `false` | Enforce shared experts fusion even when it would normally be disabled (e.g. under DeepEP). Mutually exclusive with --disable-shared-experts-fusion. |

### Mamba 缓存与线性注意力（18）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `max_mamba_cache_size` | int | `null` | `null` | The maximum size of the mamba cache. |
| `mamba_ssm_dtype` | str | `null` | `null` | The data type of the SSM states in mamba cache. If not set, will be read from model config (mamba_ssm_dtype). |
| `mamba_max_states_per_path` | int | `-1` | `-1` | Maximum number of cached Mamba states retained per root-to-tail path (-1 means unlimited). When enabled, after each insert the shallowest eligible interior states beyond the cap are removed while their full KV remains. Tail, fork, and locked nodes are preserved. Must be -1 or a positive integer. |
| `enable_mamba_cache_stochastic_rounding` | bool | `false` | `false` | Enable stochastic rounding when writing FP16 Mamba SSM cache states. Requires --mamba-ssm-dtype float16 and CUDA. With --mamba-backend triton, requires SM100. |
| `mamba_cache_philox_rounds` | int | `0` | `0` | Number of Philox rounds to use for stochastic rounding of FP16 Mamba SSM cache writes. Triton uses the Triton default when set to 0; FlashInfer uses 10 rounds when set to 0. |
| `mamba_full_memory_ratio` | float | `null` | `0.9` | The ratio of mamba state memory to full kv cache memory. |
| `mamba_radix_cache_strategy` | str | `auto` | `auto` | The strategy to use for mamba radix cache. |
| `uses_mamba_radix_cache` | bool | — | `false` | （无 CLI 定义） |
| `mamba_track_interval` | int | `256` | `256` | The interval to track the mamba state during decode. |
| `enable_int8_mamba_checkpoint` | bool | `false` | `false` | Store radix-cached linear-attn (mamba) states in int8 (separate checkpoint pool) for ~2x cached-prefix capacity at fixed memory. |
| `int8_mamba_ckpt_size` | int | `null` | `null` | Number of int8 mamba checkpoint slots (default: 2x the active mamba pool size). |
| `linear_attn_backend` | str | `triton` | `triton` | The default kernel backend for linear attention (GDN/KDA). Can be overridden per-mode by --linear-attn-decode-backend and --linear-attn-prefill-backend. The Helion backend is KDA-only. |
| `linear_attn_decode_backend` | str | `null` | `null` | Override the kernel backend for linear attention decode. If not set, uses --linear-attn-backend. |
| `linear_attn_prefill_backend` | str | `null` | `null` | Override the kernel backend for linear attention prefill/extend. If not set, uses --linear-attn-backend; compatible SM100 GDN models may automatically select FlashInfer. |
| `linear_attn_verify_backend` | str | `null` | `null` | Override the kernel backend for linear attention speculative target-verify. If not set, follows the decode backend (flashinfer decode -> flashinfer verify, otherwise triton). KDA supports triton, nv_cutedsl, and flashinfer verify backends. |
| `enable_linear_replayssm` | bool | `false` | `false` | Enable the ReplaySSM buffered output-only linear-attn decode kernel. Primarily a GDN (scalar-gate) decode-bandwidth optimization (~1.2-1.5x at batch >= 64). KDA uses its selected Triton or Helion implementation, but its per-K gate ring is larger and ReplaySSM is typically slower than packed KDA decode; benchmark before enabling it. Requires the Triton linear-attn decode backend, or Helion for KDA, and --mamba-radix-cache-strategy no_buffer (the default). |
| `linear_replayssm_cache_len` | int | `16` | `16` | Ring-buffer length L for ReplaySSM linear-attn decode. The full recurrent state is flushed to HBM every L decode steps. |
| `enable_linear_replayssm_spec` | bool | `false` | `false` | Enable the ReplaySSM spec-verify: fold-every-commit -- a per-slot raw-input window replaces the recurrent verify's per-draft full-state snapshots. GDN or KDA hybrid linear-attn models, linear-chain (--speculative-eagle-topk in {None, 1}) only. |

### 分层缓存 (HiCache)（14）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `enable_hierarchical_cache` | bool | `false` | `false` | Enable hierarchical cache |
| `hicache_host_memory_mode` | str | `cache` | `cache` | Whether host memory is a persistent HiCache tier (cache) or a transient staging buffer between GPU and the storage backend (buffer_only). buffer_only requires --hicache-storage-backend. |
| `hicache_ratio` | float | `null` | `2` | The ratio of the size of host KV cache memory pool to the size of device pool. Defaults to 2.0 in cache mode, 1.2 in buffer_only mode, or 0.2 for backup-only host-pool decode retraction. |
| `hicache_size` | int | `0` | `0` | The size of host KV cache memory pool in gigabytes. Overrides --hicache-ratio in either host memory mode. |
| `hicache_write_policy` | str | `write_through` | `write_through` | The write policy of hierarchical cache. |
| `hicache_io_backend` | str | `kernel` | `kernel` | The IO backend for KV cache transfer between CPU and GPU |
| `hicache_mem_layout` | str | `page_first` | `page_first` | The layout of host memory pool for hierarchical cache. |
| `hicache_storage_backend` | str | `null` | `null` | The storage backend for hierarchical KV cache. Built-in backends: file, mooncake, npu_memcache, hf3fs, nixl, aibrix, tensorcast. For dynamic backend, use --hicache-storage-backend-extra-config to specify: backend_name (custom name), module_path (Python module path), class_name (backend class name). |
| `hicache_storage_prefetch_policy` | str | `timeout` | `timeout` | Control when prefetching from the storage backend should stop. |
| `hicache_storage_backend_extra_config` | str | `null` | `null` | A dictionary in JSON string format, or a string starting with a leading '@' and a config file in JSON/YAML/TOML format, containing extra configuration for the storage backend. |
| `hicache_storage_prefetch_retry_poll_interval` | int | `8` | `8` | Scheduling passes a queued request waits before its storage availability check is re-issued, when the prefetch found nothing and a backup may still be committing (under load the first check can run before it does). A re-issue that waits on staging or a moved match instead goes out on the next pass. Only passes that reach prefill scheduling count. 0 disables miss retries; known-hit deferrals are always re-issued. |
| `hicache_storage_prefetch_retry_max_attempts` | int | `8` | `8` | Storage availability re-issues a queued request may make, paced miss polls and immediate re-issues alike; past the cap it is admitted with whatever the device holds. 0 disables re-issues. |
| `enable_unified_cache_external_linker` | bool | `false` | `false` | Link UnifiedRadixCache directly to an external KV store (direct L3), with no host cache tier. |
| `unified_cache_external_linker_backend` | str | `mooncake` | `mooncake` | Storage backend for --enable-unified-cache-external-linker. |

### 分层稀疏注意力 (HiSparse)（2）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `enable_hisparse` | bool | `false` | `false` | Enable hierarchical sparse attention |
| `hisparse_config` | str | `null` | `null` | A dictionary in JSON string format for hierarchical sparse attention configuration. Example: '{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 2}' |

### 多模态优化（16）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `enable_broadcast_mm_inputs_process` | bool | `false` | `false` | Enable broadcast mm-inputs process in scheduler. |
| `enable_prefix_mm_cache` | bool | `false` | `false` | Enable prefix multimodal cache. Currently only supports mm-only. |
| `mm_enable_dp_encoder` | bool | `false` | `false` | Enabling data parallelism for mm encoder. The dp size will be set to the tp size automatically. |
| `mm_process_config` | loads | `null` | `{}` | Multimodal preprocessing config, a json config contains keys: `image`, `video`, `audio` |
| `mm_processor_worker_num` | int | `0` | `0` | Number of threads for multimodal processor calls. 0 selects the model-specific default. Only processors with isolated-worker support can use more than one thread. |
| `mm_io_worker_num` | int | `0` | `0` | Number of threads for multimodal data loading and decoding. 0 selects the model-specific default. SGLANG_IO_WORKERS remains supported as an environment override when this argument is 0. |
| `allowed_media_domains` | str | `[]` | `[]` | Restrict client-supplied HTTP(S) image, video, and audio URLs to these exact hostnames. Redirect destinations are checked against the same allowlist. When unset, remote media from any domain is allowed. |
| `media_url_max_file_size_mb` | int | `64` | `64` | Maximum size in MiB for one client-supplied remote media download. The limit is enforced while streaming; set to 0 to disable it. |
| `mm_preprocess_cache_size_mb` | int | `null` | `null` | CPU memory budget for content-addressed multimodal preprocessing artifacts. Unset selects a model-specific default (256 MiB for Kimi-K3); 0 disables the cache. The budget is divided across tokenizer workers and does not reserve GPU memory. |
| `trust_mm_content_hashes` | bool | `false` | `false` | Trust caller-provided multimodal SHA-256 content hashes. This can skip reading media on a hot metadata-cache hit; only enable it when the caller guarantees that hashes identify immutable media bytes. |
| `limit_mm_data_per_request` | loads | `null` | `null` | Limit the number of multimodal inputs per request. e.g. '{"image": 1, "video": 1, "audio": 1}' |
| `enable_mm_global_cache` | bool | `false` | `false` | Enable global multimodal embedding cache to skip redundant ViT inference. |
| `image_processor_backend` | str | `auto` | `auto` | Image processor backend. 'auto' lets Transformers select the best available backend. |
| `mm_global_cache_backend` | str | `mooncake` | `mooncake` | Storage backend for the multimodal global embedding cache. Used when --enable-mm-global-cache is set. |
| `disable_fast_image_processor` | bool | `false` | `false` | Deprecated. Use --image-processor-backend=pil instead. |
| `mm_feature_transport` | str | `null` | `cpu` | Transport multimodal features through CPU memory, a bounded CUDA IPC pool, or a bounded CUDA VMM pool. Unset uses cpu except for validated multi-node GB200/GB300 MNNVL models, which use cuda_vmm when an IMEX channel is available. Select cuda_ipc explicitly for single-node GPU transport. GPU transports reserve SGLANG_MM_FEATURE_CACHE_MB (default 1024 MiB) on the base GPU and fall back to CPU transport when the pool is full. |

### LoRA（14）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `enable_lora` | bool | `null` | `null` | Enable LoRA support for the model. This argument is automatically set to True if `--lora-paths` is provided for backward compatibility. |
| `enable_lora_overlap_loading` | bool | `null` | `null` | Enable asynchronous LoRA weight loading in order to overlap H2D transfers with GPU compute. This should be enabled if you find that your LoRA workloads are bottlenecked by adapter weight loading, for example when frequently loading large LoRA adapters. |
| `max_lora_rank` | int | `null` | `null` | The maximum rank of LoRA adapters. If not specified, it will be automatically inferred from the adapters provided in --lora-paths. |
| `lora_target_modules` | str | `null` | `null` | The union set of all target modules where LoRA should be applied. If not specified, it will be automatically inferred from the adapters provided in --lora-paths. If 'all' is specified, all supported modules will be targeted. |
| `lora_paths` | str | `null` | `null` | The list of LoRA adapters to load. Each adapter must be specified in one of the following formats: <PATH> \| <NAME>=<PATH> \| JSON with schema {"lora_name":str,"lora_path":str,"pinned":bool} |
| `max_loaded_loras` | int | `null` | `null` | If specified, it limits the maximum number of LoRA adapters loaded in CPU memory at a time. The value must be greater than or equal to `--max-loras-per-batch`. |
| `max_loras_per_batch` | int | `8` | `8` | Maximum number of adapters for a running batch, include base-only request. |
| `lora_eviction_policy` | str | `lru` | `lru` | LoRA adapter eviction policy when memory pool is full. 'lru': Least Recently Used (default, better cache efficiency). 'fifo': First-In-First-Out. |
| `lora_backend` | str | `csgmv` | `csgmv` | Choose the kernel backend for multi-LoRA serving. |
| `max_lora_chunk_size` | int | `16` | `16` | Maximum chunk size for the ChunkedSGMV LoRA backend. Only used when --lora-backend is 'csgmv'. Choosing a larger value might improve performance. |
| `experts_shared_outer_loras` | bool | `null` | `null` | Force shared outer LoRA mode for MoE models. When set, w1/w3 lora_A and w2 lora_B are shared across experts (expert_dim=1). Use --no-experts-shared-outer-loras to force disable. By default this is auto-detected from adapter weights. |
| `lora_use_virtual_experts` | bool | `false` | `false` | Enable virtual expert computation for MoE models. When set, the model will use virtual expert computation. |
| `lora_strict_loading` | bool | `false` | `false` | Enable strict loading for LoRA adapters. When set, mismatched or missing keys in the adapter weights will raise an error. |
| `lora_drain_wait_threshold` | float | `0` | `0` | When any LoRA adapter request waits longer than this threshold (in seconds), the scheduler will selectively drain one running adapter to make room. This mitigates extreme tail latency under high or skewed workloads by preventing a small set of adapters from monopolizing batch slots. Set to 0 to disable draining (default). |

### 双批次重叠 (TBO)（3）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `enable_two_batch_overlap` | bool | `false` | `false` | Enabling two micro batches to overlap. |
| `enable_single_batch_overlap` | bool | `false` | `false` | Let computation and communication overlap within one micro batch. |
| `tbo_token_distribution_threshold` | float | `0.48` | `0.48` | The threshold of token distribution between two batches in micro-batch-overlap, determines whether to two-batch-overlap or two-chunk-overlap. Set to 0 denote disable two-chunk-overlap. |

### CPU 卸载 (Offloading)（5）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `cpu_offload_gb` | int | `0` | `0` | How many GBs of RAM to reserve for CPU offloading. |
| `offload_group_size` | int | `-1` | `-1` | Number of layers per group in offloading. |
| `offload_num_in_group` | int | `1` | `1` | Number of layers to be offloaded within a group. |
| `offload_prefetch_step` | int | `1` | `1` | Steps to prefetch in offloading. |
| `offload_mode` | str | `cpu` | `cpu` | Mode of offloading. |

### LMCache（2）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `enable_lmcache` | bool | `false` | `false` | Using LMCache as an alternative hierarchical cache solution |
| `lmcache_config_file` | str | `null` | `null` | Path to the LMCache YAML configuration file |

### FlexKV（2）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `enable_flexkv` | bool | `false` | `false` | Route the default RadixCache through FlexKV's KVManager for host-tier (CPU / SSD / Remote) KV cache offload. Equivalent to --radix-cache-backend=flexkv but also participates in the auto-selection chain alongside --enable-lmcache. |
| `flexkv_config_file` | str | `null` | `null` | Path to the FlexKV YAML / JSON configuration file. Equivalent to setting the FLEXKV_CONFIG_PATH environment variable. |

### KTransformers / AMX 专家并行（6）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `kt_weight_path` | str | `null` | `null` | [ktransformers parameter] The path of the quantized expert weights for amx kernel. A local folder. |
| `kt_method` | str | `AMXINT4` | `AMXINT4` | [ktransformers parameter] Quantization formats for CPU execution. |
| `kt_cpuinfer` | int | `null` | `null` | [ktransformers parameter] The number of CPUInfer threads. |
| `kt_threadpool_count` | int | `2` | `2` | [ktransformers parameter] One-to-one with the number of NUMA nodes (one thread pool per NUMA). |
| `kt_num_gpu_experts` | int | `null` | `null` | [ktransformers parameter] The number of GPU experts. |
| `kt_max_deferred_experts_per_token` | int | `null` | `null` | [ktransformers parameter] Maximum number of experts deferred to CPU per token. All MoE layers except the final one use this value; the final layer always uses 0. |

### 扩散 LLM (dLLM)（3）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `dllm_algorithm` | str | `null` | `null` | The diffusion LLM algorithm, such as LowConfidence. |
| `dllm_algorithm_config` | str | `null` | `null` | The diffusion LLM algorithm configurations. Must be a YAML file. |
| `dllm_fdfo` | bool | `true` | `true` | Enable First-Done-First-Out (FDFO) scheduling for diffusion LLM inference. Enabled by default; use --no-dllm-fdfo to fall back to synchronous block scheduling. |

### PD 分离（11）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `disaggregation_mode` | str | `null` | `null` | Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated |
| `disaggregation_transfer_backend` | str | `mooncake` | `mooncake` | The backend for disaggregation transfer. Default is mooncake. |
| `disaggregation_bootstrap_port` | int | `8998` | `8998` | Bootstrap server port on the prefill server. Default is 8998. |
| `disaggregation_ib_device` | str | `null` | `null` | The InfiniBand devices for disaggregation transfer. Supports a single device (e.g., --disaggregation-ib-device mlx5_0), a shared comma-separated list (e.g., --disaggregation-ib-device mlx5_0,mlx5_1), a per-GPU JSON mapping (e.g., --disaggregation-ib-device '{"0": "mlx5_0,mlx5_1", "1": "mlx5_2"}'), or a path to a JSON file containing that mapping. Default is None, which triggers automatic device detection when mooncake backend is enabled. |
| `disaggregation_decode_enable_radix_cache` | bool | `false` | `false` | Enable radix cache on decode server (PD mode). Caches KV prefixes to avoid redundant transfers. Incompatible with --enable-hisparse, speculative decoding, and --disaggregation-transfer-backend fake. |
| `disaggregation_decode_enable_offload_kvcache` | bool | `false` | `false` | Enable async KV cache offloading on decode server (PD mode). |
| `disaggregation_decode_retraction_backup` | str | `null` | `null` | Storage backend for KV preserved across PD decode retraction. 'cpu_tensor' uses per-request CPU tensors. 'host_pool' uses a reserved HiCache pool and does not fall back on exhaustion. 'none' keeps no backup: a retracted request is aborted with 503 for the client to retry. If omitted, the backend is inferred from the decode KV pool. |
| `num_reserved_decode_tokens` | int | `512` | `512` | Number of decode tokens that will have memory reserved when adding new request to the running batch. |
| `disaggregation_decode_extra_slots` | int | `null` | `null` | Number of extra decode req_to_token slots pre-allocated for in-transfer requests (PD mode). If unset, defaults to 0 (or 2x the per-worker running batch for small batches). |
| `disaggregation_decode_polling_interval` | int | `1` | `1` | The interval to poll requests in decode server. Can be set to >1 to reduce the overhead of this. |
| `optimistic_prefill_attempts` | int | `0` | `0` | Number of optimistic prefill forward passes that skip the bootstrap wait. |

### 编码器预填充分离（8）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `encoder_only` | bool | `false` | `false` | For MLLM with an encoder, launch an encoder-only server |
| `language_only` | bool | `false` | `false` | For VLM, load weights for the language model only. |
| `language_model_only` | bool | `false` | `false` | Skip the multimodal encoder entirely: its weights are never loaded and the tower is never built, freeing that GPU memory for KV cache. Multimodal requests are rejected. Unlike --language-only this is a standalone mode, not part of encoder/decoder disaggregation. |
| `encoder_transfer_backend` | str | `auto` | `zmq_to_scheduler` | The backend for encoder disaggregation transfer. Auto selects a model- and TP-aware backend. |
| `encoder_urls` | str | `[]` | `[]` | List of encoder server urls. |
| `encoder_bootstrap_port` | int | `8997` | `8997` | Port for the EncoderBootstrapServer that runs in the language-only tokenizer manager process. Encoders register here, and language-only receivers fetch the current URL list from here. |
| `encoder_register_urls` | str | `[]` | `[]` | One or more EncoderBootstrapServer URLs to register this encoder with on startup, for dynamic encoder discovery. Example: --encoder-register-urls http://prefill0:8997 http://prefill1:8997. Used with --encoder-only servers. |
| `enable_adaptive_dispatch_to_encoder` | bool | `false` | `false` | When enabled, adaptively dispatch: multi-image requests go to encoder in language_only epd mode, single-image requests are processed locally. |

### PD-Multiplexing（3）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `enable_pdmux` | bool | `false` | `false` | Enable PD-Multiplexing, PD running on greenctx stream. |
| `pdmux_config_path` | str | `null` | `null` | The path of the PD-Multiplexing config file. |
| `sm_group_num` | int | `8` | `8` | Number of sm partition groups. |

### 权重加载与更新（19）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `startup_weight_load_mode` | str | `serial` | `serial` | Control startup weight loading relative to CUDA graph capture. 'serial' preserves the existing startup order; 'overlap' stages checkpoint files while CUDA graphs are captured and commits the real weights afterward. |
| `custom_weight_loader` | str | `null` | `[]` | The custom dataloader which used to update the model. Should be set with a valid import path, such as my_package.weight_load_func |
| `weight_loader_disable_mmap` | bool | `false` | `false` | Disable mmap while loading weight using safetensors. |
| `weight_loader_prefetch_checkpoints` | bool | `false` | `false` | Prefetch checkpoint files into OS page cache before loading. Each rank prefetches a fraction of the shards, reducing total network I/O on shared filesystems (NFS/Lustre) from N*checkpoint to 1*checkpoint. Recommended for models on network storage. When enabled, multi-threaded safetensors loading is disabled by default to avoid I/O oversubscription with the prefetch threads; set enable_multithread_load=true in --model-loader-extra-config to keep multi-threaded loading (e.g. on local NVMe where prefetch is a no-op). |
| `weight_loader_prefetch_num_threads` | int | `4` | `4` | Number of threads per rank for checkpoint prefetching (default: 4). |
| `weight_loader_drop_cache_after_load` | bool | `false` | `false` | Call posix_fadvise(DONTNEED) on each safetensors shard after loading it. |
| `remote_instance_weight_loader_seed_instance_ip` | str | `null` | `null` | The ip of the seed instance for loading weights from remote instance. |
| `remote_instance_weight_loader_seed_instance_service_port` | int | `null` | `null` | The service port of the seed instance for loading weights from remote instance. |
| `remote_instance_weight_loader_send_weights_group_ports` | json_list_type | `null` | `null` | The communication group ports for loading weights from remote instance. |
| `remote_instance_weight_loader_backend` | str | `nccl` | `nccl` | The backend for loading weights from remote instance. Can be 'transfer_engine', 'nccl', or 'modelexpress'. Default is 'nccl'. |
| `remote_instance_weight_loader_start_seed_via_transfer_engine` | bool | `false` | `false` | Start seed server via transfer engine backend for remote instance weight loader. |
| `engine_info_bootstrap_port` | int | `6789` | `6789` | Port for the engine info bootstrap server. Default is 6789. Must be set explicitly when running multiple instances on the same node. |
| `modelexpress_config` | str | `null` | `null` | JSON config for ModelExpress P2P weight loading. Keys: "url" (optional gRPC host:port override), "transport" ("nixl" or "transfer_engine"). Example: '{"url": "localhost:8001", "transport": "nixl"}' |
| `download_dir` | str | `null` | `null` | Model download directory for huggingface. |
| `model_checksum` | str | `null` | `null` | Model file integrity verification. If provided without value, uses model-path as HF repo ID. Otherwise, provide checksums JSON file path or HuggingFace repo ID. |
| `delete_ckpt_after_loading` | bool | `false` | `false` | Delete the model checkpoint after loading the model. |
| `decrypted_config_file` | str | `null` | `null` | The path of the decrypted config file. |
| `decrypted_draft_config_file` | str | `null` | `null` | The path of the decrypted draft config file. |
| `checkpoint_engine_wait_weights_before_ready` | bool | `false` | `false` | If set, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods before serving inference requests. |

### 预填充延迟器（7）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `enable_prefill_delayer` | bool | `false` | `false` | Enable prefill delayer for DP attention to reduce idle time. |
| `prefill_delayer_max_delay_passes` | int | `30` | `30` | Maximum forward passes to delay prefill. |
| `prefill_delayer_token_usage_low_watermark` | float | `null` | `null` | Token usage low watermark for prefill delayer. |
| `prefill_delayer_forward_passes_buckets` | float | `null` | `null` | Custom buckets for prefill delayer forward passes histogram. 0 and max_delay_passes-1 will be auto-added. |
| `prefill_delayer_wait_seconds_buckets` | float | `null` | `null` | Custom buckets for prefill delayer wait seconds histogram. 0 will be auto-added. |
| `prefill_delayer_queue_min_ratio` | float | `null` | `null` | Opt-in to the adaptive queue-based delay trigger (independent of the slot-based one). Delays prefill until the waiting queue reaches min(running_req * ratio, prefill_max_requests), falling back to the observed max_prefill_bs when no request limit is set. Unset (default) keeps the original slot-only behavior. Typical: 0.1 ~ 0.5. |
| `prefill_delayer_max_delay_ms` | float | `null` | `null` | Wall-clock cap (ms) on a single queue-trigger delay; once exceeded, prefill is force-released to bound worst-case TTFT. Only consulted when --prefill-delayer-queue-min-ratio is set. Typical: 1000 ~ 5000; defaults to 5000 if unset. |

### 最小空闲槽延迟（1）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `min_free_slots_delay` | int | `null` | `null` | Hold new prefills until at least N running-request slots have freed up, so they are admitted in one batch instead of one at a time. Useful when each admission is disproportionately expensive, e.g. speculative decoding with a separate draft prefill pass. An explicit value always wins, capped by max-running-requests (1 disables). When unset, DFlash workloads auto-enable the formula; other workloads stay disabled. Not supported with pipeline parallelism. |

### 确定性推理（2）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `enable_deterministic_inference` | bool | `false` | `false` | Enable deterministic inference mode with batch invariant ops. |
| `rl_on_policy_target` | str | `null` | `null` | The training system that SGLang needs to match for true on-policy. |

### KV Canary（3）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `kv_canary` | str | `none` | `none` | KV cache canary mode. 'none' disables the canary (default). 'log' prints them while the server keeps running (production-safe). 'raise' fails the server on the first detected mismatch (CI lane). |
| `kv_canary_real_data` | str | `none` | `none` | Check the real KV-cache in the canary. 'none' (default) disables the feature. 'partial' checks the first 16 bytes of each real-KV slot. 'all' checks the full real-KV slot. |
| `kv_canary_sweep_interval` | int | `0` | `0` | Every N forward steps, run a full-pool sweep. |

### 动态批处理分词器（5）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `enable_dynamic_batch_tokenizer` | bool | `false` | `false` | Enable async dynamic batch tokenizer for improved performance when multiple requests arrive concurrently. |
| `dynamic_batch_tokenizer_batch_size` | int | `32` | `32` | [Only used if --enable-dynamic-batch-tokenizer is set] Maximum batch size for dynamic batch tokenizer. |
| `dynamic_batch_tokenizer_batch_timeout` | float | `0.002` | `0.002` | [Only used if --enable-dynamic-batch-tokenizer is set] Timeout in seconds for batching tokenization requests. |
| `enable_tokenizer_batch_encode` | bool | `false` | `false` | Enable batch tokenization for improved performance when processing multiple text inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds. |
| `disable_tokenizer_batch_decode` | bool | `false` | `false` | Disable batch decoding when decoding multiple completions. |

### 调试张量转储（3）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `debug_tensor_dump_output_folder` | str | `null` | `null` | The output folder for dumping tensors. In Eagle mode, tensor outputs from draft and target models are stored in separate subdirectories ('draft' and 'target'). |
| `debug_tensor_dump_layers` | int | `null` | `null` | The layer ids to dump. Dump all layers if not specified. |
| `debug_tensor_dump_input_file` | str | `null` | `null` | The input filename for dumping tensors |

### 其他运行时特性 (Misc)（11）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `enable_memory_saver` | bool | `false` | `false` | Allow saving memory using release_memory_occupation and resume_memory_occupation |
| `enable_weights_cpu_backup` | bool | `false` | `false` | Save model weights (both main model and draft model, if any) to CPU memory during release_weights_occupation and resume_weights_occupation |
| `enable_draft_weights_cpu_backup` | bool | `false` | `false` | Save draft model weights to CPU memory during release_weights_occupation and resume_weights_occupation |
| `enable_custom_logit_processor` | bool | `false` | `false` | Enable users to pass custom logit processors to the server (disabled by default for security) |
| `enable_return_hidden_states` | bool | `false` | `false` | Enable returning full hidden states with responses. Equivalent to `--return-hidden-states-mode full`. |
| `return_hidden_states_mode` | str | `null` | `null` | Set the maximum hidden-state return mode supported by the server. `last` allows requests with return_hidden_states=False or `last`; `full` also allows return_hidden_states=True. |
| `enable_return_routed_experts` | bool | `false` | `false` | Enable returning routed experts of each layer with responses. |
| `enable_return_indexer_topk` | bool | `false` | `false` | Enable returning indexer topk indices of layers with indexer with responses. |
| `sampling_mask_max_tokens` | int | `4096` | `4096` | The maximum number of token IDs in a returned sampling mask. Requests are aborted if their realized sampling support exceeds this limit. Use the same value on disaggregated prefill and decode nodes; clients should set top_k below the limit to leave headroom for cutoff ties. |
| `disable_outlines_disk_cache` | bool | `false` | `false` | Disable disk cache of outlines to avoid possible crashes related to file system or high concurrency. |
| `enable_mis` | bool | `false` | `false` | Enable Multi-Item Scoring optimization. Combines query and multiple items into a single sequence for efficient batch processing. Requires --attention-backend flashinfer; auto-disables CUDA graph, radix cache, and chunked prefill. |

### 权重缓存（3）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `weight_cache_mode` | str | `off` | `off` | Weight cache mode. 'off': normal disk loading. 'daemon': launch weight cache daemon (holds weights in GPU memory). Engine-spawned daemons are co-terminal with the engine and do NOT persist across restarts, so this alone does not speed up restart (the first start is slower). For fast recovery, run the standalone daemon (python -m sglang.srt.weight_cache.daemon) and connect with 'client'. 'client': connect to existing daemon and load via IPC. |
| `weight_cache_socket` | str | `null` | `null` | Unix socket path for weight cache daemon (client mode).If not set, derives the path from SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE using the caller's physical GPU UUID. |
| `weight_cache_timeout` | int | `1800` | `1800` | Timeout in seconds for weight cache daemon readiness (default: 1800). |

### 自定义钩子与探针（2）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `forward_hooks` | json_list_type | `null` | `null` | JSON-formatted forward hook specifications to attach to the model. |
| `msprobe_dump_config` | str | `null` | `null` | The path of the JSON configuration file for msProbe. If specified, enables msProbe dump. |

### PLE 嵌入卸载（3）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `ple_offload_embedding` | bool | `null` | `null` | Offload Qwen4 PLE n-gram embedding weights to CPU pinned memory. Enabled by default for BF16 Qwen4-Exp on CUDA; use --no-ple-offload-embedding to disable. |
| `ple_offload_backend` | str | `pinned` | `pinned` | Host storage for the offloaded Qwen4 PLE n-gram table. 'pinned' (default) uses CPU pinned memory. 'file' maps a sparse file under --ple-offload-dir and lets the gather kernel read it directly; use it on unified-memory devices (e.g. GB10 / DGX Spark) where pinned host memory comes out of the same pool as the model weights. Requires a device that reports cudaDevAttrPageableMemoryAccessUsesHostPageTables. |
| `ple_offload_dir` | str | `null` | `null` | Directory for the file-backed PLE table when --ple-offload-backend is 'file'. Defaults to $SGLANG_CACHE_DIR/ple/<model path>, one directory per checkpoint. The file is sparse and reused across restarts; put it on fast local storage (NVMe). |

### PD 扩展与内部标志（17）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `disaggregation_enable_kv_checksum` | bool | `false` | `false` | Compute an Adler-32 checksum over each request's KV pages on prefill and verify it on decode. Enable on both prefill and decode engines. A mismatch aborts the request, or raises in CI. Disabled by default. |
| `enable_pd_role_switch` | bool | `false` | `false` | Allow runtime prefill<->decode role switch via /pd_role_switch (PD mode). |
| `speculative_domino_candidate_pool_size` | int | `2048` | `2048` | Domino only. Size of the approximate block-shared base-logit candidate pool. Set to 0 to score the full vocabulary. |
| `flashinfer_a2a_dispatch_type` | str | `null` | `null` | Select FlashInfer A2A dispatcher activation dtype. |
| `cuda_graph_max_seq_len_prefill` | int | `null` | `null` | Longest sequence a prefill cuda graph replay admits; longer batches run eager prefill. Folds into cuda_graph_config[prefill].max_seq_len. |
| `prefill_kv_cache_dequant_dtype` | str | `auto` | `auto` | Online dequantization dtype used by prefill attention when --kv-cache-dtype=nvfp4. 'nvfp4' reads the packed cache directly without additional dequantization; 'fp8_e4m3' dequantizes it into a temporary FP8 workspace. This does not change the stored KV-cache dtype. 'auto' selects NVFP4 without additional dequantization on SM100 and FP8 E4M3 otherwise. |
| `dsv4_attn_backend` | str | `auto` | `auto` | DeepSeek V4 attention backend. 'auto' (default) resolves to 'flashmla'. 'trtllm' (opt-in, SM100/SM103 with FP8 KV cache) switches the SWA/compressed KV pools to a uniform 512-dim FP8 layout and runs decode and sparse prefill through the flashinfer trtllm-gen sparse MLA kernel. The backend choice is shared by prefill and decode. |
| `otlp_service_name` | str | `null` | `null` | Service name for OTLP traces (displayed as 'service.name' in trace backends). If unset, falls back to the OTEL_SERVICE_NAME env var, then to 'sglang'. |
| `_radix_eviction_policy_explicitly_set` | bool | — | `false` | （无 CLI 定义） |
| `hicache_host_memory_fraction` | float | `0.8` | `0.8` | Fraction of the available host memory, bounded by visible cgroup memory.max/memory.high or v1 memory limits (after a 10 GiB reserve) that the HiCache host pools of all ranks on this machine may use. Applies only when neither --hicache-ratio nor --hicache-size is set: the default ratio is then reduced until the pools fit. Lower it when several engines share a memory cgroup. |
| `enable_linker_mla_dedup` | bool | `false` | `false` | Load replicated MLA KV on rank 0 and broadcast each layer with the Mooncake linker. |
| `_swa_full_tokens_ratio_explicitly_set` | bool | — | `false` | （无 CLI 定义） |
| `swa_prefix_tails` | int | `null` | `null` | When the SWA KV pool is sized from the request cap (DeepSeek-V4 family), how many radix-cached prefix tails it keeps room for. Each tail is one sliding window plus one page. Default: 4 x max_running_requests per attention-DP rank, 0 when the radix cache is disabled. |
| `enable_encoder_swa_bounded_replay` | bool | `false` | `false` | DeepSeek-V4.1 encoder SWA bounded replay: cache Main KV and Indexer keys only, rebuild request-owned SWA windows on prefix hits. Experimental; CUDA only. |
| `enable_decoder_swa_bounded_replay` | bool | `false` | `false` | DeepSeek-V4.1 decoder SWA bounded replay: after the last kv_source layer, run the remaining layers over only the last window_size tokens of a prefill. Main and indexer KV stay exact; nothing is replayed. Deterministic for a fixed prompt and chunk size. |
| `enable_response_store` | bool | `false` | `false` | Enable in-memory Responses storage for retrieval, chaining, and background requests. Disabled by default; unsupported with prefill-decode disaggregation. Storage has no TTL or size limit. |
| `disaggregation_decode_host_receive_threshold` | float | `0` | `0` | Device token usage fraction at which incoming KV is received in the decode retraction host pool, excluding evictable cache pages. Range [0, 1]; 0 disables host receive. Size with --hicache-size or --hicache-ratio; requires dense MHA and a transfer backend that supports host destinations. No built-in backend currently supports this. |

### 追加字段（未列入 field_order）（4）

| 字段 | 类型 | 默认 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `speculative_draft_sink_size` | int | `null` | `null` | Number of leading 'attention sink' tokens the draft always attends to, in addition to the --speculative-draft-window-size recent window (StreamingLLM-style). Honored only by the built-in EAGLE/MTP draft-decode path on the Triton and FlashInfer draft backends; the Llama EAGLE-3 and DFLASH windows ignore it. 0/unset => pure recent window. Requires --speculative-draft-window-size. |
| `disable_aiter_allreduce_fusion_in_prefill` | bool | `false` | `false` | Disable Aiter AllReduce Fusion for prefill batches (EXTEND / MIXED / SPLIT_PREFILL) while keeping it for decode. Only meaningful with --enable-aiter-allreduce-fusion. |
| `disable_aiter_allreduce_fusion_in_decode` | bool | `false` | `false` | Disable Aiter AllReduce Fusion for decode batches (DECODE / TARGET_VERIFY / draft-extend / IDLE) while keeping it for prefill. Only meaningful with --enable-aiter-allreduce-fusion. |
| `grpc_response_timeout_secs` | int | `300` | `300` | Timeout in seconds waiting for each native gRPC response chunk, including the first. Must be positive. Defaults to 300. |

## 4. `internal_states[*]` 的运行时专有字段

> `internal_states` = 每个 DP rank 一份的调度器状态字典 = `resolved_server_args_dict()` + 运行态字段。
> 与 §3 同名的字段含义一致（实时解析值）；此处仅列 §3 之外的运行时专有字段（共 7 个，其余 516 个见 §3）。

| 字段 | 类型 | 当前值 | 说明 |
| --- | --- | --- | --- |
| `world_size` | int | `8` | 世界大小 = dp × tp × pp |
| `last_gen_throughput` | float | `0` | 最近统计窗口的生成吞吐 (token/s) |
| `memory_usage` | dict | `{"weight": 37.889, "kvcache": 0, "startup_available": 19.…` | 显存明细：weight / kvcache / startup_available / token_capacity(/swa) / graph{cuda graph} |
| `startup_time` | dict | `{"load_weight": 373.13464560359716, "kv_cache_allocation"…` | 启动各阶段耗时（同顶层 startup_time） |
| `effective_max_running_requests_per_dp` | int | `256` | 单 DP rank 实际生效的 max_running_requests |
| `decode_cuda_graph_bs` | list | `[]` | 已捕获的 decode CUDA graph batch size 列表 |
| `decode_cuda_graph_memory_gb` | float | `2.096` | decode CUDA graph 显存占用 (GB) |
