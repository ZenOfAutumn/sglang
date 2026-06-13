# srt

## 目录用途
`srt`（SGLang RunTime）是 SGLang 推理引擎的运行时核心包，承载从服务参数解析、HTTP/引擎入口、请求调度、KV 缓存管理，到模型加载、各类计算层与推理执行的全部服务端逻辑。本目录直接存放全局常量、环境变量描述符和服务器启动参数等基础设施，其余功能按职责拆分到大量子目录中。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `constants.py` | 定义全局常量，如 GPU 显存类型（KV 缓存/权重/CUDA Graph）枚举与健康检查请求 ID 前缀。 |
| `environ.py` | 集中管理 SGLANG_*/SGL_* 环境变量，提供 `EnvField` 描述符、`Envs` 注册表与 `temp_set_env` 临时设置上下文。 |
| `server_args.py` | 定义 `ServerArgs`、`PortArgs` 等核心数据类及 argparse 参数注册，是服务器所有可配置项的单一来源。 |
| `server_args_config_parser.py` | 提供 `ConfigArgumentMerger`，负责将 YAML 配置文件与命令行参数合并。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `batch_invariant_ops` | 批次无关（确定性）算子实现，详见其 README_zh.md。 |
| `batch_overlap` | 单/双批次计算与通信重叠（SBO/TBO）调度，详见其 README_zh.md。 |
| `checkpoint_engine` | 通过 checkpoint-engine 进行权重热更新的集成，详见其 README_zh.md。 |
| `compilation` | torch.compile / 分段 CUDA Graph 编译后端与 FX pass，详见其 README_zh.md。 |
| `configs` | 各模型与加载相关的配置类。 |
| `connector` | 远程存储/KV 连接器抽象。 |
| `constrained` | 约束/结构化解码（语法、JSON 等）后端。 |
| `debug_utils` | 调试与精度比对工具。 |
| `disaggregation` | PD 分离（Prefill/Decode 解耦）相关组件。 |
| `distributed` | 分布式并行状态与通信原语。 |
| `dllm` | 扩散式 LLM（Diffusion LLM）推理支持，详见其 README_zh.md。 |
| `elastic_ep` | 弹性专家并行与专家权重备份，详见其 README_zh.md。 |
| `entrypoints` | HTTP server、Engine、OpenAI 兼容接口等入口。 |
| `eplb` | 专家并行负载均衡（EPLB）。 |
| `function_call` | 工具/函数调用解析。 |
| `grpc` | gRPC 服务模块，详见其 README_zh.md。 |
| `hardware_backend` | 不同硬件后端适配。 |
| `layers` | 注意力、MoE、量化等核心计算层。 |
| `lora` | LoRA 适配器加载与推理。 |
| `managers` | 调度器、分词管理器等运行时管理组件。 |
| `mem_cache` | KV 缓存与 Radix Cache 等内存管理。 |
| `model_executor` | 前向批处理与模型运行器。 |
| `model_loader` | 模型权重加载逻辑。 |
| `models` | 各模型架构实现。 |
| `multimodal` | 多模态输入处理。 |
| `multiplex` | PD 复用（SM 分割并发）调度，详见其 README_zh.md。 |
| `observability` | 监控、指标与请求耗时统计。 |
| `parser` | 提示词模板与推理内容解析。 |
| `ray` | 基于 Ray actor 的引擎与服务启动，详见其 README_zh.md。 |
| `sampling` | 采样参数与惩罚处理器。 |
| `speculative` | 投机解码（EAGLE 等）。 |
| `tokenizer` | 分词器封装。 |
| `utils` | 通用工具函数。 |
| `weight_sync` | 训练侧到推理侧的权重同步，详见其 README_zh.md。 |

## 完整目录树导航

下表按目录层级列出 `srt/` 下全部 130 个子目录，点击可进入各目录的 `README_zh.md` 查看该目录用途与文件清单。

- [`batch_invariant_ops/`](batch_invariant_ops/README_zh.md)
- [`batch_overlap/`](batch_overlap/README_zh.md)
- [`checkpoint_engine/`](checkpoint_engine/README_zh.md)
- [`compilation/`](compilation/README_zh.md)
- [`configs/`](configs/README_zh.md)
- [`connector/`](connector/README_zh.md)
  - [`serde/`](connector/serde/README_zh.md)
- [`constrained/`](constrained/README_zh.md)
  - [`triton_ops/`](constrained/triton_ops/README_zh.md)
- [`debug_utils/`](debug_utils/README_zh.md)
  - [`comparator/`](debug_utils/comparator/README_zh.md)
    - [`aligner/`](debug_utils/comparator/aligner/README_zh.md)
      - [`entrypoint/`](debug_utils/comparator/aligner/entrypoint/README_zh.md)
      - [`reorderer/`](debug_utils/comparator/aligner/reorderer/README_zh.md)
      - [`token_aligner/`](debug_utils/comparator/aligner/token_aligner/README_zh.md)
        - [`concat_steps/`](debug_utils/comparator/aligner/token_aligner/concat_steps/README_zh.md)
        - [`smart/`](debug_utils/comparator/aligner/token_aligner/smart/README_zh.md)
      - [`unsharder/`](debug_utils/comparator/aligner/unsharder/README_zh.md)
    - [`dims_spec/`](debug_utils/comparator/dims_spec/README_zh.md)
    - [`tensor_comparator/`](debug_utils/comparator/tensor_comparator/README_zh.md)
    - [`visualizer/`](debug_utils/comparator/visualizer/README_zh.md)
  - [`schedule_simulator/`](debug_utils/schedule_simulator/README_zh.md)
    - [`data_source/`](debug_utils/schedule_simulator/data_source/README_zh.md)
    - [`routers/`](debug_utils/schedule_simulator/routers/README_zh.md)
    - [`schedulers/`](debug_utils/schedule_simulator/schedulers/README_zh.md)
  - [`source_patcher/`](debug_utils/source_patcher/README_zh.md)
- [`disaggregation/`](disaggregation/README_zh.md)
  - [`ascend/`](disaggregation/ascend/README_zh.md)
  - [`base/`](disaggregation/base/README_zh.md)
  - [`common/`](disaggregation/common/README_zh.md)
  - [`fake/`](disaggregation/fake/README_zh.md)
  - [`mooncake/`](disaggregation/mooncake/README_zh.md)
  - [`mori/`](disaggregation/mori/README_zh.md)
  - [`nixl/`](disaggregation/nixl/README_zh.md)
- [`distributed/`](distributed/README_zh.md)
  - [`device_communicators/`](distributed/device_communicators/README_zh.md)
- [`dllm/`](dllm/README_zh.md)
  - [`algorithm/`](dllm/algorithm/README_zh.md)
  - [`mixin/`](dllm/mixin/README_zh.md)
- [`elastic_ep/`](elastic_ep/README_zh.md)
- [`entrypoints/`](entrypoints/README_zh.md)
  - [`anthropic/`](entrypoints/anthropic/README_zh.md)
  - [`ollama/`](entrypoints/ollama/README_zh.md)
  - [`openai/`](entrypoints/openai/README_zh.md)
- [`eplb/`](eplb/README_zh.md)
  - [`eplb_algorithms/`](eplb/eplb_algorithms/README_zh.md)
  - [`eplb_simulator/`](eplb/eplb_simulator/README_zh.md)
- [`function_call/`](function_call/README_zh.md)
- [`grpc/`](grpc/README_zh.md)
- [`hardware_backend/`](hardware_backend/README_zh.md)
  - [`mlx/`](hardware_backend/mlx/README_zh.md)
  - [`npu/`](hardware_backend/npu/README_zh.md)
    - [`attention/`](hardware_backend/npu/attention/README_zh.md)
    - [`graph_runner/`](hardware_backend/npu/graph_runner/README_zh.md)
    - [`modules/`](hardware_backend/npu/modules/README_zh.md)
    - [`moe/`](hardware_backend/npu/moe/README_zh.md)
    - [`quantization/`](hardware_backend/npu/quantization/README_zh.md)
- [`layers/`](layers/README_zh.md)
  - [`attention/`](layers/attention/README_zh.md)
    - [`fla/`](layers/attention/fla/README_zh.md)
    - [`linear/`](layers/attention/linear/README_zh.md)
      - [`kernels/`](layers/attention/linear/kernels/README_zh.md)
    - [`mamba/`](layers/attention/mamba/README_zh.md)
      - [`ops/`](layers/attention/mamba/ops/README_zh.md)
    - [`nsa/`](layers/attention/nsa/README_zh.md)
    - [`triton_ops/`](layers/attention/triton_ops/README_zh.md)
    - [`wave_ops/`](layers/attention/wave_ops/README_zh.md)
  - [`deep_gemm_wrapper/`](layers/deep_gemm_wrapper/README_zh.md)
  - [`moe/`](layers/moe/README_zh.md)
    - [`ep_moe/`](layers/moe/ep_moe/README_zh.md)
    - [`fused_moe_triton/`](layers/moe/fused_moe_triton/README_zh.md)
      - [`configs/`](layers/moe/fused_moe_triton/configs/README_zh.md)
        - [`triton_3_1_0/`](layers/moe/fused_moe_triton/configs/triton_3_1_0/README_zh.md)
        - [`triton_3_2_0/`](layers/moe/fused_moe_triton/configs/triton_3_2_0/README_zh.md)
        - [`triton_3_3_0/`](layers/moe/fused_moe_triton/configs/triton_3_3_0/README_zh.md)
        - [`triton_3_3_1/`](layers/moe/fused_moe_triton/configs/triton_3_3_1/README_zh.md)
        - [`triton_3_4_0/`](layers/moe/fused_moe_triton/configs/triton_3_4_0/README_zh.md)
        - [`triton_3_5_1/`](layers/moe/fused_moe_triton/configs/triton_3_5_1/README_zh.md)
    - [`moe_runner/`](layers/moe/moe_runner/README_zh.md)
    - [`token_dispatcher/`](layers/moe/token_dispatcher/README_zh.md)
  - [`quantization/`](layers/quantization/README_zh.md)
    - [`compressed_tensors/`](layers/quantization/compressed_tensors/README_zh.md)
      - [`schemes/`](layers/quantization/compressed_tensors/schemes/README_zh.md)
    - [`configs/`](layers/quantization/configs/README_zh.md)
    - [`modelslim/`](layers/quantization/modelslim/README_zh.md)
      - [`schemes/`](layers/quantization/modelslim/schemes/README_zh.md)
    - [`quark/`](layers/quantization/quark/README_zh.md)
      - [`schemes/`](layers/quantization/quark/schemes/README_zh.md)
  - [`rotary_embedding/`](layers/rotary_embedding/README_zh.md)
  - [`utils/`](layers/utils/README_zh.md)
- [`lora/`](lora/README_zh.md)
  - [`backend/`](lora/backend/README_zh.md)
  - [`torch_ops/`](lora/torch_ops/README_zh.md)
  - [`triton_ops/`](lora/triton_ops/README_zh.md)
- [`managers/`](managers/README_zh.md)
- [`mem_cache/`](mem_cache/README_zh.md)
  - [`cpp_radix_tree/`](mem_cache/cpp_radix_tree/README_zh.md)
  - [`hybrid_cache/`](mem_cache/hybrid_cache/README_zh.md)
  - [`sparsity/`](mem_cache/sparsity/README_zh.md)
    - [`algorithms/`](mem_cache/sparsity/algorithms/README_zh.md)
    - [`backend/`](mem_cache/sparsity/backend/README_zh.md)
    - [`core/`](mem_cache/sparsity/core/README_zh.md)
  - [`storage/`](mem_cache/storage/README_zh.md)
    - [`aibrix_kvcache/`](mem_cache/storage/aibrix_kvcache/README_zh.md)
    - [`eic/`](mem_cache/storage/eic/README_zh.md)
    - [`hf3fs/`](mem_cache/storage/hf3fs/README_zh.md)
      - [`docs/`](mem_cache/storage/hf3fs/docs/README_zh.md)
    - [`lmcache/`](mem_cache/storage/lmcache/README_zh.md)
    - [`mooncake_store/`](mem_cache/storage/mooncake_store/README_zh.md)
    - [`nixl/`](mem_cache/storage/nixl/README_zh.md)
- [`model_executor/`](model_executor/README_zh.md)
- [`model_loader/`](model_loader/README_zh.md)
- [`models/`](models/README_zh.md)
  - [`deepseek_common/`](models/deepseek_common/README_zh.md)
    - [`attention_forward_methods/`](models/deepseek_common/attention_forward_methods/README_zh.md)
- [`multimodal/`](multimodal/README_zh.md)
  - [`evs/`](multimodal/evs/README_zh.md)
  - [`processors/`](multimodal/processors/README_zh.md)
- [`multiplex/`](multiplex/README_zh.md)
- [`observability/`](observability/README_zh.md)
- [`parser/`](parser/README_zh.md)
- [`ray/`](ray/README_zh.md)
- [`sampling/`](sampling/README_zh.md)
  - [`penaltylib/`](sampling/penaltylib/README_zh.md)
- [`speculative/`](speculative/README_zh.md)
  - [`cpp_ngram/`](speculative/cpp_ngram/README_zh.md)
- [`tokenizer/`](tokenizer/README_zh.md)
- [`utils/`](utils/README_zh.md)
- [`weight_sync/`](weight_sync/README_zh.md)
