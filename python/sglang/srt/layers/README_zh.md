# srt/layers

## 目录用途
本目录是 SGLang 模型计算层的实现集合，提供构建大语言模型所需的各类神经网络层与算子，包括线性层（含张量并行）、归一化、激活函数、词表嵌入、注意力封装、采样、logits 处理以及张量并行通信原语等。这些层被 `srt/models` 下的各模型组装调用，并对接量化、MoE、注意力后端等子模块。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `activation.py` | 激活函数层：SiluAndMul、GeluAndMul、NewGELU、ReLU2、QuickGELU、XIELU、ScaledActivation 等多平台实现。 |
| `amx_utils.py` | CPU AMX 相关工具：权重加载后处理、维度/类型支持判断、`PackWeightMethod` 权重打包方法。 |
| `communicator.py` | 张量并行/层间通信核心：`LayerCommunicator`、`ScatterMode`、各种 all-reduce + layernorm 融合通信函数与上下文。 |
| `communicator_nsa_cp.py` | 面向 NSA（Native Sparse Attention）上下文并行的通信器，继承并扩展 `LayerCommunicator` 系列类。 |
| `conv.py` | Conv2d/Conv3d 层，当 kernel==stride 时用 unfold+linear 优化 patch 嵌入并规避 CuDNN 的 Conv3d bug。 |
| `dp_attention.py` | 数据并行注意力（DP attention）支持：DP 缓冲区管理、padding 模式、并行 world/local 信息计算与初始化。 |
| `elementwise.py` | 逐元素融合 Triton 算子：softcap、双残差 RMSNorm、RMSNorm、experts 合并、gelu_and_mul 等。 |
| `flashinfer_comm_fusion.py` | FlashInfer 通信融合：`FlashInferWorkspaceManager` 管理 all-reduce 与归一化融合工作区。 |
| `fused_sampling.py` | 采样流水线的融合 Triton kernel，将温度缩放+softmax 融合为单/多遍 kernel 以降低开销。 |
| `int4fp8_utils.py` | Quark 量化相关工具：fp8 张量缩放量化、int4 列向量化、int4 打包为 int32。 |
| `layernorm.py` | 归一化层：RMSNorm、LayerNorm、GemmaRMSNorm、Gemma3RMSNorm 多平台实现。 |
| `linear.py` | 线性层全家族：Replicated/Column/Row 并行、Merged/QKV 并行线性层及量化方法接入（改编自 vLLM）。 |
| `logits_processor.py` | logits 处理器：`LogitsProcessor`、`LogitsProcessorOutput`、`LogitsMetadata`，负责输出 logits 计算与元数据。 |
| `model_parallel.py` | torch 张量并行通用工具：`ColwiseParallelSharded`、`RowwiseParallelMaybeWait` 等并行风格。 |
| `modelopt_utils.py` | NVIDIA ModelOpt 量化相关常量，如量化配置选择 `QUANT_CFG_CHOICES`。 |
| `multimodal.py` | 多模态相关 GPU 张量哈希（MurmurHash 变体 Triton kernel）：`gpu_tensor_hash` 等。 |
| `n_gram_embedding.py` | `NgramEmbedding` 层，用于 n-gram 推测解码等场景的嵌入。 |
| `parameter.py` | vLLM 风格参数封装：`BasevLLMParameter` 及列/行/分组/块量化缩放等参数类型（改编自 vLLM）。 |
| `pooler.py` | 池化层：`Pooler`、`CrossEncodingPooler`、`PoolingType`，用于嵌入/重排序模型输出聚合。 |
| `radix_attention.py` | `RadixAttention` 注意力层封装及 `AttentionType`，对接各注意力后端与 radix 缓存。 |
| `radix_linear_attention.py` | `RadixLinearAttention` 线性注意力层封装。 |
| `rocm_linear_utils.py` | ROCm/AITER 线性算子工具：DSv3 router gemm、融合 qk-rope 等。 |
| `sampler.py` | `Sampler` 采样层，执行温度/top-p/top-k 等采样逻辑（支持 DP attention）。 |
| `sparse_pooler.py` | 稀疏嵌入池化：`SparsePooler`、`SparseEmbeddingOutput`。 |
| `torchao_utils.py` | torchao 量化工具：投影过滤器与将 torchao 配置应用到模型。 |
| `vocab_parallel_embedding.py` | 词表并行嵌入：`VocabParallelEmbedding`、`ParallelLMHead` 及分片索引（改编自 vLLM）。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `attention/` | 各注意力后端实现（FlashInfer、FlashAttention、Triton、NSA 等），由其它任务单独说明。 |
| `moe/` | 混合专家（MoE）层与路由、专家并行实现，由其它任务单独说明。 |
| `quantization/` | 量化方法集合（FP8、INT4、AWQ、GPTQ、ModelOpt 等），由其它任务单独说明。 |
| `deep_gemm_wrapper/` | DeepGEMM 库封装：JIT 编译、配置与 fp8 grouped GEMM 入口。 |
| `rotary_embedding/` | 旋转位置编码（RoPE）各变体实现（YaRN、Llama3、Deepseek、MRoPE 等）。 |
| `utils/` | 层相关通用工具：层 id 解析、上下文并行、哈希、logprob、多平台算子等。 |
