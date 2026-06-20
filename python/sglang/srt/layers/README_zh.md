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

## layers 计算层学习路线

本路线面向想吃透 `srt/layers` 的工程师，按**数据流顺序**（嵌入 → 线性/归一化/激活 → 注意力 → 输出 logits → 采样）拆成 6 个阶段，每阶段含「阅读清单 / 动手任务 / 自检问题」。建议先通读 `srt/models/llama.py` 的 `forward`，建立"一个模型如何把这些层串起来"的整体印象，再逐层深入。

> 前置：先理解张量并行（TP）基本概念（列并行/行并行），因为 layers 里大量层都是 TP 感知的。可参考 `docs/sglang_learning_plan_zh.md` 的并行小节。

### 阶段 L0：建立全景（0.5 天）

- **阅读**：本 README 文件清单；`srt/models/llama.py` 的 `LlamaDecoderLayer` / `LlamaForCausalLM.forward`，看各层调用顺序。
- **动手**：在 llama 的 `forward` 里按层打印各 tensor 的 shape（embedding→attention→mlp→logits），画出一张数据流图。
- **自检**：能说出一次前向里 layers 各组件的调用顺序，以及哪些层是张量并行的。

### 阶段 L1：嵌入与线性层全家族（1.5 天）★ 重点

- **阅读**：
  1. `vocab_parallel_embedding.py`：`VocabParallelEmbedding` / `ParallelLMHead` 如何按词表分片，分片索引怎么算。
  2. `linear.py`：`ReplicatedLinear`、`ColumnParallelLinear`、`RowParallelLinear`、`MergedColumnParallelLinear`、`QKVParallelLinear` 的区别与各自 all-gather/all-reduce 时机。
  3. `parameter.py`：vLLM 风格的参数封装与量化缩放参数如何挂载。
- **动手**：对照画出 QKV 投影用 `QKVParallelLinear`（列并行）、o_proj 用 `RowParallelLinear`（行并行）的通信路径；用 2 卡 TP 启动一个小模型，确认权重确实被切分。
- **自检**：为什么 attention 的 QKV 用列并行、output proj 用行并行？两者各自在何处触发通信？

### 阶段 L2：归一化、激活与逐元素融合（1 天）

- **阅读**：
  1. `layernorm.py`：`RMSNorm` / `LayerNorm` / `GemmaRMSNorm` 多平台实现与 residual 融合签名。
  2. `activation.py`：`SiluAndMul` / `GeluAndMul` 等"激活+逐元素乘"的融合层。
  3. `elementwise.py`：softcap、双残差 RMSNorm 等 Triton 融合算子。
- **动手**：找出 llama MLP 里 `SiluAndMul` 的调用点，理解 gate/up 投影后为何能融合；对比开关融合算子的 decode 延迟。
- **自检**：为什么把"激活 + 逐元素乘 + norm + residual"融合成单个 kernel 能提速？

### 阶段 L3：注意力封装与位置编码（2 天）★ 重点

- **阅读**：
  1. `radix_attention.py`：`RadixAttention` 如何作为统一入口对接 radix 缓存与各注意力后端、`AttentionType` 的含义。
  2. `radix_linear_attention.py`：线性注意力封装（如涉及 Mamba/GDN 类模型）。
  3. `rotary_embedding/`：选 RoPE 基础实现 + 一个变体（YaRN 或 Deepseek）看长度外推。
  4. `dp_attention.py`：DP attention 的缓冲区管理与 padding 模式（结合 Glossary 的「DP 注意力」「非填充 token 数」条目）。
  5. 概览 `attention/` 子目录的 `base_attn_backend.py` 与一个后端实现。
- **动手**：开关 `--enable-dp-attention`（或相应参数）观察 padding 与显存变化；在 `RadixAttention.forward` 打点确认走了哪个后端。
- **自检**：`RadixAttention` 与具体 attention backend 是什么关系？RoPE 变体如何支持超过训练长度的外推？

### 阶段 L4：输出 logits 处理（1 天）

- **阅读**：
  1. `logits_processor.py`：`LogitsProcessor` / `LogitsProcessorOutput` / `LogitsMetadata`，重点看 prefill 只需最后一个 token、decode 每步一个 token 的差异，以及 `return_logprob` 路径如何用 `log_softmax`（参考 Glossary「对数 softmax」条目）。
  2. `pooler.py` / `sparse_pooler.py`：嵌入/重排序模型如何聚合输出（与生成式 logits 路径对比）。
- **动手**：开 `return_logprob` 发一个请求，跟踪 logits → logprob 的计算路径。
- **自检**：prefill 与 decode 在 logits 计算上各取哪些位置的 hidden states？为什么？

### 阶段 L5：采样层（1.5 天）★ 重点

- **阅读**：
  1. `sampler.py`：`Sampler` 主流程，简单情形（直接 multinomial）与复杂情形（top-k/top-p/min-p 截断）的分叉；确定性采样 `multinomial_with_seed` 的 Gumbel-Max 实现。
  2. `fused_sampling.py`：温度缩放 + softmax 的融合 Triton kernel。
  3. 配套理论：`docs/theory/Glossary.md` 的「torch.multinomial」「随机种子」「Gumbel-Max」「PRNG」条目，以及 `docs/theory/op/flashinfer_top_k_top_p_sampling_from_probs.md`。
- **动手**：固定 `sampling_seed` 发两次相同请求验证可复现；对比 flashinfer 与 pytorch 采样后端的延迟。
- **自检**：top-k/top-p/min-p 的执行顺序是怎样的？为什么 flashinfer 后端不支持逐请求确定性采样？

### 阶段 L6（可选）：并行通信与量化接入（按需）

- **阅读**：`communicator.py`（`LayerCommunicator` / `ScatterMode` 与 all-reduce+layernorm 融合）；`model_parallel.py`；`quantization/` 子目录的 `base_config.py` + 一个具体方案（fp8/awq）看量化方法如何接进 `linear.py`。
- **动手**：在多卡下用 profiler 抓一次 forward，观察通信算子（all-reduce/all-gather）出现的位置与耗时占比。
- **自检**：能指出一层 decoder 里发生 TP 通信的所有位置，并说明量化方法是如何挂到线性层上的。

### 学习建议

1. **顺数据流读**：始终沿 `embedding → linear/norm/act → attention → logits → sampler` 这条线，不要孤立看单个文件。
2. **对照一个真实模型**：以 `llama.py` 为锚点，每学一层就回到模型里找它的调用点。
3. **融合算子重在"为什么融合"**：`elementwise.py` / `fused_sampling.py` 的价值是减少 kernel 启动与显存往返，理解动机比抠 Triton 细节更重要。
4. **理论与代码互证**：采样、logits 部分配合 `docs/theory/` 下的 Glossary 与 op 文档一起看。
