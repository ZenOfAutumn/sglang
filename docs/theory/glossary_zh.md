# SGLang 术语表（Glossary）

本文件收录 SGLang 源码与文档中常见的术语解释，便于查阅。

## 非填充 token 数（num_token_non_padded）

指一个批次（batch）里**真正有效、需要计算的 token 数量**，不包含为了对齐而补进去的 padding（占位）token。

**为什么会有 padding：**
在 DP 注意力（数据并行注意力）、MoE 专家并行（EP）等场景下，各 rank 的 batch 需要对齐到相同的 token 数；同时 CUDA Graph 也要求固定的输入 shape。为满足这些约束，会用占位 token 把 batch 补齐到统一长度。

**作用：**
`num_token_non_padded` 用来告诉 kernel “前多少个是真实 token”，从而跳过 padding 部分，避免无效计算和对结果的污染。

**相关字段（见 `python/sglang/srt/model_executor/forward_batch_info.py`）：**

- `num_token_non_padded`：GPU 侧张量，仅在 MoE 专家并行（EP>1）时构建，供 kernel 使用。
- `num_token_non_padded_cpu`：CPU 侧整数，始终记录，供调度/统计逻辑使用。

## DP 注意力（Data Parallel Attention，数据并行注意力）

指在多卡部署时，对**注意力层**采用数据并行：每个 rank 持有完整的注意力权重，各自独立处理 batch 中不同的请求（不同序列），不切分单个序列内部的计算。

**主要用途：**
在 DeepSeek 这类 MLA 模型上，注意力部分用 DP（每卡算各自的请求），而 MoE/MLP 部分仍用 TP/EP。这样可以避免 MLA 的 KV cache 在 TP 下被重复存储，显著节省显存、提升吞吐。

**代价：**
各 rank 的请求数 / token 数不同，需要在进入 MoE 等共享层之前做 token 数对齐（即 padding 与 `global_num_tokens` 同步，参见“非填充 token 数”）。

## 扩散式 LLM（Diffusion LLM / DLLM）

指在 SGLang 中以扩散模型（diffusion model）方式运行的 LLM：把生成过程建模为逐步去噪（denoising），而非传统的逐 token 自回归。

**特点：**
推理以固定大小的 **block（块）** 为单位进行，每个 block 包含多个 token 的位置，因此位置编码需要按块偏移展开。

**相关字段（见 `python/sglang/srt/model_executor/forward_batch_info.py`）：**

- `dllm_config.block_size`：每个块包含的 token 数。
- `dllm_block_offsets`：各块的起始偏移，配合 `block_size` 展开出完整的 `positions`。

## ngram embedding（N-gram 嵌入）

LongCat 等模型使用的一种嵌入机制：除常规的 token embedding 外，还根据 token 的 n-gram（连续若干 token 的组合）从一张 **token 表（`ne_token_table` / `token_table`）** 中查出额外的嵌入信息并融合进去，以增强模型对局部上下文 / 组合模式的表达。

**实现：**
通过 `NgramEmbeddingInfo` 记录每个请求在 token 表中的起始列（`column_starts`）与长度（`req_lens`），按 decode / extend 模式分别计算后供模型前向使用。

**相关字段（见 `python/sglang/srt/model_executor/forward_batch_info.py`）：**

- `NgramEmbeddingInfo`：保存 ngram embedding 的状态（token 表、列起始、请求长度等）。
- `_init_ngram_embedding_info`：按 decode / extend 模式构建本批次的 ngram embedding 信息。
- `model_runner.use_ngram_embedding`：是否启用该机制的开关。

