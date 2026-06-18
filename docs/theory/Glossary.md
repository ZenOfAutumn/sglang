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

## 在线 softmax（Online Softmax）

一种**单遍（one-pass）、数值稳定地增量计算 softmax** 的方法。它是 FlashAttention 等高效注意力 kernel 的数学基础：在不把完整的注意力分数矩阵 $S = QK^\top$ 写回显存的前提下，一边遍历 Key/Value 分块，一边累积出最终的注意力输出。

### 为什么需要它

标准 softmax 为了数值稳定，需要先减去最大值，公式为：

$$
\mathrm{softmax}(x_i) = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}, \quad m = \max_j x_j
$$

这要求**先扫一遍**所有 $x_j$ 求出全局最大值 $m$ 和分母 $\sum_j \exp(x_j - m)$，**再扫一遍**做归一化——即“两遍（two-pass）”，且必须把整行分数都保存下来。在注意力里，分数矩阵 $S$ 的大小是 $\text{seq\_len} \times \text{seq\_len}$，长序列下既占显存又频繁读写 HBM，成为带宽瓶颈。

在线 softmax 把“求最大值、求分母、加权求和”三件事**融合进同一遍遍历**，只需维护几个标量/向量状态，无需保存整行分数。

### 核心：增量更新

遍历到一个新的分块时，只维护三个运行中的量：

- $m$：到目前为止见过的最大分数（running max）
- $l$：到目前为止的指数和分母（running sum，已按当前 $m$ 校正）
- $O$：到目前为止的加权 value 累积（running output）

当新分块带来更大的最大值 $m_{\text{new}} > m_{\text{old}}$ 时，**已经累积的 $l$ 和 $O$ 都要乘以一个校正因子** $\exp(m_{\text{old}} - m_{\text{new}})$ 进行“缩放对齐”，再把新分块的贡献加进来：

$$
\begin{aligned}
m_{\text{new}} &= \max(m_{\text{old}}, m_{\text{block}}) \\
\text{correction} &= \exp(m_{\text{old}} - m_{\text{new}}) \\
l_{\text{new}} &= l_{\text{old}} \cdot \text{correction} + \sum_{\text{block}} \exp(s - m_{\text{new}}) \\
O_{\text{new}} &= O_{\text{old}} \cdot \text{correction} + \sum_{\text{block}} \exp(s - m_{\text{new}}) \cdot V_{\text{block}}
\end{aligned}
$$

遍历结束后，用 $O / l$ 得到最终注意力输出。可以证明，这个结果与一次性对整行做标准 softmax **完全等价**（不是近似），同时全程数值稳定（始终减去当前最大值，不会指数溢出）。

### 在 SGLang 中的意义

- 各类 FlashAttention / FlashInfer / Triton 注意力后端（见 `python/sglang/srt/layers/attention/`）都依赖在线 softmax 实现分块（tiling）计算，从而支持长上下文而不爆显存。
- 它也是 **KV cache 分页（paged）注意力**、**chunked prefill** 等机制能够分块/分页处理注意力的前提——因为可以按 KV 块逐步累积，无需一次性看到完整序列。
- 与之相关的 “**log-sum-exp（LSE）**” 即上面的 $m + \log(l)$，在 speculative decoding、注意力结果跨设备/跨分块合并（如 ring attention、DP attention 的分块归并）时用于把多段局部 softmax 结果正确地拼接起来。

## 预热（Warmup）

指服务启动后、正式对外提供服务之前，先用**少量构造好的请求**把整条推理链路完整地“跑通”几遍，让各种**一次性的、惰性触发（lazy）的初始化开销**提前发生，从而保证**真实用户的首个请求不会被这些首次开销拖慢**。

### 目的

- **消除首请求的高延迟尖刺（cold start）**：许多耗时的初始化是“用到才做”的，如果不预热，这些开销会全部砸在第一个真实请求上，造成首 token 延迟（TTFT）异常高、超时甚至触发上游熔断。
- **稳定性与可预测性**：预热后系统进入“热”状态，延迟分布更平稳，便于做容量规划、压测基线与 SLA 评估。
- **暴露启动期问题**：在真正接客前，用可控请求验证显存是否够用、各后端能否正常前向，提前发现 OOM 或配置错误。

### 原理：预热到底“热”了什么

预热请求与真实请求走**完全相同的代码路径**，因此会触发以下首次开销，并把结果缓存下来供后续复用：

- **CUDA Graph 捕获**：SGLang 在启动阶段会按一组预设 batch size 捕获 decode/prefill 的 CUDA Graph（见 `python/sglang/srt/model_executor/model_runner.py` 中的 `init_decode_cuda_graph` / `init_prefill_cuda_graph`）。预热请求确保这些图被实际执行并校验通过。
- **JIT / 编译**：`torch.compile`、Triton kernel 的首次编译，以及 deep_gemm 等 JIT kernel 的生成与缓存。
- **kernel autotune**：部分注意力 / MoE / GEMM kernel 在首次运行时会做自动调优（autotune）选择最优配置（参见 FLA 的 `autotune_cache`），预热让调优在接客前完成。
- **显存分配与内存池预热**：KV cache 池、各类工作缓冲区的首次分配，以及 CUDA caching allocator 的内存块预留，避免真实请求时再临时申请、触碰碎片化路径。
- **通信链路初始化**：多卡场景下 NCCL / 自定义 all-reduce 等通信算子的首次建链与缓冲区分配（见 `python/sglang/srt/distributed/`）。
- **惰性导入与对象构建**：部分 Python 模块、采样器、约束解码（grammar）等组件的首次构建。

### 在 SGLang 中的实现

- **通用启动预热**：服务启动时由后台线程执行（见 `model_runner.py` 同级 entrypoints 中的 `_wait_and_warmup` / `_execute_server_warmup`）——先轮询 `/model_info` 等待服务就绪，再发送一个真实推理请求把上述开销跑通，日志打印 `Warmup ended`。
- **自定义预热任务**：通过 `--warmups` 指定（如 `voice_chat` 等），由 `entrypoints/warmup.py` 的 `execute_warmups` 按名称注册并执行，可针对特定场景（多模态、PD 分离等）定制预热流量。

