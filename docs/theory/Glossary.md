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

## 对数 softmax（log_softmax）

指对 softmax 的结果再取自然对数，即 $\log(\mathrm{softmax}(x))$，把一组 logits 转换成 **对数概率（log-probabilities）**。在 SGLang 中，采样器（`python/sglang/srt/layers/sampler.py`）用它来计算返回给用户的 token logprob，约束解码、speculative decoding 的接受判定等也依赖对数概率。

### 定义

对长度为 $K$ 的 logits 向量 $x=(x_1,\dots,x_K)$，第 $i$ 个分量的对数 softmax 为：

$$
\mathrm{log\_softmax}(x)_i = \log\frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}} = x_i - \log\sum_{j=1}^{K} e^{x_j}.
$$

右边的 $\log\sum_j e^{x_j}$ 就是 **log-sum-exp（LSE）**。由于它是概率的对数，所有输出都 $\le 0$，且对同一维度 $\exp$ 后求和为 1。

### 为什么不直接 `log(softmax(x))`

朴素做法「先算 softmax 再取 log」有两个数值问题：

- **指数溢出**：$x_j$ 较大时 $e^{x_j}$ 会上溢为 `inf`。
- **log(0) 下溢**：softmax 结果中极小的概率会被舍入成 0，再取 $\log$ 得到 $-\infty$。

`log_softmax` 通过 **减最大值（max-shift）** 的等价变形规避这两点。令 $m=\max_j x_j$：

$$
\mathrm{log\_softmax}(x)_i = (x_i - m) - \log\sum_{j=1}^{K} e^{x_j - m}.
$$

减去 $m$ 后，求和中至少有一项为 $e^0=1$，分母不会下溢为 0；同时所有指数项 $\le 1$，不会上溢。这个变形与原式**完全等价**（分子分母同乘 $e^{-m}$），但全程数值稳定——这也正是「在线 softmax」里维护 running max 的同一思想。

### 与 softmax 的关系

- $\mathrm{softmax}(x) = \exp(\mathrm{log\_softmax}(x))$，二者互为指数/对数关系。
- 配合 `NLLLoss` 使用时，`log_softmax + NLLLoss` 等价于交叉熵损失，但比「softmax → log → 乘加」更稳更快，因此训练里常直接用 `log_softmax`。
- 推理侧返回 logprob 时，对一批 logits 调用 `torch.nn.functional.log_softmax(logits, dim=-1)`，再按采样到的 token id 取出对应分量即可。

### 在 SGLang 中的意义

- **logprob 输出**：当请求开启 `return_logprob` 时，采样器对 logits 做 `log_softmax` 得到每个候选/选中 token 的对数概率返回给上层（见 `python/sglang/srt/layers/sampler.py`、`python/sglang/srt/layers/logits_processor.py`）。
- **数值稳定的接受判定**：speculative decoding 在比较 draft / target 分布、约束解码在做 mask 归一时，都在对数空间运算以避免极小概率被舍成 0。

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

## 多 tokenizer 模式（Multi-Tokenizer / Multi-HTTP-Worker Mode）

指用**多个进程并行承担 HTTP 接入 + 分词/反分词（tokenize / detokenize）** 的部署模式，用来突破单进程 Python（GIL）与单 HTTP server 的吞吐瓶颈。通过命令行参数 `--tokenizer-worker-num`（分词进程数）和 `--detokenizer-worker-num`（反分词进程数）开启，二者大于 1 时即进入该模式。

**为什么需要：**
默认情况下整个前端（HTTP 接收、请求校验、分词、把结果反分词成字符串再返回）都在单进程内完成。当并发请求很多、或分词/反分词本身较重（长文本、多模态）时，单进程会成为瓶颈，GPU 反而“吃不饱”。多 tokenizer 模式把这部分 CPU 密集工作横向扩展到多个进程，从而提升整体吞吐、降低排队延迟。

**架构（见 `python/sglang/srt/managers/multi_tokenizer_mixin.py`）：**

- **`TokenizerWorker`**：继承自 `TokenizerManager` 的工作进程，每个进程独立处理一部分 HTTP 请求并完成分词；启动时向路由注册自己的 IPC 地址。
- **`MultiTokenizerRouter`**：位于多个 worker 与 scheduler/detokenizer 之间的路由进程。前向：`worker → router → scheduler`；后向：`detokenizer → router → 对应 worker`；同时把 pause/continue 等控制广播给所有 worker，保证状态一致。
- **`MultiDetokenizerRouter` / `MultiHttpWorkerDetokenizerMixin`**：反分词侧的对应路由与混入逻辑，把调度器产出的 token 结果分发给多个反分词进程并行处理。
- 进程间通过 **ZMQ**（PUSH/PULL）通信，部分一次性资源（如 load snapshot 的 PULL socket）由单一 router 进程持有并经共享内存（SHM）下发给各 worker，避免多进程重复绑定。

**约束与注意事项（见 `python/sglang/srt/server_args.py`）：**

- 与 `--skip-tokenizer-init` 互斥：跳过分词器初始化时会强制把 worker 数重置为 1。
- 暂不支持与 `--enable-http2` 同时使用（`tokenizer_worker_num > 1` 会报错）。
- 请求需要由 router 正确路由回**发起该请求的那个 worker**，以便把反分词后的字符串结果返回给对应的 HTTP 连接。

## 增量解码（Incremental Detokenization）

指在**流式生成**过程中，每生成一步就把**新产生的 token** 实时还原成文本片段返回给用户，而不是等整条序列全部生成完再一次性解码。它是流式输出（如 SSE 逐字返回）的基础，由 `DetokenizerManager`（见 `python/sglang/srt/managers/detokenizer_manager.py`）实现。

### 为什么不能「每步只解码新 token」

直觉上，每步只把新增的那几个 token 单独 `decode` 一下、拼到已有文本后面即可。但这样会出错，根因是 **「token 序列 → 文本」不是逐 token 拼接**：

- 一个字符（尤其中文、emoji）可能由**多个 token** 编码，单独解码其中一个 token 得到的是不完整字节；
- tokenizer 在拼接处对**空格、特殊符号**的处理依赖上下文，孤立解码单个 token 会丢失这些信息。

因此必须**带着上下文**解码，并维护每个请求的解码进度状态。

### SGLang 的实现：带上下文的「窗口解码 + 相减」

每个请求在 `decode_status` 字典里（按 `rid` 索引）维护一份增量解码状态，核心是两个游标：

- **`surr_offset`（环绕上下文起点）**：解码时额外多带的一段前文起点，用来让跨 token 的字符 / 空格能正确拼接。
- **`read_offset`（已读取起点）**：已经提交给用户的文本对应的 token 边界。

每一步的处理（见 `_decode_batch_token_id_output`）：

1. 解码 `[surr_offset, 末尾]` 得到 `read` 文本（含本次新 token）；
2. 解码 `[surr_offset, read_offset)` 得到 `surr` 上下文文本；
3. **本次真正新增的文本 = `read` 去掉 `surr` 前缀**——多带上下文只是为了正确拼接，最后减掉避免重复输出；
4. 推进 `surr_offset = read_offset`、`read_offset = len(decode_ids)`，供下一步接续。

由于 `decode_status` 需为每个在途请求长期保存状态，它用「有界 + 自动淘汰最旧」（`LimitedCapacityDict`）来约束内存上限。

### 与边界问题的关系

增量解码的**正确性难点**集中在 token 交界处——这正是下一条「反向解码的边界问题」要解决的（UTF-8 字符被切断、批量 vs 逐行解码差异等）。简言之：增量解码是**机制**，边界问题是该机制必须处理好的**坑**。

## 反向解码的边界问题（Detokenization Edge Cases）

「反向解码（detokenization）」指把 token id 还原成文本字符串。它的**边界问题**指：在 **token 与 token 的交界处、文本片段的拼接处**，增量解码结果可能出错（乱码、多/少空格、特殊符号异常等）。这是因为「一个字符 ↔ 一个 token」并非一一对应——一个字符可能跨多个 token，文本也不是简单把每个 token 的解码结果拼接起来。

### 两类典型边界问题

**1. 跨 token 的字符被截断（UTF-8 边界问题）**

一个字符（尤其中文、emoji）可能由多个 token 编码而成。增量（流式）解码时若只解码到该字符的一半，会得到不完整的字节，显示为乱码 `�`。

SGLang 的处理方式（见 `python/sglang/srt/managers/detokenizer_manager.py` 的 `_decode_batch_token_id_output`）：

- **多带一段上下文 token（`surr`）一起解码**：每次解码 `[surr_offset, 末尾]` 得到 `read` 文本，同时解码 `[surr_offset, read_offset)` 得到 `surr` 文本，本次真正新增的文本 = `read` 去掉 `surr` 前缀。多带上下文是为了让跨 token 的字符能正确拼接。
- **`�` 检测 + 延迟提交**：若新增文本以 `�` 结尾，说明字符不完整，则只发送可打印前缀（`find_printable_text`）、**不推进 offset**，等下一批 token 到达后再重试解码，避免把乱码发给用户。

**2. 批量解码 vs 逐行解码结果不一致**

某些 tokenizer（如 **gpt-oss**）在 `batch_decode`（多行一起解码）与单行 `decode` 下，对特殊 token、token 间空格的处理存在细微差异，导致批量路径在边界处产生错误文本。

SGLang 的处理方式：提供 `--disable-tokenizer-batch-decode` 开关（`server_args.disable_tokenizer_batch_decode`）。开启后改为**逐行解码**来规避此类问题；默认走批量解码以获得更高性能。

### 小结

边界问题 = **多个 token 拼接成文本时，在交界处产生的解码错误**，主要包括 UTF-8 字符被切断（靠 `surr` 上下文 + `�` 检测延迟提交解决）和批量解码的行为差异（靠禁用批量、逐行解码解决）。

## 伪随机数生成器（PRNG）

**伪随机数生成器（Pseudo-Random Number Generator, PRNG）** 是一种用**确定性算法**产生「看起来随机」数字序列的方法。它不依赖物理熵源（如热噪声），而是从一个初始的**种子（seed）** 出发，用固定的递推公式不断算出下一个数。因此 PRNG 产生的并非真随机，而是**伪随机**：序列完全由种子决定——相同种子必得相同序列，这正是「随机种子」可复现性的根基。

### 基本原理：状态 + 递推 + 输出

任何 PRNG 都可抽象为三部分：

- **内部状态 $s$**：一段被持续更新的内存（从几十位到上千位不等）。
- **状态转移函数 $s_{t+1}=f(s_t)$**：用确定性公式把当前状态推进到下一个状态。
- **输出函数 $x_t=g(s_t)$**：从状态中提取出对外可见的随机数（常再归一化到 $[0,1)$）。

种子的作用就是设定初始状态 $s_0$。由于状态空间有限，序列最终一定会循环，循环前的长度称为**周期（period）**，好的 PRNG 周期极长（如 $2^{19937}-1$）。

### 衡量「随机性」的标准

伪随机虽非真随机，但要求在统计上**不可区分于真随机**：

- **均匀性**：输出在取值范围内近似均匀分布。
- **独立性 / 无相关**：前后数之间无可察觉的规律，能通过 TestU01、Diehard 等统计测试。
- **长周期**：避免在实际使用量级内重复。
- 注意：常规 PRNG **不要求密码学安全**——已知部分输出可能反推状态。需要安全性时要用 CSPRNG（如基于 AES/ChaCha）。

### 两类典型实现

**1. 有状态、序列式（stateful）**——主流通用 PRNG

- **线性同余（LCG）**：$s_{t+1}=(a\,s_t+c)\bmod m$，最简单但质量一般。
- **Mersenne Twister（MT19937）**：周期 $2^{19937}-1$、统计性质优秀，是 NumPy 旧默认、CPython `random` 模块的底层。
- **Philox / Threefry（counter-based，基于计数器）**：状态即「种子 + 计数器」，$x = f(\text{key},\ \text{counter})$。优点是**无需串行推进状态**，给定计数器即可并行、随机地直接算出第 $n$ 个数——非常适合 GPU。PyTorch 的 CUDA 随机数（`torch.multinomial`、dropout 等）正是基于 Philox。

**2. 无状态、哈希式（stateless / hash-based）**

不维护可变状态，而是把「种子 + 坐标（如位置、下标）」直接哈希成随机数：$x=\mathrm{hash}(\text{seed},\ \text{key})$。它本质上是 counter-based 思路的极端形式，天然可复现、可并行、与调用顺序无关。SGLang 的确定性采样用的 `murmur_hash32(seed, position, col)` 就属于此类（见下文「随机种子」与「Gumbel-Max」条目）。

### 在 SGLang 中的意义

- **全局 RNG（有状态）**：`torch.multinomial`、`torch.manual_seed` 等走 PyTorch 的全局生成器（CPU 用 MT19937，CUDA 用 Philox）。其状态随每次调用推进，因此在连续批处理下会被相邻请求「串扰」，难以按单个请求复现。
- **无状态 PRNG（哈希式）**：为实现与 batch 组合 / 调度顺序无关的**确定性采样**，SGLang 改用 `murmur_hash32` 这一无状态 PRNG，把「请求种子 + token 位置 + 词表列」直接哈希成随机源，再转成 Gumbel 噪声做 Gumbel-Max 采样（见 `python/sglang/srt/layers/sampler.py` 的 `multinomial_with_seed`）。

## 随机种子（Random Seed）

**随机种子**是喂给伪随机数生成器（PRNG）的一个整数初值。计算机里的“随机”其实是**确定性算法**算出来的伪随机序列：给定相同的种子，就会得到**完全相同**的随机数序列。因此种子的核心价值是 **可复现性（reproducibility）**——固定种子后，同样的输入能稳定复现同样的输出，便于调试、对拍、回归测试与做基准评测。

在 LLM 推理里，“随机”主要出现在**采样（sampling）**环节：当 `temperature > 0` 时，下一个 token 不是取概率最高的那个，而是按概率分布**随机抽样**得到，于是同一个 prompt 多次生成会得到不同结果。引入随机种子，就能让这种随机采样变得可控、可复现。

SGLang 中的随机种子分两个层面：

**1. 全局框架种子（`random_seed`）**

服务级别的种子，用于框架初始化阶段各处的随机性（如部分权重初始化、调试用随机数据等）。

- 见 `python/sglang/srt/server_args.py`：CLI 参数 `--random-seed`；字段 `random_seed`，**默认为 `None`，此时会随机取一个值** `random.randint(0, 1 << 30)`，所以不显式指定时每次启动的种子并不固定。

**2. 请求级采样种子（`sampling_seed`）——确定性采样**

每个请求可单独携带的采样种子，用于让该请求的 token 采样**确定可复现**。

- 见 `python/sglang/srt/sampling/sampling_params.py`：`SamplingParams.sampling_seed`（请求级，默认 `None`）。
- 见 `python/sglang/srt/sampling/sampling_batch_info.py`：批次内各请求的 `sampling_seed` 被收集成张量，随 batch 一起下发到采样 kernel。

**确定性采样的实现原理（见 `python/sglang/srt/layers/sampler.py` 的 `multinomial_with_seed`）：**

普通采样用 `torch.multinomial`，其随机性依赖**全局 RNG 状态**——在连续批处理（continuous batching）下，请求的批次组合、执行顺序随时变化，全局 RNG 状态会被“串扰”，导致同一请求难以复现。SGLang 用一种**无状态、按位置可复现**的方案替代：

1. 用 `murmur_hash32(seed, positions, col_indices)` 把「请求种子 + token 在序列中的位置 + 词表列下标」哈希成一个均匀随机值——种子和位置一起参与哈希，保证每个位置都有**唯一且可复现**的随机源，且不依赖任何全局状态。
2. 把哈希值映射到 $[0,1]$ 均匀分布，再转成 **Gumbel 噪声**（$-\log(-\log(x))$）。
3. 给 logits 加上 Gumbel 噪声后取 `argmax`——这就是 **Gumbel-Max 技巧**，在数学上等价于按 softmax 概率分布做一次随机抽样，但全程确定（同样的种子+位置必得同样结果）。

### Gumbel-Max 技巧为何等价于按 softmax 抽样

**结论（Gumbel-Max 定理）：** 设有一组未归一化的对数概率（logits）$\ell_1,\dots,\ell_K$，对应的 softmax 概率为

$$
p_k = \frac{e^{\ell_k}}{\sum_{j=1}^{K} e^{\ell_j}}.
$$

独立地为每个类别采一份 Gumbel(0,1) 噪声 $g_k$，则

$$
\arg\max_k\ (\ell_k + g_k)
$$

所选中类别的分布**恰好就是** $\mathrm{Categorical}(p_1,\dots,p_K)$。也就是说，“加噪声取最大”与“直接按 $p_k$ 抽样”在分布上完全一致。

**Gumbel 噪声怎么来：** 取 $x\sim\mathrm{Uniform}(0,1)$，令

$$
g = -\log(-\log x),
$$

得到的 $g$ 服从标准 Gumbel 分布。其累积分布函数（CDF）为 $F(g)=\exp(-e^{-g})$（这正对应代码里 `x.log_().neg_(); x.log_().neg_()` 两次 $-\log$ 的操作）。

**证明：** 记 $z_k=\ell_k+g_k$。由 Gumbel 的 CDF 可得每个 $z_k$ 的 CDF 是平移后的 Gumbel：

$$
\Pr(z_k \le t)=\exp\!\big(-e^{-(t-\ell_k)}\big)=\exp\!\big(-e^{\ell_k}e^{-t}\big),
$$

对应密度为 $f_k(t)=e^{\ell_k}e^{-t}\exp(-e^{\ell_k}e^{-t})$。类别 $k$ 被选中，当且仅当 $z_k$ 是所有 $z_j$ 中的最大值，即对所有 $j\neq k$ 有 $z_j\le z_k$。对 $z_k=t$ 积分，并利用各 $z_j$ 相互独立：

$$
\begin{aligned}
\Pr(k \text{ 最大})
&= \int_{-\infty}^{\infty} f_k(t)\prod_{j\neq k}\Pr(z_j\le t)\,dt \\
&= \int_{-\infty}^{\infty} e^{\ell_k}e^{-t}\exp\!\big(-e^{\ell_k}e^{-t}\big)\prod_{j\neq k}\exp\!\big(-e^{\ell_j}e^{-t}\big)\,dt \\
&= \int_{-\infty}^{\infty} e^{\ell_k}e^{-t}\exp\!\Big(-\big(\textstyle\sum_{j} e^{\ell_j}\big)e^{-t}\Big)\,dt.
\end{aligned}
$$

令 $S=\sum_j e^{\ell_j}$，并换元 $u=e^{-t}$（则 $du=-e^{-t}\,dt$，即 $e^{-t}\,dt=-du$，积分限 $t:-\infty\to\infty$ 对应 $u:\infty\to 0$）：

$$
\Pr(k \text{ 最大})
= \int_{0}^{\infty} e^{\ell_k}\,e^{-S u}\,du
= e^{\ell_k}\cdot\frac{1}{S}
= \frac{e^{\ell_k}}{\sum_{j} e^{\ell_j}}
= p_k.
$$

正好等于 softmax 概率 $p_k$，证毕。

**为什么对确定性采样有用：** 抽样的全部随机性都被“外包”给了 Gumbel 噪声 $g_k$；而 SGLang 用 `murmur_hash32(seed, position, col)` 来**确定性地生成**这份噪声——只要 `seed` 和 token 位置相同，每个类别（词表列）拿到的 $g_k$ 就完全相同。于是 $\arg\max_k(\ell_k+g_k)$ 也完全确定。这样既保留了“按 softmax 概率分布抽样”的正确统计行为（不是贪心、不是近似），又彻底摆脱了对全局 RNG 状态的依赖，从而与 batch 组合、调度顺序无关。

**两个实现细节：**

- **温度（temperature）** 体现在 $\ell_k$ 上：采样前 logits 通常已除以温度（$\ell_k/T$）。$T\to 0$ 时分布趋于 one-hot，Gumbel-Max 退化为对原始 logits 取 argmax，即贪心解码；$T$ 越大分布越平、采样越随机。
- **数值稳定**：代码中 Gumbel 噪声与 logits 的运算保持在 `float64`，并对 $-\log x$ 做了 `clamp`，避免 $x$ 极小时 $\log$ 溢出（见 `multinomial_with_seed` 的注释）。

因此：**只要设定相同的 `sampling_seed`，无论该请求和哪些请求拼成一个 batch、batch 怎么调度，采样结果都完全一致**，从而在高吞吐的连续批处理下依然实现可复现的确定性推理。

## 多项式采样（torch.multinomial）

`torch.multinomial(input, num_samples)` 按给定的**概率权重分布**进行**有/无放回的随机抽样**，返回被抽中类别的**下标**。在 LLM 推理里，它是「随机采样」路径的核心算子：把模型输出经 softmax 得到的概率 $p=(p_1,\dots,p_K)$ 当作 input，抽出下一个 token 的词表下标。

### 输入与语义

- `input`：形状 $(K,)$ 或 $(B, K)$ 的**非负权重**张量，**不要求每行和为 1**（内部会自动归一化）；为 0 的项不会被抽中。
- `num_samples`：每行抽取的样本数。SGLang 解码每步只需一个 token，故固定 `num_samples=1`。
- `replacement`：是否放回，默认 `False`。当 `num_samples=1` 时放回与否无差别。
- 返回被抽中类别的**下标**（不是概率值），再据此从词表/排序索引里取回真正的 token id。

数学上，第 $i$ 类被抽中的概率为按权重归一化的结果：

$$
\Pr(\text{抽中 } i) = \frac{p_i}{\sum_{j=1}^{K} p_j}.
$$

### 实现原理：逆变换采样（inverse-CDF）

无放回 / `num_samples=1` 时，常见实现是**累积分布 + 均匀随机数二分**：

1. 归一化权重并求前缀和（CDF）：$C_i=\sum_{j\le i} p_j / \sum_j p_j$，得到单调递增、终值为 1 的序列。
2. 从全局 RNG 取一个均匀随机数 $u\sim\mathrm{Uniform}(0,1)$。
3. 找到第一个满足 $C_i \ge u$ 的下标 $i$（对 CDF 做二分查找），即为抽样结果。

由于 $u$ 落入区间 $[C_{i-1}, C_i)$ 的概率正好等于该区间长度 $p_i/\sum_j p_j$，所以抽中第 $i$ 类的概率恰为其归一化权重——这就是「逆变换采样」。多样本有放回时则重复取 $u$；GPU 上 PyTorch 会用并行化的变体（如别名法 / 批量二分）实现。

### 关键特性：依赖全局 RNG 状态

`torch.multinomial` 的随机性来自步骤 2 的 $u$，而 $u$ 取自 **全局 RNG 状态**（CPU/CUDA generator）。这意味着：

- **结果可被全局种子影响**：`torch.manual_seed` 会改变后续所有 `multinomial` 的输出。
- **顺序敏感、难以按请求复现**：在连续批处理（continuous batching）下，请求每步拼成的 batch、执行顺序不断变化，全局 RNG 会被相邻请求「串扰」，导致同一请求难以稳定复现。这正是 SGLang 在需要确定性采样时改用 **Gumbel-Max + `murmur_hash32`**（见「随机种子」条目的 `multinomial_with_seed`）而非直接 `torch.multinomial` 的原因。

### 在 SGLang 中的用法（见 `python/sglang/srt/layers/sampler.py`）

- **简单情形（无截断）**：直接对 softmax 概率调用 `torch.multinomial(probs, num_samples=1)` 抽下一个 token。
- **复杂情形（top-k / top-p / min-p）**：先对概率排序、按阈值做掩码置零并重归一化，再对处理后的 `probs_sort` 调 `torch.multinomial`，最后用 `torch.gather` 把排序下标映射回真实 token id。
- **确定性采样开关**：当请求带 `sampling_seed` 时，对应分支改走 `multinomial_with_seed`（Gumbel-Max），与 `torch.multinomial` 在分布上等价，但**不依赖全局 RNG**，从而可复现。

## WAR 屏障（Write-After-Read Barrier，写后读屏障）

**WAR 屏障**是 SGLang 重叠调度（`event_loop_overlap`）中用来防止**「写后读」数据竞争（Write-After-Read hazard）**的一个 CUDA stream 间同步点。它确保「本轮调度对某块共享 GPU 缓冲的**写入**」一定发生在「上一轮前向对同一块缓冲的**读取**完成之后」。

### 背景：两条并行的 CUDA stream

重叠调度把工作拆到两条 stream 上并发执行（见「在线 softmax」无关，这里是流水线并行机制）：

- **调度 stream（`schedule_stream`）**：跑 CPU 端调度准备所产生的 GPU 操作（如把下一批的输入写入共享缓冲）。
- **前向 stream（`forward_stream`）**：跑模型前向计算。

正是这种「第 N 轮调度」与「第 N-1 轮前向」并行（参见「`event_loop_overlap`」的重叠原理），才会引出跨 stream 的读写顺序问题。

### 为什么需要它：WAR 冒险

「写后读冒险」指：**一个读操作尚未完成，另一个写操作就抢先覆盖了它要读的数据**，导致读到被污染的新值。在重叠调度里：

- 第 N-1 轮的**前向**还在 `forward_stream` 上**读取**某块共享 GPU 缓冲（如 input_ids / seq_lens 等暂存区）；
- 第 N 轮的**调度**已在 `schedule_stream` 上准备**写入**同一块缓冲。

两条 stream 各自异步执行，若不加约束，调度的写可能在前向的读还没做完时就发生，造成数据竞争、结果错误。

### 实现：用 `wait_stream` 跨 stream 等待

在每轮调度开始前插入一条屏障，让调度 stream 等待前向 stream 把上一轮的活干完：

```python
# WAR barrier: this iter's schedule writes to shared GPU buffers
# wait for prev forward's reads.
if self._war_barrier_enabled:
    self.schedule_stream.wait_stream(self.forward_stream)
```

`wait_stream` 不阻塞 CPU，只是在 GPU 上让 `schedule_stream` 后续的操作排在 `forward_stream` 当前已入队操作之后执行，从而把「写」排到「读」之后。

### 开关与例外（`_war_barrier_enabled`）

- **默认在 CUDA 上开启**（或显式设 `SGLANG_ENABLE_WAR_BARRIER`）。
- **DFLASH 投机解码下关闭**：DFLASH 用 `verify_done` / plan-stream 依赖等**更细粒度**的同步自行保护其对共享 `req_to_token` 的写入，无需这个**全局**屏障，关闭可减少不必要的串行化。

> 注意区分方向：本屏障是「调度等前向」（写等读，WAR）；而 `run_batch` 里还有一处反向的 `forward_stream.wait_stream(schedule_stream)`，那是「前向等调度」，保证前向所依赖的调度准备已就绪（属于 RAW，读等写），两者配合维持跨 stream 的正确时序。

