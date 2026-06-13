# Attention Backend(注意力后端)

SGLang 支持种类繁多的注意力后端。每一种都各有优缺点。
你可以根据自己的需求进行测试。

```{important}
选择最优的注意力后端对最大化性能至关重要。不同的后端在不同场景下表现各异,因此请根据你的模型、硬件和使用场景进行选择。并非所有后端都在所有平台和模型架构上受支持。

如果你不指定 `--attention-backend`,SGLang 会尽最大努力根据你的硬件和模型架构自动选择性能最佳的后端。
```

## 支持矩阵

支持矩阵分为两部分:MHA(标准注意力)和 MLA(multi-head latent attention,多头潜在注意力)。关于 MHA 与 MLA 之间关键区别的解释,请参阅 [SGLang 关于 DeepSeek MLA 的文档](../basic_usage/deepseek_v3.md#multi-head-latent-attention-mla-throughput-optimizations) 以及原始 [DeepSeek MLA 论文](https://arxiv.org/pdf/2405.04434)。

### MHA 后端

| **Backend**                     | **Page Size > 1 (native)** | **FP8 KV Cache** | **FP4 KV Cache** | **Spec topk=1** | **Spec topk>1** | **Sliding Window** | **MultiModal** |
|---------------------------------|-----------------------------|------------------|-----------------|-----------------|-----------------|--------------------|----------------|
| **FlashInfer**                  | ✅                          | ✅               | ❌              | ✅              | ✅              | ✅                 | ❌             |
| **FA3 (FlashAttention 3)**      | ✅                          | ✅               | ❌              | ✅              | ✅              | ✅                 | ✅             |
| **FA4 (FlashAttention 4)**      | 128                         | ❌               | ✅              | ❌              | ❌              | ❌                 | ✅             |
| **Triton**                      | ❌                          | ✅               | ✅              | ✅              | ✅              | ✅                 | ✅             |
| **Torch Native (SDPA)**         | ❌                          | ✅               | ✅              | ❌              | ❌              | ❌                 | ✅             |
| **FlexAttention (PyTorch)**     | ❌                          | ❌               | ✅              | ❌              | ❌              | ❌                 | ❌             |
| **TRTLLM MHA**                  | 16, 32 or 64                | ✅               | ✅              | ✅              | ❌              | ✅                 | ❌             |
| **Dual Chunk FlashAttention**   | ✅                          | ❌               | ❌              | ❌              | ❌              | ❌                 | ❌             |
| **AITER (ROCm)**                | ✅                          | ✅               | ❌              | ✅              | ✅              | ✅                 | ✅             |
| **Wave (ROCm)**                 | ✅                          | ❌               | ❌              | ❌              | ❌              | ❌                 | ❌             |
| **Ascend (NPU)**                | ✅                          | ❌               | ❌              | ✅              | ❌              | ✅                 | ✅             |
| **Intel XPU**                   | ✅                          | ❌               | ❌              | ❌              | ❌              | ✅                 | ❌             |
| **Intel AMX (CPU)**             | ❌                          | ❌               | ❌              | ❌              | ❌              | ❌                 | ❌             |

### MLA 后端

| **Backend**                | **Native Page Sizes**     | **FP8 KV Cache** | **FP4 KV Cache** | **Chunked Prefix Cache** | **Spec topk=1** | **Spec topk>1** |
|----------------------------|---------------------------|------------------|------------------|--------------------------|-----------------|-----------------|
| **FlashInfer MLA**         | 1                         | ❌               | ✅               | ✅                       | ✅              | ❌              |
| **FlashMLA**               | 64                        | ✅               | ✅               | ✅                       | ✅              | ❌              |
| **Cutlass MLA**            | 128                       | ✅               | ✅               | ✅                       | ✅              | ❌              |
| **TRTLLM MLA (Blackwell)** | 32 or 64                  | ✅               | ✅               | ✅                       | ✅              | ❌              |
| **FA3 (FlashAttention 3)** | n/a                       | ❌               | ❌               | ✅                       | ✅              | ⚠️ (page_size=1 only) |
| **Triton**                 | n/a                       | ❌               | ❌               | ❌                       | ✅              | ⚠️ (page_size=1 only) |
| **FA4**                    | 1                         | ❌               | ✅               | ✅                       | ❌              | ❌              |
| **Ascend MLA (NPU)**       | 128                       | ❌               | ❌               | ❌                       | ❌              | ❌              |

```{note}
多模态注意力通过 `--mm-attention-backend` 选择。"MultiModal" 列表示该后端族是否存在对应的多模态实现。
```

```{note}
- FlashAttention 4 在 SM90(Hopper)和 SM100(Blackwell)上同时支持 prefill 和 decode。FA4 MLA 支持 `page_size = 1`;FA4 MHA 要求 `page_size = 128`。在 SM100 上,这由服务器自动强制;在 SM90 上,用户必须手动设置 `--page-size 128`。
- NSA 专为 [DeepSeek V3.2 DSA](https://lmsys.org/blog/2025-09-29-deepseek-V32/) 设计。详情请参阅 [DSA Attention Backend (NSA)](#dsa-attention-backend-nsa) 章节和 [DeepSeek V3.2 部署指南](../basic_usage/deepseek_v32.md)。
```

```{warning}
**Hopper(SM90)上的 FA4:** 由于缺乏 SplitKV 支持,FA4 的 decode 速度会随序列长度增长而下降。在 batch=1 时,相比 H100 上的 FA3:在 2K tokens 时约 -10%,4K 时约 -18%,8K 时约 -31%,16K 时约 -49%。更大的 batch size 会缩小差距(例如 batch=8:2K 时约 -2%,4K 时约 -8%)。Blackwell(SM100)不受影响。
```

```{note}
对于 KV4 FA4 场景,FA4 需要使用不同的 --decode-attention-backend 来运行。除了 trtllm_mha 与 FA4 不兼容之外,所有其他 decode 后端的行为均如表所示。
```

```{tip}
Speculative decoding topk:`topk` 是每一步从草稿模型采样的草稿 token 数量。`topk = 1` 遵循经典的 EAGLE;`topk > 1` 会探索多个分支,且要求草稿和验证路径中的后端都提供支持。
```

```{note}
**Speculative Decoding V2 (Spec V2):** Spec V2 使用重叠调度(`SGLANG_ENABLE_SPEC_V2=True`),可使各种注意力后端受益。它需要 `--speculative-eagle-topk 1`,并且目前适用于 EAGLE 和 EAGLE3。

**已验证的后端:** TRTLLM MLA、TRTLLM MHA、FA3、Ascend(NPU)、Triton。

**有限支持:** FlashInfer 可在 Spec V2 下运行,但其 plan stream(用于 split-KV 优化)引入了一个同步点,限制了重叠带来的收益。
```

```{tip}
Page size 控制将多少个 token 分组到一个 KV cache 块中。要使 prefix cache 生效,token 数量必须至少填满一个完整的 page。例如,如果你的 prompt 只有 32 个 token 而 `page_size = 64`,它无法填满一个完整的 page,因此无法在 prefix cache 中被匹配(page 不能被填充)。当有 65 个 token 且 `page_size = 64` 时,只有前 64 个 token 的第一个 page 会被缓存和匹配;剩余的 1 个 token 会被丢弃。使用 `page_size = 1` 可获得最大的前缀复用(token 级别匹配)。请注意,较大的 page size 通常能提升注意力 kernel 性能,因此当 prefix cache 复用不重要时,优先选择 `page_size > 1`。
```

许多本身不在 page 上原生运行的后端,可以通过将 page table 扩展为逐 token 索引,在 wrapper 层模拟 `page_size > 1`。"Page Size > 1 (native)" 列表示真正的 kernel 内分页。某些后端要求固定的原生 page size,无法被缩减/以其他方式模拟:TRTLLM MHA(16/32/64)、TRTLLM MLA(32/64)、FlashMLA(64)、Cutlass MLA(128)、Ascend(128)。

MLA page-size 约束:
- FlashInfer MLA:page_size = 1。
- FlashMLA:page_size = 64。
- Cutlass MLA:page_size = 128。
- TRTLLM MLA:page_size ∈ {32, 64}。

### GDN 注意力后端

GDN(Gated Delta Network)是一种具有 O(n) 复杂度的线性注意力机制,用于将 GDN 线性注意力层与标准全注意力层交替排列的混合模型中。GDN **不**通过 `--attention-backend` 选择;当模型架构需要时(例如 Qwen 3.5、Qwen 3 Next、Jet Nemotron、Jet VLM),它会自动激活。

GDN 线性注意力层有自己的 kernel 后端,通过 `--linear-attn-backend` 选择(默认:`triton`)。你可以使用 `--linear-attn-decode-backend` 和 `--linear-attn-prefill-backend` 按阶段覆盖 kernel。

| **Backend**              | **Decode** | **Prefill / Extend** | **Spec Decoding (Target Verify)** |
|--------------------------|------------|----------------------|-----------------------------------|
| **Triton (CUDA)**        | ✅         | ✅                   | ✅                                |
| **Triton (AMD/ROCm)**    | ✅         | ✅                   | ✅                                |
| **Triton (NPU)**         | ✅         | ✅                   | ❌                                |
| **Triton (CPU)**         | ✅         | ✅                   | ❌                                |
| **CuTe DSL (CUDA only)**| ✅         | ❌                   | ❌                                |

```{important}
GDN 模型是混合模型:全注意力层仍然需要一个标准的 `--attention-backend`。混合 GDN 模型上全注意力后端的平台约束:
- **Blackwell(例如 B200)**:仅 `triton`、`trtllm_mha` 或 `fa4`。
- **NPU(Ascend)**:仅 `ascend`。
- **AMD(ROCm)**:推荐 `triton`。
- **其他 CUDA(Hopper、Ampere 等)**:自动选择即可工作;无特殊约束。
```

### DSA Attention Backend (NSA)

DSA(Deepseek Sparse Attention)是 [DeepSeek V3.2](https://lmsys.org/blog/2025-09-29-deepseek-V32/) 使用的原生稀疏注意力机制。当模型架构需要时它会自动激活,并通过 `--attention-backend nsa` 选择。

在内部,NSA 后端会为 prefill 和 decode 阶段分派到不同的子后端。你可以通过 `--nsa-prefill-backend` 和 `--nsa-decode-backend` 覆盖这些设置:

| **Sub-backend**       | **Prefill** | **Decode** | **Notes**                                     |
|-----------------------|-------------|------------|-----------------------------------------------|
| **flashmla_sparse**   | ✅          | ✅         | Hopper 和 Blackwell(bf16)上的默认 prefill |
| **flashmla_kv**       | ✅          | ✅         | Blackwell 上带 DP 的 FP8 默认 decode   |
| **flashmla_auto**     | ✅          | ❌         | 根据 kv_cache_dtype 自动选择 flashmla_sparse 或 flashmla_kv |
| **fa3**               | ✅          | ✅         | Hopper(bf16)上的默认 decode               |
| **trtllm**            | ✅          | ✅         | Blackwell(bf16)上的默认 decode;在 Blackwell 上不带 DP 时两者的默认值 |
| **tilelang**          | ✅          | ✅         | AMD(ROCm)上的默认值                         |
| **aiter**             | ✅          | ✅         | AMD 专用 kernel 库(需要 aiter 包) |

部署示例请参阅 [DeepSeek V3.2 部署指南](../basic_usage/deepseek_v32.md)。

### 混合注意力(prefill 和 decode 使用不同后端)(实验性)

```{warning}
混合注意力是一项实验性功能。
```

你可以为 prefill 和 decode 混搭注意力后端。当某个后端擅长 prefill 而另一个擅长 decode 时,这非常有用。关于实现细节,请参阅 `python/sglang/srt/layers/attention/hybrid_attn_backend.py`。

```bash
# Example: Prefill with FA4, Decode with TRTLLM MLA (Blackwell)
python3 -m sglang.launch_server \
  --model-path nvidia/DeepSeek-R1-FP4 \
  --tp 8 \
  --attention-backend trtllm_mla \
  --moe-runner-backend flashinfer_trtllm \
  --quantization modelopt_fp4 \
  --prefill-attention-backend fa4
```

#### 使用混合注意力的 speculative decoding

混合注意力也可与 speculative decoding 配合使用。用于草稿 decode 和目标验证的后端取决于 `--speculative-attention-mode`:

- `--speculative-attention-mode decode`(推荐):草稿/验证使用 decode 后端。
- `--speculative-attention-mode prefill`(默认):草稿/验证使用 prefill 后端。

将混合注意力与 speculative decoding 结合时的约束:

- 如果任一注意力后端是 `trtllm_mha`,speculative decoding 仅支持 `--speculative-eagle-topk 1`。
- 对于带 `--page-size > 1` 和 `--speculative-eagle-topk > 1` 的分页 MHA 后端,仅支持 `flashinfer`。
- CUDA Graph:decode 后端总是被捕获;只有当 `--speculative-attention-mode prefill` 时才会捕获 prefill 后端。


```{tip}
如果你只设置了 `--prefill-attention-backend` 或 `--decode-attention-backend` 其中之一,未指定的阶段会继承 `--attention-backend`。
如果两者都指定且不同,SGLang 会自动启用一个混合 wrapper,按阶段分派到所选后端。
```

## 注意力后端选择指南(CUDA)

如果未指定 `--attention-backend` 参数,SGLang 会根据硬件(CUDA)和模型架构自动选择最佳后端。

### 自动选择逻辑

**1. MHA 模型(例如 Llama、Qwen)**
- **Hopper(例如 H100、H200)**:如果使用 CUDA 12.3+ 且模型配置受支持,默认为 `fa3`。
- **Blackwell(例如 B200)**:默认为 `trtllm_mha`,除非使用带 `topk > 1` 的 speculative decoding。
- **其他架构(Ampere、Ada 等)**:如果可用则默认为 `flashinfer`;否则回退到 `triton`。

**2. MLA 模型(例如 DeepSeek V3)**
- **Hopper**:默认为 `fa3`(需要 CUDA 12.3+)。
- **Blackwell**:默认为 `flashinfer`;针对 DeepSeek V3 模型会专门自动选择 `trtllm_mla`。
- **其他架构**:默认为 `triton`。


## 用户指南

### 不同注意力后端的启动命令

- FlashInfer(非 Hopper 机器的默认值,例如 A100、A40)
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend flashinfer
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-V3 \
  --attention-backend flashinfer \
  --trust-remote-code
```

- FlashAttention 3(Hopper 机器的默认值,例如 H100、H200、H20)
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend fa3
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-V3 \
  --trust-remote-code \
  --attention-backend fa3
```

- Triton
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend triton
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-V3 \
  --attention-backend triton \
  --trust-remote-code
```

- FlashMLA
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend flashmla \
  --trust-remote-code
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend flashmla \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code
```

- TRTLLM MLA(针对 Blackwell 架构优化,例如 B200)
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend trtllm_mla \
  --trust-remote-code
```

- 带 FP8 KV Cache 的 TRTLLM MLA(更高并发,更低内存占用)
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend trtllm_mla \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code
```

- TRTLLM MHA(针对 Blackwell 架构优化,例如 B200)
```bash
python3 -m sglang.launch_server \
  --tp 4 \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --attention-backend trtllm_mha \
  --trust-remote-code
```

- TRTLLM MHA(XQA backend)(针对 SM90 和 SM120 优化,例如 H20、H200、5090)
  请注意,TRTLLM XQA 后端仅在 pagesize 64 时工作良好。
```bash
python3 -m sglang.launch_server \
  --tp 4 \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --decode-attention-backend trtllm_mha \
  --trust-remote-code
```

- FlashAttention 4(MHA & MLA)
```bash
# FA4 for both prefill and decode on SM90/SM100
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --attention-backend fa4 \
  --page-size 128 \
  --trust-remote-code

python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --prefill-attention-backend fa4 \
  --trust-remote-code
```

- Cutlass MLA
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend cutlass_mla \
  --trust-remote-code
```

- Ascend
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend ascend
```

- Intel XPU
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend intel_xpu
```

- Wave
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend wave
```

- FlexAttention
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend flex_attention
```

- Dual Chunk FlashAttention
```bash
python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-14B-Instruct-1M \
  --attention-backend dual_chunk_flash_attn
```

- Torch Native
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend torch_native
```

## 添加新注意力后端的步骤
要添加新的注意力后端,你可以参考现有的后端
(`python/sglang/srt/layers/attention/triton_backend.py`、`python/sglang/srt/layers/attention/flashattention_backend.py`)
并遵循以下步骤。

```{note}
线性注意力 kernel 后端(GDN、KDA)遵循不同的模式。它们在 `python/sglang/srt/layers/attention/linear/kernels/` 中实现 `LinearAttnKernelBase`,并由 `GDNKernelDispatcher` / `KDAKernelDispatcher` 分派,而不是通过 `@register_attention_backend` 注册。
```

1. 不使用 cuda graph 运行。支持两个 forward 函数
- forward_extend
  - 将用于 prefill、带 KV cache 的 prefill 以及目标验证
  - 每层会被调用一次
- forward_decode
  - 将用于普通 decode 和草稿 decode
  - 每层会被调用一次
- init_forward_metadata
  - 初始化类以及所有层共享的公共元数据
  - 调用 plan 函数进行诸如 split_kv 之类的优化
  - 每次 forward 会被调用一次
2. 使用 cuda graph 运行。它有两个阶段(capture 和 replay),你需要实现三个函数
- init_cuda_graph_state
  - 在整个生命周期内会被调用一次
  - 创建所有公共共享缓冲区
- init_forward_metadata_capture_cuda_graph
  - 在捕获 cuda graph 之前会被调用
  - 它与 init_forward_metadata 类似,但将元数据写入一些预定义的缓冲区
- init_forward_metadata_replay_cuda_graph
  - 在重放 cuda graph 之前会被调用
  - 此函数处于关键路径上,需要执行得快
