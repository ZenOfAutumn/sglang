# DeepSeek V3/V3.1/R1 使用指南

SGLang 提供了许多专为 DeepSeek 模型设计的优化，使其从第一天起就成为官方 [DeepSeek 团队](https://github.com/deepseek-ai/DeepSeek-V3/tree/main?tab=readme-ov-file#62-inference-with-sglang-recommended)推荐的推理引擎。

本文档概述了当前针对 DeepSeek 的优化。
有关已实现特性的概览，请参阅已完成的 [Roadmap](https://github.com/sgl-project/sglang/issues/2591)。

## 使用 SGLang 启动 DeepSeek V3.1/V3/R1

要运行 DeepSeek V3.1/V3/R1 模型，推荐的设置如下：

| Weight Type | Configuration |
|------------|-------------------|
| **全精度 [FP8](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)**<br>*（推荐）* | 8 x H200 |
| | 8 x B200 |
| | 8 x MI300X |
| | 2 x 8 x H100/800/20 |
| | Xeon 6980P CPU |
| **全精度 ([BF16](https://huggingface.co/unsloth/DeepSeek-R1-0528-BF16))**（由原始 FP8 上转换而来） | 2 x 8 x H200 |
| | 2 x 8 x MI300X |
| | 4 x 8 x H100/800/20 |
| | 4 x 8 x A100/A800 |
| **量化权重 ([INT8](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8))** | 16 x A100/800 |
| | 32 x L40S |
| | Xeon 6980P CPU |
| | 4 x Atlas 800I A3 |
| **量化权重 ([W4A8](https://huggingface.co/novita/Deepseek-R1-0528-W4AFP8))** | 8 x H20/100, 4 x H200 |
| **量化权重 ([AWQ](https://huggingface.co/QuixiAI/DeepSeek-R1-0528-AWQ))** | 8 x H100/800/20 |
| | 8 x A100/A800 |
| **量化权重 ([MXFP4](https://huggingface.co/amd/DeepSeek-R1-MXFP4-Preview))** | 8, 4 x MI355X/350X |
| **量化权重 ([NVFP4](https://huggingface.co/nvidia/DeepSeek-R1-0528-NVFP4-v2))** | 8, 4 x B200 |

<style>
.md-typeset__table {
  width: 100%;
}

.md-typeset__table table {
  border-collapse: collapse;
  margin: 1em 0;
  border: 2px solid var(--md-typeset-table-color);
  table-layout: fixed;
}

.md-typeset__table th {
  border: 1px solid var(--md-typeset-table-color);
  border-bottom: 2px solid var(--md-typeset-table-color);
  background-color: var(--md-default-bg-color--lighter);
  padding: 12px;
}

.md-typeset__table td {
  border: 1px solid var(--md-typeset-table-color);
  padding: 12px;
}

.md-typeset__table tr:nth-child(2n) {
  background-color: var(--md-default-bg-color--lightest);
}
</style>

```{important}
官方的 DeepSeek V3 已经是 FP8 格式，因此你不应使用任何量化参数（如 `--quantization fp8`）来运行它。
```

供参考的详细命令：

- [8 x H200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#using-docker-recommended)
- [4 x B200, 8 x B200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-one-b200-node)
- [8 x MI300X](../platforms/amd_gpu.md#running-deepseek-v3)
- [2 x 8 x H200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h2008-nodes-and-docker)
- [4 x 8 x A100](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-four-a1008-nodes)
- [8 x A100 (AWQ)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-8-a100a800-with-awq-quantization)
- [16 x A100 (INT8)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-16-a100a800-with-int8-quantization)
- [32 x L40S (INT8)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-32-l40s-with-int8-quantization)
- [Xeon 6980P CPU](../platforms/cpu_server.md#example-running-deepseek-r1)
- [4 x Atlas 800I A3 (int8)](../platforms/ascend/ascend_npu_deepseek_example.md#running-deepseek-with-pd-disaggregation-on-4-x-atlas-800i-a3)

### 下载权重
如果你在启动服务器时遇到错误，请确保权重已完成下载。建议事先下载它们，或多次重启直到所有权重下载完毕。请参阅 [DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base#61-inference-with-deepseek-infer-demo-example-only) 官方指南来下载权重。

### 在一个 8 x H200 节点上启动
请参阅[该示例](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#installation--launch)。

### 在多节点上运行示例

- [Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP](https://lmsys.org/blog/2025-06-16-gb200-part-1/)（[Part I](https://lmsys.org/blog/2025-06-16-gb200-part-1/)、[Part II](https://lmsys.org/blog/2025-09-25-gb200-part-2/)）——关于 GB200 优化的全面指南。

- [Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs](https://lmsys.org/blog/2025-05-05-large-scale-ep/)——关于 PD disaggregation 和大规模 EP 的指南。

- [Serving with two H20*8 nodes](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h208-nodes)。

- [Best Practices for Serving DeepSeek-R1 on H20](https://lmsys.org/blog/2025-09-26-sglang-ant-group/)——关于 H20 优化、部署和性能的全面指南。

- [Serving with two H200*8 nodes and docker](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h2008-nodes-and-docker)。

- [Serving with four A100*8 nodes](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-four-a1008-nodes)。

## 优化

### Multi-head Latent Attention (MLA) 吞吐量优化

**说明**：[MLA](https://arxiv.org/pdf/2405.04434) 是 DeepSeek 团队提出的一种创新注意力机制，旨在提升推理效率。SGLang 为此实现了特定优化，包括：

- **权重吸收（Weight Absorption）**：通过应用矩阵乘法的结合律来重新排序计算步骤，此方法在解码阶段平衡了计算与访存并提升了效率。

- **MLA 注意力后端**：目前 SGLang 支持多种优化的 MLA 注意力后端，包括 [FlashAttention3](https://github.com/Dao-AILab/flash-attention)、[Flashinfer](https://docs.flashinfer.ai/api/attention.html#flashinfer-mla)、[FlashMLA](https://github.com/deepseek-ai/FlashMLA)、[CutlassMLA](https://github.com/sgl-project/sglang/pull/5390)、**TRTLLM MLA**（针对 Blackwell 架构优化）以及 [Triton](https://github.com/triton-lang/triton) 后端。默认的 FA3 在广泛的工作负载下都提供良好的性能。

- **FP8 量化**：W8A8 FP8 和 KV Cache FP8 量化实现了高效的 FP8 推理。此外，我们实现了批量矩阵乘法（BMM）算子，以便在带权重吸收的 MLA 中实现 FP8 推理。

- **CUDA Graph 与 Torch.compile**：MLA 和 Mixture of Experts (MoE) 都兼容 CUDA Graph 和 Torch.compile，这能降低延迟并在小批量时加速解码速度。

- **Chunked Prefix Cache**：分块前缀缓存优化通过将前缀缓存切分为若干块、用 multi-head attention 处理它们并合并其状态，从而提升吞吐量。当对长序列进行 chunked prefill 时，其提升可能很显著。目前此优化仅在 FlashAttention3 后端可用。

总体而言，借助这些优化，我们在输出吞吐量上相比前一版本实现了高达 **7x** 的加速。

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_3/deepseek_mla.svg" alt="Multi-head Latent Attention for DeepSeek Series Models">
</p>

**用法**：MLA 优化默认启用。

**参考**：更多详情请查看 [Blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations) 和 [Slides](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/lmsys_1st_meetup_deepseek_mla.pdf)。

### 数据并行注意力（Data Parallelism Attention）

**说明**：此优化涉及对 DeepSeek 系列模型 MLA 注意力机制的数据并行（DP），它能显著减小 KV cache 大小，从而支持更大的批量。每个 DP worker 独立处理不同类型的批次（prefill、decode、idle），随后在通过 Mixture-of-Experts (MoE) 层处理前后进行同步。如果你不使用 DP attention，KV cache 将在所有 TP rank 之间被复制。

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_4/dp_attention.svg" alt="Data Parallelism Attention for DeepSeek Series Models">
</p>

启用数据并行注意力后，我们相比前一版本实现了高达 **1.9x** 的解码吞吐量提升。

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_4/deepseek_coder_v2.svg" alt="Data Parallelism Attention Performance Comparison">
</p>

**用法**：
- 在使用 8 块 H200 GPU 时，将 `--enable-dp-attention --tp 8 --dp 8` 追加到服务器参数中。此优化能在服务器受 KV cache 容量限制的大批量场景下提升峰值吞吐量。
- DP 和 TP 注意力可以灵活组合。例如，要在 2 个各有 8 块 H100 GPU 的节点上部署 DeepSeek-V3/R1，你可以指定 `--enable-dp-attention --tp 16 --dp 2`。此配置以 2 个 DP 组运行注意力，每组包含 8 块 TP GPU。

```{caution}
数据并行注意力不推荐用于低延迟、小批量的场景。它是针对大批量的高吞吐量场景进行优化的。
```

**参考**：请查看 [Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models)。

### 多节点张量并行（Multi-Node Tensor Parallelism）

**说明**：对于单节点内存有限的用户，SGLang 支持使用张量并行跨多个节点部署 DeepSeek 系列模型（包括 DeepSeek V3）。此方法将模型参数划分到多块 GPU 或多个节点上，以处理对单节点内存而言过大的模型。

**用法**：使用示例请查看[此处](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h2008-nodes-and-docker)。

### 块级 FP8（Block-wise FP8）

**说明**：SGLang 实现了带两项关键优化的块级 FP8 量化：

- **激活（Activation）**：E4M3 格式，使用 per-token-per-128-channel 的子向量缩放（sub-vector scales）并进行在线转换（online casting）。

- **权重（Weight）**：Per-128x128-block 量化，以获得更好的数值稳定性。

- **DeepGEMM**：针对 FP8 矩阵乘法优化的 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) kernel 库。

**用法**：上述激活和权重优化对 DeepSeek V3 模型默认开启。DeepGEMM 在 NVIDIA Hopper/Blackwell GPU 上默认启用，在其他设备上默认禁用。也可以通过设置环境变量 `SGLANG_ENABLE_JIT_DEEPGEMM=0` 手动关闭 DeepGEMM。

```{tip}
在部署 DeepSeek 模型之前，请预编译 DeepGEMM kernel 以改善首次运行性能。预编译过程通常需要约 10 分钟完成。
```

```bash
python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
```

### 多 token 预测（Multi-token Prediction）
**说明**：SGLang 基于 [EAGLE 投机解码](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding) 实现了 DeepSeek V3 的多 token 预测（MTP）。借助此优化，在 H200 TP8 设置下，batch size 为 1 时解码速度可提升 **1.8x**，batch size 为 32 时可提升 **1.5x**。

**用法**：
添加 `--speculative-algorithm EAGLE`。其他标志，如 `--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 是可选的。例如：
```
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --speculative-algorithm EAGLE \
  --trust-remote-code \
  --tp 8
```
- DeepSeek 模型的默认配置为 `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`。针对给定的 batch size，`--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 的最佳配置可以使用 [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) 脚本搜索得到。最小配置是 `--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`，它能在较大的 batch size 下实现加速。
- 大多数 MLA 注意力后端完全支持 MTP 的使用。详情请参阅 [MLA Backends](../advanced_features/attention_backend.md#mla-backends)。

```{note}
要在大 batch size（>48）下启用 DeepSeek MTP，你需要调整一些参数（参考[这个讨论](https://github.com/sgl-project/sglang/issues/4543#issuecomment-2737413756)）：
- 将 `--max-running-requests` 调大。MTP 的默认值为 `48`。对于更大的 batch size，你应将此值增大到默认值以上。
- 设置 `--cuda-graph-bs`。它是用于 cuda graph 捕获的 batch size 列表。[投机解码默认捕获的 batch size](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L888-L895) 为 48。你可以通过加入更多的 batch size 来自定义它。
```

```{tip}
要为 EAGLE 投机解码启用实验性的 overlap scheduler，请设置环境变量 `SGLANG_ENABLE_SPEC_V2=1`。这可以通过在草稿（draft）和验证（verification）阶段之间启用 overlap scheduling 来提升性能。
```


### DeepSeek R1 与 V3.1 的推理内容（Reasoning Content）

请参阅 [Reasoning Parser](https://docs.sglang.io/advanced_features/separate_reasoning.html) 和 [Thinking Parameter for DeepSeek V3.1](https://docs.sglang.io/basic_usage/openai_api_completions.html#Example:-DeepSeek-V3-Models)。


### DeepSeek 模型的函数调用（Function calling）

添加参数 `--tool-call-parser deepseekv3` 和 `--chat-template ./examples/chat_template/tool_chat_template_deepseekv3.jinja`（推荐）来启用此特性。例如（在 1 * H20 节点上运行）：

```
python3 -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3-0324 \
  --tp 8 \
  --port 30000 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.9 \
  --tool-call-parser deepseekv3 \
  --chat-template ./examples/chat_template/tool_chat_template_deepseekv3.jinja
```

请求示例：

```
curl "http://127.0.0.1:30000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{"temperature": 0, "max_tokens": 100, "model": "deepseek-ai/DeepSeek-V3-0324", "tools": [{"type": "function", "function": {"name": "query_weather", "description": "Get weather of a city, the user should supply a city first", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city, e.g. Beijing"}}, "required": ["city"]}}}], "messages": [{"role": "user", "content": "How'\''s the weather like in Qingdao today"}]}'
```

预期响应

```
{"id":"6501ef8e2d874006bf555bc80cddc7c5","object":"chat.completion","created":1745993638,"model":"deepseek-ai/DeepSeek-V3-0324","choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[{"id":"0","index":null,"type":"function","function":{"name":"query_weather","arguments":"{\"city\": \"Qingdao\"}"}}]},"logprobs":null,"finish_reason":"tool_calls","matched_stop":null}],"usage":{"prompt_tokens":116,"total_tokens":138,"completion_tokens":22,"prompt_tokens_details":null}}

```
流式请求示例：
```
curl "http://127.0.0.1:30000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{"temperature": 0, "max_tokens": 100, "model": "deepseek-ai/DeepSeek-V3-0324","stream":true,"tools": [{"type": "function", "function": {"name": "query_weather", "description": "Get weather of a city, the user should supply a city first", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city, e.g. Beijing"}}, "required": ["city"]}}}], "messages": [{"role": "user", "content": "How'\''s the weather like in Qingdao today"}]}'
```
预期的流式数据块（为清晰起见已简化）：
```
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"city"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\":\""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"Q"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"ing"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"dao"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\"}"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":null}}], "finish_reason": "tool_calls"}
data: [DONE]
```
客户端需要拼接所有 arguments 片段以重建完整的工具调用：
```
{"city": "Qingdao"}
```

```{important}
1. 使用较低的 `"temperature"` 值以获得更好的结果。
2. 为获得更一致的工具调用结果，建议使用 `--chat-template examples/chat_template/tool_chat_template_deepseekv3.jinja`。它提供了改进的统一 prompt。
```


### DeepSeek R1 的思考预算（Thinking Budget）

在 SGLang 中，我们可以使用 `CustomLogitProcessor` 来实现思考预算。

启动服务器时开启 `--enable-custom-logit-processor` 标志。

```
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --tp 8 --port 30000 --host 0.0.0.0 --mem-fraction-static 0.9 --disable-cuda-graph --reasoning-parser deepseek-r1 --enable-custom-logit-processor
```

请求示例：

```python
import openai
from rich.pretty import pprint
from sglang.srt.sampling.custom_logit_processor import DeepSeekR1ThinkingBudgetLogitProcessor


client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="*")
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {
            "role": "user",
            "content": "Question: Is Paris the Capital of France?",
        }
    ],
    max_tokens=1024,
    extra_body={
        "custom_logit_processor": DeepSeekR1ThinkingBudgetLogitProcessor().to_str(),
        "custom_params": {
            "thinking_budget": 512,
        },
    },
)
pprint(response)
```

## 常见问题（FAQ）

**问：模型加载耗时过长，并且我遇到了 NCCL 超时。我该怎么办？**

答：如果你遇到模型加载时间过长和 NCCL 超时，可以尝试增大超时时长。在启动模型时添加参数 `--dist-timeout 3600`。这会将超时设置为一小时，通常能解决该问题。
