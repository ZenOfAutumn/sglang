# Speculative Decoding

SGLang 提供了多种 speculative decoding 选项,包括 EAGLE-2/EAGLE-3、MTP、经典的 draft-model 解码以及一种基于 NGRAM 的变体。我们的实现旨在最大化速度和效率,被认为是开源 LLM 引擎中最快的之一。

## Summary

### Jump to sections

- [EAGLE Decoding](#eagle-decoding)
  - [EAGLE-2 Decoding](#eagle-2-decoding)
  - [EAGLE-2 Decoding with torch.compile](#eagle-2-decoding-with-torchcompile)
  - [EAGLE-2 Decoding via Frequency-Ranked Speculative Sampling](#eagle-2-decoding-via-frequency-ranked-speculative-sampling)
  - [EAGLE-3 Decoding](#eagle-3-decoding)
- [Multi Token Prediction](#multi-token-prediction)
- [Standalone Speculative Decoding (Small Draft Model)](#standalone-speculative-decoding-small-draft-model)
- [Speculative Decoding V2 (Overlap Scheduler)](#speculative-decoding-v2-overlap-scheduler)
- [Ngram Speculative Decoding](#ngram-speculative-decoding)
- [Full Parameter Reference](#full-parameter-reference)
- [OOM Troubleshooting](#oom-troubleshooting)
- [References](#references)

### Quick guidance

- **最佳速度/质量(推荐)**:使用 **EAGLE-3**,配合 `--speculative-algorithm EAGLE3`。
- **强力默认 / 广泛兼容**:使用 **EAGLE-2**,配合 `--speculative-algorithm EAGLE`。
- **降低 EAGLE-2 的 `lm_head` 开销**:使用 `--speculative-token-map` 启用 **FR-Spec**。
- **模型支持 MTP**:使用 **通过 speculative decoding 的 MTP**(通常使用较小的 `speculative_num_steps/topk/num_draft_tokens`,参见示例章节)。
- **你有一个较小的 draft LLM**:使用 **STANDALONE**(`--speculative-algorithm STANDALONE`)。
- **没有额外的模型可用**:使用 **NGRAM**(`--speculative-algorithm NGRAM`,仅 CUDA)。
- **想要 overlap scheduler(实验性)**:使用 `SGLANG_ENABLE_SPEC_V2=True` 启用 **SpecV2**(需要 `--speculative-eagle-topk 1`)。

### Method comparison (mini table)

| Method | Draft source | Separate draft model? | How to enable | Notes / constraints |
|---|---|---:|---|---|
| EAGLE-2 | EAGLE draft model(feature drafting + tree) | 通常是 | `--speculative-algorithm EAGLE` + `--speculative-draft-model-path ...` | 调优 `--speculative-num-steps`、`--speculative-eagle-topk`、`--speculative-num-draft-tokens` |
| EAGLE-2 + `torch.compile` | 与 EAGLE-2 相同 | 通常是 | 添加 `--enable-torch-compile`(可选 `--torch-compile-max-bs`) | 收益因硬件/模型而异;请 benchmark 验证 |
| EAGLE-2 + FR-Spec | 与 EAGLE-2 相同 + token 子集 | 通常是 | 添加 `--speculative-token-map ...` | 通过高频 token 词表降低 `lm_head` 开销 |
| EAGLE-3 | EAGLE3 draft model | 是 | `--speculative-algorithm EAGLE3` + `--speculative-draft-model-path ...` | 在下方 benchmark 中 throughput 最佳 |
| MTP | 内置 multi-token heads(模型特定) | 通常否 | 参见 **Multi Token Prediction** 章节 | 使用 speculative 工作流;对某些模型 draft 路径可能自动处理 |
| STANDALONE | 较小的 draft LLM(token 级) | 是 | `--speculative-algorithm STANDALONE` + `--speculative-draft-model-path ...` | **不支持** `--enable-dp-attention` |
| SpecV2(实验性) | V2 worker + overlap scheduler | N/A | `SGLANG_ENABLE_SPEC_V2=True` | 仅支持 `--speculative-eagle-topk 1`;适用于 `EAGLE`、`EAGLE3`、`STANDALONE` |
| NGRAM | 来自先前 token 的 ngram cache | 否 | `--speculative-algorithm NGRAM` | 仅 CUDA;不支持 `--enable-dp-attention`;禁用 overlap scheduler 和 mixed chunked prefill |

### Performance Highlights

请参见下方在 MT bench 上测试的 LLaMA-Instruct 3.1 8B 通过 EAGLE3 解码所能实现的巨大 throughput 改进。
更多细节请参见 [EAGLE3 paper](https://arxiv.org/pdf/2503.01840)。

| Method | Throughput (tokens/s) |
|--------|----------------|
| SGLang (w/o speculative, 1x H100) | 158.34 tokens/s |
| SGLang + EAGLE-2 (1x H100) | 244.10 tokens/s |
| SGLang + EAGLE-3 (1x H100) | 373.25 tokens/s |

---

## EAGLE Decoding

要启用 EAGLE speculative decoding,以下参数相关:

| Parameter | Description | Default |
|---|---|---|
| `--speculative-draft-model-path` | Draft model 路径/权重。对于 EAGLE/EAGLE3 和 STANDALONE **通常必需**。对于某些支持 MTP 的模型,可以省略。 | `None` |
| `--speculative-num-steps` | 自回归 drafting 的深度。增加 speculation 范围,但有 rejection cascade 的风险。 | Auto(Llama/Grok 为 `5`;许多其他模型为 `3`) |
| `--speculative-eagle-topk` | 每步的分支因子。提升候选多样性和接受率,但增加内存/计算消耗。 | Auto(Llama/Grok 为 `4`;许多其他模型为 `1`) |
| `--speculative-num-draft-tokens` | 最大并行验证容量。允许更深的树评估,但增加 GPU 内存使用。 | Auto(Llama/Grok 为 `8`;许多其他模型为 `4`)。如果 `topk=1`,则调整为 `num_steps + 1`。 |
| `--speculative-accept-threshold-single` | 单 token 验证的接受阈值。值越低接受越激进。 | `1.0` |
| `--speculative-accept-threshold-acc` | 跨步骤的累积接受阈值。 | `1.0` |
| `--speculative-attention-mode` | speculative 操作的 attention 模式(`prefill` 或 `decode`),影响 target 验证和 draft 扩展两者。 | `"prefill"` |
| `--speculative-draft-attention-backend` | 覆盖 draft model 的 attention 后端。 | `None`(与 target 相同) |
| `--speculative-draft-model-quantization` | draft model 的 quantization 方法。使用 `"unquant"` 即使在 target model 已量化时也强制不进行量化。 | 与 target model 相同 |
| `--speculative-draft-model-revision` | 要加载的 draft model 的特定 revision/commit。 | `None`(当设置了 `--speculative-draft-model-path` 且省略 revision 时,自动设置为 `"main"`) |
| `--speculative-draft-load-format` | draft model 权重的加载格式。 | `None` |

对于 EAGLE-2 和 EAGLE-3,这些参数大部分相同。`--speculative-token-map` 对 EAGLE-3 模型被忽略。
对于 `--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens`:三者全部不设置以使用自动调优,或在调优时三者全部显式设置。

你可以使用 [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) 找到这些参数的最佳组合。


### EAGLE-2 Decoding

你可以通过设置 `--speculative-algorithm EAGLE` 并选择合适的模型来启用 EAGLE-2 Decoding。

**Launch the server:**

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**Send a request:**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

### EAGLE-2 Decoding with `torch.compile`

你可以选择启用 `torch.compile`,以对 draft model 应用 kernel 级优化(算子融合、autotune)。实际加速取决于你的硬件、模型架构和 batch size。在某些配置下(例如 H100 上的小 draft model,此时 cuBLAS 已经最优且启用了 CUDA graph),收益可能可以忽略。我们建议在你的具体设置上分别使用和不使用此 flag 进行 benchmark,以验证它是否有帮助。

要启用它,添加 `--enable-torch-compile`,并可选地设置 `--torch-compile-max-bs`:

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.7 \
    --enable-torch-compile \
    --torch-compile-max-bs 8 \
    --log-level warning
```

**Send a request:**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

### EAGLE-2 Decoding via Frequency-Ranked Speculative Sampling

通过在 draft model 中采用截断的高频 token 词表,EAGLE speculative decoding 在加速流水线的同时减少 `lm_head` 计算开销,且不会带来质量下降。更多细节请查看[论文](https://arxiv.org/pdf/2502.14856)。

在我们的实现中,设置 `--speculative-token-map` 以启用此优化。你可以从[这个模型](https://huggingface.co/thunlp/LLaMA3-Instruct-8B-FR-Spec)获取 FR-Spec 中的高频 token。或者你可以通过从[这个仓库](https://github.com/thunlp/FR-Spec/tree/main?tab=readme-ov-file#prepare-fr-spec-vocabulary-subset)直接下载这些 token 来获取高频 token。

感谢 [Weilin Zhao](https://github.com/Achazwl) 和 [Zhousx](https://github.com/Zhou-sx) 的贡献。

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-LLaMA3-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --speculative-token-map thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --dtype float16 \
    --log-level warning
```

**Send a request:**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

### EAGLE-3 Decoding

你可以通过设置 `--speculative-algorithm EAGLE3` 并选择合适的模型来启用 EAGLE-3 解码。

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --dtype float16 \
    --log-level warning
```

**Send a request:**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## Multi Token Prediction

我们通过使用 speculative decoding 在 SGLang 中支持 [MTP (Multi-Token Prediction)](https://arxiv.org/pdf/2404.19737)。这里我们以 `XiaomiMiMo/MiMo-7B-RL` 为例(关于 DeepSeek MTP 的用法,请参考 [deepseek_v32 doc](../basic_usage/deepseek_v32.md#multi-token-prediction))。

```bash
python3 -m sglang.launch_server \
    --model XiaomiMiMo/MiMo-7B-RL \
    --host 0.0.0.0 \
    --trust-remote-code \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 2 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**Send a request:**

```python
import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "XiaomiMiMo/MiMo-7B-RL",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

response = requests.post(url, json=data)
print(response.json())
```

---

## Standalone Speculative Decoding (Small Draft Model)

除了 EAGLE/MTP,SGLang 还支持使用较小的 **draft model** 进行 **token 级 speculative decoding**。通过 `--speculative-algorithm STANDALONE` 启用它,并通过 `--speculative-draft-model-path` 提供一个 draft model。

相关参数:

| Parameter | Description | Default |
|---|---|---|
| `--speculative-draft-model-path` | Draft model 权重(比 target model 小)。 | `None` |
| `--speculative-num-steps` | Draft 深度(draft model 自回归运行多少步)。 | `3`(STANDALONE 的自动默认值) |
| `--speculative-eagle-topk` | 分支因子(每步的 token 候选数)。 | `1`(STANDALONE 的自动默认值) |
| `--speculative-num-draft-tokens` | 验证容量。 | `4`(STANDALONE 的自动默认值) |
| `--speculative-draft-model-quantization` | draft model 的 quantization。使用 `"unquant"` 即使在 target 已量化时也禁用 draft 上的量化。 | 与 target 相同 |

> **注意:** Standalone speculative decoding 目前 **不支持** `--enable-dp-attention`。

```bash
python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 2 \
    --speculative-num-draft-tokens 7 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**Send a request:**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## Speculative Decoding V2 (Overlap Scheduler)

SGLang 提供了一个**实验性的 Speculative Decoding V2** 实现,它启用 overlap scheduler 并使用 V2 speculative worker(例如 `StandaloneWorkerV2`、`EAGLEWorkerV2`)。

要启用它,设置环境变量:
- `SGLANG_ENABLE_SPEC_V2=True`

注意:
- SpecV2 目前仅支持 `--speculative-eagle-topk 1`。启用 SpecV2 时,**显式设置 `--speculative-eagle-topk 1`**。
- 如果你显式设置 `--speculative-eagle-topk > 1`,服务器将报错。
- 如果你省略 `--speculative-eagle-topk`,自动调优可能为某些模型(例如 Llama)选择 `topk > 1`。这与 SpecV2 不兼容,且不一定总会立即触发配置错误,所以请显式设置 `--speculative-eagle-topk 1`。
- 这适用于 `EAGLE`、`EAGLE3` 和 `STANDALONE`。

```bash
SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 5 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**Send a request:**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## Ngram Speculative Decoding

SGLang 还支持**基于 ngram 的 speculative decoding**(无需单独的 draft model)。它从由先前生成的 token 构建的 ngram cache 中检索 draft token,然后用 target model 验证它们。

通过以下方式启用:
- `--speculative-algorithm NGRAM`

### Ngram-specific parameters

| Parameter | Description | Default |
|---|---|---|
| `--speculative-num-draft-tokens` | 每步验证的 draft token 数量。 | `12` |
| `--speculative-ngram-min-bfs-breadth` | 最小 BFS 宽度。 | `1` |
| `--speculative-ngram-max-bfs-breadth` | 最大 BFS 宽度。 | `10` |
| `--speculative-ngram-match-type` | Ngram 树构建模式:`"BFS"` 表示基于新近度(recency)的扩展,`"PROB"` 表示基于频率的扩展。 | `"BFS"` |
| `--speculative-ngram-max-trie-depth` | ngram trie 存储和匹配的最大后缀长度。 | `18` |
| `--speculative-ngram-capacity` | 缓存容量(条目数量)。 | `10,000,000` |

注意:
- Ngram speculative decoding **仅支持 CUDA**。
- 它目前 **不支持** `--enable-dp-attention`。
- 它会禁用 overlap scheduler 和 mixed chunked prefill。
- 如果 `--speculative-ngram-max-bfs-breadth > 1`(因此 `speculative_eagle_topk > 1`)且 `page_size > 1`,使用 `--attention-backend flashinfer`;否则服务器将报错。
- 可选:设置 `SGLANG_NGRAM_FORCE_GREEDY_VERIFY=True` 以强制 greedy 验证。

```bash
python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm NGRAM \
    --speculative-num-draft-tokens 16 \
    --speculative-ngram-max-bfs-breadth 10 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**Send a request:**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## Full Parameter Reference

下面是 SGLang 中所有可用的 speculative decoding 参数的完整列表:

### Core parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--speculative-algorithm` | `str` | `None` | 要使用的算法:`EAGLE`、`EAGLE3`、`STANDALONE`、`NGRAM`、`NEXTN`(`EAGLE` 的别名) |
| `--speculative-draft-model-path` | `str` | `None` | draft model 权重的路径 |
| `--speculative-draft-model-revision` | `str` | `None` | draft model 的特定 revision/commit(当设置了 draft 路径且省略 revision 时自动使用 `"main"`) |
| `--speculative-draft-load-format` | `str` | `None` | draft model 权重的加载格式 |
| `--speculative-num-steps` | `int` | `None`(省略时自动选择) | 自回归 drafting 深度 |
| `--speculative-eagle-topk` | `int` | `None`(省略时自动选择) | 每个 drafting 步骤的分支因子 |
| `--speculative-num-draft-tokens` | `int` | `None`(省略时自动选择) | 用于验证的最大 draft token 数量 |
| `--speculative-accept-threshold-single` | `float` | `1.0` | 单 token 接受阈值 |
| `--speculative-accept-threshold-acc` | `float` | `1.0` | 累积接受阈值 |
| `--speculative-token-map` | `str` | `None` | FR-Spec 高频 token map 的路径 |
| `--speculative-attention-mode` | `str` | `"prefill"` | speculative 操作的 attention 模式(`"prefill"` 或 `"decode"`) |
| `--speculative-draft-attention-backend` | `str` | `None` | 覆盖 draft model 的 attention 后端 |
| `--speculative-moe-runner-backend` | `str` | `None` | draft model 的 MoE runner 后端 |
| `--speculative-moe-a2a-backend` | `str` | `None` | draft model 的 MoE all-to-all 后端 |
| `--speculative-draft-model-quantization` | `str` | 与 target 相同 | draft model 的 quantization(`"unquant"` 表示禁用) |

### Ngram-specific parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--speculative-ngram-min-bfs-breadth` | `int` | `1` | 最小 BFS 宽度 |
| `--speculative-ngram-max-bfs-breadth` | `int` | `10` | 最大 BFS 宽度 |
| `--speculative-ngram-match-type` | `str` | `"BFS"` | Ngram 树构建模式:`"BFS"` 表示基于新近度的扩展,`"PROB"` 表示基于频率的扩展 |
| `--speculative-ngram-max-trie-depth` | `int` | `18` | ngram trie 存储和匹配的最大后缀长度 |
| `--speculative-ngram-capacity` | `int` | `10,000,000` | 缓存容量 |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `SGLANG_ENABLE_SPEC_V2` | `False` | 启用 Speculative Decoding V2(overlap scheduler) |
| `SGLANG_NGRAM_FORCE_GREEDY_VERIFY` | `False` | 为 ngram 解码强制 greedy 验证 |

### Other related flags

| Parameter | Description |
|---|---|
| `--enable-multi-layer-eagle` | 启用 multi-layer EAGLE(对 MiMoV2 和 Step3p5 模型自动启用) |
| `--enable-torch-compile` | 为 kernel 级优化启用 `torch.compile` |
| `--torch-compile-max-bs` | `torch.compile` 的最大 batch size |

---

## OOM Troubleshooting

> [!WARNING]
> **Out of Memory (OOM)?** Speculative decoding 可能增加 GPU 内存使用,因为 draft tree、CUDA graph 和验证相关的缓冲区会消耗额外的 VRAM。如果你遇到 OOM 错误,尝试以下调整。

### Step 1: Lower static memory fraction (most effective)

```bash
--mem-fraction-static 0.5   # when omitted, this value is auto-computed
```

- `--mem-fraction-static` 控制模型权重 + KV cache 池的内存预算。
- 降低它会直接增加用于激活和 CUDA graph 缓冲区的动态余量。
- 如果省略,SGLang 会根据其他设置自动估算此值,而那些自动设置对某些工作负载可能仍然过于激进。

### Step 2: Reduce CUDA graph batch size

```bash
# Fewer CUDA graph captures = less memory reserved
--cuda-graph-max-bs 4   # or even 2 for tight memory situations
```

- 如果省略,`--cuda-graph-max-bs` 会根据 GPU 内存和 TP 大小自动选择,在高内存 GPU 上可能大得多。

### Step 3: Reduce draft tree size

这三个参数直接控制 draft tree 消耗多少内存:

```bash
# Before (aggressive, high memory)
--speculative-num-steps 5 --speculative-eagle-topk 8 --speculative-num-draft-tokens 64

# After (conservative, lower memory)
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

### Step 4: Limit concurrent requests

```bash
# Fewer concurrent requests lowers in-flight load and can reduce OOM risk
--max-running-requests 4
```

### Quick OOM recovery recipe

如果你正在遇到 OOM,只想要一个能用的配置,从这个最小配置开始,然后逐步增加:

```bash
python3 -m sglang.launch_server \
    --model <your-model> \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path <your-draft-model> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --cuda-graph-max-bs 2 \
    --mem-fraction-static 0.5 \
    --max-running-requests 4 \
    --log-level warning
```

然后逐步增加 `--speculative-num-draft-tokens`、`--speculative-eagle-topk` 和 `--cuda-graph-max-bs`。最后再增加 `--mem-fraction-static`,且仅在运行稳定之后。

---

## References

EAGLE 过程如下:

- 在 EAGLE 中,draft model 使用特征序列 $(f_1, ..., f_k)$ 和 token 序列 $(t_2, ..., t_{k+1})$ 来预测下一个特征向量,即原始 LLM 的最后一个 hidden state。
- 然后从 $p_{k+2}=\text{LMHead}(f_{k+1})$ 中采样下一个 token。之后,这两个序列以树状方式扩展——分支出多个潜在的延续,每步的分支因子由 `speculative_eagle_topk` 参数控制——以确保上下文更连贯的连接,并再次作为输入给出。
- 在 SGLang 的 EAGLE-2 实现中,draft tree 按配置的步数扩展,然后重新排序以选择 top `speculative_num_draft_tokens` 个最终节点作为 draft token。
- EAGLE-3 移除了特征预测目标,纳入了低层和中层特征,并以 on-policy 方式训练。

这通过对特征(而非 token)进行操作以获得更规整的输入,并额外传入下一时间步的 token 以减少采样随机性,从而提升 drafting 准确性。更多细节请参见 [EAGLE-2](https://arxiv.org/abs/2406.16858) 和 [EAGLE-3](https://arxiv.org/abs/2503.01840) 论文。

关于如何训练你自己的 EAGLE 模型的指南,请参见 [EAGLE repo](https://github.com/SafeAILab/EAGLE/tree/main?tab=readme-ov-file#train)。专门针对 EAGLE-3 训练,请查看 [SpecForge](https://github.com/sgl-project/SpecForge),这是 SGLang 团队为 EAGLE-3 speculative decoding 模型设计的训练框架,可无缝移植到 SGLang 服务。详情请参见 [SpecForge documentation](https://docs.sglang.ai/SpecForge/) 和 [blog post](https://lmsys.org/blog/2025-07-25-spec-forge)。
