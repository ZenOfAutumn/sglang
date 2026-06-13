# 采样参数（Sampling Parameters）

本文档描述 SGLang Runtime 的采样参数。它是 runtime 的底层端点。
如果你想要一个能自动处理 chat template 的高层端点，可以考虑使用 [OpenAI Compatible API](openai_api_completions.ipynb)。

## `/generate` 端点

`/generate` 端点接受以下 JSON 格式的参数。详细用法请参阅 [native API doc](native_api.ipynb)。该对象定义在 `io_struct.py::GenerateReqInput` 中。你也可以阅读源代码以查找更多参数和文档。

| Argument                   | Type/Default                                                                 | Description                                                                                                                                                     |
|----------------------------|------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| text                       | `Optional[Union[List[str], str]] = None`                                     | 输入 prompt。可以是单个 prompt 或一批 prompt。                                                                                                 |
| input_ids                  | `Optional[Union[List[List[int]], List[int]]] = None`                         | 文本对应的 token ID；可以指定 text 或 input_ids 之一。                                                                                               |
| input_embeds               | `Optional[Union[List[List[List[float]]], List[List[float]]]] = None`         | input_ids 对应的 embeddings；可以指定 text、input_ids 或 input_embeds 之一。                                                                          |
| image_data                 | `Optional[Union[List[List[ImageDataItem]], List[ImageDataItem], ImageDataItem]] = None` | 图像输入。支持三种格式：(1) **原始图像**：PIL Image、文件路径、URL 或 base64 字符串；(2) **处理器输出**：包含 HuggingFace 处理器输出、带 `format: "processor_output"` 的 Dict；(3) **预计算 embeddings**：带 `format: "precomputed_embedding"` 且 `feature` 包含预先计算好的视觉 embeddings 的 Dict。可以是单张图像、图像列表或图像列表的列表。详情见 [Multimodal Input Formats](#multimodal-input-formats)。 |
| audio_data                 | `Optional[Union[List[AudioDataItem], AudioDataItem]] = None`                 | 音频输入。可以是文件名、URL 或 base64 编码的字符串。                                                                                             |
| sampling_params            | `Optional[Union[List[Dict], Dict]] = None`                                   | 采样参数，如下文各节所述。                                                                                                     |
| rid                        | `Optional[Union[List[str], str]] = None`                                     | 请求 ID。                                                                                                                                                 |
| return_logprob             | `Optional[Union[List[bool], bool]] = None`                                   | 是否返回 token 的对数概率（log probabilities）。                                                                                                                 |
| logprob_start_len          | `Optional[Union[List[int], int]] = None`                                     | 如果 return_logprob，则为 prompt 中返回 logprobs 的起始位置。默认为 "-1"，即仅返回输出 token 的 logprobs。                     |
| top_logprobs_num           | `Optional[Union[List[int], int]] = None`                                     | 如果 return_logprob，则为每个位置返回的 top logprobs 数量。                                                                                       |
| token_ids_logprob          | `Optional[Union[List[List[int]], List[int]]] = None`                         | 如果 return_logprob，则为要返回 logprob 的 token ID。                                                                                         |
| return_text_in_logprobs    | `bool = False`                                                               | 是否在返回的 logprobs 中将 token 反 tokenize 为文本。                                                                                                  |
| stream                     | `bool = False`                                                               | 是否流式输出。                                                                                                                                       |
| lora_path                  | `Optional[Union[List[Optional[str]], Optional[str]]] = None`                 | LoRA 的路径。                                                                                                                                           |
| custom_logit_processor     | `Optional[Union[List[Optional[str]], str]] = None`                           | 用于高级采样控制的自定义 logit 处理器。必须是使用 `to_str()` 方法序列化的 `CustomLogitProcessor` 实例。用法见下文。 |
| return_hidden_states       | `Union[List[bool], bool] = False`                                            | 是否返回 hidden states。                                                                                                                                |
| return_routed_experts      | `bool = False`                                                               | 是否为 MoE 模型返回路由专家（routed experts）。需要 `--enable-return-routed-experts` 服务器标志。返回 base64 编码的 int32 专家 ID，作为逻辑形状为 `[num_tokens, num_layers, top_k]` 的扁平化数组。 |

## 采样参数

该对象定义在 `sampling_params.py::SamplingParams` 中。你也可以阅读源代码以查找更多参数和文档。

### 关于默认值的说明

默认情况下，SGLang 会从模型的 `generation_config.json` 初始化若干采样参数（当服务器以 `--sampling-defaults model`（即默认值）启动时）。若要改用 SGLang/OpenAI 的常量默认值，请以 `--sampling-defaults openai` 启动服务器。你总是可以通过 `sampling_params` 在每个请求中覆盖任意参数。

```bash
# 使用 generation_config.json 提供的模型默认值（默认行为）
python -m sglang.launch_server --model-path <MODEL> --sampling-defaults model

# 改用 SGLang/OpenAI 的常量默认值
python -m sglang.launch_server --model-path <MODEL> --sampling-defaults openai
```

### 核心参数

| Argument        | Type/Default                                 | Description                                                                                                                                    |
|-----------------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| max_new_tokens  | `int = 128`                                  | 以 token 衡量的最大输出长度。                                                                                                  |
| stop            | `Optional[Union[str, List[str]]] = None`     | 一个或多个[停止词（stop words）](https://platform.openai.com/docs/api-reference/chat/create#chat-create-stop)。如果采样到其中一个词，生成将停止。 |
| stop_token_ids  | `Optional[List[int]] = None`                 | 以 token ID 形式提供停止词。如果采样到其中一个 token ID，生成将停止。                                        |
| stop_regex      | `Optional[Union[str, List[str]]] = None`     | 当命中此列表中的任意正则表达式模式时停止 |
| temperature     | `float (model default; fallback 1.0)`        | 采样下一个 token 时的[温度（Temperature）](https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature)。`temperature = 0` 对应贪婪采样，温度越高多样性越大。 |
| top_p           | `float (model default; fallback 1.0)`        | [Top-p](https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_p) 从累积概率超过 `top_p` 的最小有序集合中选取 token。当 `top_p = 1` 时，这退化为从所有 token 中无限制地采样。 |
| top_k           | `int (model default; fallback -1)`           | [Top-k](https://developer.nvidia.com/blog/how-to-get-better-outputs-from-your-large-language-model/#predictability_vs_creativity) 从概率最高的 `k` 个 token 中随机选取。 |
| min_p           | `float (model default; fallback 0.0)`        | [Min-p](https://github.com/huggingface/transformers/issues/27670) 从概率大于 `min_p * highest_token_probability` 的 token 中采样。 |

### 惩罚项（Penalizers）

| Argument           | Type/Default           | Description                                                                                                                                    |
|--------------------|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| frequency_penalty  | `float = 0.0`          | 基于 token 在迄今生成中的频率进行惩罚。必须在 `-2` 和 `2` 之间，其中负数鼓励 token 重复，正数鼓励采样新 token。惩罚的缩放随 token 每次出现而线性增长。 |
| presence_penalty   | `float = 0.0`          | 如果 token 在迄今生成中出现过，则对其进行惩罚。必须在 `-2` 和 `2` 之间，其中负数鼓励 token 重复，正数鼓励采样新 token。一旦 token 出现，惩罚的缩放是恒定的。 |
| repetition_penalty | `float = 1.0`          | 缩放先前生成 token 的 logits，以抑制（值 > 1）或鼓励（值 < 1）重复。有效范围是 `[0, 2]`；`1.0` 不改变概率。 |
| min_new_tokens     | `int = 0`              | 强制模型至少生成 `min_new_tokens` 个 token，直到采样到停止词或 EOS token。注意这可能导致意外行为，例如当分布高度偏向这些 token 时。 |

### 约束解码（Constrained decoding）

以下参数请参阅我们专门的[约束解码](../advanced_features/structured_outputs.ipynb)指南。

| Argument        | Type/Default                    | Description                                                                                                                                    |
|-----------------|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| json_schema     | `Optional[str] = None`          | 用于结构化输出的 JSON schema。                                                                                                            |
| regex           | `Optional[str] = None`          | 用于结构化输出的正则表达式。                                                                                                                  |
| ebnf            | `Optional[str] = None`          | 用于结构化输出的 EBNF。                                                                                                                   |
| structural_tag  | `Optional[str] = None`          | 用于结构化输出的 structural tag。                                                                                                       |

### 其他选项

| Argument                      | Type/Default                    | Description                                                                                                                                    |
|-------------------------------|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| n                             | `int = 1`                       | 指定每个请求生成的输出序列数量。（不鼓励在一个请求中生成多个输出（n > 1）；多次重复相同的 prompt 可以提供更好的控制和效率。） |
| ignore_eos                    | `bool = False`                  | 采样到 EOS token 时不停止生成。                                                                                               |
| skip_special_tokens           | `bool = True`                   | 解码期间移除特殊 token。                                                                                                         |
| spaces_between_special_tokens | `bool = True`                   | 反 tokenize 期间是否在特殊 token 之间添加空格。                                                                     |
| no_stop_trim                  | `bool = False`                  | 不从生成的文本中裁剪停止词或 EOS token。                                                                                    |
| custom_params                 | `Optional[List[Optional[Dict[str, Any]]]] = None` | 在使用 `CustomLogitProcessor` 时使用。用法见下文。                                                                              |

## 示例

### 普通（Normal）

启动服务器：

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000
```

发送请求：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)
print(response.json())
```

详细示例见 [send request](./send_request.ipynb)。

### 流式（Streaming）

发送请求并流式获取输出：

```python
import requests, json

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
        "stream": True,
    },
    stream=True,
)

prev = 0
for chunk in response.iter_lines(decode_unicode=False):
    chunk = chunk.decode("utf-8")
    if chunk and chunk.startswith("data:"):
        if chunk == "data: [DONE]":
            break
        data = json.loads(chunk[5:].strip("\n"))
        output = data["text"].strip()
        print(output[prev:], end="", flush=True)
        prev = len(output)
print("")
```

详细示例见 [openai compatible api](openai_api_completions.ipynb)。

### 多模态（Multimodal）

启动服务器：

```bash
python3 -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-7b-ov
```

下载一张图像：

```bash
curl -o example_image.png -L https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true
```

发送请求：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<image>\nDescribe this image in a very short sentence.<|im_end|>\n"
                "<|im_start|>assistant\n",
        "image_data": "example_image.png",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)
print(response.json())
```

`image_data` 可以是文件名、URL 或 base64 编码的字符串。另请参阅 `python/sglang/srt/utils.py:load_image`。

流式以与[上文](#streaming)类似的方式支持。

详细示例见 [OpenAI API Vision](openai_api_vision.ipynb)。

### 结构化输出（JSON、Regex、EBNF）

你可以指定 JSON schema、正则表达式或 [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) 来约束模型输出。模型输出将保证遵循给定的约束。一个请求只能指定一个约束参数（`json_schema`、`regex` 或 `ebnf`）。

SGLang 支持两种 grammar 后端：

- [XGrammar](https://github.com/mlc-ai/xgrammar)（默认）：支持 JSON schema、正则表达式和 EBNF 约束。
  - XGrammar 目前使用 [GGML BNF format](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)。
- [Outlines](https://github.com/dottxt-ai/outlines)：支持 JSON schema 和正则表达式约束。

如果你想改为初始化 Outlines 后端，可以使用 `--grammar-backend outlines` 标志：

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
--port 30000 --host 0.0.0.0 --grammar-backend [xgrammar|outlines] # xgrammar or outlines (default: xgrammar)
```

```python
import json
import requests

json_schema = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": "^[\\w]+$"},
        "population": {"type": "integer"},
    },
    "required": ["name", "population"],
})

# JSON (works with both Outlines and XGrammar)
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Here is the information of the capital of France in the JSON format.\n",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json_schema,
        },
    },
)
print(response.json())

# Regular expression (Outlines backend only)
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Paris is the capital of",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "regex": "(France|England)",
        },
    },
)
print(response.json())

# EBNF (XGrammar backend only)
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Write a greeting.",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "ebnf": 'root ::= "Hello" | "Hi" | "Hey"',
        },
    },
)
print(response.json())
```

详细示例见 [structured outputs](../advanced_features/structured_outputs.ipynb)。

### 自定义 logit 处理器（Custom logit processor）

启动服务器时开启 `--enable-custom-logit-processor` 标志。

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --port 30000 \
  --enable-custom-logit-processor
```

定义一个总是采样特定 token id 的自定义 logit 处理器。

```python
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

class DeterministicLogitProcessor(CustomLogitProcessor):
    """A dummy logit processor that changes the logits to always
    sample the given token id.
    """

    def __call__(self, logits, custom_param_list):
        # Check that the number of logits matches the number of custom parameters
        assert logits.shape[0] == len(custom_param_list)
        key = "token_id"

        for i, param_dict in enumerate(custom_param_list):
            # Mask all other tokens
            logits[i, :] = -float("inf")
            # Assign highest probability to the specified token
            logits[i, param_dict[key]] = 0.0
        return logits
```

发送请求：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "custom_logit_processor": DeterministicLogitProcessor().to_str(),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 32,
            "custom_params": {"token_id": 5},
        },
    },
)
print(response.json())
```

发送 OpenAI chat completion 请求：

```python
import openai
from sglang.utils import print_highlight

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0.0,
    max_tokens=32,
    extra_body={
        "custom_logit_processor": DeterministicLogitProcessor().to_str(),
        "custom_params": {"token_id": 5},
    },
)

print_highlight(f"Response: {response}")
```
