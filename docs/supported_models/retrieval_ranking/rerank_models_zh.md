# 重排序模型（Rerank Models）

SGLang 通过将优化的服务框架与灵活的编程接口相结合，为重排序模型提供了全面的支持。这种设置实现了对 cross-encoder 重排序任务的高效处理，提升了搜索结果排序的准确性和相关性。SGLang 的设计确保了在重排序模型部署期间的高吞吐量和低延迟，使其成为大规模检索系统中基于语义的结果精炼的理想选择。

```{important}
SGLang 中的重排序模型分为两类：

- **Cross-encoder 重排序模型**：使用 `--is-embedding` 运行（embedding runner）。
- **仅解码器（decoder-only）重排序模型**：在运行时**不使用** `--is-embedding`，而是使用下一个 token 的 logprob 打分（yes/no）。
  - 纯文本（例如 Qwen3-Reranker）
  - 多模态（例如 Qwen3-VL-Reranker）：还支持图像/视频内容

部分模型可能需要 `--trust-remote-code`。
```

## 支持的重排序模型

| Model Family (Rerank)                          | Example HuggingFace Identifier       | Chat Template | Description                                                                                                                      |
|------------------------------------------------|--------------------------------------|---------------|----------------------------------------------------------------------------------------------------------------------------------|
| **BGE-Reranker (BgeRerankModel)**              | `BAAI/bge-reranker-v2-m3`            | N/A           | 目前仅支持 `attention-backend` 为 `triton` 和 `torch_native`。来自 BAAI 的高性能 cross-encoder 重排序模型。适用于基于语义相关性对搜索结果重排序。   |
| **Qwen3-Reranker (decoder-only yes/no)**       | `Qwen/Qwen3-Reranker-8B`             | `examples/chat_template/qwen3_reranker.jinja` | 仅解码器的重排序模型，使用下一个 token 的 logprob 对标签（yes/no）打分。启动时**不使用** `--is-embedding`。 |
| **Qwen3-VL-Reranker (multimodal yes/no)**      | `Qwen/Qwen3-VL-Reranker-2B`          | `examples/chat_template/qwen3_vl_reranker.jinja` | 支持文本、图像和视频的多模态仅解码器重排序模型。使用 yes/no logprob 打分。启动时**不使用** `--is-embedding`。 |


## Cross-Encoder 重排序（embedding runner）

### 启动命令

```shell
python3 -m sglang.launch_server \
  --model-path BAAI/bge-reranker-v2-m3 \
  --host 0.0.0.0 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --attention-backend triton \
  --is-embedding \
  --port 30000
```

### 示例客户端请求

```python
import requests

url = "http://127.0.0.1:30000/v1/rerank"

payload = {
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "what is panda?",
    "documents": [
        "hi",
        "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
    ],
    "top_n": 1,
    "return_documents": True
}

response = requests.post(url, json=payload)
response_json = response.json()

for item in response_json:
    if item.get("document"):
        print(f"Score: {item['score']:.2f} - Document: '{item['document']}'")
    else:
        print(f"Score: {item['score']:.2f} - Index: {item['index']}")
```

**请求参数：**

- `query`（必填）：用于对文档进行排序的查询文本
- `documents`（必填）：待排序的文档列表
- `model`（必填）：用于重排序的模型
- `top_n`（可选）：返回文档的最大数量。默认返回所有文档。如果指定的值大于文档总数，则返回所有文档。
- `return_documents`（可选）：是否在响应中返回文档。默认为 `True`。

## Qwen3-Reranker（仅解码器 yes/no 重排序）

### 启动命令

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Reranker-0.6B \
  --trust-remote-code \
  --disable-radix-cache \
  --host 0.0.0.0 \
  --port 8001 \
  --chat-template examples/chat_template/qwen3_reranker.jinja
```

```{note}
Qwen3-Reranker 使用仅解码器的 logprob 打分（yes/no）。请勿使用 `--is-embedding` 启动它。
```

### 示例客户端请求（支持可选的 instruct、top_n 和 return_documents）

```shell
curl -X POST http://127.0.0.1:8001/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Reranker-0.6B",
    "query": "法国首都是哪里？",
    "documents": [
      "法国的首都是巴黎。",
      "德国的首都是柏林。",
      "香蕉是黄色的水果。"
    ],
    "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
    "top_n": 2,
    "return_documents": true
  }'
```

**请求参数：**

- `query`（必填）：用于对文档进行排序的查询文本
- `documents`（必填）：待排序的文档列表
- `model`（必填）：用于重排序的模型
- `instruct`（可选）：重排序器的指令文本
- `top_n`（可选）：返回文档的最大数量。默认返回所有文档。如果指定的值大于文档总数，则返回所有文档。
- `return_documents`（可选）：是否在响应中返回文档。默认为 `True`。

### 响应格式

`/v1/rerank` 返回一个对象列表（按分数降序排列）：

- `score`：浮点数，越高表示越相关
- `document`：原始文档字符串（仅在 `return_documents` 为 `true` 时包含）
- `index`：在输入 `documents` 中的原始索引
- `meta_info`：可选的调试/使用信息（某些模型可能会出现）

返回结果的数量由 `top_n` 参数控制。如果未指定 `top_n` 或其值大于文档总数，则返回所有文档。

示例（`return_documents: true`）：

```json
[
  {"score": 0.99, "document": "法国的首都是巴黎。", "index": 0},
  {"score": 0.01, "document": "德国的首都是柏林。", "index": 1},
  {"score": 0.00, "document": "香蕉是黄色的水果。", "index": 2}
]
```

示例（`return_documents: false`）：

```json
[
  {"score": 0.99, "index": 0},
  {"score": 0.01, "index": 1},
  {"score": 0.00, "index": 2}
]
```

示例（`top_n: 2`）：

```json
[
  {"score": 0.99, "document": "法国的首都是巴黎。", "index": 0},
  {"score": 0.01, "document": "德国的首都是柏林。", "index": 1}
]
```

### 常见陷阱

- **`--chat-template` 是必需的。** 如果不加 `--chat-template examples/chat_template/qwen3_reranker.jinja`，服务器不会将该模型识别为仅解码器重排序模型，并会返回 400 错误：`"This model does not appear to be an embedding model by default. Please add `--is-embedding`..."`。正确的修复方式是添加 chat template 标志，而**不是** `--is-embedding`。
- 如果你使用 `--is-embedding` 启动 Qwen3-Reranker，`/v1/rerank` 将无法计算 yes/no 的 logprob 分数。请在**不使用** `--is-embedding` 的情况下重新启动。
- 如果你看到类似 "score should be a valid number" 的校验错误，且后端返回的是一个列表，请升级到能够将 `embedding[0]` 强制转换为 rerank 响应中 `score` 的版本。

## Qwen3-VL-Reranker（多模态仅解码器重排序）

Qwen3-VL-Reranker 扩展了 Qwen3-Reranker 以支持多模态内容，允许对包含文本、图像和视频的文档进行重排序。

### 启动命令

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-Reranker-2B \
  --trust-remote-code \
  --disable-radix-cache \
  --host 0.0.0.0 \
  --port 30000 \
  --chat-template examples/chat_template/qwen3_vl_reranker.jinja
```

```{note}
Qwen3-VL-Reranker 与 Qwen3-Reranker 一样使用仅解码器的 logprob 打分（yes/no）。请勿使用 `--is-embedding` 启动它。
```

### 纯文本重排序（向后兼容）

```python
import requests

url = "http://127.0.0.1:30000/v1/rerank"

payload = {
    "model": "Qwen3-VL-Reranker-2B",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a branch of artificial intelligence that enables computers to learn from data.",
        "The weather in Paris is usually mild with occasional rain.",
        "Deep learning is a subset of machine learning using neural networks with many layers.",
    ],
    "instruct": "Retrieve passages that answer the question.",
    "return_documents": True
}

response = requests.post(url, json=payload)
results = response.json()

for item in results:
    print(f"Score: {item['score']:.4f} - {item['document'][:60]}...")
```

### 图像重排序（文本查询，图像/混合文档）

```python
import requests

url = "http://127.0.0.1:30000/v1/rerank"

payload = {
    "query": "A woman playing with her dog on a beach at sunset.",
    "documents": [
        # Document 1: Text description
        "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset.",
        # Document 2: Image URL
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/beach_dog.jpeg"
                }
            }
        ],
        # Document 3: Text + Image (mixed)
        [
            {"type": "text", "text": "A joyful scene at the beach:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/beach_dog.jpeg"
                }
            }
        ]
    ],
    "instruct": "Retrieve images or text relevant to the user's query.",
    "return_documents": False
}

response = requests.post(url, json=payload)
results = response.json()

for item in results:
    print(f"Index: {item['index']}, Score: {item['score']:.4f}")
```

### 多模态查询重排序（带图像的查询）

```python
import requests

url = "http://127.0.0.1:30000/v1/rerank"

payload = {
    # Query with text and image
    "query": [
        {"type": "text", "text": "Find similar images to this:"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/reference_image.jpeg"
            }
        }
    ],
    "documents": [
        "A cat sleeping on a couch.",
        "A woman and her dog enjoying the sunset at the beach.",
        "A busy city street with cars and pedestrians.",
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/similar_image.jpeg"
                }
            }
        ]
    ],
    "instruct": "Find images or descriptions similar to the query image."
}

response = requests.post(url, json=payload)
results = response.json()

for item in results:
    print(f"Index: {item['index']}, Score: {item['score']:.4f}")
```

### 请求参数（多模态）

- `query`（必填）：可以是字符串（纯文本）或内容部件列表：
  - `{"type": "text", "text": "..."}` 用于文本
  - `{"type": "image_url", "image_url": {"url": "..."}}` 用于图像
  - `{"type": "video_url", "video_url": {"url": "..."}}` 用于视频
- `documents`（必填）：列表，其中每个文档可以是字符串或内容部件列表（格式与 query 相同）
- `instruct`（可选）：重排序器的指令文本
- `top_n`（可选）：返回文档的最大数量
- `return_documents`（可选）：是否在响应中返回文档（默认：`false`）

### 常见陷阱

- 对于 Qwen3-VL-Reranker，请始终使用 `--chat-template examples/chat_template/qwen3_vl_reranker.jinja`。
- 请勿使用 `--is-embedding` 启动。
- 为获得最佳效果，请使用 `--disable-radix-cache` 以避免多模态内容的缓存问题。
- **注意**：目前仅测试并支持 `Qwen3-VL-Reranker-2B`。8B 模型可能有不同的行为，不保证能与此模板正常工作。
