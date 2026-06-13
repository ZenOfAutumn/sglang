# DeepSeek OCR (OCR-1 / OCR-2)

DeepSeek OCR 模型是用于 OCR 和文档理解的多模态（图像 + 文本）模型。

## 启动服务器

```shell
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-OCR-2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30000
```

> 你可以将 `deepseek-ai/DeepSeek-OCR-2` 替换为 `deepseek-ai/DeepSeek-OCR`。

## 提示词示例

模型卡片中推荐的提示词：

```
<image>
<|grounding|>Convert the document to markdown.
```

```
<image>
Free OCR.
```

## OpenAI 兼容请求示例

```python
import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "deepseek-ai/DeepSeek-OCR-2",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>\n<|grounding|>Convert the document to markdown."},
                {"type": "image_url", "image_url": {"url": "https://example.com/your_image.jpg"}},
            ],
        }
    ],
    "max_tokens": 512,
}

response = requests.post(url, json=data)
print(response.text)
```
