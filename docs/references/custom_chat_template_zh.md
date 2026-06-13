# 自定义聊天模板

**注意**:SGLang 项目中有两套聊天模板系统。本文档讲的是为 OpenAI 兼容 API 服务器设置自定义聊天模板(定义在 [conversation.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/parser/conversation.py))。它与 SGLang 语言前端中使用的聊天模板(定义在 [chat_template.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/lang/chat_template.py))无关。

默认情况下,服务器使用来自 Hugging Face 的模型分词器中指定的聊天模板。
对于大多数官方模型(如 Llama-2/Llama-3),它应该可以直接工作。

如果需要,你也可以在启动服务器时覆盖聊天模板:

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-2-7b-chat-hf \
  --port 30000 \
  --chat-template llama-2
```

如果你想要的聊天模板缺失,欢迎贡献它,或者从文件加载它。

## JSON 格式

你可以加载由 `conversation.py` 定义的 JSON 格式。

```json
{
  "name": "my_model",
  "system": "<|im_start|>system",
  "user": "<|im_start|>user",
  "assistant": "<|im_start|>assistant",
  "sep_style": "CHATML",
  "sep": "<|im_end|>",
  "stop_str": ["<|im_end|>", "<|im_start|>"]
}
```

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-2-7b-chat-hf \
  --port 30000 \
  --chat-template ./my_model_template.json
```

## Jinja 格式

你也可以使用 Hugging Face Transformers 定义的 [Jinja 模板格式](https://huggingface.co/docs/transformers/main/en/chat_templating)。

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-2-7b-chat-hf \
  --port 30000 \
  --chat-template ./my_model_template.jinja
```
