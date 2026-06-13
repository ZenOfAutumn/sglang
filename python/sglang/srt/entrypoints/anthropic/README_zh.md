# srt/entrypoints/anthropic

## 目录用途
本目录实现 Anthropic Messages API 的兼容层。它将 Anthropic 风格的请求转换为内部的 OpenAI ChatCompletion 格式，复用 `OpenAIServingChat` 处理，再把结果转换回 Anthropic 响应格式，使外部客户端可用 Anthropic 协议访问 SGLang。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化文件。 |
| `protocol.py` | Anthropic Messages API 的 Pydantic 协议模型，包括错误、用量、内容块、消息、工具、ToolChoice、计数 token 请求/响应、消息请求及流式事件等。 |
| `serving.py` | `AnthropicServing` 请求处理器，负责 Anthropic 与 OpenAI 格式互转、委托 OpenAIServingChat 处理，并支持 SSE 流式响应。 |
