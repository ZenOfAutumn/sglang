# srt/entrypoints/ollama

## 目录用途
本目录实现 Ollama 兼容 API 层。它把 Ollama API 请求转换为 SGLang 内部格式并返回 Ollama 兼容响应，同时提供本地 Ollama 与远端 SGLang 之间的智能路由能力，便于已有 Ollama 客户端无缝接入。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化文件（SGLang 的 Ollama 兼容 API）。 |
| `protocol.py` | Ollama API 的 Pydantic 协议模型，覆盖 chat/generate 请求与（流式）响应、消息、模型信息、tags 与 show 等接口格式。 |
| `serving.py` | `OllamaServing` 请求处理器，将 Ollama 请求转为 SGLang 内部格式并返回 Ollama 兼容（含流式）响应。 |
| `smart_router.py` | `SmartRouter` 智能路由，使用 LLM 评判将任务分类为简单/复杂，分别路由到本地 Ollama 或远端 SGLang；含命令行入口 main。 |
