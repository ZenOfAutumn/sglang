# srt/entrypoints/openai

## 目录用途
本目录实现 OpenAI 兼容 API 层，是 SGLang 对外服务入口中最核心的协议适配部分。它定义统一的请求/响应协议模型与服务处理基类，并为 chat、completions、embedding、classify、rerank、score、tokenize、transcription、responses 等多种端点提供具体的服务实现，将 OpenAI 风格请求转换为内部生成调用并组织返回结果。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化文件。 |
| `encoding_dsv32.py` | DeepSeek-V3.2 工具调用编码工具，在 OpenAI 工具格式与 DSML 之间互转（改编自 DeepSeek 官方实现）。 |
| `protocol.py` | OpenAI 兼容 API 的 Pydantic 协议模型集合，含 ModelCard/ModelList、ErrorResponse、LogProbs 及各端点的请求/响应结构。 |
| `serving_base.py` | 服务处理抽象基类 `OpenAIServingBase`，封装请求校验、错误处理、流式响应等通用逻辑。 |
| `serving_chat.py` | `/v1/chat/completions` 处理器 `OpenAIServingChat`，支持模板渲染、工具调用、多模态与流式输出。 |
| `serving_classify.py` | 分类端点处理器 `OpenAIServingClassify`，基于模型输出做文本分类。 |
| `serving_completions.py` | `/v1/completions` 处理器 `OpenAIServingCompletion`，处理文本补全请求与流式响应。 |
| `serving_embedding.py` | 嵌入端点处理器 `OpenAIServingEmbedding`，处理 embedding 请求并返回向量结果。 |
| `serving_rerank.py` | 重排序端点处理器，含 Qwen3 reranker 模板/后端检测与打分逻辑。 |
| `serving_responses.py` | `/v1/responses` 处理器 `OpenAIServingResponses`（继承 Chat），支持有状态响应与工具调用（改编自 vLLM）。 |
| `serving_score.py` | 打分端点处理器 `OpenAIServingScore`，对输入计算相关性分数。 |
| `serving_tokenize.py` | 分词/反分词端点处理器 `OpenAIServingTokenize` 与 `OpenAIServingDetokenize`。 |
| `serving_transcription.py` | 语音转写端点处理器 `OpenAIServingTranscription`，处理音频转录请求。 |
| `tool_server.py` | 工具服务器抽象 `ToolServer` 及实现（MCPToolServer、DemoToolServer），用于发现并暴露可调用工具（改编自 vLLM）。 |
| `usage_processor.py` | 无状态用量处理器 `UsageProcessor`，将原始 token 计数汇总为 UsageInfo。 |
| `utils.py` | 通用辅助函数，处理 logprobs 转换、隐藏状态、路由专家、缓存 token 明细等返回字段。 |
