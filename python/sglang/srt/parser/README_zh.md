# srt/parser

## 目录用途
本目录集中存放 SGLang 的各类文本解析与模板处理工具，包括对话/聊天模板管理、Jinja 聊天模板内容格式检测与处理、代码补全（FIM）模板、推理（思维链）内容解析，以及 GPT-OSS 的 Harmony 结构化输出解析。它们主要服务于请求输入的提示词构造与模型输出的结构化拆分。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `conversation.py` | 对话模板核心：`Conversation`、`SeparatorStyle`、模板注册与按模型路径匹配，生成聊天/嵌入对话提示词及多模态文本拼接 |
| `jinja_template_utils.py` | Jinja 聊天模板工具：检测模板的内容格式（string/openai），并按格式处理消息内容（含多模态图片占位） |
| `code_completion_parser.py` | 代码补全模板：`FimPosition`、`CompletionTemplate` 及注册/设置逻辑，从补全请求生成 FIM 提示词 |
| `reasoning_parser.py` | 推理（reasoning/thinking）内容解析器，把模型输出拆分为 reasoning_text 与 normal_text，含 DeepSeek-R1、Qwen3、Kimi、GLM4.5、GPT-OSS、MiniMax、Nemotron3、Mistral 等多种 detector，支持流式 |
| `harmony_parser.py` | GPT-OSS 的 Harmony（T4）格式解析器，按 analysis/commentary/final 频道拆分特殊标记输出，提供 `HarmonyParser` 及流式策略 |
