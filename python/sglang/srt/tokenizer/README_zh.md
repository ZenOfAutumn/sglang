# srt/tokenizer

## 目录用途
本目录提供 SGLang 中基于 OpenAI `tiktoken` 的分词器封装，用于支持使用 tiktoken 词表的模型（如 GPT 系列风格词表）在推理服务中的编码与解码。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `tiktoken_tokenizer.py` | `TiktokenProcessor` 与 `TiktokenTokenizer`，封装 tiktoken 词表的加载、特殊 token 处理及编码/解码，适配 SGLang 的分词器接口 |
