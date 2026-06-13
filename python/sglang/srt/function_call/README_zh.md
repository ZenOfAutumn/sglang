# srt/function_call

## 目录用途
本目录实现 SGLang 的工具调用（function calling / tool calling）输出解析框架。它负责把各家大模型生成的、格式各异的工具调用文本（XML、JSON、Pythonic、Harmony 等）解析为统一的 `ToolCallItem` 结构，同时支持一次性（非流式）解析与流式增量解析。每种模型对应一个 detector，由 `FunctionCallParser` 统一调度。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `function_call_parser.py` | 顶层入口 `FunctionCallParser`，维护 `tool_call_parser` 名称到各 detector 类的映射表，对外提供 `has_tool_call` / `parse_non_stream` / 流式解析接口 |
| `base_format_detector.py` | 抽象基类 `BaseFormatDetector`，定义一次性与流式增量两套解析接口及公共缓冲、JSON 解析逻辑 |
| `core_types.py` | 核心数据类型：`ToolCallItem`（解析结果）、`StreamingParseResult`（流式结果）、`StructureInfo` 等 |
| `utils.py` | 工具函数：公共前缀、部分 JSON 解析、JSON 完整性判断、从 tool schema 推断参数类型、生成 JSON schema 约束等 |
| `deepseekv3_detector.py` | DeepSeek-V3 工具调用解析器 |
| `deepseekv31_detector.py` | DeepSeek-V3.1 工具调用解析器 |
| `deepseekv32_detector.py` | DeepSeek-V3.2 工具调用解析器 |
| `glm4_moe_detector.py` | GLM-4 / GLM-4.5 MoE 工具调用解析器（XML 转 JSON 状态机） |
| `glm47_moe_detector.py` | GLM-4.7 MoE 工具调用解析器 |
| `gpt_oss_detector.py` | GPT-OSS（T4 / Harmony 格式）工具调用解析器，基于 `HarmonyParser` |
| `kimik2_detector.py` | Kimi-K2 工具调用解析器，含特殊 token 清理 |
| `lfm2_detector.py` | LFM2 工具调用解析器 |
| `llama32_detector.py` | Llama 3 / 3.2 工具调用解析器 |
| `mimo_detector.py` | MiMo 工具调用解析器 |
| `mistral_detector.py` | Mistral 工具调用解析器 |
| `pythonic_detector.py` | Pythonic（Python 函数调用语法）工具调用解析器 |
| `qwen25_detector.py` | Qwen / Qwen2.5 工具调用解析器 |
| `qwen3_coder_detector.py` | Qwen3-Coder（及 step3p5）工具调用解析器，使用 `<tool_call>`/`<function=>` 标记 |
| `step3_detector.py` | Step3 工具调用解析器 |
| `minimax_m2.py` | MiniMax M2 工具调用解析器（`<minimax:tool_call>` XML 格式） |
| `trinity_detector.py` | Trinity 工具调用解析器，继承 Qwen2.5 并先剥离 `<think>` 标记 |
| `internlm_detector.py` | InternLM / InternS1 工具调用解析器（改编自 lmdeploy） |
| `hermes_detector.py` | Hermes 风格工具调用解析器 |
| `gigachat3_detector.py` | GigaChat3 工具调用解析器 |
| `json_array_parser.py` | 当 `tool_choice="required"` 或指定具体工具、启用 JSON schema 约束时使用的纯 JSON 数组解析器，绕过模型专用解析器 |
