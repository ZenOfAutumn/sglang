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

---

## 学习计划

> 目标：从「一次工具调用如何被解析出来」入手，吃透本目录的**统一抽象 + 每模型 detector + 流式状态机 + 约束生成**四条主线。建议按阶段推进，每阶段都配了「阅读 → 动手 → 自检 → 产出物」。

### 阶段 0：建立全局认知（0.5 天）

**目标**：搞清楚工具调用解析在整条推理链路里的位置——它处理的是**模型已经生成的文本**，把非结构化输出还原成结构化的 `ToolCallItem`。

- **阅读**：本 README 的「目录用途」+ `core_types.py`（只有 35 行，先把三个数据结构记牢）。
- **核心数据结构**：
  - `ToolCallItem`：解析结果，字段 `tool_index` / `name` / `parameters`（**注意 parameters 是 JSON 字符串，不是 dict**）；
  - `StreamingParseResult`：`normal_text`（普通文本）+ `calls`（本次解析出的调用）；
  - `StructureInfo`：`begin` / `end` / `trigger`，用于生成结构化标签约束。
- **自检**：① 为什么 `parameters` 用 JSON 字符串而非 dict？（提示：流式增量拼接）② `ToolCallItem` 里为什么 `name` 可空？（提示：流式里 name 和参数可能分多次到达）
- **产出物**：一句话回答「function_call 目录的输入是什么、输出是什么」。

### 阶段 1：顶层入口 FunctionCallParser（0.5 天）

**目标**：理解「名称 → detector」的调度机制与四个对外接口。

| 阅读 | 关键点 |
| --- | --- |
| `function_call_parser.py` | `ToolCallParserEnum` 映射表；`__init__` 如何按名称实例化 detector |
| 同上 | `has_tool_call` / `parse_non_stream` / `parse_stream_chunk` 三个解析入口（都先判 `if not self.tools` 短路） |
| 同上 | `get_legacy_structural_tag` / `get_structure_constraint`（约束生成，先了解入口即可，阶段 4 深入） |

- **动手**：用 `--tool-call-parser qwen25` 起服务，发一个带 `tools` 的请求，在 `parse_non_stream` 打点，打印 `full_text` 与解析出的 `tool_call_list`。
- **自检**：① 非流式 `parse_non_stream` 与流式 `parse_stream_chunk` 返回值结构有何异同？② 为什么解析失败时要**原样返回整段文本**而不是丢弃？
- **产出物**：`FunctionCallParser` 的调用时序：输入文本 → 选 detector → detect_and_parse → ToolCallItem。

### 阶段 2：统一抽象 BaseFormatDetector ★ 本目录核心（1–2 天）

**目标**：吃透所有 detector 共享的基类逻辑，尤其是**流式增量解析的状态机与缓冲**。

| 阅读 | 关键函数 |
| --- | --- |
| `base_format_detector.py` | `detect_and_parse`（`:104`，抽象方法，非流式一次性解析） |
| 同上 | `parse_streaming_increment`（`:125`，**流式增量核心**：跨 chunk 缓冲、部分 JSON 解析、逐步吐 name/参数） |
| 同上 | `parse_base_json`（`:77`，把 `{name, arguments}` 匹配结果转成 `ToolCallItem` 列表，校验工具是否存在） |
| 同上 | `has_tool_call`（`:347`）、`structure_info`（`:358`）、`supports_structural_tag` / `get_structural_tag`（`:353` / `:375`） |
| `utils.py` | 部分 JSON 解析、JSON 完整性判断、公共前缀、从 tool schema 推断参数类型 |

- **动手**：流式发一个工具调用请求，把响应按 token 切成小 chunk，逐个喂给 `parse_stream_chunk`，打印每次的 `StreamingParseResult`——观察 name 先出现、参数分多次拼接、`tool_index` 如何递增。
- **自检**：① 一个 JSON 参数被切成两半到达时，detector 靠什么保证不解析出错？（提示：缓冲 + 部分 JSON 解析）② `tool_index` 在并行多工具调用时如何区分不同调用？③ `normal_text` 与 `calls` 何时互斥、何时同时出现？
- **产出物**：`parse_streaming_increment` 的状态机图（未进入调用 / 正在解析 name / 正在拼接参数 / 调用结束），标出缓冲区的进出。

### 阶段 3：具体模型 detector 对比（1 天）

**目标**：通过对比几种代表性格式，理解「同一套抽象如何适配各家格式」。

| detector | 格式特点 |
| --- | --- |
| `qwen25_detector.py` | `<tool_call>{"name":..,"arguments":..}</tool_call>`，JSON 风格，最经典 |
| `qwen3_coder_detector.py` | `<tool_call><function=name><parameter=k>v</parameter></function>`，XML 嵌套风格 |
| `pythonic_detector.py` | `[func(a=1, b="x")]`，用 `ast.parse` 解析 Python 语法（Llama 3.2/4） |
| `glm4_moe_detector.py` | XML→JSON 状态机 |
| `kimik2_detector.py` | 特殊 token（`<|tool_call_begin|>` 等）+ 正则，含 ID/计数器两种格式 |
| `gpt_oss_detector.py` | 基于 `HarmonyParser` 的 Harmony 格式 |

- **动手**：任选两个格式差异大的 detector（如 `qwen25` vs `pythonic`），对同一组 tools 构造各自格式的文本，分别跑 `detect_and_parse`，对比它们如何殊途同归产出相同的 `ToolCallItem`。
- **自检**：① Pythonic 格式为什么不能用 JSON 解析、要用 `ast`？② 为什么 `trinity_detector` 要先剥离 `<think>` 标记？③ 遇到未定义的函数名，detector 默认怎么处理？（提示：`SGLANG_FORWARD_UNKNOWN_TOOLS`）
- **产出物**：一张「格式 → 解析手段（JSON / 正则 / ast / 状态机）」对照表。

### 阶段 4：约束生成与 tool_choice（1 天）

**目标**：理解解析的「反向」——如何在**解码阶段**用结构化标签 / JSON schema **约束**模型只产出合法工具调用。

| 阅读 | 关键点 |
| --- | --- |
| `function_call_parser.py` | `get_structure_constraint`：`required` / `auto` / 指定工具的判定；原生 structural_tag → legacy → json_schema 三级回退 |
| 同上 | `get_legacy_structural_tag`：`$defs` 校验、strict 与 `tool_strict_level` 如何决定是否带 schema |
| `utils.py` | `get_json_schema_constraint`、`_get_tool_schema_defs`、`get_json_schema_constraint` |
| `json_array_parser.py` | `tool_choice="required"` 时绕过模型专用解析器的纯 JSON 数组路径 |

- **动手**：分别用 `tool_choice="auto"`、`"required"`、指定具体工具发三个请求，在 `get_structure_constraint` 打点，观察各自返回的约束类型（`structural_tag` / `json_schema` / None）。
- **自检**：① `auto` 模式下什么条件才会加约束？② `required` 与 `strict=True` 对是否带 schema 的影响分别是什么？③ 为什么约束生成异常时选择返回 None 而非抛错？
- **产出物**：`tool_choice`（auto/required/named）× `strict` → 最终约束类型的决策表。

### 阶段 5：测试与验证（0.5 天）

- **阅读**：`test/registered/unit/function_call/test_function_call_parser.py`（含流式分块解析用例）、各 `test_*_detector.py`。
- **动手**：仿照现有用例，为某个 detector 补一个「无参数工具」或「并行多工具调用」的流式解析测试（**按仓库规范：单测不使用 mock**）。
- **产出物**：一个能跑通的新增测试用例 + 一句话说明它覆盖了哪个边界。

### 总线索图

```
API 请求(tools, tool_choice)
   │
   ├─[解码前] get_structure_constraint ──► 结构化标签/JSON schema 约束 ──► 采样器
   │
   ▼
模型生成文本
   │
   ▼
FunctionCallParser（按名称选 detector）
   ├─ 非流式：detect_and_parse ─────────► StreamingParseResult(normal_text, calls)
   └─ 流式：  parse_streaming_increment ─► 逐 chunk 增量吐 ToolCallItem
   │
   ▼
ToolCallItem{tool_index, name, parameters(JSON str)} ──► OpenAI message.tool_calls
