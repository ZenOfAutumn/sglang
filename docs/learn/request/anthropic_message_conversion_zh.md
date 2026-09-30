# Anthropic Messages 协议转换详解：`_convert_to_chat_completion_request` 与 `_convert_response`

本文回答两个问题：

1. **请求方向**：客户端 POST `/v1/messages`（Anthropic 格式）后，`_convert_to_chat_completion_request` 如何把它翻译成 SGLang 内部的 `ChatCompletionRequest`（OpenAI 格式）？
2. **响应方向**：OpenAI 的 `ChatCompletionResponse` / `ChatCompletionStreamResponse` 如何被 `_convert_response`（以及流式的 `_generate_anthropic_stream`）翻译回 Anthropic 的 `AnthropicMessagesResponse` / SSE 事件流？

Anthropic 接口本身不是本仓库的原生协议，而是一层**协议翻译适配层**：请求转成 OpenAI 格式后复用既有 chat 推理链路，得到响应后再转回 Anthropic 格式。

## 0. 全局视图

```text
客户端 POST /v1/messages  (AnthropicMessagesRequest)
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ 入口  http_server.py: anthropic_v1_messages                       │
│   app.state.anthropic_serving.handle_messages(request, raw_request)│
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ AnthropicServing.handle_messages        (serving.py:206)          │
│   ① _convert_to_chat_completion_request (serving.py:229)  ← 请求转换│
│   ② stream ? _handle_streaming : _handle_non_streaming  (分流)     │
└──────────────────────────────────────────────────────────────────┘
        │  ChatCompletionRequest（OpenAI 协议模型）
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ OpenAIServingChat（复用既有 chat 链路：校验 → 模板渲染 → 采样）       │
│   _validate_request / _convert_to_internal_request                │
│   _handle_non_streaming_request  → ChatCompletionResponse          │
│   _generate_chat_stream          → OpenAI SSE chunk 流              │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ 响应转换                                                          │
│   非流式: _convert_response            (serving.py:1259)          │
│   流式:   _generate_anthropic_stream   (serving.py:838)           │
│   错误:   _convert_openai_error_response (serving.py:1337)        │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
AnthropicMessagesResponse (JSON)  或  Anthropic SSE 事件流
```

关键文件：

| 路径 | 作用 |
| --- | --- |
| `python/sglang/srt/entrypoints/anthropic/protocol.py` | Anthropic 侧协议模型（请求/响应/内容块/流事件） |
| `python/sglang/srt/entrypoints/anthropic/serving.py` | `AnthropicServing`：全部双向转换逻辑 |
| `python/sglang/srt/entrypoints/http_server.py` | 路由 `/v1/messages`、`/v1/messages/count_tokens` 与 Anthropic 风格错误处理 |

---

## 1. 请求方向：`_convert_to_chat_completion_request`

签名（`serving.py:229`）：

```python
def _convert_to_chat_completion_request(
    self, anthropic_request: AnthropicMessagesRequest
) -> ChatCompletionRequest:
```

输入 `AnthropicMessagesRequest`（`protocol.py:361`）：`model / messages / max_tokens / system / stream / temperature / top_p / top_k / stop_sequences / tools / tool_choice / thinking / output_config / betas / metadata`。

输出 `ChatCompletionRequest`（`openai/protocol.py` 中的同名类）。

整个函数可以拆成 4 个阶段：

```text
① system 收敛     → 生成 messages[0]（role=system）
② messages 逐条翻译 → 内容块映射 + tool_result 拆成 role=tool
③ 顶层标量字段映射  → temperature / top_p / top_k / stop / stream_options
④ tools / tool_choice / thinking / output_config 收尾
```

### 1.1 阶段①：system 的处理

Anthropic 的 system 有两处来源：

- 顶层字段 `anthropic_request.system`（字符串，或 `[{type:"text", ...}]` 块数组）；
- `messages` 中 `role == "system"` 的内联消息（Claude Code / SDK 有时这么发）。

处理逻辑（`serving.py:412-433`）：

1. 先把顶层 `system` 里的 text 块收集进 `system_parts`；
2. 若 `self._merge_inline_system` 为真，再把内联 system 消息的文本也收集进来；
3. 只要 `system_parts` 非空，就在所有消息**之前**插入一条 `{"role": "system", "content": "\n".join(system_parts)}`。

`_merge_inline_system` 在构造函数中确定（`serving.py:193`）：

```python
self._merge_inline_system = not detect_inline_system_support(self._chat_template())
```

含义：**当前模型的 chat template 不支持内联 system 消息时**，才把内联 system 合并进首条 system 消息；若模板支持，则保留其原本位置与 `role="system"`（见下一节：内联 system 会走通用消息分支）。

注意内联 system 的文本抽取使用 `_extract_system_text`（`serving.py:136`）：只取 `type=="text"` 的块，逐块 `strip()`，丢弃空串，最后用 `\n` 连接；全空返回 `None`（即不产生空 system 消息）。

### 1.2 阶段②：messages 逐条翻译

对每条 `AnthropicMessage`（`protocol.py:126`，`role ∈ {user, assistant, system}`，`content` 为字符串或内容块数组）：

- 若 `role == "system"` 且 `_merge_inline_system` 为真 → 已在阶段①处理，`continue` 跳过；
- 若 `content` 是**字符串** → 直接产出 `{"role": role, "content": content}`；
- 否则进入**内容块循环**，逐块映射（见下表）。

落盘规则在循环末尾（`serving.py:540-562`）：

| 情况 | 产出 |
| --- | --- |
| `content_parts` 只有一个 text 块 | `content` 收缩为**字符串**（贴近 OpenAI 常见形态） |
| `content_parts` 有多个/非文本块 | `content` 为**数组**（多模态 parts） |
| 无 `content_parts` 但有 `tool_calls` | 产出仅含 `tool_calls` 的 assistant 消息 |
| 无内容、`role == "user"` | 不产出（该轮的 tool_result 已经翻译成独立的 `role="tool"` 消息） |
| 无内容、非 user（assistant） | 产出 `content: ""`，**保证角色交替合法**，避免严格 chat template 报错 |

#### 内容块 → OpenAI 映射表

| Anthropic 块（`protocol.py`） | OpenAI 形态 | 关键细节 |
| --- | --- | --- |
| `text` | `{"type":"text","text":...}` | 用 `is not None` 判断，**空字符串也保留占位**（`serving.py:484`），防止 assistant 空轮次导致 user→user 相邻 |
| `image` | `{"type":"image_url","image_url":{"url":...}}` | base64 → `data:{media_type};base64,{data}`；有 `url` 则直接透传；`_convert_anthropic_image_source_to_openai_part`（`serving.py:235`） |
| `search_result` | `{"type":"text","text":...}` | 由 `_text_from_search_result` 拍平成 `Title: / Source: / Content:` 多行文本（`serving.py:269`） |
| `tool_use` | assistant 消息的 `tool_calls[]` | `{"id": block.id or f"call_{uuid4hex}", "type":"function", "function":{"name":..., "arguments": json.dumps(block.input or {})}}`（`serving.py:499-508`） |
| `tool_result`（user 侧） | 独立的 `{"role":"tool","tool_call_id":...,"content":...}` 消息 | 可能拆成**多条**（见 1.3） |
| `tool_result`（assistant 侧） | `{"type":"text","text":"Tool result: ..."}` | 非 user 角色时不拆 tool 消息（`serving.py:532-538`） |
| `thinking` | assistant 的 `reasoning_content`，或重新包裹后并入 content | 见 1.4 |
| `redacted_thinking` | **不支持**，直接抛错 → 外层返回 400 | `ValueError("Anthropic redacted_thinking history is not supported")`（`serving.py:385`） |

循环开头会跳过 `thinking` / `redacted_thinking`（`serving.py:477`），因为它们的文本已经由 1.4 的重建逻辑统一处理，避免重复注入 prompt。

### 1.3 顺序保持：`tool_result` 与 `_emit_user_message`

Anthropic 允许在同一条 user 消息里混排 `text` / `image` / `tool_result`。OpenAI 侧 `role="tool"` 必须与 assistant 的 `tool_calls` 相邻配套，因此顺序必须严格保持。

做法（`serving.py:510-538`）：

1. 遇到 `tool_result` 时，若 `msg.role == "user"`：先调用 `_emit_user_message(content_parts)`（`serving.py:435`）把**此前累积**的 text/image 内容冲刷成一条 user 消息，然后才逐条追加 `role="tool"` 消息；
2. 这样线序保持 `user(pre)` → `tool` → `user(post)`，而不是把 `tool_result` 之前的文本挪到 tool 消息之后。

`_emit_user_message` 的单 text 收缩规则与主循环一致。

`tool_call_id` 取值：`block.tool_use_id or block.id or ""`（`serving.py:516`），兼容老 payload 使用 `id` 的写法。

### 1.4 assistant 历史 thinking 的重建

`_convert_assistant_thinking_blocks`（`serving.py:369`）把历史轮次的 thinking 块重建成 `(reasoning_content, text)` 二元组，二者至多一个非空：

- 若存在 `redacted_thinking` → 抛错（无法本地解析加密内容）；
- 没有任何 thinking 块 → 返回 `(None, None)`；
- 把多个 `thinking` 块用 `\n` 连接成 `reasoning_text`；
- 若 `OpenAIServingChat.supports_native_reasoning_history()` 为真（编码器本身认 `reasoning_content` 通道）→ 返回 `reasoning_text`，写入 `openai_msg["reasoning_content"]`；
- 否则调用 `wrap_reasoning_history(reasoning_text)` 用该模型的 reasoning 定界符**重新包裹**，作为 text part 拼回 content；若抛 `ValueError`（例如非 reasoning 模型），只打 WARNING 并**丢弃**该段 thinking，而不是让整个历史回放 400。

### 1.5 `tool_result.content` 的转换与 `tool_reference` 分组

`_convert_tool_result_content`（`serving.py:298`）返回 `(tool_contents, tool_text)`：

- `content` 是字符串 → `([content], content)`；
- `content` 是块数组时，逐个取出 dict/Pydantic 后按 `type` 处理：

| 输入项 | 产出 OpenAI part |
| --- | --- |
| `text` | `{"type":"text","text":...}`，同时累积进 `tool_text` |
| `image` | `image_url` part（同上） |
| `search_result` | 拍平成 text part，同时累积进 `tool_text` |
| `tool_reference` | `{"type":"tool_reference","name": ref_name}`，`ref_name = item.tool_name or item.name`（Anthropic 用 `tool_name`，SGLang 侧匹配 `name`，在此处做字段名翻译） |

随后做**连续性分组**（`serving.py:347-363`）：把 `tool_reference` 的连续片段与其他 part 分开成组。原因是 GLM 模板只在 tool 消息**开头**展开工具引用，分组能在不改变 part 顺序的前提下满足该约束。分组结果若 `tool_reference` 把内容切成多段，就会产出**多条** `role="tool"` 消息（都带同一个 `tool_call_id`）。

### 1.6 阶段③：顶层标量字段映射

构造 `request_data` 后直接 `ChatCompletionRequest(**request_data)`（`serving.py:565-588`）：

| Anthropic 字段 | OpenAI(`ChatCompletionRequest`) 字段 | 条件/说明 |
| --- | --- | --- |
| `messages` | `messages` | 上面转换结果 |
| `model` | `model` | 必填 |
| `max_tokens` | `max_tokens` | 必填；后续 `to_sampling_params` 中成为 `max_new_tokens`（`max_completion_tokens or max_tokens`） |
| `stream` | `stream` | `None` → `False` |
| `temperature` | `temperature` | 仅非 `None` 时写入 |
| `top_p` | `top_p` | 仅非 `None` 时写入 |
| `top_k` | `top_k` | SGLang 扩展，仅非 `None` 时写入 |
| `stop_sequences` | `stop` | 仅非 `None` 时写入 |
| `stream=True` | `stream_options` | 固定写 `StreamOptions(include_usage=True, continuous_usage_stats=True)`，这样才能拿到 usage 转成 Anthropic 的 `input/output_tokens` |
| `metadata` | — | 直接丢弃 |
| `system` | 见阶段① | |
| `thinking` | `chat_template_kwargs` 的 `thinking`/`enable_thinking` | 经 `apply_reasoning_enabled`，见 1.8 |
| `output_config.effort` | `reasoning_effort` | `xhigh` 折叠为 `max`（OpenAI 的 Literal 无 `xhigh`） |
| `output_config.task_budget` | — | 当前实现**仅打 INFO 日志**（注释所说的 “custom param 转发” 并未写入 `chat_request.custom_params`） |
| `betas` | — | 仅 INFO 日志（本地无 beta 开关体系） |
| `tools` | `tools` | 见 1.7 |
| `tool_choice` | `tool_choice` | 见 1.7 |

### 1.7 `tools` 与 `tool_choice` 转换

**tools**（`serving.py:657-689`）：

- `is_server_tool(tool)`（`protocol.py:235`）为真 —— 即 Anthropic 内置服务端工具（`web_search_*`、`computer_*`、`bash_*`、`text_editor_*`）—— **跳过**并打 INFO 日志，因为 OpenAI `tools` 数组要求 `input_schema`，而这些工具的实现由 Anthropic 提供；
- 自定义工具转成 `Tool(type="function", defer_loading=tool.defer_loading, function={"name":..., "description": tool.description or "", "parameters": tool.input_schema})`；
- `defer_loading=True` 的工具保留在列表里，chat template 会在初始 `<tools>` 块中隐藏它，等 `tool_reference` 指名时再按需渲染；
- 若转换后列表为空，则不设置 `chat_request.tools`。

**tool_choice**（`serving.py:696-731`）：

| Anthropic `tool_choice.type` | OpenAI `tool_choice` | 说明 |
| --- | --- | --- |
| `none` | `"none"` | 无条件生效（即使没有工具） |
| `auto` | `"auto"` | 仅在 `chat_request.tools` 非空时 |
| `any` | `"required"` | 语义都是“必须调用某个工具” |
| `tool` | `ToolChoice(type="function", function=ToolChoiceFuncName(name=tool_name))` | 先校验 `tool_name` 确实在转发过去的 tools 列表里（用属性访问 `t.function.name`），否则抛错 → 400 |
| —（未提供但 `tools` 非空） | `"auto"` | 兜底默认 |

若 `tool_choice.type ∈ {any, tool}` 但所有工具都是被跳过的服务端内置工具（`chat_request.tools` 为空），则**显式抛错**而不是悄悄降级为“不调用工具”，避免欺骗调用方。

### 1.8 `thinking` / `output_config` / `betas`

`thinking`（`serving.py:590-621`）：

- 协议层已经按 SDK 的三种变体做了形状校验（`enabled` 必须有 `budget_tokens ≥ 1024`；`disabled` 不允许其他字段；`adaptive` 不允许 `budget_tokens`，见 `protocol.py:255`）；到这一步 `budget_tokens` 只可能出现在 `enabled` 上，本地后端没有对应的硬上限旋钮，因此只打 WARNING 而不拒绝；
- `display == "omitted"` 同样只打 WARNING（本地无法在流中抑制 reasoning 文本）；
- 真正的开关是：`enabled = thinking.type != "disabled"`，随后 `self.openai_serving_chat.apply_reasoning_enabled(chat_request, enabled)`。`adaptive` 与 `enabled` 等价处理（本地无自动降档能力）。

`output_config`（`serving.py:629-643`）：`effort` → `reasoning_effort`（`xhigh` → `max`）；`task_budget` 只打 INFO。

`betas`（`serving.py:648-652`）：INFO 日志，`accept-and-log`，避免 400。

### 1.9 请求方向伪代码

```python
openai_messages = []

# ① system
system_parts = [顶层 system 的 text 块]
if _merge_inline_system: system_parts += [内联 system 消息的文本]
if system_parts: openai_messages.append({"role":"system","content":"\n".join(system_parts)})

# ② messages
for msg in anthropic_request.messages:
    if msg.role == "system" and _merge_inline_system: continue
    if isinstance(msg.content, str): openai_messages.append({role, content}); continue

    content_parts, tool_calls = [], []
    if msg.role == "assistant":
        reasoning_content, reasoning_history = 重建 thinking(...)      # 1.4
    for block in msg.content:
        text/image/search_result → content_parts
        tool_use                 → tool_calls
        tool_result(user)        → flush(content_parts) 后逐条 role=tool
        tool_result(assistant)   → content_parts += "Tool result: ..."
    if tool_calls: openai_msg["tool_calls"] = tool_calls
    按 1.2 表落盘 content_parts（含空串占位兜底）

# ③④ 标量 + tools/tool_choice/thinking
return ChatCompletionRequest(**request_data)
```

---

## 2. 响应方向（非流式）：`_convert_response`

签名（`serving.py:1259`）：`_convert_response(response: ChatCompletionResponse) -> AnthropicMessagesResponse`。

只取 `choices[0]`（Anthropic Messages 语义是单候选）。

### 2.1 内容映射（顺序固定）

| OpenAI | Anthropic 输出 | 说明 |
| --- | --- | --- |
| `choices[0].message.reasoning_content` | `ThinkingBlock(thinking=...)` | 放在 content **最前**；无 signature 时该字段整体省略（空串会让下游签名校验失败） |
| `choices[0].message.content` | `TextBlock(text=...)` | 紧随 thinking 之后 |
| `choices[0].message.tool_calls[]` | `ToolUseBlock(id, name, input)` | `input = json.loads(function.arguments)`；解析失败打 WARNING 并退化为 `{}`（`serving.py:1288-1300`） |

即输出 content 的块顺序恒为 **thinking → text → tool_use**，与原 OpenAI 侧的交错顺序无关。

### 2.2 `stop_reason` 映射

`STOP_REASON_MAP`（`serving.py:68`）：

| OpenAI `finish_reason` | Anthropic `stop_reason` |
| --- | --- |
| `stop` | `end_turn` |
| `length` | `max_tokens` |
| `tool_calls` | `tool_use` |
| `content_filter` / `abort` / 其他 | **无对应**：打 WARNING 并回落 `end_turn` |

`finish_reason` 为 `None` 时按 `"stop"` 处理。注意 `stop_sequence` 这个取值虽然协议里存在（`protocol.py:436`），但当前实现**不会**产生它。

### 2.3 `usage` 映射

由 `_anthropic_usage_from_openai`（`serving.py:110`）统一完成：

| OpenAI | Anthropic | 规则 |
| --- | --- | --- |
| `prompt_tokens` − `prompt_tokens_details.cached_tokens` | `input_tokens` | `_anthropic_input_tokens`（`serving.py:93`）：`max(prompt - cached, 0)`；若 `cached > prompt`（上游统计异常）打 WARNING 并夹到 0 |
| `prompt_tokens_details.cached_tokens` | `cache_read_input_tokens` | 仅当非 0 时输出 |
| `completion_tokens` | `output_tokens` | `force_zero_output=True` 时强制 0 |

`usage is None` 时返回全 0 的 `AnthropicUsage`。

### 2.4 兜底

| 情况 | 兜底 |
| --- | --- |
| `response.choices` 为空 | 返回 `content=[TextBlock(text="")]`、`stop_reason="end_turn"`、usage 全 0（`serving.py:1263`） |
| 转换后 `content` 为空 | 补 `TextBlock(text="")`，因为 Anthropic 要求 content 至少有一个块，否则严格 SDK 解析失败（`serving.py:1322`） |

响应 id 由 `f"msg_{uuid.uuid4().hex}"` 重新生成（不复用 OpenAI 的 `id`）。

---

## 3. 响应方向（流式）：`_generate_anthropic_stream`

非流式转换是「整体映射」，流式转换是「**状态机 +增量映射**」。上游是 `OpenAIServingChat._generate_chat_stream` 产出的 OpenAI SSE chunk，下游要产出 Anthropic 的事件序列。

### 3.1 事件序列总览

```text
event: message_start          ← 惰性触发（首个有内容的 chunk）
event: content_block_start    ← 打开一个块（text / thinking / tool_use）
event: content_block_delta    ← text_delta / thinking_delta / input_json_delta
...
event: content_block_stop
（可重复 N 个块，每块一对 start/stop，index 递增）
event: message_delta          ← stop_reason + output usage
event: message_stop
```

异常时改为：`message_start`（若未发）→ 关闭已打开的块 → `error` → `message_stop`。

### 3.2 状态机变量（`serving.py:850-858`）

| 变量 | 含义 |
| --- | --- |
| `content_block_index` | 当前块的 index，关闭块时 `+1` |
| `content_block_open` / `content_block_type` | 当前是否打开了块、块类型 |
| `captured_thinking_signature` | 缓存的 thinking 签名 |
| `finish_reason` / `final_usage` | 捕获的结束原因与最终 usage |
| `message_started` | `message_start` 是否已发送（保证只发一次） |
| `had_content_delta` | 是否产生过任何内容增量 |
| `message_id` | `msg_<uuid4hex>`，整个流共享 |

三个关键辅助闭包：

- `_ensure_content_block_events(block_type, content_block, force_new=False)`（`serving.py:910`）：按需**先关旧块再开新块**。`force_new=True` 用于连续两个 `tool_use`：每个工具必须有自己的 index 与 start/stop 对，否则第二个工具的 `input_json_delta` 会追加到第一个工具的 JSON 参数里，两个 tool call 全部损坏。
- `_close_content_block_events()`（`serving.py:882`）：类型为 `thinking` 且存在真实签名时先补一个 `signature_delta`，再发 `content_block_stop`，并把 index 推进、类型清空。
  说明：`captured_thinking_signature` 当前仅有初始化与重置、没有从上游取值，因此 `signature_delta` **实际不会发出**（等价于 Anthropic 语义中的“未签名 thinking”）。
- `_ensure_message_started(usage)`（`serving.py:941`）：只发一次 `message_start`。

### 3.3 delta 映射

| OpenAI chunk 内容 | Anthropic 事件 |
| --- | --- |
| `delta.reasoning_content` | 确保打开 `thinking` 块 → `content_block_delta(ThinkingDelta)` |
| `delta.tool_calls[*].function.name` 存在（新工具调用） | `force_new` 关闭旧块 → `content_block_start(ToolUseBlock(id/name, input={}))` → 若有 `arguments` 再发 `content_block_delta(InputJsonDelta(partial_json=...))` |
| `delta.tool_calls[*].function.arguments`（同一调用的续传） | 仅当当前块类型为 `tool_use` 时追加 `InputJsonDelta`；否则 WARNING 并**丢弃**（`serving.py:1226`） |
| `delta.content` | 确保打开 `text` 块 → `content_block_delta(TextDelta)` |
| `chunk.usage` | 记录为 `final_usage`（`include_input=False, include_output=True`）；`choices` 为空的 usage-only chunk 直接跳过 |
| `choice.finish_reason` | 仅**记录**，不提前结束循环（同一 chunk 可能还带着最后一个内容增量，提前 return 会丢内容，`serving.py:1134`） |
| `data: [DONE]` | 关闭当前块 → `message_delta(stop_reason, usage)` → `message_stop` |

`message_start` 的惰性触发条件（`serving.py:1145-1163`）：chunk 带有 reasoning / tool_calls / 非空 content / usage，或该 chunk 带 `finish_reason`。原因见代码注释：OpenAI 流会先发一个只有 role 的 chunk，此时 usage 还不可用，若在那儿发 `message_start`，客户端会拿到 `input_tokens=0`。

`message_start` 的 usage 通过 `force_zero_output=True` 把 `output_tokens` 固定为 0（此刻还没生成）；真正的 output 数量在 `message_delta` 里给。

空流兜底（`serving.py:1053`）：若到 `[DONE]` 时既没有内容增量、也没有 `finish_reason`，判定为后端静默丢流，发 `error(api_error, "Backend produced no content")` + `message_stop`；若 `finish_reason` 存在（例如 `max_tokens=1` 命中停止、内容过滤），则按正常空 completion 处理，避免客户端 SDK 误触发重试。

### 3.4 流式错误路径

`_flush_on_error(error_type, message)`（`serving.py:954`）保证错误时线序仍然合法：

1. `message_start`（若未发）；
2. 关闭已打开的 content block（严格 SDK 客户端不接受没有 `content_block_stop` 的 `content_block_start`）；
3. `error` 事件；
4. `message_stop`。

上游 chunk 解析失败时会先尝试 `_parse_upstream_error`（`serving.py:972`）识别 OpenAI handler 的流式错误信封（`{"error": {"message", "code", ...}}`，它本就通不过 `ChatCompletionStreamResponse` 校验），把真实的 type/message 转发给客户端，而不是统一压成 “Stream processing error”。

另有两处前置保护：`openai_stream.__aiter__()` 抛异常、以及首个 chunk 之前 `ValueError`（分词失败等），都转成上述规范错误序列，避免客户端看到半开 SSE / TCP 直接断开。

---

## 4. 错误响应转换

`_convert_openai_error_response`（`serving.py:1337`）把 OpenAI handler 的错误响应改成 Anthropic 的 `{"type":"error","error":{"type","message"}}`：

- 状态码 → `error.type` 由 `ERROR_TYPE_MAP`（`serving.py:74`）映射：`400 invalid_request_error`、`401 authentication_error`、`403 permission_error`、`404 not_found_error`、`408 request_timeout_error`、`429 rate_limit_error`、`503 overloaded_error`，其余 4xx/5xx 多为 `api_error`；
- 解析上游 body：JSON 里取 `error.message` / `message`，非 JSON 时取前 500 个字符；4xx 会沿用上游的 `error.type`，5xx 一律归一化；
- 文本清洗由 `_scrub_error_message`（`serving.py:161`）负责：5xx 只返回 `"Internal server error"`，4xx 保留原始信息但剔除 `Traceback` 行与 `File "/...` 行、截断到 500 字符，防止泄露堆栈/路径/PII。

`http_server.py` 里的 `validation_exception_handler` / `validation_exception_handler(RequestValidationError)` 以及 `_anthropic_validation_message` 对 `/v1/messages` 前缀的请求做同样风格的错误包装（把默认 422 改成 400）。

---

## 5. 有损 / 不一致点清单

调试时最容易踩的几处：

1. **`metadata` 丢弃**，Anthropic 侧的 user_id 等信息不会到达后端。
2. **`thinking.budget_tokens` 不生效**：只打 WARNING，本地没有硬上限旋钮，真实上限仍是 `max_tokens`。
3. **`thinking.display="omitted"` 不生效**：reasoning 文本仍会照常发出。
4. **`output_config.task_budget` 未真正转发**：代码只打 INFO 日志（注释提到的 “custom param” 未写入 `custom_params`）。
5. **`betas` 无实际作用**：accept-and-log。
6. **服务端内置工具被静默跳过**（INFO 日志）；若 `tool_choice` 指向它们则会明确报错。
7. **`redacted_thinking` 历史直接 400**；非 reasoning 模型的 `thinking` 历史可能被丢弃（WARNING）。
8. **`signature_delta` 实际不会发出**（`captured_thinking_signature` 未从上游取值）。
9. **非流式响应只取 `choices[0]`**，且 content 块顺序被规范化为 thinking → text → tool_use。
10. **`stop_sequence` 不会被产出**：停止原因只在 `end_turn / max_tokens / tool_use` 三者中取值，`content_filter` / `abort` 会以 WARNING 形式退化成 `end_turn`。
11. **`input_tokens = prompt_tokens − cached_tokens`**：Anthropic 的 `input_tokens` 不含缓存命中部分，与 OpenAI 的 `prompt_tokens` 语义不同，对账时容易对不上。

---

## 6. 相关代码索引

| 函数 / 常量 | 位置 |
| --- | --- |
| `AnthropicServing.handle_messages` | `anthropic/serving.py:206` |
| `_convert_to_chat_completion_request` | `anthropic/serving.py:229` |
| `_convert_anthropic_image_source_to_openai_part` | `anthropic/serving.py:235` |
| `_text_from_search_result` | `anthropic/serving.py:269` |
| `_convert_tool_result_content` | `anthropic/serving.py:298` |
| `_convert_assistant_thinking_blocks` | `anthropic/serving.py:369` |
| `_emit_user_message` | `anthropic/serving.py:435` |
| `_handle_non_streaming` | `anthropic/serving.py:735` |
| `_handle_streaming` | `anthropic/serving.py:789` |
| `_generate_anthropic_stream` | `anthropic/serving.py:838` |
| `_convert_response` | `anthropic/serving.py:1259` |
| `_convert_openai_error_response` | `anthropic/serving.py:1337` |
| `handle_count_tokens` | `anthropic/serving.py:1416` |
| `STOP_REASON_MAP` / `ERROR_TYPE_MAP` | `anthropic/serving.py:68` / `:74` |
| `_anthropic_usage_from_openai` | `anthropic/serving.py:110` |
| Anthropic 协议模型 | `anthropic/protocol.py` |
| 路由 `/v1/messages` | `entrypoints/http_server.py:2096` |
