---
title: 保留思考（Preserved thinking）
url: https://platform.claude.com/docs/en/build-with-claude/preserved-thinking
description: 修改对话现在会导致报错或丢弃思考块；本文介绍如何检查你的集成是否存在此类修改，以及如何迁移。
---

在 Claude Fable 5.1 中，修改对话之前的轮次（`system` 提示词、`tools` 或任何更早的消息）会影响 API 响应。默认情况下，API 会拒绝该请求并返回错误；你也可以选择让 API 从模型可见的上下文中丢弃受影响的思考块（`prefix_mismatch_behavior: "drop_block"`）。对于在 2026 年 8 月 31 日 00:00 UTC 或之后创建的新账户，默认强制执行此检查。更多详情请参阅 *[工作原理](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking#how-it-works)* 和 *[受影响的用户](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking#who-is-affected)*。

当你把一个思考块传回 API 时，API 会使用其 `signature` 检查此前的对话是否保持不变，以及当前模型能否读取该思考块。设置这项检查是为了防止在一组指令下产生的推理被重放到另一组可能具有对抗性的指令下。

API 提供了一系列一等能力，可在对话进行过程中修改对话，覆盖大部分需要编辑对话记录的场景：使用[对话中途系统消息](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)添加新指令，使用[轮次级系统消息](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking#per-turn-reminders)添加每轮提醒，使用[对话中途工具变更](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages#mid-conversation-tool-changes)添加或移除工具，以及使用[逐消息 effort](https://platform.claude.com/docs/en/build-with-claude/effort#change-effort-mid-conversation-beta)调整每一轮的思考深度。本文其余部分介绍如何判断你的集成是否受到影响，以及如何把常见的代理框架模式迁移到这些能力上。另一个好处是，让每个思考块之前的所有内容逐字节保持不变，也能使[提示词缓存](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)的前缀保持稳定。

是否需要进行修改，取决于由谁管理你的对话历史：

* **你使用 Claude 官方产品或 SDK：** Claude Code、claude.ai、[Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) 或 [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview)。它们会替你保持前缀不变。

* **你直接调用 Messages API**，例如在自己的 Agent 循环或其他环境中调用。你应检查代码，确保将 `messages` 数组视为只可追加。以下常见模式会编辑前缀，并使编辑位置之后的思考失效：

  * 裁剪或删除较早的轮次
  * 在客户端总结较早轮次，仅保留最近轮次
  * 向较早的轮次注入提醒，然后在下一次请求中移除
  * 每次请求都重新构建 `system` 提示词（当前时间、token 预算、模式标志等）
  * 在会话中途向 `tools` 添加条目或从中移除条目

## 工作原理

对于新请求，API 会检查：

* **模型相同或更新。** 一个思考块可以被生成它的模型及后续更新的模型读取，但不能被更早的模型读取。对话切换到更新模型时会保留其推理；切换到更早模型时，这些思考块无法通过模型检查，API 会在该次请求中丢弃它们。各模型的具体列表请参阅[保留思考](https://platform.claude.com/docs/en/build-with-claude/thinking#preserved-for-model)。
* **思考块之前的内容均未发生变化。** 包括顶层 `system` 提示词、`tools` 中的工具集合，以及该思考块之前的每一条消息。使用服务端压缩时，被检查的前缀从最近的[压缩块](https://platform.claude.com/docs/en/build-with-claude/compaction)开始。
* **此前思考块的链条未被破坏。** 更早的 `thinking` 和 `redacted_thinking` 块不属于前缀，但每个思考块都会跨轮次记录它之前的思考块。你可以从历史记录开头移除思考块；如果从中间移除一个思考块，它后面的所有思考块都会失效。

未通过模型检查的思考块一定会被丢弃。对于前缀不匹配，可以通过 `thinking.block_binding.prefix_mismatch_behavior` 选择处理方式；使用该字段需要添加 `thinking-binding-controls-2026-08-01` [Beta Header](https://platform.claude.com/docs/en/api/beta-headers)：

* `"drop_block"`：API 会移除该思考块以及对话中位于它之后的所有思考块，请求仍会成功。被丢弃的思考块不计费。响应会在顶层 `input_transformations` 数组中列出这些块（流式传输时位于 `message_start` 事件中）。
* `"error"`：API 拒绝请求并返回 400 `invalid_request_error`，其中会指出第一个校验失败的思考块。

默认值为 `"error"`。添加上述 Header 后，你可以设置该字段，并在响应中获得 `input_transformations`。

## 受影响的用户

Claude Fable 5.1。模型列表请参阅[保留思考](https://platform.claude.com/docs/en/build-with-claude/thinking#preserved-thinking)。

对于 Claude Fable 5.1，API 会对新账户强制执行检查。新账户指在 2026 年 8 月 31 日 00:00 UTC 或之后创建的账户。Claude API 和云平台采用相同定义。后续模型将对所有用户强制执行此检查。

无论账户创建时间为何，只要请求设置了 `prefix_mismatch_behavior`，就代表主动启用该检查。因此，旧账户也可以用这种方式测试。若要检查账户是否已默认启用检查，请在不添加 Beta Header 的情况下发送一个编辑过历史记录的请求：如果 API 返回 400，且错误信息中提到该 Header，就说明已默认启用。

<Note>
  如果你维护的是一个由用户使用自己 API Key 运行的工具或框架，那么新账户用户会比你更早触发这项检查，因为你自己的 Key 很可能属于旧账户。测试时请设置 `prefix_mismatch_behavior`，以便看到用户会遇到的行为。
</Note>

## 如何判断你的集成是否受影响

捕获集成在几轮正常对话中发出的精确请求体；如果产品会执行压缩或变更工具，也要覆盖这些操作。对于每一对连续请求，比较 `system`、`tools` 和 `messages` 的共有部分。在新追加轮次之前，这些内容应逐字节完全一致。

然后通过 API 进行验证。添加 `thinking-binding-controls-2026-08-01` [Beta Header](https://platform.claude.com/docs/en/api/beta-headers)，使用 `claude-fable-5-1`，并将 `thinking.block_binding.prefix_mismatch_behavior` 设置为 `"drop_block"`，然后通过你的集成运行一次正常的多轮会话。下面是该会话第二轮的请求，它把第一轮响应中的 assistant 消息完全按收到时的形式传回：

```bash
curl https://api.anthropic.com/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: thinking-binding-controls-2026-08-01" \
  -d '{
    "model": "claude-fable-5-1",
    "max_tokens": 16000,
    "thinking": {
      "type": "adaptive",
      "block_binding": { "prefix_mismatch_behavior": "drop_block" }
    },
    "system": "You are a coding agent.",
    "messages": [
      { "role": "user", "content": "Fix the failing test." },
      {
        "role": "assistant",
        "content": [
          { "type": "thinking", "thinking": "", "signature": "EqQBCkYIBxgCKkD..." },
          { "type": "text", "text": "I need to see the test first. Which file is it in?" }
        ]
      },
      { "role": "user", "content": "tests/test_auth.py" }
    ]
  }'
```

此后，每个响应都会包含顶层 `input_transformations` 数组。请在每轮记录它：

```json
{
  "input_transformations": [
    {
      "type": "thinking_dropped",
      "path": "messages.1.content.0",
      "reason": "prefix_binding_mismatch"
    }
  ]
}
```

* **每一轮都为空：** 你的集成完整保留了历史记录。
* **`reason: "prefix_binding_mismatch"`：** 在本次请求与上一次请求之间，`path` 所指思考块之前的某些内容发生了变化。对该轮之前的 `system`、`tools` 和 `messages` 做差异比较即可定位。
* **`reason: "model_binding_mismatch"`：** 对话被切换到了无法读取早期模型思考块的模型，例如经过路由或回退。这并非集成中的错误。继续发送这些思考块，让 API 丢弃当前模型无法读取的部分即可。

此方法适用于任何账户，因为设置该字段会让请求主动启用检查。如果想让 CI 在检查失败时明确报错，请改为设置 `"error"`。400 错误信息开头如下：

```text wrap
messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to "drop_block".
```

如果请求中没有 Beta Header，错误信息还会包含：``That setting requires the `thinking-binding-controls-2026-08-01` value in the `anthropic-beta` header.`` 错误信息末尾通常还会说明发生了什么变化，例如 `system` 提示词或 `tools` 列表与创建该思考块时不同。

所有错误变体请参阅[思考功能故障排查](https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting#error-thinking-block-signature)。

## 哪些操作属于编辑

在两次连续请求之间：

| 请求之间的变化 | 后续思考块 |
| --- | --- |
| 在末尾追加消息 | 有效 |
| 添加一个尚未被任何内容引用、且设置了 `defer_loading: true` 的工具 | 有效 |
| 从历史记录开头移除 `thinking` 块（移除某个位置之前的全部思考块） | 有效 |
| 修改 `system`、`tools` 和 `messages` 之外的任意请求参数（`max_tokens`、`output_config`、`tool_choice`、`metadata` 等） | 有效 |
| 添加、移动或移除 `cache_control` 标记 | 有效 |
| 使用会轮换但始终返回相同字节内容的签名 URL | 有效 |
| 服务端压缩或上下文编辑移除或替换内容 | 有效（检查比较的是你发送的内容，而非服务端编辑后的副本） |
| 将一个已清除的[轮次级系统消息](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking#per-turn-reminders)留在原位 | 有效 |
| 编辑、重新排序或删除任何较早的 `user`、`assistant` 或 `system` 消息 | 无效 |
| 向较早的用户轮次添加文本块，或移除上次添加的文本块 | 无效 |
| 修改顶层 `system` 字符串或块 | 无效 |
| 在 `tools` 中添加、移除、重命名或编辑工具 | 无效 |
| 从历史记录中间移除一个 `thinking` 块，同时保留后续思考块 | 之后的每个思考块都无效 |
| 图片或文档 URL 在下一次请求中返回不同的字节内容 | 无效 |
| 在后续请求中删除或改写同一条轮次级消息 | 无效 |

## 更新你的集成

下面的每一种模式，都会使用效果相同的 API 能力替代一种历史记录编辑方式，同时不改变较早内容的字节。

### 严格按原样追加 assistant 轮次

保存每次响应中的 `content` 数组，并在下一次请求中把它原封不动地作为 assistant 轮次传回。所有块类型都必须按照接收顺序保留，包括 `thinking` 字段为空的 `thinking` 块。不要通过会丢弃未知块类型或空字段的中间数据类型进行重新序列化。

### 使用对话中途系统消息添加指令，而非编辑 `system`

如果代码会在每次请求时重新构建顶层 `system` 提示词（例如当前时间、token 预算、模式标志、新发现的项目上下文），对话中的每个思考块都会检查失败。请在会话开始时固定 `system`；发生变化时，在变化生效的 `messages` 位置追加一条 [`role: "system"` 消息](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)：

```json
{
  "role": "system",
  "content": "The user switched the workspace to read-only mode. Do not write files until told otherwise."
}
```

模型会以系统提示词的权限级别处理该消息，同时消息之前的一切保持不变。在 Claude Fable 5.1 上不需要 Beta Header。在工具循环中，应将它放在 `tool_result` 用户消息之后，绝不能放在 assistant 的 `tool_use` 与对应 `tool_result` 之间（参阅[限制](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages#limitations)）。

### 使用轮次级系统消息发送每轮提醒

最常见的历史记录编辑是每轮提示：在每批工具结果后追加一行提醒（例如“把相互独立的读取请求放在一起”“你已经有一段时间没有向用户同步进度了”），然后在下一次请求中移除它，以免提醒不断累积。移除提醒本身就是一次编辑。

应改为在 `tool_result` 用户消息之后发送一条带有 `clear_at: "next_user_message"` 的[对话中途系统消息](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)（需要 Beta Header `mid-conversation-system-clear-at-2026-08-21`）。下面的 `messages` 数组来自经过两轮工具调用后的请求。`messages[3]` 是上一次请求的提醒，它被保留在原位；`messages[6]` 是本次请求追加的副本：

```json
[
  { "role": "user", "content": "Fix the failing test." },
  {
    "role": "assistant",
    "content": [
      { "type": "thinking", "thinking": "", "signature": "..." },
      {
        "type": "tool_use",
        "id": "toolu_01",
        "name": "read_file",
        "input": { "path": "tests/test_auth.py" }
      }
    ]
  },
  {
    "role": "user",
    "content": [{ "type": "tool_result", "tool_use_id": "toolu_01", "content": "..." }]
  },
  {
    "role": "system",
    "clear_at": "next_user_message",
    "content": "Request every independent read in one turn."
  },
  {
    "role": "assistant",
    "content": [
      { "type": "thinking", "thinking": "", "signature": "..." },
      {
        "type": "tool_use",
        "id": "toolu_02",
        "name": "read_file",
        "input": { "path": "src/auth.py" }
      }
    ]
  },
  {
    "role": "user",
    "content": [{ "type": "tool_result", "tool_use_id": "toolu_02", "content": "..." }]
  },
  {
    "role": "system",
    "clear_at": "next_user_message",
    "content": "Request every independent read in one turn."
  }
]
```

一条仅包含 `tool_result` 的用户消息也算“下一条用户消息”，因此 `messages[3]` 已经被清除：它不会渲染任何内容，也不消耗输入 token，但仍然保留在数组中，所以 `messages[4]` 中的思考仍然有效。`messages[6]` 是模型在本轮实际看到的内容。后续请求应把二者都保留在原位，并在下一个 `tool_result` 消息之后继续追加下一份提醒。轮次级消息只能包含 `text`，不能使用 `cache_control`。请把缓存断点放在前一个用户轮次上。参阅[轮次级系统消息](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages#turn-scoped-system-messages)。

如果不使用该 Beta 功能，则可在同一条用户消息的 `tool_result` 块之后追加一个 `text` 块作为提醒，并将之前的副本保留在原位。模型会遵循最新的一份提醒。

### 使用 `tool_addition` 和 `tool_removal` 更改工具，而非编辑 `tools`

如果工具集合会在会话中途变化（例如身份认证后解锁一个工具，或者模式切换后撤下危险工具），不要编辑 `tools`。应在会话开始时声明完整工具集合，并使用[对话中途工具变更](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages#mid-conversation-tool-changes)，从指定位置开始提供或撤下工具（需要 Beta Header `mid-conversation-tool-changes-2026-07-01`）。尚不可用的工具应设置 `defer_loading: true`，之后再通过 `tool_addition` 块启用；其结构与下面的 `tool_removal` 相同：

```json
{
  "role": "system",
  "content": [
    { "type": "tool_removal", "tool": { "type": "tool_reference", "name": "delete_branch" } },
    { "type": "text", "text": "Branch deletion is disabled for the rest of this session." }
  ]
}
```

如果某个工具的 Schema 到会话中途才能获知，例如运行时发现的 MCP 服务器工具，可以将其以 `defer_loading: true` 追加到 `tools`，再通过 `tool_addition` 提供。尚未被引用的延迟加载工具不属于前缀，因此追加它是安全的；追加普通工具则不安全。

### 尽可能在服务端裁剪上下文

客户端截断和总结是第二常见的编辑模式：删除或总结最早的轮次，并逐字保留最近轮次。但是，最近轮次中的思考块是在你删掉的完整历史仍然存在时生成的，因此检查会失败。服务端的等价功能不算编辑，因为检查比较的是你所发送的对话：

* [压缩](https://platform.claude.com/docs/en/build-with-claude/compaction)：当上下文接近你设定的阈值时，把较早轮次总结为一个压缩块；被检查的前缀会从该块重新开始。它的 [`instructions` 参数](https://platform.claude.com/docs/en/build-with-claude/compaction#custom-summarization-instructions)可以接收自定义总结提示词（例如“保留每个股票代码、持仓规模和明确陈述的假设”）。
* [上下文编辑](https://platform.claude.com/docs/en/build-with-claude/context-editing)：按照规则清除旧工具结果（`clear_tool_uses_20250919`），或按照从旧到新的顺序清除旧思考块（`clear_thinking_20251015`）。

### 在客户端自定义压缩

这项检查并不禁止客户端压缩。规则要更具体：**不要在被你重写的前缀后保留思考块。**

**简单压缩**是推荐形式，无需改变现有机制。当对话过长时，将其总结成一条消息，然后让下一个请求只包含该总结和新的用户轮次，不重放任何更早的轮次或思考块：`messages` 变为 `[{"role": "user", "content": "<summary of the session so far>\n\n<the next instruction>"}]`。由于不再有更早的思考块，因此不会发生校验失败，模型会基于压缩后的对话重新思考。Claude 模型使用这种方案进行过长程任务训练；对于大多数工作负载，其表现与更复杂的方案相当。与任何压缩一样，它会在压缩点重置提示词缓存。

另外两种常见模式如果保持原样会失败，各自需要一处修改：

* **保留尾部的压缩（Keep-tail compaction）**：总结较早轮次，并逐字保留最近轮次。保留轮次中的思考块是针对完整历史生成的，因此放到总结之后会检查失败。修复方法：从所有要继续保留的 assistant 轮次中删除 `thinking` 和 `redacted_thinking`，保留 `text` 和 `tool_use`；或者发送 `prefix_mismatch_behavior: "drop_block"`，由 API 将它们删除。
* **后台压缩（Background compaction）**：在非关键路径上生成总结，并在对话继续期间切换为总结，因此切换前产生的每一轮思考都早于该切换。修复方法：对于仍包含切换前思考块的每个请求都发送 `"drop_block"`（也可以自行移除这些块；切换后的第一个响应会通过 `input_transformations` 准确列出被丢弃的块），或者采用同步压缩。

从对话记录中间剪掉个别轮次，会使它们之后的所有内容失效，任何客户端方案都无法规避。对于指令变更，请使用对话中途系统消息；如需选择性移除内容，请使用服务端[上下文编辑](https://platform.claude.com/docs/en/build-with-claude/context-editing)。

不要在一轮工具调用中途压缩：如果 assistant 轮次中的 `tool_use` 仍在等待对应的 `tool_result`，就应该连同其完整思考一起传回，让模型带着原有推理完成该轮工具调用（参阅[保留思考块](https://platform.claude.com/docs/en/build-with-claude/thinking#preserving-thinking-blocks)）。

### 使用 ID 引用文件，不要使用内容会变化的 URL

对于来源为 `url` 的 `image` 或 `document` 块，检查前缀包含的是 URL 获取到的字节，而不是 URL 字符串。“最新截图”端点或已被编辑的文档会让后续思考失效；如果轮换的签名 URL 始终返回同一个文件，则不会失效。对于需要跨轮次引用的内容，请通过 [Files API](https://platform.claude.com/docs/en/build-with-claude/files) 上传一次并使用 `file_id`，或直接发送 Base64。

### 决定发生不匹配时如何处理

将集成改为只追加后，请为生产环境选择一种 `prefix_mismatch_behavior`。该字段只控制前缀不匹配。如果当前模型无法读取某个思考块（例如路由切换或[服务端回退](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback#server-side-fallback)后），该块始终会被丢弃；发送 Beta Header 时，丢弃行为会记录在 `input_transformations` 中。

* 如果前缀不匹配只可能意味着代码存在 Bug，请使用 **`"error"`**（默认值）。这样你会在测试中通过 400 错误发现问题，而不是让思考块被静默丢弃。在 Message Batches API 中，未设置该字段时默认丢弃失败的块，而不会让批处理项失败；如果希望批处理项报错，请显式设置 `"error"`。
* 如果你更希望丢弃受影响的思考块而不是让请求失败，请使用 **`"drop_block"`**，并记录 `input_transformations`。

如果在生产环境中捕获到 400，重试同一个请求无法消除错误。应添加 Beta Header，并使用 `prefix_mismatch_behavior: "drop_block"` 重试；这会准确移除校验失败的块，包括那些所在 assistant 轮次中的 `tool_use` 仍在等待 `tool_result` 的块。丢弃操作只对当次请求有效，因此该会话余下的每个请求都应继续发送 `"drop_block"` 和 Beta Header。如果不使用 Beta 功能，请从历史记录中移除全部 `thinking` 和 `redacted_thinking` 块，同时保留每轮的 `text` 和 `tool_use` 块，然后重试一次。之后应修复引发不匹配的编辑行为。

## 本文使用的 API 功能

| 功能 | 替代的操作 | 状态 | Header |
| --- | --- | --- | --- |
| [未保留思考块的控制项](https://platform.claude.com/docs/en/build-with-claude/thinking#preserved-thinking-controls)（`thinking.block_binding.prefix_mismatch_behavior`、`input_transformations`） | 在前缀不匹配时选择拒绝或丢弃，并查看被丢弃的内容 | Beta | `thinking-binding-controls-2026-08-01` |
| [对话中途系统消息](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)（`messages` 中的 `role: "system"`） | 重新构建顶层 `system` 提示词 | 稳定 | 无 |
| [轮次级系统消息](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking#per-turn-reminders)（`clear_at: "next_user_message"`） | 注入提醒并在下一次请求中删除 | Beta | `mid-conversation-system-clear-at-2026-08-21` |
| [对话中途工具变更](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages#mid-conversation-tool-changes)（`tool_addition`、`tool_removal`） | 编辑 `tools` 数组 | Beta | `mid-conversation-tool-changes-2026-07-01` |
| [压缩](https://platform.claude.com/docs/en/build-with-claude/compaction)（通过 `instructions` 自定义总结提示词） | 在客户端总结旧轮次 | Beta | `compact-2026-01-12` |
| [上下文编辑](https://platform.claude.com/docs/en/build-with-claude/context-editing)（`clear_tool_uses_20250919`、`clear_thinking_20251015`） | 在客户端删除旧工具结果或思考 | Beta | `context-management-2025-06-27` |
| [Files API](https://platform.claude.com/docs/en/build-with-claude/files)（使用 `file_id` 来源） | 请求之间内容会发生变化的 URL | 稳定 | 无 |
| [逐消息 effort](https://platform.claude.com/docs/en/build-with-claude/effort#change-effort-mid-conversation-beta)（`role: "system"` 消息中的 `output_config.effort`） | 在请求之间修改顶层 effort（保护的是提示词缓存，而非思考；effort 不属于前缀） | Beta | `mid-conversation-output-config-2026-07-01` |

在同一个请求中组合多个 Header：

```text wrap
anthropic-beta: thinking-binding-controls-2026-08-01,mid-conversation-system-clear-at-2026-08-21,mid-conversation-tool-changes-2026-07-01
```

Amazon Bedrock 和 Google Cloud 使用相同的 Beta 名称。如何通过各 SDK 发送这些 Header，请参阅 [Beta Header](https://platform.claude.com/docs/en/api/beta-headers)。

## 检查清单

* 如果由 Claude 官方产品或 SDK（Claude Code、claude.ai、Claude Managed Agents、Claude Agent SDK）管理对话历史，到这里即可，无需继续操作。
* 连续请求体中的 `system`、`tools` 和共有的 `messages` 前缀逐字节完全一致。
* 使用 `prefix_mismatch_behavior: "drop_block"` 运行完整会话时，日志中没有 `prefix_binding_mismatch` 条目。
* Assistant 轮次严格按照收到时的形式逐字节传回，包括所有块类型。
* 会话期间顶层 `system` 和 `tools` 保持固定。变更通过 `role: "system"` 消息以及 `tool_addition` / `tool_removal` 块传入。
* 每轮提醒使用轮次级系统消息（或尾部文本块），每次都追加新消息且绝不移除旧消息。
* 使用压缩或上下文编辑裁剪上下文；或者使用客户端压缩，但不能在重写后的前缀之后保留思考块，也不能拆开一轮工具调用。
* 跨轮次文件使用 `file_id` 或 Base64，而不是可变 URL。
* 已为生产环境设置 `prefix_mismatch_behavior`，并监控相关 400 错误或思考块丢弃记录。

## 常见问题

<AccordionGroup>
  <Accordion title="测试这项检查需要新账户吗？">
    不需要。发送 `thinking-binding-controls-2026-08-01` Beta Header，并设置 `thinking.block_binding.prefix_mismatch_behavior`。无论账户创建时间为何，设置该字段都会让请求主动启用检查：`"error"` 会拒绝编辑过的历史记录，并返回与新账户相同的 400；`"drop_block"` 会让请求继续，并在 `input_transformations` 中列出被丢弃的内容。参阅[如何判断你的集成是否受影响](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking#how-to-tell-whether-your-integration-is-impacted)。
  </Accordion>

  <Accordion title="如果思考块之前的任何内容发生变化，哪怕只是一项工具描述，对话是否就无法继续使用？">
    不是。失效的是历史记录中位于修改位置之后的现有思考，你可以选择如何处理。使用 `prefix_mismatch_behavior: "drop_block"` 时，API 会丢弃这些块并让请求成功：模型在该轮回答时不会获得这些推理，同时提示词缓存从编辑位置重新开始。使用默认值 `"error"` 时，API 会返回 400 并拒绝请求，直到你撤销编辑或使用 `"drop_block"` 重新发送。参阅[决定发生不匹配时如何处理](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking#decide-what-happens-on-a-mismatch)。[哪些操作属于编辑](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking#what-counts-as-an-edit)列出了会产生影响的变更。
  </Accordion>

  <Accordion title="在请求之间修改 effort 或其他思考设置，会使较早的思考失效吗？">
    不会。`output_config.effort`、`max_tokens` 和 `thinking` 配置不属于被检查的前缀；检查范围仅包含 `system`、`tools` 和 `messages`。修改顶层 effort 仍会使大部分提示词缓存失效；在 Claude Fable 5.1 上，使用[逐消息 effort](https://platform.claude.com/docs/en/build-with-claude/effort#change-effort-mid-conversation-beta)进行修改则可保留缓存。effort 消息一旦发送，就会成为历史记录的一部分，后续请求中应将它保留在原位。
  </Accordion>

  <Accordion title="我的工具列表会在会话中途变化，如何避免使对话失效？">
    不要编辑 `tools`：在会话开始时声明完整工具集合，把尚不可用的工具标记为 `defer_loading: true`，再通过 `tool_addition` 和 `tool_removal` 块启用或撤下工具。如果某个工具的 Schema 只能在会话中途得知，例如运行时发现的 MCP 服务器工具，仍然可以将其以 `defer_loading: true` 追加到 `tools` 并以相同方式提供，因为尚未被引用的延迟加载工具不属于前缀。承载这些块的 `role: "system"` 消息会成为后续思考的前缀，因此之后不要移动、改写或删除它们。参阅[使用 `tool_addition` 和 `tool_removal` 更改工具](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking#change-tools-with-tool-addition-and-tool-removal-not-by-editing-tools)。
  </Accordion>

  <Accordion title="我通过总结较早轮次并逐字保留最近轮次来压缩对话，这种方式还能使用吗？">
    如果保留的轮次仍带有其思考块，就不能直接使用：这些块是在被你替换掉的历史记录基础上生成的，因此会检查失败。应从继续保留的轮次中移除 `thinking` 和 `redacted_thinking`（其 `text` 和 `tool_use` 块继续保留），或发送 `prefix_mismatch_behavior: "drop_block"` 让 API 丢弃这些块。简单压缩（一条总结消息加下一条用户消息，不重放之前的轮次）不会留下可供检查失败的思考块，因此是推荐形式。服务端[压缩](https://platform.claude.com/docs/en/build-with-claude/compaction)和[上下文编辑](https://platform.claude.com/docs/en/build-with-claude/context-editing)不算编辑。参阅[在客户端自定义压缩](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking#custom-compaction-on-the-client)。
  </Accordion>

  <Accordion title="如何处理在会话中途发生变化的 AGENTS.md 或 CLAUDE.md 等指令文件？">
    在会话开始时加载一次，并保持顶层 `system` 提示词和 `tools` 固定。文件发生变化时，不要编辑原始内容，而是在 `messages` 中对应的位置追加新版本：如果指令来自作为操作者的你，请使用[对话中途系统消息](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)；如果文件文本被视为不可信内容，则将其放入下一条 `user` 消息，不应赋予系统提示词权限。参阅[使用对话中途系统消息添加指令](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking#add-instructions-with-a-mid-conversation-system-message-not-by-editing-system)和[限制](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages#limitations)。
  </Accordion>

  <Accordion title="重启后或第二天，我还能恢复已保存的会话吗？">
    可以。恢复的会话就是一次普通的后续请求：`system`、`tools` 和之前的 `messages` 必须与你上次发送的内容逐字节一致。请持久化实际发送和接收的内容（渲染后的系统提示词、工具定义，以及按原样返回的每个 assistant 轮次）并重放，而不是根据可能已经变化的输入重新渲染，例如日期、更新后的指令文件或新的工具版本。任何新内容都应放在追加的消息中。参阅[严格按原样追加 assistant 轮次](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking#append-assistant-turns-exactly-as-returned)。
  </Accordion>

  <Accordion title="当对话从 Claude Fable 5.1 切换到较早模型，然后再切换回来时，思考会发生什么？">
    始终发送完整历史记录，让 API 针对每次请求自行判断。较早的模型无法读取 Claude Fable 5.1 的思考块，因此 API 只会在该次请求中不向该模型展示这些块（不报错、不计费；发送 Beta Header 时，会在 `input_transformations` 中记录为 `model_binding_mismatch`）。API 从不编辑你的 `messages` 数组，所以这些块仍保留在历史记录中。当同一份历史记录切回 Claude Fable 5.1 时，这些块会重新变为可读，较早模型生成的思考也会一并保留。只有客户端自行移除这些块时，推理才会丢失，例如框架在切换模型时清除思考，或根据每个模型实际使用的内容重新构建历史记录。哪些模型能读取哪些思考块，请参阅[仅供生成它的模型或更新模型使用](https://platform.claude.com/docs/en/build-with-claude/thinking#preserved-for-model)。

    ![动画：切换到 Claude Opus 时，该轮会跳过 Claude Fable 5.1 的思考；切换回来后，所有内容会再次被读取](https://platform.claude.com/docs/images/preserved-thinking-model-switch.gif)
  </Accordion>

  <Accordion title="我的代理框架可以把某一轮路由到非 Claude 模型，这会使 Claude 较早的思考失效吗？">
    不会，前提是这些轮次追加在现有历史之后，且之前的内容均未发生变化。不含思考块的 assistant 消息与其他追加消息没有区别。请把其他模型的输出作为 `text` 和 `tool_use` 内容发送。
  </Accordion>

  <Accordion title="可以把一个对话的推理带到新对话中吗？">
    不能带入不同的对话。思考块只能用在与生成它时完全相同的 `system`、`tools` 和 `messages` 之后。因此，如果一个分支在分叉点之前原样重放历史记录，就能保留其中的思考；以其他任何内容开始的新对话都不能使用这些思考块。新对话应从任务状态总结开始，包括目标、已作出的决策、当前涉及的文件和结果，以及下一步操作，具体方式参阅[简单压缩](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking#custom-compaction-on-the-client)。
  </Accordion>
</AccordionGroup>

## 后续步骤

<CardGroup cols={2}>
  <Card title="思考功能故障排查" icon="hammer" href="https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting">
    诊断并修复最常见的思考功能故障：配置导致的 400 错误、思考块为空或缺失、因 max\_tokens 停止，以及缓存未命中。
  </Card>

  <Card title="对话中途系统消息与工具变更" icon="messages" href="https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages">
    在对话中途修改系统指令或工具可用性，同时不使此前已缓存的前缀失效。
  </Card>

  <Card title="压缩" icon="stack" href="https://platform.claude.com/docs/en/build-with-claude/compaction">
    通过服务端上下文压缩，管理接近上下文窗口限制的长对话。
  </Card>

  <Card title="提示词缓存" icon="database" href="https://platform.claude.com/docs/en/build-with-claude/prompt-caching">
    使用 `cache_control` 缓存提示词前缀，通过自动缓存或带有 5 分钟、1 小时 TTL 的显式断点降低成本和延迟。
  </Card>
</CardGroup>
