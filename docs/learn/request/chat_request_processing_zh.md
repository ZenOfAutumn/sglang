# 推理请求全过程处理详解：从 POST /v1/chat/completions、POST /v1/messages 到 SSE/JSON 响应

本文回答一个问题：**客户端发起一次推理请求后，到收到完整响应（非流式 JSON 或流式 SSE）为止，服务端到底发生了什么？**

覆盖两条对外协议入口，它们在服务处理层汇合后共用同一条主干：

- **OpenAI 兼容**：`POST /v1/chat/completions`（`OpenAIServingChat`）；
- **Anthropic 兼容**：`POST /v1/messages`（`AnthropicServing`，先把 Anthropic 请求翻译成 OpenAI 请求，复用 `OpenAIServingChat`，再把响应翻译回 Anthropic 形状）。

本文只讲「一次推理请求（request）」的处理链路，即 HTTP 接入 → 协议转换 → 分词 → 调度 → 生成 → 回程 → 响应组装。进程如何启动、三类进程如何划分，见 `docs/learn/runtime/http_server_startup_zh.md` 与 `docs/learn/runtime/why_three_processes_zh.md`，本文不重复。

## 0. 一张图看全流程

```text
                         客户端请求
        ┌────────────────────┴─────────────────────┐
        ▼                                          ▼
POST /v1/chat/completions (OpenAI)         POST /v1/messages (Anthropic)
        │                                          │
        ▼                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ① HTTP 接入层  srt/entrypoints/http_server.py                             │
│   validate_json_request（校验 Content-Type）                              │
│   openai_v1_chat_completions            anthropic_v1_messages             │
│     → app.state.openai_serving_chat       → app.state.anthropic_serving   │
└─────────────────────────────────────────────────────────────────────────┘
        │  ChatCompletionRequest（OpenAI）          │ AnthropicMessagesRequest
        │                                          ▼
        │                     ┌──────────────────────────────────────────────┐
        │                     │ ②a Anthropic 兼容层  anthropic/serving.py     │
        │                     │   AnthropicServing.handle_messages           │
        │                     │    ├─ _convert_to_chat_completion_request    │
        │                     │    │    内容块/工具/thinking → OpenAI messages │
        │                     │    ├─ 复用 OpenAIServingChat 的校验与转换      │
        │                     │    └─ 生成后将 OpenAI 结果翻译回 Anthropic     │
        │                     └──────────────────────────────────────────────┘
        │  ChatCompletionRequest（转换后）                 │
        ▼                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ②b 服务处理层（两路汇合）openai/serving_base.py + serving_chat.py          │
│   OpenAIServingBase.handle_request   ← 通用骨架（校验/转换/分流/异常）      │
│     ├─ _validate_request        参数合法性                                │
│     ├─ _convert_to_internal_request                                       │
│     │    ├─ _process_messages    渲染 chat template / 工具 / reasoning     │
│     │    ├─ to_sampling_params   采样参数                                  │
│     │    └─ 构造 GenerateReqInput（内部统一请求结构）                       │
│     ├─ stream=true  → _handle_streaming_request → _generate_chat_stream    │
│     └─ stream=false → _handle_non_streaming_request → _build_chat_response │
└─────────────────────────────────────────────────────────────────────────┘
        │  GenerateReqInput
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ③ 分词与分发层  srt/managers/tokenizer_manager.py                         │
│   TokenizerManager.generate_request（异步生成器）                          │
│     ├─ normalize_batch_and_arguments / _init_req_state                    │
│     ├─ 暂停门控 + model_update_lock（读锁，与权重更新互斥）                  │
│     ├─ _tokenize_one_request → _validate_one_request → _create_tokenized_object │
│     ├─ _send_one_request（ZMQ PUSH 发往 Scheduler）                        │
│     └─ _stream_one_response（等 ReqState.event，逐片 yield 回上层）          │
└─────────────────────────────────────────────────────────────────────────┘
        │  TokenizedGenerateReqInput (ZMQ)
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ④ 调度与执行  srt/managers/scheduler.py + model_executor                 │
│   event_loop：recv_requests → process_input_requests → get_next_batch     │
│              → run_batch（GPU 前向/采样）→ process_batch_result            │
└─────────────────────────────────────────────────────────────────────────┘
        │  Batch*Output (ZMQ，经 Detokenizer)
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ⑤ 回程  TokenizerManager.handle_loop → _handle_batch_output               │
│   按 rid 写入 ReqState.out_list 并 set(event)，唤醒 _stream_one_response   │
└─────────────────────────────────────────────────────────────────────────┘
        │  yield 中间片段/最终结果
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ⑥ 响应组装（按入口分流）                                                   │
│   OpenAI   ：SSE chunk（流式）/ ChatCompletionResponse（非流式）           │
│   Anthropic：_generate_anthropic_stream（OpenAI SSE → Anthropic 事件流）   │
│              _convert_response（ChatCompletionResponse → Messages 响应）  │
└─────────────────────────────────────────────────────────────────────────┘
```

关键点先说清楚：

- **协议模型（`ChatCompletionRequest`）只活在 ② 层**。进入 ③ 层时已经变成内部结构 `GenerateReqInput`，`rid`、`sampling_params`、`input_ids/text` 都在这时定型。
- **一次请求的身份是 `rid`（request id）**。`rid` 是贯穿 ②~⑤ 的唯一键，`TokenizerManager.rid_to_state` 用它在回程里找到等待协程。
- **流式与非流式的差异只体现在 ② 层如何消费 ③ 层 yield 出来的片段**，③~⑤ 的主干完全一致。
- **Anthropic 是一条“翻译层 + 复用主干”的入口**：`AnthropicServing` 只做「Anthropic ↔ OpenAI」的形状/语义转换，转换完成后与 OpenAI 入口走完全相同的 `OpenAIServingChat` 与 ③~⑤ 主干。

---

## 1. HTTP 接入层

### 1.1 路由与依赖校验

`/v1/chat/completions` 是一个 FastAPI 路由，挂了一个 `validate_json_request` 依赖：

```1816:1823:python/sglang/srt/entrypoints/http_server.py
@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI 兼容的对话补全端点 /v1/chat/completions。"""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )
```

`validate_json_request` 只做一件事：要求 `Content-Type: application/json`，否则返回 415 语义的校验错误：

```679:692:python/sglang/srt/entrypoints/http_server.py
async def validate_json_request(raw_request: Request):
    """校验请求的 Content-Type 必须为 application/json，否则抛出校验错误。"""
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=[
                {
                    "loc": ["header", "content-type"],
                    "msg": "Unsupported Media Type: Only 'application/json' is allowed",
                    "type": "value_error",
                }
            ]
        )
```

注意：**请求体到 `ChatCompletionRequest` 的解析由 FastAPI/Pydantic 自动完成**，字段的合法性（token 越界、类型不符等）在这一步就会以 422 返回，不会进入 handler。请求体协议模型定义在 `protocol.py`：

```823:830:python/sglang/srt/entrypoints/openai/protocol.py
class ChatCompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[ChatCompletionMessageParam]
    model: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="Model name. Supports LoRA adapters via 'base-model:adapter-name' syntax.",
    )
```

### 1.2 handler 是何时、如何注册的

`app.state.openai_serving_chat` 并不是硬编码的类，而是在 `lifespan` 启动钩子里用 `tokenizer_manager.serving_chat_class` 动态实例化的（允许按模型插件替换实现）：

```330:338:python/sglang/srt/entrypoints/http_server.py
    # Initialize OpenAI serving handlers
    fast_api_app.state.openai_serving_completion = OpenAIServingCompletion(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_chat = (
        _global_state.tokenizer_manager.serving_chat_class(
            _global_state.tokenizer_manager, _global_state.template_manager
        )
    )
```

因此路由函数本身极薄：它做的全部工作就是**把解析好的 `ChatCompletionRequest` 转交给 chat handler**。`raw_request` 被一并传入，后续用于读取自定义 header（如 `x-data-parallel-rank`）、检测客户端断连、提取 trace 上下文。

---

## 2. 服务处理层

### 2.1 通用骨架：OpenAIServingBase.handle_request

所有 OpenAI 端点共享同一个处理骨架。它把「校验 → 转换 → 分流 → 异常归一化」串起来：

```73:109:python/sglang/srt/entrypoints/openai/serving_base.py
    async def handle_request(
        self, request: OpenAIServingRequest, raw_request: Request
    ) -> Union[Any, StreamingResponse, ErrorResponse]:
        """Handle the specific request type with common pattern
        If you want to override this method, you should be careful to record the validation time.
        """
        received_time = monotonic_time()

        try:
            # Validate request
            error_msg = self._validate_request(request)
            if error_msg:
                return self.create_error_response(error_msg)

            # Log the raw OpenAI request payload before conversion to tokenized form.
            request_logger = self.tokenizer_manager.request_logger
            if request_logger.log_requests and request_logger.log_requests_level >= 2:
                request_logger.log_openai_received_request(request, request=raw_request)

            # Convert to internal format
            adapted_request, processed_request = self._convert_to_internal_request(
                request, raw_request
            )

            if isinstance(adapted_request, (GenerateReqInput, EmbeddingReqInput)):
                # Only set timing fields if adapted_request supports them
                adapted_request.received_time = received_time

            # Note(Xinyuan): raw_request below is only used for detecting the connection of the client
            if hasattr(request, "stream") and request.stream:
                return await self._handle_streaming_request(
                    adapted_request, processed_request, raw_request
                )
            else:
                return await self._handle_non_streaming_request(
                    adapted_request, processed_request, raw_request
                )
```

异常路径在这一层被统一收敛（`serving_base.py:110-133`）：

| 异常 | 返回 |
| --- | --- |
| `HTTPException` | 按 `status_code` 转成 `ErrorResponse` |
| `ValueError` | 400 BadRequest |
| `DS32EncodingError` | 400 BadRequest |
| 其它 `Exception` | 500 InternalServerError |

这点很重要：**`serving_chat` 内部抛 `ValueError` 即可自动变成 400**，例如改流式约束、超长上下文等。

### 2.2 请求校验：_validate_request

`serving_chat.py:865-942` 的 `_validate_request` 是 chat 特有的校验，覆盖：

- `messages` 非空；
- `return_sampling_mask` 依赖 `return_meta_info`；
- 非多模态模型收到 image/video/audio content part 时拒绝（`_validate_media_content`）；
- `tool_choice=required` 时必须有工具；指定具名工具必须存在于工具列表；工具名去重；
- 工具 `parameters` 必须是合法 JSON Schema（`Draft202012Validator.check_schema`，且先做 DB/ORM 别名归一化 `normalize_json_schema_types`，防递归炸栈）；
- `max_completion_tokens/max_tokens` 超过模型 `context_length` 且未开启 `allow_auto_truncate` 时拒绝；
- `response_format=json_schema` 必须带 `schema_`。

```925:941:python/sglang/srt/entrypoints/openai/serving_chat.py
        max_output_tokens = request.max_completion_tokens or request.max_tokens
        server_context_length = self.tokenizer_manager.server_args.context_length
        if (
            max_output_tokens
            and server_context_length
            and max_output_tokens > server_context_length
        ) and not self.tokenizer_manager.server_args.allow_auto_truncate:
            return (
                f"max_completion_tokens is too large: {max_output_tokens}."
                f"This model supports at most {server_context_length} completion tokens."
            )
```

### 2.3 核心转换：_convert_to_internal_request

这是 ② 层最重的一步，把 OpenAI 协议请求变成内部 `GenerateReqInput`。它做了四类事情：

**(1) 提取并归一化 reasoning_effort / 校验流式互斥参数**（`serving_chat.py:971-1000`）。例如 `return_token_ids`、`return_meta_info`、`return_prompt_token_ids` 都不允许和 `stream=true` 同用。

**(2) 渲染 prompt 并构造采样参数**：

```1002:1012:python/sglang/srt/entrypoints/openai/serving_chat.py
        is_multimodal = self.tokenizer_manager.model_config.is_multimodal

        # Process messages and apply chat template
        processed_messages = self._process_messages(request, is_multimodal)
        # Build sampling parameters
        sampling_params = request.to_sampling_params(
            stop=processed_messages.stop,
            model_generation_config=self.default_sampling_params,
            tool_call_constraint=processed_messages.tool_call_constraint,
            renderer_handles_response_format=self.chat_encoding_spec == "kimi_k3",
        )
```

`_process_messages`（`serving_chat.py:1100-1204`）内部逻辑：

1. 合并 `default_chat_template_kwargs`（服务端默认）与请求里的 `chat_template_kwargs`；
2. 若是 gpt-oss / gemma4，强制保留 special tokens；
3. 判定是否进入 thinking/reasoning 模式（`_get_reasoning_from_request`）；
4. 处理工具：把工具定义交给 `FunctionCallParser.get_structure_constraint` 生成结构化约束（工具调用语法约束），并把 tool 定义注入模板参数；
5. 渲染模板：
   - 有 `input_ids`：跳过模板，直接用；
   - `template_manager.chat_template_name is None`：走 `_apply_jinja_template`（Jinja / 自定义 encoder 分支）；
   - 否则走 `_apply_conversation_template`；
6. 回填 `stop`、`tool_call_constraint`、`require_reasoning`、`skip_special_tokens`。

采样参数转换在协议层完成（`protocol.py:1073` 起），优先级为 **用户显式值 > 模型 generation_config > OpenAI 默认值**，并把 `response_format`（json_schema / json_object / structural_tag）也折叠进 `sampling_params`。

**(3) 根据模型类型决定 prompt 载体**（`serving_chat.py:1014-1037`）：`input_ids`、`text`、或「多模态文本 + 占位符」。这一步决定了后续 ③ 层是直接走 token ids 还是重新分词。

**(4) 构造内部请求 `GenerateReqInput`**：

```1052:1098:python/sglang/srt/entrypoints/openai/serving_chat.py
        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            image_data=processed_messages.image_data,
            video_data=processed_messages.video_data,
            audio_data=processed_messages.audio_data,
            sampling_params=sampling_params,
            return_logprob=request.logprobs,
            logprob_start_len=-1,
            top_logprobs_num=request.top_logprobs or 0,
            return_sampling_mask=request.return_sampling_mask,
            stream=request.stream,
            return_text_in_logprobs=True,
            modalities=processed_messages.modalities,
            lora_path=lora_path,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
            routed_dp_rank=effective_routed_dp_rank,
            disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,
            return_hidden_states=request.return_hidden_states,
            return_routed_experts=request.return_routed_experts,
            routed_experts_start_len=request.routed_experts_start_len,
            rid=request.rid,
            session_id=request.session_id,
            extra_key=request.extra_key,
            cache_salt=request.cache_salt,
            require_reasoning=processed_messages.require_reasoning,
            priority=request.priority,
            routing_key=self.extract_routing_key(raw_request),
            custom_labels=custom_labels,
            custom_logit_processor=request.custom_logit_processor,
            images_config=getattr(request, "images_config", None),
            video_config=_build_video_config(request),
            image_max_dynamic_patch=img_max_dynamic_patch,
            video_max_dynamic_patch=vid_max_dynamic_patch,
            max_dynamic_patch=getattr(request, "max_dynamic_patch", None),
            use_audio_in_video=getattr(request, "use_audio_in_video", False),
            return_prompt_token_ids=request.return_prompt_token_ids
            or request.return_token_ids,
        )
```

另外，`raw_request` 的 header 里若带了自定义标签、`x-smg-routing-key`、`X-Data-Parallel-Rank`，会在这里被抽出来写入内部请求；开启 `SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES` 时还会做 header 覆盖。返回的第二个值就是原 `ChatCompletionRequest`（下游用它来组织响应，而不是再解析一遍内部结构）。

### 2.4 流式分支：_handle_streaming_request

流式的关键技巧是**先把生成器 kick-start 一步**，让可能的校验错误（如上下文超限）在 HTTP 200 发送之前暴露，从而还能返回 400：

```1515:1541:python/sglang/srt/entrypoints/openai/serving_chat.py
    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse]:
        """Handle streaming chat completion request"""
        generator = self._generate_chat_stream(adapted_request, request, raw_request)

        # Kick-start the generator to trigger validation before HTTP 200 is sent.
        # If validation fails (e.g., context length exceeded), we can still return
        # a proper HTTP 400 error response instead of streaming it as SSE payload.
        try:
            first_chunk = await generator.__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        async def prepend_first_chunk():
            yield first_chunk
            async for chunk in generator:
                yield chunk

        return StreamingResponse(
            prepend_first_chunk(),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )
```

注意 `background=self.tokenizer_manager.create_abort_task(...)`：它把「请求结束/客户端断开时如何清理」挂到 FastAPI 的后台任务上，是取消路径的一部分。

真正的 SSE 生成在 `_generate_chat_stream`（`serving_chat.py:1543-1807`）。它的主干是一个 `async for content in self.tokenizer_manager.generate_request(...)`，即**直接消费 ③ 层这个异步生成器**。对每个片段它：

- 累计 usage（prompt/completion/reasoning/cached、多模态 token 等）；
- 处理 logprobs（增量切片）；
- 判定 `finish_reason`，其中 `abort + status_code` 会走「流式错误 chunk」；
- 首个片段先发一个 `role=assistant, content=""` 的空 delta（OpenAI 语义）；
- 调用 `_generate_stream_content` 产出内容/reasoning/tool_call 三类 delta，经 `build_sse_content` 序列化成 `data: {...}\n\n`；
- 收尾时补发 `finish_reason` chunk、hidden_states、sglext（routed_experts 等）、usage chunk，最后 `data: [DONE]\n\n`。

SSE 序列化的格式在 `sse_utils.py`：

```52:99:python/sglang/srt/entrypoints/openai/sse_utils.py
def build_sse_content(
    chunk_id: str,
    created: int,
    model: str,
    index: int,
    role: Optional[str] = None,
    content: Optional[str] = None,
    reasoning_content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    logprobs: Optional[dict] = None,
    matched_stop: Union[None, int, str] = None,
    usage: Optional[dict] = None,
) -> str:
    """Build an SSE chunk string for content/reasoning updates.
    ...
    """
    delta = StreamDelta(role=role, content=content, reasoning_content=reasoning_content)
    choice = StreamChoice(
        index=index,
        delta=delta,
        logprobs=logprobs,
        finish_reason=finish_reason,
        matched_stop=matched_stop,
    )
    chunk = StreamChunk(
        id=chunk_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[choice],
        usage=usage,
    )
    return (_SSE_DATA_B + _stream_encoder.encode(chunk) + _SSE_NL_B).decode()
```

### 2.5 非流式分支：_handle_non_streaming_request

非流式只取生成器的**第一个（也是唯一一个）结果**，然后一次性组装：

```1809:1832:python/sglang/srt/entrypoints/openai/serving_chat.py
    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[ChatCompletionResponse, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming chat completion request"""
        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_chat_response(
            request,
            ret,
            int(time.time()),
        )

        return response
```

`_build_chat_response`（`serving_chat.py:1834` 起）对每个 choice 做：解码文本、拆分 reasoning、解析 tool calls、处理 logprobs/hidden states，最终拼成 `ChatCompletionResponse`（或按需返回带 meta_info 的 `ORJSONResponse`）。

---

### 2.6 Anthropic Messages API 兼容层（/v1/messages）

Anthropic 入口不是另起一条推理链，而是**一个协议翻译层**：把 Anthropic Messages 请求转成 OpenAI `ChatCompletionRequest`，复用 `OpenAIServingChat` 的校验、转换与生成，再把结果翻译回 Anthropic 形状。到 `GenerateReqInput` 之后，两条入口共用完全相同的主干（第 3~5 节）。

#### 2.6.1 路由与注册

```2096:2113:python/sglang/srt/entrypoints/http_server.py
@app.post("/v1/messages", dependencies=[Depends(validate_json_request)])
async def anthropic_v1_messages(
    request: AnthropicMessagesRequest, raw_request: Request
):
    """Anthropic 兼容的 Messages API 端点 /v1/messages。"""
    return await raw_request.app.state.anthropic_serving.handle_messages(
        request, raw_request
    )


@app.post("/v1/messages/count_tokens", dependencies=[Depends(validate_json_request)])
async def anthropic_v1_count_tokens(
    request: AnthropicCountTokensRequest, raw_request: Request
):
    """Anthropic 兼容的 token 计数端点 /v1/messages/count_tokens。"""
    return await raw_request.app.state.anthropic_serving.handle_count_tokens(
        request, raw_request
    )
```

`AnthropicServing` 在 `lifespan` 里用已经建好的 chat handler 构造，即它从实例化起就持有 `OpenAIServingChat` 的引用：

```365:367:python/sglang/srt/entrypoints/http_server.py
    # Initialize Anthropic-compatible serving handler
    fast_api_app.state.anthropic_serving = AnthropicServing(
        fast_api_app.state.openai_serving_chat
    )
```

#### 2.6.2 主入口：handle_messages

```206:227:python/sglang/srt/entrypoints/anthropic/serving.py
    async def handle_messages(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> Union[JSONResponse, StreamingResponse]:
        """Main entry point for /v1/messages endpoint."""
        try:
            chat_request = self._convert_to_chat_completion_request(request)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error converting Anthropic request: %s", e)
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=str(e),
            )

        if request.stream:
            return await self._handle_streaming(chat_request, request, raw_request)
        else:
            return await self._handle_non_streaming(chat_request, request, raw_request)
```

要点：**Anthropic → OpenAI 的转换失败会直接返回 Anthropic 风格的 400**，不会进入推理层。

#### 2.6.3 请求翻译：_convert_to_chat_completion_request

这是兼容层最重的一步（`serving.py:229-733`）。它把 Anthropic 的 `messages` / `content blocks` / `tools` 逐项映射为 OpenAI 结构：

| Anthropic 侧 | OpenAI 侧 | 说明 |
| --- | --- | --- |
| `system`（字符串或 text block 列表） | 一条 `role=system` message | 多个 text block 用 `\n` 合并；模板不支持内联 system 时（`_merge_inline_system`），还会把历史里 `role=system` 的消息一并合并 |
| `text` block | `{"type":"text"}` content part | 空字符串也保留占位，避免 role 交替断裂 |
| `image` block | `{"type":"image_url"}` part | base64 → `data:<media_type>;base64,...`；也支持 `url` |
| `tool_use` block（assistant） | `tool_calls` | `id`（缺省补 `call_<uuid>`）、`name`、`input` 序列化为 arguments JSON |
| `tool_result` block（user） | 独立的 `role=tool` message | 展开 `tool_call_id` + content；content 内的 text/image/tool_reference/search_result 分别处理 |
| `thinking` / `redacted_thinking` | `reasoning_content` 或文本 history | 见下方 thinking 处理 |
| `search_result` block | 文本 part | 拼成 `Title/Source/Content` 文本 |

几处关键设计：

- **wire 顺序保持**：`_emit_user_message`（`serving.py:435-449`）在遇到 `tool_result` 前先把已累积的 user 文本/图片 flush 成一条 user 消息，保证 `user(pre) → tool → user(post)` 的顺序；否则 text/image 会被挪到 tool 消息之后。
- **thinking 历史重建**：`_convert_assistant_thinking_blocks`（`serving.py:369-410`）把历史 assistant 的 thinking 块还原：原生支持推理历史（`supports_native_reasoning_history()`）时作为 `reasoning_content`；否则用 `wrap_reasoning_history` 重新包成文本拼进 content。`redacted_thinking`（加密内容无法解析）直接报错；非推理模型上 rewrap 失败则告警丢弃，避免历史回显把整个请求打成 400。
- **采样参数**：`model / max_tokens / temperature / top_p / top_k / stop_sequences` 直接映射；`stream=true` 时强制注入 `stream_options = StreamOptions(include_usage=True, continuous_usage_stats=True)`，以便流式回传 usage。
- **thinking → 推理开关**：`thinking.type != "disabled"` 时调用 `apply_reasoning_enabled`；`budget_tokens` 与 `display="omitted"` 本地无对应能力，accept-and-log。
- **output_config**：`effort` 映射到 OpenAI `reasoning_effort`（`xhigh` 折叠为 `max`），`task_budget` 作为软提示记录。
- **tools**：自定义工具映射为 OpenAI `function`（`input_schema → parameters`）；Anthropic 服务端内置工具（`web_search_*`、`computer_*` 等）无 client schema，跳过并打日志。
- **tool_choice**：`none→"none"`、`auto→"auto"`、`any→"required"`、`tool→`具名 `ToolChoice`（并校验该工具确实存在于转发列表，否则报错）。
- `betas`：accept-and-log。

#### 2.6.4 校验与生成：复用 OpenAIServingChat

Anthropic 路径**不经过** `OpenAIServingBase.handle_request`，而是直接调用 chat handler 的内部方法：

```756:768:python/sglang/srt/entrypoints/anthropic/serving.py
        try:
            # Convert to internal request
            adapted_request, processed_request = (
                self.openai_serving_chat._convert_to_internal_request(
                    chat_request, raw_request
                )
            )
            adapted_request.received_time = received_time

            # Get response from OpenAI handler
            response = await self.openai_serving_chat._handle_non_streaming_request(
                adapted_request, processed_request, raw_request
            )
```

- 非流式：`_validate_request` → `_convert_to_internal_request` → `_handle_non_streaming_request`（`serving.py:735-787`）。
- 流式：`_validate_request` → `_convert_to_internal_request` → 直接消费 `_generate_chat_stream` 产出的 OpenAI SSE（`serving.py:838-848`）。

因为复用同一批内部方法，**两条入口的 `rid`、`ReqState`、ZMQ 分发、调度与回程完全一致**，差别仅在外层如何解释输入与输出。

#### 2.6.5 响应翻译

**非流式**：`_handle_non_streaming` 拿到 `ChatCompletionResponse` 后交给 `_convert_response`（`serving.py:1259-1335`）翻译为 Anthropic `MessagesResponse`：

- `message.reasoning_content` → `ThinkingBlock`；
- `message.content` → `TextBlock`；
- `message.tool_calls` → `ToolUseBlock`（arguments JSON 反序列化为 `input`，非法 JSON 降级为空 dict 并告警）；
- `finish_reason → stop_reason` 映射：`STOP_REASON_MAP = {stop→end_turn, length→max_tokens, tool_calls→tool_use}`（`serving.py:68-72`）；
- usage 映射（`_anthropic_usage_from_openai`）；
- Anthropic 要求 content 至少一个块，空补 `TextBlock(text="")`。

**流式**：`_generate_anthropic_stream`（`serving.py:838` 起）逐行读取 OpenAI SSE，翻译为 Anthropic 事件序列：

```text
message_start
  → [content_block_start → content_block_delta* → content_block_stop] × 块数（thinking / text / tool_use）
  → message_delta（stop_reason + usage）
  → message_stop
```

- 内容块管理用 `content_block_index` + `_ensure_content_block_events`；连续两个 `tool_use` 必须 `force_new=True` 各自开新块，否则第二个工具的 `input_json_delta` 会追加到第一个工具的参数上。
- `message_start` 延后到首个带真实 usage/content 的 chunk，避免发出 `input_tokens=0`。
- 错误路径由 `_flush_on_error` 兜底：保证补发 `message_start`、关闭未闭合的 content block、发 `ErrorEvent` 和 `message_stop`，让严格 SDK 也能解析。
- OpenAI handler 的流式错误 envelope 会被 `_parse_upstream_error` 识别并转成 Anthropic 错误事件。

**错误映射**：`ERROR_TYPE_MAP`（`serving.py:74-85`）把 HTTP 状态码映射到 Anthropic `error.type`；`_convert_openai_error_response`（`serving.py:1337-1384`）对 4xx 保留原始 message（轻度清洗），对 5xx 一律返回通用 `"Internal server error"`，避免泄露 `str(e)` 中的路径/prompt 片段。

#### 2.6.6 count_tokens

`handle_count_tokens`（`serving.py:1416` 起）同样先做 Anthropic → OpenAI 转换，然后复用 `_process_messages` 得到 `prompt_ids` 并计数：

```1452:1466:python/sglang/srt/entrypoints/anthropic/serving.py
            processed = self.openai_serving_chat._process_messages(
                chat_request, is_multimodal
            )

            if isinstance(processed.prompt_ids, list):
                input_tokens = len(processed.prompt_ids)
            else:
                # prompt_ids is a string (multimodal case) — tokenize it
                tokenizer = self.openai_serving_chat.tokenizer_manager.tokenizer
                input_tokens = len(tokenizer.encode(processed.prompt_ids))

            return JSONResponse(
                content=AnthropicCountTokensResponse(
                    input_tokens=input_tokens
                ).model_dump()
            )
```

---

## 3. 分词与分发层：TokenizerManager

`generate_request` 是 ③ 层的核心入口，本身是一个**异步生成器**：调用方要么 `__anext__()` 取一次（非流式），要么 `async for` 持续消费（流式）。

```864:938:python/sglang/srt/managers/tokenizer_manager.py
    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        # 中译：核心入口——异步生成器。每个 HTTP/Engine 请求都从这里进入，逐个 yield 输出片段。
        #       流程：确保后台 handle_loop 已启动 -> 归一化参数 -> 校验 -> 暂停门控 ->
        #       持「读锁」(与权重更新互斥) -> 分词并发给调度器 -> 等待并 yield 各响应。
        # 中译：首次调用时惰性创建后台事件循环任务（handle_loop / sigterm_watchdog）。
        self.auto_create_handle_loop()

        # Normalize the request
        # 中译：归一化批量与各项参数（把单条/批量统一成可索引的形态）。
        obj.normalize_batch_and_arguments()
        self._set_default_priority(obj)
        ...
        # 中译：为请求在 rid_to_state 中登记 ReqState（含 event、时间统计等）。
        self._init_req_state(obj, request)
        try:
            if get_disagg().language_only:
                self._handle_epd_disaggregation_encode_request(obj)

            # Log the request
            self.request_logger.log_received_request(obj, self.tokenizer, request)

            # 中译：暂停门控——若处于暂停态，挂起在条件变量上直到 continue_generation 唤醒。
            # 中译：持「读锁」整个生成过程。权重更新持写锁，故此处保证不会在权重更新中途用到半新半旧权重。
            # 中译：单条请求：分词 -> 发给调度器 -> 逐个 yield 响应；批量请求转交 _handle_batch_request。
            async with self.is_pause_cond:
                await self.is_pause_cond.wait_for(lambda: not self.is_pause)

            async with self.model_update_lock.reader_lock:
                await self._validate_and_resolve_lora(obj)
                # Tokenize the request and send it to the scheduler
                if obj.is_single:
                    tokenized_obj = await self._tokenize_one_request(obj)
                    state = self.rid_to_state[obj.rid]
                    if obj.return_prompt_token_ids:
                        state.prompt_token_ids = list(tokenized_obj.input_ids)
                    self._send_one_request(tokenized_obj)
                    async for response in self._wait_one_response(obj, request):
                        yield response
                else:
                    async for response in self._handle_batch_request(obj, request):
                        yield response
```

分步说明：

- **`auto_create_handle_loop()`**：惰性启动回程事件循环 `handle_loop`（见第 5 节）。
- **`normalize_batch_and_arguments()`**：把单条/批量统一成可索引形态，`rid` 在此确定。
- **`_init_req_state`**：为请求在 `rid_to_state` 登记 `ReqState`（含 `asyncio.Event`、`out_list`、时间统计）。`rid` 重复会直接报错：

```3670:3678:python/sglang/srt/managers/tokenizer_manager.py
        for rid, sub_obj, bootstrap_room in items:
            if rid in self.rid_to_state:
                raise ValueError(f"Duplicate request ID detected: {rid}")
            time_stats = APIServerReqTimeStats(disagg_mode=self.disaggregation_mode)
            state = ReqState([], False, asyncio.Event(), sub_obj, time_stats)
            self.rid_to_state[rid] = state
            if self.enable_trace:
                time_stats.init_trace_ctx(rid, bootstrap_room, external_trace_header)
            time_stats.set_created_time(created_time)
```

- **暂停门控**：`is_pause` 为真时挂起，保证与 `pause_generation/continue_generation` 协作。
- **`model_update_lock.reader_lock`**：整个生成期间持读锁，权重更新（warm 更新）持写锁，二者互斥——这就是「压测后 flush kv cache 而不重启服务」这类热更新能安全进行的前提。
- **`_tokenize_one_request`**：确定 `input_ids` 来源（`input_embeds` / `input_ids` / 对 text 分词），多模态请求交给 `mm_processor` 处理得到含占位符的 `input_ids` 与 `mm_inputs`。
- **`_validate_one_request`**：校验输入长度、`input + max_new_tokens` 是否超 `context_len`（超限按 `allow_auto_truncate` 截断或报错），以及 embedding / hidden states / custom logit 等专属字段。
- **`_create_tokenized_object`**：把结果封装成 `TokenizedGenerateReqInput`（`input_ids` 用 `array('q')` 紧凑存储），采样参数做 `normalize` + `verify`：

```1494:1516:python/sglang/srt/managers/tokenizer_manager.py
        sampling_params = self.sampling_params_class(**sampling_kwargs)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify(self.model_config.vocab_size)

        # Build return object
        if isinstance(obj, GenerateReqInput):
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )
            ...
            tokenized_obj = TokenizedGenerateReqInput(
                input_text=input_text,
                input_ids=input_ids_arr,
                mm_inputs=mm_inputs,
                sampling_params=sampling_params,
                ...
```

- **`_send_one_request`**：经 ZMQ PUSH 发往 Scheduler；同时记录 `dispatch_time`，多模态大块特征走共享内存（`wrap_shm_features`）：

```1705:1726:python/sglang/srt/managers/tokenizer_manager.py
    def _send_one_request(
        self,
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
    ):
        # 中译：把单个分词后的请求经 ZMQ 发给调度器。wrap_shm_features 会把大块多模态特征
        prepared_mm_items = []
        dispatched = False
        try:
            prepared_mm_items = self.cuda_vmm_feature_transport.prepare_for_dispatch(
                (tokenized_obj.mm_inputs,)
            )
            tokenized_obj.time_stats.set_api_server_dispatch_time()
            tokenized_obj = wrap_shm_features(tokenized_obj)
            time_stats = tokenized_obj.time_stats
            tokenized_obj.wrap_pickle_fields()
            self._dispatch_to_scheduler(tokenized_obj)
            dispatched = True
            tokenized_obj.time_stats = time_stats
            tokenized_obj.time_stats.set_api_server_dispatch_finish_time()
        finally:
            if not dispatched:
                self.cuda_vmm_feature_transport.cancel_for_dispatch(prepared_mm_items)
```

- 单条请求随后进入 `_wait_one_response`（实际转发到 `_stream_one_response`）等待调度器输出。

### 3.1 等待与 yield：_stream_one_response

这是「请求协程」的等待循环。它阻塞在 `state.event` 上，醒来后把 `out_list` 原子地取空，再根据是否结束决定 yield 还是继续等：

```1853:1886:python/sglang/srt/managers/tokenizer_manager.py
    async def _stream_one_response(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        state: ReqState,
        request: Optional[fastapi.Request] = None,
    ):
        # Not all request types have `stream` (e.g., EmbeddingReqInput). Default to non-streaming.
        # 中译：并非所有请求类型都有 stream 字段（如 EmbeddingReqInput），默认非流式。
        is_stream = getattr(obj, "stream", False)
        while True:
            try:
                await asyncio.wait_for(
                    state.event.wait(), timeout=_REQUEST_STATE_WAIT_TIMEOUT
                )
            except asyncio.TimeoutError:
                if (
                    request is not None
                    and not obj.background
                    and await request.is_disconnected()
                ):
                    # Abort the request for disconnected requests (non-streaming, waiting queue)
                    self.abort_request(obj.rid)
                    # Use exception to kill the whole call stack and asyncio task
                    raise ValueError(
                        f"Request is disconnected from the client side (type 1). Abort request {obj.rid=}"
                    )
                continue

            # Drain all pending outputs atomically.
            # 中译：原子地取走所有待发输出（换上新空列表），并清掉 event，等待下一批。
            out_list = state.out_list
            state.out_list = []
            finished = state.finished
            state.event.clear()
```

要点：

- **超时不是错误**：`asyncio.wait_for` 超时后只做「客户端是否断连」检查，未断连则 `continue` 继续等。
- **客户端断连检测**就在这里：非后台请求若发现 `request.is_disconnected()`，调用 `abort_request(rid)` 并抛 `ValueError` 杀穿调用栈。
- **结束处理**（`tokenizer_manager.py:1918-1947`）：请求完成时记录 `response_sent_to_client_time`、写请求日志与指标、处理 `abort/error` 的 `finish_reason`，最后 `yield out` 并 `break`。
- **流式中间片段**（`tokenizer_manager.py:1949-1957`）直接 `yield out`；**非流式中间片段不 yield**，只借机做断连检测。这就是「非流式只在最后拿到一次结果」的原因——上层 `_handle_non_streaming_request` 的 `__anext__()` 正好等到这个最终 `yield`。

---

## 4. 调度与执行层（Scheduler）

Scheduler 是**每 TP rank 一个的子进程**，把第 ③ 层经 ZMQ 送来的 `TokenizedGenerateReqInput` 真正跑成 token：一个「调度–执行」循环负责「收请求 → 选批 → 前向（GPU）→ 采样 → 出结果」，结束后把新 token 经 Detokenizer 回传（接第 ⑤ 节）。

本层细节较多（含通用主循环、`result_queue` 一拍错位、`FutureMap` 中继，以及 **PD 分离**下 prefill / decode 两侧的循环、bootstrap 与 KV 传输），已**单独成文**：

> **《调度与执行层：Scheduler 详解（PD 分离 + 重叠调度）》**
> 见 `docs/learn/request/scheduler_processing_zh.md`

该文涵盖：Scheduler 的 Mixin 组成与事件循环分派（`dispatch_event_loop`）；非重叠 / 重叠主循环（`event_loop_normal` / `event_loop_overlap`，含 WAR 屏障与延后采样）；选批 `get_next_batch_to_run` / `update_running_batch`（含 OOM 回退）与入队 `_add_request_to_queue`；前向 `run_batch`、结果处理 `process_batch_result`；重叠原理（`schedule_stream` / `forward_stream` / `copy_stream` 与 `FutureMap`）；以及 **PD 分离**的总体架构、prefill 节点（bootstrap、KV 分块发送、在途传输轮询）与 decode 节点（预分配 / 传输 / 等待三级队列、PREBUILT 批）。

---

## 5. 输出回程层

Scheduler/Detokenizer 产出的是 `BatchStrOutput` / `BatchTokenIDOutput` / `BatchEmbeddingOutput`，经 ZMQ 回到 TokenizerManager 的后台事件循环：

```2355:2374:python/sglang/srt/managers/tokenizer_manager.py
    async def handle_loop(self):
        """The event loop that handles requests

        中译：「出口」主事件循环（生产者侧）。持续从 detokenizer 收消息：
              批量输出交给 _handle_batch_output（按 rid 写回各 ReqState 并唤醒等待协程）；
              其余控制类消息交给类型分发器。等待收消息期间临时关闭软看门狗（等消息不算卡死）。
        """
        while True:
            with self.soft_watchdog.disable():
                recv_obj = await async_sock_recv(self.recv_from_detokenizer)
            if isinstance(
                recv_obj,
                (BatchStrOutput, BatchEmbeddingOutput, BatchTokenIDOutput),
            ):
                await self._handle_batch_output(recv_obj)
            else:
                self._result_dispatcher(recv_obj)
            self.last_receive_tstamp = real_time()
            # 中译：喂狗，表明本轮处理正常推进。
            self.soft_watchdog.feed()
```

`_handle_batch_output` 是回程的核心：

1. 对批次里每个 `rid`，用 `rid_to_state.get(rid)` 找到 `ReqState`（找不到且不是健康检查就记 error 跳过）；
2. 构造 `meta_info`（`id`、`finish_reason`、`prompt_tokens`、时间统计、logprobs、`reasoning_tokens`、`completion_tokens`、`cached_tokens`、多模态 token 计数、hidden_states 等）；
3. 累积文本/ids 到 `state`；
4. 把新片段 append 到 `state.out_list`、设置 `state.finished`，**`state.event.set()` 唤醒第 3.1 节的等待协程**。

于是第 3.1 节的 `_stream_one_response` 从 `state.event.wait()` 返回，取走 `out_list` 并向上 `yield`，最终由 ② 层的 `_generate_chat_stream` / `_build_chat_response` 组装为 SSE chunk 或完整 JSON。

请求正常完成后，`rid_to_state` 中的条目会被移除（`logging/metrics` 与 abort 清理在 `_stream_one_response` 的 finished 分支完成），避免状态泄漏。

---

## 6. 关键数据结构与时间统计

| 结构 | 位置 | 作用 |
| --- | --- | --- |
| `ChatCompletionRequest` | `entrypoints/openai/protocol.py:823` | 对外 OpenAI 协议请求模型 |
| `AnthropicMessagesRequest` / `AnthropicMessagesResponse` | `entrypoints/anthropic/protocol.py` | 对外 Anthropic 协议请求/响应模型 |
| `AnthropicStreamEvent`（MessageStart/Delta/Stop、ContentBlock*、Error 等） | `entrypoints/anthropic/protocol.py` | Anthropic SSE 事件模型 |
| `GenerateReqInput` | `managers/io_struct.py:175` | 内部统一请求（② → ③） |
| `TokenizedGenerateReqInput` | `managers/io_struct.py` | 分词后请求（③ → ④，走 ZMQ） |
| `ReqState` | `managers/tokenizer_manager.py:237` | 回程汇聚点：`out_list` + `asyncio.Event` + 时间统计 |
| `StreamChunk` / `ChatCompletionResponse` | `entrypoints/openai/sse_utils.py` / `protocol.py` | 对外响应模型 |

`APIServerReqTimeStats` 贯穿全程，串起 created / tokenize / dispatch / response_sent_to_client 等阶段时间戳，是 P50/P99 与 trace 的数据来源。

---

## 7. 异常与取消路径小结

| 触发点 | 行为 |
| --- | --- |
| 请求体非法（Pydantic） | FastAPI 直接 422，未进入 handler |
| `Content-Type` 非 JSON | `validate_json_request` 报校验错误 |
| `_validate_request` 返回错误字符串 | `create_error_response` 返回对应错误 |
| `_convert_to_internal_request` / 生成中抛 `ValueError` | `serving_base` 统一转 400；流式在首片前抛仍可 400 |
| 上下文超限（`_validate_one_request`） | 非流式 400；流式若已开始则走 SSE 错误 chunk |
| 客户端断连 | `_stream_one_response` 检测后 `abort_request(rid)` 并抛错清理；流式另有 `create_abort_task` 后台清理 |
| Scheduler abort（含 `status_code`） | 流式发错误 chunk；非流式抛 `HTTPException`/`ValueError` |
| Anthropic 请求转换失败（`_convert_to_chat_completion_request`） | 直接返回 Anthropic 风格 400 `invalid_request_error` |
| Anthropic 流式生成中途异常 | `_flush_on_error` 补全 `message_start`、关闭 content block，再发 `ErrorEvent` + `message_stop` |
| Anthropic 上游 5xx | 统一转 `api_error` 且只回通用 message，不泄露原始异常细节 |

---

## 附：一句话总结调用链

```text
【OpenAI 入口】
POST /v1/chat/completions
  → http_server.openai_v1_chat_completions                 (路由)
  → OpenAIServingChat.handle_request                        (serving_base 骨架)
  → _validate_request → _convert_to_internal_request        (校验 + 渲染模板 + 采样参数)
  ┌─ 主干（两入口共用）───────────────────────────────────────┐
  │ → TokenizerManager.generate_request                     (分词 + 分发 + 等待)
  │     → _tokenize_one_request → _send_one_request (ZMQ)   (去程)
  │     → Scheduler.event_loop → run_batch → Detokenizer    (生成)
  │     → handle_loop → _handle_batch_output → event.set()  (回程)
  └──────────────────────────────────────────────────────────┘
  → _generate_chat_stream / _build_chat_response            (组装 SSE / JSON)
  → 客户端收到 data: [DONE] 或完整 ChatCompletionResponse

【Anthropic 入口】
POST /v1/messages
  → http_server.anthropic_v1_messages                       (路由)
  → AnthropicServing.handle_messages
  → _convert_to_chat_completion_request                     (Anthropic → OpenAI 翻译)
  → （复用 OpenAIServingChat 的校验/转换/生成，主干同上）
  → _generate_anthropic_stream / _convert_response          (OpenAI → Anthropic 翻译)
  → 客户端收到 Anthropic 事件流或 Messages 响应
```
