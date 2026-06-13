from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Union

import jinja2
import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from jsonschema import Draft202012Validator, SchemaError

from sglang.srt.entrypoints.openai.encoding_dsv32 import encode_messages
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    DeltaMessage,
    ErrorResponse,
    FunctionResponse,
    LogProbs,
    MessageProcessingResult,
    SglExt,
    ToolCall,
    ToolCallProcessingResult,
    ToolChoice,
    TopLogprob,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai.utils import (
    process_cached_tokens_details_from_ret,
    process_hidden_states_from_ret,
    process_routed_experts_from_ret,
    to_openai_style_logprobs,
)
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.utils import get_json_schema_constraint
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.parser.jinja_template_utils import process_content_for_template_format
from sglang.srt.parser.reasoning_parser import ReasoningParser

if TYPE_CHECKING:
    from sglang.srt.managers.template_manager import TemplateManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


def _extract_max_dynamic_patch(request: ChatCompletionRequest):
    # 从多模态消息中提取图像/视频的 max_dynamic_patch（动态分块上限）。
    # 分别收集 image_url / video_url 中携带的值，最后取最小值作为全局限制。
    img_vals = []
    vid_vals = []
    for msg in request.messages or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            # part 可能是 pydantic 对象或 dict 类型
            if getattr(part, "type", None) == "image_url":
                iu = getattr(part, "image_url", None)
                mdp = getattr(iu, "max_dynamic_patch", None) if iu else None
                if mdp is not None:
                    img_vals.append(int(mdp))
            elif getattr(part, "type", None) == "video_url":
                vu = getattr(part, "video_url", None)
                mdp = getattr(vu, "max_dynamic_patch", None) if vu else None
                if mdp is not None:
                    vid_vals.append(int(mdp))

    # TODO(yuan-luo): 为 image 和 video 实现 per-item 的 max_dynamic_patch
    # 取最小值：多个图/视频取最保守（最小）的分块上限。
    img_max_dynamic_patch = min(img_vals) if img_vals else None
    vid_max_dynamic_patch = min(vid_vals) if vid_vals else None
    return img_max_dynamic_patch, vid_max_dynamic_patch


class OpenAIServingChat(OpenAIServingBase):
    """/v1/chat/completions 请求的处理器。继承自 OpenAIServingBase。"""

    # 类级标志：默认采样参数只打印一次日志。
    _default_sampling_params_logged = False

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager
        # 工具调用解析器类型（如 llama3 / qwen / mistral 等）。
        self.tool_call_parser = self.tokenizer_manager.server_args.tool_call_parser
        # 推理（reasoning）解析器类型。
        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser

        # 从模型的 generation config 获取默认采样参数。
        self.default_sampling_params = (
            self.tokenizer_manager.model_config.get_default_sampling_params()
        )
        if (
            self.default_sampling_params
            and not OpenAIServingChat._default_sampling_params_logged
        ):
            logger.info(
                f"Using default chat sampling params from model generation config: {self.default_sampling_params}",
            )
            OpenAIServingChat._default_sampling_params_logged = True

        # 判断是否为 GPT-OSS 模型（harmony 格式，需保留特殊 token）。
        self.is_gpt_oss = (
            hasattr(self.tokenizer_manager.model_config, "hf_config")
            and hasattr(self.tokenizer_manager.model_config.hf_config, "model_type")
            and self.tokenizer_manager.model_config.hf_config.model_type == "gpt_oss"
        )

        self.use_dpsk_v32_encoding = self._use_dpsk_v32_encoding()

    def _handle_last_assistant_message(
        self,
        messages: List[Dict[str, Any]],
        request: ChatCompletionRequest,
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        处理 continue_final_message 特性：分离最后一条 assistant 消息。

        如果启用了 continue_final_message 且最后一条消息来自 assistant，
        则提取其内容并从消息列表中移除（后续作为生成前缀续写）。
        如果 continue_final_message 为 False 且最后一条消息来自 assistant，
        则把它转换为 user 消息，以确保最后一条消息始终来自 user。

        仅处理纯文本内容（字符串），忽略多模态内容（列表）。

        Args:
            messages: 消息字典列表
            request: 携带 continue_final_message 标志的 ChatCompletionRequest

        Returns:
            元组 (processed_messages, assistant_prefix)
            - processed_messages: 已对最后一条 assistant 消息做妥善处理的消息列表
            - assistant_prefix: 最后一条 assistant 消息的内容（仅字符串），否则为 None
        """
        assistant_prefix = None
        if messages and messages[-1].get("role") == "assistant":
            last_content = messages[-1].get("content")
            # 仅处理字符串内容，忽略多模态内容（列表）
            if isinstance(last_content, str):
                if request.continue_final_message:
                    # 提取内容并移除该 assistant 消息
                    assistant_prefix = last_content
                    messages = messages[:-1]
                else:
                    # 把最后一条 assistant 消息转换为 user 消息
                    messages[-1] = {"role": "user", "content": last_content}
        return messages, assistant_prefix

    def _append_assistant_prefix_to_prompt_ids(
        self, prompt_ids: List[int], assistant_prefix: str
    ) -> List[int]:
        """
        把 assistant 前缀追加到 prompt_ids（用于续写最后一条 assistant 消息）。

        Args:
            prompt_ids: 当前的 prompt token ID 列表
            assistant_prefix: 要追加的 assistant 消息内容

        Returns:
            追加 assistant 前缀后的 prompt_ids
        """
        encoded = self.tokenizer_manager.tokenizer.encode(assistant_prefix)
        # 由于前缀是拼接在已有 prompt 之后，需去掉编码结果开头多出的 BOS token，避免重复。
        if encoded and encoded[0] == self.tokenizer_manager.tokenizer.bos_token_id:
            encoded = encoded[1:]
        return prompt_ids + encoded

    def _use_dpsk_v32_encoding(self) -> bool:
        # 判断是否需要使用 DeepSeek V3.2 的专用编码：
        # 当模型是 DeepseekV3 架构且没有内置 chat_template 时启用。
        has_chat_template = (
            self.tokenizer_manager.tokenizer is not None
            and self.tokenizer_manager.tokenizer.chat_template is not None
        )
        architectures = self.tokenizer_manager.model_config.hf_config.architectures
        is_dpsk_v32 = "DeepseekV3" in architectures[0] if architectures else False
        return not has_chat_template and is_dpsk_v32

    def _request_id_prefix(self) -> str:
        return "chatcmpl-"

    def _validate_request(self, request: ChatCompletionRequest) -> Optional[str]:
        """校验请求合法性（消息非空、工具选择与工具定义、输出长度、json_schema 等）。返回错误信息或 None。"""
        # 消息不能为空。
        if not request.messages:
            return "Messages cannot be empty."

        if (
            isinstance(request.tool_choice, str)
            and request.tool_choice.lower() == "required"
            and not request.tools
        ):
            return "Tools cannot be empty if tool choice is set to required."

        if request.tool_choice is not None and not isinstance(request.tool_choice, str):
            if not request.tools:
                return "Tools cannot be empty if tool choice is set to a specific tool."
            tool_name = request.tool_choice.function.name
            tool_exists = any(tool.function.name == tool_name for tool in request.tools)
            if not tool_exists:
                return f"Tool '{tool_name}' not found in tools list."

        # 校验工具定义：检查每个工具的 parameters 是否为合法的 JSON Schema
        for i, tool in enumerate(request.tools or []):
            if tool.function.parameters is None:
                continue
            try:
                Draft202012Validator.check_schema(tool.function.parameters)
            except SchemaError as e:
                return f"Tool {i} function has invalid 'parameters' schema: {str(e)}"

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

        if request.response_format and request.response_format.type == "json_schema":
            schema = getattr(request.response_format.json_schema, "schema_", None)
            if schema is None:
                return "schema_ is required for json_schema response format request."

        return None

    def _convert_to_internal_request(
        self,
        request: ChatCompletionRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, ChatCompletionRequest]:
        # 把 OpenAI 的 ChatCompletionRequest 转换为内部 GenerateReqInput。
        # 从 chat_template_kwargs 中弹出 reasoning_effort（推理强度）。
        reasoning_effort = (
            request.chat_template_kwargs.pop("reasoning_effort", None)
            if request.chat_template_kwargs
            else None
        )
        # GPT-OSS（harmony）不支持 reasoning_effort=none。
        if self.is_gpt_oss and reasoning_effort == "none":
            raise ValueError(
                f"Harmony does not support reasoning effort {reasoning_effort}"
            )

        if reasoning_effort is not None:
            request.reasoning_effort = reasoning_effort

        # 是否为多模态模型（决定后续传 text 还是 input_ids）。
        is_multimodal = self.tokenizer_manager.model_config.is_multimodal

        # 处理消息并套用聊天模板（得到 prompt / prompt_ids / 多模态数据等）。
        processed_messages = self._process_messages(request, is_multimodal)

        # 构建采样参数（含停止串与工具调用约束）。
        sampling_params = request.to_sampling_params(
            stop=processed_messages.stop,
            model_generation_config=self.default_sampling_params,
            tool_call_constraint=processed_messages.tool_call_constraint,
        )

        # 多模态走 text；纯文本视 prompt_ids 是字符串还是 token id 列表分别传 text/input_ids。
        if is_multimodal:
            prompt_kwargs = {"text": processed_messages.prompt}
        else:
            if isinstance(processed_messages.prompt_ids, str):
                prompt_kwargs = {"text": processed_messages.prompt_ids}
            else:
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}

        # 从请求头提取自定义标签（用于监控/追踪）。
        custom_labels = self.extract_custom_labels(raw_request)

        # 从请求头提取 routed_dp_rank（优先级高于请求体）。
        effective_routed_dp_rank = self.extract_routed_dp_rank_from_header(
            raw_request, request.routed_dp_rank
        )

        # 解析 LoRA adapter（来自 model 参数的 base:adapter 语法或显式 lora_path）。
        lora_path = self._resolve_lora_path(request.model, request.lora_path)
        img_max_dynamic_patch, vid_max_dynamic_patch = _extract_max_dynamic_patch(
            request
        )
        # 组装内部生成请求对象，把 OpenAI 字段映射到 SGLang 内部字段。
        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            image_data=processed_messages.image_data,
            video_data=processed_messages.video_data,
            audio_data=processed_messages.audio_data,
            sampling_params=sampling_params,
            return_logprob=request.logprobs,
            logprob_start_len=-1,
            top_logprobs_num=request.top_logprobs or 0,
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
            rid=request.rid,
            extra_key=self._compute_extra_key(request),
            require_reasoning=self._get_reasoning_from_request(request),
            priority=request.priority,
            routing_key=self.extract_routing_key(raw_request),
            custom_labels=custom_labels,
            custom_logit_processor=request.custom_logit_processor,
            image_max_dynamic_patch=img_max_dynamic_patch,
            video_max_dynamic_patch=vid_max_dynamic_patch,
            max_dynamic_patch=getattr(request, "max_dynamic_patch", None),
        )

        return adapted_request, request

    def _process_messages(
        self, request: ChatCompletionRequest, is_multimodal: bool
    ) -> MessageProcessingResult:
        """处理聊天消息并套用聊天模板，返回含 prompt/多模态数据/工具约束的结果。"""
        # GPT-OSS 模型需保留特殊 token 以供 harmony 解析。
        if self.is_gpt_oss:
            request.skip_special_tokens = False

        # Mistral 的特殊 token 处理补丁。
        self._patch_mistral_skip_special_tokens(request)

        tool_call_constraint = None

        # 若有工具且 tool_choice != none，则准备工具定义并保留特殊 token。
        tools = None
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
            if not isinstance(request.tool_choice, str):
                tools = [
                    item.model_dump()
                    for item in request.tools
                    if item.function.name == request.tool_choice.function.name
                ]
            else:
                tools = [item.model_dump() for item in request.tools]
            if self.tool_call_parser:
                # 通过函数调用解析器获取结构约束（用于约束解码）。
                parser = FunctionCallParser(request.tools, self.tool_call_parser)
                tool_call_constraint = parser.get_structure_constraint(
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                )
            # 对 required 或指定工具的选择，直接用 JSON schema 约束。
            if request.tool_choice == "required" or isinstance(
                request.tool_choice, ToolChoice
            ):
                json_schema = get_json_schema_constraint(
                    request.tools,
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                )
                tool_call_constraint = ("json_schema", json_schema)

        # 选择模板：无自定义模板名时用 Jinja 模板；否则用内置会话模板。
        if self.template_manager.chat_template_name is None:
            result = self._apply_jinja_template(request, tools, is_multimodal)
        else:
            result = self._apply_conversation_template(request, is_multimodal)

        result.tool_call_constraint = tool_call_constraint
        return result

    def _apply_jinja_template(
        self,
        request: ChatCompletionRequest,
        tools: Optional[List[Dict]],
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """套用 HuggingFace tokenizer 自带的 Jinja 聊天模板，把消息渲染为 prompt token。"""
        prompt = ""
        prompt_ids = []
        openai_compatible_messages = []
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        template_content_format = self.template_manager.jinja_template_content_format

        if self.use_dpsk_v32_encoding:
            thinking_mode = (
                "thinking"
                if (request.chat_template_kwargs or {}).get("thinking")
                else "chat"
            )
            messages = request.messages
            messages = [msg.model_dump() for msg in messages]

            for msg in messages:
                if msg.get("content") is None:
                    msg["content"] = ""
                processed_msg = process_content_for_template_format(
                    msg,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                    use_dpsk_v32_encoding=self.use_dpsk_v32_encoding,
                )
                msg.update(processed_msg)

            # 处理 continue_final_message：分离最后一条 assistant 消息
            messages, assistant_prefix = self._handle_last_assistant_message(
                messages, request
            )

            if messages[0]["role"] != "system":
                # 插入一个空的 system prompt，便于渲染工具相关的 system 提示
                messages.insert(0, {"role": "system", "content": ""})
            if request.tools:
                messages[0]["tools"] = [tool.model_dump() for tool in request.tools]
            real_input = encode_messages(messages, thinking_mode=thinking_mode)
            prompt_ids = self.tokenizer_manager.tokenizer.encode(real_input)

            # 若启用了 continue_final_message，则追加 assistant 前缀以续写
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )
        else:
            for message in request.messages:
                if message.content is None:
                    message.content = ""
                msg_dict = message.model_dump()

                # 根据检测到的模板内容格式处理消息内容（提取多模态数据等）
                processed_msg = process_content_for_template_format(
                    msg_dict,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                )

                # 根据 Transformers 官方文档与维护者的说明：assistant 角色消息中
                # tool_calls 的参数（arguments）需要是 dict 而非 JSON 字符串——
                # 这是工具调用聊天模板未来期望的格式。
                # 因此对带 tool_calls 的消息，把（来自 OpenAI 格式的）字符串解析为 dict
                if (
                    processed_msg["role"] == "assistant"
                    and "tool_calls" in processed_msg
                    and isinstance(processed_msg["tool_calls"], list)
                ):
                    for item in processed_msg["tool_calls"]:
                        if "arguments" in item["function"] and isinstance(
                            item["function"]["arguments"], str
                        ):
                            item["function"]["arguments"] = orjson.loads(
                                item["function"]["arguments"]
                            )

                openai_compatible_messages.append(processed_msg)

            # 处理 continue_final_message：分离最后一条 assistant 消息
            openai_compatible_messages, assistant_prefix = (
                self._handle_last_assistant_message(openai_compatible_messages, request)
            )

            extra_template_kwargs = {}
            if request.reasoning_effort is not None:
                extra_template_kwargs["reasoning_effort"] = request.reasoning_effort
            if request.chat_template_kwargs:
                extra_template_kwargs.update(request.chat_template_kwargs)

            try:
                prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
                    openai_compatible_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    tools=tools,
                    return_dict=False,
                    **extra_template_kwargs,
                )
            except Exception as e:
                # 若第一次尝试失败，则改用扁平的「仅 function」格式重试。
                # 某些模板（如 Mistral）期望 tools 不带 OpenAI 的外层包装。
                tools = (
                    [t["function"] if "function" in t else t for t in tools]
                    if tools
                    else None
                )
                try:
                    prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
                        openai_compatible_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        tools=tools,
                        return_dict=False,
                        **extra_template_kwargs,
                    )
                except jinja2.TemplateError as template_error:
                    # 模板错误（如 Jinja 模板中的 raise_exception）应视为客户端错误（400）。
                    raise ValueError(str(template_error)) from template_error

            # 若启用了 continue_final_message，则追加 assistant 前缀以续写
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )

            if is_multimodal:
                prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)

        stop = request.stop
        image_data = image_data if image_data else None
        audio_data = audio_data if audio_data else None
        video_data = video_data if video_data else None
        modalities = modalities if modalities else []
        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    def _apply_conversation_template(
        self,
        request: ChatCompletionRequest,
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """套用 SGLang 内置的会话（conversation）模板渲染 prompt。"""
        prompt = ""
        prompt_ids = []
        conv = generate_chat_conv(request, self.template_manager.chat_template_name)

        # 若需要续写最后一条 assistant 消息，则相应调整会话结构。
        if (
            request.continue_final_message
            and request.messages
            and request.messages[-1].role == "assistant"
        ):
            # 移除自动添加的空 assistant 轮次（若存在）。
            if conv.messages and conv.messages[-1][1] is None:
                conv.messages.pop()
            # 从会话重新生成 prompt。
            prompt = conv.get_prompt()
            # 去掉末尾表示 assistant 结束的 stop token 或分隔符，以便续写。
            if isinstance(conv.stop_str, list):
                for stop_token in conv.stop_str:
                    if prompt.endswith(stop_token):
                        prompt = prompt[: -len(stop_token)]
            elif isinstance(conv.stop_str, str) and prompt.endswith(conv.stop_str):
                prompt = prompt[: -len(conv.stop_str)]
            if conv.sep and prompt.endswith(conv.sep):
                prompt = prompt[: -len(conv.sep)]
            if getattr(conv, "sep2", None) and prompt.endswith(conv.sep2):
                prompt = prompt[: -len(conv.sep2)]
        else:
            prompt = conv.get_prompt()
            # 如需推理且解析器不是 qwen3/glm4（它们内部思考、无需前置 <think>），则手动追加 <think>。
            if self._get_reasoning_from_request(
                request
            ) and self.reasoning_parser not in ["qwen3", "qwen3-thinking", "glm4"]:
                # qwen3 与 glm4 在内部进行思考，无需前置的 <think> token
                prompt += "<think>"  # Note(Xinyuan): 硬编码 thinking token

        image_data = conv.image_data if conv.image_data else None
        video_data = conv.video_data if conv.video_data else None
        audio_data = conv.audio_data if conv.audio_data else None
        modalities = conv.modalities if conv.modalities else []
        stop = copy.copy(conv.stop_str or [] if not request.ignore_eos else [])

        if request.stop:
            if isinstance(request.stop, str):
                stop.append(request.stop)
            else:
                stop.extend(request.stop)

        if not is_multimodal:
            prompt_ids = self.tokenizer_manager.tokenizer.encode(prompt)

        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse]:
        """处理流式（SSE）聊天请求。"""
        generator = self._generate_chat_stream(adapted_request, request, raw_request)

        # 预先拉取第一个 chunk 以触发校验：在返回 HTTP 200 之前发现错误
        # （如上下文超长）时，能返回正常的 400 错误响应，而不是把错误当作 SSE 负载。
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

    async def _generate_chat_stream(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """生成流式聊天响应（逐 chunk yield SSE 文本）。"""
        # 工具调用与推理的解析器（按 index/多路并行分别维护）。
        parser_dict = {}
        reasoning_parser_dict = {}

        # 流式状态跟踪：是否首块、已输出缓冲、已处理 logprob 数、是否有工具调用、结束原因。
        is_firsts = {}
        stream_buffers = {}
        n_prev_tokens = {}
        has_tool_calls = {}
        finish_reasons = {}

        # 用量（token 统计）跟踪。
        prompt_tokens = {}
        completion_tokens = {}
        cached_tokens = {}
        hidden_states = {}
        routed_experts = {}

        stream_started = False
        try:
            # 逐个消费 tokenizer_manager 产出的增量结果。
            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                index = content.get("index", 0)

                prompt_tokens[index] = content["meta_info"].get("prompt_tokens", 0)
                completion_tokens[index] = content["meta_info"].get(
                    "completion_tokens", 0
                )
                cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)
                hidden_states[index] = content["meta_info"].get("hidden_states", None)
                routed_experts[index] = content["meta_info"].get("routed_experts", None)

                # 处理 logprobs：仅输出新增部分的 logprob。
                choice_logprobs = None
                if request.logprobs:
                    n_prev_token = n_prev_tokens.get(index, 0)
                    total_output_logprobs = content["meta_info"][
                        "output_token_logprobs_length"
                    ]
                    if n_prev_token < total_output_logprobs:
                        choice_logprobs = self._process_streaming_logprobs(
                            content, n_prev_token, total_output_logprobs
                        )
                    n_prev_tokens[index] = total_output_logprobs

                finish_reason = content["meta_info"].get("finish_reason", None)
                finish_reason_type = finish_reason["type"] if finish_reason else None

                # 记录每个 index 的结束原因。
                if finish_reason_type:
                    # 若是调度器主动 abort（中止）。
                    if finish_reason_type == "abort":
                        code = finish_reason.get(
                            "status_code", HTTPStatus.INTERNAL_SERVER_ERROR
                        )
                        error = self.create_streaming_error_response(
                            finish_reason.get("message", "Generation aborted."),
                            code.name,
                            code.value,
                        )
                        yield f"data: {error}\n\n"
                        break
                    else:
                        finish_reasons[index] = finish_reason

                # 首块：先发送含 role=assistant 的空内容 delta（符合 OpenAI 流格式）。
                if is_firsts.get(index, True):
                    is_firsts[index] = False
                    delta = DeltaMessage(role="assistant", content="")
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=delta,
                        finish_reason=None,
                        logprobs=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    stream_started = True

                # 计算本次增量 delta = 新文本 - 已缓冲文本。
                stream_buffer = stream_buffers.get(index, "")
                delta = content["text"][len(stream_buffer) :]
                stream_buffers[index] = stream_buffer + delta

                # 处理推理内容：若启用推理解析且要求分离，把 reasoning 部分单独输出为 reasoning_content。
                if self.reasoning_parser and request.separate_reasoning:
                    reasoning_text, delta = self._process_reasoning_stream(
                        index, delta, reasoning_parser_dict, content, request
                    )
                    if reasoning_text:
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(reasoning_content=reasoning_text),
                            finish_reason=None,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )

                        # 若启用了 continuous_usage_stats，在每个 chunk 中附带用量统计
                        if (
                            request.stream_options
                            and request.stream_options.continuous_usage_stats
                        ):
                            chunk.usage = UsageProcessor.calculate_token_usage(
                                prompt_tokens=prompt_tokens.get(index, 0),
                                completion_tokens=completion_tokens.get(index, 0),
                            )

                        yield f"data: {chunk.model_dump_json()}\n\n"

                # 处理工具调用：增量解析 delta，按需输出工具调用 chunk。
                if (
                    request.tool_choice != "none"
                    and request.tools
                    and self.tool_call_parser
                ):
                    async for chunk in self._process_tool_call_stream(
                        index,
                        delta,
                        parser_dict,
                        content,
                        request,
                        has_tool_calls,
                    ):
                        if chunk:
                            yield chunk

                    # 生成结束时，补发尚未输出的工具调用参数。
                    if finish_reason_type is not None and index in parser_dict:
                        parser = parser_dict[index]
                        remaining_chunk = self._check_for_unstreamed_tool_args(
                            parser, content, request, index
                        )
                        if remaining_chunk:
                            yield remaining_chunk

                else:
                    # 普通文本内容（无工具调用）。
                    if delta:
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(content=delta),
                            finish_reason=None,
                            matched_stop=None,
                            logprobs=choice_logprobs,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )

                        # 若启用了 continuous_usage_stats，在每个 chunk 中附带用量统计
                        if (
                            request.stream_options
                            and request.stream_options.continuous_usage_stats
                        ):
                            chunk.usage = UsageProcessor.calculate_token_usage(
                                prompt_tokens=prompt_tokens.get(index, 0),
                                completion_tokens=completion_tokens.get(index, 0),
                            )

                        yield f"data: {chunk.model_dump_json()}\n\n"

            # 为每个完成的 index 发送带 finish_reason 的 chunk。
            for idx, finish_reason_data in finish_reasons.items():
                finish_reason_type = finish_reason_data["type"]

                # 若有工具调用且是自然 stop，则把 finish_reason 改为 tool_calls。
                final_finish_reason = finish_reason_type
                if has_tool_calls.get(idx, False) and finish_reason_type == "stop":
                    final_finish_reason = "tool_calls"

                finish_reason_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"][
                        "id"
                    ],  # NOTE: OpenAI 对所有 index（多路采样）使用相同的 chatcmpl-id
                    created=int(time.time()),
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=idx,
                            delta=DeltaMessage(),
                            finish_reason=final_finish_reason,
                            matched_stop=(
                                finish_reason_data["matched"]
                                if "matched" in finish_reason_data
                                else None
                            ),
                        )
                    ],
                    model=request.model,
                    usage=None,
                )
                yield f"data: {finish_reason_chunk.model_dump_json()}\n\n"

            # 若请求要求返回 hidden states，则发送
            if request.return_hidden_states and hidden_states:
                for index, choice_hidden_states in hidden_states.items():
                    if choice_hidden_states:
                        last_token_hidden_states = (
                            choice_hidden_states[-1]
                            if len(choice_hidden_states) > 1
                            else []
                        )
                        hidden_states_chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=index,
                                    delta=DeltaMessage(
                                        hidden_states=last_token_hidden_states
                                    ),
                                    finish_reason=None,  # hidden states 无需 finish_reason
                                )
                            ],
                            model=request.model,
                        )
                        yield f"data: {hidden_states_chunk.model_dump_json()}\n\n"

            if request.return_routed_experts and routed_experts:
                # 取第一个非 None 的 routed_experts 值
                first_routed_experts = next(
                    (v for v in routed_experts.values() if v is not None), None
                )
                if first_routed_experts is not None:
                    routed_experts_chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[],  # sglext 位于响应级别，不属于某个 choice
                        model=request.model,
                        sglext=SglExt(routed_experts=first_routed_experts),
                    )
                    yield f"data: {routed_experts_chunk.model_dump_json()}\n\n"

            # 额外的 usage chunk：若请求 include_usage，在最后补发累计 token 用量。
            if request.stream_options and request.stream_options.include_usage:
                usage = UsageProcessor.calculate_streaming_usage(
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    n_choices=request.n,
                    enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
                )
                usage_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=int(time.time()),
                    choices=[],  # 按 OpenAI 规范，usage chunk 的 choices 为空数组
                    model=request.model,
                    usage=usage,
                )
                yield f"data: {usage_chunk.model_dump_json()}\n\n"

        except ValueError as e:
            # 若流还未开始，抛出交由上层转为 400；否则以 SSE 错误事件输出。
            if not stream_started:
                raise
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        # OpenAI SSE 结束标志。
        yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[ChatCompletionResponse, ErrorResponse, ORJSONResponse]:
        """处理非流式聊天请求（一次性返回完整响应）。"""
        try:
            # 非流式只取生成器的最终一个结果（包含完整输出）。
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

    def _build_chat_response(
        self,
        request: ChatCompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
    ) -> Union[ChatCompletionResponse, ORJSONResponse]:
        """从生成结果构建完整的 ChatCompletionResponse（逐 choice 组装）。"""
        choices = []

        # 在响应级别构建 sglext（取第一个 ret，因为这些是整个请求级别的）。
        first_ret = ret[0]
        routed_experts = process_routed_experts_from_ret(first_ret, request)
        cached_tokens_details = process_cached_tokens_details_from_ret(
            first_ret, request
        )
        response_sglext = None
        if routed_experts or cached_tokens_details:
            response_sglext = SglExt(
                routed_experts=routed_experts,
                cached_tokens_details=cached_tokens_details,
            )

        # 逐个 choice（多路采样 n>1 时多个）组装。
        for idx, ret_item in enumerate(ret):
            # 处理 logprobs。
            choice_logprobs = None
            if request.logprobs:
                choice_logprobs = self._process_response_logprobs(ret_item)

            # 处理 hidden states。
            hidden_states = process_hidden_states_from_ret(ret_item, request)

            finish_reason = ret_item["meta_info"]["finish_reason"]
            text = ret_item["text"]

            # 处理推理内容：把 reasoning 从正文中分离出来。
            reasoning_text = None
            reasoning_parser = self.reasoning_parser
            if reasoning_parser and request.separate_reasoning:
                is_force_reasoning = (
                    self.template_manager.force_reasoning
                    or self._get_reasoning_from_request(request)
                )
                try:
                    parser = ReasoningParser(
                        model_type=reasoning_parser,
                        stream_reasoning=False,
                        force_reasoning=is_force_reasoning,
                        request=request,
                    )
                    reasoning_text, text = parser.parse_non_stream(text)
                except Exception as e:
                    logger.error(f"Reasoning parsing error: {e}")
                    return self.create_error_response(
                        "Failed to parse reasoning content",
                        err_type="InternalServerError",
                        status_code=500,
                    )

            # 处理工具调用：从文本中解析出工具调用并从正文剔除。
            tool_calls = None
            if (
                request.tool_choice != "none"
                and request.tools
                and self.tool_call_parser
            ):
                history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
                tool_calls, text, finish_reason = self._process_tool_calls(
                    text,
                    request.tools,
                    finish_reason,
                    request.tool_choice,
                    history_tool_calls_cnt,
                )

            choice_data = ChatCompletionResponseChoice(
                index=idx,
                message=ChatMessage(
                    role="assistant",
                    content=text if text else None,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_text if reasoning_text else None,
                ),
                logprobs=choice_logprobs,
                finish_reason=finish_reason["type"] if finish_reason else None,
                matched_stop=(
                    finish_reason["matched"]
                    if finish_reason and "matched" in finish_reason
                    else None
                ),
                hidden_states=hidden_states,
            )
            choices.append(choice_data)

        # 计算 token 用量。
        usage = UsageProcessor.calculate_response_usage(
            ret,
            n_choices=request.n,
            enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
        )

        return ChatCompletionResponse(
            id=ret[0]["meta_info"]["id"],
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
            metadata={"weight_version": ret[0]["meta_info"]["weight_version"]},
            sglext=response_sglext,
        )

    def _process_logprobs_tokens(
        self, logprobs: LogProbs, use_token_index: bool = False
    ) -> List[ChatCompletionTokenLogprob]:
        """流式与非流式共用的辅助方法，把 logprobs 中的 token 转为 OpenAI 结构。

        Args:
            logprobs: 来自模型的 LogProbs 数据
            use_token_index: 非流式为 True（按 token_idx 取完整数据），流式为 False（取 index 0 的已切片数据）
        """
        token_logprobs = []

        for token_idx, (token, logprob) in enumerate(
            zip(logprobs.tokens, logprobs.token_logprobs)
        ):
            token_bytes = list(token.encode("utf-8"))
            top_logprobs = []
            if logprobs.top_logprobs:
                # - 非流式（use_token_index=True）：用 token_idx 索引完整数据
                # - 流式（use_token_index=False）：数据已预先切片，固定取 index 0
                top_logprobs_idx = token_idx if use_token_index else 0
                for top_token, top_logprob in logprobs.top_logprobs[
                    top_logprobs_idx
                ].items():
                    top_token_bytes = list(top_token.encode("utf-8"))
                    top_logprobs.append(
                        TopLogprob(
                            token=top_token,
                            bytes=top_token_bytes,
                            logprob=top_logprob,
                        )
                    )
            token_logprobs.append(
                ChatCompletionTokenLogprob(
                    token=token,
                    bytes=token_bytes,
                    logprob=logprob,
                    top_logprobs=top_logprobs,
                )
            )

        return token_logprobs

    def _process_response_logprobs(self, ret_item: Dict[str, Any]) -> ChoiceLogprobs:
        """处理非流式响应的 logprobs，转为 OpenAI 风格。"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=ret_item["meta_info"]["output_token_logprobs"],
            output_top_logprobs=ret_item["meta_info"].get("output_top_logprobs", None),
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=True)
        return ChoiceLogprobs(content=token_logprobs)

    def _process_tool_call_id(
        self,
        call_item: ToolCallItem,
        history_tool_calls_cnt: int,
    ) -> str:
        """生成唯一的 tool_call_id（Kimi-K2 需用 functions.{name}:{index} 格式）。"""
        if self.tool_call_parser != "kimi_k2":
            # 除 Kimi-K2 外，所有模型用一个简单的 uuid 即可。
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
            return tool_call_id
        else:
            # 对齐 Kimi-K2 格式：functions.{name}:{index}
            # Kimi-K2 允许一条消息中包含多个 tool_calls；SGLang 把 call_item.tool_index 设为该消息内部的*局部*位置。
            # 因此需用 `history_tool_calls_cnt + call_item.tool_index` 修正 index，以保证全局唯一且顺序正确。
            tool_call_id = f"functions.{call_item.name}:{history_tool_calls_cnt+call_item.tool_index}"
            logger.debug(
                f"Process tool call idx, parser: {self.tool_call_parser}, tool_call_id: {tool_call_id}, history_cnt: {history_tool_calls_cnt}"
            )
            return tool_call_id

    def _process_tool_calls(
        self,
        text: str,
        tools: List[Any],
        finish_reason: Dict[str, Any],
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        history_tool_calls_cnt: int = 0,
    ) -> ToolCallProcessingResult:
        """从响应文本中解析工具调用（非流式）。"""

        # 处理 required 或指定工具：此时输出被 JSON schema 约束，直接解析 JSON 数组。
        if tool_choice == "required" or (
            isinstance(tool_choice, ToolChoice) and tool_choice.type == "function"
        ):
            # 既然在处理工具调用，把 finish_reason 设为 tool_calls
            if finish_reason["type"] == "stop":
                finish_reason["type"] = "tool_calls"
                finish_reason["matched"] = None
            try:
                # 对 required 的 tool_choice，预期输出是一个工具调用的 JSON 数组
                tool_call_data = orjson.loads(text)
                tool_calls = []
                for i, tool in enumerate(tool_call_data):
                    # 从 JSON 数据构建 ToolCallItem
                    call_info = ToolCallItem(
                        tool_index=i,  # 用循环下标作为 tool_index
                        name=tool["name"],
                        parameters=json.dumps(tool["parameters"], ensure_ascii=False),
                    )
                    tool_id = self._process_tool_call_id(
                        call_info, history_tool_calls_cnt
                    )
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            index=i,
                            function=FunctionResponse(
                                name=tool["name"],
                                arguments=json.dumps(
                                    tool["parameters"], ensure_ascii=False
                                ),
                            ),
                        )
                    )
                return ToolCallProcessingResult(tool_calls, "", finish_reason)
            except json.JSONDecodeError as e:
                logger.error(f"Tool call parsing error: {e}")
                return ToolCallProcessingResult(None, text, finish_reason)

        # 非约束情况：输出未被 JSON schema 约束，用函数调用解析器从文本中提取。
        parser = FunctionCallParser(tools, self.tool_call_parser)
        if parser.has_tool_call(text):
            if finish_reason["type"] == "stop":
                finish_reason["type"] = "tool_calls"
                finish_reason["matched"] = None
            try:
                text, call_info_list = parser.parse_non_stream(text)
                tool_calls = []
                for call_info in call_info_list:
                    tool_id = self._process_tool_call_id(
                        call_info, history_tool_calls_cnt
                    )
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            index=getattr(call_info, "tool_index", None),
                            function=FunctionResponse(
                                name=call_info.name, arguments=call_info.parameters
                            ),
                        )
                    )
                return ToolCallProcessingResult(tool_calls, text, finish_reason)
            except Exception as e:
                logger.error(f"Tool call parsing error: {e}")
                # 返回解析失败结果，但不让整个请求失败
                return ToolCallProcessingResult(None, text, finish_reason)

        return ToolCallProcessingResult(None, text, finish_reason)

    def _process_streaming_logprobs(
        self,
        content: Dict[str, Any],
        n_prev_token: int,
        total_output_logprobs: int,
    ) -> ChoiceLogprobs:
        """处理流式响应的 logprobs（只取 [n_prev_token, total] 区间的增量）。"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=content["meta_info"]["output_token_logprobs"][
                n_prev_token:total_output_logprobs
            ],
            output_top_logprobs=content["meta_info"].get("output_top_logprobs", [])[
                n_prev_token:total_output_logprobs
            ],
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=False)
        return ChoiceLogprobs(content=token_logprobs)

    def _process_reasoning_stream(
        self,
        index: int,
        delta: str,
        reasoning_parser_dict: Dict[int, ReasoningParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
    ) -> tuple[Optional[str], str]:
        """处理流式响应中的推理内容，返回 (reasoning_text, 剩余正文)。"""
        # 按 index 懒初始化推理解析器。
        if index not in reasoning_parser_dict:
            is_force_reasoning = (
                self.template_manager.force_reasoning
                or self._get_reasoning_from_request(request)
            )
            reasoning_parser_dict[index] = ReasoningParser(
                self.reasoning_parser,
                request.stream_reasoning,
                is_force_reasoning,
                request,
            )
        reasoning_parser = reasoning_parser_dict[index]
        return reasoning_parser.parse_stream_chunk(delta)

    def _get_history_tool_calls_cnt(self, request: ChatCompletionRequest) -> int:
        """统计请求消息历史中已有的工具调用数量。

        NOTE: 该方法仅对在 tool_calls id 中包含「自增的历史工具调用下标」的模型有用，
        例如 kimi-k2。

        Args:
            request: 聊天补全请求对象。

        Returns:
            历史中工具调用的总数；若不适用则返回 0。
        """
        messages = getattr(request, "messages", [])
        idx = 0
        for msg in messages:
            if msg.role == "assistant":
                tool_calls = getattr(msg, "tool_calls", None)
                idx += len(list(tool_calls)) if tool_calls is not None else 0  # noqa
        return idx

    def _patch_mistral_skip_special_tokens(
        self, request: ChatCompletionRequest
    ) -> None:
        """Mistral 用特殊 token（[THINK]/[/THINK]）作为推理标记，
        skip_special_tokens=True 时会被剔除，故需关闭该选项。"""
        if (
            self.reasoning_parser in ["mistral"]
            and request.reasoning_effort is not None
            and request.reasoning_effort != "none"
        ):
            request.skip_special_tokens = False

    def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:
        """判断混合推理模型是否需要开启推理（不同模型默认行为与开关字段不同）。"""
        if not self.reasoning_parser:
            return False
        if self.reasoning_parser in ["deepseek-v3"]:
            # 需显式开启思考的模型（thinking=True）
            return (
                request.chat_template_kwargs is not None
                and request.chat_template_kwargs.get("thinking") is True
            )
        if self.reasoning_parser in ["kimi_k2"]:
            # 默认思考、可通过 thinking=False 关闭的模型
            return (
                not request.chat_template_kwargs
                or request.chat_template_kwargs.get("thinking") is not False
            )
        if self.reasoning_parser in ["qwen3", "glm45", "nemotron_3", "interns1"]:
            # 默认思考、可通过 enable_thinking=False 关闭的模型
            return (
                not request.chat_template_kwargs
                or request.chat_template_kwargs.get("enable_thinking") is not False
            )
        if self.reasoning_parser in ["mimo"]:
            # 需显式开启思考的模型（enable_thinking=True）
            return (
                request.chat_template_kwargs is not None
                and request.chat_template_kwargs.get("enable_thinking") is True
            )
        if self.reasoning_parser in ["mistral"]:
            # Mistral 模型仅当 reasoning_effort 被显式设为非 None/"none" 的值
            # （通常为 "high"）时才进行推理。
            return (
                request.reasoning_effort is not None
                and request.reasoning_effort != "none"
            )
        return True  # 默认开启推理

    async def _process_tool_call_stream(
        self,
        index: int,
        delta: str,
        parser_dict: Dict[int, FunctionCallParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        has_tool_calls: Dict[int, bool],
    ):
        """处理流式响应中的工具调用（增量 yield 正文与工具调用 chunk）。"""
        if index not in parser_dict:
            # required 或指定工具时直接用 JSON 数组解析器。
            if request.tool_choice == "required" or isinstance(
                request.tool_choice, ToolChoice
            ):
                parser_dict[index] = JsonArrayParser()
            else:
                parser_dict[index] = FunctionCallParser(
                    tools=request.tools,
                    tool_call_parser=self.tool_call_parser,
                )

        parser = parser_dict[index]

        # 同时兼容 FunctionCallParser 与 JsonArrayParser 两种解析器。
        if isinstance(parser, JsonArrayParser):
            result = parser.parse_streaming_increment(delta, request.tools)
            normal_text, calls = result.normal_text, result.calls
        else:
            normal_text, calls = parser.parse_stream_chunk(delta)

        # 先输出被工具解析器“退出”的普通文本。
        if normal_text:
            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(content=normal_text),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            # 若启用了 continuous_usage_stats，在每个 chunk 中附带用量统计
            if request.stream_options and request.stream_options.continuous_usage_stats:
                prompt_tokens = content["meta_info"].get("prompt_tokens", 0)
                completion_tokens = content["meta_info"].get("completion_tokens", 0)
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

        # 再输出解析出的工具调用增量。
        history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
        for call_item in calls:
            # 标记该 choice 含有工具调用。
            has_tool_calls[index] = True

            # tool_call_id 每个工具调用只生成一次。
            if call_item.name:
                # 首块：包含 ID 与函数名。
                tool_call_id = self._process_tool_call_id(
                    call_item, history_tool_calls_cnt
                )
                function_name = call_item.name
            else:
                # 后续块：只传参数增量，ID 与 name 为空。
                tool_call_id = None
                function_name = None

            tool_call = ToolCall(
                id=tool_call_id,
                index=call_item.tool_index,
                function=FunctionResponse(
                    name=function_name,
                    arguments=call_item.parameters,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            # 若启用了 continuous_usage_stats，在每个 chunk 中附带用量统计
            if request.stream_options and request.stream_options.continuous_usage_stats:
                prompt_tokens = content["meta_info"].get("prompt_tokens", 0)
                completion_tokens = content["meta_info"].get("completion_tokens", 0)
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

    def _check_for_unstreamed_tool_args(
        self,
        parser: Union[FunctionCallParser, JsonArrayParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        index: int,
    ) -> Optional[str]:
        """生成结束时，检查是否还有未输出的工具调用参数需补发，
        确保即使模型在最后一块才生成完整参数也能被正确输出。"""
        # 获取检测器：来自 FunctionCallParser.detector 或直接是 json 检测器。
        detector = parser.detector if hasattr(parser, "detector") else parser

        # 仅当存在工具调用且检测器已跟踪到数据时才检查
        if (
            not hasattr(detector, "prev_tool_call_arr")
            or not detector.prev_tool_call_arr
        ):
            return None

        if (
            not hasattr(detector, "streamed_args_for_tool")
            or not detector.streamed_args_for_tool
        ):
            return None

        # 取正在处理的最后一个工具调用
        tool_index = len(detector.prev_tool_call_arr) - 1
        if tool_index < 0 or tool_index >= len(detector.streamed_args_for_tool):
            return None

        # 比较期望参数与实际已输出参数
        expected_args = detector.prev_tool_call_arr[tool_index].get("arguments", {})
        expected_call = json.dumps(expected_args, ensure_ascii=False)
        actual_call = detector.streamed_args_for_tool[tool_index]

        # 计算剩余未输出的参数部分。
        remaining_call = (
            expected_call.replace(actual_call, "", 1)
            if actual_call in expected_call
            else ""
        )

        if remaining_call:
            # 用剩余参数构建一个工具调用 chunk
            tool_call = ToolCall(
                id=None,  # 参数增量块不携带 ID
                index=tool_index,
                function=FunctionResponse(
                    name=None,  # 参数增量块不携带 name
                    arguments=remaining_call,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,  # 该 chunk 不发送 finish_reason
            )

            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            return f"data: {chunk.model_dump_json()}\n\n"

        return None
