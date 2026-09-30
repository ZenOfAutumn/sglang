# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pydantic models for OpenAI API protocol"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TypeAlias,
    Union,
    get_args,
    runtime_checkable,
)

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextConfig,
)
from openai.types.responses.response import ToolChoice
from openai.types.responses.response_format_text_json_schema_config import (
    ResponseFormatTextJSONSchemaConfig,
)
from openai.types.shared.response_format_json_object import ResponseFormatJSONObject
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from typing_extensions import Literal

try:
    from xgrammar import StructuralTag
except:
    StructuralTag = Any

from sglang.utils import convert_json_schema_to_str

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "default"


class ModelCard(BaseModel):
    """Model cards."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "sglang"
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None


class ModelList(BaseModel):
    """Model list consists of model cards."""

    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


@runtime_checkable
class ParsedResponseFields(Protocol):
    """Protocol for parsed response fields from custom renderers."""

    content: Optional[str]
    tool_calls: Optional[List[Dict]]
    reasoning_content: Optional[str]


class ResponseParserProtocol(Protocol):
    """Protocol for custom response parsers.

    Implementations parse model output tokens into structured OpenAI response fields.
    """

    def parse_response(
        self, output_ids: List[int]
    ) -> Union[ParsedResponseFields, ErrorResponse]:
        """Parse complete response from output token IDs."""
        ...

    def build_streaming_sse_chunks(
        self,
        output_ids: List[int],
        index: int,
        chunk_id: str,
        model: str,
        usage: Optional[Dict],
    ) -> Tuple[List[str], bool, Optional[str]]:
        """Parse streaming tokens and build SSE chunks.

        Returns: (sse_chunks, has_tool_calls, error_message)
        """
        ...


class LogProbs(BaseModel):
    """legacy/文本补全风格的 logprobs 结构（/v1/completions 返回）。"""

    text_offset: List[int] = Field(default_factory=list)  # 每个 token 在原文中的字符偏移
    token_logprobs: List[Optional[float]] = Field(default_factory=list)  # 每个生成 token 的 log 概率
    tokens: List[str] = Field(default_factory=list)  # 生成的 token 文本列表
    top_logprobs: List[Optional[Dict[str, float]]] = Field(
        default_factory=list
    )  # 每个位置上 top-k 候选 token 及其 log 概率


class TopLogprob(BaseModel):
    """某个位置上候选 token 的 logprob（chat 风格 logprobs 用）。"""

    token: str  # 候选 token 的文本
    bytes: List[int]  # token 对应的 UTF-8 字节序列（便于还原非 ASCII 字符）
    logprob: float  # 该候选 token 的 log 概率


class ChatCompletionTokenLogprob(BaseModel):
    """chat 风格 logprobs：单个生成 token 及其 top-k 候选（对齐 OpenAI 格式）。"""

    token: str  # 实际选中的 token 文本
    bytes: List[int]  # 实际选中 token 的 UTF-8 字节序列
    logprob: float  # 实际选中 token 的 log 概率
    top_logprobs: List[TopLogprob]  # 该位置上 top-k 候选 token 及其 log 概率


class ChoiceLogprobs(BaseModel):
    # 用于 /v1/chat/completions 响应中的 logprobs 字段
    content: List[ChatCompletionTokenLogprob]  # 逐生成 token 的 logprob 明细


class CachedTokensDetails(BaseModel):
    """命中的缓存 token 按来源的明细拆分。"""

    device: int = 0  # 来自 device 缓存（GPU）的 token 数
    host: int = 0  # 来自 host 缓存（CPU 内存）的 token 数
    # 以下 L3 存储字段仅在开启 storage backend 时出现
    storage: Optional[int] = None  # 来自 L3 存储后端的 token 数
    storage_backend: Optional[str] = None  # 使用的存储后端类型

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        # 移除 None 字段，避免 L3 未开启时响应里出现冗余字段
        if self.storage is None:
            data.pop("storage", None)
        if self.storage_backend is None:
            data.pop("storage_backend", None)
        return data


class PromptTokensDetails(BaseModel):
    """输入（prompt）token 的明细。"""

    cached_tokens: int = 0  # 命中前缀缓存的 token 数
    # 多模态 prompt token 数（仅在 prompt 中存在对应模态时填充）
    image_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    video_tokens: Optional[int] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        # 不存在的多模态字段直接丢弃，使纯文本/纯缓存场景仍保持 {"cached_tokens": N} 的原始结构
        for key in ("image_tokens", "audio_tokens", "video_tokens"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class UsageInfo(BaseModel):
    """token 用量统计（chat/completion 响应通用）。"""

    prompt_tokens: int = 0  # 输入（prompt）token 数
    total_tokens: int = 0  # 总 token 数 = prompt_tokens + completion_tokens
    completion_tokens: Optional[int] = 0  # 输出（生成）token 数
    # 开启 --enable-cache-report 时，返回命中缓存的 token 明细
    prompt_tokens_details: Optional[PromptTokensDetails] = None
    reasoning_tokens: Optional[int] = 0  # 输出中属于思维链（reasoning）的 token 数


class StreamOptions(BaseModel):
    """流式请求的选项（chat/completion 的 stream_options）。"""

    include_usage: Optional[bool] = False  # 是否在最后一个 chunk 中附带 usage 统计
    continuous_usage_stats: Optional[bool] = False  # 是否每个 chunk 都持续带上累计 usage 统计


class JsonSchemaResponseFormat(BaseModel):
    """response_format={"type": "json_schema"} 时的结构化输出约束。"""

    name: str  # 该 schema 的名字
    description: Optional[str] = None  # schema 描述
    # 用 alias 规避与 pydantic BaseModel.schema 的命名冲突
    schema_: Optional[Dict[str, object]] = Field(alias="schema", default=None)
    # The OpenAI wire contract accepts JSON booleans only; StrictBool rejects
    # the values lax pydantic would coerce ("yes", "on", 0, 1, ...), matching
    # OpenAI's 422 behavior. Omitted (None) keeps its meaning.
    strict: Optional[StrictBool] = None


class ResponseFormat(BaseModel):
    """输出格式约束：纯文本 / JSON 对象 / 指定 JSON Schema。"""

    type: Literal["text", "json_object", "json_schema"]  # 输出格式类型
    json_schema: Optional[JsonSchemaResponseFormat] = None  # type=json_schema 时的具体 schema


class StructuresResponseFormat(BaseModel):
    """结构化输出模板：begin/schema/end 三部分拼成完整约束。"""

    begin: str  # 起始片段
    schema_: Optional[Dict[str, object]] = Field(alias="schema", default=None)  # 中间的 JSON Schema
    end: str  # 结束片段


# NOTE(dark): 保留该类以兼容旧版调用
class LegacyStructuralTagResponseFormat(BaseModel):
    """旧版结构化标签 response_format（保留向后兼容）。"""

    type: Literal["structural_tag"]  # 类型，固定为 structural_tag
    structures: List[StructuresResponseFormat]  # 可用的结构模板列表
    triggers: List[str]  # 触发各结构的字符串
    at_least_one: bool = False  # 是否要求至少匹配到一个结构


StructuralTagResponseFormat: TypeAlias = Union[
    LegacyStructuralTagResponseFormat, StructuralTag
]

ToolCallConstraint: TypeAlias = Union[
    Tuple[Literal["structural_tag"], StructuralTagResponseFormat],
    Tuple[Literal["json_schema"], Any],  # json_schema can be dict/str/None
]


class FileRequest(BaseModel):
    # https://platform.openai.com/docs/api-reference/files/create
    file: bytes  # The File object (not file name) to be uploaded
    purpose: str = (
        "batch"  # The intended purpose of the uploaded file, default is "batch"
    )


class FileResponse(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str


class FileDeleteResponse(BaseModel):
    id: str
    object: str = "file"
    deleted: bool


class BatchRequest(BaseModel):
    input_file_id: (
        str  # The ID of an uploaded file that contains requests for the new batch
    )
    endpoint: str  # The endpoint to be used for all requests in the batch
    completion_window: str  # The time frame within which the batch should be processed
    metadata: Optional[dict] = None  # Optional custom metadata for the batch


class BatchResponse(BaseModel):
    id: str
    object: str = "batch"
    endpoint: str
    errors: Optional[dict] = None
    input_file_id: str
    completion_window: str
    status: str = "validating"
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    created_at: int
    in_progress_at: Optional[int] = None
    expires_at: Optional[int] = None
    finalizing_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    expired_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    request_counts: Optional[dict] = None
    metadata: Optional[dict] = None


def _migrate_deprecated_dp_rank(values: dict) -> dict:
    if isinstance(values, dict) and values.get("data_parallel_rank") is not None:
        import warnings

        warnings.warn(
            "'data_parallel_rank' is deprecated, use 'routed_dp_rank' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if values.get("routed_dp_rank") is None:
            values["routed_dp_rank"] = values["data_parallel_rank"]
    return values


class CompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="Model name. Supports LoRA adapters via 'base-model:adapter-name' syntax.",
    )
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: bool = False
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: int = 16
    n: int = 1
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: float = 1.0
    top_p: float = 1.0
    user: Optional[str] = None
    return_hidden_states: Union[bool, Literal["last"]] = False
    return_routed_experts: bool = False
    routed_experts_start_len: int = 0
    return_cached_tokens_details: bool = False
    return_spec_tokens_details: bool = False
    return_token_ids: bool = False

    # Extra parameters for SRT backend only and will be ignored by OpenAI models.
    top_k: int = -1
    min_p: float = 0.0
    min_tokens: int = 0
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    repetition_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = None
    stop_regex: Optional[Union[str, List[str]]] = None
    no_stop_trim: bool = False
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    session_id: Optional[str] = None
    session_params: Optional[Dict] = None
    response_format: Optional[Union[ResponseFormat, StructuralTagResponseFormat]] = None
    custom_params: Optional[Dict] = None
    custom_logit_processor: Optional[str] = None

    images_config: Optional[Dict] = None

    # For PD disaggregation
    bootstrap_host: Optional[Union[List[str], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[int], int]] = None

    # For DP routing — external router assigns a specific DP worker
    routed_dp_rank: Optional[int] = None
    # For PD disagg — hint telling decode which prefill DP worker has the KV cache
    disagg_prefill_dp_rank: Optional[int] = None
    # Deprecated: use routed_dp_rank instead
    data_parallel_rank: Optional[int] = None

    # For request id
    rid: Optional[Union[List[str], str]] = None
    # Extra key for caller-defined request classification
    extra_key: Optional[Union[List[str], str]] = None
    # Cache salt for request caching
    cache_salt: Optional[Union[List[str], str]] = None
    # Priority for the request
    priority: Optional[int] = None

    # For custom metric labels
    custom_labels: Optional[Dict[str, str]] = None

    @model_validator(mode="before")
    @classmethod
    def _handle_deprecated_dp_rank(cls, values):
        return _migrate_deprecated_dp_rank(values)

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class SpecTokensDetails(BaseModel):
    """单次请求的投机解码（speculative decoding）统计。"""

    spec_accept_rate: float = 0.0  # 投机解码接受率
    spec_accept_length: float = 0.0  # 平均每次接受的 token 数
    spec_cap_length: float = 0.0  # 平均每次上限内的接受长度
    spec_block_accept_length: float = 0.0  # 块粒度下的平均接受长度
    spec_num_correct_drafts: int = 0  # 被验证正确的 draft token 数
    spec_num_proposed_drafts: int = 0  # 提议的 draft token 数
    spec_verify_ct: int = 0  # 验证（verify）调用次数
    spec_correct_drafts_histogram: List[int] = Field(default_factory=list)  # 正确 draft 数的直方图
    spec_cap_lens_histogram: List[int] = Field(default_factory=list)  # 接受长度上限的直方图


class SglExt(BaseModel):
    """OpenAI 兼容响应中的 SGLang 扩展字段容器。

    后续凡是对 OpenAI 兼容响应对象的 SGLang 专有扩展，都应新增到这里，
    而不要直接挂在 choice 对象上。
    """

    routed_experts: Optional[str] = None  # 每个 token 路由到的专家（MoE）
    cached_tokens_details: Optional[CachedTokensDetails] = None  # 缓存命中明细
    spec_tokens_details: Optional[Union[SpecTokensDetails, List[SpecTokensDetails]]] = (
        None
    )  # 投机解码统计（按候选/请求维度）

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        # 移除 None 字段，保持响应简洁
        return {k: v for k, v in data.items() if v is not None}


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter", "abort"]] = None
    matched_stop: Union[None, int, str] = None
    hidden_states: Optional[object] = None
    token_ids: Optional[List[int]] = None
    prompt_token_ids: Optional[List[int]] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.hidden_states is None:
            data.pop("hidden_states", None)
        if self.token_ids is None:
            data.pop("token_ids", None)
        if self.prompt_token_ids is None:
            data.pop("prompt_token_ids", None)
        return data


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo
    metadata: Optional[Dict[str, Any]] = None
    sglext: Optional[SglExt] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.sglext is None:
            data.pop("sglext", None)
        return data


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter", "abort"]] = None
    matched_stop: Union[None, int, str] = None
    hidden_states: Optional[object] = None
    token_ids: Optional[List[int]] = None
    prompt_token_ids: Optional[List[int]] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.hidden_states is None:
            data.pop("hidden_states", None)
        if self.token_ids is None:
            data.pop("token_ids", None)
        if self.prompt_token_ids is None:
            data.pop("prompt_token_ids", None)
        return data


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None
    sglext: Optional[SglExt] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.sglext is None:
            data.pop("sglext", None)
        return data


class ChatCompletionMessageContentTextPart(BaseModel):
    """多模态消息内容块：纯文本部分。"""

    type: Literal["text"]  # 内容块类型，固定为 "text"
    text: str  # 该块的文本内容


class ChatCompletionMessageContentThinkingPart(BaseModel):
    """多模态消息内容块：思维链（thinking/reasoning）部分，仅允许出现在 assistant 消息中。"""

    type: Literal["thinking", "reasoning"]  # 类型标识，兼容 thinking 与 reasoning 两种叫法
    thinking: Optional[str] = None  # 思维链内容（与 text 二选一）
    text: Optional[str] = None  # 兼容字段：部分模型把思维链放在 text 中

    @model_validator(mode="after")
    def validate_payload(self):
        # 校验：thinking 与 text 必须且只能提供一个
        if (self.thinking is None) == (self.text is None):
            raise ValueError(
                "thinking parts require exactly one of 'thinking' or 'text'"
            )
        return self


class ChatCompletionMessageContentImageURL(BaseModel):
    """图片输入：URL 或 base64 data URI，附带清晰度与切图控制。"""

    url: str  # 图片地址（http(s) URL 或 data:image/...;base64,...）
    detail: Optional[Literal["auto", "low", "high"]] = "auto"  # OpenAI 的图片清晰度档位
    max_dynamic_patch: Optional[int] = None  # SGLang 扩展：动态切图（patch）数量的上限
    min_dynamic_patch: Optional[int] = None  # SGLang 扩展：动态切图（patch）数量的下限
    content_hash: Optional[str] = None  # SGLang 扩展：图片内容哈希，用于多模态缓存命中判定

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: Optional[str]) -> Optional[str]:
        # 解析并校验 content_hash 的格式（非法格式会被拒绝/规范化）
        from sglang.srt.multimodal.cache import parse_content_hash

        return parse_content_hash(value)


class ChatCompletionMessageContentVideoURL(BaseModel):
    """视频输入：URL 或 base64 data URI，附带切图控制。"""

    url: str  # 视频地址
    max_dynamic_patch: Optional[int] = None  # 动态抽帧/切图数量的上限
    min_dynamic_patch: Optional[int] = None  # 动态抽帧/切图数量的下限


class ChatCompletionMessageContentAudioURL(BaseModel):
    """音频输入：URL 或 base64 data URI。"""

    url: str  # 音频地址


class ChatCompletionMessageContentImagePart(BaseModel):
    """多模态消息内容块：图片。"""

    type: Literal["image_url"]  # 内容块类型，固定为 "image_url"
    image_url: ChatCompletionMessageContentImageURL  # 图片地址及其参数
    modalities: Optional[Literal["image", "multi-images", "video"]] = "image"  # 图片承载的模态语义


class ChatCompletionMessageContentVideoPart(BaseModel):
    """多模态消息内容块：视频。"""

    type: Literal["video_url"]  # 内容块类型，固定为 "video_url"
    video_url: ChatCompletionMessageContentVideoURL  # 视频地址及其参数


class ChatCompletionMessageContentInputAudio(BaseModel):
    """内联（base64）音频输入，对应 OpenAI 的 input_audio。"""

    data: str  # base64 编码后的音频数据
    format: Literal["wav", "mp3"]  # 音频格式


_AUDIO_FORMAT_TO_MIME_TYPE = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
}  # 音频格式到 MIME type 的映射，用于拼 data URI


class ChatCompletionMessageContentAudioURLPart(BaseModel):
    """多模态消息内容块：按引用传入的音频（音频 URL / data URI 统一形式）。"""

    type: Literal["audio_url"]  # 内容块类型，固定为 "audio_url"
    audio_url: ChatCompletionMessageContentAudioURL  # 音频地址


class ChatCompletionMessageContentAudioInlinePart(BaseModel):
    """多模态消息内容块：内联 base64 音频（OpenAI 的 input_audio 形式）。"""

    type: Literal["input_audio"]  # 内容块类型，固定为 "input_audio"
    input_audio: ChatCompletionMessageContentInputAudio  # 内联音频数据


def _to_audio_url_part(
    part: Union[
        ChatCompletionMessageContentAudioURLPart,
        ChatCompletionMessageContentAudioInlinePart,
    ],
) -> ChatCompletionMessageContentAudioURLPart:
    # 校验辅助函数：把内联 base64 音频统一转换成 audio_url（data URI）形式
    if isinstance(part, ChatCompletionMessageContentAudioURLPart):
        return part

    audio = part.input_audio  # 内联音频：data + format -> data URI
    return ChatCompletionMessageContentAudioURLPart(
        type="audio_url",
        audio_url=ChatCompletionMessageContentAudioURL(
            url=f"data:{_AUDIO_FORMAT_TO_MIME_TYPE[audio.format]};base64,{audio.data}"
        ),
    )


# 音频既可按引用以 `audio_url` 传入（URL 或 data URI），
# 也可按 OpenAI 的 `input_audio` 内联 base64 传入；
# 内联形式在字段校验时会被自动转换成等价的 data URI。
ChatCompletionMessageContentAudioPart = Annotated[
    Union[
        ChatCompletionMessageContentAudioURLPart,
        ChatCompletionMessageContentAudioInlinePart,
    ],
    AfterValidator(_to_audio_url_part),
]


class ChatCompletionMessageContentToolReferenceBlock(BaseModel):
    """工具引用块（GLM 扩展）：按名字引用已有工具的 schema，由 chat template 内联展开。"""

    # GLM 特有扩展，配合 `defer_loading` 工具使用：chat template 会根据
    # `tools[*].function.name == tr.name` 找到对应工具，并在当前轮把其 schema
    # 内联渲染出来。不属于任何 OpenAI API；放在这里是为了让 Pydantic 能通过
    # Chat Completions 路径接受这类 content（Anthropic 端点会在转发前把其
    # `tool_name` 字段翻译成 `name`）。
    type: Literal["tool_reference"]  # 内容块类型，固定为 "tool_reference"
    name: str  # 被引用的工具名


ChatCompletionMessageContentPart = Union[
    ChatCompletionMessageContentTextPart,  # 文本
    ChatCompletionMessageContentThinkingPart,  # 思维链
    ChatCompletionMessageContentImagePart,  # 图片
    ChatCompletionMessageContentVideoPart,  # 视频
    ChatCompletionMessageContentAudioPart,  # 音频（URL 或内联）
    ChatCompletionMessageContentToolReferenceBlock,  # 工具引用（GLM 扩展）
]  # 一条消息 content 可为字符串，也可为上述内容块组成的列表

# 多模态重排（如 Qwen3-VL-Reranker）的内容类型：既可以是纯字符串（纯文本），
# 也可以是上述多模态内容块组成的列表
RerankContentPart = Union[
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageContentVideoPart,
]
RerankContent = Union[str, List[RerankContentPart]]  # 重排输入内容：纯文本或文本/图片/视频内容块列表


class FunctionResponse(BaseModel):
    """工具调用的函数信息（响应侧）。"""

    name: Optional[str] = None  # 函数名
    arguments: Optional[str | Dict[str, Any]] = None  # 函数入参，JSON 字符串或已解析的 dict


class ToolCall(BaseModel):
    """一次工具（函数）调用的完整描述，流式与非流式共用。"""

    id: Optional[str] = None  # 调用 id，用于把 tool 结果回填到对应调用
    index: Optional[int] = None  # 在同一条消息的多次调用中的序号（流式拼接用）
    type: Literal["function"] = "function"  # 调用类型，目前仅支持 function
    function: FunctionResponse  # 被调用的函数及其入参


_GenericMessageRole = Literal[
    "system", "assistant", "tool", "function", "developer", "latest_reminder"
]  # 通用消息角色集合（除 user 外的角色，user 有单独定义）
_GENERIC_MESSAGE_ROLES: Tuple[str, ...] = get_args(_GenericMessageRole)


class ChatCompletionMessageGenericParam(BaseModel):
    """通用消息（system/assistant/tool/developer 等，除 user 外的角色）。"""

    role: _GenericMessageRole  # 消息角色
    content: Union[str, List[ChatCompletionMessageContentPart], None] = Field(
        default=None
    )  # 消息内容：字符串或多模态内容块列表，允许为空
    tool_call_id: Optional[str] = None  # role=tool 时，对应的 tool_call id
    name: Optional[str] = None  # 参与者名字（如 function 模式下指定函数名）
    reasoning_content: Optional[str] = None  # 历史 assistant 消息的思维链内容
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])  # assistant 消息中产生的工具调用
    tools: Optional[List[Tool]] = Field(default=None, examples=[None])  # 消息级工具定义（system/developer 消息可用）

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, v):
        # 角色名大小写不敏感：统一转小写并校验合法性
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower not in _GENERIC_MESSAGE_ROLES:
                allowed = ", ".join(repr(r) for r in _GENERIC_MESSAGE_ROLES)
                raise ValueError(f"'role' must be one of {allowed} (case-insensitive).")
            return v_lower
        raise ValueError("'role' must be a string")

    @model_validator(mode="after")
    def validate_thinking_parts_role(self):
        # 校验：thinking 内容块只能出现在 assistant 消息中
        if self.role != "assistant" and isinstance(self.content, list):
            for part in self.content:
                if isinstance(part, ChatCompletionMessageContentThinkingPart):
                    raise ValueError(
                        "thinking content parts are only valid in assistant messages"
                    )
        return self


class ChatCompletionMessageUserParam(BaseModel):
    """user 角色消息，content 必填。"""

    role: Literal["user"]  # 消息角色，固定为 user
    content: Union[str, List[ChatCompletionMessageContentPart]]  # 用户消息内容（文本或多模态）

    @model_validator(mode="after")
    def validate_thinking_parts_role(self):
        # 校验：user 消息中不允许出现 thinking 内容块
        if isinstance(self.content, list):
            for part in self.content:
                if isinstance(part, ChatCompletionMessageContentThinkingPart):
                    raise ValueError(
                        "thinking content parts are only valid in assistant messages"
                    )
        return self


# 请求中 messages 元素的类型：通用消息或 user 消息
ChatCompletionMessageParam = Union[
    ChatCompletionMessageGenericParam, ChatCompletionMessageUserParam
]


class Function(BaseModel):
    """函数（工具）描述，供模型判断何时以及如何调用。"""

    description: Optional[str] = Field(default=None, examples=[None])  # 函数功能描述
    name: str  # 函数名
    parameters: Optional[object] = None  # 入参的 JSON Schema
    strict: bool = False  # 是否开启严格 schema 模式（严格遵循 parameters）
    defer_loading: Optional[bool] = None  # 是否延迟加载该工具（按需展开 schema）

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        # defer_loading 为 None 时不输出该字段
        if self.defer_loading is None:
            data.pop("defer_loading", None)
        return data


class Tool(BaseModel):
    """工具包装体：type + function。"""

    type: str = Field(default="function", examples=["function"])  # 工具类型，目前为 function
    function: Function  # 具体的函数定义
    defer_loading: Optional[bool] = None  # 是否延迟加载（会向下透传到 function）

    @model_validator(mode="after")
    def _propagate_defer_loading(self) -> Tool:
        # 把外层的 defer_loading 透传到内层 function（内层未显式设置时）
        if self.defer_loading is not None and self.function.defer_loading is None:
            self.function.defer_loading = self.defer_loading
        return self


# Tool is defined after the message params that reference it, so the forward
# reference has to be resolved explicitly.
ChatCompletionMessageGenericParam.model_rebuild()


class ToolChoiceFuncName(BaseModel):
    """指定工具时使用的函数名包装。"""

    name: Optional[str] = None  # 要强制调用的函数名


class ToolChoice(BaseModel):
    """tool_choice 的强指定形式：指定具体某个函数。"""

    function: ToolChoiceFuncName  # 目标函数
    type: Literal["function"] = Field(default="function", examples=["function"])  # 工具类型


# OpenAI 规范中 reasoning effort 的字符串档位（当前 Responses/Chat API）：
# none/minimal/low/medium/high/xhigh/max，/v1/responses 直接原样使用。
ReasoningEffortTier = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max"
]
# /v1/responses 的强类型流事件会额外校验 OpenAI 更窄的 ``Reasoning.effort``；
# 若回显超出该集合的档位，会导致流中断。
ECHOABLE_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})
# Chat Completions 与 /v1/tokenize 还额外接受 [0.0, 0.99] 的细粒度浮点数，
# 这是 sglang 扩展（不属于 OpenAI schema，因此 /v1/responses 仅保留字符串档位）。
# 统一在一处定义，避免多个接口之间的取值漂移。
ReasoningEffortType = Optional[
    Union[
        ReasoningEffortTier,
        Annotated[float, Field(ge=0.0, le=0.99, allow_inf_nan=False)],  # sglang 扩展：细粒度推理强度浮点值
    ]
]


def _has_message_level_tools(messages: Any) -> bool:
    # 判断 messages 中是否存在“消息级”的工具定义（system/developer 消息携带 tools）
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and message["role"].lower() in ("system", "developer")
        and bool(message.get("tools"))
        for message in messages
    )


class ChatCompletionRequest(BaseModel):
    """OpenAI 兼容的 /v1/chat/completions 请求体。

    字段顺序基本对齐 OpenAI 官方文档；除标准字段外还包含大量 SGLang 扩展字段
    （采样控制、调度/PD 分离、多模态、调试返回信息等）。
    """

    # 按照 OpenAI 官方 API 文档字段顺序排列
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[ChatCompletionMessageParam]  # 对话消息列表（必填），含 system/user/assistant/tool 等角色
    model: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="Model name. Supports LoRA adapters via 'base-model:adapter-name' syntax.",
    )  # 模型名；支持 '基础模型:adapter名' 语法指定 LoRA 适配器
    frequency_penalty: float = 0.0  # 频率惩罚：按 token 出现次数线性惩罚，抑制重复
    logit_bias: Optional[Dict[str, float]] = None  # 对指定 token id 的 logit 偏置
    logprobs: bool = False  # 是否返回输出 token 的 log 概率
    top_logprobs: Optional[int] = None  # logprobs=true 时返回每个位置的 top-k 候选数
    max_tokens: Optional[int] = Field(
        default=None,
        deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
        description="The maximum number of tokens that can be generated in the chat completion. ",
    )  # 已废弃：请改用 max_completion_tokens
    max_completion_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of completion tokens for a chat completion request, "
        "including visible output tokens and reasoning tokens. Input tokens are not included. ",
    )  # 本次最多生成的 token 数（含思维链 token，不含输入）
    n: int = 1  # 为每条消息生成多少个候选回复
    presence_penalty: float = 0.0  # 存在惩罚：只要 token 已出现过就惩罚，鼓励话题多样性
    response_format: Optional[Union[ResponseFormat, StructuralTagResponseFormat]] = None  # 输出格式约束（文本/JSON/结构化标签）
    seed: Optional[int] = None  # 随机种子，用于采样可复现
    stop: Optional[Union[str, List[str]]] = None  # 停止字符串（生成到该串时停止）
    stream: bool = False  # 是否流式返回（SSE）
    stream_options: Optional[StreamOptions] = None  # 流式选项（是否返回 usage 等）
    temperature: Optional[float] = None  # 采样温度，越大越随机（None 时回落模型默认/全局默认）
    top_p: Optional[float] = None  # 核采样阈值（nucleus sampling）
    user: Optional[str] = None  # 终端用户标识，便于 OpenAI 侧统计与风控
    tools: Optional[List[Tool]] = Field(default=None, examples=[None])  # 可供模型调用的工具（函数）列表
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = Field(
        default="auto", examples=["none"]
    )  # noqa  # 工具选择策略：auto/required/none 或指定某个工具
    parallel_tool_calls: bool = True  # 是否允许模型并行发起多个工具调用
    return_hidden_states: Union[bool, Literal["last"]] = False  # 调试：返回 hidden states（True 全量 / "last" 最后一层）
    return_routed_experts: bool = False  # 调试：返回每个 token 路由到的专家（MoE）
    routed_experts_start_len: int = 0  # 调试：从第几个 token 位置开始记录 routed experts
    return_cached_tokens_details: bool = False  # 调试：返回命中前缀缓存的 token 明细
    return_spec_tokens_details: bool = False  # 调试：返回投机解码（speculative）的 token 明细
    return_prompt_token_ids: bool = False  # 调试：返回输入的 prompt token ids
    return_token_ids: bool = False  # 调试：返回输出的 token ids
    return_meta_info: bool = False  # 调试：返回底层调度/采样的 meta 信息
    return_sampling_mask: bool = False  # 调试：返回采样掩码（如约束解码时被屏蔽的 token）
    reasoning_effort: ReasoningEffortType = Field(
        default=None,
        description="Constrains effort on reasoning for reasoning models. "
        "Accepts string levels ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max') or a "
        "float in [0.0, 0.99] for fine-grained control. "
        "'none' disables reasoning entirely, 'low' is the least effort, 'high' is the most effort. "
        "Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning "
        "in a response. 'none' defaults thinking and enable_thinking to false in "
        "chat_template_kwargs (unless explicitly overridden). Not supported in the harmony path."
        "'max' is an sglang extension to the OpenAI schema for "
        "models that expose a maximum-effort tier above 'high'; models that don't "
        "support it treat it the same as 'high'.",
    )  # 推理强度：字符串档位或 [0.0, 0.99] 浮点数；'none' 关闭思考，会影响 chat_template_kwargs 中的 thinking/enable_thinking
    task: Optional[
        Literal["action", "query", "authority", "domain", "title", "read_url"]
    ] = Field(
        default=None,
        description="DeepSeek-V4 quick instruction task. When set, the last "
        "user/developer message is treated as a single-shot classification prompt "
        "and the corresponding task special token (e.g. `<｜domain｜>`) is appended "
        "before generation. Only honored by the dsv4 chat encoder; ignored otherwise.",
    )  # DeepSeek-V4 快捷指令任务：把最后一条 user/developer 消息当作单次分类提示，并追加对应任务特殊 token

    # 以下为 SRT 后端专有参数，OpenAI 模型会忽略
    top_k: Optional[int] = None  # top-k 采样：只从概率最高的 k 个 token 中采样（-1 表示不限制）
    min_p: Optional[float] = None  # min-p 采样：保留概率 >= min_p * 最大概率 的 token
    min_tokens: int = 0  # 至少生成多少个 token 后才允许停止
    regex: Optional[str] = None  # 正则约束解码：输出必须匹配该正则
    ebnf: Optional[str] = None  # EBNF 语法约束解码
    repetition_penalty: Optional[float] = None  # 重复惩罚系数（>1 抑制重复）
    stop_token_ids: Optional[List[int]] = None  # 以 token id 形式指定停止符
    stop_regex: Optional[Union[str, List[str]]] = None  # 以正则形式指定停止符
    no_stop_trim: bool = False  # 命中停止符时是否保留停止符本身（不裁剪）
    ignore_eos: bool = False  # 是否忽略 EOS，一直生成到 max_new_tokens
    continue_final_message: bool = False  # 是否续写最后一条消息（不重新加 assistant 起始标记）
    skip_special_tokens: bool = True  # 输出解码时是否跳过特殊 token
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None  # LoRA 适配器路径（可批量指定多个）
    session_id: Optional[str] = None  # 会话 id，用于多轮复用/会话级缓存
    session_params: Optional[Dict] = None  # 会话相关参数
    separate_reasoning: bool = True  # 是否把思维链从 content 中拆分到 reasoning_content
    stream_reasoning: bool = True  # 流式时是否输出思维链内容
    chat_template_kwargs: Optional[Dict] = None  # 透传给 chat template 的额外参数（如 thinking/enable_thinking）

    # SGLang 多模态控制参数（扩展）
    max_dynamic_patch: Optional[int] = None  # 图像/视频动态切块（patch）数量上限
    min_dynamic_patch: Optional[int] = None  # 图像/视频动态切块（patch）数量下限
    use_audio_in_video: bool = False  # 是否从视频中提取并使用音轨

    images_config: Optional[Dict] = None  # 图片处理配置
    video_config: Optional[Dict] = None  # 视频处理配置

    # 自定义 logit processor，用于高级采样控制
    custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None  # 自定义 logit 处理器（代码或路径）
    custom_params: Optional[Dict] = None  # 传给自定义处理器/采样参数的额外参数

    # 预先算好的 prompt token ids：提供后完全跳过 chat template 分词；
    # 但 messages 仍会用于推导停止符与 tool_call_constraint。
    input_ids: Optional[List[int]] = None  # 预分词后的输入 token ids

    # 请求 id
    rid: Optional[Union[List[str], str]] = None
    # 调用方自定义的请求分类 key
    extra_key: Optional[Union[List[str], str]] = None
    # 请求缓存的盐值（cache salt）
    cache_salt: Optional[Union[List[str], str]] = None
    # 请求优先级
    priority: Optional[int] = None

    # PD 分离（prefill/decode 分离）相关
    bootstrap_host: Optional[Union[List[str], str]] = None  # 对端 bootstrap 服务地址
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None  # 对端 bootstrap 端口
    bootstrap_room: Optional[Union[List[int], int]] = None  # bootstrap 房间号（配对 prefill/decode 请求）

    # DP 路由：由外部 router 指定具体的 DP worker
    routed_dp_rank: Optional[int] = None
    # PD 分离场景：提示 decode 侧应去哪个 prefill DP worker 取 KV cache
    disagg_prefill_dp_rank: Optional[int] = None
    # 已废弃：请改用 routed_dp_rank
    data_parallel_rank: Optional[int] = None

    # OpenAI/SGLang 默认采样参数（用户未显式指定且模型配置缺失时的兜底值）
    _DEFAULT_SAMPLING_PARAMS = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }

    @model_validator(mode="before")
    @classmethod
    def _handle_deprecated_dp_rank(cls, values):
        # 兼容已废弃的 data_parallel_rank：统一迁移到 routed_dp_rank
        return _migrate_deprecated_dp_rank(values)

    @model_validator(mode="before")
    @classmethod
    def set_tool_choice_default(cls, values):
        # 未显式指定 tool_choice 时的默认值：无工具（含消息级工具）则为 "none"，否则为 "auto"
        if values.get("tool_choice") is None:
            if values.get("tools") is None and not _has_message_level_tools(
                values.get("messages")
            ):
                values["tool_choice"] = "none"
            else:
                values["tool_choice"] = "auto"
        return values

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def validate_reasoning_effort_type(cls, value):
        # 布尔值会被 pydantic 宽松地当成 0/1 数字，这里显式拒绝
        if isinstance(value, bool):
            raise ValueError("reasoning_effort must not be a boolean")
        return value

    @model_validator(mode="before")
    @classmethod
    def normalize_reasoning_inputs(cls, values: Dict):
        # 归一化两种推理开关的写法：
        # 1) reasoning_effort（字符串档位或 [0.0,0.99] 浮点数）
        # 2) 嵌套的 reasoning 对象（effort/reasoning_effort + enabled/enable）
        # 最终统一转写为 reasoning_effort 与 chat_template_kwargs 中的 thinking 标志。
        r = values.get("reasoning")
        thinking = None

        if r is not None and isinstance(r, dict):
            effort = r.get("effort")
            if effort is None:
                effort = r.get("reasoning_effort")
            if isinstance(effort, str) and effort in {
                "none",
                "minimal",
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
            }:
                values["reasoning_effort"] = effort
            elif isinstance(effort, (int, float)) and not isinstance(effort, bool):
                values["reasoning_effort"] = float(effort)
            elif isinstance(effort, str):
                # 与顶层 reasoning_effort 字段保持一致：宽松 union 会把数字字符串
                # 转成浮点数，后续再做范围校验。
                try:
                    values["reasoning_effort"] = float(effort)
                except ValueError as exc:
                    raise ValueError(f"invalid reasoning effort: {effort!r}") from exc
            elif effort is not None:
                raise ValueError(f"invalid reasoning effort: {effort!r}")

            enabled = (
                r.get("enabled")
                if r.get("enabled") is not None
                else r.get("enable", False)
            )  # 兼容 enabled 与 enable 两种写法
            if isinstance(enabled, str):
                enabled = enabled.strip().lower() in {"1", "true", "yes", "y", "on"}  # 字符串形式的布尔值
            if enabled:
                thinking = True

        effort = values.get("reasoning_effort")
        if effort is not None:
            thinking = effort != "none"  # effort=none 即关闭思考

        if thinking is not None:
            ctk = values.get("chat_template_kwargs")
            if not isinstance(ctk, dict):
                ctk = {}
            # 不同模型检查的键不同：
            # - "thinking"：deepseek-v3、kimi_k2
            # - "enable_thinking"：qwen3、glm45、nemotron_3、interns1
            ctk.setdefault("thinking", thinking)
            ctk.setdefault("enable_thinking", thinking)
            values["chat_template_kwargs"] = ctk

        return values

    @model_validator(mode="before")
    @classmethod
    def set_json_schema(cls, values):
        # 兼容简写写法：response_format={"type":"json_schema", "schema": {...}}
        # 将顶层 schema 规范化到 OpenAI 标准的 {"json_schema": {"name", "schema", "strict"}} 结构
        response_format = values.get("response_format")
        if not response_format:
            return values

        if response_format.get("type") != "json_schema":
            return values

        schema = response_format.pop("schema", None)
        json_schema = response_format.get("json_schema")

        if json_schema:
            # 已是标准写法，无需转换
            return values

        if schema:
            name_ = schema.get("title", "Schema")
            strict_ = None
            # 支持把 strict 写在 schema.properties 中作为约定的兼容写法
            if "properties" in schema and "strict" in schema["properties"]:
                item = schema["properties"].pop("strict", None)
                strict_ = bool(item and item.get("default", False))

            response_format["json_schema"] = {
                "name": name_,
                "schema": schema,
                "strict": strict_,
            }

        return values

    def to_sampling_params(
        self,
        stop: List[str],
        model_generation_config: Dict[str, Any],
        tool_call_constraint: Optional[ToolCallConstraint] = None,
        renderer_handles_response_format: bool = False,
    ) -> Dict[str, Any]:
        """
        把请求转换成底层采样参数。
        优先级：用户显式传值 > 模型 generation_config > OpenAI 默认值
        """

        def get_param(param_name: str):
            # 取参数：用户值优先，否则回落模型配置，最后回落默认值
            value = getattr(self, param_name)
            if value is None:
                return model_generation_config.get(
                    param_name, self._DEFAULT_SAMPLING_PARAMS[param_name]
                )
            return value

        # 特殊 token 之间是否插入空格（可通过 chat_template_kwargs 覆盖）
        spaces_between_special_tokens = (
            True
            if self.chat_template_kwargs is None
            else self.chat_template_kwargs.get("spaces_between_special_tokens", True)
        )

        # 汇总采样参数（注意 max_new_tokens 优先取 max_completion_tokens，兼容旧字段 max_tokens）
        sampling_params = {
            "temperature": get_param("temperature"),  # 采样温度
            "max_new_tokens": self.max_completion_tokens or self.max_tokens,  # 最多生成 token 数
            "min_new_tokens": self.min_tokens,  # 最少生成 token 数
            "stop": stop,  # 停止字符串（由上层渲染/解析得到）
            "stop_token_ids": self.stop_token_ids,  # 停止 token id
            "stop_regex": self.stop_regex,  # 停止正则
            "top_p": get_param("top_p"),  # 核采样
            "top_k": get_param("top_k"),  # top-k 采样
            "min_p": get_param("min_p"),  # min-p 采样
            "presence_penalty": self.presence_penalty,  # 存在惩罚
            "frequency_penalty": self.frequency_penalty,  # 频率惩罚
            "repetition_penalty": get_param("repetition_penalty"),  # 重复惩罚
            "regex": self.regex,  # 正则约束
            "ebnf": self.ebnf,  # EBNF 约束
            "n": self.n,  # 生成候选数
            "no_stop_trim": self.no_stop_trim,  # 是否保留停止符
            "ignore_eos": self.ignore_eos,  # 是否忽略 EOS
            "skip_special_tokens": self.skip_special_tokens,  # 解码是否跳过特殊 token
            "logit_bias": self.logit_bias,  # logit 偏置
            "custom_params": self.custom_params,  # 自定义参数
            "sampling_seed": self.seed,  # 采样随机种子
            "spaces_between_special_tokens": spaces_between_special_tokens,  # 特殊 token 间是否加空格
        }

        if self.response_format and self.response_format.type == "json_schema":
            # strict=false 时，只有当 renderer 会把 response_format 透传给模型时才允许不加约束；
            # 普通 chat template 根本看不到它。
            if (
                self.response_format.json_schema.strict is not False
                or not renderer_handles_response_format
            ):
                sampling_params["json_schema"] = convert_json_schema_to_str(
                    self.response_format.json_schema.schema_
                )
        elif self.response_format and self.response_format.type == "json_object":
            # json_object 退化为“任意 JSON 对象”约束
            sampling_params["json_schema"] = '{"type": "object"}'
        elif self.response_format and self.response_format.type == "structural_tag":
            # 结构化标签约束（SGLang 扩展）
            sampling_params["structural_tag"] = convert_json_schema_to_str(
                self.response_format.model_dump(by_alias=True)
            )

        # 检查是否已存在输出约束（regex/ebnf/structural_tag/json_schema）
        has_existing_constraints = (
            sampling_params.get("regex")
            or sampling_params.get("ebnf")
            or sampling_params.get("structural_tag")
            or sampling_params.get("json_schema")
        )

        if tool_call_constraint and has_existing_constraints:
            # 工具调用约束与输出约束互斥：required/指定工具时直接报错
            if self.tool_choice == "required" or isinstance(
                self.tool_choice, ToolChoice
            ):
                raise ValueError(
                    "tool_choice 'required' or a named tool cannot be combined with "
                    "response_format, regex, or ebnf: the tool-call constraint and the "
                    "output constraint cannot both be honored."
                )
            logger.warning("Constrained decoding is not compatible with tool calls.")
        elif tool_call_constraint:
            # 无其他约束时，把工具调用约束转成对应的约束参数
            constraint_type, constraint_value = tool_call_constraint
            if constraint_type == "structural_tag":
                sampling_params[constraint_type] = convert_json_schema_to_str(
                    constraint_value.model_dump(by_alias=True)
                )
            elif constraint_type == "json_schema":
                sampling_params[constraint_type] = convert_json_schema_to_str(
                    constraint_value  # type: ignore
                )
            else:
                sampling_params[constraint_type] = constraint_value

        return sampling_params


class ChatMessage(BaseModel):
    """响应中的完整消息（非流式）。"""

    role: Optional[str] = None  # 消息角色，通常为 assistant
    content: Optional[str] = None  # 正文内容（思维链会被拆到 reasoning_content）
    reasoning_content: Optional[str] = None  # 思维链内容（separate_reasoning=true 时单独返回）
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])  # 模型产生的工具调用


class ChatCompletionResponseChoice(BaseModel):
    """非流式响应中的单个候选（一条 messages 生成一个 choice）。"""

    index: int  # 候选序号（与请求的 n 对应）
    message: ChatMessage  # 生成的消息
    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None  # token 级 log 概率明细
    finish_reason: Optional[
        Literal[
            "stop", "length", "tool_calls", "content_filter", "function_call", "abort"
        ]
    ] = None  # 结束原因：stop=命中停止符，length=达长度上限，tool_calls=触发工具调用，abort=被中断
    matched_stop: Union[None, int, str] = None  # 实际命中的停止符（token id 或字符串）
    hidden_states: Optional[object] = None  # 调试：hidden states
    prompt_token_ids: Optional[List[int]] = None  # 调试：输入 token ids
    response_token_ids: Optional[List[int]] = None  # 调试：输出 token ids
    meta_info: Optional[Dict[str, Any]] = None  # 调试：底层调度/采样 meta 信息

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        # 调试字段为 None 时不出现在响应中，保持响应体精简
        if self.hidden_states is None:
            data.pop("hidden_states", None)
        if self.prompt_token_ids is None:
            data.pop("prompt_token_ids", None)
        if self.response_token_ids is None:
            data.pop("response_token_ids", None)
        if self.meta_info is None:
            data.pop("meta_info", None)
        return data


class ChatCompletionResponse(BaseModel):
    """非流式的 /v1/chat/completions 响应体（对齐 OpenAI）。"""

    id: str  # 本次 completion 的唯一 id
    object: str = "chat.completion"  # 对象类型，固定为 chat.completion
    created: int = Field(default_factory=lambda: int(time.time()))  # 创建时间（unix 秒）
    model: str  # 实际使用的模型名
    choices: List[ChatCompletionResponseChoice]  # 候选列表（对应 n）
    usage: UsageInfo  # token 用量统计
    metadata: Optional[Dict[str, Any]] = None  # 额外元信息
    sglext: Optional[SglExt] = None  # SGLang 扩展字段

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        # 无扩展字段时不输出 sglext
        if self.sglext is None:
            data.pop("sglext", None)
        return data


class DeltaMessage(BaseModel):
    """流式响应中的增量消息（每个 SSE chunk 携带）。"""

    role: Optional[str] = None  # 首个 chunk 会带上 role
    content: Optional[str] = None  # 本 chunk 增量正文
    reasoning_content: Optional[str] = None  # 本 chunk 增量思维链内容
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])  # 本 chunk 增量的工具调用
    hidden_states: Optional[object] = None  # 调试：hidden states

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.hidden_states is None:
            data.pop("hidden_states", None)
        return data


class ChatCompletionResponseStreamChoice(BaseModel):
    """流式响应中的单个候选。"""

    index: int  # 候选序号（与请求的 n 对应）
    delta: DeltaMessage  # 本 chunk 的增量内容
    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None  # token 级 log 概率明细
    finish_reason: Optional[
        Literal[
            "stop", "length", "tool_calls", "content_filter", "function_call", "abort"
        ]
    ] = None  # 结束原因，仅在最后一个 chunk 给出
    matched_stop: Union[None, int, str] = None  # 实际命中的停止符


class ChatCompletionStreamResponse(BaseModel):
    """流式的 /v1/chat/completions 响应（每个 SSE chunk 一个）。"""

    id: str  # 与整个流共享的 completion id
    object: str = "chat.completion.chunk"  # 对象类型，固定为 chat.completion.chunk
    created: int = Field(default_factory=lambda: int(time.time()))  # 创建时间（unix 秒）
    model: str  # 实际使用的模型名
    choices: List[ChatCompletionResponseStreamChoice]  # 候选增量列表
    usage: Optional[UsageInfo] = None  # token 用量（仅在 include_usage=true 的最后 chunk 返回）
    sglext: Optional[SglExt] = None  # SGLang 扩展字段

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        # 无扩展字段时不输出 sglext
        if self.sglext is None:
            data.pop("sglext", None)
        return data


class MultimodalEmbeddingInput(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None
    video: Optional[str] = None


EmbeddingInput = Union[
    List[int], List[List[int]], str, List[str], List[MultimodalEmbeddingInput]
]


class EmbeddingRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings/create
    input: EmbeddingInput
    model: str = DEFAULT_MODEL_NAME
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None

    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Priority for the request
    priority: Optional[int] = None
    # LoRA adapter path(s)
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    # Placeholder token id used to locate embedding override positions in input token IDs.
    embed_override_token_id: Optional[int] = None
    # Per-input embedding overrides (null entries skip that input).
    # Shape: [num_inputs][num_replacements][hidden_size]
    embed_overrides: Optional[List[Optional[List[List[float]]]]] = None


class EmbeddingObject(BaseModel):
    embedding: Union[List[float], str]
    index: int
    object: str = "embedding"


ClassifyInput = Union[str, List[str], List[int]]


class ClassifyRequest(BaseModel):
    # OpenAI-compatible classification request
    model: str = DEFAULT_MODEL_NAME
    input: ClassifyInput
    user: Optional[str] = None

    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Priority for the request
    priority: Optional[int] = None


class ClassifyData(BaseModel):
    index: int
    label: str
    probs: List[float]
    num_classes: int


class ClassifyResponse(BaseModel):
    id: str
    object: str = "list"
    created: int
    model: str
    data: List[ClassifyData]
    usage: UsageInfo


class EmbeddingResponse(BaseModel):
    data: List[EmbeddingObject]
    model: str
    object: str = "list"
    usage: Optional[UsageInfo] = None


class ScoringRequest(BaseModel):
    query: Optional[Union[str, List[int]]] = (
        None  # Query text or pre-tokenized token IDs
    )
    items: Optional[Union[str, List[str], List[List[int]]]] = (
        None  # Item text(s) or pre-tokenized token IDs
    )
    # Placeholder token id used to locate embedding override positions in query/items.
    embed_override_token_id: Optional[int] = None
    # Query embedding overrides.
    query_embed_overrides: Optional[List[List[float]]] = (
        None  # [num_query_embed_overrides][hidden_size]
    )
    # Per-item embedding overrides (null entries skip that item).
    item_embed_overrides: Optional[List[Optional[List[List[float]]]]] = (
        None  # [num_items][num_item_embed_overrides][hidden_size]
    )
    label_token_ids: Optional[List[int]] = (
        None  # Token IDs to compute probabilities for
    )
    apply_softmax: bool = False
    item_first: bool = False
    return_pooled_hidden_states: bool = False
    model: str = DEFAULT_MODEL_NAME


class ScoringResponse(BaseModel):
    scores: List[
        List[float]
    ]  # List of lists of probabilities, each in the order of label_token_ids
    pooled_hidden_states: Optional[List[Optional[List[float]]]] = None
    model: str
    usage: Optional[UsageInfo] = None
    object: str = "scoring"


class V1RerankReqInput(BaseModel):
    query: RerankContent = Field(
        ...,
        description="The query to match against documents. Can be a string (text-only) "
        "or a list of content parts for multimodal queries (text, image_url, video_url).",
    )
    documents: List[RerankContent] = Field(
        ...,
        description="List of documents to rank. Each document can be a string (text-only) "
        "or a list of content parts for multimodal documents (text, image_url, video_url).",
    )
    instruct: Optional[str] = Field(
        default=None,
        description="The instruct to the reranker model.",
    )
    top_n: Optional[int] = Field(
        default=None,
        description="Maximum number of documents to return. Defaults to returning all documents. "
        "If specified value is greater than the total number of documents, all documents will be returned.",
    )
    return_documents: bool = Field(
        default=True,
        description="Whether to return documents in the response. Only included when set to true.",
    )

    @field_validator("top_n")
    @classmethod
    def validate_top_n(cls, v):
        if v is not None and v < 1:
            raise ValueError("Value error, parameter top_n should be larger than 0.")
        return v

    def is_multimodal(self) -> bool:
        """Check if the request contains any multimodal content."""
        if isinstance(self.query, list):
            return True
        for doc in self.documents:
            if isinstance(doc, list):
                return True
        return False


class RerankResponse(BaseModel):
    score: float
    document: Optional[str] = None
    index: int
    meta_info: Optional[dict] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        # Exclude document field if it's None
        if self.document is None:
            data.pop("document", None)
        return data


class TokenizeRequest(BaseModel):
    """Request schema for the /tokenize endpoint.

    /tokenize 接口的请求体：把「文本」或「对话消息」编码成 token id 序列。
    两种输入二选一（prompt 或 messages），由下面的校验器保证。
    """

    # 允许携带额外字段（不作为请求参数校验），例如透传的模型侧自定义参数
    model_config = ConfigDict(extra="allow")

    model: str = DEFAULT_MODEL_NAME  # 目标模型名，默认取全局默认模型
    prompt: Optional[Union[str, List[str]]] = None  # 纯文本输入，支持单条或多条
    messages: Optional[List[ChatCompletionMessageParam]] = None  # 对话消息输入，走 chat template 渲染
    tools: Optional[List[Tool]] = Field(default=None, examples=[None])  # 工具定义，用于渲染 chat template
    tool_choice: Optional[Union[ToolChoice, Literal["auto", "required", "none"]]] = (
        Field(default=None, examples=["auto"])  # 工具选择策略：auto/required/none 或指定某个工具
    )
    reasoning_effort: ReasoningEffortType = None  # 推理强度（reasoning 模型相关，影响模板渲染）
    continue_final_message: bool = False  # 是否续写最后一条消息（不重新加 assistant 起始标记）
    chat_template_kwargs: Optional[Dict] = None  # 透传给 chat template 的额外参数
    add_special_tokens: bool = Field(
        default=True,
        description="whether to add model-specific special tokens (e.g. BOS/EOS) during encoding.",
    )  # 编码时是否添加模型特有特殊 token（如 BOS/EOS）

    @model_validator(mode="after")
    def validate_tokenize_input(self) -> TokenizeRequest:
        # 校验：prompt 与 messages 必须且只能提供一个
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("Exactly one of 'prompt' or 'messages' must be provided.")
        return self

    def to_chat_completion_request(self) -> ChatCompletionRequest:
        # 把本请求转换成 ChatCompletionRequest，以便复用 chat 侧的模板渲染/编码逻辑
        data = self.model_dump(
            exclude={"prompt", "add_special_tokens"},  # prompt 需转成 messages；add_special_tokens 非 chat 参数
            exclude_none=True,
        )
        # 合并 model_config 允许的额外字段，一并透传给 ChatCompletionRequest
        extra = getattr(self, "__pydantic_extra__", None)
        if extra:
            data.update(extra)
        return ChatCompletionRequest.model_validate(data)


class TokenizeResponse(BaseModel):
    """Response schema for the /tokenize endpoint.

    /tokenize 接口的响应体：返回编码后的 token id。
    """

    tokens: Union[List[int], List[List[int]]]  # token id；单条输入为 List[int]，批量输入为 List[List[int]]
    count: Union[int, List[int]]  # 每条输入的 token 数量，形状与 tokens 对应
    max_model_len: int  # 模型支持的最大上下文长度，便于调用方自行截断


class DetokenizeRequest(BaseModel):
    """Request schema for the /detokenize endpoint.

    /detokenize 接口的请求体：把 token id 序列解码回文本，是 tokenize 的逆过程。
    """

    model: str = DEFAULT_MODEL_NAME  # 目标模型名，默认取全局默认模型
    tokens: Union[List[int], List[List[int]]]  # 待解码的 token id；单条为 List[int]，批量为 List[List[int]]
    skip_special_tokens: bool = Field(
        default=True,
        description="whether to exclude special tokens (e.g. padding or EOS) during decoding.",
    )  # 解码时是否过滤掉特殊 token（如 padding / EOS）


class DetokenizeResponse(BaseModel):
    """Response schema for the /detokenize endpoint.

    /detokenize 接口的响应体：返回解码后的文本。
    """

    text: Union[str, List[str]]  # 解码结果；单条输入返回 str，批量输入返回 List[str]


OpenAIServingRequest = Union[
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    ClassifyRequest,
    ScoringRequest,
    V1RerankReqInput,
    TokenizeRequest,
    DetokenizeRequest,
]


# Response API protocol definitions
class ResponseReasoningParam(BaseModel):
    """Reasoning parameters for responses."""

    effort: Optional[ReasoningEffortTier] = Field(
        default="medium",
        description="Constrains effort on reasoning for reasoning models. "
        "Accepts the OpenAI string tiers "
        "('none','minimal','low','medium','high','xhigh','max').",
    )
    summary: Optional[Literal["auto", "concise", "detailed"]] = Field(
        default=None,
        description="Include a summary of the model's reasoning trace on the response.",
    )


# Only ``function`` / ``web_search*`` / ``code_interpreter`` are wired to
# execution paths; the rest pass validation so clients aren't rejected.
RESPONSE_TOOL_TYPES = Literal[
    "function",
    "web_search",
    "web_search_preview",
    "code_interpreter",
    "file_search",
    "image_generation",
    "computer_use_preview",
    "local_shell",
    "mcp",
    "custom",
    "namespace",
    "tool_search",
]


class ResponseTool(BaseModel):
    """Tool definition for responses."""

    type: RESPONSE_TOOL_TYPES = Field(description="Type of tool to enable")
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: bool = False
    # Inner schemas for ``namespace`` tools.
    tools: Optional[List[Dict[str, Any]]] = None

    @model_validator(mode="after")
    def validate_function_tool(self) -> ResponseTool:
        if self.type == "function" and not self.name:
            raise ValueError("Function tools must include a name.")
        return self


ResponseInputOutputItem: TypeAlias = Union[
    ResponseInputItemParam,
    "ResponseReasoningItem",
    ResponseFunctionToolCall,
]


class ResponsesRequest(BaseModel):
    """Request body for v1/responses endpoint."""

    # Core OpenAI API fields (ordered by official documentation)
    background: Optional[bool] = False
    include: Optional[
        List[
            Literal[
                "code_interpreter_call.outputs",
                "computer_call_output.output.image_url",
                "file_search_call.results",
                "message.input_image.image_url",
                "message.output_text.logprobs",
                "reasoning.encrypted_content",
            ]
        ]
    ] = None
    # Accept dict-shaped items as the loose arm; downstream normalization
    # handles replayed shapes that don't satisfy every openai TypedDict.
    input: Union[str, List[ResponseInputOutputItem], List[Dict[str, Any]]]
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    model: Optional[str] = None  # Made optional to match vLLM
    parallel_tool_calls: Optional[bool] = True
    previous_response_id: Optional[str] = None
    reasoning: Optional[ResponseReasoningParam] = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] = "auto"
    store: Optional[bool] = True
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    text: Optional[ResponseTextConfig] = None
    tool_choice: Union[Literal["auto", "required", "none"], Dict[str, Any]] = "auto"
    tools: List[ResponseTool] = Field(default_factory=list)
    top_logprobs: Optional[int] = 0
    top_p: Optional[float] = None
    truncation: Optional[Literal["auto", "disabled"]] = "disabled"
    user: Optional[str] = None

    # Extra SGLang parameters
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    request_id: str = Field(
        default_factory=lambda: f"resp_{uuid.uuid4().hex}",
        description="The request_id related to this request. If the caller does not set it, a random uuid will be generated.",
    )
    session_id: Optional[str] = None
    priority: int = Field(default=0, description="Request priority")
    extra_key: Optional[str] = Field(
        default=None,
        description="Extra key for caller-defined request classification",
    )
    cache_salt: Optional[str] = Field(
        default=None, description="Cache salt for request caching"
    )

    # SGLang sampling extras. ``None`` defers to ``--preferred-sampling-params``.
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[Union[str, List[str]]] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None

    # Default sampling parameters
    _DEFAULT_SAMPLING_PARAMS = {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }

    @model_validator(mode="before")
    @classmethod
    def normalize_reasoning_to_thinking(cls, values):
        """Turn reasoning.effort == "none" into a thinking toggle, as
        ChatCompletionRequest does. Both keys are set: families differ
        ("thinking" for deepseek/kimi, "enable_thinking" for qwen3/glm)."""
        if not isinstance(values, dict):
            return values
        # mode="before", so reasoning is still raw: a dict from JSON, or a
        # built param object when constructed in Python.
        r = values.get("reasoning")
        effort = None
        if isinstance(r, dict):
            effort = r.get("effort") or r.get("reasoning_effort")
        elif isinstance(r, ResponseReasoningParam):
            effort = r.effort

        if effort == "none":
            existing = values.get("chat_template_kwargs")
            existing = existing if isinstance(existing, dict) else {}
            # existing last: an explicit caller value wins.
            values["chat_template_kwargs"] = {
                "thinking": False,
                "enable_thinking": False,
                **existing,
            }
        return values

    @model_validator(mode="before")
    @classmethod
    def normalize_responses_input(cls, values):
        if not isinstance(values, dict):
            return values

        input_value = values.get("input")
        if not isinstance(input_value, list):
            return values

        values = values.copy()
        values["input"] = [
            cls._normalize_input_item_for_validation(item) for item in input_value
        ]
        return values

    @staticmethod
    def _normalize_input_item_for_validation(item):
        if not isinstance(item, dict):
            return item

        # an output item replayed into input carries a string id; without this it'd
        # be read as an item-reference, drop its content, and fail as an empty {}.
        # input-item ids aren't resolved server-side, so just drop it.
        if isinstance(item.get("id"), str) and item.get("content") is not None:
            item = {k: v for k, v in item.items() if k != "id"}

        content = item.get("content")
        if not isinstance(content, list):
            return item

        item = dict(item)
        item["content"] = [
            ResponsesRequest._normalize_content_part_for_validation(part)
            for part in content
        ]
        return item

    @staticmethod
    def _normalize_content_part_for_validation(part):
        if not isinstance(part, dict):
            return part

        part_type = part.get("type")
        if part_type != "input_image" or part.get("detail") is not None:
            return part

        part = part.copy()
        part["detail"] = "auto"
        return part

    @staticmethod
    def _json_schema_from_text_format(
        text: Optional[ResponseTextConfig],
    ) -> Optional[str]:
        """Map a Responses ``text.format`` to a json_schema string, or None when
        no JSON constraint applies (``text`` and anything unrecognized)."""
        response_format = text.format if text is not None else None
        if isinstance(response_format, ResponseFormatJSONObject):
            return '{"type": "object"}'
        if not isinstance(response_format, ResponseFormatTextJSONSchemaConfig):
            return None
        schema = response_format.schema_
        return convert_json_schema_to_str(schema) if schema is not None else None

    def is_include_output_logprobs(self) -> bool:
        return bool(self.include and "message.output_text.logprobs" in self.include)

    def has_json_schema_constraint(self) -> bool:
        return self._json_schema_from_text_format(self.text) is not None

    def effective_tool_choice(self) -> Union[str, Dict[str, Any]]:
        """``tool_choice`` reduced to what the server can actually honor: of the
        object forms only a named ``function`` survives, the rest (web_search,
        mcp, ...) can't be forced through the tool-call parser."""
        tool_choice = self.tool_choice
        if not isinstance(tool_choice, dict):
            return tool_choice
        name = tool_choice.get("name") or (tool_choice.get("function") or {}).get(
            "name"
        )
        if tool_choice.get("type") == "function" and name:
            return {"type": "function", "name": name}
        return "auto"

    def to_sampling_params(
        self,
        default_max_tokens: int,
        default_params: Optional[Dict] = None,
        stop: Optional[Union[str, List[str]]] = None,
        tool_call_constraint: Optional[ToolCallConstraint] = None,
    ) -> Dict[str, Any]:
        """Convert to sampling parameters for generation."""
        if default_params is None:
            default_params = {}

        # Use max_output_tokens if available, otherwise use max_tokens for backwards compatibility
        if self.max_output_tokens is not None:
            max_tokens = min(self.max_output_tokens, default_max_tokens)
        else:
            max_tokens = default_max_tokens

        # Headroom for BOS/EOS the engine appends on top of prompt+budget.
        max_tokens -= 2

        temperature = self.temperature
        if temperature is None:
            temperature = default_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"]
            )

        top_p = self.top_p
        if top_p is None:
            top_p = default_params.get("top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"])

        # Omit None entries so they fall through to ``--preferred-sampling-params``
        # rather than overriding it with a literal default.
        params: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop if stop is None else stop,
        }
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.min_p is not None:
            params["min_p"] = self.min_p
        if self.repetition_penalty is not None:
            params["repetition_penalty"] = self.repetition_penalty

        # Apply any additional default parameters
        for key, value in default_params.items():
            if key not in params or params[key] is None:
                params[key] = value

        json_schema = self._json_schema_from_text_format(self.text)
        if json_schema is not None:
            params["json_schema"] = json_schema

        has_existing_constraints = (
            params.get("regex")
            or params.get("ebnf")
            or params.get("structural_tag")
            or params.get("json_schema")
        )
        if tool_call_constraint and has_existing_constraints:
            # Refuse rather than silently drop the tool-call grammar.
            raise ValueError(
                "Cannot combine tool calls with constrained decoding "
                "(text.format / regex / ebnf / structural_tag / json_schema). "
                "Remove one."
            )
        if tool_call_constraint:
            constraint_type, constraint_value = tool_call_constraint
            if constraint_type in ("structural_tag", "json_schema"):
                params[constraint_type] = convert_json_schema_to_str(
                    constraint_value.model_dump(by_alias=True)
                    if hasattr(constraint_value, "model_dump")
                    else constraint_value
                )
            else:
                params[constraint_type] = constraint_value

        return params


class PromptTokenUsageInfo(BaseModel):
    """Prompt token usage details."""

    cached_tokens: int = 0


class ResponsesResponse(BaseModel):
    """Response body for v1/responses endpoint."""

    id: str = Field(default_factory=lambda: f"resp_{time.time()}")
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    model: str

    output: List[
        Union[ResponseOutputItem, ResponseReasoningItem, ResponseFunctionToolCall]
    ] = Field(default_factory=list)
    status: Literal[
        "queued", "in_progress", "completed", "incomplete", "failed", "cancelled"
    ]
    usage: Optional[UsageInfo] = None
    parallel_tool_calls: bool = True
    tool_choice: Union[str, Dict[str, Any]] = "auto"
    tools: List[ResponseTool] = Field(default_factory=list)

    # OpenAI compatibility fields. not all are used at the moment.
    # Recommend checking https://platform.openai.com/docs/api-reference/responses
    error: Optional[dict] = None
    incomplete_details: Optional[dict] = None  # TODO(v) support this input
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    previous_response_id: Optional[str] = None
    reasoning: Optional[dict] = (
        # Unused. No model supports this. For GPT-oss, system prompt sets
        # the field, not server args.
        None  # {"effort": Optional[str], "summary": Optional[str]}
    )
    store: Optional[bool] = None
    temperature: Optional[float] = None
    text: Optional[dict] = None  # e.g. {"format": {"type": "text"}}
    top_p: Optional[float] = None
    truncation: Optional[str] = None
    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @field_serializer("usage")
    def _serialize_usage(self, usage: Optional[UsageInfo], _info):
        """Emit the Responses usage shape, not the chat one UsageInfo carries."""
        if usage is None:
            return None
        cached = (
            usage.prompt_tokens_details.cached_tokens
            if usage.prompt_tokens_details
            else 0
        )
        return {
            "input_tokens": usage.prompt_tokens,
            "input_tokens_details": {
                "cached_tokens": cached,
                # required (no default) in the SDK's InputTokensDetails model
                "cache_write_tokens": 0,
            },
            "output_tokens": usage.completion_tokens or 0,
            "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens or 0},
            "total_tokens": usage.total_tokens,
        }

    @classmethod
    def from_request(
        cls,
        request: ResponsesRequest,
        sampling_params: Any,
        model_name: str,
        created_time: int,
        output: List[
            Union[ResponseOutputItem, ResponseReasoningItem, ResponseFunctionToolCall]
        ],
        status: str,
        usage: Optional[UsageInfo],
    ) -> ResponsesResponse:
        """Create a response from a request."""

        # Determine if the output is plain text only to set text.format
        def _is_text_only(
            items: List[
                Union[
                    ResponseOutputItem, ResponseReasoningItem, ResponseFunctionToolCall
                ]
            ],
        ) -> bool:
            if not items:
                return False
            for it in items:
                # tool call -> not pure text.
                if isinstance(it, ResponseReasoningItem) or isinstance(
                    it, ResponseFunctionToolCall
                ):
                    return False
                try:
                    if isinstance(it, ResponseOutputText):
                        continue
                    elif isinstance(it, ResponseOutputMessage):
                        if not it.content:
                            continue
                        for c in it.content:
                            if not isinstance(c, ResponseOutputText):
                                return False
                    else:
                        # Unknown type, not considered text-only
                        return False
                except AttributeError:
                    return False
            return True

        if request.text is not None:
            # by_alias keeps the wire key "schema" rather than the attr schema_.
            text_format = request.text.model_dump(by_alias=True, exclude_none=True)
        else:
            text_format = (
                {"format": {"type": "text"}} if _is_text_only(output) else None
            )

        return cls(
            id=request.request_id,
            created_at=created_time,
            model=model_name,
            output=output,
            status=status,
            usage=usage,
            parallel_tool_calls=(
                request.parallel_tool_calls
                if request.parallel_tool_calls is not None
                else True
            ),
            tool_choice=request.effective_tool_choice(),
            tools=request.tools,
            # fields for parity with v1/responses
            error=None,
            incomplete_details=(
                {"reason": "max_output_tokens"} if status == "incomplete" else None
            ),
            instructions=request.instructions,
            max_output_tokens=request.max_output_tokens,
            previous_response_id=request.previous_response_id,  # TODO(v): ensure this is propagated if retrieved from store
            reasoning={
                "effort": (
                    request.reasoning.effort
                    if request.reasoning
                    and request.reasoning.effort in ECHOABLE_REASONING_EFFORTS
                    else None
                ),
                "summary": None,  # unused
            },
            store=request.store,
            temperature=request.temperature,
            text=text_format,  # TODO(v): Expand coverage per https://platform.openai.com/docs/api-reference/responses/list
            top_p=request.top_p,
            truncation=request.truncation,
            user=request.user,
            metadata=request.metadata or {},
        )


class RequestResponseMetadata(BaseModel):
    """Metadata for request/response tracking."""

    request_id: str
    final_usage_info: Optional[UsageInfo] = None


@dataclass
class MessageProcessingResult:
    prompt: str
    prompt_ids: Union[str, List[int]]
    image_data: Optional[Any]
    audio_data: Optional[Any]
    video_data: Optional[Any]
    modalities: List[str]
    stop: List[str]
    tool_call_constraint: Optional[ToolCallConstraint] = None
    skip_special_tokens: bool = True
    require_reasoning: bool = False


class ToolCallProcessingResult(NamedTuple):
    """Result of processing tool calls in a response."""

    tool_calls: Optional[
        List[Any]
    ]  # List of ToolCall objects or None if parsing failed
    remaining_text: str  # Text remaining after parsing tool calls
    finish_reason: Dict[str, Any]  # Updated finish reason dictionary


class ResponseReasoningTextContent(BaseModel):
    text: str
    type: Literal["reasoning_text"] = "reasoning_text"


ResponseInputOutputItem: TypeAlias = Union[
    ResponseInputItemParam, "ResponseReasoningItem", ResponseFunctionToolCall
]


# ================== Transcription API Protocol Definitions ==================


class TranscriptionRequest(BaseModel):
    """Request model for audio transcription (OpenAI-compatible)."""

    model: str = DEFAULT_MODEL_NAME
    language: Optional[str] = None
    response_format: str = "json"
    temperature: float = 0.0
    timestamp_granularities: Optional[List[str]] = None
    stream: bool = False
    # Internal fields (not from API)
    audio_data: Optional[bytes] = None
    audio_duration_s: float = 0.0


class TranscriptionUsage(BaseModel):
    """Usage info for transcription response (duration-based)."""

    type: Literal["duration"] = "duration"
    seconds: int  # Audio duration in seconds (rounded up)


class TranscriptionResponse(BaseModel):
    """Non-streaming transcription response (OpenAI-compatible)."""

    text: str
    usage: Optional[TranscriptionUsage] = None


class TranscriptionSegment(BaseModel):
    """A segment with timestamp information."""

    id: int
    start: float
    end: float
    text: str


class TranscriptionVerboseResponse(BaseModel):
    """Verbose transcription response with timestamps (OpenAI-compatible)."""

    task: str = "transcribe"
    language: Optional[str] = None
    duration: Optional[float] = None
    text: str
    segments: List[TranscriptionSegment] = []
    usage: Optional[TranscriptionUsage] = None


class TranscriptionStreamChoice(BaseModel):
    """Delta content for streaming transcription."""

    delta: DeltaMessage
    finish_reason: Optional[str] = None


class TranscriptionStreamResponse(BaseModel):
    """Streaming transcription chunk (OpenAI-compatible)."""

    id: str = Field(default_factory=lambda: f"trsc-{uuid.uuid4().hex}")
    object: Literal["transcription.chunk"] = "transcription.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[TranscriptionStreamChoice]
    usage: Optional[UsageInfo] = None
