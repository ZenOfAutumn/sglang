# =============================================================================
# 推理（reasoning / thinking）内容解析器
#
# 很多大模型会在正式回答前输出一段“思考/推理”内容，并用特殊标记
# 包裹（如 <think>...</think>、[THINK]...[/THINK]、◁think▷...◁/think▷ 等）。
# 本模块负责把模型输出拆分为两部分：
#   - reasoning_text：推理/思考内容（可单独展示为 reasoning_content）
#   - normal_text：面向用户的正式回答
# 同时提供两套接口：一次性解析（非流式）与增量解析（流式）。
# =============================================================================

from typing import Dict, Optional, Tuple, Type

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.parser.harmony_parser import HarmonyParser


class StreamingParseResult:
    """解析结果容器：分别保存正文与推理内容。"""

    def __init__(
        self,
        normal_text: Optional[str] = None,
        reasoning_text: Optional[str] = None,
    ):
        # 正式回答文本。
        self.normal_text = normal_text or ""
        # 推理/思考文本。
        self.reasoning_text = reasoning_text or ""


class BaseReasoningFormatDetector:
    """推理格式检测器基类：提供一次性解析与流式增量解析两套接口。"""

    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
        tool_start_token: Optional[str] = None,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        # 推理块的开始/结束标记（如 <think> / </think>）。
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        # 可选的“工具调用开始标记”：某些模型未发出 </think> 就转入工具调用。
        self.tool_start_token = tool_start_token
        # 是否处于推理状态（force_reasoning 表示默认一开始就是推理，如 R1）。
        self._in_reasoning = force_reasoning
        # 是否边生成边流式输出推理内容。
        self.stream_reasoning = stream_reasoning

        # 流式解析缓冲区；是否已剔除起始 <think> 标记。
        self._buffer = ""
        self.stripped_think_start = False

        # continue_final_message：续写最后一条 assistant 消息时，需考虑已有的前文。
        self.continue_final_message = continue_final_message
        if self.continue_final_message:
            self.previous_content = previous_content
            self.previous_count = len(previous_content)
        else:
            self.previous_content = ""
            self.previous_count = 0

        # 根据前文中已出现的起始/结束标记，修正初始推理状态。
        if self.think_start_token in self.previous_content:
            self._in_reasoning = True
        if self.think_end_token in self.previous_content:
            self._in_reasoning = False

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """一次性（非流式）解析：从完整文本中拆出推理内容与正文。"""
        # 是否处于推理：强制推理，或文本中出现了 <think> 起始标记。
        in_reasoning = self._in_reasoning or self.think_start_token in text

        # 不在推理块，整段都是正文。
        if not in_reasoning:
            return StreamingParseResult(normal_text=text)

        # 进入推理块：先去掉起始标记并去首尾空白。
        processed_text = text.replace(self.think_start_token, "").strip()

        # 没有结束标记（本文与前文都没有）：推理尚未结束。
        if (
            self.think_end_token not in processed_text
            and self.think_end_token not in self.previous_content
        ):
            # 检查是否被工具调用标记打断（未发 </think> 就转入工具调用）。
            if (
                in_reasoning
                and self.tool_start_token is not None
                and self.tool_start_token in processed_text
            ):
                # 在第一个工具标记处切分：前半是推理，后半（含标记）作为正文保留。
                tool_idx = processed_text.find(self.tool_start_token)
                reasoning_text = processed_text[:tool_idx].strip()
                normal_text = processed_text[tool_idx:]
                return StreamingParseResult(
                    normal_text=normal_text, reasoning_text=reasoning_text
                )
            # 否则视为推理在结束标记前被截断，整段都是推理。
            return StreamingParseResult(reasoning_text=processed_text)

        # 有结束标记：以其为界分割，前半为推理，后半为正文。
        if self.think_end_token in processed_text:
            splits = processed_text.split(self.think_end_token, maxsplit=1)
            reasoning_text = splits[0]
            normal_text = splits[1].strip()

            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text
            )
        else:
            # 结束标记在前文里（continue_final_message=True 场景），本次全是正文。
            return StreamingParseResult(normal_text=processed_text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """流式增量解析：处理不完整的推理标记与内容。

        stream_reasoning=False：累积推理内容，直到遇到结束标记才输出。
        stream_reasoning=True：推理内容随到随输出。
        """
        # 累加新增文本到缓冲区。
        self._buffer += new_text
        current_text = self._buffer

        # 若当前文本是某个标记的不完整前缀（可能跨 chunk），先继续缓冲等后续。
        tokens_to_check = [self.think_start_token, self.think_end_token]
        if self.tool_start_token:
            tokens_to_check.append(self.tool_start_token)
        if any(
            token.startswith(current_text) and token != current_text
            for token in tokens_to_check
        ):
            return StreamingParseResult()

        # 若出现起始 <think> 标记且尚未剔除，则剔除并进入推理状态。
        if not self.stripped_think_start and self.think_start_token in current_text:
            current_text = current_text.replace(self.think_start_token, "")
            self.stripped_think_start = True
            self._in_reasoning = True

        # 处理推理块结束：在推理中且出现结束标记。
        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)

            # 结束标记之前为推理，之后为正文；清空缓冲并退出推理状态。
            end_idx = current_text.find(self.think_end_token)
            reasoning_text = current_text[:end_idx]

            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]

            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text.rstrip()
            )

        # 仍在推理中。
        if self._in_reasoning:
            # 检查是否被工具调用标记打断：是则切出推理并转交正文。
            if self.tool_start_token and self.tool_start_token in current_text:
                tool_idx = current_text.find(self.tool_start_token)
                reasoning_text = current_text[:tool_idx]
                normal_text = current_text[tool_idx:]
                self._buffer = ""
                self._in_reasoning = False
                return StreamingParseResult(
                    normal_text=normal_text, reasoning_text=reasoning_text
                )
            if self.stream_reasoning:
                # 流式：立即输出已缓冲的推理内容并清空缓冲。
                self._buffer = ""
                return StreamingParseResult(reasoning_text=current_text)
            else:
                # 非流式：继续累积，不输出。
                return StreamingParseResult()

        # 不在推理块：作为正文输出。
        if not self._in_reasoning:
            self._buffer = ""
            return StreamingParseResult(normal_text=current_text)

        return StreamingParseResult()


class DeepSeekR1Detector(BaseReasoningFormatDetector):
    """
    DeepSeek-R1 模型检测器。推理格式：(<think>)*(.*)</think>
    把 </think> 之前的文本作为 reasoning_text，之后的作为 normal_text。

    支持：R1（不带 <think> 起始标记、默认即推理）、R1-0528（带 <think> 起始标记）。
    以下为原英文说明：
    Returns all the text before the </think> tag as `reasoning_text`
    and the rest of the text as `normal_text`.

    Supported models:
      - DeepSeek-R1: Always generates thinking content without <think> start tag
      - DeepSeek-R1-0528: Generates thinking content with <think> start tag

    Format patterns:
      - DeepSeek-R1: "I need to think about this...</think>The answer is 42."
      - DeepSeek-R1-0528: "<think>I need to think about this...</think>The answer is 42."

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        # DeepSeek-R1 默认一开始就处于推理（force_reasoning=True），直到 </think>。
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )
        # https://github.com/sgl-project/sglang/pull/3202#discussion_r1950153599


class Qwen3Detector(BaseReasoningFormatDetector):
    """
    Qwen3 系列检测器（如 Qwen/Qwen3-235B-A22B）。推理格式：(<think>)*(.*)</think>
    可通过请求参数 enable_thinking 切换思考/普通模式。以下为原英文说明：

    Qwen3 models released before 07/2025 supports switching between thinking mode and normal
    mode using `enable_thinking` parameter in the request parameter.
      - enable_thinking=True: "<think>reasoning content</think>The answer is 42."
      - enable_thinking=False: "The answer is 42." (no thinking tokens)

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )


class KimiDetector(BaseReasoningFormatDetector):
    """
    Kimi Thinking 模型检测器。推理格式使用特殊字符：◁think▷*(.*)◁/think▷
    把 ◁/think▷ 之前作为 reasoning_text，之后作为 normal_text。
    Returns all the text before the ◁/think▷ tag as `reasoning_text`
    and the rest of the text as `normal_text`.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "◁think▷",
            "◁/think▷",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )


class KimiK2Detector(BaseReasoningFormatDetector):
    """
    Kimi K2 检测器。推理格式：(<think>)*(.*)</think>
    特点：K2 可能在发出 </think> 之前就用 <|tool_calls_section_begin|> 转入工具调用。
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<|tool_calls_section_begin|>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )


class Glm45Detector(BaseReasoningFormatDetector):
    """
    GLM-4.5 检测器。推理格式：(<think>)*(.*)</think>
    GLM-4.5 用 <tool_call> 作为工具起始标记，从推理模式切到普通模式。

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_call>",
        )


class GptOssDetector(BaseReasoningFormatDetector):
    """
    GPT-OSS（T4 风格 harmony 格式）检测器，内部委托专用的 HarmonyParser 解析。
    输出由 <|channel|>analysis<|message|> ... <|end|> 等结构化标记组成。
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "<|channel|>analysis<|message|>",
            "<|end|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )
        self.parser = HarmonyParser()

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        # 用 HarmonyParser 解析事件流，并以空字符串冲刷缓冲（一次性解析）。
        events = self.parser.parse(text)
        events += self.parser.parse("")

        # 收集 reasoning 事件作为推理文本。
        reasoning_text = "".join(
            [e.content for e in events if e.event_type == "reasoning"]
        )
        normal_parts = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":
                # 工具调用事件保留 raw_text（含结构标记），供后续函数调用解析器识别。
                normal_parts.append(e.raw_text if e.raw_text else e.content)
        normal_text = "".join(normal_parts)

        return StreamingParseResult(
            normal_text=normal_text,
            reasoning_text=reasoning_text,
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        # 流式：逐块交给 HarmonyParser，转换为 reasoning / normal / tool_call 事件。
        events = self.parser.parse(new_text)

        reasoning_text = "".join(
            [e.content for e in events if e.event_type == "reasoning"]
        )
        normal_parts = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":
                # Use raw_text to preserve structural markers for function call detector
                normal_parts.append(e.raw_text if e.raw_text else e.content)
        normal_text = "".join(normal_parts)

        return StreamingParseResult(
            normal_text=normal_text,
            reasoning_text=reasoning_text,
        )


class MiniMaxAppendThinkDetector(BaseReasoningFormatDetector):
    """
    MiniMax 专用：在输出开头补上 <think> 标记（模型输出不自带起始标记）。
    注：本检测器不拆分推理/正文，只负责补标记，后续交由上层处理。
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        # scheduler.py need `reasoning_parser.detector.think_end_token`
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )
        self.is_first_chunk = False

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        # 仅在首块前补上 <think> 标记。
        if not self.is_first_chunk:
            self.is_first_chunk = True
            new_text = self.think_start_token + new_text
        return StreamingParseResult(normal_text=new_text)

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        # 一次性：直接在文本开头拼上 <think> 标记。
        return StreamingParseResult(normal_text=self.think_start_token + text)


class Nemotron3Detector(BaseReasoningFormatDetector):
    """
    Nemotron3 检测器。推理格式与 DeepSeek-R1 相同：(<think>)*(.*)</think>。
    额外支持 force_nonempty_content：当正文为空时，把推理与正文互换，避免正文为空。
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )
        self._force_nonempty_content = force_nonempty_content

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        ret = super().detect_and_parse(text)
        # 若要求正文非空但解析出的正文为空，则与推理互换。
        if self._force_nonempty_content and not ret.normal_text:
            ret.normal_text, ret.reasoning_text = ret.reasoning_text, ret.normal_text
        return ret


class MistralDetector(BaseReasoningFormatDetector):
    """
    带推理的 Mistral 模型检测器（如 Mistral-Small-4-119B-2603）。
    推理格式：[THINK]推理内容[/THINK]回答。
    推理是可选的：仅当 reasoning_effort="high" 时出现；="none" 时直接输出无思考标记。
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "[THINK]",
            "[/THINK]",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )


class ReasoningParser:
    """推理解析统一入口：根据模型类型选择对应检测器，处理流式与非流式两种场景。

    参数：
        model_type: 模型类型（决定用哪个检测器）。
        stream_reasoning: False 累积到推理完成才输出；True 随到随输出。
    """

    # 模型类型 → 检测器类的映射表（多个模型可复用同一检测器，如 deepseek-v3/mimo/qwen3 都用 Qwen3Detector）。
    DetectorMap: Dict[str, Type[BaseReasoningFormatDetector]] = {
        "deepseek-r1": DeepSeekR1Detector,
        "deepseek-v3": Qwen3Detector,
        "glm45": Glm45Detector,
        "gpt-oss": GptOssDetector,
        "kimi": KimiDetector,
        "kimi_k2": KimiK2Detector,
        "mimo": Qwen3Detector,
        "qwen3": Qwen3Detector,
        "qwen3-thinking": Qwen3Detector,
        "minimax": Qwen3Detector,
        "minimax-append-think": MiniMaxAppendThinkDetector,
        "step3": DeepSeekR1Detector,
        "step3p5": DeepSeekR1Detector,
        "mistral": MistralDetector,
        "nemotron_3": Nemotron3Detector,
        "interns1": Qwen3Detector,
    }

    def __init__(
        self,
        model_type: Optional[str] = None,
        stream_reasoning: bool = True,
        force_reasoning: Optional[bool] = None,
        request: ChatCompletionRequest = None,
    ):
        if not model_type:
            raise ValueError("Model type must be specified")

        # 根据模型类型（不区分大小写）查找检测器类。
        detector_class = self.DetectorMap.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        # 特殊情况：这几类模型强制开启推理。
        if model_type.lower() in {"qwen3-thinking", "gpt-oss", "minimax"}:
            force_reasoning = True

        # 仅在显式设置时才传 force_reasoning，否则使用检测器自身默认值。
        kwargs = {"stream_reasoning": stream_reasoning}
        if force_reasoning is not None:
            kwargs["force_reasoning"] = force_reasoning

        # 续写场景：最后一条是 assistant 且 continue_final_message=True，需把前文传给检测器。
        if (
            request is not None
            and isinstance(request, ChatCompletionRequest)
            and request.continue_final_message
            and request.messages[-1].role == "assistant"
        ):
            kwargs["continue_final_message"] = True
            kwargs["previous_content"] = request.messages[-1].content

        # 模板参数请求强制正文非空时（仅 Nemotron3 等支持），传递该标志。
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
        if chat_template_kwargs.get("force_nonempty_content") is True:
            kwargs["force_nonempty_content"] = True

        # 实例化具体检测器。
        self.detector = detector_class(**kwargs)

    def parse_non_stream(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """非流式调用：一次性解析，返回 (推理文本, 正文)。"""
        ret = self.detector.detect_and_parse(full_text)
        return ret.reasoning_text, ret.normal_text

    def parse_stream_chunk(
        self, chunk_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """流式调用：增量解析，返回 (推理文本, 正文)。"""
        ret = self.detector.parse_streaming_increment(chunk_text)
        return ret.reasoning_text, ret.normal_text
