import logging
from typing import Dict, List, Literal, Optional, Set, Tuple, Type, Union

from sglang.srt.entrypoints.openai.protocol import (
    LegacyStructuralTagResponseFormat,
    StructuralTagResponseFormat,
    StructuresResponseFormat,
    Tool,
    ToolCallConstraint,
    ToolChoice,
)
from sglang.srt.environ import ToolStrictLevel, envs
from sglang.srt.function_call.apertus2509_detector import Apertus2509Detector
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.cohere_command4_detector import CohereCommand4Detector
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.gemma4_detector import Gemma4Detector
from sglang.srt.function_call.gigachat3_detector import GigaChat3Detector
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sglang.srt.function_call.gpt_oss_detector import GptOssDetector
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.srt.function_call.hunyuan_detector import HunyuanDetector
from sglang.srt.function_call.internlm_detector import InternlmDetector
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.srt.function_call.lfm2_detector import Lfm2Detector
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.srt.function_call.minicpm5_detector import MiniCPM5Detector
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.srt.function_call.poolside_v1_detector import PoolsideV1Detector
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.step3_detector import Step3Detector
from sglang.srt.function_call.trinity_detector import TrinityDetector
from sglang.srt.function_call.utils import (
    _get_tool_schema_defs,
    get_json_schema_constraint,
)

logger = logging.getLogger(__name__)


class FunctionCallParser:
    """
    Parser for function/tool calls in model outputs.

    This class handles both streaming and non-streaming parsing of function calls using a detector.
    In streaming scenarios, each time new_text is received, it calls detector.parse_streaming_increment
    and returns the resulting normal_text and calls to the upper layer (or SSE).

    中译：模型输出中「工具调用（function/tool call）」的顶层解析器。

    它是整个 function_call 框架的对外入口，本身不含具体格式的解析规则，而是根据
    `tool_call_parser` 名称选出对应模型的 detector（见 ToolCallParserEnum 映射表），
    再把解析工作委托给该 detector。同时支持两种模式：
      - 非流式：parse_non_stream，一次性解析完整输出；
      - 流式：parse_stream_chunk，每收到一段 new_text 就调用
        detector.parse_streaming_increment 做增量解析，把 normal_text 与 calls
        逐步返回给上层（或 SSE 流）。
    """

    # 中译：工具解析器名称 → detector 类的注册表。用户通过 --tool-call-parser 传入的名称
    #       在此查表得到对应模型的解析器类（一种模型格式对应一个 detector）。
    #       注意多个名称可映射到同一实现（如 glm/glm45 共用、step3p5 复用 Qwen3Coder）。
    ToolCallParserEnum: Dict[str, Type[BaseFormatDetector]] = {
        "apertus2509": Apertus2509Detector,
        "cohere_command4": CohereCommand4Detector,
        "deepseekv3": DeepSeekV3Detector,
        "deepseekv31": DeepSeekV31Detector,
        "deepseekv32": DeepSeekV32Detector,
        "deepseekv4": DeepSeekV4Detector,
        "glm": Glm4MoeDetector,
        "glm45": Glm4MoeDetector,
        "glm47": Glm47MoeDetector,
        "gpt-oss": GptOssDetector,
        "kimi_k2": KimiK2Detector,
        "lfm2": Lfm2Detector,
        "llama3": Llama32Detector,
        "mimo": MiMoDetector,
        "minicpm5": MiniCPM5Detector,
        "mistral": MistralDetector,
        "poolside_v1": PoolsideV1Detector,
        "pythonic": PythonicDetector,
        "qwen": Qwen25Detector,
        "qwen25": Qwen25Detector,
        "qwen3_coder": Qwen3CoderDetector,
        "step3": Step3Detector,
        "step3p5": Qwen3CoderDetector,
        "minimax-m2": MinimaxM2Detector,
        "trinity": TrinityDetector,
        "interns1": InternlmDetector,
        "hermes": HermesDetector,
        "hunyuan": HunyuanDetector,
        "gigachat3": GigaChat3Detector,
        "gemma4": Gemma4Detector,
    }

    def __init__(self, tools: List[Tool], tool_call_parser: str):
        # 中译：按名称从注册表取出 detector 类并实例化；名称不在表中则报错，
        #       避免静默地用错误格式解析导致 tool_calls 解析为空。
        detector_class = self.ToolCallParserEnum.get(tool_call_parser)
        if detector_class:
            detector = detector_class()
        else:
            raise ValueError(f"Unsupported tool_call_parser: {tool_call_parser}")

        # 中译：self.detector —— 实际执行解析的模型专用检测器；
        #       self.tools —— 本次请求可用的工具列表（无工具时各解析方法直接短路返回）；
        #       self.tool_strict_level —— 工具参数约束严格级别（由环境变量控制，
        #       影响是否用 schema 约束模型输出，见 get_legacy_structural_tag）。
        self.detector = detector
        self.tools = tools
        self.tool_strict_level = envs.SGLANG_TOOL_STRICT_LEVEL.get()

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a tool call in the format supported by this parser.
        This delegates to the detector's implementation.

        Args:
            text: The text to check for tool calls

        Returns:
            True if the text contains a tool call, False otherwise

        中译：判断给定文本是否包含本解析器所支持格式的工具调用。
        实际判断委托给 detector（通常是检测其起始标记，如 <tool_call> / 特殊 token）。

        参数：
            text：待检查的文本。
        返回：
            含工具调用返回 True，否则 False。
        """
        # 中译：本次请求未提供任何工具时，不可能有工具调用，直接返回 False。
        if not self.tools:
            return False
        return self.detector.has_tool_call(text)

    def parse_non_stream(self, full_text: str) -> Tuple[str, list[ToolCallItem]]:
        """
        One-time parsing of the full text to extract tool calls.

        Args:
            full_text: The complete text to parse

        Returns:
            A tuple containing:
            - The remaining text after parsing that was not consumed by the detector (can be treated as normal text)
            - A list of tool calls parsed from the text

        中译：对完整输出文本做一次性（非流式）解析，抽取其中的工具调用。

        参数：
            full_text：待解析的完整文本。
        返回：
            二元组：
            - 解析后未被 detector 消费的剩余文本（可当作普通文本展示给用户）；
            - 从文本中解析出的工具调用列表。
        """
        # 中译：无工具则原样返回文本、空调用列表。
        if not self.tools:
            return full_text, []
        # 中译：委托 detector 解析，得到 normal_text（普通文本）与 calls（工具调用）。
        parsed_result = self.detector.detect_and_parse(full_text, self.tools)
        tool_call_list = parsed_result.calls
        # 中译：解析到工具调用时返回剩余普通文本 + 调用列表；否则视为无工具调用，
        #       原样返回整段文本（避免误吞正常内容）。
        if tool_call_list:
            return parsed_result.normal_text, tool_call_list
        else:
            return full_text, []

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, list[ToolCallItem]]:
        """
        Streaming incremental parsing of chunks of text as they arrive.

        Args:
            chunk_text: The new chunk of text to parse

        Returns:
            A tuple containing:
            - The normal text that should be displayed to the user
            - A list of tool calls parsed from the chunk

        中译：随着文本分块（chunk）陆续到达，做流式增量解析。
        detector 内部维护跨 chunk 的缓冲与状态机（如半个 JSON、未闭合标记），
        因此本方法需对同一请求按到达顺序反复调用。

        参数：
            chunk_text：新到达的一段文本。
        返回：
            二元组：
            - 本次应展示给用户的普通文本（可能为空，若当前正处在工具调用中间）；
            - 从本 chunk 解析出的工具调用（可能是增量片段，如仅 name 或部分参数）。
        """
        # 中译：无工具则原样返回本 chunk、空调用列表。
        if not self.tools:
            return chunk_text, []
        # 中译：final_normal_text —— 本次要吐出的普通文本；final_calls —— 本次解析出的调用增量。
        final_normal_text = ""
        final_calls = []

        # 中译：委托 detector 做增量解析（内部会消费/累积缓冲并推进状态机）。
        sp_result = self.detector.parse_streaming_increment(chunk_text, self.tools)
        if sp_result.normal_text:
            final_normal_text = sp_result.normal_text
        if sp_result.calls:
            # 中译：有工具调用增量时，追加到结果，并以 detector 返回的 normal_text 为准
            #       （此处覆盖而非累加，因 detector 已给出本轮应输出的普通文本）。
            final_calls.extend(sp_result.calls)
            final_normal_text = sp_result.normal_text

        return final_normal_text, final_calls

    def get_legacy_structural_tag(
        self, at_least_one: bool = False
    ) -> StructuralTagResponseFormat:
        """
        Generate a structural tag response format for all available tools.

        This creates the necessary structural tags that guide the model's output format.

        Args:
            at_least_one: If True, the grammar forces at least one tool call
                (no free text allowed). Used for required/named tool_choice.

        Raises:
            ValueError: If tools have conflicting $defs schemas.

        中译：为所有可用工具生成「结构化标签（structural tag）」响应格式。
        结构化标签会在解码时约束模型输出，使其按工具的起止标记与 schema 生成，
        从而保证产出可被 detector 正确解析。

        参数：
            at_least_one：为 True 时，语法强制至少产生一个工具调用（不允许自由文本），
                用于 required / 指定具体工具的 tool_choice。
        异常：
            ValueError：当各工具的 $defs schema 存在冲突时抛出。
        """
        # 中译：构建结构化标签前，先校验各工具 $defs（JSON schema 引用定义）的一致性，
        #       冲突则直接抛错。
        _get_tool_schema_defs(self.tools)

        # 中译：tool_structures —— 每个工具对应的「起始标记 + schema + 结束标记」；
        #       tool_trigger_set —— 触发进入结构化约束的触发词集合（去重）。
        tool_structures: List[StructuresResponseFormat] = list()
        tool_trigger_set: Set[str] = set()

        # 中译：向 detector 取「按工具名生成起止标记等结构信息」的回调。
        get_structure_info = self.detector.structure_info()
        for tool in self.tools:
            function = tool.function
            name = function.name
            assert name is not None
            info = get_structure_info(name)

            # accept all if not strict, otherwise only accept the schema
            # 中译：非严格模式接受任意参数（schema 置空 {}）；严格模式才用工具声明的
            #       parameters schema 约束参数。严格与否由工具自身 strict 或全局
            #       tool_strict_level（达到 PARAMETER 级）决定。
            is_strict = (
                function.strict or self.tool_strict_level >= ToolStrictLevel.PARAMETER
            )
            schema = function.parameters if is_strict else {}

            tool_structures.append(
                StructuresResponseFormat(
                    begin=info.begin,
                    schema=schema or {},  # type: ignore
                    end=info.end,
                )
            )
            tool_trigger_set.add(info.trigger)

        # TODO(dark): move this into new structural tag format
        # This requires all grammar backend support the new format
        # 中译：TODO(dark)：待所有 grammar 后端支持新格式后，迁移到新的结构化标签格式。
        #       目前仍返回 legacy 版本。
        return LegacyStructuralTagResponseFormat(
            type="structural_tag",
            structures=tool_structures,
            triggers=list(tool_trigger_set),
            at_least_one=at_least_one,
        )

    def get_structure_constraint(
        self,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]],
        parallel_tool_calls: bool = True,
        thinking_mode: bool = False,
    ) -> Optional[ToolCallConstraint]:
        """
        Returns the appropriate structure constraint for tool calls based on the tool_choice.
        The constraint is used to guide the model's output format.

        Args:
            tool_choice: The tool choice setting from the request

        Returns:
            A tuple of (constraint_type, constraint_value) to be added to sampling parameters,
            or None if no constraint applies.

        中译：根据请求里的 tool_choice，返回合适的「工具调用约束」，用于在解码阶段
        引导/强制模型输出符合工具格式。约束会被加入采样参数（sampling params）。

        参数：
            tool_choice：请求中的工具选择设置（"auto" / "required" / 指定具体工具）。
            parallel_tool_calls：是否允许一次生成多个工具调用。
            thinking_mode：是否处于思考（reasoning）模式，影响原生结构化标签的生成。
        返回：
            (约束类型, 约束值) 二元组加入采样参数；无约束时返回 None。
        """
        # 中译：is_required —— tool_choice 为 "required" 或指定了具体工具（必须调用）。
        is_required = tool_choice == "required" or isinstance(tool_choice, ToolChoice)
        # 中译：should_constrain_auto —— "auto" 模式下是否也需要约束：仅当存在 strict 工具
        #       或全局严格级别达到 FUNCTION 级时才约束（否则 auto 允许自由发挥、不加约束）。
        should_constrain_auto = tool_choice == "auto" and (
            any(tool.function.strict for tool in self.tools)
            or self.tool_strict_level >= ToolStrictLevel.FUNCTION
        )

        # Highest priority: model-native structural_tag when available.
        # 中译：最高优先级——优先使用模型「原生」结构化标签（保留模型自身的工具调用格式）。
        try:
            if is_required or should_constrain_auto:
                structural_tag = self.detector.get_structural_tag(
                    tools=self.tools,
                    thinking_mode=thinking_mode,
                    tool_choice=tool_choice,
                )
                if structural_tag is not None:
                    return ("structural_tag", structural_tag)

                # Fallback to legacy structural tag if model-native tag is not supported.
                # 中译：模型不支持原生结构化标签时，回退到 legacy 结构化标签。
                if self.detector.supports_structural_tag():
                    # For "required"/named: always use structural_tag to preserve the
                    # model's native tool call format. Schema is only included when
                    # strict=True, per OpenAI protocol semantics.
                    # For "auto": only constrain when strict is enabled.
                    # 中译：required / 指定工具：始终用 structural_tag 以保留模型原生格式，
                    #       仅当 strict=True 时才带上 schema（遵循 OpenAI 协议语义）；
                    #       auto：仅在启用 strict 时才约束。
                    tag = self.get_legacy_structural_tag(at_least_one=is_required)
                    return ("structural_tag", tag)

            # 中译：required / 指定工具但上面未命中结构化标签时，退化为 JSON schema 约束
            #       （直接约束输出为符合工具参数 schema 的 JSON）。
            if tool_choice == "required" or isinstance(tool_choice, ToolChoice):
                json_schema = get_json_schema_constraint(
                    self.tools, tool_choice, parallel_tool_calls=parallel_tool_calls
                )
                return ("json_schema", json_schema)
        except Exception as e:
            # 中译：构建约束过程中任何异常都不应中断请求，记录错误后返回 None（不加约束）。
            logger.error(f"Error getting structure constraint: {e}")
            return None
