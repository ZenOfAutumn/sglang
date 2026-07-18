import json
import logging
import re

from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import _find_common_prefix, _partial_json_loads

logger = logging.getLogger(__name__)


class DeepSeekV32Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V3.2 model function call format.

    The DeepSeek V3.2 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    中译：DeepSeek-V3.2 模型工具调用格式的解析器（V4 解析器即继承本类）。
    DeepSeek-V3.2 使用类 XML 的 DSML（DeepSeek Markup Language）标签划定工具调用，
    标签分隔符为全角竖线 `｜DSML｜`。支持两种参数格式：

    中译：格式 1——XML 参数标签（每个参数一个 <｜DSML｜parameter> 标签）：
    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    中译：格式 2——直接 JSON（所有参数写成一个 JSON 对象）：
    Format 2 - Direct JSON:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Examples:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>

    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜function_calls>` and `</｜DSML｜function_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V3.2 format specification

    中译：关键组成：
    - 工具调用整体段：包在 `<｜DSML｜function_calls>` 与 `</｜DSML｜function_calls>` 之间；
    - 单个工具调用：包在 `<｜DSML｜invoke name="...">` 与 `</｜DSML｜invoke>` 之间；
    - 参数：XML 标签或直接 JSON 两种格式均可；
    - 支持一次多个工具调用。

    参考：DeepSeek-V3.2 格式规范。
    """

    def __init__(self):
        super().__init__()
        # 中译：bot_token / eot_token——工具调用整体段的起始 / 结束标记；
        #       invoke_end_token——单个工具调用的结束标记。
        self.bot_token = "<｜DSML｜function_calls>"
        self.eot_token = "</｜DSML｜function_calls>"
        self.invoke_end_token = "</｜DSML｜invoke>"
        # 中译：parameter_regex——匹配「完整」的单个 XML 参数标签，捕获组依次为
        #       参数名 / string 标志 / 值（.*? 非贪婪）。
        self.parameter_regex = r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</｜DSML｜parameter>'
        # 中译：partial_parameter_regex——匹配「未闭合」的参数标签（值直到文本末尾 $），
        #       用于流式下参数还没流完的部分解析。
        self.partial_parameter_regex = (
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*)$'
        )
        # 中译：非贪婪提取起止标记之间的整个 function_calls 段内容。
        self.function_calls_regex = (
            r"<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>"
        )
        # Long-form `<｜DSML｜invoke name="x">...</｜DSML｜invoke>` and the
        # self-closing `<｜DSML｜invoke name="x"/>` shape V4 emits for zero-arg
        # tools. The `end` group is empty when the closer hasn't streamed in.
        # 中译：invoke_regex 同时匹配两种 invoke 形态：
        #   ① 长形式 `<｜DSML｜invoke name="x">...</｜DSML｜invoke>`；
        #   ② 自闭合 `<｜DSML｜invoke name="x"/>`（V4 对零参数工具会产出这种）。
        #   流式下当结束标记尚未到达时，end 捕获组为空（用于判断是否已完整）。
        self.invoke_regex = (
            r'<｜DSML｜invoke\s+name="(?P<name>[^"]+)"\s*'
            r"(?:(?P<self_close>/>)"
            r"|>(?P<body>.*?)(?P<end>(?:</｜DSML｜invoke>|$)))"
        )
        # 中译：两个「结束标记的逐段前缀」列表。流式下结束标记可能被切成多段陆续到达，
        #       部分前缀会被误当作参数值捕获；据此从参数值末尾剔除这些残缺前缀。
        self.prefix_parameter_end_call = ["</", "｜DSML｜", "parameter"]
        self.prefix_invoke_end_call = ["</", "｜DSML｜", "inv", "oke"]
        # 中译：当前正在处理的工具调用索引，-1 表示尚未开始任何工具调用。
        self.current_tool_id = -1

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v32 format tool call.

        中译：判断文本是否含 DeepSeek-V3.2 格式的工具调用（整体起始标记，
        或单个 invoke 起始标记）。
        """
        return self.bot_token in text or "<｜DSML｜invoke" in text

    @staticmethod
    def _unpack_invoke_match(m: "re.Match[str]") -> tuple[str, str, bool]:
        """Returns (name, body, is_complete) for an invoke_regex match.

        Self-closing invokes have empty body and are always complete.
        Long-form bodies are always strings (possibly empty); they're
        incomplete when matched against `$` because the closing tag
        hasn't streamed in yet.

        中译：从 invoke_regex 的匹配结果中解包出 (name, body, is_complete)。
        - 自闭合 invoke（name="x"/）：body 为空且总是完整（is_complete=True）；
        - 长形式 invoke：body 总是字符串（可能为空），当它是因匹配到 `$`
          （而非真正的结束标记）时表示未完成，is_complete=False。
        """
        name = m.group("name").strip()
        # 中译：命中自闭合形式——无参数且已完整。
        if m.group("self_close"):
            return name, "", True
        # 中译：长形式——end 组非空（即匹配到真正的 </｜DSML｜invoke>）才算完整。
        return name, m.group("body"), bool(m.group("end"))

    def _parse_parameters_from_xml(
        self, invoke_content: str, allow_partial: bool = False
    ) -> str:
        """
        Parse parameters from either XML-like format or JSON format to str.

        Supports two formats:
        1. XML parameter tags: <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter>
        2. Direct JSON: { "key": "value" }
        """
        # First, try to parse as direct JSON (new format)
        invoke_content_stripped = invoke_content.strip()
        if invoke_content_stripped.startswith("{"):
            if allow_partial:
                # Remove incomplete invoke end call prefix in case they are captured by param
                for token in reversed(self.prefix_invoke_end_call):
                    invoke_content_stripped = invoke_content_stripped.rstrip(token)
                return invoke_content_stripped
            elif invoke_content_stripped.endswith("}"):
                return invoke_content_stripped

        # Fall back to XML parameter tag parsing (original format)
        parameters = {}
        # Find all complete parameter matches
        param_matches = list(
            re.finditer(self.parameter_regex, invoke_content, re.DOTALL)
        )

        last_match_end = 0
        for match in param_matches:
            param_name = match.group(1)
            param_type = match.group(2)
            param_value = match.group(3)
            last_match_end = match.end()

            # Convert value based on type
            if param_type == "true":  # string type
                parameters[param_name] = param_value.strip()
            else:
                # Try to parse as JSON for other types
                try:
                    parameters[param_name] = json.loads(param_value.strip())
                except (json.JSONDecodeError, ValueError):
                    parameters[param_name] = param_value.strip()

        # If allowed, try to parse a partial parameter at the end
        if allow_partial:
            remaining_content = invoke_content[last_match_end:]

            # Remove incomplete parameter_end_call prefix in case they are captured by param
            for token in reversed(self.prefix_parameter_end_call):
                remaining_content = remaining_content.rstrip(token)

            # Match start of a parameter tag + value (potentially incomplete)
            # Regex: <tag name="..." string="...">VALUE... (no end tag)
            partial_match = re.search(
                self.partial_parameter_regex, remaining_content, re.DOTALL
            )

            if partial_match and (param_value := partial_match.group(3)):
                param_name = partial_match.group(1)
                if partial_match.group(2) == "true":
                    parameters[param_name] = param_value.strip()
                else:
                    try:
                        parameters[param_name] = _partial_json_loads(
                            param_value, Allow.ALL
                        )[0]
                    except json.JSONDecodeError:
                        parameters[param_name] = param_value.strip()

        return json.dumps(parameters, ensure_ascii=False)

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.

        中译：一次性（非流式）解析：检测并解析文本中的工具调用。
        :param text: 待解析的完整文本。
        :param tools: 可用工具列表。
        :return: StreamingParseResult，包含剩余普通文本与解析出的调用列表。
        """
        # 中译：定位工具调用起始标记；起始标记之前的部分为普通文本（去掉尾部双换行）。
        idx = text.find(self.bot_token)
        normal_text = text[:idx].removesuffix("\n\n") if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        try:
            # Extract content between function_calls tags
            # 中译：提取 function_calls 标签之间的整段内容。
            function_calls_match = re.search(
                self.function_calls_regex,
                text,
                re.DOTALL,
            )
            if not function_calls_match:
                return StreamingParseResult(normal_text=normal_text, calls=[])

            function_calls_content = function_calls_match.group(1)

            # Find all invoke blocks
            # 中译：逐个匹配所有 invoke 块（支持一次多个工具调用）。
            for invoke_match in re.finditer(
                self.invoke_regex, function_calls_content, re.DOTALL
            ):
                func_name, invoke_content, _ = self._unpack_invoke_match(invoke_match)
                func_args = self._parse_parameters_from_xml(invoke_content)
                # construct match_result for parse_base_json
                # 中译：组装成 {name, parameters} 交给基类 parse_base_json（校验工具存在并转 ToolCallItem）。
                match_result = {"name": func_name, "parameters": json.loads(func_args)}
                calls.extend(self.parse_base_json(match_result, tools))

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            # 中译：解析失败时降级——把原文本当作普通文本返回，避免丢失内容。
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV32 format.
        Supports multiple consecutive invoke blocks and argument streaming.

        中译：DeepSeek-V3.2 格式的流式增量解析。支持多个连续 invoke 块与参数流式传输。
        核心思路：把 new_text 追加到跨 chunk 缓冲 _buffer，尝试匹配 invoke 块；
        名字先发，参数按「稳定前缀 diff」逐步增量发送，直到看到结束标记才完成一个调用。
        """
        # 中译：先把新到达的文本追加进跨 chunk 缓冲区。
        self._buffer += new_text
        current_text = self._buffer

        # Check if buffer contains any DSML markers or ends with potential tag prefix
        # This handles partial/streaming DSML content
        # 中译：判断缓冲是否含（部分）DSML 标记，以处理流式下标记分多段到达的情况。
        dsml_markers = ["｜DSML｜", "<｜", "</｜"]
        potentially_dsml = any(marker in current_text for marker in dsml_markers)

        # Also check if text ends with start of a tag (to handle "<" arriving separately)
        # 中译：再判断文本是否以标签起始片段结尾（处理 "<" 单独先到达的情况）。
        dsml_prefixes = ["<", "<｜", "</", "</｜"]
        ends_with_prefix = any(
            current_text.rstrip().endswith(prefix) for prefix in dsml_prefixes
        )

        # 中译：既不含工具调用、也不像 DSML 片段时，本段全是普通文本：清空缓冲并直接返回。
        if (
            not self.has_tool_call(current_text)
            and not potentially_dsml
            and not ends_with_prefix
        ):
            self._buffer = ""
            for e_token in [self.eot_token, self.invoke_end_token]:
                if e_token in current_text:
                    current_text = current_text.replace(e_token, "")
            return StreamingParseResult(normal_text=current_text)

        all_calls: list[ToolCallItem] = []
        try:
            # Loop to handle multiple consecutive invoke blocks
            # 中译：循环处理多个连续的 invoke 块（一次 chunk 里可能完成不止一个工具调用）。
            while True:
                # Try to match an invoke block (may be partial)
                # 中译：尝试匹配一个 invoke 块（可能是尚未闭合的部分）。
                invoke_match = re.search(
                    pattern=self.invoke_regex,
                    string=current_text,
                    flags=re.DOTALL,
                )
                if not invoke_match:
                    break

                func_name, invoke_content, is_tool_end = self._unpack_invoke_match(
                    invoke_match
                )

                # Initialize state if this is the first tool call
                # 中译：首个工具调用时初始化状态（当前 id、已记录参数、已发送参数）。
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Ensure arrays are large enough for current tool
                # 中译：确保状态数组长度足够容纳当前工具索引。
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                # 1. Send tool name if not sent yet
                # 中译：① 若工具名尚未发送，先吐出一个仅带 name 的增量（parameters 为空）。
                if not self.current_tool_name_sent:
                    all_calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True

                # 2. Parse current parameters (partial or complete)
                # 中译：② 解析当前参数（未结束时允许部分解析）。
                current_params = self._parse_parameters_from_xml(
                    invoke_content, allow_partial=not is_tool_end
                )

                # 3. Calculate and send incremental arguments
                # 中译：③ 计算并发送参数增量。sent_len 为已发送长度，prev_params 为上一次解析值。
                sent_len = len(self.streamed_args_for_tool[self.current_tool_id])
                prev_params = self.prev_tool_call_arr[self.current_tool_id].get(
                    "arguments"
                )

                argument_diff = None

                if is_tool_end:
                    # If complete, send everything remaining
                    # 中译：已完成，把剩下的参数全部发出。
                    argument_diff = current_params[sent_len:]
                elif prev_params is not None:
                    # If partial, send stable prefix diff
                    # 中译：部分解析时，只发送与上次值的「稳定公共前缀」中超出已发送部分，
                    #       避免把会变化的尾部提前发出。
                    if current_params != prev_params:
                        prefix = _find_common_prefix(current_params, prev_params)
                        if len(prefix) > sent_len:
                            argument_diff = prefix[sent_len:]

                if argument_diff:
                    all_calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=argument_diff,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff

                # Update the stored arguments
                # 中译：更新已记录的参数快照（供下一次算 diff 用）。
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": func_name,
                    "arguments": current_params,
                }

                # Check if tool call is complete (has closing tag)
                # 中译：工具调用已完成（看到结束标记）时——
                if is_tool_end:
                    # Remove the completed tool call from buffer
                    # 中译：从缓冲中移除已完成的调用，剩余留给下一轮。
                    self._buffer = current_text[invoke_match.end() :]
                    current_text = self._buffer  # Update for next iteration

                    # Move to next tool call
                    # 中译：推进到下一个工具调用，重置「名字已发送」标志。
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False

                    # Continue loop to check for more invoke blocks
                    continue
                else:
                    # Tool call not complete yet, don't return anything
                    # Wait for more chunks until we see </｜DSML｜invoke>
                    # 中译：尚未完成，不再继续，等后续 chunk 直到出现 </｜DSML｜invoke>。
                    break

            # No more invoke blocks found
            # 中译：本轮无更多 invoke 块，返回本次累积的调用增量（普通文本为空）。
            return StreamingParseResult(normal_text="", calls=all_calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def structure_info(self) -> _GetInfoFunc:
        # 中译：返回「工具名 → StructureInfo」的回调，供生成结构化标签约束：
        #       begin/end 为单个 invoke 的起止标记，trigger 为触发约束的前缀。
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}">',
            end="</｜DSML｜invoke>",
            trigger="<｜DSML｜invoke",
        )

    def get_structural_tag_name(self) -> str:
        # 中译：返回结构化标签名称（供上层识别本模型的原生 structural_tag）。
        return "deepseek_v3_2"

