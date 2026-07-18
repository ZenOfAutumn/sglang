import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import orjson
from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

try:
    from xgrammar import StructuralTag, get_model_structural_tag
except ImportError:
    StructuralTag = Any
    get_model_structural_tag = None

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.environ import envs
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import (
    _find_common_prefix,
    _is_complete_json,
    _partial_json_loads,
)

logger = logging.getLogger(__name__)


class BaseFormatDetector(ABC):
    """工具调用格式检测器的基类,提供两套接口:一次性解析(one-time)和流式增量解析(streaming incremental)。"""

    def __init__(self):
        # 流式解析状态管理
        # 缓冲区,用于累积跨多个流式分片(chunk)到达的、尚不完整的模式片段
        self._buffer = ""
        # 保存正在解析的每个工具调用的完整信息(名称和参数)。
        # 供 serving 层在流式结束时做补全处理使用。
        # 格式:[{"name": str, "arguments": dict}, ...]
        self.prev_tool_call_arr: List[Dict] = []
        # 当前正在流式输出的工具调用的索引。初始为 -1(无活跃工具),
        # 每完成一个工具就自增。用于追踪当前正在流式输出哪个工具的参数。
        self.current_tool_id: int = -1
        # 标记当前工具的名称是否已发送给客户端。
        # 工具名称会先以空参数发送,随后参数再增量地流式输出。
        self.current_tool_name_sent: bool = False
        # 记录已流式发送给客户端的、每个工具参数的原始 JSON 字符串内容。
        # 对 serving 层在流式结束时计算剩余待发送内容至关重要。
        # 每个下标对应一个 tool_id。例如:['{"location": "San Francisco"', '{"temp": 72']
        self.streamed_args_for_tool: List[str] = []

        # Token 配置(由子类覆盖)
        self.bot_token = ""  # begin-of-tool token,工具调用起始标记
        self.eot_token = ""  # end-of-tool token,工具调用结束标记
        self.tool_call_separator = ", "  # 多个工具调用之间的分隔符

    def _get_tool_indices(self, tools: List[Tool]) -> Dict[str, int]:
        """
        获取工具名称到其在 tools 列表中下标的映射。

        这个工具方法构建一个从函数名到其在 tools 列表中下标的字典,
        在工具校验以及创建 ToolCallItem 时经常用到。

        Args:
            tools: 可用工具列表

        Returns:
            工具名称到下标的映射字典
        """
        return {
            tool.function.name: i for i, tool in enumerate(tools) if tool.function.name
        }

    def parse_base_json(self, action: Any, tools: List[Tool]) -> List[ToolCallItem]:
        """将已解析出的 JSON 对象(单个或列表)转换为 ToolCallItem 列表,并做工具名校验。"""
        tool_indices = self._get_tool_indices(tools)
        if not isinstance(action, list):
            action = [action]

        results = []
        for act in action:
            name = act.get("name")
            if not (name and name in tool_indices):
                logger.warning(f"Model attempted to call undefined function: {name}")
                if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                    continue  # 跳过未知工具(默认的历史行为)

            results.append(
                ToolCallItem(
                    tool_index=tool_indices.get(name, -1),
                    name=name,
                    parameters=json.dumps(
                        act.get("parameters") or act.get("arguments", {}),
                        ensure_ascii=False,
                    ),
                )
            )

        return results

    @abstractmethod
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        一次性解析全部文本。若格式匹配则返回 success=True,否则返回 False。
        注意这里的 leftover_text 表示"本解析器不会再进一步消费的内容"。
        """
        action = orjson.loads(text)
        return StreamingParseResult(calls=self.parse_base_json(action, tools))

    def _ends_with_partial_token(self, buffer: str, bot_token: str) -> int:
        """
        检查 buffer 是否以 bot_token 的一部分(前缀)结尾。
        返回该部分 bot_token 的长度。

        对某些格式而言,bot_token 并非模型词表中的单个 token,
        例如 Mistral 中的 `[TOOL_CALLS] [`。
        """
        for i in range(1, min(len(buffer) + 1, len(bot_token))):
            if bot_token.startswith(buffer[-i:]):
                return i
        return 0

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        带工具校验的流式增量解析。

        这个基类实现最适合以下特征的格式:
        1. bot_token 后紧跟 JSON(例如 bot_token + JSON 数组)
        2. JSON 可以用 partial_json_loads 增量解析
        3. 多个工具调用之间以 "; " 或 ", " 分隔

        不兼容格式的例子(需要自定义实现,但可复用本类的部分逻辑):
        - 每个工具调用被包裹在独立的代码块中:见 Qwen25Detector
        - 多个独立块:[TOOL_CALLS] [...] \n [TOOL_CALLS] [...]
        - 工具调用是 Pythonic 风格

        对于不兼容的格式,检测器应覆盖此方法并实现自定义逻辑。
        """
        # 将新到达的文本追加到缓冲区
        self._buffer += new_text
        current_text = self._buffer

        # 满足以下任一条件即视为 current_text 含有工具调用:
        # 它是一个新工具调用序列的开头;或者在已有前一个工具调用的情况下,
        # 它以工具调用分隔符开头(即在分隔符之后开始一个新工具调用)。
        if not (
            self.has_tool_call(current_text)
            or (
                self.current_tool_id > 0
                and current_text.startswith(self.tool_call_separator)
            )
        ):
            # 只有在确定没有工具调用正在开始时才清空缓冲区
            if not self._ends_with_partial_token(self._buffer, self.bot_token):
                normal_text = self._buffer
                self._buffer = ""
                if self.eot_token in normal_text:
                    normal_text = normal_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=normal_text)
            else:
                # 可能是 bot_token 的一部分(前缀),继续缓冲等待后续内容
                return StreamingParseResult()

        # 若尚未构建工具下标映射则构建之
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        # 解析标志位:工具名已发送后允许全部类型(含 STR),
        # 否则禁止字符串类型(~Allow.STR),避免把不完整的字符串误解析为完整值
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            try:
                # 优先检查:如果正在处理后续工具(current_tool_id > 0),
                # 先检查文本是否以工具分隔符开头。这对并行工具调用至关重要,
                # 因为 bot_token(例如 '[')也可能出现在当前工具的数组参数内部,
                # 我们绝不能把那种情况误判为一个新工具的开始。
                used_separator_branch = False
                if self.current_tool_id > 0 and current_text.startswith(
                    self.tool_call_separator
                ):
                    start_idx = len(self.tool_call_separator)
                    used_separator_branch = True
                else:
                    tool_call_pos = current_text.find(self.bot_token)
                    if tool_call_pos != -1:
                        start_idx = tool_call_pos + len(self.bot_token)
                    else:
                        start_idx = 0

                if start_idx >= len(current_text):
                    return StreamingParseResult()

                try:
                    obj, end_idx = _partial_json_loads(current_text[start_idx:], flags)
                except (MalformedJSON, json.JSONDecodeError):
                    # 分隔符落在了非 JSON 的标记文本上;退回到用 bot_token 定位,
                    # 它能跳过所有对象之间的标记文本。
                    # 例如 Qwen25:分隔符 "," 会匹配到 eot/bot 标签之间的位置。
                    if used_separator_branch and self.bot_token in current_text:
                        start_idx = current_text.find(self.bot_token) + len(
                            self.bot_token
                        )
                        if start_idx >= len(current_text):
                            return StreamingParseResult()
                        obj, end_idx = _partial_json_loads(
                            current_text[start_idx:], flags
                        )
                    else:
                        raise

                is_current_complete = _is_complete_json(
                    current_text[start_idx : start_idx + end_idx]
                )

                # 若存在工具名则校验之
                if "name" in obj and obj["name"] not in self._tool_indices:
                    # 工具名无效——重置状态
                    self._buffer = ""
                    self.current_tool_id = -1
                    self.current_tool_name_sent = False
                    if self.streamed_args_for_tool:
                        self.streamed_args_for_tool.pop()
                    return StreamingParseResult()

                # 处理 parameters/arguments 字段的一致性
                # 注意:这里假设 obj 始终是单个工具调用的(可能不完整的)片段
                if "parameters" in obj:
                    assert (
                        "arguments" not in obj
                    ), "model generated both parameters and arguments"
                    obj["arguments"] = obj["parameters"]

                current_tool_call = obj

            except (MalformedJSON, json.JSONDecodeError):
                return StreamingParseResult()

            if not current_tool_call:
                return StreamingParseResult()

            # 情况 1:处理工具名的流式输出
            # 当遇到一个工具但尚未发送其名称时进入此分支
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")

                if function_name and function_name in self._tool_indices:
                    # 如果这是一个新工具(current_tool_id 曾为 -1),初始化它
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.streamed_args_for_tool.append("")
                    # 如果这是后续工具,确保 streamed_args_for_tool 足够长
                    elif self.current_tool_id >= len(self.streamed_args_for_tool):
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                    # 发送工具名,参数暂时为空
                    res = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        ],
                    )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            # 情况 2:处理参数的流式输出
            # 当已经发送过工具名、现在需要增量地流式输出参数时进入此分支
            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()

                if cur_arguments is not None:
                    # 计算参数中已经流式发送出去的部分有多长
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_arguments = None
                    if self.current_tool_id < len(self.prev_tool_call_arr):
                        prev_arguments = self.prev_tool_call_arr[
                            self.current_tool_id
                        ].get("arguments")

                    argument_diff = None

                    # 如果当前工具的 JSON 已完整,则发送所有剩余的参数
                    if is_current_complete:
                        argument_diff = cur_args_json[sent:]
                        completing_tool_id = (
                            self.current_tool_id
                        )  # 保存即将完成的工具的 ID

                        # 只移除已处理的部分,保留尚未处理的内容
                        self._buffer = current_text[start_idx + end_idx :]

                    # 如果工具仍在解析中,则发送增量变化
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                        if cur_args_json != prev_args_json:
                            # 取上一次与本次参数 JSON 的公共前缀,增量即为该前缀中尚未发送的部分
                            prefix = _find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    # 用当前状态更新 prev_tool_call_arr
                    if self.current_tool_id >= 0:
                        # 确保 prev_tool_call_arr 足够长
                        while len(self.prev_tool_call_arr) <= self.current_tool_id:
                            self.prev_tool_call_arr.append({})
                        self.prev_tool_call_arr[self.current_tool_id] = (
                            current_tool_call
                        )

                    # 若当前工具已完成,则推进到下一个工具
                    if is_current_complete:
                        self.current_tool_name_sent = False
                        self.current_tool_id += 1

                    # 如果有新增内容,则发送参数增量
                    if argument_diff is not None:
                        # 使用正确的 tool_index:已完成的工具用 completing_tool_id,进行中的工具用 current_tool_id
                        tool_index_to_use = (
                            completing_tool_id
                            if is_current_complete
                            else self.current_tool_id
                        )
                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=tool_index_to_use,
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        self.streamed_args_for_tool[tool_index_to_use] += argument_diff

            return res

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult()

    @abstractmethod
    def has_tool_call(self, text: str) -> bool:
        """
        检查给定文本是否包含本格式特有的函数调用标记。
        """
        raise NotImplementedError()

    def supports_structural_tag(self) -> bool:
        """如果本检测器支持 structural tag 格式则返回 True。"""
        return True

    @abstractmethod
    def structure_info(self) -> _GetInfoFunc:
        """
        返回一个用于生成 StructureInfo 的函数,供受约束生成(constrained generation)使用。

        返回的函数接受一个工具名,并返回一个 StructureInfo 对象,
        其中包含在本格式下受约束生成函数调用所需的 begin/end 模式以及触发 token(trigger tokens)。

        Returns:
            一个接受工具名(str)并返回 StructureInfo 的函数
        """
        raise NotImplementedError()

    def get_structural_tag_name(self) -> Optional[str]:
        """如果支持模型原生 structural tag,则返回对应的 XGrammar 模型名。"""
        return None

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
    ) -> Optional[StructuralTag]:
        """
        在支持的情况下,返回模型原生的 XGrammar structural tag。

        Args:
            tools: 可用工具列表
            tool_choice: 请求中的 tool choice 设置
            thinking_mode: 返回的 structural tag 中是否包含模型的推理前缀。
                当 SGLang 的 ReasonerGrammarBackend 会负责 <think>...</think> 前缀时
                (典型情况是配置了 --reasoning-parser),传入 False,
                以保证只有一层去约束推理部分。

        Returns:
            如果本检测器支持模型原生 tag 则返回 StructuralTag,否则返回 None
        """
        structural_tag_name = self.get_structural_tag_name()
        if not structural_tag_name or get_model_structural_tag is None:
            return None

        converted_tools = [tool.model_dump() for tool in tools or []]
        converted_tool_choice = (
            tool_choice.model_dump()
            if isinstance(tool_choice, ToolChoice)
            else tool_choice
        )
        return get_model_structural_tag(
            model=structural_tag_name,
            tools=converted_tools,
            tool_choice=converted_tool_choice,
            reasoning=thinking_mode,
        )
