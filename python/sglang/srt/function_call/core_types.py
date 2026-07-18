from dataclasses import dataclass
from typing import Callable, List, Optional

from pydantic import BaseModel


class ToolCallItem(BaseModel):
    """Simple encapsulation of the parsed ToolCall result for easier usage in streaming contexts.

    中译：解析出的单个工具调用结果的简单封装，便于在流式（streaming）场景下使用。
    流式下一个工具调用可能分多次增量产出，故字段设计为可部分填充。
    """

    # 中译：本次响应中该工具调用的索引（从 0 开始），用于在并行多工具调用时
    #       区分不同调用，也用于流式时把同一调用的多个增量片段拼回同一条。
    tool_index: int
    # 中译：被调用的工具名。可为 None——流式下 name 与参数可能分多次到达，
    #       仅传递参数增量的片段会把 name 置空。
    name: Optional[str] = None
    # 中译：工具参数，**为 JSON 字符串而非 dict**。用字符串是为了支持流式下
    #       逐段拼接（上层直接把片段追加到已有字符串后），最后再整体反序列化。
    parameters: str  # JSON string


class StreamingParseResult(BaseModel):
    """Result of streaming incremental parsing.

    中译：一次（流式或非流式）解析的结果，同时携带普通文本与工具调用。
    """

    # 中译：本次解析出的、应展示给用户的普通文本（非工具调用部分）；
    #       当前正处在工具调用中间时可能为空字符串。
    normal_text: str = ""
    # 中译：本次解析出的工具调用列表（流式下可能是增量片段，如仅 name 或部分参数）。
    calls: List[ToolCallItem] = []


@dataclass
class StructureInfo:
    # 中译：某个工具调用在模型输出中的「结构信息」，用于生成结构化标签约束解码：
    #   begin   —— 工具调用的起始标记（如 <tool_call>、特殊 token）；
    #   end     —— 工具调用的结束标记；
    #   trigger —— 触发进入结构化约束的触发词。
    begin: str
    end: str
    trigger: str


"""
Helper alias of function
Usually it is a function that takes a name string and returns a StructureInfo object,
which can be used to construct a structural_tag object

中译：函数类型别名。通常是一个「接收工具名字符串 → 返回 StructureInfo 对象」的函数，
用于据此构造 structural_tag（结构化标签）对象。
"""
_GetInfoFunc = Callable[[str], StructureInfo]
