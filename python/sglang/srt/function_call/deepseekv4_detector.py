import logging

from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector

logger = logging.getLogger(__name__)


class DeepSeekV4Detector(DeepSeekV32Detector):
    """
    Detector for DeepSeek V4 model function call format.

    The DeepSeek V4 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    中译：DeepSeek-V4 模型工具调用格式的解析器。

    DeepSeek-V4 使用类 XML 的 DSML 标签来划定工具调用。本类直接继承
    DeepSeekV32Detector，复用其解析逻辑，仅覆盖起止标记与正则（见 __init__）。
    支持两种参数格式：

    中译：格式 1——XML 参数标签（每个参数用一个 <｜DSML｜parameter> 标签）：
    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    中译：格式 2——直接 JSON（所有参数写成一个 JSON 对象）：
    Format 2 - Direct JSON:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Examples:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜tool_calls>` and `</｜DSML｜tool_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V4 format specification

    中译：关键组成：
    - 工具调用整体段：包在 `<｜DSML｜tool_calls>` 与 `</｜DSML｜tool_calls>` 之间；
    - 单个工具调用：包在 `<｜DSML｜invoke name="...">` 与 `</｜DSML｜invoke>` 之间；
    - 参数：XML 标签或直接 JSON 两种格式均可；
    - 支持一次多个工具调用。

    参考：DeepSeek-V4 格式规范。
    """

    def __init__(self):
        # 中译：复用父类（V3.2）的全部解析逻辑，仅覆盖 V4 特有的起止标记与正则。
        super().__init__()
        # 中译：bot_token / eot_token——工具调用整体段的起始 / 结束标记
        #       （bot = begin-of-tool-calls，eot = end-of-tool-calls）。
        self.bot_token = "<｜DSML｜tool_calls>"
        self.eot_token = "</｜DSML｜tool_calls>"
        # 中译：非贪婪地提取起止标记之间的工具调用内容（.*? 搭配 DOTALL 跨行匹配）。
        self.function_calls_regex = r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>"

    def get_structural_tag_name(self) -> str:
        # 中译：返回结构化标签的名称（供上层识别本模型的原生 structural_tag）。
        return "deepseek_v4"
