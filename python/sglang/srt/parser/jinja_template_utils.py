"""Template utilities for Jinja template processing.

This module provides utilities for analyzing and processing Jinja chat templates,
including content format detection and message processing.
"""

import logging

import jinja2
import transformers.utils.chat_template_utils as hf_chat_utils

from sglang.srt.utils import ImageData

logger = logging.getLogger(__name__)

# ============================================================================
# JINJA TEMPLATE CONTENT FORMAT DETECTION
# ============================================================================
#
# This adapts vLLM's approach for detecting chat template content format:
# https://github.com/vllm-project/vllm/blob/02f0c7b220422792f5e53de2a7d51d2d3ff2df28/vllm/entrypoints/chat_utils.py#L296-L313
# - Analyzes Jinja template AST to detect content iteration patterns
# - 'openai' format: templates with {%- for content in message['content'] -%} loops
# - 'string' format: templates that expect simple string content
# - Processes content accordingly to match template expectations


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    """判断节点是否为“读取变量”访问，如 {{ varname }}。"""
    # Name 节点且 ctx==load（读取上下文）、名称匹配。
    if isinstance(node, jinja2.nodes.Name):
        return node.ctx == "load" and node.name == varname
    return False


def _is_attr_access(node: jinja2.nodes.Node, varname: str, key: str) -> bool:
    """判断节点是否为属性/下标访问，如 {{ varname['key'] }} 或 {{ varname.key }}。"""
    # 下标形式：varname['key']。
    if isinstance(node, jinja2.nodes.Getitem):
        return (
            _is_var_access(node.node, varname)
            and isinstance(node.arg, jinja2.nodes.Const)
            and node.arg.value == key
        )

    # 属性形式：varname.key。
    if isinstance(node, jinja2.nodes.Getattr):
        return _is_var_access(node.node, varname) and node.attr == key

    return False


def _is_var_or_elems_access(
    node: jinja2.nodes.Node,
    varname: str,
    key: str = None,
) -> bool:
    """判断节点是否（透过过滤器/测试/切片）访问了 varname 或 varname[key]。

    递归剖开 Jinja 中常见的包裹：
      - Filter：如 message['content'] | selectattr(...)
      - Test：如 ... is ...
      - Slice：如 message['content'][1:]
    最终落到“属性访问”或“变量访问”的判断。
    """
    # 过滤器：继续看其被过滤的对象。
    if isinstance(node, jinja2.nodes.Filter):
        return node.node is not None and _is_var_or_elems_access(
            node.node, varname, key
        )
    # 测试表达式：继续看其左侧对象。
    if isinstance(node, jinja2.nodes.Test):
        return _is_var_or_elems_access(node.node, varname, key)

    # 切片：继续看被切片的对象。
    if isinstance(node, jinja2.nodes.Getitem) and isinstance(
        node.arg, jinja2.nodes.Slice
    ):
        return _is_var_or_elems_access(node.node, varname, key)

    # 有 key 时判断属性/下标访问；否则判断变量访问。
    return _is_attr_access(node, varname, key) if key else _is_var_access(node, varname)


def _try_extract_ast(chat_template: str):
    """尝试将 Jinja 模板解析为 AST（抽象语法树），失败返回 None。"""
    try:
        # 复用 HuggingFace 的模板编译器，再拿环境去 parse 出 AST。
        jinja_compiled = hf_chat_utils._compile_jinja_template(chat_template)
        return jinja_compiled.environment.parse(chat_template)
    except Exception as e:
        logger.debug(f"Error when compiling Jinja template: {e}")
        return None


def detect_jinja_template_content_format(chat_template: str) -> str:
    """
    检测聊天模板期望的内容格式是 'string' 还是 'openai'。

    - 'string'：content 是简单字符串（如 DeepSeek 模板）。
    - 'openai'：content 是结构化 dict 列表（如 Llama4 模板）。

    检测逻辑：
    - 若模板含如 {%- for content in message['content'] -%} 的循环 → 'openai'。
    - 否则 → 'string'。
    """
    # 多模态模板快捷判断：含 image/audio/video/vision 关键字则直接当作 openai 格式。
    if any(
        keyword in chat_template for keyword in ["image", "audio", "video", "vision"]
    ):
        return "openai"

    # 解析 AST；解析失败则保守返回 string。
    jinja_ast = _try_extract_ast(chat_template)
    if jinja_ast is None:
        return "string"

    try:
        # 遍历所有 for 循环，查找对 content 的迭代。
        for loop_ast in jinja_ast.find_all(jinja2.nodes.For):
            loop_iter = loop_ast.iter

            # 是否在迭代 message['content']（或带过滤器/切片的变体）。
            if _is_var_or_elems_access(loop_iter, "message", "content"):
                return "openai"  # 发现内容迭代 → openai 格式

            # 也检查 msg.content / m.content 这类变量名（如 glm4v 模板）。
            if _is_var_or_elems_access(
                loop_iter, "msg", "content"
            ) or _is_var_or_elems_access(loop_iter, "m", "content"):
                return "openai"  # 发现内容迭代 → openai 格式（glm4v）

        return "string"  # 未发现内容循环 → string 格式
    except Exception as e:
        logger.debug(f"Error when parsing AST of Jinja template: {e}")
        return "string"


def process_content_for_template_format(
    msg_dict: dict,
    content_format: str,
    image_data: list,
    video_data: list,
    audio_data: list,
    modalities: list,
    use_dpsk_v32_encoding: bool = False,
) -> dict:
    """
    根据检测到的模板格式处理消息内容（并抽取多模态数据）。

    参数：
        msg_dict: 含 content 的消息字典。
        content_format: 'string' 或 'openai'（由 AST 分析得出）。
        image_data: 输出参数，追加提取出的图像。
        video_data: 输出参数，追加提取出的视频。
        audio_data: 输出参数，追加提取出的音频。
        modalities: 输出参数，追加模态信息。
        use_dpsk_v32_encoding: 为 True 时，抽出多模态数据并把 content 转为字符串（用于 DeepSeek-V3.2 编码）。

    返回：处理后的消息字典。
    """
    if not isinstance(msg_dict.get("content"), list):
        # content 已是字符串或 None，无需处理；顺便过滤掉值为 None 的字段。
        return {k: v for k, v in msg_dict.items() if v is not None}

    if content_format == "openai" or use_dpsk_v32_encoding:
        # openai 格式：保留结构化内容列表，并将类型归一化（image_url → image 等）。
        # V32 编码：抽出多模态数据，但把 content 压成纯文本字符串。
        processed_content_parts = []
        text_parts = []
        for chunk in msg_dict["content"]:
            if isinstance(chunk, dict):
                chunk_type = chunk.get("type")

                if chunk_type == "image_url":
                    # 提取图像 URL、detail、max_dynamic_patch，存入 image_data。
                    image_obj = chunk.get("image_url") or {}
                    mdp = image_obj.get("max_dynamic_patch", None)
                    # Also allow flat style: chunk["max_dynamic_patch"]
                    image_data.append(
                        ImageData(
                            url=image_obj["url"],
                            detail=image_obj.get("detail", "auto"),
                            max_dynamic_patch=mdp,
                        )
                    )

                    if chunk.get("modalities"):
                        modalities.append(chunk.get("modalities"))
                    # 归一化为简单的 'image' 类型以兼容模板。
                    processed_content_parts.append({"type": "image"})
                elif chunk_type == "video_url":
                    # 视频：无 max_dynamic_patch 时只存 url；有时保留结构信息供后端使用。
                    video_obj = chunk.get("video_url") or {}
                    mdp = video_obj.get("max_dynamic_patch", None)
                    if mdp is None:
                        video_data.append(chunk["video_url"]["url"])
                    else:
                        # Keep structured info for backend, but template only sees {"type":"video"}
                        video_data.append(
                            {
                                "url": video_obj["url"],
                                "max_dynamic_patch": mdp,
                            }
                        )
                    if chunk.get("modalities"):
                        modalities.append(chunk.get("modalities"))
                    # 归一化为简单的 'video' 类型以兼容模板。
                    processed_content_parts.append({"type": "video"})
                elif chunk_type == "audio_url":
                    audio_data.append(chunk["audio_url"]["url"])
                    # 归一化为简单的 'audio' 类型。
                    processed_content_parts.append({"type": "audio"})
                elif chunk_type == "text":
                    # V32 编码：文本单独收集，稍后拼接为一个字符串。
                    if use_dpsk_v32_encoding:
                        text_parts.append(chunk["text"])
                    else:
                        # openai 格式：文本块原样保留。
                        processed_content_parts.append(chunk)

        # 重建消息：除 content 外的非空字段原样保留。
        new_msg = {
            k: v for k, v in msg_dict.items() if v is not None and k != "content"
        }
        if use_dpsk_v32_encoding:
            # V32：content 是拼接后的纯文本。
            new_msg["content"] = " ".join(text_parts) if text_parts else ""
        else:
            # openai：content 是归一化后的结构化列表。
            new_msg["content"] = processed_content_parts
        return new_msg

    elif content_format == "string":
        # string 格式：只保留文本，展平为纯文本（适用于 DeepSeek 等模板）。
        text_parts = []
        for chunk in msg_dict["content"]:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                text_parts.append(chunk["text"])
            # 注：string 格式下忽略图像/音频，因为模板不期望结构化内容；
            # 多模态占位符需以其他方式插入。

        new_msg = msg_dict.copy()
        new_msg["content"] = " ".join(text_parts) if text_parts else ""
        new_msg = {k: v for k, v in new_msg.items() if v is not None}
        return new_msg

    else:
        # 未知格式：报错。
        raise ValueError(f"Invalid content format: {content_format}")
