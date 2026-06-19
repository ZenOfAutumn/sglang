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
"""
Centralized template management for chat templates and completion templates.

This module provides a unified interface for managing both chat conversation templates
and code completion templates, eliminating global state and improving modularity.

中译：聊天模板（chat template）与补全模板（completion template）的集中式管理模块。
      本模块提供统一接口来同时管理「对话模板」与「代码补全模板」，把原本散落的全局状态
      收敛到一个 TemplateManager 实例里，从而消除全局变量、提升模块化程度。
      职责包括：从命令行参数/文件/模型路径/HuggingFace 等多种来源加载模板，并从聊天模板中
      自动探测「推理模式（reasoning/thinking）」以及建议使用的 reasoning/tool-call 解析器。
"""

import json
import logging
import os
from typing import Dict, Optional

from sglang.srt.managers.template_detection import (
    REASONING_PARSER_RULES,
    TOOL_CALL_PARSER_RULES,
    ReasoningToggleConfig,
    build_detection_context,
    detect_reasoning_pattern,
    match_rules,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.parser.code_completion_parser import (
    CompletionTemplate,
    FimPosition,
    completion_template_exists,
    register_completion_template,
    set_completion_template,
)
from sglang.srt.parser.conversation import (
    Conversation,
    SeparatorStyle,
    chat_template_exists,
    get_conv_template_by_model_path,
    register_conv_template,
)
from sglang.srt.parser.jinja_template_utils import detect_jinja_template_content_format

logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Centralized manager for chat and completion templates.

    This class encapsulates all template-related state and operations,
    eliminating the need for global variables and providing a clean
    interface for template management.

    中译：聊天与补全模板的集中式管理器。
          本类把所有与模板相关的状态和操作封装在一起，免去全局变量，对外提供干净的模板管理接口。
          典型协作对象是 TokenizerManager（从中取出/写回 tokenizer 与 processor 的 chat_template），
          以及 template_detection 模块（用于自动探测推理模式与解析器）。
          所有状态字段以下划线开头，外部通过只读属性（@property）访问，避免被随意改写。
    """

    def __init__(self):
        # 中译：当前选用的内置聊天模板名（None 表示未选用内置模板，可能改用 Jinja 模板）。
        self._chat_template_name: Optional[str] = None
        # 中译：当前选用的补全模板名。
        self._completion_template_name: Optional[str] = None
        # 中译：Jinja 模板的内容格式（'string' / 'openai' / None），决定多模态消息如何渲染，默认 'openai'。
        self._jinja_template_content_format: Optional[str] = "openai"
        # 中译：是否强制开启推理（thinking）模式，由聊天模板自动探测得出。
        self._force_reasoning: bool = False
        # 中译：从聊天模板推断出的「推理开关配置」（开关参数名、默认是否开启、特殊情况等）。
        self._reasoning_config: Optional[ReasoningToggleConfig] = None
        # 中译：自动探测到的推理解析器名（如未识别则为 None）。
        self._suggested_reasoning_parser: Optional[str] = None
        # 中译：自动探测到的工具调用解析器名（如未识别则为 None）。
        self._suggested_tool_call_parser: Optional[str] = None

    @property
    def chat_template_name(self) -> Optional[str]:
        """Get the current chat template name.

        中译：只读属性——返回当前选用的聊天模板名。
        """
        return self._chat_template_name

    @property
    def completion_template_name(self) -> Optional[str]:
        """Get the current completion template name.

        中译：只读属性——返回当前选用的补全模板名。
        """
        return self._completion_template_name

    @property
    def jinja_template_content_format(self) -> Optional[str]:
        """Get the detected template content format ('string' or 'openai' or None).

        中译：只读属性——返回探测到的 Jinja 模板内容格式（'string' / 'openai' / None）。
        """
        return self._jinja_template_content_format

    @property
    def force_reasoning(self) -> bool:
        """
        Check if the current chat template enforces reasoning/thinking.

        Returns:
            True if the template contains reasoning patterns like <think> tags

        中译：只读属性——判断当前聊天模板是否强制开启推理（thinking）。
              返回值：若模板含有 <think> 之类推理标记（强制开启）则为 True。
        """
        return self._force_reasoning

    @property
    def reasoning_config(self) -> Optional[ReasoningToggleConfig]:
        """Get the reasoning toggle config inferred from chat template.

        中译：只读属性——返回从聊天模板推断出的「推理开关配置」。
        """
        return self._reasoning_config

    @property
    def suggested_reasoning_parser(self) -> Optional[str]:
        """Get the auto-detected reasoning parser name, or None.

        中译：只读属性——返回自动探测到的推理解析器名（未识别则为 None）。
        """
        return self._suggested_reasoning_parser

    @property
    def suggested_tool_call_parser(self) -> Optional[str]:
        """Get the auto-detected tool-call parser name, or None.

        中译：只读属性——返回自动探测到的工具调用解析器名（未识别则为 None）。
        """
        return self._suggested_tool_call_parser

    def _run_template_detection(self, template, tokenizer) -> None:
        """Run reasoning pattern and parser detection on a template.

        中译：对给定模板执行「推理模式 + 解析器」的自动探测，并把结果写入本实例的对应字段。
              参数 template 为聊天模板字符串，tokenizer 用于取词表（vocab）以辅助识别特定模型族。
              副作用：更新 _force_reasoning、_reasoning_config、_suggested_reasoning_parser、
              _suggested_tool_call_parser 四个字段。
        """
        # 中译：先探测推理模式（是否强制思考、开关配置）。
        self._force_reasoning, self._reasoning_config = detect_reasoning_pattern(
            template
        )
        # Build context once, reuse for both parser detections (avoids
        # duplicate tokenizer.get_vocab() calls).
        # 中译：只构建一次探测上下文，供 reasoning 与 tool-call 两类解析器探测复用，
        #       避免重复调用 tokenizer.get_vocab()（取词表开销不小）。
        ctx = build_detection_context(
            template, tokenizer, self._reasoning_config, self._force_reasoning
        )
        if ctx is None:
            # 中译：模板为空（无法构建上下文）时直接返回，保持解析器字段为 None。
            return
        # 中译：分别按 reasoning / tool-call 规则集匹配，命中第一条规则即返回其建议的解析器名。
        self._suggested_reasoning_parser = match_rules(
            ctx, REASONING_PARSER_RULES, "reasoning parser"
        )
        self._suggested_tool_call_parser = match_rules(
            ctx, TOOL_CALL_PARSER_RULES, "tool-call parser"
        )

    def load_chat_template(
        self,
        tokenizer_manager: TokenizerManager,
        chat_template_arg: Optional[str],
        model_path: str,
    ) -> None:
        """
        Load a chat template from various sources.

        Args:
            tokenizer_manager: The tokenizer manager instance
            chat_template_arg: Template name, file path, or None to auto-detect
            model_path: Path to the model

        中译：从多种来源加载聊天模板（对外主入口之一）。
              参数：
                tokenizer_manager：分词器管理器实例（用于读写其 tokenizer/processor 的 chat_template）。
                chat_template_arg：模板名、文件路径，或 None（None 表示走自动探测）。
                model_path：模型路径。
              逻辑：显式指定了模板就直接加载；否则先按模型路径猜测内置模板，再回退到 HuggingFace 模板。
              最后无论走哪条路径，都会基于最终的聊天模板做一次推理/解析器自动探测。
        """
        if chat_template_arg:
            # 中译：显式给定了模板参数——按内置名/文件路径加载。
            self._load_explicit_chat_template(tokenizer_manager, chat_template_arg)
        else:
            # Guess chat template from model path
            # 中译：未显式指定——先尝试根据模型路径猜测内置聊天模板。
            self.guess_chat_template_from_model_path(model_path)

            # If no pre-defined template was found, fallback to HuggingFace template
            # 中译：若没猜中任何内置模板，则回退使用 HuggingFace（tokenizer/processor 自带）的模板。
            if self._chat_template_name is None:
                # Try HuggingFace template first
                # 中译：优先解析 HuggingFace 自带模板。
                hf_template = self._resolve_hf_chat_template(tokenizer_manager)
                if hf_template:
                    # override the chat template
                    # 中译：把解析到的 HF 模板写回 tokenizer，覆盖其原 chat_template。
                    if tokenizer_manager.tokenizer:
                        tokenizer_manager.tokenizer.chat_template = hf_template
                    # 中译：探测该 HF 模板的内容格式（string/openai）。
                    self._jinja_template_content_format = (
                        detect_jinja_template_content_format(hf_template)
                    )
                    logger.info(
                        f"Using default HuggingFace chat template with detected content format: {self._jinja_template_content_format}"
                    )
                else:
                    # Default to string content format if no template was found
                    # 中译：连 HF 模板也没有时，内容格式默认回退为 'string'。
                    self._jinja_template_content_format = "string"
                    logger.info(
                        "No chat template found, defaulting to 'string' content format"
                    )

        # Detect reasoning pattern and suggest parser from chat template
        # 中译：基于最终生效的聊天模板，探测推理模式并给出建议的解析器；随后把探测结果汇总打日志。
        if tokenizer_manager.tokenizer:
            template = tokenizer_manager.tokenizer.chat_template
            self._run_template_detection(template, tokenizer_manager.tokenizer)
            parts = []
            if self._reasoning_config:
                parts.append(f"reasoning_config={self._reasoning_config}")
            if self._suggested_reasoning_parser:
                parts.append(f"reasoning_parser={self._suggested_reasoning_parser}")
            if self._suggested_tool_call_parser:
                parts.append(f"tool_call_parser={self._suggested_tool_call_parser}")
            if parts:
                logger.info(f"Auto-detected template features: {', '.join(parts)}")

    def _load_explicit_chat_template(
        self, tokenizer_manager: TokenizerManager, chat_template_arg: str
    ) -> None:
        """Load explicitly specified chat template.

        中译：加载用户显式指定的聊天模板。
              参数 chat_template_arg 可能是：内置模板名、.jinja 文件路径，或 .json 文件路径。
              若既非内置名也非有效文件路径，则抛 RuntimeError。
        """
        logger.info(f"Loading chat template from argument: {chat_template_arg}")

        # 中译：若是已注册的内置模板名，直接记录名字即可返回。
        if chat_template_exists(chat_template_arg):
            self._chat_template_name = chat_template_arg
            return

        # 中译：既不是内置名，也不是存在的文件路径——无法识别，报错。
        if not os.path.exists(chat_template_arg):
            raise RuntimeError(
                f"Chat template {chat_template_arg} is not a built-in template name "
                "or a valid chat template file path."
            )

        # 中译：按文件后缀分流——.jinja 走 Jinja 模板加载，否则按 JSON 模板加载。
        if chat_template_arg.endswith(".jinja"):
            self._load_jinja_template(tokenizer_manager, chat_template_arg)
        else:
            self._load_json_chat_template(chat_template_arg)

    def guess_chat_template_from_model_path(self, model_path: str) -> None:
        """
        Infer chat template name from model path.

        Args:
            model_path: Path to the model

        中译：根据模型路径推断内置聊天模板名（命中则写入 _chat_template_name，否则保持不变）。
              参数 model_path：模型路径。
        """
        template_name = get_conv_template_by_model_path(model_path)
        if template_name is not None:
            logger.info(f"Inferred chat template from model path: {template_name}")
            self._chat_template_name = template_name

    def load_completion_template(self, completion_template_arg: str) -> None:
        """
        Load completion template for code completion.

        Args:
            completion_template_arg: Template name or file path

        中译：加载用于「代码补全」的补全模板（FIM, fill-in-the-middle）。
              参数 completion_template_arg：内置补全模板名或 .json 文件路径。
              副作用：最后调用 set_completion_template 把选定模板设置为全局生效的补全模板。
        """
        logger.info(f"Loading completion template: {completion_template_arg}")

        # 中译：不是已注册的内置补全模板名时——尝试当作 JSON 文件路径加载（路径无效则报错）。
        if not completion_template_exists(completion_template_arg):
            if not os.path.exists(completion_template_arg):
                raise RuntimeError(
                    f"Completion template {completion_template_arg} is not a built-in template name "
                    "or a valid completion template file path."
                )

            self._load_json_completion_template(completion_template_arg)
        else:
            # 中译：是内置名，直接记录。
            self._completion_template_name = completion_template_arg

        # 中译：把最终选定的补全模板名设置为全局生效。
        set_completion_template(self._completion_template_name)

    def initialize_templates(
        self,
        tokenizer_manager: TokenizerManager,
        model_path: str,
        chat_template: Optional[str] = None,
        completion_template: Optional[str] = None,
    ) -> None:
        """
        Initialize all templates based on provided configuration.

        Args:
            tokenizer_manager: The tokenizer manager instance
            model_path: Path to the model
            chat_template: Optional chat template name/path
            completion_template: Optional completion template name/path

        中译：根据配置一次性初始化所有模板（聊天模板必加载，补全模板按需加载）——对外总入口。
              参数：
                tokenizer_manager：分词器管理器实例。
                model_path：模型路径（聊天模板自动探测时用）。
                chat_template：可选的聊天模板名/路径（None 表示自动探测）。
                completion_template：可选的补全模板名/路径（None 表示不加载）。
        """
        # Load chat template
        # 中译：加载聊天模板（必做）。
        self.load_chat_template(tokenizer_manager, chat_template, model_path)

        # Load completion template
        # 中译：仅当指定了补全模板时才加载它。
        if completion_template:
            self.load_completion_template(completion_template)

    def _load_jinja_template(
        self, tokenizer_manager: TokenizerManager, template_path: str
    ) -> None:
        """Load a Jinja template file.

        中译：加载一个 .jinja 聊天模板文件，并写回 tokenizer.chat_template。
              副作用：读取文件内容、把转义的 "\\n" 还原为真实换行符后写入 tokenizer；
              清空 _chat_template_name（因为用的是 Jinja 模板而非内置具名模板）；并探测其内容格式。
        """
        with open(template_path, "r") as f:
            # 中译：读入整个模板文件，并去掉首尾换行。
            chat_template = "".join(f.readlines()).strip("\n")
        # 中译：把文本中字面的 "\n" 转为真正的换行符后写回 tokenizer。
        tokenizer_manager.tokenizer.chat_template = chat_template.replace("\\n", "\n")
        # 中译：用的是 Jinja 模板，没有内置具名模板，故名字置空。
        self._chat_template_name = None
        # Detect content format from the loaded template
        # 中译：从加载到的模板内容中探测其内容格式（string/openai）。
        self._jinja_template_content_format = detect_jinja_template_content_format(
            chat_template
        )
        logger.info(
            f"Detected user specified Jinja chat template with content format: {self._jinja_template_content_format}"
        )

    def _load_json_chat_template(self, template_path: str) -> None:
        """Load a JSON chat template file.

        中译：加载一个 JSON 格式的聊天模板文件，构造 Conversation 并注册为可用模板。
              JSON 中需含 name/system/user/assistant/sep_style/stop_str 等字段；
              sep_style 必须是 SeparatorStyle 枚举里的合法值，否则抛 ValueError。
        """
        assert template_path.endswith(
            ".json"
        ), "unrecognized format of chat template file"

        with open(template_path, "r") as filep:
            template = json.load(filep)
            try:
                # 中译：把字符串形式的分隔风格转成 SeparatorStyle 枚举。
                sep_style = SeparatorStyle[template["sep_style"]]
            except KeyError:
                # 中译：未知的分隔风格——抛出明确错误（from None 抑制原始 KeyError 链）。
                raise ValueError(
                    f"Unknown separator style: {template['sep_style']}"
                ) from None

            # 中译：用 JSON 字段构造 Conversation 并注册（override=True 表示同名可覆盖）。
            register_conv_template(
                Conversation(
                    name=template["name"],
                    system_template=template["system"] + "\n{system_message}",
                    system_message=template.get("system_message", ""),
                    roles=(template["user"], template["assistant"]),
                    sep_style=sep_style,
                    sep=template.get("sep", "\n"),
                    stop_str=template["stop_str"],
                ),
                override=True,
            )
        self._chat_template_name = template["name"]

    def _load_json_completion_template(self, template_path: str) -> None:
        """Load a JSON completion template file.

        中译：加载一个 JSON 格式的补全模板文件，构造 CompletionTemplate 并注册。
              JSON 中需含 name 及 fim_begin/middle/end_token、fim_position 等字段；
              fim_position 必须是 FimPosition 枚举里的合法值，否则抛 ValueError。
        """
        assert template_path.endswith(
            ".json"
        ), "unrecognized format of completion template file"

        with open(template_path, "r") as filep:
            template = json.load(filep)
            try:
                # 中译：把字符串形式的 FIM 位置转成 FimPosition 枚举。
                fim_position = FimPosition[template["fim_position"]]
            except KeyError:
                # 中译：未知的 FIM 位置——抛出明确错误。
                raise ValueError(
                    f"Unknown fim position: {template['fim_position']}"
                ) from None

            # 中译：用 JSON 字段构造补全模板并注册（override=True 表示同名可覆盖）。
            register_completion_template(
                CompletionTemplate(
                    name=template["name"],
                    fim_begin_token=template["fim_begin_token"],
                    fim_middle_token=template["fim_middle_token"],
                    fim_end_token=template["fim_end_token"],
                    fim_position=fim_position,
                ),
                override=True,
            )
        self._completion_template_name = template["name"]

    def _resolve_hf_chat_template(
        self, tokenizer_manager: TokenizerManager
    ) -> Optional[str]:
        """中译：解析 HuggingFace 自带的聊天模板字符串。

        优先取多模态 processor 的 chat_template，其次取 tokenizer 的 chat_template；
        若模板是 dict（含多个具名模板）则交给 _select_named_template 选择；都没有则返回 None。
        任何异常都被吞掉并记 warning，返回 None（不让模板探测影响主流程）。
        """
        try:
            # Try (mm-)processor first, then tokenizer
            # 中译：先试多模态 processor，再退回到 tokenizer 上的 chat_template。
            template = (
                getattr(tokenizer_manager.processor, "chat_template", None)
                if tokenizer_manager.processor
                else None
            ) or (
                getattr(tokenizer_manager.tokenizer, "chat_template", None)
                if tokenizer_manager.tokenizer
                else None
            )

            if template is None:
                logger.warning("No HuggingFace chat template found")
                return None

            # Handle dict templates (multiple named templates)
            # 中译：模板为 dict 时表示存在多个具名模板，交给 _select_named_template 决策。
            if isinstance(template, dict):
                return self._select_named_template(template, tokenizer_manager)

            # Single string template
            # 中译：单个字符串模板，直接返回。
            return template

        except Exception as e:
            logger.warning(f"Error getting chat template: {e}")
            return None

    def _select_named_template(
        self, templates: Dict[str, str], tokenizer_manager: TokenizerManager
    ) -> str:
        """中译：从多个「具名」HuggingFace 聊天模板中选出一个。

        参数 templates 为 {名字: 模板字符串} 字典。
        若 server_args.hf_chat_template_name 指定了名字则用它（不存在则报错）；
        否则回退使用字典里的第一个模板。空字典直接抛 ValueError。
        """
        if not templates:
            raise ValueError("Empty templates dict provided")

        available_names = list(templates.keys())
        logger.info(f"Multiple HuggingFace chat templates available: {available_names}")

        # Use specified template if provided
        # 中译：若用户通过 hf_chat_template_name 指定了模板名，则优先使用（找不到就报错）。
        if preferred_name := tokenizer_manager.server_args.hf_chat_template_name:
            if preferred_name not in templates:
                raise ValueError(
                    f"Specified template '{preferred_name}' not found. "
                    f"Available templates: {available_names}"
                )
            logger.info(f"Using specified chat template: '{preferred_name}'")
            return templates[preferred_name]

        # Fallback: Use first available template
        # 中译：未指定时——回退使用可用模板中的第一个。
        first_name = available_names[0]
        logger.info(f"Using first available template: '{first_name}'")
        return templates[first_name]
