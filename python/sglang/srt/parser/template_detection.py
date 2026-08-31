# Copyright 2026 SGLang Team
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
Template detection utilities for auto-detecting reasoning and tool-call parsers.

Provides rule-based detection of reasoning mode, reasoning parser, and tool-call
parser from chat templates and tokenizer vocabularies.

中译：模板探测工具——用于自动识别「推理（reasoning/thinking）解析器」与「工具调用（tool-call）解析器」。
      本模块通过一套「规则（DetectionRule）」对聊天模板文本与 tokenizer 词表进行匹配，从而推断出：
        1) 推理模式（是否强制开启、开关参数名及其默认值等，见 ReasoningToggleConfig）；
        2) 应使用的推理解析器名（如 deepseek-r1、qwen3、gpt-oss 等）；
        3) 应使用的工具调用解析器名。
      设计要点：每条规则用一个 predicate 谓词函数判定是否命中；规则按顺序匹配，命中第一条即返回。
      探测所需的全部输入被打包进 TemplateDetectionContext，从而让谓词只依赖该上下文、便于测试与复用。
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import jinja2
import jinja2.ext
import jinja2.nodes
import jinja2.sandbox

from sglang.srt.arg_groups.overrides import declare_late_resolution, resolving_view

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemplateDetectionContext:
    """中译：模板探测上下文（不可变数据类）。

    把一次探测所需的全部输入打包在一起，供各规则的 predicate 谓词使用，避免谓词直接依赖外部状态：
      - template：聊天模板字符串。
      - reasoning_config：已探测出的推理开关配置（部分解析器规则会复用它）。
      - force_reasoning：是否强制开启推理。
      - vocab：tokenizer 词表的 token 集合（用于判断某些特殊 token 是否存在）。
    另提供三个便捷判定方法：文本包含 / 词表包含 / 正则匹配。
    """

    template: str
    reasoning_config: Optional["ReasoningToggleConfig"]
    force_reasoning: bool
    vocab: set[str]

    def has_text(self, needle: str) -> bool:
        # 中译：判断模板文本中是否「字面包含」给定子串。
        return needle in self.template

    def has_vocab(self, token: str) -> bool:
        # 中译：判断给定 token 是否存在于 tokenizer 词表中。
        return token in self.vocab

    def has_pattern(self, pattern: str, flags: int = 0) -> bool:
        # 中译：判断模板文本是否匹配给定正则（flags 可传 re.DOTALL 等）。
        return re.search(pattern, self.template, flags) is not None

    def has_vocab_pattern(self, pattern: str) -> bool:
        compiled = re.compile(pattern)
        return any(isinstance(tok, str) and compiled.search(tok) for tok in self.vocab)


@dataclass(frozen=True)
class DetectionRule:
    """中译：单条探测规则（不可变数据类）。

      - name：规则名（仅用于日志/调试）。
      - value：命中后要返回的结果值（如解析器名字符串，或 ReasoningToggleConfig）。
      - predicate：谓词函数，接收 TemplateDetectionContext，返回该规则是否命中。
    """

    name: str
    value: object
    predicate: Callable[[TemplateDetectionContext], bool]


@dataclass(frozen=True)
class ReasoningToggleConfig:
    """中译：推理（thinking）开关配置（不可变数据类）。

    描述某个聊天模板如何控制是否进入推理模式：
      - toggle_param：控制推理的模板参数名（如 "enable_thinking" / "thinking"），无则为 None。
      - default_enabled：该参数未显式传入时的默认值（是否默认开启推理）。
      - special_case：特殊情形标记（如 "always" 表示恒开、"mistral" 表示 Mistral 专属处理）。
    """

    toggle_param: Optional[str] = None
    default_enabled: Optional[bool] = None
    special_case: Optional[str] = None
    effort_kwarg: Optional[str] = None

    @property
    def always_on(self) -> bool:
        # 中译：是否「恒定开启」推理（special_case == "always"）。
        return self.special_case == "always"


class _GenerationTagExtension(jinja2.ext.Extension):
    """Parse-only support for the ``{% generation %}`` blocks emitted by
    transformers chat templates (assistant token masking). Needed so the
    toggle-default parser below can build an AST for templates that use it."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.Scope(body).set_lineno(lineno)


def _has_toggle_default_assignment(
    ctx: TemplateDetectionContext, param: str, default: bool
) -> bool:
    """True if the template applies Jinja's ``default`` filter (or its ``d``
    alias) to ``param`` with working-toggle semantics and the given default,
    e.g. ``{%- set enable_thinking = enable_thinking | default(false) -%}``
    (Laguna-XS-2.1) or ``default(true)`` (Laguna-S-2.1).

    Matches on the parsed ``Filter`` node, so comments, ``{% raw %}`` blocks,
    and string literals cannot shadow a live assignment. Boolean-mode handling:
    ``default(x, false)`` is equivalent to ``default(x)``; ``default(false,
    true)`` still maps False -> False and True -> True, so it is a working
    default-off toggle; only ``default(true, true)`` collapses the toggle
    (an explicit False is replaced by True) and is not matched.
    """
    try:
        env = jinja2.Environment(
            extensions=[jinja2.ext.loopcontrols, _GenerationTagExtension]
        )
        tree = env.parse(ctx.template)
    except jinja2.TemplateError:
        return False
    except Exception:
        # e.g. RecursionError on a pathologically nested template; detection
        # must never take down server startup (mirrors
        # detect_inline_system_support).
        return False
    for node in tree.find_all(jinja2.nodes.Filter):
        if node.name not in ("default", "d") or node.dyn_args or node.dyn_kwargs:
            continue
        if not isinstance(node.node, jinja2.nodes.Name) or node.node.name != param:
            continue
        args = list(node.args)
        kwargs = {kw.key: kw.value for kw in node.kwargs}
        if not args or not isinstance(args[0], jinja2.nodes.Const):
            continue
        default_value = args[0].value
        if not isinstance(default_value, bool):
            continue
        boolean_arg = args[1] if len(args) > 1 else kwargs.get("boolean")
        if boolean_arg is not None and not isinstance(boolean_arg, jinja2.nodes.Const):
            continue
        boolean_mode = bool(boolean_arg.value) if boolean_arg is not None else False
        if boolean_mode and default_value:
            # default(true, true): any falsy value is replaced by True, so an
            # explicit enable_thinking=false still renders thinking on -- the
            # assignment is not a working toggle.
            continue
        if default_value is default:
            return True
    return False


# ---------------------------------------------------------------------------
# Reasoning mode rules (detect toggle config from template)
# 中译：推理模式规则集——从模板探测「推理开关配置（ReasoningToggleConfig）」。
#       按顺序匹配，命中第一条即返回其 value（一个 ReasoningToggleConfig）。
#       既覆盖「恒开」类模板（gpt-oss 通道标记、强制 <think>），也覆盖各类
#       enable_thinking / thinking 开关参数（默认 true / 默认 false）。
# ---------------------------------------------------------------------------

REASONING_MODE_RULES = (
    DetectionRule(
        name="gpt_oss_channel_markers",
        value=ReasoningToggleConfig(special_case="always"),
        predicate=lambda ctx: ctx.has_text("<|channel|>"),
    ),
    DetectionRule(
        name="force_reasoning_pattern",
        value=ReasoningToggleConfig(special_case="always"),
        predicate=lambda ctx: ctx.has_pattern(r"<\|im_start\|>assistant\\n<think>\\n")
        and not ctx.has_text("enable_thinking")
        and not ctx.has_text("thinking"),
    ),
    DetectionRule(
        name="mistral_reasoning_effort",
        value=ReasoningToggleConfig(special_case="mistral"),
        predicate=lambda ctx: ctx.has_text("reasoning_effort")
        and ctx.has_text("[THINK]"),
    ),
    DetectionRule(
        name="explicit_enable_thinking_default_false",
        value=ReasoningToggleConfig(
            toggle_param="enable_thinking", default_enabled=False
        ),
        predicate=lambda ctx: ctx.has_pattern(
            r"{%\s*if\s+not\s+enable_thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+enable_thinking\s*=\s*(?:false|False)\s*%}",
            re.DOTALL,
        )
        or _has_toggle_default_assignment(ctx, "enable_thinking", False),
    ),
    DetectionRule(
        name="nemotron_3_super_low_effort",
        value=ReasoningToggleConfig(
            toggle_param="enable_thinking",
            default_enabled=True,
            effort_kwarg="low_effort",
        ),
        predicate=lambda ctx: ctx.has_text("low_effort")
        and ctx.has_text("truncate_history_thinking"),
    ),
    DetectionRule(
        name="enable_thinking_default_true",
        value=ReasoningToggleConfig(
            toggle_param="enable_thinking", default_enabled=True
        ),
        predicate=lambda ctx: ctx.has_pattern(
            r"{%\s*if\s+not\s+enable_thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+enable_thinking\s*=\s*(?:true|True)\s*%}",
            re.DOTALL,
        )
        or ctx.has_pattern(
            r"set\s+enable_thinking\s*=\s*enable_thinking\s+if\s+enable_thinking\s+is\s+defined\s+else\s+(?:true|True)"
        )
        or ctx.has_pattern(
            r"enable_thinking\s+is\s+defined\s+and\s+(?:enable_thinking\s+is\s+false|not\s+enable_thinking)"
        )
        or ctx.has_pattern(
            r"enable_thinking\s+is\s+not\s+defined\s+or\s+enable_thinking"
        )
        or ctx.has_pattern(r"namespace\([^)]*enable_thinking\s*=\s*true")
        or _has_toggle_default_assignment(ctx, "enable_thinking", True),
    ),
    DetectionRule(
        name="explicit_thinking_default_false",
        value=ReasoningToggleConfig(toggle_param="thinking", default_enabled=False),
        predicate=lambda ctx: ctx.has_pattern(
            r"{%\s*if\s+not\s+thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+thinking\s*=\s*(?:false|False)\s*%}",
            re.DOTALL,
        )
        or _has_toggle_default_assignment(ctx, "thinking", False),
    ),
    DetectionRule(
        name="thinking_default_true",
        value=ReasoningToggleConfig(toggle_param="thinking", default_enabled=True),
        predicate=lambda ctx: ctx.has_pattern(
            r"{%\s*if\s+not\s+thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+thinking\s*=\s*(?:true|True)\s*%}",
            re.DOTALL,
        )
        or ctx.has_pattern(
            r"set\s+thinking\s*=\s*thinking\s+if\s+thinking\s+is\s+defined\s+else\s+(?:true|True)"
        )
        or ctx.has_pattern(
            r"thinking\s+is\s+defined\s+and\s+(?:thinking\s+is\s+false|not\s+thinking)"
        )
        or ctx.has_pattern(r"thinking\s+is\s+not\s+defined\s+or\s+thinking")
        or ctx.has_pattern(r"namespace\([^)]*thinking\s*=\s*true")
        or _has_toggle_default_assignment(ctx, "thinking", True),
    ),
)


# ---------------------------------------------------------------------------
# Shared predicates for model-family detection
# 中译：模型族识别用的共享谓词函数。
#       这些函数被 reasoning 与 tool-call 两个规则集复用（同一个谓词，可对应不同的 value）。
#       每个 _is_xxx(ctx) 返回该模板/词表是否符合某模型族的特征。
# ---------------------------------------------------------------------------


def _is_apertus2509(ctx):
    # 中译：Apertus-2509 族——词表含特殊 token <|inner_prefix|>。
    return ctx.has_vocab("<|inner_prefix|>")


def _is_gemma4(ctx):
    # 中译：Gemma-4 族——模板含 <|channel> 文本。
    return ctx.has_text("<|channel>")


def _is_kimi(ctx):
    # 中译：Kimi 族——模板含其特有的思考标记 ◁think▷。
    return ctx.has_text("◁think▷")


def _is_interns1(ctx):
    # 中译：InternS1 族——模板含 default_thinking_sys，且推理配置恰为 enable_thinking 默认开启。
    return ctx.has_text("default_thinking_sys") and ctx.reasoning_config == (
        ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True)
    )


def _is_mistral(ctx):
    # 中译：Mistral 族——推理配置的 special_case 标记为 "mistral"。
    return (
        ctx.reasoning_config is not None
        and ctx.reasoning_config.special_case == "mistral"
    )


def _is_gpt_oss(ctx):
    # 中译：gpt-oss 族——模板含通道标记 <|channel|>。
    return ctx.has_text("<|channel|>")


def _is_kimi_k2(ctx):
    # 中译：Kimi-K2 族——词表含工具调用段起始 token <|tool_calls_section_begin|>。
    return ctx.has_vocab("<|tool_calls_section_begin|>")


def _is_nemotron_3(ctx):
    # 中译：Nemotron-3 族——模板含 truncate_history_thinking，且推理配置为 enable_thinking 默认开启。
    return ctx.has_text("truncate_history_thinking") and (
        ctx.reasoning_config is not None
        and ctx.reasoning_config.toggle_param == "enable_thinking"
        and ctx.reasoning_config.default_enabled is True
    )


def _is_glm45(ctx):
    # 中译：GLM-4.5/4.6 族——综合判断多种特征（详见函数体内联注释）。
    return (
        (
            ctx.has_text("[gMASK]<sop>")
            or ctx.has_pattern(r"(?<!<)/nothink")
            or ctx.has_pattern(r"(?<!<)/think")
        )
        and ctx.has_vocab("<tool_call>")
        and ctx.reasoning_config
        == ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True)
        and (ctx.has_vocab("<|user|>") or ctx.has_vocab("<|endoftext|>"))
    )


def _is_glm47(ctx):
    return _is_glm45(ctx) and ctx.has_pattern(
        r"\{\{[-\s]*['\"]<tool_call>['\"]\s*\+\s*tc\.name"
    )


def _is_xml_kv_tool_call(ctx):
    # Structural signature for the GLM-4.5 / GLM-4.6 style tool-call format
    # (`<tool_call>name<arg_key>k</arg_key>\n<arg_value>v</arg_value>...</tool_call>`).
    # Matches any model whose tokenizer carries `<arg_key>` and `<arg_value>` as
    # added tokens — e.g., inclusionAI/Ring-2.6, which borrows GLM's tool-call
    # format but doesn't share the `[gMASK]<sop>` / `enable_thinking` family
    # signature checked by `_is_glm45`.
    # 中译：识别 GLM-4.5/4.6 风格工具调用格式的「结构性特征」
    #       （形如 <tool_call>name<arg_key>k</arg_key>\n<arg_value>v</arg_value>...</tool_call>）。
    #       只要 tokenizer 把 <arg_key> 与 <arg_value> 作为附加 token 收录即命中——
    #       例如 inclusionAI/Ring-2.6 借用了 GLM 的工具调用格式，但不具备 _is_glm45 检查的
    #       [gMASK]<sop> / enable_thinking 等模型族特征，因此用本谓词单独覆盖。
    return ctx.has_vocab("<arg_key>") and ctx.has_vocab("<arg_value>")


def _is_deepseek_v31(ctx):
    return ctx.has_text("<｜tool▁calls▁begin｜>") and ctx.has_text("<｜tool▁sep｜>")


def _is_deepseek_v32(ctx):
    return ctx.has_text("<｜DSML｜function_calls>")


def _is_deepseek_v4(ctx):
    return ctx.has_text("<｜DSML｜tool_calls>")


def _is_hunyuan(ctx):
    # The shipping Hy3 tokenizer appends a shared suffix to each special token
    # (e.g. ``<tool_calls:opensource>``), so match the bare or suffixed form.
    tc = ctx.has_text("<tool_calls>") or ctx.has_vocab_pattern(
        r"^<tool_calls(?::[^>]+)?>$"
    )
    sep = ctx.has_text("<tool_sep>") or ctx.has_vocab_pattern(
        r"^<tool_sep(?::[^>]+)?>$"
    )
    return (tc and sep) or (
        ctx.has_text("reasoning_effort") and ctx.has_text("interleaved_thinking")
    )


def _is_poolside_v1(ctx):
    has_poolside_tool_format = (
        ctx.has_text("<arg_key>")
        and ctx.has_text("<arg_value>")
        and (
            # Laguna-XS.2 spells out the tool-call format in prose;
            # Laguna-S-2.1 does not, so also key on the tool preamble both
            # template families share. The Poolside identity must not depend
            # on the enable_thinking default, which differs between families.
            ctx.has_text("unescaped XML-like object")
            or ctx.has_text("All available function signatures are listed below")
        )
    )
    return has_poolside_tool_format or (
        ctx.reasoning_config
        == ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=False)
        and not _is_hunyuan(ctx)
        and (ctx.has_text("<arg_key>") or ctx.has_vocab("<arg_key>"))
        and (ctx.has_text("<arg_value>") or ctx.has_vocab("<arg_value>"))
    )


def _is_mimo(ctx):
    # 中译：MiMo 族——推理配置为 enable_thinking 且默认关闭。
    return ctx.reasoning_config == ReasoningToggleConfig(
        toggle_param="enable_thinking", default_enabled=False
    )


def _is_minimax(ctx):
    # 中译：MiniMax 族——模板含其工具调用标记 <minimax:tool_call>。
    return ctx.has_text("<minimax:tool_call>")


def _is_minimax_m3(ctx):
    return ctx.has_text("<mm:think>") or ctx.has_text("]<]minimax[>[")


def _is_minicpm5(ctx):
    # 中译：MiniCPM-5 族——词表含 <function 与 <param，或模板出现 <function name= / <param name= 形式。
    if ctx.has_vocab("<function") and ctx.has_vocab("<param"):
        return True
    return ctx.has_pattern(r"<function\s+name=") and ctx.has_pattern(r"<param\s+name=")


def _is_lfm2(ctx):
    return (
        ctx.has_text("<|tool_call_start|>") or ctx.has_vocab("<|tool_call_start|>")
    ) and (ctx.has_text("<|tool_call_end|>") or ctx.has_vocab("<|tool_call_end|>"))


def _is_step3p5(ctx):
    return ctx.has_pattern(r"Step-?3(?:\.|p)?[57]", re.IGNORECASE) or (
        ctx.has_text("reasoning_effort")
        and ctx.has_text("Reasoning: ")
        and _is_qwen3_coder(ctx)
    )


def _is_step3(ctx):
    return ctx.has_text("<steptml:invoke") or (
        ctx.has_text("<｜tool_calls_begin｜>") and ctx.has_text("<｜tool_sep｜>")
    )


def _is_qwen3_coder(ctx):
    return ctx.has_text("<function=") and ctx.has_text("<parameter=")


def _is_qwen3(ctx):
    # 中译：Qwen3 族——推理配置为 enable_thinking 且默认开启。
    return ctx.reasoning_config == ReasoningToggleConfig(
        toggle_param="enable_thinking", default_enabled=True
    )


def _is_deepseek_v3(ctx):
    # 中译：DeepSeek-V3 族——推理配置为 thinking 且默认关闭。
    return ctx.reasoning_config == ReasoningToggleConfig(
        toggle_param="thinking", default_enabled=False
    )


def _is_deepseek_r1(ctx):
    # 中译：DeepSeek-R1 族——模板强制开启推理（force_reasoning 为真）。
    return ctx.force_reasoning


def _is_deepseek_r1_think_tags(ctx):
    # 中译：DeepSeek-R1 族（兜底）——模板直接含 <think> 或 </think> 标签。
    return not _is_lfm2(ctx) and (ctx.has_text("<think>") or ctx.has_text("</think>"))


# ---------------------------------------------------------------------------
# Reasoning parser rules
# 中译：推理解析器规则集——按顺序匹配，命中第一条即返回对应的解析器名（value）。
#       顺序很重要：更具体/优先级更高的模型族规则应排在更宽泛的兜底规则之前。
# ---------------------------------------------------------------------------

REASONING_PARSER_RULES = (
    DetectionRule(name="apertus2509", value="apertus2509", predicate=_is_apertus2509),
    DetectionRule(name="gemma4", value="gemma4", predicate=_is_gemma4),
    DetectionRule(name="kimi", value="kimi", predicate=_is_kimi),
    DetectionRule(name="interns1", value="interns1", predicate=_is_interns1),
    DetectionRule(name="mistral", value="mistral", predicate=_is_mistral),
    DetectionRule(name="gpt_oss", value="gpt-oss", predicate=_is_gpt_oss),
    DetectionRule(name="kimi_k2", value="kimi_k2", predicate=_is_kimi_k2),
    DetectionRule(name="nemotron_3", value="nemotron_3", predicate=_is_nemotron_3),
    DetectionRule(name="glm45", value="glm45", predicate=_is_glm45),
    DetectionRule(name="hunyuan", value="hunyuan", predicate=_is_hunyuan),
    DetectionRule(name="poolside_v1", value="poolside_v1", predicate=_is_poolside_v1),
    DetectionRule(name="mimo", value="mimo", predicate=_is_mimo),
    DetectionRule(name="minimax_m3", value="minimax-m3", predicate=_is_minimax_m3),
    DetectionRule(name="minimax", value="minimax", predicate=_is_minimax),
    DetectionRule(name="step3p5", value="step3p5", predicate=_is_step3p5),
    DetectionRule(name="step3", value="step3", predicate=_is_step3),
    DetectionRule(name="qwen3", value="qwen3", predicate=_is_qwen3),
    DetectionRule(name="deepseek_v4", value="deepseek-v4", predicate=_is_deepseek_v4),
    DetectionRule(name="deepseek_v3", value="deepseek-v3", predicate=_is_deepseek_v3),
    DetectionRule(
        name="deepseek_r1_force", value="deepseek-r1", predicate=_is_deepseek_r1
    ),
    DetectionRule(
        name="deepseek_r1_think_tags",
        value="deepseek-r1",
        predicate=_is_deepseek_r1_think_tags,
    ),
)

# ---------------------------------------------------------------------------
# Tool-call parser rules (reuse shared predicates, different values)
# 中译：工具调用解析器规则集——复用上面的共享谓词，但返回的 value（解析器名）可能不同。
#       例如同一个 _is_minimax 谓词，在推理规则里对应 "minimax"，这里对应 "minimax-m2"。
# ---------------------------------------------------------------------------

TOOL_CALL_PARSER_RULES = (
    DetectionRule(name="apertus2509", value="apertus2509", predicate=_is_apertus2509),
    DetectionRule(name="gemma4", value="gemma4", predicate=_is_gemma4),
    DetectionRule(name="gpt_oss", value="gpt-oss", predicate=_is_gpt_oss),
    DetectionRule(name="kimi_k2", value="kimi_k2", predicate=_is_kimi_k2),
    DetectionRule(name="minimax_m3", value="minimax-m3", predicate=_is_minimax_m3),
    DetectionRule(name="minimax", value="minimax-m2", predicate=_is_minimax),
    DetectionRule(name="interns1", value="interns1", predicate=_is_interns1),
    DetectionRule(name="mistral", value="mistral", predicate=_is_mistral),
    DetectionRule(name="deepseek_v4", value="deepseekv4", predicate=_is_deepseek_v4),
    DetectionRule(name="deepseek_v32", value="deepseekv32", predicate=_is_deepseek_v32),
    DetectionRule(name="deepseek_v31", value="deepseekv31", predicate=_is_deepseek_v31),
    DetectionRule(name="lfm2", value="lfm2", predicate=_is_lfm2),
    DetectionRule(name="glm47", value="glm47", predicate=_is_glm47),
    DetectionRule(name="glm45", value="glm45", predicate=_is_glm45),
    DetectionRule(name="minicpm5", value="minicpm5", predicate=_is_minicpm5),
    DetectionRule(name="hunyuan", value="hunyuan", predicate=_is_hunyuan),
    DetectionRule(name="poolside_v1", value="poolside_v1", predicate=_is_poolside_v1),
    DetectionRule(name="step3p5", value="step3p5", predicate=_is_step3p5),
    DetectionRule(name="step3", value="step3", predicate=_is_step3),
    DetectionRule(
        name="xml_kv_tool_call", value="glm45", predicate=_is_xml_kv_tool_call
    ),
    DetectionRule(name="mimo", value="mimo", predicate=_is_mimo),
    DetectionRule(name="qwen3_coder", value="qwen3_coder", predicate=_is_qwen3_coder),
    DetectionRule(name="qwen", value="qwen", predicate=_is_qwen3),
    DetectionRule(name="deepseek_v3", value="deepseekv3", predicate=_is_deepseek_v3),
    DetectionRule(name="deepseek_r1", value="deepseekv3", predicate=_is_deepseek_r1),
)


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------


def build_detection_context(
    template: Optional[str],
    tokenizer,
    reasoning_config: Optional[ReasoningToggleConfig] = None,
    force_reasoning: bool = False,
) -> Optional[TemplateDetectionContext]:
    """中译：构建模板探测上下文 TemplateDetectionContext。

    参数：
      template：聊天模板字符串（为 None 时直接返回 None，表示无法探测）。
      tokenizer：分词器，用于取词表 vocab；取词表失败仅告警并退化为空集合（跳过依赖词表的规则）。
      reasoning_config / force_reasoning：可选的已知推理信息，会被放入上下文供部分规则复用。
    返回：构建好的上下文；template 为 None 时返回 None。
    """
    if template is None:
        return None
    vocab = set()
    if tokenizer is not None:
        try:
            # 中译：取 tokenizer 词表的全部 token 作为集合，供 has_vocab 判定使用。
            vocab = set(tokenizer.get_vocab().keys())
        except Exception as e:
            # 中译：取词表失败不致命——记 warning 并退化为空集合，依赖词表的规则会被跳过。
            logger.warning(
                "Failed to load tokenizer vocab for template detection: %s. "
                "Vocab-dependent detection rules will be skipped.",
                e,
            )
    return TemplateDetectionContext(
        template=template,
        reasoning_config=reasoning_config,
        force_reasoning=force_reasoning,
        vocab=vocab,
    )


def match_rules(
    ctx: TemplateDetectionContext,
    rules: Tuple[DetectionRule, ...],
    label: str,
) -> Optional[str]:
    """中译：按顺序遍历规则集，返回第一条命中规则的 value；都不命中返回 None。

    参数：
      ctx：探测上下文。
      rules：有序规则元组（顺序即优先级）。
      label：用于日志的标签（如 "reasoning parser" / "tool-call parser"）。
    健壮性：单条规则谓词抛异常时仅记 warning 并跳过，不影响后续规则匹配。
    """
    for rule in rules:
        try:
            if rule.predicate(ctx):
                return rule.value
        except Exception as e:
            # 中译：某条规则谓词执行出错——记录并跳过该规则，继续尝试后续规则。
            logger.warning(
                "Detection rule '%s' for %s raised an exception: %s. Skipping.",
                rule.name,
                label,
                e,
                exc_info=True,
            )
    return None


def detect_reasoning_pattern(
    template: Optional[str],
) -> Tuple[bool, Optional[ReasoningToggleConfig]]:
    """Detect if the chat template contains reasoning/thinking patterns.

    中译：探测聊天模板是否含有推理（thinking）模式。
          返回二元组 (force_reasoning, reasoning_config)：
            - force_reasoning：是否强制开启推理（取命中规则配置的 always_on）。
            - reasoning_config：命中的推理开关配置；都不命中则返回 (False, None)。
          注意：本函数只用模板文本即可判定，故构造的上下文 vocab 为空集合。
    """
    if template is None:
        return False, None

    ctx = TemplateDetectionContext(
        template=template,
        reasoning_config=None,
        force_reasoning=False,
        vocab=set(),
    )
    # 中译：按顺序匹配推理模式规则，命中第一条即返回其 (always_on, 配置)。
    for rule in REASONING_MODE_RULES:
        if rule.predicate(ctx):
            return rule.value.always_on, rule.value

    return False, None


def detect_reasoning_parser(
    template: Optional[str],
    tokenizer,
    reasoning_config: Optional[ReasoningToggleConfig] = None,
    force_reasoning: bool = False,
) -> Optional[str]:
    """Auto-detect which reasoning parser to use from the chat template.

    中译：从聊天模板自动探测应使用的「推理解析器」名（未识别返回 None）。
          先构建探测上下文，再按 REASONING_PARSER_RULES 匹配。
    """
    ctx = build_detection_context(
        template, tokenizer, reasoning_config, force_reasoning
    )
    if ctx is None:
        return None
    return match_rules(ctx, REASONING_PARSER_RULES, "reasoning parser")


def detect_tool_call_parser(
    template: Optional[str],
    tokenizer,
    reasoning_config: Optional[ReasoningToggleConfig] = None,
    force_reasoning: bool = False,
) -> Optional[str]:
    """Auto-detect which tool-call parser to use from the chat template.

    中译：从聊天模板自动探测应使用的「工具调用解析器」名（未识别返回 None）。
          先构建探测上下文，再按 TOOL_CALL_PARSER_RULES 匹配。
    """
    ctx = build_detection_context(
        template, tokenizer, reasoning_config, force_reasoning
    )
    if ctx is None:
        return None
    return match_rules(ctx, TOOL_CALL_PARSER_RULES, "tool-call parser")


def detect_inline_system_support(chat_template: Optional[str]) -> bool:
    """True if mid-conversation ``role: "system"`` renders inline; False if the
    template raises or silently drops it (then merge into the leading block).

    The probe requires the second system's sentinel to appear in the output —
    not raising isn't enough, since some templates ignore non-leading system."""
    if not chat_template:
        return False
    sentinel = "__sglang_inline_system_sentinel__"
    try:
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[jinja2.ext.loopcontrols],
        )
        rendered = env.from_string(chat_template).render(
            messages=[
                {"role": "system", "content": "t"},
                {"role": "user", "content": "t"},
                {"role": "system", "content": sentinel},
                {"role": "user", "content": "t"},
            ],
            add_generation_prompt=False,
        )
        return sentinel in rendered
    except jinja2.TemplateError:
        return False
    except Exception:
        return False


def _detect_auto_parser(
    attr: str,
    ctx: TemplateDetectionContext,
    rules: Tuple[DetectionRule, ...],
    label: str,
        # 中译：探测成功——写回属性并记录。
) -> Optional[str]:
    """The parser one auto field resolves to (``None`` disables it)."""
    detected = match_rules(ctx, rules, label)
    if detected:
        logger.info(
            f"Auto-detected --{attr.replace('_', '-')} as '{detected}' from chat template"
        )
        return detected
    _log_undetected_parser(attr, label)
    return None


def _load_explicit_jinja_template(chat_template_arg: Optional[str]) -> Optional[str]:
    if not chat_template_arg or not isinstance(chat_template_arg, str):
        return None
    if not chat_template_arg.endswith(".jinja") or not os.path.exists(
        chat_template_arg
    ):
        return None
    with open(chat_template_arg, encoding="utf-8") as f:
        return f.read().replace("\\n", "\n")


def _log_undetected_parser(attr: str, label: str) -> None:
    logger.warning(
        f"--{attr.replace('_', '-')}=auto specified but could not detect "
        f"{label} from chat template. Disabling {label}."
    )


def _architecture_auto_parsers(server_args, needs: Tuple[str, ...]) -> Dict[str, str]:
    """The parsers the model architecture implies, for the fields still on auto."""
    from sglang.srt.utils.hf_transformers_utils import get_config

    cfg = resolving_view(server_args)
    config = get_config(
        cfg.model_path,
        trust_remote_code=cfg.trust_remote_code,
        revision=getattr(cfg, "revision", None),
        model_config_parser=getattr(cfg, "model_config_parser", "auto"),
    )
    architectures = getattr(config, "architectures", None) or []
    arch = architectures[0] if architectures else ""
    model_type = getattr(config, "model_type", "")

    if "KimiK3" in arch or model_type == "kimi_k3":
        reasoning_parser, tool_call_parser = "kimi_k3", "kimi_k3"
    elif "DeepseekV4" in arch:
        reasoning_parser, tool_call_parser = "deepseek-v4", "deepseekv4"
    elif "DeepseekV3" in arch:
        reasoning_parser, tool_call_parser = "deepseek-v3", "deepseekv32"
    else:
        # 中译：用户指定了 =auto 但探测失败——禁用该解析器（置 None）并告警。
        return {}

    resolved = {}
    for attr, detected in (
        ("reasoning_parser", reasoning_parser),
        ("tool_call_parser", tool_call_parser),
    ):
        if attr in needs:
            resolved[attr] = detected
            logger.info(
                f"Auto-detected --{attr.replace('_', '-')} as '{detected}' "
                f"from model architecture '{arch}'"
            )
    return resolved


def resolve_auto_parsers(server_args) -> None:
    """Resolve ``--reasoning-parser=auto`` / ``--tool-call-parser=auto`` from the
    chat template, before anything publishes ``server_args``.

    # 中译：两项都无需自动探测，直接返回。
    Performs a lightweight tokenizer load, so it runs once in engine init. The
    decision goes to this instance's declaration stash, so every holder of it
    carries it -- the schedulers it forks, the HTTP server, the tokenizer
    workers it is serialized for -- and each publishes bags projected from it.
    The fields stay what the operator passed.
    """
    cfg = resolving_view(server_args)
    needs = tuple(
        attr
        for attr in ("reasoning_parser", "tool_call_parser")
        if getattr(cfg, attr) == "auto"
    )
    if not needs:
        return

    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    chat_template_arg = getattr(cfg, "chat_template", None)
    try:
        explicit_jinja_template = _load_explicit_jinja_template(chat_template_arg)
    except Exception as e:
        logger.warning("Failed to load explicit Jinja chat template: %s", e)
        explicit_jinja_template = None
    has_explicit_template_without_detection = (
        chat_template_arg is not None and explicit_jinja_template is None
    )

    tokenizer = None
    try:
        # 中译：轻量加载 tokenizer 并取其聊天模板，作为探测输入。
        tokenizer = get_tokenizer(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
        )
    except Exception as e:
        # 中译：tokenizer 加载失败——无法探测，把需要 auto 的项分别禁用为 None 并告警后返回。
        logger.warning(f"Failed to load tokenizer for auto-detection: {e}")

    template = explicit_jinja_template
    if template is None and tokenizer is not None:
        template = getattr(tokenizer, "chat_template", None)

    # 中译：先探测推理模式，再用它构建完整探测上下文（含词表），供后续解析器解析复用。
    force_reasoning, reasoning_config = detect_reasoning_pattern(template)
    ctx = build_detection_context(
        template, tokenizer, reasoning_config, force_reasoning
    )

    detected: Dict[str, Optional[str]] = {}
    if ctx is None:
        if has_explicit_template_without_detection:
            logger.warning(
                "--chat-template=%s is explicit but is not a readable Jinja template, so "
                "parser auto-detection from chat template is not available.",
                chat_template_arg,
            )
        else:
            try:
                detected.update(_architecture_auto_parsers(server_args, needs))
            except Exception as e:
                logger.warning(
                    "Failed to load model config for architecture-based auto-detection: %s",
                    e,
                )
        for attr, label in (
            ("reasoning_parser", "reasoning parser"),
            ("tool_call_parser", "tool-call parser"),
        ):
            if attr in needs and attr not in detected:
                _log_undetected_parser(attr, label)
                detected[attr] = None
    else:
        for attr, rules, label in (
            ("reasoning_parser", REASONING_PARSER_RULES, "reasoning parser"),
            ("tool_call_parser", TOOL_CALL_PARSER_RULES, "tool-call parser"),
        ):
            if attr in needs:
                detected[attr] = _detect_auto_parser(attr, ctx, rules, label)

    # 中译：分别解析需要 auto 的解析器项（就地写回 server_args）。
    if detected:
        declare_late_resolution(server_args, "template-detection", **detected)
