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
import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

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

    @property
    def always_on(self) -> bool:
        # 中译：是否「恒定开启」推理（special_case == "always"）。
        return self.special_case == "always"


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
        ),
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
        or ctx.has_pattern(r"namespace\([^)]*enable_thinking\s*=\s*true"),
    ),
    DetectionRule(
        name="explicit_thinking_default_false",
        value=ReasoningToggleConfig(toggle_param="thinking", default_enabled=False),
        predicate=lambda ctx: ctx.has_pattern(
            r"{%\s*if\s+not\s+thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+thinking\s*=\s*(?:false|False)\s*%}",
            re.DOTALL,
        ),
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
        or ctx.has_pattern(r"namespace\([^)]*thinking\s*=\s*true"),
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
    return ctx.has_text("truncate_history_thinking") and ctx.reasoning_config == (
        ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True)
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


def _is_mimo(ctx):
    # 中译：MiMo 族——推理配置为 enable_thinking 且默认关闭。
    return ctx.reasoning_config == ReasoningToggleConfig(
        toggle_param="enable_thinking", default_enabled=False
    )


def _is_minimax(ctx):
    # 中译：MiniMax 族——模板含其工具调用标记 <minimax:tool_call>。
    return ctx.has_text("<minimax:tool_call>")


def _is_minicpm5(ctx):
    # 中译：MiniCPM-5 族——词表含 <function 与 <param，或模板出现 <function name= / <param name= 形式。
    if ctx.has_vocab("<function") and ctx.has_vocab("<param"):
        return True
    return ctx.has_pattern(r"<function\s+name=") and ctx.has_pattern(r"<param\s+name=")


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
    return ctx.has_text("<think>") or ctx.has_text("</think>")


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
    DetectionRule(name="mimo", value="mimo", predicate=_is_mimo),
    DetectionRule(name="minimax", value="minimax", predicate=_is_minimax),
    DetectionRule(name="qwen3", value="qwen3", predicate=_is_qwen3),
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
    DetectionRule(name="minimax", value="minimax-m2", predicate=_is_minimax),
    DetectionRule(name="interns1", value="interns1", predicate=_is_interns1),
    DetectionRule(name="mistral", value="mistral", predicate=_is_mistral),
    DetectionRule(name="glm45", value="glm45", predicate=_is_glm45),
    DetectionRule(name="minicpm5", value="minicpm5", predicate=_is_minicpm5),
    DetectionRule(
        name="xml_kv_tool_call", value="glm45", predicate=_is_xml_kv_tool_call
    ),
    DetectionRule(name="mimo", value="mimo", predicate=_is_mimo),
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


def _resolve_auto_parser(
    server_args,
    attr: str,
    ctx: TemplateDetectionContext,
    rules: Tuple[DetectionRule, ...],
    label: str,
) -> None:
    """Resolve a single auto parser, updating server_args in place.

    中译：解析单个「=auto」解析器配置，并就地更新 server_args 上对应属性。
          参数：
            server_args：服务端参数对象（将被原地修改）。
            attr：要写入的属性名（如 "reasoning_parser" / "tool_call_parser"）。
            ctx / rules / label：探测上下文、规则集、日志标签。
          命中则把探测到的解析器名写入属性；未命中则写入 None（即禁用该解析器）并告警。
    """
    detected = match_rules(ctx, rules, label)
    if detected:
        # 中译：探测成功——写回属性并记录。
        setattr(server_args, attr, detected)
        logger.info(
            f"Auto-detected --{attr.replace('_', '-')} as '{detected}' from chat template"
        )
    else:
        # 中译：用户指定了 =auto 但探测失败——禁用该解析器（置 None）并告警。
        logger.warning(
            f"--{attr.replace('_', '-')}=auto specified but could not detect "
            f"{label} from chat template. Disabling {label}."
        )
        setattr(server_args, attr, None)


def resolve_auto_parsers(server_args) -> None:
    """Resolve --reasoning-parser=auto and --tool-call-parser=auto before scheduler.

    This performs a lightweight tokenizer load to detect parsers from the chat
    template. Called early in engine init before scheduler subprocesses are spawned.

    中译：在调度器启动前解析 --reasoning-parser=auto 与 --tool-call-parser=auto。
          会做一次轻量级的 tokenizer 加载，从聊天模板中探测解析器；在引擎初始化早期、
          调度器子进程被 fork 之前调用，从而让探测结果通过 server_args 传递给子进程。
          副作用：就地修改 server_args 的 reasoning_parser / tool_call_parser 属性。
          若两者都不是 "auto" 则直接返回；若 tokenizer 加载失败，则把对应的 auto 项禁用为 None。
    """
    needs_reasoning = server_args.reasoning_parser == "auto"
    needs_tool_call = server_args.tool_call_parser == "auto"

    # 中译：两项都无需自动探测，直接返回。
    if not needs_reasoning and not needs_tool_call:
        return

    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    try:
        # 中译：轻量加载 tokenizer 并取其聊天模板，作为探测输入。
        tokenizer = get_tokenizer(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
        )
        template = getattr(tokenizer, "chat_template", None)
    except Exception as e:
        # 中译：tokenizer 加载失败——无法探测，把需要 auto 的项分别禁用为 None 并告警后返回。
        logger.warning(f"Failed to load tokenizer for auto-detection: {e}")
        if needs_reasoning:
            logger.warning(
                "--reasoning-parser=auto specified but could not detect "
                "reasoning parser from chat template. Disabling reasoning parser."
            )
            server_args.reasoning_parser = None
        if needs_tool_call:
            logger.warning(
                "--tool-call-parser=auto specified but could not detect "
                "tool-call parser from chat template. Disabling tool-call parser."
            )
            server_args.tool_call_parser = None
        return

    # 中译：先探测推理模式，再用它构建完整探测上下文（含词表），供后续解析器解析复用。
    force_reasoning, reasoning_config = detect_reasoning_pattern(template)
    ctx = build_detection_context(
        template, tokenizer, reasoning_config, force_reasoning
    )
    if ctx is None:
        return

    # 中译：分别解析需要 auto 的解析器项（就地写回 server_args）。
    if needs_reasoning:
        _resolve_auto_parser(
            server_args,
            "reasoning_parser",
            ctx,
            REASONING_PARSER_RULES,
            "reasoning parser",
        )

    if needs_tool_call:
        _resolve_auto_parser(
            server_args,
            "tool_call_parser",
            ctx,
            TOOL_CALL_PARSER_RULES,
            "tool-call parser",
        )
