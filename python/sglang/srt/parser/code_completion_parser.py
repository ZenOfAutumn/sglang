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
# =============================================================================
# 代码补全模板（FIM，Fill-In-the-Middle / 中间填充）
#
# 代码补全场景下，输入分为“前缀代码（prefix/prompt）”与“后缀代码（suffix）”，
# 模型需要填补中间缺失的部分。不同模型用不同的特殊 token 标记 prefix/middle/suffix，
# 且拼接顺序不同（MIDDLE 或 END 两种风格）。
# 本模块维护一个全局模板注册表，并根据模板把请求组装为 FIM 提示词。
# =============================================================================

"""Completion templates."""

import dataclasses
import logging
from enum import Enum, auto
from typing import Optional

from sglang.srt.entrypoints.openai.protocol import CompletionRequest

logger = logging.getLogger(__name__)
# 全局当前使用的补全模板名（启动时设置一次）。
completion_template_name: Optional[str] = None


class FimPosition(Enum):
    """FIM middle token 的位置：MIDDLE 表示拼在中间，END 表示拼在末尾。"""

    MIDDLE = auto()
    END = auto()


@dataclasses.dataclass
class CompletionTemplate:
    """补全提示词模板（目前仅用于代码补全）。"""

    # 模板名称。
    name: str

    # FIM 起始（prefix）标记。
    fim_begin_token: str

    # FIM 中间（middle）标记。
    fim_middle_token: str

    # FIM 结束（suffix）标记。
    fim_end_token: str

    # middle 标记的位置（决定拼接顺序）。
    fim_position: FimPosition


# 全局补全模板注册表：模板名 → 模板对象。
completion_templates: dict[str, CompletionTemplate] = {}


def register_completion_template(template: CompletionTemplate, override: bool = False):
    """注册一个补全模板；默认不允许重复注册（override=True 可覆盖）。"""
    if not override:
        assert (
            template.name not in completion_templates
        ), f"{template.name} has been registered."

    completion_templates[template.name] = template


def completion_template_exists(template_name: str) -> bool:
    """判断指定名称的补全模板是否已注册。"""
    return template_name in completion_templates


def set_completion_template(template_name: str) -> None:
    """设置全局补全模板名（仅首次生效，后续调用不覆盖）。"""
    global completion_template_name
    if completion_template_name is None:
        completion_template_name = template_name


def is_completion_template_defined() -> bool:
    """是否已设置全局补全模板。"""
    global completion_template_name
    return completion_template_name is not None


def generate_completion_prompt_from_request(request: CompletionRequest) -> str:
    """从补全请求生成 FIM 提示词。无 suffix 时直接返回 prompt。"""
    global completion_template_name
    # 没有后缀，不是 FIM 场景，直接用原 prompt。
    if request.suffix == "":
        return request.prompt

    return generate_completion_prompt(
        request.prompt, request.suffix, completion_template_name
    )


def generate_completion_prompt(prompt: str, suffix: str, template_name: str) -> str:
    """根据模板把 prefix(prompt) 与 suffix 用 FIM 标记拼接成最终提示词。"""
    completion_template = completion_templates[template_name]
    fim_begin_token = completion_template.fim_begin_token
    fim_middle_token = completion_template.fim_middle_token
    fim_end_token = completion_template.fim_end_token
    fim_position = completion_template.fim_position

    # 两种拼接风格：
    # MIDDLE：begin + prefix + middle + suffix + end（如 DeepSeek Coder）。
    # END：begin + prefix + end + suffix + middle（如 StarCoder / Qwen Coder）。
    if fim_position == FimPosition.MIDDLE:
        prompt = f"{fim_begin_token}{prompt}{fim_middle_token}{suffix}{fim_end_token}"
    elif fim_position == FimPosition.END:
        prompt = f"{fim_begin_token}{prompt}{fim_end_token}{suffix}{fim_middle_token}"

    return prompt


# 以下预注册三个常见代码模型的 FIM 模板。
register_completion_template(
    CompletionTemplate(
        name="deepseek_coder",
        fim_begin_token="<｜fim▁begin｜>",
        fim_middle_token="<｜fim▁hole｜>",
        fim_end_token="<｜fim▁end｜>",
        fim_position=FimPosition.MIDDLE,
    )
)


register_completion_template(
    CompletionTemplate(
        name="star_coder",
        fim_begin_token="<fim_prefix>",
        fim_middle_token="<fim_middle>",
        fim_end_token="<fim_suffix>",
        fim_position=FimPosition.END,
    )
)

register_completion_template(
    CompletionTemplate(
        name="qwen_coder",
        fim_begin_token="<|fim_prefix|>",
        fim_middle_token="<|fim_middle|>",
        fim_end_token="<|fim_suffix|>",
        fim_position=FimPosition.END,
    )
)
