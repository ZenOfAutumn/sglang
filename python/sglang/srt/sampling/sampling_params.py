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
"""Sampling parameters for text generation."""
# 本文件定义了文本生成过程中的采样参数（SamplingParams），
# 包含参数的存储、合法性校验（verify）以及归一化处理（normalize），
# 并提供了基于正则表达式计算停止匹配所需最大 token 缓冲长度的辅助函数。

import logging
from typing import Any, Dict, List, Optional, Union

# sre_parse 在 Python 3.11+ 中已被弃用，需改用 re._parser；
# 这里通过 try/except 兼容新旧版本的 Python。
try:
    import re._parser as sre_parse
except ImportError:
    import sre_parse  # Python 3.11 之前的版本使用旧的 sre_parse 模块

# 温度参数的浮点判等阈值：当 temperature 小于该值时视为 0（即贪婪采样）。
_SAMPLING_EPS = 1e-6
# top_k 的“全词表”取值：表示对整个词表生效（即不限制候选 token 数量）。
TOP_K_ALL = 1 << 30

logger = logging.getLogger(__name__)


class SamplingParams:
    """
    采样参数集合。

    封装了控制文本生成行为的全部采样相关参数（如温度、top_p、top_k、
    各类惩罚项、停止条件、结构化约束等）。

    详细文档见 docs/backend/sampling_params.md 或
    https://docs.sglang.io/backend/sampling_params.html
    """

    def __init__(
        self,
        max_new_tokens: int = 128,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        stop_regex: Optional[Union[str, List[str]]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        n: int = 1,
        json_schema: Optional[str] = None,
        regex: Optional[str] = None,
        ebnf: Optional[str] = None,
        structural_tag: Optional[str] = None,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        no_stop_trim: bool = False,
        custom_params: Optional[Dict[str, Any]] = None,
        stream_interval: Optional[int] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        sampling_seed: Optional[int] = None,
    ) -> None:
        # 本次请求最多生成的新 token 数量。
        self.max_new_tokens = max_new_tokens
        # 停止字符串：生成内容中出现这些字符串时终止生成（可为单个字符串或字符串列表）。
        self.stop_strs = stop
        if stop_token_ids:
            # 停止 token id 集合：生成到其中任一 token 时终止；转为 set 便于快速查找。
            self.stop_token_ids = set(stop_token_ids)
        else:
            self.stop_token_ids = None
        # 停止正则：生成内容匹配这些正则时终止（可为单个或列表）。
        self.stop_regex_strs = stop_regex
        # 温度：控制采样随机性，越大越随机，越小越确定。
        self.temperature = temperature
        # top_p（核采样）：仅在累积概率达到 top_p 的最小候选集合中采样。
        self.top_p = top_p
        # top_k：仅在概率最高的 k 个候选 token 中采样；-1 表示不限制。
        self.top_k = top_k
        # min_p：过滤概率低于 (min_p * 最大概率) 的候选 token。
        self.min_p = min_p
        # 频率惩罚：根据 token 已出现的次数施加惩罚，降低重复。
        self.frequency_penalty = frequency_penalty
        # 存在惩罚：只要 token 出现过就施加惩罚，鼓励引入新内容。
        self.presence_penalty = presence_penalty
        # 重复惩罚：对已出现过的 token 的 logits 进行缩放以抑制重复。
        self.repetition_penalty = repetition_penalty
        # 本次请求至少要生成的新 token 数量（在此之前不允许出现 EOS/停止）。
        self.min_new_tokens = min_new_tokens
        # 正则约束：强制生成结果匹配该正则表达式。
        self.regex = regex
        # 为同一输入生成的候选序列数量。
        self.n = n
        # JSON Schema 约束：强制生成结果符合该 JSON Schema。
        self.json_schema = json_schema
        # EBNF 文法约束：强制生成结果符合该 EBNF 文法。
        self.ebnf = ebnf
        # 结构化标签约束：用于结构化输出的标签定义。
        self.structural_tag = structural_tag
        # 是否忽略 EOS：为 True 时遇到 EOS 不停止，继续生成直到达到 max_new_tokens。
        self.ignore_eos = ignore_eos
        # 解码时是否跳过特殊 token（如 <s>、</s> 等）。
        self.skip_special_tokens = skip_special_tokens
        # 解码时特殊 token 之间是否保留空格。
        self.spaces_between_special_tokens = spaces_between_special_tokens
        # 是否在命中停止条件时不裁剪掉停止串本身（保留停止字符串）。
        self.no_stop_trim = no_stop_trim
        # 自定义参数：透传给自定义 logits 处理器等扩展逻辑。
        self.custom_params = custom_params
        # 流式输出的间隔（每生成多少 token 推送一次）。
        self.stream_interval = stream_interval
        # logit 偏置：对指定 token id 的 logits 加上偏置值，键为 token id，值为偏置量。
        self.logit_bias = logit_bias
        # 采样随机种子：用于复现可重复的采样结果。
        self.sampling_seed = sampling_seed

        # 处理一些特殊情况
        if 0 <= self.temperature < _SAMPLING_EPS:
            # 当温度近似为 0 时退化为贪婪采样：
            # 将温度重置为 1.0 以避免后续除零，并将 top_k 设为 1（只取概率最高的 token）。
            self.temperature = 1.0
            self.top_k = 1
        if self.top_k == -1:
            # top_k 为 -1 表示不限制，等价于对整个词表生效。
            self.top_k = TOP_K_ALL  # 整个词表

    def verify(self, vocab_size):
        # 校验各采样参数的取值范围是否合法，非法时抛出 ValueError。
        # vocab_size：词表大小，用于校验 logit_bias 的 token id 范围。
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.top_k < 1 or self.top_k == -1:
            raise ValueError(
                f"top_k must be -1 (disable) or at least 1, got {self.top_k}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                "frequency_penalty must be in [-2, 2], got "
                f"{self.frequency_penalty}."
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}."
            )
        if not 0.0 <= self.repetition_penalty <= 2.0:
            raise ValueError(
                "repetition_penalty must be in [0, 2], got "
                f"{self.repetition_penalty}."
            )
        if not 0 <= self.min_new_tokens:
            raise ValueError(
                f"min_new_tokens must be in [0, max_new_tokens], got "
                f"{self.min_new_tokens}."
            )
        if self.max_new_tokens is not None:
            if self.max_new_tokens < 0:
                raise ValueError(
                    f"max_new_tokens must be at least 0, got {self.max_new_tokens}."
                )
            if not self.min_new_tokens <= self.max_new_tokens:
                raise ValueError(
                    f"min_new_tokens must be in [0, max_new_tokens({self.max_new_tokens})], got "
                    f"{self.min_new_tokens}."
                )
        if self.logit_bias is not None:
            for token_id in self.logit_bias:
                if not 0 <= int(token_id) < vocab_size:
                    raise ValueError(
                        f"logit_bias must has keys in [0, {vocab_size - 1}], got "
                        f"{token_id}."
                    )

        # json_schema、regex、ebnf 三种结构化约束互斥，最多只能设置其中一种。
        grammars = [
            self.json_schema,
            self.regex,
            self.ebnf,
        ]  # 互斥，最多只能设置一种
        if sum(x is not None for x in grammars) > 1:
            raise ValueError("Only one of regex, json_schema, or ebnf can be set.")

    def normalize(self, tokenizer):
        # 归一化处理停止条件，并预计算停止匹配所需的最大缓冲长度，
        # 便于在生成过程中高效地检测停止字符串/停止正则是否命中。
        # 处理停止字符串
        if self.stop_strs is None:
            # 未设置停止字符串：统一为空列表，最大长度为 0。
            self.stop_strs = []
            self.stop_str_max_len = 0
        else:
            # 单个字符串统一包装为列表，便于后续遍历处理。
            if isinstance(self.stop_strs, str):
                self.stop_strs = [self.stop_strs]

            # 计算所有停止字符串中最长者对应的长度：
            # 有 tokenizer 时按编码后的 token 数计算，否则退化为按字符数计算。
            stop_str_max_len = 0
            for stop_str in self.stop_strs:
                if tokenizer is not None:
                    stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                    stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))
                else:
                    stop_str_max_len = max(stop_str_max_len, len(stop_str))
            self.stop_str_max_len = stop_str_max_len

        # 处理停止正则字符串
        if self.stop_regex_strs is None:
            # 未设置停止正则：统一为空列表，最大长度为 0。
            self.stop_regex_strs = []
            self.stop_regex_max_len = 0
        else:
            # 单个正则统一包装为列表。
            if isinstance(self.stop_regex_strs, str):
                self.stop_regex_strs = [self.stop_regex_strs]

            # 计算匹配所有停止正则所需缓冲的最大 token 长度上界。
            stop_regex_max_len = 0
            for stop_regex in self.stop_regex_strs:
                stop_regex_max_len = max(
                    stop_regex_max_len, get_max_seq_length(stop_regex)
                )

            self.stop_regex_max_len = stop_regex_max_len


# 该函数计算为了匹配输入正则字符串，最多需要缓冲的 token 数量的严格上界。
# 注意：在最坏情况下，一个需要缓冲的字符对应一个 token。
def get_max_seq_length(regex_str: str):
    # 先将正则解析为 sre_parse 的语法树（SubPattern），再递归计算其最大匹配长度。
    return _max_length_from_subpattern(sre_parse.parse(regex_str))


# 无界重复（如 *、+、{n,}）时使用的“足够大”的长度上界值。
MAX_LEN = 2**30


def _max_length_from_subpattern(subpattern: sre_parse.SubPattern):
    # 递归遍历正则语法树的每个节点，累加其可能匹配的最大字符（token）数量。
    total = 0
    for token, value in subpattern:
        if token in {
            sre_parse.LITERAL,  # `value` 为某一个具体字符
            sre_parse.IN,  # 匹配 `value` 字符集合中的任一字符
            sre_parse.ANY,  # "."，匹配任意单个字符
        }:
            # 上述三类节点均最多匹配 1 个字符。
            total += 1
        elif token == sre_parse.SUBPATTERN:
            # 子模式（分组），例如 (a\d+) 解析为：
            # [(SUBPATTERN,
            #   (1, 0, 0, [(LITERAL, 97),
            #              (MAX_REPEAT, (1, MAXREPEAT, [(IN, [(CATEGORY, CATEGORY_DIGIT)])]))]))]
            # 取出内部子模式并递归计算其最大长度。
            _, _, _, inner_subpattern = value
            total += _max_length_from_subpattern(inner_subpattern)
        elif token == sre_parse.BRANCH:
            # 分支（如 a|bc）：取各分支中最大长度者作为上界。
            _, branches = value
            total += max(_max_length_from_subpattern(branch) for branch in branches)
        elif token in {sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT}:
            # 重复（贪婪/非贪婪）：value 为 (最小次数, 最大次数, 内部子模式)。
            _, max_num_repeat, inner_subpattern = value
            if max_num_repeat == sre_parse.MAXREPEAT:
                # 无界重复（如 *、+、{n,}）：长度无上界，使用 MAX_LEN 兜底。
                total += MAX_LEN
            else:
                # 有界重复：最大重复次数 * 内部子模式的最大长度。
                total += max_num_repeat * _max_length_from_subpattern(inner_subpattern)
        elif token == sre_parse.AT:
            # 零宽断言（如 ^、$、\b）不消耗字符，不增加最大长度。
            total += 0
        else:
            # 未处理的正则节点类型：保守起见按最大长度兜底，并打印告警。
            logger.warning(f"Got unhandled regex token: {token}")

            total += MAX_LEN

    return total
