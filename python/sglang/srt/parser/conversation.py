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
"""Conversation chat templates.

This module provides conversation template definitions, data structures, and utilities
for managing chat templates across different model types in SGLang.

Key components:
- Conversation class: Defines the structure and behavior of chat templates
- SeparatorStyle enum: Different conversation formatting styles
- Template registry: Functions to register and retrieve templates by name or model path
- Built-in templates: Pre-defined templates for popular models
"""

# Adapted from
# https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
import dataclasses
import json
import os
import re
from enum import IntEnum, auto
from typing import Callable, Dict, List, Optional, Tuple, Union

from typing_extensions import Literal

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.utils import ImageData, read_system_prompt_from_file


class SeparatorStyle(IntEnum):
    """分隔符风格枚举。

    每种风格对应一种拼接 system / 角色 / 消息 / 分隔符的方式，
    在 Conversation.get_prompt 中根据该枚举选择不同的拼接逻辑。
    不同模型（LLaMA2/3/4、ChatGLM、ChatML、Qwen 等）有各自的提示词格式。
    """

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    LLAMA3 = auto()
    LLAMA4 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    DEEPSEEK_CHAT = auto()
    METAMATH = auto()
    DeepSeekVL2 = auto()
    QWEN2_VL_EMBED = auto()
    QWEN2_AUDIO = auto()
    GEMMA3 = auto()
    MPT = auto()
    PADDLE_OCR = auto()


@dataclasses.dataclass
class Conversation:
    """管理提示词模板并保存整个对话历史的类。

    一个 Conversation 实例描述某个模型的聊天模板（角色名、分隔符、多模态占位符等），
    并能把累积的消息通过 get_prompt() 拼接成最终输入模型的提示词。
    """

    # 模板名称。
    name: str
    # system 提示词的模板（{system_message} 会被填充）。
    system_template: str = "{system_message}"
    # 实际的 system 消息内容。
    system_message: str = ""
    # 两个角色名（通常为 用户 与 助手）。
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # 所有消息，每项为 [role, message]。
    messages: List[List[str]] = ()
    # few-shot 示例数量（转换为 chatbot 格式时跳过前 offset 条）。
    offset: int = 0
    # 分隔符风格及其配置。
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # 停止条件（默认为 EOS token）。
    stop_str: Union[str, List[str]] = None
    # 提示词中表示图像/视频/音频的占位 token。
    image_token: str = "<image>"
    video_token: str = "<video>"
    audio_token: str = "<audio>"

    image_data: Optional[List[ImageData]] = None
    video_data: Optional[List[str]] = None
    modalities: Optional[List[str]] = None
    stop_token_ids: Optional[int] = None

    audio_data: Optional[List[str]] = None
    image_token_at_prefix: bool = False

    def get_prompt(self) -> str:
        """根据分隔符风格，把 system 与所有消息拼接成用于生成的最终提示词。

        下面一大串 if/elif 分支对应 SeparatorStyle 的每种风格，逻辑雷同但分隔符/
        角色标记细节不同；未填写（message 为空）的最后一条通常是助手占位，
        用于提示模型从此处开始生成。
        """
        # 先把 system 消息填入 system 模板。
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.QWEN2_VL_EMBED:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            ret += self.stop_str
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA4:
            # begin_of_text is added by default
            if self.system_message:
                ret = system_prompt
            else:
                ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += f"<|header_start|>{role}<|header_end|>\n\n"
                    ret += f"{message.strip()}<|eot|>"
                else:
                    ret += f"<|header_start|>{role}<|header_end|>\n\n"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA3:
            if self.system_message:
                ret = system_prompt
            else:
                ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    ret += f"{message.strip()}<|eot_id|>"
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == "chatglm2" else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i // 2 + round_add_n}]{self.sep}"

                if message:
                    ret += f"{role}：{message}{self.sep}"
                else:
                    ret += f"{role}："
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if system_prompt == "" else system_prompt + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ""
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += "<s>"
                if message:
                    ret += role + ":" + message + seps[i % 2] + "\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ""
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.METAMATH:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for i, (role, message) in enumerate(self.messages):
                # For MetaMath, sep2 is used to prefix the message.
                starting_sep = ":\n" if i % 2 == 0 else ": " + self.sep2
                ending_sep = self.sep if i % 2 == 0 else ""
                if message:
                    ret += role + starting_sep + message + ending_sep
                else:
                    ret += role + starting_sep
            return ret
        elif self.sep_style == SeparatorStyle.DEEPSEEK_CHAT:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DeepSeekVL2:
            seps = [self.sep, self.sep2]
            if system_prompt == "" or system_prompt is None:
                ret = ""
            else:
                ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.GEMMA3:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i == 0:
                        ret += message + self.sep
                    else:
                        ret += role + message + self.sep
                else:
                    ret += role
            return ret

        elif self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.QWEN2_AUDIO:
            ret = "" if system_prompt == "" else system_prompt + self.sep

            counter = 1
            for role, message in self.messages:
                if message:
                    while self.audio_token in message:
                        message = message.replace(
                            self.audio_token, self.audio_token.format(idx=counter), 1
                        )
                        counter += 1

                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"

            return ret
        elif self.sep_style == SeparatorStyle.PADDLE_OCR:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ": "
                    if role == self.roles[0]:
                        if self.image_token in message:
                            ret += message.replace(
                                self.image_token + "\n", self.image_token
                            )
                        else:
                            ret += message
                        ret += "\n"
                    else:
                        ret += message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        else:
            # 未知风格，报错。
            raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        """设置 system 消息。"""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """追加一条消息（message 为 None 表示待生成的占位）。"""
        self.messages.append([role, message])

    def append_image(self, image: str, detail: Literal["auto", "low", "high"]):
        """追加一张图像数据。"""
        self.image_data.append(ImageData(url=image, detail=detail))

    def append_video(self, video: str):
        """追加一段视频数据。"""
        self.video_data.append(video)

    def append_audio(self, audio: str):
        """追加一段音频数据。"""
        self.audio_data.append(audio)

    def update_last_message(self, message: str):
        """原地更新最后一条消息。

        构造提示词时最后一条通常为 None（占位），拿到模型响应后需原地填回。
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """将对话转换为 gradio chatbot 格式（[[用户, 助手], ...]）。"""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """将对话转换为 OpenAI chat completion 的 messages 格式。"""
        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            image_token=self.image_token,
            video_token=self.video_token,
            audio_token=self.audio_token,
            image_token_at_prefix=self.image_token_at_prefix,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# 全局对话模板注册表：模板名 → Conversation。
chat_templates: Dict[str, Conversation] = {}
# 模型路径 → 模板名 的匹配函数列表（依次尝试）。
matching_function_registry: List[Callable] = []


def register_conv_template(template: Conversation, override: bool = False):
    """注册一个对话模板；默认不允许重复注册（override=True 可覆盖）。"""
    if not override:
        assert (
            template.name not in chat_templates
        ), f"{template.name} has been registered."

    chat_templates[template.name] = template


def register_conv_template_matching_function(func):
    """注册一个“模型路径 → 模板名”的匹配函数（可作装饰器使用）。"""
    matching_function_registry.append(func)


def get_conv_template_by_model_path(model_path):
    """依次尝试所有匹配函数，返回第一个命中的模板名；都未命中返回 None。"""
    for matching_func in matching_function_registry:
        conv_name = matching_func(model_path)
        if conv_name is not None:
            return conv_name
    return None


def chat_template_exists(template_name: str) -> bool:
    """判断指定名称的对话模板是否已注册。"""
    return template_name in chat_templates


def generate_embedding_convs(
    texts: List[str], images: List[str], videos: List[str], template_name: str
) -> List[Conversation]:
    """为 embedding 场景批量构造对话：每组 (text, image, video) 生成一个只有用户输入的对话。"""
    conv_template = chat_templates[template_name].copy()
    convs = []
    for text, image, video in zip(texts, images, videos):
        conv = Conversation(
            name=conv_template.name,
            system_template=conv_template.system_template,
            system_message=conv_template.system_message,
            roles=conv_template.roles,
            messages=list(conv_template.messages),  # prevent in-place modification
            offset=conv_template.offset,
            sep_style=SeparatorStyle(conv_template.sep_style),
            sep=conv_template.sep,
            sep2=conv_template.sep2,
            stop_str=conv_template.stop_str,
            image_data=[],
            video_data=[],
            audio_data=[],
            modalities=[],
            image_token=conv_template.image_token,
            video_token=conv_template.video_token,
            audio_token=conv_template.audio_token,
            image_token_at_prefix=conv_template.image_token_at_prefix,
        )
        real_content = ""

        if image is not None:
            image_token = (
                conv.image_token + "\n"
                if conv.name != "gme-qwen2-vl"
                else conv.image_token
            )
            real_content += image_token
        if video is not None:
            real_content += conv.video_token
        if text is not None:
            real_content += text
        conv.append_message(conv.roles[0], real_content)
        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        convs.append(conv)

    return convs


# Models in which system adds modality tokens at prompt start automatically
# when media inputs exceed modality tokens in prompt (e.g. 3 images but 2 <image> tokens)
_MODELS_REQUIRING_MODALITY_SUPPLEMENT = {"deepseek-vl2"}


# adapted from https://github.com/vllm-project/vllm/blob/5124f5bf51b83e6f344c1bc6652e8c4d81313b34/vllm/entrypoints/chat_utils.py#L856
def _get_full_multimodal_text_prompt(
    modality_token: str, modality_count: int, text_prompt: str
) -> str:
    """为多模态模型补全缺失的模态占位 token。

    当实际多模态输入数（modality_count）多于文本中已有的占位 token 数时，
    在提示词开头补上缺少的 token；若占位多于实际数据则报错。
    """
    # 文本中已有的占位保留不动，只计算还缺多少个。
    left: int = modality_count - text_prompt.count(modality_token)
    if left < 0:
        raise ValueError(
            f"Found more '{modality_token}' placeholders in input prompt than "
            "actual multimodal data items."
        )

    # NOTE: For now we always add missing modality_token at the front of
    # the prompt. This may change to be customizable in the future.
    return "\n".join([modality_token] * left + [text_prompt])


def generate_chat_conv(
    request: ChatCompletionRequest, template_name: str
) -> Conversation:
    """根据 ChatCompletion 请求与指定模板名，构造并填充一个 Conversation。

    处理 system / user / assistant 三种角色的消息，并在 user 消息中插入
    图像/视频/音频占位 token，最后追加一条空助手消息作为生成占位。
    """
    conv = chat_templates[template_name].copy()
    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message=conv.system_message,
        roles=conv.roles,
        messages=list(conv.messages),  # prevent in-place modification
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        image_data=[],
        video_data=[],
        audio_data=[],
        modalities=[],
        image_token=conv.image_token,
        audio_token=conv.audio_token,
        video_token=conv.video_token,
        image_token_at_prefix=conv.image_token_at_prefix,
    )

    if isinstance(request.messages, str):
        raise ValueError("The messages should be a list of dict.")
    for message in request.messages:
        msg_role = message.role
        if msg_role == "system":
            if isinstance(message.content, str):
                conv.system_message = message.content
            elif isinstance(message.content, list):
                if (
                    len(message.content) != 1
                    or getattr(message.content[0], "type", None) != "text"
                ):
                    raise ValueError("The system message should be a single text.")
                else:
                    conv.system_message = getattr(message.content[0], "text", "")
        elif msg_role == "user":
            # Handle the various types of Chat Request content types here.
            if isinstance(message.content, str):
                conv.append_message(conv.roles[0], message.content)
            else:
                real_content = ""
                # calculate number of image_url
                num_image_url = 0
                for content in message.content:
                    if content.type == "image_url":
                        num_image_url += 1
                        conv.modalities.append(content.modalities)
                image_token = (
                    conv.image_token + "\n"
                    if conv.name != "qwen2-vl"
                    else conv.image_token
                )
                add_token_as_needed: bool = (
                    conv.name in _MODELS_REQUIRING_MODALITY_SUPPLEMENT
                )
                if add_token_as_needed:
                    image_token = ""

                audio_token = conv.audio_token
                video_token = conv.video_token
                for content in message.content:
                    if content.type == "text":
                        if num_image_url > 16:
                            real_content += "\n"  # for video
                        real_content += content.text
                    elif content.type == "image_url":
                        # NOTE: works for llava and intervl2_5
                        if conv.image_token_at_prefix:
                            real_content = image_token + real_content
                        else:
                            real_content += image_token
                        conv.append_image(
                            content.image_url.url, content.image_url.detail
                        )
                    elif content.type == "video_url":
                        real_content += video_token
                        conv.append_video(content.video_url.url)
                    elif content.type == "audio_url":
                        real_content += audio_token
                        conv.append_audio(content.audio_url.url)
                if add_token_as_needed:
                    real_content = _get_full_multimodal_text_prompt(
                        conv.image_token, num_image_url, real_content
                    )
                conv.append_message(conv.roles[0], real_content)
        elif msg_role == "assistant":
            parsed_content = ""
            if isinstance(message.content, str):
                parsed_content = message.content
            elif isinstance(message.content, list):
                if (
                    len(message.content) != 1
                    or getattr(message.content[0], "type", None) != "text"
                ):
                    raise ValueError(
                        "The assistant's response should be a single text."
                    )
                else:
                    parsed_content = getattr(message.content[0], "text", "")
            conv.append_message(conv.roles[1], parsed_content)
        else:
            raise ValueError(f"Unknown role: {msg_role}")

    # 追加一条空的助手消息，作为模型生成的起点占位。
    conv.append_message(conv.roles[1], None)
    return conv


# =============================================================================
# 以下为各流行模型预注册的内置对话模板。
# 每个 register_conv_template(...) 描述一个模型的角色名、system 模板、
# 分隔符风格、停止词与多模态占位 token 等。结构雷同，不逐个注释。
# =============================================================================

# llama2 template
# reference: https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
# reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
register_conv_template(
    Conversation(
        name="llama-2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
        stop_str=["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"],
    )
)

# reference: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/chat_template.json
register_conv_template(
    Conversation(
        name="mistral",
        system_template="[SYSTEM_PROMPT]\n{system_message}\n[/SYSTEM_PROMPT]\n\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
        stop_str=["[INST]", "[/INST]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]"],
        image_token="[IMG]",
    )
)

register_conv_template(
    Conversation(
        name="devstral",
        system_template="[SYSTEM_PROMPT]\n{system_message}\n[/SYSTEM_PROMPT]\n\n",
        system_message=read_system_prompt_from_file("mistralai/Devstral-Small-2505"),
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
        stop_str=["[INST]", "[/INST]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]"],
        image_token="[IMG]",
    )
)

# reference: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/chat_template.json
register_conv_template(
    Conversation(
        name="llama-4",
        system_template="<|header_start|>system<|header_end|>\n\n{system_message}<|eot|>",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.LLAMA4,
        sep="",
        stop_str=["<|end_of_text|>", "<|eot|>", "<|eom|>"],
        image_token="<|image|>",
    )
)

# TODO (lifuhuang): Refactor BaseMultimodalProcessor to support the default image token "<|image_{index}|>" in the future.
register_conv_template(
    Conversation(
        name="phi-4-mm",
        system_message="",
        system_template="{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="<|end|>",
        stop_str="<|end|>",
        image_token="<|endoftext10|>",
        audio_token="<|endoftext11|>",
    )
)

register_conv_template(
    Conversation(
        name="chatml",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_str=["<|endoftext|>", "<|im_end|>"],
    )
)

register_conv_template(
    Conversation(
        name="chatml-llava",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_str=["<|endoftext|>", "<|im_end|>"],
    )
)

register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

register_conv_template(
    Conversation(
        name="llama_3_vision",
        system_message="You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.LLAMA3,
        sep="",
        stop_str=["<|end_of_text|>", "<|eot_id|>"],
        image_token="<|image|>",
    )
)

register_conv_template(
    Conversation(
        name="llava_llama_3",
        system_message="You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.LLAMA3,
        sep="",
        stop_str=["<|end_of_text|>", "<|eot_id|>"],
    )
)
# Reference: https://github.com/InternLM/lmdeploy/blob/387bf54b4f124e72aab30ae9755f562e435d3d01/lmdeploy/model.py#L425-L442
register_conv_template(
    Conversation(
        name="internlm2-chat",
        system_template="<|im_start|>system\n{system_message}",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep="\n",
        stop_str=["<|im_end|>", "<|action_end|>"],
    )
)

register_conv_template(
    Conversation(
        name="internvl-2-5",
        system_template="<|im_start|>system\n{system_message}",
        system_message="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
        stop_str=["<|im_end|>", "<|action_end|>"],
        image_token="<IMG_CONTEXT>",
        image_token_at_prefix=True,
    )
)

# Reference: https://huggingface.co/docs/transformers/main/model_doc/qwen2_vl#usage-example
register_conv_template(
    Conversation(
        name="qwen2-vl",
        system_message="You are a helpful assistant.",
        system_template="<|im_start|>system\n{system_message}",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep="<|im_end|>\n",
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        stop_str=["<|im_end|>"],
        image_token="<|vision_start|><|image_pad|><|vision_end|>",
        video_token="<|vision_start|><|video_pad|><|vision_end|>",
    )
)

register_conv_template(
    Conversation(
        name="deepseek-ocr",
        system_message="",
        system_template="",
        roles=("", ""),
        sep="",
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        stop_str=["<｜end▁of▁sentence｜>"],
        image_token="<image>",
        image_token_at_prefix=True,
    )
)

register_conv_template(
    Conversation(
        name="paddle-ocr",
        system_message="",
        system_template="<|begin_of_sentence|>{system_message}",
        roles=("User", "Assistant"),
        sep="<|end_of_sentence|>",
        sep_style=SeparatorStyle.PADDLE_OCR,
        stop_str=["<|end_of_sentence|>"],
        image_token="<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>",
    )
)

register_conv_template(
    Conversation(
        name="deepseek-vl2",
        system_template="{system_message}",
        # system_message="You are a helpful assistant. Please answer truthfully and write out your "
        # "thinking step by step to be sure you get the right answer.",
        system_message="",
        roles=("<|User|>", "<|Assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.DeepSeekVL2,
        sep="\n\n",
        sep2="<｜end▁of▁sentence｜>",
        stop_str=["User:", "<｜end▁of▁sentence｜>"],
    )
)

# Reference: https://huggingface.co/google/gemma-3-4b-it/blob/main/config.json
register_conv_template(
    Conversation(
        name="gemma-it",
        system_message="You are a helpful assistant.",
        system_template="<start_of_turn>user\n{system_message}\n\n",
        roles=("<start_of_turn>user\n", "<start_of_turn>model\n"),
        sep="<end_of_turn>\n",
        sep_style=SeparatorStyle.GEMMA3,
        stop_str=["<end_of_turn>"],
        image_token="<start_of_image>",
        audio_token="<start_of_audio>",
    )
)

# Reference: https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct#usage
register_conv_template(
    Conversation(
        name="gme-qwen2-vl",
        system_message="You are a helpful assistant.",
        system_template="<|im_start|>system\n{system_message}",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep="<|im_end|>\n",
        sep_style=SeparatorStyle.QWEN2_VL_EMBED,
        stop_str="<|endoftext|>",
        image_token="<|vision_start|><|image_pad|><|vision_end|>",
    )
)

# Reference: https://huggingface.co/openbmb/MiniCPM-V-2_6#usage
register_conv_template(
    Conversation(
        name="minicpmv",
        system_message="You are a helpful assistant",
        system_template="<|im_start|>system\n{system_message}.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep="<|im_end|>\n",
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        stop_str=("<|im_end|>", "<|endoftext|>"),
        image_token="(<image>./</image>)",
        video_token="(<video>./</video>)",
    )
)

# Reference: https://github.com/deepseek-ai/Janus?tab=readme-ov-file#janus-pro
register_conv_template(
    Conversation(
        name="janus-pro",
        system_message="You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language",
        system_template="{system_message}.",
        roles=("User", "Assistant"),
        sep="\n\n",
        sep2="<｜end▁of▁sentence｜>",
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        stop_str=["<|User|>", "<｜end▁of▁sentence｜>"],
        image_token="<image_placeholder>",
    )
)

# Reference: https://huggingface.co/openbmb/MiniCPM-o-2_6#usage
register_conv_template(
    Conversation(
        name="minicpmo",
        system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        system_template="<|im_start|>system\n{system_message}",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep="<|im_end|>\n",
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        stop_str=("<|im_end|>", "<|endoftext|>"),
        image_token="(<image>./</image>)",
        audio_token="(<audio>./</audio>)",
    )
)

# Reference: https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/chat_template.jinja
register_conv_template(
    Conversation(
        name="kimi-vl",
        system_message="You are a helpful assistant",
        system_template="<|im_system|>system<|im_middle|>{system_message}",
        roles=(
            "<|im_user|>user<|im_middle|>",
            "<|im_assistant|>assistant<|im_middle|>",
        ),
        messages=[],
        sep="<|im_end|>",
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        stop_str="<|im_end|>",
        image_token="<|media_start|>image<|media_content|><|media_pad|><|media_end|>",
    )
)

register_conv_template(
    Conversation(
        name="qwen2-audio",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep="<|im_end|>\n",
        sep_style=SeparatorStyle.QWEN2_AUDIO,
        stop_str=["<|im_end|>"],
        audio_token="Audio {idx}: <|audio_bos|><|AUDIO|><|audio_eos|>\n",
    )
)

register_conv_template(
    Conversation(
        name="points-v15-chat",
        system_message="",
        system_template="",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep="<|im_end|>\n",
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        stop_str=["<|im_end|>"],
        image_token="<|vision_start|><|image_pad|><|vision_end|>",
        video_token="<|vision_start|><|video_pad|><|vision_end|>",
    )
)

# Whisper speech-to-text template
# Whisper uses special tokens: <|startoftranscript|>, <|en|>, <|transcribe|>, etc.
# Audio features are processed by encoder separately, not inserted into text
# The decoder start tokens (task, language) should be set via generation config
register_conv_template(
    Conversation(
        name="whisper",
        system_template="",
        system_message="",
        roles=("", ""),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_str=["<|endoftext|>"],
        audio_token="",  # Empty - audio is handled by encoder, not as text token
    )
)

# config.json 中的 model_type → 对话模板名的映射（供匹配函数兼底使用）。
MODEL_TYPE_TO_TEMPLATE = {
    "internvl_chat": "internvl-2-5",
    "deepseek_vl_v2": "deepseek-vl2",
    "multi_modality": "janus-pro",
    "phi4mm": "phi-4-mm",
    "minicpmv": "minicpmv",
    "minicpmo": "minicpmo",
    "deepseek-ocr": "deepseek-ocr",
    "paddleocr_vl": "paddle-ocr",
    "whisper": "whisper",
}


# =============================================================================
# 以下为模型路径匹配函数：用 @register_conv_template_matching_function 注册，
# get_conv_template_by_model_path 会依次调用它们。每个函数先按模型名关键词
# （正则）匹配，命中则返回模板名；否则兼底用 config.json 的 model_type 查映射表。
# =============================================================================


@register_conv_template_matching_function
def match_points_v15_chat(model_path: str):
    # reference: https://github.com/sgl-project/sglang/issues/12791
    if re.search(r"\bpoints\b", model_path, re.IGNORECASE):
        return "points-v15-chat"


def get_model_type(model_path: str) -> Optional[str]:
    """读取模型目录下的 config.json 并返回其 model_type；读不到或解析失败返回 None。"""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("model_type")
    except (IOError, json.JSONDecodeError):
        return None


@register_conv_template_matching_function
def match_internvl(model_path: str):
    if re.search(r"internvl", model_path, re.IGNORECASE):
        return "internvl-2-5"
    model_type = get_model_type(model_path)
    return MODEL_TYPE_TO_TEMPLATE.get(model_type)


@register_conv_template_matching_function
def match_deepseek_janus_pro(model_path: str):
    if re.search(r"janus", model_path, re.IGNORECASE):
        return "janus-pro"
    model_type = get_model_type(model_path)
    return MODEL_TYPE_TO_TEMPLATE.get(model_type)


@register_conv_template_matching_function
def match_vicuna(model_path: str):
    if re.search(r"vicuna|llava-v1\.5|llava-next-video-7b", model_path, re.IGNORECASE):
        return "vicuna_v1.1"


@register_conv_template_matching_function
def match_deepseek_vl(model_path: str):
    if re.search(r"deepseek.*vl2", model_path, re.IGNORECASE):
        return "deepseek-vl2"
    model_type = get_model_type(model_path)
    return MODEL_TYPE_TO_TEMPLATE.get(model_type)


@register_conv_template_matching_function
def match_qwen_chat_ml(model_path: str):
    if re.search(
        r"llava-v1\.6-34b|llava-v1\.6-yi-34b|llava-next-video-34b|llava-onevision-qwen2",
        model_path,
        re.IGNORECASE,
    ):
        return "chatml-llava"


@register_conv_template_matching_function
def match_minicpm(model_path: str):
    match = re.search(r"minicpm-(v|o)", model_path, re.IGNORECASE)
    if match:
        return f"minicpm{match.group(1).lower()}"
    model_type = get_model_type(model_path)
    return MODEL_TYPE_TO_TEMPLATE.get(model_type)


@register_conv_template_matching_function
def match_phi_4_mm(model_path: str):
    if "phi-4-multimodal" in model_path.lower():
        return "phi-4-mm"
    model_type = get_model_type(model_path)
    return MODEL_TYPE_TO_TEMPLATE.get(model_type)


@register_conv_template_matching_function
def match_deepseek_ocr(model_path: str):
    if "deepseek-ocr" in model_path.lower():
        return "deepseek-ocr"
    model_type = get_model_type(model_path)
    return MODEL_TYPE_TO_TEMPLATE.get(model_type)


@register_conv_template_matching_function
def match_paddle_ocr(model_path: str):
    if "paddleocr" in model_path.lower():
        return "paddle-ocr"
    model_type = get_model_type(model_path)
    return MODEL_TYPE_TO_TEMPLATE.get(model_type)


@register_conv_template_matching_function
def match_whisper(model_path: str):
    if "whisper" in model_path.lower():
        return "whisper"
    model_type = get_model_type(model_path)
    return MODEL_TYPE_TO_TEMPLATE.get(model_type)
