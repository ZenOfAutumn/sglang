# TODO: also move pad_input_ids into this module
# 中译：本模块是「多模态处理器（multimodal processor）」的注册中心与工厂。
#       - 在导入期扫描某个包下的所有模块，把每个处理器类声明支持的模型架构注册进
#         PROCESSOR_MAPPING（架构类 -> 处理器类）。
#       - 运行期再根据模型的 HuggingFace 配置（hf_config.architectures）查表，
#         实例化出对应的多模态处理器，用于把图像/音频/视频等多模态输入预处理成模型可用的张量。
#       TODO（保留原注释）：未来也把 pad_input_ids 逻辑搬进本模块。
import importlib
import inspect
import logging
import pkgutil

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# 中译：全局注册表——键为「模型架构类」，值为「对应的多模态处理器类」。
#       由 import_processors() 在导入期填充，get_mm_processor() 在运行期按架构名查表。
PROCESSOR_MAPPING = {}


def import_processors(package_name: str, overwrite: bool = False):
    """中译：扫描并注册指定包下的所有多模态处理器。

    作用：
        遍历 package_name 包中的每个子模块，找出其中继承自 BaseMultimodalProcessor 的
        处理器类，并按其 `models` 属性声明支持的每个模型架构，登记到 PROCESSOR_MAPPING。

    关键参数：
        package_name: 待扫描的包名（如多模态处理器所在的包），用 importlib 动态导入。
        overwrite: 为 True 时，若已有「同名架构」的旧注册项，则先删除旧项再写入新项，
                   用于覆盖/替换已注册的处理器。

    副作用：
        原地修改全局字典 PROCESSOR_MAPPING；对导入失败的子模块仅打印 warning 并跳过。
    """
    package = importlib.import_module(package_name)
    # 中译：遍历包内的直接子模块；跳过子包（ispkg 为 True 者），只处理具体模块。
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                # 中译：单个模块导入失败不应中断整体注册流程，记录 warning 后跳过。
                logger.warning(f"Ignore import error when loading {name}: {e}")
                continue
            all_members = inspect.getmembers(module, inspect.isclass)
            # 中译：只保留「定义在该模块内」的类（排除从别处 import 进来的类），避免重复/误注册。
            classes = [
                member
                for name, member in all_members
                if member.__module__ == module.__name__
            ]
            # 中译：在这些类中筛出多模态处理器子类（BaseMultimodalProcessor 的子类）逐个注册。
            for cls in (
                cls for cls in classes if issubclass(cls, BaseMultimodalProcessor)
            ):
                # 中译：每个处理器类必须声明 `models`，列出它支持的模型架构类。
                assert hasattr(cls, "models")
                for arch in getattr(cls, "models"):
                    # 中译：overwrite 模式下，先按「架构类名」匹配并删除已有的旧注册项，再写入新项。
                    if overwrite:
                        for model_cls, processor_cls in PROCESSOR_MAPPING.items():
                            if model_cls.__name__ == arch.__name__:
                                del PROCESSOR_MAPPING[model_cls]
                                break
                    # 中译：登记「架构类 -> 处理器类」映射。
                    PROCESSOR_MAPPING[arch] = cls


def get_mm_processor(
    hf_config,
    server_args: ServerArgs,
    processor,
    transport_mode,
    model_config=None,
    **kwargs,
) -> BaseMultimodalProcessor:
    """中译：多模态处理器工厂——根据模型配置选出并实例化合适的处理器。

    作用：
        依据 hf_config.architectures（模型架构名）在 PROCESSOR_MAPPING 中查表，
        并结合是否走 Transformers 后端，返回一个具体的 BaseMultimodalProcessor 实例。

    关键参数：
        hf_config: HuggingFace 模型配置，其中 architectures 列出模型架构名，用于查表。
        server_args: 服务启动参数；其 model_impl 决定使用的实现后端（auto/transformers 等）。
        processor: 底层的 HF 处理器/分词器对象，传入具体处理器构造。
        transport_mode: 多模态数据的传输方式（如进程间如何传递张量）。
        model_config: 可选；当 model_impl 为 "auto" 时用它解析最终实现类型。
        **kwargs: 透传给处理器构造函数的其他参数。

    返回值：
        匹配到的多模态处理器实例。

    异常：
        若没有任何已注册架构匹配且未启用 Transformers 后端，抛出 ValueError。
    """
    # 中译：解析使用的模型实现后端；显式为 "transformers" 时直接走 Transformers 后端。
    model_impl = str(getattr(server_args, "model_impl", "auto")).lower()
    uses_transformers_backend = model_impl == "transformers"
    # 中译："auto" 时进一步解析模型最终实现类型，判断是否落到 Transformers 后端。
    if model_impl == "auto" and model_config is not None:
        from sglang.srt.model_loader.utils import get_resolved_model_impl

        uses_transformers_backend = (
            get_resolved_model_impl(model_config) == ModelImpl.TRANSFORMERS
        )

    # 中译：遍历注册表，找出架构名匹配的处理器类。
    for model_cls, processor_cls in PROCESSOR_MAPPING.items():
        if model_cls.__name__ not in hf_config.architectures:
            continue
        # 中译：若未走 Transformers 后端，或该处理器显式支持 Transformers 后端，则使用它。
        if not uses_transformers_backend or getattr(
            processor_cls, "supports_transformers_backend", False
        ):
            return processor_cls(
                hf_config, server_args, processor, transport_mode, **kwargs
            )

    # 中译：未命中专用处理器但走 Transformers 后端时，回退到通用的 Transformers 自动处理器。
    if uses_transformers_backend:
        from sglang.srt.multimodal.processors.transformers_auto import (
            TransformersAutoMultimodalProcessor,
        )

        return TransformersAutoMultimodalProcessor(
            hf_config, server_args, processor, transport_mode, **kwargs
        )

    # 中译：既无专用处理器也不走 Transformers 后端——无法处理该架构，报错并列出已注册架构。
    raise ValueError(
        f"No processor registered for architecture: {hf_config.architectures}.\n"
        f"Registered architectures: {[model_cls.__name__ for model_cls in PROCESSOR_MAPPING.keys()]}"
    )
