# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 改编自 https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/base_config.py
#
# 本文件定义了 SGLang 量化体系的核心抽象基类：
#   1. QuantizeMethodBase     —— 所有量化方法（linear / MoE / embedding 等）的基类
#   2. LinearMethodBase       —— 普通线性层（Linear）量化方法的基类
#   3. FusedMoEMethodBase     —— 融合 MoE 层量化方法的基类
#   4. QuantizationConfig     —— 量化方案配置（GPTQ / AWQ / FP8 / Marlin ...）的基类
#
# 整体调用流程：
#   启动时根据模型目录中的量化配置文件（如 quant_config.json）构造具体的
#   QuantizationConfig 子类实例 → 模型构建时通过 get_quant_method() 为每个
#   层拿到对应的 QuantizeMethodBase → create_weights() 创建量化权重 →
#   加载 checkpoint 后 process_weights_after_loading() 做后处理 →
#   推理时调用 apply() 执行量化矩阵乘。
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import torch
from sglang.srt.layers.modelopt_utils import canonicalize_modelopt_quant_algo
from torch import nn

# 仅在类型检查时导入，避免运行时产生循环依赖
if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.base import MoeQuantInfo
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackendLike
    from sglang.srt.models.utils import WeightsMapper


class QuantizeMethodBase(ABC):
    """不同量化方法的基类。

    一个 "量化方法" 描述了如何为一个层创建量化权重、以及如何用这些
    量化权重执行前向计算。所有具体的量化方法（如 GPTQ、AWQ、FP8、
    Marlin 等）都应继承此类，并按需实现 create_weights / apply /
    process_weights_after_loading。

    典型实现示例：
        GPTQMarlinLinearMethod、AwqLinearMethod、Fp8LinearMethod 等
    """

    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        """为层创建量化权重。

        权重将作为属性（attribute）直接挂载到 layer 上，例如
        layer.weight、layer.weight_scale 等。

        输入示例：
            layer       : 一个未量化/待量化的 nn.Module（如 Linear 层）
            weight_args : 各量化方案自定义的位置参数
                          （如 LinearMethodBase 中是 input_size_per_partition 等）
            extra_weight_attrs: 传递给 torch.nn.Parameter 的额外属性，
                          例如 {"weight_loader": ...} 用于分片加载

        输出示例：
            无显式返回值；副作用是在 layer 上挂载权重属性，例如：
                layer.weight       -> Parameter(shape=[output_size, input_size])
                layer.weight_scale -> 量化缩放系数 Parameter
        """
        raise NotImplementedError()

    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """将层中的量化权重作用于输入张量（即执行量化前向计算）。

        前提：必须先在该层上调用过 create_weights()。

        输入示例：
            layer : 已创建好量化权重的 nn.Module
            args  : 前向计算的输入，例如 LinearMethodBase 中是 (x, bias)

        输出示例：
            量化矩阵乘的输出张量，例如：
                x: [num_tokens, 512] (fp16)
                -> 输出 [num_tokens, 1024] (fp16)，
                   内部实际执行的是 INT4/FP8 量化后的 GEMM
        """
        raise NotImplementedError()

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """在权重加载完成后对权重做后处理。

        典型用途：转置权重以匹配 CUDA kernel 的内存布局、
        重打包（repack）量化权重、校验量化参数等。

        输入示例：
            layer : 已加载完 checkpoint 权重的 nn.Module

        输出示例：
            无返回值；副作用示例：将 layer.weight 原地转置，
            或将 INT4 打包权重 repack 为 Marlin kernel 需要的格式
        """
        return


class LinearMethodBase(QuantizeMethodBase):
    """普通线性层（可能是量化的）线性方法的基类。

    用于管理 Transformer 中所有 nn.Linear 类层的量化，
    例如 qkv_proj、o_proj、gate_up_proj、down_proj 等。

    典型实现示例：
        GPTQMarlinLinearMethod、AwqMarlinLinearMethod、Fp8LinearMethod
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """为线性层创建量化权重，权重将作为属性挂载到 layer 上。

        输入示例（以 8 卡张量并行中的 QKVLinear 为例）：
            layer: QKVParallelLinear 实例
            input_size_per_partition: 本 rank 上权重的输入维度，
                如 TP=8 时 hidden_size=4096 -> 每卡输入维度仍为 4096
            output_partition_sizes: 本 rank 上每个逻辑权重的输出维度列表，
                例如 QKV 层为 [Wq 的宽度, Wk 的宽度, Wv 的宽度]，
                如 [1024, 128, 128]（假设 32 头、8 组 KV、TP=8）
            input_size: 所有 rank 合计的输入维度，如 4096
            output_size: 所有 rank 合计的输出维度，如 4096 + 1024 + 1024 = 6144
            params_dtype: 参数的数据类型，如 torch.float16
            extra_weight_attrs: 额外属性，如 {"weight_loader": ...}

        输出示例：
            无显式返回值；副作用是在 layer 上挂载量化权重属性：
                layer.weight        -> 形状如 [1280, 4096]（打包后的量化权重）
                layer.weight_scale  -> 形状如 [1280, 1] 或按 block 分组的 scale
        """
        raise NotImplementedError()

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """将层中的量化权重作用于输入张量（执行量化 GEMM）。

        前提：必须先在该层上调用过 create_weights()。

        输入示例：
            layer: 已创建好量化权重的 nn.Linear
            x   : 输入张量，形状 [num_tokens, input_size]，如 [32, 4096]，fp16
            bias: 可选偏置，形状 [output_size]，如 [6144]

        输出示例：
            输出张量，形状 [num_tokens, output_size]，如 [32, 6144]，fp16；
            内部执行的是反量化后的矩阵乘：y = x @ W_q^T * scale + bias
        """
        raise NotImplementedError()


class FusedMoEMethodBase(QuantizeMethodBase):
    """融合 MoE（Mixture of Experts）层量化方法的基类。

    MoE 层与普通 Linear 不同：权重是 [num_experts, ...] 形状的多专家
    堆叠权重，前向计算需要经过 token 分发（dispatch）与结果合并
    （combine）两个阶段，因此 apply() 的输入输出是 dispatch/combine
    结构体而非裸张量。

    典型实现示例：
        Fp8MoEMethod、GPTQMarlinMoEMethod 等
    """

    # 每个 FusedMoE 层实例持有的 MoE 运行器（负责 dispatch -> kernel -> combine 流程）
    runner: MoeRunner | None = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """为 MoE 层创建量化权重。

        输入示例（以 Mixtral 风格的 MoE 层为例，EP/TP 并行下）：
            layer: FusedMoE 实例
            num_experts: 专家总数，如 8
            hidden_size: 隐藏层维度，如 4096
            intermediate_size_per_partition: 本 rank 上每个专家的
                FFN 中间层宽度，如 TP=2 时 14336 -> 7168
            params_dtype: 参数数据类型，如 torch.float16

        输出示例：
            无显式返回值；副作用是在 layer 上挂载多专家权重属性：
                layer.w13_weight -> 形状 [num_experts, 2*I, H] 的 gate+up 合并权重
                layer.w2_weight  -> 形状 [num_experts, H, I] 的 down 投影权重
                以及对应的 scale 张量
        """
        raise NotImplementedError

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        """为该 MoE 层创建 MoE 运行器（MoeRunner）。

        运行器封装了 "token 分发 -> 调用量化 MoE kernel -> 结果合并"
        的完整执行流程。

        输入示例：
            layer: FusedMoE 实例
            moe_runner_config: 运行器配置，例如包含
                inplace=True（原地写回）、activation="silu" 等

        输出示例：
            无显式返回值；副作用通常是设置 self.runner = MoeRunner(...)
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        """执行量化 MoE 前向计算。

        输入示例：
            layer: 已创建好量化权重的 FusedMoE 实例
            dispatch_output: token 分发结果，包含每个 token 被路由到
                哪些专家、对应的 topk 权重、以及按专家分组排列后的
                隐层激活张量等

        输出示例：
            CombineInput: 待合并的专家输出，供后续 combine 阶段
            （如 allreduce / token 还原排序）使用，最终还原为
            [num_tokens, hidden_size] 的输出
        """
        raise NotImplementedError

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        """返回描述该层上量化状态的 ``TritonMoeQuantInfo``。

        LoRA MoE 运行器会调用本方法，使 ``invoke_fused_moe_kernel``
        能拿到正确的标志位 / 缩放系数 / 分块形状（block-shape）来处理
        基础权重。每个量化方法都必须重写此方法，其构造方式应与
        ``apply()`` 内部已有的构造保持一致。

        输入示例：
            layer: FusedMoE 实例（fp8 量化）

        输出示例：
            TritonMoeQuantInfo，例如包含：
                w13_weight, w13_weight_scale, w2_weight, w2_weight_scale,
                block_shape=[128, 128] 等，供 triton kernel 使用
        """
        raise NotImplementedError(
            f"{type(self).__name__} 必须实现 get_triton_quant_info()"
        )

    def get_moe_quant_info(
        self, layer: torch.nn.Module, runner_backend: MoeRunnerBackendLike
    ) -> MoeQuantInfo:
        """根据运行后端返回对应的 MoE 量化信息。

        输入示例：
            layer          : FusedMoE 实例
            runner_backend : 运行后端枚举，如 "triton"

        输出示例：
            runner_backend 为 "triton" 时 -> 返回 TritonMoeQuantInfo
            其他后端（如 flashinfer）     -> 抛出 NotImplementedError
        """
        if runner_backend.is_triton():
            return self.get_triton_quant_info(layer)
        raise NotImplementedError(
            f"{type(self).__name__} 未提供后端 {runner_backend.value!r} 的量化信息"
        )


class QuantizationConfig(ABC):
    """量化方案配置的基类。

    每个 QuantizationConfig 子类对应一种量化方案（GPTQ、AWQ、FP8、
    Marlin、ModelOpt 等），负责：
      1. 声明自己的名称、支持的激活数据类型、最低 GPU 算力要求
      2. 从模型目录的量化配置文件中解析配置并构造自身
      3. 为每个层分配具体的量化方法（get_quant_method）

    典型实现示例：
        GPTQConfig、AWQConfig、Fp8Config、ModelOptFp8Config 等
    """

    # 权重量化的分组块大小，如 DeepSeek 的细粒度量化使用 [128, 128]，
    # None 表示使用默认（按列分组）的 scale 布局
    weight_block_size: Optional[List[int]] = None

    def __init__(self):
        super().__init__()
        # packed_modules_mapping 由各模型在初始化时填充：
        # 将 sglang 模型中 "合并后的模块名" 映射到 "checkpoint 中的原始子模块名列表"。
        # 例如 {"gate_up_proj": ["gate_proj", "up_proj"]}，
        # 供加载器在按名字匹配量化权重时使用。
        self.packed_modules_mapping: Dict[str, List[str]] = dict()

    def update_packed_modules_mapping(self, mapping: Dict[str, List[str]]) -> None:
        """更新合并模块的映射表。

        输入示例：
            mapping: {"gate_up_proj": ["gate_proj", "up_proj"]}

        输出示例：
            无返回值；副作用是 self.packed_modules_mapping 被整体替换
        """
        self.packed_modules_mapping = mapping

    @abstractmethod
    def get_name(self) -> str:
        """量化方法的名称。

        输入示例：无
        输出示例："gptq" / "awq" / "fp8" / "modelopt_fp8"
        """
        raise NotImplementedError()

    @abstractmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """支持的激活（计算）数据类型列表。

        输入示例：无
        输出示例：[torch.float16, torch.bfloat16]
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        """支持该量化方法所需的最低 GPU 算力（compute capability）。

        该限制源于量化方法所依赖的自定义 CUDA kernel。

        输入示例：无（类方法）
        输出示例：
            70 -> Volta, 75 -> Turing, 80 -> Ampere, 89 -> Ada
            例如 FP8 方法要求 89（仅 Hopper 之前不支持 FP8 Tensor Core）
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> List[str]:
        """需要在模型目录中搜索的量化配置文件名列表。

        输入示例：无
        输出示例：
            GPTQ  -> ["quantize_config.json", "quant_config.json", "config.json"]
            AWQ   -> ["quant_config.json", "quantize_config.json"]
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> QuantizationConfig:
        """从模型的量化配置字典创建配置类实例。

        输入示例：
            config: {"bits": 4, "group_size": 128, "sym": True, ...}
                    （来自 quantize_config.json）

        输出示例：
            返回对应的 QuantizationConfig 实例，
            如 GPTQConfig(bits=4, group_size=128, sym=True)
        """
        raise NotImplementedError()

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        """
        检测该量化方法是否可以通过"覆盖用户指定的量化方法"来支持给定的
        checkpoint 格式——只有极少数特殊情况下子类才应重写此方法。

        输入示例：
            hf_quant_cfg: checkpoint 的量化配置字典，如 {"quant_algo": "NVFP4"}
            user_quant  : 用户通过 --quantization 指定的方法名，如 "modelopt"

        输出示例：
            返回覆盖后的方法名字符串（如 "modelopt_fp4"），
            若无需覆盖则返回 None
        """
        return None

    @classmethod
    def _modelopt_override_quantization_method(
        cls, hf_quant_config, user_quant
    ) -> Optional[str]:
        """ModelOpt 量化方法覆盖的共享逻辑。

        输入示例：
            hf_quant_config: {"quant_algo": "NVFP4", "quant_method": "modelopt_fp4"}
            user_quant: "modelopt" 或用户显式指定的方法名

        输出示例：
            - checkpoint 为 ModelOpt 格式时返回具体方法名，
              如 "modelopt_fp4" / "modelopt_fp8"
            - 否则返回 None

        处理规则：
            1. 用户显式要求在线重量化（如 quark_mxfp4）时不覆盖
            2. 用户指定通用 "modelopt" 时，根据 checkpoint 的 quant_algo
               自动探测具体方法
            3. 检查 quant_method 字段是否为 modelopt_fp8 / modelopt_fp4
        """
        if hf_quant_config is None:
            return None

        # 如果用户显式请求了在线重量化（例如在 NVFP4 checkpoint 上做
        # quark_mxfp4 在线重量化），不要把它覆盖回源格式。
        from sglang.srt.configs.model_config import REQUANTIZATION_METHODS

        if user_quant == "nvfp4_online" or user_quant in REQUANTIZATION_METHODS:
            return None

        # 检查是否为 ModelOpt 配置
        quant_algo = hf_quant_config.get("quant_algo", "").upper()

        # 如果用户指定了通用的 "modelopt"，则自动探测具体方法
        if user_quant == "modelopt":
            canonical_method = canonicalize_modelopt_quant_algo(quant_algo)
            if canonical_method is not None:
                return canonical_method

        # hf_quant_config 可能是已解析过的量化配置，
        # 因此还需要检查 quant_method 字段。
        if hf_quant_config.get("quant_method", "") == "modelopt_fp8":
            return "modelopt_fp8"
        elif hf_quant_config.get("quant_method", "") == "modelopt_fp4":
            return "modelopt_fp4"

        return None

    @staticmethod
    def get_from_keys(config: Dict[str, Any], keys: List[str]) -> Any:
        """从模型的量化配置字典中获取指定键的值（按顺序尝试多个候选键）。

        输入示例：
            config: {"weight_bits": 4, "group_size": 128}
            keys  : ["bits", "weight_bits"]

        输出示例：
            返回 4（"bits" 不存在但 "weight_bits" 存在）；
            若所有候选键都不存在，抛出 ValueError
        """
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(
            f"Cannot find any of {keys} in the model's " "quantization config."
        )

    @staticmethod
    def get_from_keys_or(config: Dict[str, Any], keys: List[str], default: Any) -> Any:
        """从模型的量化配置字典中获取可选键的值，找不到时返回默认值。

        输入示例：
            config : {"group_size": 128}
            keys   : ["sym", "symmetric"]
            default: True

        输出示例：
            两个键都不存在 -> 返回 True（default）；
            若存在则返回对应值
        """
        try:
            return QuantizationConfig.get_from_keys(config, keys)
        except ValueError:
            return default

    @abstractmethod
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """获取量化层应使用的量化方法。

        输入示例：
            layer : 正在构建的层实例（如 Linear / FusedMoE / Embedding）
            prefix: 该层在 state dict 中的完整名字，
                    如 "model.layers.0.self_attn.qkv_proj"

        输出示例：
            - Linear 层  -> 返回对应的 LinearMethodBase 子类实例
              （如 GPTQMarlinLinearMethod）
            - MoE 层     -> 返回对应的 FusedMoEMethodBase 子类实例
            - 不支持量化的层（如 LayerNorm）-> 返回 None
        """
        raise NotImplementedError()

    @abstractmethod
    def get_scaled_act_names(self) -> List[str]:
        """返回需要后置缩放（post-scale）的激活函数名称列表。

        目前仅 AWQ 使用该接口（AWQ 的激活感知缩放需要作用于
        gate 后的激活值）。

        输入示例：无
        输出示例：
            AWQ  -> ["gelu", "silu"]（视模型激活函数而定）
            其他 -> []
        """
        raise NotImplementedError()

    def apply_weight_name_mapper(
        self, hf_to_sglang_mapper: WeightsMapper
    ):  # noqa: B027
        """
        供模型更新量化配置中引用的模块名的接口，
        使其反映 sglang 的模型结构。

        输入示例：
            hf_to_sglang_mapper: HF 结构到 sglang 结构的权重名映射表，
                如 {"self_attn.q_proj": "self_attn.q_proj"} 等更名规则

        输出示例：
            无返回值；副作用是量化配置内部记录的模块名（如
            ignore 列表、模块映射等）被更新为 sglang 结构下的名字
        """
        pass


def method_has_implemented_embedding(method_class: Type[QuantizeMethodBase]) -> bool:
    """判断量化方法类是否实现了 embedding 方法。

    并非所有量化方法都实现了 embedding（量化 Embedding 层），
    因此需要检查给定的方法类是否实现了它。检查方式是：确认该方法
    与基类 QuantizeMethodBase 中的实现不是同一个（即被子类重写过）。

    输入示例：
        method_class: GPTQMarlinLinearMethod（或任意 QuantizeMethodBase 子类）

    输出示例：
        - 子类定义了 embedding -> 返回 True
        - 子类未定义（沿用了基类的，或基类中根本没有）-> 返回 False
    """
    base_embedding = inspect.getattr_static(QuantizeMethodBase, "embedding", None)
    class_embedding = inspect.getattr_static(method_class, "embedding", None)

    return class_embedding is not None and class_embedding is not base_embedding
