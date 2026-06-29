# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kv_cache.py

import logging

import torch

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

logger = logging.getLogger(__name__)


class BaseKVCacheMethod(QuantizeMethodBase):
    """KV 缓存量化方法的基类。

    它会给 Attention（注意力）层添加 ``k_scale`` 与 ``v_scale`` 两个属性，
    以支持从 checkpoint（模型权重文件）中加载这两个缩放因子。
    k_scale / v_scale 的用途是：
        - 在把 k/v_cache 条目写入缓存前，用它对其做量化（quantize）；
        - 在从缓存读取 k/v_cache 条目时，用它对其做反量化（dequantize）。

    :param quant_config: 对应的量化配置 QuantizationConfig
    """

    def __init__(self, quant_config: QuantizationConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        """为某个 Attention 层创建“权重”（即 k_scale 和 v_scale）。"""
        # 把 KV 缓存的缩放因子初始化为 -1.0 —— 这是一个无效值（标记位）。
        # 如果 checkpoint 中确实带有 k/v_scale，加载权重时会把这里的 -1.0 覆盖掉；
        # 若 checkpoint 中没有，则后续 process_weights_after_loading 会据此回退到默认值 1.0。
        layer.k_scale = torch.nn.Parameter(
            torch.tensor(-1.0, dtype=torch.float32), requires_grad=False
        )
        layer.v_scale = torch.nn.Parameter(
            torch.tensor(-1.0, dtype=torch.float32), requires_grad=False
        )
        # 这两个标量缩放因子不是常规权重，跳过权重加载时的存在性校验，
        # 避免框架因 checkpoint 里没有同名权重而报错。
        layer.k_scale._skip_weight_check = True
        layer.v_scale._skip_weight_check = True

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        # 本类只负责管理缩放因子，不参与前向计算；apply 被调用说明用法有误。
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")

    def process_weights_after_loading(self, layer) -> None:
        """权重加载完成后，最终确定 k_scale / v_scale 的取值。

        根据 create_weights 留下的初始值（-1.0）是否被 checkpoint 覆盖，
        分三种情况处理：两者都有效、两者都无效、只有其一有效。
        """
        if layer.k_scale > 0.0 and layer.v_scale > 0.0:
            # 情况一：checkpoint 同时提供了 k_scale 和 v_scale，优先各自独立使用。
            k_scale = layer.k_scale.to("cpu").tolist()
            v_scale = layer.v_scale.to("cpu").tolist()
            # FNUZ 是 AMD ROCm 上的 fp8 变体（e4m3fnuz），其数值范围只有 OCP fp8 的一半，
            # 因此需要把缩放因子乘以 2 来补偿。
            if is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2
        elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
            # 情况二：checkpoint 中没有任何缩放因子（两者仍是无效的负初始值），
            # 退回到默认值 1.0（相当于不缩放）。
            k_scale = 1.0
            v_scale = 1.0
        else:
            # 情况三：checkpoint 里只有单一的 kv_scale。加载权重时它会被映射到 k_scale，
            # 这里再把 k_scale 复制给 v_scale，使两者共用同一缩放因子。
            assert layer.k_scale > 0.0
            scale_to_duplicate = max(layer.k_scale, layer.v_scale)
            k_scale = scale_to_duplicate.to("cpu").tolist()
            v_scale = scale_to_duplicate.to("cpu").tolist()
            if is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2

        # 当前仅支持 per-tensor（整张张量共用一个标量）的缩放因子；
        # 若取到的不是单个 float（例如 per-channel 列表），则直接报错。
        if not isinstance(k_scale, float) or not isinstance(v_scale, float):
            raise ValueError(
                "Only support per-tensor scaling factor " "for fp8 KV cache"
            )

        # 这两个值会在最终的 Attention.forward() 中被使用。
        # 既写回 Parameter（供持久化/查询），也另存一份 python float 以便高频访问时免去 GPU 取值开销。
        layer.k_scale.copy_(k_scale)
        layer.v_scale.copy_(v_scale)
        layer.k_scale_float = k_scale
        layer.v_scale_float = v_scale
