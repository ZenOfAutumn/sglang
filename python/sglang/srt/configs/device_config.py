import logging
from typing import Optional

import torch

from sglang.srt.platforms import current_platform

logger = logging.getLogger(__name__)

# SGLang 内置支持的设备类型白名单。
# cuda: NVIDIA GPU; xpu: Intel GPU; hpu: Habana Gaudi; cpu: CPU;
# npu: 华为昇腾; musa: 摩尔线程 GPU; mps: Apple Silicon GPU。
SUPPORTED_DEVICES = ["cuda", "xpu", "hpu", "cpu", "npu", "musa", "mps"]


class DeviceConfig:
    """设备配置类。

    描述模型运行所在的硬件设备(设备类型 + 具体卡号)，供加载与执行链路
    决定张量应放置到哪个 `torch.device` 上。
    """

    # 解析后的 torch 设备对象(如 torch.device("cuda"))。
    device: Optional[torch.device]
    # 设备序号(GPU/NPU 卡号)；-1 表示未显式指定，由运行时自行决定。
    gpu_id: Optional[int]

    def __init__(self, device: str = "cuda", gpu_id: int = -1) -> None:
        # 校验设备类型：必须在内置白名单中，或属于外部注册(out-of-tree)平台，
        # 否则视为不支持并抛出异常。
        if device in SUPPORTED_DEVICES or current_platform.is_out_of_tree():
            self.device_type = device
        else:
            raise RuntimeError(f"Not supported device type: {device}")
        # 由设备类型字符串构造 torch.device 对象。
        self.device = torch.device(self.device_type)
        # 记录具体卡号。
        self.gpu_id = gpu_id
