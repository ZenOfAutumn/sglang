# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
# 中译：本文件移植自 flash-linear-attention（FLA）项目的 fla/utils.py，
#       为 SGLang 中的线性注意力（linear attention）相关 Triton kernel 提供通用工具函数，
#       包括：环境检查、张量缓存、输入规整（contiguous）、设备/平台探测、共享内存能力查询等。
# -*- coding: utf-8 -*-

import contextlib
import functools
import inspect
import logging
import os
import sys
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import torch
import triton
from packaging import version

from sglang.srt.utils.common import torch_release

logger = logging.getLogger(__name__)

# 中译：以下三个环境变量开关，均通过环境变量控制 FLA 的行为。
# COMPILER_MODE：是否处于编译模式（FLA_COMPILER_MODE=1 时开启）。
COMPILER_MODE = os.getenv("FLA_COMPILER_MODE") == "1"
# FLA_CI_ENV：是否处于 CI（持续集成）环境，影响精度断言的宽松程度（见 assert_close）。
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
# FLA_CACHE_RESULTS：是否缓存 Triton autotune 的结果（默认开启），可加速重复运行。
FLA_CACHE_RESULTS = os.getenv("FLA_CACHE_RESULTS", "1") == "1"


# 中译：检测当前 Triton 版本的 autotune 是否支持 cache_results 参数（新版本才有）。
SUPPORTS_AUTOTUNE_CACHE = (
    "cache_results" in inspect.signature(triton.autotune).parameters
)

# 中译：根据上面的探测结果，决定传给 triton.autotune 的额外 kwargs；
#       若不支持 cache_results 则传空字典，避免在旧版本上报错。
autotune_cache_kwargs = (
    {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}
)


@lru_cache(maxsize=1)
def check_environments():
    """
    Checks the current operating system, Triton version, and Python version,
    issuing warnings if they don't meet recommendations.
    This function's body only runs once due to lru_cache.

    中译：检查当前的操作系统、Triton 版本和 Python 版本，若不满足推荐配置则发出警告。
          由于使用了 @lru_cache(maxsize=1)，函数体实际上只会执行一次（结果被缓存）。
    """
    # Check Operating System
    # 中译：检查操作系统——Triton 没有官方的 Windows 版本，因此在 Windows 上不做适配，潜在错误也不会修复。
    if sys.platform == "win32":
        logger.warning(
            "Detected Windows operating system. Triton does not have an official Windows release, "
            "thus FLA will not be adapted for Windows, and any potential errors will not be fixed. "
            "Please consider using a Linux environment for compatibility."
        )

    # 中译：检查 Triton 版本——低于推荐的 3.2.0 时发出警告（可能出错且不会修复）。
    triton_version = version.parse(triton.__version__)
    required_triton_version = version.parse("3.2.0")

    if triton_version < required_triton_version:
        logger.warning(
            f"Current Triton version {triton_version} is below the recommended 3.2.0 version. "
            "Errors may occur and these issues will not be fixed. "
            "Please consider upgrading Triton."
        )

    # Check Python version
    # 中译：检查 Python 版本——建议使用 3.11 或更高版本以获得最佳体验。
    py_version = version.parse(f"{sys.version_info.major}.{sys.version_info.minor}")
    required_py_version = version.parse("3.11")

    if py_version < required_py_version:
        logger.warning(
            f"Current Python version {py_version} is below the recommended 3.11 version. "
            "It is recommended to upgrade to Python 3.11 or higher for the best experience."
        )

    return None


def get_abs_err(x, y):
    # 中译：计算两个张量之间的最大绝对误差（先做差，展平后取绝对值的最大值）。常用于精度校验。
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    # 中译：计算相对误差比率 = RMSE(x - y) / RMS(x)。
    #       err 为差值的均方根（RMSE），base 为参考张量 x 的均方根，分母加 1e-8 防止除零。
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    # 中译：断言参考结果 ref 与待测结果 tri 足够接近，常用于对比「参考实现」与「Triton 实现」的输出。
    #       参数：prefix 日志前缀；ref 参考值；tri 待测值；ratio 允许的相对误差上限；
    #            warning 为 True 时只告警不报错；err_atol 绝对误差容忍阈值。
    abs_atol = get_abs_err(ref, tri)  # 最大绝对误差
    msg = f"{prefix} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    logger.info(msg)
    error_rate = get_err_ratio(ref, tri)  # 相对误差比率
    # 中译：绝对误差已足够小，直接通过。
    if abs_atol <= err_atol:
        return
    # 中译：在「告警模式」或「CI 环境且误差较小」时，超过阈值只发 warning 而不中断。
    if warning or (FLA_CI_ENV and (error_rate < 0.01 or abs_atol <= 0.3)):
        if error_rate > ratio:
            import warnings

            warnings.warn(msg)
    else:
        # 中译：否则严格断言相对误差必须小于阈值，否则抛出 AssertionError。
        assert error_rate < ratio, msg


# 中译：GDN（Gated Delta Net）重计算的抑制级别，通过环境变量配置，用于控制反向重计算的行为。
SUPPRESS_LEVEL = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.
    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 4). When the cache is full, the oldest entry will be removed.
    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.
    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.

    中译：一个针对「张量输入」函数的最近结果缓存装饰器。
          它会缓存最近若干组输入张量对应的输出，缓存容量固定（默认 4 条），
          缓存满后会淘汰最旧的一条（LRU 风格）。
          注意：缓存命中判定使用「身份比较」（is，即同一对象），而非值相等，
               因此只有传入完全相同的张量对象时才会命中缓存。
          参数 fn：被装饰的函数，应当接收张量输入并返回张量输出。
          返回：带缓存能力的包装函数。
    """

    # 中译：缓存条目列表，每条形如 (args, kwargs, result)。
    cache_entries: Tuple[Optional[Tuple], Optional[Dict], Any] = []
    cache_size = 4  # 最大缓存条目数

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries, cache_size
        # 中译：遍历已有缓存，逐条比较位置参数与关键字参数是否「是同一对象」。
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    # 中译：命中缓存——把该条目移动到列表末尾（标记为最近使用），并返回缓存结果。
                    cache_entries = (
                        cache_entries[:i]
                        + cache_entries[i + 1 :]
                        + [(args, kwargs, last_result)]
                    )
                    return last_result

        # 中译：未命中——实际调用原函数计算结果。
        result = fn(*args, **kwargs)

        # 中译：缓存已满则淘汰最旧的一条（列表头部），再把新结果追加到末尾。
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.

    中译：输入守卫装饰器。作用有二：
          1) 把所有输入张量转为内存连续（contiguous），避免 Triton kernel 因非连续布局出错；
          2) 根据输入张量所在的设备，把执行上下文切到对应 GPU 设备上。
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 中译：对位置参数中的张量逐一调用 .contiguous()，非张量保持原样。
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        # 中译：对关键字参数中的张量同样做连续化处理。
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        # 中译：寻找第一个张量参数（先找位置参数，再找关键字参数），用它来确定目标设备。
        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        # 中译：若找到张量则切到其所在设备的上下文；否则用空上下文（不切换设备）。
        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


# 中译：contiguous 是 input_guard 的别名，便于在 kernel 代码中以更直观的名字使用。
contiguous = input_guard


def require_version(version, hint):
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.

    中译：运行时依赖版本检查装饰器，使用与 pip 完全相同的版本约束语法（如 "torch>=2.4"）。
          被装饰函数在执行前会先校验版本，同时顺带把张量参数做连续化处理。
          参数：version 版本约束字符串；hint 校验失败时给用户的提示信息。
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            from transformers.utils.versions import require_version

            require_version(version, hint)
            return fn(
                ctx,
                *(
                    i if not isinstance(i, torch.Tensor) else i.contiguous()
                    for i in args
                ),
                **{
                    k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
                    for k, v in kwargs.items()
                },
            )

        return wrapper

    return decorator


def checkpoint(fn):
    # 中译：把函数包装为「梯度检查点（gradient checkpointing）」模式，
    #       前向不保存中间激活，反向时重算，以显存换计算，降低训练显存占用。
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)

    return wrapper


def _cpu_device_warning():
    # 中译：当前平台不支持 Triton 时发出警告，提示已回退到 CPU。
    import warnings

    warnings.warn(
        ("Triton is not supported on current platform, roll back to CPU."), stacklevel=1
    )


@lru_cache(maxsize=None)
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    # 中译：查询指定设备的 SM（流多处理器，multiprocessor）数量，常用于 kernel 的并行度配置。
    #       查询失败（如 CPU 环境）则告警并返回 -1。
    try:
        return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)[
            "multiprocessor_count"
        ]
    except BaseException:
        _cpu_device_warning()
        return -1


@lru_cache(maxsize=None)
def get_available_device() -> str:
    # 中译：返回当前 Triton 后端名称（如 cuda / hip / xpu），探测失败则回退为 "cpu"。
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return "cpu"


@lru_cache(maxsize=None)
def _check_platform() -> Literal["nvidia", "amd", "intel", "musa"]:
    # 中译：把 Triton 后端名称映射为更直观的厂商名：cuda→nvidia、hip→amd、xpu→intel，其余原样返回。
    device = get_available_device()
    if device == "cuda":
        return "nvidia"
    elif device == "hip":
        return "amd"
    elif device == "xpu":
        return "intel"
    else:
        return device


# For AMD GPUs, the triton backend is 'hip', while for Nvidia GPUs, the triton backend is 'cuda'.
# However, the torch backend is 'cuda' for both Nvidia and AMD GPUs.
# Therefore, we need to check the triton backend to determine the actual GPU vendor.
# 中译：对于 AMD GPU，Triton 后端是 'hip'；而 Nvidia GPU 的 Triton 后端是 'cuda'。
#       但 torch 侧无论 Nvidia 还是 AMD 后端都是 'cuda'，
#       因此需要通过 Triton 后端来判断实际的 GPU 厂商。这里把 hip 统一映射为 torch 认识的 cuda。
device = get_available_device() if get_available_device() != "hip" else "cuda"
device_torch_lib = getattr(torch, device)  # 对应设备的 torch 子模块（如 torch.cuda）
device_platform = _check_platform()  # 当前平台厂商（nvidia/amd/intel/...）

# 中译：以下布尔量用于区分不同 GPU 厂商与架构，供 kernel 选择不同代码路径。
is_amd = device_platform == "amd"  # 是否为 AMD GPU
is_intel = device_platform == "intel"  # 是否为 Intel GPU
is_nvidia = device_platform == "nvidia"  # 是否为 Nvidia GPU
# 中译：是否为 Intel Alchemist（Arc A 系列）显卡。
is_intel_alchemist = is_intel and "Intel(R) Arc(TM) A" in torch.xpu.get_device_name(0)
# 中译：是否为 Nvidia Hopper 架构（H 系列，或计算能力主版本 >= 9）。
is_nvidia_hopper = is_nvidia and (
    "NVIDIA H" in torch.cuda.get_device_name(0)
    or torch.cuda.get_device_capability()[0] >= 9
)
# 中译：是否启用 CUDA Graph（仅 Nvidia 且环境变量 FLA_USE_CUDA_GRAPH=1 时）。
use_cuda_graph = is_nvidia and os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"

# Nvidia Ampere or newer, haven't check AMD and intel yet.
# 中译：是否支持 TF32（Nvidia Ampere 及更新架构，计算能力主版本 >= 8）；AMD/Intel 尚未核实。
is_tf32_supported = is_nvidia and torch.cuda.get_device_capability(0)[0] >= 8
# 中译：Triton 语言是否提供 gather 原语（新版 Triton 才有）。
is_gather_supported = hasattr(triton.language, "gather")


def get_all_max_shared_mem():
    # 中译：获取所有设备的「最大共享内存（max_shared_mem）」列表，单位为字节。
    #       该值决定了 Triton kernel 可用的 SRAM 容量，进而影响分块（tiling）策略。
    #       查询失败（如 CPU 环境）则告警并返回 [-1]。
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)[
                "max_shared_mem"
            ]
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        _cpu_device_warning()
        return [-1]


class Backend(Enum):
    # 中译：枚举不同 GPU 架构对应的单 SM 共享内存容量（字节），用于判断设备能否跑某些 kernel。
    ADA = 101376  # RTX 4090
    AMPERE = 166912  # A100
    HOPPER = 232448  # H100
    DEFAULT = 102400  # Default（中译：未知架构的默认值）

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        # 中译：根据架构名（如 "hopper"）返回对应的共享内存阈值；未知架构返回默认值。
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@lru_cache(maxsize=None)
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    # 中译：判断指定设备的共享内存是否足以满足某架构需求，用于在运行前筛选可用的 kernel 实现。
    #       任何异常（如无法查询）均返回 False，表示不满足。
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


# 中译：根据 PyTorch 版本选择不同的混合精度（AMP）装饰器与设备上下文构造方式。
if torch_release >= (2, 4):
    # 中译：≥ 2.4 使用新的 torch.amp 通用 API，可通过 device_type 指定设备类型；若为 cpu 则回退为 cuda。
    device = "cuda" if device == "cpu" else device
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)

    def custom_device_ctx(index: int):
        # 中译：返回指定索引设备的上下文管理器（如 torch.cuda.device(index)）。
        return device_torch_lib.device(index)

else:
    # 中译：< 2.4 的旧版本只支持 cuda 设备，并使用旧的 device_torch_lib.amp API。
    assert (
        device == "cuda"
    ), "Only cuda device is supported for PyTorch version < 2.4.0."
    autocast_custom_fwd = device_torch_lib.amp.custom_fwd
    autocast_custom_bwd = device_torch_lib.amp.custom_bwd

    def custom_device_ctx(index: int):
        return torch.cuda.device(index)


# 中译：重新获取一次可用设备名称（此处复用为平台标识，注意与前面的 device_platform 含义略有不同）。
device_platform = get_available_device()
