"""插件注册表内部存储，支撑 ``SpeculativeAlgorithm.register``。

插件应使用 ``SpeculativeAlgorithm.register`` 类方法 API；
请勿直接从此模块导入。
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type

import torch

from sglang.srt.arg_groups.overrides import resolving_view

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpecInput

# Worker 工厂类型：接收 ServerArgs，返回 Worker 类
WorkerFactory = Callable[["ServerArgs"], Type]
# 服务参数校验器类型：接收 ServerArgs，无返回值（异常即校验失败）
ServerArgsValidator = Callable[["ServerArgs"], None]

logger = logging.getLogger(__name__)


class CustomSpecAlgo:
    """插件注册的投机算法。

    通过鸭子类型模拟 ``SpeculativeAlgorithm`` 枚举值
    （提供相同的 ``is_*()`` / ``create_worker`` 接口）。

    插件可以继承此类来覆盖任何 ``is_*()`` / ``supports_*()`` / ``create_worker``
    方法（例如为了与调度器/model_runner 中针对内置算法的分支集成，
    如 ``if spec_algorithm.is_eagle():``）。在注册时通过 ``spec_class=...`` 传入子类。

    默认行为：除 ``is_speculative`` 返回 True 外，所有 ``is_*()`` 返回 False。

    ``supports_overlap=False`` 已废弃：spec V1 worker 路径已移除，
    此类算法在 V2 调度器架构上以非 overlap（同步）方式运行。
    请将插件 worker 迁移到 V2 架构和 overlap 调度。

    示例::

        >>> class MySpecAlgo(CustomSpecAlgo):
        ...     def is_eagle(self) -> bool:
        ...         return True  # 让调度器将此算法视为 EAGLE 家族
        ...
        >>> @SpeculativeAlgorithm.register("MY_SPEC", spec_class=MySpecAlgo)
        ... def _factory(server_args):
        ...     return MySpecWorker
    """

    def __init__(
        self,
        name: str,
        factory: WorkerFactory,
        *,
        supports_overlap: bool = False,
        validate_server_args: Optional[ServerArgsValidator] = None,
    ):
        """
        输入:
            name: 算法名称（大写）
            factory: Worker 工厂函数，接收 server_args 返回 Worker 类
            supports_overlap: 是否支持 overlap 调度（已废弃，建议迁移为 True）
            validate_server_args: 可选的参数校验回调
        """
        self.name = name
        self.factory = factory
        self.supports_overlap = supports_overlap
        self.validate_server_args = validate_server_args

    def __repr__(self) -> str:
        return f"CustomSpecAlgo({self.name!r})"

    def is_some(self) -> bool:
        """是否启用了某种投机算法。插件算法始终返回 True。"""
        return True

    def is_none(self) -> bool:
        """是否未启用投机解码。插件算法始终返回 False。"""
        return False

    def is_speculative(self) -> bool:
        """是否启用了投机解码。插件算法始终返回 True。"""
        return True

    def is_eagle(self) -> bool:
        """是否为 EAGLE 家族。默认 False，可由子类覆盖。"""
        return False

    def is_eagle3(self) -> bool:
        """是否为 EAGLE3。默认 False。"""
        return False

    def is_frozen_kv_mtp(self) -> bool:
        """是否为 Frozen-KV MTP。默认 False。"""
        return False

    def is_dflash(self) -> bool:
        """是否为 DFlash。默认 False。"""
        return False

    def is_dspark(self) -> bool:
        return False

    def is_dflash_family(self) -> bool:
        return False

    def is_standalone(self) -> bool:
        """是否为 Standalone。默认 False。"""
        return False

    def is_ngram(self) -> bool:
        """是否为 NGRAM。默认 False。"""
        return False

    def supports_target_verify_for_draft(self) -> bool:
        """草稿阶段是否需要 target verify。默认 False。"""
        return False

    def supports_ragged_verify(self) -> bool:
        return False

    def supports_grammar_overlap(self) -> bool:
        # Whether the worker advances the grammar FSM inside verify() (via the
        # scheduler's grammar barrier), letting spec + grammar decode overlap.
        return False

    def has_draft_kv(self) -> bool:
        """草稿阶段是否写入 KV 链。保守默认值：返回 True（更大的 KV 预留）。"""
        return True

    def handle_server_args(self, server_args: ServerArgs) -> None:
        pass

    def create_worker(self, server_args: ServerArgs) -> Type:

        cfg = resolving_view(server_args)
        if not cfg.disable_overlap_schedule and not self.supports_overlap:
            raise ValueError(
                f"投机算法 {self.name} 不支持 overlap 调度。"
            )
        if not self.supports_overlap:
            # 仅在 overlap 禁用时到达此处，算法确实在 V2 架构上同步运行
            logger.warning(
                "投机算法 %s 注册时 supports_overlap=False，此配置已废弃："
                "spec V1 worker 路径已移除，算法现在在 V2 调度器架构上以非 overlap"
                "（同步）方式运行。请将插件 worker 迁移以支持 overlap 调度。",
                self.name,
            )
        return self.factory(server_args)

    def get_num_tokens_per_req_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:
        """获取 target verify 阶段每个请求的 token 数。

        输入:
            num_draft_tokens: 草稿 token 数
            is_draft_worker: 是否为草稿 worker

        输出:
            每个请求的验证 token 数（当前等于 num_draft_tokens）

        FIXME: 在 forward mode 重构后移除此方法。
        """
        return num_draft_tokens

    def get_num_tokens_per_bs_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:
        # Deprecated alias; remove together with the FIXME above.
        warnings.warn(
            "get_num_tokens_per_bs_for_target_verify is deprecated; use "
            "get_num_tokens_per_req_for_target_verify instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_num_tokens_per_req_for_target_verify(
            num_draft_tokens, is_draft_worker
        )

    def build_disagg_draft_input(
        self,
        batch: ScheduleBatch,
        last_tokens_tensor: torch.Tensor,
        future_map: FutureMap,
    ) -> Optional[SpecInput]:
        """Build the disaggregation draft input for ``batch``, or ``None``.

        The speculative config comes from ``runtime_context.get_spec()``, which
        follows a runtime override where the startup record does not.
        """
        return None


# 全局插件注册表：算法名称 -> CustomSpecAlgo 实例
_REGISTRY: Dict[str, CustomSpecAlgo] = {}

# CLI 别名：不是 ``SpeculativeAlgorithm`` 成员但仍映射到内置算法的名称
# （如 NEXTN -> EAGLE）。与枚举成员一起保留，防止插件覆盖。
_RESERVED_ALIASES = frozenset({"NEXTN"})


def _reserved_names() -> frozenset:
    """获取插件不能注册的名称集合。

    包括所有 ``SpeculativeAlgorithm`` 枚举成员名和 ``_RESERVED_ALIASES``。

    从枚举惰性派生（避免循环导入——``spec_info`` 导入此模块），
    因此新增内置算法会自动被保留，无需编辑第二个列表。

    输出:
        frozenset[str]：所有保留名称

    示例::

        >>> names = _reserved_names()
        >>> 'EAGLE' in names
        True
        >>> 'NEXTN' in names
        True
    """
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    return frozenset(algo.name for algo in SpeculativeAlgorithm) | _RESERVED_ALIASES


def _assert_custom_spec_algo_conforms(spec_class: Type[CustomSpecAlgo]) -> None:
    """校验 ``spec_class`` 是否符合 ``SpeculativeAlgorithm`` 的鸭子类型契约。

    ``from_string`` 返回任意一种类型，调用方基于共享的 ``is_*()`` / ``supports_*()``
    接口分发而无需 isinstance 检查。因此枚举上每个此类方法也必须存在于
    注册的 spec class 上——否则插件算法在调用点会遇到 ``AttributeError``
    （这就是 ``is_some`` / ``is_frozen_kv_mtp`` 曾经遗漏的原因）。
    新增断言方法会被自动覆盖，无需维护第二个列表。

    在 ``register_algorithm`` 中调用而非导入时调用，因为 ``spec_info`` 导入
    此模块，此模块加载时 ``SpeculativeAlgorithm`` 尚未定义；注册时它已完全定义。

    输入:
        spec_class: 待校验的 CustomSpecAlgo 子类

    异常:
        TypeError: 当 spec_class 缺少枚举上的 is_*/supports_* 方法时抛出

    示例::

        >>> # 如果 MySpecAlgo 缺少 is_frozen_kv_mtp 方法：
        >>> _assert_custom_spec_algo_conforms(MySpecAlgo)
        TypeError: MySpecAlgo is missing duck-typed methods from
        SpeculativeAlgorithm: ['is_frozen_kv_mtp']. ...
    """
    # 注意：使用 ``vars()`` 而非 ``dir()``——``EnumMeta.__dir__`` 会隐藏
    # 实例方法，因此 ``dir(SpeculativeAlgorithm)`` 会返回空接口，
    # 使此守卫变成静默空操作。
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    interface = {
        name
        for name in vars(SpeculativeAlgorithm)
        if name.startswith(("is_", "supports_"))
    }
    missing = sorted(interface - set(dir(spec_class)))
    if missing:
        raise TypeError(
            f"{spec_class.__name__} 缺少 SpeculativeAlgorithm 的鸭子类型方法: "
            f"{missing}。请将它们添加到 {spec_class.__name__}，"
            "以确保插件注册的算法可被正确分发。"
        )


def register_algorithm(
    name: str,
    *,
    supports_overlap: bool = False,
    validate_server_args: Optional[ServerArgsValidator] = None,
    spec_class: Type[CustomSpecAlgo] = CustomSpecAlgo,
) -> Callable[[WorkerFactory], WorkerFactory]:
    """返回一个装饰器，将插件算法以 ``name`` 注册到全局注册表。

    输入:
        name: 算法名称（不区分大小写，内部转大写）
        supports_overlap: 是否支持 overlap 调度（默认 False，已废弃）
        validate_server_args: 可选的参数校验回调
        spec_class: CustomSpecAlgo 子类，用于覆盖 is_*() / supports_*() / create_worker

    输出:
        装饰器函数，接收 factory(server_args) -> worker_class 并原样返回

    异常:
        ValueError: 当 name 是保留名或已注册时抛出
        TypeError: 当 spec_class 不符合鸭子类型契约时抛出

    示例::

        >>> @register_algorithm("MY_SPEC")
        ... def _factory(server_args):
        ...     return MySpecWorker
        ...
        >>> algo = get_spec("MY_SPEC")
        >>> algo.name
        'MY_SPEC'
    """
    upper = name.upper()
    if upper in _reserved_names():
        raise ValueError(
            f"'{upper}' 是保留的投机算法名称，不可重新注册。"
        )
    if upper in _REGISTRY:
        raise ValueError(f"投机算法 '{upper}' 已注册。")
    _assert_custom_spec_algo_conforms(spec_class)

    def decorator(factory: WorkerFactory) -> WorkerFactory:
        _REGISTRY[upper] = spec_class(
            name=upper,
            factory=factory,
            supports_overlap=supports_overlap,
            validate_server_args=validate_server_args,
        )
        return factory

    return decorator


def get_spec(name: Optional[str]) -> Optional[CustomSpecAlgo]:
    """根据名称获取已注册的投机算法。

    输入:
        name: 算法名称字符串（不区分大小写），或 None

    输出:
        CustomSpecAlgo 实例，或 None（内置算法名/未知名/None 时返回 None）

    示例::

        >>> get_spec("MY_SPEC")
        CustomSpecAlgo('MY_SPEC')
        >>> get_spec("EAGLE")  # 内置算法不在注册表中
        None
        >>> get_spec(None)
        None
    """
    if name is None:
        return None
    return _REGISTRY.get(name.upper())
