from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    OrderedDict,
    Protocol,
    Tuple,
    TypeGuard,
    Union,
    runtime_checkable,
)

import torch

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs
    from sglang.srt.layers.moe.token_dispatcher import (
        AscendTPCombineInput,
        AscendTPDispatchOutput,
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
        DeepEPv2CombineInput,
        DeepEPv2DispatchOutput,
        FlashinferCombineInput,
        FlashinferDispatchOutput,
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import TopKOutput


# ------------------------------ Dispatcher 钩子机制 -----------------------------------


class _RemovableDispatcherHandle:
    """一个可移除已注册钩子的轻量句柄。

    句柄只弱引用钩子字典，因此不会延长 dispatcher 或钩子容器的生命周期。
    调用 :meth:`remove` 后，对应钩子不再参与后续 dispatch/combine。

    输入输出示例：
        输入：包含 ``{0: hook_fn}`` 的 ``OrderedDict``。
        输出：句柄的 ``id`` 为 0；调用 ``remove()`` 后字典变为空。
    """

    next_id = 0  # 全局递增计数器，保证不同钩子句柄的 ID 唯一

    def __init__(self, hooks_dict: OrderedDict):
        """保存钩子字典的弱引用并分配唯一 ID。

        输入示例：``hooks_dict = OrderedDict()``。
        输出示例：创建首个句柄后 ``self.id == 0``，原字典内容不变。
        """
        self.id = _RemovableDispatcherHandle.next_id
        _RemovableDispatcherHandle.next_id += 1
        self.weak_hooks_dict = weakref.ref(hooks_dict)

    def remove(self):
        """从原钩子字典中删除本句柄对应的钩子。

        如果钩子已经删除，或原字典已被回收，则该操作静默返回，因而可重复调用。

        输入示例：字典为 ``{self.id: hook_fn}``。
        输出示例：调用后字典中不再包含 ``self.id``，函数返回 ``None``。
        """
        hooks_dict = self.weak_hooks_dict()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]


class DispatcherBaseHooks:
    """dispatcher 钩子链的基础容器。

    钩子按注册顺序保存在 ``OrderedDict`` 中。具体的 pre/post 子类负责定义参数、
    返回值以及多个钩子之间如何串联。
    """

    def __init__(self):
        """创建空钩子链。

        输入：无。
        输出示例：``self.hook_dict == OrderedDict()``。
        """
        self.hook_dict = OrderedDict[int, Callable]()

    def register_hook(self, hook_fun: Callable) -> _RemovableDispatcherHandle:
        """把一个钩子追加到调用链并返回可移除句柄。

        输入示例：``hook_fun(dispatcher, value) -> value``。
        输出示例：返回 ``handle``，且 ``hook_dict[handle.id] is hook_fun``。
        """
        handle = _RemovableDispatcherHandle(self.hook_dict)
        self.hook_dict[handle.id] = hook_fun
        return handle

    def __call__(self, *args, **kwargs) -> Optional[Any]:
        """调用钩子链；参数和返回值由子类定义。

        输入示例：pre-dispatch 子类接收 dispatcher、hidden states 和 Top-K 结果。
        输出示例：返回钩子逐个变换后的 hidden states 与 Top-K 结果。
        """
        raise NotImplementedError("子类必须实现此方法")


class _PreDispatchHooks(DispatcherBaseHooks):
    """在真正执行 token dispatch 之前串行变换输入。"""

    def __call__(
        self,
        dispatcher: BaseDispatcher,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> Optional[Tuple[torch.Tensor, TopKOutput]]:
        """依注册顺序执行所有 pre-dispatch 钩子。

        钩子返回 ``None`` 表示只执行副作用、不替换输入；返回二元组则将其作为后续
        钩子的输入。这样可用于量化、埋点或 overlap 前处理。

        输入示例：``hidden_states.shape == [8, 4096]``，Top-K shape 为 ``[8, 2]``。
        输出示例：返回 ``(hidden_states, topk_output)``，shape 通常保持不变。
        """
        for hook_fun in self.hook_dict.values():
            hook_output = hook_fun(dispatcher, hidden_states, topk_output)
            if hook_output is not None:
                hidden_states, topk_output = hook_output
        return hidden_states, topk_output


class _PostDispatchHooks(DispatcherBaseHooks):
    """在 dispatcher 产生通信结果后串行变换 ``DispatchOutput``。"""

    def __call__(
        self, dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
    ) -> Optional[DispatchOutput]:
        """依注册顺序执行所有 post-dispatch 钩子。

        输入示例：格式为 ``DEEPEP_LL`` 的 dispatch 输出。
        输出示例：返回补充了异步接收信息的同格式输出；钩子返回 ``None`` 时保留原值。
        """
        for hook_fun in self.hook_dict.values():
            hook_output = hook_fun(dispatcher, dispatch_output)
            if hook_output is not None:
                dispatch_output = hook_output
        return dispatch_output


class _PreCombineHooks(DispatcherBaseHooks):
    """在反向通信 combine 之前串行变换 ``CombineInput``。"""

    def __call__(
        self, dispatcher: BaseDispatcher, combine_input: CombineInput
    ) -> Optional[CombineInput]:
        """依注册顺序执行所有 pre-combine 钩子。

        输入示例：本地专家计算生成的 ``StandardCombineInput([8, 4096])``。
        输出示例：返回相同格式的 combine 输入，hidden states 可被钩子替换。
        """
        for hook_fun in self.hook_dict.values():
            hook_output = hook_fun(dispatcher, combine_input)
            if hook_output is not None:
                combine_input = hook_output
        return combine_input


class _PostCombineHooks(DispatcherBaseHooks):
    """在 combine 恢复来源 token 顺序后串行变换最终 hidden states。"""

    def __call__(
        self, dispatcher: BaseDispatcher, hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """依注册顺序执行所有 post-combine 钩子。

        输入示例：``hidden_states.shape == [8, 4096]``。
        输出示例：返回钩子处理后的 ``[8, 4096]`` 张量；通常不改变 shape。
        """
        for hook_fun in self.hook_dict.values():
            hook_output = hook_fun(dispatcher, hidden_states)
            if hook_output is not None:
                hidden_states = hook_output
        return hidden_states


# ------------------------------ Dispatch 输出协议 -------------------------------------


class DispatchOutputChecker:
    """按 ``format`` 判断 dispatch 输出的具体类型。

    这些方法返回 ``TypeGuard``，不仅执行运行时判断，也帮助类型检查器把协议类型收窄为
    对应的 ``NamedTuple``，从而安全访问该后端专属字段。
    """

    @staticmethod
    def format_is_standard(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[StandardDispatchOutput]:
        """判断是否为标准格式。

        输入示例：``StandardDispatchOutput(hidden, None, topk_output)``。
        输出示例：标准格式返回 ``True``，DeepEP 格式返回 ``False``。
        """
        return dispatch_output.format.is_standard()

    @staticmethod
    def format_is_triton_kernels(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[StandardDispatchOutput]:
        """判断输出是否可作为 Triton-kernels 的标准 dispatch 格式使用。

        当前 Triton-kernels 复用 ``STANDARD`` 协议，因此该方法与
        :meth:`format_is_standard` 的判断相同。

        输入示例：``dispatch_output.format == STANDARD``。
        输出示例：返回 ``True``。
        """
        return dispatch_output.format.is_standard()

    @staticmethod
    def format_is_ascend_tp(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[AscendTPDispatchOutput]:
        return dispatch_output.format.is_ascend_tp()

    @staticmethod
    def format_is_deepep_normal(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[DeepEPNormalDispatchOutput]:
        """判断是否为 DeepEP normal 兼容格式。

        输入示例：``dispatch_output.format == DEEPEP_NORMAL``。
        输出示例：返回 ``True``，调用方随后可读取 ``num_recv_tokens_per_expert``。
        """
        return dispatch_output.format.is_deepep_normal()

    @staticmethod
    def format_is_deepep_ll(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[DeepEPLLDispatchOutput]:
        """判断是否为 DeepEP low-latency 兼容格式。

        输入示例：``dispatch_output.format == DEEPEP_LL``。
        输出示例：返回 ``True``，调用方可进入 masked grouped GEMM 路径。
        """
        return dispatch_output.format.is_deepep_ll()

    @staticmethod
    def format_is_deepep(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput]]:
        """判断是否属于任一 DeepEP 兼容格式。

        输入示例：格式为 ``DEEPEP_NORMAL`` 或 ``DEEPEP_LL``。
        输出示例：两者均返回 ``True``；``STANDARD`` 返回 ``False``。
        """
        return dispatch_output.format.is_deepep()

    @staticmethod
    def format_is_flashinfer(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[FlashinferDispatchOutput]:
        """判断是否为 FlashInfer A2A 格式。

        输入示例：``dispatch_output.format == FLASHINFER``。
        输出示例：返回 ``True``，调用方可读取可选的 ``moe_output`` workspace。
        """
        return dispatch_output.format.is_flashinfer()

    @staticmethod
    def format_is_deepep_v2(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[DeepEPv2DispatchOutput]:
        return dispatch_output.format.is_deepep_v2()


class DispatchOutputFormat(Enum):
    """dispatcher 与 MoE runner 之间的 dispatch 输出格式。

    格式描述数据布局契约，而不一定与通信库一一对应。例如 Mooncake、NIXL 和 Mori
    可以复用 DeepEP 的 normal 或 low-latency 格式。
    """

    STANDARD = "standard"
    DEEPEP_NORMAL = "deepep_normal"
    DEEPEP_LL = "deepep_ll"
    FLASHINFER = "flashinfer"
    DEEPEP_V2 = "deepep_v2"
    ASCEND_TP = "ascend_tp"

    def is_standard(self) -> bool:
        """是否为标准格式。

        输入示例：``DispatchOutputFormat.STANDARD``；输出：``True``。
        """
        return self == DispatchOutputFormat.STANDARD

    def is_ascend_tp(self) -> bool:
        return self == DispatchOutputFormat.ASCEND_TP

    def is_deepep_normal(self) -> bool:
        """是否为 DeepEP normal 兼容格式。

        输入示例：``DispatchOutputFormat.DEEPEP_NORMAL``；输出：``True``。
        """
        return self == DispatchOutputFormat.DEEPEP_NORMAL

    def is_deepep_ll(self) -> bool:
        """是否为 DeepEP low-latency 兼容格式。

        输入示例：``DispatchOutputFormat.DEEPEP_LL``；输出：``True``。
        """
        return self == DispatchOutputFormat.DEEPEP_LL

    def is_deepep(self) -> bool:
        """是否为任一 DeepEP 兼容格式。

        输入示例：``DEEPEP_NORMAL``、``DEEPEP_LL``；输出：二者均为 ``True``。
        """
        return self in [
            DispatchOutputFormat.DEEPEP_NORMAL,
            DispatchOutputFormat.DEEPEP_LL,
        ]

    def is_flashinfer(self) -> bool:
        """是否为 FlashInfer A2A 格式。

        输入示例：``DispatchOutputFormat.FLASHINFER``；输出：``True``。
        """
        return self == DispatchOutputFormat.FLASHINFER

    def is_deepep_v2(self) -> bool:
        return self == DispatchOutputFormat.DEEPEP_V2


@runtime_checkable
class DispatchOutput(Protocol):
    """不同 dispatcher 输出共同遵守的结构化协议。

    ``hidden_states`` 是通信后、交给本地专家计算的数据；首维可能已经不等于本 rank
    原始 token 数。``format`` 告诉 runner 应使用哪种 pre-permute 和计算布局。

    输入输出示例：
        dispatch 输入 ``hidden_states.shape == [8, 4096]``，Top-K 为 ``[8, 2]``；
        某 EP rank 可能输出 ``hidden_states.shape == [11, 4096]``，表示它从所有 rank
        共接收了 11 个 token。具体的 Top-K 和计数字段由实现类型补充。
    """

    hidden_states: torch.Tensor

    @property
    def format(self) -> DispatchOutputFormat:
        """返回当前对象的数据布局格式。

        输入：当前协议对象。
        输出示例：``DispatchOutputFormat.DEEPEP_LL``。
        """
        ...


# ------------------------------ Combine 输入协议 --------------------------------------


class CombineInputChecker:
    """按 ``format`` 判断 combine 输入的具体类型，并为类型检查器收窄类型。"""

    @staticmethod
    def format_is_standard(
        combine_input: CombineInput,
    ) -> TypeGuard[StandardCombineInput]:
        """判断是否为标准 combine 格式。

        输入示例：``StandardCombineInput(hidden_states=[8, 4096])``。
        输出示例：返回 ``True``。
        """
        return combine_input.format == CombineInputFormat.STANDARD

    @staticmethod
    def format_is_ascend_tp(
        combine_input: CombineInput,
    ) -> TypeGuard[AscendTPCombineInput]:
        return combine_input.format == CombineInputFormat.ASCEND_TP

    @staticmethod
    def format_is_deepep_normal(
        combine_input: CombineInput,
    ) -> TypeGuard[DeepEPNormalCombineInput]:
        """判断是否为 DeepEP normal 兼容 combine 格式。

        输入示例：``combine_input.format == DEEPEP_NORMAL``；输出：``True``。
        """
        return combine_input.format == CombineInputFormat.DEEPEP_NORMAL

    @staticmethod
    def format_is_deepep_ll(
        combine_input: CombineInput,
    ) -> TypeGuard[DeepEPLLCombineInput]:
        """判断是否为 DeepEP low-latency 兼容 combine 格式。

        输入示例：``combine_input.format == DEEPEP_LL``；输出：``True``。
        """
        return combine_input.format == CombineInputFormat.DEEPEP_LL

    @staticmethod
    def format_is_deepep(
        combine_input: CombineInput,
    ) -> TypeGuard[Union[DeepEPNormalCombineInput, DeepEPLLCombineInput]]:
        """判断是否属于任一 DeepEP 兼容 combine 格式。

        输入示例：格式为 ``DEEPEP_NORMAL`` 或 ``DEEPEP_LL``。
        输出示例：两者均返回 ``True``。
        """
        return combine_input.format in [
            CombineInputFormat.DEEPEP_NORMAL,
            CombineInputFormat.DEEPEP_LL,
        ]

    @staticmethod
    def format_is_flashinfer(
        combine_input: CombineInput,
    ) -> TypeGuard[FlashinferCombineInput]:
        """判断是否为 FlashInfer combine 格式。

        输入示例：``combine_input.format == FLASHINFER``；输出：``True``。
        """
        return combine_input.format == CombineInputFormat.FLASHINFER

    @staticmethod
    def format_is_deepep_v2(
        combine_input: CombineInput,
    ) -> TypeGuard[DeepEPv2CombineInput]:
        return combine_input.format == CombineInputFormat.DEEPEP_V2


class CombineInputFormat(Enum):
    """本地专家计算结果传回 dispatcher.combine 时采用的数据格式。"""

    STANDARD = "standard"
    DEEPEP_NORMAL = "deepep_normal"
    DEEPEP_LL = "deepep_ll"
    FLASHINFER = "flashinfer"
    DEEPEP_V2 = "deepep_v2"
    ASCEND_TP = "ascend_tp"


@runtime_checkable
class CombineInput(Protocol):
    """不同后端 combine 输入共同遵守的结构化协议。

    当前公共协议只要求提供 ``format``。大多数具体实现还包含专家输出
    ``hidden_states``，DeepEP 兼容格式还会携带 ``topk_ids`` 和 ``topk_weights``。

    输入输出示例：
        输入对象可表示本 rank 专家计算后的 ``hidden_states.shape == [11, 4096]``；
        经过 ``dispatcher.combine`` 后，来源 rank 恢复为原始 ``[8, 4096]``。
    """

    # TODO：待所有后端统一字段后，将 hidden_states 加入公共协议。

    @property
    def format(self) -> CombineInputFormat:
        """返回当前 combine 输入的数据布局格式。

        输入：当前协议对象。
        输出示例：``CombineInputFormat.STANDARD``。
        """
        ...


# ------------------------------ Dispatcher 抽象基类 -----------------------------------


class BaseDispatcherConfig(ABC):
    """dispatcher 配置对象的抽象基类。

    目前基类不规定公共字段，各后端可定义自己的配置内容。

    输入输出示例：
        ``DeepEPConfig`` 继承该类，输入配置文件内容，输出 normal/combine 通信配置。
    """

    pass


class BaseDispatcher(ABC):
    """所有 MoE token dispatcher 的统一抽象基类。

    子类实现 ``dispatch`` 和 ``combine`` 两个方向的通信：前者把来源 token 发送到
    专家所在 rank，后者把专家输出送回 token 来源 rank。基类额外提供钩子、量化配置
    和通信计算重叠所需的临时参数。
    """

    def __init__(self):
        """初始化公共配置、overlap 状态和四类钩子容器。

        输入：无。
        输出示例：``quant_config == {}``，所有 overlap 参数和钩子均为 ``None``。
        """
        self.quant_config: dict = {}

        # 通信计算重叠参数：仅在一次重叠执行窗口内有效，由调用方设置并及时清理。
        self.overlap_args: Optional[CombineOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

        # 钩子按需创建；未注册钩子时不包装 dispatch/combine，避免正常路径额外开销。
        self._pre_dispatch_hooks: Optional[_PreDispatchHooks] = None
        self._post_dispatch_hooks: Optional[_PostDispatchHooks] = None
        self._pre_combine_hooks: Optional[_PreCombineHooks] = None
        self._post_combine_hooks: Optional[_PostCombineHooks] = None
        self._original_dispatch_func: Optional[Callable] = None
        self._original_combine_func: Optional[Callable] = None

    @abstractmethod
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        """按 Top-K 路由结果把 token 分发到目标专家所在 rank。

        参数：
            hidden_states：本 rank 当前 token，典型 shape 为 ``[T_local, H]``。
            topk_output：每个 token 的专家 ID、权重和可选 router logits，Top-K 字段
                典型 shape 为 ``[T_local, K]``。

        返回：
            某种 ``DispatchOutput``。其 hidden 首维由本 rank 实际接收的 token 或
            token-expert assignment 决定，不保证等于 ``T_local``。

        输入输出示例：
            输入 hidden 为 ``[8, 4096]``、Top-K 为 ``[8, 2]``；该 rank 从 EP 组
            接收 11 个 token 后，可输出 hidden ``[11, 4096]`` 及对应路由元数据。
        """
        pass

    def _dispatch_with_hook(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        """依次执行 pre-dispatch、原始 dispatch 和 post-dispatch 钩子。

        输入示例：hidden ``[8, 4096]`` 和 Top-K ``[8, 2]``。
        输出示例：返回原始 dispatch 结果，或 post hook 替换后的同协议结果。
        """
        if self._pre_dispatch_hooks is not None:
            hidden_states, topk_output = self._pre_dispatch_hooks(
                self, hidden_states, topk_output
            )
        # 该包装器只会由 _override_dispatch_func 安装，此时原方法必定已保存。
        assert self._original_dispatch_func is not None
        dispatch_output = self._original_dispatch_func(
            hidden_states=hidden_states, topk_output=topk_output
        )
        if self._post_dispatch_hooks is not None:
            dispatch_output = self._post_dispatch_hooks(self, dispatch_output)
        return dispatch_output

    def _override_dispatch_func(self) -> None:
        """首次注册 dispatch 钩子时，用带钩子的包装函数替换实例方法。

        原始绑定方法只保存一次，避免多次注册钩子造成重复包装或递归调用。

        输入：当前 dispatcher 实例。
        输出示例：``self.dispatch`` 指向 ``_dispatch_with_hook``，原实现保存在
        ``_original_dispatch_func``；函数返回 ``None``。
        """
        if self._original_dispatch_func is None:
            self._original_dispatch_func = self.dispatch
            self.dispatch = self._dispatch_with_hook

    @abstractmethod
    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        """把本地专家输出送回来源 rank，并恢复来源 token 的输出顺序。

        参数：
            combine_input：由 MoE runner 生成、格式与 dispatch 输出匹配的专家计算结果。

        返回：
            来源 rank 的 hidden states，典型 shape 恢复为 ``[T_local, H]``。

        输入输出示例：
            输入可能包含本 rank 处理的 11 个接收 token，即 ``[11, 4096]``；
            combine 完成后，当前来源 rank 得到其原始 8 个 token，即 ``[8, 4096]``。
        """
        pass

    def _combine_with_hook(self, combine_input: CombineInput) -> torch.Tensor:
        """依次执行 pre-combine、原始 combine 和 post-combine 钩子。

        输入示例：格式为 ``DEEPEP_LL`` 的本地专家计算结果。
        输出示例：返回 combine 后的 ``[T_local, H]`` hidden states。
        """
        if self._pre_combine_hooks is not None:
            combine_input = self._pre_combine_hooks(self, combine_input)
        # 该包装器只会由 _override_combine_func 安装，此时原方法必定已保存。
        assert self._original_combine_func is not None
        hidden_states = self._original_combine_func(combine_input=combine_input)
        if self._post_combine_hooks is not None:
            hidden_states = self._post_combine_hooks(self, hidden_states)
        return hidden_states

    def _override_combine_func(self) -> None:
        """首次注册 combine 钩子时，用带钩子的包装函数替换实例方法。

        输入：当前 dispatcher 实例。
        输出示例：保存原始 combine 方法并安装 ``_combine_with_hook``；返回 ``None``。
        """
        if self._original_combine_func is None:
            self._original_combine_func = self.combine
            self.combine = self._combine_with_hook

    def register_pre_dispatch_hook(
        self,
        hook: Callable[
            [BaseDispatcher, torch.Tensor, TopKOutput],
            Optional[Tuple[torch.Tensor, TopKOutput]],
        ],
    ) -> _RemovableDispatcherHandle:
        """注册 dispatch 前置钩子。

        钩子可返回新的 ``(hidden_states, topk_output)``；返回 ``None`` 表示只执行副作用。

        输入示例：``hook(dispatcher, hidden, topk) -> (hidden, topk)``。
        输出示例：返回 ``handle``；调用 ``handle.remove()`` 可取消该钩子。
        """
        if self._pre_dispatch_hooks is None:
            self._pre_dispatch_hooks = _PreDispatchHooks()
            self._override_dispatch_func()
        handle = self._pre_dispatch_hooks.register_hook(hook)
        return handle

    def register_post_dispatch_hook(
        self, hook: Callable[[BaseDispatcher, DispatchOutput], Optional[DispatchOutput]]
    ) -> _RemovableDispatcherHandle:
        """注册 dispatch 后置钩子。

        输入示例：``hook(dispatcher, output) -> output``。
        输出示例：返回可移除句柄；钩子可替换 dispatch 输出，也可返回 ``None``。
        """
        if self._post_dispatch_hooks is None:
            self._post_dispatch_hooks = _PostDispatchHooks()
            self._override_dispatch_func()
        handle = self._post_dispatch_hooks.register_hook(hook)
        return handle

    def register_pre_combine_hook(
        self, hook: Callable[[BaseDispatcher, CombineInput], Optional[CombineInput]]
    ) -> _RemovableDispatcherHandle:
        """注册 combine 前置钩子。

        输入示例：``hook(dispatcher, combine_input) -> combine_input``。
        输出示例：返回可移除句柄；钩子可替换 combine 输入，也可返回 ``None``。
        """
        if self._pre_combine_hooks is None:
            self._pre_combine_hooks = _PreCombineHooks()
            self._override_combine_func()
        handle = self._pre_combine_hooks.register_hook(hook)
        return handle

    def register_post_combine_hook(
        self, hook: Callable[[BaseDispatcher, torch.Tensor], Optional[torch.Tensor]]
    ) -> _RemovableDispatcherHandle:
        """注册 combine 后置钩子。

        输入示例：``hook(dispatcher, hidden_states) -> hidden_states``，其中 hidden shape
        为 ``[T_local, H]``。
        输出示例：返回可移除句柄；钩子可替换最终 hidden，也可返回 ``None``。
        """
        if self._post_combine_hooks is None:
            self._post_combine_hooks = _PostCombineHooks()
            self._override_combine_func()
        handle = self._post_combine_hooks.register_hook(hook)
        return handle

    def set_quant_config(self, quant_config: dict) -> None:
        """保存 dispatcher 通信或 payload 量化所需的配置。

        配置内容由后端解释，例如 FlashInfer dispatch 可读取 ``input_global_scale``。

        输入示例：``{"input_global_scale": scale_tensor}``。
        输出示例：``self.quant_config`` 指向该字典；函数返回 ``None``。
        """
        self.quant_config = quant_config

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ) -> None:
        """设置当前通信计算重叠窗口使用的运行时参数。

        参数：
            combine_overlap_args：SBO combine 与后续计算重叠需要的结构化参数。
            meta_overlap_args：后端或 runner 共享的附加元数据。

        输入示例：``set_overlap_args(combine_args, {"stream": comm_stream})``。
        输出示例：两个参数保存在实例上；函数返回 ``None``。
        """
        self.overlap_args = combine_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        """清除一次 overlap 执行结束后遗留的临时参数。

        输入：无。
        输出示例：``overlap_args`` 和 ``meta_overlap_args`` 均恢复为 ``None``。
        """
        self.overlap_args = None
        self.meta_overlap_args = None
