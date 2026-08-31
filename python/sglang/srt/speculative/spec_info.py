from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type, Union

import torch

from sglang.srt.speculative.spec_registry import (
    CustomSpecAlgo,
    ServerArgsValidator,
    WorkerFactory,
)
from sglang.srt.speculative.spec_registry import get_spec as _get_registered_spec
from sglang.srt.speculative.spec_registry import (
    register_algorithm as _register_algorithm,
)

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.ngram_worker import NGRAMWorker


class SpeculativeAlgorithm(Enum):
    """内置投机解码算法枚举。

    插件注册的算法是 ``CustomSpecAlgo`` 实例；
    ``from_string`` 返回任意一种类型，两者都暴露相同的 ``is_*()`` / ``create_worker``
    接口，因此调用方无需 isinstance 检查即可统一分发。

    示例::

        >>> algo = SpeculativeAlgorithm.from_string("EAGLE")
        >>> algo.is_eagle()
        True
        >>> algo.is_ngram()
        False
    """

    DFLASH = auto()          # DeepSeek Flash 投机，线性验证
    EAGLE = auto()           # EAGLE 单层草稿模型 + 树形验证
    EAGLE3 = auto()          # EAGLE v3，支持 aux hidden state
    FROZEN_KV_MTP = auto()   # 冻结 KV 的 Multi-Token Prediction
    STANDALONE = auto()      # 独立草稿模型，不与目标模型共享 embedding/lm_head
    NGRAM = auto()           # 基于 N-gram 语料库匹配，无需草稿模型
    NONE = auto()            # 未启用投机解码

    @classmethod
    def from_string(
        cls, name: Optional[str]
    ) -> Union[SpeculativeAlgorithm, CustomSpecAlgo]:
        """从字符串解析投机算法。

        输入:
            name: 算法名称字符串（大小写不敏感），如 "EAGLE"、"ngram"、None

        输出:
            SpeculativeAlgorithm 枚举值，或 CustomSpecAlgo 插件实例。
            name 为 None 时返回 NONE。

        示例::

            >>> SpeculativeAlgorithm.from_string("EAGLE")
            <SpeculativeAlgorithm.EAGLE: ...>
            >>> SpeculativeAlgorithm.from_string(None)
            <SpeculativeAlgorithm.NONE: ...>
            >>> SpeculativeAlgorithm.from_string("MY_PLUGIN_ALGO")
            CustomSpecAlgo('MY_PLUGIN_ALGO')
        """
        if name is None:
            return cls.NONE
        upper = name.upper()
        try:
            return cls[upper]
        except KeyError:
            pass
        # 如果不是内置枚举，尝试从插件注册表中查找
        spec = _get_registered_spec(upper)
        if spec is not None:
            return spec
        raise ValueError(f"未知的投机解码算法名称: {name}")

    @classmethod
    def register(
        cls,
        name: str,
        *,
        supports_overlap: bool = False,
        validate_server_args: Optional[ServerArgsValidator] = None,
        spec_class: Type[CustomSpecAlgo] = CustomSpecAlgo,
    ) -> Callable[[WorkerFactory], WorkerFactory]:
        """注册插件投机算法的装饰器。

        输入:
            name: 算法名称（大写注册）
            supports_overlap: 是否支持 overlap 调度（默认 False，已废弃）
            validate_server_args: 可选的服务参数校验函数
            spec_class: CustomSpecAlgo 子类，用于覆盖 is_*() / create_worker 方法

        输出:
            装饰器函数，接收 factory(server_args) -> worker_class

        示例::

            @SpeculativeAlgorithm.register("MY_SPEC", supports_overlap=False)
            def _factory(server_args):
                return MySpecWorker
        """
        return _register_algorithm(
            name,
            supports_overlap=supports_overlap,
            validate_server_args=validate_server_args,
            spec_class=spec_class,
        )

    def is_some(self) -> bool:
        """是否启用了某种投机算法（非 NONE）。"""
        return self != SpeculativeAlgorithm.NONE

    def is_none(self) -> bool:
        """是否未启用投机解码。"""
        return self == SpeculativeAlgorithm.NONE

    def is_speculative(self) -> bool:
        """是否启用了投机解码（等价于 is_some()）。"""
        return self != SpeculativeAlgorithm.NONE

    def is_eagle(self) -> bool:
        """是否为 EAGLE 家族（含 EAGLE / EAGLE3 / FROZEN_KV_MTP）。

        FIXME(kpham_sgl): 等调度器完全支持 FROZEN_KV_MTP 后，从这里移除它。
        """
        return self in (
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.FROZEN_KV_MTP,
        )

    def is_eagle3(self) -> bool:
        """是否为 EAGLE3。"""
        return self == SpeculativeAlgorithm.EAGLE3

    def is_frozen_kv_mtp(self) -> bool:
        """是否为 Frozen-KV MTP。"""
        return self == SpeculativeAlgorithm.FROZEN_KV_MTP

    def is_dflash(self) -> bool:
        """是否为 DFlash。"""
        return self == SpeculativeAlgorithm.DFLASH

    def is_standalone(self) -> bool:
        """是否为 Standalone（独立草稿模型）。"""
        return self == SpeculativeAlgorithm.STANDALONE

    def is_ngram(self) -> bool:
        """是否为 NGRAM。"""
        return self == SpeculativeAlgorithm.NGRAM

    def supports_target_verify_for_draft(self) -> bool:
        """草稿阶段是否需要目标模型做 target verify。目前仅 DFlash 返回 True。"""
        return self.is_dflash()

    def has_draft_kv(self) -> bool:
        """草稿阶段是否写入 KV 链。

        NGRAM 不写 KV（其树仅存在于 verify 掩码中），因此 per-decode KV 分配
        不需要 per-topk 页对齐；参见 get_alloc_len_per_decode。
        """
        return not self.is_ngram()

    def carries_draft_hidden_states(self) -> bool:
        """PD 分离 prefill→decode 传输时是否携带草稿 hidden states。

        仅 EAGLE 家族需要；STANDALONE 的普通草稿模型忽略它。
        """
        return self.is_eagle()

    def create_future_map(
        self,
        device: torch.device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
    ) -> FutureMap:
        """创建 overlap 模式下的 FutureMap，用于跨阶段传递数据。

        输入:
            device: GPU 设备
            req_to_token_pool: 请求到 token 的映射池
            needs_cpu_seq_lens: 是否需要 CPU 侧的 seq_lens

        输出:
            FutureMap 实例，用于 overlap 调度中 draft/verify/extend 阶段的数据传递
        """
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(device, self, req_to_token_pool, needs_cpu_seq_lens)

    def build_disagg_draft_input(
        self,
        batch: ScheduleBatch,
        server_args: ServerArgs,
        last_tokens_tensor: torch.Tensor,
        future_map: FutureMap,
    ) -> Optional[SpecInput]:
        """在 PD 分离场景下构建草稿输入。

        输入:
            batch: 调度批次
            server_args: 服务参数
            last_tokens_tensor: 最后的 token 张量
            future_map: overlap 模式的 FutureMap

        输出:
            SpecInput 实例或 None（非 EAGLE 算法返回 None）
        """
        if self.is_eagle():
            from sglang.srt.speculative.eagle_disaggregation import (
                build_eagle_disagg_draft_input,
            )

            return build_eagle_disagg_draft_input(
                batch, server_args, last_tokens_tensor, future_map
            )
        return None

    def need_topk(self) -> bool:
        """是否需要 top-k 选择。EAGLE 和 Standalone 需要。"""
        return self.is_eagle() or self.is_standalone()

    def get_num_tokens_per_bs_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:
        """获取 target verify 阶段每个请求的 token 数。

        输入:
            num_draft_tokens: 草稿 token 数
            is_draft_worker: 是否为草稿 worker

        输出:
            每个请求的验证 token 数（当前等于 num_draft_tokens）

        FIXME: 在 forward mode 重构后移除此方法。target verify 本质上是
        固定序列长度的 prefill/extend，完整支持 CUDA Graph。我们也可以
        将它用于非 target verify 的固定长度 prefill 场景。这里暴露接口
        以允许其他用例。
        """
        return num_draft_tokens

    def create_worker(
        self, server_args: ServerArgs
    ) -> Optional[Union[Type[BaseSpecWorker], Type[TpModelWorker], Type[NGRAMWorker]]]:
        """根据当前算法创建对应的 Worker 类。

        输入:
            server_args: 服务参数，包含投机解码相关配置

        输出:
            Worker 类（注意是类本身，不是实例），如 EAGLEWorkerV2 / NGRAMWorker 等

        示例::

            >>> algo = SpeculativeAlgorithm.EAGLE
            >>> worker_cls = algo.create_worker(server_args)
            >>> worker_cls.__name__
            'EAGLEWorkerV2'
        """
        assert (
            not self.is_none()
        ), "无法为 NONE 投机算法创建 worker。"

        if self.is_dflash():
            # V2 worker 同时驱动 overlap 和非 overlap 模式
            # （当 overlap 禁用时调度器同步运行），与 EAGLE 相同。
            from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

            return DFlashWorkerV2

        if self.is_frozen_kv_mtp():
            from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import (
                FrozenKVMTPWorkerV2,
            )

            return FrozenKVMTPWorkerV2

        # EAGLE / EAGLE3 / STANDALONE / MULTI_LAYER 始终使用 V2 worker，
        # 即使 overlap 禁用（调度器同步驱动）。
        if self.is_eagle() and server_args.enable_multi_layer_eagle:
            from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                MultiLayerEagleWorkerV2,
            )

            return MultiLayerEagleWorkerV2

        elif self.is_eagle():
            from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

            return EAGLEWorkerV2
        elif self.is_standalone():
            from sglang.srt.speculative.standalone_worker_v2 import (
                StandaloneWorkerV2,
            )

            return StandaloneWorkerV2
        elif self.is_ngram():
            from sglang.srt.speculative.ngram_worker import NGRAMWorker

            return NGRAMWorker

        raise ValueError("create_worker 中的不可达代码路径。")


class SpecInputType(IntEnum):
    """投机解码输入类型枚举。

    用于在注意力后端断言时区分不同算法的 SpecInput 类型。
    如果所有算法能共享相同的 draft_input 和 verify_input 数据结构，
    可以考虑简化此枚举。
    """

    EAGLE_DRAFT = auto()              # EAGLE 草稿前向输入
    EAGLE_DRAFT_EXTEND = auto()       # EAGLE 草稿 extend（KV 补齐）输入
    EAGLE_VERIFY = auto()             # EAGLE 目标验证输入
    FROZEN_KV_MTP_DRAFT = auto()      # Frozen-KV MTP 草稿输入
    FROZEN_KV_MTP_VERIFY = auto()     # Frozen-KV MTP 验证输入
    DFLASH_DRAFT = auto()             # DFlash 草稿输入
    DFLASH_VERIFY = auto()            # DFlash 验证输入
    NGRAM_VERIFY = auto()             # NGRAM 验证输入


class SpecInput(ABC):
    """投机解码输入的抽象基类。

    所有算法的 draft/verify 输入都继承此类，通过 ``spec_input_type``
    标识自身属于哪个算法的哪个阶段。

    示例::

        >>> class MyVerifyInput(SpecInput):
        ...     def __init__(self, draft_token, custom_mask):
        ...         super().__init__(SpecInputType.EAGLE_VERIFY)
        ...         self.draft_token = draft_token
        ...         self.custom_mask = custom_mask
        ...     def get_spec_adjust_token_coefficient(self):
        ...         return (8, 8)  # (num_tokens_per_req, num_tokens_for_logprob_per_req)
    """

    def __init__(self, spec_input_type: SpecInputType):
        self.spec_input_type = spec_input_type

    # 跨算法阶段守卫。用于注意力后端和 ForwardBatch padding 逻辑中，
    # 根据阶段分发而不硬编码具体算法类（EAGLE / FROZEN_KV_MTP / DFLASH / NGRAM
    # 各有自己的 draft / verify SpecInput 子类）。
    def is_draft_input(self) -> bool:
        """是否为草稿阶段输入（draft 或 draft_extend）。"""
        return self.spec_input_type in {
            SpecInputType.EAGLE_DRAFT,
            SpecInputType.EAGLE_DRAFT_EXTEND,
            SpecInputType.FROZEN_KV_MTP_DRAFT,
            SpecInputType.DFLASH_DRAFT,
        }

    def is_verify_input(self) -> bool:
        """是否为验证阶段输入。"""
        return self.spec_input_type in {
            SpecInputType.EAGLE_VERIFY,
            SpecInputType.FROZEN_KV_MTP_VERIFY,
            SpecInputType.DFLASH_VERIFY,
            SpecInputType.NGRAM_VERIFY,
        }

    @abstractmethod
    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        """获取 token 计数调整系数。

        输出:
            (c1, c2) 元组:
            - c1: 每个请求的 token 数系数（用于 global_num_tokens）
            - c2: 每个请求的 logprob token 数系数（用于 global_num_tokens_for_logprob）

        示例::
            >>> # EAGLE 的 draft_token_num=8 时
            >>> coeff = input.get_spec_adjust_token_coefficient()
            >>> coeff
            (8, 8)
        """
        pass

    def get_spec_adjusted_global_num_tokens(
        self, batch: ScheduleBatch
    ) -> Tuple[List[int], List[int]]:
        """获取调整后的全局 token 数（乘以系数后）。

        输入:
            batch: 调度批次，包含 global_num_tokens 和 global_num_tokens_for_logprob

        输出:
            (global_num_tokens, global_num_tokens_for_logprob) 两个列表，
            分别是原始值乘以 c1 和 c2 的结果。

        示例::

            >>> # 假设 batch.global_num_tokens = [1, 1], 系数为 (8, 8)
            >>> tokens, logprob_tokens = input.get_spec_adjusted_global_num_tokens(batch)
            >>> tokens
            [8, 8]
            >>> logprob_tokens
            [8, 8]
        """
        c1, c2 = self.get_spec_adjust_token_coefficient()
        global_num_tokens = [x * c1 for x in batch.global_num_tokens]
        global_num_tokens_for_logprob = [
            x * c2 for x in batch.global_num_tokens_for_logprob
        ]
        return global_num_tokens, global_num_tokens_for_logprob


def create_dummy_verify_input(
    spec_algorithm: SpeculativeAlgorithm,
    server_args: ServerArgs,
    custom_mask: torch.Tensor,
    num_tokens_per_bs: int,
    is_draft_worker: bool,
) -> Optional[SpecInput]:
    """为 CUDA Graph 捕获创建虚拟验证 SpecInput（按算法分发）。

    CUDA Graph 捕获时需要形状正确的输入但不需要真实数据，此函数按算法类型
    构造一个"空壳" SpecInput，仅包含必要的元信息（如 draft_token_num、
    spec_steps、topk 等），张量字段设为 None。

    输入:
        spec_algorithm: 投机算法枚举
        server_args: 服务参数
        custom_mask: 自定义注意力掩码张量（CUDA Graph 捕获用）
        num_tokens_per_bs: 每个请求的 token 数
        is_draft_worker: 是否为草稿 worker

    输出:
        SpecInput 实例或 None。不同算法返回不同子类:
        - EAGLE/STANDALONE → EagleVerifyInput
        - DFLASH → DFlashVerifyInput
        - NGRAM → NgramVerifyInput
        - 其他 → None

    示例::

        >>> algo = SpeculativeAlgorithm.EAGLE
        >>> dummy = create_dummy_verify_input(
        ...     algo, server_args, custom_mask=torch.ones((8, 8)),
        ...     num_tokens_per_bs=8, is_draft_worker=False)
        >>> type(dummy).__name__
        'EagleVerifyInput'
    """
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    spec_info = None
    if spec_algorithm.is_eagle() or spec_algorithm.is_standalone():
        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        if is_draft_worker:
            raise RuntimeError("草稿 worker 不应创建 verify 输入。")
        else:
            spec_info = EagleVerifyInput(
                draft_token=None,
                custom_mask=custom_mask,
                positions=None,
                retrieve_index=None,
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                retrieve_cum_len=None,
                spec_steps=server_args.speculative_num_steps,
                topk=server_args.speculative_eagle_topk,
                draft_token_num=server_args.speculative_num_draft_tokens,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                seq_lens_sum=None,
                seq_lens_cpu=None,
            )
    elif spec_algorithm.is_dflash():
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        # 虚拟预热只需形状元数据；不强制使用自定义掩码模式。
        spec_info = DFlashVerifyInput(
            draft_token=None,
            positions=None,
            draft_token_num=server_args.speculative_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=(
                CaptureHiddenMode.NULL if is_draft_worker else CaptureHiddenMode.FULL
            ),
        )

    elif spec_algorithm.is_ngram():
        from sglang.srt.speculative.ngram_info import NgramVerifyInput

        spec_info = NgramVerifyInput(
            draft_token=None,
            custom_mask=custom_mask,
            positions=None,
            retrieve_index=None,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            draft_token_num=num_tokens_per_bs,
        )
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

    return spec_info
