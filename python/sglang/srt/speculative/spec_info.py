from __future__ import annotations

import warnings
from abc import ABC
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type, Union

import torch

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.runtime_context import get_spec as get_spec_config
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
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


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

    DFLASH = auto()
    DSPARK = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    FROZEN_KV_MTP = auto()
    STANDALONE = auto()
    NGRAM = auto()
    NONE = auto()

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

    def is_dspark(self) -> bool:
        return self == SpeculativeAlgorithm.DSPARK

    def is_dflash_family(self) -> bool:
        return self.is_dflash() or self.is_dspark()

    def is_standalone(self) -> bool:
        """是否为 Standalone（独立草稿模型）。"""
        return self == SpeculativeAlgorithm.STANDALONE

    def is_ngram(self) -> bool:
        """是否为 NGRAM。"""
        return self == SpeculativeAlgorithm.NGRAM

    def supports_target_verify_for_draft(self) -> bool:
        return self.is_dflash_family()

    def supports_ragged_verify(self) -> bool:
        """Whether this algorithm's verify step may carry a RaggedVerifyLayout
        (per-request verify lengths); gates the token-bucket-keyed verify
        graphs in the decode cuda graph runner."""
        return self.is_dspark()

    def supports_grammar_overlap(self) -> bool:
        # Whether the worker advances the grammar FSM inside verify() (via the
        # scheduler's grammar barrier), letting spec + grammar decode overlap.
        # Needs a GPU draft phase to hide the grammar CPU work under: NGRAM drafts
        # from a host corpus lookup, so it stays synchronous by design.
        # STANDALONE inherits the EAGLE V2 worker's verify path, barrier included.
        return self.is_eagle() or self.is_standalone() or self.is_dflash_family()

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
        needs_confidence_relay: bool = False,
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

        return FutureMap(
            device,
            self,
            req_to_token_pool,
            needs_cpu_seq_lens,
            needs_confidence_relay,
        )

    def build_disagg_draft_input(
        self,
        batch: ScheduleBatch,
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

            return build_eagle_disagg_draft_input(batch, last_tokens_tensor, future_map)
        if self.is_dspark():
            from sglang.srt.speculative.dspark_disaggregation import (
                build_dspark_disagg_draft_input,
            )

            return build_dspark_disagg_draft_input(
                batch, last_tokens_tensor, future_map
            )
        return None

    def need_topk(self) -> bool:
        """是否需要 top-k 选择。EAGLE 和 Standalone 需要。"""
        return self.is_eagle() or self.is_standalone()

    def handle_server_args(self, server_args: ServerArgs) -> None:
        """Hook for per-algorithm server args mutation.

        In-place updated.
        """
        from sglang.srt.arg_groups.speculative_hook import (
            _handle_dflash,
            _handle_dspark,
            _handle_eagle_family,
            _handle_frozen_kv_mtp,
            _handle_ngram,
        )

        # Validate for every algorithm at startup: the metrics paths read the
        # ragged-verify mode env and must not be where a typo'd value raises.
        from sglang.srt.speculative.ragged_verify import read_ragged_verify_mode

        read_ragged_verify_mode()

        if self.is_dflash():
            _handle_dflash(server_args)
        elif self.is_dspark():
            _handle_dspark(server_args)
        elif self.is_frozen_kv_mtp():
            _handle_frozen_kv_mtp(server_args)
        elif self.is_eagle() or self.is_standalone():
            _handle_eagle_family(server_args)
        elif self.is_ngram():
            _handle_ngram(server_args)

    def get_num_tokens_per_req_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:
        # FIXME: Remove this after the forward mode refactor. Target verify is
        # essentially a fixed sequence length prefill/extend with full cuda
        # graph support. We can use it for target verify, or we can use it for
        # other cases which is not target verify but fixed length prefill.
        # Here, we expose this interface to allow the other use cases.
        if self.is_dspark() and is_draft_worker:
            return num_draft_tokens - 1
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

    def create_worker(
        self, server_args: ServerArgs
    ) -> Optional[Union[Type[BaseSpecWorker], Type[TpModelWorker], Type[NGRAMWorker]]]:

        cfg = resolving_view(server_args)
        assert (
            not self.is_none()
        ), "无法为 NONE 投机算法创建 worker。"

        if self.is_dflash():
            # V2 worker 同时驱动 overlap 和非 overlap 模式
            # （当 overlap 禁用时调度器同步运行），与 EAGLE 相同。
            from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

            return DFlashWorkerV2

        if self.is_dspark():
            from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
                DSparkWorkerV2,
            )

            return DSparkWorkerV2

        if self.is_frozen_kv_mtp():
            from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import (
                FrozenKVMTPWorkerV2,
            )

            return FrozenKVMTPWorkerV2

        # EAGLE / EAGLE3 / STANDALONE / MULTI_LAYER always use the V2 worker,
        # even with overlap disabled (scheduler drives it synchronously).
        if self.is_eagle() and cfg.enable_multi_layer_eagle:
            from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                MultiLayerEagleWorkerV2,
            )

            return MultiLayerEagleWorkerV2

        elif self.is_eagle():
            from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

            return EAGLEWorkerV2
        elif self.is_standalone():
            from sglang.srt.speculative.standalone_worker_v2 import StandaloneWorkerV2

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
    # Per-request verify lengths for the ragged-verify graphs (see
    # sglang.srt.speculative.ragged_verify); verify inputs of algorithms with
    # supports_ragged_verify() override it per step. Must stay a class-level
    # default, not an __init__ assignment: dataclass subclasses declare it as
    # a field and run __post_init__ -> super().__init__ *after* field
    # assignment, so an init-time default would clobber the passed layout.
    ragged_verify_layout: Optional[RaggedVerifyLayout] = None

    # Uniform per-request token width of this forward (and its logits-row
    # counterpart). Doubles as the DP-attention global_num_tokens multiplier
    # (ragged forwards carry 1 there). -1 = not set by this flow.
    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = -1

    # DSA MTP IndexShare seed relay. Class-level defaults (same rationale as
    # ragged_verify_layout) so scheduler/relay/attention code reads them
    # uniformly on any SpecInput; only the EAGLE-family inputs override them.
    dsa_topk_indices: Optional[torch.Tensor] = None
    future_dsa_topk_indices_available: bool = False
    dsa_seed_topk_capture: Optional[torch.Tensor] = None

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


def spec_scale_global_num_tokens(
    spec_info: SpecInput,
    global_num_tokens: List[int],
    global_num_tokens_for_logprob: List[int],
) -> Tuple[List[int], List[int]]:
    """Scale the raw per-rank sync values (request counts on decode-family
    rounds) into this forward's token units using the spec input's uniform
    per-request widths."""
    return (
        [x * spec_info.num_tokens_per_req for x in global_num_tokens],
        [
            x * spec_info.num_tokens_for_logprob_per_req
            for x in global_num_tokens_for_logprob
        ],
    )


def create_dummy_verify_input(
    spec_algorithm: SpeculativeAlgorithm,
    custom_mask: torch.Tensor,
    num_tokens_per_req: int,
    is_draft_worker: bool,
) -> Optional[SpecInput]:
    """Dummy verify ``SpecInput`` for CUDA-graph capture (per-algorithm dispatch).

    The tree shape comes from the bags so the dummy matches the step config
    being captured; adaptive spec captures each candidate with that config's
    leaves overridden.
    """
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    spec = get_spec_config()

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
                spec_steps=spec.speculative_num_steps,
                topk=spec.speculative_eagle_topk,
                draft_token_num=spec.speculative_num_draft_tokens,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                seq_lens_sum=None,
                seq_lens_cpu=None,
            )
    elif spec_algorithm.is_dflash_family():
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        # 虚拟预热只需形状元数据；不强制使用自定义掩码模式。
        spec_info = DFlashVerifyInput(
            draft_token=None,
            positions=None,
            draft_token_num=spec.speculative_num_draft_tokens,
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
            draft_token_num=num_tokens_per_req,
        )
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

    return spec_info
