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

from __future__ import annotations

import logging
import math
import time
from abc import ABC
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import einops
import torch
import torch.distributed

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_EXPERT_DISPATCH,
    ExpertDispatchCollector,
    resolve_collector_class,
)
from sglang.srt.runtime_context import get_device as get_device_namespace
from sglang.srt.runtime_context import (
    get_exec,
    get_schedule,
    logs_expert_balancedness_to_server_log,
    reports_expert_balancedness,
)
from sglang.srt.utils import Withable, get_device, get_int_env_var

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

logger = logging.getLogger(__name__)

# --------------------------------------- Entrypoint -----------------------------------------

_OutputMode = Literal["file", "object"]
EPLB_BALANCEDNESS_WINDOW_SIZES = (10, 100, 1000)


@dataclass
class ExpertDistributionMetrics:
    forward_pass_id: int
    eplb_balancedness: torch.Tensor
    gpu_physical_count_sum: Optional[torch.Tensor]
    reset_server_log_history: bool

    def map_device_tensors(self, fn):
        # Device-tensor fields only; caller injects the copy+safety primitive
        # (see GenerationBatchResult.copy_to_cpu).
        self.eplb_balancedness = fn(self.eplb_balancedness)
        if self.gpu_physical_count_sum is not None:
            self.gpu_physical_count_sum = fn(self.gpu_physical_count_sum)


class ExpertDistributionRecorder(ABC):
    """Global expert distribution recording"""

    @staticmethod
    def init_new(
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
    ):
        if get_exec().moe.expert_distribution_recorder_mode is not None:
            assert (
                expert_location_metadata is not None
            ), "ExpertLocationMetadata is required for expert distribution recording. One possible"
            "reason is that you are using a model that does not support expert distribution"
            "recording. Try setting `get_model_config_for_expert_location` in your model."
            return _ExpertDistributionRecorderReal(expert_location_metadata, rank)
        else:
            return _ExpertDistributionRecorderNoop()

    @contextmanager
    def with_current_layer(self, layer_idx):
        yield

    @contextmanager
    def with_debug_name(self, debug_name):
        yield

    @contextmanager
    def disable_this_region(self):
        yield

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int, forward_batch: ForwardBatch):
        yield {}

    def on_select_experts(self, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(
        self,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        pass

    def on_deepep_dispatch_low_latency(
        self, local_physical_count_of_layer: torch.Tensor
    ):
        pass

    def start_record(self):
        self._on_not_implemented()

    def stop_record(self):
        self._on_not_implemented()

    def dump_record(self, output_mode: _OutputMode = "file"):
        self._on_not_implemented()

    @property
    def recording(self):
        return False

    def _on_not_implemented(self):
        raise Exception(
            "Please set ServerArgs.expert_distribution_recorder_mode to use ExpertDistributionRecorder."
        )


class _ExpertDistributionRecorderNoop(ExpertDistributionRecorder):
    pass


class _ExpertDistributionRecorderReal(ExpertDistributionRecorder):
    """专家分布记录器的真实（非空）实现，负责在运行时采集各专家的负载分布。

    整体机制：
    - 通过三个 hook（on_select_experts / on_deepep_dispatch_normal /
      on_deepep_dispatch_low_latency）在 MoE 路由、DeepEP 分发等关键路径上埋点采集。
    - 每次 forward pass 内由 _SinglePassGatherer 收集当次数据，结束后汇总到 _Accumulator。
    - 通过 with_forward_pass / with_current_layer 等上下文管理器维护当前所处的
      forward_pass_id / 层 id / 调试名，供 hook 定位数据归属。
    - 支持 start/stop 控制记录开关，以及 dump 导出（并重置）。
    """

    def __init__(
        self,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
    ):
        self._expert_location_metadata = expert_location_metadata

        # 是否正在记录（开关）
        self._recording = False
        # 是否整体禁用（用于 disable_this_region 临时屏蔽）
        self._disable_all = False
        self._current_forward_pass_id = Withable()
        self._current_layer_idx = Withable()
        self._current_debug_name = Withable()
        self._accumulator = _Accumulator.init_new(expert_location_metadata, rank)
        self._single_pass_gatherers = {
            k: _SinglePassGatherer.init_new(expert_location_metadata, rank)
            for k in self._accumulator.get_single_pass_gatherer_keys()
        }

        if reports_expert_balancedness():
            logger.info(
                "ExpertDistributionRecorder auto start record since "
                f"expert_balancedness_report_mode={get_exec().moe.expert_balancedness_report_mode}"
            )
            self.start_record()

    def with_current_layer(self, layer_idx):
        """上下文管理器：在进入某层计算时记录当前层 id，退出时自动恢复。

        示例：
            with recorder.with_current_layer(3):
                ...  # 此区域内 hook 采集的数据会归属到第 3 层
        """
        return self._current_layer_idx.with_value(layer_idx)

    def with_debug_name(self, debug_name):
        """上下文管理器：设置当前调试名，用于选择对应的单次采集器。"""
        return self._current_debug_name.with_value(debug_name)

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int, forward_batch: ForwardBatch):
        """上下文管理器：包住一次完整的 forward pass，自动处理开始/结束钩子。

        进入时记录 forward_pass_id 并重置各采集器；退出时把本次采集数据汇总进累加器。

        示例：
            with recorder.with_forward_pass(42, forward_batch) as outputs:
                ...  # 运行模型 forward；hook 采集的数据归属到 pass 42
            # 退出后 outputs 中可能含本次 pass 产生的 metrics
        """
        outputs = {}
        with self._current_forward_pass_id.with_value(forward_pass_id):
            self._on_forward_pass_start(forward_batch)
            try:
                yield outputs
            finally:
                self._on_forward_pass_end(forward_pass_id, outputs)

    @contextmanager
    def disable_this_region(self):
        """上下文管理器：临时禁用记录（区域内的 hook 全部不采集）。

        退出时恢复之前的启用状态（支持嵌套）。
        示例：
            with recorder.disable_this_region():
                ...  # 例如预热/捕获图等不希望计入统计的区域
        """
        previous_disable_all = self._disable_all
        self._disable_all = True
        try:
            yield
        finally:
            self._disable_all = previous_disable_all

    def _on_forward_pass_start(self, forward_batch: ForwardBatch):
        """forward pass 开始回调：重置各单次采集器并通知其本次 forward 开始。

        未处于记录状态（_recording=False）时直接返回，不做任何采集。
        """
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            gatherer.reset()
            gatherer.on_forward_pass_start(forward_batch)

    def _on_forward_pass_end(self, forward_pass_id: int, outputs: Dict[str, Any]):
        """forward pass 结束回调：从各采集器 collect 本次数据并 append 到累加器。

        outputs 用于回传本次 pass 可能产生的指标（如均衡度）。
        """
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            single_pass_data = gatherer.collect()
            self._accumulator.append(
                forward_pass_id, gatherer_key, single_pass_data, outputs
            )

    def on_select_experts(self, topk_ids: torch.Tensor):
        """采集钩子：在 topk 专家选择处埋点，记录每个 token 选中的专家 id。

        输入：topk_ids，形状 (num_tokens, top_k)，元素为选中的（物理）专家 id。
        示例：topk_ids=[[0,3],[0,1]] 表示 token0 选中专家 0/3，token1 选中 0/1。
        """
        self._on_hook("on_select_experts", topk_ids=topk_ids)

    def on_deepep_dispatch_normal(
        self,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        """采集钩子：DeepEP normal（高吞吐）分发路径上埋点。

        输入：
        - local_physical_count_of_layer: 本层各本地物理专家的 token 计数。
        - num_tokens_per_rank / num_tokens_per_rdma_rank / num_tokens_per_expert:
          DeepEP 分发产生的各维度 token 数量统计。
        """
        self._on_hook(
            "on_deepep_dispatch_normal",
            local_physical_count_of_layer=local_physical_count_of_layer,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

    def on_deepep_dispatch_low_latency(
        self, local_physical_count_of_layer: torch.Tensor
    ):
        """采集钩子：DeepEP low-latency（低延迟）分发路径上埋点。

        输入：local_physical_count_of_layer，本层各本地物理专家的 token 计数张量。
        """
        self._on_hook(
            "on_deepep_dispatch_low_latency",
            local_physical_count_of_layer=local_physical_count_of_layer,
        )

    def _on_hook(self, hook_name: str, **kwargs):
        """钩子统一分发入口：根据当前调试名选到对应采集器，并调用其同名方法。

        会自动带上当前层 id（layer_idx）。以下两种情况直接跳过（不采集）：
        - 整体被禁用（_disable_all=True）；
        - 既不在记录状态、当前 stream 也未在做 CUDA graph 捕获时。
        """
        if self._disable_all:
            return
        if not (
            self._recording or torch.get_device_module().is_current_stream_capturing()
        ):
            return
        gatherer = self._single_pass_gatherers[
            self._accumulator.get_single_pass_gatherer_key(
                self._current_debug_name.value
            )
        ]
        getattr(gatherer, hook_name)(layer_idx=self._current_layer_idx.value, **kwargs)

    def _reset(self):
        """重置专家分布记录器：清空各采集器与累加器的状态。"""
        logger.info("Resetting ExpertDistributionRecorder...")
        assert (
            self._current_layer_idx.value is None
        ), f"{self._current_layer_idx.value=}"
        for gatherer in self._single_pass_gatherers.values():
            gatherer.reset()
        self._accumulator.reset()

    def start_record(self):
        """开始记录专家分布（会先重置状态）。"""
        if self._recording:
            logger.warning(
                "SGLang server is already recording expert ids. Did you forget to dump the expert ids recorded so far by sending requests to the `/stop_expert_distribution_record` and `/dump_expert_distribution_record` endpoints?"
            )
        self._reset()
        self._recording = True

    def stop_record(self):
        """停止记录专家分布。"""
        if not self._recording:
            logger.warning(
                "SGLang server has not been recording expert ids. Did you forget to start recording by sending request to the `/start_expert_distribution_record` endpoint?"
            )
        self._recording = False

    def dump_record(self, output_mode: _OutputMode = "file"):
        """导出专家分布记录，并在导出后重置记录器。

        输入：output_mode，"file" 写入文件，"object" 直接返回结果 dict。
        返回：由累加器 dump 出的结果（object 模式下为含 logical_count 等字段的 dict）。
        """
        output = self._accumulator.dump(output_mode=output_mode)
        self._reset()
        return output

    @property
    def recording(self):
        return self._recording


def get_global_expert_distribution_recorder():
    from sglang.srt.runtime_context import get_resources

    resources = get_resources()
    if resources.expert_distribution_recorder is None:
        # Call sites expect a recorder unconditionally; default to the noop.
        resources.expert_distribution_recorder = _ExpertDistributionRecorderNoop()
    return resources.expert_distribution_recorder


def set_global_expert_distribution_recorder(value):
    from sglang.srt.runtime_context import get_resources

    get_resources().expert_distribution_recorder = value


# --------------------------------------- SinglePassGatherer -----------------------------------------


class _SinglePassGatherer(ABC):
    """单次 forward pass 的专家负载收集器（抽象基类）。

    定位
    ----
    在 EPLB 的负载采集链路中，``ExpertDistributionRecorder``（记录器）负责跨多次
    forward pass 的累积与 dump，而本类只负责"一次 forward pass 之内"的负载收集：
    模型前向过程中通过若干 hook（``on_select_experts`` / ``on_deepep_dispatch_*``）
    把各层、各专家被路由到的 token 计数喂进来，本次 forward 结束后由记录器调用
    ``collect()`` 取走结果，再调用 ``reset()`` 复位以复用同一实例。

    为什么要有多个子类
    ------------------
    不同的 MoE 后端 / 采集精度，其"负载信息从哪个 hook、以什么形态到来"各不相同，
    因此派生出多个子类（见 ``init_new`` 的分派）：

    - ``_DetailSinglePassGatherer``：per-token 逐 token 记录 topk_ids，信息最细（可离线复盘）。
    - ``_SelectExpertsSinglePassGatherer``：在 topk 选择处统计（全局物理专家计数）。
    - ``_DeepepNormalSinglePassGatherer``：DeepEP normal 模式，从 dispatch 处拿近似统计（CPU 累加）。
    - ``_DeepepLowLatencySinglePassGatherer``：DeepEP low-latency 模式（GPU 上按本地物理专家累加）。

    统一输出
    --------
    所有子类的 ``collect()`` 都会返回含 ``global_physical_count``
    （形状 ``[num_layers, num_physical_experts]``，各物理专家被路由到的 token 数）的 dict，
    使上层记录器无需关心具体后端差异。
    """

    @staticmethod
    def init_new(
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
    ) -> _SinglePassGatherer:
        if get_exec().moe.expert_distribution_recorder_mode == "per_token":
            return _DetailSinglePassGatherer(expert_location_metadata, rank)

        if get_exec().moe.moe_a2a_backend == "mori":
            return _DeepepLowLatencySinglePassGatherer(expert_location_metadata, rank)

        if get_exec().moe.expert_distribution_recorder_mode == "stat_approx":
            if get_exec().moe.moe_a2a_backend != "none" and (
                get_exec().moe.deepep_mode == "normal"
            ):
                return _DeepepNormalSinglePassGatherer(expert_location_metadata, rank)
            else:
                raise NotImplementedError

        if get_exec().moe.moe_a2a_backend == "deepep":
            if get_exec().moe.deepep_mode == "normal":
                return _SelectExpertsSinglePassGatherer(expert_location_metadata, rank)
            elif get_exec().moe.deepep_mode == "low_latency":
                return _DeepepLowLatencySinglePassGatherer(
                    expert_location_metadata,
                    rank,
                    elastic_ep_enabled=get_exec().moe.elastic_ep_backend is not None,
                )
            else:
                raise NotImplementedError

        # Non-DeepEP a2a backends (flashinfer / nixl / mooncake / megamoe) and
        # no-a2a path dispatch through the standard topk select_experts.
        return _SelectExpertsSinglePassGatherer(expert_location_metadata, rank)

    def __init__(self, expert_location_metadata: ExpertLocationMetadata, rank: int):
        """保存专家位置元数据与当前 rank。

        参数
        ----
        expert_location_metadata: 专家位置元数据，提供 num_layers / num_physical_experts /
            num_local_physical_experts 等维度信息，供各子类分配缓冲区与换算索引。
        rank: 当前进程的 EP rank，用于把"本地物理专家计数"映射到"全局物理专家计数"中的偏移。
        """
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank

    def on_forward_pass_start(self, forward_batch: ForwardBatch):
        """forward pass 开始时的 hook（默认空实现）。

        仅需要逐 token 元信息的子类（``_DetailSinglePassGatherer``）会重写它以记录
        input_ids / positions 等；其余子类无需该信息。
        """
        pass

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        """topk 专家选择处的 hook（默认空实现）。

        参数
        ----
        layer_idx: 当前 MoE 层的下标。
        topk_ids: 该层每个 token 选中的 topk 专家 id，形状约为 ``[num_tokens, top_k]``；
            其中 ``-1`` 表示无效/填充位。
        """
        pass

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        """DeepEP normal 模式 dispatch 处的 hook（默认空实现）。

        参数
        ----
        layer_idx: 当前层下标。
        local_physical_count_of_layer: 本层各"本地物理专家"收到的 token 计数列表。
        num_tokens_per_rank / num_tokens_per_rdma_rank / num_tokens_per_expert:
            dispatch 过程中的分发统计（分别按 rank / RDMA rank / 专家维度），
            仅详细收集器会保存，用于离线分析。
        """
        pass

    def on_deepep_dispatch_low_latency(
        self, layer_idx: int, local_physical_count_of_layer: torch.Tensor
    ):
        """DeepEP low-latency 模式 dispatch 处的 hook（默认空实现）。

        参数
        ----
        layer_idx: 当前层下标。
        local_physical_count_of_layer: 本层各本地物理专家收到的 token 计数（张量）。
        """
        pass

    def reset(self):
        """复位内部缓冲区，以便同一实例用于下一次 forward pass。子类必须实现。"""
        raise NotImplementedError

    def collect(self) -> Dict:
        """汇总并返回本次 forward pass 的采集结果（含 ``global_physical_count``）。子类必须实现。"""
        raise NotImplementedError


class _DetailSinglePassGatherer(_SinglePassGatherer):
    """逐 token 详细收集器（per_token 模式）。

    除了统计各物理专家的 token 数，还会完整保留每个 token 的 topk_ids、
    输入元信息（input_ids/positions 等）以及 dispatch 统计，信息最细，
    主要用于离线分析/复盘（数据量较大，不适合高频在线重平衡）。
    """

    # DeepSeek V3 的 topk 取值为 8；后续应该做成可配置/通用的
    _TOP_K_NUM = 8

    def __init__(
        self,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
    ):
        super().__init__(expert_location_metadata, rank)
        self._metadata: Optional[Dict[str, Any]] = None
        self._topk_ids_of_layer = torch.zeros(
            (
                expert_location_metadata.num_layers,
                # TODO determine the max number
                get_schedule().chunked_prefill_size * 8,
                self._TOP_K_NUM,
            ),
            dtype=torch.int32,
            device=get_device_namespace().device,
        )
        self._misc_objects: List[Dict[str, Any]] = []
        # 详细收集器暂不支持 TBO（two-batch-overlap，双 batch 重叠）
        assert (
            not get_exec().overlap.enable_two_batch_overlap
        ), "DetailSinglePassGatherer does not support TBO yet"
        # TODO 需断言共享专家融合（shared experts fusion）已关闭，否则数据会出错

    def on_forward_pass_start(self, forward_batch: ForwardBatch):
        """记录本次 forward 的输入元信息（input_ids / positions / 序列长 / forward 模式）。

        断言 ``self._metadata is None`` 确保一次 forward 只记录一次（未被 reset 前不重入）。
        """
        assert self._metadata is None
        self._metadata = dict(
            # TODO pr-chain
            # rids=forward_batch.rids,
            input_ids=forward_batch.input_ids.cpu().tolist(),
            positions=forward_batch.positions.cpu().tolist(),
            extend_seq_lens=forward_batch.extend_seq_lens_cpu,
            forward_mode=forward_batch.forward_mode.value,
        )

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        """把该层每个 token 的 topk_ids 原样写入预分配的缓冲区对应位置。

        输入输出示例
        ------------
        >>> # layer_idx=0, topk_ids 形状 [3 个 token, top_k=2]
        >>> topk_ids = torch.tensor([[1, 5], [2, 5], [1, 3]])
        >>> gatherer.on_select_experts(0, topk_ids)
        >>> # 之后 self._topk_ids_of_layer[0, :3, :2] == topk_ids
        """
        self._topk_ids_of_layer[layer_idx, : topk_ids.shape[0], : topk_ids.shape[1]] = (
            topk_ids
        )

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        """把本层 dispatch 的各维度统计（转到 CPU list）追加到 ``_misc_objects`` 中保存。"""
        self._misc_objects.append(
            dict(
                layer_id=layer_idx,
                num_tokens_per_rank=num_tokens_per_rank.cpu().tolist(),
                num_tokens_per_rdma_rank=(
                    num_tokens_per_rdma_rank.cpu().tolist()
                    if num_tokens_per_rdma_rank is not None
                    else None
                ),
                num_tokens_per_expert=num_tokens_per_expert.cpu().tolist(),
            )
        )

    def reset(self):
        """复位：topk 缓冲区填 -1（无效位）、清空 misc 列表、元信息置 None。"""
        self._topk_ids_of_layer[...] = -1
        self._misc_objects.clear()
        self._metadata = None

    def collect(self) -> Dict:
        """汇总本次 forward 的详细结果。

        会把逐 token 的 topk_ids 聚合成各物理专家的计数 ``global_physical_count``，
        并连同输入元信息、逐 token topk_ids、dispatch 统计一并返回。

        返回 dict 包含：
          - input_ids / positions / extend_seq_lens / forward_mode（来自元信息）
          - topk_ids_of_layer: ``[num_layers, num_tokens, top_k]`` 逐 token 的选专家结果
          - misc_objects: 各层 dispatch 统计
          - global_physical_count: ``[num_layers, num_physical_experts]`` 各物理专家 token 数
        """
        num_tokens = len(self._metadata["input_ids"])

        global_physical_count = _convert_per_token_to_global_physical_count(
            num_tokens,
            num_layers=self._expert_location_metadata.num_layers,
            num_physical_experts=self._expert_location_metadata.num_physical_experts,
            _topk_ids_of_layer=self._topk_ids_of_layer,
        )

        return dict(
            **self._metadata,
            topk_ids_of_layer=self._topk_ids_of_layer[:, :num_tokens, :].clone().cpu(),
            misc_objects=self._misc_objects,
            global_physical_count=global_physical_count,
        )


class _LayerBasedCpuSinglePassGatherer(_SinglePassGatherer):
    """基于层的 CPU 收集器（抽象中间类）。

    用一个 ``{layer_idx: List[int]}`` 字典在 CPU 上累加各层的逐专家计数；
    同一层多次写入时按位相加（适用于 DeepEP normal 等以 Python list 形式上报的场景）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # {层下标: 该层各专家的累加计数列表}
        self._objects_of_layer = {}

    def _on_layer_data(self, layer_idx: int, objects: List[int]):
        """向指定层写入/累加一组计数。

        若该层已有数据则逐位相加（``_list_sum``），否则直接写入。

        输入输出示例
        ------------
        >>> g._on_layer_data(0, [1, 2, 3])   # 首次写入
        >>> g._on_layer_data(0, [4, 0, 1])   # 逐位相加
        >>> g._objects_of_layer[0]
        [5, 2, 4]
        """
        assert 0 <= layer_idx < self._expert_location_metadata.num_layers
        if layer_idx in self._objects_of_layer:
            self._objects_of_layer[layer_idx] = _list_sum(
                self._objects_of_layer[layer_idx], objects
            )
        else:
            self._objects_of_layer[layer_idx] = objects

    def reset(self):
        """复位：清空逐层计数字典。"""
        self._objects_of_layer.clear()

    def _collect_objects(self, pad_len: int) -> torch.Tensor:
        """把逐层计数字典汇总成 ``[num_layers, pad_len]`` 的张量；缺失的层用全 0 补齐。

        输入输出示例
        ------------
        >>> # 共 3 层，仅第 1 层有数据 [2, 5]，pad_len=2
        >>> g._collect_objects(pad_len=2)
        tensor([[0, 0], [2, 5], [0, 0]])
        """
        data = [
            self._objects_of_layer.get(layer_index) or ([0] * pad_len)
            for layer_index in range(self._expert_location_metadata.num_layers)
        ]
        return torch.tensor(data)


def _list_sum(a: List, b: List) -> List:
    """两个等长列表逐位相加（长度不一致会因 strict=True 报错）。

    >>> _list_sum([1, 2, 3], [4, 0, 1])
    [5, 2, 4]
    """
    return [x + y for x, y in zip(a, b, strict=True)]


class _LayerBasedGpuSinglePassGatherer(_SinglePassGatherer):
    """基于层的 GPU 收集器（抽象中间类）。

    用一个预分配的 GPU 张量 ``_data`` 直接在设备上累加计数，避免频繁的 CPU 交互。

    ``enable_global_physical_experts`` 控制 ``_data`` 的列宽：
      - True：直接按全局物理专家数 ``num_physical_experts`` 开辟（collect 时无需换算）；
      - False：按本地物理专家数 ``num_local_physical_experts`` 开辟，collect 时再根据 rank
        映射到全局计数。
    """

    def __init__(self, *args, enable_global_physical_experts: bool, **kwargs):
        super().__init__(*args, **kwargs)

        device = get_device()

        self._enable_global_physical_experts = enable_global_physical_experts
        self._data = torch.zeros(
            (
                self._expert_location_metadata.num_layers,
                (
                    self._expert_location_metadata.num_physical_experts
                    if enable_global_physical_experts
                    else self._expert_location_metadata.num_local_physical_experts
                ),
            ),
            dtype=torch.int,
            device=device,
        )

    def reset(self):
        """复位：把 GPU 计数张量清零。"""
        self._data[...] = 0

    def collect(self) -> Dict:
        """返回 ``{"global_physical_count": [num_layers, num_physical_experts]}``。

        若本来就是全局计数则直接返回；否则把本地计数按 rank 映射到全局位置。
        """
        if self._enable_global_physical_experts:
            global_physical_count = self._data
        else:
            # 若成为瓶颈可以优化
            global_physical_count = _convert_local_to_global_physical_count(
                self._data,
                rank=self._rank,
                num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
                num_physical_experts=self._expert_location_metadata.num_physical_experts,
            )

        return dict(global_physical_count=global_physical_count)


class _SelectExpertsSinglePassGatherer(_LayerBasedGpuSinglePassGatherer):
    """在 topk 选择处统计的收集器（默认兜底实现）。

    直接根据每层选中的 topk_ids 累加各全局物理专家的计数（``enable_global_physical_experts=True``），
        适用于未启用专门 a2a 后端的常规场景。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, enable_global_physical_experts=True)

    # 可优化（例如算子融合 / 编译）
    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        """把本层 topk_ids 展平，排除 -1 无效位后，用 scatter_add 累加到各专家计数。

        输入输出示例
        ------------
        >>> # num_physical_experts=6, layer_idx=0
        >>> topk_ids = torch.tensor([[1, 5], [1, 3], [-1, 5]])  # 含一个无效位 -1
        >>> g.on_select_experts(0, topk_ids)
        >>> # 专家1 被选 2 次、专家5 选 2 次、专家3 选 1 次，-1 不计
        >>> # => self._data[0] == tensor([0, 2, 0, 1, 0, 2])
        """
        topk_ids = topk_ids.flatten()
        mask = topk_ids != -1
        self._data[layer_idx, :].scatter_add_(
            dim=0, index=topk_ids.masked_fill(~mask, 0).long(), src=mask.int()
        )


class _DeepepNormalSinglePassGatherer(_LayerBasedCpuSinglePassGatherer):
    """DeepEP normal 模式收集器。

    从 DeepEP normal dispatch 处拿到的是本地物理专家计数（Python list），在 CPU 上逐层累加；
    注意这是近似统计，小 batch 时精度有限（见构造时的提示日志）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.distributed.get_rank() == 0:
            logger.info(
                "DeepepNormalSinglePassGatherer gathers approximate statistics. "
                "If used with small batch size, consider using expert_distribution_recorder_mode=stat."
            )

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        """把本层的本地物理专家计数（list）累加到对应层。"""
        assert isinstance(local_physical_count_of_layer, list)
        self._on_layer_data(layer_idx, local_physical_count_of_layer)

    def collect(self) -> Dict:
        """汇总逐层本地计数并按 rank 映射为全局计数，返回 ``{"global_physical_count": ...}``。"""
        local_physical_count = super()._collect_objects(
            pad_len=self._expert_location_metadata.num_local_physical_experts
        )
        global_physical_count = _convert_local_to_global_physical_count(
            local_physical_count,
            rank=self._rank,
            num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
            num_physical_experts=self._expert_location_metadata.num_physical_experts,
        )
        return dict(global_physical_count=global_physical_count)


class _DeepepLowLatencySinglePassGatherer(_LayerBasedGpuSinglePassGatherer):
    def __init__(self, *args, elastic_ep_enabled: bool = False, **kwargs):
        super().__init__(*args, **kwargs, enable_global_physical_experts=False)
        self._elastic_ep_enabled = elastic_ep_enabled

    def on_deepep_dispatch_low_latency(
        self, layer_idx: int, local_physical_count_of_layer: torch.Tensor
    ):
        if local_physical_count_of_layer.shape[0] != self._data.shape[1]:
            if not self._elastic_ep_enabled:
                self._data[layer_idx, :] += local_physical_count_of_layer
                return

            n = self._data.shape[1]
            if local_physical_count_of_layer.shape[0] > n:
                local_physical_count_of_layer = local_physical_count_of_layer[:n]
            else:
                local_physical_count_of_layer = torch.nn.functional.pad(
                    local_physical_count_of_layer,
                    (0, n - local_physical_count_of_layer.shape[0]),
                )
        self._data[layer_idx, :] += local_physical_count_of_layer


def _convert_per_token_to_global_physical_count(
    num_tokens: int,
    num_layers: int,
    num_physical_experts: int,
    _topk_ids_of_layer: torch.Tensor,
) -> torch.Tensor:
    topk_ids_layer_major = _topk_ids_of_layer[:, :num_tokens, :].reshape(num_layers, -1)
    mask = topk_ids_layer_major != -1

    index = topk_ids_layer_major.masked_fill(~mask, 0).long()
    src = mask.int()

    ans = torch.zeros(
        (num_layers, num_physical_experts),
        dtype=_topk_ids_of_layer.dtype,
        device=_topk_ids_of_layer.device,
    )
    ans.scatter_add_(dim=1, index=index, src=src)
    return ans


def _convert_local_to_global_physical_count(
    local_physical_count: torch.Tensor,
    rank: int,
    num_local_physical_experts: int,
    num_physical_experts: int,
) -> torch.Tensor:
    dtype = local_physical_count.dtype
    device = local_physical_count.device
    num_layers, _ = local_physical_count.shape

    ans = torch.zeros((num_layers, num_physical_experts), dtype=dtype, device=device)
    ans[
        :, num_local_physical_experts * rank : num_local_physical_experts * (rank + 1)
    ] = local_physical_count
    return ans


# --------------------------------------- Accumulator -----------------------------------------

_SINGLE_PASS_GATHERER_KEY_PRIMARY = "primary"


class _Accumulator(ABC):
    @staticmethod
    def init_new(
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
    ) -> _Accumulator:
        return _Accumulator.get_class()(expert_location_metadata, rank)

    @staticmethod
    def get_class() -> Type[_Accumulator]:
        return {
            "stat": _StatAccumulator,
            "stat_approx": _StatAccumulator,
            "per_pass": _DetailAccumulator,
            "per_token": _DetailAccumulator,
        }[get_exec().moe.expert_distribution_recorder_mode]

    def __init__(
        self,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
    ):
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank

    def get_single_pass_gatherer_keys(self):
        return [_SINGLE_PASS_GATHERER_KEY_PRIMARY]

    def get_single_pass_gatherer_key(self, debug_name: Optional[str]):
        return _SINGLE_PASS_GATHERER_KEY_PRIMARY

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
        outputs: Dict[str, Any],
    ):
        pass

    def reset(self):
        pass

    def dump(self, output_mode: _OutputMode):
        pass


class _UtilizationRateAccumulatorMixin(_Accumulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._enable = reports_expert_balancedness()

        if self._enable:
            self.window_sizes = EPLB_BALANCEDNESS_WINDOW_SIZES
            self._history = _DequeCollection(maxlens=self.window_sizes)
            self._reset_server_log_history = True
            self._rank = torch.distributed.get_rank()
            expert_dispatch_cls = resolve_collector_class(
                STAT_LOGGER_ROLE_EXPERT_DISPATCH,
                ExpertDispatchCollector,
            )
            self._expert_dispatch_collector = expert_dispatch_cls(
                self._expert_location_metadata.ep_size
            )
            self._metric_heatmap_collection_counter = 0

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
        outputs: Dict[str, Any],
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data, outputs)
        if self._enable:
            return self._append_utilization_rate(
                forward_pass_id, single_pass_data["global_physical_count"], outputs
            )

    def reset(self):
        super().reset()
        if self._enable:
            self._history.clear()
            self._reset_server_log_history = True

    def _append_utilization_rate(
        self,
        forward_pass_id: int,
        single_pass_global_physical_count: torch.Tensor,
        outputs: Dict[str, Any],
    ):
        gpu_physical_count = compute_gpu_physical_count(
            single_pass_global_physical_count,
            num_gpu=self._expert_location_metadata.ep_size,
        )
        gpu_physical_count = gpu_physical_count.to(get_device_namespace().device)
        torch.distributed.reduce(
            gpu_physical_count, dst=0, op=torch.distributed.ReduceOp.SUM
        )

        if self._rank == 0:
            self._handle_metric_eplb_heatmap(gpu_physical_count)

            utilization_rate_gpu = torch.mean(
                compute_utilization_rate(gpu_physical_count)
            )
            should_track_history = not math.isclose(
                get_exec().moe.eplb_min_rebalancing_utilization_threshold, 1.0
            )

            should_log = logs_expert_balancedness_to_server_log()
            outputs["metrics"] = ExpertDistributionMetrics(
                forward_pass_id=forward_pass_id,
                eplb_balancedness=utilization_rate_gpu,
                gpu_physical_count_sum=(
                    gpu_physical_count.sum() if should_log else None
                ),
                reset_server_log_history=self._reset_server_log_history,
            )
            self._reset_server_log_history = False

            if should_track_history:
                self._history.append(utilization_rate_gpu.item())

    # TODO refactor
    def _handle_metric_eplb_heatmap(self, gpu_physical_count: torch.Tensor):
        # sglang:eplb_gpu_physical_count metric is disabled if SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL <= 0
        interval = get_int_env_var("SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL", 0)
        if interval > 0 and self._metric_heatmap_collection_counter % interval == 0:
            for layer_idx in range(self._expert_location_metadata.num_layers):
                count_of_layer = (
                    self._expert_dispatch_collector.eplb_gpu_physical_count.labels(
                        layer=str(layer_idx)
                    )
                )
                # Exclude the +Inf bucket.
                assert (
                    self._expert_location_metadata.ep_size
                    == len(count_of_layer._buckets) - 1
                ), f"{self._expert_location_metadata.ep_size=}, {len(count_of_layer._buckets)=}"
                for gpu_rank in range(self._expert_location_metadata.ep_size):
                    count = gpu_physical_count[layer_idx, gpu_rank]
                    if count > 0:
                        count_of_layer._sum.inc(count * gpu_rank)
                        count_of_layer._buckets[gpu_rank].inc(count)
        self._metric_heatmap_collection_counter += 1


class _DequeCollection:
    def __init__(self, maxlens: Sequence[int]):
        self._dequeues = [deque(maxlen=maxlen) for maxlen in maxlens]

    def append(self, value):
        for d in self._dequeues:
            d.append(value)

    def clear(self):
        for d in self._dequeues:
            d.clear()

    def mean(self) -> Dict[int, float]:
        return {d.maxlen: sum(d) / len(d) for d in self._dequeues if len(d) > 0}


class _DetailAccumulator(_UtilizationRateAccumulatorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._records = []

    def get_single_pass_gatherer_keys(self):
        return super().get_single_pass_gatherer_keys()

    def get_single_pass_gatherer_key(self, debug_name: Optional[str]):
        return super().get_single_pass_gatherer_key(debug_name)

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
        outputs: Dict[str, Any],
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data, outputs)

        def _process_object(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().clone()
            return obj

        single_pass_data_processed = {
            k: _process_object(v) for k, v in single_pass_data.items()
        }

        self._records.append(
            dict(
                forward_pass_id=forward_pass_id,
                rank=self._rank,
                gatherer_key=gatherer_key,
                **single_pass_data_processed,
            )
        )

    def reset(self):
        super().reset()
        self._records.clear()

    def dump(self, output_mode: _OutputMode):
        assert output_mode == "file"
        output = dict(
            records=self._records,
            # NOTE: This may change during recording, so here we say it is the "last" one
            last_physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
        )
        _dump_to_file(
            f"expert_distribution_recorder_{time.time()}_{self._rank}.pt", output
        )


class _StatAccumulator(_UtilizationRateAccumulatorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_physical_count_of_buffered_step = _Buffer.init_new(
            item_shape=(
                self._expert_location_metadata.num_layers,
                # Cannot use local_physical_count to support select_experts
                self._expert_location_metadata.num_physical_experts,
            ),
            buffer_size=get_exec().moe.expert_distribution_recorder_buffer_size,
            dtype=torch.int32,
            device=get_device_namespace().device,
        )
        self._first_dump = True

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
        outputs: Dict[str, Any],
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data, outputs)
        # Can optimize if overhead here is large
        self._global_physical_count_of_buffered_step.append(
            single_pass_data["global_physical_count"]
        )

    def reset(self):
        super().reset()
        self._global_physical_count_of_buffered_step.reset()

    def dump(self, output_mode: _OutputMode):
        logical_count_of_buffered_step = _convert_global_physical_count_to_logical_count(
            self._global_physical_count_of_buffered_step.get_all(),
            num_layers=self._expert_location_metadata.num_layers,
            num_logical_experts=self._expert_location_metadata.num_logical_experts,
            physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
        )

        if self._first_dump:
            self._first_dump = False
            torch.get_device_module().empty_cache()

        torch.distributed.all_reduce(
            logical_count_of_buffered_step, op=torch.distributed.ReduceOp.SUM
        )

        output = dict(
            rank=self._rank,
            logical_count=logical_count_of_buffered_step,
            average_utilization_rate_over_window=self._get_global_average_utilization_rate(),
        )

        if output_mode == "file":
            if self._rank == 0:
                _dump_to_file(f"expert_distribution_recorder_{time.time()}.pt", output)
        elif output_mode == "object":
            return output
        else:
            raise NotImplementedError

    def _get_global_average_utilization_rate(self):
        """计算「窗口内平均均衡度」并广播给所有 rank。

        这里的 utilization rate 并非 GPU 硬件利用率，而是各 GPU 之间专家负载的
        均衡程度（balancedness）：越接近 1 表示越均衡，越小表示越不均衡（存在热点 GPU）。
        单次值由 compute_utilization_rate 计算（avg/max），逐次 append 到滑动窗口 _history。

        行为：
        - 若未启用记录，或阈值 eplb_min_rebalancing_utilization_threshold == 1.0（默认），
          直接返回 None，表示不启用“已均衡则跳过重平衡”的判断（即每次都重平衡）。
        - 否则取最大窗口（window_sizes[-1]，即最近 1000 次 forward）内的均值；
          只有 rank 0 持有真实值，再通过 broadcast 同步给其他 rank。

        返回：float（均衡度，0~1）或 None。
        """
        # 未启用，或阈值为 1.0（不做跳过判断）时直接返回 None
        if not self._enable or math.isclose(
            get_exec().moe.eplb_min_rebalancing_utilization_threshold, 1.0
        ):
            return None

        if self._rank == 0:
            # _history.mean() 返回各窗口大小对应的均值；window_sizes = [10, 100, 1000]
            utilization_mean_rates = self._history.mean()
            # 取最大窗口（最近 1000 次 forward pass）的平均均衡度
            window_index = self.window_sizes[-1]
            average_utilization_rate_over_window = (
                utilization_mean_rates[window_index]
                if window_index in utilization_mean_rates
                else 0
            )

            avg_rate_tensor = torch.tensor(
                [average_utilization_rate_over_window],
                dtype=torch.float32,
                device="cuda",
            )
        else:
            avg_rate_tensor = torch.empty(1, dtype=torch.float32, device="cuda")
        torch.distributed.broadcast(avg_rate_tensor, src=0)
        return avg_rate_tensor.item()


def _dump_to_file(name, data):
    save_dir = Path(envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.get())
    path_output = save_dir / name
    logger.info(f"Write expert distribution to {path_output}")
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(path_output))


class _Buffer:
    @staticmethod
    def init_new(item_shape: Tuple, buffer_size: int, dtype, device):
        if buffer_size < 0:
            return _InfiniteBuffer(item_shape, dtype=dtype, device=device)
        else:
            return _CircularBuffer(item_shape, buffer_size, dtype=dtype, device=device)

    def append(self, value: torch.Tensor):
        raise NotImplementedError

    def get_all(self) -> torch.Tensor:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class _CircularBuffer(_Buffer):
    def __init__(self, item_shape: Tuple, buffer_size: int, dtype, device):
        self._buffer = torch.zeros(
            (buffer_size, *item_shape), dtype=dtype, device=device
        )
        self._curr_index = 0

    def append(self, value: torch.Tensor):
        self._buffer[self._curr_index] = value
        self._curr_index = (self._curr_index + 1) % len(self._buffer)

    def get_all(self) -> torch.Tensor:
        return self._buffer

    def reset(self):
        self._buffer[...] = 0


class _InfiniteBuffer(_Buffer):
    def __init__(self, item_shape: Tuple, dtype, device):
        self._item_shape = item_shape
        self._buffer = torch.zeros((128, *item_shape), dtype=dtype, device=device)
        self._size = 0

    def append(self, value: torch.Tensor):
        curr_buffer_size = len(self._buffer)
        dtype = self._buffer.dtype
        device = self._buffer.device

        if self._size == curr_buffer_size:
            new_buffer = torch.zeros(
                (2 * curr_buffer_size, *self._item_shape), dtype=dtype, device=device
            )
            new_buffer[:curr_buffer_size] = self._buffer
            self._buffer = new_buffer

        self._buffer[self._size] = value
        self._size += 1

    def get_all(self) -> torch.Tensor:
        return self._buffer[: self._size]

    def reset(self):
        self._buffer[...] = 0
        self._size = 0


def _convert_global_physical_count_to_logical_count(
    # (whatever, num_layers, num_physical_experts)
    global_physical_count: torch.Tensor,
    num_layers: int,
    num_logical_experts: int,
    physical_to_logical_map: torch.Tensor,
):
    dim_extra, _, _ = global_physical_count.shape
    dtype = global_physical_count.dtype
    device = global_physical_count.device
    logical_count = torch.zeros(
        (dim_extra, num_layers, num_logical_experts), dtype=dtype, device=device
    )
    logical_count.scatter_add_(
        dim=2,
        index=physical_to_logical_map.unsqueeze(0)
        .expand(dim_extra, -1, -1)
        .to(torch.int64),
        src=global_physical_count,
    )
    return logical_count


def compute_gpu_physical_count(
    physical_count_of_whatever: torch.Tensor,  # (..., num_layer, num_physical_expert)
    num_gpu: int,
):
    """output: gpu_physical_count_of_batch (..., num_layer, num_gpu)"""
    return einops.reduce(
        physical_count_of_whatever,
        "... num_layer (num_gpu num_expert_per_gpu) -> ... num_layer num_gpu",
        "sum",
        num_gpu=num_gpu,
    )


def compute_utilization_rate(
    gpu_physical_count_of_batch: torch.Tensor,  # (..., num_layer, num_gpu)
):
    """计算每层的“均衡度”（balancedness），输出形状 (..., num_layer)。

    本质是“平均负载 / 峰值负载”：对每一层，先把物理专家计数按 GPU 聚合成每张卡的
    token 处理量，再取所有 GPU 的均值与最大值之比：

        utilization_rate = (avg_gpu_physical_count + eps) / (max_gpu_physical_count + eps)

    含义：
    - 值 = 1.0：完全均衡，每张卡负载相同。
    - 值越小：越不均衡，存在热点 GPU（最忙的卡远高于平均），成为瓶颈。
    - eps=1e-5 用于避免除零。

    示例：
        某层 4 张卡负载 = [100, 100, 100, 100] -> 100/100 = 1.0（完全均衡）
        某层 4 张卡负载 = [400, 0, 0, 0]     -> 100/400 = 0.25（严重不均衡）
    """
    gpu_physical_count_of_batch = gpu_physical_count_of_batch.float()
    max_gpu_physical_count = einops.reduce(
        gpu_physical_count_of_batch,
        "... num_layer num_gpu -> ... num_layer",
        "max",
    )
    avg_gpu_physical_count = einops.reduce(
        gpu_physical_count_of_batch,
        "... num_layer num_gpu -> ... num_layer",
        "mean",
    )
    return (avg_gpu_physical_count + 1e-5) / (max_gpu_physical_count + 1e-5)
