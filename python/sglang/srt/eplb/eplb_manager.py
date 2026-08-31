from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, List

import torch.cuda
import torch.distributed as dist
from torch import nn

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    format_expert_location_layout,
    format_expert_location_layout_diff,
    get_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.runtime_context import get_exec, get_model, get_parallel

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


class EPLBManager:
    def __init__(
        self,
        *,
        model_config: ModelConfig,
        ps: Any,
        get_model: Callable[[], nn.Module],
        get_expert_location_updater: Callable[[], ExpertLocationUpdater],
        get_expert_backup_client: Callable[[], Any],
        get_weight_updater: Callable[[], Any],
    ):
        super().__init__()
        # These collaborators are set on ModelRunner AFTER EPLBManager is
        # constructed (model load, expert_backup_client, weight_updater), so
        # they are read through getters at rebalance time, not captured here.
        self._model_config = model_config
        self._ps = ps
        self._get_model = get_model
        self._get_expert_location_updater = get_expert_location_updater
        self._get_expert_backup_client = get_expert_backup_client
        self._get_weight_updater = get_weight_updater
        self._rebalance_layers_per_chunk = (
            get_exec().moe.eplb_rebalance_layers_per_chunk
        )
        self._rebalance_num_iterations = get_exec().moe.eplb_rebalance_num_iterations
        self._rebalance_disabled_reason = None
        self._rebalance_disabled_logged = False

        # 重平衡的迭代周期必须 >= 分布记录器的环形缓冲区大小，
        # 否则环形缓冲区里会残留上一周期的过期数据（stale data）。
        # 如果确实需要支持更短周期，可以另行实现。
        assert (
            get_exec().moe.eplb_rebalance_num_iterations
            >= get_exec().moe.expert_distribution_recorder_buffer_size
        ), "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"

        # 如果全局的专家分布记录器还没有在记录，则启动记录（开始采集各专家负载）
        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )

        # 创建主生成器（协程），后续通过 on_forward_pass_end 逐步推进
        self._main_generator = self._entrypoint()

    def on_forward_pass_end(self):
        """在每次 forward pass 结束时由外部（ModelRunner/Scheduler）调用，推进一步状态机。

        每调用一次就 ``next`` 一次主生成器：大多数时候只是消耗一个空转的 yield（计数），
        当累计次数达到阈值时，这一步会实际触发（并分块推进）重平衡。

        输入输出示例
        ------------
        >>> # 该方法无参数、无返回值，仅推进内部生成器
        >>> for _ in range(2000):
        ...     manager.on_forward_pass_end()
        >>> # 若 _rebalance_num_iterations == 1000，则上面 2000 次调用期间
        >>> # 会触发 2 次重平衡
        """
        next(self._main_generator)

    def reset_generator(self):
        """重置主生成器，回到初始计数状态。

        典型用于权重更新、专家位置变更等场景后，重新开始计数周期。

        输入输出示例
        ------------
        >>> manager.reset_generator()
        >>> # 之后重新从 0 开始累计迭代次数
        """
        self._main_generator = self._entrypoint()

    def disable_rebalance(self, reason: str):
        self._rebalance_disabled_reason = reason
        self._rebalance_disabled_logged = False
        self.reset_generator()

    def enable_rebalance(self):
        self._rebalance_disabled_reason = None
        self._rebalance_disabled_logged = False
        self.reset_generator()

    # can be more complex if needed
    def _entrypoint(self):
        """主状态机（生成器）：无限循环地"先计数、后重平衡”。

        循环逻辑：
          1. 先 yield ``_rebalance_num_iterations`` 次（每次对应一个 forward pass，纯计数）；
          2. 然后把控制权委托给 ``self.rebalance()``（``yield from``），执行一次重平衡，
             其间若分块也会 yield，从而跨多个 forward pass 完成；
          3. 回到第 1 步，进入下一个周期。

        输入输出示例
        ------------
        >>> gen = manager._entrypoint()
        >>> next(gen)  # 第 1 次 forward pass（计数）
        >>> # ... 累计到阈值后，某一次 next(gen) 会进入 rebalance 流程
        """
        while True:
            # 阶段一：空转计数，等待累计到设定的迭代次数
            for _ in range(self._rebalance_num_iterations):
                yield

            # 阶段二：触发一次重平衡（可能分多步 yield 完成）
            yield from self.rebalance()

    def rebalance(self):
        if self._rebalance_disabled_reason is not None:
            if not self._rebalance_disabled_logged:
                logger.debug(
                    "[EPLBManager] rebalance disabled: %s",
                    self._rebalance_disabled_reason,
                )
                self._rebalance_disabled_logged = True
            return

        elastic_state = ElasticEPStateManager.instance()
        is_post_scale_rebalance = elastic_state is not None and elastic_state.has_scaled
        # A failed later scale leaves the previously committed world serving.
        if is_post_scale_rebalance and (
            elastic_state.pending_ep_size is not None
            or elastic_state.scale_phase not in ("serving_expanded", "failed")
        ):
            return

        logger.info("[EPLBManager] rebalance start")

        # 未设置分块（一次性更新所有层）时才做计时统计，
        # 因为分块模式下 rebalance 会跨多个 forward pass，计时没有意义
        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            torch.get_device_module().synchronize()
            time_start = time.time()

        # 从全局分布记录器 dump 出窗口内的负载统计，并在 dump 后重置记录器
        dump_record_output = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )
        # logical_count：各（层, 专家）在窗口内被路由到的 token 计数，形状约为
        #   [num_layers, num_logical_experts] 或带 EP 维度，用于驱动重平衡算法
        logical_count = dump_record_output["logical_count"]
        # average_utilization_rate_over_window：窗口内的平均 GPU 利用率（0~1）
        average_utilization_rate_over_window = dump_record_output[
            "average_utilization_rate_over_window"
        ]

        # 检查是否确实需要重平衡（利用率已经很高则跳过，避免无谓的搬运开销）
        if not self._check_rebalance_needed(average_utilization_rate_over_window):
            return

        expert_location_metadata = self._compute_expert_location_metadata(
            logical_count,
            broadcast_over_world=is_post_scale_rebalance,
        )

        from sglang.srt.model_executor.model_runner_components.moe_ep_setup import (
            init_lplb_solvers,
        )

        # 把需要更新的层 id 切分成若干 chunk，分批应用
        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        all_update_layer_ids = [
            layer_id for chunk in update_layer_ids_chunks for layer_id in chunk
        ]
        self._log_rebalance_layout_before_update(
            expert_location_metadata,
            update_layer_ids=all_update_layer_ids,
        )
        for chunk_layer_ids in update_layer_ids_chunks:
            if len(update_layer_ids_chunks) > 1:
                yield
            update_expert_location_with_recovery(
                expert_location_updater=self._get_expert_location_updater(),
                model=self._get_model(),
                new_expert_location_metadata=expert_location_metadata,
                update_layer_ids=chunk_layer_ids,
                nnodes=get_parallel().nnodes,
                tp_rank=(
                    self._elastic_global_rank()
                    if is_post_scale_rebalance
                    else self._ps.tp_rank
                ),
                use_flat_topology=is_post_scale_rebalance,
                expert_backup_client=self._get_expert_backup_client(),
                update_weights_from_disk_callable=self._get_weight_updater().update_weights_from_disk,
                ep_dispatch_algorithm=get_exec().moe.ep_dispatch_algorithm,
                init_lplb_solvers_callable=lambda: init_lplb_solvers(
                    model_config=self._model_config
                ),
            )
            if is_post_scale_rebalance:
                # P2P waits only synchronize participating peers. Ranks without
                # moves must also install this chunk before NIXL resumes.
                dist.barrier()

        self._log_rebalance_layout_after_update(update_layer_ids=all_update_layer_ids)

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.get_device_module().synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

    def _compute_expert_location_metadata(
        self, logical_count, *, broadcast_over_world: bool
    ) -> ExpertLocationMetadata:
        if not broadcast_over_world:
            return ExpertLocationMetadata.init_by_eplb(
                self._model_config,
                logical_count,
            )

        current_metadata = get_global_expert_location_metadata()
        assert current_metadata is not None
        # One owner prevents process-local launch topology from influencing
        # the mapping chosen for the expanded world.
        if dist.get_rank() == 0:
            computed_metadata = ExpertLocationMetadata.init_by_eplb(
                self._model_config,
                logical_count,
                # Arbitrary append topologies may not preserve node divisibility.
                use_flat_topology=True,
            )
            physical_to_logical_map = (
                computed_metadata.physical_to_logical_map.contiguous()
            )
        else:
            physical_to_logical_map = torch.empty_like(
                current_metadata.physical_to_logical_map
            )

        dist.broadcast(physical_to_logical_map, src=0)
        return ExpertLocationMetadata.init_by_mapping(
            self._model_config,
            physical_to_logical_map,
            moe_ep_rank=self._elastic_global_rank(),
        )

    def _elastic_global_rank(self) -> int:
        return self._ps.tp_rank + get_parallel().ep_join_rank_offset

    def _check_rebalance_needed(self, average_utilization_rate_over_window):
        """根据窗口内平均利用率判断是否需要重平衡。

        参数
        ----
        average_utilization_rate_over_window:
            窗口内的平均 GPU 利用率（0~1）；为 None 表示暂无统计数据。

        返回
        ----
        bool：True 表示需要重平衡，False 表示可以跳过。

        判定逻辑：当利用率已经高于阈值
        （``eplb_min_rebalancing_utilization_threshold``）时，说明 GPU 已被充分利用，
        此时重平衡带来的收益不足以抵消搬运开销，因此跳过。

        输入输出示例
        ------------
        >>> # 阈值 eplb_min_rebalancing_utilization_threshold = 0.90
        >>> manager._check_rebalance_needed(None)   # 无数据 -> 需要重平衡
        True
        >>> manager._check_rebalance_needed(0.40)   # 利用率低 -> 需要重平衡
        True
        >>> manager._check_rebalance_needed(0.95)   # 利用率已很高 -> 跳过
        False
        """
        # 尚无统计数据时，默认需要重平衡
        if average_utilization_rate_over_window is None:
            return True

        # 利用率已经高于阈值：GPU 已被充分利用，跳过重平衡
        if (
            average_utilization_rate_over_window
            > get_exec().moe.eplb_min_rebalancing_utilization_threshold
        ):
            logger.info(
                f"[EPLBManager] Skipped ep rebalancing: current GPU utilization {average_utilization_rate_over_window:.2f} > minimum rebalance threshold {get_exec().moe.eplb_min_rebalancing_utilization_threshold:.2f}"
            )
            return False

        return True

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        """计算本次重平衡需要更新的层 id，并按 chunk 大小切分。

        返回
        ----
        List[List[int]]：切分后的层 id 分块列表；每个子列表是一个 chunk。

        chunk 大小由 ``eplb_rebalance_layers_per_chunk`` 决定；若为 None，则用一个
        很大的数（1000000）作为 chunk 大小，等价于"不分块，所有层放在一个 chunk 里”。

        输入输出示例
        ------------
        >>> # 模型有 4 个含路由专家的层：[0, 1, 2, 3]
        >>> # 情形 A：eplb_rebalance_layers_per_chunk = None（不分块）
        >>> manager._compute_update_layer_ids_chunks()
        [[0, 1, 2, 3]]
        >>> # 情形 B：eplb_rebalance_layers_per_chunk = 2（每块 2 层）
        >>> manager._compute_update_layer_ids_chunks()
        [[0, 1], [2, 3]]
        """
        # 取出模型中所有含"路由专家权重”的层 id，并排序保证顺序稳定
        all_layer_ids = sorted(
            list(self._get_model().routed_experts_weights_of_layer.keys())
        )
        # None 时用一个足够大的值，等价于不分块
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))

    def _should_log_expert_location_metadata(self) -> bool:
        return self._ps.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get()

    def _log_rebalance_layout_before_update(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        if not self._should_log_expert_location_metadata():
            return

        old_expert_location_metadata = get_global_expert_location_metadata()
        logger.info(
            "[EPLBManager] rebalance layout before:\n%s",
            format_expert_location_layout(
                old_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )
        logger.info(
            "[EPLBManager] rebalance layout target:\n%s",
            format_expert_location_layout(
                new_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )
        logger.info(
            "[EPLBManager] rebalance layout diff:\n%s",
            format_expert_location_layout_diff(
                old_expert_location_metadata,
                new_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )

    def _log_rebalance_layout_after_update(self, update_layer_ids: List[int]):
        if not self._should_log_expert_location_metadata():
            return

        logger.info(
            "[EPLBManager] rebalance layout after:\n%s",
            format_expert_location_layout(
                get_global_expert_location_metadata(),
                layer_ids=update_layer_ids,
            ),
        )


def update_expert_location_with_recovery(
    *,
    expert_location_updater: ExpertLocationUpdater,
    model: nn.Module,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: List[int],
    nnodes: int,
    tp_rank: int,
    use_flat_topology: bool = False,
    expert_backup_client,
    update_weights_from_disk_callable,
    ep_dispatch_algorithm: str,
    init_lplb_solvers_callable,
):
    p2p_missing_logical_experts = expert_location_updater.update(
        model.routed_experts_weights_of_layer,
        new_expert_location_metadata,
        update_layer_ids=update_layer_ids,
        nnodes=nnodes,
        rank=tp_rank,
        use_flat_topology=use_flat_topology,
    )

    if len(p2p_missing_logical_experts) > 0:
        # Load the missing expert weights from disk
        if callable(getattr(model, "generate_weight_name_filter", None)):
            # Filter and load only missing expert weights
            weight_name_filter = model.generate_weight_name_filter(
                p2p_missing_logical_experts
            )
        else:
            # Do a full reload from disk/DRAM
            logger.info(
                "[Elastic EP] Model does not implement generate_weight_name_filter. "
                "Performing full weight reload."
            )
            weight_name_filter = None

        if expert_backup_client is not None and expert_backup_client.use_backup:
            # Load the missing weights from the DRAM backup
            expert_backup_client.update_weights(weight_name_filter)
        else:
            # Load the missing weights from disk
            update_weights_from_disk_callable(
                get_model().model_path,
                get_model().load_format,
                weight_name_filter=weight_name_filter,
            )

    # Re-init LPLB solvers after expert location update
    if ep_dispatch_algorithm == "lp":
        init_lplb_solvers_callable()


def _chunk_list(items: List, chunk_size):
    """把列表按固定大小切分成若干块的生成器工具函数。

    参数
    ----
    items:      待切分的列表。
    chunk_size: 每块的最大元素个数。

    产出
    ----
    依次 yield 出每个切片（子列表）；最后一块可能不足 chunk_size。

    输入输出示例
    ------------
    >>> list(_chunk_list([0, 1, 2, 3, 4], chunk_size=2))
    [[0, 1], [2, 3], [4]]
    >>> list(_chunk_list([0, 1, 2], chunk_size=5))
    [[0, 1, 2]]
    """
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
