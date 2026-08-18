import logging
import time
from typing import TYPE_CHECKING, List

import torch.cuda

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ExpertLocationMetadata

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class EPLBManager:
    """EPLB（Expert Parallelism Load Balancer，专家并行负载均衡）管理器。

    背景
    ----
    在 MoE（Mixture-of-Experts）模型中，每个 token 会被路由到若干个"专家”（expert），
    这些专家分布在多张 GPU（多个 EP rank）上。由于路由是数据相关的，实际运行时不同专家
    的负载往往并不均衡：某些"热门专家”所在的 GPU 会成为瓶颈，而其他 GPU 则空闲。

    EPLB 的作用就是周期性地统计一段时间窗口内各专家的实际负载（logical_count），
    然后重新计算专家到物理设备的映射（必要时对热门专家做复制/冗余），从而让各 GPU 的
    负载尽可能均衡，提升整体吞吐。

    职责
    ----
    本类是"调度层”，负责决定"什么时候”触发重平衡、以及如何把重平衡拆分成多个 chunk
    分批执行；真正的负载采集由 ExpertDistributionRecorder 完成，真正的重平衡算法由
    eplb_algorithms 完成，物理权重的搬运由 ModelRunner.update_expert_location 完成。

    工作机制
    --------
    采用"生成器 + yield”的协程式设计（见 ``_entrypoint``）：每次 forward pass 结束时
    由外部调用 ``on_forward_pass_end`` 推进一步；累计到设定的迭代次数后触发一次 rebalance，
    而 rebalance 本身在分块（chunk）执行时也会 yield，从而把耗时操作分摊到多个 forward pass，
    避免单次卡顿过久。
    """

    def __init__(self, model_runner: "ModelRunner"):
        """初始化 EPLB 管理器。

        参数
        ----
        model_runner:
            所属的 ModelRunner 实例，通过它可以访问 server_args、model_config、model，
            以及执行专家位置更新（update_expert_location）。

        输入输出示例
        ------------
        >>> # 假设 server_args 中：
        >>> #   eplb_rebalance_num_iterations = 1000
        >>> #   expert_distribution_recorder_buffer_size = 100
        >>> #   eplb_rebalance_layers_per_chunk = None
        >>> manager = EPLBManager(model_runner)
        >>> # 构造完成后：
        >>> #   - 已启动专家分布记录器（开始采集负载）
        >>> #   - self._rebalance_num_iterations == 1000（每 1000 次 forward 触发一次重平衡）
        >>> #   - self._main_generator 是一个待推进的生成器
        """
        super().__init__()
        self._model_runner = model_runner
        self._server_args = model_runner.server_args
        # 每个 chunk 更新多少层；为 None 表示不分块（一次性更新所有层）
        self._rebalance_layers_per_chunk = (
            self._server_args.eplb_rebalance_layers_per_chunk
        )
        # 每隔多少次 forward pass 触发一次重平衡
        self._rebalance_num_iterations = self._server_args.eplb_rebalance_num_iterations

        # 重平衡的迭代周期必须 >= 分布记录器的环形缓冲区大小，
        # 否则环形缓冲区里会残留上一周期的过期数据（stale data）。
        # 如果确实需要支持更短周期，可以另行实现。
        assert (
            self._server_args.eplb_rebalance_num_iterations
            >= self._server_args.expert_distribution_recorder_buffer_size
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

    # 如有需要，这里的调度逻辑可以做得更复杂
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
        """执行一次专家负载重平衡（生成器）。

        流程：
          1. 从分布记录器 dump 出窗口内的负载统计（logical_count 与平均利用率）；
          2. 根据平均利用率判断是否需要重平衡（利用率已足够高则跳过）；
          3. 用 EPLB 算法计算新的专家位置元数据（ExpertLocationMetadata）；
          4. 按 chunk 分批把新的专家位置应用到模型（搬运/复制专家权重）。

        为什么是生成器：当按 chunk 分批更新时，每完成一个 chunk 就 ``yield`` 让出控制权，
        使得耗时的权重搬运分摊到多个 forward pass，避免单次阻塞过久。

        输入输出示例
        ------------
        >>> # dump_record(output_mode="object") 返回一个 dict，示意如下
        >>> # （EP world size=2, num_layers=1, num_logical_experts=4）：
        >>> dump_record_output = {
        ...     "rank": 0,
        ...     # logical_count 形状为 (dim_extra, num_layers, num_logical_experts)，
        ...     # 这里 dim_extra=1；数值为窗口内各(层,逻辑专家)累计的 token 计数，
        ...     # 且已跨 rank all_reduce 求和
        ...     "logical_count": [[[100, 5, 5, 5]]],  # 逻辑专家0是热门专家，负载远高于其他
        ...     # 窗口内平均 GPU 利用率（0~1）；仅 rank 0 有真实值
        ...     "average_utilization_rate_over_window": 0.40,
        ... }
        >>> # 取出两个关键字段：
        >>> logical_count = dump_record_output["logical_count"]                       # [[[100, 5, 5, 5]]]
        >>> util = dump_record_output["average_utilization_rate_over_window"]         # 0.40
        >>> # 由于 0.40 <= 阈值（例如 0.90），需要重平衡：
        >>> gen = manager.rebalance()
        >>> for _ in gen:  # 逐个 chunk 推进
        ...     pass
        >>> # 结果：逻辑专家0被复制到多张卡（获得多个物理副本），各 EP rank 负载趋于均衡
        """
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

        # 依据负载统计用 EPLB 算法计算新的专家位置元数据
        expert_location_metadata = ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count
        )

        # 把需要更新的层 id 切分成若干 chunk，分批应用
        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            # 只有在确实分成多块时才让出控制权（单块无需 yield）
            if len(update_layer_ids_chunks) > 1:
                yield
            # 把新的专家位置应用到模型对应的层（真正搬运/复制专家权重）
            self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.get_device_module().synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

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
            > self._server_args.eplb_min_rebalancing_utilization_threshold
        ):
            logger.info(
                f"[EPLBManager] Skipped ep rebalancing: current GPU utilization {average_utilization_rate_over_window:.2f} > minimum rebalance threshold {self._server_args.eplb_min_rebalancing_utilization_threshold:.2f}"
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
            list(self._model_runner.model.routed_experts_weights_of_layer.keys())
        )
        # None 时用一个足够大的值，等价于不分块
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))


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
