"""DP-attention scheduling helpers.

中译：DP（数据并行）attention 相关的调度辅助模块。
      背景：启用 DP attention 时，多个 DP 组各自独立调度本地批次，但 MLP/MoE 等层
      仍需跨组协同（要么各组 token 数一致才能跑同一张 CUDA Graph，要么需 all-gather
      汇总全局 token 数）。本模块负责这层「MLP 同步」：
      - MLPSyncBatchInfo：封装本地批次的关键信息（token 数、能否走 CUDA Graph、前向模式等），
        通过 all_gather 在各 rank 间汇总，得出全局视图。
      - prepare_mlp_sync_batch_raw：核心流程——计算本地信息 → all-gather → 决定是否需要
        插入「空闲（idle）批次」让本组也参与集合通信 → 回填到批次对象。
      - SchedulerDPAttnAdapter：把上述逻辑封装为调度器可调用的适配器，并能按需生成 idle 批次。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.metrics_collector import DPCooperationInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import require_mlp_tp_gather

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


_ENABLE_METRICS_DP_ATTENTION = envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.get()


@dataclass
class MLPSyncBatchInfo:
    # 中译：MLP 同步所需的批次信息。前半部分是各 rank 的「本地」信息（待汇总），
    #       后半部分（some gathered elements）是 all_gather 之后填入的「全局」结果。
    dp_size: int  # DP 组数
    tp_size: int  # attn TP 大小
    cp_size: int  # attn CP（context parallel）大小

    num_tokens: int  # 本地批次 token 数
    num_tokens_for_logprob: int  # 本地需算 logprob 的 token 数
    can_cuda_graph: bool  # 本地能否走（普通）CUDA Graph
    is_extend_in_batch: bool  # 本地批次是否含 extend（prefill）
    local_can_run_tbo: bool  # 本地能否做 two-batch-overlap（TBO）
    local_forward_mode: int  # 本地前向模式（ForwardMode 的 int 值）
    can_run_breakable_cuda_graph: bool  # 本地能否走「可打断」CUDA Graph

    # some gathered elements
    # 中译：以下为 all_gather 汇总得到的全局结果（初始为 None，调用 all_gather 后填充）。
    tp0_info: torch.Tensor = None  # 各 (dp, tp0) rank 的原始信息张量
    global_num_tokens: list[int] = None  # 各 DP 组的 token 数
    global_num_tokens_for_logprob: list[int] = None  # 各 DP 组需算 logprob 的 token 数
    tbo_split_seq_index: torch.Tensor = None  # TBO 的序列切分位置
    global_forward_mode: int = None  # 全局统一的前向模式
    dp_cooperation_info: Optional[DPCooperationInfo] = None  # DP 协作度量（指标用）

    def _get_local_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        # 中译：把本地的 7 个标量信息打包成一个张量，供 all_gather 传输。
        return torch.tensor(
            [
                self.num_tokens,
                self.num_tokens_for_logprob,
                int(self.can_cuda_graph),
                int(self.is_extend_in_batch),
                int(self.local_can_run_tbo),
                self.local_forward_mode,
                int(self.can_run_breakable_cuda_graph),
            ],
            device=device,
            dtype=dtype,
        )

    def _get_fallback_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        # 中译：「不活跃 rank」的兜底信息张量：0 token、视作 idle 且可走 CUDA Graph，
        #       使其不影响全局取 min/max 的结果（详见 all_gather 中对 inactive rank 的覆盖）。
        return torch.tensor(
            [
                0,  # num_tokens
                0,  # num_tokens_for_logprob
                1,  # can_cuda_graph
                0,  # is_extend_in_batch
                1,  # local_can_run_tbo
                ForwardMode.IDLE.value,  # local_forward_mode
                0,  # can_run_breakable_cuda_graph
            ],
            device=device,
            dtype=dtype,
        )

    def all_gather(self, device, group: torch.distributed.ProcessGroup):
        # 中译：跨所有 rank 汇总本地信息，得到全局视图并回填到本对象。
        local_info_tensor = self._get_local_tensor(device=device)
        global_info_tensor = torch.empty(
            (self.dp_size, self.tp_size * self.cp_size, 7),
            dtype=torch.int64,
            device=device,
        )

        torch.distributed.all_gather_into_tensor(
            global_info_tensor.flatten(),
            local_info_tensor,
            group=group,
        )
        if device == "cpu":
            tp_active_ranks = get_tp_group().active_ranks_cpu
        else:
            tp_active_ranks = get_tp_group().active_ranks

        # Set fallback values for inactive ranks
        # 中译：把不活跃 rank 的信息整体覆盖为兜底值，避免它们干扰后续 min/max 聚合。
        tp_info = global_info_tensor.view(self.dp_size * self.tp_size * self.cp_size, 7)
        tp_info[tp_active_ranks == 0] = self._get_fallback_tensor(device=device)

        # 中译：只取每个 DP 组内 tp_rank=0 的那行（同组 TP 内信息一致，取代表即可）。
        tp0_info = global_info_tensor[:, 0, :]
        self.tp0_info = tp0_info
        # Perform only one Device-to-Host (D2H) memory copy
        # 中译：只做一次设备→主机（D2H）拷贝，把前两列（token 数）一起搬下来，减少同步开销。
        cpu_data = tp0_info[:, :2].cpu()
        self.global_num_tokens = cpu_data[:, 0].tolist()
        self.global_num_tokens_for_logprob = cpu_data[:, 1].tolist()
        # 中译：所有组都能走 graph 才算能走（min）；任一组含 extend 即视为有 extend（max）。
        self.can_cuda_graph = bool(tp0_info[:, 2].min().item())
        self.is_extend_in_batch = bool(tp0_info[:, 3].max().item())
        self.can_run_breakable_cuda_graph = bool(tp0_info[:, 6].min().item())
        if _ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(tp0_info[:, 5].tolist())


def _update_gather_batch(
    batch: ScheduleBatch,
    mlp_sync_info: MLPSyncBatchInfo,
    require_mlp_tp_gather: bool,
    skip_all_gather=False,
):
    # 中译：把已汇总的 MLP 同步信息回填到批次对象上（全局 token 数、前向模式、能否走 DP CUDA Graph 等）。
    # TODO: handle the case when moe_dense_tp_size != 1
    if not require_mlp_tp_gather:
        batch.global_num_tokens = [mlp_sync_info.num_tokens]
        batch.global_num_tokens_for_logprob = [mlp_sync_info.num_tokens_for_logprob]
    else:
        batch.global_num_tokens = mlp_sync_info.global_num_tokens
        batch.global_num_tokens_for_logprob = (
            mlp_sync_info.global_num_tokens_for_logprob
        )
    if not skip_all_gather:
        batch.is_extend_in_batch = mlp_sync_info.is_extend_in_batch
        batch.tbo_split_seq_index = mlp_sync_info.tbo_split_seq_index
        batch.global_forward_mode = mlp_sync_info.global_forward_mode

    # Check forward mode for cuda graph
    batch.can_run_dp_cuda_graph = mlp_sync_info.can_cuda_graph
    batch.can_run_dp_breakable_cuda_graph = mlp_sync_info.can_run_breakable_cuda_graph


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    dp_size: int,
    attn_tp_size: int,
    attn_cp_size: int,
    tp_group: GroupCoordinator,
    get_idle_batch: Callable[[], ScheduleBatch],
    disable_cuda_graph: bool,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
):
    # 中译：DP attention 下的 MLP 同步主流程（无状态 raw 版）。
    #       步骤：算本地 token 数/标志 → all_gather 得全局信息 → 据此决定是否需插入 idle 批次
    #       让本组也参与集合通信 → 回填到批次。返回处理后的 local_batch（可能被替换为 idle 批次）。
    # Check if other DP workers have running batches
    # 中译：先根据本地前向模式确定本地 token 数。prebuilt / idle 视为 0 token。
    if (
        local_batch is None
        or local_batch.forward_mode.is_prebuilt()
        or local_batch.forward_mode.is_idle()
    ):
        num_tokens = 0
        num_tokens_for_logprob = 0
    elif local_batch.forward_mode.is_decode():
        # 中译：decode 每个请求只产 1 token，token 数即批大小。
        num_tokens = local_batch.batch_size()
        num_tokens_for_logprob = num_tokens
    else:
        # 中译：extend（prefill）：token 数为 extend_num_tokens；
        #       需算 logprob 的 token 数按各请求 (extend_len - logprob_start_len) 累加。
        num_tokens = local_batch.extend_num_tokens
        num_tokens_for_logprob = sum(
            # We should have at least 1 token for sample in every case.
            # 中译：任何情况下都至少留 1 个 token 用于采样（故 max(..., 1)）。
            max(extend_len - logprob_start_len, 1)
            for logprob_start_len, extend_len in zip(
                local_batch.extend_logprob_start_lens,
                local_batch.extend_lens,
            )
        )
        assert (
            local_batch.return_logprob
            or num_tokens_for_logprob == local_batch.batch_size()
        )

    # 中译：skip_all_gather——attn-dp=1 等场景可跳过 all-gather（无需跨组同步）。
    skip_all_gather = envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
    # 中译：本地能否走普通 CUDA Graph：decode/idle/prebuilt 模式且未全局禁用 graph。
    can_cuda_graph = (
        local_batch is None
        or local_batch.forward_mode.is_decode_or_idle()
        or local_batch.forward_mode.is_prebuilt()
    ) and not disable_cuda_graph
    can_run_breakable_cuda_graph = (
        local_batch is not None
        and local_batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED)
        and not disable_cuda_graph
    )

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False
    if local_batch is not None:
        local_batch.is_extend_in_batch = is_extend_in_batch

    tbo_preparer = TboDPAttentionPreparer()
    # 中译：选择 all-gather 走的设备/通信组：满足条件时走 GPU（device）组，否则走 CPU（gloo）组。
    #       走 GPU 组更快，但需无 offload 且 overlap 调度被禁用或显式允许在 overlap 中做 NCCL all-gather。
    if len(offload_tags) == 0 and (
        disable_overlap_schedule
        or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
    ):
        group = tp_group.device_group
        device = tp_group.device
    else:
        group = tp_group.cpu_group
        device = "cpu"

    local_can_run_tbo, local_forward_mode = tbo_preparer.prepare_all_gather(local_batch)

    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=dp_size,
        tp_size=attn_tp_size,
        cp_size=attn_cp_size,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens_for_logprob,
        can_cuda_graph=can_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,
        local_can_run_tbo=local_can_run_tbo,
        local_forward_mode=local_forward_mode,
        can_run_breakable_cuda_graph=can_run_breakable_cuda_graph,
    )

    if not skip_all_gather:
        # 中译：执行 all-gather 汇总，并据 TBO 相关列计算切分位置与全局前向模式。
        mlp_sync_info.all_gather(device=device, group=group)

        mlp_sync_info.tbo_split_seq_index, mlp_sync_info.global_forward_mode = (
            tbo_preparer.compute_output(
                mlp_sync_info.tp0_info[:, 4:6],
            )
        )

    # Decide whether to emit idle batch
    # 中译：决定本组是否需要发一个「空闲批次」。只要全局存在 token（别的组在跑），
    #       本组即使本地为空也得发 idle 批次参与集合通信，否则会造成各 rank 不对齐而挂起。
    if skip_all_gather:
        # Skip idle batch when attn-dp=1
        # 中译：attn-dp=1 时无需跨组对齐，dp_size>1 才可能需要 idle 批次。
        need_idle_batch = dp_size > 1
    else:
        need_idle_batch = max(mlp_sync_info.global_num_tokens) > 0

    batch_to_gather = local_batch
    if need_idle_batch:
        if local_batch is None:
            # 中译：本地无批次——新建一个 idle 批次顶上。
            batch_to_gather = local_batch = get_idle_batch()
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            # 中译：prebuilt 批次不能直接改，挂一个 inner idle 批次专门用于跑 MLP 同步。
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()

    if batch_to_gather is not None:
        _update_gather_batch(
            batch_to_gather, mlp_sync_info, require_mlp_tp_gather, skip_all_gather
        )

    if _ENABLE_METRICS_DP_ATTENTION and local_batch is not None:
        local_batch.dp_cooperation_info = mlp_sync_info.dp_cooperation_info

    return local_batch


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerDPAttnAdapter:
    # 中译：DP attention 调度适配器。把上面的 raw 函数封装成调度器可直接调用的接口，
    #       并持有构造 idle 批次所需的各内存池/配置。
    tp_group: GroupCoordinator
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: BasePrefixCache
    offload_tags: set[str]
    ps: ParallelState
    server_args: ServerArgs
    model_config: ModelConfig
    enable_overlap: bool
    spec_algorithm: SpeculativeAlgorithm
    get_require_mlp_sync: Callable[[], bool]

    def prepare_mlp_sync_batch(self, local_batch: ScheduleBatch):
        # 中译：用本适配器持有的参数调用 raw 实现，执行 MLP 同步。
        return prepare_mlp_sync_batch_raw(
            local_batch,
            dp_size=self.server_args.dp_size,
            attn_tp_size=self.ps.attn_tp_size,
            attn_cp_size=self.ps.attn_cp_size,
            tp_group=self.tp_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=cuda_graph_fully_disabled(),
            require_mlp_tp_gather=require_mlp_tp_gather(self.server_args),
            disable_overlap_schedule=self.server_args.disable_overlap_schedule,
            offload_tags=self.offload_tags,
        )

    def maybe_prepare_mlp_sync_batch(
        self,
        batch: Optional[ScheduleBatch],
        need_sync: Optional[bool] = None,
    ) -> Optional[ScheduleBatch]:
        """
        Helper to prepare MLP sync batch for DP attention.
        Should be called after get_new_batch_prefill().

        Args:
            batch: The batch to process
            need_sync: If specified, overrides self.get_require_mlp_sync() for prepare_mlp_sync_batch decision

        中译：按需执行 MLP 同步的便捷封装，应在 get_new_batch_prefill() 之后调用。
              need_sync 显式给定时覆盖默认的 get_require_mlp_sync() 判定；否则用后者决定是否同步。
        """
        if need_sync if need_sync is not None else self.get_require_mlp_sync():
            batch = self.prepare_mlp_sync_batch(batch)
        return batch

    def get_idle_batch(self) -> ScheduleBatch:
        # 中译：构造一个空（idle）批次：不含任何请求，仅用于让本 DP 组参与集合通信/MLP 同步。
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch
