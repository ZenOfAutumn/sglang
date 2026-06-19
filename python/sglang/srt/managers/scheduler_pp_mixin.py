from __future__ import annotations

# 中译：本文件实现流水线并行（Pipeline Parallelism, PP）的调度逻辑，作为 Scheduler 的 mixin。
#       PP 把模型按层切成多个 stage，每个 stage 在一个 PP rank 上。一条请求要依次流经
#       stage 0 → stage 1 → ... → 最后一个 stage 才完成一次 forward。为了不让前面的 stage
#       在等后面的 stage 时空转，引入 micro-batch（mb）：把多个批次错开送入流水线，
#       使各 stage 同时在算不同的 micro-batch，从而填满流水线、提高利用率。
#       核心内容：
#       - event_loop_pp / event_loop_pp_disagg_prefill / event_loop_pp_disagg_decode：
#         三种 PP 调度循环（普通、PD 分离的 prefill 端、PD 分离的 decode 端）。
#       - PPBatchMetadata：随 micro-batch 传递的轻量元数据（如是否可用 CUDA Graph）。
#       - ChunkSizePredictor：用二次模型拟合 prefill 延迟，动态预测下一个 chunk 大小。
#       关键难点都在「各 PP stage 间有序的点对点收发」与「计算/通信重叠」：用异步 send + 同步 recv
#       避免错位，用按 rank 奇偶决定收发顺序避免环形死锁，用 micro-batch 缓冲实现重叠。

import logging
import math
import time
from array import array
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed
from tqdm import tqdm

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.utils import poll_and_all_reduce_attn_cp_tp_group
from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_dp_size,
    is_dp_attention_enabled,
    set_is_extend_in_batch,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_logprob_dict_from_result,
    get_logprob_from_pp_outputs,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import DynamicGradMode, broadcast_pyobj, point_to_point_pyobj
from sglang.srt.utils.common import get_device_module, is_xpu

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


def _pp_can_skip_output_comm(batch: ScheduleBatch) -> bool:
    """Check if output send/recv can be skipped for this batch.

    中译：判断本批次能否跳过「输出张量」在 PP 环上的收发。
          满足下列全部条件即可跳过（纯中间 chunk 的 prefill，输出对外层无用）：
          打开了对应开关、是 EXTEND（prefill）模式、批中只有 1 个请求、
          不是最后一个 prefill chunk、且不需要返回 logprob。跳过通信可省一次 P2P 开销。
    """
    return (
        envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get()
        and batch is not None
        and batch.forward_mode == ForwardMode.EXTEND
        and len(batch.reqs) == 1
        and not batch.contains_last_prefill_chunk
        and not batch.return_logprob
    )


@dataclass
class PPBatchMetadata:
    # 中译：随 micro-batch 一起保存的轻量元数据。目前只记录该批次本轮是否启用了 CUDA Graph，
    #       后续处理批结果时需要这个标志来正确还原执行路径。
    can_run_cuda_graph: bool


class SchedulerPPMixin:
    @DynamicGradMode()
    def event_loop_pp(self: Scheduler):
        """
        A scheduler loop for pipeline parallelism.
        Notes:
        1. Each stage runs in the same order and is notified by the previous stage.
        2. We use async send but sync recv to avoid desynchronization while minimizing the communication overhead.
        3. We can use async batch depth to buffer the outputs in the last stage for to allow overlapping the GPU computation and CPU processing and avoid last PP rank staggler.

        中译：流水线并行（PP）的调度主循环。
        要点：
        1. 各 stage 以相同顺序运行，由上一 stage 通知（驱动）下一 stage，保证全环步调一致。
        2. 采用「异步 send + 同步 recv」：发送不阻塞以减小通信开销，接收同步以避免收发错位（desync）。
        3. 通过 pp_async_batch_depth（异步批深度）在最后一个 stage 缓冲输出，使「GPU 计算」与
           「CPU 后处理」重叠，避免最后一个 PP rank 成为拖尾瓶颈（straggler）。
        下面的「Unified Schedule」描述了单个 stage 在一轮里对第 i 个 micro-batch 的收发/计算顺序，
        其中 (i+1)%mb_size 指「上一轮已发出、本轮该回收处理结果」的那个 micro-batch。

        Unified Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1)% mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage(can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.

        ====================================================================
        """
        self.init_pp_loop_state()
        while True:
            server_is_idle = True
            # 中译：遍历所有 micro-batch 槽位。pp_loop_size = pp_size + 异步批深度，
            #       让流水线里同时在飞的 micro-batch 数量略多于 stage 数以实现重叠。
            for mb_id in range(self.pp_loop_size):
                # 中译：把本槽位对应的运行态切换到当前调度上下文（running/last 都是按 mb_id 保存的）。
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                # 中译：next_first_rank_mb_id 指「最后一个 stage 该把哪个 mb 的输出回送给 rank0」；
                #       next_mb_id 指「该回收并后处理结果」的那个 micro-batch 槽位。
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size
                with torch.profiler.record_function("recv_requests"):
                    recv_reqs = self.request_receiver.recv_requests()
                    self.process_input_requests(recv_reqs)
                if not self.pp_group.is_last_rank:
                    # 中译：先等上一轮发出的「请求转发」完成（提交异步通信），再发本轮新请求，
                    #       保证下游 stage 收到的请求顺序与本 stage 一致。
                    self._pp_commit_comm_work(self.send_req_work)
                    with torch.profiler.record_function("send_reqs_to_next_stage"):
                        # 中译：把本 stage 收到的请求异步转发给下一 stage（沿 PP 环单向传递请求元数据）。
                        self.send_req_work = self._pp_send_pyobj_to_next_stage(
                            recv_reqs,
                            async_send=True,
                        )
                with torch.profiler.record_function("get_next_batch_to_run"):
                    # 中译：为本 micro-batch 槽位挑选下一个要跑的批次（prefill/decode 由内部策略决定）。
                    self.mbs[mb_id] = self.get_next_batch_to_run()
                self.running_mbs[mb_id] = self.running_batch
                self.cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    # 中译：非首 stage 需要先从上一 stage 接收 proxy 张量（隐藏态/残差），作为本 stage 的输入。
                    pp_proxy_tensors = self._pp_recv_proxy_tensors()
                next_pp_outputs = None
                next_batch_result = None
                d2h_event = None
                if self.server_args.pp_async_batch_depth > 0:
                    # 中译：异步批深度>0 时，提前（在本批 launch 之前）收发上一轮输出并预处理，
                    #       让这部分通信/CPU 工作与紧接着的 GPU 计算重叠。
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                # 中译：等上一轮发出的 proxy 张量发送完成后，再启动本批前向，避免缓冲区被覆写。
                self._pp_commit_comm_work(self.send_proxy_work)
                if self.cur_batch:
                    # 中译：在独立 forward_stream 上启动本 micro-batch 的前向计算（不阻塞 CPU），
                    #       返回 result（含本 stage 输出的 proxy 张量）与 launch_event（前向已入队的事件）。
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )
                if self.server_args.pp_async_batch_depth == 0:
                    # 中译：异步批深度为 0 时，在 launch 之后才收发上一轮输出（无额外缓冲，不做提前重叠）。
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                if self.mbs[next_mb_id] is not None:
                    # 中译：等设备到主机（D2H）拷贝完成，确保 next_batch_result 中的 token 已就绪，
                    #       再处理上一轮该 micro-batch 的结果（此处与本批 GPU 计算重叠进行）。
                    d2h_event.synchronize()
                    with torch.profiler.record_function("process_batch_result"):
                        self._pp_process_batch_result(
                            self.mbs[next_mb_id],
                            next_batch_result,
                        )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]
                if not self.pp_group.is_last_rank:
                    if self.cur_batch:
                        # 中译：让默认计算流等待前向 launch_event，确保 proxy 张量已算完再发送给下一 stage。
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        with torch.profiler.record_function(
                            "send_proxy_dict_to_next_stage"
                        ):
                            # 中译：把本 stage 算出的隐藏态/残差 proxy 张量异步发给下一 stage。
                            self.send_proxy_work = self._pp_send_dict_to_next_stage(
                                result.pp_hidden_states_proxy_tensors.tensors,
                                async_send=True,
                                msg_type="proxy",
                            )

                # 中译：把本轮收到的输出留给下一轮（pp_outputs 供非末位 stage 转发给后继）。
                self.pp_outputs = next_pp_outputs

            # When the server is idle, self-check and re-init some states
            # 中译：整轮所有 micro-batch 都没有可跑的批次时，服务空闲，做自检并重置部分状态。
            if server_is_idle:
                self.on_idle()

    @DynamicGradMode()
    def event_loop_pp_disagg_prefill(self: Scheduler):
        """
        This is the prefill server event loop for pipeline parallelism.

        Notes:
        1. Following the same rules as the event_loop_pp.
        2. Adds extra steps for KV transfer process: bootstrap + release.

        Prefill Server Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith bootstrap req from previous stage
        recv ith transferred req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1) % mb_size th consensus bootstrapped req from previous stage
        local consensus on bootstrapped req
        recv prev (i+1) % mb_size th release req from previous stage
        local consensus on release req
        recv prev (i+1) % mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith bootstrap req to next stage
        send ith transferred req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage (can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.
        ====================================================================

        There are two additional elements compared to the regular schedule:

        Bootstrap Requests + Release Requests:
        - Both can have local failure and need to be consensus on. PP needs to guarantee eventual consistency of local failure and flush malfunc requests out as soft error.

        中译：PD 分离（Prefill/Decode disaggregation）下 prefill 端的 PP 调度循环。
        相比普通 event_loop_pp，额外增加了 KV 传输相关的两类步骤：bootstrap（建链）与 release（释放）。
        难点在于：bootstrap 和 release 都可能在某个 rank 上局部失败，必须在所有 PP rank 间「达成共识」
        （取交集为成功、并集为失败），以保证最终一致性，并把出故障的请求作为软错误剔除。
        实现手法：每个 rank 把本地状态沿 PP 环传给下一 rank，逐 rank 求交/并，最后一个 rank 形成共识后
        再回送给 rank0，rank0 据此真正推进各队列。
        """
        self.init_pp_loop_state()

        # PD additional state initialization
        # 中译：PD 场景下的额外状态。bmbs/tmbs 按 mb_id 记录各 micro-batch 的 bootstrap/transferred rid，
        #       供「下一轮回收时」与上一 stage 的结果求共识；各 send_*_work 缓存待提交的异步发送句柄。
        bmbs = [None] * self.pp_loop_size
        tmbs = [None] * self.pp_loop_size
        consensus_bootstrapped_rids: Optional[List[str]] = None
        transferred_rids: List[str] = []
        release_rids: Optional[List[str]] = None
        send_bootstrapped_work = []
        send_transfer_work = []
        send_consensus_bootstrapped_work = []
        send_release_work = []

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_release_rids = None
                next_consensus_bootstrapped_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.request_receiver.recv_requests()
                self.process_input_requests(recv_reqs)

                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)

                # 中译：收集本 stage 已完成 bootstrap（建链）的请求 rid，并与上一 stage 求共识；
                #       记入 bmbs[mb_id] 备下一轮使用，并提交上一轮的 bootstrap 发送工作。
                bootstrapped_rids = self._pp_pd_get_bootstrapped_ids()
                bmbs[mb_id] = bootstrapped_rids
                self._pp_commit_comm_work(send_bootstrapped_work)

                # 中译：收集本 stage 已完成 KV 传输（transferred）的请求 rid，逻辑同上。
                transferred_rids = self._pp_pd_get_prefill_transferred_ids()
                self._pp_commit_comm_work(send_transfer_work)
                tmbs[mb_id] = transferred_rids

                self.process_prefill_chunk()
                batch = self.get_new_batch_prefill()
                batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = self._pp_recv_proxy_tensors()

                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)
                if self.cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )
                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                # 中译：最后一个 stage 据本地结果形成「bootstrap 共识」并回送给 rank0；
                #       中间 stage 则把已收到的共识继续向下传。release 共识同理。
                send_consensus_bootstrapped_work, consensus_bootstrapped_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        bmbs,
                        next_first_rank_mb_id,
                        consensus_bootstrapped_rids,
                        bootstrapped_rids,
                    )
                )
                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                # 中译：若上一轮该 micro-batch 发过 bootstrap，则本轮接收共识结果并据此推进 bootstrap 队列。
                if bmbs[next_mb_id] is not None:
                    next_consensus_bootstrapped_rids = (
                        self._pp_recv_pyobj_from_prev_stage()
                    )
                    next_consensus_bootstrapped_rids = self.process_bootstrapped_queue(
                        next_consensus_bootstrapped_rids
                    )
                self._pp_commit_comm_work(send_consensus_bootstrapped_work)
                if tmbs[next_mb_id] is not None:
                    next_release_rids = self._pp_recv_pyobj_from_prev_stage()
                self._pp_commit_comm_work(send_release_work)
                # post-process the coming microbatch
                # 中译：后处理「即将到来」的 micro-batch（即上一轮发出、本轮回收的那个）的批结果。
                if self.mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    self._pp_process_batch_result(
                        self.mbs[next_mb_id],
                        next_batch_result,
                    )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]

                if tmbs[next_mb_id] is not None:
                    self.process_disagg_prefill_inflight_queue(next_release_rids)
                if not self.pp_group.is_last_rank:
                    self.send_req_work = self._pp_send_pyobj_to_next_stage(
                        recv_reqs, async_send=True
                    )
                    send_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                        bootstrapped_rids, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if self.cur_batch:
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            async_send=True,
                            msg_type="proxy",
                        )

                # 中译：把本轮接收到的输出/共识结果结转到下一轮对应变量，供下一轮该 micro-batch 处理。
                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_bootstrapped_rids = next_consensus_bootstrapped_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            # 中译：空闲且没有在途（inflight）KV 传输时才算真正空闲，做自检与状态重置。
            if server_is_idle and len(self.disagg_prefill_inflight_queue) == 0:
                self.on_idle()

    @DynamicGradMode()
    def event_loop_pp_disagg_decode(self: Scheduler):
        """中译：PD 分离下 decode 端的 PP 调度循环。

        与 prefill 端结构对称，但需在 PP rank 间对三类事件达成共识：
        - retract（回撤）：KV 空间不足时把请求踢回，等空闲再恢复；
        - prealloc（预分配）：为新请求预分配 KV 空间；
        - release/transfer（传输完成、释放）：KV 从 prefill 端传到位后放行进入解码。
        共识方式同 prefill：沿环逐 rank 求交/并，末位 rank 形成共识后回送 rank0。
        另外 decode 批可能是 prebuilt（已预构建）的，prebuilt 批不参与 proxy 收发与结果后处理。
        """
        self.init_pp_loop_state()

        # PD additional state initialization
        # 中译：rmbs/pmbs/tmbs 分别按 mb_id 记录各 micro-batch 的 retract/prealloc/transferred rid，
        #       连同各 send_*_work 句柄一起用于跨轮、跨 stage 的共识传递。
        rmbs = [None] * self.pp_loop_size
        pmbs = [None] * self.pp_loop_size
        tmbs = [None] * self.pp_loop_size
        consensus_retract_rids: Optional[List[str]] = None
        consensus_prealloc_rids: Optional[List[str]] = None
        release_rids: Optional[List[str]] = None  # consensus transferred rids
        send_retract_work = []
        send_prealloc_work = []
        send_transfer_work = []
        send_consensus_retract_work = []
        send_consensus_prealloc_work = []
        send_release_work = []

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_consensus_retract_rids = None
                next_consensus_prealloc_rids = None
                next_release_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.request_receiver.recv_requests()
                self.process_input_requests(recv_reqs)

                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)

                # reaching consensus through PP ranks
                # 中译：沿 PP 环收集并求共识——本 stage 的 retract/prealloc/transferred rid 与上一 stage 求交/并。
                retract_rids = self._pp_pd_get_retract_ids(mb_id)
                rmbs[mb_id] = retract_rids
                self._pp_commit_comm_work(send_retract_work)

                prealloc_rids = self._pp_pd_get_prealloc_ids()
                pmbs[mb_id] = prealloc_rids
                self._pp_commit_comm_work(send_prealloc_work)

                transferred_rids = self._pp_pd_get_decode_transferred_ids()
                tmbs[mb_id] = transferred_rids
                self._pp_commit_comm_work(send_transfer_work)

                # get batch to run and proxy tensors if needed
                # 中译：取下一个 decode 批；若非 prebuilt 批，则需从上一 stage 接收 proxy 张量作为输入。
                batch = self.get_next_disagg_decode_batch_to_run()
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = None
                    if not self.cur_batch.forward_mode.is_prebuilt():
                        pp_proxy_tensors = self._pp_recv_proxy_tensors()

                # early send output if possible
                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)

                if self.cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )

                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )

                # reach consensus on last rank and send to PP=0
                # otherwise, just pass along previous consensus
                send_consensus_retract_work, consensus_retract_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        rmbs,
                        next_first_rank_mb_id,
                        consensus_retract_rids,
                        retract_rids,
                    )
                )

                send_consensus_prealloc_work, consensus_prealloc_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        pmbs,
                        next_first_rank_mb_id,
                        consensus_prealloc_rids,
                        prealloc_rids,
                    )
                )

                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                if self.server_args.disaggregation_decode_enable_offload_kvcache:
                    self.decode_offload_manager.check_offload_progress()

                if rmbs[next_mb_id] is not None:
                    next_consensus_retract_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_consensus_retract_rids = self.process_retract_queue(
                        next_consensus_retract_rids
                    )
                self._pp_commit_comm_work(send_consensus_retract_work)

                if pmbs[next_mb_id] is not None:
                    next_consensus_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_consensus_prealloc_rids = self.process_prealloc_queue(
                        next_consensus_prealloc_rids
                    )
                self._pp_commit_comm_work(send_consensus_prealloc_work)

                if tmbs[next_mb_id] is not None:
                    next_release_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_release_rids = self.process_decode_transfer_queue(
                        next_release_rids
                    )
                self._pp_commit_comm_work(send_release_work)

                # post-process the coming microbatch
                # 中译：后处理上一轮发出、本轮回收的 micro-batch；prebuilt 批无需等 D2H、也不做结果后处理。
                if self.mbs[next_mb_id] is not None:
                    if not self.mbs[next_mb_id].forward_mode.is_prebuilt():
                        d2h_event.synchronize()
                        self._pp_process_batch_result(
                            self.mbs[next_mb_id],
                            next_batch_result,
                        )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]

                if not self.pp_group.is_last_rank:
                    self.send_req_work = self._pp_send_pyobj_to_next_stage(
                        recv_reqs, async_send=True
                    )
                    send_retract_work = self._pp_send_pyobj_to_next_stage(
                        retract_rids, async_send=True
                    )
                    send_prealloc_work = self._pp_send_pyobj_to_next_stage(
                        prealloc_rids, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if self.cur_batch and not self.cur_batch.forward_mode.is_prebuilt():
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            async_send=True,
                            msg_type="proxy",
                        )

                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_retract_rids = next_consensus_retract_rids
                consensus_prealloc_rids = next_consensus_prealloc_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)

            if server_is_idle and queue_size == 0:
                self.on_idle()

    def init_pp_loop_state(self: Scheduler):
        # 中译：初始化 PP 调度循环所需的全部状态。
        # 中译：流水线槽位数 = PP stage 数 + 异步批深度；多出的深度用来缓冲输出以实现计算/通信重叠。
        self.pp_loop_size: int = self.ps.pp_size + self.server_args.pp_async_batch_depth
        # In CP mode, attention weights are duplicated, eliminating the need for the attention TP all-gather operation.
        # 中译：CP（上下文并行）模式下 attention 权重是复制的，因此无需做 attention TP 的 all-gather。
        self.require_attn_tp_allgather = (
            not self.server_args.enable_dsa_prefill_context_parallel
        )
        # 中译：按槽位保存各 micro-batch 的当前批、上一批、运行态批；mb_metadata 保存对应元数据。
        self.mbs = [None] * self.pp_loop_size
        self.last_mbs = [None] * self.pp_loop_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False)
            for _ in range(self.pp_loop_size)
        ]
        self.mb_metadata: List[Optional[PPBatchMetadata]] = [None] * self.pp_loop_size
        self.pp_outputs: Optional[PPProxyTensors] = None
        # 中译：仅最后一个 stage 使用的输出缓冲队列，元素为 (前向完成事件, 待发送的 proxy 输出)，
        #       配合异步批深度延迟发送，避免末位 rank 成为拖尾瓶颈。
        self.last_rank_comm_queue: deque[Tuple[torch.Event, PPProxyTensors]] = deque()

        self.send_req_work = []
        self.send_proxy_work = []
        self.send_output_work = []
        self.launch_event = None
        # 中译：收到「类型不符」的张量字典时的暂存收件箱：按 msg_type 分桶缓存，
        #       等到需要该类型时再取出（见 _pp_recv_typed_dict 的解复用逻辑）。
        self._pp_tensor_dict_inbox: Dict[str, deque[Dict[str, torch.Tensor]]] = (
            defaultdict(deque)
        )

    def profile_and_init_predictor(self: Scheduler):
        """
        Profile prefill latency for dynamic chunk sizing.

        Only runs on PP0 (first rank), then broadcasts data to all ranks.
        All ranks fit coefficients using the same data.

        中译：对 prefill 延迟做离线 profiling，用于「动态分块大小」预测。
              仅在 PP0（首 stage）真正跑测量，再把 (seq_lens, latencies) 数据广播给所有 rank；
              各 rank 用同一份数据拟合二次模型系数，保证所有 rank 的预测器完全一致。
        """
        seq_lens: List[int] = []
        latencies: List[float] = []

        if self.pp_group.is_first_rank:
            model_runner = self.tp_worker.model_runner
            model_config = model_runner.model_config
            input_ids_list: List[array[int]] = []
            # 中译：构造一组从大到小、覆盖不同长度的随机输入，用于采样「序列长度 → 前向延迟」的数据点。
            #       起点取 1.25 倍 chunked_prefill_size，等差递减，最多 128 个样本。
            for i in range(128):
                chunk_size = int(
                    self.chunked_prefill_size * 1.25
                    - i * (self.chunked_prefill_size * 1.25 // 128)
                )
                if chunk_size <= 0:
                    break
                input_ids = array(
                    "q",
                    np.random.randint(
                        0, 10000, size=chunk_size, dtype=np.int64
                    ).tobytes(),
                )
                input_ids_list.append(input_ids)

            sampling_params = SamplingParams(
                temperature=0,
                max_new_tokens=1,
            )
            # Create and profile requests
            for i, input_ids in enumerate(
                tqdm(
                    input_ids_list,
                    desc="Profiling prefill latency for dynamic chunking",
                )
            ):
                req = Req(
                    rid=str(i),
                    origin_input_text="",
                    origin_input_ids=input_ids,
                    sampling_params=sampling_params,
                )
                req.full_untruncated_fill_ids = req.origin_input_ids
                req.fill_len = len(req.full_untruncated_fill_ids)
                req.logprob_start_len = -1
                req.set_extend_input_len(req.fill_len - len(req.prefix_indices))

                # Prepare batch
                batch = ScheduleBatch.init_new(
                    [req],
                    self.req_to_token_pool,
                    self.token_to_kv_pool_allocator,
                    self.tree_cache,
                    self.model_config,
                    False,
                    self.spec_algorithm,
                )

                current_seq_len = req.fill_len

                if is_dp_attention_enabled():
                    # For profiling, we only have one request on PP0
                    # Set global_num_tokens to indicate this rank has tokens, others have 0
                    dp_size = get_attention_dp_size()
                    global_num_tokens = [0] * dp_size
                    dp_rank = get_attention_dp_rank()
                    global_num_tokens[dp_rank] = current_seq_len
                    batch.global_num_tokens = global_num_tokens
                    batch.global_num_tokens_for_logprob = global_num_tokens

                hs = (
                    getattr(model_config, "hc_hidden_size", None)
                    or model_config.hidden_size
                )
                proxy_tensors = {
                    "hidden_states": torch.zeros(
                        (current_seq_len, hs),
                        dtype=model_config.dtype,
                        device=self.device,
                    ),
                    "residual": torch.zeros(
                        (current_seq_len, model_config.hidden_size),
                        dtype=model_config.dtype,
                        device=self.device,
                    ),
                }

                pp_proxy = PPProxyTensors(proxy_tensors)

                # Measure latency with device synchronization for accurate timing
                device_module = get_device_module()
                # Synchronize before starting timing to ensure clean measurement
                device_module.synchronize()

                # 中译：从准备输入到前向完成、再做一次设备同步，精确测量这次 prefill 的端到端延迟。
                start = time.perf_counter()
                batch.prepare_for_extend()

                # Resolve deferred H2D: prepare_for_extend now leaves input_ids=None
                if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
                    batch.input_ids = batch.prefill_input_ids_cpu.to(
                        self.device, non_blocking=True
                    )
                    batch.prefill_input_ids_cpu = None

                forward_batch = ForwardBatch.init_new(batch, model_runner)
                set_is_extend_in_batch(batch.forward_mode.is_extend())

                _ = model_runner.forward(
                    forward_batch=forward_batch, pp_proxy_tensors=pp_proxy
                )

                # Synchronize after forward to ensure GPU operations complete
                device_module.synchronize()

                latency_seconds = time.perf_counter() - start
                latency_ms = latency_seconds * 1e3  # Convert to milliseconds
                seq_lens.append(len(input_ids))
                latencies.append(latency_ms)

                # Release KV cache
                if req.req_pool_idx is not None:
                    kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, : req.fill_len
                    ]
                    self.token_to_kv_pool_allocator.free(kv_indices)
                    self.req_to_token_pool.free(req)

            logger.info(
                f"[PP Dynamic Chunk] [PP0] Profiled {len(seq_lens)} samples: "
                f"seq_lens={seq_lens}, latencies_ms={latencies}"
            )

            if self.ps.attn_tp_size > 1:
                data_to_sync_tp = [seq_lens, latencies]
                data_to_sync_tp = broadcast_pyobj(
                    data_to_sync_tp,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
                seq_lens, latencies = data_to_sync_tp

            if self.ps.attn_cp_size > 1:
                data_to_sync_tp = [seq_lens, latencies]
                data_to_sync_tp = broadcast_pyobj(
                    data_to_sync_tp,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

        # Broadcast data to all ranks
        # 中译：把 PP0 测得的数据沿 PP 组广播到所有 rank，确保各 rank 拟合出相同系数。
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            data_to_sync = [seq_lens, latencies]
            self.pp_group.broadcast_object_list(data_to_sync, src=0)
            seq_lens, latencies = data_to_sync

        # Quadratic model: f(l) = al^2 + bl + c
        # 中译：用二次模型 f(l)=al^2+bl+c 拟合延迟，并按 base chunk size 设定目标延迟，标记预测器就绪。
        self.length_predictor = ChunkSizePredictor()
        self.length_predictor.fit(seq_lens, latencies)
        self.length_predictor.set_target_latency(self.chunked_prefill_size)
        self.length_predictor.is_ready = True
        logger.info(
            f"[PP Dynamic Chunk] [PP{self.ps.pp_rank}] Predictor ready (quadratic). "
            f"Target latency: {self.length_predictor.target_latency:.2f}ms"
        )

    def predict_next_chunk_size(self: Scheduler, history_len: int) -> Optional[int]:
        """
        Predict next chunk size dynamically based on current history length.

        Args:
            history_len: Current sequence length

        Returns:
            Predicted chunk size, or None to use default chunked_prefill_size

        中译：根据当前已处理的历史长度，动态预测下一个 prefill chunk 的大小。
              返回 None 表示退回使用默认的 chunked_prefill_size（未启用/预测器未就绪/预测失败时）。
        """
        if (
            not self.enable_dynamic_chunking
            or self.length_predictor is None
            or not self.length_predictor.is_ready
        ):
            return None

        max_chunk_size = self.max_prefill_tokens
        predicted_size = self.length_predictor.predict_next_chunk_size(
            history_len=history_len,
            base_chunk_size=self.chunked_prefill_size,
            page_size=self.page_size,
            context_len=self.model_config.context_len,
            max_chunk_size=max_chunk_size,
        )

        if predicted_size is not None:
            logger.debug(
                f"[PP Dynamic Chunk] [PP{self.ps.pp_rank}] Predicted chunk size: "
                f"{predicted_size} (history_len={history_len})"
            )

        return predicted_size

    def process_bootstrapped_queue(
        self: Scheduler, bootstrapped_rids: Optional[List[str]]
    ):
        # finished consensus bootstrapped reqs and prepare the waiting queue
        # 中译：依据已达成共识的 bootstrap 结果（成功/失败两组 rid），从 bootstrap 队列弹出对应请求，
        #       成功的放入等待队列；返回 [成功 rid, 失败 rid] 供继续向下游传播。
        if bootstrapped_rids is not None:
            (
                good_consensus_bootstrapped_rids,
                bad_consensus_bootstrapped_rids,
            ) = bootstrapped_rids
            good_reqs, failed_reqs = (
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped(
                    return_failed_reqs=True,
                    rids_to_check=good_consensus_bootstrapped_rids
                    + bad_consensus_bootstrapped_rids,
                )
            )
            self.waiting_queue.extend(good_reqs)
            return [[req.rid for req in good_reqs], [req.rid for req in failed_reqs]]
        return None

    def _pp_pd_get_bootstrapped_ids(self: Scheduler):
        # communicate pre-consensus bootstrapp reqs
        # 中译：收集「待共识」的 bootstrap rid。首 rank 直接读本地状态；其余 rank 先收上一 rank 的结果，
        #       再与本地求交（good 取交集=全员都好，bad 取并集=任一失败即失败），逐 rank 累积成全环共识。
        if self.pp_group.is_first_rank:
            # First rank, pop the bootstrap reqs from the bootstrap queue
            good_bootstrapped_rids, bad_bootstrapped_rids = self.get_rids(
                self.disagg_prefill_bootstrap_queue.queue,
                True,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
        else:
            # Other ranks, receive the bootstrap reqs info from the previous rank and ensure the consensus
            prev_bootstrapped_rids = self._pp_recv_pyobj_from_prev_stage()
            prev_good_bootstrapped_rids, prev_bad_bootstrapped_rids = (
                prev_bootstrapped_rids
            )
            curr_good_bootstrapped_rids, curr_bad_bootstrapped_rids = self.get_rids(
                self.disagg_prefill_bootstrap_queue.queue,
                True,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
            good_bootstrapped_rids = list(
                set(prev_good_bootstrapped_rids) & set(curr_good_bootstrapped_rids)
            )
            bad_bootstrapped_rids = list(
                set(prev_bad_bootstrapped_rids) | set(curr_bad_bootstrapped_rids)
            )
        return [good_bootstrapped_rids, bad_bootstrapped_rids]

    def _pp_pd_get_prefill_transferred_ids(self: Scheduler):
        # get the current stage transfer success
        if self.pp_group.is_first_rank:
            transferred_rids = self.get_rids(
                self.disagg_prefill_inflight_queue,
                True,
                [KVPoll.Success, KVPoll.Failed],
            )
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self._pp_recv_pyobj_from_prev_stage()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.get_rids(
                self.disagg_prefill_inflight_queue,
                True,
                [KVPoll.Success, KVPoll.Failed],
            )
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = list(
                set(prev_transferred_rids) & set(curr_transferred_rids)
            )
        return transferred_rids

    def _pp_pd_send_consensus_bootstrapped_ids(
        self: Scheduler,
        bmbs: List[List[str]],
        next_first_rank_mb_id: int,
        consensus_bootstrapped_rids: List[str],
        bootstrapped_rids: List[str],
    ):
        # 3 (Release): send the release rids from last stage to the first stage
        send_consensus_bootstrapped_work = []
        if self.pp_group.is_last_rank:
            # 中译：最后一个 rank 已掌握全环共识，把它回送给 rank0（PP 环上「绕回」第一个 stage）。
            if bmbs[next_first_rank_mb_id] is not None:
                consensus_bootstrapped_rids = bootstrapped_rids
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if consensus_bootstrapped_rids is not None:
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        return send_consensus_bootstrapped_work, consensus_bootstrapped_rids

    def _pp_pd_send_consensus_release_ids(
        self: Scheduler,
        tmbs: List[List[str]],
        next_first_rank_mb_id: int,
        release_rids: List[str],
        transferred_rids: List[str],
    ):
        send_release_work = []
        if self.pp_group.is_last_rank:
            if tmbs[next_first_rank_mb_id] is not None:
                release_rids = transferred_rids
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if release_rids is not None:
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        return send_release_work, release_rids

    def _pp_commit_comm_work(self: Scheduler, work: List[P2PWork]) -> None:
        # 中译：「提交/兑现」一批异步点对点通信——逐个等待其底层 work 完成，再清空列表。
        #       用于在复用发送缓冲区或推进下一步之前，确保上一轮的异步 send 已真正落地。
        for p2p_work in work:
            p2p_work.work.wait()
        work.clear()

    def _pp_commit_send_output_work_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
    ) -> Tuple[
        Optional[PPProxyTensors],
        Optional[GenerationBatchResult],
        Optional[torch.Event],
    ]:
        self._pp_commit_comm_work(work=self.send_output_work)
        (
            next_pp_outputs,
            next_batch_result,
            d2h_event,
            self.send_output_work,
        ) = self._pp_send_recv_and_preprocess_output_tensors(
            next_first_rank_mb_id,
            next_mb_id,
            self.mbs,
            self.mb_metadata,
            self.last_rank_comm_queue,
            self.pp_outputs,
        )
        return next_pp_outputs, next_batch_result, d2h_event

    def _pp_send_pyobj_to_next_stage(self: Scheduler, data, async_send: bool = False):
        # 中译：把任意 Python 对象（请求列表、rid 共识等）发给「下一个 PP stage」。
        #       仅每个 attn TP/CP 组的 rank0 真正参与发送（其余 rank 内容由组内广播获得），
        #       目标 rank = ((pp_rank+1)%pp_size)*tp_size + dp_offset，即环上的后继 stage。
        p2p_work = []
        if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
            dp_offset = self.ps.attn_dp_rank * self.ps.attn_tp_size
            p2p_work = point_to_point_pyobj(
                data,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                self.world_group.cpu_group,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                ((self.ps.pp_rank + 1) % self.ps.pp_size) * self.ps.tp_size + dp_offset,
                async_send=async_send,
            )
        return p2p_work

    def _pp_recv_pyobj_from_prev_stage(self: Scheduler):
        # 中译：从「上一个 PP stage」接收 Python 对象。同样只有 attn TP/CP 组 rank0 实际收，
        #       收到后再在 attn TP / attn CP 组内广播，使组内所有 rank 拿到一致的数据。
        if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
            dp_offset = self.ps.attn_dp_rank * self.ps.attn_tp_size
            data = point_to_point_pyobj(
                [],
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                self.world_group.cpu_group,
                ((self.ps.pp_rank - 1) % self.ps.pp_size) * self.ps.tp_size + dp_offset,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
            )
        else:
            data = None

        if self.ps.attn_tp_size > 1:
            data = broadcast_pyobj(
                data,
                self.attn_tp_group.rank,
                self.attn_tp_cpu_group,
                src=self.attn_tp_group.ranks[0],
            )

        if self.ps.attn_cp_size > 1:
            data = broadcast_pyobj(
                data,
                self.attn_cp_group.rank,
                self.attn_cp_cpu_group,
                src=self.attn_cp_group.ranks[0],
            )

        return data

    def _pp_prepare_tensor_dict(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {
            "next_token_ids": result.next_token_ids,
        }

        if batch.return_logprob:
            logprob_dict = get_logprob_dict_from_result(result)
            tensor_dict = {
                **tensor_dict,
                **logprob_dict,
            }
        return tensor_dict

    def _pp_send_dict_to_next_stage(
        self: Scheduler,
        tensor_dict: Dict[str, torch.Tensor],
        async_send: bool = True,
        msg_type: str = "default",
    ):
        # Warn once if using default untyped messages
        # 中译：发送张量字典到下一 stage。同一对 stage 间会交错传 proxy（前向输入）与 output（结果）两类，
        #       因此用 __msg_type__ 给消息打标签，接收端据此解复用；未打标签（default）时告警一次。
        if msg_type == "default":
            logger.warning_once(
                "PP send: using default untyped message. "
                "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
            )
        tensor_dict["__msg_type__"] = msg_type
        p2p_work = []
        p2p_work.extend(
            self.pp_group.send_tensor_dict(
                tensor_dict=tensor_dict,
                all_gather_group=(
                    self.attn_tp_group if self.require_attn_tp_allgather else None
                ),
                async_send=async_send,
            )
        )
        return p2p_work

    def _pp_recv_typed_dict(
        self: Scheduler,
        expected_kind: str = "default",
        all_gather_group: Optional = None,
    ) -> Dict[str, torch.Tensor]:
        """Receive a typed tensor dict, demultiplexing by msg_type.

        If a message of the wrong kind is received, it's stashed in the queue
        and we continue receiving until we get the expected kind.

        中译：接收指定类型（expected_kind）的张量字典，按 msg_type 解复用。
              因为 proxy 与 output 两类消息在同一通道上交错到达，若先收到「类型不符」的消息，
              先把它暂存进 inbox 队列，继续接收直到拿到期望类型；下次再需要被暂存的类型时直接取出。
        """
        # 中译：若收件箱里已有期望类型的消息，直接取用，无需再走网络接收。
        if expected_kind in self._pp_tensor_dict_inbox:
            inbox_queue = self._pp_tensor_dict_inbox[expected_kind]
            if inbox_queue:
                return inbox_queue.popleft()

        while True:
            tensor_dict = self.pp_group.recv_tensor_dict(
                all_gather_group=all_gather_group
            )
            received_kind = tensor_dict.get("__msg_type__", "default")
            if received_kind == expected_kind:
                if received_kind == "default":
                    logger.warning_once(
                        f"PP recv: got default untyped message. Content keys: {tensor_dict.keys()}"
                        "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
                    )
                return tensor_dict
            else:
                # 中译：类型不符——暂存到对应类型的收件箱，继续循环接收，直到拿到期望类型。
                logger.debug(
                    f"PP recv: expected {expected_kind}, got {received_kind}, stashing"
                )
                self._pp_tensor_dict_inbox[received_kind].append(tensor_dict)

    def _pp_recv_proxy_tensors(self: Scheduler) -> Optional[PPProxyTensors]:
        pp_proxy_tensors = None
        if not self.pp_group.is_first_rank:
            pp_proxy_tensors = PPProxyTensors(
                self._pp_recv_typed_dict(
                    expected_kind="proxy",
                    all_gather_group=(
                        self.attn_tp_group if self.require_attn_tp_allgather else None
                    ),
                )
            )
        return pp_proxy_tensors

    def _pp_recv_dict_from_prev_stage(
        self: Scheduler,
    ) -> Dict[str, torch.Tensor]:
        return self._pp_recv_typed_dict(
            expected_kind="output",
            all_gather_group=(
                self.attn_tp_group if self.require_attn_tp_allgather else None
            ),
        )

    def _pp_make_skip_output_result(
        self: Scheduler,
        batch: ScheduleBatch,
        mb_metadata: Optional[PPBatchMetadata],
    ):
        # 中译：当本批可跳过输出通信（见 _pp_can_skip_output_comm）时，造一个占位结果，避免真的收发。
        bs = len(batch.reqs)
        placeholder = torch.zeros(bs, dtype=torch.int64, device=self.device)
        # next_pp_outputs = None so non-last ranks skip forwarding
        # (pp_outputs is None gate). Placeholder carried in
        # batch_result.next_token_ids for process_batch_result_prefill.
        batch.output_ids = placeholder
        batch_result = GenerationBatchResult(
            logits_output=None,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=placeholder,
            can_run_cuda_graph=(
                mb_metadata.can_run_cuda_graph if mb_metadata else False
            ),
            skipped_output_comm=True,
        )
        d2h_event = self.device_module.Event()
        d2h_event.record(self.device_module.current_stream())
        return None, batch_result, d2h_event

    def _pp_prep_batch_result(
        self: Scheduler,
        batch: ScheduleBatch,
        mb_metadata: PPBatchMetadata,
        pp_outputs: PPProxyTensors,
    ):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        logits_output = None
        extend_input_len_per_req = None
        extend_logprob_start_len_per_req = None

        # 中译：rank0 收到末位 stage 回送的 next_token_ids（及可选 logprob），据此组装本批的最终结果。
        if batch.return_logprob:
            (
                logits_output,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = get_logprob_from_pp_outputs(pp_outputs)
        batch.input_ids = pp_outputs["next_token_ids"].to(torch.int64)
        # PP rank 0 also relays into output_tokens_buf so the next iter's
        # resolve_forward_inputs finds these tokens for the decode portion
        # of mixed-chunk batches (which gather via mix_running_indices).
        self.future_map.stash(batch.req_pool_indices, batch.input_ids)
        output_result = GenerationBatchResult(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=pp_outputs["next_token_ids"],
            extend_input_len_per_req=extend_input_len_per_req,
            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
            can_run_cuda_graph=mb_metadata.can_run_cuda_graph,
        )
        return output_result

    def _pp_process_batch_result(
        self: Scheduler, batch: ScheduleBatch, output_result: GenerationBatchResult
    ):
        self.process_batch_result(batch, output_result)

    def _pp_send_output_to_next_stage(
        self: Scheduler,
        next_first_rank_mb_id: int,
        mbs: List[ScheduleBatch],
        last_rank_comm_queue: deque,
        pp_outputs: PPProxyTensors | None,
    ) -> List[P2PWork]:
        send_output_work = []
        if self.pp_group.is_last_rank:
            # send ready PP output to rank 0
            # 中译：末位 stage 从缓冲队列取出最早一笔输出（含其前向完成事件），绕回发给 rank0；
            #       prebuilt 批或可跳过通信的批则不发。
            target = mbs[next_first_rank_mb_id]
            if target is not None:
                q_event, pp_outputs_to_send = last_rank_comm_queue.popleft()
                if (
                    not target.forward_mode.is_prebuilt()
                    and not _pp_can_skip_output_comm(target)
                ):
                    self.device_module.current_stream().wait_event(q_event)
                    with torch.profiler.record_function("send_res_dict_to_next_stage"):
                        send_output_work = self._pp_send_dict_to_next_stage(
                            pp_outputs_to_send.tensors,
                            async_send=True,
                            msg_type="output",
                        )
        # send the outputs from the last round to let the next stage worker run post processing
        # 中译：非末位 stage 把上一轮收到的输出继续转发给下一 stage，让后继 worker 去做结果后处理。
        if not self.pp_group.is_last_rank:
            if pp_outputs:
                with torch.profiler.record_function("send_res_dict_to_next_stage"):
                    send_output_work = self._pp_send_dict_to_next_stage(
                        pp_outputs.tensors,
                        async_send=True,
                        msg_type="output",
                    )
        return send_output_work

    def _pp_send_recv_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
        mbs: List[ScheduleBatch],
        mb_metadata: List[PPBatchMetadata],
        last_rank_comm_queue: deque[Tuple[torch.Event, PPProxyTensors]],
        pp_outputs: PPProxyTensors | None,
    ) -> Tuple[
        Optional[PPProxyTensors],
        Optional[GenerationBatchResult],
        Optional[torch.Event],
        List[P2PWork],
    ]:
        next_pp_outputs = None
        d2h_event = None
        batch_result = None
        send_output_work = []

        # On CUDA, isend is async: it enqueues to the stream and returns,
        # so every rank can send first safely. On some backends isend is
        # effectively blocking and does not return until the peer posts a
        # matching recv; if every PP rank sends first, all ranks block
        # waiting for a receiver and the ring deadlocks. Order send/recv
        # by pp_rank parity (even: send->recv, odd: recv->send) so each
        # adjacent pair has one sender and one receiver posted at the
        # same time.

        # 中译：上面解释了为何要按 rank 奇偶决定收发顺序——
        #       CUDA 上 isend 异步入队即返回，人人先发都安全；但某些后端（如 XPU）isend 是阻塞的，
        #       要等对端 post 出匹配的 recv 才返回。若所有 PP rank 都先发，就会全员等收方而导致环形死锁。
        #       因此按 pp_rank 奇偶错开：偶数 rank「先发后收」，奇数 rank「先收后发」，
        #       使每对相邻 stage 总有一发一收同时就绪。
        # CUDA: send first
        # XPU: even ranks send first, odd ranks recv first.
        send_first = (not is_xpu()) or ((self.ps.pp_rank % 2) == 0)

        def _do_send():
            return self._pp_send_output_to_next_stage(
                next_first_rank_mb_id,
                mbs,
                last_rank_comm_queue,
                pp_outputs,
            )

        def _do_recv():
            nonlocal next_pp_outputs, batch_result, d2h_event
            target = mbs[next_mb_id]
            if target is None or target.forward_mode.is_prebuilt():
                return
            if _pp_can_skip_output_comm(target):
                next_pp_outputs, batch_result, d2h_event = (
                    self._pp_make_skip_output_result(target, mb_metadata[next_mb_id])
                )
                return
            with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
                next_pp_outputs = PPProxyTensors(self._pp_recv_dict_from_prev_stage())
            with self.copy_stream_ctx:
                self.copy_stream.wait_stream(self.schedule_stream)
                batch_result = self._pp_prep_batch_result(
                    target, mb_metadata[next_mb_id], next_pp_outputs
                )
                d2h_event = self.device_module.Event()
                d2h_event.record(self.device_module.current_stream())

        # 中译：按上面算出的顺序执行收发，避免环形死锁。
        if send_first:
            send_output_work = _do_send()
            _do_recv()
        else:
            _do_recv()
            send_output_work = _do_send()

        return next_pp_outputs, batch_result, d2h_event, send_output_work

    def _pp_launch_batch(
        self: Scheduler,
        mb_id: int,
        pp_proxy_tensors: PPProxyTensors,
        mb_metadata: List[Optional[PPBatchMetadata]],
        last_rank_comm_queue: deque,
    ):
        # 中译：在独立 forward_stream 上启动本 micro-batch 的前向，记录起止时间并生成完成事件；
        #       末位 stage 额外把输出 proxy 连同事件压入缓冲队列，供后续延迟发送（异步批深度重叠）。
        with torch.profiler.record_function("run_batch"):
            with self.forward_stream_ctx:
                self.forward_stream.wait_stream(self.schedule_stream)
                set_time_batch(
                    self.cur_batch.reqs,
                    "set_run_batch_cpu_start_time",
                    trace_only=True,
                )
                result = self.run_batch(self.cur_batch, pp_proxy_tensors)
                set_time_batch(
                    self.cur_batch.reqs,
                    "set_run_batch_cpu_end_time",
                    trace_only=True,
                    attrs={"pp_mb_id": mb_id},
                )
                mb_metadata[mb_id] = PPBatchMetadata(
                    can_run_cuda_graph=result.can_run_cuda_graph,
                )
                event = self.device_module.Event()
                event.record(self.device_module.current_stream())
                if self.pp_group.is_last_rank:
                    # (last rank) buffer the outputs for async batch depth
                    last_rank_comm_queue.append(
                        (
                            event,
                            PPProxyTensors(
                                self._pp_prepare_tensor_dict(result, self.cur_batch)
                            ),
                        )
                    )
        return result, event

    def get_rids(
        self: Scheduler, req_queue: List[Req], is_send: bool, *poll_statuses_group
    ):
        """
        Used by PP, get the required rids with the given poll statuses.

        中译：PP 专用——按给定的 KV 轮询状态（poll status）筛选出请求 rid。
              先对每个请求的 KV sender/receiver 轮询状态做 attn CP/TP 组内 all-reduce（保证组内一致），
              再按传入的多组状态分别过滤出对应 rid；可一次传多组状态（如 [成功], [失败]）分别返回。
        """
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender if is_send else req.kv_receiver for req in req_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        rids: List = []
        for poll_statuses in poll_statuses_group:
            rids.append(
                [
                    req.rid if is_send else req.req.rid
                    for req, poll in zip(req_queue, polls)
                    if poll in poll_statuses
                ]
            )
        return tuple(rids) if len(rids) > 1 else rids[0]

    def _pp_pd_get_retract_ids(self: Scheduler, mb_id: int):
        # communicate pre-consensus retracted reqs
        # 中译：把尚未归属 micro-batch 的回撤请求绑定到当前 mb_id，确保每个回撤请求只由一个 micro-batch
        #       负责共识，避免在不同槽位间重复处理。
        for req in self.disagg_decode_prealloc_queue.retracted_queue:
            # assign retracted reqs to the current microbatch
            if req.retraction_mb_id is None:
                req.retraction_mb_id = mb_id
        curr_retract_rids = [
            req.rid
            for req in self.disagg_decode_prealloc_queue.retracted_queue
            if req.retraction_mb_id == mb_id
        ]
        if self.pp_group.is_first_rank:
            # First rank, get all retracted req ids for the microbatch
            return curr_retract_rids
        else:
            # Other ranks, receive the retracted reqs info from the previous rank and ensure the consensus
            prev_retract_rids = self._pp_recv_pyobj_from_prev_stage()
            return list(set(prev_retract_rids) & set(curr_retract_rids))

    def _pp_pd_get_prealloc_ids(self: Scheduler):
        # communicate pre-consensus prealloc reqs
        if self.pp_group.is_first_rank:
            # First rank, pop the preallocated reqs from the prealloc queue
            good_prealloc_rids, bad_prealloc_rids = self.get_rids(
                self.disagg_decode_prealloc_queue.queue,
                False,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
        else:
            # Other ranks, receive the preallocated reqs info from the previous rank and ensure the consensus
            prev_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
            prev_good_prealloc_rids, prev_bad_prealloc_rids = prev_prealloc_rids
            curr_good_prealloc_rids, curr_bad_prealloc_rids = self.get_rids(
                self.disagg_decode_prealloc_queue.queue,
                False,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
            good_prealloc_rids = list(
                set(prev_good_prealloc_rids) & set(curr_good_prealloc_rids)
            )
            bad_prealloc_rids = list(
                set(prev_bad_prealloc_rids) | set(curr_bad_prealloc_rids)
            )
        return [good_prealloc_rids, bad_prealloc_rids]

    def _pp_pd_get_decode_transferred_ids(self: Scheduler):
        # get the current stage transfer success
        if self.pp_group.is_first_rank:
            transferred_rids = self.get_rids(
                self.disagg_decode_transfer_queue.queue,
                False,
                [KVPoll.Success, KVPoll.Failed],
            )
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self._pp_recv_pyobj_from_prev_stage()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.get_rids(
                self.disagg_decode_transfer_queue.queue,
                False,
                [KVPoll.Success, KVPoll.Failed],
            )
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = list(
                set(prev_transferred_rids) & set(curr_transferred_rids)
            )
        return transferred_rids

    def process_retract_queue(self: Scheduler, retract_rids: Optional[List[str]]):
        # 中译：依据共识后的回撤 rid，尝试恢复被回撤的请求（KV 空间够再多跑若干步解码时），放回等待队列。
        if retract_rids is not None:
            # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
            resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs(
                retract_rids
            )
            self.waiting_queue.extend(resumed_reqs)
            return [req.rid for req in resumed_reqs]
        return None

    def process_prealloc_queue(self: Scheduler, prealloc_rids: Optional[List[str]]):
        # 中译：依据共识后的预分配 rid，把已预分配 KV 的请求弹出并转入「等待 KV 传输」队列。
        #       若仍有待恢复的回撤请求，则优先让位、本轮不分配新请求。
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return [[], []]

        if prealloc_rids is not None:
            (
                good_consensus_prealloc_rids,
                bad_consensus_prealloc_rids,
            ) = prealloc_rids
            good_reqs, failed_reqs = self.disagg_decode_prealloc_queue.pop_preallocated(
                rids_to_check=good_consensus_prealloc_rids
                + bad_consensus_prealloc_rids,
            )
            self.disagg_decode_transfer_queue.extend(good_reqs)
            return [
                [req.req.rid for req in good_reqs],
                [req.req.rid for req in failed_reqs],
            ]
        return None

    def process_decode_transfer_queue(
        self: Scheduler, release_rids: Optional[List[str]]
    ):
        # 中译：依据共识后的 release rid，把 KV 已传输到位的请求弹出并放入等待队列（可正式开始解码）。
        if release_rids is not None:
            released_reqs = self.disagg_decode_transfer_queue.pop_transferred(
                release_rids
            )
            if self.enable_hisparse:
                for req in released_reqs:
                    self.hisparse_coordinator.admit_request_direct(req)
            self.waiting_queue.extend(released_reqs)
            return [req.rid for req in released_reqs]
        return None


class ChunkSizePredictor:
    """
    Predictor for dynamic chunk size based on quadratic latency model.

    Models latency as: f(l) = a*l^2 + b*l + c
    Predicts next chunk size x such that: f(L+x) - f(L) = target_latency

    中译：基于二次延迟模型的「动态分块大小」预测器。
          把累计前向延迟建模为 f(l)=a*l^2+b*l+c（attention 的 O(n^2) 复杂度使 a>0）。
          给定已处理长度 L 和目标单步延迟 target_latency，求解 x 使 f(L+x)-f(L)=target_latency，
          即「在保持每步延迟大致恒定的前提下，下一个 chunk 还能再吃多少 token」。
          随着 L 增大，attention 越来越贵，故预测的 x 会自动变小，让各步耗时更均衡。
    """

    def __init__(self):
        self.quadratic_coeff_a = 0.0
        self.linear_coeff_b = 0.0
        self.constant_coeff_c = 0.0
        self.target_latency: Optional[float] = None
        self.is_ready = False

    def fit(self, seq_lens: List[int], latencies: List[float]):
        """Fit quadratic coefficients f(l) = al^2 + bl + c from data points.

        中译：用最小二乘从 (序列长度, 延迟) 数据点拟合二次系数 a, b, c。
        """
        # Skip the first data point to reduce fitting bias, as the first run is slower without warmup
        # 中译：跳过第一个数据点——首次运行无 warmup 偏慢，会污染拟合结果。
        L = np.array(seq_lens[1:], dtype=np.float64)
        T = np.array(latencies[1:], dtype=np.float64)

        if len(L) < 8:
            raise ValueError(
                f"Not enough data points for quadratic fitting ({len(L)} < 8). "
                "Need at least 8 samples with different sequence lengths."
            )

        # Build design matrix for f(l) = al^2 + bl + c
        # 中译：构造设计矩阵 [l^2, l, 1]，用 lstsq 解出系数 [a, b, c]。
        X = np.column_stack([L * L, L, np.ones_like(L)])  # [l^2, l, 1]

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, T, rcond=None)
            if len(coeffs) >= 3:
                fitted_a = float(coeffs[0])  # quadratic coefficient
                fitted_b = float(coeffs[1])  # linear coefficient
                fitted_c = float(coeffs[2])  # constant coefficient
            else:
                raise ValueError("Failed to fit coefficients: insufficient rank")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to fit f(l) = al^2 + bl + c: {e}")

        # Validate coefficients
        # 中译：校验系数合理性。a 必须为正（attention 是 O(n^2)，二次项不应非正），否则视为 warmup 数据有问题。
        if fitted_a <= 0:
            raise ValueError(
                f"Fitted quadratic coefficient a={fitted_a:.2e} is not positive. "
                "Attention has O(n^2) complexity, so a must be positive. "
                "Check warmup data quality."
            )

        if fitted_b < 0:
            logger.warning(
                f"Fitted linear coefficient b={fitted_b:.2e} is negative. Setting b=0."
            )
            fitted_b = 0.0

        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c

        logger.info(
            f"[ChunkSizePredictor] Fitted coefficients: a={fitted_a:.2e}, "
            f"b={fitted_b:.2e}, c={fitted_c:.2e}"
        )

    def set_target_latency(self, base_chunk_size: int):
        """Set target latency based on base chunk size: target = f(base_chunk_size) - f(0).

        中译：以「默认 base chunk size 跑一个 chunk 的延迟」作为每步目标延迟，
              即 target = f(base_chunk_size) - f(0)，后续动态分块都向这个延迟看齐。
        """

        def f(l: float) -> float:
            """Total latency function: f(l) = al^2 + bl + c (or bl + c for linear)"""
            return (
                self.quadratic_coeff_a * l * l
                + self.linear_coeff_b * l
                + self.constant_coeff_c
            )

        self.target_latency = f(float(base_chunk_size)) - f(0.0)

        if self.target_latency <= 0:
            raise ValueError(
                f"Calculated target_latency={self.target_latency:.2f}ms is not positive. "
                "Check warmup data quality."
            )

        logger.info(
            f"[ChunkSizePredictor] Target latency: {self.target_latency:.2f}ms "
            f"(base_chunk_size={base_chunk_size})"
        )

    def predict_next_chunk_size(
        self,
        history_len: int,
        base_chunk_size: int,
        page_size: int,
        context_len: int,
        max_chunk_size: Optional[int] = None,
    ) -> Optional[int]:
        """
        Predict next chunk size x such that f(history_len + x) - f(history_len) = target_latency.

        Args:
            history_len: Current sequence length (L)
            base_chunk_size: Base chunk size
            page_size: Page size for alignment
            context_len: Maximum context length
            max_chunk_size: Maximum allowed chunk size (optional)

        Returns:
            Predicted chunk size, or None if prediction fails

        中译：求解 x 使 f(history_len+x)-f(history_len)=target_latency，即下一个 chunk 的大小。
              随后还会做平滑、按 page_size 对齐、限制不超过剩余上下文/上限等修正，失败时返回 None。
        """
        if not self.is_ready or self.target_latency is None:
            return None

        # Handle quadratic model: f(l) = al^2 + bl + c
        if self.quadratic_coeff_a <= 0:
            return None

        # Solve f(L+x) - f(L) = T
        # where f(L) = a*L^2 + b*L + c
        # This expands to: ax^2 + (2aL+b)x - T = 0
        # A = a, B = 2aL + b, C = -T
        # 中译：把 f(L+x)-f(L)=T 展开为关于 x 的一元二次方程 ax^2+(2aL+b)x-T=0，下面用求根公式解 x。
        A = self.quadratic_coeff_a
        B = 2 * self.quadratic_coeff_a * history_len + self.linear_coeff_b
        C = -self.target_latency

        # 中译：判别式 <0 说明无实数解（无法在目标延迟内吃下任何 token），放弃动态预测。
        discriminant = B * B - 4 * A * C

        if discriminant < 0:
            logger.warning(
                f"Discriminant is negative ({discriminant:.2e}). "
                f"No real solution for chunk size. L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        # 中译：取正根（-B+sqrt(Δ))/(2A) 作为理论 chunk 大小（另一根为负、无意义）。
        sqrt_discriminant = math.sqrt(discriminant)
        calculated_chunk_size_float = (-B + sqrt_discriminant) / (2 * A)

        if calculated_chunk_size_float <= 0:
            logger.warning(
                f"Calculated chunk size is non-positive ({calculated_chunk_size_float:.2f}). "
                f"L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        # Use a smooth coefficient to reduce the abrupt decrease in chunk size
        # 中译：用平滑系数在 base_chunk_size 与理论值之间插值，避免 chunk 大小骤降造成抖动；
        #       并兜底不小于 base 的 1/4。
        smooth_coeff = envs.SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR.get()
        smoothed_chunk_size = base_chunk_size + smooth_coeff * (
            calculated_chunk_size_float - base_chunk_size
        )
        # Make sure the dynamic chunk size is at least 1/4 of the base chunk size
        calculated_chunk_size = max(int(smoothed_chunk_size), base_chunk_size // 4)

        # Align to page_size (minimum alignment size is 64)
        # 中译：向下对齐到 page_size（最小对齐粒度 64），使 chunk 与 KV 分页边界一致。
        alignment_size = max(page_size, 64)
        dynamic_chunk_size = (calculated_chunk_size // alignment_size) * alignment_size

        # Ensure aligned size is at least alignment_size
        if dynamic_chunk_size < alignment_size:
            dynamic_chunk_size = alignment_size

        # Apply constraints
        # 中译：施加上限——不超过剩余上下文（预留 100 token 余量），也不超过传入的 max_chunk_size。
        max_allowed = context_len - history_len - 100  # Leave 100 tokens margin
        if max_chunk_size is not None:
            max_allowed = min(max_allowed, max_chunk_size)
        dynamic_chunk_size = min(dynamic_chunk_size, max_allowed)

        # Align again after min operation
        # 中译：取 min 后可能破坏对齐，这里再对齐一次。
        dynamic_chunk_size = (dynamic_chunk_size // alignment_size) * alignment_size

        if dynamic_chunk_size < alignment_size:
            return None

        return dynamic_chunk_size
