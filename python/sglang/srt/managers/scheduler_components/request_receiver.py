"""Request receiving for the scheduler.

中译：调度器的「请求接收」组件。
      负责把上游（TokenizerManager / RPC）发来的请求拉进调度器，并在张量并行（TP）/
      流水并行（PP）/ 数据并行（DP attention）等多 rank 拓扑下，把请求正确地广播或
      点对点转发到各 rank，保证所有 rank 看到一致的请求列表。
      关键流程（见 recv_requests）：
        1) 拉取原始请求（_pull_raw_reqs）：仅特定 rank（attn_tp/cp rank 0）真正收，
           其余 rank 等待广播；PP 非首段则从上一段点对点接收。
        2) 跨 rank 广播（_broadcast_reqs_across_ranks）：DP attention 下把请求拆成
           work（推理请求）与 control（控制消息）分别广播，避免一次全员 gloo 同步。
        3) 处理多模态接收、最后解包共享内存（shm）特征（顺序很重要，见各方法注释）。
"""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Union,
)

import zmq
from torch.distributed import barrier

from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.mm_utils import (
    has_shm_features,
    unwrap_shm_features,
)
from sglang.srt.utils import (
    broadcast_pyobj,
    point_to_point_pyobj,
)
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.server_args import ServerArgs
    from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerRequestReceiver:
    # 中译：请求接收器。冻结数据类，持有各通信组（tp / attn_tp / attn_cp / world 等）
    #       与并行状态 ps，以及若干回调；负责接收并在各 rank 间对齐请求。
    recv_from_tokenizer: Union[zmq.Socket, ScriptedTokenizerRecvProxy]  # 来自 TokenizerManager 的 socket
    recv_from_rpc: Optional[zmq.Socket]  # 来自 RPC 的 socket（控制类请求）
    recv_skipper: Any
    input_blocker: Any
    mm_receiver: Any
    ps: ParallelState
    tp_group: Any
    tp_cpu_group: Any
    attn_tp_group: Any
    attn_tp_cpu_group: Any
    attn_cp_group: Any
    attn_cp_cpu_group: Any
    world_group: Any
    server_args: ServerArgs
    model_config: ModelConfig
    max_recv_per_poll: int
    stream_output: Callable[..., None]
    get_last_forward_mode: Callable[[], Any]
    scripted_scheduler_hook: Optional[ScriptedSchedulerHook] = None

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        # 中译：判断单次轮询接收的请求数是否已达上限（<0 表示不限），用于避免一次拉空对端而饿死前向。
        if self.max_recv_per_poll < 0:
            return False
        return num_recv_reqs >= self.max_recv_per_poll

    @scheduler_nvtx_method("scheduler.recv_requests")
    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks.

        中译：对外主入口。在 tp_rank=0 上真正接收请求，再广播给其余所有 TP rank，
              使各 rank 的请求列表一致。依次执行：拉取 → 输入阻断器 → 跨 rank 广播 →
              多模态接收 → 解包 shm 特征。
        """

        # 中译：脚本化运行时（测试）钩子推进一步。
        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.step()

        # 中译：recv_skipper——某些前向模式下可跳过本轮接收（如忙于 decode 时减少干扰）。
        if self.recv_skipper is not None:
            if not self.recv_skipper.handle(self.get_last_forward_mode()):
                return []

        recv_reqs = self._pull_raw_reqs()

        # 中译：输入阻断器（input blocker）——用于暂停/缓冲输入（如热更新权重期间）。
        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        recv_reqs = self._broadcast_reqs_across_ranks(recv_reqs)

        recv_reqs = self._apply_mm_receiver(recv_reqs)

        self._finalize_shm_features(recv_reqs)

        return recv_reqs

    def _pull_raw_reqs(self) -> Optional[List]:
        # 中译：拉取原始请求。
        #       PP 首段（pp_rank==0）：在 attn_tp/cp rank 0 上非阻塞地从 tokenizer 与 rpc 两个
        #         socket 拉取，直到拉空或达上限；其余 rank 返回 None（等待后续广播）。
        #       PP 非首段：在 attn_tp/cp rank 0 上从上一流水段点对点接收请求。
        if self.ps.pp_rank == 0:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                recv_reqs = []

                # 中译：非阻塞循环拉取 tokenizer 端请求，直到 socket 拉空（ZMQError）或达上限。
                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)

                # 中译：同样地拉取 rpc 端的控制请求。
                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_rpc = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_rpc)
            else:
                # 中译：非接收 rank，置空，稍后由广播补齐。
                recv_reqs = None
        else:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                # 中译：PP 非首段：从上一段（pp_rank-1）点对点接收请求；dp_offset 用于定位同 DP 组内的对端 rank。
                dp_offset = self.ps.attn_dp_rank * self.ps.attn_tp_size
                recv_reqs = point_to_point_pyobj(
                    [],
                    self.ps.pp_rank * self.ps.tp_size + dp_offset,
                    self.world_group.cpu_group,
                    (self.ps.pp_rank - 1) * self.ps.tp_size + dp_offset,
                    self.ps.pp_rank * self.ps.tp_size + dp_offset,
                )
            else:
                recv_reqs = None
        return recv_reqs

    def _broadcast_reqs_across_ranks(self, recv_reqs: Optional[List]) -> List:
        # 中译：把（仅在接收 rank 上有内容的）请求列表广播到所有 rank。
        #       DP attention 路径：拆成 work（推理）与 control（控制）两类分别广播——
        #         work 只需在 attn_tp / attn_cp 组内广播；control 视开关在本地组或全 tp 组广播。
        #       非 DP attention 路径：直接在 tp 组内广播整份列表。
        if self.server_args.enable_dp_attention:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                # 中译：仅在 DP 组 leader 上做拆分，其余 rank 占位 None 等待广播填充。
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
            else:
                work_reqs = None
                control_reqs = None

            if self.ps.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )

            if self.ps.attn_cp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

            # When dp_attention_local_control_broadcast is enabled, each DP
            # group leader already receives control messages from the DP
            # controller, so we broadcast within attn_tp_group + attn_cp_group
            # instead of the full tp_group.  This avoids an expensive
            # all-ranks gloo sync.
            # 中译：启用 local_control_broadcast 时，各 DP 组 leader 已从 DP 控制器收到控制消息，
            #       因此只需在 attn_tp + attn_cp 组内广播即可，无需在整个 tp 组做昂贵的全员 gloo 同步。
            _local_ctrl = self.server_args.enable_dp_attention_local_control_broadcast
            if _local_ctrl:
                if self.ps.attn_tp_size != 1:
                    control_reqs = broadcast_pyobj(
                        control_reqs,
                        self.attn_tp_group.rank,
                        self.attn_tp_cpu_group,
                        src=self.attn_tp_group.ranks[0],
                    )
                if self.ps.attn_cp_size != 1:
                    control_reqs = broadcast_pyobj(
                        control_reqs,
                        self.attn_cp_group.rank,
                        self.attn_cp_cpu_group,
                        src=self.attn_cp_group.ranks[0],
                    )
            elif self.ps.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif self.ps.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return recv_reqs

    def _apply_mm_receiver(self, recv_reqs: List) -> List:
        # Process MM requests under EPD-disaggregation mode
        # 中译：在 EPD 分离（encoder/prefill/decode 分离）模式下处理多模态（MM）请求：
        #       等待编码特征到位；对超时/出错的请求构造 abort 并直接回流输出。
        if (
            self.ps.pp_rank == 0
            and self.server_args.language_only
            and self.server_args.encoder_transfer_backend
            in ["zmq_to_scheduler", "mooncake"]
        ):
            recv_reqs, abort_reqs = self.mm_receiver.process_waiting_requests(recv_reqs)
            for req, error_msg, error_code in abort_reqs:
                status_code = (
                    HTTPStatus.BAD_REQUEST
                    if error_code == 400
                    else HTTPStatus.INTERNAL_SERVER_ERROR
                )
                prepare_abort(req, error_msg, status_code=status_code)
                self.stream_output([req], req.return_logprob)
        return recv_reqs

    def _finalize_shm_features(self, recv_reqs: Optional[List]) -> None:
        # Unwrap shared memory features AFTER all broadcasts complete,
        # so that ShmPointerMMData metadata (not full tensor data) is what
        # gets serialized during broadcast_pyobj.
        # 中译：必须在所有广播完成「之后」才解包共享内存（shm）特征——这样广播 broadcast_pyobj
        #       序列化的是 ShmPointerMMData 这类「指针元数据」而非整块张量，避免重复搬运大张量。
        if recv_reqs:
            # Barrier for the non-DP-attention path only: there is a single
            # broadcast_pyobj on tp_cpu_group where the source rank returns
            # the original objects immediately while other ranks are still in
            # pickle.loads (-> __setstate__ -> shm_open).  Without a barrier
            # the source can call materialize() / shm_unlink before others
            # open the segment.  recv_reqs is consistent across all ranks
            # here (same broadcast), so the guard is deadlock-free.
            #
            # Under DP-attention no barrier is needed: the control_reqs
            # broadcast on tp_cpu_group (step 3) is a collective that forces
            # every rank to complete the earlier attn_tp / attn_cp work_reqs
            # deserializations (steps 1-2, which call shm_open) before any
            # rank returns from step 3.  POSIX guarantees shm_unlink only
            # removes the name; already-open handles stay valid.
            # 中译：仅「非 DP attention」路径需要这道屏障（barrier）。该路径只有一次
            #       tp_cpu_group 上的 broadcast_pyobj：源 rank 会立即拿到原对象返回，而其余
            #       rank 还在 pickle.loads（-> __setstate__ -> shm_open）打开共享内存段。
            #       若无屏障，源 rank 可能在别的 rank 打开段之前就 materialize()/shm_unlink，
            #       导致后者打不开。此处 recv_reqs 在各 rank 一致（同一次广播），故加屏障不会死锁。
            #       DP attention 路径无需屏障：control_reqs 在 tp_cpu_group 上的广播（步骤 3）本身是
            #       集合通信，会强制各 rank 先完成前面 attn_tp/attn_cp 的 work_reqs 反序列化（步骤 1-2，
            #       即 shm_open）再返回；且 POSIX 保证 shm_unlink 只删名字，已打开的句柄仍有效。
            if (
                not self.server_args.enable_dp_attention
                and self.ps.tp_size > 1
                and self.model_config.is_multimodal
                and has_shm_features(recv_reqs)
            ):
                barrier(group=self.tp_cpu_group)
            for req in recv_reqs:
                unwrap_shm_features(req)

    def _split_work_and_control_reqs(self, recv_reqs: List):
        # 中译：把请求拆成 work（实际推理请求：生成/嵌入及其批量版）与 control（其余控制消息）两类，
        #       以便在 DP attention 下分别走不同的广播组（见 _broadcast_reqs_across_ranks）。
        work_reqs = [
            req
            for req in recv_reqs
            if isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        control_reqs = [
            req
            for req in recv_reqs
            if not isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        return work_reqs, control_reqs
