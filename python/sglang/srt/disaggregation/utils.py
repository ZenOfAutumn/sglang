from __future__ import annotations

import os
import random
from collections import deque
from contextlib import nullcontext
from enum import Enum
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Type, overload

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.environ import envs
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.disaggregation.base.conn import KVArgs, StateType
    from sglang.srt.disaggregation.common.conn import (
        CommonKVBootstrapServer,
        CommonKVManager,
        CommonKVReceiver,
        CommonKVSender,
    )
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs

#########################
# Constants & Enums
# 中译：常量与枚举。本文件是 PD 分离（Prefill/Decode 分离部署）的通用工具集，
#       汇集了：运行模式枚举、跨 rank 状态轮询同步、传输元数据缓冲区、KV 传输后端工厂、
#       上下文并行（CP）下的页索引切分、以及请求中止等杂项工具。
#########################
# 中译：FAKE_BOOTSTRAP_HOST 是「伪 bootstrap 主机」的哨兵地址。当请求的 bootstrap_host
#       等于此值时，表示走 fake 后端（不做真实 KV 传输，常用于 warmup），见 _is_fake_transfer。
FAKE_BOOTSTRAP_HOST = "2.2.2.2"


class DisaggregationMode(Enum):
    # 中译：当前进程的分离部署角色。
    #   NULL   —— 非分离（unified，prefill+decode 同一实例）；
    #   PREFILL—— 仅承担预填充；
    #   DECODE —— 仅承担解码。
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

    @staticmethod
    def to_engine_type(mode: str) -> str:
        # 中译：把模式字符串映射为引擎类型标签（prefill/decode/unified），供上层标识使用。
        if mode == DisaggregationMode.PREFILL.value:
            return "prefill"
        elif mode == DisaggregationMode.DECODE.value:
            return "decode"
        return "unified"


#########################
# Synchronization
# 中译：同步。PD 分离下 KV 传输是异步的，需要周期性 poll 各请求的传输状态（KVPoll），
#       并在多个并行 rank（TP/CP）之间对齐状态——只有所有 rank 都就绪，某请求才能推进，
#       这靠 all_reduce(MIN) 实现（任一 rank 未完成，取最小值就会把整体拉回未完成态）。
#########################


def _get_failure_prob() -> float:
    # 中译：读取「故障注入概率」（仅测试用）。优先读新环境变量，读不到再回退到旧变量，
    #       都没有则视为 0（不注入故障）。用于模拟 KV 传输失败以验证容错路径。
    try:
        return float(envs.SGLANG_TEST_DISAGG_FAILURE_PROB.get())
    except Exception:
        # fallback to legacy env var
        # 中译：回退到旧版环境变量 DISAGGREGATION_TEST_FAILURE_PROB。
        return float(os.getenv("DISAGGREGATION_TEST_FAILURE_PROB", "0"))


def _poll_with_failure_injection(pollers) -> List[int]:
    # 中译：对一组 poller 逐个取传输状态。若开启了故障注入（概率>0），
    #       则以该概率把状态强制改成 Failed，用于测试；否则原样返回各 poller.poll() 的结果。
    if (failure_prob := _get_failure_prob()) > 0:
        return [
            int(KVPoll.Failed) if random.random() < failure_prob else int(poller.poll())
            for poller in pollers
        ]
    return [int(poller.poll()) for poller in pollers]


def _is_fake_transfer(req: Req, server_args: ServerArgs) -> bool:
    # 中译：判断该请求是否走「伪传输」（不做真实 KV 搬运）。两种情形：
    #   1) 请求显式带哨兵 bootstrap_host（FAKE_BOOTSTRAP_HOST）；
    #   2) 未带 bootstrap_host 且服务端配置的传输后端就是 fake。
    return req.bootstrap_host == FAKE_BOOTSTRAP_HOST or (
        req.bootstrap_host is None
        and server_args.disaggregation_transfer_backend == "fake"
    )


def _apply_metadata_gate(polls, decode_reqs, metadata_buffers, server_args) -> None:
    """Downgrade Success → Transferring for requests whose metadata hasn't landed.

    Mutates `polls` in-place. Called before all-reduce so that MIN across TP
    ranks naturally prevents any rank from committing before all ranks are ready.

    中译：「元数据门闸」。KV 数据传完不代表随行元数据（首 token、logprob、bootstrap_room 等）
          也已落地；本函数把「KV 已 Success 但元数据尚未到达」的请求状态回退为 Transferring，
          避免过早提交。就地修改 polls，且在 all_reduce(MIN) 之前调用——这样只要有一个 rank
          还没就绪，取 MIN 就会让全体保持未完成，天然保证跨 TP rank 的一致提交。
          判定依据：decode 侧元数据缓冲里的 bootstrap_room 是否已被写入（==0 视为未到达）；
          伪传输（fake）请求不参与该门闸。
    """
    for i, poll_val in enumerate(polls):
        if poll_val == int(KVPoll.Success):
            decode_req = decode_reqs[i]
            if _is_fake_transfer(decode_req.req, server_args):
                continue
            actual_room = metadata_buffers.bootstrap_room[
                decode_req.metadata_buffer_index, 0
            ].item()
            if actual_room == 0:
                polls[i] = int(KVPoll.Transferring)


def poll_and_all_reduce(
    pollers,
    gloo_group: dist.ProcessGroup,
    decode_reqs=None,
    metadata_buffers: Optional[MetadataBuffers] = None,
    server_args: Optional[ServerArgs] = None,
):
    # 中译：轮询一组传输状态并在进程组内做 all_reduce(MIN) 对齐。
    #       返回对齐后的状态列表：某请求只有当所有 rank 都 Success 时，其 MIN 结果才是 Success。
    # at a certain prob, the poll is failed to simulate failure
    # 中译：先取各 poller 状态（可能按概率注入 Failed）。
    polls = _poll_with_failure_injection(pollers)

    # Apply metadata gate on the decode requests to downgrade Success → Transferring for requests whose metadata hasn't landed.
    # 中译：若提供了 decode 请求与元数据缓冲，则先过一遍元数据门闸（见 _apply_metadata_gate）。
    if (
        decode_reqs is not None
        and metadata_buffers is not None
        and server_args is not None
    ):
        _apply_metadata_gate(polls, decode_reqs, metadata_buffers, server_args)
    # 中译：把状态转成 uint8 张量，用 MIN 归约。KVPoll 的取值需保证「越小越未完成」，
    #       从而 MIN 天然表达「木桶效应」——以最慢 rank 为准。
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()


def poll_and_all_reduce_attn_cp_tp_group(
    pollers,
    attn_cp_cpu_group: dist.ProcessGroup,
    attn_tp_cpu_group: dist.ProcessGroup,
):
    # 中译：在「注意力 TP × 注意力 CP」两级进程组上分两步对齐状态，用于同时开启 TP 与
    #       上下文并行（CP）的场景，确保一个 DP 分片内所有 TP×CP 参与者看到一致的状态迁移。
    # First sync across attn-tp ranks so all TP participants for a given (dp, cp)
    # shard observe the same status transitions.
    # 中译：第一步——先在 attn-tp 组内对齐，使同一 (dp, cp) 分片下的所有 TP 成员状态一致。
    polls = poll_and_all_reduce(pollers, attn_tp_cpu_group)

    # Then sync across attn-cp ranks, so all TPxCP participants in one DP shard
    # converge to the same global status.
    # 中译：第二步——再在 attn-cp 组内对齐，使一个 DP 分片内的所有 TP×CP 成员收敛到同一全局状态。
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(
        tensor_to_reduce,
        op=dist.ReduceOp.MIN,
        group=attn_cp_cpu_group,
    )
    return tensor_to_reduce.tolist()


def poll_and_all_reduce_with_staging(
    decode_reqs,
    staging_handler,
    gloo_group: dist.ProcessGroup,
    metadata_buffers: Optional[MetadataBuffers] = None,
    server_args: Optional[ServerArgs] = None,
):
    """Staging-aware polling: advance scatter, demote incomplete transfers, all_reduce.

    中译：面向「暂存缓冲（staging buffer）」路径的轮询版本。当 prefill 与 decode 的 TP 尺寸
          不同、需要经暂存缓冲做 gather→整块 RDMA→scatter 时使用。三件事：
          ① 对需要 staging 且尚未完成的请求推进 scatter（把暂存区数据散射到目标 KV 槽）；
          ② 取传输状态，并把「KV 报 Success 但 scatter 还没做完」的请求回退为 Transferring；
          ③ 再过元数据门闸，最后 all_reduce(MIN) 跨 rank 对齐。
    """
    # 中译：① 对每个「需要 staging 且未完成」的请求推进一步 scatter。
    for decode_req in decode_reqs:
        if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
            decode_req
        ):
            staging_handler.advance_scatter(decode_req)

    # allow test injection of failure probability at runtime
    # 中译：② 取各 receiver 状态（允许运行时注入故障概率）。
    receivers = [dr.kv_receiver for dr in decode_reqs]
    raw_polls = _poll_with_failure_injection(receivers)
    # 中译：即便 KV 已 Success，只要 scatter 尚未完成，就把状态回退为 Transferring，防止过早提交。
    for i, decode_req in enumerate(decode_reqs):
        if raw_polls[i] == int(KVPoll.Success):
            if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
                decode_req
            ):
                raw_polls[i] = int(KVPoll.Transferring)
    # Apply metadata gate on the decode requests to downgrade Success → Transferring for requests whose metadata hasn't landed.
    # 中译：③ 元数据门闸 + all_reduce(MIN) 对齐（同 poll_and_all_reduce）。
    if metadata_buffers is not None and server_args is not None:
        _apply_metadata_gate(raw_polls, decode_reqs, metadata_buffers, server_args)
    poll_tensor = torch.tensor(raw_polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(poll_tensor, op=dist.ReduceOp.MIN, group=gloo_group)
    return poll_tensor.tolist()


#########################
# Metadata Buffers
# 中译：元数据缓冲区。除了 KV 本体，prefill 还需把「首个输出 token 及其随行元数据」
#       （包括缓存 token 数、logprob、投机解码的 topk / 隐藏态、用于校验的 bootstrap_room 等）
#       传给 decode。这些字段预先分配为固定形状的缓冲张量，每个请求占一个行下标，供 RDMA 搬运。
#########################


class ReqToMetadataIdxAllocator:
    """A memory pool that maps a request to its first output token location.

    中译：一个简单的索引分配器，把请求映射到元数据缓冲里的一个行位置（存放其首个输出 token）。
          本质是定长槽位池：用双端队列维护空闲槽，alloc/free 均为 O(1)。
    """

    def __init__(
        self,
        size: int,
    ):
        # 中译：size 为可容纳的并发请求上限，初始时所有槽位 [0, size) 均空闲。
        self.size = size
        self.free_slots = deque(list(range(size)))

    def available_size(self):
        # 中译：当前可用（空闲）槽位数量。
        return len(self.free_slots)

    def alloc(self) -> Optional[int]:
        # 中译：分配一个空闲槽位下标；无可用则返回 None（由调用方决定排队或拒绝）。
        if len(self.free_slots) == 0:
            return None

        return self.free_slots.popleft()

    def free(self, free_index: int):
        # 中译：归还槽位下标，供后续请求复用。
        self.free_slots.append(free_index)


class MetadataBuffers:
    def __init__(
        self,
        size: int,
        hidden_size: int,
        hidden_states_dtype: torch.dtype,
        max_top_logprobs_num: int = 128,
        custom_mem_pool: torch.cuda.MemPool = None,
    ):
        self.custom_mem_pool = custom_mem_pool
        bootstrap_room_dtype = torch.uint64
        device = "cpu"
        if is_npu():
            # For ascend backend, output tokens are placed in the NPU and will be transferred by D2D channel.
            device = "npu"
            # TODO: Fix me when npu backend supports torch.uint64
            bootstrap_room_dtype = torch.int64
        elif self.custom_mem_pool:
            # TODO(shangming): Fix me (use 'cuda') when nvlink_transport of Mooncake is bug-free
            device = "cpu"
        elif envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() == "INTRA_NODE_NVLINK":
            device = "cuda"
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            # TODO: abort top_logprobs_num > 128 in PD

            # We transfer the metadata of first output token to decode
            # The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes
            # 中译：需把首个输出 token 的元数据传给 decode。RDMA 传输的最小粒度是 64 字节，
            #       因此这里统一把每行 padding 到＞64B（如 (size,16) 的 int32 = 64B）。
            self.output_ids = torch.zeros((size, 16), dtype=torch.int32, device=device)
            self.cached_tokens = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_token_logprobs_val = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_token_logprobs_idx = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_top_logprobs_val = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.float32, device=device
            )
            self.output_top_logprobs_idx = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.int32, device=device
            )
            # For PD + spec decode
            self.output_topk_p = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_topk_index = torch.zeros(
                (size, 16), dtype=torch.int64, device=device
            )
            self.output_hidden_states = torch.zeros(
                (size, hidden_size), dtype=hidden_states_dtype, device=device
            )
            # Request validation: store bootstrap_room to detect metadata corruption
            self.bootstrap_room = torch.zeros(
                (size, 8), dtype=bootstrap_room_dtype, device=device
            )

    def get_buf_infos(self):
        ptrs = [
            self.output_ids.data_ptr(),
            self.cached_tokens.data_ptr(),
            self.output_token_logprobs_val.data_ptr(),
            self.output_token_logprobs_idx.data_ptr(),
            self.output_top_logprobs_val.data_ptr(),
            self.output_top_logprobs_idx.data_ptr(),
            self.output_topk_p.data_ptr(),
            self.output_topk_index.data_ptr(),
            self.output_hidden_states.data_ptr(),
            self.bootstrap_room.data_ptr(),
        ]
        data_lens = [
            self.output_ids.nbytes,
            self.cached_tokens.nbytes,
            self.output_token_logprobs_val.nbytes,
            self.output_token_logprobs_idx.nbytes,
            self.output_top_logprobs_val.nbytes,
            self.output_top_logprobs_idx.nbytes,
            self.output_topk_p.nbytes,
            self.output_topk_index.nbytes,
            self.output_hidden_states.nbytes,
            self.bootstrap_room.nbytes,
        ]
        item_lens = [
            self.output_ids[0].nbytes,
            self.cached_tokens[0].nbytes,
            self.output_token_logprobs_val[0].nbytes,
            self.output_token_logprobs_idx[0].nbytes,
            self.output_top_logprobs_val[0].nbytes,
            self.output_top_logprobs_idx[0].nbytes,
            self.output_topk_p[0].nbytes,
            self.output_topk_index[0].nbytes,
            self.output_hidden_states[0].nbytes,
            self.bootstrap_room[0].nbytes,
        ]
        return ptrs, data_lens, item_lens

    def get_buf(self, idx: int):
        return (
            self.output_ids[idx].clone(),
            self.cached_tokens[idx].clone(),
            self.output_token_logprobs_val[idx].clone(),
            self.output_token_logprobs_idx[idx].clone(),
            self.output_top_logprobs_val[idx].clone(),
            self.output_top_logprobs_idx[idx].clone(),
            self.output_topk_p[idx].clone(),
            self.output_topk_index[idx].clone(),
            self.output_hidden_states[idx].clone(),
            self.bootstrap_room[idx].clone(),
        )

    def set_buf(self, req: Req):
        # 中译：prefill 侧调用：把请求的首 token、缓存 token 统计（总/GPU/Host/Storage 四级）、
        #       （可选）logprob、（PD+投机解码时的）topk 与隐藏态，以及用于校验的 bootstrap_room
        #       写入该请求对应的元数据缓冲行，随后由传输后端搬运给 decode。
        self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
        self.cached_tokens[req.metadata_buffer_index][0] = req.cached_tokens
        self.cached_tokens[req.metadata_buffer_index][1] = req.cached_tokens_device
        self.cached_tokens[req.metadata_buffer_index][2] = req.cached_tokens_host
        self.cached_tokens[req.metadata_buffer_index][3] = req.cached_tokens_storage
        if req.return_logprob:
            if req.logprob.output_token_logprobs_val:  # not none or empty list
                self.output_token_logprobs_val[req.metadata_buffer_index][0] = (
                    req.logprob.output_token_logprobs_val[0]
                )
            if req.logprob.output_token_logprobs_idx:  # not none or empty list
                self.output_token_logprobs_idx[req.metadata_buffer_index][0] = (
                    req.logprob.output_token_logprobs_idx[0]
                )

            if req.logprob.output_top_logprobs_val:  # not none or empty list
                self.output_top_logprobs_val[req.metadata_buffer_index][
                    : len(req.logprob.output_top_logprobs_val[0])
                ] = torch.tensor(
                    req.logprob.output_top_logprobs_val[0],
                    dtype=torch.float32,
                    device="cpu",
                )
            if req.logprob.output_top_logprobs_idx:  # not none or empty list
                self.output_top_logprobs_idx[req.metadata_buffer_index][
                    : len(req.logprob.output_top_logprobs_idx[0])
                ] = torch.tensor(
                    req.logprob.output_top_logprobs_idx[0],
                    dtype=torch.int32,
                    device="cpu",
                )
        # For PD + spec decode
        if req.hidden_states_tensor is not None:
            # speculative_eagle_topk should not be greater than 16 currently
            topk = req.output_topk_p.size(0)

            self.output_topk_p[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_p
            )
            self.output_topk_index[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_index
            )
            self.output_hidden_states[req.metadata_buffer_index].copy_(
                req.hidden_states_tensor
            )
        # Store bootstrap_room for validation on decode side
        # 中译：存入 bootstrap_room 供 decode 侧校验——搬运后如果该值与预期不符，则说明元数据
        #       尚未落地或发生错位（配合 _apply_metadata_gate 使用，==0 视为未到达）。
        self.bootstrap_room[req.metadata_buffer_index, 0] = (
            req.bootstrap_room if req.bootstrap_room is not None else 0
        )


#########################
# Transfer Backend
# 中译：传输后端。PD 分离支持多种 KV 搬运实现（Mooncake / Mori / NIXL / Ascend / Fake），
#       下面的工厂函数 get_kv_class 根据后端类型返回对应的一组实现类，实现后端可插拔。
#########################


class TransferBackend(Enum):
    # 中译：可选的 KV 传输后端类型。fake 为测试/暖机用的空实现。
    MOONCAKE = "mooncake"
    MORI = "mori"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"


class KVClassType(Enum):
    # 中译：同一传输后端下的四类角色：
    #   KVARGS           —— KV 参数描述（指针/长度/TP 信息等）；
    #   MANAGER          —— 传输管理器（维护连接、队列、状态）；
    #   SENDER / RECEIVER—— prefill 侧发送 / decode 侧接收；
    #   BOOTSTRAP_SERVER —— 握手引导服务，用于 P/D 互相发现。
    KVARGS = "kvargs"
    MANAGER = "manager"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"


@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.KVARGS]
) -> Type[KVArgs]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.MANAGER]
) -> Type[CommonKVManager]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.SENDER]
) -> Type[CommonKVSender]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.RECEIVER]
) -> Type[CommonKVReceiver]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.BOOTSTRAP_SERVER]
) -> Type[CommonKVBootstrapServer]: ...


def get_kv_class(
    transfer_backend: TransferBackend, class_type: KVClassType
) -> Optional[Type]:
    # 中译：KV 类工厂：根据（传输后端, 角色类型）返回对应的实现类。
    #       采用延迟导入（函数内 import），避免未安装某后端依赖时在模块加载阶段就报错。

    if transfer_backend == TransferBackend.MOONCAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mooncake import (
            MooncakeKVBootstrapServer,
            MooncakeKVManager,
            MooncakeKVReceiver,
            MooncakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MooncakeKVManager,
            KVClassType.SENDER: MooncakeKVSender,
            KVClassType.RECEIVER: (MooncakeKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.MORI:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mori import (
            MoriKVBootstrapServer,
            MoriKVManager,
            MoriKVReceiver,
            MoriKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MoriKVManager,
            KVClassType.SENDER: MoriKVSender,
            KVClassType.RECEIVER: (MoriKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MoriKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.ASCEND:
        from sglang.srt.disaggregation.ascend import (
            AscendKVBootstrapServer,
            AscendKVManager,
            AscendKVReceiver,
            AscendKVSender,
        )
        from sglang.srt.disaggregation.base import KVArgs

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: AscendKVManager,
            KVClassType.SENDER: AscendKVSender,
            KVClassType.RECEIVER: (AscendKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: AscendKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.NIXL:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.nixl import (
            NixlKVBootstrapServer,
            NixlKVManager,
            NixlKVReceiver,
            NixlKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: NixlKVManager,
            KVClassType.SENDER: NixlKVSender,
            KVClassType.RECEIVER: (NixlKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: NixlKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.FAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.fake import (
            FakeKVManager,
            FakeKVReceiver,
            FakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: FakeKVManager,
            KVClassType.SENDER: FakeKVSender,
            KVClassType.RECEIVER: (FakeKVReceiver),
        }
        return class_mapping.get(class_type)

    # 中译：未知传输后端，直接报错。
    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


def page_indices_to_cp_rank_page_indices(
    page_indices: np.ndarray,
    total_pages: int,
    cp_rank: int,
    cp_size: int,
) -> np.ndarray:
    """
    Filter page_indices (which are *global* page ids in the KV pool) to those
    belonging to the given CP rank for this request.

    For a single request, its pages occupy a contiguous global range
    [first_page, first_page + total_pages). We first compute the local
    split [0, total_pages) across cp_size ranks, then shift that local
    range by first_page back into the global page id space and take
    the intersection with page_indices.

    Returns:
        Subset of page_indices that fall in this rank's global
        [start_page, end_page) slice for the given CP rank.

    中译：在开启上下文并行（CP）时，一个请求的 KV 页会按 CP rank 切分到不同卡上。
          本函数从全局页 id 列表中筛出属于指定 cp_rank 的那一段。
          前提：单个请求的页占据一段连续的全局区间 [first_page, first_page+total_pages)。
          做法：先在本地区间 [0, total_pages) 上按 cp_size 均匀切分（余数优先分给靠前的 rank），
                再把本地区间平移 first_page 回到全局页 id 空间，取交集。
    """
    # 中译：未开 CP（cp_size<=1）则无需切分，原样返回。
    if cp_size <= 1:
        return page_indices

    # 中译：空页列表直接返回。
    if page_indices.size == 0:
        return np.asarray(page_indices)

    # 中译：first_page 为该请求页区间的起点；base/rem 为均分的商与余数。
    first_page = int(page_indices.min())
    base = total_pages // cp_size
    rem = total_pages % cp_size

    # 中译：计算本 rank 在本地坐标下的 [local_start, local_end)。
    #       能整除时每 rank 均分 base 页；否则前 rem 个 rank 各多分 1 页。
    if rem == 0:
        local_start = cp_rank * base
        local_end = local_start + base
    else:
        local_start = cp_rank * base + min(cp_rank, rem)
        n_pages = base + (1 if cp_rank < rem else 0)
        local_end = local_start + n_pages

    # Map back to global page ids.
    # 中译：把本地区间平移回全局页 id 空间，再用掩码取出落在该区间内的页。
    start_page = first_page + local_start
    end_page = first_page + local_end

    mask = (page_indices >= start_page) & (page_indices < end_page)
    return np.asarray(page_indices)[mask]


def filter_kv_indices_for_cp_rank(
    kv_mgr: CommonKVManager, kv_indices: np.ndarray, index_slice: slice
) -> Tuple[np.ndarray, slice]:
    """Filters kv_indices and index_slice for the current CP rank.

    中译：把一个请求的 kv_indices 及其对应的 index_slice，裁剪为当前 CP rank 实际拥有的那一段。
          返回（裁剪后的 kv_indices, 对应的子 slice）；若本 rank 不拥有任何页，则返回空切片。
    """
    total_pages = len(kv_indices)
    cp_rank = kv_mgr.attn_cp_rank
    cp_size = kv_mgr.attn_cp_size

    # 中译：先算出本 rank 拥有的全局页 id 子集。
    rank_page_indices = page_indices_to_cp_rank_page_indices(
        page_indices=kv_indices,
        total_pages=total_pages,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )

    # 中译：本 rank 无页，返回空的 kv_indices 与零长度 slice。
    if rank_page_indices.size == 0:
        new_kv_indices = kv_indices[:0]
        new_index_slice = slice(index_slice.start, index_slice.start)
    else:
        # 中译：用掩码定位属于本 rank 的页在 kv_indices 中的位置。
        mask = np.isin(kv_indices, rank_page_indices)
        if not mask.any():
            new_kv_indices = kv_indices[:0]
            new_index_slice = slice(index_slice.start, index_slice.start)
        else:
            # 中译：因为本 rank 的页在全局上连续，取掩码的首/尾 True 位置即得连续区间，
            #       并同步把 index_slice 也平移、裁剪到同一子区间，保证与 kv_indices 对齐。
            first_pos = int(mask.argmax())
            last_pos = len(mask) - int(mask[::-1].argmax())

            new_kv_indices = kv_indices[first_pos:last_pos]
            new_index_slice = slice(
                index_slice.start + first_pos,
                index_slice.start + last_pos,
            )
    return new_kv_indices, new_index_slice


#########################
# Misc
# 中译：杂项工具。包括：MLA 后端判定、向 KVArgs 追加状态组件（兼容 SWA/Mamba/DSA 等异构 KV）、
#       以及请求中止相关的辅助函数。
#########################


def is_mla_backend(target_kv_pool) -> bool:
    # 中译：判断目标 KV 池是否为 MLA（多头潜在注意力，如 DeepSeek 系列）类型，
    #       传输时 MLA 与常规 MHA 的 KV 布局/切分方式不同。
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return isinstance(target_kv_pool, (MLATokenToKVPool, DeepSeekV4TokenToKVPool))


def append_state_component(
    kv_args: KVArgs,
    state_type: StateType,
    data_ptrs: List[int],
    data_lens: List[int],
    item_lens: List[int],
    dim_per_tensor: Optional[List[int]] = None,
) -> None:
    """Append one state component. Caller orders state_types consistently
    on prefill and decode sides.

    中译：向 kv_args 追加一个「状态组件」。除 KV 外，有些模型还有需随 KV 一同传输的额外状态（
          如滞窗注意力 SWA 环形缓冲、Mamba 卷积/时序状态、DSA 状态等）。这里把各状态的
          类型、数据指针、总长、单项长、（可选）每张量维度分别追加到平行列表中。
          关键约束：调用方必须在 prefill 与 decode 两侧以相同顺序追加，以保证两侧状态一一对应。
    """
    kv_args.state_types.append(state_type)
    kv_args.state_data_ptrs.append(data_ptrs)
    kv_args.state_data_lens.append(data_lens)
    kv_args.state_item_lens.append(item_lens)
    kv_args.state_dim_per_tensor.append(dim_per_tensor or [])


def setup_state_kv_args(
    kv_args: KVArgs,
    token_to_kv_pool,
    draft_token_to_kv_pool=None,
    total_kv_layers: int = None,
    req_to_token_pool=None,
) -> None:
    """Populate ``kv_args`` state-buffer fields from the given pool.
    Shared by prefill and decode bootstrap paths so the state_type dispatch
    lives in one place.

    中译：根据给定的 KV 池类型，填充 kv_args 里的「状态缓冲」字段。prefill 与 decode 的 bootstrap
          路径共用本函数，把「各种 KV 池 → 对应 StateType」的分发逻辑集中在一处，保证两侧一致。
          支持的异构状态：SWA / SWA_RING（滞窗注意力）、MAMBA（线性注意力）、DSA（DeepSeek 稀疏注意力）。
    """
    from sglang.srt.disaggregation.base.conn import StateType
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool

    # 中译：先清空各状态列表，再按池类型逐一填充。
    kv_args.state_types = []
    kv_args.state_data_ptrs = []
    kv_args.state_data_lens = []
    kv_args.state_item_lens = []
    kv_args.state_dim_per_tensor = []

    # 中译：若 KV 池提供了 get_state_buf_infos，说明有需随行传输的状态缓冲，取出其指针/长度信息。
    if hasattr(token_to_kv_pool, "get_state_buf_infos"):
        data_ptrs, data_lens, item_lens = token_to_kv_pool.get_state_buf_infos()

        # DeepSeekV4TokenToKVPool inherits BaseSWAKVPool; its heterogeneous
        # state list is described per-entry via get_state_buf_infos.
        # 中译：SWA 类池：追加 SWA 状态；若是 unified_kv（滞窗环形缓冲内嵌在统一缓冲、
        #       按行寻址，无单独 swa_kv_pool），则额外以 SWA_RING 形式追加环形缓冲。
        if isinstance(token_to_kv_pool, BaseSWAKVPool):
            append_state_component(
                kv_args, StateType.SWA, data_ptrs, data_lens, item_lens
            )
            # unified_kv: the SWA ring lives in the unified buffers (no separate
            # swa_kv_pool) and is addressed per-row, so ship it as SWA_RING.
            if getattr(token_to_kv_pool, "_unified_kv", False) and hasattr(
                token_to_kv_pool, "get_unified_swa_ring_buf_infos"
            ):
                ring_ptrs, ring_lens, ring_item_lens = (
                    token_to_kv_pool.get_unified_swa_ring_buf_infos()
                )
                if ring_ptrs:
                    append_state_component(
                        kv_args,
                        StateType.SWA_RING,
                        ring_ptrs,
                        ring_lens,
                        ring_item_lens,
                    )
        # 中译：混合线性池（如 Mamba 系）：追加 MAMBA 状态，并附带每张量维度以支持 TP 尺寸不同时的切片。
        elif isinstance(token_to_kv_pool, HybridLinearKVPool):
            dim = (
                token_to_kv_pool.get_state_dim_per_tensor()
                if hasattr(token_to_kv_pool, "get_state_dim_per_tensor")
                else None
            )
            append_state_component(
                kv_args, StateType.MAMBA, data_ptrs, data_lens, item_lens, dim
            )
        # 中译：DSA / NPU-MLA 池：如带 draft KV 池（投机解码）则先并入其状态缓冲；
        #       NPU-MLA 走特殊分组（kv_buf_groups + total_kv_layers），其余情形追加 DSA 状态。
        elif isinstance(token_to_kv_pool, (DSATokenToKVPool, NPUMLATokenToKVPool)):
            if draft_token_to_kv_pool is not None and isinstance(
                draft_token_to_kv_pool, DSATokenToKVPool
            ):
                (
                    draft_data_ptrs,
                    draft_data_lens,
                    draft_item_lens,
                ) = draft_token_to_kv_pool.get_state_buf_infos()
                data_ptrs = data_ptrs + draft_data_ptrs
                data_lens = data_lens + draft_data_lens
                item_lens = item_lens + draft_item_lens
            if isinstance(token_to_kv_pool, NPUMLATokenToKVPool):
                kv_args.kv_buf_groups = (
                    len(kv_args.kv_data_ptrs) // token_to_kv_pool.layer_num
                )
                kv_args.total_kv_layers = total_kv_layers
            else:
                append_state_component(
                    kv_args, StateType.DSA, data_ptrs, data_lens, item_lens
                )

    if (
        StateType.MAMBA not in kv_args.state_types
        and req_to_token_pool is not None
        and hasattr(req_to_token_pool, "get_state_buf_infos")
    ):
        data_ptrs, data_lens, item_lens = req_to_token_pool.get_state_buf_infos()
        if data_ptrs:
            dim = (
                req_to_token_pool.get_state_dim_per_tensor()
                if hasattr(req_to_token_pool, "get_state_dim_per_tensor")
                else None
            )
            append_state_component(
                kv_args, StateType.MAMBA, data_ptrs, data_lens, item_lens, dim
            )


def prepare_abort(req: Req, error_message: str, status_code=None):
    # 中译：把请求标记为「中止」。设置其 finished_reason 为 FINISH_ABORT（携带错误信息与状态码），
    #       以便后续能把错误结果回流给客户端；若请求要求返回 logprob，则清空各 logprob 列表避免脏数据。
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    # populate finish metadata and stream output
    req.finished_reason = FINISH_ABORT(error_message, status_code)

    if req.return_logprob:
        req.logprob.input_token_logprobs_val = []
        req.logprob.input_token_logprobs_idx = []
        req.logprob.input_top_logprobs_val = []
        req.logprob.input_top_logprobs_idx = []
        req.logprob.input_token_ids_logprobs_val = []
        req.logprob.input_token_ids_logprobs_idx = []


def is_aborted(req: Req) -> bool:
    # 中译：判断请求是否已被中止：待结束标记 to_finish 或已定结束原因 finished_reason
    #       任一为 FINISH_ABORT 即视为中止。
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    return isinstance(req.to_finish, FINISH_ABORT) or isinstance(
        req.finished_reason, FINISH_ABORT
    )
