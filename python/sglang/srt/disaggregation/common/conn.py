# 中译：PD 分离（Prefill/Decode 分离部署）下 KV cache 跨实例传输的
#       「后端无关」通用实现。本文件抽出 Mooncake / NIXL / Mori / Ascend
#       等传输后端共享的连接管理与元数据交换逻辑，四个核心类分工如下：
#         - CommonKVManager        每个 rank 一个，管理连接、状态、并行拓扑映射；
#                                  Prefill 侧向 bootstrap server 注册自身地址，
#                                  Decode 侧从 bootstrap server 拉取拓扑并做心跳。
#         - CommonKVSender         Prefill 侧「每请求」对象，负责发送 KV cache。
#         - CommonKVReceiver       Decode 侧「每请求」对象，负责拉取/接收 KV cache。
#         - CommonKVBootstrapServer 仅 Prefill 实例启动的轻量 aiohttp HTTP 服务，
#                                  充当 rendezvous / 服务发现的元数据交换点。
#       注意：真正的 KV 数据面是 Prefill<->Decode 点对点直连（各后端子类实现），
#             bootstrap server 只负责「牵线」，不搬运 KV 数据。
from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import requests
import torch.distributed as dist
import zmq
from aiohttp import web

from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer,
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
    KVTransferMetric,
    StateType,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    filter_kv_indices_for_cp_rank,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_dp_size,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_parallel,
    get_serving,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import (
    NetworkAddress,
    get_local_ip_auto,
    get_zmq_socket_on_host,
)

logger = logging.getLogger(__name__)


# Reuse a keep-alive session per bootstrap_addr for decode-side bootstrap queries
# so we don't open a fresh TCP connection per query (that churns short-lived
# sockets and can exhaust ephemeral ports under high concurrency). Thread-local
# because requests.Session is not safe for concurrent cross-thread use.
_bootstrap_sessions = threading.local()


def _get_bootstrap_session(bootstrap_addr: str) -> requests.Session:
    sessions = getattr(_bootstrap_sessions, "by_addr", None)
    if sessions is None:
        sessions = {}
        _bootstrap_sessions.by_addr = sessions
    session = sessions.get(bootstrap_addr)
    if session is None:
        # Not evicted: the number of bootstrap_addr is bounded (one per prefill
        # server). Bootstrap is http-only, so only http:// is mounted.
        session = requests.Session()
        session.mount(
            "http://", requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
        )
        sessions[bootstrap_addr] = session
    return session


class KVTransferError(Exception):
    """KV 传输失败异常。

    携带 bootstrap_room（请求在 PD 两侧的唯一关联 id）与失败原因；
    is_from_another_rank 标记该失败是否由同实例其它 rank 上报（用于区分
    本 rank 自身失败还是被其它 rank 传染，便于日志与状态处理）。
    """

    def __init__(
        self,
        bootstrap_room: int,
        failure_reason: str,
        is_from_another_rank: bool = False,
    ):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason
        self.is_from_another_rank = is_from_another_rank

    def __str__(self):
        return f"KVTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


@dataclasses.dataclass
class PrefillServerInfo:
    """Decode 侧缓存的某个 Prefill 实例的并行拓扑与派生的 rank 映射。

    前半部分（拓扑字段）由 Decode 端通过 `GET /route`（哨兵查询）从 bootstrap
    server 拉取；后半部分（target_* / required_*）由 `_resolve_rank_mapping`
    在 Decode 端本地计算并回填，描述「本 Decode rank 应向哪些 Prefill rank
    取 KV、需要多少路响应」。同一 (bootstrap_addr, decode 引擎) 组合下结果确定。
    """

    # Topology fields (fetched from bootstrap server)
    # 中译：以下为从 bootstrap server 拉取的 Prefill 端并行拓扑信息。
    attn_tp_size: int
    attn_cp_size: int
    dp_size: int
    pp_size: int
    page_size: Optional[int]
    kv_cache_dtype: Optional[str]
    follow_bootstrap_room: bool
    enable_dsa_cache_layer_split: bool = False

    # PD true-retraction rebootstrap: the prefill's HTTP API port. The decode
    # already knows the prefill host (the bootstrap_addr host), so it can POST
    # /generate to http://{bootstrap_host}:{prefill_http_port} to trigger a KV
    # recompute -- no router-injected pd_rebootstrap_prefill_url needed.
    prefill_http_port: Optional[int] = None

    # Pre-computed rank mapping (set by try_ensure_parallel_info on decode side)
    target_tp_rank: Optional[int] = None
    target_tp_ranks: Optional[List[int]] = None
    target_cp_ranks: Optional[List[int]] = None
    target_pp_ranks: Optional[List[int]] = None
    required_dst_info_num: Optional[int] = None
    required_prefill_response_num: Optional[int] = None

    def __post_init__(self):
        self.attn_tp_size = int(self.attn_tp_size)
        self.attn_cp_size = int(self.attn_cp_size)
        self.dp_size = int(self.dp_size)
        self.pp_size = int(self.pp_size)
        self.page_size = int(self.page_size) if self.page_size is not None else None
        self.kv_cache_dtype = (
            str(self.kv_cache_dtype) if self.kv_cache_dtype is not None else None
        )
        self.follow_bootstrap_room = bool(self.follow_bootstrap_room)
        self.enable_dsa_cache_layer_split = bool(self.enable_dsa_cache_layer_split)
        self.prefill_http_port = (
            int(self.prefill_http_port) if self.prefill_http_port is not None else None
        )


@dataclasses.dataclass
class PrefillRankInfo:
    """单个 Prefill rank 的 KV 传输端点（IP+端口）。

    由该 rank 在启动时 PUT 注册到 bootstrap server 的 prefill_port_table 中，
    供 Decode 端按 (dp, cp, tp, pp) 索引查询后建立点对点连接。
    """

    rank_ip: str
    rank_port: int

    def __post_init__(self):
        self.rank_ip = str(self.rank_ip)
        self.rank_port = int(self.rank_port)


class CommonKVManager(BaseKVManager):
    """KV 传输管理器（后端无关基类），每个 rank（TP/CP/DP/PP）一个实例。

    职责：
      - 统一读取当前 rank 的并行坐标（attn tp/cp/dp、system dp、pp）；
      - 绑定用于 KV 传输元数据/控制信令的 ZMQ PULL socket；
      - PREFILL 角色：向 bootstrap server 注册本 rank 的 (ip, port) 与拓扑；
      - DECODE 角色：维护连接池、心跳检测、拓扑缓存与失败处理。
    子类（Mooncake/NIXL/Mori/Ascend）在此基础上实现具体的数据面传输。
    """

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        # 中译：缓存 KV 参数与预算每条/每组数据项的字节长度总和（供传输量统计用）。
        # 中译：KVArgs——包含 KV pool 显存地址/长度、每项字节数、head 数、page size、
        #       state 元数据等底层传输所需的全部参数。
        self.kv_args = args
        # 中译：单个 token 的 KV cache 总字节数（各层/各分量 item_len 之和），用于把
        self.kv_cache_dtype_str = args.kv_cache_dtype_str
        self.kv_item_lens_sum = sum(args.kv_item_lens)
        # 中译：单个 token 的特殊 state（Mamba/SWA/DSA 等）总字节数，同样用于字节数统计。
        self.state_item_lens_sum = sum(x for comp in args.state_item_lens for x in comp)
        # 中译：是否为 MLA 后端（DeepSeek 类模型）。MLA 的 KV 组织方式不同，影响传输布局。
        self.is_mla_backend = is_mla_backend
        # 中译：本实例角色——PREFILL 或 DECODE，决定 __init__ 走哪条初始化分支。
        # Per-sender fan-out of a KV copy onto N decode destinations
        # (MLA under Prefill-CP + Decode-TP, or decode_tp > prefill_tp).
        # MLA is resolved lazily at bootstrap (see resolve_kv_replica_factor);
        # MHA never replicates, so it stays pinned at 1.
        self._kv_replica_factor: Optional[int] = None if is_mla_backend else 1
        self.is_hybrid_mla_backend = getattr(args, "is_hybrid_mla_backend", False)
        self.disaggregation_mode = disaggregation_mode
        # 中译：服务端启动参数（并行规模、超时、负载均衡策略等），后续多处从中读取配置。
        self.server_args = server_args
        self.enable_deferred_decode_kv_release = (
            envs.SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE.get()
        )
        self._dcp_pack_buffers = None
        # for p/d multi node infer
        # 中译：bootstrap server 的 host（PD 双侧建立关联的控制面服务地址）。
        # 中译：bootstrap server 的端口（多机 prefill 下会在后面跨节点同步为 leader 端口）。
        # 中译：分布式初始化地址（rank 0 地址），多机注册时以它为目标。
        # 中译：以下为当前 rank 在各并行维度下的坐标（size=该维度总规模，rank=本进程序号）。
        # 中译：attention 张量并行（TP）规模与本 rank 序号。
        # 中译：attention 上下文并行（CP）规模与本 rank 序号。
        # 中译：attention 数据并行（DP）规模与本 rank 序号（用于按 room 路由到正确 dp 组）。
        self.bootstrap_host = get_serving().host
        self.bootstrap_port = get_disagg().disaggregation_bootstrap_port
        self.dist_init_addr = get_parallel().dist_init_addr
        parallel = get_parallel()
        self.attn_tp_size = parallel.attn_tp_size
        self.attn_tp_rank = parallel.attn_tp_rank
        self.attn_cp_size = parallel.attn_cp_size
        self.attn_cp_rank = parallel.attn_cp_rank
        self.dcp_size = parallel.attn_dcp_size
        self.dcp_rank = parallel.attn_dcp_rank
        self.attn_dp_size = get_attention_dp_size()
        self.attn_dp_rank = get_attention_dp_rank()
        # 中译：系统级 DP 规模——启用 dp attention 时视为 1，否则等于 server_args.dp_size。
        self.system_dp_size = (
            1 if get_parallel().enable_dp_attention else get_parallel().dp_size
        )
        # 中译：系统级 DP rank，来自 KVArgs（未提供则默认为 0）。
        self.system_dp_rank = (
            self.kv_args.system_dp_rank if self.kv_args.system_dp_rank else 0
        )
        # 中译：流水线并行（PP）规模与本 rank 序号。
        self.pp_size = get_parallel().pp_size
        self.pp_rank = self.kv_args.pp_rank
        # 中译：本机自动探测到的本地 IP，作为本 rank KV 传输控制通道对外暴露的地址。
        self.local_ip = get_local_ip_auto()
        # 中译：为 True 时所有 CP rank 都参与 KV 传输；否则仅 CP rank 0 发送（其余为 dummy）。
        cp_sharded_prefill = self.attn_cp_size > 1 and (
            self.is_hybrid_mla_backend or get_parallel().enable_dsa_cache_layer_split
        )

        hybrid_decode_pulls_all_ranks = (
            self.is_hybrid_mla_backend
            and disaggregation_mode == DisaggregationMode.DECODE
        )
        self.enable_all_cp_ranks_for_transfer = (
            envs.SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER.get()
            or cp_sharded_prefill
            or hybrid_decode_pulls_all_ranks
        )

        # bind zmq socket
        # 中译：绑定本 rank 的 ZMQ PULL 套接字，自动选取空闲端口；该 (ip, port)
        #       会随后注册到 bootstrap server，作为本 rank 的 KV 传输控制通道地址。
        # 中译：ZMQ 上下文对象，管理本 rank 所有 ZMQ 套接字的生命周期。
        self._zmq_ctx = zmq.Context()
        # 中译：rank_port——自动分配到的 PULL 监听端口；server_socket——接收对端
        # Raise libzmq's per-context socket cap because this manager caches two
        # sockets per decode endpoint, so large fleets can exceed the default.
        self._zmq_ctx.set(
            zmq.MAX_SOCKETS, envs.SGLANG_DISAGGREGATION_ZMQ_MAX_SOCKETS.get()
        )
        self.rank_port, self.server_socket = get_zmq_socket_on_host(
            self._zmq_ctx, zmq.PULL, host=self.local_ip
        )
        logger.debug(f"kv manager bind to {self.local_ip}:{self.rank_port}")

        # 中译：请求状态表——bootstrap_room -> KVPoll 状态（本 rank 上所有在管请求的状态机）。
        self.request_status: Dict[int, KVPoll] = {}
        # 中译：出站 PUSH 套接字缓存——按目标地址复用 ZMQ 连接，避免重复建连。
        self._socket_cache: Dict[str, zmq.Socket] = {}
        # 中译：套接字监控（socket monitor）缓存——按地址复用，用于观察连接事件/断连检测。
        self._monitor_cache: Dict[str, zmq.Socket] = {}
        # 中译：保护上述套接字缓存的并发访问锁（多线程收发时避免竞态）。
        self._socket_send_locks: Dict[str, threading.Lock] = {}
        self._socket_lock = threading.Lock()
        # 中译：失败记录表——bootstrap_room -> 失败原因字符串，供 failure_exception() 回溯。
        self.failure_records: Dict[int, str] = {}
        # 中译：保护失败记录表的并发访问锁。
        self.failure_lock = threading.Lock()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # 中译：PREFILL 角色：同步 leader 端口 -> 向 bootstrap server 注册本 rank
            #       -> 初始化传输信息表与超时阈值。
            # 当 SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER 为 True 时，所有 CP rank
            # 都参与 KV 传输；否则仅 CP rank 0 发送（其余为 dummy）。
            self.is_dummy_cp_rank = (
                not self.enable_all_cp_ranks_for_transfer
                and self.attn_cp_size > 1
                and self.attn_cp_rank != 0
            )
            # 在注册前把 leader 的 bootstrap 端口同步给每个 rank：多节点 prefill 下，
            # 注册目标是 `dist_init_addr`（rank 0），但当启动器为每台主机自动预留
            # 空闲端口时，各 rank 的本地端口可能不同，需先同步统一。
            self.bootstrap_port = self._sync_bootstrap_port_across_nodes(
                self.bootstrap_port
            )
            self.register_to_bootstrap()
            # 中译：传输信息表——bootstrap_room -> 该请求从 Decode 侧收到的传输目标信息
            #       （目标地址、目标 KV 索引、aux 目标位置等），是发送 KV 的依据。
            self.transfer_infos = {}
            # 中译：room -> decode 端已缓存的前缀长度，Prefill 据此跳过前缀、只发送剩余 KV。
            # Deferred KV release: aborted room -> (decode_ip, decode_port);
            # ack held until the transfer drains.
            self._deferred_ack_targets: Dict[int, Tuple[str, int]] = {}
            self.req_to_decode_prefix_len: Dict[int, int] = {}
            # 中译：decode 端 KV 参数注册表——缓存各 Decode 对端注册来的 KVArgs 信息。
            self.decode_kv_args_table = {}
            # 中译：流水线并行进程组（PP group），用于跨 PP rank 的通信/同步。
            self.pp_group = get_pp_group()
            # 若 prefill 侧发生超时，说明 prefill 实例未能从该请求对应的 decode 实例
            # 收到 KV 索引。这些超时请求应被中止以释放 tree cache。
            self.bootstrap_timeout = envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # 中译：DECODE 角色：维护连接池/拓扑缓存/心跳失败计数/HTTP 会话池，
            #       并记录 bootstrap_addr <-> 请求 room 的反向映射，用于节点故障时定位受影响请求。
            # 中译：是否启用 staging（中转缓冲）传输模式，默认关闭，由后续协商开启。
            self.enable_staging: bool = False
            # 中译：连接池——缓存到各 Prefill rank 端点的连接信息（地址/端口），供复用。
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            # 中译：保护连接池的并发访问锁。
            self.connection_lock = threading.Lock()
            # 中译：room -> 该请求需要收齐的 Prefill 响应数（多 rank 取 KV 时的应答计数目标）。
            self.required_prefill_response_num_table: Dict[int, int] = {}
            # 中译：Prefill 拓扑缓存——bootstrap_addr -> PrefillServerInfo（并行拓扑 + rank 映射）。
            self.prefill_info_table: Dict[str, PrefillServerInfo] = {}
            # 中译：心跳失败计数——bootstrap_addr -> 连续失败次数，达到 max_failures 判定对端故障。
            self.heartbeat_failures: Dict[str, int] = {}
            # 中译：HTTP 会话池——按地址复用 requests.Session（连接保活，降低握手开销）。
            self.session_pool: Dict = defaultdict(requests.Session)
            # 中译：保护会话池的并发访问锁。
            self.session_pool_lock = threading.Lock()
            # 中译：地址 -> 该地址上活跃请求 room 集合的反向映射，用于对端故障时快速定位受影响请求。
            self.addr_to_rooms_tracker: Dict[str, Set[int]] = defaultdict(set)
            # 中译：room -> 已收到响应的 Prefill 端点集合，用于判断某请求的 KV 是否已全部取齐。
            self.prefill_response_tracker: Dict[int, Set[int]] = defaultdict(set)
            # Deferred KV release: room -> prefill ranks that acked their transfer
            # drained. Entry exists only while the room is held, so a stale/late
            # ack for a reused bootstrap_room is dropped.
            self._deferred_abort_ack_tracker: Dict[int, Set[int]] = {}
            # Heartbeat interval should be at least 2 seconds
            self.heartbeat_interval = max(
                envs.SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL.get(), 2.0
            )
            # 心跳失败阈值至少为 1。
            self.max_failures = max(
                envs.SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE.get(), 1
            )
            # 若 decode 侧发生超时，说明 decode 实例在 bootstrap 之后未能收到
            # KV Cache 传输完成信号。这些超时请求应被中止以释放 tree cache。
            self.waiting_timeout = envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
            # PD true-retraction rebootstrap: a shared executor + per-thread HTTP
            # sessions used to drive the original prefill worker's ``/generate``
            # endpoint so it recomputes a retracted request's prefix KV under the
            # current weights. Created lazily on first use so deployments that
            # never retract pay nothing.
            self._prefill_recompute_executor: Optional[
                concurrent.futures.ThreadPoolExecutor
            ] = None
            self._prefill_recompute_executor_lock = threading.Lock()
            self._prefill_recompute_sessions = threading.local()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def _should_skip_cp_replicated_state_transfer(self) -> bool:
        """Whether this prefill rank should omit CP-replicated state.

        Prefill CP materializes global token order before writing state pools, so
        every CP rank holds the same state. When all CP ranks transfer their KV
        shards, only rank 0 needs to send that state. Cache layer split is the
        exception because each CP rank owns different state layers.
        """
        return (
            self.attn_cp_size > 1
            and self.attn_cp_rank != 0
            and not get_parallel().enable_dsa_cache_layer_split
        )

    def requires_dcp_relayout(self, dst_dcp_size: int, dst_dcp_rank: int) -> bool:
        if self.dcp_size == dst_dcp_size:
            if self.dcp_rank != dst_dcp_rank:
                raise RuntimeError(
                    "PD peers must connect matching DCP ranks, got "
                    f"prefill={self.dcp_rank}, decode={dst_dcp_rank}"
                )
            return False

        if (
            self.dcp_size == 1
            and dst_dcp_size > 1
            and (self.is_mla_backend or self.is_hybrid_mla_backend)
        ):
            return True

        raise RuntimeError(
            f"Unsupported PD DCP topology: {self.dcp_size} -> {dst_dcp_size}"
        )

    def prepare_dcp_token_item_lens(self, dst_page_item_lens: List[int]) -> List[int]:
        page_size = self.kv_args.page_size
        src_token_lens = [
            item_len // page_size for item_len in self.kv_args.kv_item_lens
        ]
        dst_token_lens = [item_len // page_size for item_len in dst_page_item_lens]
        if src_token_lens != dst_token_lens:
            raise RuntimeError(
                "PD DCP source/destination KV geometry differs: "
                f"src={src_token_lens}, dst={dst_token_lens}"
            )
        return src_token_lens

    def _register_staging_memory(self, ptr: int, size: int) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support staging memory registration"
        )

    def _init_dcp_pack_buffers_once(self, dcp_size: int) -> None:
        if self._dcp_pack_buffers is not None:
            return
        if not self.kv_args.kv_item_lens:
            return
        from sglang.srt.disaggregation.common.dcp_pack import init_dcp_pack_buffers

        self._dcp_pack_buffers = init_dcp_pack_buffers(
            self._register_staging_memory,
            self.kv_args,
            len(self.transfer_queues),
            dcp_size,
        )

    def check_status(self, bootstrap_room: int) -> KVPoll:
        # 中译：返回指定请求（bootstrap_room）当前的 KV 传输轮询状态。
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        # 中译：更新请求状态。状态只能单调推进（取 max），Failed 为终态；
        #       但已被 clear() 清除的 room 不得被迟到的 Failed “复活”，否则会
        #       污染复用同一 bootstrap_room 的未来请求。
        if bootstrap_room not in self.request_status:
            # 不要用 Failed 复活一个已被清除的条目：一旦 clear() 已把该 room 从
            # request_status 中弹出，任何迟到的 update_status(Failed)（如来自 abort()）
            # 都必须是空操作。否则一个 Failed 条目可能污染复用同一 bootstrap_room 的
            # 未来请求。
            if status == KVPoll.Failed:
                return
            self.request_status[bootstrap_room] = status
        else:
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        # 中译：线程安全地记录某请求的失败原因，供后续构造 KVTransferError / 日志使用。
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

    def register_deferred_abort_room(self, bootstrap_room: int) -> None:
        """Arm drain-ack accounting for a held room; a fresh set wipes stale acks
        from a prior request that reused this bootstrap_room."""
        self._deferred_abort_ack_tracker[bootstrap_room] = set()

    def note_abort_ack(self, bootstrap_room: int, prefill_rank: int) -> None:
        """Record a prefill rank's drain ack (decode receiver thread). Only counts
        while the room is held; grabs the set by reference to avoid racing clear."""
        acks = self._deferred_abort_ack_tracker.get(bootstrap_room)
        if acks is not None:
            acks.add(prefill_rank)

    def is_abort_release_safe(self, bootstrap_room: int, required_acks: int) -> bool:
        """True once every prefill rank that could still write these pages has acked."""
        return (
            len(self._deferred_abort_ack_tracker.get(bootstrap_room, ()))
            >= required_acks
        )

    def clear_deferred_abort_state(self, bootstrap_room: int) -> None:
        self._deferred_abort_ack_tracker.pop(bootstrap_room, None)

    def _prefill_unique_rank(self) -> int:
        """Stable per-sender id, matching what the transfer worker syncs on Success."""
        return (
            self.attn_tp_rank * (self.pp_size * self.attn_cp_size)
            + self.pp_rank * self.attn_cp_size
            + self.attn_cp_rank
        )

    def _send_abort_ack(self, decode_ip: str, decode_port: int, room: int) -> None:
        """Best-effort ack that this rank's transfer for an aborted room drained."""
        try:
            na = NetworkAddress(decode_ip, decode_port)
            self._send_multipart_locked(
                na.to_tcp(),
                [
                    b"ABORT_ACK",
                    str(room).encode("ascii"),
                    str(self._prefill_unique_rank()).encode("ascii"),
                ],
                is_ipv6=na.is_ipv6,
            )
        except Exception as e:
            logger.debug(f"Failed to send drained ABORT_ACK for room {room}: {e}")

    def _maybe_ack_drained_abort(self, room: int) -> None:
        """Send the deferred ack once an aborted room's chunks have drained
        (outstanding == 0). pop() makes it fire at most once."""
        if self._staging_outstanding.get(room, 0) > 0:
            return
        target = self._deferred_ack_targets.pop(room, None)
        if target is not None:
            self._send_abort_ack(target[0], target[1], room)

    def register_deferred_ack_target(
        self, room: int, decode_ip: str, decode_port: int
    ) -> None:
        """Hold this room's ack until its transfer drains. Callers must mark the
        room Failed FIRST -- registering while it still accepts chunks lets the
        worker ack, then a new chunk writes pages the decode already released."""
        self._deferred_ack_targets[room] = (decode_ip, decode_port)

    def get_kv_replica_factor(self) -> int:
        if self._kv_replica_factor is None:
            logger.warning_once(
                "get_kv_replica_factor called before resolve_kv_replica_factor; "
                "assuming 1, but the metrics may be inaccurate."
            )
            return 1
        return self._kv_replica_factor

    def resolve_kv_replica_factor(self, transfer_infos: Dict) -> None:
        if not self.is_mla_backend:
            # Only MLA replicates its KV across decode ranks; non-MLA head slices are
            # disjoint and stay pinned at the factor of 1 set in __init__.
            return

        info = next(iter(transfer_infos.values()), None)
        if info is None or info.required_dst_info_num is None:
            logger.warning_once(
                "resolve_kv_replica_factor: no decode destinations available; "
                "KV transfer metrics may be inaccurate."
            )
            return
        self._kv_replica_factor = info.required_dst_info_num

    def _ensure_prefill_recompute_executor(
        self,
    ) -> concurrent.futures.ThreadPoolExecutor:
        """Lazily create the shared executor that drives PD-retract rebootstrap
        ``/generate`` calls. One executor per (decode) kv manager, shared across
        all receivers."""
        executor = self._prefill_recompute_executor
        if executor is not None:
            return executor
        with self._prefill_recompute_executor_lock:
            if self._prefill_recompute_executor is None:
                workers = envs.SGLANG_DISAGGREGATION_THREAD_POOL_SIZE.get()
                if workers is None:
                    workers = 16
                self._prefill_recompute_executor = (
                    concurrent.futures.ThreadPoolExecutor(
                        max_workers=max(1, workers),
                        thread_name_prefix="pd-rebootstrap-prefill",
                    )
                )
            return self._prefill_recompute_executor

    def _get_prefill_recompute_session(self) -> requests.Session:
        """Per-thread ``requests.Session`` for the rebootstrap executor threads
        (``requests.Session`` is not safe for concurrent cross-thread use)."""
        session = getattr(self._prefill_recompute_sessions, "session", None)
        if session is None:
            session = requests.Session()
            self._prefill_recompute_sessions.session = session
        return session

    def _resolve_rebootstrap_prefill_url(
        self, kv_receiver: CommonKVReceiver
    ) -> Optional[str]:
        """Derive the prefill ``/generate`` base URL for a PD true-retraction
        rebootstrap from bootstrap info.

        The decode already knows the prefill host (the ``bootstrap_addr`` host,
        which the router/client set to the prefill's HTTP host), and the prefill
        self-registers its HTTP API port in ``PrefillServerInfo`` at bootstrap
        registration. Combining them yields ``http://{host}:{prefill_http_port}``
        with no router-injected ``pd_rebootstrap_prefill_url``.
        """
        prefill_info = self.prefill_info_table.get(kv_receiver.bootstrap_addr)
        if prefill_info is None or prefill_info.prefill_http_port is None:
            return None
        host = NetworkAddress.parse(kv_receiver.bootstrap_addr).host
        return NetworkAddress(host, prefill_info.prefill_http_port).to_url()

    def submit_prefill_recompute(
        self, kv_receiver: CommonKVReceiver, payload: dict
    ) -> None:
        """Dispatch a PD true-retraction rebootstrap ``/generate`` to the
        original prefill worker so it recomputes the retracted request's prefix
        KV under the current weights and transfers it back over the
        already-bootstrapped channel.

        The target prefill ``/generate`` URL is derived from bootstrap info (the
        prefill self-registered its HTTP port), not from a router-injected field.

        Non-blocking from the scheduler's perspective: the HTTP POST runs on the
        shared executor. Any failure (unresolved URL, HTTP error, exception) is
        surfaced through the standard ``KVPoll.Failed`` path via
        ``kv_receiver.abort()`` so the scheduler's existing transfer-failure
        handling streams the aborted request back to the client. ``payload`` is
        prebuilt by the decode scheduler (``Req.build_rebootstrap_payload``) so
        HTTP/sampling concerns stay on the kv manager.

        The decode scheduler broadcasts each retracted request to every rank in
        its attention TP/CP group and every PP stage, so all of them reach this
        call and would each POST an identical ``/generate`` -- making the prefill
        worker recompute the same request once per decode rank. The ``/generate``
        is a server-level call: the prefill frontend fans it out to its own
        workers and transfers the recomputed KV back to *all* decode ranks, so
        exactly one decode rank must issue it. Elect the same leader the request
        receiver uses (attn-tp/attn-cp group leader, first PP stage); the other
        ranks still bootstrap and receive their KV shard as usual, and on failure
        the leader-only abort matches the leader-only output streaming (other
        ranks fall back to the per-request waiting-timeout safety net).
        """
        if self.attn_tp_rank != 0 or self.attn_cp_rank != 0 or self.pp_rank != 0:
            return
        prefill_url = self._resolve_rebootstrap_prefill_url(kv_receiver)
        if not prefill_url:
            logger.error(
                "PD retract rebootstrap could not resolve the prefill /generate "
                "URL from bootstrap info (rid=%s bootstrap_room=%s bootstrap_addr=%s).",
                payload.get("rid"),
                payload.get("bootstrap_room"),
                kv_receiver.bootstrap_addr,
            )
            self._fail_prefill_recompute(
                kv_receiver,
                "PD retract rebootstrap could not resolve the prefill /generate "
                "URL from bootstrap info.",
            )
            return
        self._ensure_prefill_recompute_executor().submit(
            self._run_prefill_recompute, kv_receiver, prefill_url, payload
        )

    def _fail_prefill_recompute(
        self, kv_receiver: CommonKVReceiver, reason: str
    ) -> None:
        """Fail a rebootstrap request via the standard ``KVPoll.Failed`` path.

        ``abort()`` transitions the receiver to Failed and notifies the prefill
        worker to release its orphaned bootstrap entry, but records a generic
        reason; we overwrite it with a descriptive one so the eventual
        ``failure_exception`` (and the client-facing abort message) explains that
        the rebootstrap ``/generate`` failed rather than reporting a spurious
        ``AbortReq``.
        """
        kv_receiver.abort()
        self.record_failure(kv_receiver.bootstrap_room, reason)

    def _run_prefill_recompute(
        self, kv_receiver: CommonKVReceiver, prefill_url: str, payload: dict
    ) -> None:
        rid = payload.get("rid")
        try:
            response = self._get_prefill_recompute_session().post(
                prefill_url.rstrip("/") + "/generate",
                json=payload,
                timeout=self.waiting_timeout,
            )
            if response.status_code >= 400:
                logger.error(
                    "PD rebootstrap prefill failed for rid=%s status=%s body=%s",
                    rid,
                    response.status_code,
                    response.text[:512],
                )
                self._fail_prefill_recompute(
                    kv_receiver,
                    f"PD retract rebootstrap /generate failed for rid={rid} "
                    f"(status={response.status_code}).",
                )
        except Exception:
            logger.exception("PD rebootstrap prefill request failed for rid=%s", rid)
            self._fail_prefill_recompute(
                kv_receiver,
                f"PD retract rebootstrap /generate request errored for rid={rid}.",
            )

    def try_ensure_parallel_info(self, bootstrap_addr: str) -> bool:
        """Single non-blocking attempt to fetch and cache prefill parallel info.
        Returns True if info is available (cached or freshly fetched).

        中译：（Decode 侧）尝试一次拉取并缓存指定 Prefill 实例的并行拓扑信息，
        非阻塞（单次尝试，失败返回 False）。命中缓存直接返回，不再访问
        bootstrap server。拉取后会校验 page_size / kv_cache_dtype 是否与 Decode 端一致，
        并计算好 rank 映射后写入 prefill_info_table。
        """
        if bootstrap_addr in self.prefill_info_table:
            return True

        info: PrefillServerInfo = None
        try:
            # 中译：四个 rank 参数均为 -1 是“哨兵查询”，表示只要拓扑元信息而非具体 rank 地址。
            url = (
                f"http://{bootstrap_addr}/route?"
                f"prefill_dp_rank={-1}&prefill_cp_rank={-1}&"
                f"target_tp_rank={-1}&target_pp_rank={-1}"
            )
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                info = PrefillServerInfo(**data)
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return False
        except Exception as e:
            logger.error(f"Error fetching prefill server info from bootstrap: {e}")
            return False

        # 合理性校验（Prefill 与 Decode 两侧的 page_size / kv_cache_dtype 必须一致）。
        if info.page_size is not None and info.page_size != self.kv_args.page_size:
            raise RuntimeError(
                f"Page size mismatch: prefill server has page_size={info.page_size}, "
                f"but decode server has page_size={self.kv_args.page_size}. "
                f"Both servers must use the same --page-size value."
            )

        if (
            info.kv_cache_dtype is not None
            and info.kv_cache_dtype != self.kv_cache_dtype_str
        ):
            raise RuntimeError(
                f"KV cache dtype mismatch: prefill server has kv_cache_dtype={info.kv_cache_dtype}, "
                f"but decode server has kv_cache_dtype={self.kv_cache_dtype_str}. "
                f"Both servers must use the same --kv-cache-dtype value."
            )

        if self.dcp_size > 1:
            if not (self.is_mla_backend or self.is_hybrid_mla_backend):
                raise RuntimeError(
                    "PD decode DCP requires an MLA or hybrid-MLA KV pool."
                )
            if info.attn_cp_size != 1:
                raise RuntimeError(
                    "PD decode DCP currently requires prefill attention CP=1, "
                    f"got {info.attn_cp_size}."
                )

        self._resolve_rank_mapping(info)
        self.prefill_info_table[bootstrap_addr] = info
        logger.debug(f"Prefill parallel info for [{bootstrap_addr}]: {info}")
        return True

    def _resolve_rank_mapping(self, info: PrefillServerInfo) -> None:
        """Compute TP/CP/PP rank mapping and store on the PrefillServerInfo object.
        Deterministic for a given (bootstrap_addr, decode engine) pair.

        中译：根据 Prefill 与 Decode 两侧的 TP/CP/PP size 差异，计算本 Decode rank
        应向哪些 Prefill rank 拉取 KV（target_*_ranks）以及需要多少路响应
        （required_*_num），并回填到 info 对象上。TP 不对齐时：
          - decode_tp == prefill_tp：一对一；
          - decode_tp >  prefill_tp：多个 decode rank 共享一个 prefill rank；
          - decode_tp <  prefill_tp：一个 decode rank 需从多个 prefill rank 取（非MLA）。
        """
        # TP rank 映射
        if self.attn_tp_size == info.attn_tp_size:
            target_tp_rank = self.kv_args.engine_rank % self.attn_tp_size
            required_dst_info_num = 1
            required_prefill_response_num = 1
            target_tp_ranks = [target_tp_rank]
        elif self.attn_tp_size > info.attn_tp_size:
            if not self.is_mla_backend and not self.is_hybrid_mla_backend:
                logger.warning_once(
                    "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
                )
            target_tp_rank = (self.kv_args.engine_rank % self.attn_tp_size) // (
                self.attn_tp_size // info.attn_tp_size
            )
            required_dst_info_num = self.attn_tp_size // info.attn_tp_size
            required_prefill_response_num = 1
            target_tp_ranks = [target_tp_rank]
        else:
            if not self.is_mla_backend and not self.is_hybrid_mla_backend:
                logger.warning_once(
                    "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
                )
            # 对于非 MLA 模型，一个 decode rank 需从多个 prefill rank 拉取 KVCache。
            target_tp_ranks = list(
                range(
                    (self.kv_args.engine_rank % self.attn_tp_size)
                    * (info.attn_tp_size // self.attn_tp_size),
                    (self.kv_args.engine_rank % self.attn_tp_size + 1)
                    * (info.attn_tp_size // self.attn_tp_size),
                )
            )
            # 对于 MLA 模型，只需从一个 prefill rank 拉取 KVCache，但仍需在连接池中维持
            # 多个连接，并向其它 prefill rank 发送 dummy 请求，否则 KVPoll 永远无法被
            # 正确置位。
            target_tp_rank = target_tp_ranks[0]
            required_dst_info_num = 1
            if self.is_mla_backend:
                required_prefill_response_num = 1
            else:
                required_prefill_response_num = info.attn_tp_size // self.attn_tp_size

        # CP rank 映射 —— decode 的 cp size 应等于 1。
        assert self.attn_cp_size == 1, (
            f"Decode cp size ({self.attn_cp_size}) should be equal to 1",
        )
        if self.attn_cp_size == info.attn_cp_size:
            assert info.attn_cp_size == 1, (
                f"When prefill cp size is 1, attn cp size should be 1, but got {self.attn_cp_size}",
            )
            target_cp_ranks = [self.attn_cp_rank]
        else:
            target_cp_ranks = list(range(info.attn_cp_size))
            pull_from_all_cp_ranks = (
                self.enable_all_cp_ranks_for_transfer
                or info.enable_dsa_cache_layer_split
            )
            if not pull_from_all_cp_ranks:
                # Only retrieve from prefill CP rank 0 when not using all ranks
                target_cp_ranks = target_cp_ranks[:1]
                required_prefill_response_num *= 1
            else:
                required_prefill_response_num *= info.attn_cp_size // self.attn_cp_size

        # PP rank 映射 —— decode 的 pp size 应等于 prefill 的 pp size 或为 1。
        assert self.pp_size == info.pp_size or self.pp_size == 1, (
            f"Decode pp size ({self.pp_size}) should be equal to prefill pp size ({info.pp_size}) or 1",
        )
        if info.pp_size == self.pp_size:
            target_pp_ranks = [self.pp_rank]
        else:
            target_pp_ranks = list(range(info.pp_size))
            required_prefill_response_num *= info.pp_size // self.pp_size

        info.target_tp_rank = target_tp_rank
        info.target_tp_ranks = target_tp_ranks
        info.target_cp_ranks = target_cp_ranks
        info.target_pp_ranks = target_pp_ranks
        info.required_dst_info_num = required_dst_info_num
        info.required_prefill_response_num = required_prefill_response_num

    def _sync_bootstrap_port_across_nodes(self, local_port: int) -> int:
        """Broadcast world-rank-0's bootstrap port to all prefill ranks.

        Required for multi-node prefill when the launcher auto-reserves a
        free port per host (e.g. Dynamo's
        `_reserve_disaggregation_bootstrap_port`): without sync, non-leader
        ranks register to `<leader_ip>:<their_local_port>`, hit
        `Connection refused`, and the leader's `prefill_port_table` ends
        up missing rows.

        中译：把 world-rank-0（leader）的 bootstrap 端口广播给所有 prefill rank。
        多节点 prefill 且启动器为每台主机自动预留空闲端口（如 Dynamo 的
        `_reserve_disaggregation_bootstrap_port`）时必需：若不同步，非 leader
        rank 会注册到 `<leader_ip>:<自己的本地端口>`，导致 `Connection refused`，
        使 leader 的 `prefill_port_table` 缺失部分行。
        """
        if not self.dist_init_addr or get_parallel().nnodes == 1:
            return local_port

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "torch.distributed must be initialised before "
                "CommonKVManager registers to the bootstrap server in "
                "multi-node prefill mode."
            )

        world_group = get_world_group()
        synced_port = world_group.broadcast_object(local_port, src=0)
        if synced_port != local_port:
            logger.info(
                f"Synced disaggregation bootstrap port from leader: "
                f"local={local_port} -> leader={synced_port} "
                f"(world_rank={world_group.rank_in_group})"
            )
        return synced_port

    def register_to_bootstrap(self):
        """Register prefill server info to bootstrap server via HTTP PUT.

        中译：（Prefill 侧）通过 HTTP PUT /route 将本 rank 的 (ip, port) 与并行拓扑
        注册到 bootstrap server。多节点时 server 位于 dist_init_addr（rank 0），
        单节点时即本机（若绑定到通配地址 0.0.0.0/:: 则改用真实本地 IP，
        因为 aiohttp>=3.9 会拒绝 Host 为 0.0.0.0 的请求）。带指数退避重试。
        """
        if self.dist_init_addr:
            # 多节点情形：bootstrap server 的 host 为 dist_init_addr。
            host = NetworkAddress.parse(self.dist_init_addr).resolved().host
        else:
            # 单节点情形：bootstrap server 的 host 与 http server 的 host 相同。
            host = self.bootstrap_host
            # A wildcard bind address (0.0.0.0 / ::) is not a valid HTTP Host
            # and can't be connected to; rewrite it to the same-family loopback,
            # which the wildcard listener also binds.  (self.local_ip is wrong
            # here — it can resolve to a different family than the listener,
            # e.g. IPv6 while the server is bound to 0.0.0.0.)
            host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)

        bootstrap_na = NetworkAddress(host, self.bootstrap_port)
        url = f"{bootstrap_na.to_url()}/route"
        payload = {
            "attn_tp_size": self.attn_tp_size,
            "attn_tp_rank": self.attn_tp_rank,
            "attn_cp_size": self.attn_cp_size,
            "attn_cp_rank": self.attn_cp_rank,
            "attn_dp_size": self.attn_dp_size,
            "attn_dp_rank": self.attn_dp_rank,
            "pp_size": self.pp_size,
            "pp_rank": self.pp_rank,
            "system_dp_size": self.system_dp_size,
            "system_dp_rank": self.system_dp_rank,
            "rank_ip": self.local_ip,
            "rank_port": self.rank_port,
            "page_size": self.kv_args.page_size,
            "kv_cache_dtype": self.kv_cache_dtype_str,
            "load_balance_method": get_parallel().load_balance_method,
            "enable_dsa_cache_layer_split": get_parallel().enable_dsa_cache_layer_split,
            # Self-register the HTTP API port so the decode can derive the PD
            # retract rebootstrap /generate URL from bootstrap info instead of a
            # router-injected pd_rebootstrap_prefill_url.
            "prefill_http_port": get_serving().port,
        }

        max_retries, initial_delay, max_delay = 5, 1.0, 30.0
        for attempt in range(max_retries):
            try:
                response = requests.put(url, json=payload, timeout=5)
                if response.status_code == 200:
                    logger.debug("Prefill successfully registered to bootstrap server.")
                    return
                logger.warning(
                    f"Prefill register attempt {attempt + 1}/{max_retries} failed: status {response.status_code}"
                )
            except Exception as e:
                # 递归到根因，跳过 urllib3 包装层产生的误导性错误信息。
                cause = e
                while cause.__cause__ is not None:
                    cause = cause.__cause__
                logger.warning(
                    f"Prefill register attempt {attempt + 1}/{max_retries} failed: {cause}"
                )
            if attempt == max_retries - 1:
                break
            delay = min(initial_delay * (2**attempt), max_delay) * (
                0.75 + 0.25 * (time.monotonic() % 1)
            )
            time.sleep(delay)
        logger.error(
            f"Prefill instance failed to register to bootstrap server after {max_retries} retries"
        )

    def _connect(self, endpoint: str, is_ipv6: bool = False):
        # 中译：获取（或创建）到指定端点的 ZMQ PUSH 套接字，带连接缓存与断连监控：
        #       若缓存套接字已断开则关闭重建；并设置 TCP keepalive/无限重连等选项。
        with self._socket_lock:
            sock = self._socket_cache.get(endpoint)
            if sock is not None:
                monitor = self._monitor_cache.get(endpoint)
                disconnected = False
                if monitor is not None:
                    try:
                        monitor.recv_multipart(zmq.NOBLOCK)
                        disconnected = True
                    except zmq.Again:
                        pass
                    except zmq.ZMQError:
                        disconnected = True
                if not disconnected:
                    return sock
                sock.close(linger=0)
                if monitor is not None:
                    monitor.close()
                self._socket_cache.pop(endpoint, None)
                self._monitor_cache.pop(endpoint, None)

            sock = self._zmq_ctx.socket(zmq.PUSH)
            if is_ipv6:
                sock.setsockopt(zmq.IPV6, 1)
            sock.setsockopt(zmq.RECONNECT_IVL, -1)
            sock.setsockopt(zmq.SNDTIMEO, 30000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.TCP_KEEPALIVE, 1)
            sock.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 30)
            sock.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 5)
            sock.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)
            sock.connect(endpoint)
            self._socket_cache[endpoint] = sock
            self._monitor_cache[endpoint] = sock.get_monitor_socket(
                zmq.EVENT_DISCONNECTED
            )
            self._socket_send_locks.setdefault(endpoint, threading.Lock())
            return sock

    def _send_multipart_locked(
        self, endpoint: str, parts: List[bytes], is_ipv6: bool = False
    ):
        # Cached sockets are shared across sender threads and zmq sockets are
        # not thread-safe; serialize sends per endpoint.
        sock = self._connect(endpoint, is_ipv6=is_ipv6)
        with self._socket_send_locks[endpoint]:
            sock.send_multipart(parts)

    def get_mha_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int]
    ) -> Tuple[List[int], List[int], List[int], List[int], int]:
        start_layer = self.kv_args.prefill_start_layer
        num_kv_layers = len(src_kv_ptrs) // 2
        end_layer = start_layer + num_kv_layers
        dst_num_total_layers = len(dst_kv_ptrs) // 2
        src_k_ptrs = src_kv_ptrs[:num_kv_layers]
        src_v_ptrs = src_kv_ptrs[num_kv_layers:]
        if num_kv_layers == dst_num_total_layers:
            dst_k_ptrs = dst_kv_ptrs[:dst_num_total_layers]
            dst_v_ptrs = dst_kv_ptrs[dst_num_total_layers:]
        elif (
            num_kv_layers < dst_num_total_layers
            and dst_num_total_layers % num_kv_layers != 0
        ):
            # 情形：Decode 带有 draft 模型的 KV，而 Prefill 部署时未开启投机解码。
            # dst_kv_ptrs 布局：[K_main..., V_main..., draft_K..., draft_V...]。
            multiplier_ratio = dst_num_total_layers // num_kv_layers
            dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
            v_ptr_offset = num_kv_layers * multiplier_ratio
            dst_v_ptrs = dst_kv_ptrs[
                v_ptr_offset + start_layer : v_ptr_offset + end_layer
            ]
        else:
            # decode 的 pp size 应等于 prefill 的 pp size 或为 1。
            dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
            dst_v_ptrs = dst_kv_ptrs[
                dst_num_total_layers + start_layer : dst_num_total_layers + end_layer
            ]
        layers_current_pp_stage = len(src_k_ptrs)
        return src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage

    def get_mla_kv_ptrs_with_pp(
        self,
        src_kv_ptrs: List[int],
        dst_kv_ptrs: List[int],
        state_type: Optional[StateType] = None,
    ) -> Tuple[List[int], List[int], int]:
        # 快路径：两侧使用完全相同的 PP 布局。
        if len(src_kv_ptrs) == len(dst_kv_ptrs):
            return src_kv_ptrs, dst_kv_ptrs, len(src_kv_ptrs)

        mla_ratios = getattr(self.kv_args, "mla_compression_ratios", None)
        if mla_ratios:
            # 压缩型 MLA（如 DeepSeek V4）：扁平列表按 buffer 类型（压缩率桶）而非
            # 按层组织，因此需在 dst 扁平列表的每个分段内定位本 PP stage 对应的子区间。
            sliced_src_kv_ptrs, sliced_dst_kv_ptrs = self._mla_slice_ptrs_for_pp(
                src_kv_ptrs, dst_kv_ptrs, mla_ratios, state_type
            )
            return (
                sliced_src_kv_ptrs,
                sliced_dst_kv_ptrs,
                len(sliced_src_kv_ptrs),
            )

        # 常规 MLA 的 PP 切片。
        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + len(src_kv_ptrs)
        # decode 的 pp size 应等于 prefill 的 pp size 或为 1。
        sliced_dst_kv_ptrs = dst_kv_ptrs[start_layer:end_layer]
        return src_kv_ptrs, sliced_dst_kv_ptrs, len(src_kv_ptrs)

    def _mla_slice_ptrs_for_pp(
        self,
        src_kv_ptrs: List[int],
        dst_kv_ptrs: List[int],
        mla_ratios: List[int],
        state_type: Optional[StateType] = None,
    ) -> Tuple[List[int], List[int]]:
        """Produce aligned (src, dst) pointer lists for compressed-MLA
        pools (e.g. DeepSeek V4) under PP.

        The pool produces two possible flat-list layouts (selected via dst
        length):

        - kv_data layout, length = 2 * c4_L + c128_L:
            [c4_layer_{0..c4_L-1},
             c4_indexer_layer_{0..c4_L-1},
             c128_layer_{0..c128_L-1}]
          Each section is indexed by compressed-layer id within that
          compression bucket.

        - SWA state_data layout, length = swa_L + 2 * c4_L:
            [swa_layer_{0..swa_L-1},
             c4_compress_state_{0..c4_L-1},
             c4_indexer_compress_state_{0..c4_L-1}]
          ``swa_L`` is the SWA pool's actual buffer count
          (``num_effective_layers``), which can be smaller than
          ``len(mla_ratios)`` when the HF config's ``compress_ratios``
          list contains entries for layers not materialized into the SWA
          pool (e.g. an MTP/nextn slot at the tail).

        - C128_STATE layout, length = c128_L:
            [c128_compress_state_{0..c128_L-1}]

        src is already PP-filtered on the prefill side. dst is the
        decode-side full-model list (when decode is PP=1). We slice dst to
        match src's PP stage. If src itself is also full-model, it is
        returned unchanged.

        中译：在 PP 下为压缩型 MLA pool（如 DeepSeek V4）生成对齐的（src, dst）指针列表。

        该 pool 可能产生两种扁平列表布局（根据 dst 长度区分）：

        - kv_data 布局，长度 = 2 * c4_L + c128_L：
            [c4_layer_{0..c4_L-1},
             c4_indexer_layer_{0..c4_L-1},
             c128_layer_{0..c128_L-1}]
          每个分段内部按该压缩桶内的“压缩层 id”索引。

        - state_data 布局，长度 = swa_L + 2 * c4_L + c128_L：
            [swa_layer_{0..swa_L-1},
             compress_state_{非 None, c4_L + c128_L},
             indexer_compress_state_{非 None, c4_L}]
          ``swa_L`` 是 SWA pool 的实际 buffer 数（``num_effective_layers``），
          当 HF 配置的 ``compress_ratios`` 包含未落入 SWA pool 的层（如尾部的
          MTP/nextn 槽位）时，它可能小于 ``len(mla_ratios)``。

        src 已在 prefill 侧经 PP 过滤；dst 是 decode 侧的全模型列表（当 decode 为 PP=1）。
        我们对 dst 切片以匹配 src 的 PP stage；若 src 本身也是全模型，则原样返回。
        """
        start_layer = self.kv_args.prefill_start_layer
        end_layer = getattr(self.kv_args, "prefill_end_layer", None)
        assert (
            end_layer is not None
        ), "KVArgs.prefill_end_layer must be set when using compressed-MLA PD with PP"

        c4_full = sum(1 for r in mla_ratios if r == 4)
        c128_full = sum(1 for r in mla_ratios if r == 128)
        kv_layout_len = 2 * c4_full + c128_full

        c4_off_s = sum(1 for r in mla_ratios[:start_layer] if r == 4)
        c4_off_e = sum(1 for r in mla_ratios[:end_layer] if r == 4)
        c128_off_s = sum(1 for r in mla_ratios[:start_layer] if r == 128)
        c128_off_e = sum(1 for r in mla_ratios[:end_layer] if r == 128)

        if state_type == StateType.C128_STATE:
            return src_kv_ptrs, list(dst_kv_ptrs[c128_off_s:c128_off_e])

        if state_type == StateType.SWA_RING:
            swa_s = min(start_layer, len(dst_kv_ptrs))
            swa_e = min(end_layer, len(dst_kv_ptrs))
            return src_kv_ptrs, list(dst_kv_ptrs[swa_s:swa_e])

        if (
            state_type not in (StateType.SWA, StateType.SWA_RING, StateType.C128_STATE)
            and len(dst_kv_ptrs) == kv_layout_len
        ):
            sliced_dst = (
                list(dst_kv_ptrs[c4_off_s:c4_off_e])
                + list(dst_kv_ptrs[c4_full + c4_off_s : c4_full + c4_off_e])
                + list(dst_kv_ptrs[2 * c4_full + c128_off_s : 2 * c4_full + c128_off_e])
            )
            return src_kv_ptrs, sliced_dst

        # SWA state-data layout. ``swa_L`` is derived from the actual dst
        # length so we tolerate cases where the SWA pool has fewer buffers
        # than ``len(mla_ratios)`` (e.g. nextn padding). C128 state ships as
        # a separate StateType.C128_STATE component and must not be counted
        # here.
        swa_L = len(dst_kv_ptrs) - 2 * c4_full
        if swa_L < 0 or swa_L > len(mla_ratios):
            raise ValueError(
                f"Unexpected compressed-MLA dst_kv_ptrs length "
                f"{len(dst_kv_ptrs)}; expected either {kv_layout_len} "
                f"(kv_data) or swa_L + {2 * c4_full} "
                f"(state_data) given compression_ratios "
                f"(c4={c4_full}, c128={c128_full}, "
                f"total={len(mla_ratios)})."
            )

        swa_s = min(start_layer, swa_L)
        swa_e = min(end_layer, swa_L)
        compress_section_start = swa_L
        indexer_section_start = swa_L + c4_full
        sliced_dst = (
            list(dst_kv_ptrs[swa_s:swa_e])
            + list(
                dst_kv_ptrs[
                    compress_section_start
                    + c4_off_s : compress_section_start
                    + c4_off_e
                ]
            )
            + list(
                dst_kv_ptrs[
                    indexer_section_start + c4_off_s : indexer_section_start + c4_off_e
                ]
            )
        )

        return src_kv_ptrs, sliced_dst

    def _start_heartbeat_checker_thread(self):
        """Start the heartbeat checker thread for Decode worker.

        中译：（Decode 侧）启动后台心跳线程，周期性向已知的各 Prefill bootstrap
        server 发 GET /health；连续失败达到 max_failures 则视为节点故障，
        调用 _handle_node_failure 清理连接并将受影响请求置 Failed。
        """

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_info_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0
                            self._on_heartbeat_success(bootstrap_addr)
                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=heartbeat_checker, daemon=True).start()

    def _on_heartbeat_success(self, bootstrap_addr: str):
        """Hook called on successful heartbeat. Override for backend-specific cleanup.

        中译：心跳成功时调用的钩子。子类可重写以做后端特定的清理。
        """
        pass

    def _handle_node_failure(self, failed_bootstrap_addr: str):
        """Handle failure of a prefill node.

        中译：处理某 Prefill 节点故障：从连接池/拓扑缓存中剔除该地址，断开
        残留的 ZMQ 端点，并将所有尚未成功且关联该节点的请求标记为 Failed。
        """
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            # 在删除前从缓存的 bootstrap_infos 中收集 TCP 端点。
            stale_endpoints = set()
            for k in keys_to_remove:
                for info in self.connection_pool[k]:
                    ip = info.get("rank_ip")
                    port = info.get("rank_port")
                    if ip and port:
                        na = NetworkAddress(ip, int(port))
                        stale_endpoints.add(na.to_tcp())
            for k in keys_to_remove:
                del self.connection_pool[k]
            self.prefill_info_table.pop(failed_bootstrap_addr, None)

            possible_affected_rooms = list(
                self.addr_to_rooms_tracker.get(failed_bootstrap_addr, [])
            )
            self.addr_to_rooms_tracker.pop(failed_bootstrap_addr, None)

        for endpoint in stale_endpoints:
            CommonKVReceiver.disconnect_endpoint(endpoint)

        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.request_status
                and self.check_status(room) != KVPoll.Success
            ):
                self.record_failure(
                    room,
                    f"Lost connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                affected_rooms.append(room)

        logger.error(
            f"Lost connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), "
            f"{len(affected_rooms)} requests affected"
        )


class CommonKVSender(BaseKVSender):
    """Prefill 侧的「每请求」KV 发送器（后端无关基类）。

    每个待传输的请求（以 bootstrap_room 标识）对应一个 Sender，负责：
      - 维护该请求的发送进度（curr_idx / num_kv_indices）与状态机；
      - 处理 CP dummy rank、dp_rank 路由校验与注册；
      - 统计传输量、处理 bootstrap 超时与中止。
    具体的 send()/poll() 由各后端子类实现。
    """

    def __init__(
        self,
        mgr: CommonKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        # 中译：后端无关的 KV 传输管理器（CommonKVManager）。所有跨请求共享的资源
        #       （状态表、路由信息、KV/state 元数据、传输引擎句柄等）都挂在它上面，
        #       本 Sender 通过它读写状态、发送数据、记录失败。
        self.kv_mgr = mgr
        # 中译：本次传输请求的唯一标识（一次 PD 传输的“房间号”）。Prefill 与 Decode
        #       通过同一 bootstrap_room 配对，也是本 Sender 在状态表中的键。
        self.bootstrap_room = bootstrap_room
        # 中译：辅助数据（aux）在 metadata buffer 中的槽位索引，如首 token / hidden states 等。
        #       初始为 None，稍后由 init() 填入。
        self.aux_index = None
        # 中译：bootstrap server 的地址（host:port），用于向其注册 dp_rank、查询路由等控制面交互。
        self.bootstrap_server_url = bootstrap_addr
        # 中译：本次传输的“最终态”缓存（Success/Failed 等）。一旦得出终态就记在这里，
        #       后续 poll() 直接返回它，避免重复计算或状态回退。None 表示尚未有终态。
        self.conclude_state: Optional[KVPoll] = None
        # 中译：本请求的传输性能指标（时延、总字节数、速度等），由 get_transfer_metric() 汇总输出。
        self._transfer_metric = KVTransferMetric()
        # 中译：累计已发送的 KV 索引数量（token 粒度），用于计算传输总字节数。
        self._transfer_num_kv_indices = 0
        # 中译：累计已发送的 state 索引数量（Mamba/SWA/DSA 等特殊状态），用于计算传输总字节数。
        self._transfer_num_state_indices = 0
        # inner state
        # 中译：当前发送进度游标——已发送到的 KV 索引位置，配合 num_kv_indices 支持分块续发。
        self.curr_idx = 0
        # 中译：本次传输的起始时间戳（首次真正发送时记录），用于 bootstrap 超时判定与时延统计。None 表示尚未开始。
        self.init_time: Optional[float] = None
        if self.kv_mgr.is_dummy_cp_rank:
            # Non-authoritative CP ranks are dummy participants.
            # 中译：非权威 CP rank 为 dummy 参与者，不实际发送，直接置为等待输入。
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)
            return

        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        # 中译：多 DP 时需确定该请求归属哪个 prefill dp_rank：
        if get_parallel().dp_size > 1 and not req_has_disagg_prefill_dp_rank:
            if get_parallel().load_balance_method != "follow_bootstrap_room":
                self._register_prefill_dp_rank()
            elif (
                self.kv_mgr.attn_dp_rank != self.bootstrap_room % get_parallel().dp_size
            ):
                # follow_bootstrap_room was overridden by external routed_dp_rank
                if envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.get():
                    self._register_prefill_dp_rank()
                else:
                    self.kv_mgr.record_failure(
                        self.bootstrap_room,
                        f"follow_bootstrap_room conflict: dispatched to dp_rank "
                        f"{self.kv_mgr.attn_dp_rank} but bootstrap_room "
                        f"{self.bootstrap_room} implies dp_rank "
                        f"{self.bootstrap_room % get_parallel().dp_size}. "
                        f"Set SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK=1 "
                        f"to allow mixed routing.",
                    )
                    self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                    return

    def _register_prefill_dp_rank(self):
        """Register this request's prefill dp_rank to the bootstrap server.

        中译：将本请求（bootstrap_room）实际落到的 prefill dp_rank 注册到
        bootstrap server，供 Decode 端后续通过 /query_dp_ranks 查到正确的 dp 组。
        """
        url = f"http://{self.bootstrap_server_url}/register_dp_rank"
        payload = {
            "bootstrap_room": self.bootstrap_room,
            "dp_rank": self.kv_mgr.attn_dp_rank,
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code != 200:
                logger.error(
                    f"Failed to register prefill dp_rank: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(f"Failed to register prefill dp_rank: {e}")

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        # 中译：记录本请求待发送的 KV 索引总数与辅助数据索引（如 aux 缓冲区位置），
        #       作为后续分块发送与“是否最后一块”判断的依据。
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index
        logger.debug(
            f"CommonKVSender init with num_kv_indices: {num_kv_indices} and aux_index: {aux_index}"
        )

    def pop_decode_prefix_len(self) -> int:
        return self.kv_mgr.req_to_decode_prefix_len.pop(self.bootstrap_room, 0)

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        return num_pages > 0 or last_chunk

    def get_transfer_metric(self) -> KVTransferMetric:
        total_bytes = self._transfer_num_kv_indices * self.kv_mgr.kv_item_lens_sum
        total_bytes += (
            self._transfer_num_state_indices * self.kv_mgr.state_item_lens_sum
        )
        # Pinned to 1 for MHA (disjoint slices); only MLA replication makes it > 1.
        total_bytes *= self.kv_mgr.get_kv_replica_factor()
        self._transfer_metric.transfer_total_bytes = total_bytes
        return self._transfer_metric

    def _record_transfer_indices(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List],
    ):
        self._transfer_num_kv_indices += len(kv_indices)
        if state_indices:
            for component_indices in state_indices:
                if component_indices is not None:
                    self._transfer_num_state_indices += len(component_indices)

    def _prepare_send_indices(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ) -> Tuple[npt.NDArray[np.int32], slice, bool, bool]:
        """Common pre-processing for send(): index tracking and CP-rank handling.

        中译：send() 的通用前置处理：推进发送游标 curr_idx、判定是否最后一块，
        并根据 CP 配置过滤/跳过本 rank 不负责的索引（dummy CP rank 仅在最后
        一块时置成功）。若返回 should_skip=True，调用方应立即返回。

        Returns:
            (kv_indices, index_slice, is_last_chunk, should_skip)
            If should_skip is True, the caller should return immediately.
        """
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last_chunk = self.curr_idx == self.num_kv_indices

        if (
            self.kv_mgr.enable_all_cp_ranks_for_transfer
            and not get_parallel().enable_dsa_cache_layer_split
        ):
            kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
                total_pages=self.num_kv_indices,
            )
        elif self.kv_mgr.is_dummy_cp_rank:
            if not is_last_chunk:
                return kv_indices, index_slice, is_last_chunk, True
            else:
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
                return kv_indices, index_slice, is_last_chunk, True

        return kv_indices, index_slice, is_last_chunk, False

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
    ):
        # 中译：发送一批 KV 索引对应的 KV cache（数据面）。基类为空实现，
        #       由各后端子类（Mooncake/NIXL/…）结合具体传输机制实现。
        pass

    def _check_bootstrap_timeout(self) -> Optional[KVPoll]:
        # 中译：（Prefill 侧）检查是否在 Bootstrapping 阶段超时（未收到 Decode 侧的
        #       KV 索引）。超时则记录失败并置 Failed，以释放 tree cache；
        #       可通过 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT 放宽阈值。
        if self.init_time is None:
            return None
        elapsed = time.time() - self.init_time
        if elapsed < self.kv_mgr.bootstrap_timeout:
            return None
        logger.warning_once(
            "Some requests timed out when bootstrapping, "
            "which means prefill instances fail to receive the KV indices from the decode instance of this request. "
            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
        )
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s "
            f"in KVPoll.Bootstrapping",
        )
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        return KVPoll.Failed

    def clear(self) -> None:
        # 中译：清理本请求在管理器上的残留状态（状态、前缀长度、传输信息），
        #       供 bootstrap_room 安全复用。
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        if hasattr(self.kv_mgr, "req_to_decode_prefix_len"):
            self.kv_mgr.req_to_decode_prefix_len.pop(self.bootstrap_room, None)
        if hasattr(self.kv_mgr, "transfer_infos"):
            self.kv_mgr.transfer_infos.pop(self.bootstrap_room, None)
        if hasattr(self.kv_mgr, "_deferred_ack_targets"):
            # Drop a held ack target if the room concluded without draining
            # (e.g. aborted before any chunk enqueued); else it leaks on prefill.
            self.kv_mgr._deferred_ack_targets.pop(self.bootstrap_room, None)

    def abort(self):
        # 中译：响应 AbortReq 中止本请求的发送，记录失败原因并置 Failed 终态。
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        self.conclude_state = KVPoll.Failed


class CommonKVReceiver(BaseKVReceiver):
    """Decode 侧的「每请求」KV 接收器（后端无关基类）。

    每个需从 Prefill 取 KV 的请求对应一个 Receiver，负责：
      - 根据预先计算好的 rank 映射，从 bootstrap server 拉取各目标 Prefill
        rank 的 (ip, port)（_setup_bootstrap_infos）；
      - 向 Prefill 侧发送本请求的 KV 索引元数据（由子类实现）；
      - 处理连接池复用、等待超时与中止通知。
    类级共享一套 ZMQ PUSH 套接字缓存（多个请求复用到同一 Prefill 端点的连接）。
    """

    _ctx = zmq.Context()
    _ctx.set(zmq.MAX_SOCKETS, envs.SGLANG_DISAGGREGATION_ZMQ_MAX_SOCKETS.get())
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        mgr: CommonKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self.conclude_state: Optional[KVPoll] = None
        self.require_staging: bool = False
        self.init_time: Optional[float] = None
        self.abort_notified: bool = False
        self._connection_pool_entries: Dict[str, List[Dict]] = {}
        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(self.bootstrap_room)
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)

    def init(self, prefill_dp_rank: int):
        # 中译：初始化本请求的接收：校验目标 Prefill 拓扑已缓存（否则视为节点已下线），
        #       读取预先计算好的 target_tp/cp/pp rank 映射，建立各目标 rank 的连接信息，
        #       成功后置为 WaitingForInput。
        if self.bootstrap_addr not in self.kv_mgr.prefill_info_table:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Prefill server with bootstrap_addr: {self.bootstrap_addr} is healthy before, but now it is down. Request (bootstrap_room: {self.bootstrap_room}) has been marked as failed.",
            )
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        # Read pre-computed rank mapping from prefill_info (computed in try_ensure_parallel_info)
        self.prefill_info = self.kv_mgr.prefill_info_table[self.bootstrap_addr]
        self.target_tp_rank = self.prefill_info.target_tp_rank
        self.target_tp_ranks = self.prefill_info.target_tp_ranks
        self.target_cp_ranks = self.prefill_info.target_cp_ranks
        self.target_pp_ranks = self.prefill_info.target_pp_ranks
        self.required_dst_info_num = self.prefill_info.required_dst_info_num
        self.required_prefill_response_num = (
            self.prefill_info.required_prefill_response_num
        )

        self.kv_mgr.required_prefill_response_num_table[self.bootstrap_room] = (
            self.required_prefill_response_num
        )

        if self.kv_mgr.enable_staging:
            self.require_staging = (
                self.prefill_info.attn_tp_size != 0
                and self.prefill_info.attn_tp_size != self.kv_mgr.attn_tp_size
            )

        self.prefill_dp_rank = prefill_dp_rank
        self._setup_bootstrap_infos()
        if self.conclude_state == KVPoll.Failed:
            return
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _setup_bootstrap_infos(self):
        # 中译：为本请求需要连接的每个 (cp, tp, pp) 目标 Prefill rank 从 bootstrap server
        #       拉取其 (ip, port)，并缓存到连接池（按 bootstrap_key 去重）；
        #       MLA 下仅 target_tp_rank 为真实 rank，其余为 dummy（仅维持连接以正确推进 KVPoll）。
        all_bootstrap_infos = []
        # NOTE: key distinguished by bootstrap_addr, prefill_dp_rank, prefill_cp_rank, and target_tp_rank
        for target_cp_rank in self.target_cp_ranks:
            bootstrap_key = f"{self.bootstrap_addr}_{self.prefill_dp_rank}_{target_cp_rank}_{self.target_tp_rank}"

            with self.kv_mgr.connection_lock:
                cached_bootstrap_infos = self.kv_mgr.connection_pool.get(bootstrap_key)

            if cached_bootstrap_infos is None:
                bootstrap_infos = []
                for target_tp_rank in self.target_tp_ranks:
                    # Enable higher PP ranks to be bootstrapped earlier to make PP PD requests bootstrap more robust
                    for target_pp_rank in reversed(self.target_pp_ranks):
                        bootstrap_info = self._get_bootstrap_info_from_server(
                            self.prefill_dp_rank,
                            target_cp_rank,
                            target_tp_rank,
                            target_pp_rank,
                        )
                        if bootstrap_info is not None:
                            if self.kv_mgr.is_mla_backend:
                                # For MLA: target_tp_rank is the selected real rank, others are dummy ranks
                                bootstrap_info["is_dummy"] = not bool(
                                    target_tp_rank == self.target_tp_rank
                                    or self.target_tp_rank is None
                                )
                            else:
                                # For non-MLA: all target_tp_ranks are selected real ranks
                                bootstrap_info["is_dummy"] = False
                            logger.debug(
                                f"Fetched bootstrap info: {bootstrap_info} for DP {self.prefill_dp_rank} CP {target_cp_rank} TP {target_tp_rank} PP {target_pp_rank}"
                            )
                            bootstrap_infos.append(bootstrap_info)
                        else:
                            self.kv_mgr.record_failure(
                                self.bootstrap_room,
                                f"Could not fetch bootstrap info for: prefill_dp_rank: {self.prefill_dp_rank} prefill_cp_rank: {target_cp_rank} target_tp_rank: {target_tp_rank} and target_pp_rank {target_pp_rank}",
                            )
                            self.conclude_state = KVPoll.Failed
                            self.kv_mgr.update_status(
                                self.bootstrap_room, KVPoll.Failed
                            )
                            self.bootstrap_infos = None
                            self.invalidate_cached_bootstrap_infos()
                            return

                self.bootstrap_infos = bootstrap_infos
                self._connection_pool_entries[bootstrap_key] = self.bootstrap_infos

                # Register kv_args only once to prefill KVManager according to the info fetched
                # from the bootstrap server. Do this before caching in connection_pool so a failed
                # registration does not leave a stale entry that later requests would reuse.
                if not self._register_kv_args():
                    self.invalidate_cached_bootstrap_infos()
                    return

                with self.kv_mgr.connection_lock:
                    cached_bootstrap_infos = self.kv_mgr.connection_pool.setdefault(
                        bootstrap_key, self.bootstrap_infos
                    )

                if cached_bootstrap_infos is not self.bootstrap_infos:
                    self.bootstrap_infos = cached_bootstrap_infos
            else:
                self.bootstrap_infos = cached_bootstrap_infos

            self._connection_pool_entries[bootstrap_key] = self.bootstrap_infos

            assert len(self.bootstrap_infos) > 0
            all_bootstrap_infos.extend(self.bootstrap_infos)

        self.bootstrap_infos = all_bootstrap_infos

    def invalidate_cached_bootstrap_infos(self) -> None:
        with self.kv_mgr.connection_lock:
            for bootstrap_key, bootstrap_infos in self._connection_pool_entries.items():
                if self.kv_mgr.connection_pool.get(bootstrap_key) is bootstrap_infos:
                    del self.kv_mgr.connection_pool[bootstrap_key]
            self._connection_pool_entries.clear()

    def _get_bootstrap_info_from_server(
        self, prefill_dp_rank, prefill_cp_rank, target_tp_rank, target_pp_rank
    ):
        """Fetch the bootstrap info from the bootstrap server.

        中译：通过 GET /route 按 (dp, cp, tp, pp) 四元组向 bootstrap server 查询
        具体目标 Prefill rank 的 (rank_ip, rank_port)；失败返回 None。
        """
        try:
            url = f"http://{self.bootstrap_addr}/route?prefill_dp_rank={prefill_dp_rank}&prefill_cp_rank={prefill_cp_rank}&target_tp_rank={target_tp_rank}&target_pp_rank={target_pp_rank}"
            response = _get_bootstrap_session(self.bootstrap_addr).get(url, timeout=5)
            if response.status_code == 200:
                bootstrap_info = response.json()
                bootstrap_info["pp_rank"] = int(target_pp_rank)
                return bootstrap_info
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    @staticmethod
    def query_prefill_dp_ranks(
        bootstrap_addr: str, bootstrap_rooms: List[int]
    ) -> Dict[str, int]:
        """Batch query prefill dp_ranks for given bootstrap_rooms.

        中译：通过 POST /query_dp_ranks 批量查询一组 bootstrap_room 各自实际落到的
        prefill dp_rank（与 Sender 侧 _register_prefill_dp_rank 配对使用）。
        """
        try:
            url = f"http://{bootstrap_addr}/query_dp_ranks"
            response = _get_bootstrap_session(bootstrap_addr).post(
                url,
                json={"bootstrap_rooms": bootstrap_rooms},
                timeout=5,
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    f"Failed to query dp_ranks: {response.status_code}, {response.text}"
                )
                return {}
        except Exception as e:
            logger.error(f"Error querying dp_ranks from bootstrap: {e}")
            return {}

    @classmethod
    def _connect(cls, endpoint: str, is_ipv6: bool = False):
        # 中译：类级共享的 ZMQ PUSH 套接字缓存：同一端点只建一次连接，多请求复用。
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                if is_ipv6:
                    sock.setsockopt(zmq.IPV6, 1)
                sock.setsockopt(zmq.LINGER, 0)
                # Bound send so a dead peer cannot block the scheduler forever.
                sock.setsockopt(
                    zmq.SNDTIMEO,
                    envs.SGLANG_DISAGGREGATION_ZMQ_SEND_TIMEOUT.get() * 1000,
                )
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    @classmethod
    def disconnect_endpoint(cls, endpoint: str):
        # 中译：从类级缓存中移除并关闭指定端点的套接字（节点故障时清理残留连接）。
        with cls._global_lock:
            sock = cls._socket_cache.pop(endpoint, None)
            lock = cls._socket_locks.pop(endpoint, None)
        if sock:
            if lock:
                with lock:
                    sock.close()
            else:
                sock.close()
            logger.debug(f"Disconnected stale ZMQ PUSH socket (receiver): {endpoint}")

    @classmethod
    def _connect_to_bootstrap_server(cls, bootstrap_info: dict):
        ip_address = bootstrap_info["rank_ip"]
        port = bootstrap_info["rank_port"]
        na = NetworkAddress(ip_address, port)
        sock, lock = cls._connect(na.to_tcp(), is_ipv6=na.is_ipv6)
        return sock, lock

    def _register_kv_args(self) -> bool:
        return True

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        raise NotImplementedError

    def _check_waiting_timeout(self) -> Optional[KVPoll]:
        # 中译：（Decode 侧）检查 WaitingForInput 阶段是否超时（bootstrap 后未收到
        #       KV 传输完成信号）。超时则置 Failed 并向 Prefill 侧发送中止通知；
        #       可通过 SGLANG_DISAGGREGATION_WAITING_TIMEOUT 放宽阈值。
        if self.init_time is None:
            return None
        elapsed = time.time() - self.init_time
        if elapsed < self.kv_mgr.waiting_timeout:
            return None
        logger.warning_once(
            "Some requests fail to receive KV Cache transfer done signal after bootstrapping. "
            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
        )
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s "
            f"in KVPoll.WaitingForInput",
        )
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        self.invalidate_cached_bootstrap_infos()
        if (
            not self.abort_notified
            and hasattr(self, "bootstrap_infos")
            and self.bootstrap_infos is not None
        ):
            self._send_abort_notification()
            self.abort_notified = True
        return KVPoll.Failed

    def clear(self) -> None:
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        self.kv_mgr.required_prefill_response_num_table.pop(self.bootstrap_room, None)
        self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room, None)

    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        self.conclude_state = KVPoll.Failed
        if (
            not self.abort_notified
            and hasattr(self, "bootstrap_infos")
            and self.bootstrap_infos is not None
        ):
            self._send_abort_notification()
            self.abort_notified = True

    def _send_abort_notification(self):
        # 中译：向所有目标 Prefill rank 发送 ABORT 控制消息（best-effort），
        #       告知其本请求已中止，便于 Prefill 侧释放对应资源。
        for bootstrap_info in self.bootstrap_infos:
            # Best-effort notification to prefill side that this request was aborted.
            try:
                sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
                with lock:
                    sock.send_multipart(
                        [
                            b"ABORT",
                            str(self.bootstrap_room).encode("ascii"),
                            self.kv_mgr.local_ip.encode("ascii"),
                            str(self.kv_mgr.rank_port).encode("ascii"),
                        ]
                    )
                logger.debug(
                    f"Sent abort notification for room {self.bootstrap_room} "
                    f"to {bootstrap_info.get('rank_ip', 'unknown')}:{bootstrap_info.get('rank_port', 'unknown')}"
                )
            except Exception as e:
                logger.debug(
                    f"Failed to send abort notification for room {self.bootstrap_room}: {e}"
                )


class CommonKVBootstrapServer(BaseKVBootstrapServer):
    """轻量的 aiohttp HTTP 服务，仅由 Prefill 实例启动（跑在 TokenizerManager 后台线程）。

    本质上是一个「注册中心 / 元数据交换点」，全部状态保存在进程内存，
    不是独立分布式集群，也不参与 KV 数据面传输。提供的路由：
      - PUT  /route            Prefill 各 rank 注册自身 (ip, port) 与并行拓扑；
      - GET  /route            Decode 端查拓扑元信息（哨兵）或具体 rank 地址；
      - POST /register_dp_rank  登记某 bootstrap_room 实际落到的 dp_rank；
      - POST /query_dp_ranks    批量查询上述 dp_rank 映射；
      - GET  /health            健康检查（供 Decode 侧心跳）。
    因为状态在内存，进程重启即丢失；靠多 Prefill 实例部署与上层路由分散单点风险。
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.store = dict()
        self.lock = asyncio.Lock()
        # The event loop only keeps weak references to tasks, so a long-lived
        # task needs a strong reference to survive garbage collection.
        self._background_tasks: Set[asyncio.Task] = set()
        self._setup_routes()
        self.pp_size = None
        self.attn_tp_size = None
        self.attn_cp_size = None
        self.dp_size = None
        self.page_size = None
        self.kv_cache_dtype: Optional[str] = None
        self.follow_bootstrap_room: Optional[bool] = None
        self.enable_dsa_cache_layer_split: Optional[bool] = None
        self.prefill_http_port: Optional[int] = None
        self.prefill_port_table: Dict[
            int, Dict[int, Dict[int, Dict[int, PrefillRankInfo]]]
        ] = {}
        self.room_to_dp_rank: Dict[int, Dict[str, Union[int, float]]] = {}
        self._registered_count = 0
        self.entry_cleanup_interval = (
            envs.SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL.get()
        )

        # Start bootstrap server
        # 中译：在后台 daemon 线程中启动 aiohttp 事件循环（与主进程同生命周期）。
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        self.thread.start()

    def _is_ready(self) -> bool:
        # 中译：判断是否所有预期的 Prefill worker 均已注册（期望数 = dp*cp*tp*pp）。
        #       未就绪时 GET /route 会返回 503，避免 Decode 端拿到不完整拓扑。
        if (
            self.attn_tp_size is None
            or self.attn_cp_size is None
            or self.pp_size is None
            or self.dp_size is None
        ):
            return False
        expected = self.dp_size * self.attn_cp_size * self.attn_tp_size * self.pp_size
        logger.debug(
            f"Expected {expected} prefill servers to be registered, {self._registered_count} registered so far"
        )
        return self._registered_count >= expected

    def _setup_routes(self):
        # 中译：注册 HTTP 路由（/route 同时处理 PUT 注册与 GET 查询）。
        self.app.router.add_route("*", "/route", self._handle_route)
        self.app.router.add_post("/register_dp_rank", self._handle_register_dp_rank)
        self.app.router.add_post("/query_dp_ranks", self._handle_query_dp_ranks)
        self.app.router.add_get("/health", self._handle_health_check)

    async def _handle_health_check(self, request):
        return web.Response(text="OK", status=200)

    async def _handle_route(self, request: web.Request):
        method = request.method
        if method == "PUT":
            return await self._handle_route_put(request)
        elif method == "GET":
            return await self._handle_route_get(request)
        else:
            return web.Response(
                text="Method not allowed", status=405, content_type="application/json"
            )

    async def _handle_route_put(self, request: web.Request):
        # 中译：处理 Prefill rank 的注册：首次注册时记录全局拓扑（tp/cp/dp/pp size、
        #       page_size、kv_cache_dtype、负载均衡策略），并把本 rank 的 (ip, port)
        #       按 (dp, cp, tp, pp) 层级存入 prefill_port_table（加锁保证线程安全）。
        data = await request.json()
        attn_tp_size = data["attn_tp_size"]
        attn_tp_rank = data["attn_tp_rank"]
        attn_cp_size = data["attn_cp_size"]
        attn_cp_rank = data["attn_cp_rank"]
        attn_dp_size = data["attn_dp_size"]
        attn_dp_rank = data["attn_dp_rank"]
        pp_size = data["pp_size"]
        pp_rank = data["pp_rank"]
        system_dp_size = data["system_dp_size"]
        system_dp_rank = data["system_dp_rank"]
        rank_ip = data["rank_ip"]
        rank_port = int(data["rank_port"])
        page_size = int(data["page_size"])
        kv_cache_dtype = data["kv_cache_dtype"]
        prefill_http_port = data.get("prefill_http_port")

        if self.attn_tp_size is None:
            self.attn_tp_size = attn_tp_size

        if self.attn_cp_size is None:
            self.attn_cp_size = attn_cp_size

        if self.dp_size is None:
            self.dp_size = attn_dp_size if system_dp_size == 1 else system_dp_size

        if self.pp_size is None:
            self.pp_size = pp_size

        if self.page_size is None and page_size is not None:
            self.page_size = page_size

        if self.kv_cache_dtype is None and kv_cache_dtype is not None:
            self.kv_cache_dtype = kv_cache_dtype

        if self.prefill_http_port is None and prefill_http_port is not None:
            self.prefill_http_port = int(prefill_http_port)

        if self.follow_bootstrap_room is None:
            load_balance_method = data.get(
                "load_balance_method", "follow_bootstrap_room"
            )
            self.follow_bootstrap_room = load_balance_method == "follow_bootstrap_room"

        if self.enable_dsa_cache_layer_split is None:
            self.enable_dsa_cache_layer_split = bool(
                data.get("enable_dsa_cache_layer_split", False)
            )

        if system_dp_size == 1:
            dp_group = attn_dp_rank
        else:
            dp_group = system_dp_rank

        # Add lock to make sure thread-safe
        async with self.lock:
            dp_group_table = self.prefill_port_table.setdefault(dp_group, {})
            cp_group_table = dp_group_table.setdefault(attn_cp_rank, {})
            tp_group_table = cp_group_table.setdefault(attn_tp_rank, {})

            tp_group_table[pp_rank] = PrefillRankInfo(
                rank_ip=rank_ip,
                rank_port=rank_port,
            )

            self._registered_count += 1

        expected = self.dp_size * self.attn_cp_size * self.attn_tp_size * self.pp_size
        logger.debug(
            f"Register prefill bootstrap: DP{dp_group} CP{attn_cp_rank} TP{attn_tp_rank} PP{pp_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
            f" ({self._registered_count}/{expected} registered)"
        )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        # 中译：处理 Decode 端查询。若四个 rank 参数均为 -1（哨兵），返回整体拓扑
        #       元信息（PrefillServerInfo）；否则按 (dp, cp, tp, pp) 返回具体 rank 的
        #       (ip, port)。未完成注册时返 503，找不到对应项时返 404。
        prefill_dp_rank = request.query.get("prefill_dp_rank")
        prefill_cp_rank = request.query.get("prefill_cp_rank")
        target_tp_rank = request.query.get("target_tp_rank")
        target_pp_rank = request.query.get("target_pp_rank")
        if (
            not prefill_dp_rank
            or not prefill_cp_rank
            or not target_tp_rank
            or not target_pp_rank
        ):
            return web.Response(text="Missing inputs for bootstrap server.", status=400)

        if (
            int(prefill_dp_rank) == -1
            and int(prefill_cp_rank) == -1
            and int(target_tp_rank) == -1
            and int(target_pp_rank) == -1
        ):
            if not self._is_ready():
                return web.Response(
                    text=f"Prefill server not fully registered yet"
                    f" ({self._registered_count} workers registered).",
                    status=503,
                )
            info = PrefillServerInfo(
                attn_tp_size=self.attn_tp_size,
                attn_cp_size=self.attn_cp_size,
                dp_size=self.dp_size,
                pp_size=self.pp_size,
                page_size=self.page_size,
                kv_cache_dtype=self.kv_cache_dtype,
                follow_bootstrap_room=(
                    self.follow_bootstrap_room
                    if self.follow_bootstrap_room is not None
                    else True
                ),
                enable_dsa_cache_layer_split=bool(self.enable_dsa_cache_layer_split),
                prefill_http_port=self.prefill_http_port,
            )
            return web.json_response(dataclasses.asdict(info), status=200)

        if not self._is_ready():
            return web.Response(
                text=f"Prefill server not fully registered yet"
                f" ({self._registered_count} workers registered).",
                status=503,
            )

        # Find corresponding prefill info
        try:
            async with self.lock:
                bootstrap_info = self.prefill_port_table[int(prefill_dp_rank)][
                    int(prefill_cp_rank)
                ][int(target_tp_rank)][int(target_pp_rank)]
        except KeyError:
            return web.Response(
                text=f"Bootstrap info not found for dp_rank={prefill_dp_rank} cp_rank={prefill_cp_rank} "
                f"tp_rank={target_tp_rank} pp_rank={target_pp_rank}",
                status=404,
            )

        return web.json_response(dataclasses.asdict(bootstrap_info), status=200)

    async def _handle_register_dp_rank(self, request: web.Request):
        data = await request.json()
        bootstrap_room = int(data["bootstrap_room"])
        dp_rank = int(data["dp_rank"])
        async with self.lock:
            self.room_to_dp_rank[bootstrap_room] = {
                "dp_rank": dp_rank,
                "timestamp": time.time(),
            }
        logger.debug(f"Registered dp_rank={dp_rank} for {bootstrap_room=}")
        return web.Response(text="OK", status=200)

    async def _handle_query_dp_ranks(self, request: web.Request):
        data = await request.json()
        bootstrap_rooms = data["bootstrap_rooms"]
        result = {}
        async with self.lock:
            for room in bootstrap_rooms:
                room_int = int(room)
                if room_int in self.room_to_dp_rank:
                    result[str(room_int)] = self.room_to_dp_rank[room_int]["dp_rank"]
        return web.json_response(result, status=200)

    async def _cleanup_expired_entries(self):
        """Remove entries older than cleanup interval from room_to_dp_rank.

        中译：后台协程，周期性清理 room_to_dp_rank 中过期的条目，
        避免已完成的请求映射无限堆积占用内存。
        """
        while True:
            await asyncio.sleep(self.entry_cleanup_interval)
            current_time = time.time()
            async with self.lock:
                expired_keys = [
                    key
                    for key, value in self.room_to_dp_rank.items()
                    if current_time - value["timestamp"] > self.entry_cleanup_interval
                ]
                for key in expired_keys:
                    del self.room_to_dp_rank[key]
            if expired_keys:
                logger.debug(
                    f"Cleaned up {len(expired_keys)} expired entries from room_to_dp_rank"
                )

    def _run_server(self):
        # 中译：在后台线程内创建独立事件循环，启动 aiohttp 服务并常驻；
        #       同时拉起过期条目清理协程。
        try:
            # Event Loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            cleanup_task = self._loop.create_task(self._cleanup_expired_entries())
            self._background_tasks.add(cleanup_task)
            cleanup_task.add_done_callback(self._background_tasks.discard)

            access_log = None
            if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
                access_log = self.app.logger

            self._runner = web.AppRunner(self.app, access_log=access_log)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, host=self.host, port=self.port)
            self._loop.run_until_complete(site.start())
            logger.info(
                f"CommonKVBootstrapServer started successfully on {self.host}:{self.port}"
            )
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Server error: {str(e)}", exc_info=True)
        finally:
            # Cleanup
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    def close(self):
        """Shutdown

        中译：优雅关闭：线程安全地停止事件循环并等待后台线程退出。
        """
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("Stopping server loop...")

        if self.thread.is_alive():
            self.thread.join(timeout=2)
            logger.info("Server thread stopped")

    def poll(self) -> KVPoll: ...
