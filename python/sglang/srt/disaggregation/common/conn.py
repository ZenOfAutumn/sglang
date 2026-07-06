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
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    filter_kv_indices_for_cp_rank,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_dp_rank,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import (
    NetworkAddress,
    get_local_ip_auto,
    get_zmq_socket_on_host,
)

logger = logging.getLogger(__name__)


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
        self.kv_args = args
        self.kv_item_lens_sum = sum(args.kv_item_lens)
        self.state_item_lens_sum = sum(x for comp in args.state_item_lens for x in comp)
        self.is_mla_backend = is_mla_backend
        self.disaggregation_mode = disaggregation_mode
        self.server_args = server_args
        # for p/d multi node infer
        self.bootstrap_host = server_args.host
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_cp_size = get_attention_cp_size()
        self.attn_cp_rank = get_attention_cp_rank()
        self.attn_dp_size = get_attention_dp_size()
        self.attn_dp_rank = get_attention_dp_rank()
        self.system_dp_size = (
            1 if server_args.enable_dp_attention else server_args.dp_size
        )
        self.system_dp_rank = (
            self.kv_args.system_dp_rank if self.kv_args.system_dp_rank else 0
        )
        self.pp_size = server_args.pp_size
        self.pp_rank = self.kv_args.pp_rank
        self.local_ip = get_local_ip_auto()
        # 中译：为 True 时所有 CP rank 都参与 KV 传输；否则仅 CP rank 0 发送（其余为 dummy）。
        self.enable_all_cp_ranks_for_transfer = (
            envs.SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER.get()
        )

        # bind zmq socket
        # 中译：绑定本 rank 的 ZMQ PULL 套接字，自动选取空闲端口；该 (ip, port)
        #       会随后注册到 bootstrap server，作为本 rank 的 KV 传输控制通道地址。
        self._zmq_ctx = zmq.Context()
        self.rank_port, self.server_socket = get_zmq_socket_on_host(
            self._zmq_ctx, zmq.PULL, host=self.local_ip
        )
        logger.debug(f"kv manager bind to {self.local_ip}:{self.rank_port}")

        self.request_status: Dict[int, KVPoll] = {}
        self._socket_cache: Dict[str, zmq.Socket] = {}
        self._monitor_cache: Dict[str, zmq.Socket] = {}
        self._socket_lock = threading.Lock()
        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # 中译：PREFILL 角色：同步 leader 端口 -> 向 bootstrap server 注册本 rank
            #       -> 初始化传输信息表与超时阈值。
            # When SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER is True, all CP ranks
            # participate in KV transfer; Otherwise only CP rank 0 sends.
            self.is_dummy_cp_rank = (
                not self.enable_all_cp_ranks_for_transfer
                and self.attn_cp_size > 1
                and self.attn_cp_rank != 0
            )
            # Sync the leader's bootstrap port to every rank before
            # registering: in multi-node prefill, registration targets
            # `dist_init_addr` (rank 0) but each rank's local port may
            # differ when the launcher auto-reserves a free port per host.
            self.bootstrap_port = self._sync_bootstrap_port_across_nodes(
                self.bootstrap_port
            )
            self.register_to_bootstrap()
            self.transfer_infos = {}
            self.req_to_decode_prefix_len: Dict[int, int] = {}
            self.decode_kv_args_table = {}
            self.pp_group = get_pp_group()
            # If a timeout happens on the prefill side, it means prefill instances
            # fail to receive the KV indices from the decode instance of this request.
            # These timeout requests should be aborted to release the tree cache.
            self.bootstrap_timeout = envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # 中译：DECODE 角色：维护连接池/拓扑缓存/心跳失败计数/HTTP 会话池，
            #       并记录 bootstrap_addr <-> 请求 room 的反向映射，用于节点故障时定位受影响请求。
            self.enable_staging: bool = False
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            self.connection_lock = threading.Lock()
            self.required_prefill_response_num_table: Dict[int, int] = {}
            self.prefill_info_table: Dict[str, PrefillServerInfo] = {}
            self.heartbeat_failures: Dict[str, int] = {}
            self.session_pool: Dict = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            self.addr_to_rooms_tracker: Dict[str, Set[int]] = defaultdict(set)
            self.prefill_response_tracker: Dict[int, Set[int]] = defaultdict(set)
            # Heartbeat interval should be at least 2 seconds
            self.heartbeat_interval = max(
                envs.SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL.get(), 2.0
            )
            # Heartbeat failure should be at least 1
            self.max_failures = max(
                envs.SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE.get(), 1
            )
            # If a timeout happens on the decode side, it means decode instances
            # fail to receive the KV Cache transfer done signal after bootstrapping.
            # These timeout requests should be aborted to release the tree cache.
            self.waiting_timeout = envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def check_status(self, bootstrap_room: int) -> KVPoll:
        # 中译：返回指定请求（bootstrap_room）当前的 KV 传输轮询状态。
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        # 中译：更新请求状态。状态只能单调推进（取 max），Failed 为终态；
        #       但已被 clear() 清除的 room 不得被迟到的 Failed “复活”，否则会
        #       污染复用同一 bootstrap_room 的未来请求。
        if bootstrap_room not in self.request_status:
            # Do not resurrect a cleared entry with Failed: once clear() has
            # popped the room from request_status, any late update_status(Failed)
            # (e.g. from abort()) must be a no-op. Otherwise a Failed entry could
            # pollute a future request that reuses the same bootstrap_room.
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
            url = f"http://{bootstrap_addr}/route?prefill_dp_rank={-1}&prefill_cp_rank={-1}&target_tp_rank={-1}&target_pp_rank={-1}"
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

        # Sanity checks
        if info.page_size is not None and info.page_size != self.kv_args.page_size:
            raise RuntimeError(
                f"Page size mismatch: prefill server has page_size={info.page_size}, "
                f"but decode server has page_size={self.kv_args.page_size}. "
                f"Both servers must use the same --page-size value."
            )

        if (
            info.kv_cache_dtype is not None
            and info.kv_cache_dtype != self.server_args.kv_cache_dtype
        ):
            raise RuntimeError(
                f"KV cache dtype mismatch: prefill server has kv_cache_dtype={info.kv_cache_dtype}, "
                f"but decode server has kv_cache_dtype={self.server_args.kv_cache_dtype}. "
                f"Both servers must use the same --kv-cache-dtype value."
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
        # TP rank mapping
        if self.attn_tp_size == info.attn_tp_size:
            target_tp_rank = self.kv_args.engine_rank % self.attn_tp_size
            required_dst_info_num = 1
            required_prefill_response_num = 1
            target_tp_ranks = [target_tp_rank]
        elif self.attn_tp_size > info.attn_tp_size:
            if not self.is_mla_backend:
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
            if not self.is_mla_backend:
                logger.warning_once(
                    "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
                )
            # For non-MLA models, one decode rank needs to retrieve KVCache from multiple prefill ranks
            target_tp_ranks = list(
                range(
                    (self.kv_args.engine_rank % self.attn_tp_size)
                    * (info.attn_tp_size // self.attn_tp_size),
                    (self.kv_args.engine_rank % self.attn_tp_size + 1)
                    * (info.attn_tp_size // self.attn_tp_size),
                )
            )
            # For MLA models, we can retrieve KVCache from only one prefill rank, but we still need to maintain
            # multiple connections in the connection pool and have to send dummy requests to other prefill ranks,
            # or the KVPoll will never be set correctly
            target_tp_rank = target_tp_ranks[0]
            required_dst_info_num = 1
            if self.is_mla_backend:
                required_prefill_response_num = 1
            else:
                required_prefill_response_num = info.attn_tp_size // self.attn_tp_size

        # CP rank mapping — decode cp size should be equal to 1
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
            if not self.enable_all_cp_ranks_for_transfer:
                # Only retrieve from prefill CP rank 0 when not using all ranks
                target_cp_ranks = target_cp_ranks[:1]
                required_prefill_response_num *= 1
            else:
                required_prefill_response_num *= info.attn_cp_size // self.attn_cp_size

        # PP rank mapping — decode pp size should be equal to prefill pp size or 1
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
        """
        if not self.dist_init_addr or self.server_args.nnodes == 1:
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
            # Multi-node case: bootstrap server's host is dist_init_addr
            host = NetworkAddress.parse(self.dist_init_addr).resolved().host
        else:
            # Single-node case: bootstrap server's host is the same as http server's host
            host = self.bootstrap_host
            # If the server was bound to the wildcard address (0.0.0.0 / ::), use the
            # actual local IP instead — a PUT to http://0.0.0.0:<port>/route is rejected
            # with 403 by aiohttp ≥3.9 because 0.0.0.0 is not a valid HTTP Host value.
            if host in ("0.0.0.0", "::"):
                host = self.local_ip

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
            "kv_cache_dtype": self.server_args.kv_cache_dtype,
            "load_balance_method": self.server_args.load_balance_method,
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
                # Walk to root cause to skip misleading urllib3 wrapper messages
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
            return sock

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
            # Case: Decode has draft model KV while Prefill is deployed without speculative decoding
            # dst_kv_ptrs layout: [K_main..., V_main..., draft_K..., draft_V...]
            multiplier_ratio = dst_num_total_layers // num_kv_layers
            dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
            v_ptr_offset = num_kv_layers * multiplier_ratio
            dst_v_ptrs = dst_kv_ptrs[
                v_ptr_offset + start_layer : v_ptr_offset + end_layer
            ]
        else:
            # Decode pp size should be equal to prefill pp size or 1
            dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
            dst_v_ptrs = dst_kv_ptrs[
                dst_num_total_layers + start_layer : dst_num_total_layers + end_layer
            ]
        layers_current_pp_stage = len(src_k_ptrs)
        return src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage

    def get_mla_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int]
    ) -> Tuple[List[int], List[int], int]:
        # Fast path: both sides use exactly the same PP layout
        if len(src_kv_ptrs) == len(dst_kv_ptrs):
            return src_kv_ptrs, dst_kv_ptrs, len(src_kv_ptrs)

        mla_ratios = getattr(self.kv_args, "mla_compression_ratios", None)
        if mla_ratios:
            # Compressed-MLA (e.g. DeepSeek V4): the flat list is organized
            # by buffer type (compression-ratio bucket) rather than by
            # layer, so we locate the sub-range for this PP stage inside each
            # section of the dst flat list.
            sliced_src_kv_ptrs, sliced_dst_kv_ptrs = self._mla_slice_ptrs_for_pp(
                src_kv_ptrs, dst_kv_ptrs, mla_ratios
            )
            return (
                sliced_src_kv_ptrs,
                sliced_dst_kv_ptrs,
                len(sliced_src_kv_ptrs),
            )

        # Regular MLA PP slicing
        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + len(src_kv_ptrs)
        # Decode pp size should be equal to prefill pp size or 1
        sliced_dst_kv_ptrs = dst_kv_ptrs[start_layer:end_layer]
        return src_kv_ptrs, sliced_dst_kv_ptrs, len(src_kv_ptrs)

    def _mla_slice_ptrs_for_pp(
        self,
        src_kv_ptrs: List[int],
        dst_kv_ptrs: List[int],
        mla_ratios: List[int],
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

        - state_data layout, length = swa_L + 2 * c4_L + c128_L:
            [swa_layer_{0..swa_L-1},
             compress_state_{non-None, c4_L + c128_L},
             indexer_compress_state_{non-None, c4_L}]
          ``swa_L`` is the SWA pool's actual buffer count
          (``num_effective_layers``), which can be smaller than
          ``len(mla_ratios)`` when the HF config's ``compress_ratios``
          list contains entries for layers not materialized into the SWA
          pool (e.g. an MTP/nextn slot at the tail).

        src is already PP-filtered on the prefill side. dst is the
        decode-side full-model list (when decode is PP=1). We slice dst to
        match src's PP stage. If src itself is also full-model, it is
        returned unchanged.
        """
        start_layer = self.kv_args.prefill_start_layer
        end_layer = getattr(self.kv_args, "prefill_end_layer", None)
        assert end_layer is not None, (
            "KVArgs.prefill_end_layer must be set when using "
            "compressed-MLA PD with PP"
        )

        c4_full = sum(1 for r in mla_ratios if r == 4)
        c128_full = sum(1 for r in mla_ratios if r == 128)
        kv_layout_len = 2 * c4_full + c128_full

        c4_off_s = sum(1 for r in mla_ratios[:start_layer] if r == 4)
        c4_off_e = sum(1 for r in mla_ratios[:end_layer] if r == 4)
        c128_off_s = sum(1 for r in mla_ratios[:start_layer] if r == 128)
        c128_off_e = sum(1 for r in mla_ratios[:end_layer] if r == 128)

        if len(dst_kv_ptrs) == kv_layout_len:
            sliced_dst = (
                list(dst_kv_ptrs[c4_off_s:c4_off_e])
                + list(dst_kv_ptrs[c4_full + c4_off_s : c4_full + c4_off_e])
                + list(dst_kv_ptrs[2 * c4_full + c128_off_s : 2 * c4_full + c128_off_e])
            )
            return src_kv_ptrs, sliced_dst

        # State-data layout. ``swa_L`` is derived from the actual dst
        # length so we tolerate cases where the SWA pool has fewer
        # buffers than ``len(mla_ratios)`` (e.g. nextn padding).
        swa_L = len(dst_kv_ptrs) - 2 * c4_full - c128_full
        if swa_L < 0 or swa_L > len(mla_ratios):
            raise ValueError(
                f"Unexpected compressed-MLA dst_kv_ptrs length "
                f"{len(dst_kv_ptrs)}; expected either {kv_layout_len} "
                f"(kv_data) or swa_L + {2 * c4_full + c128_full} "
                f"(state_data) given compression_ratios "
                f"(c4={c4_full}, c128={c128_full}, "
                f"total={len(mla_ratios)})."
            )
        # Guard against asking the prefill side to read past the SWA
        # pool boundary.
        assert end_layer <= swa_L, (
            f"prefill_end_layer ({end_layer}) exceeds dst SWA pool "
            f"buffer count ({swa_L}); compression_ratios may include "
            f"layers (e.g. nextn) that the SWA pool does not cover."
        )

        # compress_state non-None count up to L = count(r != 0).
        c_non_zero_s = sum(1 for r in mla_ratios[:start_layer] if r != 0)
        c_non_zero_e = sum(1 for r in mla_ratios[:end_layer] if r != 0)
        compress_section_start = swa_L
        indexer_section_start = swa_L + (c4_full + c128_full)
        sliced_dst = (
            list(dst_kv_ptrs[start_layer:end_layer])
            + list(
                dst_kv_ptrs[
                    compress_section_start
                    + c_non_zero_s : compress_section_start
                    + c_non_zero_e
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
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
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
        """Hook called on successful heartbeat. Override for backend-specific cleanup."""
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
            # Collect TCP endpoints from cached bootstrap_infos before deletion
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

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
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
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.conclude_state: Optional[KVPoll] = None
        self._transfer_metric = KVTransferMetric()
        self._transfer_num_kv_indices = 0
        self._transfer_num_state_indices = 0
        # inner state
        self.curr_idx = 0
        self.init_time: Optional[float] = None
        if self.kv_mgr.is_dummy_cp_rank:
            # Non-authoritative CP ranks are dummy participants.
            # 中译：非权威 CP rank 为 dummy 参与者，不实际发送，直接置为等待输入。
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)
            return

        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        # 中译：多 DP 时需确定该请求归属哪个 prefill dp_rank：
        #       非 follow_bootstrap_room 策略直接注册；若为 follow_bootstrap_room
        #       但实际路由与 room%dp_size 不一致，则根据开关决定强制注册或直接失败。
        if self.kv_mgr.server_args.dp_size > 1:
            if self.kv_mgr.server_args.load_balance_method != "follow_bootstrap_room":
                self._register_prefill_dp_rank()
            elif (
                self.kv_mgr.attn_dp_rank
                != self.bootstrap_room % self.kv_mgr.server_args.dp_size
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
                        f"{self.bootstrap_room % self.kv_mgr.server_args.dp_size}. "
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

        if self.kv_mgr.enable_all_cp_ranks_for_transfer:
            kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
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

    def poll(self) -> KVPoll:
        pass

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")

    def clear(self) -> None:
        # 中译：清理本请求在管理器上的残留状态（状态、前缀长度、传输信息），
        #       供 bootstrap_room 安全复用。
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        if hasattr(self.kv_mgr, "req_to_decode_prefix_len"):
            self.kv_mgr.req_to_decode_prefix_len.pop(self.bootstrap_room, None)
        if hasattr(self.kv_mgr, "transfer_infos"):
            self.kv_mgr.transfer_infos.pop(self.bootstrap_room, None)

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

            if bootstrap_key not in self.kv_mgr.connection_pool:
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
                            return

                self.bootstrap_infos = bootstrap_infos
                self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos

                # Register kv_args only once to prefill KVManager according to the info fetched from the bootstrap server
                self._register_kv_args()
            else:
                self.bootstrap_infos = self.kv_mgr.connection_pool[bootstrap_key]

            assert len(self.bootstrap_infos) > 0
            all_bootstrap_infos.extend(self.bootstrap_infos)

        self.bootstrap_infos = all_bootstrap_infos

    def _get_bootstrap_info_from_server(
        self, prefill_dp_rank, prefill_cp_rank, target_tp_rank, target_pp_rank
    ):
        """Fetch the bootstrap info from the bootstrap server.

        中译：通过 GET /route 按 (dp, cp, tp, pp) 四元组向 bootstrap server 查询
        具体目标 Prefill rank 的 (rank_ip, rank_port)；失败返回 None。
        """
        try:
            url = f"http://{self.bootstrap_addr}/route?prefill_dp_rank={prefill_dp_rank}&prefill_cp_rank={prefill_cp_rank}&target_tp_rank={target_tp_rank}&target_pp_rank={target_pp_rank}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bootstrap_info = response.json()
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
            response = requests.post(
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

    def _register_kv_args(self):
        pass

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
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
        if (
            not self.abort_notified
            and hasattr(self, "bootstrap_infos")
            and self.bootstrap_infos is not None
        ):
            self._send_abort_notification()
            self.abort_notified = True
        return KVPoll.Failed

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")

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
        self._setup_routes()
        self.pp_size = None
        self.attn_tp_size = None
        self.attn_cp_size = None
        self.dp_size = None
        self.page_size = None
        self.kv_cache_dtype: Optional[str] = None
        self.follow_bootstrap_room: Optional[bool] = None
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

        if self.follow_bootstrap_room is None:
            load_balance_method = data.get(
                "load_balance_method", "follow_bootstrap_room"
            )
            self.follow_bootstrap_room = load_balance_method == "follow_bootstrap_room"

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

            self._loop.create_task(self._cleanup_expired_entries())

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
