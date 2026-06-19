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
"""A controller that dispatches requests to multiple data parallel workers.

中译：数据并行（Data Parallel, DP）控制器。它位于 TokenizerManager（分词器进程）与多个
      Scheduler（调度器）工作进程之间，负责把已分词的请求按某种负载均衡策略分发到不同的
      DP worker（每个 worker 内部又是一组 TP/PP 进程）。
      核心职责：
      1. 启动并管理所有 DP worker 的调度进程（普通 DP 模式 / DP attention 模式两套启动路径）。
      2. 维护各 worker 的存活状态，并以多种负载均衡方法（轮询、bootstrap room、按请求数、
         按 token 数）选择目标 worker。
      3. 通过 ZMQ 与上游（TokenizerManager）和下游（各 Scheduler）通信，转发请求与控制消息。
"""

import faulthandler
import logging
import multiprocessing as mp
import signal
import threading
import time
from enum import Enum, auto
from typing import Callable, List, Optional

import psutil
import setproctitle
import zmq

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    BlockReqInput,
    ProfileReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.load_snapshot import create_load_snapshot_reader
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.observability.req_time_stats import DPControllerReqTimeStats
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.server_args import (
    DP_ATTENTION_HANDSHAKE_PORT_DELTA,
    PortArgs,
    ServerArgs,
)
from sglang.srt.utils import numa_utils
from sglang.srt.utils.common import (
    configure_logger,
    kill_itself_when_parent_died,
    maybe_reindex_device_id,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    bind_port,
    get_zmq_socket,
    get_zmq_socket_on_host,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)

# 中译：父进程向上游回传 scheduler 进程 PID 列表时所用的键名（见进程入口函数末尾）。
SCHEDULER_PIDS_ARG = "scheduler_pids"


class LoadBalanceMethod(Enum):
    """Load balance method.

    中译：负载均衡方法枚举。决定 DataParallelController 把请求分发到哪个 DP worker。
    - ROUND_ROBIN：简单轮询。
    - FOLLOW_BOOTSTRAP_ROOM：按请求的 bootstrap_room 取模，保证同一会话落到固定 worker
      （PD 分离/预填充-解码协同场景需要 prefill 和 decode 命中同一 rank）。
    - TOTAL_REQUESTS：选当前累计请求数最少的 worker。
    - TOTAL_TOKENS：选当前累计 token 数最少的 worker（以请求数做次级 tie-break）。
    """

    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
    TOTAL_REQUESTS = auto()
    TOTAL_TOKENS = auto()

    @classmethod
    def from_str(cls, method: str):
        # 中译：从字符串（不区分大小写）解析为枚举值，非法值抛 ValueError。
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DPBudget:
    """中译：DP 负载预算表。为「按请求数 / 按 token 数」均衡策略维护每个 DP rank 的当前负载估计。

    数据来源是各 Scheduler 写入共享内存的负载快照（load snapshot）；本类在两次快照刷新之间
    用「推测式 +1」累加（每分发一个请求就给目标 rank 的计数加一），避免一波突发请求因读到的是
    同一份过期快照而全部涌向同一个 rank。
    """

    def __init__(self, dp_size: int):
        # 中译：dp_size 个 DP rank；分别维护累计请求数、累计 token 数、上次快照时间戳。
        self.dp_size = dp_size
        self.total_requests = [0] * dp_size
        self.total_tokens = [0] * dp_size
        self.last_timestamp = [0.0] * dp_size

    def update_budget(self, loads):
        """Update budget from shm snapshots, skipping stale reads.

        中译：用共享内存里的负载快照刷新预算表，跳过时间戳未变化的过期读取。
              刷新会用快照里的真实负载覆盖此前的推测式 +1 累加值。
        """
        for load in loads:
            # 中译：时间戳未变化说明该 rank 的快照还没更新，跳过避免回退到旧值。
            if load.timestamp == self.last_timestamp[load.dp_rank]:
                continue
            self.last_timestamp[load.dp_rank] = load.timestamp
            self.total_requests[load.dp_rank] = (
                load.num_running_reqs + load.num_waiting_reqs
            )
            self.total_tokens[load.dp_rank] = load.num_total_tokens

    def dispatch(self, method: LoadBalanceMethod, estimated_tokens: int = 0):
        """中译：按指定均衡方法选出目标 DP rank。

        参数 estimated_tokens：本请求预估的 token 数（仅 TOTAL_TOKENS 用到）。
        返回选中的 rank 下标；方法不支持时返回 None。副作用：会对选中 rank 做推测式累加。
        """
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            # 中译：选累计请求数最少的 rank。
            target_rank = self.total_requests.index(min(self.total_requests))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            # Use total_requests as a tie-breaker when total_tokens are equal
            # 中译：选累计 token 数最少的 rank；token 数相同时用累计请求数做次级排序。
            target_rank = min(
                range(self.dp_size),
                key=lambda i: (self.total_tokens[i], self.total_requests[i]),
            )
        else:
            return None

        # Increment the load of that worker by one as a heuristic
        # 中译：推测式累加——把选中 worker 的负载先加上，使同一波突发请求能被打散到不同 rank，
        #       而不必等下一次快照刷新。
        self.total_requests[target_rank] += 1
        self.total_tokens[target_rank] += estimated_tokens
        return target_rank


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers.

    中译：数据并行控制器。把分词后的请求分发到多个 DP worker，并维护它们的存活状态、
          负载均衡策略与进程间通信。关键协作对象：
          - 上游 TokenizerManager（经 recv_from_tokenizer 拉取已分词请求）。
          - 下游各 Scheduler worker（经 self.workers 的 PUSH 套接字发送请求/控制消息）。
          - DPBudget / load_snapshot_reader（负载均衡所需的负载视图）。
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> None:
        # Parse args
        # 中译：解析配置，并把负载均衡方法字符串解析为枚举；run_scheduler_process_func 是
        #       启动单个 scheduler 进程的入口函数（可注入以便测试/定制）。
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )
        self.run_scheduler_process_func = run_scheduler_process_func

        # Init inter-process communication
        # 中译：初始化 ZMQ 上下文（IO 线程数 = 1 + dp_size）。仅 node_rank==0 的主节点直接从
        #       TokenizerManager 拉取请求；其他节点不接请求，只跟着主节点跑 worker。
        self.context = zmq.Context(1 + server_args.dp_size)
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )

        # Dispatch method
        # 中译：根据负载均衡方法选定对应的分发函数 self.dispatching；并记录哪些方法在每次分发前
        #       需要刷新负载预算（仅 TOTAL_REQUESTS / TOTAL_TOKENS 依赖实时负载）。
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
            LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
            LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]
        self.refresh_load_budget_on_dispatch = self.load_balance_method in (
            LoadBalanceMethod.TOTAL_REQUESTS,
            LoadBalanceMethod.TOTAL_TOKENS,
        )

        # Load balance budget
        # 中译：负载预算表与负载快照读取器（从共享内存读各 scheduler 的运行/等待请求数等）。
        self.dp_budget = DPBudget(server_args.dp_size)
        self.load_snapshot_reader = create_load_snapshot_reader(
            server_args,
            port_args,
            caller="DataParallelController",
        )
        self._last_refresh_time = 0.0

        # To protect changing env vars to set CUDA_VISIBLE_DEVICES.
        # 中译：保护「修改环境变量设置 CUDA_VISIBLE_DEVICES」这一全局操作的锁，避免多线程并发
        #       启动 worker 时彼此踩到对方设置的可见设备环境变量。
        self.env_lock = threading.Lock()

        # Launch data parallel workers
        # 中译：scheduler_procs 保存所有已启动的 scheduler 子进程；workers 是发往各 DP worker 的
        #       PUSH 套接字（下标即 dp_rank）；status 标记各 worker 是否存活/可用。
        self.scheduler_procs = []
        self.workers: List[zmq.Socket] = [None] * server_args.dp_size
        self.status: List[bool] = [True] * server_args.dp_size

        if server_args.enable_dp_attention:
            # 中译：DP attention 模式——所有 DP rank 复用同一 TP group，启动路径与端口分配不同。
            self.launch_dp_attention_schedulers(server_args, port_args)
            # When local control broadcast is enabled, send control messages to
            # every DP group leader (attn_tp_rank=0) so each leader broadcasts
            # within its own attn_tp_group instead of the full tp_group.
            # Otherwise fall back to the original behaviour: send to only the
            # first leader, which then broadcasts over the full tp_group.
            local_ctrl = server_args.enable_dp_attention_local_control_broadcast
            # 中译：control_message_step 是发送控制消息时在 workers 列表上的步长——开启本地控制
            #       广播时每个 DP 组的 leader 都要收到（步长 1）；否则只发给首个 leader，由它在整个
            #       tp_group 内广播（步长 = tp_size）。
            self.control_message_step = 1 if local_ctrl else server_args.tp_size
        else:
            # 中译：普通 DP 模式——每个 DP rank 各自独立的 TP group，逐个启动。
            self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        # 中译：初始化基于消息类型的请求分发器（路由不同请求/控制消息到对应处理函数）。
        self.init_dispatcher()

        # 中译：软看门狗，检测控制器是否卡死（soft 模式只告警不直接杀进程）。
        self.soft_watchdog = Watchdog.create(
            debug_name="DataParallelController",
            watchdog_timeout=server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DP_CONTROLLER.get(),
        )

        if server_args.enable_metrics:
            # 中译：启用指标时启动 CPU 监控线程，上报本控制器进程的 CPU 使用情况。
            start_cpu_monitor_thread("data_parallel_controller")

    def send_to_all_workers(self, obj):
        # 中译：把对象广播给所有「存活」的 worker（用于阻塞/解阻塞、profile 等全局控制消息）。
        for i, worker in enumerate(self.workers):
            if self.status[i]:
                worker.send_pyobj(obj)

    def send_control_message(self, obj):
        # Send control messages to first worker of tp group
        # 中译：把控制消息按 control_message_step 步长发给各 TP group 的 leader，由 leader 在组内广播。
        for worker in self.workers[:: self.control_message_step]:
            worker.send_pyobj(obj)

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        # 中译：更新各 DP rank 的存活状态（如某 worker 故障/恢复后由调度侧通知）。
        self.status = ranks.status

    def refresh_load_budget(self):
        # 中译：从负载快照刷新预算表，但做 20ms 节流。突发请求时若每次分发都刷新，会用（同一份
        #       尚未更新的）快照覆盖掉本波已累加的推测式 +1，导致整波请求全压到同一个 DP rank。
        #       节流让一波请求先靠推测式计数打散，下一批再用真实负载刷新。
        # Throttle to at most once per 20ms.  When a burst of requests
        # arrives, dispatching_with_trace() calls this before every
        # dispatch.  Each call reads the latest scheduler snapshot and
        # overwrites the speculative +1 increments that DPBudget.dispatch()
        # added for previously dispatched requests in this burst.  Without
        # throttling, the budget resets to the (stale) scheduler-reported
        # value on every request, causing the entire burst to land on a
        # single DP rank.  The 20ms interval lets the burst complete
        # using speculative counters, then refreshes from the real
        # scheduler load for the next batch.
        now = time.perf_counter()
        if now - self._last_refresh_time < 0.02:
            return
        self._last_refresh_time = now
        self.dp_budget.update_budget(self.load_snapshot_reader.read_all())

    def dispatching_with_trace(self, req: Req, refresh_load_budget: bool = True):
        # 中译：分发单个请求并打点（trace）。需要时先刷新负载预算，记录分发开始/完成时间戳，
        #       再调用选定的均衡函数 self.dispatching 真正发送。
        #       refresh_load_budget=False 用于批量分发时避免每条都重复刷新。
        if refresh_load_budget and self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()

        req.time_stats = DPControllerReqTimeStats.new_from_obj(req.time_stats)

        req.time_stats.set_dp_dispatch_time()
        self.dispatching(req)
        req.time_stats.set_dp_dispatch_finish_time()

    def dispatch_batch_generate(self, batch_req: BatchTokenizedGenerateReqInput):
        # 中译：分发一批生成请求。先统一刷新一次负载预算，再逐条分发（逐条不再重复刷新）。
        if self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()
        for req in batch_req:
            self.dispatching_with_trace(req, refresh_load_budget=False)

    def dispatch_batch_embedding(self, batch_req: BatchTokenizedEmbeddingReqInput):
        # 中译：分发一批 embedding 请求，逻辑同上。
        if self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()
        for req in batch_req:
            self.dispatching_with_trace(req, refresh_load_budget=False)

    def init_dispatcher(self):
        # 中译：初始化基于类型的请求分发器，将各类输入路由到对应处理函数：
        #       单条生成/embedding → 带 trace 的分发；批量 → 批量分发；
        #       Block/Profile 请求 → 广播给所有 worker；ActiveRanks 输出 → 更新存活状态。
        #       兜底（未匹配类型）走 send_control_message，按 leader 广播控制消息。
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.dispatching_with_trace),
                (TokenizedEmbeddingReqInput, self.dispatching_with_trace),
                (BatchTokenizedGenerateReqInput, self.dispatch_batch_generate),
                (BatchTokenizedEmbeddingReqInput, self.dispatch_batch_embedding),
                (BlockReqInput, self.send_to_all_workers),
                (ProfileReq, self.send_to_all_workers),
                (ActiveRanksOutput, self.update_active_ranks),
            ]
        )
        self._request_dispatcher.add_fallback_fn(self.send_control_message)

    def launch_dp_schedulers(self, server_args, port_args):
        """中译：普通 DP 模式下逐个启动各 DP rank 的 TP group。

        每个 DP rank 用独立线程启动其 scheduler 进程组，并各自分配独立的 nccl 端口与 GPU 区间；
        最后阻塞等待所有 worker 就绪（ready_event）。共享同一对 tokenizer/detokenizer IPC 名。
        """
        # 中译：base_gpu_id 是当前 DP rank 起始 GPU 偏移，每启动一个 rank 就按其 TP*PP 规模递增。
        base_gpu_id = 0

        threads = []
        sockets = []
        ready_events = []
        for dp_rank in range(server_args.dp_size):
            # 中译：为该 dp_rank 生成一套新端口，但复用全局共享的 tokenizer/detokenizer IPC 名与 instance_id。
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name
            tmp_port_args.instance_id = port_args.instance_id

            # This port is checked free in PortArgs.init_new.
            # We hold it first so that the next dp worker gets a different port
            # 中译：nccl 端口已在 init_new 里确认空闲；这里先占住它，确保下一个 dp worker 分到不同端口。
            sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            # Create a thread for each worker
            # 中译：为每个 worker 创建一个启动线程（并行启动以缩短总启动时间）。
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            base_gpu_id += (
                server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
            )

            if server_args.node_rank == 0:
                # 中译：主节点为该 dp_rank 建立 PUSH 套接字，后续请求经此发往对应 scheduler。
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,
                )

        # Free all sockets before starting the threads to launch TP workers
        # 中译：启动线程前先释放占位的 nccl 端口，让真正的 TP worker 能用上这些端口。
        for sock in sockets:
            sock.close()

        # Start all threads
        # 中译：启动所有线程，并阻塞等待每个 worker 就绪信号。
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

    def launch_tensor_parallel_group_thread(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        ready_event: threading.Event,
    ):
        # 中译：线程目标函数——启动该 dp_rank 的 TP group，启动完成后置位 ready_event 通知主流程。
        self.launch_tensor_parallel_group(server_args, port_args, base_gpu_id, dp_rank)
        ready_event.set()

        # This thread cannot be closed because otherwise the `kill_itself_when_parent_died`
        # function in scheduler.py will kill the scheduler.
        # 中译：此线程必须常驻不能退出，否则 scheduler.py 里的 kill_itself_when_parent_died
        #       会把对应 scheduler 进程当作「父进程已死」而误杀。故用超长 sleep 永久挂起。
        while True:
            time.sleep(30 * 24 * 3600)

    def _broadcast_worker_ports(
        self, server_args: ServerArgs, worker_ports: Optional[List[int]] = None
    ) -> List[int]:
        """Broadcast worker ports from node 0 to all other nodes.

        Node 0 acts as the server, waiting for all other nodes to connect and
        sending them the pre-allocated worker ports. Other nodes act as clients,
        connecting to node 0 to receive their copy of the worker ports.

        Args:
            server_args: Server arguments containing node configuration.
            worker_ports: Pre-allocated worker ports to broadcast.

        Returns:
            List of worker ports (same on all nodes after broadcast).

        中译：在 DP attention 多机部署中，把 node 0 预分配好的 worker 端口广播给其他所有节点。
              node 0 充当服务端，等待其余节点连接并下发端口；其余节点充当客户端，连接 node 0
              获取这份端口列表。广播后所有节点持有相同的端口列表，从而协调一致地建立连接。
              参数 worker_ports：node 0 预分配待广播的端口（其他节点传入 None）。
              返回：广播后各节点一致的 worker 端口列表。
        """
        # Determine the endpoint for inter-node communication
        # 中译：确定跨节点通信的端点地址（在分布式初始化地址或 host:port 基础上加固定偏移）。
        if server_args.dist_init_addr is None:
            na = NetworkAddress(
                server_args.host or "127.0.0.1",
                server_args.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA,
            )
        else:
            na = NetworkAddress.parse(server_args.dist_init_addr)
            na = NetworkAddress(na.host, na.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA)
        endpoint = na.to_tcp()

        if server_args.node_rank == 0:
            # Node 0: Broadcast worker ports to all other nodes
            # 中译：node 0 作为服务端广播端口（等待 nnodes-1 个客户端节点连接）。
            return self._broadcast_ports_as_server(
                endpoint, server_args.nnodes - 1, worker_ports
            )
        else:
            # Other nodes: Receive worker ports from node 0
            # 中译：其余节点作为客户端从 node 0 接收端口。
            return self._receive_ports_as_client(endpoint, server_args.node_rank)

    def _broadcast_ports_as_server(
        self, endpoint: str, expected_clients: int, worker_ports: List[int]
    ) -> List[int]:
        """Broadcast worker ports to all client nodes.

        中译：作为服务端（node 0）逐个回应客户端握手并下发端口，直到所有期望的客户端都已连接。
              结束后：若未启用弹性 EP（elastic_ep_backend 为 None）则关闭套接字；否则另起后台线程
              继续应答，以便后续恢复的 EP rank 也能拿到端口。
        """
        logger.debug(f"Broadcasting worker ports to {expected_clients} client nodes")
        logger.debug(f"Worker ports: {worker_ports}")

        # 中译：建立 REP（应答）套接字，按「收握手 → 回端口」的请求-应答模式服务各客户端。
        rep_socket = get_zmq_socket(self.context, zmq.REP, endpoint, True)

        try:
            connected_clients = 0
            while connected_clients < expected_clients:
                # Wait for client handshake
                # 中译：等待某个客户端节点发来握手（携带其 node rank）。
                client_rank = rep_socket.recv().decode()
                logger.debug(f"Received handshake from node {client_rank}")

                # Send worker ports to client
                # 中译：把端口列表回给该客户端，并累计已连接数。
                rep_socket.send_pyobj(worker_ports)
                connected_clients += 1
                logger.debug(
                    f"Sent worker ports to {connected_clients}/{expected_clients} nodes"
                )

            logger.debug("Worker port broadcast completed")
            return worker_ports
        finally:
            # 中译：未启用弹性 EP 时直接关闭套接字；否则交给后台线程持续应答（供 EP 恢复使用）。
            if self.server_args.elastic_ep_backend is None:
                rep_socket.close()
            else:
                threading.Thread(
                    target=self._reply_ports_as_server,
                    args=(rep_socket, worker_ports),
                    daemon=True,
                ).start()

    def _reply_ports_as_server(self, rep_socket: zmq.Socket, worker_ports: List[int]):
        """
        Runs as a background thread to broadcast worker ports for recovered EP ranks

        中译：作为后台线程长期运行，为「恢复后的 EP rank」持续应答端口请求（弹性 EP 场景）。
              单次 recv/decode 失败不致命，记录异常后继续循环等待下一次握手。
        """
        while True:
            # Wait for client handshake
            # 中译：等待客户端握手；失败则记录异常并继续，不退出线程。
            try:
                client_rank = rep_socket.recv().decode()
            except Exception:
                logger.exception(
                    "Failed to recv/decode handshake in reply thread; continue"
                )
                continue
            logger.debug(f"Received handshake from node {client_rank}")

            # Send worker ports to client
            rep_socket.send_pyobj(worker_ports)
            logger.debug(f"Sent worker ports to node {client_rank}")

    def _receive_ports_as_client(self, endpoint: str, node_rank: int) -> List[int]:
        """Receive worker ports from the server node.

        中译：作为客户端连接 node 0，发送本节点 rank 握手后接收端口列表。收发均设 10 分钟超时，
              超时则抛 RuntimeError（说明 node 0 未在限定时间内下发端口）。
        """
        logger.debug(f"Connecting to node 0 to receive worker ports")

        # 中译：建立 REQ（请求）套接字并设置收发超时（10 分钟）。
        req_socket = get_zmq_socket(self.context, zmq.REQ, endpoint, False)
        req_socket.setsockopt(zmq.RCVTIMEO, 600 * 1000)  # 10 minute timeout
        req_socket.setsockopt(zmq.SNDTIMEO, 600 * 1000)

        try:
            # Send handshake with our node rank
            # 中译：发送握手，携带本节点的 node_rank。
            req_socket.send(str(node_rank).encode())

            # Receive worker ports
            # 中译：接收 node 0 回传的端口列表。
            worker_ports = req_socket.recv_pyobj()
            logger.debug(f"Received {len(worker_ports)} worker ports from node 0")
            return worker_ports
        except zmq.Again:
            logger.error("Timeout waiting for worker ports from node 0")
            raise RuntimeError(
                "Failed to receive worker ports from node 0 within timeout"
            )
        finally:
            req_socket.close()

    def launch_dp_attention_schedulers(
        self, server_args: ServerArgs, port_args: PortArgs
    ):
        """中译：DP attention 模式下启动调度器。所有 DP rank 复用同一个 TP group，

        因此只调用一次 launch_tensor_parallel_group，由内部按 attention 分片逻辑切分出各 dp_rank。
        端口由 node 0 预分配并广播给所有节点，避免多机端口冲突。
        """
        if server_args.dist_init_addr is None:
            bind_host = "127.0.0.1"
        else:
            bind_host = NetworkAddress.parse(server_args.dist_init_addr).host

        # Pre-allocate worker ports on node 0 to avoid conflicts
        # 中译：在 node 0 上为每个 dp_rank 预分配 PUSH 端口/套接字，避免后续端口冲突。
        worker_ports = []
        if server_args.node_rank == 0:
            for dp_rank in range(server_args.dp_size):
                worker_port, worker_socket = get_zmq_socket_on_host(
                    self.context, zmq.PUSH, host=bind_host
                )
                worker_ports.append(worker_port)
                self.workers[dp_rank] = worker_socket
                logger.debug(
                    "Assigned port %s to worker %s on host %s",
                    worker_port,
                    dp_rank,
                    bind_host,
                )

        # 中译：把端口广播给所有节点，得到全局一致的端口列表。
        broadcasted_ports = self._broadcast_worker_ports(
            server_args, worker_ports if worker_ports else None
        )
        # 中译：以单个 TP group 启动（dp_rank 传 None，由内部按 attention 分片推导各 rank）。
        self.launch_tensor_parallel_group(
            server_args, port_args, 0, None, broadcasted_ports
        )

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):
        """中译：启动一个 TP group 内的全部 scheduler 进程（按本节点负责的 PP/TP rank 区间）。

        负责：根据节点拓扑计算本节点应承载的 pp_rank/tp_rank 范围，为每个 (pp_rank, tp_rank)
        组合算出 GPU id 及各并行维度（attn_cp/moe_dp/moe_ep）的 rank，逐个以子进程方式拉起
        scheduler，并通过管道等待各进程上报模型加载完成的信息（如 max_total_num_tokens）。
        参数 dp_rank：普通 DP 模式下为具体 rank；DP attention 模式下传 None，由内部推导。
        """
        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        # 中译：显存节省适配器（可在空闲时把权重暂存到 CPU 等），按配置启用。
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        # 中译：保存与各 scheduler 子进程通信的管道读端，用于稍后接收其就绪信息。
        scheduler_pipe_readers = []

        # 中译：以下一段根据节点数 nnodes 把 PP/TP 维度切分到各节点，算出本节点负责的 rank 区间。
        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
        )

        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        attn_cp_rank = 0
        moe_dp_rank = 0
        # 中译：遍历本节点负责的每个 (pp_rank, tp_rank) 组合，各拉起一个 scheduler 进程。
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                rank_port_args = port_args

                if server_args.enable_dp_attention:
                    # dp attention has different sharding logic
                    # 中译：DP attention 的分片逻辑不同——由 tp_rank 反推该进程所属的 dp_rank，
                    #       并为该 dp_rank 计算独立的 zmq 端口。
                    _, _, dp_rank, _ = compute_dp_attention_world_info(
                        server_args.enable_dp_attention,
                        tp_rank,
                        server_args.tp_size,
                        server_args.dp_size,
                        server_args.attn_cp_size,
                    )
                    # compute zmq ports for this dp rank
                    rank_port_args = PortArgs.init_new(
                        server_args, dp_rank, worker_ports
                    )
                    # Data parallelism reuses the tensor parallelism group,
                    # so all dp ranks should use the same nccl port.
                    # 中译：DP attention 复用同一 TP group，故所有 dp rank 共用同一个 nccl 端口。
                    rank_port_args.nccl_port = port_args.nccl_port
                    rank_port_args.instance_id = port_args.instance_id

                # 中译：建立单向管道，子进程（writer）启动后用它回传模型加载就绪信息。
                reader, writer = mp.Pipe(duplex=False)
                # 中译：依据 base/偏移/各 rank 计算该进程实际使用的 GPU id。
                gpu_id = (
                    server_args.base_gpu_id
                    + base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                attn_dp_size = (
                    server_args.dp_size if server_args.enable_dp_attention else 1
                )

                # Parallelism hierarchy (outermost to innermost):
                # - Attention: Global(TP) -> DP -> ATTN_CP -> ATTN_TP (innermost)
                # - MoE: Global(TP) -> MOE_DP -> EP -> MOE_TP (innermost)
                # 中译：并行层级（从外到内）。注意力侧：全局 TP → DP → ATTN_CP → ATTN_TP；
                #       MoE 侧：全局 TP → MOE_DP → EP → MOE_TP。
                #       下面据此从 tp_rank 反推该进程在各并行维度上的 rank。
                attn_tp_size = (
                    server_args.tp_size // attn_dp_size // server_args.attn_cp_size
                )
                attn_cp_rank = (tp_rank // attn_tp_size) % server_args.attn_cp_size
                moe_dp_rank = tp_rank // (
                    server_args.tp_size // server_args.moe_dp_size
                )
                moe_ep_rank = (
                    tp_rank
                    % (server_args.tp_size // server_args.moe_dp_size)
                    // (
                        server_args.tp_size
                        // server_args.moe_dp_size
                        // server_args.ep_size
                    )
                )

                # 中译：env_lock 保护设置设备可见性的临界区；maybe_reindex_device_id 可能把
                #       gpu_id 重映射为进程内可见的设备序号。随后以子进程启动 scheduler。
                with self.env_lock, maybe_reindex_device_id(gpu_id) as gpu_id:
                    proc = mp.Process(
                        target=self.run_scheduler_process_func,
                        args=(
                            server_args,
                            rank_port_args,
                            gpu_id,
                            tp_rank,
                            attn_cp_rank,
                            moe_dp_rank,
                            moe_ep_rank,
                            pp_rank,
                            dp_rank,
                            writer,
                        ),
                    )
                    with (
                        memory_saver_adapter.configure_subprocess(),
                        numa_utils.configure_subprocess(server_args, gpu_id),
                    ):
                        proc.start()
                self.scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)

        # Wait for model to finish loading
        # 中译：阻塞等待每个 scheduler 子进程经管道回传「模型加载完成」的信息。
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        # 中译：取首个进程上报的 KV 缓存总 token 容量与最大请求输入长度，作为本控制器的全局上限。
        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]

    def maybe_external_dp_rank_routing(self, req: Req):
        # 中译：若请求已显式指定目标 DP rank（routed_dp_rank），直接路由到该 worker 并返回 True；
        #       否则返回 False，交由后续负载均衡策略选择目标。
        if req.routed_dp_rank is not None:
            logger.debug(f"Direct routing to DP rank {req.routed_dp_rank}")
            self.workers[req.routed_dp_rank].send_pyobj(req)
            return True
        return False

    def round_robin_scheduler(self, req: Req):
        # 中译：轮询调度。跳过已被显式路由的请求；否则从计数器位置起找到下一个存活 worker 发送，
        #       并把计数器向后推进（取模回绕）。
        if self.maybe_external_dp_rank_routing(req):
            return

        while True:
            if self.status[self.round_robin_counter]:
                logger.debug(f"Choose worker {self.round_robin_counter}")
                self.workers[self.round_robin_counter].send_pyobj(req)
                self.round_robin_counter = (self.round_robin_counter + 1) % len(
                    self.workers
                )
                break
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )

    def follow_bootstrap_room_scheduler(self, req: Req):
        # 中译：按 bootstrap_room 取模选 worker，保证同一会话的请求落到固定 rank（PD 分离需 prefill
        #       与 decode 命中同一实例）。bootstrap_room 为空说明请求被直接发到 prefill/decode 而非
        #       经由 router，属误用，故断言报错。
        if self.maybe_external_dp_rank_routing(req):
            return

        assert req.bootstrap_room is not None, (
            "req.bootstrap_room should not be None. Do not send requests directly to "
            "prefill or decode instances; send to the router instead."
        )
        target_rank = req.bootstrap_room % len(self.workers)
        self.workers[target_rank].send_pyobj(req)

    def total_requests_scheduler(self, req: Req):
        # 中译：按「累计请求数最少」选 worker 发送。
        if self.maybe_external_dp_rank_routing(req):
            return
        target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        self.workers[target_worker].send_pyobj(req)

    def total_tokens_scheduler(self, req: Req):
        # 中译：按「累计 token 数最少」选 worker 发送；用输入 token 数作为本请求的预估负载。
        if self.maybe_external_dp_rank_routing(req):
            return
        estimated_tokens = len(req.input_ids)
        target_worker = self.dp_budget.dispatch(
            LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=estimated_tokens
        )
        self.workers[target_worker].send_pyobj(req)

    def event_loop(self):
        # 中译：主事件循环。持续非阻塞地从 TokenizerManager 拉取请求并交分发器处理；
        #       队列暂空（ZMQError）时跳出内层循环，外层再继续轮询。每轮喂软看门狗表示存活。
        while True:
            while True:
                self.soft_watchdog.feed()
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                self._request_dispatcher(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    run_scheduler_process_func: Callable = run_scheduler_process,
):
    # 中译：DP 控制器进程的入口函数。设置进程名、启用故障处理器、注册「父死自杀」，
    #       配置日志与可选的链路追踪，创建控制器并把就绪信息回传给父进程，最后进入事件循环；
    #       出异常时记录堆栈并向父进程发 SIGQUIT 触发整体退出。
    setproctitle.setproctitle("sglang::data_parallel_controller")
    # 中译：启用 faulthandler，崩溃时打印各线程的 Python 调用栈，便于排查。
    faulthandler.enable()
    # 中译：父进程死亡时本进程自杀，避免遗留孤儿进程。
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    configure_logger(server_args)
    if server_args.enable_trace:
        # 中译：启用追踪时初始化追踪上下文，并按 PD 分离角色设置线程标签。
        process_tracing_init(
            server_args.otlp_traces_endpoint,
            "sglang",
            trace_modules=server_args.trace_modules,
        )
        thread_label = "DP Controller"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill DP Controller"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode DP Controller"
        trace_set_thread_info(thread_label)

    try:
        # 中译：创建控制器（其构造过程会启动所有 scheduler 子进程并等待就绪）。
        controller = DataParallelController(
            server_args, port_args, run_scheduler_process_func
        )
        # 中译：收集所有 scheduler 子进程 PID，连同容量上限一起回传给父进程，告知整体就绪。
        scheduler_pids = [
            proc.pid for proc in controller.scheduler_procs if proc is not None
        ]
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
                SCHEDULER_PIDS_ARG: scheduler_pids,
            }
        )
        # 中译：仅主节点跑请求分发事件循环；其余节点不接请求。
        if server_args.node_rank == 0:
            controller.event_loop()
        # 中译：阻塞等待各 scheduler 子进程退出（正常情况下不会发生，退出即视为异常并记录）。
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        # 中译：捕获所有异常，记录完整堆栈，并向父进程发 SIGQUIT 通知整体退出。
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
