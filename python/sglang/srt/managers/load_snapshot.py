"""Load snapshot: publish scheduler load metrics for DP balancing and /v1/loads.

Architecture
------------

Each scheduler periodically publishes a ``LoadSnapshot`` containing its
current load metrics (running reqs, tokens, throughput, ...).  Two
transport backends are supported:

**SHM mode** (single-node, default)::

    Scheduler  ──ShmLoadSnapshotWriter──▶  /dev/shm mmap file
                                               ▲
    TokenizerManager  ──ShmLoadSnapshotReader───┘  (for /v1/loads)
    DataParallelController  ──ShmLoadSnapshotReader─┘  (for dispatch)

**ZMQ mode** (multi-node DP attention, or ``SGLANG_LOAD_SNAPSHOT_USE_ZMQ=1``)::

    Scheduler (any node)  ──ZmqLoadSnapshotWriter (PUSH)──▶  network
                                                               │
    ZmqShmLoadSnapshotReader (PULL, node 0)  ◀─────────────────┘
        │  drains zmq, writes to SHM
        ▼
    /dev/shm mmap file (node 0)
        ▲
    TokenizerManager / DataParallelController  ──ShmLoadSnapshotReader──┘

Shared memory does not work across nodes, so multi-node DP attention
requires the ZMQ transport.  The ``ZmqShmLoadSnapshotReader`` on node 0
receives snapshots from all schedulers via zmq PUSH/PULL and writes them
into the local SHM file.  All readers (TokenizerManager,
DataParallelController) on
node 0 then read from SHM.

``zmq_reader_owner()`` decides which process on node 0 binds the zmq
PULL socket (only one can bind); the other reads plain SHM.

中译：本模块负责「负载快照（LoadSnapshot）」的发布与读取，用于数据并行（DP）的
      负载均衡调度以及对外的 /v1/loads 查询接口。

总体架构：每个 Scheduler 会周期性发布一份 LoadSnapshot，内含其当前负载指标
（运行中请求数、token 数、吞吐、缓存命中率等）。支持两种传输后端：

- SHM 模式（单节点，默认）：Scheduler 用 ShmLoadSnapshotWriter 把快照写入
  /dev/shm 上的 mmap 文件；TokenizerManager（供 /v1/loads）和
  DataParallelController（供调度分发）用 ShmLoadSnapshotReader 直接读该文件。

- ZMQ 模式（多节点 DP attention，或设置 SGLANG_LOAD_SNAPSHOT_USE_ZMQ=1）：
  共享内存无法跨节点，因此各节点的 Scheduler 通过 zmq PUSH 把快照发到网络上；
  0 号节点上的 ZmqShmLoadSnapshotReader（PULL）收取后写入本地 SHM 文件，
  0 号节点上的其他读取者再从 SHM 读取。

  zmq_reader_owner() 决定 0 号节点上由哪个进程来 bind zmq PULL socket
  （只能有一个进程 bind），其余进程退化为直接读 SHM。
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import mmap
import os
import struct
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import msgspec
import msgspec.msgpack
import msgspec.structs

from sglang.srt.environ import envs
from sglang.srt.utils.network import is_zmq_endpoint_ipv6

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import GetLoadsReqOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 中译：PD 分离模式字符串与整数编码的双向映射。快照里用 int 存储更紧凑，
#       读出时再反查回字符串。
DISAGG_MODE_TO_INT = {"null": 0, "prefill": 1, "decode": 2}
INT_TO_DISAGG_MODE = {v: k for k, v in DISAGG_MODE_TO_INT.items()}


def _native(v):
    """Coerce numpy scalars to Python int/float for msgpack encoding.

    中译：把 numpy 标量（如 np.int64/np.float32）转换为 Python 原生 int/float，
          以便 msgpack 能正确编码。numpy 标量带 .item() 方法，调用它即可取出原生值。
    """
    if hasattr(v, "item"):
        return v.item()
    return v


def should_use_zmq(server_args) -> bool:
    """Whether to use zmq PUSH/PULL instead of shared memory for load snapshots.

    Shared memory (mmap) only works within a single node.  When schedulers
    run on multiple nodes (multi-node DP attention), they cannot write to
    the SHM file on node 0, so we fall back to zmq transport.  The env var
    ``SGLANG_LOAD_SNAPSHOT_USE_ZMQ`` forces zmq mode for testing.

    中译：判断负载快照是否应使用 zmq PUSH/PULL 而非共享内存。
          共享内存（mmap）只在单节点内有效；当 Scheduler 跨多节点运行
          （多节点 DP attention）时，它们无法写入 0 号节点的 SHM 文件，
          因此退回到 zmq 传输。环境变量 SGLANG_LOAD_SNAPSHOT_USE_ZMQ 可强制
          开启 zmq 模式（主要用于测试）。
    """
    return (
        server_args.enable_dp_attention and server_args.nnodes > 1
    ) or envs.SGLANG_LOAD_SNAPSHOT_USE_ZMQ.get()


_LOAD_AWARE_METHODS = frozenset({"total_requests", "total_tokens"})


def _tokenizer_load_snapshot_owner_caller(server_args) -> str:
    """The caller that plays the tokenizer-side zmq owner role.

    In multi-tokenizer mode (``tokenizer_worker_num > 1``) there are N
    independent ``TokenizerWorker`` processes that would all try to bind the
    same zmq PULL endpoint.  Instead, the single ``MultiTokenizerRouter``
    process owns the socket (polls zmq -> SHM) and every worker reads SHM.

    中译：返回「tokenizer 侧」充当 zmq owner 角色的那个调用方名称。
          在多 tokenizer 模式（tokenizer_worker_num > 1）下存在 N 个独立的
          TokenizerWorker 进程，它们都想 bind 同一个 zmq PULL 端点会冲突；
          因此改由唯一的 MultiTokenizerRouter 进程持有 socket（轮询 zmq -> 写 SHM），
          每个 worker 只读 SHM。单 tokenizer 模式下则由 TokenizerManager 充当。
    """
    if server_args.tokenizer_worker_num > 1:
        return "MultiTokenizerRouter"
    return "TokenizerManager"


def zmq_reader_owner(server_args, caller: str) -> bool:
    """Decide which process owns the zmq PULL socket.

    Exactly one of ``"DataParallelController"``, ``"TokenizerManager"``, or
    ``"MultiTokenizerRouter"`` must return True when zmq mode is active.  The
    owner polls zmq -> SHM; the others read SHM.

    Rules:
      - Non-zero node_rank: no TokenizerManager, DataParallelController only
        launches schedulers and waits -> nobody owns it.
      - dp_size == 1: no DataParallelController exists -> tokenizer-side owner
        owns it.
      - dp_size > 1, load-aware method: DataParallelController polls on every
        dispatch via refresh_load_budget() -> DataParallelController owns it.
      - dp_size > 1, round-robin / other: DataParallelController never reads
        load data -> tokenizer-side owner owns it (polls on /v1/loads calls).

    The tokenizer-side owner is the ``"MultiTokenizerRouter"`` caller in
    multi-tokenizer mode, otherwise the ``"TokenizerManager"`` caller.

    中译：决定由哪个进程持有（bind）zmq PULL socket。
          zmq 模式激活时，DataParallelController / TokenizerManager /
          MultiTokenizerRouter 三者中必须且只能有一个返回 True；owner 负责
          轮询 zmq 并写入 SHM，其余进程只读 SHM。规则：
          - node_rank 非 0：该节点没有 TokenizerManager，DataParallelController
            只负责拉起 Scheduler 并等待，因此无人持有 -> 返回 False。
          - dp_size == 1：没有 DataParallelController -> 由 tokenizer 侧 owner 持有。
          - dp_size > 1 且为「负载感知」调度方法：DataParallelController 在每次分发时
            都会通过 refresh_load_budget() 轮询 -> 由它持有。
          - dp_size > 1 且为轮询/其他方法：DataParallelController 从不读负载数据
            -> 由 tokenizer 侧 owner 持有（在 /v1/loads 调用时轮询）。
          其中 tokenizer 侧 owner 在多 tokenizer 模式下是 MultiTokenizerRouter，
          否则是 TokenizerManager。
    """
    if not should_use_zmq(server_args):
        return False
    if server_args.node_rank != 0:
        return False
    tokenizer_owner = _tokenizer_load_snapshot_owner_caller(server_args)
    if server_args.dp_size == 1:
        return caller == tokenizer_owner
    if server_args.load_balance_method.lower() in _LOAD_AWARE_METHODS:
        return caller == "DataParallelController"
    return caller == tokenizer_owner


# ---------------------------------------------------------------------------
# LoadSnapshot data class
# ---------------------------------------------------------------------------

# 中译：核心指标字段名清单。这些是「扁平」字段，直接从 GetLoadsReqOutput 同名属性拷贝。
CORE_METRIC_FIELDS = (
    "timestamp",
    "dp_rank",
    "num_running_reqs",
    "num_waiting_reqs",
    "num_waiting_uncached_tokens",
    "num_used_tokens",
    "num_total_tokens",
    "max_total_num_tokens",
    "max_running_requests",
    "token_usage",
    "gen_throughput",
    "cache_hit_rate",
    "utilization",
)
# 中译：可选「分节（section）」字段表。每个元组为
#       (include 键, GetLoadsReqOutput 上的子对象属性名, 快照里的 has_xxx 存在标志,
#        ((子对象属性名, 快照扁平字段名), ...))。
#       由于 LoadSnapshot 是扁平结构，这里把嵌套子对象（memory/spec/lora/disagg/queues）
#       拍平成带前缀的字段；has_xxx 标志位记录该分节当时是否存在。
SECTION_FIELDS = (
    (
        "memory",
        "memory",
        "has_memory",
        (
            ("weight_gb", "memory_weight_gb"),
            ("kv_cache_gb", "memory_kv_cache_gb"),
            ("graph_gb", "memory_graph_gb"),
            ("token_capacity", "memory_token_capacity"),
        ),
    ),
    (
        "spec",
        "speculative",
        "has_speculative",
        (
            ("accept_length", "speculative_accept_length"),
            ("accept_rate", "speculative_accept_rate"),
        ),
    ),
    (
        "lora",
        "lora",
        "has_lora",
        (
            ("slots_used", "lora_slots_used"),
            ("slots_total", "lora_slots_total"),
            ("utilization", "lora_utilization"),
        ),
    ),
    (
        "disagg",
        "disaggregation",
        "has_disaggregation",
        (
            ("mode", "disagg_mode"),
            ("prefill_bootstrap_queue_reqs", "prefill_bootstrap_queue_reqs"),
            ("prefill_inflight_queue_reqs", "prefill_inflight_queue_reqs"),
            ("decode_prealloc_queue_reqs", "decode_prealloc_queue_reqs"),
            ("decode_transfer_queue_reqs", "decode_transfer_queue_reqs"),
            ("decode_retracted_queue_reqs", "decode_retracted_queue_reqs"),
            ("kv_transfer_speed_gb_s", "kv_transfer_speed_gb_s"),
            ("kv_transfer_latency_ms", "kv_transfer_latency_ms"),
        ),
    ),
    (
        "queues",
        "queues",
        "has_queues",
        (
            ("waiting", "queue_waiting"),
            ("grammar", "queue_grammar"),
            ("paused", "queue_paused"),
            ("retracted", "queue_retracted"),
        ),
    ),
)


class LoadSnapshot(msgspec.Struct, omit_defaults=True):
    """单个 dp_rank 的负载快照（可序列化的扁平结构）。

    中译：用 msgspec.Struct 定义，omit_defaults=True 表示编码时省略默认值字段以减小体积。
          字段分为「核心指标」与若干可选分节（memory/speculative/lora/disaggregation/queues），
          每个分节用 has_xxx 标志位表示当时是否采集到。该对象会被 msgpack 编码后写入
          SHM 槽位或经 zmq 传输。
    """

    timestamp: float = 0.0
    dp_rank: int = 0
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    num_waiting_uncached_tokens: int = 0
    num_used_tokens: int = 0
    num_total_tokens: int = 0
    max_total_num_tokens: int = 0
    max_running_requests: int = 0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    cache_hit_rate: float = 0.0
    utilization: float = 0.0

    has_memory: int = 0
    memory_weight_gb: float = 0.0
    memory_kv_cache_gb: float = 0.0
    memory_graph_gb: float = 0.0
    memory_token_capacity: int = 0

    has_speculative: int = 0
    speculative_accept_length: float = 0.0
    speculative_accept_rate: float = 0.0

    has_lora: int = 0
    lora_slots_used: int = 0
    lora_slots_total: int = 0
    lora_utilization: float = 0.0

    has_disaggregation: int = 0
    disagg_mode: int = 0
    prefill_bootstrap_queue_reqs: int = 0
    prefill_inflight_queue_reqs: int = 0
    decode_prealloc_queue_reqs: int = 0
    decode_transfer_queue_reqs: int = 0
    decode_retracted_queue_reqs: int = 0
    kv_transfer_speed_gb_s: float = 0.0
    kv_transfer_latency_ms: float = 0.0

    has_queues: int = 0
    queue_waiting: int = 0
    queue_grammar: int = 0
    queue_paused: int = 0
    queue_retracted: int = 0

    @classmethod
    def from_get_loads_output(cls, output: GetLoadsReqOutput) -> LoadSnapshot:
        """从 Scheduler 产出的 GetLoadsReqOutput 构造一个扁平的 LoadSnapshot。

        中译：先逐个拷贝核心指标字段（dp_rank 做空值兜底，其余经 _native 归一化）；
              再遍历分节表，把每个嵌套子对象拍平成带前缀的字段，并写入 has_xxx 标志位；
              disagg_mode 字符串转为整数编码。
        参数 output：调度器汇报的原始负载对象（含可选子对象）。
        返回：填好字段的 LoadSnapshot 实例。
        """
        snapshot: dict = {}
        for name in CORE_METRIC_FIELDS:
            value = getattr(output, name)
            if name == "dp_rank":
                snapshot[name] = int(value) if value is not None else 0
            else:
                snapshot[name] = _native(value)

        for _, section_name, present_attr, attrs in SECTION_FIELDS:
            # 中译：取出子对象（如 output.memory）；不存在则 has_xxx=0 并跳过该分节。
            section = getattr(output, section_name, None)
            snapshot[present_attr] = int(section is not None)
            if section is None:
                continue
            for section_attr, snapshot_attr in attrs:
                value = getattr(section, section_attr)
                if snapshot_attr == "disagg_mode":
                    value = DISAGG_MODE_TO_INT.get(value, 0)
                else:
                    value = _native(value)
                snapshot[snapshot_attr] = value

        return cls(**snapshot)

    # 中译：to_dict 的 include 参数允许的合法分节名集合。
    VALID_SECTIONS = frozenset(
        {"core", "memory", "spec", "lora", "disagg", "queues", "all"}
    )

    def to_dict(self, include: Optional[set[str]] = None) -> dict:
        """把快照转回带嵌套结构的 dict（供 /v1/loads 等接口返回）。

        中译：核心指标始终包含；include 控制返回哪些可选分节。
              - include 为 None 或含 "all"：返回全部存在的分节。
              - include == {"core"}：只返回核心指标。
              - 其他：校验 include 是否都在 VALID_SECTIONS 内（否则报错），
                按需挑选分节。仅当 has_xxx 为真的分节才会出现在结果里。
              disagg_mode 整数会被反查回字符串。
        """
        load = {
            "dp_rank": self.dp_rank,
            "num_running_reqs": self.num_running_reqs,
            "num_waiting_reqs": self.num_waiting_reqs,
            "num_waiting_uncached_tokens": self.num_waiting_uncached_tokens,
            "num_used_tokens": self.num_used_tokens,
            "num_total_tokens": self.num_total_tokens,
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_running_requests": self.max_running_requests,
            "token_usage": self.token_usage,
            "gen_throughput": self.gen_throughput,
            "cache_hit_rate": self.cache_hit_rate,
            "utilization": self.utilization,
        }

        if include is None or "all" in include:
            include_all = True
        else:
            # 中译：include 必须是 VALID_SECTIONS 的子集，否则抛出明确的错误提示。
            if not (include <= self.VALID_SECTIONS):
                raise ValueError(
                    f"Invalid include sections: {include - self.VALID_SECTIONS}. "
                    f"Valid options: {sorted(self.VALID_SECTIONS)}"
                )
            if include == {"core"}:
                return load
            include_all = False

        for include_key, section_name, present_attr, attrs in SECTION_FIELDS:
            # 中译：该分节当时未采集到（has_xxx 为 0）则跳过。
            if not getattr(self, present_attr):
                continue
            # 中译：非「全选」模式下，未被 include 请求的分节也跳过。
            if not include_all and include_key not in include:
                continue

            # 中译：把扁平字段重新组装回嵌套子 dict。
            section = {}
            for section_attr, snapshot_attr in attrs:
                value = getattr(self, snapshot_attr)
                if snapshot_attr == "disagg_mode":
                    value = INT_TO_DISAGG_MODE.get(value, "null")
                section[section_attr] = value
            load[section_name] = section

        return load


# 中译：全局共享的 msgpack 编/解码器（编码任意对象、解码为 LoadSnapshot）。
snapshot_encoder = msgspec.msgpack.Encoder()
snapshot_decoder = msgspec.msgpack.Decoder(LoadSnapshot)


# ---------------------------------------------------------------------------
# SHM file layout utilities
# ---------------------------------------------------------------------------

# 中译：SHM 文件二进制布局相关常量。
#   MAGIC：文件魔数（标识 SGLang Load Snapshot），用于读端校验。
#   VERSION：布局版本号，不匹配则拒绝读取。
#   HEADER_STRUCT：文件头格式 = 4字节魔数 + 2字节版本 + 2字节 dp_size + 4字节 slot_size（小端）。
#   SLOT_LEN_STRUCT：每个槽位开头的 4 字节 payload 长度。
#   SLOT_SIZE：每个 dp_rank 占用的固定槽位大小（16KB）。
# 文件整体布局：[Header][slot_0][slot_1]...[slot_{dp_size-1}]，每个 dp_rank 独占一个槽位。
MAGIC = b"SLNS"
VERSION = 2
HEADER_STRUCT = struct.Struct("<4sHHI")
SLOT_LEN_STRUCT = struct.Struct("<I")
SLOT_SIZE = 16 * 1024


@contextmanager
def file_lock(fd: int, lock_type: int):
    """文件锁上下文管理器：进入时按指定类型加锁，退出时必定解锁。

    中译：lock_type 取 fcntl.LOCK_EX（写者独占）或 LOCK_SH（读者共享），
          用 flock 协调多进程对同一 SHM 文件的并发读写，避免读到撕裂的数据。
    """
    fcntl.flock(fd, lock_type)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


def shm_path_for(ipc_name: str) -> str:
    """由 IPC 名称推导出确定性的 /dev/shm SHM 文件路径。

    中译：取 ipc_name 的 basename 做可读前缀（非字母数字字符替换为下划线），
          再附加其 blake2s 摘要的十六进制，避免不同 ipc_name 路径冲突，
          同时保证同一 ipc_name 始终映射到同一文件。
    """
    name = os.path.basename(ipc_name.rstrip("/")) or "default"
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    digest = hashlib.blake2s(ipc_name.encode(), digest_size=4).hexdigest()
    return f"/dev/shm/sglang_loads_{safe_name}_{digest}.shm"


def file_size(dp_size: int, slot_size: int = SLOT_SIZE) -> int:
    # 中译：整个 SHM 文件大小 = 文件头 + dp_size 个槽位。
    return HEADER_STRUCT.size + dp_size * slot_size


def slot_offset(dp_rank: int, slot_size: int = SLOT_SIZE) -> int:
    # 中译：第 dp_rank 个槽位在文件中的字节偏移（跳过文件头后按槽位定位）。
    return HEADER_STRUCT.size + dp_rank * slot_size


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


class ShmLoadSnapshotWriter:
    """SHM 写者：把本 dp_rank 的负载快照写入 /dev/shm mmap 文件的对应槽位。

    中译：单节点默认传输后端。构造时创建/打开文件、加写锁、写入文件头并初始化本
          rank 的槽位。每个 Scheduler 持有一个对应自己 dp_rank 的 writer。
          publish_interval/publish_counter 供调用方做发布节流（本类只存储不强制）。
    """

    def __init__(
        self, path: str, dp_size: int, dp_rank: int, publish_interval: int = 1
    ):
        """打开/创建 SHM 文件、写入文件头并初始化本 rank 槽位。

        中译：校验 dp_rank 合法性；以读写方式打开文件并加独占锁，
              ftruncate 到所需大小后 mmap 映射，写入文件头，再写入一份空快照占位。
              出错时确保关闭已打开的 fd 后再抛出。
        """
        if dp_rank < 0 or dp_rank >= dp_size:
            raise ValueError(f"invalid dp_rank={dp_rank} for dp_size={dp_size}")
        self.publish_interval = max(1, publish_interval)
        self.publish_counter = 0

        self.path = path
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.slot_size = SLOT_SIZE
        self.fd = -1
        size = file_size(dp_size, self.slot_size)

        # 中译：O_CREAT 不存在则创建；0o600 仅属主可读写。
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            with file_lock(self.fd, fcntl.LOCK_EX):
                os.ftruncate(self.fd, size)
                self.mmap = mmap.mmap(self.fd, size, access=mmap.ACCESS_WRITE)
                HEADER_STRUCT.pack_into(
                    self.mmap, 0, MAGIC, VERSION, dp_size, self.slot_size
                )
                self._write_payload(LoadSnapshot(dp_rank=dp_rank))
        except Exception:
            if self.fd >= 0:
                os.close(self.fd)
            raise

    def write(self, snapshot: LoadSnapshot) -> None:
        """加写锁后把一份快照写入本 rank 槽位。

        中译：校验快照的 dp_rank 与本 writer 一致（防止串槽），随后独占写入。
        """
        if snapshot.dp_rank != self.dp_rank:
            raise ValueError(
                f"snapshot dp_rank={snapshot.dp_rank} does not match writer dp_rank={self.dp_rank}"
            )

        with file_lock(self.fd, fcntl.LOCK_EX):
            self._write_payload(snapshot)

    def _write_payload(self, snapshot: LoadSnapshot) -> None:
        """把快照编码并写入槽位（调用方需已持有写锁）。

        中译：写入采用「先清零长度 -> 写 payload -> 清空槽位剩余空间 -> 最后写回真实长度」
              的顺序。先把长度字段置 0 再最后写真实长度，是为了让并发读者要么读到旧的完整
              数据、要么读到长度 0（视为无数据），而不会读到半截的新数据（撕裂读）。
              若 payload 超过槽位容量则报错。
        """
        payload = snapshot_encoder.encode(snapshot)
        max_payload_size = self.slot_size - SLOT_LEN_STRUCT.size
        if len(payload) > max_payload_size:
            raise ValueError(
                f"load snapshot payload size {len(payload)} exceeds slot payload "
                f"capacity {max_payload_size}"
            )

        offset = slot_offset(self.dp_rank, self.slot_size)
        payload_start = offset + SLOT_LEN_STRUCT.size
        payload_end = payload_start + len(payload)
        slot_end = offset + self.slot_size

        SLOT_LEN_STRUCT.pack_into(self.mmap, offset, 0)  # 中译：先把长度置 0，使读者跳过
        self.mmap[payload_start:payload_end] = payload  # 中译：写入新 payload
        self.mmap[payload_end:slot_end] = b"\0" * (slot_end - payload_end)  # 中译：清空残留
        SLOT_LEN_STRUCT.pack_into(self.mmap, offset, len(payload))  # 中译：最后写回真实长度

    def close(self) -> None:
        """释放 mmap 映射并关闭文件描述符。"""
        self.mmap.close()
        os.close(self.fd)


class ZmqLoadSnapshotWriter:
    """Sends load snapshots via zmq PUSH to a ZmqShmLoadSnapshotReader.

    CONFLATE is set so only the latest message is kept in the send
    buffer when the reader is slower than the writer.

    中译：ZMQ 写者：通过 zmq PUSH 把快照发送给（0 号节点上的）ZmqShmLoadSnapshotReader。
          多节点场景下替代 SHM 写者。设置了 CONFLATE 选项，当读者慢于写者时发送缓冲区
          只保留最新一条消息（丢弃旧的），保证拿到的是最新负载。
    """

    def __init__(
        self, endpoint: str, dp_size: int, dp_rank: int, publish_interval: int = 1
    ):
        """创建 PUSH socket 并连接到收集端 endpoint。

        中译：校验 dp_rank；按 endpoint 是否 IPv6 设置 IPV6 选项；
              LINGER=0 表示关闭时不等待未发完数据；CONFLATE=1 只保留最新消息。
        """
        import zmq as _zmq

        if dp_rank < 0 or dp_rank >= dp_size:
            raise ValueError(f"invalid dp_rank={dp_rank} for dp_size={dp_size}")
        self.publish_interval = max(1, publish_interval)
        self.publish_counter = 0
        self.dp_size = dp_size
        self.dp_rank = dp_rank

        self._zmq = _zmq
        self._ctx = _zmq.Context.instance()
        self._socket = self._ctx.socket(_zmq.PUSH)
        if is_zmq_endpoint_ipv6(endpoint):
            self._socket.setsockopt(_zmq.IPV6, 1)
        self._socket.setsockopt(_zmq.LINGER, 0)
        self._socket.setsockopt(_zmq.CONFLATE, 1)
        self._socket.connect(endpoint)

    def write(self, snapshot: LoadSnapshot) -> None:
        """非阻塞地 PUSH 一份快照。

        中译：校验 dp_rank 一致后用 NOBLOCK 发送；若发送缓冲区暂时不可用
              （抛出 zmq.Again）则直接丢弃本次快照（下一次还会再发，不阻塞调度）。
        """
        if snapshot.dp_rank != self.dp_rank:
            raise ValueError(
                f"snapshot dp_rank={snapshot.dp_rank} does not match "
                f"writer dp_rank={self.dp_rank}"
            )
        try:
            self._socket.send(snapshot_encoder.encode(snapshot), self._zmq.NOBLOCK)
        except self._zmq.Again:
            pass

    def close(self) -> None:
        """关闭 PUSH socket。"""
        self._socket.close()


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


class ShmLoadSnapshotReader:
    """SHM 读者：从 /dev/shm mmap 文件读取各 dp_rank 的负载快照。

    中译：供 TokenizerManager（/v1/loads）和 DataParallelController（调度分发）使用。
          采用「懒附加（lazy attach）」：文件可能尚未被写者创建，首次读取失败后下次会重试，
          一旦成功映射就缓存 mmap/fd。读取时加共享锁，与写者的独占锁配合避免撕裂读。
    """

    def __init__(self, path: str, dp_size: int):
        """记录路径与 dp_size，并尝试首次附加到 SHM 文件（允许失败）。"""
        self.path = path
        self.dp_size = dp_size
        self.mmap: Optional[mmap.mmap] = None
        self.fd: Optional[int] = None
        self.slot_size = SLOT_SIZE
        self._header_warning_logged = False
        self._attach()

    def _attach(self) -> bool:
        """尝试打开并 mmap 映射 SHM 文件，成功后缓存句柄。

        中译：已附加则直接返回 True。否则只读打开文件（不存在返回 False）；
              校验文件大小、文件头（魔数/版本/dp_size/slot_size）是否匹配，
              任一不符则关闭并返回 False（头不匹配只告警一次，避免刷屏）。
        返回：是否成功附加。
        """
        if self.mmap is not None:
            return True

        try:
            fd = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return False

        size = os.fstat(fd).st_size
        if size < HEADER_STRUCT.size:
            os.close(fd)
            return False

        try:
            with file_lock(fd, fcntl.LOCK_SH):
                mapped = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
                magic, version, dp_size, slot_size = HEADER_STRUCT.unpack_from(
                    mapped, 0
                )
        except (OSError, ValueError):
            os.close(fd)
            return False

        # 中译：文件头任一项不匹配（魔数/版本/dp_size 错，或槽位/文件过小）都视为无效。
        if (
            magic != MAGIC
            or version != VERSION
            or dp_size != self.dp_size
            or slot_size < SLOT_LEN_STRUCT.size
            or size < file_size(self.dp_size, slot_size)
        ):
            mapped.close()
            os.close(fd)
            if not self._header_warning_logged:
                logger.warning("load shm header mismatch at %s", self.path)
                self._header_warning_logged = True
            return False

        self.mmap = mapped
        self.fd = fd
        self.slot_size = slot_size
        return True

    def read(self, dp_rank: int) -> Optional[LoadSnapshot]:
        """读取指定 dp_rank 的最新快照（加共享锁）。

        中译：dp_rank 越界或文件尚未就绪时返回 None；否则加共享锁读取该槽位。
        """
        if dp_rank < 0 or dp_rank >= self.dp_size:
            return None
        if not self._attach():
            return None

        assert self.fd is not None
        with file_lock(self.fd, fcntl.LOCK_SH):
            return self._read_slot(dp_rank)

    def _read_slot(self, dp_rank: int) -> Optional[LoadSnapshot]:
        """解析单个槽位的字节为 LoadSnapshot（调用方需已持锁）。

        中译：先读长度字段；长度为 0 或超界视为无效返回 None；
              否则切出 payload 字节并 msgpack 解码。解码异常时记调试日志并返回 None
              （容忍写者正在更新的瞬态情况）。
        """
        assert self.mmap is not None
        offset = slot_offset(dp_rank, self.slot_size)
        (payload_len,) = SLOT_LEN_STRUCT.unpack_from(self.mmap, offset)
        max_payload_size = self.slot_size - SLOT_LEN_STRUCT.size
        if payload_len == 0 or payload_len > max_payload_size:
            return None

        payload_start = offset + SLOT_LEN_STRUCT.size
        payload_end = payload_start + payload_len
        try:
            return snapshot_decoder.decode(self.mmap[payload_start:payload_end])
        except Exception as e:
            logger.debug("load snapshot decode failed for rank %s: %s", dp_rank, e)
            return None

    def read_all(self) -> list[LoadSnapshot]:
        """一次性读取所有 dp_rank 的有效快照（一把共享锁覆盖全部槽位）。

        中译：文件未就绪返回空列表；否则遍历所有槽位，跳过无效（None）的，返回有效列表。
        """
        if not self._attach():
            return []

        assert self.fd is not None
        with file_lock(self.fd, fcntl.LOCK_SH):
            loads = []
            for r in range(self.dp_size):
                load = self._read_slot(r)
                if load is not None:
                    loads.append(load)
            return loads

    def close(self) -> None:
        """释放 mmap 与 fd（幂等，可重复调用）。"""
        if self.mmap is not None:
            self.mmap.close()
            self.mmap = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class ZmqShmLoadSnapshotReader:
    """Receives snapshots via zmq PULL from writers, writes to SHM, reads from SHM.

    Transparently wraps a ShmLoadSnapshotReader.  Every read() / read_all()
    first drains the PULL socket into SHM so callers always see fresh data.

    中译：0 号节点上的「桥接读者」：通过 zmq PULL 收取各节点写者发来的快照，
          写入本地 SHM，再从 SHM 读取。它透明地包装了一个 ShmLoadSnapshotReader，
          每次 read()/read_all() 都会先把 PULL socket 里的消息排空并落入 SHM，
          从而保证调用方读到的是最新数据。它同时为同节点其他纯 SHM 读者维护 SHM 文件。
    """

    def __init__(self, endpoint: str, shm_path: str, dp_size: int):
        """绑定 PULL socket 并创建内部 SHM 读者与按 rank 的 SHM 写者缓存。

        中译：bind（而非 connect）zmq PULL 端点（owner 角色）；同样设置 IPV6/LINGER/CONFLATE；
              内部持有一个 ShmLoadSnapshotReader 负责读，_shm_writers 按 dp_rank 懒创建写者。
        """
        import zmq as _zmq

        self._zmq = _zmq
        self._ctx = _zmq.Context.instance()
        self._socket = self._ctx.socket(_zmq.PULL)
        if is_zmq_endpoint_ipv6(endpoint):
            self._socket.setsockopt(_zmq.IPV6, 1)
        self._socket.setsockopt(_zmq.LINGER, 0)
        self._socket.setsockopt(_zmq.CONFLATE, 1)
        self._socket.bind(endpoint)

        self._endpoint = endpoint
        self._shm_path = shm_path
        self.dp_size = dp_size
        self._shm_reader = ShmLoadSnapshotReader(shm_path, dp_size)
        self._shm_writers: dict[int, ShmLoadSnapshotWriter] = {}

    def _poll(self) -> None:
        """Drain zmq messages and write latest per dp_rank to SHM.

        中译：排空 zmq PULL 队列，并把每个 dp_rank 的「最新一条」写入 SHM。
              先非阻塞循环 recv 直到 zmq.Again（队列空），过程中按 dp_rank 只保留最新快照
              （后到覆盖先到），解码失败仅告警；随后为每个 rank 懒创建写者并落盘，
              单个 rank 写失败也只告警、不影响其他 rank。
        """
        latest: dict[int, LoadSnapshot] = {}
        while True:
            try:
                data = self._socket.recv(self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            try:
                snapshot = snapshot_decoder.decode(data)
                if 0 <= snapshot.dp_rank < self.dp_size:
                    latest[snapshot.dp_rank] = snapshot
            except Exception as e:
                logger.warning("load snapshot zmq decode failed: %s", e)

        for dp_rank, snapshot in latest.items():
            # 中译：首次见到某 dp_rank 时再为其创建 SHM 写者（懒初始化）。
            if dp_rank not in self._shm_writers:
                self._shm_writers[dp_rank] = ShmLoadSnapshotWriter(
                    self._shm_path, self.dp_size, dp_rank
                )
            try:
                self._shm_writers[dp_rank].write(snapshot)
            except Exception as e:
                logger.warning(
                    "load snapshot shm write failed for rank %d: %s", dp_rank, e
                )

    def fileno(self) -> int:
        """Edge-triggered fd that becomes readable when zmq messages arrive.

        Lets an owner process register the reader with an event loop and drain
        it via ``poll()`` instead of polling on a timer.

        中译：返回 zmq socket 的边沿触发 fd，消息到达时变为可读。
              owner 进程可把它注册进事件循环，由事件驱动调用 poll() 排空，
              而不必用定时器轮询。
        """
        return self._socket.getsockopt(self._zmq.FD)

    def poll(self) -> None:
        """Drain the zmq PULL socket into SHM.

        Public entry point so an owner process (e.g. MultiTokenizerRouter) can
        keep SHM fresh without touching internals.

        中译：_poll 的公开入口，便于 owner 进程（如 MultiTokenizerRouter）在不触碰
              内部实现的前提下，把 zmq 数据排空到 SHM，保持 SHM 新鲜。
        """
        self._poll()

    def read(self, dp_rank: int) -> Optional[LoadSnapshot]:
        """先排空 zmq 到 SHM，再从 SHM 读取指定 rank 的快照。"""
        self._poll()
        return self._shm_reader.read(dp_rank)

    def read_all(self) -> list[LoadSnapshot]:
        """先排空 zmq 到 SHM，再从 SHM 读取全部 rank 的快照。"""
        self._poll()
        return self._shm_reader.read_all()

    def close(self) -> None:
        """关闭全部资源：所有 SHM 写者、内部读者、PULL socket。

        中译：若端点是 ipc:// 形式，还会尝试删除对应的 unix socket 文件（清理临时文件）。
        """
        for w in self._shm_writers.values():
            w.close()
        self._shm_writers.clear()
        self._shm_reader.close()
        self._socket.close()
        if self._endpoint.startswith("ipc://"):
            try:
                os.unlink(self._endpoint[len("ipc://") :])
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _zmq_addr_for(port_args) -> str:
    """Return the zmq PUSH/PULL address from PortArgs.

    For dp_attention (TCP mode), uses the ``load_collector_ipc_name`` field
    stored in PortArgs.  For single-node IPC (env-var override), derives
    a deterministic IPC path from ``instance_id``.

    中译：从 PortArgs 推导 zmq PUSH/PULL 地址。
          dp_attention（TCP 模式）下直接用 PortArgs 里的 load_collector_ipc_name；
          若该字段为空（如单节点经环境变量强制开启 zmq），则基于 instance_id
          生成一个确定性的 ipc:// unix socket 路径（含可读前缀 + blake2s 摘要避免冲突）。
    """
    ipc_name = getattr(port_args, "load_collector_ipc_name", "")
    if ipc_name:
        return ipc_name
    safe = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in port_args.instance_id
    )
    digest = hashlib.blake2s(port_args.instance_id.encode(), digest_size=4).hexdigest()
    return f"ipc:///tmp/sglang_load_collector_{safe}_{digest}.sock"


def create_load_snapshot_writer(
    server_args,
    port_args,
    dp_size: int,
    dp_rank: int,
    publish_interval: int = 1,
):
    """Return a SHM or ZMQ writer based on server configuration.

    中译：工厂函数——根据服务配置返回 SHM 写者或 ZMQ 写者。
          should_use_zmq 为真时用 ZmqLoadSnapshotWriter（地址来自 _zmq_addr_for），
          否则用 ShmLoadSnapshotWriter（路径来自 shm_path_for(instance_id)）。
    """
    if should_use_zmq(server_args):
        return ZmqLoadSnapshotWriter(
            _zmq_addr_for(port_args), dp_size, dp_rank, publish_interval
        )
    return ShmLoadSnapshotWriter(
        shm_path_for(port_args.instance_id), dp_size, dp_rank, publish_interval
    )


def create_load_snapshot_reader(server_args, port_args, caller: str):
    """Create a load snapshot reader.

    Args:
        caller: ``"DataParallelController"``, ``"TokenizerManager"``, or
            ``"MultiTokenizerRouter"`` -- determines who binds the zmq PULL
            socket when zmq mode is active.

    中译：工厂函数——创建负载快照读者。
          根据 zmq_reader_owner(server_args, caller) 判断本调用方是否为 zmq owner：
          是则返回会 bind PULL 端点的 ZmqShmLoadSnapshotReader（桥接 zmq->SHM），
          否则返回纯 ShmLoadSnapshotReader（仅读 SHM）。
          caller 取值见上，用于决定 zmq 模式下由谁来 bind PULL socket。
    """
    dp_size = server_args.dp_size
    if zmq_reader_owner(server_args, caller):
        return ZmqShmLoadSnapshotReader(
            _zmq_addr_for(port_args), shm_path_for(port_args.instance_id), dp_size
        )
    return ShmLoadSnapshotReader(shm_path_for(port_args.instance_id), dp_size)
