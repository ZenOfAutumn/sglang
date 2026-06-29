from __future__ import annotations

import atexit
import heapq
import json
import logging
import os
import threading
import time
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PrefetchTimeoutConfig,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    attach_hybrid_dsa_pool_to_hiradix_cache,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MLATokenToKVPoolHost,
    get_mha_host_pool_cls,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
)
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class HiRadixCache(RadixCache):
    """分级（Hierarchical）前缀缓存：在 RadixCache 之上叠加多级存储。

    背景：纯 RadixCache 只在 GPU device 显存里保存 KV，显存一满就得淘汰、再用时只能重算。
    HiRadixCache 在其下方挂接更廉价、更大容量的存储层，形成「分级缓存」：

        L1：device（GPU 显存）—— 由父类 RadixCache 的 value（device 槽位索引）管理，最快但最贵。
        L2：host（CPU 内存）—— 由 token_to_kv_pool_host 管理，作为显存的备份/溢出层。
        L3：storage backend（磁盘/远端 KV 存储，可选）—— 容量最大、最慢，跨请求/跨进程复用。

    数据在各级之间通过 HiCacheController 异步搬运：
        * write-through / backup：把 device 上的 KV 写穿透到 host，再可选地备份到 L3 storage。
        * load-back：把 host 上的 KV 回载到 device，使其重新可用于计算。
        * prefetch：从 L3 storage 预取命中的前缀到 host，再按需 load-back 到 device。

    通过让热点前缀「下沉」到 host/storage 而非直接丢弃，HiCache 显著提升前缀命中率、
    降低重算开销，尤其适合长上下文、多轮对话、RAG 等场景。本类继承 RadixCache 的树结构与
    淘汰框架，并重写相关方法以协调 device/host/storage 三级之间的搬运与一致性。
    """

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._enable_metrics_flag = params.enable_metrics

        self.page_size = params.page_size
        # 取出底层 device 侧的 KV pool（按模型注意力类型分为 MHA / MLA / DSA）。
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()

        # 依据 device KV pool 的类型，构造与之匹配的 host（CPU）侧备份池。
        # host 池容量由 hicache_ratio / hicache_size 控制，是 L2 缓存的载体。
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = get_mha_host_pool_cls(self.kv_cache)(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, DSATokenToKVPool):
            # DSA（混合注意力）场景：host 池要等 storage 的 extra_config 解析完，
            # 由 attach_hybrid_dsa_pool_to_hiradix_cache 在稍后填充，这里先置空。
            self.token_to_kv_pool_host = None
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        else:
            raise ValueError("HiRadixCache only supports MHA, MLA, and DSA models")

        # 各类分布式通信组：HiCache 在多卡/多级并行下需要跨 rank 同步缓存元数据。
        self.tp_group = params.tp_cache_group  # 张量并行（TP）通信组
        self.attn_cp_group = params.attn_cp_cache_group  # 注意力上下文并行（CP）组
        self.attn_tp_group = params.attn_tp_cache_group  # 注意力张量并行子组
        self.pp_group = params.pp_cache_group  # 流水线并行（PP）通信组
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.pp_rank = params.pp_rank  # 当前进程在 PP 流水线中的 rank
        self.pp_size = params.pp_size  # PP 流水线总级数
        # 是否启用 L3 storage 后端（None 表示只用 device+host 两级）。
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_config,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        # TODO: 支持更多的超时检查函数（目前只实现了线性超时）。
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        # 预取停止策略：best_effort（尽力而为）/ wait_complete（等待完成）/ timeout（超时）。
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        # 调度线程与缓存搬运线程之间的同步事件：有 load-back 完成时被唤醒。
        self.load_cache_event = threading.Event()
        # cache_controller 是 device/host/storage 三级之间所有异步搬运的执行者。
        # DSA 走专门的混合控制器组装路径；其余模型用通用的 HiCacheController。
        if isinstance(self.kv_cache, DSATokenToKVPool):
            attach_hybrid_dsa_pool_to_hiradix_cache(
                self,
                params,
                server_args,
                extra_config=extra_config,
                prefetch_threshold=prefetch_threshold,
                enable_storage_metrics=self.enable_storage_metrics,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
            )
        else:
            self.cache_controller = HiCacheController(
                params.token_to_kv_pool_allocator,
                self.token_to_kv_pool_host,
                self.page_size,
                self.tp_group,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
                pp_group=self.pp_group,
                write_policy=server_args.hicache_write_policy,
                io_backend=server_args.hicache_io_backend,
                storage_backend=server_args.hicache_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=extra_config,
                enable_storage_metrics=self.enable_storage_metrics,
            )
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_config=prefetch_timeout_config,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        # 记录正在进行 write-through（device→host 写穿透）的节点。
        self.ongoing_write_through = {}
        # 记录正在进行 load-back（host→device 回载）的节点片段。
        self.ongoing_load_back = {}
        # 记录正在进行的 prefetch（storage→host 预取）请求。
        self.ongoing_prefetch = {}
        # 记录正在进行的 backup（host→storage 备份）操作。
        self.ongoing_backup = {}
        # 按请求统计从 L3 storage 实际载入的 token 数（用于 L3 命中率指标）。
        # key: request_id，value: 实际从 storage 载入的 token 数。
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        # 待回收的异步分布式通信句柄列表（如 PP 间的 isend）。
        self.work_list: List[torch.distributed.Work] = []
        # 写穿透阈值：一个节点被命中达到该次数后才触发 write-through。
        # write_through 策略下为 1（首次即写），否则为 2（更保守，减少写放大）。
        # todo: 动态调整该阈值。
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        # 回载阈值：host 命中的前缀长度达到该值才值得做一次 load-back。
        self.load_back_threshold = 10

        # 进程退出时自动 detach storage 后端，避免残留连接/线程。
        atexit.register(self.shutdown)

        # host 侧的可淘汰叶子集合（与父类 device 侧 evictable_leaves 对应）。
        self.evictable_host_leaves = set()

        super().__init__(params=params)

    def _all_reduce_attn_groups(self, tensor: torch.Tensor, op):
        # 在注意力相关的并行组内做 all_reduce，让各 rank 对缓存元数据（如命中长度）达成一致。
        # 优先在 attn_cp / attn_tp 子组上规约；若它们都不存在或只有单 rank，则回退到整个 TP 组。
        reduced = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(tensor, op=op, group=group)
                reduced = True
        if not reduced and self.tp_world_size > 1:
            torch.distributed.all_reduce(tensor, op=op, group=self.tp_group)

    def _barrier_attn_groups(self):
        # 与 _all_reduce_attn_groups 同样的「优先子组、否则回退 TP 组」逻辑，但只做同步屏障，
        # 用于在跨 rank 的搬运操作前后对齐执行节奏。
        waited = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.barrier(group=group)
                waited = True
        if not waited and self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)

    def _reap_completed_async_work(self):
        """轮询并回收已完成的异步通信句柄。

        work_list 按入队顺序保存异步通信句柄（如 PP 间的 isend）。由于通信按序完成，
        只需从头连续回收已完成的那些即可。

        必须在调度（scheduler）线程中调用。
        """
        count = 0
        while count < len(self.work_list) and self.work_list[count].is_completed():
            count += 1
        if count > 0:
            logger.debug(f"Reap {count} completed async work")
            self.work_list = self.work_list[count:]

    def _all_reduce(self, data: torch.Tensor, tp_reduce_op: torch.distributed.ReduceOp):
        """在所有 TP 与 PP rank 之间同步数据。

        具体做法：先在「第 0 个 PP 级」的所有 TP rank 上执行 tp_reduce_op 规约，
        再把规约结果沿流水线传播给后续所有 PP 级（PP1、PP2…）。这样既保证 TP 内一致，
        又让整条 PP 流水线看到同一份结果。

        必须在调度（scheduler）线程中调用。
        """
        if self.pp_rank == 0:
            self._all_reduce_attn_groups(data, tp_reduce_op)
        self._pp_sync(data)

    def _pp_sync(self, data: torch.Tensor) -> None:
        """沿 PP 流水线同步数据：PPn（n>0）会接收来自 PP0 的数据。

        下图说明 _pp_sync 的行为（data 从 PP0 逐级向后传递）。

        time  | pp0                     | pp1                     | pp2
        ------|-------------------------|-------------------------|-----------------------------
        0     | _pp_sync(data=1) starts | _pp_sync(data=?) starts | _pp_sync(data=?) starts
        1     | _pp_sync(data=1) ends   |                         |
        2     |                         | _pp_sync(data=1) ends   |
        3     |                         |                         | _pp_sync(data=1) ends

        _pp_sync 不要求各 rank 之间存在统一的同步点（无需全局对齐），下面这种交错执行也可能发生。

        time  | pp0                     | pp1                     | pp2
        ------|-------------------------|-------------------------|-----------------------------
        0     | _pp_sync(data=1) starts |                         |
        1     | _pp_sync(data=1) ends   |                         |
        2     |                         | _pp_sync(data=?) starts |
        3     |                         | _pp_sync(data=1) ends   |
        4     |                         |                         | _pp_sync(data=?) starts
        5     |                         |                         | _pp_sync(data=1) ends
        """
        if self.pp_size <= 1 or self.pp_group is None:
            return
        # 非首级：先从上一级 PP 阻塞接收数据。
        if self.pp_rank > 0:
            torch.distributed.recv(
                data, group_src=self.pp_rank - 1, group=self.pp_group, tag=2
            )
        # 非末级：把数据异步转发给下一级 PP。
        if self.pp_rank + 1 < self.pp_size:
            # 先克隆一份 data 再发送，使调用方在本次调用后可安全修改 data。
            # 由于 _pp_sync 仅用于传输小数据，这份拷贝开销可忽略。
            copy_of_data = data.clone()
            send_work = torch.distributed.isend(
                copy_of_data, group_dst=self.pp_rank + 1, group=self.pp_group, tag=2
            )
            self.work_list.append(send_work)

    def shutdown(self):
        """进程退出时尽力（best-effort）自动 detach storage 后端。

        这样可保持启动期与运行期行为一致：只要后端被挂接过（无论是通过 CLI 参数还是
        管理 API），退出时都尝试 detach。失败也只记录日志，不影响进程退出。
        """
        try:
            if self.enable_storage:
                self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_config: PrefetchTimeoutConfig,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        # 把解析好的 storage 运行期配置落到实例字段，并按需初始化指标采集器。
        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_config = prefetch_timeout_config
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics

        if self.enable_storage_metrics:
            attn_cp_rank, attn_cp_size = (
                self.cache_controller.get_attn_cp_rank_and_size()
            )
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
                "attn_cp_rank": attn_cp_rank,
                "attn_cp_size": attn_cp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            existing_collector = getattr(self, "storage_metrics_collector", None)
            if existing_collector is None:
                # 首次：构造指标采集器。
                from sglang.srt.server_args import get_global_server_args

                storage_cls = resolve_collector_class(
                    get_global_server_args(),
                    STAT_LOGGER_ROLE_STORAGE,
                    StorageMetricsCollector,
                )
                self.storage_metrics_collector = storage_cls(labels=labels)
            elif set(existing_collector.labels.keys()) == set(labels.keys()):
                # 标签键集合不变：只更新标签值，复用同一采集器。
                existing_collector.labels = labels
            else:
                # 标签键集合变了：保留旧标签，避免重复注册同名指标导致冲突。
                logger.warning(
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to "
                    "avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """运行期挂接（启用）storage 后端。

        这会在 `HiCacheController` 内部启动 storage 线程，并启用 prefetch/backup 路径。
        调用方必须确保此时没有正在运行/排队的请求，以避免竞态。

        返回 (是否成功, 提示信息)。
        """
        # 先校验入参（不产生副作用）。
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: {hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        # 若已经启用：
        # - 后端未变：视为成功，仅更新策略。
        # - 后端改变：视为失败，且不更新任何策略（必须先 detach 再换后端）。
        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type

            if current_backend == storage_backend:
                if hicache_storage_prefetch_policy is not None:
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                    logger.info(
                        f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
                    )
                if hicache_write_policy is not None:
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                    logger.info(f"Set hicache_write_policy to {hicache_write_policy}")
                return (
                    True,
                    "HiCache storage backend already enabled with same backend; policies updated.",
                )

            return (
                False,
                f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                f"Cannot attach different backend '{storage_backend}'. Detach first.",
            )

        # 尚未启用：在 controller attach 之前先更新策略，使新启动的 storage 线程能读到新值。
        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
            logger.info(
                f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
            )

        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )
            logger.info(f"Set hicache_write_policy to {hicache_write_policy}")

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_config,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json '{storage_backend_extra_config_json}': {e}",
            )

        try:
            self.cache_controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
                **self._get_hybrid_storage_attach_kwargs(),
            )
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_config=prefetch_timeout_config,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple[bool, str]:
        """运行期 detach（停用）storage 后端。

        调用方必须确保此时没有正在运行/排队的请求，以避免竞态。
        返回 (是否成功, 提示信息)。
        """
        try:
            # 在拆除 storage 线程/后端之前，先排空所有待处理的控制队列。
            # 重要：必须早于清空 `ongoing_*`，否则后到的 ack/release 将无法匹配到对应节点，
            # 可能造成 host 页或锁的泄漏。
            self._drain_storage_control_queues_local()
            # 幂等 detach：始终请求 controller 做尽力清理，即使 `self.enable_storage`
            # 已经是 False（可能是上一次未完成 detach 残留的状态）。
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # 管理类操作不应让 server 崩溃。返回失败并附带详情。
            return False, f"Failed to detach HiCache storage backend: {e}"

        # 尽力清理任何残留的簿记信息。
        self._drain_storage_control_queues_local()
        # controller 线程完全停止后，可以安全地强制释放任何残留的待处理操作
        # （例如未收到 revoke/ack 的异步 prefetch/backup）。
        self._force_release_pending_storage_ops()

        self.enable_storage = False
        self.enable_storage_metrics = False
        return True, "Detached HiCache storage backend successfully."

    def _force_release_pending_storage_ops(self):
        """强制释放任何残留的 prefetch/backup 簿记信息。

        这是 detach/shutdown 路径上的安全兜底。它假定 storage 线程已经停止
        （通过 controller.detach），因此不会有并发访问这些数据结构。
        """
        cc = self.cache_controller

        # 强制释放残留的 prefetch 操作：归还预分配的 host 页，并解除匹配前缀节点上的 host 保护。
        try:
            for req_id, info in list(self.ongoing_prefetch.items()):
                try:
                    last_host_node, token_ids, host_indices, _operation = info
                except Exception:
                    # 结构不符合预期，直接丢弃该项。
                    self.ongoing_prefetch.pop(req_id, None)
                    continue

                try:
                    if host_indices is not None:
                        cc.mem_pool_host.free(host_indices)
                except Exception:
                    logger.exception(
                        "Failed to free host indices for prefetch %s", req_id
                    )

                try:
                    last_host_node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for prefetch %s", req_id
                    )

                try:
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0
                except Exception:
                    pass

                self.ongoing_prefetch.pop(req_id, None)
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")

        # 强制释放残留的 backup 操作：解除相关节点上的 host 保护。
        try:
            for ack_id, node in list(self.ongoing_backup.items()):
                try:
                    node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for backup op %s", ack_id
                    )
                self.ongoing_backup.pop(ack_id, None)
        except Exception:
            logger.exception("Force release pending backup ops failed.")

    def _drain_storage_control_queues_local(self):
        """排空 storage 控制队列，且不做 TP 同步。

        用于 shutdown/detach 路径：即便各 rank 的队列长度暂时不一致，也要尽力清理，
        因此这里不强求跨 rank 对齐队列长度。
        """
        self._drain_storage_control_queues_impl(
            n_revoke=None,
            n_backup=None,
            n_release=None,
            log_metrics=False,
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        log_metrics: bool,
    ):
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is not None:
                    last_host_node, token_ids, _, _ = info
                    last_host_node.release_host()
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0

        def _drain_backup():
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                ack_id = operation.id
                entry = self.ongoing_backup.pop(ack_id, None)
                if entry is not None:
                    entry.release_host()
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)

        _drain_revoke()
        _drain_backup()
        _drain_release()

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """解析 storage 后端的 extra config（JSON）并抽取出若干具体参数。

        Args:
            storage_backend_extra_config: 包含额外配置的 JSON 字符串（或以 "@" 前缀的文件路径）。

        Returns:
            tuple: (extra_config_dict, prefetch_threshold, prefetch_timeout_config, hicache_storage_pass_prefix_keys)
        """
        # 若提供了 extra config 则解析。它可以是 JSON 字符串，
        # 也可以是以 "@" 为前缀的 json/toml/yaml 文件路径。
        extra_config = {}
        if storage_backend_extra_config:
            try:
                if storage_backend_extra_config.startswith("@"):
                    # 从 json/toml/yaml 文件读取配置。
                    path = storage_backend_extra_config[1:]
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "rb" if ext == ".toml" else "r") as f:
                        if ext == ".json":
                            extra_config = json.load(f)
                        elif ext == ".toml":
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (config format: {ext})"
                            )
                else:
                    # 直接从 JSON 字符串读取配置。
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        # 从 extra_config 中弹出预取相关参数（弹出后剩余键继续传给后端）。
        defaults = PrefetchTimeoutConfig()
        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # 单位：token
        prefetch_timeout_base = extra_config.pop(
            "prefetch_timeout_base", defaults.base
        )  # 单位：秒，线性超时的基础时间
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", defaults.per_ki_token
        )  # 单位：秒/1024 token，超时随 token 数线性增长的斜率
        prefetch_timeout_max = extra_config.pop(
            "prefetch_timeout_max", defaults.max
        )  # 单位：秒，线性超时的上限
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(prefetch_timeout_max, (int, float)):
            raise ValueError(
                f"prefetch_timeout_max must be number, got {type(prefetch_timeout_max).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        prefetch_timeout_config = PrefetchTimeoutConfig(
            base=float(prefetch_timeout_base),
            per_ki_token=float(prefetch_timeout_per_ki_token),
            max=float(prefetch_timeout_max),
        )

        return (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_config,
            hicache_storage_pass_prefix_keys,
        )

    def reset(self):
        # 重置整个分级缓存：清空节点计数器、controller、host 池及各类簿记，最后重置父类树结构。
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        # 清空按请求维度的跟踪字典。
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.evictable_host_leaves.clear()
        super().reset()

    def get_height(self, node: TreeNode):
        # 计算从给定节点回到根节点的路径长度（树高），调试/统计用。
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def _get_extra_pools(self) -> dict:
        if not isinstance(self.cache_controller, HybridCacheController):
            return {}
        if isinstance(self.kv_cache, DSATokenToKVPool):
            pool = PoolTransfer(
                name=PoolName.INDEXER,
                hit_policy=PoolHitPolicy.ALL_PAGES,
                indices_from_pool=PoolName.KV,
            )
            return {"extra_pools": [pool]}
        else:
            return {}

    def _get_hybrid_storage_attach_kwargs(self) -> dict:
        """当 controller 为 HybridCacheController 时，attach_storage_backend 所需的额外 kwargs。"""
        if isinstance(self.cache_controller, HybridCacheController):
            return {"host_pools": self.cache_controller.mem_pool_host.entries}
        return {}

    def clear_storage_backend(self) -> bool:
        # 清空 L3 storage 后端的全部内容（仅部分后端如 nixl 支持 clear 操作）。
        if self.enable_storage:
            try:
                # 检查 storage 后端是否实现了 clear 方法。
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def write_backup(self, node: TreeNode, write_back=False) -> int:
        # 把单个节点的 KV 从 device 写穿透（write-through）到 host，返回写入的 token 数。
        # 备份不变式（write-through 模式下）：已备份的节点必须构成从根开始的连续前缀，中间不能有空洞。
        # 因此若父节点尚未备份，则跳过本节点（write_back 模式不受此约束）。
        if not write_back and (
            node.parent != self.root_node and not node.parent.backuped
        ):
            return 0

        # 请求 controller 在 host 池分配空间并发起 device→host 的异步搬运。
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
            **self._get_extra_pools(),
        )
        if host_indices is None:
            # host 池满了：先淘汰出足够空间，再重试一次写入。
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
                **self._get_extra_pools(),
            )
        if host_indices is not None:
            node.host_value = host_indices.clone()
            assert len(node.host_value) > 0
            # 登记为「写穿透进行中」，等待 DMA 完成的 ack。
            self._track_write_through_node(node, len(node.key))
            if not write_back:
                # 写穿透期间锁住该节点，防止其 device KV 在搬运完成前被淘汰。
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def _track_write_through_node(self, node: TreeNode, backup_len: int) -> None:
        # 登记一个进行中的写穿透：以节点 id 为 ack_id，记录 (锁定节点, 备份长度, 待发布节点列表)。
        node.write_through_pending_id = node.id
        self.ongoing_write_through[node.id] = (node, backup_len, [node])

    def _replace_pending_write_through_node(
        self, old_node: TreeNode, new_nodes: List[TreeNode]
    ) -> None:
        # 当某个「写穿透进行中」的节点被分裂时，用分裂出的新节点替换待发布列表中的旧节点，
        # 保证 ack 到达时能正确地把状态发布到分裂后的所有新节点上。
        ack_id = old_node.write_through_pending_id
        if ack_id is None:
            return

        pending = self.ongoing_write_through.get(ack_id)
        if pending is None:
            return

        lock_node, backup_len, publish_nodes = pending
        updated_nodes = []
        replaced = False
        for node in publish_nodes:
            if node is old_node:
                updated_nodes.extend(new_nodes)
                replaced = True
            else:
                updated_nodes.append(node)

        if not replaced:
            return

        for node in new_nodes:
            node.write_through_pending_id = ack_id
        self.ongoing_write_through[ack_id] = (lock_node, backup_len, updated_nodes)

    def _finish_write_through_ack(self, ack_id: int, *, release_lock: bool) -> None:
        # 收到写穿透完成的 ack：清除各节点的 pending 标记并上报「已落到 host」事件，
        # 若启用了 L3 则继续把该前缀备份到 storage，最后按需解锁。
        lock_node, backup_len, publish_nodes = self.ongoing_write_through.pop(ack_id)
        for node in publish_nodes:
            if node.write_through_pending_id == ack_id:
                node.write_through_pending_id = None
            # DMA 已确认——数据块此刻已在 host 上。
            self._record_store_event(node, medium=StorageMedium.CPU)
        if self.enable_storage:
            self.write_backup_storage(lock_node, backup_len)
        if release_lock:
            self.dec_lock_ref(lock_node)

    def write_backup_storage(self, node: TreeNode, backup_len: Optional[int] = None):
        # 把已在 host 上的 KV 进一步备份到 L3 storage。
        # 若节点在入队后被分裂，则通过「沿链向上遍历并拼接」恢复出分裂前的完整数据；
        # prefix_keys 锚定在链顶节点，避免前缀哈希被重复计入。
        if backup_len is None or len(node.key) == backup_len:
            top, key, hash_value, host_value = (
                node,
                node.key,
                node.hash_value,
                node.host_value,
            )
        else:
            top, key, hash_value, host_value = self._concat_split_chain(
                node, backup_len
            )

        prefix_keys = (
            top.get_prefix_hash_values(top.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )

        # 发起 host→storage 的异步备份，登记到 ongoing_backup，并保护 host 值不被淘汰直到完成。
        operation_id = self.cache_controller.write_storage(
            host_value, key, hash_value, prefix_keys, **self._get_extra_pools()
        )
        self.ongoing_backup[operation_id] = node
        node.protect_host()

    def _concat_split_chain(self, node: TreeNode, backup_len: int):
        """沿分裂链向上遍历，恢复出「入队时刻」的 key/hash/host_value。"""
        # 从 node 向根回溯，累计长度恰好等于 backup_len，收集这条链上的所有节点。
        chain, accumulated = [], 0
        current = node
        while current is not self.root_node and accumulated < backup_len:
            chain.append(current)
            accumulated += len(current.key)
            current = current.parent
        assert accumulated == backup_len, (
            f"backup chain length mismatch for node {node.id}: "
            f"expected {backup_len}, got {accumulated}"
        )
        chain.reverse()  # 反转为「父在前」的顺序，便于按前缀顺序拼接
        top = chain[0]
        if top.key.is_bigram:
            # bigram 片段之间共享一个边界 token，因此第一段之后都要丢掉首个重叠 token。
            token_ids = list(chain[0].key.token_ids)
            for n in chain[1:]:
                token_ids.extend(n.key.token_ids[1:])
        else:
            token_ids = []
            for n in chain:
                token_ids.extend(n.key.token_ids)
        key = RadixKey(token_ids, top.key.extra_key, top.key.is_bigram)

        # 仅当链上所有节点的哈希都已计算时才拼接出完整哈希，否则置 None（留待惰性计算）。
        if all(n.hash_value is not None for n in chain):
            hash_value = []
            for n in chain:
                hash_value.extend(n.hash_value)
        else:
            hash_value = None
        host_value = torch.cat([n.host_value for n in chain])
        return top, key, hash_value, host_value

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        # write_back 策略或分块请求都跳过命中计数更新：
        # write_back 模式下只在淘汰时才回写 host，无需靠命中计数触发；
        # 分块请求跳过的原因同父类（避免自我引用式膨胀）。
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1

        if not node.backuped:
            if node.hit_count >= self.write_through_threshold:
                # 节点尚未备份且命中次数达到阈值：触发 device→host 的写穿透。
                self.write_backup(node)

    def writing_check(self, write_back=False):
        # 检查/收割已完成的「写穿透」操作。write_back=True 时阻塞直到全部完成。
        if write_back:
            # 阻塞，直到所有 write-back 完成。
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        self._finish_write_through_ack(ack_id, release_lock=False)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # 注意：所有 rank 的 ongoing_write_through 完全一致，为空时可跳过跨 rank 同步。
        if len(self.ongoing_write_through) == 0:
            return

        # 非阻塞路径：统计本地已完成的写穿透数，再用 MIN 规约取各 rank 的最小值，
        # 确保所有 rank 步调一致地只处理「大家都已完成」的那部分。
        finish_count = 0
        if self.pp_rank == 0:
            for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                if not finish_event.query():
                    break
                finish_count += 1
        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
        finish_count = finish_count_tensor.item()

        if finish_count > 0:
            logger.debug(f"Process {finish_count} write back operations")
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                self._finish_write_through_ack(ack_id, release_lock=True)
            finish_count -= 1

    def loading_check(self):
        # 收割已完成的「load-back（host→device 回载）」操作，逻辑与 writing_check 的非阻塞路径对称：
        # 各 rank MIN 规约取已完成数的最小值，再逐个解锁对应的末端节点。
        finish_count = 0
        if self.pp_rank == 0:
            for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
                if not finish_event.query():
                    break
                finish_count += 1
        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
        finish_count = finish_count_tensor.item()

        if finish_count > 0:
            logger.debug(f"Process {finish_count} load operations")
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_load_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                end_node = self.ongoing_load_back.pop(ack_id)
                # 回载完成，解开回载期间对末端节点施加的锁。
                self.dec_lock_ref(end_node)
            finish_count -= 1

    def is_load_back_event_done(self, consumer_index: int) -> bool:
        """本地 load-back 事件完成后返回 True。"""
        if consumer_index < 0:
            return True

        finish_event = self.cache_controller.layer_done_counter.events[
            consumer_index
        ].finish_event
        if not finish_event.query():
            return False

        self.loading_check()
        return True

    def evictable_size(self):
        return self.evictable_size_

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        # 重写父类加锁：除维护 device 侧可淘汰叶子外，还需同步更新 host 侧可淘汰叶子集合。
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        # 重写父类解锁：同样需要在解锁路径上同步更新 host 侧可淘汰叶子集合。
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def _update_host_leaf_status(self, node: TreeNode):
        # 维护 host 侧可淘汰叶子集合。注意与 device 侧的差异：
        # host 叶子的候选条件是「device 上已被淘汰、且未加锁」——只有 device KV 已不在了，
        # 其 host 备份才谈得上被 host 淘汰；只要还有任一子节点在 host 有备份，它就不是 host 叶子。
        if not node.evicted or node.lock_ref > 0:
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            return

        for child in node.children.values():
            if child.backuped:
                if node in self.evictable_host_leaves:
                    self.evictable_host_leaves.remove(node)
                return

        if node not in self.evictable_host_leaves:
            self.evictable_host_leaves.add(node)

    def evict(self, params: EvictParams) -> EvictResult:
        # 重写父类 device 淘汰：在淘汰前，依据写策略决定是否先把未备份节点回写到 host
        # （write_back 模式下淘汰即回写，避免直接丢弃造成 L2 缺失）。
        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    # 节点尚未备份：在淘汰前先把它回写到 host，保住 L2 缓存。
                    written = self.write_backup(x, write_back=True)
                    num_evicted += written
                    if written > 0:
                        write_back_nodes.append(x)
                else:
                    num_evicted += self._evict_regular(x)
            else:
                num_evicted += self._evict_backuped(x)

            # 检查 x 的父节点：若其所有子节点都已被淘汰（write_back_nodes 视作即将淘汰），
            # 则父节点新晋为可淘汰叶子，入堆继续参与淘汰。
            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # 所有子节点都已淘汰，或根本没有子节点。
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

        # write_back 模式：等所有回写到 host 的搬运确认完成后，再正式把这些节点从 device 淘汰。
        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _evict_backuped(self, node: TreeNode):
        # GPU -> CPU 降级：数据块从 device 退到 host（host 上已有备份，节点保留在树中）。
        # 上报 remove(GPU)，让下游索引器不再把它当作「device 本地」来打分；
        # 对应的 store(CPU) 事件已在 write_backup() 拷贝到 host 时上报过。
        self._record_remove_event(node, medium=StorageMedium.GPU)
        num_evicted = self.cache_controller.evict_device(node.value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None  # 标记 device KV 已淘汰（但 host_value 仍在）
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        # 该节点被淘汰后，父节点的叶子状态也需重新评估。
        self._update_leaf_status(node.parent)
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # 淘汰一个从未发起过 host 写入的节点——直接从树上删除并上报 BlockRemoved。
        assert len(node.children) == 0, f"non-leaf, {node.id=}"

        self._record_remove_event(node)
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def evict_host(self, num_tokens: int):
        # 淘汰 host（L2）侧缓存：从 host 可淘汰叶子中按策略优先级淘汰，腾出 host 池空间。
        leaves = list(self.evictable_host_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            # 只淘汰「device 已被淘汰」节点的 host 值（device 还在的不能先丢 host 备份）。
            if not x.evicted:
                continue

            # host 引用计数 > 0：正被某次 storage 操作占用，跳过。
            if x.host_ref_counter > 0:
                continue

            # 数据块被彻底删除（GPU 早已淘汰，现在连 CPU 也释放）——
            # 上报 remove(CPU)，让 router 丢弃 host 层的该条目。
            self._record_remove_event(x, medium=StorageMedium.CPU)
            num_evicted += self.cache_controller.evict_host(x.host_value)

            # 从父节点 children 中摘除该节点，并维护 host 叶子集合。
            key = x.key.child_key(self.page_size)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            self._update_host_leaf_status(x.parent)

            # 父节点若变成「无子且 device 已淘汰」，则成为新的 host 淘汰候选，入堆。
            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        # 把一段「device 已淘汰、但 host 仍有备份」的前缀回载（load-back）回 device，
        # 使其重新可用于计算。返回回载到 device 的索引；放弃回载时返回 None。

        start_time = time.perf_counter()
        last_hit_node = node
        # 从 node 向上收集所有「已被 device 淘汰」的节点（按从根到叶的顺序），
        # 直到遇到第一个仍在 device 上的祖先 ancester_node。
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # 锁住祖先节点，防止它在本次回载过程中被淘汰。
        result = self.inc_lock_ref(ancester_node)
        delta = result.delta

        # 要么全部回载，要么完全不回载（避免只回载一半造成前缀不连续）。
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # 总量太小（不值得回载）或超出显存配额：放弃回载并解锁。
            self.dec_lock_ref(ancester_node)
            return None

        # 发起 host→device 的回载搬运。
        device_indices = self.cache_controller.load(
            host_indices=host_indices,
            node_id=last_hit_node.id,
            **self._get_extra_pools(),
        )
        if device_indices is None:
            # device 空间不足：先淘汰出足够空间，再重试一次回载。
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices,
                node_id=last_hit_node.id,
                **self._get_extra_pools(),
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # 即便淘汰后仍无足够显存来回载 KV，放弃。
            logger.warning(
                "load_back: FAILED to load %d tokens for node %d "
                "even after eviction (evictable_size=%d)",
                len(host_indices),
                last_hit_node.id,
                self.evictable_size_,
            )
            return None

        # 回载成功：登记为进行中，并把回载到的 device 索引按段写回各节点的 value。
        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)].clone()
            offset += len(node.host_value)
            # 数据块从 host 提升回 GPU——上报 store(GPU)，让下游索引器重新视其为 device 本地。
            self._record_store_event(node, medium=StorageMedium.GPU)
        self.evictable_size_ += len(device_indices)
        # 锁住末端节点，直到 loading_check 确认搬运完成后再解锁。
        self.inc_lock_ref(last_hit_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        return device_indices

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ):
        # 匹配的末端节点若已被 device 淘汰，则尝试把它回载回 device。
        # 回载成功返回回载到的索引与该末端节点；失败则退回到第一个仍在 device 上的祖先。
        last_node = params.best_match_node
        mem_quota = params.mem_quota
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            # 回载失败：沿父链上溯，找到第一个未被淘汰的祖先作为可用前缀末端。
            while last_node.evicted:
                last_node = last_node.parent

        return (
            self._empty_match_result.device_indices,
            last_node,
        )

    def query_storage_hit_length(
        self,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        # 向 L3 storage 查询：新输入 token 中有多长的前缀已存在于 storage 后端（用于决定是否预取）。
        # 未启用 storage、或预取被限流时直接返回 0。
        if not self.enable_storage or self.cache_controller.prefetch_rate_limited():
            return 0

        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=last_host_node.key.extra_key,
            is_bigram=self.is_eagle,
        ).page_aligned(self.page_size)
        # 待查前缀短于阈值则不值得查询 storage（预取收益太小）。
        if len(prefetch_key) < self.prefetch_threshold:
            return 0

        # 构造一个「仅查询」的探测操作（数据页为空切片），向后端问命中长度而不真正搬运。
        operation = PrefetchOperation(
            "__storage_hit_query__",
            self.cache_controller.mem_pool_host.get_dummy_flat_data_page()[:0],
            prefetch_key,
            last_hash,
            prefix_keys,
        )
        hash_values, storage_hit_count = self.cache_controller._storage_hit_query(
            operation
        )
        # 各 rank 取命中长度的最小值（MIN 规约），保证 TP/CP 组内一致地按「最短共识」预取。
        storage_hit_count_tensor = torch.tensor(storage_hit_count, dtype=torch.int)
        self._all_reduce_attn_groups(
            storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
        )
        storage_hit_count = storage_hit_count_tensor.item()
        # 把命中长度向下取整到 page_size 整数倍（页是搬运的最小单位）。
        storage_hit_count = storage_hit_count - (storage_hit_count % self.page_size)
        return storage_hit_count

    def ready_to_load_host_cache(self) -> int:
        """通知 cache controller 开始 KV 缓存的回载（load）。

        返回一个 consumer index，供调度批次管理器用于追踪本次回载的完成情况。
        """
        return self.cache_controller.start_loading()

    def flush_write_through_acks(self) -> None:
        # 收割已完成的写穿透 ack（非阻塞）。
        self.writing_check()

    def check_hicache_events(self):
        # 调度循环每步调用：集中收割写穿透/回载完成、排空 storage 控制队列、回收异步通信、上报指标。
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        self._reap_completed_async_work()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def drain_storage_control_queues(self):
        """合并处理 prefetch revoke、backup ack、host 内存释放三类控制消息。

        合并成一次跨 rank 同步，以尽量减少 TP 同步次数与 Python 开销。
        """
        cc = self.cache_controller

        # 先用 MIN 规约取三类队列在各 rank 上的最小长度，保证各 rank 只处理「大家都已就绪」的那部分。
        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.ack_backup_queue.qsize(),
                cc.host_mem_release_queue.qsize(),
            ],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(qsizes, torch.distributed.ReduceOp.MIN)

        n_revoke, n_backup, n_release = map(int, qsizes.tolist())
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
        )

    # 超时阈值随页数线性增长：timeout = min(max, base + 斜率 * token数/1024)。
    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        cfg = self.prefetch_timeout_config
        num_tokens = len(operation.hash_value) * self.page_size
        timeout = min(cfg.max, cfg.base + cfg.per_ki_token * num_tokens / 1024)
        return time.monotonic() - operation.start_time > timeout

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        # 判断一个预取操作此刻能否终止，行为取决于 prefetch_stop_policy。
        can_terminate = True

        # best_effort：随时可终止（不强求取满）。
        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        # 判断预取是否已「取满」（完成 token 数 == 期望页数 * 页大小）。
        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            # wait_complete：必须取满才允许终止。
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            # timeout：取满或超时之一即可终止。
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            # 未知策略：直接允许终止。
            return True

        if (
            completed
            and getattr(operation, "pool_transfers", None)
            and not getattr(operation, "pool_transfers_done", True)
        ):
            can_terminate = False

        # 跨 rank 用 MAX 规约对齐终止决策：
        # states[0] = 1-can_terminate（取 MAX 即「只要有一个 rank 不可终止，结果就是不可终止」），
        # states[1] = operation_terminated（取 MAX 即「只要有一个 rank 已终止，就视为已终止」）。
        operation_terminated = operation.is_terminated()
        states = torch.tensor(
            [1 - int(can_terminate), int(operation_terminated)],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(states, torch.distributed.ReduceOp.MAX)
        can_terminate = states[0].item() == 0
        operation_terminated = states[1].item() == 1
        # 终止条件：已在任一 TP worker 上被终止，或在所有 TP worker 上都满足了终止条件。
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def check_prefetch_progress(self, req_id: str) -> bool:
        # 检查某请求的预取进度：若已可终止，则把已取到的部分插入 host 树并清理簿记，返回 True。
        if req_id not in self.ongoing_prefetch:
            # 该请求没有进行中的预取，或预取已被撤销。
            return True

        # todo: 引入更多预取进度策略（如超时）。
        # 当前策略是尽力预取，并在排队结束时终止。
        last_host_node, prefetch_key, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]

        if operation.host_indices is None:
            # 因 host 内存不足，预取尚未真正发起。
            return True

        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        logger.debug(f"Prefetch {req_id} completed with {completed_tokens} tokens")

        min_completed_tokens = completed_tokens
        # 在改动 host 缓存树状态前，先跨 worker 同步「取到的最小完成 token 数」，保证各 rank 一致。
        completed_tokens_tensor = torch.tensor(min_completed_tokens, dtype=torch.int)
        self._all_reduce_attn_groups(
            completed_tokens_tensor, torch.distributed.ReduceOp.MIN
        )
        min_completed_tokens = completed_tokens_tensor.item()
        fetched_key = prefetch_key[:min_completed_tokens]
        written_indices = host_indices[:min_completed_tokens]
        matched_length = self._insert_helper_host(
            last_host_node,
            fetched_key,
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
        )

        # 释放已与树中既有 host 备份重复的那部分（matched_length），
        # 以及超出共识完成长度的多取部分（min..completed），只保留真正新增并入树的 host 页。
        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        last_host_node.release_host()
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

        # 统计本请求真正从 L3 storage 载入的 token 数（= 共识完成长度 - 树中已有的命中长度）。
        loaded_from_storage = min_completed_tokens - matched_length
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)

        return True

    def terminate_prefetch(self, req_id: str):
        # 主动标记某请求的预取为「终止」，使其在下一次进度检查时尽快收尾。
        if req_id not in self.ongoing_prefetch:
            return

        _, _, _, operation = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        """弹出并返回某请求从 storage 载入的 token 数。

        若未做过预取或预取被撤销则返回 0。应在 check_prefetch_progress() 返回 True 之后调用。
        """
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def match_prefix(self, params: MatchPrefixParams):
        # 重写父类前缀匹配：除返回 device 上命中的前缀外，还额外计算 host（L2）侧的命中信息，
        # 以便调度器据此决定是否触发 load-back / prefetch。
        if self.disable:
            return self._empty_match_result

        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result

        # 先做与父类一致的 device 侧匹配，得到 device 命中索引与末端节点。
        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = self._empty_match_result.device_indices

        # 从 device 末端继续沿父链上溯：
        # - 累计「device 已淘汰但 host 仍有备份」节点的长度，得到可回载的 host 命中长度；
        # - 同时定位最近一个仍有 host 备份的节点 last_host_node。
        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent
        while not last_host_node.backuped:
            last_host_node = last_host_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            # TODO(ispobock): 后续应使用 best_match_node 作为 load_back 的起始节点
            best_match_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        """从 L3 存储后端把命中的前缀预取到 L2（host 内存）。

        当请求的输入在 device/host 都未命中、但可能存在于存储后端（磁盘/远端）时，
        发起一次异步预取：先在 host 内存池中分配落地空间，再交给 cache_controller
        异步搬运。预取受阈值、限流与 host 可用内存约束，必要时会缩减长度做
        「尽力而为」的部分预取，最终把进行中的预取记录登记到 ``ongoing_prefetch``。
        """
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=last_host_node.key.extra_key,
            is_bigram=self.is_eagle,
        )
        # 把待拉取的 token 数对齐到 page_size 的整数倍（页粒度管理的要求）。
        prefetch_key = prefetch_key.page_aligned(self.page_size)
        prefetch_length = len(prefetch_key)
        # 以下任一情况都不发起预取：未启用存储后端、可预取长度低于阈值、
        # 或控制器当前处于预取限流状态（避免占满带宽）。
        if (
            not self.enable_storage
            or prefetch_length < self.prefetch_threshold
            or self.cache_controller.prefetch_rate_limited()
        ):
            return

        # 先把 last_host_node 标记为受保护，避免预取过程中它被 host 侧淘汰回收。
        last_host_node.protect_host()
        # 在 host 内存池中为本次预取分配落地空间；分配失败则先淘汰部分 host 缓存再重试。
        host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            self.evict_host(prefetch_length)
            host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            # 淘汰后仍无法满额分配：退而求其次，按当前 host 可用空间（向下对齐到页）
            # 缩减本次预取长度，做「尽力而为」的部分预取。
            available_size = self.cache_controller.mem_pool_host.available_size()
            prefetch_length = available_size - (available_size % self.page_size)
            if prefetch_length >= self.prefetch_threshold:
                prefetch_key = prefetch_key[:prefetch_length]
                host_indices = self.cache_controller.mem_pool_host.alloc(
                    prefetch_length
                )
                if host_indices is None:
                    last_host_node.release_host()
                    return
            else:
                # 缩减后的长度仍低于阈值，放弃预取并释放此前的保护。
                last_host_node.release_host()
                return
        operation = self.cache_controller.prefetch(
            req_id,
            host_indices,
            prefetch_key,
            last_hash,
            prefix_keys,
            **self._get_extra_pools(),
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)

    def _insert_helper_host(
        self, node: TreeNode, key: RadixKey, host_value, hash_value
    ):
        """把一段「仅 host 存在」的前缀（含 host_value/hash_value）插入到树中。

        用于预取/回写完成后，将搬运到 host 内存的 KV 数据登记进 radix 树：
        沿已有前缀逐段下行匹配，必要时拆分节点；当存在尾部未匹配部分时，新建一个
        只持有 ``host_value``（device 侧 ``value`` 为空）的节点挂到树上，并立即
        发布对应的 store 事件。返回已匹配（已存在）的前缀长度。
        """
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = node.key.match(key, page_size=self.page_size)
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=node.priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.host_value = host_value.clone()
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
            self._update_host_leaf_status(new_node)
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            # 立即发布新落地的 host 后缀，使下游缓存索引器能够解析那些
            # 在此「仅 L2 存在」前缀之上继续延展的后代节点。
            self._record_store_event(new_node, medium=StorageMedium.CPU)

        return matched_length

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        """HiCache 版本的前缀匹配辅助函数（device 侧）。

        与父类 RadixCache 的区别在于：本类的节点可能「device 已淘汰但 host 仍有备份」
        （即 ``node.evicted`` 为真）。这类节点虽然仍在树结构中以维持前缀链，但其
        device 侧的 ``value`` 已被回收。因此这里在收集命中的 device 索引时，会
        **跳过已淘汰节点的 value**，只把仍驻留在 device 上的部分拼接返回。
        """
        node.last_access_time = time.monotonic()
        child_key = key.child_key(self.page_size)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                # 部分匹配：把 child 拆分，使新节点恰好对应公共前缀。
                new_node = self._split_node(child.key, child, prefix_len)
                # 仅当新节点未被淘汰（device 上仍有 value）时才收集其索引。
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                # 整段匹配：同样跳过已淘汰节点，只收集 device 仍存在的部分。
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        """HiCache 版本的节点拆分。

        在 ``split_len`` 处把 ``child`` 拆成两段，插入一个新的前缀节点：
        拆分后结构变为 ``new_node -> child``，new_node 承载公共前缀
        ``key[:split_len]``，原 child 保留剩余后缀。

        与父类相比，本实现需要**同时拆分 device 侧的 ``value`` 与 host 侧的
        ``host_value``**（若存在备份），并相应地拆分 ``hash_value``、迁移挂起的
        write-through 记录，以保证多层缓存在拆分后仍然自洽。
        """
        # 新节点作为公共前缀，原 child 降为其子节点（new_node -> child）。
        new_node = TreeNode(priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        # 同步拆分 device 侧 value 与 host 侧 host_value（若存在备份）。
        if child.evicted:
            # device 已淘汰：新节点 device 侧无 value。
            new_node.value = None
        else:
            new_node.value = child.value[:split_len].clone()
            child.value = child.value[split_len:].clone()
        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        # 哈希值也按 split_len 切分，使两段各自携带正确的页级哈希。
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        if child.backuped:
            # 若 child 上存在挂起的 write-through 写回任务，需把它重定向到
            # 拆分后的两个节点，避免回写记录指向已失效的旧节点。
            self._replace_pending_write_through_node(child, [new_node, child])

        return new_node

    def insert(self, params: InsertParams) -> InsertResult:
        """HiCache 版本的插入：把一段（page 对齐的）KV 序列写入 radix 树。

        相对父类的关键差异：节点可能处于「device 已淘汰但 host 仍有备份」状态
        （``node.evicted`` 为真）。当插入命中这类节点时，会把传入的 device ``value``
        重新挂回该节点（常见于 KV cache 重算场景），并相应更新 device 侧的叶子状态
        与可淘汰空间。返回已存在的命中前缀总长度。
        """
        key = params.key
        value = params.value
        chunked = params.chunked
        priority = params.priority

        if priority is None:
            priority = 0

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]

        if len(key) == 0:
            return InsertResult(prefix_len=0)

        node = self.root_node
        child_key = key.child_key(self.page_size)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prefix_len == len(node.key):
                if node.evicted:
                    # 节点 device 侧已被淘汰：把传入的 value 重新挂回该节点。
                    # 这种情况常见于 KV cache 重算（recomputation）后回填。
                    node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(node.value)
                    self._update_leaf_status(node)
                    self._update_host_leaf_status(node)
                    # 由于 device 上新增了一个叶子，需同步更新父节点的叶子状态。
                    self._update_leaf_status(node.parent)
                else:
                    self._inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # 部分匹配：在 prefix_len 处拆分节点，使公共前缀单独成节点。
                new_node = self._split_node(node.key, node, prefix_len)
                # 共享前缀节点也应反映本次插入带来的最大 priority。
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    new_node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(new_node.value)
                    self._update_leaf_status(new_node)
                    self._update_host_leaf_status(new_node)
                    # 由于 device 上新增了一个叶子，需同步更新父节点的叶子状态。
                    self._update_leaf_status(new_node.parent)
                else:
                    self._inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

            # 仅当启用存储后端或 KV cache 事件时才计算 hash_value（按页哈希）。
            if self.enable_storage or self.enable_kv_cache_events:
                new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

            # 发出 BlockStored 事件，让 router 把这个 block 纳入索引。
            self._record_store_event(new_node)

            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(new_node, chunked)
        return InsertResult(prefix_len=total_prefix_length)

    def release_aborted_request(self, rid: str):
        """请求被中止时，清理其相关的预取与存储命中跟踪状态。

        终止进行中的预取操作、释放被保护的 host 节点，回收已分配但未用完的
        host 内存，并从 ``ongoing_prefetch`` 中移除该请求记录，防止资源泄漏。
        """
        # 清理该中止请求的存储命中跟踪信息。
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)

        if rid not in self.ongoing_prefetch:
            return

        last_host_node, prefetch_key, host_indices, operation = self.ongoing_prefetch[
            rid
        ]
        if operation.host_indices is None:
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        self._barrier_attn_groups()
        last_host_node.release_host()
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)
