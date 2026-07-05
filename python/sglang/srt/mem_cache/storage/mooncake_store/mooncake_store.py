import ctypes
import json
import logging
import os
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import requests
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.memory_pool_host import (
    HostKVCache,
    HostTensorAllocator,
    MLATokenToKVPoolHost,
)
from sglang.srt.observability.metrics_collector import StorageMetrics

# Mooncake 零拷贝接口不需要本地缓冲区，但 setup 仍要求传入一个默认大小占位。
DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MB
SETUP_TIMEOUT = 600  # 等待 mooncake store server 启动的超时时间（10 分钟）

logger = logging.getLogger(__name__)


class MooncakeHostTensorAllocator(HostTensorAllocator):
    """host 侧张量分配器，底层内存由 Mooncake 的 MooncakeHostMemAllocator 提供。

    standalone（独立 client）模式下要求使用本分配器，以便真实的 mooncake_client
    进程能够按指针映射这些 host 缓冲区（配合 register_buffer 做零拷贝传输）。
    """

    def __init__(self):
        super().__init__()
        from mooncake.store import MooncakeHostMemAllocator

        self.allocator = MooncakeHostMemAllocator()
        self.ptr = None

    def allocate(
        self, dims: tuple, dtype: torch.dtype, device: str = "cpu"
    ) -> torch.Tensor:
        """用 MooncakeHostMemAllocator 分配内存，并包装成 PyTorch 张量返回。"""
        self.dims = dims
        self.dtype = dtype
        # 按各维乘积算出元素个数，再乘以单元素字节数得到总字节数。
        size = 1
        for d in dims:
            size *= d
        size *= torch.tensor([], dtype=self.dtype).element_size()
        # 从 Mooncake 分配器拿到一块裸内存指针（整数地址）。
        ptr_int = self.allocator.alloc(size)
        self.ptr = ptr_int
        # 通过 ctypes 把裸地址包成字节数组，再用 frombuffer 零拷贝地映射为 uint8 张量。
        c_type = ctypes.c_byte * size
        c_array = c_type.from_address(ptr_int)

        tensor = torch.frombuffer(c_array, dtype=torch.uint8, count=size)

        # 若目标 dtype 非 uint8，需按元素大小整除校验后再 view 成目标类型。
        if dtype != torch.uint8:
            element_size = torch.tensor([], dtype=dtype).element_size()
            assert size % element_size == 0, "Size must be divisible by element size"
            tensor = tensor.view(dtype)

        return tensor.view(dims)


def _parse_global_segment_size(value) -> int:
    """解析 global_segment_size 配置：支持整数、纯数字字符串以及带 'gb' 后缀的字符串。"""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        # 形如 "32gb" -> 32 * 1024^3 字节。
        if s.endswith("gb"):
            num = s[:-2].strip()
            if not num:
                raise ValueError(
                    "Invalid global_segment_size: missing number before 'gb'"
                )
            return int(num) * 1024 * 1024 * 1024
        return int(s)
    return int(value)


@dataclass
class MooncakeStoreConfig:
    """MooncakeStore 的连接与部署配置。

    支持三种加载来源（优先级：extra_config > 配置文件 > 环境变量），三者字段一致。
    """

    local_hostname: str  # 本地主机名（transfer engine 会话标识用）
    metadata_server: str  # 元数据服务地址（如 "P2PHANDSHAKE"）
    global_segment_size: int  # 全局段大小（字节），会按 tp_size 均摊到每个 rank
    protocol: str  # 传输协议，如 "rdma" / "tcp"
    device_name: str  # RDMA 设备名；也可传 JSON 按 tp_rank 映射不同网卡
    master_server_address: str  # master 服务地址
    master_metrics_port: int  # master 指标端口（check_server 时轮询）
    check_server: bool  # 启动时是否等待并检查 server 就绪
    standalone_storage: bool  # 是否为独立存储（dummy client）模式
    client_server_address: str  # standalone 模式下真实 client 的地址
    enable_ssd_offload: bool = False  # 是否启用 SSD 卸载
    ssd_offload_path: Optional[str] = None  # SSD 卸载路径

    @staticmethod
    def from_file() -> "MooncakeStoreConfig":
        """从 JSON 配置文件加载配置（路径由 SGLANG_HICACHE_MOONCAKE_CONFIG_PATH 指定）。"""
        if not envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():
            raise RuntimeError(
                f"Config file path not set. Please set {envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.name}"
            )
        file_path = envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.get()
        try:
            with open(file_path) as fin:
                config = json.load(fin)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {file_path}: {str(e)}")

        # master_server_address 与 client_server_address 至少要提供其一。
        if (
            "master_server_address" not in config
            and "client_server_address" not in config
        ):
            raise ValueError(
                "Either master_server_address or client_server_address is required in config file"
            )

        return MooncakeStoreConfig(
            local_hostname=config.get(
                "local_hostname", envs.MOONCAKE_LOCAL_HOSTNAME.default
            ),
            metadata_server=config.get(
                "metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default
            ),
            global_segment_size=_parse_global_segment_size(
                config.get(
                    "global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default
                )
            ),
            protocol=config.get("protocol", envs.MOONCAKE_PROTOCOL.default),
            device_name=config.get("device_name", envs.MOONCAKE_DEVICE.default),
            master_server_address=config.get(
                "master_server_address", envs.MOONCAKE_MASTER.default
            ),
            master_metrics_port=config.get(
                "master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default
            ),
            check_server=config.get("check_server", envs.MOONCAKE_CHECK_SERVER.default),
            standalone_storage=config.get(
                "standalone_storage", envs.MOONCAKE_STANDALONE_STORAGE.default
            ),
            client_server_address=config.get(
                "client_server_address", envs.MOONCAKE_CLIENT.default
            ),
            enable_ssd_offload=config.get(
                "enable_ssd_offload", envs.MOONCAKE_ENABLE_SSD_OFFLOAD.default
            ),
            ssd_offload_path=config.get(
                "ssd_offload_path", envs.MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.default
            ),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """从环境变量加载配置。示例：
        export MOONCAKE_MASTER=10.13.3.232:50051
        export MOONCAKE_PROTOCOL="rdma"
        export MOONCAKE_DEVICE=""
        export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
        """
        # MOONCAKE_MASTER 与 MOONCAKE_CLIENT 至少要设置其一。
        if not envs.MOONCAKE_MASTER.is_set() and not envs.MOONCAKE_CLIENT.is_set():
            raise ValueError(
                "Either the environment variable 'MOONCAKE_MASTER' or 'MOONCAKE_CLIENT' is not set."
            )

        # local_hostname 特殊处理：优先取 MOONCAKE_LOCAL_HOSTNAME，
        # 未设置时回退到旧的 LOCAL_HOSTNAME（保持对遗留环境变量的前向兼容）。
        if envs.MOONCAKE_LOCAL_HOSTNAME.is_set():
            local_hostname = envs.MOONCAKE_LOCAL_HOSTNAME.get()
        else:
            local_hostname = os.getenv(
                "LOCAL_HOSTNAME", envs.MOONCAKE_LOCAL_HOSTNAME.default
            )

        return MooncakeStoreConfig(
            local_hostname=local_hostname,
            metadata_server=envs.MOONCAKE_TE_META_DATA_SERVER.get(),
            global_segment_size=_parse_global_segment_size(
                envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.get()
            ),
            protocol=envs.MOONCAKE_PROTOCOL.get(),
            device_name=envs.MOONCAKE_DEVICE.get(),
            master_server_address=envs.MOONCAKE_MASTER.get(),
            master_metrics_port=envs.MOONCAKE_MASTER_METRICS_PORT.get(),
            check_server=envs.MOONCAKE_CHECK_SERVER.get(),
            standalone_storage=envs.MOONCAKE_STANDALONE_STORAGE.get(),
            client_server_address=envs.MOONCAKE_CLIENT.get(),
            enable_ssd_offload=envs.MOONCAKE_ENABLE_SSD_OFFLOAD.get(),
            ssd_offload_path=envs.MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.get(),
        )

    @staticmethod
    def load_from_extra_config(extra_config: dict) -> "MooncakeStoreConfig":
        """从 extra_config 字典加载配置（HiCacheStorageConfig.extra_config 传入）。"""
        # 同样要求 master/client 地址至少提供其一。
        if (
            "master_server_address" not in extra_config
            and "client_server_address" not in extra_config
        ):
            raise ValueError(
                "Either master_server_address or client_server_address is required in extra_config"
            )

        return MooncakeStoreConfig(
            local_hostname=extra_config.get(
                "local_hostname", envs.MOONCAKE_LOCAL_HOSTNAME.default
            ),
            metadata_server=extra_config.get(
                "metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default
            ),
            global_segment_size=_parse_global_segment_size(
                extra_config.get(
                    "global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default
                )
            ),
            protocol=extra_config.get("protocol", envs.MOONCAKE_PROTOCOL.default),
            device_name=extra_config.get("device_name", envs.MOONCAKE_DEVICE.default),
            master_server_address=extra_config.get(
                "master_server_address", envs.MOONCAKE_MASTER.default
            ),
            master_metrics_port=extra_config.get(
                "master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default
            ),
            check_server=extra_config.get(
                "check_server", envs.MOONCAKE_CHECK_SERVER.default
            ),
            standalone_storage=extra_config.get(
                "standalone_storage", envs.MOONCAKE_STANDALONE_STORAGE.default
            ),
            client_server_address=extra_config.get(
                "client_server_address", envs.MOONCAKE_CLIENT.default
            ),
            enable_ssd_offload=extra_config.get(
                "enable_ssd_offload", envs.MOONCAKE_ENABLE_SSD_OFFLOAD.default
            ),
            ssd_offload_path=extra_config.get(
                "ssd_offload_path", envs.MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.default
            ),
        )


class MooncakeBaseStore:
    """Mooncake store 的基础封装：负责导入依赖、加载配置、注册零拷贝缓冲区。"""

    def __init__(self):
        self.store = None
        self.config = None

    def _import_mooncake_store(self):
        """延迟导入 mooncake 依赖；缺失时给出安装指引。"""
        try:
            from mooncake.store import MooncakeDistributedStore

            return MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html "
                "to run SGLang with MooncakeConnector."
            ) from e

    def _load_config(self, storage_config: Any = None):
        """按优先级加载配置：extra_config > 配置文件 > 环境变量。"""
        extra_config = (
            getattr(storage_config, "extra_config", None) if storage_config else None
        )

        # 1) extra_config 中显式给出 master/client 地址时优先使用。
        if extra_config and (
            extra_config.get("master_server_address") is not None
            or extra_config.get("client_server_address") is not None
        ):
            config = MooncakeStoreConfig.load_from_extra_config(extra_config)
            logger.info("Mooncake Configuration loaded from extra_config successfully.")

        # 2) 否则若设置了配置文件路径环境变量，则从文件加载。
        elif envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():
            config = MooncakeStoreConfig.from_file()
            logger.info("Mooncake Configuration loaded from file successfully.")

        # 3) 兜底：从环境变量加载。
        else:
            config = MooncakeStoreConfig.load_from_env()
            logger.info("Mooncake Configuration loaded from env successfully.")

        return config

    def register_buffer(self, tensor: torch.Tensor):
        """把 host 张量按 (指针, 字节数) 注册给 Mooncake，用于后续零拷贝 RDMA 传输。"""
        if self.store is None:
            raise RuntimeError("Mooncake store is not initialized.")
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        ret_code = self.store.register_buffer(ptr, size)
        if ret_code != 0:
            logger.error(f"Failed to register buffer, error code: {ret_code}")
            raise RuntimeError(
                f"Failed to register buffer to Mooncake Store, error code: {ret_code}"
            )


class MooncakeStore(HiCacheStorage, MooncakeBaseStore):
    """HiCache 的 Mooncake L3 存储后端实现。

    通过零拷贝接口把 host KV 缓冲区按 (指针, 字节数) 注册给 Mooncake，
    以 page 为粒度做 batch get/set；同时支持 hybrid 模型（Mamba/DSA/DeepSeek V4 等）
    的多类 side pool，以及 PD/TP/PP/CP 各维度的 key 命名区分。
    """

    @staticmethod
    def _standalone_required_bytes(mem_pool: Any) -> int:
        """计算 standalone 模式下必须让真实 client 可见的 host 缓冲区总字节数。

        standalone（dummy client）模式下，真正的 mooncake_client 进程需要映射所有
        之后会经 register_buffer() 按指针传入的 host 缓冲区。对于 hybrid 模型，
        这既包含 KV 主缓冲，也包含 side pool（如 Mamba 的 temporal/conv 状态）。
        """
        # 优先使用通用的 "hybrid pool" 访问器（若存在）。
        total = 0
        seen_ptrs: set[int] = set()

        def _add_tensor(t: Optional[torch.Tensor]):
            """累加单个张量的字节数，并按指针去重（避免同一块内存重复计入）。"""
            nonlocal total
            if t is None:
                return
            try:
                ptr = int(t.data_ptr())
            except Exception:
                return
            if ptr in seen_ptrs:
                return
            seen_ptrs.add(ptr)
            total += int(t.numel() * t.element_size())

        # 始终计入锚点 KV 缓冲区（若存在）。
        _add_tensor(getattr(mem_pool, "kv_buffer", None))

        # HostPoolGroup：逐个 pool 计入其 hybrid 缓冲区。
        entries = getattr(mem_pool, "entries", None)
        if entries:
            for entry in entries:
                host_pool = getattr(entry, "host_pool", None)
                if host_pool is None:
                    continue
                # KV pool 的锚点内存前面已计入，这里重复添加也无害（有去重）。
                _add_tensor(getattr(host_pool, "kv_buffer", None))
                for buf in getattr(host_pool, "get_hybrid_pool_buffer", lambda: [])():
                    _add_tensor(buf)
            return total

        # 单个 HostKVCache 型 pool：追加其 side 缓冲区（若有）。
        for buf in getattr(mem_pool, "get_hybrid_pool_buffer", lambda: [])():
            _add_tensor(buf)
        return total

    def __init__(
        self, storage_config: HiCacheStorageConfig = None, mem_pool: HostKVCache = None
    ):
        MooncakeBaseStore.__init__(self)
        MooncakeDistributedStore = self._import_mooncake_store()
        try:
            self.store = MooncakeDistributedStore()

            self.config = self._load_config(storage_config)
            extra_config = (
                getattr(storage_config, "extra_config", None)
                if storage_config
                else None
            )
            # 全局段大小按 tp_size 均摊：每个 TP rank 只占用其中一份。
            tp_scale_factor = 1 if storage_config is None else storage_config.tp_size

            per_tp_global_segment_size = (
                self.config.global_segment_size // tp_scale_factor
            )

            # extra_backend_tag：可选的 key 前缀，用于隔离不同后端/租户的对象命名空间。
            self.extra_backend_tag = None
            if extra_config and "extra_backend_tag" in extra_config:
                self.extra_backend_tag = extra_config["extra_backend_tag"]
                logger.info(f"Using extra_backend_tag: {self.extra_backend_tag}")

            # 若配置要求，先阻塞等待 server 就绪。
            if self.config.check_server:
                self.check_server()

            # device_name 支持传 JSON：按 tp_rank 为不同 rank 指定不同的 RDMA 网卡。
            device_name = self.config.device_name
            if device_name and device_name.strip().startswith("{"):
                try:
                    device_config = json.loads(device_name)
                    if storage_config and hasattr(storage_config, "tp_rank"):
                        tp_rank = storage_config.tp_rank
                        # 整数键与字符串键都尝试，因为 JSON 解析可能把键转成字符串。
                        device_name = device_config.get(tp_rank, "")
                        if not device_name:
                            device_name = device_config.get(str(tp_rank), "")
                    else:
                        device_name = ""
                except (json.JSONDecodeError, AttributeError):
                    logger.warning(
                        f"Failed to parse device_name as JSON: {device_name}"
                    )
                    device_name = ""
            if self.config.standalone_storage:
                # standalone 模式：本进程只是 dummy client，需向真实 client 声明要映射的字节数。
                if not isinstance(mem_pool.allocator, MooncakeHostTensorAllocator):
                    raise RuntimeError(
                        "MooncakeStore with standalone_storage=True requires MooncakeHostTensorAllocator. "
                        "Please set standalone_storage=False "
                        "or upgrade Mooncake by 'pip install mooncake --upgrade'."
                    )
                required_bytes = self._standalone_required_bytes(mem_pool)
                ret_code = self.store.setup_dummy(
                    required_bytes,
                    DEFAULT_LOCAL_BUFFER_SIZE,  # 零拷贝接口不需要本地缓冲区
                    self.config.client_server_address,
                )
            else:
                # 非 standalone：尝试复用进程内已初始化的共享 transfer engine，避免重复建连。
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_mooncake_transfer_engine,
                    )

                    self._shared_mooncake_transfer_engine = (
                        get_mooncake_transfer_engine()
                    )
                except Exception:
                    self._shared_mooncake_transfer_engine = None
                    logger.debug("Failed to reuse initialized mooncake transfer engine")

                # 仅当共享 transfer engine 的配置与本 store 一致时才复用
                # （同一网卡、P2PHANDSHAKE 元数据、rdma 协议）。
                if (
                    self._shared_mooncake_transfer_engine is not None
                    and device_name
                    == self._shared_mooncake_transfer_engine.get_ib_device()
                    and self.config.metadata_server == "P2PHANDSHAKE"
                    and self.config.protocol == "rdma"
                ):
                    client_hostname = (
                        self._shared_mooncake_transfer_engine.get_session_id()
                    )
                    transfer_engine = self._shared_mooncake_transfer_engine.get_engine()
                    logger.info(
                        f"Reuse initialized mooncake transfer engine: {self._shared_mooncake_transfer_engine}"
                    )
                else:
                    # 不复用：用本地主机名新建 transfer engine（传 None 让 setup 内部创建）。
                    client_hostname = self.config.local_hostname
                    transfer_engine = None

                # SSD 卸载相关的可选参数，仅在启用时加入。
                setup_kwargs = {}
                if self.config.enable_ssd_offload:
                    setup_kwargs["enable_ssd_offload"] = True
                if self.config.ssd_offload_path is not None:
                    setup_kwargs["ssd_offload_path"] = self.config.ssd_offload_path

                # 循环重试：若安装的 Mooncake 版本不支持某些 setup 参数，剔除后重试。
                while True:
                    try:
                        ret_code = self.store.setup(
                            client_hostname,
                            self.config.metadata_server,
                            per_tp_global_segment_size,
                            DEFAULT_LOCAL_BUFFER_SIZE,  # 零拷贝接口不需要本地缓冲区
                            self.config.protocol,
                            device_name,
                            self.config.master_server_address,
                            transfer_engine,
                            **setup_kwargs,
                        )
                        break
                    except TypeError as e:
                        # 从异常信息里挑出不被支持的 kwargs；若非此原因则直接抛出。
                        unsupported_kwargs = [
                            key for key in list(setup_kwargs) if key in str(e)
                        ]
                        if not unsupported_kwargs:
                            raise
                        logger.warning(
                            "The installed Mooncake version does not support the "
                            f"{', '.join(unsupported_kwargs)} parameter(s) in setup(). "
                            f"Retrying without {', '.join(unsupported_kwargs)}. "
                            "Please upgrade Mooncake to enable SSD offload support."
                        )
                        for key in unsupported_kwargs:
                            setup_kwargs.pop(key, None)
            if ret_code:
                raise RuntimeError(
                    f"Failed to setup Mooncake store, error code: {ret_code}"
                )
            logger.info("Mooncake store setup successfully.")

            self.local_rank = (
                storage_config.tp_rank if storage_config is not None else 0
            )
            # 预热：put/get 一个小对象，规避 transfer engine 启动竞态。
            self.warmup()
            logger.info("Mooncake store warmup successfully.")

            # 从 storage_config 读取并行维度信息（无配置时取单卡默认值）。
            self.enable_storage_metrics = False
            if storage_config is not None:
                self.is_mla_backend = storage_config.is_mla_model
                self.pp_rank = storage_config.pp_rank
                self.pp_size = storage_config.pp_size
                self.attn_cp_rank = storage_config.attn_cp_rank
                self.attn_cp_size = storage_config.attn_cp_size
                self.enable_storage_metrics = storage_config.enable_storage_metrics
            else:
                self.is_mla_backend = False
                self.local_rank = 0
                self.pp_rank = 0
                self.pp_size = 1
                self.attn_cp_rank = 0
                self.attn_cp_size = 1

            # 构造对象 key 的 rank 后缀：
            # MHA 每个 TP rank 的 K/V 独立存放，故后缀含 local_rank；
            # MLA 的 KV 在 TP 间共享，故后缀不含 local_rank（仅 PP 维度区分）。
            self.enable_pp = self.pp_size > 1
            if self.enable_pp:
                self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"
                self.mla_suffix = f"{self.pp_rank}"
            else:
                self.mha_suffix = f"{self.local_rank}"
                self.mla_suffix = ""

            # should_split_heads：当本 rank 需把注意力头拆分写到多个逻辑 rank 时，
            # split_factor 表示拆分份数，mha_suffix 相应扩展为多后缀列表。
            self.storage_config = storage_config
            self.split_factor = 0
            if self.storage_config.should_split_heads:
                self.split_factor = (
                    self.storage_config.tp_lcm_size // self.storage_config.tp_size
                )
                base_rank = self.local_rank * self.split_factor
                target_ranks = [base_rank + i for i in range(self.split_factor)]
                if self.enable_pp:
                    self.mha_suffix = [
                        f"{rank}_{self.pp_rank}" for rank in target_ranks
                    ]
                else:
                    self.mha_suffix = [f"{rank}" for rank in target_ranks]

            # name->host_pool 映射，供 v2 批量接口按 PoolTransfer.name 找到对应 host pool。
            self.registered_pools = {}

            # 指标统计：每 page 的 GB 数，以及预取/备份的页数与带宽采样缓存。
            self.gb_per_page = None
            self.prefetch_pgs = []
            self.backup_pgs = []
            self.prefetch_bandwidth = []
            self.backup_bandwidth = []

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

    @staticmethod
    def _iter_host_pool_buffers(host_pool: HostKVCache):
        """遍历一个 host pool 需注册的所有缓冲区。

        优先用 get_hybrid_pool_buffer()（hybrid pool 可能有多个 side 缓冲区），
        缺省时回退到单个 kv_buffer；跳过 None。
        """
        get_buffers = getattr(
            host_pool,
            "get_hybrid_pool_buffer",
            lambda: [getattr(host_pool, "kv_buffer", None)],
        )
        for buf in get_buffers():
            if buf is not None:
                yield buf

    def check_server(self):
        """轮询 master 的 /get_all_segments 接口，阻塞等待 server 就绪（超时抛错）。"""
        master_server_ip = self.config.master_server_address.split(":")[0]
        segments_url = f"http://{master_server_ip}:{self.config.master_metrics_port}/get_all_segments"
        start_time = time.perf_counter()

        check_result = False
        while time.perf_counter() - start_time < SETUP_TIMEOUT:
            try:
                check_segments_resp = requests.get(segments_url, timeout=3)
            except Exception:
                logger.info(
                    "waiting mooncake store server started, cost_time: %.2f seconds.",
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            if check_segments_resp.text == "":
                logger.info(
                    "waiting mooncake store server started, cost_time: %.2f seconds.",
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            logger.info("Mooncake store server started successfully.")
            check_result = True
            break

        if not check_result:
            logger.error("Launch mooncake store server timeout")
            raise ValueError("Launch mooncake store server timeout")

    def warmup(self):
        """预热：写入并校验一个小对象，规避 transfer engine 刚启动时的竞态。"""
        warmup_key = "sglang_mooncake_store_warmup_key" + uuid.uuid4().hex
        warmup_value = bytes(4 * 1024)  # 4 KB

        # 重试逻辑：应对 transfer engine 启动竞态导致的首次 put 失败。
        max_retries = 10
        retry_delay = 1.0  # 秒

        for attempt in range(max_retries):
            ret = self.store.put(warmup_key, warmup_value)
            if ret == 0:
                break
            logger.warning(
                f"[TP{self.local_rank}] Warmup put failed (attempt {attempt + 1}/{max_retries}), "
                f"ret={ret}, retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)
        else:
            raise RuntimeError(
                f"[TP{self.local_rank}] Warmup put failed after {max_retries} attempts, "
                "Transfer Engine might not be ready"
            )

        assert self.store.is_exist(warmup_key) == 1
        assert self.store.get(warmup_key) == warmup_value

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        """注册主 host KV 池的缓冲区，并计算每 page 的 GB 数（供带宽统计）。"""
        super().register_mem_pool_host(mem_pool_host)
        if getattr(self.mem_pool_host, "kv_buffer", None) is None:
            # hybrid 逻辑锚点只拥有分配索引、无物理张量；其物理缓冲经
            # register_mem_host_pool_v2() 注册。
            return
        try:
            for buffer in self._iter_host_pool_buffers(self.mem_pool_host):
                super().register_buffer(buffer)
        except TypeError as err:
            logger.error("Failed to register buffer to Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Register Buffer Error.") from err

        bytes_per_page = mem_pool_host.get_ksize_per_token() * mem_pool_host.page_size
        self.gb_per_page = bytes_per_page / (1 << 30)

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        """v2：注册除主 KV 锚点之外的额外 hybrid/side pool 缓冲区。"""
        # KV 锚点内存已在 register_mem_pool_host() 注册，这里跳过。
        if host_pool_name == PoolName.KV:
            return
        # DRAFT（投机解码草稿）池：登记映射并注册其单个 kv_buffer。
        if host_pool_name == PoolName.DRAFT:
            self.registered_pools[host_pool_name] = host_pool
            super().register_buffer(host_pool.kv_buffer)
            return

        # 维护 name->pool 映射，使 v2 批量接口能在运行时按 PoolTransfer.name
        # 找到对应的 host pool 实现。
        self.registered_pools[host_pool_name] = host_pool

        # 非锚点池：要么是带自有访问器的 side 专用池，要么是用作 SWA 侧的普通 KV 型池。
        for buf in self._iter_host_pool_buffers(host_pool):
            super().register_buffer(buf)

    def _tag_keys(self, keys: List[str]) -> List[str]:
        """为所有 key 加上 extra_backend_tag 前缀（未配置则原样返回）。"""
        if self.extra_backend_tag is None:
            return keys
        return [f"{self.extra_backend_tag}_{key}" for key in keys]

    def _get_hybrid_page_component_keys(
        self, page_keys: List[str], transfer: PoolTransfer
    ) -> Tuple[List[str], int]:
        """把一个 hybrid pool 的每个 page key 展开为其组成对象的 key 列表。

        返回 (component_keys, key_multiplier)：一个逻辑 page 会对应
        key_multiplier 个存储对象（如 K+V 两个、Mamba 的 temporal+多个 conv 等）。
        后缀顺序必须与该 pool 的 get_page_buffer_meta() 输出顺序一致，因为
        Mooncake 会把对象 key 与注册的缓冲区指针按序 zip 起来。
        """
        host_pool = getattr(self, "registered_pools", {}).get(transfer.name)
        if host_pool is None:
            raise ValueError(f"Unregistered Mooncake hybrid pool: {transfer.name}")

        pool_name = transfer.name
        suffixes = []
        if pool_name == PoolName.MAMBA:
            # Mamba：一个 temporal 对象 + 每个 conv 状态一个对象。
            conv_num = len(getattr(host_pool, "conv_buffer", None) or [])
            suffixes = [f"_{self.mha_suffix}_temporal"] + [
                f"_{self.mha_suffix}_conv_{i}" for i in range(conv_num)
            ]
        elif pool_name == PoolName.DRAFT:
            # 草稿池的 MLA/MHA 布局与目标模型相互独立（如 MLA 目标上挂 EAGLE-MHA 草稿），
            # 因此后缀方案取自草稿池自身的类。`_draft` 标签用于避免这些 key 与目标模型的
            # `{rank}_k` / `{rank}_k`+`{rank}_v` 命名冲突。
            draft_pool = self.registered_pools.get(PoolName.DRAFT)
            if isinstance(draft_pool, MLATokenToKVPoolHost):
                suffixes = [f"_{self.mla_suffix}_{PoolName.DRAFT}_k"]
            else:
                suffixes = [
                    f"_{self.mha_suffix}_{PoolName.DRAFT}_k",
                    f"_{self.mha_suffix}_{PoolName.DRAFT}_v",
                ]
        elif pool_name in (
            PoolName.INDEXER,
            PoolName.DEEPSEEK_V4_C4,
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            PoolName.DEEPSEEK_V4_C128,
            PoolName.DEEPSEEK_V4_C4_STATE,
            PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            PoolName.DEEPSEEK_V4_C128_STATE,
        ):
            # DSA indexer 与 DeepSeek V4 side pool 都是「按 page 打包的单对象」池。
            suffixes = [f"_{self.mla_suffix}_{pool_name}"]
        elif pool_name == PoolName.SWA:
            if not self.is_mla_backend and hasattr(host_pool, "v_buffer"):
                # 普通 MHA 的 SWA 与 K/V 池一样，拆成 K、V 两个对象。
                suffixes = [
                    f"_{self.mha_suffix}_{pool_name}_k",
                    f"_{self.mha_suffix}_{pool_name}_v",
                ]
            elif self.is_mla_backend:
                suffixes = [f"_{self.mla_suffix}_{pool_name}"]

        if not suffixes:
            raise ValueError(
                f"Unsupported Mooncake hybrid pool name: {pool_name}, "
                f"host_pool={type(host_pool)}"
            )
        # 每个 page 对应 len(suffixes) 个对象；用嵌套推导按 (page × 后缀) 顺序展开 key。
        key_multiplier = len(suffixes)
        component_keys = [
            f"{page_key}{suffix}" for page_key in page_keys for suffix in suffixes
        ]
        return component_keys, key_multiplier

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        """v2 批量存在性查询：先查 KV 主池命中页数，再对每个 side pool 求交，返回可用前缀。"""
        if self.mem_pool_host.kv_buffer is None:
            # 逻辑锚点：Mooncake 中没有物理 KV 对象，可用前缀完全由所需的 side 对象决定。
            kv_pages = len(keys)
        else:
            kv_pages = self.batch_exists(keys, extra_info)

        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            # 已无可用前缀则提前退出。
            if final_pages == 0:
                break
            component_keys, key_multiplier = self._get_hybrid_page_component_keys(
                keys, transfer
            )
            component_keys = self._tag_keys(component_keys)
            ex = self._batch_exist(component_keys)
            # 一个 page 命中 = 其全部组成对象都存在（组内 AND）。
            if key_multiplier > 0:
                page_exists = [
                    all(
                        r == 1
                        for r in ex[i * key_multiplier : (i + 1) * key_multiplier]
                    )
                    for i in range(kv_pages)
                ]
            else:
                page_exists = [False] * kv_pages
            boundary = 0
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                # ALL_PAGES：从头连续命中的最长前缀（遇到第一个 False 即截断）。
                try:
                    boundary = page_exists.index(False)
                except ValueError:
                    boundary = kv_pages
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                # TRAILING_PAGES：只要末尾 trailing 个 page 命中即可，从大到小找满足的前缀长度。
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        page_exists[i]
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[transfer.name] = boundary
            # 最终可用页数取各 pool 命中边界的最小值（求交）。
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _batch_io_v2(self, transfers: List[PoolTransfer], is_set: bool):
        """统一的 v2 读写路径：每个 PoolTransfer 可展开为每页一或多个存储对象，
        但对外仍按 page 级别汇总结果。is_set 区分写入(set)与读取(get)。"""
        results: dict = {}
        for transfer in transfers:
            host_pool = getattr(self, "registered_pools", {}).get(transfer.name)
            keys = transfer.keys
            page_size = getattr(host_pool, "page_size", 1) or 1
            host_indices = transfer.host_indices
            assert len(keys) > 0
            assert len(keys) == len(host_indices) // page_size

            # 展开成组件级 key，并拿到对应的 (缓冲区指针, 元素字节数) 元数据。
            key_strs, key_multiplier = self._get_hybrid_page_component_keys(
                keys, transfer
            )
            key_strs = self._tag_keys(key_strs)
            ptr_list, element_size_list = host_pool.get_page_buffer_meta(host_indices)
            # DeepSeek V4 C4 布局：一个对象由多段缓冲拼成，需打包成 multi-buffer 结构。
            if transfer.name == PoolName.DEEPSEEK_V4_C4:
                ptr_list, element_size_list = self._pack_multi_buffer_meta(
                    key_strs, ptr_list, element_size_list
                )

            if is_set:
                # 写入前先查存在性：已存在的对象记为成功(0)、跳过，只写缺失(-1)的部分。
                exist_result = self._batch_exist(key_strs)
                io_results = [0 if state == 1 else -1 for state in exist_result]
                missing_idx = [i for i, state in enumerate(exist_result) if state != 1]
                if missing_idx:
                    put_results = self._put_batch_zero_copy_impl(
                        [key_strs[i] for i in missing_idx],
                        [ptr_list[i] for i in missing_idx],
                        [element_size_list[i] for i in missing_idx],
                    )
                    for i, res in zip(missing_idx, put_results):
                        io_results[i] = res
            else:
                io_results = self._get_batch_zero_copy_impl(
                    key_strs, ptr_list, element_size_list
                )
            # 把组件级结果聚合回 page 级布尔结果。
            results[transfer.name] = self._batch_postprocess(
                io_results, is_set_operate=is_set, key_multiplier=key_multiplier
            )
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict:
        """v2 批量读取（L3->host 零拷贝）。"""
        return self._batch_io_v2(transfers, is_set=False)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict:
        """v2 批量写入（host->L3 零拷贝，已存在的对象自动跳过）。"""
        return self._batch_io_v2(transfers, is_set=True)

    def _get_mha_split_heads_buffer_meta(self, keys, indices):
        """拆头 MHA：每个 page 按 split_factor 个逻辑 rank 展开，各产生 K、V 两个对象。"""
        ptr_list, element_size_list = (
            self.mem_pool_host.get_split_heads_page_buffer_meta(
                indices, self.split_factor
            )
        )
        key_list = []
        for key_ in keys:
            for suffix in self.mha_suffix:
                key_list.append(f"{key_}_{suffix}_k")
                key_list.append(f"{key_}_{suffix}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    @staticmethod
    def _uses_multi_buffer(buffer_ptrs: List[Any]) -> bool:
        """判断是否为 multi-buffer 结构：每个对象由多段缓冲组成（首元素是序列）。"""
        return bool(buffer_ptrs) and isinstance(buffer_ptrs[0], Sequence)

    @staticmethod
    def _pack_multi_buffer_meta(
        key_strs: List[str],
        ptr_list: List[int],
        element_size_list: List[int],
    ) -> Tuple[List[Any], List[Any]]:
        """当每个 key 对应多段缓冲时，把扁平的 ptr/size 列表按 key 分组成嵌套列表。"""
        # 一一对应（每 key 单段缓冲）时无需打包，直接返回。
        if len(ptr_list) == len(key_strs):
            return ptr_list, element_size_list

        assert len(key_strs) > 0
        assert len(ptr_list) == len(element_size_list)
        assert len(ptr_list) % len(key_strs) == 0

        # nbuf：每个 key 的缓冲段数；按 nbuf 切分成 [[seg0, seg1, ...], ...]。
        nbuf = len(ptr_list) // len(key_strs)
        return [ptr_list[i : i + nbuf] for i in range(0, len(ptr_list), nbuf)], [
            element_size_list[i : i + nbuf]
            for i in range(0, len(element_size_list), nbuf)
        ]

    def _get_mha_buffer_meta(self, keys, indices):
        """普通 MHA：每个 page 产生 K、V 两个对象；仅 page-first 布局能一一对应。"""
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mha_suffix}_k")
            key_list.append(f"{key_}_{self.mha_suffix}_v")
        # layer_first 布局会产生 multi-buffer（数量不匹配），MHA 不支持，需用 page_first。
        if len(key_list) != len(ptr_list):
            raise RuntimeError(
                "Mooncake layer_first multi-buffer is only supported for MLA "
                "host KV pool. Use page_first/page_first_direct for MHA."
            )
        return key_list, ptr_list, element_size_list

    def _get_mla_buffer_meta(self, keys, indices):
        """MLA：KV 在 TP 间共享，每个 page 仅一个对象（可能是多段缓冲，需打包）。"""
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mla_suffix}_k")
        ptr_list, element_size_list = self._pack_multi_buffer_meta(
            key_list, ptr_list, element_size_list
        )
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _batch_preprocess(self, keys, host_indices):
        """v1 路径预处理：按 MLA / 拆头 MHA / 普通 MHA 选择对应的 key 与缓冲元数据构造方式。"""
        assert len(keys) > 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size
        if self.is_mla_backend:
            return self._get_mla_buffer_meta(keys, host_indices)
        else:
            if self.storage_config.should_split_heads:
                return self._get_mha_split_heads_buffer_meta(keys, host_indices)
            else:
                return self._get_mha_buffer_meta(keys, host_indices)

    def _batch_postprocess(
        self, results: List[int], is_set_operate=False, key_multiplier=None
    ):
        """把组件级 I/O 结果按 key_multiplier 分组，聚合成 page 级布尔成功标记。

        参考 https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-store/include/pybind_client.h
        batch_get_into：每个元素为成功读取的字节数（>0），失败为负值；
        batch_put_from：每个元素成功为 0，失败为负值。
        """
        # 未显式给出时按后端类型推断每 page 的对象数：MLA=1；MHA=2（拆头再乘 split_factor）。
        if key_multiplier is None:
            if self.is_mla_backend:
                key_multiplier = 1
            else:
                key_multiplier = 2
                if self.storage_config.should_split_heads:
                    key_multiplier *= self.split_factor

        # 每 key_multiplier 个结果为一组（对应一个逻辑 page），组内全成功才算该 page 成功。
        result_groups = [
            results[i : i + key_multiplier]
            for i in range(0, len(results), key_multiplier)
        ]
        return [
            (
                all(res == 0 for res in group)  # set：全为 0 视为成功
                if is_set_operate
                else all(res > 0 for res in group)  # get：全 > 0（读到字节）视为成功
            )
            for group in result_groups
        ]

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """v1 批量读取（KV 主池）：零拷贝把 L3 数据读入 host_indices 指向的缓冲区。"""
        if self.mem_pool_host.kv_buffer is None:
            # DeepSeek V4 的 KV 锚点是逻辑的、无物理数据；实际数据由 v2 side pool 承载。
            return [True] * len(keys)

        # 若配置了 extra_backend_tag，先给 key 加前缀。
        keys = self._tag_keys(keys)

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)

        start_time = time.perf_counter()
        get_results = self._get_batch_zero_copy_impl(
            key_strs, buffer_ptrs, buffer_sizes
        )
        end_time = time.perf_counter()

        # 记录预取页数与带宽（GB/s）指标。
        if self.enable_storage_metrics:
            self.prefetch_pgs.append(len(keys))
            self.prefetch_bandwidth.append(
                len(keys) / (end_time - start_time) * self.gb_per_page
            )

        return self._batch_postprocess(get_results, is_set_operate=False)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """v1 批量写入（KV 主池）：仅写入 L3 中尚不存在的对象，已存在的跳过。"""
        if self.mem_pool_host.kv_buffer is None:
            # DeepSeek V4 的 KV 锚点是逻辑的、无物理数据；实际数据由 v2 side pool 承载。
            return [True] * len(keys)

        # 若配置了 extra_backend_tag，先给 key 加前缀。
        keys = self._tag_keys(keys)

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)
        exist_result = self._batch_exist(key_strs)

        # 挑出尚不存在的 key 待写入；已存在的直接记为成功(0)。
        set_keys = []
        set_buffer_ptrs = []
        set_buffer_sizes = []
        set_indices = []
        set_results = [-1] * len(key_strs)
        for i in range(len(key_strs)):
            if exist_result[i] != 1:
                set_keys.append(key_strs[i])
                set_buffer_ptrs.append(buffer_ptrs[i])
                set_buffer_sizes.append(buffer_sizes[i])
                set_indices.append(i)
            else:
                set_results[i] = 0

        # 仅把不存在的 key 写入存储。
        if len(set_keys) > 0:
            start_time = time.perf_counter()
            put_results = self._put_batch_zero_copy_impl(
                set_keys, set_buffer_ptrs, set_buffer_sizes
            )
            end_time = time.perf_counter()

            # 记录备份页数与带宽（GB/s）指标。
            if self.enable_storage_metrics:
                self.backup_pgs.append(len(set_keys))
                self.backup_bandwidth.append(
                    len(set_keys) / (end_time - start_time) * self.gb_per_page
                )

            # 把实际写入结果回填到对应位置。
            for i in range(len(set_indices)):
                set_results[set_indices[i]] = put_results[i]

        return self._batch_postprocess(set_results, is_set_operate=True)

    def set(
        self,
        key,
        value: Optional[Any] = None,
        target_location: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> bool:
        """单 key 零拷贝写入（已存在则视为成功直接返回）。"""
        # 目前仅支持零拷贝写入，location/sizes 必填。
        assert target_location is not None and target_sizes is not None
        exist_result = self._batch_exist([key])
        if exist_result[0] == 1:
            return True
        put_result = self._put_batch_zero_copy_impl(
            [key], [target_location], [target_sizes]
        )
        return put_result[0] == 0

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[torch.Tensor]] = None,
        target_locations: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> bool:
        """批量零拷贝写入：跳过已存在的 key，仅写入缺失的部分。"""
        # 目前仅支持零拷贝写入，location/sizes 必填且与 keys 一一对应。
        assert target_locations is not None and target_sizes is not None
        assert len(keys) == len(target_locations) == len(target_sizes)

        if len(keys) == 0:
            return False

        # 任一参数为 None 直接失败。
        for i in range(len(keys)):
            if (
                keys[i] is None
                or target_locations[i] is None
                or target_sizes[i] is None
            ):
                return False

        # 挑出尚不存在的 key 待写入。
        exist_result = self._batch_exist(keys)
        set_keys = []
        set_target_locations = []
        set_target_sizes = []
        set_indices = []
        for i in range(len(keys)):
            if exist_result[i] != 1:
                set_keys.append(keys[i])
                set_target_locations.append(target_locations[i])
                set_target_sizes.append(target_sizes[i])
                set_indices.append(i)
        # 仅把不存在的 key 写入存储。
        start_time = time.perf_counter()
        put_result = self._put_batch_zero_copy_impl(
            set_keys, set_target_locations, set_target_sizes
        )
        end_time = time.perf_counter()

        if self.enable_storage_metrics:
            self.backup_pgs.append(len(set_keys))
            self.backup_bandwidth.append(
                len(set_keys) / (end_time - start_time) * self.gb_per_page
            )

        # 把本次成功写入的位置也标记为已存在，便于统一计算连续成功数。
        for i in range(len(set_indices)):
            if put_result[i] == 0:
                exist_result[set_indices[i]] = 1

        # 统计从头开始连续成功的数量。
        success_count = 0
        for i in range(len(keys)):
            if exist_result[i] == 0:
                break
            success_count += 1
        # TODO: 直接返回从头开始连续成功的操作数量。
        return success_count == len(keys)

    def get(
        self,
        key,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """单 key 零拷贝读取。"""
        assert target_location is not None and target_sizes is not None
        get_result = self._get_batch_zero_copy_impl(
            [key], [target_location], [target_sizes]
        )
        return get_result[0] >= 0

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> int:
        """批量零拷贝读取，返回从头开始连续成功的 page 数。"""
        assert len(keys) == len(target_locations) == len(target_sizes)
        if len(keys) == 0:
            return 0

        start_time = time.perf_counter()
        get_result = self._get_batch_zero_copy_impl(
            keys, target_locations, target_sizes
        )
        end_time = time.perf_counter()

        # MLA 每 page 1 个对象；MHA 每 page 2 个对象（K/V）。
        if self.is_mla_backend:
            key_multiplier = 1
        else:
            key_multiplier = 2

        if self.enable_storage_metrics:
            self.prefetch_pgs.append(len(keys))
            self.prefetch_bandwidth.append(
                len(keys) / (end_time - start_time) * self.gb_per_page
            )

        # 一旦遇到失败的对象，返回其之前已完整命中的 page 数。
        for i in range(len(keys)):
            if get_result[i] < 0:
                return i // key_multiplier
        return len(keys) // key_multiplier

    def exists(self, key) -> bool:
        """判断单个 key 是否存在。"""
        exist_result = self._batch_exist([key])
        return exist_result[0] == 1

    def batch_exists(
        self, keys, extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """批量存在性查询，返回从头开始连续存在的 page 数。"""
        # 若配置了 extra_backend_tag，先给 key 加前缀。
        keys = self._tag_keys(keys)

        # 按后端类型拼出实际查询 key，并确定每 page 的对象数 key_multiplier。
        if self.is_mla_backend:
            query_keys = [f"{key}_{self.mla_suffix}_k" for key in keys]
            key_multiplier = 1
        else:
            query_keys = []
            if self.storage_config.should_split_heads:
                # 拆头 MHA：每 page 拆成 split_factor 个 rank，各含 K、V。
                for key in keys:
                    for suffix in self.mha_suffix:
                        query_keys.append(f"{key}_{suffix}_k")
                        query_keys.append(f"{key}_{suffix}_v")
                key_multiplier = 2 * self.split_factor
            else:
                for key in keys:
                    query_keys.append(f"{key}_{self.mha_suffix}_k")
                    query_keys.append(f"{key}_{self.mha_suffix}_v")
                key_multiplier = 2

        # 遇到第一个缺失对象时，返回之前完整命中的 page 数。
        exist_result = self._batch_exist(query_keys)
        for i in range(len(query_keys)):
            if exist_result[i] != 1:
                return i // key_multiplier
        return len(query_keys) // key_multiplier

    def close(self):
        # MooncakeDistributedStore 析构时会自动清理，无需手动 close。
        pass

    def clear(self) -> None:
        """清空存储中的所有对象。"""
        self.store.remove_all()

    def _put_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[Any], buffer_sizes: List[Any]
    ) -> List[int]:
        """零拷贝批量写入底层实现：按是否 multi-buffer 选择对应的 Mooncake 接口。"""
        if self._uses_multi_buffer(buffer_ptrs):
            return self.store.batch_put_from_multi_buffers(
                key_strs, buffer_ptrs, buffer_sizes
            )
        return self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)

    def _get_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[Any], buffer_sizes: List[Any]
    ) -> List[int]:
        """零拷贝批量读取底层实现：按是否 multi-buffer 选择对应的 Mooncake 接口。"""
        if self._uses_multi_buffer(buffer_ptrs):
            return self.store.batch_get_into_multi_buffers(
                key_strs, buffer_ptrs, buffer_sizes
            )
        return self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)

    def _batch_exist(self, key_strs: List[str]) -> List[int]:
        """批量存在性查询底层实现（返回每个 key 的存在标记，1 表示存在）。"""
        return self.store.batch_is_exist(key_strs)

    def get_stats(self):
        """导出并清空累积的预取/备份指标（页数与带宽采样）。"""
        storage_metrics = StorageMetrics()
        storage_metrics.prefetch_pgs.extend(self.prefetch_pgs)
        storage_metrics.backup_pgs.extend(self.backup_pgs)
        storage_metrics.prefetch_bandwidth.extend(self.prefetch_bandwidth)
        storage_metrics.backup_bandwidth.extend(self.backup_bandwidth)
        self.prefetch_pgs.clear()
        self.backup_pgs.clear()
        self.prefetch_bandwidth.clear()
        self.backup_bandwidth.clear()
        return storage_metrics
