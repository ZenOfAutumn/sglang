# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StorageBackendFactory:
    """存储后端实例的工厂类，支持动态加载。

    HiCache 的 L3 存储后端（file/nixl/mooncake/hf3fs/aibrix/eic/simm 等）通过本工厂
    统一创建。内置后端以「懒加载」方式注册到 _registry：注册时只记录模块路径与类名，
    真正 import 推迟到创建实例时，从而避免为未使用的后端引入额外依赖。
    除内置后端外，还支持通过配置在运行时动态加载自定义后端（backend_name="dynamic"）。
    """

    # 后端注册表：name -> {loader, module_path, class_name}。类级共享。
    _registry: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _load_backend_class(
        module_path: str, class_name: str, backend_name: str
    ) -> type[HiCacheStorage]:
        """根据模块路径加载并校验后端类（必须继承自 HiCacheStorage）。"""
        try:
            module = importlib.import_module(module_path)
            backend_class = getattr(module, class_name)
            # 强制约束：所有存储后端都必须实现 HiCacheStorage 接口。
            if not issubclass(backend_class, HiCacheStorage):
                raise TypeError(
                    f"Backend class {class_name} must inherit from HiCacheStorage"
                )
            return backend_class
        except ImportError as e:
            raise ImportError(
                f"Failed to import backend '{backend_name}' from '{module_path}': {e}"
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_path}': {e}"
            ) from e

    @classmethod
    def register_backend(cls, name: str, module_path: str, class_name: str) -> None:
        """以懒加载方式注册一个存储后端。

        Args:
            name: 后端标识符
            module_path: 包含后端类的 Python 模块路径
            class_name: 后端类名
        """
        # 同名后端重复注册会覆盖旧的，并打印告警提示。
        if name in cls._registry:
            logger.warning(f"Backend '{name}' is already registered, overwriting")

        def loader() -> type[HiCacheStorage]:
            """懒加载函数：真正被调用时才 import 后端类。"""
            return cls._load_backend_class(module_path, class_name, name)

        cls._registry[name] = {
            "loader": loader,
            "module_path": module_path,
            "class_name": class_name,
        }

    @classmethod
    def create_backend(
        cls,
        backend_name: str,
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Any,
        **kwargs,
    ) -> HiCacheStorage:
        """创建一个存储后端实例。
        Args:
            backend_name: 要创建的后端名称
            storage_config: 存储配置
            mem_pool_host: host 侧内存池对象
            **kwargs: 传递给外部（动态）后端的额外参数
        Returns:
            已初始化的存储后端实例
        Raises:
            ValueError: 后端未注册且无法动态加载
            ImportError: 后端模块无法导入
            Exception: 后端初始化失败
        """
        # 优先走已注册的内置后端：触发懒加载拿到类，再按各后端约定创建实例。
        if backend_name in cls._registry:
            registry_entry = cls._registry[backend_name]
            backend_class = registry_entry["loader"]()
            logger.info(
                f"Creating storage backend '{backend_name}' "
                f"({registry_entry['module_path']}.{registry_entry['class_name']})"
            )
            return cls._create_builtin_backend(
                backend_name, backend_class, storage_config, mem_pool_host
            )

        # 名为 "dynamic" 时，尝试根据 extra_config 动态加载自定义后端。
        if backend_name == "dynamic" and storage_config.extra_config is not None:
            backend_config = storage_config.extra_config
            return cls._create_dynamic_backend(
                backend_config, storage_config, mem_pool_host, **kwargs
            )

        # 既非内置也无法动态加载：报错并列出所有已注册后端便于排查。
        available_backends = list(cls._registry.keys())

        raise ValueError(
            f"Unknown storage backend '{backend_name}'. "
            f"Registered backends: {available_backends}. "
        )

    @classmethod
    def _create_dynamic_backend(
        cls,
        backend_config: Dict[str, Any],
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Any,
        **kwargs,
    ) -> HiCacheStorage:
        """根据配置动态创建一个后端实例。"""
        # 动态后端配置必须包含以下三个字段，缺一不可。
        required_fields = ["backend_name", "module_path", "class_name"]
        for field in required_fields:
            if field not in backend_config:
                raise ValueError(
                    f"Missing required field '{field}' in backend config for 'dynamic' backend"
                )

        backend_name = backend_config["backend_name"]
        module_path = backend_config["module_path"]
        class_name = backend_config["class_name"]

        try:
            # 导入后端类（同样要求继承 HiCacheStorage）。
            backend_class = cls._load_backend_class(
                module_path, class_name, backend_name
            )

            logger.info(
                f"Creating dynamic storage backend '{backend_name}' "
                f"({module_path}.{class_name})"
            )

            # 动态后端统一以 (storage_config, kwargs) 的签名构造实例。
            return backend_class(storage_config, kwargs)
        except Exception as e:
            logger.error(
                f"Failed to create dynamic storage backend '{backend_name}': {e}"
            )
            raise

    @classmethod
    def _create_builtin_backend(
        cls,
        backend_name: str,
        backend_class: type[HiCacheStorage],
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Any,
    ) -> HiCacheStorage:
        """按各内置后端各自的初始化约定创建实例。

        不同后端的构造签名并不统一：file/nixl 只需 storage_config；
        mooncake/aibrix/eic/simm 还需要 mem_pool_host；hf3fs 则需要先根据内存池
        布局算出每 page 字节数与 dtype，再走 from_env_config 从环境配置构造。
        """
        if backend_name == "file":
            return backend_class(storage_config)
        elif backend_name == "nixl":
            return backend_class(storage_config)
        elif backend_name == "mooncake":
            backend = backend_class(storage_config, mem_pool_host)
            return backend
        elif backend_name == "aibrix":
            backend = backend_class(storage_config, mem_pool_host)
            return backend
        elif backend_name == "hf3fs":
            # 根据内存池布局计算每个 page 的字节数（bytes_per_page）。
            if mem_pool_host.layout in ["page_first", "page_first_direct"]:
                # page-first 布局用 K 侧每 token 字节数（K/V 已拆分存放）。
                bytes_per_page = (
                    mem_pool_host.get_ksize_per_token() * mem_pool_host.page_size
                )
            elif mem_pool_host.layout == "layer_first":
                # layer-first 布局用整体每 token 字节数（K/V 合并计）。
                bytes_per_page = (
                    mem_pool_host.get_size_per_token() * mem_pool_host.page_size
                )

            dtype = mem_pool_host.dtype
            return backend_class.from_env_config(bytes_per_page, dtype, storage_config)
        elif backend_name == "eic":
            return backend_class(storage_config, mem_pool_host)
        elif backend_name == "simm":
            return backend_class(storage_config, mem_pool_host)
        else:
            raise ValueError(f"Unknown built-in backend: {backend_name}")


# 注册内置存储后端（此处仅登记模块路径与类名，import 推迟到首次创建时）。
StorageBackendFactory.register_backend(
    "file", "sglang.srt.mem_cache.hicache_storage", "HiCacheFile"
)

StorageBackendFactory.register_backend(
    "nixl",
    "sglang.srt.mem_cache.storage.nixl.hicache_nixl",
    "HiCacheNixl",
)

StorageBackendFactory.register_backend(
    "mooncake",
    "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store",
    "MooncakeStore",
)

StorageBackendFactory.register_backend(
    "hf3fs",
    "sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs",
    "HiCacheHF3FS",
)

StorageBackendFactory.register_backend(
    "aibrix",
    "sglang.srt.mem_cache.storage.aibrix_kvcache.aibrix_kvcache_storage",
    "AibrixKVCacheStorage",
)

StorageBackendFactory.register_backend(
    "eic",
    "sglang.srt.mem_cache.storage.eic.eic_storage",
    "EICStorage",
)

StorageBackendFactory.register_backend(
    "simm",
    "sglang.srt.mem_cache.storage.simm.hicache_simm",
    "HiCacheSiMM",
)
