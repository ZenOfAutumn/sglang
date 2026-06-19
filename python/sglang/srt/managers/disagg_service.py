"""Start bootstrap/kv-store-related server

中译：在「PD 分离（disaggregation，Prefill/Decode 分离部署）」模式下，
      负责启动与 KV 缓存传输相关的引导（bootstrap）服务 / KV 存储。
      Prefill 实例需要先启动一个 bootstrap server，供 Decode 实例连接并建立
      KV cache 的跨实例传输通道；不同传输后端（如 Mooncake、NIXL、Ascend）
      使用各自的 bootstrap server 实现，由 get_kv_class 根据后端类型选出。
"""

import os

from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.srt.server_args import ServerArgs


def start_disagg_service(
    server_args: ServerArgs,
):
    """根据 PD 分离配置，按需启动 KV bootstrap 服务并返回其实例。

    作用：解析分离模式与传输后端；仅当本实例为 Prefill 角色时，创建并返回
          对应后端的 bootstrap server（用于 Decode 实例建链与 KV 传输协商）。
    关键参数：
        server_args: 全局服务配置，提供分离模式、传输后端、host、bootstrap 端口、
                     node_rank 等信息。
    返回值：Prefill 角色返回 bootstrap_server 实例；其他角色（Decode/非分离）
            隐式返回 None。
    副作用：绑定网络端口启动服务；在 Ascend 后端且 node_rank==0 时还会创建
            memfabric 配置存储（config store）。
    """
    # Start kv bootstrap server on prefill
    # 中译：仅在 Prefill 端启动 KV bootstrap server。
    disagg_mode = DisaggregationMode(server_args.disaggregation_mode)
    transfer_backend = TransferBackend(server_args.disaggregation_transfer_backend)

    if disagg_mode == DisaggregationMode.PREFILL:
        # only start bootstrap server on prefill tm
        # 中译：只有 Prefill 侧的 TokenizerManager(tm) 才启动 bootstrap server。
        # 中译：根据传输后端选出对应的 bootstrap server 类（不同后端实现不同）。
        kv_bootstrap_server_class = get_kv_class(
            transfer_backend, KVClassType.BOOTSTRAP_SERVER
        )
        bootstrap_server = kv_bootstrap_server_class(
            host=server_args.host,
            port=server_args.disaggregation_bootstrap_port,
        )
        # 中译：仅当为 0 号节点且使用 Ascend 后端时，需要额外创建 memfabric 配置存储。
        is_create_store = (
            server_args.node_rank == 0 and transfer_backend == TransferBackend.ASCEND
        )
        if is_create_store:
            try:
                # 中译：从环境变量读取 Ascend memfabric 存储地址并初始化配置存储。
                from memfabric_hybrid import create_config_store

                ascend_url = os.getenv("ASCEND_MF_STORE_URL")
                create_config_store(ascend_url)
            except Exception as e:
                error_message = f"Failed create mf store, invalid ascend_url."
                error_message += f" With exception {e}"
                raise error_message

        return bootstrap_server
