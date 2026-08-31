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
from sglang.srt.runtime_context import (
    get_disagg,
    get_parallel,
    get_serving,
)


    # 中译：仅在 Prefill 端启动 KV bootstrap server。
def start_disagg_service():
    # Start kv bootstrap server on prefill
    disagg_mode = DisaggregationMode(get_disagg().disaggregation_mode)
    transfer_backend = TransferBackend(get_disagg().disaggregation_transfer_backend)

    if disagg_mode == DisaggregationMode.PREFILL:
        # only start bootstrap server on prefill tm
        # 中译：只有 Prefill 侧的 TokenizerManager(tm) 才启动 bootstrap server。
        # 中译：根据传输后端选出对应的 bootstrap server 类（不同后端实现不同）。
        kv_bootstrap_server_class = get_kv_class(
            transfer_backend, KVClassType.BOOTSTRAP_SERVER
        )
        bootstrap_server = kv_bootstrap_server_class(
            host=get_serving().host,
            port=get_disagg().disaggregation_bootstrap_port,
        )
        # 中译：仅当为 0 号节点且使用 Ascend 后端时，需要额外创建 memfabric 配置存储。
                # 中译：从环境变量读取 Ascend memfabric 存储地址并初始化配置存储。
        maybe_create_ascend_config_store(transfer_backend=transfer_backend)

        return bootstrap_server


def maybe_create_ascend_config_store(transfer_backend: TransferBackend) -> None:
    """Also called directly by the rust-server scheduler: there the KV
    bootstrap registry is served by the embedded rust server's api listener
    (one rust implementation covers every transfer backend — their
    bootstrap-server subclasses are all plain ``CommonKVBootstrapServer``,
    which the rust registry ports verbatim), leaving this store as the only
    ``start_disagg_service`` duty left to perform."""
    if not (
        get_parallel().node_rank == 0 and transfer_backend == TransferBackend.ASCEND
    ):
        return
    try:
        from memfabric_hybrid import create_config_store

        ascend_url = os.getenv("ASCEND_MF_STORE_URL")
        create_config_store(ascend_url)
    except Exception as e:
        raise RuntimeError(
            f"Failed create mf store, invalid ascend_url. With exception {e}"
        )
