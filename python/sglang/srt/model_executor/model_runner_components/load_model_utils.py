from __future__ import annotations

import datetime
import logging
import os
import socket
import threading
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch
import torch.distributed as dist

from sglang.srt.arg_groups.overrides import (
    modelexpress_transport_of,
    modelexpress_url_of,
)
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.model_loader.loader import get_model_loader
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    trigger_init_weights_send_group_for_remote_instance_request,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_model,
    get_observability,
)
from sglang.srt.utils.common import is_npu
from sglang.srt.utils.network import NetworkAddress

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

# 中译：=====================================================================================
# 本文件是「权重加载编排层」：`ModelRunner.load_model()` 里真正读 checkpoint 的那一段都在这儿。
# 它把「选哪个 loader / 是否把加载包进 memory saver 的 weights region / 加载前后的显存与并行状态处理」
# 拆成一堆小函数，真正的主链路只有三个：
#   · build_load_config(...)             —— 打包 LoadConfig（loader 靠它决定从哪读、怎么读）
#   · maybe_enable_ipc_weight_cache(...) —— 可选：把 load_format 切到 IPC 零拷贝权重缓存
#   · load_model_with_memory_saver(...)  —— 真正加载：在 GPU_MEMORY_TYPE_WEIGHTS region 里读权重
# 其余是旁路辅助：dtype 回退、KV scale 加载、滑窗大小解析、在线量化上报、张量 dump hook、
# 远程实例 NCCL 发送组、跨 rank barrier 等。
# 调用顺序与耗时口径见 docs/learn/run/init_tp_model_worker_zh.md 第 5 节。
# =====================================================================================

logger = logging.getLogger(__name__)

_is_npu = is_npu()


# 中译：跨 rank 权重加载对齐（monitored_barrier）的超时时间：480 秒。
#       比纯读盘时间给得宽，因为读完权重后有的 rank 还要做额外处理（kernel 预编译、KV scale 加载、
#       在线量化上报等），不能按「谁读得慢」直接定超时。
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # leave more time for post data processing


def maybe_precompile_model_kernels_after_loading(model, device: str) -> None:
    """加载权重后按需预编译模型 kernel（模型未实现该钩子则直接跳过）。

    【中译】模型可以自定义 `precompile_kernels_after_loading()`（例如需要 JIT/AOT 生成
           kernel 的模型）。若该钩子存在，则：CUDA 上先 synchronize + empty_cache 把加载阶段
           留下的碎片/缓存清干净，再执行预编译，之后再来一次 synchronize + empty_cache——
           这样预编译产生的临时显存不会污染后续「按剩余显存给 KV 池定尺寸」的统计。
    【输入示例】model=<已加载权重的 LlamaForCausalLM>, device="cuda"
    【输出】无返回值；副作用是可能已完成 kernel 预编译、并把显存缓存清空（仅 CUDA 分支）；
           非 CUDA（npu/cpu）只执行 precompile()，不做同步与清缓存。
    """
    precompile = getattr(model, "precompile_kernels_after_loading", None)
    if precompile is None:
        return

    if device == "cuda":
        current_platform.synchronize()
        current_platform.empty_cache()
    precompile()
    if device == "cuda":
        current_platform.synchronize()
        current_platform.empty_cache()


class LoadedModel(msgspec.Struct, frozen=True, kw_only=True):
    """`load_model_with_memory_saver()` 的返回值：一次权重加载的全部产物（不可变结构体）。

    【中译】用 `msgspec.Struct` 而不是 dataclass：字段存取更快、内存更省。
           `frozen=True`  → 实例不可变（加载结果不该被中途改写）；
           `kw_only=True` → 只能用关键字传参（避免位置参数错位——这几个字段含义差异很大）。

    【字段】
        loader                      —— 实际执行加载的 loader（由 load_format 决定是哪个实现），
                                       后续赋给 `model_runner.loader`（导出/重载权重时会用到）。
        model                       —— 加载好的 nn.Module（PP 下只包含本 stage 的层）。
        remote_instance_weight_info —— 远程实例权重加载时由 loader 带出的元信息；非该模式为 None。
        startup_weight_load         —— 仅 `--is-startup-weight-load-overlap` 时非 None：
                                       权重尚未提交的句柄，之后由 `finalize_startup_weight_load()` 提交。

    【输入示例】
        LoadedModel(
            loader=<DefaultModelLoader>,
            model=<LlamaForCausalLM>,
            remote_instance_weight_info=None,
            startup_weight_load=None,
        )

    【输出示例】（调用方 `ModelRunner.load_model` 的取用方式）
        self.loader = loaded.loader
        self.model = loaded.model
        self.startup_weight_load = loaded.startup_weight_load
    """

    loader: Any
    model: Any
    remote_instance_weight_info: Optional[Any]
    startup_weight_load: Optional[Any] = None


def maybe_downgrade_dtype_for_legacy_gpu(*, model_config: ModelConfig) -> None:
    """老旧 GPU 上把模型 dtype 回退到 float16（sm80 以下没有 bfloat16 支持）。

    【中译】判定依据是设备的 compute capability：
           · major < 8（sm80 之前）→ 没有 bfloat16 张量核心，强制 float16；
           · 若 minor < 5（低于 sm75）→ 直接报错，SGLang 不支持。
           由于这是「设备驱动」的结论，进程内所有 runner 的判定必然一致，所以除了改
           `model_config.dtype`（每个 runner 自己那份事实来源），还往 runtime context 写一条
           override 记录，便于日志/观测看出「发生过回退」。

    【输入示例】model_config=ModelConfig(model_path="meta-llama/Llama-2-7b", dtype=torch.bfloat16)
    【输出】无返回值；副作用：把 model_config.dtype 就地改成 torch.float16
           （低于 sm75 时抛 RuntimeError: SGLang only supports sm75 and above.）。
    """
    if torch.cuda.get_device_capability()[0] < 8:
        logger.info(
            "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
        )

        # Device-driven, so every runner in the process resolves the same way;
        # the per-runner truth is model_config.dtype, this is the record.
        # 中译：这是「设备驱动」的结论，所以同进程内每个 runner 解析结果必然相同；
        #       每个 runner 真正生效的是 model_config.dtype，这里写的是一份「记录」。
        get_context().override("ModelRunner._sm80_dtype_fallback", dtype="float16")
        model_config.dtype = torch.float16
        if torch.cuda.get_device_capability()[1] < 5:
            raise RuntimeError("SGLang only supports sm75 and above.")


def maybe_trigger_remote_instance_nccl_send_group(
    *, tp_rank: int, load_format: Optional[str] = None
) -> None:
    """``load_format`` is this runner's effective format: a draft loading under
    ``--speculative-draft-draft-load-format`` needs its own send group, and the
    target's format cannot answer for it.

    【中译】`load_format` 是本 runner「实际生效」的格式：draft 用
           `--speculative-draft-load-format` 加载时它需要自己的发送组，target 的格式不能代表它
           （所以这个参数不能省，也不能从全局取）。

    【行为】仅当（本 runner 的 load_format 或全局 `model.load_format`）为 REMOTE_INSTANCE
           且后端为 NCCL 时：由 tp_rank 0 起一个后台线程，向 seed 实例发起
           「初始化权重发送组」请求，请求里带上本机 IP 与各 rank 的发送端口。

    【输入示例】tp_rank=0, load_format="remote_instance"
    【输出】无返回值；副作用是后台线程已启动（不 join、不等结果，后续握手由加载器侧完成）。
    """
    if (
        load_format or get_model().load_format
    ) == LoadFormat.REMOTE_INSTANCE and get_model().remote_instance_weight_loader_backend == RemoteInstanceWeightLoaderBackend.NCCL:
        if tp_rank == 0:
            instance_ip = NetworkAddress.resolve_host(socket.gethostname())
            t = threading.Thread(
                target=trigger_init_weights_send_group_for_remote_instance_request,
                args=(
                    get_model().remote_instance_weight_loader_seed_instance_ip,
                    get_model().remote_instance_weight_loader_seed_instance_service_port,
                    get_model().remote_instance_weight_loader_send_weights_group_ports,
                    instance_ip,
                ),
            )
            t.start()


def load_kv_cache_scales(*, model, kv_cache_dtype: str) -> None:
    """``kv_cache_dtype`` is the caller's resolved value. Required rather than
    defaulted: a fallback to ``server_args`` would be a hidden global read for
    any future caller that forgets to pass one.

    【中译】`kv_cache_dtype` 由调用方解析后显式传入，而且**故意不给默认值**：
           若默认回退去读 `server_args`，未来某个忘了传参的调用方就会静默拿到全局值，
           这种「隐藏的全局读取」是刻意避免的。

    【行为】kv_cache_dtype == "fp8_e4m3" 时：
           · 指定了 `--quantization-param-path` → 调 `model.load_kv_cache_scales(path)`
             把 FP8 的 scale 读进来；模型未实现该方法则直接抛 RuntimeError；
           · 未指定 → 打 warning：缩放因子按 1.0 处理，精度可能下降。
           非 fp8_e4m3 时什么都不做。

    【输入示例】model=<LlamaForCausalLM>, kv_cache_dtype="fp8_e4m3"
    【输出】无返回值；副作用是把 scale 写进模型的 KV 相关参数（或仅打印告警）。
    """
    if kv_cache_dtype == "fp8_e4m3":
        if get_model().quantization_param_path is not None:
            if callable(getattr(model, "load_kv_cache_scales", None)):
                model.load_kv_cache_scales(get_model().quantization_param_path)
                logger.info(
                    "Loaded KV cache scaling factors from %s",
                    get_model().quantization_param_path,
                )
            else:
                raise RuntimeError(
                    "Using FP8 KV cache and scaling factors provided but "
                    "model %s does not support loading scaling factors.",
                    model.__class__,
                )
        else:
            logger.warning(
                "Using FP8 KV cache but no scaling factors "
                "provided. Defaulting to scaling factors of 1.0. "
                "This may lead to less accurate results!"
            )


def resolve_sliding_window_size(model, model_config: ModelConfig) -> Optional[int]:
    """解析本模型实际使用的滑动窗口大小（sliding window size）。

    【中译】按优先级三级回退：
           1) 模型自己实现了 `get_attention_sliding_window_size()` → 用模型给的（最权威）；
           2) 混合滑窗模型（is_hybrid_swa）且 config 里配了 sliding_window_size → 用它；
              （各模型的 sliding_window 字段含义可能不同，如 dllm，这里只认 SWA 场景）
           3) 配了 attention_chunk_size → 用 chunk size 顶上，并打一条日志说明。

    【输入示例】model=<支持滑窗的 Qwen 系列模型>,
               model_config=ModelConfig(is_hybrid_swa=True, sliding_window_size=4096)
    【输出示例】4096   # 命中分支 2
               131072 # 命中分支 3（attention_chunk_size 顶替）
               None   # 三条都不命中：该模型不使用滑窗
    """
    # Parse other args
    sliding_window_size = None
    if hasattr(model, "get_attention_sliding_window_size"):
        sliding_window_size = model.get_attention_sliding_window_size()
    elif model_config.is_hybrid_swa and model_config.sliding_window_size is not None:
        # sliding window field in model config may have different meaning for different kinds of models (e.g., dllm), here we only consider the sliding window in SWA model
        # 中译：model_config 里的 sliding window 字段在不同模型（如 dllm）上含义可能不同，
        #       这里只处理 SWA 模型下的滑窗语义。
        sliding_window_size = model_config.sliding_window_size
    elif model_config.attention_chunk_size is not None:
        sliding_window_size = model_config.attention_chunk_size
        logger.info(
            f"Setting sliding_window_size to be attention_chunk_size: {sliding_window_size}"
        )
    return sliding_window_size


def report_online_quantization(*, model) -> None:
    """在线量化时上报「量化了多少层、都是哪些层类型」（纯日志，不改变任何行为）。

    【中译】只在两个条件同时满足时打印：① 开启了在线量化（`model.quantization` 非空）；
           ② 模型的 `quant_config.quantized_layers` 是二元组 `(层类型, 层数)`。
           其余情况静默返回（并非所有模型都提供该字段）。

    【输入示例】model=<权重已加载、quant_config.quantized_layers=(["linear"], 28) 的模型>
    【输出】无返回值；副作用是打印一行：
           `Online <quant> quantization: quantized 28 layers of types: ["linear"]`
    """
    # TODO: Make sure all models have `quant_config` attribute, and all online quantization methods register which layers they actually quantize.
    # 中译：TODO：需保证所有模型都有 `quant_config` 属性，并且所有在线量化方法都登记了它们
    #       实际量化的层（否则这里的统计会不完整，甚至直接不打印）。
    quantized_layers = getattr(
        getattr(model, "quant_config", None), "quantized_layers", None
    )
    if (
        get_model().quantization is not None
        and isinstance(quantized_layers, tuple)
        and len(quantized_layers) == 2
    ):
        layer_types, quantized_layers_count = quantized_layers
        logger.info(
            f"Online {get_model().quantization} quantization: quantized {quantized_layers_count} layers of types: {layer_types}"
        )


def maybe_register_debug_tensor_dump_hook(
    *,
    model,
    spec_algorithm: SpeculativeAlgorithm,
    is_draft_worker: bool,
    tp_size: int,
    tp_rank: int,
    pp_rank: int,
) -> None:
    """调试用：按需给模型注册 forward hook，把指定层的张量 dump 到磁盘。

    【中译】只有指定了 `--debug-tensor-dump-output-folder` 才注册（默认 None = 不注册，
           对生产零开销）。EAGLE 投机解码时 target 与 draft 各自 dump 到子目录
           `target/`、`draft/`，避免两套张量混在一起；并行信息（tp_size / tp_rank / pp_rank）
           会传进文件名，便于按 rank 归位。

    【输入示例】model=<目标模型>, spec_algorithm=<SpeculativeAlgorithm.EAGLE>,
               is_draft_worker=False, tp_size=8, tp_rank=0, pp_rank=0
    【输出】无返回值；副作用是模型上多了 forward hook（写文件到 dump_folder）。
    """
    if get_observability().debug_tensor_dump_output_folder is not None:
        dump_folder = get_observability().debug_tensor_dump_output_folder
        if spec_algorithm.is_eagle():
            role = "draft" if is_draft_worker else "target"
            dump_folder = os.path.join(dump_folder, role)
        register_forward_hook_for_model(
            model,
            dump_folder,
            get_observability().debug_tensor_dump_layers,
            tp_size,
            tp_rank,
            pp_rank,
        )


def build_load_config(
    *,
    server_args: ServerArgs,
    tp_rank: int,
    load_format: Optional[str] = None,
    remote_instance_weight_transporter_engine: Any,
    remote_instance_weight_transporter_session_id: str,
    draft_model_idx: Optional[int],
    weight_cache_mode: str,
    weight_cache_socket: Optional[str],
) -> LoadConfig:
    """把「从哪儿、怎么读权重」所需的全部信息打包成一个 LoadConfig（loader 的唯一输入）。

    【中译】三类信息的汇聚点：
           1) 来源与格式：load_format（为空则回退全局 `model.load_format`）、download_dir、
              model_loader_extra_config；ModelOpt 量化配置单独抽成 ModelOptConfig；
           2) 远程实例权重加载：seed 实例 ip/port、发送组端口、后端类型、传输引擎与会话 id
              （由调用方传入的 engine / session_id 决定）；
           3) 权重缓存与 ModelExpress：weight_cache_mode / weight_cache_socket、
              modelexpress 的 url 与 transport；
           另外带上 tp_rank 与 draft_model_idx，让 loader 知道「我是哪个 rank、哪份 draft」。

    【输入示例】
        server_args=ServerArgs(model_path="Qwen/Qwen3-8B", tp_size=8),
        tp_rank=0,
        load_format=None,                                  # → 用全局 model.load_format（如 "safetensors"）
        remote_instance_weight_transporter_engine=<engine or None>,
        remote_instance_weight_transporter_session_id="sess-xxx",
        draft_model_idx=None,                              # target 传 None；draft 传 0/1/...
        weight_cache_mode="off",
        weight_cache_socket=None,
    【输出示例】LoadConfig(load_format="safetensors", tp_rank=0, draft_model_idx=None,
                          weight_cache_mode="off", modelexpress_url=None, ...)
    """
    from sglang.srt.configs.modelopt_config import ModelOptConfig

    modelopt_config = ModelOptConfig(
        quant=get_model().modelopt_quant,
        checkpoint_restore_path=get_model().modelopt_checkpoint_restore_path,
        checkpoint_save_path=get_model().modelopt_checkpoint_save_path,
        export_path=get_model().modelopt_export_path,
        quantize_and_serve=get_model().quantize_and_serve,
    )

    return LoadConfig(
        load_format=load_format or get_model().load_format,
        download_dir=get_model().download_dir,
        model_loader_extra_config=get_model().model_loader_extra_config,
        tp_rank=tp_rank,
        remote_instance_weight_loader_seed_instance_ip=get_model().remote_instance_weight_loader_seed_instance_ip,
        remote_instance_weight_loader_seed_instance_service_port=get_model().remote_instance_weight_loader_seed_instance_service_port,
        remote_instance_weight_loader_send_weights_group_ports=get_model().remote_instance_weight_loader_send_weights_group_ports,
        remote_instance_weight_loader_backend=get_model().remote_instance_weight_loader_backend,
        remote_instance_weight_loader_transfer_engine=remote_instance_weight_transporter_engine,
        remote_instance_weight_loader_transfer_engine_session_id=remote_instance_weight_transporter_session_id,
        modelexpress_url=modelexpress_url_of(server_args),
        modelexpress_transport=modelexpress_transport_of(server_args),
        modelopt_config=modelopt_config,
        rl_quant_profile=get_model().rl_quant_profile,
        draft_model_idx=draft_model_idx,
        weight_cache_mode=weight_cache_mode,
        weight_cache_socket=weight_cache_socket,
    )


def maybe_enable_ipc_weight_cache(
    *,
    load_config: LoadConfig,
    tp_size: int,
    pp_rank: int,
    tp_rank: int,
) -> None:
    """Switch ``load_config`` onto the IPC weight-cache path, in place.

    Overrides the load format to ``IPC_CACHE`` (remembering the original as the
    disk fallback) and derives the per-rank daemon socket if unset. Idempotent:
    the format swap is guarded on ``!= IPC_CACHE`` so a second call (e.g. a
    weight reload) can't overwrite the captured fallback format.

    【中译】就地（in place）把 `load_config` 切到 IPC 权重缓存路径：
           · 把 load_format 改成 `IPC_CACHE`，并把原格式记到 `fallback_load_format`
             （作为「缓存不可用时的磁盘回退格式」）；
           · socket 未设置时按 rank 推导出本 rank 的 daemon socket；
           该函数**幂等**：格式替换有 `!= IPC_CACHE` 守卫，因此第二次调用（如权重重载）
           不会把已经捕获的回退格式覆盖掉。

    【输入示例】load_config=LoadConfig(load_format="safetensors", weight_cache_socket=None),
               tp_size=8, pp_rank=0, tp_rank=3（且 server_args 开了 --weight-cache-mode ipc）
    【输出】无返回值；副作用（就地修改 load_config）：
           load_config.load_format            == "ipc_cache"
           load_config.fallback_load_format   == "safetensors"
           load_config.weight_cache_socket    == get_socket_path(global_rank=3)
           若 --weight-cache-mode off → 直接返回，load_config 原样不动。
    """
    if get_model().weight_cache_mode == "off":
        return

    if load_config.load_format != LoadFormat.IPC_CACHE:
        load_config.fallback_load_format = load_config.load_format
        load_config.load_format = LoadFormat.IPC_CACHE

    # Compute socket path using global rank (tp_size * pp_rank + tp_rank) so
    # each daemon has a unique socket even across PP stages and nodes.
    # 中译：用 global rank（tp_size * pp_rank + tp_rank）算 socket 路径，
    #       保证跨 PP stage、跨节点时每个 daemon 的 socket 也唯一。
    if load_config.weight_cache_socket is None:
        from sglang.srt.weight_cache.protocol import (
            compute_global_rank,
            get_socket_path,
        )

        global_rank = compute_global_rank(tp_size, pp_rank, tp_rank)
        load_config.weight_cache_socket = get_socket_path(global_rank=global_rank)


def load_model_with_memory_saver(
    *,
    server_args: ServerArgs,
    model_config: ModelConfig,
    load_config: LoadConfig,
    device: str,
    gpu_id: int,
    memory_saver_adapter: Any,
    is_draft_worker: bool,
) -> LoadedModel:
    """真正读 checkpoint 的函数：在 `GPU_MEMORY_TYPE_WEIGHTS` region 里完成权重加载。

    【中译】它只干三件事：
           1) 进 region 前：临时 monkey patch vllm 的并行状态（量化层还要用）、
              决定 `enable_cpu_backup`、以及在 IPC 零拷贝模式下强制关掉 CPU 备份；
           2) 在 `memory_saver_adapter.region(GPU_MEMORY_TYPE_WEIGHTS, ...)` 内：
              选 loader → 造 DeviceConfig → （overlap 模式走 prepare，否则直接 load_model）；
              这段期间分配的权重显存都被打上 weights 标签，之后才能被 pause/resume 回收；
           3) 出 region 后：NPU 上清一下缓存、把 monkey patch 还原。

    【输入示例】
        server_args=ServerArgs(model_path="Qwen/Qwen3-8B", load_format="safetensors"),
        model_config=<ModelConfig>,
        load_config=<build_load_config(...) 的结果>,
        device="cuda", gpu_id=0,
        memory_saver_adapter=<ModelRunner.memory_saver_adapter>,
        is_draft_worker=False,
    【输出示例】
        LoadedModel(loader=<DefaultModelLoader>, model=<LlamaForCausalLM>,
                    remote_instance_weight_info=None, startup_weight_load=None)
        耗时口径：单次 TP=8 的 8B 模型约数十秒~分钟级，这就是 init_tp_model_worker 的主体耗时。
    """
    # Remove monkey_patch when linear.py quant remove dependencies with vllm
    # 中译：临时给 vllm 的并行状态打 monkey patch（部分量化实现还依赖 vllm 的
    #       linear 层）——待量化层彻底去除对 vllm 的依赖后这行就该删掉。
    monkey_patch_vllm_parallel_state()

    # 中译：是否需要把权重备份到 CPU（决定 pause 时是「D2H 备份后释放」还是「直接释放」）：
    #       target 看 --enable-weights-cpu-backup；draft 另有 --enable-draft-weights-cpu-backup。
    enable_cpu_backup = get_exec().features.enable_weights_cpu_backup or (
        is_draft_worker and get_exec().features.enable_draft_weights_cpu_backup
    )

    # In zero-copy IPC mode, the weights are shared with the daemon via
    # CUDA IPC and must not be offloaded/reloaded by the memory saver.
    # 中译：IPC 零拷贝模式下权重通过 CUDA IPC 与 daemon 共享，不能被 memory saver
    #       释放/重载，所以这里强制关掉 CPU 备份。
    is_ipc_zero_copy = get_model().weight_cache_mode != "off"
    if is_ipc_zero_copy and enable_cpu_backup:
        logger.warning(
            "[ModelRunner] Disabling weights CPU backup in zero-copy IPC mode — "
            "IPC-mapped weights cannot be offloaded to CPU."
        )
        enable_cpu_backup = False

    remote_instance_weight_info = None
    startup_weight_load = None
    # 中译：关键的 memory saver region：这段区间内的显存分配都会被登记到
    #       GPU_MEMORY_TYPE_WEIGHTS 标签下（供 release/resume_memory_occupation 回收）。
    #       所以下面所有「读权重」的动作都必须在 with 里面。
    with memory_saver_adapter.region(
        GPU_MEMORY_TYPE_WEIGHTS,
        enable_cpu_backup=enable_cpu_backup,
    ):
        loader = get_model_loader(
            load_config=load_config,
            model_config=model_config,
        )
        device_config = DeviceConfig(device, gpu_id)
        # 中译：两条加载路径：
        #       · overlap 模式（--is-startup-weight-load-overlap）：这里不真读权重，而是建
        #         StartupWeightLoadManager 并 prepare()，返回一个「带 sentinel 的模型」；真正的
        #         权重提交推迟到 init_model_worker 里的 finalize_startup_weight_load()，
        #         以便与 KV 池分配 / CUDA graph capture 重叠。
        #       · 常规模式：直接 loader.load_model(...)，这里就是耗时主体。
        if server_args.is_startup_weight_load_overlap:
            from sglang.srt.model_executor.model_runner_components.startup_weight_load import (
                StartupWeightLoadManager,
            )

            startup_weight_load = StartupWeightLoadManager.create_from_published_config(
                loader=loader,
                model_config=model_config,
                load_config=load_config,
                device_config=device_config,
                is_draft_worker=is_draft_worker,
            )
            model = startup_weight_load.prepare()
        else:
            model = loader.load_model(
                model_config=model_config,
                device_config=device_config,
            )
        if hasattr(loader, "remote_instance_transfer_engine_weight_info"):
            # 中译：远程实例权重加载时，loader 会把「权重元信息」挂在自身属性上，
            #       这里取出来一并回传给调用方（后续用于权重同步/对账）。
            remote_instance_weight_info = (
                loader.remote_instance_transfer_engine_weight_info
            )
    # Cache needs to be cleared after loading model weights (in the loader.load_model function).
    # To avoid conflict with memory_saver_adapter.region, empty_cache operation is now moved here.
    # 中译：权重加载后本应清一次缓存，但为保证与 memory_saver_adapter.region 不冲突，
    #       empty_cache 从 region 内部移到了 region 之后（这里）。
    if _is_npu:
        torch.npu.empty_cache()
    # 中译：还原刚才的 monkey patch（无论加载成功与否走到这里都还原）。
    monkey_patch_vllm_parallel_state(reverse=True)

    return LoadedModel(
        loader=loader,
        model=model,
        remote_instance_weight_info=remote_instance_weight_info,
        startup_weight_load=startup_weight_load,
    )


def dist_barrier_after_load(
    *,
    elastic_ep_backend: Optional[str],
    tp_rank: int,
    is_ep_joiner: bool = False,
) -> None:
    """加载完权重后的跨 rank 对齐：保证「没人提前开跑」。

    【中译】两条分支：
           · elastic_ep_backend == "mooncake"：Mooncake 不支持 `monitored_barrier`，
             所以用普通 `dist.barrier`（弹性 EP 的 joiner 不参与）；
           · 其他情况：用 `dist.monitored_barrier` + 480s 超时 + wait_all_ranks——
             能检测出「某些 rank 没加载完」。若超时/报错，则抛 ValueError，提示可能是
             OOM 类意外失败或某个节点太慢（比直接挂死更容易定位）。

    【输入示例】elastic_ep_backend=None, tp_rank=3, is_ep_joiner=False
    【输出】无返回值；副作用是阻塞到全体 rank 加载完成（否则抛异常）。
    """
    if elastic_ep_backend == "mooncake":
        # Mooncake does not support `monitored_barrier`
        # 中译：Mooncake 不支持 `monitored_barrier`，只能用普通 barrier。
        if not is_ep_joiner:
            dist.barrier(group=get_tp_group().cpu_group)
    else:
        # Handle the case where some ranks do not finish loading.
        # 中译：处理「有 rank 没加载完」的情况：带超时的 monitored_barrier。
        try:
            dist.monitored_barrier(
                group=get_tp_group().cpu_group,
                timeout=datetime.timedelta(seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S),
                wait_all_ranks=True,
            )
        except RuntimeError:
            raise ValueError(
                f"TP rank {tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
            ) from None
