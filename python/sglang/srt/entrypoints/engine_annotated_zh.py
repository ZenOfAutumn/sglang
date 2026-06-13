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
"""
推理服务器的入口（SRT = SGLang Runtime，即 SGLang 运行时）。

本文件实现了推理引擎的 Python API。

【本文件中文注释版】
本文件是 `engine.py` 的带详细中文注释副本，代码逻辑与原文件保持完全一致，
仅添加注释用于学习理解，请勿在生产中直接引用本副本。

整体架构（Engine 由三大组件组成，进程间通过 ZMQ 做 IPC 通信）：
1. TokenizerManager（主进程）：对请求做分词，并把请求发送给 Scheduler。
2. Scheduler（子进程）：从 TokenizerManager 接收请求，组 batch、调度并前向计算，
   再把输出 token 发给 DetokenizerManager。
3. DetokenizerManager（子进程）：对输出 token 反分词，并把结果回传给 TokenizerManager。
"""

from __future__ import annotations

import asyncio
import atexit
import dataclasses
import logging
import multiprocessing as mp
import os
import random
import signal
import threading
import time
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

# 修复 Python threading 的一个 bug：
# 将 threading 内部的 _register_atexit 替换为空操作，避免在多进程退出时触发异常。
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import torch
import uvloop
import zmq

from sglang.srt.elastic_ep.expert_backup_manager import run_expert_backup_manager
from sglang.srt.entrypoints.engine_info_bootstrap_server import (
    EngineInfoBootstrapServer,
)
from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    DestroyWeightsUpdateGroupReqInput,
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    MultimodalDataInputFormat,
    OpenSessionReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.multi_tokenizer_mixin import MultiTokenizerRouter
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.managers.tokenizer_manager_multiitem_mixin import ScoreResult
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    MultiprocessingSerializer,
    assert_pkg_version,
    configure_logger,
    get_bool_env_var,
    is_cuda,
    kill_process_tree,
    launch_dummy_health_check_server,
    maybe_reindex_device_id,
    numa_utils,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.srt.utils.network import get_zmq_socket, is_port_available
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.watchdog import SubprocessWatchdog
from sglang.version import __version__

# 模块级 logger
logger = logging.getLogger(__name__)
# 使用 uvloop 替换默认事件循环策略，以获得更高的异步性能。
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# 缓存当前是否为 CUDA 环境，避免重复检测。
_is_cuda = is_cuda()


@dataclasses.dataclass
class SchedulerInitResult:
    """启动调度器（scheduler）后的返回结果。"""

    # 各调度器进程上报的信息列表（如 max_req_input_len 等）。
    scheduler_infos: List[Dict[str, Any]]
    # 等待所有调度器就绪的回调（默认空操作）。
    wait_for_ready: Callable[[], None] = lambda: None
    # 等待所有调度器进程结束的回调（默认空操作）。
    wait_for_completion: Callable[[], None] = lambda: None
    # 可选的引擎信息 bootstrap server（用于远程权重加载等场景）。
    engine_info_bootstrap_server: Optional[Any] = None


def init_tokenizer_manager(
    server_args: ServerArgs,
    port_args: PortArgs,
    TokenizerManagerClass: Optional[TokenizerManager] = None,
) -> Tuple[TokenizerManager, TemplateManager]:
    # 启动 tokenizer（分词器）管理器。
    # 允许外部传入自定义的 TokenizerManager 子类，否则使用默认实现。
    TokenizerManagerClass = TokenizerManagerClass or TokenizerManager
    tokenizer_manager = TokenizerManagerClass(server_args, port_args)

    # 初始化对话/补全模板。
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )

    return tokenizer_manager, template_manager


class Engine(EngineBase):
    """
    推理引擎的入口类。

    - 引擎由三个组件构成：
        1. TokenizerManager：对请求分词并发送给调度器。
        2. Scheduler（子进程）：从 TokenizerManager 接收请求、组 batch、前向计算，
           并把输出 token 发送给 DetokenizerManager。
        3. DetokenizerManager（子进程）：对输出 token 反分词，并把结果回传给 TokenizerManager。

    注意：
    1. HTTP 服务器、Engine 和 TokenizerManager 都运行在主进程中。
    2. 进程间通信（IPC）通过 ZMQ 库完成（每个进程使用不同端口）。
    """

    # 以下字段允许用户覆盖 server args，并为其私有分支启动自定义进程。
    server_args_class: ServerArgs = ServerArgs
    # 初始化 tokenizer manager 的函数（可被子类替换）。
    init_tokenizer_manager_func: Callable = staticmethod(init_tokenizer_manager)
    # 运行 scheduler 进程的函数（可被子类替换）。
    run_scheduler_process_func: Callable = staticmethod(run_scheduler_process)
    # 运行 detokenizer 进程的函数（可被子类替换）。
    run_detokenizer_process_func: Callable = staticmethod(run_detokenizer_process)

    def __init__(self, **kwargs):
        """
        本函数的参数与 `sglang/srt/server_args.py::ServerArgs` 相同。
        参数文档请参阅 `ServerArgs`。
        """

        # 解析 server_args。
        if "server_args" in kwargs:
            # 情况一：直接传入了已构造好的 server_args。
            server_args = kwargs["server_args"]
        else:
            # 情况二：从 kwargs 构造 server_args。
            if "log_level" not in kwargs:
                # 默认不打印日志（仅 error 级别），避免作为 Python API 使用时日志过多。
                kwargs["log_level"] = "error"
            server_args = self.server_args_class(**kwargs)
        self.server_args = server_args
        logger.info(f"{server_args=}")

        # 预先把 tokenizer_manager 置为 None，
        # 这样 shutdown() 中的 atexit 回调即使提前触发也不会因属性缺失而报错。
        self.tokenizer_manager = None

        # 注册退出回调：程序退出时自动关闭所有子进程。
        atexit.register(self.shutdown)

        # 启动各子进程（tokenizer / scheduler / detokenizer）。
        (
            tokenizer_manager,
            template_manager,
            port_args,
            scheduler_init_result,
            subprocess_watchdog,
        ) = self._launch_subprocesses(
            server_args=server_args,
            init_tokenizer_manager_func=self.init_tokenizer_manager_func,
            run_scheduler_process_func=self.run_scheduler_process_func,
            run_detokenizer_process_func=self.run_detokenizer_process_func,
        )
        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager
        self._scheduler_init_result = scheduler_init_result
        # 把子进程看门狗挂到 tokenizer_manager 上，便于后续统一管理。
        if tokenizer_manager is not None:
            tokenizer_manager._subprocess_watchdog = subprocess_watchdog
        self.port_args = port_args
        # 如果启动了 bootstrap server，则记录传输引擎信息（用于远程实例权重加载）。
        if scheduler_init_result.engine_info_bootstrap_server is not None:
            self.remote_instance_transfer_engine_info = (
                scheduler_init_result.engine_info_bootstrap_server.transfer_engine_info
            )

        # 初始化 ZMQ socket，用于向调度器发送 RPC 请求。
        context = zmq.Context(2)
        if self.server_args.node_rank == 0:
            # 仅 node_rank == 0 的主节点创建 RPC 发送 socket。
            self.send_to_rpc = get_zmq_socket(
                context, zmq.DEALER, self.port_args.rpc_ipc_name, True
            )
        else:
            self.send_to_rpc = None

        # 启用链路追踪（tracing）。
        if server_args.enable_trace:
            process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
            thread_label = "Tokenizer"
            # 根据 PD 分离模式设置线程标签，便于追踪区分。
            if server_args.disaggregation_mode == "prefill":
                thread_label = "Prefill Tokenizer"
            elif server_args.disaggregation_mode == "decode":
                thread_label = "Decode Tokenizer"
            trace_set_thread_info(thread_label)

        # 获取或新建事件循环（同步 API 内部依赖该循环驱动异步逻辑）。
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def _resolve_routed_dp_rank(
        self,
        routed_dp_rank: Optional[int],
        data_parallel_rank: Optional[int],
    ) -> Optional[int]:
        # 处理已废弃的 data_parallel_rank 参数，兼容旧调用方式。
        if data_parallel_rank is not None:
            import warnings

            warnings.warn(
                "'data_parallel_rank' is deprecated, use 'routed_dp_rank' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            # 若未显式给出 routed_dp_rank，则回退使用废弃参数的值。
            if routed_dp_rank is None:
                routed_dp_rank = data_parallel_rank

        if routed_dp_rank is not None:
            dp_size = self.server_args.dp_size
            # dp_size <= 1 时指定 rank 0 无意义，直接忽略。
            if dp_size <= 1 and routed_dp_rank == 0:
                logger.warning(
                    f"routed_dp_rank={routed_dp_rank} is ignored because dp_size={dp_size}"
                )
                return None
            # 越界检查：rank 必须落在 [0, dp_size) 区间内。
            if routed_dp_rank < 0 or routed_dp_rank >= dp_size:
                raise ValueError(
                    f"routed_dp_rank={routed_dp_rank} out of range [0, {dp_size})"
                )

        logger.debug(f"routed_dp_rank: {routed_dp_rank}")
        return routed_dp_rank

    def generate(
        self,
        # 输入 prompt：可以是单个 prompt，也可以是一批 prompt。
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # 文本对应的 token id；text 与 input_ids 二选一即可。
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # 图像输入：可以是图像对象、文件名、URL 或 base64 编码字符串。
        # 支持的格式：
        # - 单请求的单张图像
        # - 图像列表（批量中每个请求一张）
        # - 图像列表的列表（每个请求多张图像）
        # - HuggingFace processor 的预处理输出列表，每项为含 `format`: 'processor_output' 等字段的 dict
        # - 预计算的图像 embedding 列表，每项为含 `format`: 'precomputed_embedding' 与 `feature` 字段的 dict
        # 更多细节参见 python/sglang/srt/utils.py:load_image。
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: bool = False,
        return_routed_experts: bool = False,
        stream: bool = False,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        routed_dp_rank: Optional[int] = None,
        disagg_prefill_dp_rank: Optional[int] = None,
        # 已废弃：请改用 routed_dp_rank。
        data_parallel_rank: Optional[int] = None,
        external_trace_header: Optional[Dict] = None,
        rid: Optional[Union[List[str], str]] = None,
        session_params: Optional[Dict] = None,
        priority: Optional[int] = None,
    ) -> Union[Dict, Iterator[Dict]]:
        """
        本函数的参数与 `sglang/srt/managers/io_struct.py::GenerateReqInput` 相同。
        参数文档请参阅 `GenerateReqInput`。

        这是【同步】生成接口。
        """
        # 解析并校验数据并行 rank（兼容废弃参数）。
        routed_dp_rank = self._resolve_routed_dp_rank(
            routed_dp_rank, data_parallel_rank
        )

        # 把所有参数封装成统一的请求对象。
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
            return_hidden_states=return_hidden_states,
            return_routed_experts=return_routed_experts,
            stream=stream,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            routed_dp_rank=routed_dp_rank,
            disagg_prefill_dp_rank=disagg_prefill_dp_rank,
            external_trace_header=external_trace_header,
            rid=rid,
            session_params=session_params,
            priority=priority,
        )
        # 通过 tokenizer_manager 提交请求，得到一个异步生成器。
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream:
            # 流式模式：把异步生成器包装成同步生成器逐块返回。
            def generator_wrapper():
                while True:
                    try:
                        # 在事件循环上驱动异步生成器，取出下一个数据块。
                        chunk = self.loop.run_until_complete(generator.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        # 异步迭代结束，退出循环。
                        break

            return generator_wrapper()
        else:
            # 非流式模式：直接驱动事件循环取出唯一结果并返回。
            ret = self.loop.run_until_complete(generator.__anext__())
            return ret

    async def async_generate(
        self,
        # 输入 prompt：可以是单个 prompt，也可以是一批 prompt。
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # 文本对应的 token id；text 与 input_ids 二选一即可。
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # 图像输入：含义与 generate() 完全一致（详见上方注释）。
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: bool = False,
        return_routed_experts: bool = False,
        stream: bool = False,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        routed_dp_rank: Optional[int] = None,
        disagg_prefill_dp_rank: Optional[int] = None,
        # 已废弃：请改用 routed_dp_rank。
        data_parallel_rank: Optional[int] = None,
        external_trace_header: Optional[Dict] = None,
        rid: Optional[Union[List[str], str]] = None,
        session_params: Optional[Dict] = None,
        priority: Optional[int] = None,
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """
        本函数的参数与 `sglang/srt/managers/io_struct.py::GenerateReqInput` 相同。
        参数文档请参阅 `GenerateReqInput`。

        这是 generate() 的【异步】版本。
        """
        # 解析并校验数据并行 rank（兼容废弃参数）。
        routed_dp_rank = self._resolve_routed_dp_rank(
            routed_dp_rank, data_parallel_rank
        )

        # 封装请求对象。
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            return_hidden_states=return_hidden_states,
            return_routed_experts=return_routed_experts,
            stream=stream,
            custom_logit_processor=custom_logit_processor,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            routed_dp_rank=routed_dp_rank,
            disagg_prefill_dp_rank=disagg_prefill_dp_rank,
            external_trace_header=external_trace_header,
            rid=rid,
            session_params=session_params,
            priority=priority,
        )
        # 提交请求获取异步生成器。
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream is True:
            # 流式：直接返回异步生成器交由调用方 await 迭代。
            return generator
        else:
            # 非流式：await 取出唯一结果返回。
            return await generator.__anext__()

    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
        dimensions: Optional[int] = None,
        lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None,
        external_trace_header: Optional[Dict] = None,
        rid: Optional[Union[List[str], str]] = None,
    ) -> Dict:
        """
        本函数的参数与 `sglang/srt/managers/io_struct.py::EmbeddingReqInput` 相同。
        参数文档请参阅 `EmbeddingReqInput`。

        用于生成 embedding（向量化）的【同步】接口。
        """
        # 封装为 embedding 请求对象。
        obj = EmbeddingReqInput(
            text=prompt,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            dimensions=dimensions,
            lora_path=lora_path,
            external_trace_header=external_trace_header,
            rid=rid,
        )
        # 提交请求并在事件循环上取结果。
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = self.loop.run_until_complete(generator.__anext__())
        return ret

    async def async_encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
        dimensions: Optional[int] = None,
        lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None,
        external_trace_header: Optional[Dict] = None,
        rid: Optional[Union[List[str], str]] = None,
    ) -> Dict:
        """
        encode 方法的【异步】版本。

        本函数的参数与 `sglang/srt/managers/io_struct.py::EmbeddingReqInput` 相同。
        参数文档请参阅 `EmbeddingReqInput`。
        """
        obj = EmbeddingReqInput(
            text=prompt,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            dimensions=dimensions,
            lora_path=lora_path,
            external_trace_header=external_trace_header,
            rid=rid,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)
        return await generator.__anext__()

    def rerank(
        self,
        prompt: Union[List[List[str]]],
    ) -> Dict:
        """
        本函数的参数与 `sglang/srt/managers/io_struct.py::EmbeddingReqInput` 相同。
        参数文档请参阅 `EmbeddingReqInput`。

        重排序（rerank）接口，内部以 cross-encoder 方式请求模型。
        """
        # is_cross_encoder_request=True 表示这是 cross-encoder 的重排序请求。
        obj = EmbeddingReqInput(text=prompt, is_cross_encoder_request=True)
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = self.loop.run_until_complete(generator.__anext__())
        return ret

    @classmethod
    def _launch_scheduler_processes(
        cls,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> Tuple[SchedulerInitResult, Optional[List]]:
        """使用 multiprocessing 启动调度器进程。
        子类可重写本方法以适配不同后端（例如 Ray）。

        返回：
            (SchedulerInitResult, scheduler_procs) 二元组。
            对于 RayEngine，scheduler_procs 为 None（它改用 Ray actor）。
        """
        scheduler_procs = []

        if server_args.dp_size == 1:
            # 分支一：dp_size == 1，直接启动张量并行（TP）的调度器进程。
            memory_saver_adapter = TorchMemorySaverAdapter.create(
                enable=server_args.enable_memory_saver
            )
            scheduler_pipe_readers = []

            # 计算本节点负责的 PP / TP rank 区间，以及每节点的 PP / TP 规模。
            pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node = (
                _calculate_rank_ranges(
                    server_args.nnodes,
                    server_args.pp_size,
                    server_args.tp_size,
                    server_args.node_rank,
                )
            )

            # 为本节点上的每个 (pp_rank, tp_rank) 组合启动一个调度器进程。
            for pp_rank in pp_rank_range:
                for tp_rank in tp_rank_range:
                    # 创建单向管道（子进程通过 writer 回传就绪信息）。
                    reader, writer = mp.Pipe(duplex=False)
                    # 根据 base_gpu_id 与并行 rank 计算实际使用的 GPU id。
                    gpu_id = (
                        server_args.base_gpu_id
                        + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                        + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                    )
                    # 计算注意力 CP、MoE DP、MoE EP 等并行 rank。
                    attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(
                        server_args, tp_rank
                    )

                    # maybe_reindex_device_id 可能会对 GPU id 重新映射。
                    with maybe_reindex_device_id(gpu_id) as gpu_id:
                        proc = mp.Process(
                            target=run_scheduler_process_func,
                            args=(
                                server_args,
                                port_args,
                                gpu_id,
                                tp_rank,
                                attn_cp_rank,
                                moe_dp_rank,
                                moe_ep_rank,
                                pp_rank,
                                None,
                                writer,
                            ),
                        )
                        # 在子进程配置上下文（显存节省、NUMA 绑核）中启动进程。
                        with memory_saver_adapter.configure_subprocess(), numa_utils.configure_subprocess(
                            server_args, gpu_id
                        ):
                            proc.start()

                    scheduler_procs.append(proc)
                    scheduler_pipe_readers.append(reader)
        else:
            # 分支二：dp_size > 1，启动数据并行控制器（由它再去拉起调度器）。
            reader, writer = mp.Pipe(duplex=False)
            scheduler_pipe_readers = [reader]
            proc = mp.Process(
                target=run_data_parallel_controller_process,
                kwargs=dict(
                    server_args=server_args,
                    port_args=port_args,
                    pipe_writer=writer,
                    run_scheduler_process_func=run_scheduler_process_func,
                ),
            )
            proc.start()
            scheduler_procs.append(proc)

        scheduler_infos = []

        def wait_for_ready():
            # 等待所有调度器进程完成模型加载并上报就绪信息。
            infos = _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs)
            scheduler_infos.extend(infos)

        def wait_for_completion():
            # 阻塞等待所有调度器/数据并行控制器进程退出，并记录退出码。
            for proc in scheduler_procs:
                proc.join()
                logger.error(
                    f"Scheduler or DataParallelController {proc.pid} "
                    f"terminated with {proc.exitcode}"
                )

        return (
            SchedulerInitResult(
                scheduler_infos=scheduler_infos,
                wait_for_ready=wait_for_ready,
                wait_for_completion=wait_for_completion,
            ),
            scheduler_procs,
        )

    @classmethod
    def _launch_subprocesses(
        cls,
        server_args: ServerArgs,
        init_tokenizer_manager_func: Callable,
        run_scheduler_process_func: Callable,
        run_detokenizer_process_func: Callable,
        port_args: Optional[PortArgs] = None,
    ) -> Tuple[
        TokenizerManager,
        TemplateManager,
        PortArgs,
        SchedulerInitResult,
        Optional[SubprocessWatchdog],
    ]:
        """在主进程启动 TokenizerManager，在子进程启动 Scheduler，在另一子进程启动 DetokenizerManager。

        返回：
            (tokenizer_manager, template_manager, port_args, scheduler_init_result, subprocess_watchdog) 元组。
        """
        # 配置全局环境（日志、环境变量、参数校验、GC 设置）。
        configure_logger(server_args)
        _set_envs_and_config(server_args)
        server_args.check_server_args()
        _set_gc(server_args)

        # 为进程间通信分配端口。
        if port_args is None:
            port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

        # 若需要按 rank 提供信息，则启动引擎信息 bootstrap server。
        engine_info_bootstrap_server = None
        if (
            server_args.remote_instance_weight_loader_start_seed_via_transfer_engine
            and server_args.node_rank == 0
        ):
            bootstrap_port = server_args.engine_info_bootstrap_port
            # 端口占用检查：同机多实例时必须使用不同端口。
            if not is_port_available(bootstrap_port):
                raise RuntimeError(
                    f"engine_info_bootstrap_port {bootstrap_port} is already in use. "
                    f"When running multiple instances on the same node, each instance must use a "
                    f"different --engine-info-bootstrap-port."
                )
            engine_info_bootstrap_server = EngineInfoBootstrapServer(
                host=server_args.host, port=bootstrap_port
            )

        # 启动调度器进程。
        scheduler_init_result, scheduler_procs = cls._launch_scheduler_processes(
            server_args, port_args, run_scheduler_process_func
        )
        scheduler_init_result.engine_info_bootstrap_server = (
            engine_info_bootstrap_server
        )

        # 如启用弹性专家备份（elastic expert backup），则启动对应管理器。
        if (
            server_args.enable_elastic_expert_backup
            and server_args.elastic_ep_backend is not None
        ):
            run_expert_backup_manager(server_args, port_args)

        if server_args.node_rank >= 1:
            # 多节点场景下，非 0 号节点不需要运行 tokenizer/detokenizer，在此等待即可。
            scheduler_init_result.wait_for_ready()

            if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
                # 当把 Engine 当作 Python API 使用时，不希望在此阻塞，直接返回。
                return (
                    None,
                    None,
                    port_args,
                    scheduler_init_result,
                    None,
                )

            # 启动一个 dummy 健康检查服务器（供负载均衡器探活）。
            launch_dummy_health_check_server(
                server_args.host, server_args.port, server_args.enable_metrics
            )

            # 阻塞等待调度器进程结束。
            scheduler_init_result.wait_for_completion()
            return (
                None,
                None,
                port_args,
                scheduler_init_result,
                None,
            )

        # 启动 detokenizer（反分词器）进程。
        detoken_proc = mp.Process(
            target=run_detokenizer_process_func,
            args=(
                server_args,
                port_args,
            ),
        )
        detoken_proc.start()

        # 先初始化 tokenizer manager（bootstrap server 也在这里完成初始化）。
        if server_args.tokenizer_worker_num == 1:
            # 单 worker：使用标准初始化函数。
            tokenizer_manager, template_manager = init_tokenizer_manager_func(
                server_args, port_args
            )
        else:
            # 多 worker：启动多分词器路由（MultiTokenizerRouter）。
            tokenizer_manager = MultiTokenizerRouter(server_args, port_args)
            template_manager = None

        # 等待模型加载完成。
        scheduler_init_result.wait_for_ready()

        # 从调度器拿回一些信息给 tokenizer_manager（例如最大请求输入长度）。
        tokenizer_manager.max_req_input_len = scheduler_init_result.scheduler_infos[0][
            "max_req_input_len"
        ]

        # 设置子进程存活看门狗，用于检测崩溃。
        # 注意：RayEngine 返回 scheduler_procs=None，因为它使用 Ray actor 而非 mp.Process。
        processes = list(scheduler_procs or [])
        names = [f"scheduler_{i}" for i in range(len(processes))]
        processes.append(detoken_proc)
        names.append("detokenizer")
        subprocess_watchdog = SubprocessWatchdog(
            processes=processes, process_names=names
        )
        subprocess_watchdog.start()

        return (
            tokenizer_manager,
            template_manager,
            port_args,
            scheduler_init_result,
            subprocess_watchdog,
        )

    def shutdown(self):
        """关闭引擎，停止看门狗并杀掉子进程树。"""
        if (
            self.tokenizer_manager is not None
            and self.tokenizer_manager._subprocess_watchdog is not None
        ):
            # 先停止看门狗，避免它误判子进程被杀而再次触发清理。
            self.tokenizer_manager._subprocess_watchdog.stop()
        # 杀掉当前进程的所有子进程（不含父进程自身）。
        kill_process_tree(os.getpid(), include_parent=False)

    def __enter__(self):
        # 支持 with 语句上下文管理。
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # 退出 with 块时自动关闭引擎。
        self.shutdown()
        return False

    def flush_cache(self):
        # 刷新（清空）缓存，例如 RadixAttention 的前缀缓存。
        return self.loop.run_until_complete(self.tokenizer_manager.flush_cache())

    def open_session(
        self,
        capacity_of_str_len: int,
        session_id: Optional[str] = None,
        streaming: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        """打开一个共享上下文的多轮对话会话（session）。

        参数：
            capacity_of_str_len：会话的最大字符串长度容量。
            session_id：可选的会话 ID；若不提供则自动生成 UUID。
            streaming：对实时流式使用低开销路径（仅追加模式）。
            timeout：若设置，会话在空闲该秒数后自动关闭。
                空闲时间从会话打开或最近一次请求提交开始计算。

        返回：
            会话 ID（传入的或新生成的 UUID）。
        """
        obj = OpenSessionReqInput(
            capacity_of_str_len=capacity_of_str_len,
            session_id=session_id,
            streaming=streaming,
            timeout=timeout,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.open_session(obj, None)
        )

    def close_session(self, session_id: str) -> None:
        """关闭会话并释放其资源。

        参数：
            session_id：要关闭的会话 ID。
        """
        obj = CloseSessionReqInput(session_id=session_id)
        self.loop.run_until_complete(self.tokenizer_manager.close_session(obj, None))

    def start_profile(self, **kwargs):
        # 开始性能采集（profiling）。
        self.loop.run_until_complete(self.tokenizer_manager.start_profile(**kwargs))

    def stop_profile(self):
        # 停止性能采集。
        self.loop.run_until_complete(self.tokenizer_manager.stop_profile())

    def start_expert_distribution_record(self):
        # 开始记录专家（MoE expert）分布信息。
        self.loop.run_until_complete(
            self.tokenizer_manager.start_expert_distribution_record()
        )

    def stop_expert_distribution_record(self):
        # 停止记录专家分布信息。
        self.loop.run_until_complete(
            self.tokenizer_manager.stop_expert_distribution_record()
        )

    def dump_expert_distribution_record(self):
        # 导出已记录的专家分布信息。
        self.loop.run_until_complete(
            self.tokenizer_manager.dump_expert_distribution_record()
        )

    def get_server_info(self):
        # 获取服务器信息：合并 server_args、调度器信息、内部状态与版本号。
        internal_states = self.loop.run_until_complete(
            self.tokenizer_manager.get_internal_state()
        )
        return {
            **dataclasses.asdict(self.tokenizer_manager.server_args),
            **self._scheduler_init_result.scheduler_infos[0],
            "internal_states": internal_states,
            "version": __version__,
        }

    def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ):
        """初始化参数更新通信组（用于分布式权重更新）。"""
        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.init_weights_update_group(obj, None)
        )

    def destroy_weights_update_group(
        self,
        group_name: str,
    ):
        """销毁参数更新通信组。"""
        obj = DestroyWeightsUpdateGroupReqInput(
            group_name=group_name,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.destroy_weights_update_group(obj, None)
        )

    def update_weights_from_distributed(
        self,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str = "weight_update_group",
        flush_cache: bool = True,
        load_format: Optional[str] = None,
    ):
        """从分布式源更新权重。"""
        obj = UpdateWeightsFromDistributedReqInput(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=flush_cache,
            load_format=load_format,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_distributed(obj, None)
        )

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """从张量更新权重。若后续还会有多次更新，可将 `flush_cache` 设为 False，
        以避免重复的缓存清理操作。"""
        if load_format == "flattened_bucket":
            # flattened_bucket 格式无需序列化，直接透传。
            serialized_named_tensors = named_tensors
        else:
            # 否则按 TP 大小逐份序列化张量（每个 TP rank 一份）。
            serialized_named_tensors = [
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(self.server_args.tp_size)
            ]
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=flush_cache,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_tensor(obj, None)
        )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: Optional[str] = None,
    ):
        """在不重启引擎的情况下，从磁盘原地更新权重。

        本方法允许从磁盘更新模型权重而无需重启引擎，
        可用于加载不同模型或用新训练得到的权重进行更新。
        """
        obj = UpdateWeightFromDiskReqInput(
            model_path=model_path,
            load_format=load_format,
        )

        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_disk(obj, None)
        )

    def update_weights_from_ipc(
        self,
        zmq_handles: Dict[str, str],
        flush_cache: bool = True,
    ):
        """通过 IPC 更新权重（用于 checkpoint-engine 集成）。"""
        obj = UpdateWeightsFromIPCReqInput(
            zmq_handles=zmq_handles,
            flush_cache=flush_cache,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_ipc(obj, None)
        )

    def get_weights_by_name(self, name: str, truncate_size: int = 100):
        """按参数名获取权重（可截断返回前 truncate_size 个元素）。"""
        obj = GetWeightsByNameReqInput(name=name, truncate_size=truncate_size)
        return self.loop.run_until_complete(
            self.tokenizer_manager.get_weights_by_name(obj, None)
        )

    def load_lora_adapter_from_tensors(
        self,
        lora_name: str,
        tensors,
        config_dict: Dict,
        load_format: Optional[str] = None,
    ):
        # 从张量加载 LoRA 适配器。
        if load_format == "flattened_bucket":
            # flattened_bucket 格式无需序列化。
            serialized_tensors = tensors
        else:
            # 否则序列化为字符串形式。
            serialized_tensors = MultiprocessingSerializer.serialize(
                tensors, output_str=True
            )
        lora_req = LoadLoRAAdapterFromTensorsReqInput(
            lora_name=lora_name,
            config_dict=config_dict,
            serialized_tensors=serialized_tensors,
            load_format=load_format,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.load_lora_adapter_from_tensors(lora_req, None)
        )

    def load_lora_adapter(self, lora_name: str, lora_path: str, pinned: bool = False):
        """在不重启引擎的情况下加载一个新的 LoRA 适配器。"""

        obj = LoadLoRAAdapterReqInput(
            lora_name=lora_name,
            lora_path=lora_path,
            pinned=pinned,  # pinned=True 表示常驻、不被换出。
        )

        return self.loop.run_until_complete(
            self.tokenizer_manager.load_lora_adapter(obj, None)
        )

    def unload_lora_adapter(self, lora_name: str):
        """在不重启引擎的情况下卸载一个 LoRA 适配器。"""

        obj = UnloadLoRAAdapterReqInput(lora_name=lora_name)

        return self.loop.run_until_complete(
            self.tokenizer_manager.unload_lora_adapter(obj, None)
        )

    async def async_load_lora_adapter(
        self, lora_name: str, lora_path: str, pinned: bool = False
    ):
        """
        load_lora_adapter 的【异步】版本。

        详细文档参见 load_lora_adapter()。
        """

        obj = LoadLoRAAdapterReqInput(
            lora_name=lora_name,
            lora_path=lora_path,
            pinned=pinned,
        )

        return await self.tokenizer_manager.load_lora_adapter(obj, None)

    async def async_unload_lora_adapter(self, lora_name: str):
        """
        unload_lora_adapter 的【异步】版本。

        详细文档参见 unload_lora_adapter()。
        """

        obj = UnloadLoRAAdapterReqInput(lora_name=lora_name)

        return await self.tokenizer_manager.unload_lora_adapter(obj, None)

    def release_memory_occupation(self, tags: Optional[List[str]] = None):
        # 释放显存占用（按可选的 tags 指定释放范围）。
        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        return self.loop.run_until_complete(
            self.tokenizer_manager.release_memory_occupation(obj, None)
        )

    def resume_memory_occupation(self, tags: Optional[List[str]] = None):
        # 恢复之前释放的显存占用。
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        return self.loop.run_until_complete(
            self.tokenizer_manager.resume_memory_occupation(obj, None)
        )

    def freeze_gc(self):
        """
        为了维持低延迟的高性能服务，我们希望减少垃圾回收器扫描大量对象时造成的卡顿。

        通常的做法是：先启动服务并用真实请求进行预热，从而初始化许多无需被回收的长生命周期对象。

        在充分预热之后，可调用本函数“冻结”垃圾回收器，
        使此刻之前创建的所有对象都被视为无需参与垃圾回收。
        """

        self.loop.run_until_complete(self.tokenizer_manager.freeze_gc())

    """
    在所有调度器进程上执行一次 RPC 调用。
    """

    def collective_rpc(self, method: str, **kwargs):
        # 构造 RPC 请求并通过 ZMQ 发送给调度器。
        obj = RpcReqInput(method=method, parameters=kwargs)
        self.send_to_rpc.send_pyobj(obj)
        # 阻塞接收 RPC 结果。
        recv_req = self.send_to_rpc.recv_pyobj(zmq.BLOCKY)
        assert isinstance(recv_req, RpcReqOutput)
        # 断言 RPC 执行成功，否则抛出携带错误信息的异常。
        assert recv_req.success, recv_req.message

    def save_remote_model(self, **kwargs):
        # 通过 RPC 让调度器把模型保存到远端存储。
        self.collective_rpc("save_remote_model", **kwargs)

    def save_sharded_model(self, **kwargs):
        # 通过 RPC 让调度器以分片方式保存模型。
        self.collective_rpc("save_sharded_model", **kwargs)

    def score(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> ScoreResult:
        """
        给定 (query + item) 对，计算指定 token ID 作为下一个 token 出现的概率。例如：
        query = "<|user|>Is the following city the capital of France? "
        items = ["Paris <|assistant|>", "London <|assistant|>", "Berlin <|assistant|>"]
        label_token_ids = [2332, 1223]  # 分别是 "Yes" 和 "No" 的 token ID
        item_first = False

        上述配置会向模型传入如下 prompt：
        "<|user|>Is the following city the capital of France? Paris <|assistant|>"
        "<|user|>Is the following city the capital of France? London <|assistant|>"
        "<|user|>Is the following city the capital of France? Berlin <|assistant|>"
        然后接口会返回模型把 "Yes" 与 "No" 作为下一个 token 的概率，
        输出形如：
        [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]]

        参数：
            query：query 文本或预分词后的 query token ID（必填）。
            items：item 文本或预分词后的 item token ID（必填）。
            label_token_ids：要计算概率的 token ID 列表；若为 None 则不计算任何 token 概率。
            apply_softmax：是否对概率做 softmax 归一化。
            item_first：若为 True，则把 items 拼到 query 前面；否则拼到后面。

        返回：
            ScoreResult，包含：
                scores：嵌套列表，含每个 item、每个 label token 的概率。
                prompt_tokens：处理的 prompt token 数量。

        抛出：
            ValueError：当未提供 query 或 items、或 token ID 超出词表、
                      或指定 token 不可用 logprob 时。
        """
        return self.loop.run_until_complete(
            self.tokenizer_manager.score_request(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=item_first,
                request=None,
            )
        )

    async def async_score(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> ScoreResult:
        """
        score 方法的【异步】版本。

        详细文档参见 score()。
        """
        return await self.tokenizer_manager.score_request(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
            item_first=item_first,
            request=None,
        )


def _set_envs_and_config(server_args: ServerArgs):
    # 设置全局环境变量与配置。
    # NCCL_CUMEM_ENABLE：控制 NCCL 的 cuMem 分配；启用对称内存时需要打开。
    if "NCCL_CUMEM_ENABLE" not in os.environ or server_args.enable_symm_mem:
        os.environ["NCCL_CUMEM_ENABLE"] = str(int(server_args.enable_symm_mem))
    # NCCL_NVLS_ENABLE：控制 NVLink SHARP；启用 NCCL NVLS 或对称内存时打开。
    if (
        "NCCL_NVLS_ENABLE" not in os.environ
        or server_args.enable_nccl_nvls
        or server_args.enable_symm_mem
    ):
        os.environ["NCCL_NVLS_ENABLE"] = str(
            int(server_args.enable_nccl_nvls or server_args.enable_symm_mem)
        )
    # 限制单卡的最大连接数；加速 CUDA module 加载。
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    if os.environ.get("TRTLLM_ENABLE_PDL", "1") != "0":
        # flashinfer 会用该环境变量控制从 MoE 到量化等多种 kernel 的 PDL 行为。
        os.environ["TRTLLM_ENABLE_PDL"] = "1"

    if os.environ.get("CUTE_DSL_LOG_LEVEL") is None:
        # 默认设为 warning 级别（30），避免日志过多。
        os.environ["CUTE_DSL_LOG_LEVEL"] = "30"

    if os.environ.get("CUTE_DSL_LOG_TO_CONSOLE") is None:
        # 需要把日志输出到控制台，否则日志级别设置不会生效。
        os.environ["CUTE_DSL_LOG_TO_CONSOLE"] = "1"

    # 生成本次运行的唯一 ID（也可作为参数传入）。
    os.environ["SGLANG_RUN_ID"] = (
        f"sglang-run-{time.time()}-{random.randint(0, 100000000)}"
    )

    # 若启用指标采集，则设置 Prometheus 多进程目录。
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # 设置 ulimit（提高文件描述符等限制）。
    set_ulimit()

    # 检查 flashinfer / sgl-kernel 的版本是否满足要求。
    if not get_bool_env_var("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"):
        if server_args.attention_backend == "flashinfer":
            assert_pkg_version(
                "flashinfer_python",
                "0.6.7",
                "Please uninstall the old version and "
                "reinstall the latest version by following the instructions "
                "at https://docs.flashinfer.ai/installation.html.",
            )
        if _is_cuda:
            assert_pkg_version(
                "sglang-kernel",
                "0.4.0",
                "Please reinstall the latest version with `pip install sglang-kernel --force-reinstall`",
            )

    # 信号处理器只能在主线程中注册。
    if threading.current_thread() is threading.main_thread():
        if server_args.custom_sigquit_handler is None:
            # 注册默认 SIGQUIT 处理器。
            # 子进程发生错误时会向本进程发送 SIGQUIT，本进程随后清理整个进程树。
            # 注意：该处理器用于启动阶段，grpc server 启动后可能会被 tokenizer manager
            # 中的运行阶段处理器（running_phase_sigquit_handler）替换。
            def launch_phase_sigquit_handler(signum, frame):
                logger.error(
                    "Received sigquit from a child process. It usually means the child failed."
                )
                kill_process_tree(os.getpid())

            signal.signal(signal.SIGQUIT, launch_phase_sigquit_handler)
        else:
            # 允许用户注册自定义 SIGQUIT 处理器（例如做 crash dump）。
            logger.error(
                f"Using custom SIGQUIT handler: {server_args.custom_sigquit_handler}"
            )
            signal.signal(signal.SIGQUIT, server_args.custom_sigquit_handler)
    else:
        # 非主线程下无法注册信号处理器，给出告警。
        logger.warning(
            "Signal handler is not added because the engine is not in the "
            "main thread. This disables the SIGQUIT handler for cleaning up "
            "the process tree when a child process fails."
        )

    # 设置多进程启动方式为 spawn（CUDA 场景下必须）。
    mp.set_start_method("spawn", force=True)


def _set_gc(server_args: ServerArgs):
    # 如配置了 GC 阈值，则覆盖默认垃圾回收阈值。
    if gc_threshold := server_args.gc_threshold:
        import gc

        gc.set_threshold(*gc_threshold)


def _scheduler_died_error(rank: int, proc) -> RuntimeError:
    """为初始化期间死亡的调度器进程构造一条描述性错误。"""
    # 给进程留 10 秒回收，以拿到退出码。
    proc.join(timeout=10)
    return RuntimeError(
        f"Rank {rank} scheduler died during initialization "
        f"(exit code: {proc.exitcode}). "
        f"If exit code is -9 (SIGKILL), a common cause is the OS OOM killer. "
        f"Run `dmesg -T | grep -i oom` to check."
    )


def _wait_for_scheduler_ready(
    scheduler_pipe_readers: List,
    scheduler_procs: List,
) -> List[Dict]:
    """等待模型加载完成并返回各调度器信息。

    使用带超时的 poll() 而非阻塞式 recv()，
    这样在子进程死亡（例如 OOM 被 SIGKILL）时能及时发现，而不会一直挂起。
    """
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        while True:
            # 每 5 秒轮询一次管道是否有数据可读。
            if scheduler_pipe_readers[i].poll(timeout=5.0):
                try:
                    data = scheduler_pipe_readers[i].recv()
                except EOFError:
                    # 管道被关闭说明子进程已死。
                    raise _scheduler_died_error(i, scheduler_procs[i])
                if data["status"] != "ready":
                    # 子进程上报非就绪状态，说明初始化失败。
                    raise RuntimeError(
                        "Initialization failed. Please see the error messages above."
                    )
                scheduler_infos.append(data)
                break

            # poll 超时——检查所有进程是否有提前死亡的。
            for j in range(len(scheduler_procs)):
                if not scheduler_procs[j].is_alive():
                    raise _scheduler_died_error(j, scheduler_procs[j])

    return scheduler_infos


def _calculate_rank_ranges(
    nnodes: int, pp_size: int, tp_size: int, node_rank: int
) -> Tuple[range, range, int, int]:
    """计算指定节点上的 pp_rank_range 与 tp_rank_range。

    参数：
        nnodes：节点总数。
        pp_size：流水线并行（pipeline parallel）大小。
        tp_size：张量并行（tensor parallel）大小。
        node_rank：要计算区间的节点 rank。

    返回：
        (pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node) 四元组：
        - pp_rank_range：分配给本节点的流水线并行 rank 区间。
        - tp_rank_range：分配给本节点的张量并行 rank 区间。
        - pp_size_per_node：每节点的 PP rank 数。
        - tp_size_per_node：每节点的 TP rank 数。
    """
    # 每节点的 PP 规模（至少为 1）。
    pp_size_per_node = max(pp_size // nnodes, 1)
    # 每个 PP rank 占用的节点数（至少为 1）。
    nnodes_per_pp_rank = max(nnodes // pp_size, 1)
    # 计算本节点负责的 PP rank 区间。
    pp_rank_range = range(
        pp_size_per_node * (node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (node_rank // nnodes_per_pp_rank + 1),
    )

    # 每个 TP 组占用的节点数与 PP 相同。
    nnodes_per_tp_group = nnodes_per_pp_rank
    tp_size_per_node = tp_size // nnodes_per_tp_group
    # 计算本节点负责的 TP rank 区间。
    tp_rank_range = range(
        tp_size_per_node * (node_rank % nnodes_per_tp_group),
        tp_size_per_node * (node_rank % nnodes_per_tp_group + 1),
    )

    return pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node


def _compute_parallelism_ranks(
    server_args: ServerArgs, tp_rank: int
) -> Tuple[int, int, int]:
    """为某个 TP rank 计算注意力-CP、MoE-DP、MoE-EP 的 rank。"""
    # 启用 DP attention 时 attn_dp_size 等于 dp_size，否则为 1。
    attn_dp_size = server_args.dp_size if server_args.enable_dp_attention else 1

    # 并行层级（从最外层到最内层）：
    # - 注意力：Global(TP) -> DP -> ATTN_CP -> ATTN_TP（最内层）
    # - MoE：Global(TP) -> MOE_DP -> EP -> MOE_TP（最内层）
    attn_tp_size = server_args.tp_size // attn_dp_size // server_args.attn_cp_size
    # 注意力上下文并行（context parallel）rank。
    attn_cp_rank = (tp_rank // attn_tp_size) % server_args.attn_cp_size
    # MoE 数据并行 rank。
    moe_dp_rank = tp_rank // (server_args.tp_size // server_args.moe_dp_size)
    # MoE 专家并行（expert parallel）rank。
    moe_ep_rank = (
        tp_rank
        % (server_args.tp_size // server_args.moe_dp_size)
        // (server_args.tp_size // server_args.moe_dp_size // server_args.ep_size)
    )
    return attn_cp_rank, moe_dp_rank, moe_ep_rank

