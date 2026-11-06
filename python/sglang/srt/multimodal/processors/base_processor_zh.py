"""多模态处理器基类（中文注释版）。

本文件是 ``base_processor.py`` 的逐段注解副本：代码逻辑与原文件完全一致，
只额外插入中文注释，便于对照阅读。真正的实现以 ``base_processor.py`` 为准，
本文件不参与运行。

────────────────────────────────────────────────────────────────────
一、这个文件解决的问题
────────────────────────────────────────────────────────────────────
一次带图片/视频/音频的请求，在进入 scheduler 之前要完成三件事：

1. 取数据（load）：把 URL / base64 / 本地路径变成真正的像素与采样点，
   即 PIL.Image、torch.Tensor、np.ndarray。IO 密集 → 放线程池并发做。
2. 预处理（process）：交给 HF 的 AutoProcessor 抽 patch、算 grid_thw，
   得到 pixel_values 等特征张量。CPU/GPU 密集 → 放进程池或隔离线程做。
3. 对齐（combine）：把 prompt 里的占位 token 展开成真实 token 数，并算出
   每个多模态 item 在 input_ids 中的 [start, end] 偏移，供 scheduler 做
   prefix cache 命中与 M-RoPE 位置计算。

────────────────────────────────────────────────────────────────────
二、两条 load 路径（本文件最容易看晕的地方）
────────────────────────────────────────────────────────────────────
``load_mm_data`` 是个分拣器，根据「prompt 中占位符个数」与「实际给的数据
个数」是否严格 1:1 对齐，决定走哪条路：

* ``fast_load_mm_data``：个数完全对齐，把三路数据一次性提交到 IO 线程池，
  按原索引回填，不扫描 prompt。快，但无法表达「一个占位符展开成多帧」。
* ``legacy_load_mm_data``：个数对不上（或 skip_tokenizer_init、或模型支持
  动态帧扩展，如 MiniCPM-V / MiniCPM-O）。做法是按 prompt 的切分结果逐个
  占位符配一个数据，加载完再把占位符替换成 N 个（N = 该数据展开的帧数/图数）。
  兼容性强，但依赖三个迭代器的隐式同步，逻辑绕，后文逐行注释。

────────────────────────────────────────────────────────────────────
三、并发模型速记
────────────────────────────────────────────────────────────────────
* ``io_executor``：线程池，只做取数据；PIL/ffmpeg 解码期间会释放 GIL。
* ``cpu_executor``：进程池，做重 CPU 预处理（子类通过 multiprocessing 接口使用）。
* ``mm_processor_executor``：线程池 + 每线程 deepcopy 一份 HF processor，
  用于并发执行 ``process_and_combine_mm_data``。
* 事件循环线程只做调度与 await，任何阻塞操作都必须挪到上面几个池子里。
"""

import asyncio
import concurrent
import concurrent.futures
import concurrent.futures.process
import dataclasses
import multiprocessing as mp
import os
import re
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from PIL import Image
from transformers import BaseImageProcessor

from sglang.srt import platforms
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.cache import (
    MultimodalPreprocessCache,
    PreprocessFingerprintProvider,
    build_processor_fingerprint,
)
from sglang.srt.multimodal.processors.executor import MultimodalProcessorExecutor
from sglang.srt.multimodal.processors.hash_executor import MultimodalHashExecutor
from sglang.srt.multimodal.transport.cuda_ipc import (
    MM_FEATURE_CACHE_SIZE,
    MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
    get_mm_feature_pool_size_per_worker,
)
from sglang.srt.multimodal.transport.memory_pool import get_mm_feature_consumer_count
from sglang.srt.runtime_context import (
    get_exec,
    get_mm,
    get_serving,
)
from sglang.srt.utils import (
    CLIENT_MEDIA_EXCEPTIONS,
    configure_media_url_security,
    envs,
    is_cpu,
    is_npu,
    is_xpu,
    load_audio,
    load_image,
    load_video,
    logger,
    smart_to_rgb,
)

# 进程级缓存平台判断结果：这些探测函数会在每个请求的热路径上被反复调用，
# 因此在模块导入时只算一次。
_is_cpu = is_cpu()
_is_npu = is_npu()
_is_xpu = is_xpu()


@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    """``load_mm_data`` 的返回值：已展开占位符的文本 + 已加载好的原始数据。

    注意这里装的还是「原料」：images 是 PIL.Image、videos 是 torch.Tensor、
    audios 是 np.ndarray，尚未经过 HF processor。真正变成 pixel_values 是在
    后续 ``process_and_combine_mm_data`` 中完成的。

    * input_text：占位符**已展开**后的 prompt 文本。例如 Qwen-VL 中一张图
      原本只是一个占位符，展开后变成 N 个；legacy 路径下 N 就是该图/该视频
      实际解码出的帧数。
    * input_ids：调用方原本就带过来的 token ids（预分词输入），用于避免
      decode→retokenize 引起的 token 漂移；普通请求下为 None。
    * images/videos/audios：按 prompt 中出现顺序收集好的原始数据，元素可能是
      原始数据，也可能是 dict（processor_output / 预计算 embedding）。
    """

    # input_text with all multimodality placeholder token expanded
    input_text: str

    # original pre-tokenized ids, useful for processor_output/precomputed inputs,
    # when they already carry the input ids
    # 预分词输入：调用方直接给了 token ids 而非字符串。
    # 这类输入不能再 decode 回文本，否则会引入 token 漂移。
    input_ids: Optional[Union[List[int], torch.Tensor]] = None

    # frames loaded from image, in given order
    # 「已加载、未处理」的图像帧，顺序与 prompt 中占位符出现顺序一致。
    # 元素类型：PIL.Image / torch.Tensor（nvJPEG 在 GPU 上解码的结果）/ dict。
    images: Optional[list[Union[Image.Image, dict]]] = dataclasses.field(
        default_factory=list
    )

    # videos
    # 已加载的视频张量（形状通常是 T x H x W x C）或 dict。
    videos: Optional[list[Union[torch.Tensor, dict]]] = dataclasses.field(
        default_factory=list
    )

    # audios
    # 已加载的音频采样点，np.ndarray（采样点 × 声道）。
    audios: Optional[list[Union[np.ndarray, dict]]] = dataclasses.field(
        default_factory=list
    )

    def organize_results(self) -> List[Tuple[Modality, Any]]:
        """

        :return: a list of results, with their corresponding modalities
        """
        # 把「按模态分桶」的结果摊平成 (模态, 数据) 的有序列表，供
        # process_and_combine_mm_data 逐项分流。
        # 这里 image → video → audio 的固定顺序不影响正确性：后续每个 item 的
        # offset 是根据 input_ids 中占位符的实际位置重新计算的，而不是靠列表下标。
        return (
            [(Modality.IMAGE, data) for data in self.images]
            + [(Modality.VIDEO, data) for data in self.videos]
            + [(Modality.AUDIO, data) for data in self.audios]
        )


@dataclasses.dataclass
class MultimodalSpecialTokens:
    """描述「某模型用哪些占位符代表哪种模态」的小工具类。

    三层信息：
    1. ``*_token``：占位符的**字符串**形式（如 "<|image_pad|>"），用来切分 prompt；
    2. ``*_token_id``：占位符的 **token id**，用来在 input_ids 中定位偏移；
    3. ``*_token_regex`` / ``combined_regex``：由字符串编译出的正则，切分时
       一次性匹配三种模态。

    构造时通常只填 id（由模型 processor 从 tokenizer 查出），``build()`` 负责
    反查字符串并编译正则；之后同一实例反复使用，不会重复编译。
    """

    image_token: Optional[Union[str, List[str]]] = None
    video_token: Optional[Union[str, List[str]]] = None
    audio_token: Optional[Union[str, List[str]]] = None

    image_token_id: Optional[int] = None
    video_token_id: Optional[int] = None
    audio_token_id: Optional[int] = None

    image_token_regex: Optional[re.Pattern] = None
    video_token_regex: Optional[re.Pattern] = None
    audio_token_regex: Optional[re.Pattern] = None

    combined_regex: Optional[re.Pattern] = None

    def build(self, processor):
        # 一次性完成三件准备：
        # 1) 用 tokenizer 把 token id 反查成字符串；
        # 2) 把字符串编译成单模态正则；
        # 3) 拼接成「一次匹配三种模态」的合并正则。
        # 后续切分 prompt 时只需用 combined_regex，不用反复试三个正则。
        self.convert_to_strs(processor)
        self.parse_regex()
        self.get_combined_regex()
        return self

    def convert_to_str(self, token: Union[str, int], processor) -> str:
        # 三种情况：未配置→原样返回 None；已经是字符串→直接用；
        # 是 token id→走 tokenizer 反查。注意用 convert_ids_to_tokens 而非 decode：
        # 单个 id 的 decode 可能带上多余空白，而后面的正则匹配要求逐字符精确。
        if token is None:
            return token
        if isinstance(token, str):
            return token
        return processor.tokenizer.convert_ids_to_tokens([token])[0]

    def convert_to_strs(self, processor):
        # 只在「字符串缺失而 id 已给出」时反查，避免覆盖掉模型显式配置的写法
        # （同一个 token 在不同 tokenizer 下可能有多种等价字面量）。
        if not self.image_token:
            self.image_token = self.convert_to_str(self.image_token_id, processor)
        if not self.video_token:
            self.video_token = self.convert_to_str(self.video_token_id, processor)
        if not self.audio_token:
            self.audio_token = self.convert_to_str(self.audio_token_id, processor)

    def get_modality_of_token(self, token: str) -> Optional[Modality]:
        """
        :return: the modality associated with the given token, if the token is a special_token or matches with the multimodal token regex
        """
        # 先做 O(1) 的精确字符串比较（四个枚举项），再做正则兑底。
        # 兑底是必要的：某些模型的占位符是一个序列，如 "<image_1>"、"<image_2>"，
        # 字符串不相等但正则能匹配。
        modality = {
            self.image_token: Modality.IMAGE,
            self.video_token: Modality.VIDEO,
            self.audio_token: Modality.AUDIO,
        }.get(token)
        if modality:
            return modality

        for regex, modality in [
            (self.image_token_regex, Modality.IMAGE),
            (self.video_token_regex, Modality.VIDEO),
            (self.audio_token_regex, Modality.AUDIO),
        ]:
            if regex and regex.match(token):
                return modality

        return None

    def get_token_id_by_modality(self, modality: Modality) -> Optional[int]:
        # 反向映射：给定模态返回占位符 id，用于在 input_ids 里扫描偏移。
        return {
            Modality.IMAGE: self.image_token_id,
            Modality.VIDEO: self.video_token_id,
            Modality.AUDIO: self.audio_token_id,
        }.get(modality)

    def parse_regex(self):
        # re.escape 不可或缺：占位符里常有 "<|image_pad|>" 这类特殊字符，
        # 不转义会被当成正则语法。
        if self.image_token_regex is None and self.image_token is not None:
            self.image_token_regex = re.compile(re.escape(self.image_token))
        if self.video_token_regex is None and self.video_token is not None:
            self.video_token_regex = re.compile(re.escape(self.video_token))
        if self.audio_token_regex is None and self.audio_token is not None:
            self.audio_token_regex = re.compile(re.escape(self.audio_token))

    def get_combined_regex(self) -> re.Pattern:
        """
        Builds and returns a regex, used to split input str into tokens (with mm special tokens)
        """
        # 结果形如 "((?:<|image_pad|>)|(?:<|video_pad|>))"，外层的捕获组很关键：
        # ``re.split`` 只有带捕获组时才会把「分隔符本身」也保留在结果列表里，
        # 而 legacy_load_mm_data 恰恰需要拿到分隔符去判断模态。
        # flags 用 |= 合并，保证各子模式的行内标志（如 re.IGNORECASE）不丢。
        if self.combined_regex:
            return self.combined_regex
        tokens = [
            self.image_token_regex,
            self.video_token_regex,
            self.audio_token_regex,
        ]
        patterns = []
        flags = 0
        for t in tokens:
            if t is not None:
                patterns.append(t.pattern)
                flags |= t.flags
        combined = "(" + "|".join(f"(?:{p})" for p in patterns) + ")"
        self.combined_regex = re.compile(combined, flags)
        return self.combined_regex


def _tokenizer_of(processor):
    """The tokenizer reached from an HF processor.

    Some processors (e.g. InternVL) are handed a tokenizer directly as their
    ``_processor`` rather than one that wraps a tokenizer. Every path that
    resolves a tokenizer -- construction and per-worker processor clones alike --
    goes through here, so a clone cannot resolve differently from the original.

    中文小结：有 tokenizer 属性就用它，否则把 processor 自身当 tokenizer
    （InternVL 这种情况）。克隆出的 processor 也走同一入口，保证不会分化出
    两种解析结果。
    """
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def _forget_parent_executor_wakeups():
    # A forked worker inherits the parent's executor wakeups, whose locks a parent
    # thread may hold at fork time; since Python 3.12.8 / 3.13.1 the worker's exit
    # hook takes those locks, so the worker would never exit.
    # 中文：fork 出来的子进程会继承父进程的 executor 唤醒对象，而 fork 那一刻
    # 父进程的某个线程可能正持着里面那把锁。Python 3.12.8 / 3.13.1 起，子进程
    # 退出钩子会去获取这把锁 → 永远拿不到 → 进程无法退出（表现为进程池关不掉）。
    # 这里直接清掉继承来的表，让子进程自己重建。
    # 作为 ProcessPoolExecutor 的 initializer 调用。
    concurrent.futures.process._threads_wakeups.clear()


class BaseMultimodalProcessor(ABC):
    """所有模型多模态处理器的基类。

    职责边界（记住这三步，整份文件就清楚了）：
      1. load：把用户给的 URL/base64/路径 变成内存中的数据（``load_mm_data``）；
      2. process：调 HF processor 得到 pixel_values 等特征（``process_mm_data``）；
      3. combine：把特征切分/包装成 ``MultimodalDataItem`` 并算出 token 偏移
         （``process_and_combine_mm_data``）。

    子类通常只需：设置类属性（``models``、``IMAGE_TOKEN`` 等）+
    实现 ``process_mm_data_async``。只有行为特殊（MiniCPM-V 动态帧、GLM4V 等）
    才需要重写 load 或 combine 环节。

    下面这组类属性是「模型声明自己的默认策略」，用户命令行参数优先级更高：
      * ``gpu_image_decode``：是否用 nvJPEG 在 GPU 上解码 JPEG；
      * ``video_preprocessing_device``：视频预处理放哪个设备（长视频建议 "cpu"，
        避免与 scheduler 抢同一个 GPU 显存池）；
      * ``auto_mm_processor_worker_num``：并发预处理线程数的推荐值；
      * ``auto_mm_preprocess_cache_size_mb``：预处理缓存预算（MB）；
      * ``supports_mm_processor_concurrency``：预处理是否线程安全，
        不自洽的 processor 置 False 会退回单线程。
    """

    models = []
    gpu_image_decode = True  # Enable GPU decoding by default
    smart_rgb_conversion = False
    video_preprocessing_device = None
    prefer_tokenized_input = False
    precompute_hash_before_cpu_transfer = False
    # Set by processors that already build input_ids from the request's own
    # tokens, so the retokenize-avoidance rebuild below has nothing to add.
    preserve_processor_input_ids = False
    # None lets the worker count follow where preprocessing actually runs; a
    # model that measured its own optimum assigns a number instead. See
    # `_resolve_auto_mm_processor_worker_num`.
    auto_mm_processor_worker_num = None
    auto_mm_io_worker_num = 4
    # Models opt in by assigning a non-zero default. A user-provided server
    # argument overrides this value; zero disables storage and cache-key work.
    auto_mm_preprocess_cache_size_mb = 0
    # Artifact-based processors may keep their prompt/M-RoPE fast path even
    # when artifact retention is disabled.
    uses_media_artifacts_without_cache = False
    # Processors opt out only when their preprocessing is not thread-safe. The
    # worker pool gives each thread its own `copy.deepcopy` of the HF processor
    # and injects it, and the single function it runs --
    # `process_and_combine_mm_data` -- resolves that clone instead of
    # `self._processor`, so isolation does not depend on the subclass.
    supports_mm_processor_concurrency = True

    def __init__(
        self, hf_config, server_args, _processor, transport_mode, *args, **kwargs
    ):
        self.hf_config = hf_config
        self._processor = _processor
        self.server_args = server_args
        self.transport_mode = transport_mode
        configure_media_url_security(
            get_mm().allowed_media_domains,
            get_mm().media_url_max_file_size_mb,
        )
        # 特征回传通道：tokenizer 进程算出的 pixel_values 要交给 scheduler 所在
        # 的 GPU 进程。cpu 走普通 CPU tensor 拷贝；cuda_ipc / cuda_vmm 则用
        # 共享显存传递，省一次 H2D。值不可识别时保守回退到 cpu。
        configured_mm_feature_transport = get_mm().mm_feature_transport
        self.mm_feature_transport = (
            configured_mm_feature_transport
            if configured_mm_feature_transport in ("cpu", "cuda_ipc", "cuda_vmm")
            else "cpu"
        )
        self.use_cuda_ipc = self.mm_feature_transport == "cuda_ipc"
        self.use_ipc_pool_handle_cache = (
            self.use_cuda_ipc and envs.SGLANG_USE_IPC_POOL_HANDLE_CACHE.get()
        )
        self.image_processor_backend = get_mm().image_processor_backend
        if get_mm().disable_fast_image_processor:
            self.image_processor_backend = "pil"
        self.disable_fast_image_processor = self.image_processor_backend == "pil"
        self.skip_tokenizer_init = get_serving().skip_tokenizer_init

        mm_process_config = get_mm().mm_process_config
        self.image_config = mm_process_config.get("image", {})
        self.video_config = mm_process_config.get("video", {})
        self.audio_config = mm_process_config.get("audio", {})

        # 每个 tokenizer worker 是独立进程，各自持有一份 CPU 缓存。
        # 这里把「全局预算」按 worker 数均分，否则增加 worker 会无声地
        # 成倍抬高主机内存占用。
        # Each tokenizer worker is a separate process with its own CPU cache.
        # Split the requested service-wide budget so increasing worker count
        # does not silently multiply host-memory usage.
        requested_cache_mb = get_mm().mm_preprocess_cache_size_mb
        total_cache_mb = (
            self.auto_mm_preprocess_cache_size_mb
            if requested_cache_mb is None
            else requested_cache_mb
        )
        tokenizer_worker_num = max(int(get_serving().tokenizer_worker_num), 1)
        worker_cache_bytes = total_cache_mb * 1024 * 1024 // tokenizer_worker_num
        self.mm_preprocess_cache = MultimodalPreprocessCache(
            max_size_bytes=worker_cache_bytes,
            max_entries=8192,
        )
        self.trust_mm_content_hashes = bool(get_mm().trust_mm_content_hashes)
        # fingerprint（配置指纹）只用于构造 artifact 缓存 key。
        # 若本 processor 永远不会保留 artifact，就不去遍历 processor 状态，
        # 避免无意义的启动开销（有些 processor 的属性访问很重）。
        # The fingerprint is needed only to build artifact keys. Avoid inspecting
        # processor state when this processor will never retain artifacts.
        self.processor_fingerprint = (
            build_processor_fingerprint(self, hf_config)
            if self.mm_preprocess_cache.enabled
            or self.uses_media_artifacts_without_cache
            else None
        )
        if self.mm_preprocess_cache.enabled:
            logger.info(
                "Multimodal preprocess cache enabled for %s: %d MiB total "
                "(%d MiB per tokenizer worker), at most 8192 entries; "
                "caller content hashes are %s.",
                type(self).__name__,
                total_cache_mb,
                worker_cache_bytes // (1024 * 1024),
                "trusted" if self.trust_mm_content_hashes else "verified",
            )

        # 统一从 processor 解析 tokenizer（兼容"processor 本身就是 tokenizer"）。
        self._tokenizer = _tokenizer_of(self._processor)

        # Same guard as in serving_chat.py against double BOS.
        try:
            self._tokenizer_auto_adds_specials = len(self._tokenizer.encode("")) > 0
        except Exception:
            self._tokenizer_auto_adds_specials = False

        # FIXME: not accurate, model and image specific
        self.NUM_TOKEN_PER_FRAME = 330

        # IO 线程数优先级：命令行显式指定 > 环境变量 SGLANG_IO_WORKERS > 模型默认值。
        # 线程数 > 4 才打日志，避免默认配置下刷屏。
        requested_mm_io_worker_num = get_mm().mm_io_worker_num
        env_mm_io_worker_num = os.environ.get("SGLANG_IO_WORKERS")
        if requested_mm_io_worker_num:
            self.mm_io_worker_num = requested_mm_io_worker_num
            io_worker_source = "explicit"
        elif env_mm_io_worker_num is not None:
            self.mm_io_worker_num = int(env_mm_io_worker_num)
            io_worker_source = "environment"
        else:
            self.mm_io_worker_num = self.auto_mm_io_worker_num
            io_worker_source = "auto"
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.mm_io_worker_num,
            thread_name_prefix="sglang-mm-io",
        )
        self.hash_executor = MultimodalHashExecutor()
        if self.mm_io_worker_num > 4:
            logger.info(
                "Multimodal data loading enabled with %d worker threads (%s).",
                self.mm_io_worker_num,
                io_worker_source,
            )
        # 预处理并发数：显式配置优先，否则自动推导（见 _resolve_auto_mm_processor_worker_num）。
        # skip_mm_pool=True 一般用于「子进程内部又构造了一份 processor」的场景，
        # 再开池子只会套娃，因此强制为 1。
        skip_mm_pool = kwargs.get("skip_mm_pool", False)
        requested_mm_processor_worker_num = get_mm().mm_processor_worker_num
        self.mm_processor_worker_num = (
            1
            if skip_mm_pool
            else requested_mm_processor_worker_num
            or self._resolve_auto_mm_processor_worker_num()
        )
        if (
            self.mm_processor_worker_num > 1
            and not self.supports_mm_processor_concurrency
        ):
            logger.warning(
                "Concurrent multimodal processing is not supported by %s; "
                "using synchronous processing.",
                type(self).__name__,
            )
            self.mm_processor_worker_num = 1
        self.mm_processor_executor = None
        if self.mm_processor_worker_num > 1:
            try:
                # A callable, not the object: subclasses finish customizing
                # `_processor` after this returns, and the workers must clone it
                # as the subclass left it.
                self.mm_processor_executor = MultimodalProcessorExecutor(
                    lambda: self._processor, self.mm_processor_worker_num
                )
            except Exception:
                logger.warning(
                    "Unable to clone the multimodal processor for concurrent "
                    "workers; falling back to synchronous processing.",
                    exc_info=True,
                )
                self.mm_processor_worker_num = 1
        if self.mm_processor_executor is not None:
            logger.info(
                "Multimodal processor concurrency enabled with %d isolated "
                "worker threads (%s).",
                self.mm_processor_worker_num,
                "auto" if requested_mm_processor_worker_num == 0 else "explicit",
            )
        self._cpu_executor_lock = threading.Lock()
        self.cpu_executor = self._create_cpu_executor()

        # HF processor 输出的属性名 → 模态。
        # collect_mm_items_from_processor_output 靠它把 pixel_values、input_features
        # 等字段归到对应的 MultimodalDataItem 上；同名但属于不同模态的模型需在
        # 子类里覆盖这张表。
        # Mapping from attribute names to modality types
        self.ATTR_NAME_TO_MODALITY = {
            # Image-related attributes
            "pixel_values": Modality.IMAGE,
            "image_sizes": Modality.IMAGE,
            "image_grid_thw": Modality.IMAGE,
            "image_attention_mask": Modality.IMAGE,
            "image_emb_mask": Modality.IMAGE,
            "images_spatial_crop": Modality.IMAGE,
            "images_crop": Modality.IMAGE,
            "has_local_crops": Modality.IMAGE,
            "has_images": Modality.IMAGE,
            "tgt_size": Modality.IMAGE,
            "image_grid_hws": Modality.IMAGE,
            "aspect_ratio_ids": Modality.IMAGE,
            "aspect_ratio_mask": Modality.IMAGE,
            "num_patches": Modality.IMAGE,
            "patch_pixel_values": Modality.IMAGE,
            "block_sizes": Modality.IMAGE,
            "grid_thws": Modality.IMAGE,  # for kimi k2.5
            # Audio-related attributes
            "audio_features": Modality.AUDIO,
            "audio_feature_lens": Modality.AUDIO,
            "input_features": Modality.AUDIO,
            "input_features_mask": Modality.AUDIO,
            "audio_attention_mask": Modality.AUDIO,
            "feature_attention_mask": Modality.AUDIO,
            # Video-related attributes
            "pixel_values_videos": Modality.VIDEO,
            "second_per_grid_ts": Modality.VIDEO,
            "video_grid_thw": Modality.VIDEO,
            # Generic attributes that could apply to multiple modalities
            # "precomputed_embeddings" - handled specially as it can be any modality
        }

        # 哪些字段算「特征张量」（需要走 transport，可能在 GPU/CPU 间搬运）。
        # 命名保留原拼写 filed，不改动以避免无关 diff。
        # name of the feature filed
        # TODO: pass from processors
        self.FEATURE_NAMES = [
            "pixel_values",
            "pixel_values_videos",
            "audio_features",
            "input_features",
        ]

        # 只有走 cuda_ipc 时才需要在 GPU 上预留一块特征内存池。
        # 预算同样是「全局总量」，按 worker 数均分，避免多 worker 叠加占显存。
        if self.use_cuda_ipc and not skip_mm_pool:
            # SGLANG_MM_FEATURE_CACHE_MB is the total pool budget across all
            # tokenizer workers. Each worker gets an equal share so that adding
            # workers doesn't multiply the GPU-side footprint.
            worker_num = get_serving().tokenizer_worker_num
            per_worker_pool_size = get_mm_feature_pool_size_per_worker(
                MM_FEATURE_CACHE_SIZE, worker_num
            )
            total_pool_size = per_worker_pool_size * worker_num
            logger.info(
                "CUDA IPC multimodal feature pools reserve %.0f MiB total on "
                "GPU %d (%.0f MiB per tokenizer worker × %d; configured "
                "budget %.0f MiB).",
                total_pool_size / (1024 * 1024),
                self.server_args.base_gpu_id,
                per_worker_pool_size / (1024 * 1024),
                worker_num,
                MM_FEATURE_CACHE_SIZE / (1024 * 1024),
            )
            self.cudaipc_mmfeature_pool = MmItemMemoryPool(
                per_worker_pool_size,
                MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
                self.server_args.base_gpu_id,
                get_mm_feature_consumer_count(),
            )

    @property
    def keep_mm_features_on_device(self) -> bool:
        """Whether feature transport expects processor outputs to stay on GPU.

        中文：若特征要经 GPU 共享内存回传，就不能在 process 完又把 tensor 扫到
        CPU（否则白跑一趟 H2D→D2H）。这个属性就是各处抽 tensor 代码的统一判据。
        """
        return self.mm_feature_transport in ("cuda_ipc", "cuda_vmm")

    def preprocess_fingerprint_payload(self) -> dict[str, Any]:
        """Return every stable setting that can change a media artifact.

        The payload is hashed once at startup and becomes part of every
        artifact key. Model processors must extend this method when they add an
        output-affecting option. The wrapped HF processor can expose its own
        typed payload through ``PreprocessFingerprintProvider``.
        """
        # 除本包装层的选项外，还要把内部 HF processor 自己声明的指纹合并进来
        # （即它实现了 PreprocessFingerprintProvider 接口时）。
        # 漏加会导致：改了预处理超参但缓存 key 不变 → 静默命中旧 artifact。
        wrapped_processor = (
            self._processor.preprocess_fingerprint_payload()
            if isinstance(self._processor, PreprocessFingerprintProvider)
            else None
        )
        return {
            "wrapper_class": (
                f"{type(self._processor).__module__}."
                f"{type(self._processor).__qualname__}"
            ),
            "gpu_image_decode": self.gpu_image_decode,
            "image_processor_backend": self.image_processor_backend,
            "feature_transport": self.mm_feature_transport,
            "image_config": self.image_config,
            "video_config": self.video_config,
            "audio_config": self.audio_config,
            "wrapped_processor": wrapped_processor,
        }

    def clear_preprocess_cache(self) -> None:
        """Drop artifacts and reject cache writes from pre-flush work.

        Active requests continue and still receive their preprocessing result;
        they simply cannot repopulate the freshly cleared cache.

        中文：清空缓存但不打断在途请求。在途请求仍会拿到自己的预处理结果，
        只是不能把结果又写回刚被清空的缓存（避免 flush 后旧条目立即复活）。
        """
        self.mm_preprocess_cache.clear()

    def shutdown(self) -> None:
        """Drop cached artifacts and stop every processor-side executor."""
        # 关闭顺序：先清缓存（避免关闭过程中还有写入），再依次关哈希/IO/CPU/预处理池。
        # IO 与 CPU 池用 wait=False + cancel_futures=True：不再等待已排队的任务，
        # 因为此时服务正在退出，只求尽快释放资源。
        self.clear_preprocess_cache()
        self.hash_executor.shutdown()
        self.io_executor.shutdown(wait=False, cancel_futures=True)
        self.cpu_executor.shutdown(wait=False, cancel_futures=True)
        if self.mm_processor_executor is not None:
            self.mm_processor_executor.shutdown()

    def _create_cpu_executor(self) -> concurrent.futures.ProcessPoolExecutor:
        # 为何 cuda_vmm 必须用 spawn：fork 会继承父进程已初始化的 CUDA 上下文，
        # 而 CUDA 不允许子进程复用父进程的 context（重初始化会报错）。
        # 其它模式用 fork：启动快，且能直接共享已加载的 tokenizer 内存。
        start_method = "spawn" if self.mm_feature_transport == "cuda_vmm" else "fork"
        return concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context(start_method),
            max_workers=int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),
            initializer=_forget_parent_executor_wakeups,
        )

    def _replace_broken_cpu_executor(
        self, failed_executor: concurrent.futures.ProcessPoolExecutor
    ) -> None:
        """Replace a failed preprocess pool once across concurrent requests.

        中文：进程池可能因子进程被 OOM kill 而整体报废。并发请求会同时发现这点，
        用锁 + 身份比对（``is not failed_executor``）保证只重建一次，其余请求直接
        复用新池；旧池的关闭扔到后台守护线程，不阻塞当前请求。
        """
        with self._cpu_executor_lock:
            if self.cpu_executor is not failed_executor:
                return
            self.cpu_executor = self._create_cpu_executor()
        logger.warning("Replaced a broken multimodal CPU preprocess pool")
        threading.Thread(
            target=self._shutdown_broken_cpu_executor,
            args=(failed_executor,),
            name="sglang-mm-cpu-pool-cleanup",
            daemon=True,
        ).start()

    @staticmethod
    def _shutdown_broken_cpu_executor(
        failed_executor: concurrent.futures.ProcessPoolExecutor,
    ) -> None:
        try:
            failed_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            logger.warning(
                "Failed to shut down a broken multimodal CPU preprocess pool",
                exc_info=True,
            )

    def compute_mrope_positions(self, input_ids, mm_items):
        """Compute M-RoPE positions from expanded input_ids and multimodal items.

        Returns (mrope_positions, mrope_position_delta) or (None, None) if the
        model does not use M-RoPE.

        中文：M-RoPE（Qwen2-VL 系）对文本/高/宽三个维度分别维护位置，
        图像对应的 token 要据其 grid 形状重新排列位置。基类默认返回 (None, None)
        表示不需要；需要该能力的子类覆盖并给出三个分量。
        """
        return None, None

    @property
    def spatial_merge_size(self):
        # 视觉侧 patch 合并边长（如 Qwen-VL 为 2）：一个视觉 token 对应
        # spatial_merge_size² 个 patch，因此「token 数 = patch 数 / size²」。
        # 这个换算在 build_input_ids 里反复出现。
        return self.hf_config.vision_config.spatial_merge_size

    def build_input_ids(
        self, prompt, img_grid_thw=None, video_grid_thw=None, audio_seq_lens=None
    ):
        """
        Use prompt, img_grid_thw, video_grid_thw, and audio_seq_lens to build input_ids.
        Supports image, video, and audio tokens.

        中文：这是「模型自己定义了 IM_TOKEN_ID / VIDEO_TOKEN_ID」时用的手拼实现，
        不依赖 HF processor 返回的 input_ids。做法：
          1. 先扫一遍 prompt，找到形如 <im_start><mm_token> 的位置，记录其模态；
          2. 对每处按 grid_thw 算出应展开的 token 数 N，把那个 mm_token 换成 N 个；
          3. 同时记录每个 item 在最终 input_ids 中的 [start, end] 与模态顺序。
        这里的 offsets 与 modality_list 会被 get_mm_data 用来切分预计算好的 embedding，
        两边必须严格一致，否则 EPD（encode-prefill-decode）会行数对不上。
        """
        if not isinstance(prompt, list):
            prompt = self._tokenizer.encode(prompt)

        img_token_id = getattr(self, "IM_TOKEN_ID", None)
        video_token_id = getattr(self, "VIDEO_TOKEN_ID", None)
        audio_token_id = getattr(self, "audio_token_id", None)
        spatial_merge_size = getattr(self, "spatial_merge_size", 1)

        input_ids = []
        offsets = []

        cur_idx = 0

        # Use img_token_id instead of im_start_id, because a dummy im_start_id
        # may be generated by the tokenizer.
        # 中文：不能拿 im_start_id 当哨兵——tokenizer 可能凭空插一个假的 im_start。
        # 改成「看下一个 token 是不是 mm 占位符」，更稳。
        # 注意这里是向前看一位（i+1），所以循环只跑到 len(prompt)-1。
        vision_start_indices = []
        for i in range(len(prompt) - 1):
            if img_token_id is not None and prompt[i + 1] == img_token_id:
                vision_start_indices.append((i, Modality.IMAGE))
            elif video_token_id is not None and prompt[i + 1] == video_token_id:
                vision_start_indices.append((i, Modality.VIDEO))
            elif audio_token_id is not None and prompt[i + 1] == audio_token_id:
                vision_start_indices.append((i, Modality.AUDIO))
        # get modality list with order preserved
        modality_list = [modality for _, modality in vision_start_indices]

        img_idx = 0
        video_idx = 0
        audio_idx = 0
        for mm_start_idx, modality in vision_start_indices:
            if modality == Modality.IMAGE:
                # grid_thw = (t, h, w) 是「patch 数」，除 spatial_merge_size²
                # 才是实际视觉 token 数。
                mm_token_num = img_grid_thw[img_idx].prod() // (spatial_merge_size**2)
                mm_token_id = img_token_id
                img_idx += 1
            elif modality == Modality.VIDEO:
                mm_token_num = video_grid_thw[video_idx].prod() // (
                    spatial_merge_size**2
                )
                mm_token_id = video_token_id
                video_idx += 1
            elif modality == Modality.AUDIO:
                mm_token_num = int(audio_seq_lens[audio_idx].item())
                mm_token_id = audio_token_id
                audio_idx += 1
            else:
                raise ValueError(f"Invalid modality: {modality}")
            assert cur_idx <= mm_start_idx

            input_ids.extend(prompt[cur_idx : mm_start_idx + 1])
            mm_offset_start = len(input_ids)
            input_ids.extend([mm_token_id] * mm_token_num)
            # +2 是为了跳过 <im_start> 和旧的那个 mm_token，直接落到 end 标记；
            # 旧的 mm_token 已在上一步被【重写为 N 个】。
            cur_idx = (
                mm_start_idx + 2
            )  # jump to img_end_id, video_end_id, or audio_end_id
            offsets.append((mm_offset_start, len(input_ids) - 1))
        else:
            input_ids.extend(prompt[cur_idx:])

        return input_ids, offsets, modality_list

    def get_mm_data(self, prompt, embeddings, **kwargs):
        # EPD 场景：embedding 已由 encoder 算好，这里只负责「重建 prompt 布局 +
        # 把每个模态的 embedding 按 offsets 切成一段一段，包成 MultimodalDataItem」。
        # consumed_per_modality 记录每种模态已经用掉多少行，保证按 prompt 顺序
        # 消费、不多不少。
        img_grid_thw = kwargs.get("img_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        audio_feature_lens = kwargs.get("audio_feature_lens", None)

        input_ids, offsets, modality_list = self.build_input_ids(
            prompt,
            img_grid_thw=img_grid_thw,
            video_grid_thw=video_grid_thw,
            audio_seq_lens=audio_feature_lens,
        )
        assert all(isinstance(modality, Modality) for modality in modality_list)

        mm_items = []
        consumed_per_modality = {}

        for modality, offset in zip(modality_list, offsets):
            num_tokens = offset[1] - offset[0] + 1
            embedding_start = consumed_per_modality.get(modality, 0)
            embedding_slice = embeddings[modality][
                embedding_start : embedding_start + num_tokens
            ]
            consumed_per_modality[modality] = embedding_start + num_tokens
            mm_items.append(
                MultimodalDataItem(
                    modality=modality,
                    offsets=[offset],
                    precomputed_embeddings=embedding_slice,
                )
            )

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.IM_TOKEN_ID,
            video_token_id=getattr(self, "VIDEO_TOKEN_ID", None),
        )

    def get_validated_mm_data(
        self,
        prompt,
        embeddings: Dict[Modality, torch.Tensor],
        **kwargs,
    ) -> MultimodalProcessorOutput:
        """Build EPD multimodal inputs and validate the embedding layout.

        Model processors may override ``get_mm_data`` to rebuild their prompt
        layout. This shared wrapper ensures every override consumes exactly the
        encoder rows it received before the result reaches the scheduler.

        中文：子类可以重写 get_mm_data 自定义排版，但入口统一包一层校验：
        每个 item 持有的行数必须等于 offsets 推出的行数，且各模态总消费行数
        必须等于 encoder 给的行数。不满足就报错——宁可在 CPU 侧快速失败，
        也不要让形状错位流到 GPU 变成难查的非法内存访问。
        """
        output = self.get_mm_data(prompt, embeddings, **kwargs)
        self._validate_precomputed_embedding_layout(output, embeddings)
        return output

    @staticmethod
    def _validate_precomputed_embedding_layout(
        output: MultimodalProcessorOutput,
        embeddings: Dict[Modality, torch.Tensor],
    ) -> None:
        # 校验分两步：先逐 item 对「自身 offsets 与行数」，再对「每个模态的总账」。
        consumed_per_modality = {modality: 0 for modality in embeddings}

        for item in output.mm_items:
            embedding = item.precomputed_embeddings
            if not isinstance(embedding, torch.Tensor):
                raise RuntimeError(
                    "EPD multimodal items must contain tensor embeddings; "
                    f"got {type(embedding).__name__} for "
                    f"{item.modality.name.lower()}"
                )

            num_rows = embedding.shape[0]
            if item.offsets is not None:
                expected_rows = sum(end - start + 1 for start, end in item.offsets)
                if num_rows != expected_rows:
                    raise RuntimeError(
                        "Precomputed multimodal embedding length mismatch for "
                        f"{item.modality.name.lower()}: expected {expected_rows} "
                        f"rows from prompt offsets, got {num_rows}"
                    )

            if item.modality not in consumed_per_modality:
                raise RuntimeError(
                    "EPD processor returned an unexpected embedding modality: "
                    f"{item.modality.name.lower()}"
                )
            consumed_per_modality[item.modality] += num_rows

        for modality, embedding in embeddings.items():
            if not isinstance(embedding, torch.Tensor):
                raise RuntimeError(
                    "EPD encoder output must contain tensor embeddings; "
                    f"got {type(embedding).__name__} for {modality.name.lower()}"
                )
            consumed_rows = consumed_per_modality[modality]
            if consumed_rows != embedding.shape[0]:
                raise RuntimeError(
                    "Precomputed multimodal embedding consumption mismatch for "
                    f"{modality.name.lower()}: received {embedding.shape[0]} rows, "
                    f"consumed {consumed_rows}"
                )

    def _resolve_processor(self, processor=None):
        # processor 参数存在的原因是并发 worker：每个线程拿到的是自己那份
        # deepcopy 的 processor，必须用它而不是 self._processor，否则又回到共享状态。
        if processor is None:
            return self._processor, self._tokenizer
        return processor, _tokenizer_of(processor)

    def _preprocessing_competes_with_the_scheduler(self) -> bool:
        """Whether image preprocessing contends with the serving accelerator.

        The fast image processor runs inside the tokenizer process but may run on
        the same accelerator as the scheduler. A second preprocessing worker there
        adds device contention rather than CPU preprocessing parallelism.

        中文：判断「预处理会不会和 scheduler 抢同一张卡」。fast image processor
        在 tokenizer 进程里跑，但可以指定跑在 GPU 上；若真是 GPU 路径，再开一个
        worker 只会抢显存/算力，不会带来真并行（已实测 Qwen-VL 在 GB300 上
        9.30 → 4.02 req/s 的倒退）。
        """
        if _is_cpu or get_exec().deterministic.rl_on_policy_target is not None:
            return False
        if self.disable_fast_image_processor:
            return False
        image_processor = getattr(self._processor, "image_processor", None)
        if not isinstance(image_processor, BaseImageProcessor):
            return False
        if _is_xpu or _is_npu:
            return True
        platform = platforms.current_platform
        return platform.is_cuda_alike()

    def _resolve_auto_mm_processor_worker_num(self) -> int:
        """The worker count to use when the user did not ask for one.

        Two workers overlap preprocessing that runs on the CPU, where the second
        thread is real parallelism: measured on Qwen2.5-VL with full-page images
        at 32-way concurrency, 4.46 -> 6.08 req/s on H200 and 7.07 -> 8.76 on
        GB300.

        The GPU path is capped at one worker even when a model declares more.
        A declaration records what its author measured on one platform and one
        image shape; contending for the device the scheduler is serving from is a
        property of the path itself, and it does not go away because a subclass
        asked for concurrency. Qwen-VL declares two and is the model that
        measures 9.30 -> 4.02 req/s on GB300 full-page images, so honouring the
        declaration here would exempt exactly the case that regresses.
        `--mm-processor-worker-num` still overrides this.

        中文：未显式指定时的默认取值。
          * 若预处理会占用 scheduler 的加速卡 → 强制 1；
          * 否则 CPU 路径下默认 2（第二个线程是真并行，实测 H200 上
            4.46 → 6.08 req/s）；
          * 模型自己声明了数字就用声明的。
        """
        if self._preprocessing_competes_with_the_scheduler():
            return 1
        declared = self.auto_mm_processor_worker_num
        return 2 if declared is None else declared

    def _fast_image_processor_device(self, processor) -> Optional[str]:
        """The device for the fast image processor, or None to leave it unset.

        Resolved from this processor's own ``server_args``: engines sharing a
        tokenizer process each carry their own ``base_gpu_id``.

        中文：决定 fast image processor 跑在哪个设备上，返回 None 表示不传
        device（让它自己选）。注意必须用 self.server_args 里的 base_gpu_id，
        因为一个 tokenizer 进程可能同时服务多个 engine，每个 engine 的卡号不同。
        NPU 分支里的 patch 都是为了绕开异腾对张量维度的限制。
        """
        server_args = self.server_args
        if _is_cpu or get_exec().deterministic.rl_on_policy_target is not None:
            return "cpu"
        if _is_xpu:
            return "xpu"
        if not _is_npu:
            platform = platforms.current_platform
            if not platform.is_cuda_alike():
                return None
            # Per-worker placement travels as a constructor argument, and
            # this record is that argument.
            return f"{platform.device_type}:{server_args.base_gpu_id}"
        if processor.__class__.__name__ == "MiniMaxVLProcessor":
            # MiniMax's image/video processors create 10-dim tensors during
            # patch extraction, exceeding the Ascend 8-dim limit; patch them
            # (same pattern as qwen-vl / GLM-4.6V) and run on NPU.
            from sglang.srt.hardware_backend.npu.modules.minimax_m3_processor import (
                npu_apply_minimax_m3_image_preprocess_patch,
                npu_apply_minimax_m3_video_preprocess_patch,
            )

            npu_apply_minimax_m3_image_preprocess_patch(processor.image_processor)
            if (
                hasattr(processor, "video_processor")
                and processor.video_processor is not None
            ):
                npu_apply_minimax_m3_video_preprocess_patch(processor.video_processor)
            return "npu"
        if processor.__class__.__name__ not in {"Glm4vProcessor", "Glm46VProcessor"}:
            # For qwen-vl, the processor hits a reshape issue from the Ascend
            # dims restriction.
            from sglang.srt.hardware_backend.npu.modules.qwen_vl_processor import (
                npu_apply_qwen_image_preprocess_patch,
            )

            npu_apply_qwen_image_preprocess_patch()
            return "npu"
        if processor.__class__.__name__ == "Glm46VProcessor":
            from sglang.srt.hardware_backend.npu.modules.glm46v_processor import (
                npu_apply_glm46v_image_preprocess_patch,
            )

            npu_apply_glm46v_image_preprocess_patch()
            return "npu"
        return None

    @contextmanager
    def _temporary_fast_processor_cuda_pool(self, device: Optional[str]):
        """Release fast-processor CUDA temporaries after CPU feature transport.

        中文：如果结果是走 CPU 回传的（不需要留在显存），那么 fast image processor
        在 GPU 上产生的中间临时张量就是纯浪费显存。这里用一个临时 CUDA 内存池把
        这段推理圈起来，退出 with 时整池释放，不用逐块等 refcount。
        两个例外：特征必须留在 GPU（keep_mm_features_on_device）、
        或需要先算 hash 再下 CPU（precompute_hash_before_cpu_transfer）。
        """
        can_release = (
            device is not None
            and torch.device(device).type == "cuda"
            and not self.keep_mm_features_on_device
            and not self.precompute_hash_before_cpu_transfer
        )
        if not can_release:
            yield
            return

        with torch.cuda.device(device):
            pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool, device=device):
            yield

    def process_mm_data(
        self,
        input_text,
        images=None,
        videos=None,
        audios=None,
        processor=None,
        processor_video_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> dict:
        """
        process multimodal data with transformers AutoProcessor

        中文：真正调用 HF processor 的地方，输入是「文本 + 原始数据」，
        输出是 processor 的 BatchFeature（含 input_ids、pixel_values 等）。
        这里做了四件事：拼 kwargs、决定算力设备、避免重复 BOS、把特征搬回 CPU。
        注意：图像/音频参数名在不同 processor 上不一致（audios vs audio、
        是否传 truncation），下面的特例名单就是为了适配这种历史差异。
        """
        processor, tokenizer = self._resolve_processor(processor)

        if images:
            kwargs["images"] = images
            if self.image_config:
                kwargs.setdefault("images_kwargs", {}).update(self.image_config)
        if videos:
            kwargs["videos"] = videos
            video_config = (
                self.video_config
                if processor_video_config is None
                else processor_video_config
            )
            if video_config:
                kwargs.setdefault("videos_kwargs", {}).update(video_config)
        if audios:
            if processor.__class__.__name__ in {
                "Gemma3nProcessor",
                "Gemma4Processor",
                "Gemma4UnifiedProcessor",
                "GlmAsrProcessor",
                "GraniteSpeechProcessor",
                "Qwen2AudioProcessor",
                "Qwen3ASRProcessor",
                "Qwen3OmniMoeProcessor",
            }:
                # Note(Xinyuan): for gemma3n, ref: https://github.com/huggingface/transformers/blob/ccf2ca162e33f381e454cdb74bf4b41a51ab976d/src/transformers/models/gemma3n/processing_gemma3n.py#L107
                kwargs["audio"] = audios
                kwargs.setdefault("audio_kwargs", {})
                kwargs["audio_kwargs"].setdefault("truncation", False)
            else:
                kwargs["audios"] = audios
            if self.audio_config:
                kwargs.setdefault("audio_kwargs", {}).update(self.audio_config)

        processor_device = None
        if (
            hasattr(processor, "image_processor")
            and isinstance(processor.image_processor, BaseImageProcessor)
            and not self.disable_fast_image_processor
        ):
            processor_device = self._fast_image_processor_device(processor)
            if processor_device is not None:
                kwargs["device"] = processor_device

        # Long-video preprocessing stays on CPU to avoid competing with scheduler GPU pools.
        # 中文：视频预处理只做 CPU 侧时也显式传 device="cpu"，防止上游默认值
        # 把它又拉回 GPU（长视频动辄几百帧，抢显存后果很严重）。
        if videos and self.video_preprocessing_device is not None:
            kwargs["device"] = self.video_preprocessing_device

        # Avoid double BOS when the chat template already wrote one.
        # 中文：chat template 渲染完通常已自带 BOS，若 tokenizer 还会自动再加
        # 一个，就会变成 <s><s>，影响生成质量。这里只在「文本确实以 BOS 开头」时
        # 关掉自动添加，而不是无脑关。
        if self._tokenizer_auto_adds_specials and isinstance(input_text, str):
            bos = getattr(tokenizer, "bos_token", None)
            if bos and input_text.startswith(bos):
                kwargs.setdefault("add_special_tokens", False)

        with self._temporary_fast_processor_cuda_pool(processor_device):
            result = processor.__call__(
                text=[input_text],
                padding=True,
                return_tensors="pt",
                **kwargs,
            )
            # Deferred: the hash is computed on the GPU tensor first, and
            # _precompute_hashes_before_cpu_transfer moves it down afterwards.
            # 中文：两种延迟策略——若需要算 hash，就先把 GPU tensor 留着，等需要
            # hash 时先算再下 CPU（顺序不能反，否则 hash 要针对 CPU 副本重算）；
            # 若特征要留在 GPU，则根本不应该拷。两种都不成立时才在这里 D2H。
            if (
                not self.keep_mm_features_on_device
                and not self.precompute_hash_before_cpu_transfer
            ):
                # move feature tensors to cpu
                for feature_name in self.FEATURE_NAMES:
                    if feature_name in result and isinstance(
                        result[feature_name], torch.Tensor
                    ):
                        result[feature_name] = result[feature_name].to("cpu")

        return result

    @abstractmethod
    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        pass

    def get_estimated_frames_list(self, image_data):
        """
        estimate the total frame count from all visual input

        中文：预估每个输入会展开成多少帧。图片固定 1 帧；视频需要打开容器读
        帧数（``video:`` 前缀是内部约定）。这个预估用于提前决定采样上限，
        避免一次把几分钟的视频全部解码进内存。
        """
        from sglang.srt.utils.video_decoder import VideoDecoderWrapper

        # Before processing inputs
        if not image_data or len(image_data) == 0:
            return []
        estimated_frames_list = []
        for image in image_data:
            if isinstance(image, str) and image.startswith("video:"):
                path = image[len("video:") :]
                decoder = VideoDecoderWrapper(path)
                num_frames = len(decoder)
            else:
                # For images, each contributes one frame
                num_frames = 1
            estimated_frames_list.append(num_frames)

        return estimated_frames_list

    @classmethod
    def _load_single_item(
        cls,
        data,
        modality: Modality,
        frame_count_limit=None,
        audio_sample_rate: Optional[int] = None,
        discard_alpha_channel=True,
    ):
        """
        Load a single multimodal data.

        If data is processor_output or precomputed embedding, return directly.

        Class method that can be pickled for multiprocessing

        中文：一个 item 的取数据入口，跑在 IO 线程池/进程池里，所以：
          * 必须是 ``classmethod``（或模块级函数）——实例方法会连带 self 一起
            被打包，而 self 里有线程池/锁，pickle 不过去；
          * 不能碰 self 的任何状态，所以参数里的开关（gpu_image_decode、
            smart_rgb_conversion）都从 cls 读；
          * 已经是「预处理过的 dict」时直接透传，不重复 load。
        异常分两类：客户端数据问题（URL 404、格式不支持）报 ValueError，
        服务端问题（OOM、解码器内部错误）报 RuntimeError，便于上层区分重试策略。
        """
        if cls._is_preprocessed_input(data):
            return data
        try:
            if modality == Modality.IMAGE:
                # load_image 内部会先试 nvJPEG（GPU）再回退 PIL；
                # 第二个返回值是原始格式信息，这里用不到。
                img, _ = load_image(data, cls.gpu_image_decode)
                if isinstance(img, torch.Tensor):
                    return img  # JPEG already decoded on GPU by nvJPEG
                # PIL decodes lazily; do it here in the io worker so the decode
                # doesn't run later on the event-loop thread.
                if discard_alpha_channel:
                    if cls.smart_rgb_conversion:
                        return smart_to_rgb(img)
                    if img.mode != "RGB":
                        return img.convert("RGB")
                # 显式 load()：PIL 打开文件时是惰性的（只读 header），像素数据还在
                # 文件句柄里。这里主动把像素解码进内存并关闭文件句柄，目的是：
                #   1) 把解码开销留在 IO worker 里，不让它落到事件循环线程上；
                #   2) 避免后续线程/进程使用时才发现文件被删/被截断；
                #   3) PIL 对象脱离源文件后可以安全跨线程传递。
                img.load()
                return img
            elif modality == Modality.VIDEO:
                return load_video(data, use_gpu=cls.video_preprocessing_device != "cpu")
            elif modality == Modality.AUDIO:
                return load_audio(data, audio_sample_rate)

        except CLIENT_MEDIA_EXCEPTIONS as e:
            data_str = str(data)
            if len(data_str) > 100:
                data_str = data_str[:100] + "..."
            raise ValueError(f"Error while loading data {data_str}: {e}") from e
        except Exception as e:
            data_str = str(data)
            if len(data_str) > 100:
                data_str = data_str[:100] + "..."
            raise RuntimeError(f"Error while loading data {data_str}: {e}") from e

    @staticmethod
    def _get_preprocessed_input_format(data):
        """returns the detailed format if the provided data is already preprocessed.
        returns none if the provided data is not preprocessed

        中文：识别离线 API 传入的「免预处理」载荷。两种取值形式都认：
        ``MultimodalInputFormat`` 枚举本身，或者它的字符串名字
        （客户端通常只能传 JSON，所以字符串形式必须兼容）。
        """
        if not isinstance(data, dict):
            return None
        data_format = data.get("format")
        if isinstance(data_format, MultimodalInputFormat):
            return data_format
        if data_format in (
            MultimodalInputFormat.PROCESSOR_OUTPUT.name,
            "processor_output",
        ):
            return MultimodalInputFormat.PROCESSOR_OUTPUT
        if data_format in (
            MultimodalInputFormat.PRECOMPUTED_EMBEDDING.name,
            "precomputed_embedding",
        ):
            return MultimodalInputFormat.PRECOMPUTED_EMBEDDING
        return None

    @classmethod
    def _is_preprocessed_input(cls, data):
        """returns if the data is already preprocessed (by the vlm processor)

        中文：整个 load 链路里被反复调用的「短路判据」。
        """
        return cls._get_preprocessed_input_format(data) is not None

    @classmethod
    def _all_mm_data_is_preprocessed(cls, *data_lists):
        # 判断「本次请求是否完全不需要 load」：每个非空模态列表里的每一项
        # 都得是预处理载荷；且至少要有一项（纯文本请求不应走这条捷径）。
        # None 元素直接跳过，因为调用方常把 [None] 当成「该位置无数据」的占位。
        has_mm_data = False
        for data_list in data_lists:
            if not data_list:
                continue
            if not isinstance(data_list, list):
                data_list = [data_list]
            for item in data_list:
                if item is None:
                    continue
                has_mm_data = True
                if not cls._is_preprocessed_input(item):
                    return False
        return has_mm_data

    def _submit_mm_data_loading_tasks_simple(
        self,
        data_list: Optional[list],
        modality: Modality,
        audio_sample_rate: Optional[int],
        discard_alpha_channel: bool,
    ) -> List[Tuple[Modality, int, concurrent.futures.Future]]:
        """
        Simple version: For one modal data submit IO load task.
        Return:
            List[(modality, index_in_that_modality, future)]

        中文：fast 路径用。把「一个模态的所有 item」全部丢进 IO 线程池，
        并把【下标】一起带回，以便结果按原顺序回填（而不是按完成先后）；
        这里的 frame_count_limit 传 None——fast 路径不做帧数截断。
        """
        futures: List[Tuple[Modality, int, concurrent.futures.Future]] = []

        if not data_list:
            logger.debug(
                "[_submit_mm_data_loading_tasks_simple] no data for modality=%s",
                modality.name,
            )
            return futures

        for idx, data in enumerate(data_list):
            logger.debug(
                "[_submit_mm_data_loading_tasks_simple] submit load task: "
                "modality=%s, index=%d, data_type=%s",
                modality.name,
                idx,
                type(data),
            )
            future = self.io_executor.submit(
                self.__class__._load_single_item,
                data,
                modality,
                None,  # frame_count_limit: no consider for fast path
                audio_sample_rate,
                discard_alpha_channel,
            )
            futures.append((modality, idx, future))

        return futures

    def submit_data_loading_tasks(
        self,
        text_parts: List[str],
        multimodal_tokens: MultimodalSpecialTokens,
        data_iterators: dict[Modality, Iterator[Any]],
        discard_alpha_channel: bool = True,
        image_estimated_frames_iter: Optional[iter] = None,
        image_scaling_factor: float = 1.0,
        max_image_frames: int = 30,
        audio_sample_rate: Optional[int] = None,
    ) -> Tuple[List, List]:
        """
        load multimodal data parallelly using iterators.

        中文：legacy 路径用。核心技巧是——**用 prompt 切分得到的占位符列表作为
        调度表**，逐项从对应模态的迭代器 next() 取数据并提交 IO 任务。
        于是 futures 的顺序严格对应 text_parts 中占位符的顺序，后续消费时
        两个迭代器同步 next() 即可，不需要额外记录下标。

        两个方向的个数不匹配都在这里报：
          * 占位符多于数据：next() 抛 StopIteration → 打 warn 并直接返回，
            调用方会拿短列表，最终在拼接阶段暴露问题；
          * 数据多于占位符：循环结束后再 next() 一次探测，能取到就说明剩余
            数据永远用不上，同样只 warn（容忍多余输入，不报错）。
        """
        futures = []
        task_info = []

        for text_part in text_parts:
            modality = multimodal_tokens.get_modality_of_token(text_part)
            if modality is not None:
                data_iterator = data_iterators.get(modality)
                if data_iterator is None:
                    raise ValueError(f"No data iterator found for token: {text_part}")

                try:
                    data = next(data_iterator)
                except StopIteration:
                    logger.warning(
                        f"Mismatch: More '{modality.name}' tokens found than corresponding data provided."
                    )
                    return futures, task_info

                # 每张图可展开的帧数上限 = 预估帧数 × 缩放系数，至少 1。
                # 注意这不是「实际解码帧数」，实际帧数由解码器按上限截断后返回。
                frame_count_limit = None
                if modality == Modality.IMAGE and image_estimated_frames_iter:
                    try:
                        estimated_frames = next(image_estimated_frames_iter)
                        # Use the pre-calculated scaling factor and max frames
                        frame_count_limit = max(
                            1, int(estimated_frames * image_scaling_factor)
                        )
                        # Ensure we don't exceed the absolute max (redundant if scaling_factor handles it)
                        # frame_count_limit = min(frame_count_limit, max_image_frames)
                    except StopIteration:
                        raise ValueError(
                            "Mismatch between image tokens and estimated frame counts."
                        )

                futures.append(
                    self.io_executor.submit(
                        self.__class__._load_single_item,
                        data,
                        modality,
                        frame_count_limit,
                        audio_sample_rate,
                        discard_alpha_channel,
                    )
                )
                task_info.append((modality, data, frame_count_limit))

        for modality, iterator in data_iterators.items():
            try:
                next(iterator)
                logger.warning(
                    f"Warning: More {modality.name.lower()} data items provided than corresponding tokens found in the prompt."
                )
            except StopIteration:
                pass
            except Exception:
                pass

        return futures, task_info

    @staticmethod
    def _validate_one_modality(modality: Modality, data_list: Optional[list]):
        # 单个模态的校验规则：要么整份列表都是「原始数据」，
        # 要么列表长度为 1 且那唯一一项是预处理载荷。
        # 不允许「预处理载荷 + 原始图」混合，因为后者的 offset 计算方式完全不同，
        # 混在一起无法保证 input_ids 与 mm_items 对齐。
        if data_list is None:
            return
        if not isinstance(data_list, list):
            raise TypeError(
                f"{modality.name} must be a list or None, got {type(data_list)}"
            )

        formatted_indices = []
        for idx, item in enumerate(data_list):
            if BaseMultimodalProcessor._is_preprocessed_input(item):
                formatted_indices.append(idx)

        # 预处理载荷自带 input_ids / offsets，仅支持「单条」语义；
        # 多张图已预处理的情况应由调用方合并成一个条目再传。
        if formatted_indices:
            if len(data_list) != 1:
                raise ValueError(
                    f"For {modality}, when providing a 'processor_output' or "
                    f"'precomputed_embedding', you must pass exactly one item; "
                    f"received {len(data_list)} items (formatted at indices {formatted_indices})."
                )

    @staticmethod
    def validate_mm_data(
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
    ):
        """
        Validate multimodal input lists per modality.

        Rule per modality (image/video/audio):
        - Either the list has exactly one item and that single item is a dict with
          format in {"processor_output", "precomputed_embedding"};
        - Or, the list contains only "normal" items (i.e., does not include any
          item whose format is one of the two above).

        Empty or None lists are considered valid.

        中文：注意这里只校验「形态」，不校验「个数是否与 prompt 对得上」——
        个数对不上是合法场景（legacy 路径专门处理），所以不能在这里报错。
        """

        BaseMultimodalProcessor._validate_one_modality(Modality.IMAGE, image_data)
        BaseMultimodalProcessor._validate_one_modality(Modality.VIDEO, video_data)
        BaseMultimodalProcessor._validate_one_modality(Modality.AUDIO, audio_data)

    def _process_loaded_mm_data(self, modality, raw_data, result):
        # 把一个已加载的结果归到对应模态的列表里。
        # 图像特例：普通图像可能被返回成 list（如 PDF 拆页），需要展平；
        # 而预处理载荷本身就是一个整体，不能再展平。
        images, videos, audios = [], [], []

        is_precomputed = self._is_preprocessed_input(raw_data)

        if modality == Modality.IMAGE:
            if is_precomputed:
                images.append(result)
            else:
                if isinstance(result, list):
                    images.extend(result)
                else:
                    images.append(result)
        elif modality == Modality.VIDEO:
            videos.append(result)
        elif modality == Modality.AUDIO:
            audios.append(result)

        return is_precomputed, images, videos, audios

    async def load_mm_data(
        self,
        prompt: str,
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
        audio_sample_rate: Optional[int] = None,
    ) -> BaseMultiModalProcessorOutput:
        """取数据的统一入口（分拣器）。

        中文流程：
          1. 先校验参数形态（同一模态不允许混搭预处理载荷与原始数据）；
          2. 若调用方直接给了 input_ids 且所有数据都是预处理载荷 → 原样返回，
             一个字节都不搬（离线回放 / benchmark 走这条）；
          3. 用合并正则把 prompt 切成 [文本, 占位符, 文本, ...]；
          4. 数一遍各模态占位符个数，与传入数据个数比对：
             严格相等 → fast 路径；否则 → legacy 路径。

        为何个数不等必须走 legacy：fast 路径靠【下标】把结果回填到列表，
        隐含「第 i 个占位符 = 第 i 个数据」；一图多帧时这个一一映射不成立，
        必须由 legacy 路径在拼接阶段按实际帧数复制占位符。
        """
        BaseMultimodalProcessor.validate_mm_data(image_data, video_data, audio_data)

        input_ids = prompt if isinstance(prompt, list) else None
        if input_ids is not None and self._all_mm_data_is_preprocessed(
            image_data, video_data, audio_data
        ):
            # fast path for preprocessed data: early return
            # 中文：数据已经是 processor_output / 预计算 embedding，且调用方带了
            # input_ids，那就无需 load、无需重分词，直接把三路数据透传下去，
            # input_text 留空（后续 combine 会从 dict 里取 input_ids）。
            return BaseMultiModalProcessorOutput(
                input_text="",
                input_ids=input_ids,
                images=list(image_data or []),
                videos=list(video_data or []),
                audios=list(audio_data or []),
            )

        multimodal_tokens_pattern = multimodal_tokens.get_combined_regex()
        if isinstance(prompt, list) and return_text:
            assert len(prompt) and isinstance(prompt[0], int)
            prompt = self._tokenizer.decode(prompt)
        else:
            prompt = prompt

        assert isinstance(prompt, str)
        # split text into list of normal text and special tokens
        text_parts = re.split(multimodal_tokens_pattern, prompt)

        cnt = {Modality.IMAGE: 0, Modality.VIDEO: 0, Modality.AUDIO: 0}
        for text_part in text_parts:
            modality = multimodal_tokens.get_modality_of_token(text_part)
            if modality is not None:
                cnt[modality] += 1

        n_image = len(image_data) if image_data else 0
        n_video = len(video_data) if video_data else 0
        n_audio = len(audio_data) if audio_data else 0

        # For MiniCPMO and MiniCPMV or multimodal_tokens not totally align, legacy show path
        # 中文：满足任一条件就走 legacy：
        #   * skip_tokenizer_init：没分词器，无法把 prompt 解码成字符串去数占位符；
        #   * 三种模态的占位符个数与数据个数不相等（典型：一图多帧/视频帧数可变）；
        #   * 模型声明支持动态帧扩展（MiniCPM-V/O），帧数要等解码后才确定。
        if (
            self.skip_tokenizer_init
            or cnt[Modality.IMAGE] != n_image
            or cnt[Modality.VIDEO] != n_video
            or cnt[Modality.AUDIO] != n_audio
            or getattr(self, "support_dynamic_frame_expansion", False)
        ):
            # 注意 input_ids 一并传入：legacy 路径只负责拼文本，原始 token 仍透传，
            # 供下游在 SGLANG_MM_AVOID_RETOKENIZE 下重建 input_ids。
            return await self.legacy_load_mm_data(
                prompt=prompt,
                multimodal_tokens=multimodal_tokens,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                return_text=return_text,
                discard_alpha_channel=discard_alpha_channel,
                audio_sample_rate=audio_sample_rate,
                input_ids=input_ids,
            )
        # For models other than MiniCPMO and MiniCPMV,
        # totally align multimodal_tokens, fast path
        return await self.fast_load_mm_data(
            prompt=prompt,
            multimodal_tokens=multimodal_tokens,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            return_text=return_text,
            discard_alpha_channel=discard_alpha_channel,
            audio_sample_rate=audio_sample_rate,
            input_ids=input_ids,
        )

    async def fast_load_mm_data(
        self,
        prompt: str,
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
        audio_sample_rate: Optional[int] = None,
        input_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> BaseMultiModalProcessorOutput:
        """
        A fast version of `load_mm_data` that loads multimodal data directly.
        This version does not scan the prompt to recognize tokens. It assumes
        that the caller has already aligned the tokens and data in a 1:1 manner.
        The behavior is as follows:
        1. It runs `_load_single_item` for all input data concurrently.
        2. It returns the loaded images, videos, and audios in their original order.
        3. It returns the input prompt as a string.

        中文：前提是「占位符个数 == 数据个数」，所以这里【不需要】扫描 prompt，
        也不需要逐项同步迭代器：三路数据全提交，按 index 回填即可。
        代价是无法表达「一个占位符展开成多个 token」，那类情况走 legacy 路径。
        文本部分原样返回（prompt 已由调用方处理好，占位符数量本就对）。

        注意：本方法【不校验】个数是否真的对齐（校验在 ``load_mm_data`` 里做）。
        一般不要直接调用它，而是调用 ``load_mm_data`` 让它自动选路。

        Args:
            prompt: 提示文本。两种形态都支持：
                * ``str``：普通文本，占位符以字面量形式出现在串里（最常见）；
                * ``List[int]``：预分词后的 token ids，需要 ``return_text=True``
                  才会被 decode 成字符串（decode 只用于切分/回显，
                  真正的 ``input_ids`` 由入参原样透传，不丢信息）。
            multimodal_tokens: 描述「哪些 token 代表哪种模态」的对象。
                **fast 路径内部并不使用它**，保留这个参数纯粹是为了与
                ``legacy_load_mm_data`` 保持同一签名，方便 ``load_mm_data``
                两条分支透传。真正会用到它的是 legacy 路径（要 regex 切分）。
            image_data: 图像列表，**一个元素对应 prompt 里的一个图像占位符**。
                元素可以是 URL / 本地路径 / base64 / bytes / ``PIL.Image`` /
                ``torch.Tensor``，也可以是预处理载荷 dict（原样透传、不解码）。
                为 None 或空列表表示该模态无数据。
            video_data: 视频列表，语义同上；元素通常是视频路径/URL，
                加载后得到 ``T×H×W×C`` 的 tensor。
            audio_data: 音频列表，语义同上；加载后得到 ``np.ndarray``
                （采样点 × 声道）。
            return_text: prompt 为 ``List[int]`` 时是否 decode 成字符串。
                为 True 时走 ``tokenizer.decode``；为 False 时要求 prompt 本身
                已经是 ``str``（否则下面的 assert 会失败）。
                注意它只影响 ``input_text``，不影响 ``input_ids`` 的透传。
            discard_alpha_channel: 图像是否丢弃 alpha 通道（转 RGB）。
                走 PIL 分支且为 True 时，RGBA/LA/P 图会被 ``convert("RGB")``，
                避免下游 processor 因通道数不一致报错。
                nvJPEG（GPU 解码）分支不经过这里，它输出的是 tensor。
            audio_sample_rate: 目标采样率。非 None 时会重采样到该采样率，
                保证同一请求内多个音频的采样率一致（否则拼 batch 会失败）。
            input_ids: 调用方已有的 token ids，仅做透传，写入返回值的
                ``input_ids`` 字段。用于 ``SGLANG_MM_AVOID_RETOKENIZE``
                或离线回放场景，避免 decode→retokenize 引入 token 漂移。

        Returns:
            ``BaseMultiModalProcessorOutput``，字段含义：
                * ``input_text``：**原样**的 prompt 字符串（fast 路径不展开
                  占位符，这是它与 legacy 路径最大的行为差异）；
                * ``input_ids``：入参 ``input_ids`` 原样透传，没有则为 None；
                * ``images`` / ``videos`` / ``audios``：按输入下标对齐的已加载
                  数据列表（``images[i]`` 就是 ``image_data[i]`` 的加载结果），
                  元素类型为 PIL.Image / torch.Tensor / np.ndarray / dict。
                  列表长度等于对应入参列表的长度。

        Raises:
            AssertionError: prompt 是 list 但元素不是 int（应传 token ids），
                或者 prompt 是 list 且 ``return_text=False``。
            ValueError: 客户端数据问题（URL 404、格式不支持、文件损坏等），
                消息里带模态名和下标，便于定位是第几个输入坏了。
            RuntimeError: 服务端问题（解码器内部错误、OOM 等），
                同样带模态名和下标。

        Examples:
            >>> # 1) 两个占位符 + 两张图：1:1 对齐，走 fast 路径
            >>> tokens = MultimodalSpecialTokens(image_token="<|image_pad|>")
            >>> out = asyncio.run(self.fast_load_mm_data(
            ...     prompt="<|image_pad|>和<|image_pad|>有什么不同？",
            ...     multimodal_tokens=tokens,
            ...     image_data=["/data/a.jpg", "https://example.com/b.png"],
            ... ))
            >>> out.input_text
            '<|image_pad|>和<|image_pad|>有什么不同？'   # 未展开，原样返回
            >>> len(out.images)                          # 顺序与 image_data 一致
            2

            >>> # 2) 预分词 ids 入参：decode 用于回显，input_ids 原样透传
            >>> ids = [151652, 151655, 9707]            # <vision_start><image_pad>hi
            >>> out = asyncio.run(self.fast_load_mm_data(
            ...     prompt="<|image_pad|>hi",
            ...     multimodal_tokens=tokens,
            ...     image_data=[open("/data/a.jpg", "rb").read()],
            ...     input_ids=ids,                      # 下游用它，避免重分词漂移
            ... ))
            >>> out.input_ids == ids
            True

            >>> # 3) 预处理载荷（离线回放 / benchmark）：dict 原样透传，不解码
            >>> payload = {"format": "processor_output", "input_ids": ids, ...}
            >>> out = asyncio.run(self.fast_load_mm_data(
            ...     prompt="<|image_pad|>描述",
            ...     multimodal_tokens=tokens,
            ...     image_data=[payload],
            ... ))
            >>> out.images[0] is payload
            True

            >>> # 4) 个数不对齐时【不要】直接调本方法：它不会展开占位符，
            >>> #    下游会因 token 数与特征数不匹配而报错。应先走 load_mm_data：
            >>> out = asyncio.run(self.load_mm_data(      # 自动选择 fast / legacy
            ...     prompt="<|video_pad|>总结这个视频",
            ...     multimodal_tokens=tokens,
            ...     video_data=["/data/clip.mp4"],       # 1 个占位符 vs T 帧
            ... ))
        """

        # Convert prompt into str
        if isinstance(prompt, list) and return_text:
            assert len(prompt) and isinstance(prompt[0], int)
            prompt_str = self._tokenizer.decode(prompt)
        else:
            assert isinstance(prompt, str)
            prompt_str = prompt

        futures: List[Tuple[Modality, int, concurrent.futures.Future]] = []

        modalities_data = [
            (image_data, Modality.IMAGE),
            (video_data, Modality.VIDEO),
            (audio_data, Modality.AUDIO),
        ]

        for data_list, modality in modalities_data:
            futures.extend(
                self._submit_mm_data_loading_tasks_simple(
                    data_list, modality, audio_sample_rate, discard_alpha_channel
                )
            )

        logger.debug("[load_mm_data(simple)] total futures submitted: %d", len(futures))

        images: List[Any] = [None] * len(image_data) if image_data else []
        videos: List[Any] = [None] * len(video_data) if video_data else []
        audios: List[Any] = [None] * len(audio_data) if audio_data else []

        for modality, idx, future in futures:
            try:
                result = await asyncio.wrap_future(future)
            except ValueError as e:
                logger.info(
                    "[load_mm_data(simple)] invalid %s data at index=%d: %s",
                    modality.name,
                    idx,
                    e,
                )
                raise ValueError(
                    f"An exception occurred while loading {modality.name} data "
                    f"at index {idx}: {e}"
                ) from e
            except Exception as e:
                logger.exception(
                    "[load_mm_data(simple)] error loading %s data at index=%d",
                    modality.name,
                    idx,
                )
                raise RuntimeError(
                    f"An exception occurred while loading {modality.name} data at index {idx}: {e}"
                )

            if modality == Modality.IMAGE:
                images[idx] = result
            elif modality == Modality.VIDEO:
                videos[idx] = result
            elif modality == Modality.AUDIO:
                audios[idx] = result

        logger.debug(
            "[load_mm_data(simple)] loaded counts: images=%d, videos=%d, audios=%d",
            len(images),
            len(videos),
            len(audios),
        )

        return BaseMultiModalProcessorOutput(
            images=images,
            audios=audios,
            videos=videos,
            input_text=prompt_str,
            input_ids=input_ids,
        )

    async def legacy_load_mm_data(
        self,
        prompt: str,
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
        audio_sample_rate: Optional[int] = None,
        input_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> BaseMultiModalProcessorOutput:
        """
        Each frame of video/image will be replaced by a single image token

        Args:
            multimodal_tokens (list[str]): list of special token which denoting a single multimodal data
                e.g. image token or audio token
            discard_alpha_channel: if True, discards the alpha channel in the returned images

        ────────────────────────────────────────────────────────────
        中文详解：为什么需要这条路径
        ────────────────────────────────────────────────────────────
        fast 路径假设「prompt 里 N 个占位符 ↔ N 个数据」。以下情况不成立：
          * 视频：prompt 里只有 1 个 <video>，解码后却是 T 帧，最终要展开成 T 个占位符；
          * MiniCPM-V 这类动态切图模型：一张图切成若干子图，占位符数量取决于
            运行时算出的切图方案。
        所以 legacy 路径的原则是：先按占位符逐项取数据，再用「实际展开结果」
        重写占位符，而不是用「数据个数」去对齐「占位符个数」。

        ────────────────────────────────────────────────────────────
        执行步骤（对着代码看）
        ────────────────────────────────────────────────────────────
        Step 1  把 prompt 切成 [文本, 占位符, 文本, 占位符, ...]。
                合并正则外层有捕获组，分隔符本身会保留在列表里。
        Step 2  为每种模态建一个数据迭代器（仅当配置了 token 且确实有数据）。
        Step 3  submit_data_loading_tasks：遍历切分结果，遇到占位符就 next()
                对应模态的数据并提交 IO 任务；futures 与 task_info 因此严格按
                「占位符出现顺序」排列。
        Step 4  再遍历一次切分结果，两个迭代器同步 next()：
                  - 占位符 → 取出 (modality, raw_data, frame_limit) 与加载结果，
                    把结果塞进对应模态列表，并把占位符替换成实际展开出的数量个；
                  - 普通文本 → 原样保留。
        Step 5  "".join 拼回文本，返回「展开后的文本 + 已加载数据」。

        ────────────────────────────────────────────────────────────
        关键不变量 / 坑
        ────────────────────────────────────────────────────────────
        1. futures、task_info、text_parts 三个序列必须同步推进；任何一个多走
           一步，后面全错位。消费两个迭代器时用同一个循环体控制，就是为了保证这点。
        2. 图片分支把占位符替换成 count 个；视频/音频分支直接把原 text_part 放回去。
           注意 ``new_text_parts += mm_tokens`` 是把字符串【逐字符】展开进 list，
           能工作纯粹因为最后会 "".join，别把它当 append 用。
        3. 预计算输入（dict）自带 offsets，所以原样保留 text_part、不重写占位符；
           这种情况下 StopIteration 是预期行为（没有对应的 load 任务）。
        4. 空帧：count == 0 时不要追加任何占位符（过滤无效帧）。
        """

        multimodal_tokens_pattern = multimodal_tokens.get_combined_regex()
        if isinstance(prompt, list) and return_text:
            assert len(prompt) and isinstance(prompt[0], int)
            prompt = self._tokenizer.decode(prompt)
        else:
            prompt = prompt

        assert isinstance(prompt, str)
        # split text into list of normal text and special tokens
        # 中文：re.split 带捕获组时会保留分隔符，所以结果里「普通文本」和「占位符」
        # 交替出现；后续判断某一项是占位符还是文本，靠 pattern.match 而不是下标奇偶。
        text_parts = re.split(multimodal_tokens_pattern, prompt)
        # collect all data
        # 中文：只为「配了 token 且确实给了数据」的模态建迭代器。
        # 注意这是【单次迭代器】，一旦 next() 过就不能回头，所以下面只能顺序消费。
        data_iterators = {}
        if multimodal_tokens.image_token and image_data:
            data_iterators[Modality.IMAGE] = iter(image_data)
        if multimodal_tokens.video_token and video_data:
            data_iterators[Modality.VIDEO] = iter(video_data)
        if multimodal_tokens.audio_token and audio_data:
            data_iterators[Modality.AUDIO] = iter(audio_data)

        # futures: the futures of loaded data
        # task_info: modality, raw_data, and other metadata of each data
        # 中文：两个列表长度相同、顺序相同，都等价于「prompt 中占位符的出现顺序」。
        # 之所以还要把 raw_data 带回来：后面要靠它判断该 item 是不是预处理载荷。
        futures, task_info = self.submit_data_loading_tasks(
            text_parts=text_parts,
            multimodal_tokens=multimodal_tokens,
            data_iterators=data_iterators,
            discard_alpha_channel=discard_alpha_channel,
            audio_sample_rate=audio_sample_rate,
        )
        # 中文：把两个列表包成迭代器，与 text_parts 的遍历「对齐步伐」。
        # 这里不用下标访问，是为了防止某个分支漏走了占位符还能静默跑下去——
        # 一旦错位，后面的 StopIteration 会立刻暴露问题。
        task_info_iter = iter(task_info)
        futures_iter = iter(futures)

        # Process results
        images, videos, audios = [], [], []
        new_text_parts = []
        has_precomputed_input = False
        for text_part in text_parts:
            try:
                # 中文：match 而不是 search：text_part 要么整段都是占位符，
                # 要么是普通文本，不存在「部分匹配」的情况。
                if multimodal_tokens_pattern.match(text_part):
                    # 三个迭代器在此处严格同步推进：text_parts 由 for 驱动，
                    # task_info / futures 各取一个。顺序是提交时就固定好的。
                    modality, raw_data, frame_limit = next(task_info_iter)
                    # wrap_future 把 concurrent 的 Future 适配成 asyncio Future，
                    # await 期间事件循环可以继续处理其它请求；
                    # 注意这里是串行 await，但 IO 任务本身已经并发提交了。
                    result = await asyncio.wrap_future(next(futures_iter))

                    is_precomputed, new_imgs, new_vids, new_auds = (
                        self._process_loaded_mm_data(modality, raw_data, result)
                    )

                    has_precomputed_input |= is_precomputed
                    images.extend(new_imgs)
                    videos.extend(new_vids)
                    audios.extend(new_auds)

                    if modality == Modality.IMAGE:
                        # 中文：图像是一对多关系——一次加载可能得到多帧（PDF 拆页、
                        # 多页 TIFF），所以占位符要复制 count 份；
                        # count 就是本次实际加载出的图/帧数量。
                        # 预处理载荷例外：它自身就是一个整体，占位符必须原样保留。
                        if is_precomputed:
                            new_text_parts += [text_part]
                        else:
                            count = len(new_imgs)
                            if count > 0:
                                new_text_parts += [
                                    multimodal_tokens.image_token
                                ] * count
                    elif modality == Modality.VIDEO:
                        # load as video
                        # 中文：视频不在这里展开成 T 个占位符，而是原样写回一个 video
                        # token。具体占位多少个 token 由下游 processor 根据
                        # video_grid_thw 决定（见 build_input_ids / combine 阶段）。
                        mm_tokens = (
                            text_part
                            if is_precomputed
                            else multimodal_tokens.video_token
                        )
                        new_text_parts += mm_tokens
                    elif modality == Modality.AUDIO:
                        # audio
                        # 中文：与视频同理，音频占多少 token 由采样点数在后续
                        # processor（或 build_input_ids 里的 audio_seq_lens）决定，
                        # 这里只保留一个 audio 占位符。
                        mm_tokens = (
                            text_part
                            if is_precomputed
                            else multimodal_tokens.audio_token
                        )
                        new_text_parts += mm_tokens
                else:
                    # normal text
                    new_text_parts += [text_part]

            except StopIteration as e:
                # when precomputed_input is presented with multi-images, StopIteration is expected
                # 中文：走到这里说明「prompt 里的占位符比实际提交的 load 任务多」。
                # 已出现过预处理载荷时，多出来的占位符本就对应「自带 offset 的数据」，
                # 不需要 load 任务，因此属正常情况，把文本原样保留即可；
                # 否则说明用户少给了数据，当作错误抛出。
                if has_precomputed_input:
                    new_text_parts += [text_part]
                    continue
                raise RuntimeError(
                    f"An exception occurred while loading multimodal data: {e}"
                )
            except ValueError as e:
                raise ValueError(
                    f"An exception occurred while loading multimodal data: {e}"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"An exception occurred while loading multimodal data: {e}"
                )
        # 中文：join 把「文本片段 + 复制后的占位符」拼回完整 prompt。
        # 注意这里不带任何分隔符，所以前面把字符串逐字符丢进列表也不会出错。
        # input_ids 原样透传：若调用方带了预分词 id，后续 combine 可据此避免重分词。
        return BaseMultiModalProcessorOutput(
            images=images,
            audios=audios,
            videos=videos,
            input_text="".join(new_text_parts),
            input_ids=input_ids,
        )

    @staticmethod
    def get_mm_items_offset(
        input_ids: torch.Tensor, mm_token_id: int
    ) -> List[Tuple[int, int]]:
        """
        Get a set of range for mm_items from input_ids
        Example:
            input_ids = [1, 2, 3, 3, 3, 4, 3, 3]
            mm_token_id = 3
            return result = [(2,4),(6,7)]
        """
        # 中文：用「当前为 True 且左邻为 False」找每一段连续占位符的起点，
        # 用「当前为 True 且右邻为 False」找终点。
        # torch.roll(mask, 1) 把序列右移一位，相当于拿到「前一个元素」的 mask；
        # 首尾会越界回绕，但首元素若为 True 时其左邻回绕成末元素，
        # 由于末元素必为「目标 token」才会配对成功，实际不会误判（序列首尾恰好相接时需子类重写）。
        mask = input_ids == mm_token_id
        start_positions = (mask & ~torch.roll(mask, 1)).nonzero(as_tuple=True)[0]
        end_positions = (mask & ~torch.roll(mask, -1)).nonzero(as_tuple=True)[0]
        return list(zip(start_positions.tolist(), end_positions.tolist()))

    @staticmethod
    def get_mm_items_offset_by_pair(
        input_ids: torch.Tensor, mm_start_id: int, mm_end_id: int
    ) -> List[Tuple[int, int]]:
        # 中文：占位符被一对边界 token 包着（如 <im_start> ... <im_end>），
        # 真正的占位内容在 start 之后、end 之前，所以 start+1 / end-1。
        # 这里直接用 zip 配对，隐含「start/end 严格交替且数量相等」的不变量。
        indices_start = (input_ids == mm_start_id).nonzero(as_tuple=True)[0] + 1
        indices_end = (input_ids == mm_end_id).nonzero(as_tuple=True)[0] - 1

        return list(zip(indices_start.tolist(), indices_end.tolist()))

    def get_mm_item_offsets(
        self,
        input_ids: torch.Tensor,
        mm_tokens: MultimodalSpecialTokens,
        modality: Modality,
    ) -> List[Tuple[int, int]]:
        """Return placeholder offsets belonging to one modality.

        Processors that reuse one token ID for multiple modalities can override
        this method and use surrounding boundary tokens to disambiguate spans.

        中文：按模态取占位符区间。默认实现就是「扫该模态的 token id」，
        但像 GLM4V 这类图像/视频共用一个 pad token 的模型，必须靠前后边界
        才能区分，因此留了虚方法。
        """
        mm_token_id = mm_tokens.get_token_id_by_modality(modality)
        if mm_token_id is None:
            raise ValueError(f"No token id found for modality: {modality}")
        return self.get_mm_items_offset(input_ids, mm_token_id)

    def collect_mm_items_from_processor_output(
        self, data_dict: dict, modality: Modality = None
    ) -> List[MultimodalDataItem]:
        """
        Create mm_items from processor output.

        Initially creates one item per modality; these are later split into per-image/video items by get_new_expanded_mm_items.

        Note that the data_dict can be hf processor output, or passed via offline engine api

        Args:
            modality: if provided, force the data into a single MultimodalDataItem of that modality
        """

        # universal getter for data_dict
        # 中文：data_dict 可能是 dict，也可能是 HF 的 BatchFeature（支持属性访问），
        # 所以先抽象出一个统一取值函数。
        get_data_value = (
            data_dict.get
            if hasattr(data_dict, "get")
            else lambda name, default=None: getattr(data_dict, name, default)
        )

        # decide explicitly-set modality
        explicit_modality = modality
        modality_value = get_data_value("modality")
        if explicit_modality is None and modality_value is not None:
            explicit_modality = (
                modality_value
                if isinstance(modality_value, Modality)
                else Modality.from_str(str(modality_value))
            )

        items: dict[Modality, MultimodalDataItem] = {}
        for attr_name, value in data_dict.items():
            if attr_name in (
                "input_ids",
                "format",
                "modality",
                "hash",
                "pad_value",
                "offsets",
            ):
                # metadata fields need explicit handling, skip generic item.set
                # 中文：这些字段不属于「某个模态的特征」，而是描述整条数据的元信息，
                # 所以不能在循环里直接 set 到 item 上，而是在下面 len(items)==1 时
                # 统一处理（只有单 item 时才能确定这些元信息归谁）。
                continue

            # Get modality for this attribute
            current_modality = explicit_modality or self.ATTR_NAME_TO_MODALITY.get(
                attr_name
            )

            if attr_name == "precomputed_embeddings":
                current_modality = current_modality or Modality.IMAGE

            if current_modality:
                # Create item if needed
                if current_modality not in items:
                    items[current_modality] = MultimodalDataItem(
                        modality=current_modality
                    )

                # 中文：不同模型的特征字段名不同（pixel_values / input_features / ...），
                # 但下游统一只认 item.feature，所以这里做一次改名归一。
                if attr_name in self.FEATURE_NAMES:
                    attr_name = "feature"

                items[current_modality].set(attr_name, value)

        # deal with metadata fields when data_dict is preprocessed input: convert from tensor to expected python types
        # the attribution of the metadata fields is only clear when number of MultimodalDataItem is 1
        if len(items) == 1:
            item = next(iter(items.values()))

            # adjust offset
            offsets = get_data_value("offsets")
            if offsets is not None:
                if isinstance(offsets, torch.Tensor):
                    offsets = offsets.detach().cpu().tolist()
                item.offsets = [(int(start), int(end)) for start, end in offsets]

            # adjust hash_value
            hash_value = get_data_value("hash")
            if hash_value is not None:
                if isinstance(hash_value, torch.Tensor):
                    hash_value = hash_value.item()
                item.hash = int(hash_value)
                pad_value = get_data_value("pad_value")
                if pad_value is not None:
                    if isinstance(pad_value, torch.Tensor):
                        pad_value = pad_value.item()
                    item.pad_value = int(pad_value)

        return list(items.values())

    def _process_and_collect_mm_items(
        self,
        input_text: str,
        images=None,
        audios=None,
        videos=None,
        processor=None,
        **kwargs,
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """
        Helper method to process multimodal data and create mm_items in one step.

        Returns:
            Tuple of (created mm_items, input_ids)

        中文：把「调 HF processor」与「把输出转换成 mm_items」两件事包成一步，
        顺便把 input_ids 拉平（HF 返回的通常是 [1, L] 的二维张量）。
        """
        if processor is not None:
            # 中文：把 worker 专属的 processor 透传下去，见 _resolve_processor 的说明。
            kwargs["processor"] = processor
        ret = self.process_mm_data(
            input_text=input_text,
            images=images,
            audios=audios,
            videos=videos,
            **kwargs,
        )

        input_ids = ret["input_ids"].flatten()
        collected_items = self.collect_mm_items_from_processor_output(ret)

        return collected_items, input_ids, ret

    @staticmethod
    def _ensure_input_ids_is_tensor(input_ids) -> Optional[torch.Tensor]:
        """make sure the input_ids is a flattened tensor

        中文：统一成「一维 long 张量」。下游的 get_mm_items_offset 需要按值比较，
        dtype 不一致（如 int32 vs int64）在部分后端上会报错或静默升位，
        所以这里显式转 long。
        """
        if input_ids is None:
            return None
        if isinstance(input_ids, torch.Tensor):
            return input_ids.flatten().to(dtype=torch.long)
        return torch.tensor(input_ids, dtype=torch.long).flatten()

    def _wrap_tensor_for_cuda_ipc(self, tensor: torch.Tensor):
        """helper function to turn a tensor into a cuda-ipc tensor

        中文：把 GPU 张量放进共享内存池，并返回一个可跨进程传输的代理对象。
        两个兜底：
          * 张量本来就在 CPU → 无需包装，直接返回；
          * 内存池已满（wrap_tensor 返回 None）→ 退回 tensor.cpu()。
        兜底很重要：池满时宁可多一次拷贝，也不能让请求直接失败。
        """
        if not tensor.is_cuda:
            return tensor

        proxy = self.cudaipc_mmfeature_pool.wrap_tensor(
            tensor,
            use_pool_handle_cache=self.use_ipc_pool_handle_cache,
        )
        return proxy if proxy is not None else tensor.cpu()

    @staticmethod
    def _move_feature_to_cpu(value):
        # 中文：特征可能是 tensor、list[tensor] 或 tuple，逐层递归下 CPU。
        # 用递归而不是假定结构，是因为不同模型把 pixel_values 组织成不同嵌套层次。
        if isinstance(value, torch.Tensor):
            return value.cpu()
        if isinstance(value, list):
            return [BaseMultimodalProcessor._move_feature_to_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(BaseMultimodalProcessor._move_feature_to_cpu(v) for v in value)
        return value

    def _precompute_hashes_before_cpu_transfer(
        self, mm_items: List[MultimodalDataItem]
    ) -> None:
        if not self.precompute_hash_before_cpu_transfer:
            return

        # 中文：顺序不能反——必须先 set_pad_value（它会读 feature 算哈希），
        # 再把 feature 搬去 CPU；否则哈希就是基于 CPU 副本算的，跟 GPU 上真正
        # 参与计算的数据存在微小差异风险。
        for item in mm_items:
            item.set_pad_value()
            if not self.keep_mm_features_on_device:
                item.feature = self._move_feature_to_cpu(item.feature)
                item.precomputed_embeddings = self._move_feature_to_cpu(
                    item.precomputed_embeddings
                )

    def resolve_image_token_counts(self, images: List) -> List[int]:
        """Per-image expanded token counts, computed without re-tokenizing.

        Default implementation uses the transformers in-tree convention
        ``_get_num_multimodal_tokens(image_sizes=...)`` (present on the in-tree
        VLM processors, e.g. Qwen-VL, Gemma3, GLM4V). Models whose processor
        does not implement it (e.g. Kimi) override this method.

        中文：不重新分词就能知道「第 i 张图要占多少 token」，靠的是 HF processor
        的 ``_get_num_multimodal_tokens``（只做尺寸/网格算术，不读像素）。
        这一步是 SGLANG_MM_AVOID_RETOKENIZE 能成立的前提。
        """
        assert images is not None
        image_sizes = [
            (
                tuple(image.shape[-2:])
                if isinstance(image, torch.Tensor)
                else (image.height, image.width)
            )
            for image in images
        ]
        num_image_tokens = self._processor._get_num_multimodal_tokens(
            image_sizes=image_sizes
        ).num_image_tokens
        return [int(count) for count in num_image_tokens]

    @staticmethod
    def _expand_input_ids(
        original_ids: List[int],
        counts: List[int],
        placeholder_token_id: Optional[int],
    ) -> List[int]:
        """Rebuild final input_ids for a pre-tokenized (list[int]) prompt.

        Keep the user's ORIGINAL tokens verbatim and expand the i-th image
        placeholder into ``counts[i]`` copies of ``placeholder_token_id``. The HF
        processor's re-tokenization is discarded, so non-media tokens cannot
        drift.

        中文：只在「原始 input_ids + 占位符个数匹配」时才会走到这里。
        典型漂移场景：用户传的 token 里含特殊空白或非规范编码，decode 再 encode
        后长度会变，导致前后端看到的 prompt 不一致。
        万一占位符个数对不上（比如 processor 多塞了一个 pad），这里直接报
        ValueError，由上层捕获后回退到 decode+retokenize（并打 warn）。
        """
        if placeholder_token_id is None:
            raise ValueError("placeholder_token_id is not set for this processor")

        num_placeholders = sum(
            1 for token_id in original_ids if token_id == placeholder_token_id
        )
        if num_placeholders != len(counts):
            raise ValueError(
                f"prompt has {num_placeholders} image placeholder token(s) but "
                f"{len(counts)} image(s) were provided"
            )

        rebuilt: List[int] = []
        next_image_idx = 0
        for token_id in original_ids:
            if token_id == placeholder_token_id:
                rebuilt.extend([placeholder_token_id] * counts[next_image_idx])
                next_image_idx += 1
            else:
                rebuilt.append(token_id)
        return rebuilt

    def process_and_combine_mm_data(
        self,
        base_output: BaseMultiModalProcessorOutput,
        mm_tokens: MultimodalSpecialTokens,
        processor=None,
        **kwargs,
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """
        Process multimodal data and return the combined multimodal items and input_ids.
        Supports mixed modalities (images and audio in the same request).

        Returns:
            Tuple of (list of mm_items, input_ids)

        中文：load 之后的第二阶段，也是整条链路最关键的构造环节。
        输入的 base_output 里装着已加载的原始数据（或预处理 dict），
        输出是 scheduler 直接可用的 (mm_items, input_ids, 原始 processor 输出)。

        三类数据的分流（按元素类型而非模态）：
          * dict + format=processor_output：上游已跑过 processor，直接拆成 mm_items；
          * dict + format=precomputed_embedding：上游已算好 embedding，包成一个
            item 并把其余字段放进 model_specific_data；
          * 其余原始数据：合并后一次调 process_mm_data，拿回特征再拆成 mm_items。

        输出前还要做三件事：
          1. 定 input_ids（优先用调用方透传的，其次 dict 里的，最后才重新分词）；
          2. 给每个 item 算 offsets（占位符在 input_ids 中的区间）；
          3. 拆分 + 补 pad_value/hash + 可能包装成 CUDA IPC 代理。
        """
        processor_override = processor
        processor, tokenizer = self._resolve_processor(processor)

        # Collect all items and categorize them
        all_loaded_data = base_output.organize_results()
        # Handle text-only case
        # 中文：纯文本请求不经过 HF processor，直接用 tokenizer 编码。
        # 不用 processor 的原因是它可能对纯文本加额外 padding/模板字段，
        # 反而与文本模型的输入约定不一致。
        if not all_loaded_data:
            input_ids = tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()
            return [], input_ids, {}

        # 中文：按「元素是不是 dict」分流，而不是按模态——因为同一个模态下
        # 可能一部分是预处理载荷、一部分是原始图（虽然 validate_mm_data 会拦，
        # 但组织阶段仍然保持这个判据更清晰）。
        dict_items, raw_images, raw_audios, raw_videos = [], [], [], []
        for modality, item in all_loaded_data:
            if isinstance(item, dict):
                dict_items.append((modality, item))
            elif modality == Modality.IMAGE:
                raw_images.append(item)
            elif modality == Modality.AUDIO:
                raw_audios.append(item)
            elif modality == Modality.VIDEO:
                raw_videos.append(item)
            else:
                raise ValueError(f"Unknown multimodal item type: {type(item)}")
        # Process items and get input_ids
        all_collected_items: list[MultimodalDataItem] = []
        input_ids = None
        # Handle raw items (need processing)
        if raw_images or raw_audios or raw_videos:
            if processor_override is not None:
                kwargs["processor"] = processor
            collected_items, input_ids, ret = self._process_and_collect_mm_items(
                input_text=base_output.input_text,
                images=raw_images,
                audios=raw_audios,
                videos=raw_videos,
                **kwargs,
            )
            all_collected_items = collected_items

            # When SGLANG_MM_AVOID_RETOKENIZE is on, keep the user's exact tokens to avoid retokenize drift.
            # 中文：默认路径下 input_ids 是「文本→tokenizer→ids」，而文本是
            # 之前从 ids decode 出来的，一来一回可能变长/变短（token 漂移）。
            # 开关打开且只有图片时，就用用户原始 ids + 重新算的图片 token 数
            # 自己拼 input_ids，彻底绕开重分词。
            # 限制条件很严：仅图片、无音频/视频、processor 没自己造 ids——
            # 只要有一条不满足就放弃，因为混合模态下逐项对齐很容易算错。
            # Drift happens when Retokenization is not identity: Decode(X) => String => Re-tokenize => Y, X != Y.
            if (
                envs.SGLANG_MM_AVOID_RETOKENIZE.get()
                and not self.preserve_processor_input_ids
                and base_output.input_ids is not None
                and input_ids is not None
                and raw_images
                and not raw_audios
                and not raw_videos
            ):
                assert isinstance(base_output.input_ids, list), (
                    f"expected list[int] input_ids, got {type(base_output.input_ids)}"
                )
                try:
                    counts = self.resolve_image_token_counts(raw_images)
                    image_placeholder_token_id = mm_tokens.image_token_id
                    if image_placeholder_token_id is None:
                        raise ValueError(
                            "image placeholder token id is not set for this processor"
                        )
                    processor_placeholder_count = int(
                        (input_ids == image_placeholder_token_id).sum().item()
                    )
                    if processor_placeholder_count != sum(counts):
                        raise ValueError(
                            "processor image placeholder count mismatch: "
                            f"processor={processor_placeholder_count}, "
                            f"resolved={sum(counts)}"
                        )
                    input_ids = torch.tensor(
                        self._expand_input_ids(
                            base_output.input_ids,
                            counts,
                            image_placeholder_token_id,
                        ),
                        dtype=input_ids.dtype,
                    )
                except Exception as e:
                    logger.warning(
                        f"Due to {e}, falling back to decode+retokenize, which may change prompt length (token drift)."
                    )
        else:
            ret = None

        # Handle dict items (processed or precomputed)
        dict_ret = None
        for modality, dict_item in dict_items:
            input_format = self._get_preprocessed_input_format(dict_item)
            if input_format is not None and dict_ret is None:
                dict_ret = dict_item
            if input_format == MultimodalInputFormat.PROCESSOR_OUTPUT:
                items = self.collect_mm_items_from_processor_output(dict_item)
                for item in items:
                    item.format = MultimodalInputFormat.PROCESSOR_OUTPUT
                all_collected_items.extend(items)
            elif input_format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:
                dict_item = dict(dict_item)
                feature = dict_item.pop("feature")
                all_collected_items.append(
                    MultimodalDataItem(
                        modality=modality,
                        feature=feature,
                        format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
                        model_specific_data=dict_item,
                    )
                )
        # Fallback tokenization if no raw items were processed
        if ret is None and dict_ret is not None:
            ret = dict_ret

        if input_ids is None:
            input_ids = self._ensure_input_ids_is_tensor(base_output.input_ids)

        if input_ids is None:
            for _, dict_item in dict_items:
                input_ids = self._ensure_input_ids_is_tensor(dict_item.get("input_ids"))
                if input_ids is not None:
                    break

        if input_ids is None:
            input_ids = tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()

        # Add offsets to all items
        # 中文：预处理载荷自带 offsets，跳过；其余按占位符位置现算。
        # offsets 是 prefix cache / KV 复用 / M-RoPE 的基础，算错会直接导致输出乱码。
        for mm_item in all_collected_items:
            if mm_item.offsets is not None:
                continue
            mm_item.offsets = self.get_mm_item_offsets(
                input_ids=input_ids,
                mm_tokens=mm_tokens,
                modality=mm_item.modality,
            )

        # Split bundled items into per-image/video items for better cache granularity
        # 中文：HF processor 通常把一批图打包成一个 pixel_values 张量。这里按 offsets
        # 拆成「一图一个 item」，好处是 prefix cache 可以按图粒度命中（改一张图
        # 不会让其他图的 KV 全部失效）。
        from sglang.srt.managers.mm_utils import get_new_expanded_mm_items

        all_collected_items = get_new_expanded_mm_items(all_collected_items)
        all_collected_items = self._finalize_mm_items(
            all_collected_items,
            images=base_output.images,
        )

        return all_collected_items, input_ids, ret

    def _finalize_mm_items(
        self,
        mm_items: List[MultimodalDataItem],
        *,
        images: Optional[List[Any]],
    ) -> List[MultimodalDataItem]:
        # 中文：收尾流水线，顺序有讲究：
        #   1) 子类先做模型特有的 reshape（此时 feature 还是普通 tensor）；
        #   2) 预处理载荷补 pad_value（用于批内 padding 对齐）；
        #   3) 若需要先算 hash，在 D2H 之前算掉；
        #   4) 最后才考虑包装成 CUDA IPC 代理（一旦包装，feature 就不再是普通 tensor）。
        mm_items = self._postprocess_mm_items_before_transport(
            mm_items,
            images=images,
        )

        for item in mm_items:
            if item.format in (
                MultimodalInputFormat.PROCESSOR_OUTPUT,
                MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
            ):
                item.set_pad_value()

        self._precompute_hashes_before_cpu_transfer(mm_items)
        return self._prepare_mm_items_for_transport(mm_items)

    def _postprocess_mm_items_before_transport(
        self,
        mm_items: List[MultimodalDataItem],
        *,
        images: Optional[List[Any]],
    ) -> List[MultimodalDataItem]:
        """Apply model-specific item reshaping while features are still tensors.

        中文：留给子类的钩子。必须在 transport 之前做，因为一旦变成 IPC 代理，
        就不能再自由 reshape/切分了。
        """
        return mm_items

    def _prepare_mm_items_for_transport(
        self, mm_items: List[MultimodalDataItem]
    ) -> List[MultimodalDataItem]:
        """Wrap final GPU features, rolling back every lease if one wrap fails.

        中文：把一个请求里所有 GPU 特征包装成 IPC 代理。若中途任一个失败，
        必须把已申请的池切片全部归还（cancel_proxy）并把 item 恢复成原 tensor，
        否则内存池会被泄漏的切片逐渐占满，最终只能全量回退到 CPU 拷贝。
        回滚本身也可能失败，所以收集 rollback_errors 并用 add_note 挂到原异常上，
        既不丢原始错误，也保留了回滚失败的信息。
        """
        if not self.use_cuda_ipc:
            return mm_items

        # Pool misses fall back to plain CPU tensors. The scheduler copies out
        # and releases each successful pool slice.
        updates = []
        try:
            for item in mm_items:
                fields = (
                    ("feature", item.feature),
                    ("precomputed_embeddings", item.precomputed_embeddings),
                )
                for field, tensor in fields:
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    wrapped = self._wrap_tensor_for_cuda_ipc(tensor)
                    setattr(item, field, wrapped)
                    updates.append((item, field, tensor, wrapped))
        except BaseException as error:
            rollback_errors = []
            for item, field, tensor, wrapped in reversed(updates):
                try:
                    if isinstance(wrapped, CudaIpcTensorTransportProxy):
                        self.cudaipc_mmfeature_pool.cancel_proxy(wrapped)
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
                finally:
                    setattr(item, field, tensor)
            if rollback_errors:
                error.add_note(
                    f"{len(rollback_errors)} CUDA IPC rollback operation(s) also failed"
                )
                raise error from rollback_errors[0]
            raise
        return mm_items

    async def process_and_combine_mm_data_async(
        self,
        base_output: BaseMultiModalProcessorOutput,
        mm_tokens: MultimodalSpecialTokens,
        **kwargs,
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """Run multimodal preprocessing without blocking the event loop.

        中文：没开并发池时直接同步调（就是同一线程，会阻塞事件循环，但至少能用）；
        开了就把活交给 worker 线程，事件循环在 await 期间可以继续处理其它请求。
        worker 内部用的是它自己那份 deepcopy 的 processor，因此线程安全。
        """
        if self.mm_processor_executor is None:
            return self.process_and_combine_mm_data(base_output, mm_tokens, **kwargs)

        return await self.mm_processor_executor.run(
            self.process_and_combine_mm_data,
            base_output,
            mm_tokens,
            **kwargs,
        )
