"""
Multi-modality utils

中译：多模态（图像 / 视频 / 音频）处理工具集。
      本模块是 SGLang 多模态推理的核心辅助层，主要职责包括：
      1. 把多模态数据项（MultimodalDataItem）的特征编码（embedding）按「分块预填充
         （chunked prefill）」的需要切分、缓存与提取；
      2. 将多模态 embedding 散布（scatter）到文本 token 的 embedding 序列中对应的占位符位置，
         再交给语言模型前向；
      3. 提供占位符 token 的填充（padding）策略、多模态数据边界识别、特征哈希；
      4. 跨进程高效传输张量：CUDA IPC（TransportProxyTensor）与共享内存（ShmPointerMMData）。
      关键协作对象：MultimodalInputs / MultimodalDataItem（schedule_batch）、ForwardBatch、
      MultiModalStaticCache（embedding 缓存）、各模型的 get_image/video/audio_feature 编码函数。
"""

import copy
import hashlib
import pickle
from abc import abstractmethod
from collections import defaultdict
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.multimodal import gpu_tensor_hash
from sglang.srt.managers.schedule_batch import (
    CudaIpcTensorTransportProxy,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.multimodal.evs import EVSEmbeddingResult
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import flatten_nested_list, is_npu, print_warning_once
from sglang.utils import logger

_is_npu = is_npu()

# NOTE: Using the shared logger from sglang.utils instead of creating a module-specific logger
# to ensure consistent logging behavior across the codebase. This prevents issues with log
# propagation that can cause some log messages (like 'server is fired up') to not appear
# in the console when multimodal support is enabled.

# TODO(mick): nccl
# cuda_ipc: for intranode tensor sharing
# 中译：张量跨进程传输模式。cuda_ipc 用于「节点内」GPU 张量共享（零拷贝）；
#       auto / default 走常规序列化路径。
TensorTransportMode = Literal["cuda_ipc", "auto", "default"]


# 中译：预分配的 GPU 特征缓冲区与当前写入偏移。复用一块大缓冲区避免反复申请显存。
_GPU_FEATURE_BUFFER: Optional[torch.Tensor] = None
_BUFFER_OFFSET = 0

# 中译：是否使用默认张量传输模式（惰性判定一次后缓存，见 _get_is_default_transport）。
_is_default_tensor_transport = None


def init_feature_buffer(device):
    # 中译：初始化全局 GPU 特征缓冲区。仅在 GPU 设备、且配置了缓冲区大小、且尚未初始化时分配。
    #       缓冲区大小由环境变量 SGLANG_MM_BUFFER_SIZE_MB 控制（单位 MB，按 float32 计算元素数）。
    global _GPU_FEATURE_BUFFER, _BUFFER_OFFSET
    if (
        device == "cpu"
        or envs.SGLANG_MM_BUFFER_SIZE_MB.get() == 0
        or _GPU_FEATURE_BUFFER is not None
    ):
        return
    try:
        size_mb = envs.SGLANG_MM_BUFFER_SIZE_MB.get()
        num_elements = int(size_mb * 1024 * 1024 / 4)
        _GPU_FEATURE_BUFFER = torch.empty(
            num_elements, dtype=torch.float32, device=device
        )
        logger.info(f"Preallocated {size_mb}MB GPU buffer")
    except RuntimeError as e:
        # 中译：显存不足等导致分配失败时，退回 None（后续逻辑会回退到普通分配）。
        _GPU_FEATURE_BUFFER = None


def reset_buffer_offset():
    # 中译：重置缓冲区写入偏移（一般在每个 batch 开始时调用，以便复用整块缓冲区）。
    global _BUFFER_OFFSET
    _BUFFER_OFFSET = 0


def is_feature_buffer_initialized():
    # 中译：判断全局 GPU 特征缓冲区是否已初始化。
    global _GPU_FEATURE_BUFFER
    if _GPU_FEATURE_BUFFER is None:
        return False
    return True


def try_add_to_buffer(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    # 中译：尝试把张量拷入预分配缓冲区并返回指向缓冲区的视图（view）。
    #       若缓冲区未初始化或剩余空间不足，则原样返回输入张量（不使用缓冲区）。
    global _BUFFER_OFFSET

    if _GPU_FEATURE_BUFFER is None:
        return tensor

    tensor_size = tensor.numel()

    if _BUFFER_OFFSET + tensor_size <= _GPU_FEATURE_BUFFER.numel():
        # 中译：剩余空间足够——拷贝到缓冲区对应片段，按原 shape 取视图，并前移偏移。
        buffer_view = _GPU_FEATURE_BUFFER[_BUFFER_OFFSET : _BUFFER_OFFSET + tensor_size]
        buffer_view.copy_(tensor.flatten(), non_blocking=True)
        result = buffer_view.view(tensor.shape)
        _BUFFER_OFFSET += tensor_size
        return result
    else:
        return tensor


class TransportProxyTensor(torch.Tensor):
    """
    A convenient torch.Tensor subclass that carries extra metadata and supports
    efficient inter-process communications

    中译：携带额外元数据、支持高效进程间通信的 torch.Tensor 子类。
          核心价值在于自定义 pickle 行为（__getstate__/__setstate__）：当 transport_mode
          为 "cuda_ipc" 且张量在 GPU 上时，序列化的不是张量数据本身，而是 CUDA IPC 句柄，
          从而在节点内进程间「零拷贝」共享同一块显存；否则退回普通张量序列化。
          额外携带 name / fields 等元数据，供调度链路传递上下文。
    """

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        name: Optional[str] = None,
        fields: Optional[Dict[str, Any]] = None,
        transport_mode: TensorTransportMode = "default",
        *args,
        **kwargs,
    ):
        # 中译：构造方法。把已有张量「就地」转为本子类（as_subclass，不复制数据），
        #       并挂上 _metadata（名称、附加字段、传输模式）。
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"Input 'data' must be a torch.Tensor, but got {type(data)}"
            )

        instance = data.as_subclass(cls)

        instance._metadata = {
            "name": name,
            "fields": fields if fields is not None else {},
            "transport_mode": transport_mode,
        }

        return instance

    def __getstate__(self):
        """
        Called during pickling. Implements the serialization logic.

        中译：pickle 序列化时调用，实现自定义序列化逻辑。
              cuda_ipc 模式下导出显存的 IPC 句柄（含 shape/dtype/stride 等重建信息）而非张量数据；
              获取句柄失败（如张量并行场景）时自动回退为 default 模式、序列化普通张量数据。
        """
        # acquire all serialize metadata from _metadata
        state = {
            "metadata": self._metadata,
            "tensor_data": None,
            "ipc_extra": None,
        }
        transport_mode = self._metadata.get("transport_mode", "default")

        if transport_mode == "cuda_ipc" and self.is_cuda:
            # 中译：导出 CUDA IPC 共享句柄，接收方可据此映射到同一块显存，避免数据拷贝。
            try:
                storage = self.untyped_storage()
                handle = storage._share_cuda_()

                state["ipc_extra"] = {
                    "handle": handle,
                    "shape": self.shape,
                    "dtype": self.dtype,
                    "stride": self.stride(),
                    "device_index": self.device.index,
                    "storage_offset": self.storage_offset(),
                }
                state["tensor_data"] = None
            except Exception as e:
                # Failed to get CUDA IPC handle (possibly tp). Falling back to default transport.
                state["metadata"]["transport_mode"] = "default"
                state["tensor_data"] = self.as_subclass(torch.Tensor)
        else:
            state["metadata"]["transport_mode"] = "default"
            state["tensor_data"] = self.as_subclass(torch.Tensor)

        return state

    def __setstate__(self, state: Dict[str, Any]):
        """
        Called during unpickling. Implements the deserialization logic.

        中译：pickle 反序列化时调用，实现自定义反序列化逻辑。
              cuda_ipc 模式下用句柄在「源设备」上重建张量（映射到同一显存）；
              default 模式下直接用序列化的张量数据恢复；两者皆无时报错。
        """
        self._metadata = state["metadata"]

        transport_mode = self._metadata.get("transport_mode", "default")

        if transport_mode == "cuda_ipc" and state["ipc_extra"] is not None:
            ipc_extra = state["ipc_extra"]
            handle, shape, dtype, stride, source_device_index, s_offset = (
                ipc_extra["handle"],
                ipc_extra["shape"],
                ipc_extra["dtype"],
                ipc_extra["stride"],
                ipc_extra["device_index"],
                ipc_extra["storage_offset"],
            )

            try:
                # 中译：在源 GPU 上用 IPC 句柄重建 storage 与张量，set_ 到 self（共享同一显存）。
                target_device = torch.device(f"cuda:{source_device_index}")
                with torch.cuda.device(target_device):
                    storage = torch.UntypedStorage._new_shared_cuda(*handle)
                    reconstructed_tensor = torch.empty(
                        0, dtype=dtype, device=target_device
                    ).set_(storage, storage_offset=s_offset, size=shape, stride=stride)
                    self.set_(reconstructed_tensor)
            except Exception as e:
                print(f"Error: Failed to deserialize from CUDA IPC handle ({e}).")
                raise e

        elif state["tensor_data"] is not None:
            self.set_(state["tensor_data"])
        else:
            raise pickle.UnpicklingError(
                "Invalid state for TransportProxyTensor: no tensor data found."
            )

    @property
    def name(self) -> Optional[str]:
        # 中译：张量的名称元数据（可选）。
        return self._metadata.get("name")

    @property
    def fields(self) -> Dict[str, Any]:
        # 中译：附加字段字典（携带额外上下文）。
        return self._metadata.get("fields", {})

    @property
    def transport_mode(self) -> TensorTransportMode:
        # 中译：当前张量的传输模式（cuda_ipc / auto / default）。
        return self._metadata.get("transport_mode", "default")


class MultiModalityDataPaddingPattern:
    """
    Data tokens (like image tokens) often need special handling during padding
    to maintain model compatibility. This class provides the interface for
    implementing different padding strategies for data tokens

    中译：多模态占位符 token 填充策略的抽象基类。
          图像 / 音频等「数据 token」在输入序列里通常需要被替换为该多模态项的 pad_value
          （一个与内容相关的哈希值，用于 RadixAttention 前缀匹配）。不同模型的标记方式不同
          （成对标记 vs 重复单 token），故抽象出统一接口，由子类实现具体替换逻辑。
    """

    @abstractmethod
    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """
        Pad the input ids sequence containing data tokens, and replace them with pad_values

        中译：把输入序列中的数据 token 替换为对应的 pad_value，返回处理后的 input_ids。
        """
        pass


class MultiModalityDataPaddingPatternTokenPairs(MultiModalityDataPaddingPattern):
    """In this pattern, data tokens should be enclosed by special token pairs (e.g. <image>...</image>, data_token_pairs)

    The padded value in a region enclosed by a token pair with be the same one, as the MultimodalDataItem's pad value

    This strategy should be applied when data content is marked by start/end token pairs in the input sequence.

    中译：「成对标记」填充策略。数据 token 被特殊起止标记对包围（如 <image>...</image>）。
          被一对标记包围的区域内的 token 全部替换为同一个 pad_value（即该多模态项的 pad value）。
          适用于「用起止 token 对标记多模态内容」的模型。
    """

    def __init__(
        self,
        data_token_pairs: Optional[List[Tuple[int, int]]],
        data_start_token_ids: Optional[List[int]] = None,
    ) -> None:
        """

        Args:
            data_start_token_ids marks the start of a single multimodal data
            See Minicpmo's slice_start_id for example

        中译：
        参数：
            data_token_pairs：起止标记 token 对列表，如 [(im_start, im_end), ...]。
            data_start_token_ids：标记「单个多模态数据起点」的 token id 集合，
                用于区分多个数据项的边界（参考 Minicpmo 的 slice_start_id）；
                未提供时默认取每个 pair 的起始 token。
        """
        self.data_token_id_pairs = data_token_pairs
        self.data_start_token_ids = data_start_token_ids or [
            s for s, _e in data_token_pairs
        ]

    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """
        This function will replace the data-tokens in between with pad_values accordingly

        中译：把每对起止标记之间的数据 token 替换为相应的 pad_value。
              逐对扫描起止标记，区间内的 token 全部填充为当前数据项的 pad_value，
              并在 mm_inputs.data_offsets 中记录各数据项的起点偏移。
        """
        pad_values = [item.pad_value for item in mm_inputs.mm_items]
        data_token_pairs = self.data_token_id_pairs
        mm_inputs.data_offsets = []
        if data_token_pairs is None:
            # 中译：未显式给出标记对时，退回使用 mm_inputs 自带的图像起止 token。
            data_token_pairs = [mm_inputs.im_start_id, mm_inputs.im_end_id]
        if data_token_pairs is None:
            print_warning_once(
                "No data_token_pairs provided, RadixAttention might be influenced."
            )
            return input_ids
        start_token_ids = {s for s, _e in data_token_pairs}
        end_tokens_ids = {e for _s, e in data_token_pairs}

        padded_ids = []
        last_idx = 0
        data_idx = -1

        # 中译：找出所有起始标记与结束标记在序列中的下标位置。
        start_indices = [i for i, x in enumerate(input_ids) if x in start_token_ids]
        end_indices = [i for i, x in enumerate(input_ids) if x in end_tokens_ids]

        if len(start_indices) != len(end_indices):
            # 中译：起止标记数量不匹配（如被截断）时不做处理，原样返回。
            return input_ids

        for start_idx, end_idx in zip(start_indices, end_indices):
            # 中译：先原样拷贝上一区间结束到本起始标记（含起始标记本身）的部分。
            padded_ids.extend(input_ids[last_idx : start_idx + 1])

            # 中译：若该起始标记是「数据项起点」，则切到下一个数据项并记录其偏移。
            if input_ids[start_idx] in self.data_start_token_ids:
                data_idx += 1
                mm_inputs.data_offsets += [start_idx]

            # 中译：防御性夹取——data_idx 不超过 pad_values 数量上限。
            if data_idx >= len(pad_values):
                data_idx = len(pad_values) - 1

            # 中译：起止标记之间的 token 数量，全部用当前数据项的 pad_value 填充。
            num_tokens = end_idx - start_idx - 1
            pad_value = pad_values[data_idx]
            padded_ids.extend([pad_value] * num_tokens)

            last_idx = end_idx

        # 中译：补上最后一个结束标记到序列末尾的剩余部分。
        padded_ids.extend(input_ids[last_idx:])

        assert len(input_ids) == len(padded_ids), "Length validation fails"
        return padded_ids


class MultiModalityDataPaddingPatternMultimodalTokens(MultiModalityDataPaddingPattern):
    """In this pattern, data tokens should be represented as repetitions of a single token
    e.g. <image><image>....<image>, or <audio><audio>...<audio>

    中译：「重复单 token」填充策略。数据 token 表现为同一个占位 token 的连续重复
          （如 <image><image>...<image>）。这里不依赖起止标记对，而是直接按每个数据项
          预先记录的偏移区间（offsets）把对应区段替换为该项的 pad_value。
    """

    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """
        Replaces multimodal tokens in input_ids with corresponding pad_values from mm_items.
        Each modality (image, audio, video) is handled separately based on its token_id.

        中译：按各数据项的偏移区间，把 input_ids 中的多模态占位 token 替换为对应 pad_value。
              图像 / 音频 / 视频各模态按其 token_id 分别处理。
        """
        if not input_ids or not mm_inputs.mm_items:
            return input_ids

        input_ids_tensor = torch.as_tensor(input_ids)

        # Replace multimodal tokens using per-item offsets
        # 中译：按模态对数据项分组，再按各项偏移区间逐段替换。
        items_by_modality = defaultdict(list)
        for item in mm_inputs.mm_items:
            items_by_modality[item.modality].append(item)

        token_id_map = {
            Modality.IMAGE: mm_inputs.im_token_id,
            Modality.AUDIO: mm_inputs.audio_token_id,
            Modality.VIDEO: mm_inputs.video_token_id,
        }

        for modality, items in items_by_modality.items():
            token_id = token_id_map.get(modality)

            if not items or token_id is None:
                continue

            for i, item in enumerate(items):
                # 中译：offset 为闭区间 (start, end)，故右端 +1 才能覆盖到 end 位置。
                for offset in items[i].offsets:
                    input_ids_tensor[offset[0] : offset[1] + 1] = item.pad_value

        ret_input_ids = input_ids_tensor.tolist()
        return ret_input_ids


# 中译：进程级的多模态 embedding 静态缓存（按多模态项哈希复用编码结果，避免重复 ViT 前向）。
embedding_cache: Optional[MultiModalStaticCache] = None


def init_mm_embedding_cache(max_size: int = 0):
    # 中译：初始化全局多模态 embedding 缓存，max_size 为缓存容量上限（0 表示按实现默认）。
    global embedding_cache
    embedding_cache = MultiModalStaticCache(max_size)


def get_embedding_chunk(
    embedding: torch.Tensor,
    extend_prefix_len: int,
    extend_seq_len: int,
    items_offset: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, int, int]:
    """
    Extract a chunk of embeddings based on the specified prefix length, sequence length, and offset ranges.

    Args:
        embedding: The full embedding tensor to extract a chunk from
        extend_prefix_len: The starting position (prefix length) for extraction
        extend_seq_len: The number of tokens to extract
        items_offset: List of [start, end] offset ranges for multimodal items in the input sequence

    Returns:
        A tuple containing:
        - The extracted embedding chunk as a tensor
        - The start index used for extraction
        - The end index used for extraction

    Note:
        If there's no overlap between the requested range and the offset ranges,
        an empty tensor is returned with zeros for start and end indices.

    中译：根据给定的前缀长度、序列长度与各多模态项的偏移区间，从完整 embedding 中切出
          本次「分块预填充」所需的那一段。
          背景：分块预填充时一个请求被拆成多个 batch 处理，每个 chunk 只覆盖 token 序列的
          一段区间 [extend_prefix_len, extend_prefix_len + extend_seq_len)，而 embedding 是
          按多模态项顺序连续排布的，需要把 token 区间映射到 embedding 行区间。
          返回：(切出的 embedding 片段, 起始行下标, 结束行下标)。若无重叠则返回空张量与 0。
    """
    start_index, end_index = 0, 0
    # 中译：本次 chunk 覆盖的 token 区间（闭区间）：[extend_start_index, extend_end_index]。
    extend_start_index = extend_prefix_len
    extend_end_index = extend_prefix_len + extend_seq_len - 1

    # 中译：累加各多模态项落在 chunk 起点之前 / 内部的 token 数，换算出 embedding 行的起止下标。
    for start, end in items_offset:
        if extend_start_index >= start and extend_start_index <= end:
            start_index += extend_start_index - start
        elif extend_start_index > end:
            start_index += end - start + 1

        if extend_end_index >= start and extend_end_index <= end:
            end_index += extend_end_index - start + 1
        elif extend_end_index > end:
            end_index += end - start + 1
    # some models' embedding is 3-dim, reshape it to 2-dim
    # 中译：部分模型的 embedding 是三维，统一 reshape 成二维 (tokens, hidden) 再切片。
    embedding = embedding.reshape(-1, embedding.shape[-1])
    embedding_chunk = embedding[start_index:end_index]
    return embedding_chunk, start_index, end_index


def _get_precomputed_embedding(
    items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Optional[torch.Tensor]:
    """
    If all items have precomputed_embeddings, return their concatenation.
    If some but not all have precomputed_embeddings, raise NotImplementedError.
    If none have precomputed_embeddings, return None.

    中译：尝试走「预计算 embedding」快路径。
          某些请求的多模态特征已在别处编码好（precomputed_embeddings），无需再跑 ViT。
          - 全部数据项都有预计算 embedding：拼接后按 chunk 切出并返回；
          - 部分有部分没有：抛 NotImplementedError（暂不支持混合）；
          - 都没有：返回 None，交由后续的分块编码路径处理。
    """
    precomputed_embeddings = []
    max_iterations = min(len(items_size) - 1, len(prefix_length))

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue

        items_per_req = items[items_size[i] : items_size[i + 1]]
        extend_len = extend_length[i] if i < len(extend_length) else 0
        items_offset = items_offset_list[i]

        if any(item.precomputed_embeddings is None for item in items_per_req):
            chunk = None
        else:
            req_embeddings = torch.concat(
                [item.precomputed_embeddings for item in items_per_req]
            )
            chunk, _, _ = get_embedding_chunk(
                embedding=req_embeddings,
                extend_prefix_len=prefix_length[i],
                extend_seq_len=extend_len,
                items_offset=items_offset,
            )

        if chunk is None and len(items_per_req) > 1:
            return None
        precomputed_embeddings.append(chunk)

    if any(feature is not None for feature in precomputed_embeddings):
        # 中译：要么全部有预计算 embedding，要么全部没有；混合情形（部分预计算）暂不支持。
        if not all(feature is not None for feature in precomputed_embeddings):
            raise NotImplementedError(
                "MM inputs where only some items are precomputed."
            )

        # Normalize device across chunks before concat.
        # 中译：拼接前先统一各片段所在设备（优先选 CUDA），避免 concat 跨设备报错。
        target_device = next(
            (t.device for t in precomputed_embeddings if t.is_cuda),
            precomputed_embeddings[0].device,
        )
        precomputed_embeddings = [
            t if t.device == target_device else t.to(target_device, non_blocking=True)
            for t in precomputed_embeddings
        ]
        result = torch.concat(precomputed_embeddings)
        # some models embedding is 3-dim, reshape it to 2-dim (similar to get_embedding_chunk)
        result = result.reshape(-1, result.shape[-1])
        return result
    return None


# 中译：多模态特征编码函数的类型别名——输入数据项列表，输出 embedding 张量或 EVS 结果。
DataEmbeddingFunc = Callable[
    [List[MultimodalDataItem]], torch.Tensor | EVSEmbeddingResult
]


def _can_skip_pre_embed_feature_move(data_embedding_func: DataEmbeddingFunc) -> bool:
    """qwen-vl visual forward already moves batched features to the target device.

    instead of performing multiple H2D for each mm feature from all mm_items (followed by concatenation on device),
    for some models which internally performs H2D on concated mm feature, these small H2D calls could be replaced with a single big H2D

    中译：判断是否可以跳过「编码前把特征搬到目标设备」这一步。
          某些模型（如 Qwen3-VL 系列）的视觉前向内部会自行把整批特征一次性 H2D 到目标设备，
          这样就无需为每个 mm_item 单独做小批 H2D（多次小拷贝 + 设备上拼接），
          可用一次大 H2D 替代，减少拷贝次数。仅对特定模型的 get_image/video_feature 返回 True。
    """
    owner = getattr(data_embedding_func, "__self__", None)
    if owner is None:
        return False
    if getattr(data_embedding_func, "__name__", None) not in (
        "get_image_feature",
        "get_video_feature",
    ):
        return False
    return owner.__class__.__name__ in {
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }


def _move_items_to_device(
    items: List[MultimodalDataItem], device: torch.device
) -> None:
    """Move item features to the target device (in-place, non-blocking).

    中译：把各数据项的特征张量「就地、非阻塞」搬到目标设备（仅当尚未在该设备时）。
    """
    for item in items:
        if isinstance(item.feature, torch.Tensor) and item.feature.device != device:
            item.feature = item.feature.to(device, non_blocking=True)


def _get_chunked_embedding_full(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items_per_req: List[MultimodalDataItem],
    items_offset: List[Tuple[int, int]],
    extend_prefix_len: int,
    extend_seq_len: int,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Fallback: encode all items at once, cache combined result, extract chunk.
    Used for non-bundled items or EVS results.

    中译：回退路径——把一个请求的全部数据项「一次性」编码，缓存合并结果，再切出当前 chunk。
          用于「未按图切分（non-bundled）」的项或 EVS（按帧裁剪）结果。
          相比按图编码，此路径可能会编码到不属于当前 chunk 的项，但实现简单、通用。
    """
    item_hashes = [item.hash for item in embedding_items_per_req]
    embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)
    embedding_per_req = embedding_cache.get(item_hashes)

    if embedding_per_req is None:
        # 中译：缓存未命中——必要时把特征搬到设备，调用编码函数，并写回缓存。
        if not _can_skip_pre_embed_feature_move(data_embedding_func):
            _move_items_to_device(embedding_items_per_req, device)
        embedding = data_embedding_func(embedding_items_per_req)
        embedding_per_req = (
            EmbeddingResult(embedding=embedding)
            if isinstance(embedding, torch.Tensor)
            else embedding
        )
        embedding_cache.set(embedding_items_hash, embedding_per_req)

    if isinstance(embedding_per_req, EVSEmbeddingResult):
        # 中译：EVS（高效视频采样）会裁剪掉部分帧，需相应地重排占位符并更新 input_ids 与偏移。
        item = embedding_items_per_req[0]
        input_ids, items_offset = (
            embedding_per_req.redistribute_pruned_frames_placeholders(
                input_ids,
                items_offset,
                item=item,
                extend_prefix_len=extend_prefix_len,
                extend_seq_len=extend_seq_len,
            )
        )

    embedding_per_req_chunk, _, _ = get_embedding_chunk(
        embedding=embedding_per_req.embedding,
        extend_prefix_len=extend_prefix_len,
        extend_seq_len=extend_seq_len,
        items_offset=items_offset,
    )
    return embedding_per_req_chunk, input_ids


def _get_chunked_embedding_by_item(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items_per_req: List[MultimodalDataItem],
    items_offset: List[Tuple[int, int]],
    extend_prefix_len: int,
    extend_seq_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Per-image chunk-aware encoding: only encode images overlapping with the
    current chunk, cache each image individually.
    Items must already be split per-image (each item has exactly one offset).

    中译：「按图、感知 chunk」的编码路径。只编码与当前 chunk 有重叠的图像，并按单图各自缓存。
          要求数据项已按单图切分（每项恰好一个 offset）。相比一次性全编码，能避免对不在
          当前 chunk 的图像做无用的 ViT 前向，且缓存粒度更细、命中率更高。
    """
    chunk_start = extend_prefix_len
    chunk_end = extend_prefix_len + extend_seq_len  # exclusive
    # 中译：chunk_end 为开区间右端（不含）。

    if extend_seq_len <= 0:
        return None

    # 1. Find items overlapping with current chunk
    # offsets are (start, end) inclusive on both ends
    # 中译：第一步——找出与当前 chunk 有重叠的数据项（offset 为左右都闭的区间）。
    overlapping = []
    for idx, (item, offset) in enumerate(zip(embedding_items_per_req, items_offset)):
        start, end = offset
        if end >= chunk_start and start < chunk_end:
            overlapping.append((idx, item, start, end))

    if not overlapping:
        return None

    # 2. Check per-image cache for each overlapping item
    # 中译：第二步——逐个查单图缓存，命中的直接取，未命中的收集起来稍后批量编码。
    cached_embeddings = {}  # idx -> tensor
    miss_items = []  # (idx, item, start, end)
    for idx, item, start, end in overlapping:
        cached = embedding_cache.get_single(item.hash)
        if cached is not None:
            cached_embeddings[idx] = cached.embedding
        else:
            miss_items.append((idx, item, start, end))

    # 3. Batch encode all cache-miss items in one ViT call
    # 中译：第三步——把所有未命中的项合并为一次 ViT 调用（批量编码更高效），
    #       再按每项的 token 数把输出切回各项，并写入单图缓存。
    if miss_items:
        miss_item_list = [item for _, item, _, _ in miss_items]
        _move_items_to_device(miss_item_list, device)
        all_miss_embedding = data_embedding_func(miss_item_list)
        all_miss_embedding = all_miss_embedding.reshape(
            -1, all_miss_embedding.shape[-1]
        )

        # Split output by per-item token count
        # 中译：按每项 token 数（end - start + 1）切分批量编码输出。
        token_counts = [end - start + 1 for _, _, start, end in miss_items]
        split_embeddings = torch.split(all_miss_embedding, token_counts, dim=0)

        for (idx, item, _, _), emb in zip(miss_items, split_embeddings):
            cached_embeddings[idx] = emb
            emb_result = EmbeddingResult(embedding=emb)
            embedding_cache.set(item.hash, emb_result)

    # 4. Assemble chunk: for each overlapping item, extract the overlap slice
    # 中译：第四步——对每个重叠项，仅取它与当前 chunk 相交的那一段，按顺序拼成 chunk embedding。
    chunk_slices = []
    for idx, _, start, end in overlapping:
        emb = cached_embeddings[idx]  # shape: (end - start + 1, hidden)
        # 中译：计算重叠区间（闭区间），再转为相对该项的本地下标做切片。
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end - 1)  # inclusive
        local_start = overlap_start - start
        local_end = overlap_end - start + 1  # exclusive for slicing
        chunk_slices.append(emb[local_start:local_end])

    return torch.cat(chunk_slices, dim=0)


def _get_chunked_prefill_embedding(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """
    Chunked prefill embedding: encode per-request items and extract the chunk.
    Items are already split per-image at processor stage.

    中译：分块预填充的 embedding 入口。逐请求编码其多模态项并切出当前 chunk 的 embedding。
          数据项在预处理阶段已按单图切分。内部按是否「每项单 offset」选择按图路径或合并路径，
          最后把各请求的 chunk embedding 拼接成一整段返回（同时返回可能被 EVS 改写的 input_ids）。
    """
    embedding_list = []
    device = input_ids.device
    # FIXME(Xinyuan): temporary workaround for eagle3
    # 中译：FIXME——针对 eagle3（投机解码）的临时处理，取两者较小值以防越界。
    max_iterations = min(len(items_size) - 1, len(prefix_length))

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            # 中译：该请求没有本模态的数据项，跳过。
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None, items_offset

        extend_prefix_len = prefix_length[i]
        extend_seq_len = extend_length[i] if i < len(extend_length) else 0

        # Skip if all items already prefilled
        # 中译：若该请求的所有数据项都落在已预填充的前缀里（不在本 chunk），整体跳过。
        if all(offset_end < prefix_length[i] for _, offset_end in items_offset):
            continue

        # Use per-image path when all items have exactly one offset (already
        # split per-image) — this avoids encoding images not in this chunk.
        # Fall back to combined path for non-split items or EVS.
        # 中译：所有项都恰好只有一个 offset（已按单图切分）时走「按图路径」，避免编码 chunk 外的图；
        #       否则（未切分或 EVS）走「合并路径」。
        is_per_image = all(len(item.offsets) == 1 for item in embedding_items_per_req)

        if is_per_image:
            chunk_embedding = _get_chunked_embedding_by_item(
                data_embedding_func,
                embedding_items_per_req,
                items_offset,
                extend_prefix_len,
                extend_seq_len,
                device,
            )
            if chunk_embedding is not None:
                embedding_list.append(chunk_embedding)
        else:
            chunk_embedding, input_ids = _get_chunked_embedding_full(
                data_embedding_func,
                embedding_items_per_req,
                items_offset,
                extend_prefix_len,
                extend_seq_len,
                input_ids,
                device,
            )
            if chunk_embedding is not None:
                embedding_list.append(chunk_embedding)

    if len(embedding_list) == 0:
        return None, input_ids
    return torch.concat(embedding_list, dim=0), input_ids


def _get_multimodal_mask(
    input_ids: torch.Tensor, placeholder_tensor: torch.Tensor
) -> torch.Tensor:
    # 中译：生成多模态占位符掩码——input_ids 中等于任一 pad_value 的位置为 True，
    #       末尾扩一维 (N, 1) 以便后续 masked_scatter_ 按 hidden 维广播。
    return torch.isin(input_ids, placeholder_tensor).unsqueeze(-1)


def _adjust_embedding_length(
    embedding: torch.Tensor,
    mask: torch.Tensor,
    logger,
) -> torch.Tensor:
    # 中译：校正 embedding 行数，使其与 input_ids 中占位符 token 的数量一致。
    #       多了：多为分块预填充导致，从尾部截取所需数量（折中方案，并提示可调大 chunked_prefill_size）；
    #       少了：属于内部错误，直接抛异常。
    num_mm_tokens_in_embedding = embedding.shape[0]
    num_mm_tokens_in_input_ids = mask.sum().item()
    if num_mm_tokens_in_input_ids != num_mm_tokens_in_embedding:
        logger.warning(
            f"Number of tokens in multimodal embedding does not match those in the input text. "
            f"Got {num_mm_tokens_in_input_ids} tokens in the text but {num_mm_tokens_in_embedding} "
            f"tokens from multimodal embeddings."
        )
        if num_mm_tokens_in_input_ids < num_mm_tokens_in_embedding:
            chunked_prefill_size = get_global_server_args().chunked_prefill_size
            if chunked_prefill_size != -1:
                logger.warning(
                    "You may want to avoid this issue by raising `chunked_prefill_size`, or disabling chunked prefill"
                )
            # extract from the end: this is a compromise
            if embedding.dim() == 2:
                embedding = embedding[-num_mm_tokens_in_input_ids:, :]
            else:
                num_multimodal = num_mm_tokens_in_input_ids // embedding.shape[0]
                embedding = embedding[-num_multimodal:, :]
        else:
            raise RuntimeError(
                f"Insufficient multimodal embedding length: {num_mm_tokens_in_input_ids=} vs {num_mm_tokens_in_embedding=}. This is an internal error"
            )
    return embedding


def get_embedding_and_mask(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    placeholder_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """
    Generate multimodal embeddings and create a mask for identifying their positions in the input sequence.

    Args:
        data_embedding_func: Function that generates embeddings for multimodal items
        embedding_items: List of multimodal items to embed
        placeholder_tensor: Tensor containing token IDs that serve as placeholders for multimodal content
        input_ids: The input token IDs tensor
        items_size: Cumulative sizes of multimodal items per request
        prefix_length: Prefix lengths for each request
        extend_length: Sequence lengths for each request
        items_offset_list: List of offset ranges for multimodal items in each request

    Returns:
        A tuple containing:
        - The generated embeddings tensor
        - A boolean mask tensor indicating where these embeddings should be placed
        - If EVS is used, the pruned input ids tensor; otherwise, the original input ids tensor

    中译：生成多模态 embedding，并构造一个掩码标识它们在输入序列中的位置。
          流程：先尝试预计算 embedding 快路径，没有则走分块编码路径；再据占位符生成掩码；
          最后校正 embedding 长度与掩码一致。
          返回 (embedding, 掩码, input_ids)；EVS 场景下 input_ids 可能被裁剪改写。
    """
    # 1. Get embedding
    # 中译：第一步——取 embedding。优先用预计算结果，否则走分块预填充编码。
    embedding = _get_precomputed_embedding(
        embedding_items, items_size, prefix_length, extend_length, items_offset_list
    )
    if embedding is None:
        embedding, input_ids = _get_chunked_prefill_embedding(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
            input_ids,
        )
        if embedding is None:
            return None, None, input_ids
    # 2. Get mask
    # 中译：第二步——生成占位符掩码。NPU 上需先同步当前流，确保前面的异步算子完成。
    if _is_npu:
        torch.npu.current_stream().synchronize()
    special_multimodal_mask = _get_multimodal_mask(input_ids, placeholder_tensor)
    # 3. Adjust embedding length if needed
    # 中译：第三步——必要时校正 embedding 行数与掩码中占位符数量一致。
    embedding = _adjust_embedding_length(embedding, special_multimodal_mask, logger)
    return embedding, special_multimodal_mask, input_ids


def embed_mm_inputs(
    mm_inputs_list: List[MultimodalInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    input_embedding: nn.Embedding,
    multimodal_model: nn.Module = None,
    data_embedding_func_mapping: Dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: dict[Modality, List[int]] = None,
    use_deepstack: Dict[Modality, bool] = {},
) -> Optional[torch.Tensor]:
    """
    Embed multimodal inputs and integrate them with text token embeddings.

    Args:
        mm_inputs_list: List of multimodal inputs to process
        extend_prefix_lens: Prefix lengths for each request
        extend_seq_lens: Sequence lengths for each request
        input_ids: Input token IDs tensor
        input_embedding: Embedding layer for text tokens
        placeholder_tokens: Token IDs for multimodal placeholders (uses pad_values if None)

    Returns:
        Combined embedding tensor with multimodal content integrated

    中译：把多模态输入编码为 embedding，并与文本 token 的 embedding 融合成最终输入 embedding。
          总体步骤：
          1. 汇总所有多模态项；2. 按模态分别编码出 embedding 与占位符掩码；
          3. 取文本 token 的 embedding（先把占位符 id 夹取到合法词表范围）；
          4. 用 masked_scatter_ 把多模态 embedding 散布到占位符位置。
          deepstack 模型还会额外维护一组 deepstack embedding。返回 (input_embeds, other_info)。
    """
    other_info = {}
    if mm_inputs_list is None:
        return None

    # 1. Calculate the multimodal data which exists in input_ids, with the help of pad_values
    # we assume that multimodal data are represented with its pad_values in input_ids
    # 中译：第一步——汇总所有请求的多模态项。约定：input_ids 中的多模态内容以其 pad_value 表示。
    item_flatten_list = []
    for mm_inputs in mm_inputs_list:
        item_flatten_list += [item for item in mm_inputs.mm_items if item is not None]

    # deepstack_embeddings: per-modality
    # 中译：deepstack_embeddings 按模态各存一份（deepstack 模型用，非 deepstack 模态置 None）。
    modalities, embeddings, masks, deepstack_embeddings = [], [], [], []

    # 2. Get multimodal embedding separately
    # Try get mm embedding if any
    # 中译：第二步——逐模态分别编码。先按模态筛出数据项，再找到对应的编码函数。
    for modality in Modality.all():
        items = [
            item for item in item_flatten_list if item.is_modality(modality=modality)
        ]
        embedder = (
            None
            if data_embedding_func_mapping is None
            else data_embedding_func_mapping.get(modality, None)
        )
        if embedder is None:
            # "image", "video", etc
            # 中译：未显式给出编码函数时，按模态名约定从模型上取 get_<modality>_feature。
            modality_id = modality.name.lower()
            embedder = getattr(multimodal_model, f"get_{modality_id}_feature", None)
        if len(items) != 0:
            assert embedder is not None, f"no embedding method found for {modality}"
            placeholder_tensor = torch.as_tensor(
                [item.pad_value for item in items],
                device=input_ids.device,
            )
            # calculate per request items length offset
            # 中译：计算每个请求的「数据项累计数量」与「占位符偏移区间」，供分块编码定位。
            items_size = [0]
            items_offsets = []
            for mm_inputs in mm_inputs_list:
                mm_items = [
                    item
                    for item in mm_inputs.mm_items
                    if item.is_modality(modality=modality)
                ]
                items_size.append(items_size[-1] + len(mm_items))
                items_offsets.append(
                    flatten_nested_list([item.offsets for item in mm_items])
                )

            embedding, mask, input_ids = get_embedding_and_mask(
                data_embedding_func=embedder,
                embedding_items=items,
                placeholder_tensor=placeholder_tensor,
                input_ids=input_ids,
                items_size=items_size,
                prefix_length=extend_prefix_lens,
                extend_length=extend_seq_lens,
                items_offset_list=items_offsets,
            )

            if use_deepstack.get(modality, None) and embedding is not None:
                # 中译：deepstack 模型把编码结果拆成「主 embedding」与「deepstack embedding」两部分。
                embedding, deepstack_embedding = (
                    multimodal_model.separate_deepstack_embeds(embedding)
                )
                deepstack_embeddings += [deepstack_embedding]
            else:
                deepstack_embeddings += [None]
            modalities += [modality]
            embeddings += [embedding]
            masks += [mask]

    # 3. Get input embeddings
    # 中译：第三步——取文本 token 的 embedding。
    vocab_size = input_embedding.num_embeddings
    # Important: clamp after getting original multimodal regions
    # Clamp input ids. This is because the input_ids for the multimodal tokens are
    # filled with the hash values of the multimodal for the prefix matching in the radix attention.
    # There values are useless because their embeddings will be replaced by vision embeddings anyway.
    # 中译：务必在取得多模态区域信息之后再夹取 input_ids。多模态 token 的 id 实为多模态内容的
    #       哈希值（用于 RadixAttention 前缀匹配），超出词表范围，但其 embedding 反正会被
    #       视觉 embedding 覆盖，故先夹取到合法范围以便安全查 embedding 表。
    input_ids.clamp_(min=0, max=vocab_size - 1)
    input_embeds = input_embedding(input_ids)

    # deepstack embedding
    # 中译：deepstack 模型需额外准备一块零填充的 deepstack embedding（hidden 维放大若干倍）。
    if use_deepstack:
        num_deepstack_embeddings = len(multimodal_model.deepstack_visual_indexes)

        deepstack_embedding_shape = input_embeds.shape[:-1] + (
            input_embeds.shape[-1] * num_deepstack_embeddings,
        )
        # a zero-filled embedding, with the same length of input_embeds, but different hidden_size
        input_deepstack_embeds = torch.zeros(
            deepstack_embedding_shape,
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )

        other_info["input_deepstack_embeds"] = input_deepstack_embeds

    # 4. scatter embeddings into input embedding
    # masked_scatter_ avoids the cudaStreamSynchronize that torch.where triggers.
    # 中译：第四步——把各模态 embedding 散布（scatter）到文本 embedding 的占位符位置。
    #       用 masked_scatter_ 而非 torch.where，可避免后者触发的 cudaStreamSynchronize（更快）。
    def _scatter(dest, mask, src):
        dest.masked_scatter_(mask.expand_as(dest), src.to(dest.device, dest.dtype))

    for i, modality, embedding, mask in zip(
        range(len(embeddings)), modalities, embeddings, masks
    ):
        if embedding is None or mask is None:
            continue
        _scatter(input_embeds, mask, embedding)
        if use_deepstack.get(modality, None):
            _scatter(input_deepstack_embeds, mask, deepstack_embeddings[i])

    return input_embeds, other_info


def _embed_mm_inputs_with_split(
    mm_inputs_list: List[MultimodalInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embedding: nn.Embedding,
    multimodal_model: nn.Module = None,
    data_embedding_func_mapping: Dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: dict[Modality, List[int]] = None,
    use_deepstack: Dict[Modality, bool] = {},
):
    """Split batch into precomputed vs non-precomputed, embed each group, merge back.

    中译：把一个 batch 按「全部预计算 embedding」与「非全预计算」分成两组，各自调用
          embed_mm_inputs 编码后，再按原请求位置合并回完整的 input_embeds。
          目的：让 get_embedding_and_mask 每次只面对「同质」的 batch（避免预计算与否混杂），
          配合 enable_adaptive_dispatch_to_encoder 使用。
    """
    precomputed_req_indices = []
    non_precomputed_req_indices = []
    for idx, mm_input in enumerate(mm_inputs_list):
        items = [item for item in mm_input.mm_items if item is not None]
        if items and all(
            getattr(item, "precomputed_embeddings", None) is not None for item in items
        ):
            precomputed_req_indices.append(idx)
        else:
            non_precomputed_req_indices.append(idx)

    embed_kwargs = dict(
        multimodal_model=multimodal_model,
        input_embedding=input_embedding,
        data_embedding_func_mapping=data_embedding_func_mapping,
        placeholder_tokens=placeholder_tokens,
        use_deepstack=use_deepstack,
    )

    if not precomputed_req_indices or not non_precomputed_req_indices:
        # 中译：只有单一组（全预计算或全非预计算）时无需拆分，直接整体编码。
        return embed_mm_inputs(
            mm_inputs_list=mm_inputs_list,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            input_ids=input_ids,
            **embed_kwargs,
        )

    # 中译：两组都存在——需要按请求在拼接序列中的 token 起点切片、分组编码、再写回。
    all_seq_lens = forward_batch.extend_seq_lens_cpu
    mm_batch_indices = [
        i for i, mm in enumerate(forward_batch.mm_inputs) if mm is not None
    ]
    token_starts = []
    cumulative = 0
    for sl in all_seq_lens:
        token_starts.append(cumulative)
        cumulative += sl

    vocab_size = input_embedding.num_embeddings
    input_embeds = input_embedding(input_ids.clamp(min=0, max=vocab_size - 1))
    other_info = {}

    input_deepstack_embeds = None
    if use_deepstack and multimodal_model is not None:
        num_deepstack_embeddings = len(multimodal_model.deepstack_visual_indexes)
        input_deepstack_embeds = torch.zeros(
            input_ids.shape[0],
            input_embedding.embedding_dim * num_deepstack_embeddings,
            device=input_ids.device,
            dtype=input_embedding.weight.dtype,
        )
        other_info["input_deepstack_embeds"] = input_deepstack_embeds

    for group_req_indices in [precomputed_req_indices, non_precomputed_req_indices]:
        # 中译：对每组——抽出该组各请求的输入、前缀/序列长度，并按 token 起点切出其 input_ids 拼接。
        sub_mm_inputs = [mm_inputs_list[i] for i in group_req_indices]
        sub_prefix_lens = [extend_prefix_lens[i] for i in group_req_indices]
        sub_seq_lens = [extend_seq_lens[i] for i in group_req_indices]
        group_batch_indices = [mm_batch_indices[i] for i in group_req_indices]
        sub_slices = [
            input_ids[token_starts[bi] : token_starts[bi] + all_seq_lens[bi]]
            for bi in group_batch_indices
        ]
        sub_input_ids = torch.cat(sub_slices)

        sub_embeds, sub_info = embed_mm_inputs(
            mm_inputs_list=sub_mm_inputs,
            extend_prefix_lens=sub_prefix_lens,
            extend_seq_lens=sub_seq_lens,
            input_ids=sub_input_ids,
            **embed_kwargs,
        )

        # 中译：把本组编码出的 sub_embeds 按各请求在完整序列中的原始位置写回 input_embeds。
        offset = 0
        for bi in group_batch_indices:
            req_len = all_seq_lens[bi]
            start = token_starts[bi]
            input_embeds[start : start + req_len] = sub_embeds[
                offset : offset + req_len
            ]
            if (
                input_deepstack_embeds is not None
                and "input_deepstack_embeds" in sub_info
            ):
                input_deepstack_embeds[start : start + req_len] = sub_info[
                    "input_deepstack_embeds"
                ][offset : offset + req_len]
            offset += req_len

    return input_embeds, other_info


def general_mm_embed_routine(
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    language_model: nn.Module,
    multimodal_model: Optional[nn.Module] = None,
    data_embedding_funcs: Dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: Optional[dict[Modality, List[int]]] = None,
    use_deepstack: Dict[Modality, bool] = {},
    **kwargs,
) -> torch.Tensor:
    """
    Process multimodal inputs and forward through language model.

    Args:
        input_ids: Input token IDs tensor
        forward_batch: Batch information for model forward pass
        language_model: Base language model to use
        data_embedding_funcs: A dictionary mapping from modality type to the corresponding embedding function.
        placeholder_tokens: Token IDs for multimodal placeholders
        use_deepstack: Whether to use deepstack embeddings for each modality, default False
        **kwargs: Additional arguments passed to language model

    Returns:
        Hidden states from language model forward pass

    中译：多模态推理的总入口例程。先把多模态输入融合进文本 embedding（仅在前向需要时、
          且为流水线并行的首个 rank 上执行），再调用语言模型完成前向，返回 hidden states。
          decode / target_verify 阶段或无多模态输入时，直接走普通文本 embedding。
    """
    assert hasattr(language_model, "get_input_embeddings")
    embed_tokens = language_model.get_input_embeddings()
    # 中译：非流水线并行，或处于流水线首 rank 时，才负责构造输入 embedding。
    if not hasattr(language_model, "pp_group") or language_model.pp_group.is_first_rank:
        # 中译：仅在「扩展（extend/prefill）阶段且确有多模态输入」时融合多模态 embedding。
        if (
            not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.contains_mm_inputs()
        ):
            mm_inputs_list = [
                mm_input for mm_input in forward_batch.mm_inputs if mm_input is not None
            ]
            extend_prefix_lens = [
                prefix_len
                for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            extend_seq_lens = [
                seq_len
                for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            server_args = get_global_server_args()
            if server_args and server_args.enable_adaptive_dispatch_to_encoder:
                # Split by precomputed vs non-precomputed so get_embedding_and_mask only sees uniform batches
                # 中译：开启自适应分派到编码器时，按预计算与否拆分 batch，保证下游只处理同质 batch。
                input_embeds, other_info = _embed_mm_inputs_with_split(
                    mm_inputs_list=mm_inputs_list,
                    extend_prefix_lens=extend_prefix_lens,
                    extend_seq_lens=extend_seq_lens,
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    input_embedding=embed_tokens,
                    multimodal_model=multimodal_model,
                    data_embedding_func_mapping=data_embedding_funcs,
                    placeholder_tokens=placeholder_tokens,
                    use_deepstack=use_deepstack,
                )
            else:
                input_embeds, other_info = embed_mm_inputs(
                    mm_inputs_list=mm_inputs_list,
                    extend_prefix_lens=extend_prefix_lens,
                    extend_seq_lens=extend_seq_lens,
                    input_ids=input_ids,
                    input_embedding=embed_tokens,
                    multimodal_model=multimodal_model,
                    data_embedding_func_mapping=data_embedding_funcs,
                    placeholder_tokens=placeholder_tokens,
                    use_deepstack=use_deepstack,
                )

            # add for qwen3_vl deepstack
            # 中译：Qwen3-VL 等 deepstack 模型——把 deepstack embedding 透传给语言模型前向。
            if use_deepstack:
                kwargs["input_deepstack_embeds"] = other_info["input_deepstack_embeds"]
            # Offload GPU features to CPU instead of discarding them to balance memory
            # efficiency and data persistence.
            # In chunked-prefill, a request is processed across multiple batches, and
            # the original multimodal data must remain accessible until the entire
            # prefill phase is complete. Since the multimodal embedding cache is
            # best-effort, offloading to CPU ensures we have a reliable fallback
            # if a cache miss occurs in subsequent chunks, while still freeing up
            # critical GPU memory.
            # 中译：把已用过的 GPU 特征卸载（offload）到 CPU 而非直接丢弃，兼顾显存与数据可用性。
            #       分块预填充下一个请求跨多个 batch 处理，原始多模态数据须在整个预填充期间可用；
            #       由于 embedding 缓存是「尽力而为」的，后续 chunk 万一未命中可从 CPU 副本回退，
            #       同时又能及时释放宝贵的 GPU 显存。
            if mm_inputs_list:
                for mm_input_obj in mm_inputs_list:
                    if mm_input_obj and hasattr(mm_input_obj, "mm_items"):
                        for mm_item in mm_input_obj.mm_items:
                            feature = getattr(mm_item, "feature", None)
                            if isinstance(feature, torch.Tensor) and feature.is_cuda:
                                mm_item.feature = feature.to("cpu", non_blocking=True)
                            if get_global_server_args().language_only:
                                precomputed_embeddings = getattr(
                                    mm_item, "precomputed_embeddings", None
                                )
                                if (
                                    isinstance(precomputed_embeddings, torch.Tensor)
                                    and precomputed_embeddings.is_cuda
                                ):
                                    mm_item.precomputed_embeddings = (
                                        precomputed_embeddings.to(
                                            "cpu", non_blocking=True
                                        )
                                    )
            forward_batch.mm_inputs = None
            forward_batch.mm_input_embeds = input_embeds
        else:
            # 中译：无多模态融合需求时，直接对 input_ids 取文本 embedding。
            input_embeds = embed_tokens(input_ids)
        # Copy to pre-allocated buffer if available (for CUDA graph address stability)
        # 中译：若有预分配缓冲区则拷入，保证 CUDA Graph 捕获时输入张量地址稳定。
        if forward_batch.input_embeds is not None:
            forward_batch.input_embeds.copy_(input_embeds)
            input_embeds = forward_batch.input_embeds
    else:
        # 中译：流水线并行的非首 rank 不构造 embedding（由前序 rank 经 hidden states 传入）。
        input_embeds = None

    hidden_states = language_model(
        input_ids=None,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        **kwargs,
    )
    return hidden_states


def get_multimodal_data_bounds(
    input_ids: torch.Tensor, pad_values: List[int], token_pairs: List[Tuple[int, int]]
) -> torch.Tensor:
    """
    Returns a tensor indicating the bounds of multimodal data (images, video, audio, etc.)

    Returns:
        [bounds_count, 2]

    中译：返回各多模态数据区段的边界（起、止下标），形状 [区段数, 2]。
          通过起止标记 token 在序列中定位，配对后取标记内侧区间 (start+1, end-1)。
          含一个特例修复：im_start 可能被作为前缀缓存而缺失，按条件在前面补一个起点。
    """
    # All the multimodal data in the batch should share the same special bound token ids.
    # 中译：同一 batch 内的所有多模态数据应共用同一组起止标记 token id。
    start_tokens = {s for s, _e in token_pairs}
    end_tokens = {e for _s, e in token_pairs}

    assert all(isinstance(t, int) for t in start_tokens)
    assert all(isinstance(t, int) for t in end_tokens)

    start_cond = torch.isin(
        input_ids, torch.as_tensor(start_tokens, device=input_ids.device)
    )
    end_cond = torch.isin(
        input_ids, torch.as_tensor(end_tokens, device=input_ids.device)
    )

    (data_start_tokens,) = torch.where(start_cond)
    (data_end_tokens,) = torch.where(end_cond)

    data_start_tokens_cpu = data_start_tokens.cpu().tolist()
    data_end_tokens_cpu = data_end_tokens.cpu().tolist()

    # the im_start_id sometimes can be cached as prefix, but it is needed for the embedding of the multimodal data
    # 中译：im_start 有时会被作为前缀缓存而不出现在当前序列里，但它对多模态 embedding 是必要的。
    #       若结束标记恰好比起始标记多一个、且序列首 token 是 pad_value、且首个结束标记在首个
    #       起始标记之前，则在起点列表最前补一个 0，补回缺失的起点。
    if len(data_start_tokens_cpu) != len(data_end_tokens_cpu):
        if (
            len(data_start_tokens_cpu) + 1 == len(data_end_tokens_cpu)
            and input_ids[0].item() in pad_values
            and data_end_tokens_cpu
            and data_start_tokens_cpu
            and data_end_tokens_cpu[0] < data_start_tokens_cpu[0]
        ):
            data_start_tokens_cpu.insert(0, 0)
    valid_mm_data_nums = min(len(data_start_tokens_cpu), len(data_end_tokens_cpu))

    if valid_mm_data_nums == 0:
        return torch.zeros((0, 2), device=input_ids.device)

    # Filter out pairs where start_token >= end_token
    # 中译：过滤掉起点不小于终点的非法配对；有效配对取标记内侧区间 (start+1, end-1)。
    valid_pairs = []
    for i in range(valid_mm_data_nums):
        start_token = data_start_tokens_cpu[i]
        end_token = data_end_tokens_cpu[i]
        if start_token < end_token:
            valid_pairs.append((start_token + 1, end_token - 1))

    if not valid_pairs:
        return torch.zeros((0, 2), device=input_ids.device)

    # Convert valid pairs to tensor
    valid_pairs_tensor = torch.as_tensor(valid_pairs, device=input_ids.device)
    return valid_pairs_tensor


def data_hash(data) -> int:
    # 中译：对字节数据做 SHA256，取前 8 字节转成无符号整数作为哈希值（用于多模态项去重/缓存）。
    hash_bytes = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def tensor_hash(tensor_list) -> int:
    """
    hash a tensor or a tensor list

    中译：对单个张量或张量列表计算哈希值。
          GPU 路径：拼接后用 triton 核（gpu_tensor_hash）在 GPU 上算哈希，避免回传 CPU；
          CPU 路径：逐张量增量喂入 SHA256，避免一次性 concat 的内存峰值。
    """
    tensor = tensor_list
    if isinstance(tensor_list, list):
        tensor_list = flatten_nested_list(tensor_list)
        tensors = [
            x.flatten() if isinstance(x, torch.Tensor) else x for x in tensor_list
        ]
        # GPU path: concat + triton hash (unchanged)
        if any(isinstance(t, torch.Tensor) and t.is_cuda for t in tensors):
            tensor = torch.concat(tensors)
            return gpu_tensor_hash(tensor.cuda())
        # CPU path: hash each tensor incrementally without concat
        hasher = hashlib.sha256()
        for t in tensors:
            t = t.detach().contiguous()
            hasher.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
        hash_bytes = hasher.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big", signed=False)

    # Single tensor
    if tensor.is_cuda:
        return gpu_tensor_hash(tensor.cuda())
    tensor = tensor.detach().contiguous()
    hasher = hashlib.sha256()
    hasher.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    hash_bytes = hasher.digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def hash_feature(f):
    # 中译：对多模态「特征」计算哈希值，按其具体类型分派：
    #       列表（含 ShmPointer / 张量 / 普通值）、np.ndarray、torch.Tensor、
    #       CUDA IPC 代理、共享内存指针（优先用预计算哈希），其余退回 data_hash。
    if isinstance(f, list):
        if len(f) > 0 and isinstance(f[0], ShmPointerMMData):
            return tensor_hash([x.tensor for x in f])
        if len(f) > 0 and isinstance(f[0], torch.Tensor):
            return tensor_hash(f)
        return data_hash(tuple(flatten_nested_list(f)))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        hasher = hashlib.sha256()
        hasher.update(memoryview(arr))
        hash_bytes = hasher.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big", signed=False)
    elif isinstance(f, torch.Tensor):
        return tensor_hash([f])
    elif isinstance(f, CudaIpcTensorTransportProxy):
        reconstruct_t = f.reconstruct_on_target_device(torch.cuda.current_device())
        return tensor_hash([reconstruct_t])
    elif isinstance(f, ShmPointerMMData):
        if f.precomputed_hash is not None:
            return f.precomputed_hash
        return tensor_hash([f.tensor])
    return data_hash(f)


def extend_mrope_positions_for_retracted_request(
    mrope_positions: torch.Tensor, output_ids_len: int
) -> torch.Tensor:
    """
    Extend mrope_positions for retracted requests by appending positions for output_ids.

    When a request is retracted and has multimodal inputs with mrope_positions,
    we need to extend the positions to cover the output_ids that were already generated.
    For pure text tokens, all three dimensions use the same incremental sequence.

    Args:
        mrope_positions: The original mrope positions tensor, shape (3, origin_input_ids_len)
        output_ids_len: The number of output tokens to generate positions for

    Returns:
        Extended mrope_positions tensor with shape (3, origin_input_ids_len + output_ids_len)

    中译：为「被回退（retracted）」的请求扩展 mrope_positions（多模态旋转位置编码）。
          请求被回退重算时，已生成的 output_ids 也需要对应的位置编码。纯文本 token 的三维
          位置编码相同，故直接从最后一个位置 +1 起生成递增序列，拼接到原位置之后。
    """
    if output_ids_len <= 0:
        return mrope_positions

    # Get the last position value corresponding to origin_input_ids
    # mrope_positions shape: (3, origin_input_ids_len)
    # 中译：取原始输入对应的最后一个位置值（三维各一个）。
    last_position = mrope_positions[:, -1]  # shape: (3,)

    # Generate pure text mrope positions for output_ids
    # All three dimensions for pure text are the same incremental sequence
    # 中译：为 output_ids 生成纯文本位置编码——三维共用同一段从 last+1 开始的递增序列。
    start_pos = last_position[0] + 1  # Start from last position + 1
    output_positions = (
        torch.arange(
            start_pos,
            start_pos + output_ids_len,
            dtype=torch.int64,
            device=mrope_positions.device,
        )
        .unsqueeze(0)
        .expand(3, -1)
    )  # shape: (3, output_ids_len)

    # Concatenate to the original mrope_positions
    return torch.cat([mrope_positions, output_positions], dim=1)


def _get_length(value):
    # 中译：获取「第 0 维长度」的通用辅助：张量/ndarray 取 shape[0]（标量返回 None），
    #       list/tuple 取 len，其余返回 None。用于把「打包项」按数量切分。
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.shape[0] if value.ndim > 0 else None
    if isinstance(value, np.ndarray):
        return value.shape[0] if value.ndim > 0 else None
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _slice_value(value, start, end):
    # 中译：对张量/ndarray/list/tuple 等做 [start:end] 切片的通用辅助；切片失败则原样返回。
    if isinstance(value, torch.Tensor):
        return value[start:end]
    if isinstance(value, np.ndarray):
        return value[start:end]
    if isinstance(value, list):
        return value[start:end]
    if isinstance(value, tuple):
        return value[start:end]
    try:
        return value[start:end]
    except Exception:
        return value


def _slice_model_data(
    data: dict,
    index: int,
    start: int,
    end: int,
    num_items: int,
    total_feature_len: Optional[int],
):
    # 中译：把模型专属数据（model_specific_data）按「打包项拆单」的需要做切分。
    #       按各字段第 0 维长度判断其语义：长度==项数 → 按 index 取该项；
    #       长度==特征总长 → 按 [start:end] 切对应特征段；其余原样保留（共享）。
    sliced = {}
    for key, value in data.items():
        length = _get_length(value)
        if length == num_items:
            sliced[key] = _slice_value(value, index, index + 1)
        elif total_feature_len is not None and length == total_feature_len:
            sliced[key] = _slice_value(value, start, end)
        else:
            sliced[key] = value
    return sliced


def _try_simple_split(item, num_items, expanded_mm_items):
    """Try to split a bundled item by matching feature dim-0 to offset count.
    Returns True if split succeeded, False otherwise.

    中译：尝试「简单拆分」一个打包项——当特征第 0 维长度恰等于 offset 数量（num_items）时，
          按行逐项拆成 num_items 个独立项（各持一个 offset、一行特征）追加到结果，返回 True；
          否则不动、返回 False（交由调用方回退处理）。
    """
    feature = item.feature if item.feature is not None else item.precomputed_embeddings
    if feature is None:
        return False

    if isinstance(feature, (torch.Tensor, np.ndarray)):
        feature_count = feature.shape[0]
    elif isinstance(feature, (list, tuple)):
        feature_count = len(feature)
    else:
        return False

    if feature_count != num_items:
        # 中译：特征第 0 维与 offset 数量不一致，无法简单拆分。
        return False

    for i in range(num_items):
        # 中译：浅拷贝原项，逐项取第 i 行特征/预计算 embedding、第 i 个 offset，清空哈希待重算。
        new_item = copy.copy(item)
        if item.feature is not None:
            if isinstance(item.feature, (list, tuple)):
                new_item.feature = [item.feature[i]]
            else:
                new_item.feature = item.feature[i : i + 1]
        if item.precomputed_embeddings is not None:
            if isinstance(item.precomputed_embeddings, (list, tuple)):
                new_item.precomputed_embeddings = [item.precomputed_embeddings[i]]
            else:
                new_item.precomputed_embeddings = item.precomputed_embeddings[i : i + 1]
        new_item.offsets = [item.offsets[i]]
        new_data = {}
        for k, v in item.model_specific_data.items():
            if isinstance(v, (list, tuple)) and len(v) == num_items:
                new_data[k] = [v[i]]
            elif (
                isinstance(v, (torch.Tensor, np.ndarray))
                and len(v.shape) > 0
                and v.shape[0] == num_items
            ):
                new_data[k] = v[i : i + 1]
            else:
                new_data[k] = v
        new_item.model_specific_data = new_data
        new_item.hash = None
        expanded_mm_items.append(new_item)
    return True


def get_new_expanded_mm_items(original_mm_items):
    # 中译：把「打包（bundled，一项含多个 offset）」的多模态项展开为「每项单 offset」的列表。
    #       这是按图/按帧编码路径的前置条件。图像按 image_grid_thw 的 patch 数切分，
    #       视频按 video_grid_thw 的帧/patch 数切分；缺网格信息时回退 _try_simple_split；
    #       非打包项原样保留。
    expanded_mm_items = []
    for item in original_mm_items:
        is_bundled = item.offsets is not None and len(item.offsets) > 1

        if is_bundled:
            num_items = len(item.offsets)

            if item.is_image():
                image_grid_thw = item.model_specific_data.get("image_grid_thw")
                grid_len = _get_length(image_grid_thw)
                if image_grid_thw is None or grid_len != num_items:
                    # No grid info — fall back to simple split by feature dim-0
                    # 中译：无网格信息——回退到按特征第 0 维的简单拆分。
                    if not _try_simple_split(item, num_items, expanded_mm_items):
                        expanded_mm_items.append(item)
                    continue

                # 中译：每张图的 patch 数 = grid_thw 各维之积（T*H*W），据此算特征切分下标。
                if isinstance(image_grid_thw, torch.Tensor):
                    patches_per_item = (
                        torch.prod(image_grid_thw, dim=-1).long().tolist()
                    )
                else:
                    patches_per_item = [int(np.prod(grid)) for grid in image_grid_thw]

                cumulative = torch.cumsum(
                    torch.tensor(patches_per_item, dtype=torch.long), dim=0
                )
                slice_indices = [0] + cumulative.tolist()

                feature_len = _get_length(item.feature)
                if feature_len is None:
                    feature_len = _get_length(item.precomputed_embeddings)
                if feature_len is None or slice_indices[-1] != feature_len:
                    # 中译：特征长度缺失或与累计 patch 数对不上，无法精确切分，原样保留。
                    expanded_mm_items.append(item)
                    continue

                total_feature_len = feature_len
                for i in range(num_items):
                    # 中译：按累计下标切出第 i 张图的特征/embedding 与对应 offset，组装为独立项。
                    start, end = slice_indices[i], slice_indices[i + 1]
                    new_item = copy.copy(item)
                    if item.feature is not None:
                        new_item.feature = _slice_value(item.feature, start, end)
                    if item.precomputed_embeddings is not None:
                        new_item.precomputed_embeddings = _slice_value(
                            item.precomputed_embeddings, start, end
                        )
                    new_item.offsets = [item.offsets[i]]
                    new_item.model_specific_data = _slice_model_data(
                        item.model_specific_data,
                        index=i,
                        start=start,
                        end=end,
                        num_items=num_items,
                        total_feature_len=total_feature_len,
                    )
                    new_item.hash = None
                    expanded_mm_items.append(new_item)

            elif item.is_video():
                # 中译：视频项——按 video_grid_thw（每行 [T,H,W]）把多帧拆成「每个视频一项」。
                video_grid_thw = item.model_specific_data.get("video_grid_thw")
                if video_grid_thw is None:
                    if not _try_simple_split(item, num_items, expanded_mm_items):
                        expanded_mm_items.append(item)
                    continue

                # video_grid_thw shape: [num_videos, 3] where each row is [T, H, W]
                # When T > 1, item.offsets contains frames (num_items = total frames)
                # grid_len = num_videos, num_items = sum(T for each video) = total frames
                grid_len = _get_length(video_grid_thw)
                num_videos = grid_len

                # Calculate total frames and frames per video
                if isinstance(video_grid_thw, torch.Tensor):
                    frames_per_video = video_grid_thw[:, 0].long().tolist()
                else:
                    frames_per_video = [int(grid[0]) for grid in video_grid_thw]
                total_frames = sum(frames_per_video)

                # num_items should equal total_frames when T > 1
                # 中译：T>1 时 offset 以「帧」为单位，num_items 应等于总帧数；否则不切分。
                if num_items != total_frames:
                    expanded_mm_items.append(item)
                    continue

                # Calculate patches per video: T * H * W for each video
                # 中译：每个视频的 patch 数 = T*H*W，用于切分特征。
                if isinstance(video_grid_thw, torch.Tensor):
                    patches_per_video = (
                        torch.prod(video_grid_thw, dim=-1).long().tolist()
                    )
                else:
                    patches_per_video = [int(np.prod(grid)) for grid in video_grid_thw]

                # Calculate cumulative patches to get slice indices for each video
                cumulative = torch.cumsum(
                    torch.tensor(patches_per_video, dtype=torch.long), dim=0
                )
                slice_indices = [0] + cumulative.tolist()

                feature_len = _get_length(item.feature)
                if feature_len is None:
                    feature_len = _get_length(item.precomputed_embeddings)
                if feature_len is None or slice_indices[-1] != feature_len:
                    expanded_mm_items.append(item)
                    continue

                total_feature_len = feature_len
                # Group frames by video: calculate frame indices for each video
                # 中译：按视频分组——算出每个视频在「帧维度（offsets）」上的起止下标。
                frame_start_indices = [0]
                for i in range(num_videos):
                    frame_start_indices.append(
                        frame_start_indices[-1] + frames_per_video[i]
                    )

                # Expand each video into a separate item
                # 中译：把每个视频展开为一个独立项——特征按 patch 切、offsets 按帧切。
                for video_idx in range(num_videos):
                    start, end = (
                        slice_indices[video_idx],
                        slice_indices[video_idx + 1],
                    )
                    frame_start, frame_end = (
                        frame_start_indices[video_idx],
                        frame_start_indices[video_idx + 1],
                    )

                    new_item = copy.copy(item)
                    if item.feature is not None:
                        new_item.feature = _slice_value(item.feature, start, end)
                    if item.precomputed_embeddings is not None:
                        new_item.precomputed_embeddings = _slice_value(
                            item.precomputed_embeddings, start, end
                        )
                    # Group offsets for this video (all frames of this video)
                    new_item.offsets = item.offsets[frame_start:frame_end]
                    # For video_grid_thw, slice the corresponding row [T, H, W] for this video
                    new_item.model_specific_data = _slice_model_data(
                        item.model_specific_data,
                        index=video_idx,
                        start=start,
                        end=end,
                        num_items=num_videos,
                        total_feature_len=total_feature_len,
                    )
                    new_item.hash = None
                    expanded_mm_items.append(new_item)
            else:
                # 中译：图像/视频之外的模态——尝试简单拆分，失败则原样保留。
                if not _try_simple_split(item, num_items, expanded_mm_items):
                    expanded_mm_items.append(item)

        else:
            # 中译：非打包项（单 offset 或无 offset）无需展开，直接保留。
            expanded_mm_items.append(item)
    return expanded_mm_items


class ShmPointerMMData:
    """
    Wraps a tensor to be sent via a shared memory handle.
    This acts as a "pointer" to the tensor data across process boundaries.

    中译：用共享内存（shared_memory）句柄传输张量的包装类，相当于跨进程的张量「指针」。
          构造时把 CPU 张量拷入一块共享内存，pickle 时只传共享内存名与元信息（shape/dtype）；
          接收方按名挂载并零拷贝地视图化（__setstate__），再由 materialize() 克隆为自有内存
          并释放共享内存。用于「非默认」张量传输模式下高效传递多模态特征。
    """

    def __init__(self, tensor: torch.Tensor, precomputed_hash: Optional[int] = None):
        # 中译：确保张量在 CPU 且连续；随后申请同等字节数的共享内存并把数据拷入。
        if not tensor.is_cpu:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.precomputed_hash = precomputed_hash
        nbytes = tensor.numel() * tensor.element_size()
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        try:
            # 中译：以 uint8 视图把张量原始字节整块拷入共享内存缓冲区。
            dst = torch.frombuffer(shm.buf, dtype=torch.uint8)
            dst.copy_(tensor.view(torch.uint8).reshape(-1))
        except BaseException:
            # 中译：拷贝失败需立即关闭并 unlink，避免泄漏共享内存段。
            shm.close()
            shm.unlink()
            raise
        # 中译：仅记录共享内存名；本地句柄关闭（不 unlink），数据仍在共享内存中等待接收方挂载。
        self.shm_name = shm.name
        shm.close()
        self._shm_handle = None

    def __getstate__(self):
        # 中译：序列化时只导出共享内存名与元信息，不导出张量数据本身（避免拷贝）。
        return {
            "shm_name": self.shm_name,
            "shape": self.shape,
            "dtype": self.dtype,
            "precomputed_hash": self.precomputed_hash,
        }

    def __setstate__(self, state):
        # 中译：反序列化时按名挂载共享内存，并「零拷贝」地视图化为张量（不克隆、不 unlink）。
        self.shm_name = state["shm_name"]
        self.shape = state["shape"]
        self.dtype = state["dtype"]
        self.precomputed_hash = state.get("precomputed_hash")
        self._shm_handle = shared_memory.SharedMemory(name=self.shm_name)
        # Zero-copy view into shared memory (no clone, no unlink)
        self.tensor = torch.frombuffer(self._shm_handle.buf, dtype=self.dtype).reshape(
            self.shape
        )

    def materialize(self) -> torch.Tensor:
        """Clone tensor from shm to owned memory, then release shm handle.

        中译：把共享内存中的张量克隆到「自有内存」，随后关闭并 unlink 共享内存段并返回该克隆。
              unlink 由本方法负责（见 __del__ 只 close 不 unlink）；若已被其他 rank 释放则忽略。
        """
        tensor = self.tensor.clone()
        if self._shm_handle is not None:
            self._shm_handle.close()
            try:
                self._shm_handle.unlink()
            except FileNotFoundError:
                pass  # Another rank already unlinked
            self._shm_handle = None
        return tensor

    def __del__(self):
        # Only close; never unlink. Unlinking is materialize()'s job.
        # 中译：析构时只 close 不 unlink——unlink 是 materialize() 的职责，避免提前删除被他人使用的段。
        if getattr(self, "_shm_handle", None) is not None:
            self._shm_handle.close()
            self._shm_handle = None


def _get_is_default_transport():
    # 中译：判断当前张量传输模式是否为 "default"（结果惰性计算并全局缓存一次）。
    #       default 模式下走常规序列化，无需 SHM 包装；非 default 才需要 wrap/unwrap。
    global _is_default_tensor_transport
    if _is_default_tensor_transport is None:
        from sglang.srt.managers.tokenizer_manager import (
            _determine_tensor_transport_mode,
        )

        _is_default_tensor_transport = (
            _determine_tensor_transport_mode(get_global_server_args()) == "default"
        )
    return _is_default_tensor_transport


def _wrap_tensor_or_list(value, precomputed_hash: Optional[int] = None):
    """Wrap a CPU tensor (or list of CPU tensors) in ShmPointerMMData.

    ``precomputed_hash`` is only forwarded for the single-tensor case.
    For list features the item-level hash covers all elements jointly,
    so per-element hashes are not applicable.

    中译：把 CPU 张量（或 CPU 张量列表）包装成 ShmPointerMMData（共享内存指针）。
          precomputed_hash 仅在「单张量」情形下透传；列表特征由项级哈希统一覆盖所有元素，
          故不适用逐元素哈希。非 CPU 张量或其他类型原样返回。
    """
    if isinstance(value, torch.Tensor) and value.is_cpu:
        return ShmPointerMMData(value, precomputed_hash=precomputed_hash)
    elif isinstance(value, (list, tuple)):
        wrapped = [
            (ShmPointerMMData(t) if isinstance(t, torch.Tensor) and t.is_cpu else t)
            for t in value
        ]
        return type(value)(wrapped) if isinstance(value, tuple) else wrapped
    return value


def wrap_shm_features(obj):
    """
    Scan the object for multimodal tensors and wrap them in SHM pointers.

    中译：扫描对象内的多模态张量（feature / precomputed_embeddings），用共享内存指针包装它们，
          以便跨进程高效传输。default 传输模式或跳过分词器初始化时直接原样返回。
    """
    if _get_is_default_transport() or get_global_server_args().skip_tokenizer_init:
        return obj

    if hasattr(obj, "mm_inputs") and obj.mm_inputs:
        for item in obj.mm_inputs.mm_items:
            item_hash = getattr(item, "hash", None)
            if hasattr(item, "feature") and item.feature is not None:
                item.feature = _wrap_tensor_or_list(
                    item.feature, precomputed_hash=item_hash
                )
            if (
                hasattr(item, "precomputed_embeddings")
                and item.precomputed_embeddings is not None
            ):
                item.precomputed_embeddings = _wrap_tensor_or_list(
                    item.precomputed_embeddings, precomputed_hash=item_hash
                )
    return obj


def _feature_has_shm(feat) -> bool:
    """Check whether a single feature (tensor, ShmPointer, or list) contains ShmPointerMMData.

    中译：判断单个特征（张量 / ShmPointer / 列表）中是否含有 ShmPointerMMData。
    """
    if isinstance(feat, ShmPointerMMData):
        return True
    if isinstance(feat, (list, tuple)):
        return any(isinstance(t, ShmPointerMMData) for t in feat)
    return False


def has_shm_features(recv_reqs):
    """Return True if any request in the list contains ShmPointerMMData.

    中译：判断请求列表中是否有任一请求携带共享内存包装的特征（支持嵌套的 batch 请求）。
    """
    for req in recv_reqs:
        if hasattr(req, "batch"):
            if has_shm_features(req.batch):
                return True
        elif hasattr(req, "mm_inputs") and req.mm_inputs:
            for item in req.mm_inputs.mm_items:
                if _feature_has_shm(item.feature):
                    return True
                if _feature_has_shm(getattr(item, "precomputed_embeddings", None)):
                    return True
    return False


def _unwrap_tensor_or_list(value):
    """Restore ShmPointerMMData wrappers back into standard torch.Tensors.

    中译：把 ShmPointerMMData 包装还原为标准 torch.Tensor（调用 materialize 克隆并释放共享内存）；
          列表/元组则逐元素还原；其他类型原样返回。
    """
    if isinstance(value, ShmPointerMMData):
        return value.materialize()
    elif isinstance(value, (list, tuple)):
        unwrapped = [
            t.materialize() if isinstance(t, ShmPointerMMData) else t for t in value
        ]
        return type(value)(unwrapped) if isinstance(value, tuple) else unwrapped
    return value


def unwrap_shm_features(obj):
    """
    Restore ShmPointerMMData wrappers back into standard torch.Tensors.
    Handles both single requests and batch requests.

    中译：把对象内被共享内存包装的多模态特征还原为标准张量（wrap_shm_features 的逆操作）。
          同时支持单请求与 batch 请求。default 传输或跳过分词器初始化时原样返回。
    """
    if _get_is_default_transport() or get_global_server_args().skip_tokenizer_init:
        return obj
    # Handle batch requests
    # 中译：batch 请求——对其中每个子请求递归还原。
    if hasattr(obj, "batch"):
        for sub_obj in obj.batch:
            unwrap_shm_features(sub_obj)
        return obj
    # Handle single requests
    if hasattr(obj, "mm_inputs") and obj.mm_inputs:
        for item in obj.mm_inputs.mm_items:
            if hasattr(item, "feature") and item.feature is not None:
                item.feature = _unwrap_tensor_or_list(item.feature)
            if (
                hasattr(item, "precomputed_embeddings")
                and item.precomputed_embeddings is not None
            ):
                item.precomputed_embeddings = _unwrap_tensor_or_list(
                    item.precomputed_embeddings
                )
    return obj
