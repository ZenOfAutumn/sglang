# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/linear.py"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter

from sglang.kernel_api_logging import wrap_method_with_debug_kernel_once
from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_quant_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_allocation_symmetric,
)
from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    BlockQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
    RowvLLMParameter,
    _ColumnvLLMParameter,
)
from sglang.srt.layers.utils import pad_or_narrow_weight
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_bool_env_var, is_cpu, is_hip, is_npu, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import (
        QuantizationConfig,
        QuantizeMethodBase,
    )

_is_hip = is_hip()
_disable_hip_linear_quant = _is_hip and get_bool_env_var(
    "SGLANG_ROCM_DISABLE_LINEARQUANT"
)

logger = logging.getLogger(__name__)

WEIGHT_LOADER_V2_SUPPORTED = [
    "CompressedTensorsLinearMethod",
    "AWQLinearMethod",
    "GPTQMarlinLinearMethod",
    "Fp8LinearMethod",
    "BlockInt8LinearMethod",
    "MarlinLinearMethod",
    "QQQLinearMethod",
    "GPTQMarlin24LinearMethod",
    "TPUInt8LinearMethod",
    "GPTQLinearMethod",
    "FBGEMMFp8LinearMethod",
    "GPTQLinearAscendMethod",
    "GPTQLinearIntelAMXMethod",
    "GPTQMoEAscendMethod",
    "GPTQMoEIntelAMXMethod",
    "ModelOptFp8LinearMethod",
    "ModelOptFp4LinearMethod",
    "IPEXAWQLinearMethod",
    "PetitNvFp4LinearMethod",
    "QuarkInt4Fp8LinearMethod",
    "QuarkLinearMethod",
]

_is_cpu = is_cpu()
_is_npu = is_npu()


def adjust_marlin_shard(param, shard_size, shard_offset):
    marlin_tile_size = getattr(param, "marlin_tile_size", None)
    if marlin_tile_size is None:
        return shard_size, shard_offset

    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


def adjust_bitsandbytes_4bit_shard(
    param: Parameter, shard_offsets: Dict[str, Tuple[int, int]], loaded_shard_id: str
) -> Tuple[int, int]:
    """Adjust the quantization offsets and sizes for BitsAndBytes sharding."""

    total, _ = shard_offsets["total"]
    orig_offset, orig_size = shard_offsets[loaded_shard_id]

    quantized_total = param.data.shape[0]
    quantized_offset = orig_offset * quantized_total // total
    quantized_size = orig_size * quantized_total // total

    return quantized_size, quantized_offset


def adjust_scalar_to_fused_array(param, loaded_weight, shard_id):
    """For fused modules (QKV and MLP) we have an array of length
    N that holds 1 scale for each "logical" matrix. So the param
    is an array of length N. The loaded_weight corresponds to
    one of the shards on disk. Here, we slice the param based on
    the shard_id for loading.
    """
    qkv_idxs = {"q": 0, "k": 1, "v": 2}

    if isinstance(shard_id, str):
        shard_id = qkv_idxs[shard_id]
    elif not isinstance(shard_id, int):
        raise ValueError(f"Unknown Shard Id {shard_id}")

    # AutoFP8 scales do not have a shape
    # compressed-tensors scales do have a shape
    if len(loaded_weight.shape) != 0:
        assert loaded_weight.shape[0] == 1
        loaded_weight = loaded_weight[0]

    return param[shard_id], loaded_weight


def adjust_shard_offsets(shard_offsets, loaded_weight, dim):
    actual_weight_size = loaded_weight.size(dim)
    target_weight_size = shard_offsets[-1][-1] + shard_offsets[-1][-2]
    if actual_weight_size != target_weight_size:
        new_shard_offsets = []
        new_offset = 0
        for shard_id, shard_offset, shard_size in shard_offsets:
            actual_shard_size = actual_weight_size * shard_size // target_weight_size
            new_shard_offsets.append((shard_id, new_offset, actual_shard_size))
            new_offset += actual_shard_size
        return new_shard_offsets
    return shard_offsets


class LinearBase(torch.nn.Module):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        if quant_config is None:
            from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

        if self.quant_method is not None:
            wrap_method_with_debug_kernel_once(
                self.quant_method,
                "apply",
                op_name=f"sglang.quant_method.{self.quant_method.__class__.__name__}.apply",
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
        )

        # All the linear layer supports quant method.
        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            self.input_size,
            [self.output_size],
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # If the weight on disk does not have a shape, give it one
        # (such scales for AutoFp8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        # The per-tensor quant-scale must be 1 dimension
        if _is_npu:
            if param.size() != loaded_weight.size() and param.size(0) == 1:
                if torch.allclose(loaded_weight, loaded_weight[0]):
                    loaded_weight = loaded_weight[:1]
                else:
                    raise ValueError(f"{loaded_weight} are not all equal")

            if param.dtype == torch.int8 or loaded_weight.dtype == torch.int8:
                assert (
                    param.dtype == loaded_weight.dtype
                ), "init para dtype and loaded weight dtype should be the same"

        assert (
            param.size() == loaded_weight.size()
        ), f"{param.shape=} {param.dtype=} {loaded_weight.shape=} {loaded_weight.dtype=}"
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None
        output = self.quant_method.apply(self, x, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        return s


class ColumnParallelLinear(LinearBase):
    """列并行（Column Parallel）线性层。

    该线性层的计算定义为 Y = XA + b。其中权重矩阵 A 沿它的第二个维度
    （即输出维度 / 列方向）被切分为 A = [A_1, ..., A_p]，每张 GPU 只持有
    其中一个分片 A_i，因此每张卡各自算出 Y_i = X A_i。

    列切分的关键性质：输入 X 在所有卡上是完整（复制）的，计算前不需要通信；
    输出被切分在列方向上，只有当下游需要完整输出时才需要 all-gather。
    这也是为什么列并行常与紧随其后的行并行（RowParallelLinear）配对使用
    （如 MLP 的 up/gate + down、Attention 的 qkv_proj + o_proj）：
    中间结果无需 all-gather，整块只需要一次 all-reduce。

    参数说明:
        input_size: 矩阵 A 的第一个维度（输入特征数，不切分）。
        output_size: 矩阵 A 的第二个维度（全局输出特征数，会被切分）。
        bias: 为 True 时添加 bias（bias 同样按列切分）。
        gather_output: 为 True 时对输出做 all-gather，使每张 GPU 都拿到完整的 Y；
                       为 False 时每张 GPU 只保留自己的分片输出 Y_i = X A_i
                       （TP 中的常见做法）。
        skip_bias_add: 为性能优化预留的开关。开启后本层不会把 bias 加到输出上，
                       而是把 bias 直接返回，交由调用方与其他 element-wise
                       算子（如激活函数）融合执行。
        params_dtype: 参数的数据类型，默认取全局默认 dtype。
        quant_config: 量化配置，决定使用哪个 quant_method 创建权重与执行计算。
        output_sizes: 打包进同一个输出的多段输出尺寸列表，例如 QKV 打包时
                       长度为 3（q/k/v 各一段）。
        prefix: 该层在 state dict 中的名字（含所有父模块前缀），
                       例如 model.layers.0.qkv_proj。
        tp_rank: 显式指定的 TP rank；为 None 时从全局 TP group 获取。
        tp_size: 显式指定的 TP world size；为 None 时从全局 TP group 获取。
        use_presharded_weights: 权重文件中已经是按 TP 切分好的分片，
                       加载时无需再做 narrow 切片。
        skip_block_quant_check: 跳过 block 量化的形状可整除性检查。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        output_sizes: Optional[List[int]] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        use_presharded_weights: bool = False,
        skip_block_quant_check: bool = False,
    ):
        # 基类负责记录全局 in/out 尺寸、解析 params_dtype，
        # 并根据 quant_config 选出本层使用的 quant_method。
        super().__init__(
            input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix
        )

        self.gather_output = gather_output
        self.use_presharded_weights = use_presharded_weights

        # 沿最后一个维度（输出/列方向）切分权重矩阵。
        # tp_rank / tp_size 允许外部显式传入（例如某些模块使用自定义子通信组），
        # 未传入时回退到全局张量并行通信组的 rank 与 world size。
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        assert self.quant_method is not None
        # 本 rank 负责的输出列数；divide() 会断言必须整除，避免出现不均匀切分。
        self.output_size_per_partition = divide(self.output_size, tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # 如果是 QKVParallelLinear 或 MergedColumnParallelLinear 这类打包层，
        # 子类会预先设置 self.output_sizes（每段的全局输出大小），
        # 此处按段分别切分，保证每段在本 rank 上的切片尺寸都被量化方法感知到
        # （block 量化 / scale 的粒度需要按段对齐，不能只看合并后的总长度）。
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, tp_size) for output_size in self.output_sizes
            ]

        if output_sizes is None:
            output_sizes = [output_size]

        # 由量化方法负责创建权重（及可能的 scale / zero point 等附属参数），
        # 并把 weight_loader 绑定到参数上，供模型加载时按 TP 切片写入。
        # 注意：列并行下 input_size_per_partition == input_size（输入不切分）。
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            skip_block_quant_check=skip_block_quant_check,
            weight_loader=(
                # 新版参数类型（vLLM Parameter 体系）走 v2 加载路径，
                # 由参数对象自身实现切分逻辑；否则走本类的传统 weight_loader。
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if bias:
            # bias 与输出同维度，因此也按列切分：每个 rank 只持有自己那段 bias。
            self.bias = Parameter(
                torch.zeros(self.output_size_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    # output_dim=0 告知 loader：bias 的第 0 维是被切分的输出维。
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            # 注册为 None，保持 state_dict / named_parameters 结构一致。
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        """传统（v1）权重加载入口：把 checkpoint 中的完整权重切出本 rank 的分片。

        参数:
            param: 本层已创建好的参数（目标，形状是切分后的）。
            loaded_weight: 从 checkpoint 读出的权重（通常是未切分的全局权重）。
        """
        # output_dim 由 create_weights 时设置，指示该参数哪一维是被切分的输出维。
        # 为 None 表示该参数不按输出维切分（例如 per-tensor scale 之类的标量参数）。
        output_dim = getattr(param, "output_dim", None)
        param_data = param.data

        # GGUF 格式的特殊处理
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            # GGUF 会把「量化类型」作为一个标量张量单独存储，这里取出记录到参数上。
            param.weight_type = loaded_weight.item()

        # 实例化 GGUF 的 UninitializedParameter：
        # GGUF 权重的真实形状要等到读到 checkpoint 才知道，
        # 因此先按加载到的形状（输出维除以 tp_size）分配实际显存。
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            weight_shape = list(loaded_weight.shape)
            if output_dim is not None:
                weight_shape[output_dim] = weight_shape[output_dim] // self.tp_size
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)
            param_data = param.data

        # bitsandbytes 在加载时已经只读取了本 rank 需要的那部分权重，
        # 因此这里无需再做 narrow 切片。
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        if output_dim is not None and not use_bitsandbytes_4bit:
            # 目标参数在输出维上的长度就是本 rank 的分片大小，
            # 起始偏移 = tp_rank * shard_size（列方向连续等分切分）。
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size

            if _is_cpu:
                from sglang.srt.model_loader.weight_utils import (
                    narrow_padded_param_and_loaded_weight,
                )

                # CPU 后端可能对权重做了 padding，需要同时对
                # 目标参数和源权重做「对齐后的 narrow」，避免形状错位。
                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    param_data,
                    loaded_weight,
                    0,  # param_data_start：目标参数中的写入起点
                    start_idx,
                    output_dim,
                    shard_size,
                    not self.use_presharded_weights,
                )
            else:
                # 权重已按 TP 预切分时，loaded_weight 本身就是本 rank 的分片，
                # 直接使用；否则从全局权重中切出属于本 rank 的那一段。
                if not self.use_presharded_weights:
                    loaded_weight = loaded_weight.narrow(
                        output_dim, start_idx, shard_size
                    )

        # 特殊情况：从磁盘加载的 scale 常常是没有形状的 0 维张量
        #（例如 AutoFP8 的 per-tensor scale），统一 reshape 成 [1] 以便拷贝。
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        # 形状必须完全一致，否则说明切分参数（tp_size/tp_rank）或权重布局有误。
        assert (
            param_data.shape == loaded_weight.shape
        ), f"param_data.shape={param_data.shape} != loaded_weight.shape={loaded_weight.shape}"
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: Parameter, loaded_weight: torch.Tensor):
        """新版（v2）权重加载入口。

        与 v1 的区别：切分逻辑下沉到参数对象自身
        （由 vLLM Parameter 体系的 load_column_parallel_weight 实现），
        本层只负责把 tp_rank 等上下文传下去。
        """
        # 特殊情况：从磁盘加载的 scale 常常是没有形状的 0 维张量
        #（例如 AutoFP8 的情况），统一 reshape 成 [1]。
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)

        if isinstance(param, _ColumnvLLMParameter):
            # 标准路径：列并行参数自己知道如何按 tp_rank 切分并写入。
            param.load_column_parallel_weight(
                loaded_weight,
                tp_rank=self.tp_rank,
                use_presharded_weights=self.use_presharded_weights,
            )
        else:
            # FIXME: 这个分支是为了能加载 deepseek v3 awq 权重而存在的，
            # 后续应当修掉它，避免在此处做类型分叉。
            # 在 QuantizedRL 重新加载权重后，参数可能仍然需要 tp_rank。
            try:
                param.load_column_parallel_weight(
                    loaded_weight,
                    tp_rank=self.tp_rank,
                    use_presharded_weights=self.use_presharded_weights,
                )
            except TypeError:
                # 兜底：某些参数实现不接受额外的关键字参数。
                param.load_column_parallel_weight(loaded_weight)

    def forward(self, input_):
        # skip_bias_add 时不把 bias 传给 GEMM，而是在最后原样返回给调用方，
        # 由调用方与后续 element-wise 算子融合。
        bias = self.bias if not self.skip_bias_add else None

        # 矩阵乘：输入是完整的 X，权重是本 rank 的列分片 A_i，
        # 因此输出 output_parallel = X @ A_i 只是完整输出的一个列分片。
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)
        if self.gather_output:
            # 跨各分片做 all-gather，拼回完整输出 Y（沿最后一维拼接）。
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            # 常规 TP 路径：保持输出切分状态，交给下游（通常是 RowParallelLinear）
            # 直接消费，从而省掉一次 all-gather 通信。
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        # print(model) 时展示的额外信息。
        # 注意这里的 output_features 是「本 rank 分片后」的输出维度，而非全局输出维度，
        # 便于直接从打印结果确认切分是否符合预期。
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", gather_output={self.gather_output}"
        return s


class MergedColumnParallelLinear(ColumnParallelLinear):
    """多段打包（Packed）的列并行线性层。

    与 ColumnParallelLinear 类似，但多个逻辑上独立的线性层的权重矩阵
    沿输出维度拼接成一个大矩阵，从而把多次 GEMM 合成一次、提升计算效率。
    典型用法是 SwiGLU MLP 的 gate_proj 与 up_proj 合并为 gate_up_proj。

    关键难点在权重加载：checkpoint 里通常仍是分开存储的多个权重，
    因此每一段需要「先各自按 TP 切分，再写入合并大矩阵中对应的偏移位置」，
    而不能把合并后的大矩阵直接当作一个整体切分。
    举例（output_sizes=[N, N], tp_size=2）：
        全局列布局: [ gate(0..N) | up(N..2N) ]
        rank0 持有:  [ gate[0:N/2] , up[0:N/2] ]
        rank1 持有:  [ gate[N/2:N] , up[N/2:N] ]
    即每个 rank 都同时抽取了每一段的一部分，而不是整段归属某个 rank。
    这正是下面 weight_loader 中 shard_offset / start_idx 两层偏移计算的原因。

    参数说明:
        input_size: 线性层的输入维度（不切分，各段共享）。
        output_sizes: 各段输出维度的列表（均为全局尺寸），
                       合并后的全局输出维度为 sum(output_sizes)。
        bias: 为 True 时添加 bias。
        gather_output: 为 True 时对输出做 all-gather，使每张 GPU 都拿到完整输出；
                       为 False 时每张 GPU 只保留自己的分片输出。
        skip_bias_add: 为性能优化预留的开关。开启后本层不把 bias 加到输出上，
                       而是直接返回，交由调用方与其他 element-wise 算子融合。
        params_dtype: 参数的数据类型。
        quant_config: 量化配置。
        prefix: 该层在 state dict 中的名字（含所有父模块前缀），
                       例如 model.layers.0.qkv_proj。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        use_presharded_weights: bool = False,
    ):
        # 注意：output_sizes 必须在 super().__init__() 之前设置。
        # 因为父类 ColumnParallelLinear.__init__ 会通过
        # hasattr(self, "output_sizes") 来判定要不要按段计算 output_partition_sizes。
        self.output_sizes = output_sizes
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        # 每一段都必须能被 tp_size 整除：否则无法在保持段边界的前提下均匀切分。
        assert all(output_size % tp_size == 0 for output_size in output_sizes)
        self.use_presharded_weights = use_presharded_weights
        super().__init__(
            input_size=input_size,
            # 对父类而言，本层就是一个输出维为 sum(output_sizes) 的普通列并行层。
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
            use_presharded_weights=use_presharded_weights,
        )
        self.prefix = prefix

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ):
        """传统（v1）权重加载入口。

        参数:
            param: 本层已创建的合并参数（目标）。
            loaded_weight: 从 checkpoint 读出的权重。
            loaded_shard_id: 本次要写入的是第几段（对应 output_sizes 的下标）。
                为 None 表示 checkpoint 中已经是合并好的大权重，
                需要在本函数内先拆成各段再递归加载。
        """
        # tuple 形式的 shard id（一次写入多段）仅 v2 参数体系支持。
        if isinstance(loaded_shard_id, tuple):
            if hasattr(param, "load_merged_column_weight"):
                return self.weight_loader_v2(param, loaded_weight, loaded_shard_id)
            raise NotImplementedError(
                "Shard id with multiple indices is not supported in weight_loader, "
                "please use weight_loader_v2 instead."
            )

        # GGUF 的特殊处理：
        # 必须先知道量化类型，才能初始化 GGUF 参数。
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            # 逐段记录每一段的量化类型（各段可能不同）。
            param.data[loaded_shard_id].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            return

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            # GGUF 不预先分配合并大矩阵，而是把各段切好的分片存入
            # data_container 列表，并用 shard_id_map 记录段 -> 下标的映射。
            shard_size = loaded_weight.size(output_dim) // self.tp_size
            start_idx = self.tp_rank * shard_size

            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # AQLM codebook 的特殊情况（固定尺寸、沿 dim 0 拼接）。
        is_metadata = getattr(param, "is_metadata", False)
        # per-tensor scale 的特殊情况：需要把一个标量写入合并后的数组槽位。
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # 进入此分支说明 checkpoint 中权重已经是合并好的（qkv/mlp 融合存储）。
            if output_dim is None:
                # 不按输出维切分的参数（如 per-tensor scale），直接整体拷贝。
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            # 计算各段在「已合并的 loaded_weight」中的 (段号, 起始偏移, 长度)。
            # 若权重已按 TP 预切分，则 loaded_weight 里每段只有 1/tp_size 的长度，
            # 因此偏移也要用缩小后的 effective_size 累加。
            current_shard_offset = 0
            shard_offsets: List[Tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                effective_size = (
                    output_size // self.tp_size
                    if self.use_presharded_weights
                    else output_size
                )
                shard_offsets.append((i, current_shard_offset, effective_size))
                current_shard_offset += effective_size
            # packed_dim：被「多个低位宽数值打包进一个存储单元」的维度
            #（如 4bit 量化下 8 个值存入一个 int32）。
            packed_dim = getattr(param, "packed_dim", None)

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            if _is_cpu:
                # CPU 后端可能对各段做过 padding，需重算偏移以对齐真实布局。
                shard_offsets = adjust_shard_offsets(
                    shard_offsets, loaded_weight, output_dim
                )

            for shard_id, shard_offset, shard_size in shard_offsets:
                # 量化的特殊情况：
                # 若已量化，需要把偏移和长度换算到「打包后」的坐标系上
                #（除以 pack_factor），否则会切错位置。
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset
                    )

                if use_bitsandbytes_4bit:
                    # bitsandbytes 的打包布局与上面的通用推算不同，
                    # 需要基于未量化的原始段偏移表单独换算。
                    index = list(itertools.accumulate([0] + self.output_sizes))
                    orig_offsets = {
                        str(i): (index[i], size)
                        for i, size in enumerate(self.output_sizes)
                    }
                    orig_offsets["total"] = (self.output_size, 0)
                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_offsets, str(shard_id)
                    )

                # 切出该段，然后递归调用自己（带上具体 shard_id），
                # 进入下面的「单段写入」路径完成 TP 切分与拷贝。
                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        # 以下是「单段写入」路径：loaded_weight 对应 output_sizes[loaded_shard_id] 这一段。
        assert loaded_shard_id < len(self.output_sizes)
        if output_dim is not None:
            # 第一层偏移：本段在「本 rank 的合并参数」中的起始位置。
            # 因为每个 rank 只持有每段的 1/tp_size，所以前缀和也要除以 tp_size。
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
            shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
            # 量化的特殊情况：
            # 若已量化，需要把偏移和长度换算到打包后的坐标系上。
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor
                # Marlin 布局的特殊修正。
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            if use_bitsandbytes_4bit:
                # bitsandbytes 已只读本 rank 需要的部分，
                # 因此直接用 loaded_weight 的实际长度作为段长与偏移步长。
                shard_size = loaded_weight.shape[output_dim]
                shard_offset = loaded_weight.shape[output_dim] * loaded_shard_id

            # 把目标参数裁剪到「本段对应的子区域」，后续只往这个子区域写。
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            # 第二层偏移：在「全局的本段权重」中，本 rank 应该取哪一段。
            start_idx = self.tp_rank * shard_size

            if _is_cpu:
                from sglang.srt.model_loader.weight_utils import (
                    narrow_padded_param_and_loaded_weight,
                )

                # CPU 后端：可能存在 padding，需要对目标与源同时做对齐切片。
                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    param_data,
                    loaded_weight,
                    0,  # param_data_start：目标（已裁剪至本段）中的写入起点
                    start_idx,
                    output_dim,
                    shard_size,
                    not use_bitsandbytes_4bit and not self.use_presharded_weights,
                )
            else:
                # bitsandbytes 与预切分权重都已只包含本 rank 需要的部分，
                # 无需在此再做 narrow。
                if not use_bitsandbytes_4bit and not self.use_presharded_weights:
                    # 补齐：处理尺寸未对齐的特例（如 qwen2_5_VL 的 mlp 非 8 对齐），
                    # 此时末端会越过源权重边界，需要 pad 而不能直接 narrow。
                    end_idx = start_idx + shard_size
                    if end_idx > loaded_weight.shape[output_dim]:
                        loaded_weight = pad_or_narrow_weight(
                            loaded_weight, output_dim, start_idx, shard_size
                        )
                    else:
                        loaded_weight = loaded_weight.narrow(
                            output_dim, start_idx, shard_size
                        )

        # AQLM codebook 的特殊情况。
        elif is_metadata:
            # metadata 表示各段尺寸固定、沿 dim 0 拼接，不参与 TP 切分。
            shard_size = loaded_weight.shape[0]
            shard_offset = loaded_shard_id * shard_size
            param_data = param_data.narrow(0, shard_offset, shard_size)

        # 融合场景下 per-tensor scale 的特殊情况：把标量写入合并数组的第 shard_id 个槽位。
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )

        else:
            # 既没有 output_dim，也不属于上述特例：
            # 只能假定该权重在所有分片上都相同（复制而非切分），并给出警告。
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def _load_fused_module_from_checkpoint(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        output_sizes: list[int] | None = None,
    ):
        """处理「MLP 层在 checkpoint 中已经融合存储」的特殊情况。

        这种情况下调用方拿不到 shard id（因为磁盘上就是一个大权重），
        本函数负责把它拆成各段、推导出每段的 shard id，
        再带着 shard id 回调 weight_loader_v2 逐段加载。

        一个存在这种融合层的模型例子：
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

        参数:
            output_sizes: 可选。仅拆分其中部分段时传入；
                为 None 时使用本层完整的 self.output_sizes。
        """

        # 注意：这里的偏移是在「全局（未按 TP 切分）的大权重」坐标系下计算的，
        # 所以直接累加 output_size 而不除 tp_size；TP 切分由后续的
        # load_merged_column_weight 在参数内部完成。
        current_shard_offset = 0
        shard_offsets: List[Tuple[int, int, int]] = []
        output_sizes = output_sizes or self.output_sizes
        for i, output_size in enumerate(output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size
        if _is_cpu:
            from sglang.srt.model_loader.weight_utils import (
                pad_loaded_weight,
            )

            # CPU 后端的参数可能带 padding，先把源权重补齐到同样布局再切。
            loaded_weight = pad_loaded_weight(
                loaded_weight, param.output_dim, output_sizes
            )

        for shard_id, shard_offset, shard_size in shard_offsets:
            # 量化的特殊情况：
            # 若已量化，需要把偏移和长度换算到打包后的坐标系上。
            # v2 体系下这一换算由参数对象自己提供。
            if (
                isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )
            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def _load_merged_block_scale(
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        """处理 MergedColumnParallelLinear 的分块（block-wise）量化 scale 加载。

        与 QKVParallelLinear._load_qkv_block_scale 类似，但面向合并列并行层。
        核心差异：block 量化下 scale 的粒度是「每 block_n 个输出通道一个 scale」，
        因此所有偏移与长度都要先从「通道数」换算成「block 数」。
        """
        weight_block_size = self.quant_method.quant_config.weight_block_size
        block_n, _ = weight_block_size[0], weight_block_size[1]
        # UE8M0 格式下 scale 已是逐通道存储（相当于 block_n == 1），无需再按块换算。
        block_n = 1 if getattr(param, "format_ue8m0", False) else block_n

        # 计算每一段占多少个 block（向上取整）及其起始 block 偏移。
        shard_block_sizes = []
        shard_block_offsets = []
        current_block_offset = 0
        for output_size in self.output_sizes:
            shard_block_size = (output_size + block_n - 1) // block_n
            shard_block_sizes.append(shard_block_size)
            shard_block_offsets.append(current_block_offset)
            current_block_offset += shard_block_size

        if _is_cpu:
            from sglang.srt.model_loader.weight_utils import (
                pad_loaded_weight,
            )

            # CPU 后端可能带 padding，先对齐到相同的分块布局。
            loaded_weight = pad_loaded_weight(
                loaded_weight, param.output_dim, shard_block_sizes
            )

        # 逐段加载
        for shard_id, (shard_block_offset, shard_block_size) in enumerate(
            zip(shard_block_offsets, shard_block_sizes)
        ):
            # 从已合并的 loaded_weight 中切出本段的 scale（全局范围）。
            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_block_offset, shard_block_size
            )

            # 计算考虑 TP 后的单 rank 偏移与长度（仍以 block 为单位）。
            rank_shard_offset = shard_block_offset // self.tp_size
            rank_shard_size = shard_block_size // self.tp_size

            # 写入参数；具体的 rank 切片由参数对象根据 tp_rank 完成。
            param.load_merged_column_weight(
                loaded_weight=loaded_weight_shard,
                shard_id=shard_id,
                shard_offset=rank_shard_offset,
                shard_size=rank_shard_size,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                use_presharded_weights=self.use_presharded_weights,
            )

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ):
        """新版（v2）权重加载入口。

        本层只负责算出「本段在本 rank 参数中的 shard_offset / shard_size」，
        具体的切片与写入交由参数对象的 load_merged_column_weight 完成。

        参数:
            loaded_shard_id: int 表示单段；tuple 表示一次写入多段；
                None 表示 checkpoint 中已融合存储，需先拆段。
        """
        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            if isinstance(param, PerTensorScaleParameter):
                # per-tensor scale 全层只有一个值，不存在段的概念，固定用 shard_id=0。
                param.load_merged_column_weight(
                    loaded_weight=loaded_weight,
                    shard_id=0,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
                return
            elif isinstance(param, BlockQuantScaleParameter):
                # block 量化 scale 需要按 block 粒度拆段，走专用路径。
                self._load_merged_block_scale(param, loaded_weight)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                # 这两类参数不按输出维切分（如沿输入维切分或不切分），
                # 因此无需拆段，直接整体交给参数对象处理。
                param.load_merged_column_weight(
                    loaded_weight=loaded_weight,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
                return
            # loaded_shard_id 为 tuple 时，只拆它指定的那几段；
            # 为 None 时传 None，由下游使用完整的 self.output_sizes。
            output_sizes = (
                [self.output_sizes[idx] for idx in loaded_shard_id]
                if loaded_shard_id
                else None
            )
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(
                param, loaded_weight, output_sizes=output_sizes
            )
            return

        # 单段路径：计算本段在本 rank 合并参数中的偏移与长度。
        assert loaded_shard_id < len(self.output_sizes)

        if isinstance(param, BlockQuantScaleParameter):
            # block 量化 scale：先把通道数向上取整换算成 block 数，再除 tp_size。
            weight_block_size = self.quant_method.quant_config.weight_block_size
            raw_block_n, _ = weight_block_size[0], weight_block_size[1]
            block_n = 1 if getattr(param, "format_ue8m0", False) else raw_block_n
            shard_offset = (
                (sum(self.output_sizes[:loaded_shard_id]) + block_n - 1) // block_n
            ) // self.tp_size
            shard_size = (
                (self.output_sizes[loaded_shard_id] + block_n - 1)
                // block_n
                // self.tp_size
            )
        else:
            # 普通权重：直接用通道数的前缀和 / 段长除以 tp_size。
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
            shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        param.load_merged_column_weight(
            loaded_weight=loaded_weight,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            use_presharded_weights=self.use_presharded_weights,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )


class QKVParallelLinear(ColumnParallelLinear):
    """用于 Attention 的 QKV 投影的列并行线性层。

    把 Attention 中 q / k / v 三个投影矩阵沿输出维度拼接成一个大矩阵，
    从而用一次 GEMM 同时算出 Q、K、V。

    与 MergedColumnParallelLinear 的关键区别：切分粒度是「注意力头」而非任意列。
    当 kv 头数少于 q 头数时（如 MQA / GQA 多查询 / 分组查询注意力），
    q 头被切分，而 kv 头可能需要在多个 rank 上「复制」。

    举例（total_num_heads=8, total_num_kv_heads=2, tp_size=4）：
        num_heads         = 8 / 4 = 2      → 每个 rank 2 个 q 头
        tp_size(4) >= total_num_kv_heads(2) → num_kv_heads = 1
        num_kv_head_replicas = 4 / 2 = 2   → 每个 kv 头被 2 个 rank 共享
        rank0/rank1 都持有 kv 头 0；rank2/rank3 都持有 kv 头 1
    这就是下面加载 k/v 时用 tp_rank // num_kv_head_replicas
    （而不是直接用 tp_rank）来定位源分片的原因。

    参数说明:
        hidden_size: Transformer 的输入隐藏维度。
        head_size: 每个注意力头的维度。
        total_num_heads: 全局（未切分的）query 头数。
        total_num_kv_heads: 全局（未切分的）key/value 头数；
                            为 None 时假定等于 total_num_heads（即标准 MHA）。
        bias: 为 True 时添加 bias。
        skip_bias_add: 为性能优化预留的开关。开启后本层不把 bias 加到输出上，
                       而是直接返回，交由调用方与其他 element-wise 算子融合。
        params_dtype: 参数的数据类型。
        quant_config: 量化配置。
        prefix: 该层在 state dict 中的名字（含所有父模块前缀），
                       例如 model.layers.0.qkv_proj。
        tp_rank: 显式指定的 TP rank；为 None 时从全局 TP group 获取。
        tp_size: 显式指定的 TP world size；为 None 时从全局 TP group 获取。
        load_presharded_attn: attention 权重已按 TP 预切分，加载时无需再切。
        v_head_size: v 头的维度；为 None 时与 head_size 相同。
                       部分模型（如 MLA 类结构）的 v 头维度与 q/k 不一致。
        skip_block_quant_check: 跳过 block 量化的形状可整除性检查。
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        load_presharded_attn: bool = False,
        v_head_size: Optional[int] = None,
        skip_block_quant_check: bool = False,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # 沿最后一个维度（输出/头方向）切分权重矩阵。
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        # q 头常规切分：每个 rank 拿 total_num_heads / tp_size 个头。
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            # kv 头数不够分（GQA/MQA 且 TP 很大）：
            # 每个 rank 至少保留 1 个 kv 头，并在 num_kv_head_replicas 个 rank 之间复制。
            # 这会带来冗余的 KV cache 开销，但能避开跟不上切分粒度的问题。
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            # kv 头数足够：与 q 头一样常规切分，无需复制。
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        # 本 rank 上 q / k / v 各自占用的输出列数。
        self.q_proj_shard_size = self.num_heads * self.head_size
        self.kv_proj_shard_size = self.num_kv_heads * self.head_size
        self.v_proj_shard_size = self.num_kv_heads * self.v_head_size
        input_size = self.hidden_size
        # 注意：这里是先算出本 rank 的列数再乘 tp_size，
        # 而不是直接用 total_num_* 求和。
        # 在 kv 头被复制的情况下（num_kv_head_replicas > 1），
        # 这个「名义上的全局输出维」会大于真实的 kv 权重总量，
        # 因为它把每个副本都计入了——这正是父类能用 divide() 均分的前提。
        output_size = (
            self.num_heads * self.head_size
            + self.num_kv_heads * self.head_size
            + self.num_kv_heads * self.v_head_size
        ) * tp_size
        # 供父类 ColumnParallelLinear 按段计算 output_partition_sizes；
        # 必须在 super().__init__() 之前设置。
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.v_head_size * tp_size,  # v_proj
        ]
        self.use_presharded_weights = load_presharded_attn
        # HIP（ROCm）上某些场景需要禁用线性层量化，此时强制走非量化路径。
        quant_config = None if _disable_hip_linear_quant else quant_config

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
            use_presharded_weights=self.use_presharded_weights,
            skip_block_quant_check=skip_block_quant_check,
        )

    def _get_shard_offset_mapping(self, loaded_shard_id: str):
        """返回 q/k/v 各段在「本 rank 合并参数」中的起始列偏移。

        注意均使用本 rank 的 num_heads / num_kv_heads，因此得到的是本地偏移。
        "total" 表示三段合计的总长度。
        """
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + self.num_kv_heads) * self.head_size
            + self.num_kv_heads * self.v_head_size,
        }
        return shard_offset_mapping.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str):
        """返回 q/k/v 各段在「本 rank 合并参数」中占用的列数。

        注意 v 使用 v_head_size（可能与 q/k 的 head_size 不同）。
        """
        shard_size_mapping = {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.v_head_size,
        }
        return shard_size_mapping.get(loaded_shard_id)

    def _load_fused_module_from_checkpoint(
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        """处理「QKV 层在 checkpoint 中已经融合存储」的特殊情况。

        这种情况下调用方拿不到 shard id（磁盘上就是一个大的 qkv 权重），
        本函数负责把它拆成 q/k/v 三段，再带着 shard id 回调
        weight_loader_v2 逐段加载。

        一个存在这种融合层的模型例子：
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """
        # 注意：这里统一使用 total_num_*，因为偏移是在
        # 「全局（未按 TP 切分）的 qkv 大权重」坐标系下计算的；
        # TP 切分由后续的 load_qkv_weight 在参数内部完成。
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, self.total_num_heads * self.head_size),
            (
                "k",
                self.total_num_heads * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
            (
                "v",
                (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                self.total_num_kv_heads * self.v_head_size,
            ),
        ]

        for shard_id, shard_offset, shard_size in shard_offsets:
            # 量化的特殊情况：
            # 若已量化，需要把偏移和长度换算到打包后的坐标系上。
            if (
                isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )

            # 权重已预切分时直接沿用上一轮的值（无需再从全局权重中切）。
            if not self.use_presharded_weights:
                loaded_weight_shard = loaded_weight.narrow(
                    param.output_dim, shard_offset, shard_size
                )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def _load_qkv_block_scale(
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        """处理 QKV 层的分块（block-wise）量化 scale 加载。

        block 量化下 scale 的粒度是「每 block_n 个输出通道一个 scale」，
        因此所有偏移与长度都需先从「通道数」换算成「block 数」。
        """
        block_n, _ = self.quant_method.quant_config.weight_block_size
        # 全局坐标系下（未切分）各段占多少个 block。
        q_size = self.total_num_heads * self.head_size // block_n
        k_size = self.total_num_kv_heads * self.head_size // block_n
        v_size = self.total_num_kv_heads * self.v_head_size // block_n
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, q_size),
            ("k", q_size, k_size),
            ("v", q_size + k_size, v_size),
        ]
        for shard_id, shard_offset, shard_size in shard_offsets:
            # 先在全局坐标系下切出本段的 scale。
            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )
            # 再算出本段在「本 rank 参数」中的 block 级偏移与长度。
            rank_shard_offset = self._get_shard_offset_mapping(shard_id) // block_n
            rank_shard_size = self._get_shard_size_mapping(shard_id) // block_n
            param.load_qkv_weight(
                loaded_weight=loaded_weight_shard,
                num_heads=self.num_kv_head_replicas,
                shard_id=shard_id,
                shard_offset=rank_shard_offset,
                shard_size=rank_shard_size,
                tp_rank=self.tp_rank,
                use_presharded_weights=self.use_presharded_weights,
            )

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ):
        """新版（v2）权重加载入口。

        本层只负责算出「本段在本 rank 参数中的 shard_offset / shard_size」，
        具体的切片与写入交由参数对象的 load_qkv_weight 完成。

        参数:
            loaded_shard_id: "q" / "k" / "v" 之一；
                为 None 表示 checkpoint 中已融合存储，需先拆段。
        """
        if loaded_shard_id is None:  # 部分模型的特殊情况
            if isinstance(param, PerTensorScaleParameter):
                # per-tensor scale 全层只有一个值，不区分 q/k/v。
                param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=0)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                # 不按输出维（头方向）切分的参数，无需拆段。
                param.load_qkv_weight(loaded_weight=loaded_weight)
                return
            elif isinstance(param, BlockQuantScaleParameter):
                # block 量化 scale 需要按 block 粒度拆段，走专用路径。
                self._load_qkv_block_scale(param, loaded_weight)
                return
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        # 本段在本 rank 合并参数中的偏移与长度（以通道为单位）。
        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)

        if isinstance(param, BlockQuantScaleParameter):
            # block 量化 scale：把通道数向上取整换算成 block 数。
            weight_block_size = self.quant_method.quant_config.weight_block_size
            raw_block_n, _ = weight_block_size[0], weight_block_size[1]
            # UE8M0 格式下 scale 已是逐通道存储，相当于 block_n == 1。
            block_n = 1 if getattr(param, "format_ue8m0", False) else raw_block_n
            shard_offset = (shard_offset + block_n - 1) // block_n
            shard_size = (shard_size + block_n - 1) // block_n

        # num_heads 传的是 num_kv_head_replicas：
        # 参数对象靠它把 tp_rank 换算成源权重中的 kv 头下标
        #（k/v 被多个 rank 共享时，多个 rank 会读取同一份 kv 权重）。
        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=self.num_kv_head_replicas,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
            use_presharded_weights=self.use_presharded_weights,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ):
        """传统（v1）权重加载入口。

        参数:
            param: 本层已创建的合并 qkv 参数（目标）。
            loaded_weight: 从 checkpoint 读出的权重。
            loaded_shard_id: "q" / "k" / "v" 之一；
                为 None 表示 checkpoint 中已融合存储，
                需要在本函数内先拆成三段再递归加载。
        """

        # GGUF 的特殊处理：
        # 必须先知道量化类型，才能初始化 GGUF 参数。
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type and loaded_shard_id is not None:
            # 逐段记录 q/k/v 各自的量化类型（各段可能不同）。
            idx_map = {"q": 0, "k": 1, "v": 2}
            param.data[idx_map[loaded_shard_id]].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            return

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            # GGUF 不预先分配合并大矩阵，而是把各段切好的分片存入
            # data_container 列表，并用 shard_id_map 记录段 -> 下标的映射。
            shard_size = loaded_weight.size(output_dim) // self.tp_size
            start_idx = self.tp_rank * shard_size

            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # AQLM codebook 的特殊情况（固定尺寸、沿 dim 0 拼接）。
        is_metadata = getattr(param, "is_metadata", False)

        # 融合场景下 per-tensor scale 的特殊情况。
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # 进入此分支说明 checkpoint 中权重已经是合并好的（qkv/mlp 融合存储）。
            if output_dim is None:
                # 不按输出维切分的参数（如 per-tensor scale），直接整体拷贝。
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            # 在「全局（未切分）的 qkv 大权重」坐标系下列出三段的偏移与长度，
            # 因此统一使用 total_num_*。
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.total_num_heads * self.head_size),
                (
                    "k",
                    self.total_num_heads * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
                (
                    "v",
                    (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                    self.total_num_kv_heads * self.v_head_size,
                ),
            ]
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

            # packed_dim：被「多个低位宽数值打包进一个存储单元」的维度。
            packed_dim = getattr(param, "packed_dim", None)
            if _is_cpu:
                # CPU 后端可能对各段做过 padding，需重算偏移以对齐真实布局。
                shard_offsets = adjust_shard_offsets(
                    shard_offsets, loaded_weight, output_dim
                )

            for shard_id, shard_offset, shard_size in shard_offsets:
                # 量化权重的特殊情况：
                # 若已量化，需要把偏移和长度换算到打包后的坐标系上。
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor

                    # Marlin 布局的特殊修正。
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset
                    )

                if use_bitsandbytes_4bit:
                    # bitsandbytes 的打包布局与上面的通用推算不同，
                    # 需要基于未量化的原始 qkv 偏移表单独换算。
                    orig_qkv_offsets = {
                        "q": (0, self.total_num_heads * self.head_size),
                        "k": (
                            self.total_num_heads * self.head_size,
                            self.total_num_kv_heads * self.head_size,
                        ),
                        "v": (
                            (self.total_num_heads + self.total_num_kv_heads)
                            * self.head_size,
                            self.total_num_kv_heads * self.v_head_size,
                        ),
                        "total": (
                            (self.total_num_heads + self.total_num_kv_heads)
                            * self.head_size
                            + self.total_num_kv_heads * self.v_head_size,
                            0,
                        ),
                    }

                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_qkv_offsets, shard_id
                    )

                # 权重已预切分时直接沿用上一轮的值（无需再从全局权重中切）。
                if not self.use_presharded_weights:
                    loaded_weight_shard = loaded_weight.narrow(
                        output_dim, shard_offset, shard_size
                    )
                # 递归调用自己（带上具体 shard_id），
                # 进入下面的「单段写入」路径完成 TP 切分与拷贝。
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        # 以下是「单段写入」路径：loaded_weight 对应 q / k / v 中的某一段。
        assert loaded_shard_id in ["q", "k", "v"]

        # 存在 output_dim 时走默认的切分加载流程。
        if output_dim is not None:
            # 第一层偏移：本段在「本 rank 的合并参数」中的起始位置。
            # 注意这里用的是本 rank 的 num_heads / num_kv_heads（本地坐标系）。
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.v_head_size
            # 量化权重的特殊情况：
            # 若已量化，需要把偏移和长度换算到打包后的坐标系上。
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor

                # Marlin 布局的特殊修正。
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            if use_bitsandbytes_4bit:
                # bitsandbytes 已只读本 rank 需要的部分，
                # 此处基于本地（已切分）的头数重算偏移。
                orig_qkv_offsets = {
                    "q": (0, self.num_heads * self.head_size),
                    "k": (
                        self.num_heads * self.head_size,
                        self.num_kv_heads * self.head_size,
                    ),
                    "v": (
                        (self.num_heads + self.num_kv_heads) * self.head_size,
                        self.num_kv_heads * self.v_head_size,
                    ),
                    "total": (
                        (self.num_heads + self.num_kv_heads) * self.head_size
                        + self.num_kv_heads * self.v_head_size,
                        0,
                    ),
                }
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_qkv_offsets, loaded_shard_id
                )

            # 把目标参数裁剪到「本段对应的子区域」，后续只往这个子区域写。
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            # 第二层偏移：在「全局的本段权重」中，本 rank 应该取哪一段。
            # 这是 QKV 层区别于普通合并列并行层的关键之处：
            #   - q 头完全切分，每个 rank 取自己的那一份，所以直接用 tp_rank；
            #   - k/v 头在 num_kv_head_replicas 个 rank 之间复制，
            #     因此需除以副本数，使共享同一 kv 头的多个 rank 映射到
            #     源权重的同一个分片（即读取完全相同的 kv 权重）。
            if loaded_shard_id == "q":
                shard_id = self.tp_rank
            else:
                shard_id = self.tp_rank // self.num_kv_head_replicas
            start_idx = shard_id * shard_size

            if _is_cpu:
                from sglang.srt.model_loader.weight_utils import (
                    narrow_padded_param_and_loaded_weight,
                )

                # CPU 后端：可能存在 padding，需要对目标与源同时做对齐切片。
                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    param_data,
                    loaded_weight,
                    0,  # param_data_start：目标（已裁剪至本段）中的写入起点
                    start_idx,
                    output_dim,
                    shard_size,
                    not use_bitsandbytes_4bit and not self.use_presharded_weights,
                )
            else:
                # bitsandbytes 与预切分权重都已只包含本 rank 需要的部分，
                # 无需在此再做 narrow。
                if not use_bitsandbytes_4bit and not self.use_presharded_weights:
                    loaded_weight = loaded_weight.narrow(
                        output_dim, start_idx, shard_size
                    )

        # AQLM codebook 的特殊情况。
        elif is_metadata:
            # metadata 表示各段尺寸固定、沿 dim 0 拼接，不参与 TP 切分。
            shard_size = loaded_weight.shape[0]
            shard_index = ["q", "k", "v"].index(loaded_shard_id)
            param_data = param_data.narrow(0, shard_index * shard_size, shard_size)
        # 融合场景下 per-tensor scale 的特殊情况：把标量写入合并数组对应的槽位。
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )
        else:
            # 既没有 output_dim，也不属于上述特例：
            # 只能假定该权重在所有分片上都相同（复制而非切分），并给出警告。
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions."
                )

        assert (
            param_data.shape == loaded_weight.shape
        ), f"{param_data.shape=} {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """行并行（Row Parallel）线性层。

    该线性层的计算定义为 Y = XA + b。权重矩阵 A 沿它的第一个维度
    （即输入维度 / 行方向）切分，输入 X 则沿它的第二个维度切分：
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -

    行切分的关键性质（与列并行正好互补）：
    每张卡算出的 X_i A_i 是「形状完整但数值不完整」的部分和，
    因为数学上有 $Y = XA = \\sum_{i=1}^{p} X_i A_i$。
    因此必须做一次 all-reduce 把各卡的部分和相加，才能得到正确结果。
    这与列并行形成对比：列并行输出是「数值完整但形状不完整」（需 all-gather 拼接）。

    两者配对使用（ColumnParallel → RowParallel）时：
    列并行的切分输出可直接作为行并行的切分输入（input_is_parallel=True），
    中间无需任何通信，整个 MLP / Attention 块只需末尾一次 all-reduce。

    参数说明:
        input_size: 矩阵 A 的第一个维度（全局输入特征数，会被切分）。
        output_size: 矩阵 A 的第二个维度（输出特征数，不切分）。
        bias: 为 True 时添加 bias。注意 bias 不参与切分
              （因为输出维不切分，每张卡都持有完整的 bias）。
        input_is_parallel: 为 True 时假定输入已经在各 GPU 上切分好了
                           （通常来自上游的列并行层），本层不再切分；
                           为 False 时本层会把完整输入沿最后一维切开并取本 rank 那份。
        skip_bias_add: 为性能优化预留的开关。开启后本层不把 bias 加到输出上，
                       而是直接返回，交由调用方与其他 element-wise 算子融合。
        params_dtype: 参数的数据类型。
        reduce_results: 为 True 时在本层内完成 all-reduce；
                       为 False 时返回未归约的部分和，由调用方自行归约
                       （例如想把多个层的归约合并成一次通信时）。
                       注意：此时输出在数值上是不完整的。
        quant_config: 量化配置。
        prefix: 该层在 state dict 中的名字（含所有父模块前缀）。
        tp_rank: 显式指定的 TP rank；为 None 时从全局 TP group 获取。
        tp_size: 显式指定的 TP world size；为 None 时从全局 TP group 获取。
        use_presharded_weights: 权重文件中已经是按 TP 切分好的分片，
                       加载时无需再做 narrow 切片。
        use_dp_attention_reduce: 在 attention 的 TP 子组内做 all-reduce
                       （用于 DP attention 场景，此时 attention 与 FFN 的
                       并行组划分不一致），而不是用全局 TP 组。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        use_presharded_weights: bool = False,
        use_dp_attention_reduce: bool = False,
    ):
        # HIP（ROCm）上某些场景需要禁用线性层量化，此时强制走非量化路径。
        quant_config = None if _disable_hip_linear_quant else quant_config
        # 基类负责记录全局 in/out 尺寸、解析 params_dtype，
        # 并根据 quant_config 选出本层使用的 quant_method。
        super().__init__(
            input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        self.use_dp_attention_reduce = use_dp_attention_reduce

        # 沿输入（行）方向切分权重矩阵。
        # tp_rank / tp_size 允许外部显式传入（例如某些模块使用自定义子通信组），
        # 未传入时回退到全局张量并行通信组的 rank 与 world size。
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        # 本 rank 负责的输入行数；divide() 会断言必须整除。
        self.input_size_per_partition = divide(input_size, self.tp_size)
        assert self.quant_method is not None
        self.use_presharded_weights = use_presharded_weights

        # 由量化方法负责创建权重（及可能的 scale 等附属参数）。
        # 与列并行相反：这里传入的是切分后的 input_size_per_partition，
        # 而 output_partition_sizes 直接用完整的 output_size（输出不切分）。
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                # 新版参数类型（vLLM Parameter 体系）走 v2 加载路径，
                # 由参数对象自身实现切分逻辑；否则走本类的传统 weight_loader。
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )

        if bias:
            # bias 与输出同维度，而输出维在行并行下不切分，
            # 因此每张卡都持有尺寸为 output_size 的完整 bias。
            # 为避免 all-reduce 时 bias 被重复累加 tp_size 次，
            # forward 中只在 rank 0 把它融入 GEMM。
            self.bias = Parameter(torch.zeros(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    # output_dim=0 告知 loader：bias 的第 0 维是输出维。
                    # 由于本层 bias 不切分，weight_loader 中只看 input_dim，
                    # 而 bias 没有 input_dim 属性，因此会被整体拷贝。
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            # 注册为 None，保持 state_dict / named_parameters 结构一致。
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        """传统（v1）权重加载入口：把 checkpoint 中的完整权重切出本 rank 的分片。

        与列并行的对应方法相比，此处看的是 input_dim 而不是 output_dim。
        bias 因为没有 input_dim 属性，会自然地跳过切片、被整体拷贝。

        参数:
            param: 本层已创建好的参数（目标，形状是切分后的）。
            loaded_weight: 从 checkpoint 读出的权重（通常是未切分的全局权重）。
        """
        # input_dim 由 create_weights 时设置，指示该参数哪一维是被切分的输入维。
        # 为 None 表示不按输入维切分（如 bias、per-tensor scale 等）。
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        # GGUF 格式的特殊处理
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            # GGUF 会把「量化类型」作为一个标量张量单独存储，这里取出记录到参数上。
            param.weight_type = loaded_weight.item()

        # 实例化 GGUF 的 UninitializedParameter：
        # GGUF 权重的真实形状要等到读到 checkpoint 才知道，
        # 因此先按加载到的形状（输入维除以 tp_size）分配实际显存。
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            weight_shape = list(loaded_weight.shape)
            if input_dim:
                weight_shape[input_dim] = weight_shape[input_dim] // self.tp_size
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

        param_data = param.data
        # bitsandbytes 在加载时已经只读取了本 rank 需要的那部分权重，
        # 预切分权重也同理，因此这里无需再做 narrow 切片。
        if (
            input_dim is not None
            and not use_bitsandbytes_4bit
            and not self.use_presharded_weights
        ):
            # 目标参数在输入维上的长度就是本 rank 的分片大小，
            # 起始偏移 = tp_rank * shard_size（行方向连续等分切分）。
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size

            if _is_cpu:
                from sglang.srt.model_loader.weight_utils import (
                    narrow_padded_param_and_loaded_weight,
                )

                # CPU 后端可能对权重做了 padding，需要同时对
                # 目标参数和源权重做「对齐后的 narrow」，避免形状错位。
                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    param_data,
                    loaded_weight,
                    0,  # param_data_start：目标参数中的写入起点
                    start_idx,
                    input_dim,
                    shard_size,
                )
            else:
                # 补齐：处理尺寸未对齐的特例（如 qwen2_5_VL 的 mlp 非 8 对齐），
                # 此时末端会越过源权重边界，需要 pad 而不能直接 narrow。
                end_idx = start_idx + shard_size
                if end_idx > loaded_weight.shape[input_dim]:
                    loaded_weight = pad_or_narrow_weight(
                        loaded_weight, input_dim, start_idx, shard_size
                    )
                else:
                    loaded_weight = loaded_weight.narrow(
                        input_dim, start_idx, shard_size
                    )

        # 特殊情况：从磁盘加载的 scale 常常是没有形状的 0 维张量
        #（例如 AutoFP8 的 per-tensor scale），统一 reshape 成 [1] 以便拷贝。
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        # 形状必须完全一致，否则说明切分参数（tp_size/tp_rank）或权重布局有误。
        assert (
            param_data.shape == loaded_weight.shape
        ), f"{param_data.shape=} {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor):
        """新版（v2）权重加载入口。

        与 v1 的区别：沿输入维的切分逻辑下沉到参数对象自身
        （由 load_row_parallel_weight 实现），本层只负责传递 tp_rank 等上下文。
        """

        # 特殊情况：从磁盘加载的 scale 常常是没有形状的 0 维张量
        #（例如 AutoFP8 的情况），统一 reshape 成 [1]。
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)

        if isinstance(param, RowvLLMParameter):
            # 这个 `BasevLLMParameter` 定义在 sglang/srt/layers/parameter.py 中，
            # 它支持 tp_rank 和 use_presharded_weights 等额外参数。
            param.load_row_parallel_weight(
                loaded_weight,
                tp_rank=self.tp_rank,
                use_presharded_weights=self.use_presharded_weights,
            )
        else:
            # 这里的 `params` 定义在 `vllm/model_executor/parameter.py` 中，
            # 它不支持额外参数。
            # 不过在 QuantizedRL 重新加载权重后，参数可能仍然需要 tp_rank，
            # 因此先尝试带参调用，失败再兜底。
            try:
                param.load_row_parallel_weight(
                    loaded_weight,
                    tp_rank=self.tp_rank,
                    use_presharded_weights=self.use_presharded_weights,
                )
            except TypeError:
                # 兜底：某些参数实现不接受额外的关键字参数。
                param.load_row_parallel_weight(loaded_weight)

    def forward(self, input_, skip_all_reduce=False, forward_batch=None):
        """前向计算。

        参数:
            input_: 输入张量。是否已沿最后一维切分由 input_is_parallel 决定。
            skip_all_reduce: 临时跳过本次 all-reduce（例如调用方打算把多个
                层的归约合并，或已在外层自行归约）。
            forward_batch: 当前批次信息，仅用于判定能否开启量化通信。
        """
        if self.input_is_parallel:
            # 上游（通常是列并行层）已经交付了切分好的输入，直接用。
            input_parallel = input_
        else:
            # 输入是完整的：本地沿最后一维切开并取本 rank 那份。
            # 这是纯本地切片操作，不涉及通信；contiguous() 是为了后续 GEMM 的内存布局。
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # 矩阵乘。
        assert self.quant_method is not None
        # 只在 rank 0 上把 bias 融入 GEMM（这确保了 TP>1 时
        # bias 不会被重复累加）。
        # 原因：后面的 all-reduce 会把各 rank 的输出求和，
        # 若每个 rank 都加一次 bias，结果就会变成 XA + tp_size * b。
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        # 对称内存（symmetric memory）上下文：让 GEMM 直接把结果写入
        # 可被通信内核直读的缓冲区，从而使后续 all-reduce 能走
        # one-shot / 融合路径，避开一次额外拷贝。
        # 需与实际用于归约的通信组保持一致，所以这里也要区分 DP attention。
        if self.use_dp_attention_reduce:
            symm_ctx = use_symmetric_memory(get_attention_tp_group())
        else:
            symm_ctx = use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            )
        with symm_ctx:
            # 输出是「形状完整但数值不完整」的部分和 X_i A_i。
            output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)

        if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
            # 把各 rank 的部分和相加，得到完整的 Y = sum_i X_i A_i。
            if self.use_dp_attention_reduce:
                # DP attention 场景：在 attention 的 TP 子组内归约。
                output = get_attention_tp_group().all_reduce(output_parallel)
            else:
                # 量化通信：用低精度传输以降低通信量。
                # 仅在非 decode/idle（即 prefill 类、通信量大且对精度不敏感）
                # 且服务端开关打开时才启用；decode 阶段张量小，量化收益低而误差风险高。
                quantize_communications = (
                    (
                        not forward_batch.forward_mode.is_decode_or_idle()
                        and get_global_server_args().enable_quant_communications
                    )
                    if forward_batch is not None
                    else False
                )
                if quantize_communications:
                    output = tensor_model_parallel_quant_all_reduce(output_parallel)
                else:
                    output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            # 不归约：返回的是部分和，调用方必须自己负责完成 all-reduce，
            # 否则结果在数值上是错的。
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias

    def extra_repr(self) -> str:
        # print(model) 时展示的额外信息。
        # 注意与列并行相反：这里被切分的是 input_features（本 rank 分片后），
        # 而 output_features 是完整的全局输出维度。
        s = f"input_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s


class MergedColumnParallelRepeatedLinear(LinearBase):
    """Merged column parallel linear and repeated linear layer.

    TODO: quantization is not supported yet.
    Args:
        input_size: input dimension of the linear layer.
        column_output_sizes: output dimension of the column linear layers.
        repeated_output_sizes: output dimension of the repeated linear layers.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        input_size: int,
        column_output_sizes: List[int],
        repeated_output_sizes: List[int],
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        output_size = sum(column_output_sizes) + sum(repeated_output_sizes)
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.num_column_parallel = len(column_output_sizes)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.output_partition_sizes = [
            divide(x, self.tp_size) for x in column_output_sizes
        ] + repeated_output_sizes
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            skip_block_quant_check=True,
            weight_loader=self.weight_loader,
        )

        self.prefix = prefix

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, input_)

    def weight_loader(
        self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ) -> torch.Tensor:
        output_dim = param.output_dim
        shard_offset = sum(self.output_partition_sizes[:loaded_shard_id])
        shard_size = self.output_partition_sizes[loaded_shard_id]
        param_data = param.data.narrow(output_dim, shard_offset, shard_size)

        if loaded_shard_id < self.num_column_parallel:
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        param_data.copy_(loaded_weight)


class ColumnParallelBatchedLinear(nn.Module):
    """Column parallel batched linear layer.

    TODO: quantization is not supported yet.
    Args:
        batch: batch dimension of the linear layer.
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        dtype: Data type for the parameters.
    """

    def __init__(
        self, batch: int, input_size: int, output_size: int, dtype: torch.dtype
    ):
        super().__init__()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.weight = nn.Parameter(
            torch.empty(batch, output_size // self.tp_size, input_size, dtype=dtype),
            requires_grad=False,
        )
        setattr(self.weight, "weight_loader", self.weight_loader)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.bmm(input, self.weight.transpose(-1, -2))

    def weight_loader(
        self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ) -> torch.Tensor:
        shard_size = self.weight.shape[-2]
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param.data[loaded_shard_id].copy_(loaded_weight)
