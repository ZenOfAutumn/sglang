# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/parameter.py"""

import logging
from fractions import Fraction
from typing import Callable, Optional, Union

import torch
from torch.nn import Parameter

from sglang.srt.environ import envs
from sglang.srt.layers.utils import pad_or_narrow_weight
from sglang.srt.utils import is_cpu

__all__ = [
    "BasevLLMParameter",
    "PackedvLLMParameter",
    "PerTensorScaleParameter",
    "ModelWeightParameter",
    "ChannelQuantScaleParameter",
    "GroupQuantScaleParameter",
    "BlockQuantScaleParameter",
    "PackedColumnParameter",
    "RowvLLMParameter",
]

logger = logging.getLogger(__name__)

_is_cpu = is_cpu()


def _dtype_rank(dtype: torch.dtype) -> Optional[int]:
    """
    把浮点 dtype 映射为一个「精度等级」整数，用于比较两种 dtype 的精度高低。

    等级约定（数值越大精度越高）：
        0 -> 所有 fp8 变体
        1 -> fp16 / bf16
        2 -> fp32
        3 -> fp64

    两个刻意的设计：
      - **同级合并**：fp16 与 bf16 归为同一级，各种 fp8 变体也归为同一级。
        因为它们位宽相同，彼此转换不属于「精度降级」，只是格式差异
        （如 e4m3 与 e5m2 是指数位/尾数位的不同分配，
        fnuz 后缀是 ROCm 上不支持 inf/NaN 的变体）。
      - **返回 None 表示不认识**：整型、bool、fp4 等未列出的类型一律返回 None，
        由调用方决定如何处理（`copy_with_check` 会直接报错，拒绝静默拷贝）。

    :param dtype: 待判定的 torch dtype。
    :return: 精度等级；无法归类时返回 None。
    """
    if dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
        torch.float8_e8m0fnu,
    ):
        return 0
    if dtype in (torch.float16, torch.bfloat16):
        return 1
    if dtype == torch.float32:
        return 2
    if dtype == torch.float64:
        return 3
    return None


def copy_with_check(target: torch.Tensor, loaded_weight: torch.Tensor):
    """
    把 `loaded_weight` 拷贝进 `target`，并在拷贝前**禁止精度降级（downcast）**。

    为什么需要这层检查：
      `Tensor.copy_` 会静默做 dtype 转换。若 checkpoint 里是 bf16 权重，
      而目标参数因为量化配置被建成了 fp8，直接 copy_ 不会报错，
      只会悄悄损失精度——最终表现为模型输出异常，且极难定位。
      因此这里显式拦截「高精度 -> 低精度」的拷贝，让问题在加载阶段就暴露。

    精度比较基于 `_dtype_rank`：bf16 与 fp16 同级，各 fp8 变体同级，
    所以同级之间的互转（如 bf16 -> fp16）被视为合法，不会被拦。

    逃生舱：设置环境变量 `SGLANG_QUANT_ALLOW_DOWNCASTING=1`（默认关闭）
    可放行降级拷贝，用于「checkpoint 是高精度、但确实想在线量化」的场景。

    :param target: 目标参数张量（原地写入）。
    :param loaded_weight: 从 checkpoint 读出的源张量，形状必须与 target 完全一致。
    :raises ValueError: dtype 无法归类，或发生了未被允许的精度降级。
    """

    # 形状必须严格相等：本函数只负责 dtype 校验，切分/narrow 应由调用方提前完成
    assert (
        target.shape == loaded_weight.shape
    ), f"{target.shape=}, {loaded_weight.shape=}"

    # 快路径：dtype 完全相同，不存在任何转换，直接拷贝
    if target.dtype == loaded_weight.dtype:
        target.copy_(loaded_weight)
        return

    # 走到这里说明 dtype 不同，需要判断这次转换是升级还是降级
    target_rank = _dtype_rank(target.dtype)
    loaded_rank = _dtype_rank(loaded_weight.dtype)

    # 任一方是未纳入等级体系的类型（整型、fp4 等）：无法判断精度关系，
    # 宁可报错也不静默转换
    if target_rank is None or loaded_rank is None:
        raise ValueError(
            f"Unsupported copy between dtypes: {target.dtype=}, {loaded_weight.dtype=}"
        )

    # 核心拦截：目标精度低于源精度即为降级，除非用户显式放行
    if target_rank < loaded_rank and not envs.SGLANG_QUANT_ALLOW_DOWNCASTING.get():
        raise ValueError(
            f"Downcasting not allowed: {target.dtype=}, {loaded_weight.dtype=}"
        )

    # 针对 e8m0（DeepSeek-V4 引入的纯指数格式，仅用于存放缩放因子 scale）的额外约束：
    # 它不是普通数值格式，只应拷进 e8m0 本身或 fp32，不能落到 e4m3/e5m2 等尾数格式上。
    #
    # 注意：此处是上游遗留的**无效判断**——`loaded_rank` 是 `_dtype_rank` 返回的 int，
    # 却在和 `torch.dtype` 比较，条件恒为 False，断言实际从未执行。
    # 保持原样以避免改变现有行为；若要修复，应改为比较
    # `loaded_weight.dtype == torch.float8_e8m0fnu` 且
    # `target.dtype in {torch.float8_e8m0fnu, torch.float32}`。
    if loaded_rank == torch.float8_e8m0fnu:
        assert target_rank in {torch.float8_e8m0fnu, torch.float32}

    # 校验通过：执行拷贝，必要的 dtype 转换由 copy_ 自行完成
    target.copy_(loaded_weight)


class BasevLLMParameter(Parameter):
    """
    Base parameter for vLLM linear layers. Extends the torch.nn.parameter
    by taking in a linear weight loader. Will copy the loaded weight
    into the parameter when the provided weight loader is called.
    """

    def __new__(cls, data: torch.Tensor, **kwargs):

        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(self, data: torch.Tensor, weight_loader: Callable):
        """
        Initialize the BasevLLMParameter

        :param data: torch tensor with the parameter data
        :param weight_loader: weight loader callable

        :returns: a torch.nn.parameter
        """

        self._weight_loader = weight_loader

    @property
    def weight_loader(self):
        return self._weight_loader

    def _assert_and_load(self, loaded_weight: torch.Tensor):
        """不做任何切分，直接整体拷贝（要求形状完全一致）。"""
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    # 下面四个方法是权重加载的统一入口（由各个线性层的 weight_loader_v2 调用）。
    # 基类的默认实现均为「不切分、整体拷贝」，适用于不需要沿任何维度
    # 切分的参数（例如 per-tensor scale 这类标量）。
    # 需要切分的子类（_ColumnvLLMParameter / RowvLLMParameter）会覆盖对应方法。
    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        self._assert_and_load(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        self._assert_and_load(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._assert_and_load(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._assert_and_load(loaded_weight)


class _ColumnvLLMParameter(BasevLLMParameter):
    """
    私有基类：为「按列并行（column parallel）切分」的线性层参数提供权重加载能力
    （load_column_parallel_weight / load_merged_column_weight / load_qkv_weight）。

    适用范围：
      - QKV 投影、MLP 的 gate/up 投影等**在磁盘上尚未融合**的权重，
        它们在 SGLang 中会被融合成一个大 Parameter（QKVParallelLinear /
        MergedColumnParallelLinear），因此加载时需要把多个 checkpoint 张量
        分别写进同一块参数的不同区间。

    列并行的核心语义：
      对 y = x @ W^T，W 的形状是 [output_size, input_size]。
      列并行沿 **output_size 维**切分（即本类的 `output_dim`），
      每个 TP rank 只持有 output_size / tp_size 行，最终输出需要 all-gather 才完整。

    因此本类要求子类必须提供 `output_dim`（该参数在张量中对应输出维度的下标，
    通常权重是 0，某些 scale 张量可能不同）。这些方法由各列并行线性层的
    weight_loader 内部调用。
    """

    def __init__(self, output_dim: int, **kwargs):
        # output_dim: 输出维（被 TP 切分的那一维）在本张量中的维度下标。
        # 例如权重 [output_size, input_size] -> output_dim = 0。
        self._output_dim = output_dim
        super().__init__(**kwargs)

    @property
    def output_dim(self):
        """被张量并行切分的输出维下标。"""
        return self._output_dim

    def load_column_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
    ):
        """
        加载「单个未融合」的列并行权重（对应 ColumnParallelLinear）。

        语义：checkpoint 中是完整的 [output_size, ...]，
        本 rank 只取属于自己的那一段 [tp_rank * shard_size, +shard_size)。

        :param loaded_weight: 从 checkpoint 读出的张量。
            - use_presharded_weights=False 时为**完整未切分**权重；
            - use_presharded_weights=True 时已经是本 rank 的分片，直接拷贝。
        :param tp_rank: 当前进程在 TP 组内的 rank。
        :param use_presharded_weights: checkpoint 是否已按 TP 预切分
            （如某些量化/预处理产物），为 True 则跳过 narrow。
        """
        if not use_presharded_weights:
            # 目标参数在输出维上的长度即本 rank 应持有的分片大小
            shard_size = self.data.shape[self.output_dim]

            from sglang.srt.model_loader.weight_utils import (
                narrow_padded_param_and_loaded_weight,
            )

            if _is_cpu:
                # CPU 后端：权重可能被 pad 过（本 rank 分片超出 loaded_weight 实际范围），
                # 该工具函数会同时裁剪 param 与 loaded_weight，保证二者形状一致。
                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    self.data,
                    loaded_weight,
                    0,  # param_data_start：目标参数从 0 开始写
                    tp_rank * shard_size,  # loaded_weight 上本 rank 分片的起点
                    self.output_dim,
                    shard_size,
                )
                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            else:
                # GPU 路径：直接在输出维上切出本 rank 负责的连续区间
                loaded_weight = loaded_weight.narrow(
                    self.output_dim, tp_rank * shard_size, shard_size
                )

        # 带 dtype 校验的拷贝：禁止精度降级（如 fp32 -> fp8），
        # 除非显式开启 SGLANG_QUANT_ALLOW_DOWNCASTING。
        copy_with_check(self.data, loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        """
        加载「融合列并行层」中的**某一个子分片**（对应 MergedColumnParallelLinear）。

        典型场景：MLP 的 gate_proj 与 up_proj 在磁盘上是两个独立张量，
        但运行时被融合为一个 [2 * intermediate, hidden] 的参数。
        本方法每次只负责把其中一个（gate 或 up）写入参数的对应区间。

        与 load_column_parallel_weight 的区别在于「双重定位」：
          - 目标侧：用 shard_offset / shard_size 定位在**融合参数**中的写入位置；
          - 源侧：用 tp_rank * shard_size 定位在**该子权重完整张量**中的读取位置。

        kwargs:
          - shard_offset: 该子权重在融合参数输出维上的起始偏移（已按 TP 折算）。
          - shard_size:   该子权重在本 rank 上的分片长度。
          - tp_rank:      当前 TP rank。
          - use_presharded_weights: checkpoint 是否已预切分。
        """

        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        tp_rank = kwargs.get("tp_rank")
        use_presharded_weights = kwargs.get("use_presharded_weights")

        # 打包（packed）参数场景：如 int4/int8 权重会把多个元素压进一个存储单元。
        # 若被打包的维度恰好就是输出维，则外部传入的 offset/size 是「逻辑元素数」，
        # 需要按 packed_factor 折算成「实际存储下标」。
        if (
            isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
            and self.packed_dim == self.output_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = self.data

        # 目标侧：在融合参数中切出本子权重（且属于本 rank）的写入窗口
        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)

        from sglang.srt.model_loader.weight_utils import (
            narrow_padded_param_and_loaded_weight,
        )

        if _is_cpu:
            # CPU 后端：统一交由工具函数处理 pad/narrow，保证两侧形状对齐
            param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                param_data,
                loaded_weight,
                0,  # param_data_start：窗口内从 0 开始写
                tp_rank * shard_size,  # 源侧本 rank 分片起点
                self.output_dim,
                shard_size,
                not use_presharded_weights,  # 是否需要对 loaded_weight 做 narrow
            )
        else:
            if not use_presharded_weights:
                # 源侧：从该子权重的完整张量中取出本 rank 分片。
                # 特例：某些模型（如 qwen2_5_VL 的 mlp）中间维不是 8 的倍数，
                # 平均切分后最后一个 rank 的区间会越界，此时需要补零对齐。
                start_idx = tp_rank * shard_size
                end_idx = start_idx + shard_size
                if end_idx > loaded_weight.shape[self.output_dim]:
                    loaded_weight = pad_or_narrow_weight(
                        loaded_weight, self.output_dim, start_idx, shard_size
                    )
                else:
                    loaded_weight = loaded_weight.narrow(
                        self.output_dim, start_idx, shard_size
                    )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_qkv_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
        **kwargs,
    ):
        """
        加载融合 QKV 层中的 q / k / v **某一个分量**（对应 QKVParallelLinear）。

        与 load_merged_column_weight 的结构基本一致，唯一关键差异在**源侧分片下标**：
        Q 的头数通常远多于 KV（GQA / MQA）。当 tp_size > num_kv_heads 时，
        多个 TP rank 会**共享同一份 KV 头**（即 KV 被复制而非切分），
        因此 KV 的读取下标不能直接用 tp_rank，而要除以复制倍数。

        kwargs:
          - shard_offset: q/k/v 该分量在融合参数输出维上的起始偏移。
          - shard_size:   该分量在本 rank 上的分片长度。
          - shard_id:     "q" / "k" / "v"，标识当前加载的是哪个分量。
          - num_heads:    KV 头的复制倍数（由调用方按 tp_size / num_kv_heads 计算），
                          仅对 k/v 生效。
        """

        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        shard_id = kwargs.get("shard_id")
        num_heads = kwargs.get("num_heads")

        # 同 load_merged_column_weight：打包量化权重需把逻辑下标折算为存储下标
        if (
            isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
            and self.output_dim == self.packed_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = self.data

        # 关键：确定在源张量上读取第几个分片。
        #   - q：每个 rank 各持有不同的 Q 头 -> 直接用 tp_rank；
        #   - k/v：num_heads 个 rank 共享同一份 KV 分片 -> tp_rank // num_heads，
        #          从而实现 KV 头在这些 rank 上的复制。
        shard_id = tp_rank if shard_id == "q" else tp_rank // num_heads

        # 目标侧：在融合 QKV 参数中切出本分量的写入窗口
        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)

        if _is_cpu:
            from sglang.srt.model_loader.weight_utils import (
                narrow_padded_param_and_loaded_weight,
            )

            # CPU 后端：交由工具函数统一处理 pad/narrow
            param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                param_data,
                loaded_weight,
                0,  # param_data_start：窗口内从 0 开始写
                shard_id * shard_size,  # 源侧起点（已按 q / kv 规则换算）
                self.output_dim,
                shard_size,
                not use_presharded_weights,
            )
        else:
            if not use_presharded_weights:
                # 源侧：从该分量的完整张量中取出本 rank 应加载的区间
                loaded_weight = loaded_weight.narrow(
                    self.output_dim, shard_id * shard_size, shard_size
                )

        assert (
            param_data.shape == loaded_weight.shape
        ), f"{param_data.shape=}, {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)


class RowvLLMParameter(BasevLLMParameter):
    """为「行并行」线性层提供权重加载能力（load_row_parallel_weight）的参数类。

    适用于需要沿输入维（行方向）切分的参数，要求必须定义 input_dim。

    与 _ColumnvLLMParameter 构成对称：后者沿 output_dim 切分。
    ModelWeightParameter 同时继承两者，因此一个权重既能被列并行层
    也能被行并行层使用，具体走哪条切分路径由调用方（即所属线性层）决定。
    """

    def __init__(self, input_dim: int, **kwargs):
        # input_dim：该参数中「输入特征」对应的维度下标，
        # 也就是行并行下需要被切分的那一维。
        self._input_dim = input_dim
        super().__init__(**kwargs)

    @property
    def input_dim(self):
        return self._input_dim

    def load_row_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
    ):
        """把 checkpoint 中的权重沿输入维切出本 rank 的分片并写入本参数。

        行并行下每个 rank 只持有权重的一个行分块 A_i，
        因此需从全局权重中截取 [tp_rank * shard_size, (tp_rank+1) * shard_size)。

        参数:
            loaded_weight: 从 checkpoint 读出的权重；
                use_presharded_weights=False 时它是未切分的全局权重，
                为 True 时它已经就是本 rank 的分片。
            tp_rank: 本进程在张量并行组中的 rank，用于定位切片起点。
            use_presharded_weights: 权重已按 TP 预切分，跳过本函数内的切片逻辑。
        """
        if not use_presharded_weights:
            # 目标参数在输入维上的长度就是本 rank 应得的分片大小。
            shard_size = self.data.shape[self.input_dim]

            from sglang.srt.model_loader.weight_utils import (
                narrow_padded_param_and_loaded_weight,
            )

            if _is_cpu:
                # CPU 后端的参数可能带 padding，需要对「目标参数」与「源权重」
                # 同时做对齐后的 narrow，否则两者形状会错位。
                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    self.data,
                    loaded_weight,
                    0,  # param_data_start：目标参数中的写入起点
                    tp_rank * shard_size,
                    self.input_dim,
                    shard_size,
                )

                # 此分支已完成拷贝，直接返回，
                # 不再走下面针对 self.data 的通用写入路径。
                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)

                return
            else:
                # 补齐：处理尺寸未对齐的特例（如 qwen2_5_VL 的 mlp 非 8 对齐）。
                # 此时最后一个 rank 的切片末端会越过源权重边界，
                # 只能先 pad 再取，不能直接 narrow（否则会越界报错）。
                start_idx = tp_rank * shard_size
                end_idx = start_idx + shard_size
                if end_idx > loaded_weight.shape[self.input_dim]:
                    loaded_weight = pad_or_narrow_weight(
                        loaded_weight, self.input_dim, start_idx, shard_size
                    )
                else:
                    loaded_weight = loaded_weight.narrow(
                        self.input_dim, start_idx, shard_size
                    )

        # 特殊情况：从磁盘加载的 scale 常常是没有形状的 0 维张量
        #（例如 AutoFP8），统一 reshape 成 [1] 以便与参数形状对齐。
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        # 形状必须完全一致，否则说明 tp_rank/tp_size 或权重布局有误。
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)


class ModelWeightParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """
    Parameter class for linear layer weights. Uses both column and
    row parallelism.
    """

    pass


class GroupQuantScaleParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """
    Parameter class for weight scales loaded for weights with
    grouped quantization. Uses both column and row parallelism.
    """

    pass


class ChannelQuantScaleParameter(_ColumnvLLMParameter):
    """
    Parameter class for weight scales loaded for weights with
    channel-wise quantization. Equivalent to _ColumnvLLMParameter.
    """

    pass


class BlockQuantScaleParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """
    Parameter class for weight scales loaded for weights with
    block-wise quantization. Uses both column and row parallelism.
    """

    pass


class PerTensorScaleParameter(BasevLLMParameter):
    """
    Parameter class for scales where the number of scales is
    equivalent to the number of logical matrices in fused linear
    layers (e.g. for QKV, there are 3 scales loaded from disk).
    This is relevant to weights with per-tensor quantization.
    Adds functionality to map the scalers to a shard during
    weight loading.

    Note: additional parameter manipulation may be handled
    for each quantization config specifically, within
    process_weights_after_loading
    """

    def __init__(self, **kwargs):
        self.qkv_idxs = {"q": 0, "k": 1, "v": 2}
        super().__init__(**kwargs)

    def _shard_id_as_int(self, shard_id: Union[str, int]) -> int:
        if isinstance(shard_id, int):
            return shard_id

        # if not int, assume shard_id for qkv
        # map to int and return
        assert isinstance(shard_id, str)
        assert shard_id in self.qkv_idxs
        return self.qkv_idxs[shard_id]

    # 对于行并行层，无需切分，
    # 直接把权重原样加载进参数。
    # 原因：per-tensor scale 是整个张量共用的一个标量，
    # 不随输入/输出维切分，所以每个 rank 上的值都相同。
    # 因此丢弃 tp_rank / use_presharded_weights 两个切分相关参数（基类不接受它们），
    # 再转给基类的「整体拷贝」实现。
    def load_row_parallel_weight(self, *args, **kwargs):
        kwargs.pop("tp_rank", None)
        kwargs.pop("use_presharded_weights", None)
        super().load_row_parallel_weight(*args, **kwargs)

    def load_merged_column_weight(self, *args, **kwargs):
        self._load_into_shard_id(*args, **kwargs)

    def load_qkv_weight(self, *args, **kwargs):
        self._load_into_shard_id(*args, **kwargs)

    def load_column_parallel_weight(self, *args, **kwargs):
        kwargs.pop("tp_rank", None)
        kwargs.pop("use_presharded_weights", None)
        super().load_row_parallel_weight(*args, **kwargs)

    def _load_into_shard_id(
        self, loaded_weight: torch.Tensor, shard_id: Union[str, int], **kwargs
    ):
        """
        Slice the parameter data based on the shard id for
        loading.
        """

        param_data = self.data
        shard_id = self._shard_id_as_int(shard_id)

        # AutoFP8 scales do not have a shape
        # compressed-tensors scales do have a shape
        if len(loaded_weight.shape) != 0:
            assert loaded_weight.shape[0] == 1
            loaded_weight = loaded_weight[0]

        param_data = param_data[shard_id]
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class PackedColumnParameter(_ColumnvLLMParameter):
    """
    Parameter for model parameters which are packed on disk
    and support column parallelism only. See PackedvLLMParameter
    for more details on the packed properties.
    """

    def __init__(
        self,
        packed_factor: Union[int, Fraction],
        packed_dim: int,
        marlin_tile_size: Optional[int] = None,
        **kwargs,
    ):
        self._packed_factor = packed_factor
        self._packed_dim = packed_dim
        self._marlin_tile_size = marlin_tile_size
        super().__init__(**kwargs)

    @property
    def packed_dim(self):
        return self._packed_dim

    @property
    def packed_factor(self):
        return self._packed_factor

    @property
    def marlin_tile_size(self):
        return self._marlin_tile_size

    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        return _adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=self.packed_factor,
            marlin_tile_size=self.marlin_tile_size,
        )


class PackedvLLMParameter(ModelWeightParameter):
    """
    Parameter for model weights which are packed on disk.
    Example: GPTQ Marlin weights are int4 or int8, packed into int32.
    Extends the ModelWeightParameter to take in the
    packed factor, the packed dimension, and optionally, marlin
    tile size for marlin kernels. Adjusts the shard_size and
    shard_offset for fused linear layers model weight loading
    by accounting for packing and optionally, marlin tile size.
    """

    def __init__(
        self,
        packed_factor: Union[int, Fraction],
        packed_dim: int,
        marlin_tile_size: Optional[int] = None,
        **kwargs,
    ):
        self._packed_factor = packed_factor
        self._packed_dim = packed_dim
        self._marlin_tile_size = marlin_tile_size
        super().__init__(**kwargs)

    @property
    def packed_dim(self):
        return self._packed_dim

    @property
    def packed_factor(self):
        return self._packed_factor

    @property
    def marlin_tile_size(self):
        return self._marlin_tile_size

    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        return _adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=self.packed_factor,
            marlin_tile_size=self.marlin_tile_size,
        )


def permute_param_layout_(
    param: BasevLLMParameter, input_dim: int, output_dim: int, **kwargs
) -> BasevLLMParameter:
    """
    Permute a parameter's layout to the specified input and output dimensions,
    useful for forcing the parameter into a known layout, for example, if I need
    a packed (quantized) weight matrix to be in the layout
        {input_dim = 0, output_dim = 1, packed_dim = 0}
    then I can call:
        permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
    to ensure x is in the correct layout (permuting it to the correct layout if
    required, asserting if it cannot get it to the correct layout)
    """

    curr_input_dim = getattr(param, "input_dim", None)
    curr_output_dim = getattr(param, "output_dim", None)

    if curr_input_dim is None or curr_output_dim is None:
        assert param.data.dim() == 2, (
            "permute_param_layout_ only supports 2D parameters when either "
            "input_dim or output_dim is not set"
        )

    # if one of the dimensions is not set, set it to the opposite of the other
    #  we can only do this since we asserted the parameter is 2D above
    if curr_input_dim is None:
        assert curr_output_dim is not None, "either input or output dim must be set"
        curr_input_dim = (curr_output_dim + 1) % 2
    if curr_output_dim is None:
        assert curr_input_dim is not None, "either input or output dim must be set"
        curr_output_dim = (curr_input_dim + 1) % 2

    # create permutation from the current layout to the layout with
    # self.input_dim at input_dim and self.output_dim at output_dim preserving
    # other dimensions
    perm = [
        i for i in range(param.data.dim()) if i not in [curr_input_dim, curr_output_dim]
    ]
    perm.insert(input_dim, curr_input_dim)
    perm.insert(output_dim, curr_output_dim)

    if "packed_dim" in kwargs:
        assert (
            hasattr(param, "packed_dim")
            and param.packed_dim == perm[kwargs["packed_dim"]]
        ), "permute_param_layout_ currently doesn't support repacking"

    param.data = param.data.permute(*perm)
    if hasattr(param, "_input_dim"):
        param._input_dim = input_dim
    if hasattr(param, "_output_dim"):
        param._output_dim = output_dim
    if "packed_dim" in kwargs and hasattr(param, "_packed_dim"):
        param._packed_dim = kwargs["packed_dim"]

    return param


def _adjust_shard_indexes_for_marlin(shard_size, shard_offset, marlin_tile_size):
    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


def _adjust_shard_indexes_for_packing(
    shard_size, shard_offset, packed_factor, marlin_tile_size
):
    shard_size = shard_size // packed_factor
    shard_offset = shard_offset // packed_factor
    if marlin_tile_size is not None:
        return _adjust_shard_indexes_for_marlin(
            shard_size=shard_size,
            shard_offset=shard_offset,
            marlin_tile_size=marlin_tile_size,
        )
    return shard_size, shard_offset
