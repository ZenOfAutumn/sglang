# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/model_executor/layers/vocab_parallel_embedding.py

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch.nn.parameter import Parameter, UninitializedParameter

from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.amx_utils import PackWeightMethod
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import (
    attn_tp_all_reduce,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_allocation_symmetric,
    is_dp_attention_enabled,
)
from sglang.srt.layers.parameter import BasevLLMParameter
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_compiler_backend,
    is_cpu,
    is_npu,
    set_weight_attrs,
)
from sglang.srt.utils.async_probe import maybe_detect_oob

DEFAULT_VOCAB_PADDING_SIZE = 64

_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_npu = is_npu()

logger = logging.getLogger(__name__)


def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """把词表大小向上取整到 pad_to 的整数倍。

    为什么必须 padding：
      1. 词表维要被 tp_size 整除，才能均匀切分到各张卡；
      2. 对齐到 64（或更大）的倍数对 GEMM / Tensor Core 更友好；
      3. 真实词表大小往往是奇怪的数字（如 32000、151936、129280），
         直接切分会出现各 rank 分片不等长，后续 all-reduce/all-gather
         的形状对不上。

    padding 出来的那些行是「不对应任何真实 token」的空位，
    权重填 0，且在 forward 时通过 mask 保证永远不会被选中。
    """
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def vocab_range_from_per_partition_vocab_size(
    per_partition_vocab_size: int, rank: int, offset: int = 0
) -> Sequence[int]:
    """已知每个分片的大小，算出本 rank 负责的词表区间 [index_f, index_l)。

    Args:
        per_partition_vocab_size: 每张卡负责的词表行数。
        rank: 本进程在 TP 组内的编号。
        offset: 区间整体平移量。用于 LoRA 新增词表段——
            新增段在全局 token_id 空间里是接在原始词表之后的，
            所以要加上 org_vocab_size 作为起点偏移。
    """
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f + offset, index_l + offset


def vocab_range_from_global_vocab_size(
    global_vocab_size: int, rank: int, world_size: int, offset: int = 0
) -> Sequence[int]:
    """已知全局词表大小，均分后算出本 rank 负责的词表区间。

    注意这里用的是 divide（要求整除），所以传进来的 global_vocab_size
    必须已经 padding 过，否则会直接断言失败。
    """
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size, rank, offset=offset
    )


@dataclass
class VocabParallelEmbeddingShardIndices:
    """词表并行 embedding 中「本 rank 分片」的各种索引边界。

    这里同时维护了两套边界，理解它们的区别是读懂整个文件的关键：

      - ``padded_*``：**按 padding 后的词表均分**得到的边界。
        它决定了本 rank 的参数张量有多少行（各 rank 严格相等，
        因为要保证集合通信的形状一致）。
      - 无前缀的 ``org_*`` / ``added_*``：把 padding 裁掉后，
        本 rank 实际持有的**真实 token** 边界。
        它决定了权重加载时该从 checkpoint 里切哪一段、
        以及 forward 时哪些 token id 算「命中本 rank」。

    两者之差就是本 rank 的 padding 行数（``num_*_padding``），
    这些行权重为 0 且永远不会被 mask 放行。

    命名中的 ``org`` 指原始词表，``added`` 指 LoRA 新增词表。
    """

    # ---- padding 后均分得到的边界（决定参数张量形状）----
    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int

    # ---- 裁掉 padding 后的真实 token 边界（决定加载与 mask）----
    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int

    @property
    def num_org_elements(self) -> int:
        """本 rank 持有的原始词表**真实** token 数。"""
        return self.org_vocab_end_index - self.org_vocab_start_index

    @property
    def num_added_elements(self) -> int:
        """本 rank 持有的 LoRA 新增词表**真实** token 数。"""
        return self.added_vocab_end_index - self.added_vocab_start_index

    @property
    def num_org_elements_padded(self) -> int:
        """本 rank 原始词表段占用的行数（含 padding）。"""
        return self.padded_org_vocab_end_index - self.padded_org_vocab_start_index

    @property
    def num_added_elements_padded(self) -> int:
        """本 rank 新增词表段占用的行数（含 padding）。"""
        return self.padded_added_vocab_end_index - self.padded_added_vocab_start_index

    @property
    def num_org_vocab_padding(self) -> int:
        """原始词表段中的空洞行数。

        这个值在 forward 计算局部索引时至关重要：新增词表段紧跟在
        「原始段 + 其 padding」之后，所以把全局 token id 映射到
        本 rank 局部行号时必须把这段空洞算进偏移里。
        """
        return self.num_org_elements_padded - self.num_org_elements

    @property
    def num_added_vocab_padding(self) -> int:
        """新增词表段中的空洞行数。"""
        return self.num_added_elements_padded - self.num_added_elements

    @property
    def num_elements_padded(self) -> int:
        """本 rank 参数张量的总行数，必须等于 num_embeddings_per_partition。"""
        return self.num_org_elements_padded + self.num_added_elements_padded

    def __post_init__(self):
        # 一致性检查：区间必须非空且有序
        assert self.padded_org_vocab_start_index <= self.padded_org_vocab_end_index
        assert self.padded_added_vocab_start_index <= self.padded_added_vocab_end_index

        assert self.org_vocab_start_index <= self.org_vocab_end_index
        assert self.added_vocab_start_index <= self.added_vocab_end_index

        # 真实区间必须被包含在 padding 区间内（padding 只会向外扩，不会向内缩）
        assert self.org_vocab_start_index <= self.padded_org_vocab_start_index
        assert self.added_vocab_start_index <= self.padded_added_vocab_start_index
        assert self.org_vocab_end_index <= self.padded_org_vocab_end_index
        assert self.added_vocab_end_index <= self.padded_added_vocab_end_index

        # 真实元素数不可能超过带 padding 的容量
        assert self.num_org_elements <= self.num_org_elements_padded
        assert self.num_added_elements <= self.num_added_elements_padded


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def get_masked_input_and_mask(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把全局 token id 映射为本 rank 的局部行号，并返回「不属于本 rank」的掩码。

    这是词表并行的核心计算。因为每张卡只持有词表的一个切片，
    而输入的 token id 是全局的，所以必须：
      1. 判断每个 id 是否落在本 rank 负责的区间内；
      2. 命中的减掉起始偏移，变成本地行号；
      3. 未命中的必须置 0（而不能保留原值）——否则 embedding 查表会越界。
         置 0 后会取到本 rank 第 0 行的垃圾值，因此还需要把 mask 一并返回，
         由调用方在查表**之后**把这些位置的输出归零。

    步骤 3 这个「先算垃圾值、再事后归零」的做法看上去浪费，
    但它避开了数据相关的分支，使整个函数保持纯逐元素运算，
    从而能被 torch.compile 融合成单个 kernel。

    词表布局回顾（参见 VocabParallelEmbedding 类注释的图）：

        本 rank 张量:  [ 原始段真实 | 原始段 padding | 新增段真实 | 新增段 padding ]
        局部行号:      0 ................................................ N-1

    原始段的映射很直接：``local = global - org_vocab_start_index``。
    新增段需跳过前面「原始段真实行 + 原始段 padding 行」，所以偏移是：

        added_offset = added_vocab_start_index
                     - (org_vocab_end_index - org_vocab_start_index)  # 原始段真实行数
                     - num_org_vocab_padding                          # 原始段空洞行数

    Returns:
        (masked_input, out_of_range_mask)。前者是可直接用于查表的局部行号，
        后者为 True 表示该 token 不属于本 rank（需将对应输出置零）。
    """
    # torch.compile 会把下面所有逐元素算子融合成一个 kernel，因此非常快
    # 命中原始词表段（仅真实 token 区间，padding 区不算命中）
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    # 命中 LoRA 新增词表段
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index
    )
    # 新增段的偏移：除了减掉自身起点，还要把它在本地张量中靠后的
    # 那部分位置加回来（即减掉一个负的量），故这里是三项相减。
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    # 用乘法代替分支选择偏移：两个 mask 互斥，所以至多一项生效；
    # 都未命中时 valid_offset 为 0（但下一行会整体置 0，所以无影响）。
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = org_vocab_mask | added_vocab_mask
    # 乘上 vocab_mask：未命中的位置直接变 0，保证索引不越界。
    input_ = vocab_mask * (input_ - valid_offset)
    # 返回取反的 mask：True 表示「不属于本 rank」，供调用方 masked_fill_ 置零。
    return input_, ~vocab_mask


def get_embedding_tp_kwargs() -> dict:
    """返回*输入 embedding* 的词表并行布局参数，仅适用于支持 embedding 副本化
    的模型（DeepSeek-V2 target 系列：DeepSeek V3.1 / Kimi K2.5，
    以及它们对应的 EAGLE3 / NextN draft 模型）。

    为什么要把这个逻辑收拢到一个 helper：
    EAGLE / NextN 的 draft 模型会与 target **共享同一个**
    ``embed_tokens.weight`` 张量（通过 ``set_embed`` / ``set_embed_and_head``）。
    因此 target 和所有共享它的 draft **必须使用完全相同的词表并行布局**；
    否则 draft 侧的 mask / 索引计算会面对一个布局不同的张量，
    后果是 accept_len **静默下降**（不报错，只是推断质量变差）。
    全部走这一个 helper 就能从结构上消除两边配置漂移的可能。
    """
    if envs.SGLANG_ENABLE_EMBED_REPLICATION.get():
        # 副本化：每张卡都存完整词表。
        # 收益是省掉 embedding 后的 all-reduce，代价是权重重复存储。
        # 对超大词表（如 129280）而言这笔显存开销不小，所以默认关闭。
        return {"enable_tp": False}
    # 否则沿词表维切分。开启 DP attention 时每个 rank 只持有自己的局部 token，
    # 因此归约要在 attention-TP 子组内做，而不是在完整 TP 组内做。
    return {"enable_tp": True, "use_attn_tp_group": is_dp_attention_enabled()}


class VocabParallelEmbedding(torch.nn.Module):
    """沿**词表维**切分的 embedding 层。

    与 torch.nn.Embedding 的关系：行为上等价，但词表大小会被 padding
    到能被 TP 卡数整除的值。

    为什么沿词表维而不是 hidden 维切分：
    embedding 本质是一次「查表」（gather）。沿词表维切分后，
    每张卡只能查到自己负责的那些 token，其余位置输出 0，
    最后一次 all-reduce 就能把各卡的结果拼起来（因为互不重叠，
    求和等价于拼接）。若沿 hidden 维切，则每张卡都需存完整词表，
    而词表维（十万量级）恰好是这个张量最大的一维，节约不了显存。

    **布局约定**：为了兼容各种加载方式，LoRA 新增的 embedding
    总是放在 TP 分片张量的末尾。也就是说，基础词表与 LoRA 词表
    各自独立切分（且各自 padding），然后拼在同一个张量里。

    下面例子中：原始词表 = 1010，新增词表 = 16，padding 到 64 的倍数。
    因此 padding 后总词表大小为 1088（先把 1010 补到 1024，
    加上 16 变成 1040，再补到 1088）。张量布局如下：

    TP1，rank 0（不切分）：
                            |< --------BASE-------- >|< -BASE PADDING-- >|< -----LORA------ >|< -LORA PADDING-- >|
    对应的 token_id：     |  0  |  1  | ... | 1009 |  -1  | ... |  -1  | 1010 | ... | 1015 |  -1  | ... |  -1  |
                     行号： |  0  |  1  | ... | 1009 | 1010 | ... | 1023 | 1024 | ... | 1039 | 1040 | ... | 1087 |

    TP2，rank 0：
                            |< --------------------BASE--------------------- >|< -----LORA------ >|< -LORA PADDING- >|
    对应的 token_id：     |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 1000 | ... | 1015 |  -1  | ... |  -1 |
                     行号： |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 527  |  520 | ... | 543 |
    TP2，rank 1：
                            |< -----------BASE----------- >|< -BASE PADDING- >|< -----------LORA PADDING----------- >|
    对应的 token_id：     | 512 | 513 | 514 | ... | 1009 | -1  | ...  | -1  |  -1  | ... |  -1  | -1  | ... |   -1 |
                     行号： |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 519  | 520 | ... |  543 |

    看图时注意两个要点：
      - ``token_id = -1`` 的位置就是 padding 空洞，权重填 0，
        且永远不会被 ``get_masked_input_and_mask`` 放行。
      - 各 rank 的行数严格相等（例中都是 544 = 1088/2），
        这是集合通信形状一致的前提；但各 rank 的**真实** token 数可以不同
        （rank 1 的 LoRA 段全是 padding）。

    Args:
        num_embeddings: 词表大小（含 LoRA 新增部分）。
        embedding_dim: 隐藏层维度。
        params_dtype: 参数类型，默认取 torch 全局默认 dtype。
        org_num_embeddings: 原始词表大小（不含 LoRA）。
        padding_size: 词表的 padding 对齐粒度。
        quant_config: 本层的量化配置。
        prefix: 本层在 state dict 中的完整名称，用于量化时按名匹配规则。
        enable_tp: 是否启用词表并行。为 False 时退化为每张卡持有完整词表
            （副本化），可省掉 all-reduce。
        use_attn_tp_group: 在 attention-TP 子组内而非完整 TP 组内做切分与归约，
            用于 DP attention 场景。
        use_presharded_weights: checkpoint 中的权重已按 TP 切好，
            加载时不需再 narrow。
    """  # noqa: E501

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: Optional[torch.dtype] = None,
        org_num_embeddings: Optional[int] = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        enable_tp: bool = True,
        use_attn_tp_group: bool = False,
        use_presharded_weights: bool = False,
    ):
        super().__init__()
        self.quant_config = quant_config

        self.enable_tp = enable_tp
        self.use_attn_tp_group = use_attn_tp_group
        if self.enable_tp:
            if use_attn_tp_group:
                # DP attention 场景：只在 attention 的 TP 子组内切分。
                # 此时各 DP 副本各自持有完整词表的一份切分，
                # 归约也只在子组内进行。
                tp_rank = get_attention_tp_rank()
                self.tp_size = get_attention_tp_size()
            else:
                tp_rank = get_tensor_model_parallel_rank()
                self.tp_size = get_tensor_model_parallel_world_size()
        else:
            # 副本化模式：等价于 tp_size=1，各卡持有完整词表。
            # 此时不存在「子组」概念，所以 use_attn_tp_group 必须为 False。
            assert use_attn_tp_group is False
            tp_rank = 0
            self.tp_size = 1

        self.num_embeddings = num_embeddings
        self.org_vocab_size = org_num_embeddings or num_embeddings

        # 兼容词表大小无法被 TP 卡数整除的情形。
        # 把对齐粒度乘上 tp_size，可以保证 padding 后一定能被 tp_size 整除
        # （代价是多浪费一些空行）。
        # 注意这里限定了 _is_cpu：GPU 路径上依赖调用方保证可整除，
        # 不在此处隐式改大 padding，以免与已有的权重布局假设冲突。
        if (
            _is_cpu
            and pad_vocab_size(self.org_vocab_size, padding_size) % self.tp_size != 0
        ):
            padding_size *= self.tp_size
        self.padding_size = padding_size

        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.use_presharded_weights = use_presharded_weights
        if use_presharded_weights:
            # 预切分权重意味着 checkpoint 已经按某个固定布局切好，
            # 而 LoRA 新增词表会改变张量布局（末尾多出一段），
            # 两者无法共存。
            assert (
                num_added_embeddings == 0
            ), "Lora is not supported with presharded weights."

        # 两阶段 padding，顺序不可颠倒：
        #   1. 先把原始词表补齐（保证原始段自身对齐）；
        #   2. 再把「对齐后的原始段 + 新增段」整体补齐。
        # 这正是类注释中 1010 -> 1024 -> 1040 -> 1088 的来源，
        # 也是新增段能稳定落在末尾的前提。
        self.org_vocab_size_padded = pad_vocab_size(
            self.org_vocab_size, self.padding_size
        )
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings, self.padding_size
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded

        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            tp_rank,
            self.tp_size,
        )
        self.embedding_dim = embedding_dim

        quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if quant_method is None:
            quant_method = UnquantizedEmbeddingMethod()

        # 如果本层确实是 embedding 层，那么量化方法必须实现 embedding 操作；
        # 如果是其他类型（比如子类 ParallelLMHead，它只用权重做 matmul、
        # 从不调 embedding），则不要求这一点。
        #
        # 注意：这里写的是 type(self.__class__)，结果是元类（通常为 type），
        # 永远不等于 VocabParallelEmbedding，因此 is_embedding_layer 恒为 False，
        # 下面的校验实际上从不生效。这是从 vLLM 原封不动继承过来的已知缺陷，
        # 此处保留原样以便与上游保持一致（正确写法应为 type(self)）。
        is_embedding_layer = type(self.__class__) is VocabParallelEmbedding
        quant_method_implements_embedding = method_has_implemented_embedding(
            type(quant_method)
        )
        if is_embedding_layer and not quant_method_implements_embedding:
            raise NotImplementedError(
                f"The class {type(quant_method).__name__} must implement "
                "the 'embedding' method, see UnquantizedEmbeddingMethod."
            )

        self.quant_method: QuantizeMethodBase = quant_method

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        # 沿词表维切分权重矩阵。
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        # 本 rank 参数张量的行数（含 padding，各 rank 相等）。
        self.num_embeddings_per_partition = divide(
            self.num_embeddings_padded, self.tp_size
        )
        # 与 _get_indices 算出的布局互相校验：两条独立路径得出的总行数必须一致。
        assert (
            self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        )
        # 下面两个是本 rank 的**真实** token 数（不含 padding），
        # 各 rank 可以不相等；LoRA 等上层逻辑会读取它们。
        self.num_org_embeddings_per_partition = (
            self.shard_indices.org_vocab_end_index
            - self.shard_indices.org_vocab_start_index
        )
        self.num_added_embeddings_per_partition = (
            self.shard_indices.added_vocab_end_index
            - self.shard_indices.added_vocab_start_index
        )

        # 由量化方法负责建参数（不量化时就是一个普通的 Parameter）。
        # 把 weight_loader 一并传进去，使权重加载时能找到本层的切分逻辑。
        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

    @classmethod
    def _get_indices(
        cls,
        vocab_size_padded: int,
        org_vocab_size_padded: int,
        vocab_size: int,
        org_vocab_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> VocabParallelEmbeddingShardIndices:
        """根据给定的 tp_rank / tp_size，算出本分片的起止索引，
        布局遵循类注释中的图。

        这里的核心套路是：**先按 padding 后的尺寸均分，再用 min() 裁回真实边界**。
        均分保证了各 rank 张量形状一致；裁回则得到实际持有的真实 token 区间。
        当某个 rank 的均分区间完全落在 padding 区时，
        min() 会使 start == end，得到一个空区间（该 rank 没有真实 token）。
        这正是类注释中 TP2/rank 1 的 LoRA 段全为 padding 的情形。
        """
        num_added_embeddings_padded = vocab_size_padded - org_vocab_size_padded
        # 原始词表段：从 0 开始均分，无需偏移。
        padded_org_vocab_start_index, padded_org_vocab_end_index = (
            vocab_range_from_global_vocab_size(org_vocab_size_padded, tp_rank, tp_size)
        )
        # 新增词表段：在全局 token_id 空间里接在原始词表之后，
        # 所以要以 org_vocab_size（注意是**未** padding 的真实大小）作为偏移。
        padded_added_vocab_start_index, padded_added_vocab_end_index = (
            vocab_range_from_global_vocab_size(
                num_added_embeddings_padded, tp_rank, tp_size, offset=org_vocab_size
            )
        )
        # 裁掉 padding，得到真实 token 边界。
        # 用 min 而不是 clamp：起点也可能超过真实词表大小，
        # 此时 start == end == 词表大小，区间为空。
        org_vocab_start_index = min(padded_org_vocab_start_index, org_vocab_size)
        org_vocab_end_index = min(padded_org_vocab_end_index, org_vocab_size)
        added_vocab_start_index = min(padded_added_vocab_start_index, vocab_size)
        added_vocab_end_index = min(padded_added_vocab_end_index, vocab_size)
        return VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index,
            padded_org_vocab_end_index,
            padded_added_vocab_start_index,
            padded_added_vocab_end_index,
            org_vocab_start_index,
            org_vocab_end_index,
            added_vocab_start_index,
            added_vocab_end_index,
        )

    def get_sharded_to_full_mapping(self) -> Optional[List[int]]:
        """返回一个索引映射，用于重排 all-gather 后的 logits，供采样使用。

        采样时会从所有 rank 收集 logits。收集后的张量布局仍然遵循
        类注释中的格式——也就是说它是「按 rank 拼接」的，
        每个 rank 内部又是「原始段 | 原始 padding | 新增段 | 新增 padding」，
        因此 index 与 token_id **并不相等**。例如 TP2 时，
        拼接后的 index 512 对应的却是 token_id 1000（LoRA 段）。

        而采样需要 index 与 token_id 严格一一对应（index 就是 token_id），
        否则 argmax / top-k 算出的下标无法直接当作 token_id 使用。
        本方法返回的索引就是用来做这个重排的。

        返回列表的组织方式：``所有 rank 的基础词表 + 所有 rank 的新增词表 + 全部 padding``。
        这样重排后，前 ``org_vocab_size`` 个位置恰好是 token_id 0..N，
        接着是 LoRA token，padding 全部被赶到末尾（采样时可直接忽略）。

        Returns:
            tp_size < 2 时返回 None（无需重排，因为单卡布局下 index 已经就是 token_id）。
        """
        if self.tp_size < 2:
            return None

        # 分三个桶收集「拼接后张量中的下标」，最后按顺序连接。
        base_embeddings: List[int] = []
        added_embeddings: List[int] = []
        padding: List[int] = []
        for tp_rank in range(self.tp_size):
            shard_indices = self._get_indices(
                self.num_embeddings_padded,
                self.org_vocab_size_padded,
                self.num_embeddings,
                self.org_vocab_size,
                tp_rank,
                self.tp_size,
            )
            # 本 rank 在拼接后张量中的位置区间。
            range_start = self.num_embeddings_per_partition * tp_rank
            range_end = self.num_embeddings_per_partition * (tp_rank + 1)
            # 按「原始真实 -> 原始 padding -> 新增真实 -> 新增 padding」的
            # 顺序逐段扫过本 rank，把下标分发到对应的桶里。
            base_embeddings.extend(
                range(range_start, range_start + shard_indices.num_org_elements)
            )
            padding.extend(
                range(
                    range_start + shard_indices.num_org_elements,
                    range_start + shard_indices.num_org_elements_padded,
                )
            )
            added_embeddings.extend(
                range(
                    range_start + shard_indices.num_org_elements_padded,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                )
            )
            padding.extend(
                range(
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements_padded,
                )
            )
            # 四段加起来必须恰好填满本 rank 的区间，不多不少。
            assert (
                range_start
                + shard_indices.num_org_elements_padded
                + shard_indices.num_added_elements_padded
                == range_end
            )
        # 拼接顺序决定了重排后的语义：真实 token 在前（且 id 连续），padding 在后。
        ret = base_embeddings + added_embeddings + padding
        assert len(ret) == self.num_embeddings_padded
        return ret

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        """从 checkpoint 加载权重，并取出属于本 rank 的词表分片。

        注意这里只处理**原始词表**的切分（``org_vocab_*``）：
        LoRA 新增词表不从 base checkpoint 里来，而是后续由 LoRA 加载器
        写入张量末尾的那一段。因此本函数把本 rank 真实 token 以外的
        所有尾部行（padding + 预留的 LoRA 位）一律填 0。

        Args:
            param: 本层的参数张量（可能带有 output_dim / packed_dim 等属性）。
            loaded_weight: 从 checkpoint 读出的完整（或已预切分的）权重。
        """
        # output_dim：该参数沿哪一维做 TP 切分（embedding 权重为 0，即词表维）。
        # packed_dim：量化时多个低位宽数值被打包进一个整数的那一维。
        output_dim = getattr(param, "output_dim", None)
        packed_dim = getattr(param, "packed_dim", None)

        # GGUF 权重类型标记：这不是真正的权重张量，而是一个标量（量化类型枚举），
        # 直接拷贝并记录类型即可，不适用任何切分逻辑。
        if getattr(param, "is_gguf_weight_type", None):
            param.data.copy_(loaded_weight)
            param.weight_type = loaded_weight.item()
            return
        elif isinstance(param, UninitializedParameter):
            # 延迟初始化的参数（常见于 GGUF）：此时才知道真实形状。
            # 注意要把词表维除以 tp_size，因为本 rank 只存一份分片。
            shape = list(loaded_weight.shape)
            if output_dim is not None:
                shape[output_dim] = shape[output_dim] // self.tp_size
            param.materialize(tuple(shape), dtype=loaded_weight.dtype)

        # 参数没有 output_dim，说明它不需切分，应该完整复制到每张卡上
        # （例如 act_order GPTQ 的 g_idx，它是一个全局置换表）。
        if output_dim is None:
            assert param.data.shape == loaded_weight.shape
            param.data.copy_(loaded_weight)
            return

        # 本 rank 该从全局权重里取哪一段。
        # 用的是裁掉 padding 的**真实**边界，因为 checkpoint 里只有真实 token。
        start_idx = self.shard_indices.org_vocab_start_index
        shard_size = self.shard_indices.org_vocab_end_index - start_idx

        # 如果打包维恰好就是我们要切分的那一维，
        # 则偏移量需除以 packed_factor（因为一个存储单元装了 packed_factor 个值）。
        if packed_dim is not None and packed_dim == output_dim:
            packed_factor = (
                param.packed_factor
                if isinstance(param, BasevLLMParameter)
                else param.packed_factor
            )
            assert loaded_weight.shape[output_dim] == (
                self.org_vocab_size // param.packed_factor
            )
            start_idx = start_idx // packed_factor
            shard_size = shard_size // packed_factor
        else:
            # 校验 checkpoint 权重的词表维长度是否符合预期：
            #   - 普通情况下应为完整的 org_vocab_size；
            #   - 预切分权重下已经除过 tp_size。
            assert loaded_weight.shape[output_dim] == (
                self.org_vocab_size
                // (self.tp_size if self.use_presharded_weights else 1)
            ), f"{self.org_vocab_size=} {self.use_presharded_weights=} {loaded_weight.shape[output_dim]=}"

        # 拷贝数据。
        # 预切分权重已经是本 rank 的那一段，再 narrow 会重复切一次，所以要跳过。
        if not self.use_presharded_weights:
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        param[: loaded_weight.shape[0]].data.copy_(loaded_weight)
        # 尾部剩余行全部置 0。这些行包括 padding 空洞以及预留的 LoRA 位。
        # 必须显式清零而不能依赖未初始化内存：虽然 forward 会用 mask 屏蔽它们，
        # 但 tp_size == 1 时不走 mask 分支，残留的 NaN/Inf 会污染输出。
        param[loaded_weight.shape[0] :].data.fill_(0)

    def forward(self, input_):
        """查表得到 token embedding。

        TP > 1 时的三步式：
          1. mask + 重映射：把全局 token id 变为局部行号，非本 rank 的置 0；
          2. 查表：非本 rank 的位置会取到第 0 行的无意义值；
          3. 把那些位置归零，再 all-reduce——各 rank 负责的 token 互不重叠，
             所以求和的效果等价于「每个位置取唯一拥有者的结果」。

        第 3 步的归零不可省：若不归零，垃圾值会在 all-reduce 中被累加进结果。
        """
        # 把非法 token id（>= vocab_size，或负数 / 未被 mask 的哨兵值）
        # 以一个定位明确的异步断言暴露出来，
        # 而不是静默地做一次越界的 embedding 查表（tp=1 时不走 mask，无保护）。
        # 异步断言不会引入 GPU-CPU 同步，错误在下一个同步点才浮现。
        maybe_detect_oob(
            input_, 0, self.num_embeddings, "VocabParallelEmbedding input id"
        )
        if self.tp_size > 1:
            # 构造掩码，并把全局 id 重映射为本 rank 局部行号。
            masked_input, input_mask = get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            # 单卡（或副本化）：持有完整词表，id 无需任何平移。
            masked_input = input_

        # 查表取 embedding。
        # 包在 use_symmetric_memory 里：对称内存允许后续的 NCCL 集合通信
        # 直接在这块显存上做，避开一次拷贝；不满足条件时自动禁用。
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            output_parallel = self.quant_method.embedding(self, masked_input.long())

        if self.tp_size > 1:
            # 把不属于本 rank 的位置置零（它们刚才取到的是第 0 行的垃圾值）。
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            # input_scattered 模式下，下游会自行做 all-gather 把分散的激活汇总，
            # 此处再归约一次就重复了，所以跳过。
            if not get_attn_tp_context().input_scattered:
                if self.use_attn_tp_group:
                    # DP attention：仅在 attention 的 TP 子组内归约。
                    output_parallel = attn_tp_all_reduce(output_parallel)
                else:
                    # 在全部模型并行 GPU 上做归约。
                    output_parallel = tensor_model_parallel_all_reduce(output_parallel)
        return output_parallel

    def extra_repr(self) -> str:
        # 注意这里打印的 num_embeddings 是**本 rank 分片**的行数（含 padding），
        # 而不是全局词表大小，看 print(model) 输出时容易误读。
        s = f"num_embeddings={self.num_embeddings_per_partition}"
        s += f", embedding_dim={self.embedding_dim}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", num_embeddings_padded={self.num_embeddings_padded}"
        if self.enable_tp:
            s += f", tp_size={self.tp_size}"
        return s


class ParallelLMHead(VocabParallelEmbedding):
    """并行化的语言模型输出头（LM head）。

    存放用于算 logits 的权重矩阵（供 Sampler 使用）。
    权重与 bias 都会被 padding，以保证能被模型并行 GPU 数整除。

    **为何继承 VocabParallelEmbedding**：两者权重形状完全相同
    （``[vocab_size, hidden]``）、切分方式也相同（都沿词表维），
    因此可以直接复用父类的 padding 计算、分片索引与权重加载逻辑。
    这也是 weight tying（输入 embedding 与输出头共享权重）能成立的前提。

    **但计算方向相反**，这是两者本质区别：

    | | VocabParallelEmbedding | ParallelLMHead |
    | --- | --- | --- |
    | 运算 | 查表 gather（id -> 向量） | 矩阵乘（向量 -> 全词表得分） |
    | 输出切分情况 | 完整的 hidden，需 all-reduce | 部分词表的 logits，需 all-gather |
    | 本类是否实现 forward | 是 | **不**实现（见下） |

    因为输出需要 all-gather 而不是 all-reduce，且往往要与采样、
    logits 后处理（温度、惩罚项等）融合在一起，
    本类故意不实现 forward，而是只当作一个「权重容器」，
    由 LogitsProcessor / Sampler 取走 ``self.weight`` 自行计算。

    Args:
        num_embeddings: 词表大小。
        embedding_dim: 隐藏层维度。
        bias: 是否使用 bias。
        params_dtype: 参数类型。
        org_num_embeddings: 原始词表大小（不含 LoRA）。
        padding_size: 词表的 padding 对齐粒度。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        org_num_embeddings: Optional[int] = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_attn_tp_group: bool = False,
        use_presharded_weights: bool = False,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            org_num_embeddings=org_num_embeddings,
            padding_size=padding_size,
            quant_config=quant_config,
            prefix=prefix,
            use_attn_tp_group=use_attn_tp_group,
            use_presharded_weights=use_presharded_weights,
        )
        self.quant_config = quant_config

        # 仅在未量化时才支持对 LMHead 做权重打包。
        # CPU + AMX 后端下把权重预先重排为 AMX 堆叠布局，可显著提升 GEMM 吞吐；
        # 但重排后的内存布局与量化 kernel 预期的不一致，所以二者不能共存。
        if _is_cpu and _is_cpu_amx_available:
            if hasattr(self, "weight") and self.weight.dtype in [
                torch.bfloat16,
                torch.float16,
            ]:
                self.quant_method = PackWeightMethod(weight_names=["weight"])

        if bias:
            # bias 沿词表维切分，每张卡只持有自己那一段（长度与权重行数一致）。
            self.bias = Parameter(
                torch.empty(self.num_embeddings_per_partition, dtype=params_dtype)
            )
            # 标上 output_dim=0 并复用父类的 weight_loader，
            # 使 bias 能与权重走完全相同的词表切分逻辑。
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            # 显式注册为 None，使 state_dict / 属性访问行为一致。
            self.register_parameter("bias", None)

    def tie_weights(self, embed_tokens: VocabParallelEmbedding):
        """与词向量（输入 embedding）共享权重。

        weight tying 是很多模型（如 Llama-3.2 小型号、Gemma）的标配：
        输入 embedding 与输出头形状相同，共享后可省下一份巨大的词表权重。

        返回值的两种情形必须区分开：
          - 一般情况：把 ``embed_tokens.weight`` 挂到自己身上，返回 ``self``；
          - GGUF：直接返回 ``embed_tokens`` 本身。因为 GGUF 的量化权重
            伴随一整套额外张量（scale、量化类型等），
            单拿一个 ``weight`` 过来是不完整的，必须整个模块一起用。
        """
        # GGUF 量化的 embed_tokens。
        if self.quant_config and self.quant_config.get_name() == "gguf":
            return embed_tokens
        else:
            self.weight = embed_tokens.weight
            return self

    def forward(self, input_):
        """本类不参与前向计算，调用即报错。

        LM head 的权重应由 LogitsProcessor / Sampler 取用，
        因为算 logits 需要与采样、词表并行的 all-gather、
        以及各种 logits 后处理协同，不能在本层内部完成。
        这里主动抛异常而不是默默算一个 matmul，
        是为了把「误把 LMHead 当普通层调用」这类错误在第一时间暴露。
        """
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")
