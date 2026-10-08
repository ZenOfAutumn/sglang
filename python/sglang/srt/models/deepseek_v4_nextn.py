"""DeepSeek-V4 的 NextN（MTP 草稿）模型。

本文件实现 DeepSeek-V4 的 Multi-Token Prediction 草稿模块，用于投机解码
（--speculative-algorithm EAGLE 时作为 draft model 加载，权重来自主
checkpoint 的 nextn 层，无需单独的 draft checkpoint）。

与 DeepSeek-V3 的 NextN（deepseek_nextn.py）结构对比：

  V3:  concat[enorm(embed) ; hnorm(hidden)] --eh_proj(2d->d)--> 1 层 decoder
  V4:  e_proj(enorm(embed)) + h_proj(hnorm(hidden_i))，i = 1..hc_mult
       即目标模型给每个 token 传来 hc_mult 份 hidden state（多分支），
       embedding 投影后广播加到每个分支上，decoder 以 [n, hc_mult, d]
       形状前向，最后由 hc_head 做可学习的门控加权求和，把 hc_mult 个
       分支收敛回一个 [n, d] 的 hidden state。

术语对照（与 docs/learn/speculative/mtp.md 一致）：
  - hidden state 通路（hnorm/h_proj）= 上下文通道 ①（主模型前缀摘要）；
  - decoder 层自己的 attention 回看历史 KV = 上下文通道 ②。
"""

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.attention.dsa.utils import (
    can_dsa_cp_split,
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
    is_dsa_prefill_cp_round_robin_split,
)
from sglang.srt.layers.cp.utils import (
    cp_round_robin_input_ids_v2,
    is_cp_v2_active,
)
from sglang.srt.layers.dp_attention import (
    dp_gather_replicate,
    get_global_dp_buffer_len,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_round_robin_input_ids,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer, DeepseekV4ForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

# NextN 层不做 KV 压缩（压缩比 0 = 关闭）：草稿层只有 1 层，压缩收益小，
# 且需要与目标模型 verify 时的 KV 布局对齐。
COMPRESS_RATIO_NEXTN_LAYER = 0


class DeepseekV4ModelNextN(nn.Module):
    """V4 NextN 主体：embedding 融合 + 1 层 decoder + hc_head 分支聚合。

    输入输出概览（记 n = 本次前向的 token 数，d = hidden_size，
    m = hc_mult，以 DeepSeek-V4 典型配置 d=7168、m=2 为例）：

      input_ids                          [n]          草稿输入 token
      spec_info.hidden_states            [n, m*d]     目标模型传来的 m 份 hidden
      --> forward 返回
      hidden_states（聚合+norm 后）       [n, d]       喂给共享 lm_head 出草稿分布
      pre_hc_head（聚合前的展平特征）      [n, m*d]     供 logits_processor 捕获，
                                                      作为下一步草稿的输入 hidden
    """
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # 词嵌入表。权重与主模型共享（load_weights 时从主 checkpoint 加载），
        # 开启 DP attention 时不做 TP 切分（各 DP rank 持有完整词表）。
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        # 两个独立的 RMSNorm，分别归一化 embedding 通路与 hidden state 通路，
        # 消除两个表征空间的量级差（作用同 V3 的 enorm/hnorm，见 mtp.md 1.2 节）。
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rms_norm_eps = config.rms_norm_eps

        # hc = hidden combination：V4 NextN 的多分支 hidden 聚合机制。
        # hc_mult(m)：目标模型为每个 token 提供的 hidden state 分支数；
        # hc_dim = m*d：m 个分支展平后的总维度。
        # hc_head_* 是 hc_head() 门控聚合的参数（详见 hc_head docstring）：
        #   hc_head_fn    [m, m*d]  由全部分支特征算出 m 个门控 logit 的线性映射
        #   hc_head_base  [m]       每个分支门控的偏置
        #   hc_head_scale [1]       门控 logit 的全局温度缩放
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        # 融合投影：V3 用 concat[e;h] + eh_proj(2d->d)，V4 改为两路各自过
        # d->d 投影后相加（e_proj(enorm(embed)) + h_proj(hnorm(hidden_i))），
        # 加法形式便于 embedding 一次投影后广播到全部 m 个 hidden 分支。
        self.e_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("e_proj", prefix),
        )
        self.h_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("h_proj", prefix),
        )
        if isinstance(quant_config, ModelSlimConfig):
            prefix = "mtp.0"
        else:
            prefix = add_prefix("decoder", prefix)

        # NextN 只有 1 层完整的 V4 decoder layer（attention + MoE/FFN）。
        # 该层自己的 attention 会读写草稿侧 KV cache——这是上下文通道 ②。
        self.decoder = DeepseekV4DecoderLayer(
            config,
            layer_id=0,
            quant_config=quant_config,
            is_nextn=True,
            prefix=prefix,
            alt_streams=None,
            compress_ratio_override=COMPRESS_RATIO_NEXTN_LAYER,
        )

        # DSA prefill 阶段的 context parallel（CP）开关与 CP 组大小，
        # 用于 forward 中对长 prefill 做序列维切分。
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_size = get_parallel().attn_cp_size
        else:
            self.cp_size = None

        # 输出 norm。挂在 shared_head 命名空间下是为了对齐 checkpoint 里
        # "model.shared_head.norm" 的权重名（lm_head 本体与主模型共享，
        # 在 DeepseekV4ForCausalLMNextN 中单独构造）。
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        """把 m 个 hidden 分支做可学习门控加权求和，收敛为单个 hidden。

        计算：gate_i = sigmoid(scale * <fn_i, RMS 归一化后的展平特征>) + base_i
             再加 hc_eps 下限；y = sum_i gate_i * x_i。
        注意门控是 sigmoid 而非 softmax——各分支权重独立、不互斥，
        且 +hc_eps 保证每个分支至少有微小贡献（梯度不会完全断流）。

        入参：
          x         [n, m, d]  decoder 输出的 m 个分支，n=token 数
          hc_fn     [m, m*d]   门控线性映射（float32）
          hc_scale  [1]        门控温度
          hc_base   [m]        门控偏置
        出参：
          y         [n, d]     聚合后的 hidden，dtype 还原为 x 的 dtype

        数值示例（toy：n=1, m=2, d=2，eps 取 0 便于口算）：
          x = [[[1, 0],           # 分支 0
                [0, 2]]]          # 分支 1
          flatten -> [[1, 0, 0, 2]]，RMS = sqrt((1+0+0+4)/4) ≈ 1.118，
          rsqrt ≈ 0.894
          设 hc_fn = [[1,0,0,0],[0,0,0,1]]，hc_scale=1，hc_base=[0,0]：
          mixes = [1, 2] * 0.894 = [0.894, 1.789]
          gate  = sigmoid([0.894, 1.789]) = [0.710, 0.857]
          y = 0.710*[1,0] + 0.857*[0,2] = [0.710, 1.714]   # [n=1, d=2]
        """
        shape, dtype = x.size(), x.dtype
        # [n, m, d] -> [n, m*d]，转 float32 保证门控数值稳定
        x = x.flatten(1).float()
        # 对展平后的 m*d 维特征做 RMS 归一（只算系数，不改 x 本身）
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.rms_norm_eps)
        # [n, m*d] @ [m*d, m]^T -> [n, m]：每个分支一个门控 logit
        mixes = F.linear(x, hc_fn) * rsqrt
        # sigmoid 门控 + eps 下限 -> [n, m]
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        # [n, m, 1] * [n, m, d] 按分支加权后沿 m 求和 -> [n, d]
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        cp_v2_active = is_cp_v2_active(forward_batch)
        use_prefill_cp = dsa_use_prefill_cp(forward_batch)
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)  # [n] -> [n, d]
        else:
            hidden_states = input_embeds

        if hidden_states.shape[0] > 0:
            n_tokens = hidden_states.shape[0]
            d = self.config.hidden_size
            # [n, m*d] -> [n*m, d]：把 m 个分支摊平成独立行，
            # 便于 RMSNorm / h_proj 按 d 维统一处理。
            # 例：n=2, m=2, d=7168 时 [2, 14336] -> [4, 7168]
            hc_flat = forward_batch.spec_info.hidden_states.view(
                n_tokens * self.hc_mult, d
            )
            # hidden 通路（上下文通道 ①）：hnorm 归一 + h_proj 投影
            h_proj_out, _ = self.h_proj(self.hnorm(hc_flat))
            h_proj_hidden_states = h_proj_out.view(n_tokens, self.hc_mult, d)

            # embedding 通路：enorm 归一 + e_proj 投影，[n, d]
            e_proj_hidden_states, _ = self.e_proj(self.enorm(hidden_states))
            # 广播相加：[n, 1, d] + [n, m, d] -> [n, m, d]
            # （embedding 只有一份，加到全部 m 个 hidden 分支上）
            hidden_states = e_proj_hidden_states[:, None, :] + h_proj_hidden_states
        else:
            # 空 batch（如 DP attention 下本 rank 无请求）：只补齐形状
            # [0, d] -> [0, m, d]，保证后续算子/通信的维度一致。
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        if get_parallel().attn_dp_size > 1 and get_moe_a2a_backend().is_none():
            input_ids_global = torch.empty(
                (get_global_dp_buffer_len(), 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            # Token IDs are replicated within an attention-TP group. Use replicate
            # gather to avoid summing duplicated IDs when attention_tp_size > 1.
            # Clone because the MAX_LEN gather may zero its local input in place.
            dp_gather_replicate(
                input_ids_global, input_ids[:, None].clone(), forward_batch
            )
            input_ids_global = input_ids_global.squeeze(-1)
        else:
            input_ids_global = input_ids

        if use_prefill_cp:
            if cp_v2_active:
                input_ids = cp_round_robin_input_ids_v2(input_ids, forward_batch)
            else:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
                positions = cp_split_and_rebuild_position(forward_batch, positions)
                input_ids = cp_round_robin_input_ids(input_ids)
            input_ids_global = input_ids

        # 唯一一层 decoder：attention 读写草稿 KV（上下文通道 ②）+ MoE/FFN。
        # 入出形状均为 [n, m, d]。
        hidden_states, residual, post, comb = self.decoder(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
        )
        if residual is not None:
            # NextN 只有一层 decoder，不存在"下一层"来消费延迟融合的
            # hc_post 状态，所以必须在这里就地完成 hc_post 融合。
            hidden_states = self.decoder.hc_post(hidden_states, residual, post, comb)

        if use_prefill_cp and not cp_v2_active:
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )

        # 聚合前先留一份展平特征 [n, m, d] -> [n, m*d]：
        # 它会经 logits_processor 捕获，成为下一步草稿的输入 hidden
        #（对应 mtp.md 中"自递归时 g 顶替 h"的 g）。
        pre_hc_head = hidden_states.flatten(1)

        # 门控聚合 m 个分支 [n, m, d] -> [n, d]，再过输出 norm，
        # 结果交给共享 lm_head 产生草稿分布。
        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        hidden_states = self.shared_head.norm(hidden_states)

        return hidden_states, pre_hc_head


class DeepseekV4ForCausalLMNextN(DeepseekV4ForCausalLM):
    """V4 NextN 的对外入口：NextN 主体 + 共享 lm_head + logits 处理。

    继承 DeepseekV4ForCausalLM 只为复用其权重加载/融合逻辑
    （load_weights(is_nextn=True) 只抽取 nextn 层权重），
    __init__ 里刻意用 nn.Module.__init__ 跳过父类构造（不建 L 层主干）。
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_parallel().tp_size
        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_rank = get_parallel().attn_cp_rank
            self.cp_size = get_parallel().attn_cp_size
        else:
            self.cp_rank = None
            self.cp_size = None

        self.model = DeepseekV4ModelNextN(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        # lm_head 与主模型共享权重（prefix 指向 checkpoint 中
        # model.shared_head.head），草稿分布与目标分布用同一张输出投影表。
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.dsa_enable_prefill_cp and not is_cp_v2_active(forward_batch):
            if can_dsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )
                if is_dsa_prefill_cp_round_robin_split():
                    attn_backend = get_attn_backend()
                    metadata = attn_backend.forward_metadata
                    core_meta = metadata.core_attn_metadata
                    core_meta.apply_cp_reindex()
                    core_meta.init_flashmla_related(is_prefill=True)
                    if metadata.indexer_metadata is not None:
                        metadata.indexer_metadata = (
                            attn_backend.init_forward_metadata_indexer(core_meta)
                        )

        hidden_states, pre_hc_head = self.model(input_ids, positions, forward_batch)
        # hidden_states [n, d] 过 lm_head 出草稿 logits；
        # pre_hc_head [n, m*d] 作为 hidden_states_before_norm 传入，
        # capture 模式下它才是被记录的 hidden（下一步草稿的输入）。
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            hidden_states_before_norm=pre_hc_head,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # 复用 V4 主模型的加载逻辑，is_nextn=True 时只挑 nextn 层权重。
        super().load_weights(weights, is_nextn=True)

    def post_load_weights(self, is_nextn=False, weight_names=None):
        super().post_load_weights(is_nextn=True, weight_names=weight_names)


EntryClass = [DeepseekV4ForCausalLMNextN]
