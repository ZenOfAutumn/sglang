import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.utils.hash import murmur_hash32
from sglang.srt.layers.utils.logprob import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.async_probe import sanitize_nan_logits
from sglang.srt.utils.common import (
    get_bool_env_var,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
)

if is_cuda():
    from flashinfer.sampling import (
        min_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs,
    )
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
    )

if is_musa():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
if _use_aiter:
    from aiter import greedy_sample as _aiter_greedy_sample

# The aiter greedy_sample kernel can return an out-of-range token id (== vocab_size,
# e.g. 151666 for MiniCPM-V) for all-NaN / all -inf logit rows on ROCm, which decodes
# to an empty string and breaks downstream consumers. Set this to 1 to fall back to
# torch.argmax (which always returns a valid index). Default off so behavior is
# unchanged elsewhere.
_disable_aiter_greedy_sample = get_bool_env_var("SGLANG_DISABLE_AITER_GREEDY_SAMPLE")

if is_npu():
    import torch_npu

logger = logging.getLogger(__name__)

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")
_CUSTOM_SAMPLER_FACTORIES: Dict[str, Callable[[], "Sampler"]] = {}
_BUILT_IN_SAMPLING_BACKENDS = {"flashinfer", "pytorch", "ascend"}


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.tp_sync_group = get_tp_group().device_group
        if is_dp_attention_enabled():
            self.tp_sync_group = get_attention_tp_group().device_group

        self.rl_on_policy_target = get_global_server_args().rl_on_policy_target
        # In RL on-policy mode, deterministic inference is automatically enabled.
        self.enable_deterministic = (
            get_global_server_args().enable_deterministic_inference
        )
        # In RL on-policy mode, we use log_softmax to compute logprobs to match the trainer.
        self.use_log_softmax_logprob = self.rl_on_policy_target is not None
        self.use_ascend_backend = get_global_server_args().sampling_backend == "ascend"

    def _preprocess_logits(
        self, logits: torch.Tensor, sampling_info: SamplingBatchInfo
    ) -> torch.Tensor:
        """Apply custom logit processors and sanitize non-finite logits."""
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(logits, sampling_info)
        sanitize_nan_logits(logits, "sampler: next_token_logits")
        return logits

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            token_ids_logprobs: Per-sequence list of specific token IDs to retrieve
                logprobs for. Each element is a list of token IDs (or None) for one
                sequence in the batch. This is used in speculative decoding.
            positions: The positions of the tokens in the sequence. Used for deterministic sampling
                to get the unique seed for each position.

        中译：执行真实采样：对模型前向产出的 logits 做处理并采样出每条序列的下一个 token id，
              （按需）计算 logprob 并就地写回 logits_output。整体分为「贪心」与「随机采样」两条
              主路径，随机路径下还区分 Ascend 后端、RL on-policy、标准 softmax 采样三种情况。

        参数：
            logits_output：模型前向输出的 logits 容器（其 next_token_logits 为输入）。
            sampling_info：采样元信息（温度、top-p/top-k/min-p 开关、自定义处理器等）。
            return_logprob：为 True 时把 logprob 信息写回 logits_output。
            top_logprobs_nums：每条序列要返回的 top-k logprob 数量。
            token_ids_logprobs：每条序列额外指定要取 logprob 的 token id 列表（投机解码用）。
            positions：各 token 在序列中的位置，用于确定性采样时为每个位置生成唯一随机种子。
        """
        logits = logits_output.next_token_logits

        # Preprocess logits (custom processors and NaN handling)
        # 中译：采样前预处理——应用自定义 logit processor，并把非有限值（NaN/Inf）清洗掉。
        logits = self._preprocess_logits(logits, sampling_info)

        if sampling_info.is_all_greedy:
            # 中译：贪心路径——整批都不需要随机性，直接取概率最大的 token（argmax）。
            if _use_aiter and not _disable_aiter_greedy_sample:
                # 中译：aiter greedy kernel —— AMD/ROCm 平台上 aiter 库提供的「贪心采样」融合算子。
                #   它做的事等价于 torch.argmax(logits, -1)：对每条序列在词表维度取 logit 最大的
                #   token id。但实现为单个手写 ROCm kernel，把「写入预分配输出张量 + 求 argmax」
                #   融合在一起，省去了 torch.argmax 的额外开销与中间张量，在 AMD GPU 上更快。
                #
                #   启用条件（见文件顶部）：
                #     _use_aiter = 环境变量 SGLANG_USE_AITER 打开且当前是 HIP/ROCm 平台；
                #     且未通过 SGLANG_DISABLE_AITER_GREEDY_SAMPLE 显式禁用。
                #
                #   调用约定：结果写入预先分配好的 int32 输出张量（按惯例 kernel 不返回值，
                #   而是原地写 batch_next_token_ids），长度为 batch（logits.shape[0]）。
                #
                #   已知坑（见 _disable_aiter_greedy_sample 处注释）：当某行 logits 全为 NaN /
                #   全为 -inf 时，该 kernel 可能返回越界 token id（== vocab_size），解码成空串并
                #   破坏下游。遇到此情况可设 SGLANG_DISABLE_AITER_GREEDY_SAMPLE=1 回退到
                #   torch.argmax（总是返回合法下标）。
                batch_next_token_ids = torch.empty(
                    logits.shape[0], device=logits.device, dtype=torch.int32
                )
                _aiter_greedy_sample(batch_next_token_ids, logits)
            else:
                # 中译：通用回退路径——用 torch.argmax 取每条序列 logit 最大的 token id。
                batch_next_token_ids = torch.argmax(logits, -1)
            if return_logprob:
                # 中译：贪心也可按需返回 logprob，对原始 logits 直接做 log_softmax。
                original_logprobs = logprobs = torch.nn.functional.log_softmax(
                    logits, dim=-1
                )
        else:
            # 中译：随机采样路径。simple_sampling_case 表示不需要 top-p/top-k/min-p 的「简单情形」，
            #       可走更快的采样实现。
            simple_sampling_case = (
                not sampling_info.need_top_p_sampling
                and not sampling_info.need_top_k_sampling
                and not sampling_info.need_min_p_sampling
            )

            # If requested, cache original logprobs before temperature scaling.
            # 中译：若要求返回「应用温度前」的原始 logprob，则在温度缩放前先缓存一份。
            if return_logprob and SGLANG_RETURN_ORIGINAL_LOGPROB:
                original_logprobs = torch.log_softmax(logits, dim=-1)

            # In RL on-policy mode, we use log_softmax to compute logprobs to match the trainer.
            # 中译：RL on-policy 模式下，用 log_softmax 计算 logprob 以与训练器口径对齐。
            logprobs_via_logsoftmax_kernel = None
            if self.rl_on_policy_target is not None:
                # TODO: use more inplace ops to save memory
                logits_div_temperature = (
                    logits.bfloat16().div(sampling_info.temperatures).bfloat16()
                )
                logprobs_via_logsoftmax_kernel = torch.log_softmax(
                    logits_div_temperature, dim=-1
                )
                del logits_div_temperature

            if self.use_ascend_backend:
                # Ascend backend: sample from logits directly.
                # 中译：Ascend 后端——直接从 logits 采样（其 kernel 内部完成温度/概率处理）。
                batch_next_token_ids, logprobs = self._forward_ascend_backend(
                    logits,
                    sampling_info,
                    simple_sampling_case,
                    return_logprob,
                    positions,
                )
            elif (
                self.use_log_softmax_logprob
                and self.enable_deterministic
                and simple_sampling_case
            ):
                # RL on-policy path: sample from logprobs to match the trainer.
                # 中译：RL on-policy 路径——直接从 logprob 分布采样，使结果与训练器一致。
                batch_next_token_ids = self._sample_from_logprobs(
                    logprobs_via_logsoftmax_kernel,
                    sampling_info,
                    positions,
                )
                if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:
                    logprobs = logprobs_via_logsoftmax_kernel
            else:
                # Standard path: do softmax and sample from probs.
                # 中译：标准路径——温度缩放后做 softmax，再从概率分布中采样。
                logits.div_(sampling_info.temperatures)  # 除以温度（原地操作）

                # In-place op to save memory
                # 中译：原地把 logits 变成概率分布（softmax），复用同一块显存以省内存。
                logits[:] = torch.softmax(logits, dim=-1)
                probs = logits

                # 中译：从概率分布采样（内部按需应用 top-p/top-k/min-p，调对应后端 kernel）。
                batch_next_token_ids = self._sample_from_probs(
                    probs, sampling_info, positions, simple_sampling_case
                )
                if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:
                    # 中译：需要 logprob 时，优先用 RL 路径算好的，否则对概率取对数。
                    logprobs = (
                        logprobs_via_logsoftmax_kernel
                        if logprobs_via_logsoftmax_kernel is not None
                        else torch.log(probs)
                    )
                del probs

        # Attach logprobs to logits_output (in-place modification)
        # 中译：把计算好的 logprob（含 top-k、指定 token id 的 logprob）就地写回 logits_output。
        if return_logprob:
            if SGLANG_RETURN_ORIGINAL_LOGPROB:
                # 中译：若要求返回温度缩放前的原始 logprob，则改用之前缓存的那份。
                logprobs = original_logprobs
            self._attach_logprobs_to_output(
                logits_output,
                logprobs,
                top_logprobs_nums,
                token_ids_logprobs,
                sampling_info,
                batch_next_token_ids,
            )

        # 中译：在张量并行（TP）各 rank 间同步采样得到的 token id，保证各 rank 结果一致。
        self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)

        return batch_next_token_ids

    def _sample_from_probs(
        self,
        probs: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        positions: torch.Tensor,
        simple_sampling_case: bool,
    ) -> torch.Tensor:
        """Sample from probability distribution (after softmax).

        Used for standard sampling with flashinfer/pytorch backends.
        Handles both simple (direct multinomial) and complex (top-k/top-p/min-p) cases.

        中译：从（softmax 之后的）概率分布中采样下一个 token。

        用于 flashinfer / pytorch 后端的标准采样路径。会同时处理两种情形：
        - 简单情形（simple）：无 top-k/top-p/min-p 截断，直接做多项式（multinomial）采样；
        - 复杂情形（complex）：需要先做 top-k/top-p/min-p 截断与重归一化再采样。

        参数：
            probs: 形状 (batch, vocab) 的概率张量（已过 softmax，每行和为 1）。
            sampling_info: 批次采样参数（top_ks/top_ps/min_ps、是否需要 min_p、确定性采样种子等）。
            positions: 各 token 在序列中的位置，用于确定性采样按位置生成唯一随机源。
            simple_sampling_case: 是否为简单情形（无任何截断），决定走快速直采还是带截断的路径。
        返回：
            batch_next_token_ids: 形状 (batch,) 的下一个 token id。
        """
        if simple_sampling_case:
            # 中译：简单情形——无截断，直接对概率分布做多项式采样。
            # 当 sampling_seed 不为 None 时，sampling_from_probs_torch 内部会改走
            # Gumbel-Max 的确定性采样（与 batch 组合/调度顺序无关，可复现）。
            batch_next_token_ids = sampling_from_probs_torch(
                probs,
                sampling_seed=sampling_info.sampling_seed,
                positions=positions,
            )
        else:
            # 中译：复杂情形——按全局配置选择采样后端实现。
            backend = get_global_server_args().sampling_backend
            if backend == "flashinfer":
                # 中译：flashinfer 后端是高性能 CUDA kernel，但其内部 RNG 不接受
                # 外部传入的逐请求种子，因此无法支持确定性采样，这里断言种子为空。
                assert (
                    sampling_info.sampling_seed is None
                ), "Sampling seed is not supported for flashinfer backend"
                if sampling_info.need_min_p_sampling:
                    # 中译：min-p 采样路径——先按 top-k、再按 top-p 对概率做截断并重归一化，
                    # 最后基于 min-p 阈值采样。
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                    batch_next_token_ids = min_p_sampling_from_probs(
                        probs, sampling_info.min_ps
                    )
                else:
                    # 中译：top-k + top-p 联合采样（filter_apply_order="joint" 表示
                    # 两个过滤条件联合应用，而非依次串行）。
                    batch_next_token_ids = top_k_top_p_sampling_from_probs(
                        probs.contiguous(),
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        filter_apply_order="joint",
                    )
            elif backend == "pytorch":
                # A slower fallback implementation with torch native operations.
                # 中译：使用 torch 原生算子的较慢回退实现；它支持确定性采样
                # （接收 sampling_seed 与 positions），并在一函数内统一处理
                # top-k / top-p / min-p 三种截断。
                batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                    probs,
                    sampling_info.top_ks,
                    sampling_info.top_ps,
                    sampling_info.min_ps,
                    sampling_info.need_min_p_sampling,
                    sampling_info.sampling_seed,
                    positions,
                )
            else:
                raise ValueError(f"Invalid sampling backend: {backend}")
        return batch_next_token_ids

    def _sample_from_logprobs(
        self,
        logprobs: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from log-probabilities using the Gumbel trick.

        Used for deterministic sampling with simple cases (no top-k/top-p/min-p).
        Requires sampling_seed to be set in sampling_info.

        中译：用 Gumbel-Max 技巧从对数概率（log-probabilities）中采样。

        用于「简单情形（无 top-k/top-p/min-p 截断）下的确定性采样」。原理：给每个
        类别的对数概率加上由种子确定性生成的 Gumbel 噪声后取 argmax，在分布上等价于
        按 softmax 概率抽样，但全程可复现（详见术语表「随机种子 / Gumbel-Max 技巧」）。
        必须在 sampling_info 中设置 sampling_seed，否则无法生成确定性噪声。

        参数：
            logprobs: 形状 (batch, vocab) 的对数概率张量。
            sampling_info: 批次采样参数，此处必须包含非空的 sampling_seed。
            positions: 各 token 在序列中的位置，与种子一起决定每个位置唯一且可复现的噪声。
        返回：
            形状 (batch,)、dtype 为 int32 的下一个 token id。
        """
        # 中译：本路径专为确定性采样设计，缺少种子则无意义，直接断言失败。
        assert (
            sampling_info.sampling_seed is not None
        ), "sampling_seed is required for sampling from logprobs"
        # 中译：multinomial_with_seed 内部用 murmur_hash32(seed, positions, 列下标)
        # 生成 Gumbel 噪声并对 (logprobs + 噪声) 取 argmax，返回形状 (batch, 1) 的列索引。
        sampled_index = multinomial_with_seed(
            logprobs, sampling_info.sampling_seed, positions
        )
        # 中译：展平为一维并转 int32，作为 token id 返回。
        return sampled_index.view(-1).to(torch.int32)

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        simple_sampling_case: bool,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from temperature-scaled logits without softmax.

        Used for the Ascend NPU backend which handles softmax internally.
        """
        if simple_sampling_case:
            probs = torch.softmax(logits, dim=-1)
            if sampling_info.sampling_seed is not None:
                probabilities = probs.to(torch.float64).log_()
                batch_next_token_ids = multinomial_with_seed(
                    probabilities, sampling_info.sampling_seed, positions
                ).view(-1)
            else:
                batch_next_token_ids = torch.multinomial(probs, num_samples=1).view(-1)
            return batch_next_token_ids.to(torch.int32)
        else:
            assert (
                self.use_ascend_backend
            ), "Only ascend backend supports sampling from logits"
            batch_next_token_ids = top_k_top_p_min_p_sampling_from_logits_ascend(
                logits,
                sampling_info.top_ks,
                sampling_info.top_ps,
                sampling_info.min_ps,
                sampling_info.need_min_p_sampling,
                sampling_info.sampling_seed,
                positions,
            )
            return batch_next_token_ids.to(torch.int32)

    def _forward_ascend_backend(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        simple_sampling_case: bool,
        return_logprob: bool,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Handle the full Ascend backend sampling path.

        Ascend backend has fused kernels that handle softmax internally,
        so we sample directly from temperature-scaled logits.

        Returns:
            A tuple of (batch_next_token_ids, logprobs). logprobs is None
            when return_logprob is False or SGLANG_RETURN_ORIGINAL_LOGPROB is set.
        """
        logits.div_(sampling_info.temperatures)
        batch_next_token_ids = self._sample_from_logits(
            logits, sampling_info, simple_sampling_case, positions
        )
        logprobs = None
        if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:
            logprobs = torch.log_softmax(logits, dim=-1)
        return batch_next_token_ids, logprobs

    def _attach_logprobs_to_output(
        self,
        logits_output: LogitsProcessorOutput,
        logprobs: torch.Tensor,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        sampling_info: SamplingBatchInfo,
        batch_next_token_ids: torch.Tensor,
    ):
        # clamp to avoid -inf values
        logprobs.clamp_(min=torch.finfo(logprobs.dtype).min)

        # Attach logprobs to logits_output (in-place modification)
        if any(x > 0 for x in top_logprobs_nums):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums, no_copy_to_cpu=True)

        if any(x is not None for x in token_ids_logprobs):
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(
                logprobs, token_ids_logprobs, no_copy_to_cpu=True
            )

        logits_output.next_token_logprobs = logprobs[
            torch.arange(len(batch_next_token_ids), device=sampling_info.device),
            batch_next_token_ids,
        ]

    def _sync_token_ids_across_tp(
        self, batch_next_token_ids: torch.Tensor, sampling_info: SamplingBatchInfo
    ):
        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
    ) -> None:
        """
        Compute logprobs for requested token IDs without performing sampling.

        Optimized for prefill-only scoring requests that need token probabilities
        but don't require next token generation.
        """

        if logits_output.next_token_logits is None:
            logger.warning("No logits available for logprob computation")
            return

        # Check if any requests actually need logprobs computation
        needs_token_ids_logprobs = any(
            token_ids is not None and len(token_ids) > 0
            for token_ids in token_ids_logprobs
        )
        needs_top_logprobs = any(x > 0 for x in top_logprobs_nums)

        if not (needs_token_ids_logprobs or needs_top_logprobs):
            return

        # Preprocess logits (custom processors and NaN handling)
        logits = self._preprocess_logits(logits_output.next_token_logits, sampling_info)

        # Compute logprobs
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Handle top logprobs if requested
        if needs_top_logprobs:
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums, no_copy_to_cpu=True)

        # Handle token_ids logprobs if requested
        if needs_token_ids_logprobs:
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs_batch_optimized(logprobs, token_ids_logprobs)


def register_sampler_backend(backend: str, factory: Callable[[], "Sampler"]) -> None:
    """Register a custom sampler factory for a backend string."""

    if not backend:
        raise ValueError("backend must be a non-empty string")

    from sglang.srt.server_args import SAMPLING_BACKEND_CHOICES

    if backend in _CUSTOM_SAMPLER_FACTORIES:
        logger.warning("Overriding existing sampler factory for backend '%s'", backend)
    SAMPLING_BACKEND_CHOICES.add(backend)
    _CUSTOM_SAMPLER_FACTORIES[backend] = factory


def create_sampler(backend: Optional[str] = None) -> "Sampler":
    """Create a sampler honoring custom backend registrations."""

    server_args = get_global_server_args()
    backend = backend or (server_args.sampling_backend if server_args else None)

    if backend in _CUSTOM_SAMPLER_FACTORIES:
        sampler = _CUSTOM_SAMPLER_FACTORIES[backend]()
        if not isinstance(sampler, Sampler):
            raise TypeError(
                f"Custom sampler factory for backend '{backend}' must return a Sampler"
            )
        return sampler

    if backend is None or backend in _BUILT_IN_SAMPLING_BACKENDS:
        return Sampler()

    raise ValueError(
        f"Unknown sampling backend '{backend}'. Register it via register_sampler_backend()."
    )


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
    sampling_seed: Optional[torch.Tensor],
    positions: torch.Tensor,
):
    """
    A top-k, top-p and min-p sampling implementation with native pytorch operations.
    When sampling_seed is not None, deterministic inference will be enabled, it will sample
    with the sampling_seed of each request.

    中译：用 torch 原生算子实现的 top-k / top-p / min-p「带截断」采样（即「复杂情形」）。
    当 sampling_seed 不为 None 时启用确定性推理，按每个请求各自的种子采样（结果可复现）。

    整体流程：先把概率降序排序，依次施加 top-k、top-p（以及可选 min-p）三种截断把不合格
    候选的概率清零，再在剩余候选上采样，最后用排序索引映射回原始 token id。

    参数：
        probs: 形状 (batch, vocab) 的概率张量（已过 softmax）。
        top_ks: 每个请求的 top-k 值（只保留概率最高的 k 个候选）。
        top_ps: 每个请求的 top-p 阈值（保留累积概率达 p 的最小候选集，即 nucleus 采样）。
        min_ps: 每个请求的 min-p 系数（阈值 = 该行最大概率 × min_p，过滤掉相对过小的候选）。
        need_min_p_sampling: 是否启用 min-p 截断。
        sampling_seed: 逐请求确定性采样种子；为 None 时走非确定性 multinomial。
        positions: 各 token 的位置，确定性采样时与种子一起生成唯一可复现的随机源。
    返回：
        batch_next_token_ids: 形状 (batch,) 的下一个 token id。
    """
    # 中译：按概率降序排序；probs_sort 为排序后的概率，probs_idx 记录其对应的原始列下标。
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    # 中译：沿排序维做累积和，用于后续 top-p 的累积概率判断。
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # 中译：top-k 截断——排序后下标 >= top_k 的位置（即第 k 名之后的候选）概率清零。
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    # 中译：top-p 截断——(累积和 - 当前项) 即「该候选之前的累积概率」，若已超过 top_p
    # 说明 nucleus 已集满，把该候选及其后的概率清零。用减去自身的写法可保证 nucleus
    # 至少包含第 1 个候选（避免阈值过小时全被清零）。
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        # TODO: probs_sort should be re-normalized for the use of multinomial_with_seed
        # 中译：min-p 与确定性采样暂不兼容——multinomial_with_seed 要求传入归一化概率，
        # 而此处 probs_sort 经多重截断后未重归一化，直接用会得到错误结果，故断言种子为空。
        assert (
            sampling_seed is None
        ), "With sampling seed, multinomial_with_seed will provide wrong results"
        # 中译：min-p 截断——阈值 = 当前行最大概率(已排序首位) × min_p，低于阈值的候选清零，
        # 即只保留与「最可能 token」相对接近的候选。
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    if sampling_seed is None:
        # 中译：非确定性路径——直接在截断后的（未归一化）权重上做多项式采样。
        # torch.multinomial 接受非归一化权重，故无需重归一化。
        sampled_index = torch.multinomial(probs_sort, num_samples=1)
    else:
        # NOTE: when using top-k/top-p/min-p sampling, we need to modify probs before we
        # apply log to get logprobs. Therefore, we cannot use log_softmax directly.
        # For now, we use log to the modified probs to get logprobs, but for numerical
        # stability, we'd better come up with a solution to use log_softmax.
        # 中译：确定性路径——由于要在「截断后的概率」上采样，不能直接用 log_softmax，
        # 只能对修改后的 probs 取 log 得到 logprobs（被清零项 log 后为 -inf，argmax 时
        # 自然不会被选中）。这里用 float64 提升数值稳定性；TODO 中提到更稳的做法待优化。
        logprobs = probs_sort.to(torch.float64)  # Using float64 for numerical stability
        del probs_sort
        logprobs.log_()
        # 中译：用基于种子的 Gumbel-Max 采样，保证确定可复现。
        sampled_index = multinomial_with_seed(logprobs, sampling_seed, positions)

    # int32 range is enough to represent the token ids
    # 中译：int32 足以表示 token id 范围，转 int32 省显存。
    probs_idx = probs_idx.to(torch.int32)
    # 中译：sampled_index 是「排序后」的下标，需经 probs_idx 映射回原始词表 token id。
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def top_k_top_p_min_p_sampling_from_logits_ascend(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
    sampling_seed: Optional[torch.Tensor],
    positions: torch.Tensor,
):
    """A top-k, top-p and min-p sampling implementation for ascend npu with torch_npu interface.

    Takes temperature-scaled logits as input (softmax is applied internally).
    """
    # torch_npu.npu_top_k_top_p requires top_k value range in [1, 1024]
    if hasattr(torch_npu, "npu_top_k_top_p") and torch.all(
        (top_ks <= 1024) & (top_ks >= 1)
    ):
        logits_top_k_top_p = torch_npu.npu_top_k_top_p(logits, top_ps, top_ks)
        probs_top_k_top_p = logits_top_k_top_p.softmax(dim=-1)

        if need_min_p_sampling:
            min_p_thresholds = probs_top_k_top_p.max(dim=-1) * min_ps
            min_p_mask = probs_top_k_top_p < min_p_thresholds.view(-1, 1)
            probs_top_k_top_p.masked_fill_(min_p_mask, 0.0)

        if sampling_seed is None:
            batch_next_token_ids = torch.multinomial(probs_top_k_top_p, num_samples=1)
        else:
            logprobs_top_k_top_p = probs_top_k_top_p.to(
                torch.float64
            )  # Using float64 for numerical stability
            del probs_top_k_top_p
            logprobs_top_k_top_p.log_()
            batch_next_token_ids = multinomial_with_seed(
                logprobs_top_k_top_p, sampling_seed, positions
            )
    else:
        probs = torch.softmax(logits, dim=-1)
        probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

        # when top_k is -1 (in which sglang turns it to TOP_K_ALL), make it explicitly equal to logit's size
        topk_all_mask = top_ks == TOP_K_ALL
        top_ks.masked_fill_(topk_all_mask, probs.shape[1])
        top_k_mask = torch.arange(0, probs.shape[-1], device=probs.device).view(
            1, -1
        ) >= top_ks.view(-1, 1)
        probs_sort.masked_fill_(top_k_mask, 0.0)

        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = probs_sum - probs_sort > top_ps.view(-1, 1)
        probs_sort.masked_fill_(top_p_mask, 0.0)

        if need_min_p_sampling:
            min_p_thresholds = probs_sort[:, 0] * min_ps
            min_p_mask = probs_sort < min_p_thresholds.view(-1, 1)
            probs_sort.masked_fill_(min_p_mask, 0.0)

        if sampling_seed is None:
            sampled_index = torch.multinomial(probs_sort, num_samples=1)
        else:
            logprobs = probs_sort.to(
                torch.float64
            )  # Using float64 for numerical stability
            del probs_sort
            logprobs.log_()
            sampled_index = multinomial_with_seed(logprobs, sampling_seed, positions)
        probs_idx = probs_idx.to(torch.int32)
        batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index)

    return batch_next_token_ids.view(-1)


@torch.compile(dynamic=True, disable=is_npu())
def multinomial_with_seed(
    logprobs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """
    Samples n elements from an input tensor `inputs` of shape (n, m) using
    a unique random seed for each row. This is a deterministic batched alternative to
    `torch.multinomial`.

    Args:
        inputs: A float tensor of shape (n, m) representing n categorical
                distributions with m categories each. The values are treated
                as weights and do not need to sum to 1.
        seed:   An integer tensor of shape (n,) containing the random seed
                for each corresponding row in `inputs`.
        positions: The positions of the tokens in the sequence. Used for deterministic sampling
                to get the unique seed for each position.

    Returns:
        A tensor of shape (n,) where the i-th element is an index sampled
        from the distribution in `inputs[i]` using `seed[i]`.

    中译：基于 Gumbel-Max 技巧的「确定性、可批处理」采样，是 torch.multinomial 的
    确定性替代实现。对形状 (n, m) 的输入（n 个分布、每个 m 个类别），用「逐请求种子 +
    token 位置」哈希出每个 (行, 列) 的随机源，从而保证：相同的 seed 与 position 必得
    完全相同的采样结果——与该请求和哪些请求拼成 batch、batch 如何调度无关，因此可复现。
    数学原理（加 Gumbel 噪声取 argmax 等价于按 softmax 概率抽样）见术语表
    「随机种子 / Gumbel-Max 技巧」条目。

    注意：参数虽名为 logprobs，但按 Gumbel-Max 的要求其语义应为「对数概率/对数权重」；
    docstring 中的 `inputs` 即指该入参。各行不要求归一化（差一个常数偏移不影响 argmax）。

    参数：
        logprobs: 形状 (n, m) 的对数概率/对数权重张量（n 个分布，每个 m 个类别）。
        seed:     形状 (n,) 的整数种子张量，逐行（逐请求）一个种子。
        positions: 各 token 在序列中的位置，与 seed 一起保证每个位置的随机源唯一且可复现。
    返回：
        形状 (n, 1) 的索引张量，第 i 行是用 seed[i] 从第 i 个分布中采到的类别下标。
    """
    n, m = logprobs.shape
    # 中译：种子转 uint64，作为哈希输入；避免有符号溢出并匹配哈希函数的入参类型。
    seed = seed.to(torch.uint64)
    # 中译：列下标 [0, 1, ..., m-1]，代表词表中的每个类别；与 seed、position 一起哈希，
    # 使「同一请求、同一位置」下每个候选 token 都拿到各自确定的随机值。
    col_indices = torch.arange(m, device=logprobs.device)
    # 中译：murmur_hash32 对 (seed, positions, col_indices) 三元组做哈希并广播成
    # (n, m) 的 uint32 张量——这是全部随机性的确定性来源（无全局 RNG 状态）。
    hashed = murmur_hash32(seed, positions, col_indices)

    # NOTE (sehoon): it is critical to keep gumbel noise calculation in float64 to avoid numerical instability.
    # keeping logprobs in float64 is less critical, but we found it's still safer to keep it in float64.
    # 中译：Gumbel 噪声计算必须在 float64 下进行以避免数值不稳定；logprobs 也一并用
    # float64（重要性稍低但更安全）。先把 uint32 哈希值归一化到 [0, 1] 的均匀分布。
    x = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max

    # x is a uniform sample in [0, 1]. get gumbel noise from it.
    # which is equivalent to -log(-log(x))
    # keep everything in in-place operations to avoid unnecessary memory allocations.
    # 中译：x 为 [0,1] 均匀样本，经 -log(-log(x)) 变换即得标准 Gumbel 噪声。
    # 全程使用原地（in-place）操作，避免额外的显存分配。
    x.log_().clamp_(min=torch.finfo(x.dtype).min).neg_()  # -log(x)
    # 中译：clamp 下界用于防止 x 极小时 log 结果为 -inf 导致溢出（数值保护）。
    x.log_().neg_()  # -log(-log(x)) == gumbel noise

    # add gumbel noise to logprobs
    # 中译：把 Gumbel 噪声加到对数概率上——这正是 Gumbel-Max 技巧的关键一步。
    x.add_(logprobs.to(torch.float64))

    # 中译：对每行取 argmax 即完成一次「按 softmax 概率抽样」（数学上等价、且确定可复现）。
    return torch.argmax(x, dim=1, keepdim=True)


def sampling_from_probs_torch(
    probs: torch.Tensor,
    sampling_seed: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
):
    """A sampling implementation with native pytorch operations, without
    top-k, top-p, or min-p filtering.

    Note: For deterministic sampling from logprobs, use Sampler._sample_from_logprobs instead.

    中译：用 torch 原生算子实现的采样，不做 top-k / top-p / min-p 任何截断（即「简单情形」）。

    根据是否传入 sampling_seed，分两条路径：
    - 未传种子：用 torch.multinomial 直接按概率分布随机抽样，随机性来自全局 RNG 状态。
    - 传入种子：走确定性采样——把概率转成对数概率后用 Gumbel-Max 技巧采样，结果可复现
      且与 batch 组合/调度顺序无关（详见术语表「随机种子 / Gumbel-Max 技巧」）。

    注意：若已持有对数概率（logprobs），应直接用 Sampler._sample_from_logprobs，
    避免这里额外的 torch.log(probs) 转换及由此引入的精度损失。

    参数：
        probs: 形状 (batch, vocab) 的概率张量（已过 softmax，每行和为 1）。
        sampling_seed: 逐请求确定性采样种子张量；为 None 时走非确定性的 multinomial。
        positions: 各 token 在序列中的位置，与种子一起为每个位置生成唯一且可复现的噪声。
    返回：
        batch_next_token_ids: 形状 (batch,)、dtype 为 int32 的下一个 token id。
    """
    if sampling_seed is None:
        # 中译：非确定性路径——标准多项式采样，每行抽取 1 个样本。
        sampled_index = torch.multinomial(probs, num_samples=1)
    else:
        # Deterministic sampling: convert probs to logprobs and use gumbel trick
        # 中译：确定性路径——先 log 把概率转成对数概率，再用基于种子的 Gumbel-Max 采样。
        sampled_index = multinomial_with_seed(
            torch.log(probs), sampling_seed, positions
        )
    # 中译：把形状 (batch, 1) 的列索引展平为一维并转 int32，作为 token id 返回。
    batch_next_token_ids = sampled_index.view(-1).to(torch.int32)
    return batch_next_token_ids


def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def get_token_ids_logprobs_batch_optimized(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[List[int]],
) -> Tuple[List, List]:
    """
    Vectorized batch processing for token ID logprobs extraction.

    Uses a single GPU kernel call for the entire batch instead of multiple
    separate calls, significantly improving performance for large batches.

    Args:
        logprobs: Log probabilities tensor [batch_size, vocab_size]
        token_ids_logprobs: List of token IDs to extract logprobs for

    Example:
        # Input: batch_size=3, vocab_size=5
        logprobs = torch.tensor([
            [-1.2, -2.1, -0.8, -3.0, -1.5],  # batch 0
            [-0.5, -1.8, -2.2, -1.1, -2.7],  # batch 1
            [-2.0, -0.9, -1.4, -2.8, -1.6],  # batch 2
        ])
        token_ids_logprobs = [[1, 3], [2], [0, 2, 4]]

        # Output:
        # values = [tensor([-2.1, -3.0]), tensor([-2.2]), tensor([-2.0, -1.4, -1.6])]
        # indices = [[1, 3], [2], [0, 2, 4]]
    """
    batch_size = len(token_ids_logprobs)
    device = logprobs.device

    # Step 1: Calculate lengths for each request, treating None as empty list
    # Example: [[1, 3], [2], [0, 2, 4]] -> token_lengths = tensor([2, 1, 3])
    token_lengths = torch.tensor(
        [len(token_ids or []) for token_ids in token_ids_logprobs], device=device
    )
    total_tokens = int(token_lengths.sum().item())  # 2 + 1 + 3 = 6

    # Handle edge case where no tokens are requested
    if total_tokens == 0:
        return [logprobs.new_empty(0) for _ in token_ids_logprobs], [
            [] for _ in token_ids_logprobs
        ]

    # Step 2: Build flattened indices using torch operations
    # Example: row_indices = [0, 0, 1, 2, 2, 2] (batch indices repeated by their lengths)
    row_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device), token_lengths
    )
    # Example: col_indices = [1, 3, 2, 0, 2, 4] (flattened token IDs from all requests)
    col_indices = torch.tensor(
        [
            token_id
            for token_ids in token_ids_logprobs
            for token_id in (token_ids or [])
        ],
        device=device,
        dtype=torch.long,
    )

    # Step 3: Single vectorized gather operation
    # Example: logprobs[row_indices, col_indices] -> [-2.1, -3.0, -2.2, -2.0, -1.4, -1.6]
    gathered_logprobs = logprobs[row_indices, col_indices]

    # Step 4: Split results back per request using torch operations
    # Example: split tensor [6] into chunks of sizes [2, 1, 3] -> [tensor(2), tensor(1), tensor(3)]
    split_logprobs = torch.split_with_sizes(
        gathered_logprobs, token_lengths.tolist(), dim=0
    )

    # Step 5: Format output to match expected return structure
    # Example: Convert split tensors back to list format with proper empty handling
    # i=0: [1,3] -> append split_logprobs[0] and [1,3]
    # i=1: [2] -> append split_logprobs[1] and [2]
    # i=2: [0,2,4] -> append split_logprobs[2] and [0,2,4]
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []

    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None and len(token_ids) > 0:
            output_token_ids_logprobs_val.append(split_logprobs[i])
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append(logprobs.new_empty(0))
            output_token_ids_logprobs_idx.append([])

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx


def apply_custom_logit_processor(
    logits: torch.Tensor,
    sampling_batch_info: SamplingBatchInfo,
    num_tokens_in_batch: int = 1,
):
    """Apply custom logit processors to the logits.
    This function will modify the logits in-place.
    num_tokens_in_batch is needed to support spec decoding, where each batch can contain multiple
    tokens. By default, we assume each batch contains only 1 token.
    """

    assert logits.shape[0] == len(sampling_batch_info) * num_tokens_in_batch, (
        f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "
        f"sampling_batch_info ({len(sampling_batch_info)}) x num_tokens_in_batch "
        f"({num_tokens_in_batch})"
    )

    for _, (
        processor,
        batch_mask,
    ) in sampling_batch_info.custom_logit_processor.items():
        # Get the batch indices that need to be processed
        batch_indices = batch_mask.nonzero(as_tuple=True)[0]

        assert batch_mask.shape[0] == len(sampling_batch_info), (
            f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )
        batch_mask = torch.repeat_interleave(batch_mask, num_tokens_in_batch)

        # Apply the processor to the logits
        logits[batch_mask] = processor(
            logits[batch_mask],
            [sampling_batch_info.custom_params[i] for i in batch_indices],
        )

        logger.debug(
            f"Custom logit processor {processor.__class__.__name__} is applied."
        )
