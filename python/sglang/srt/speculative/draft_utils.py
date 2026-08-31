"""草稿模型注意力后端工厂模块。

本模块提供 DraftBackendFactory，用于根据 server_args 和硬件平台
为草稿模型（EAGLE / Frozen-KV MTP 等）创建 decode 和 extend 注意力后端。

支持的后端类型：
    - flashinfer / flashinfer_mla
    - triton
    - aiter (AMD ROCm)
    - fa3 / fa4 (FlashAttention v3/v4)
    - flashmla
    - trtllm_mha / trtllm_mla
    - cutedsl_mla
    - tokenspeed_mla
    - dsa / nsa (DeepseekSparseAttn)
    - ascend (NPU)
    - dsv4 (Deepseek V4)
    - hybrid_linear_attn
"""

import logging

from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils.common import is_blackwell, is_hip, is_musa

logger = logging.getLogger(__name__)


class DraftBackendFactory:
    """草稿模型注意力后端工厂。

    根据 server_args 配置和硬件平台，为草稿模型创建 decode 和 extend 注意力后端。
    支持多种后端类型，并通过 draft_attn_backend 配置覆盖默认选择。

    属性：
        server_args:           服务器参数
        draft_model_runner:    草稿模型 runner
        topk:                  每步每请求候选数
        speculative_num_steps: 草稿推理步数
        draft_attn_backend:    草稿专用注意力后端名称（可选覆盖）

    示例：
        >>> factory = DraftBackendFactory(server_args, runner, topk=8, num_steps=4)
        >>> decode_backend = factory.create_decode_backend()
        >>> extend_backend = factory.create_draft_extend_backend()
    """

    def __init__(
        self,
        server_args: ServerArgs,
        draft_model_runner,
        topk: int,
        speculative_num_steps: int,
    ):
        """初始化草稿后端工厂。

        参数：
            server_args:           服务器参数
            draft_model_runner:    草稿模型 runner
            topk:                  每步每请求候选数
            speculative_num_steps: 草稿推理步数
        """
        self.server_args = server_args
        self.draft_model_runner = draft_model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.draft_attn_backend = server_args.speculative_draft_attention_backend

    def _create_backend(
        self, backend_name: str, backend_map: dict, error_template: str
    ):
        """根据配置从 backend_map 中创建指定的注意力后端。

        优先级：draft_attn_backend > server_args[backend_name] > server_args.attention_backend

        参数：
            backend_name:   server_args 中的后端字段名（如 decode_attention_backend）
            backend_map:    后端名称 → 创建函数的映射
            error_template: 后端不支持时的错误信息模板（含 {backend_type}）

        返回：注意力后端实例

        异常：ValueError - 当后端类型不在 backend_map 中时
        """
        backend_type = (
            self.draft_attn_backend
            if self.draft_attn_backend
            else getattr(self.server_args, backend_name)
        )
        if backend_type is None:
            backend_type = self.server_args.attention_backend

        if backend_type not in backend_map:
            raise ValueError(error_template.format(backend_type=backend_type))

        return backend_map[backend_type]()

    def create_decode_backend(self):
        """创建草稿模型的 decode 注意力后端。

        当 speculative_num_steps == 1 时返回 None（无需多步后端）。

        返回：注意力后端实例或 None

        示例：
            >>> backend = factory.create_decode_backend()
            >>> # 返回 FlashInferMultiStepDraftBackend 等
        """
        if self.speculative_num_steps == 1:
            return None

        backend_map = {
            "flashinfer": self._create_flashinfer_decode_backend,
            "triton": self._create_triton_decode_backend,
            "aiter": self._create_aiter_decode_backend,
            "fa3": self._create_fa3_decode_backend,
            "hybrid_linear_attn": (
                self._create_fa3_decode_backend
                if not is_blackwell()
                else self._create_triton_decode_backend
            ),
            "flashmla": self._create_flashmla_decode_backend,
            "trtllm_mha": self._create_trtllm_mha_decode_backend,
            "trtllm_mla": self._create_trtllm_mla_decode_backend,
            "cutedsl_mla": self._create_cutedsl_mla_decode_backend,
            "tokenspeed_mla": self._create_tokenspeed_mla_decode_backend,
            "dsa": self._create_dsa_decode_backend,
            "nsa": self._create_dsa_decode_backend,  # "nsa" 是 "dsa" 的废弃别名
            "ascend": self._create_ascend_decode_backend,
            "fa4": self._create_fa4_decode_backend,
            "dsv4": self._create_dsv4_decode_backend,
        }

        return self._create_backend(
            "decode_attention_backend",
            backend_map,
            "EAGLE is not supported in decode attention backend {backend_type}",
        )

    def create_draft_extend_backend(self):
        """创建草稿模型的 extend（prefill）注意力后端。

        根据 speculative_attention_mode 选择使用 decode 或 prefill 后端字段。

        返回：注意力后端实例
        """
        backend_map = {
            "flashinfer": self._create_flashinfer_prefill_backend,
            "triton": self._create_triton_prefill_backend,
            "aiter": self._create_aiter_prefill_backend,
            "fa3": self._create_fa3_prefill_backend,
            "hybrid_linear_attn": (
                self._create_fa3_prefill_backend
                if not is_blackwell()
                else self._create_triton_prefill_backend
            ),
            "flashmla": self._create_flashmla_prefill_backend,
            "trtllm_mha": self._create_trtllm_mha_prefill_backend,
            "trtllm_mla": self._create_trtllm_mla_prefill_backend,
            # cute-dsl MLA 仅支持 decode；draft-extend 回退到 trtllm-gen
            "cutedsl_mla": self._create_trtllm_mla_prefill_backend,
            "tokenspeed_mla": self._create_tokenspeed_mla_prefill_backend,
            "dsa": self._create_dsa_prefill_backend,
            "nsa": self._create_dsa_prefill_backend,  # "nsa" 是 "dsa" 的废弃别名
            "ascend": self._create_ascend_prefill_backend,
            "fa4": self._create_fa4_prefill_backend,
            "dsv4": self._create_dsv4_prefill_backend,
        }
        backend_name = (
            "decode_attention_backend"
            if self.server_args.speculative_attention_mode == "decode"
            else "prefill_attention_backend"
        )
        return self._create_backend(
            backend_name,
            backend_map,
            "EAGLE is not supported in attention backend {backend_type}",
        )

    def _create_dsa_decode_backend(self):
        """创建 DeepseekSparseAttn 多步 decode 后端。"""
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnMultiStepBackend,
        )

        return DeepseekSparseAttnMultiStepBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_dsa_prefill_backend(self):
        """创建 DeepseekSparseAttn prefill 后端。"""
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        return DeepseekSparseAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_flashinfer_decode_backend(self):
        """创建 FlashInfer 多步 decode 后端（自动区分 MLA/非 MLA）。"""
        if not get_global_server_args().use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackend,
            )

            return FlashInferMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )
        else:
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAMultiStepDraftBackend,
            )

            return FlashInferMLAMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )

    def _create_triton_decode_backend(self):
        """创建 Triton 多步 decode 后端。"""
        from sglang.srt.layers.attention.triton_backend import (
            TritonMultiStepDraftBackend,
        )

        return TritonMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_aiter_decode_backend(self):
        """创建 Aiter（AMD ROCm）多步 decode 后端。"""
        from sglang.srt.layers.attention.aiter_backend import AiterMultiStepDraftBackend

        return AiterMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_fa_decode_backend(self, fa_impl_ver: int = 3):
        """创建 FlashAttention 多步 decode 后端（指定 FA 版本）。

        参数：
            fa_impl_ver: FlashAttention 实现版本（3 或 4）
        """
        if not is_musa():
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionMultiStepBackend,
            )
        else:
            from sglang.srt.hardware_backend.musa.attention.flashattention_backend import (
                MusaFlashAttentionMultiStepBackend as FlashAttentionMultiStepBackend,
            )

        return FlashAttentionMultiStepBackend(
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
            fa_impl_ver=fa_impl_ver,
        )

    def _create_fa3_decode_backend(self):
        """创建 FlashAttention v3 多步 decode 后端。"""
        return self._create_fa_decode_backend(fa_impl_ver=3)

    def _create_fa4_decode_backend(self):
        """创建 FlashAttention v4 多步 decode 后端。"""
        return self._create_fa_decode_backend(fa_impl_ver=4)

    def _create_flashmla_decode_backend(self):
        """创建 FlashMLA 多步 decode 后端。"""
        from sglang.srt.layers.attention.flashmla_backend import (
            FlashMLAMultiStepDraftBackend,
        )

        return FlashMLAMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_trtllm_mha_decode_backend(self):
        """创建 TRT-LLM MHA 多步 decode 后端。"""
        from sglang.srt.layers.attention.trtllm_mha_backend import (
            TRTLLMHAAttnMultiStepDraftBackend,
        )

        return TRTLLMHAAttnMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_trtllm_mla_decode_backend(self, backend: str = "trtllm-gen"):
        """创建 TRT-LLM MLA 多步 decode 后端。

        参数：
            backend: 后端实现类型（"trtllm-gen" 或 "cute-dsl"）

        异常：ValueError - 当模型非 MLA 时
        """
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "trtllm_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.trtllm_mla_backend import (
            TRTLLMMLAMultiStepDraftBackend,
        )

        return TRTLLMMLAMultiStepDraftBackend(
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
            backend=backend,
        )

    def _create_cutedsl_mla_decode_backend(self):
        """创建 cute-dsl MLA 多步 decode 后端（基于 trtllm_mla）。"""
        return self._create_trtllm_mla_decode_backend(backend="cute-dsl")

    def _create_tokenspeed_mla_decode_backend(self):
        """创建 Tokenspeed MLA 多步 decode 后端。

        异常：ValueError - 当模型非 MLA 时
        """
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "tokenspeed_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLAMultiStepDraftBackend,
        )

        return TokenspeedMLAMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_ascend_decode_backend(self):
        """创建昇腾 NPU 多步 decode 后端。"""
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnMultiStepDraftBackend,
        )

        return AscendAttnMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_dsv4_decode_backend(self):
        """创建 Deepseek V4 多步 decode 后端（自动区分 HIP/CUDA）。"""
        if is_hip():
            from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                DeepseekV4MultiStepBackend,
            )
        else:
            from sglang.srt.layers.attention.deepseek_v4_backend import (
                DeepseekV4MultiStepBackend,
            )

        return DeepseekV4MultiStepBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_flashinfer_prefill_backend(self):
        """创建 FlashInfer prefill 后端（自动区分 MLA/非 MLA）。"""
        if not get_global_server_args().use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferAttnBackend,
            )

            return FlashInferAttnBackend(self.draft_model_runner, skip_prefill=False)
        else:
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAAttnBackend,
            )

            return FlashInferMLAAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_triton_prefill_backend(self):
        """创建 Triton prefill 后端。"""
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        return TritonAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_aiter_prefill_backend(self):
        """创建 Aiter prefill 后端。"""
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        return AiterAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_fa_prefill_backend(self, fa_impl_ver: int = 3):
        """创建 FlashAttention prefill 后端（指定 FA 版本）。

        参数：
            fa_impl_ver: FlashAttention 实现版本（3 或 4）
        """
        if not is_musa():
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
            )
        else:
            from sglang.srt.hardware_backend.musa.attention.flashattention_backend import (
                MusaFlashAttentionBackend as FlashAttentionBackend,
            )
        return FlashAttentionBackend(
            self.draft_model_runner, skip_prefill=False, fa_impl_ver=fa_impl_ver
        )

    def _create_fa3_prefill_backend(self):
        """创建 FlashAttention v3 prefill 后端。"""
        return self._create_fa_prefill_backend(fa_impl_ver=3)

    def _create_fa4_prefill_backend(self):
        """创建 FlashAttention v4 prefill 后端。"""
        return self._create_fa_prefill_backend(fa_impl_ver=4)

    def _create_trtllm_mha_prefill_backend(self):
        """创建 TRT-LLM MHA prefill 后端。"""
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        return TRTLLMHAAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_trtllm_mla_prefill_backend(self):
        """创建 TRT-LLM MLA prefill 后端。

        异常：ValueError - 当模型非 MLA 时
        """
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "trtllm_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

        return TRTLLMMLABackend(self.draft_model_runner, skip_prefill=False)

    def _create_tokenspeed_mla_prefill_backend(self):
        """创建 Tokenspeed MLA prefill 后端。

        异常：ValueError - 当模型非 MLA 时
        """
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "tokenspeed_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
        )

        return TokenspeedMLABackend(self.draft_model_runner, skip_prefill=False)

    def _create_ascend_prefill_backend(self):
        """创建昇腾 NPU prefill 后端。"""
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )

        return AscendAttnBackend(self.draft_model_runner)

    def _create_flashmla_prefill_backend(self):
        """创建 FlashMLA prefill 后端（暂不支持，返回 None）。"""
        logger.warning(
            "flashmla prefill backend is not yet supported for draft extend."
        )
        return None

    def _create_dsv4_prefill_backend(self):
        """创建 Deepseek V4 prefill 后端（自动区分 HIP/CUDA）。"""
        if is_hip():
            from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                DeepseekV4HipRadixBackend,
            )

            return DeepseekV4HipRadixBackend(
                self.draft_model_runner, skip_prefill=False
            )
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        return DeepseekV4AttnBackend(self.draft_model_runner, skip_prefill=False)
