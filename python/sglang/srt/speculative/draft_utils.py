from sglang.srt.runtime_context import attention_backends, get_spec
from sglang.srt.utils.common import (
    cpu_has_amx_support,
    is_blackwell,
    is_cpu,
    is_hip,
    is_musa,
    is_npu,
)


def _assert_draft_needs_no_conv_sidecar(draft_model_runner) -> None:
    """Refuse a multi-step draft decode backend for a draft with conv layers."""
    from sglang.srt.configs.inkling import InklingMMConfig, InklingModelConfig

    if isinstance(
        draft_model_runner.model_config.hf_config,
        (InklingModelConfig, InklingMMConfig),
    ):
        raise NotImplementedError(
            "Inkling's draft model runs its own short convs, which need the "
            "conv-state sidecar the multi-step draft decode backend cannot carry. "
            "Use --enable-multi-layer-eagle."
        )


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
        draft_model_runner,
        topk: int,
        speculative_num_steps: int,
        seed_dsa_topk_from_draft_extend: bool = False,
    ):
        self.draft_model_runner = draft_model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.seed_dsa_topk_from_draft_extend = seed_dsa_topk_from_draft_extend
        # The draft runner's own backend, not the process-wide config.
        self.draft_attn_backend = draft_model_runner.draft_attention_backend

    def _create_backend(
        self,
        backend_name: str,
        backend_map: dict,
        error_template: str,
        stamps_children: bool = False,
    ):
        # The split pair with the base-backend fallback already applied.
        prefill_backend, decode_backend = attention_backends()
        configured = (
            decode_backend
            if backend_name == "decode_attention_backend"
            else prefill_backend
        )
        backend_type = self.draft_attn_backend or configured

        if backend_type not in backend_map:
            raise ValueError(error_template.format(backend_type=backend_type))

        stamp, backend = backend_map[backend_type]()
        if backend is not None:
            if stamps_children:
                from sglang.srt.layers.attention.attention_registry import (
                    attn_backend_wrapper_for_draft_decode,
                )

                backend = attn_backend_wrapper_for_draft_decode(
                    self.draft_model_runner, backend
                )
            backend.prefill_attention_backend_str = stamp
            backend.decode_attention_backend_str = stamp
            if stamps_children:
                for child in backend.attn_backends:
                    child.prefill_attention_backend_str = stamp
                    child.decode_attention_backend_str = stamp
        return backend

    def create_decode_backend(self):
        # No multi-step draft backend for steps=0 (nospec) or steps=1.
        if self.speculative_num_steps <= 1:
            return None

        # Returns a per-step CONTAINER, not an AttentionBackend, so
        # attn_backend_wrapper_for_draft_extend cannot give it a conv sidecar.
        _assert_draft_needs_no_conv_sidecar(self.draft_model_runner)

        backend_map = {
            "flashinfer": self._create_flashinfer_decode_backend,
            "triton": self._create_triton_decode_backend,
            "intel_amx": self._create_intel_amx_decode_backend,
            "aiter": self._create_aiter_decode_backend,
            "fa3": self._create_fa3_decode_backend,
            "hybrid_linear_attn": self._create_hybrid_linear_attn_decode_backend,
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
            stamps_children=True,
        )

    def create_draft_extend_backend(self):
        """创建草稿模型的 extend（prefill）注意力后端。

        根据 speculative_attention_mode 选择使用 decode 或 prefill 后端字段。

        返回：注意力后端实例
        """
        backend_map = {
            "flashinfer": self._create_flashinfer_prefill_backend,
            "triton": self._create_triton_prefill_backend,
            "intel_amx": self._create_intel_amx_prefill_backend,
            "aiter": self._create_aiter_prefill_backend,
            "fa3": self._create_fa3_prefill_backend,
            "hybrid_linear_attn": self._create_hybrid_linear_attn_prefill_backend,
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
            if get_spec().speculative_attention_mode == "decode"
            else "prefill_attention_backend"
        )
        backend = self._create_backend(
            backend_name,
            backend_map,
            "EAGLE is not supported in attention backend {backend_type}",
        )
        # A draft with conv layers of its own (Inkling) needs its sidecar here too.
        from sglang.srt.layers.attention.attention_registry import (
            attn_backend_wrapper_for_draft_extend,
        )

        wrapped = attn_backend_wrapper_for_draft_extend(
            self.draft_model_runner, backend
        )
        if wrapped is not backend and wrapped is not None and backend is not None:
            wrapped.prefill_attention_backend_str = (
                backend.prefill_attention_backend_str
            )
            wrapped.decode_attention_backend_str = backend.decode_attention_backend_str
        return wrapped

    def _create_dsa_decode_backend(self):
        """创建 DeepseekSparseAttn 多步 decode 后端。"""
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnMultiStepBackend,
        )

        return (
            "dsa",
            DeepseekSparseAttnMultiStepBackend(
                self.draft_model_runner,
                self.topk,
                self.speculative_num_steps,
                seed_dsa_topk_from_draft_extend=self.seed_dsa_topk_from_draft_extend,
            ),
        )

    def _create_dsa_prefill_backend(self):
        """创建 DeepseekSparseAttn prefill 后端。"""
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        return (
            "dsa",
            DeepseekSparseAttnBackend(
                self.draft_model_runner,
                skip_prefill=False,
                seed_dsa_topk_from_draft_extend=self.seed_dsa_topk_from_draft_extend,
            ),
        )

    def _create_flashinfer_decode_backend(self):
        if not self.draft_model_runner.use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackend,
            )

            return (
                "flashinfer",
                FlashInferMultiStepDraftBackend(
                    self.draft_model_runner, self.topk, self.speculative_num_steps
                ),
            )
        else:
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAMultiStepDraftBackend,
            )

            return (
                "flashinfer",
                FlashInferMLAMultiStepDraftBackend(
                    self.draft_model_runner, self.topk, self.speculative_num_steps
                ),
            )

    def _create_triton_decode_backend(self):
        """创建 Triton 多步 decode 后端。"""
        from sglang.srt.layers.attention.triton_backend import (
            TritonMultiStepDraftBackend,
        )

        return (
            "triton",
            TritonMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            ),
        )

    def _create_intel_amx_decode_backend(self):
        from sglang.srt.layers.attention.intel_amx_backend import (
            IntelAMXMultiStepDraftBackend,
        )

        return (
            "intel_amx",
            IntelAMXMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            ),
        )

    def _create_hybrid_linear_attn_decode_backend(self):
        if is_cpu() and cpu_has_amx_support():
            return self._create_intel_amx_decode_backend()
        if is_blackwell():
            return self._create_triton_decode_backend()
        return self._create_fa3_decode_backend()

    def _create_hybrid_linear_attn_prefill_backend(self):
        if is_cpu() and cpu_has_amx_support():
            return self._create_intel_amx_prefill_backend()
        if is_blackwell():
            return self._create_triton_prefill_backend()
        return self._create_fa3_prefill_backend()

    def _create_aiter_decode_backend(self):
        """创建 Aiter（AMD ROCm）多步 decode 后端。"""
        from sglang.srt.layers.attention.aiter_backend import AiterMultiStepDraftBackend

        return (
            "aiter",
            AiterMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            ),
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

        return (
            f"fa{fa_impl_ver}",
            FlashAttentionMultiStepBackend(
                self.draft_model_runner,
                self.topk,
                self.speculative_num_steps,
                fa_impl_ver=fa_impl_ver,
            ),
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

        return (
            "flashmla",
            FlashMLAMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            ),
        )

    def _create_trtllm_mha_decode_backend(self):
        """创建 TRT-LLM MHA 多步 decode 后端。"""
        from sglang.srt.layers.attention.trtllm_mha_backend import (
            TRTLLMHAAttnMultiStepDraftBackend,
        )

        return (
            "trtllm_mha",
            TRTLLMHAAttnMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            ),
        )

    def _create_trtllm_mla_decode_backend(self, backend: str = "trtllm-gen"):
        if not self.draft_model_runner.use_mla_backend:
            raise ValueError(
                "trtllm_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.trtllm_mla_backend import (
            TRTLLMMLAMultiStepDraftBackend,
        )

        return (
            "trtllm_mla",
            TRTLLMMLAMultiStepDraftBackend(
                self.draft_model_runner,
                self.topk,
                self.speculative_num_steps,
                backend=backend,
            ),
        )

    def _create_cutedsl_mla_decode_backend(self):
        if not self.draft_model_runner.use_mla_backend:
            raise ValueError(
                "cutedsl_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.cutedsl_mla_backend import (
            CuteDslMLAMultiStepDraftBackend,
        )

        return (
            "cutedsl_mla",
            CuteDslMLAMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            ),
        )

    def _create_tokenspeed_mla_decode_backend(self):
        if not self.draft_model_runner.use_mla_backend:
            raise ValueError(
                "tokenspeed_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLAMultiStepDraftBackend,
        )

        return (
            "tokenspeed_mla",
            TokenspeedMLAMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            ),
        )

    def _create_ascend_decode_backend(self):
        """创建昇腾 NPU 多步 decode 后端。"""
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnMultiStepDraftBackend,
        )

        return (
            "ascend",
            AscendAttnMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            ),
        )

    def _create_dsv4_decode_backend(self):
        # Decode here is the EAGLE multi-step draft decode path.
        if is_npu():
            from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
                DeepseekV4AscendMultiStepDraftBackend,
            )

            return (
                "dsv4",
                DeepseekV4AscendMultiStepDraftBackend(
                    self.draft_model_runner, self.topk, self.speculative_num_steps
                ),
            )
        elif is_hip():
            from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                DeepseekV4MultiStepBackend,
            )
        else:
            from sglang.srt.layers.attention.deepseek_v4_backend import (
                DeepseekV4MultiStepBackend,
            )

        return (
            "dsv4",
            DeepseekV4MultiStepBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            ),
        )

    def _create_flashinfer_prefill_backend(self):
        if not self.draft_model_runner.use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferAttnBackend,
            )

            return (
                "flashinfer",
                FlashInferAttnBackend(self.draft_model_runner, skip_prefill=False),
            )
        else:
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAAttnBackend,
            )

            return (
                "flashinfer",
                FlashInferMLAAttnBackend(self.draft_model_runner, skip_prefill=False),
            )

    def _create_triton_prefill_backend(self):
        """创建 Triton prefill 后端。"""
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        return (
            "triton",
            TritonAttnBackend(self.draft_model_runner, skip_prefill=False),
        )

    def _create_intel_amx_prefill_backend(self):
        from sglang.srt.layers.attention.intel_amx_backend import IntelAMXAttnBackend

        return ("intel_amx", IntelAMXAttnBackend(self.draft_model_runner))

    def _create_aiter_prefill_backend(self):
        """创建 Aiter prefill 后端。"""
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        return ("aiter", AiterAttnBackend(self.draft_model_runner, skip_prefill=False))

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
        return (
            f"fa{fa_impl_ver}",
            FlashAttentionBackend(
                self.draft_model_runner, skip_prefill=False, fa_impl_ver=fa_impl_ver
            ),
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

        return (
            "trtllm_mha",
            TRTLLMHAAttnBackend(self.draft_model_runner, skip_prefill=False),
        )

    def _create_trtllm_mla_prefill_backend(self):
        if not self.draft_model_runner.use_mla_backend:
            raise ValueError(
                "trtllm_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

        return (
            "trtllm_mla",
            TRTLLMMLABackend(self.draft_model_runner, skip_prefill=False),
        )

    def _create_tokenspeed_mla_prefill_backend(self):
        if not self.draft_model_runner.use_mla_backend:
            raise ValueError(
                "tokenspeed_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
        )

        return (
            "tokenspeed_mla",
            TokenspeedMLABackend(self.draft_model_runner, skip_prefill=False),
        )

    def _create_ascend_prefill_backend(self):
        """创建昇腾 NPU prefill 后端。"""
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )

        return ("ascend", AscendAttnBackend(self.draft_model_runner))

    def _create_flashmla_prefill_backend(self):
        from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

        return (
            "flashmla",
            FlashMLABackend(self.draft_model_runner, skip_prefill=False),
        )

    def _create_dsv4_prefill_backend(self):
        # On NPU the "dsv4" backend resolves to the Ascend V4 subclass; its
        # draft-extend path uses the registered DSV4 prefill backend.
        if is_npu():
            from sglang.srt.layers.attention.attention_registry import (
                ATTENTION_BACKENDS,
            )

            return ("dsv4", ATTENTION_BACKENDS["dsv4"](self.draft_model_runner))
        elif is_hip():
            from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                DeepseekV4HipRadixBackend,
            )

            return (
                "dsv4",
                DeepseekV4HipRadixBackend(self.draft_model_runner, skip_prefill=False),
            )
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        return (
            "dsv4",
            DeepseekV4AttnBackend(self.draft_model_runner, skip_prefill=False),
        )
