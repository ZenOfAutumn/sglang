# 注意力后端注册表模块。
#
# 本模块通过装饰器 `register_attention_backend(name)` 把「后端名称字符串」
# 映射到「创建该后端实例的工厂函数」，统一收集进全局字典 `ATTENTION_BACKENDS`。
# ModelRunner 的 `_get_attention_backend_from_str()` 会按名称在此表中查找并调用
# 对应工厂函数来构造注意力后端。各工厂内部采用延迟导入(lazy import)，避免在
# 模块加载期就引入重型/平台相关依赖，也能规避循环导入。
import logging
import warnings
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    attention_backends_of,
    resolved_view,
)
from sglang.srt.configs.hybrid_arch import (
    hybrid_gdn_config,
    hybrid_lightning_config,
    kimi_linear_config,
    mamba2_config,
    mambaish_config,
)
from sglang.srt.configs.linear_attn_model_registry import (
    get_linear_attn_config,
    import_backend_class,
)
from sglang.srt.runtime_context import (
    get_parallel,
    get_platform,
    get_spec,
)
from sglang.srt.utils import get_device_capability, is_hip, is_musa, is_npu

# 是否运行在摩尔线程 MUSA 平台(在 fa3 分支里据此走 MUSA 专用实现)。
_is_musa = is_musa()
_is_npu = is_npu()
_is_hip = is_hip()

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # 仅用于类型标注的导入，放在 TYPE_CHECKING 下以规避运行期循环导入。
    # evade circular imports
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.model_runner import ModelRunner

# 全局注册表：{后端名称: 工厂函数}。由下方各 @register_attention_backend 填充。
ATTENTION_BACKENDS = {}


def register_attention_backend(name):
    """装饰器工厂：把被装饰的工厂函数以 `name` 为键注册进 ATTENTION_BACKENDS。

    用法：在工厂函数上加 `@register_attention_backend("xxx")`，之后即可通过
    名称 "xxx" 查表拿到该工厂并创建后端。
    """

    def decorator(fn):
        # 以名称为键登记工厂函数；原函数原样返回，不改变其行为。
        ATTENTION_BACKENDS[name] = fn
        return fn

    return decorator


@register_attention_backend("flashinfer")
def create_flashinfer_backend(runner):
    # FlashInfer 后端：非 MLA 模型走标准实现，MLA 模型走专用的 MLA 实现。
    import torch

    if not runner.use_mla_backend:
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )

        # Init streams
        if get_spec().speculative_algorithm == "EAGLE":
            if (
                not hasattr(runner, "plan_stream_for_flashinfer")
                or not runner.plan_stream_for_flashinfer
            ):
                runner.plan_stream_for_flashinfer = torch.cuda.Stream()
        return FlashInferAttnBackend(
            runner, init_new_workspace=runner.init_new_workspace
        )
    else:
        from sglang.srt.layers.attention.flashinfer_mla_backend import (
            FlashInferMLAAttnBackend,
        )

        return FlashInferMLAAttnBackend(runner)


@register_attention_backend("trtllm_mla")
def create_trtllm_mla_backend(runner):
    # TensorRT-LLM 的 MLA 后端：仅适用于 MLA 模型，否则报错。
    if not runner.use_mla_backend:
        raise ValueError("trtllm_mla backend can only be used with MLA models.")
    if get_parallel().dcp_enabled and get_spec().speculative_algorithm is not None:

        _, decode_backend = attention_backends_of(resolved_view(runner.server_args))
        if decode_backend == "trtllm_mla":
            raise ValueError(
                "trtllm_mla cannot serve decode context parallelism with speculative "
                "decoding: it does not forward the cyclic DCP metadata to its decode "
                "kernel and returns no rank-local LSE for the cross-rank merge. "
                "Select cutedsl_mla or tokenspeed_mla."
            )
    from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

    return TRTLLMMLABackend(runner)


@register_attention_backend("tokenspeed_mla")
def create_tokenspeed_mla_backend(runner):
    if not runner.use_mla_backend:
        raise ValueError("tokenspeed_mla backend can only be used with MLA models.")
    from sglang.srt.layers.attention.tokenspeed_mla_backend import (
        TokenspeedMLABackend,
    )

    return TokenspeedMLABackend(runner)


@register_attention_backend("cutedsl_mla")
def create_cutedsl_mla_backend(runner):
    # CUTE-DSL 的 MLA 后端：复用 TRTLLMMLABackend，但以 backend="cute-dsl" 切换实现；仅限 MLA 模型。
    if not runner.use_mla_backend:
        raise ValueError("cutedsl_mla backend can only be used with MLA models.")
    from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend

    return CuteDslMLABackend(runner)


@register_attention_backend("aiter")
def create_aiter_backend(runner):
    # AMD AITER 后端(面向 AMD/HIP 平台的注意力实现)。
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

    return AiterAttnBackend(runner)


@register_attention_backend("wave")
def create_wave_backend(runner):
    from sglang.srt.layers.attention.wave_backend import WaveAttnBackend

    return WaveAttnBackend(runner)


@register_attention_backend("ascend")
def create_ascend_backend(runner):
    # 华为昇腾 NPU 注意力后端。
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )

    return AscendAttnBackend(runner)


@register_attention_backend("dsa")
def create_dsa_backend(runner):
    # DeepSeek 稀疏注意力(DSA)后端。
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    return DeepseekSparseAttnBackend(runner)


@register_attention_backend("nsa")
def _create_nsa_compat(runner):
    # "nsa" 是 "dsa" 的已废弃别名，仅作兼容，发出弃用告警后转调 dsa 工厂。
    warnings.warn(
        "attention-backend='nsa' is deprecated; use 'dsa' instead. "
        "The alias will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_dsa_backend(runner)


@register_attention_backend("dsv4")
def create_dsv4_backend(runner):
    if _is_npu:
        from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
            DeepseekV4AscendAttnBackend,
        )

        return DeepseekV4AscendAttnBackend(runner)
    elif _is_hip:
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            DeepseekV4HipRadixBackend,
        )

        logger.info(
            "Using DeepseekV4HipRadixBackend for compressed attention backend (HIP)."
        )
        return DeepseekV4HipRadixBackend(runner)
    else:
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        logger.info("Using DeepseekV4AttnBackend for dsv4 attention backend (CUDA).")
        return DeepseekV4AttnBackend(runner)


@register_attention_backend("triton")
def create_triton_backend(runner):
    # Triton 注意力后端：不支持 encoder-decoder 的 cross attention。
    assert not runner.model_config.is_encoder_decoder, (
        "Cross attention is not supported in the triton attention backend. "
        "Please use `--attention-backend flashinfer`."
    )
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    return TritonAttnBackend(runner)


@register_attention_backend("torch_native")
def create_torch_native_backend(runner):
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    return TorchNativeAttnBackend(runner)


@register_attention_backend("flex_attention")
def create_flex_attention_backend(runner):
    from sglang.srt.layers.attention.torch_flex_backend import TorchFlexAttnBackend

    return TorchFlexAttnBackend(runner)


@register_attention_backend("flashmla")
def create_flashmla_backend(runner):
    from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

    return FlashMLABackend(runner)


@register_attention_backend("fa3")
def create_flashattention_v3_backend(runner):
    # FlashAttention v3 后端：依据设备算力(SM 版本)选择实现。
    # 非 MUSA: 要求 SM>=80 且 <=90(SM80 仅非 MLA，SM90 均可)；MUSA: 走 MUSA 专用实现。
    major, minor = get_device_capability()
    if not _is_musa:
        assert (major == 8 and not runner.use_mla_backend) or major == 9, (
            "FlashAttention v3 Backend requires SM>=80 and SM<=90. "
            "Please use `--attention-backend flashinfer`."
        )
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )

        return FlashAttentionBackend(runner)
    else:
        assert major == 3 and minor >= 1, (
            "FlashAttention v3 Backend requires MP>=31. "
            "Please use `--attention-backend triton`."
        )
        from sglang.srt.hardware_backend.musa.attention import (
            MusaFlashAttentionBackend,
        )

        return MusaFlashAttentionBackend(runner)


@register_attention_backend("fa4")
def create_flashattention_v4_backend(runner):
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionBackend,
    )

    return FlashAttentionBackend(runner, fa_impl_ver=4)


@register_attention_backend("cutlass_mla")
def create_cutlass_mla_backend(runner):
    from sglang.srt.layers.attention.cutlass_mla_backend import CutlassMLABackend

    return CutlassMLABackend(runner)


@register_attention_backend("trtllm_mha")
def create_trtllm_mha_backend(runner):
    # TensorRT-LLM 的 MHA 后端：仅适用于非 MLA 模型，否则报错。
    if runner.use_mla_backend:
        raise ValueError("trtllm_mha backend can only be used with non-MLA models.")
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

    return TRTLLMHAAttnBackend(runner)


@register_attention_backend("hpc_ops")
def create_hpc_ops_backend(runner):
    if runner.use_mla_backend:
        raise ValueError("hpc_ops backend can only be used with non-MLA models.")
    if runner.model_config.is_encoder_decoder:
        raise ValueError(
            "Cross attention is not supported in the hpc_ops attention backend."
        )
    if get_spec().speculative_algorithm is not None:
        raise ValueError(
            "hpc_ops backend does not support speculative decoding for now."
        )
    from sglang.srt.layers.attention.hpc_ops_backend import HPCOpsAttnBackend

    return HPCOpsAttnBackend(runner)


@register_attention_backend("intel_amx")
def create_intel_amx_backend(runner):
    from sglang.srt.layers.attention.intel_amx_backend import IntelAMXAttnBackend

    return IntelAMXAttnBackend(runner)


@register_attention_backend("dual_chunk_flash_attn")
def create_dual_chunk_flash_attn_backend(runner):
    from sglang.srt.layers.attention.dual_chunk_flashattention_backend import (
        DualChunkFlashAttentionBackend,
    )

    return DualChunkFlashAttentionBackend(runner)


def attn_backend_wrapper_for_draft_extend(
    runner: "ModelRunner", full_attn_backend: "AttentionBackend"
):
    """Apply the model's attention wrapper to a DRAFT-EXTEND backend, if it needs one.

    ``DraftBackendFactory`` skips :func:`attn_backend_wrapper`, which is right for
    the mamba hybrids whose MTP draft is all softmax attention. Inkling's draft has
    its own short convs, so it must expose ``conv_state_metadata`` too.
    """
    from sglang.srt.configs.dots3 import Dots3Config
    from sglang.srt.configs.inkling import InklingMMConfig, InklingModelConfig

    if isinstance(runner.model_config.hf_config, (InklingModelConfig, InklingMMConfig)):
        return attn_backend_wrapper(runner, full_attn_backend)
    if isinstance(runner.model_config.hf_text_config, Dots3Config):
        return attn_backend_wrapper(runner, full_attn_backend)
    return full_attn_backend


def attn_backend_wrapper_for_draft_decode(runner: "ModelRunner", backend):
    """Apply the Dots model wrapper to per-step draft backends."""
    from sglang.srt.configs.dots3 import Dots3Config

    if not hasattr(runner, "model_config"):
        return backend
    hf_text_config = runner.model_config.hf_text_config
    if isinstance(hf_text_config, Dots3Config):
        return hf_text_config.wrap_draft_decode_attention_backend(backend)
    return backend


@register_attention_backend("minicpm_flashattn")
def create_minicpm_flashattn_backend(runner):
    from sglang.srt.layers.attention.minicpm.backend import MiniCPMSparseBackend

    return MiniCPMSparseBackend(runner, use_flashinfer=False)


@register_attention_backend("minicpm_flashinfer")
def create_minicpm_flashinfer_backend(runner):
    from sglang.srt.layers.attention.minicpm.backend import MiniCPMSparseBackend

    return MiniCPMSparseBackend(runner, use_flashinfer=True)


def attn_backend_wrapper(runner: "ModelRunner", full_attn_backend: "AttentionBackend"):
    """
    Wrapper for special models like hybrid GDN, so we don't
    need to change the code of the original attention backend.

    中译：用于「混合(hybrid)」模型(如 hybrid GDN、Mamba2、KDA、Lightning 等线性
    注意力混合架构)的包装器。它在不改动原始全注意力(full attention)后端代码的前提下，
    把「全注意力后端」与「线性注意力后端」按层组合成 HybridLinearAttnBackend。
    若模型不是混合架构，则原样返回传入的 full_attn_backend。
    """
    # 混合 GDN 仅支持非 MLA 模型。
    assert not (
        hybrid_gdn_config(runner.model_config) is not None and runner.use_mla_backend
    ), "hybrid_gdn can only be used with non-MLA models."

    from sglang.srt.configs.dots3 import Dots3Config
    from sglang.srt.configs.model_config import is_minimax_sparse

    if isinstance(runner.model_config.hf_text_config, Dots3Config):
        return runner.model_config.hf_text_config.wrap_attention_backend(
            runner, full_attn_backend
        )

    if is_minimax_sparse(runner.model_config.hf_config):
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxHybridAttnBackend,
            MiniMaxSparseAttnBackend,
        )

        sparse_backend = MiniMaxSparseAttnBackend(runner)
        return MiniMaxHybridAttnBackend(
            full_attn_backend, sparse_backend, sparse_backend.sparse_layer_ids
        )

    if cfg := mambaish_config(runner.model_config):
        from sglang.srt.configs.inkling import InklingMMConfig, InklingModelConfig

        if isinstance(
            runner.model_config.hf_config, (InklingModelConfig, InklingMMConfig)
        ):
            from sglang.srt.layers.attention.linear.inkling_sconv_backend import (
                InklingShortConvAttnBackend,
                InklingShortConvHybridAttnBackend,
            )

            return InklingShortConvHybridAttnBackend(
                full_attn_backend,
                InklingShortConvAttnBackend(runner),
                cfg.full_attention_layer_ids,
            )

        from sglang.kernels.ops.attention.fla.utils import check_environments
        from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
        from sglang.srt.layers.attention.linear.lightning_backend import (
            LightningAttentionBackend,
        )
        from sglang.srt.layers.attention.linear.utils import (
            resolve_linear_attn_backends,
        )
        from sglang.srt.utils import (
            is_blackwell,
            is_npu,
            is_xpu,
        )

        if not is_npu():
            from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
                HybridLinearAttnBackend,
                Mamba2AttnBackend,
            )
            from sglang.srt.layers.attention.linear.gdn_backend import (
                GDNAttnBackend,
                flashinfer_gdn_prefill_default,
            )

            if is_xpu():
                from sglang.srt.hardware_backend.xpu.attention.xpu_gdn_backend import (
                    XpuGDNAttnBackend as GDNAttnBackend,
                )
        else:
            from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
                AscendGDNAttnBackend as GDNAttnBackend,
            )
            from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
                AscendHybridLinearAttnBackend as HybridLinearAttnBackend,
            )
            from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
                AscendMamba2AttnBackend as Mamba2AttnBackend,
            )

        # 校验线性注意力所需的运行环境，并初始化其配置。
        check_environments()
        prefill_default = None
        if hybrid_gdn_config(runner.model_config) is not None and not is_npu():
            prefill_default = flashinfer_gdn_prefill_default(runner)
        runner.linear_attn_backends = resolve_linear_attn_backends(
            prefill_default=prefill_default
        )
        hybrid_backend_cls = HybridLinearAttnBackend
        if hybrid_gdn_config(runner.model_config) is not None:
            if is_blackwell():
                if get_platform().is_sm120:
                    allowed = {"triton", "trtllm_mha", "flashinfer"}
                else:
                    allowed = {"triton", "trtllm_mha", "fa4"}
                prefill_be = runner.prefill_attention_backend_str
                decode_be = runner.decode_attention_backend_str
                assert prefill_be in allowed and decode_be in allowed, (
                    f"Only {allowed} backends are supported on Blackwell GPUs for hybrid GDN models. "
                    f"Got prefill={prefill_be}, decode={decode_be}."
                )
            elif is_npu():
                assert (
                    runner.prefill_attention_backend_str == "ascend"
                    and runner.decode_attention_backend_str == "ascend"
                ), "ascend backend is the only supported backend on NPU for hybrid GDN models, use --attention-backend ascend to specify the backend."
            logger.info(f"Using hybrid linear attention backend for hybrid GDN models.")
            linear_attn_backend = GDNAttnBackend(runner)
        elif mamba2_config(runner.model_config) is not None:
            from sglang.srt.configs.lfm2 import Lfm2Config
            from sglang.srt.configs.lfm2_moe import Lfm2MoeConfig
            from sglang.srt.configs.lfm2_vl import Lfm2VlConfig
            from sglang.srt.configs.zaya import ZayaConfig

            # Short-conv hybrids (ZAYA1 CCA, LFM2 short conv) share a conv-state
            # sidecar that owns the per-request state plumbing and is invoked by
            # the model via conv_state_metadata (never as a full-vs-linear
            # alternative). Other mamba2 models keep the full Mamba2 SSM backend.
            short_conv_cfgs = (
                ZayaConfig,
                Lfm2Config,
                Lfm2MoeConfig,
                Lfm2VlConfig,
            )
            if isinstance(mamba2_config(runner.model_config), short_conv_cfgs):
                if is_npu():
                    # The model conv layers call
                    # get_attn_backend().conv_state_metadata() unconditionally,
                    # but the Ascend hybrid/mamba backend has no such method.
                    # Fail here (before model execution) with a clear message
                    # rather than an AttributeError deep in the first conv layer.
                    raise NotImplementedError(
                        "Short-conv hybrid models (ZAYA1 CCA, LFM2 / LFM2-MoE) "
                        "are not yet supported on NPU: the conv-state sidecar "
                        "(ShortConvAttnBackend.conv_state_metadata) has no Ascend "
                        "implementation. Add an Ascend conv-state backend before "
                        "serving these models on NPU."
                    )
                from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
                    ShortConvHybridAttnBackend,
                )
                from sglang.srt.layers.attention.linear.short_conv_backend import (
                    ShortConvAttnBackend,
                )

                linear_attn_backend = ShortConvAttnBackend(runner)
                hybrid_backend_cls = ShortConvHybridAttnBackend
            else:
                linear_attn_backend = Mamba2AttnBackend(runner)
        elif kimi_linear_config(runner.model_config) is not None:
            if _is_npu:
                from sglang.srt.hardware_backend.npu.attention.ascend_kda_backend import (
                    AscendKDAAttnBackend,
                    AscendKDAHybridLinearAttnBackend,
                )

                linear_attn_backend = AscendKDAAttnBackend(runner)
                hybrid_backend_cls = AscendKDAHybridLinearAttnBackend
            else:
                linear_attn_backend = KDAAttnBackend(runner)
        elif hybrid_lightning_config(runner.model_config) is not None:
            linear_attn_backend = LightningAttentionBackend(runner)
        else:
            spec_result = get_linear_attn_config(runner.model_config.hf_config)
            if spec_result is not None:
                spec, _ = spec_result
                cfg = runner.model_config
                BackendClass = import_backend_class(spec.backend_class_name)
                linear_attn_backend = BackendClass(runner)
                if spec.hybrid_backend_class_name is not None:
                    hybrid_backend_cls = import_backend_class(
                        spec.hybrid_backend_class_name
                    )
            else:
                raise ValueError(
                    "Expected hybrid GDN or NemotronH models, but got unknown model. "
                    "If this is a custom hybrid model, use register_linear_attn_model() "
                    "from sglang.srt.configs.linear_attn_model_registry."
                )
        # 确定哪些层使用「全注意力」(其余层走线性注意力)。
        if runner.is_draft_worker:
            # FIXME: we assume that MTP/NEXTN always use full-attention.
            # 中译：FIXME：这里假设 MTP/NEXTN 草稿模型始终使用全注意力。
            full_attn_layers = [0]
        else:
            full_attn_layers = cfg.full_attention_layer_ids
        return hybrid_backend_cls(
            full_attn_backend, linear_attn_backend, full_attn_layers
        )

    # 非混合模型：原样返回全注意力后端。
    return full_attn_backend


@register_attention_backend("intel_xpu")
def create_intel_xpu_backend(runner):
    # Intel XPU(Intel GPU)注意力后端。
    from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend

    return XPUAttentionBackend(runner)
