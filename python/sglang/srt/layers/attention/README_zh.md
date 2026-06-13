# srt/layers/attention

## 目录用途
本目录汇集 SGLang 的各种注意力后端（Attention Backend）实现及其公共基础设施。每个后端封装了在不同硬件/算子库（FlashAttention、FlashInfer、Triton、TensorRT-LLM、Aiter、Mamba 线性注意力、NSA 等）上执行 prefill/decode/CUDA Graph/投机解码（draft）的注意力计算逻辑，并通过统一的 `AttentionBackend` 抽象接入模型运行时。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| base_attn_backend.py | 定义所有后端的抽象基类 `AttentionBackend`，规定 metadata 初始化、CUDA Graph、prefill/decode forward 等接口。 |
| attention_registry.py | 注意力后端注册表与工厂函数，按名称创建各后端（flashinfer/triton/aiter/wave/ascend/nsa/flashmla/cutlass_mla/trtllm 等）。 |
| aiter_backend.py | AMD ROCm Aiter 后端（`AiterAttnBackend`），含索引更新器与多步 draft 后端。 |
| cutlass_mla_backend.py | 基于 CUTLASS 的 MLA decode 后端（`CutlassMLABackend`，继承 FlashInfer MLA）。 |
| double_sparsity_backend.py | Double Sparsity 近似稀疏注意力后端（`DoubleSparseAttnBackend`）。 |
| dual_chunk_flashattention_backend.py | Dual Chunk FlashAttention 长上下文后端（`DualChunkFlashAttentionBackend`）及垂直/斜线稀疏算子。 |
| flashattention_backend.py | FlashAttention v3/v4 后端（`FlashAttentionBackend`），含本地注意力虚拟批、多步 draft 后端与 metadata kernel。 |
| flashinfer_backend.py | FlashInfer 后端（`FlashInferAttnBackend`），含 decode/prefill 索引更新器与多步 draft 后端。 |
| flashinfer_mla_backend.py | FlashInfer MLA（DeepSeek 多头潜在注意力）后端（`FlashInferMLAAttnBackend`）及分块 KV runner。 |
| flashmla_backend.py | FlashMLA decode 后端（`FlashMLABackend`，继承 FlashInfer MLA）。 |
| hybrid_attn_backend.py | 混合后端（`HybridAttnBackend`），prefill 与 decode 分别使用不同后端。 |
| hybrid_linear_attn_backend.py | 混合线性注意力后端（`HybridLinearAttnBackend`/`Mamba2AttnBackend`），融合 Mamba 状态管理与全注意力。 |
| intel_amx_backend.py | Intel AMX CPU 后端（`IntelAMXAttnBackend`）。 |
| nsa_backend.py | Native Sparse Attention 后端（`NativeSparseAttnBackend`）及多步 draft 后端、NSA 索引器 metadata。 |
| tbo_backend.py | Two-Batch-Overlap（TBO）封装后端（`TboAttnBackend`），对子批做 CUDA Graph 切分调度。 |
| torch_flex_backend.py | PyTorch FlexAttention 后端（`TorchFlexAttnBackend`）。 |
| torch_native_backend.py | PyTorch 原生 SDPA 后端（`TorchNativeAttnBackend`）。 |
| triton_backend.py | Triton 后端（`TritonAttnBackend`），含多步 draft 后端与滑动窗口缓冲管理。 |
| trtllm_mha_backend.py | TensorRT-LLM MHA 后端（`TRTLLMHAAttnBackend`，继承 FlashInfer）。 |
| trtllm_mla_backend.py | TensorRT-LLM MLA 后端（`TRTLLMMLABackend`），含 draft extend 的 query padding/unpad 与 FP8 量化算子。 |
| wave_backend.py | AMD Wave 后端（`WaveAttnBackend`），基于 wave_ops 算子。 |
| xpu_backend.py | Intel XPU 后端（`XPUAttentionBackend`）。 |
| merge_state.py | 注意力分块结果合并（merge state，attention state combine）的对外封装。 |
| utils.py | 后端公共工具：构建 FlashInfer/FlashMLA KV 索引、KV cache 写入、RoPE 融合、量化等 Triton 算子。 |
| vision.py | 视觉编码器注意力实现（SDPA/Triton/Flash3/Flash4/FlashInfer/Aiter/Ascend 及统一 `VisionAttention`）。 |
| vision_utils.py | 视觉注意力辅助：ViT dummy head 配置更新与权重 padding。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| fla | Fused Linear Attention（gated delta rule/KDA 等）的 Triton 算子集合。 |
| linear | 线性注意力后端（GDN/KDA/Lightning/SegLA）及其元数据与工具。 |
| mamba | Mamba/Mamba2 混合器、因果卷积与 SSM 相关实现。 |
| nsa | Native Sparse Attention 的索引器、KV 量化/反量化与稀疏算子。 |
| triton_ops | 通用 Triton 注意力算子（decode/extend/prefill/merge 等）。 |
| wave_ops | 基于 AMD Wave 的注意力算子（decode/extend/prefill）。 |
