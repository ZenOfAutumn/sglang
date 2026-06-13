# srt/layers/rotary_embedding

## 目录用途
本目录实现旋转位置编码（RoPE）及其各类缩放变体，是 `rotary_embedding.py` 的模块化替代实现。涵盖基础 RoPE、YaRN、Llama3、Deepseek、动态 NTK、Phi3 LongRoPE、Fourier、DualChunk 以及多模态 MRoPE 等，并提供 `get_rope` 工厂函数、底层算子与 Triton kernel。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 公共 API 入口，重导出 `RotaryEmbedding`、`get_rope`、`MRotaryEmbedding`、`apply_rotary_pos_emb` 及 YaRN 辅助函数。 |
| `base.py` | `RotaryEmbedding` 基类与 `LinearScalingRotaryEmbedding`，含多平台（CUDA/HIP/NPU/CPU 等）分发。 |
| `factory.py` | 工厂函数 `get_rope`、`get_rope_cpu`、`get_rope_wrapper`，按配置选择并构造对应 RoPE 变体。 |
| `mrope.py` | 多模态 RoPE：`MRotaryEmbedding`、`YaRNScalingMRotaryEmbedding`、`Ernie4_5_VLRotaryEmbedding` 及交错 RoPE 应用。 |
| `mrope_rope_index.py` | 多模态位置索引计算 `get_rope_index`，支持 Qwen2/3-VL、Qwen3-Omni、GLM4V、Ernie4.5。 |
| `rope_variant.py` | RoPE 缩放变体：Phi3LongRoPE、Fourier、DeepseekScaling、Llama3、Llama4Vision、DynamicNTK(Alpha)、DualChunk。 |
| `triton_kernels.py` | 多模态 RoPE 的 Triton JIT kernel：融合 mrope 前向、Ernie4.5 qk 融合等。 |
| `utils.py` | 基础旋转算子：`rotate_neox`、`rotate_gptj`、`apply_rotary_emb`、`apply_rotary_pos_emb_native/npu` 等。 |
| `yarn.py` | YaRN 缩放：`YaRNScalingRotaryEmbedding` 及修正维度/范围、线性 ramp mask、mscale 等辅助函数。 |

## 子目录
无子目录。
