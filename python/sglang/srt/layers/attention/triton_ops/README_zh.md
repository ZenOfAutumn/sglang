# srt/layers/attention/triton_ops

## 目录用途
本目录提供通用的 Triton 注意力算子实现，作为 Triton 后端及部分其它后端的底层算子，涵盖 decode、extend（含 prefix cache）、prefill 注意力，以及 Double Sparsity、ROCm MLA RoPE 解码、注意力 state 合并与 FP8 KV 写入等。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| decode_attention.py | 解码阶段注意力（普通/分组 GQA 两阶段 flash-decoding）Triton 算子。 |
| extend_attention.py | extend/prefill 带前缀 KV 的扩展注意力算子（含 unified 索引版本）。 |
| prefill_attention.py | 无前缀缓存的 prefill 上下文注意力算子（`context_attention_fwd`）。 |
| double_sparsity_attention.py | Double Sparsity 的 flash-decode 近似稀疏注意力多阶段算子。 |
| rocm_mla_decode_rope.py | ROCm 平台 MLA 解码 + RoPE 融合的分组注意力算子。 |
| merge_state.py | 注意力分块结果合并（merge state）的 Triton kernel 与封装。 |
| trtllm_fp8_kv_kernel.py | 向 KV cache 写入 FP8 量化 K/V 的融合 Triton 算子（TRT-LLM 路径）。 |
