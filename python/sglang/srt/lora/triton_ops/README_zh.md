# srt/lora/triton_ops

## 目录用途
该目录用 Triton 实现 LoRA 计算所需的各类高性能 GPU 核，包括分段 SGMV 的 shrink(LoRA A)与 expand(LoRA B)、嵌入层 LoRA、QKV 与 gate_up 投影的 LoRA B、以及 MoE 融合 LoRA 核。供 `TritonLoRABackend` 与 `ChunkedSgmvLoRABackend` 调用。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 汇总导出各前向函数(sgemm a/b、qkv_b、gate_up_b、embedding、chunked 系列、fused_moe_lora)。 |
| `sgemm_lora_a.py` | `sgemm_lora_a_fwd`：分段 LoRA A(shrink)的 Triton 核与前向封装。 |
| `sgemm_lora_b.py` | `sgemm_lora_b_fwd`：分段 LoRA B(expand)的 Triton 核与前向封装。 |
| `qkv_lora_b.py` | `qkv_lora_b_fwd`：QKV 合并投影的 LoRA B 计算核。 |
| `gate_up_lora_b.py` | `gate_up_lora_b_fwd`：gate_up 合并投影的 LoRA B 计算核。 |
| `embedding_lora_a.py` | `embedding_lora_a_fwd`：词嵌入层 LoRA A 计算核。 |
| `chunked_sgmv_shrink.py` | `chunked_sgmv_lora_shrink_forward`：固定分块版 SGMV shrink 核(缓存特化)。 |
| `chunked_sgmv_expand.py` | `chunked_sgmv_lora_expand_forward`：固定分块版 SGMV expand 核(缓存特化)。 |
| `chunked_embedding_lora_a.py` | `chunked_embedding_lora_a_forward`：分块版嵌入层 LoRA A 核。 |
| `fused_moe_lora_kernel.py` | `fused_moe_lora` 及 shrink/expand 子核：MoE 融合 LoRA 计算(改编自 vLLM)。 |
