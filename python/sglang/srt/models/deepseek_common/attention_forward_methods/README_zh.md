# srt/models/deepseek_common/attention_forward_methods

## 目录用途
本目录为 DeepSeek 系列模型（DeepSeek V2/V3/V3.2 等）的注意力前向计算提供可插拔的实现集合。它把多头注意力（MHA）与吸收式多隐注意力（MLA）的多种变体拆分为独立的 Mixin 类，按运行设备（CUDA/CPU/ROCm/NPU）和后端能力进行分发，供 `deepseek_v2.py` 等模型层组合复用。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 导出 `AttnForwardMethod` 枚举及 MHA/MLA/CPU/ROCm 各 Mixin 类，统一对外暴露接口。 |
| `forward_methods.py` | 定义 `AttnForwardMethod` 枚举，枚举所有注意力前向方法（MHA、MLA、MHA_CHUNKED_KV、MHA_ONE_SHOT、MLA_FUSED_ROPE_ROCM/CPU、各类 NPU/DSA 变体等）。 |
| `forward_mha.py` | `DeepseekMHAForwardMixin`：多头注意力前向实现，含分块 KV、一次性（one-shot）等针对长前缀/显存优化的路径。 |
| `forward_mla.py` | `DeepseekMLAForwardMixin`：吸收式多隐注意力（MLA）前向实现，含 FP8 量化、deep_gemm 等优化路径。 |
| `forward_mla_fused_rope_cpu.py` | `DeepseekMLACpuForwardMixin`：CPU（Intel AMX）上带融合 RoPE 的 MLA 前向实现。 |
| `forward_mla_fused_rope_rocm.py` | `DeepseekMLARocmForwardMixin`：ROCm/AMD GPU 上带融合 RoPE 的 MLA 解码前向实现。 |
