# srt/batch_invariant_ops

## 目录用途
本目录提供"批次无关"（batch-invariant）算子实现，移植自 thinking-machines-lab/batch_invariant_ops。其目标是让矩阵乘、log_softmax、求均值、RMSNorm 等运算的数值结果不随批大小变化，从而保证推理结果的可复现性与确定性。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 聚合并导出对外 API，包括 `set/enable/disable_batch_invariant_mode`、`is_batch_invariant_mode_enabled`、`matmul_persistent`、`log_softmax`、`mean_dim`、`rms_norm_batch_invariant`、`get_batch_invariant_attention_block_size` 及 `AttentionBlockSize`。 |
| `batch_invariant_ops.py` | 核心实现，包含 Triton 持久化 matmul/bmm/log_softmax/mean/RMSNorm 内核、可选 DeepGEMM 后端、批次无关算子（mm/addmm/bmm 等）以及通过 `torch.library` 切换批次无关模式的开关逻辑。 |
