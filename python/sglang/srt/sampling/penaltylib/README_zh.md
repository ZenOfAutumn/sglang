# srt/sampling/penaltylib

## 目录用途
本目录实现批处理（batched）的采样惩罚项库，用于在生成过程中按 batch 调整 logits。包含频率、存在、重复等惩罚以及最小新增 token 数限制，所有惩罚项由统一的编排器（orchestrator）管理生命周期与张量更新。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 汇总导出各惩罚器与编排器（`BatchedFrequencyPenalizer`、`BatchedMinNewTokensPenalizer`、`BatchedPenalizerOrchestrator`、`BatchedPresencePenalizer`、`BatchedRepetitionPenalizer`） |
| `orchestrator.py` | `BatchedPenalizerOrchestrator` 与惩罚器抽象基类 `_BatchedPenalizer`，统一管理各惩罚项的初始化、过滤、合并与对 logits 的应用 |
| `frequency_penalty.py` | `BatchedFrequencyPenalizer`，按 token 出现频次施加频率惩罚 |
| `presence_penalty.py` | `BatchedPresencePenalizer`，对已出现过的 token 施加存在惩罚 |
| `repetition_penalty.py` | `BatchedRepetitionPenalizer` 及 `apply_scaling_penalties`，按缩放系数施加重复惩罚 |
| `min_new_tokens.py` | `BatchedMinNewTokensPenalizer`，在达到最小新增 token 数前抑制 EOS 等结束 token |
