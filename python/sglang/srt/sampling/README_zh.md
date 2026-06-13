# srt/sampling

## 目录用途
本目录负责 SGLang 的采样（sampling）逻辑，包括请求级采样参数的定义与校验、批处理采样状态的组织与张量化，以及可插拔的自定义 logits 处理器。采样所需的各类惩罚项（重复、频率、存在、最小新增 token 数）单独放在 `penaltylib` 子目录中。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `sampling_params.py` | `SamplingParams` 数据类，定义并校验 temperature、top-p/top-k、各类惩罚、最大长度、正则约束等单请求采样参数 |
| `sampling_batch_info.py` | `SamplingBatchInfo`，把一个 batch 内多请求的采样参数聚合为张量，整合惩罚项编排器与自定义 logits 处理器，并提供 batch 合并工具 |
| `custom_logit_processor.py` | 自定义 logits 处理器抽象基类及内置实现（禁用 token、思考预算控制 ThinkingBudget、DeepSeek-OCR 防重复 n-gram 等） |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `penaltylib` | 批处理惩罚项库：频率、存在、重复惩罚与最小新增 token 数限制，由编排器统一调度 |
