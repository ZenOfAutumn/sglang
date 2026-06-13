# srt/layers/utils

## 目录用途
本目录为 `srt/layers` 提供层相关的通用工具，包括权重名解析与参数处理、上下文并行（context parallel）拆分与聚合、Triton 哈希 kernel、logprob 计算以及多平台算子分发基类等。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包入口，重导出 `common` 中的公共符号与 `MultiPlatformOp`。 |
| `common.py` | 通用工具：`get_layer_id` 解析层号、`pad_or_narrow_weight`、`copy_or_rebind_param`、`PPMissingLayer` 占位层。 |
| `cp_utils.py` | 上下文并行工具：`ContextParallelMetadata`、序列拆分/重建、all-gather 重排、KV cache 保存与 prefill CP 注意力前向。 |
| `hash.py` | MurmurHash32 的 Triton kernel 实现：`rotl32`、`fmix32`、`murmur3_mix`、`murmur_hash32` 等。 |
| `logprob.py` | logprob 计算：`LogprobStage`、`InputLogprobsResult`、top/token-ids logprobs 提取及温度归一化与推测解码相关逻辑。 |
| `multi_platform.py` | `MultiPlatformOp` 基类，按硬件平台（CUDA/HIP/NPU/CPU 等）分发算子实现并支持 kernel API 调试日志。 |

## 子目录
无子目录。
