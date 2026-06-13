# srt/weight_sync

## 目录用途
本目录提供训练侧到推理侧的权重同步工具，主要服务于 RL 等"训练-推理一体"场景。它支持将（可能是分片的）张量打平为桶进行高效传输，并将权重更新写入正在运行的 SGLang Engine。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `tensor_bucket.py` | 定义 `FlattenedTensorMetadata` 与 `FlattenedTensorBucket`，将多个张量打平到单个桶中并记录元数据，便于批量传输与还原。 |
| `utils.py` | 提供异步 `update_weights`，将 `DTensor`/`DeviceMesh` 下的参数批转换为序列化张量并经 `UpdateWeightsFromTensorReqInput` 推送给 Engine，含 `_preprocess_tensor_for_update_weights` 预处理。 |
