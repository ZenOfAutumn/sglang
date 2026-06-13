# srt/debug_utils/comparator/tensor_comparator

## 目录用途
本目录实现单张量对的核心数值比对：计算张量统计量（均值、标准差、分位数等）与差异指标（相对误差、最大绝对差及其坐标、per-token 相对误差等），并将比对结果格式化为可读文本。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化，导出 `compare_tensor_pair`。 |
| `comparator.py` | 计算单张量信息与一对张量的差异（含形状统一、分位数采样阈值、per-token 相对误差）。 |
| `formatter.py` | 将比对信息（`TensorComparisonInfo`、replicated 校验等）格式化为 rich/文本输出。 |
| `types.py` | 比对数据类型：`TensorStats`、`TensorInfo`、`DiffInfo`、`TensorComparisonInfo` 及默认分位数。 |
