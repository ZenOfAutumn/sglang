# srt/mem_cache/sparsity/algorithms

## 目录用途
稀疏注意力算法实现集合。定义统一抽象基类，并提供具体的可检索 KV cache 压缩算法（Quest 边界框估计、DeepSeek NSA 原生索引器），将 token 级稀疏统一为 page_size=1 的按页稀疏处理。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| __init__.py | 导出 `BaseSparseAlgorithm`、`BaseSparseAlgorithmImpl`、`DeepSeekNSAAlgorithm`、`QuestAlgorithm`。 |
| base_algorithm.py | 稀疏算法抽象基类 `BaseSparseAlgorithm` 及其实现基类，定义统一 TopK 检索接口。 |
| deepseek_nsa.py | `DeepSeekNSAAlgorithm`：基于 NSA 原生索引器进行 TopK 检索的稀疏算法。 |
| quest_algorithm.py | `QuestAlgorithm`：Quest 论文的按页边界框（min/max）criticality 稀疏估计算法。 |
