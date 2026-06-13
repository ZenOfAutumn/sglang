# srt/debug_utils/comparator/dims_spec

## 目录用途
本目录定义并解析比对器使用的 dims 规格 DSL，把诸如 `t[cp:zigzag] (h___d)[tp] s # dp:=x tp:replicated` 的维度字符串解析为结构化的维名、并行修饰符与注释声明，并提供基于 torch named tensor 的维度命名/解析工具。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化，统一导出维名常量、枚举类型与解析/命名函数。 |
| `comment_parser.py` | 解析 dims 字符串中 `#` 注释段的 dp 别名（`dp:=x`）与 replicated 轴声明。 |
| `dim_parser.py` | 解析单个维 token（含融合维 `(h d)` 与方括号内的修饰符）为 `DimSpec`。 |
| `dims_parser.py` | 解析完整 dims 字符串为 `DimsSpec`，并提供 squeeze 维（`1`）相关的 `_SingletonDimUtil`。 |
| `modifier_parser.py` | 解析并行修饰符 token（如 `cp:zigzag+partial`）为 `ParallelModifier`（轴/顺序/规约）。 |
| `tensor_naming.py` | torch named tensor 工具：按名查维索引、应用/剥离维名、按名解析维度。 |
| `types.py` | 核心类型与常量：维名常量、`TokenLayout`、`ParallelAxis`、`Ordering`、`Reduction`、`DimSpec`、`DimsSpec` 等。 |
