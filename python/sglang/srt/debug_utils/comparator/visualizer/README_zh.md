# srt/debug_utils/comparator/visualizer

## 目录用途
本目录负责把一对待比对张量渲染为多面板对比图（baseline/target/diff 热力图、差异直方图、2D 直方图、采样散点等），用于直观查看数值差异分布。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化，导出 `generate_comparison_figure`。 |
| `figure.py` | 对比图生成的主编排逻辑：构建面板上下文与面板列表并组装整图。 |
| `panels.py` | 各类面板的绘制函数（baseline/target/diff 热力图、直方图、散点等）。 |
| `preprocessing.py` | 张量预处理与可视化工具（降维到 2D、平衡长宽比、下采样、log10 变换、统计格式化）。 |
