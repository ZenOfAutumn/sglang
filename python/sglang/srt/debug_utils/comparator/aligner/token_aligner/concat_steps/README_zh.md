# srt/debug_utils/comparator/aligner/token_aligner/concat_steps

## 目录用途
本目录实现 token 对齐的 `concat_steps`（简单拼接）模式：按 step 顺序把各步骤张量在 token 维拼接，再截断到两侧公共 token 数。同时提供 THD 布局下的全局序列长度加载辅助。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化，导出 `execute_token_aligner_concat_steps`。 |
| `executor.py` | 拼接各 step 张量并截断到 `min(total_x, total_y)` 个 token；含 token 维解析与回退逻辑。 |
| `thd_seq_lens_loader.py` | 仅加载 THD 布局下每 step 的全局 per-seq token 数（借助 smart 模块的辅助插件）。 |
