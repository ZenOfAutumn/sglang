# srt/debug_utils/source_patcher

## 目录用途
本目录提供运行时源码热补丁工具，按 YAML 配置对目标函数的源码做匹配/替换/前插/后插编辑，并重新编译加载，从而在不改动仓库源文件的情况下注入调试逻辑（如插入转储调用）。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化，导出 `CodePatcher`、`apply_patches_from_config`、`patch_function` 及补丁类型。 |
| `code_patcher.py` | 核心补丁器：解析 YAML 配置、对函数取源码并应用编辑、注入前导 import 并重新加载补丁后的函数。 |
| `source_editor.py` | 源码文本编辑引擎：按 `EditSpec` 顺序做匹配定位与替换/前插/后插，并处理缩进对齐。 |
| `types.py` | 补丁类型定义：`EditSpec`、`PatchSpec`、`PatchConfig`、`PatchState` 及 `PatchApplicationError`。 |
