# srt/constrained

## 目录用途
本目录实现 SGLang 的约束解码（constrained decoding）与语法（grammar）后端，用于强制模型输出符合指定结构（如 JSON、正则、EBNF、结构化标签）。它定义统一的语法对象与后端抽象，接入 xgrammar、outlines、llguidance 等多种第三方引擎，并支持跳跃前进（jump-forward）加速及与推理流程的衔接。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `base_grammar_backend.py` | 语法后端抽象层：语法对象基类 `BaseGrammarObject`、后端基类 `BaseGrammarBackend`、统计 `GrammarStats`、无效语法占位 `InvalidGrammarObject`，及后端注册/创建函数 |
| `grammar_manager.py` | 语法管理器 `GrammarManager`，负责异步初始化、缓存与调度各请求的语法对象 |
| `xgrammar_backend.py` | xgrammar 后端实现 `XGrammarGrammarBackend` 及语法对象 `XGrammarGrammar` |
| `outlines_backend.py` | outlines 后端实现 `OutlinesGrammarBackend`，含从对象构建正则的工具 |
| `outlines_jump_forward.py` | outlines 的跳跃前进映射 `OutlinesJumpForwardMap` 及相关状态推进/磁盘缓存逻辑 |
| `llguidance_backend.py` | llguidance（Guidance）后端实现 `GuidanceBackend` 及语法对象 `GuidanceGrammar` |
| `reasoner_grammar_backend.py` | 推理（reasoner）包装后端 `ReasonerGrammarBackend`，在推理思考段落后再施加语法约束 |
| `utils.py` | 约束相关工具函数，如判断是否为旧版结构化标签 `is_legacy_structural_tag` |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `triton_ops` | 约束解码所需的 Triton 算子（token bitmask 原地应用） |
