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

## 第三方语法引擎实现原理

### 统一抽象：mask logits

约束解码的本质是：**在每一步采样前，把所有"当前语法状态下不允许出现"的 token 的 logits 置为 `-inf`**，从而保证采样结果一定符合语法。SGLang 通过 `BaseGrammarObject` 抽象统一了这套流程，所有第三方引擎都实现同一组接口：

| 接口 | 作用 |
| --- | --- |
| `allocate_vocab_mask` | 分配本批次的词表掩码张量 |
| `fill_vocab_mask` | 由引擎根据**当前语法状态**计算出允许/禁止的 token，写入掩码 |
| `move_vocab_mask` / `apply_vocab_mask` | 把掩码搬到 GPU 并原地作用到 logits（禁止项置 `-inf`） |
| `accept_token` | 采样出 token 后，**推进语法状态机** |
| `rollback` | 回退 k 个 token 的状态（投机解码 verify 失败时使用） |
| `try_jump_forward` / `jump_and_retokenize` | 跳跃前进加速（见下文） |

调度流程：`GrammarManager` 异步编译语法（编译较慢，故用 `ThreadPoolExecutor` + 缓存），每个请求持有一个语法对象，解码循环里 `fill_vocab_mask → apply_vocab_mask → 采样 → accept_token` 循环推进。

三个后端的差异主要在**「语法如何编译成状态机」与「掩码如何表示」**：

### 1. xgrammar（`xgrammar_backend.py`，默认推荐）

- **原理**：把 JSON Schema / EBNF / 正则 / 结构化标签编译成 `CompiledGrammar`，运行时用 `GrammarMatcher` 维护下推自动机（PDA，支持上下文无关文法，能处理 JSON 的嵌套括号匹配）。
- **掩码表示**：使用**位掩码（token bitmask）**——每个 token 用 1 bit 表示是否允许，`allocate_token_bitmask` 分配，`fill_vocab_mask` 由 matcher 填充，再用 `apply_token_bitmask_inplace_triton`（ROCm 上为 `..._cuda`）的 **Triton/CUDA 算子** 原地作用到 logits。位掩码比 bool 张量省 32 倍显存、应用更快。
- **状态推进**：`matcher.accept_token` 接受 token 并推进 PDA；支持 `rollback`（默认上限 `MAX_ROLLBACK_TOKENS=200`），因此**兼容投机解码**。
- **特点**：性能最好、功能最全（CFG/结构化标签），是 SGLang 的默认后端。

### 2. outlines（`outlines_backend.py`）

- **原理**：把约束统一**转换成正则表达式**（JSON Schema 经 `build_regex_from_schema` 转正则），再用 `interegular` 把正则编译成 **有限状态自动机（FSM）**，由 `RegexGuide` 维护。
- **掩码表示**：使用 **bool 张量**（`torch.zeros(..., dtype=torch.bool)`）。`fill_vocab_mask` 调 `guide.get_next_instruction(state).tokens` 拿到当前状态允许的 token 列表，把这些位置置 0、其余置 1；`apply_vocab_mask` 用 `masked_fill_(..., -inf)`。
- **状态推进**：`state = guide.get_next_state(state, token)`，是纯 FSM 状态转移。
- **局限**：正则/FSM 表达能力弱于 PDA，**无法表达任意上下文无关文法**（如任意深度嵌套），但对正则约束足够；不支持 `rollback`，与投机解码兼容性弱。
- **跳跃前进**：通过 `OutlinesJumpForwardMap`（`outlines_jump_forward.py`）实现，见下文。

### 3. llguidance / Guidance（`llguidance_backend.py`）

- **原理**：用 Guidance 项目的 `LLMatcher` + `LLTokenizer`，把语法 `grammar_from` 序列化后构建 matcher，支持 JSON Schema、正则与结构化标签。
- **掩码表示**：同样是**位掩码**，但用 llguidance 自带的 `allocate_token_bitmask` / `fill_next_token_bitmask` / `apply_token_bitmask_inplace`（库内置的 torch 算子）。
- **状态推进**：`ll_matcher.consume_token` 消费 token；对 EOS / stop 有专门处理；支持 `rollback`（注意 stop 后的 EOS 不计入 matcher 计数，回退时需 -1）。
- **特点**：作为 xgrammar 之外的可选高性能后端，rollback 友好。

### 跳跃前进（Jump-Forward）加速

当语法在某个状态下**只有唯一一条确定路径**（例如 JSON 里键名后必然跟 `":"`、或固定字段名）时，无需逐 token 让模型生成，可以**直接把这段确定文本"跳"过去**，省掉若干次前向。流程：`try_jump_forward` 探测可跳跃的字符串 → `jump_forward_str_state` 得到跳跃文本与新状态 → `jump_and_retokenize` 重新分词并对齐语法状态（因为跳过的文本可能与原 token 边界不一致，需重新 tokenize）。outlines 通过预计算的 `OutlinesJumpForwardMap`（带磁盘缓存）实现，xgrammar 由 matcher 内部支持。

### reasoner 包装（`reasoner_grammar_backend.py`）

`ReasonerGrammarBackend` 包装上述任一后端：对带"思考段落（reasoning）"的模型（如 DeepSeek-R1），**在思考段（`<think>...</think>`）内不施加语法约束**（让模型自由推理），只在思考结束后才对最终答案施加结构约束。

### 小结对照

| 后端 | 状态机 | 表达能力 | 掩码 | rollback / 投机解码 | 定位 |
| --- | --- | --- | --- | --- | --- |
| xgrammar | 下推自动机（PDA） | CFG（最强） | token bitmask + Triton/CUDA | 支持 | 默认推荐 |
| outlines | 有限状态机（FSM） | 正则（较弱） | bool 张量 | 不支持 | 正则约束/轻量 |
| llguidance | LLMatcher | CFG/正则/结构标签 | token bitmask（库内置） | 支持 | 高性能可选 |
