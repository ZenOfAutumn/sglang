# xgrammar：从概率生成到结构正确的约束解码

> **30 字解释**：xgrammar 在每次采样前封住不合语法的 token，只让模型沿合法路径生成。
>
> **一句话原理**：把 JSON Schema、正则、EBNF 或结构化标签编译成可逐 token 推进的语法状态机；每一步根据当前状态生成词表位掩码，将非法 token 的 logits 置为 \(-\infty\)，采样后再用新 token 推进状态。

## 目录

1. [为什么需要 xgrammar](#1-为什么需要-xgrammar)
2. [第一性原理：结构正确性从哪里来](#2-第一性原理结构正确性从哪里来)
3. [核心机制：编译、掩码、采样、推进](#3-核心机制编译掩码采样推进)
4. [逐 token 示例：生成一个 JSON 对象](#4-逐-token-示例生成一个-json-对象)
5. [SGLang 中的完整执行链路](#5-sglang-中的完整执行链路)
6. [四类约束及示例](#6-四类约束及示例)
7. [高级机制](#7-高级机制)
8. [对立面、边界与反模式](#8-对立面边界与反模式)
9. [性能成本与适用场景](#9-性能成本与适用场景)
10. [源码阅读地图](#10-源码阅读地图)
11. [语义压缩：最后只记住这些](#11-语义压缩最后只记住这些)

---

## 1. 为什么需要 xgrammar

### 1.1 没有约束解码时，工程师如何妥协

大语言模型的原生输出是一个概率分布，而不是 JSON 对象、函数调用或程序语法树。即使 prompt 明确要求“只输出 JSON”，模型仍可能生成：

```text
当然可以，结果如下：
```json
{"name": "Alice", "age": 18,}
```
```

这里至少存在四类失败：

1. 输出了 JSON 之外的解释文字；
2. 输出了 Markdown 代码围栏；
3. 最后一个字段多了逗号，JSON 语法非法；
4. 即使 JSON 能解析，字段类型也可能不符合业务要求。

过去常见的补救路线是：

```text
加强 prompt
   ↓ 仍可能失败
解析后校验
   ↓ 失败则修复 JSON
修复失败
   ↓ 重试模型
增加延迟、成本和不确定性
```

这些方案都在**生成完成后**处理错误。xgrammar 改变了问题：不再问“错误输出如何修”，而是问“错误 token 能否从一开始就不允许被采样”。

### 1.2 xgrammar 推翻了什么假设

旧假设是：

> 模型先自由生成完整文本，应用再判断文本是否合法。

xgrammar 的新假设是：

> 在每一个 token 被采样之前，系统已经知道当前哪些 token 仍可能构成合法结果。

因此，结构合法性从“事后概率事件”变成了“采样空间的硬约束”。模型仍决定合法候选中哪个最合适，但不能选择会使语法立即失败的 token。

---

## 2. 第一性原理：结构正确性从哪里来

先把问题降到不可再拆的基本事实。

### 2.1 基本事实一：模型只提供分数

在第 \(t\) 个解码步骤，模型输出词表中每个 token 的 logit：

\[
z_t \in \mathbb{R}^{V}
\]

其中 \(V\) 是词表大小。采样器将 logits 转成概率：

\[
p(x_t=i)=\frac{e^{z_{t,i}}}{\sum_{j=1}^{V}e^{z_{t,j}}}
\]

模型并不知道应用程序最终要把文本交给 JSON 解析器、正则匹配器还是工具执行器。它只预测“下一个 token 像什么”。

### 2.2 基本事实二：语法是前缀相关的

一个 token 是否合法，取决于已经生成的前缀。

以 JSON 为例：

```text
前缀                              下一步可能合法的内容
空字符串                          {
{"age"                            :
{"age":                           数字或空白
{"age": 1                         数字、空白或 }
{"age": 18}                       EOS
```

所以不能预先制作一个永远不变的“JSON 合法 token 列表”。同一个 `}` token：

- 在对象字段值结束后可能合法；
- 在空输出开头一定非法；
- 在字符串内部只是普通字符的一部分，其含义又不同。

约束器必须保存一个随输出前缀变化的**语法状态**。

### 2.3 基本事实三：token 不等于字符

模型按 token 生成，语法通常按字符定义。一个 token 可能是：

- 一个字符，例如 `{`；
- 多个字符，例如 `"age"`；
- 字符的一部分；
- 带前导空格的片段，例如 ` age`；
- 含特殊字节或特殊 token。

因此不能只检查“下一个字符是否合法”。约束引擎必须结合 tokenizer 词表，判断**每个候选 token 对当前语法状态的影响**。

### 2.4 基本事实四：概率归零即可禁止采样

如果非法 token 的 logit 被设为 \(-\infty\)，那么：

\[
e^{-\infty}=0
\]

它经过 softmax 后概率就是 0，任何正常采样策略都不会选中它。因此约束解码最小闭环只有四步：

```text
知道当前语法状态
      ↓
找出仍合法的 token
      ↓
把其余 token 的 logits 置为 -inf
      ↓
采样并用结果推进语法状态
```

这就是 xgrammar 最核心、不可再简化的原理。

---

## 3. 核心机制：编译、掩码、采样、推进

### 3.1 结构类比：机场安检门

可以把一次解码类比为旅客逐个通过机场安检：

| 约束解码 | 机场安检 |
| --- | --- |
| 词表中的全部 token | 等待过检的全部旅客 |
| 当前语法状态 | 当前安检规则与已通过记录 |
| token bitmask | 放行名单 |
| 非法 logit 置 \(-\infty\) | 禁止登机 |
| 采样 | 从放行者中选出下一位 |
| `accept_token` | 记录该旅客，并更新后续规则 |

安检门不决定“谁最值得登机”，只决定“谁有资格登机”。同样，xgrammar 不替代模型：

- **模型**负责在合法候选中表达语义偏好；
- **xgrammar**负责删除语法上不可能的候选。

### 3.2 两个阶段

xgrammar 的工作分为编译期和运行期。

#### 编译期：规则变成可执行状态

输入规则可以是：

- JSON Schema；
- 正则表达式；
- EBNF；
- Structural Tag（结构化标签）。

`GrammarCompiler` 结合 `TokenizerInfo` 将规则编译为 `CompiledGrammar`。编译结果不仅理解字符级语法，还建立语法与模型 token 词表之间的联系。

```text
JSON Schema / Regex / EBNF / Structural Tag
                     │
                     ▼
              GrammarCompiler
                     │ 结合 TokenizerInfo
                     ▼
              CompiledGrammar
```

编译可能比单步状态推进昂贵，因此 SGLang 会异步编译并缓存结果。

#### 运行期：逐 token 匹配

每个请求持有独立的 `GrammarMatcher`。即使两个请求共享同一个 `CompiledGrammar`，它们生成到的位置不同，运行状态也必须隔离。

```text
                     CompiledGrammar（可缓存、可共享）
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
       Request A: GrammarMatcher   Request B: GrammarMatcher
       已生成 {"age":             已生成 {"name":"A"
```

#### 实现原理速览

**CompiledGrammar：编译成下推自动机 + 预算好的词表掩码缓存。**

四类规则先被统一降解为字节级 EBNF/内部 IR，规则之间通过非终结符相互引用。由于 JSON、括号等结构可以任意嵌套，光靠有限状态机（FSA）无法记住“还差几个 `}`”，因此编译目标是**下推自动机（PDA）**：每条规则是一张字节级状态图，规则间的引用通过压栈/弹栈表达递归。

真正让 xgrammar 快的关键是 **adaptive token mask cache（自适应词表掩码缓存）**。编译期会把 tokenizer 的整个词表在每个语法状态节点上“预跑”一遍，把 token 分成两类：

- **上下文无关 token（context-independent）**：是否合法只取决于当前展开位置的状态节点，与栈上下文无关。绝大多数 token 属于此类，可在**编译期**直接算好并存成位掩码。
- **上下文相关 token（context-dependent）**：合法性依赖完整栈（例如某个 token 会跨越规则边界去闭合外层结构），只能在运行时结合当前栈判定。通常数量很少。

所以 `CompiledGrammar` = PDA 结构 + `TokenizerInfo` + 预计算掩码缓存，全部**只读**，可被同一 schema 的所有请求共享。这份预计算既是编译比单步推进昂贵的原因，也是必须异步化与缓存的原因。

**GrammarMatcher：在 PDA 上并行推进的一组栈。**

matcher 保存当前在 PDA 中的位置，用**栈**处理任意嵌套。但 token 与字符并非一一对齐、且语法可能有歧义（同一前缀可能对应多条解析路径），因此它维护的不是单个栈，而是**一组并行栈状态**（类似 NFA 的多状态并行）；只有当某 token 让所有分支都走不通时，才判定它非法。三个核心动作因此都很轻：

- `fill_next_token_bitmask`：先把上下文无关掩码**直接查表拷入**，再只对少量上下文相关 token 结合当前栈逐个判定后合并。成本约为 $O(\text{上下文相关 token 数})$，无需遍历整个词表。
- `accept_token`：用被采样的 token 推进所有活跃栈分支，并剪掉走不通的分支。
- `rollback` / jump-forward：栈用可回溯结构保存最近 $N$ 步（SGLang 设 `max_rollback_tokens=200`），回退即恢复历史快照；当所有活跃分支的下一段是同一确定字符串时，`find_jump_forward_string` 直接吐出该串，跳过模型前向。

一句话对照：**`CompiledGrammar` 承载“语言长什么样 + 词表大部分合法性”，是重、只读、可共享的；`GrammarMatcher` 承载“当前走到哪、还能往哪走”，是轻、可变、每请求独立的。**

#### CompiledGrammar 实例：`{"age": integer}` 的内部结构

下面用一个最小 JSON Schema 把三块结构拆开看。为聚焦内部结构，下列内容是概念示意，非 xgrammar 真实序列化格式。

```json
{
  "type": "object",
  "properties": {"age": {"type": "integer"}},
  "required": ["age"],
  "additionalProperties": false
}
```

它先被降解为等价的字节级 EBNF：

```ebnf
root    ::= "{" "\"age\"" ":" integer "}"
integer ::= "-"? digit+
digit   ::= [0-9]
```

**① PDA 结构**：每条规则编译成一张字节级状态图，规则引用（`root` 调用 `integer`）通过压栈/弹栈表达。状态节点与转移大致如下：

```text
S0  ──"{"──►  S1 ──"\"age\""──► S2 ──":"──► S3
                                             │  调用 integer：push(返回点=S4)
                                             ▼
                                      [integer 子图]
                            ┌── "-"? ──► D1 ──[0-9]──► D2 ⇄ [0-9]（循环）
                            └───────────────────────────► (数字读完) pop ─► S4
S4  ──"}"──►  S5(接受，可发 EOS)
```

- 栈内容记录“还欠哪些结构未闭合 + 子规则返回点”，例如读到 `{"age":1` 时栈约为 `[根对象未闭合, integer 返回点]`；
- 这正是 §4.2 所说的 PDA：`integer` 的 `digit+` 循环让长度无界，靠栈而非固定状态数来处理。

**② TokenizerInfo**：把字节级语法连接到具体模型的 token id 层。它保存词表、每个 token 解码出的字节串、以及权威 stop token（由模型 EOS 注入）。示意词表：

```text
token_id   文本片段     解码字节
--------------------------------
 101        {           7B
 102        "age"       22 61 67 65 22
 103        ":"         3A
 104        18          31 38
 105        }           7D
 106        true        74 72 75 65
 107        <eos>       （停止符）
```

没有它，语法只知道“下一字节该是 `0x7B`”，却不知道**哪个 token 解码后正好是 `{`**。

**③ 预计算掩码缓存（adaptive token mask cache）**：编译期遍历整个词表，在每个 PDA 状态节点上预跑，算出**上下文无关**部分并存成位掩码；少数**上下文相关** token 标记为“运行时再判”。

```text
PDA 状态   预计算允许（上下文无关）      标记为上下文相关
------------------------------------------------------------
S0         {101}                        —
S2         {103 ":"}                    —
S3/D*      {104 "18"、其他数字片段}      —（integer 内部）
S4         {105 "}"}                    —
S5         {107 <eos>}                  —
```

例如「`}` 之后能否直接发 `<eos>`」在只有单层对象时是确定的（上下文无关）；但在多层嵌套里，`}` 到底闭合哪一层、之后还能否再 `}`，要看栈上下文——这类判定就留给运行时的 `GrammarMatcher`。

合起来，这份 `CompiledGrammar` 只读且可被所有 `json_schema` 相同的请求共享；每个请求再各自 `copy()` 出轻量 `GrammarMatcher`，在同一份 PDA + 掩码缓存上独立推进自己的栈。

### 3.3 一步解码的完整数据流

```text
模型前向
  │
  ▼
logits [batch, vocab_size]
  │
  ├─ matcher.fill_next_token_bitmask(...)
  │        根据每个请求当前的语法状态填充 bitmask
  ▼
apply_token_bitmask_inplace(...)
  │        非法 token 的 logit → -inf
  ▼
温度 / top-k / top-p / softmax / 采样
  │
  ▼
next_token_id
  │
  └─ matcher.accept_token(next_token_id)
           推进语法状态，进入下一轮
```

抽象伪代码如下：

```text
while not matcher.is_terminated():
    logits = model.forward(input_ids)

    bitmask = allocate_token_bitmask(batch_size, vocab_size)
    matcher.fill_next_token_bitmask(bitmask, batch_index)
    apply_token_bitmask_inplace(logits, bitmask)

    next_token = sample(logits)
    matcher.accept_token(next_token)
    input_ids = next_token
```

### 3.4 为什么使用位掩码

若使用 `bool` 张量描述每个 token 是否可用，每个位置通常至少占 1 byte。xgrammar 使用 token bitmask，每个 token 只占 1 bit。

对于词表大小 \(V\)：

\[
\text{bitmask 大小} \approx \left\lceil\frac{V}{32}\right\rceil \times 4\text{ bytes}
\]

相较 1 byte/token 的 bool 表示，理论上约节省 8 倍；相较常见的 32-bit 整数逐 token 表示，则节省 32 倍。更重要的是，位掩码适合用 Triton/CUDA kernel 批量作用到 GPU logits 上。

在 SGLang 中：

- NVIDIA 等路径使用 `apply_token_bitmask_inplace_triton`；
- ROCm 路径使用 `apply_token_bitmask_inplace_cuda`；
- NPU 使用对应的 NPU 算子。

---

## 4. 逐 token 示例：生成一个 JSON 对象

假设输出必须满足：

```json
{
  "type": "object",
  "properties": {
    "age": {
      "type": "integer"
    }
  },
  "required": ["age"],
  "additionalProperties": false
}
```

期望结果之一是：

```json
{"age":18}
```

为便于理解，下面按“字符片段”展示。真实运行时单位是 tokenizer token，一个 token 可能覆盖多个字符。

| 步骤 | 已生成前缀 | 允许的下一类输出 | 被禁止的典型输出 |
| --- | --- | --- | --- |
| 0 | 空 | `{`、允许的前导空白 | `[`、普通文本、`"` |
| 1 | `{` | 空白、字段名 `"age"` | `"name"`、数字 |
| 2 | `{"age"` | 空白、`:` | `,`、`}` |
| 3 | `{"age":` | 空白、合法整数起始 | `"18"`、`true`、`[` |
| 4 | `{"age":1` | 下一位数字、空白、`}` | 字母、`[` |
| 5 | `{"age":18` | 下一位数字、空白、`}` | `"`、`true` |
| 6 | `{"age":18}` | EOS | 额外字段、解释文字 |

假设某一步模型给出下列 logits：

```text
当前前缀：{"age":

候选 token       原始 logit      是否符合 schema      mask 后
------------------------------------------------------------
"18"               8.2              是               8.2
"unknown"          9.5              否              -inf
"true"             7.4              否              -inf
"null"             6.9              否              -inf
" 18"              6.5              是               6.5
```

即使模型最偏好 `"unknown"`，它也无法被采样，因为其概率被强制归零。约束不会让模型“更懂年龄”，但能确保它只能从满足当前语法和 schema 的候选中选择。JSON Schema 各关键字的实际支持范围取决于 xgrammar 版本；年龄上下界等业务约束仍应由服务端复验。

### 4.1 为什么不是简单的字符串白名单

假设词表包含这些 token：

```text
"{"       "{\"age\":"       "18"       "18}"       "}"       "age"
```

在起始状态，`"{"` 和 `"{\"age\":"` 都可能合法；生成 `{"age":` 后，`"18"` 与 `"18}"` 都可能合法。xgrammar 需要判断完整 token 消费后是否仍存在合法语法路径，而不只是检查 token 的第一个字符。

### 4.2 状态为什么需要栈语义

JSON 可以任意嵌套：

```json
{"user":{"name":"Alice","tags":["a","b"]}}
```

解析器必须记住当前处于：对象 → 对象 → 数组，并在遇到 `]`、`}` 时按相反顺序闭合。仅靠固定数量状态的简单规则很难表达任意深度嵌套；这也是上下文无关文法及其栈式匹配能力比普通正则/FSM 更适合复杂嵌套结构的原因。

#### 什么是 PDA（下推自动机）

前面多次提到 xgrammar 把语法编译成 **PDA（Pushdown Automaton，下推自动机）**。一句话：**PDA = 有限状态机（FSM）+ 一个栈。**

先看 FSM 的局限。有限状态机只有固定数量的状态、没有额外记忆，只能识别正则语言，例如匹配 `ORD-[0-9]{8}` 毫无问题。但它**记不住计数**——无法判断「左右括号是否一一配对」，因为嵌套深度可以无限大，而状态数是有限的。JSON 恰恰要求这种配对记忆：到底欠几个 `}`，取决于当前嵌套多深，FSM 存不下这个「欠账数」，PDA 用**栈**来存。

PDA 有三个部件，每读入一个符号（字符/token），根据「当前状态 + 栈顶」决定：转到哪个状态，以及**压栈还是弹栈**：

```text
   输入： { " a " : { " b " ... }   ← 逐个符号读入
            │
   ┌────────▼────────┐
   │   有限状态控制    │  ← 当前在读键？读值？
   └────────┬────────┘
            │ 读到 { 就 push，读到 } 就 pop
   ┌────────▼────────┐
   │       栈         │  ← 记住「还没闭合的结构」
   │   [ obj, obj ]   │
   └─────────────────┘
```

- 遇到 `{`：把「等待一个 `}`」**压栈**；
- 遇到 `}`：检查栈顶确实是对应的对象，然后**弹栈**；
- 读完输入且**栈为空** → 结构完整合法。

栈可以无限增长，所以任意深度嵌套都能表达。这类语言称为「上下文无关语言（CFL）」，正是 EBNF/JSON 所属的层级。映射到 xgrammar：

| PDA 部件 | 在 xgrammar 中的对应 |
| --- | --- |
| 有限状态控制 | 当前语法规则展开到哪一步 |
| 栈 | `GrammarMatcher` 里那个（组）栈——记住外层还欠哪些结构未闭合 |
| push | 进入 `{`、`[` 或子规则时压入 |
| pop | 闭合 `}`、`]` 时弹出 |
| 「下一步能读哪些符号」 | 就是生成 token bitmask 的依据 |

一句话记忆：**FSM 管「顺序」，PDA 在此之上多管「配对/嵌套」——多出来的能力全部来自那个栈。** 这也解释了为什么 §3.2 说编译目标是 PDA、而 `GrammarMatcher` 的核心状态是一组栈。

---

## 5. SGLang 中的完整执行链路

### 5.1 启动阶段

SGLang 默认选择 xgrammar：

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --grammar-backend xgrammar
```

若不显式传 `--grammar-backend`，当前 `ServerArgs` 也会将其设为 `xgrammar`。初始化链路如下：

```text
ServerArgs.grammar_backend = "xgrammar"
                │
                ▼
create_grammar_backend(...)
                │
                ▼
XGrammarGrammarBackend(tokenizer, vocab_size, eos_token_ids)
                │
                ├─ TokenizerInfo.from_huggingface(...)
                └─ GrammarCompiler(tokenizer_info)
```

`TokenizerInfo` 很关键：语法定义在文本层，而 mask 必须作用于 token id 层，二者需要通过具体模型的 tokenizer 连接。

### 5.2 请求进入 Scheduler

请求可携带四种约束之一：

```text
sampling_params.json_schema
sampling_params.regex
sampling_params.ebnf
sampling_params.structural_tag
```

`GrammarManager.process_req_with_grammar` 按优先级构造缓存键：

```text
("json", schema_string)
("regex", regex_string)
("ebnf", ebnf_string)
("structural_tag", structural_tag_string)
```

随后发生两种情况：

- **缓存命中**：复制一份 grammar matcher 状态，直接进入等待队列；
- **缓存未命中**：在线程池异步编译，请求暂存于 `grammar_queue`，编译完成后再进入调度队列。

为什么缓存的是编译结果而不是共享运行状态？因为规则可以共享，但每个请求的生成前缀不同。`copy()` 会基于同一 `CompiledGrammar` 创建新的 `GrammarMatcher`。

### 5.3 编译分派

`XGrammarGrammarBackend` 根据类型调用不同入口：

| 请求类型 | xgrammar 编译入口 |
| --- | --- |
| 任意合法 JSON | `compile_builtin_json_grammar()` |
| JSON Schema | `compile_json_schema(...)` |
| EBNF | `compile_grammar(...)` |
| Regex | `compile_regex(...)` |
| Structural Tag | `compile_structural_tag(...)` |

编译错误会变成 `InvalidGrammarObject`，请求被明确终止，而不是静默退化成无约束生成。

### 5.4 解码阶段

每轮 decode 时，SGLang 将同一 batch 内不同请求的语法 mask 填入对应行：

```text
vocab_mask
┌───────────────────────────────────────────┐
│ request 0：JSON schema 当前状态的允许集合   │
│ request 1：无约束或其他约束                 │
│ request 2：EBNF 当前状态的允许集合          │
└───────────────────────────────────────────┘
```

mask 被搬到设备端并原地作用于 logits。采样成功后，`accept_token` 更新对应请求的 matcher。请求完成、抢占或投机验证失败时，语法状态还必须与 token 序列保持一致。

---

## 6. 四类约束及示例

下面示例假设 SGLang 服务运行在 `http://localhost:30000`。

### 6.1 JSON Schema：约束业务对象

适用于 API 返回对象、表单数据、工具参数等“字段和类型已知”的场景。

```python
import json
import requests

schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "temperature": {"type": "number"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    "required": ["city", "temperature", "unit"],
    "additionalProperties": False,
}

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Return the weather of Beijing as JSON.",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 128,
            "json_schema": json.dumps(schema),
        },
    },
)

print(response.json()["text"])
```

可能输出：

```json
{"city":"Beijing","temperature":26,"unit":"celsius"}
```

JSON Schema 保证的是结构属性，例如：

- 输出是对象而不是数组；
- 必须包含三个 required 字段；
- `temperature` 必须是 number；
- `unit` 只能二选一；
- 不允许额外字段。

它不保证天气数值真实。真实性仍取决于模型知识、上下文或外部工具。

### 6.2 Regex：约束扁平文本格式

适用于电话号码、ID、日期、固定枚举等不需要递归嵌套的格式。

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Generate one order ID.",
        "sampling_params": {
            "temperature": 0.7,
            "max_new_tokens": 32,
            "regex": r"ORD-[0-9]{8}",
        },
    },
)

print(response.json()["text"])
```

合法输出：

```text
ORD-20260719
```

非法输出包括：

```text
order-20260719
ORD-123
ORD-ABCDEFGH
```

Regex 擅长扁平模式，但不适合表达任意深度的括号嵌套。需要递归结构时应优先考虑 EBNF 或 JSON Schema。

### 6.3 EBNF：直接描述一门输出语言

#### 6.3.1 第一性原理降维：生成约束本质是“前缀能否完成”

EBNF（Extended Backus–Naur Form，扩展巴科斯范式）不是输出模板，也不是给模型看的提示词。它是一套**有限规则对无限字符串集合的递归定义**：

\[
G=(N,\Sigma,P,S)
\]

| 符号 | 含义 | 例子 |
| --- | --- | --- |
| \(N\) | 非终结符：可继续展开的抽象结构 | `expr`、`number`、`pair` |
| \(\Sigma\) | 终结符：最终实际输出的字符 | `"{"`、`":"`、数字 |
| \(P\) | 产生式：结构如何展开 | `pair ::= key ":" value` |
| \(S\) | 起始符号：整门语言的入口 | `root` |

约束解码面对的真正问题不是“当前文本是否合法”，而是：

> 对当前前缀 \(p\) 和候选 token \(t\)，是否至少存在某个后缀 \(s\)，使 \(p+t+s\in L(G)\)？

如果不存在这样的后缀，选择 \(t\) 就会进入**永远无法修复的死前缀**，其 logit 必须立即置为 \(-\infty\)。因此 EBNF 在 xgrammar 中的作用，是定义语言 \(L(G)\)；`GrammarMatcher` 则持续回答“哪个 token 仍能通向 \(L(G)\) 中的某个完整句子”。

EBNF 常见构造可以压缩为：

| 构造 | 语义 | 示例 |
| --- | --- | --- |
| 顺序 | A 后必须接 B | `A B` |
| 选择 | A、B 二选一 | `A \| B` |
| 重复 | 同一结构出现多次 | `[0-9]+` 或规则递归 |
| 可选 | 结构可出现也可不出现 | 具体写法取决于 grammar 方言 |
| 分组/引用 | 组合并复用子结构 | `pair ::= key ":" value` |
| 递归 | 结构内部再次包含自身 | 嵌套数组、括号表达式 |

> 不同 EBNF 实现的表面语法并不完全统一。SGLang 把字符串交给 xgrammar 的 `compile_grammar`，应以当前 xgrammar 支持的 grammar 语法为准，而不能假定任意 EBNF 方言都可直接互换。

#### 6.3.2 高阶结构化类比：它对应编译器前端，而不是字符串过滤器

EBNF 与成熟编译器前端具有同构关系：

```text
传统编译器
源代码 ─► 词法单元 ─► 文法解析器 ─► 语法树 / 接受或拒绝
                       ▲
                       │ EBNF/CFG 描述合法程序

约束解码
输出前缀 ─► tokenizer token ─► GrammarMatcher ─► 下一步 token mask
                                ▲
                                │ EBNF 描述合法输出
```

两者拓扑相同，时间方向相反：

- 编译器是**判定式**：完整 token 流已经存在，解析器判断它是否属于语言；
- 约束解码是**生成式**：文本尚未产生，解析器反向暴露所有仍可接受的下一 token。

递归语法需要记住尚未闭合的层级，其结构与下推自动机（PDA）的栈相似。例如解析：

```json
{"users":[{"name":"Alice"}]}
```

状态必须记住尚待闭合的结构：

```text
读取 {       栈：[object]
读取 [       栈：[object, array]
读取 {       栈：[object, array, object]
读取 }       栈：[object, array]
读取 ]       栈：[object]
读取 }       栈：[]
```

这不是“数括号”技巧，而是递归语言的结构记忆：结束符必须与最近尚未闭合的开始符配对。

#### 6.3.3 示例一：有限选择

下面的 grammar 把完整输出语言限定为三个字符串：

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")

grammar = r'''
root ::= "Hello" | "Hi" | "Hey"
'''

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Say one short greeting."}],
    temperature=0,
    max_tokens=16,
    extra_body={"ebnf": grammar},
)

print(response.choices[0].message.content)
```

此时模型仍可以在 `Hello`、`Hi`、`Hey` 中按概率选择，但任何第四种文本都没有可达路径。

#### 6.3.4 示例二：分层组合

只允许 `{"name":"Alice"}` 这种单字段 JSON 形状，其中值只能包含英文字母：

```ebnf
root    ::= "{" pair "}"
pair    ::= "\"name\"" ":" string
string  ::= "\"" [A-Za-z]+ "\""
```

合法输出：

```json
{"name":"Alice"}
```

非法输出：

```text
{"name":123}
{"name":"Alice","age":18}
{"other":"Alice"}
```

按产生式展开，它等价于：

```text
root
  └─ "{" + pair + "}"
              └─ "name" + ":" + string
                                  └─ "\"" + letters + "\""
```

规则复用比把整个结构平铺成一条巨大正则更接近语言本身的层级。

#### 6.3.5 边界与对立面

EBNF、Regex、JSON Schema 的分界不在“谁更高级”，而在它们描述的对象不同：

| 方案 | 直接描述什么 | 强项 | 天然弱项 |
| --- | --- | --- | --- |
| Regex | 字符串局部模式 | ID、日期、固定前后缀、扁平格式 | 任意深度递归结构 |
| EBNF | 一门语言的生成规则 | DSL、表达式、嵌套结构、严格序列 | 业务字段语义表达繁琐 |
| JSON Schema | JSON 数据模型 | 字段、类型、必填项、枚举、对象/数组约束 | 非 JSON 语言与任意协议外壳 |

能力边界可以用形式语言层级理解：有限状态机适合正则语言；当语言要求无界的配对记忆，例如 \(n\) 个左括号必须对应 \(n\) 个右括号时，需要栈式状态。工程实现会做大量优化，但“是否需要无界结构记忆”仍是最核心的分界线。

极端情况：

- **语言为空**：所有分支互相矛盾，起始状态就没有可接受 token，生成无法启动；
- **只有一个句子**：每一步几乎都唯一确定，模型前向的信息增益趋近于零，应考虑 jump-forward；
- **语言接近全集**：几乎所有 token 都合法，约束退化为空操作，却仍支付匹配成本；
- **无终止递归**：每个前缀都可能合法，但模型可以永不完成，仍必须用 `max_new_tokens` 限制；
- **二义性**：同一前缀可由多条产生式解释。只要存在合法路径就可继续，但状态集合与匹配成本可能膨胀。

典型反模式：

1. **把 EBNF 当业务规则引擎**：用文法编码库存、权限、实时价格等外部状态；这些不是字符串语言属性，应由业务层验证；
2. **把动态数据全展开成文法**：将十万个商品 ID 写成十万个选择分支，导致编译、缓存和匹配成本失控；
3. **写出只能生成、不能结束的递归**：结构始终有下一步，却没有终止产生式；
4. **忽略 tokenizer 边界**：以为 grammar 每次只消费一个字符，实际 token 可能跨越多个终结符；
5. **为 JSON 手写巨型 EBNF**：若需求本质是字段/类型约束，JSON Schema 通常更短、更可维护。

#### 6.3.6 动机演进与语义压缩

演进路线是：

```text
硬编码字符串模板
  └─ 只能表达一个固定句子
       ↓
Regex
  └─ 能表达大量扁平模式，但缺少递归结构记忆
       ↓
BNF
  └─ 用产生式与递归定义上下文无关语言
       ↓
EBNF
  └─ 加入更紧凑的选择、重复、可选等表达方式
       ↓
tokenizer-aware 约束解码
  └─ 不再等文本生成后判错，而是在生成途中阻断死前缀
```

**无行业黑话的一句话**：

> EBNF 用少量可递归规则划出所有允许写出的句子，任何无法走到完整句子的下一步都被禁止。

### 6.4 Structural Tag：受控地切换输出协议

#### 6.4.1 第一性原理降维

JSON Schema 只解决“**参数体内部是否合法**”，不能解决工具调用更底层的三个问题：

1. 当前输出是普通正文，还是可执行的工具调用？
2. 如果是工具调用，具体选择了哪个工具？
3. 参数从哪里开始、在哪里结束，之后是否还能继续正文或下一个调用？

如果没有明确边界，下面这段 JSON 可能是回答中的示例，也可能是要执行的参数，应用层无法仅凭内容判定：

```json
{"city":"Beijing"}
```

Structural Tag 的最小单元不是“标签”，而是一个三元组：

$$
\mathcal{T}_i = \left(b_i,\; L_i,\; e_i\right)
$$

其中：

- $b_i$ 是第 $i$ 种结构的开始序列；
- $L_i$ 是结构体内部允许的语言，工具调用中通常由 JSON Schema 定义；
- $e_i$ 是结束序列。

一个完整结构片段属于：

$$
L\left(\mathcal{T}_i\right)=\left\{b_i x e_i \mid x\in L_i\right\}
$$

但仅有三元组还不够。系统还需要一个**外层控制器**决定：

- 哪些前缀会触发结构分支；
- 多个结构之间是顺序、选择还是重复；
- 是否至少出现一个结构；
- 多个调用之间使用什么分隔符。

所以 Structural Tag 本质上是两层约束：

```text
外层协议状态机：正文 / 触发 / 选工具 / 多调用 / 结束
                           │
                           ▼
内层载荷状态机：当前工具参数必须满足对应 JSON Schema
```

对第 $t$ 步的状态 $q_t$，允许 token 集合为：

$$
A(q_t)=
\begin{cases}
A_{\text{outer}}(q_t), & q_t\text{ 位于外层协议状态},\\
A_{\text{schema}_i}(q_t), & q_t\text{ 位于工具 }i\text{ 的参数体},\\
A_{\text{end}_i}(q_t), & q_t\text{ 正在闭合工具 }i.
\end{cases}
$$

最终 mask 仍遵循统一规则：

$$
z'_{t,j}=
\begin{cases}
z_{t,j}, & j\in A(q_t),\\
-\infty, & j\notin A(q_t).
\end{cases}
$$

这解释了为什么 Structural Tag 不是普通 XML 标签：它决定的是**何时切换约束器，以及切换到哪一个约束器**。

##### 逐 token 状态示例

假设模型原生格式为：

```text
<tool_call>get_weather
{"city":"Beijing"}</tool_call>
```

对应结构为：

```json
{
  "begin": "<tool_call>get_weather\n",
  "schema": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": false
  },
  "end": "</tool_call>"
}
```

生成过程中的约束切换如下：

| 已生成前缀 | 当前状态 | 下一步约束来源 | 典型禁止项 |
| --- | --- | --- | --- |
| 空或普通正文 | `FREE_TEXT` | 外层协议：继续正文或进入合法 trigger | 非法的半截控制标记 |
| `<tool_call>` | `SELECT_STRUCTURE` | 只能选择已声明工具对应的 begin | 未声明工具名 |
| `<tool_call>get_weather\n` | `BODY(get_weather)` | `get_weather.parameters` JSON Schema | 错误类型、额外字段；必填字段缺失时禁止提前闭合 |
| `...{"city":"Beijing"}` | `CLOSE(get_weather)` | 固定结束序列 | 普通正文、错误结束标记 |
| `...</tool_call>` | `AFTER_STRUCTURE` | 结束、恢复正文或进入下一调用，由外层格式决定 | 不符合外层组合规则的分支 |

关键点是：**同一个 token 是否允许，取决于当前处于哪一层状态**。例如 `{` 在正文中只是普通字符；进入 `BODY(get_weather)` 后，它成为 JSON 参数的起始符，并受该工具 schema 控制。

#### 6.4.2 高阶结构化类比

Structural Tag 与网络协议的“**多路复用器 + 分帧器 + 载荷解码器**”同构：

```text
网络协议栈
字节流 ─► 识别帧边界 ─► 判断消息类型 ─► 按对应 payload 协议解码

Structural Tag
文本流 ─► 识别 trigger ─► 判断工具类型 ─► 按对应 JSON Schema 约束参数
```

映射关系：

| Structural Tag | 网络协议结构 | 共同机制 |
| --- | --- | --- |
| `trigger` | 同步字 / magic number | 从普通流切入受控帧 |
| `begin` | 帧头 + 消息类型 | 确定后续使用哪个子协议 |
| `schema` | payload 编码规则 | 约束载荷内部结构 |
| `end` | 帧边界 | 确定载荷已完整结束 |
| 多个 structure | 多种消息类型 | 根据类型切换不同解码规则 |
| separator / repetition | 连续帧协议 | 规定多个消息如何排列 |

真正重要的相似拓扑是：

$$
\text{控制平面选择协议}\quad+\quad\text{数据平面按所选协议处理载荷}
$$

Structural Tag 的外层格式是控制平面，内部 JSON Schema 是数据平面。它不像 CRC，不负责检测传输错误；也不像工具执行器，不负责验证权限或产生业务结果。

在 SGLang 中还可以类比为**嵌套状态机的动态分派**：

```text
OuterMatcher
   ├─ 普通文本状态
   ├─ 命中 get_weather begin ─► JsonSchemaMatcher(weather_schema)
   ├─ 命中 search begin      ─► JsonSchemaMatcher(search_schema)
   └─ 参数结束               ─► 回到 OuterMatcher
```

#### 6.4.3 边界与对立面限定

##### 与邻近概念的绝对边界

| 概念 | 它回答的问题 | 工作时机 | Structural Tag 不替代它的原因 |
| --- | --- | --- | --- |
| JSON Schema | 一个 JSON 值内部是否合法？ | 生成中 | 不描述正文与参数体的切换边界 |
| EBNF | 整门字符语言如何生成？ | 生成中 | 更通用，但不直接承载工具、trigger、schema 等领域语义 |
| Chat Template | 模型输入和历史消息如何编码？ | 生成前 | 它提供模型熟悉的格式，不保证模型输出严格遵守 |
| Structural Tag | 何时进入哪个结构化子协议？ | 生成中 | 只负责协议路径与参数结构 |
| Tool-call Parser | 已生成文本如何拆成工具名和参数？ | 生成后/流式增量 | 它不能回到过去阻止非法 token |
| Tool Executor | 工具是否允许执行、执行结果是什么？ | 解析后 | 结构合法不等于安全或业务合法 |

理论上可用一份巨大 EBNF 重写 Structural Tag，但这是“表达能力相同”而不是“抽象相同”。Structural Tag 把常见模式直接提升为领域对象：

$$
\text{trigger}\rightarrow\text{tool selection}\rightarrow
\text{schema body}\rightarrow\text{end}\rightarrow\text{next state}
$$

这样工具定义可以直接映射到约束，而不必为每个模型手工拼接整份 EBNF。

##### `auto`、`required` 与 `strict` 的正交关系

| 配置 | 约束维度 | 语义 |
| --- | --- | --- |
| `tool_choice="auto"` | 是否进入工具分支 | 可以不调用；若触发调用，则按结构规则生成 |
| `tool_choice="required"` | 工具调用次数下界 | 至少进入一次工具分支，通常对应 `at_least_one=True` |
| 指定工具 | 工具选择集合 | 目标语义是限制到指定工具；具体约束形态取决于 detector 与回退路径 |
| `strict=True` | 参数体语言 | 使用工具声明的 parameters schema |
| 非 strict | 参数体语言 | schema 可退化为 `{}`，主要保证外壳可解析 |

因此：

- `required` 约束“**必须调用**”，不等于参数严格；
- `strict` 约束“**调用后参数必须合法**”，不等于必须调用；
- `auto + strict` 表示“可以不调用；一旦调用，参数必须符合 schema”。

##### 极限边界

- **零个 structure**：若还要求 `at_least_one=True`，合法路径为空；
- **唯一 structure + 唯一参数值**：输出几乎完全确定，模型前向信息增益趋近于零；
- **trigger 是普通高频前缀**：正文可能意外进入结构状态，破坏自然语言输出；
- **begin/end 互为前缀或可在 payload 中无转义出现**：协议边界产生歧义；
- **多个 structure 拥有相同 begin**：直到更长前缀出现前都无法确定子协议，状态分支膨胀；
- **并行调用无 separator/repetition 规则**：单次调用合法，但调用序列无从判定；
- **schema 无可满足实例**：已经进入工具分支，却不存在任何合法参数体，生成陷入死状态。

##### 典型反模式

1. **手写与模型训练格式不一致的标签**：结构可以被强制生成，但模型在合法分支中的工具选择和参数质量可能显著下降；
2. **用普通自然语言作为 trigger**：高频前缀误触发状态切换；
3. **只有 parser，没有生成约束**：只能事后发现标签或 JSON 损坏；
4. **只有约束，没有 parser**：得到合法的模型原生文本，却无法转成统一 `tool_calls[]`；
5. **把 schema 合法当成可安全执行**：攻击性字符串也可能完全符合 `type: string`，执行前仍需权限、注入、范围和业务校验；
6. **把 Structural Tag 当通用 XML 校验**：它关注协议切换与载荷约束，不提供 XML 命名空间、属性语义或 DOM 验证；
7. **错误组合 `required` 与空/冲突 schema**：所有正常结束路径都被封死，最终不存在合法下一 token。

#### 6.4.4 语义压缩与动机演进

##### 为什么会演进出 Structural Tag

```text
自然语言描述“请调用某工具”
  └─ 工具名和参数无法稳定提取
       ↓
只输出一个 JSON
  └─ 参数可解析，但无法区分正文、代码示例和真实调用
       ↓
增加 begin/end 标记
  └─ 能分帧，但内部参数仍可能缺字段或类型错误
       ↓
标签 + JSON Schema
  └─ 单次调用可解析且参数结构合法
       ↓
Structural Tag 外层组合
  └─ 支持 auto/required、多工具、重复调用、分隔符和模型原生格式
       ↓
Structural Tag + Tool-call Parser
  └─ 生成中阻断非法路径，生成后转换为统一 API 对象
```

它推翻的旧假设是：

> 先让模型自由输出，再靠正则或 parser 猜测哪一段是工具调用。

新的假设是：

> 在每个 token 产生前，系统就维护“当前处于正文、哪个工具参数体或结束阶段”的确定状态。

##### SGLang 中的实际落点

SGLang 支持两类输入：

- legacy：`structures + triggers + at_least_one`；
- xgrammar 新格式：`tag`、`sequence`、`or`、`triggered_tags`、`tags_with_separator` 等组合结构。

`XGrammarGrammarBackend.dispatch_structural_tag` 将 legacy 格式转换为 xgrammar `StructuralTag`，或直接编译新格式，最终都进入 `compile_structural_tag`。

工具调用时通常不需要用户手写 Structural Tag。`FunctionCallParser` 会结合：

- `tools` 与每个工具的 parameters；
- `tool_choice`；
- `strict` / `ToolStrictLevel`；
- 模型对应的 detector 与原生标签格式；

优先生成模型原生 Structural Tag，不支持时回退到 legacy Structural Tag；必须调用但无法使用结构标签时，还可能回退到 JSON Schema 约束。不同路径对“指定工具”和并行调用的编码方式并不完全相同，应以 detector 产出的实际约束为准。

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "北京天气怎么样？"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询城市天气",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    tool_choice="required",
)

print(response.choices[0].message.tool_calls)
```

端到端职责链：

```text
tools / tool_choice / strict
            │
            ▼
FunctionCallParser + model detector
  生成模型原生或 legacy Structural Tag
            │
            ▼
xgrammar.compile_structural_tag
  编译外层协议和内层 schema
            │
            ▼
逐 token mask
  生成阶段阻断非法协议路径
            │
            ▼
模型原生 tool-call 文本
            │
            ▼
detector.parse_*
  解析为 normal_text + ToolCallItem
            │
            ▼
OpenAI tool_calls[]
            │
            ▼
业务层重新校验并执行工具
```

**无行业黑话的一句话**：

> Structural Tag 让系统在模型写每个词时，都明确它正在写普通回答、哪个工具的参数，还是工具调用的结束标记。

---

## 7. 高级机制

### 7.1 Rollback：让语法状态跟随 token 回退

普通自回归解码每轮接受一个 token，状态只向前走。但投机解码会先提出多个 draft token，再由主模型验证；验证失败的后缀必须撤销。

```text
原状态 S0
  ├─ accept A → S1
  ├─ accept B → S2
  └─ accept C → S3

验证结果：只接受 A，拒绝 B/C

rollback(2)
  └─ S3 → S1
```

`XGrammarGrammar.rollback(k)` 同时回退 matcher 和已接受 token 记录。SGLang 当前创建 matcher 时设置的最大回退 token 数为 200。

如果只回退模型 token、没有回退 grammar 状态，下一轮 mask 会基于不存在的前缀计算，轻则错误屏蔽合法 token，重则使生成无路可走。

### 7.2 Jump-Forward：确定内容不必让模型逐字生成

若当前语法状态后只有唯一固定文本，例如 JSON Schema 决定下一个字段名只能是：

```text
,"temperature":
```

那么逐 token 调模型没有信息增益，因为模型没有选择空间。xgrammar 可通过 `find_jump_forward_string()` 找出确定片段并直接跳过。

类比填写固定模板：

```text
姓名：[模型填写]
固定标签“年龄：”无需模型逐字抄写
年龄：[模型填写]
```

跳跃后需要重新 tokenize，因为“旧输出 token + 新字符串”的整体分词边界可能变化：

```text
旧 token 边界： ["age"] [:]
拼接后重分词： ["age":]
```

SGLang 的 `jump_and_retokenize` 会找到新旧 token 序列的最长共同前缀，回退分歧部分，再按新 token 序列重新推进 matcher。

### 7.3 编译缓存与请求状态隔离

相同 schema 可能被大量请求复用。正确设计是：

```text
共享：CompiledGrammar（昂贵、只读）
隔离：GrammarMatcher（便宜、每请求可变）
```

这与数据库“共享查询计划、每次执行有独立游标”类似。若多个请求共享同一个 matcher，一个请求生成 `{` 后会污染另一个请求的状态。

### 7.4 Reasoning 模型：思考阶段与答案阶段使用不同规则

对 DeepSeek-R1、Qwen3 等 reasoning 模型，如果最终答案要求 JSON，不能从第一个思考 token 开始就强制 JSON，否则模型无法自由生成思维过程。

`ReasonerGrammarBackend` 用两阶段状态机包装真实 grammar：

```text
THINKING
  - 通常不应用最终答案 grammar
  - 允许自由推理
  - 遇到 think_end_id
          │
          ▼
GENERATION
  - 开始调用 xgrammar
  - 最终答案必须满足 JSON/EBNF/Regex/Structural Tag
```

严格思考模式还可以：

- 在思考阶段排除某些 token；
- 设置 `thinking_budget`；
- budget 用尽时只允许 `think_end_id`，强制退出思考。

---

## 8. 对立面、边界与反模式

理解 xgrammar 最快的方法，是明确它不是什么。

### 8.1 xgrammar 与 prompt 指令

| 维度 | Prompt：“请只输出 JSON” | xgrammar |
| --- | --- | --- |
| 约束性质 | 软约束 | 硬约束 |
| 实现位置 | 模型上下文 | 采样前的 logits |
| 是否可能违反 | 可能 | 语法路径正确时不会选择非法 token |
| 是否消耗上下文 token | 是 | grammar 本身不必放入 prompt |
| 是否影响语义理解 | 可以通过说明影响模型 | 只删除非法候选，不提供事实知识 |

最佳实践通常不是二选一：prompt 告诉模型“应该表达什么”，xgrammar 保证“必须以什么形式表达”。

### 8.2 xgrammar 与事后 JSON 校验

| 维度 | 事后校验 | xgrammar |
| --- | --- | --- |
| 发生时间 | 完整输出之后 | 每个 token 采样之前 |
| 失败处理 | 修复或重试 | 非法路径不会被采样 |
| 流式输出 | 可能流到一半才发现非法 | 每个前缀都仍存在合法完成路径 |
| 业务语义校验 | 可以做 | 不是主要职责 |

二者仍可组合：xgrammar 保证结构，业务校验器检查数据库约束、权限、事实一致性等无法仅靠语法表达的规则。

### 8.3 xgrammar 与 tool-call parser

这是最容易混淆的一组概念：

```text
xgrammar：生成前/生成中，限制哪些 token 能出现
parser：  生成后，把模型文本拆成 name、arguments 等字段
```

只有 parser 没有 grammar，格式可能生成失败；只有 grammar 没有 parser，得到的仍是模型原生标记文本，应用层还没有结构化 `tool_calls[]`。

### 8.4 它不能保证什么

xgrammar 能保证“形式合法”，不能自动保证：

1. **事实正确**：`{"capital":"London"}` 结构合法，但法国首都答案错误；
2. **工具存在**：除非 schema/结构标签已限制工具名；
3. **工具执行成功**：网络、权限和业务异常属于执行层；
4. **跨系统业务一致性**：例如库存必须大于下单量；
5. **高质量文本**：语法越窄，模型可选空间越小，但内容不必更聪明。

### 8.5 边界条件

#### 合法候选趋近于整个词表

若 grammar 在当前状态允许几乎所有 token，mask 接近空操作，但仍有状态计算和 mask 应用开销。此时约束收益很小。

#### 合法候选只有一个

模型失去选择权。继续执行完整前向只是在重复确认一个已知答案，适合尝试 jump-forward。

#### 合法候选为空

表示当前前缀、grammar、tokenizer 或停止条件之间出现矛盾。系统无法通过调高温度解决，因为所有概率都是 0。应检查：

- grammar 是否可满足；
- EOS/stop token 是否配置一致；
- tokenizer 是否受支持；
- schema 是否把所有分支都封死；
- 是否错误复用了其他请求的 matcher 状态。

#### grammar 允许无限输出

例如 EBNF 中存在无终止递归，或者规则允许字符串无限增长。xgrammar 只能保证前缀合法，仍需要 `max_new_tokens` 等外部上限保证请求最终结束。

#### schema 极大或频繁变化

缓存命中率降低，编译延迟和缓存占用上升。不要把每个请求的随机值都写进一个全新 schema；能作为 prompt 数据表达的内容，不应全部固化成语法规则。

### 8.6 典型反模式

#### 反模式一：用超大枚举承载动态业务数据

```json
{"enum": ["十万个动态商品 ID……"]}
```

可能导致 grammar 编译和状态匹配成本过高。更合理的做法是让模型生成较小的检索条件，再由业务系统查询和校验。

#### 反模式二：以为约束越严，答案越准确

过严的 grammar 可能删除语义正确但未被规则覆盖的答案。例如只允许 `yes|no`，却忽略“信息不足”。最终输出一定合法，但可能被迫错误二选一。

#### 反模式三：同时给出互相冲突的 prompt 和 grammar

Prompt 要求自然语言解释，grammar 只允许 JSON。模型会尽力表达解释，但所有相关 token 都被屏蔽，结果可能结构合法却内容怪异。

#### 反模式四：手工解析 token 字符串代替 tokenizer-aware matcher

直接按字符检查会忽略 token 跨字符、前导空格、UTF-8 字节和特殊 token 边界，容易误封合法 token 或放行非法 token。

#### 反模式五：认为结构约束可以替代输入与输出校验

Grammar 是输出生成层的防线，不是权限、业务规则和安全审计的替代品。工具参数在真正执行前仍需按服务端信任边界重新校验。

---

## 9. 性能成本与适用场景

### 9.1 成本分解

xgrammar 的额外成本主要来自：

1. **一次性编译**：规则 + tokenizer → `CompiledGrammar`；
2. **每步状态查询**：计算当前允许 token；
3. **每步 mask 填充与传输**；
4. **GPU mask kernel**：将非法 logits 置为 \(-\infty\)；
5. **状态推进**：接受采样 token；
6. **可选重分词**：jump-forward 后修正 token 边界。

收益来自：

- 避免格式错误后的重试；
- 减少 JSON 修复逻辑；
- jump-forward 跳过确定片段；
- 工具调用参数更稳定；
- 流式输出始终保持“仍可完成为合法结果”的前缀。

### 9.2 何时值得使用

推荐：

- JSON API 响应；
- tool/function calling；
- SQL/DSL/配置片段；
- ID、日期、枚举等固定格式；
- 下游解析失败代价高的自动化流程。

谨慎使用：

- 自由写作、开放问答；
- schema 每次都巨大且完全不同；
- 约束无法准确覆盖真实答案空间；
- 只需简单后处理即可稳定解决的低风险场景。

### 9.3 与其他后端的边界

| 后端 | 核心表示 | 适合场景 | 主要特点 |
| --- | --- | --- | --- |
| xgrammar | 编译语法 + tokenizer-aware matcher + bitmask | JSON、EBNF、Regex、Structural Tag | SGLang 默认；支持 rollback、jump-forward |
| outlines | Regex/FSM + bool mask | 以正则为主的约束 | 模型简单，但递归表达能力弱 |
| llguidance | LLMatcher + bitmask | JSON、Regex、结构化标签 | 可选高性能后端，支持 rollback |

三者在 SGLang 中服从同一个抽象：`allocate_vocab_mask → fill_vocab_mask → apply_vocab_mask → accept_token`。差别主要在规则如何编译、状态如何表示、mask 如何生成，以及对 rollback/jump-forward 的支持。

---

## 10. 源码阅读地图

| 文件 | 阅读重点 |
| --- | --- |
| `python/sglang/srt/constrained/base_grammar_backend.py` | 统一接口、异步编译线程池、缓存、后端创建 |
| `python/sglang/srt/constrained/xgrammar_backend.py` | `GrammarCompiler`、`GrammarMatcher`、bitmask、rollback、jump-forward |
| `python/sglang/srt/constrained/grammar_manager.py` | 请求如何进入 grammar queue、跨 rank 同步编译状态、失败处理 |
| `python/sglang/srt/constrained/reasoner_grammar_backend.py` | thinking/generation 两阶段状态机 |
| `python/sglang/srt/constrained/triton_ops/bitmask_ops.py` | GPU 上如何应用 token bitmask |
| `python/sglang/srt/function_call/function_call_parser.py` | 工具 schema 如何转成 Structural Tag / JSON Schema 约束 |
| `python/sglang/srt/speculative/eagle_utils.py` | 投机验证时如何应用 grammar mask |
| `python/sglang/srt/speculative/spec_utils.py` | draft 树遍历时如何推进 grammar 并生成各节点 mask |

建议阅读顺序：

```text
BaseGrammarObject 接口
        ↓
XGrammarGrammar 的六个核心方法
        ↓
XGrammarGrammarBackend 的四类 dispatch
        ↓
GrammarManager 的请求与缓存生命周期
        ↓
Reasoner / Speculative / Function Call 集成
```

其中六个核心方法是：

```text
allocate_vocab_mask
fill_vocab_mask
apply_vocab_mask
accept_token
rollback
try_jump_forward
```

抓住这六个方法，就抓住了约束解码的主干。

---

## 11. 语义压缩：最后只记住这些

### 11.1 门外汉版本

> 模型每次写下一个词前，xgrammar 都先检查这个词会不会把格式写坏；会写坏的词不让选。

### 11.2 工程师版本

> xgrammar 将结构规则编译为 tokenizer-aware 状态机，每步生成词表 bitmask，把非法 logits 置为 \(-\infty\)，并在采样后推进或回退状态。

### 11.3 一张图

```text
规则：JSON Schema / Regex / EBNF / Structural Tag
                         │
                         ▼
               编译：CompiledGrammar         ← 可缓存共享
                         │
                         ▼
               运行：GrammarMatcher          ← 每请求独立
                         │
          ┌──────────────┴──────────────┐
          ▼                             │
根据当前前缀生成 token bitmask           │
          │                             │
          ▼                             │
非法 logits = -inf                     │
          │                             │
          ▼                             │
模型在合法候选中采样 next token          │
          │                             │
          └──── accept_token / rollback ┘
```

### 11.4 最关键的边界

```text
xgrammar 保证：输出形式符合规则
xgrammar 不保证：输出事实正确、工具执行成功、业务逻辑正确
```

因此，完整可靠链路应是：

```text
Prompt 提供任务语义
        +
xgrammar 保证生成结构
        +
Parser 转成应用对象
        +
业务层重新校验并执行

