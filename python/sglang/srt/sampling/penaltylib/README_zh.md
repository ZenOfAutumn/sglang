# srt/sampling/penaltylib

## 目录用途
本目录实现批处理（batched）的采样惩罚项库，用于在生成过程中按 batch 调整 logits。包含频率、存在、重复等惩罚以及最小新增 token 数限制，所有惩罚项由统一的编排器（orchestrator）管理生命周期与张量更新。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 汇总导出各惩罚器与编排器（`BatchedFrequencyPenalizer`、`BatchedMinNewTokensPenalizer`、`BatchedPenalizerOrchestrator`、`BatchedPresencePenalizer`、`BatchedRepetitionPenalizer`） |
| `orchestrator.py` | `BatchedPenalizerOrchestrator` 与惩罚器抽象基类 `_BatchedPenalizer`，统一管理各惩罚项的初始化、过滤、合并与对 logits 的应用 |
| `frequency_penalty.py` | `BatchedFrequencyPenalizer`，按 token 出现频次施加频率惩罚 |
| `presence_penalty.py` | `BatchedPresencePenalizer`，对已出现过的 token 施加存在惩罚 |
| `repetition_penalty.py` | `BatchedRepetitionPenalizer` 及 `apply_scaling_penalties`，按缩放系数施加重复惩罚 |
| `min_new_tokens.py` | `BatchedMinNewTokensPenalizer`，在达到最小新增 token 数前抑制 EOS 等结束 token |

## 学习说明

### 一、各惩罚的原理与区别

四种惩罚都作用在 logits（softmax 之前的打分）上，但「看什么」和「怎么改」各不相同：

| 惩罚 | 触发依据 | 作用方式 | 累计算子 | 是否随次数加重 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| repetition（重复） | token 是否出现过 | 乘/除（乘法型） | `scatter_`（赋值） | 否 | 1.0 |
| presence（存在） | token 是否出现过 | 减去固定值（加法型） | `scatter_`（赋值） | 否 | 0.0 |
| frequency（频率） | token 出现的次数 | 减去 次数×系数（加法型） | `scatter_add_`（累加） | 是 | 0.0 |
| min_new_tokens | 已生成长度是否达标 | 把 EOS 等结束 token 压到 -inf | —（按步判断） | — | 0 |

关键对比：
- **presence vs frequency**：二者都是「加法型」，区别只在累计算子。presence 用赋值（`scatter_`），出现过就是那个固定惩罚；frequency 用累加（`scatter_add_`），出现 N 次惩罚就是 N 倍。
- **repetition vs presence**：二者都「只看是否出现过」，区别在作用方式。repetition 是乘法型（对 logit 乘或除惩罚系数），presence 是加法型（直接减一个常数）。

### 二、乘法型 vs 加法型（为什么要分两类）

惩罚器用 `is_multiplicative` 标志区分两类，orchestrator 据此分别处理：
- **加法型**（presence/frequency/min_new_tokens）：维护一个 `[bs, vocab]` 的累积惩罚矩阵，应用时 `logits.sub_(累积矩阵)`。多个加法惩罚可直接相加。
- **乘法型**（repetition）：维护一个缩放系数矩阵（默认 1.0），应用时对 logit 按正负分别乘/除。多个乘法惩罚需累乘。

repetition 的正负对称处理（见 `apply_scaling_penalties`）：
- `logit < 0` → 乘以 penalty（penalty>1 时更负，更不可能被采样）；
- `logit >= 0` → 除以 penalty（penalty>1 时变小，同样降低概率）。
- 这样保证无论 logit 正负，`penalty>1` 始终抑制、`penalty<1` 始终鼓励。

### 三、架构与数据流（orchestrator 如何调度）

`BatchedPenalizerOrchestrator` 统一管理一个 batch 内的所有惩罚器，核心数据流：

1. **构造**：为每种惩罚器建实例，并对「当前 batch 确实需要」的惩罚器执行 `prepare`（分配张量）。只要有一个需要，整个 orchestrator 标记 `is_required=True`。
2. **每步累计**：解码每生成一步，`cumulate_output_tokens(output_ids)` 把新 token 喂给各惩罚器更新其累积矩阵。
3. **应用惩罚**：`apply(logits)` 把所有惩罚作用到 logits 上。
   - 常规路径：逐个惩罚器直接 `apply`。
   - 投机解码路径（`repeat` 不为空）：一个请求对应 `repeat` 行 logits（draft token），先按「每请求」算惩罚，再用 `repeat_interleave` 扩展到「每 draft token」——加法型先汇总到全零张量再扩展相加，乘法型先累乘再扩展应用。

### 四、生命周期与「按需准备」（lazy prepare）

为了零开销地跳过未启用的惩罚，框架采用惰性策略：
- **`_is_required()`**：检查 batch 中是否有任一请求设了非默认值。全为默认值则返回 False，**完全不分配张量、不参与计算**。
- **`prepare` / `teardown`**：需要时才分配 `[bs, vocab]` 矩阵；不再需要时释放张量引用便于 GC。
- **`filter(keep_indices)`**：部分请求完成被移出 batch 时，按保留行索引裁剪各惩罚器张量，保持与 batch 对齐；若某惩罚器不再被需要则 teardown。
- **`merge(their)`**：两个 batch 合并时沿 batch 维拼接张量。**必须在 `batch.reqs` 被更新前调用**，且未准备的惩罚器需先 prepare 再合并。
- **`release` / 上下文管理器**：释放所有惩罚器并断开对 `ScheduleBatch` 的引用。orchestrator 用 **弱引用（weakref）** 持有 batch，避免循环引用导致显存无法及时回收。

### 五、扩展：如何新增一个惩罚器

继承 `_BatchedPenalizer`，按需实现这几个抽象方法即可被 orchestrator 自动调度：
- `_is_required()`：判断本惩罚在当前 batch 是否需要启用；
- `_prepare()`：分配内部张量（通常是 `[bs, vocab]` 累积矩阵 + `[bs, 1]` 系数）；
- `_cumulate_output_tokens()`：每步把新 token 计入累积矩阵（赋值用 `scatter_`，累加用 `scatter_add_`）；
- `_apply()`：把惩罚作用到 logits（加法型 `sub_`，乘法型重写 `get_scaling_penalties` 并设 `is_multiplicative=True`）；
- `_filter()` / `_merge()` / `_teardown()`：分别处理裁剪、合并、释放。
