# PrefillAdder：请求准入与分块（核心骨架）

> 只看普通请求从等待到进入 prefill 批次的主线，不展开特殊模型、缓存后端和并行细节。

## 1. 一句话定位

**`PrefillAdder` 是本轮 prefill 组批的准入与预算管理器：决定一个请求能否加入、本轮处理多少输入，以及是否还能继续接纳请求。**

它在每次实际组批时创建，不是跨轮保存请求的队列，也不负责执行模型前向。

职责分工：

- `SchedulePolicy`：决定先考虑谁，对 `waiting_queue` 排序。
- `Scheduler`：遍历候选请求、检查请求名额、维护队列并构造批次。
- `PrefillAdder`：依据资源预算选出请求，确定本轮输入范围。

## 2. 调用位置与核心流程

`Scheduler.get_new_batch_prefill` → `_get_new_batch_prefill_raw` → `PrefillAdder.add_one_req`

其中 `_get_new_batch_prefill_raw` 负责：

1. **建立预算**：创建 `PrefillAdder`，传入 KV 分配器、前缀缓存、运行中批次和本轮计算预算。
2. **优先续跑旧分块**：若有 `chunked_req`，先调用 `add_chunked_req`，推进它的下一段输入。
3. **逐个考虑新请求**：遍历已排序的 `waiting_queue`；先用 `req.init_next_round_input(tree_cache)` 匹配缓存前缀，再调用 `add_one_req`。
4. **停止选人并收集结果**：根据返回状态决定是否继续遍历，以 `can_run_list` 为本轮入选结果。
5. **正式组批**：移出入选请求、保存未完成分块，构造 `ScheduleBatch`，调用 `prepare_for_extend` 准备输入及新算部分的 KV 分配。

### `add_one_req` 内部只抓四件事

- **估需求**：结合已命中的前缀，估算新增输入与后续生成所需的 KV 空间。
- **查预算**：不满足准入条件就停止接纳；锁定复用前缀时会复核 KV 余量。
- **定范围**：可以完整处理就接纳剩余输入；开启分块且超出本轮 chunk 预算时，只接纳一段，记录 `new_chunked_req`。
- **记结果**：设置 `req.extend_range`，加入 `can_run_list`、保护缓存前缀、扣减预算，再由 `budget_state` 判断是否还能继续添加。

**切块限制的是单轮计算量，不会绕过 KV 容量检查。**

## 3. 只记四类预算

| 字段 | 回答的问题 |
| --- | --- |
| `rem_total_tokens` | 考虑后续生成预留后，还能接纳多少 KV 需求？ |
| `cur_rem_tokens` | 只看当前这一步，还能容纳多少新增 KV？ |
| `rem_input_tokens` | 本轮 prefill 还剩多少输入 token 预算？ |
| `rem_chunk_tokens` | 开启分块时，本轮还剩多少 chunk token 预算？ |

KV 预算综合**分配器空闲量、可驱逐缓存量和已记账需求**；计算预算则限制本轮工作量。两者不是一回事。

后续生成空间的“预留”是预算记账，不代表立即分配整个生成阶段的 KV。对运行中请求，会用 `new_token_ratio` 估算未来需求。

## 4. 看哪些输出

- `can_run_list`：本轮实际入选的请求。
- `req.extend_range`：该请求本轮处理的输入区间。
- `new_chunked_req`：本轮新切出的、后续还需续跑的请求；由 Scheduler 保存到下一轮。

`add_one_req` 的返回值主要控制**是否继续尝试下一个请求**：

| 返回值 | 含义 |
| --- | --- |
| `CONTINUE` | 可以继续尝试添加下一个请求 |
| `NO_TOKEN` | KV 等内存预算不足，停止添加 |
| `OTHER` | 输入、分块、请求数等其他约束要求停止添加 |

**停止添加不等于当前请求未入选**：它可能已经加入，随后预算恰好耗尽。是否入选应看 `can_run_list`。未入选的新请求通常留在等待队列，不是被中止。

优先级抢占是可选扩展，由 `preempt_to_schedule` 配合 `preempt_list` 完成；理解上述主线时可先忽略。

## 5. 源码入口

- `python/sglang/srt/managers/scheduler.py`：`get_new_batch_prefill`（3682）、`_get_new_batch_prefill_raw`（3709）。
- `python/sglang/srt/managers/schedule_policy.py`：`PrefillAdder`（624）、`budget_state`（992）、`_update_prefill_budget`（1021）、`add_chunked_req`（1168）、`add_one_req`（1387）。

**记忆主线：按顺序看请求 → 算需求、查预算 → 完整接纳或切一段 → 记账 → 交回 Scheduler 组批。**
