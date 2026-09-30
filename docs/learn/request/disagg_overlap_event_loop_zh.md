# PD 分离重叠调度详解：`event_loop_overlap_disagg_prefill` 与 `event_loop_overlap_disagg_decode`

> 本文是《调度与执行层：Scheduler 详解（PD 分离 + 重叠调度）》（`docs/learn/request/scheduler_processing_zh.md`）的专题延伸篇，聚焦 PD 分离部署下 P 端、D 端两条**重叠（overlap）事件循环**的逐行行为、跨轮时序和容易踩坑的边界条件。
>
> 通用概念——`result_queue`、`batch.copy()`、`_apply_war_barrier`、`future_map`、三条 CUDA Stream、`launch_batch_sample_if_needed`——在原文 §4.1.1、§4.3、§4.6、§4.7 中已有介绍，本文只在与 PD 分离相关的地方展开。文中「原文 §x」均指该文档的章节。

---

## 1. 定位：两条循环在整体中的位置

### 1.1 入口分派

`Scheduler.run_event_loop` 先建立 `schedule_stream`，确保它与 `forward_stream` 不是同一个底层 stream，并确定是否启用 WAR 屏障；然后在 `schedule_stream` 上下文中调用 `dispatch_event_loop`：

```5749:5762:python/sglang/srt/managers/scheduler.py
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if get_parallel().pp_size > 1:
            scheduler.event_loop_pp_disagg_prefill()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_prefill()
        else:
            scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if get_parallel().pp_size > 1:
            scheduler.event_loop_pp_disagg_decode()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_decode()
        else:
            scheduler.event_loop_normal_disagg_decode()
```

进入本文两条循环需要同时满足：

| 条件 | Prefill 侧 | Decode 侧 |
| --- | --- | --- |
| `disaggregation_mode` | `PREFILL` | `DECODE` |
| `pp_size` | `== 1`（PP 优先级更高，`> 1` 时走 `event_loop_pp_disagg_*`） | `== 1` |
| `enable_overlap` | `True`（即未设置 `--disable-overlap-schedule`） | `True` |

两者分别定义在 `SchedulerDisaggregationPrefillMixin`（`srt/disaggregation/prefill.py:768-847`）与 `SchedulerDisaggregationDecodeMixin`（`srt/disaggregation/decode.py:2724-2788`）中，通过 Mixin 混入 `Scheduler`（原文 §4.1）。

### 1.2 两侧职责对比

```text
            Prefill 实例（本文 §2）                          Decode 实例（本文 §3）
 ┌────────────────────────────────────────┐        ┌────────────────────────────────────────┐
 │ bootstrap_queue ──► waiting_queue       │        │ prealloc_queue ──► transfer_queue       │
 │   (握手/乐观抢跑)     │                  │        │   (预分配 KV 落点)     │                │
 │                      ▼                  │  KV +  │                        ▼                │
 │          EXTEND 前向（产首 token）      │ 元数据 │               waiting_queue             │
 │                      │                  │ ─────► │                        │                │
 │                      ▼                  │  RDMA  │     PREBUILT（不跑模型，只写元数据）   │
 │ inflight_queue（轮询发送完成）          │        │                        ▼                │
 │   └► 成功：释放 KV，回包首 token         │        │   running_batch ──► DECODE 前向（循环）  │
 └────────────────────────────────────────┘        └────────────────────────────────────────┘
```

- **P 端只跑 `EXTEND`**：每个请求的 `max_new_tokens` 在进 bootstrap 队列时被强制改为 1（`PrefillBootstrapQueue._process_req`，`prefill.py:475-481`），前向产出首 token 后即把 KV 发给 D 端；P 端**永远不进入 decode**。
- **D 端基本只跑 `DECODE`**：KV 到达后，请求经过 `PREBUILT` 这一「伪 prefill」阶段（只写元数据、不跑模型）并入 `running_batch`，之后按普通 decode 逐 token 生成。

这一差异直接决定了两条重叠循环的写法：P 端的「上一批」和「本批」都是 extend 批，D 端的「上一批」和「本批」都是 decode 批。

---

## 2. Prefill 侧：`event_loop_overlap_disagg_prefill`

### 2.1 完整代码

```767:847:python/sglang/srt/disaggregation/prefill.py
    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        self.result_queue = deque()

        while True:
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )

            # Get the next batch to run
            plan = self.get_next_disagg_prefill_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                batch_result = self.run_batch(batch)
                self._apply_war_barrier()
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                self.on_idle()

            self.process_disagg_prefill_inflight_queue()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result, batch)

            self.last_batch = batch
```

（为便于阅读，上面省略了源码中的逐行中文注释。）

### 2.2 单轮流程分解

每轮循环分为 9 步，按顺序标号为 P1~P9：

| 步骤 | 代码 | 作用 | 关键细节 |
| --- | --- | --- | --- |
| P1 | `recv_requests` + `process_input_requests` | 收新请求/控制请求 | 生成请求经 `_add_request_to_queue` 进入 `disagg_prefill_bootstrap_queue`：创建 KV sender，检查是否超出 KV 总容量，强制 `max_new_tokens=1`，并置 `pending_bootstrap=True` |
| P2 | `if self._engine_paused: continue` | 引擎暂停（如权重更新期间）时不推进 | 控制请求仍会在 P1 中处理 |
| P3 | `pop_bootstrapped()` → `waiting_queue` | 把握手已完成或允许乐观抢跑的请求放进 `waiting_queue` | 见 §2.3 |
| P4 | `get_next_disagg_prefill_batch_to_run` | 组 prefill 批 | 内部依次调用 `process_pending_chunked_abort`、重置 `batch_is_full`、`resolve_waiting_queue_bootstrap`、`process_prefill_chunk`、`get_new_batch_prefill`、`maybe_prepare_mlp_sync_batch` |
| P5 | `ngram_embedding_manager.prepare_for_forward` | ngram embedding 预处理 | 未启用时原样返回 |
| P6 | `maybe_prefetch_staging_for_batch` → `run_batch` → `_apply_war_barrier` → 入队 | **提交本批**前向 | `run_batch` 内对每个请求调用 `maybe_send_cached_prefix_chunk`（前缀 KV 提前发送，见 §2.6） |
| P7 | `popleft` + `process_batch_result` | **处理上一批**结果 | 按 forward_mode 分派到 `process_batch_result_disagg_prefill` |
| P8 | `process_disagg_prefill_inflight_queue` | 轮询在途 KV 传输 | 每轮都执行，包括空闲轮 |
| P9 | `launch_batch_sample_if_needed` → `last_batch = batch` | 补交本批延迟采样，然后进入下一拍 | 本批需要 grammar/延迟采样时才实际执行 |

### 2.3 P3：bootstrap 放行与乐观 prefill

`pop_bootstrapped`（`prefill.py:483-583`）在 attn-CP/TP 组内通过 `poll_and_all_reduce_attn_cp_tp_group` 让各 rank 对握手状态达成一致，然后逐请求分流：

| poll 状态 | 处理 | 进入 `waiting_queue` 时的 `pending_bootstrap` |
| --- | --- | --- |
| `WaitingForInput` | `finalize_bootstrap`：分配 metadata 槽位，读取 `decode_prefix_len`，调用 `sender.init(num_pages, idx)` | `False`（已就绪） |
| `Bootstrapping`，且 `prefill_attempt_count < optimistic_prefill_attempts`，请求也未被回撤 | 只预留 metadata 槽位，**乐观抢跑** | `True`（仍待确认） |
| `Failed` | `handle_bootstrap_failure`：中止、释放资源、回包 | 不进入 |
| metadata 槽位不足 | `continue`，下一轮再尝试 | 不进入 |

乐观 prefill 的意义在于：D 端握手慢时，P 端不用干等，可以先把 prompt 算掉。代价是：如果前向完成时握手仍未完成，KV 就不能发送，只能先挂起，或者释放后重新排队。在重叠循环中，这类请求的确认和回退点更多，见 §2.5。

### 2.4 P4：组批里与重叠相关的两处

#### (1) `resolve_waiting_queue_bootstrap`：入批前的最后一次握手补查

```620:676:python/sglang/srt/disaggregation/prefill.py
    def resolve_waiting_queue_bootstrap(self: Scheduler) -> None:
        ...
        candidates = [req for req in self.waiting_queue if not is_aborted(req)]
        ...
        for req, poll in zip(candidates, polls):
            if poll == KVPoll.Failed:
                self.handle_bootstrap_failure(req)
                failed.add(req)
            elif (
                poll == KVPoll.WaitingForInput
                and req.pending_bootstrap
                and not should_force_retry(req)
            ):
                self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
```

这一步会在 `waiting_queue` 中提前剔除对端已失效的请求，避免白算 prefill；对已完成握手的乐观请求补做 `finalize`，这样前向完成后就可以直接发送 KV。

#### (2) `process_prefill_chunk`：重叠模式下**推迟**中间 chunk 的 KV 发送

分块 prefill（chunked prefill）会把长 prompt 拆成多段，分多轮计算。每轮开始时，`process_prefill_chunk` 处理 `self.chunked_req`：

```1336:1374:python/sglang/srt/disaggregation/prefill.py
        if (req := self.chunked_req) is not None:
            chunked_req_to_exclude.add(req)
            maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)

            if not self.check_bootstrap(req):
                if is_aborted(req):
                    # bootstrap failed
                    self.chunked_req = None
                elif self.has_bootstrapped_waiting_req():
                    # optimistic request yields to waiting requests
                    self.chunked_req = None
                    if not self.enable_overlap:
                        self.optimistic_release_and_requeue(req)
                # else: still bootstrapping, keep computing without sending
            elif self.enable_overlap:
                # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                req.tmp_end_idx = min(
                    req.extend_range.end,
                    len(req.origin_input_ids),
                )
            else:
                self.send_kv_chunk(req)
```

这里是重叠版和非重叠版**最关键的行为差异**：

| 场景 | 非重叠 | 重叠 |
| --- | --- | --- |
| 握手已完成，发送上一 chunk 的 KV | 立即调用 `send_kv_chunk(req)` | 只记录 `req.tmp_end_idx`，实际发送推迟到 P7 的 `process_batch_result_disagg_prefill` |
| 乐观请求让位给已就绪请求 | 立即 `optimistic_release_and_requeue` | 推迟：上一 chunk 的前向结果尚未处理，不能此时释放 KV，由 P7 延后处理 |

原因是：重叠模式下，执行到 P4 时，上一轮提交的 chunk 前向**可能仍在 GPU 上写入 KV**，其结果也尚未被 CPU 消费。如果此时发送，RDMA 可能读到尚未写完的 KV 页；如果此时释放，前向可能还在写这些页。因此只能先记录「这一块的右边界」，等 P7 确认结果后再处理。

> `get_new_batch_prefill` 每组出一个包含分块请求的批，都会执行 `chunked_req.inflight_middle_chunks += 1`（`scheduler.py:3973-3974`）。在重叠模式下，这个计数跨越「已提交但尚未处理」的批次，P7 依据它判断当前处理的是中间 chunk 还是最后一个 chunk。

### 2.5 P7：`process_batch_result_disagg_prefill` 的重叠语义

P7 处理的是 `tmp_batch`（上一批的快照），此时**本批已经提交**。函数入口首先调用 `copy_done.synchronize()`，等待上一批的 D2H 拷贝完成；这也是 P 端 CPU 可能阻塞的主要位置。

对每个请求，函数按 `inflight_middle_chunks` 分成两条路径：

**A. 最后一个 chunk（`inflight_middle_chunks <= 0`）：prefill 全部完成**

```951:1007:python/sglang/srt/disaggregation/prefill.py
            if req.inflight_middle_chunks <= 0:
                req.time_stats.set_prefill_finished_time()
                if req.pending_bootstrap and should_force_retry(req):
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    continue
                req.output_ids.append(next_token_id)
                maybe_cache_unfinished_req(req, self.tree_cache)
                self.disagg_prefill_inflight_queue.append(req)
                ...
                if not req.pending_bootstrap:
                    self.send_kv_chunk(req, last_chunk=True)
                req.time_stats.set_prefill_transfer_queue_entry_time()
```

- 追加首 token，把 KV 写入 radix tree，然后放入 `disagg_prefill_inflight_queue`；
- 握手已就绪时，立即发送最后一块 KV：`last_chunk=True` 会同时通过 `set_buf` 写入 metadata（首 token、logprob、hidden states 等），并处理 Mamba/SWA 等状态页；
- 仍处于乐观 pending 状态时，**先挂在 inflight 队列中不发送**，由 P8 轮询到握手完成后再补发（见 §2.7）。

**B. 中间 chunk（`inflight_middle_chunks > 0`）**

```1025:1073:python/sglang/srt/disaggregation/prefill.py
                req.inflight_middle_chunks -= 1
                still_chunking = self.chunked_req is req or (
                    req.extend_range is not None
                    and req.extend_range.end >= len(req.origin_input_ids)
                )
                if req.pending_bootstrap and not still_chunking:
                    self.optimistic_release_and_requeue(req)
                    ...
                    continue
                if is_aborted(req):
                    self.clear_pending_chunk_send(req)
                    ...
                    continue
                ...
                if self.enable_overlap and not req.pending_bootstrap:
                    ...
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
```

- `still_chunking`：用来判断该请求的下一块是否已经提交。如果 P4 中该请求已经让位，也就是 `chunked_req` 被置为 `None`，且下一块没有提交，那么这里补做 P4 推迟的 `optimistic_release_and_requeue`；
- `is_aborted`：乐观握手可能在这一块已经开始执行后才失败，此时丢弃该 chunk，不发送；
- 正常情况下，按 P4 记录的 `tmp_end_idx` 发送这一块 KV。`send_kv_chunk` 在非最后一块时会把 `end_idx` 向下对齐到页边界；开启 staging 时还会对齐到 staging grid，剩余部分随下一次发送。

> **一句话总结**：在重叠版 P 端中，**发送 KV、释放 KV、乐观回退这三件事都只在「结果已确认」的 P7 中执行**，而非重叠版可以在 P4 中直接执行。

### 2.6 P6：前向提交与前缀 KV 提前发送

`run_batch` 在 PD prefill 模式下会先对每个请求调用 `maybe_send_cached_prefix_chunk`，然后再提交前向：

```4293:4296:python/sglang/srt/managers/scheduler.py
        # PD prefill: early-send cached prefix KV, overlapping the suffix forward.
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            for req in batch.reqs:
                self.maybe_send_cached_prefix_chunk(req)
```

只有开启 `SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX` 时，这一步才会生效：它把 radix 命中的设备侧前缀 KV 先发出去，与后缀前向重叠。重叠模式下还有一个细节：

```1430:1434:python/sglang/srt/disaggregation/prefill.py
        if self.enable_overlap:
            ev = torch.cuda.Event()
            ev.record(self.forward_stream)
            req.disagg_kv_sender._early_send_wait_event = ev
        self.send_kv_chunk(req, last_chunk=False, end_idx=cached_end)
```

此时**上一批的 prefill 前向可能还在 `forward_stream` 上写这些前缀页**。例如，前缀就是上一批刚算完、刚插入 radix tree 的内容。因此代码会先在 `forward_stream` 上记录一个事件，让传输线程在 RDMA 读取前等待该事件，避免读写竞争。

`maybe_prefetch_staging_for_batch` 则在开启 staging 时向 D 端预发 `STAGING_REQ`，让 D 端在 P 端前向期间预先分配 staging 空间。

### 2.7 P8：在途队列轮询

`process_disagg_prefill_inflight_queue`（`prefill.py:1093-1215`）**每轮都会执行**，无论本轮是否有批次。它在组内对 poll 结果做 all-reduce，然后逐请求处理：

| 状态 | 处理 |
| --- | --- |
| `req.pending_bootstrap` 为真，说明请求在 P7 中因握手未完成而被挂起 | `handle_pending_bootstrap`：握手完成时 `finalize` 并补发 `send_kv_chunk(last_chunk=True)`；握手失败时中止；仍在握手时继续等待 |
| `WaitingForInput` / `Transferring` | 保留到下一轮 |
| `Success` | 设置 `FINISH_LENGTH(0)`，调用 `release_kv_cache` 解锁 radix tree，执行 `sender.clear()`，上报传输指标 |
| `Failed` | `handle_inflight_transfer_failure` 中止请求 |

最后，函数对所有已完成的请求统一调用 `stream_output`，把首 token 流式返回给客户端，并释放 metadata 槽位。

P8 放在 P7 之后有两个好处：P7 中刚加入 inflight 队列的请求可以在同一轮就被首次轮询；而且 P8 的 CPU 工作同样会与本批 GPU 前向重叠。

### 2.8 P 端单轮时序图

```text
第 k 轮（上一轮已提交批次 N；本轮提交批次 N+1）

CPU: |P1 收请求|P3 放行|P4 组批 N+1|P6 run_batch(N+1)|P7 处理 N 的结果       |P8 轮询 inflight|P9|
     |         |       | ├resolve   | ├早发前缀 KV     | ├copy_done.sync(N)    | ├补发 pending   |  |
     |         |       | └chunk:记录| ├提交前向        | ├首 token/入 inflight | └成功→回包      |  |
     |         |       |  tmp_end   | └WAR 屏障        | └send_kv_chunk        |                 |  |
GPU: |=================== G[N]（extend） ================|=========== G[N+1]（extend） ===========...
                                                        ↑
                                  G[N+1] 在 forward_stream 上排在 G[N] 之后执行
```

与聚合版重叠循环一样，`G[N]` 期间 CPU 可以完成 P1~P6；`G[N+1]` 期间 CPU 可以完成 P7~P9。**KV 发送是异步的**：`send_kv_chunk` 只把任务交给传输后端，由后台线程执行 RDMA，不阻塞调度线程。

### 2.9 与聚合版 `event_loop_overlap` 的差异

| 项目 | 聚合版 `event_loop_overlap` | P 端 `event_loop_overlap_disagg_prefill` |
| --- | --- | --- |
| 选批 | `get_next_batch_to_run`（prefill 优先，否则 decode） | `get_next_disagg_prefill_batch_to_run`（**只组 prefill**，不调用 `update_running_batch`） |
| 放行 | 无 | `pop_bootstrapped()` |
| `is_disable_overlap_for_batch` | 有：连续 prefill 且开启 `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` 时，或 spec+grammar 需要同步时 | **没有**：P 端连续批次都是 extend，始终保持一拍错位 |
| `gracefully_exit` 检查 | 有 | 无 |
| 每轮额外步骤 | — | `process_disagg_prefill_inflight_queue()` |
| 空闲判断 | `last_batch is None and batch is None` → `on_idle` | 相同，但 `on_idle` 内部只有在 `is_fully_idle()` 时才做完整自检；inflight 或 bootstrap 队列非空时只发布负载快照 |
| `launch_batch_sample_if_needed` | 仅当 `is_generation` 时调用 | 无条件调用（P 端必然是生成模型） |
| `running_batch.batch_is_full` | 由 `update_running_batch` 重置 | 由于不调用 `update_running_batch`，需在选批入口**手动重置为 `False`**，否则高并发下可能卡住 |

---

## 3. Decode 侧：`event_loop_overlap_disagg_decode`

### 3.1 完整代码

```2723:2788:python/sglang/srt/disaggregation/decode.py
    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        self.result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None

        def pop_and_process():
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            # Pending rooms from the prior cycle can overlap request intake and
            # the tail of the in-flight decode graph.
            if not self._engine_paused:
                self.disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries()
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            self.process_decode_queue()

            # Get the next batch to run
            plan = self.get_next_disagg_decode_batch_to_run(
                running_batch=self.running_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch
            # overlap + spec + grammar is unsupported (would desync DP ranks).
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(
                batch, last_batch=self.last_batch
            )

            if disable_overlap_for_batch and self.last_batch:
                pop_and_process()

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self._apply_war_barrier()
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                self.on_idle()

            self.launch_batch_sample_if_needed(batch_result, batch)

            # Update last_batch
            self.last_batch = batch
```

### 3.2 单轮流程分解

每轮循环分为 10 步，按顺序标号为 D1~D10：

| 步骤 | 代码 | 作用 | 关键细节 |
| --- | --- | --- | --- |
| D1 | `prefetch_prefill_dp_rank_queries` | 提前异步查询 P 端 DP rank | 把上一轮留在 `pending_reqs` 中的 `bootstrap_room` 按地址分组，提交到线程池查询，使网络查询与收请求、GPU 前向尾部重叠 |
| D2 | `recv_requests` + `process_input_requests` | 收请求 | 生成请求进入 `disagg_decode_prealloc_queue.add`：创建 `DecodeRequest` 和 receiver，并尝试从缓存中解析 `prefill_dp_rank` |
| D3 | `if self._engine_paused: continue` | 暂停时不推进 | — |
| D4 | `process_decode_queue` | 推进「预分配 → 传输 → 等待」三级队列 | 见 §3.3 |
| D5 | `get_next_disagg_decode_batch_to_run` | **同步完成 PREBUILT 收尾**，合并进 `running_batch`，然后为 DECODE 做准备 | 见 §3.4 |
| D6 | `prepare_for_forward` / `cur_batch_for_debug` | 预处理与调试现场记录 | — |
| D7 | `is_disable_overlap_for_batch` →（必要时）`pop_and_process` | 判断是否临时关闭重叠 | 见 §3.6 |
| D8 | `run_batch` → `_apply_war_barrier` → 入队 | **提交本批** DECODE 前向 | — |
| D9 | `pop_and_process` / `on_idle` | **处理上一批**结果 | `process_batch_result_decode`；本轮 D7 已处理过上一批时跳过 |
| D10 | `launch_batch_sample_if_needed` → `last_batch = batch` | 补交延迟采样 | 结构化输出（grammar）时常见 |

### 3.3 D4：`process_decode_queue` 队列推进

```2917:2982:python/sglang/srt/disaggregation/decode.py
    def process_decode_queue(self: Scheduler):
        if self.enable_decode_hicache:
            self.tree_cache.check_hicache_events()
        if get_disagg().disaggregation_decode_enable_offload_kvcache:
            self.decode_offload_manager.check_offload_progress()
        self.disagg_decode_transfer_queue.resolve_deferred_releases()

        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            return
        ...
        self.polling_count = (self.polling_count + 1) % self.polling_interval
        if self.polling_count % self.polling_interval == 0:
            req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
            self.disagg_decode_transfer_queue.extend(req_conns)
            transferred_reqs = (
                self.disagg_decode_transfer_queue.pop_transferred()
            )
            ...
            self.waiting_queue.extend(transferred_reqs)
```

按执行顺序：

1. **HiCache / offload 进度**：检查分层缓存回载和 KV 卸载等异步事件。
2. **`resolve_deferred_releases`**：由 D 端发起 abort 时，P 端可能仍在向这些页写入 RDMA，所以相关页被暂时持有；收到 P 端的全部 drain ack 或超时后才真正释放。该步骤**每轮都会执行**，并且位于后面的回撤/轮询门控之前，确保内存紧张时超时释放仍能生效。
3. **优先恢复回撤请求**：`resume_retracted_reqs` 在显存足够时重新预分配 KV，并调用 `retraction_restore` 从 CPU 回载 KV。只要 `retracted_queue` 非空，本轮就**不接纳新请求**，防止回撤请求饿死。
4. **轮询节流**：只有每隔 `disaggregation_decode_polling_interval` 轮，才执行一次开销较大的跨 rank poll。
5. **`pop_preallocated`**：先调用 `_resolve_pending_reqs`，消费 D1 预取的 DP rank 查询结果；再在多项预算约束下预分配 KV 落点，并通过 `send_metadata` 告知 P 端。反死锁约束是：「可回退 token + 可分配 token」必须大于任一在途请求的最大需求。详见原文 §4.11.3。
6. **`pop_transferred`**：`Success` 时调用 `_commit_transfer_to_req`，校验 `bootstrap_room`，并把首 token、`cached_tokens`、logprob 等写回 `Req`，然后放入 `waiting_queue`；`Failed` 时中止并释放，D 端主动 abort 时则走延迟释放。

> **重叠下的一个注意点**：D4 运行时，上一轮的 DECODE 前向可能仍在 GPU 上执行。D4 中的 KV 分配和释放都发生在 `schedule_stream` 上，并由 D8 之后的 WAR 屏障约束：`schedule_stream` 上后续的写入要等上一批前向对共享缓冲的读取完成后才会执行，因此不会覆盖前向正在使用的数据。

### 3.4 D5：PREBUILT 在选批阶段被「同步消化」

```2803:2835:python/sglang/srt/disaggregation/decode.py
    def get_next_disagg_decode_batch_to_run(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> NextBatchPlan:
        new_prebuilt_batch = self.get_new_prebuilt_batch(running_batch)
        if new_prebuilt_batch:
            assert self.chunked_req is None
            self.batch_result_processor.process_batch_result_prebuilt(
                new_prebuilt_batch
            )
            new_prebuilt_batch.filter_batch()
            if not new_prebuilt_batch.is_empty():
                if running_batch.is_empty():
                    running_batch = new_prebuilt_batch
                    ...
                else:
                    running_batch.merge_batch(new_prebuilt_batch)

        if running_batch.is_empty():
            ret = None
        else:
            running_batch = self.update_running_batch(running_batch)
            ret = running_batch if not running_batch.is_empty() else None

        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret)
        ...
        return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)
```

这里有一个容易误解的点：**PREBUILT 批不经过 `result_queue`，也不参与一拍错位**。它在 D5 中被**同步**完成三步处理：

1. `get_new_prebuilt_batch`：按 `running_batch` 的剩余容量，从 `waiting_queue` 取出请求，然后调用 `prepare_for_prebuilt` 和 `process_prebuilt`；
2. `process_batch_result_prebuilt`：调用 `update_finish_state`，例如首 token 就命中 stop 条件时，直接完成并释放 KV；随后 `stream_output` 把**首 token** 推送给客户端；
3. `filter_batch` 去掉已完成请求，然后执行 `merge_batch` 并入 `running_batch`。

随后 `update_running_batch` 对合并后的 `running_batch` 调用 `filter_batch`、`check_decode_mem`，必要时执行 `retract_decode`，最后调用 `prepare_for_decode`，生成**本轮真正执行的 DECODE 批**。所以在非 PP 的重叠循环中，D8 实际提交的 `batch` 通常是 `DECODE` 批，或者 DP 同步所需的 `IDLE` 批。`run_batch` 中 `is_prebuilt()` → `_run_batch_prebuilt` 这条短路主要服务于 PP 循环：在 PP 中，PREBUILT 批会作为独立的 micro-batch 流转。

#### `get_new_prebuilt_batch` 中与重叠相关的屏障

```2906:2913:python/sglang/srt/disaggregation/decode.py
        # construct fake completed prefill
        new_batch.prepare_for_prebuilt()
        if self.enable_overlap:
            # A finished request can still have one redundant forward in flight.
            # Drain it before a prebuilt request seeds a potentially reused row.
            self.schedule_stream.wait_stream(self.forward_stream)
        new_batch.process_prebuilt(self.future_map)
```

`process_prebuilt` 会通过 `future_map.stash(req_pool_indices, bonus_tokens=首 token)` 把首 token 写入**以 `req_pool_idx` 为下标**的中继缓冲，供第一次 DECODE 的 `resolve_forward_inputs` 读取。问题在于，在重叠模式下，某个已完成请求可能**还有一次多余的前向正在 GPU 上执行**（§3.5），而它释放的 `req_pool_idx` 可能刚好被新的 PREBUILT 请求复用。如果不等待，这次在途前向结束时发布的旧 token 可能覆盖新请求刚写入的首 token。因此这里要求 `schedule_stream` 等待 `forward_stream`。注意，这是 stream 间依赖，不会阻塞 CPU。

### 3.5 D9：延迟一拍带来的「多余前向」

D 端在稳态下，每轮都是「提交 DECODE[N+1]，然后处理 DECODE[N] 的结果」。这里存在一个不可避免的问题：

```text
轮 k   : 提交 DECODE[N]    （请求 r 在其中生成的 token 恰好是 EOS）
轮 k+1 : D5 为 N+1 选批时 r 看起来仍未完成 → r 被放进 DECODE[N+1] 并提交
         D9 处理 DECODE[N] 结果 → r.update_finish_state() → r 完成、释放 KV、回包
轮 k+2 : D9 处理 DECODE[N+1] 结果 → r 已 finished → 跳过
```

对应代码在 `process_batch_result_decode` 中：

```1038:1044:python/sglang/srt/managers/scheduler_components/batch_result_processor.py
            if (self.enable_overlap or self.enable_overlap_mlx) and (
                req.finished() or req.is_retracted
            ):
                # NOTE: This (req.finished() or req.is_retracted) should only happen when overlap scheduling is enabled.
                # And all the over-allocated tokens will be freed in `release_kv_cache`.
                ...
                continue
```

- 每个请求最多会多执行 **1 次**无效前向，这是重叠调度隐藏 CPU 开销的代价；
- 多分配的 KV token 会在 `release_kv_cache` 中统一回收；
- 回撤同理：DECODE[N] 在执行时，D5 中的 `update_running_batch` 可能已经回撤了其中某些请求，这些请求的结果也会被跳过；
- §3.4 中 `get_new_prebuilt_batch` 的 `wait_stream` 屏障，就是为了避免这次多余前向影响复用该槽位的新请求。

### 3.6 D7：`is_disable_overlap_for_batch` 在 D 端的实际效果

```2054:2090:python/sglang/srt/managers/scheduler.py
        disable_overlap_for_batch = (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            and batch_is_extend
            and last_batch_is_extend
        )
        need_grammar_sync = (
            batch
            and not batch.spec_algorithm.is_none()
            and batch.grammar_needs_sync()
            and batch.forward_mode.is_decode()
            and len(self.result_queue) > 0
        )
        return disable_overlap_for_batch or need_grammar_sync
```

- **连续 prefill 分支**：D 端提交的批不是 extend 批，所以该分支基本不会触发；
- **`need_grammar_sync` 分支**：使用**投机解码 + grammar 且需要同步**时（host-draft 类算法），必须先处理上一批，让 grammar FSM 前进一步，才能生成本批的 vocab mask。此时本轮退化为「先处理上批，再提交本批」。源码注释中的「overlap + spec + grammar is unsupported (would desync DP ranks)」指的就是这种组合下不能保持重叠。

与聚合版相比有一个细微差异：D 端写的是 `if disable_overlap_for_batch and self.last_batch:`，多了一个 `last_batch` 判空；聚合版只写了 `if disable_overlap_for_batch:`。由于 `need_grammar_sync` 自身已要求 `len(result_queue) > 0`，实际行为是等价的，D 端的写法只是更防御。

### 3.7 D10：延迟采样

非投机模式下，如果启用重叠且批中带 grammar，或开启 `SGLANG_ENABLE_DELAY_SAMPLE`，`TpModelWorker.forward_batch_generation` 只完成前向，不执行采样，而是返回 `delay_sample_func`（`tp_worker.py:748-767`）。`launch_batch_sample_if_needed` 会在**上一批结果处理完之后**执行以下操作：

1. 在 `forward_stream` 上执行采样，因为 grammar 的 vocab mask 依赖上一批 token 使 FSM 前进后的状态；
2. 调用 `_relay_forward_payload`，把采样出的 token 写入 `future_map`；
3. 在 `copy_stream` 上发起 D2H 拷贝；
4. 清空 `delay_sample_func` 和 `next_token_logits`，避免闭包长期持有大张量，导致显存持续泄漏。

所以在 grammar 场景下，本批「前向」与上批「结果处理」仍然可以重叠，只有「采样」这一小步需要串行等待。

### 3.8 D 端单轮时序图

```text
第 k 轮（上一轮已提交 DECODE[N]；本轮提交 DECODE[N+1]）

CPU: |D1 预取 dp rank|D2 收请求|D4 队列推进           |D5 选批                          |D8 run_batch|D9 处理 N 结果  |D10|
     |               |         | ├deferred release    | ├PREBUILT: 首 token 回包         | (N+1)      | ├copy_done.sync|   |
     |               |         | ├恢复回撤            | │  └wait_stream(fwd) + stash     | └WAR 屏障  | ├追加 token     |   |
     |               |         | ├pop_preallocated    | ├merge 入 running_batch          |            | ├完成→释放 KV   |   |
     |               |         | └pop_transferred     | └update_running_batch→DECODE     |            | └stream_output |   |
GPU: |============================= G[N]（decode） =============================|======== G[N+1]（decode） ========...
```

- `G[N]` 期间，CPU 完成 D1~D8，包括网络轮询、预分配和 PREBUILT 处理；
- `G[N+1]` 期间，CPU 完成 D9~D10；
- `G[N+1]` 的 `input_ids` 不依赖 CPU 端的 D9：`resolve_forward_inputs` 在 GPU 上直接从 `future_map.output_tokens_buf[req_pool_indices]` 读取 `G[N]` 的采样结果。新并入的 PREBUILT 请求则读取 `process_prebuilt` 写入的首 token。这是 D 端 decode 可以做到一拍错位的根本原因（原文 §4.7）。

### 3.9 与聚合版 `event_loop_overlap` 的差异

| 项目 | 聚合版 `event_loop_overlap` | D 端 `event_loop_overlap_disagg_decode` |
| --- | --- | --- |
| 每轮开头 | — | `prefetch_prefill_dp_rank_queries()`（仅在未暂停时） |
| 队列推进 | — | `process_decode_queue()` |
| 选批 | `get_next_batch_to_run`（合并上一个 prefill 批 → 新 prefill → decode） | `get_next_disagg_decode_batch_to_run`（同步处理 PREBUILT → merge → decode） |
| 新请求进入 running | 上一个 prefill 批**需要一轮错位后**，才在下一轮 `get_next_batch_to_run` 中 merge | PREBUILT 在**当轮 D5** 同步完成并 merge，首 token 当轮回包 |
| `last_batch` 参数 | 选批需要依赖它合并 prefill 批 | 选批不依赖 `last_batch`（签名中没有该参数）；它只用于重叠判断、`pop_preallocated` 预算计算等 |
| 关闭重叠时 | `if disable: pop_and_process()`，并在 unified memory 下机会性 flush | `if disable and last_batch: pop_and_process()`，没有 flush |
| `gracefully_exit` / `_sched_idled` / 严格内存自检 | 有 | 无 |

---

## 4. 两条循环的横向对照

### 4.1 结构对照

```text
                  event_loop_overlap_disagg_prefill      event_loop_overlap_disagg_decode
开头              —                                        prefetch_prefill_dp_rank_queries
收请求            recv + process_input_requests            recv + process_input_requests
暂停门控          _engine_paused → continue                _engine_paused → continue
队列推进          pop_bootstrapped → waiting_queue         process_decode_queue
选批              get_next_disagg_prefill_batch_to_run     get_next_disagg_decode_batch_to_run
重叠开关          （无）                                    is_disable_overlap_for_batch
提交本批          [staging 预取] run_batch + WAR + 入队     run_batch + WAR + 入队
处理上批          popleft + process_batch_result           pop_and_process
                  （→ process_batch_result_disagg_prefill） （→ process_batch_result_decode）
每轮尾部          process_disagg_prefill_inflight_queue    —
延迟采样          launch_batch_sample_if_needed            launch_batch_sample_if_needed
```

### 4.2 语义对照

| 维度 | P 端 | D 端 |
| --- | --- | --- |
| 提交批的 `forward_mode` | `EXTEND`（或 DP 同步用的 `IDLE`） | `DECODE`（或 `IDLE`） |
| 跨轮依赖 | 分块 prefill：KV 发送边界 `tmp_end_idx`、`inflight_middle_chunks` | decode token：`future_map` 中继 |
| 「上一批结果」主要用于 | 追加首 token、**发起 KV 传输**、乐观回退 | 追加 token、判断是否完成、释放 KV、流式输出 |
| 重叠带来的额外约束 | 中间 chunk 的发送和释放必须推迟到 P7；早发前缀需要等待 `forward_stream` 事件 | 已完成请求会多跑 1 次前向；PREBUILT 写入 `future_map` 前需要 `wait_stream` |
| 首 token 回包时机 | KV 传输 `Success` 后，在 P8 中回包 | KV 到达后，在 D5 PREBUILT 阶段回包 |
| 空闲时仍需推进 | inflight 轮询（P8）、bootstrap 放行（P3） | 队列推进（D4），包括 deferred release |
| 跨 rank 一致性 | `poll_and_all_reduce_attn_cp_tp_group` | 同左，外加 `polling_interval` 节流 |

### 4.3 一个请求跨两端的完整时间线（重叠模式）

```text
P 端轮次:  k            k+1                    k+2                        k+3 ...
           │            │                      │                          │
           P1 入 bootstrap
           P3 握手完成→waiting
           P4 组批
           P6 提交 EXTEND ─── G(extend) ───►
                        P7 处理结果：首 token、
                           入 inflight、send_kv_chunk(last) ──RDMA──┐
                        P8 poll: Transferring                        │
                                               P8 poll: Success      │
                                               → 释放 KV、回包首 token│
                                                                     ▼
D 端轮次:  j ...                                    j'             j'+1          j'+2
                                                    D4 pop_transferred
                                                    →waiting_queue
                                                    D5 PREBUILT: 首 token 回包，
                                                       merge→DECODE[M]
                                                    D8 提交 DECODE[M] ── G ──►
                                                                   D8 提交 DECODE[M+1]
                                                                   D9 处理 M：第 2 个 token 回包
                                                                                 ...
```

> 注：握手（bootstrap）由 D 端 `add` 和 `pop_preallocated` 驱动，与 P 端 P1~P3 并行进行，图中做了简化。首 token 实际会通过 Router/网关，只经由其中一端返回给客户端；具体取决于部署方式，本文不展开。

---

## 5. 常见问题

**Q1：为什么 P 端没有 `is_disable_overlap_for_batch`？**
P 端连续批次都是 extend 批。如果要复用聚合版「连续 prefill 时关闭重叠」的逻辑，就必须每轮关闭重叠，重叠也就失去意义。P 端的 TTFT 主要取决于 KV 传输和 D 端接入，而不是延迟一拍处理结果，所以代码中没有这个开关。

**Q2：P 端重叠下，中间 chunk 的 KV 为什么不在 `process_prefill_chunk` 中直接发送？**
执行到那里时，上一 chunk 的前向可能还没有写完 KV，RDMA 读取会产生竞争。推迟到 `process_batch_result_disagg_prefill` 后，`copy_done.synchronize()` 已经保证该批 GPU 工作完成，此时再发送是安全的。

**Q3：D 端的 PREBUILT 为什么不放进 `result_queue` 走错位？**
PREBUILT 不跑模型，没有 GPU 结果需要等待。同步处理可以让首 token 尽早回包，并让请求立即参与本轮 DECODE，减少一拍延迟。唯一的 GPU 依赖是 `future_map` 槽位复用，已经通过 `schedule_stream.wait_stream(forward_stream)` 解决。

**Q4：`_apply_war_barrier` 会让 CPU 等待 GPU 吗？**
不会。它只是在 `schedule_stream` 上插入 `wait_event` / `wait_stream`，约束**后续 GPU 写操作**的执行顺序。CPU 真正可能阻塞的位置是 P7/D9 中的 `copy_done.synchronize()`，以及 `.tolist()` 等同步点。

**Q5：D 端 `polling_interval > 1` 会影响重叠吗？**
不会影响前向重叠，只会降低预分配和传输轮询的频率，减少跨 rank 通信开销；代价是 KV 到达后进入 `waiting_queue` 的时间最多会延迟 `polling_interval - 1` 轮。

**Q6：暂停引擎（`_engine_paused`）时，在途的前向结果会怎样？**
取决于 `pause_generation` 的模式（`scheduler.py:5416-5490`）：

- `in_place`：只设置 `_engine_paused=True`，不改动 `running_batch`、`last_batch`、`chunked_req` 或 `result_queue`。循环在 P2/D3 执行 `continue`，队列中的那一批结果会保留下来，恢复后在 P7/D9 中照常处理。
- `retract`：如果启用了重叠且 `last_batch` 非空，会**先 `popleft` 处理完在途结果**，然后回撤 running 请求，并把 `last_batch` 置为 `None`。D 端回撤的请求会进入 `hold_rebootstrap`，之后重新从 P 端获取 KV。P 端不会把上一批 extend 的请求合并到回撤列表中，因为它们已经在 inflight 队列里；也不会清空处于 chunk 中途的 `chunked_req`，因为如果 sender 仍然存活，释放它的 KV 会发送已被复用的页。
