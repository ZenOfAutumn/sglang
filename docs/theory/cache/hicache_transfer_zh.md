# HiCache 三级 KV 缓存传输机制详解（Device ↔ Host ↔ Storage）

> 本文系统讲解 SGLang **分层 KV 缓存（HiCache）** 中，KV 缓存如何在 **L1（GPU 显存）、L2（主机内存）、L3（外部存储）** 三级之间搬运，涉及哪些核心类、线程与方法，以及完整的端到端数据流。
>
> 配套整图见同目录 **`hicache_transfer_zh.drawio`**（用 draw.io / diagrams.net 打开），一张图覆盖 write / load-back / prefetch / backup / evict 全部传输路径。
>
> 所有描述均对账当前仓库代码：
> - 传输执行者：`python/sglang/srt/managers/cache_controller.py`（`HiCacheController`）
> - 调用方 / 调度侧：`python/sglang/srt/mem_cache/hiradix_cache.py`（`HiRadixCache`）
> - 主机侧 KV 池：`python/sglang/srt/mem_cache/memory_pool_host.py`
> - 相关阅读：`mem_cache/README_zh.md`、`docs/theory/cache/kv_cache_capacity_zh.md`

---

## 0. 一句话结论

**HiCache 把 KV 缓存组织成三级存储，`HiCacheController` 用「独立 CUDA stream + 后台线程」把 KV 在三级之间异步搬运，以扩展可缓存总量、提升前缀命中率，同时与模型前向计算重叠（overlap）以隐藏拷贝延迟。**

| 级别 | 介质 | 数据结构 | 定位方式 | 特点 |
| --- | --- | --- | --- | --- |
| **L1** | GPU 显存 | `mem_pool_device`（KVCache 物理张量） | `device_indices` | 最快、最稀缺 |
| **L2** | 主机内存（CPU） | `mem_pool_host`（`HostKVCache`） | `host_indices` | 容量更大、稍慢 |
| **L3** | 外部存储后端 | `storage_backend` | 逐页哈希键 | 容量最大、最慢，可跨进程/跨节点复用 |

---

## 1. 五条传输路径总览

HiCache 一共有 **5 条**核心数据流动路径，前四条是搬运、第五条是回收：

| # | 路径 | 方向 | 执行方式 | 入口方法 |
| --- | --- | --- | --- | --- |
| ① | **write / write-through** | L1 → L2 | 同步（`write_stream`） | `write()` / `start_writing()` |
| ② | **load-back** | L2 → L1 | 同步（`load_stream`，逐层事件） | `load()` / `start_loading()` |
| ③ | **prefetch** | L3 → L2 | 异步后台线程 | `prefetch()` / `prefetch_thread_func()` |
| ④ | **backup / write_storage** | L2 → L3 | 异步后台线程 | `write_storage()` / `backup_thread_func()` |
| ⑤ | **evict** | 释放 L1 / L2 | 同步 | `evict_device()` / `evict_host()` |

> 关键设计：**L1↔L2 是同步搬运**（用专用 CUDA stream 与前向计算并行），**L2↔L3 是异步后台线程搬运**（因存储 IO 慢且不确定，不能阻塞调度循环）。

---

## 2. 核心类与数据结构（`cache_controller.py`）

### 2.1 `HiCacheController`——总执行者

被 `HiRadixCache` 持有，统筹全部三级搬运。关键成员：

| 成员 | 作用 |
| --- | --- |
| `mem_pool_device` / `mem_pool_device_allocator` | L1 设备侧 KV 池及其分配器 |
| `mem_pool_host` | L2 主机侧 KV 池 |
| `storage_backend` | L3 外部存储后端（可运行时 attach/detach） |
| `write_stream` / `load_stream` | L1↔L2 专用 CUDA stream（与主计算流并发） |
| `write_queue` / `load_queue` | 待发起的写回 / 加载操作队列 |
| `ack_write_queue` / `ack_load_queue` | 已发起、等待事件完成的回执队列 |
| `prefetch_thread` / `backup_thread` | L2↔L3 异步后台线程 |
| `layer_done_counter` | 逐层加载完成计数器（overlap 关键） |
| `has_draft` / `mem_pool_*_draft` | 投机解码草稿 KV 池（搭车搬运） |

### 2.2 操作描述类

- **`CacheOperation`**：一次 **L1↔L2** 搬运的描述，含 `host_indices`、`device_indices`、`node_ids`、`priority`。可用 `merge_ops()` 合并多个操作为一次大批量拷贝。
- **`StorageOperation`**：一次 **L2↔L3** 存储操作的描述，含 `host_indices`、`token_ids`、`hash_value`（逐页哈希键）、`completed_tokens`（进度）。
- **`PrefetchOperation`**（继承 `StorageOperation`）：可中断的预取操作，用锁保护 `completed_tokens` 与终止标志，支持 `mark_terminate()` / `increment()`。

### 2.3 同步与重叠机制

- **`LayerLoadingEvent`**：为「逐层加载」维护一组 CUDA 事件（`load_events[i]` = 第 i 层加载完成）。消费者（前向）可**按层等待**——加载完一层就能算那一层，不必等整批加载完。
- **`LayerDoneCounter`**：管理多套（`num_counters=3`）`LayerLoadingEvent`，用生产者/消费者索引轮转，让相邻批次的加载/消费能并发而不互相覆盖。
- **`HiCacheAck`**：一次 write/load 的完成回执，携带 `(start_event, finish_event, node_ids)`，供调度器轮询完成后标记节点缓存状态。
- **`TransferBuffer`**：有界队列，把「缓冲准备」与「实际传输」解耦重叠以提升吞吐。

---

## 3. 路径①：write / write-through（L1 → L2）

**目的**：把 GPU 上算好的 KV 写回主机内存备份，为后续驱逐显存后仍可复用做准备。

**触发**（`hiradix_cache.py`）：`_inc_hit_count()` 中节点命中次数达到 `write_through_threshold`（`write_through` 策略为 1，其它为 2）时调用 `write_backup()`。

**流程**（`cache_controller.py`）：

```
write_backup(node)                       # HiRadixCache
  └─ controller.write(device_indices)    # 在 host 池 alloc 落点，失败返回 None
       └─ 入 write_queue，调用 start_writing()
            └─ merge_ops 合并队列 → move_indices 调整索引布局
               └─ 在 write_stream 上：backup_from_device_all_layer（整批逐层拷贝）
                  └─ record_stream 保护索引张量生命周期
                  └─ HiCacheAck(start,finish,node_ids) 入 ack_write_queue
```

**完成收割**：`writing_check()` 每步轮询 `ack_write_queue` 里的 `finish_event`，完成后把对应节点标记为 `backuped`。

> `write_policy` 三种：`write_through`（首次命中即写）、`write_through_selective`（默认，更保守，减少写放大）、`write_back`（仅在驱逐 L1 前才回写 host）。

---

## 4. 路径②：load-back（L2 → L1）

**目的**：把 host 上备份的命中前缀回载到 GPU，使其重新可用于计算。

**触发**：`match_prefix()` 计算出 `host_hit_length`（device 已淘汰但 host 仍有备份的前缀长度）后，调度器经 `init_load_back()` → `load_back()` 发起；仅当 host 命中长度 ≥ `load_back_threshold`(=10) 才值得回载。

**流程**：

```
load_back(node) → controller.load(host_indices)   # 在 device 池 alloc 落点，失败返回 None
  └─ 入 load_queue，返回 device_indices
     └─ start_loading()：
        · update_producer() 轮转占用一套逐层事件（返回 producer_id）
        · 在 load_stream 上逐层：load_to_device_per_layer(i)
          → 每层 producer_event.complete(i) 打点
        · ack_load_queue 登记回执，返回 producer_id
```

**逐层重叠**：消费者（前向）用 `set_consumer(producer_id)` + `wait_until(layer)` **按层等待**，实现加载与计算流水线重叠。**完成收割**由 `loading_check()` 负责。

---

## 5. 路径③：prefetch（L3 → L2，异步）

**目的**：从外部存储预取命中的前缀到 host，供后续 load-back 到 GPU 复用。

**触发**：`match_prefix()` 发现 storage 可能命中且未被限流（`prefetch_rate_limited()`）时，经 `prefetch()` 投入 `prefetch_queue`。

**后台线程流程**（`prefetch_thread_func`）：

```
1. _storage_hit_query()：逐页算哈希键（链式），batch_exists 探测「前缀连续命中」页数
2. 跨 rank all_reduce(MIN) 对齐命中页数（存储按 rank 分片/复制，必须一致）
3. 命中 < prefetch_threshold → revoke 撤销，append_host_mem_release 归还 host 内存
4. 否则裁剪 host_indices/hash_value 到命中范围 → 投入 prefetch_buffer
5. prefetch_io_aux_func 从 buffer 取出 → _page_transfer()：
   · 分批 STORAGE_BATCH_SIZE 传输
   · 若 has_draft：先读 draft L3（避免竞态），再读 target
   · page_get_func：_generic_page_get（待废弃）/ _page_get_zero_copy（batch_get_v1 零拷贝）
```

> **命中必须连续**：前缀缓存复用要求哈希链连续命中，一旦某页缺失即停止。
>
> `prefetch_stop_policy`：`best_effort`（尽力而为）/ `wait_complete`（等待完成）/ `timeout`（超时终止）。

---

## 6. 路径④：backup / write_storage（L2 → L3，异步）

**目的**：把 host 上的 KV 进一步持久化到外部存储，供跨请求/跨进程复用。

**触发**：`write_backup()` 中若 `enable_storage`，则对已在 host 的 KV 调用 `write_backup_storage()` → `controller.write_storage()`，投入 `backup_queue`。

**后台线程流程**（`backup_thread_func` → `_page_backup`）：

```
· 分批 STORAGE_BATCH_SIZE 写 target 页（page_set_func：_generic_page_set / _page_set_zero_copy）
· 写失败则告警并中止（暂不支持部分成功）
· 若 has_draft：尽力附带写 draft 页（_draft_page_set）
· MLA 模型下非 rank0 跳过实际写出（backup_skip，避免重复写相同 latent）
· 完成后放入 ack_backup_queue
```

**完成收割**：`drain_storage_control_queues()` 排空 `ack_backup_queue`，释放对应 host 锁。

---

## 6.5 什么是「零拷贝（zero-copy）」

在 prefetch（③）和 backup（④）里，你会看到成对的两套接口：`_generic_page_get/set`（待废弃）与 `_page_get/set_zero_copy`（零拷贝）。它们的差别就在于**数据在 L2 host 池与 L3 存储后端之间搬运时，中途要不要多经过一块临时缓冲区**。

### 6.5.1 核心思想

**零拷贝 = 让存储后端直接读写 host 池的目标槽位，省掉「先落到中间张量、再拷进 host 池」这一次多余的内存拷贝。**

- host 池（`mem_pool_host`）本身就是一块**预分配好、地址固定**的大内存。每页 KV 在其中的落点由 `host_indices` 精确定位。
- 既然目的地地址已知，就没必要让后端先把数据读到一个临时 tensor、再由 Python 侧 `memcpy` 搬进 host 池——**直接把 `host_indices` 交给后端，让它把数据一步到位写进最终位置**即可。

### 6.5.2 两种接口对照（以预取读取为例）

| 维度 | 旧接口 `batch_get`（`_generic_page_get`） | 零拷贝 `batch_get_v1`（`_page_get_zero_copy`） |
| --- | --- | --- |
| 传给后端的东西 | 一批**临时空张量**（`get_dummy_flat_data_page()`） | 目标 **`host_indices`**（host 池真实槽位） |
| 后端返回 | `List[torch.Tensor]`（读回的数据） | `List[bool]`（每页是否成功） |
| 落库方式 | 调用方再逐页 `set_from_flat_data_page()` **拷贝**进 host 池 | 后端**直接写入** host 池槽位，无需再拷 |
| 内存拷贝次数 | 至少 1 次额外的中转拷贝 | 0 次额外拷贝 |

用代码对照更直观：

```python
# 旧接口：后端 → 临时张量 → 再拷进 host 池（多一次拷贝）
page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
for i in ...:
    self.mem_pool_host.set_from_flat_data_page(host_indices[...], page_data[i])  # ← 额外拷贝

# 零拷贝：直接把 host_indices 给后端，一步到位写入
results = self.storage_backend.batch_get_v1(hash_values, host_indices, extra_info)  # ← 无中转
```

备份写出（④）对称：`_generic_page_set` 先 `get_data_page()` 取出数据再 `batch_set`；而 `_page_set_zero_copy` 直接把 `host_indices` 交给 `batch_set_v1`，让后端从 host 池原地读走。

### 6.5.3 为什么值得

- **省内存带宽**：KV 页动辄几十上百 KB，一次预取几十上百页，省掉的中转拷贝在高吞吐下相当可观。
- **省临时内存与 GC**：不再为每页分配临时张量。
- **更低延迟**：少一次同步拷贝，IO 路径更短。

> 注意：零拷贝需要后端支持 v1/v2 接口（如 `mooncake` 支持零拷贝 v2）。不支持的后端仍回退到 `_generic_*` 路径，所以两套实现会并存一段时间（旧接口标记为 `todo: deprecate`）。

---

## 7. 路径⑤：驱逐（释放 L1 / L2 容量）

- **`evict_device(device_indices)`**：`allocator.free` 释放 L1 GPU 槽位。`write_back` 策略下，驱逐 L1 前会先 `write_backup(write_back=True)` 把 KV 回写到 host。
- **`evict_host(host_indices)`**：`mem_pool_host.free` 释放 L2 主机槽位（目前仅支持 `backup_only` 策略）。

---

## 8. `io_backend`：L1↔L2 索引布局适配

`move_indices()` 按 `io_backend` 调整索引张量的所在设备与布局：

| io_backend | 处理 |
| --- | --- |
| `kernel` | 用自定义 kernel 拷贝，`host_indices` 需移到 GPU |
| `direct` | 直接按索引拷贝；`layer_first` 布局需排序、`device_indices` 移到 CPU；`page_first_direct` 布局 `device_indices` 移到 CPU |
| `kernel_ascend` | 昇腾 kernel 路径，`device_indices` 移到 CPU |

---

## 9. 端到端典型链路（把五条路径串起来）

```
首次见到某前缀
  → GPU 计算 KV（L1）
  → ① write-through 写回 Host（L2）
  → ④ backup 备份到 Storage（L3）
  → 显存不足时 ⑤ evict_device 驱逐 L1（KV 仍在 L2/L3）

再次命中同一前缀
  → match_prefix：device 未命中，但 host_hit_length>0 或 storage 命中
  → ③ prefetch：从 Storage 预取到 Host（若仅 L3 有）
  → ② load-back：从 Host 回载到 GPU
  → 复用前缀，跳过重算
```

**跨线程协作枢纽**：`HiRadixCache.check_hicache_events()` 在**每步调度循环**调用：

- `writing_check()`：收割 ① 写回完成 → 标记节点 `backuped`
- `loading_check()`：收割 ② 回载完成 → 解锁末端节点
- `drain_storage_control_queues()`：排空 ③ 的 revoke、④ 的 backup-ack、host 内存释放三类控制消息（跨 rank MIN 对齐，减少同步次数）

---

## 10. 投机解码草稿（draft）KV 的搭车搬运

启用投机解码时存在 draft + target 两套 KV 池。HiCache 让 **draft KV 搭车在 target 的 L2/L3 操作上一起搬运**：

- **L1↔L2**：`start_writing` / `start_loading` 中若 `has_draft`，对 draft 池同样做 `backup_from_device_all_layer` / `load_to_device_per_layer`。
- **L2↔L3**：按后端选择实现——`mooncake` 用零拷贝 v2（`_draft_page_*_v2`）；通用后端用 `{hash}.draft` 键区分（`_draft_page_*_generic`）；`hf3fs/eic/nixl/simm` 暂不支持。
- **关键顺序**：预取时**先读 draft、再发布 target 完成**，否则 `wait_complete` 可能在 draft KV 到达前就把 target 加载回来，造成不一致。

---

## 11. 小结

1. HiCache 三级：**L1(GPU) ↔ L2(Host) ↔ L3(Storage)**，由 `HiCacheController` 统一搬运。
2. **五条路径**：① write（L1→L2）、② load-back（L2→L1）、③ prefetch（L3→L2）、④ backup（L2→L3）、⑤ evict（释放）。
3. **L1↔L2 同步**（专用 CUDA stream + 逐层事件，与前向 overlap）；**L2↔L3 异步**（后台线程，因存储 IO 慢）。
4. `HiRadixCache` 是调用方，用 `match_prefix` 决策、`check_hicache_events` 每步收割完成事件与控制消息。
5. 投机解码 draft KV 搭车在 target 操作上一起搬运，注意「先 draft 后 target」的顺序以避免竞态。

> 配套整图：`hicache_transfer_zh.drawio`（一张图覆盖全部五条路径、核心类、线程与策略）。

