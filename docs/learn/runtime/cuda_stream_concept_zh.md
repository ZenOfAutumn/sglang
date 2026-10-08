# CUDA Stream（流）：概念与 SGLang 中的用法

> 本文回答三件事：**stream 是什么**、**为什么要用多条流**、**多流之间怎么同步**，然后把 SGLang 里实际用到的流列全。
>
> 与 `scheduler_subprocess_startup_zh.md` 的关系：那篇 §4.1 讲 `schedule_stream` / `forward_stream` 在启动链里怎么诞生、各步耗时；本文先把 stream 本身讲清楚，再给出完整的流清单与依赖关系。
>
> 代码路径相对仓库根目录（`sglang/`）。标注「部署树」的条目指 `launch.sh` 实际加载的那份源码（`SGLANG_SOURCE` 下的 `.../main-20260926/baseline/sglang`），工作区快照尚未包含。

---

## 0. 一页速览

| 问题 | 一句话答案 |
| --- | --- |
| stream 是什么 | GPU 上的**任务队列**：同一 stream 内严格 FIFO 串行，不同 stream 之间可并发 |
| 谁在排队 | kernel、memcpy（H2D / D2H / D2D）、event 的 record / wait |
| 「当前流」 | `with torch.cuda.stream(s):` 块内，所有**未显式指定流**的 GPU 操作都排到 `s` |
| 怎么同步 | `Event.record()` + `wait_event()`（细粒度），或 `wait_stream()`（等待对方全部已入队工作） |
| `synchronize()` | 阻塞 **CPU** 直到该流跑完——不要放在热路径 |
| 为什么要多流 | 让「计算」与「拷贝 / 通信 / 调度端的 GPU 准备」**重叠**，把串行变并行 |
| SGLang 主循环几条 | **3 条**：`schedule_stream` + `forward_stream` + `copy_stream`（非 overlap 时前两条是同一条）；PP 再加 `pp_comm_stream` |

---

## 1. stream 是什么

### 1.1 基本模型

一次 kernel 下发（launch）对 CPU 来说是**异步**的：CPU 只是把任务塞进某条流的队列就返回，GPU 自己按队列执行。因此「排队顺序」和「执行顺序」是同一个东西：

- **同一条流内**：FIFO，前一个任务没做完，后一个不会开始；
- **不同流之间**：没有任何顺序，硬件有多余资源就并发跑。

```
streamA: [K1 ][K2      ][K3]
streamB:      [K4 ][K5        ]     ← 与 A 无依赖时可与 A 并发
CPU:     launch…launch…launch…（入队即返回，不等待）
```

### 1.2 「当前流」与 stream context

PyTorch 里每条流是一个 `torch.cuda.Stream`（设备无关 API 为 `torch.get_device_module(dev).Stream()`）。**不显式指定流的算子会排到「当前流」**，而当前流由上下文管理器切换：

```python
with torch.cuda.stream(s):      # 块内所有 GPU 操作 → 流 s
    tensor.copy_(other)         # 这个 D2H 拷贝排在 s 上
```

SGLang 用 `self.device_module.stream(...)` 构造 context（等价物），把「一批工作」整体挂到某条流上。

- **默认流（default / legacy stream）**：进程启动时当前流就是它；它语义上会与其他流隐式同步（除非用 per-thread default stream），所以**多流重叠要把工作显式挪出默认流**。
- **优先级**：`Stream(priority=0)` 是默认优先级；数值越小（越负）越优先。SGLang 建调度流/前向流都用 `priority=0`，不靠优先级，只靠「分流的语义」来重叠。

### 1.3 同步原语

| 原语 | 语义 | 代价 |
| --- | --- | --- |
| `ev = Event(); ev.record(stream=a)` | 在流 `a` 的「当前位置」打一个标记 | 极低 |
| `b.wait_event(ev)` | 流 `b` 等到该标记完成才继续 | 低（细粒度） |
| `b.wait_stream(a)` | 流 `b` 等 `a` **当前已入队的全部工作** | 中（粗粒度，容易把无关工作也拖进来） |
| `stream.synchronize()` | 阻塞 CPU 直到该流清空 | 高，**热路径禁用** |
| `torch.cuda.current_stream()` | 取当前流，用于「在别的流前插依赖」 | — |

> 经验法则：**能用一个 event 表达的最小依赖，就不要用 `wait_stream`**。SGLang 在统一内存池场景下就是这么做的：能用 `shared_read_done_event` 就不用整条前向的 fence。

### 1.4 跨流共享显存的两个必踩坑

1. **读-写竞争（RAW / WAR）**：流 B 要写的 buffer，可能是流 A（前向）正在读的。必须让 B 等 A 读完——SGLang 用 WAR 屏障（见 §4.2）。
2. **张量提前释放**：PyTorch 的 caching allocator 按流记账。跨流使用时，若分配流已「用完」而使用流还在跑，显存可能被回收再分配给别人。需要 `tensor.record_stream(other_stream)` 或**引用保活**（SGLang 用 `batch_record_buf` 环形缓冲钉住最近两轮的 GPU 张量）。

---

## 2. 为什么要多流

单流模型下，GPU 的时间线是一条串行链：算 → 拷 → 算 → 拷。而这些阶段用的硬件单元并不相同（SM / Copy Engine / NVLink / PCIe），串行执行等于把资源浪费在互相等待上。多流的目标就是**把这些阶段拉到不同流上重叠**：

| 重叠对 | 收益 | SGLang 中的实现 |
| --- | --- | --- |
| 调度（KV 分配/拷贝、cache、输入准备） vs 前向计算 | 隐藏 CPU 侧与调度侧开销 | `schedule_stream` vs `forward_stream` |
| 结果 D2H 拷贝 vs 下一轮前向 | 结果回传不阻塞计算 | `copy_stream` |
| PP 张量通信 vs 计算 | 掩盖 send/recv 延迟 | `pp_comm_stream`（部署树，`SGLANG_PP_COMM_OVERLAP=1`） |
| KV 加载/落盘 / 权重搬运 vs 计算 | 隐藏 IO | cache / storage 侧的功能流（见 §4.5） |
| 模型内部子模块 vs 主路径（如视觉塔、MoE 通信） | 隐藏子模块延迟 | 各模型的 `alt_stream` |

---

## 3. SGLang 里的流全景

### 3.1 调度器主循环：3 条（+1）

| 流 | 归属 / 创建点 | 主要工作 |
| --- | --- | --- |
| `schedule_stream` | `Scheduler`，`init_overlap` 内 | 调度侧的 GPU 准备与收尾：KV 池分配/释放、KV 索引写入、radix cache match/insert 与 KV 拷贝（v2p/p2v remap、compaction）、批输入准备、结果处理；整个 event loop 都跑在它的 context 里 |
| `forward_stream` | 来自 `model_runner.forward_stream`（`ModelRunner` 初始化时建） | 模型前向 kernel 的主体 |
| `copy_stream` | `Scheduler`，无条件创建 | 结果 D2H 拷贝，让它与下一轮前向重叠 |
| `pp_comm_stream` | **部署树** `scheduler_pp_mixin.py` | PP 张量 send/recv 专用流，`SGLANG_PP_COMM_OVERLAP=1` 且 CUDA 时才建 |

```1873:1875:python/sglang/srt/managers/scheduler.py
        self.schedule_stream = self.device_module.Stream(priority=0)
        if self.device == "cpu":
            self.schedule_stream.synchronize = lambda: None  # No-op for CPU
```

```1652:1658:python/sglang/srt/managers/scheduler.py
        self.forward_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.forward_stream
        )
        self.copy_stream: CudaStream = self.device_module.Stream()
        self.copy_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.copy_stream
        )
```

```478:478:python/sglang/srt/model_executor/model_runner.py
        self.forward_stream = torch.get_device_module(self.device).Stream()
```

注意 `forward_stream` 是**从 ModelRunner 借来的**：非 overlap 且无 PP 时，`init_model_worker` 里会让 `forward_stream = schedule_stream`（二者合一，退回单流语义）；overlap 或 PP>1 时才真正分成两条。

```494:496:python/sglang/srt/managers/scheduler.py
        self.enable_overlap = (
            not get_schedule().disable_overlap_schedule and not use_mlx()
        )
```

即**重叠调度默认开启**，用 `--disable-overlap-schedule` 关闭。

### 3.2 流的依赖与 WAR 屏障

`schedule_stream` 是 producer，`forward_stream` 是 consumer；反向还要一个 WAR 屏障防止「下一批的写」抢在「上一轮前向的读」之前：

```1895:1906:python/sglang/srt/managers/scheduler.py
    def _apply_war_barrier(self):
        # WAR: keep later schedule_stream writes behind this forward's shared reads.
        # Clearing matters: a phase that skips the publish then falls back to coarse.
        if not self._war_barrier_enabled:
            return
        runner = self.model_worker.last_shared_read_runner
        ev = runner.shared_read_done_event
        runner.shared_read_done_event = None
        if ev is not None and not envs.SGLANG_FORCE_COARSE_WAR_BARRIER.get():
            self.schedule_stream.wait_event(ev)
        else:
            self.schedule_stream.wait_stream(self.forward_stream)
```

`copy_stream` 是**叶子流**：它等 `forward_stream`，但没有任何人等它（只由 `copy_done` event 收尾），因此天然与前向重叠：

```4370:4378:python/sglang/srt/managers/scheduler.py
                                # Result D2H on copy_stream overlaps the next forward
                                # instead of serializing on forward_stream; it's a leaf
                                # gated by copy_done, so nothing on forward_stream waits.
                                self.copy_stream.wait_stream(self.forward_stream)
                                with self.copy_stream_ctx:
                                    batch_result.copy_to_cpu(
                                        return_logprob=batch.return_logprob,
                                        return_hidden_states=batch.return_hidden_states,
                                    )
```

依赖关系图：

```
    (CPU 选批 / 准备，在 schedule_stream 上下文内入队)
                 │
                 ▼
   ┌────────────────────────────────────────────┐
   │ schedule_stream：KV 分配/拷贝、cache、输入   │
   │ 准备、结果处理；跑整个 event loop           │
   └───────────────┬────────────────────────────┘
                   │ forward_stream.wait_stream(schedule_stream)
                   ▼
   ┌────────────────────────────────────────────┐
   │ forward_stream：模型前向                     │
   └───────┬────────────────────────────┬───────┘
           │ record(shared_read_done)    │ wait_stream
           ▼                             ▼
   ┌───────────────────────┐   ┌───────────────────────────┐
   │ schedule_stream 的 WAR │   │ copy_stream：结果 D2H      │
   │ 屏障（等前向读完再写） │   │ （叶子流，无人等它）       │
   └───────────────────────┘   └───────────────────────────┘
```

### 3.3 坑：CUDA 流来自固定池，`Stream()` 可能「别名」

CUDA/HIP 的 `Stream()` 不是无限新建，而是从一个小型 round-robin 池里取。这意味着**新取的流可能和已有的流是同一条**（`s.stream` 句柄相同），重叠直接失效。所以 SGLang 启动时会检查并重画：

- 工作区版：`while` 循环重取，直到 `schedule_stream.cuda_stream != forward_stream.cuda_stream`；
- **部署树**额外抽了个工具 `allocate_distinct_stream(device_module, avoid_streams)`（`managers/utils.py`），PP 的 `pp_comm_stream` 就用它避开 `schedule_stream` / `forward_stream` / `copy_stream` 三条。

> 只有 CUDA/HIP 暴露 `cuda_stream` 句柄，NPU/XPU 直接跳过这个别名检查。

### 3.4 组件级流（归 Scheduler 的子组件）

| 位置 | 流 | 用途 |
| --- | --- | --- |
| `srt/managers/overlap_utils.py` | `fwd_prepare_d2h_stream` | 重叠路径下把「前向准备」的 device→host 拷贝挪出主路径 |
| `srt/managers/hisparse_coordinator.py` | `write_staging_stream`、`decode_backup_stream`、`prefetch_stream` | HiSparse：暂存写、decode 备份、KV 预取与计算重叠 |
| `srt/managers/cache_controller.py` | 自建流 | 层级 cache 的离线加载与在线前向同步 |

### 3.5 model / cache / storage 级（不归主循环）

| 位置 | 流 | 用途 |
| --- | --- | --- |
| `srt/models/*.py` | `alt_stream`（llama4、nemotron_h、inkling、grok、giga-chat…；`kimi_k3` 有 3 条；`deepseek_v4_dspark` 的 multi-stream） | 模型内部的旁路并发（MoE 通信、视觉塔等） |
| `srt/models/qwen3_vl.py` | `copy_stream` | 视觉塔张量搬运 |
| `srt/mem_cache/storage/flexkv/flexkv_radix_cache.py` | `load_stream`、`store_stream` | KV 换入/换出与计算重叠 |
| `srt/mem_cache/storage/umbp/*`（部署树） | `_offload_index_stream` | UM 卸载索引维护 |
| `srt/utils/cuda_vmm_transport_utils.py` | `_publish_stream`、`_recycle_stream` | VMM 页发布/回收 |
| `srt/speculative/dspark_components/dspark_observability.py` | `_d2h_stream` | 观测数据异步回传 |

### 3.6 特例：MLX / CPU / HIP

- **MLX（Apple）**：不用 CUDA 流，靠 `mx.async_eval` 做 CPU/GPU 重叠，只建 `result_queue`；
- **CPU 后端**：`schedule_stream.synchronize` 被替换成 no-op（`scheduler.py:1875`）；
- **HIP（ROCm）**：`copy_to_cpu` 走同步路径——注释明确写了「跨流同步的开销大于它要重叠的那点 D2H」。

---

## 4. 相关开关

| 开关 | 默认 | 作用 |
| --- | --- | --- |
| `--disable-overlap-schedule` | 关闭（即重叠默认开） | 关掉后 `enable_overlap=False`，`forward_stream = schedule_stream`，退回单流 |
| `SGLANG_PP_COMM_OVERLAP`（部署树） | `False` | 为 PP 建独立的 `pp_comm_stream` 做通信/计算重叠 |
| `SGLANG_ENABLE_WAR_BARRIER` | `False`（CUDA 上强制开） | 调度侧写入统一内存池前等前向读完 |
| `SGLANG_FORCE_COARSE_WAR_BARRIER` | `False` | 屏障降级为粗粒度 `wait_stream(forward_stream)` |

---

## 5. 排查多流问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 开了重叠但没提速 | 两条流被 CUDA 池「别名」成同一条 | 看启动日志的流不别名重画；NPU/XPU 无此检查 |
| 偶发数值错 / 花屏 | 跨流 RAW/WAR 或张量提前释放 | 检查 `wait_event` 是否漏、是否需要 `record_stream`/引用保活 |
| 想复现「串行是否变慢」 | — | `--disable-overlap-schedule` 对比 |
| 定位 kernel 落在哪条流 | — | `nsys profile` / `torch.profiler` 看 stream id；`CUDA_LAUNCH_BLOCKING=1` 可临时串行化 |
