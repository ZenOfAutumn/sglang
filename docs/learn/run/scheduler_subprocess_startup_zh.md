# Scheduler 子进程启动详解

> 本文是 `docs/learn/run/http_server_startup_zh.md` 第 3.2 节「Scheduler 子进程内部启动」的**展开版**：原文只保留启动链路上的必要顺序，这里把每个环节展开到代码级细节。
>
> 不重复的内容仍在原文：父进程侧的进程编排（`_launch_subprocesses` / `SubprocessWatchdog`）、Detokenizer/TokenizerManager 启动、HTTP 装配（`_setup_and_run_http_server` / `lifespan`）、预热与就绪判定（`_wait_and_warmup` / `/health`）。
>
> 代码块路径均相对仓库根目录（`sglang/`），行号对应当前版本。
>
> 每个代码块前的「**代码归属**」统一标注 `源文件 · 类 → 方法`；模块级函数会显式注明「无类」。

---

## 0. 一页速览

Scheduler 是**唯一持有并驱动 GPU 的进程**，每个 TP rank 一个。它的启动可以概括成六步：

```
父进程 mp.Process(target=run_scheduler_process, args=(..., pipe_writer))
        │
        ▼
① 进程入口      run_scheduler_process        load_plugins → publish → configure → tracing
        ▼
② 进程级配置    configure_scheduler_process 父死子死 / 进程名 / faulthandler / logger / 亲和性 / NUMA
        ▼
③ 构造          Scheduler.__init__           一串 init_*（编排器风格，顺序有强依赖）
        ▼
④ 握手          pipe_writer.send(get_init_info())   ← 父进程 _wait_for_scheduler_ready 在此收
        ▼
⑤ 事件循环      run_event_loop → dispatch_event_loop   schedule_stream + 选具体 loop，阻塞运行
        ▼
⑥ 退出清理      finally: metrics_reporter._shutdown_fpm / release_host_resources
```

关键约束（后文逐条解释）：

| 约束 | 原因 |
| --- | --- |
| `init_soft_watchdog` 必须在 `__init__` **最前面** | 看门狗守护线程首个 tick 就读 `forward_ct` / `cur_batch_for_debug` |
| `is_initializing` 覆盖整个构造期 | 构造期看门狗判活语义、`dump_info` 是否输出现场都依赖它 |
| KV cache 构建在 **模型 worker 之后** | 显存被权重占掉多少，决定了 KV 池能开多大 |
| `maybe_init_rust_server` 在 `init_disaggregation` 与 `init_request_receiver` **之前** | Rust server 上的 KV bootstrap registry 要被 KVManager 同步注册；接收器要读它选 ingress |
| 握手发生在**事件循环之前** | 若先进 loop 再握手，父进程的 pipe 超时判定会失去意义 |

---

## 1. 进程入口：`run_scheduler_process`

**代码归属**：`python/sglang/srt/managers/scheduler.py` · 模块级函数 `run_scheduler_process`（无类，子进程入口）

```5808:5889:python/sglang/srt/managers/scheduler.py
    # 载荷：加载插件 -> 配置进程（日志/进程名/CPU 亲和性）-> 可选初始化 tracing ->
    #       创建 Scheduler 实例 -> 通过 pipe 把初始化信息回传父进程 -> 进入事件循环。
    load_plugins()
    # Publish before anything in this process reads configuration.
    publish(server_args, role="scheduler")
    dp_rank = configure_scheduler_process(...)
    parent_process = psutil.Process().parent()

    if get_observability().enable_trace:
        process_tracing_init(...)
        trace_set_thread_info(thread_label, tp_rank, dp_rank, pp_rank)

    scheduler = None
    try:
        scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank,
                              moe_ep_rank, pp_rank, attn_cp_rank, moe_dp_rank, dp_rank)
        pipe_writer.send(scheduler.get_init_info())
        scheduler.run_event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
        if envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION.get():
            os.killpg(os.getpgrp(), signal.SIGKILL)
    finally:
        if scheduler is not None:
            scheduler.metrics_reporter._shutdown_fpm()
            if scheduler.gracefully_exit:
                scheduler.release_host_resources()
```

逐步说明（`scheduler.py:5793` 起）：

1. **`load_plugins()`**：插件钩子可以覆盖 `Scheduler` 及其依赖，所以必须在**读取任何配置之前**加载。
2. **`publish(server_args, role="scheduler")`**：把 `server_args` 发布到本进程的 runtime context，之后代码统一走 `get_serving()` / `get_schedule()` / `get_parallel()` / `get_disagg()` 等访问，而不是层层传参。
3. **`configure_scheduler_process(...)`**：进程级配置，返回规范化后的 `dp_rank`（见第 2 节）。
4. **`parent_process = psutil.Process().parent()`**：提前拿到父进程句柄，异常路径上要给它发 `SIGQUIT`（让它别继续等一个已死的子进程）。
5. **tracing**：`enable_trace` 时初始化 OTLP；`thread_label` 会按 PD 角色变成 `Prefill Scheduler` / `Decode Scheduler`，并带上 `(tp_rank, dp_rank, pp_rank)` 便于在 trace 里区分 rank。
6. **构造 + 握手 + 事件循环**：注意 `Scheduler(...)` 的参数顺序是 `(server_args, port_args, gpu_id, tp_rank, moe_ep_rank, pp_rank, attn_cp_rank, moe_dp_rank, dp_rank)`——与 `__init__` 的形参顺序（`moe_ep_rank` 在前、`attn_cp_rank` 在后）**不同**，改签名时要一起改。
7. **异常路径**：打日志 → 给父进程发 `SIGQUIT`；可选地 `killpg` 整个进程组（`SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION=1`），避免兄弟 rank 在 NCCL/TCPStore 上刷几千行回溯后才一起死。
8. **`finally`**：
   - `metrics_reporter._shutdown_fpm()`：FPM（指标上报）有后台 ZMQ 发布线程，需要显式收尾来冲刷队列并干净关 socket；
   - `release_host_resources()` **只在优雅退出路径**调用（`gracefully_exit`）——异常路径上 GPU 可能已经卡死，`destroy()` 里的 `synchronize()` 自己就会 hang。

---

## 2. 进程级配置：`configure_scheduler_process`

**代码归属**：`python/sglang/srt/managers/scheduler.py` · 模块级函数 `configure_scheduler_process`（无类；返回规范化后的 `dp_rank`）

```5725:5790:python/sglang/srt/managers/scheduler.py
    # 中译：调度器 worker 的日志与进程标题配置；display_* 仅用于显示，运行时 rank 保持局部值。
    kill_itself_when_parent_died()

    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        # [For Router] 走 router 时由环境变量指定 dp_rank
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    shown_dp = display_dp_rank if display_dp_rank is not None else dp_rank
    shown_tp = display_tp_rank if display_tp_rank is not None else tp_rank

    prefix = ""
    if shown_dp is not None:      prefix += f" DP{shown_dp}"
    if pp_size > 1:               prefix += f" PP{pp_rank}"
    if attn_cp_size > 1:          prefix += f" ATTN_CP{attn_cp_rank}"
    if moe_dp_size > 1:           prefix += f" MOE_DP{moe_dp_rank}"
    if tp_size > 1:               prefix += f" TP{shown_tp}"
    if ep_size > 1:               prefix += f" EP{shown_moe_ep}"

    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    if envs.SGLANG_SET_CPU_AFFINITY.get():
        set_gpu_proc_affinity(pp_size, tp_size, nnodes, gpu_id)
    if not envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = get_numa_node_if_available(server_args, gpu_id)
        if numa_node is not None:
            numa_bind_to_node(numa_node)
    return dp_rank
```

要点：

| 动作 | 说明 |
| --- | --- |
| `kill_itself_when_parent_died()` | 父进程死亡时子进程自尽，避免孤儿进程霸占 GPU |
| `SGLANG_DP_RANK` 环境变量 | router / 外部编排场景下 `dp_rank` 不在启动参数里，靠环境变量补齐 |
| `display_*_rank` | **纯显示用**（跨节点时把全局 rank 打进进程名/日志前缀），运行逻辑仍用局部 rank |
| `prefix` 拼装 | 只在对应并行维度 `size > 1` 时才出现，所以单卡启动的进程名就是干净的 `sglang::scheduler` |
| `faulthandler.enable()` | 段错误 / 卡死时能打出 Python 栈，配合 `SIGQUIT` 排查 |
| `configure_logger` + `suppress_other_loggers` | 给本进程日志加 rank 前缀，并压掉第三方库噪声 |
| CPU 亲和性 / NUMA | `SGLANG_SET_CPU_AFFINITY=1` 时把进程绑到对应 GPU 邻近的核；`SGLANG_NUMA_BIND_V2` 打开时改由别的机制绑定（这里跳过） |

---

## 3. `Scheduler.__init__`：编排器风格

`__init__`（`scheduler.py:448`）刻意保持「编排器」风格，注释里写死了这条代码风格约定：

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler`（`:423`）→ `Scheduler.__init__`（代码风格约定注释）

```461:464:python/sglang/srt/managers/scheduler.py
        # NOTE: KEEP THE FOLLOWING CODE STYLE for this function:
        # Keep __init__ as an orchestrator: sequence init_* and maybe_init_* calls
        # with minimal glue. Move substantial component-specific logic into
        # dedicated methods instead of adding inline blocks here.
```

也就是说：**新逻辑请写成 `init_xxx()` 方法**，不要在 `__init__` 里塞大段内联代码。

### 3.1 前置三件套（顺序不可动）

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.__init__`（前置三件套）

```465:472:python/sglang/srt/managers/scheduler.py
        self.init_startup_timing_begin()
        self.is_initializing = True
        # init_soft_watchdog starts a daemon thread that reads these on its first tick.
        # 中译：看门狗守护线程一启动就会读取 forward_ct / cur_batch，所以必须先置好初值。
        self.forward_ct: int = 0
        self.cur_batch_for_debug: Optional[ScheduleBatch] = None
        self.init_soft_watchdog()
```

- `init_startup_timing_begin()` 记下 `perf_counter()` 起点，构造最后用它算 `scheduler_e2e`。
- `is_initializing = True` 在整个构造期保持为 `True`；看门狗判活与 `dump_info` 都读它。
- `forward_ct` / `cur_batch_for_debug` **必须先手工初始化**，因为 `init_soft_watchdog` 会立刻起一个 daemon 线程，线程首个 tick 就读这两个属性——晚一步就是「未定义属性」崩溃。

### 3.2 软看门狗

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.init_soft_watchdog`

```1406:1410:python/sglang/srt/managers/scheduler.py
    def init_soft_watchdog(self):
        if (x := get_device().soft_watchdog_timeout) is not None:
            self.soft_watchdog = create_scheduler_watchdog(
                self, watchdog_timeout=x, soft=True
            )
```

- `soft_watchdog_timeout` 默认 `None` → **默认不创建**，方法直接返回。
- `soft=True`：超时只 dump 诊断信息，**不发 SIGQUIT、不杀进程**；与 `init_watch_dog_memory_saver_input_blocker` 里的硬看门狗（默认 300s、超时 crash）形成对照。
- 判活靠 `scheduler.forward_ct`：主循环每轮喂狗使计数增长，超时未增长即判定卡死。
- 现场信息在 `create_scheduler_watchdog`（`scheduler_components/invariant_checker.py:522`，模块级函数，返回 `WatchdogRaw`——即 `srt/utils/watchdog.py:145` 的看门狗底层实现类，软/硬看门狗都走它）的 `dump_info` 里生成：`cur_batch_for_debug.batch_size()` / `.reqs` + `invariant_checker._check_all_pools()` 的各池状态。构造期 `is_initializing=True` 时直接返回空串（此时还没有有效 batch 与池状态）。

### 3.3 参数展开与 `ParallelState`

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.__init__`（参数快照与 `ParallelState`）

```474:553:python/sglang/srt/managers/scheduler.py
        # Parse args：把 server_args 里的大量配置项展开成 self.* 字段
        self.server_args = server_args
        self.nccl_port = port_args.nccl_port
        self.schedule_policy = get_schedule().schedule_policy
        ...
        self.enable_overlap = (
            not get_schedule().disable_overlap_schedule and not use_mlx()
        )
        self.enable_overlap_mlx = (
            not get_schedule().disable_overlap_schedule and use_mlx()
        )
        ...
        attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size = (
            compute_dp_attention_world_info(
                get_parallel().enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size,
            )
        )
        self.ps = ParallelState(  # tp/pp/dp/attn_tp/attn_cp/attn_dp/moe_ep/moe_dp/gpu_id
            ...
        )
```

- 热路径上直接读 `self.enable_overlap`、`self.page_size` 这类字段，避免每轮都查 runtime context。
- `enable_overlap` 与 `enable_overlap_mlx` 互斥：前者要求 CUDA/HIP/NPU 类后端，后者是 MLX（用 `mx.async_eval` 而非 stream）。
- `enable_pdmux` / `skip_tokenizer_init` / `enable_hierarchical_cache` / `enable_unified_memory` / `max_recv_per_poll` 等都在这里快照，后续 init 只读。
- **`self.ps: ParallelState`** 是后续所有 rank 判断的统一入口（`self.ps.attn_tp_rank`、`self.ps.pp_rank`、`self.ps.gpu_id`…）。DP attention 打开时注意力部分是数据并行，需要单独算一套 `attn_*` world info。

### 3.4 完整 init 顺序表

按 `__init__` 中的实际调用顺序（`scheduler.py:448-763`）：

| # | 调用 | 作用 | 备注 / 依赖 |
| --- | --- | --- | --- |
| 0 | `init_startup_timing_begin` / `is_initializing=True` | 计时起点、构造标记 | 必须最先 |
| 1 | `init_soft_watchdog` | 软看门狗 | 依赖 `forward_ct`、`cur_batch_for_debug` 已置初值 |
| 2 | 参数展开 + `ParallelState` | 配置快照、rank 视图 | 后续所有 init 都读 `self.ps` |
| 3 | `init_model_config` | `ModelConfig.from_server_args` | NPU 下顺带建 `dllm_config` |
| 4 | `init_metrics_collector` | Prometheus 采集器 | 需要 `ps` |
| 5 | `init_ipc_channels` | 与 tokenizer / RPC 的 ZMQ 通道 | 只有 rank0（`pp_rank==0 & attn_tp_rank==0 & attn_cp_rank==0`）建收包端 |
| 6 | `init_idle_sleeper` | 空闲时阻塞在 socket 上省 CPU | 仅 rank0 且 `--sleep-on-idle` |
| 7 | `init_zbal_on_npu` | NPU switch allocator | 必须在任何 torch 分配之前 |
| 8 | `init_pdmux`（条件） | PD 复用上下文 | 仅 `enable_pdmux` |
| 9 | `init_tokenizer` | 加载 tokenizer / processor | 供停符判定、reasoning parser、M-RoPE 回退 |
| 10 | `init_moe_gemm_config` | MoE top-k、GEMM 配置探测 | 兼容多种 HF 命名 |
| 11 | `init_mamba_backend` | mamba selective-scan 后端 | 仅混合 SSM 模型 |
| 12 | `maybe_revert_pr_fix()` | 回滚实验性 patch | **必须在 `init_model_worker` 前**（否则 patch 打在已 revert 的目标上是 no-op） |
| 13 | **`init_model_worker`** | 加载权重、建 KV 池、capture CUDA graph | 最重、最耗显存 |
| 14 | 测试钩子 `SGLANG_TEST_STUCK_SCHEDULER_INIT` | 人为 sleep 复现启动卡死 | 仅测试 |
| 15 | **`kv_cache_builder.build_kv_cache`** | 建 `req_to_token_pool` / allocator / `tree_cache` | 依赖 #13 的显存占用 |
| 16 | `emit_metrics_constants` | 上报常量指标 | 需要 KV 池已建 |
| 17 | `maybe_init_hccl_dp_prewarm` | NPU DeepSeek-V4 HCCL 预热 | 仅特定组合 |
| 18 | canary `attach_radix_cache` | 一致性探针挂 radix cache | 可选 |
| 19 | `init_hisparse_coordinator` | 稀疏 KV 协调器 | 仅 `enable_hisparse` |
| 20 | `DecodeKVCacheOffloadManager`（条件） | decode 侧 KV offload | 仅 disagg decode + offload |
| 21 | `init_running_status` | 队列/批次/计数等运行态 | 连续批处理核心字段 |
| 22 | `init_chunked_prefill` | chunked prefill 配置 | 依赖 `model_config`、`ps.pp_size` |
| 23 | `init_diffusion_llm` | 扩散 LLM（dLLM）状态 | 仅 dLLM |
| 24 | `init_metrics_reporter` | 指标上报器 | 依赖 #4 |
| 25 | `init_schedule_policy` | FCFS/LPM、prefill delayer、new_token_ratio | 依赖 `tree_cache` |
| 26 | `init_watch_dog_memory_saver_input_blocker` | 硬看门狗 + 显存节省 + 输入阻断 + 收包节流 + GC 日志 | 见 4.8 |
| 27 | `init_profiler` | profiler 管理器 | |
| 28 | `maybe_init_rust_server` | 内嵌 Rust 前端（rank0） | **必须在 `init_disaggregation` 与 `init_request_receiver` 之前** |
| 29 | `init_disaggregation` | PD 分离队列 / 传输后端 | |
| 30 | `init_overlap` | `FutureMap`、`forward_stream_ctx`、`copy_stream` | 依赖 KV 池（`req_to_token_pool`） |
| 31 | `maybe_init_ngram_embedding` | ngram embedding 表 | 可选 |
| 32 | `init_deterministic_inference_config` | 确定性推理的截断对齐 | 仅确定性推理 |
| 33 | `init_weight_updater` | 权重热更新管理器 | 依赖 `memory_saver_adapter` |
| 34 | `init_request_dispatcher` | `TypeBasedDispatcher` 路由表 | |
| 35 | `init_lora_drainer` / `init_lora_overlap_loader` | LoRA 相关 | 可选 |
| 36 | `init_grammar_manager` | 约束生成（grammar） | |
| 37 | `maybe_init_scripted_scheduler_hook` | 测试用脚本化钩子 | 仅测试 |
| 38 | `init_request_receiver` | 统一收包器 | 依赖 `recv_skipper` / `input_blocker` / `output_streamer` |
| 39 | `init_dp_attn_adapter` | DP attention 适配器 | |
| 40 | `init_pool_stats_observer` | 池统计观察者 | 依赖 `tree_cache`、allocator |
| 41 | `init_invariant_checker` | 不变量自检器 | |
| 42 | `init_kv_events_publisher` | KV 事件发布 | 可选 |
| 43 | `init_load_publisher` / `init_load_inquirer` | 负载对外发布 / 内部查询 | |
| 44 | `init_output_streamer` | 输出流式回传 | 依赖 `ipc_channels` |
| 45 | `init_batch_result_processor` | 批结果处理器 | 含 beam coordinator |
| 46 | `init_rank_consensus_checker` | rank 一致性检查 | |
| 47 | `is_initializing = False` | 清除构造标记 | 看门狗自此按运行态判死 |
| 48 | `init_startup_timing_summary` | 汇总启动耗时 | 填 `self.startup_time`，供握手回传 |

> 表中「可选/仅 XX」的分支都是靠 `server_args` / runtime context 快照判断的，跑默认配置时大部分会直接 `return`。

---

## 4. 重点环节详解

### 4.1 `init_model_worker`：最重的一步

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.init_model_worker`

```1110:1163:python/sglang/srt/managers/scheduler.py
    def init_model_worker(self):
        # 加载 TP worker 权重 -> 按需建 draft worker -> 分配 KV 内存池 ->
        # 初始化 attention 后端并 capture CUDA graph -> 取回运行参数
        # 1. 构造 TP worker（MlxTpModelWorker / 常规 worker），这一步真正把权重读进显存
        self.init_tp_model_worker()
        if self.server_args.is_startup_weight_load_overlap:
            # 2. --is-startup-weight-load-overlap：权重加载与后续步骤重叠，先 start，后面再 finalize
            self.tp_worker.start_startup_weight_load()
        # 3. 投机解码（EAGLE/NGRAM/DFLASH…）时才建 draft worker；否则 draft_worker = external_corpus_manager = None
        self.maybe_init_draft_worker()

        # 4. init_memory_pools：先 init_target_memory_pool()（runner 上已有 pool 就跳过），
        #    再 kv_cache_builder.resolve_decode_retraction_backup()，
        #    最后把同一套 (pool, allocator) 给 draft worker 复用；计时存进 kv_cache_allocation_time
        tic = time.perf_counter()
        self.init_memory_pools()
        self.kv_cache_allocation_time = time.perf_counter() - tic

        # 5. attention 后端初始化 + CUDA graph capture（draft 同理）；
        #    capture 后的 post_capture_resize_kv_pool() 会把 resize 耗时也计入 kv_cache_allocation_time
        self.init_all_attention_backends()
        self.init_all_cuda_graphs()

        model_runner = self.tp_worker.model_runner
        if model_runner.token_to_kv_pool.post_capture_active:
            model_runner.post_capture_resize_kv_pool()

        if self.server_args.is_startup_weight_load_overlap:
            self.tp_worker.finalize_startup_weight_load()
        ...
        # 6. 无投机解码时就是 tp_worker，否则是 draft_worker——这是「真正干活」的 worker 引用
        self.model_worker = (
            self.tp_worker if self.spec_algorithm.is_none() else self.draft_worker
        )

        # 7. get_worker_info() 12 元组：一次取回 max_total_num_tokens、max_prefill_tokens、max_running_requests、
        #    max_queued_requests、max_req_len、max_req_input_len、random_seed、device、forward_stream 等；
        #    max_total_num_tokens 与 max_req_input_len 是后面握手回传给父进程的两个关键字段；
        #    forward_stream 在此诞生，init_overlap 只是把它包成 stream context；KV cache 构建（下一步）依赖此处算出的 max_total_num_tokens
        (self.max_total_num_tokens, self.max_prefill_tokens, self.max_running_requests,
         self.max_queued_requests, self.max_req_len, self.max_req_input_len,
         self.random_seed, self.device, self.forward_stream, _, _, _) = (
            self.tp_worker.get_worker_info()
        )
        # 8. 派生配置：min_free_slots_delayer（DFlash 自动开启）、pp_max_micro_batch_size 兜底覆盖、set_random_seed；
        #    tp_rank == 0 时打印一行 max_total_num_tokens=... available_gpu_mem=... 的启动日志（排查显存问题时先看这行）
```

#### 4.1.1 实测：各步耗时

打点方式：在 `init_model_worker` 内用 `perf_counter()` 逐步打点，函数末尾汇总一行日志（改动落在 `launch.sh` 实际加载的部署树 `SGLANG_SOURCE` 下，非本仓库）。

环境：单节点 8× NVIDIA H20-3e，`bash launch.sh agg`（TP=8 / EP=8，带 `--skip-server-warmup`）；日志 `bench-results/init-mw-timing-20260928/server.log`。

| 步骤 | 耗时 (s) | 占比 |
| --- | --- | --- |
| `init_tp_model_worker`（权重加载） | 272.09 | 77.4% |
| `init_all_cuda_graphs` | 78.57 | 22.4% |
| `init_memory_pools` | 0.42 | 0.12% |
| `init_all_attention_backends` | 0.21 | 0.06% |
| 其余 15 个打点 | ≈0 | 条件分支未触发 / 微秒级 |
| **sum** | **351.29** | 100% |

原始日志（rank0；8 个 rank 数值基本一致）：

```
init_model_worker per-step timings (s) [tp_rank=0]: init_tp_model_worker=272.083, start_startup_weight_load=0.000, maybe_init_draft_worker=0.000, init_memory_pools=0.423, init_all_attention_backends=0.204, init_all_cuda_graphs=78.571, prepare_hicache_staging=0.000, create_streams_and_prewarm=0.000, post_capture_resize_kv_pool=0.000, finalize_startup_weight_load=0.000, refresh_deep_gemm_layout_memory_budget=0.000, post_capture_elastic_ep_recover=0.000, dispatch_model_worker=0.000, get_worker_info=0.000, derived_scheduler_config=0.000, setup_parallel_groups=0.000, dp_tp_groups=0.000, pad_input_ids_and_seed=0.001, available_gpu_mem_and_log=0.000 | sum=351.286
```

引擎口径对照（同一次启动）：

```
Engine startup timings (s): load_weight=268.43, kv_cache_allocation=0.42, scheduler_e2e=365.27, cuda_graph={prefill=0.00, decode=7.17, target_verify=0.00, draft_prefill=0.00, draft_decode=0.00, draft_extend=0.00}, tokenizer_e2e=386.10
```

读法：

- **≈98% 的时间只花在两块**：权重加载（`init_tp_model_worker`，272 s）+ CUDA graph capture（`init_all_cuda_graphs`，78.6 s）；KV 池分配与其余调度器初始化合计 < 1 s。
- `init_all_cuda_graphs` 的 78.6 s 里真正 capture 只有 `decode=7.17 s`，其余 ≈70 s 是 **FlashInfer autotune**（日志里 `FlashInfer autotune completed`）；prefill graph 在本配置被 `cuda-graph-backend-prefill=disabled` 关掉，所以 `cuda_graph.prefill=0`。
  - **FlashInfer autotune**：启动时按真实 shape 让 AutoTuner 试跑候选 kernel，选最快 tactic 落盘缓存复用；确定性模式跳过。
- 与引擎口径的差值来源：`load_weight`(268.43) 略小于 `init_tp_model_worker`(272.08)，差在读 checkpoint 等前置；`scheduler_e2e`(365.27) 覆盖整个 `Scheduler.__init__`，故大于本函数的 `sum`(351.29)。

### 4.2 KV cache 与内存池

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.__init__`（`build_kv_cache` 调用点）

```608:672:python/sglang/srt/managers/scheduler.py
        result = kv_cache_builder.build_kv_cache(
            server_args=self.server_args,
            model_config=self.model_config,
            tp_worker=self.tp_worker,
            page_size=self.page_size,
            spec_algorithm=self.spec_algorithm,
            attn_tp_cpu_group=..., tp_cpu_group=..., attn_cp_cpu_group=...,
            ps=self.ps,
            tp_group=self.tp_group, pp_group=self.pp_group,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
            hicache_draft_plan=(... if self.draft_worker is not None else None),
        )
        self.is_hybrid_swa = result.is_hybrid_swa
        self.is_hybrid_ssm = result.is_hybrid_ssm
        self.sliding_window_size = result.sliding_window_size
        self.full_tokens_per_layer = result.full_tokens_per_layer
        self.swa_tokens_per_layer = result.swa_tokens_per_layer
        self.req_to_token_pool = result.req_to_token_pool
        self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
        self.disable_radix_cache = result.disable_radix_cache
        self.tree_cache = result.tree_cache
```

- 产物四件套：`req_to_token_pool`（请求 → token 槽位映射）、`token_to_kv_pool_allocator`（KV 张量分配器）、`tree_cache`（radix 前缀树，前缀复用）、以及 `disable_radix_cache` 开关。
- 混合注意力模型多出 `is_hybrid_swa` / `is_hybrid_ssm` / `sliding_window_size` / `full|swa_tokens_per_layer`，这些决定了后面各类 self-check 与 KV 事件的语义。
- 开了 hierarchical cache 时，把 `cache_controller.load_fence_stream` 指向 `forward_stream`（离线加载与在线前向的同步点）。
- 紧随其后的 `emit_metrics_constants()`（上报 KV 显存占用 / graph 显存 / num_pages）、canary 探针挂载、`init_hisparse_coordinator()`、decode offload manager，都**依赖这里的产物**。

### 4.3 运行态与 chunked prefill

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.init_running_status`

```1258:1284:python/sglang/srt/managers/scheduler.py
    def init_running_status(self):
        self.gracefully_exit = False
        self.waiting_queue: List[Req] = []
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        self.cur_batch_for_debug: Optional[ScheduleBatch] = None
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.return_health_check_ipcs: Deque[Optional[str]] = deque()
        self.flush_wrapper = SchedulerFlushWrapper(...)
        self.session_controller = SessionController(self.tree_cache)
        self.forward_sleep_time = None
        self._engine_paused = False
```

| 字段 | 语义 |
| --- | --- |
| `waiting_queue` | 等待被调度（尚未跑前向）的请求 |
| `running_batch` | 当前持续解码的批次（连续批处理主体） |
| `cur_batch_for_debug` | 本轮要跑前向的批次，也是软看门狗 dump 的对象 |
| `last_batch` | 上一轮批次；重叠调度里它的结果延后到本轮处理 |
| `forward_ct` | 累计前向次数：看门狗判活、测试钩子、周期任务都用它 |
| `_engine_paused` | 引擎暂停（如权重热更新期）时只收请求不跑前向 |
| `flush_wrapper` / `session_controller` | `/flush_cache` 与 session（多轮会话）管理 |

`init_chunked_prefill`（`scheduler.py:1286`）要点：`chunked_prefill_size <= 0` 归一为 `None`（关闭）；**多模态 + Transformers 后端会强制关闭 chunked prefill**（避免分块后多模态输入错位）；`is_mixed_chunk` 需要 `--enable-mixed-chunk`；PP 下可选 `enable_dynamic_chunking`，启动期会 profile 一次 prefill 延迟建预测器，失败则降级关闭。

### 4.4 调度策略

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.init_schedule_policy`

```1361:1404:python/sglang/srt/managers/scheduler.py
    def init_schedule_policy(self):
        self.policy = SchedulePolicy(
            self.schedule_policy, self.tree_cache, self.enable_hierarchical_cache,
            self.enable_priority_scheduling, self.schedule_low_priority_values_first,
        )
        self.prefill_delayer: Optional[PrefillDelayer] = None
        self.prefill_bs_tracker = RecentPrefillBatchSizeTracker(...)
        self.max_prefill_bs: int = 0
        if get_schedule().enable_prefill_delayer:
            ...
            self.prefill_delayer = PrefillDelayer(...)
        self.enable_priority_preemption = (
            self.enable_priority_scheduling
            and not get_schedule().disable_priority_preemption
        )
        self.new_token_ratio_tracker = NewTokenRatioTracker.from_config()
```

- `SchedulePolicy` 需要 `tree_cache`（LPM 要查前缀命中），所以**必须在 KV cache 之后**。
- decode 引擎上配 `--enable-prefill-delayer` 会打日志忽略（没有 prefill 调度路径，delayer 是 no-op）。
- `new_token_ratio_tracker` 估计 decode 每步新增 token 比例，用来预估 KV 消耗、决定是否接纳新 prefill。

下面把 `SchedulePolicy` 的候选策略，以及 `PrefillDelayer` 的作用各展开一段。

**`SchedulePolicy` 的策略种类**

`--schedule-policy` 默认 `fcfs`，可选 6 个字符串，分两个谱系（`managers/schedule_policy.py:231` / `:244`）：

| 谱系 | 枚举 | 取值 | 全称 | 排序依据 | 适用部署（PD 分离 / 不分离） |
| --- | --- | --- | --- | --- | --- |
| 感知缓存 `CacheAwarePolicy` | `LPM` | `lpm` | Longest Prefix Match | 最长前缀匹配：命中前缀越长越靠前 | 非分离 ✅；PD-prefill ✅；PD-decode ❌（默认无 radix cache → 降级 FCFS） |
| | `DFS_WEIGHT` | `dfs-weight` | Depth-First Search Weighting | 沿缓存树 DFS + 子树请求数加权，让共享前缀的请求聚集 | 同 `LPM` |
| 不感知缓存 `CacheAgnosticPolicy` | `FCFS` | `fcfs` | First Come First Serve | 先到先服务（按入队时间） | 全场景 ✅（PD-decode 的默认策略） |
| | `LOF` | `lof` | Longest Output First | 最长输出优先（`max_new_tokens` 降序） | 全场景 ✅ |
| | `RANDOM` | `random` | Random | 随机打乱 | 全场景 ✅ |
| | `ROUTING_KEY` | `routing-key` | Routing Key | 路由键在运行批中出现频率高的排前（利于 MoE 专家/缓存局部性） | 全场景 ✅ |

```231:257:python/sglang/srt/managers/schedule_policy.py
class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""
    # ...（此处省略逐项中译注释）...
    LPM = "lpm"  # longest prefix match
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting

class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""
    # ...（此处省略逐项中译注释）...
    FCFS = "fcfs"  # first come first serve
    LOF = "lof"  # longest output first
    RANDOM = "random"
    ROUTING_KEY = "routing-key"  # prioritize by routing key frequency in running batch
```

`SchedulePolicy.calc_priority` 是排序入口，几条实际生效的规则：

- **自动降级**：`tree_cache` 被禁用时 cache-aware 策略退化为 `FCFS`；`LPM` 在等待队列 > 128 时本轮临时退化为 `FCFS`（避免前缀匹配 + 排序开销过大）。
- **优先级叠加**：`--enable-priority-scheduling`（仅 `fcfs` / `lof` 合法）在排序键前再叠一层 `priority`，方向由 `--schedule-low-priority-values-first` 决定；与 cache-aware 策略不兼容。
- **批内前缀缓存**：cache-aware 下用一棵「模拟 RadixCache」在等待队列内部做前缀匹配——对真实缓存命中很短、但队列内共享前缀超过阈值（默认 32）的请求，会被临时降优先级推后，让先行者先把该前缀写进缓存，从而提升整体命中率。
- **是否适合 PD 分离**：判据是「该引擎有没有 radix cache」——cache-aware 策略依赖前缀命中信息。
  - **PD-prefill 引擎**：保留 radix cache，`LPM` / `DFS_WEIGHT` 有效且推荐（按前缀复用排序）。
  - **PD-decode 引擎**：默认强制 `disable_radix_cache=True`，树缓存退化为 `ChunkCache`，`LPM` / `DFS_WEIGHT` 会被**静默降级为 `FCFS`**（除非显式开 `--disaggregation-decode-enable-radix-cache`，实验特性）。
  - **非分离（`disaggregation_mode=null`）**：radix cache 正常，6 种策略都可用；四种 cache-agnostic 策略不依赖缓存，任意部署形态都可用。

降级逻辑与「decode 引擎强制关缓存」两处代码：

```364:385:python/sglang/srt/managers/schedule_policy.py
    def _validate_and_adjust_policy(
        self, policy: str, tree_cache: BasePrefixCache
    ) -> Policy:
        # ...（校验策略名并据缓存设置调整）...
        try:
            policy_enum = CacheAwarePolicy(policy)
            if getattr(tree_cache, "disable", True):
                # If tree_cache is disabled, using CacheAgnosticPolicy policy
                return CacheAgnosticPolicy.FCFS
            return policy_enum
        except ValueError:
            # ...（退回 cache-agnostic）...
```

```106:112:python/sglang/srt/arg_groups/pd_disaggregation_hook.py
        else:
            declare_resolution(
                server_args,
                "handle_pd_disaggregation",
                disable_radix_cache=True,
            )
            logger.warning("KV cache is forced as chunk cache for decode server")
```

**`ROUTING_KEY`：`routing_key` 的作用（简表）**

| 维度 | 说明 |
| --- | --- |
| 是什么 | 前端路由（SMG）经 HTTP header `x-smg-routing-key` 打的**亲和标签**；engine 只读不改，**不是 session 标识**（session 另有字段） |
| 谁指定 | 调用方/路由层；engine 从 header 取出后透传 `GenerateReqInput → TokenizedGenerateReqInput → Req.routing_key` |
| 主要作用 | `ROUTING_KEY` 策略按「运行批中出现频率」把同键请求聚到同一批优先调度 |
| 收益来源 | 同键请求的局部性复用：共享前缀（缓存命中）、MoE 专家、LoRA/租户等 |
| 何时更有用 | 前缀匹配看不见的亲和、或缓存不可用（PD-decode 默认 ChunkCache）、或队列大前缀匹配太贵时；实现是 O(1) 计数、开销低 |
| 其他用途 | 路由键指标统计（`num_unique_running_routing_keys`、`routing_key_*_req_count`、`routing_keys_active`）；会话重建时透传以保持一致 |

**`PrefillDelayer` 的作用**

`PrefillDelayer`（`managers/prefill_delayer.py:107`）用于 **PD 不分离（`disaggregation_mode == "null"`）+ 开启 overlap 调度** 的场景，决定「本轮要不要推迟 prefill」。

```107:116:python/sglang/srt/managers/prefill_delayer.py
class PrefillDelayer:
    """Prefill 延迟器：在 PD 不分离 + overlap 调度下，决定是否推迟本轮 prefill。
    ...
    约束：仅在 disaggregation_mode == "null"（PD 不分离）且启用 overlap 调度时可用。
    """
```

- **动机**：prefill 批次会和正在跑的 decode 合批，若过早插入 prefill，会让 decode 达不到最大 batch size、拉低吞吐；而不同 DP rank 负载不一，需要跨 rank 统一协调。
- **做法**：每轮把各 rank 的 5 个字段（是否可 prefill、是否低水位强制放行、运行批大小、最大 prefill 批大小、等待队列长度）all-gather 汇总，得到全局形态后决策：

| 全局形态 | 含义 | 决策 |
| --- | --- | --- |
| `all` | 所有 rank 可 prefill | 满足 slot 条件（空位 < `max_prefill_bs`）或 queue 条件时延迟，否则放行 |
| `none` | 都不可 prefill | 直接放行 |
| `mixed` | 部分可、部分不可 | 为对齐各 rank，倾向等待慢的 rank 跟上 |

- **兜底（避免无限等待）**：`--prefill-delayer-max-delay-passes`（默认 30）限制最多连续推迟的 forward 轮数，超时 `wait_timeout` 放行；`--prefill-delayer-token-usage-low-watermark` 在任一 rank 的 KV 使用率低于水位线（GPU 没吃饱）时 `token_watermark` 强制放行；队列触发另用 wall-clock（默认 5000ms）封顶最坏 TTFT；首次满足延迟条件时先放行一次（`skip_first_delayer`），避免冷启动空等。
- **落地**：`Scheduler` 侧用 `RecentPrefillBatchSizeTracker` 估计 `max_prefill_bs`，用 `PrefillDelayerSinglePassExecutor` 保证每轮只真正协商一次；决策原因（`no_wait` / `wait_success` / `delay` / `wait_timeout` / `token_watermark`）打点到指标。

### 4.5 运维组件（硬看门狗 / 显存节省 / 输入阻断）

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.init_watch_dog_memory_saver_input_blocker`

```1412:1433:python/sglang/srt/managers/scheduler.py
    def init_watch_dog_memory_saver_input_blocker(self):
        self.watchdog = create_scheduler_watchdog(
            self, watchdog_timeout=get_device().watchdog_timeout
        )
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=get_exec().features.enable_memory_saver
        )
        self.recv_skipper = SchedulerRecvSkipper.maybe_create()
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.ps.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )
        if envs.SGLANG_LOG_GC.get():
            configure_gc_logger()
```

1. **硬看门狗**：`watchdog_timeout`（默认 300s）且**不传 `soft=True`** → 超时发 SIGQUIT 让服务 crash，防止永久 hang。与 3.2 的软看门狗共用同一实现。
2. **显存节省**：`TorchMemorySaverAdapter`，供权重/显存热插拔式释放与恢复（`init_weight_updater` 依赖它）。
3. **收包节流**：`SchedulerRecvSkipper.maybe_create()` 仅在 `scheduler_recv_interval > 1` 时创建——按上一轮 forward mode 加权计数，隔几轮才收一次新请求，省 ZMQ 收包开销（不支持 DP attention）。
4. **输入阻断**：仅 `SGLANG_ENABLE_COLOCATED_BATCH_GEN=1` 时创建；全局屏障期间把新请求暂存（pending），等所有 rank 到达屏障后统一放行，用于权重热更等「全局静止」操作；非 `attn_tp_rank==0` 用 `noop` 只参与屏障协同。
5. **GC 日志**：`SGLANG_LOG_GC=1` 时打到日志，便于排查 GC 抖动。

### 4.6 其它成对/成组的 init

| 方法 | 关键点 |
| --- | --- |
| `init_ipc_channels` | 封装成 `SchedulerIpcChannels`；只有 rank0 拿 `recv_from_tokenizer`，非 rank0 保持 `None`；顺带建 load snapshot writer |
| `init_idle_sleeper` | 仅 rank0 + `--sleep-on-idle`；Rust server 模式换成 `RustServerIdleSleeper` |
| `maybe_init_rust_server` | rank0 且有 `SGLANG_RUST_SERVER` 时 `RustServer.launch`；**必须在 `init_disaggregation`（KV bootstrap registry 同步注册）和 `init_request_receiver`（选 ingress）之前**；未启用也会显式置 `self.rust_server = None` |
| `init_overlap` | `FutureMap` **始终创建**（非重叠路径也靠它中继 decode 的 `input_ids`）；MLX 分支只建 `result_queue`；否则建 `forward_stream_ctx` / `copy_stream`，开了 `enable_overlap` 再建长度为 2 的 `batch_record_buf` 环形缓冲（钉住最近两轮 GPU 张量引用，防跨 stream 提前释放） |
| `init_request_receiver` | 需要 `recv_skipper`、`input_blocker`、`mm_receiver`、`output_streamer.stream_output`，所以这三者必须更早建好 |
| `init_hisparse_coordinator` | 复用 `model_runner` 内已创建的 coordinator，并把 decode producer stream 指到 `forward_stream` |

### 4.7 收尾：`is_initializing = False` 与耗时汇总

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.__init__`（尾部）+ `Scheduler.init_startup_timing_summary`

```760:782:python/sglang/srt/managers/scheduler.py
        self.init_rank_consensus_checker()
        self.is_initializing = False
        self.init_startup_timing_summary()

    def init_startup_timing_summary(self) -> None:
        self.startup_time = build_scheduler_startup_time(
            target_load_weight=self.tp_worker.weight_load_time,
            draft_load_weight=(
                0.0 if self.draft_worker is None else self.draft_worker.weight_load_time
            ),
            kv_cache_allocation=self.kv_cache_allocation_time,
            scheduler_e2e=time.perf_counter() - self.scheduler_startup_begin,
            target_cuda_graph=self.tp_worker.graph_time_usage,
            draft_cuda_graph=(...),
        )
```

- `is_initializing = False` 之后，软看门狗的 `dump_info` 才会真正输出诊断现场；`_check_all_pools` 之类也才有意义。
- `build_scheduler_startup_time` 把 **权重加载 / KV 分配 / CUDA graph / scheduler 端到端** 四类耗时打出来（日志里的 `Engine startup timings (s): ...`）；这个 `startup_time` 随后通过握手回传父进程。

### 4.8 组件类一览（上面的 init_* 各自构造了什么）

上面所有 `init_*` 里出现的组件都是**专用的类**（绝大多数在 `managers/scheduler_components/` 下），`Scheduler` 只持有实例引用：

| 组件 | 类 | 定义位置 | 由谁创建 |
| --- | --- | --- | --- |
| ZMQ 通道集合 | `SchedulerIpcChannels` | `managers/scheduler_components/ipc_channels.py:25` | `init_ipc_channels` |
| 请求分发 | `TypeBasedDispatcher` | `python/sglang/utils.py:611` | `init_request_dispatcher` |
| 收包器 | `SchedulerRequestReceiver` | `managers/scheduler_components/request_receiver.py:63` | `init_request_receiver` |
| 收包节流 | `SchedulerRecvSkipper` | `managers/scheduler_components/recv_skipper.py:13`（旧版 `managers/scheduler_recv_skipper.py:6`） | `init_watch_dog_memory_saver_input_blocker` |
| 输入阻断 | `SchedulerInputBlocker` | `managers/scheduler_input_blocker.py:34` | 同上 |
| 看门狗底层 | `Watchdog`（基类）/ `_WatchdogReal` / `_WatchdogNoop` / `WatchdogRaw` | `srt/utils/watchdog.py:20` / `:77` / `:139` / `:145` | `create_scheduler_watchdog`（软/硬看门狗都走它） |
| 池统计 | `SchedulerPoolStatsObserver` | `managers/scheduler_components/pool_stats_observer.py:177` | `init_pool_stats_observer` |
| 不变量自检 | `SchedulerInvariantChecker` | `managers/scheduler_components/invariant_checker.py:58` | `init_invariant_checker` |
| 批结果处理 | `SchedulerBatchResultProcessor` | `managers/scheduler_components/batch_result_processor.py:88` | `init_batch_result_processor` |
| 输出回传 | `SchedulerOutputStreamer` | `managers/scheduler_components/output_streamer.py:67` | `init_output_streamer` |
| 权重热更新 | `SchedulerWeightUpdaterManager` | `managers/scheduler_components/weight_updater.py:91` | `init_weight_updater` |
| 指标上报 | `SchedulerMetricsReporter` | `managers/scheduler_components/metrics_reporter.py:157` | `init_metrics_reporter` |
| 负载发布 / 查询 | `SchedulerLoadPublisher` / `SchedulerLoadInquirer` | `managers/scheduler_components/load_publisher.py:114` / `load_inquirer.py:43` | `init_load_publisher` / `init_load_inquirer` |
| KV 事件 | `SchedulerKvEventsPublisher` | `managers/scheduler_components/kv_events_publisher.py:59` | `init_kv_events_publisher` |
| DP attention 适配 | `SchedulerDPAttnAdapter` | `managers/scheduler_components/dp_attn.py:545` | `init_dp_attn_adapter` |
| profiler | `SchedulerProfilerManager` | `managers/scheduler_components/profiler_manager.py:63` | `init_profiler` |
| flush 封装 | `SchedulerFlushWrapper` | `managers/scheduler_components/flush_wrapper.py:11` | `init_running_status` |
| 内嵌 Rust 前端 | `RustServer` | `managers/rust_server.py:358` | `maybe_init_rust_server` |

> 注意：`Scheduler` 类本身（`scheduler.py:423`）只负责编排与热路径逻辑，具体职责全部下放到这些组件类；而 `run_scheduler_process` / `configure_scheduler_process` / `dispatch_event_loop` 是 `scheduler.py` 的模块级函数，`_launch_scheduler_processes` / `_wait_for_scheduler_ready` 分属 `Engine` 类方法与 `engine.py` 模块级函数。

---

## 5. 握手：`get_init_info` ↔ `_wait_for_scheduler_ready`

子进程侧（`scheduler.py:1823`）：

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.get_init_info`

```1823:1836:python/sglang/srt/managers/scheduler.py
    def get_init_info(self) -> Dict[str, Any]:
        """Return scheduler initialization info for handshake."""
        result_dict = {
            "status": "ready",
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_req_input_len": self.max_req_input_len,
            "startup_time": self.startup_time,
        }
        return result_dict
```

父进程侧（`entrypoints/engine.py`）：每个 rank 一对 `mp.Pipe(duplex=False)`，`writer` 传进子进程、`reader` 留给父进程：

**代码归属**：`python/sglang/srt/entrypoints/engine.py` · `class Engine(EngineScoreMixin, EngineBase)`（`:227`）→ classmethod `Engine._launch_scheduler_processes`

```886:918:python/sglang/srt/entrypoints/engine.py
            for pp_rank in pp_rank_range:
                for tp_rank in tp_rank_range:
                    reader, writer = mp.Pipe(duplex=False)
                    gpu_id = (
                        server_args.base_gpu_id
                        + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                        + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                    )
                    attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(
                        server_args, tp_rank
                    )
                    proc = mp.Process(
                        target=run_scheduler_process_func,
                        args=(server_args, port_args, gpu_id, tp_rank, attn_cp_rank,
                              moe_dp_rank, moe_ep_rank, pp_rank, None, writer),
                    )
                    proc.start()
```

父进程等到「模型加载完成」的判定是**轮询 + 子进程存活检查**，而不是阻塞 `recv()`：

**代码归属**：`python/sglang/srt/entrypoints/engine.py` · 模块级函数 `_wait_for_scheduler_ready`（无类）

```1903:1932:python/sglang/srt/entrypoints/engine.py
def _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs) -> List[Dict]:
    """Uses poll() with timeout instead of blocking recv(), so that child process
    death (e.g. OOM SIGKILL) is detected promptly instead of hanging forever."""
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        while True:
            if scheduler_pipe_readers[i].poll(timeout=5.0):
                try:
                    data = scheduler_pipe_readers[i].recv()
                except EOFError:
                    raise _scheduler_died_error(i, scheduler_procs[i])
                if data["status"] != "ready":
                    raise RuntimeError("Initialization failed. ...")
                scheduler_infos.append(data)
                break

            # Poll timed out — check all processes for early death
            for j in range(len(scheduler_procs)):
                if not scheduler_procs[j].is_alive():
                    raise _scheduler_died_error(j, scheduler_procs[j])
    return scheduler_infos
```

要点：

- **为什么用 `poll(5s)`**：阻塞 `recv()` 在子进程被 OOM SIGKILL 时会永久挂住父进程；`poll` 超时后立刻检查 `proc.is_alive()`，能给出「Rank N scheduler died during initialization (exit code: -9)」这种可操作错误（`-9` 基本就是 OS OOM Killer，`dmesg -T | grep -i oom` 印证）。
- 回传的 `max_total_num_tokens` / `max_req_input_len` 会被父进程继续下传：`max_req_input_len` 用于 HTTP 侧请求长度校验，`startup_time` 汇总进引擎启动耗时日志。
- `SchedulerInitResult.wait_for_ready()` 是个可调用对象，把「等就绪」延迟到 `_launch_subprocesses` 的某个时点才触发（见原文 3.1 节）。
- 握手完成**之后**才 `run_event_loop()`：这样「回传 ready」与「开始服务」是两件事，父进程不会因为一个已开始跑 loop 但尚未真正可用的子进程而误判。

---

## 6. 进入事件循环：`run_event_loop` 与 `dispatch_event_loop`

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.run_event_loop`

```1849:1891:python/sglang/srt/managers/scheduler.py
    def run_event_loop(self) -> None:
        # Engine init (graph capture, warmups) is done; from here on any
        # Triton kernel device-load is a lazy first-use at serving time.
        triton_load_watch.install()
        triton_load_watch.mark_serving_started()

        if use_mlx():
            dispatch_event_loop(self)
            return

        self.schedule_stream = self.device_module.Stream(priority=0)
        if self.device == "cpu":
            self.schedule_stream.synchronize = lambda: None
        elif is_cuda() or _is_hip:
            # 重抽：避免 schedule_stream 与 forward_stream 别名，否则重叠调度失效
            _redraws = 0
            while (
                self.schedule_stream.cuda_stream == self.forward_stream.cuda_stream
                and _redraws < 64
            ):
                self.schedule_stream = self.device_module.Stream(priority=0)
                _redraws += 1
        self._war_barrier_enabled = is_cuda() or envs.SGLANG_ENABLE_WAR_BARRIER.get()
        with self.device_module.StreamContext(self.schedule_stream):
            dispatch_event_loop(self)
```

1. **`triton_load_watch`**：标记「图捕获/warmup 已结束」，之后任何 Triton kernel 的 device-load 都算**服务期懒加载**，可被监控到（这正是原文预热那节说的「首个真实请求会触发编译」的观测点）。
2. **`schedule_stream`**：所有 CPU 端调度准备产生的 GPU 操作跑在这条 stream 上，与 `forward_stream` 分离——这是重叠调度的基础。CUDA/HIP 的 stream 来自固定轮转池，可能抽到与 `forward_stream` 相同的 handle，所以要**重抽最多 64 次**直到不别名。
3. **WAR（Write-After-Read）屏障**：CUDA 下默认开启；作用是把「本轮调度写入共享缓冲」挡在「上轮前向读取」之后（`_apply_war_barrier` 优先用 `shared_read_done_event`，否则退化为 `wait_stream(forward_stream)`）。
4. **`with StreamContext(schedule_stream)`** 包住 dispatch：保证事件循环里所有默认 stream 操作都落在 schedule stream 上。

之后按「PD 模式 × PP × pdmux/MLX × overlap」组合选循环：

**代码归属**：`python/sglang/srt/managers/scheduler.py` · 模块级函数 `dispatch_event_loop`（无类，不在 `Scheduler` 内）

```5694:5722:python/sglang/srt/managers/scheduler.py
def dispatch_event_loop(scheduler: Scheduler):
    disaggregation_mode = scheduler.disaggregation_mode
    if disaggregation_mode == DisaggregationMode.NULL:
        if scheduler.enable_pdmux:            scheduler.event_loop_pdmux()
        elif pp_size > 1:                     scheduler.event_loop_pp()
        elif scheduler.enable_overlap_mlx:    scheduler.event_loop_overlap_mlx()
        elif scheduler.enable_overlap:        scheduler.event_loop_overlap()
        else:                                 scheduler.event_loop_normal()
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if pp_size > 1:                       scheduler.event_loop_pp_disagg_prefill()
        elif scheduler.enable_overlap:        scheduler.event_loop_overlap_disagg_prefill()
        else:                                 scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if pp_size > 1:                       scheduler.event_loop_pp_disagg_decode()
        elif scheduler.enable_overlap:        scheduler.event_loop_overlap_disagg_decode()
        else:                                 scheduler.event_loop_normal_disagg_decode()
```

两个代表型循环的骨架差异：

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.event_loop_normal`

```1914:1947:python/sglang/srt/managers/scheduler.py
    def event_loop_normal(self):
        # 非重叠：每轮严格串行 收请求 → 选批 → 前向 → 处理结果
        while True:
            if self.gracefully_exit:
                break
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            plan = self.get_next_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            self.cur_batch_for_debug = batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self._sched_idled = True
                self.on_idle()
            self.last_batch = batch
```

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.event_loop_overlap`

```1952:1976:python/sglang/srt/managers/scheduler.py
    def event_loop_overlap(self):
        # 重叠：把"上一批的结果处理"推迟到"本批前向已启动之后"，用 result_queue 中转
        self.result_queue: Deque[Tuple[ScheduleBatch, ...]] = deque()

        def pop_and_process():
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            if self.gracefully_exit:
                break
```

一句话区别：`event_loop_normal` 的 `run_batch` 与 `process_batch_result` 同步串行（好调试、吞吐低）；`event_loop_overlap` 只把 `(batch, result)` 入队，让 CPU 调度与 GPU 前向并行（结果可能还没拷回 CPU，靠 `FutureMap` 中继）。

---

## 7. 启动期常见问题（Scheduler 视角）

| 现象 | 定位 |
| --- | --- |
| 卡在 `wait_for_ready` | 某个 scheduler 初始化失败（权重 OOM、算子编译报错）。父进程 `poll` 超时会检查存活，退出码 `-9` 多为 OS OOM Killer：`dmesg -T \| grep -i oom` |
| 端口已监听但请求 503 | warmup 未完成（`ServerStatus.Starting`），等日志 `The server is fired up and ready to roll!` |
| 想复现「启动卡死」 | `SGLANG_TEST_STUCK_SCHEDULER_INIT=<秒>` 在 `init_model_worker` 后人为 sleep |
| 异常时兄弟 rank 刷屏 | `SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION=1` 让整组进程一起被 SIGKILL |
| 想看卡死现场 | 打开 `--soft-watchdog-timeout`（只 dump 不杀）；现场含 `cur_batch_for_debug` 与各池状态 |
| 显存相关问题 | 先看 tp_rank0 那行 `max_total_num_tokens=..., available_gpu_mem=... GB` 日志 |
| 想省空闲 CPU | `--sleep-on-idle`（rank0 才会建 `idle_sleeper`） |
| 想控制启动日志噪声 | `SGLANG_LOG_GC` 只在需要看 GC 时打开；`faulthandler` 已默认启用 |

---

## 8. 小结

**Scheduler 子进程启动 = 「配置进程 → 编排式构造 → 握手回传 → 进入事件循环」四段。**

- **配置进程**：父死子死、进程名带 rank 前缀、faulthandler、logger 前缀、CPU 亲和性 / NUMA。
- **构造**：`init_soft_watchdog` 抢在最前（线程首个 tick 要读 `forward_ct`）；`init_model_config` → `init_ipc_channels` → **`init_model_worker`（权重 + KV 池 + CUDA graph，最重）** → KV cache 构建 → 运行态 / chunked prefill → 调度策略 → 运维组件 → Rust server / PD / overlap → 收尾清标记并汇总耗时。顺序不是风格问题，而是**依赖问题**（见第 3.4 节的依赖列）。
- **握手**：`get_init_info()` 通过 `mp.Pipe` 回传 `status/max_total_num_tokens/max_req_input_len/startup_time`；父进程用 `poll(5s)` + 存活检查，保证 OOM 时快速失败而不是永久挂住。
- **事件循环**：建 `schedule_stream`（避免与 `forward_stream` 别名 + WAR 屏障）后按 PD/PP/overlap 组合分派；`normal` 串行、`overlap` 用 `result_queue` 让 CPU 调度叠在 GPU 前向上。
