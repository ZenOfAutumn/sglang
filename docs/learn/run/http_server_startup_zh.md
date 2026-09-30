# SGLang HTTP Server 启动全过程：从 launch_server.run_server 到 /model_info 就绪

本文回答一个问题：**执行 `sglang serve`（或 `python -m sglang.launch_server`）后，到 HTTP 接口可用，中间到底发生了什么？**

行文只讲「运行期（run）」的调用顺序与进程编排，不重复解释「为什么要拆三类进程」这类设计动机——那部分见 `docs/learn/arch/why_three_processes_zh.md`。本文凡是与已有文档重叠的地方，一律改为引用，避免同一份内容维护两遍。

## 0. 一张图看全流程

```text
  python -m sglang.launch_server                    sglang serve
            │  (__main__)                                │  cli/serve.py::_run_llm
            │  load_plugins()                            │  prepare_server_args()
            │  prepare_server_args()                     │
            └──────────────►  launch_server.run_server  ◄┘
                                     │
        ┌──────────────┬─────────────┼──────────────────────────────┐
        │ encoder_only │ smg_grpc    │ use_ray                      │ 默认分支（本文主角）
        ▼              ▼             ▼                              ▼
  encoder gRPC/HTTP  grpc_server   ray.http_server   srt.entrypoints.http_server.launch_server
                                                                    │
                        ┌───────────────────────────────────────────┘
                        ▼
             Engine._launch_subprocesses  ← 编排「三类进程」
                        │
   ┌────────────────────┼───────────────────────────┬──────────────────────┐
   ▼                    ▼                           ▼                      ▼
 Scheduler 子进程   Detokenizer 子进程      TokenizerManager(主进程)   SubprocessWatchdog
 (每 TP rank 一个)  (独立进程)             + TemplateManager          (崩溃监控)
   │                    │                           │
   └── pipe 回传 "ready" ┘                           │  (ZMQ PUSH/PULL 互联)
                        │
                        ▼
             _setup_and_run_http_server
                        │  set_global_state / 中间件 / 鉴权 / 日志
                        ▼
             uvicorn(or Granian) 拉起 FastAPI app
                        │
                        ▼
             lifespan 启动钩子：构建 serving handler → 起 warmup 线程 → yield 开始收请求
                        │
                        ▼
             _wait_and_warmup：等权重 → 发真实预热请求 → freeze_gc
                        │
                        ▼
             "The server is fired up and ready to roll!"   ← 此时 HTTP 服务真正可用
```

关键点先说清楚：**`uvicorn` 开始监听端口 ≠ 服务可用**。端口在 `lifespan` 的 `yield` 之前就已绑定，但模型权重加载、KV cache 分配、CUDA graph 捕获都在子进程里进行；真正「可用」以 `_wait_and_warmup` 成功完成一次真实推理为准。

## 1. 入口层：两种启动方式汇聚到 run_server

`python -m sglang.launch_server` 的 `__main__` 与 `sglang serve` 最终都调用同一个分发函数 `run_server`。

```104:127:python/sglang/launch_server.py
if __name__ == "__main__":
    warnings.warn("'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.", ...)
    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
```

`sglang serve` 走 CLI 分支后再汇聚：

```90:99:python/sglang/cli/serve.py
def _run_llm(request: ServeRequest) -> None:
    if any(arg in request.argv for arg in ("-h", "--help")):
        _print_llm_help(request)
        return
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    server_args = prepare_server_args(list(request.argv))
    run_server(server_args)
```

两个入口有几个共同约定：

- `suppress_noisy_warnings()` 在模块导入阶段就屏蔽告警噪音（`launch_server.py` 顶部）。
- `load_plugins()` 允许用户插件在正式启动前打补丁；这一步在 `Engine.__init__`/`_launch_subprocesses` 里还会被**防御性地再调用一次**（插件注册必须幂等）。
- `prepare_server_args()` 返回的是**未经解析（raw）**的 `ServerArgs`（`python/sglang/srt/server_args.py`）。解析（resolution）是独立动作，由 `resolve_once()` 触发，因此游标传入 `run_server` 的参数还是「原始输入」。

```4632:4666:python/sglang/srt/server_args.py
def prepare_server_args(argv: List[str]) -> ServerArgs:
    parser = argparse.ArgumentParser(prog="sglang serve")
    ServerArgs.add_cli_args(parser)
    # ... 合并 --config 配置文件 ...
    raw_args = parser.parse_args(argv)
    logging.basicConfig(...)
    return ServerArgs.from_cli_args(raw_args)
```

> 说明：`ServerArgs.__post_init__` 刻意保持「什么都不做」（见其 docstring），解析逻辑集中在 `resolve_once()` 的 pipeline 里。这样一份 `ServerArgs` 可以被 pickle 传给子进程后再各自解析，不会因为重复解析而出错。

## 2. 分发层：run_server 选哪条路

`run_server` 是一个纯分发器，本身不实现服务逻辑，只决定「当前进程该以哪种形态运行」。

```57:101:python/sglang/launch_server.py
    server_args.resolve_once()
    cfg = resolving_view(server_args)

    if cfg.encoder_only:
        if cfg.smg_grpc_mode or cfg.grpc_mode:
            from sglang.srt.disaggregation.encoder.grpc_server import serve_grpc_encoder
            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encoder.http_server import launch_server
            launch_server(server_args)
    elif cfg.smg_grpc_mode:
        from sglang.srt.entrypoints.grpc_server import serve_grpc
        asyncio.run(serve_grpc(server_args))
    elif cfg.use_ray:
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(...)
        launch_server(server_args)
    else:
        from sglang.srt.entrypoints.http_server import launch_server
        launch_server(server_args)
```

| 分支 | 触发条件 | 目标实现 | 本文覆盖 |
| --- | --- | --- | --- |
| 一 | `--encoder-only` + gRPC | `disaggregation/encoder/grpc_server.py::serve_grpc_encoder` | 否 |
| 一' | `--encoder-only` | `disaggregation/encoder/http_server.py::launch_server` | 否 |
| 二 | `--smg-grpc-mode`（旧版 `--grpc-mode` 会折叠进来） | `entrypoints/grpc_server.py::serve_grpc` | 否 |
| 三 | `--use-ray` | `srt/ray/http_server.py::launch_server`（把 Scheduler 换成 Ray actor） | 否 |
| 四 | 默认 | `srt/entrypoints/http_server.py::launch_server` | **是** |

注意 `cfg.encoder_only` / `cfg.smg_grpc_mode` 这些标志是**解析后**才确定的（例如 `--grpc-mode` 在 `__post_init__` 阶段被折叠进 `smg_grpc_mode`），所以函数开头必须先 `resolve_once()`，再通过 `resolving_view` 读取最终配置。另外，Rust 原生 gRPC（`--grpc-port`）是**与默认 HTTP 服务并行启动**的独立路径，不走上面任何分支（见第 4、7 节）。

## 3. 引擎启动层：http_server.launch_server

默认分支进入 `srt/entrypoints/http_server.py::launch_server`。它的骨架只有两步：**先编排引擎子进程，再装配 HTTP 服务**。

```2949:2987:python/sglang/srt/entrypoints/http_server.py
    # Launch subprocesses
    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
        _weight_cache_daemon_procs,
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager_func,
        run_scheduler_process_func=run_scheduler_process_func,
        run_detokenizer_process_func=run_detokenizer_process_func,
    )

    if envs.SGLANG_RUST_SERVER.get():
        # Rust 服务器同时承担 api-server / tokenizer / detokenizer，
        # 主进程没有 Python HTTP server 可跑：直接预热后阻塞等 scheduler 退出。
        if not get_serving().skip_server_warmup:
            _execute_server_warmup(server_args)
        logger.info("The server is fired up and ready to roll!")
        if launch_callback is not None:
            launch_callback()
        scheduler_init_result.block_until_scheduler_exits()
    else:
        _setup_and_run_http_server(
            server_args, tokenizer_manager, template_manager, port_args,
            scheduler_init_result.scheduler_infos, subprocess_watchdog,
            execute_warmup_func=execute_warmup_func, launch_callback=launch_callback,
        )
```

几个可替换的注入点（`launch_server` 的形参默认值）值得记住，测试与派生实现常靠它们打桩：

- `init_tokenizer_manager_func` → `engine.py::init_tokenizer_manager`
- `run_scheduler_process_func` → `scheduler.py::run_scheduler_process`
- `run_detokenizer_process_func` → `detokenizer_manager.py::run_detokenizer_process`
- `execute_warmup_func` → `_execute_server_warmup`

### 3.1 子进程编排：Engine._launch_subprocesses

这是整个启动流程最核心的一环，负责把「三类进程」按正确顺序拉起，并在它们之间建立握手。返回的是一个 6 元组：`tokenizer_manager, template_manager, port_args, scheduler_init_result, subprocess_watchdog, weight_cache_daemon_procs`。

```1067:1088:python/sglang/srt/entrypoints/engine.py
    @classmethod
    def _launch_subprocesses(
        cls, server_args, init_tokenizer_manager_func, run_scheduler_process_func,
        run_detokenizer_process_func, port_args=None, placement_group=None,
    ) -> Tuple[TokenizerManager, TemplateManager, PortArgs, SchedulerInitResult,
               Optional[SubprocessWatchdog], ...]:
        """Launch the TokenizerManager in the main process, the Scheduler in a subprocess,
        and the DetokenizerManager in another subprocess."""
        startup_tic = time.perf_counter()
```

按执行顺序拆开看：

**① 进程级环境配置（子进程尚未派生）**

```1092:1095:python/sglang/srt/entrypoints/engine.py
        configure_logger(server_args)
        server_args.resolve_once()
        _set_envs_and_config(server_args)
```

`_set_envs_and_config` 做的事（`engine.py:1757`）：

- 设置 `NCCL_*` / `TRTLLM_ENABLE_PDL` / `CUTE_DSL_*` 等环境变量；
- 生成 `SGLANG_RUN_ID`；开启 metrics 时 `set_prometheus_multiproc_dir()`；
- `set_ulimit()`、版本校验（flashinfer / sglang-kernel）；
- **注册 SIGQUIT 处理函数** `launch_phase_sigquit_handler`：子进程出错会向主进程发 SIGQUIT，主进程据此清理整个进程树；
- `mp.set_start_method("spawn", force=True)`：**子进程一律用 spawn 派生**（这也是 `ServerArgs` 要能被 pickle 的原因）；
- 按需设置 GC 阈值、提示旧版 kernel 缓存目录。

**② 参数校验与解析器检测**

```1110:1126:python/sglang/srt/entrypoints/engine.py
        server_args.check_server_args()
        parsers = resolving_view(server_args)
        if parsers.reasoning_parser == "auto" or parsers.tool_call_parser == "auto":
            resolve_auto_parsers(server_args)
        context_before_publish = snapshot_context()
        publish(server_args, role="tokenizer")
```

这里出现的 `publish` / `snapshot_context` / `resolving_view` 属于 SGLang 的**运行时上下文（runtime context）**机制：`ServerArgs` 解析后被「发布」成进程内的只读配置，各组件通过 `get_serving()` / `get_parallel()` / `get_model()` 等读取，而不是到处传 `server_args`。发布一旦完成，`ServerArgs` 就变为只读（对其重新赋值会抛 `AttributeError`，见 `server_args.py:__setattr__`）。

**③ 分配 IPC 端口：PortArgs.init_new**

```1136:1140:python/sglang/srt/entrypoints/engine.py
            if port_args is None:
                port_args = PortArgs.init_new(server_args)
            logger.info(f"server_args={server_args.resolved_dict()}")
```

`PortArgs` 集中保存所有进程间通信端点（`server_args.py:4678`）：

| 字段 | 方向（ZMQ socket） | 用途 |
| --- | --- | --- |
| `tokenizer_ipc_name` | Detokenizer → TokenizerManager（PULL） | 反分词结果回程；`--skip-tokenizer-init` 时由 rank0 Scheduler 直接推给 TokenizerManager |
| `scheduler_input_ipc_name` | Tokenizer 前端 → Scheduler rank0（PULL） | 已分词请求；多 tokenizer/DP 时经 MultiTokenizerRouter 或 DP controller 转发 |
| `detokenizer_ipc_name` | Scheduler → Detokenizer（PULL） | 输出 token id 批次 |
| `rpc_ipc_name` | Engine ↔ Scheduler（DEALER，双向） | **控制面 RPC**：承载 `Engine.collective_rpc()`——把 `flush_cache`、更新权重、`save_model`、`profile`、`freeze_gc`、`get_internal_state` 等管理命令下发到所有 Scheduler 并等结果；与三条 PUSH/PULL 单向数据通道分开，**不承载逐 token 推理数据** |
| `metrics_ipc_name` | Scheduler → 外部订阅方（PUSH） | 推送 `KvMetrics`（KV/负载指标）给 KV 感知路由/监控 |
| `tokenizer_worker_ipc_name` | TokenizerWorker → MultiTokenizerRouter（router 侧 PULL bind，各 worker PUSH connect） | **仅** `tokenizer_worker_num > 1` 时创建（单 worker 为 `None`）：N 个 TokenizerWorker 的请求/worker 注册/pause-continue 广播先汇聚到 router，再由 router 统一转发 Scheduler；**只承载前向，不承载结果回程**（回程靠请求上打标的 `http_worker_ipc` 直接推回各 worker 私有的 `tokenizer_ipc_name`） |
| `load_collector_ipc_name` | Scheduler → LoadCollector（PUSH/PULL） | dp-attention TCP 模式的负载快照采集地址；单机为空，改由 `instance_id` 推导 |
| `decoupled_spec_ipc_config` | verifier Scheduler ↔ drafter Scheduler（bind/connect 端点组） | 分离式投机解码的 IPC 配置：含 `bind_endpoint` / `connect_endpoints` / `rank`；非单个名字而是一组端点 |
| `forward_pass_metrics_ipc_name` | Scheduler → 外部订阅方（PUSH；**`ServerArgs` 字段**） | `--enable-forward-pass-metrics` 时向外暴露逐轮 forward 指标，供外部（如 Dynamo planner）订阅 |
| `nccl_port` | 端口（非 ZMQ） | torch.dist / NCCL 初始化；未指定时自动选空闲端口 |

普通单机场景用 `ipc://<临时文件>`（Unix domain socket）；开启 **DP attention** 时 `init_new` 会改用 **TCP + 端口**（`ZMQ_TCP_PORT_DELTA = 233`，单机从 `--port` 偏移派生）。

> 说明：上表除 `nccl_port`（端口）外均为 IPC 端点，且已收全 `PortArgs` 的全部 7 个 `*_ipc_name`；其中 `decoupled_spec_ipc_config` 属 `PortArgs`（一组 bind/connect 端点，非单名），`forward_pass_metrics_ipc_name` 属 **`ServerArgs`**。

**④ 启动 Scheduler 子进程（每个 TP rank 一个）**

```1182:1187:python/sglang/srt/entrypoints/engine.py
        scheduler_init_result, scheduler_procs = cls._launch_scheduler_processes(
            server_args, port_args, run_scheduler_process_func, **launch_kwargs
        )
        scheduler_init_result.engine_info_bootstrap_server = engine_info_bootstrap_server
```

`_launch_scheduler_processes`（`engine.py:849`）分两种拓扑：

- **无 DP controller**（`dp_size == 1` 且非 elastic-EP `scale`）：双层循环 `for pp_rank / for tp_rank`，为每个 rank 建一个 `mp.Pipe(duplex=False)` 和 `mp.Process(target=run_scheduler_process_func, ...)`。GPU 由 `base_gpu_id + pp*tp + tp_rank*gpu_id_step` 计算，并在此处算出 `attn_cp_rank / moe_dp_rank / moe_ep_rank`。
- **有 DP controller**：只起一个 `run_data_parallel_controller_process`，由它再去拉起各 DP rank 的 scheduler。

每个 scheduler 通过 pipe 把「ready」消息回传父进程，`wait_for_ready()` 就是读这些 pipe（见下）。

**⑤ 启动 Detokenizer 子进程**

```1257:1261:python/sglang/srt/entrypoints/engine.py
        detoken_procs, detoken_names = cls._launch_detokenizer_subprocesses(
            server_args=server_args, port_args=port_args,
            run_detokenizer_process_func=run_detokenizer_process_func,
        )
```

`_launch_detokenizer_subprocesses`（`engine.py:967`）：`detokenizer_worker_num == 1` 时起一个 `DetokenizerManager` 进程；`> 1` 时每个 worker 一个私有 IPC，再加一个 `MultiDetokenizerRouter` 进程持有原 `detokenizer_ipc_name` 做扇出。

**⑥ 在主进程初始化 TokenizerManager 与 TemplateManager**

```1265:1275:python/sglang/srt/entrypoints/engine.py
        # Init tokenizer manager first, as the bootstrap server is initialized here
        if server_args.tokenizer_worker_num == 1:
            tokenizer_manager, template_manager = init_tokenizer_manager_func(
                server_args, port_args
            )
        else:
            tokenizer_manager = MultiTokenizerRouter(server_args, port_args)
            template_manager = None
```

`init_tokenizer_manager`（`engine.py:160`）构造 `TokenizerManager`，再用 `TemplateManager.initialize_templates()` 加载对话模板，最后**回填 auto 解析器**：若 `--reasoning-parser` / `--tool-call-parser` 是 `auto`，就从模板检测结果里取值（检测不到则告警并置空）。

**⑦ 等待模型加载完成（握手）**

```1278:1306:python/sglang/srt/entrypoints/engine.py
        try:
            scheduler_init_result.wait_for_ready()
            cls._set_startup_time(tokenizer_manager, scheduler_init_result, startup_tic)
            tokenizer_manager.max_req_input_len = scheduler_init_result.scheduler_infos[0]["max_req_input_len"]
            processes = list(scheduler_procs or [])
            names = [f"scheduler_{i}" for i in range(len(processes))]
            processes.extend(detoken_procs)
            names.extend(detoken_names)
            subprocess_watchdog = SubprocessWatchdog(processes=processes, process_names=names)
            subprocess_watchdog.start()
            startup_complete = True
        finally:
            if not startup_complete and isinstance(tokenizer_manager, TokenizerManager):
                tokenizer_manager.cuda_vmm_feature_transport.shutdown()
```

`wait_for_ready()` 内部是 `_wait_for_scheduler_ready`（`engine.py:1903`）：对每个 pipe 用 `poll(timeout=5.0)` 轮询，收到 `{"status": "ready", ...}` 才继续；**超时的同时会检查子进程是否已死**（OOM SIGKILL 之类），避免父进程永久卡死。`_set_startup_time` 把各 scheduler 上报的分阶段耗时（load_weight / kv_cache_allocation / cuda_graph / scheduler_e2e）与 tokenizer 端到端耗时汇总打印。

**⑧ 装配 SubprocessWatchdog**

把 scheduler + detokenizer 子进程交给 `SubprocessWatchdog`，运行期检测到崩溃即可通知主进程退出，避免「半死不活」。

> 非 0 rank 节点的特殊处理：`node_rank >= 1` 时不跑 tokenizer 进程，`wait_for_ready()` 后启动一个 dummy health server 并阻塞（`engine.py:1195`）。Rust 服务器模式下则会跳过 Python detokenizer / tokenizer manager（`engine.py:1233`）。

### 3.2 Scheduler 子进程内部启动

Scheduler 是唯一持有并驱动 GPU 的进程，其启动分「进程配置 → 构造 → 回传 → 事件循环」四步。

```5808:5867:python/sglang/srt/managers/scheduler.py
    # 加载插件 -> 配置进程（日志/进程名/CPU 亲和性）-> 可选初始化 tracing ->
    # 创建 Scheduler 实例 -> 通过 pipe 把初始化信息回传父进程 -> 进入事件循环。
    load_plugins()
    publish(server_args, role="scheduler")
    dp_rank = configure_scheduler_process(server_args, gpu_id, tp_rank, attn_cp_rank,
                                          moe_dp_rank, moe_ep_rank, pp_rank, dp_rank, ...)
    # ... tracing 初始化 ...
    scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank, moe_ep_rank,
                          pp_rank, attn_cp_rank, moe_dp_rank, dp_rank)
    pipe_writer.send(scheduler.get_init_info())
    scheduler.run_event_loop()
```

`configure_scheduler_process`（`scheduler.py:5725`）：`kill_itself_when_parent_died()`、`setproctitle("sglang::scheduler_TP0_...")`、`faulthandler.enable()`、按 rank 前缀配置 logger、按需设置 CPU 亲和性 / NUMA 绑定。

`Scheduler.__init__`（`scheduler.py:448`）保持「编排器」风格，把重活拆进一串 `init_*` / `maybe_init_*`。按顺序最关键的几步：

| 阶段 | 方法 | 作用 | 补充说明 |
| --- | --- | --- | --- |
| 软看门狗 | `init_soft_watchdog` | 启动后检测初始化/运行卡死 | 1. **默认关闭**：仅当 `soft_watchdog_timeout` 非 `None` 才创建调用；<br>2. **soft 语义**：超时**只 dump 诊断信息、不杀进程**；<br>3. **检测原理**：独立 daemon 线程轮询 `scheduler.forward_ct`，超时即打印 `cur_batch_for_debug`（batch_size/reqs）与各显存池状态，用于定位死锁/泄漏；<br>4. **位置约束**：必须在 `__init__` 最前面调用（线程首个 tick 就读 `forward_ct`、`cur_batch_for_debug`），初始化阶段 `dump_info` 返回空串 |
| 模型配置 | `init_model_config` | 读 HF config、上下文长度、是否生成/多模态 | 早于模型 worker，决定后续加载哪套实现（如 NPU 下校验 page size） |
| IPC | `init_ipc_channels` | 建与 tokenizer / detokenizer / RPC 的 ZMQ 通道 | 端点按 `pp_rank`/`attn_tp_rank` 等是否为 rank0 决定是否创建 |
| **模型 worker** | `init_model_worker` | **加载权重**（最重、最耗显存的一步） | 流程为 TP worker → 可选 draft worker → 分配 KV 内存池 → 初始化 attention 后端并 capture CUDA graph → 回取 `max_total_num_tokens`/`forward_stream` 等运行参数 |
| **KV cache** | `kv_cache_builder.build_kv_cache` | 建 `req_to_token_pool` / `token_to_kv_pool_allocator` / radix `tree_cache` | 依赖上一步 worker 给出的 `max_total_num_tokens` 等参数 |
| 运行态 | `init_running_status` / `init_chunked_prefill` | waiting_queue、running_batch、chunked prefill | 连续批处理的核心状态；chunked prefill 的 size / interval 取自调度配置 |
| 调度策略 | `init_schedule_policy` | FCFS/LPM、new_token_ratio 估计 | 依赖已建好的 `tree_cache` 与优先级调度开关 |
| 运维 | `init_watch_dog_memory_saver_input_blocker` | 硬看门狗、显存节省、输入阻断 | 一站式的「运维类组件」初始化：<br>1. **硬看门狗**：用 `watchdog_timeout`（默认 300s、**不传 `soft=True`**），超时发 SIGQUIT 让服务 crash 以防永久 hang，与上面的软看门狗相对；<br>2. **显存节省**：按 `enable_memory_saver` 建 `TorchMemorySaverAdapter`，供权重/显存热插拔式释放与恢复；<br>3. **收包节流**：`SchedulerRecvSkipper` 仅在 `scheduler_recv_interval > 1` 时创建，按上一轮 forward mode 加权计数，隔几轮才收一次新请求，省掉每轮 ZMQ 收包开销；<br>4. **输入阻断**：`SchedulerInputBlocker` 仅在 `SGLANG_ENABLE_COLOCATED_BATCH_GEN` 时创建，全局屏障期间先暂存新请求、等所有 rank 到齐屏障再统一放行（非 `attn_tp_rank==0` 用 noop 只参与协同），用于权重热更等全局静止操作；<br>5. **GC 日志**：若 `SGLANG_LOG_GC` 打开则额外配置 GC 日志，便于排查 GC 抖动导致的延迟尖刺 |
| PD/重叠 | `init_disaggregation` / `init_overlap` | 分离部署队列、schedule_stream/forward_stream | `FutureMap` 始终创建——非重叠路径也靠它跨迭代中继 decode 的 input_ids |
| 分发器 | `init_request_dispatcher` | 按消息类型路由 | `TypeBasedDispatcher`，把 `TokenizedGenerateReqInput` 等分派到对应 handler |
| 收尾 | `is_initializing = False` + `init_startup_timing_summary` | 清除初始化标记、汇总耗时 | 清除标记后软看门狗的 `dump_info` 才会真正输出诊断信息 |

`get_init_info()` 回传的正是父进程 `_wait_for_scheduler_ready` 要的握手数据：

```1823:1836:python/sglang/srt/managers/scheduler.py
    def get_init_info(self) -> Dict[str, Any]:
        result_dict = {
            "status": "ready",
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_req_input_len": self.max_req_input_len,
            "startup_time": self.startup_time,
        }
        return result_dict
```

`run_event_loop()`（`scheduler.py:1849`）建立 `schedule_stream`（与 `forward_stream` 分离，是重叠调度的关键），再按 PD/PP/overlap 分派到具体事件循环，阻塞运行直到 `ShutdownReq`。

### 3.3 Detokenizer 子进程内部启动

```646:667:python/sglang/srt/managers/detokenizer_manager.py
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer")
    configure_logger(server_args)
    publish(server_args, role="detokenizer")
    parent_process = psutil.Process().parent()
    manager = detokenizer_manager_class(server_args, port_args)
    if server_args.tokenizer_worker_num == 1:
        manager.event_loop()
    else:
        manager.multi_http_worker_event_loop()
```

`DetokenizerManager.__init__`（`detokenizer_manager.py:135`）：`init_ipc_channels`（PULL 从 scheduler 收、PUSH 回 tokenizer）、`init_tokenizer`（加载反分词 tokenizer）、`init_running_status`（`decode_status` 增量解码进度、软看门狗、CPU 监控）、`init_request_dispatcher`（按类型路由 `BatchEmbeddingOutput` / `BatchTokenIDOutput` / `FreezeGCReq` / `ConfigureLoggingReq`）。单 tokenizer 模式的 `event_loop` 就是「收 → 分发 → 回传」的无限循环。

### 3.4 TokenizerManager 主进程初始化

`TokenizerManager.__init__`（`tokenizer_manager.py:453`）与 Scheduler 同款编排风格，`init_*` 顺序有依赖（`model_config` 必须先于 tokenizer / metrics）：

```619:644:python/sglang/srt/managers/tokenizer_manager.py
    def init_ipc_channels(self, port_args: PortArgs):
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        if get_serving().tokenizer_worker_num == 1:
            self.send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
            )
            self.tokenizer_ipc_name = None
        else:
            # 多 tokenizer 模式改用 tokenizer_worker_ipc_name
            ...
```

注意它用的是 `zmq.asyncio.Context`——TokenizerManager 跑在 asyncio 事件循环里，这就是「主进程同时也承载 HTTP 服务」的原因。

## 4. 进程与 IPC 全景（启动完成后）

> 三类进程的职责分工与「为什么必须拆进程」，见 `docs/learn/arch/why_three_processes_zh.md`；这里只给出启动装配需要记住的连接关系。

| 进程 | 数量 | 承载方 | 启动函数 | 主要 ZMQ 端点 |
| --- | --- | --- | --- | --- |
| 主进程（HTTP + TokenizerManager） | 1 | `launch_server` 所在进程 | `init_tokenizer_manager` | 收 `tokenizer_ipc_name`，发 `scheduler_input_ipc_name` |
| Scheduler | 每 TP rank 一个 | 子进程 | `run_scheduler_process` | 收 `scheduler_input_ipc_name`，发 `detokenizer_ipc_name` |
| Detokenizer | 1 | 子进程 | `run_detokenizer_process` | 收 `detokenizer_ipc_name`，发 `tokenizer_ipc_name` |

主链路：`HTTP → TokenizerManager(tokenize) →[ZMQ]→ Scheduler(forward)→[ZMQ]→ Detokenizer→[ZMQ]→ TokenizerManager → HTTP 响应`。

## 5. HTTP 装配层：_setup_and_run_http_server

子进程编排完成后，回到主进程装配 HTTP 服务。这一步的顺序很重要：**先把全局状态和中间件准备好，再拉起 uvicorn**，否则 `lifespan` 里访问不到 `_global_state`。

```2664:2723:python/sglang/srt/entrypoints/http_server.py
    # 发布进程级全局状态，供各端点与 lifespan 读取
    set_global_state(_GlobalState(
        tokenizer_manager=tokenizer_manager,
        template_manager=template_manager,
        scheduler_info=scheduler_infos[0],
    ))
    if tokenizer_manager is not None:
        tokenizer_manager._subprocess_watchdog = subprocess_watchdog
    if get_observability().enable_metrics:
        add_prometheus_track_response_middleware(app)

    if server_args.tokenizer_worker_num == 1:
        app.is_single_tokenizer_mode = True
        app.server_args = server_args
        app.warmup_thread_kwargs = dict(
            server_args=server_args, launch_callback=launch_callback,
            execute_warmup_func=execute_warmup_func,
        )
        # API key 鉴权（仅单 tokenizer 模式支持）
        if (server_args.api_key or server_args.admin_api_key
                or app_has_admin_force_endpoints(app)):
            add_api_key_middleware(app, api_key=..., admin_api_key=...)
    else:
        app.is_single_tokenizer_mode = False
        multi_tokenizer_args_shm = write_data_for_multi_tokenizer(
            port_args, server_args, {**scheduler_infos[0], "startup_time": tokenizer_manager.startup_time},
        )
```

单/多 tokenizer 模式的分野就发生在这里：

- **单 tokenizer**（默认）：通过 **app 属性**把 `server_args` 等传给 `lifespan`，HTTP 服务与 TokenizerManager 同进程；
- **多 tokenizer**：把参数写进**共享内存**，uvicorn 以多 worker 模式加载 `app`，每个 worker 在 `lifespan` 里从共享内存恢复自己的 TokenizerManager（且不支持 API key）。

随后根据开关选择真正的监听服务器（`http_server.py:2724`）：

| 条件 | 启动方式 | 备注 |
| --- | --- | --- |
| `--enable-http2` | `_run_granian_server` 嵌入式 Granian | HTTP/2；`tokenizer_worker_num>1` 时退回 Granian 多进程 |
| `--enable-ssl-refresh` | `uvicorn.Config` + `uvicorn.Server` + `SSLCertRefresher` | 证书热刷新，退出时停刷新线程 |
| 单 worker 默认 | `uvicorn.run(app, loop="uvloop", ...)` | 直接传 app 对象 |
| 多 worker 默认 | `uvicorn.run("sglang.srt.entrypoints.http_server:app", workers=N, ...)` | 传字符串让各 worker 各自 import |

`set_uvicorn_logging_configs(server_args)` 会先根据 `--log-level-http` 调整 uvicorn 日志，再启动。`finally` 块负责多 tokenizer 模式下的共享内存卸载与 socket 清理。

## 6. FastAPI app 与路由注册（导入期完成）

`app` 是**模块级全局对象**，在 `import sglang.srt.entrypoints.http_server` 时就被构建出来——这解释了为什么多 worker 模式可以用 `"module:app"` 字符串让 uvicorn 自行导入。

> **uvicorn 与 FastAPI 的分工**：`uvicorn` 是 ASGI 服务器，负责绑定端口、收发 HTTP/WS 连接、驱动事件循环与 `lifespan`；`FastAPI` 是 Web 框架，负责定义路由、校验/序列化请求、挂中间件。二者都不含推理逻辑，业务一律转交 TokenizerManager/Scheduler。

```494:536:python/sglang/srt/entrypoints/http_server.py
app = FastAPI(
    lifespan=lifespan,
    openapi_url=None if get_bool_env_var("DISABLE_OPENAPI_DOC") else "/openapi.json",
)
app.router.route_class = ORJSONRoute
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

if envs.SGLANG_ENABLE_REQUEST_DECOMPRESSION.get():
    app.add_middleware(RequestDecompressionMiddleware)

# Include routers
from sglang.srt.entrypoints.v1_loads import router as v1_loads_router
v1_loads_router.route_class = ORJSONRoute
app.include_router(v1_loads_router)

from sglang.srt.entrypoints.elastic_ep import router as elastic_ep_router
elastic_ep_router.route_class = ORJSONRoute
app.include_router(elastic_ep_router)
```

装配期值得注意的几点：

- **`ORJSONRoute` / `ORJSONRequest`**：把请求体解析从标准库 `json` 换成 `orjson`，面向几十 MB 的多模态请求；同时更严格（裸 `NaN`/`Infinity`、超 64 位整数直接 400）。`route_class` 是**每个 router 各自维护**的，所以 `include_router` 的子路由也要单独设置。
- **中间件顺序**：CORS、可选解压、Prometheus（在 `_setup_and_run_http_server`/`lifespan` 里追加）、API key 鉴权（在 `_setup_and_run_http_server` 里追加）。
- **路由分组**（`@app.get/@app.post/@app.websocket`）：
  - 原生接口：`/health`、`/health_generate`、`/model_info`、`/server_info`、`/get_load`、`/generate`、`/encode`、`/classify`、`/flush_cache`、权重更新系列、`/pause_generation`、`/abort_request` 等；
  - OpenAI 兼容：`/v1/completions`、`/v1/chat/completions`、`/v1/embeddings`、`/v1/models`、`/v1/responses`、`/v1/audio/transcriptions`、`/v1/realtime`(WS) 等；
  - Ollama 兼容：`/api/chat`、`/api/generate`、`/api/tags`；
  - Anthropic 兼容：`/v1/messages` 系列。
- **端点实现依赖 lifespan 里挂到 `app.state` 的 serving handler**：路由是在导入期注册的，但 handler 实例在 lifespan 启动阶段才创建，因此「路由可见」早于「handler 可用」。

`/server_info` 返回的是「解析后」的配置，这与本文第 1 节提到的 `resolved_dict()` 相呼应；字段含义见 `docs/metrics/server_info_fields_zh.md`。

## 7. lifespan：uvicorn 接管后的启动钩子

`uvicorn.run(...)` 起来后，第一件事就是执行 FastAPI 的 `lifespan` 启动段。这里是「HTTP 服务从内存到可用」的分水岭。

`lifespan` 的作用可以用三句话概括：

- **启动段（`yield` 之前）**：在真正对外服务前把依赖备齐——多 tokenizer 下先 `init_multi_tokenizer()` 恢复参数，再按需开 metrics / tracing，构造各类 serving handler（OpenAI / Ollama / Anthropic / `/v1/responses`）与 tool server，执行自定义 warmup，启动原生 gRPC + sidecar + 预热线程；
- **`yield`**：分界点，把控制权交回 uvicorn，HTTP 开始接收请求；
- **退出段（`finally`）**：资源回收的唯一入口——停 sidecar → 关 gRPC → `tool_server.aclose()` → `warmup_thread.join()`；因为都在 `try` 内，中途失败也能走到清理。

```302:328:python/sglang/srt/entrypoints/http_server.py
    if getattr(fast_api_app, "is_single_tokenizer_mode", False):
        server_args = fast_api_app.server_args
        warmup_thread_kwargs = fast_api_app.warmup_thread_kwargs
        thread_label = "Tokenizer"
    else:
        # 多 tokenizer worker：从共享内存恢复参数并建立本 worker 的 TokenizerManager
        server_args = await init_multi_tokenizer()
        warmup_thread_kwargs = dict(server_args=server_args)
        thread_label = f"MultiTokenizer-{_global_state.tokenizer_manager.worker_id}"

    if get_observability().enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    if server_args.enable_trace:
        process_tracing_init(...)
        trace_set_thread_info(thread_label)
```

接着按顺序做这些事（`http_server.py:330-458`）：

1. **构建 OpenAI serving handler**：completion / chat / embedding / classify / score / rerank / tokenize / detokenize / transcription，全部挂到 `fast_api_app.state.*`；
2. **构建 Ollama 与 Anthropic serving**（Anthropic 复用 chat handler）；
3. **工具服务器**：`--tool-server demo` → `DemoToolServer`；指定地址 → `MCPToolServer`；`EXA_API_KEY` → `NativeToolServer`；
4. **`/v1/responses`**：构造 `OpenAIServingResponses`。**这一步失败只告警不致命**（例如 gpt-oss harmony vocab 下载失败），端点会被禁用但不影响整体启动；
5. **自定义 warmup**：`--warmups` 指定的预置请求；
6. **原生 gRPC + sidecar**：单 tokenizer 且 `--grpc-port` 且非 SMG/legacy gRPC 时，`_start_native_grpc_server_for_runtime` 启动 Rust 原生 gRPC；配了 `--sidecar` 再启 sidecar；
7. **启动通用预热线程** `warmup_thread = Thread(target=_wait_and_warmup, ...)`；
8. **`yield`**：把控制权交回 uvicorn，开始接收 HTTP 请求。

退出阶段（`yield` 之后的 `finally`）：停 sidecar → 关原生 gRPC → `await tool_server.aclose()` → `warmup_thread.join()`。

注意顺序：**warmup 线程是在 `yield` 前启动的**（`http_server.py:440`），也就是说端口已监听、预热在后台跑。这就是第 0 节那句「端口绑定 ≠ 可用」的来源——期间 `/health` 会返回 503。

## 8. 就绪判定：从 /health 到一次真实推理

`lifespan` 里启动的预热线程执行 `_wait_and_warmup`：

```2501:2543:python/sglang/srt/entrypoints/http_server.py
def _wait_and_warmup(server_args, launch_callback=None, execute_warmup_func=_execute_server_warmup):
    if server_args.checkpoint_engine_wait_weights_before_ready:
        _wait_weights_ready()
    # elastic joiner 由主节点代管，跳过预热
    skip_elastic_joiner_warmup = server_args.is_ep_scale_joiner
    if not get_serving().skip_server_warmup and not skip_elastic_joiner_warmup:
        if not execute_warmup_func(server_args):
            return
    else:
        _global_state.tokenizer_manager.server_status = ServerStatus.Up
    _freeze_gc_after_server_warmup(server_args)
    logger.info("The server is fired up and ready to roll!")
    if server_args.delete_ckpt_after_loading:
        delete_directory(get_model().model_path)
    if launch_callback is not None:
        launch_callback()
```

`_execute_server_warmup`（`http_server.py:2299`）做的是**真正的端到端预热**：

1. 最多轮询 120 次（每次 sleep 1s）请求 `/model_info`，直到返回 200，确认 HTTP 层已起；
2. 根据 `/model_info` 判断是生成模型还是 embedding，多模态是否走 `/v1/chat/completions`；
3. 发一次真实推理请求（生成模型 `max_new_tokens=8`，embedding `1`）——把权重加载、CUDA graph / kernel 编译的一次性开销提前到启动阶段；
4. PD/EP 分离部署改为按 DP rank 并发发送带 bootstrap 信息的请求；
5. 成功后把 `tokenizer_manager.server_status` 置为 `ServerStatus.Up`；异常则 `kill_process_tree` 并返回 False。

预热到底「热」了哪些东西、以及收益从大到小（收益 ≈ 能提前消掉的**首请求延迟尖刺**）：

| 排序 | 预热对象 | 触发方式 | 量级 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | Triton kernel JIT 编译 / autotune（fused_moe、attention extend 等） | `--warmups voice_chat` / `prefill_shapes`，以及首个真实请求 | 秒级 ~ 十几秒 | 收益最大；`warmup.py` 注释明确写了不预热会破坏语音实时推理 |
| 2 | 结构化输出 grammar FSM 编译（xgrammar） | `--warmups whisper_autodetect` | ~15–20s（一次性） | 绝对值很大，但只影响用结构化输出的场景 |
| 3 | CUDA graph 未覆盖 shape 的 re-capture / torch.compile | graph 在 `init_model_worker` capture，预热负责走通 | 百 ms ~ 秒级 | 覆盖面广，MoE / 长 prefill 更显著 |
| 4 | 权重加载完成、KV/显存池与 radix cache 首次分配 | `_wait_weights_ready()` + 首个 forward | 百 ms 级 | 只影响首个请求，但常体现在冷启动 P99 |
| 5 | VLM 视觉编码器编译（MoonViT position interpolation） | 预热用 512×512 图走 `/v1/chat/completions`（Kimi VLM） | 百 ms ~ 秒级 | 仅多模态模型 |
| 6 | Python 侧路径缓存（tokenizer / chat template / 正则、detokenizer 增量解码、ZMQ 通道） | 首个真实请求 | ms ~ 几十 ms | 最小，但影响每条冷路径的首次命中 |

> `--warmups` 的内置项定义在 `entrypoints/warmup.py` 的 `_warmup_registry`：`voice_chat`（size 4→2044 的 fused_moe 内核）、`prefill_shapes`（64→32K 的 prefill 长度，覆盖 fused_moe / attention extend）、`whisper_autodetect`（预编译两个 Whisper 自动检测正则的 FSM）。它们是**按需选择的补充预热**，而上面第 1–5 步那次真实请求是默认必做的。

预热完成后通过 `POST /freeze_gc` 冻结 scheduler / detokenizer 的 GC（长跑时静态对象不再参与 gen2 回收）。

健康检查端点与就绪状态联动（`http_server.py:698`）：

- `/health`：默认只在 `server_status != Starting` 时返回 200；`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=1` 时才真正走一次「生成 1 个 token」的链路探测；
- `/health_generate`：真的发一个 `max_new_tokens=1` 的请求，只要在 `HEALTH_CHECK_TIMEOUT`（默认 20s）内收到任何响应即算健康，否则 503 并标记 `UnHealthy`；
- 优雅退出中（`gracefully_exit`）一律返回 503。

启动期间客户端会先看到 `/health` 503（`server_status == Starting`），直到 warmup 把状态推到 `Up`。

## 9. 启动时序总表

| 阶段 | 位置 | 产物 / 标志 |
| --- | --- | --- |
| 参数解析 | `prepare_server_args` | raw `ServerArgs` |
| 解析生效 | `run_server.resolve_once` | `resolving_view` 可用 |
| 选分支 | `run_server` | 进入 `http_server.launch_server` |
| 进程环境 | `_set_envs_and_config` | NCCL env、SIGQUIT、`mp spawn` |
| 分配端口 | `PortArgs.init_new` | 一组 IPC/TCP 端点 |
| 起 Scheduler | `_launch_scheduler_processes` | 每 TP rank 一进程 |
| 起 Detokenizer | `_launch_detokenizer_subprocesses` | 1 进程（或 router + N） |
| 建 Tokenizer | `init_tokenizer_manager` | 主进程 `TokenizerManager` |
| 等模型加载 | `wait_for_ready` | 收到 `status="ready"` |
| 起看门狗 | `SubprocessWatchdog.start` | 崩溃监控 |
| 装配 HTTP | `_setup_and_run_http_server` | 全局状态、中间件、鉴权 |
| 监听端口 | `uvicorn.run` / Granian | 端口 bind（此时 `/health` 仍 503） |
| 启动钩子 | `lifespan`（yield 前） | serving handler、gRPC、warmup 线程 |
| 热身 | `_wait_and_warmup` | 真实推理 + `freeze_gc` |
| 就绪 | 日志 | `The server is fired up and ready to roll!` |

一个典型成功启动的日志锚点：

```text
server_args={...}
INFO: Started server process [pid]
INFO: Waiting for application startup.
... Engine startup timings (s): load_weight=.., kv_cache_allocation=.., scheduler_e2e=.. ...
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
The server is fired up and ready to roll!
```

## 10. 与已有文档的关系（去重说明）

本文只讲「启动期的调用顺序」，以下内容刻意**不重复**，需要时请跳转：

| 已有文档 | 覆盖内容 | 与本文的关系 |
| --- | --- | --- |
| `docs/learn/arch/why_three_processes_zh.md` | 为什么拆三类进程、GIL、ZMQ 协议、延迟/吞吐收益 | 本文第 4 节只保留启动所需的连接关系，动机部分引用该文 |
| `docs/metrics/endpoints_metrics_and_loads_zh.md` | `/metrics` 与 `/v1/loads` 的请求/返回值、中间件挂载机制 | 本文只在第 6 节指出中间件装配时机 |
| `docs/metrics/server_info_fields_zh.md` | `/server_info` 字段逐项对照 | 本文第 6 节引用其作为字段字典 |
| `docs/metrics/v1_loads_zh.md` | `/v1/loads` 细节 | 同上，负载接口细节不在本文展开 |
| `docs/metrics/kv_cache_scheduling_and_pd_disagg_zh.md` | KV cache 调度与 PD 分离 | 本文只提启动期队列占位，机制见该文 |
| `docs/advanced_features/server_arguments_zh.md` | CLI 参数全表 | 本文不重复列举参数，只说明其何时被解析 |

## 11. 排查启动问题的几个常见坑

- **卡在 `wait_for_ready`**：通常是某个 scheduler 进程初始化失败（权重加载 OOM、算子编译报错）。`_wait_for_scheduler_ready` 会在 poll 超时时检查子进程存活，退出码 `-9` 基本是 OS OOM Killer，可用 `dmesg -T | grep -i oom` 确认。
- **端口已监听但请求 503**：处于 `ServerStatus.Starting`，warmup 还没完成。看日志里有没有 `The server is fired up and ready to roll!`。
- **`/v1/responses` 404**：`OpenAIServingResponses` 初始化失败（例如 harmony vocab 下载失败）只告警禁用，检查启动日志中的 `OpenAI Responses API (/v1/responses) disabled`。
- **多 worker 模式没有 API key 鉴权**：`--api-key` 只在单 tokenizer 模式下生效；多 tokenizer 模式会在 `lifespan` 里断言失败或忽略。
- **`server_args` 被改写报错**：解析完成后 `ServerArgs` 只读，运行期改配置要走 runtime context 的 `override`，不是给字段赋值。

## 12. 一句话总结

**`sglang serve` 的启动 = 一次「先编排进程、再装配 HTTP、最后预热」的三段式：`run_server` 选型 → `Engine._launch_subprocesses` 把 Scheduler / Detokenizer / TokenizerManager 按 ZMQ 拓扑拉起并握手 → `_setup_and_run_http_server` + `lifespan` 装配 FastAPI/uvicorn → `_wait_and_warmup` 用一次真实推理确认端到端可用。端口先绑定，可用性由 warmup 决定。**
