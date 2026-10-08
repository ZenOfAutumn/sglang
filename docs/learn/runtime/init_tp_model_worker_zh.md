# `init_tp_model_worker` 详解

> 本文是 `docs/learn/runtime/scheduler_subprocess_startup_zh.md` 第 4.1 节 `init_model_worker` 第一小步 `init_tp_model_worker` 的**展开版**：原文把它概括成「加载权重」，这里把「这一行里到底做了什么」展开到代码级。
>
> 不重复的内容仍在原文：`Scheduler.__init__` 的整体编排顺序、KV cache 构建（`kv_cache_builder.build_kv_cache`）、attention 后端初始化、CUDA graph capture、握手回传等。它们都在 `init_tp_model_worker` **之后**发生。
>
> 代码块路径均相对仓库根目录（`sglang/`），行号对应当前版本。
>
> 每个代码块前的「**代码归属**」统一标注 `源文件 · 类 → 方法`；模块级函数会显式注明「无类」。

---

## 0. 一页速览

`init_tp_model_worker` 是 `Scheduler.init_model_worker` 的**第一步**，也是整个启动过程中**唯一真正把权重读进显存**的地方。它的全部产出就是给调度器挂上一个 `self.tp_worker`：

```
Scheduler.__init__
        │
        ▼
Scheduler.init_model_worker()                     ← scheduler.py:1112
        │
        ├─① init_tp_model_worker()                ← 本文主题，272 s（权重加载）
        │        │
        │        ▼
        │   TpModelWorker.__init__                ← tp_worker.py:361
        │        ├─ _init_model_config()          # ModelConfig.from_server_args
        │        ├─ _init_model_runner()          # ModelRunner.__init__ ─┐
        │        │                                                        │
        │        │   ModelRunner.__init__          ← model_runner.py:336   │
        │        │        ├─ ...（解析参数 / 建流 / 初始化分布式）         │
        │        │        ├─ initialize()          ← model_runner.py:828  │
        │        │        │        ├─ load_model() ← 真正读 checkpoint    │
        │        │        │        │       └─ load_model_with_memory_saver()
        │        │        │        └─ ...（EPLB / LoRA / kv_cache_dtype）  │
        │        │        └─ init_weight_updater()/init_weight_exporter() │
        │        ├─ _init_multi_layer_eagle_model_runners()  # 仅多层 EAGLE
        │        ├─ _init_dllm_algorithm()                   # 仅 dLLM
        │        ├─ tokenizer / processor                    # 非 draft 且未 skip
        │        ├─ pp_group / world_group                   # NCCL 通信组
        │        └─ 跨 rank 广播随机种子
        │
        └─② start_startup_weight_load()           # 仅 --is-startup-weight-load-overlap
        └─③ maybe_init_draft_worker()             # 投机解码才建
        └─④ init_memory_pools() / attention / cuda graph / get_worker_info()
```

关键结论（后文逐条解释）：

| 结论 | 说明 |
| --- | --- |
| 它**只**建 `TpModelWorker`，不建 KV 池 | KV 池在后面的 `init_memory_pools()` 里按「权重占完之后的剩余显存」来定尺寸 |
| 它**只**建 `ModelRunner`，不初始化 attention 后端 | `init_all_attention_backends()` / `init_all_cuda_graphs()` 在 `init_model_worker` 的后半段 |
| 它的耗时基本等于「读 checkpoint」 | 实测 272 s / 351 s ≈ **77%**，见第 7 节 |
| 必须排在 `maybe_revert_pr_fix()` 之后 | 回滚补丁的目标（如 `_init_pools`）就在这条链路里，晚了就是 no-op |
| target worker 与 draft worker 走**同一个类** | 区别只在 `is_draft_worker=True`、外部传入共享内存池、以及 `draft_load_format` |

---

## 1. 入口与上下文

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.__init__`（调用点）

```591:598:python/sglang/srt/managers/scheduler.py
        # Must precede init_model_worker: revert targets like _init_pools run during it,
        # so patching them afterwards is a no-op.
        maybe_revert_pr_fix()

        # Launch a model worker and draft model worker if using speculative decoding
        # 中译：启动模型 worker（加载权重）；若启用投机解码还会额外启动 draft（草稿）模型 worker。
        #       这是初始化中最重、最耗显存的一步。
        self.init_model_worker()
```

`init_tp_model_worker` 在 `Scheduler.__init__` 里排在第 13 位（原文 3.4 节顺序表），前面必须已经完成：

| 前置步骤 | 为什么必须先做 |
| --- | --- |
| 参数展开 + `ParallelState`（`self.ps`） | `init_tp_model_worker` 直接读 `self.ps.gpu_id` |
| `self.nccl_port = port_args.nccl_port` | 传给 `ModelRunner` 做 NCCL 建组端口 |
| `init_model_config()` | 调度器侧的 `ModelConfig`（worker 侧会用自己的那份） |
| `init_tokenizer()` | 供停符判定 / reasoning parser；worker 内部的 tokenizer 是另一份（三份对照见 §3.3.1） |
| `init_moe_gemm_config()` | FP8/FP4/BF16 GEMM 配置、MoE top-k 配置 |
| `init_mamba_backend()` | 混合 SSM 模型的选择性扫描后端 |
| `init_zbal_on_npu()` | NPU switch allocator，必须在任何 torch 分配之前 |
| `maybe_revert_pr_fix()` | 回滚实验性 patch，**必须在权重加载之前** |

`Scheduler.init_model_worker` 的整体骨架（本文只覆盖第 1 行，其余在原文 4.1 节展开）：

```1112:1128:python/sglang/srt/managers/scheduler.py
    def init_model_worker(self):
        # 中译：初始化模型 worker 的完整流程：加载 TP worker 权重 -> 按需建 draft worker ->
        #       分配 KV 内存池 -> 初始化 attention 后端并 capture CUDA graph ->
        #       从 worker 取回 max_total_num_tokens、device、forward_stream 等关键运行参数。
        # Load model weights.
        self.init_tp_model_worker()
        if self.server_args.is_startup_weight_load_overlap:
            self.tp_worker.start_startup_weight_load()
        self.maybe_init_draft_worker()

        # Prepare KV cache pools for all workers
        tic = time.perf_counter()
        self.init_memory_pools()
        self.kv_cache_allocation_time = time.perf_counter() - tic

        self.init_all_attention_backends()
        self.init_all_cuda_graphs()
```

---

## 2. `Scheduler.init_tp_model_worker`：只有 4 个入参

**代码归属**：`python/sglang/srt/managers/scheduler.py` · `class Scheduler` → `Scheduler.init_tp_model_worker`

```1020:1036:python/sglang/srt/managers/scheduler.py
    def init_tp_model_worker(self):
        worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.ps.gpu_id,
            ps=self.ps,
            nccl_port=self.nccl_port,
        )

        # FIXME: move tp worker's init logic outside of the scheduler.
        # ...（此处省略其他后端分支）...
        from sglang.srt.managers.tp_worker import TpModelWorker

        self.tp_worker = TpModelWorker(**worker_kwargs)
```

要点：

1. **只传 4 个参数**。注意 `TpModelWorker.__init__` 还有 `is_draft_worker` / `req_to_token_pool` / `token_to_kv_pool_allocator` / `memory_pool_config` / `context_length` / `draft_attention_backend` / `random_seed` 等形参，这里**全部走默认值**——也就是说 target worker 此时：
   - `is_draft_worker=False`；
   - 两级内存池为 `None`（稍后 `init_target_memory_pool()` 才分配）；
   - `memory_pool_config=None`（稍后由 `alloc_memory_pool()` 解析）；
   - `random_seed=None`（走广播路径）。
2. **只建标准 `TpModelWorker`**。它是 `BaseTpWorker` 的具体实现，调度器只依赖基类接口。
3. **`FIXME` 注释**说明作者也知道「worker 的初始化逻辑不该放在 scheduler 里」，但目前仍保留这个入口。
4. 方法**没有返回值**，副作用是挂上 `self.tp_worker`。

---

## 3. `TpModelWorker.__init__`：8 件事，按顺序做完

**代码归属**：`python/sglang/srt/managers/tp_worker.py` · `class TpModelWorker`（`:354`）→ `TpModelWorker.__init__`（`:361`）

方法体可以切成 8 段，顺序有依赖：

### 3.1 解析参数 + 建主 `ModelRunner`

```376:413:python/sglang/srt/managers/tp_worker.py
        # Parse args
        # 中译：解析并暂存各类并行维度的 size 与本 worker 所处的 rank。
        #       tp=张量并行、ep=专家并行(MoE)、pp=流水并行、dp=数据并行、cp=上下文并行。
        self.server_args = server_args
        self.ps = ps
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        # 中译：is_draft_worker 标识本 worker 是否为投机解码（speculative）中的 draft 模型 worker。
        self.is_draft_worker = is_draft_worker
        # 中译：is_multi_layer_eagle 表示多层 EAGLE 投机解码（每个 draft 步用一个独立 ModelRunner）。
        self.is_multi_layer_eagle = is_multi_layer_eagle
        # 中译：两级内存池可由外部传入（如 draft worker 复用 target worker 的池），也可后续自行分配。
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        # Draft worker: target's resolved MemoryPoolConfig (forwarded to ModelRunner).
        # 中译：draft worker 复用 target（主模型）已确定的内存池配置，直接转发给 ModelRunner。
        self.memory_pool_config = memory_pool_config
        # Draft worker: target's effective context length; the draft runs at
        # absolute target positions. None keeps server_args.context_length.
        self.context_length = context_length
        # Draft worker: the attention backend the algorithm resolved for it.
        self.draft_attention_backend = draft_attention_backend

        # MTP model runners
        # 中译：MTP（multi-token prediction，如多层 EAGLE）会用到多个 ModelRunner，存于此列表；
        #       常规情况下该列表为空，只用单个 self._model_runner。
        self.model_runner_list: List[ModelRunner] = []

        # 中译：依次构建模型配置与 ModelRunner（真正加载权重、占显存的重操作在此发生）。
        self._init_model_config()
        self._init_model_runner()

        if is_multi_layer_eagle:
            # 中译：多层 EAGLE 时为每个投机步额外创建一个 ModelRunner。
            self._init_multi_layer_eagle_model_runners()

        # 中译：初始化扩散式 LLM（dLLM）算法（若未启用则为 None）。
        self._init_dllm_algorithm()
```

- `self.ps`（`ParallelState`）是本 worker 的 rank 视图，后面 `get_worker_info()` 里还会用它算 `max_req_len`。
- `_init_model_config()` 与 `_init_model_runner()` 是**这条链路上唯一的重活**：前者造 `ModelConfig`，后者造 `ModelRunner`（权重在里面读进来）。
- `model_runner_list` 默认空；只有多层 EAGLE 才会被填成「第 0 层已存在 + 每层各一个 runner」。

`_init_model_config` 的差异点：draft worker 用**投机 draft 模型**的路径/版本，target worker 用主模型：

```533:551:python/sglang/srt/managers/tp_worker.py
    def _init_model_config(self):
        # 中译：构建 ModelConfig。draft worker 用投机 draft 模型的路径/版本，否则用主模型的。
        from sglang.srt.configs.model_config import ModelConfig

        self.model_config = ModelConfig.from_server_args(
            self.server_args,
            model_path=(
                get_model().model_path
                if not self.is_draft_worker
                else get_spec().speculative_draft_model_path
            ),
            model_revision=(
                get_model().revision
                if not self.is_draft_worker
                else get_spec().speculative_draft_model_revision
            ),
            is_draft_model=self.is_draft_worker,
            context_length=self.context_length,
        )
```

`_init_model_runner` 把 `self.ps`、`nccl_port`、内存池、`memory_pool_config`、`draft_attention_backend`、`draft_model_idx` 等原样转发给 `ModelRunner`：

```553:570:python/sglang/srt/managers/tp_worker.py
    def _init_model_runner(self):
        # 中译：创建主 ModelRunner，传入全部并行 rank/size 与内存池等。这是真正加载权重的重操作。
        from sglang.srt.model_executor.model_runner import ModelRunner

        self._model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=get_schedule().mem_fraction_static,
            gpu_id=self.gpu_id,
            ps=self.ps,
            nccl_port=self.nccl_port,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            draft_attention_backend=self.draft_attention_backend,
            draft_model_idx=0 if self.is_multi_layer_eagle else None,
        )
```

### 3.2 多层 EAGLE 与 dLLM（条件分支）

- `_init_multi_layer_eagle_model_runners()`：把主 runner 放进列表，再为 `speculative_num_steps - 1` 个投机步各建一个 `ModelRunner`（`draft_model_idx=i` 区分层），它们**共享同一套内存池**，只有模型层不同。
- `_init_dllm_algorithm()`：`get_exec().dllm.dllm_algorithm` 非空时 `DllmAlgorithm.from_server_args(...)`，否则 `= None`。

### 3.3 tokenizer / processor

只有**非 draft 且未 `skip_tokenizer_init`** 时才加载：

- 多模态模型：`get_processor(...)` → `get_tokenizer_from_processor(...)`；
- 纯文本模型：`get_tokenizer(...)`；
- draft worker 直接 `self.tokenizer = self.processor = None`——因为 `tokenizer_path` 永远指向 target 模型，加载它只会重复。

```415:440:python/sglang/srt/managers/tp_worker.py
        # 中译：跳过分词器初始化时（调用方自行传入 token id），分词器与处理器均置 None。
        if get_serving().skip_tokenizer_init or self.is_draft_worker:
            # A draft worker's tokenizer would only duplicate the target's:
            # tokenizer_path always points at the target model.
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    get_serving().tokenizer_path,
                    tokenizer_mode=get_serving().tokenizer_mode,
                    trust_remote_code=get_model().trust_remote_code,
                    revision=get_model().revision,
                    tokenizer_backend=get_serving().tokenizer_backend,
                    model_name=get_model().model_path,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    get_serving().tokenizer_path,
                    tokenizer_mode=get_serving().tokenizer_mode,
                    trust_remote_code=get_model().trust_remote_code,
                    revision=get_model().revision,
                    tokenizer_backend=get_serving().tokenizer_backend,
                )
```

#### 3.3.1 三份 tokenizer 的对照

同一次服务里，同一个 `tokenizer_path` 会被**独立加载三份**（前端进程一份、每个 Scheduler 进程两份），彼此不共享对象：

| 维度 | 前端 `TokenizerManager` | `Scheduler.self.tokenizer` | `self.tp_worker.tokenizer` |
| --- | --- | --- | --- |
| 代码归属 | `managers/tokenizer_manager.py` | `managers/scheduler.py` · `Scheduler.init_tokenizer`（`:919`） | `managers/tp_worker.py` · `TpModelWorker.__init__`（`:415`） |
| 所在进程 | 前端进程（HTTP server 同进程；`tokenizer_worker_num > 1` 时由 Granian 起多个 worker 进程） | Scheduler 子进程（每 rank 一个） | **同一个** Scheduler 子进程（每 rank 一个） |
| 实例份数 | 每个前端 worker 一份 | 每 rank 一份 | 每 rank 一份 |
| 创建时机 | 前端启动时 | `Scheduler.__init__` 第 12 步 | 第 13 步 `init_tp_model_worker()` → `TpModelWorker.__init__` |
| 置 `None` 条件 | `skip_tokenizer_init=True` | 仅 `skip_tokenizer_init=True` | `skip_tokenizer_init=True` **或** `is_draft_worker=True` |
| 多模态构造参数 | 含 `image_processor_backend` 解析 | 多传 `image_processor_backend=resolve_image_processor_backend(get_mm())` | 不传该参数，只传 `tokenizer_backend` / `model_name` |
| 是否处理用户输入 | **是**：文本 → token ids、多模态预处理 | 否，只接收 token 化后的请求 | 否，只接收 token 化后的请求 |
| 主要用途 | 请求入口：tokenize、封装 `Tokenized*ReqInput`、结果回传用户 | `req.tokenizer = self.tokenizer`（Req/采样参数规范化、语法与 stop 处理）、编码 reasoning parser 的 think_end token 写回 `model_config.think_end_ids`、构建 `_mm_processor`（M-RoPE 回退） | 主路径**只赋值不使用**；仅投机解码侧借用：`ngram_worker.py` 的 `target_worker.tokenizer`、`standalone_worker_v2.py` 的 `target_tokenizer` / `draft_worker.tokenizer` |
| 副作用 | 无 | 有：写 `model_config.think_end_ids`、建 `self._mm_processor` | 无 |
| 与其它组件的通信 | ZMQ（PUSH/PULL）与 Scheduler / Detokenizer 交互 | 进程内直接调用 | 进程内直接调用（被 Scheduler 持有） |
| 启动耗时归属 | 前端进程启动 | `init_tokenizer()` | `init_tp_model_worker()`（第 7 节里「读 checkpoint 之外」那约 3.6 s 固定开销的一部分） |

由此可得 tokenizer 实例总数（纯文本、非 draft 时）：

$$N_{\text{tokenizer}} = \underbrace{2 \times N_{\text{rank}}}_{\text{Scheduler} + \text{TpModelWorker}} + N_{\text{前端 worker}}$$

即 **Scheduler 进程内的两份里，只有 Scheduler 那份真正参与判定逻辑，worker 那份在 target worker 的主流程中不被使用，仅投机解码侧会借用**；draft worker 已显式跳过加载（见上文 `:416`），target worker 那份仍会重复加载一次。

### 3.4 设备 + NCCL 通信组

```441:446:python/sglang/srt/managers/tp_worker.py
        self.device = self.model_runner.device

        # Init nccl groups
        # 中译：取得 NCCL 通信组——pp_group（流水并行组）与 world_group（全体进程组）。
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()
```

`pp_group` 决定前向时是否 `is_last_rank`（只有最后一段 PP 才采样），`world_group` 用于下面的随机种子广播。

### 3.5 跨 rank 同步随机种子

```452:470:python/sglang/srt/managers/tp_worker.py
        # Sync random seed across TP workers.
        # Elastic joiners and last-stage-only draft workers cannot enter the WORLD
        # broadcast, so they reuse the target's already-broadcast seed.
        if random_seed is not None:
            self.random_seed = random_seed
        elif server_args.is_ep_joiner:
            self.random_seed = get_device().random_seed
        else:
            self.random_seed = broadcast_pyobj(
                [get_device().random_seed],
                self.ps.tp_size * self.ps.pp_rank + self.ps.tp_rank,
                self.world_group.cpu_group,
                src=self.world_group.ranks[0],
            )[0]
        set_random_seed(self.random_seed)

        self.enable_overlap = not get_schedule().disable_overlap_schedule
        self.enable_spec = get_spec().speculative_algorithm is not None
        self.hicache_layer_transfer_counter = None
```

三条路径的区别：

| 分支 | 取值来源 | 场景 |
| --- | --- | --- |
| `random_seed is not None` | 调用方显式传入 | draft worker 复用 target 的种子 |
| `server_args.is_ep_joiner` | `get_device().random_seed` | 弹性 EP 新加入的 rank 进不了 WORLD 广播 |
| 其他 | 以 `world_group.ranks[0]` 为 src 广播 | 常规路径，保证各 rank 采样一致 |

---

## 4. `ModelRunner.__init__`：权重加载的宿主

**代码归属**：`python/sglang/srt/model_executor/model_runner.py` · `class ModelRunner` → `ModelRunner.__init__`（`:336`）

`__init__` 很长，按「先轻后重」排：先解析参数与建流，**中间用 `initialize()` 读权重**，最后补上权重更新/导出器。关键顺序：

```833:851:python/sglang/srt/model_executor/model_runner.py
        self.init_memory_saver_adapter()
        self.maybe_init_remote_instance_transfer_engine()
        self.maybe_init_expert_location_metadata()
        self.maybe_init_lplb_solvers()
        self.maybe_init_eplb_manager()
        self.expert_location_updater = ExpertLocationUpdater()
        self.maybe_init_elastic_ep()
        self.init_token_oracle()
        self.sampler = create_sampler()
        self.load_model()
        prepare_moe_topk(
            model=self.model,
            model_config=self.model_config,
            moe_ep_size=self.ps.moe_ep_size,
            moe_ep_rank=self.ps.moe_ep_rank,
        )
```

`__init__` 里几处**顺序敏感**的点：

| 位置 | 动作 | 为什么敏感 |
| --- | --- | --- |
| `:549` | `torch...set_device(ps.gpu_id)` | TransferEngine 初始化需要先有设备上下文 |
| `:564` | `init_shared_mooncake_transfer_engine()` | 必须在 `init_torch_distributed` 之前，避免重复建 TransferEngine |
| `:573` | `init_torch_distributed()` | 建 TP/PP/attention-TP 进程组，并测出 `pre_model_load_memory` |
| `:578` | `self.forward_stream = ...Stream()` | 重叠调度用的前向 stream，随后被 `get_worker_info` 交给调度器 |
| `:589` | `set_offloader(...)` | 按 `dp_rank` 建 CPU offload 加载器 |
| `:606` | `deep_gemm_wrapper.update_deep_gemm_config(gpu_id)` | DeepGEMM JIT 配置，需在权重加载前 |
| `:612` | `self.hisparse_coordinator = None` | 必须在 `initialize()` 之前，CUDA graph capture 要能看到它 |
| `:628` | `self.initialize()` | **权重加载发生在这里** |
| `:661` | `init_weight_updater()` / `init_weight_exporter()` | 依赖已加载的 `self.model` |

`initialize()` 的完整步骤：

```828:874:python/sglang/srt/model_executor/model_runner.py
    def initialize(self):
        # 中译：核心初始化流程。依次完成：内存节省器创建、专家位置/分布记录初始化、
        # 中译：创建 torch 显存节省器适配器（可在空闲时释放/重建显存以节省占用）。
        # 中译：专家并行（EP）。若开启 EPLB（专家负载均衡）且非草稿 worker，则创建 EPLBManager。
        # 中译：创建采样器、加载模型权重、准备 MoE 的 top-k 路由。
        self.init_memory_saver_adapter()
        self.maybe_init_remote_instance_transfer_engine()
        self.maybe_init_expert_location_metadata()
        self.maybe_init_lplb_solvers()
        self.maybe_init_eplb_manager()
        self.expert_location_updater = ExpertLocationUpdater()
        self.maybe_init_elastic_ep()
        self.init_token_oracle()
        self.sampler = create_sampler()
        self.load_model()
        prepare_moe_topk(
            model=self.model,
            model_config=self.model_config,
            moe_ep_size=self.ps.moe_ep_size,
            moe_ep_rank=self.ps.moe_ep_rank,
        )

        # 中译：加载专家权重备份客户端（仅在弹性 EP + 开启专家备份时创建）。
        self.maybe_init_dwdp()

        # Must run before backend/graph init so no draft graph records a
        # routed-experts capture-write kernel.
        if self.is_draft_worker:
            disable_routed_experts_capture_for_draft(self.model)
        self.maybe_init_expert_backup_client()
        self.remote_instance_weight_transporter.maybe_register_and_publish_weight_info()
        self.layer_info: ModelLayerInfo = resolve_layer_indices(
            model=self.model,
            model_config=self.model_config,
            is_draft_worker=self.is_draft_worker,
            spec_algorithm=self.spec_algorithm,
        )
        adjust_hybrid_swa_layer_ids(
            model_config=self.model_config,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            is_hybrid_swa=self.is_hybrid_swa,
        )
        self.maybe_apply_post_load_model_transforms()
        self.maybe_init_lora_manager()
        self.maybe_enable_batch_invariant_mode()
        self.configure_kv_cache_dtype()
```

读法：`initialize()` 里 **只有 `load_model()` 是重活**，其余都是条件分支（EPLB / 弹性 EP / dLLM / LoRA / 确定性推理），默认配置下大多直接 `return`。`layer_info` 必须在权重加载之后才能 `resolve_layer_indices`（要知道实际模型结构）；`configure_kv_cache_dtype()` 也放在最后，保证 KV 量化类型与实际权重一致。

---

## 5. `ModelRunner.load_model()`：真正读 checkpoint

**代码归属**：`python/sglang/srt/model_executor/model_runner.py` · `class ModelRunner` → `ModelRunner.load_model`（`:1342`）

这是 `init_tp_model_worker` 耗时的**绝大部分来源**。骨架（省略条件分支细节）：

```1345:1405:python/sglang/srt/model_executor/model_runner.py
        tic_total = time.perf_counter()
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        # 中译：将线程数设为 1 可减少线程冲突、加速权重加载。
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            maybe_downgrade_dtype_for_legacy_gpu(model_config=self.model_config)

        set_cuda_arch()

        draft_load_format = self.draft_load_format
        self.load_config = build_load_config(
            server_args=self.server_args,
            load_format=draft_load_format,
            tp_rank=self.ps.tp_rank,
            remote_instance_weight_transporter_engine=self.remote_instance_weight_transporter.engine,
            remote_instance_weight_transporter_session_id=self.remote_instance_weight_transporter.session_id,
            draft_model_idx=self.draft_model_idx,
            weight_cache_mode=get_model().weight_cache_mode,
            weight_cache_socket=get_model().weight_cache_socket,
        )
        # ... more code ...
        with self._load_format_scope(draft_load_format):
            loaded = load_model_with_memory_saver(
                server_args=self.server_args,
                model_config=self.model_config,
                load_config=self.load_config,
                device=self.device,
                gpu_id=self.gpu_id,
                memory_saver_adapter=self.memory_saver_adapter,
                is_draft_worker=self.is_draft_worker,
            )
        self.loader = loaded.loader
        self.model = loaded.model
        self.startup_weight_load = loaded.startup_weight_load
```

逐步说明：

| 步骤 | 作用 |
| --- | --- |
| 记录 `before_avail_memory` | 与加载后的可用显存相减，得到 `weight_load_mem_usage` |
| `torch.set_num_threads(1)` | 减少线程争抢，加快权重加载（非 CPU 设备） |
| `maybe_downgrade_dtype_for_legacy_gpu` | sm80 以下没有 bfloat16 支持，回退 float16 |
| `set_cuda_arch()` | 设置 CUDA 架构相关开关（影响后续 kernel 选择） |
| `build_load_config(...)` | 把 `server_args` + ModelOpt 量化配置 + 远程权重传输引擎等打包成 `LoadConfig` |
| `maybe_enable_ipc_weight_cache(...)` | `weight_cache_mode != off` 时把 load_format 改成 `IPC_CACHE`（零拷贝），并推导 per-rank socket |
| `maybe_trigger_remote_instance_nccl_send_group(...)` | 远程实例权重加载场景，提前建 NCCL 发送组 |
| `load_model_with_memory_saver(...)` | **核心**：在 memory-saver region 内选 loader 并加载 |
| `get_offloader().post_init()` | 非 draft worker 的 CPU offload 后置初始化 |
| `maybe_precompile_model_kernels_after_loading()` | 预编译模型 kernel |
| `load_kv_cache_scales(...)` | 读 FP8 KV 的 scale，供后续 `configure_kv_cache_dtype` 用 |
| `resolve_sliding_window_size(...)` | 解析滑窗大小（不同模型字段含义不同） |
| 末尾 | 打日志 `Load weight end. elapsed=... mem usage=...`，并 `dist_barrier_after_load` 让掉队的 rank 对齐 |

日志与计时的落点：

```1445:1463:python/sglang/srt/model_executor/model_runner.py
        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        self.weight_load_mem_usage = before_avail_memory - after_avail_memory
        self.weight_load_time = time.perf_counter() - tic_total
        # Get quantization config from ModelConfig
        # This handles both config.json (standard) and hf_quant_config.json (ModelOpt)
        # 中译：从 ModelConfig 获取量化配置（同时兼容标准的 config.json 与 ModelOpt 的
        #       hf_quant_config.json），仅用于日志展示。
        quant_str = self.model_config.get_quantization_config_log_str()

        # 中译：TODO：确保所有模型都有 `quant_config` 属性，且所有在线量化方法都注册它们
        if self.startup_weight_load is None:
            logger.info(
                f"Load weight end. "
                f"elapsed={self.weight_load_time:.2f} s, "
                f"type={type(self.model).__name__}, "
                f"{quant_str + ', ' if quant_str else ''}"
                f"avail mem={after_avail_memory:.2f} GB, "
                f"mem usage={self.weight_load_mem_usage:.2f} GB."
            )
```

- `self.weight_load_time` 就是最终通过 `TpModelWorker.weight_load_time` → `startup_time` 汇总进 `Engine startup timings (s): load_weight=...` 的那个值。
- `self.weight_load_mem_usage` 被 `emit_metrics_constants()` 上报为 `weight_memory_usage_gb`（在 `init_model_worker` 之后）。
- 注意：**只有 `startup_weight_load is None` 时才打这行日志**——开了 `--is-startup-weight-load-overlap` 时，由 `finalize_startup_weight_load()` 负责后续。

---

## 6. `load_model_with_memory_saver()`：选 loader + 加载器分发

**代码归属**：`python/sglang/srt/model_executor/model_runner_components/load_model_utils.py` · 模块级函数 `load_model_with_memory_saver`（无类）

```502:540:python/sglang/srt/model_executor/model_runner_components/load_model_utils.py
    with memory_saver_adapter.region(
        GPU_MEMORY_TYPE_WEIGHTS,
        enable_cpu_backup=enable_cpu_backup,
    ):
        loader = get_model_loader(
            load_config=load_config,
            model_config=model_config,
        )
        device_config = DeviceConfig(device, gpu_id)
        if server_args.is_startup_weight_load_overlap:
            from sglang.srt.model_executor.model_runner_components.startup_weight_load import (
                StartupWeightLoadManager,
            )

            startup_weight_load = StartupWeightLoadManager.create_from_published_config(
                loader=loader,
                model_config=model_config,
                load_config=load_config,
                device_config=device_config,
                is_draft_worker=is_draft_worker,
            )
            model = startup_weight_load.prepare()
        else:
            model = loader.load_model(
                model_config=model_config,
                device_config=device_config,
            )
        if hasattr(loader, "remote_instance_transfer_engine_weight_info"):
            remote_instance_weight_info = (
                loader.remote_instance_transfer_engine_weight_info
            )
```

要点：

1. **整个加载过程包在 `memory_saver_adapter.region(GPU_MEMORY_TYPE_WEIGHTS, ...)` 里**——权重属于可整体释放/重建的显存区域，供权重热更新 / `enable_weights_cpu_backup` 使用。零拷贝 IPC 模式下会强制关掉 CPU 备份（IPC 映射的权重无法 offload）。
2. **`get_model_loader(load_config, model_config)` 按 `load_format` 分发**到具体 loader（`python/sglang/srt/model_loader/loader.py:4535`）。常见分支：

| load_format | Loader | 说明 |
| --- | --- | --- |
| 默认 / `safetensors` | `DefaultModelLoader` | 从磁盘按 safetensors 读，支持多线程 |
| `layered` | `LayeredModelLoader` | 分层加载，配合显存节省 |
| `sharded_state` / `presharded` | `ShardedStateLoader` / `PreshardedModelLoader` | 预分片权重 |
| `remote` / `remote_instance` | `RemoteModelLoader` 等 | 从远端实例拉权重（PD 场景） |
| `bitsandbytes` / `gguf` | `BitsAndBytesModelLoader` / `GGUFModelLoader` | 量化格式 |
| `dummy` | `DummyModelLoader` | 无权重（测试） |
| `ipc_cache` | 由 `maybe_enable_ipc_weight_cache` 切换 | 零拷贝共享权重 |

3. **`model = loader.load_model(...)` 的返回值就是 `self.model`**——`ModelRunner` 后续所有 `self.model.xxx`（`support_pp` 检测、`resolve_layer_indices`、`configure_kv_cache_dtype`……）都基于它。
4. **`--is-startup-weight-load-overlap`** 时不直接加载，而是建 `StartupWeightLoadManager` 并 `prepare()`（返回带 sentinel 的模型），真正的权重提交推迟到 `init_model_worker` 里的 `finalize_startup_weight_load()`——这就是 `TpModelWorker` 那两个转发方法存在的原因。

---

## 7. 实测耗时与口径

打点方式：在 `init_model_worker` 内用 `perf_counter()` 逐步打点（改动落在 `launch.sh` 实际加载的部署树 `SGLANG_SOURCE` 下，非本仓库）。

环境：单节点 8× NVIDIA H20-3e，`bash launch.sh agg`（TP=8 / EP=8，带 `--skip-server-warmup`）。

| 步骤 | 耗时 (s) | 占 `init_model_worker` 比例 |
| --- | --- | --- |
| **`init_tp_model_worker`（本文）** | **272.09** | **77.4%** |
| `init_all_cuda_graphs` | 78.57 | 22.4% |
| `init_memory_pools` | 0.42 | 0.12% |
| `init_all_attention_backends` | 0.21 | 0.06% |
| 其余 15 个打点 | ≈0 | 条件分支未触发 / 微秒级 |
| **sum** | **351.29** | 100% |

引擎口径对照（同一次启动）：`load_weight=268.43`。

读法：

- `init_tp_model_worker`(272.08) 与引擎口径 `load_weight`(268.43) 相差约 3.6 s，差在**「读 checkpoint 之外」**的固定开销：`ModelConfig.from_server_args`、`init_torch_distributed` 建进程组、`init_shared_mooncake_transfer_engine`、tokenizer 加载、随机种子广播等。
- 这 272 s 里，`load_model()` 的 `load_model_with_memory_saver` 是主体；`weight_load_time` 由于从 `tic_total` 计到加载结束，会略小于整函数耗时。
- 与 `init_all_cuda_graphs` 的分工：**权重先占显存，KV 池和 CUDA graph 后做**。`init_tp_model_worker` 结束时显存里只有权重；KV 池尺寸由 `init_target_memory_pool()` 依「剩余显存」再算。

---

## 8. 常见问题（`init_tp_model_worker` 视角）

| 现象 | 定位 |
| --- | --- |
| 启动卡在这 272 s | 正常：在读 checkpoint。`weight_load_time` 与 `load_weight=` 可对齐；可用 `load_format` 换更快格式（`fastsafetensors` / 预分片 / IPC 权重缓存） |
| `Load weight begin` 后立刻 `-9` | OS OOM Killer 杀了子进程：`dmesg -T \| grep -i oom`。父进程侧会报 `Rank N scheduler died during initialization` |
| 显存不够 | 先看 `Load weight end ... mem usage=... GB`，再看之后 tp_rank0 的 `max_total_num_tokens=..., available_gpu_mem=...` |
| 分数卡显存不均 | `init_torch_distributed` 里有 TP 显存均衡检查；`before_avail_memory` 采样点在此之前 |
| 想省这 272 s | `--is-startup-weight-load-overlap`：`init_tp_model_worker` 只 `prepare()`，权重提交延后到 `finalize_startup_weight_load()`，与 KV 池 / graph capture 重叠 |
| 想复现「启动卡死」 | `SGLANG_TEST_STUCK_SCHEDULER_INIT=<秒>` 在 `init_model_worker` 后人为 sleep（**不在** `init_tp_model_worker` 内） |
| draft worker 没加载 tokenizer | 预期行为：`is_draft_worker=True` 时 `tokenizer = processor = None`；同一 `tokenizer_path` 在服务里共加载几份见 §3.3.1 |

---

## 9. 小结

`init_tp_model_worker` 是**四层套娃里最里层的入口**：

```
Scheduler.init_tp_model_worker           # 4 个入参 → 建 worker 实例
   └─ TpModelWorker.__init__             # 8 件事：ModelConfig / ModelRunner / tokenizer / NCCL 组 / 种子
        └─ ModelRunner.__init__          # 建流、初始化分布式、采样器
             └─ ModelRunner.initialize()
                  └─ ModelRunner.load_model()      # 272 s 的主体
                       └─ load_model_with_memory_saver()  # 按 load_format 选 loader 读权重
```

- **产物**：`self.tp_worker`（内含 `model_runner` / `model` / `tokenizer` / `pp_group` / `world_group` / 随机种子），以及 `model_runner.weight_load_time`、`weight_load_mem_usage`。
- **不做的事**：不分配 KV 池（`init_memory_pools`）、不初始化 attention 后端（`init_all_attention_backends`）、不 capture CUDA graph（`init_all_cuda_graphs`）、不取 `get_worker_info`——这四件事都在 `init_model_worker` 后续步骤。
- **关键约束**：必须晚于 `maybe_revert_pr_fix()`；必须早于 KV 池分配（因为 KV 池尺寸依赖权重占用的显存）。
- **耗时构成**：实测 272 s / 351 s ≈ 77%，几乎全是读 checkpoint 的时间，与「加载参数是否变更」无关——所以压测后如果没改部署参数，只需 flush KV cache，不需要重启模型服务。
