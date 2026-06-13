# 用于 RL 系统的 SGLang

本文档是面向将 SGLang 集成到 RL 和后训练系统中的基础设施团队的实用指南。它聚焦于循环中(rollout、评估、训练、权重同步)的运维痛点,并将其映射到具体的 SGLang API、标志和集成模式。重点在于最大化 rollout 效率、准确性和稳定性,同时在生产环境中保持 rollout-serving 行为的一致。

## 为什么在 RL 生命周期中使用 SGLang?

让我们秉承早期 DeepMind RL 工程的一条指导原则:

**做一个库,而不是一个框架。**

这一理念通过将 SGLang 作为灵活的工具(而非僵化的结构)提供,从而赋能创新。以下是在你的 RL 生命周期中使用 SGLang 的五个理由:

* **细粒度的引擎休眠与唤醒**:促成最大算力的 rollout 和训练
* **开放可用的 Refit 功能**:为 co-location 或 disaggregation 提供多样化方法
* **易于推迟生成**:支持 partial rollout 和专用的 rollout 控制
* **确定性推理**:实现确定性推理,从而做到训练-推理零失配
* **负载均衡 Router**:为高吞吐 rollout 提供 cache-aware 负载均衡

以下各节将详细介绍这些方面。

## 细粒度的引擎休眠与唤醒

Rollout 和训练都是内存密集型的,将它们共置(co-locate)在相同的 GPU 上常常导致内存压力和缓慢的交接。SGLang 提供了一种内存感知的 sleep/wake 机制,在保持 server 进程存活的同时释放 KV cache 和权重,然后为 rollout 恢复它们,而无需完全重启。这避免了每个 RL 步骤期间重复的磁盘 I/O 和 CUDA graph 重新捕获。

在底层,RL 团队通过 [torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver) 使用 CUDA-graph 感知的权重 offload 来保留用于 graph replay 的虚拟内存地址。详情请参阅:[Efficient RL Training - Optimizing Memory Usage in verl](https://hebiao064.github.io/rl-memory-management)。

### Server 标志

在启动 server 时启用 memory saver 支持:

```
--enable-memory-saver
```

### 释放内存

**端点:** `POST /release_memory_occupation`

**请求体:**

| 字段 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `tags` | 要释放哪些内存区域。如果省略,则全部释放。 | `None` | 类型:list[str],取值:`kv_cache`、`weights` |
<!-- python/sglang/srt/managers/io_struct.py#L1381 currently only supports `kv_cache`, `weights` -->
**行为说明:**

- 此调用会断言没有正在进行的请求。在调用之前请确保引擎处于空闲状态。
- 如果释放了 `kv_cache`,SGLang 会刷新缓存;后续请求将按需重建 KV cache。

### 恢复内存

**端点:** `POST /resume_memory_occupation`

**请求体:**

| 字段 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `tags` | 要恢复哪些内存区域。如果省略,则全部恢复。 | `None` | 类型:list[str],取值:`kv_cache`、`weights` |
<!-- python/sglang/srt/managers/io_struct.py#L1393 currently only supports `kv_cache`, `weights` -->

## 开放可用的 Refit 功能

每一步训练完成后,rollout 引擎必须用新权重进行 refit。SGLang 支持三种 refit 策略,以便你匹配自己的基础设施风格(co-located vs disaggregated)和扩展需求。每种策略都映射到一个具有清晰请求 schema 的具体 API。要更深入地了解 SGLang 的权重更新工具,请参阅 [RL System Deep Thinking: Weight Update Mechanisms](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/sys-design/readme-1-EN.md)。

**如何选择:**

- **From disk** 最简单,最适合弹性 rollout 扩展和 checkpointing。
- **From tensor** 最适合 co-located 训练/rollout,当你可以传入内存中的 tensor 时。
- **From distributed** 最适合带专用通信组(NCCL/IB)的 disaggregated 训练/rollout。

### 从磁盘更新权重

**何时使用:**

- 将 checkpoint 保存到磁盘并从磁盘更新权重
- 动态扩展(新的 rollout 实例可以从同一个 checkpoint 加载)

**为什么它效果好:**

这条路径以一些 I/O 开销换取了简单性和灵活性。它能自然地与 checkpointing 集成,并使添加新的 rollout 引擎变得轻而易举:将它们指向同一个 checkpoint 并调用 API 即可。对于高可用性来说,它也是最安全的选择,因为 checkpoint 本身就是真相之源(source of truth)。

**端点:** `POST /update_weights_from_disk`

**请求体:**

| 字段 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `model_path` | 带有新权重的模型路径。 | 必填 | 类型:str |
| `load_format` | 加载权重的格式。 | `None` | 类型:str |
| `abort_all_requests` | 在更新前中止所有正在运行的请求。 | `False` | 类型:bool |
| `weight_version` | server 跟踪的可选权重版本标签。 | `None` | 类型:str |
| `is_async` | 异步执行权重加载。 | `False` | 类型:bool |
| `torch_empty_cache` | 清空 torch 缓存。 | `False` | 类型:bool |
| `keep_pause` | 更新后保持 scheduler 处于暂停状态。 | `False` | 类型:bool |
| `recapture_cuda_graph` | 更新后重新捕获 CUDA graph。 | `False` | 类型:bool |
| `token_step` | 用于 rollout 簿记的训练器步骤 id。 | `0` | 类型:int |
| `flush_cache` | 更新后刷新 KV cache。 | `True` | 类型:bool |

**响应体:**

| 字段 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `success` | 更新是否成功。 | - | 类型:bool |
| `message` | 状态 / 错误消息。 | - | 类型:str |
| `num_paused_requests` | 更新期间暂停的请求数量。 | `0` | 类型:int |

**Python Engine API:** `engine.update_weights_from_disk(model_path, load_format=None)`

**扩散引擎(SGLang-Diffusion):** 扩散引擎暴露相同的 `POST /update_weights_from_disk` 端点,行为如下:

- **全有或全无,带回滚:** 如果任何模块加载失败,所有先前已更新的模块都会通过从原始模型路径重新加载而回滚到原始权重。不会留下部分更新。如果回滚本身失败,异常会向上传播,以便调用方知道模型处于不一致状态。
- **Offload 感知:** 当启用 layerwise offload(`--dit-layerwise-offload`)时,扩散 offload manager 会用小的 `torch.empty((1,))` 占位符替换 GPU 参数,而真正的权重存放在合并的 pinned CPU buffer 中。一个朴素的 `param.data.copy_()` 会因 shape 不匹配而失败。相反,updater 会动态检测活动的 offload manager,并将新权重直接写入它们的 CPU buffer,完全绕过占位符。对于在更新时恰好被预取到 GPU 的任何层,活动的 GPU tensor 也会被更新,以便更改立即生效。这不需要额外的 GPU 内存,也不会扰乱 offload 状态。
- **DTensor 感知:** 通过 `torch.distributed.tensor`(tensor parallelism)分布的参数会通过 `distribute_tensor` 更新,以便每个 shard 都正确放置在正确的 device mesh 上。

**请求体:**

| 字段 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `model_path` | 带有新权重的模型路径。 | 必填 | 类型:str |
| `flush_cache` | 更新后刷新 TeaCache 状态。 | `True` | 类型:bool |
| `target_modules` | 要更新的模块名称列表(例如 `["transformer"]`)。如果省略,所有 `nn.Module` 组件都会被更新。 | `None` | 类型:list[str] |

**响应体:**

| 字段 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `success` | 更新是否成功。 | - | 类型:bool |
| `message` | 状态 / 错误消息。 | - | 类型:str |

> **注意:** 扩散引擎(SGLang-Diffusion)目前不支持热 refit(在推理进行中更新权重)。扩散 scheduler 一次处理一个请求,并在处理下一个请求之前完成整个推理,因此权重更新和推理永远不会并发运行。

### 从 Tensor 更新权重

**何时使用:**

- Co-located 训练和 rollout,训练可以直接提供 tensor
- 快速的内存中更新

**重要约束:**

此策略要求训练进程和 rollout 引擎共享对 tensor 的访问。Co-located 设置必须将模型保持在 GPU 上;将 tensor 移到 CPU 会破坏更新路径。对于高性能 MoE 或专用的注意力 kernel,与 disaggregated rollout 相比,co-location 可能会限制某些优化。

**端点:** `POST /update_weights_from_tensor`

**请求体:**

| 字段 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `serialized_named_tensors` | 每 TP 的序列化 tensor 负载。 | 必填 | 类型:list[str\|bytes] |
| `load_format` | 可选的加载格式选择器。 | `None` | `None`、`direct`、`flattened_bucket`,或自定义 loader 路径字符串 |
| `flush_cache` | 更新后刷新 KV cache。 | `True` | 类型:bool |
| `abort_all_requests` | 在更新前中止所有正在运行的请求。 | `False` | 类型:bool |
| `weight_version` | server 跟踪的可选版本标签。 | `None` | 类型:str |

**注意:** 序列化的 tensor 负载必须使用 `MultiprocessingSerializer.serialize(...)` 创建,并且应当是 base64 安全的字符串。

**Python Engine API:** `engine.update_weights_from_tensor(named_tensors, load_format=None, flush_cache=True)`

### 从分布式组更新权重

**何时使用:**

- Disaggregated 训练和 rollout
- 由 NCCL 或 IB 支持的、从训练 worker 到 rollout worker 的权重广播

**工作原理:**

训练 worker 收集权重(通常在 TP rank 0 上),将它们广播到 rollout 组,然后每个 rollout TP shard 加载它需要的参数。这避免了磁盘 I/O,并保持训练和 rollout 解耦,代价是需要管理一个专用的通信组。

**初始化权重更新组**

**端点:** `POST /init_weights_update_group`

**请求体:**

| 字段 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `master_address` | 组 master 地址。 | 必填 | 类型:str |
| `master_port` | 组 master 端口。 | 必填 | 类型:int |
| `rank_offset` | 本地 rank 映射的偏移量。 | 必填 | 类型:int |
| `world_size` | 总 world size。 | 必填 | 类型:int |
| `group_name` | 组名。 | `weight_update_group` | 类型:str |
| `backend` | 通信后端。 | `nccl` | 类型:str |

**更新权重**

**端点:** `POST /update_weights_from_distributed`

**请求体:**

| 字段 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `names` | 要更新的参数名称。 | 必填 | 类型:list[str] |
| `dtypes` | 每个参数的 dtype 字符串。 | 必填 | 类型:list[str] |
| `shapes` | Tensor 形状。 | 必填 | 类型:list[list[int]] |
| `group_name` | 组名。 | `weight_update_group` | 类型:str |
| `flush_cache` | 更新后刷新 KV cache。 | `True` | 类型:bool |
| `abort_all_requests` | 在更新前中止所有正在运行的请求。 | `False` | 类型:bool |
| `weight_version` | 可选的版本标签。 | `None` | 类型:str |
| `load_format` | 可选的格式选择器。 | `None` | `None` 或 `flattened_bucket` |

**销毁权重更新组**

**端点:** `POST /destroy_weights_update_group`

**请求体:**

| 字段 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `group_name` | 组名。 | `weight_update_group` | 类型:str |

**Python Engine API:**

- `engine.init_weights_update_group(...)`
- `engine.update_weights_from_distributed(names, dtypes, shapes, ...)`
- `engine.destroy_weights_update_group(group_name)`

## 易于推迟生成

多轮 RL rollout 常常受到长尾请求的困扰,这些请求会阻塞整个批次。少数缓慢的交互可能使所有 GPU 停滞,而长尾行为使得性能分析和监控变得困难。

SGLang 暴露了显式的 pause/resume API,以便你可以暂停缓慢的请求并在之后继续它们。这种模式与 [APRIL](https://arxiv.org/abs/2509.18521) 等系统相匹配:在收集到足够的响应后即终止,并在下一步回收未完成的响应。其结果是在不丢弃部分工作的情况下提高 GPU 利用率。

`pause_generation` --- 更新权重 --- `continue_generation` 是在从训练更新权重时正确的执行流程。只有当 SGLang 没有在主动处理推理任务时,才能进行更新。

### 暂停生成

**端点:** `POST /pause_generation`

**请求体:**

| 字段 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `mode` | 暂停模式。 | `abort` | `abort`、`retract`、`in_place` |

**模式:**

- `abort`:默认行为,与设置了 `abort_all` 的 `abort` 端点相同。来自 `waiting_queue` 和 `running_queue` 的待处理请求会立即返回给调用方。
- `retract`:将引擎置于 "paused" 状态。将正在运行的请求移回 waiting queue。KV cache 可被刷新并在之后重新计算。
- `in_place`:将引擎置于 "paused" 状态,但不改变请求的状态。正在运行的请求依赖 KV cache 的可用性来继续,因此任何后续的 `flush_cache` 调用都将不会成功。

### 继续生成

**端点:** `POST /continue_generation`

## 确定性推理

在许多 RL 技术栈中,rollout 和训练是用不同的 kernel 或批处理行为实现的。即使权重完全相同,token 概率也可能漂移,从而悄然破坏 on-policy 假设。这就是训练-推理失配问题。

SGLang 支持一种确定性推理模式,可降低不同 batch shape 间的非确定性。这缓解了运行时批处理和 kernel 选择引入的方差。要进一步实现真正的 on-policy 训练,你需要修改训练引擎以使用相同的确定性 kernel。关于实现细节,请参阅这些 miles 示例:[True On-Policy](https://github.com/radixark/miles/tree/main/examples/true_on_policy) 和 [True On-Policy for VLM](https://github.com/radixark/miles/tree/main/examples/true_on_policy_vlm)。更多背景信息请参阅博客文章 [Let Speed Be With Stability: All-In-One Solution to Training-Inference Mismatch with Miles](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/mismatch/blog-en.md)。

**Server 标志:**

```
--enable-deterministic-inference
```

更多详情请参阅 [Deterministic Inference](deterministic_inference.md)

## 负载均衡 Router

SGLang Model Gateway 是大规模 RL rollout 推荐的控制平面。它提供异步、非阻塞的请求处理,cache-aware 负载均衡,以及跨 rollout 和 reward server 的容错路由。这让你能够在保持 GPU 饱和的同时,避免长尾停滞以及脆弱的、引擎本地的并发逻辑。它已在 GLM 4.5+ 模型的训练中部署,并在生产级大规模 RL 工作负载中被证明是高效的。

对 RL 基础设施的关键收益:

- **异步非阻塞效率**:SGLang 原生的异步 server/router 架构(HTTPS/gRPC)自动管理并发。这保证了最大的 GPU 饱和度和有效的 continuous batching,而无需工程师进行复杂的手动实现。
- **弹性与容错**:通过将 reward model 和 rollout 封装为独立的 server,SGLang 在逻辑上和物理上将它们解耦。该架构为大规模分布式训练提供了健壮的灾难恢复能力;如果一个 server 失败,router 会自动将流量重定向到健康的节点,确保训练过程不中断地继续。
- **训练-推理对齐**:在训练和推理中都使用 SGLang Model Gateway 可确保 "所见即所得"。这消除了得分差异,以及因训练和部署使用不同引擎而常常导致的痛苦的后端对齐问题。
- **动态负载均衡与长尾缓解**:与静态分区不同,SGLang Model Gateway 为多轮 RL 实现了请求级别的动态分派。它可以将一次对话的不同轮次分布到不同的 server 上,以平衡工作负载,并消除由序列长度变化引起的长尾延迟。

关于部署和配置,请参阅:[SGLang Model Gateway](sgl_model_gateway.md)
