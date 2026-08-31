# srt/eplb 学习指南

> 本文档是 `srt/eplb`（专家并行负载均衡，EPLB）目录的中文学习指南：前半部分为目录/文件概览，后半部分为按运行时调用链组织的深入代码走读计划。

## 目录用途
该目录实现专家并行负载均衡（Expert Parallelism Load Balancing, EPLB）。在 MoE 模型的专家并行部署中，它负责统计各专家的 token 负载分布、计算逻辑专家到物理副本的映射、周期性地重平衡专家在各 GPU/节点上的放置，并在运行时动态更新专家位置与分发路由，从而缓解专家负载不均带来的性能瓶颈。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化文件（空）。 |
| `eplb_manager.py` | `EPLBManager` 调度入口，按配置的迭代周期触发专家负载统计与重平衡流程，驱动专家位置更新。 |
| `expert_distribution.py` | 专家负载分布记录器，采集并聚合各 forward pass/rank 的物理专家计数，支持 per_pass 等记录模式并提供全局单例。 |
| `expert_location.py` | `ExpertLocationMetadata` 专家位置元数据，维护物理↔逻辑专家映射，结合 eplb 算法计算专家放置。 |
| `expert_location_dispatch.py` | `ExpertLocationDispatchInfo` 专家分发信息，按 static/random 算法生成逻辑专家到物理副本的运行时分发映射。 |
| `expert_location_updater.py` | `ExpertLocationUpdater`，在重平衡后通过点对点通信（P2POp）在各 rank 间迁移专家权重并更新位置元数据，支持弹性 EP。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `eplb_algorithms` | 专家重平衡算法实现（DeepSeek 原版、向量化版、弹性感知版）。详见其 `README_zh.md`。 |
| `eplb_simulator` | EPLB 离线仿真/分析工具，从记录的专家分布数据回放并评估均衡策略。详见其 `README_zh.md`。 |

### `eplb_algorithms/` 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 定义 `EplbAlgorithm` 枚举与统一入口 `rebalance_experts`/`compute_algorithm`，按算法类型分派到 deepseek / deepseek_vec / elasticity_aware，并支持层次化（hierarchical）变体。 |
| `deepseek.py` | DeepSeek 官方 EPLB 算法（自其仓库拷贝），含均衡装箱 `balanced_packing`、专家复制与层次化重平衡 `rebalance_experts_hierarchical`。 |
| `deepseek_vec.py` | DeepSeek 算法的向量化实现，按分块（chunkwise）批量生成冗余专家与物理↔逻辑映射，提升大规模计算效率。 |
| `elasticity_aware.py` | 弹性感知重平衡算法，依据 `active_ranks` 在部分 rank 失效/变动时回退到全局负载均衡策略并复用 deepseek 层次化算法。 |
| `README_zh.md` | 该子目录的中文说明文档。 |

### `eplb_simulator/` 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 导入并暴露 `reader` 模块。 |
| `reader.py` | 读取 `per_pass` 模式记录的专家分布数据（`.pt`），聚合各 rank 的全局物理专家计数，并复用 `_convert_global_physical_count_to_logical_count` 转换为逻辑计数，用于离线评估重平衡策略。 |
| `README_zh.md` | 该子目录的中文说明文档。 |

---

## 深入学习 eplb 代码路径计划

本计划按 **运行时真实调用链** 组织，从触发入口出发逐层深入，边读代码边验证核心数据结构，帮助建立完整的 EPLB 心智模型。建议按阶段顺序推进，每个阶段先读"关键代码路径"，再对照"要点/自测"确认理解。

### 前置知识
- MoE / 专家并行（EP）基础：逻辑专家（logical expert）与物理专家（physical expert / 副本）的区别。
- 分布式基础：`torch.distributed` 的 world size、rank、`P2POp` 点对点通信、`batch_isend_irecv`。
- 相关配置项（`ServerArgs`，定义见 `python/sglang/srt/server_args.py`）：

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `enable_eplb` | `bool` / `False` | 是否启用 EPLB（专家并行负载均衡）。启用时要求 `ep_size > 1`；若未显式设置 `expert_distribution_recorder_mode` 会自动设为 `"stat"`，且 `ep_dispatch_algorithm` 默认设为 `"static"`。 |
| `ep_num_redundant_experts` | `int` / `0` | 每层在整个 EP group 中额外分配的**冗余物理专家**总数。它使 `num_physical_experts = num_logical_experts + ep_num_redundant_experts`，是热点专家可被多副本分担的前提。 |
| `eplb_algorithm` | `str` / `"auto"` | EPLB 重平衡算法。`"auto"` 会按 `num_groups % num_nodes` 自动在 `deepseek` / `deepseek_hierarchical` 间选择；可显式指定 `deepseek*`、`deepseek_vec*`、`elasticity_aware*`。启用弹性 EP 时 `"auto"` 会被改写为 `"elasticity_aware"`。 |
| `eplb_rebalance_num_iterations` | `int` / `1000` | 每隔多少次 forward 迭代自动触发一次重平衡。必须 `>= expert_distribution_recorder_buffer_size`（否则循环缓冲区含陈旧数据，`EPLBManager.__init__` 有断言）。 |
| `eplb_rebalance_layers_per_chunk` | `Optional[int]` / `None` | 每次 forward pass 重平衡的层数（分块大小）。`None` 表示一次性处理所有层（内部按超大 chunk `1000000` 处理）；设为较小值可把权重迁移分摊到多次 forward，降低单次停顿。 |
| `eplb_min_rebalancing_utilization_threshold` | `float` / `1.0` | 触发重平衡的最小 GPU 平均利用率阈值，取值 `[0.0, 1.0]`。当窗口内平均利用率**高于**该阈值时跳过重平衡；默认 `1.0` 表示始终重平衡。 |
| `expert_distribution_recorder_mode` | `Optional[Literal["stat","stat_approx","per_pass","per_token"]]` / `None` | 专家分布记录器模式，决定负载统计的粒度：`stat`（聚合统计）、`stat_approx`（近似统计）、`per_pass`（按 forward 记录，供 `eplb_simulator` 离线回放）、`per_token`（按 token 记录，最细）。启用 EPLB 时若为 `None` 会自动设为 `"stat"`。 |
| `expert_distribution_recorder_buffer_size` | `Optional[int]` / `None` | 记录器循环缓冲区大小。`None` 时默认取 `eplb_rebalance_num_iterations`（否则回退 `1000`）；设为 `-1` 表示无限缓冲。 |

> 关联行为参见 `server_args.py` 的 `_handle_eplb_and_dispatch` / `_handle_elastic_ep` / `_handle_expert_distribution_metrics`。

### 阶段 0：全局数据模型与术语（先建立坐标系）
- 关键代码路径：
  - `expert_location.py` → `ExpertLocationMetadata`（dataclass 字段与 `@property`：`physical_to_logical_map`、`logical_to_all_physical_map`、`logical_to_all_physical_map_num_valid`、`logical_to_rank_dispatch_physical_map`；`num_layers/num_physical_experts/num_local_physical_experts/num_logical_experts/ep_size`）。
- 要点/自测：
  - 画出 `physical→logical`（多对一）与 `logical→all_physical`（一对多，含冗余副本）两张映射表的形状 `(layers, ...)`。
  - 解释 `num_physical_experts = num_logical_experts + ep_num_redundant_experts` 的含义。

### 阶段 1：调度入口与生命周期（EPLBManager）
- 关键代码路径：
  - `eplb_manager.py` → `EPLBManager.__init__`（读取配置、断言 `num_iterations >= buffer_size`、启动分布记录器 `start_record()`）。
  - `_entrypoint()` 生成器 + `on_forward_pass_end()`（每次 forward 结束调用 `next()`，累计到周期后触发 `rebalance()`）。
  - `rebalance()`（`dump_record` → `_check_rebalance_needed` → `init_by_eplb` → 按 chunk 调 `model_runner.update_expert_location`）。
- 要点/自测：
  - 为什么用 **生成器 + yield** 来控制周期而不是简单计数器？（分块 rebalance 时可跨 forward 让出控制权）
  - `_check_rebalance_needed` 依据 `average_utilization_rate_over_window` 与阈值决定是否跳过，理解"高利用率时不重平衡"的动机。
  - `_compute_update_layer_ids_chunks` 如何按 `eplb_rebalance_layers_per_chunk` 分块。
- 上游触发点（可选追溯）：搜索 `EPLBManager` / `on_forward_pass_end` 在 `model_executor/model_runner.py` 与 scheduler 中的调用位置。

### 阶段 2：负载统计与采集（ExpertDistributionRecorder）
- 关键代码路径：
  - `expert_distribution.py` → `ExpertDistributionRecorder.init_new`（Noop vs Real 分派）、全局单例 `get_global_expert_distribution_recorder`/`set_...`。
  - `_ExpertDistributionRecorderReal`：`with_forward_pass` / `with_current_layer` 上下文管理器、`on_select_experts`、`on_deepep_dispatch_normal`、`on_deepep_dispatch_low_latency` 三个采集 hook、`_on_hook` 分派、`dump_record`。
  - `_SinglePassGatherer`（单次 forward 的收集器）与其子类，理解 per_pass / 累积 buffer 模式。
- 要点/自测：
  - 三个 hook 分别在什么路径被调用？（在 topk 选择、DeepEP normal / low-latency dispatch 处埋点）
  - `dump_record(output_mode="object")` 返回的 `logical_count` 与 `average_utilization_rate_over_window` 是如何聚合出来的。

#### `_SinglePassGatherer` 采集的是什么

`_SinglePassGatherer` 是 EPLB 负载采集的**最底层"单次统计单元"**，回答的问题是：**这一次 forward pass 里，每个 MoE 层的每个专家各接收了多少 token**——即重平衡所依赖的"专家负载分布"。

- **核心产出**：`collect()` 返回的 dict 中，关键字段是 `global_physical_count`，形状 `[num_layers, num_physical_experts]`，含义是"第 `i` 层第 `j` 个**物理专家**在本次 forward 被路由到的 token 数"。计数越高即越热门，是判断"谁需要复制/重平衡"的直接依据。

- **数据来源（三个埋点 hook）**：数据在模型前向过程中通过 hook 喂入，不同子类使用不同 hook（因为不同 MoE 后端的负载信息来源与形态不同）：

| Hook | 埋点位置 | 采集内容 |
| --- | --- | --- |
| `on_select_experts` | topk 路由选择处 | 每个 token 选中的 topk 专家 id（`topk_ids`），`-1` 为无效位 |
| `on_deepep_dispatch_normal` | DeepEP normal dispatch | 本地物理专家计数 + 各维度分发统计（近似统计）|
| `on_deepep_dispatch_low_latency` | DeepEP low-latency dispatch | 本地物理专家计数 |

- **子类分工**（由 `init_new` 按 `server_args` 分派）：
  - `_SelectExpertsSinglePassGatherer`（默认兜底）：从 topk 选择处用 `scatter_add` 直接累加各**全局**物理专家计数。
  - `_DeepepNormalSinglePassGatherer` / `_DeepepLowLatencySinglePassGatherer`：从 DeepEP 的 dispatch 处按**本地**物理专家累加，`collect` 时再按 rank 映射为全局计数。
  - `_DetailSinglePassGatherer`（`per_token` 模式）：采集最细，除计数外还完整保留每个 token 的 `topk_ids`、输入元信息（`input_ids`/`positions`）与 dispatch 统计，主要用于离线复盘。

- **在链路中的位置**：单次的物理专家计数会被 `ExpertDistributionRecorder` 跨多次 forward 累积（滑动窗口），并按专家位置映射聚合成**逻辑专家**的 `logical_count`，后者才是重平衡算法（阶段 3）真正的输入。

  ```
  每次 forward（_SinglePassGatherer 采集单次）
     └─ on_select_experts / on_deepep_dispatch_* 埋点喂数据
     └─ collect() 产出 global_physical_count
          └─ ExpertDistributionRecorder 跨多次 forward 累积（滑动窗口）
               └─ dump_record() 得到窗口内的 logical_count + 平均利用率
                    └─ EPLBManager.rebalance() 据此重算专家位置
  ```

### 阶段 3：重平衡算法（eplb_algorithms）
- 关键代码路径：
  - `eplb_algorithms/__init__.py` → `EplbAlgorithm` 枚举、`compute_algorithm`（`auto` 如何选择 hierarchical）、`rebalance_experts` 分派逻辑。
  - `deepseek.py` → `balanced_packing`（均衡装箱）、`replicate_experts`（专家复制/加冗余）、`rebalance_experts_hierarchical` 与非层次化路径。
  - `deepseek_vec.py` → 向量化 / chunkwise 实现，对比其与 `deepseek.py` 的等价性与性能差异。
  - `elasticity_aware.py` → `active_ranks` 传入、rank 失效时回退到全局均衡并复用 deepseek 层次化。
- 要点/自测：
  - 手推 `balanced_packing` 的贪心装箱过程（把带权重的专家均匀分到桶/GPU）。
  - 说明 $\text{num\_groups} \bmod \text{num\_nodes} = 0$ 时为何倾向 hierarchical。
  - 弹性感知与普通算法的唯一关键差异是什么（`active_ranks`）。

#### EPLB 算法原理与横向比较

`EplbAlgorithm` 中虽然有 6 个枚举值，但可以理解为 **$3$ 个算法家族 $\times$ $2$ 种拓扑模式**：

- 算法家族决定“依据什么负载目标分配冗余副本”：`deepseek`、`deepseek_vec`、`elasticity_aware`。
- 拓扑模式决定“是否保留专家组和节点边界”：普通版本执行全局均衡，`*_hierarchical` 执行分层均衡。

所有算法解决的共同问题是：给定逻辑专家负载和有限的冗余物理专家名额，为热点逻辑专家增加副本，并安排物理副本所在的位置，使最繁忙 GPU 或副本的预计负载尽可能低。若逻辑专家 $e$ 的负载为 $w_e$，副本数为 $c_e$，则每个副本的理想平均负载近似为：

$$
\ell_e = \frac{w_e}{c_e}
$$

算法不断把冗余名额分给当前 $\ell_e$ 较高的专家，然后再把副本均匀装箱到节点或 GPU。它优化的是基于历史 token 计数估算出的**放置均衡**，并不直接迁移 token；新映射生效后，运行时分发逻辑才会把后续 token 路由到不同副本。

##### 1. `deepseek`：聚合负载上的贪心复制与装箱

统一入口先沿统计步维度求和：

$$
w_{l,e} = \sum_t x_{t,l,e}
$$

其中 $x_{t,l,e}$ 是统计步 $t$、MoE 层 $l$、逻辑专家 $e$ 的 token 数。因此，`deepseek` 只关心整个统计窗口的累计热度，不保留每一步的波动信息。

核心过程如下：

1. 每个逻辑专家先拥有一个物理副本。
2. 对每个冗余名额，选择当前 $\frac{w_e}{c_e}$ 最大的逻辑专家，将其副本数 $c_e$ 加一。
3. 按副本分摊后的预计负载，使用 LPT 风格的贪心装箱：先处理重副本，再放入当前总负载最低且尚未满员的 GPU。
4. 生成物理到逻辑、逻辑到物理以及逻辑专家副本数三个映射。

特点：参考实现最直观、结果易解释；但它把时间维度压缩成总和。例如两个专家的分步负载分别为 $[10,0]$ 和 $[0,10]$，聚合后都为 $10$，算法无法利用“热点出现于不同时刻”这一信息。

##### 2. `deepseek_vec`：保留时间步峰值的 chunkwise 分配

`deepseek_vec` 接收完整的三维负载 $x_{t,l,e}$，不是简单把 `deepseek` 改写成批量张量操作。它在每次分配冗余副本时，会评估“把副本给哪个专家后，各统计步中最热专家的负载之和最小”。对候选专家 $e$，其近似目标可写为：

$$
J(e) = \sum_t \max_j
\begin{cases}
\dfrac{x_{t,l,j}}{c_j + 1}, & j=e \\
\dfrac{x_{t,l,j}}{c_j}, & j\ne e
\end{cases}
$$

选择使 $J(e)$ 最小的专家获得下一个副本。这让算法能区分稳定热点和随时间交替的热点，更关注统计窗口内各步的峰值，而不是只看累计总量。实现还具有以下特点：

- 通过 `make_redundant_experts_chunkwise` 在 chunk 内并行处理层和专家组；冗余名额仍按轮次迭代分配。
- 用与专家 id 相关的微小扰动打破分数相同的情况，使 `argmax` 选择确定化。
- 分配副本后，按累计平均负载排序，并通过蛇形排列把重、轻副本分散到 GPU。
- 普通版本把全部物理专家视为一个 chunk；分层版本以每个节点的物理专家空间为一个 chunk。

因此，`deepseek_vec` 的主要优势不只是“向量化更快”，还包括**保留统计步信息、优化逐步峰值**。代价是约束更多：物理专家、逻辑专家和冗余专家数量都必须能按 chunk 整齐划分；其启发式目标与 `deepseek` 不完全相同，所以两者不保证产生相同映射。

##### 3. `elasticity_aware`：仅在活跃 rank 上重平衡

`elasticity_aware` 的副本分配与装箱核心复用 `deepseek.rebalance_experts_hierarchical`，区别在于额外读取 `active_ranks`：

- 所有 rank 都活跃时，其行为分别等价于对应的 `deepseek` 全局版或分层版。
- 存在失效 rank 时，不论是否选择 `_hierarchical`，都会强制退化为全局策略，只在活跃 rank 和它们实际拥有的物理槽位上计算映射。
- 算法随后把失效 rank 对应的槽位补回输出张量，并修正物理专家索引，使输出形状仍与完整 EP world 一致；这些占位槽位不会被记录为逻辑专家的有效副本。

发生缩容时放弃分层策略，是因为原来的“专家组均匀分节点”前提可能已被破坏；先保证所有逻辑专家在存活 rank 上仍有可用副本，比维持原节点亲和性更重要。该算法解决的是**可用性与形状兼容性**，并不会让失效 rank 继续承载负载。

##### 4. 全局版与分层版

全局版实际上仍调用 `rebalance_experts_hierarchical`，但传入 $\text{num\_groups}=1$、$\text{num\_nodes}=1$，把全部设备视为一个资源池。分层版则执行三层约束：

1. 统计每个专家组的负载，并把完整专家组均衡分配到节点；同一组不会被拆到不同节点。
2. 在各节点内部独立分配冗余专家。
3. 在节点内部把物理专家均衡装箱到 GPU。

| 模式 | 优点 | 代价与前提 | 更适合的场景 |
| --- | --- | --- | --- |
| 全局 | 放置自由度最大，通常更容易得到更低的全局最大负载 | 可能增加跨节点 token 分发通信，不保留专家组局部性 | 单节点；网络层级不明显；组数无法均分到节点；弹性缩容 |
| 分层 | 保留专家组的节点亲和性，减少昂贵的跨节点通信 | 受节点边界约束，纯计算负载可能不如全局版均匀；要求组数、GPU 数、物理专家数满足整除条件 | 多节点且节点内互联明显快于节点间网络 |

分层算法并非必然更快：最终吞吐取决于**计算不均衡代价**与**跨节点通信代价**的权衡。需要结合真实模型、并行拓扑和流量分布压测。

##### 5. 六种配置的选择建议

| 配置值 | 负载信息 | 拓扑策略 | rank 故障感知 | 建议用途 |
| --- | --- | --- | --- | --- |
| `deepseek` | 各步求和后的累计负载 | 全局 | 否 | 通用、单节点或希望优先追求全局均衡；也是不满足分层条件时的安全选择 |
| `deepseek_hierarchical` | 各步求和后的累计负载 | 节点→节点内 GPU | 否 | 多节点、专家组可均分到节点，且跨节点通信昂贵 |
| `deepseek_vec` | 保留每个统计步的负载 | 全局 chunk | 否 | 负载随请求或时间明显波动，希望降低逐步峰值 |
| `deepseek_vec_hierarchical` | 保留每个统计步的负载 | 节点 chunk→节点内 GPU | 否 | 同时需要逐步峰值感知和多节点通信局部性，并满足严格整除约束 |
| `elasticity_aware` | 各步求和后的累计负载 | 正常时全局；缩容时仅活跃 rank 全局 | 是 | 启用 Elastic EP，rank 集合可能动态变化 |
| `elasticity_aware_hierarchical` | 各步求和后的累计负载 | 全员时分层；缩容时回退全局 | 是 | Elastic EP 的稳定阶段希望保留节点局部性 |

当前 `compute_algorithm("auto", ...)` 只在 `deepseek` 与 `deepseek_hierarchical` 之间选择：当 `num_groups` 非空且 $\text{num\_groups} \bmod \text{num\_nodes} = 0$ 时选择分层版，否则选择全局版；它不会自动选择 `deepseek_vec`。启用 Elastic EP 时，参数处理阶段会把默认 `auto` 改写为 `elasticity_aware`。因此，若要使用向量版或弹性感知分层版，需要显式配置。

##### 6. 共同限制与评估方法

- **整除约束**：分层算法至少要求 $\text{num\_logical\_experts} \bmod \text{num\_groups} = 0$、$\text{num\_groups} \bmod \text{num\_nodes} = 0$、$\text{num\_gpus} \bmod \text{num\_nodes} = 0$ 以及 $\text{num\_physical\_experts} \bmod \text{num\_gpus} = 0$；向量版还要求各 chunk 能均分逻辑专家与冗余名额。
- **历史负载假设**：所有算法都假设下一个窗口与统计窗口具有一定相似性。流量突变时，即使历史映射最优，也可能很快失效。
- **副本均分假设**：估算使用 $\frac{w_e}{c_e}$，隐含运行时能把同一逻辑专家的 token 较均匀地分发到各物理副本；若 dispatch 严重偏斜，放置算法的收益会打折。
- **算法开销不是全部成本**：应同时考虑映射计算时间、权重迁移停顿、稳态 GPU 最大负载、跨节点通信量和端到端吞吐/延迟。
- **推荐验证方式**：使用 `per_pass` 记录同一份真实负载，通过 `eplb_simulator` 分别回放候选算法，先比较预计最大负载和跨节点流量，再进行在线压测；不要仅根据算法名称判断优劣。

### 阶段 4：位置元数据的构建与更新（ExpertLocationMetadata）
- 关键代码路径：
  - `expert_location.py` → `_init_common`、`init_trivial`、`init_by_mapping`、`init_by_eplb`（调用 `eplb_algorithms.rebalance_experts` 生成新映射）。
  - `update(...)`（按 `update_layer_ids` 局部更新映射张量）、CPU 副本字段的用途。
  - 全局单例 `get_global_expert_location_metadata` / `set_...`。
- 要点/自测：
  - `init_by_eplb` 如何把算法输出（物理/逻辑映射）填充回 `ExpertLocationMetadata` 各字段。
  - 为什么需要 `*_cpu` 副本（性能/避免频繁 device↔host 拷贝）。

### 阶段 5：权重迁移与位置切换（ExpertLocationUpdater）
- 关键代码路径：
  - `expert_location_updater.py` → `ExpertLocationUpdater.update`（`_first_execution` 清缓存、取旧元数据、`_update_expert_weights`、旧元数据 `update`）。
  - `_update_expert_weights` 内部：如何基于新旧映射 diff 出需迁移的物理专家，构造 `P2POp` 收发，处理弹性 EP 下的 `missing_logical_experts`。
- 要点/自测：
  - 迁移权重时如何避免死锁 / 保证收发配对（`batch_isend_irecv` 的用法）。
  - 弹性 EP 场景下"缺失逻辑专家"如何被识别并向上返回。

### 阶段 6：运行时分发（ExpertLocationDispatchInfo）
- 关键代码路径：
  - `expert_location_dispatch.py` → `ExpertLocationDispatchInfo.init_new`、`transform_select_experts_inputs`、`topk_ids_logical_to_physical`。
  - static vs dynamic：`_topk_ids_logical_to_physical_static` / `_topk_ids_logical_to_physical_dynamic`。
- 要点/自测：
  - forward 时逻辑专家 id 如何被映射到具体物理副本，static 与 random/dynamic 策略的取舍。
  - 该路径与阶段 2 的 `on_select_experts` 埋点如何配合。

### 阶段 7：离线仿真与验证（eplb_simulator）
- 关键代码路径：
  - `eplb_simulator/reader.py` → 读取 `per_pass` 的 `.pt`、聚合全局物理计数、`_convert_global_physical_count_to_logical_count`。
- 要点/自测：
  - 用一份 dump 数据离线跑不同算法，比较均衡指标，理解如何"无需上机即可评估策略"。

### 端到端串联（一句话主线）
`forward 结束` → `EPLBManager.on_forward_pass_end`（阶段 1）→ 周期到达触发 `rebalance` → `ExpertDistributionRecorder.dump_record` 出 `logical_count`（阶段 2）→ `ExpertLocationMetadata.init_by_eplb` 调 `eplb_algorithms.rebalance_experts` 算新放置（阶段 3、4）→ `ModelRunner.update_expert_location` → `ExpertLocationUpdater.update` 迁移权重并切换元数据（阶段 5）→ 后续 forward 经 `ExpertLocationDispatchInfo` 按新映射分发（阶段 6）；`eplb_simulator` 可离线复盘（阶段 7）。

### 建议的动手练习
1. 在 `rebalance()` / 各 hook 处加日志，用小 MoE 模型跑几轮，观察 `logical_count` 与映射变化。
2. 打开 `expert_distribution_recorder_mode=per_pass` dump 数据，用 `eplb_simulator/reader.py` 加载并跑不同 `eplb_algorithm` 对比均衡度。
3. 单元级验证 `balanced_packing`：构造已知权重，手算期望分桶结果并断言。
