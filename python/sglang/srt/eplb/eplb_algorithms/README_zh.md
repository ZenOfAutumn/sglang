# srt/eplb/eplb_algorithms

## 目录用途
该目录汇集专家并行负载均衡（EPLB, Expert Parallelism Load Balancer）的具体重平衡算法。给定各专家的 token 负载统计，这些算法计算逻辑专家到物理副本（含冗余副本）的放置方案，使各 GPU/节点的负载尽可能均衡。`__init__.py` 提供统一入口，按 `EplbAlgorithm` 枚举分派到对应实现，并支持层次化（hierarchical）变体。

## 核心概念

在阅读具体算法前，先厘清几个贯穿全目录的术语：

- **逻辑专家（logical expert）**：模型定义里的专家，数量固定为 `num_logical_experts`。路由（router/topk）产出的永远是逻辑专家 id。
- **物理专家 / 副本（physical expert / replica）**：逻辑专家在设备上的实际实例，数量为 `num_physical_experts`（= `num_replicas`）。一个热门逻辑专家可以有多个物理副本以分摊负载。
- **冗余专家（redundant expert）**：`num_physical_experts - num_logical_experts` 个额外副本，正是这些冗余名额被分配给负载最高的逻辑专家。
- **负载统计（weight / tokens_per_expert）**：窗口内各逻辑专家被路由到的 token 计数，是所有算法的输入依据（越大表示越热门）。
- **三个输出映射**（几乎所有算法都返回这三个，是下游 `ExpertLocationMetadata` 的核心）：
  - `physical_to_logical_map` `[layers, num_physical]`：每个物理副本对应哪个逻辑专家。
  - `logical_to_physical_map` `[layers, num_logical, X]`：每个逻辑专家拥有的物理副本 id 列表（不足 X 的用 `-1` 填充）。
  - `logical_count` / `expert_count` `[layers, num_logical]`：每个逻辑专家拥有的物理副本个数。
- **层次化（hierarchical）**：区分节点内（node，NVLink 更快）与节点间网络。层次化策略先把专家组打包到节点、再在节点内构造冗余、最后打包到 GPU，从而让高频通信尽量留在节点内。非层次化（global）则把 `num_groups=num_nodes=1` 当作单一大池统一均衡。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 定义 `EplbAlgorithm` 枚举与统一入口 `rebalance_experts`，按算法类型分派到 deepseek / deepseek_vec / elasticity_aware；`compute_algorithm` 负责把 `"auto"` 解析成具体算法。 |
| `deepseek.py` | DeepSeek 官方 EPLB 算法（自其仓库拷贝），含均衡装箱 `balanced_packing`、专家复制 `replicate_experts` 与层次化重平衡 `rebalance_experts_hierarchical`。逐层用 Python 循环实现，逻辑最直观。 |
| `deepseek_vec.py` | DeepSeek 算法的向量化实现，按分块（chunkwise）批量生成冗余专家与物理↔逻辑映射，避免逐层 Python 循环，提升大规模计算效率；区分 prefill（层次化）与 decode（非层次化）两条路径。 |
| `elasticity_aware.py` | 弹性感知重平衡算法，依据 `active_ranks` 在部分 rank 失效/变动时回退到全局负载均衡策略并复用 deepseek 层次化算法，再把失效 rank 的位置用占位副本补回，保证输出形状不变。 |

## 各文件详细介绍

### `__init__.py`（统一入口与算法选择）

- **`EplbAlgorithm` 枚举**：列出全部可选算法——`deepseek` / `deepseek_hierarchical` / `deepseek_vec` / `deepseek_vec_hierarchical` / `elasticity_aware` / `elasticity_aware_hierarchical`。带 `_hierarchical` 后缀的即启用层次化策略。
- **`rebalance_experts(...)`**：唯一对外入口。根据传入的 `algorithm` 分派：
  - `deepseek*` → 先对 `tokens_per_expert.sum(dim=0)`（把多个统计步聚合成 `[layers, num_logical]`）再调用 `deepseek.rebalance_experts`；
  - `deepseek_vec*` → 直接把带步维度的 `tokens_per_expert` 交给 `deepseek_vec.rebalance_experts`；
  - `elasticity_aware*` → 额外从 `ElasticEPStateManager` 取 `active_ranks`（当前存活的 rank 掩码）传入。
  - `num_gpus` 由 `num_physical_experts // num_local_physical_experts` 推导。
- **`compute_algorithm(raw_algorithm, num_groups, num_nodes)`**：把配置里的字符串解析成枚举。非 `"auto"` 时直接查表；`"auto"` 时的启发式规则是：当 `num_groups` 能被 `num_nodes` 整除时选层次化（`deepseek_hierarchical`），否则选全局（`deepseek`）。

输入输出示例：
```python
algo = compute_algorithm("auto", num_groups=8, num_nodes=2)  # 8 % 2 == 0
# => EplbAlgorithm.deepseek_hierarchical

phy2log, log2phy, logcnt = rebalance_experts(
    tokens_per_expert=torch.tensor([[[100, 5, 5, 5]]]),  # [steps=1, layers=1, num_logical=4]
    num_physical_experts=8,        # 4 个逻辑专家 + 4 个冗余名额
    num_local_physical_experts=2,  # 每卡 2 个物理专家 => num_gpus = 8 // 2 = 4
    num_groups=1, num_nodes=1, algorithm=EplbAlgorithm.deepseek,
)
# 冗余名额优先分给热门的逻辑专家 0，使各卡负载趋于均衡
```

### `deepseek.py`（官方参考实现，逐层循环）

三个函数层层组合：

1. **`balanced_packing(weight, num_packs)`** —— 均衡装箱。
   把 `n` 个带权对象平均分到 `m` 个 pack，要求每个 pack 恰好装 `n/m` 个且各 pack 权重尽量均衡。做法是按权重降序，用贪心把每个对象放进"当前权重最小且未装满”的 pack。
   - 输入：`weight [X, n]`、`num_packs`；输出：`pack_index [X, n]`（各对象所属 pack）、`rank_in_pack [X, n]`（在 pack 内的次序）。
   - 特例：`groups_per_pack == 1` 时直接顺序分配。
   ```python
   # weight = [[10, 1, 8, 3]], num_packs = 2  => 每 pack 2 个对象
   # 降序处理 10,8,3,1：pack0={10}, pack1={8}, pack1+={3}=11, pack0+={1}=11 => 均衡
   ```

2. **`replicate_experts(weight, num_phy)`** —— 专家复制。
   把 `num_log` 个逻辑专家复制成 `num_phy` 个物理副本，使所有副本的最大负载最小。做法是重复 `num_redundant = num_phy - num_log` 次：每次给"当前平均负载（`weight / logcnt`）最高”的逻辑专家再加一个副本。
   - 输出：`phy2log [X, num_phy]`、`rank [X, num_phy]`（副本序号）、`logcnt [X, num_log]`（各逻辑专家副本数）。
   ```python
   # weight = [[100, 5, 5, 5]], num_phy = 6 => 2 个冗余名额
   # 第1个冗余给专家0(100/1最大) -> logcnt[0]=2
   # 第2个冗余给专家0(100/2=50 仍最大) -> logcnt[0]=3
   # 结果 logcnt = [[3, 1, 1, 1]]，专家0 拥有 3 个副本
   ```

3. **`rebalance_experts_hierarchical(...)`** —— 层次化三步走：
   - Step 1：把专家按组的 token 总量 `balanced_packing` 到各 **节点**；
   - Step 2：在每个节点内用 `replicate_experts` 构造冗余副本；
   - Step 3：把节点内的物理专家再 `balanced_packing` 到各 **GPU**。
   通过 `log2mlog` / `mlog2log`（逻辑↔节点内重排逻辑）等置换在各步之间换算索引。

4. **`rebalance_experts(...)`（入口）**：`enable_hierarchical=True` 走上面的三步；否则退化为 `num_groups=num_nodes=1` 的"全局单池”均衡。最后根据 `phy2log`/`phyrank` 反算出 `log2phy`（逻辑→物理副本列表），返回 `(phy2log, log2phy, logcnt)`。

### `deepseek_vec.py`（向量化实现）

与 `deepseek.py` 语义等价，但用张量运算替代 Python 逐层循环，适合层数/专家数较大的场景。

- **`pack_groups(tokens_per_group, num_nodes)`**：`balanced_packing` 的向量化对应物，把专家组打包到节点并返回每组的目标位置。
- **`make_redundant_experts_chunkwise(...)`**：核心。把物理专家空间切成 `num_chunks` 个 chunk，在每个 chunk 内并行地：先放置基础逻辑专家，再迭代 `num_redundancy_experts_per_group` 次，用 `score = tokens / logical_count` 挑出最需要副本的逻辑专家分配冗余名额（`+1e-4 * id` 的小扰动用于打破并列，保证 argmax 唯一）。当 `num_local_physical_experts > 1` 时，再做一次 GPU 间的负载均衡（含蛇形 `flip` 排布以均摊）。
- **`decode_rebalance_experts(...)`**：非层次化路径，令 `num_physical_experts_per_chunk = num_physical_experts`（单块），直接调用 chunkwise。
- **`prefill_rebalance_experts(...)`**：层次化路径，先 `pack_groups` 把组打包到节点、构造 `log2mlog`/`mlog2log` 置换，再以 `num_physical_experts // num_nodes` 为 chunk 大小调用 chunkwise，最后把节点内索引换算回逻辑空间。
- **`rebalance_experts(...)`（入口）**：`enable_hierarchical` 为真走 prefill 路径，否则走 decode 路径。

> 命名约定：`log` = 逻辑，`mlog` = 节点内重排后的逻辑（node-local logical），`phy` = 物理，`pphy` = GPU 打包后的物理。`a2b` 表示"a 索引 → b 索引”的映射张量。

### `elasticity_aware.py`（弹性感知）

面向"部分 rank 可能掉线/恢复”的弹性 EP 场景，复用 `deepseek.rebalance_experts_hierarchical` 作为底层算法。

- 入参多一个 `active_ranks`（各 rank 是否存活的掩码张量）。
- 分派逻辑：
  - `num_active_ranks < num_gpus`（有 rank 失效）→ **强制回退全局策略**，只对 `num_active_ranks` 个活跃 rank、`num_local_experts * num_active_ranks` 个物理专家做均衡；
  - 否则按 `enable_hierarchical` 选层次化或全局策略。
- 失效补位：回退时先在活跃 rank 上算好映射，再遍历 `active_ranks`，对每个失效的 rank 在对应位置**插入一段占位副本**（`phy2log` 补 0 段、`log2phy` 索引整体后移 `num_local_experts`），保证最终返回的张量形状与"全部 rank 在线”时一致，下游无需感知失效。

## 调用关系与数据流

```
EPLBManager.rebalance()                     # eplb_manager.py（调度层）
  └─ ExpertLocationMetadata.init_by_eplb()  # expert_location.py
       └─ eplb_algorithms.rebalance_experts(..., algorithm)   # 本目录 __init__.py（分派）
            ├─ deepseek.rebalance_experts()            # 逐层循环实现
            ├─ deepseek_vec.rebalance_experts()        # 向量化实现
            └─ elasticity_aware.rebalance_experts()    # 复用 deepseek 层次化 + 弹性补位
       # 三者统一返回 (physical_to_logical_map, logical_to_physical_map, logical_count)
```

即：管理层决定"何时重平衡”，本目录负责"给定负载如何放置专家”，产出的三个映射交由 `ExpertLocationMetadata` 封装、再由 `ModelRunner.update_expert_location` 实际搬运权重。
