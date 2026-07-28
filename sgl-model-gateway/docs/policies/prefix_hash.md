# Prefix Hash（前缀哈希）算法详解

> 本文档是 [ROUTING_POLICIES_ZH.md](./ROUTING_POLICIES_ZH.md) 中「Prefix Hash」策略的深入拆解，
> 包含 `load_factor` 计算方式与选路分支等细节。主文档仅保留概览。

**实现**：`src/policies/prefix_hash.rs` — 取前 $N$ 个 token 做 xxhash，落到一致性哈希环，再做有界负载检查。

## 第一性原理降维

这是第一个显式优化 $H_i$（缓存复用）的策略，但用的是**近似手段**：

$$
\text{worker} = \text{Ring.lookup}\big(\text{xxh3}(\text{tokens}[0{:}N])\big), \quad N = \texttt{prefix\_token\_count}\ (\text{默认 }256)
$$

核心假设：**相同前缀 → 相同哈希 → 相同 Worker → KV Cache 命中**。但纯亲和会造成热点，因此加了**有界负载均衡**。


### `load_factor` 的详细计算方式

对应实现见 `PrefixHashPolicy::load_ok`（`src/policies/prefix_hash.rs`）。每次选路时，先在**健康 Worker**集合上计算全局指标，再对候选 Worker 做「负载是否可接受」判定：

> **注意**：这里的 `worker.load()` 是「活跃请求数」（在途 / 并发请求数），而非 token 数——它由请求进入时 +1、完成时 −1 的原子计数器维护。详见 [worker_load_note.md](./worker_load_note.md)。

```text
// 1) 仅统计健康 Worker
total_load  = Σ worker.load()   // 所有健康 Worker 的当前活跃请求数之和
num_workers = 健康 Worker 数量

// 2) 计算“含本次请求”的人均负载（+1 用于模拟即将进入的这一个请求）
avg_load    = (total_load + 1) / num_workers

// 3) 由 load_factor 放大得到可接受负载阈值
threshold   = avg_load * load_factor        // load_factor 默认 1.25

// 4) 判定：候选 Worker 负载 ≤ 阈值 即视为“负载 OK”
load_ok     = worker.load() <= threshold
```

用公式表示，即候选 Worker $w$ 被判定为可接受当且仅当：

$$
\text{load}(w) \;\le\; \underbrace{\frac{\left(\sum_{i} \text{load}(w_i)\right) + 1}{\text{num\_workers}}}_{\text{avg\_load}} \times \texttt{load\_factor}
$$

**关键细节：**

- **`+1` 的含义**：把「即将到来的这次请求」计入人均，避免在低负载时阈值被算得过低而误判过载。
- **`load_factor` 的语义**：允许单个 Worker 的负载最多达到人均的 `load_factor` 倍。默认 `1.25` 表示「最多超出平均 25%」。
  - 取值越大 → 越偏向**缓存亲和**（更容忍热点，命中率高，但负载更不均）。
  - 取值越接近 `1.0` → 越偏向**均衡**（更早触发迁移，命中率下降）。
- **边界短路**：当 `total_load == 0` 或 `num_workers == 0` 时，`load_ok` 直接返回 `true`（无负载可比，直接走缓存亲和）。

### 判定后的选路分支（结合阈值）

1. **RingHit**：哈希环命中的初始 Worker 满足 `load_ok` → 直接选它（最佳情况，缓存亲和成立）。
2. **LoadBalanceWalk**：初始 Worker 过载（不满足 `load_ok`）→ 在**同样满足 `load_ok`** 的健康 Worker 中挑负载最小者；若**所有** Worker 都过载，则退回使用初始 Worker（放弃迁移，避免无意义抖动）。
3. **FallbackLeastLoad**：无哈希环或环查找失败 → 忽略亲和，直接选负载最小的健康 Worker。

即在「缓存亲和」与「负载上限」之间由 `load_factor` 设定了一个可调的安全阀。

## 高阶结构化类比

等价于 **NUMA 调度的缓存亲和 + 负载封顶**：优先让线程回到其缓存所在的 NUMA 节点，但当该节点过载时允许迁移，避免亲和性演变成拥塞。

## 边界与反模式


| 维度 | prefix_hash                    | cache_aware（基数树）     |
| ---- | ------------------------------ | ------------------------- |
| 查找 | $O(\log n)$                    | $O(\text{prefix\_len})$   |
| 内存 | $O(\text{workers} \times v_n)$ | $O(\text{total\_tokens})$ |
| 精度 | 前缀分组（粗）                 | 精确最长前缀匹配          |

- **边界**：与 cache_aware 的分界线是**精度 vs 可预测性**——prefix_hash 用固定长度前缀分组换取稳定的 $O(\log n)$，放弃了精确匹配。
- **反模式**：`prefix_token_count` 设得过短会把不同请求错误聚合（假共享），过长则失去分组意义。

## 演进动机

它是 cache_aware「太重」时的轻量替代：用哈希近似代替显式前缀树，把更新成本从 $O(\text{prefix\_len})$ 降到 $O(1)$。

## 参数配置

策略参数定义见 `PolicyConfig::PrefixHash`（`src/config/types.rs`），两个参数都带 serde 默认值。

| 参数 | 类型 | 默认值 | 含义 | 调参影响 |
|---|---|---|---|---|
| `prefix_token_count` | `usize` | `256` | 参与哈希的前缀 token 数：只对请求前 $N$ 个 token 做 xxhash 判定缓存归属 | 调短 → 易把不同请求错误聚合（假共享）；调长 → 分组过细、失去亲和意义 |
| `load_factor` | `f64` | `1.25` | 有界负载因子：候选 Worker 负载 > 人均 × 此值时视为过载，沿环换下一个 | 越大越偏**缓存亲和**（容忍热点、命中率高）；越接近 `1.0` 越偏**均衡**（更早迁移、命中率降） |

> 有界负载判据：候选 $w$ 可接受 $\iff \text{load}(w) \le \frac{(\sum_i \text{load}(w_i)) + 1}{\text{num\_workers}} \times \texttt{load\_factor}$；其中 `load()` 为活跃请求数（见 [worker_load_note.md](./worker_load_note.md)）。

## 下游 Worker 节点增删对本策略的影响

Prefix Hash 的亲和来自**哈希环**，而环由 `WorkerRegistry` 在 worker 变更时预构建/重建，再由选路阶段传入策略（策略自身不持久保存环，见 `src/policies/prefix_hash.rs`）。因此拓扑变更的影响主要体现在环的重建上：

- **新增 Worker**：Worker 上环后，**约 $1/N$ 的前缀**的顺时针后继会改指向新 Worker，这部分请求的亲和目标发生迁移，需在新 Worker 上重新 prefill；其余前缀归属不变。
- **移除 Worker**：被移除节点所负责的那段环弧上的前缀，会顺时针**改由后继 Worker 承接**，同样约 $1/N$ 规模发生迁移；未受影响的前缀保持亲和。
- **有界负载判据**：`load_ok` 用的是**当次传入的健康 Worker 集合**（活跃请求数，见 [worker_load_note.md](./worker_load_note.md)），Worker 数变化会立即改变 `avg_load` 与阈值，无需额外状态迁移。
- **少 Worker 场景**：Worker 很少时增删会明显改变环的均匀性，虚拟节点可缓解但迁移比例的波动更大。

**一句话**：增删仅引起环上**约 $1/N$ 的前缀重映射**（远优于取模），受影响前缀需重新 prefill，其余亲和稳定；策略无需持久状态迁移，环重建后即恢复一致。

## 新增 Router（网关实例）对本策略的影响

上一节是**下游 Worker**变更；这一节是**网关自身**横向扩容——新增一个 Router 实例。两者影响机制完全不同：Worker 变更改变环的拓扑，Router 变更则涉及多实例间的**状态是否共享**。

**① 哈希环：天然一致，无需同步环本身**

`HashRing::new` 用 **blake3** 构建，结果在不同 Rust 版本/进程间稳定可复现（`src/core/worker_registry.rs`）；而 `WorkerRegistry` 已被 mesh 同步（`src/server.rs` 对 worker_registry 调用 `set_mesh_sync`），worker 列表在各 Router 间收敛一致。二者叠加的结果：

- 每个 Router **本地**从相同的 worker 列表重建出**完全相同的环** → 同一前缀无论落到哪个 Router，都路由到**同一个 Worker** → **缓存亲和跨 Router 保持**，且无需同步环本身（即「分布式友好」优点的实现基础）。
- 新增 Router 上线后，只要它同步到相同 worker 列表，其路由决策与既有 Router **零差异**，不会打散已建立的前缀亲和。

**② 负载视图：不跨 Router 共享（关键差异）**

与 cache_aware 不同，`PrefixHashPolicy` **未覆盖 `set_mesh_sync`**（只有 `CacheAwarePolicy` 覆盖了，见 `src/policies/cache_aware.rs`），因此走 trait 默认空实现（`src/policies/mod.rs`）——**它的负载状态不参与 mesh 同步**。而 `load_ok` 依赖的 `worker.load()` 是**每个 Router 本地维护的在途请求计数**（见 [worker_load_note.md](./worker_load_note.md)），只统计「本 Router 派发、尚未返回」的请求。由此：

- 有 $R$ 个 Router 时，每个 Router 的 `total_load` / `avg_load` 只覆盖**自己那 $\approx 1/R$ 的流量**，看不到全局真实负载。
- **后果**：某 Worker 因**其他 Router**的流量而变热时，在本 Router 视角里仍显示「空闲」，`load_ok` 判定偏乐观 → 各 Router 无法协同迁移，可能一起把请求压向同一个实际已热的 Worker。Router 越多，本地视图与全局负载的偏差越大。
- **缓解**：这正是 [优化迭代方向](#优化迭代方向) 中「token 级负载判据」的额外价值——引擎 `total_tokens` 是**引擎侧的全局真值**，各 Router 拉取到的是同一数字，天然跨 Router 一致，可同时修复「本地负载视图偏差」与「长短请求等权」两个问题。

**一句话**：新增 Router 对**缓存亲和无影响**（环本地可复现、worker 列表已 mesh 同步，各实例路由一致）；但**负载判据是各 Router 本地视图、不跨实例共享**，Router 越多 `load_ok` 越偏乐观，改用引擎 `total_tokens` 可一并解决。

## 优缺点、使用场景、局限性与优化迭代方向

### 优点

- **轻量高效**：无需维护前缀树，查找 $O(\log n)$、更新 $O(1)$，内存仅 $O(\text{workers} \times v_n)$，远低于 cache_aware。
- **可预测**：相同前缀恒定映射到同一 Worker，延迟与归属稳定，便于容量规划。
- **兼顾负载**：`load_factor` 有界负载检查提供可调安全阀，避免亲和演变成热点。
- **分布式友好**：哈希环可由 worker 列表本地重建，多节点天然一致，无需同步环本身。

### 缺点

- **前缀精度粗**：按**固定长度**前缀分组，无法区分「前 N token 相同、之后分叉」的请求（假共享），命中率低于 cache_aware 的精确最长匹配。
- **参数敏感**：`prefix_token_count` 过短易假共享、过长则失去分组意义。
- **不感知 token 级成本**：`load_ok` 用的是活跃请求数而非 token 负载（见 [worker_load_note.md](./worker_load_note.md)），负载判定精度有限。

### 使用场景

- 有前缀共享、但更看重**稳定延迟与低开销**、可接受近似命中率的服务。
- Worker 规模较大、cache_aware 的树内存/更新成本不划算的场景。
- 需要缓存亲和又要求 $O(\log n)$ 可预测查找的通用流量。

### 局限性

- 前缀分叉点晚于 `prefix_token_count` 时无法感知差异，长公共前缀 + 短差异后缀的负载表现不佳。
- 有界负载判据基于活跃请求数，长短请求等权，无法反映真实计算成本差异。
- 纯哈希亲和不感知引擎侧 KV 实际驻留，命中收益依赖引擎缓存未淘汰。

### 优化迭代方向

- **前缀长度自适应**：按请求分布动态调整 `prefix_token_count`，或多粒度前缀哈希兼顾精度与开销。
- **token 级负载判据**：将 `load_ok` 的活跃请求数替换为引擎 `total_tokens`，提升有界负载判定精度（详见下节）。
- **虚拟节点均匀化**：增大环上虚拟节点密度，改善少 Worker 时的分布均匀性。
- **与 cache_aware 混合**：热前缀走精确树、长尾走哈希近似，在命中率与成本间取折中。

#### 深入：用 `total_tokens` 替换 `load_ok` 的活跃请求数

当前 `load_ok`（`src/policies/prefix_hash.rs`）用的是 `worker.load()`——网关本地维护的**在途请求条数**，把长短请求等权（见 [worker_load_note.md](./worker_load_note.md)）。而 Power of Two 主路径已经在用引擎 `/v1/loads` 的 `total_tokens`（running + waiting 的 token 总债，见 [power_of_two.md](./power_of_two.md)），保真度显著更高。将后者引入 `load_ok`，可让「有界负载」判据从「请求条数」升级为「真实计算成本」，尤其改善长短请求混合场景下的判定精度。

架构上完全可行，但**主要工作量不在 prefix_hash 本身**，而在打通负载采集链路并处理新引入的脆弱点：

**① 两者负载来源的本质差异**

| 维度 | prefix_hash 现状 | Power of Two 主路径 |
| ---- | ---- | ---- |
| 数据源 | `worker.load()`（本地原子计数，在途请求数） | `cached_loads` ← 引擎 `total_tokens` |
| 获取方式 | **同步、恒可用**，选路时直接读 | **后台异步拉取**，`LoadMonitor` 每 $\tau_\text{load}$（默认 5s）刷新快照 |
| 注入通道 | 无（直接访问 `Worker`） | `LoadBalancingPolicy::update_loads()` |

**② 需要改动的三处**

1. **让 prefix_hash 能接收 token 负载**：`update_loads()` 在 trait 中已有默认空实现（`src/policies/mod.rs`）。仿照 `PowerOfTwoPolicy`，为 `PrefixHashPolicy` 增加 `cached_loads: RwLock<HashMap<String, isize>>` 并实现 `update_loads`。
2. **打通 `LoadMonitor` 的拉取开关（关键卡点）**：`monitor_loop`（`src/core/worker_manager.rs`）**仅在存在 Power of Two 策略时才拉取负载**，否则整轮跳过（零开销优化）；`get_all_power_of_two_policies`（`src/policies/registry.rs`）也是按策略名 `"power_of_two"` 硬编码筛选。要让 prefix_hash 生效，需把「谁需要负载」的判定从「是否 PowerOfTwo」泛化为「是否负载感知策略」（如在 trait 加 `needs_load_monitoring()`，或改为收集所有覆盖了 `update_loads` 的策略）。
3. **`load_ok` 内部改用 token 负债**：`total_load` / `worker_load` 从 `Σ worker.load()` 换成从缓存读 `total_tokens`。阈值语义（人均 × `load_factor`）仍然成立，量纲自洽。

**③ 三个必须处理的权衡**

- **降级一致性（比 P2C 更脆弱）**：token 负载来自异步拉取，某 Worker 缺失时得到 `-1`。`load_ok` 依赖**全局求和** `total_load`，一个 `-1` 混入会污染整个平均值——比 P2C 的两两比较更敏感。必须设计缺失时的回退（整体回退到 `worker.load()`，或将缺失项按人均补齐），**绝不能直接求和**。
- **陈旧窗口**：`worker.load()` 实时；换成 token 负载后最多陈旧 $\tau_\text{load}$ 秒。`load_ok` 是「是否迁移出亲和 Worker」的判据，陈旧信号可能造成迁移抖动或迟滞。
- **响应格式对齐**：网关解析 `aggregate.total_tokens`，而 SGLang 原生 `/v1/loads` 返回 `loads[].num_total_tokens`（见 [power_of_two.md](./power_of_two.md) 的字段结构注意）。此隐患会被 prefix_hash 原样继承，接入时需确认格式已对齐，否则解析得 `-1` 触发上述降级。
