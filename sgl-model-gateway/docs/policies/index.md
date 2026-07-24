# 负载均衡路由策略深度拆解

> 本文档以第一性原理拆解 `sgl-model-gateway` 中 `src/policies/` 下的全部路由策略，
> 说明每种算法「解决什么底层矛盾、以什么结构运转、边界在哪里、为何演化出来」。
> 所有策略均实现统一的 [`LoadBalancingPolicy`](./mod.rs) trait，
> `select_worker` 返回被选中 Worker 在候选数组中的下标。

---

## 0. 所有策略的公共底座

在讨论任何具体算法之前，必须先厘清一个所有策略共享的事实：**策略只在「健康且熔断允许执行」的 Worker 子集上做选择**。

```text
// src/policies/mod.rs
w.is_healthy() && w.circuit_breaker().can_execute()
```

即候选集并非全部 Worker，而是：

$$
\mathcal{H} = \{\, i \mid \text{healthy}(i) \wedge \text{can\_execute}(i) \,\}
$$

这条不变量把「可靠性」与「负载均衡」解耦：熔断/健康检查负责剔除坏节点，策略只负责在好节点里做**成本最优的位置选择**。这也是理解后续所有算法的前提——它们的差异只在于「如何在 $\mathcal{H}$ 内挑一个」。

### 0.1 路由问题的第一性矛盾

大模型推理路由的底层矛盾是：

$$
\text{决策必须在 } t_0 \text{ 完成，但真实成本 } C = C_{\text{prefill}}(L_\text{in}) + \sum_{t=1}^{L_\text{out}} C_{\text{decode}}(L_\text{in}+t) \text{ 依赖未知的 } L_\text{out}
$$

且可复用状态（KV/Prefix Cache）**绑定在特定 Worker 上**。因此路由不是「均分流量」，而是在信息不完备下求解：

$$
\arg\min_i \big( \underbrace{Q_i}_{\text{排队}} + \underbrace{C_i}_{\text{预估计算}} - \underbrace{H_i}_{\text{缓存复用收益}} + \underbrace{R_i}_{\text{故障/热点风险}} \big)
$$

**每种策略本质上都是对上式中某几项的近似估计与取舍**。下表先给出全局对照，后续章节逐一深挖。

| 策略 | 主要估计项 | 决策输入 | 内部状态 | 时间复杂度 | 核心取舍 |
|---|---|---|---|---|---|
| Random | 无（假设同质） | 候选集 | 无 | $O(1)$ | 零成本 vs 零信息 |
| Round Robin | 无（均分请求数） | 候选集 | 原子游标 | $O(1)$ | 请求数均衡 vs 忽略成本差异 |
| Power of Two | $Q_i$ | 负载采样 | 缓存负载表 | $O(1)$ | 极低观测成本改善尾延迟 |
| Prefix Hash | $H_i$（近似） | token 前缀 + 哈希环 | 无（依赖外部环） | $O(\log n)$ | 稳定前缀亲和 vs 精度 |
| Consistent Hashing | 亲和稳定性 | routing key / header | 哈希环 | $O(\log n)$ | 拓扑变化下最小迁移 |
| Cache Aware | $H_i$（精确）+ $Q_i$ | 请求文本 | 近似基数树 | $O(\text{prefix\_len})$ | 精确复用 vs 内存/维护成本 |
| Manual | 强会话亲和 | routing key | key→Worker 映射 | $O(1)$ | 绝对粘性 vs 负载偏斜 |
| Bucket | PD 分桶均衡 | PD 请求特征 | 分桶状态 | ~$O(1)$ | PD 场景专用 |

---

## 1. Random（随机）

**实现**：`src/policies/random.rs` — 在 $\mathcal{H}$ 中均匀随机取一个下标。

### 第一性原理降维
Random 显式假设 $C_i \approx C,\ H_i = 0$，即**所有 Worker 同质、无可复用状态**。在此假设下，任何单次决策都无法比随机更优（因为没有任何可利用的信息），随机反而以 $O(1)$ 无状态开销获得长期期望均衡：

$$
\mathbb{E}[\text{load}_i] = \frac{\lambda}{|\mathcal{H}|}
$$

### 高阶结构化类比
等价于操作系统中最原始的**无优先级随机调度**，或哈希表的**均匀散列假设**——它不追求单次最优，只保证在大数定律下不产生系统性偏斜。

### 边界与反模式
- **边界**：与 Round Robin 的区别在于 Random 是无记忆的（马尔可夫性），不保证任意窗口内的均匀，只保证期望均匀。
- **反模式**：当请求成本方差极大（大模型典型场景）时，随机会以一定概率把多个长请求砸到同一 Worker，制造尾延迟。它是**基线**，不是优化解。

### 演进动机
它是「假设后端同质」时代的遗留最优解。一旦承认 $C_i$ 方差大、$H_i \neq 0$，Random 立即退位为兜底基线。

---

## 2. Round Robin（轮询）

**实现**：`src/policies/round_robin.rs` — 原子计数器 `fetch_add(1)` 后对 $|\mathcal{H}|$ 取模。

### 第一性原理降维
Round Robin 用一个**全局原子游标**把「随机的期望均匀」升级为「确定性的请求数均衡」：

$$
\text{selected} = (\text{counter}_{k}) \bmod |\mathcal{H}|
$$

它消除了 Random 的方差，但代价是引入了**共享可变状态**（原子变量），并隐含一个更强的错误假设：**每个请求成本相等**。

### 高阶结构化类比
等价于时间片轮转调度（Time-Slice Round Robin）在「假设每个进程时间片等长」下的退化形式；也类似 DNS 轮询。

### 边界与反模式
- **边界**：与 Random 的分界线是「有无记忆」。Round Robin 有 $O(1)$ 状态，能保证任意 $N$ 个连续请求恰好覆盖各 Worker 一次。
- **极限**：健康集合 $\mathcal{H}$ 动态变化时，取模基准会漂移，严格轮转性被破坏（这是不可避免的，因为候选集在变）。
- **反模式**：把「请求数均衡」误当「负载均衡」。10 个短请求 + 1 个超长请求下，轮询会让承接长请求的 Worker 严重过载。

### 演进动机
它是对 Random 方差的第一次修正。`reset()` 存在的意义正是承认其**有状态性**需要可清零。

---

## 3. Power of Two Choices（二选一）

**实现**：`src/policies/power_of_two.rs` — 随机抽 2 个 Worker，比较负载取低者。

### 第一性原理降维
这是本仓库唯一真正估计 $Q_i$（实时负载）的**轻量**策略。它的精髓不在「选负载低的」，而在「**只看两个**」：

$$
\text{selected} = \arg\min_{i \in \{a,b\}} \text{load}(i), \quad a,b \sim \text{Uniform}(\mathcal{H}),\ a \neq b
$$

理论结论（Mitzenmacher）：全局最忙 Worker 的期望负载从随机的 $O(\log n / \log\log n)$ 降到 $O(\log\log n)$——**只用两次采样就获得接近全局最优的尾部收益**，却避免了扫描全局带来的观测成本与「羊群效应」。

#### 举例：为什么「看两个」就够了

设有 $n = 100$ 个 Worker，用「全局最忙 Worker 的队列长度」衡量是否均衡（越小越均衡）。

- **随机选 1（Random）**：每个请求闭眼丢给一台。总有机器「连续中奖」堆积，最忙者队列长度约 $\dfrac{\log n}{\log\log n} = \dfrac{\log 100}{\log\log 100} \approx \dfrac{4.6}{1.5} \approx 3$（量级示意）。
- **随机选 2 取较空（Power of Two）**：每个请求随机挑 2 台、比较后丢给更空的那台。最忙者队列长度骤降到约 $\log\log n = \log\log 100 \approx 1.5$（量级示意）。

用超市收银台类比：100 个收银台，「随机选 1」= 蒙眼冲向某个台子，必然有的排长队、有的空着；「随机选 2」= 随便瞟 **2 个**台子、走人少的那个——不必比较全部 100 个，只看 2 个就足以避开长队。

**为什么是 2 而不是 3、4？** 关键在概率抑制：一台机器要变「最忙」，在 Power of Two 下必须**连续**在「被抽中的两台里更空」，其概率约为 $(1/n)^2$ 量级，相比随机选 1 的 $1/n$ 急剧变小——这正是负载从 $\log n$ 降到 $\log\log n$ 的根源。而选 3 个变成 $(1/n)^3$，**边际收益已微乎其微**，却要多付一次负载比较成本。因此「2」是观测成本与均衡质量的甜点：

| 方案 | 决策成本 | 最忙 Worker 负载（量级） | 效果 |
|---|---|---|---|
| 随机选 1 | 最低 | $\sim \log n / \log\log n$ | 易出热点 |
| **随机选 2 取优** | 低（多看 1 台） | $\sim \log\log n$ | 显著均衡 |
| 查全部 $n$ 台取最优 | 高（$O(n)$ + 羊群效应） | $\sim 1$ | 均衡但成本高且易振荡 |

#### 数学原理：最忙 Worker 负载为何是 $\log\log n$

设把 $n$ 个球（请求）投入 $n$ 个桶（Worker），关注**最大桶高**（即最忙 Worker 的负载）。用 $\beta_k$ 表示「高度 $\ge k$ 的桶所占比例」，通过归纳递推估计其量级。

**① 随机选 1（$d=1$）—— 独立泊松近似**

每个桶的高度近似服从 $\text{Poisson}(1)$。桶高 $\ge k$ 的概率约 $\dfrac{1}{k!}$，故：

$$
\Pr[\text{某桶高度} \ge k] \approx \frac{1}{k!}, \qquad \mathbb{E}[\text{高度}\ge k \text{ 的桶数}] \approx \frac{n}{k!}
$$

令该期望降到 $O(1)$（即最大高度阈值），需 $k! \approx n$。由 Stirling 公式 $k! \approx (k/e)^k$ 反解：

$$
k^* \approx \frac{\ln n}{\ln \ln n} = \Theta\!\left(\frac{\log n}{\log\log n}\right)
$$

这就是随机选 1 的最大负载量级。

**② 随机选 2 取较空（$d=2$）—— 递推的「平方坍缩」**

核心变化：一个球**只有**当它随机选中的 **两个**桶**都**已达到高度 $\ge k$ 时，才可能把某桶推到 $k+1$。两次独立采样，故：

$$
\beta_{k+1} \;\lesssim\; \big(\beta_k\big)^{2}
$$

这是一个**平方递推**。从某个常数基准 $\beta_{k_0} \le \tfrac{1}{2}$ 出发迭代：

$$
\beta_{k_0+j} \;\lesssim\; \left(\tfrac{1}{2}\right)^{2^{j}}
$$

指数上出现 $2^{j}$——**双重指数衰减**。要让 $\beta_k$ 小到对应「不足一个桶」（即 $\beta_k \cdot n < 1$，$\beta_k < 1/n$），只需：

$$
2^{\,j} \gtrsim \log_2 n \;\Longrightarrow\; j \gtrsim \log_2 \log_2 n
$$

于是最大高度：

$$
k^{*} \;=\; k_0 + j \;=\; \frac{\ln\ln n}{\ln 2} + \Theta(1) \;=\; \Theta(\log\log n)
$$

**③ 一般 $d$ 选一（$d\ge 2$）**

同理递推变为 $\beta_{k+1} \lesssim (\beta_k)^{d}$，指数塔底数变成 $d$：

$$
k^{*} \;=\; \frac{\ln\ln n}{\ln d} + \Theta(1)
$$

对照三种情形，可一眼看出「$d=1 \to d=2$ 是质变，$d=2 \to d\ge3$ 只是常数因子改良」：

| 采样数 $d$ | 递推关系 | 最大负载量级 | 相对 $d=2$ |
|---|---|---|---|
| $1$ | $\beta_{k+1}\approx \beta_k/k$（无平方） | $\Theta\!\big(\tfrac{\log n}{\log\log n}\big)$ | — |
| $2$ | $\beta_{k+1}\lesssim \beta_k^{2}$ | $\Theta(\log\log n)$ | 基准 |
| $d\ge 3$ | $\beta_{k+1}\lesssim \beta_k^{d}$ | $\dfrac{\log\log n}{\log d}+\Theta(1)$ | 仅差常数因子 $\tfrac{1}{\log d}$ |

**结论**：$d=1$ 无平方项，衰减是「阶乘级」，反解得 $\log n/\log\log n$；$d\ge2$ 引入 $\beta_k^{d}$ 的**多重指数（幂塔）衰减**，反解得 $\log\log n$。从 1 到 2 把「单指数」变成「双重指数」，是量级跃迁；从 2 再往上只把幂塔底数由 2 变成 $d$，仅改变常数系数 $1/\log d$——这从数学上精确解释了前文「2 是甜点」的直觉。

> 说明：以上为 Azar–Broder–Karlin–Upfal / Mitzenmacher 结果的直觉化推导，省略了误差项与集中不等式的严格证明，量级（$\Theta$）结论成立。

代码中的关键工程细节（**指标兼容性降级**）：

```text
// 若任一 Worker 缺 token 级负载，两者都降级为请求计数比较
match (load1_tokens, load2_tokens) {
    (Some(t1), Some(t2)) => 使用 (t1, t2)                // 同为 token 负载
    否则                 => 使用 (worker1.load(), worker2.load())  // 同为请求计数
}
```

这修复了一个隐蔽 bug：绝不能拿「token 负载(如 50000)」与「请求数(如 5)」比较，否则量纲不可比会导致灾难性误判。

### `load_tokens` 的含义与更新周期

Power of Two 的选择质量取决于「负载信号有多准、多新」。本仓库对负载采用**两级信号 + 后台异步刷新**：

**① `load_tokens` 是什么（高保真信号）**

- 语义：某个 Worker 上**当前尚未处理完的总 token 数**，而非「请求条数」。它直接对应 GPU 上排队/在算的实际计算量，因此比请求计数更能反映真实压力（1 个 8k-token 的长请求远重于 5 个 32-token 的短请求）。
- 来源：`LoadMonitor` 向每个 HTTP Worker 的负载接口拉取，解析响应 JSON 的 `aggregate.total_tokens` 字段：

```text
// src/core/worker_manager.rs · parse_load_response
json["aggregate"]["total_tokens"]  // 解析为 isize
// 请求失败 / 非 2xx / JSON 解析失败 / 字段缺失 → 返回 -1（视为无效）
```

- 存储：策略内部用 `cached_loads: RwLock<HashMap<String, isize>>` 缓存，key 为 Worker URL，value 即该 token 负载快照。
- **降级信号**：若某 Worker 在缓存中缺失（监控失败、字段缺失等），`select_worker` 会把参与比较的**两个** Worker **都**回退到本地请求计数 `worker.load()`，保证量纲一致（即前述兼容性降级）。

**① bis 引擎侧 `total_tokens` 究竟怎么算（数值语义在 Worker 端定义）**

该数值由**推理引擎（SGLang server）**自己维护，网关只是拉取。SGLang 调度器在 `/v1/loads` 的负载查询逻辑（`scheduler_components/load_inquirer.py`）中如下计算：

$$
\text{total\_tokens} \;=\; \underbrace{\text{num\_used\_tokens}}_{\text{运行中已占用}} \;+\; \sum_{\text{req}\,\in\,\text{等待队列}} \text{req.seqlen}
$$

```python
# load_inquirer.py（引擎侧）
num_used_tokens, _ = pool_stats_observer.get_pool_stats().get_kv_token_stats()
num_total_tokens = num_used_tokens + sum(
    req.seqlen for queue in waiting_queues for req in queue
)
```

两项的物理含义：

- **第一项 `num_used_tokens`（KV Cache 真实占用）**：由 `pool_stats_observer` 计算——
  $$
  \text{num\_used\_tokens} = \text{max\_total\_num\_tokens} - (\text{available\_size} + \text{evictable\_size})
  $$
  即「KV 池总容量 − 空闲可分配槽位 − 可淘汰的前缀缓存」，反映 GPU 上正在运行的请求实际吃掉的 KV 槽位（含受保护、不可淘汰的前缀缓存）。hybrid-SWA 模型取 `max(full, swa)`，SSM/Mamba、HiSparse 会叠加各自的分层统计。

- **第二项 `Σ req.seqlen`（排队负债）**：所有等待队列中每个请求的完整输入序列长度之和。等待队列组成随模式而变：普通模式为主等待队列；PD-Prefill 追加 `bootstrap_queue`；PD-Decode 追加 `prealloc/transfer/retracted` 等子队列。

**一句话**：`total_tokens` = 该 Worker「已在 GPU 上跑着的（running）+ 已收到但排队等 prefill 的（waiting）」token 总债，比「请求条数」精确得多，也正是 SGLang DP 负载均衡 `total_tokens` 方法所用的核心信号。

> ⚠️ 字段结构注意：SGLang 原生 `/v1/loads` 返回形如 `{"loads":[{"num_total_tokens":N, ...}]}`（无 `aggregate` 层、字段名为 `num_total_tokens`），而网关当前解析的是 `aggregate.total_tokens`。若直连该版本引擎且中间无格式转换层，网关会解析失败得到 `-1`，从而使 P2C 退化为按请求计数比较——接入时需确认二者的响应格式已对齐。

**② 更新周期（后台定时，非每请求）**

负载采集被**从请求热路径中剥离**，改由 `LoadMonitor` 后台任务周期性拉取，避免每个请求都去查询负载：

$$
\text{每隔 } \tau_\text{load} \text{ 秒：并发拉取所有 Worker 负载} \;\to\; \text{整体替换策略缓存} \;\to\; \text{watch 广播快照}
$$

关键参数与行为：

- **间隔** $\tau_\text{load}$ = `PolicyConfig::PowerOfTwo { load_check_interval_secs }`，CLI/默认值为 **5 秒**（见 `main.rs` 的 `parse_policy`）。
- **触发条件**：`monitor_loop` 每个 tick 先检查是否存在 Power of Two 策略；**没有则跳过拉取**（零开销），有才并发采集。
- **更新方式**：`update_loads` 用最新快照**整体替换**旧缓存（`*cached = loads.clone()`），而非增量合并；写锁获取失败则静默跳过，等下一周期补上。
- **空结果保护**：若本轮没拉到任何负载，则**不覆盖**旧缓存并告警，避免把「采集失败」误当「负载为 0」。

**③ 这带来一个固有的「陈旧窗口」**

由于是定时快照，策略看到的负载**最多陈旧 $\tau_\text{load}$ 秒**。当状态陈旧程度远超负载变化速度（$\Delta_\text{state} \gg \tau_\text{load}$ 的反向情形）时，二选一会退化为「近似随机」——这正是「边界与反模式」中所述的极限：$\tau_\text{load}$ 调大则观测开销低但信号旧，调小则信号新但采集压力大，需按流量抖动幅度权衡。

### 高阶结构化类比
等价于分布式哈希负载均衡中的 **"the power of two random choices"**，也类比 CPU 调度里的**局部窥探**（只看邻近核心而非全局队列）以规避全局锁竞争。

### 边界与反模式
- **边界**：与「全局最少负载」的分界线是**观测范围**。全局最少负载会因所有请求同时看到同一个「最空闲」节点而振荡（$A$ 空 → 全涌向 $A$ → $A$ 过载 → 全涌向 $B$）；二选一用随机采样天然打散这种同步。
- **极限**：负载数据严重过期（$\Delta_\text{state} \gg \tau_\text{load}$）时退化为近似随机。
- **反模式**：在负载信号缺失/延迟大的环境里强行相信采样值。

### 演进动机
推翻了「必须扫描全局才能优化尾延迟」的假设。它是**观测成本**与**均衡质量**之间的帕累托最优点。

---

## 4. Prefix Hash（前缀哈希）

**实现**：`src/policies/prefix_hash.rs` — 取前 $N$ 个 token 做 xxhash，落到一致性哈希环，再做有界负载检查。

### 第一性原理降维
这是第一个显式优化 $H_i$（缓存复用）的策略，但用的是**近似手段**：

$$
\text{worker} = \text{Ring.lookup}\big(\text{xxh3}(\text{tokens}[0{:}N])\big), \quad N = \texttt{prefix\_token\_count}\ (\text{默认 }256)
$$

核心假设：**相同前缀 → 相同哈希 → 相同 Worker → KV Cache 命中**。但纯亲和会造成热点，因此加了**有界负载均衡**。

#### `load_factor` 的详细计算方式

对应实现见 `PrefixHashPolicy::load_ok`（`src/policies/prefix_hash.rs`）。每次选路时，先在**健康 Worker**集合上计算全局指标，再对候选 Worker 做「负载是否可接受」判定：

```text
// 1) 仅统计健康 Worker
//    注意：这里的 worker.load() 是“活跃请求数”（在途/并发请求数），
//    而非 token 数——它由请求进入时 +1、完成时 -1 的原子计数器维护。
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

#### 判定后的选路分支（结合阈值）

1. **RingHit**：哈希环命中的初始 Worker 满足 `load_ok` → 直接选它（最佳情况，缓存亲和成立）。
2. **LoadBalanceWalk**：初始 Worker 过载（不满足 `load_ok`）→ 在**同样满足 `load_ok`** 的健康 Worker 中挑负载最小者；若**所有** Worker 都过载，则退回使用初始 Worker（放弃迁移，避免无意义抖动）。
3. **FallbackLeastLoad**：无哈希环或环查找失败 → 忽略亲和，直接选负载最小的健康 Worker。

即在「缓存亲和」与「负载上限」之间由 `load_factor` 设定了一个可调的安全阀。

### 高阶结构化类比
等价于 **NUMA 调度的缓存亲和 + 负载封顶**：优先让线程回到其缓存所在的 NUMA 节点，但当该节点过载时允许迁移，避免亲和性演变成拥塞。

### 边界与反模式
| 维度 | prefix_hash | cache_aware（基数树） |
|---|---|---|
| 查找 | $O(\log n)$ | $O(\text{prefix\_len})$ |
| 内存 | $O(\text{workers} \times v_n)$ | $O(\text{total\_tokens})$ |
| 精度 | 前缀分组（粗） | 精确最长前缀匹配 |

- **边界**：与 cache_aware 的分界线是**精度 vs 可预测性**——prefix_hash 用固定长度前缀分组换取稳定的 $O(\log n)$，放弃了精确匹配。
- **反模式**：`prefix_token_count` 设得过短会把不同请求错误聚合（假共享），过长则失去分组意义。

### 演进动机
它是 cache_aware「太重」时的轻量替代：用哈希近似代替显式前缀树，把更新成本从 $O(\text{prefix\_len})$ 降到 $O(1)$。

---

## 5. Consistent Hashing（一致性哈希）

**实现**：`src/policies/consistent_hashing.rs` — 预计算哈希环，支持 header 直连与 routing key 亲和。

### 第一性原理降维
一致性哈希优化的**不是实时负载**，而是**拓扑变化下的映射稳定性**。普通取模 $\text{hash}(k) \bmod n$ 在 $n$ 变化时几乎全部重映射；一致性哈希把 Worker 和 key 都映射到同一个环上，key 顺时针找第一个健康 Worker：

$$
\text{扩缩容时迁移的 key 比例} \approx \frac{1}{N} \quad (\text{而非取模的近 } 100\%)
$$

代码中环由 `WorkerRegistry` 预构建缓存（非每请求重建），保证 $O(\log n)$ 二分查找 + $O(k)$ 跳过连续不健康节点。其决策优先级为：

1. `X-SMG-Target-Worker`：按下标直连（$O(1)$）
2. `X-SMG-Routing-Key`：显式 key 一致性哈希
3. 隐式 key（authorization/x-forwarded-for/cookie 等稳定头）做会话亲和
4. 随机兜底

### 高阶结构化类比
经典的 **Chord DHT / Amazon Dynamo** 环形拓扑——用「环 + 顺时针查找」把「成员变更的爆炸半径」限制在 $1/N$。

#### 什么是「环形拓扑」

把哈希值空间（如 $[0, 2^{32})$ 或 $[0, 2^{160})$）想象成一个**首尾相接的圆环**：最大值的下一个位置又回到 $0$。然后：

1. **节点上环**：对每个 Worker（节点）用哈希函数 $\text{hash}(\text{节点标识})$ 算出一个位置，把它「钉」在环上。
2. **数据/请求上环**：对每个 key（这里是前缀哈希或 routing key）同样算 $\text{hash}(k)$，落到环上某点。
3. **归属规则**：从 key 的位置**顺时针**走，遇到的**第一个节点**就是它的归属 Worker（若不健康则继续顺时针跳到下一个）。

这样「查找」退化为「在有序的节点位置数组里做二分查找」——即代码中的 $O(\log n)$。

```text
        0 / 2^32
          ┌───●B───┐
       ●A │        │ key k  → 顺时针遇到的第一个节点是 C，归属 C
          │        ●C
          └───●D───┘
   环上顺序: A → B → C → D → （回到 A）
```

#### Chord DHT（2001，MIT）

- **背景**：P2P 分布式哈希表，目标是在没有中心目录的情况下，让任意节点都能高效定位「某个 key 存在哪个节点」。
- **核心贡献**：
  - 将节点与 key 映射到同一个 $2^{m}$ 的环（**identifier ring**）；key 归属其顺时针后继节点（successor）。
  - 用 **finger table（指针表）** 让每个节点缓存若干「间隔指数级增大」的后继，从而把查找从 $O(N)$ 降到 $O(\log N)$ 跳。
  - 节点加入/离开时，只影响其**相邻区间**的 key，迁移量约 $1/N$。
- **对本项目的映射**：我们不需要 finger table（Worker 数量少、环在网关本地预构建），但「节点与 key 同环、顺时针找后继」的**归属规则完全一致**。

#### Amazon Dynamo（2007）

- **背景**：亚马逊购物车等高可用存储，追求「永远可写」与弹性伸缩。
- **在一致性哈希上的关键改进**：
  - **虚拟节点（virtual nodes / vnodes）**：每个物理节点在环上放置**多个**虚拟位置，解决「节点少时环分布不均、扩容负载迁移不均」的问题。这正是本文档下方“极限”里提到的**虚拟节点缓解均匀性**的来源。
  - **副本与 N/R/W**：key 顺时针的前 $N$ 个节点各存一份副本（本项目不涉及存储副本，仅借用环定位思想）。
- **对本项目的映射**：`HashRing` 为每个 Worker 生成多个虚拟节点以改善均匀性，思路直接来自 Dynamo 的 vnodes。

#### 为什么迁移量是 $1/N$

普通取模 $\text{hash}(k) \bmod n$：当 $n$ 从 $N$ 变为 $N+1$ 时，**绝大多数 key 的取模结果都会变**，接近 100% 重映射。
环形拓扑下增删一个节点，只有**落在该节点所负责的那段环弧**上的 key 需要改归属，其余 key 的「顺时针后继」不变——受影响比例约为该弧长占整个环的比例，即约 $1/N$。这就是「成员变更的爆炸半径被限制在 $1/N$」的直观来源。

### 边界与反模式
- **边界**：与 Manual 的分界线是**扩容行为**。一致性哈希在加节点时**会**重分布约 $1/N$ 的 key；Manual 加节点时**完全不**重分布已有会话。
- **极限**：Worker 数很少（如 2~3 个）时环的均匀性差，需虚拟节点缓解。
- **反模式**：把一致性哈希当负载均衡器用——它天然不感知实时负载，热 key 会持续压在同一节点。

### 演进动机
推翻了「取模映射」在动态拓扑下的可用性假设，是所有需要「稳定亲和 + 弹性伸缩」场景的基石。

---

## 6. Cache Aware（缓存感知）

**实现**：`src/policies/cache_aware.rs` + `src/policies/tree.rs` — 每个 Worker 维护近似基数树，在「缓存亲和」与「最短队列」间动态切换。

### 第一性原理降维
Cache Aware 是唯一**精确**估计 $H_i$ 的策略，也是最能体现「路由=在线调度」本质的策略。它对每个 Worker 用**近似基数树**（存原始字符而非 token，省去分词开销）记录历史请求前缀，从而精确计算最长前缀匹配率。

其核心是一个**双模式动态切换**：

$$
\text{imbalanced} \iff (\text{max} - \text{min}) > \tau_\text{abs} \;\wedge\; \text{max} > \tau_\text{rel} \cdot \text{min}
$$

- **系统均衡** → 走**缓存亲和**：
  - 若最高匹配率 > `cache_threshold`（默认 0.5）→ 路由到匹配最高的 Worker（复用 KV）；
  - 否则 → 路由到**树最小**的 Worker（缓存容量最富余）。
- **系统失衡** → 走**最短队列**：直接路由到 pending 最少的 Worker，牺牲缓存换取负载再平衡。

背景任务周期性 LRU 淘汰树叶节点（`eviction_interval_secs`、`max_tree_size`）以约束内存。特别地，树以 `pool::model` 为 key 隔离 prefill/decode/regular 池，防止交替调用互相驱逐。

### 高阶结构化类比
最贴切的类比是 **NUMA 感知操作系统调度器**：
| OS 调度器 | Cache Aware |
|---|---|
| CPU 本地缓存亲和 | KV/Prefix Cache 亲和 |
| 工作集迁移成本 | 重新 Prefill 成本 |
| 负载失衡时抢占迁移 | 失衡时切最短队列 |

它精确回答了那个反直觉命题：**最空闲的实例未必最优**——只要 $Q_A - H_A < Q_B - H_B$，选更忙但有缓存的 $A$ 反而完成更快。

### 边界与反模式
- **边界**：与 prefix_hash 的分界线是**精确 vs 近似**（基数树最长匹配 vs 固定前缀哈希）。
- **极限**：请求完全无共享前缀时，基数树退化为纯开销，此时应回退到更轻的策略。
- **反模式**：**缓存亲和绝对优先**。若只追 $\arg\max_i H_i$，会形成「缓存越热→流量越集中→缓存越热」的正反馈灾难。代码用「双阈值失衡检测 + 切最短队列」正是为破除此正反馈。

### 演进动机
它推翻了「所有健康实例等价」的假设，把路由从「无状态副本选择」进化为「**状态位置选择**」——这是模型网关区别于普通 L4/L7 负载均衡器的分水岭。

---

## 7. Manual（手动/强会话亲和）

**实现**：`src/policies/manual.rs` — 每个 routing key 粘死到固定 Worker，仅在其失联时才重映射，保留最多 2 个候选做快速故障切换。

### 第一性原理降维
Manual 追求的是**绝对粘性**，其不变量强于一致性哈希：

$$
\text{key} \mapsto \text{Worker} \text{ 一经建立，除非该 Worker 不健康，否则永不改变（即使扩容）}
$$

它维护 `key → [worker₁, worker₂]`（`MAX_CANDIDATE_WORKERS = 2`）映射，主 Worker 挂了立即切备用，避免会话中断。

### 高阶结构化类比
等价于**有状态服务的 sticky session**（如传统 Java 应用服务器的会话粘滞），或数据库的**主-备绑定**——上下文存在特定节点上，迁移意味着状态丢失。

### 边界与反模式
- **边界**：与一致性哈希的唯一但关键区别——**扩容不触发任何已有会话迁移**（一致性哈希会迁移 $1/N$）。用于「会话上下文存储在 Worker 本地」的强状态场景。
- **反模式**：长尾会话导致的**负载偏斜**——某些 key 极热而绑定关系又不允许迁移，会让个别 Worker 持续过载。需靠 `max_idle_secs` 淘汰空闲绑定缓解。

### 演进动机
当「会话状态无法廉价重建」时，一致性哈希的 $1/N$ 迁移都不可接受，于是演化出「只在故障时才动」的绝对粘性策略。

---

## 8. Bucket（分桶）

**实现**：`src/policies/bucket.rs` — 按 **请求长度（字符数）** 把 Prefill Worker 划分为若干「桶」，每个 Worker 负责一段长度区间；后台线程按滑动窗口内的真实负载**动态重划边界**，主要服务 PD（Prefill/Decode 分离）场景的 **prefill_policy**。

### 第一性原理降维
Bucket 的核心洞察是：**Prefill 阶段的计算成本近似正比于输入长度**。因此，与其在请求到达时估计每个 Worker 的排队/缓存状态，不如**按输入长度做静态分片**——让"长请求"和"短请求"分别落到固定的 Worker，从而把「重活」隔离开，避免长短请求混在同一实例上互相拖尾。

它把连续的长度轴 $[0, \infty)$ 切成 $N$ 段（$N$ = prefill worker 数），每个 Worker 认领一段区间 `[min, max]`：

$$
\text{route}(x) = \text{Worker}_k \quad\text{s.t.}\quad x \in [\text{min}_k,\ \text{max}_k]
$$

其中 $x$ 是本次请求的**字符数**（`request_text.chars().count()`，故 `needs_request_text() = true`）。选桶用**二分查找** `find_boundary`（边界有序），复杂度 $O(\log N)$。

**初始边界**是等分的：`gap = l_max / worker_cnt`（`l_max` 初值 4096），最后一个桶的上界扩到 `usize::MAX` 兜底任意超长请求。

### 双层机制：静态分桶 + 动态再平衡

**① 请求期（select_worker）——分桶为主、失衡时降级为最小负载**

每次选 Worker 时，先读取滑动窗口内各 Worker 的累计字符负载 `chars_per_url`，用**双阈值**判断是否失衡（与 Cache Aware 同构）：

$$
\text{imbalanced} \iff (\text{max} - \text{min}) > \tau_\text{abs} \;\wedge\; \text{max} > \tau_\text{rel} \cdot \text{min}
$$

- **未失衡** → 走 **Bucket 分桶**：按请求长度二分命中所属区间的 Worker（长度决定归属，可预测、稳定）。
- **已失衡** → 临时降级为**最小负载优先**：直接选当前累计字符最少的 Worker，牺牲长度亲和换取快速再平衡。

选定后 `post_process_request` 会：把本次 `char_cnt` 累加到该 URL 的负载、生成一条带时间戳的 `SequencerRequest` 入队，并**淘汰超出滑动窗口**（`period = bucket_adjust_interval_secs × 1000` ms）的历史请求、回滚其负载。即负载统计是一个**时间滑动窗口**，只反映最近一段时间的流量。

**② 后台期（adjust_boundary）——按负载分位重划边界**

后台线程每隔 `bucket_adjust_interval_secs`（默认 5s）对每个模型的桶做一次边界重算，目标是让**每个桶承担的总负载尽量均等**：

1. 计算目标单桶负载 `new_single_bucket_load = 总负载 / worker_cnt`；
2. **迟滞（hysteresis）保护**：若新旧单桶负载相差不到 2 倍（且旧值非 0），认为无需调整，直接跳过——避免边界频繁抖动；
3. 否则把窗口内所有请求长度**排序**，按"累积负载达到单桶目标"为切点，依次给每个 Worker 划定新的 `[min, max]` 区间（本质是**按负载做等分位切分**，而非按长度等分）。

这样，如果短请求特别多，短请求区间会被切得更细（多个 Worker 分摊）；长请求稀疏，则由少数 Worker 覆盖大长度区间——**边界随真实长度分布自适应**。

### 关键数据结构
| 字段 | 作用 |
|---|---|
| `boundary: Vec<Boundary>` | 有序的 `{url, [min,max]}` 列表，二分选桶的依据 |
| `chars_per_url` | 各 Worker 在滑动窗口内的累计字符负载（失衡判断 + 重划分位） |
| `request_list: VecDeque<SequencerRequest>` | 按时间排序的请求队列，用于滑动窗口过期淘汰 |
| `period` | 滑动窗口长度（ms），等于调整间隔 |

桶以 `normalize_model_key(model_id)` 为 key 隔离，不同模型各自维护独立分桶与负载窗口。Worker 增删（`add_prefill_url` / `remove_prefill_url`）会重置边界并同步 `chars_per_url`。

### 高阶结构化类比
类似磁盘的**分区/分级存储**或数据库的**范围分片（range sharding）**——按 key（这里是"请求长度"）的区间把负载路由到固定分片，再用后台任务按实际数据分布**动态调整分片边界**（类比 HBase Region Split / 自动 rebalance）。也可类比 CPU 调度里把长短任务分到不同队列的**多级队列**思想。

### 边界与反模式
- **边界**：适用范围窄，主要作为 PD 分离的 **prefill_policy**（长度 ≈ prefill 计算量时最有效），不建议作为通用 Regular 流量默认策略。
- **与 Cache Aware 的区别**：两者都用"双阈值失衡检测 + 降级"，但 Bucket 的一等公民是**请求长度分片**（无状态、可预测），Cache Aware 的一等公民是**前缀缓存亲和**（有状态）。Bucket 不感知 KV 缓存。
- **反模式**：
  - 当请求长度与真实计算成本**弱相关**时（如长度相近但难度差异大），长度分桶失去意义。
  - 迟滞阈值（2×）设置过松会导致边界长期不更新、分桶僵化；过紧则边界抖动、路由不稳定。
  - 长度分布极度长尾时，个别覆盖超长区间的 Worker 可能持续偏载，依赖失衡降级兜底。

### 演进动机
在 PD 分离下，prefill 是"算力密集、成本随长度线性增长"的阶段。把"按长度分片 + 按负载自适应重划边界"结合，既保留了**同长度请求路由稳定**的可预测性，又通过后台再平衡与请求期失衡降级，避免了静态分片在流量倾斜时的僵化——这是为"长度即成本"这一 prefill 特性量身定制的均衡器。

---

## 9. 统一收敛：如何选择策略

回到第 0 节的目标函数，各策略只是对不同项的侧重：

$$
\arg\min_i \big( Q_i + C_i - H_i + R_i \big)
$$

| 工作负载特征 | 推荐策略 | 理由 |
|---|---|---|
| 成本相近、无共享前缀 | Round Robin / Random | 无可利用信息，均分即最优 |
| 突发负载、成本方差大 | Power of Two | 低成本估计 $Q_i$，抑制尾延迟 |
| 大量相同 system prompt | Cache Aware | 精确最大化 $H_i$，兼顾失衡切换 |
| 前缀共享但要求稳定延迟 | Prefix Hash | 近似 $H_i$，$O(\log n)$ 可预测 |
| 多租户会话、动态扩缩容 | Consistent Hashing | 亲和 + $1/N$ 迁移 |
| 强状态会话、不可迁移 | Manual | 绝对粘性 + 故障切换 |
| PD 分离部署 | Bucket | 阶段异构分桶均衡 |

### 核心本质结论

> **路由策略的全部差异，本质是在「未来成本未知、可复用状态分散在不同机器」这一约束下，用不同精度、不同观测成本去估计「排队 + 计算 − 缓存复用 + 风险」，并在『利用已有状态』与『避免因利用状态而制造热点』之间划定各自的安全边界。**

