# Power of Two Choices（二选一）算法详解

> 本文档是 [ROUTING_POLICIES_ZH.md](./ROUTING_POLICIES_ZH.md) 中「Power of Two」策略的深入拆解，
> 包含数学推导、负载信号语义与更新机制等细节。主文档仅保留概览。

**实现**：`src/policies/power_of_two.rs` — 随机抽 2 个 Worker，比较负载取低者。

## 第一性原理降维

这是本仓库唯一真正估计 $Q_i$（实时负载）的**轻量**策略。它的精髓不在「选负载低的」，而在「**只看两个**」：

$$
\text{selected} = \arg\min_{i \in \{a,b\}} \text{load}(i), \quad a,b \sim \text{Uniform}(\mathcal{H}),\ a \neq b
$$

理论结论（Mitzenmacher）：全局最忙 Worker 的期望负载从随机的 $O(\log n / \log\log n)$ 降到 $O(\log\log n)$——**只用两次采样就获得接近全局最优的尾部收益**，却避免了扫描全局带来的观测成本与「羊群效应」。

### 举例：为什么「看两个」就够了

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

### 数学原理：最忙 Worker 负载为何是 $\log\log n$

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

## `load_tokens` 的含义与更新周期

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

## 高阶结构化类比

等价于分布式哈希负载均衡中的 **"the power of two random choices"**，也类比 CPU 调度里的**局部窥探**（只看邻近核心而非全局队列）以规避全局锁竞争。

## 边界与反模式

- **边界**：与「全局最少负载」的分界线是**观测范围**。全局最少负载会因所有请求同时看到同一个「最空闲」节点而振荡（$A$ 空 → 全涌向 $A$ → $A$ 过载 → 全涌向 $B$）；二选一用随机采样天然打散这种同步。
- **极限**：负载数据严重过期（$\Delta_\text{state} \gg \tau_\text{load}$）时退化为近似随机。
- **反模式**：在负载信号缺失/延迟大的环境里强行相信采样值。

## 演进动机

推翻了「必须扫描全局才能优化尾延迟」的假设。它是**观测成本**与**均衡质量**之间的帕累托最优点。

