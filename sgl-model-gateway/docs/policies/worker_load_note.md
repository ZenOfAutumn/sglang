# 注意：`worker.load()` 的语义

> 本文档抽取自各路由策略共用的一个易混淆点，供 [power_of_two.md](./power_of_two.md)、
> [prefix_hash.md](./prefix_hash.md)、[index.md](./index.md) 等文档引用。

## 结论

**`worker.load()` 返回的是「活跃请求数」（在途 / 并发请求数），而不是 token 数。**

它由一个原子计数器维护：

- 请求进入该 worker 时 **+1**（`increment_load`）；
- 请求完成（或失败、被取消）时 **−1**（`decrement_load`）。

因此任意时刻 `worker.load()` 等于「当前正在该 worker 上处理、尚未返回的请求条数」。

对应实现见 `src/core/worker.rs`：

```text
/// 获取当前负载（在途请求数）
fn load(&self) -> usize;
/// 递增负载计数器
fn increment_load(&self);
/// 递减负载计数器
fn decrement_load(&self);
```

## 为什么需要特别强调

多个策略在计算负载时会用到两类量纲完全不同的「负载」，容易混淆：

| 量纲 | 来源 | 含义 | 使用方 |
|---|---|---|---|
| **活跃请求数** | `worker.load()`（网关本地原子计数） | 在途 / 并发请求**条数** | Prefix Hash 的 `load_ok`、Power of Two 的降级路径、Manual 的 `MinLoad` 模式 |
| **token 负债** | 引擎 `/v1/loads` 的 `total_tokens` | running + waiting 的 **token** 总数 | Power of Two 的主路径（`cached_loads`） |

关键区别：

- **量纲不同**：一个是「请求条数」，一个是「token 数」，两者不可直接比较或混用。
- **保真度不同**：`total_tokens` 能反映请求的真实计算成本（长短请求差异巨大），而「活跃请求数」把每个请求视为等权，精度较低。
- **可得性不同**：`worker.load()` 是网关本地维护、恒可用；`total_tokens` 需要向引擎拉取，失败时会降级回 `worker.load()`——**正是在这种降级场景下，两个量纲必须保持一致**（同时使用请求计数），否则比较无意义。

## 一句话

凡文档中出现 `worker.load()`，一律指**在途请求条数**；凡出现 `total_tokens` / token 负载，指**引擎侧的 token 负债**。二者量纲不同，仅在 Power of Two 拉取 token 负载失败降级时，才统一回退到前者。

