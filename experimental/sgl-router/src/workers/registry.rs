// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use crate::health::circuit_breaker::CircuitBreakerConfig;
use crate::workers::worker::Worker;
use dashmap::DashMap;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

/// [`WorkerRegistry::add`] 调用拒绝某个 spec（worker 规格）的原因。
#[derive(Debug, Clone, thiserror::Error)]
pub enum AddWorkerError {
    /// spec 的模式（plain 普通模式 vs prefill/decode 预填充/解码模式）与
    /// 该 spec 的某个 `model_ids` 下已注册的 worker 发生冲突。router 不
    /// 支持在单个模型上混用 PD（PD 分离）与 plain（普通）worker 池：
    /// resolver（解析器）会根据已注册的 worker 推导出该模型是 PD 还是
    /// plain 形态；一旦混用，当其中一侧因熔断器打开（breaker-open）而不
    /// 可用时，就会静默降级到恰好健康的另一侧形态，从而向客户端返回错误
    /// 的错误码。
    #[error(
        "worker {worker:?} for model {model:?} would mix PD ({pd_mode}) with plain workers on \
         the same model — sgl-router does not support mixed pools. Use one of: only Plain \
         workers, or only Prefill+Decode workers."
    )]
    MixedPdAndPlain {
        worker: WorkerId,
        model: ModelId,
        /// 触发本次冲突的*新加入* worker 的角色
        /// （*已存在*的 worker 则是相反的角色）。
        pd_mode: &'static str,
    },
}

/// Worker 注册表：维护 worker 到 ID、模型到 worker 的双向索引。
#[derive(Debug, Default)]
pub struct WorkerRegistry {
    /// 主索引：WorkerId -> Worker。所有 worker 实体都存放在这里，
    /// 其余索引仅持有 WorkerId 引用。
    by_id: DashMap<WorkerId, Arc<Worker>>,
    /// 反向索引：ModelId -> 服务该模型的 WorkerId 集合。
    /// 用于按模型快速检索候选 worker（`workers_for`）。
    by_model: DashMap<ModelId, HashSet<WorkerId>>,
    /// 将 `add_with_cb` 中“校验→插入”这段临界区串行化，使得
    /// `MixedPdAndPlain`（混用检查）与随后的写入操作保持原子性。否则，
    /// 来自 `manager::register_one` 的两个并发注册，若针对同一模型且模式
    /// 冲突，可能都观察到空池并都执行插入，从而让注册表进入混用状态——
    /// 这正是该检查要预防的破坏。
    /// 读操作（`workers_for`、`get`、`len` 等）对底层 DashMap 保持无锁；
    /// 只有经过 `add_with_cb` / `remove` 的写操作才会获取此锁，因此锁竞争
    /// 仅受注册表变更频率（worker 发现事件）约束，而非请求速率。
    write: Mutex<()>,
}

impl WorkerRegistry {
    /// 添加一个 worker，使用默认熔断器配置（阈值 = 3）。
    /// 是 `add_with_cb(spec, None)` 的便捷封装。
    pub fn add(&self, spec: WorkerSpec) -> Result<(), AddWorkerError> {
        self.add_with_cb(spec, None)
    }

    /// 添加一个 worker，可选地提供熔断器（circuit-breaker）配置。
    /// 传入 `None` 表示使用熔断器默认配置（阈值 = 3）。
    ///
    /// 重复添加一个已存在的 `WorkerId` 属于 upsert（更新插入）：会先清除
    /// 旧条目在 `by_model` 中的所有成员关系，这样新 spec 不再服务的模型
    /// 就不会再解析到该 worker。若缺少这一“先移除”步骤，一个模型集合缩
    /// 小了的 worker 仍会出现在 `workers_for(<已移除的模型>)` 中，因为
    /// `by_id.get(...)` 会通过陈旧的 模型→ID 索引返回新的 worker。
    ///
    /// 当添加该 spec 会导致同一模型上 PD（prefill/decode）worker 与 plain
    /// worker 混用时，返回 [`AddWorkerError::MixedPdAndPlain`]。冲突是相对
    /// *现有*注册表状态来检测的——重复添加相同的 worker id 没问题（会先移
    /// 除旧条目）；只要某个 worker 自身的 `model_ids` 都不混用，即便进程内
    /// 其他模型存在模式混用，添加它也没问题。
    ///
    /// 被拒绝时，注册表**不会**被修改。如果被拒绝的 spec 携带的 id 已有对
    /// 应条目，旧条目会原样保留——是否驱逐它由调用方决定（并且，重要的是，
    /// 若驱逐还需清理 `KvEventIndex` / `ActiveLoadRegistry` 中的附属状态）。
    /// 如果在这里做清理，当调用方本想保留旧条目时，就会向那些附属结构泄漏
    /// 出孤儿状态。
    pub fn add_with_cb(
        &self,
        spec: WorkerSpec,
        cb: Option<CircuitBreakerConfig>,
    ) -> Result<(), AddWorkerError> {
        // 记录新加入 worker 的模式，用于后续混用检查。
        let incoming_mode = spec.mode;
        // 在整个“校验→插入”序列期间持有写锁。
        // 否则，两个针对同一模型、模式冲突的并发调用方，可能都看到空池并都
        // 继续执行插入，从而产生该检查本应预防的 PD+plain 混用状态。
        //
        // 此处 Mutex 中毒（poisoning）意味着上一个写入方在持锁时发生了
        // panic——而由于临界区涵盖 `remove_locked` + 若干次 `by_model` 更
        // 新 + 最后的 `by_id.insert`，中途 panic 可能在两个 DashMap 之间留
        // 下一个残缺条目。通过 `PoisonError::into_inner` 恢复会针对这个写
        // 了一半的状态静默继续；相反，向上传播 panic 会把这一破坏暴露给
        // `manager::register_one` 的任务，并最终触发
        // `supervise_critical_tasks → mark_unready`，使该 pod 停止接收流量。
        // 这才是正确的结果。
        let _guard = self.write.lock().unwrap();
        // 在修改之前先针对现有 worker 做校验。重复添加相同 id 属于 upsert；
        // 为了检查目的，假装旧条目已不存在（否则，一个不混用的 worker 做
        // upsert 时，如果它当前的条目已在服务该模型，就会与自身冲突）。
        for model in &spec.model_ids {
            for existing in self.workers_for(model) {
                // 跳过自身：同 id 的重复添加是 upsert，不算冲突。
                if existing.id == spec.id {
                    continue;
                }
                // 若新旧模式不能共存（plain 与 PD 混用），拒绝本次添加。
                if modes_are_mixed(incoming_mode, existing.mode()) {
                    return Err(AddWorkerError::MixedPdAndPlain {
                        worker: spec.id,
                        model: model.clone(),
                        pd_mode: mode_name(incoming_mode),
                    });
                }
            }
        }
        // 校验通过，构造 worker 实体（携带熔断器配置）。
        let w = Arc::new(Worker::with_cb_config(spec, cb));
        let id = w.id.clone();
        // 先移除旧条目（若存在），确保 upsert 时陈旧的模型索引被清理。
        self.remove_locked(&id);
        // 为该 worker 服务的每个模型建立 模型→ID 反向索引。
        for m in &w.model_ids {
            self.by_model
                .entry(m.clone())
                .or_default()
                .insert(id.clone());
        }
        // 最后写入主索引。
        self.by_id.insert(id, w);
        Ok(())
    }

    /// 从注册表中移除指定 worker。
    pub fn remove(&self, id: &WorkerId) {
        // 与 `add_with_cb` 一样获取写锁，使移除操作不会与并发的添加操作竞
        // 争（否则一份陈旧的 `workers_for` 快照可能让一次添加相对一个即将
        // 被移除的对端成功，反之亦然）。
        let _guard = self.write.lock().unwrap();
        self.remove_locked(id);
    }

    /// 内部移除逻辑，假定调用方已持有写锁。
    /// 任何已获取 `self.write` 的路径都应调用此函数。
    fn remove_locked(&self, id: &WorkerId) {
        // 先从主索引移除；若确实存在，再清理它在各模型下的反向索引。
        if let Some((_, w)) = self.by_id.remove(id) {
            for m in &w.model_ids {
                if let Some(mut set) = self.by_model.get_mut(m) {
                    set.remove(id);
                }
            }
        }
    }

    /// 返回服务指定模型的所有 worker（不区分熔断器状态）。
    /// 通过 模型→ID 反向索引查出 ID 集合，再回主索引取出 worker 实体。
    pub fn workers_for(&self, model: &ModelId) -> Vec<Arc<Worker>> {
        self.by_model
            .get(model)
            .map(|ids| {
                ids.iter()
                    // ID 可能因并发移除而失效，用 filter_map 跳过缺失项。
                    .filter_map(|i| self.by_id.get(i).map(|w| Arc::clone(&w)))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// 返回服务指定模型且当前可用（熔断器未打开）的 worker 列表。
    pub fn healthy_workers_for(&self, model: &ModelId) -> Vec<Arc<Worker>> {
        // 过滤时使用 `would_allow`（不产生副作用）——而 `allow()` 会为每个
        // 被枚举的候选占用一个半开（half-open）探测名额，从而让策略真正选
        // 中的那个 worker 反而被“饿死”。探测名额是在派发时由
        // [`crate::proxy`] 中的 `forward_*_to` 占用的。
        self.workers_for(model)
            .into_iter()
            .filter(|w| w.breaker.would_allow())
            .collect()
    }

    /// 返回服务指定模型且模式匹配（Plain/Prefill/Decode）的 worker 列表。
    pub fn workers_for_mode(&self, model: &ModelId, mode: WorkerMode) -> Vec<Arc<Worker>> {
        self.workers_for(model)
            .into_iter()
            .filter(|w| w.mode() == mode)
            .collect()
    }

    /// 返回已注册 worker 的总数。
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// 注册表是否为空。
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// 按 ID 获取单个 worker。
    pub fn get(&self, id: &WorkerId) -> Option<Arc<Worker>> {
        self.by_id.get(id).map(|w| Arc::clone(&w))
    }

    /// 返回所有已注册 worker 的快照，跨全部模型与模式，且不区分熔断器状态。
    /// 顺序未定义（直接遍历底层 `DashMap`）。
    ///
    /// 用于面向整个集群的管理性 fan-out（例如 `/flush_cache`）——它面向
    /// router 已知的每一个 worker，而非某个模型的池；也用于 `/metrics`
    /// 抓取路径，渲染每个 worker 的指标（`sgl_router_worker_health`、
    /// `_cb_state`、`_inflight_requests`）以及池大小指标。metrics 路径在每
    /// 次抓取时都重新采样（而非推送），因此被移除的 worker 会立即不再出现。
    pub fn all(&self) -> Vec<Arc<Worker>> {
        self.by_id.iter().map(|e| Arc::clone(e.value())).collect()
    }
}

/// 当两种模式无法在同一模型上共存时返回 `true`——即一方是 `Plain`
/// 而另一方是 `Prefill` 或 `Decode`。
fn modes_are_mixed(a: WorkerMode, b: WorkerMode) -> bool {
    matches!(
        (a, b),
        (WorkerMode::Plain, WorkerMode::Prefill | WorkerMode::Decode)
            | (WorkerMode::Prefill | WorkerMode::Decode, WorkerMode::Plain)
    )
}

fn mode_name(m: WorkerMode) -> &'static str {
    match m {
        WorkerMode::Plain => "plain",
        WorkerMode::Prefill => "prefill",
        WorkerMode::Decode => "decode",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};

    fn spec(id: &str, mode: WorkerMode, models: &[&str]) -> WorkerSpec {
        WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode,
            model_ids: models.iter().map(|m| ModelId((*m).into())).collect(),
            bootstrap_port: None,
        }
    }

    #[test]
    fn add_then_query_by_model() {
        let r = WorkerRegistry::default();
        let _ = r.add(spec("w1", WorkerMode::Plain, &["m1", "m2"]));
        let _ = r.add(spec("w2", WorkerMode::Plain, &["m1"]));
        let m1 = r.workers_for(&ModelId("m1".into()));
        let m2 = r.workers_for(&ModelId("m2".into()));
        let m_missing = r.workers_for(&ModelId("missing".into()));
        assert_eq!(m1.len(), 2);
        assert_eq!(m2.len(), 1);
        assert!(m_missing.is_empty());
    }

    #[test]
    fn all_returns_every_worker_across_models_and_modes() {
        let r = WorkerRegistry::default();
        let _ = r.add(spec("w1", WorkerMode::Plain, &["m1"]));
        let _ = r.add(spec("p", WorkerMode::Prefill, &["m2"]));
        let _ = r.add(spec("d", WorkerMode::Decode, &["m2"]));
        let mut ids: Vec<String> = r.all().into_iter().map(|w| w.id.0.clone()).collect();
        ids.sort();
        assert_eq!(ids, vec!["d", "p", "w1"]);
    }

    #[test]
    fn all_is_empty_for_fresh_registry() {
        assert!(WorkerRegistry::default().all().is_empty());
    }

    #[test]
    fn remove_drops_from_all_models() {
        let r = WorkerRegistry::default();
        let _ = r.add(spec("w1", WorkerMode::Plain, &["m1", "m2"]));
        r.remove(&WorkerId("w1".into()));
        assert!(r.workers_for(&ModelId("m1".into())).is_empty());
        assert!(r.workers_for(&ModelId("m2".into())).is_empty());
    }

    #[test]
    fn all_lists_multi_model_worker_once() {
        let r = WorkerRegistry::default();
        // “a”服务两个模型；`all` 必须仍只列出它一次，
        // 不同于按模型枚举会重复计数。
        let _ = r.add(spec("a", WorkerMode::Plain, &["m1", "m2"]));
        let _ = r.add(spec("b", WorkerMode::Plain, &["m1"]));
        let mut urls: Vec<String> = r.all().iter().map(|w| w.url.clone()).collect();
        urls.sort();
        assert_eq!(urls, vec!["http://a:30000", "http://b:30000"]);
    }

    /// `healthy_workers_for` 必须剔除熔断器处于 Open（打开）状态的
    /// worker。本测试的早期版本对两个熔断器未被触碰的 worker 断言
    /// `healthy.len() == 2`——即它只针对空操作情形（两者都 Closed），
    /// 即便 `healthy_workers_for` 完全忽略熔断器、仅仅是 `workers_for`
    /// 的薄封装别名，也能通过。真正的契约是：触发一个熔断器，并
    /// 断言存活集合不包含它。
    #[test]
    fn healthy_subset_filters_via_breaker() {
        use crate::health::circuit_breaker::CircuitBreakerConfig;
        use std::num::NonZeroU32;
        use std::time::Duration;

        let r = WorkerRegistry::default();
        let _ = r.add_with_cb(spec("ok", WorkerMode::Plain, &["m"]), None);
        // 给 "bad" 一个阈值=1 的熔断器，这样一次 record_failure 就能把它
        // 翻转到 Open 状态。
        let _ = r.add_with_cb(
            spec("bad", WorkerMode::Plain, &["m"]),
            Some(CircuitBreakerConfig {
                threshold: NonZeroU32::new(1).unwrap(),
                cool_down: Duration::from_secs(30),
            }),
        );
        let bad = r.get(&WorkerId("bad".into())).expect("bad worker present");
        bad.breaker.record_failure();
        assert!(
            !bad.breaker.would_allow(),
            // 健康性断言：阈值=1 + 一次失败必须使熔断器进入 Open。
            "sanity: threshold=1 + one failure must Open the breaker",
        );

        let healthy = r.healthy_workers_for(&ModelId("m".into()));
        assert_eq!(
            healthy.len(),
            1,
            // 只有熔断器非 Open 的 worker 应当存活。
            "only the worker with a non-Open breaker should survive",
        );
        assert_eq!(healthy[0].id, WorkerId("ok".into()));
    }

    /// PD prefill/decode worker 与 plain worker 无法在同一模型上共存。
    /// resolver 基于已注册 worker 确定其 PD-vs-plain 形态；一旦混用，当其
    /// 中一侧因熔断器打开而不可用时，就会被迫回退到恰好健康的另一桶，
    /// 从而返回错误的 5xx 码（返回 `no_healthy_workers` 而不是
    /// `no_prefill_workers_available`）。先行拒绝冲突的添加，让运维人员
    /// 立即发现该配置错误。
    #[test]
    fn plain_then_pd_for_same_model_is_rejected() {
        let r = WorkerRegistry::default();
        assert!(r.add(spec("plain", WorkerMode::Plain, &["m"])).is_ok());
        let err = r
            .add(spec("p", WorkerMode::Prefill, &["m"]))
            .expect_err("PD worker must be rejected when model already has Plain workers");
        let msg = err.to_string();
        assert!(
            msg.contains("PD") && msg.contains("plain"),
            "error must name both modes; got: {msg}"
        );
        // 已存在的 plain worker 在拒绝后应存活。
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Plain)
                .len(),
            1,
        );
        assert!(r
            .workers_for_mode(&ModelId("m".into()), WorkerMode::Prefill)
            .is_empty());
    }

    #[test]
    fn pd_then_plain_for_same_model_is_rejected() {
        let r = WorkerRegistry::default();
        assert!(r.add(spec("p", WorkerMode::Prefill, &["m"])).is_ok());
        assert!(r.add(spec("d", WorkerMode::Decode, &["m"])).is_ok());
        let err = r
            .add(spec("plain", WorkerMode::Plain, &["m"]))
            .expect_err("plain worker must be rejected when model already has PD workers");
        let msg = err.to_string();
        assert!(
            msg.contains("PD") && msg.contains("plain"),
            "error must name both modes; got: {msg}"
        );
    }

    #[test]
    fn plain_only_pool_admits_more_plain_workers() {
        let r = WorkerRegistry::default();
        assert!(r.add(spec("a", WorkerMode::Plain, &["m"])).is_ok());
        assert!(r.add(spec("b", WorkerMode::Plain, &["m"])).is_ok());
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Plain)
                .len(),
            2,
        );
    }

    #[test]
    fn pd_pool_admits_more_pd_workers_in_both_roles() {
        let r = WorkerRegistry::default();
        assert!(r.add(spec("p1", WorkerMode::Prefill, &["m"])).is_ok());
        assert!(r.add(spec("p2", WorkerMode::Prefill, &["m"])).is_ok());
        assert!(r.add(spec("d1", WorkerMode::Decode, &["m"])).is_ok());
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Prefill)
                .len(),
            2,
        );
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Decode)
                .len(),
            1,
        );
    }

    /// 以缩小的 `model_ids` 重新添加一个 worker 时，必须将该 worker 从它
    /// 不再服务的模型中移除。早期实现只更新了 `by_id`，遗留了陈旧的
    /// `by_model` 条目仍指向新 worker。
    #[test]
    fn re_add_with_shrunken_model_set_drops_stale_indexes() {
        let r = WorkerRegistry::default();
        let _ = r.add(spec("w1", WorkerMode::Plain, &["m1", "m2"]));
        assert_eq!(r.workers_for(&ModelId("m2".into())).len(), 1);

        let _ = r.add(spec("w1", WorkerMode::Plain, &["m1"]));
        assert_eq!(
            r.workers_for(&ModelId("m2".into())).len(),
            0,
            // 重新添加后 w1 不再服务 m2。
            "w1 no longer serves m2 after re-add"
        );
        assert_eq!(
            r.workers_for(&ModelId("m1".into())).len(),
            1,
            // w1 仍服务 m1。
            "w1 still serves m1"
        );
    }

    /// 以不同模式重新添加相同 id，应体现在 `workers_for_mode` 中。
    #[test]
    fn re_add_with_different_mode_updates_mode_filter() {
        let r = WorkerRegistry::default();
        let _ = r.add(spec("w1", WorkerMode::Prefill, &["m"]));
        let _ = r.add(spec("w1", WorkerMode::Decode, &["m"]));
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Prefill)
                .len(),
            0,
        );
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Decode)
                .len(),
            1,
        );
    }

    /// 当 upsert 因 `MixedPdAndPlain` 被拒绝时，注册表**不会**被修改——
    /// 被拒绝 id 的旧条目会原样保留。驱逐（及配套的 `KvEventIndex` /
    /// `ActiveLoadRegistry` 清理）是 manager 的职责；在这里做会向那些
    /// 附属结构泄漏孤儿状态。
    #[test]
    fn upsert_rejected_with_mixed_modes_leaves_registry_unchanged() {
        let r = WorkerRegistry::default();
        // 模型 m 上有一个健康的 PD 池。
        let _ = r.add(spec("p", WorkerMode::Prefill, &["m"]));
        let _ = r.add(spec("d", WorkerMode::Decode, &["m"]));
        // 以 Plain 模式重新添加 "p"——发现机制报告了一次角色翻转。
        // 解码 worker "d" 仍在 m 上，因此校验拒绝。
        let err = r
            .add(spec("p", WorkerMode::Plain, &["m"]))
            .expect_err("plain upsert must be rejected when peer decode worker remains");
        assert!(err.to_string().contains("plain"), "got: {err}");
        // 旧的 "p" 条目存活（仍为 Prefill）。注册表故意不在拒绝时自动驱
        // 逐——驱逐（及配套的附属状态清理）由调用方决定。
        let p = r
            .get(&WorkerId("p".into()))
            .expect("prior entry must remain — caller owns the cleanup");
        assert_eq!(p.mode(), WorkerMode::Prefill);
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Prefill)
                .len(),
            1,
        );
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Decode)
                .len(),
            1,
        );
    }

    /// 一个*新的*（尚未注册的）worker 因 `MixedPdAndPlain` 被拒绝时，不得
    /// 影响池。结合上面的 upsert 测试，这锁定了：拒绝本身绝不会修改
    /// 注册表状态。
    #[test]
    fn rejected_new_add_leaves_pool_untouched() {
        let r = WorkerRegistry::default();
        let _ = r.add(spec("plain", WorkerMode::Plain, &["m"]));
        let err = r
            .add(spec("p", WorkerMode::Prefill, &["m"]))
            .expect_err("PD worker must be rejected against existing plain pool");
        assert!(err.to_string().contains("plain"), "got: {err}");
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Plain)
                .len(),
            1,
        );
        assert!(r.get(&WorkerId("p".into())).is_none());
    }

    /// 来自 `manager::register_one` 的并发注册会相互竞争：每个被生成的
    /// 任务都并行调用 `add_with_cb`，而该方法内部的“先校验后插入”序
    /// 列**不是**原子的。两个为同一模型添加模式冲突 worker 的线程，
    /// 可能都通过现有 worker 检查（各自看到空池）并都继续插入，从而让
    /// 注册表进入 PD+plain 混用状态——正是 `MixedPdAndPlain` 检查本应
    /// 预防的破坏。
    ///
    /// 我们锁定的不变式：对每个模型，最终的池必须要么全为 Plain，要
    /// 么全为 PD，绝不混用。我们不关心究竟选中哪种“胜出”模式——竞
    /// 争中的 manager 已按 WorkerId 串行化，因此这里需要原子性的是跨 id
    /// 的情形。
    #[test]
    fn concurrent_conflicting_modes_never_produce_mixed_pool() {
        use std::sync::Arc;
        use std::sync::Barrier;
        use std::thread;

        // 所有线程都针对同一个共享模型，因此每个 `add_with_cb` 竞争者都在
        // 同一个 `workers_for("m")` 槽位上争用——这正是使一个线程的“读-
        // 校验-写”窗口与另一个线程的修改重叠的原因。早期变体把负载分
        // 散到 4 个模型上，无法稳定复现该 bug（每槽位竞争被稀释到约
        // N/4 个线程）。200 次迭代 × 16 线程在作者机器上能在前几次迭
        // 代内触发该竞争；修复后，该不变式必须在每一次迭代中都成立。
        const N_THREADS: usize = 16;
        const ITER: usize = 200;

        for iter in 0..ITER {
            let r = Arc::new(WorkerRegistry::default());
            let barrier = Arc::new(Barrier::new(N_THREADS));
            let mut handles = Vec::with_capacity(N_THREADS);
            for t in 0..N_THREADS {
                let r = Arc::clone(&r);
                let barrier = Arc::clone(&barrier);
                // 一半线程注册 Plain worker，另一半注册 Prefill，全部在同一模型
                // 上。若 `add_with_cb` 内部的 校验→写入 非原子，一个 Plain 线
                // 程和一个 Prefill 线程会都看到空池并都成功。
                let mode = if t % 2 == 0 {
                    WorkerMode::Plain
                } else {
                    WorkerMode::Prefill
                };
                let id = format!("iter{iter}-t{t}");
                handles.push(thread::spawn(move || {
                    barrier.wait();
                    let _ = r.add(spec(&id, mode, &["m"]));
                }));
            }
            for h in handles {
                h.join().unwrap();
            }

            // 不变式检查：模型为单一模式。
            let model = ModelId("m".into());
            let plain = r.workers_for_mode(&model, WorkerMode::Plain).len();
            let prefill = r.workers_for_mode(&model, WorkerMode::Prefill).len();
            let decode = r.workers_for_mode(&model, WorkerMode::Decode).len();
            let pd = prefill + decode;
            assert!(
                plain == 0 || pd == 0,
                // 第 {iter} 次迭代：注册表持有混用池——plain={plain}，
                // prefill={prefill}，decode={decode}。`add_with_cb` 中的
                // MixedPdAndPlain 检查在并发调用方之间不是原子的。
                "iter {iter}: registry holds a mixed pool — \
                 plain={plain}, prefill={prefill}, decode={decode}. \
                 The MixedPdAndPlain check in `add_with_cb` is not atomic \
                 across concurrent callers.",
            );
            // 健康性断言：第一个拿到锁的线程必须成功（此时尚无对端）。
            // 防御一种退化的“修复”——它通过静默拒绝每一次添加来满足单
            // 一模式不变式。
            assert!(
                plain + pd >= 1,
                "iter {iter}: no workers were registered — \
                 the lock or mixed-mode check is starving every caller.",
            );
        }
    }
}
