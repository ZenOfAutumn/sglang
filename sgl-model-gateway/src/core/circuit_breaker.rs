use std::{
    sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering},
    time::{Duration, Instant},
};

use tracing::info;

use crate::observability::metrics::Metrics;

/// Worker 熔断器配置。
///
/// 熔断器按 `Closed -> Open -> HalfOpen -> Closed` 的状态机工作：
/// - `Closed`：正常放行请求，并统计连续成功/失败次数；
/// - `Open`：拒绝路由请求，使故障 Worker 进入冷却期；
/// - `HalfOpen`：冷却结束后的探测状态，暂时放行请求以判断 Worker 是否恢复。
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// `Closed` 状态下触发熔断所需的连续失败次数。
    /// 任意一次成功都会把连续失败计数清零。
    pub failure_threshold: u32,
    /// `HalfOpen` 状态下恢复为 `Closed` 所需的连续成功次数。
    pub success_threshold: u32,
    /// `Open` 状态的冷却时长；到期后在下一次状态检查时转为 `HalfOpen`。
    pub timeout_duration: Duration,
    /// 预留的失败统计窗口配置。
    ///
    /// 注意：当前实现只统计连续失败，尚未依据该时间窗口淘汰历史失败。
    pub window_duration: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout_duration: Duration::from_secs(30),
            window_duration: Duration::from_secs(60),
        }
    }
}

/// 熔断器状态的原子存储编码，避免在请求热路径上加锁。
const STATE_CLOSED: u8 = 0;
const STATE_OPEN: u8 = 1;
const STATE_HALF_OPEN: u8 = 2;

/// Worker 熔断器状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// 关闭：Worker 正常，请求可以执行。
    Closed,
    /// 打开：Worker 正在冷却，请求不可执行。
    Open,
    /// 半开：冷却期已结束，允许探测请求判断 Worker 是否恢复。
    ///
    /// 当前实现没有限制并发探测数，因此处于半开状态时可能同时放行多个请求。
    HalfOpen,
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitState::Closed => write!(f, "Closed"),
            CircuitState::Open => write!(f, "Open"),
            CircuitState::HalfOpen => write!(f, "HalfOpen"),
        }
    }
}

impl CircuitState {
    pub fn as_str(&self) -> &'static str {
        match self {
            CircuitState::Closed => "closed",
            CircuitState::Open => "open",
            CircuitState::HalfOpen => "half_open",
        }
    }

    pub fn to_int(&self) -> u8 {
        match self {
            CircuitState::Closed => STATE_CLOSED,
            CircuitState::Open => STATE_OPEN,
            CircuitState::HalfOpen => STATE_HALF_OPEN,
        }
    }

    fn from_int(v: u8) -> Self {
        match v {
            STATE_CLOSED => CircuitState::Closed,
            STATE_OPEN => CircuitState::Open,
            STATE_HALF_OPEN => CircuitState::HalfOpen,
            // 原子值理论上只可能是上述三个常量；遇到异常值时按 Closed 处理，避免永久阻断流量。
            _ => CircuitState::Closed,
        }
    }
}

/// 返回进程启动后经过的单调时钟毫秒数。
///
/// 这里不用系统墙上时钟，避免系统时间回拨或校时导致冷却时长计算异常；
/// 转换为 `u64` 是为了能通过原子变量无锁保存时间点。
#[inline]
fn now_ms() -> u64 {
    // 所有熔断器共享同一个进程内时间原点，确保不同时间戳可直接相减。
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_millis() as u64
}

/// 面向 Worker 请求的无锁熔断器。
///
/// 路由选择会频繁调用 `can_execute`，因此状态、计数器和时间戳均使用原子变量，
/// 避免 `RwLock` 在高并发热路径上的竞争。状态转换使用 CAS 或原子交换完成。
///
/// 典型状态流转：
/// 1. `Closed` 下连续失败达到 `failure_threshold`，转为 `Open`；
/// 2. `Open` 持续 `timeout_duration` 后，下一次查询状态时惰性转为 `HalfOpen`；
/// 3. `HalfOpen` 下连续成功达到 `success_threshold`，转回 `Closed`；
/// 4. `HalfOpen` 下任意一次失败，立即重新转为 `Open` 并开始新一轮冷却。
#[derive(Debug)]
pub struct CircuitBreaker {
    /// 当前状态：0=`Closed`、1=`Open`、2=`HalfOpen`。
    state: AtomicU8,
    /// 当前连续失败次数；记录成功或进入 `HalfOpen`/`Closed` 时清零。
    consecutive_failures: AtomicU32,
    /// 当前连续成功次数；记录失败或发生状态转换时清零。
    consecutive_successes: AtomicU32,
    /// 生命周期内累计失败次数，仅用于统计，不参与状态判断。
    total_failures: AtomicU64,
    /// 生命周期内累计成功次数，仅用于统计，不参与状态判断。
    total_successes: AtomicU64,
    /// 最近一次失败相对于进程时间原点的毫秒数；0 表示尚无失败。
    last_failure_time_ms: AtomicU64,
    /// 最近一次状态变化相对于进程时间原点的毫秒数，用于计算 Open 冷却期。
    last_state_change_ms: AtomicU64,
    /// 状态转换阈值与冷却时长配置。
    config: CircuitBreakerConfig,
    /// 指标标签，通常用于区分不同 Worker。
    metric_label: String,
}

impl CircuitBreaker {
    /// 使用默认配置创建熔断器，初始状态为 `Closed`。
    pub fn new() -> Self {
        Self::with_config_and_label(CircuitBreakerConfig::default(), String::new())
    }

    /// 使用指定配置和指标标签创建熔断器，并发布初始 `Closed` 状态指标。
    pub fn with_config_and_label(config: CircuitBreakerConfig, metric_label: String) -> Self {
        let init_state = CircuitState::Closed;
        Metrics::set_worker_cb_state(&metric_label, init_state.to_int());
        Self {
            state: AtomicU8::new(STATE_CLOSED),
            consecutive_failures: AtomicU32::new(0),
            consecutive_successes: AtomicU32::new(0),
            total_failures: AtomicU64::new(0),
            total_successes: AtomicU64::new(0),
            last_failure_time_ms: AtomicU64::new(0),
            last_state_change_ms: AtomicU64::new(now_ms()),
            config,
            metric_label,
        }
    }

    /// 返回该熔断器上报监控指标时使用的标签。
    pub fn metric_label(&self) -> &str {
        &self.metric_label
    }

    /// 判断当前是否允许向 Worker 发送请求。
    ///
    /// 这是路由选择的无锁热路径。调用 `state()` 时也会顺便检查 `Open`
    /// 冷却期是否结束，因此该方法可能触发 `Open -> HalfOpen` 状态转换。
    #[inline]
    pub fn can_execute(&self) -> bool {
        let state = self.state();
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => false,
            CircuitState::HalfOpen => true,
        }
    }

    /// 返回当前状态；若 `Open` 冷却已到期，会先惰性切换到 `HalfOpen`。
    #[inline]
    pub fn state(&self) -> CircuitState {
        self.check_and_update_state_returning()
    }

    /// 检查冷却期限并返回最新状态，全程无锁。
    #[inline]
    fn check_and_update_state_returning(&self) -> CircuitState {
        let current_state_int = self.state.load(Ordering::Acquire);
        let current_state = CircuitState::from_int(current_state_int);

        if current_state == CircuitState::Open {
            let last_change_ms = self.last_state_change_ms.load(Ordering::Acquire);
            let elapsed_ms = now_ms().saturating_sub(last_change_ms);
            let timeout_ms = self.config.timeout_duration.as_millis() as u64;

            if elapsed_ms >= timeout_ms {
                // 多线程可能同时发现冷却到期；通过 CAS 保证只有一个线程完成状态转换。
                if self
                    .state
                    .compare_exchange(
                        STATE_OPEN,
                        STATE_HALF_OPEN,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    self.last_state_change_ms.store(now_ms(), Ordering::Release);
                    self.consecutive_failures.store(0, Ordering::Release);
                    self.consecutive_successes.store(0, Ordering::Release);

                    info!("Circuit breaker state transition: open -> half_open");
                    Metrics::record_worker_cb_transition(&self.metric_label, "open", "half_open");
                    Metrics::set_worker_cb_state(&self.metric_label, STATE_HALF_OPEN);
                    self.publish_gauge_metrics();
                    return CircuitState::HalfOpen;
                }
                // CAS 失败说明其他线程已修改状态，重新读取其最终结果。
                return CircuitState::from_int(self.state.load(Ordering::Acquire));
            }
        }
        current_state
    }

    /// 记录一次真实 Worker 请求的结果，并同步更新熔断状态和监控指标。
    pub fn record_outcome(&self, success: bool) {
        if success {
            self.record_success();
        } else {
            self.record_failure();
        }

        let outcome_str = if success { "success" } else { "failure" };
        Metrics::record_worker_cb_outcome(&self.metric_label, outcome_str);
        self.publish_gauge_metrics();
    }

    /// 记录一次成功。
    ///
    /// 成功会清空连续失败计数；若当前为 `HalfOpen`，连续成功达到
    /// `success_threshold` 后关闭熔断器。`Closed` 下的成功只更新计数。
    pub fn record_success(&self) {
        self.total_successes.fetch_add(1, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Release);
        let successes = self.consecutive_successes.fetch_add(1, Ordering::AcqRel) + 1;

        let current_state = CircuitState::from_int(self.state.load(Ordering::Acquire));

        match current_state {
            CircuitState::HalfOpen => {
                if successes >= self.config.success_threshold {
                    self.transition_to(CircuitState::Closed);
                }
            }
            CircuitState::Closed => {}
            CircuitState::Open => {
                tracing::warn!("Success recorded while circuit is open");
            }
        }
    }

    /// 记录一次失败。
    ///
    /// 失败会清空连续成功计数并更新时间戳：
    /// - `Closed`：连续失败达到阈值后进入 `Open`；
    /// - `HalfOpen`：一次失败就立即回到 `Open`，重新开始冷却；
    /// - `Open`：只更新统计，不重复执行状态转换。
    pub fn record_failure(&self) {
        self.total_failures.fetch_add(1, Ordering::Relaxed);
        self.consecutive_successes.store(0, Ordering::Release);
        let failures = self.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1;

        // 最近失败时间用于统计展示；Open 冷却本身从 last_state_change_ms 开始计算。
        self.last_failure_time_ms.store(now_ms(), Ordering::Release);

        let current_state = CircuitState::from_int(self.state.load(Ordering::Acquire));

        match current_state {
            CircuitState::Closed => {
                if failures >= self.config.failure_threshold {
                    self.transition_to(CircuitState::Open);
                }
            }
            CircuitState::HalfOpen => {
                self.transition_to(CircuitState::Open);
            }
            CircuitState::Open => {}
        }
    }

    /// 原子切换到目标状态，并重置与新状态不兼容的连续计数。
    ///
    /// `swap` 返回旧状态；只有状态确实发生变化时才更新时间戳、日志和指标，
    /// 避免多个并发请求重复发布同一状态转换。
    fn transition_to(&self, new_state: CircuitState) {
        let new_state_int = new_state.to_int();
        let old_state_int = self.state.swap(new_state_int, Ordering::AcqRel);
        let old_state = CircuitState::from_int(old_state_int);

        if old_state != new_state {
            self.last_state_change_ms.store(now_ms(), Ordering::Release);

            match new_state {
                CircuitState::Closed => {
                    self.consecutive_failures.store(0, Ordering::Release);
                    self.consecutive_successes.store(0, Ordering::Release);
                }
                CircuitState::Open => {
                    self.consecutive_successes.store(0, Ordering::Release);
                }
                CircuitState::HalfOpen => {
                    self.consecutive_failures.store(0, Ordering::Release);
                    self.consecutive_successes.store(0, Ordering::Release);
                }
            }

            let from = old_state.as_str();
            let to = new_state.as_str();
            info!("Circuit breaker state transition: {} -> {}", from, to);
            Metrics::record_worker_cb_transition(&self.metric_label, from, to);
            Metrics::set_worker_cb_state(&self.metric_label, new_state.to_int());
            self.publish_gauge_metrics();
        }
    }

    /// 返回当前连续失败次数。
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures.load(Ordering::Acquire)
    }

    /// 返回当前连续成功次数。
    pub fn consecutive_successes(&self) -> u32 {
        self.consecutive_successes.load(Ordering::Acquire)
    }

    /// 返回熔断器生命周期内累计失败次数。
    pub fn total_failures(&self) -> u64 {
        self.total_failures.load(Ordering::Relaxed)
    }

    /// 返回熔断器生命周期内累计成功次数。
    pub fn total_successes(&self) -> u64 {
        self.total_successes.load(Ordering::Relaxed)
    }

    /// 返回距最近一次失败经过的时间；从未失败时返回 `None`。
    pub fn time_since_last_failure(&self) -> Option<Duration> {
        let last_ms = self.last_failure_time_ms.load(Ordering::Acquire);
        if last_ms == 0 {
            None
        } else {
            let elapsed_ms = now_ms().saturating_sub(last_ms);
            Some(Duration::from_millis(elapsed_ms))
        }
    }

    /// 返回距最近一次状态变化经过的时间。
    pub fn time_since_last_state_change(&self) -> Duration {
        let last_ms = self.last_state_change_ms.load(Ordering::Acquire);
        let elapsed_ms = now_ms().saturating_sub(last_ms);
        Duration::from_millis(elapsed_ms)
    }

    /// 判断是否处于 `HalfOpen`；调用时可能触发冷却到期后的惰性状态转换。
    pub fn is_half_open(&self) -> bool {
        self.state() == CircuitState::HalfOpen
    }

    /// 记录一次探测成功；仅在 `HalfOpen` 状态下计入恢复判断。
    pub fn record_test_success(&self) {
        if self.is_half_open() {
            self.record_success();
        }
    }

    /// 记录一次探测失败；仅在 `HalfOpen` 状态下重新打开熔断器。
    pub fn record_test_failure(&self) {
        if self.is_half_open() {
            self.record_failure();
        }
    }

    /// 手动重置为 `Closed`，并清空连续成功和失败计数。累计计数不会清零。
    pub fn reset(&self) {
        self.transition_to(CircuitState::Closed);
        self.consecutive_failures.store(0, Ordering::Release);
        self.consecutive_successes.store(0, Ordering::Release);
        self.publish_gauge_metrics();
    }

    /// 手动强制进入 `Open`，从此刻开始计算新的冷却周期。
    pub fn force_open(&self) {
        self.transition_to(CircuitState::Open);
    }

    /// 获取当前状态及累计/连续计数的统计快照。
    pub fn stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            state: self.state(),
            consecutive_failures: self.consecutive_failures(),
            consecutive_successes: self.consecutive_successes(),
            total_failures: self.total_failures(),
            total_successes: self.total_successes(),
            time_since_last_failure: self.time_since_last_failure(),
            time_since_last_state_change: self.time_since_last_state_change(),
        }
    }

    fn publish_gauge_metrics(&self) {
        Metrics::set_worker_cb_consecutive_failures(
            &self.metric_label,
            self.consecutive_failures(),
        );
        Metrics::set_worker_cb_consecutive_successes(
            &self.metric_label,
            self.consecutive_successes(),
        );
    }
}

impl Clone for CircuitBreaker {
    fn clone(&self) -> Self {
        Self {
            state: AtomicU8::new(self.state.load(Ordering::Acquire)),
            consecutive_failures: AtomicU32::new(self.consecutive_failures.load(Ordering::Acquire)),
            consecutive_successes: AtomicU32::new(
                self.consecutive_successes.load(Ordering::Acquire),
            ),
            total_failures: AtomicU64::new(self.total_failures.load(Ordering::Relaxed)),
            total_successes: AtomicU64::new(self.total_successes.load(Ordering::Relaxed)),
            last_failure_time_ms: AtomicU64::new(self.last_failure_time_ms.load(Ordering::Acquire)),
            last_state_change_ms: AtomicU64::new(self.last_state_change_ms.load(Ordering::Acquire)),
            config: self.config.clone(),
            metric_label: self.metric_label.clone(),
        }
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

/// 熔断器统计快照。
///
/// `total_*` 仅用于观测，`consecutive_*` 才参与当前状态转换判断。
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    pub state: CircuitState,
    pub consecutive_failures: u32,
    pub consecutive_successes: u32,
    pub total_failures: u64,
    pub total_successes: u64,
    pub time_since_last_failure: Option<Duration>,
    pub time_since_last_state_change: Duration,
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
    fn test_circuit_breaker_initial_state() {
        let cb = CircuitBreaker::new();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.can_execute());
        assert_eq!(cb.consecutive_failures(), 0);
        assert_eq!(cb.consecutive_successes(), 0);
    }

    #[test]
    fn test_circuit_opens_on_threshold() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config_and_label(config, String::new());

        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();

        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.can_execute());
        assert_eq!(cb.consecutive_failures(), 3);
    }

    #[test]
    fn test_circuit_half_open_after_timeout() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout_duration: Duration::from_millis(100),
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config_and_label(config, String::new());

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        thread::sleep(Duration::from_millis(150));

        assert_eq!(cb.state(), CircuitState::HalfOpen);
        assert!(cb.can_execute());
    }

    #[test]
    fn test_circuit_closes_on_success_threshold() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout_duration: Duration::from_millis(50),
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config_and_label(config, String::new());

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        thread::sleep(Duration::from_millis(100));
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();

        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.can_execute());
    }

    #[test]
    fn test_circuit_reopens_on_half_open_failure() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout_duration: Duration::from_millis(50),
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config_and_label(config, String::new());

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        thread::sleep(Duration::from_millis(100));
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_failure();

        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.can_execute());
    }

    #[test]
    fn test_success_resets_failure_count() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config_and_label(config, String::new());

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.consecutive_failures(), 2);

        cb.record_success();
        assert_eq!(cb.consecutive_failures(), 0);
        assert_eq!(cb.consecutive_successes(), 1);

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_manual_reset() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config_and_label(config, String::new());

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.consecutive_failures(), 0);
        assert_eq!(cb.consecutive_successes(), 0);
    }

    #[test]
    fn test_force_open() {
        let cb = CircuitBreaker::new();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.force_open();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.can_execute());
    }

    #[test]
    fn test_stats() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config_and_label(config, String::new());

        cb.record_success();
        cb.record_failure();
        cb.record_failure();

        let stats = cb.stats();
        assert_eq!(stats.state, CircuitState::Open);
        assert_eq!(stats.consecutive_failures, 2);
        assert_eq!(stats.consecutive_successes, 0);
        assert_eq!(stats.total_failures, 2);
        assert_eq!(stats.total_successes, 1);
    }

    #[test]
    fn test_clone() {
        let cb1 = CircuitBreaker::new();
        cb1.record_failure();

        let cb2 = cb1.clone();
        assert_eq!(cb2.consecutive_failures(), 1);

        cb1.record_failure();
        assert_eq!(cb1.consecutive_failures(), 2);
        assert_eq!(cb2.consecutive_failures(), 1); // cb2 is unchanged
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;

        let cb = Arc::new(CircuitBreaker::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let cb_clone = Arc::clone(&cb);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    cb_clone.record_failure();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(cb.total_failures(), 1000);
    }
}
