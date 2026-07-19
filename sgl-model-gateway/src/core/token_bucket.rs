use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use tokio::sync::Notify;
use tracing::{debug, trace};

/// 用于限流的令牌桶(Token Bucket)。
///
/// 本实现提供:
/// - 基于可配置补充速率的平滑限流
/// - 突发(burst)容量处理（桶容量允许短时间内集中消耗）
/// - 通过 `Notify` 为等待中的请求提供公平排队
/// - 为 Drop 处理器提供同步归还令牌的能力（见 `return_tokens_sync`）
///
/// 使用 `parking_lot::Mutex` 以兼容同步上锁（无需 async）。
#[derive(Clone)]
pub struct TokenBucket {
    /// 受锁保护的内部可变状态（当前令牌数与上次补充时间）。
    inner: Arc<Mutex<TokenBucketInner>>,
    /// 当令牌被归还时，用于唤醒等待中的获取者。
    notify: Arc<Notify>,
    /// 桶的最大容量（即最大可累积/突发令牌数）。
    capacity: f64,
    /// 补充速率（每秒新增令牌数）；为 0 时退化为纯并发限制。
    refill_rate: f64, // 每秒令牌数
}

/// 令牌桶的内部可变状态（需在锁保护下访问）。
struct TokenBucketInner {
    /// 当前可用令牌数（用 f64 以支持按时间比例精确补充）。
    tokens: f64,
    /// 上一次执行补充计算的时间点，用于按流逝时间惰性补充。
    last_refill: Instant,
}

impl TokenBucket {
    /// 创建一个新的令牌桶
    ///
    /// # 参数
    /// * `capacity` - 最大令牌数（突发容量）
    /// * `refill_rate` - 每秒补充的令牌数（为 0 表示纯并发限制）
    pub fn new(capacity: usize, refill_rate: usize) -> Self {
        let capacity = capacity as f64;
        // 允许 refill_rate=0 以实现纯并发限制（类似信号量 semaphore 的行为）
        // 当 refill_rate=0 时，令牌只能通过 return_tokens() 归还
        let refill_rate = refill_rate as f64;

        Self {
            inner: Arc::new(Mutex::new(TokenBucketInner {
                tokens: capacity,
                last_refill: Instant::now(),
            })),
            notify: Arc::new(Notify::new()),
            capacity,
            refill_rate,
        }
    }

    /// 尝试立即获取令牌。
    ///
    /// 获取成功返回 `Ok(())`；令牌不足时返回 `Err(())`。
    pub async fn try_acquire(&self, tokens: f64) -> Result<(), ()> {
        self.try_acquire_sync(tokens)
    }

    /// try_acquire 的同步版本（供内部使用）。
    fn try_acquire_sync(&self, tokens: f64) -> Result<(), ()> {
        let mut inner = self.inner.lock();

        // 根据距上次补充经过的时间，按补充速率惰性计算应新增的令牌
        let now = Instant::now();
        let elapsed = now.duration_since(inner.last_refill).as_secs_f64();
        let refill_amount = elapsed * self.refill_rate;

        // 补充后不得超过桶容量，并更新上次补充时间
        inner.tokens = (inner.tokens + refill_amount).min(self.capacity);
        inner.last_refill = now;

        trace!(
            "Token bucket: {} tokens available, requesting {}",
            inner.tokens,
            tokens
        );

        // 令牌足够则扣除并放行，否则获取失败
        if inner.tokens >= tokens {
            inner.tokens -= tokens;
            debug!(
                "Token bucket: acquired {} tokens, {} remaining",
                tokens, inner.tokens
            );
            Ok(())
        } else {
            Err(())
        }
    }

    /// 获取令牌，必要时进行等待。
    ///
    /// 当 `refill_rate=0` 时，将无限期等待令牌通过 `return_tokens()` 归还。
    /// 可使用 `acquire_timeout()` 设置合适的超时。
    pub async fn acquire(&self, tokens: f64) -> Result<(), tokio::time::error::Elapsed> {
        // 快路径:若当前令牌充足则直接成功返回
        if self.try_acquire(tokens).await.is_ok() {
            return Ok(());
        }

        // 当 refill_rate=0（纯并发限制）时，令牌只能通过 return_tokens() 归还，
        // 因此仅等待 notify 信号。
        if self.refill_rate == 0.0 {
            debug!(
                "Token bucket: waiting indefinitely for {} tokens (refill_rate=0)",
                tokens
            );

            loop {
                // 等待来自 return_tokens() 的唤醒信号
                self.notify.notified().await;

                if self.try_acquire(tokens).await.is_ok() {
                    return Ok(());
                }
            }
        }

        // 根据还需多少令牌与补充速率，估算需要等待的时长
        let wait_time = {
            let inner = self.inner.lock();
            let tokens_needed = tokens - inner.tokens;
            let wait_secs = (tokens_needed / self.refill_rate).max(0.0);
            Duration::from_secs_f64(wait_secs)
        };

        debug!(
            "Token bucket: waiting {:?} for {} tokens",
            wait_time, tokens
        );

        // 在估算的等待时长内循环重试获取:
        // 要么被 return_tokens() 的信号唤醒，要么每 10ms 轮询一次（兼顾补充到位的情况）
        tokio::time::timeout(wait_time, async {
            loop {
                if self.try_acquire(tokens).await.is_ok() {
                    return;
                }
                tokio::select! {
                    _ = self.notify.notified() => {},
                    _ = tokio::time::sleep(Duration::from_millis(10)) => {},
                }
            }
        })
        .await?;

        Ok(())
    }

    /// 以自定义超时时长获取令牌。
    pub async fn acquire_timeout(
        &self,
        tokens: f64,
        timeout: Duration,
    ) -> Result<(), tokio::time::error::Elapsed> {
        tokio::time::timeout(timeout, self.acquire(tokens)).await?
    }

    /// 将令牌归还到桶中（同步版本）。
    ///
    /// 可安全地在同步上下文中调用（如 Drop 处理器）。
    /// 使用 `parking_lot::Mutex`，不会无限期阻塞。
    pub fn return_tokens_sync(&self, tokens: f64) {
        {
            let mut inner = self.inner.lock();
            // 归还后令牌数同样不得超过桶容量
            inner.tokens = (inner.tokens + tokens).min(self.capacity);
            debug!(
                "Token bucket: returned {} tokens, {} available",
                tokens, inner.tokens
            );
        } // 在发出 notify 前先释放锁，避免被唤醒者立即争锁
        self.notify.notify_waiters();
    }

    /// 将令牌归还到桶中（异步版本，以保持 API 兼容）。
    pub async fn return_tokens(&self, tokens: f64) {
        self.return_tokens_sync(tokens);
    }

    /// 获取当前可用令牌数（用于监控）。
    pub async fn available_tokens(&self) -> f64 {
        let mut inner = self.inner.lock();

        // 读取前先按流逝时间完成一次惰性补充，以返回最新的可用令牌数
        let now = Instant::now();
        let elapsed = now.duration_since(inner.last_refill).as_secs_f64();
        let refill_amount = elapsed * self.refill_rate;

        inner.tokens = (inner.tokens + refill_amount).min(self.capacity);
        inner.last_refill = now;

        inner.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_bucket_basic() {
        let bucket = TokenBucket::new(10, 5);

        assert!(bucket.try_acquire(5.0).await.is_ok());
        assert!(bucket.try_acquire(5.0).await.is_ok());

        assert!(bucket.try_acquire(1.0).await.is_err());

        tokio::time::sleep(Duration::from_millis(300)).await;

        assert!(bucket.try_acquire(1.0).await.is_ok());
    }

    #[tokio::test]
    async fn test_token_bucket_refill() {
        let bucket = TokenBucket::new(10, 10);

        assert!(bucket.try_acquire(10.0).await.is_ok());

        tokio::time::sleep(Duration::from_millis(500)).await;

        let available = bucket.available_tokens().await;
        assert!((4.0..=6.0).contains(&available));
    }

    #[tokio::test]
    async fn test_token_bucket_zero_refill_rate() {
        // With refill_rate=0, tokens should only come back via return_tokens()
        let bucket = TokenBucket::new(2, 0);

        // Acquire both tokens
        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_ok());

        // No more tokens available
        assert!(bucket.try_acquire(1.0).await.is_err());

        // Wait - should NOT refill automatically
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(bucket.try_acquire(1.0).await.is_err());

        // Return a token - now we should be able to acquire
        bucket.return_tokens(1.0).await;
        assert!(bucket.try_acquire(1.0).await.is_ok());

        // No more tokens again
        assert!(bucket.try_acquire(1.0).await.is_err());
    }

    #[tokio::test]
    async fn test_token_bucket_zero_refill_with_notify() {
        // Test that acquire wakes up when tokens are returned
        let bucket = Arc::new(TokenBucket::new(1, 0));

        // Acquire the only token
        assert!(bucket.try_acquire(1.0).await.is_ok());

        let bucket_clone = bucket.clone();

        // Spawn a task that will return the token after a delay
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            bucket_clone.return_tokens(1.0).await;
        });

        // This should wait and then succeed when token is returned
        let result = bucket.acquire_timeout(1.0, Duration::from_secs(1)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_return_tokens_sync() {
        // Test that sync return works correctly
        let bucket = TokenBucket::new(2, 0);

        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_err());

        // Use sync return
        bucket.return_tokens_sync(1.0);
        assert!(bucket.try_acquire(1.0).await.is_ok());
    }
}
