//! Retry policy with exponential backoff and jitter.
//!
//! Provides configurable retry behavior for transient failures with:
//! - Exponential backoff between retries
//! - Jitter to avoid thundering herd problem
//! - Configurable retry conditions
//! - Maximum delay capping

use std::future::Future;
use std::time::Duration;

use rand::Rng;
use tokio::time::sleep;
use tracing::{debug, warn};

use super::{NagualError, RetryError};

/// Condition for determining if an operation should be retried.
pub trait RetryCondition<E>: Send + Sync {
    /// Returns true if the error should trigger a retry.
    fn should_retry(&self, error: &E, attempt: u32) -> bool;
}

/// Default retry condition that checks if errors are retryable.
#[derive(Clone, Debug, Default)]
pub struct DefaultRetryCondition;

impl RetryCondition<NagualError> for DefaultRetryCondition {
    fn should_retry(&self, error: &NagualError, _attempt: u32) -> bool {
        error.is_retryable()
    }
}

/// Always retry condition (useful for testing or specific scenarios).
#[derive(Clone, Debug, Default)]
pub struct AlwaysRetry;

impl<E> RetryCondition<E> for AlwaysRetry {
    fn should_retry(&self, _error: &E, _attempt: u32) -> bool {
        true
    }
}

/// Never retry condition.
#[derive(Clone, Debug, Default)]
pub struct NeverRetry;

impl<E> RetryCondition<E> for NeverRetry {
    fn should_retry(&self, _error: &E, _attempt: u32) -> bool {
        false
    }
}

/// Custom retry condition using a closure.
pub struct CustomRetryCondition<F>(pub F);

impl<E, F> RetryCondition<E> for CustomRetryCondition<F>
where
    F: Fn(&E, u32) -> bool + Send + Sync,
{
    fn should_retry(&self, error: &E, attempt: u32) -> bool {
        (self.0)(error, attempt)
    }
}

/// Retry policy configuration with exponential backoff.
#[derive(Clone, Debug)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (not including the initial attempt).
    pub max_retries: u32,
    /// Base delay between retries (will be multiplied by backoff factor).
    pub base_delay: Duration,
    /// Maximum delay between retries (caps the exponential growth).
    pub max_delay: Duration,
    /// Backoff multiplier applied to base_delay for each retry.
    pub backoff_factor: f64,
    /// Jitter factor (0.0 to 1.0) - adds randomness to avoid thundering herd.
    pub jitter_factor: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_factor: 2.0,
            jitter_factor: 0.25,
        }
    }
}

impl RetryPolicy {
    /// Create a new retry policy with custom settings.
    pub fn new(
        max_retries: u32,
        base_delay: Duration,
        max_delay: Duration,
    ) -> Self {
        Self {
            max_retries,
            base_delay,
            max_delay,
            backoff_factor: 2.0,
            jitter_factor: 0.25,
        }
    }

    /// Create a policy optimized for fast failures.
    pub fn fast() -> Self {
        Self {
            max_retries: 2,
            base_delay: Duration::from_millis(50),
            max_delay: Duration::from_millis(500),
            backoff_factor: 2.0,
            jitter_factor: 0.1,
        }
    }

    /// Create a policy for slow, persistent retries.
    pub fn persistent() -> Self {
        Self {
            max_retries: 10,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(300),
            backoff_factor: 2.0,
            jitter_factor: 0.25,
        }
    }

    /// Create a policy for database operations.
    pub fn database() -> Self {
        Self {
            max_retries: 5,
            base_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(10),
            backoff_factor: 2.0,
            jitter_factor: 0.2,
        }
    }

    /// Create a policy for network/HTTP operations.
    pub fn network() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_factor: 2.0,
            jitter_factor: 0.3,
        }
    }

    /// Set the maximum number of retries.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set the base delay.
    pub fn with_base_delay(mut self, base_delay: Duration) -> Self {
        self.base_delay = base_delay;
        self
    }

    /// Set the maximum delay.
    pub fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.max_delay = max_delay;
        self
    }

    /// Set the backoff factor.
    pub fn with_backoff_factor(mut self, factor: f64) -> Self {
        self.backoff_factor = factor;
        self
    }

    /// Set the jitter factor (0.0 to 1.0).
    pub fn with_jitter_factor(mut self, factor: f64) -> Self {
        self.jitter_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Calculate the delay for a given attempt number (0-indexed).
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        // Calculate exponential backoff
        let exponential_delay = self.base_delay.as_secs_f64()
            * self.backoff_factor.powi(attempt as i32);

        // Cap at max_delay
        let capped_delay = exponential_delay.min(self.max_delay.as_secs_f64());

        // Apply jitter
        let jitter_range = capped_delay * self.jitter_factor;
        let jitter = rand::thread_rng().gen_range(-jitter_range..=jitter_range);
        let final_delay = (capped_delay + jitter).max(0.0);

        Duration::from_secs_f64(final_delay)
    }

    /// Execute an async operation with retry logic.
    pub async fn with_retry<F, Fut, T, E, C>(
        &self,
        operation_name: &str,
        condition: &C,
        mut operation: F,
    ) -> Result<T, RetryError>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::fmt::Display,
        C: RetryCondition<E>,
    {
        let mut last_error: Option<E> = None;
        let mut total_delay_ms: u64 = 0;

        for attempt in 0..=self.max_retries {
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        debug!(
                            operation = operation_name,
                            attempt = attempt,
                            "Operation succeeded after retry"
                        );
                    }
                    return Ok(result);
                }
                Err(error) => {
                    let is_last_attempt = attempt == self.max_retries;

                    if is_last_attempt || !condition.should_retry(&error, attempt) {
                        if is_last_attempt {
                            warn!(
                                operation = operation_name,
                                attempt = attempt,
                                max_retries = self.max_retries,
                                error = %error,
                                "Max retries exceeded"
                            );
                            return Err(RetryError::MaxRetriesExceeded {
                                max_retries: self.max_retries,
                                total_delay_ms,
                                last_error: error.to_string(),
                            });
                        } else {
                            debug!(
                                operation = operation_name,
                                attempt = attempt,
                                error = %error,
                                "Error is not retryable"
                            );
                            return Err(RetryError::NotRetryable {
                                reason: error.to_string(),
                            });
                        }
                    }

                    let delay = self.calculate_delay(attempt);
                    total_delay_ms += delay.as_millis() as u64;

                    debug!(
                        operation = operation_name,
                        attempt = attempt,
                        next_attempt = attempt + 1,
                        delay_ms = delay.as_millis(),
                        error = %error,
                        "Retrying after delay"
                    );

                    last_error = Some(error);
                    sleep(delay).await;
                }
            }
        }

        Err(RetryError::MaxRetriesExceeded {
            max_retries: self.max_retries,
            total_delay_ms,
            last_error: last_error.map(|e| e.to_string()).unwrap_or_default(),
        })
    }
}

/// Convenience function to retry an operation with default settings.
pub async fn with_retry<F, Fut, T, E>(
    operation_name: &str,
    operation: F,
) -> Result<T, RetryError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let policy = RetryPolicy::default();
    let condition = AlwaysRetry;
    policy.with_retry(operation_name, &condition, operation).await
}

/// Convenience function to retry an operation with a custom policy.
pub async fn with_retry_policy<F, Fut, T, E>(
    operation_name: &str,
    policy: &RetryPolicy,
    operation: F,
) -> Result<T, RetryError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let condition = AlwaysRetry;
    policy.with_retry(operation_name, &condition, operation).await
}

/// Retry with NagualError and automatic retry condition detection.
pub async fn with_retry_nagual<F, Fut, T>(
    operation_name: &str,
    policy: &RetryPolicy,
    operation: F,
) -> Result<T, RetryError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, NagualError>>,
{
    let condition = DefaultRetryCondition;
    policy.with_retry(operation_name, &condition, operation).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_default_policy() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
        assert_eq!(policy.base_delay, Duration::from_millis(100));
        assert_eq!(policy.max_delay, Duration::from_secs(30));
    }

    #[test]
    fn test_calculate_delay_exponential() {
        let policy = RetryPolicy::default().with_jitter_factor(0.0);

        let delay0 = policy.calculate_delay(0);
        let delay1 = policy.calculate_delay(1);
        let delay2 = policy.calculate_delay(2);

        // With no jitter, delays should double
        assert_eq!(delay0, Duration::from_millis(100));
        assert_eq!(delay1, Duration::from_millis(200));
        assert_eq!(delay2, Duration::from_millis(400));
    }

    #[test]
    fn test_calculate_delay_capped() {
        let policy = RetryPolicy::new(
            10,
            Duration::from_secs(1),
            Duration::from_secs(5),
        ).with_jitter_factor(0.0);

        // After many retries, delay should be capped
        let delay = policy.calculate_delay(10);
        assert_eq!(delay, Duration::from_secs(5));
    }

    #[test]
    fn test_calculate_delay_with_jitter() {
        let policy = RetryPolicy::default().with_jitter_factor(0.5);

        // Run multiple times to verify jitter adds variability
        let delays: Vec<Duration> = (0..10)
            .map(|_| policy.calculate_delay(1))
            .collect();

        // Not all delays should be the same (with 50% jitter)
        let all_same = delays.windows(2).all(|w| w[0] == w[1]);
        assert!(!all_same, "Jitter should cause variation in delays");
    }

    #[tokio::test]
    async fn test_retry_success_first_attempt() {
        let policy = RetryPolicy::fast();
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let result: Result<i32, RetryError> = policy
            .with_retry("test_op", &AlwaysRetry, || {
                let attempts = attempts_clone.clone();
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, &str>(42)
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let policy = RetryPolicy::fast();
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let result: Result<i32, RetryError> = policy
            .with_retry("test_op", &AlwaysRetry, || {
                let attempts = attempts_clone.clone();
                async move {
                    let count = attempts.fetch_add(1, Ordering::SeqCst);
                    if count < 2 {
                        Err("transient error")
                    } else {
                        Ok(42)
                    }
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_max_retries_exceeded() {
        let policy = RetryPolicy::fast().with_max_retries(2);
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let result: Result<i32, RetryError> = policy
            .with_retry("test_op", &AlwaysRetry, || {
                let attempts = attempts_clone.clone();
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    Err::<i32, _>("persistent error")
                }
            })
            .await;

        assert!(matches!(result, Err(RetryError::MaxRetriesExceeded { .. })));
        assert_eq!(attempts.load(Ordering::SeqCst), 3); // Initial + 2 retries
    }

    #[tokio::test]
    async fn test_retry_not_retryable() {
        let policy = RetryPolicy::fast();
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let result: Result<i32, RetryError> = policy
            .with_retry("test_op", &NeverRetry, || {
                let attempts = attempts_clone.clone();
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    Err::<i32, _>("not retryable")
                }
            })
            .await;

        assert!(matches!(result, Err(RetryError::NotRetryable { .. })));
        assert_eq!(attempts.load(Ordering::SeqCst), 1); // Only initial attempt
    }

    #[tokio::test]
    async fn test_custom_retry_condition() {
        let policy = RetryPolicy::fast().with_max_retries(5);
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        // Only retry on "retry" errors, not "fatal" errors
        let condition = CustomRetryCondition(|error: &&str, _| *error == "retry");

        let result: Result<i32, RetryError> = policy
            .with_retry("test_op", &condition, || {
                let attempts = attempts_clone.clone();
                async move {
                    let count = attempts.fetch_add(1, Ordering::SeqCst);
                    if count < 2 {
                        Err("retry")
                    } else {
                        Err("fatal")
                    }
                }
            })
            .await;

        assert!(matches!(result, Err(RetryError::NotRetryable { .. })));
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }
}
