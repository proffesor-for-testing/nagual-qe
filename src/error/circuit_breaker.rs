//! Circuit breaker pattern for fault tolerance.
//!
//! Implements the circuit breaker pattern with three states:
//! - Closed: Normal operation, requests flow through
//! - Open: Requests are rejected immediately (fail fast)
//! - HalfOpen: Limited requests are allowed to test recovery
//!
//! This helps prevent cascading failures and allows systems to recover gracefully.

use std::future::Future;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tracing::{debug, info, warn};

use super::CircuitBreakerError;

/// State of the circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests flow through normally.
    Closed,
    /// Circuit is open, requests are rejected immediately.
    Open,
    /// Circuit is half-open, limited requests are allowed to test recovery.
    HalfOpen,
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitState::Closed => write!(f, "CLOSED"),
            CircuitState::Open => write!(f, "OPEN"),
            CircuitState::HalfOpen => write!(f, "HALF_OPEN"),
        }
    }
}

/// Configuration for the circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Name of the service protected by this circuit breaker.
    pub service_name: String,
    /// Number of consecutive failures before opening the circuit.
    pub failure_threshold: u32,
    /// Number of consecutive successes in half-open state before closing.
    pub success_threshold: u32,
    /// Duration the circuit stays open before transitioning to half-open.
    pub reset_timeout: Duration,
    /// Maximum number of requests allowed in half-open state.
    pub half_open_max_requests: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            service_name: "default".to_string(),
            failure_threshold: 5,
            success_threshold: 3,
            reset_timeout: Duration::from_secs(30),
            half_open_max_requests: 3,
        }
    }
}

impl CircuitBreakerConfig {
    /// Create a new circuit breaker configuration.
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            ..Default::default()
        }
    }

    /// Set the failure threshold.
    pub fn with_failure_threshold(mut self, threshold: u32) -> Self {
        self.failure_threshold = threshold;
        self
    }

    /// Set the success threshold for half-open state.
    pub fn with_success_threshold(mut self, threshold: u32) -> Self {
        self.success_threshold = threshold;
        self
    }

    /// Set the reset timeout.
    pub fn with_reset_timeout(mut self, timeout: Duration) -> Self {
        self.reset_timeout = timeout;
        self
    }

    /// Set the maximum requests in half-open state.
    pub fn with_half_open_max_requests(mut self, max_requests: u32) -> Self {
        self.half_open_max_requests = max_requests;
        self
    }
}

/// Internal state tracking for the circuit breaker.
struct CircuitBreakerState {
    state: CircuitState,
    consecutive_failures: u32,
    consecutive_successes: u32,
    last_failure_time: Option<Instant>,
    half_open_requests: u32,
}

impl Default for CircuitBreakerState {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            consecutive_failures: 0,
            consecutive_successes: 0,
            last_failure_time: None,
            half_open_requests: 0,
        }
    }
}

/// Circuit breaker for protecting services from cascading failures.
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: RwLock<CircuitBreakerState>,
    // Metrics
    total_requests: AtomicU64,
    total_failures: AtomicU64,
    total_successes: AtomicU64,
    total_rejections: AtomicU64,
    state_transitions: AtomicU32,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given configuration.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        info!(
            service = %config.service_name,
            failure_threshold = config.failure_threshold,
            reset_timeout_ms = config.reset_timeout.as_millis(),
            "Circuit breaker created"
        );

        Self {
            config,
            state: RwLock::new(CircuitBreakerState::default()),
            total_requests: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
            total_successes: AtomicU64::new(0),
            total_rejections: AtomicU64::new(0),
            state_transitions: AtomicU32::new(0),
        }
    }

    /// Create a circuit breaker with default settings for a service.
    pub fn for_service(service_name: impl Into<String>) -> Self {
        Self::new(CircuitBreakerConfig::new(service_name))
    }

    /// Get the current state of the circuit breaker.
    pub fn state(&self) -> CircuitState {
        let mut state = self.state.write();
        self.maybe_transition_to_half_open(&mut state);
        state.state
    }

    /// Get the service name.
    pub fn service_name(&self) -> &str {
        &self.config.service_name
    }

    /// Check if the circuit is allowing requests.
    pub fn is_allowing_requests(&self) -> bool {
        let state = self.state();
        state != CircuitState::Open
    }

    /// Get circuit breaker metrics.
    pub fn metrics(&self) -> CircuitBreakerMetrics {
        let state = self.state.read();
        CircuitBreakerMetrics {
            state: state.state,
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_failures: self.total_failures.load(Ordering::Relaxed),
            total_successes: self.total_successes.load(Ordering::Relaxed),
            total_rejections: self.total_rejections.load(Ordering::Relaxed),
            consecutive_failures: state.consecutive_failures,
            consecutive_successes: state.consecutive_successes,
            state_transitions: self.state_transitions.load(Ordering::Relaxed),
        }
    }

    /// Execute an async operation protected by the circuit breaker.
    pub async fn call<F, Fut, T, E>(
        &self,
        operation: F,
    ) -> Result<T, CircuitBreakerError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        // Check if we can make a request
        if !self.can_execute()? {
            self.total_rejections.fetch_add(1, Ordering::Relaxed);
            let state = self.state.read();
            let retry_after_ms = state
                .last_failure_time
                .map(|t| {
                    let elapsed = t.elapsed();
                    if elapsed < self.config.reset_timeout {
                        (self.config.reset_timeout - elapsed).as_millis() as u64
                    } else {
                        0
                    }
                })
                .unwrap_or(0);

            return Err(CircuitBreakerError::Open {
                service: self.config.service_name.clone(),
                retry_after_ms,
            });
        }

        // Execute the operation
        match operation().await {
            Ok(result) => {
                self.record_success();
                Ok(result)
            }
            Err(error) => {
                warn!(
                    service = %self.config.service_name,
                    error = %error,
                    "Circuit breaker recorded failure"
                );
                self.record_failure();

                // Return the original error wrapped appropriately
                let state = self.state.read();
                if state.state == CircuitState::Open {
                    Err(CircuitBreakerError::Open {
                        service: self.config.service_name.clone(),
                        retry_after_ms: self.config.reset_timeout.as_millis() as u64,
                    })
                } else {
                    // HalfOpen or Closed with failures
                    Err(CircuitBreakerError::HalfOpen {
                        service: self.config.service_name.clone(),
                    })
                }
            }
        }
    }

    /// Check if we can execute a request.
    fn can_execute(&self) -> Result<bool, CircuitBreakerError> {
        let mut state = self.state.write();

        // Check for timeout-based transition to half-open
        self.maybe_transition_to_half_open(&mut state);

        match state.state {
            CircuitState::Closed => Ok(true),
            CircuitState::Open => Ok(false),
            CircuitState::HalfOpen => {
                if state.half_open_requests < self.config.half_open_max_requests {
                    state.half_open_requests += 1;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// Check if we should transition from Open to HalfOpen.
    fn maybe_transition_to_half_open(&self, state: &mut CircuitBreakerState) {
        if state.state == CircuitState::Open {
            if let Some(last_failure) = state.last_failure_time {
                if last_failure.elapsed() >= self.config.reset_timeout {
                    self.transition_to(state, CircuitState::HalfOpen);
                }
            }
        }
    }

    /// Record a successful operation.
    fn record_success(&self) {
        self.total_successes.fetch_add(1, Ordering::Relaxed);

        let mut state = self.state.write();
        state.consecutive_failures = 0;
        state.consecutive_successes += 1;

        match state.state {
            CircuitState::HalfOpen => {
                if state.consecutive_successes >= self.config.success_threshold {
                    self.transition_to(&mut state, CircuitState::Closed);
                }
            }
            CircuitState::Closed => {
                // Already closed, nothing to do
            }
            CircuitState::Open => {
                // Shouldn't happen, but handle gracefully
                debug!(
                    service = %self.config.service_name,
                    "Success recorded while circuit is open (unexpected)"
                );
            }
        }
    }

    /// Record a failed operation.
    fn record_failure(&self) {
        self.total_failures.fetch_add(1, Ordering::Relaxed);

        let mut state = self.state.write();
        state.consecutive_successes = 0;
        state.consecutive_failures += 1;
        state.last_failure_time = Some(Instant::now());

        match state.state {
            CircuitState::Closed => {
                if state.consecutive_failures >= self.config.failure_threshold {
                    self.transition_to(&mut state, CircuitState::Open);
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open state opens the circuit
                self.transition_to(&mut state, CircuitState::Open);
            }
            CircuitState::Open => {
                // Already open, nothing to do
            }
        }
    }

    /// Transition to a new state.
    fn transition_to(&self, state: &mut CircuitBreakerState, new_state: CircuitState) {
        let old_state = state.state;
        if old_state != new_state {
            info!(
                service = %self.config.service_name,
                from = %old_state,
                to = %new_state,
                consecutive_failures = state.consecutive_failures,
                consecutive_successes = state.consecutive_successes,
                "Circuit breaker state transition"
            );

            state.state = new_state;
            self.state_transitions.fetch_add(1, Ordering::Relaxed);

            // Reset counters on transition
            match new_state {
                CircuitState::Closed => {
                    state.consecutive_failures = 0;
                    state.consecutive_successes = 0;
                }
                CircuitState::HalfOpen => {
                    state.half_open_requests = 0;
                    state.consecutive_successes = 0;
                }
                CircuitState::Open => {
                    state.consecutive_successes = 0;
                }
            }
        }
    }

    /// Manually reset the circuit breaker to closed state.
    pub fn reset(&self) {
        let mut state = self.state.write();
        info!(
            service = %self.config.service_name,
            from = %state.state,
            "Circuit breaker manually reset"
        );
        state.state = CircuitState::Closed;
        state.consecutive_failures = 0;
        state.consecutive_successes = 0;
        state.last_failure_time = None;
        state.half_open_requests = 0;
        self.state_transitions.fetch_add(1, Ordering::Relaxed);
    }

    /// Manually trip the circuit breaker to open state.
    pub fn trip(&self) {
        let mut state = self.state.write();
        info!(
            service = %self.config.service_name,
            from = %state.state,
            "Circuit breaker manually tripped"
        );
        state.state = CircuitState::Open;
        state.last_failure_time = Some(Instant::now());
        self.state_transitions.fetch_add(1, Ordering::Relaxed);
    }
}

/// Metrics from a circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitBreakerMetrics {
    pub state: CircuitState,
    pub total_requests: u64,
    pub total_failures: u64,
    pub total_successes: u64,
    pub total_rejections: u64,
    pub consecutive_failures: u32,
    pub consecutive_successes: u32,
    pub state_transitions: u32,
}

impl CircuitBreakerMetrics {
    /// Calculate the failure rate.
    pub fn failure_rate(&self) -> f64 {
        let total = self.total_successes + self.total_failures;
        if total == 0 {
            0.0
        } else {
            self.total_failures as f64 / total as f64
        }
    }

    /// Calculate the rejection rate.
    pub fn rejection_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_rejections as f64 / self.total_requests as f64
        }
    }
}

/// A registry of circuit breakers for managing multiple services.
#[derive(Default)]
pub struct CircuitBreakerRegistry {
    breakers: RwLock<hashbrown::HashMap<String, Arc<CircuitBreaker>>>,
}

impl CircuitBreakerRegistry {
    /// Create a new registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a circuit breaker for a service.
    pub fn get_or_create(&self, service_name: &str) -> Arc<CircuitBreaker> {
        // Fast path: check if it exists
        {
            let breakers = self.breakers.read();
            if let Some(cb) = breakers.get(service_name) {
                return cb.clone();
            }
        }

        // Slow path: create a new one
        let mut breakers = self.breakers.write();
        breakers
            .entry(service_name.to_string())
            .or_insert_with(|| Arc::new(CircuitBreaker::for_service(service_name)))
            .clone()
    }

    /// Get or create a circuit breaker with custom config.
    pub fn get_or_create_with_config(
        &self,
        config: CircuitBreakerConfig,
    ) -> Arc<CircuitBreaker> {
        let mut breakers = self.breakers.write();
        breakers
            .entry(config.service_name.clone())
            .or_insert_with(|| Arc::new(CircuitBreaker::new(config)))
            .clone()
    }

    /// Get all circuit breaker metrics.
    pub fn all_metrics(&self) -> Vec<(String, CircuitBreakerMetrics)> {
        let breakers = self.breakers.read();
        breakers
            .iter()
            .map(|(name, cb)| (name.clone(), cb.metrics()))
            .collect()
    }

    /// Reset all circuit breakers.
    pub fn reset_all(&self) {
        let breakers = self.breakers.read();
        for cb in breakers.values() {
            cb.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_initial_state() {
        let cb = CircuitBreaker::for_service("test");
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.is_allowing_requests());
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_on_failures() {
        let config = CircuitBreakerConfig::new("test")
            .with_failure_threshold(3);
        let cb = CircuitBreaker::new(config);

        // Simulate failures
        for _ in 0..3 {
            let _ = cb.call(|| async { Err::<(), _>("error") }).await;
        }

        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.is_allowing_requests());
    }

    #[tokio::test]
    async fn test_circuit_breaker_success() {
        let cb = CircuitBreaker::for_service("test");

        let result = cb.call(|| async { Ok::<_, &str>(42) }).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        let metrics = cb.metrics();
        assert_eq!(metrics.total_successes, 1);
        assert_eq!(metrics.total_failures, 0);
    }

    #[tokio::test]
    async fn test_circuit_breaker_rejects_when_open() {
        let config = CircuitBreakerConfig::new("test")
            .with_failure_threshold(2)
            .with_reset_timeout(Duration::from_secs(60));
        let cb = CircuitBreaker::new(config);

        // Open the circuit
        for _ in 0..2 {
            let _ = cb.call(|| async { Err::<(), _>("error") }).await;
        }

        assert_eq!(cb.state(), CircuitState::Open);

        // Should be rejected
        let result = cb.call(|| async { Ok::<_, &str>(42) }).await;
        assert!(matches!(result, Err(CircuitBreakerError::Open { .. })));

        let metrics = cb.metrics();
        assert_eq!(metrics.total_rejections, 1);
    }

    #[tokio::test]
    async fn test_circuit_breaker_transitions_to_half_open() {
        let config = CircuitBreakerConfig::new("test")
            .with_failure_threshold(2)
            .with_reset_timeout(Duration::from_millis(50));
        let cb = CircuitBreaker::new(config);

        // Open the circuit
        for _ in 0..2 {
            let _ = cb.call(|| async { Err::<(), _>("error") }).await;
        }

        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for reset timeout
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Should transition to half-open
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[tokio::test]
    async fn test_circuit_breaker_closes_after_successes() {
        let config = CircuitBreakerConfig::new("test")
            .with_failure_threshold(2)
            .with_success_threshold(2)
            .with_reset_timeout(Duration::from_millis(50));
        let cb = CircuitBreaker::new(config);

        // Open the circuit
        for _ in 0..2 {
            let _ = cb.call(|| async { Err::<(), _>("error") }).await;
        }

        // Wait for reset timeout
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Should be half-open, successes should close it
        let _ = cb.call(|| async { Ok::<_, &str>(1) }).await;
        let _ = cb.call(|| async { Ok::<_, &str>(2) }).await;

        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_manual_reset() {
        let config = CircuitBreakerConfig::new("test")
            .with_failure_threshold(1);
        let cb = CircuitBreaker::new(config);

        cb.trip();
        assert_eq!(cb.state(), CircuitState::Open);

        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_metrics() {
        let cb = CircuitBreaker::for_service("test");
        let metrics = cb.metrics();

        assert_eq!(metrics.state, CircuitState::Closed);
        assert_eq!(metrics.failure_rate(), 0.0);
        assert_eq!(metrics.rejection_rate(), 0.0);
    }

    #[test]
    fn test_registry() {
        let registry = CircuitBreakerRegistry::new();

        let cb1 = registry.get_or_create("service1");
        let cb2 = registry.get_or_create("service2");
        let cb1_again = registry.get_or_create("service1");

        assert_eq!(cb1.service_name(), "service1");
        assert_eq!(cb2.service_name(), "service2");
        assert!(Arc::ptr_eq(&cb1, &cb1_again));
    }
}
