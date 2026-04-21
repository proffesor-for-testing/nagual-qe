//! Chaos and Resilience Tests for Nagual
//!
//! This module contains chaos engineering tests to verify system resilience
//! under various failure conditions:
//!
//! - PostgreSQL outage simulation
//! - GCloud unavailability simulation
//! - Concurrent write stress testing
//! - Disk space exhaustion handling
//! - Process crash during write recovery
//!
//! These tests follow the principles of chaos engineering:
//! 1. Define steady state behavior
//! 2. Hypothesize that steady state continues during/after chaos
//! 3. Introduce real-world events (network failures, crashes, etc.)
//! 4. Try to disprove the hypothesis
//!
//! # Running Chaos Tests
//!
//! ```bash
//! # Run all chaos tests
//! cargo test --test chaos_tests
//!
//! # Run specific chaos test
//! cargo test --test chaos_tests pg_outage
//!
//! # Run with verbose output
//! cargo test --test chaos_tests -- --nocapture
//! ```

pub mod concurrent_test;
pub mod crash_test;
pub mod disk_test;
pub mod gcloud_outage_test;
pub mod pg_outage_test;

/// Common test utilities for chaos tests
pub mod common {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Tracks metrics during chaos tests
    #[derive(Debug, Clone)]
    pub struct ChaosMetrics {
        pub operations_attempted: Arc<AtomicUsize>,
        pub operations_succeeded: Arc<AtomicUsize>,
        pub operations_failed: Arc<AtomicUsize>,
        pub operations_recovered: Arc<AtomicUsize>,
        pub data_loss_bytes: Arc<AtomicUsize>,
    }

    impl Default for ChaosMetrics {
        fn default() -> Self {
            Self::new()
        }
    }

    impl ChaosMetrics {
        pub fn new() -> Self {
            Self {
                operations_attempted: Arc::new(AtomicUsize::new(0)),
                operations_succeeded: Arc::new(AtomicUsize::new(0)),
                operations_failed: Arc::new(AtomicUsize::new(0)),
                operations_recovered: Arc::new(AtomicUsize::new(0)),
                data_loss_bytes: Arc::new(AtomicUsize::new(0)),
            }
        }

        pub fn record_attempt(&self) {
            self.operations_attempted.fetch_add(1, Ordering::SeqCst);
        }

        pub fn record_success(&self) {
            self.operations_succeeded.fetch_add(1, Ordering::SeqCst);
        }

        pub fn record_failure(&self) {
            self.operations_failed.fetch_add(1, Ordering::SeqCst);
        }

        pub fn record_recovery(&self) {
            self.operations_recovered.fetch_add(1, Ordering::SeqCst);
        }

        pub fn record_data_loss(&self, bytes: usize) {
            self.data_loss_bytes.fetch_add(bytes, Ordering::SeqCst);
        }

        pub fn get_attempted(&self) -> usize {
            self.operations_attempted.load(Ordering::SeqCst)
        }

        pub fn get_succeeded(&self) -> usize {
            self.operations_succeeded.load(Ordering::SeqCst)
        }

        pub fn get_failed(&self) -> usize {
            self.operations_failed.load(Ordering::SeqCst)
        }

        pub fn get_recovered(&self) -> usize {
            self.operations_recovered.load(Ordering::SeqCst)
        }

        pub fn get_data_loss(&self) -> usize {
            self.data_loss_bytes.load(Ordering::SeqCst)
        }

        pub fn success_rate(&self) -> f64 {
            let attempted = self.get_attempted() as f64;
            if attempted == 0.0 {
                return 100.0;
            }
            (self.get_succeeded() as f64 / attempted) * 100.0
        }

        pub fn recovery_rate(&self) -> f64 {
            let failed = self.get_failed() as f64;
            if failed == 0.0 {
                return 100.0;
            }
            (self.get_recovered() as f64 / failed) * 100.0
        }
    }

    /// Simulates a service outage
    pub struct OutageSimulator {
        is_available: Arc<AtomicBool>,
        outage_start: Option<Instant>,
        outage_duration: Duration,
    }

    impl OutageSimulator {
        pub fn new() -> Self {
            Self {
                is_available: Arc::new(AtomicBool::new(true)),
                outage_start: None,
                outage_duration: Duration::from_secs(0),
            }
        }

        pub fn trigger_outage(&mut self, duration: Duration) {
            self.is_available.store(false, Ordering::SeqCst);
            self.outage_start = Some(Instant::now());
            self.outage_duration = duration;
        }

        pub fn restore_service(&mut self) {
            self.is_available.store(true, Ordering::SeqCst);
            self.outage_start = None;
        }

        pub fn is_available(&self) -> bool {
            if let Some(start) = self.outage_start {
                if start.elapsed() >= self.outage_duration {
                    self.is_available.store(true, Ordering::SeqCst);
                    return true;
                }
            }
            self.is_available.load(Ordering::SeqCst)
        }

        pub fn get_availability_flag(&self) -> Arc<AtomicBool> {
            Arc::clone(&self.is_available)
        }

        pub fn time_until_recovery(&self) -> Option<Duration> {
            self.outage_start.map(|start| {
                let elapsed = start.elapsed();
                if elapsed >= self.outage_duration {
                    Duration::from_secs(0)
                } else {
                    self.outage_duration - elapsed
                }
            })
        }
    }

    impl Default for OutageSimulator {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Chaos test configuration
    #[derive(Debug, Clone)]
    pub struct ChaosTestConfig {
        /// Duration of the simulated outage
        pub outage_duration: Duration,
        /// Number of operations to attempt during outage
        pub operations_during_outage: usize,
        /// Time to wait for recovery after outage ends
        pub recovery_timeout: Duration,
        /// Maximum acceptable data loss (in bytes)
        pub max_data_loss_bytes: usize,
        /// Expected DLQ capture rate during outage
        pub expected_dlq_capture_rate: f64,
        /// Recovery time SLA
        pub recovery_sla: Duration,
    }

    impl Default for ChaosTestConfig {
        fn default() -> Self {
            Self {
                outage_duration: Duration::from_secs(30),
                operations_during_outage: 100,
                recovery_timeout: Duration::from_secs(60),
                max_data_loss_bytes: 0, // Zero tolerance for data loss
                expected_dlq_capture_rate: 100.0, // 100% DLQ capture
                recovery_sla: Duration::from_secs(300), // 5 minutes
            }
        }
    }

    impl ChaosTestConfig {
        pub fn pg_outage_30min() -> Self {
            Self {
                outage_duration: Duration::from_secs(30 * 60), // 30 minutes
                operations_during_outage: 1000,
                recovery_timeout: Duration::from_secs(5 * 60), // 5 minutes
                max_data_loss_bytes: 0,
                expected_dlq_capture_rate: 100.0,
                recovery_sla: Duration::from_secs(5 * 60),
            }
        }

        pub fn gcloud_outage() -> Self {
            Self {
                outage_duration: Duration::from_secs(15 * 60), // 15 minutes
                operations_during_outage: 500,
                recovery_timeout: Duration::from_secs(5 * 60),
                max_data_loss_bytes: 0,
                expected_dlq_capture_rate: 100.0,
                recovery_sla: Duration::from_secs(5 * 60),
            }
        }

        pub fn concurrent_stress() -> Self {
            Self {
                outage_duration: Duration::from_secs(0), // No outage
                operations_during_outage: 100, // 100 concurrent operations
                recovery_timeout: Duration::from_secs(30),
                max_data_loss_bytes: 0,
                expected_dlq_capture_rate: 100.0,
                recovery_sla: Duration::from_secs(10),
            }
        }
    }

    /// Assertion helpers for chaos tests
    pub fn assert_no_data_loss(metrics: &ChaosMetrics) {
        let data_loss = metrics.get_data_loss();
        assert_eq!(
            data_loss, 0,
            "Data loss detected: {} bytes lost",
            data_loss
        );
    }

    pub fn assert_dlq_capture_rate(metrics: &ChaosMetrics, expected_rate: f64) {
        let failed = metrics.get_failed() as f64;
        let recovered = metrics.get_recovered() as f64;
        if failed > 0.0 {
            let capture_rate = (recovered / failed) * 100.0;
            assert!(
                capture_rate >= expected_rate,
                "DLQ capture rate {:.1}% is below expected {:.1}%",
                capture_rate,
                expected_rate
            );
        }
    }

    pub fn assert_recovery_within_sla(
        recovery_time: Duration,
        sla: Duration,
    ) {
        assert!(
            recovery_time <= sla,
            "Recovery time {:?} exceeded SLA {:?}",
            recovery_time,
            sla
        );
    }

    pub fn assert_no_data_corruption(
        expected: &[u8],
        actual: &[u8],
    ) {
        assert_eq!(
            expected, actual,
            "Data corruption detected"
        );
    }
}
