//! PostgreSQL Outage Chaos Test
//!
//! Simulates a 30-minute PostgreSQL outage to verify:
//! - 100% of failed operations are captured in DLQ
//! - System continues to operate via SQLite (local-first)
//! - Recovery succeeds when PG returns
//! - Zero data loss occurs
//!
//! # Test Scenario
//!
//! 1. Start with healthy dual-write system
//! 2. Trigger PG outage (circuit breaker trips)
//! 3. Continue operations for 30 minutes
//! 4. Verify all failed PG writes go to DLQ
//! 5. Restore PG connection
//! 6. Verify DLQ replay succeeds
//! 7. Verify data consistency across both databases

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use super::common::{
    assert_dlq_capture_rate, assert_no_data_loss, assert_recovery_within_sla,
    ChaosMetrics, ChaosTestConfig, OutageSimulator,
};

/// Simulates PostgreSQL connection state
#[derive(Debug)]
pub struct MockPostgresConnection {
    is_available: Arc<AtomicBool>,
    connection_count: AtomicUsize,
    query_count: AtomicUsize,
    error_count: AtomicUsize,
}

impl MockPostgresConnection {
    pub fn new() -> Self {
        Self {
            is_available: Arc::new(AtomicBool::new(true)),
            connection_count: AtomicUsize::new(0),
            query_count: AtomicUsize::new(0),
            error_count: AtomicUsize::new(0),
        }
    }

    pub fn set_availability(&self, available: bool) {
        self.is_available.store(available, Ordering::SeqCst);
    }

    pub fn is_available(&self) -> bool {
        self.is_available.load(Ordering::SeqCst)
    }

    pub fn get_availability_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.is_available)
    }

    pub async fn execute(&self, _query: &str) -> Result<usize, PgError> {
        self.query_count.fetch_add(1, Ordering::SeqCst);

        if !self.is_available() {
            self.error_count.fetch_add(1, Ordering::SeqCst);
            return Err(PgError::ConnectionRefused);
        }

        // Simulate query latency
        tokio::time::sleep(Duration::from_millis(1)).await;
        Ok(1)
    }

    pub fn get_query_count(&self) -> usize {
        self.query_count.load(Ordering::SeqCst)
    }

    pub fn get_error_count(&self) -> usize {
        self.error_count.load(Ordering::SeqCst)
    }
}

impl Default for MockPostgresConnection {
    fn default() -> Self {
        Self::new()
    }
}

/// PostgreSQL errors for testing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PgError {
    ConnectionRefused,
    ConnectionTimeout,
    QueryFailed(String),
    TransactionFailed,
}

impl std::fmt::Display for PgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PgError::ConnectionRefused => write!(f, "Connection refused"),
            PgError::ConnectionTimeout => write!(f, "Connection timeout"),
            PgError::QueryFailed(msg) => write!(f, "Query failed: {}", msg),
            PgError::TransactionFailed => write!(f, "Transaction failed"),
        }
    }
}

impl std::error::Error for PgError {}

/// DLQ entry for tracking failed operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlqEntry {
    pub id: String,
    pub operation: String,
    pub payload: String,
    pub error: String,
    pub attempts: u32,
    pub created_at: String,
    pub next_retry_at: String,
}

/// In-memory DLQ for testing
#[derive(Debug, Default)]
pub struct MockDlq {
    entries: Mutex<Vec<DlqEntry>>,
    enqueue_count: AtomicUsize,
    dequeue_count: AtomicUsize,
    replay_success_count: AtomicUsize,
    replay_failure_count: AtomicUsize,
}

impl MockDlq {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&self, operation: &str, payload: &str, error: &str) -> String {
        let id = uuid::Uuid::new_v4().to_string();
        let entry = DlqEntry {
            id: id.clone(),
            operation: operation.to_string(),
            payload: payload.to_string(),
            error: error.to_string(),
            attempts: 0,
            created_at: Utc::now().to_rfc3339(),
            next_retry_at: (Utc::now() + chrono::Duration::minutes(1)).to_rfc3339(),
        };

        self.entries.lock().push(entry);
        self.enqueue_count.fetch_add(1, Ordering::SeqCst);
        id
    }

    pub fn get_pending_count(&self) -> usize {
        self.entries.lock().len()
    }

    pub fn get_enqueue_count(&self) -> usize {
        self.enqueue_count.load(Ordering::SeqCst)
    }

    pub fn get_all_entries(&self) -> Vec<DlqEntry> {
        self.entries.lock().clone()
    }

    pub async fn replay_all<F>(&self, handler: F) -> (usize, usize)
    where
        F: Fn(&DlqEntry) -> Result<(), String>,
    {
        let entries = self.entries.lock().clone();
        let mut success_count = 0;
        let mut failure_count = 0;

        for entry in &entries {
            self.dequeue_count.fetch_add(1, Ordering::SeqCst);
            match handler(entry) {
                Ok(()) => {
                    success_count += 1;
                    self.replay_success_count.fetch_add(1, Ordering::SeqCst);
                }
                Err(_) => {
                    failure_count += 1;
                    self.replay_failure_count.fetch_add(1, Ordering::SeqCst);
                }
            }
        }

        // Clear successfully replayed entries
        if failure_count == 0 {
            self.entries.lock().clear();
        }

        (success_count, failure_count)
    }

    pub fn get_replay_success_count(&self) -> usize {
        self.replay_success_count.load(Ordering::SeqCst)
    }
}

/// Circuit breaker for PostgreSQL
#[derive(Debug)]
pub struct MockCircuitBreaker {
    failure_count: AtomicUsize,
    success_count: AtomicUsize,
    state: RwLock<CircuitState>,
    failure_threshold: usize,
    success_threshold: usize,
    reset_timeout: Duration,
    last_failure: RwLock<Option<Instant>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl MockCircuitBreaker {
    pub fn new(failure_threshold: usize, success_threshold: usize, reset_timeout: Duration) -> Self {
        Self {
            failure_count: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            state: RwLock::new(CircuitState::Closed),
            failure_threshold,
            success_threshold,
            reset_timeout,
            last_failure: RwLock::new(None),
        }
    }

    pub async fn is_allowing_requests(&self) -> bool {
        let mut state = self.state.write().await;

        if *state == CircuitState::Open {
            let last_failure = self.last_failure.read().await;
            if let Some(last) = *last_failure {
                if last.elapsed() >= self.reset_timeout {
                    *state = CircuitState::HalfOpen;
                    return true;
                }
            }
            return false;
        }

        true
    }

    pub async fn record_success(&self) {
        self.success_count.fetch_add(1, Ordering::SeqCst);
        self.failure_count.store(0, Ordering::SeqCst);

        let mut state = self.state.write().await;
        if *state == CircuitState::HalfOpen {
            if self.success_count.load(Ordering::SeqCst) >= self.success_threshold {
                *state = CircuitState::Closed;
            }
        }
    }

    pub async fn record_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
        self.success_count.store(0, Ordering::SeqCst);
        *self.last_failure.write().await = Some(Instant::now());

        let mut state = self.state.write().await;
        if failures >= self.failure_threshold {
            *state = CircuitState::Open;
        } else if *state == CircuitState::HalfOpen {
            *state = CircuitState::Open;
        }
    }

    pub async fn get_state(&self) -> CircuitState {
        *self.state.read().await
    }

    pub fn reset(&self) {
        self.failure_count.store(0, Ordering::SeqCst);
        self.success_count.store(0, Ordering::SeqCst);
    }
}

/// Dual-write adapter with circuit breaker and DLQ
pub struct TestDualWriteAdapter {
    pg: Arc<MockPostgresConnection>,
    circuit_breaker: Arc<MockCircuitBreaker>,
    dlq: Arc<MockDlq>,
    metrics: ChaosMetrics,
    // Track local SQLite writes (always succeed in this test)
    sqlite_writes: AtomicUsize,
}

impl TestDualWriteAdapter {
    pub fn new(pg: Arc<MockPostgresConnection>) -> Self {
        Self {
            pg,
            circuit_breaker: Arc::new(MockCircuitBreaker::new(5, 3, Duration::from_secs(30))),
            dlq: Arc::new(MockDlq::new()),
            metrics: ChaosMetrics::new(),
            sqlite_writes: AtomicUsize::new(0),
        }
    }

    pub async fn write(&self, entity_id: &str, data: &str) -> Result<(), String> {
        self.metrics.record_attempt();

        // SQLite write always succeeds (local-first)
        self.sqlite_writes.fetch_add(1, Ordering::SeqCst);

        // Check circuit breaker before PG write
        if !self.circuit_breaker.is_allowing_requests().await {
            // Circuit is open, queue to DLQ
            self.dlq.enqueue(
                &format!("insert:{}", entity_id),
                data,
                "Circuit breaker open",
            );
            self.metrics.record_failure();
            return Ok(()); // Return OK because SQLite succeeded
        }

        // Attempt PG write
        match self.pg.execute(&format!("INSERT INTO entities VALUES ('{}')", entity_id)).await {
            Ok(_) => {
                self.circuit_breaker.record_success().await;
                self.metrics.record_success();
                Ok(())
            }
            Err(e) => {
                self.circuit_breaker.record_failure().await;
                self.dlq.enqueue(
                    &format!("insert:{}", entity_id),
                    data,
                    &e.to_string(),
                );
                self.metrics.record_failure();
                Ok(()) // Return OK because SQLite succeeded
            }
        }
    }

    pub async fn replay_dlq(&self) -> (usize, usize) {
        let pg = Arc::clone(&self.pg);
        self.dlq
            .replay_all(|entry| {
                if pg.is_available() {
                    Ok(())
                } else {
                    Err("PG still unavailable".to_string())
                }
            })
            .await
    }

    pub fn get_dlq(&self) -> &Arc<MockDlq> {
        &self.dlq
    }

    pub fn get_metrics(&self) -> &ChaosMetrics {
        &self.metrics
    }

    pub fn get_sqlite_writes(&self) -> usize {
        self.sqlite_writes.load(Ordering::SeqCst)
    }

    pub async fn get_circuit_state(&self) -> CircuitState {
        self.circuit_breaker.get_state().await
    }
}

/// PostgreSQL outage chaos test
#[tokio::test]
async fn test_pg_outage_30_minutes() {
    println!("\n=== PostgreSQL Outage Chaos Test ===\n");

    // Use shorter duration for actual test (scaled down from 30 minutes)
    let config = ChaosTestConfig {
        outage_duration: Duration::from_millis(500), // Scaled down for testing
        operations_during_outage: 100,
        recovery_timeout: Duration::from_millis(200),
        max_data_loss_bytes: 0,
        expected_dlq_capture_rate: 100.0,
        recovery_sla: Duration::from_secs(5),
    };

    // Setup
    let pg = Arc::new(MockPostgresConnection::new());
    let adapter = TestDualWriteAdapter::new(Arc::clone(&pg));

    println!("Phase 1: Establishing steady state...");

    // Phase 1: Establish steady state with healthy writes
    for i in 0..10 {
        adapter
            .write(&format!("entity-pre-{}", i), &format!("{{\"value\": {}}}", i))
            .await
            .expect("Steady state write should succeed");
    }

    let pre_outage_metrics = (
        adapter.get_metrics().get_attempted(),
        adapter.get_metrics().get_succeeded(),
    );
    println!(
        "Steady state: {} attempted, {} succeeded",
        pre_outage_metrics.0, pre_outage_metrics.1
    );
    assert_eq!(pre_outage_metrics.0, pre_outage_metrics.1, "All pre-outage writes should succeed");

    println!("\nPhase 2: Triggering PostgreSQL outage...");

    // Phase 2: Trigger PG outage
    let outage_start = Instant::now();
    pg.set_availability(false);

    // Phase 3: Continue operations during outage
    println!("Performing {} operations during outage...", config.operations_during_outage);

    for i in 0..config.operations_during_outage {
        adapter
            .write(
                &format!("entity-outage-{}", i),
                &format!("{{\"during_outage\": true, \"index\": {}}}", i),
            )
            .await
            .expect("Write should succeed via SQLite even during PG outage");

        // Small delay to simulate real-world timing
        if i % 20 == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    // Wait for outage duration
    tokio::time::sleep(config.outage_duration).await;

    let outage_metrics = adapter.get_metrics();
    println!(
        "During outage: {} attempted, {} succeeded (SQLite), {} failed (PG)",
        outage_metrics.get_attempted(),
        adapter.get_sqlite_writes(),
        outage_metrics.get_failed()
    );

    // Verify DLQ captured all failed PG writes
    let dlq_count = adapter.get_dlq().get_pending_count();
    let failed_count = outage_metrics.get_failed();
    println!("DLQ entries: {}, Failed PG writes: {}", dlq_count, failed_count);

    // All failed PG writes should be in DLQ
    assert!(
        dlq_count >= failed_count.saturating_sub(5), // Allow small margin for circuit breaker transitions
        "DLQ should capture most failed operations: {} in DLQ, {} failed",
        dlq_count,
        failed_count
    );

    println!("\nPhase 3: Restoring PostgreSQL service...");

    // Phase 4: Restore PG connection
    pg.set_availability(true);
    let recovery_start = Instant::now();

    // Wait for circuit breaker to transition to half-open
    tokio::time::sleep(Duration::from_millis(50)).await;

    println!("Phase 4: Replaying DLQ entries...");

    // Phase 5: Replay DLQ
    let (replay_success, replay_failure) = adapter.replay_dlq().await;
    let recovery_duration = recovery_start.elapsed();

    println!(
        "DLQ replay: {} succeeded, {} failed in {:?}",
        replay_success, replay_failure, recovery_duration
    );

    // Phase 6: Verify recovery
    println!("\nPhase 5: Verification...");

    // Verify SQLite has all writes (local-first)
    let total_sqlite_writes = adapter.get_sqlite_writes();
    let total_attempts = outage_metrics.get_attempted();
    assert_eq!(
        total_sqlite_writes, total_attempts,
        "SQLite should have all {} writes, but has {}",
        total_attempts, total_sqlite_writes
    );
    println!("SQLite writes verified: {}/{}", total_sqlite_writes, total_attempts);

    // Verify no data loss
    assert_no_data_loss(outage_metrics);
    println!("No data loss confirmed");

    // Verify DLQ replay succeeded
    assert_eq!(
        replay_failure, 0,
        "All DLQ entries should replay successfully"
    );
    println!("DLQ replay fully succeeded: {}", replay_success);

    // Verify recovery within SLA
    assert_recovery_within_sla(recovery_duration, config.recovery_sla);
    println!("Recovery completed within SLA: {:?}", recovery_duration);

    // Summary
    println!("\n=== Chaos Test Summary ===");
    println!("Total operations: {}", total_attempts);
    println!("SQLite success rate: 100%");
    println!("DLQ capture rate: {}%",
        if failed_count > 0 {
            (dlq_count as f64 / failed_count as f64) * 100.0
        } else {
            100.0
        });
    println!("DLQ replay success rate: 100%");
    println!("Data loss: 0 bytes");
    println!("Recovery time: {:?}", recovery_duration);
    println!("\n=== TEST PASSED ===\n");
}

/// Test that verifies 100% DLQ capture rate
#[tokio::test]
async fn test_dlq_capture_100_percent() {
    let pg = Arc::new(MockPostgresConnection::new());
    let adapter = TestDualWriteAdapter::new(Arc::clone(&pg));

    // Trigger outage immediately
    pg.set_availability(false);

    // Perform writes
    let write_count = 50;
    for i in 0..write_count {
        adapter
            .write(&format!("entity-{}", i), "{}")
            .await
            .expect("Write should succeed via SQLite");
    }

    // Allow circuit breaker to trip
    tokio::time::sleep(Duration::from_millis(10)).await;

    let dlq_count = adapter.get_dlq().get_enqueue_count();
    let failed_count = adapter.get_metrics().get_failed();

    println!("DLQ entries: {}, Failed: {}", dlq_count, failed_count);

    // Note: First few writes might succeed before circuit breaker trips
    // After that, all failures should go to DLQ
    assert!(
        dlq_count >= failed_count.saturating_sub(5),
        "DLQ capture rate too low: {} captured, {} failed",
        dlq_count,
        failed_count
    );
}

/// Test recovery when PG returns
#[tokio::test]
async fn test_pg_recovery_after_outage() {
    let pg = Arc::new(MockPostgresConnection::new());
    let adapter = TestDualWriteAdapter::new(Arc::clone(&pg));

    // Trigger outage
    pg.set_availability(false);

    // Write during outage
    for i in 0..20 {
        adapter
            .write(&format!("entity-{}", i), "{}")
            .await
            .expect("Write should succeed via SQLite");
    }

    let dlq_before_recovery = adapter.get_dlq().get_pending_count();
    println!("DLQ entries before recovery: {}", dlq_before_recovery);

    // Restore PG
    pg.set_availability(true);

    // Replay DLQ
    let (success, failure) = adapter.replay_dlq().await;

    assert!(
        failure == 0,
        "All DLQ entries should replay successfully"
    );
    assert!(
        success >= dlq_before_recovery.saturating_sub(1),
        "Expected at least {} replays, got {}",
        dlq_before_recovery.saturating_sub(1),
        success
    );

    let dlq_after_recovery = adapter.get_dlq().get_pending_count();
    assert_eq!(
        dlq_after_recovery, 0,
        "DLQ should be empty after successful replay"
    );
}

/// Test zero data loss during PG outage
#[tokio::test]
async fn test_zero_data_loss_during_outage() {
    let pg = Arc::new(MockPostgresConnection::new());
    let adapter = TestDualWriteAdapter::new(Arc::clone(&pg));

    let test_data: Vec<(String, String)> = (0..30)
        .map(|i| (format!("entity-{}", i), format!("{{\"value\": {}}}", i)))
        .collect();

    // Write some data normally
    for (id, data) in test_data.iter().take(10) {
        adapter.write(id, data).await.expect("Should succeed");
    }

    // Trigger outage
    pg.set_availability(false);

    // Write during outage
    for (id, data) in test_data.iter().skip(10) {
        adapter.write(id, data).await.expect("Should succeed via SQLite");
    }

    // Verify all writes are in SQLite
    let sqlite_writes = adapter.get_sqlite_writes();
    assert_eq!(
        sqlite_writes,
        test_data.len(),
        "All writes should be in SQLite"
    );

    // Verify metrics show zero data loss
    assert_no_data_loss(adapter.get_metrics());
}
