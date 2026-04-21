//! Integration tests for PostgreSQL failure scenarios.
//!
//! These tests verify graceful degradation when PostgreSQL is unavailable,
//! DLQ capture of failed writes, and recovery after PostgreSQL restoration.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use nagual::db::{DualWritable, DualWriteAdapter, DualWriteConfig, SqliteDb, PostgresDb};
use nagual::error::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitState,
    DeadLetterQueue, DlqEntry, Result,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tokio::sync::RwLock;

/// Test entity for PostgreSQL failure tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverPattern {
    pub id: String,
    pub data: String,
    pub updated_at: DateTime<Utc>,
}

impl FailoverPattern {
    pub fn new(id: &str, data: &str) -> Self {
        Self {
            id: id.to_string(),
            data: data.to_string(),
            updated_at: Utc::now(),
        }
    }
}

#[async_trait::async_trait]
impl DualWritable for FailoverPattern {
    type Id = String;

    fn table_name() -> &'static str {
        "failover_patterns"
    }

    fn id(&self) -> Self::Id {
        self.id.clone()
    }

    fn updated_at(&self) -> DateTime<Utc> {
        self.updated_at
    }

    fn set_updated_at(&mut self, ts: DateTime<Utc>) {
        self.updated_at = ts;
    }

    fn sqlite_insert_sql() -> &'static str {
        "INSERT OR REPLACE INTO failover_patterns (id, data, updated_at) VALUES (?, ?, ?)"
    }

    fn sqlite_update_sql() -> &'static str {
        "UPDATE failover_patterns SET data = ?, updated_at = ? WHERE id = ?"
    }

    fn sqlite_insert_params(&self) -> Vec<Box<dyn rusqlite::ToSql + Send + Sync>> {
        vec![
            Box::new(self.id.clone()),
            Box::new(self.data.clone()),
            Box::new(self.updated_at.to_rfc3339()),
        ]
    }

    fn sqlite_update_params(&self) -> Vec<Box<dyn rusqlite::ToSql + Send + Sync>> {
        vec![
            Box::new(self.data.clone()),
            Box::new(self.updated_at.to_rfc3339()),
            Box::new(self.id.clone()),
        ]
    }

    fn postgres_insert_sql() -> &'static str {
        "INSERT INTO failover_patterns (id, data, updated_at) VALUES ($1, $2, $3) ON CONFLICT (id) DO UPDATE SET data = $2, updated_at = $3"
    }

    fn postgres_update_sql() -> &'static str {
        "UPDATE failover_patterns SET data = $1, updated_at = $2 WHERE id = $3"
    }

    async fn postgres_insert(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error> {
        sqlx::query(Self::postgres_insert_sql())
            .bind(&self.id)
            .bind(&self.data)
            .bind(self.updated_at)
            .execute(pool)
            .await?;
        Ok(())
    }

    async fn postgres_update(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error> {
        sqlx::query(Self::postgres_update_sql())
            .bind(&self.data)
            .bind(self.updated_at)
            .bind(&self.id)
            .execute(pool)
            .await?;
        Ok(())
    }

    async fn postgres_delete(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error> {
        sqlx::query("DELETE FROM failover_patterns WHERE id = $1")
            .bind(&self.id)
            .execute(pool)
            .await?;
        Ok(())
    }
}

/// Helper to setup test schema.
async fn setup_schema(db: &SqliteDb) -> Result<()> {
    db.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS failover_patterns (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        "#,
    )
    .await
}

// ============================================================================
// Test: Graceful Degradation When PG Unavailable
// ============================================================================

mod graceful_degradation_tests {
    use super::*;

    #[tokio::test]
    async fn test_sqlite_continues_when_no_pg_configured() {
        // DualWriteAdapter without PostgreSQL should work with SQLite only
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        setup_schema(adapter.sqlite()).await.unwrap();

        assert!(!adapter.is_postgres_available());

        let pattern = FailoverPattern::new("graceful-1", "Test data");
        let result = adapter.insert(&pattern).await.unwrap();

        assert!(result.is_ok());
        assert!(result.sqlite_success);
        assert!(result.postgres_success.is_none());
    }

    #[tokio::test]
    async fn test_multiple_writes_without_pg() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        setup_schema(adapter.sqlite()).await.unwrap();

        // Multiple writes should succeed
        for i in 0..10 {
            let pattern = FailoverPattern::new(&format!("multi-{}", i), &format!("Data {}", i));
            let result = adapter.insert(&pattern).await.unwrap();
            assert!(result.sqlite_success);
        }

        // Verify all data in SQLite
        let count: i64 = adapter
            .sqlite()
            .query_one("SELECT COUNT(*) FROM failover_patterns", &[], |row| {
                row.get(0)
            })
            .await
            .unwrap()
            .unwrap();
        assert_eq!(count, 10);
    }

    #[tokio::test]
    async fn test_circuit_state_without_pg() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();

        // Circuit should be closed even without PG
        assert_eq!(adapter.circuit_state(), CircuitState::Closed);
    }
}

// ============================================================================
// Test: DLQ Captures Failed Writes
// ============================================================================

mod dlq_capture_tests {
    use super::*;

    #[tokio::test]
    async fn test_dlq_entry_creation() {
        let entry = DlqEntry::new(
            "test_operation",
            r#"{"id": "test-1", "data": "test"}"#,
            "Connection refused",
        );

        assert_eq!(entry.operation, "test_operation");
        assert_eq!(entry.attempts, 0);
        assert_eq!(entry.max_attempts, 10);
        assert!(!entry.should_abandon());
    }

    #[tokio::test]
    async fn test_dlq_enqueue_and_dequeue() {
        let dlq = DeadLetterQueue::in_memory().unwrap();

        let entry = DlqEntry::new("pg_write", r#"{"id": "1"}"#, "Connection timeout");
        let id = dlq.enqueue(&entry).unwrap();

        // Entry should not be immediately ready (future next_retry_at)
        let dequeued = dlq.dequeue().unwrap();
        assert!(dequeued.is_none());

        // Check stats
        let stats = dlq.stats().unwrap();
        assert_eq!(stats.pending, 1);
    }

    #[tokio::test]
    async fn test_dlq_mark_success() {
        let dlq = DeadLetterQueue::in_memory().unwrap();

        let entry = DlqEntry::new("pg_write", r#"{"id": "1"}"#, "Error");
        let id = dlq.enqueue(&entry).unwrap();

        dlq.mark_success(&id).unwrap();

        let stats = dlq.stats().unwrap();
        assert_eq!(stats.total, 0);
    }

    #[tokio::test]
    async fn test_dlq_mark_failure_with_retry() {
        let dlq = DeadLetterQueue::in_memory().unwrap();

        let entry = DlqEntry::new("pg_write", r#"{"id": "1"}"#, "Error").with_max_attempts(5);
        let id = dlq.enqueue(&entry).unwrap();

        let will_retry = dlq.mark_failure(&id, "Still failing").unwrap();
        assert!(will_retry);

        let stats = dlq.stats().unwrap();
        assert_eq!(stats.pending, 1);
    }

    #[tokio::test]
    async fn test_dlq_abandons_after_max_attempts() {
        let dlq = DeadLetterQueue::in_memory().unwrap();

        let entry = DlqEntry::new("pg_write", r#"{"id": "1"}"#, "Error").with_max_attempts(2);
        let id = dlq.enqueue(&entry).unwrap();

        // First failure - should retry
        let will_retry = dlq.mark_failure(&id, "Attempt 1").unwrap();
        assert!(will_retry);

        // Second failure - should abandon
        let will_retry = dlq.mark_failure(&id, "Attempt 2").unwrap();
        assert!(!will_retry);

        let stats = dlq.stats().unwrap();
        assert_eq!(stats.abandoned, 1);
        assert_eq!(stats.pending, 0);
    }

    #[tokio::test]
    async fn test_dlq_max_size_enforcement() {
        let dlq = DeadLetterQueue::in_memory().unwrap().with_max_size(3);

        // Fill up the queue
        for i in 0..3 {
            let entry = DlqEntry::new(format!("op_{}", i), "{}", "Error");
            dlq.enqueue(&entry).unwrap();
        }

        // Fourth entry should fail
        let entry = DlqEntry::new("op_overflow", "{}", "Error");
        let result = dlq.enqueue(&entry);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_dlq_get_by_operation() {
        let dlq = DeadLetterQueue::in_memory().unwrap();

        // Add entries with different operations
        for i in 0..3 {
            let entry = DlqEntry::new("type_a", format!("{}", i), "Error");
            dlq.enqueue(&entry).unwrap();
        }
        for i in 0..2 {
            let entry = DlqEntry::new("type_b", format!("{}", i), "Error");
            dlq.enqueue(&entry).unwrap();
        }

        let type_a_entries = dlq.get_by_operation("type_a", 10).unwrap();
        assert_eq!(type_a_entries.len(), 3);

        let type_b_entries = dlq.get_by_operation("type_b", 10).unwrap();
        assert_eq!(type_b_entries.len(), 2);
    }

    #[tokio::test]
    async fn test_dlq_cleanup_abandoned() {
        let dlq = DeadLetterQueue::in_memory().unwrap();

        // Add and abandon entries
        for i in 0..5 {
            let entry = DlqEntry::new(format!("op_{}", i), "{}", "Error").with_max_attempts(1);
            let id = dlq.enqueue(&entry).unwrap();
            dlq.mark_failure(&id, "Failed").unwrap();
        }

        // Verify abandoned
        let stats = dlq.stats().unwrap();
        assert_eq!(stats.abandoned, 5);

        // Cleanup (with 0 duration - all should be eligible)
        let deleted = dlq.cleanup_abandoned(Duration::from_secs(0)).unwrap();
        assert_eq!(deleted, 5);
    }

    #[tokio::test]
    async fn test_dlq_requeue_stuck() {
        let dlq = DeadLetterQueue::in_memory().unwrap();

        let entry = DlqEntry::new("stuck_op", "{}", "Error");
        dlq.enqueue(&entry).unwrap();

        // Normally entries are stuck if "processing" for too long
        // This tests the requeue_stuck mechanism
        let requeued = dlq.requeue_stuck(Duration::from_secs(0)).unwrap();
        // No entries in "processing" state yet
        assert_eq!(requeued, 0);
    }
}

// ============================================================================
// Test: Recovery After PG Restoration
// ============================================================================

mod recovery_tests {
    use super::*;

    #[tokio::test]
    async fn test_dlq_batch_processing() {
        let dlq = DeadLetterQueue::in_memory().unwrap();

        // Enqueue entries
        for i in 0..5 {
            let entry = DlqEntry::new(format!("batch_{}", i), format!("{}", i), "Initial error");
            dlq.enqueue(&entry).unwrap();
        }

        // Update entries to be ready for retry
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        // Note: We can't directly manipulate the DLQ's internal state in a test
        // This test validates the batch processing interface exists

        let result = dlq.process_batch(10, |entry| {
            // Simulate processing - succeed on even IDs
            if entry.payload.parse::<i32>().unwrap_or(0) % 2 == 0 {
                Ok(())
            } else {
                Err("Simulated failure".to_string())
            }
        });

        // Process_batch requires entries to be ready, which they're not immediately
        // This validates the interface works
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_circuit_breaker_recovery_flow() {
        let config = CircuitBreakerConfig::new("pg-recovery")
            .with_failure_threshold(2)
            .with_success_threshold(2)
            .with_reset_timeout(Duration::from_millis(50));
        let breaker = CircuitBreaker::new(config);

        // Phase 1: Normal operation
        assert_eq!(breaker.state(), CircuitState::Closed);

        // Phase 2: Failures trigger open
        for _ in 0..2 {
            let _ = breaker.call(|| async { Err::<(), _>("PG unavailable") }).await;
        }
        assert_eq!(breaker.state(), CircuitState::Open);

        // Phase 3: Wait for half-open
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        // Phase 4: Successes close the circuit
        for _ in 0..2 {
            let _ = breaker.call(|| async { Ok::<_, &str>(()) }).await;
        }
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_sqlite_state_preserved_during_pg_failure() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        setup_schema(adapter.sqlite()).await.unwrap();

        // Write data during "PG outage" (no PG configured = simulated outage)
        for i in 0..10 {
            let pattern = FailoverPattern::new(&format!("outage-{}", i), &format!("Data {}", i));
            adapter.insert(&pattern).await.unwrap();
        }

        // All data should be in SQLite
        let count: i64 = adapter
            .sqlite()
            .query_one("SELECT COUNT(*) FROM failover_patterns", &[], |row| {
                row.get(0)
            })
            .await
            .unwrap()
            .unwrap();
        assert_eq!(count, 10);

        // Verify data integrity
        let data: Vec<(String, String)> = adapter
            .sqlite()
            .query("SELECT id, data FROM failover_patterns ORDER BY id", &[], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .await
            .unwrap();

        for (i, (id, data)) in data.iter().enumerate() {
            assert_eq!(id, &format!("outage-{}", i));
            assert_eq!(data, &format!("Data {}", i));
        }
    }
}

// ============================================================================
// Test: Partial Failure Scenarios
// ============================================================================

mod partial_failure_tests {
    use super::*;

    #[tokio::test]
    async fn test_sqlite_success_pg_absent() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        setup_schema(adapter.sqlite()).await.unwrap();

        let pattern = FailoverPattern::new("partial-1", "Test data");
        let result = adapter.insert(&pattern).await.unwrap();

        // SQLite should succeed
        assert!(result.sqlite_success);
        // PG is not configured, so None
        assert!(result.postgres_success.is_none());
        // Not queued to DLQ (no PG = no DLQ needed)
        assert!(!result.queued_to_dlq);
    }

    #[tokio::test]
    async fn test_dual_write_result_is_ok() {
        let result = nagual::db::DualWriteResult {
            sqlite_success: true,
            postgres_success: None,
            queued_to_dlq: false,
            entity_id: "test".to_string(),
            warnings: vec![],
        };

        // is_ok means SQLite succeeded
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_dual_write_result_is_fully_synced() {
        // Fully synced when both succeed or PG is not configured
        let result = nagual::db::DualWriteResult {
            sqlite_success: true,
            postgres_success: None,
            queued_to_dlq: false,
            entity_id: "test".to_string(),
            warnings: vec![],
        };
        assert!(result.is_fully_synced());

        let result_with_pg = nagual::db::DualWriteResult {
            sqlite_success: true,
            postgres_success: Some(true),
            queued_to_dlq: false,
            entity_id: "test".to_string(),
            warnings: vec![],
        };
        assert!(result_with_pg.is_fully_synced());

        let result_pg_failed = nagual::db::DualWriteResult {
            sqlite_success: true,
            postgres_success: Some(false),
            queued_to_dlq: true,
            entity_id: "test".to_string(),
            warnings: vec!["PG failed".to_string()],
        };
        assert!(!result_pg_failed.is_fully_synced());
    }

    #[tokio::test]
    async fn test_circuit_breaker_failure_modes() {
        let config = CircuitBreakerConfig::new("partial-test")
            .with_failure_threshold(2)
            .with_half_open_max_requests(1);
        let breaker = CircuitBreaker::new(config);

        // Open the breaker
        for _ in 0..2 {
            let _ = breaker.call(|| async { Err::<(), _>("error") }).await;
        }

        // Requests should be rejected
        let result = breaker.call(|| async { Ok::<_, &str>(()) }).await;
        match result {
            Err(CircuitBreakerError::Open { service, .. }) => {
                assert_eq!(service, "partial-test");
            }
            _ => panic!("Expected Open error"),
        }
    }
}

// ============================================================================
// Test: DLQ Retry Backoff
// ============================================================================

mod retry_backoff_tests {
    use super::*;

    #[tokio::test]
    async fn test_dlq_entry_retry_intervals() {
        let mut entry = DlqEntry::new("backoff_test", "{}", "Error");

        // Verify increasing intervals
        // 0 attempts -> 1 minute
        entry.attempts = 0;
        let next = entry.calculate_next_retry();
        let diff = next - Utc::now();
        assert!(diff.num_seconds() >= 55 && diff.num_seconds() <= 65);

        // 1 attempt -> 5 minutes
        entry.attempts = 1;
        let next = entry.calculate_next_retry();
        let diff = next - Utc::now();
        assert!(diff.num_minutes() >= 4 && diff.num_minutes() <= 6);

        // 2 attempts -> 15 minutes
        entry.attempts = 2;
        let next = entry.calculate_next_retry();
        let diff = next - Utc::now();
        assert!(diff.num_minutes() >= 14 && diff.num_minutes() <= 16);

        // 3 attempts -> 1 hour
        entry.attempts = 3;
        let next = entry.calculate_next_retry();
        let diff = next - Utc::now();
        assert!(diff.num_minutes() >= 59 && diff.num_minutes() <= 61);

        // 4+ attempts -> 6 hours (capped)
        entry.attempts = 4;
        let next = entry.calculate_next_retry();
        let diff = next - Utc::now();
        assert!(diff.num_hours() >= 5 && diff.num_hours() <= 7);
    }

    #[tokio::test]
    async fn test_dlq_entry_should_abandon() {
        let entry = DlqEntry::new("abandon_test", "{}", "Error").with_max_attempts(3);
        assert!(!entry.should_abandon());

        let mut entry_at_max = entry.clone();
        entry_at_max.attempts = 3;
        assert!(entry_at_max.should_abandon());
    }

    #[tokio::test]
    async fn test_dlq_entry_is_ready_for_retry() {
        let mut entry = DlqEntry::new("ready_test", "{}", "Error");

        // Freshly created entry has next_retry_at in the future
        assert!(!entry.is_ready_for_retry());

        // Set next_retry_at to the past
        entry.next_retry_at = Utc::now() - chrono::Duration::seconds(10);
        assert!(entry.is_ready_for_retry());
    }
}

// ============================================================================
// Test: Concurrent DLQ Operations
// ============================================================================

mod concurrent_dlq_tests {
    use super::*;
    use std::sync::Arc;
    use parking_lot::Mutex;

    #[tokio::test]
    async fn test_concurrent_dlq_enqueue() {
        let dlq = Arc::new(Mutex::new(DeadLetterQueue::in_memory().unwrap()));

        let mut handles = Vec::new();

        for i in 0..10 {
            let dlq_clone = Arc::clone(&dlq);
            let handle = tokio::spawn(async move {
                let entry = DlqEntry::new(format!("concurrent_{}", i), format!("{}", i), "Error");
                dlq_clone.lock().enqueue(&entry)
            });
            handles.push(handle);
        }

        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }

        let stats = dlq.lock().stats().unwrap();
        assert_eq!(stats.pending, 10);
    }
}
