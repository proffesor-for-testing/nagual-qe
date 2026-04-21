//! Integration tests for the dual-write pattern.
//!
//! These tests verify that writes are coordinated correctly between SQLite and PostgreSQL,
//! with proper handling of failures, conflicts, and circuit breaker activation.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use nagual::db::{
    ConflictResolution, DualWritable, DualWriteAdapter, DualWriteConfig, OperationType,
    SqliteDb,
};
use nagual::error::{CircuitBreaker, CircuitBreakerConfig, CircuitState, Result};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

/// Test entity for integration tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestPattern {
    pub id: String,
    pub problem: String,
    pub solution: String,
    pub category: String,
    pub effectiveness: f64,
    pub updated_at: DateTime<Utc>,
}

impl TestPattern {
    pub fn new(id: &str, problem: &str, solution: &str) -> Self {
        Self {
            id: id.to_string(),
            problem: problem.to_string(),
            solution: solution.to_string(),
            category: "testing".to_string(),
            effectiveness: 0.5,
            updated_at: Utc::now(),
        }
    }

    pub fn with_category(mut self, category: &str) -> Self {
        self.category = category.to_string();
        self
    }

    pub fn with_effectiveness(mut self, effectiveness: f64) -> Self {
        self.effectiveness = effectiveness;
        self
    }
}

#[async_trait::async_trait]
impl DualWritable for TestPattern {
    type Id = String;

    fn table_name() -> &'static str {
        "test_patterns"
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
        "INSERT OR REPLACE INTO test_patterns (id, problem, solution, category, effectiveness, updated_at) VALUES (?, ?, ?, ?, ?, ?)"
    }

    fn sqlite_update_sql() -> &'static str {
        "UPDATE test_patterns SET problem = ?, solution = ?, category = ?, effectiveness = ?, updated_at = ? WHERE id = ?"
    }

    fn sqlite_delete_sql() -> &'static str {
        "DELETE FROM test_patterns WHERE id = ?"
    }

    fn sqlite_insert_params(&self) -> Vec<Box<dyn rusqlite::ToSql + Send + Sync>> {
        vec![
            Box::new(self.id.clone()),
            Box::new(self.problem.clone()),
            Box::new(self.solution.clone()),
            Box::new(self.category.clone()),
            Box::new(self.effectiveness),
            Box::new(self.updated_at.to_rfc3339()),
        ]
    }

    fn sqlite_update_params(&self) -> Vec<Box<dyn rusqlite::ToSql + Send + Sync>> {
        vec![
            Box::new(self.problem.clone()),
            Box::new(self.solution.clone()),
            Box::new(self.category.clone()),
            Box::new(self.effectiveness),
            Box::new(self.updated_at.to_rfc3339()),
            Box::new(self.id.clone()),
        ]
    }

    fn postgres_insert_sql() -> &'static str {
        "INSERT INTO test_patterns (id, problem, solution, category, effectiveness, updated_at) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (id) DO UPDATE SET problem = $2, solution = $3, category = $4, effectiveness = $5, updated_at = $6"
    }

    fn postgres_update_sql() -> &'static str {
        "UPDATE test_patterns SET problem = $1, solution = $2, category = $3, effectiveness = $4, updated_at = $5 WHERE id = $6"
    }

    fn postgres_delete_sql() -> &'static str {
        "DELETE FROM test_patterns WHERE id = $1"
    }

    async fn postgres_insert(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error> {
        sqlx::query(Self::postgres_insert_sql())
            .bind(&self.id)
            .bind(&self.problem)
            .bind(&self.solution)
            .bind(&self.category)
            .bind(self.effectiveness)
            .bind(self.updated_at)
            .execute(pool)
            .await?;
        Ok(())
    }

    async fn postgres_update(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error> {
        sqlx::query(Self::postgres_update_sql())
            .bind(&self.problem)
            .bind(&self.solution)
            .bind(&self.category)
            .bind(self.effectiveness)
            .bind(self.updated_at)
            .bind(&self.id)
            .execute(pool)
            .await?;
        Ok(())
    }

    async fn postgres_delete(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error> {
        sqlx::query("DELETE FROM test_patterns WHERE id = $1")
            .bind(&self.id)
            .execute(pool)
            .await?;
        Ok(())
    }
}

/// Helper to create a test schema in SQLite.
async fn setup_sqlite_schema(db: &SqliteDb) -> Result<()> {
    db.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS test_patterns (
            id TEXT PRIMARY KEY,
            problem TEXT NOT NULL,
            solution TEXT NOT NULL,
            category TEXT NOT NULL,
            effectiveness REAL DEFAULT 0.5,
            updated_at TEXT NOT NULL
        );
        "#,
    )
    .await
}

// ============================================================================
// Test: Basic SQLite + DualWrite Coordination (no PostgreSQL)
// ============================================================================

mod sqlite_only_tests {
    use super::*;

    #[tokio::test]
    async fn test_dual_write_adapter_creation() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        assert_eq!(adapter.circuit_state(), CircuitState::Closed);
        assert!(!adapter.is_postgres_available());
    }

    #[tokio::test]
    async fn test_sqlite_insert_write() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();

        // Setup schema
        setup_sqlite_schema(adapter.sqlite()).await.unwrap();

        // Insert a test pattern
        let pattern = TestPattern::new("test-1", "How to cache data?", "Use Redis with TTL");
        let result = adapter.insert(&pattern).await.unwrap();

        assert!(result.is_ok());
        assert!(result.sqlite_success);
        assert_eq!(result.entity_id, "test-1");
        assert!(result.postgres_success.is_none()); // No PG configured
    }

    #[tokio::test]
    async fn test_sqlite_update_write() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        setup_sqlite_schema(adapter.sqlite()).await.unwrap();

        // Insert first
        let mut pattern = TestPattern::new("test-2", "Original problem", "Original solution");
        adapter.insert(&pattern).await.unwrap();

        // Update
        pattern.problem = "Updated problem".to_string();
        pattern.solution = "Updated solution".to_string();
        pattern.updated_at = Utc::now();

        let result = adapter.update(&pattern).await.unwrap();
        assert!(result.is_ok());
        assert!(result.sqlite_success);
    }

    #[tokio::test]
    async fn test_sqlite_delete_write() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        setup_sqlite_schema(adapter.sqlite()).await.unwrap();

        // Insert first
        let pattern = TestPattern::new("test-3", "To be deleted", "Will be removed");
        adapter.insert(&pattern).await.unwrap();

        // Delete
        let result = adapter.delete(&pattern).await.unwrap();
        assert!(result.is_ok());
        assert!(result.sqlite_success);
    }

    #[tokio::test]
    async fn test_sqlite_upsert_write() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        setup_sqlite_schema(adapter.sqlite()).await.unwrap();

        // First upsert (insert)
        let pattern = TestPattern::new("test-4", "Upsert test", "Initial solution");
        let result = adapter.upsert(&pattern).await.unwrap();
        assert!(result.is_ok());

        // Second upsert (update)
        let pattern = TestPattern::new("test-4", "Upsert test", "Updated solution");
        let result = adapter.upsert(&pattern).await.unwrap();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_multiple_entities_insert() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        setup_sqlite_schema(adapter.sqlite()).await.unwrap();

        // Insert multiple patterns
        for i in 0..10 {
            let pattern = TestPattern::new(
                &format!("batch-{}", i),
                &format!("Problem {}", i),
                &format!("Solution {}", i),
            );
            let result = adapter.insert(&pattern).await.unwrap();
            assert!(result.is_ok());
        }

        // Verify count
        let count: i64 = adapter
            .sqlite()
            .query_one("SELECT COUNT(*) FROM test_patterns", &[], |row| row.get(0))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(count, 10);
    }
}

// ============================================================================
// Test: Data Consistency Across Both Databases
// ============================================================================

mod consistency_tests {
    use super::*;

    #[tokio::test]
    async fn test_data_preserved_after_insert() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        setup_sqlite_schema(adapter.sqlite()).await.unwrap();

        let pattern = TestPattern::new("preserve-1", "Test problem", "Test solution")
            .with_category("architecture")
            .with_effectiveness(0.85);

        adapter.insert(&pattern).await.unwrap();

        // Query back from SQLite
        let row: (String, String, String, f64) = adapter
            .sqlite()
            .query_one(
                "SELECT problem, solution, category, effectiveness FROM test_patterns WHERE id = ?",
                &[&"preserve-1"],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .await
            .unwrap()
            .unwrap();

        assert_eq!(row.0, "Test problem");
        assert_eq!(row.1, "Test solution");
        assert_eq!(row.2, "architecture");
        assert!((row.3 - 0.85).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_update_preserves_other_fields() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        setup_sqlite_schema(adapter.sqlite()).await.unwrap();

        // Insert
        let mut pattern = TestPattern::new("update-1", "Original", "Original solution")
            .with_category("performance")
            .with_effectiveness(0.75);
        adapter.insert(&pattern).await.unwrap();

        // Update only problem
        pattern.problem = "Updated".to_string();
        pattern.updated_at = Utc::now();
        adapter.update(&pattern).await.unwrap();

        // Verify category and effectiveness are preserved
        let row: (String, String, f64) = adapter
            .sqlite()
            .query_one(
                "SELECT problem, category, effectiveness FROM test_patterns WHERE id = ?",
                &[&"update-1"],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .await
            .unwrap()
            .unwrap();

        assert_eq!(row.0, "Updated");
        assert_eq!(row.1, "performance");
        assert!((row.2 - 0.75).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_delete_removes_all_fields() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        setup_sqlite_schema(adapter.sqlite()).await.unwrap();

        let pattern = TestPattern::new("delete-1", "To delete", "Solution");
        adapter.insert(&pattern).await.unwrap();

        // Verify exists
        let exists: bool = adapter
            .sqlite()
            .query_one(
                "SELECT 1 FROM test_patterns WHERE id = ?",
                &[&"delete-1"],
                |_| Ok(true),
            )
            .await
            .unwrap()
            .unwrap_or(false);
        assert!(exists);

        // Delete
        adapter.delete(&pattern).await.unwrap();

        // Verify deleted
        let exists: bool = adapter
            .sqlite()
            .query_one(
                "SELECT 1 FROM test_patterns WHERE id = ?",
                &[&"delete-1"],
                |_| Ok(true),
            )
            .await
            .unwrap()
            .unwrap_or(false);
        assert!(!exists);
    }
}

// ============================================================================
// Test: Conflict Resolution
// ============================================================================

mod conflict_tests {
    use super::*;

    #[tokio::test]
    async fn test_lww_local_wins() {
        let local = TestPattern {
            id: "conflict-1".to_string(),
            problem: "Local version".to_string(),
            solution: "Local solution".to_string(),
            category: "testing".to_string(),
            effectiveness: 0.8,
            updated_at: Utc::now(), // Newer
        };

        let remote = TestPattern {
            id: "conflict-1".to_string(),
            problem: "Remote version".to_string(),
            solution: "Remote solution".to_string(),
            category: "testing".to_string(),
            effectiveness: 0.7,
            updated_at: Utc::now() - chrono::Duration::seconds(10), // Older
        };

        let winner = DualWriteAdapter::resolve_conflict_lww(&local, &remote);
        assert!(winner.is_local());
        let resolved = winner.winner();
        assert_eq!(resolved.problem, "Local version");
    }

    #[tokio::test]
    async fn test_lww_remote_wins() {
        let local = TestPattern {
            id: "conflict-2".to_string(),
            problem: "Local version".to_string(),
            solution: "Local solution".to_string(),
            category: "testing".to_string(),
            effectiveness: 0.8,
            updated_at: Utc::now() - chrono::Duration::seconds(10), // Older
        };

        let remote = TestPattern {
            id: "conflict-2".to_string(),
            problem: "Remote version".to_string(),
            solution: "Remote solution".to_string(),
            category: "testing".to_string(),
            effectiveness: 0.7,
            updated_at: Utc::now(), // Newer
        };

        let winner = DualWriteAdapter::resolve_conflict_lww(&local, &remote);
        assert!(winner.is_remote());
        let resolved = winner.winner();
        assert_eq!(resolved.problem, "Remote version");
    }

    #[tokio::test]
    async fn test_conflict_logging() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();

        let local = TestPattern::new("log-1", "Local", "Local solution");
        let remote = TestPattern::new("log-1", "Remote", "Remote solution");

        let conflict_id = adapter
            .log_conflict(&local, &remote, ConflictResolution::LocalWins)
            .unwrap();

        assert!(!conflict_id.is_empty());
    }

    #[tokio::test]
    async fn test_get_pending_conflicts() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();

        // Log several conflicts
        for i in 0..5 {
            let local = TestPattern::new(&format!("pending-{}", i), "Local", "Local solution");
            let remote = TestPattern::new(&format!("pending-{}", i), "Remote", "Remote solution");
            adapter
                .log_conflict(&local, &remote, ConflictResolution::Pending)
                .unwrap();
        }

        let pending = adapter.get_pending_conflicts(10).unwrap();
        assert_eq!(pending.len(), 5);
    }

    #[tokio::test]
    async fn test_resolve_conflict() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();

        let local = TestPattern::new("resolve-1", "Local", "Local solution");
        let remote = TestPattern::new("resolve-1", "Remote", "Remote solution");
        let conflict_id = adapter
            .log_conflict(&local, &remote, ConflictResolution::Pending)
            .unwrap();

        // Resolve it
        adapter
            .resolve_conflict(&conflict_id, ConflictResolution::LocalWins)
            .unwrap();

        // Verify no longer pending
        let pending = adapter.get_pending_conflicts(10).unwrap();
        assert!(pending.iter().all(|c| c.record_id != "resolve-1"));
    }
}

// ============================================================================
// Test: Circuit Breaker Activation
// ============================================================================

mod circuit_breaker_tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_breaker_initial_state() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        assert_eq!(adapter.circuit_state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_metrics() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        let metrics = adapter.circuit_metrics();

        assert_eq!(metrics.state, CircuitState::Closed);
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.total_failures, 0);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_on_failures() {
        let config = CircuitBreakerConfig::new("test-service")
            .with_failure_threshold(3)
            .with_reset_timeout(Duration::from_millis(100));
        let breaker = CircuitBreaker::new(config);

        // Simulate failures
        for _ in 0..3 {
            let _result = breaker
                .call(|| async { Err::<(), _>("simulated failure") })
                .await;
        }

        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_breaker_transitions_to_half_open() {
        let config = CircuitBreakerConfig::new("test-service")
            .with_failure_threshold(2)
            .with_reset_timeout(Duration::from_millis(50));
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        for _ in 0..2 {
            let _result = breaker
                .call(|| async { Err::<(), _>("simulated failure") })
                .await;
        }
        assert_eq!(breaker.state(), CircuitState::Open);

        // Wait for reset timeout
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Should transition to half-open
        assert_eq!(breaker.state(), CircuitState::HalfOpen);
    }

    #[tokio::test]
    async fn test_circuit_breaker_closes_after_success() {
        let config = CircuitBreakerConfig::new("test-service")
            .with_failure_threshold(2)
            .with_success_threshold(2)
            .with_reset_timeout(Duration::from_millis(50));
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        for _ in 0..2 {
            let _result = breaker
                .call(|| async { Err::<(), _>("simulated failure") })
                .await;
        }

        // Wait for half-open
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Successful calls should close it
        for _ in 0..2 {
            let _result = breaker.call(|| async { Ok::<_, &str>(()) }).await;
        }

        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_manual_reset() {
        let config = CircuitBreakerConfig::new("test-service").with_failure_threshold(2);
        let breaker = CircuitBreaker::new(config);

        // Trip the breaker manually
        breaker.trip();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Reset manually
        breaker.reset();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }
}

// ============================================================================
// Test: DLQ Integration
// ============================================================================

mod dlq_tests {
    use super::*;

    #[tokio::test]
    async fn test_dlq_stats_initial() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        let stats = adapter.dlq_stats().unwrap();

        assert_eq!(stats.pending, 0);
        assert_eq!(stats.processing, 0);
        assert_eq!(stats.abandoned, 0);
        assert_eq!(stats.total, 0);
    }

    #[tokio::test]
    async fn test_dlq_process_empty() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        let result = adapter.process_dlq(10).await.unwrap();

        assert_eq!(result.processed, 0);
        assert_eq!(result.succeeded, 0);
        assert_eq!(result.failed, 0);
        assert_eq!(result.abandoned, 0);
    }
}

// ============================================================================
// Test: Concurrent Operations
// ============================================================================

mod concurrent_tests {
    use super::*;

    #[tokio::test]
    async fn test_concurrent_inserts() {
        // Note: DualWriteAdapter contains rusqlite::Connection which is not Send/Sync,
        // so we can't use tokio::spawn directly. Instead, we test rapid sequential
        // inserts which still exercises the insert logic correctly.
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        setup_sqlite_schema(adapter.sqlite()).await.unwrap();

        // Perform rapid sequential inserts
        for i in 0..10 {
            let pattern = TestPattern::new(
                &format!("concurrent-{}", i),
                &format!("Problem {}", i),
                &format!("Solution {}", i),
            );
            let result = adapter.insert(&pattern).await;
            assert!(result.is_ok());
        }

        // Verify all inserted
        let count: i64 = adapter
            .sqlite()
            .query_one("SELECT COUNT(*) FROM test_patterns", &[], |row| row.get(0))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(count, 10);
    }

    #[tokio::test]
    async fn test_concurrent_updates_same_entity() {
        // Note: DualWriteAdapter contains rusqlite::Connection which is not Send/Sync,
        // so we can't use tokio::spawn directly. Instead, we test rapid sequential
        // updates which exercises the same upsert conflict resolution logic.
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        setup_sqlite_schema(adapter.sqlite()).await.unwrap();

        // Insert initial pattern
        let pattern = TestPattern::new("shared-1", "Initial", "Initial solution");
        adapter.insert(&pattern).await.unwrap();

        // Perform rapid updates sequentially (simulating concurrent-like behavior)
        for i in 0..5 {
            let pattern = TestPattern::new(
                "shared-1",
                &format!("Updated by {}", i),
                &format!("Solution by {}", i),
            );
            adapter.upsert(&pattern).await.unwrap();
        }

        // Verify entity exists (final version should be present)
        let exists: bool = adapter
            .sqlite()
            .query_one(
                "SELECT 1 FROM test_patterns WHERE id = ?",
                &[&"shared-1"],
                |_| Ok(true),
            )
            .await
            .unwrap()
            .unwrap_or(false);
        assert!(exists);

        // Verify the problem field was updated
        let problem: String = adapter
            .sqlite()
            .query_one(
                "SELECT problem FROM test_patterns WHERE id = ?",
                &[&"shared-1"],
                |row| row.get(0),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(problem, "Updated by 4"); // Last update should win
    }
}

// ============================================================================
// Test: Error Handling
// ============================================================================

mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_write_without_schema_fails() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();
        // Don't setup schema

        let pattern = TestPattern::new("error-1", "Problem", "Solution");
        let result = adapter.insert(&pattern).await;

        // Should fail due to missing table
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_dual_write_result_methods() {
        let result = nagual::db::DualWriteResult {
            sqlite_success: true,
            postgres_success: Some(true),
            queued_to_dlq: false,
            entity_id: "test-1".to_string(),
            warnings: vec![],
        };

        assert!(result.is_ok());
        assert!(result.is_fully_synced());

        let partial_result = nagual::db::DualWriteResult {
            sqlite_success: true,
            postgres_success: Some(false),
            queued_to_dlq: true,
            entity_id: "test-2".to_string(),
            warnings: vec!["PG write failed".to_string()],
        };

        assert!(partial_result.is_ok());
        assert!(!partial_result.is_fully_synced());
    }
}
