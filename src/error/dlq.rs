//! Dead Letter Queue (DLQ) for failed operations.
//!
//! Provides persistent storage for failed operations that can be retried later.
//! Uses SQLite for durability and supports:
//! - Enqueuing failed operations with metadata
//! - Dequeuing for retry processing
//! - Batch processing
//! - Automatic cleanup of abandoned entries

use std::path::Path;
use std::time::Duration;

use chrono::{DateTime, Utc};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::DlqError;

/// Entry in the dead letter queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlqEntry {
    /// Unique identifier for this entry.
    pub id: String,
    /// Name of the operation that failed.
    pub operation: String,
    /// JSON payload of the operation.
    pub payload: String,
    /// Error message from the last failure.
    pub error: String,
    /// Number of retry attempts made.
    pub attempts: u32,
    /// Maximum number of attempts before abandoning.
    pub max_attempts: u32,
    /// When the entry was created.
    pub created_at: DateTime<Utc>,
    /// When to next attempt retry.
    pub next_retry_at: DateTime<Utc>,
    /// Additional metadata.
    pub metadata: Option<String>,
}

impl DlqEntry {
    /// Create a new DLQ entry.
    pub fn new(
        operation: impl Into<String>,
        payload: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            operation: operation.into(),
            payload: payload.into(),
            error: error.into(),
            attempts: 0,
            max_attempts: 10,
            created_at: now,
            next_retry_at: now + chrono::Duration::minutes(1),
            metadata: None,
        }
    }

    /// Set the maximum number of attempts.
    pub fn with_max_attempts(mut self, max: u32) -> Self {
        self.max_attempts = max;
        self
    }

    /// Set metadata.
    pub fn with_metadata(mut self, metadata: impl Serialize) -> Result<Self, DlqError> {
        self.metadata = Some(
            serde_json::to_string(&metadata)
                .map_err(|e| DlqError::EnqueueFailed(e.to_string()))?,
        );
        Ok(self)
    }

    /// Check if this entry should be abandoned.
    pub fn should_abandon(&self) -> bool {
        self.attempts >= self.max_attempts
    }

    /// Check if this entry is ready for retry.
    pub fn is_ready_for_retry(&self) -> bool {
        Utc::now() >= self.next_retry_at
    }

    /// Calculate the next retry time using exponential backoff.
    /// Retry intervals: 1min, 5min, 15min, 1hr, 6hr (then capped)
    pub fn calculate_next_retry(&self) -> DateTime<Utc> {
        let delay_minutes = match self.attempts {
            0 => 1,
            1 => 5,
            2 => 15,
            3 => 60,
            _ => 360, // 6 hours
        };
        Utc::now() + chrono::Duration::minutes(delay_minutes)
    }
}

/// Dead Letter Queue backed by SQLite.
pub struct DeadLetterQueue {
    conn: Connection,
    max_size: Option<usize>,
}

impl DeadLetterQueue {
    /// Create a new DLQ at the specified path.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, DlqError> {
        let conn = Connection::open(path)?;
        let dlq = Self { conn, max_size: None };
        dlq.initialize_schema()?;
        Ok(dlq)
    }

    /// Create an in-memory DLQ (useful for testing).
    pub fn in_memory() -> Result<Self, DlqError> {
        let conn = Connection::open_in_memory()?;
        let dlq = Self { conn, max_size: None };
        dlq.initialize_schema()?;
        Ok(dlq)
    }

    /// Set the maximum number of entries in the queue.
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = Some(max_size);
        self
    }

    /// Initialize the database schema.
    fn initialize_schema(&self) -> Result<(), DlqError> {
        self.conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS dlq (
                id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                payload TEXT NOT NULL,
                error TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 10,
                created_at TEXT NOT NULL,
                next_retry_at TEXT NOT NULL,
                metadata TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                updated_at TEXT NOT NULL
            )
            "#,
            [],
        )?;

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dlq_status_next_retry ON dlq(status, next_retry_at)",
            [],
        )?;

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dlq_operation ON dlq(operation)",
            [],
        )?;

        info!("DLQ schema initialized");
        Ok(())
    }

    /// Enqueue a failed operation.
    pub fn enqueue(&self, entry: &DlqEntry) -> Result<String, DlqError> {
        // Check size limit
        if let Some(max_size) = self.max_size {
            let count: usize = self.conn.query_row(
                "SELECT COUNT(*) FROM dlq WHERE status = 'pending'",
                [],
                |row| row.get(0),
            )?;
            if count >= max_size {
                return Err(DlqError::Full { max_size });
            }
        }

        let now = Utc::now();
        self.conn.execute(
            r#"
            INSERT INTO dlq (id, operation, payload, error, attempts, max_attempts,
                           created_at, next_retry_at, metadata, status, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'pending', ?10)
            "#,
            params![
                entry.id,
                entry.operation,
                entry.payload,
                entry.error,
                entry.attempts,
                entry.max_attempts,
                entry.created_at.to_rfc3339(),
                entry.next_retry_at.to_rfc3339(),
                entry.metadata,
                now.to_rfc3339(),
            ],
        )?;

        debug!(
            id = %entry.id,
            operation = %entry.operation,
            "Enqueued entry to DLQ"
        );

        Ok(entry.id.clone())
    }

    /// Dequeue the next entry ready for retry.
    pub fn dequeue(&self) -> Result<Option<DlqEntry>, DlqError> {
        let now = Utc::now().to_rfc3339();

        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, operation, payload, error, attempts, max_attempts,
                   created_at, next_retry_at, metadata
            FROM dlq
            WHERE status = 'pending' AND next_retry_at <= ?1
            ORDER BY next_retry_at ASC
            LIMIT 1
            "#,
        )?;

        let entry = stmt.query_row([&now], |row| {
            let created_at: String = row.get(6)?;
            let next_retry_at: String = row.get(7)?;
            Ok(DlqEntry {
                id: row.get(0)?,
                operation: row.get(1)?,
                payload: row.get(2)?,
                error: row.get(3)?,
                attempts: row.get(4)?,
                max_attempts: row.get(5)?,
                created_at: DateTime::parse_from_rfc3339(&created_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                next_retry_at: DateTime::parse_from_rfc3339(&next_retry_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                metadata: row.get(8)?,
            })
        });

        match entry {
            Ok(e) => {
                // Mark as processing
                self.conn.execute(
                    "UPDATE dlq SET status = 'processing', updated_at = ?1 WHERE id = ?2",
                    params![Utc::now().to_rfc3339(), e.id],
                )?;
                Ok(Some(e))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(DlqError::DequeueFailed(e.to_string())),
        }
    }

    /// Mark an entry as successfully processed (removes it from the queue).
    pub fn mark_success(&self, id: &str) -> Result<(), DlqError> {
        self.conn.execute(
            "DELETE FROM dlq WHERE id = ?1",
            params![id],
        )?;
        debug!(id = %id, "DLQ entry marked as success and removed");
        Ok(())
    }

    /// Mark an entry as failed and schedule for retry.
    pub fn mark_failure(&self, id: &str, error: &str) -> Result<bool, DlqError> {
        // Get current entry
        let (attempts, max_attempts): (u32, u32) = self.conn.query_row(
            "SELECT attempts, max_attempts FROM dlq WHERE id = ?1",
            params![id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;

        let new_attempts = attempts + 1;
        let now = Utc::now();

        if new_attempts >= max_attempts {
            // Mark as abandoned
            self.conn.execute(
                "UPDATE dlq SET status = 'abandoned', attempts = ?1, error = ?2, updated_at = ?3 WHERE id = ?4",
                params![new_attempts, error, now.to_rfc3339(), id],
            )?;

            warn!(
                id = %id,
                attempts = new_attempts,
                "DLQ entry abandoned after max attempts"
            );

            return Ok(false); // Entry was abandoned
        }

        // Calculate next retry time
        let temp_entry = DlqEntry {
            id: id.to_string(),
            operation: String::new(),
            payload: String::new(),
            error: error.to_string(),
            attempts: new_attempts,
            max_attempts,
            created_at: now,
            next_retry_at: now,
            metadata: None,
        };
        let next_retry = temp_entry.calculate_next_retry();

        self.conn.execute(
            r#"
            UPDATE dlq
            SET status = 'pending', attempts = ?1, error = ?2,
                next_retry_at = ?3, updated_at = ?4
            WHERE id = ?5
            "#,
            params![
                new_attempts,
                error,
                next_retry.to_rfc3339(),
                now.to_rfc3339(),
                id
            ],
        )?;

        debug!(
            id = %id,
            attempts = new_attempts,
            next_retry = %next_retry,
            "DLQ entry scheduled for retry"
        );

        Ok(true) // Entry will be retried
    }

    /// Get entries ready for retry processing.
    pub fn get_ready_entries(&self, limit: usize) -> Result<Vec<DlqEntry>, DlqError> {
        let now = Utc::now().to_rfc3339();

        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, operation, payload, error, attempts, max_attempts,
                   created_at, next_retry_at, metadata
            FROM dlq
            WHERE status = 'pending' AND next_retry_at <= ?1
            ORDER BY next_retry_at ASC
            LIMIT ?2
            "#,
        )?;

        let entries = stmt.query_map([&now, &limit.to_string()], |row| {
            let created_at: String = row.get(6)?;
            let next_retry_at: String = row.get(7)?;
            Ok(DlqEntry {
                id: row.get(0)?,
                operation: row.get(1)?,
                payload: row.get(2)?,
                error: row.get(3)?,
                attempts: row.get(4)?,
                max_attempts: row.get(5)?,
                created_at: DateTime::parse_from_rfc3339(&created_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                next_retry_at: DateTime::parse_from_rfc3339(&next_retry_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                metadata: row.get(8)?,
            })
        })?;

        entries.collect::<Result<Vec<_>, _>>().map_err(DlqError::Database)
    }

    /// Process a batch of ready entries with a handler function.
    pub fn process_batch<F>(
        &self,
        batch_size: usize,
        handler: F,
    ) -> Result<BatchProcessResult, DlqError>
    where
        F: Fn(&DlqEntry) -> Result<(), String>,
    {
        let entries = self.get_ready_entries(batch_size)?;
        let mut result = BatchProcessResult::default();

        for entry in entries {
            // Mark as processing
            self.conn.execute(
                "UPDATE dlq SET status = 'processing', updated_at = ?1 WHERE id = ?2",
                params![Utc::now().to_rfc3339(), entry.id],
            )?;

            match handler(&entry) {
                Ok(()) => {
                    self.mark_success(&entry.id)?;
                    result.succeeded += 1;
                }
                Err(error) => {
                    let will_retry = self.mark_failure(&entry.id, &error)?;
                    if will_retry {
                        result.failed += 1;
                    } else {
                        result.abandoned += 1;
                    }
                }
            }
        }

        info!(
            succeeded = result.succeeded,
            failed = result.failed,
            abandoned = result.abandoned,
            "DLQ batch processing complete"
        );

        Ok(result)
    }

    /// Get statistics about the queue.
    pub fn stats(&self) -> Result<DlqStats, DlqError> {
        let pending: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM dlq WHERE status = 'pending'",
            [],
            |row| row.get(0),
        )?;

        let processing: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM dlq WHERE status = 'processing'",
            [],
            |row| row.get(0),
        )?;

        let abandoned: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM dlq WHERE status = 'abandoned'",
            [],
            |row| row.get(0),
        )?;

        let oldest_entry: Option<String> = self.conn.query_row(
            "SELECT MIN(created_at) FROM dlq WHERE status = 'pending'",
            [],
            |row| row.get(0),
        ).ok();

        Ok(DlqStats {
            pending,
            processing,
            abandoned,
            total: pending + processing + abandoned,
            oldest_entry: oldest_entry.and_then(|s| DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
        })
    }

    /// Clean up abandoned entries older than the specified duration.
    pub fn cleanup_abandoned(&self, older_than: Duration) -> Result<usize, DlqError> {
        let cutoff = Utc::now() - chrono::Duration::from_std(older_than)
            .expect("Duration should be within valid chrono range (less than ~292 billion years)");
        let deleted = self.conn.execute(
            "DELETE FROM dlq WHERE status = 'abandoned' AND updated_at < ?1",
            params![cutoff.to_rfc3339()],
        )?;
        info!(deleted = deleted, "Cleaned up abandoned DLQ entries");
        Ok(deleted)
    }

    /// Requeue a stuck processing entry (e.g., after worker crash).
    pub fn requeue_stuck(&self, stuck_threshold: Duration) -> Result<usize, DlqError> {
        let cutoff = Utc::now() - chrono::Duration::from_std(stuck_threshold)
            .expect("Duration should be within valid chrono range (less than ~292 billion years)");
        let requeued = self.conn.execute(
            "UPDATE dlq SET status = 'pending', updated_at = ?1 WHERE status = 'processing' AND updated_at < ?2",
            params![Utc::now().to_rfc3339(), cutoff.to_rfc3339()],
        )?;
        if requeued > 0 {
            info!(requeued = requeued, "Requeued stuck DLQ entries");
        }
        Ok(requeued)
    }

    /// Get entries by operation type.
    pub fn get_by_operation(&self, operation: &str, limit: usize) -> Result<Vec<DlqEntry>, DlqError> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, operation, payload, error, attempts, max_attempts,
                   created_at, next_retry_at, metadata
            FROM dlq
            WHERE operation = ?1 AND status = 'pending'
            ORDER BY created_at ASC
            LIMIT ?2
            "#,
        )?;

        let entries = stmt.query_map([operation, &limit.to_string()], |row| {
            let created_at: String = row.get(6)?;
            let next_retry_at: String = row.get(7)?;
            Ok(DlqEntry {
                id: row.get(0)?,
                operation: row.get(1)?,
                payload: row.get(2)?,
                error: row.get(3)?,
                attempts: row.get(4)?,
                max_attempts: row.get(5)?,
                created_at: DateTime::parse_from_rfc3339(&created_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                next_retry_at: DateTime::parse_from_rfc3339(&next_retry_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                metadata: row.get(8)?,
            })
        })?;

        entries.collect::<Result<Vec<_>, _>>().map_err(DlqError::Database)
    }
}

/// Result of batch processing.
#[derive(Debug, Default, Clone)]
pub struct BatchProcessResult {
    pub succeeded: usize,
    pub failed: usize,
    pub abandoned: usize,
}

impl BatchProcessResult {
    pub fn total(&self) -> usize {
        self.succeeded + self.failed + self.abandoned
    }
}

/// Statistics about the DLQ.
#[derive(Debug, Clone)]
pub struct DlqStats {
    pub pending: usize,
    pub processing: usize,
    pub abandoned: usize,
    pub total: usize,
    pub oldest_entry: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dlq_entry_creation() {
        let entry = DlqEntry::new("test_op", r#"{"key": "value"}"#, "test error");
        assert_eq!(entry.operation, "test_op");
        assert_eq!(entry.attempts, 0);
        assert_eq!(entry.max_attempts, 10);
        assert!(!entry.should_abandon());
    }

    #[test]
    fn test_dlq_entry_retry_intervals() {
        let mut entry = DlqEntry::new("test", "{}", "error");

        // Test increasing intervals
        let intervals = vec![1, 5, 15, 60, 360];
        for (attempt, expected_minutes) in intervals.into_iter().enumerate() {
            entry.attempts = attempt as u32;
            let next = entry.calculate_next_retry();
            let diff = next - Utc::now();
            // Allow some tolerance for test execution time
            assert!(
                diff.num_minutes() >= expected_minutes - 1 && diff.num_minutes() <= expected_minutes,
                "Expected ~{} minutes for attempt {}, got {} minutes",
                expected_minutes,
                attempt,
                diff.num_minutes()
            );
        }
    }

    #[test]
    fn test_dlq_enqueue_dequeue() {
        let dlq = DeadLetterQueue::in_memory().unwrap();

        let entry = DlqEntry::new("test_op", r#"{"id": 1}"#, "test error");
        let id = dlq.enqueue(&entry).unwrap();

        // Entry should not be immediately ready (next_retry_at is in the future)
        let dequeued = dlq.dequeue().unwrap();
        assert!(dequeued.is_none());

        // Manually update next_retry_at to now
        dlq.conn.execute(
            "UPDATE dlq SET next_retry_at = ?1 WHERE id = ?2",
            params![Utc::now().to_rfc3339(), id],
        ).unwrap();

        let dequeued = dlq.dequeue().unwrap();
        assert!(dequeued.is_some());
        assert_eq!(dequeued.unwrap().id, id);
    }

    #[test]
    fn test_dlq_mark_success() {
        let dlq = DeadLetterQueue::in_memory().unwrap();

        let entry = DlqEntry::new("test", "{}", "error");
        let id = dlq.enqueue(&entry).unwrap();

        dlq.mark_success(&id).unwrap();

        // Entry should be removed
        let stats = dlq.stats().unwrap();
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_dlq_mark_failure_with_retry() {
        let dlq = DeadLetterQueue::in_memory().unwrap();

        let entry = DlqEntry::new("test", "{}", "error").with_max_attempts(5);
        let id = dlq.enqueue(&entry).unwrap();

        // First failure - should schedule retry
        let will_retry = dlq.mark_failure(&id, "new error").unwrap();
        assert!(will_retry);

        let stats = dlq.stats().unwrap();
        assert_eq!(stats.pending, 1);
    }

    #[test]
    fn test_dlq_mark_failure_abandoned() {
        let dlq = DeadLetterQueue::in_memory().unwrap();

        let entry = DlqEntry::new("test", "{}", "error").with_max_attempts(2);
        let id = dlq.enqueue(&entry).unwrap();

        // First failure
        dlq.mark_failure(&id, "error 1").unwrap();
        // Second failure - should abandon
        let will_retry = dlq.mark_failure(&id, "error 2").unwrap();
        assert!(!will_retry);

        let stats = dlq.stats().unwrap();
        assert_eq!(stats.abandoned, 1);
        assert_eq!(stats.pending, 0);
    }

    #[test]
    fn test_dlq_max_size() {
        let dlq = DeadLetterQueue::in_memory().unwrap().with_max_size(2);

        let entry1 = DlqEntry::new("test1", "{}", "error");
        let entry2 = DlqEntry::new("test2", "{}", "error");
        let entry3 = DlqEntry::new("test3", "{}", "error");

        dlq.enqueue(&entry1).unwrap();
        dlq.enqueue(&entry2).unwrap();
        let result = dlq.enqueue(&entry3);

        assert!(matches!(result, Err(DlqError::Full { max_size: 2 })));
    }

    #[test]
    fn test_dlq_stats() {
        let dlq = DeadLetterQueue::in_memory().unwrap();

        for i in 0..5 {
            let entry = DlqEntry::new(format!("op_{}", i), "{}", "error");
            dlq.enqueue(&entry).unwrap();
        }

        let stats = dlq.stats().unwrap();
        assert_eq!(stats.pending, 5);
        assert_eq!(stats.total, 5);
    }

    #[test]
    fn test_dlq_get_by_operation() {
        let dlq = DeadLetterQueue::in_memory().unwrap();

        for i in 0..3 {
            let entry = DlqEntry::new("type_a", format!("{}", i), "error");
            dlq.enqueue(&entry).unwrap();
        }
        for i in 0..2 {
            let entry = DlqEntry::new("type_b", format!("{}", i), "error");
            dlq.enqueue(&entry).unwrap();
        }

        let type_a = dlq.get_by_operation("type_a", 10).unwrap();
        assert_eq!(type_a.len(), 3);

        let type_b = dlq.get_by_operation("type_b", 10).unwrap();
        assert_eq!(type_b.len(), 2);
    }
}
