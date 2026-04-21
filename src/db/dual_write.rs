//! Dual-Write Adapter for synchronized SQLite and PostgreSQL writes.
//!
//! Implements the dual-write pattern where:
//! - SQLite writes are synchronous and return immediately
//! - PostgreSQL writes happen asynchronously via `tokio::spawn`
//! - Circuit breaker protects against cascading PG failures
//! - DLQ captures failed PG writes for later retry
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐
//! │   Entity    │────▶│DualWriteAdapter│
//! └─────────────┘     └──────┬──────┘
//!                            │
//!            ┌───────────────┼───────────────┐
//!            ▼               ▼               ▼
//!     ┌──────────┐    ┌──────────┐    ┌──────────┐
//!     │  SQLite  │    │  Circuit │    │   DLQ    │
//!     │  (sync)  │    │  Breaker │    │(failures)│
//!     └──────────┘    └────┬─────┘    └──────────┘
//!                          │
//!                          ▼
//!                   ┌──────────────┐
//!                   │ PostgreSQL   │
//!                   │ (background) │
//!                   └──────────────┘
//! ```

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{debug, error, info, warn};

use crate::error::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitState, DeadLetterQueue,
    DlqEntry, DlqError, Result,
};
use crate::sync::pii::global_redactor;

use super::{PostgresDb, SqliteDb};

// Re-export for convenience
pub use super::conflicts::{ConflictLog, ConflictLogEntry, ConflictResolution};

/// Operation types for dual-write operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationType {
    /// Insert a new record
    Insert,
    /// Update an existing record
    Update,
    /// Delete a record
    Delete,
    /// Upsert (insert or update)
    Upsert,
}

impl std::fmt::Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationType::Insert => write!(f, "insert"),
            OperationType::Update => write!(f, "update"),
            OperationType::Delete => write!(f, "delete"),
            OperationType::Upsert => write!(f, "upsert"),
        }
    }
}

/// Metadata for a DLQ entry containing information about the failed operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualWriteDlqMeta {
    /// The operation type that failed
    pub operation: OperationType,
    /// The table name
    pub table_name: String,
    /// The entity's unique identifier
    pub entity_id: String,
    /// Timestamp when the operation was attempted
    pub attempted_at: DateTime<Utc>,
    /// Updated_at timestamp for conflict resolution
    pub updated_at: DateTime<Utc>,
}

/// Trait for entities that can be dual-written to both SQLite and PostgreSQL.
///
/// Implementing this trait enables automatic synchronization between the local
/// SQLite database and cloud PostgreSQL with conflict resolution support.
#[async_trait]
pub trait DualWritable: Serialize + DeserializeOwned + Send + Sync + Clone + 'static {
    /// The unique identifier type for this entity.
    type Id: ToString + Clone + Send + Sync;

    /// The table name in the database.
    fn table_name() -> &'static str;

    /// Get the unique identifier for this entity.
    fn id(&self) -> Self::Id;

    /// Get the last updated timestamp for conflict resolution.
    fn updated_at(&self) -> DateTime<Utc>;

    /// Set the updated timestamp (for conflict resolution).
    fn set_updated_at(&mut self, ts: DateTime<Utc>);

    /// Generate SQL for inserting into SQLite.
    fn sqlite_insert_sql() -> &'static str;

    /// Generate SQL for updating in SQLite.
    fn sqlite_update_sql() -> &'static str;

    /// Generate SQL for deleting from SQLite.
    fn sqlite_delete_sql() -> &'static str {
        // Default implementation - can be overridden
        "DELETE FROM {table} WHERE id = ?"
    }

    /// Bind parameters for SQLite insert.
    fn sqlite_insert_params(&self) -> Vec<Box<dyn rusqlite::ToSql + Send + Sync>>;

    /// Bind parameters for SQLite update.
    fn sqlite_update_params(&self) -> Vec<Box<dyn rusqlite::ToSql + Send + Sync>>;

    /// Bind parameters for SQLite delete.
    fn sqlite_delete_params(&self) -> Vec<Box<dyn rusqlite::ToSql + Send + Sync>> {
        vec![Box::new(self.id().to_string())]
    }

    /// Generate SQL for inserting into PostgreSQL.
    fn postgres_insert_sql() -> &'static str;

    /// Generate SQL for updating in PostgreSQL.
    fn postgres_update_sql() -> &'static str;

    /// Generate SQL for deleting from PostgreSQL.
    fn postgres_delete_sql() -> &'static str {
        "DELETE FROM {table} WHERE id = $1"
    }

    /// Execute PostgreSQL insert with bound parameters.
    async fn postgres_insert(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error>;

    /// Execute PostgreSQL update with bound parameters.
    async fn postgres_update(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error>;

    /// Execute PostgreSQL delete with bound parameters.
    async fn postgres_delete(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error>;
}

/// Result of a dual-write operation.
#[derive(Debug, Clone)]
pub struct DualWriteResult {
    /// Whether SQLite write succeeded
    pub sqlite_success: bool,
    /// Whether PostgreSQL write succeeded (None if PG not configured)
    pub postgres_success: Option<bool>,
    /// Whether the write was queued to DLQ
    pub queued_to_dlq: bool,
    /// The entity ID that was written
    pub entity_id: String,
    /// Any warning messages
    pub warnings: Vec<String>,
}

impl DualWriteResult {
    /// Check if the overall operation succeeded (at least SQLite succeeded).
    pub fn is_ok(&self) -> bool {
        self.sqlite_success
    }

    /// Check if both databases succeeded.
    pub fn is_fully_synced(&self) -> bool {
        self.sqlite_success && self.postgres_success.unwrap_or(true)
    }
}

/// Configuration for the DualWriteAdapter.
#[derive(Debug, Clone)]
pub struct DualWriteConfig {
    /// Number of consecutive PG failures before opening circuit breaker.
    pub circuit_breaker_threshold: u32,
    /// Duration to keep circuit breaker open before half-open.
    pub circuit_breaker_reset_timeout: Duration,
    /// Maximum size of the DLQ.
    pub dlq_max_size: Option<usize>,
    /// Path to DLQ database file.
    pub dlq_path: String,
    /// Whether to log conflicts automatically.
    pub auto_log_conflicts: bool,
}

impl Default for DualWriteConfig {
    fn default() -> Self {
        Self {
            circuit_breaker_threshold: 5,
            circuit_breaker_reset_timeout: Duration::from_secs(30),
            dlq_max_size: Some(10000),
            dlq_path: "nagual_dlq.db".to_string(),
            auto_log_conflicts: true,
        }
    }
}

/// Dual-Write Adapter for synchronized database writes.
///
/// This adapter ensures writes go to both SQLite (local) and PostgreSQL (cloud)
/// with proper failure handling via circuit breaker and DLQ patterns.
pub struct DualWriteAdapter {
    /// Local SQLite database (always available)
    sqlite: Arc<SqliteDb>,
    /// Cloud PostgreSQL database (optional)
    postgres: Option<Arc<PostgresDb>>,
    /// Circuit breaker for PostgreSQL
    circuit_breaker: Arc<CircuitBreaker>,
    /// Dead letter queue for failed operations
    dlq: Arc<parking_lot::Mutex<DeadLetterQueue>>,
    /// Conflict logging
    conflict_log: Arc<parking_lot::Mutex<ConflictLog>>,
    /// Configuration
    config: DualWriteConfig,
}

impl DualWriteAdapter {
    /// Create a new DualWriteAdapter.
    pub fn new(
        sqlite: Arc<SqliteDb>,
        postgres: Option<Arc<PostgresDb>>,
        config: DualWriteConfig,
    ) -> Result<Self> {
        let circuit_breaker = Arc::new(CircuitBreaker::new(
            CircuitBreakerConfig::new("postgresql")
                .with_failure_threshold(config.circuit_breaker_threshold)
                .with_reset_timeout(config.circuit_breaker_reset_timeout)
                .with_success_threshold(3)
                .with_half_open_max_requests(2),
        ));

        let dlq = DeadLetterQueue::new(&config.dlq_path)?;
        let dlq = if let Some(max_size) = config.dlq_max_size {
            dlq.with_max_size(max_size)
        } else {
            dlq
        };

        let conflict_log = ConflictLog::new(&config.dlq_path.replace(".db", "_conflicts.db"))?;

        info!(
            sqlite_path = %sqlite.path(),
            postgres_configured = postgres.is_some(),
            circuit_breaker_threshold = config.circuit_breaker_threshold,
            "DualWriteAdapter initialized"
        );

        Ok(Self {
            sqlite,
            postgres,
            circuit_breaker,
            dlq: Arc::new(parking_lot::Mutex::new(dlq)),
            conflict_log: Arc::new(parking_lot::Mutex::new(conflict_log)),
            config,
        })
    }

    /// Create a DualWriteAdapter for testing (in-memory databases).
    pub fn new_for_testing() -> Result<Self> {
        let sqlite = Arc::new(SqliteDb::open_in_memory()?);
        let dlq = DeadLetterQueue::in_memory()?;
        let conflict_log = ConflictLog::in_memory()?;

        let circuit_breaker = Arc::new(CircuitBreaker::new(
            CircuitBreakerConfig::new("postgresql")
                .with_failure_threshold(5)
                .with_reset_timeout(Duration::from_secs(30)),
        ));

        Ok(Self {
            sqlite,
            postgres: None,
            circuit_breaker,
            dlq: Arc::new(parking_lot::Mutex::new(dlq)),
            conflict_log: Arc::new(parking_lot::Mutex::new(conflict_log)),
            config: DualWriteConfig::default(),
        })
    }

    /// Get the circuit breaker state.
    pub fn circuit_state(&self) -> CircuitState {
        self.circuit_breaker.state()
    }

    /// Get the circuit breaker metrics.
    pub fn circuit_metrics(&self) -> crate::error::CircuitBreakerMetrics {
        self.circuit_breaker.metrics()
    }

    /// Check if PostgreSQL is available (configured and circuit not open).
    pub fn is_postgres_available(&self) -> bool {
        self.postgres.is_some() && self.circuit_breaker.is_allowing_requests()
    }

    /// Get the DLQ statistics.
    pub fn dlq_stats(&self) -> std::result::Result<crate::error::DlqStats, DlqError> {
        self.dlq.lock().stats()
    }

    /// Write an entity to both databases.
    ///
    /// This is the main entry point for dual-write operations.
    pub async fn write<E: DualWritable>(
        &self,
        entity: &E,
        op: OperationType,
    ) -> Result<DualWriteResult> {
        let entity_id = entity.id().to_string();
        let mut result = DualWriteResult {
            sqlite_success: false,
            postgres_success: None,
            queued_to_dlq: false,
            entity_id: entity_id.clone(),
            warnings: Vec::new(),
        };

        // Step 1: Synchronous SQLite write (must succeed)
        match self.write_sqlite(entity, op).await {
            Ok(()) => {
                result.sqlite_success = true;
                debug!(
                    entity_id = %entity_id,
                    operation = %op,
                    table = E::table_name(),
                    "SQLite write succeeded"
                );
            }
            Err(e) => {
                error!(
                    entity_id = %entity_id,
                    operation = %op,
                    error = %e,
                    "SQLite write failed"
                );
                return Err(e);
            }
        }

        // Step 2: Async PostgreSQL write (if configured)
        if let Some(ref postgres) = self.postgres {
            let pg = postgres.clone();
            let cb = self.circuit_breaker.clone();
            let dlq = self.dlq.clone();
            let config_auto_log = self.config.auto_log_conflicts;
            let conflict_log = self.conflict_log.clone();

            // PII redaction: create a redacted copy for the cloud-bound PG write.
            // Local SQLite data is NEVER modified.
            let entity_clone = Self::redact_entity_for_cloud(entity);
            let entity_id_clone = entity_id.clone();

            // Spawn background task for PostgreSQL write
            tokio::spawn(async move {
                Self::write_postgres_with_circuit_breaker(
                    &pg,
                    &cb,
                    &dlq,
                    &conflict_log,
                    &entity_clone,
                    op,
                    config_auto_log,
                )
                .await;
            });

            result.postgres_success = Some(true); // Optimistic - actual result is async
            debug!(
                entity_id = %entity_id_clone,
                "PostgreSQL write spawned in background"
            );
        }

        Ok(result)
    }

    /// Insert an entity into both databases.
    pub async fn insert<E: DualWritable>(&self, entity: &E) -> Result<DualWriteResult> {
        self.write(entity, OperationType::Insert).await
    }

    /// Update an entity in both databases.
    pub async fn update<E: DualWritable>(&self, entity: &E) -> Result<DualWriteResult> {
        self.write(entity, OperationType::Update).await
    }

    /// Delete an entity from both databases.
    pub async fn delete<E: DualWritable>(&self, entity: &E) -> Result<DualWriteResult> {
        self.write(entity, OperationType::Delete).await
    }

    /// Upsert an entity (insert or update) in both databases.
    pub async fn upsert<E: DualWritable>(&self, entity: &E) -> Result<DualWriteResult> {
        self.write(entity, OperationType::Upsert).await
    }

    /// Create a PII-redacted clone of an entity for cloud-bound writes.
    ///
    /// Serializes the entity to JSON, strips PII from known text fields
    /// (`problem`, `solution`, `context`, `critique`), and deserializes back.
    /// If redaction fails (e.g., serialization error), returns the original
    /// entity unmodified — safety over availability.
    fn redact_entity_for_cloud<E: DualWritable>(entity: &E) -> E {
        let redactor = global_redactor();

        // Serialize entity to mutable JSON value
        let mut json = match serde_json::to_value(entity) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "Failed to serialize entity for PII redaction, using original");
                return entity.clone();
            }
        };

        // Redact known text fields (reasoning_patterns schema).
        // Includes `title` and `summary` which can contain free-form text
        // derived from problem/solution content.
        let text_fields = ["problem", "solution", "context", "critique", "title", "summary"];
        let mut total_redactions = 0usize;

        if let Some(obj) = json.as_object_mut() {
            for field in &text_fields {
                if let Some(serde_json::Value::String(val)) = obj.get(*field) {
                    let result = redactor.strip_pii(val);
                    if result.redactions_count > 0 {
                        total_redactions += result.redactions_count;
                        obj.insert(field.to_string(), serde_json::Value::String(result.text));
                    }
                }
            }
        }

        if total_redactions > 0 {
            debug!(
                total_redactions = total_redactions,
                "PII stripped from entity before PG write"
            );
        }

        // Deserialize back to the entity type
        match serde_json::from_value(json) {
            Ok(redacted) => redacted,
            Err(e) => {
                warn!(error = %e, "Failed to deserialize PII-redacted entity, using original");
                entity.clone()
            }
        }
    }

    /// Write to SQLite synchronously.
    async fn write_sqlite<E: DualWritable>(&self, entity: &E, op: OperationType) -> Result<()> {
        let params = match op {
            OperationType::Insert | OperationType::Upsert => entity.sqlite_insert_params(),
            OperationType::Update => entity.sqlite_update_params(),
            OperationType::Delete => entity.sqlite_delete_params(),
        };

        let sql = match op {
            OperationType::Insert | OperationType::Upsert => E::sqlite_insert_sql(),
            OperationType::Update => E::sqlite_update_sql(),
            OperationType::Delete => E::sqlite_delete_sql(),
        };

        // Use with_connection to keep non-Send &dyn ToSql refs inside
        // the synchronous closure (they must not live across .await).
        self.sqlite
            .with_connection(|conn| {
                let param_refs: Vec<&dyn rusqlite::ToSql> = params
                    .iter()
                    .map(|p| p.as_ref() as &dyn rusqlite::ToSql)
                    .collect();
                conn.execute(sql, &param_refs[..])
                    .map_err(super::DatabaseError::from)?;
                Ok(())
            })
            .await
    }

    /// Write to PostgreSQL with circuit breaker protection.
    async fn write_postgres_with_circuit_breaker<E: DualWritable>(
        postgres: &PostgresDb,
        circuit_breaker: &CircuitBreaker,
        dlq: &parking_lot::Mutex<DeadLetterQueue>,
        _conflict_log: &parking_lot::Mutex<ConflictLog>,
        entity: &E,
        op: OperationType,
        _auto_log_conflicts: bool,
    ) {
        let entity_id = entity.id().to_string();
        let pool = postgres.pool();

        // Attempt write through circuit breaker
        let result = circuit_breaker
            .call(|| async {
                match op {
                    OperationType::Insert | OperationType::Upsert => {
                        entity.postgres_insert(pool).await
                    }
                    OperationType::Update => entity.postgres_update(pool).await,
                    OperationType::Delete => entity.postgres_delete(pool).await,
                }
            })
            .await;

        match result {
            Ok(()) => {
                debug!(
                    entity_id = %entity_id,
                    operation = %op,
                    table = E::table_name(),
                    "PostgreSQL write succeeded"
                );
            }
            Err(CircuitBreakerError::Open {
                service,
                retry_after_ms,
            }) => {
                warn!(
                    service = %service,
                    retry_after_ms = retry_after_ms,
                    entity_id = %entity_id,
                    "Circuit breaker OPEN - enqueueing to DLQ"
                );

                Self::enqueue_to_dlq(dlq, entity, op, "Circuit breaker open").await;
            }
            Err(CircuitBreakerError::HalfOpen { service }) => {
                warn!(
                    service = %service,
                    entity_id = %entity_id,
                    "Circuit breaker HALF_OPEN - PostgreSQL write failed, enqueueing to DLQ"
                );

                Self::enqueue_to_dlq(dlq, entity, op, "Write failed during half-open state").await;
            }
        }
    }

    /// Enqueue a failed operation to the DLQ.
    async fn enqueue_to_dlq<E: DualWritable>(
        dlq: &parking_lot::Mutex<DeadLetterQueue>,
        entity: &E,
        op: OperationType,
        error_msg: &str,
    ) {
        let entity_id = entity.id().to_string();
        let now = Utc::now();

        // Serialize entity to JSON
        let payload = match serde_json::to_string(entity) {
            Ok(json) => json,
            Err(e) => {
                error!(
                    entity_id = %entity_id,
                    error = %e,
                    "Failed to serialize entity for DLQ"
                );
                return;
            }
        };

        let meta = DualWriteDlqMeta {
            operation: op,
            table_name: E::table_name().to_string(),
            entity_id: entity_id.clone(),
            attempted_at: now,
            updated_at: entity.updated_at(),
        };

        let operation_name = format!("dual_write:{}:{}", E::table_name(), op);
        let entry = DlqEntry::new(&operation_name, &payload, error_msg);

        // Add metadata and enqueue
        let entry_to_enqueue = match entry.with_metadata(&meta) {
            Ok(e) => e,
            Err(e) => {
                error!(
                    entity_id = %entity_id,
                    error = %e,
                    "Failed to add metadata to DLQ entry"
                );
                // Create a new entry without metadata
                DlqEntry::new(&operation_name, &payload, error_msg)
            }
        };

        // Enqueue (don't block on failures)
        match dlq.lock().enqueue(&entry_to_enqueue) {
            Ok(id) => {
                info!(
                    dlq_id = %id,
                    entity_id = %entity_id,
                    operation = %op,
                    "Operation enqueued to DLQ"
                );
            }
            Err(e) => {
                error!(
                    entity_id = %entity_id,
                    error = %e,
                    "Failed to enqueue to DLQ - operation will be lost!"
                );
            }
        }
    }

    /// Resolve a conflict using Last-Write-Wins strategy.
    ///
    /// Compares `updated_at` timestamps and returns the winning version.
    pub fn resolve_conflict_lww<E: DualWritable>(local: &E, remote: &E) -> ConflictWinner<E> {
        let local_ts = local.updated_at();
        let remote_ts = remote.updated_at();

        if local_ts >= remote_ts {
            info!(
                local_ts = %local_ts,
                remote_ts = %remote_ts,
                "Conflict resolved: local wins (LWW)"
            );
            ConflictWinner::Local(local.clone())
        } else {
            info!(
                local_ts = %local_ts,
                remote_ts = %remote_ts,
                "Conflict resolved: remote wins (LWW)"
            );
            ConflictWinner::Remote(remote.clone())
        }
    }

    /// Log a conflict for manual review.
    pub fn log_conflict<E: DualWritable>(
        &self,
        local: &E,
        remote: &E,
        resolution: ConflictResolution,
    ) -> Result<String> {
        let local_json = serde_json::to_value(local)?;
        let remote_json = serde_json::to_value(remote)?;

        let entry = ConflictLogEntry::new(
            E::table_name(),
            &local.id().to_string(),
            local_json,
            remote_json,
            resolution,
        );

        let id = self.conflict_log.lock().log(&entry)?;
        info!(
            conflict_id = %id,
            table = E::table_name(),
            entity_id = %local.id().to_string(),
            resolution = ?resolution,
            "Conflict logged"
        );

        Ok(id)
    }

    /// Get pending conflicts from the conflict log.
    pub fn get_pending_conflicts(&self, limit: usize) -> Result<Vec<ConflictLogEntry>> {
        Ok(self.conflict_log.lock().get_pending(limit)?)
    }

    /// Resolve a conflict by ID.
    pub fn resolve_conflict(
        &self,
        conflict_id: &str,
        resolution: ConflictResolution,
    ) -> Result<()> {
        self.conflict_log.lock().resolve(conflict_id, resolution)?;
        info!(conflict_id = %conflict_id, resolution = ?resolution, "Conflict resolved");
        Ok(())
    }

    /// Process DLQ entries and retry failed PostgreSQL writes.
    pub async fn process_dlq(&self, batch_size: usize) -> Result<DlqProcessResult> {
        let postgres = match &self.postgres {
            Some(pg) => pg.clone(),
            None => {
                return Ok(DlqProcessResult {
                    processed: 0,
                    succeeded: 0,
                    failed: 0,
                    abandoned: 0,
                });
            }
        };

        // Check circuit breaker state
        if !self.circuit_breaker.is_allowing_requests() {
            debug!("Circuit breaker open, skipping DLQ processing");
            return Ok(DlqProcessResult {
                processed: 0,
                succeeded: 0,
                failed: 0,
                abandoned: 0,
            });
        }

        let entries = self.dlq.lock().get_ready_entries(batch_size)?;
        let mut result = DlqProcessResult {
            processed: entries.len(),
            succeeded: 0,
            failed: 0,
            abandoned: 0,
        };

        for entry in entries {
            // Parse metadata to determine operation type
            let meta: Option<DualWriteDlqMeta> = entry
                .metadata
                .as_ref()
                .and_then(|m| serde_json::from_str(m).ok());

            if meta.is_none() {
                warn!(
                    dlq_id = %entry.id,
                    "DLQ entry has no valid metadata, marking as failure"
                );
                let _ = self.dlq.lock().mark_failure(&entry.id, "Invalid metadata");
                result.failed += 1;
                continue;
            }

            let meta = meta.unwrap();

            // Replay the operation through circuit breaker
            let pool = postgres.pool();
            let payload = entry.payload.clone();
            let table_name = meta.table_name.clone();
            let operation = meta.operation;

            let replay_result = self
                .circuit_breaker
                .call(|| async {
                    Self::replay_operation(pool, &table_name, &payload, operation).await
                })
                .await;

            match replay_result {
                Ok(()) => {
                    self.dlq.lock().mark_success(&entry.id)?;
                    result.succeeded += 1;
                    debug!(dlq_id = %entry.id, "DLQ entry replayed successfully");
                }
                Err(_) => {
                    let will_retry = self.dlq.lock().mark_failure(&entry.id, "Replay failed")?;
                    if will_retry {
                        result.failed += 1;
                    } else {
                        result.abandoned += 1;
                        warn!(dlq_id = %entry.id, "DLQ entry abandoned after max retries");
                    }
                }
            }
        }

        info!(
            processed = result.processed,
            succeeded = result.succeeded,
            failed = result.failed,
            abandoned = result.abandoned,
            "DLQ processing complete"
        );

        Ok(result)
    }

    /// Get the underlying SQLite database.
    pub fn sqlite(&self) -> &Arc<SqliteDb> {
        &self.sqlite
    }

    /// Get the underlying PostgreSQL database (if configured).
    pub fn postgres(&self) -> Option<&Arc<PostgresDb>> {
        self.postgres.as_ref()
    }

    /// Get the conflict log.
    pub fn conflict_log(&self) -> &Arc<parking_lot::Mutex<ConflictLog>> {
        &self.conflict_log
    }

    /// Replay a DLQ operation by deserializing the entity and calling the appropriate PostgreSQL method.
    ///
    /// This function handles dynamic dispatch based on table name since Rust requires
    /// compile-time type information. Each registered entity type is matched and replayed.
    async fn replay_operation(
        pool: &PgPool,
        table_name: &str,
        payload: &str,
        operation: OperationType,
    ) -> std::result::Result<(), sqlx::Error> {
        match table_name {
            "reasoning_patterns" => Self::replay_reasoning_pattern(pool, payload, operation).await,
            "test_entities" => {
                // Test entities are only used in unit tests, skip in production
                debug!(table_name = %table_name, "Skipping test entity replay");
                Ok(())
            }
            _ => {
                // Unknown table - log warning and succeed to clear from DLQ
                // This prevents unknown tables from blocking the queue indefinitely
                warn!(
                    table_name = %table_name,
                    "Unknown table in DLQ entry, cannot replay - marking as processed"
                );
                Ok(())
            }
        }
    }

    /// Replay a reasoning pattern operation to PostgreSQL.
    async fn replay_reasoning_pattern(
        pool: &PgPool,
        payload: &str,
        operation: OperationType,
    ) -> std::result::Result<(), sqlx::Error> {
        // Deserialize the pattern from the DLQ payload
        // The payload is the serialized StorablePattern (which wraps a Pattern)
        let pattern_data: serde_json::Value =
            serde_json::from_str(payload).map_err(|e| sqlx::Error::Decode(Box::new(e)))?;

        // Extract fields from the JSON payload
        let id = pattern_data["id"]
            .as_str()
            .ok_or_else(|| sqlx::Error::Decode("Missing id field".into()))?;

        match operation {
            OperationType::Delete => {
                sqlx::query("DELETE FROM reasoning_patterns WHERE id = $1")
                    .bind(id)
                    .execute(pool)
                    .await?;

                debug!(pattern_id = %id, "Replayed DELETE for reasoning_patterns");
            }
            OperationType::Insert | OperationType::Update | OperationType::Upsert => {
                // Extract all fields for insert/update/upsert
                let timestamp = pattern_data["timestamp"]
                    .as_str()
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(Utc::now);

                let updated_at = pattern_data["updated_at"]
                    .as_str()
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(Utc::now);

                let category = pattern_data["category"].as_str().unwrap_or("general");

                let raw_problem = pattern_data["problem"].as_str().unwrap_or("");
                let raw_solution = pattern_data["solution"].as_str().unwrap_or("");
                let raw_context = pattern_data["context"].as_str().unwrap_or("");
                let raw_critique = pattern_data["critique"].as_str().unwrap_or("");

                // PII redaction: strip sensitive data from cloud-bound text fields.
                // Local SQLite is NEVER modified — only the PG copy is redacted.
                let redactor = global_redactor();
                let problem_r = redactor.strip_pii(raw_problem);
                let solution_r = redactor.strip_pii(raw_solution);
                let context_r = redactor.strip_pii(raw_context);
                let critique_r = redactor.strip_pii(raw_critique);

                let total_redactions = problem_r.redactions_count
                    + solution_r.redactions_count
                    + context_r.redactions_count
                    + critique_r.redactions_count;

                if total_redactions > 0 {
                    debug!(
                        pattern_id = %id,
                        total_redactions = total_redactions,
                        "PII stripped from pattern before PG replay"
                    );
                }

                let problem = problem_r.text.as_str();
                let solution = solution_r.text.as_str();
                let context = context_r.text.as_str();
                let critique = critique_r.text.as_str();

                let effectiveness = pattern_data["effectiveness"].as_f64().unwrap_or(0.5);

                let reuse_count = pattern_data["reuse_count"].as_i64().unwrap_or(0) as i32;

                let reward = pattern_data["reward"].as_f64().unwrap_or(0.5);

                let success = pattern_data["success"].as_bool().unwrap_or(true);

                let agent_id = pattern_data["agent_id"].as_str().map(|s| s.to_string());

                let session_id = pattern_data["session_id"].as_str().map(|s| s.to_string());

                let confidence = pattern_data["confidence"].as_f64().unwrap_or(0.5);

                // Handle embedding - may be array or null
                let embedding: Option<Vec<f32>> = pattern_data["embedding"].as_array().map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                });

                // Handle tags, related_patterns, and metadata as JSON values
                let tags = pattern_data
                    .get("tags")
                    .cloned()
                    .unwrap_or(serde_json::json!([]));

                let related_patterns = pattern_data
                    .get("related_patterns")
                    .cloned()
                    .unwrap_or(serde_json::json!([]));

                let metadata = pattern_data
                    .get("metadata")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));

                // Use upsert query for all insert/update/upsert operations
                // This is safe and idempotent
                sqlx::query(
                    r#"
                    INSERT INTO reasoning_patterns (
                        id, timestamp, updated_at, category, problem, solution, context,
                        effectiveness, reuse_count, reward, success, critique,
                        agent_id, session_id, confidence, embedding, tags,
                        related_patterns, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                    ON CONFLICT (id) DO UPDATE SET
                        updated_at = EXCLUDED.updated_at,
                        category = EXCLUDED.category,
                        problem = EXCLUDED.problem,
                        solution = EXCLUDED.solution,
                        context = EXCLUDED.context,
                        effectiveness = EXCLUDED.effectiveness,
                        reuse_count = EXCLUDED.reuse_count,
                        reward = EXCLUDED.reward,
                        success = EXCLUDED.success,
                        critique = EXCLUDED.critique,
                        agent_id = EXCLUDED.agent_id,
                        session_id = EXCLUDED.session_id,
                        confidence = EXCLUDED.confidence,
                        embedding = EXCLUDED.embedding,
                        tags = EXCLUDED.tags,
                        related_patterns = EXCLUDED.related_patterns,
                        metadata = EXCLUDED.metadata
                    "#,
                )
                .bind(id)
                .bind(timestamp)
                .bind(updated_at)
                .bind(category)
                .bind(problem)
                .bind(solution)
                .bind(context)
                .bind(effectiveness)
                .bind(reuse_count)
                .bind(reward)
                .bind(success)
                .bind(critique)
                .bind(&agent_id)
                .bind(&session_id)
                .bind(confidence)
                .bind(&embedding)
                .bind(&tags)
                .bind(&related_patterns)
                .bind(&metadata)
                .execute(pool)
                .await?;

                debug!(
                    pattern_id = %id,
                    operation = %operation,
                    "Replayed {} for reasoning_patterns",
                    operation
                );
            }
        }

        Ok(())
    }
}

/// Result of conflict resolution.
#[derive(Debug, Clone)]
pub enum ConflictWinner<E> {
    /// Local version wins
    Local(E),
    /// Remote version wins
    Remote(E),
}

impl<E> ConflictWinner<E> {
    /// Get the winning entity.
    pub fn winner(self) -> E {
        match self {
            ConflictWinner::Local(e) | ConflictWinner::Remote(e) => e,
        }
    }

    /// Check if local won.
    pub fn is_local(&self) -> bool {
        matches!(self, ConflictWinner::Local(_))
    }

    /// Check if remote won.
    pub fn is_remote(&self) -> bool {
        matches!(self, ConflictWinner::Remote(_))
    }
}

/// Result of DLQ processing.
#[derive(Debug, Clone, Default)]
pub struct DlqProcessResult {
    /// Total entries processed
    pub processed: usize,
    /// Entries successfully replayed
    pub succeeded: usize,
    /// Entries that failed but will be retried
    pub failed: usize,
    /// Entries abandoned after max retries
    pub abandoned: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Example entity for testing
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestEntity {
        id: String,
        name: String,
        value: i32,
        updated_at: DateTime<Utc>,
    }

    impl TestEntity {
        fn new(id: &str, name: &str, value: i32) -> Self {
            Self {
                id: id.to_string(),
                name: name.to_string(),
                value,
                updated_at: Utc::now(),
            }
        }
    }

    #[async_trait]
    impl DualWritable for TestEntity {
        type Id = String;

        fn table_name() -> &'static str {
            "test_entities"
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
            "INSERT OR REPLACE INTO test_entities (id, name, value, updated_at) VALUES (?, ?, ?, ?)"
        }

        fn sqlite_update_sql() -> &'static str {
            "UPDATE test_entities SET name = ?, value = ?, updated_at = ? WHERE id = ?"
        }

        fn sqlite_insert_params(&self) -> Vec<Box<dyn rusqlite::ToSql + Send + Sync>> {
            vec![
                Box::new(self.id.clone()),
                Box::new(self.name.clone()),
                Box::new(self.value),
                Box::new(self.updated_at.to_rfc3339()),
            ]
        }

        fn sqlite_update_params(&self) -> Vec<Box<dyn rusqlite::ToSql + Send + Sync>> {
            vec![
                Box::new(self.name.clone()),
                Box::new(self.value),
                Box::new(self.updated_at.to_rfc3339()),
                Box::new(self.id.clone()),
            ]
        }

        fn postgres_insert_sql() -> &'static str {
            "INSERT INTO test_entities (id, name, value, updated_at) VALUES ($1, $2, $3, $4) ON CONFLICT (id) DO UPDATE SET name = $2, value = $3, updated_at = $4"
        }

        fn postgres_update_sql() -> &'static str {
            "UPDATE test_entities SET name = $1, value = $2, updated_at = $3 WHERE id = $4"
        }

        async fn postgres_insert(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error> {
            sqlx::query(Self::postgres_insert_sql())
                .bind(&self.id)
                .bind(&self.name)
                .bind(self.value)
                .bind(self.updated_at)
                .execute(pool)
                .await?;
            Ok(())
        }

        async fn postgres_update(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error> {
            sqlx::query(Self::postgres_update_sql())
                .bind(&self.name)
                .bind(self.value)
                .bind(self.updated_at)
                .bind(&self.id)
                .execute(pool)
                .await?;
            Ok(())
        }

        async fn postgres_delete(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error> {
            sqlx::query("DELETE FROM test_entities WHERE id = $1")
                .bind(&self.id)
                .execute(pool)
                .await?;
            Ok(())
        }
    }

    #[test]
    fn test_operation_type_display() {
        assert_eq!(format!("{}", OperationType::Insert), "insert");
        assert_eq!(format!("{}", OperationType::Update), "update");
        assert_eq!(format!("{}", OperationType::Delete), "delete");
        assert_eq!(format!("{}", OperationType::Upsert), "upsert");
    }

    #[test]
    fn test_conflict_winner() {
        let local = TestEntity::new("1", "local", 100);
        let remote = TestEntity::new("1", "remote", 200);

        let winner = ConflictWinner::Local(local.clone());
        assert!(winner.is_local());
        assert!(!winner.is_remote());

        let winner = ConflictWinner::Remote(remote.clone());
        assert!(!winner.is_local());
        assert!(winner.is_remote());
    }

    #[test]
    fn test_lww_resolution() {
        let mut local = TestEntity::new("1", "local", 100);
        let mut remote = TestEntity::new("1", "remote", 200);

        // Local is newer
        local.updated_at = Utc::now();
        remote.updated_at = local.updated_at - chrono::Duration::seconds(10);

        let winner = DualWriteAdapter::resolve_conflict_lww(&local, &remote);
        assert!(winner.is_local());

        // Remote is newer
        remote.updated_at = Utc::now();
        local.updated_at = remote.updated_at - chrono::Duration::seconds(10);

        let winner = DualWriteAdapter::resolve_conflict_lww(&local, &remote);
        assert!(winner.is_remote());
    }

    #[tokio::test]
    async fn test_dual_write_adapter_creation() {
        let adapter = DualWriteAdapter::new_for_testing().unwrap();

        assert_eq!(adapter.circuit_state(), CircuitState::Closed);
        assert!(!adapter.is_postgres_available()); // No PG configured
    }

    #[test]
    fn test_dual_write_result() {
        let result = DualWriteResult {
            sqlite_success: true,
            postgres_success: Some(true),
            queued_to_dlq: false,
            entity_id: "test-1".to_string(),
            warnings: vec![],
        };

        assert!(result.is_ok());
        assert!(result.is_fully_synced());

        let result_partial = DualWriteResult {
            sqlite_success: true,
            postgres_success: Some(false),
            queued_to_dlq: true,
            entity_id: "test-2".to_string(),
            warnings: vec!["PG write failed".to_string()],
        };

        assert!(result_partial.is_ok());
        assert!(!result_partial.is_fully_synced());
    }
}
