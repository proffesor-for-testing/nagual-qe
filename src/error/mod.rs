//! Error types for the Nagual system.
//!
//! Provides a comprehensive error hierarchy using thiserror for all
//! operations including database, migration, and coordination errors.
//!
//! This module also provides resilience patterns:
//! - Retry with exponential backoff and jitter
//! - Circuit breaker for fault isolation
//! - Dead letter queue for failed operations

mod circuit_breaker;
mod dlq;
mod dlq_worker;
mod retry;

pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerMetrics, CircuitBreakerRegistry, CircuitState};
pub use dlq::{BatchProcessResult, DeadLetterQueue, DlqEntry, DlqStats};
pub use dlq_worker::{DlqStorage, DlqWorker, DlqWorkerConfig, DlqWorkerHandle, HandlerFn, OperationRouter};
pub use retry::{with_retry, with_retry_nagual, with_retry_policy, AlwaysRetry, CustomRetryCondition, DefaultRetryCondition, NeverRetry, RetryCondition, RetryPolicy};

use thiserror::Error;

/// Main result type for Nagual operations.
pub type Result<T> = std::result::Result<T, NagualError>;

/// Primary error type for the Nagual system.
#[derive(Error, Debug)]
pub enum NagualError {
    /// Database-related errors
    #[error("Database error: {0}")]
    Database(#[from] DatabaseError),

    /// Migration-related errors
    #[error("Migration error: {0}")]
    Migration(#[from] MigrationError),

    /// Coordination errors for dual-database operations
    #[error("Coordination error: {0}")]
    Coordination(#[from] CoordinationError),

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Config { message: String },

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// HTTP/network errors (e.g. Screenpipe API)
    #[error("HTTP error: {0}")]
    Http(String),

    /// Generic internal errors
    #[error("Internal error: {message}")]
    Internal { message: String },
}

/// Database-specific errors.
#[derive(Error, Debug)]
pub enum DatabaseError {
    /// SQLite-specific error
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// PostgreSQL-specific error (via sqlx)
    #[error("PostgreSQL error: {0}")]
    Postgres(#[from] sqlx::Error),

    /// Connection pool exhausted
    #[error("Connection pool exhausted")]
    PoolExhausted,

    /// Connection timeout
    #[error("Connection timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// Database not found
    #[error("Database not found: {path}")]
    NotFound { path: String },

    /// Schema mismatch
    #[error("Schema mismatch: expected version {expected}, found {found}")]
    SchemaMismatch { expected: i32, found: i32 },
}

/// Migration-specific errors.
#[derive(Error, Debug)]
pub enum MigrationError {
    /// Migration already applied
    #[error("Migration '{name}' (version {version}) already applied")]
    AlreadyApplied { version: i64, name: String },

    /// Migration not found
    #[error("Migration version {version} not found")]
    NotFound { version: i64 },

    /// Migration checksum mismatch (script was modified after being applied)
    #[error("Checksum mismatch for migration {version}: expected {expected}, found {found}")]
    ChecksumMismatch {
        version: i64,
        expected: String,
        found: String,
    },

    /// Migration lock could not be acquired
    #[error("Could not acquire migration lock: {reason}")]
    LockFailed { reason: String },

    /// Migration lock is held by another process
    #[error("Migration lock held by process {pid} since {acquired_at}")]
    LockHeld { pid: i64, acquired_at: String },

    /// Migration script execution failed
    #[error("Migration {version} failed: {message}")]
    ExecutionFailed { version: i64, message: String },

    /// Invalid migration script
    #[error("Invalid migration script '{name}': {reason}")]
    InvalidScript { name: String, reason: String },

    /// Rollback not possible
    #[error("Cannot rollback migration {version}: {reason}")]
    RollbackFailed { version: i64, reason: String },

    /// No migrations to apply
    #[error("No pending migrations")]
    NoPendingMigrations,

    /// Checkpoint error
    #[error("Checkpoint error: {message}")]
    CheckpointFailed { message: String },

    /// Migration file I/O error
    #[error("Migration file error: {0}")]
    FileError(#[from] std::io::Error),

    /// SQL execution error
    #[error("SQL error during migration: {0}")]
    SqlError(String),
}

/// Coordination errors for dual-database operations.
#[derive(Error, Debug)]
pub enum CoordinationError {
    /// SQLite operation failed
    #[error("SQLite operation failed: {message}")]
    SqliteFailed { message: String },

    /// PostgreSQL operation failed
    #[error("PostgreSQL operation failed: {message}")]
    PostgresFailed { message: String },

    /// Both databases failed
    #[error("Both databases failed - SQLite: {sqlite_error}, PostgreSQL: {postgres_error}")]
    BothFailed {
        sqlite_error: String,
        postgres_error: String,
    },

    /// Partial failure - one database succeeded, other failed
    #[error("Partial failure - {succeeded} succeeded, {failed} failed: {message}")]
    PartialFailure {
        succeeded: String,
        failed: String,
        message: String,
    },

    /// Inconsistent state between databases
    #[error("Inconsistent state: SQLite at version {sqlite_version}, PostgreSQL at version {postgres_version}")]
    InconsistentState {
        sqlite_version: i64,
        postgres_version: i64,
    },

    /// Coordination timeout
    #[error("Coordination timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// Recovery required
    #[error("Manual recovery required: {message}")]
    RecoveryRequired { message: String },
}

/// Circuit breaker specific errors.
#[derive(Error, Debug, Clone)]
pub enum CircuitBreakerError {
    /// Circuit is open, requests are being rejected
    #[error("Circuit breaker open for service '{service}', will retry after {retry_after_ms}ms")]
    Open { service: String, retry_after_ms: u64 },

    /// Circuit is half-open, limited requests allowed
    #[error("Circuit breaker half-open for service '{service}'")]
    HalfOpen { service: String },
}

/// Retry-specific errors.
#[derive(Error, Debug)]
pub enum RetryError {
    /// Maximum retries exceeded
    #[error("Max retries ({max_retries}) exceeded after {total_delay_ms}ms: {last_error}")]
    MaxRetriesExceeded {
        max_retries: u32,
        total_delay_ms: u64,
        last_error: String,
    },

    /// Operation is not retryable
    #[error("Operation is not retryable: {reason}")]
    NotRetryable { reason: String },

    /// Retry was cancelled
    #[error("Retry cancelled")]
    Cancelled,
}

/// Dead letter queue errors.
#[derive(Error, Debug)]
pub enum DlqError {
    /// Failed to enqueue operation
    #[error("Failed to enqueue operation: {0}")]
    EnqueueFailed(String),

    /// Failed to dequeue operation
    #[error("Failed to dequeue operation: {0}")]
    DequeueFailed(String),

    /// Operation was abandoned after too many attempts
    #[error("Operation '{operation_id}' abandoned after {attempts} attempts")]
    Abandoned {
        operation_id: String,
        attempts: u32,
    },

    /// DLQ database error
    #[error("DLQ database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// DLQ is full
    #[error("DLQ is full (max {max_size} entries)")]
    Full { max_size: usize },
}

impl From<String> for MigrationError {
    fn from(s: String) -> Self {
        MigrationError::SqlError(s)
    }
}

impl NagualError {
    /// Create a configuration error with a message.
    pub fn config(message: impl Into<String>) -> Self {
        NagualError::Config {
            message: message.into(),
        }
    }

    /// Create an internal error with a message.
    pub fn internal(message: impl Into<String>) -> Self {
        NagualError::Internal {
            message: message.into(),
        }
    }

    /// Check if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            NagualError::Database(db_err) => db_err.is_retryable(),
            NagualError::Coordination(coord_err) => coord_err.is_retryable(),
            NagualError::Io(_) => true,
            _ => false,
        }
    }

    /// Check if this error is transient (temporary).
    pub fn is_transient(&self) -> bool {
        match self {
            NagualError::Database(db_err) => db_err.is_transient(),
            NagualError::Coordination(CoordinationError::Timeout { .. }) => true,
            NagualError::Io(_) => true,
            _ => false,
        }
    }

    /// Get the error code for logging and metrics.
    pub fn error_code(&self) -> &'static str {
        match self {
            NagualError::Database(_) => "DATABASE_ERROR",
            NagualError::Migration(_) => "MIGRATION_ERROR",
            NagualError::Coordination(_) => "COORDINATION_ERROR",
            NagualError::Config { .. } => "CONFIG_ERROR",
            NagualError::Io(_) => "IO_ERROR",
            NagualError::Serde(_) => "SERDE_ERROR",
            NagualError::Http(_) => "HTTP_ERROR",
            NagualError::Internal { .. } => "INTERNAL_ERROR",
        }
    }
}

impl DatabaseError {
    /// Check if this database error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            DatabaseError::PoolExhausted
                | DatabaseError::Timeout { .. }
        )
    }

    /// Check if this database error is transient.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            DatabaseError::PoolExhausted
                | DatabaseError::Timeout { .. }
        )
    }
}

impl CoordinationError {
    /// Check if this coordination error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            CoordinationError::Timeout { .. }
                | CoordinationError::SqliteFailed { .. }
                | CoordinationError::PostgresFailed { .. }
        )
    }
}

impl From<CircuitBreakerError> for NagualError {
    fn from(err: CircuitBreakerError) -> Self {
        NagualError::Internal {
            message: err.to_string(),
        }
    }
}

impl From<RetryError> for NagualError {
    fn from(err: RetryError) -> Self {
        NagualError::Internal {
            message: err.to_string(),
        }
    }
}

impl From<DlqError> for NagualError {
    fn from(err: DlqError) -> Self {
        NagualError::Internal {
            message: err.to_string(),
        }
    }
}

impl From<sqlx::Error> for NagualError {
    fn from(err: sqlx::Error) -> Self {
        NagualError::Database(DatabaseError::Postgres(err))
    }
}

impl From<MigrationError> for DatabaseError {
    fn from(err: MigrationError) -> Self {
        // Convert MigrationError to a DatabaseError by wrapping the message
        // This is used when migration operations are performed within database closures
        match err {
            MigrationError::SqlError(msg) => DatabaseError::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(1),
                Some(msg),
            )),
            other => DatabaseError::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(1),
                Some(other.to_string()),
            )),
        }
    }
}

/// Extension trait for adding context to Results.
pub trait ResultExt<T> {
    /// Add context to an error.
    fn context(self, context: impl Into<String>) -> Result<T>;

    /// Add lazy context to an error.
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}

impl<T> ResultExt<T> for Result<T> {
    fn context(self, context: impl Into<String>) -> Result<T> {
        self.map_err(|e| NagualError::Internal {
            message: format!("{}: {}", context.into(), e),
        })
    }

    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| NagualError::Internal {
            message: format!("{}: {}", f(), e),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_is_retryable() {
        let timeout = NagualError::Database(DatabaseError::Timeout { timeout_ms: 5000 });
        assert!(timeout.is_retryable());

        let config_err = NagualError::config("bad config");
        assert!(!config_err.is_retryable());
    }

    #[test]
    fn test_error_code() {
        let err = NagualError::config("test");
        assert_eq!(err.error_code(), "CONFIG_ERROR");

        let db_err = NagualError::Database(DatabaseError::PoolExhausted);
        assert_eq!(db_err.error_code(), "DATABASE_ERROR");
    }

    #[test]
    fn test_circuit_breaker_error() {
        let err = CircuitBreakerError::Open {
            service: "database".to_string(),
            retry_after_ms: 30000,
        };
        assert!(err.to_string().contains("database"));
        assert!(err.to_string().contains("30000"));
    }

    #[test]
    fn test_dlq_error() {
        let err = DlqError::Abandoned {
            operation_id: "op-123".to_string(),
            attempts: 10,
        };
        assert!(err.to_string().contains("op-123"));
        assert!(err.to_string().contains("10"));
    }
}
