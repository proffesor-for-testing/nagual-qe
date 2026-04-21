//! Migration system for Nagual.
//!
//! Provides a robust migration framework supporting:
//! - Schema versioning with checksums
//! - Advisory locking to prevent concurrent migrations
//! - Up/down migrations with rollback support
//! - Checkpoint system for resumable migrations
//! - Dual-database coordination (SQLite + PostgreSQL)

mod coordinator;
mod runner;

pub use coordinator::{CoordinationResult, DualMigrationCoordinator, MigrationOutcome};
pub use runner::{
    Checkpoint, Migration, MigrationRunner, MigrationStatus, PostgresMigrationRunner, SchemaVersion,
};

use crate::error::{MigrationError, Result};
use chrono::{DateTime, Utc};
use ring::digest::{digest, SHA256};
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// SQL for creating the schema_version table in SQLite.
pub const SQLITE_SCHEMA_VERSION_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    checksum TEXT NOT NULL,
    description TEXT NOT NULL,
    execution_time_ms INTEGER,
    rolled_back_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_schema_version_applied
ON schema_version(applied_at);
"#;

/// SQL for creating the schema_version table in PostgreSQL.
pub const POSTGRES_SCHEMA_VERSION_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS schema_version (
    version BIGINT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum TEXT NOT NULL,
    description TEXT NOT NULL,
    execution_time_ms BIGINT,
    rolled_back_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_schema_version_applied
ON schema_version(applied_at);
"#;

/// SQL for creating the migration_lock table in SQLite.
/// Uses a simple row-based lock pattern.
pub const SQLITE_MIGRATION_LOCK_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS migration_lock (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    locked_by TEXT NOT NULL,
    locked_at TEXT NOT NULL,
    pid INTEGER NOT NULL,
    host TEXT
);
"#;

/// SQL for creating the migration_lock table in PostgreSQL.
/// Uses PostgreSQL advisory locks for better coordination.
pub const POSTGRES_MIGRATION_LOCK_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS migration_lock (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    locked_by TEXT NOT NULL,
    locked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pid INTEGER NOT NULL,
    host TEXT
);
"#;

/// SQL for creating the migration_checkpoint table.
/// Tracks partial progress during long-running migrations.
pub const SQLITE_CHECKPOINT_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS migration_checkpoint (
    version INTEGER PRIMARY KEY,
    step_index INTEGER NOT NULL,
    total_steps INTEGER NOT NULL,
    last_completed_step TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"#;

/// PostgreSQL checkpoint table.
pub const POSTGRES_CHECKPOINT_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS migration_checkpoint (
    version BIGINT PRIMARY KEY,
    step_index INTEGER NOT NULL,
    total_steps INTEGER NOT NULL,
    last_completed_step TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"#;

/// Lock information for advisory locking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockInfo {
    /// Who acquired the lock (usually process identifier).
    pub locked_by: String,
    /// When the lock was acquired.
    pub locked_at: DateTime<Utc>,
    /// Process ID that holds the lock.
    pub pid: i64,
    /// Hostname where the lock was acquired.
    pub host: Option<String>,
}

impl LockInfo {
    /// Create a new lock info for the current process.
    pub fn current() -> Self {
        Self {
            locked_by: format!("nagual-{}", std::process::id()),
            locked_at: Utc::now(),
            pid: std::process::id() as i64,
            host: hostname::get()
                .ok()
                .and_then(|h| h.into_string().ok()),
        }
    }
}

/// Advisory lock manager for preventing concurrent migrations.
pub struct MigrationLock {
    /// Lock key identifier.
    pub key: String,
    /// Lock info if currently held.
    pub info: Option<LockInfo>,
}

impl MigrationLock {
    /// Create a new migration lock manager.
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            info: None,
        }
    }

    /// Try to acquire a lock in SQLite.
    pub fn try_acquire_sqlite(
        conn: &rusqlite::Connection,
    ) -> std::result::Result<Option<LockInfo>, MigrationError> {
        // First ensure the lock table exists
        conn.execute_batch(SQLITE_MIGRATION_LOCK_TABLE)
            .map_err(|e| MigrationError::LockFailed {
                reason: format!("Failed to create lock table: {}", e),
            })?;

        // Check if lock is held
        let existing: Option<(String, String, i64)> = conn
            .query_row(
                "SELECT locked_by, locked_at, pid FROM migration_lock WHERE id = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()
            .map_err(|e| MigrationError::LockFailed {
                reason: format!("Failed to check lock: {}", e),
            })?;

        if let Some((locked_by, locked_at, pid)) = existing {
            // Lock is held - check if it's stale (older than 1 hour)
            if let Ok(lock_time) = DateTime::parse_from_rfc3339(&locked_at) {
                let age = Utc::now().signed_duration_since(lock_time.with_timezone(&Utc));
                if age.num_hours() >= 1 {
                    // Stale lock - can be forcibly released
                    tracing::warn!(
                        "Found stale migration lock held by {} since {}, releasing",
                        locked_by,
                        locked_at
                    );
                    conn.execute("DELETE FROM migration_lock WHERE id = 1", [])
                        .map_err(|e| MigrationError::LockFailed {
                            reason: format!("Failed to release stale lock: {}", e),
                        })?;
                } else {
                    return Err(MigrationError::LockHeld {
                        pid,
                        acquired_at: locked_at,
                    });
                }
            }
        }

        // Try to acquire lock
        let info = LockInfo::current();
        conn.execute(
            "INSERT INTO migration_lock (id, locked_by, locked_at, pid, host)
             VALUES (1, ?, ?, ?, ?)",
            rusqlite::params![
                &info.locked_by,
                info.locked_at.to_rfc3339(),
                info.pid,
                &info.host,
            ],
        )
        .map_err(|e| MigrationError::LockFailed {
            reason: format!("Failed to acquire lock: {}", e),
        })?;

        Ok(Some(info))
    }

    /// Release a SQLite lock.
    pub fn release_sqlite(conn: &rusqlite::Connection) -> std::result::Result<(), MigrationError> {
        conn.execute("DELETE FROM migration_lock WHERE id = 1", [])
            .map_err(|e| MigrationError::LockFailed {
                reason: format!("Failed to release lock: {}", e),
            })?;
        Ok(())
    }
}

/// Calculate SHA-256 checksum of migration content.
pub fn calculate_checksum(content: &str) -> String {
    let hash = digest(&SHA256, content.as_bytes());
    hex::encode(hash.as_ref())
}

/// Validate that a migration script is properly formatted.
pub fn validate_migration_script(content: &str, name: &str) -> Result<()> {
    if content.trim().is_empty() {
        return Err(MigrationError::InvalidScript {
            name: name.to_string(),
            reason: "Migration script is empty".to_string(),
        }
        .into());
    }

    // Check for dangerous statements without transaction
    let dangerous_patterns = ["DROP TABLE", "DROP DATABASE", "TRUNCATE"];
    for pattern in dangerous_patterns {
        if content.to_uppercase().contains(pattern) {
            tracing::warn!(
                "Migration {} contains dangerous statement: {}",
                name,
                pattern
            );
        }
    }

    Ok(())
}

/// Generate a new migration file template.
pub fn generate_migration_template(name: &str, version: i64) -> (String, String) {
    let up_content = format!(
        r#"-- Migration: {}
-- Version: {}
-- Created: {}

-- UP migration
-- Add your schema changes here

"#,
        name,
        version,
        Utc::now().to_rfc3339()
    );

    let down_content = format!(
        r#"-- Migration: {} (rollback)
-- Version: {}
-- Created: {}

-- DOWN migration
-- Add rollback statements here

"#,
        name,
        version,
        Utc::now().to_rfc3339()
    );

    (up_content, down_content)
}

/// Get the migrations directory path.
pub fn migrations_dir() -> PathBuf {
    PathBuf::from("migrations")
}

/// Hex encoding helper (since we're not using the hex crate).
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_checksum() {
        let content = "CREATE TABLE test (id INTEGER);";
        let checksum = calculate_checksum(content);
        assert!(!checksum.is_empty());
        assert_eq!(checksum.len(), 64); // SHA-256 produces 32 bytes = 64 hex chars
    }

    #[test]
    fn test_validate_empty_script() {
        let result = validate_migration_script("", "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_valid_script() {
        let result = validate_migration_script("CREATE TABLE test (id INTEGER);", "test");
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_template() {
        let (up, down) = generate_migration_template("create_users", 1);
        assert!(up.contains("create_users"));
        assert!(up.contains("UP migration"));
        assert!(down.contains("DOWN migration"));
    }

    #[test]
    fn test_lock_info_current() {
        let info = LockInfo::current();
        assert!(info.locked_by.starts_with("nagual-"));
        assert!(info.pid > 0);
    }
}
