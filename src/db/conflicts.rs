//! Conflict logging and resolution for dual-write operations.
//!
//! Provides persistent storage for conflicts between local SQLite and cloud PostgreSQL
//! databases. Supports Last-Write-Wins (LWW) resolution strategy with manual override
//! capabilities.
//!
//! ## Conflict Log Schema
//!
//! ```sql
//! CREATE TABLE conflict_log (
//!     id TEXT PRIMARY KEY,
//!     table_name TEXT NOT NULL,
//!     record_id TEXT NOT NULL,
//!     local_data TEXT NOT NULL,      -- JSON
//!     remote_data TEXT NOT NULL,     -- JSON
//!     resolution TEXT NOT NULL,      -- pending, local_wins, remote_wins, merged, manual
//!     resolved_at TEXT,              -- RFC3339 timestamp
//!     created_at TEXT NOT NULL,      -- RFC3339 timestamp
//!     metadata TEXT                  -- Additional context (JSON)
//! );
//! ```

use std::path::Path;

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::error::{DlqError, NagualError, Result};

/// Resolution status for a conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictResolution {
    /// Conflict is pending resolution
    Pending,
    /// Local version wins (Last-Write-Wins or manual choice)
    LocalWins,
    /// Remote version wins (Last-Write-Wins or manual choice)
    RemoteWins,
    /// Versions were merged
    Merged,
    /// Manually resolved with custom data
    Manual,
    /// Conflict was skipped/ignored
    Skipped,
}

impl std::fmt::Display for ConflictResolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConflictResolution::Pending => write!(f, "pending"),
            ConflictResolution::LocalWins => write!(f, "local_wins"),
            ConflictResolution::RemoteWins => write!(f, "remote_wins"),
            ConflictResolution::Merged => write!(f, "merged"),
            ConflictResolution::Manual => write!(f, "manual"),
            ConflictResolution::Skipped => write!(f, "skipped"),
        }
    }
}

impl std::str::FromStr for ConflictResolution {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "pending" => Ok(ConflictResolution::Pending),
            "local_wins" => Ok(ConflictResolution::LocalWins),
            "remote_wins" => Ok(ConflictResolution::RemoteWins),
            "merged" => Ok(ConflictResolution::Merged),
            "manual" => Ok(ConflictResolution::Manual),
            "skipped" => Ok(ConflictResolution::Skipped),
            _ => Err(format!("Unknown resolution: {}", s)),
        }
    }
}

/// Entry in the conflict log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictLogEntry {
    /// Unique identifier for this conflict
    pub id: String,
    /// Name of the table where the conflict occurred
    pub table_name: String,
    /// Primary key of the conflicting record
    pub record_id: String,
    /// Local (SQLite) version of the data
    pub local_data: serde_json::Value,
    /// Remote (PostgreSQL) version of the data
    pub remote_data: serde_json::Value,
    /// Current resolution status
    pub resolution: ConflictResolution,
    /// When the conflict was resolved (if resolved)
    pub resolved_at: Option<DateTime<Utc>>,
    /// When the conflict was detected
    pub created_at: DateTime<Utc>,
    /// Additional metadata
    pub metadata: Option<serde_json::Value>,
}

impl ConflictLogEntry {
    /// Create a new conflict log entry.
    pub fn new(
        table_name: &str,
        record_id: &str,
        local_data: serde_json::Value,
        remote_data: serde_json::Value,
        resolution: ConflictResolution,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            table_name: table_name.to_string(),
            record_id: record_id.to_string(),
            local_data,
            remote_data,
            resolution,
            resolved_at: None,
            created_at: Utc::now(),
            metadata: None,
        }
    }

    /// Add metadata to the entry.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Check if this conflict is pending resolution.
    pub fn is_pending(&self) -> bool {
        self.resolution == ConflictResolution::Pending
    }

    /// Get the local updated_at timestamp if present.
    pub fn local_updated_at(&self) -> Option<DateTime<Utc>> {
        self.local_data
            .get("updated_at")
            .and_then(|v| v.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
    }

    /// Get the remote updated_at timestamp if present.
    pub fn remote_updated_at(&self) -> Option<DateTime<Utc>> {
        self.remote_data
            .get("updated_at")
            .and_then(|v| v.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
    }

    /// Determine the winner using Last-Write-Wins strategy.
    pub fn lww_winner(&self) -> Option<ConflictResolution> {
        match (self.local_updated_at(), self.remote_updated_at()) {
            (Some(local_ts), Some(remote_ts)) => {
                if local_ts >= remote_ts {
                    Some(ConflictResolution::LocalWins)
                } else {
                    Some(ConflictResolution::RemoteWins)
                }
            }
            (Some(_), None) => Some(ConflictResolution::LocalWins),
            (None, Some(_)) => Some(ConflictResolution::RemoteWins),
            (None, None) => None,
        }
    }
}

/// Conflict log backed by SQLite.
pub struct ConflictLog {
    conn: Connection,
}

impl ConflictLog {
    /// Create a new conflict log at the specified path.
    pub fn new(path: impl AsRef<Path>) -> std::result::Result<Self, DlqError> {
        let conn = Connection::open(path)?;
        let log = Self { conn };
        log.initialize_schema()?;
        Ok(log)
    }

    /// Create an in-memory conflict log (for testing).
    pub fn in_memory() -> std::result::Result<Self, DlqError> {
        let conn = Connection::open_in_memory()?;
        let log = Self { conn };
        log.initialize_schema()?;
        Ok(log)
    }

    /// Initialize the database schema.
    fn initialize_schema(&self) -> std::result::Result<(), DlqError> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS conflict_log (
                id TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                record_id TEXT NOT NULL,
                local_data TEXT NOT NULL,
                remote_data TEXT NOT NULL,
                resolution TEXT NOT NULL DEFAULT 'pending',
                resolved_at TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_conflict_log_resolution
                ON conflict_log(resolution);

            CREATE INDEX IF NOT EXISTS idx_conflict_log_table_name
                ON conflict_log(table_name);

            CREATE INDEX IF NOT EXISTS idx_conflict_log_created_at
                ON conflict_log(created_at);

            CREATE INDEX IF NOT EXISTS idx_conflict_log_table_record
                ON conflict_log(table_name, record_id);
            "#,
        )?;

        info!("Conflict log schema initialized");
        Ok(())
    }

    /// Log a new conflict.
    pub fn log(&self, entry: &ConflictLogEntry) -> std::result::Result<String, DlqError> {
        let local_data = serde_json::to_string(&entry.local_data)
            .map_err(|e| DlqError::EnqueueFailed(e.to_string()))?;
        let remote_data = serde_json::to_string(&entry.remote_data)
            .map_err(|e| DlqError::EnqueueFailed(e.to_string()))?;
        let metadata = entry
            .metadata
            .as_ref()
            .map(|m| serde_json::to_string(m))
            .transpose()
            .map_err(|e| DlqError::EnqueueFailed(e.to_string()))?;

        self.conn.execute(
            r#"
            INSERT INTO conflict_log (
                id, table_name, record_id, local_data, remote_data,
                resolution, resolved_at, created_at, metadata
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
            params![
                entry.id,
                entry.table_name,
                entry.record_id,
                local_data,
                remote_data,
                entry.resolution.to_string(),
                entry.resolved_at.map(|dt| dt.to_rfc3339()),
                entry.created_at.to_rfc3339(),
                metadata,
            ],
        )?;

        debug!(
            conflict_id = %entry.id,
            table = %entry.table_name,
            record_id = %entry.record_id,
            "Conflict logged"
        );

        Ok(entry.id.clone())
    }

    /// Get a conflict by ID.
    pub fn get(&self, id: &str) -> std::result::Result<Option<ConflictLogEntry>, DlqError> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, table_name, record_id, local_data, remote_data,
                   resolution, resolved_at, created_at, metadata
            FROM conflict_log
            WHERE id = ?1
            "#,
        )?;

        let entry = stmt.query_row([id], |row| {
            Self::row_to_entry(row)
        });

        match entry {
            Ok(e) => Ok(Some(e)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(DlqError::DequeueFailed(e.to_string())),
        }
    }

    /// Get all pending conflicts.
    pub fn get_pending(&self, limit: usize) -> std::result::Result<Vec<ConflictLogEntry>, DlqError> {
        self.get_by_resolution(ConflictResolution::Pending, limit)
    }

    /// Get conflicts by resolution status.
    pub fn get_by_resolution(
        &self,
        resolution: ConflictResolution,
        limit: usize,
    ) -> std::result::Result<Vec<ConflictLogEntry>, DlqError> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, table_name, record_id, local_data, remote_data,
                   resolution, resolved_at, created_at, metadata
            FROM conflict_log
            WHERE resolution = ?1
            ORDER BY created_at ASC
            LIMIT ?2
            "#,
        )?;

        let entries = stmt.query_map([resolution.to_string(), limit.to_string()], |row| {
            Self::row_to_entry(row)
        })?;

        entries
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| DlqError::DequeueFailed(e.to_string()))
    }

    /// Get conflicts by table name.
    pub fn get_by_table(
        &self,
        table_name: &str,
        limit: usize,
    ) -> std::result::Result<Vec<ConflictLogEntry>, DlqError> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, table_name, record_id, local_data, remote_data,
                   resolution, resolved_at, created_at, metadata
            FROM conflict_log
            WHERE table_name = ?1
            ORDER BY created_at DESC
            LIMIT ?2
            "#,
        )?;

        let entries = stmt.query_map([table_name, &limit.to_string()], |row| {
            Self::row_to_entry(row)
        })?;

        entries
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| DlqError::DequeueFailed(e.to_string()))
    }

    /// Resolve a conflict.
    pub fn resolve(
        &self,
        id: &str,
        resolution: ConflictResolution,
    ) -> std::result::Result<(), DlqError> {
        let now = Utc::now();
        let rows = self.conn.execute(
            "UPDATE conflict_log SET resolution = ?1, resolved_at = ?2 WHERE id = ?3",
            params![resolution.to_string(), now.to_rfc3339(), id],
        )?;

        if rows == 0 {
            warn!(conflict_id = %id, "Conflict not found");
            return Err(DlqError::DequeueFailed(format!("Conflict {} not found", id)));
        }

        info!(
            conflict_id = %id,
            resolution = %resolution,
            "Conflict resolved"
        );

        Ok(())
    }

    /// Auto-resolve pending conflicts using Last-Write-Wins.
    pub fn auto_resolve_lww(&self, limit: usize) -> std::result::Result<AutoResolveResult, DlqError> {
        let pending = self.get_pending(limit)?;
        let mut result = AutoResolveResult::default();

        for entry in pending {
            if let Some(winner) = entry.lww_winner() {
                match self.resolve(&entry.id, winner) {
                    Ok(()) => result.resolved += 1,
                    Err(e) => {
                        warn!(
                            conflict_id = %entry.id,
                            error = %e,
                            "Failed to resolve conflict"
                        );
                        result.failed += 1;
                    }
                }
            } else {
                // Can't determine winner automatically
                result.skipped += 1;
                debug!(
                    conflict_id = %entry.id,
                    "Cannot auto-resolve: missing timestamps"
                );
            }
        }

        info!(
            resolved = result.resolved,
            failed = result.failed,
            skipped = result.skipped,
            "Auto-resolve LWW complete"
        );

        Ok(result)
    }

    /// Get conflict statistics.
    pub fn stats(&self) -> std::result::Result<ConflictStats, DlqError> {
        let pending: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM conflict_log WHERE resolution = 'pending'",
            [],
            |row| row.get(0),
        )?;

        let resolved: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM conflict_log WHERE resolution != 'pending'",
            [],
            |row| row.get(0),
        )?;

        let local_wins: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM conflict_log WHERE resolution = 'local_wins'",
            [],
            |row| row.get(0),
        )?;

        let remote_wins: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM conflict_log WHERE resolution = 'remote_wins'",
            [],
            |row| row.get(0),
        )?;

        let oldest_pending: Option<String> = self.conn.query_row(
            "SELECT MIN(created_at) FROM conflict_log WHERE resolution = 'pending'",
            [],
            |row| row.get(0),
        ).ok();

        Ok(ConflictStats {
            pending,
            resolved,
            local_wins,
            remote_wins,
            total: pending + resolved,
            oldest_pending: oldest_pending.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            }),
        })
    }

    /// Clean up old resolved conflicts.
    pub fn cleanup_resolved(&self, older_than_days: u32) -> std::result::Result<usize, DlqError> {
        let cutoff = Utc::now() - chrono::Duration::days(older_than_days as i64);
        let deleted = self.conn.execute(
            "DELETE FROM conflict_log WHERE resolution != 'pending' AND resolved_at < ?1",
            params![cutoff.to_rfc3339()],
        )?;

        info!(deleted = deleted, "Cleaned up old resolved conflicts");
        Ok(deleted)
    }

    /// Export conflicts to JSON.
    pub fn export_to_json(&self, limit: usize) -> std::result::Result<String, DlqError> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, table_name, record_id, local_data, remote_data,
                   resolution, resolved_at, created_at, metadata
            FROM conflict_log
            ORDER BY created_at DESC
            LIMIT ?1
            "#,
        )?;

        let entries: Vec<ConflictLogEntry> = stmt
            .query_map([limit.to_string()], |row| Self::row_to_entry(row))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| DlqError::DequeueFailed(e.to_string()))?;

        serde_json::to_string_pretty(&entries)
            .map_err(|e| DlqError::EnqueueFailed(e.to_string()))
    }

    /// Convert a database row to a ConflictLogEntry.
    ///
    /// Note: This function uses defensive defaults for parsing failures:
    /// - Invalid JSON in local_data/remote_data becomes Value::Null
    /// - Invalid resolution string defaults to Pending
    /// - Invalid created_at timestamp defaults to current time
    fn row_to_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<ConflictLogEntry> {
        let local_data_str: String = row.get(3)?;
        let remote_data_str: String = row.get(4)?;
        let resolution_str: String = row.get(5)?;
        let resolved_at_str: Option<String> = row.get(6)?;
        let created_at_str: String = row.get(7)?;
        let metadata_str: Option<String> = row.get(8)?;

        Ok(ConflictLogEntry {
            id: row.get(0)?,
            table_name: row.get(1)?,
            record_id: row.get(2)?,
            // Use defensive defaults for potentially malformed data from the database
            local_data: serde_json::from_str(&local_data_str).unwrap_or(serde_json::Value::Null),
            remote_data: serde_json::from_str(&remote_data_str).unwrap_or(serde_json::Value::Null),
            resolution: resolution_str
                .parse()
                .unwrap_or(ConflictResolution::Pending),
            resolved_at: resolved_at_str.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            }),
            created_at: DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
        })
    }
}

/// Result of auto-resolution.
#[derive(Debug, Clone, Default)]
pub struct AutoResolveResult {
    /// Number of conflicts resolved
    pub resolved: usize,
    /// Number of conflicts that failed to resolve
    pub failed: usize,
    /// Number of conflicts skipped (couldn't determine winner)
    pub skipped: usize,
}

impl AutoResolveResult {
    /// Total conflicts processed.
    pub fn total(&self) -> usize {
        self.resolved + self.failed + self.skipped
    }
}

/// Statistics about the conflict log.
#[derive(Debug, Clone)]
pub struct ConflictStats {
    /// Number of pending conflicts
    pub pending: usize,
    /// Number of resolved conflicts
    pub resolved: usize,
    /// Number resolved as local wins
    pub local_wins: usize,
    /// Number resolved as remote wins
    pub remote_wins: usize,
    /// Total conflicts
    pub total: usize,
    /// Oldest pending conflict timestamp
    pub oldest_pending: Option<DateTime<Utc>>,
}

/// Public function to log a conflict (used by CLI and other modules).
pub fn log_conflict(
    conflict_log: &ConflictLog,
    table_name: &str,
    record_id: &str,
    local_data: serde_json::Value,
    remote_data: serde_json::Value,
) -> Result<String> {
    let entry = ConflictLogEntry::new(
        table_name,
        record_id,
        local_data,
        remote_data,
        ConflictResolution::Pending,
    );

    conflict_log.log(&entry).map_err(NagualError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_conflict_resolution_display() {
        assert_eq!(format!("{}", ConflictResolution::Pending), "pending");
        assert_eq!(format!("{}", ConflictResolution::LocalWins), "local_wins");
        assert_eq!(format!("{}", ConflictResolution::RemoteWins), "remote_wins");
        assert_eq!(format!("{}", ConflictResolution::Merged), "merged");
        assert_eq!(format!("{}", ConflictResolution::Manual), "manual");
        assert_eq!(format!("{}", ConflictResolution::Skipped), "skipped");
    }

    #[test]
    fn test_conflict_resolution_parse() {
        assert_eq!(
            "pending".parse::<ConflictResolution>().unwrap(),
            ConflictResolution::Pending
        );
        assert_eq!(
            "local_wins".parse::<ConflictResolution>().unwrap(),
            ConflictResolution::LocalWins
        );
        assert!("invalid".parse::<ConflictResolution>().is_err());
    }

    #[test]
    fn test_conflict_log_entry_creation() {
        let local = json!({"id": "1", "name": "local", "updated_at": "2024-01-15T10:00:00Z"});
        let remote = json!({"id": "1", "name": "remote", "updated_at": "2024-01-15T11:00:00Z"});

        let entry = ConflictLogEntry::new(
            "test_table",
            "1",
            local,
            remote,
            ConflictResolution::Pending,
        );

        assert!(entry.is_pending());
        assert_eq!(entry.table_name, "test_table");
        assert_eq!(entry.record_id, "1");
    }

    #[test]
    fn test_lww_winner_detection() {
        let local = json!({"id": "1", "updated_at": "2024-01-15T10:00:00Z"});
        let remote = json!({"id": "1", "updated_at": "2024-01-15T11:00:00Z"});

        let entry = ConflictLogEntry::new(
            "test_table",
            "1",
            local,
            remote,
            ConflictResolution::Pending,
        );

        // Remote is newer, should win
        assert_eq!(entry.lww_winner(), Some(ConflictResolution::RemoteWins));

        // Swap timestamps
        let local = json!({"id": "1", "updated_at": "2024-01-15T12:00:00Z"});
        let remote = json!({"id": "1", "updated_at": "2024-01-15T11:00:00Z"});

        let entry = ConflictLogEntry::new(
            "test_table",
            "1",
            local,
            remote,
            ConflictResolution::Pending,
        );

        // Local is newer, should win
        assert_eq!(entry.lww_winner(), Some(ConflictResolution::LocalWins));
    }

    #[test]
    fn test_conflict_log_crud() {
        let log = ConflictLog::in_memory().unwrap();

        let local = json!({"id": "1", "name": "local"});
        let remote = json!({"id": "1", "name": "remote"});

        let entry = ConflictLogEntry::new(
            "users",
            "1",
            local.clone(),
            remote.clone(),
            ConflictResolution::Pending,
        );

        // Log conflict
        let id = log.log(&entry).unwrap();
        assert!(!id.is_empty());

        // Get by ID
        let retrieved = log.get(&id).unwrap().unwrap();
        assert_eq!(retrieved.table_name, "users");
        assert_eq!(retrieved.record_id, "1");
        assert!(retrieved.is_pending());

        // Get pending
        let pending = log.get_pending(10).unwrap();
        assert_eq!(pending.len(), 1);

        // Resolve
        log.resolve(&id, ConflictResolution::LocalWins).unwrap();

        let resolved = log.get(&id).unwrap().unwrap();
        assert_eq!(resolved.resolution, ConflictResolution::LocalWins);
        assert!(resolved.resolved_at.is_some());

        // Pending should be empty now
        let pending = log.get_pending(10).unwrap();
        assert!(pending.is_empty());
    }

    #[test]
    fn test_conflict_stats() {
        let log = ConflictLog::in_memory().unwrap();

        // Add some conflicts
        for i in 0..5 {
            let entry = ConflictLogEntry::new(
                "users",
                &i.to_string(),
                json!({"id": i}),
                json!({"id": i}),
                ConflictResolution::Pending,
            );
            log.log(&entry).unwrap();
        }

        let stats = log.stats().unwrap();
        assert_eq!(stats.pending, 5);
        assert_eq!(stats.resolved, 0);
        assert_eq!(stats.total, 5);

        // Resolve some
        let pending = log.get_pending(3).unwrap();
        for entry in pending {
            log.resolve(&entry.id, ConflictResolution::LocalWins).unwrap();
        }

        let stats = log.stats().unwrap();
        assert_eq!(stats.pending, 2);
        assert_eq!(stats.resolved, 3);
        assert_eq!(stats.local_wins, 3);
    }

    #[test]
    fn test_auto_resolve_lww() {
        let log = ConflictLog::in_memory().unwrap();

        // Add conflicts with timestamps
        let entry1 = ConflictLogEntry::new(
            "users",
            "1",
            json!({"id": "1", "updated_at": "2024-01-15T10:00:00Z"}),
            json!({"id": "1", "updated_at": "2024-01-15T11:00:00Z"}),
            ConflictResolution::Pending,
        );
        log.log(&entry1).unwrap();

        let entry2 = ConflictLogEntry::new(
            "users",
            "2",
            json!({"id": "2", "updated_at": "2024-01-15T12:00:00Z"}),
            json!({"id": "2", "updated_at": "2024-01-15T11:00:00Z"}),
            ConflictResolution::Pending,
        );
        log.log(&entry2).unwrap();

        // Add one without timestamps (will be skipped)
        let entry3 = ConflictLogEntry::new(
            "users",
            "3",
            json!({"id": "3", "name": "no_timestamp"}),
            json!({"id": "3", "name": "no_timestamp"}),
            ConflictResolution::Pending,
        );
        log.log(&entry3).unwrap();

        let result = log.auto_resolve_lww(10).unwrap();
        assert_eq!(result.resolved, 2);
        assert_eq!(result.skipped, 1);

        let stats = log.stats().unwrap();
        assert_eq!(stats.pending, 1); // Only the one without timestamps
        assert_eq!(stats.local_wins, 1);
        assert_eq!(stats.remote_wins, 1);
    }

    #[test]
    fn test_export_to_json() {
        let log = ConflictLog::in_memory().unwrap();

        let entry = ConflictLogEntry::new(
            "users",
            "1",
            json!({"id": "1", "name": "local"}),
            json!({"id": "1", "name": "remote"}),
            ConflictResolution::Pending,
        );
        log.log(&entry).unwrap();

        let json_str = log.export_to_json(10).unwrap();
        let parsed: Vec<ConflictLogEntry> = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].table_name, "users");
    }

    #[test]
    fn test_get_by_table() {
        let log = ConflictLog::in_memory().unwrap();

        // Add conflicts for different tables
        for i in 0..3 {
            let entry = ConflictLogEntry::new(
                "users",
                &i.to_string(),
                json!({"id": i}),
                json!({"id": i}),
                ConflictResolution::Pending,
            );
            log.log(&entry).unwrap();
        }

        for i in 0..2 {
            let entry = ConflictLogEntry::new(
                "orders",
                &i.to_string(),
                json!({"id": i}),
                json!({"id": i}),
                ConflictResolution::Pending,
            );
            log.log(&entry).unwrap();
        }

        let users = log.get_by_table("users", 10).unwrap();
        assert_eq!(users.len(), 3);

        let orders = log.get_by_table("orders", 10).unwrap();
        assert_eq!(orders.len(), 2);
    }

    #[test]
    fn test_log_conflict_function() {
        let log = ConflictLog::in_memory().unwrap();

        let id = log_conflict(
            &log,
            "test_table",
            "record_1",
            json!({"name": "local"}),
            json!({"name": "remote"}),
        )
        .unwrap();

        assert!(!id.is_empty());

        let entry = log.get(&id).unwrap().unwrap();
        assert_eq!(entry.table_name, "test_table");
        assert_eq!(entry.record_id, "record_1");
    }
}
