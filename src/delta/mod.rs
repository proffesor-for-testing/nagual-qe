//! Delta Event Sourcing for the Knowledge Operating System (KOS P2).
//!
//! Tracks field-level changes to patterns over time, enabling time-travel
//! queries ("what did this pattern look like 30 days ago?") and change
//! velocity analysis. Each mutation produces a delta recording exactly
//! which fields changed, with periodic full snapshots for fast reconstruction.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::db::SqliteDb;
use crate::error::Result;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single field-level difference between two pattern states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDiff {
    /// Name of the field that changed
    pub field: String,
    /// Previous value (null for new fields or Create operations)
    pub old_value: serde_json::Value,
    /// New value (null for removed fields or Delete operations)
    pub new_value: serde_json::Value,
}

impl FieldDiff {
    /// Create a new field diff.
    pub fn new(
        field: impl Into<String>,
        old_value: serde_json::Value,
        new_value: serde_json::Value,
    ) -> Self {
        Self {
            field: field.into(),
            old_value,
            new_value,
        }
    }
}

/// The kind of delta operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeltaOperation {
    /// Pattern was created (all fields are "new")
    Create,
    /// Pattern was updated (only changed fields recorded)
    Update,
    /// Pattern was deleted (snapshot of final state preserved)
    Delete,
}

impl DeltaOperation {
    /// Canonical string for storage.
    pub fn as_str(&self) -> &'static str {
        match self {
            DeltaOperation::Create => "create",
            DeltaOperation::Update => "update",
            DeltaOperation::Delete => "delete",
        }
    }
}

impl std::fmt::Display for DeltaOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for DeltaOperation {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "create" => DeltaOperation::Create,
            "update" => DeltaOperation::Update,
            "delete" => DeltaOperation::Delete,
            _ => DeltaOperation::Update, // default
        }
    }
}

/// A single delta record capturing what changed in a pattern mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternDelta {
    /// Unique ID for this delta record
    pub id: String,
    /// The pattern this delta applies to
    pub pattern_id: String,
    /// Monotonically increasing sequence number per pattern
    pub seq: u64,
    /// When this change occurred
    pub timestamp: DateTime<Utc>,
    /// Type of operation
    pub operation: DeltaOperation,
    /// Field-level diffs (empty for Delete)
    pub field_diffs: Vec<FieldDiff>,
    /// ID of the agent that made the change
    pub agent_id: Option<String>,
    /// Full pattern JSON snapshot (included every Nth delta, and always on Delete)
    pub snapshot: Option<serde_json::Value>,
}

/// Summary statistics for a pattern's change history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaSummary {
    /// Total number of deltas recorded
    pub total_deltas: u64,
    /// Number of create operations
    pub creates: u64,
    /// Number of update operations
    pub updates: u64,
    /// Number of delete operations
    pub deletes: u64,
    /// Timestamp of the first delta
    pub first_change: Option<DateTime<Utc>>,
    /// Timestamp of the most recent delta
    pub last_change: Option<DateTime<Utc>>,
    /// Number of snapshots available for fast reconstruction
    pub snapshot_count: u64,
}

// ---------------------------------------------------------------------------
// compute_diffs: field-level diff engine
// ---------------------------------------------------------------------------

/// Compute field-level diffs between two pattern JSON representations.
///
/// Compares top-level keys. If a key exists in `old` but not `new`, it is
/// treated as removed. If a key exists in `new` but not `old`, it is treated
/// as added. If both exist but differ, it is a modification.
pub fn compute_diffs(old: &serde_json::Value, new: &serde_json::Value) -> Vec<FieldDiff> {
    let mut diffs = Vec::new();

    let old_obj = match old.as_object() {
        Some(o) => o,
        None => return diffs,
    };
    let new_obj = match new.as_object() {
        Some(o) => o,
        None => return diffs,
    };

    // Check fields in old: modified or removed
    for (key, old_val) in old_obj {
        match new_obj.get(key) {
            Some(new_val) if new_val != old_val => {
                diffs.push(FieldDiff::new(key, old_val.clone(), new_val.clone()));
            }
            None => {
                diffs.push(FieldDiff::new(
                    key,
                    old_val.clone(),
                    serde_json::Value::Null,
                ));
            }
            _ => {} // unchanged
        }
    }

    // Check fields only in new: added
    for (key, new_val) in new_obj {
        if !old_obj.contains_key(key) {
            diffs.push(FieldDiff::new(
                key,
                serde_json::Value::Null,
                new_val.clone(),
            ));
        }
    }

    // Sort for deterministic output
    diffs.sort_by(|a, b| a.field.cmp(&b.field));
    diffs
}

/// Create diffs for a Create operation (all fields are "new").
pub fn diffs_for_create(state: &serde_json::Value) -> Vec<FieldDiff> {
    let obj = match state.as_object() {
        Some(o) => o,
        None => return Vec::new(),
    };

    let mut diffs: Vec<FieldDiff> = obj
        .iter()
        .map(|(key, val)| FieldDiff::new(key, serde_json::Value::Null, val.clone()))
        .collect();

    diffs.sort_by(|a, b| a.field.cmp(&b.field));
    diffs
}

/// Create diffs for a Delete operation (all fields become null).
pub fn diffs_for_delete(state: &serde_json::Value) -> Vec<FieldDiff> {
    let obj = match state.as_object() {
        Some(o) => o,
        None => return Vec::new(),
    };

    let mut diffs: Vec<FieldDiff> = obj
        .iter()
        .map(|(key, val)| FieldDiff::new(key, val.clone(), serde_json::Value::Null))
        .collect();

    diffs.sort_by(|a, b| a.field.cmp(&b.field));
    diffs
}

// ---------------------------------------------------------------------------
// DeltaStore: persistence layer
// ---------------------------------------------------------------------------

/// Persistent store for pattern deltas.
///
/// Records create/update/delete deltas, stores periodic snapshots, and
/// provides time-travel reconstruction and change velocity queries.
pub struct DeltaStore {
    db: Arc<SqliteDb>,
    snapshot_interval: u64,
    lineage: Option<Arc<crate::lineage::LineageQuery>>,
}

impl DeltaStore {
    /// Create a new DeltaStore.
    pub fn new(db: Arc<SqliteDb>, snapshot_interval: u64) -> Self {
        Self {
            db,
            snapshot_interval,
            lineage: None,
        }
    }

    /// Attach a lineage query for recording delta-to-lineage ancestry.
    pub fn with_lineage(mut self, lineage: Arc<crate::lineage::LineageQuery>) -> Self {
        self.lineage = Some(lineage);
        self
    }

    /// Create a DeltaStore with default snapshot interval (every 10 deltas).
    pub fn with_defaults(db: Arc<SqliteDb>) -> Self {
        Self::new(db, 10)
    }

    /// Initialize the schema (create table if not exists).
    pub async fn init(&self) -> Result<()> {
        self.db
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS pattern_deltas (
                    id          TEXT PRIMARY KEY,
                    pattern_id  TEXT NOT NULL,
                    seq         INTEGER NOT NULL,
                    timestamp   TEXT NOT NULL,
                    operation   TEXT NOT NULL,
                    field_diffs TEXT NOT NULL,
                    agent_id    TEXT,
                    snapshot    TEXT,
                    UNIQUE(pattern_id, seq)
                );
                CREATE INDEX IF NOT EXISTS idx_deltas_pattern_id ON pattern_deltas(pattern_id);
                CREATE INDEX IF NOT EXISTS idx_deltas_timestamp ON pattern_deltas(timestamp);",
            )
            .await?;
        Ok(())
    }

    /// Record a delta between old and new pattern state.
    ///
    /// For Create: `old` should be None, `new_state` is the full JSON.
    /// For Update: both `old` and `new_state` provided; only changed fields recorded.
    /// For Delete: `old` is the final state, `new_state` can be empty JSON.
    pub async fn record(
        &self,
        pattern_id: &str,
        old: Option<&serde_json::Value>,
        new_state: &serde_json::Value,
        op: DeltaOperation,
        agent_id: Option<&str>,
    ) -> Result<PatternDelta> {
        let next_seq = self.next_seq(pattern_id).await?;
        let now = Utc::now();
        let id = uuid::Uuid::new_v4().to_string();

        let field_diffs = match op {
            DeltaOperation::Create => diffs_for_create(new_state),
            DeltaOperation::Update => match old {
                Some(old_val) => compute_diffs(old_val, new_state),
                None => diffs_for_create(new_state),
            },
            DeltaOperation::Delete => match old {
                Some(old_val) => diffs_for_delete(old_val),
                None => Vec::new(),
            },
        };

        // Include snapshot on Create, Delete, or every Nth delta
        let snapshot = if op == DeltaOperation::Create
            || op == DeltaOperation::Delete
            || (self.snapshot_interval > 0 && next_seq % self.snapshot_interval == 0)
        {
            match op {
                DeltaOperation::Delete => old.cloned(),
                _ => Some(new_state.clone()),
            }
        } else {
            None
        };

        let diffs_json = serde_json::to_string(&field_diffs)
            .unwrap_or_else(|_| "[]".to_string());
        let snapshot_json: Option<String> = snapshot
            .as_ref()
            .map(|s| serde_json::to_string(s).unwrap_or_else(|_| "{}".to_string()));
        let ts_str = now.to_rfc3339();
        let op_str = op.as_str().to_string();
        let agent_str: Option<String> = agent_id.map(|s| s.to_string());
        let seq_i64 = next_seq as i64;

        self.db
            .execute(
                "INSERT INTO pattern_deltas (id, pattern_id, seq, timestamp, operation, field_diffs, agent_id, snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                &[
                    &id as &dyn rusqlite::ToSql,
                    &pattern_id as &dyn rusqlite::ToSql,
                    &seq_i64 as &dyn rusqlite::ToSql,
                    &ts_str as &dyn rusqlite::ToSql,
                    &op_str as &dyn rusqlite::ToSql,
                    &diffs_json as &dyn rusqlite::ToSql,
                    &agent_str as &dyn rusqlite::ToSql,
                    &snapshot_json as &dyn rusqlite::ToSql,
                ],
            )
            .await?;

        Ok(PatternDelta {
            id,
            pattern_id: pattern_id.to_string(),
            seq: next_seq,
            timestamp: now,
            operation: op,
            field_diffs,
            agent_id: agent_id.map(|s| s.to_string()),
            snapshot,
        })
    }

    /// Get the next sequence number for a pattern.
    async fn next_seq(&self, pattern_id: &str) -> Result<u64> {
        let pid = pattern_id.to_string();
        let max_seq = self
            .db
            .query_one(
                "SELECT COALESCE(MAX(seq), -1) FROM pattern_deltas WHERE pattern_id = ?",
                &[&pid as &dyn rusqlite::ToSql],
                |row| row.get::<_, i64>(0),
            )
            .await?
            .unwrap_or(-1);

        Ok((max_seq + 1) as u64)
    }

    /// Get all deltas for a pattern, ordered by sequence number.
    pub async fn history(&self, pattern_id: &str) -> Result<Vec<PatternDelta>> {
        let pid = pattern_id.to_string();
        self.db
            .query(
                "SELECT id, pattern_id, seq, timestamp, operation, field_diffs, agent_id, snapshot
                 FROM pattern_deltas WHERE pattern_id = ? ORDER BY seq ASC",
                &[&pid as &dyn rusqlite::ToSql],
                row_to_delta,
            )
            .await
    }

    /// Get deltas within a time window.
    pub async fn window(
        &self,
        pattern_id: &str,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<PatternDelta>> {
        let pid = pattern_id.to_string();
        let from_str = from.to_rfc3339();
        let to_str = to.to_rfc3339();
        self.db
            .query(
                "SELECT id, pattern_id, seq, timestamp, operation, field_diffs, agent_id, snapshot
                 FROM pattern_deltas
                 WHERE pattern_id = ? AND timestamp >= ? AND timestamp <= ?
                 ORDER BY seq ASC",
                &[
                    &pid as &dyn rusqlite::ToSql,
                    &from_str as &dyn rusqlite::ToSql,
                    &to_str as &dyn rusqlite::ToSql,
                ],
                row_to_delta,
            )
            .await
    }

    /// Reconstruct pattern state at a given timestamp.
    ///
    /// Finds the most recent snapshot at or before `at`, then replays
    /// subsequent deltas up to `at` to rebuild the exact state.
    pub async fn reconstruct_at(
        &self,
        pattern_id: &str,
        at: DateTime<Utc>,
    ) -> Result<Option<serde_json::Value>> {
        let pid = pattern_id.to_string();
        let at_str = at.to_rfc3339();

        // Find the most recent snapshot at or before the target time
        let snapshot_opt = self
            .db
            .query_one(
                "SELECT seq, snapshot, operation FROM pattern_deltas
                 WHERE pattern_id = ? AND snapshot IS NOT NULL AND snapshot != ''
                   AND timestamp <= ?
                 ORDER BY seq DESC LIMIT 1",
                &[
                    &pid as &dyn rusqlite::ToSql,
                    &at_str as &dyn rusqlite::ToSql,
                ],
                |row| {
                    let seq: i64 = row.get(0)?;
                    let snapshot_str: String = row.get(1)?;
                    let op_str: String = row.get(2)?;
                    Ok((seq, snapshot_str, op_str))
                },
            )
            .await?;

        let (base_state, base_seq) = match snapshot_opt {
            Some((_, _, ref op_str)) if op_str == "delete" => {
                // The most recent snapshot came from a Delete — pattern was deleted
                return Ok(None);
            }
            Some((seq, snapshot_str, _)) => {
                if snapshot_str.is_empty() {
                    return Ok(None);
                }
                let state: serde_json::Value =
                    serde_json::from_str(&snapshot_str).unwrap_or(serde_json::Value::Null);
                (state, seq)
            }
            None => return Ok(None),
        };

        // Get all deltas after the snapshot up to the target time
        let deltas = self
            .db
            .query(
                "SELECT id, pattern_id, seq, timestamp, operation, field_diffs, agent_id, snapshot
                 FROM pattern_deltas
                 WHERE pattern_id = ? AND seq > ? AND timestamp <= ?
                 ORDER BY seq ASC",
                &[
                    &pid as &dyn rusqlite::ToSql,
                    &base_seq as &dyn rusqlite::ToSql,
                    &at_str as &dyn rusqlite::ToSql,
                ],
                row_to_delta,
            )
            .await?;

        // Replay deltas on top of the snapshot
        let mut state = base_state;
        for delta in &deltas {
            if delta.operation == DeltaOperation::Delete {
                return Ok(None); // Pattern was deleted before target time
            }
            state = apply_diffs(&state, &delta.field_diffs);
        }

        Ok(Some(state))
    }

    /// Compute aggregate change velocity: number of field changes per day
    /// within the specified window.
    pub async fn change_velocity(
        &self,
        pattern_id: &str,
        window_days: u32,
    ) -> Result<f64> {
        let pid = pattern_id.to_string();

        let diffs_strings: Vec<String> = if window_days == 0 {
            // window_days=0 means count all deltas (no date filter)
            self.db
                .query(
                    "SELECT field_diffs FROM pattern_deltas WHERE pattern_id = ?",
                    &[&pid as &dyn rusqlite::ToSql],
                    |row| row.get::<_, String>(0),
                )
                .await?
        } else {
            let cutoff = Utc::now() - chrono::Duration::days(window_days as i64);
            let cutoff_str = cutoff.to_rfc3339();
            self.db
                .query(
                    "SELECT field_diffs FROM pattern_deltas
                     WHERE pattern_id = ? AND timestamp >= ?",
                    &[
                        &pid as &dyn rusqlite::ToSql,
                        &cutoff_str as &dyn rusqlite::ToSql,
                    ],
                    |row| row.get::<_, String>(0),
                )
                .await?
        };

        let total_changes: usize = diffs_strings
            .iter()
            .map(|diffs_str| {
                let diffs: Vec<FieldDiff> =
                    serde_json::from_str(diffs_str).unwrap_or_default();
                diffs.len()
            })
            .sum();

        if window_days == 0 {
            return Ok(total_changes as f64);
        }

        Ok(total_changes as f64 / window_days as f64)
    }

    /// Get the latest delta for a pattern.
    pub async fn latest(&self, pattern_id: &str) -> Result<Option<PatternDelta>> {
        let pid = pattern_id.to_string();
        self.db
            .query_one(
                "SELECT id, pattern_id, seq, timestamp, operation, field_diffs, agent_id, snapshot
                 FROM pattern_deltas WHERE pattern_id = ? ORDER BY seq DESC LIMIT 1",
                &[&pid as &dyn rusqlite::ToSql],
                row_to_delta,
            )
            .await
    }

    /// Get summary statistics for a pattern's deltas.
    pub async fn summary(&self, pattern_id: &str) -> Result<DeltaSummary> {
        let pid = pattern_id.to_string();
        let result = self
            .db
            .query_one(
                "SELECT
                    COUNT(*),
                    SUM(CASE WHEN operation = 'create' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN operation = 'update' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN operation = 'delete' THEN 1 ELSE 0 END),
                    MIN(timestamp),
                    MAX(timestamp),
                    SUM(CASE WHEN snapshot IS NOT NULL AND snapshot != '' THEN 1 ELSE 0 END)
                 FROM pattern_deltas WHERE pattern_id = ?",
                &[&pid as &dyn rusqlite::ToSql],
                |row| {
                    let total: i64 = row.get(0)?;
                    let creates: i64 = row.get(1)?;
                    let updates: i64 = row.get(2)?;
                    let deletes: i64 = row.get(3)?;
                    let first_ts: Option<String> = row.get(4)?;
                    let last_ts: Option<String> = row.get(5)?;
                    let snapshots: i64 = row.get(6)?;
                    Ok((total, creates, updates, deletes, first_ts, last_ts, snapshots))
                },
            )
            .await?;

        match result {
            Some((total, creates, updates, deletes, first_ts, last_ts, snapshots)) => {
                let first_change = first_ts
                    .and_then(|ts| DateTime::parse_from_rfc3339(&ts).ok())
                    .map(|dt| dt.with_timezone(&Utc));
                let last_change = last_ts
                    .and_then(|ts| DateTime::parse_from_rfc3339(&ts).ok())
                    .map(|dt| dt.with_timezone(&Utc));

                Ok(DeltaSummary {
                    total_deltas: total as u64,
                    creates: creates as u64,
                    updates: updates as u64,
                    deletes: deletes as u64,
                    first_change,
                    last_change,
                    snapshot_count: snapshots as u64,
                })
            }
            None => Ok(DeltaSummary {
                total_deltas: 0,
                creates: 0,
                updates: 0,
                deletes: 0,
                first_change: None,
                last_change: None,
                snapshot_count: 0,
            }),
        }
    }

    /// Get the number of deltas recorded for a pattern.
    pub async fn count(&self, pattern_id: &str) -> Result<u64> {
        let pid = pattern_id.to_string();
        let cnt = self
            .db
            .query_one(
                "SELECT COUNT(*) FROM pattern_deltas WHERE pattern_id = ?",
                &[&pid as &dyn rusqlite::ToSql],
                |row| row.get::<_, i64>(0),
            )
            .await?
            .unwrap_or(0);

        Ok(cnt as u64)
    }

    /// Get the snapshot interval.
    pub fn snapshot_interval(&self) -> u64 {
        self.snapshot_interval
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Apply a set of field diffs to a JSON value, producing the new state.
fn apply_diffs(state: &serde_json::Value, diffs: &[FieldDiff]) -> serde_json::Value {
    let mut obj = match state.as_object() {
        Some(o) => o.clone(),
        None => return state.clone(),
    };

    for diff in diffs {
        if diff.new_value.is_null() {
            obj.remove(&diff.field);
        } else {
            obj.insert(diff.field.clone(), diff.new_value.clone());
        }
    }

    serde_json::Value::Object(obj)
}

/// Convert a database row to a PatternDelta.
fn row_to_delta(row: &rusqlite::Row<'_>) -> rusqlite::Result<PatternDelta> {
    let id: String = row.get(0)?;
    let pattern_id: String = row.get(1)?;
    let seq: i64 = row.get(2)?;
    let ts_str: String = row.get(3)?;
    let op_str: String = row.get(4)?;
    let diffs_str: String = row.get(5)?;
    let agent_id: Option<String> = row.get(6)?;
    let snapshot_str: Option<String> = row.get(7)?;

    let timestamp = DateTime::parse_from_rfc3339(&ts_str)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());

    let operation = DeltaOperation::from(op_str.as_str());

    let field_diffs: Vec<FieldDiff> =
        serde_json::from_str(&diffs_str).unwrap_or_default();

    let snapshot = snapshot_str
        .filter(|s| !s.is_empty())
        .and_then(|s| serde_json::from_str(&s).ok());

    Ok(PatternDelta {
        id,
        pattern_id,
        seq: seq as u64,
        timestamp,
        operation,
        field_diffs,
        agent_id,
        snapshot,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- compute_diffs tests ---

    #[test]
    fn test_compute_diffs_no_change() {
        let old = serde_json::json!({"a": 1, "b": "hello"});
        let new = serde_json::json!({"a": 1, "b": "hello"});
        let diffs = compute_diffs(&old, &new);
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_compute_diffs_single_field_modified() {
        let old = serde_json::json!({"reward": 0.5, "domain": "rust"});
        let new = serde_json::json!({"reward": 0.8, "domain": "rust"});
        let diffs = compute_diffs(&old, &new);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].field, "reward");
        assert_eq!(diffs[0].old_value, serde_json::json!(0.5));
        assert_eq!(diffs[0].new_value, serde_json::json!(0.8));
    }

    #[test]
    fn test_compute_diffs_multiple_fields() {
        let old = serde_json::json!({"a": 1, "b": 2, "c": 3});
        let new = serde_json::json!({"a": 10, "b": 2, "c": 30});
        let diffs = compute_diffs(&old, &new);
        assert_eq!(diffs.len(), 2);
        let fields: Vec<&str> = diffs.iter().map(|d| d.field.as_str()).collect();
        assert!(fields.contains(&"a"));
        assert!(fields.contains(&"c"));
    }

    #[test]
    fn test_compute_diffs_field_added() {
        let old = serde_json::json!({"a": 1});
        let new = serde_json::json!({"a": 1, "b": 2});
        let diffs = compute_diffs(&old, &new);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].field, "b");
        assert_eq!(diffs[0].old_value, serde_json::Value::Null);
        assert_eq!(diffs[0].new_value, serde_json::json!(2));
    }

    #[test]
    fn test_compute_diffs_field_removed() {
        let old = serde_json::json!({"a": 1, "b": 2});
        let new = serde_json::json!({"a": 1});
        let diffs = compute_diffs(&old, &new);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].field, "b");
        assert_eq!(diffs[0].old_value, serde_json::json!(2));
        assert_eq!(diffs[0].new_value, serde_json::Value::Null);
    }

    #[test]
    fn test_compute_diffs_type_change() {
        let old = serde_json::json!({"val": "text"});
        let new = serde_json::json!({"val": 42});
        let diffs = compute_diffs(&old, &new);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].old_value, serde_json::json!("text"));
        assert_eq!(diffs[0].new_value, serde_json::json!(42));
    }

    #[test]
    fn test_compute_diffs_non_object_returns_empty() {
        let old = serde_json::json!("string");
        let new = serde_json::json!(42);
        let diffs = compute_diffs(&old, &new);
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_compute_diffs_sorted_output() {
        let old = serde_json::json!({"z": 1, "a": 1, "m": 1});
        let new = serde_json::json!({"z": 2, "a": 2, "m": 2});
        let diffs = compute_diffs(&old, &new);
        assert_eq!(diffs.len(), 3);
        assert_eq!(diffs[0].field, "a");
        assert_eq!(diffs[1].field, "m");
        assert_eq!(diffs[2].field, "z");
    }

    // --- diffs_for_create / diffs_for_delete ---

    #[test]
    fn test_diffs_for_create() {
        let state = serde_json::json!({"problem": "test", "reward": 0.5});
        let diffs = diffs_for_create(&state);
        assert_eq!(diffs.len(), 2);
        for d in &diffs {
            assert_eq!(d.old_value, serde_json::Value::Null);
        }
    }

    #[test]
    fn test_diffs_for_delete() {
        let state = serde_json::json!({"problem": "test", "reward": 0.5});
        let diffs = diffs_for_delete(&state);
        assert_eq!(diffs.len(), 2);
        for d in &diffs {
            assert_eq!(d.new_value, serde_json::Value::Null);
        }
    }

    // --- DeltaOperation ---

    #[test]
    fn test_delta_operation_display() {
        assert_eq!(DeltaOperation::Create.to_string(), "create");
        assert_eq!(DeltaOperation::Update.to_string(), "update");
        assert_eq!(DeltaOperation::Delete.to_string(), "delete");
    }

    #[test]
    fn test_delta_operation_from_str() {
        assert_eq!(DeltaOperation::from("create"), DeltaOperation::Create);
        assert_eq!(DeltaOperation::from("update"), DeltaOperation::Update);
        assert_eq!(DeltaOperation::from("delete"), DeltaOperation::Delete);
        assert_eq!(DeltaOperation::from("CREATE"), DeltaOperation::Create);
        assert_eq!(DeltaOperation::from("unknown"), DeltaOperation::Update);
    }

    #[test]
    fn test_delta_operation_serde_roundtrip() {
        let op = DeltaOperation::Create;
        let json = serde_json::to_string(&op).unwrap();
        let deserialized: DeltaOperation = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, op);
    }

    // --- apply_diffs ---

    #[test]
    fn test_apply_diffs_modify_field() {
        let state = serde_json::json!({"a": 1, "b": 2});
        let diffs = vec![FieldDiff::new("a", serde_json::json!(1), serde_json::json!(10))];
        let result = apply_diffs(&state, &diffs);
        assert_eq!(result, serde_json::json!({"a": 10, "b": 2}));
    }

    #[test]
    fn test_apply_diffs_add_field() {
        let state = serde_json::json!({"a": 1});
        let diffs = vec![FieldDiff::new(
            "b",
            serde_json::Value::Null,
            serde_json::json!(2),
        )];
        let result = apply_diffs(&state, &diffs);
        assert_eq!(result["a"], serde_json::json!(1));
        assert_eq!(result["b"], serde_json::json!(2));
    }

    #[test]
    fn test_apply_diffs_remove_field() {
        let state = serde_json::json!({"a": 1, "b": 2});
        let diffs = vec![FieldDiff::new(
            "b",
            serde_json::json!(2),
            serde_json::Value::Null,
        )];
        let result = apply_diffs(&state, &diffs);
        assert_eq!(result, serde_json::json!({"a": 1}));
    }

    #[test]
    fn test_apply_diffs_empty() {
        let state = serde_json::json!({"a": 1});
        let result = apply_diffs(&state, &[]);
        assert_eq!(result, state);
    }

    // --- FieldDiff serde ---

    #[test]
    fn test_field_diff_serde_roundtrip() {
        let diff = FieldDiff::new("reward", serde_json::json!(0.5), serde_json::json!(0.9));
        let json = serde_json::to_string(&diff).unwrap();
        let deserialized: FieldDiff = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, diff);
    }

    // --- PatternDelta serde ---

    #[test]
    fn test_pattern_delta_serde_roundtrip() {
        let delta = PatternDelta {
            id: "delta-1".to_string(),
            pattern_id: "pat-1".to_string(),
            seq: 0,
            timestamp: Utc::now(),
            operation: DeltaOperation::Create,
            field_diffs: vec![
                FieldDiff::new("problem", serde_json::Value::Null, serde_json::json!("test")),
            ],
            agent_id: Some("agent-1".to_string()),
            snapshot: Some(serde_json::json!({"problem": "test"})),
        };
        let json = serde_json::to_string(&delta).unwrap();
        let deserialized: PatternDelta = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, delta.id);
        assert_eq!(deserialized.pattern_id, delta.pattern_id);
        assert_eq!(deserialized.seq, delta.seq);
        assert_eq!(deserialized.operation, delta.operation);
        assert_eq!(deserialized.field_diffs.len(), delta.field_diffs.len());
    }

    // --- DeltaStore async tests ---

    async fn test_db() -> Arc<SqliteDb> {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let store = DeltaStore::new(Arc::clone(&db), 10);
        store.init().await.unwrap();
        db
    }

    #[tokio::test]
    async fn test_store_init() {
        let db = test_db().await;
        // Table should exist; a second init should not fail
        let store = DeltaStore::new(db, 10);
        store.init().await.unwrap();
    }

    #[tokio::test]
    async fn test_store_record_create() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);
        let state = serde_json::json!({"problem": "test", "reward": 0.5});

        let delta = store
            .record("pat-1", None, &state, DeltaOperation::Create, Some("agent-1"))
            .await
            .unwrap();

        assert_eq!(delta.pattern_id, "pat-1");
        assert_eq!(delta.seq, 0);
        assert_eq!(delta.operation, DeltaOperation::Create);
        assert_eq!(delta.field_diffs.len(), 2); // problem + reward
        assert!(delta.snapshot.is_some()); // Create always includes snapshot
        assert_eq!(delta.agent_id, Some("agent-1".to_string()));
    }

    #[tokio::test]
    async fn test_store_record_update_only_changed_fields() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);

        let old = serde_json::json!({"problem": "test", "reward": 0.5, "domain": "rust"});
        let new = serde_json::json!({"problem": "test", "reward": 0.8, "domain": "rust"});

        // First create a Create delta so seq starts at 0
        store
            .record("pat-1", None, &old, DeltaOperation::Create, None)
            .await
            .unwrap();

        let delta = store
            .record("pat-1", Some(&old), &new, DeltaOperation::Update, None)
            .await
            .unwrap();

        assert_eq!(delta.seq, 1);
        assert_eq!(delta.operation, DeltaOperation::Update);
        assert_eq!(delta.field_diffs.len(), 1); // only reward changed
        assert_eq!(delta.field_diffs[0].field, "reward");
    }

    #[tokio::test]
    async fn test_store_record_delete_includes_snapshot() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);

        let state = serde_json::json!({"problem": "test", "reward": 0.5});

        store
            .record("pat-1", None, &state, DeltaOperation::Create, None)
            .await
            .unwrap();

        let delta = store
            .record(
                "pat-1",
                Some(&state),
                &serde_json::json!({}),
                DeltaOperation::Delete,
                None,
            )
            .await
            .unwrap();

        assert_eq!(delta.operation, DeltaOperation::Delete);
        assert!(delta.snapshot.is_some()); // Delete always includes snapshot
        assert_eq!(delta.snapshot.unwrap(), state);
    }

    #[tokio::test]
    async fn test_store_history() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);

        let s1 = serde_json::json!({"problem": "v1", "reward": 0.5});
        let s2 = serde_json::json!({"problem": "v1", "reward": 0.8});
        let s3 = serde_json::json!({"problem": "v2", "reward": 0.8});

        store
            .record("pat-1", None, &s1, DeltaOperation::Create, None)
            .await
            .unwrap();
        store
            .record("pat-1", Some(&s1), &s2, DeltaOperation::Update, None)
            .await
            .unwrap();
        store
            .record("pat-1", Some(&s2), &s3, DeltaOperation::Update, None)
            .await
            .unwrap();

        let history = store.history("pat-1").await.unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].seq, 0);
        assert_eq!(history[1].seq, 1);
        assert_eq!(history[2].seq, 2);
    }

    #[tokio::test]
    async fn test_store_latest() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);

        let s1 = serde_json::json!({"problem": "v1"});
        let s2 = serde_json::json!({"problem": "v2"});

        store
            .record("pat-1", None, &s1, DeltaOperation::Create, None)
            .await
            .unwrap();
        store
            .record("pat-1", Some(&s1), &s2, DeltaOperation::Update, None)
            .await
            .unwrap();

        let latest = store.latest("pat-1").await.unwrap();
        assert!(latest.is_some());
        assert_eq!(latest.unwrap().seq, 1);
    }

    #[tokio::test]
    async fn test_store_latest_none() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);
        let latest = store.latest("nonexistent").await.unwrap();
        assert!(latest.is_none());
    }

    #[tokio::test]
    async fn test_store_count() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);

        assert_eq!(store.count("pat-1").await.unwrap(), 0);

        let s1 = serde_json::json!({"problem": "v1"});
        store
            .record("pat-1", None, &s1, DeltaOperation::Create, None)
            .await
            .unwrap();

        assert_eq!(store.count("pat-1").await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_store_summary() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);

        let s1 = serde_json::json!({"problem": "v1", "reward": 0.5});
        let s2 = serde_json::json!({"problem": "v1", "reward": 0.8});

        store
            .record("pat-1", None, &s1, DeltaOperation::Create, None)
            .await
            .unwrap();
        store
            .record("pat-1", Some(&s1), &s2, DeltaOperation::Update, None)
            .await
            .unwrap();

        let summary = store.summary("pat-1").await.unwrap();
        assert_eq!(summary.total_deltas, 2);
        assert_eq!(summary.creates, 1);
        assert_eq!(summary.updates, 1);
        assert_eq!(summary.deletes, 0);
        assert!(summary.first_change.is_some());
        assert!(summary.last_change.is_some());
        assert!(summary.snapshot_count >= 1); // Create includes snapshot
    }

    #[tokio::test]
    async fn test_snapshot_interval() {
        let db = test_db().await;
        // Snapshot every 3 deltas
        let store = DeltaStore::new(db, 3);

        let s = serde_json::json!({"problem": "test"});

        // seq=0: Create (always has snapshot)
        let d0 = store
            .record("pat-1", None, &s, DeltaOperation::Create, None)
            .await
            .unwrap();
        assert!(d0.snapshot.is_some());

        // seq=1: Update (no snapshot, 1 % 3 != 0)
        let d1 = store
            .record("pat-1", Some(&s), &s, DeltaOperation::Update, None)
            .await
            .unwrap();
        assert!(d1.snapshot.is_none());

        // seq=2: Update (no snapshot, 2 % 3 != 0)
        let d2 = store
            .record("pat-1", Some(&s), &s, DeltaOperation::Update, None)
            .await
            .unwrap();
        assert!(d2.snapshot.is_none());

        // seq=3: Update (snapshot! 3 % 3 == 0)
        let d3 = store
            .record("pat-1", Some(&s), &s, DeltaOperation::Update, None)
            .await
            .unwrap();
        assert!(d3.snapshot.is_some());
    }

    #[tokio::test]
    async fn test_reconstruct_at_returns_none_for_unknown_pattern() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);
        let result = store
            .reconstruct_at("nonexistent", Utc::now())
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_reconstruct_at_from_create_snapshot() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);

        let s1 = serde_json::json!({"problem": "v1", "reward": 0.5});
        store
            .record("pat-1", None, &s1, DeltaOperation::Create, None)
            .await
            .unwrap();

        // Reconstruct should return the create state
        let result = store
            .reconstruct_at("pat-1", Utc::now())
            .await
            .unwrap();
        assert!(result.is_some());
        let state = result.unwrap();
        assert_eq!(state["problem"], "v1");
        assert_eq!(state["reward"], 0.5);
    }

    #[tokio::test]
    async fn test_reconstruct_at_replays_updates() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);

        let s1 = serde_json::json!({"problem": "v1", "reward": 0.5});
        let s2 = serde_json::json!({"problem": "v1", "reward": 0.8});
        let s3 = serde_json::json!({"problem": "v2", "reward": 0.8, "domain": "rust"});

        store
            .record("pat-1", None, &s1, DeltaOperation::Create, None)
            .await
            .unwrap();
        store
            .record("pat-1", Some(&s1), &s2, DeltaOperation::Update, None)
            .await
            .unwrap();
        store
            .record("pat-1", Some(&s2), &s3, DeltaOperation::Update, None)
            .await
            .unwrap();

        let result = store
            .reconstruct_at("pat-1", Utc::now())
            .await
            .unwrap();
        assert!(result.is_some());
        let state = result.unwrap();
        assert_eq!(state["problem"], "v2");
        assert_eq!(state["reward"], 0.8);
        assert_eq!(state["domain"], "rust");
    }

    #[tokio::test]
    async fn test_reconstruct_at_after_delete_returns_none() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);

        let s1 = serde_json::json!({"problem": "v1"});
        store
            .record("pat-1", None, &s1, DeltaOperation::Create, None)
            .await
            .unwrap();
        store
            .record(
                "pat-1",
                Some(&s1),
                &serde_json::json!({}),
                DeltaOperation::Delete,
                None,
            )
            .await
            .unwrap();

        let result = store
            .reconstruct_at("pat-1", Utc::now())
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_change_velocity() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);

        let s1 = serde_json::json!({"a": 1, "b": 2});
        let s2 = serde_json::json!({"a": 10, "b": 20});

        store
            .record("pat-1", None, &s1, DeltaOperation::Create, None)
            .await
            .unwrap();
        store
            .record("pat-1", Some(&s1), &s2, DeltaOperation::Update, None)
            .await
            .unwrap();

        // Over 30 days window, should have field changes
        let velocity = store.change_velocity("pat-1", 30).await.unwrap();
        assert!(velocity > 0.0);
    }

    #[tokio::test]
    async fn test_change_velocity_zero_window() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);

        let s1 = serde_json::json!({"a": 1});
        store
            .record("pat-1", None, &s1, DeltaOperation::Create, None)
            .await
            .unwrap();

        // window_days=0 returns absolute count
        let velocity = store.change_velocity("pat-1", 0).await.unwrap();
        assert!(velocity >= 1.0);
    }

    #[tokio::test]
    async fn test_multi_pattern_isolation() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);

        let s = serde_json::json!({"problem": "test"});

        store
            .record("pat-1", None, &s, DeltaOperation::Create, None)
            .await
            .unwrap();
        store
            .record("pat-2", None, &s, DeltaOperation::Create, None)
            .await
            .unwrap();
        store
            .record("pat-1", Some(&s), &s, DeltaOperation::Update, None)
            .await
            .unwrap();

        assert_eq!(store.count("pat-1").await.unwrap(), 2);
        assert_eq!(store.count("pat-2").await.unwrap(), 1);

        let h1 = store.history("pat-1").await.unwrap();
        let h2 = store.history("pat-2").await.unwrap();
        assert_eq!(h1.len(), 2);
        assert_eq!(h2.len(), 1);
    }

    #[tokio::test]
    async fn test_seq_monotonic() {
        let db = test_db().await;
        let store = DeltaStore::with_defaults(db);

        let s = serde_json::json!({"problem": "test"});

        for _ in 0..5 {
            store
                .record("pat-1", None, &s, DeltaOperation::Create, None)
                .await
                .unwrap();
        }

        let history = store.history("pat-1").await.unwrap();
        for (i, delta) in history.iter().enumerate() {
            assert_eq!(delta.seq, i as u64);
        }
    }

    #[test]
    fn test_delta_summary_serde_roundtrip() {
        let summary = DeltaSummary {
            total_deltas: 10,
            creates: 1,
            updates: 8,
            deletes: 1,
            first_change: Some(Utc::now()),
            last_change: Some(Utc::now()),
            snapshot_count: 2,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let deserialized: DeltaSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.total_deltas, summary.total_deltas);
        assert_eq!(deserialized.creates, summary.creates);
        assert_eq!(deserialized.snapshot_count, summary.snapshot_count);
    }
}
