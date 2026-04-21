//! Witness Chains for the Knowledge Operating System (KOS P1).
//!
//! Provides an append-only tamper-evident log for all pattern mutations.
//! Each entry is BLAKE3-chained to its predecessor, forming an integrity chain
//! that can be verified at any time. This builds on P0 (Lineage) to add
//! cryptographic proof that the lineage history has not been altered.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::db::SqliteDb;
use crate::error::Result;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// The kind of mutation that occurred on a pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WitnessOperation {
    Store,
    Update,
    Delete,
    Merge,
}

impl WitnessOperation {
    /// Return the canonical string representation used in hashing and storage.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Store => "store",
            Self::Update => "update",
            Self::Delete => "delete",
            Self::Merge => "merge",
        }
    }
}

impl std::fmt::Display for WitnessOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for WitnessOperation {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "store" => Self::Store,
            "update" => Self::Update,
            "delete" => Self::Delete,
            "merge" => Self::Merge,
            _ => Self::Store,
        }
    }
}

/// Classification of what the witness entry attests to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum WitnessType {
    /// Attests to origin / authorship.
    Provenance = 1,
    /// Attests to a computation (embedding, scoring, etc.).
    Computation = 2,
    /// Attests to a consolidation / merge operation.
    Consolidation = 3,
}

impl WitnessType {
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            2 => Self::Computation,
            3 => Self::Consolidation,
            _ => Self::Provenance,
        }
    }
}

impl std::fmt::Display for WitnessType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provenance => write!(f, "provenance"),
            Self::Computation => write!(f, "computation"),
            Self::Consolidation => write!(f, "consolidation"),
        }
    }
}

/// A single entry in the witness chain.
#[derive(Debug, Clone)]
pub struct WitnessEntry {
    /// Auto-incrementing sequence number (primary key).
    pub seq: u64,
    /// The pattern this witness attests to.
    pub pattern_id: String,
    /// What mutation occurred.
    pub operation: WitnessOperation,
    /// BLAKE3(pattern_id || operation || metadata).
    pub action_hash: [u8; 32],
    /// BLAKE3(previous entry) or all-zeros for the genesis entry.
    pub prev_hash: [u8; 32],
    /// BLAKE3(prev_hash || action_hash || timestamp || witness_type).
    pub entry_hash: [u8; 32],
    /// When the witness was recorded.
    pub timestamp: DateTime<Utc>,
    /// What this witness attests to.
    pub witness_type: WitnessType,
    /// Optional agent / user that performed the action.
    pub agent_id: Option<String>,
    /// Optional free-form metadata (JSON, field snapshot, etc.).
    pub metadata: Option<String>,
}

/// Result of verifying a witness chain.
#[derive(Debug, Clone)]
pub struct WitnessVerification {
    /// Whether the entire checked range is valid.
    pub valid: bool,
    /// How many entries were checked.
    pub entries_checked: u64,
    /// The first sequence number where the chain broke (if any).
    pub first_broken_seq: Option<u64>,
    /// Total length of the chain that was inspected.
    pub chain_length: u64,
}

// ---------------------------------------------------------------------------
// BLAKE3 hash helpers
// ---------------------------------------------------------------------------

/// Compute the action hash: BLAKE3(pattern_id || operation || metadata).
fn compute_action_hash(
    pattern_id: &str,
    operation: &WitnessOperation,
    metadata: Option<&str>,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(pattern_id.as_bytes());
    hasher.update(operation.as_str().as_bytes());
    if let Some(m) = metadata {
        hasher.update(m.as_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// Compute the entry hash: BLAKE3(prev_hash || action_hash || timestamp || witness_type).
fn compute_entry_hash(
    prev_hash: &[u8; 32],
    action_hash: &[u8; 32],
    timestamp: &str,
    witness_type: u8,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(prev_hash);
    hasher.update(action_hash);
    hasher.update(timestamp.as_bytes());
    hasher.update(&[witness_type]);
    *hasher.finalize().as_bytes()
}

// ---------------------------------------------------------------------------
// WitnessChain
// ---------------------------------------------------------------------------

/// Append-only tamper-evident log backed by SQLite.
pub struct WitnessChain {
    db: Arc<SqliteDb>,
    lineage: Option<Arc<crate::lineage::LineageQuery>>,
}

impl WitnessChain {
    /// Create a new WitnessChain backed by the given SQLite database.
    pub fn new(db: Arc<SqliteDb>) -> Self {
        Self {
            db,
            lineage: None,
        }
    }

    /// Attach a lineage query for automatic lineage recording on witness entries.
    pub fn with_lineage(mut self, lineage: Arc<crate::lineage::LineageQuery>) -> Self {
        self.lineage = Some(lineage);
        self
    }

    /// Create the `witness_log` table and indexes if they do not exist.
    pub async fn init_schema(&self) -> Result<()> {
        self.db
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS witness_log (
                    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id   TEXT NOT NULL,
                    operation    TEXT NOT NULL,
                    action_hash  BLOB NOT NULL,
                    prev_hash    BLOB NOT NULL,
                    entry_hash   BLOB NOT NULL,
                    timestamp    TEXT NOT NULL,
                    witness_type INTEGER NOT NULL DEFAULT 1,
                    agent_id     TEXT,
                    metadata     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_witness_pattern_id ON witness_log(pattern_id);
                CREATE INDEX IF NOT EXISTS idx_witness_timestamp ON witness_log(timestamp);",
            )
            .await
    }

    /// Append a new witness entry, automatically chaining to the previous entry.
    ///
    /// Returns the sequence number of the newly appended entry.
    pub async fn append(
        &self,
        pattern_id: &str,
        operation: WitnessOperation,
        witness_type: WitnessType,
        agent_id: Option<&str>,
        metadata: Option<&str>,
    ) -> Result<u64> {
        let action_hash = compute_action_hash(pattern_id, &operation, metadata);
        let now = Utc::now();
        let ts = now.to_rfc3339();

        // Get the previous entry's hash (or zeros for genesis).
        let prev_hash = self.last_entry_hash().await?;

        let entry_hash = compute_entry_hash(&prev_hash, &action_hash, &ts, witness_type.as_u8());

        let op_str = operation.as_str().to_string();
        let wt = witness_type.as_u8() as i64;
        let pid = pattern_id.to_string();
        let aid = agent_id.map(|s| s.to_string());
        let md = metadata.map(|s| s.to_string());
        let action_vec = action_hash.to_vec();
        let prev_vec = prev_hash.to_vec();
        let entry_vec = entry_hash.to_vec();
        let ts_clone = ts.clone();

        self.db
            .execute(
                "INSERT INTO witness_log (pattern_id, operation, action_hash, prev_hash, entry_hash, timestamp, witness_type, agent_id, metadata) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                &[
                    &pid as &dyn rusqlite::ToSql,
                    &op_str as &dyn rusqlite::ToSql,
                    &action_vec as &dyn rusqlite::ToSql,
                    &prev_vec as &dyn rusqlite::ToSql,
                    &entry_vec as &dyn rusqlite::ToSql,
                    &ts_clone as &dyn rusqlite::ToSql,
                    &wt as &dyn rusqlite::ToSql,
                    &aid as &dyn rusqlite::ToSql,
                    &md as &dyn rusqlite::ToSql,
                ],
            )
            .await?;

        // Return the seq of the just-inserted row.
        let seq = self
            .db
            .query_one(
                "SELECT seq FROM witness_log ORDER BY seq DESC LIMIT 1",
                &[],
                |row| row.get::<_, i64>(0),
            )
            .await?
            .unwrap_or(0) as u64;

        Ok(seq)
    }

    /// Verify the integrity of the full witness chain.
    pub async fn verify(&self) -> Result<WitnessVerification> {
        let entries = self.all_entries().await?;
        verify_entries(&entries)
    }

    /// Verify only the entries belonging to a specific pattern.
    ///
    /// Because entries for different patterns share the same global chain,
    /// this method re-verifies the full chain but only reports the first break
    /// that affects an entry for the given pattern.
    pub async fn verify_pattern(&self, pattern_id: &str) -> Result<WitnessVerification> {
        let all = self.all_entries().await?;
        let full = verify_entries(&all)?;

        if full.valid {
            // Count how many entries belong to this pattern.
            let pattern_count = all.iter().filter(|e| e.pattern_id == pattern_id).count() as u64;
            return Ok(WitnessVerification {
                valid: true,
                entries_checked: pattern_count,
                first_broken_seq: None,
                chain_length: pattern_count,
            });
        }

        // Chain is broken somewhere. Report the first break that touches this pattern.
        let pattern_entries: Vec<&WitnessEntry> =
            all.iter().filter(|e| e.pattern_id == pattern_id).collect();
        let pattern_count = pattern_entries.len() as u64;

        // Check if the broken seq belongs to one of this pattern's entries.
        let broken_for_pattern = full.first_broken_seq.and_then(|broken_seq| {
            if pattern_entries.iter().any(|e| e.seq == broken_seq) {
                Some(broken_seq)
            } else {
                None
            }
        });

        Ok(WitnessVerification {
            valid: broken_for_pattern.is_none(),
            entries_checked: pattern_count,
            first_broken_seq: broken_for_pattern,
            chain_length: pattern_count,
        })
    }

    /// Get the latest witness entry for a given pattern.
    pub async fn latest(&self, pattern_id: &str) -> Result<Option<WitnessEntry>> {
        let pid = pattern_id.to_string();
        self.db
            .query_one(
                "SELECT seq, pattern_id, operation, action_hash, prev_hash, entry_hash, \
                        timestamp, witness_type, agent_id, metadata \
                 FROM witness_log WHERE pattern_id = ? ORDER BY seq DESC LIMIT 1",
                &[&pid as &dyn rusqlite::ToSql],
                row_to_entry,
            )
            .await
    }

    /// Get the full audit trail for a pattern (all witness entries, oldest first).
    pub async fn audit_trail(&self, pattern_id: &str) -> Result<Vec<WitnessEntry>> {
        let pid = pattern_id.to_string();
        self.db
            .query(
                "SELECT seq, pattern_id, operation, action_hash, prev_hash, entry_hash, \
                        timestamp, witness_type, agent_id, metadata \
                 FROM witness_log WHERE pattern_id = ? ORDER BY seq ASC",
                &[&pid as &dyn rusqlite::ToSql],
                row_to_entry,
            )
            .await
    }

    /// Return the total number of witness entries.
    pub async fn count(&self) -> Result<u64> {
        let c = self
            .db
            .query_one("SELECT COUNT(*) FROM witness_log", &[], |row| {
                row.get::<_, i64>(0)
            })
            .await?
            .unwrap_or(0);
        Ok(c as u64)
    }

    // -- internal helpers ---------------------------------------------------

    /// Fetch the entry_hash of the last entry in the chain, or zeros if empty.
    async fn last_entry_hash(&self) -> Result<[u8; 32]> {
        let hash_opt = self
            .db
            .query_one(
                "SELECT entry_hash FROM witness_log ORDER BY seq DESC LIMIT 1",
                &[],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .await?;

        match hash_opt {
            Some(v) if v.len() == 32 => {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&v);
                Ok(arr)
            }
            _ => Ok([0u8; 32]),
        }
    }

    /// Load every entry in sequence order.
    async fn all_entries(&self) -> Result<Vec<WitnessEntry>> {
        self.db
            .query(
                "SELECT seq, pattern_id, operation, action_hash, prev_hash, entry_hash, \
                        timestamp, witness_type, agent_id, metadata \
                 FROM witness_log ORDER BY seq ASC",
                &[],
                row_to_entry,
            )
            .await
    }
}

// ---------------------------------------------------------------------------
// Row mapping
// ---------------------------------------------------------------------------

fn row_to_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<WitnessEntry> {
    let seq: i64 = row.get(0)?;
    let pattern_id: String = row.get(1)?;
    let operation_str: String = row.get(2)?;
    let action_hash_vec: Vec<u8> = row.get(3)?;
    let prev_hash_vec: Vec<u8> = row.get(4)?;
    let entry_hash_vec: Vec<u8> = row.get(5)?;
    let timestamp_str: String = row.get(6)?;
    let witness_type_int: i64 = row.get(7)?;
    let agent_id: Option<String> = row.get(8)?;
    let metadata: Option<String> = row.get(9)?;

    let mut action_hash = [0u8; 32];
    if action_hash_vec.len() == 32 {
        action_hash.copy_from_slice(&action_hash_vec);
    }
    let mut prev_hash = [0u8; 32];
    if prev_hash_vec.len() == 32 {
        prev_hash.copy_from_slice(&prev_hash_vec);
    }
    let mut entry_hash = [0u8; 32];
    if entry_hash_vec.len() == 32 {
        entry_hash.copy_from_slice(&entry_hash_vec);
    }

    let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());

    Ok(WitnessEntry {
        seq: seq as u64,
        pattern_id,
        operation: WitnessOperation::from(operation_str.as_str()),
        action_hash,
        prev_hash,
        entry_hash,
        timestamp,
        witness_type: WitnessType::from_u8(witness_type_int as u8),
        agent_id,
        metadata,
    })
}

// ---------------------------------------------------------------------------
// Verification logic (pure function, no DB access)
// ---------------------------------------------------------------------------

fn verify_entries(entries: &[WitnessEntry]) -> Result<WitnessVerification> {
    if entries.is_empty() {
        return Ok(WitnessVerification {
            valid: true,
            entries_checked: 0,
            first_broken_seq: None,
            chain_length: 0,
        });
    }

    let total = entries.len() as u64;

    // Genesis entry must have all-zero prev_hash.
    if entries[0].prev_hash != [0u8; 32] {
        return Ok(WitnessVerification {
            valid: false,
            entries_checked: 1,
            first_broken_seq: Some(entries[0].seq),
            chain_length: total,
        });
    }

    // Verify that each entry's action_hash is correct.
    for entry in entries {
        let expected_action =
            compute_action_hash(&entry.pattern_id, &entry.operation, entry.metadata.as_deref());
        if expected_action != entry.action_hash {
            return Ok(WitnessVerification {
                valid: false,
                entries_checked: entry.seq,
                first_broken_seq: Some(entry.seq),
                chain_length: total,
            });
        }
    }

    // Verify that each entry's entry_hash is correct and chains to predecessor.
    for (i, entry) in entries.iter().enumerate() {
        let expected_prev = if i == 0 {
            [0u8; 32]
        } else {
            entries[i - 1].entry_hash
        };

        if entry.prev_hash != expected_prev {
            return Ok(WitnessVerification {
                valid: false,
                entries_checked: (i + 1) as u64,
                first_broken_seq: Some(entry.seq),
                chain_length: total,
            });
        }

        let ts = entry.timestamp.to_rfc3339();
        let expected_entry = compute_entry_hash(
            &entry.prev_hash,
            &entry.action_hash,
            &ts,
            entry.witness_type.as_u8(),
        );
        if expected_entry != entry.entry_hash {
            return Ok(WitnessVerification {
                valid: false,
                entries_checked: (i + 1) as u64,
                first_broken_seq: Some(entry.seq),
                chain_length: total,
            });
        }
    }

    Ok(WitnessVerification {
        valid: true,
        entries_checked: total,
        first_broken_seq: None,
        chain_length: total,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Arc<SqliteDb> {
        let db = SqliteDb::open_in_memory().unwrap();
        Arc::new(db)
    }

    async fn setup_chain() -> WitnessChain {
        let db = test_db();
        let chain = WitnessChain::new(db);
        chain.init_schema().await.unwrap();
        chain
    }

    // -- Schema tests -------------------------------------------------------

    #[tokio::test]
    async fn test_init_schema_creates_table() {
        let db = test_db();
        let chain = WitnessChain::new(db.clone());
        chain.init_schema().await.unwrap();

        let exists = db.table_exists("witness_log").await.unwrap();
        assert!(exists, "witness_log table should exist after init_schema");
    }

    #[tokio::test]
    async fn test_init_schema_idempotent() {
        let db = test_db();
        let chain = WitnessChain::new(db.clone());
        chain.init_schema().await.unwrap();
        chain.init_schema().await.unwrap(); // second call should not fail

        let exists = db.table_exists("witness_log").await.unwrap();
        assert!(exists);
    }

    // -- Hash computation tests ---------------------------------------------

    #[test]
    fn test_action_hash_deterministic() {
        let h1 = compute_action_hash("p1", &WitnessOperation::Store, Some("meta"));
        let h2 = compute_action_hash("p1", &WitnessOperation::Store, Some("meta"));
        assert_eq!(h1, h2, "Same inputs must produce the same action hash");
    }

    #[test]
    fn test_action_hash_differs_for_different_pattern() {
        let h1 = compute_action_hash("p1", &WitnessOperation::Store, None);
        let h2 = compute_action_hash("p2", &WitnessOperation::Store, None);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_action_hash_differs_for_different_operation() {
        let h1 = compute_action_hash("p1", &WitnessOperation::Store, None);
        let h2 = compute_action_hash("p1", &WitnessOperation::Update, None);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_action_hash_differs_for_different_metadata() {
        let h1 = compute_action_hash("p1", &WitnessOperation::Store, Some("a"));
        let h2 = compute_action_hash("p1", &WitnessOperation::Store, Some("b"));
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_action_hash_none_vs_some_metadata() {
        let h1 = compute_action_hash("p1", &WitnessOperation::Store, None);
        let h2 = compute_action_hash("p1", &WitnessOperation::Store, Some("data"));
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_entry_hash_chains_correctly() {
        let prev = [0u8; 32];
        let action = compute_action_hash("p1", &WitnessOperation::Store, None);
        let ts = "2026-01-01T00:00:00+00:00";

        let h1 = compute_entry_hash(&prev, &action, ts, WitnessType::Provenance.as_u8());

        // Changing prev_hash changes entry_hash
        let mut different_prev = [0u8; 32];
        different_prev[0] = 1;
        let h2 = compute_entry_hash(&different_prev, &action, ts, WitnessType::Provenance.as_u8());
        assert_ne!(h1, h2, "Different prev_hash should produce different entry_hash");
    }

    #[test]
    fn test_entry_hash_differs_for_witness_type() {
        let prev = [0u8; 32];
        let action = compute_action_hash("p1", &WitnessOperation::Store, None);
        let ts = "2026-01-01T00:00:00+00:00";

        let h1 = compute_entry_hash(&prev, &action, ts, WitnessType::Provenance.as_u8());
        let h2 = compute_entry_hash(&prev, &action, ts, WitnessType::Computation.as_u8());
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_entry_hash_differs_for_timestamp() {
        let prev = [0u8; 32];
        let action = compute_action_hash("p1", &WitnessOperation::Store, None);

        let h1 = compute_entry_hash(&prev, &action, "2026-01-01T00:00:00+00:00", 1);
        let h2 = compute_entry_hash(&prev, &action, "2026-01-02T00:00:00+00:00", 1);
        assert_ne!(h1, h2);
    }

    // -- Append tests -------------------------------------------------------

    #[tokio::test]
    async fn test_append_genesis_entry_has_zero_prev_hash() {
        let chain = setup_chain().await;

        let seq = chain
            .append("p1", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();

        assert_eq!(seq, 1);

        let entry = chain.latest("p1").await.unwrap().unwrap();
        assert_eq!(entry.prev_hash, [0u8; 32], "Genesis entry must have zero prev_hash");
    }

    #[tokio::test]
    async fn test_append_subsequent_entries_chain() {
        let chain = setup_chain().await;

        chain
            .append("p1", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        let first = chain.latest("p1").await.unwrap().unwrap();

        chain
            .append("p1", WitnessOperation::Update, WitnessType::Computation, None, None)
            .await
            .unwrap();

        // The second entry in the global chain
        let entries = chain.all_entries().await.unwrap();
        let second = &entries[1];

        assert_eq!(
            second.prev_hash, first.entry_hash,
            "Second entry's prev_hash must equal first entry's entry_hash"
        );
    }

    #[tokio::test]
    async fn test_append_tracks_operations() {
        let chain = setup_chain().await;

        chain
            .append("p1", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        chain
            .append("p1", WitnessOperation::Update, WitnessType::Computation, None, None)
            .await
            .unwrap();
        chain
            .append("p1", WitnessOperation::Delete, WitnessType::Provenance, None, None)
            .await
            .unwrap();

        let trail = chain.audit_trail("p1").await.unwrap();
        assert_eq!(trail.len(), 3);
        assert_eq!(trail[0].operation, WitnessOperation::Store);
        assert_eq!(trail[1].operation, WitnessOperation::Update);
        assert_eq!(trail[2].operation, WitnessOperation::Delete);
    }

    #[tokio::test]
    async fn test_append_with_agent_and_metadata() {
        let chain = setup_chain().await;

        chain
            .append(
                "p1",
                WitnessOperation::Store,
                WitnessType::Provenance,
                Some("agent-42"),
                Some(r#"{"field":"value"}"#),
            )
            .await
            .unwrap();

        let entry = chain.latest("p1").await.unwrap().unwrap();
        assert_eq!(entry.agent_id.as_deref(), Some("agent-42"));
        assert_eq!(entry.metadata.as_deref(), Some(r#"{"field":"value"}"#));
    }

    #[tokio::test]
    async fn test_append_returns_incrementing_seq() {
        let chain = setup_chain().await;

        let s1 = chain
            .append("p1", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        let s2 = chain
            .append("p2", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        let s3 = chain
            .append("p1", WitnessOperation::Update, WitnessType::Computation, None, None)
            .await
            .unwrap();

        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(s3, 3);
    }

    // -- Verify tests -------------------------------------------------------

    #[tokio::test]
    async fn test_verify_valid_chain() {
        let chain = setup_chain().await;

        chain
            .append("p1", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        chain
            .append("p1", WitnessOperation::Update, WitnessType::Computation, None, Some("data"))
            .await
            .unwrap();
        chain
            .append("p2", WitnessOperation::Store, WitnessType::Provenance, Some("bot"), None)
            .await
            .unwrap();

        let v = chain.verify().await.unwrap();
        assert!(v.valid, "Untampered chain must verify as valid");
        assert_eq!(v.entries_checked, 3);
        assert!(v.first_broken_seq.is_none());
        assert_eq!(v.chain_length, 3);
    }

    #[tokio::test]
    async fn test_verify_empty_chain_is_valid() {
        let chain = setup_chain().await;
        let v = chain.verify().await.unwrap();
        assert!(v.valid);
        assert_eq!(v.entries_checked, 0);
        assert_eq!(v.chain_length, 0);
    }

    #[tokio::test]
    async fn test_verify_tampered_action_hash_detected() {
        let chain = setup_chain().await;

        chain
            .append("p1", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        chain
            .append("p1", WitnessOperation::Update, WitnessType::Computation, None, None)
            .await
            .unwrap();

        // Tamper with the first entry's action_hash.
        chain
            .db
            .execute(
                "UPDATE witness_log SET action_hash = ? WHERE seq = 1",
                &[&vec![0xFFu8; 32] as &dyn rusqlite::ToSql],
            )
            .await
            .unwrap();

        let v = chain.verify().await.unwrap();
        assert!(!v.valid, "Tampered chain must fail verification");
        assert_eq!(v.first_broken_seq, Some(1));
    }

    #[tokio::test]
    async fn test_verify_tampered_prev_hash_detected() {
        let chain = setup_chain().await;

        chain
            .append("p1", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        chain
            .append("p1", WitnessOperation::Update, WitnessType::Computation, None, None)
            .await
            .unwrap();

        // Tamper with the second entry's prev_hash.
        chain
            .db
            .execute(
                "UPDATE witness_log SET prev_hash = ? WHERE seq = 2",
                &[&vec![0xAAu8; 32] as &dyn rusqlite::ToSql],
            )
            .await
            .unwrap();

        let v = chain.verify().await.unwrap();
        assert!(!v.valid);
        assert_eq!(v.first_broken_seq, Some(2));
    }

    #[tokio::test]
    async fn test_verify_single_entry_valid() {
        let chain = setup_chain().await;

        chain
            .append("p1", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();

        let v = chain.verify().await.unwrap();
        assert!(v.valid);
        assert_eq!(v.entries_checked, 1);
        assert_eq!(v.chain_length, 1);
    }

    // -- Pattern-specific tests ---------------------------------------------

    #[tokio::test]
    async fn test_verify_pattern_only_checks_that_pattern() {
        let chain = setup_chain().await;

        chain
            .append("p1", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        chain
            .append("p2", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        chain
            .append("p1", WitnessOperation::Update, WitnessType::Computation, None, None)
            .await
            .unwrap();

        let v = chain.verify_pattern("p1").await.unwrap();
        assert!(v.valid);
        assert_eq!(v.entries_checked, 2, "p1 has 2 entries");
        assert_eq!(v.chain_length, 2);
    }

    #[tokio::test]
    async fn test_verify_pattern_nonexistent_is_valid() {
        let chain = setup_chain().await;

        chain
            .append("p1", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();

        let v = chain.verify_pattern("does-not-exist").await.unwrap();
        assert!(v.valid);
        assert_eq!(v.entries_checked, 0);
    }

    #[tokio::test]
    async fn test_audit_trail_returns_correct_entries() {
        let chain = setup_chain().await;

        chain
            .append("p1", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        chain
            .append("p2", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        chain
            .append("p1", WitnessOperation::Update, WitnessType::Computation, None, None)
            .await
            .unwrap();

        let trail = chain.audit_trail("p1").await.unwrap();
        assert_eq!(trail.len(), 2);
        assert!(trail.iter().all(|e| e.pattern_id == "p1"));

        let trail_p2 = chain.audit_trail("p2").await.unwrap();
        assert_eq!(trail_p2.len(), 1);
        assert_eq!(trail_p2[0].pattern_id, "p2");
    }

    #[tokio::test]
    async fn test_latest_returns_most_recent() {
        let chain = setup_chain().await;

        chain
            .append("p1", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        chain
            .append("p1", WitnessOperation::Update, WitnessType::Computation, None, None)
            .await
            .unwrap();
        chain
            .append("p1", WitnessOperation::Merge, WitnessType::Consolidation, None, None)
            .await
            .unwrap();

        let latest = chain.latest("p1").await.unwrap().unwrap();
        assert_eq!(latest.operation, WitnessOperation::Merge);
        assert_eq!(latest.witness_type, WitnessType::Consolidation);
    }

    #[tokio::test]
    async fn test_latest_nonexistent_returns_none() {
        let chain = setup_chain().await;
        let result = chain.latest("nope").await.unwrap();
        assert!(result.is_none());
    }

    // -- Edge cases ---------------------------------------------------------

    #[tokio::test]
    async fn test_concurrent_patterns_interleave_correctly() {
        let chain = setup_chain().await;

        // Interleave entries for different patterns in the global chain.
        chain
            .append("a", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        chain
            .append("b", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        chain
            .append("a", WitnessOperation::Update, WitnessType::Computation, None, None)
            .await
            .unwrap();
        chain
            .append("b", WitnessOperation::Update, WitnessType::Computation, None, None)
            .await
            .unwrap();

        // Full chain must still be valid.
        let v = chain.verify().await.unwrap();
        assert!(v.valid);
        assert_eq!(v.entries_checked, 4);

        // Per-pattern verification also valid.
        let va = chain.verify_pattern("a").await.unwrap();
        assert!(va.valid);
        assert_eq!(va.chain_length, 2);

        let vb = chain.verify_pattern("b").await.unwrap();
        assert!(vb.valid);
        assert_eq!(vb.chain_length, 2);
    }

    #[tokio::test]
    async fn test_count() {
        let chain = setup_chain().await;
        assert_eq!(chain.count().await.unwrap(), 0);

        chain
            .append("p1", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        assert_eq!(chain.count().await.unwrap(), 1);

        chain
            .append("p2", WitnessOperation::Store, WitnessType::Provenance, None, None)
            .await
            .unwrap();
        assert_eq!(chain.count().await.unwrap(), 2);
    }

    // -- WitnessOperation tests ---------------------------------------------

    #[test]
    fn test_witness_operation_as_str() {
        assert_eq!(WitnessOperation::Store.as_str(), "store");
        assert_eq!(WitnessOperation::Update.as_str(), "update");
        assert_eq!(WitnessOperation::Delete.as_str(), "delete");
        assert_eq!(WitnessOperation::Merge.as_str(), "merge");
    }

    #[test]
    fn test_witness_operation_from_str() {
        assert_eq!(WitnessOperation::from("store"), WitnessOperation::Store);
        assert_eq!(WitnessOperation::from("update"), WitnessOperation::Update);
        assert_eq!(WitnessOperation::from("delete"), WitnessOperation::Delete);
        assert_eq!(WitnessOperation::from("merge"), WitnessOperation::Merge);
        assert_eq!(WitnessOperation::from("STORE"), WitnessOperation::Store);
        assert_eq!(WitnessOperation::from("unknown"), WitnessOperation::Store);
    }

    #[test]
    fn test_witness_operation_display() {
        assert_eq!(format!("{}", WitnessOperation::Store), "store");
        assert_eq!(format!("{}", WitnessOperation::Delete), "delete");
    }

    #[test]
    fn test_witness_operation_serde_roundtrip() {
        let ops = vec![
            WitnessOperation::Store,
            WitnessOperation::Update,
            WitnessOperation::Delete,
            WitnessOperation::Merge,
        ];
        for op in ops {
            let json = serde_json::to_string(&op).unwrap();
            let back: WitnessOperation = serde_json::from_str(&json).unwrap();
            assert_eq!(op, back);
        }
    }

    // -- WitnessType tests --------------------------------------------------

    #[test]
    fn test_witness_type_as_u8() {
        assert_eq!(WitnessType::Provenance.as_u8(), 1);
        assert_eq!(WitnessType::Computation.as_u8(), 2);
        assert_eq!(WitnessType::Consolidation.as_u8(), 3);
    }

    #[test]
    fn test_witness_type_from_u8() {
        assert_eq!(WitnessType::from_u8(1), WitnessType::Provenance);
        assert_eq!(WitnessType::from_u8(2), WitnessType::Computation);
        assert_eq!(WitnessType::from_u8(3), WitnessType::Consolidation);
        assert_eq!(WitnessType::from_u8(99), WitnessType::Provenance); // fallback
    }

    #[test]
    fn test_witness_type_display() {
        assert_eq!(format!("{}", WitnessType::Provenance), "provenance");
        assert_eq!(format!("{}", WitnessType::Computation), "computation");
        assert_eq!(format!("{}", WitnessType::Consolidation), "consolidation");
    }

    #[test]
    fn test_witness_type_serde_roundtrip() {
        let types = vec![
            WitnessType::Provenance,
            WitnessType::Computation,
            WitnessType::Consolidation,
        ];
        for wt in types {
            let json = serde_json::to_string(&wt).unwrap();
            let back: WitnessType = serde_json::from_str(&json).unwrap();
            assert_eq!(wt, back);
        }
    }

    // -- Verification pure function tests -----------------------------------

    #[test]
    fn test_verify_entries_empty() {
        let v = verify_entries(&[]).unwrap();
        assert!(v.valid);
        assert_eq!(v.entries_checked, 0);
    }
}
