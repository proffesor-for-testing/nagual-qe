//! Knowledge Epochs for the Knowledge Operating System (KOS P5).
//!
//! Epochs freeze validated knowledge into named, immutable snapshots.
//! They support Copy-on-Write (COW) branching for experimentation,
//! rollback to restore previous states, and diff to compare epoch contents.
//!
//! Epochs depend on the lineage feature (P0) to track pattern derivation
//! across snapshots.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::db::SqliteDb;
use crate::error::{NagualError, Result};

/// A frozen snapshot of knowledge at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEpoch {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub pattern_count: u64,
    pub domain_count: u32,
    pub avg_reward: f64,
    pub parent_epoch: Option<String>,
    pub frozen: bool,
    pub metadata: Option<serde_json::Value>,
}

/// Result of comparing two epochs.
#[derive(Debug, Clone)]
pub struct EpochDiff {
    pub epoch_a: String,
    pub epoch_b: String,
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub common: usize,
}

/// Result of a (non-destructive) rollback analysis.
///
/// Rollback does NOT modify the database. It reports what *would* change
/// if the current `reasoning_patterns` were rolled back to match the epoch.
#[derive(Debug, Clone)]
pub struct RollbackResult {
    pub epoch_name: String,
    /// Patterns that are in the epoch but NOT currently in reasoning_patterns.
    pub patterns_restored: u64,
    /// Patterns that are currently in reasoning_patterns but NOT in the epoch.
    pub patterns_removed: u64,
}

/// Manages knowledge epochs -- snapshots, branching, rollback.
pub struct EpochManager {
    db: Arc<SqliteDb>,
    lineage: Option<Arc<crate::lineage::LineageQuery>>,
}

impl EpochManager {
    /// Create a new EpochManager backed by the given database.
    pub fn new(db: Arc<SqliteDb>) -> Self {
        Self {
            db,
            lineage: None,
        }
    }

    /// Attach a lineage query for recording epoch-boundary lineage snapshots.
    pub fn with_lineage(mut self, lineage: Arc<crate::lineage::LineageQuery>) -> Self {
        self.lineage = Some(lineage);
        self
    }

    /// Initialize the epoch schema tables. Idempotent.
    pub async fn init_schema(&self) -> Result<()> {
        self.db
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS knowledge_epochs (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL UNIQUE,
                    description TEXT,
                    created_at  TEXT NOT NULL,
                    pattern_count INTEGER NOT NULL,
                    domain_count  INTEGER NOT NULL,
                    avg_reward    REAL NOT NULL,
                    parent_epoch  TEXT,
                    frozen        INTEGER DEFAULT 1,
                    metadata      TEXT
                );

                CREATE TABLE IF NOT EXISTS epoch_membership (
                    epoch_id    TEXT NOT NULL,
                    pattern_id  TEXT NOT NULL,
                    PRIMARY KEY (epoch_id, pattern_id)
                );

                CREATE INDEX IF NOT EXISTS idx_epoch_membership_epoch
                    ON epoch_membership(epoch_id);
                CREATE INDEX IF NOT EXISTS idx_epoch_membership_pattern
                    ON epoch_membership(pattern_id);",
            )
            .await?;

        Ok(())
    }

    /// Create a new epoch that snapshots all current reasoning_patterns.
    ///
    /// The epoch is frozen by default. Errors if the name already exists.
    pub async fn create_epoch(
        &self,
        name: &str,
        description: Option<&str>,
    ) -> Result<KnowledgeEpoch> {
        // Check for duplicate name
        let n = name.to_string();
        let existing = self
            .db
            .query_one(
                "SELECT id FROM knowledge_epochs WHERE name = ?",
                &[&n as &dyn rusqlite::ToSql],
                |row| row.get::<_, String>(0),
            )
            .await?;

        if existing.is_some() {
            return Err(NagualError::Internal {
                message: format!("Epoch with name '{}' already exists", name),
            });
        }

        let epoch_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let now_str = now.to_rfc3339();
        let desc = description.map(|s| s.to_string());

        // Query stats from reasoning_patterns
        let eid = epoch_id.clone();
        let stats = self
            .db
            .with_connection(move |conn| {
                let pattern_count: i64 = conn
                    .query_row(
                        "SELECT COUNT(*) FROM reasoning_patterns",
                        [],
                        |row| row.get(0),
                    )
                    .map_err(crate::error::DatabaseError::from)?;

                let domain_count: i64 = conn
                    .query_row(
                        "SELECT COUNT(DISTINCT domain) FROM reasoning_patterns WHERE domain != ''",
                        [],
                        |row| row.get(0),
                    )
                    .map_err(crate::error::DatabaseError::from)?;

                let avg_reward: f64 = conn
                    .query_row(
                        "SELECT COALESCE(AVG(reward), 0.0) FROM reasoning_patterns",
                        [],
                        |row| row.get(0),
                    )
                    .map_err(crate::error::DatabaseError::from)?;

                // Insert all pattern IDs into epoch_membership
                conn.execute(
                    "INSERT INTO epoch_membership (epoch_id, pattern_id)
                     SELECT ?, id FROM reasoning_patterns",
                    rusqlite::params![eid],
                )
                .map_err(crate::error::DatabaseError::from)?;

                Ok((pattern_count as u64, domain_count as u32, avg_reward))
            })
            .await?;

        let (pattern_count, domain_count, avg_reward) = stats;

        // Insert the epoch record
        self.db
            .execute(
                "INSERT INTO knowledge_epochs
                    (id, name, description, created_at, pattern_count, domain_count, avg_reward, parent_epoch, frozen, metadata)
                 VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 1, NULL)",
                &[
                    &epoch_id as &dyn rusqlite::ToSql,
                    &name as &dyn rusqlite::ToSql,
                    &desc as &dyn rusqlite::ToSql,
                    &now_str as &dyn rusqlite::ToSql,
                    &(pattern_count as i64) as &dyn rusqlite::ToSql,
                    &(domain_count as i32) as &dyn rusqlite::ToSql,
                    &avg_reward as &dyn rusqlite::ToSql,
                ],
            )
            .await?;

        Ok(KnowledgeEpoch {
            id: epoch_id,
            name: name.to_string(),
            description: description.map(|s| s.to_string()),
            created_at: now,
            pattern_count,
            domain_count,
            avg_reward,
            parent_epoch: None,
            frozen: true,
            metadata: None,
        })
    }

    /// COW branch: copy parent's membership to a new, non-frozen epoch.
    ///
    /// Errors if parent doesn't exist or branch_name already exists.
    pub async fn branch(
        &self,
        parent_name: &str,
        branch_name: &str,
    ) -> Result<KnowledgeEpoch> {
        let parent = self.get(parent_name).await?;
        let parent = parent.ok_or_else(|| NagualError::Internal {
            message: format!("Parent epoch '{}' not found", parent_name),
        })?;

        // Check branch_name doesn't already exist
        let bn = branch_name.to_string();
        let existing = self
            .db
            .query_one(
                "SELECT id FROM knowledge_epochs WHERE name = ?",
                &[&bn as &dyn rusqlite::ToSql],
                |row| row.get::<_, String>(0),
            )
            .await?;

        if existing.is_some() {
            return Err(NagualError::Internal {
                message: format!("Epoch with name '{}' already exists", branch_name),
            });
        }

        let branch_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let now_str = now.to_rfc3339();

        // Copy membership from parent
        let bid = branch_id.clone();
        let pid = parent.id.clone();
        self.db
            .with_connection(move |conn| {
                conn.execute(
                    "INSERT INTO epoch_membership (epoch_id, pattern_id)
                     SELECT ?, pattern_id FROM epoch_membership WHERE epoch_id = ?",
                    rusqlite::params![bid, pid],
                )
                .map_err(crate::error::DatabaseError::from)?;
                Ok(())
            })
            .await?;

        // Insert the branch epoch record (frozen = false)
        self.db
            .execute(
                "INSERT INTO knowledge_epochs
                    (id, name, description, created_at, pattern_count, domain_count, avg_reward, parent_epoch, frozen, metadata)
                 VALUES (?, ?, NULL, ?, ?, ?, ?, ?, 0, NULL)",
                &[
                    &branch_id as &dyn rusqlite::ToSql,
                    &branch_name as &dyn rusqlite::ToSql,
                    &now_str as &dyn rusqlite::ToSql,
                    &(parent.pattern_count as i64) as &dyn rusqlite::ToSql,
                    &(parent.domain_count as i32) as &dyn rusqlite::ToSql,
                    &parent.avg_reward as &dyn rusqlite::ToSql,
                    &parent.id as &dyn rusqlite::ToSql,
                ],
            )
            .await?;

        Ok(KnowledgeEpoch {
            id: branch_id,
            name: branch_name.to_string(),
            description: None,
            created_at: now,
            pattern_count: parent.pattern_count,
            domain_count: parent.domain_count,
            avg_reward: parent.avg_reward,
            parent_epoch: Some(parent.id),
            frozen: false,
            metadata: None,
        })
    }

    /// Non-destructive rollback analysis.
    ///
    /// Compares the epoch's membership against current `reasoning_patterns`
    /// and reports what would change, without actually modifying any data.
    ///
    /// - `patterns_restored`: IDs in epoch membership but missing from reasoning_patterns
    /// - `patterns_removed`: IDs in reasoning_patterns but not in epoch membership
    pub async fn rollback(&self, epoch_name: &str) -> Result<RollbackResult> {
        let epoch = self.get(epoch_name).await?;
        let epoch = epoch.ok_or_else(|| NagualError::Internal {
            message: format!("Epoch '{}' not found", epoch_name),
        })?;

        let eid = epoch.id.clone();
        let eid2 = epoch.id.clone();

        // Patterns in the epoch but not currently in reasoning_patterns
        let restored: Vec<String> = self
            .db
            .query(
                "SELECT pattern_id FROM epoch_membership WHERE epoch_id = ?
                 EXCEPT
                 SELECT id FROM reasoning_patterns",
                &[&eid as &dyn rusqlite::ToSql],
                |row| row.get(0),
            )
            .await?;

        // Patterns currently in reasoning_patterns but not in the epoch
        let removed: Vec<String> = self
            .db
            .query(
                "SELECT id FROM reasoning_patterns
                 EXCEPT
                 SELECT pattern_id FROM epoch_membership WHERE epoch_id = ?",
                &[&eid2 as &dyn rusqlite::ToSql],
                |row| row.get(0),
            )
            .await?;

        Ok(RollbackResult {
            epoch_name: epoch_name.to_string(),
            patterns_restored: restored.len() as u64,
            patterns_removed: removed.len() as u64,
        })
    }

    /// Compute the diff between two named epochs.
    ///
    /// Returns which pattern IDs were added, removed, and how many are common.
    pub async fn diff(
        &self,
        epoch_a_name: &str,
        epoch_b_name: &str,
    ) -> Result<EpochDiff> {
        let a = self.get(epoch_a_name).await?;
        let a = a.ok_or_else(|| NagualError::Internal {
            message: format!("Epoch '{}' not found", epoch_a_name),
        })?;

        let b = self.get(epoch_b_name).await?;
        let b = b.ok_or_else(|| NagualError::Internal {
            message: format!("Epoch '{}' not found", epoch_b_name),
        })?;

        let aid = a.id.clone();
        let bid = b.id.clone();

        // Added: in B but not A
        let aid2 = aid.clone();
        let bid2 = bid.clone();
        let added: Vec<String> = self
            .db
            .query(
                "SELECT pattern_id FROM epoch_membership WHERE epoch_id = ?
                 EXCEPT
                 SELECT pattern_id FROM epoch_membership WHERE epoch_id = ?",
                &[
                    &bid2 as &dyn rusqlite::ToSql,
                    &aid2 as &dyn rusqlite::ToSql,
                ],
                |row| row.get(0),
            )
            .await?;

        // Removed: in A but not B
        let aid3 = aid.clone();
        let bid3 = bid.clone();
        let removed: Vec<String> = self
            .db
            .query(
                "SELECT pattern_id FROM epoch_membership WHERE epoch_id = ?
                 EXCEPT
                 SELECT pattern_id FROM epoch_membership WHERE epoch_id = ?",
                &[
                    &aid3 as &dyn rusqlite::ToSql,
                    &bid3 as &dyn rusqlite::ToSql,
                ],
                |row| row.get(0),
            )
            .await?;

        // Common: in both A and B
        let aid4 = aid.clone();
        let bid4 = bid.clone();
        let common_rows: Vec<String> = self
            .db
            .query(
                "SELECT pattern_id FROM epoch_membership WHERE epoch_id = ?
                 INTERSECT
                 SELECT pattern_id FROM epoch_membership WHERE epoch_id = ?",
                &[
                    &aid4 as &dyn rusqlite::ToSql,
                    &bid4 as &dyn rusqlite::ToSql,
                ],
                |row| row.get(0),
            )
            .await?;

        Ok(EpochDiff {
            epoch_a: epoch_a_name.to_string(),
            epoch_b: epoch_b_name.to_string(),
            added,
            removed,
            common: common_rows.len(),
        })
    }

    /// List all epochs ordered by created_at DESC.
    pub async fn list(&self) -> Result<Vec<KnowledgeEpoch>> {
        self.db
            .query(
                "SELECT id, name, description, created_at, pattern_count, domain_count,
                        avg_reward, parent_epoch, frozen, metadata
                 FROM knowledge_epochs ORDER BY created_at DESC",
                &[],
                |row| {
                    let id: String = row.get(0)?;
                    let name: String = row.get(1)?;
                    let description: Option<String> = row.get(2)?;
                    let created_at_str: String = row.get(3)?;
                    let pattern_count: i64 = row.get(4)?;
                    let domain_count: i32 = row.get(5)?;
                    let avg_reward: f64 = row.get(6)?;
                    let parent_epoch: Option<String> = row.get(7)?;
                    let frozen: bool = row.get(8)?;
                    let metadata_str: Option<String> = row.get(9)?;

                    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now());

                    let metadata = metadata_str
                        .and_then(|s| serde_json::from_str(&s).ok());

                    Ok(KnowledgeEpoch {
                        id,
                        name,
                        description,
                        created_at,
                        pattern_count: pattern_count as u64,
                        domain_count: domain_count as u32,
                        avg_reward,
                        parent_epoch,
                        frozen,
                        metadata,
                    })
                },
            )
            .await
    }

    /// Get an epoch by name.
    pub async fn get(&self, name: &str) -> Result<Option<KnowledgeEpoch>> {
        let n = name.to_string();
        self.db
            .query_one(
                "SELECT id, name, description, created_at, pattern_count, domain_count,
                        avg_reward, parent_epoch, frozen, metadata
                 FROM knowledge_epochs WHERE name = ?",
                &[&n as &dyn rusqlite::ToSql],
                |row| {
                    let id: String = row.get(0)?;
                    let name: String = row.get(1)?;
                    let description: Option<String> = row.get(2)?;
                    let created_at_str: String = row.get(3)?;
                    let pattern_count: i64 = row.get(4)?;
                    let domain_count: i32 = row.get(5)?;
                    let avg_reward: f64 = row.get(6)?;
                    let parent_epoch: Option<String> = row.get(7)?;
                    let frozen: bool = row.get(8)?;
                    let metadata_str: Option<String> = row.get(9)?;

                    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now());

                    let metadata = metadata_str
                        .and_then(|s| serde_json::from_str(&s).ok());

                    Ok(KnowledgeEpoch {
                        id,
                        name,
                        description,
                        created_at,
                        pattern_count: pattern_count as u64,
                        domain_count: domain_count as u32,
                        avg_reward,
                        parent_epoch,
                        frozen,
                        metadata,
                    })
                },
            )
            .await
    }

    /// Freeze an epoch (set frozen = true). Useful after branch modifications.
    pub async fn freeze(&self, name: &str) -> Result<()> {
        let epoch = self.get(name).await?;
        let _epoch = epoch.ok_or_else(|| NagualError::Internal {
            message: format!("Epoch '{}' not found", name),
        })?;

        let n = name.to_string();
        self.db
            .execute(
                "UPDATE knowledge_epochs SET frozen = 1 WHERE name = ?",
                &[&n as &dyn rusqlite::ToSql],
            )
            .await?;

        Ok(())
    }

    /// Delete an epoch and its membership.
    ///
    /// If `force` is false, frozen epochs cannot be deleted.
    /// If `force` is true, even frozen epochs are deleted.
    pub async fn delete(&self, name: &str, force: bool) -> Result<()> {
        let epoch = self.get(name).await?;
        let epoch = epoch.ok_or_else(|| NagualError::Internal {
            message: format!("Epoch '{}' not found", name),
        })?;

        if epoch.frozen && !force {
            return Err(NagualError::Internal {
                message: format!("Cannot delete frozen epoch '{}' (use force=true)", name),
            });
        }

        let eid = epoch.id.clone();
        let eid2 = epoch.id.clone();

        self.db
            .execute(
                "DELETE FROM epoch_membership WHERE epoch_id = ?",
                &[&eid as &dyn rusqlite::ToSql],
            )
            .await?;

        self.db
            .execute(
                "DELETE FROM knowledge_epochs WHERE id = ?",
                &[&eid2 as &dyn rusqlite::ToSql],
            )
            .await?;

        Ok(())
    }

    /// Count patterns in an epoch by name.
    pub async fn pattern_count(&self, name: &str) -> Result<u64> {
        let epoch = self.get(name).await?;
        let epoch = epoch.ok_or_else(|| NagualError::Internal {
            message: format!("Epoch '{}' not found", name),
        })?;

        let eid = epoch.id.clone();
        let count: Option<i64> = self
            .db
            .query_one(
                "SELECT COUNT(*) FROM epoch_membership WHERE epoch_id = ?",
                &[&eid as &dyn rusqlite::ToSql],
                |row| row.get(0),
            )
            .await?;

        Ok(count.unwrap_or(0) as u64)
    }

    /// Return all pattern IDs belonging to the named epoch.
    pub async fn members(&self, name: &str) -> Result<Vec<String>> {
        let epoch = self.get(name).await?;
        let epoch = epoch.ok_or_else(|| NagualError::Internal {
            message: format!("Epoch '{}' not found", name),
        })?;

        let eid = epoch.id.clone();
        self.db
            .query(
                "SELECT pattern_id FROM epoch_membership WHERE epoch_id = ? ORDER BY pattern_id",
                &[&eid as &dyn rusqlite::ToSql],
                |row| row.get(0),
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create in-memory DB with reasoning_patterns + epoch tables.
    async fn setup_test_db() -> (Arc<SqliteDb>, EpochManager) {
        let db = SqliteDb::open_in_memory().unwrap();

        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT NOT NULL,
                solution TEXT NOT NULL,
                domain TEXT DEFAULT '',
                reward REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );",
        )
        .await
        .unwrap();

        let db = Arc::new(db);
        let mgr = EpochManager::new(db.clone());
        mgr.init_schema().await.unwrap();

        (db, mgr)
    }

    /// Helper: insert a test pattern.
    async fn insert_pattern(db: &SqliteDb, id: &str, domain: &str, reward: f64) {
        let now = Utc::now().to_rfc3339();
        db.execute(
            "INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            &[
                &id as &dyn rusqlite::ToSql,
                &format!("problem-{}", id) as &dyn rusqlite::ToSql,
                &format!("solution-{}", id) as &dyn rusqlite::ToSql,
                &domain as &dyn rusqlite::ToSql,
                &reward as &dyn rusqlite::ToSql,
                &now as &dyn rusqlite::ToSql,
                &now as &dyn rusqlite::ToSql,
            ],
        )
        .await
        .unwrap();
    }

    // -----------------------------------------------------------------------
    // KnowledgeEpoch serde (2 tests)
    // -----------------------------------------------------------------------

    #[test]
    fn test_knowledge_epoch_serde_roundtrip() {
        let epoch = KnowledgeEpoch {
            id: "abc-123".to_string(),
            name: "v1.0".to_string(),
            description: Some("First stable release".to_string()),
            created_at: Utc::now(),
            pattern_count: 42,
            domain_count: 3,
            avg_reward: 0.75,
            parent_epoch: None,
            frozen: true,
            metadata: Some(serde_json::json!({"key": "value"})),
        };

        let json = serde_json::to_string(&epoch).unwrap();
        let deser: KnowledgeEpoch = serde_json::from_str(&json).unwrap();

        assert_eq!(deser.id, "abc-123");
        assert_eq!(deser.name, "v1.0");
        assert_eq!(deser.pattern_count, 42);
        assert_eq!(deser.domain_count, 3);
        assert!((deser.avg_reward - 0.75).abs() < f64::EPSILON);
        assert!(deser.frozen);
    }

    #[test]
    fn test_knowledge_epoch_serde_default_fields() {
        let epoch = KnowledgeEpoch {
            id: "id".to_string(),
            name: "minimal".to_string(),
            description: None,
            created_at: Utc::now(),
            pattern_count: 0,
            domain_count: 0,
            avg_reward: 0.0,
            parent_epoch: None,
            frozen: false,
            metadata: None,
        };

        let json = serde_json::to_string(&epoch).unwrap();
        let deser: KnowledgeEpoch = serde_json::from_str(&json).unwrap();

        assert!(deser.description.is_none());
        assert!(deser.parent_epoch.is_none());
        assert!(!deser.frozen);
        assert!(deser.metadata.is_none());
    }

    // -----------------------------------------------------------------------
    // Schema tests (2 tests)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_init_schema_creates_tables() {
        let (db, _mgr) = setup_test_db().await;
        assert!(db.table_exists("knowledge_epochs").await.unwrap());
        assert!(db.table_exists("epoch_membership").await.unwrap());
    }

    #[tokio::test]
    async fn test_init_schema_idempotent() {
        let (db, mgr) = setup_test_db().await;
        // Second call should not fail
        mgr.init_schema().await.unwrap();
        assert!(db.table_exists("knowledge_epochs").await.unwrap());
    }

    // -----------------------------------------------------------------------
    // create_epoch tests (5 tests)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_create_epoch_correct_stats() {
        let (db, mgr) = setup_test_db().await;
        insert_pattern(&db, "p1", "rust", 0.8).await;
        insert_pattern(&db, "p2", "rust", 0.6).await;
        insert_pattern(&db, "p3", "python", 0.9).await;

        let epoch = mgr.create_epoch("v1", Some("First")).await.unwrap();

        assert_eq!(epoch.name, "v1");
        assert_eq!(epoch.pattern_count, 3);
        assert_eq!(epoch.domain_count, 2); // rust, python
        let expected_avg = (0.8 + 0.6 + 0.9) / 3.0;
        assert!((epoch.avg_reward - expected_avg).abs() < 0.001);
        assert!(epoch.frozen);
        assert!(epoch.parent_epoch.is_none());
    }

    #[tokio::test]
    async fn test_create_epoch_captures_membership() {
        let (db, mgr) = setup_test_db().await;
        insert_pattern(&db, "p1", "rust", 0.8).await;
        insert_pattern(&db, "p2", "python", 0.6).await;

        let epoch = mgr.create_epoch("snap", None).await.unwrap();

        let count = mgr.pattern_count("snap").await.unwrap();
        assert_eq!(count, 2);
        assert_eq!(epoch.pattern_count, 2);
    }

    #[tokio::test]
    async fn test_create_epoch_duplicate_name_errors() {
        let (_db, mgr) = setup_test_db().await;
        mgr.create_epoch("unique", None).await.unwrap();

        let result = mgr.create_epoch("unique", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_create_epoch_empty_db() {
        let (_db, mgr) = setup_test_db().await;
        let epoch = mgr.create_epoch("empty", None).await.unwrap();

        assert_eq!(epoch.pattern_count, 0);
        assert_eq!(epoch.domain_count, 0);
        assert!((epoch.avg_reward - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_create_epoch_description_stored() {
        let (_db, mgr) = setup_test_db().await;
        let epoch = mgr
            .create_epoch("described", Some("A detailed description"))
            .await
            .unwrap();

        assert_eq!(
            epoch.description.as_deref(),
            Some("A detailed description")
        );

        let fetched = mgr.get("described").await.unwrap().unwrap();
        assert_eq!(
            fetched.description.as_deref(),
            Some("A detailed description")
        );
    }

    // -----------------------------------------------------------------------
    // branch tests (4 tests)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_branch_copies_parent_membership() {
        let (db, mgr) = setup_test_db().await;
        insert_pattern(&db, "p1", "rust", 0.8).await;
        insert_pattern(&db, "p2", "python", 0.7).await;

        mgr.create_epoch("parent", None).await.unwrap();
        let branch = mgr.branch("parent", "child").await.unwrap();

        let count = mgr.pattern_count("child").await.unwrap();
        assert_eq!(count, 2);
        assert_eq!(branch.pattern_count, 2);
    }

    #[tokio::test]
    async fn test_branch_is_not_frozen() {
        let (_db, mgr) = setup_test_db().await;
        mgr.create_epoch("parent", None).await.unwrap();
        let branch = mgr.branch("parent", "child").await.unwrap();

        assert!(!branch.frozen);
    }

    #[tokio::test]
    async fn test_branch_has_parent_epoch_set() {
        let (_db, mgr) = setup_test_db().await;
        let parent = mgr.create_epoch("parent", None).await.unwrap();
        let branch = mgr.branch("parent", "child").await.unwrap();

        assert_eq!(branch.parent_epoch.as_deref(), Some(parent.id.as_str()));
    }

    #[tokio::test]
    async fn test_branch_nonexistent_parent_errors() {
        let (_db, mgr) = setup_test_db().await;
        let result = mgr.branch("nonexistent", "child").await;
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // diff tests (4 tests)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_diff_shows_added_patterns() {
        let (db, mgr) = setup_test_db().await;
        insert_pattern(&db, "p1", "rust", 0.8).await;

        mgr.create_epoch("before", None).await.unwrap();

        insert_pattern(&db, "p2", "python", 0.9).await;
        mgr.create_epoch("after", None).await.unwrap();

        let diff = mgr.diff("before", "after").await.unwrap();

        assert_eq!(diff.added.len(), 1);
        assert!(diff.added.contains(&"p2".to_string()));
        assert_eq!(diff.common, 1);
    }

    #[tokio::test]
    async fn test_diff_shows_removed_patterns() {
        let (db, mgr) = setup_test_db().await;
        insert_pattern(&db, "p1", "rust", 0.8).await;
        insert_pattern(&db, "p2", "python", 0.7).await;

        mgr.create_epoch("full", None).await.unwrap();

        // Create a branch from full, then delete p2's membership to simulate removal
        let branch = mgr.branch("full", "partial").await.unwrap();
        let bid = branch.id.clone();
        db.execute(
            "DELETE FROM epoch_membership WHERE epoch_id = ? AND pattern_id = 'p2'",
            &[&bid as &dyn rusqlite::ToSql],
        )
        .await
        .unwrap();

        let diff = mgr.diff("full", "partial").await.unwrap();

        assert_eq!(diff.removed.len(), 1);
        assert!(diff.removed.contains(&"p2".to_string()));
        assert_eq!(diff.common, 1);
    }

    #[tokio::test]
    async fn test_diff_same_epoch_no_changes() {
        let (db, mgr) = setup_test_db().await;
        insert_pattern(&db, "p1", "rust", 0.8).await;

        mgr.create_epoch("same-a", None).await.unwrap();
        // Create a second epoch with the same patterns
        mgr.create_epoch("same-b", None).await.unwrap();

        let diff = mgr.diff("same-a", "same-b").await.unwrap();

        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert_eq!(diff.common, 1);
    }

    #[tokio::test]
    async fn test_diff_disjoint_epochs() {
        let (db, mgr) = setup_test_db().await;

        insert_pattern(&db, "p1", "rust", 0.8).await;
        mgr.create_epoch("only-p1", None).await.unwrap();

        // Remove p1, add p2
        db.execute(
            "DELETE FROM reasoning_patterns WHERE id = 'p1'",
            &[],
        )
        .await
        .unwrap();
        insert_pattern(&db, "p2", "python", 0.9).await;
        mgr.create_epoch("only-p2", None).await.unwrap();

        let diff = mgr.diff("only-p1", "only-p2").await.unwrap();

        assert_eq!(diff.added.len(), 1);
        assert!(diff.added.contains(&"p2".to_string()));
        assert_eq!(diff.removed.len(), 1);
        assert!(diff.removed.contains(&"p1".to_string()));
        assert_eq!(diff.common, 0);
    }

    // -----------------------------------------------------------------------
    // rollback tests (3 tests) - non-destructive
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_rollback_reports_patterns_to_remove() {
        let (db, mgr) = setup_test_db().await;
        insert_pattern(&db, "p1", "rust", 0.8).await;

        mgr.create_epoch("baseline", None).await.unwrap();

        // Add extra patterns after epoch
        insert_pattern(&db, "p2", "python", 0.7).await;
        insert_pattern(&db, "p3", "go", 0.6).await;

        let result = mgr.rollback("baseline").await.unwrap();

        assert_eq!(result.epoch_name, "baseline");
        assert_eq!(result.patterns_restored, 0); // all epoch patterns still present
        assert_eq!(result.patterns_removed, 2); // p2 and p3 would be removed

        // Verify rollback is non-destructive: all 3 patterns still exist
        let remaining: Vec<String> = db
            .query(
                "SELECT id FROM reasoning_patterns ORDER BY id",
                &[],
                |row| row.get(0),
            )
            .await
            .unwrap();

        assert_eq!(remaining.len(), 3);
    }

    #[tokio::test]
    async fn test_rollback_reports_patterns_to_restore() {
        let (db, mgr) = setup_test_db().await;
        insert_pattern(&db, "p1", "rust", 0.8).await;
        insert_pattern(&db, "p2", "python", 0.9).await;

        mgr.create_epoch("snapshot", None).await.unwrap();

        // Delete p2 from reasoning_patterns after creating epoch
        db.execute(
            "DELETE FROM reasoning_patterns WHERE id = 'p2'",
            &[],
        )
        .await
        .unwrap();

        let result = mgr.rollback("snapshot").await.unwrap();

        assert_eq!(result.patterns_restored, 1); // p2 needs restoring
        assert_eq!(result.patterns_removed, 0); // nothing extra to remove
    }

    #[tokio::test]
    async fn test_rollback_mixed_restore_and_remove() {
        let (db, mgr) = setup_test_db().await;
        insert_pattern(&db, "p1", "rust", 0.8).await;
        insert_pattern(&db, "p2", "python", 0.7).await;

        mgr.create_epoch("v1", None).await.unwrap();

        // Delete p2 from reasoning_patterns and add p3, p4, p5
        db.execute("DELETE FROM reasoning_patterns WHERE id = 'p2'", &[])
            .await
            .unwrap();
        insert_pattern(&db, "p3", "go", 0.6).await;
        insert_pattern(&db, "p4", "java", 0.5).await;
        insert_pattern(&db, "p5", "ts", 0.4).await;

        let result = mgr.rollback("v1").await.unwrap();

        assert_eq!(result.patterns_restored, 1); // p2
        assert_eq!(result.patterns_removed, 3); // p3, p4, p5
    }

    // -----------------------------------------------------------------------
    // list/get tests (2 tests)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_list_returns_all_epochs() {
        let (_db, mgr) = setup_test_db().await;
        mgr.create_epoch("alpha", None).await.unwrap();
        mgr.create_epoch("beta", None).await.unwrap();
        mgr.create_epoch("gamma", None).await.unwrap();

        let all = mgr.list().await.unwrap();
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn test_get_by_name_returns_correct_epoch() {
        let (db, mgr) = setup_test_db().await;
        insert_pattern(&db, "p1", "rust", 0.8).await;

        let created = mgr.create_epoch("findme", Some("desc")).await.unwrap();
        let found = mgr.get("findme").await.unwrap();

        assert!(found.is_some());
        let found = found.unwrap();
        assert_eq!(found.id, created.id);
        assert_eq!(found.name, "findme");
        assert_eq!(found.description.as_deref(), Some("desc"));
        assert_eq!(found.pattern_count, 1);
    }

    // -----------------------------------------------------------------------
    // freeze/delete tests (2 tests)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_freeze_makes_epoch_frozen() {
        let (_db, mgr) = setup_test_db().await;
        mgr.create_epoch("parent", None).await.unwrap();
        let branch = mgr.branch("parent", "mutable").await.unwrap();
        assert!(!branch.frozen);

        mgr.freeze("mutable").await.unwrap();

        let frozen = mgr.get("mutable").await.unwrap().unwrap();
        assert!(frozen.frozen);
    }

    #[tokio::test]
    async fn test_delete_removes_non_frozen_epoch() {
        let (_db, mgr) = setup_test_db().await;
        mgr.create_epoch("parent", None).await.unwrap();
        mgr.branch("parent", "deletable").await.unwrap();

        // deletable is not frozen, so we can delete it without force
        mgr.delete("deletable", false).await.unwrap();
        let gone = mgr.get("deletable").await.unwrap();
        assert!(gone.is_none());

        // parent is frozen, so delete without force should fail
        let result = mgr.delete("parent", false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delete_force_removes_frozen_epoch() {
        let (_db, mgr) = setup_test_db().await;
        mgr.create_epoch("frozen-epoch", None).await.unwrap();

        // Without force, should fail
        let result = mgr.delete("frozen-epoch", false).await;
        assert!(result.is_err());

        // With force, should succeed
        mgr.delete("frozen-epoch", true).await.unwrap();
        let gone = mgr.get("frozen-epoch").await.unwrap();
        assert!(gone.is_none());
    }

    // -----------------------------------------------------------------------
    // members tests (2 tests)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_members_returns_pattern_ids() {
        let (db, mgr) = setup_test_db().await;
        insert_pattern(&db, "p1", "rust", 0.8).await;
        insert_pattern(&db, "p2", "python", 0.7).await;
        insert_pattern(&db, "p3", "go", 0.6).await;

        mgr.create_epoch("snap", None).await.unwrap();

        let members = mgr.members("snap").await.unwrap();
        assert_eq!(members.len(), 3);
        assert!(members.contains(&"p1".to_string()));
        assert!(members.contains(&"p2".to_string()));
        assert!(members.contains(&"p3".to_string()));
    }

    #[tokio::test]
    async fn test_members_nonexistent_epoch_errors() {
        let (_db, mgr) = setup_test_db().await;
        let result = mgr.members("ghost").await;
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // rollback nonexistent epoch test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_rollback_nonexistent_epoch_errors() {
        let (_db, mgr) = setup_test_db().await;
        let result = mgr.rollback("nope").await;
        assert!(result.is_err());
    }
}
