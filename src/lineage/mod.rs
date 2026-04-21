//! Pattern Lineage tracking for the Knowledge Operating System (KOS P0).
//!
//! Tracks parent-child relationships between patterns so that merges,
//! consolidations, improvements, and derivations have a provable lineage chain.
//! This is the foundation feature that P1 (Witness Chains), P2 (Delta Event Sourcing),
//! and P5 (Epochs) depend on.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::db::SqliteDb;
use crate::error::{DatabaseError, NagualError, Result};
use crate::reasoning_bank::pattern::PatternId;

/// How a pattern was derived from its parent.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivationType {
    /// New pattern, no parent
    Original,
    /// Two patterns merged
    Merge,
    /// Similar patterns consolidated
    Consolidation,
    /// Improved based on feedback
    Improvement,
    /// Branched for experimentation
    Fork,
    /// Cross-domain transfer (P4)
    Transfer,
}

impl std::fmt::Display for DerivationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Original => write!(f, "original"),
            Self::Merge => write!(f, "merge"),
            Self::Consolidation => write!(f, "consolidation"),
            Self::Improvement => write!(f, "improvement"),
            Self::Fork => write!(f, "fork"),
            Self::Transfer => write!(f, "transfer"),
        }
    }
}

impl From<&str> for DerivationType {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "merge" => Self::Merge,
            "consolidation" => Self::Consolidation,
            "improvement" => Self::Improvement,
            "fork" => Self::Fork,
            "transfer" => Self::Transfer,
            _ => Self::Original,
        }
    }
}

/// A record representing a pattern's position in the lineage chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageRecord {
    /// The pattern this record describes.
    pub pattern_id: PatternId,
    /// The parent pattern, if any (None for originals).
    pub parent_id: Option<PatternId>,
    /// How this pattern was derived from its parent.
    pub derivation_type: DerivationType,
    /// Depth in the lineage chain (0 = original, 1 = first derivation, etc.).
    pub lineage_depth: u32,
    /// When this lineage relationship was established.
    pub created_at: DateTime<Utc>,
}

impl LineageRecord {
    /// Create a new lineage record for an original pattern (no parent).
    pub fn new_original(pattern_id: PatternId) -> Self {
        Self {
            pattern_id,
            parent_id: None,
            derivation_type: DerivationType::Original,
            lineage_depth: 0,
            created_at: Utc::now(),
        }
    }

    /// Create a new lineage record for a derived pattern.
    pub fn new_derived(
        pattern_id: PatternId,
        parent_id: PatternId,
        derivation_type: DerivationType,
        lineage_depth: u32,
    ) -> Self {
        Self {
            pattern_id,
            parent_id: Some(parent_id),
            derivation_type,
            lineage_depth,
            created_at: Utc::now(),
        }
    }

    /// Check if this is an original pattern (no parent).
    pub fn is_original(&self) -> bool {
        self.parent_id.is_none()
    }
}

/// Query interface for navigating the lineage tree.
pub struct LineageQuery {
    db: Arc<SqliteDb>,
}

impl LineageQuery {
    /// Create a new lineage query instance.
    pub fn new(db: Arc<SqliteDb>) -> Self {
        Self { db }
    }

    /// Record a lineage relationship for a pattern.
    pub async fn record(
        &self,
        pattern_id: &PatternId,
        parent_id: Option<&PatternId>,
        derivation_type: &DerivationType,
        lineage_depth: u32,
    ) -> Result<()> {
        let pid = pattern_id.as_str().to_string();
        let par = parent_id.map(|p| p.as_str().to_string());
        let dt = derivation_type.to_string();
        let depth = lineage_depth;

        self.db
            .execute(
                "UPDATE reasoning_patterns SET parent_id = ?, derivation_type = ?, lineage_depth = ? WHERE id = ?",
                &[
                    &par as &dyn rusqlite::ToSql,
                    &dt as &dyn rusqlite::ToSql,
                    &depth as &dyn rusqlite::ToSql,
                    &pid as &dyn rusqlite::ToSql,
                ],
            )
            .await?;

        Ok(())
    }

    /// Get the lineage record for a specific pattern.
    pub async fn get(&self, id: &PatternId) -> Result<Option<LineageRecord>> {
        let id_str = id.as_str().to_string();

        self.db
            .query_one(
                "SELECT id, parent_id, derivation_type, lineage_depth, created_at \
                 FROM reasoning_patterns WHERE id = ?",
                &[&id_str as &dyn rusqlite::ToSql],
                |row| {
                    let pattern_id: String = row.get(0)?;
                    let parent_id: Option<String> = row.get(1)?;
                    let derivation_type: Option<String> = row.get(2)?;
                    let lineage_depth: Option<i64> = row.get(3)?;
                    let created_at: String = row.get(4)?;

                    Ok(LineageRecord {
                        pattern_id: PatternId::from_string(pattern_id),
                        parent_id: parent_id.map(PatternId::from_string),
                        derivation_type: DerivationType::from(
                            derivation_type.as_deref().unwrap_or("original"),
                        ),
                        lineage_depth: lineage_depth.unwrap_or(0) as u32,
                        created_at: DateTime::parse_from_rfc3339(&created_at)
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(|_| Utc::now()),
                    })
                },
            )
            .await
    }

    /// Get all ancestors of a pattern (walk up the parent chain).
    ///
    /// Returns ancestors from immediate parent to root, in order.
    /// The pattern itself is NOT included in the result.
    pub async fn ancestors(&self, id: &PatternId) -> Result<Vec<LineageRecord>> {
        let mut result = Vec::new();
        let mut current_id = id.clone();

        // Walk up the parent chain with a safety limit to prevent infinite loops
        for _ in 0..100 {
            let record = self.get(&current_id).await?;

            match record {
                Some(rec) => {
                    if let Some(ref parent_id) = rec.parent_id {
                        // Get the parent's record and add it
                        if let Some(parent_rec) = self.get(parent_id).await? {
                            current_id = parent_id.clone();
                            result.push(parent_rec);
                        } else {
                            break;
                        }
                    } else {
                        break; // Reached root
                    }
                }
                None => break,
            }
        }

        Ok(result)
    }

    /// Get all descendants of a pattern (all children, grandchildren, etc.).
    ///
    /// Uses breadth-first traversal. The pattern itself is NOT included.
    pub async fn descendants(&self, id: &PatternId) -> Result<Vec<LineageRecord>> {
        let mut result = Vec::new();
        let mut queue = vec![id.clone()];

        // BFS with safety limit
        for _ in 0..1000 {
            if queue.is_empty() {
                break;
            }

            let current_id = queue.remove(0);
            let children = self.children(&current_id).await?;

            for child in children {
                queue.push(child.pattern_id.clone());
                result.push(child);
            }
        }

        Ok(result)
    }

    /// Get the lineage depth for a pattern.
    ///
    /// Returns 0 for original patterns, or the stored lineage_depth value.
    pub async fn depth(&self, id: &PatternId) -> Result<u32> {
        let id_str = id.as_str().to_string();

        let depth = self
            .db
            .query_one(
                "SELECT lineage_depth FROM reasoning_patterns WHERE id = ?",
                &[&id_str as &dyn rusqlite::ToSql],
                |row| {
                    let d: Option<i64> = row.get(0)?;
                    Ok(d.unwrap_or(0) as u32)
                },
            )
            .await?;

        Ok(depth.unwrap_or(0))
    }

    /// Get direct children of a pattern.
    pub async fn children(&self, id: &PatternId) -> Result<Vec<LineageRecord>> {
        let id_str = id.as_str().to_string();

        self.db
            .query(
                "SELECT id, parent_id, derivation_type, lineage_depth, created_at \
                 FROM reasoning_patterns WHERE parent_id = ?",
                &[&id_str as &dyn rusqlite::ToSql],
                |row| {
                    let pattern_id: String = row.get(0)?;
                    let parent_id: Option<String> = row.get(1)?;
                    let derivation_type: Option<String> = row.get(2)?;
                    let lineage_depth: Option<i64> = row.get(3)?;
                    let created_at: String = row.get(4)?;

                    Ok(LineageRecord {
                        pattern_id: PatternId::from_string(pattern_id),
                        parent_id: parent_id.map(PatternId::from_string),
                        derivation_type: DerivationType::from(
                            derivation_type.as_deref().unwrap_or("original"),
                        ),
                        lineage_depth: lineage_depth.unwrap_or(0) as u32,
                        created_at: DateTime::parse_from_rfc3339(&created_at)
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(|_| Utc::now()),
                    })
                },
            )
            .await
    }

    /// Get all patterns of a specific derivation type.
    pub async fn by_derivation_type(
        &self,
        derivation_type: &DerivationType,
    ) -> Result<Vec<LineageRecord>> {
        let dt = derivation_type.to_string();

        self.db
            .query(
                "SELECT id, parent_id, derivation_type, lineage_depth, created_at \
                 FROM reasoning_patterns WHERE derivation_type = ?",
                &[&dt as &dyn rusqlite::ToSql],
                |row| {
                    let pattern_id: String = row.get(0)?;
                    let parent_id: Option<String> = row.get(1)?;
                    let derivation_type: Option<String> = row.get(2)?;
                    let lineage_depth: Option<i64> = row.get(3)?;
                    let created_at: String = row.get(4)?;

                    Ok(LineageRecord {
                        pattern_id: PatternId::from_string(pattern_id),
                        parent_id: parent_id.map(PatternId::from_string),
                        derivation_type: DerivationType::from(
                            derivation_type.as_deref().unwrap_or("original"),
                        ),
                        lineage_depth: lineage_depth.unwrap_or(0) as u32,
                        created_at: DateTime::parse_from_rfc3339(&created_at)
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(|_| Utc::now()),
                    })
                },
            )
            .await
    }

    /// Count patterns at each lineage depth level.
    pub async fn depth_distribution(&self) -> Result<Vec<(u32, u64)>> {
        self.db
            .query(
                "SELECT COALESCE(lineage_depth, 0) as depth, COUNT(*) as cnt \
                 FROM reasoning_patterns \
                 GROUP BY depth ORDER BY depth",
                &[],
                |row| {
                    let depth: i64 = row.get(0)?;
                    let count: i64 = row.get(1)?;
                    Ok((depth as u32, count as u64))
                },
            )
            .await
    }

    /// Get the full lineage chain from root to a given pattern.
    ///
    /// Returns records from root to the pattern (inclusive), in order.
    pub async fn chain_to_root(&self, id: &PatternId) -> Result<Vec<LineageRecord>> {
        let mut ancestors = self.ancestors(id).await?;
        ancestors.reverse(); // Now root-first

        // Add the pattern itself at the end
        if let Some(self_record) = self.get(id).await? {
            ancestors.push(self_record);
        }

        Ok(ancestors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- DerivationType tests --

    #[test]
    fn test_derivation_type_display_original() {
        assert_eq!(DerivationType::Original.to_string(), "original");
    }

    #[test]
    fn test_derivation_type_display_merge() {
        assert_eq!(DerivationType::Merge.to_string(), "merge");
    }

    #[test]
    fn test_derivation_type_display_consolidation() {
        assert_eq!(DerivationType::Consolidation.to_string(), "consolidation");
    }

    #[test]
    fn test_derivation_type_display_improvement() {
        assert_eq!(DerivationType::Improvement.to_string(), "improvement");
    }

    #[test]
    fn test_derivation_type_display_fork() {
        assert_eq!(DerivationType::Fork.to_string(), "fork");
    }

    #[test]
    fn test_derivation_type_display_transfer() {
        assert_eq!(DerivationType::Transfer.to_string(), "transfer");
    }

    #[test]
    fn test_derivation_type_from_str_exact() {
        assert_eq!(DerivationType::from("merge"), DerivationType::Merge);
        assert_eq!(
            DerivationType::from("consolidation"),
            DerivationType::Consolidation
        );
        assert_eq!(
            DerivationType::from("improvement"),
            DerivationType::Improvement
        );
        assert_eq!(DerivationType::from("fork"), DerivationType::Fork);
        assert_eq!(DerivationType::from("transfer"), DerivationType::Transfer);
        assert_eq!(DerivationType::from("original"), DerivationType::Original);
    }

    #[test]
    fn test_derivation_type_from_str_case_insensitive() {
        assert_eq!(DerivationType::from("MERGE"), DerivationType::Merge);
        assert_eq!(DerivationType::from("Merge"), DerivationType::Merge);
        assert_eq!(DerivationType::from("Fork"), DerivationType::Fork);
        assert_eq!(DerivationType::from("TRANSFER"), DerivationType::Transfer);
    }

    #[test]
    fn test_derivation_type_from_str_unknown() {
        assert_eq!(DerivationType::from("unknown"), DerivationType::Original);
        assert_eq!(DerivationType::from(""), DerivationType::Original);
        assert_eq!(DerivationType::from("blah"), DerivationType::Original);
    }

    #[test]
    fn test_derivation_type_serde_roundtrip() {
        let types = vec![
            DerivationType::Original,
            DerivationType::Merge,
            DerivationType::Consolidation,
            DerivationType::Improvement,
            DerivationType::Fork,
            DerivationType::Transfer,
        ];

        for dt in types {
            let json = serde_json::to_string(&dt).unwrap();
            let deserialized: DerivationType = serde_json::from_str(&json).unwrap();
            assert_eq!(dt, deserialized, "Failed roundtrip for {:?}", dt);
        }
    }

    #[test]
    fn test_derivation_type_serde_json_values() {
        assert_eq!(
            serde_json::to_string(&DerivationType::Original).unwrap(),
            "\"original\""
        );
        assert_eq!(
            serde_json::to_string(&DerivationType::Merge).unwrap(),
            "\"merge\""
        );
        assert_eq!(
            serde_json::to_string(&DerivationType::Consolidation).unwrap(),
            "\"consolidation\""
        );
    }

    #[test]
    fn test_derivation_type_equality() {
        assert_eq!(DerivationType::Merge, DerivationType::Merge);
        assert_ne!(DerivationType::Merge, DerivationType::Fork);
    }

    #[test]
    fn test_derivation_type_clone() {
        let dt = DerivationType::Improvement;
        let cloned = dt.clone();
        assert_eq!(dt, cloned);
    }

    #[test]
    fn test_derivation_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DerivationType::Merge);
        set.insert(DerivationType::Merge); // duplicate
        set.insert(DerivationType::Fork);
        assert_eq!(set.len(), 2);
    }

    // -- LineageRecord tests --

    #[test]
    fn test_lineage_record_new_original() {
        let id = PatternId::from_string("p1");
        let record = LineageRecord::new_original(id.clone());

        assert_eq!(record.pattern_id, id);
        assert!(record.parent_id.is_none());
        assert_eq!(record.derivation_type, DerivationType::Original);
        assert_eq!(record.lineage_depth, 0);
        assert!(record.is_original());
    }

    #[test]
    fn test_lineage_record_new_derived() {
        let child_id = PatternId::from_string("child");
        let parent_id = PatternId::from_string("parent");

        let record = LineageRecord::new_derived(
            child_id.clone(),
            parent_id.clone(),
            DerivationType::Improvement,
            1,
        );

        assert_eq!(record.pattern_id, child_id);
        assert_eq!(record.parent_id, Some(parent_id));
        assert_eq!(record.derivation_type, DerivationType::Improvement);
        assert_eq!(record.lineage_depth, 1);
        assert!(!record.is_original());
    }

    #[test]
    fn test_lineage_record_is_original() {
        let original = LineageRecord::new_original(PatternId::from_string("p1"));
        assert!(original.is_original());

        let derived = LineageRecord::new_derived(
            PatternId::from_string("p2"),
            PatternId::from_string("p1"),
            DerivationType::Fork,
            1,
        );
        assert!(!derived.is_original());
    }

    #[test]
    fn test_lineage_record_fields_accessible() {
        let record = LineageRecord::new_derived(
            PatternId::from_string("c1"),
            PatternId::from_string("p1"),
            DerivationType::Merge,
            3,
        );

        assert_eq!(record.pattern_id.as_str(), "c1");
        assert_eq!(record.parent_id.as_ref().unwrap().as_str(), "p1");
        assert_eq!(record.lineage_depth, 3);
        // created_at should be very recent
        let elapsed = Utc::now()
            .signed_duration_since(record.created_at)
            .num_seconds();
        assert!(elapsed < 2);
    }

    #[test]
    fn test_lineage_record_serde_roundtrip() {
        let record = LineageRecord::new_derived(
            PatternId::from_string("c1"),
            PatternId::from_string("p1"),
            DerivationType::Consolidation,
            2,
        );

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: LineageRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.pattern_id, record.pattern_id);
        assert_eq!(deserialized.parent_id, record.parent_id);
        assert_eq!(deserialized.derivation_type, record.derivation_type);
        assert_eq!(deserialized.lineage_depth, record.lineage_depth);
    }

    #[test]
    fn test_lineage_record_original_serde_roundtrip() {
        let record = LineageRecord::new_original(PatternId::from_string("root"));

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: LineageRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.pattern_id.as_str(), "root");
        assert!(deserialized.parent_id.is_none());
        assert_eq!(deserialized.derivation_type, DerivationType::Original);
        assert_eq!(deserialized.lineage_depth, 0);
    }

    #[test]
    fn test_lineage_record_clone() {
        let record = LineageRecord::new_derived(
            PatternId::from_string("c"),
            PatternId::from_string("p"),
            DerivationType::Transfer,
            5,
        );
        let cloned = record.clone();
        assert_eq!(cloned.pattern_id, record.pattern_id);
        assert_eq!(cloned.lineage_depth, record.lineage_depth);
    }

    // -- LineageQuery integration tests (use in-memory SQLite) --

    /// Helper to create an in-memory database with the lineage schema.
    async fn setup_test_db() -> Arc<SqliteDb> {
        let db = SqliteDb::open_in_memory().unwrap();

        // Create minimal reasoning_patterns table with lineage columns
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT NOT NULL DEFAULT '',
                solution TEXT NOT NULL DEFAULT '',
                domain TEXT DEFAULT '',
                parent_id TEXT,
                derivation_type TEXT,
                lineage_depth INTEGER DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_patterns_parent_id ON reasoning_patterns(parent_id);
            CREATE INDEX IF NOT EXISTS idx_patterns_derivation_type ON reasoning_patterns(derivation_type);",
        )
        .await
        .unwrap();

        Arc::new(db)
    }

    /// Helper to insert a pattern into the test database.
    async fn insert_pattern(
        db: &SqliteDb,
        id: &str,
        parent_id: Option<&str>,
        derivation_type: Option<&str>,
        lineage_depth: u32,
    ) {
        let now = Utc::now().to_rfc3339();
        db.execute(
            "INSERT INTO reasoning_patterns (id, problem, solution, parent_id, derivation_type, lineage_depth, created_at) \
             VALUES (?, 'test problem', 'test solution', ?, ?, ?, ?)",
            &[
                &id as &dyn rusqlite::ToSql,
                &parent_id as &dyn rusqlite::ToSql,
                &derivation_type as &dyn rusqlite::ToSql,
                &lineage_depth as &dyn rusqlite::ToSql,
                &now as &dyn rusqlite::ToSql,
            ],
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_lineage_query_get_existing() {
        let db = setup_test_db().await;
        insert_pattern(&db, "p1", None, Some("original"), 0).await;

        let query = LineageQuery::new(db);
        let record = query.get(&PatternId::from_string("p1")).await.unwrap();

        assert!(record.is_some());
        let record = record.unwrap();
        assert_eq!(record.pattern_id.as_str(), "p1");
        assert!(record.parent_id.is_none());
        assert_eq!(record.derivation_type, DerivationType::Original);
        assert_eq!(record.lineage_depth, 0);
    }

    #[tokio::test]
    async fn test_lineage_query_get_nonexistent() {
        let db = setup_test_db().await;
        let query = LineageQuery::new(db);
        let record = query
            .get(&PatternId::from_string("nonexistent"))
            .await
            .unwrap();
        assert!(record.is_none());
    }

    #[tokio::test]
    async fn test_lineage_query_record() {
        let db = setup_test_db().await;
        insert_pattern(&db, "p1", None, None, 0).await;

        let query = LineageQuery::new(db);
        query
            .record(
                &PatternId::from_string("p1"),
                Some(&PatternId::from_string("p0")),
                &DerivationType::Improvement,
                1,
            )
            .await
            .unwrap();

        let record = query.get(&PatternId::from_string("p1")).await.unwrap().unwrap();
        assert_eq!(record.parent_id.as_ref().unwrap().as_str(), "p0");
        assert_eq!(record.derivation_type, DerivationType::Improvement);
        assert_eq!(record.lineage_depth, 1);
    }

    #[tokio::test]
    async fn test_lineage_query_children() {
        let db = setup_test_db().await;
        insert_pattern(&db, "root", None, Some("original"), 0).await;
        insert_pattern(&db, "child1", Some("root"), Some("improvement"), 1).await;
        insert_pattern(&db, "child2", Some("root"), Some("fork"), 1).await;
        insert_pattern(&db, "unrelated", None, Some("original"), 0).await;

        let query = LineageQuery::new(db);
        let children = query
            .children(&PatternId::from_string("root"))
            .await
            .unwrap();

        assert_eq!(children.len(), 2);
        let child_ids: Vec<&str> = children.iter().map(|c| c.pattern_id.as_str()).collect();
        assert!(child_ids.contains(&"child1"));
        assert!(child_ids.contains(&"child2"));
    }

    #[tokio::test]
    async fn test_lineage_query_children_empty() {
        let db = setup_test_db().await;
        insert_pattern(&db, "leaf", None, Some("original"), 0).await;

        let query = LineageQuery::new(db);
        let children = query
            .children(&PatternId::from_string("leaf"))
            .await
            .unwrap();
        assert!(children.is_empty());
    }

    #[tokio::test]
    async fn test_lineage_query_ancestors_3_levels() {
        let db = setup_test_db().await;
        // root -> middle -> leaf
        insert_pattern(&db, "root", None, Some("original"), 0).await;
        insert_pattern(&db, "middle", Some("root"), Some("improvement"), 1).await;
        insert_pattern(&db, "leaf", Some("middle"), Some("consolidation"), 2).await;

        let query = LineageQuery::new(db);
        let ancestors = query
            .ancestors(&PatternId::from_string("leaf"))
            .await
            .unwrap();

        // Should be [middle, root] (walking up from leaf)
        assert_eq!(ancestors.len(), 2);
        assert_eq!(ancestors[0].pattern_id.as_str(), "middle");
        assert_eq!(ancestors[1].pattern_id.as_str(), "root");
    }

    #[tokio::test]
    async fn test_lineage_query_ancestors_original_has_none() {
        let db = setup_test_db().await;
        insert_pattern(&db, "root", None, Some("original"), 0).await;

        let query = LineageQuery::new(db);
        let ancestors = query
            .ancestors(&PatternId::from_string("root"))
            .await
            .unwrap();
        assert!(ancestors.is_empty());
    }

    #[tokio::test]
    async fn test_lineage_query_descendants() {
        let db = setup_test_db().await;
        // root -> child1 -> grandchild1
        //      -> child2
        insert_pattern(&db, "root", None, Some("original"), 0).await;
        insert_pattern(&db, "child1", Some("root"), Some("improvement"), 1).await;
        insert_pattern(&db, "child2", Some("root"), Some("fork"), 1).await;
        insert_pattern(&db, "grandchild1", Some("child1"), Some("merge"), 2).await;

        let query = LineageQuery::new(db);
        let descendants = query
            .descendants(&PatternId::from_string("root"))
            .await
            .unwrap();

        assert_eq!(descendants.len(), 3);
        let desc_ids: Vec<&str> = descendants.iter().map(|d| d.pattern_id.as_str()).collect();
        assert!(desc_ids.contains(&"child1"));
        assert!(desc_ids.contains(&"child2"));
        assert!(desc_ids.contains(&"grandchild1"));
    }

    #[tokio::test]
    async fn test_lineage_query_descendants_leaf() {
        let db = setup_test_db().await;
        insert_pattern(&db, "leaf", None, Some("original"), 0).await;

        let query = LineageQuery::new(db);
        let descendants = query
            .descendants(&PatternId::from_string("leaf"))
            .await
            .unwrap();
        assert!(descendants.is_empty());
    }

    #[tokio::test]
    async fn test_lineage_query_depth() {
        let db = setup_test_db().await;
        insert_pattern(&db, "root", None, Some("original"), 0).await;
        insert_pattern(&db, "child", Some("root"), Some("improvement"), 1).await;
        insert_pattern(&db, "grandchild", Some("child"), Some("fork"), 2).await;

        let query = LineageQuery::new(db);

        assert_eq!(
            query.depth(&PatternId::from_string("root")).await.unwrap(),
            0
        );
        assert_eq!(
            query.depth(&PatternId::from_string("child")).await.unwrap(),
            1
        );
        assert_eq!(
            query
                .depth(&PatternId::from_string("grandchild"))
                .await
                .unwrap(),
            2
        );
    }

    #[tokio::test]
    async fn test_lineage_query_depth_nonexistent() {
        let db = setup_test_db().await;
        let query = LineageQuery::new(db);

        let depth = query
            .depth(&PatternId::from_string("nonexistent"))
            .await
            .unwrap();
        assert_eq!(depth, 0);
    }

    #[tokio::test]
    async fn test_lineage_query_by_derivation_type() {
        let db = setup_test_db().await;
        insert_pattern(&db, "p1", None, Some("original"), 0).await;
        insert_pattern(&db, "p2", Some("p1"), Some("merge"), 1).await;
        insert_pattern(&db, "p3", Some("p1"), Some("merge"), 1).await;
        insert_pattern(&db, "p4", Some("p1"), Some("fork"), 1).await;

        let query = LineageQuery::new(db);
        let merges = query
            .by_derivation_type(&DerivationType::Merge)
            .await
            .unwrap();

        assert_eq!(merges.len(), 2);
        for m in &merges {
            assert_eq!(m.derivation_type, DerivationType::Merge);
        }
    }

    #[tokio::test]
    async fn test_lineage_query_depth_distribution() {
        let db = setup_test_db().await;
        insert_pattern(&db, "r1", None, Some("original"), 0).await;
        insert_pattern(&db, "r2", None, Some("original"), 0).await;
        insert_pattern(&db, "c1", Some("r1"), Some("improvement"), 1).await;
        insert_pattern(&db, "g1", Some("c1"), Some("merge"), 2).await;

        let query = LineageQuery::new(db);
        let dist = query.depth_distribution().await.unwrap();

        // Should have 3 levels: depth 0 (2 patterns), depth 1 (1), depth 2 (1)
        assert_eq!(dist.len(), 3);
        assert_eq!(dist[0], (0, 2));
        assert_eq!(dist[1], (1, 1));
        assert_eq!(dist[2], (2, 1));
    }

    #[tokio::test]
    async fn test_lineage_query_chain_to_root() {
        let db = setup_test_db().await;
        insert_pattern(&db, "root", None, Some("original"), 0).await;
        insert_pattern(&db, "mid", Some("root"), Some("improvement"), 1).await;
        insert_pattern(&db, "leaf", Some("mid"), Some("fork"), 2).await;

        let query = LineageQuery::new(db);
        let chain = query
            .chain_to_root(&PatternId::from_string("leaf"))
            .await
            .unwrap();

        // Should be [root, mid, leaf]
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].pattern_id.as_str(), "root");
        assert_eq!(chain[1].pattern_id.as_str(), "mid");
        assert_eq!(chain[2].pattern_id.as_str(), "leaf");
    }

    #[tokio::test]
    async fn test_lineage_query_chain_to_root_original() {
        let db = setup_test_db().await;
        insert_pattern(&db, "root", None, Some("original"), 0).await;

        let query = LineageQuery::new(db);
        let chain = query
            .chain_to_root(&PatternId::from_string("root"))
            .await
            .unwrap();

        // Just the root itself
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].pattern_id.as_str(), "root");
    }
}
