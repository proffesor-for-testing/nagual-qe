//! Pattern storage with dual-write to SQLite and PostgreSQL.
//!
//! Provides persistent storage for patterns with automatic synchronization
//! between local SQLite and cloud PostgreSQL databases.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{debug, info, warn};

use crate::db::fts::{fts_search, FtsSearchOptions, PatternFts};
use crate::coherence::{CoherenceAction, CoherenceGate, StoreWithCoherenceResult};

use super::pattern::{BetaParams, FailureMode, Pattern, PatternCategory, PatternId, PatternMetadata};
use crate::db::{DualWritable, DualWriteAdapter};
use crate::error::{DatabaseError, NagualError, Result};

/// Configuration for pattern storage.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Default embedding dimension (for validation).
    pub embedding_dim: usize,

    /// Whether to auto-generate embeddings if not provided.
    pub auto_generate_embeddings: bool,

    /// Table name for patterns in the database.
    pub table_name: String,

    /// Maximum number of tags per pattern.
    pub max_tags: usize,

    /// Maximum number of related patterns to store.
    pub max_related_patterns: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            auto_generate_embeddings: false, // Requires ML model, so off by default
            table_name: "reasoning_patterns".to_string(),
            max_tags: 20,
            max_related_patterns: 50,
        }
    }
}

/// Storable wrapper for Pattern that implements DualWritable.
///
/// This is a newtype wrapper that provides the database interface
/// for Pattern while keeping Pattern itself database-agnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorablePattern {
    /// The underlying pattern
    #[serde(flatten)]
    pattern: Pattern,
}

impl StorablePattern {
    /// Create from a Pattern.
    pub fn new(pattern: Pattern) -> Self {
        Self { pattern }
    }

    /// Get the underlying pattern.
    pub fn into_pattern(self) -> Pattern {
        self.pattern
    }

    /// Get a reference to the underlying pattern.
    pub fn pattern(&self) -> &Pattern {
        &self.pattern
    }
}

#[async_trait]
impl DualWritable for StorablePattern {
    type Id = String;

    fn table_name() -> &'static str {
        "reasoning_patterns"
    }

    fn id(&self) -> Self::Id {
        self.pattern.id().to_string()
    }

    fn updated_at(&self) -> DateTime<Utc> {
        self.pattern.updated_at()
    }

    fn set_updated_at(&mut self, ts: DateTime<Utc>) {
        self.pattern.set_updated_at(ts);
    }

    fn sqlite_insert_sql() -> &'static str {
        r#"
        INSERT OR REPLACE INTO reasoning_patterns (
            id, timestamp, updated_at, category, problem, solution, context,
            effectiveness, reuse_count, reward, success, critique,
            agent_id, session_id, confidence, embedding, tags,
            related_patterns, metadata,
            surprise_score, failure_mode, chunk_embeddings,
            satisfaction_score, satisfaction_trials, content_hash,
            title, summary,
            quality_alpha, quality_beta,
            embedding_method
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#
    }

    fn sqlite_update_sql() -> &'static str {
        r#"
        UPDATE reasoning_patterns SET
            updated_at = ?,
            category = ?,
            problem = ?,
            solution = ?,
            context = ?,
            effectiveness = ?,
            reuse_count = ?,
            reward = ?,
            success = ?,
            critique = ?,
            agent_id = ?,
            session_id = ?,
            confidence = ?,
            embedding = ?,
            tags = ?,
            related_patterns = ?,
            metadata = ?,
            surprise_score = ?,
            failure_mode = ?,
            chunk_embeddings = ?,
            satisfaction_score = ?,
            satisfaction_trials = ?,
            content_hash = ?,
            title = ?,
            summary = ?,
            quality_alpha = ?,
            quality_beta = ?,
            embedding_method = ?,
            tier = CASE
                WHEN CAST(? AS REAL) >= 0.9 AND CAST(? AS INTEGER) >= 20 THEN 'reflex'
                WHEN CAST(? AS REAL) >= 0.7 AND CAST(? AS INTEGER) >= 5 THEN 'crystal'
                ELSE 'booster'
            END
        WHERE id = ?
        "#
    }

    fn sqlite_delete_sql() -> &'static str {
        "DELETE FROM reasoning_patterns WHERE id = ?"
    }

    fn sqlite_insert_params(&self) -> Vec<Box<dyn rusqlite::ToSql + Send + Sync>> {
        let embedding_json = self
            .pattern
            .embedding()
            .map(|e| serde_json::to_string(e).unwrap_or_default());

        let tags_json = serde_json::to_string(self.pattern.tags()).unwrap_or_else(|_| "[]".to_string());

        let related_json = serde_json::to_string(self.pattern.related_patterns())
            .unwrap_or_else(|_| "[]".to_string());

        let metadata_json =
            serde_json::to_string(self.pattern.metadata()).unwrap_or_else(|_| "{}".to_string());

        let chunk_embeddings_json = self
            .pattern
            .chunk_embeddings()
            .map(|c| serde_json::to_string(c).unwrap_or_default());

        vec![
            Box::new(self.pattern.id().to_string()),
            Box::new(self.pattern.timestamp().to_rfc3339()),
            Box::new(self.pattern.updated_at().to_rfc3339()),
            Box::new(self.pattern.category().to_string()),
            Box::new(self.pattern.problem().to_string()),
            Box::new(self.pattern.solution().to_string()),
            Box::new(self.pattern.context().to_string()),
            Box::new(self.pattern.effectiveness() as f64),
            Box::new(self.pattern.reuse_count() as i32),
            Box::new(self.pattern.reward() as f64),
            Box::new(self.pattern.success()),
            Box::new(self.pattern.critique().to_string()),
            Box::new(self.pattern.agent_id().map(|s| s.to_string())),
            Box::new(self.pattern.session_id().map(|s| s.to_string())),
            Box::new(self.pattern.confidence() as f64),
            Box::new(embedding_json),
            Box::new(tags_json),
            Box::new(related_json),
            Box::new(metadata_json),
            // New columns
            Box::new(self.pattern.surprise_score() as f64),
            Box::new(self.pattern.failure_mode().map(|m| m.to_string())),
            Box::new(chunk_embeddings_json),
            // Software Factory enhancements (Week 1 WS-A)
            Box::new(self.pattern.satisfaction_score() as f64),
            Box::new(self.pattern.satisfaction_trials() as i32),
            Box::new(self.pattern.content_hash().map(|s| s.to_string())),
            // Pyramid summary fields (Week 1 WS-C)
            Box::new(self.pattern.title().map(|s| s.to_string())),
            Box::new(self.pattern.summary().map(|s| s.to_string())),
            // Bayesian quality score (Beta distribution)
            Box::new(self.pattern.bayesian_score().alpha()),
            Box::new(self.pattern.bayesian_score().beta()),
            // Embedding method (hash or onnx)
            Box::new(self.pattern.embedding_method().map(|s| s.to_string())),
        ]
    }

    fn sqlite_update_params(&self) -> Vec<Box<dyn rusqlite::ToSql + Send + Sync>> {
        let embedding_json = self
            .pattern
            .embedding()
            .map(|e| serde_json::to_string(e).unwrap_or_default());

        let tags_json = serde_json::to_string(self.pattern.tags()).unwrap_or_else(|_| "[]".to_string());

        let related_json = serde_json::to_string(self.pattern.related_patterns())
            .unwrap_or_else(|_| "[]".to_string());

        let metadata_json =
            serde_json::to_string(self.pattern.metadata()).unwrap_or_else(|_| "{}".to_string());

        let chunk_embeddings_json = self
            .pattern
            .chunk_embeddings()
            .map(|c| serde_json::to_string(c).unwrap_or_default());

        vec![
            Box::new(self.pattern.updated_at().to_rfc3339()),
            Box::new(self.pattern.category().to_string()),
            Box::new(self.pattern.problem().to_string()),
            Box::new(self.pattern.solution().to_string()),
            Box::new(self.pattern.context().to_string()),
            Box::new(self.pattern.effectiveness() as f64),
            Box::new(self.pattern.reuse_count() as i32),
            Box::new(self.pattern.reward() as f64),
            Box::new(self.pattern.success()),
            Box::new(self.pattern.critique().to_string()),
            Box::new(self.pattern.agent_id().map(|s| s.to_string())),
            Box::new(self.pattern.session_id().map(|s| s.to_string())),
            Box::new(self.pattern.confidence() as f64),
            Box::new(embedding_json),
            Box::new(tags_json),
            Box::new(related_json),
            Box::new(metadata_json),
            // New columns
            Box::new(self.pattern.surprise_score() as f64),
            Box::new(self.pattern.failure_mode().map(|m| m.to_string())),
            Box::new(chunk_embeddings_json),
            // Software Factory enhancements (Week 1 WS-A)
            Box::new(self.pattern.satisfaction_score() as f64),
            Box::new(self.pattern.satisfaction_trials() as i32),
            Box::new(self.pattern.content_hash().map(|s| s.to_string())),
            // Pyramid summary fields (Week 1 WS-C)
            Box::new(self.pattern.title().map(|s| s.to_string())),
            Box::new(self.pattern.summary().map(|s| s.to_string())),
            // Bayesian quality score (Beta distribution)
            Box::new(self.pattern.bayesian_score().alpha()),
            Box::new(self.pattern.bayesian_score().beta()),
            // Embedding method (hash or onnx)
            Box::new(self.pattern.embedding_method().map(|s| s.to_string())),
            // Tier CASE params: reward, reuse_count (bound twice for two conditions)
            Box::new(self.pattern.reward() as f64),
            Box::new(self.pattern.reuse_count() as i32),
            Box::new(self.pattern.reward() as f64),
            Box::new(self.pattern.reuse_count() as i32),
            // WHERE clause
            Box::new(self.pattern.id().to_string()),
        ]
    }

    fn postgres_insert_sql() -> &'static str {
        r#"
        INSERT INTO reasoning_patterns (
            id, timestamp, updated_at, category, problem, solution, context,
            effectiveness, reuse_count, reward, success, critique,
            agent_id, session_id, confidence, embedding, tags,
            related_patterns, metadata,
            surprise_score, failure_mode, chunk_embeddings,
            satisfaction_score, satisfaction_trials, content_hash,
            title, summary,
            quality_alpha, quality_beta,
            embedding_method
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30)
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
            metadata = EXCLUDED.metadata,
            surprise_score = EXCLUDED.surprise_score,
            failure_mode = EXCLUDED.failure_mode,
            chunk_embeddings = EXCLUDED.chunk_embeddings,
            satisfaction_score = EXCLUDED.satisfaction_score,
            satisfaction_trials = EXCLUDED.satisfaction_trials,
            content_hash = EXCLUDED.content_hash,
            title = EXCLUDED.title,
            summary = EXCLUDED.summary,
            quality_alpha = EXCLUDED.quality_alpha,
            quality_beta = EXCLUDED.quality_beta,
            embedding_method = EXCLUDED.embedding_method
        "#
    }

    fn postgres_update_sql() -> &'static str {
        r#"
        UPDATE reasoning_patterns SET
            updated_at = $1,
            category = $2,
            problem = $3,
            solution = $4,
            context = $5,
            effectiveness = $6,
            reuse_count = $7,
            reward = $8,
            success = $9,
            critique = $10,
            agent_id = $11,
            session_id = $12,
            confidence = $13,
            embedding = $14,
            tags = $15,
            related_patterns = $16,
            metadata = $17,
            surprise_score = $18,
            failure_mode = $19,
            chunk_embeddings = $20,
            satisfaction_score = $21,
            satisfaction_trials = $22,
            content_hash = $23,
            title = $24,
            summary = $25,
            quality_alpha = $26,
            quality_beta = $27,
            embedding_method = $28
        WHERE id = $29
        "#
    }

    async fn postgres_insert(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error> {
        let embedding_vec: Option<Vec<f32>> = self.pattern.embedding().map(|e| e.to_vec());
        let tags_json = serde_json::to_value(self.pattern.tags()).unwrap_or(serde_json::json!([]));
        let related_json =
            serde_json::to_value(self.pattern.related_patterns()).unwrap_or(serde_json::json!([]));
        let metadata_json =
            serde_json::to_value(self.pattern.metadata()).unwrap_or(serde_json::json!({}));
        let chunk_embeddings_json: Option<serde_json::Value> = self
            .pattern
            .chunk_embeddings()
            .map(|c| serde_json::to_value(c).unwrap_or(serde_json::json!([])));

        sqlx::query(Self::postgres_insert_sql())
            .bind(self.pattern.id().to_string())
            .bind(self.pattern.timestamp())
            .bind(self.pattern.updated_at())
            .bind(self.pattern.category().to_string())
            .bind(self.pattern.problem())
            .bind(self.pattern.solution())
            .bind(self.pattern.context())
            .bind(self.pattern.effectiveness() as f64)
            .bind(self.pattern.reuse_count() as i32)
            .bind(self.pattern.reward() as f64)
            .bind(self.pattern.success())
            .bind(self.pattern.critique())
            .bind(self.pattern.agent_id())
            .bind(self.pattern.session_id())
            .bind(self.pattern.confidence() as f64)
            .bind(embedding_vec)
            .bind(tags_json)
            .bind(related_json)
            .bind(metadata_json)
            .bind(self.pattern.surprise_score() as f64)
            .bind(self.pattern.failure_mode().map(|m| m.to_string()))
            .bind(chunk_embeddings_json)
            // Software Factory enhancements (Week 1 WS-A)
            .bind(self.pattern.satisfaction_score() as f64)
            .bind(self.pattern.satisfaction_trials() as i32)
            .bind(self.pattern.content_hash())
            // Pyramid summary fields (Week 1 WS-C)
            .bind(self.pattern.title())
            .bind(self.pattern.summary())
            // Bayesian quality score (Beta distribution)
            .bind(self.pattern.bayesian_score().alpha())
            .bind(self.pattern.bayesian_score().beta())
            // Embedding method (hash or onnx)
            .bind(self.pattern.embedding_method())
            .execute(pool)
            .await?;

        Ok(())
    }

    async fn postgres_update(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error> {
        let embedding_vec: Option<Vec<f32>> = self.pattern.embedding().map(|e| e.to_vec());
        let tags_json = serde_json::to_value(self.pattern.tags()).unwrap_or(serde_json::json!([]));
        let related_json =
            serde_json::to_value(self.pattern.related_patterns()).unwrap_or(serde_json::json!([]));
        let metadata_json =
            serde_json::to_value(self.pattern.metadata()).unwrap_or(serde_json::json!({}));
        let chunk_embeddings_json: Option<serde_json::Value> = self
            .pattern
            .chunk_embeddings()
            .map(|c| serde_json::to_value(c).unwrap_or(serde_json::json!([])));

        sqlx::query(Self::postgres_update_sql())
            .bind(self.pattern.updated_at())
            .bind(self.pattern.category().to_string())
            .bind(self.pattern.problem())
            .bind(self.pattern.solution())
            .bind(self.pattern.context())
            .bind(self.pattern.effectiveness() as f64)
            .bind(self.pattern.reuse_count() as i32)
            .bind(self.pattern.reward() as f64)
            .bind(self.pattern.success())
            .bind(self.pattern.critique())
            .bind(self.pattern.agent_id())
            .bind(self.pattern.session_id())
            .bind(self.pattern.confidence() as f64)
            .bind(embedding_vec)
            .bind(tags_json)
            .bind(related_json)
            .bind(metadata_json)
            .bind(self.pattern.surprise_score() as f64)
            .bind(self.pattern.failure_mode().map(|m| m.to_string()))
            .bind(chunk_embeddings_json)
            // Software Factory enhancements (Week 1 WS-A)
            .bind(self.pattern.satisfaction_score() as f64)
            .bind(self.pattern.satisfaction_trials() as i32)
            .bind(self.pattern.content_hash())
            // Pyramid summary fields (Week 1 WS-C)
            .bind(self.pattern.title())
            .bind(self.pattern.summary())
            // Bayesian quality score (Beta distribution)
            .bind(self.pattern.bayesian_score().alpha())
            .bind(self.pattern.bayesian_score().beta())
            // Embedding method (hash or onnx)
            .bind(self.pattern.embedding_method())
            .bind(self.pattern.id().to_string())
            .execute(pool)
            .await?;

        Ok(())
    }

    async fn postgres_delete(&self, pool: &PgPool) -> std::result::Result<(), sqlx::Error> {
        sqlx::query("DELETE FROM reasoning_patterns WHERE id = $1")
            .bind(self.pattern.id().to_string())
            .execute(pool)
            .await?;
        Ok(())
    }
}

/// Pattern storage with dual-write capability.
pub struct PatternStorage {
    /// The dual-write adapter for synchronized persistence.
    adapter: Arc<DualWriteAdapter>,

    /// Storage configuration.
    config: StorageConfig,
}

impl PatternStorage {
    /// Create a new PatternStorage.
    pub async fn new(adapter: Arc<DualWriteAdapter>, config: StorageConfig) -> Result<Self> {
        let storage = Self { adapter, config };

        // Initialize the database schema
        storage.init_schema().await?;

        Ok(storage)
    }

    /// Initialize the database schema for pattern storage.
    async fn init_schema(&self) -> Result<()> {
        let create_table_sql = r#"
            CREATE TABLE IF NOT EXISTS reasoning_patterns (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                category TEXT NOT NULL,
                problem TEXT NOT NULL,
                solution TEXT NOT NULL,
                context TEXT DEFAULT '',
                effectiveness REAL DEFAULT 0.5,
                reuse_count INTEGER DEFAULT 0,
                reward REAL DEFAULT 0.5,
                success INTEGER DEFAULT 1,
                critique TEXT DEFAULT '',
                agent_id TEXT,
                session_id TEXT,
                confidence REAL DEFAULT 0.5,
                embedding TEXT,
                tags TEXT DEFAULT '[]',
                related_patterns TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_patterns_category ON reasoning_patterns(category);
            CREATE INDEX IF NOT EXISTS idx_patterns_timestamp ON reasoning_patterns(timestamp);
            CREATE INDEX IF NOT EXISTS idx_patterns_updated_at ON reasoning_patterns(updated_at);
            CREATE INDEX IF NOT EXISTS idx_patterns_effectiveness ON reasoning_patterns(effectiveness);
            CREATE INDEX IF NOT EXISTS idx_patterns_agent_id ON reasoning_patterns(agent_id);
            CREATE INDEX IF NOT EXISTS idx_patterns_session_id ON reasoning_patterns(session_id);
        "#;

        self.adapter.sqlite().execute_batch(create_table_sql).await?;

        // Migrate: add new columns if they don't exist (safe to run multiple times)
        let migrate_sql = r#"
            ALTER TABLE reasoning_patterns ADD COLUMN surprise_score REAL DEFAULT 0.0;
            ALTER TABLE reasoning_patterns ADD COLUMN failure_mode TEXT;
            ALTER TABLE reasoning_patterns ADD COLUMN chunk_embeddings TEXT;
            ALTER TABLE reasoning_patterns ADD COLUMN satisfaction_score REAL DEFAULT 0.5;
            ALTER TABLE reasoning_patterns ADD COLUMN satisfaction_trials INTEGER DEFAULT 0;
            ALTER TABLE reasoning_patterns ADD COLUMN content_hash TEXT;
            ALTER TABLE reasoning_patterns ADD COLUMN title TEXT;
            ALTER TABLE reasoning_patterns ADD COLUMN summary TEXT;
            ALTER TABLE reasoning_patterns ADD COLUMN tier TEXT DEFAULT 'booster';
            ALTER TABLE reasoning_patterns ADD COLUMN quality_alpha REAL DEFAULT 1.0;
            ALTER TABLE reasoning_patterns ADD COLUMN quality_beta REAL DEFAULT 1.0;
            ALTER TABLE reasoning_patterns ADD COLUMN embedding_method TEXT;
        "#;
        // ALTER TABLE ADD COLUMN fails if column exists; ignore errors per-statement
        for stmt in migrate_sql.trim().split(';') {
            let stmt = stmt.trim();
            if !stmt.is_empty() {
                let _ = self.adapter.sqlite().execute(stmt, &[]).await;
            }
        }

        // Strategy cache table (EGUR-inspired)
        let strategy_sql = r#"
            CREATE TABLE IF NOT EXISTS strategy_cache (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                steps TEXT DEFAULT '[]',
                embedding TEXT,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_reward REAL DEFAULT 0.5,
                source_pattern_ids TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_strategy_category ON strategy_cache(category);
            CREATE INDEX IF NOT EXISTS idx_strategy_reward ON strategy_cache(avg_reward);
        "#;
        self.adapter.sqlite().execute_batch(strategy_sql).await?;

        // Create index on content_hash for deduplication lookups
        let _ = self.adapter.sqlite().execute(
            "CREATE INDEX IF NOT EXISTS idx_patterns_content_hash ON reasoning_patterns(content_hash)",
            &[],
        ).await;

        // Sessions table for token tracking (Week 1 WS-A)
        let sessions_sql = r#"
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                ended_at TEXT,
                tokens_used INTEGER DEFAULT 0,
                patterns_learned INTEGER DEFAULT 0,
                patterns_retrieved INTEGER DEFAULT 0,
                domain TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);
            CREATE INDEX IF NOT EXISTS idx_sessions_domain ON sessions(domain);
        "#;
        self.adapter.sqlite().execute_batch(sessions_sql).await?;

        // Pattern usage log table for auto-promotion tracking
        let usage_log_sql = r#"
            CREATE TABLE IF NOT EXISTS pattern_usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT NOT NULL,
                session_id TEXT DEFAULT '',
                task_id TEXT DEFAULT '',
                outcome TEXT NOT NULL DEFAULT 'retrieval',
                used_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (pattern_id) REFERENCES reasoning_patterns(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_usage_log_pattern_id ON pattern_usage_log(pattern_id);
            CREATE INDEX IF NOT EXISTS idx_usage_log_session_id ON pattern_usage_log(session_id);
            CREATE INDEX IF NOT EXISTS idx_usage_log_used_at ON pattern_usage_log(used_at);
        "#;
        self.adapter.sqlite().execute_batch(usage_log_sql).await?;

        info!("Pattern storage schema initialized");
        Ok(())
    }

    /// Store a new pattern.
    ///
    /// The pattern will be written to both SQLite and PostgreSQL.
    /// Returns the pattern ID (either existing or newly generated).
    pub async fn store_pattern(&self, pattern: &Pattern) -> Result<PatternId> {
        // Validate required fields
        self.validate_pattern(pattern)?;

        let storable = StorablePattern::new(pattern.clone());
        let id = pattern.id().clone();

        // Use dual-write adapter for synchronized persistence
        let result = self.adapter.upsert(&storable).await?;

        if result.is_ok() {
            debug!(
                pattern_id = %id,
                sqlite_success = result.sqlite_success,
                postgres_success = ?result.postgres_success,
                "Pattern stored successfully"
            );
        } else {
            warn!(
                pattern_id = %id,
                "Pattern storage failed"
            );
        }

        Ok(id)
    }

    /// Store a pattern with coherence checking.
    ///
    /// This method first checks the pattern against the CoherenceGate.
    /// Based on the coherence result, the pattern may be:
    /// - Stored directly (Accept)
    /// - Stored with warnings (AcceptWithWarning)
    /// - Held for review (RequireReview)
    /// - Rejected (Reject)
    /// - Flagged for merging (Merge)
    ///
    /// When a pattern is successfully stored, its beliefs are also persisted
    /// to the BeliefGraph for future coherence checking.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The pattern to store
    /// * `gate` - The coherence gate to use for checking (mutable for belief persistence)
    ///
    /// # Returns
    ///
    /// A `StoreWithCoherenceResult` indicating what happened to the pattern.
    pub async fn store_with_coherence_check(
        &self,
        pattern: &Pattern,
        gate: &mut CoherenceGate,
    ) -> Result<StoreWithCoherenceResult> {
        // Validate required fields first
        self.validate_pattern(pattern)?;

        let id = pattern.id().to_string();

        // Check coherence and get extracted beliefs
        let (coherence, beliefs) = gate.check_with_beliefs(
            &id,
            pattern.problem(),
            pattern.solution(),
            &pattern.category().to_string(),
        ).await?;

        info!(
            pattern_id = %id,
            coherent = coherence.is_coherent,
            energy = coherence.energy,
            conflicts = coherence.conflicts.len(),
            beliefs_extracted = beliefs.len(),
            "Coherence check completed"
        );

        // Act based on recommendation
        match coherence.recommendation {
            CoherenceAction::Accept => {
                self.store_pattern(pattern).await?;
                // Persist beliefs to BeliefGraph
                let beliefs_persisted = gate.persist_beliefs_for_pattern(&id, beliefs).await?;
                info!(
                    pattern_id = %id,
                    beliefs_persisted = beliefs_persisted,
                    "Pattern and beliefs stored"
                );
                Ok(StoreWithCoherenceResult::Stored { pattern_id: id })
            }
            CoherenceAction::AcceptWithWarning { warnings } => {
                self.store_pattern(pattern).await?;
                // Persist beliefs to BeliefGraph
                let beliefs_persisted = gate.persist_beliefs_for_pattern(&id, beliefs).await?;
                info!(
                    pattern_id = %id,
                    warning_count = warnings.len(),
                    beliefs_persisted = beliefs_persisted,
                    "Pattern stored with coherence warnings"
                );
                Ok(StoreWithCoherenceResult::StoredWithWarnings {
                    pattern_id: id,
                    warnings,
                })
            }
            CoherenceAction::RequireReview { conflicts } => {
                // Don't store the pattern - it needs manual review
                warn!(
                    pattern_id = %id,
                    conflict_count = conflicts.len(),
                    "Pattern held for review due to coherence conflicts"
                );
                Ok(StoreWithCoherenceResult::PendingReview {
                    pattern_id: id,
                    conflicts,
                })
            }
            CoherenceAction::Reject { reason } => {
                warn!(
                    pattern_id = %id,
                    reason = %reason,
                    "Pattern rejected by coherence gate"
                );
                Ok(StoreWithCoherenceResult::Rejected { reason })
            }
            CoherenceAction::Merge { merge_with } => {
                // Don't store the pattern - it should be merged with existing
                info!(
                    pattern_id = %id,
                    merge_target = %merge_with,
                    "Pattern should be merged with existing pattern"
                );
                Ok(StoreWithCoherenceResult::MergeRequired {
                    pattern_id: id,
                    merge_with,
                })
            }
        }
    }

    /// Store a pattern, forcing storage even if coherence check would reject.
    ///
    /// Use this method when you need to store a pattern regardless of
    /// coherence issues (e.g., user explicitly confirmed override).
    /// Beliefs are still persisted to the BeliefGraph for future checks.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The pattern to store
    /// * `gate` - The coherence gate to use for checking (mutable for belief persistence)
    ///
    /// # Returns
    ///
    /// A `StoreWithCoherenceResult` with the stored pattern ID and any warnings.
    pub async fn store_with_coherence_override(
        &self,
        pattern: &Pattern,
        gate: &mut CoherenceGate,
    ) -> Result<StoreWithCoherenceResult> {
        // Validate required fields first
        self.validate_pattern(pattern)?;

        let id = pattern.id().to_string();

        // Check coherence but don't enforce - also get beliefs for persistence
        let (coherence, beliefs) = gate.check_with_beliefs(
            &id,
            pattern.problem(),
            pattern.solution(),
            &pattern.category().to_string(),
        ).await?;

        // Always store the pattern
        self.store_pattern(pattern).await?;

        // Always persist beliefs (even if coherence check would have rejected)
        let beliefs_persisted = gate.persist_beliefs_for_pattern(&id, beliefs).await?;

        if coherence.is_coherent {
            info!(
                pattern_id = %id,
                beliefs_persisted = beliefs_persisted,
                "Pattern and beliefs stored"
            );
            Ok(StoreWithCoherenceResult::Stored { pattern_id: id })
        } else {
            let warnings: Vec<String> = coherence.conflicts.iter()
                .map(|c| c.description.clone())
                .collect();

            warn!(
                pattern_id = %id,
                warning_count = warnings.len(),
                beliefs_persisted = beliefs_persisted,
                "Pattern stored with coherence override"
            );

            Ok(StoreWithCoherenceResult::StoredWithWarnings {
                pattern_id: id,
                warnings,
            })
        }
    }

    /// Validate a pattern before storage.
    fn validate_pattern(&self, pattern: &Pattern) -> Result<()> {
        // Problem and solution are required
        if pattern.problem().is_empty() {
            return Err(NagualError::Config {
                message: "Pattern problem cannot be empty".to_string(),
            });
        }

        if pattern.solution().is_empty() {
            return Err(NagualError::Config {
                message: "Pattern solution cannot be empty".to_string(),
            });
        }

        // Validate embedding dimension if present
        if let Some(embedding) = pattern.embedding() {
            if embedding.len() != self.config.embedding_dim {
                return Err(NagualError::Config {
                    message: format!(
                        "Embedding dimension mismatch: expected {}, got {}",
                        self.config.embedding_dim,
                        embedding.len()
                    ),
                });
            }
        }

        // Validate tag count
        if pattern.tags().len() > self.config.max_tags {
            return Err(NagualError::Config {
                message: format!(
                    "Too many tags: {} exceeds maximum of {}",
                    pattern.tags().len(),
                    self.config.max_tags
                ),
            });
        }

        // Validate related patterns count
        if pattern.related_patterns().len() > self.config.max_related_patterns {
            return Err(NagualError::Config {
                message: format!(
                    "Too many related patterns: {} exceeds maximum of {}",
                    pattern.related_patterns().len(),
                    self.config.max_related_patterns
                ),
            });
        }

        Ok(())
    }

    /// Get a pattern by ID.
    pub async fn get_pattern(&self, id: &PatternId) -> Result<Option<Pattern>> {
        let sql = "SELECT * FROM reasoning_patterns WHERE id = ?";
        let id_str = id.to_string();

        // Use with_connection to keep &dyn ToSql refs inside
        // the synchronous closure (they must not live across .await).
        let pattern = self
            .adapter
            .sqlite()
            .with_connection(|conn| {
                let mut stmt = conn.prepare(sql).map_err(crate::error::DatabaseError::from)?;
                let mut rows = stmt
                    .query(rusqlite::params![id_str])
                    .map_err(crate::error::DatabaseError::from)?;
                match rows.next().map_err(crate::error::DatabaseError::from)? {
                    Some(row) => {
                        let p = Self::pattern_from_row(row)
                            .map_err(crate::error::DatabaseError::from)?;
                        Ok(Some(p))
                    }
                    None => Ok(None),
                }
            })
            .await?;

        Ok(pattern)
    }

    /// Update an existing pattern.
    pub async fn update_pattern(&self, pattern: &Pattern) -> Result<()> {
        self.validate_pattern(pattern)?;

        let storable = StorablePattern::new(pattern.clone());
        self.adapter.update(&storable).await?;

        debug!(pattern_id = %pattern.id(), "Pattern updated");
        Ok(())
    }

    /// Delete a pattern by ID.
    pub async fn delete_pattern(&self, id: &PatternId) -> Result<()> {
        // Create a minimal pattern for deletion
        let pattern = Pattern::builder()
            .id(id.clone())
            .problem("deleted")
            .solution("deleted")
            .build();

        let storable = StorablePattern::new(pattern);
        self.adapter.delete(&storable).await?;

        info!(pattern_id = %id, "Pattern deleted");
        Ok(())
    }

    /// Increment the reuse count for a pattern.
    pub async fn increment_reuse_count(&self, id: &PatternId) -> Result<()> {
        let sql = "UPDATE reasoning_patterns SET reuse_count = reuse_count + 1, updated_at = ? WHERE id = ?";
        let now = Utc::now().to_rfc3339();
        let id_str = id.to_string();

        self.adapter.sqlite().execute(sql, &[&now, &id_str]).await?;

        debug!(pattern_id = %id, "Pattern reuse count incremented");
        Ok(())
    }

    /// Get patterns by category.
    pub async fn get_by_category(
        &self,
        category: &PatternCategory,
        limit: usize,
    ) -> Result<Vec<Pattern>> {
        let sql = r#"
            SELECT * FROM reasoning_patterns
            WHERE category = ?
            ORDER BY effectiveness DESC, updated_at DESC
            LIMIT ?
        "#;

        let cat_str = category.to_string();
        let limit_i64 = limit as i64;

        let patterns = self
            .adapter
            .sqlite()
            .query(sql, &[&cat_str, &limit_i64], |row| {
                Self::pattern_from_row(row)
            })
            .await?;

        Ok(patterns)
    }

    /// Get recently added patterns.
    pub async fn get_recent(&self, limit: usize) -> Result<Vec<Pattern>> {
        let sql = r#"
            SELECT * FROM reasoning_patterns
            ORDER BY timestamp DESC
            LIMIT ?
        "#;

        let limit_i64 = limit as i64;

        let patterns = self
            .adapter
            .sqlite()
            .query(sql, &[&limit_i64], |row| {
                Self::pattern_from_row(row)
            })
            .await?;

        Ok(patterns)
    }

    /// Get patterns with highest effectiveness.
    pub async fn get_top_effective(&self, limit: usize) -> Result<Vec<Pattern>> {
        let sql = r#"
            SELECT * FROM reasoning_patterns
            ORDER BY effectiveness DESC
            LIMIT ?
        "#;

        let limit_i64 = limit as i64;

        let patterns = self
            .adapter
            .sqlite()
            .query(sql, &[&limit_i64], |row| {
                Self::pattern_from_row(row)
            })
            .await?;

        Ok(patterns)
    }

    /// Get all patterns with embeddings (for building HNSW index).
    pub async fn get_all_with_embeddings(&self) -> Result<Vec<Pattern>> {
        let sql = r#"
            SELECT * FROM reasoning_patterns
            WHERE embedding IS NOT NULL AND embedding != ''
        "#;

        let patterns = self
            .adapter
            .sqlite()
            .query(sql, &[], |row| {
                Self::pattern_from_row(row)
            })
            .await?;

        Ok(patterns)
    }

    /// Search patterns using FTS5 full-text search.
    ///
    /// This method uses SQLite FTS5 for efficient O(log n) text search with BM25 ranking,
    /// instead of loading all patterns and filtering with O(n) string matching.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query text
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of patterns matching the search query, ranked by relevance (BM25).
    pub async fn fts_search(&self, query: &str, limit: usize) -> Result<Vec<Pattern>> {
        // FTS5 table name matches the content table name with _fts suffix
        let fts_table = "reasoning_patterns_fts";
        let content_table = "reasoning_patterns";

        let options = FtsSearchOptions {
            limit,
            offset: 0,
            with_snippets: false,
            snippet_tokens: 10,
            snippet_markers: ("<mark>".to_string(), "</mark>".to_string()),
            search_columns: vec!["problem".to_string(), "solution".to_string(), "category".to_string()],
            min_rank: None,
        };

        // Clone query for use in closures
        let query_owned = query.to_string();

        // Use with_connection to access the raw SQLite connection for FTS operations
        let fts_results: Vec<(i64, f64, Option<String>)> = self
            .adapter
            .sqlite()
            .with_connection(|conn| {
                // First check if FTS table exists, if not, create it
                let fts_exists: bool = conn
                    .query_row(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                        [fts_table],
                        |row| row.get::<_, i64>(0),
                    )
                    .map(|count| count > 0)
                    .unwrap_or(false);

                if !fts_exists {
                    // Initialize FTS5 for patterns table
                    let pattern_fts = PatternFts::with_config(
                        crate::db::fts::Fts5Config::new(
                            content_table,
                            vec!["problem".to_string(), "solution".to_string(), "category".to_string()],
                        )
                        .with_fts_table(fts_table)
                        .with_content_rowid("rowid"),
                    );
                    pattern_fts.create_fts_table(conn).map_err(|e| {
                        DatabaseError::Sqlite(rusqlite::Error::ToSqlConversionFailure(
                            Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())),
                        ))
                    })?;
                    // Rebuild to index existing data
                    pattern_fts.rebuild_index(conn).map_err(|e| {
                        DatabaseError::Sqlite(rusqlite::Error::ToSqlConversionFailure(
                            Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())),
                        ))
                    })?;
                }

                // Call fts_search and convert error
                fts_search(conn, fts_table, &query_owned, &options).map_err(|e| {
                    DatabaseError::Sqlite(rusqlite::Error::ToSqlConversionFailure(
                        Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())),
                    ))
                })
            })
            .await?;

        if fts_results.is_empty() {
            return Ok(Vec::new());
        }

        // Get the rowids from FTS results (preserve order for ranking)
        let rowids: Vec<i64> = fts_results.iter().map(|(rowid, _, _)| *rowid).collect();

        // Build query to fetch full patterns by rowid
        let placeholders: Vec<String> = rowids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "SELECT rowid, * FROM {} WHERE rowid IN ({})",
            content_table,
            placeholders.join(", ")
        );

        // Create a map of rowid -> rank for sorting (lower BM25 score is better)
        let rank_map: std::collections::HashMap<i64, f64> = fts_results
            .iter()
            .map(|(rowid, rank, _)| (*rowid, *rank))
            .collect();

        // Fetch patterns with their rowids
        let patterns_with_rowids: Vec<(i64, Pattern)> = self
            .adapter
            .sqlite()
            .with_connection(move |conn| {
                let mut stmt = conn.prepare(&sql).map_err(DatabaseError::from)?;

                // Create params for the query
                let params: Vec<&dyn rusqlite::ToSql> = rowids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();

                let pattern_iter = stmt
                    .query_map(params.as_slice(), |row| {
                        let rowid: i64 = row.get(0)?;  // First column is rowid
                        let pattern = Self::pattern_from_row_with_offset(row, 1)?;
                        Ok((rowid, pattern))
                    })
                    .map_err(DatabaseError::from)?;

                let mut result = Vec::new();
                for item in pattern_iter {
                    result.push(item.map_err(DatabaseError::from)?);
                }
                Ok(result)
            })
            .await?;

        // Sort by BM25 rank (lower is better)
        let mut sorted: Vec<(i64, Pattern)> = patterns_with_rowids;
        sorted.sort_by(|a, b| {
            let rank_a = rank_map.get(&a.0).unwrap_or(&f64::MAX);
            let rank_b = rank_map.get(&b.0).unwrap_or(&f64::MAX);
            rank_a.partial_cmp(rank_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        let patterns: Vec<Pattern> = sorted.into_iter().map(|(_, p)| p).collect();

        debug!(
            query = %query,
            results = patterns.len(),
            "FTS pattern search completed"
        );

        Ok(patterns)
    }

    /// Convert a database row to a Pattern with column offset.
    /// Used when the query includes additional columns before the pattern data.
    fn pattern_from_row_with_offset(row: &rusqlite::Row<'_>, _offset: usize) -> rusqlite::Result<Pattern> {
        // Use named columns to avoid offset issues
        Self::pattern_from_row(row)
    }

    /// Get pattern count.
    pub async fn count(&self) -> Result<usize> {
        let sql = "SELECT COUNT(*) FROM reasoning_patterns";

        let count = self
            .adapter
            .sqlite()
            .query_one(sql, &[], |row| row.get::<_, i64>(0))
            .await?
            .unwrap_or(0);

        Ok(count as usize)
    }

    /// Convert a database row to a Pattern.
    fn pattern_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Pattern> {
        let id: String = row.get("id")?;
        let timestamp_str: String = row.get("timestamp")?;
        let updated_at_str: String = row.get("updated_at")?;
        let category_str: String = row.get("category")?;
        let problem: String = row.get("problem")?;
        let solution: String = row.get("solution")?;
        let context: String = row.get::<_, Option<String>>("context")?.unwrap_or_default();
        let effectiveness: f64 = row.get("effectiveness")?;
        let reuse_count: i32 = row.get("reuse_count")?;
        let reward: f64 = row.get("reward")?;
        let success: bool = row.get::<_, i32>("success")? != 0;
        let critique: String = row.get::<_, Option<String>>("critique")?.unwrap_or_default();
        let agent_id: Option<String> = row.get("agent_id")?;
        let session_id: Option<String> = row.get("session_id")?;
        let confidence: f64 = row.get("confidence")?;
        let embedding_json: Option<String> = row.get("embedding")?;
        let tags_json: String = row.get::<_, Option<String>>("tags")?.unwrap_or_else(|| "[]".to_string());
        let related_json: String = row.get::<_, Option<String>>("related_patterns")?.unwrap_or_else(|| "[]".to_string());
        let metadata_json: String = row.get::<_, Option<String>>("metadata")?.unwrap_or_else(|| "{}".to_string());

        // New columns (safe to fail if migration hasn't run yet)
        let surprise_score: f64 = row.get::<_, Option<f64>>("surprise_score")?.unwrap_or(0.0);
        let failure_mode_str: Option<String> = row.get::<_, Option<String>>("failure_mode")?;
        let chunk_embeddings_json: Option<String> = row.get::<_, Option<String>>("chunk_embeddings")?;
        // Software Factory enhancements (Week 1 WS-A)
        let satisfaction_score: f64 = row.get::<_, Option<f64>>("satisfaction_score")?.unwrap_or(0.5);
        let satisfaction_trials: i32 = row.get::<_, Option<i32>>("satisfaction_trials")?.unwrap_or(0);
        let content_hash: Option<String> = row.get::<_, Option<String>>("content_hash")?;
        // Pyramid summary fields (Week 1 WS-C)
        let title: Option<String> = row.get::<_, Option<String>>("title")?;
        let summary: Option<String> = row.get::<_, Option<String>>("summary")?;
        // Bayesian quality score (Beta distribution)
        let quality_alpha: f64 = row.get::<_, Option<f64>>("quality_alpha")?.unwrap_or(1.0);
        let quality_beta: f64 = row.get::<_, Option<f64>>("quality_beta")?.unwrap_or(1.0);
        // Embedding method (hash or onnx)
        let embedding_method: Option<String> = row.get::<_, Option<String>>("embedding_method").unwrap_or(None);

        // Parse timestamps
        let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        let _updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        // Parse embedding
        let embedding: Option<Vec<f32>> = embedding_json
            .and_then(|json| serde_json::from_str(&json).ok());

        // Parse chunk embeddings
        let chunk_embeddings: Option<Vec<Vec<f32>>> = chunk_embeddings_json
            .and_then(|json| serde_json::from_str(&json).ok());

        // Parse failure mode
        let failure_mode: Option<FailureMode> = failure_mode_str.map(|s| FailureMode::from(s.as_str()));

        // Parse tags
        let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();

        // Parse related patterns
        let related_patterns: Vec<PatternId> = serde_json::from_str(&related_json).unwrap_or_default();

        // Parse metadata
        let metadata: PatternMetadata = serde_json::from_str(&metadata_json).unwrap_or_default();

        // Build pattern
        let mut builder = Pattern::builder()
            .id(id)
            .timestamp(timestamp)
            .category(PatternCategory::from(category_str.as_str()))
            .problem(problem)
            .solution(solution)
            .context(context)
            .effectiveness(effectiveness as f32)
            .reuse_count(reuse_count as u32)
            .reward(reward as f32)
            .success(success)
            .critique(critique)
            .confidence(confidence as f32)
            .surprise_score(surprise_score as f32)
            .satisfaction_score(satisfaction_score as f32)
            .satisfaction_trials(satisfaction_trials as u32)
            .bayesian_score(BetaParams::with_params(quality_alpha, quality_beta))
            .tags(tags)
            .related_patterns(related_patterns)
            .metadata(metadata);

        if let Some(agent) = agent_id {
            builder = builder.agent_id(agent);
        }

        if let Some(session) = session_id {
            builder = builder.session_id(session);
        }

        if let Some(emb) = embedding {
            builder = builder.embedding(emb);
        }

        if let Some(mode) = failure_mode {
            builder = builder.failure_mode(mode);
        }

        if let Some(chunks) = chunk_embeddings {
            builder = builder.chunk_embeddings(chunks);
        }

        if let Some(hash) = content_hash {
            builder = builder.content_hash(hash);
        }

        if let Some(t) = title {
            builder = builder.title(t);
        }

        if let Some(s) = summary {
            builder = builder.summary(s);
        }

        if let Some(method) = embedding_method {
            builder = builder.embedding_method(method);
        }

        Ok(builder.build())
    }

    /// Get the storage configuration.
    pub fn config(&self) -> &StorageConfig {
        &self.config
    }

    /// Get the underlying adapter.
    pub fn adapter(&self) -> &Arc<DualWriteAdapter> {
        &self.adapter
    }

    /// Record that a pattern was used in the current session/task.
    ///
    /// Inserts a row into `pattern_usage_log` to track retrieval or success
    /// events. This data feeds the auto-promotion engine which promotes
    /// patterns that meet recurrence thresholds across distinct sessions.
    pub async fn record_pattern_usage(
        &self,
        pattern_id: &str,
        session_id: Option<&str>,
        task_id: Option<&str>,
        outcome: &str,
    ) -> Result<()> {
        let sql = r#"
            INSERT INTO pattern_usage_log (pattern_id, session_id, task_id, outcome)
            VALUES (?, ?, ?, ?)
        "#;
        let sid = session_id.unwrap_or("").to_string();
        let tid = task_id.unwrap_or("").to_string();
        let outcome = outcome.to_string();
        let pid = pattern_id.to_string();

        self.adapter
            .sqlite()
            .execute(sql, &[&pid, &sid, &tid, &outcome])
            .await?;

        debug!(pattern_id = %pid, "Pattern usage recorded");
        Ok(())
    }

    /// Count distinct sessions/tasks where a pattern was used within a time window.
    ///
    /// Returns `(total_usage_count, distinct_session_count)` within the window.
    /// Used by the auto-promotion engine to determine if a pattern meets
    /// recurrence thresholds for tier promotion.
    pub async fn count_pattern_usage_contexts(
        &self,
        pattern_id: &str,
        window_days: u32,
    ) -> Result<(u32, u32)> {
        let sql = r#"
            SELECT
                COUNT(*) as total_uses,
                COUNT(DISTINCT CASE WHEN session_id != '' THEN session_id END) as distinct_sessions
            FROM pattern_usage_log
            WHERE pattern_id = ?
            AND used_at >= datetime('now', ?)
        "#;

        let pid = pattern_id.to_string();
        let window = format!("-{} days", window_days);

        let result = self
            .adapter
            .sqlite()
            .query_one(sql, &[&pid, &window], |row| {
                let total: i64 = row.get(0)?;
                let distinct: i64 = row.get(1)?;
                Ok((total as u32, distinct as u32))
            })
            .await?
            .unwrap_or((0, 0));

        Ok(result)
    }

    /// Update the tier of a pattern by ID.
    ///
    /// Used by the auto-promotion engine to promote patterns:
    /// booster -> crystal -> reflex.
    pub async fn update_pattern_tier(&self, pattern_id: &str, new_tier: &str) -> Result<()> {
        let sql = "UPDATE reasoning_patterns SET tier = ?, updated_at = ? WHERE id = ?";
        let now = chrono::Utc::now().to_rfc3339();
        let pid = pattern_id.to_string();
        let tier = new_tier.to_string();

        self.adapter
            .sqlite()
            .execute(sql, &[&tier, &now, &pid])
            .await?;

        debug!(pattern_id = %pid, new_tier = %tier, "Pattern tier updated");
        Ok(())
    }

    /// List patterns by their tier.
    ///
    /// Returns patterns whose tier matches one of the provided values.
    /// Used by the auto-promotion engine to find eligible patterns.
    pub async fn list_patterns_by_tier(
        &self,
        tiers: &[&str],
        limit: usize,
    ) -> Result<Vec<Pattern>> {
        if tiers.is_empty() {
            return Ok(Vec::new());
        }

        // Build IN clause with placeholders
        let placeholders: Vec<&str> = tiers.iter().map(|_| "?").collect();
        let sql = format!(
            "SELECT * FROM reasoning_patterns WHERE COALESCE(tier, 'booster') IN ({}) LIMIT {}",
            placeholders.join(", "),
            limit
        );

        let tier_strings: Vec<String> = tiers.iter().map(|t| t.to_string()).collect();
        let params: Vec<&dyn rusqlite::types::ToSql> =
            tier_strings.iter().map(|s| s as &dyn rusqlite::types::ToSql).collect();

        self.adapter
            .sqlite()
            .query(&sql, &params, Self::pattern_from_row)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert_eq!(config.embedding_dim, 128);
        assert_eq!(config.table_name, "reasoning_patterns");
        assert!(!config.auto_generate_embeddings);
    }

    #[tokio::test]
    async fn test_storable_pattern() {
        let pattern = Pattern::builder()
            .problem("Test problem")
            .solution("Test solution")
            .category(PatternCategory::Testing)
            .build();

        let storable = StorablePattern::new(pattern.clone());

        assert_eq!(storable.id(), pattern.id().to_string());
        assert_eq!(storable.pattern().problem(), "Test problem");
    }

    #[tokio::test]
    async fn test_storage_new() {
        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let storage = PatternStorage::new(adapter, StorageConfig::default()).await.unwrap();

        assert_eq!(storage.config().embedding_dim, 128);
    }

    #[tokio::test]
    async fn test_store_and_retrieve_pattern() {
        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let storage = PatternStorage::new(adapter, StorageConfig::default()).await.unwrap();

        let pattern = Pattern::builder()
            .problem("How to cache data efficiently")
            .solution("Use LRU cache with TTL expiration")
            .category(PatternCategory::Performance)
            .effectiveness(0.9)
            .tag("caching")
            .build();

        // Store the pattern
        let id = storage.store_pattern(&pattern).await.unwrap();

        // Retrieve it
        let retrieved = storage.get_pattern(&id).await.unwrap();
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.problem(), "How to cache data efficiently");
        assert_eq!(retrieved.category(), &PatternCategory::Performance);
    }

    #[tokio::test]
    async fn test_validation_empty_problem() {
        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let storage = PatternStorage::new(adapter, StorageConfig::default()).await.unwrap();

        let pattern = Pattern::builder()
            .problem("")
            .solution("Some solution")
            .build();

        let result = storage.store_pattern(&pattern).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validation_wrong_embedding_dim() {
        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let storage = PatternStorage::new(adapter, StorageConfig::default()).await.unwrap();

        let pattern = Pattern::builder()
            .problem("Test")
            .solution("Solution")
            .embedding(vec![0.1, 0.2]) // Wrong dimension (2 instead of 128)
            .build();

        let result = storage.store_pattern(&pattern).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_by_category() {
        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let storage = PatternStorage::new(adapter, StorageConfig::default()).await.unwrap();

        // Store patterns in different categories
        let perf_pattern = Pattern::builder()
            .problem("Slow queries")
            .solution("Add index")
            .category(PatternCategory::Performance)
            .build();

        let sec_pattern = Pattern::builder()
            .problem("SQL injection")
            .solution("Use prepared statements")
            .category(PatternCategory::Security)
            .build();

        storage.store_pattern(&perf_pattern).await.unwrap();
        storage.store_pattern(&sec_pattern).await.unwrap();

        // Query by category
        let perf_patterns = storage.get_by_category(&PatternCategory::Performance, 10).await.unwrap();
        assert_eq!(perf_patterns.len(), 1);
        assert_eq!(perf_patterns[0].problem(), "Slow queries");
    }

    #[tokio::test]
    async fn test_increment_reuse_count() {
        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let storage = PatternStorage::new(adapter, StorageConfig::default()).await.unwrap();

        let pattern = Pattern::builder()
            .problem("Test")
            .solution("Solution")
            .build();

        let id = storage.store_pattern(&pattern).await.unwrap();

        // Increment reuse count
        storage.increment_reuse_count(&id).await.unwrap();
        storage.increment_reuse_count(&id).await.unwrap();

        // Verify
        let retrieved = storage.get_pattern(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.reuse_count(), 2);
    }

    #[tokio::test]
    async fn test_count_patterns() {
        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let storage = PatternStorage::new(adapter, StorageConfig::default()).await.unwrap();

        assert_eq!(storage.count().await.unwrap(), 0);

        storage.store_pattern(&Pattern::new("P1", "S1")).await.unwrap();
        storage.store_pattern(&Pattern::new("P2", "S2")).await.unwrap();

        assert_eq!(storage.count().await.unwrap(), 2);
    }
}
