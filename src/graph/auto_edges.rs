//! Auto Edge Creation for Pattern Graphs.
//!
//! Implements automatic creation of edges between patterns based on:
//! - **Similarity**: Creates `similar_to` edges when pattern similarity > 0.85
//! - **Co-Retrieval**: Creates `co_retrieved` edges when patterns are retrieved together > 3 times
//!
//! # Design
//!
//! The `AutoEdgeCreator` is the main entry point for edge creation. It provides:
//! - `create_similar_edges()`: Called when a new pattern is stored
//! - `record_co_retrieval()`: Called when patterns are retrieved together
//! - `check_and_create_coretrieval_edges()`: Periodic check for new co-retrieval edges
//!
//! # Thresholds
//!
//! - **Similarity Threshold**: 0.85 (cosine similarity)
//! - **Co-Retrieval Threshold**: 3 times (patterns retrieved together)
//! - **Max Similar Edges**: 5 per new pattern

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPool;
use sqlx::{FromRow, Row};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::error::{NagualError, Result};

/// Edge type enum matching the database enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternEdgeType {
    /// Similarity-based edge (cosine > 0.85)
    SimilarTo,
    /// Co-retrieval based edge
    CoRetrieved,
    /// Pattern derived from another
    DerivedFrom,
    /// Manual or inferred relation
    RelatedTo,
}

impl PatternEdgeType {
    /// Convert to database string representation.
    pub fn as_db_str(&self) -> &'static str {
        match self {
            PatternEdgeType::SimilarTo => "similar_to",
            PatternEdgeType::CoRetrieved => "co_retrieved",
            PatternEdgeType::DerivedFrom => "derived_from",
            PatternEdgeType::RelatedTo => "related_to",
        }
    }

    /// Parse from database string representation.
    pub fn from_db_str(s: &str) -> Option<Self> {
        match s {
            "similar_to" => Some(PatternEdgeType::SimilarTo),
            "co_retrieved" => Some(PatternEdgeType::CoRetrieved),
            "derived_from" => Some(PatternEdgeType::DerivedFrom),
            "related_to" => Some(PatternEdgeType::RelatedTo),
            _ => None,
        }
    }
}

/// Reason for edge creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeCreationReason {
    /// Edge created due to high similarity score.
    Similarity { score: f64 },
    /// Edge created due to co-retrieval frequency.
    CoRetrieval { count: i32 },
    /// Manual edge creation.
    Manual { reason: String },
}

impl std::fmt::Display for EdgeCreationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdgeCreationReason::Similarity { score } => {
                write!(f, "Similarity score: {:.4}", score)
            }
            EdgeCreationReason::CoRetrieval { count } => {
                write!(f, "Co-retrieved {} times", count)
            }
            EdgeCreationReason::Manual { reason } => {
                write!(f, "Manual: {}", reason)
            }
        }
    }
}

/// A pattern edge representing a relationship between two patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEdge {
    /// Unique edge identifier.
    pub id: Uuid,
    /// Source pattern ID.
    pub source_pattern: String,
    /// Target pattern ID.
    pub target_pattern: String,
    /// Edge type.
    pub edge_type: PatternEdgeType,
    /// Edge strength (0.0 to 1.0).
    pub strength: f64,
    /// Confidence in the edge (0.0 to 1.0).
    pub confidence: f64,
    /// Whether edge is bidirectional.
    pub bidirectional: bool,
    /// Whether edge was auto-created.
    pub auto_created: bool,
    /// Reason for edge creation.
    pub creation_reason: Option<String>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last update timestamp.
    pub updated_at: DateTime<Utc>,
}

/// Co-retrieval record for tracking pattern pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoRetrievalRecord {
    /// First pattern ID.
    pub pattern_a: String,
    /// Second pattern ID.
    pub pattern_b: String,
    /// Number of times retrieved together.
    pub count: i32,
    /// First retrieval timestamp.
    pub first_retrieved: DateTime<Utc>,
    /// Last retrieval timestamp.
    pub last_retrieved: DateTime<Utc>,
}

/// Candidate for co-retrieval edge creation.
#[derive(Debug, Clone, FromRow)]
pub struct CoRetrievalCandidate {
    /// First pattern ID.
    pub pattern_a: String,
    /// Second pattern ID.
    pub pattern_b: String,
    /// Co-retrieval count.
    pub count: i32,
    /// Last retrieval timestamp.
    pub last_retrieved: DateTime<Utc>,
}

/// Configuration for auto edge creation.
#[derive(Debug, Clone)]
pub struct AutoEdgeConfig {
    /// Minimum similarity threshold for similar_to edges.
    pub similarity_threshold: f64,
    /// Minimum co-retrieval count for co_retrieved edges.
    pub co_retrieval_threshold: i32,
    /// Maximum number of similar edges per new pattern.
    pub max_similar_edges: usize,
    /// Base confidence for auto-created edges.
    pub base_confidence: f64,
    /// Whether to create bidirectional similarity edges.
    pub bidirectional_similarity: bool,
}

impl Default for AutoEdgeConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            co_retrieval_threshold: 3,
            max_similar_edges: 5,
            base_confidence: 0.8,
            bidirectional_similarity: true,
        }
    }
}

impl AutoEdgeConfig {
    /// Create config with custom similarity threshold.
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Create config with custom co-retrieval threshold.
    pub fn with_co_retrieval_threshold(mut self, threshold: i32) -> Self {
        self.co_retrieval_threshold = threshold.max(1);
        self
    }

    /// Create config with custom max similar edges.
    pub fn with_max_similar_edges(mut self, max: usize) -> Self {
        self.max_similar_edges = max.max(1);
        self
    }
}

/// Result of auto edge creation operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoEdgeResult {
    /// Number of edges created.
    pub edges_created: usize,
    /// Number of edges that already existed.
    pub edges_existing: usize,
    /// Number of errors during creation.
    pub errors: usize,
    /// IDs of created edges.
    pub created_edge_ids: Vec<Uuid>,
    /// Time taken in milliseconds.
    pub duration_ms: u64,
}

impl AutoEdgeResult {
    /// Create an empty result.
    pub fn empty() -> Self {
        Self {
            edges_created: 0,
            edges_existing: 0,
            errors: 0,
            created_edge_ids: Vec::new(),
            duration_ms: 0,
        }
    }

    /// Check if any edges were created.
    pub fn has_changes(&self) -> bool {
        self.edges_created > 0
    }
}

/// Auto edge creator for managing pattern relationships.
///
/// This struct provides methods to automatically create edges between patterns
/// based on similarity and co-retrieval frequency.
pub struct AutoEdgeCreator {
    pool: PgPool,
    config: AutoEdgeConfig,
}

impl AutoEdgeCreator {
    /// Create a new auto edge creator with default configuration.
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            config: AutoEdgeConfig::default(),
        }
    }

    /// Create a new auto edge creator with custom configuration.
    pub fn with_config(pool: PgPool, config: AutoEdgeConfig) -> Self {
        Self { pool, config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &AutoEdgeConfig {
        &self.config
    }

    /// Create similar_to edges for a new pattern based on embedding similarity.
    ///
    /// Finds patterns with similarity > threshold and creates edges with
    /// strength = similarity score. Limited to top N similar patterns.
    ///
    /// # Arguments
    ///
    /// * `pattern_id` - ID of the new pattern
    /// * `embedding` - 128-dimensional embedding vector
    ///
    /// # Returns
    ///
    /// Result containing number of edges created and their IDs.
    pub async fn create_similar_edges(
        &self,
        pattern_id: &str,
        embedding: &[f32],
    ) -> Result<AutoEdgeResult> {
        let start = std::time::Instant::now();

        // Validate embedding dimension
        if embedding.len() != 128 {
            return Err(NagualError::internal(format!(
                "Invalid embedding dimension: expected 128, got {}",
                embedding.len()
            )));
        }

        // Convert embedding to string format for pgvector
        let embedding_str = format!(
            "[{}]",
            embedding
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        // Find similar patterns using cosine similarity
        // Note: pgvector uses <=> for cosine distance (1 - cosine_similarity)
        // So we need similarity > 0.85, which means distance < 0.15
        let distance_threshold = 1.0 - self.config.similarity_threshold;
        let limit = self.config.max_similar_edges as i64;

        let similar_patterns: Vec<(String, f64)> = sqlx::query_as(
            r#"
            SELECT
                p.id,
                1 - (p.embedding <=> $1::vector) as similarity
            FROM patterns p
            WHERE p.id != $2
                AND p.embedding IS NOT NULL
                AND (p.embedding <=> $1::vector) < $3
            ORDER BY p.embedding <=> $1::vector ASC
            LIMIT $4
            "#,
        )
        .bind(&embedding_str)
        .bind(pattern_id)
        .bind(distance_threshold)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| NagualError::internal(format!("Failed to find similar patterns: {}", e)))?;

        if similar_patterns.is_empty() {
            debug!(
                pattern_id = pattern_id,
                "No similar patterns found above threshold {}",
                self.config.similarity_threshold
            );
            return Ok(AutoEdgeResult {
                duration_ms: start.elapsed().as_millis() as u64,
                ..AutoEdgeResult::empty()
            });
        }

        let mut result = AutoEdgeResult::empty();

        // Create edges for each similar pattern
        for (similar_id, similarity) in similar_patterns {
            let edge_id = Uuid::new_v4();
            let creation_reason = EdgeCreationReason::Similarity { score: similarity };

            // Insert edge (source -> target)
            let inserted = self
                .insert_edge(
                    edge_id,
                    pattern_id,
                    &similar_id,
                    PatternEdgeType::SimilarTo,
                    similarity,
                    &creation_reason,
                )
                .await;

            match inserted {
                Ok(true) => {
                    result.edges_created += 1;
                    result.created_edge_ids.push(edge_id);

                    // If bidirectional, create reverse edge
                    if self.config.bidirectional_similarity {
                        let reverse_id = Uuid::new_v4();
                        if let Ok(true) = self
                            .insert_edge(
                                reverse_id,
                                &similar_id,
                                pattern_id,
                                PatternEdgeType::SimilarTo,
                                similarity,
                                &creation_reason,
                            )
                            .await
                        {
                            result.edges_created += 1;
                            result.created_edge_ids.push(reverse_id);
                        }
                    }
                }
                Ok(false) => {
                    result.edges_existing += 1;
                }
                Err(e) => {
                    warn!(
                        pattern_id = pattern_id,
                        similar_id = similar_id,
                        error = %e,
                        "Failed to create similar edge"
                    );
                    result.errors += 1;
                }
            }
        }

        result.duration_ms = start.elapsed().as_millis() as u64;

        if result.edges_created > 0 {
            info!(
                pattern_id = pattern_id,
                edges_created = result.edges_created,
                duration_ms = result.duration_ms,
                "Created similar edges for pattern"
            );
        }

        Ok(result)
    }

    /// Record a co-retrieval event between two patterns.
    ///
    /// Updates the retrieval_pairs table to track how often patterns
    /// are retrieved together. Call this when multiple patterns are
    /// returned in the same query result.
    ///
    /// # Arguments
    ///
    /// * `pattern_a` - First pattern ID
    /// * `pattern_b` - Second pattern ID
    /// * `session_id` - Optional session identifier
    ///
    /// # Returns
    ///
    /// The updated co-retrieval count for this pair.
    pub async fn record_co_retrieval(
        &self,
        pattern_a: &str,
        pattern_b: &str,
        session_id: Option<&str>,
    ) -> Result<i32> {
        // Ensure consistent ordering (pattern_a < pattern_b)
        let (ordered_a, ordered_b) = if pattern_a < pattern_b {
            (pattern_a, pattern_b)
        } else {
            (pattern_b, pattern_a)
        };

        // Use PostgreSQL function for atomic upsert
        let row: (i32,) = sqlx::query_as(
            r#"
            INSERT INTO retrieval_pairs (
                pattern_a, pattern_b, count, first_retrieved, last_retrieved,
                last_session_id
            ) VALUES ($1, $2, 1, NOW(), NOW(), $3)
            ON CONFLICT (pattern_a, pattern_b) DO UPDATE SET
                count = retrieval_pairs.count + 1,
                last_retrieved = NOW(),
                last_session_id = COALESCE($3, retrieval_pairs.last_session_id)
            RETURNING count
            "#,
        )
        .bind(ordered_a)
        .bind(ordered_b)
        .bind(session_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| NagualError::internal(format!("Failed to record co-retrieval: {}", e)))?;

        let count = row.0;

        debug!(
            pattern_a = ordered_a,
            pattern_b = ordered_b,
            count = count,
            "Recorded co-retrieval"
        );

        Ok(count)
    }

    /// Record co-retrieval for multiple pattern pairs.
    ///
    /// Efficiently records co-retrieval for all pairs in a set of patterns.
    /// Useful when a search returns multiple patterns at once.
    ///
    /// # Arguments
    ///
    /// * `pattern_ids` - Vector of pattern IDs that were retrieved together
    /// * `session_id` - Optional session identifier
    ///
    /// # Returns
    ///
    /// Number of pairs recorded.
    pub async fn record_co_retrieval_batch(
        &self,
        pattern_ids: &[String],
        session_id: Option<&str>,
    ) -> Result<usize> {
        if pattern_ids.len() < 2 {
            return Ok(0);
        }

        let mut pairs_recorded = 0;

        // Record all unique pairs
        for i in 0..pattern_ids.len() {
            for j in (i + 1)..pattern_ids.len() {
                if let Err(e) = self
                    .record_co_retrieval(&pattern_ids[i], &pattern_ids[j], session_id)
                    .await
                {
                    warn!(
                        pattern_a = &pattern_ids[i],
                        pattern_b = &pattern_ids[j],
                        error = %e,
                        "Failed to record co-retrieval pair"
                    );
                } else {
                    pairs_recorded += 1;
                }
            }
        }

        debug!(
            pattern_count = pattern_ids.len(),
            pairs_recorded = pairs_recorded,
            "Recorded batch co-retrieval"
        );

        Ok(pairs_recorded)
    }

    /// Check for and create co-retrieval edges.
    ///
    /// Finds pattern pairs that have been retrieved together more than
    /// the threshold times, and creates `co_retrieved` edges for them.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of edges to create in this batch
    ///
    /// # Returns
    ///
    /// Result containing number of edges created.
    pub async fn check_and_create_coretrieval_edges(
        &self,
        limit: usize,
    ) -> Result<AutoEdgeResult> {
        let start = std::time::Instant::now();

        // Find candidates above threshold that don't already have edges
        let candidates: Vec<CoRetrievalCandidate> = sqlx::query_as(
            r#"
            SELECT
                rp.pattern_a,
                rp.pattern_b,
                rp.count,
                rp.last_retrieved
            FROM retrieval_pairs rp
            WHERE rp.count >= $1
            AND NOT EXISTS (
                SELECT 1 FROM pattern_edges pe
                WHERE pe.edge_type = 'co_retrieved'
                AND (
                    (pe.source_pattern = rp.pattern_a AND pe.target_pattern = rp.pattern_b)
                    OR (pe.source_pattern = rp.pattern_b AND pe.target_pattern = rp.pattern_a)
                )
            )
            ORDER BY rp.count DESC, rp.last_retrieved DESC
            LIMIT $2
            "#,
        )
        .bind(self.config.co_retrieval_threshold)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| {
            NagualError::internal(format!("Failed to find co-retrieval candidates: {}", e))
        })?;

        if candidates.is_empty() {
            return Ok(AutoEdgeResult {
                duration_ms: start.elapsed().as_millis() as u64,
                ..AutoEdgeResult::empty()
            });
        }

        let mut result = AutoEdgeResult::empty();

        for candidate in candidates {
            // Calculate normalized strength (count / (count + 10) for smooth curve)
            let strength = candidate.count as f64 / (candidate.count as f64 + 10.0);
            let edge_id = Uuid::new_v4();
            let creation_reason = EdgeCreationReason::CoRetrieval {
                count: candidate.count,
            };

            let inserted = self
                .insert_edge(
                    edge_id,
                    &candidate.pattern_a,
                    &candidate.pattern_b,
                    PatternEdgeType::CoRetrieved,
                    strength,
                    &creation_reason,
                )
                .await;

            match inserted {
                Ok(true) => {
                    result.edges_created += 1;
                    result.created_edge_ids.push(edge_id);

                    // Create reverse edge for bidirectional co-retrieval
                    let reverse_id = Uuid::new_v4();
                    if let Ok(true) = self
                        .insert_edge(
                            reverse_id,
                            &candidate.pattern_b,
                            &candidate.pattern_a,
                            PatternEdgeType::CoRetrieved,
                            strength,
                            &creation_reason,
                        )
                        .await
                    {
                        result.edges_created += 1;
                        result.created_edge_ids.push(reverse_id);
                    }
                }
                Ok(false) => {
                    result.edges_existing += 1;
                }
                Err(e) => {
                    warn!(
                        pattern_a = candidate.pattern_a,
                        pattern_b = candidate.pattern_b,
                        error = %e,
                        "Failed to create co-retrieval edge"
                    );
                    result.errors += 1;
                }
            }
        }

        result.duration_ms = start.elapsed().as_millis() as u64;

        if result.edges_created > 0 {
            info!(
                edges_created = result.edges_created,
                duration_ms = result.duration_ms,
                "Created co-retrieval edges"
            );
        }

        Ok(result)
    }

    /// Get co-retrieval count for a pattern pair.
    pub async fn get_co_retrieval_count(
        &self,
        pattern_a: &str,
        pattern_b: &str,
    ) -> Result<i32> {
        // Ensure consistent ordering
        let (ordered_a, ordered_b) = if pattern_a < pattern_b {
            (pattern_a, pattern_b)
        } else {
            (pattern_b, pattern_a)
        };

        let row: Option<(i32,)> = sqlx::query_as(
            "SELECT count FROM retrieval_pairs WHERE pattern_a = $1 AND pattern_b = $2",
        )
        .bind(ordered_a)
        .bind(ordered_b)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| NagualError::internal(format!("Failed to get co-retrieval count: {}", e)))?;

        Ok(row.map(|r| r.0).unwrap_or(0))
    }

    /// Get all edges for a pattern.
    pub async fn get_edges_for_pattern(&self, pattern_id: &str) -> Result<Vec<PatternEdge>> {
        let rows = sqlx::query(
            r#"
            SELECT
                id, source_pattern, target_pattern, edge_type::text,
                strength, confidence, bidirectional, auto_created,
                creation_reason, created_at, updated_at
            FROM pattern_edges
            WHERE source_pattern = $1 OR target_pattern = $1
            ORDER BY strength DESC
            "#,
        )
        .bind(pattern_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| NagualError::internal(format!("Failed to get edges: {}", e)))?;

        let mut edges = Vec::with_capacity(rows.len());
        for row in rows {
            let edge_type_str: String = row.get("edge_type");
            edges.push(PatternEdge {
                id: row.get("id"),
                source_pattern: row.get("source_pattern"),
                target_pattern: row.get("target_pattern"),
                edge_type: PatternEdgeType::from_db_str(&edge_type_str)
                    .unwrap_or(PatternEdgeType::RelatedTo),
                strength: row.get("strength"),
                confidence: row.get("confidence"),
                bidirectional: row.get("bidirectional"),
                auto_created: row.get("auto_created"),
                creation_reason: row.get("creation_reason"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
            });
        }

        Ok(edges)
    }

    /// Insert an edge into the database.
    ///
    /// Returns Ok(true) if edge was inserted, Ok(false) if it already exists.
    async fn insert_edge(
        &self,
        edge_id: Uuid,
        source: &str,
        target: &str,
        edge_type: PatternEdgeType,
        strength: f64,
        reason: &EdgeCreationReason,
    ) -> Result<bool> {
        let result = sqlx::query(
            r#"
            INSERT INTO pattern_edges (
                id, source_pattern, target_pattern, edge_type,
                strength, confidence, bidirectional, auto_created,
                creation_reason, metadata
            ) VALUES (
                $1, $2, $3, $4::edge_type,
                $5, $6, $7, true,
                $8, '{}'::jsonb
            )
            ON CONFLICT (source_pattern, target_pattern, edge_type) DO NOTHING
            "#,
        )
        .bind(edge_id)
        .bind(source)
        .bind(target)
        .bind(edge_type.as_db_str())
        .bind(strength)
        .bind(self.config.base_confidence)
        .bind(self.config.bidirectional_similarity && edge_type == PatternEdgeType::SimilarTo)
        .bind(reason.to_string())
        .execute(&self.pool)
        .await
        .map_err(|e| NagualError::internal(format!("Failed to insert edge: {}", e)))?;

        let inserted = result.rows_affected() > 0;

        if inserted {
            // Log to audit trail
            let _ = self
                .log_edge_audit(
                    edge_id,
                    source,
                    target,
                    edge_type,
                    "created",
                    None,
                    Some(strength),
                    &reason.to_string(),
                    None,
                )
                .await;
        }

        Ok(inserted)
    }

    /// Log an edge operation to the audit trail.
    async fn log_edge_audit(
        &self,
        edge_id: Uuid,
        source: &str,
        target: &str,
        edge_type: PatternEdgeType,
        operation: &str,
        old_strength: Option<f64>,
        new_strength: Option<f64>,
        reason: &str,
        job_id: Option<&str>,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO edge_audit_log (
                edge_id, source_pattern, target_pattern, edge_type,
                operation, old_strength, new_strength, reason, job_id
            ) VALUES ($1, $2, $3, $4::edge_type, $5::edge_operation, $6, $7, $8, $9)
            "#,
        )
        .bind(edge_id)
        .bind(source)
        .bind(target)
        .bind(edge_type.as_db_str())
        .bind(operation)
        .bind(old_strength)
        .bind(new_strength)
        .bind(reason)
        .bind(job_id)
        .execute(&self.pool)
        .await
        .map_err(|e| NagualError::internal(format!("Failed to log edge audit: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_edge_type_conversion() {
        assert_eq!(PatternEdgeType::SimilarTo.as_db_str(), "similar_to");
        assert_eq!(PatternEdgeType::CoRetrieved.as_db_str(), "co_retrieved");
        assert_eq!(PatternEdgeType::DerivedFrom.as_db_str(), "derived_from");
        assert_eq!(PatternEdgeType::RelatedTo.as_db_str(), "related_to");

        assert_eq!(
            PatternEdgeType::from_db_str("similar_to"),
            Some(PatternEdgeType::SimilarTo)
        );
        assert_eq!(
            PatternEdgeType::from_db_str("co_retrieved"),
            Some(PatternEdgeType::CoRetrieved)
        );
        assert_eq!(PatternEdgeType::from_db_str("invalid"), None);
    }

    #[test]
    fn test_edge_creation_reason_display() {
        let similarity = EdgeCreationReason::Similarity { score: 0.95 };
        assert!(similarity.to_string().contains("0.95"));

        let co_retrieval = EdgeCreationReason::CoRetrieval { count: 5 };
        assert!(co_retrieval.to_string().contains("5 times"));

        let manual = EdgeCreationReason::Manual {
            reason: "test".to_string(),
        };
        assert!(manual.to_string().contains("test"));
    }

    #[test]
    fn test_auto_edge_config_defaults() {
        let config = AutoEdgeConfig::default();
        assert_eq!(config.similarity_threshold, 0.85);
        assert_eq!(config.co_retrieval_threshold, 3);
        assert_eq!(config.max_similar_edges, 5);
        assert!(config.bidirectional_similarity);
    }

    #[test]
    fn test_auto_edge_config_builder() {
        let config = AutoEdgeConfig::default()
            .with_similarity_threshold(0.9)
            .with_co_retrieval_threshold(5)
            .with_max_similar_edges(10);

        assert_eq!(config.similarity_threshold, 0.9);
        assert_eq!(config.co_retrieval_threshold, 5);
        assert_eq!(config.max_similar_edges, 10);
    }

    #[test]
    fn test_auto_edge_config_clamping() {
        let config = AutoEdgeConfig::default().with_similarity_threshold(1.5);
        assert_eq!(config.similarity_threshold, 1.0);

        let config = AutoEdgeConfig::default().with_similarity_threshold(-0.5);
        assert_eq!(config.similarity_threshold, 0.0);

        let config = AutoEdgeConfig::default().with_co_retrieval_threshold(-1);
        assert_eq!(config.co_retrieval_threshold, 1);
    }

    #[test]
    fn test_auto_edge_result_empty() {
        let result = AutoEdgeResult::empty();
        assert_eq!(result.edges_created, 0);
        assert!(!result.has_changes());
    }
}
