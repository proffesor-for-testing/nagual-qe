//! Vector similarity search for patterns.
//!
//! Provides high-performance similarity search using instant-distance HNSW
//! (Hierarchical Navigable Small World) algorithm for SQLite fallback,
//! with support for pgvector when PostgreSQL is available.

use std::sync::Arc;

use instant_distance::{Builder as HnswBuilder, HnswMap, Search};
use ndarray::Array1;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use super::pattern::{Pattern, PatternId};
use crate::db::DualWriteAdapter;
use crate::error::Result;
use crate::ml::{cosine_similarity, cosine_similarity_normalized, normalize_l2};

/// Configuration for vector search.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Expected embedding dimension.
    pub embedding_dim: usize,

    /// HNSW M parameter (number of connections per layer).
    pub hnsw_m: usize,

    /// HNSW M0 parameter (connections for base layer).
    pub hnsw_m0: usize,

    /// HNSW ef_construction parameter (size of dynamic candidate list).
    pub hnsw_ef_construction: usize,

    /// HNSW ef_search parameter (search depth).
    pub hnsw_ef_search: usize,

    /// Whether to normalize vectors before indexing.
    pub normalize_vectors: bool,

    /// Minimum similarity threshold for results.
    pub min_similarity: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            hnsw_m: 24,           // Increased for better recall at scale
            hnsw_m0: 48,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 200,  // Critical: must be >= 50 for 95% recall, 200 for near-perfect
            normalize_vectors: true,
            min_similarity: 0.0,
        }
    }
}

/// A search result with pattern and similarity score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matching pattern.
    pub pattern: Pattern,

    /// Cosine similarity score (0.0 to 1.0 for normalized vectors).
    pub similarity: f32,
}

impl SearchResult {
    /// Create a new search result.
    pub fn new(pattern: Pattern, similarity: f32) -> Self {
        Self { pattern, similarity }
    }
}

/// Wrapper for pattern embeddings in HNSW index.
#[derive(Clone)]
struct PatternPoint {
    /// Pattern ID
    id: PatternId,

    /// Normalized embedding vector
    embedding: Vec<f32>,
}

impl instant_distance::Point for PatternPoint {
    #[inline]
    fn distance(&self, other: &Self) -> f32 {
        // Use cosine distance (1 - cosine_similarity) for HNSW
        // Zero-allocation: direct dot product (embeddings are pre-normalized)
        let dot: f32 = self
            .embedding
            .iter()
            .zip(other.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();
        1.0 - dot
    }
}

/// Vector similarity search engine.
///
/// Uses HNSW (Hierarchical Navigable Small World) algorithm via instant-distance
/// for fast approximate nearest neighbor search. Falls back to brute-force
/// search for small datasets.
pub struct VectorSearch {
    /// The dual-write adapter for database access.
    adapter: Arc<DualWriteAdapter>,

    /// Configuration.
    config: SearchConfig,

    /// HNSW index (built lazily).
    hnsw_index: RwLock<Option<HnswMap<PatternPoint, PatternId>>>,

    /// Pattern cache for id->pattern lookup.
    pattern_cache: RwLock<hashbrown::HashMap<PatternId, Pattern>>,

    /// Whether the index needs rebuilding.
    index_dirty: RwLock<bool>,
}

impl VectorSearch {
    /// Create a new VectorSearch.
    pub fn new(adapter: Arc<DualWriteAdapter>, config: SearchConfig) -> Result<Self> {
        Ok(Self {
            adapter,
            config,
            hnsw_index: RwLock::new(None),
            pattern_cache: RwLock::new(hashbrown::HashMap::new()),
            index_dirty: RwLock::new(true),
        })
    }

    /// Perform similarity search for the query embedding.
    ///
    /// Returns the top K most similar patterns sorted by similarity.
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - The query vector
    /// * `k` - Number of results to return (max 50)
    ///
    /// # Returns
    ///
    /// Vector of SearchResult sorted by similarity (highest first)
    pub async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        // Validate query dimension
        if query_embedding.len() != self.config.embedding_dim {
            return Err(crate::error::NagualError::Config {
                message: format!(
                    "Query embedding dimension {} doesn't match expected {}",
                    query_embedding.len(),
                    self.config.embedding_dim
                ),
            });
        }

        let k = k.min(50); // Cap at 50 as per requirements

        // Ensure index is up to date
        self.ensure_index_built().await?;

        // Normalize query if configured
        let query_vec = if self.config.normalize_vectors {
            let arr = Array1::from_vec(query_embedding.to_vec());
            normalize_l2(&arr.view()).to_vec()
        } else {
            query_embedding.to_vec()
        };

        // Check if HNSW index exists
        let index_guard = self.hnsw_index.read();
        if let Some(ref index) = *index_guard {
            // Use HNSW for fast approximate search
            self.search_hnsw(index, &query_vec, k)
        } else {
            // Fall back to brute force for small datasets
            drop(index_guard);
            self.search_brute_force(&query_vec, k).await
        }
    }

    /// Search using HNSW index.
    fn search_hnsw(
        &self,
        index: &HnswMap<PatternPoint, PatternId>,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let query_point = PatternPoint {
            id: PatternId::from_string("query"),
            embedding: query.to_vec(),
        };

        let mut search = Search::default();
        let neighbors = index.search(&query_point, &mut search);

        let cache = self.pattern_cache.read();
        let mut results = Vec::with_capacity(k);

        for neighbor in neighbors.take(k) {
            let pattern_id = neighbor.value;
            if let Some(pattern) = cache.get(pattern_id) {
                // Calculate actual similarity (HNSW returns distance)
                let distance = neighbor.distance;
                let similarity = 1.0 - distance;

                if similarity >= self.config.min_similarity {
                    results.push(SearchResult::new(pattern.clone(), similarity));
                }
            }
        }

        // Sort by similarity (highest first)
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        debug!(
            query_dim = query.len(),
            results = results.len(),
            "HNSW search completed"
        );

        Ok(results)
    }

    /// Brute-force search for small datasets or when HNSW is not available.
    async fn search_brute_force(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        // Get all patterns with embeddings from cache or database
        let patterns = self.get_patterns_with_embeddings().await?;

        if patterns.is_empty() {
            return Ok(Vec::new());
        }

        let query_arr = Array1::from_vec(query.to_vec());
        let query_view = query_arr.view();

        // Calculate similarities
        let mut scored: Vec<(Pattern, f32)> = patterns
            .into_iter()
            .filter_map(|pattern| {
                let emb = pattern.embedding()?;
                let emb_arr = Array1::from_vec(emb.to_vec());
                let similarity = if self.config.normalize_vectors {
                    cosine_similarity_normalized(&query_view, &emb_arr.view())
                } else {
                    cosine_similarity(&query_view, &emb_arr.view())
                };
                Some((pattern, similarity))
            })
            .filter(|(_, sim)| *sim >= self.config.min_similarity)
            .collect();

        // Sort by similarity (highest first)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top k
        let results: Vec<SearchResult> = scored
            .into_iter()
            .take(k)
            .map(|(pattern, similarity)| SearchResult::new(pattern, similarity))
            .collect();

        debug!(
            query_dim = query.len(),
            results = results.len(),
            "Brute-force search completed"
        );

        Ok(results)
    }

    /// Ensure the HNSW index is built and up to date.
    async fn ensure_index_built(&self) -> Result<()> {
        let is_dirty = *self.index_dirty.read();
        if !is_dirty {
            return Ok(());
        }

        // Rebuild the index
        self.rebuild_index().await
    }

    /// Rebuild the HNSW index from all patterns.
    pub async fn rebuild_index(&self) -> Result<()> {
        let patterns = self.get_patterns_with_embeddings().await?;

        if patterns.is_empty() {
            info!("No patterns with embeddings, skipping HNSW index build");
            *self.index_dirty.write() = false;
            return Ok(());
        }

        // Build points for HNSW
        let mut points: Vec<PatternPoint> = Vec::with_capacity(patterns.len());
        let mut pattern_map: hashbrown::HashMap<PatternId, Pattern> = hashbrown::HashMap::new();

        for pattern in patterns {
            if let Some(embedding) = pattern.embedding() {
                let normalized = if self.config.normalize_vectors {
                    let arr = Array1::from_vec(embedding.to_vec());
                    normalize_l2(&arr.view()).to_vec()
                } else {
                    embedding.to_vec()
                };

                let point = PatternPoint {
                    id: pattern.id().clone(),
                    embedding: normalized,
                };

                points.push(point);
                pattern_map.insert(pattern.id().clone(), pattern);
            }
        }

        if points.is_empty() {
            warn!("No valid embeddings found, cannot build HNSW index");
            *self.index_dirty.write() = false;
            return Ok(());
        }

        // Build HNSW index
        let point_vec: Vec<PatternPoint> = points.iter().cloned().collect();
        let value_vec: Vec<PatternId> = points.iter().map(|p| p.id.clone()).collect();
        let hnsw = HnswBuilder::default()
            .ef_construction(self.config.hnsw_ef_construction)
            .ef_search(self.config.hnsw_ef_search)  // Critical: set search quality at build time
            .build(point_vec, value_vec);

        // Update caches
        *self.hnsw_index.write() = Some(hnsw);
        *self.pattern_cache.write() = pattern_map;
        *self.index_dirty.write() = false;

        info!(
            num_patterns = points.len(),
            "HNSW index rebuilt successfully"
        );

        Ok(())
    }

    /// Get all patterns with embeddings from the database.
    async fn get_patterns_with_embeddings(&self) -> Result<Vec<Pattern>> {
        let sql = r#"
            SELECT * FROM reasoning_patterns
            WHERE embedding IS NOT NULL AND embedding != ''
        "#;

        let patterns = self
            .adapter
            .sqlite()
            .query(sql, &[], |row| {
                parse_pattern_from_row(row)
            })
            .await?;

        Ok(patterns)
    }

    /// Mark the index as needing rebuild (e.g., after pattern updates).
    pub fn invalidate_index(&self) {
        *self.index_dirty.write() = true;
    }

    /// Add a pattern to the search index.
    ///
    /// This is more efficient than rebuilding the entire index
    /// for incremental updates.
    pub fn add_pattern(&self, pattern: &Pattern) -> Result<()> {
        if pattern.embedding().is_none() {
            return Ok(()); // Can't index without embedding
        }

        // Add to cache
        self.pattern_cache.write().insert(pattern.id().clone(), pattern.clone());

        // Mark index as dirty for rebuild on next search
        // (In a production system, we might do incremental updates)
        *self.index_dirty.write() = true;

        Ok(())
    }

    /// Remove a pattern from the search index.
    pub fn remove_pattern(&self, id: &PatternId) {
        self.pattern_cache.write().remove(id);
        *self.index_dirty.write() = true;
    }

    /// Get the search configuration.
    pub fn config(&self) -> &SearchConfig {
        &self.config
    }

    /// Get the number of indexed patterns.
    pub fn indexed_count(&self) -> usize {
        self.pattern_cache.read().len()
    }

    /// Check if the index needs rebuilding.
    pub fn is_index_dirty(&self) -> bool {
        *self.index_dirty.read()
    }
}

/// Parse a pattern from a database row.
fn parse_pattern_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Pattern> {
    use super::pattern::{PatternCategory, PatternId, PatternMetadata};
    use chrono::{DateTime, Utc};

    let id: String = row.get("id")?;
    let timestamp_str: String = row.get("timestamp")?;
    let _updated_at_str: String = row.get("updated_at")?;
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

    let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());

    let embedding: Option<Vec<f32>> = embedding_json
        .and_then(|json| serde_json::from_str(&json).ok());

    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
    let related_patterns: Vec<PatternId> = serde_json::from_str(&related_json).unwrap_or_default();
    let metadata: PatternMetadata = serde_json::from_str(&metadata_json).unwrap_or_default();

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

    Ok(builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_config_default() {
        let config = SearchConfig::default();
        assert_eq!(config.embedding_dim, 128);
        assert_eq!(config.hnsw_m, 24);
        assert_eq!(config.hnsw_ef_search, 200);
        assert!(config.normalize_vectors);
    }

    #[test]
    fn test_search_result() {
        let pattern = Pattern::builder()
            .problem("Test")
            .solution("Solution")
            .build();

        let result = SearchResult::new(pattern.clone(), 0.85);
        assert_eq!(result.pattern.problem(), "Test");
        assert!((result.similarity - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_pattern_point_distance() {
        use instant_distance::Point;

        let p1 = PatternPoint {
            id: PatternId::from_string("1"),
            embedding: vec![1.0, 0.0],
        };

        let p2 = PatternPoint {
            id: PatternId::from_string("2"),
            embedding: vec![1.0, 0.0],
        };

        // Same vector should have distance 0
        let dist = p1.distance(&p2);
        assert!(dist.abs() < 0.001);
    }

    #[test]
    fn test_pattern_point_orthogonal() {
        use instant_distance::Point;

        let p1 = PatternPoint {
            id: PatternId::from_string("1"),
            embedding: vec![1.0, 0.0],
        };

        let p2 = PatternPoint {
            id: PatternId::from_string("2"),
            embedding: vec![0.0, 1.0],
        };

        // Orthogonal vectors should have distance 1 (1 - 0)
        let dist = p1.distance(&p2);
        assert!((dist - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_vector_search_new() {
        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let search = VectorSearch::new(adapter, SearchConfig::default()).unwrap();

        assert_eq!(search.indexed_count(), 0);
        assert!(search.is_index_dirty());
    }

    /// Helper to create the reasoning_patterns table for testing
    async fn create_test_table(adapter: &Arc<DualWriteAdapter>) {
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
            )
        "#;
        adapter.sqlite().execute_batch(create_table_sql).await.unwrap();
    }

    #[tokio::test]
    async fn test_vector_search_brute_force() {
        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        create_test_table(&adapter).await;
        let search = VectorSearch::new(adapter.clone(), SearchConfig::default()).unwrap();

        // Create patterns with 128-dimensional embeddings
        let emb1: Vec<f32> = (0..128).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();

        // Search with empty database should return empty
        let results = search.similarity_search(&emb1, 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_invalidate_index() {
        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        create_test_table(&adapter).await;
        let search = VectorSearch::new(adapter, SearchConfig::default()).unwrap();

        // Initially dirty
        assert!(search.is_index_dirty());

        // After rebuild, should be clean
        search.rebuild_index().await.unwrap();
        assert!(!search.is_index_dirty());

        // After invalidation, should be dirty again
        search.invalidate_index();
        assert!(search.is_index_dirty());
    }
}
