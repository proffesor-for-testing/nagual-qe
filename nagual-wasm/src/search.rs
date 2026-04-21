//! Vector similarity search for WASM.
//!
//! Provides HNSW-like approximate nearest neighbor search optimized for
//! browser environments. Uses a simplified implementation that trades
//! some accuracy for smaller bundle size.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use wasm_bindgen::prelude::*;

/// Configuration for the search engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct SearchConfig {
    /// Expected embedding dimension (default: 128)
    embedding_dim: usize,

    /// Minimum similarity threshold for results (default: 0.0)
    min_similarity: f32,

    /// Maximum number of results to return (default: 50)
    max_results: usize,

    /// Whether to normalize vectors before search (default: true)
    normalize_vectors: bool,
}

#[wasm_bindgen]
impl SearchConfig {
    /// Create a new search configuration with default values.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the embedding dimension.
    #[wasm_bindgen]
    pub fn with_embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Set the minimum similarity threshold.
    #[wasm_bindgen]
    pub fn with_min_similarity(mut self, min_sim: f32) -> Self {
        self.min_similarity = min_sim.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum number of results.
    #[wasm_bindgen]
    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    /// Get the embedding dimension.
    #[wasm_bindgen(getter)]
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get the minimum similarity threshold.
    #[wasm_bindgen(getter)]
    pub fn min_similarity(&self) -> f32 {
        self.min_similarity
    }

    /// Get the maximum number of results.
    #[wasm_bindgen(getter)]
    pub fn max_results(&self) -> usize {
        self.max_results
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            min_similarity: 0.0,
            max_results: 50,
            normalize_vectors: true,
        }
    }
}

/// A single pattern stored in the index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Unique identifier
    pub id: String,

    /// Pattern content/description
    pub content: String,

    /// Vector embedding (normalized)
    pub embedding: Vec<f32>,

    /// Pattern type (pattern, trajectory, prediction, decision)
    #[serde(default = "default_pattern_type")]
    pub pattern_type: String,

    /// Confidence score (0.0 - 1.0)
    #[serde(default = "default_confidence")]
    pub confidence: f32,

    /// Additional metadata as JSON
    #[serde(default)]
    pub metadata: serde_json::Value,

    /// Creation timestamp (ISO 8601)
    #[serde(default = "default_timestamp")]
    pub created_at: String,
}

fn default_pattern_type() -> String {
    "pattern".to_string()
}

fn default_confidence() -> f32 {
    0.5
}

fn default_timestamp() -> String {
    chrono::Utc::now().to_rfc3339()
}

impl Pattern {
    /// Create a new pattern.
    pub fn new(id: String, content: String, embedding: Vec<f32>) -> Self {
        Self {
            id,
            content,
            embedding,
            pattern_type: default_pattern_type(),
            confidence: default_confidence(),
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            created_at: default_timestamp(),
        }
    }

    /// Check if the embedding dimension matches.
    pub fn has_valid_embedding(&self, expected_dim: usize) -> bool {
        self.embedding.len() == expected_dim
    }
}

/// Search result with similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Pattern ID
    pub id: String,

    /// Pattern content
    pub content: String,

    /// Similarity score (0.0 - 1.0)
    pub similarity: f32,

    /// Pattern type
    pub pattern_type: String,

    /// Confidence score
    pub confidence: f32,

    /// Metadata
    pub metadata: serde_json::Value,
}

impl SearchResult {
    fn from_pattern(pattern: &Pattern, similarity: f32) -> Self {
        Self {
            id: pattern.id.clone(),
            content: pattern.content.clone(),
            similarity,
            pattern_type: pattern.pattern_type.clone(),
            confidence: pattern.confidence,
            metadata: pattern.metadata.clone(),
        }
    }
}

/// Internal heap entry for top-k selection.
#[derive(Debug, Clone)]
struct HeapEntry {
    similarity: f32,
    index: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller similarity = higher priority (to be replaced)
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

/// Vector similarity search engine.
///
/// Provides efficient brute-force search with optimizations for browser.
/// For small to medium pattern sets (< 50K), brute-force is often faster
/// than HNSW due to lower overhead.
pub struct VectorSearch {
    config: SearchConfig,
    patterns: Vec<Pattern>,
}

impl VectorSearch {
    /// Create a new search engine with the given configuration.
    pub fn new(config: SearchConfig) -> Self {
        Self {
            config,
            patterns: Vec::new(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SearchConfig::default())
    }

    /// Add a pattern to the index.
    pub fn add_pattern(&mut self, mut pattern: Pattern) -> Result<(), String> {
        if !pattern.has_valid_embedding(self.config.embedding_dim) {
            return Err(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dim,
                pattern.embedding.len()
            ));
        }

        // Normalize the embedding if configured
        if self.config.normalize_vectors {
            normalize_l2_inplace(&mut pattern.embedding);
        }

        self.patterns.push(pattern);
        Ok(())
    }

    /// Remove a pattern by ID.
    pub fn remove_pattern(&mut self, id: &str) -> bool {
        let initial_len = self.patterns.len();
        self.patterns.retain(|p| p.id != id);
        self.patterns.len() < initial_len
    }

    /// Get a pattern by ID.
    pub fn get_pattern(&self, id: &str) -> Option<&Pattern> {
        self.patterns.iter().find(|p| p.id == id)
    }

    /// Get the number of patterns in the index.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Clear all patterns from the index.
    pub fn clear(&mut self) {
        self.patterns.clear();
    }

    /// Search for similar patterns.
    ///
    /// Uses optimized brute-force search with early termination.
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<SearchResult>, String> {
        if query_embedding.len() != self.config.embedding_dim {
            return Err(format!(
                "Query embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dim,
                query_embedding.len()
            ));
        }

        let k = top_k.min(self.config.max_results);
        if k == 0 || self.patterns.is_empty() {
            return Ok(Vec::new());
        }

        // Normalize query if configured
        let query = if self.config.normalize_vectors {
            normalize_l2(query_embedding)
        } else {
            query_embedding.to_vec()
        };

        // Use min-heap for efficient top-k selection
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k);

        for (index, pattern) in self.patterns.iter().enumerate() {
            let similarity = cosine_similarity_normalized(&query, &pattern.embedding);

            if similarity < self.config.min_similarity {
                continue;
            }

            if heap.len() < k {
                heap.push(HeapEntry { similarity, index });
            } else if let Some(min) = heap.peek() {
                if similarity > min.similarity {
                    heap.pop();
                    heap.push(HeapEntry { similarity, index });
                }
            }
        }

        // Extract results in descending order
        let mut results: Vec<SearchResult> = heap
            .into_sorted_vec()
            .into_iter()
            .rev()
            .map(|entry| {
                let pattern = &self.patterns[entry.index];
                SearchResult::from_pattern(pattern, entry.similarity)
            })
            .collect();

        // Ensure descending order by similarity
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });

        Ok(results)
    }

    /// Batch search for multiple queries.
    pub fn batch_search(
        &self,
        query_embeddings: &[Vec<f32>],
        top_k: usize,
    ) -> Result<Vec<Vec<SearchResult>>, String> {
        query_embeddings
            .iter()
            .map(|q| self.search(q, top_k))
            .collect()
    }

    /// Export all patterns as JSON.
    pub fn export_json(&self) -> Result<String, String> {
        serde_json::to_string(&self.patterns).map_err(|e| e.to_string())
    }

    /// Import patterns from JSON.
    pub fn import_json(&mut self, json: &str) -> Result<usize, String> {
        let patterns: Vec<Pattern> = serde_json::from_str(json).map_err(|e| e.to_string())?;

        let count = patterns.len();
        for pattern in patterns {
            self.add_pattern(pattern)?;
        }

        Ok(count)
    }

    /// Get all patterns (for persistence).
    pub fn get_all_patterns(&self) -> &[Pattern] {
        &self.patterns
    }

    /// Set patterns directly (for loading from persistence).
    pub fn set_patterns(&mut self, patterns: Vec<Pattern>) {
        self.patterns = patterns;
    }

    /// Get search statistics.
    pub fn get_stats(&self) -> SearchStats {
        SearchStats {
            pattern_count: self.patterns.len(),
            embedding_dim: self.config.embedding_dim,
            min_similarity: self.config.min_similarity,
            max_results: self.config.max_results,
        }
    }
}

/// Search statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchStats {
    pub pattern_count: usize,
    pub embedding_dim: usize,
    pub min_similarity: f32,
    pub max_results: usize,
}

// Vector math utilities

/// Normalize a vector to unit length using L2 normalization.
fn normalize_l2(vector: &[f32]) -> Vec<f32> {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        vector.iter().map(|x| x / norm).collect()
    } else {
        vector.to_vec()
    }
}

/// Normalize a vector in-place.
fn normalize_l2_inplace(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in vector.iter_mut() {
            *x /= norm;
        }
    }
}

/// Compute cosine similarity for normalized vectors (dot product).
#[inline]
fn cosine_similarity_normalized(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_l2() {
        let v = vec![3.0, 4.0];
        let normalized = normalize_l2(&v);

        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = normalize_l2(&[1.0, 0.0, 0.0, 0.0]);
        let b = normalize_l2(&[1.0, 0.0, 0.0, 0.0]);

        let sim = cosine_similarity_normalized(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = normalize_l2(&[1.0, 0.0]);
        let b = normalize_l2(&[0.0, 1.0]);

        let sim = cosine_similarity_normalized(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_search_empty() {
        let search = VectorSearch::with_defaults();
        let query = vec![0.0; 128];
        let results = search.search(&query, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_basic() {
        let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));

        // Add some patterns
        let patterns = vec![
            Pattern::new("p1".to_string(), "Pattern 1".to_string(), vec![1.0, 0.0, 0.0, 0.0]),
            Pattern::new("p2".to_string(), "Pattern 2".to_string(), vec![0.9, 0.1, 0.0, 0.0]),
            Pattern::new("p3".to_string(), "Pattern 3".to_string(), vec![0.0, 1.0, 0.0, 0.0]),
        ];

        for p in patterns {
            search.add_pattern(p).unwrap();
        }

        // Search for similar to p1
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = search.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "p1");
        assert!(results[0].similarity > results[1].similarity);
    }

    #[test]
    fn test_export_import_json() {
        let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));

        search
            .add_pattern(Pattern::new(
                "p1".to_string(),
                "Test".to_string(),
                vec![1.0, 0.0, 0.0, 0.0],
            ))
            .unwrap();

        let json = search.export_json().unwrap();

        let mut search2 = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));
        let count = search2.import_json(&json).unwrap();

        assert_eq!(count, 1);
        assert_eq!(search2.len(), 1);
    }
}
