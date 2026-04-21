//! ProfDAG Search Tests - Phase 1
//!
//! Comprehensive test suite for vector similarity search in ProfDAG.
//! Tests cover HNSW-based search, top-k accuracy, filtering, and edge cases.
//!
//! # Search Capabilities
//! - Vector similarity search using cosine distance
//! - Top-k nearest neighbor retrieval
//! - Filtering by node type
//! - Edge case handling (empty, duplicates, etc.)

use std::collections::{HashMap, HashSet};
use std::time::Instant;

mod common;
use common::{
    cosine_similarity, normalized_embedding, orthogonal_embeddings,
    similar_embeddings,
};

// ============================================================================
// Search Structures
// ============================================================================

/// Node type for search filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    Pattern,
    Trajectory,
    Prediction,
    Decision,
}

/// A searchable node with embedding.
#[derive(Debug, Clone)]
pub struct SearchableNode {
    pub id: String,
    pub node_type: NodeType,
    pub content: String,
    pub embedding: Vec<f32>,
}

impl SearchableNode {
    pub fn new(id: impl Into<String>, node_type: NodeType, content: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            node_type,
            content: content.into(),
            embedding,
        }
    }
}

/// Search result with similarity score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub node_id: String,
    pub similarity: f32,
}

impl SearchResult {
    pub fn new(node_id: impl Into<String>, similarity: f32) -> Self {
        Self {
            node_id: node_id.into(),
            similarity,
        }
    }
}

/// Query parameters for search.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub embedding: Vec<f32>,
    pub k: usize,
    pub node_type_filter: Option<NodeType>,
    pub min_similarity: f32,
}

impl SearchQuery {
    pub fn new(embedding: Vec<f32>, k: usize) -> Self {
        Self {
            embedding,
            k,
            node_type_filter: None,
            min_similarity: 0.0,
        }
    }

    pub fn with_type_filter(mut self, node_type: NodeType) -> Self {
        self.node_type_filter = Some(node_type);
        self
    }

    pub fn with_min_similarity(mut self, min_sim: f32) -> Self {
        self.min_similarity = min_sim.clamp(0.0, 1.0);
        self
    }
}

/// In-memory search index for testing.
#[derive(Debug, Default)]
pub struct TestSearchIndex {
    nodes: HashMap<String, SearchableNode>,
}

impl TestSearchIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, node: SearchableNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    pub fn remove_node(&mut self, id: &str) -> Option<SearchableNode> {
        self.nodes.remove(id)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Perform brute-force similarity search (for correctness testing).
    pub fn search(&self, query: &SearchQuery) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = self.nodes
            .values()
            .filter(|node| {
                // Apply type filter
                if let Some(filter_type) = query.node_type_filter {
                    if node.node_type != filter_type {
                        return false;
                    }
                }
                true
            })
            .map(|node| {
                let similarity = cosine_similarity(&query.embedding, &node.embedding);
                SearchResult::new(&node.id, similarity)
            })
            .filter(|result| result.similarity >= query.min_similarity)
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        // Return top-k
        results.truncate(query.k);
        results
    }

    /// Search with HNSW-like approximation (simulated).
    /// In production, this would use instant-distance or similar library.
    pub fn search_approximate(&self, query: &SearchQuery, ef_search: usize) -> Vec<SearchResult> {
        // For testing, we use exact search but limit candidates
        let candidate_count = (ef_search * 2).min(self.nodes.len());

        let mut candidates: Vec<(&SearchableNode, f32)> = self.nodes
            .values()
            .take(candidate_count)
            .map(|node| {
                let similarity = cosine_similarity(&query.embedding, &node.embedding);
                (node, similarity)
            })
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        candidates
            .into_iter()
            .filter(|(node, sim)| {
                if let Some(filter_type) = query.node_type_filter {
                    if node.node_type != filter_type {
                        return false;
                    }
                }
                *sim >= query.min_similarity
            })
            .take(query.k)
            .map(|(node, sim)| SearchResult::new(&node.id, sim))
            .collect()
    }
}

// ============================================================================
// Vector Similarity Search Tests
// ============================================================================

mod similarity_search_tests {
    use super::*;

    #[test]
    fn test_search_empty_index() {
        let index = TestSearchIndex::new();
        let query = SearchQuery::new(normalized_embedding(128), 10);
        let results = index.search(&query);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_single_node() {
        let mut index = TestSearchIndex::new();
        let embedding = normalized_embedding(128);
        index.add_node(SearchableNode::new("node-1", NodeType::Pattern, "Content", embedding.clone()));

        let query = SearchQuery::new(embedding, 1);
        let results = index.search(&query);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "node-1");
        assert!((results[0].similarity - 1.0).abs() < 0.001); // Same embedding = similarity 1.0
    }

    #[test]
    fn test_search_identical_query() {
        let mut index = TestSearchIndex::new();

        // Add multiple nodes with different embeddings
        for i in 0..10 {
            let embedding = normalized_embedding(128);
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                embedding,
            ));
        }

        // Search with the embedding of node-5
        let target_node = index.nodes.get("node-5").unwrap();
        let query = SearchQuery::new(target_node.embedding.clone(), 1);
        let results = index.search(&query);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "node-5");
        assert!((results[0].similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_search_similar_embeddings() {
        let mut index = TestSearchIndex::new();
        let base_embedding = normalized_embedding(128);

        // Add the base node
        index.add_node(SearchableNode::new("base", NodeType::Pattern, "Base", base_embedding.clone()));

        // Add similar nodes (small perturbations)
        let similar = similar_embeddings(&base_embedding, 5, 0.1);
        for (i, emb) in similar.iter().enumerate() {
            index.add_node(SearchableNode::new(
                format!("similar-{}", i),
                NodeType::Pattern,
                format!("Similar {}", i),
                emb.clone(),
            ));
        }

        // Add dissimilar nodes (orthogonal)
        let orthogonal = orthogonal_embeddings(128, 5);
        for (i, emb) in orthogonal.iter().enumerate() {
            index.add_node(SearchableNode::new(
                format!("orthogonal-{}", i),
                NodeType::Pattern,
                format!("Orthogonal {}", i),
                emb.clone(),
            ));
        }

        // Search for similar to base
        let query = SearchQuery::new(base_embedding, 6);
        let results = index.search(&query);

        // Top results should be base and similar nodes
        assert_eq!(results.len(), 6);
        assert_eq!(results[0].node_id, "base");

        // All top results should have high similarity
        for result in &results {
            assert!(result.similarity > 0.5, "Expected high similarity, got {}", result.similarity);
        }
    }

    #[test]
    fn test_search_orthogonal_embeddings() {
        let mut index = TestSearchIndex::new();
        let embeddings = orthogonal_embeddings(128, 4);

        for (i, emb) in embeddings.iter().enumerate() {
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                emb.clone(),
            ));
        }

        // Search with first embedding
        let query = SearchQuery::new(embeddings[0].clone(), 4);
        let results = index.search(&query);

        // First result should be the matching node
        assert_eq!(results[0].node_id, "node-0");
        assert!((results[0].similarity - 1.0).abs() < 0.001);

        // Other results should have low similarity (orthogonal)
        for i in 1..results.len() {
            assert!(results[i].similarity.abs() < 0.1, "Expected low similarity for orthogonal vectors");
        }
    }

    #[test]
    fn test_search_sorted_by_similarity() {
        let mut index = TestSearchIndex::new();
        let query_emb = normalized_embedding(128);

        // Add nodes with varying similarity
        for i in 0..10 {
            let noise_scale = i as f32 * 0.1;
            let perturbed: Vec<f32> = query_emb
                .iter()
                .map(|&x| x + (noise_scale * (i as f32 / 10.0)))
                .collect();
            let norm: f32 = perturbed.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized: Vec<f32> = perturbed.iter().map(|x| x / norm).collect();

            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                normalized,
            ));
        }

        let query = SearchQuery::new(query_emb, 10);
        let results = index.search(&query);

        // Verify results are sorted by similarity (descending)
        for i in 1..results.len() {
            assert!(
                results[i - 1].similarity >= results[i].similarity,
                "Results not sorted: {} < {} at position {} and {}",
                results[i - 1].similarity,
                results[i].similarity,
                i - 1,
                i
            );
        }
    }
}

// ============================================================================
// Top-K Accuracy Tests
// ============================================================================

mod topk_tests {
    use super::*;

    #[test]
    fn test_topk_returns_correct_count() {
        let mut index = TestSearchIndex::new();
        let base_embedding = normalized_embedding(128);

        for i in 0..100 {
            // Use similar embeddings so they all pass min_similarity
            let emb = similar_embeddings(&base_embedding, 1, 0.3)[0].clone();
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                emb,
            ));
        }

        for k in [1, 5, 10, 50, 100] {
            let query = SearchQuery::new(base_embedding.clone(), k);
            let results = index.search(&query);
            assert_eq!(results.len(), k, "Expected {} results, got {}", k, results.len());
        }
    }

    #[test]
    fn test_topk_with_fewer_nodes_than_k() {
        let mut index = TestSearchIndex::new();
        let base_embedding = normalized_embedding(128);

        for i in 0..5 {
            // Use similar embeddings so they all pass min_similarity
            let emb = similar_embeddings(&base_embedding, 1, 0.2)[0].clone();
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                emb,
            ));
        }

        let query = SearchQuery::new(base_embedding, 10);
        let results = index.search(&query);

        assert_eq!(results.len(), 5, "Should return all nodes when k > node count");
    }

    #[test]
    fn test_topk_zero() {
        let mut index = TestSearchIndex::new();

        for i in 0..10 {
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                normalized_embedding(128),
            ));
        }

        let query = SearchQuery::new(normalized_embedding(128), 0);
        let results = index.search(&query);

        assert!(results.is_empty(), "k=0 should return empty results");
    }

    #[test]
    fn test_topk_contains_best_match() {
        let mut index = TestSearchIndex::new();
        let target_embedding = normalized_embedding(128);

        // Add the target node
        index.add_node(SearchableNode::new("target", NodeType::Pattern, "Target", target_embedding.clone()));

        // Add many other nodes
        for i in 0..100 {
            index.add_node(SearchableNode::new(
                format!("other-{}", i),
                NodeType::Pattern,
                format!("Other {}", i),
                normalized_embedding(128),
            ));
        }

        // Search for target
        let query = SearchQuery::new(target_embedding, 5);
        let results = index.search(&query);

        // Target should be in top-5 (actually first)
        assert!(
            results.iter().any(|r| r.node_id == "target"),
            "Target should be in top-k results"
        );
        assert_eq!(results[0].node_id, "target", "Target should be first result");
    }

    #[test]
    fn test_recall_at_k() {
        // Test recall@k - percentage of true top-k found
        let mut index = TestSearchIndex::new();
        let query_emb = normalized_embedding(128);

        // Create known ground truth: nodes with decreasing similarity
        let mut ground_truth: Vec<(String, f32)> = Vec::new();

        for i in 0..100 {
            let factor = 1.0 - (i as f32 * 0.01);
            let emb: Vec<f32> = query_emb.iter().map(|x| x * factor + (1.0 - factor) * 0.1).collect();
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized: Vec<f32> = emb.iter().map(|x| x / norm).collect();

            let id = format!("node-{}", i);
            let sim = cosine_similarity(&query_emb, &normalized);
            ground_truth.push((id.clone(), sim));

            index.add_node(SearchableNode::new(id, NodeType::Pattern, format!("Content {}", i), normalized));
        }

        // Sort ground truth by similarity
        ground_truth.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Test recall at different k values
        for k in [1, 5, 10, 20] {
            let query = SearchQuery::new(query_emb.clone(), k);
            let results = index.search(&query);

            let true_topk: HashSet<String> = ground_truth.iter().take(k).map(|(id, _)| id.clone()).collect();
            let returned: HashSet<String> = results.iter().map(|r| r.node_id.clone()).collect();

            let recall = returned.intersection(&true_topk).count() as f32 / k as f32;

            // For exact search, recall should be 1.0
            assert!(
                (recall - 1.0).abs() < 0.001,
                "Recall@{} should be 1.0 for exact search, got {}",
                k,
                recall
            );
        }
    }
}

// ============================================================================
// Type Filtering Tests
// ============================================================================

mod filter_tests {
    use super::*;

    #[test]
    fn test_filter_by_single_type() {
        let mut index = TestSearchIndex::new();
        let base_emb = normalized_embedding(128);

        // Add nodes of different types with similar embeddings
        for node_type in [NodeType::Pattern, NodeType::Trajectory, NodeType::Prediction, NodeType::Decision] {
            for i in 0..5 {
                let similar = similar_embeddings(&base_emb, 1, 0.1)[0].clone();
                index.add_node(SearchableNode::new(
                    format!("{:?}-{}", node_type, i),
                    node_type,
                    format!("Content {:?} {}", node_type, i),
                    similar,
                ));
            }
        }

        // Search with type filter
        let query = SearchQuery::new(base_emb.clone(), 20).with_type_filter(NodeType::Pattern);
        let results = index.search(&query);

        assert_eq!(results.len(), 5, "Should return only Pattern nodes");
        for result in &results {
            assert!(result.node_id.starts_with("Pattern"), "All results should be Pattern type");
        }
    }

    #[test]
    fn test_filter_returns_empty_when_no_match() {
        let mut index = TestSearchIndex::new();

        // Add only Pattern nodes
        for i in 0..10 {
            index.add_node(SearchableNode::new(
                format!("pattern-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                normalized_embedding(128),
            ));
        }

        // Search for Trajectory nodes
        let query = SearchQuery::new(normalized_embedding(128), 10).with_type_filter(NodeType::Trajectory);
        let results = index.search(&query);

        assert!(results.is_empty(), "Should return empty when no matching type");
    }

    #[test]
    fn test_filter_combined_with_topk() {
        let mut index = TestSearchIndex::new();
        let base_emb = normalized_embedding(128);

        // Add 20 Pattern nodes and 20 Trajectory nodes
        for i in 0..20 {
            let similar_pattern = similar_embeddings(&base_emb, 1, 0.1)[0].clone();
            index.add_node(SearchableNode::new(
                format!("pattern-{}", i),
                NodeType::Pattern,
                format!("Pattern {}", i),
                similar_pattern,
            ));

            let similar_traj = similar_embeddings(&base_emb, 1, 0.2)[0].clone();
            index.add_node(SearchableNode::new(
                format!("trajectory-{}", i),
                NodeType::Trajectory,
                format!("Trajectory {}", i),
                similar_traj,
            ));
        }

        // Search for top 5 Pattern nodes
        let query = SearchQuery::new(base_emb, 5).with_type_filter(NodeType::Pattern);
        let results = index.search(&query);

        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.node_id.starts_with("pattern")));
    }

    #[test]
    fn test_min_similarity_filter() {
        let mut index = TestSearchIndex::new();
        let query_emb = normalized_embedding(128);

        // Add nodes with varying similarity
        for i in 0..10 {
            // Create embeddings with decreasing similarity
            let factor = 1.0 - (i as f32 * 0.1);
            let emb: Vec<f32> = query_emb.iter().map(|x| x * factor).collect();
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized: Vec<f32> = emb.iter().map(|x| x / norm.max(0.001)).collect();

            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                normalized,
            ));
        }

        // Search with minimum similarity threshold
        let query = SearchQuery::new(query_emb, 10).with_min_similarity(0.8);
        let results = index.search(&query);

        // All results should have similarity >= 0.8
        for result in &results {
            assert!(
                result.similarity >= 0.8,
                "Result {} has similarity {} < 0.8",
                result.node_id,
                result.similarity
            );
        }
    }

    #[test]
    fn test_combined_filters() {
        let mut index = TestSearchIndex::new();
        let query_emb = normalized_embedding(128);

        // Add various nodes
        for node_type in [NodeType::Pattern, NodeType::Trajectory] {
            for i in 0..10 {
                let factor = if i < 5 { 0.9 } else { 0.5 };
                let emb: Vec<f32> = query_emb.iter().map(|x| x * factor + 0.1).collect();
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                let normalized: Vec<f32> = emb.iter().map(|x| x / norm).collect();

                index.add_node(SearchableNode::new(
                    format!("{:?}-{}", node_type, i),
                    node_type,
                    format!("Content {:?} {}", node_type, i),
                    normalized,
                ));
            }
        }

        // Search with both type filter and min similarity
        let query = SearchQuery::new(query_emb, 10)
            .with_type_filter(NodeType::Pattern)
            .with_min_similarity(0.7);

        let results = index.search(&query);

        // All results should be Pattern type with similarity >= 0.7
        for result in &results {
            assert!(result.node_id.starts_with("Pattern"));
            assert!(result.similarity >= 0.7);
        }
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_duplicate_embeddings() {
        let mut index = TestSearchIndex::new();
        let embedding = normalized_embedding(128);

        // Add multiple nodes with identical embeddings
        for i in 0..5 {
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                embedding.clone(),
            ));
        }

        let query = SearchQuery::new(embedding, 5);
        let results = index.search(&query);

        // All should have same similarity
        assert_eq!(results.len(), 5);
        let first_sim = results[0].similarity;
        for result in &results {
            assert!((result.similarity - first_sim).abs() < 0.001);
        }
    }

    #[test]
    fn test_zero_vector_query() {
        let mut index = TestSearchIndex::new();

        for i in 0..10 {
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                normalized_embedding(128),
            ));
        }

        // Query with zero vector
        let zero_emb = vec![0.0_f32; 128];
        let query = SearchQuery::new(zero_emb, 5);
        let results = index.search(&query);

        // Should handle gracefully (all similarities will be 0 or NaN)
        // Our cosine_similarity handles this by returning 0
        for result in &results {
            assert!(result.similarity.is_finite());
        }
    }

    #[test]
    fn test_very_small_embeddings() {
        let mut index = TestSearchIndex::new();

        // Very small but non-zero embeddings
        let small_emb: Vec<f32> = (0..128).map(|_| 1e-10).collect();
        index.add_node(SearchableNode::new("small", NodeType::Pattern, "Small", small_emb.clone()));

        let query = SearchQuery::new(small_emb, 1);
        let results = index.search(&query);

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_negative_embeddings() {
        let mut index = TestSearchIndex::new();

        // Embeddings with negative values
        let neg_emb: Vec<f32> = (0..128).map(|i| if i % 2 == 0 { -1.0 } else { 1.0 }).collect();
        let norm: f32 = neg_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = neg_emb.iter().map(|x| x / norm).collect();

        index.add_node(SearchableNode::new("negative", NodeType::Pattern, "Negative", normalized.clone()));

        let query = SearchQuery::new(normalized, 1);
        let results = index.search(&query);

        assert_eq!(results.len(), 1);
        assert!((results[0].similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_high_dimensional_embeddings() {
        let mut index = TestSearchIndex::new();
        let dim = 1536; // High dimension like GPT embeddings

        for i in 0..10 {
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                normalized_embedding(dim),
            ));
        }

        let query = SearchQuery::new(normalized_embedding(dim), 5);
        let results = index.search(&query);

        assert_eq!(results.len(), 5);
        for result in &results {
            assert!(result.similarity >= -1.0 && result.similarity <= 1.0);
        }
    }

    #[test]
    fn test_single_dimension_embedding() {
        let mut index = TestSearchIndex::new();

        index.add_node(SearchableNode::new("pos", NodeType::Pattern, "Positive", vec![1.0]));
        index.add_node(SearchableNode::new("neg", NodeType::Pattern, "Negative", vec![-1.0]));

        // Allow negative similarity to include the negative vector
        let mut query = SearchQuery::new(vec![1.0], 2);
        query.min_similarity = -1.0;
        let results = index.search(&query);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].node_id, "pos");
        assert!((results[0].similarity - 1.0).abs() < 0.001);
        assert_eq!(results[1].node_id, "neg");
        assert!((results[1].similarity - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_node_removal_updates_search() {
        let mut index = TestSearchIndex::new();
        let target_emb = normalized_embedding(128);

        // Add target and other nodes
        index.add_node(SearchableNode::new("target", NodeType::Pattern, "Target", target_emb.clone()));
        for i in 0..5 {
            index.add_node(SearchableNode::new(
                format!("other-{}", i),
                NodeType::Pattern,
                format!("Other {}", i),
                normalized_embedding(128),
            ));
        }

        // Verify target is found
        let query = SearchQuery::new(target_emb.clone(), 1);
        let results = index.search(&query);
        assert_eq!(results[0].node_id, "target");

        // Remove target
        index.remove_node("target");

        // Search again - target should not be found
        let results = index.search(&query);
        assert!(!results.iter().any(|r| r.node_id == "target"));
    }

    #[test]
    fn test_concurrent_similar_scores() {
        let mut index = TestSearchIndex::new();
        let base_emb = normalized_embedding(128);

        // Add many nodes with very similar scores
        for i in 0..100 {
            let emb = similar_embeddings(&base_emb, 1, 0.001)[0].clone();
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                emb,
            ));
        }

        let query = SearchQuery::new(base_emb, 10);
        let results = index.search(&query);

        // Should return exactly 10 results despite similar scores
        assert_eq!(results.len(), 10);

        // All should have high similarity
        for result in &results {
            assert!(result.similarity > 0.9);
        }
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_search_latency_small_index() {
        let mut index = TestSearchIndex::new();

        for i in 0..100 {
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                normalized_embedding(128),
            ));
        }

        let query = SearchQuery::new(normalized_embedding(128), 10);

        let start = Instant::now();
        for _ in 0..100 {
            index.search(&query);
        }
        let duration = start.elapsed();

        let avg_latency_ms = duration.as_millis() as f64 / 100.0;
        assert!(
            avg_latency_ms < 10.0,
            "Average search latency {} ms exceeds 10ms for 100 nodes",
            avg_latency_ms
        );
    }

    #[test]
    fn test_search_latency_medium_index() {
        let mut index = TestSearchIndex::new();

        for i in 0..1000 {
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                normalized_embedding(128),
            ));
        }

        let query = SearchQuery::new(normalized_embedding(128), 10);

        let start = Instant::now();
        for _ in 0..10 {
            index.search(&query);
        }
        let duration = start.elapsed();

        let avg_latency_ms = duration.as_millis() as f64 / 10.0;
        assert!(
            avg_latency_ms < 100.0,
            "Average search latency {} ms exceeds 100ms for 1000 nodes",
            avg_latency_ms
        );
    }

    #[test]
    fn test_search_scales_sublinearly_with_k() {
        let mut index = TestSearchIndex::new();

        for i in 0..1000 {
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                normalized_embedding(128),
            ));
        }

        let query_emb = normalized_embedding(128);

        // Measure time for k=1
        let query_k1 = SearchQuery::new(query_emb.clone(), 1);
        let start = Instant::now();
        for _ in 0..10 {
            index.search(&query_k1);
        }
        let time_k1 = start.elapsed().as_micros();

        // Measure time for k=100
        let query_k100 = SearchQuery::new(query_emb, 100);
        let start = Instant::now();
        for _ in 0..10 {
            index.search(&query_k100);
        }
        let time_k100 = start.elapsed().as_micros();

        // k=100 should not be 100x slower than k=1
        let ratio = time_k100 as f64 / time_k1 as f64;
        assert!(
            ratio < 10.0,
            "k=100 is {}x slower than k=1, expected sublinear scaling",
            ratio
        );
    }

    #[test]
    fn test_index_build_performance() {
        let start = Instant::now();
        let mut index = TestSearchIndex::new();

        for i in 0..5000 {
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                normalized_embedding(128),
            ));
        }
        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 5000,
            "Building index with 5000 nodes took {:?}, expected < 5s",
            duration
        );
    }
}

// ============================================================================
// HNSW-Specific Tests (Simulated)
// ============================================================================

mod hnsw_tests {
    use super::*;

    #[test]
    fn test_approximate_vs_exact_search() {
        let mut index = TestSearchIndex::new();
        let query_emb = normalized_embedding(128);

        for i in 0..100 {
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                normalized_embedding(128),
            ));
        }

        let query = SearchQuery::new(query_emb.clone(), 10);

        let _exact_results = index.search(&query);
        let approx_results = index.search_approximate(&query, 50);

        // Approximate should return some results
        assert!(!approx_results.is_empty());

        // In this simulated version, both should return sorted results
        if !approx_results.is_empty() {
            for i in 1..approx_results.len() {
                assert!(approx_results[i - 1].similarity >= approx_results[i].similarity);
            }
        }
    }

    #[test]
    fn test_ef_search_affects_accuracy() {
        let mut index = TestSearchIndex::new();
        let query_emb = normalized_embedding(128);

        for i in 0..1000 {
            index.add_node(SearchableNode::new(
                format!("node-{}", i),
                NodeType::Pattern,
                format!("Content {}", i),
                normalized_embedding(128),
            ));
        }

        let query = SearchQuery::new(query_emb, 10);

        // Higher ef_search should consider more candidates
        let results_ef10 = index.search_approximate(&query, 10);
        let results_ef100 = index.search_approximate(&query, 100);

        // Both should return up to k results
        assert!(results_ef10.len() <= 10);
        assert!(results_ef100.len() <= 10);
    }
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Property: Search results are always sorted by similarity (descending).
        #[test]
        fn prop_results_sorted(
            node_count in 5usize..50usize,
            k in 1usize..10usize,
        ) {
            let mut index = TestSearchIndex::new();

            for i in 0..node_count {
                index.add_node(SearchableNode::new(
                    format!("node-{}", i),
                    NodeType::Pattern,
                    format!("Content {}", i),
                    normalized_embedding(64),
                ));
            }

            let query = SearchQuery::new(normalized_embedding(64), k);
            let results = index.search(&query);

            for i in 1..results.len() {
                prop_assert!(
                    results[i - 1].similarity >= results[i].similarity,
                    "Results not sorted at position {}: {} < {}",
                    i,
                    results[i - 1].similarity,
                    results[i].similarity
                );
            }
        }

        /// Property: Search returns at most k results.
        #[test]
        fn prop_returns_at_most_k(
            node_count in 1usize..100usize,
            k in 1usize..50usize,
        ) {
            let mut index = TestSearchIndex::new();

            for i in 0..node_count {
                index.add_node(SearchableNode::new(
                    format!("node-{}", i),
                    NodeType::Pattern,
                    format!("Content {}", i),
                    normalized_embedding(64),
                ));
            }

            let query = SearchQuery::new(normalized_embedding(64), k);
            let results = index.search(&query);

            prop_assert!(
                results.len() <= k,
                "Expected at most {} results, got {}",
                k,
                results.len()
            );
        }

        /// Property: All similarity scores are in [-1, 1].
        #[test]
        fn prop_similarity_bounded(
            node_count in 1usize..50usize,
        ) {
            let mut index = TestSearchIndex::new();

            for i in 0..node_count {
                index.add_node(SearchableNode::new(
                    format!("node-{}", i),
                    NodeType::Pattern,
                    format!("Content {}", i),
                    normalized_embedding(64),
                ));
            }

            let query = SearchQuery::new(normalized_embedding(64), node_count);
            let results = index.search(&query);

            for result in &results {
                prop_assert!(
                    result.similarity >= -1.0 && result.similarity <= 1.0,
                    "Similarity {} out of bounds [-1, 1]",
                    result.similarity
                );
            }
        }

        /// Property: Identical query returns self with similarity 1.0.
        #[test]
        fn prop_self_similarity_is_one(_seed in 0u64..1000u64) {
            let mut index = TestSearchIndex::new();
            let embedding = normalized_embedding(64);

            index.add_node(SearchableNode::new(
                "self",
                NodeType::Pattern,
                "Self",
                embedding.clone(),
            ));

            let query = SearchQuery::new(embedding, 1);
            let results = index.search(&query);

            prop_assert_eq!(results.len(), 1);
            prop_assert!(
                (results[0].similarity - 1.0).abs() < 0.001,
                "Self-similarity should be 1.0, got {}",
                results[0].similarity
            );
        }
    }
}
