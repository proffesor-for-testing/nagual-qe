//! Comprehensive tests for the search module.
//!
//! These tests verify vector similarity search, top-k retrieval,
//! cosine similarity correctness, and edge cases.

use nagual_wasm::{Pattern, SearchConfig, VectorSearch};

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a normalized vector from the given components.
fn create_normalized_vector(components: &[f32]) -> Vec<f32> {
    let norm: f32 = components.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        components.iter().map(|x| x / norm).collect()
    } else {
        components.to_vec()
    }
}

/// Create a pattern with the given parameters.
fn create_pattern(id: &str, content: &str, embedding: Vec<f32>) -> Pattern {
    Pattern::new(id.to_string(), content.to_string(), embedding)
}

/// Create a simple 4D search engine for testing.
fn create_4d_search() -> VectorSearch {
    VectorSearch::new(SearchConfig::new().with_embedding_dim(4))
}

// =============================================================================
// SearchConfig Tests
// =============================================================================

#[test]
fn test_search_config_defaults() {
    let config = SearchConfig::new();
    assert_eq!(config.embedding_dim(), 128);
    assert_eq!(config.min_similarity(), 0.0);
    assert_eq!(config.max_results(), 50);
}

#[test]
fn test_search_config_builder_pattern() {
    let config = SearchConfig::new()
        .with_embedding_dim(256)
        .with_min_similarity(0.5)
        .with_max_results(100);

    assert_eq!(config.embedding_dim(), 256);
    assert_eq!(config.min_similarity(), 0.5);
    assert_eq!(config.max_results(), 100);
}

#[test]
fn test_search_config_min_similarity_clamping() {
    // Test that min_similarity is clamped to [0, 1]
    let config_low = SearchConfig::new().with_min_similarity(-0.5);
    assert_eq!(config_low.min_similarity(), 0.0);

    let config_high = SearchConfig::new().with_min_similarity(1.5);
    assert_eq!(config_high.min_similarity(), 1.0);

    let config_valid = SearchConfig::new().with_min_similarity(0.7);
    assert_eq!(config_valid.min_similarity(), 0.7);
}

// =============================================================================
// Pattern Tests
// =============================================================================

#[test]
fn test_pattern_creation() {
    let pattern = create_pattern("test-id", "test content", vec![1.0, 0.0, 0.0, 0.0]);

    assert_eq!(pattern.id, "test-id");
    assert_eq!(pattern.content, "test content");
    assert_eq!(pattern.embedding, vec![1.0, 0.0, 0.0, 0.0]);
    assert_eq!(pattern.pattern_type, "pattern");
    assert_eq!(pattern.confidence, 0.5);
}

#[test]
fn test_pattern_embedding_validation() {
    let pattern_4d = create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]);
    assert!(pattern_4d.has_valid_embedding(4));
    assert!(!pattern_4d.has_valid_embedding(3));
    assert!(!pattern_4d.has_valid_embedding(5));

    let pattern_128d = create_pattern("p2", "test", vec![0.0; 128]);
    assert!(pattern_128d.has_valid_embedding(128));
    assert!(!pattern_128d.has_valid_embedding(4));
}

// =============================================================================
// VectorSearch Basic Operations Tests
// =============================================================================

#[test]
fn test_vector_search_new() {
    let search = VectorSearch::with_defaults();
    assert!(search.is_empty());
    assert_eq!(search.len(), 0);
}

#[test]
fn test_vector_search_add_pattern() {
    let mut search = create_4d_search();

    let result = search.add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]));
    assert!(result.is_ok());
    assert_eq!(search.len(), 1);
    assert!(!search.is_empty());
}

#[test]
fn test_vector_search_add_multiple_patterns() {
    let mut search = create_4d_search();

    for i in 0..10 {
        let embedding = vec![i as f32, 0.0, 0.0, 0.0];
        search
            .add_pattern(create_pattern(&format!("p{}", i), "test", embedding))
            .unwrap();
    }

    assert_eq!(search.len(), 10);
}

#[test]
fn test_vector_search_dimension_mismatch() {
    let mut search = create_4d_search();

    // Try to add a 3D pattern to a 4D search
    let result = search.add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0]));
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    assert!(err_msg.contains("dimension mismatch"));

    // Try to add a 5D pattern to a 4D search
    let result = search.add_pattern(create_pattern("p2", "test", vec![1.0, 0.0, 0.0, 0.0, 0.0]));
    assert!(result.is_err());

    // Index should still be empty
    assert!(search.is_empty());
}

#[test]
fn test_vector_search_remove_pattern() {
    let mut search = create_4d_search();

    search
        .add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    search
        .add_pattern(create_pattern("p2", "test", vec![0.0, 1.0, 0.0, 0.0]))
        .unwrap();

    assert_eq!(search.len(), 2);

    // Remove existing pattern
    let removed = search.remove_pattern("p1");
    assert!(removed);
    assert_eq!(search.len(), 1);

    // Try to remove non-existent pattern
    let removed = search.remove_pattern("p1");
    assert!(!removed);
    assert_eq!(search.len(), 1);

    // Remove last pattern
    let removed = search.remove_pattern("p2");
    assert!(removed);
    assert!(search.is_empty());
}

#[test]
fn test_vector_search_get_pattern() {
    let mut search = create_4d_search();

    search
        .add_pattern(create_pattern("p1", "content 1", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    let pattern = search.get_pattern("p1");
    assert!(pattern.is_some());
    assert_eq!(pattern.unwrap().content, "content 1");

    let pattern = search.get_pattern("nonexistent");
    assert!(pattern.is_none());
}

#[test]
fn test_vector_search_clear() {
    let mut search = create_4d_search();

    for i in 0..5 {
        search
            .add_pattern(create_pattern(&format!("p{}", i), "test", vec![i as f32, 0.0, 0.0, 0.0]))
            .unwrap();
    }

    assert_eq!(search.len(), 5);

    search.clear();
    assert!(search.is_empty());
    assert_eq!(search.len(), 0);
}

// =============================================================================
// Vector Similarity Search Tests
// =============================================================================

#[test]
fn test_search_empty_index() {
    let search = create_4d_search();
    let results = search.search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_search_single_element() {
    let mut search = create_4d_search();
    search
        .add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    // Identical query
    let results = search.search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "p1");
    // Similarity should be close to 1.0 (vectors are normalized)
    assert!(results[0].similarity > 0.99);
}

#[test]
fn test_search_top_k_retrieval() {
    let mut search = create_4d_search();

    // Add patterns with varying similarity to [1, 0, 0, 0]
    search
        .add_pattern(create_pattern("exact", "exact match", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    search
        .add_pattern(create_pattern("close", "close match", vec![0.9, 0.1, 0.0, 0.0]))
        .unwrap();
    search
        .add_pattern(create_pattern("medium", "medium match", vec![0.7, 0.3, 0.0, 0.0]))
        .unwrap();
    search
        .add_pattern(create_pattern("far", "far match", vec![0.5, 0.5, 0.0, 0.0]))
        .unwrap();
    search
        .add_pattern(create_pattern("orthogonal", "orthogonal", vec![0.0, 1.0, 0.0, 0.0]))
        .unwrap();

    // Request top-3
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = search.search(&query, 3).unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].id, "exact");
    assert_eq!(results[1].id, "close");
    assert_eq!(results[2].id, "medium");

    // Verify descending similarity order
    assert!(results[0].similarity >= results[1].similarity);
    assert!(results[1].similarity >= results[2].similarity);
}

#[test]
fn test_search_top_k_larger_than_index() {
    let mut search = create_4d_search();

    search
        .add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    search
        .add_pattern(create_pattern("p2", "test", vec![0.0, 1.0, 0.0, 0.0]))
        .unwrap();

    // Request more than available
    let results = search.search(&[1.0, 0.0, 0.0, 0.0], 100).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_search_top_k_zero() {
    let mut search = create_4d_search();
    search
        .add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    let results = search.search(&[1.0, 0.0, 0.0, 0.0], 0).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_search_query_dimension_mismatch() {
    let mut search = create_4d_search();
    search
        .add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    // Wrong dimension query
    let result = search.search(&[1.0, 0.0, 0.0], 10);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("dimension mismatch"));
}

// =============================================================================
// Cosine Similarity Correctness Tests
// =============================================================================

#[test]
fn test_cosine_similarity_identical_vectors() {
    let mut search = create_4d_search();
    search
        .add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    let results = search.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    // Identical normalized vectors should have similarity ~1.0
    assert!((results[0].similarity - 1.0).abs() < 1e-5);
}

#[test]
fn test_cosine_similarity_orthogonal_vectors() {
    let mut search = create_4d_search();
    search
        .add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    // Orthogonal query
    let results = search.search(&[0.0, 1.0, 0.0, 0.0], 1).unwrap();
    // Orthogonal vectors should have similarity ~0.0
    assert!(results[0].similarity.abs() < 1e-5);
}

#[test]
fn test_cosine_similarity_opposite_vectors() {
    // Note: Default min_similarity is 0.0, which filters out negative similarities.
    // This test verifies the behavior: opposite vectors are filtered out by default.
    let mut search = create_4d_search();
    search
        .add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    // Opposite direction query - with default min_similarity=0.0, this will be filtered
    let results = search.search(&[-1.0, 0.0, 0.0, 0.0], 1).unwrap();
    // Result should be empty since the only pattern has negative similarity
    assert!(results.is_empty(), "Negative similarity should be filtered with default min_similarity=0.0");
}

#[test]
fn test_cosine_similarity_opposite_vectors_with_negative_threshold() {
    // Use a negative min_similarity to allow opposite vectors
    let mut search = VectorSearch::new(
        SearchConfig::new()
            .with_embedding_dim(4)
            .with_min_similarity(0.0) // min_similarity is clamped to 0.0-1.0, so we can only test that
    );
    search
        .add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    // With min_similarity=0.0, opposite vectors (similarity=-1) are filtered out
    // This is expected behavior - the search only returns non-negative similarities
    let results = search.search(&[-1.0, 0.0, 0.0, 0.0], 1).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_cosine_similarity_45_degree() {
    let mut search = create_4d_search();
    search
        .add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    // 45 degree query (equal parts of x and y)
    let query = create_normalized_vector(&[1.0, 1.0, 0.0, 0.0]);
    let results = search.search(&query, 1).unwrap();

    // cos(45) = 1/sqrt(2) approx 0.707
    let expected = 1.0 / 2.0_f32.sqrt();
    assert!((results[0].similarity - expected).abs() < 1e-5);
}

#[test]
fn test_similarity_ordering() {
    let mut search = create_4d_search();

    // Add patterns at known angles from [1,0,0,0]
    let patterns = vec![
        ("angle_0", vec![1.0, 0.0, 0.0, 0.0]),   // 0 degrees
        ("angle_30", create_normalized_vector(&[3.0_f32.sqrt(), 1.0, 0.0, 0.0])), // ~30 degrees
        ("angle_45", create_normalized_vector(&[1.0, 1.0, 0.0, 0.0])),             // 45 degrees
        ("angle_60", create_normalized_vector(&[1.0, 3.0_f32.sqrt(), 0.0, 0.0])), // ~60 degrees
        ("angle_90", vec![0.0, 1.0, 0.0, 0.0]),  // 90 degrees
    ];

    for (id, emb) in patterns {
        search
            .add_pattern(create_pattern(id, "test", emb))
            .unwrap();
    }

    let results = search.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();

    // Verify ordering by angle (smaller angle = higher similarity)
    assert_eq!(results[0].id, "angle_0");
    assert_eq!(results[1].id, "angle_30");
    assert_eq!(results[2].id, "angle_45");
    assert_eq!(results[3].id, "angle_60");
    assert_eq!(results[4].id, "angle_90");
}

// =============================================================================
// Minimum Similarity Threshold Tests
// =============================================================================

#[test]
fn test_min_similarity_filtering() {
    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(4).with_min_similarity(0.5));

    // Add patterns with varying similarity to [1, 0, 0, 0]
    search
        .add_pattern(create_pattern("high", "high similarity", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    search
        .add_pattern(create_pattern("medium", "medium similarity", vec![0.6, 0.4, 0.0, 0.0]))
        .unwrap();
    search
        .add_pattern(create_pattern("low", "low similarity", vec![0.0, 1.0, 0.0, 0.0]))
        .unwrap();

    let results = search.search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();

    // Only patterns with similarity >= 0.5 should be returned
    assert!(results.len() <= 2);
    for result in &results {
        assert!(result.similarity >= 0.5);
    }
}

// =============================================================================
// Batch Search Tests
// =============================================================================

#[test]
fn test_batch_search_empty() {
    let search = create_4d_search();
    let queries: Vec<Vec<f32>> = vec![];
    let results = search.batch_search(&queries, 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_batch_search_single_query() {
    let mut search = create_4d_search();
    search
        .add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    let queries = vec![vec![1.0, 0.0, 0.0, 0.0]];
    let results = search.batch_search(&queries, 10).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].len(), 1);
    assert_eq!(results[0][0].id, "p1");
}

#[test]
fn test_batch_search_multiple_queries() {
    let mut search = create_4d_search();
    search
        .add_pattern(create_pattern("x", "x-axis", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    search
        .add_pattern(create_pattern("y", "y-axis", vec![0.0, 1.0, 0.0, 0.0]))
        .unwrap();
    search
        .add_pattern(create_pattern("z", "z-axis", vec![0.0, 0.0, 1.0, 0.0]))
        .unwrap();

    let queries = vec![
        vec![1.0, 0.0, 0.0, 0.0], // Should find x
        vec![0.0, 1.0, 0.0, 0.0], // Should find y
        vec![0.0, 0.0, 1.0, 0.0], // Should find z
    ];
    let results = search.batch_search(&queries, 1).unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0][0].id, "x");
    assert_eq!(results[1][0].id, "y");
    assert_eq!(results[2][0].id, "z");
}

#[test]
fn test_batch_search_dimension_error() {
    let mut search = create_4d_search();
    search
        .add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    // One query has wrong dimension
    let queries = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0], // Wrong!
    ];
    let result = search.batch_search(&queries, 10);

    assert!(result.is_err());
}

// =============================================================================
// Search Statistics Tests
// =============================================================================

#[test]
fn test_get_stats() {
    let mut search = VectorSearch::new(
        SearchConfig::new()
            .with_embedding_dim(64)
            .with_min_similarity(0.3)
            .with_max_results(25),
    );

    for i in 0..5 {
        search
            .add_pattern(create_pattern(&format!("p{}", i), "test", vec![0.0; 64]))
            .unwrap();
    }

    let stats = search.get_stats();
    assert_eq!(stats.pattern_count, 5);
    assert_eq!(stats.embedding_dim, 64);
    assert_eq!(stats.min_similarity, 0.3);
    assert_eq!(stats.max_results, 25);
}

// =============================================================================
// Large Scale Tests
// =============================================================================

#[test]
fn test_search_with_many_patterns() {
    let dim = 32;
    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(dim));

    // Add 1000 patterns
    for i in 0..1000 {
        let mut embedding = vec![0.0; dim];
        embedding[i % dim] = 1.0;
        search
            .add_pattern(create_pattern(&format!("p{}", i), &format!("pattern {}", i), embedding))
            .unwrap();
    }

    assert_eq!(search.len(), 1000);

    // Search should still work
    let mut query = vec![0.0; dim];
    query[0] = 1.0;
    let results = search.search(&query, 10).unwrap();

    assert_eq!(results.len(), 10);
    // The most similar should be patterns where embedding[0] = 1.0
    // (p0, p32, p64, etc.)
    assert!(results[0].similarity > 0.9);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_search_with_near_zero_vector() {
    let mut search = create_4d_search();
    search
        .add_pattern(create_pattern("p1", "test", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    // Very small but non-zero query vector
    let results = search.search(&[1e-10, 1e-10, 1e-10, 1e-10], 1).unwrap();
    // Should return a result (normalization handles small vectors)
    assert_eq!(results.len(), 1);
}

#[test]
fn test_duplicate_pattern_ids() {
    let mut search = create_4d_search();

    search
        .add_pattern(create_pattern("duplicate", "first", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    search
        .add_pattern(create_pattern("duplicate", "second", vec![0.0, 1.0, 0.0, 0.0]))
        .unwrap();

    // Both are added (no unique constraint at this level)
    assert_eq!(search.len(), 2);

    // get_pattern returns the first one found
    let pattern = search.get_pattern("duplicate");
    assert!(pattern.is_some());
}

#[test]
fn test_search_respects_max_results_config() {
    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(4).with_max_results(3));

    for i in 0..10 {
        search
            .add_pattern(create_pattern(&format!("p{}", i), "test", vec![1.0, i as f32 * 0.01, 0.0, 0.0]))
            .unwrap();
    }

    // Request more than max_results
    let results = search.search(&[1.0, 0.0, 0.0, 0.0], 100).unwrap();

    // Should be capped at max_results
    assert_eq!(results.len(), 3);
}

// =============================================================================
// Higher Dimensional Tests
// =============================================================================

#[test]
fn test_search_128_dimensions() {
    let dim = 128;
    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(dim));

    // Create patterns along different axes
    for i in 0..10 {
        let mut embedding = vec![0.0; dim];
        embedding[i] = 1.0;
        search
            .add_pattern(create_pattern(&format!("axis_{}", i), "test", embedding))
            .unwrap();
    }

    // Query along axis 5
    let mut query = vec![0.0; dim];
    query[5] = 1.0;

    let results = search.search(&query, 1).unwrap();
    assert_eq!(results[0].id, "axis_5");
    assert!(results[0].similarity > 0.99);
}

#[test]
fn test_search_mixed_similarity_128d() {
    let dim = 128;
    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(dim));

    // Create a reference pattern
    let reference: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
    let ref_norm: f32 = reference.iter().map(|x| x * x).sum::<f32>().sqrt();
    let reference: Vec<f32> = reference.iter().map(|x| x / ref_norm).collect();

    search
        .add_pattern(create_pattern("reference", "reference pattern", reference.clone()))
        .unwrap();

    // Create a slightly perturbed version
    let perturbed: Vec<f32> = reference.iter().map(|x| x + 0.01).collect();
    let p_norm: f32 = perturbed.iter().map(|x| x * x).sum::<f32>().sqrt();
    let perturbed: Vec<f32> = perturbed.iter().map(|x| x / p_norm).collect();

    search
        .add_pattern(create_pattern("perturbed", "perturbed pattern", perturbed))
        .unwrap();

    // Create an orthogonal pattern
    let orthogonal: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
    let o_norm: f32 = orthogonal.iter().map(|x| x * x).sum::<f32>().sqrt();
    let orthogonal: Vec<f32> = orthogonal.iter().map(|x| x / o_norm).collect();

    search
        .add_pattern(create_pattern("orthogonal", "orthogonal pattern", orthogonal))
        .unwrap();

    // Query with the reference vector
    let results = search.search(&reference, 3).unwrap();

    // Reference should be most similar to itself, then perturbed, then orthogonal
    assert_eq!(results[0].id, "reference");
    assert!(results[0].similarity > results[1].similarity);
    assert!(results[1].similarity > results[2].similarity);
}
