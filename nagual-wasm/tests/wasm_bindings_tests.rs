//! Comprehensive tests for the WasmProfDAG bindings.
//!
//! These tests verify the public interface of WasmProfDAG using native Rust tests.
//! Note that some WASM-specific functionality (like IndexedDB) cannot be tested
//! without a browser environment.

use nagual_wasm::{Pattern, SearchConfig, VectorSearch};

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a VectorSearch instance for testing (simulates WasmProfDAG core).
fn create_profdag(dim: usize) -> VectorSearch {
    VectorSearch::new(SearchConfig::new().with_embedding_dim(dim))
}

/// Create a test pattern.
fn create_pattern(id: &str, content: &str, embedding: Vec<f32>) -> Pattern {
    Pattern::new(id.to_string(), content.to_string(), embedding)
}

// =============================================================================
// Construction Tests
// =============================================================================

#[test]
fn test_profdag_new_creates_empty_instance() {
    let profdag = create_profdag(128);

    assert!(profdag.is_empty());
    assert_eq!(profdag.len(), 0);
}

#[test]
fn test_profdag_with_config() {
    let config = SearchConfig::new().with_embedding_dim(64).with_min_similarity(0.3);

    let profdag = VectorSearch::new(config);
    let stats = profdag.get_stats();

    assert_eq!(stats.embedding_dim, 64);
    assert_eq!(stats.min_similarity, 0.3);
}

#[test]
fn test_profdag_default_config_values() {
    let profdag = VectorSearch::with_defaults();
    let stats = profdag.get_stats();

    assert_eq!(stats.embedding_dim, 128);
    assert_eq!(stats.min_similarity, 0.0);
    assert_eq!(stats.max_results, 50);
}

// =============================================================================
// add_pattern Tests
// =============================================================================

#[test]
fn test_add_pattern_basic() {
    let mut profdag = create_profdag(4);

    let result = profdag.add_pattern(create_pattern("test-1", "Test content", vec![1.0, 0.0, 0.0, 0.0]));

    assert!(result.is_ok());
    assert_eq!(profdag.len(), 1);
}

#[test]
fn test_add_pattern_multiple() {
    let mut profdag = create_profdag(4);

    for i in 0..10 {
        profdag
            .add_pattern(create_pattern(&format!("p-{}", i), &format!("Pattern {}", i), vec![i as f32, 0.0, 0.0, 0.0]))
            .unwrap();
    }

    assert_eq!(profdag.len(), 10);
}

#[test]
fn test_add_pattern_invalid_dimension_fails() {
    let mut profdag = create_profdag(4);

    // Try to add 3D embedding to 4D index
    let result = profdag.add_pattern(create_pattern("bad", "Bad pattern", vec![1.0, 0.0, 0.0]));

    assert!(result.is_err());
    assert!(profdag.is_empty());
}

#[test]
fn test_add_pattern_preserves_content() {
    let mut profdag = create_profdag(4);

    profdag
        .add_pattern(create_pattern("id-1", "This is the content", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    let pattern = profdag.get_pattern("id-1").unwrap();
    assert_eq!(pattern.content, "This is the content");
}

#[test]
fn test_add_pattern_normalizes_embedding() {
    let mut profdag = create_profdag(4);

    // Add non-normalized embedding
    profdag
        .add_pattern(create_pattern("test", "test", vec![3.0, 4.0, 0.0, 0.0]))
        .unwrap();

    let pattern = profdag.get_pattern("test").unwrap();

    // Check that embedding is normalized (magnitude should be ~1.0)
    let magnitude: f32 = pattern.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((magnitude - 1.0).abs() < 1e-5);
}

// =============================================================================
// remove_pattern Tests
// =============================================================================

#[test]
fn test_remove_pattern_existing() {
    let mut profdag = create_profdag(4);
    profdag
        .add_pattern(create_pattern("to-remove", "content", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    let removed = profdag.remove_pattern("to-remove");

    assert!(removed);
    assert!(profdag.is_empty());
    assert!(profdag.get_pattern("to-remove").is_none());
}

#[test]
fn test_remove_pattern_nonexistent() {
    let mut profdag = create_profdag(4);
    profdag
        .add_pattern(create_pattern("exists", "content", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    let removed = profdag.remove_pattern("does-not-exist");

    assert!(!removed);
    assert_eq!(profdag.len(), 1);
}

#[test]
fn test_remove_pattern_from_empty_index() {
    let mut profdag = create_profdag(4);

    let removed = profdag.remove_pattern("any-id");

    assert!(!removed);
}

#[test]
fn test_remove_one_of_many() {
    let mut profdag = create_profdag(4);

    for i in 0..5 {
        profdag
            .add_pattern(create_pattern(&format!("p{}", i), "content", vec![i as f32, 0.0, 0.0, 0.0]))
            .unwrap();
    }

    profdag.remove_pattern("p2");

    assert_eq!(profdag.len(), 4);
    assert!(profdag.get_pattern("p0").is_some());
    assert!(profdag.get_pattern("p1").is_some());
    assert!(profdag.get_pattern("p2").is_none());
    assert!(profdag.get_pattern("p3").is_some());
    assert!(profdag.get_pattern("p4").is_some());
}

// =============================================================================
// get_pattern Tests
// =============================================================================

#[test]
fn test_get_pattern_existing() {
    let mut profdag = create_profdag(4);
    profdag
        .add_pattern(create_pattern("my-id", "my content", vec![0.5, 0.5, 0.0, 0.0]))
        .unwrap();

    let pattern = profdag.get_pattern("my-id");

    assert!(pattern.is_some());
    let p = pattern.unwrap();
    assert_eq!(p.id, "my-id");
    assert_eq!(p.content, "my content");
}

#[test]
fn test_get_pattern_nonexistent() {
    let profdag = create_profdag(4);

    let pattern = profdag.get_pattern("nonexistent");

    assert!(pattern.is_none());
}

#[test]
fn test_get_pattern_returns_reference() {
    let mut profdag = create_profdag(4);
    profdag
        .add_pattern(create_pattern("test", "content", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    // Getting pattern doesn't remove it
    let _pattern1 = profdag.get_pattern("test");
    let pattern2 = profdag.get_pattern("test");

    assert!(pattern2.is_some());
    assert_eq!(profdag.len(), 1);
}

// =============================================================================
// search Tests
// =============================================================================

#[test]
fn test_search_returns_correct_results() {
    let mut profdag = create_profdag(4);

    profdag
        .add_pattern(create_pattern("x", "x-axis", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    profdag
        .add_pattern(create_pattern("y", "y-axis", vec![0.0, 1.0, 0.0, 0.0]))
        .unwrap();
    profdag
        .add_pattern(create_pattern("z", "z-axis", vec![0.0, 0.0, 1.0, 0.0]))
        .unwrap();

    let results = profdag.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "x");
    assert!(results[0].similarity > 0.99);
}

#[test]
fn test_search_respects_top_k() {
    let mut profdag = create_profdag(4);

    for i in 0..10 {
        profdag
            .add_pattern(create_pattern(&format!("p{}", i), "content", vec![1.0 - i as f32 * 0.05, i as f32 * 0.05, 0.0, 0.0]))
            .unwrap();
    }

    let results = profdag.search(&[1.0, 0.0, 0.0, 0.0], 3).unwrap();

    assert_eq!(results.len(), 3);
}

#[test]
fn test_search_ordered_by_similarity() {
    let mut profdag = create_profdag(4);

    profdag
        .add_pattern(create_pattern("far", "far", vec![0.0, 1.0, 0.0, 0.0]))
        .unwrap();
    profdag
        .add_pattern(create_pattern("close", "close", vec![0.9, 0.1, 0.0, 0.0]))
        .unwrap();
    profdag
        .add_pattern(create_pattern("exact", "exact", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    let results = profdag.search(&[1.0, 0.0, 0.0, 0.0], 3).unwrap();

    assert_eq!(results[0].id, "exact");
    assert_eq!(results[1].id, "close");
    assert_eq!(results[2].id, "far");

    assert!(results[0].similarity >= results[1].similarity);
    assert!(results[1].similarity >= results[2].similarity);
}

#[test]
fn test_search_on_empty_index() {
    let profdag = create_profdag(4);

    let results = profdag.search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();

    assert!(results.is_empty());
}

#[test]
fn test_search_with_zero_top_k() {
    let mut profdag = create_profdag(4);
    profdag
        .add_pattern(create_pattern("test", "content", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    let results = profdag.search(&[1.0, 0.0, 0.0, 0.0], 0).unwrap();

    assert!(results.is_empty());
}

#[test]
fn test_search_dimension_mismatch_fails() {
    let mut profdag = create_profdag(4);
    profdag
        .add_pattern(create_pattern("test", "content", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    let result = profdag.search(&[1.0, 0.0, 0.0], 10);

    assert!(result.is_err());
}

// =============================================================================
// export_json / import_json Tests
// =============================================================================

#[test]
fn test_export_json_empty() {
    let profdag = create_profdag(4);

    let json = profdag.export_json().unwrap();

    assert_eq!(json, "[]");
}

#[test]
fn test_export_json_with_patterns() {
    let mut profdag = create_profdag(4);
    profdag
        .add_pattern(create_pattern("p1", "content 1", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    let json = profdag.export_json().unwrap();

    assert!(json.contains("p1"));
    assert!(json.contains("content 1"));
}

#[test]
fn test_import_json_empty_array() {
    let mut profdag = create_profdag(4);

    let count = profdag.import_json("[]").unwrap();

    assert_eq!(count, 0);
    assert!(profdag.is_empty());
}

#[test]
fn test_import_json_single_pattern() {
    let mut profdag = create_profdag(4);

    let json = r#"[{
        "id": "imported",
        "content": "imported content",
        "embedding": [0.5, 0.5, 0.0, 0.0]
    }]"#;

    let count = profdag.import_json(json).unwrap();

    assert_eq!(count, 1);
    assert_eq!(profdag.len(), 1);

    let pattern = profdag.get_pattern("imported").unwrap();
    assert_eq!(pattern.content, "imported content");
}

#[test]
fn test_import_json_multiple_patterns() {
    let mut profdag = create_profdag(4);

    let json = r#"[
        {"id": "p1", "content": "c1", "embedding": [1.0, 0.0, 0.0, 0.0]},
        {"id": "p2", "content": "c2", "embedding": [0.0, 1.0, 0.0, 0.0]},
        {"id": "p3", "content": "c3", "embedding": [0.0, 0.0, 1.0, 0.0]}
    ]"#;

    let count = profdag.import_json(json).unwrap();

    assert_eq!(count, 3);
    assert!(profdag.get_pattern("p1").is_some());
    assert!(profdag.get_pattern("p2").is_some());
    assert!(profdag.get_pattern("p3").is_some());
}

#[test]
fn test_export_import_roundtrip() {
    let mut profdag1 = create_profdag(4);

    profdag1
        .add_pattern(create_pattern("p1", "content 1", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    profdag1
        .add_pattern(create_pattern("p2", "content 2", vec![0.0, 1.0, 0.0, 0.0]))
        .unwrap();

    let json = profdag1.export_json().unwrap();

    let mut profdag2 = create_profdag(4);
    profdag2.import_json(&json).unwrap();

    assert_eq!(profdag2.len(), 2);

    // Search should work correctly
    let results = profdag2.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(results[0].id, "p1");
}

#[test]
fn test_import_json_invalid_fails() {
    let mut profdag = create_profdag(4);

    let result = profdag.import_json("not json");

    assert!(result.is_err());
}

#[test]
fn test_import_json_wrong_dimension_fails() {
    let mut profdag = create_profdag(4);

    // Embedding has 3 dimensions, but index expects 4
    let json = r#"[{"id": "p1", "content": "c1", "embedding": [1.0, 0.0, 0.0]}]"#;

    let result = profdag.import_json(json);

    assert!(result.is_err());
}

// =============================================================================
// pattern_count / is_empty Tests
// =============================================================================

#[test]
fn test_pattern_count_empty() {
    let profdag = create_profdag(4);

    assert_eq!(profdag.len(), 0);
    assert!(profdag.is_empty());
}

#[test]
fn test_pattern_count_after_add() {
    let mut profdag = create_profdag(4);

    for i in 0..5 {
        profdag
            .add_pattern(create_pattern(&format!("p{}", i), "content", vec![i as f32, 0.0, 0.0, 0.0]))
            .unwrap();

        assert_eq!(profdag.len(), i + 1);
        assert!(!profdag.is_empty());
    }
}

#[test]
fn test_pattern_count_after_remove() {
    let mut profdag = create_profdag(4);

    for i in 0..5 {
        profdag
            .add_pattern(create_pattern(&format!("p{}", i), "content", vec![i as f32, 0.0, 0.0, 0.0]))
            .unwrap();
    }

    profdag.remove_pattern("p2");
    assert_eq!(profdag.len(), 4);

    profdag.remove_pattern("p0");
    assert_eq!(profdag.len(), 3);
}

#[test]
fn test_is_empty_transitions() {
    let mut profdag = create_profdag(4);

    assert!(profdag.is_empty());

    profdag
        .add_pattern(create_pattern("p1", "content", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    assert!(!profdag.is_empty());

    profdag.remove_pattern("p1");
    assert!(profdag.is_empty());
}

// =============================================================================
// clear Tests
// =============================================================================

#[test]
fn test_clear_empty_index() {
    let mut profdag = create_profdag(4);

    profdag.clear();

    assert!(profdag.is_empty());
}

#[test]
fn test_clear_with_patterns() {
    let mut profdag = create_profdag(4);

    for i in 0..10 {
        profdag
            .add_pattern(create_pattern(&format!("p{}", i), "content", vec![i as f32, 0.0, 0.0, 0.0]))
            .unwrap();
    }

    assert_eq!(profdag.len(), 10);

    profdag.clear();

    assert!(profdag.is_empty());
    assert_eq!(profdag.len(), 0);

    // Can still add patterns after clear
    profdag
        .add_pattern(create_pattern("new", "content", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    assert_eq!(profdag.len(), 1);
}

// =============================================================================
// get_stats Tests
// =============================================================================

#[test]
fn test_get_stats_empty() {
    let profdag = VectorSearch::new(SearchConfig::new().with_embedding_dim(64).with_min_similarity(0.5).with_max_results(25));

    let stats = profdag.get_stats();

    assert_eq!(stats.pattern_count, 0);
    assert_eq!(stats.embedding_dim, 64);
    assert_eq!(stats.min_similarity, 0.5);
    assert_eq!(stats.max_results, 25);
}

#[test]
fn test_get_stats_with_patterns() {
    let mut profdag = create_profdag(128);

    for i in 0..50 {
        let mut embedding = vec![0.0; 128];
        embedding[i % 128] = 1.0;
        profdag
            .add_pattern(create_pattern(&format!("p{}", i), "content", embedding))
            .unwrap();
    }

    let stats = profdag.get_stats();

    assert_eq!(stats.pattern_count, 50);
    assert_eq!(stats.embedding_dim, 128);
}

// =============================================================================
// Batch Search Tests
// =============================================================================

#[test]
fn test_batch_search_empty_queries() {
    let profdag = create_profdag(4);
    let queries: Vec<Vec<f32>> = vec![];

    let results = profdag.batch_search(&queries, 10).unwrap();

    assert!(results.is_empty());
}

#[test]
fn test_batch_search_multiple_queries() {
    let mut profdag = create_profdag(4);

    profdag
        .add_pattern(create_pattern("x", "x", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    profdag
        .add_pattern(create_pattern("y", "y", vec![0.0, 1.0, 0.0, 0.0]))
        .unwrap();
    profdag
        .add_pattern(create_pattern("z", "z", vec![0.0, 0.0, 1.0, 0.0]))
        .unwrap();

    let queries = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];

    let results = profdag.batch_search(&queries, 1).unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0][0].id, "x");
    assert_eq!(results[1][0].id, "y");
    assert_eq!(results[2][0].id, "z");
}

// =============================================================================
// Integration Scenarios
// =============================================================================

#[test]
fn test_full_workflow() {
    let mut profdag = create_profdag(4);

    // 1. Add patterns
    profdag
        .add_pattern(create_pattern("pattern-1", "First pattern", vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    profdag
        .add_pattern(create_pattern("pattern-2", "Second pattern", vec![0.0, 1.0, 0.0, 0.0]))
        .unwrap();
    profdag
        .add_pattern(create_pattern("pattern-3", "Third pattern", vec![0.5, 0.5, 0.0, 0.0]))
        .unwrap();

    assert_eq!(profdag.len(), 3);

    // 2. Search
    let results = profdag.search(&[0.8, 0.2, 0.0, 0.0], 2).unwrap();
    assert_eq!(results.len(), 2);

    // 3. Export
    let json = profdag.export_json().unwrap();

    // 4. Create new instance and import
    let mut profdag2 = create_profdag(4);
    profdag2.import_json(&json).unwrap();

    // 5. Verify import worked
    assert_eq!(profdag2.len(), 3);

    let results2 = profdag2.search(&[0.8, 0.2, 0.0, 0.0], 2).unwrap();
    assert_eq!(results2.len(), 2);

    // 6. Remove and verify
    profdag2.remove_pattern("pattern-2");
    assert_eq!(profdag2.len(), 2);
    assert!(profdag2.get_pattern("pattern-2").is_none());

    // 7. Clear and verify
    profdag2.clear();
    assert!(profdag2.is_empty());
}

#[test]
fn test_high_dimensional_workflow() {
    let dim = 256;
    let mut profdag = VectorSearch::new(SearchConfig::new().with_embedding_dim(dim));

    // Add patterns with random-ish embeddings
    for i in 0..100 {
        let mut embedding = vec![0.0; dim];
        // Create a pattern that has activity in specific dimensions
        for j in 0..10 {
            embedding[(i * 7 + j * 13) % dim] = 1.0;
        }
        profdag
            .add_pattern(create_pattern(&format!("high-dim-{}", i), &format!("Pattern {}", i), embedding))
            .unwrap();
    }

    assert_eq!(profdag.len(), 100);

    // Search
    let mut query = vec![0.0; dim];
    query[0] = 1.0;
    query[7] = 1.0;

    let results = profdag.search(&query, 10).unwrap();
    assert_eq!(results.len(), 10);

    // Export and import
    let json = profdag.export_json().unwrap();
    let mut profdag2 = VectorSearch::new(SearchConfig::new().with_embedding_dim(dim));
    profdag2.import_json(&json).unwrap();

    assert_eq!(profdag2.len(), 100);
}
