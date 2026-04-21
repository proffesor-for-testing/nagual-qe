//! Comprehensive tests for the storage module.
//!
//! These tests focus on the non-WASM aspects of storage:
//! - StorageError creation and conversion
//! - StorageStats creation and access
//! - Pattern serialization/deserialization for storage
//!
//! Note: IndexedDB operations require a browser environment and cannot
//! be tested in native Rust tests. Those would require wasm-bindgen-test.

use nagual_wasm::{Pattern, SearchConfig, StorageStats, VectorSearch};

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a test pattern with the given parameters.
fn create_test_pattern(id: &str, content: &str, dim: usize) -> Pattern {
    let embedding = vec![0.1; dim];
    Pattern::new(id.to_string(), content.to_string(), embedding)
}

// =============================================================================
// StorageStats Tests
// =============================================================================

#[test]
fn test_storage_stats_creation() {
    let stats = StorageStats::new(100, true);

    assert_eq!(stats.pattern_count(), 100);
    assert!(stats.connected());
    assert_eq!(stats.db_name(), "nagual_profdag");
    assert_eq!(stats.db_version(), 1);
}

#[test]
fn test_storage_stats_disconnected() {
    let stats = StorageStats::new(0, false);

    assert_eq!(stats.pattern_count(), 0);
    assert!(!stats.connected());
}

#[test]
fn test_storage_stats_various_counts() {
    let counts = [0, 1, 100, 1000, 10000, u32::MAX];

    for count in counts {
        let stats = StorageStats::new(count, true);
        assert_eq!(stats.pattern_count(), count);
    }
}

// =============================================================================
// Pattern JSON Serialization Tests (for storage roundtrip)
// =============================================================================

#[test]
fn test_pattern_serialization_basic() {
    let pattern = create_test_pattern("test-id", "test content", 4);

    let json = serde_json::to_string(&pattern).expect("Failed to serialize pattern");
    let deserialized: Pattern = serde_json::from_str(&json).expect("Failed to deserialize pattern");

    assert_eq!(deserialized.id, pattern.id);
    assert_eq!(deserialized.content, pattern.content);
    assert_eq!(deserialized.embedding.len(), pattern.embedding.len());
}

#[test]
fn test_pattern_serialization_with_all_fields() {
    let mut pattern = create_test_pattern("full-pattern", "full content", 8);
    pattern.pattern_type = "trajectory".to_string();
    pattern.confidence = 0.85;
    pattern.metadata = serde_json::json!({
        "source": "test",
        "version": 1
    });

    let json = serde_json::to_string(&pattern).expect("Failed to serialize");
    let deserialized: Pattern = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(deserialized.pattern_type, "trajectory");
    assert_eq!(deserialized.confidence, 0.85);
    assert_eq!(deserialized.metadata["source"], "test");
    assert_eq!(deserialized.metadata["version"], 1);
}

#[test]
fn test_pattern_deserialization_with_defaults() {
    // Minimal JSON without optional fields
    let json = r#"{
        "id": "minimal",
        "content": "minimal pattern",
        "embedding": [0.1, 0.2, 0.3, 0.4]
    }"#;

    let pattern: Pattern = serde_json::from_str(json).expect("Failed to deserialize");

    assert_eq!(pattern.id, "minimal");
    assert_eq!(pattern.content, "minimal pattern");
    assert_eq!(pattern.embedding, vec![0.1, 0.2, 0.3, 0.4]);
    // Check defaults are applied
    assert_eq!(pattern.pattern_type, "pattern");
    assert_eq!(pattern.confidence, 0.5);
}

#[test]
fn test_pattern_collection_serialization() {
    let patterns: Vec<Pattern> = (0..10)
        .map(|i| create_test_pattern(&format!("p{}", i), &format!("content {}", i), 4))
        .collect();

    let json = serde_json::to_string(&patterns).expect("Failed to serialize");
    let deserialized: Vec<Pattern> = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(deserialized.len(), 10);
    for (i, p) in deserialized.iter().enumerate() {
        assert_eq!(p.id, format!("p{}", i));
    }
}

#[test]
fn test_pattern_with_unicode_content() {
    let pattern = create_test_pattern("unicode-test", "Hello \u{4e16}\u{754c} \u{1f600}", 4);

    let json = serde_json::to_string(&pattern).expect("Failed to serialize");
    let deserialized: Pattern = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(deserialized.content, "Hello \u{4e16}\u{754c} \u{1f600}");
}

#[test]
fn test_pattern_with_special_characters_in_id() {
    let special_ids = vec![
        "id-with-dashes",
        "id_with_underscores",
        "id.with.dots",
        "id/with/slashes",
        "id:with:colons",
        "id@with@at",
    ];

    for id in special_ids {
        let pattern = create_test_pattern(id, "test", 4);
        let json = serde_json::to_string(&pattern).expect("Failed to serialize");
        let deserialized: Pattern = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.id, id);
    }
}

#[test]
fn test_pattern_with_large_embedding() {
    // Test with a typical embedding size (128 dimensions)
    let pattern = create_test_pattern("large-embedding", "test", 128);

    let json = serde_json::to_string(&pattern).expect("Failed to serialize");
    let deserialized: Pattern = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(deserialized.embedding.len(), 128);
    assert_eq!(deserialized.embedding, pattern.embedding);
}

#[test]
fn test_pattern_with_very_large_embedding() {
    // Test with a large embedding size (1024 dimensions)
    let pattern = create_test_pattern("xlarge-embedding", "test", 1024);

    let json = serde_json::to_string(&pattern).expect("Failed to serialize");
    assert!(!json.is_empty());

    let deserialized: Pattern = serde_json::from_str(&json).expect("Failed to deserialize");
    assert_eq!(deserialized.embedding.len(), 1024);
}

#[test]
fn test_pattern_embedding_precision() {
    // Test that floating point values maintain precision
    let special_values = vec![
        0.0,
        1.0,
        -1.0,
        0.123456789,
        1e-10,
        std::f32::EPSILON,
        std::f32::MIN_POSITIVE,
    ];

    let mut pattern = create_test_pattern("precision-test", "test", special_values.len());
    pattern.embedding = special_values.clone();

    let json = serde_json::to_string(&pattern).expect("Failed to serialize");
    let deserialized: Pattern = serde_json::from_str(&json).expect("Failed to deserialize");

    for (original, loaded) in special_values.iter().zip(deserialized.embedding.iter()) {
        assert!((original - loaded).abs() < 1e-6, "Value mismatch: {} vs {}", original, loaded);
    }
}

#[test]
fn test_pattern_metadata_complex() {
    let mut pattern = create_test_pattern("metadata-test", "test", 4);
    pattern.metadata = serde_json::json!({
        "nested": {
            "level1": {
                "level2": "deep value"
            }
        },
        "array": [1, 2, 3],
        "mixed": [1, "two", true, null],
        "boolean": true,
        "number": 42.5
    });

    let json = serde_json::to_string(&pattern).expect("Failed to serialize");
    let deserialized: Pattern = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(deserialized.metadata["nested"]["level1"]["level2"], "deep value");
    assert_eq!(deserialized.metadata["array"][1], 2);
    assert_eq!(deserialized.metadata["boolean"], true);
    assert_eq!(deserialized.metadata["number"], 42.5);
}

// =============================================================================
// VectorSearch Export/Import Tests (storage simulation)
// =============================================================================

#[test]
fn test_export_import_empty() {
    let search = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));

    let json = search.export_json().expect("Failed to export");
    assert_eq!(json, "[]");

    let mut search2 = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));
    let count = search2.import_json(&json).expect("Failed to import");

    assert_eq!(count, 0);
    assert!(search2.is_empty());
}

#[test]
fn test_export_import_single_pattern() {
    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));
    search
        .add_pattern(create_test_pattern("single", "single pattern", 4))
        .unwrap();

    let json = search.export_json().expect("Failed to export");

    let mut search2 = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));
    let count = search2.import_json(&json).expect("Failed to import");

    assert_eq!(count, 1);
    assert_eq!(search2.len(), 1);

    let pattern = search2.get_pattern("single").expect("Pattern not found");
    assert_eq!(pattern.content, "single pattern");
}

#[test]
fn test_export_import_multiple_patterns() {
    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));

    for i in 0..100 {
        let mut pattern = create_test_pattern(&format!("p{}", i), &format!("pattern {}", i), 4);
        pattern.embedding = vec![i as f32 * 0.01, 0.1, 0.2, 0.3];
        search.add_pattern(pattern).unwrap();
    }

    let json = search.export_json().expect("Failed to export");

    let mut search2 = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));
    let count = search2.import_json(&json).expect("Failed to import");

    assert_eq!(count, 100);
    assert_eq!(search2.len(), 100);

    // Verify some patterns
    assert!(search2.get_pattern("p0").is_some());
    assert!(search2.get_pattern("p50").is_some());
    assert!(search2.get_pattern("p99").is_some());
}

#[test]
fn test_export_import_preserves_search_functionality() {
    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));

    // Add patterns with distinct embeddings
    search
        .add_pattern(Pattern::new("x".to_string(), "x-axis".to_string(), vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    search
        .add_pattern(Pattern::new("y".to_string(), "y-axis".to_string(), vec![0.0, 1.0, 0.0, 0.0]))
        .unwrap();

    let json = search.export_json().expect("Failed to export");

    let mut search2 = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));
    search2.import_json(&json).expect("Failed to import");

    // Search should work correctly after import
    let results = search2.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(results[0].id, "x");

    let results = search2.search(&[0.0, 1.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(results[0].id, "y");
}

#[test]
fn test_import_invalid_json() {
    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));

    let result = search.import_json("not valid json");
    assert!(result.is_err());
}

#[test]
fn test_import_wrong_structure() {
    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));

    // JSON is valid but not a pattern array
    let result = search.import_json(r#"{"key": "value"}"#);
    assert!(result.is_err());
}

#[test]
fn test_import_dimension_mismatch() {
    let json = r#"[{
        "id": "wrong-dim",
        "content": "test",
        "embedding": [0.1, 0.2, 0.3]
    }]"#;

    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));
    let result = search.import_json(json);

    // Should fail because embedding is 3D but search expects 4D
    assert!(result.is_err());
}

#[test]
fn test_export_import_roundtrip_preserves_metadata() {
    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));

    let mut pattern = create_test_pattern("metadata-test", "test", 4);
    pattern.pattern_type = "trajectory".to_string();
    pattern.confidence = 0.9;
    pattern.metadata = serde_json::json!({"key": "value"});

    search.add_pattern(pattern).unwrap();

    let json = search.export_json().expect("Failed to export");

    let mut search2 = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));
    search2.import_json(&json).expect("Failed to import");

    let loaded = search2.get_pattern("metadata-test").expect("Pattern not found");
    assert_eq!(loaded.pattern_type, "trajectory");
    assert_eq!(loaded.confidence, 0.9);
    assert_eq!(loaded.metadata["key"], "value");
}

#[test]
fn test_incremental_import() {
    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(4));

    // Add initial patterns
    search
        .add_pattern(create_test_pattern("initial", "initial pattern", 4))
        .unwrap();

    // Import more patterns
    let json = r#"[{
        "id": "imported",
        "content": "imported pattern",
        "embedding": [0.1, 0.1, 0.1, 0.1]
    }]"#;

    search.import_json(json).expect("Failed to import");

    // Both should exist
    assert_eq!(search.len(), 2);
    assert!(search.get_pattern("initial").is_some());
    assert!(search.get_pattern("imported").is_some());
}

// =============================================================================
// Large Data Tests
// =============================================================================

#[test]
fn test_export_import_large_dataset() {
    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(64));

    // Add 1000 patterns
    for i in 0..1000 {
        let mut embedding = vec![0.0; 64];
        embedding[i % 64] = 1.0;
        let pattern = Pattern::new(format!("p{}", i), format!("pattern {}", i), embedding);
        search.add_pattern(pattern).unwrap();
    }

    let json = search.export_json().expect("Failed to export");

    // Verify JSON size is reasonable
    assert!(json.len() > 0);

    let mut search2 = VectorSearch::new(SearchConfig::new().with_embedding_dim(64));
    let count = search2.import_json(&json).expect("Failed to import");

    assert_eq!(count, 1000);
    assert_eq!(search2.len(), 1000);
}

#[test]
fn test_json_size_estimation() {
    let mut search = VectorSearch::new(SearchConfig::new().with_embedding_dim(128));

    // Add 100 patterns with 128D embeddings
    for i in 0..100 {
        let pattern = create_test_pattern(&format!("pattern-{:03}", i), "some content here", 128);
        search.add_pattern(pattern).unwrap();
    }

    let json = search.export_json().expect("Failed to export");

    // Rough estimate: each float takes ~10-15 chars in JSON
    // 128 floats * ~12 chars * 100 patterns = ~150KB minimum
    assert!(json.len() > 100_000, "JSON should be at least 100KB");
    assert!(json.len() < 1_000_000, "JSON should be less than 1MB for 100 patterns");
}
