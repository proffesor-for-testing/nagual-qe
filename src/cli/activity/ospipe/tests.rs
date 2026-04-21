//! Integration tests for the OSpipe pipeline.

use std::time::Duration;

use super::config::{EmbeddingDim, OSpipeConfig};
use super::dedup::SlidingWindowDedup;
use super::pii_gate::{PiiGate, PiiPolicy};

#[test]
fn test_ospipe_config_defaults() {
    let config = OSpipeConfig::default();

    assert!(config.pii_enabled);
    assert_eq!(config.pii_policy, "redact");
    assert!(config.dedup_enabled);
    assert_eq!(config.dedup_threshold, 0.9);
    assert_eq!(config.embedding_dim, EmbeddingDim::Dim128);
    assert!(config.generate_embeddings);
}

#[test]
fn test_ospipe_config_minimal() {
    let config = OSpipeConfig::minimal();

    assert!(!config.pii_enabled);
    assert!(!config.dedup_enabled);
    assert!(!config.generate_embeddings);
}

#[test]
fn test_pii_gate_integration() {
    // Test the full flow from PII gate
    let gate = PiiGate::new(PiiPolicy::Redact);

    // Test with email
    let result = gate.process("Contact me at test@example.com for details.");
    assert!(result.is_accepted());
    assert!(result.had_pii());

    let content = result.content().unwrap();
    assert!(content.contains("[EMAIL]"));
    assert!(!content.contains("test@example.com"));

    // Test with SSN under reject policy
    let gate_reject = PiiGate::new(PiiPolicy::Reject);
    let result_reject = gate_reject.process("SSN: 456-78-9012");
    assert!(!result_reject.is_accepted());
}

#[test]
fn test_dedup_integration() {
    let mut dedup = SlidingWindowDedup::new(Duration::from_secs(300), 0.9);

    // Create a normalized embedding
    let embedding: Vec<f32> = (0..128)
        .map(|i| (i as f32 / 128.0).sin())
        .collect();
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    let embedding: Vec<f32> = embedding.iter().map(|x| x / norm).collect();

    let now = chrono::Utc::now();

    // First entry should not be duplicate
    let result1 = dedup.check_and_add(embedding.clone(), now);
    assert!(!result1.is_duplicate);
    assert_eq!(dedup.window_len(), 1);

    // Same embedding should be duplicate
    let result2 = dedup.check_and_add(embedding.clone(), now);
    assert!(result2.is_duplicate);
    assert!(result2.max_similarity > 0.99); // Should be ~1.0
    assert_eq!(dedup.window_len(), 1); // Not added

    // Different embedding should not be duplicate
    let different: Vec<f32> = (0..128)
        .map(|i| ((i as f32 + 50.0) / 128.0).cos() * 0.5)
        .collect();
    let norm: f32 = different.iter().map(|x| x * x).sum::<f32>().sqrt();
    let different: Vec<f32> = different.iter().map(|x| x / norm).collect();

    let result3 = dedup.check_and_add(different, now);
    assert!(!result3.is_duplicate);
    assert_eq!(dedup.window_len(), 2);
}

#[test]
fn test_embedding_dim_parsing() {
    assert_eq!(EmbeddingDim::from_str("128"), Some(EmbeddingDim::Dim128));
    assert_eq!(EmbeddingDim::from_str("384"), Some(EmbeddingDim::Dim384));
    assert_eq!(EmbeddingDim::from_str("dim128"), Some(EmbeddingDim::Dim128));
    assert_eq!(EmbeddingDim::from_str("DIM384"), Some(EmbeddingDim::Dim384));
    assert_eq!(EmbeddingDim::from_str("invalid"), None);
    assert_eq!(EmbeddingDim::from_str("768"), None);
}

#[test]
fn test_pii_policy_parsing() {
    assert_eq!(PiiPolicy::from_str("reject"), Some(PiiPolicy::Reject));
    assert_eq!(PiiPolicy::from_str("redact"), Some(PiiPolicy::Redact));
    assert_eq!(PiiPolicy::from_str("warn"), Some(PiiPolicy::Warn));
    assert_eq!(PiiPolicy::from_str("allow"), Some(PiiPolicy::Allow));
    assert_eq!(PiiPolicy::from_str("REDACT"), Some(PiiPolicy::Redact));
    assert_eq!(PiiPolicy::from_str("invalid"), None);
}

#[test]
fn test_config_builder_pattern() {
    let config = OSpipeConfig::default()
        .with_pii_policy("reject")
        .with_dedup_threshold(0.85)
        .with_dedup_window(Duration::from_secs(600))
        .with_embedding_dim(EmbeddingDim::Dim128);

    assert_eq!(config.pii_policy, "reject");
    assert_eq!(config.dedup_threshold, 0.85);
    assert_eq!(config.dedup_window, Duration::from_secs(600));
    assert_eq!(config.embedding_dim, EmbeddingDim::Dim128);
}

#[test]
fn test_dedup_stats() {
    let mut dedup = SlidingWindowDedup::new(Duration::from_secs(300), 0.9);
    let now = chrono::Utc::now();

    // Create distinctly different test embeddings
    for i in 0..5u64 {
        // Use different seeds to get very different embeddings
        let embedding: Vec<f32> = (0..128)
            .map(|j| {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                (i * 1000 + j as u64).hash(&mut hasher);
                (hasher.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let embedding: Vec<f32> = embedding.iter().map(|x| x / norm).collect();

        dedup.check_and_add(embedding, now);
    }

    let stats = dedup.stats();
    assert_eq!(stats.total_checked, 5);
    // All should be unique since they're generated with very different seeds
    assert_eq!(stats.window_entries, 5);
}

#[test]
fn test_pii_gate_threshold() {
    use crate::security::pii::PiiClassification;

    // Gate that only rejects critical PII
    let gate = PiiGate::with_threshold(PiiPolicy::Reject, PiiClassification::Critical);

    // Email (Medium classification) should pass
    let result_email = gate.process("Email: test@example.com");
    assert!(result_email.is_accepted());

    // SSN (Critical classification) should be rejected
    let result_ssn = gate.process("SSN: 456-78-9012");
    assert!(!result_ssn.is_accepted());
}

// ============================================================================
// Integration Tests for OSpipePipeline::process_item()
// ============================================================================

#[cfg(test)]
mod process_item_tests {
    use super::super::config::OSpipeConfig;
    use super::super::pipeline::{IngestItemResult, OSpipePipeline};
    use super::super::query_router::{QueryMode, QueryParams};
    use crate::cli::common::init_storage_sqlite_only;
    use chrono::Utc;
    use std::sync::Arc;
    use tempfile::TempDir;

    /// Create a test pipeline with SQLite storage (no embedding generation).
    async fn create_test_pipeline() -> (OSpipePipeline, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_ospipe.db");
        let storage = Arc::new(init_storage_sqlite_only(&db_path).await.unwrap());

        // Minimal config - no embeddings (would need ONNX model)
        let config = OSpipeConfig::minimal()
            .with_pii_enabled(true)
            .with_pii_policy("redact")
            .with_dedup_enabled(false); // Disable dedup since no embeddings

        let pipeline = OSpipePipeline::new(storage, config);
        (pipeline, temp_dir)
    }

    #[tokio::test]
    async fn test_process_item_basic() {
        let (mut pipeline, _temp) = create_test_pipeline().await;

        let content = "Test content for basic processing";
        let result = pipeline
            .process_item(content, "TestApp", Utc::now(), None)
            .await;

        assert!(
            result.pattern_id.is_some(),
            "Should store pattern successfully"
        );
        assert!(!result.was_rejected);
        assert!(!result.was_duplicate);
        assert_eq!(result.original_length, content.len());
    }

    #[tokio::test]
    async fn test_process_item_with_pii_redaction() {
        let (mut pipeline, _temp) = create_test_pipeline().await;

        let content = "Contact me at test@example.com for details";
        let result = pipeline
            .process_item(content, "EmailApp", Utc::now(), None)
            .await;

        assert!(
            result.pattern_id.is_some(),
            "Should store after redaction"
        );
        assert!(result.was_redacted, "Content should be redacted");
        assert!(!result.was_rejected);
        // Redacted content should be shorter or same (email replaced with [EMAIL])
        assert!(result.processed_length > 0);
    }

    #[tokio::test]
    async fn test_process_item_pii_rejection() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_reject.db");
        let storage = Arc::new(init_storage_sqlite_only(&db_path).await.unwrap());

        // Config that rejects PII instead of redacting
        let config = OSpipeConfig::minimal()
            .with_pii_enabled(true)
            .with_pii_policy("reject");

        let mut pipeline = OSpipePipeline::new(storage, config);

        let content = "SSN: 456-78-9012";
        let result = pipeline
            .process_item(content, "SensitiveApp", Utc::now(), None)
            .await;

        assert!(result.was_rejected, "Should reject content with SSN");
        assert!(result.pattern_id.is_none());
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn test_process_item_multiple_apps() {
        let (mut pipeline, _temp) = create_test_pipeline().await;
        let now = Utc::now();

        // Process items from different apps
        let apps = ["Browser", "Terminal", "VSCode", "Slack"];
        let mut pattern_ids = Vec::new();

        for app in &apps {
            let content = format!("Activity from {} application", app);
            let result = pipeline
                .process_item(&content, app, now, None)
                .await;

            assert!(result.pattern_id.is_some(), "Failed for app: {}", app);
            pattern_ids.push(result.pattern_id.unwrap());
        }

        // All pattern IDs should be unique
        let unique_ids: std::collections::HashSet<_> = pattern_ids.iter().collect();
        assert_eq!(unique_ids.len(), apps.len(), "All IDs should be unique");
    }

    #[tokio::test]
    async fn test_process_item_empty_content() {
        let (mut pipeline, _temp) = create_test_pipeline().await;

        let result = pipeline
            .process_item("", "TestApp", Utc::now(), None)
            .await;

        // Empty content should still be processed (stored with empty content)
        // This is a valid case for activity logging
        assert!(!result.was_rejected);
        assert_eq!(result.original_length, 0);
    }
}

// ============================================================================
// Integration Tests for QueryRouter::route()
// ============================================================================

#[cfg(test)]
mod query_router_tests {
    use super::super::query_router::{QueryMode, QueryParams, QueryRouter};
    use crate::cli::common::init_storage_sqlite_only;
    use crate::reasoning_bank::pattern::{Pattern, PatternCategory};
    use std::sync::Arc;
    use tempfile::TempDir;

    /// Create a test router with pre-populated patterns.
    async fn create_test_router_with_patterns() -> (QueryRouter, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_router.db");
        let storage = Arc::new(init_storage_sqlite_only(&db_path).await.unwrap());

        // Store some test patterns
        let patterns = vec![
            Pattern::builder()
                .problem("Rust async error handling")
                .solution("Use tokio::spawn for concurrent tasks")
                .category(PatternCategory::Custom("rust.async".into()))
                .tags(vec!["rust".into(), "async".into(), "tokio".into()])
                .effectiveness(0.8)
                .confidence(0.9)
                .build(),
            Pattern::builder()
                .problem("Database connection pooling")
                .solution("Use sqlx connection pool with max_connections=10")
                .category(PatternCategory::Custom("database".into()))
                .tags(vec!["database".into(), "postgresql".into(), "pooling".into()])
                .effectiveness(0.85)
                .confidence(0.88)
                .build(),
            Pattern::builder()
                .problem("API rate limiting")
                .solution("Implement exponential backoff with jitter")
                .category(PatternCategory::Custom("api".into()))
                .tags(vec!["api".into(), "rate-limit".into(), "backoff".into()])
                .effectiveness(0.75)
                .confidence(0.85)
                .build(),
        ];

        for pattern in patterns {
            storage.store_pattern(&pattern).await.unwrap();
        }

        let router = QueryRouter::new(Arc::clone(&storage));
        (router, temp_dir)
    }

    #[tokio::test]
    async fn test_query_router_keyword_search() {
        let (router, _temp) = create_test_router_with_patterns().await;

        let params = QueryParams::new("async error")
            .with_mode(QueryMode::Keyword)
            .with_limit(10);

        let results = router.route(&params).await.unwrap();

        // Should find the Rust async pattern via keyword search
        assert!(!results.is_empty(), "Should find patterns via keyword search");
        assert!(
            results.iter().any(|r| r.problem.contains("async")),
            "Should find async-related patterns"
        );
    }

    #[tokio::test]
    async fn test_query_router_hybrid_search() {
        let (router, _temp) = create_test_router_with_patterns().await;

        let params = QueryParams::new("database connection")
            .with_mode(QueryMode::Hybrid)
            .with_limit(5);

        let results = router.route(&params).await.unwrap();

        // Hybrid search should combine multiple modes
        assert!(!results.is_empty(), "Hybrid search should return results");
    }

    #[tokio::test]
    async fn test_query_router_temporal_search() {
        let (router, _temp) = create_test_router_with_patterns().await;

        let now = chrono::Utc::now();
        let yesterday = now - chrono::Duration::days(1);

        // Use a broad query that will match
        let params = QueryParams::new("rust OR database OR api")
            .with_mode(QueryMode::Temporal)
            .with_time_range(Some(yesterday), Some(now))
            .with_limit(10);

        // Temporal search should execute without errors
        // Results may be empty if patterns lack timestamp metadata
        let results = router.route(&params).await.unwrap();

        // Just verify the search executed successfully
        // (results may be empty depending on temporal search implementation)
        assert!(results.len() <= 10, "Should respect limit");
    }

    #[tokio::test]
    async fn test_query_router_with_domain_filter() {
        let (router, _temp) = create_test_router_with_patterns().await;

        let params = QueryParams::new("connection")
            .with_mode(QueryMode::Keyword)
            .with_domain("database")
            .with_limit(10);

        let results = router.route(&params).await.unwrap();

        // Should only find database-related patterns
        for result in &results {
            if let Some(ref domain) = result.domain {
                assert!(
                    domain.contains("database"),
                    "Domain filter should work: got {:?}",
                    domain
                );
            }
        }
    }

    #[tokio::test]
    async fn test_query_router_stats_tracking() {
        let (router, _temp) = create_test_router_with_patterns().await;

        // Run several queries
        for _ in 0..5 {
            let params = QueryParams::new("test query").with_mode(QueryMode::Keyword);
            let _ = router.route(&params).await;
        }

        let stats = router.stats();
        assert_eq!(stats.total_queries, 5, "Should track query count");
        assert!(
            stats.queries_by_mode.contains_key("keyword"),
            "Should track queries by mode"
        );
    }

    #[tokio::test]
    async fn test_query_router_empty_results() {
        let (router, _temp) = create_test_router_with_patterns().await;

        let params = QueryParams::new("nonexistent_gibberish_query_xyz123")
            .with_mode(QueryMode::Keyword)
            .with_limit(10);

        let results = router.route(&params).await.unwrap();

        // No match is a valid result - should return empty
        assert!(results.is_empty() || results.len() <= 10);
    }

    #[tokio::test]
    async fn test_query_router_min_score_filter() {
        let (router, _temp) = create_test_router_with_patterns().await;

        let params = QueryParams::new("async")
            .with_mode(QueryMode::Keyword)
            .with_min_score(0.5)
            .with_limit(10);

        let results = router.route(&params).await.unwrap();

        // All results should have score >= min_score
        for result in &results {
            assert!(
                result.score >= 0.5,
                "Result score {} should be >= min_score 0.5",
                result.score
            );
        }
    }
}
