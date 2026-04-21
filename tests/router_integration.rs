//! Integration tests for the FastGRNN-based vendor router.

use nagual::router::{
    ComplexityEstimator, ComplexityLevel, EstimatorConfig, FastGRNN, FastGRNNConfig,
    FallbackChain, RouterConfig, RoutingDecision, Vendor, VendorConfig,
    VendorRouter, VendorSelector,
};

/// Generate a sample normalized embedding for testing.
fn sample_embedding(dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Use a deterministic hash-based approach for reproducibility
    let mut embedding = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let hash = hasher.finish();
        embedding.push((hash % 1000) as f32 / 1000.0 - 0.5);
    }

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        embedding.iter_mut().for_each(|x| *x /= norm);
    }

    embedding
}

#[test]
fn test_fastgrnn_forward_pass() {
    let config = FastGRNNConfig::default();
    let model = FastGRNN::new(config.clone()).unwrap();

    let features = vec![0.5; config.input_dim];
    let result = model.forward(&features);

    assert!(result.is_ok());
    let complexity = result.unwrap();
    assert!(complexity >= 0.0 && complexity <= 1.0);
}

#[test]
fn test_fastgrnn_batch_inference() {
    let config = FastGRNNConfig::default();
    let model = FastGRNN::new(config.clone()).unwrap();

    let batch = vec![
        vec![0.1, 0.2, 0.3, 0.4, 0.5],
        vec![0.5, 0.5, 0.5, 0.5, 0.5],
        vec![0.9, 0.8, 0.7, 0.6, 0.5],
    ];

    let results = model.forward_batch(&batch);
    assert!(results.is_ok());

    let complexities = results.unwrap();
    assert_eq!(complexities.len(), 3);

    for c in complexities {
        assert!(c >= 0.0 && c <= 1.0);
    }
}

#[test]
fn test_fastgrnn_inference_speed() {
    let config = FastGRNNConfig::compact();
    let model = FastGRNN::new(config.clone()).unwrap();

    let features = vec![0.5; config.input_dim];

    // Warm up
    for _ in 0..100 {
        let _ = model.forward(&features);
    }

    model.reset_stats();

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = model.forward(&features);
    }
    let elapsed_us = start.elapsed().as_micros();

    let avg_us = elapsed_us as f64 / 1000.0;

    // Should be under 500us per inference in debug mode
    // (In release mode, this is typically under 50us)
    assert!(
        avg_us < 500.0,
        "Inference too slow: {:.2} us (expected < 500 us in debug mode)",
        avg_us
    );
}

#[test]
fn test_complexity_estimator_features() {
    let config = EstimatorConfig::default();
    let estimator = ComplexityEstimator::new(config);

    let query = "How do I implement a binary search tree in Rust?";
    let embedding = sample_embedding(128);

    let features = estimator.extract_features(query, &embedding);
    assert!(features.is_ok());

    let f = features.unwrap();

    // All features should be in [0, 1]
    assert!(f.query_length >= 0.0 && f.query_length <= 1.0);
    assert!(f.embedding_norm >= 0.0 && f.embedding_norm <= 1.0);
    assert!(f.domain_specificity >= 0.0 && f.domain_specificity <= 1.0);
    assert!(f.pattern_coverage >= 0.0 && f.pattern_coverage <= 1.0);
    assert!(f.historical_accuracy >= 0.0 && f.historical_accuracy <= 1.0);
}

#[test]
fn test_complexity_estimator_domain_detection() {
    let config = EstimatorConfig::default();
    let estimator = ComplexityEstimator::new(config);

    let embedding = sample_embedding(128);

    // Technical query
    let tech_features = estimator
        .extract_features(
            "Implement concurrent algorithm with thread-safe caching",
            &embedding,
        )
        .unwrap();

    // General query
    let gen_features = estimator
        .extract_features("How can I help you today?", &embedding)
        .unwrap();

    // Technical query should have higher domain specificity
    assert!(
        tech_features.domain_specificity > gen_features.domain_specificity,
        "Tech: {}, Gen: {}",
        tech_features.domain_specificity,
        gen_features.domain_specificity
    );
}

#[test]
fn test_vendor_selector_thresholds() {
    let config = VendorConfig::default();
    let selector = VendorSelector::new(config);

    // Low complexity -> local small
    let decision = selector.select(0.1, 0.9);
    assert_eq!(decision.vendor, Vendor::LocalSmall);
    assert_eq!(decision.level, ComplexityLevel::Low);

    // Medium-low complexity -> local large
    let decision = selector.select(0.4, 0.9);
    assert_eq!(decision.vendor, Vendor::LocalLarge);
    assert_eq!(decision.level, ComplexityLevel::Medium);

    // Medium-high complexity -> Claude
    let decision = selector.select(0.6, 0.9);
    assert_eq!(decision.vendor, Vendor::Claude);
    assert_eq!(decision.level, ComplexityLevel::High);

    // High complexity -> Claude
    let decision = selector.select(0.9, 0.9);
    assert_eq!(decision.vendor, Vendor::Claude);
    assert_eq!(decision.level, ComplexityLevel::VeryHigh);
}

#[test]
fn test_vendor_selector_fallback() {
    let config = VendorConfig::default();
    let selector = VendorSelector::new(config);

    // Mark local small as unavailable
    selector.mark_unavailable(Vendor::LocalSmall);

    // Low complexity should fallback to local large
    let decision = selector.select(0.1, 0.9);
    assert!(decision.is_fallback);
    assert_eq!(decision.vendor, Vendor::LocalLarge);
}

#[test]
fn test_vendor_selector_fallback_chain() {
    let config = VendorConfig::default();
    let selector = VendorSelector::new(config);

    // Mark multiple vendors unavailable
    selector.mark_unavailable(Vendor::LocalSmall);
    selector.mark_unavailable(Vendor::LocalLarge);

    // Should fallback to Claude
    let decision = selector.select(0.1, 0.9);
    assert!(decision.is_fallback);
    assert_eq!(decision.vendor, Vendor::Claude);
}

#[test]
fn test_fallback_chain_operations() {
    let chain = FallbackChain::default();

    assert_eq!(chain.vendors.len(), 4);
    assert!(chain.contains(Vendor::Claude));
    assert_eq!(chain.next_after(Vendor::LocalSmall), Some(Vendor::LocalLarge));
    assert_eq!(chain.next_after(Vendor::GPT), None);

    // Starting from Claude
    let claude_chain = FallbackChain::starting_from(Vendor::Claude);
    assert_eq!(claude_chain.vendors[0], Vendor::Claude);
}

#[test]
fn test_vendor_router_end_to_end() {
    let config = RouterConfig::default();
    let router = VendorRouter::new(config).unwrap();

    let query = "What is machine learning?";
    let embedding = sample_embedding(128);

    let decision = router.route(query, &embedding);
    assert!(decision.is_ok());

    let d = decision.unwrap();
    assert!(d.complexity >= 0.0 && d.complexity <= 1.0);
    assert!(d.confidence >= 0.0 && d.confidence <= 1.0);
    assert!(d.routing_latency_us > 0);
}

#[test]
fn test_vendor_router_latency_target() {
    let config = RouterConfig::low_latency();
    let router = VendorRouter::new(config).unwrap();

    let query = "Simple test query";
    let embedding = sample_embedding(128);

    // Warm up
    for _ in 0..10 {
        let _ = router.route(query, &embedding);
    }

    // Measure
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = router.route(query, &embedding);
    }
    let elapsed_ms = start.elapsed().as_millis();
    let avg_ms = elapsed_ms as f64 / 100.0;

    // Should be under 5ms per routing decision
    assert!(
        avg_ms < 5.0,
        "Routing too slow: {:.2} ms (target < 5 ms)",
        avg_ms
    );
}

#[test]
fn test_vendor_router_metrics() {
    let config = RouterConfig::default();
    let router = VendorRouter::new(config).unwrap();

    let embedding = sample_embedding(128);

    // Make several routing decisions
    let _ = router.route("query 1", &embedding);
    let _ = router.route("query 2", &embedding);
    let _ = router.route("query 3", &embedding);

    let metrics = router.metrics();
    assert_eq!(
        metrics
            .total_decisions
            .load(std::sync::atomic::Ordering::Relaxed),
        3
    );
    assert!(metrics.avg_latency_us() > 0.0);
}

#[test]
fn test_vendor_router_outcome_recording() {
    let config = RouterConfig::default();
    let router = VendorRouter::new(config).unwrap();

    // Record success
    router.record_outcome("test query", Vendor::Claude, true, 1000);

    let status = router.vendor_status(Vendor::Claude).unwrap();
    assert_eq!(
        status
            .success_count
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    // Record failure
    router.record_outcome("test query 2", Vendor::GPT, false, 500);

    let gpt_status = router.vendor_status(Vendor::GPT).unwrap();
    assert_eq!(
        gpt_status
            .failure_count
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn test_vendor_router_simple_route() {
    let config = RouterConfig::default();
    let router = VendorRouter::new(config).unwrap();

    let query = "Test query";
    let embedding = sample_embedding(128);

    // Simple route should also work
    let decision = router.route_simple(query, &embedding);
    assert!(decision.is_ok());
}

#[test]
fn test_vendor_properties() {
    assert!(Vendor::LocalSmall.is_local());
    assert!(Vendor::LocalLarge.is_local());
    assert!(Vendor::Claude.is_cloud());
    assert!(Vendor::GPT.is_cloud());

    assert!(Vendor::LocalSmall.relative_cost() < Vendor::Claude.relative_cost());
    assert!(Vendor::LocalSmall.relative_latency() < Vendor::Claude.relative_latency());
}

#[test]
fn test_complexity_levels() {
    assert_eq!(ComplexityLevel::from_score(0.1), ComplexityLevel::Low);
    assert_eq!(ComplexityLevel::from_score(0.4), ComplexityLevel::Medium);
    assert_eq!(ComplexityLevel::from_score(0.6), ComplexityLevel::High);
    assert_eq!(ComplexityLevel::from_score(0.9), ComplexityLevel::VeryHigh);
}

#[test]
fn test_routing_decision_serialization() {
    let decision = RoutingDecision::new(Vendor::Claude, 0.7, 0.95, 150);

    let json = serde_json::to_string(&decision).unwrap();
    assert!(json.contains("claude"));
    assert!(json.contains("0.7"));

    let parsed: RoutingDecision = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.vendor, Vendor::Claude);
}

#[test]
fn test_fastgrnn_model_size() {
    let config = FastGRNNConfig::default();
    let model = FastGRNN::new(config).unwrap();

    let size = model.model_size_bytes();

    // Should be under 100KB
    assert!(
        size < 100_000,
        "Model too large: {} bytes (expected < 100KB)",
        size
    );
}

#[test]
fn test_vendor_selector_record_success_failure() {
    let selector = VendorSelector::new(VendorConfig::default());

    // Record successes
    selector.record_success(Vendor::Claude, 100);
    selector.record_success(Vendor::Claude, 150);

    let status = selector.get_status(Vendor::Claude).unwrap();
    assert_eq!(
        status
            .success_count
            .load(std::sync::atomic::Ordering::Relaxed),
        2
    );
    assert!(status.avg_latency_us() > 0.0);

    // Record failure
    selector.record_failure(Vendor::GPT, "Test error".to_string());

    let gpt_status = selector.get_status(Vendor::GPT).unwrap();
    assert_eq!(
        gpt_status
            .failure_count
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn test_complexity_estimator_accuracy_cache() {
    let estimator = ComplexityEstimator::new(EstimatorConfig::default());

    let query = "test query for caching";
    estimator.record_accuracy(query, 0.95);

    let embedding = sample_embedding(128);
    let features = estimator.extract_features(query, &embedding).unwrap();

    assert!((features.historical_accuracy - 0.95).abs() < 0.01);

    // Clear and verify default
    estimator.clear_cache();
    let features2 = estimator.extract_features(query, &embedding).unwrap();
    assert!((features2.historical_accuracy - 0.5).abs() < 0.01);
}
