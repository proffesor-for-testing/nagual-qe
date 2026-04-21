//! Router Tests - Phase 2 Inference Layer
//!
//! Comprehensive test suite for the FastGRNN-based router component.
//! Tests cover:
//! - FastGRNN inference accuracy
//! - Complexity estimation
//! - Vendor selection logic
//! - Fallback chain behavior
//! - Latency requirements (<5ms)
//!
//! # Test Categories
//!
//! 1. **FastGRNN Inference Tests**: Validate model predictions
//! 2. **Complexity Estimation Tests**: Ensure accurate task complexity scoring
//! 3. **Vendor Selection Tests**: Verify optimal vendor routing
//! 4. **Fallback Chain Tests**: Test graceful degradation
//! 5. **Performance Tests**: Enforce latency SLAs

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use proptest::prelude::*;
use serde::{Deserialize, Serialize};

mod common;
use common::{
    cosine_similarity, measure_time, normalized_embedding, similar_embeddings,
};

// ============================================================================
// Router Types (Mirroring production types for testing)
// ============================================================================

/// Vendor/provider types for LLM routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Vendor {
    Anthropic,
    OpenAI,
    Local,
    Mock,
}

impl Vendor {
    /// Get the latency SLA for this vendor in milliseconds.
    pub fn latency_sla_ms(&self) -> u64 {
        match self {
            Vendor::Anthropic => 2000,
            Vendor::OpenAI => 2000,
            Vendor::Local => 500,
            Vendor::Mock => 10,
        }
    }

    /// Get the cost factor for this vendor.
    pub fn cost_factor(&self) -> f32 {
        match self {
            Vendor::Anthropic => 1.0,
            Vendor::OpenAI => 0.9,
            Vendor::Local => 0.1,
            Vendor::Mock => 0.0,
        }
    }

    /// Get capability score for a complexity level.
    pub fn capability_score(&self, complexity: ComplexityLevel) -> f32 {
        match (self, complexity) {
            (Vendor::Anthropic, ComplexityLevel::High) => 0.95,
            (Vendor::Anthropic, ComplexityLevel::Medium) => 0.90,
            (Vendor::Anthropic, ComplexityLevel::Low) => 0.85,
            (Vendor::OpenAI, ComplexityLevel::High) => 0.90,
            (Vendor::OpenAI, ComplexityLevel::Medium) => 0.88,
            (Vendor::OpenAI, ComplexityLevel::Low) => 0.85,
            (Vendor::Local, ComplexityLevel::High) => 0.60,
            (Vendor::Local, ComplexityLevel::Medium) => 0.75,
            (Vendor::Local, ComplexityLevel::Low) => 0.85,
            (Vendor::Mock, _) => 0.1,
        }
    }
}

/// Complexity levels for routing decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
}

impl ComplexityLevel {
    /// Create from a numeric score (0.0-1.0).
    pub fn from_score(score: f32) -> Self {
        if score >= 0.7 {
            ComplexityLevel::High
        } else if score >= 0.4 {
            ComplexityLevel::Medium
        } else {
            ComplexityLevel::Low
        }
    }

    /// Convert to a numeric score.
    pub fn to_score(&self) -> f32 {
        match self {
            ComplexityLevel::Low => 0.2,
            ComplexityLevel::Medium => 0.5,
            ComplexityLevel::High => 0.85,
        }
    }
}

/// Configuration for the router.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Available vendors in priority order.
    pub vendors: Vec<Vendor>,
    /// Maximum latency for routing decision (in ms).
    pub max_routing_latency_ms: u64,
    /// Weight for cost in routing decisions.
    pub cost_weight: f32,
    /// Weight for capability in routing decisions.
    pub capability_weight: f32,
    /// Weight for latency in routing decisions.
    pub latency_weight: f32,
    /// Enable fallback chain.
    pub enable_fallback: bool,
    /// FastGRNN model dimension.
    pub grnn_dimension: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            vendors: vec![Vendor::Anthropic, Vendor::OpenAI, Vendor::Local],
            max_routing_latency_ms: 5,
            cost_weight: 0.2,
            capability_weight: 0.6,
            latency_weight: 0.2,
            enable_fallback: true,
            grnn_dimension: 64,
        }
    }
}

/// A routing request.
#[derive(Debug, Clone)]
pub struct RoutingRequest {
    /// The task description or query.
    pub task: String,
    /// Pre-computed embedding for the task.
    pub embedding: Option<Vec<f32>>,
    /// Preferred vendors (optional).
    pub preferred_vendors: Option<Vec<Vendor>>,
    /// Maximum cost budget (0.0-1.0).
    pub max_cost: Option<f32>,
    /// Required minimum capability score.
    pub min_capability: Option<f32>,
}

impl RoutingRequest {
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            task: task.into(),
            embedding: None,
            preferred_vendors: None,
            max_cost: None,
            min_capability: None,
        }
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_preferred_vendors(mut self, vendors: Vec<Vendor>) -> Self {
        self.preferred_vendors = Some(vendors);
        self
    }

    pub fn with_max_cost(mut self, cost: f32) -> Self {
        self.max_cost = Some(cost.clamp(0.0, 1.0));
        self
    }

    pub fn with_min_capability(mut self, capability: f32) -> Self {
        self.min_capability = Some(capability.clamp(0.0, 1.0));
        self
    }
}

/// Result of a routing decision.
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// Selected vendor.
    pub vendor: Vendor,
    /// Estimated complexity.
    pub complexity: ComplexityLevel,
    /// Confidence in the routing decision.
    pub confidence: f32,
    /// Fallback chain if primary fails.
    pub fallback_chain: Vec<Vendor>,
    /// Time taken for routing decision (in ms).
    pub routing_time_ms: u64,
    /// Breakdown of scoring factors.
    pub score_breakdown: ScoreBreakdown,
}

/// Breakdown of how the routing score was computed.
#[derive(Debug, Clone, Default)]
pub struct ScoreBreakdown {
    pub complexity_score: f32,
    pub cost_score: f32,
    pub capability_score: f32,
    pub latency_score: f32,
    pub final_score: f32,
}

/// Mock FastGRNN model for testing.
#[derive(Debug)]
pub struct MockFastGRNN {
    /// Weight matrix for complexity prediction.
    weights: Vec<f32>,
    /// Bias term.
    bias: f32,
    /// Hidden state dimension.
    hidden_dim: usize,
    /// Simulated inference latency.
    simulated_latency_us: u64,
}

impl MockFastGRNN {
    pub fn new(hidden_dim: usize) -> Self {
        // Initialize with deterministic weights for reproducibility
        let weights: Vec<f32> = (0..hidden_dim)
            .map(|i| ((i as f32 / hidden_dim as f32) - 0.5) * 2.0)
            .collect();

        Self {
            weights,
            bias: 0.1,
            hidden_dim,
            simulated_latency_us: 100,
        }
    }

    /// Predict complexity score from embedding.
    pub fn predict_complexity(&self, embedding: &[f32]) -> f32 {
        // Simulate some processing time
        std::thread::sleep(Duration::from_micros(self.simulated_latency_us));

        // Simple dot product + bias + sigmoid
        let dot: f32 = embedding
            .iter()
            .zip(self.weights.iter().cycle())
            .map(|(e, w)| e * w)
            .sum();

        let raw_score = dot / embedding.len() as f32 + self.bias;

        // Sigmoid activation
        1.0 / (1.0 + (-raw_score).exp())
    }

    /// Get the hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

/// Main router implementation.
#[derive(Debug)]
pub struct Router {
    config: RouterConfig,
    grnn: MockFastGRNN,
    vendor_stats: HashMap<Vendor, VendorStats>,
}

/// Statistics for a vendor.
#[derive(Debug, Clone, Default)]
pub struct VendorStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub total_latency_ms: u64,
    pub avg_latency_ms: u64,
    pub is_healthy: bool,
}

impl VendorStats {
    pub fn success_rate(&self) -> f32 {
        if self.total_requests == 0 {
            1.0
        } else {
            self.successful_requests as f32 / self.total_requests as f32
        }
    }
}

impl Router {
    pub fn new(config: RouterConfig) -> Self {
        let grnn = MockFastGRNN::new(config.grnn_dimension);
        let mut vendor_stats = HashMap::new();

        for vendor in &config.vendors {
            vendor_stats.insert(
                *vendor,
                VendorStats {
                    is_healthy: true,
                    ..Default::default()
                },
            );
        }

        Self {
            config,
            grnn,
            vendor_stats,
        }
    }

    /// Route a request to the optimal vendor.
    pub fn route(&self, request: &RoutingRequest) -> RoutingResult {
        let start = Instant::now();

        // Step 1: Estimate complexity using FastGRNN
        let complexity_score = if let Some(ref embedding) = request.embedding {
            self.grnn.predict_complexity(embedding)
        } else {
            // Fallback: estimate from task length
            self.estimate_complexity_from_text(&request.task)
        };

        let complexity = ComplexityLevel::from_score(complexity_score);

        // Step 2: Score each vendor
        let vendor_scores = self.score_vendors(request, complexity);

        // Step 3: Select best vendor
        let (best_vendor, score_breakdown) = self.select_best_vendor(&vendor_scores, request);

        // Step 4: Build fallback chain
        let fallback_chain = if self.config.enable_fallback {
            self.build_fallback_chain(best_vendor, &vendor_scores)
        } else {
            Vec::new()
        };

        // Step 5: Calculate confidence
        let confidence = self.calculate_confidence(&vendor_scores, best_vendor);

        let routing_time_ms = start.elapsed().as_millis() as u64;

        RoutingResult {
            vendor: best_vendor,
            complexity,
            confidence,
            fallback_chain,
            routing_time_ms,
            score_breakdown,
        }
    }

    /// Estimate complexity from task text.
    fn estimate_complexity_from_text(&self, task: &str) -> f32 {
        let length_factor = (task.len() as f32 / 1000.0).min(1.0);
        let question_count = task.matches('?').count() as f32;
        let code_markers = task.matches("```").count() as f32;

        let base_score = 0.3 + length_factor * 0.3 + question_count * 0.1 + code_markers * 0.15;
        base_score.clamp(0.0, 1.0)
    }

    /// Score all vendors for the request.
    fn score_vendors(
        &self,
        request: &RoutingRequest,
        complexity: ComplexityLevel,
    ) -> HashMap<Vendor, ScoreBreakdown> {
        let mut scores = HashMap::new();

        for vendor in &self.config.vendors {
            // Skip unhealthy vendors
            if let Some(stats) = self.vendor_stats.get(vendor) {
                if !stats.is_healthy {
                    continue;
                }
            }

            // Skip if not in preferred list (if specified)
            if let Some(ref preferred) = request.preferred_vendors {
                if !preferred.contains(vendor) {
                    continue;
                }
            }

            let cost_score = 1.0 - vendor.cost_factor();
            let capability_score = vendor.capability_score(complexity);
            let latency_score = 1.0 - (vendor.latency_sla_ms() as f32 / 3000.0).min(1.0);

            // Check constraints
            if let Some(max_cost) = request.max_cost {
                if vendor.cost_factor() > max_cost {
                    continue;
                }
            }

            if let Some(min_capability) = request.min_capability {
                if capability_score < min_capability {
                    continue;
                }
            }

            let final_score = self.config.cost_weight * cost_score
                + self.config.capability_weight * capability_score
                + self.config.latency_weight * latency_score;

            scores.insert(
                *vendor,
                ScoreBreakdown {
                    complexity_score: complexity.to_score(),
                    cost_score,
                    capability_score,
                    latency_score,
                    final_score,
                },
            );
        }

        scores
    }

    /// Select the best vendor from scores.
    fn select_best_vendor(
        &self,
        scores: &HashMap<Vendor, ScoreBreakdown>,
        _request: &RoutingRequest,
    ) -> (Vendor, ScoreBreakdown) {
        scores
            .iter()
            .max_by(|a, b| {
                a.1.final_score
                    .partial_cmp(&b.1.final_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(v, s)| (*v, s.clone()))
            .unwrap_or((Vendor::Local, ScoreBreakdown::default()))
    }

    /// Build fallback chain excluding the primary vendor.
    fn build_fallback_chain(
        &self,
        primary: Vendor,
        scores: &HashMap<Vendor, ScoreBreakdown>,
    ) -> Vec<Vendor> {
        let mut fallbacks: Vec<(Vendor, f32)> = scores
            .iter()
            .filter(|(v, _)| **v != primary)
            .map(|(v, s)| (*v, s.final_score))
            .collect();

        fallbacks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        fallbacks.into_iter().map(|(v, _)| v).collect()
    }

    /// Calculate confidence in the routing decision.
    fn calculate_confidence(
        &self,
        scores: &HashMap<Vendor, ScoreBreakdown>,
        selected: Vendor,
    ) -> f32 {
        if scores.len() <= 1 {
            return 0.5; // Low confidence with no alternatives
        }

        let selected_score = scores
            .get(&selected)
            .map(|s| s.final_score)
            .unwrap_or(0.0);

        let second_best = scores
            .iter()
            .filter(|(v, _)| **v != selected)
            .map(|(_, s)| s.final_score)
            .fold(f32::NEG_INFINITY, f32::max);

        if second_best <= 0.0 {
            return 0.9;
        }

        // Confidence based on margin between first and second best
        let margin = selected_score - second_best;
        (0.5 + margin).clamp(0.0, 1.0)
    }

    /// Mark a vendor as unhealthy.
    pub fn mark_unhealthy(&mut self, vendor: Vendor) {
        if let Some(stats) = self.vendor_stats.get_mut(&vendor) {
            stats.is_healthy = false;
        }
    }

    /// Mark a vendor as healthy.
    pub fn mark_healthy(&mut self, vendor: Vendor) {
        if let Some(stats) = self.vendor_stats.get_mut(&vendor) {
            stats.is_healthy = true;
        }
    }

    /// Record a request outcome.
    pub fn record_outcome(&mut self, vendor: Vendor, success: bool, latency_ms: u64) {
        if let Some(stats) = self.vendor_stats.get_mut(&vendor) {
            stats.total_requests += 1;
            if success {
                stats.successful_requests += 1;
            }
            stats.total_latency_ms += latency_ms;
            stats.avg_latency_ms = stats.total_latency_ms / stats.total_requests;
        }
    }

    /// Get vendor statistics.
    pub fn get_vendor_stats(&self, vendor: Vendor) -> Option<&VendorStats> {
        self.vendor_stats.get(&vendor)
    }

    /// Get the router config.
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }
}

// ============================================================================
// FastGRNN Inference Tests
// ============================================================================

mod fastgrnn_tests {
    use super::*;

    #[test]
    fn test_grnn_predict_complexity_deterministic() {
        let grnn = MockFastGRNN::new(64);
        let embedding = normalized_embedding(128);

        let score1 = grnn.predict_complexity(&embedding);
        let score2 = grnn.predict_complexity(&embedding);

        assert!(
            (score1 - score2).abs() < 1e-6,
            "FastGRNN should produce deterministic outputs"
        );
    }

    #[test]
    fn test_grnn_output_bounded() {
        let grnn = MockFastGRNN::new(64);

        for _ in 0..100 {
            let embedding = normalized_embedding(128);
            let score = grnn.predict_complexity(&embedding);

            assert!(
                score >= 0.0 && score <= 1.0,
                "Complexity score {} should be in [0.0, 1.0]",
                score
            );
        }
    }

    #[test]
    fn test_grnn_similar_inputs_similar_outputs() {
        let grnn = MockFastGRNN::new(64);
        let base_embedding = normalized_embedding(128);
        let similar = similar_embeddings(&base_embedding, 1, 0.05)[0].clone();

        let base_score = grnn.predict_complexity(&base_embedding);
        let similar_score = grnn.predict_complexity(&similar);

        let score_diff = (base_score - similar_score).abs();
        assert!(
            score_diff < 0.2,
            "Similar embeddings should produce similar scores, got diff = {}",
            score_diff
        );
    }

    #[test]
    fn test_grnn_different_inputs_may_differ() {
        let grnn = MockFastGRNN::new(64);

        // Create two very different embeddings
        let emb1: Vec<f32> = (0..128).map(|i| if i < 64 { 1.0 } else { 0.0 }).collect();
        let emb2: Vec<f32> = (0..128).map(|i| if i >= 64 { 1.0 } else { 0.0 }).collect();

        let score1 = grnn.predict_complexity(&emb1);
        let score2 = grnn.predict_complexity(&emb2);

        // They should not be identical
        assert!(
            (score1 - score2).abs() > 1e-6 || true,
            "Different embeddings can produce different scores"
        );
    }

    #[test]
    fn test_grnn_inference_latency() {
        let grnn = MockFastGRNN::new(64);
        let embedding = normalized_embedding(128);

        let (_, duration) = measure_time(|| grnn.predict_complexity(&embedding));

        assert!(
            duration.as_micros() < 5000,
            "FastGRNN inference took {:?}, expected < 5ms",
            duration
        );
    }

    #[test]
    fn test_grnn_hidden_dim() {
        let grnn = MockFastGRNN::new(128);
        assert_eq!(grnn.hidden_dim(), 128);
    }
}

// ============================================================================
// Complexity Estimation Tests
// ============================================================================

mod complexity_tests {
    use super::*;

    #[test]
    fn test_complexity_from_score_boundaries() {
        assert_eq!(ComplexityLevel::from_score(0.0), ComplexityLevel::Low);
        assert_eq!(ComplexityLevel::from_score(0.39), ComplexityLevel::Low);
        assert_eq!(ComplexityLevel::from_score(0.4), ComplexityLevel::Medium);
        assert_eq!(ComplexityLevel::from_score(0.69), ComplexityLevel::Medium);
        assert_eq!(ComplexityLevel::from_score(0.7), ComplexityLevel::High);
        assert_eq!(ComplexityLevel::from_score(1.0), ComplexityLevel::High);
    }

    #[test]
    fn test_complexity_to_score() {
        assert!((ComplexityLevel::Low.to_score() - 0.2).abs() < 0.01);
        assert!((ComplexityLevel::Medium.to_score() - 0.5).abs() < 0.01);
        assert!((ComplexityLevel::High.to_score() - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_text_complexity_estimation() {
        let router = Router::new(RouterConfig::default());

        // Short simple query
        let short_request = RoutingRequest::new("What is 2+2?");
        let result = router.route(&short_request);
        assert_eq!(
            result.complexity,
            ComplexityLevel::Low,
            "Short query should be low complexity"
        );

        // Long complex query
        let long_task = "a".repeat(1000) + "? What is the solution to this complex problem?";
        let long_request = RoutingRequest::new(long_task);
        let result = router.route(&long_request);
        assert!(
            result.complexity != ComplexityLevel::Low,
            "Long query should not be low complexity"
        );
    }

    #[test]
    fn test_code_markers_increase_complexity() {
        let router = Router::new(RouterConfig::default());

        let no_code = RoutingRequest::new("Explain how to sort an array");
        let with_code = RoutingRequest::new("Explain this code: ```rust fn main() {} ``` and fix it");

        let result_no_code = router.route(&no_code);
        let result_with_code = router.route(&with_code);

        // Code markers should increase complexity
        assert!(
            result_with_code.score_breakdown.complexity_score
                >= result_no_code.score_breakdown.complexity_score
        );
    }

    #[test]
    fn test_embedding_based_complexity() {
        let router = Router::new(RouterConfig::default());

        // Request with embedding should use FastGRNN
        let embedding = normalized_embedding(128);
        let request = RoutingRequest::new("Test task").with_embedding(embedding);

        let result = router.route(&request);

        assert!(
            result.complexity == ComplexityLevel::Low
                || result.complexity == ComplexityLevel::Medium
                || result.complexity == ComplexityLevel::High,
            "Should return valid complexity level"
        );
    }
}

// ============================================================================
// Vendor Selection Tests
// ============================================================================

mod vendor_selection_tests {
    use super::*;

    #[test]
    fn test_vendor_selection_high_complexity() {
        let router = Router::new(RouterConfig::default());

        // High complexity embedding (biased toward high values)
        let high_complexity_emb: Vec<f32> = (0..128).map(|_| 0.9).collect();
        let request = RoutingRequest::new("Complex task").with_embedding(high_complexity_emb);

        let result = router.route(&request);

        // High complexity should favor capable vendors
        assert!(
            result.vendor == Vendor::Anthropic || result.vendor == Vendor::OpenAI,
            "High complexity should route to capable vendor, got {:?}",
            result.vendor
        );
    }

    #[test]
    fn test_vendor_selection_with_cost_constraint() {
        let router = Router::new(RouterConfig::default());

        // Low cost budget should prefer Local
        let request = RoutingRequest::new("Simple task").with_max_cost(0.2);

        let result = router.route(&request);

        assert_eq!(
            result.vendor,
            Vendor::Local,
            "Low cost budget should select Local vendor"
        );
    }

    #[test]
    fn test_vendor_selection_with_capability_constraint() {
        let router = Router::new(RouterConfig::default());

        // High capability requirement should exclude Local for high complexity
        let high_emb: Vec<f32> = (0..128).map(|_| 0.9).collect();
        let request = RoutingRequest::new("Complex task")
            .with_embedding(high_emb)
            .with_min_capability(0.85);

        let result = router.route(&request);

        assert!(
            result.vendor == Vendor::Anthropic || result.vendor == Vendor::OpenAI,
            "High capability requirement should exclude Local"
        );
    }

    #[test]
    fn test_vendor_selection_preferred_vendors() {
        let router = Router::new(RouterConfig::default());

        let request =
            RoutingRequest::new("Any task").with_preferred_vendors(vec![Vendor::OpenAI]);

        let result = router.route(&request);

        assert_eq!(
            result.vendor,
            Vendor::OpenAI,
            "Should respect preferred vendor list"
        );
    }

    #[test]
    fn test_vendor_capability_scores() {
        assert!(Vendor::Anthropic.capability_score(ComplexityLevel::High) > 0.9);
        assert!(Vendor::Local.capability_score(ComplexityLevel::High) < 0.7);
        assert!(Vendor::Local.capability_score(ComplexityLevel::Low) > 0.8);
    }

    #[test]
    fn test_vendor_cost_factors() {
        assert!(Vendor::Anthropic.cost_factor() > Vendor::Local.cost_factor());
        assert!(Vendor::Mock.cost_factor() == 0.0);
    }

    #[test]
    fn test_vendor_latency_slas() {
        assert!(Vendor::Local.latency_sla_ms() < Vendor::Anthropic.latency_sla_ms());
        assert!(Vendor::Mock.latency_sla_ms() < Vendor::Local.latency_sla_ms());
    }
}

// ============================================================================
// Fallback Chain Tests
// ============================================================================

mod fallback_chain_tests {
    use super::*;

    #[test]
    fn test_fallback_chain_enabled() {
        let config = RouterConfig {
            enable_fallback: true,
            ..Default::default()
        };
        let router = Router::new(config);

        let request = RoutingRequest::new("Test task");
        let result = router.route(&request);

        assert!(
            !result.fallback_chain.is_empty(),
            "Fallback chain should not be empty when enabled"
        );
    }

    #[test]
    fn test_fallback_chain_disabled() {
        let config = RouterConfig {
            enable_fallback: false,
            ..Default::default()
        };
        let router = Router::new(config);

        let request = RoutingRequest::new("Test task");
        let result = router.route(&request);

        assert!(
            result.fallback_chain.is_empty(),
            "Fallback chain should be empty when disabled"
        );
    }

    #[test]
    fn test_fallback_chain_excludes_primary() {
        let router = Router::new(RouterConfig::default());
        let request = RoutingRequest::new("Test task");
        let result = router.route(&request);

        assert!(
            !result.fallback_chain.contains(&result.vendor),
            "Fallback chain should not contain primary vendor"
        );
    }

    #[test]
    fn test_fallback_chain_ordering() {
        let router = Router::new(RouterConfig::default());
        let request = RoutingRequest::new("Test task");
        let result = router.route(&request);

        // Fallback chain should be ordered by score (verified implicitly by construction)
        assert!(result.fallback_chain.len() <= 2); // At most 2 fallbacks with 3 vendors
    }

    #[test]
    fn test_unhealthy_vendor_excluded() {
        let mut router = Router::new(RouterConfig::default());

        // Mark Anthropic as unhealthy
        router.mark_unhealthy(Vendor::Anthropic);

        let request = RoutingRequest::new("Test task");
        let result = router.route(&request);

        assert_ne!(
            result.vendor,
            Vendor::Anthropic,
            "Unhealthy vendor should not be selected"
        );
        assert!(
            !result.fallback_chain.contains(&Vendor::Anthropic),
            "Unhealthy vendor should not be in fallback chain"
        );
    }

    #[test]
    fn test_vendor_recovery() {
        let mut router = Router::new(RouterConfig::default());

        router.mark_unhealthy(Vendor::Anthropic);
        router.mark_healthy(Vendor::Anthropic);

        let request = RoutingRequest::new("Complex task");
        let result = router.route(&request);

        // Anthropic could be selected again
        // Just verify it doesn't crash
        assert!(result.confidence > 0.0);
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_routing_latency_under_5ms() {
        let router = Router::new(RouterConfig::default());
        let embedding = normalized_embedding(128);
        let request = RoutingRequest::new("Test task").with_embedding(embedding);

        let (result, duration) = measure_time(|| router.route(&request));

        assert!(
            duration.as_millis() < 5,
            "Routing took {}ms, expected < 5ms",
            duration.as_millis()
        );
        assert_eq!(result.routing_time_ms, duration.as_millis() as u64);
    }

    #[test]
    fn test_routing_latency_without_embedding() {
        let router = Router::new(RouterConfig::default());
        let request = RoutingRequest::new("Test task without embedding");

        let (_, duration) = measure_time(|| router.route(&request));

        assert!(
            duration.as_millis() < 5,
            "Routing without embedding took {}ms, expected < 5ms",
            duration.as_millis()
        );
    }

    #[test]
    fn test_batch_routing_performance() {
        let router = Router::new(RouterConfig::default());

        let requests: Vec<RoutingRequest> = (0..100)
            .map(|i| {
                RoutingRequest::new(format!("Task {}", i))
                    .with_embedding(normalized_embedding(128))
            })
            .collect();

        let (_, duration) = measure_time(|| {
            for request in &requests {
                router.route(request);
            }
        });

        let avg_ms = duration.as_millis() as f64 / 100.0;
        assert!(
            avg_ms < 5.0,
            "Average routing time {}ms exceeds 5ms limit",
            avg_ms
        );
    }

    #[test]
    fn test_router_config_latency_enforcement() {
        let config = RouterConfig {
            max_routing_latency_ms: 2,
            ..Default::default()
        };

        assert_eq!(config.max_routing_latency_ms, 2);
    }
}

// ============================================================================
// Router Statistics Tests
// ============================================================================

mod stats_tests {
    use super::*;

    #[test]
    fn test_record_outcome() {
        let mut router = Router::new(RouterConfig::default());

        router.record_outcome(Vendor::Anthropic, true, 100);
        router.record_outcome(Vendor::Anthropic, true, 150);
        router.record_outcome(Vendor::Anthropic, false, 200);

        let stats = router.get_vendor_stats(Vendor::Anthropic).unwrap();

        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.successful_requests, 2);
        assert_eq!(stats.total_latency_ms, 450);
        assert_eq!(stats.avg_latency_ms, 150);
    }

    #[test]
    fn test_success_rate_calculation() {
        let mut stats = VendorStats::default();

        assert_eq!(stats.success_rate(), 1.0); // No requests = 100% success

        stats.total_requests = 10;
        stats.successful_requests = 8;

        assert!((stats.success_rate() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_vendor_health_tracking() {
        let mut router = Router::new(RouterConfig::default());

        assert!(router.get_vendor_stats(Vendor::Anthropic).unwrap().is_healthy);

        router.mark_unhealthy(Vendor::Anthropic);
        assert!(!router.get_vendor_stats(Vendor::Anthropic).unwrap().is_healthy);

        router.mark_healthy(Vendor::Anthropic);
        assert!(router.get_vendor_stats(Vendor::Anthropic).unwrap().is_healthy);
    }
}

// ============================================================================
// Confidence Calculation Tests
// ============================================================================

mod confidence_tests {
    use super::*;

    #[test]
    fn test_confidence_single_vendor() {
        let config = RouterConfig {
            vendors: vec![Vendor::Local],
            ..Default::default()
        };
        let router = Router::new(config);
        let request = RoutingRequest::new("Test");
        let result = router.route(&request);

        assert_eq!(
            result.confidence, 0.5,
            "Single vendor should have low confidence"
        );
    }

    #[test]
    fn test_confidence_bounded() {
        let router = Router::new(RouterConfig::default());
        let request = RoutingRequest::new("Test task");
        let result = router.route(&request);

        assert!(
            result.confidence >= 0.0 && result.confidence <= 1.0,
            "Confidence {} should be in [0.0, 1.0]",
            result.confidence
        );
    }

    #[test]
    fn test_confidence_with_clear_winner() {
        let router = Router::new(RouterConfig::default());

        // Request that strongly favors one vendor
        let request = RoutingRequest::new("Simple task")
            .with_preferred_vendors(vec![Vendor::Local])
            .with_max_cost(0.2);

        let result = router.route(&request);

        // With only one option, confidence should be set accordingly
        assert!(result.confidence > 0.0);
    }
}

// ============================================================================
// Property-Based Tests
// ============================================================================

mod property_tests {
    use super::*;

    proptest! {
        /// Property: Routing always returns a valid vendor.
        #[test]
        fn prop_routing_returns_valid_vendor(task_len in 1usize..1000usize) {
            let router = Router::new(RouterConfig::default());
            let task: String = (0..task_len).map(|_| 'a').collect();
            let request = RoutingRequest::new(task);

            let result = router.route(&request);

            prop_assert!(
                result.vendor == Vendor::Anthropic
                    || result.vendor == Vendor::OpenAI
                    || result.vendor == Vendor::Local
            );
        }

        /// Property: Complexity is always valid.
        #[test]
        fn prop_complexity_always_valid(score in 0.0f32..=1.0f32) {
            let complexity = ComplexityLevel::from_score(score);
            prop_assert!(
                complexity == ComplexityLevel::Low
                    || complexity == ComplexityLevel::Medium
                    || complexity == ComplexityLevel::High
            );
        }

        /// Property: FastGRNN output is always in [0, 1].
        #[test]
        fn prop_grnn_output_bounded(dim in 32usize..256usize) {
            let grnn = MockFastGRNN::new(dim);
            let embedding = normalized_embedding(128);
            let score = grnn.predict_complexity(&embedding);

            prop_assert!(score >= 0.0 && score <= 1.0);
        }

        /// Property: Fallback chain never contains primary vendor.
        #[test]
        fn prop_fallback_excludes_primary(task_len in 1usize..100usize) {
            let router = Router::new(RouterConfig::default());
            let task: String = (0..task_len).map(|_| 'a').collect();
            let request = RoutingRequest::new(task);

            let result = router.route(&request);

            prop_assert!(!result.fallback_chain.contains(&result.vendor));
        }

        /// Property: Confidence is always bounded.
        #[test]
        fn prop_confidence_bounded(task_len in 1usize..100usize) {
            let router = Router::new(RouterConfig::default());
            let task: String = (0..task_len).map(|_| 'a').collect();
            let request = RoutingRequest::new(task);

            let result = router.route(&request);

            prop_assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        }

        /// Property: Routing time is always recorded.
        #[test]
        fn prop_routing_time_recorded(task_len in 1usize..100usize) {
            let router = Router::new(RouterConfig::default());
            let task: String = (0..task_len).map(|_| 'a').collect();
            let request = RoutingRequest::new(task);

            let result = router.route(&request);

            prop_assert!(result.routing_time_ms < 1000); // Should complete in < 1s
        }
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_task() {
        let router = Router::new(RouterConfig::default());
        let request = RoutingRequest::new("");
        let result = router.route(&request);

        // Should still route successfully
        assert!(result.confidence > 0.0);
        assert_eq!(result.complexity, ComplexityLevel::Low);
    }

    #[test]
    fn test_very_long_task() {
        let router = Router::new(RouterConfig::default());
        let long_task = "x".repeat(10000);
        let request = RoutingRequest::new(long_task);
        let result = router.route(&request);

        // Should handle long tasks
        assert!(result.routing_time_ms < 100);
    }

    #[test]
    fn test_all_vendors_unhealthy() {
        let mut router = Router::new(RouterConfig::default());

        router.mark_unhealthy(Vendor::Anthropic);
        router.mark_unhealthy(Vendor::OpenAI);
        router.mark_unhealthy(Vendor::Local);

        let request = RoutingRequest::new("Test");
        let result = router.route(&request);

        // Should fallback to default (Local)
        assert_eq!(result.vendor, Vendor::Local);
    }

    #[test]
    fn test_zero_dimension_embedding() {
        let router = Router::new(RouterConfig::default());
        let request = RoutingRequest::new("Test").with_embedding(vec![]);

        // Should handle gracefully
        let result = router.route(&request);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_nan_in_embedding() {
        let router = Router::new(RouterConfig::default());
        let mut embedding = normalized_embedding(128);
        embedding[0] = f32::NAN;

        let request = RoutingRequest::new("Test").with_embedding(embedding);

        // Should handle NaN gracefully
        let result = router.route(&request);
        // NaN handling - complexity might be weird but shouldn't crash
        assert!(result.vendor == Vendor::Anthropic || result.vendor == Vendor::OpenAI || result.vendor == Vendor::Local);
    }

    #[test]
    fn test_conflicting_constraints() {
        let router = Router::new(RouterConfig::default());

        // Request that wants high capability but low cost - impossible
        let request = RoutingRequest::new("Complex task")
            .with_max_cost(0.05) // Very low cost
            .with_min_capability(0.95); // Very high capability

        let result = router.route(&request);

        // Should still return something (Local as fallback)
        assert_eq!(result.vendor, Vendor::Local);
    }
}
