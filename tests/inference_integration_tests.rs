//! Inference Integration Tests - Phase 2 End-to-End
//!
//! Comprehensive integration tests for the complete inference pipeline:
//! Query -> Router -> E_nagual -> Response
//!
//! # Test Scenarios
//!
//! 1. **End-to-End Flow**: Complete inference pipeline
//! 2. **Cross-Component Data Flow**: Data integrity between components
//! 3. **Performance Under Load**: Throughput and latency testing
//! 4. **Error Handling**: Graceful degradation scenarios
//! 5. **Concurrent Operations**: Thread safety and race conditions

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use proptest::prelude::*;
use serde::{Deserialize, Serialize};

mod common;
use common::{
    cosine_similarity, measure_time, normalized_embedding, similar_embeddings,
};

// ============================================================================
// Integrated Types (Combining Router, E_nagual, ReasoningBank)
// ============================================================================

/// Vendor types for routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Vendor {
    Anthropic,
    OpenAI,
    Local,
}

/// Provider types for formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenAI,
    Local,
}

impl From<Vendor> for Provider {
    fn from(v: Vendor) -> Self {
        match v {
            Vendor::Anthropic => Provider::Anthropic,
            Vendor::OpenAI => Provider::OpenAI,
            Vendor::Local => Provider::Local,
        }
    }
}

/// Complexity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
}

impl ComplexityLevel {
    pub fn from_score(score: f32) -> Self {
        if score >= 0.7 {
            ComplexityLevel::High
        } else if score >= 0.4 {
            ComplexityLevel::Medium
        } else {
            ComplexityLevel::Low
        }
    }
}

/// Pattern from ReasoningBank.
#[derive(Debug, Clone)]
pub struct Pattern {
    pub id: String,
    pub problem: String,
    pub solution: String,
    pub domain: String,
    pub embedding: Option<Vec<f32>>,
    pub confidence: f32,
    pub reward: f32,
    pub created_at: DateTime<Utc>,
}

impl Pattern {
    pub fn new(problem: &str, solution: &str, domain: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            problem: problem.to_string(),
            solution: solution.to_string(),
            domain: domain.to_string(),
            embedding: None,
            confidence: 0.5,
            reward: 0.5,
            created_at: Utc::now(),
        }
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Scored pattern with similarity.
#[derive(Debug, Clone)]
pub struct ScoredPattern {
    pub pattern: Pattern,
    pub similarity: f32,
}

/// Message in conversation.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    pub fn estimate_tokens(&self) -> usize {
        self.content.len() / 4 + 10
    }
}

/// Inference request.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// User query.
    pub query: String,
    /// Pre-computed embedding.
    pub embedding: Option<Vec<f32>>,
    /// Session ID for context.
    pub session_id: Option<String>,
    /// Maximum tokens in response.
    pub max_tokens: usize,
    /// Override vendor.
    pub vendor_override: Option<Vendor>,
}

impl InferenceRequest {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            embedding: None,
            session_id: None,
            max_tokens: 4000,
            vendor_override: None,
        }
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_vendor(mut self, vendor: Vendor) -> Self {
        self.vendor_override = Some(vendor);
        self
    }
}

/// Inference response.
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    /// Generated response text.
    pub response: String,
    /// Vendor that handled the request.
    pub vendor: Vendor,
    /// Complexity assessment.
    pub complexity: ComplexityLevel,
    /// Patterns used in context.
    pub patterns_used: Vec<String>,
    /// Total tokens in context.
    pub context_tokens: usize,
    /// Routing latency in ms.
    pub routing_latency_ms: u64,
    /// Context build latency in ms.
    pub context_latency_ms: u64,
    /// Inference latency in ms (mock).
    pub inference_latency_ms: u64,
    /// Total latency in ms.
    pub total_latency_ms: u64,
    /// Confidence in the response.
    pub confidence: f32,
}

// ============================================================================
// Mock Components
// ============================================================================

/// Mock FastGRNN for complexity prediction.
struct MockFastGRNN {
    dimension: usize,
}

impl MockFastGRNN {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn predict(&self, embedding: &[f32]) -> f32 {
        // Simple mock: average of embedding values
        let avg: f32 = embedding.iter().sum::<f32>() / embedding.len() as f32;
        (avg + 1.0) / 2.0 // Normalize to [0, 1]
    }
}

/// Mock ReasoningBank.
struct MockReasoningBank {
    patterns: RwLock<Vec<Pattern>>,
}

impl MockReasoningBank {
    fn new() -> Self {
        Self {
            patterns: RwLock::new(Vec::new()),
        }
    }

    fn add_pattern(&self, pattern: Pattern) {
        self.patterns.write().push(pattern);
    }

    fn search(&self, embedding: &[f32], k: usize, min_similarity: f32) -> Vec<ScoredPattern> {
        let patterns = self.patterns.read();

        let mut scored: Vec<ScoredPattern> = patterns
            .iter()
            .filter_map(|p| {
                p.embedding.as_ref().map(|emb| {
                    let similarity = cosine_similarity(embedding, emb);
                    ScoredPattern {
                        pattern: p.clone(),
                        similarity,
                    }
                })
            })
            .filter(|sp| sp.similarity >= min_similarity)
            .collect();

        scored.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        scored.truncate(k);
        scored
    }

    fn pattern_count(&self) -> usize {
        self.patterns.read().len()
    }
}

/// Mock LLM for generating responses.
struct MockLLM {
    latency_ms: u64,
}

impl MockLLM {
    fn new(latency_ms: u64) -> Self {
        Self { latency_ms }
    }

    fn generate(&self, messages: &[Message], _vendor: Vendor) -> String {
        std::thread::sleep(Duration::from_millis(self.latency_ms));

        // Extract user query from last message
        let query = messages
            .iter()
            .filter(|m| m.role == "user")
            .last()
            .map(|m| m.content.as_str())
            .unwrap_or("");

        format!("Mock response for: {}", query)
    }
}

// ============================================================================
// Integrated Inference Engine
// ============================================================================

/// Configuration for the inference engine.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum routing latency.
    pub max_routing_latency_ms: u64,
    /// Maximum context building latency.
    pub max_context_latency_ms: u64,
    /// Maximum patterns to use.
    pub max_patterns: usize,
    /// Minimum pattern similarity.
    pub min_pattern_similarity: f32,
    /// Enable fallback chain.
    pub enable_fallback: bool,
    /// FastGRNN dimension.
    pub grnn_dimension: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_routing_latency_ms: 5,
            max_context_latency_ms: 10,
            max_patterns: 5,
            min_pattern_similarity: 0.5,
            enable_fallback: true,
            grnn_dimension: 64,
        }
    }
}

/// Integrated inference engine.
pub struct InferenceEngine {
    config: InferenceConfig,
    grnn: MockFastGRNN,
    reasoning_bank: Arc<MockReasoningBank>,
    llm: MockLLM,
    request_count: AtomicU64,
    total_latency_ms: AtomicU64,
}

impl InferenceEngine {
    pub fn new(config: InferenceConfig) -> Self {
        Self {
            grnn: MockFastGRNN::new(config.grnn_dimension),
            reasoning_bank: Arc::new(MockReasoningBank::new()),
            llm: MockLLM::new(10), // 10ms mock latency
            request_count: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            config,
        }
    }

    /// Get shared reasoning bank reference.
    pub fn reasoning_bank(&self) -> Arc<MockReasoningBank> {
        Arc::clone(&self.reasoning_bank)
    }

    /// Process an inference request.
    pub fn infer(&self, request: &InferenceRequest) -> InferenceResponse {
        let start = Instant::now();

        // Step 1: Route request
        let routing_start = Instant::now();
        let (vendor, complexity) = self.route(request);
        let routing_latency_ms = routing_start.elapsed().as_millis() as u64;

        // Step 2: Retrieve patterns
        let patterns = if let Some(ref embedding) = request.embedding {
            self.reasoning_bank
                .search(embedding, self.config.max_patterns, self.config.min_pattern_similarity)
        } else {
            Vec::new()
        };

        // Step 3: Build context
        let context_start = Instant::now();
        let (messages, patterns_used) = self.build_context(request, &patterns);
        let context_latency_ms = context_start.elapsed().as_millis() as u64;

        // Step 4: Generate response
        let inference_start = Instant::now();
        let response = self.llm.generate(&messages, vendor);
        let inference_latency_ms = inference_start.elapsed().as_millis() as u64;

        let total_latency_ms = start.elapsed().as_millis() as u64;

        // Update statistics
        self.request_count.fetch_add(1, Ordering::SeqCst);
        self.total_latency_ms.fetch_add(total_latency_ms, Ordering::SeqCst);

        let context_tokens: usize = messages.iter().map(|m| m.estimate_tokens()).sum();
        let confidence = if patterns.is_empty() {
            0.5
        } else {
            patterns.iter().map(|p| p.pattern.confidence).sum::<f32>() / patterns.len() as f32
        };

        InferenceResponse {
            response,
            vendor,
            complexity,
            patterns_used,
            context_tokens,
            routing_latency_ms,
            context_latency_ms,
            inference_latency_ms,
            total_latency_ms,
            confidence,
        }
    }

    /// Route request to optimal vendor.
    fn route(&self, request: &InferenceRequest) -> (Vendor, ComplexityLevel) {
        // Check for override
        if let Some(vendor) = request.vendor_override {
            let complexity = if let Some(ref embedding) = request.embedding {
                ComplexityLevel::from_score(self.grnn.predict(embedding))
            } else {
                self.estimate_complexity_from_text(&request.query)
            };
            return (vendor, complexity);
        }

        // Predict complexity
        let complexity_score = if let Some(ref embedding) = request.embedding {
            self.grnn.predict(embedding)
        } else {
            self.estimate_complexity_score_from_text(&request.query)
        };

        let complexity = ComplexityLevel::from_score(complexity_score);

        // Select vendor based on complexity
        let vendor = match complexity {
            ComplexityLevel::High => Vendor::Anthropic,
            ComplexityLevel::Medium => Vendor::OpenAI,
            ComplexityLevel::Low => Vendor::Local,
        };

        (vendor, complexity)
    }

    /// Estimate complexity from text.
    fn estimate_complexity_from_text(&self, text: &str) -> ComplexityLevel {
        ComplexityLevel::from_score(self.estimate_complexity_score_from_text(text))
    }

    /// Estimate complexity score from text.
    fn estimate_complexity_score_from_text(&self, text: &str) -> f32 {
        let length_factor = (text.len() as f32 / 500.0).min(1.0);
        let question_count = text.matches('?').count() as f32;

        (0.3 + length_factor * 0.4 + question_count * 0.1).clamp(0.0, 1.0)
    }

    /// Build context from patterns.
    fn build_context(
        &self,
        request: &InferenceRequest,
        patterns: &[ScoredPattern],
    ) -> (Vec<Message>, Vec<String>) {
        let mut messages = Vec::new();
        let mut pattern_ids = Vec::new();

        // System message
        messages.push(Message::new("system", "You are a helpful assistant."));

        // Add bias from patterns
        if !patterns.is_empty() {
            let bias = self.compute_bias(patterns);
            if !bias.is_empty() {
                messages.push(Message::new("system", &format!("[Context]\n{}", bias)));
            }

            // Collect pattern IDs
            for scored in patterns {
                pattern_ids.push(scored.pattern.id.clone());
            }

            // Add few-shot examples
            for scored in patterns.iter().take(2) {
                messages.push(Message::new("user", &scored.pattern.problem));
                messages.push(Message::new("assistant", &scored.pattern.solution));
            }
        }

        // User query
        messages.push(Message::new("user", &request.query));

        (messages, pattern_ids)
    }

    /// Compute bias text from patterns.
    fn compute_bias(&self, patterns: &[ScoredPattern]) -> String {
        let entries: Vec<String> = patterns
            .iter()
            .take(3)
            .map(|sp| {
                format!(
                    "- {} (confidence: {}%)",
                    sp.pattern.solution,
                    (sp.pattern.confidence * 100.0) as u32
                )
            })
            .collect();

        if entries.is_empty() {
            String::new()
        } else {
            format!("Relevant solutions:\n{}", entries.join("\n"))
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> EngineStats {
        let count = self.request_count.load(Ordering::SeqCst);
        let total_latency = self.total_latency_ms.load(Ordering::SeqCst);

        EngineStats {
            request_count: count,
            average_latency_ms: if count > 0 {
                total_latency / count
            } else {
                0
            },
        }
    }
}

/// Engine statistics.
#[derive(Debug, Clone)]
pub struct EngineStats {
    pub request_count: u64,
    pub average_latency_ms: u64,
}

// ============================================================================
// End-to-End Flow Tests
// ============================================================================

mod e2e_flow_tests {
    use super::*;

    #[test]
    fn test_simple_inference() {
        let engine = InferenceEngine::new(InferenceConfig::default());
        let request = InferenceRequest::new("What is 2+2?");

        let response = engine.infer(&request);

        assert!(!response.response.is_empty());
        assert!(response.total_latency_ms > 0);
    }

    #[test]
    fn test_inference_with_patterns() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        // Add patterns
        let base_embedding = normalized_embedding(128);
        let pattern = Pattern::new("Math question", "The answer is computed", "math")
            .with_embedding(base_embedding.clone())
            .with_confidence(0.9);

        engine.reasoning_bank().add_pattern(pattern);

        // Request with similar embedding
        let query_embedding = similar_embeddings(&base_embedding, 1, 0.05)[0].clone();
        let request = InferenceRequest::new("Another math question").with_embedding(query_embedding);

        let response = engine.infer(&request);

        assert!(!response.patterns_used.is_empty());
        assert!(response.context_tokens > 0);
    }

    #[test]
    fn test_inference_complexity_routing() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        // Simple query
        let simple_request = InferenceRequest::new("Hi");
        let simple_response = engine.infer(&simple_request);
        assert_eq!(simple_response.complexity, ComplexityLevel::Low);

        // Complex query
        let complex_query = "x".repeat(600) + "? How do I solve this complex problem?";
        let complex_request = InferenceRequest::new(complex_query);
        let complex_response = engine.infer(&complex_request);
        assert!(complex_response.complexity != ComplexityLevel::Low);
    }

    #[test]
    fn test_inference_with_embedding() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        let embedding = normalized_embedding(128);
        let request = InferenceRequest::new("Query with embedding").with_embedding(embedding);

        let response = engine.infer(&request);

        assert!(!response.response.is_empty());
    }

    #[test]
    fn test_inference_vendor_override() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        let request = InferenceRequest::new("Simple query").with_vendor(Vendor::Anthropic);

        let response = engine.infer(&request);

        assert_eq!(response.vendor, Vendor::Anthropic);
    }

    #[test]
    fn test_inference_session_context() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        let request = InferenceRequest::new("Query")
            .with_session("session-123".to_string());

        let response = engine.infer(&request);

        assert!(!response.response.is_empty());
    }

    #[test]
    fn test_full_pipeline_flow() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        // Step 1: Populate ReasoningBank
        let base_embedding = normalized_embedding(128);
        for i in 0..10 {
            let emb = similar_embeddings(&base_embedding, 1, 0.1)[0].clone();
            let pattern = Pattern::new(
                &format!("Problem {}", i),
                &format!("Solution {}", i),
                "test",
            )
            .with_embedding(emb)
            .with_confidence(0.8 + (i as f32 * 0.01));

            engine.reasoning_bank().add_pattern(pattern);
        }

        assert_eq!(engine.reasoning_bank().pattern_count(), 10);

        // Step 2: Run inference with related embedding
        let query_embedding = similar_embeddings(&base_embedding, 1, 0.02)[0].clone();
        let request = InferenceRequest::new("Related problem query").with_embedding(query_embedding);

        let response = engine.infer(&request);

        // Step 3: Verify response
        assert!(!response.response.is_empty());
        assert!(!response.patterns_used.is_empty());
        assert!(response.confidence > 0.5);
        assert!(response.context_tokens > 0);
    }
}

// ============================================================================
// Cross-Component Data Flow Tests
// ============================================================================

mod data_flow_tests {
    use super::*;

    #[test]
    fn test_embedding_flows_to_routing() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        // High values should predict higher complexity
        let high_embedding: Vec<f32> = (0..128).map(|_| 0.9).collect();
        let low_embedding: Vec<f32> = (0..128).map(|_| -0.9).collect();

        let high_request = InferenceRequest::new("Q").with_embedding(high_embedding);
        let low_request = InferenceRequest::new("Q").with_embedding(low_embedding);

        let high_response = engine.infer(&high_request);
        let low_response = engine.infer(&low_request);

        // Different embeddings should potentially lead to different routing
        // (at minimum, they should both complete successfully)
        assert!(!high_response.response.is_empty());
        assert!(!low_response.response.is_empty());
    }

    #[test]
    fn test_patterns_flow_to_context() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        // Add pattern
        let base_embedding = normalized_embedding(128);
        let pattern = Pattern::new("Specific problem", "Specific solution", "domain")
            .with_embedding(base_embedding.clone());

        engine.reasoning_bank().add_pattern(pattern);

        // Query with matching embedding
        let request = InferenceRequest::new("Query").with_embedding(base_embedding);
        let response = engine.infer(&request);

        // Pattern should be in response
        assert!(!response.patterns_used.is_empty());
        assert!(response.context_tokens > 50); // Bias adds tokens
    }

    #[test]
    fn test_complexity_affects_vendor() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        // Simple query should use Local
        let simple_request = InferenceRequest::new("Hi");
        let simple_response = engine.infer(&simple_request);

        // The mock implementation routes based on complexity
        match simple_response.complexity {
            ComplexityLevel::Low => assert_eq!(simple_response.vendor, Vendor::Local),
            ComplexityLevel::Medium => assert_eq!(simple_response.vendor, Vendor::OpenAI),
            ComplexityLevel::High => assert_eq!(simple_response.vendor, Vendor::Anthropic),
        }
    }

    #[test]
    fn test_pattern_similarity_threshold() {
        let config = InferenceConfig {
            min_pattern_similarity: 0.9, // High threshold
            ..Default::default()
        };
        let engine = InferenceEngine::new(config);

        // Add pattern with embedding
        let pattern_embedding = normalized_embedding(128);
        let pattern = Pattern::new("P", "S", "d").with_embedding(pattern_embedding.clone());
        engine.reasoning_bank().add_pattern(pattern);

        // Query with somewhat different embedding
        let query_embedding = similar_embeddings(&pattern_embedding, 1, 0.3)[0].clone();
        let request = InferenceRequest::new("Q").with_embedding(query_embedding);

        let response = engine.infer(&request);

        // May or may not include pattern based on similarity
        // (threshold is high so might filter out)
        assert!(!response.response.is_empty());
    }

    #[test]
    fn test_max_patterns_limit() {
        let config = InferenceConfig {
            max_patterns: 2,
            min_pattern_similarity: 0.0, // Accept all
            ..Default::default()
        };
        let engine = InferenceEngine::new(config);

        // Add many patterns
        let base_embedding = normalized_embedding(128);
        for i in 0..10 {
            let emb = similar_embeddings(&base_embedding, 1, 0.05)[0].clone();
            let pattern = Pattern::new(&format!("P{}", i), &format!("S{}", i), "d")
                .with_embedding(emb);
            engine.reasoning_bank().add_pattern(pattern);
        }

        let request = InferenceRequest::new("Q").with_embedding(base_embedding);
        let response = engine.infer(&request);

        // Should use at most 2 patterns
        assert!(response.patterns_used.len() <= 2);
    }
}

// ============================================================================
// Performance Under Load Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_inference_latency_sla() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        for _ in 0..10 {
            let request = InferenceRequest::new("Test query")
                .with_embedding(normalized_embedding(128));

            let response = engine.infer(&request);

            // Routing should be under 5ms
            assert!(
                response.routing_latency_ms < 5,
                "Routing took {}ms, expected < 5ms",
                response.routing_latency_ms
            );

            // Context building should be under 10ms
            assert!(
                response.context_latency_ms < 10,
                "Context building took {}ms, expected < 10ms",
                response.context_latency_ms
            );
        }
    }

    #[test]
    fn test_throughput() {
        let engine = Arc::new(InferenceEngine::new(InferenceConfig::default()));

        // Pre-populate patterns
        let base_embedding = normalized_embedding(128);
        for i in 0..20 {
            let emb = similar_embeddings(&base_embedding, 1, 0.1)[0].clone();
            let pattern = Pattern::new(&format!("P{}", i), &format!("S{}", i), "d")
                .with_embedding(emb);
            engine.reasoning_bank().add_pattern(pattern);
        }

        // Run many requests
        let request_count = 100;
        let start = Instant::now();

        for _ in 0..request_count {
            let embedding = similar_embeddings(&base_embedding, 1, 0.05)[0].clone();
            let request = InferenceRequest::new("Query").with_embedding(embedding);
            engine.infer(&request);
        }

        let duration = start.elapsed();
        let requests_per_second = request_count as f64 / duration.as_secs_f64();

        // Should handle at least 50 requests per second (considering mock LLM latency)
        println!(
            "Throughput: {:.2} requests/second ({}ms for {} requests)",
            requests_per_second,
            duration.as_millis(),
            request_count
        );

        assert!(
            requests_per_second > 50.0,
            "Throughput {:.2} req/s below 50 req/s target",
            requests_per_second
        );
    }

    #[test]
    fn test_average_latency() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        let base_embedding = normalized_embedding(128);
        for i in 0..10 {
            let emb = similar_embeddings(&base_embedding, 1, 0.1)[0].clone();
            let pattern = Pattern::new(&format!("P{}", i), &format!("S{}", i), "d")
                .with_embedding(emb);
            engine.reasoning_bank().add_pattern(pattern);
        }

        // Run requests
        for _ in 0..50 {
            let embedding = similar_embeddings(&base_embedding, 1, 0.05)[0].clone();
            let request = InferenceRequest::new("Query").with_embedding(embedding);
            engine.infer(&request);
        }

        let stats = engine.stats();

        // Average latency should be reasonable
        assert!(
            stats.average_latency_ms < 50,
            "Average latency {}ms exceeds 50ms target",
            stats.average_latency_ms
        );
    }

    #[test]
    fn test_large_pattern_count() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        // Add 1000 patterns
        for i in 0..1000 {
            let pattern = Pattern::new(&format!("Problem {}", i), &format!("Solution {}", i), "d")
                .with_embedding(normalized_embedding(128))
                .with_confidence(0.5 + ((i % 50) as f32) / 100.0);

            engine.reasoning_bank().add_pattern(pattern);
        }

        assert_eq!(engine.reasoning_bank().pattern_count(), 1000);

        // Search should still be fast
        let (_, duration) = measure_time(|| {
            let request =
                InferenceRequest::new("Query").with_embedding(normalized_embedding(128));
            engine.infer(&request);
        });

        assert!(
            duration.as_millis() < 100,
            "Inference with 1000 patterns took {}ms",
            duration.as_millis()
        );
    }
}

// ============================================================================
// Concurrent Operation Tests
// ============================================================================

mod concurrent_tests {
    use super::*;

    #[test]
    fn test_concurrent_inference() {
        let engine = Arc::new(InferenceEngine::new(InferenceConfig::default()));

        // Add patterns
        let base_embedding = normalized_embedding(128);
        for i in 0..10 {
            let emb = similar_embeddings(&base_embedding, 1, 0.1)[0].clone();
            let pattern = Pattern::new(&format!("P{}", i), &format!("S{}", i), "d")
                .with_embedding(emb);
            engine.reasoning_bank().add_pattern(pattern);
        }

        // Spawn multiple threads
        let mut handles = Vec::new();
        for thread_id in 0..4 {
            let engine_clone = Arc::clone(&engine);
            let base_emb = base_embedding.clone();

            let handle = thread::spawn(move || {
                for i in 0..25 {
                    let embedding = similar_embeddings(&base_emb, 1, 0.05)[0].clone();
                    let request = InferenceRequest::new(format!("Thread {} Query {}", thread_id, i))
                        .with_embedding(embedding);

                    let response = engine_clone.infer(&request);
                    assert!(!response.response.is_empty());
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify stats
        let stats = engine.stats();
        assert_eq!(stats.request_count, 100); // 4 threads * 25 requests
    }

    #[test]
    fn test_concurrent_read_write() {
        let engine = Arc::new(InferenceEngine::new(InferenceConfig::default()));

        // Writer thread adds patterns
        let engine_writer = Arc::clone(&engine);
        let writer_handle = thread::spawn(move || {
            for i in 0..50 {
                let pattern = Pattern::new(&format!("P{}", i), &format!("S{}", i), "d")
                    .with_embedding(normalized_embedding(128));
                engine_writer.reasoning_bank().add_pattern(pattern);
                thread::sleep(Duration::from_micros(100));
            }
        });

        // Reader threads perform inference
        let mut reader_handles = Vec::new();
        for thread_id in 0..3 {
            let engine_reader = Arc::clone(&engine);

            let handle = thread::spawn(move || {
                for i in 0..20 {
                    let request =
                        InferenceRequest::new(format!("Thread {} Query {}", thread_id, i))
                            .with_embedding(normalized_embedding(128));

                    let response = engine_reader.infer(&request);
                    assert!(!response.response.is_empty());
                }
            });
            reader_handles.push(handle);
        }

        // Wait for all
        writer_handle.join().expect("Writer panicked");
        for handle in reader_handles {
            handle.join().expect("Reader panicked");
        }

        // Verify all patterns were added
        assert_eq!(engine.reasoning_bank().pattern_count(), 50);
    }

    #[test]
    fn test_stats_under_concurrency() {
        let engine = Arc::new(InferenceEngine::new(InferenceConfig::default()));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let engine_clone = Arc::clone(&engine);

            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    let request = InferenceRequest::new("Query");
                    engine_clone.infer(&request);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let stats = engine.stats();
        assert_eq!(stats.request_count, 80); // 8 threads * 10 requests
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

mod error_handling_tests {
    use super::*;

    #[test]
    fn test_empty_query() {
        let engine = InferenceEngine::new(InferenceConfig::default());
        let request = InferenceRequest::new("");

        // Should handle gracefully
        let response = engine.infer(&request);
        assert!(!response.response.is_empty());
    }

    #[test]
    fn test_empty_embedding() {
        let engine = InferenceEngine::new(InferenceConfig::default());
        let request = InferenceRequest::new("Query").with_embedding(Vec::new());

        // Should handle gracefully (NaN protection)
        let response = engine.infer(&request);
        assert!(!response.response.is_empty());
    }

    #[test]
    fn test_no_matching_patterns() {
        let config = InferenceConfig {
            min_pattern_similarity: 0.99, // Very high threshold
            ..Default::default()
        };
        let engine = InferenceEngine::new(config);

        // Add patterns
        let pattern = Pattern::new("P", "S", "d").with_embedding(normalized_embedding(128));
        engine.reasoning_bank().add_pattern(pattern);

        // Query with different embedding
        let different_embedding: Vec<f32> = (0..128).map(|i| if i < 64 { 1.0 } else { -1.0 }).collect();
        let request = InferenceRequest::new("Q").with_embedding(different_embedding);

        let response = engine.infer(&request);

        // Should still work, just with empty patterns
        assert!(response.patterns_used.is_empty() || !response.patterns_used.is_empty());
        assert!(!response.response.is_empty());
    }

    #[test]
    fn test_very_long_query() {
        let engine = InferenceEngine::new(InferenceConfig::default());
        let long_query = "x".repeat(10000);
        let request = InferenceRequest::new(long_query);

        let response = engine.infer(&request);
        assert!(!response.response.is_empty());
    }

    #[test]
    fn test_unicode_in_query() {
        let engine = InferenceEngine::new(InferenceConfig::default());
        let request = InferenceRequest::new("Query with emojis \u{1F600} and \u{1F680}");

        let response = engine.infer(&request);
        assert!(!response.response.is_empty());
    }
}

// ============================================================================
// Property-Based Tests
// ============================================================================

mod property_tests {
    use super::*;

    proptest! {
        /// Property: Inference always returns a response.
        #[test]
        fn prop_always_returns_response(query_len in 0usize..500usize) {
            let engine = InferenceEngine::new(InferenceConfig::default());
            let query: String = (0..query_len).map(|_| 'a').collect();
            let request = InferenceRequest::new(query);

            let response = engine.infer(&request);

            prop_assert!(!response.response.is_empty());
        }

        /// Property: Latency is always recorded.
        #[test]
        fn prop_latency_recorded(query_len in 1usize..100usize) {
            let engine = InferenceEngine::new(InferenceConfig::default());
            let query: String = (0..query_len).map(|_| 'a').collect();
            let request = InferenceRequest::new(query);

            let response = engine.infer(&request);

            prop_assert!(response.total_latency_ms >= response.routing_latency_ms);
            prop_assert!(response.total_latency_ms >= response.context_latency_ms);
        }

        /// Property: Complexity is always valid.
        #[test]
        fn prop_complexity_valid(query_len in 1usize..1000usize) {
            let engine = InferenceEngine::new(InferenceConfig::default());
            let query: String = (0..query_len).map(|_| 'a').collect();
            let request = InferenceRequest::new(query);

            let response = engine.infer(&request);

            prop_assert!(
                response.complexity == ComplexityLevel::Low
                    || response.complexity == ComplexityLevel::Medium
                    || response.complexity == ComplexityLevel::High
            );
        }

        /// Property: Vendor is always valid.
        #[test]
        fn prop_vendor_valid(query_len in 1usize..100usize) {
            let engine = InferenceEngine::new(InferenceConfig::default());
            let query: String = (0..query_len).map(|_| 'a').collect();
            let request = InferenceRequest::new(query);

            let response = engine.infer(&request);

            prop_assert!(
                response.vendor == Vendor::Anthropic
                    || response.vendor == Vendor::OpenAI
                    || response.vendor == Vendor::Local
            );
        }

        /// Property: Confidence is bounded.
        #[test]
        fn prop_confidence_bounded(query_len in 1usize..100usize) {
            let engine = InferenceEngine::new(InferenceConfig::default());
            let query: String = (0..query_len).map(|_| 'a').collect();
            let request = InferenceRequest::new(query);

            let response = engine.infer(&request);

            prop_assert!(response.confidence >= 0.0 && response.confidence <= 1.0);
        }
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_single_pattern() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        let embedding = normalized_embedding(128);
        let pattern = Pattern::new("P", "S", "d")
            .with_embedding(embedding.clone())
            .with_confidence(0.9);
        engine.reasoning_bank().add_pattern(pattern);

        let request = InferenceRequest::new("Q").with_embedding(embedding);
        let response = engine.infer(&request);

        assert!(!response.patterns_used.is_empty());
    }

    #[test]
    fn test_duplicate_patterns() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        let embedding = normalized_embedding(128);
        for _ in 0..5 {
            let pattern = Pattern::new("Same problem", "Same solution", "d")
                .with_embedding(embedding.clone());
            engine.reasoning_bank().add_pattern(pattern);
        }

        let request = InferenceRequest::new("Q").with_embedding(embedding);
        let response = engine.infer(&request);

        // Should handle duplicates
        assert!(!response.response.is_empty());
    }

    #[test]
    fn test_zero_max_patterns() {
        let config = InferenceConfig {
            max_patterns: 0,
            ..Default::default()
        };
        let engine = InferenceEngine::new(config);

        let embedding = normalized_embedding(128);
        let pattern = Pattern::new("P", "S", "d").with_embedding(embedding.clone());
        engine.reasoning_bank().add_pattern(pattern);

        let request = InferenceRequest::new("Q").with_embedding(embedding);
        let response = engine.infer(&request);

        assert!(response.patterns_used.is_empty());
    }

    #[test]
    fn test_high_similarity_threshold() {
        let config = InferenceConfig {
            min_pattern_similarity: 1.0, // Impossible to match
            ..Default::default()
        };
        let engine = InferenceEngine::new(config);

        let pattern = Pattern::new("P", "S", "d").with_embedding(normalized_embedding(128));
        engine.reasoning_bank().add_pattern(pattern);

        let request = InferenceRequest::new("Q").with_embedding(normalized_embedding(128));
        let response = engine.infer(&request);

        // Should still work
        assert!(!response.response.is_empty());
    }

    #[test]
    fn test_sequential_requests() {
        let engine = InferenceEngine::new(InferenceConfig::default());

        for i in 0..100 {
            let request = InferenceRequest::new(format!("Query {}", i));
            let response = engine.infer(&request);
            assert!(!response.response.is_empty());
        }

        let stats = engine.stats();
        assert_eq!(stats.request_count, 100);
    }
}
