//! E_nagual Tests - Phase 2 Inference Layer
//!
//! Comprehensive test suite for the E_nagual component that handles:
//! - Bias computation from patterns
//! - Provider formatting (Anthropic, OpenAI, Local)
//! - Few-shot example generation
//! - Context building
//! - Integration with ReasoningBank
//!
//! # Test Categories
//!
//! 1. **Bias Computation Tests**: Validate bias generation from patterns
//! 2. **Provider Formatting Tests**: Ensure correct API formatting
//! 3. **Few-Shot Generation Tests**: Test example selection and formatting
//! 4. **Context Building Tests**: Verify context assembly
//! 5. **Integration Tests**: End-to-end with ReasoningBank

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use proptest::prelude::*;
use serde::{Deserialize, Serialize};

mod common;
use common::{
    cosine_similarity, measure_time, normalized_embedding, similar_embeddings, TestPattern,
};

// ============================================================================
// E_nagual Types (Mirroring production types for testing)
// ============================================================================

/// Provider types for LLM API formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Provider {
    Anthropic,
    OpenAI,
    Local,
}

impl Provider {
    /// Get the message role name for this provider.
    pub fn role_user(&self) -> &'static str {
        match self {
            Provider::Anthropic => "user",
            Provider::OpenAI => "user",
            Provider::Local => "user",
        }
    }

    /// Get the assistant role name for this provider.
    pub fn role_assistant(&self) -> &'static str {
        match self {
            Provider::Anthropic => "assistant",
            Provider::OpenAI => "assistant",
            Provider::Local => "assistant",
        }
    }

    /// Get the system role name for this provider.
    pub fn role_system(&self) -> &'static str {
        match self {
            Provider::Anthropic => "system",
            Provider::OpenAI => "system",
            Provider::Local => "system",
        }
    }

    /// Maximum context length for this provider.
    pub fn max_context_tokens(&self) -> usize {
        match self {
            Provider::Anthropic => 200000,
            Provider::OpenAI => 128000,
            Provider::Local => 8000,
        }
    }
}

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }

    /// Estimate token count (rough approximation).
    pub fn estimate_tokens(&self) -> usize {
        // Rough estimate: 4 chars per token
        (self.content.len() / 4) + 10 // +10 for role overhead
    }
}

/// A pattern from the ReasoningBank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub id: String,
    pub problem: String,
    pub solution: String,
    pub domain: String,
    pub context: Option<String>,
    pub confidence: f32,
    pub reward: f32,
    pub success_rate: f32,
    pub embedding: Option<Vec<f32>>,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
}

impl Pattern {
    pub fn new(problem: impl Into<String>, solution: impl Into<String>, domain: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            problem: problem.into(),
            solution: solution.into(),
            domain: domain.into(),
            context: None,
            confidence: 0.5,
            reward: 0.5,
            success_rate: 0.5,
            embedding: None,
            tags: Vec::new(),
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

    pub fn with_reward(mut self, reward: f32) -> Self {
        self.reward = reward.clamp(0.0, 1.0);
        self
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Reliability score combining confidence and success rate.
    pub fn reliability_score(&self) -> f32 {
        (self.confidence + self.success_rate) / 2.0
    }
}

/// Scored pattern with similarity.
#[derive(Debug, Clone)]
pub struct ScoredPattern {
    pub pattern: Pattern,
    pub similarity: f32,
    pub final_score: f32,
}

/// Configuration for bias computation.
#[derive(Debug, Clone)]
pub struct BiasConfig {
    /// Maximum number of patterns to use for bias.
    pub max_patterns: usize,
    /// Minimum similarity threshold.
    pub min_similarity: f32,
    /// Minimum reliability threshold.
    pub min_reliability: f32,
    /// Weight for recency in bias calculation.
    pub recency_weight: f32,
    /// Weight for reliability in bias calculation.
    pub reliability_weight: f32,
    /// Weight for similarity in bias calculation.
    pub similarity_weight: f32,
}

impl Default for BiasConfig {
    fn default() -> Self {
        Self {
            max_patterns: 5,
            min_similarity: 0.5,
            min_reliability: 0.3,
            recency_weight: 0.2,
            reliability_weight: 0.3,
            similarity_weight: 0.5,
        }
    }
}

/// Computed bias for prompt injection.
#[derive(Debug, Clone)]
pub struct ComputedBias {
    /// The bias text to inject.
    pub bias_text: String,
    /// Patterns used to compute bias.
    pub source_patterns: Vec<String>,
    /// Aggregate confidence in the bias.
    pub confidence: f32,
    /// Number of tokens in the bias.
    pub token_count: usize,
}

/// Configuration for few-shot example generation.
#[derive(Debug, Clone)]
pub struct FewShotConfig {
    /// Maximum number of examples.
    pub max_examples: usize,
    /// Maximum tokens per example.
    pub max_tokens_per_example: usize,
    /// Include problem statement.
    pub include_problem: bool,
    /// Include solution.
    pub include_solution: bool,
    /// Include context.
    pub include_context: bool,
    /// Format style.
    pub format_style: FewShotStyle,
}

impl Default for FewShotConfig {
    fn default() -> Self {
        Self {
            max_examples: 3,
            max_tokens_per_example: 500,
            include_problem: true,
            include_solution: true,
            include_context: true,
            format_style: FewShotStyle::Conversational,
        }
    }
}

/// Few-shot formatting style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FewShotStyle {
    Conversational,
    Structured,
    Compact,
}

/// A generated few-shot example.
#[derive(Debug, Clone)]
pub struct FewShotExample {
    pub user_message: String,
    pub assistant_message: String,
    pub source_pattern_id: String,
}

impl FewShotExample {
    pub fn estimate_tokens(&self) -> usize {
        (self.user_message.len() + self.assistant_message.len()) / 4 + 20
    }
}

/// Configuration for context building.
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Maximum total tokens for context.
    pub max_context_tokens: usize,
    /// Include system instructions.
    pub include_system: bool,
    /// Include few-shot examples.
    pub include_few_shot: bool,
    /// Include bias from patterns.
    pub include_bias: bool,
    /// Provider for formatting.
    pub provider: Provider,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 4000,
            include_system: true,
            include_few_shot: true,
            include_bias: true,
            provider: Provider::Anthropic,
        }
    }
}

/// Built context ready for API call.
#[derive(Debug, Clone)]
pub struct BuiltContext {
    /// Messages in provider format.
    pub messages: Vec<Message>,
    /// System prompt (if separate).
    pub system_prompt: Option<String>,
    /// Total estimated tokens.
    pub total_tokens: usize,
    /// Breakdown of token usage.
    pub token_breakdown: TokenBreakdown,
}

/// Breakdown of token usage.
#[derive(Debug, Clone, Default)]
pub struct TokenBreakdown {
    pub system_tokens: usize,
    pub few_shot_tokens: usize,
    pub bias_tokens: usize,
    pub user_tokens: usize,
}

/// E_nagual engine for bias computation and context building.
#[derive(Debug)]
pub struct ENagual {
    bias_config: BiasConfig,
    few_shot_config: FewShotConfig,
    context_config: ContextConfig,
}

impl ENagual {
    pub fn new(
        bias_config: BiasConfig,
        few_shot_config: FewShotConfig,
        context_config: ContextConfig,
    ) -> Self {
        Self {
            bias_config,
            few_shot_config,
            context_config,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(
            BiasConfig::default(),
            FewShotConfig::default(),
            ContextConfig::default(),
        )
    }

    /// Compute bias from patterns.
    pub fn compute_bias(&self, patterns: &[ScoredPattern]) -> ComputedBias {
        // Filter and limit patterns
        let filtered: Vec<&ScoredPattern> = patterns
            .iter()
            .filter(|p| {
                p.similarity >= self.bias_config.min_similarity
                    && p.pattern.reliability_score() >= self.bias_config.min_reliability
            })
            .take(self.bias_config.max_patterns)
            .collect();

        if filtered.is_empty() {
            return ComputedBias {
                bias_text: String::new(),
                source_patterns: Vec::new(),
                confidence: 0.0,
                token_count: 0,
            };
        }

        // Build bias text
        let mut bias_parts = Vec::new();
        let mut source_patterns = Vec::new();
        let mut total_confidence = 0.0;

        for scored in &filtered {
            let pattern = &scored.pattern;
            source_patterns.push(pattern.id.clone());

            // Compute weighted score
            let score = self.compute_pattern_score(scored);
            total_confidence += score;

            let bias_entry = format!(
                "Based on similar situation ({}% confidence): {}",
                (pattern.confidence * 100.0) as u32,
                pattern.solution
            );
            bias_parts.push(bias_entry);
        }

        let bias_text = if bias_parts.is_empty() {
            String::new()
        } else {
            format!(
                "Relevant patterns from past experience:\n{}",
                bias_parts.join("\n")
            )
        };

        let confidence = total_confidence / filtered.len() as f32;
        let token_count = bias_text.len() / 4;

        ComputedBias {
            bias_text,
            source_patterns,
            confidence,
            token_count,
        }
    }

    /// Compute weighted score for a pattern.
    fn compute_pattern_score(&self, scored: &ScoredPattern) -> f32 {
        let pattern = &scored.pattern;

        // Recency score (decay over 30 days)
        let age_days = (Utc::now() - pattern.created_at).num_days() as f32;
        let recency = (1.0 - (age_days / 30.0).min(1.0)).max(0.0);

        let reliability = pattern.reliability_score();
        let similarity = scored.similarity;

        self.bias_config.recency_weight * recency
            + self.bias_config.reliability_weight * reliability
            + self.bias_config.similarity_weight * similarity
    }

    /// Generate few-shot examples from patterns.
    pub fn generate_few_shot(&self, patterns: &[ScoredPattern]) -> Vec<FewShotExample> {
        let mut examples = Vec::new();
        let mut total_tokens = 0;

        for scored in patterns.iter().take(self.few_shot_config.max_examples) {
            let pattern = &scored.pattern;

            let user_message = if self.few_shot_config.include_context {
                if let Some(ref ctx) = pattern.context {
                    format!("{}\nContext: {}", pattern.problem, ctx)
                } else {
                    pattern.problem.clone()
                }
            } else {
                pattern.problem.clone()
            };

            let assistant_message = pattern.solution.clone();

            let example = FewShotExample {
                user_message,
                assistant_message,
                source_pattern_id: pattern.id.clone(),
            };

            let example_tokens = example.estimate_tokens();
            if total_tokens + example_tokens > self.few_shot_config.max_tokens_per_example * self.few_shot_config.max_examples {
                break;
            }

            total_tokens += example_tokens;
            examples.push(example);
        }

        examples
    }

    /// Format few-shot examples for a provider.
    pub fn format_few_shot_for_provider(
        &self,
        examples: &[FewShotExample],
        provider: Provider,
    ) -> Vec<Message> {
        let mut messages = Vec::new();

        for example in examples {
            messages.push(Message::new(
                provider.role_user(),
                &example.user_message,
            ));
            messages.push(Message::new(
                provider.role_assistant(),
                &example.assistant_message,
            ));
        }

        messages
    }

    /// Build complete context for an API call.
    pub fn build_context(
        &self,
        user_query: &str,
        patterns: &[ScoredPattern],
        system_instructions: Option<&str>,
    ) -> BuiltContext {
        let provider = self.context_config.provider;
        let mut messages = Vec::new();
        let mut breakdown = TokenBreakdown::default();

        // System instructions
        let system_prompt = if self.context_config.include_system {
            let sys = system_instructions.unwrap_or("You are a helpful assistant.");
            breakdown.system_tokens = sys.len() / 4;
            Some(sys.to_string())
        } else {
            None
        };

        // Add system message if provider expects it inline
        if let Some(ref sys) = system_prompt {
            if provider != Provider::Anthropic {
                messages.push(Message::system(sys.clone()));
            }
        }

        // Bias from patterns
        if self.context_config.include_bias && !patterns.is_empty() {
            let bias = self.compute_bias(patterns);
            if !bias.bias_text.is_empty() {
                breakdown.bias_tokens = bias.token_count;
                messages.push(Message::system(format!(
                    "[Context from past patterns]\n{}",
                    bias.bias_text
                )));
            }
        }

        // Few-shot examples
        if self.context_config.include_few_shot && !patterns.is_empty() {
            let examples = self.generate_few_shot(patterns);
            let example_messages = self.format_few_shot_for_provider(&examples, provider);
            breakdown.few_shot_tokens = example_messages
                .iter()
                .map(|m| m.estimate_tokens())
                .sum();
            messages.extend(example_messages);
        }

        // User query
        breakdown.user_tokens = user_query.len() / 4;
        messages.push(Message::user(user_query));

        let total_tokens = breakdown.system_tokens
            + breakdown.bias_tokens
            + breakdown.few_shot_tokens
            + breakdown.user_tokens;

        BuiltContext {
            messages,
            system_prompt,
            total_tokens,
            token_breakdown: breakdown,
        }
    }

    /// Get config references.
    pub fn bias_config(&self) -> &BiasConfig {
        &self.bias_config
    }

    pub fn few_shot_config(&self) -> &FewShotConfig {
        &self.few_shot_config
    }

    pub fn context_config(&self) -> &ContextConfig {
        &self.context_config
    }
}

// ============================================================================
// Test Helpers
// ============================================================================

fn create_test_pattern(problem: &str, solution: &str, domain: &str) -> Pattern {
    Pattern::new(problem, solution, domain)
        .with_embedding(normalized_embedding(128))
        .with_confidence(0.8)
        .with_reward(0.7)
}

fn create_scored_pattern(pattern: Pattern, similarity: f32) -> ScoredPattern {
    ScoredPattern {
        pattern,
        similarity,
        final_score: similarity * 0.8,
    }
}

fn create_test_patterns(count: usize) -> Vec<ScoredPattern> {
    (0..count)
        .map(|i| {
            let pattern = create_test_pattern(
                &format!("Problem {} description", i),
                &format!("Solution {} approach", i),
                &format!("domain.{}", i % 3),
            );
            create_scored_pattern(pattern, 0.9 - (i as f32 * 0.05))
        })
        .collect()
}

// ============================================================================
// Bias Computation Tests
// ============================================================================

mod bias_computation_tests {
    use super::*;

    #[test]
    fn test_compute_bias_basic() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(3);

        let bias = e_nagual.compute_bias(&patterns);

        assert!(!bias.bias_text.is_empty());
        assert!(!bias.source_patterns.is_empty());
        assert!(bias.confidence > 0.0);
    }

    #[test]
    fn test_compute_bias_empty_patterns() {
        let e_nagual = ENagual::with_defaults();
        let patterns: Vec<ScoredPattern> = Vec::new();

        let bias = e_nagual.compute_bias(&patterns);

        assert!(bias.bias_text.is_empty());
        assert!(bias.source_patterns.is_empty());
        assert_eq!(bias.confidence, 0.0);
    }

    #[test]
    fn test_compute_bias_respects_max_patterns() {
        let config = BiasConfig {
            max_patterns: 2,
            ..Default::default()
        };
        let e_nagual = ENagual::new(config, FewShotConfig::default(), ContextConfig::default());
        let patterns = create_test_patterns(10);

        let bias = e_nagual.compute_bias(&patterns);

        assert!(bias.source_patterns.len() <= 2);
    }

    #[test]
    fn test_compute_bias_filters_low_similarity() {
        let config = BiasConfig {
            min_similarity: 0.8,
            ..Default::default()
        };
        let e_nagual = ENagual::new(config, FewShotConfig::default(), ContextConfig::default());

        let patterns = vec![
            create_scored_pattern(create_test_pattern("P1", "S1", "d1"), 0.9),
            create_scored_pattern(create_test_pattern("P2", "S2", "d2"), 0.7), // Below threshold
            create_scored_pattern(create_test_pattern("P3", "S3", "d3"), 0.85),
        ];

        let bias = e_nagual.compute_bias(&patterns);

        // Should only include patterns with similarity >= 0.8
        assert!(bias.source_patterns.len() <= 2);
    }

    #[test]
    fn test_compute_bias_filters_low_reliability() {
        let config = BiasConfig {
            min_reliability: 0.6,
            min_similarity: 0.0, // Don't filter by similarity
            ..Default::default()
        };
        let e_nagual = ENagual::new(config, FewShotConfig::default(), ContextConfig::default());

        let mut low_reliability_pattern = create_test_pattern("P1", "S1", "d1");
        low_reliability_pattern.confidence = 0.3;
        low_reliability_pattern.success_rate = 0.2;

        let high_reliability_pattern = create_test_pattern("P2", "S2", "d2");

        let patterns = vec![
            create_scored_pattern(low_reliability_pattern, 0.9),
            create_scored_pattern(high_reliability_pattern, 0.8),
        ];

        let bias = e_nagual.compute_bias(&patterns);

        // Low reliability pattern should be filtered
        assert_eq!(bias.source_patterns.len(), 1);
    }

    #[test]
    fn test_compute_bias_includes_confidence() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(2);

        let bias = e_nagual.compute_bias(&patterns);

        assert!(bias.bias_text.contains("confidence"));
    }

    #[test]
    fn test_compute_bias_token_count() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(3);

        let bias = e_nagual.compute_bias(&patterns);

        assert!(bias.token_count > 0);
        assert!(bias.token_count < bias.bias_text.len()); // Tokens < chars
    }

    #[test]
    fn test_compute_bias_weighted_scoring() {
        let config = BiasConfig {
            recency_weight: 0.1,
            reliability_weight: 0.1,
            similarity_weight: 0.8, // Heavy weight on similarity
            ..Default::default()
        };
        let e_nagual = ENagual::new(config, FewShotConfig::default(), ContextConfig::default());

        let patterns = create_test_patterns(3);
        let bias = e_nagual.compute_bias(&patterns);

        // Should produce valid bias
        assert!(bias.confidence > 0.0 && bias.confidence <= 1.0);
    }
}

// ============================================================================
// Provider Formatting Tests
// ============================================================================

mod provider_formatting_tests {
    use super::*;

    #[test]
    fn test_anthropic_roles() {
        let provider = Provider::Anthropic;
        assert_eq!(provider.role_user(), "user");
        assert_eq!(provider.role_assistant(), "assistant");
        assert_eq!(provider.role_system(), "system");
    }

    #[test]
    fn test_openai_roles() {
        let provider = Provider::OpenAI;
        assert_eq!(provider.role_user(), "user");
        assert_eq!(provider.role_assistant(), "assistant");
        assert_eq!(provider.role_system(), "system");
    }

    #[test]
    fn test_local_roles() {
        let provider = Provider::Local;
        assert_eq!(provider.role_user(), "user");
        assert_eq!(provider.role_assistant(), "assistant");
        assert_eq!(provider.role_system(), "system");
    }

    #[test]
    fn test_max_context_tokens() {
        assert!(Provider::Anthropic.max_context_tokens() > Provider::OpenAI.max_context_tokens());
        assert!(Provider::OpenAI.max_context_tokens() > Provider::Local.max_context_tokens());
    }

    #[test]
    fn test_message_construction() {
        let msg = Message::new("user", "Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn test_message_shortcuts() {
        let system = Message::system("System prompt");
        let user = Message::user("User query");
        let assistant = Message::assistant("Response");

        assert_eq!(system.role, "system");
        assert_eq!(user.role, "user");
        assert_eq!(assistant.role, "assistant");
    }

    #[test]
    fn test_message_token_estimation() {
        let msg = Message::new("user", "This is a test message with some content");
        let tokens = msg.estimate_tokens();

        // Should be roughly content_len / 4 + overhead
        assert!(tokens > 0);
        assert!(tokens < msg.content.len());
    }

    #[test]
    fn test_format_few_shot_for_anthropic() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(2);
        let examples = e_nagual.generate_few_shot(&patterns);

        let messages = e_nagual.format_few_shot_for_provider(&examples, Provider::Anthropic);

        // Should have 2 messages per example (user + assistant)
        assert_eq!(messages.len(), examples.len() * 2);

        // Alternating roles
        for (i, msg) in messages.iter().enumerate() {
            if i % 2 == 0 {
                assert_eq!(msg.role, "user");
            } else {
                assert_eq!(msg.role, "assistant");
            }
        }
    }

    #[test]
    fn test_format_few_shot_for_openai() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(2);
        let examples = e_nagual.generate_few_shot(&patterns);

        let messages = e_nagual.format_few_shot_for_provider(&examples, Provider::OpenAI);

        assert_eq!(messages.len(), examples.len() * 2);
    }
}

// ============================================================================
// Few-Shot Generation Tests
// ============================================================================

mod few_shot_generation_tests {
    use super::*;

    #[test]
    fn test_generate_few_shot_basic() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(5);

        let examples = e_nagual.generate_few_shot(&patterns);

        assert!(!examples.is_empty());
        assert!(examples.len() <= e_nagual.few_shot_config().max_examples);
    }

    #[test]
    fn test_generate_few_shot_respects_max() {
        let config = FewShotConfig {
            max_examples: 2,
            ..Default::default()
        };
        let e_nagual = ENagual::new(BiasConfig::default(), config, ContextConfig::default());
        let patterns = create_test_patterns(10);

        let examples = e_nagual.generate_few_shot(&patterns);

        assert!(examples.len() <= 2);
    }

    #[test]
    fn test_generate_few_shot_includes_problem() {
        let config = FewShotConfig {
            include_problem: true,
            ..Default::default()
        };
        let e_nagual = ENagual::new(BiasConfig::default(), config, ContextConfig::default());

        let pattern = create_test_pattern("Specific problem text", "Solution", "domain");
        let patterns = vec![create_scored_pattern(pattern, 0.9)];

        let examples = e_nagual.generate_few_shot(&patterns);

        assert!(examples[0].user_message.contains("Specific problem text"));
    }

    #[test]
    fn test_generate_few_shot_includes_solution() {
        let config = FewShotConfig {
            include_solution: true,
            ..Default::default()
        };
        let e_nagual = ENagual::new(BiasConfig::default(), config, ContextConfig::default());

        let pattern = create_test_pattern("Problem", "Specific solution text", "domain");
        let patterns = vec![create_scored_pattern(pattern, 0.9)];

        let examples = e_nagual.generate_few_shot(&patterns);

        assert!(examples[0].assistant_message.contains("Specific solution text"));
    }

    #[test]
    fn test_generate_few_shot_includes_context() {
        let config = FewShotConfig {
            include_context: true,
            ..Default::default()
        };
        let e_nagual = ENagual::new(BiasConfig::default(), config, ContextConfig::default());

        let pattern = create_test_pattern("Problem", "Solution", "domain")
            .with_context("Additional context info");
        let patterns = vec![create_scored_pattern(pattern, 0.9)];

        let examples = e_nagual.generate_few_shot(&patterns);

        assert!(examples[0].user_message.contains("context") || examples[0].user_message.contains("Context"));
    }

    #[test]
    fn test_generate_few_shot_without_context() {
        let config = FewShotConfig {
            include_context: false,
            ..Default::default()
        };
        let e_nagual = ENagual::new(BiasConfig::default(), config, ContextConfig::default());

        let pattern = create_test_pattern("Problem only", "Solution", "domain")
            .with_context("Should not appear");
        let patterns = vec![create_scored_pattern(pattern, 0.9)];

        let examples = e_nagual.generate_few_shot(&patterns);

        assert!(!examples[0].user_message.contains("Should not appear"));
    }

    #[test]
    fn test_generate_few_shot_token_estimation() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(3);

        let examples = e_nagual.generate_few_shot(&patterns);

        for example in &examples {
            let tokens = example.estimate_tokens();
            assert!(tokens > 0);
        }
    }

    #[test]
    fn test_generate_few_shot_preserves_pattern_id() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(3);

        let examples = e_nagual.generate_few_shot(&patterns);

        for (i, example) in examples.iter().enumerate() {
            assert_eq!(example.source_pattern_id, patterns[i].pattern.id);
        }
    }

    #[test]
    fn test_generate_few_shot_empty_patterns() {
        let e_nagual = ENagual::with_defaults();
        let patterns: Vec<ScoredPattern> = Vec::new();

        let examples = e_nagual.generate_few_shot(&patterns);

        assert!(examples.is_empty());
    }
}

// ============================================================================
// Context Building Tests
// ============================================================================

mod context_building_tests {
    use super::*;

    #[test]
    fn test_build_context_basic() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(2);

        let context = e_nagual.build_context("User query", &patterns, None);

        assert!(!context.messages.is_empty());
        assert!(context.total_tokens > 0);
    }

    #[test]
    fn test_build_context_includes_user_query() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(1);

        let context = e_nagual.build_context("My specific question", &patterns, None);

        let user_messages: Vec<_> = context
            .messages
            .iter()
            .filter(|m| m.role == "user" && m.content.contains("My specific question"))
            .collect();

        assert!(!user_messages.is_empty());
    }

    #[test]
    fn test_build_context_with_system_instructions() {
        let config = ContextConfig {
            include_system: true,
            provider: Provider::Anthropic,
            ..Default::default()
        };
        let e_nagual = ENagual::new(BiasConfig::default(), FewShotConfig::default(), config);
        let patterns = create_test_patterns(1);

        let context =
            e_nagual.build_context("Query", &patterns, Some("Custom system instructions"));

        assert!(context.system_prompt.is_some());
        assert!(context.system_prompt.unwrap().contains("Custom system instructions"));
    }

    #[test]
    fn test_build_context_without_system() {
        let config = ContextConfig {
            include_system: false,
            ..Default::default()
        };
        let e_nagual = ENagual::new(BiasConfig::default(), FewShotConfig::default(), config);
        let patterns = create_test_patterns(1);

        let context = e_nagual.build_context("Query", &patterns, Some("System"));

        assert!(context.system_prompt.is_none());
    }

    #[test]
    fn test_build_context_with_bias() {
        let config = ContextConfig {
            include_bias: true,
            ..Default::default()
        };
        let e_nagual = ENagual::new(BiasConfig::default(), FewShotConfig::default(), config);
        let patterns = create_test_patterns(3);

        let context = e_nagual.build_context("Query", &patterns, None);

        assert!(context.token_breakdown.bias_tokens > 0);
    }

    #[test]
    fn test_build_context_without_bias() {
        let config = ContextConfig {
            include_bias: false,
            ..Default::default()
        };
        let e_nagual = ENagual::new(BiasConfig::default(), FewShotConfig::default(), config);
        let patterns = create_test_patterns(3);

        let context = e_nagual.build_context("Query", &patterns, None);

        assert_eq!(context.token_breakdown.bias_tokens, 0);
    }

    #[test]
    fn test_build_context_with_few_shot() {
        let config = ContextConfig {
            include_few_shot: true,
            ..Default::default()
        };
        let e_nagual = ENagual::new(BiasConfig::default(), FewShotConfig::default(), config);
        let patterns = create_test_patterns(3);

        let context = e_nagual.build_context("Query", &patterns, None);

        assert!(context.token_breakdown.few_shot_tokens > 0);
    }

    #[test]
    fn test_build_context_without_few_shot() {
        let config = ContextConfig {
            include_few_shot: false,
            ..Default::default()
        };
        let e_nagual = ENagual::new(BiasConfig::default(), FewShotConfig::default(), config);
        let patterns = create_test_patterns(3);

        let context = e_nagual.build_context("Query", &patterns, None);

        assert_eq!(context.token_breakdown.few_shot_tokens, 0);
    }

    #[test]
    fn test_build_context_token_breakdown_sum() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(3);

        let context = e_nagual.build_context("Query", &patterns, Some("System"));

        let breakdown_sum = context.token_breakdown.system_tokens
            + context.token_breakdown.few_shot_tokens
            + context.token_breakdown.bias_tokens
            + context.token_breakdown.user_tokens;

        assert_eq!(context.total_tokens, breakdown_sum);
    }

    #[test]
    fn test_build_context_empty_patterns() {
        let e_nagual = ENagual::with_defaults();
        let patterns: Vec<ScoredPattern> = Vec::new();

        let context = e_nagual.build_context("Query", &patterns, None);

        // Should still have user message
        assert!(!context.messages.is_empty());
    }

    #[test]
    fn test_build_context_user_token_count() {
        let e_nagual = ENagual::with_defaults();
        let patterns: Vec<ScoredPattern> = Vec::new();

        let long_query = "x".repeat(400); // ~100 tokens
        let context = e_nagual.build_context(&long_query, &patterns, None);

        assert!(context.token_breakdown.user_tokens >= 90);
        assert!(context.token_breakdown.user_tokens <= 110);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_full_pipeline_anthropic() {
        let config = ContextConfig {
            provider: Provider::Anthropic,
            include_system: true,
            include_bias: true,
            include_few_shot: true,
            ..Default::default()
        };
        let e_nagual = ENagual::new(BiasConfig::default(), FewShotConfig::default(), config);

        let patterns = create_test_patterns(5);
        let context = e_nagual.build_context(
            "How do I handle errors in async code?",
            &patterns,
            Some("You are a Rust expert."),
        );

        // Verify structure
        assert!(context.system_prompt.is_some());
        assert!(!context.messages.is_empty());
        assert!(context.total_tokens < Provider::Anthropic.max_context_tokens());
    }

    #[test]
    fn test_full_pipeline_openai() {
        let config = ContextConfig {
            provider: Provider::OpenAI,
            include_system: true,
            include_bias: true,
            include_few_shot: true,
            ..Default::default()
        };
        let e_nagual = ENagual::new(BiasConfig::default(), FewShotConfig::default(), config);

        let patterns = create_test_patterns(5);
        let context = e_nagual.build_context(
            "How do I handle errors?",
            &patterns,
            Some("You are helpful."),
        );

        // OpenAI has system message inline
        let has_system = context.messages.iter().any(|m| m.role == "system");
        assert!(has_system);
    }

    #[test]
    fn test_full_pipeline_local() {
        let config = ContextConfig {
            provider: Provider::Local,
            max_context_tokens: 2000, // Limited context for local
            ..Default::default()
        };
        let e_nagual = ENagual::new(BiasConfig::default(), FewShotConfig::default(), config);

        let patterns = create_test_patterns(3);
        let context = e_nagual.build_context("Simple query", &patterns, None);

        assert!(context.total_tokens < 2000);
    }

    #[test]
    fn test_reasoning_bank_integration() {
        // Simulate patterns from ReasoningBank
        let reasoning_patterns: Vec<ScoredPattern> = vec![
            create_scored_pattern(
                Pattern::new(
                    "How to handle timeouts in async code?",
                    "Use tokio::timeout wrapper with explicit duration",
                    "rust.async.timeout",
                )
                .with_confidence(0.9)
                .with_context("Common in network operations"),
                0.95,
            ),
            create_scored_pattern(
                Pattern::new(
                    "Error handling in async functions",
                    "Use anyhow::Result with context for better error messages",
                    "rust.error.async",
                )
                .with_confidence(0.85),
                0.88,
            ),
        ];

        let e_nagual = ENagual::with_defaults();
        let context = e_nagual.build_context(
            "My async function keeps timing out, how can I add proper timeout handling?",
            &reasoning_patterns,
            Some("You are a Rust async expert."),
        );

        // Context should include relevant patterns
        assert!(context.token_breakdown.bias_tokens > 0);
        assert!(context.token_breakdown.few_shot_tokens > 0);
    }

    #[test]
    fn test_pattern_reliability_propagation() {
        let mut high_reliability = create_test_pattern("P1", "S1", "d1");
        high_reliability.confidence = 0.95;
        high_reliability.success_rate = 0.9;

        let mut low_reliability = create_test_pattern("P2", "S2", "d2");
        low_reliability.confidence = 0.3;
        low_reliability.success_rate = 0.2;

        assert!(high_reliability.reliability_score() > 0.9);
        assert!(low_reliability.reliability_score() < 0.3);
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_bias_computation_performance() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(100);

        let (_, duration) = measure_time(|| e_nagual.compute_bias(&patterns));

        assert!(
            duration.as_millis() < 10,
            "Bias computation took {}ms, expected < 10ms",
            duration.as_millis()
        );
    }

    #[test]
    fn test_few_shot_generation_performance() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(100);

        let (_, duration) = measure_time(|| e_nagual.generate_few_shot(&patterns));

        assert!(
            duration.as_millis() < 5,
            "Few-shot generation took {}ms, expected < 5ms",
            duration.as_millis()
        );
    }

    #[test]
    fn test_context_building_performance() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(50);

        let (_, duration) = measure_time(|| {
            e_nagual.build_context("Query", &patterns, Some("System prompt"))
        });

        assert!(
            duration.as_millis() < 20,
            "Context building took {}ms, expected < 20ms",
            duration.as_millis()
        );
    }

    #[test]
    fn test_batch_context_building() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(10);

        let queries: Vec<String> = (0..50).map(|i| format!("Query {}", i)).collect();

        let (_, duration) = measure_time(|| {
            for query in &queries {
                e_nagual.build_context(query, &patterns, None);
            }
        });

        let avg_ms = duration.as_millis() as f64 / 50.0;
        assert!(
            avg_ms < 5.0,
            "Average context build time {}ms exceeds 5ms",
            avg_ms
        );
    }
}

// ============================================================================
// Property-Based Tests
// ============================================================================

mod property_tests {
    use super::*;

    proptest! {
        /// Property: Bias confidence is always in [0, 1].
        #[test]
        fn prop_bias_confidence_bounded(pattern_count in 1usize..20usize) {
            let e_nagual = ENagual::with_defaults();
            let patterns = create_test_patterns(pattern_count);

            let bias = e_nagual.compute_bias(&patterns);

            prop_assert!(bias.confidence >= 0.0 && bias.confidence <= 1.0);
        }

        /// Property: Token count is always non-negative.
        #[test]
        fn prop_token_count_non_negative(pattern_count in 0usize..20usize) {
            let e_nagual = ENagual::with_defaults();
            let patterns = create_test_patterns(pattern_count);

            let context = e_nagual.build_context("Query", &patterns, None);

            prop_assert!(context.total_tokens >= 0);
            prop_assert!(context.token_breakdown.user_tokens >= 0);
        }

        /// Property: Few-shot count never exceeds max.
        #[test]
        fn prop_few_shot_respects_max(
            pattern_count in 1usize..50usize,
            max_examples in 1usize..10usize
        ) {
            let config = FewShotConfig {
                max_examples,
                ..Default::default()
            };
            let e_nagual = ENagual::new(BiasConfig::default(), config, ContextConfig::default());
            let patterns = create_test_patterns(pattern_count);

            let examples = e_nagual.generate_few_shot(&patterns);

            prop_assert!(examples.len() <= max_examples);
        }

        /// Property: Messages always end with user message.
        #[test]
        fn prop_messages_end_with_user(pattern_count in 0usize..10usize) {
            let e_nagual = ENagual::with_defaults();
            let patterns = create_test_patterns(pattern_count);

            let context = e_nagual.build_context("Final user query", &patterns, None);

            if let Some(last_msg) = context.messages.last() {
                prop_assert_eq!(&last_msg.role, "user");
            }
        }

        /// Property: Provider max tokens are positive.
        #[test]
        fn prop_provider_max_tokens_positive(_seed in 0u64..100u64) {
            prop_assert!(Provider::Anthropic.max_context_tokens() > 0);
            prop_assert!(Provider::OpenAI.max_context_tokens() > 0);
            prop_assert!(Provider::Local.max_context_tokens() > 0);
        }
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_very_long_solution() {
        let long_solution = "x".repeat(10000);
        let pattern = create_test_pattern("Short problem", &long_solution, "domain");
        let patterns = vec![create_scored_pattern(pattern, 0.9)];

        let e_nagual = ENagual::with_defaults();
        let context = e_nagual.build_context("Query", &patterns, None);

        // Should handle without crashing
        assert!(context.total_tokens > 0);
    }

    #[test]
    fn test_empty_solution() {
        let pattern = create_test_pattern("Problem", "", "domain");
        let patterns = vec![create_scored_pattern(pattern, 0.9)];

        let e_nagual = ENagual::with_defaults();
        let examples = e_nagual.generate_few_shot(&patterns);

        // Should still create example
        assert_eq!(examples.len(), 1);
    }

    #[test]
    fn test_unicode_content() {
        let pattern = Pattern::new(
            "How to handle UTF-8 strings with emojis like \u{1F600}?",
            "Use String type which is always valid UTF-8",
            "rust.strings",
        );
        let patterns = vec![create_scored_pattern(pattern, 0.9)];

        let e_nagual = ENagual::with_defaults();
        let context = e_nagual.build_context("Query about \u{1F680}", &patterns, None);

        assert!(!context.messages.is_empty());
    }

    #[test]
    fn test_newlines_in_content() {
        let pattern = Pattern::new(
            "Problem\nwith\nmultiple\nlines",
            "Solution\nalso\nhas\nlines",
            "domain",
        );
        let patterns = vec![create_scored_pattern(pattern, 0.9)];

        let e_nagual = ENagual::with_defaults();
        let examples = e_nagual.generate_few_shot(&patterns);

        assert!(examples[0].user_message.contains('\n'));
        assert!(examples[0].assistant_message.contains('\n'));
    }

    #[test]
    fn test_all_patterns_filtered() {
        let config = BiasConfig {
            min_similarity: 0.99, // Very high threshold
            ..Default::default()
        };
        let e_nagual = ENagual::new(config, FewShotConfig::default(), ContextConfig::default());

        // All patterns have similarity below 0.99
        let patterns = create_test_patterns(10);
        let bias = e_nagual.compute_bias(&patterns);

        assert!(bias.bias_text.is_empty());
        assert!(bias.source_patterns.is_empty());
    }

    #[test]
    fn test_single_pattern() {
        let e_nagual = ENagual::with_defaults();
        let patterns = create_test_patterns(1);

        let bias = e_nagual.compute_bias(&patterns);
        let examples = e_nagual.generate_few_shot(&patterns);

        assert_eq!(bias.source_patterns.len(), 1);
        assert_eq!(examples.len(), 1);
    }

    #[test]
    fn test_old_pattern_recency() {
        let mut old_pattern = create_test_pattern("Old", "Solution", "domain");
        old_pattern.created_at = Utc::now() - Duration::days(60); // 60 days old

        let patterns = vec![create_scored_pattern(old_pattern, 0.9)];

        let e_nagual = ENagual::with_defaults();
        let bias = e_nagual.compute_bias(&patterns);

        // Should still include but with lower weight
        assert!(!bias.source_patterns.is_empty());
        assert!(bias.confidence > 0.0);
    }
}
