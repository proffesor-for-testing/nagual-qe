//! Pattern struct and related types for ReasoningBank.
//!
//! A Pattern represents a learned solution to a problem, with metadata
//! for tracking effectiveness, reuse, and relationships.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for a pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PatternId(pub String);

impl PatternId {
    /// Create a new random pattern ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a pattern ID from an existing string.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for PatternId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for PatternId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for PatternId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for PatternId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Category/domain classification for patterns.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternCategory {
    /// Code architecture and design patterns
    Architecture,
    /// Performance optimization patterns
    Performance,
    /// Security best practices
    Security,
    /// Testing strategies and patterns
    Testing,
    /// Error handling and resilience
    Resilience,
    /// Data management and storage
    DataManagement,
    /// API design patterns
    ApiDesign,
    /// DevOps and deployment patterns
    DevOps,
    /// Code quality and refactoring
    CodeQuality,
    /// Documentation patterns
    Documentation,
    /// Custom domain-specific category
    Custom(String),
}

impl Default for PatternCategory {
    fn default() -> Self {
        PatternCategory::Custom("general".to_string())
    }
}

impl std::fmt::Display for PatternCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternCategory::Architecture => write!(f, "architecture"),
            PatternCategory::Performance => write!(f, "performance"),
            PatternCategory::Security => write!(f, "security"),
            PatternCategory::Testing => write!(f, "testing"),
            PatternCategory::Resilience => write!(f, "resilience"),
            PatternCategory::DataManagement => write!(f, "data_management"),
            PatternCategory::ApiDesign => write!(f, "api_design"),
            PatternCategory::DevOps => write!(f, "devops"),
            PatternCategory::CodeQuality => write!(f, "code_quality"),
            PatternCategory::Documentation => write!(f, "documentation"),
            PatternCategory::Custom(s) => write!(f, "{}", s),
        }
    }
}

impl From<&str> for PatternCategory {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "architecture" => PatternCategory::Architecture,
            "performance" => PatternCategory::Performance,
            "security" => PatternCategory::Security,
            "testing" => PatternCategory::Testing,
            "resilience" => PatternCategory::Resilience,
            "data_management" | "datamanagement" => PatternCategory::DataManagement,
            "api_design" | "apidesign" => PatternCategory::ApiDesign,
            "devops" => PatternCategory::DevOps,
            "code_quality" | "codequality" => PatternCategory::CodeQuality,
            "documentation" => PatternCategory::Documentation,
            other => PatternCategory::Custom(other.to_string()),
        }
    }
}

/// Failure mode classification based on MAST taxonomy.
///
/// From "Why Do Multi-Agent LLM Systems Fail?" (Berkeley, 2025):
/// 14 failure modes in 3 overarching categories.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureMode {
    /// Specification issues: unclear requirements, ambiguous goals, missing context
    SpecificationIssue,
    /// Inter-agent misalignment: conflicting strategies, poor coordination, state drift
    InterAgentMisalignment,
    /// Task verification failures: wrong output, incomplete results, false positives
    TaskVerification,
    /// Resource/environment issues: timeouts, OOM, external dependency failures
    ResourceIssue,
    /// Unknown or unclassified failure
    Unknown,
}

impl std::fmt::Display for FailureMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FailureMode::SpecificationIssue => write!(f, "specification_issue"),
            FailureMode::InterAgentMisalignment => write!(f, "inter_agent_misalignment"),
            FailureMode::TaskVerification => write!(f, "task_verification"),
            FailureMode::ResourceIssue => write!(f, "resource_issue"),
            FailureMode::Unknown => write!(f, "unknown"),
        }
    }
}

impl From<&str> for FailureMode {
    fn from(s: &str) -> Self {
        match s.to_lowercase().replace('-', "_").as_str() {
            "specification_issue" | "specification" | "spec" => FailureMode::SpecificationIssue,
            "inter_agent_misalignment" | "misalignment" | "coordination" => FailureMode::InterAgentMisalignment,
            "task_verification" | "verification" | "output" => FailureMode::TaskVerification,
            "resource_issue" | "resource" | "timeout" | "oom" => FailureMode::ResourceIssue,
            _ => FailureMode::Unknown,
        }
    }
}

/// Bayesian quality score using Beta distribution.
///
/// NOTE: This is distinct from `crate::learning::transfer::BetaParams` which is used
/// for domain transfer learning. This BetaParams tracks pattern quality via Bayesian scoring.
///
/// Beta(alpha, beta) where:
/// - alpha accumulates positive evidence (successes)
/// - beta accumulates negative evidence (failures)
/// - mean = alpha / (alpha + beta)
/// - Starts at Beta(1, 1) (uniform prior -- no evidence)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaParams {
    alpha: f64,
    beta: f64,
}

impl BetaParams {
    /// Create with uniform prior Beta(1, 1).
    pub fn new() -> Self {
        Self { alpha: 1.0, beta: 1.0 }
    }

    /// Create from explicit alpha/beta.
    pub fn with_params(alpha: f64, beta: f64) -> Self {
        Self {
            alpha: alpha.max(0.01), // prevent degenerate
            beta: beta.max(0.01),
        }
    }

    /// Convert from existing linear reward [0.0, 1.0].
    /// Maps to Beta with 10 pseudo-observations.
    pub fn from_reward(reward: f32) -> Self {
        let r = (reward as f64).clamp(0.0, 1.0);
        Self {
            alpha: r * 10.0 + 1.0,
            beta: (1.0 - r) * 10.0 + 1.0,
        }
    }

    /// Mean of the distribution = alpha / (alpha + beta).
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Total observations (excluding prior).
    pub fn observations(&self) -> f64 {
        self.alpha + self.beta - 2.0 // subtract prior
    }

    /// Record positive outcome.
    pub fn upvote(&mut self) {
        self.alpha += 1.0;
    }

    /// Record negative outcome.
    pub fn downvote(&mut self) {
        self.beta += 1.0;
    }

    /// Get alpha value.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get beta value.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Variance of the distribution.
    pub fn variance(&self) -> f64 {
        let sum = self.alpha + self.beta;
        (self.alpha * self.beta) / (sum * sum * (sum + 1.0))
    }
}

impl Default for BetaParams {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for BetaParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.2} (Beta({:.1}, {:.1}), {} obs)",
            self.mean(),
            self.alpha,
            self.beta,
            self.observations() as i64
        )
    }
}

#[cfg(test)]
mod beta_tests {
    use super::*;

    #[test]
    fn test_uniform_prior() {
        let bp = BetaParams::new();
        assert!((bp.mean() - 0.5).abs() < 1e-10);
        assert!((bp.alpha() - 1.0).abs() < 1e-10);
        assert!((bp.beta() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_upvote_increases_mean() {
        let mut bp = BetaParams::new();
        let before = bp.mean();
        bp.upvote();
        assert!(bp.mean() > before);
    }

    #[test]
    fn test_downvote_decreases_mean() {
        let mut bp = BetaParams::new();
        let before = bp.mean();
        bp.downvote();
        assert!(bp.mean() < before);
    }

    #[test]
    fn test_from_reward() {
        // reward = 0.8f32 -> cast to f64 introduces minor float imprecision
        // alpha ~ 0.8*10 + 1 = 9.0, beta ~ 0.2*10 + 1 = 3.0
        let bp = BetaParams::from_reward(0.8);
        assert!((bp.alpha() - 9.0).abs() < 1e-6);
        assert!((bp.beta() - 3.0).abs() < 1e-6);
        // mean ~ 9/12 = 0.75
        assert!((bp.mean() - 0.75).abs() < 1e-6);

        // reward = 0.5: alpha = 6.0, beta = 6.0, mean = 0.5
        let bp5 = BetaParams::from_reward(0.5);
        assert!((bp5.mean() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_observations() {
        let mut bp = BetaParams::new();
        assert!((bp.observations() - 0.0).abs() < 1e-10);
        bp.upvote();
        assert!((bp.observations() - 1.0).abs() < 1e-10);
        bp.downvote();
        assert!((bp.observations() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_display() {
        let bp = BetaParams::new();
        let display = format!("{}", bp);
        assert!(display.contains("0.50"));
        assert!(display.contains("Beta(1.0, 1.0)"));
        assert!(display.contains("0 obs"));
    }

    #[test]
    fn test_serde_roundtrip() {
        let bp = BetaParams::with_params(5.0, 3.0);
        let json = serde_json::to_string(&bp).unwrap();
        let deserialized: BetaParams = serde_json::from_str(&json).unwrap();
        assert!((deserialized.alpha() - 5.0).abs() < 1e-10);
        assert!((deserialized.beta() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_default_is_uniform() {
        let bp = BetaParams::default();
        assert!((bp.mean() - 0.5).abs() < 1e-10);
        assert!((bp.observations() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance() {
        let bp = BetaParams::new();
        // Variance of Beta(1,1) = 1*1 / (2*2*3) = 1/12
        assert!((bp.variance() - 1.0 / 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_with_params_clamps_degenerate() {
        let bp = BetaParams::with_params(-1.0, 0.0);
        assert!((bp.alpha() - 0.01).abs() < 1e-10);
        assert!((bp.beta() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_from_reward_edge_cases() {
        // reward = 0.0 -> alpha=1, beta=11 -> mean = 1/12
        let bp0 = BetaParams::from_reward(0.0);
        assert!((bp0.alpha() - 1.0).abs() < 1e-6);
        assert!((bp0.beta() - 11.0).abs() < 1e-6);

        // reward = 1.0 -> alpha=11, beta=1 -> mean = 11/12
        let bp1 = BetaParams::from_reward(1.0);
        assert!((bp1.alpha() - 11.0).abs() < 1e-6);
        assert!((bp1.beta() - 1.0).abs() < 1e-6);

        // reward clamped above 1.0
        let bp_over = BetaParams::from_reward(1.5);
        assert!((bp_over.alpha() - 11.0).abs() < 1e-6);
    }
}

/// Additional metadata for a pattern.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternMetadata {
    /// Language or framework this pattern applies to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Framework this pattern is specific to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub framework: Option<String>,

    /// Version constraints for applicability
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version_constraint: Option<String>,

    /// Source of the pattern (e.g., "code_review", "documentation", "user")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// URL reference for the pattern origin
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_url: Option<String>,

    /// Additional key-value pairs
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl PatternMetadata {
    /// Create new empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the framework.
    pub fn with_framework(mut self, framework: impl Into<String>) -> Self {
        self.framework = Some(framework.into());
        self
    }

    /// Set the source.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Add an extra key-value pair.
    pub fn with_extra(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }
}

/// A learned pattern representing a problem-solution pair with metadata.
///
/// Patterns are the core unit of knowledge in the ReasoningBank. Each pattern
/// captures a problem, its solution, and metrics for tracking effectiveness
/// and reuse over time.
///
/// # Fields (15+ as per ADR-005)
///
/// 1. `id` - Unique identifier (UUID)
/// 2. `timestamp` - When the pattern was created
/// 3. `category` - Domain/category classification
/// 4. `problem` - Description of the problem
/// 5. `solution` - The solution approach
/// 6. `context` - Additional context for the pattern
/// 7. `effectiveness` - How effective this pattern is (0.0-1.0)
/// 8. `reuse_count` - Number of times this pattern has been reused
/// 9. `reward` - Reward signal from usage (0.0-1.0)
/// 10. `success` - Whether the pattern was successful
/// 11. `critique` - Self-critique or feedback on the pattern
/// 12. `agent_id` - ID of the agent that created this pattern
/// 13. `session_id` - Session during which the pattern was learned
/// 14. `confidence` - Confidence in the pattern (0.0-1.0)
/// 15. `embedding` - Vector embedding for similarity search
/// 16. `tags` - Searchable tags
/// 17. `related_patterns` - IDs of related patterns
/// 18. `metadata` - Additional flexible metadata
/// 19. `updated_at` - Last modification timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Unique identifier for this pattern
    id: PatternId,

    /// Timestamp when the pattern was first created
    timestamp: DateTime<Utc>,

    /// Last update timestamp
    updated_at: DateTime<Utc>,

    /// Category/domain for the pattern
    category: PatternCategory,

    /// Description of the problem this pattern solves
    problem: String,

    /// The solution or approach
    solution: String,

    /// Additional context about when/where this pattern applies
    #[serde(default)]
    context: String,

    /// Effectiveness score (0.0-1.0)
    #[serde(default)]
    effectiveness: f32,

    /// Number of times this pattern has been successfully reused
    #[serde(default)]
    reuse_count: u32,

    /// Reward signal from reinforcement learning (0.0-1.0)
    #[serde(default)]
    reward: f32,

    /// Whether the pattern application was successful
    #[serde(default = "default_success")]
    success: bool,

    /// Self-critique or user feedback on the pattern
    #[serde(default)]
    critique: String,

    /// ID of the agent that created/learned this pattern
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<String>,

    /// Session ID during which this pattern was learned
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,

    /// Confidence level in this pattern (0.0-1.0)
    #[serde(default = "default_confidence")]
    confidence: f32,

    /// Vector embedding for similarity search
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding: Option<Vec<f32>>,

    /// Which embedder produced the embedding vector (e.g., "onnx" or "hash").
    /// Used to prevent mixing ONNX and hash embeddings in similarity queries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embedding_method: Option<String>,

    /// Surprise score: how novel this pattern was when stored (0.0=duplicate, 1.0=unique).
    /// Inspired by Titans (Google, 2024): neural long-term memory learns to memorize
    /// what is "surprising" — i.e., furthest from existing knowledge.
    #[serde(default)]
    surprise_score: f32,

    /// Failure mode classification per MAST taxonomy (Berkeley, 2025).
    /// Tracks WHY a pattern failed, not just that it did.
    #[serde(skip_serializing_if = "Option::is_none")]
    failure_mode: Option<FailureMode>,

    /// Chunk-level embeddings for long solutions (DESC method).
    /// Each inner Vec<f32> is a 128-dim embedding of a ~300-char chunk.
    /// Enables fine-grained semantic search without diluting the signal.
    #[serde(skip_serializing_if = "Option::is_none")]
    chunk_embeddings: Option<Vec<Vec<f32>>>,

    /// Satisfaction score for user/agent feedback tracking (0.0-1.0).
    /// Uses Beta distribution: score = alpha / (alpha + beta) where
    /// alpha = satisfaction_trials * satisfaction_score + 1
    /// beta = satisfaction_trials * (1 - satisfaction_score) + 1
    /// Initialized to 0.5 (neutral prior).
    #[serde(default = "default_satisfaction_score")]
    satisfaction_score: f32,

    /// Number of satisfaction feedback trials recorded.
    /// Used with satisfaction_score for Bayesian updating.
    #[serde(default)]
    satisfaction_trials: u32,

    /// BLAKE3 hash of the problem+solution content for deduplication.
    /// Computed as blake3::hash(format!("{}\n{}", problem, solution)).to_hex()
    #[serde(skip_serializing_if = "Option::is_none")]
    content_hash: Option<String>,

    /// Short title (10 words max) for quick listing in pyramid summaries.
    /// Part of the pyramid summary system: title (10 words) -> summary (50 words) -> full content.
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,

    /// Summary (50 words max) for scanning in pyramid summaries.
    /// Part of the pyramid summary system: title (10 words) -> summary (50 words) -> full content.
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,

    /// Parent pattern ID for lineage tracking (KOS P0).
    /// Set when pattern is derived from another via merge, consolidation, improvement, fork, or transfer.
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_id: Option<PatternId>,

    /// How this pattern was derived from its parent (KOS P0).
    #[serde(skip_serializing_if = "Option::is_none")]
    derivation_type: Option<crate::lineage::DerivationType>,

    /// Depth in the lineage chain (0 = original, 1 = first derivation, etc.)
    #[serde(default)]
    lineage_depth: u32,

    /// Bayesian quality score using Beta distribution.
    /// Tracks accumulated positive/negative evidence independently of the linear `reward` field.
    /// Starts at Beta(1, 1) (uniform prior) for backward compatibility.
    #[serde(default)]
    bayesian_score: BetaParams,

    /// Searchable tags
    #[serde(default)]
    tags: Vec<String>,

    /// IDs of related patterns
    #[serde(default)]
    related_patterns: Vec<PatternId>,

    /// Additional flexible metadata
    #[serde(default)]
    metadata: PatternMetadata,
}

fn default_success() -> bool {
    true
}

fn default_confidence() -> f32 {
    0.5
}

fn default_satisfaction_score() -> f32 {
    0.5
}

impl Pattern {
    /// Create a new pattern builder.
    pub fn builder() -> PatternBuilder {
        PatternBuilder::new()
    }

    /// Create a new pattern with required fields.
    pub fn new(problem: impl Into<String>, solution: impl Into<String>) -> Self {
        Self::builder()
            .problem(problem)
            .solution(solution)
            .build()
    }

    // Getters

    /// Get the pattern ID.
    pub fn id(&self) -> &PatternId {
        &self.id
    }

    /// Get the creation timestamp.
    pub fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    /// Get the last update timestamp.
    pub fn updated_at(&self) -> DateTime<Utc> {
        self.updated_at
    }

    /// Get the category.
    pub fn category(&self) -> &PatternCategory {
        &self.category
    }

    /// Get the problem description.
    pub fn problem(&self) -> &str {
        &self.problem
    }

    /// Get the solution.
    pub fn solution(&self) -> &str {
        &self.solution
    }

    /// Get the context.
    pub fn context(&self) -> &str {
        &self.context
    }

    /// Get the effectiveness score.
    pub fn effectiveness(&self) -> f32 {
        self.effectiveness
    }

    /// Get the reuse count.
    pub fn reuse_count(&self) -> u32 {
        self.reuse_count
    }

    /// Get the reward signal.
    pub fn reward(&self) -> f32 {
        self.reward
    }

    /// Get whether the pattern was successful.
    pub fn success(&self) -> bool {
        self.success
    }

    /// Get the critique.
    pub fn critique(&self) -> &str {
        &self.critique
    }

    /// Get the agent ID.
    pub fn agent_id(&self) -> Option<&str> {
        self.agent_id.as_deref()
    }

    /// Get the session ID.
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Get the confidence level.
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Get the embedding vector.
    pub fn embedding(&self) -> Option<&[f32]> {
        self.embedding.as_deref()
    }

    /// Get the embedding method (e.g., "onnx" or "hash").
    pub fn embedding_method(&self) -> Option<&str> {
        self.embedding_method.as_deref()
    }

    /// Get the surprise score.
    pub fn surprise_score(&self) -> f32 {
        self.surprise_score
    }

    /// Get the failure mode.
    pub fn failure_mode(&self) -> Option<&FailureMode> {
        self.failure_mode.as_ref()
    }

    /// Get the chunk embeddings.
    pub fn chunk_embeddings(&self) -> Option<&[Vec<f32>]> {
        self.chunk_embeddings.as_deref()
    }

    /// Get the satisfaction score.
    pub fn satisfaction_score(&self) -> f32 {
        self.satisfaction_score
    }

    /// Get the number of satisfaction trials.
    pub fn satisfaction_trials(&self) -> u32 {
        self.satisfaction_trials
    }

    /// Get the content hash.
    pub fn content_hash(&self) -> Option<&str> {
        self.content_hash.as_deref()
    }

    /// Get the title (pyramid summary - 10 words max).
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Get the summary (pyramid summary - 50 words max).
    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    /// Get the parent pattern ID (KOS P0 lineage).
    pub fn parent_id(&self) -> Option<&PatternId> {
        self.parent_id.as_ref()
    }

    /// Get the derivation type (KOS P0 lineage).
    pub fn derivation_type(&self) -> Option<&crate::lineage::DerivationType> {
        self.derivation_type.as_ref()
    }

    /// Get the lineage depth (KOS P0 lineage).
    pub fn lineage_depth(&self) -> u32 {
        self.lineage_depth
    }

    /// Get the Bayesian quality score (Beta distribution).
    pub fn bayesian_score(&self) -> &BetaParams {
        &self.bayesian_score
    }

    /// Get a mutable reference to the Bayesian quality score.
    pub fn bayesian_score_mut(&mut self) -> &mut BetaParams {
        &mut self.bayesian_score
    }

    /// Get the tags.
    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    /// Get related pattern IDs.
    pub fn related_patterns(&self) -> &[PatternId] {
        &self.related_patterns
    }

    /// Get the metadata.
    pub fn metadata(&self) -> &PatternMetadata {
        &self.metadata
    }

    // Setters

    /// Set the pattern ID.
    pub fn set_id(&mut self, id: PatternId) {
        self.id = id;
    }

    /// Set the embedding vector.
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = Some(embedding);
    }

    /// Set the embedding method (e.g., "onnx" or "hash").
    pub fn set_embedding_method(&mut self, method: impl Into<String>) {
        self.embedding_method = Some(method.into());
        self.updated_at = Utc::now();
    }

    /// Set the surprise score (clamped to 0.0-1.0).
    pub fn set_surprise_score(&mut self, score: f32) {
        self.surprise_score = score.clamp(0.0, 1.0);
        self.updated_at = Utc::now();
    }

    /// Set the failure mode.
    pub fn set_failure_mode(&mut self, mode: FailureMode) {
        self.failure_mode = Some(mode);
        self.updated_at = Utc::now();
    }

    /// Set chunk embeddings.
    pub fn set_chunk_embeddings(&mut self, chunks: Vec<Vec<f32>>) {
        self.chunk_embeddings = Some(chunks);
    }

    /// Set the satisfaction score (clamped to 0.0-1.0).
    pub fn set_satisfaction_score(&mut self, score: f32) {
        self.satisfaction_score = score.clamp(0.0, 1.0);
        self.updated_at = Utc::now();
    }

    /// Increment the satisfaction trials count.
    pub fn increment_satisfaction_trials(&mut self) {
        self.satisfaction_trials += 1;
        self.updated_at = Utc::now();
    }

    /// Record a satisfaction feedback using Bayesian update.
    /// `satisfied`: true for positive feedback, false for negative.
    pub fn record_satisfaction(&mut self, satisfied: bool) {
        // Update trials count
        self.satisfaction_trials += 1;

        // Bayesian update: score = (old_score * old_trials + feedback) / new_trials
        // This is a simplified version; for production could use proper Beta distribution
        let feedback_value = if satisfied { 1.0 } else { 0.0 };
        let old_trials = self.satisfaction_trials - 1;

        if old_trials == 0 {
            // First feedback, just set it
            self.satisfaction_score = feedback_value;
        } else {
            // Weighted average with prior
            let total_positive = self.satisfaction_score * old_trials as f32 + feedback_value;
            self.satisfaction_score = total_positive / self.satisfaction_trials as f32;
        }

        self.updated_at = Utc::now();
    }

    /// Set the content hash.
    pub fn set_content_hash(&mut self, hash: String) {
        self.content_hash = Some(hash);
    }

    /// Set the title (pyramid summary - 10 words max).
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.title = Some(title.into());
        self.updated_at = Utc::now();
    }

    /// Set the summary (pyramid summary - 50 words max).
    pub fn set_summary(&mut self, summary: impl Into<String>) {
        self.summary = Some(summary.into());
        self.updated_at = Utc::now();
    }

    /// Generate a title from the problem (first 10 words or truncated).
    /// This is a fallback when no explicit title is set.
    pub fn generate_title(&self) -> String {
        self.problem
            .split_whitespace()
            .take(10)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Check if pattern has pyramid summaries (both title and summary set).
    pub fn has_pyramid(&self) -> bool {
        self.title.is_some() && self.summary.is_some()
    }

    /// Get the display title: returns explicit title or generates one from problem.
    pub fn display_title(&self) -> String {
        self.title.clone().unwrap_or_else(|| self.generate_title())
    }

    /// Compute and set the content hash from problem and solution.
    pub fn compute_content_hash(&mut self) {
        let content = format!("{}\n{}", self.problem, self.solution);
        let hash = blake3::hash(content.as_bytes());
        self.content_hash = Some(hash.to_hex().to_string());
    }

    /// Update the effectiveness score.
    pub fn set_effectiveness(&mut self, effectiveness: f32) {
        self.effectiveness = effectiveness.clamp(0.0, 1.0);
        self.updated_at = Utc::now();
    }

    /// Increment the reuse count.
    pub fn increment_reuse_count(&mut self) {
        self.reuse_count += 1;
        self.updated_at = Utc::now();
    }

    /// Update the reward.
    pub fn set_reward(&mut self, reward: f32) {
        self.reward = reward.clamp(0.0, 1.0);
        self.updated_at = Utc::now();
    }

    /// Set the success flag.
    pub fn set_success(&mut self, success: bool) {
        self.success = success;
        self.updated_at = Utc::now();
    }

    /// Set the critique.
    pub fn set_critique(&mut self, critique: impl Into<String>) {
        self.critique = critique.into();
        self.updated_at = Utc::now();
    }

    /// Set the confidence level.
    pub fn set_confidence(&mut self, confidence: f32) {
        self.confidence = confidence.clamp(0.0, 1.0);
        self.updated_at = Utc::now();
    }

    /// Add a tag.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.push(tag.into());
        self.updated_at = Utc::now();
    }

    /// Replace all tags.
    pub fn set_tags(&mut self, tags: Vec<String>) {
        self.tags = tags;
        self.updated_at = Utc::now();
    }

    /// Add a related pattern.
    pub fn add_related_pattern(&mut self, id: PatternId) {
        if !self.related_patterns.contains(&id) {
            self.related_patterns.push(id);
            self.updated_at = Utc::now();
        }
    }

    /// Set the parent pattern ID (KOS P0 lineage).
    pub fn set_parent_id(&mut self, parent_id: PatternId) {
        self.parent_id = Some(parent_id);
        self.updated_at = Utc::now();
    }

    /// Set the derivation type (KOS P0 lineage).
    pub fn set_derivation_type(&mut self, derivation_type: crate::lineage::DerivationType) {
        self.derivation_type = Some(derivation_type);
        self.updated_at = Utc::now();
    }

    /// Set the lineage depth (KOS P0 lineage).
    pub fn set_lineage_depth(&mut self, depth: u32) {
        self.lineage_depth = depth;
        self.updated_at = Utc::now();
    }

    /// Touch the updated_at timestamp.
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }

    /// Set the updated_at timestamp to a specific value (used by sync).
    pub fn set_updated_at(&mut self, ts: DateTime<Utc>) {
        self.updated_at = ts;
    }

    /// Check if the pattern has an embedding.
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    /// Compute a combined quality score based on multiple factors.
    ///
    /// Formula: (effectiveness * 0.4) + (confidence * 0.3) + (reward * 0.2) + (success_bonus * 0.1)
    pub fn quality_score(&self) -> f32 {
        let success_bonus = if self.success { 1.0 } else { 0.0 };
        (self.effectiveness * 0.4)
            + (self.confidence * 0.3)
            + (self.reward * 0.2)
            + (success_bonus * 0.1)
    }

    /// Get the age of the pattern in seconds.
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.timestamp).num_seconds()
    }

    /// Compute a relevance score incorporating temporal decay and surprise.
    ///
    /// Inspired by Titans (Google, 2024): memory should have different persistence
    /// levels — old, unvalidated knowledge decays; recently confirmed gets boosted.
    ///
    /// Formula: quality_score * decay(age) * surprise_boost
    /// - Decay: exponential with 90-day half-life on updated_at
    /// - Surprise boost: up to 20% bonus for highly novel patterns
    pub fn relevance_score(&self) -> f32 {
        let quality = self.quality_score();

        // Exponential decay: half-life of 90 days based on last update
        let age_days = (Utc::now() - self.updated_at).num_days().max(0) as f32;
        let decay = (-0.00770 * age_days).exp(); // ln(0.5) / 90 ≈ -0.00770

        // Surprise boost: novel patterns get up to 20% relevance boost
        let surprise_boost = 1.0 + (self.surprise_score * 0.2);

        quality * decay * surprise_boost
    }

    /// Get the combined problem + solution text for embedding.
    pub fn text_for_embedding(&self) -> String {
        format!(
            "{}\n\n{}\n\nContext: {}",
            self.problem, self.solution, self.context
        )
    }
}

/// Builder for creating Pattern instances.
#[derive(Debug, Default)]
pub struct PatternBuilder {
    id: Option<PatternId>,
    timestamp: Option<DateTime<Utc>>,
    updated_at: Option<DateTime<Utc>>,
    category: Option<PatternCategory>,
    problem: Option<String>,
    solution: Option<String>,
    context: Option<String>,
    effectiveness: Option<f32>,
    reuse_count: Option<u32>,
    reward: Option<f32>,
    success: Option<bool>,
    critique: Option<String>,
    agent_id: Option<String>,
    session_id: Option<String>,
    confidence: Option<f32>,
    embedding: Option<Vec<f32>>,
    embedding_method: Option<String>,
    surprise_score: Option<f32>,
    failure_mode: Option<FailureMode>,
    chunk_embeddings: Option<Vec<Vec<f32>>>,
    satisfaction_score: Option<f32>,
    satisfaction_trials: Option<u32>,
    content_hash: Option<String>,
    title: Option<String>,
    summary: Option<String>,
    parent_id: Option<PatternId>,
    derivation_type: Option<crate::lineage::DerivationType>,
    lineage_depth: Option<u32>,
    bayesian_score: Option<BetaParams>,
    tags: Option<Vec<String>>,
    related_patterns: Option<Vec<PatternId>>,
    metadata: Option<PatternMetadata>,
}

impl PatternBuilder {
    /// Create a new pattern builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the pattern ID.
    pub fn id(mut self, id: impl Into<PatternId>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the timestamp.
    pub fn timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Set the updated_at timestamp (used by sync to preserve remote timestamps).
    pub fn updated_at(mut self, updated_at: DateTime<Utc>) -> Self {
        self.updated_at = Some(updated_at);
        self
    }

    /// Set the category.
    pub fn category(mut self, category: PatternCategory) -> Self {
        self.category = Some(category);
        self
    }

    /// Set the problem description.
    pub fn problem(mut self, problem: impl Into<String>) -> Self {
        self.problem = Some(problem.into());
        self
    }

    /// Set the solution.
    pub fn solution(mut self, solution: impl Into<String>) -> Self {
        self.solution = Some(solution.into());
        self
    }

    /// Set the context.
    pub fn context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Set the effectiveness score.
    pub fn effectiveness(mut self, effectiveness: f32) -> Self {
        self.effectiveness = Some(effectiveness.clamp(0.0, 1.0));
        self
    }

    /// Set the reuse count.
    pub fn reuse_count(mut self, count: u32) -> Self {
        self.reuse_count = Some(count);
        self
    }

    /// Set the reward.
    pub fn reward(mut self, reward: f32) -> Self {
        self.reward = Some(reward.clamp(0.0, 1.0));
        self
    }

    /// Set the success flag.
    pub fn success(mut self, success: bool) -> Self {
        self.success = Some(success);
        self
    }

    /// Set the critique.
    pub fn critique(mut self, critique: impl Into<String>) -> Self {
        self.critique = Some(critique.into());
        self
    }

    /// Set the agent ID.
    pub fn agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Set the session ID.
    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the confidence level.
    pub fn confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Set the embedding vector.
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set the embedding method (e.g., "onnx" or "hash").
    pub fn embedding_method(mut self, method: impl Into<String>) -> Self {
        self.embedding_method = Some(method.into());
        self
    }

    /// Set the surprise score.
    pub fn surprise_score(mut self, score: f32) -> Self {
        self.surprise_score = Some(score.clamp(0.0, 1.0));
        self
    }

    /// Set the failure mode.
    pub fn failure_mode(mut self, mode: FailureMode) -> Self {
        self.failure_mode = Some(mode);
        self
    }

    /// Set chunk embeddings.
    pub fn chunk_embeddings(mut self, chunks: Vec<Vec<f32>>) -> Self {
        self.chunk_embeddings = Some(chunks);
        self
    }

    /// Set the satisfaction score.
    pub fn satisfaction_score(mut self, score: f32) -> Self {
        self.satisfaction_score = Some(score.clamp(0.0, 1.0));
        self
    }

    /// Set the satisfaction trials count.
    pub fn satisfaction_trials(mut self, trials: u32) -> Self {
        self.satisfaction_trials = Some(trials);
        self
    }

    /// Set the content hash.
    pub fn content_hash(mut self, hash: impl Into<String>) -> Self {
        self.content_hash = Some(hash.into());
        self
    }

    /// Set the title (pyramid summary - 10 words max).
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the summary (pyramid summary - 50 words max).
    pub fn summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Set the parent pattern ID (KOS P0 lineage).
    pub fn parent_id(mut self, parent_id: impl Into<PatternId>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }

    /// Set the derivation type (KOS P0 lineage).
    pub fn derivation_type(mut self, derivation_type: crate::lineage::DerivationType) -> Self {
        self.derivation_type = Some(derivation_type);
        self
    }

    /// Set the lineage depth (KOS P0 lineage).
    pub fn lineage_depth(mut self, depth: u32) -> Self {
        self.lineage_depth = Some(depth);
        self
    }

    /// Set the Bayesian quality score (Beta distribution).
    pub fn bayesian_score(mut self, score: BetaParams) -> Self {
        self.bayesian_score = Some(score);
        self
    }

    /// Set the tags.
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Add a single tag.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.get_or_insert_with(Vec::new).push(tag.into());
        self
    }

    /// Set related patterns.
    pub fn related_patterns(mut self, patterns: Vec<PatternId>) -> Self {
        self.related_patterns = Some(patterns);
        self
    }

    /// Set the metadata.
    pub fn metadata(mut self, metadata: PatternMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Build the pattern.
    ///
    /// Uses default values for any unspecified fields.
    pub fn build(self) -> Pattern {
        let now = Utc::now();
        let problem = self.problem.unwrap_or_default();
        let solution = self.solution.unwrap_or_default();

        // Compute content hash if not provided
        let content_hash = self.content_hash.or_else(|| {
            if !problem.is_empty() && !solution.is_empty() {
                let content = format!("{}\n{}", problem, solution);
                Some(blake3::hash(content.as_bytes()).to_hex().to_string())
            } else {
                None
            }
        });

        Pattern {
            id: self.id.unwrap_or_else(PatternId::new),
            timestamp: self.timestamp.unwrap_or(now),
            updated_at: self.updated_at.unwrap_or(now),
            category: self.category.unwrap_or_default(),
            problem,
            solution,
            context: self.context.unwrap_or_default(),
            effectiveness: self.effectiveness.unwrap_or(0.5),
            reuse_count: self.reuse_count.unwrap_or(0),
            reward: self.reward.unwrap_or(0.5),
            success: self.success.unwrap_or(true),
            critique: self.critique.unwrap_or_default(),
            agent_id: self.agent_id,
            session_id: self.session_id,
            confidence: self.confidence.unwrap_or(0.5),
            embedding: self.embedding,
            embedding_method: self.embedding_method,
            surprise_score: self.surprise_score.unwrap_or(0.0),
            failure_mode: self.failure_mode,
            chunk_embeddings: self.chunk_embeddings,
            satisfaction_score: self.satisfaction_score.unwrap_or(0.5),
            satisfaction_trials: self.satisfaction_trials.unwrap_or(0),
            content_hash,
            title: self.title,
            summary: self.summary,
            parent_id: self.parent_id,
            derivation_type: self.derivation_type,
            lineage_depth: self.lineage_depth.unwrap_or(0),
            bayesian_score: self.bayesian_score.unwrap_or_default(),
            tags: self.tags.unwrap_or_default(),
            related_patterns: self.related_patterns.unwrap_or_default(),
            metadata: self.metadata.unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_id_new() {
        let id1 = PatternId::new();
        let id2 = PatternId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_pattern_id_from_string() {
        let id = PatternId::from_string("test-123");
        assert_eq!(id.as_str(), "test-123");
    }

    #[test]
    fn test_pattern_category_from_str() {
        assert_eq!(PatternCategory::from("architecture"), PatternCategory::Architecture);
        assert_eq!(PatternCategory::from("SECURITY"), PatternCategory::Security);
        assert_eq!(PatternCategory::from("custom_domain"), PatternCategory::Custom("custom_domain".to_string()));
    }

    #[test]
    fn test_pattern_builder_minimal() {
        let pattern = Pattern::builder()
            .problem("Test problem")
            .solution("Test solution")
            .build();

        assert_eq!(pattern.problem(), "Test problem");
        assert_eq!(pattern.solution(), "Test solution");
        assert!(!pattern.id().as_str().is_empty());
    }

    #[test]
    fn test_pattern_builder_full() {
        let pattern = Pattern::builder()
            .id("custom-id")
            .problem("How to cache data")
            .solution("Use Redis with TTL")
            .context("Web application with high traffic")
            .category(PatternCategory::Performance)
            .effectiveness(0.95)
            .reuse_count(10)
            .reward(0.9)
            .success(true)
            .critique("Works well in production")
            .agent_id("agent-1")
            .session_id("session-1")
            .confidence(0.85)
            .embedding(vec![0.1, 0.2, 0.3])
            .tag("caching")
            .tag("redis")
            .metadata(PatternMetadata::new().with_language("rust"))
            .build();

        assert_eq!(pattern.id().as_str(), "custom-id");
        assert_eq!(pattern.problem(), "How to cache data");
        assert_eq!(pattern.category(), &PatternCategory::Performance);
        assert!((pattern.effectiveness() - 0.95).abs() < 0.001);
        assert_eq!(pattern.reuse_count(), 10);
        assert_eq!(pattern.tags().len(), 2);
        assert_eq!(pattern.embedding(), Some(&[0.1, 0.2, 0.3][..]));
    }

    #[test]
    fn test_pattern_new() {
        let pattern = Pattern::new("Problem", "Solution");
        assert_eq!(pattern.problem(), "Problem");
        assert_eq!(pattern.solution(), "Solution");
    }

    #[test]
    fn test_pattern_setters() {
        let mut pattern = Pattern::new("Test", "Solution");

        pattern.set_effectiveness(0.8);
        assert!((pattern.effectiveness() - 0.8).abs() < 0.001);

        pattern.increment_reuse_count();
        assert_eq!(pattern.reuse_count(), 1);

        pattern.add_tag("new-tag");
        assert!(pattern.tags().contains(&"new-tag".to_string()));
    }

    #[test]
    fn test_pattern_quality_score() {
        let pattern = Pattern::builder()
            .effectiveness(1.0)
            .confidence(1.0)
            .reward(1.0)
            .success(true)
            .build();

        let score = pattern.quality_score();
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pattern_clamp_values() {
        let pattern = Pattern::builder()
            .effectiveness(1.5) // Should clamp to 1.0
            .confidence(-0.5)   // Should clamp to 0.0
            .reward(2.0)        // Should clamp to 1.0
            .build();

        assert!((pattern.effectiveness() - 1.0).abs() < 0.001);
        assert!((pattern.confidence() - 0.0).abs() < 0.001);
        assert!((pattern.reward() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pattern_text_for_embedding() {
        let pattern = Pattern::builder()
            .problem("How to cache")
            .solution("Use Redis")
            .context("High traffic app")
            .build();

        let text = pattern.text_for_embedding();
        assert!(text.contains("How to cache"));
        assert!(text.contains("Use Redis"));
        assert!(text.contains("High traffic app"));
    }

    #[test]
    fn test_pattern_serialization() {
        let pattern = Pattern::builder()
            .problem("Test")
            .solution("Solution")
            .category(PatternCategory::Security)
            .tag("important")
            .build();

        let json = serde_json::to_string(&pattern).unwrap();
        let deserialized: Pattern = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.problem(), pattern.problem());
        assert_eq!(deserialized.category(), pattern.category());
    }

    #[test]
    fn test_pattern_metadata() {
        let metadata = PatternMetadata::new()
            .with_language("rust")
            .with_framework("tokio")
            .with_source("code_review")
            .with_extra("version", serde_json::json!("1.0.0"));

        assert_eq!(metadata.language, Some("rust".to_string()));
        assert_eq!(metadata.framework, Some("tokio".to_string()));
        assert_eq!(metadata.source, Some("code_review".to_string()));
        assert!(metadata.extra.contains_key("version"));
    }

    #[test]
    fn test_pyramid_summary_fields() {
        // Test with explicit title and summary
        let pattern = Pattern::builder()
            .problem("How to implement caching in a web application with high traffic")
            .solution("Use Redis with TTL-based expiration")
            .title("Web App Caching Strategy")
            .summary("Use Redis for caching with TTL expiration to handle high traffic web applications efficiently.")
            .build();

        assert_eq!(pattern.title(), Some("Web App Caching Strategy"));
        assert_eq!(pattern.summary(), Some("Use Redis for caching with TTL expiration to handle high traffic web applications efficiently."));
        assert!(pattern.has_pyramid());
    }

    #[test]
    fn test_generate_title() {
        let pattern = Pattern::builder()
            .problem("How to implement a rate limiter for API endpoints using token bucket algorithm")
            .solution("Use a token bucket implementation")
            .build();

        // Should take first 10 words: "How to implement a rate limiter for API endpoints using"
        let generated = pattern.generate_title();
        assert_eq!(generated, "How to implement a rate limiter for API endpoints using");

        // Pattern without explicit title
        assert!(pattern.title().is_none());
        assert!(!pattern.has_pyramid());

        // display_title should fall back to generated
        assert_eq!(pattern.display_title(), generated);
    }

    #[test]
    fn test_has_pyramid() {
        // No pyramid fields
        let pattern1 = Pattern::new("Problem", "Solution");
        assert!(!pattern1.has_pyramid());

        // Only title
        let pattern2 = Pattern::builder()
            .problem("Problem")
            .solution("Solution")
            .title("Title Only")
            .build();
        assert!(!pattern2.has_pyramid());

        // Only summary
        let pattern3 = Pattern::builder()
            .problem("Problem")
            .solution("Solution")
            .summary("Summary Only")
            .build();
        assert!(!pattern3.has_pyramid());

        // Both title and summary
        let pattern4 = Pattern::builder()
            .problem("Problem")
            .solution("Solution")
            .title("Title")
            .summary("Summary")
            .build();
        assert!(pattern4.has_pyramid());
    }

    #[test]
    fn test_pyramid_setters() {
        let mut pattern = Pattern::new("Problem", "Solution");

        assert!(pattern.title().is_none());
        assert!(pattern.summary().is_none());

        pattern.set_title("New Title");
        assert_eq!(pattern.title(), Some("New Title"));

        pattern.set_summary("New Summary");
        assert_eq!(pattern.summary(), Some("New Summary"));

        assert!(pattern.has_pyramid());
    }

    #[test]
    fn test_pyramid_serialization() {
        let pattern = Pattern::builder()
            .problem("Test problem")
            .solution("Test solution")
            .title("Test Title")
            .summary("Test Summary")
            .build();

        let json = serde_json::to_string(&pattern).unwrap();
        let deserialized: Pattern = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.title(), pattern.title());
        assert_eq!(deserialized.summary(), pattern.summary());
        assert!(deserialized.has_pyramid());
    }
}
