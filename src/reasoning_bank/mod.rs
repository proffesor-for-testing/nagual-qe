//! ReasoningBank - Pattern storage and retrieval for self-learning agents.
//!
//! The ReasoningBank provides a sophisticated pattern storage and retrieval system
//! that enables agents to learn from past experiences and apply relevant knowledge
//! to new situations.
//!
//! # Features
//!
//! - **Pattern Storage**: Store problems, solutions, and their contexts
//! - **Similarity Search**: Find similar patterns using vector embeddings
//! - **MMR Reranking**: Maximal Marginal Relevance for diverse results
//! - **Domain Filtering**: Filter patterns by domain/category hierarchy
//! - **Reward Thresholds**: Filter by pattern quality/success metrics
//! - **Prompt Formatting**: Convert patterns to LLM-ready format
//! - **Statistics**: Comprehensive pattern analytics
//!
//! # Example
//!
//! ```ignore
//! use nagual::reasoning_bank::{ReasoningBank, PatternQuery, RetrievalConfig};
//!
//! let bank = ReasoningBank::new(db, embedder)?;
//!
//! // Retrieve similar patterns
//! let query = PatternQuery::new("How to handle database timeouts?")
//!     .with_domains(vec!["database", "resilience"])
//!     .with_min_reward(0.7)
//!     .with_limit(5);
//!
//! let result = bank.retrieve_patterns(&query).await?;
//!
//! // Format for LLM prompt
//! let prompt_context = bank.format_for_prompt(&result.patterns, 2000)?;
//! ```

pub mod dedup;
pub mod dna;
pub mod export;
mod formatter;
pub mod mmr;
pub mod pattern;
pub mod pyramid;
mod retrieval;
pub mod scoring;
pub mod search;
pub mod staging;
mod stats;
pub mod storage;
pub mod transfusion;

pub use formatter::{
    format_for_prompt, FormatConfig, FormattedPattern, PromptFormatter, TruncationStrategy,
};
pub use retrieval::{
    retrieve_patterns, retrieve_patterns_hybrid, retrieve_patterns_hyperbolic, FactorScores,
    HybridSearchConfig, HyperbolicRetrievalConfig, MmrConfig, PatternQuery, RetrievalConfig,
    RetrievalResult, ScoredPattern, ScoringWeights,
};

/// Auto-promotion criteria for tier graduation.
///
/// Patterns seen `min_occurrences` times across `min_distinct_contexts`
/// distinct sessions within `window_days` are promoted one tier.
#[derive(Debug, Clone)]
pub struct AutoPromotionCriteria {
    /// Minimum number of times the pattern must be used.
    pub min_occurrences: u32,
    /// Minimum number of distinct sessions/tasks.
    pub min_distinct_contexts: u32,
    /// Time window in days.
    pub window_days: u32,
}

impl Default for AutoPromotionCriteria {
    fn default() -> Self {
        Self {
            min_occurrences: 3,
            min_distinct_contexts: 2,
            window_days: 30,
        }
    }
}
pub use staging::{staged_retrieve_patterns, RetrievalStaging, StagingStats};
pub use stats::{
    get_pattern_stats, DomainStats, PatternStats, ReuseDistribution, StatsConfig, TopPattern,
};

use chrono::{DateTime, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

/// Errors specific to ReasoningBank operations.
#[derive(Error, Debug)]
pub enum ReasoningBankError {
    /// Database error
    #[error("Database error: {0}")]
    Database(String),

    /// Embedding error
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Pattern not found
    #[error("Pattern not found: {id}")]
    NotFound { id: String },

    /// Invalid query
    #[error("Invalid query: {reason}")]
    InvalidQuery { reason: String },

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Invalid domain hierarchy
    #[error("Invalid domain hierarchy: {domain}")]
    InvalidDomain { domain: String },

    /// Token limit exceeded
    #[error("Token limit exceeded: {actual} > {limit}")]
    TokenLimitExceeded { actual: usize, limit: usize },

    /// No patterns available
    #[error("No patterns available for query")]
    NoPatterns,
}

/// Result type for ReasoningBank operations.
pub type ReasoningBankResult<T> = std::result::Result<T, ReasoningBankError>;

/// Pattern confidence tier representing maturity level.
///
/// Patterns progress through tiers as they prove their value:
/// - Booster: Default tier for new patterns
/// - Crystal: High-value patterns (reward >= 0.7, reuse >= 5)
/// - Reflex: Elite patterns (reward >= 0.9, reuse >= 20)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum PatternTier {
    /// Default tier for new or unproven patterns
    Booster = 0,
    /// High-value patterns with proven track record
    Crystal = 1,
    /// Elite patterns for instant retrieval
    Reflex = 2,
}

impl PatternTier {
    /// Check if pattern qualifies for promotion.
    pub fn check_promotion(reward: f32, reuse_count: u32) -> Option<PatternTier> {
        if reward >= 0.9 && reuse_count >= 20 {
            Some(PatternTier::Reflex)
        } else if reward >= 0.7 && reuse_count >= 5 {
            Some(PatternTier::Crystal)
        } else {
            None
        }
    }

    /// Check if pattern should be demoted (hysteresis: threshold - 0.1).
    pub fn check_demotion(current_tier: PatternTier, reward: f32) -> Option<PatternTier> {
        match current_tier {
            PatternTier::Reflex if reward < 0.8 => Some(PatternTier::Crystal),
            PatternTier::Crystal if reward < 0.6 => Some(PatternTier::Booster),
            _ => None,
        }
    }

    /// Get the tier name as a string.
    pub fn as_str(&self) -> &'static str {
        match self {
            PatternTier::Booster => "booster",
            PatternTier::Crystal => "crystal",
            PatternTier::Reflex => "reflex",
        }
    }
}

impl Default for PatternTier {
    fn default() -> Self {
        PatternTier::Booster
    }
}

impl std::fmt::Display for PatternTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for PatternTier {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "booster" => Ok(PatternTier::Booster),
            "crystal" => Ok(PatternTier::Crystal),
            "reflex" => Ok(PatternTier::Reflex),
            _ => Err(format!("Unknown tier: {}", s)),
        }
    }
}

/// A stored pattern representing a problem-solution pair with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Unique identifier
    pub id: String,

    /// The problem description
    pub problem: String,

    /// The solution description
    pub solution: String,

    /// Domain/category (e.g., "rust.async", "database.postgres")
    pub domain: String,

    /// Additional context for the pattern
    pub context: Option<String>,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,

    /// Reward/quality score (0.0 - 1.0)
    pub reward: f32,

    /// Number of times this pattern was successfully applied
    pub success_count: u32,

    /// Total number of times this pattern was used
    pub usage_count: u32,

    /// Success rate (success_count / usage_count)
    pub success_rate: f32,

    /// When the pattern was created
    pub created_at: DateTime<Utc>,

    /// When the pattern was last updated
    pub updated_at: DateTime<Utc>,

    /// Optional critique or notes about the pattern
    pub critique: Option<String>,

    /// Session ID that created this pattern
    pub session_id: Option<String>,

    /// Tags for additional categorization
    pub tags: Vec<String>,

    /// Pattern confidence tier (booster, crystal, reflex)
    pub tier: PatternTier,

    /// The embedding vector (stored separately, loaded on demand)
    #[serde(skip)]
    pub embedding: Option<Array1<f32>>,
}

impl Pattern {
    /// Create a new pattern with the given problem and solution.
    pub fn new(
        problem: impl Into<String>,
        solution: impl Into<String>,
        domain: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            problem: problem.into(),
            solution: solution.into(),
            domain: domain.into(),
            context: None,
            confidence: 0.5,
            reward: 0.5,
            success_count: 0,
            usage_count: 0,
            success_rate: 0.0,
            created_at: now,
            updated_at: now,
            critique: None,
            session_id: None,
            tags: Vec::new(),
            tier: PatternTier::Booster,
            embedding: None,
        }
    }

    /// Set the context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Set the confidence score.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set the reward score.
    pub fn with_reward(mut self, reward: f32) -> Self {
        self.reward = reward.clamp(0.0, 1.0);
        self
    }

    /// Set the session ID.
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Add tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set the critique.
    pub fn with_critique(mut self, critique: impl Into<String>) -> Self {
        self.critique = Some(critique.into());
        self
    }

    /// Set the embedding vector.
    pub fn with_embedding(mut self, embedding: Array1<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set the tier.
    pub fn with_tier(mut self, tier: PatternTier) -> Self {
        self.tier = tier;
        self
    }

    /// Check and apply tier promotion/demotion based on current reward and reuse.
    /// Returns Some((old_tier, new_tier)) if tier changed.
    pub fn evaluate_tier(&mut self) -> Option<(PatternTier, PatternTier)> {
        let old_tier = self.tier;

        // Check promotion first
        if let Some(new_tier) = PatternTier::check_promotion(self.reward, self.usage_count) {
            if (new_tier as u8) > (old_tier as u8) {
                self.tier = new_tier;
                return Some((old_tier, new_tier));
            }
        }

        // Check demotion
        if let Some(new_tier) = PatternTier::check_demotion(old_tier, self.reward) {
            self.tier = new_tier;
            return Some((old_tier, new_tier));
        }

        None
    }

    /// Update usage statistics after the pattern is used.
    pub fn record_usage(&mut self, success: bool) {
        self.usage_count += 1;
        if success {
            self.success_count += 1;
        }
        self.success_rate = if self.usage_count > 0 {
            self.success_count as f32 / self.usage_count as f32
        } else {
            0.0
        };
        self.updated_at = Utc::now();
    }

    /// Get the combined text for embedding generation.
    pub fn embedding_text(&self) -> String {
        let mut text = format!("{}\n{}", self.problem, self.solution);
        if let Some(ref ctx) = self.context {
            text.push('\n');
            text.push_str(ctx);
        }
        text
    }

    /// Check if this pattern matches a domain (with hierarchy support).
    ///
    /// Domain hierarchy uses dots: "rust.async" matches "rust", "rust.async", etc.
    pub fn matches_domain(&self, domain: &str) -> bool {
        if domain.is_empty() {
            return true;
        }
        // Exact match
        if self.domain == domain {
            return true;
        }
        // Hierarchy match: "rust.async" starts with "rust."
        if self.domain.starts_with(&format!("{}.", domain)) {
            return true;
        }
        // Reverse hierarchy: "rust" is a prefix of "rust.async"
        if domain.starts_with(&format!("{}.", self.domain)) {
            return true;
        }
        false
    }

    /// Get the reliability score combining confidence and success rate.
    pub fn reliability_score(&self) -> f32 {
        // Weight confidence by usage count (more usage = more reliable)
        let usage_weight = (self.usage_count as f32).min(10.0) / 10.0;
        let base_reliability = self.confidence * 0.4 + self.success_rate * 0.6;

        // Blend with base confidence based on usage
        base_reliability * usage_weight + self.confidence * (1.0 - usage_weight)
    }

    /// Calculate recency score (1.0 for patterns created now, decaying over time).
    pub fn recency_score(&self) -> f32 {
        let age_hours = (Utc::now() - self.created_at).num_hours() as f32;
        // Decay over 30 days (720 hours)
        let decay_rate = 720.0;
        (-age_hours / decay_rate).exp()
    }
}

impl Default for Pattern {
    fn default() -> Self {
        Self::new("", "", "general")
    }
}

/// Convert from the rich CLI Pattern type to the retrieval Pattern type.
impl From<&pattern::Pattern> for Pattern {
    fn from(p: &pattern::Pattern) -> Self {
        let mut pat = Self::new(p.problem(), p.solution(), p.category().to_string());
        pat.id = p.id().to_string();
        pat.context = if p.context().is_empty() {
            None
        } else {
            Some(p.context().to_string())
        };
        pat.confidence = p.confidence();
        pat.reward = p.reward();
        pat.success_count = if p.success() { 1 } else { 0 };
        pat.usage_count = p.reuse_count().max(1);
        pat.success_rate = if p.success() { 1.0 } else { 0.0 };
        pat.created_at = p.timestamp();
        pat.updated_at = p.timestamp();
        pat.tags = p.tags().to_vec();
        pat.tier = PatternTier::Booster;
        pat.embedding = p.embedding().map(|e| Array1::from_vec(e.to_vec()));
        pat
    }
}

/// Domain hierarchy utilities.
pub mod domain {
    /// Parse a domain into its hierarchy components.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let parts = parse_hierarchy("rust.async.tokio");
    /// assert_eq!(parts, vec!["rust", "rust.async", "rust.async.tokio"]);
    /// ```
    pub fn parse_hierarchy(domain: &str) -> Vec<String> {
        let parts: Vec<&str> = domain.split('.').collect();
        let mut hierarchy = Vec::with_capacity(parts.len());

        for i in 0..parts.len() {
            hierarchy.push(parts[..=i].join("."));
        }

        hierarchy
    }

    /// Get the parent domain.
    ///
    /// # Example
    ///
    /// ```ignore
    /// assert_eq!(parent("rust.async.tokio"), Some("rust.async".to_string()));
    /// assert_eq!(parent("rust"), None);
    /// ```
    pub fn parent(domain: &str) -> Option<String> {
        let parts: Vec<&str> = domain.split('.').collect();
        if parts.len() <= 1 {
            None
        } else {
            Some(parts[..parts.len() - 1].join("."))
        }
    }

    /// Get the root domain.
    ///
    /// # Example
    ///
    /// ```ignore
    /// assert_eq!(root("rust.async.tokio"), "rust");
    /// ```
    pub fn root(domain: &str) -> &str {
        domain.split('.').next().unwrap_or(domain)
    }

    /// Check if domain_a is an ancestor of domain_b.
    pub fn is_ancestor(ancestor: &str, descendant: &str) -> bool {
        if ancestor == descendant {
            return true;
        }
        descendant.starts_with(&format!("{}.", ancestor))
    }

    /// Get the depth of a domain in the hierarchy.
    pub fn depth(domain: &str) -> usize {
        domain.split('.').count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_new() {
        let pattern = Pattern::new(
            "How to handle errors?",
            "Use Result type with proper error handling",
            "rust.error_handling",
        );

        assert!(!pattern.id.is_empty());
        assert_eq!(pattern.problem, "How to handle errors?");
        assert_eq!(pattern.solution, "Use Result type with proper error handling");
        assert_eq!(pattern.domain, "rust.error_handling");
        assert_eq!(pattern.confidence, 0.5);
        assert_eq!(pattern.reward, 0.5);
    }

    #[test]
    fn test_pattern_builder() {
        let pattern = Pattern::new("problem", "solution", "test")
            .with_context("Additional context")
            .with_confidence(0.9)
            .with_reward(0.85)
            .with_session_id("session-123")
            .with_tags(vec!["tag1".to_string(), "tag2".to_string()])
            .with_critique("Works well for simple cases");

        assert_eq!(pattern.context, Some("Additional context".to_string()));
        assert_eq!(pattern.confidence, 0.9);
        assert_eq!(pattern.reward, 0.85);
        assert_eq!(pattern.session_id, Some("session-123".to_string()));
        assert_eq!(pattern.tags, vec!["tag1", "tag2"]);
        assert_eq!(
            pattern.critique,
            Some("Works well for simple cases".to_string())
        );
    }

    #[test]
    fn test_pattern_record_usage() {
        let mut pattern = Pattern::new("p", "s", "d");

        pattern.record_usage(true);
        assert_eq!(pattern.usage_count, 1);
        assert_eq!(pattern.success_count, 1);
        assert_eq!(pattern.success_rate, 1.0);

        pattern.record_usage(false);
        assert_eq!(pattern.usage_count, 2);
        assert_eq!(pattern.success_count, 1);
        assert_eq!(pattern.success_rate, 0.5);

        pattern.record_usage(true);
        assert_eq!(pattern.usage_count, 3);
        assert_eq!(pattern.success_count, 2);
        assert!((pattern.success_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_pattern_embedding_text() {
        let pattern = Pattern::new("Problem text", "Solution text", "domain");
        assert_eq!(pattern.embedding_text(), "Problem text\nSolution text");

        let pattern_with_context = pattern.with_context("Context text");
        assert_eq!(
            pattern_with_context.embedding_text(),
            "Problem text\nSolution text\nContext text"
        );
    }

    #[test]
    fn test_pattern_matches_domain() {
        let pattern = Pattern::new("p", "s", "rust.async.tokio");

        // Exact match
        assert!(pattern.matches_domain("rust.async.tokio"));

        // Ancestor match
        assert!(pattern.matches_domain("rust.async"));
        assert!(pattern.matches_domain("rust"));

        // Non-match
        assert!(!pattern.matches_domain("python"));
        assert!(!pattern.matches_domain("rust.sync"));

        // Empty domain matches all
        assert!(pattern.matches_domain(""));
    }

    #[test]
    fn test_pattern_reliability_score() {
        let mut pattern = Pattern::new("p", "s", "d").with_confidence(0.8);

        // With no usage, reliability is mostly based on confidence
        let initial_reliability = pattern.reliability_score();
        assert!(initial_reliability > 0.0);

        // Add successful usages
        for _ in 0..10 {
            pattern.record_usage(true);
        }

        // With full success rate, reliability should be high
        let final_reliability = pattern.reliability_score();
        assert!(final_reliability > initial_reliability);
        assert!(final_reliability > 0.8);
    }

    #[test]
    fn test_domain_parse_hierarchy() {
        let parts = domain::parse_hierarchy("rust.async.tokio");
        assert_eq!(parts, vec!["rust", "rust.async", "rust.async.tokio"]);

        let single = domain::parse_hierarchy("rust");
        assert_eq!(single, vec!["rust"]);
    }

    #[test]
    fn test_domain_parent() {
        assert_eq!(
            domain::parent("rust.async.tokio"),
            Some("rust.async".to_string())
        );
        assert_eq!(domain::parent("rust.async"), Some("rust".to_string()));
        assert_eq!(domain::parent("rust"), None);
    }

    #[test]
    fn test_domain_root() {
        assert_eq!(domain::root("rust.async.tokio"), "rust");
        assert_eq!(domain::root("rust"), "rust");
    }

    #[test]
    fn test_domain_is_ancestor() {
        assert!(domain::is_ancestor("rust", "rust.async.tokio"));
        assert!(domain::is_ancestor("rust.async", "rust.async.tokio"));
        assert!(domain::is_ancestor(
            "rust.async.tokio",
            "rust.async.tokio"
        )); // Self is ancestor
        assert!(!domain::is_ancestor("rust.sync", "rust.async.tokio"));
    }

    #[test]
    fn test_domain_depth() {
        assert_eq!(domain::depth("rust"), 1);
        assert_eq!(domain::depth("rust.async"), 2);
        assert_eq!(domain::depth("rust.async.tokio"), 3);
    }

    #[test]
    fn test_pattern_tier_promotion() {
        assert_eq!(PatternTier::check_promotion(0.5, 3), None);
        assert_eq!(
            PatternTier::check_promotion(0.7, 5),
            Some(PatternTier::Crystal)
        );
        assert_eq!(
            PatternTier::check_promotion(0.9, 20),
            Some(PatternTier::Reflex)
        );
        assert_eq!(
            PatternTier::check_promotion(0.95, 25),
            Some(PatternTier::Reflex)
        );
        assert_eq!(PatternTier::check_promotion(0.7, 4), None); // reuse too low
    }

    #[test]
    fn test_pattern_tier_demotion() {
        assert_eq!(
            PatternTier::check_demotion(PatternTier::Reflex, 0.75),
            Some(PatternTier::Crystal)
        );
        assert_eq!(
            PatternTier::check_demotion(PatternTier::Crystal, 0.55),
            Some(PatternTier::Booster)
        );
        assert_eq!(
            PatternTier::check_demotion(PatternTier::Crystal, 0.65),
            None
        ); // above threshold
        assert_eq!(
            PatternTier::check_demotion(PatternTier::Booster, 0.1),
            None
        ); // can't demote below booster
    }

    #[test]
    fn test_pattern_evaluate_tier() {
        let mut pattern = Pattern::new("p", "s", "d").with_reward(0.75);
        pattern.usage_count = 10;

        let result = pattern.evaluate_tier();
        assert_eq!(result, Some((PatternTier::Booster, PatternTier::Crystal)));
        assert_eq!(pattern.tier, PatternTier::Crystal);
    }

    #[test]
    fn test_pattern_tier_default() {
        let pattern = Pattern::new("p", "s", "d");
        assert_eq!(pattern.tier, PatternTier::Booster);
    }

    #[test]
    fn test_pattern_tier_from_str() {
        assert_eq!("booster".parse::<PatternTier>(), Ok(PatternTier::Booster));
        assert_eq!("crystal".parse::<PatternTier>(), Ok(PatternTier::Crystal));
        assert_eq!("reflex".parse::<PatternTier>(), Ok(PatternTier::Reflex));
        assert!("unknown".parse::<PatternTier>().is_err());
    }

    #[test]
    fn test_pattern_tier_display() {
        assert_eq!(PatternTier::Booster.to_string(), "booster");
        assert_eq!(PatternTier::Crystal.to_string(), "crystal");
        assert_eq!(PatternTier::Reflex.to_string(), "reflex");
    }

    #[test]
    fn test_pattern_tier_as_str() {
        assert_eq!(PatternTier::Booster.as_str(), "booster");
        assert_eq!(PatternTier::Crystal.as_str(), "crystal");
        assert_eq!(PatternTier::Reflex.as_str(), "reflex");
    }

    #[test]
    fn test_pattern_with_tier() {
        let pattern = Pattern::new("p", "s", "d").with_tier(PatternTier::Crystal);
        assert_eq!(pattern.tier, PatternTier::Crystal);
    }

    #[test]
    fn test_pattern_evaluate_tier_no_change() {
        let mut pattern = Pattern::new("p", "s", "d").with_reward(0.3);
        pattern.usage_count = 2;

        let result = pattern.evaluate_tier();
        assert_eq!(result, None);
        assert_eq!(pattern.tier, PatternTier::Booster);
    }

    #[test]
    fn test_pattern_evaluate_tier_demotion() {
        let mut pattern = Pattern::new("p", "s", "d")
            .with_reward(0.5)
            .with_tier(PatternTier::Crystal);
        pattern.usage_count = 10;

        let result = pattern.evaluate_tier();
        assert_eq!(result, Some((PatternTier::Crystal, PatternTier::Booster)));
        assert_eq!(pattern.tier, PatternTier::Booster);
    }

    #[test]
    fn test_pattern_evaluate_tier_reflex_promotion() {
        let mut pattern = Pattern::new("p", "s", "d")
            .with_reward(0.95)
            .with_tier(PatternTier::Crystal);
        pattern.usage_count = 25;

        let result = pattern.evaluate_tier();
        assert_eq!(result, Some((PatternTier::Crystal, PatternTier::Reflex)));
        assert_eq!(pattern.tier, PatternTier::Reflex);
    }
}
