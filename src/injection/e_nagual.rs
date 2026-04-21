//! E_nagual - Attention Bias Injection for Vendor LLMs
//!
//! E_nagual computes an attention bias matrix from ReasoningBank patterns,
//! HNSW neighbors, and trajectory history. This bias is then formatted and
//! injected into LLM prompts to guide model behavior without requiring
//! fine-tuning or direct model weight access.
//!
//! # Overview
//!
//! E_nagual = f(patterns, hnsw_neighbors, trajectory_history)
//!
//! The bias injection works by:
//! 1. Retrieving relevant patterns from ReasoningBank (using HNSW for speed)
//! 2. Extracting trajectory hints from past successful reasoning paths
//! 3. Computing confidence-weighted attention scores
//! 4. Formatting as system prompts, few-shot examples, or structured context
//!
//! # Example
//!
//! ```ignore
//! use nagual::injection::{ENagual, ENagualConfig, InjectionContext};
//!
//! let config = ENagualConfig::default();
//! let e_nagual = ENagual::compute("How to handle database timeouts?", &context, &config).await?;
//!
//! // Get formatted prompt prefix for Claude
//! let prefix = e_nagual.to_prompt_prefix();
//!
//! // Or get few-shot examples for GPT
//! let examples = e_nagual.to_few_shot_examples();
//! ```

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::learning::trajectory::{Trajectory, TrajectoryStep, StepType};
use crate::profdag::node::NodeType;
use crate::profdag::search::ProfDAGSearch;
use crate::profdag::storage::{ProfDAGStorage, SimilarNode};
use crate::reasoning_bank::{Pattern, ScoredPattern};

/// Configuration for E_nagual computation.
#[derive(Debug, Clone)]
pub struct ENagualConfig {
    /// Maximum number of patterns to include in the bias.
    pub max_patterns: usize,

    /// Maximum number of HNSW neighbors to consider.
    pub max_neighbors: usize,

    /// Maximum number of trajectory steps to include.
    pub max_trajectory_steps: usize,

    /// Minimum pattern reward threshold for inclusion.
    pub min_pattern_reward: f32,

    /// Minimum similarity threshold for HNSW neighbors.
    pub min_neighbor_similarity: f64,

    /// Weight for pattern confidence in final scoring.
    pub pattern_confidence_weight: f32,

    /// Weight for neighbor similarity in final scoring.
    pub neighbor_similarity_weight: f32,

    /// Weight for trajectory recency in final scoring.
    pub trajectory_recency_weight: f32,

    /// Whether to include failure patterns as negative examples.
    pub include_negative_examples: bool,

    /// Maximum tokens for the formatted output.
    pub max_output_tokens: usize,

    /// Whether to enable trajectory hints.
    pub enable_trajectory_hints: bool,
}

impl Default for ENagualConfig {
    fn default() -> Self {
        Self {
            max_patterns: 5,
            max_neighbors: 10,
            max_trajectory_steps: 3,
            min_pattern_reward: 0.6,
            min_neighbor_similarity: 0.7,
            pattern_confidence_weight: 0.4,
            neighbor_similarity_weight: 0.3,
            trajectory_recency_weight: 0.3,
            include_negative_examples: true,
            max_output_tokens: 2000,
            enable_trajectory_hints: true,
        }
    }
}

impl ENagualConfig {
    /// Create a minimal configuration for constrained contexts.
    pub fn minimal() -> Self {
        Self {
            max_patterns: 2,
            max_neighbors: 3,
            max_trajectory_steps: 1,
            include_negative_examples: false,
            max_output_tokens: 500,
            ..Default::default()
        }
    }

    /// Create a verbose configuration for complex tasks.
    pub fn verbose() -> Self {
        Self {
            max_patterns: 10,
            max_neighbors: 20,
            max_trajectory_steps: 5,
            include_negative_examples: true,
            max_output_tokens: 4000,
            ..Default::default()
        }
    }

    /// Set the maximum patterns.
    pub fn with_max_patterns(mut self, max: usize) -> Self {
        self.max_patterns = max;
        self
    }

    /// Set the minimum pattern reward.
    pub fn with_min_reward(mut self, min: f32) -> Self {
        self.min_pattern_reward = min.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum output tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_output_tokens = max;
        self
    }
}

/// A trajectory hint extracted from past reasoning paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryHint {
    /// The decision or action that led to success.
    pub decision: String,

    /// Confidence in this hint.
    pub confidence: f32,

    /// Step type that produced this hint.
    pub step_type: StepType,

    /// Pattern IDs that were involved.
    pub pattern_ids: Vec<String>,

    /// How recent this hint is (0.0 = oldest, 1.0 = most recent).
    pub recency_score: f32,
}

impl TrajectoryHint {
    /// Create a new trajectory hint from a step.
    pub fn from_step(step: &TrajectoryStep, recency_score: f32) -> Self {
        Self {
            decision: step.decision.clone(),
            confidence: step.confidence,
            step_type: step.step_type,
            pattern_ids: step.pattern_ids.iter().map(|id| id.to_string()).collect(),
            recency_score,
        }
    }

    /// Get the combined quality score.
    pub fn quality_score(&self) -> f32 {
        self.confidence * 0.6 + self.recency_score * 0.4
    }
}

/// A few-shot example for LLM prompting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// The input/query for this example.
    pub input: String,

    /// The expected output/response.
    pub output: String,

    /// Quality score for this example.
    pub quality_score: f32,

    /// Optional explanation of why this example is relevant.
    pub explanation: Option<String>,

    /// Tags for categorization.
    pub tags: Vec<String>,
}

impl Example {
    /// Create a new example.
    pub fn new(input: impl Into<String>, output: impl Into<String>, quality_score: f32) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            quality_score: quality_score.clamp(0.0, 1.0),
            explanation: None,
            tags: Vec::new(),
        }
    }

    /// Set the explanation.
    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = Some(explanation.into());
        self
    }

    /// Add tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Create from a pattern.
    pub fn from_pattern(pattern: &Pattern) -> Self {
        Self {
            input: pattern.problem.clone(),
            output: pattern.solution.clone(),
            quality_score: pattern.reliability_score(),
            explanation: pattern.context.clone(),
            tags: pattern.tags.clone(),
        }
    }

    /// Create from a scored pattern.
    pub fn from_scored_pattern(scored: &ScoredPattern) -> Self {
        let mut example = Self::from_pattern(&scored.pattern);
        // Boost quality score based on final score
        example.quality_score = (example.quality_score + scored.final_score) / 2.0;
        example
    }
}

/// E_nagual - The computed attention bias injection.
///
/// This struct contains all the components needed to inject learned patterns
/// into LLM prompts, enabling knowledge transfer without model fine-tuning.
#[derive(Debug, Clone)]
pub struct ENagual {
    /// High-reward patterns relevant to the query.
    pub relevant_patterns: Vec<ScoredPattern>,

    /// Hints from successful trajectory history.
    pub trajectory_hints: Vec<TrajectoryHint>,

    /// HNSW neighbor node IDs for context expansion.
    pub hnsw_neighbors: Vec<String>,

    /// Confidence scores for different aspects of the bias.
    pub confidence_scores: HashMap<String, f32>,

    /// Negative examples (patterns that failed).
    pub negative_examples: Vec<Pattern>,

    /// The original query that generated this bias.
    pub query: String,

    /// When this E_nagual was computed.
    pub computed_at: DateTime<Utc>,

    /// Configuration used for computation.
    config: Option<ENagualConfig>,
}

impl ENagual {
    /// Create a new empty E_nagual.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            relevant_patterns: Vec::new(),
            trajectory_hints: Vec::new(),
            hnsw_neighbors: Vec::new(),
            confidence_scores: HashMap::new(),
            negative_examples: Vec::new(),
            query: query.into(),
            computed_at: Utc::now(),
            config: None,
        }
    }

    /// Add relevant patterns.
    pub fn with_patterns(mut self, patterns: Vec<ScoredPattern>) -> Self {
        self.relevant_patterns = patterns;
        self
    }

    /// Add trajectory hints.
    pub fn with_trajectory_hints(mut self, hints: Vec<TrajectoryHint>) -> Self {
        self.trajectory_hints = hints;
        self
    }

    /// Add HNSW neighbors.
    pub fn with_neighbors(mut self, neighbors: Vec<String>) -> Self {
        self.hnsw_neighbors = neighbors;
        self
    }

    /// Add negative examples.
    pub fn with_negative_examples(mut self, examples: Vec<Pattern>) -> Self {
        self.negative_examples = examples;
        self
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: ENagualConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Compute the overall confidence in this bias.
    pub fn overall_confidence(&self) -> f32 {
        if self.confidence_scores.is_empty() {
            return 0.0;
        }

        let sum: f32 = self.confidence_scores.values().sum();
        sum / self.confidence_scores.len() as f32
    }

    /// Check if this E_nagual has meaningful content.
    pub fn has_content(&self) -> bool {
        !self.relevant_patterns.is_empty()
            || !self.trajectory_hints.is_empty()
            || !self.negative_examples.is_empty()
    }

    /// Get the number of patterns.
    pub fn pattern_count(&self) -> usize {
        self.relevant_patterns.len()
    }

    /// Get the number of trajectory hints.
    pub fn hint_count(&self) -> usize {
        self.trajectory_hints.len()
    }

    /// Convert to a prompt prefix for system prompts.
    ///
    /// This creates a structured text block that can be prepended to
    /// system prompts to inject learned knowledge.
    pub fn to_prompt_prefix(&self) -> String {
        let mut sections = Vec::new();

        // Header
        sections.push(format!(
            "# Learned Context (Confidence: {:.0}%)",
            self.overall_confidence() * 100.0
        ));

        // Relevant patterns section
        if !self.relevant_patterns.is_empty() {
            sections.push("\n## Relevant Patterns".to_string());
            for (i, scored) in self.relevant_patterns.iter().enumerate() {
                let pattern = &scored.pattern;
                sections.push(format!(
                    "\n### Pattern {} (Similarity: {:.0}%, Reward: {:.0}%)",
                    i + 1,
                    scored.similarity * 100.0,
                    pattern.reward * 100.0
                ));
                sections.push(format!("**Problem:** {}", pattern.problem));
                sections.push(format!("**Solution:** {}", pattern.solution));
                if let Some(ref ctx) = pattern.context {
                    if !ctx.is_empty() {
                        sections.push(format!("**Context:** {}", ctx));
                    }
                }
            }
        }

        // Trajectory hints section
        if !self.trajectory_hints.is_empty() {
            sections.push("\n## Reasoning Hints".to_string());
            sections.push(
                "Based on past successful reasoning trajectories:".to_string(),
            );
            for hint in &self.trajectory_hints {
                sections.push(format!(
                    "- {} (Confidence: {:.0}%)",
                    hint.decision,
                    hint.confidence * 100.0
                ));
            }
        }

        // Negative examples section (what to avoid)
        if !self.negative_examples.is_empty() {
            sections.push("\n## Approaches to Avoid".to_string());
            for pattern in &self.negative_examples {
                sections.push(format!(
                    "- **{}**: {} (Failed approach)",
                    pattern.problem,
                    pattern.solution
                ));
            }
        }

        sections.join("\n")
    }

    /// Convert to few-shot examples.
    ///
    /// This creates a list of input-output pairs that can be used
    /// as few-shot examples for models that support this format.
    pub fn to_few_shot_examples(&self) -> Vec<Example> {
        let mut examples = Vec::new();

        // Add positive examples from patterns
        for scored in &self.relevant_patterns {
            let example = Example::from_scored_pattern(scored);
            examples.push(example);
        }

        // Sort by quality score
        examples.sort_by(|a, b| {
            b.quality_score
                .partial_cmp(&a.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        examples
    }

    /// Convert to XML-formatted context.
    ///
    /// This creates an XML-style structured format that some models
    /// (like Claude) can parse effectively.
    pub fn to_xml_context(&self) -> String {
        let mut parts = Vec::new();

        parts.push("<learned_context>".to_string());
        parts.push(format!(
            "  <confidence>{:.2}</confidence>",
            self.overall_confidence()
        ));

        // Patterns
        if !self.relevant_patterns.is_empty() {
            parts.push("  <patterns>".to_string());
            for scored in &self.relevant_patterns {
                let pattern = &scored.pattern;
                parts.push("    <pattern>".to_string());
                parts.push(format!(
                    "      <similarity>{:.2}</similarity>",
                    scored.similarity
                ));
                parts.push(format!(
                    "      <reward>{:.2}</reward>",
                    pattern.reward
                ));
                parts.push(format!(
                    "      <problem>{}</problem>",
                    escape_xml(&pattern.problem)
                ));
                parts.push(format!(
                    "      <solution>{}</solution>",
                    escape_xml(&pattern.solution)
                ));
                if let Some(ref ctx) = pattern.context {
                    if !ctx.is_empty() {
                        parts.push(format!(
                            "      <context>{}</context>",
                            escape_xml(ctx)
                        ));
                    }
                }
                parts.push("    </pattern>".to_string());
            }
            parts.push("  </patterns>".to_string());
        }

        // Trajectory hints
        if !self.trajectory_hints.is_empty() {
            parts.push("  <reasoning_hints>".to_string());
            for hint in &self.trajectory_hints {
                parts.push("    <hint>".to_string());
                parts.push(format!(
                    "      <decision>{}</decision>",
                    escape_xml(&hint.decision)
                ));
                parts.push(format!("      <confidence>{:.2}</confidence>", hint.confidence));
                parts.push("    </hint>".to_string());
            }
            parts.push("  </reasoning_hints>".to_string());
        }

        // Negative examples
        if !self.negative_examples.is_empty() {
            parts.push("  <avoid>".to_string());
            for pattern in &self.negative_examples {
                parts.push("    <failed_approach>".to_string());
                parts.push(format!(
                    "      <problem>{}</problem>",
                    escape_xml(&pattern.problem)
                ));
                parts.push(format!(
                    "      <failed_solution>{}</failed_solution>",
                    escape_xml(&pattern.solution)
                ));
                parts.push("    </failed_approach>".to_string());
            }
            parts.push("  </avoid>".to_string());
        }

        parts.push("</learned_context>".to_string());

        parts.join("\n")
    }

    /// Convert to JSON context.
    ///
    /// This creates a JSON-formatted context that can be easily parsed
    /// by models or used in structured APIs.
    pub fn to_json_context(&self) -> String {
        // Build a serializable representation manually
        let mut obj = serde_json::Map::new();

        obj.insert("query".to_string(), serde_json::Value::String(self.query.clone()));
        obj.insert("overall_confidence".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(self.overall_confidence() as f64).unwrap()));
        obj.insert("computed_at".to_string(),
            serde_json::Value::String(self.computed_at.to_rfc3339()));

        // Add patterns
        let patterns: Vec<serde_json::Value> = self.relevant_patterns.iter().map(|scored| {
            let pattern = &scored.pattern;
            let mut p = serde_json::Map::new();
            p.insert("problem".to_string(), serde_json::Value::String(pattern.problem.clone()));
            p.insert("solution".to_string(), serde_json::Value::String(pattern.solution.clone()));
            p.insert("similarity".to_string(),
                serde_json::Value::Number(serde_json::Number::from_f64(scored.similarity as f64).unwrap()));
            p.insert("reward".to_string(),
                serde_json::Value::Number(serde_json::Number::from_f64(pattern.reward as f64).unwrap()));
            if let Some(ref ctx) = pattern.context {
                p.insert("context".to_string(), serde_json::Value::String(ctx.clone()));
            }
            serde_json::Value::Object(p)
        }).collect();
        obj.insert("patterns".to_string(), serde_json::Value::Array(patterns));

        // Add trajectory hints
        let hints: Vec<serde_json::Value> = self.trajectory_hints.iter().map(|hint| {
            let mut h = serde_json::Map::new();
            h.insert("decision".to_string(), serde_json::Value::String(hint.decision.clone()));
            h.insert("confidence".to_string(),
                serde_json::Value::Number(serde_json::Number::from_f64(hint.confidence as f64).unwrap()));
            serde_json::Value::Object(h)
        }).collect();
        obj.insert("trajectory_hints".to_string(), serde_json::Value::Array(hints));

        // Add neighbors
        let neighbors: Vec<serde_json::Value> = self.hnsw_neighbors.iter()
            .map(|id| serde_json::Value::String(id.clone()))
            .collect();
        obj.insert("hnsw_neighbors".to_string(), serde_json::Value::Array(neighbors));

        serde_json::to_string_pretty(&obj).unwrap_or_default()
    }

    /// Get patterns as references.
    pub fn patterns(&self) -> &[ScoredPattern] {
        &self.relevant_patterns
    }

    /// Get trajectory hints as references.
    pub fn hints(&self) -> &[TrajectoryHint] {
        &self.trajectory_hints
    }

    /// Get confidence score for a specific aspect.
    pub fn confidence_for(&self, aspect: &str) -> Option<f32> {
        self.confidence_scores.get(aspect).copied()
    }

    /// Set a confidence score for an aspect.
    pub fn set_confidence(&mut self, aspect: impl Into<String>, score: f32) {
        self.confidence_scores.insert(aspect.into(), score.clamp(0.0, 1.0));
    }
}

/// Builder for computing E_nagual from various sources.
pub struct ENagualBuilder {
    query: String,
    config: ENagualConfig,
    patterns: Vec<ScoredPattern>,
    trajectory_hints: Vec<TrajectoryHint>,
    neighbor_ids: Vec<String>,
    negative_patterns: Vec<Pattern>,
}

impl ENagualBuilder {
    /// Create a new builder for the given query.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            config: ENagualConfig::default(),
            patterns: Vec::new(),
            trajectory_hints: Vec::new(),
            neighbor_ids: Vec::new(),
            negative_patterns: Vec::new(),
        }
    }

    /// Set the configuration.
    pub fn config(mut self, config: ENagualConfig) -> Self {
        self.config = config;
        self
    }

    /// Add patterns from retrieval result.
    pub fn with_patterns(mut self, patterns: Vec<ScoredPattern>) -> Self {
        // Filter by minimum reward and limit
        let filtered: Vec<ScoredPattern> = patterns
            .into_iter()
            .filter(|p| p.pattern.reward >= self.config.min_pattern_reward)
            .take(self.config.max_patterns)
            .collect();
        self.patterns = filtered;
        self
    }

    /// Add trajectory hints from trajectory history.
    pub fn with_trajectories(mut self, trajectories: &[Trajectory]) -> Self {
        if !self.config.enable_trajectory_hints {
            return self;
        }

        let mut all_hints: Vec<TrajectoryHint> = Vec::new();
        let now = Utc::now();

        for trajectory in trajectories {
            if !trajectory.success {
                continue; // Only use successful trajectories
            }

            for step in &trajectory.steps {
                // Calculate recency score
                let age_hours = (now - step.timestamp).num_hours() as f32;
                let recency = (-age_hours / 720.0).exp(); // Decay over 30 days

                let hint = TrajectoryHint::from_step(step, recency);
                all_hints.push(hint);
            }
        }

        // Sort by quality and take top N
        all_hints.sort_by(|a, b| {
            b.quality_score()
                .partial_cmp(&a.quality_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_hints.truncate(self.config.max_trajectory_steps);

        self.trajectory_hints = all_hints;
        self
    }

    /// Add HNSW neighbors from search results.
    pub fn with_hnsw_neighbors(mut self, neighbors: &[SimilarNode]) -> Self {
        let filtered: Vec<String> = neighbors
            .iter()
            .filter(|n| n.similarity >= self.config.min_neighbor_similarity)
            .take(self.config.max_neighbors)
            .map(|n| n.node.id.clone())
            .collect();
        self.neighbor_ids = filtered;
        self
    }

    /// Add negative examples (failed patterns).
    pub fn with_negative_patterns(mut self, patterns: Vec<Pattern>) -> Self {
        if !self.config.include_negative_examples {
            return self;
        }

        // Take only low-reward patterns as negative examples
        let negative: Vec<Pattern> = patterns
            .into_iter()
            .filter(|p| p.reward < 0.4 && p.success_rate < 0.5)
            .take(2) // Limit negative examples
            .collect();
        self.negative_patterns = negative;
        self
    }

    /// Auto-fetch neighbors from HNSW search and add them to the builder.
    ///
    /// This runs a similarity search against the ProfDAG HNSW index using the
    /// provided query embedding and automatically adds matching neighbors. The
    /// builder's configured `min_neighbor_similarity` and `max_neighbors` are
    /// applied as filters after the search returns.
    ///
    /// # Arguments
    ///
    /// * `search` - Reference to a ProfDAGSearch engine with a built index
    /// * `query_embedding` - The query vector to search against (must match the
    ///   configured embedding dimension)
    /// * `max_neighbors` - Maximum number of neighbors to fetch from the index.
    ///   Results are further filtered by the builder's `min_neighbor_similarity`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let e_nagual = ENagualBuilder::new("database timeouts")
    ///     .from_search(&search, &query_embedding, 10)
    ///     .await
    ///     .build();
    /// ```
    pub async fn from_search(
        mut self,
        search: &ProfDAGSearch,
        query_embedding: &[f32],
        max_neighbors: usize,
    ) -> Self {
        match search
            .find_similar(query_embedding, max_neighbors, self.config.min_neighbor_similarity as f32)
            .await
        {
            Ok(results) => {
                self = self.with_hnsw_neighbors(&results);
            }
            Err(e) => {
                tracing::warn!("ENagualBuilder::from_search failed: {}", e);
            }
        }
        self
    }

    /// Auto-fetch similar nodes from ProfDAG storage and add as context neighbors.
    ///
    /// This fetches recent nodes of the specified type directly from storage
    /// (without an embedding-based search) and adds their IDs as neighbor context.
    /// Useful when you want to provide recent activity context rather than
    /// semantically similar context.
    ///
    /// # Arguments
    ///
    /// * `storage` - Reference to ProfDAG storage
    /// * `node_type` - Optional node type filter. When `None`, fetches all
    ///   four node types (Pattern, Trajectory, Prediction, Decision).
    /// * `limit` - Maximum number of nodes to fetch
    ///
    /// # Example
    ///
    /// ```ignore
    /// let e_nagual = ENagualBuilder::new("database timeouts")
    ///     .from_storage(&storage, Some(NodeType::Pattern), 10)
    ///     .await
    ///     .build();
    /// ```
    pub async fn from_storage(
        mut self,
        storage: &ProfDAGStorage,
        node_type: Option<NodeType>,
        limit: usize,
    ) -> Self {
        let types: Vec<NodeType> = match node_type {
            Some(nt) => vec![nt],
            None => NodeType::all().to_vec(),
        };

        let mut fetched_ids = Vec::new();
        for nt in types {
            match storage.get_nodes_by_type(nt, limit).await {
                Ok(nodes) => {
                    for node in nodes {
                        fetched_ids.push(node.id);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "ENagualBuilder::from_storage failed for type {}: {}",
                        nt,
                        e
                    );
                }
            }
        }

        // Truncate to the configured max_neighbors
        fetched_ids.truncate(self.config.max_neighbors);
        self.neighbor_ids.extend(fetched_ids);
        self
    }

    /// Build the E_nagual instance.
    pub fn build(self) -> ENagual {
        let mut confidence_scores = HashMap::new();

        // Calculate pattern confidence
        if !self.patterns.is_empty() {
            let avg_pattern_confidence: f32 = self.patterns
                .iter()
                .map(|p| p.pattern.confidence)
                .sum::<f32>() / self.patterns.len() as f32;
            confidence_scores.insert("patterns".to_string(), avg_pattern_confidence);
        }

        // Calculate trajectory confidence
        if !self.trajectory_hints.is_empty() {
            let avg_trajectory_confidence: f32 = self.trajectory_hints
                .iter()
                .map(|h| h.confidence)
                .sum::<f32>() / self.trajectory_hints.len() as f32;
            confidence_scores.insert("trajectories".to_string(), avg_trajectory_confidence);
        }

        // Calculate neighbor confidence (based on count)
        if !self.neighbor_ids.is_empty() {
            let neighbor_confidence = (self.neighbor_ids.len() as f32 / self.config.max_neighbors as f32).min(1.0);
            confidence_scores.insert("neighbors".to_string(), neighbor_confidence);
        }

        ENagual {
            relevant_patterns: self.patterns,
            trajectory_hints: self.trajectory_hints,
            hnsw_neighbors: self.neighbor_ids,
            confidence_scores,
            negative_examples: self.negative_patterns,
            query: self.query,
            computed_at: Utc::now(),
            config: Some(self.config),
        }
    }
}

/// Escape special characters for XML.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::reasoning_bank::FactorScores;

    fn create_test_scored_pattern() -> ScoredPattern {
        let pattern = Pattern::new(
            "How to handle database timeouts?",
            "Use connection pooling with retry logic",
            "database.resilience"
        )
        .with_context("Common in microservices")
        .with_confidence(0.85)
        .with_reward(0.9);

        ScoredPattern {
            pattern,
            similarity: 0.92,
            final_score: 0.88,
            factor_scores: FactorScores::default(),
        }
    }

    #[test]
    fn test_e_nagual_config_default() {
        let config = ENagualConfig::default();
        assert_eq!(config.max_patterns, 5);
        assert_eq!(config.max_neighbors, 10);
        assert_eq!(config.min_pattern_reward, 0.6);
    }

    #[test]
    fn test_e_nagual_config_minimal() {
        let config = ENagualConfig::minimal();
        assert_eq!(config.max_patterns, 2);
        assert!(!config.include_negative_examples);
    }

    #[test]
    fn test_e_nagual_new() {
        let e_nagual = ENagual::new("test query");
        assert_eq!(e_nagual.query, "test query");
        assert!(!e_nagual.has_content());
        assert_eq!(e_nagual.pattern_count(), 0);
    }

    #[test]
    fn test_e_nagual_with_patterns() {
        let patterns = vec![create_test_scored_pattern()];
        let e_nagual = ENagual::new("test")
            .with_patterns(patterns);

        assert!(e_nagual.has_content());
        assert_eq!(e_nagual.pattern_count(), 1);
    }

    #[test]
    fn test_e_nagual_to_prompt_prefix() {
        let patterns = vec![create_test_scored_pattern()];
        let e_nagual = ENagual::new("test")
            .with_patterns(patterns);

        let prefix = e_nagual.to_prompt_prefix();
        assert!(prefix.contains("Learned Context"));
        assert!(prefix.contains("Relevant Patterns"));
        assert!(prefix.contains("database timeouts"));
        assert!(prefix.contains("connection pooling"));
    }

    #[test]
    fn test_e_nagual_to_xml_context() {
        let patterns = vec![create_test_scored_pattern()];
        let e_nagual = ENagual::new("test")
            .with_patterns(patterns);

        let xml = e_nagual.to_xml_context();
        assert!(xml.contains("<learned_context>"));
        assert!(xml.contains("<patterns>"));
        assert!(xml.contains("<problem>"));
        assert!(xml.contains("<solution>"));
        assert!(xml.contains("</learned_context>"));
    }

    #[test]
    fn test_e_nagual_to_few_shot_examples() {
        let patterns = vec![create_test_scored_pattern()];
        let e_nagual = ENagual::new("test")
            .with_patterns(patterns);

        let examples = e_nagual.to_few_shot_examples();
        assert_eq!(examples.len(), 1);
        assert!(examples[0].input.contains("database timeouts"));
        assert!(examples[0].output.contains("connection pooling"));
    }

    #[test]
    fn test_trajectory_hint() {
        let hint = TrajectoryHint {
            decision: "Use caching".to_string(),
            confidence: 0.9,
            step_type: StepType::Decision,
            pattern_ids: vec!["pat_1".to_string()],
            recency_score: 0.8,
        };

        let quality = hint.quality_score();
        assert!(quality > 0.8);
    }

    #[test]
    fn test_example_from_pattern() {
        let pattern = Pattern::new(
            "Test problem",
            "Test solution",
            "test"
        )
        .with_context("Test context")
        .with_reward(0.9);

        let example = Example::from_pattern(&pattern);
        assert_eq!(example.input, "Test problem");
        assert_eq!(example.output, "Test solution");
    }

    #[test]
    fn test_e_nagual_builder() {
        let patterns = vec![create_test_scored_pattern()];
        let e_nagual = ENagualBuilder::new("How to optimize?")
            .config(ENagualConfig::default())
            .with_patterns(patterns)
            .build();

        assert!(e_nagual.has_content());
        assert!(e_nagual.confidence_scores.contains_key("patterns"));
    }

    #[test]
    fn test_escape_xml() {
        assert_eq!(escape_xml("a < b"), "a &lt; b");
        assert_eq!(escape_xml("a > b"), "a &gt; b");
        assert_eq!(escape_xml("a & b"), "a &amp; b");
    }

    #[test]
    fn test_overall_confidence() {
        let mut e_nagual = ENagual::new("test");
        e_nagual.set_confidence("patterns", 0.8);
        e_nagual.set_confidence("trajectories", 0.6);

        let overall = e_nagual.overall_confidence();
        assert!((overall - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_builder_with_hnsw_neighbors_filters_by_similarity() {
        use crate::profdag::node::ProfDAGNode;

        let neighbors = vec![
            SimilarNode {
                node: ProfDAGNode::pattern("high sim").with_id("n1"),
                similarity: 0.9,
            },
            SimilarNode {
                node: ProfDAGNode::pattern("low sim").with_id("n2"),
                similarity: 0.3, // Below default min_neighbor_similarity of 0.7
            },
            SimilarNode {
                node: ProfDAGNode::pattern("medium sim").with_id("n3"),
                similarity: 0.75,
            },
        ];

        let e_nagual = ENagualBuilder::new("test query")
            .with_hnsw_neighbors(&neighbors)
            .build();

        // Only n1 (0.9) and n3 (0.75) should pass the 0.7 threshold
        assert_eq!(e_nagual.hnsw_neighbors.len(), 2);
        assert!(e_nagual.hnsw_neighbors.contains(&"n1".to_string()));
        assert!(e_nagual.hnsw_neighbors.contains(&"n3".to_string()));
        assert!(!e_nagual.hnsw_neighbors.contains(&"n2".to_string()));
    }

    #[tokio::test]
    async fn test_builder_from_search() {
        use crate::db::DualWriteAdapter;
        use crate::profdag::node::ProfDAGNode;
        use crate::profdag::search::SearchConfig;

        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let storage = Arc::new(
            ProfDAGStorage::with_defaults(adapter).await.unwrap(),
        );

        // Insert test nodes with embeddings
        let dim = 128;
        let mut embedding1 = vec![0.0f32; dim];
        embedding1[0] = 1.0; // Unit vector along dim 0

        let mut embedding2 = vec![0.0f32; dim];
        embedding2[0] = 0.95;
        embedding2[1] = 0.05; // Close to embedding1

        let node1 = ProfDAGNode::pattern("Handle database timeouts with retries")
            .with_id("node-1")
            .with_embedding(embedding1.clone())
            .with_confidence(0.9);
        let node2 = ProfDAGNode::pattern("Implement circuit breakers for resilience")
            .with_id("node-2")
            .with_embedding(embedding2)
            .with_confidence(0.85);

        storage.insert_node(&node1).await.unwrap();
        storage.insert_node(&node2).await.unwrap();

        // Build search and rebuild index
        let search_config = SearchConfig {
            embedding_dim: dim,
            min_similarity: 0.0,
            ..SearchConfig::default()
        };
        let search = ProfDAGSearch::new(storage.clone(), search_config);
        search.rebuild_index().await.unwrap();

        // Use from_search to auto-fetch neighbors
        let e_nagual = ENagualBuilder::new("database resilience query")
            .config(ENagualConfig {
                min_neighbor_similarity: 0.0, // Accept all results for testing
                ..ENagualConfig::default()
            })
            .from_search(&search, &embedding1, 5)
            .await
            .build();

        // Should have fetched neighbors from HNSW search
        assert!(
            !e_nagual.hnsw_neighbors.is_empty(),
            "from_search should populate hnsw_neighbors"
        );
        // The query is identical to node-1's embedding, so node-1 should be a neighbor
        assert!(
            e_nagual.hnsw_neighbors.contains(&"node-1".to_string())
                || e_nagual.hnsw_neighbors.contains(&"node-2".to_string()),
            "Expected at least one of the inserted nodes in neighbors"
        );
        // Verify neighbor confidence is set in the built result
        assert!(
            e_nagual.confidence_scores.contains_key("neighbors"),
            "Build should set neighbor confidence when neighbors are present"
        );
    }

    #[tokio::test]
    async fn test_builder_from_search_empty_index() {
        use crate::db::DualWriteAdapter;
        use crate::profdag::search::SearchConfig;

        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let storage = Arc::new(
            ProfDAGStorage::with_defaults(adapter).await.unwrap(),
        );

        // Empty index - no nodes inserted
        let search = ProfDAGSearch::new(storage.clone(), SearchConfig::default());
        search.rebuild_index().await.unwrap();

        let query_embedding = vec![0.1f32; 128];
        let e_nagual = ENagualBuilder::new("empty search test")
            .from_search(&search, &query_embedding, 5)
            .await
            .build();

        // Should gracefully handle empty results
        assert!(
            e_nagual.hnsw_neighbors.is_empty(),
            "Empty index should yield no neighbors"
        );
        assert!(
            !e_nagual.confidence_scores.contains_key("neighbors"),
            "No neighbor confidence when no neighbors found"
        );
    }

    #[tokio::test]
    async fn test_builder_from_storage() {
        use crate::db::DualWriteAdapter;
        use crate::profdag::node::ProfDAGNode;

        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let storage = Arc::new(
            ProfDAGStorage::with_defaults(adapter).await.unwrap(),
        );

        // Insert test nodes of different types
        let node1 = ProfDAGNode::pattern("Pattern node 1").with_id("pat-1");
        let node2 = ProfDAGNode::pattern("Pattern node 2").with_id("pat-2");
        let node3 = ProfDAGNode::trajectory("Trajectory node").with_id("traj-1");

        storage.insert_node(&node1).await.unwrap();
        storage.insert_node(&node2).await.unwrap();
        storage.insert_node(&node3).await.unwrap();

        // Fetch only Pattern nodes
        let e_nagual = ENagualBuilder::new("storage fetch test")
            .from_storage(&storage, Some(NodeType::Pattern), 10)
            .await
            .build();

        assert_eq!(
            e_nagual.hnsw_neighbors.len(),
            2,
            "Should have 2 pattern node IDs"
        );
        assert!(e_nagual.hnsw_neighbors.contains(&"pat-1".to_string()));
        assert!(e_nagual.hnsw_neighbors.contains(&"pat-2".to_string()));
        assert!(!e_nagual.hnsw_neighbors.contains(&"traj-1".to_string()));
    }

    #[tokio::test]
    async fn test_builder_from_storage_all_types() {
        use crate::db::DualWriteAdapter;
        use crate::profdag::node::ProfDAGNode;

        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let storage = Arc::new(
            ProfDAGStorage::with_defaults(adapter).await.unwrap(),
        );

        let node1 = ProfDAGNode::pattern("Pattern node").with_id("pat-1");
        let node2 = ProfDAGNode::decision("Decision node").with_id("dec-1");

        storage.insert_node(&node1).await.unwrap();
        storage.insert_node(&node2).await.unwrap();

        // Fetch all types (node_type = None)
        let e_nagual = ENagualBuilder::new("all types test")
            .from_storage(&storage, None, 10)
            .await
            .build();

        assert!(
            e_nagual.hnsw_neighbors.len() >= 2,
            "Should include nodes of all types"
        );
        assert!(e_nagual.hnsw_neighbors.contains(&"pat-1".to_string()));
        assert!(e_nagual.hnsw_neighbors.contains(&"dec-1".to_string()));
    }

    #[tokio::test]
    async fn test_builder_from_storage_respects_max_neighbors() {
        use crate::db::DualWriteAdapter;
        use crate::profdag::node::ProfDAGNode;

        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let storage = Arc::new(
            ProfDAGStorage::with_defaults(adapter).await.unwrap(),
        );

        // Insert more nodes than max_neighbors allows
        for i in 0..15 {
            let node = ProfDAGNode::pattern(format!("Node {}", i))
                .with_id(format!("node-{}", i));
            storage.insert_node(&node).await.unwrap();
        }

        let config = ENagualConfig {
            max_neighbors: 5,
            ..ENagualConfig::default()
        };

        let e_nagual = ENagualBuilder::new("max neighbors test")
            .config(config)
            .from_storage(&storage, Some(NodeType::Pattern), 20)
            .await
            .build();

        assert!(
            e_nagual.hnsw_neighbors.len() <= 5,
            "Should respect max_neighbors config limit, got {}",
            e_nagual.hnsw_neighbors.len()
        );
    }

    #[tokio::test]
    async fn test_builder_from_search_combined_with_patterns() {
        use crate::db::DualWriteAdapter;
        use crate::profdag::node::ProfDAGNode;
        use crate::profdag::search::SearchConfig;

        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let storage = Arc::new(
            ProfDAGStorage::with_defaults(adapter).await.unwrap(),
        );

        // Insert a node with embedding
        let dim = 128;
        let embedding = vec![0.5f32; dim];
        let node = ProfDAGNode::pattern("Test pattern for combined usage")
            .with_id("combined-node")
            .with_embedding(embedding.clone());
        storage.insert_node(&node).await.unwrap();

        let search_config = SearchConfig {
            embedding_dim: dim,
            min_similarity: 0.0,
            ..SearchConfig::default()
        };
        let search = ProfDAGSearch::new(storage.clone(), search_config);
        search.rebuild_index().await.unwrap();

        // Combine patterns + from_search in a single builder chain
        let patterns = vec![create_test_scored_pattern()];
        let e_nagual = ENagualBuilder::new("combined usage test")
            .config(ENagualConfig {
                min_neighbor_similarity: 0.0,
                ..ENagualConfig::default()
            })
            .with_patterns(patterns)
            .from_search(&search, &embedding, 5)
            .await
            .build();

        // Should have both patterns and neighbors
        assert!(e_nagual.has_content(), "Should have pattern content");
        assert_eq!(e_nagual.pattern_count(), 1, "Should have 1 pattern");
        assert!(
            !e_nagual.hnsw_neighbors.is_empty(),
            "Should have neighbors from search"
        );
        assert!(
            e_nagual.confidence_scores.contains_key("patterns"),
            "Should have pattern confidence"
        );
        assert!(
            e_nagual.confidence_scores.contains_key("neighbors"),
            "Should have neighbor confidence"
        );
    }
}
