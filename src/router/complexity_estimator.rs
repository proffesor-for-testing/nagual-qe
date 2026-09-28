//! Query Complexity Estimator
//!
//! Extracts features from queries for complexity estimation. All five features are computed from
//! the query text (plus recorded accuracy); each is in `[0, 1]`.
//!
//! # Features
//!
//! 1. **query_length**: log-scaled length of the query text
//! 2. **reasoning_demand**: cues that the task needs design, proof, trade-offs, diagnosis or planning
//! 3. **domain_specificity**: how technical vs general the vocabulary is
//! 4. **structure**: code blocks, several questions, list items, explicit constraints
//! 5. **historical_accuracy**: past accuracy recorded for the same query
//!
//! Until 0.2.0, features 2 and 4 were `embedding_norm` and `pattern_coverage`, both derived from the
//! query embedding. For the (normalised) embeddings Nagual produces, the norm is constant and the
//! variance heuristic nearly so — they carried no information about the query, and the router
//! scored almost everything 0.47–0.53. The embedding is still validated but no longer scored.
//! The FastGRNN weights are trained on `models/router_queries.jsonl` (see `models/train_fastgrnn.py`).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::RouterResult;

/// Configuration for the complexity estimator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatorConfig {
    /// Maximum query length for normalization (characters).
    pub max_query_length: usize,

    /// Maximum token count for normalization.
    pub max_token_count: usize,

    /// Expected embedding dimension.
    pub embedding_dim: usize,

    /// Minimum similarity threshold for pattern coverage.
    pub pattern_similarity_threshold: f32,

    /// Weight for length in complexity calculation.
    pub length_weight: f32,

    /// Weight for reasoning demand.
    pub reasoning_weight: f32,

    /// Weight for domain specificity.
    pub domain_weight: f32,

    /// Weight for structural complexity.
    pub structure_weight: f32,

    /// Weight for historical accuracy (inverse).
    pub accuracy_weight: f32,

    /// Whether to use fast mode (skip some features).
    pub fast_mode: bool,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            max_query_length: 2000,
            max_token_count: 500,
            embedding_dim: 128,
            pattern_similarity_threshold: 0.7,
            length_weight: 0.15,
            reasoning_weight: 0.15,
            domain_weight: 0.25,
            structure_weight: 0.25,
            accuracy_weight: 0.20,
            fast_mode: false,
        }
    }
}

impl EstimatorConfig {
    /// Create a fast configuration that skips expensive features.
    pub fn fast() -> Self {
        Self {
            max_query_length: 2000,
            max_token_count: 500,
            embedding_dim: 128,
            pattern_similarity_threshold: 0.7,
            length_weight: 0.20,
            reasoning_weight: 0.20,
            domain_weight: 0.30,
            structure_weight: 0.20,
            accuracy_weight: 0.10,
            fast_mode: true,
        }
    }

    /// Validate that weights sum to 1.0.
    pub fn validate(&self) -> RouterResult<()> {
        let sum =
            self.length_weight + self.reasoning_weight + self.domain_weight + self.structure_weight + self.accuracy_weight;
        if (sum - 1.0).abs() > 0.01 {
            return Err(super::RouterError::InvalidConfig(format!(
                "Feature weights must sum to 1.0, got {}",
                sum
            )));
        }
        Ok(())
    }

    /// Normalize weights to sum to 1.0.
    pub fn normalized(&self) -> Self {
        let sum =
            self.length_weight + self.reasoning_weight + self.domain_weight + self.structure_weight + self.accuracy_weight;
        if sum > 0.0 {
            Self {
                length_weight: self.length_weight / sum,
                reasoning_weight: self.reasoning_weight / sum,
                domain_weight: self.domain_weight / sum,
                structure_weight: self.structure_weight / sum,
                accuracy_weight: self.accuracy_weight / sum,
                ..self.clone()
            }
        } else {
            Self::default()
        }
    }
}

/// Extracted features from a query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityFeatures {
    /// Normalized query length [0.0, 1.0].
    pub query_length: f32,

    /// Reasoning demand [0.0, 1.0]: design, proof, trade-off, diagnosis and planning cues.
    pub reasoning_demand: f32,

    /// Domain specificity score [0.0, 1.0].
    /// Higher = more specialized/technical query.
    pub domain_specificity: f32,

    /// Structural complexity [0.0, 1.0]: code blocks, several questions, list items, constraints.
    pub structure: f32,

    /// Historical accuracy on similar queries [0.0, 1.0].
    /// Higher = better past performance.
    pub historical_accuracy: f32,

    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, f32>,
}

impl ComplexityFeatures {
    /// Create features with default/neutral values.
    pub fn neutral() -> Self {
        Self {
            query_length: 0.5,
            reasoning_demand: 0.5,
            domain_specificity: 0.5,
            structure: 0.5,
            historical_accuracy: 0.5,
            metadata: HashMap::new(),
        }
    }

    /// Convert to a feature vector for the FastGRNN model.
    pub fn to_vector(&self) -> Vec<f32> {
        vec![
            self.query_length,
            self.reasoning_demand,
            self.domain_specificity,
            self.structure,
            self.historical_accuracy,
        ]
    }

    /// Create features from a vector.
    pub fn from_vector(v: &[f32]) -> Option<Self> {
        if v.len() != 5 {
            return None;
        }
        Some(Self {
            query_length: v[0],
            reasoning_demand: v[1],
            domain_specificity: v[2],
            structure: v[3],
            historical_accuracy: v[4],
            metadata: HashMap::new(),
        })
    }

    /// Calculate a simple weighted complexity score without using FastGRNN.
    pub fn simple_complexity(&self, config: &EstimatorConfig) -> f32 {
        let config = config.normalized();

        // Higher length -> higher complexity
        let length_contrib = self.query_length * config.length_weight;

        // More reasoning cues -> higher complexity
        let reasoning_contrib = self.reasoning_demand * config.reasoning_weight;

        // Higher domain specificity -> higher complexity
        let domain_contrib = self.domain_specificity * config.domain_weight;

        // More structure (code, several questions, constraints) -> higher complexity
        let structure_contrib = self.structure * config.structure_weight;

        // Lower historical accuracy -> higher complexity (inverse)
        let accuracy_contrib = (1.0 - self.historical_accuracy) * config.accuracy_weight;

        (length_contrib + reasoning_contrib + domain_contrib + structure_contrib + accuracy_contrib)
            .clamp(0.0, 1.0)
    }
}

/// Complexity score result with detailed breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityScore {
    /// Overall complexity score [0.0, 1.0].
    pub score: f32,

    /// Complexity level classification.
    pub level: ComplexityLevel,

    /// Extracted features.
    pub features: ComplexityFeatures,

    /// Confidence in the estimate [0.0, 1.0].
    pub confidence: f32,

    /// Time taken to compute (microseconds).
    pub computation_time_us: u64,
}

impl ComplexityScore {
    /// Create a new complexity score.
    pub fn new(score: f32, features: ComplexityFeatures, confidence: f32, time_us: u64) -> Self {
        Self {
            score,
            level: ComplexityLevel::from_score(score),
            features,
            confidence,
            computation_time_us: time_us,
        }
    }
}

/// Complexity level classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexityLevel {
    /// Simple query, can use local small model.
    Low,
    /// Moderate query, use local large model.
    Medium,
    /// Complex query, use cloud API.
    High,
    /// Very complex query, use best available model.
    VeryHigh,
}

impl ComplexityLevel {
    /// Convert a score to a complexity level.
    pub fn from_score(score: f32) -> Self {
        if score < 0.3 {
            ComplexityLevel::Low
        } else if score < 0.5 {
            ComplexityLevel::Medium
        } else if score < 0.7 {
            ComplexityLevel::High
        } else {
            ComplexityLevel::VeryHigh
        }
    }

    /// Get the score threshold for this level.
    pub fn threshold(&self) -> f32 {
        match self {
            ComplexityLevel::Low => 0.3,
            ComplexityLevel::Medium => 0.5,
            ComplexityLevel::High => 0.7,
            ComplexityLevel::VeryHigh => 1.0,
        }
    }

    /// Get string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            ComplexityLevel::Low => "low",
            ComplexityLevel::Medium => "medium",
            ComplexityLevel::High => "high",
            ComplexityLevel::VeryHigh => "very_high",
        }
    }
}

/// Query complexity estimator.
///
/// Extracts features from queries for complexity estimation.
pub struct ComplexityEstimator {
    /// Configuration.
    config: EstimatorConfig,

    /// Domain keywords for specificity detection.
    domain_keywords: HashMap<String, f32>,

    /// Historical accuracy cache (query_hash -> accuracy).
    accuracy_cache: parking_lot::RwLock<HashMap<u64, f32>>,
}

impl ComplexityEstimator {
    /// Create a new complexity estimator.
    pub fn new(config: EstimatorConfig) -> Self {
        Self {
            config,
            domain_keywords: Self::default_domain_keywords(),
            accuracy_cache: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Create default domain keywords with specificity scores.
    fn default_domain_keywords() -> HashMap<String, f32> {
        let mut keywords = HashMap::new();

        // Technical/programming terms (high specificity)
        for term in &[
            "algorithm", "implementation", "optimization", "architecture",
            "database", "async", "concurrent", "thread", "memory", "cache",
            "neural", "transformer", "embedding", "gradient", "backpropagation",
            "kubernetes", "docker", "microservice", "api", "graphql",
            "cryptography", "encryption", "hash", "signature", "certificate",
        ] {
            keywords.insert(term.to_string(), 0.8);
        }

        // Moderate specificity terms
        for term in &[
            "function", "class", "method", "variable", "type", "error",
            "debug", "test", "deploy", "build", "compile", "runtime",
            "server", "client", "request", "response", "data", "model",
        ] {
            keywords.insert(term.to_string(), 0.5);
        }

        // General terms (low specificity)
        for term in &[
            "how", "what", "why", "when", "where", "which", "can", "should",
            "help", "explain", "describe", "show", "tell", "give", "make",
        ] {
            keywords.insert(term.to_string(), 0.2);
        }

        keywords
    }

    /// Extract features from a query.
    ///
    /// The embedding is validated (non-empty, finite) but not scored: see the module docs.
    pub fn extract_features(&self, query: &str, embedding: &[f32]) -> RouterResult<ComplexityFeatures> {
        Self::validate_embedding(embedding)?;

        Ok(ComplexityFeatures {
            query_length: self.compute_length_feature(query),
            reasoning_demand: Self::compute_reasoning_demand(query),
            domain_specificity: self.compute_domain_specificity(query),
            structure: Self::compute_structure(query),
            historical_accuracy: self.get_historical_accuracy(query),
            metadata: HashMap::new(),
        })
    }

    fn validate_embedding(embedding: &[f32]) -> RouterResult<()> {
        if embedding.is_empty() {
            return Err(super::RouterError::FeatureExtraction(
                "Empty embedding".to_string(),
            ));
        }
        // A NaN/inf component made the complexity NaN; every threshold comparison is then false,
        // so the selector fell through to its most expensive tier (Claude). Reject at the boundary.
        if embedding.iter().any(|x| !x.is_finite()) {
            return Err(super::RouterError::FeatureExtraction(
                "Embedding contains NaN or infinite values".to_string(),
            ));
        }
        Ok(())
    }

    /// Log-scaled query length: a 12-character question scores ~0.34, 200 characters ~0.70,
    /// `max_query_length` and above 1.0. Linear scaling put every normal question below 0.05.
    fn compute_length_feature(&self, query: &str) -> f32 {
        let len = query.chars().count() as f32;
        let max = self.config.max_query_length.max(1) as f32;
        ((1.0 + len).ln() / (1.0 + max).ln()).clamp(0.0, 1.0)
    }

    /// Reasoning demand from lexical cues: design, proof, trade-offs, diagnosis, planning,
    /// optimisation and hard constraints. Saturating in the number of distinct cues.
    fn compute_reasoning_demand(query: &str) -> f32 {
        const CUES: &[&str] = &[
            "design", "architect", "prove", "proof", "derive", "trade-off", "tradeoff",
            "compare", "evaluate", "optimi", "analy", "refactor", "migrat", "debug", "diagnos",
            "investigat", "root cause", "why ", "strategy", "plan ", "step by step", "scal",
            "guarantee", "ensure", "without ", "must ", "constraint", "edge case", "benchmark",
            "threat model", "consisten", "fault", "concurren", "race condition", "deadlock",
            "distributed", "invariant", "formal", "complexity", "bottleneck", "rollout",
        ];
        let q = format!(" {} ", query.to_lowercase());
        let hits = CUES.iter().filter(|c| q.contains(*c)).count() as f32;
        1.0 - (-0.45 * hits).exp()
    }

    /// Structural complexity: code blocks, inline code / syntax, several questions or sentences,
    /// list items and explicit multi-part asks.
    fn compute_structure(query: &str) -> f32 {
        let code_blocks = (query.matches("```").count() / 2) as f32;
        let inline_code = (query.matches('`').count() as f32 - 6.0 * code_blocks).max(0.0) / 2.0;
        let syntax = ["::", "->", "()", "=>", "{", "};"]
            .iter()
            .filter(|t| query.contains(*t))
            .count() as f32;
        let questions = query.matches('?').count() as f32;
        let sentences = query
            .split(|c| c == '.' || c == '?' || c == '!' || c == ';' || c == '\n')
            .filter(|p| p.split_whitespace().count() >= 3)
            .count() as f32;
        let list_items = query
            .lines()
            .filter(|l| {
                let t = l.trim_start();
                t.starts_with("- ")
                    || t.starts_with("* ")
                    || (t.chars().next().is_some_and(|c| c.is_ascii_digit()) && t.contains(". "))
            })
            .count() as f32;
        let multi_part = [" and then ", " then ", " also ", " as well as ", " both ", " each "]
            .iter()
            .filter(|t| query.to_lowercase().contains(*t))
            .count() as f32;

        let raw = 0.8 * code_blocks
            + 0.3 * inline_code.min(3.0)
            + 0.25 * syntax
            + 0.25 * (questions - 1.0).max(0.0)
            + 0.2 * (sentences - 1.0).max(0.0)
            + 0.3 * list_items
            + 0.3 * multi_part;
        1.0 - (-raw).exp()
    }

    /// Compute domain specificity based on keyword analysis.
    fn compute_domain_specificity(&self, query: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().collect();

        if words.is_empty() {
            return 0.5; // Neutral for empty queries
        }

        let mut total_specificity = 0.0;
        let mut matched_count = 0;

        for word in &words {
            // Remove common punctuation
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if let Some(&specificity) = self.domain_keywords.get(clean_word) {
                total_specificity += specificity;
                matched_count += 1;
            }
        }

        if matched_count == 0 {
            // No keyword matches - use heuristics
            // Longer words tend to be more specific
            let avg_word_len: f32 = words.iter().map(|w| w.len() as f32).sum::<f32>() / words.len() as f32;
            let len_factor = (avg_word_len / 10.0).clamp(0.0, 1.0);

            // Technical punctuation (::, ->, etc.) indicates specificity
            let has_tech_syntax = query.contains("::") || query.contains("->") || query.contains("()");
            let syntax_factor = if has_tech_syntax { 0.3 } else { 0.0 };

            return (len_factor * 0.5 + syntax_factor + 0.2).clamp(0.0, 1.0);
        }

        (total_specificity / matched_count as f32).clamp(0.0, 1.0)
    }

    /// Get historical accuracy for similar queries.
    fn get_historical_accuracy(&self, query: &str) -> f32 {
        let hash = Self::hash_query(query);
        let cache = self.accuracy_cache.read();
        cache.get(&hash).copied().unwrap_or(0.5) // Default neutral
    }

    /// Record accuracy for a query (for future lookups).
    pub fn record_accuracy(&self, query: &str, accuracy: f32) {
        let hash = Self::hash_query(query);
        let mut cache = self.accuracy_cache.write();
        cache.insert(hash, accuracy.clamp(0.0, 1.0));
    }

    /// Hash a query for cache lookup.
    fn hash_query(query: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        hasher.finish()
    }

    /// Estimate complexity using simple weighted average (no FastGRNN).
    pub fn estimate_simple(
        &self,
        query: &str,
        embedding: &[f32],
    ) -> RouterResult<ComplexityScore> {
        let start = std::time::Instant::now();

        let features = self.extract_features(query, embedding)?;
        let score = features.simple_complexity(&self.config);
        let confidence = self.compute_confidence(&features);

        let time_us = start.elapsed().as_micros() as u64;

        Ok(ComplexityScore::new(score, features, confidence, time_us))
    }

    /// Compute confidence in the complexity estimate.
    fn compute_confidence(&self, features: &ComplexityFeatures) -> f32 {
        // Confidence is higher when:
        // 1. Historical accuracy is available (not 0.5 default)
        // 2. There is a clear reasoning/structure signal
        // 3. Features are not all neutral

        let history_conf = if (features.historical_accuracy - 0.5).abs() > 0.1 {
            0.3
        } else {
            0.1
        };

        // A clear lexical/structural signal (either way) makes the estimate more trustworthy.
        let signal_conf = if features.reasoning_demand > 0.3 || features.structure > 0.3 {
            0.3
        } else {
            0.15
        };

        let feature_variance = {
            let v = features.to_vector();
            let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
            let var: f32 = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
            var.sqrt()
        };
        let variance_conf = (feature_variance * 2.0).clamp(0.0, 0.4);

        (history_conf + signal_conf + variance_conf).clamp(0.3, 1.0)
    }

    /// Get the configuration.
    pub fn config(&self) -> &EstimatorConfig {
        &self.config
    }

    /// Clear the accuracy cache.
    pub fn clear_cache(&self) {
        self.accuracy_cache.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_embedding() -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut emb: Vec<f32> = (0..128).map(|_| rng.gen_range(-1.0..1.0)).collect();
        // Normalize
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            emb.iter_mut().for_each(|x| *x /= norm);
        }
        emb
    }

    #[test]
    fn test_estimator_config_default() {
        let config = EstimatorConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_estimator_config_fast() {
        let config = EstimatorConfig::fast();
        assert!(config.fast_mode);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_complexity_features_neutral() {
        let features = ComplexityFeatures::neutral();
        assert!((features.query_length - 0.5).abs() < 0.001);
        assert!((features.simple_complexity(&EstimatorConfig::default()) - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_complexity_features_to_vector() {
        let features = ComplexityFeatures {
            query_length: 0.1,
            reasoning_demand: 0.2,
            domain_specificity: 0.3,
            structure: 0.4,
            historical_accuracy: 0.5,
            metadata: HashMap::new(),
        };

        let vector = features.to_vector();
        assert_eq!(vector.len(), 5);
        assert!((vector[0] - 0.1).abs() < 0.001);
        assert!((vector[4] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_complexity_features_from_vector() {
        let vector = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let features = ComplexityFeatures::from_vector(&vector);
        assert!(features.is_some());

        let f = features.unwrap();
        assert!((f.query_length - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_complexity_level_from_score() {
        assert_eq!(ComplexityLevel::from_score(0.1), ComplexityLevel::Low);
        assert_eq!(ComplexityLevel::from_score(0.4), ComplexityLevel::Medium);
        assert_eq!(ComplexityLevel::from_score(0.6), ComplexityLevel::High);
        assert_eq!(ComplexityLevel::from_score(0.9), ComplexityLevel::VeryHigh);
    }

    #[test]
    fn test_estimator_creation() {
        let config = EstimatorConfig::default();
        let estimator = ComplexityEstimator::new(config);
        assert!(!estimator.domain_keywords.is_empty());
    }

    #[test]
    fn test_extract_features() {
        let estimator = ComplexityEstimator::new(EstimatorConfig::default());
        let embedding = sample_embedding();

        let features = estimator.extract_features("How do I implement a binary search?", &embedding);
        assert!(features.is_ok());

        let f = features.unwrap();
        assert!(f.query_length > 0.0);
        assert_eq!(f.reasoning_demand, 0.0, "no design/proof/diagnosis cue");

        let hard = estimator
            .extract_features("Design a distributed cache and prove it stays consistent under partitions", &embedding)
            .unwrap();
        assert!(hard.reasoning_demand > 0.5, "{}", hard.reasoning_demand);
    }

    #[test]
    fn test_domain_specificity_technical() {
        let estimator = ComplexityEstimator::new(EstimatorConfig::default());

        // Technical query
        let tech_specificity =
            estimator.compute_domain_specificity("Implement a concurrent algorithm with thread-safe caching");
        assert!(tech_specificity > 0.5);

        // General query
        let gen_specificity =
            estimator.compute_domain_specificity("How can I help you today?");
        assert!(gen_specificity < 0.5);
    }

    #[test]
    fn test_estimate_simple() {
        let estimator = ComplexityEstimator::new(EstimatorConfig::default());
        let embedding = sample_embedding();

        let result = estimator.estimate_simple("What is machine learning?", &embedding);
        assert!(result.is_ok());

        let score = result.unwrap();
        assert!(score.score >= 0.0 && score.score <= 1.0);
        assert!(score.confidence >= 0.0 && score.confidence <= 1.0);
    }

    #[test]
    fn test_record_and_retrieve_accuracy() {
        let estimator = ComplexityEstimator::new(EstimatorConfig::default());

        let query = "Test query for accuracy";
        estimator.record_accuracy(query, 0.95);

        let accuracy = estimator.get_historical_accuracy(query);
        assert!((accuracy - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_clear_cache() {
        let estimator = ComplexityEstimator::new(EstimatorConfig::default());

        estimator.record_accuracy("query1", 0.9);
        estimator.record_accuracy("query2", 0.8);
        estimator.clear_cache();

        // Should return default now
        let accuracy = estimator.get_historical_accuracy("query1");
        assert!((accuracy - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_complexity_score_creation() {
        let features = ComplexityFeatures::neutral();
        let score = ComplexityScore::new(0.6, features, 0.85, 100);

        assert!((score.score - 0.6).abs() < 0.001);
        assert_eq!(score.level, ComplexityLevel::High);
        assert!((score.confidence - 0.85).abs() < 0.001);
        assert_eq!(score.computation_time_us, 100);
    }
}
