//! Query Complexity Estimator
//!
//! Extracts features from queries for complexity estimation.
//! Features are designed to capture:
//! - Query structure (length, tokens)
//! - Semantic complexity (embedding characteristics)
//! - Domain specificity (pattern coverage)
//! - Historical performance (accuracy on similar queries)
//!
//! # Features
//!
//! 1. **query_length**: Normalized length of the query text
//! 2. **embedding_norm**: L2 norm of the query embedding (semantic density)
//! 3. **domain_specificity**: How specific vs general the query is
//! 4. **pattern_coverage**: How well existing patterns cover the query
//! 5. **historical_accuracy**: Past accuracy on similar queries

use std::collections::HashMap;

use ndarray::Array1;
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

    /// Weight for embedding norm in complexity calculation.
    pub norm_weight: f32,

    /// Weight for domain specificity.
    pub domain_weight: f32,

    /// Weight for pattern coverage (inverse).
    pub coverage_weight: f32,

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
            norm_weight: 0.15,
            domain_weight: 0.25,
            coverage_weight: 0.25,
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
            norm_weight: 0.20,
            domain_weight: 0.30,
            coverage_weight: 0.20,
            accuracy_weight: 0.10,
            fast_mode: true,
        }
    }

    /// Validate that weights sum to 1.0.
    pub fn validate(&self) -> RouterResult<()> {
        let sum =
            self.length_weight + self.norm_weight + self.domain_weight + self.coverage_weight + self.accuracy_weight;
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
            self.length_weight + self.norm_weight + self.domain_weight + self.coverage_weight + self.accuracy_weight;
        if sum > 0.0 {
            Self {
                length_weight: self.length_weight / sum,
                norm_weight: self.norm_weight / sum,
                domain_weight: self.domain_weight / sum,
                coverage_weight: self.coverage_weight / sum,
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

    /// Normalized embedding norm [0.0, 1.0].
    pub embedding_norm: f32,

    /// Domain specificity score [0.0, 1.0].
    /// Higher = more specialized/technical query.
    pub domain_specificity: f32,

    /// Pattern coverage score [0.0, 1.0].
    /// Higher = more patterns available for this query.
    pub pattern_coverage: f32,

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
            embedding_norm: 0.5,
            domain_specificity: 0.5,
            pattern_coverage: 0.5,
            historical_accuracy: 0.5,
            metadata: HashMap::new(),
        }
    }

    /// Convert to a feature vector for the FastGRNN model.
    pub fn to_vector(&self) -> Vec<f32> {
        vec![
            self.query_length,
            self.embedding_norm,
            self.domain_specificity,
            self.pattern_coverage,
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
            embedding_norm: v[1],
            domain_specificity: v[2],
            pattern_coverage: v[3],
            historical_accuracy: v[4],
            metadata: HashMap::new(),
        })
    }

    /// Calculate a simple weighted complexity score without using FastGRNN.
    pub fn simple_complexity(&self, config: &EstimatorConfig) -> f32 {
        let config = config.normalized();

        // Higher length -> higher complexity
        let length_contrib = self.query_length * config.length_weight;

        // Higher norm (more semantic content) -> higher complexity
        let norm_contrib = self.embedding_norm * config.norm_weight;

        // Higher domain specificity -> higher complexity
        let domain_contrib = self.domain_specificity * config.domain_weight;

        // Lower pattern coverage -> higher complexity (inverse)
        let coverage_contrib = (1.0 - self.pattern_coverage) * config.coverage_weight;

        // Lower historical accuracy -> higher complexity (inverse)
        let accuracy_contrib = (1.0 - self.historical_accuracy) * config.accuracy_weight;

        (length_contrib + norm_contrib + domain_contrib + coverage_contrib + accuracy_contrib)
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
    pub fn extract_features(&self, query: &str, embedding: &[f32]) -> RouterResult<ComplexityFeatures> {
        // Query length feature
        let query_length = self.compute_length_feature(query);

        // Embedding norm feature
        let embedding_norm = self.compute_embedding_norm(embedding)?;

        // Domain specificity feature
        let domain_specificity = self.compute_domain_specificity(query);

        // Pattern coverage (default for now, can be enhanced with actual pattern lookup)
        let pattern_coverage = if self.config.fast_mode {
            0.5 // Neutral default in fast mode
        } else {
            self.estimate_pattern_coverage(embedding)
        };

        // Historical accuracy
        let historical_accuracy = self.get_historical_accuracy(query);

        Ok(ComplexityFeatures {
            query_length,
            embedding_norm,
            domain_specificity,
            pattern_coverage,
            historical_accuracy,
            metadata: HashMap::new(),
        })
    }

    /// Compute normalized query length feature.
    fn compute_length_feature(&self, query: &str) -> f32 {
        let len = query.chars().count();
        let normalized = len as f32 / self.config.max_query_length as f32;
        normalized.clamp(0.0, 1.0)
    }

    /// Compute normalized embedding norm feature.
    fn compute_embedding_norm(&self, embedding: &[f32]) -> RouterResult<f32> {
        if embedding.is_empty() {
            return Err(super::RouterError::FeatureExtraction(
                "Empty embedding".to_string(),
            ));
        }

        let arr = Array1::from_vec(embedding.to_vec());
        let norm = arr.dot(&arr).sqrt();

        // Normalize to [0, 1] assuming max norm of ~1.5 for normalized embeddings
        let normalized = (norm / 1.5).clamp(0.0, 1.0);
        Ok(normalized)
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

    /// Estimate pattern coverage based on embedding similarity.
    ///
    /// In a full implementation, this would query the pattern store.
    /// For now, we use a heuristic based on embedding characteristics.
    fn estimate_pattern_coverage(&self, embedding: &[f32]) -> f32 {
        // Heuristic: embeddings with moderate variance tend to have better coverage
        let arr = Array1::from_vec(embedding.to_vec());
        let mean = arr.mean().unwrap_or(0.0);
        let variance = arr.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);

        // Normalize variance (typical range 0.001 - 0.1 for normalized embeddings)
        let normalized_var = (variance / 0.05).clamp(0.0, 1.0);

        // Higher variance = more unique = potentially lower coverage
        1.0 - (normalized_var * 0.5)
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
        // 2. Pattern coverage is clear (not near 0.5)
        // 3. Features are not all neutral

        let history_conf = if (features.historical_accuracy - 0.5).abs() > 0.1 {
            0.3
        } else {
            0.1
        };

        let coverage_conf = if (features.pattern_coverage - 0.5).abs() > 0.2 {
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

        (history_conf + coverage_conf + variance_conf).clamp(0.3, 1.0)
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
            embedding_norm: 0.2,
            domain_specificity: 0.3,
            pattern_coverage: 0.4,
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
        assert!(f.embedding_norm > 0.0);
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
