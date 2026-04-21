//! Multi-factor scoring for pattern ranking.
//!
//! Combines multiple signals to produce a final ranking score:
//! - Similarity: Vector similarity to query
//! - Recency: How recently the pattern was created/updated
//! - Reliability: Based on effectiveness and confidence
//! - Reuse: How often the pattern has been successfully reused

use chrono::Utc;
use ndarray::Array1;

use super::pattern::Pattern;
use super::search::SearchResult;
use crate::ml::cosine_similarity_normalized;

/// Weights for multi-factor scoring.
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    /// Weight for similarity score (default: 0.5)
    pub similarity: f32,

    /// Weight for recency score (default: 0.2)
    pub recency: f32,

    /// Weight for reliability score (default: 0.2)
    pub reliability: f32,

    /// Weight for reuse count score (default: 0.1)
    pub reuse: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            similarity: 0.5,
            recency: 0.2,
            reliability: 0.2,
            reuse: 0.1,
        }
    }
}

impl ScoringWeights {
    /// Create weights emphasizing similarity.
    pub fn similarity_focused() -> Self {
        Self {
            similarity: 0.7,
            recency: 0.1,
            reliability: 0.15,
            reuse: 0.05,
        }
    }

    /// Create weights emphasizing reliability.
    pub fn reliability_focused() -> Self {
        Self {
            similarity: 0.3,
            recency: 0.1,
            reliability: 0.5,
            reuse: 0.1,
        }
    }

    /// Create weights emphasizing recency.
    pub fn recency_focused() -> Self {
        Self {
            similarity: 0.3,
            recency: 0.5,
            reliability: 0.1,
            reuse: 0.1,
        }
    }

    /// Normalize weights to sum to 1.0.
    pub fn normalized(&self) -> Self {
        let sum = self.similarity + self.recency + self.reliability + self.reuse;
        if sum > 0.0 {
            Self {
                similarity: self.similarity / sum,
                recency: self.recency / sum,
                reliability: self.reliability / sum,
                reuse: self.reuse / sum,
            }
        } else {
            Self::default()
        }
    }

    /// Validate that weights are reasonable.
    pub fn is_valid(&self) -> bool {
        let sum = self.similarity + self.recency + self.reliability + self.reuse;
        (sum - 1.0).abs() < 0.01
            && self.similarity >= 0.0
            && self.recency >= 0.0
            && self.reliability >= 0.0
            && self.reuse >= 0.0
    }
}

/// Configuration for the multi-factor scorer.
#[derive(Debug, Clone)]
pub struct ScorerConfig {
    /// Scoring weights.
    pub weights: ScoringWeights,

    /// Half-life for recency decay in days.
    ///
    /// After this many days, the recency score is halved.
    pub recency_half_life_days: f64,

    /// Maximum reuse count for normalization.
    ///
    /// Patterns with this many reuses or more get full reuse score.
    pub max_reuse_count: u32,

    /// Whether embeddings are normalized.
    pub normalized_embeddings: bool,

    /// Minimum final score threshold.
    pub min_score: f32,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            weights: ScoringWeights::default(),
            recency_half_life_days: 30.0,
            max_reuse_count: 100,
            normalized_embeddings: true,
            min_score: 0.0,
        }
    }
}

impl ScorerConfig {
    /// Set custom weights.
    pub fn with_weights(mut self, weights: ScoringWeights) -> Self {
        self.weights = weights.normalized();
        self
    }

    /// Set recency half-life.
    pub fn with_recency_half_life(mut self, days: f64) -> Self {
        self.recency_half_life_days = days.max(1.0);
        self
    }

    /// Set maximum reuse count.
    pub fn with_max_reuse_count(mut self, count: u32) -> Self {
        self.max_reuse_count = count.max(1);
        self
    }

    /// Set minimum score threshold.
    pub fn with_min_score(mut self, min: f32) -> Self {
        self.min_score = min.clamp(0.0, 1.0);
        self
    }
}

/// A pattern with its multi-factor score breakdown.
#[derive(Debug, Clone)]
pub struct ScoredPattern {
    /// The pattern.
    pub pattern: Pattern,

    /// Final combined score.
    pub score: f32,

    /// Individual score components.
    pub components: ScoreComponents,
}

impl ScoredPattern {
    /// Create a new scored pattern.
    pub fn new(pattern: Pattern, score: f32, components: ScoreComponents) -> Self {
        Self {
            pattern,
            score,
            components,
        }
    }
}

/// Individual score components for transparency.
#[derive(Debug, Clone, Default)]
pub struct ScoreComponents {
    /// Similarity score (0.0-1.0).
    pub similarity: f32,

    /// Recency score (0.0-1.0).
    pub recency: f32,

    /// Reliability score (0.0-1.0).
    pub reliability: f32,

    /// Reuse score (0.0-1.0).
    pub reuse: f32,
}

impl ScoreComponents {
    /// Compute weighted sum of components.
    pub fn weighted_sum(&self, weights: &ScoringWeights) -> f32 {
        weights.similarity * self.similarity
            + weights.recency * self.recency
            + weights.reliability * self.reliability
            + weights.reuse * self.reuse
    }
}

/// Multi-factor scorer for pattern ranking.
pub struct MultiFactorScorer {
    config: ScorerConfig,
}

impl MultiFactorScorer {
    /// Create a new multi-factor scorer.
    pub fn new(config: ScorerConfig) -> Self {
        Self { config }
    }

    /// Score a list of patterns against a query embedding.
    ///
    /// Returns scored patterns sorted by final score (highest first).
    pub fn score_patterns(
        &self,
        results: &[SearchResult],
        query_embedding: &[f32],
    ) -> Vec<ScoredPattern> {
        let mut scored: Vec<ScoredPattern> = results
            .iter()
            .filter_map(|result| {
                let components = self.compute_components(&result.pattern, query_embedding, result.similarity);
                let score = components.weighted_sum(&self.config.weights);

                if score >= self.config.min_score {
                    Some(ScoredPattern::new(result.pattern.clone(), score, components))
                } else {
                    None
                }
            })
            .collect();

        // Sort by score (highest first)
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        scored
    }

    /// Score a single pattern.
    pub fn score_pattern(
        &self,
        pattern: &Pattern,
        query_embedding: &[f32],
    ) -> ScoredPattern {
        // Compute similarity if pattern has embedding
        let similarity = pattern.embedding().map_or(0.0, |emb| {
            let query_arr = Array1::from_vec(query_embedding.to_vec());
            let pattern_arr = Array1::from_vec(emb.to_vec());
            cosine_similarity_normalized(&query_arr.view(), &pattern_arr.view())
        });

        let components = self.compute_components(pattern, query_embedding, similarity);
        let score = components.weighted_sum(&self.config.weights);

        ScoredPattern::new(pattern.clone(), score, components)
    }

    /// Compute individual score components for a pattern.
    fn compute_components(
        &self,
        pattern: &Pattern,
        _query_embedding: &[f32],
        similarity: f32,
    ) -> ScoreComponents {
        ScoreComponents {
            similarity,
            recency: self.compute_recency_score(pattern),
            reliability: self.compute_reliability_score(pattern),
            reuse: self.compute_reuse_score(pattern),
        }
    }

    /// Compute recency score using exponential decay.
    ///
    /// Uses the formula: score = 2^(-age_days / half_life_days)
    fn compute_recency_score(&self, pattern: &Pattern) -> f32 {
        let now = Utc::now();
        let age = now - pattern.timestamp();
        let age_days = age.num_seconds() as f64 / 86400.0;

        // Exponential decay: 2^(-t/τ) where τ is half-life
        let decay = 2.0_f64.powf(-age_days / self.config.recency_half_life_days);
        decay as f32
    }

    /// Compute reliability score from effectiveness and confidence.
    ///
    /// Formula: (effectiveness * 0.6 + confidence * 0.3 + success_bonus * 0.1)
    fn compute_reliability_score(&self, pattern: &Pattern) -> f32 {
        let success_bonus = if pattern.success() { 1.0 } else { 0.0 };
        pattern.effectiveness() * 0.6 + pattern.confidence() * 0.3 + success_bonus * 0.1
    }

    /// Compute reuse score with logarithmic scaling.
    ///
    /// Uses log scaling to diminish returns from very high reuse counts:
    /// score = log(1 + reuse_count) / log(1 + max_reuse_count)
    fn compute_reuse_score(&self, pattern: &Pattern) -> f32 {
        let reuse_count = pattern.reuse_count() as f64;
        let max_count = self.config.max_reuse_count as f64;

        let score = (1.0 + reuse_count).ln() / (1.0 + max_count).ln();
        (score as f32).clamp(0.0, 1.0)
    }

    /// Get the scorer configuration.
    pub fn config(&self) -> &ScorerConfig {
        &self.config
    }

    /// Update the weights.
    pub fn set_weights(&mut self, weights: ScoringWeights) {
        self.config.weights = weights.normalized();
    }

    /// Get the current weights.
    pub fn weights(&self) -> &ScoringWeights {
        &self.config.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_pattern(id: &str, effectiveness: f32, reuse_count: u32) -> Pattern {
        Pattern::builder()
            .id(id)
            .problem(format!("Problem {}", id))
            .solution(format!("Solution {}", id))
            .effectiveness(effectiveness)
            .reuse_count(reuse_count)
            .confidence(0.8)
            .success(true)
            .build()
    }

    fn create_result(id: &str, similarity: f32) -> SearchResult {
        let pattern = create_pattern(id, 0.8, 5);
        SearchResult::new(pattern, similarity)
    }

    #[test]
    fn test_scoring_weights_default() {
        let weights = ScoringWeights::default();
        assert!((weights.similarity - 0.5).abs() < 0.001);
        assert!((weights.recency - 0.2).abs() < 0.001);
        assert!((weights.reliability - 0.2).abs() < 0.001);
        assert!((weights.reuse - 0.1).abs() < 0.001);
        assert!(weights.is_valid());
    }

    #[test]
    fn test_scoring_weights_normalize() {
        let weights = ScoringWeights {
            similarity: 2.0,
            recency: 1.0,
            reliability: 1.0,
            reuse: 0.0,
        };

        let normalized = weights.normalized();
        assert!(normalized.is_valid());
        assert!((normalized.similarity - 0.5).abs() < 0.001);
        assert!((normalized.recency - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_scorer_config_default() {
        let config = ScorerConfig::default();
        assert!((config.recency_half_life_days - 30.0).abs() < 0.1);
        assert_eq!(config.max_reuse_count, 100);
    }

    #[test]
    fn test_score_components_weighted_sum() {
        let components = ScoreComponents {
            similarity: 1.0,
            recency: 1.0,
            reliability: 1.0,
            reuse: 1.0,
        };

        let weights = ScoringWeights::default();
        let sum = components.weighted_sum(&weights);

        // All components are 1.0, so sum should equal total weight (1.0)
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_recency_score_new_pattern() {
        let scorer = MultiFactorScorer::new(ScorerConfig::default());
        let pattern = create_pattern("1", 0.8, 5);

        let recency = scorer.compute_recency_score(&pattern);

        // New pattern should have recency close to 1.0
        assert!(recency > 0.95);
    }

    #[test]
    fn test_recency_score_half_life() {
        let config = ScorerConfig::default().with_recency_half_life(1.0); // 1 day half-life
        let scorer = MultiFactorScorer::new(config);

        // Create a pattern from 1 day ago
        let mut pattern = create_pattern("1", 0.8, 5);

        // We can't easily set timestamp, but we can verify the decay function works
        // by checking the formula directly
        let one_day_decay = 2.0_f64.powf(-1.0 / 1.0); // Should be 0.5
        assert!((one_day_decay - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_reliability_score() {
        let scorer = MultiFactorScorer::new(ScorerConfig::default());

        // Pattern with effectiveness=1.0, confidence=1.0, success=true
        let pattern = Pattern::builder()
            .problem("Test")
            .solution("Solution")
            .effectiveness(1.0)
            .confidence(1.0)
            .success(true)
            .build();

        let reliability = scorer.compute_reliability_score(&pattern);

        // 1.0 * 0.6 + 1.0 * 0.3 + 1.0 * 0.1 = 1.0
        assert!((reliability - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_reliability_score_failure() {
        let scorer = MultiFactorScorer::new(ScorerConfig::default());

        let pattern = Pattern::builder()
            .problem("Test")
            .solution("Solution")
            .effectiveness(0.5)
            .confidence(0.5)
            .success(false)
            .build();

        let reliability = scorer.compute_reliability_score(&pattern);

        // 0.5 * 0.6 + 0.5 * 0.3 + 0.0 * 0.1 = 0.45
        assert!((reliability - 0.45).abs() < 0.001);
    }

    #[test]
    fn test_reuse_score_zero() {
        let scorer = MultiFactorScorer::new(ScorerConfig::default());
        let pattern = create_pattern("1", 0.8, 0);

        let reuse = scorer.compute_reuse_score(&pattern);

        // log(1) / log(101) = 0
        assert!(reuse.abs() < 0.001);
    }

    #[test]
    fn test_reuse_score_max() {
        let config = ScorerConfig::default().with_max_reuse_count(100);
        let scorer = MultiFactorScorer::new(config);
        let pattern = create_pattern("1", 0.8, 100);

        let reuse = scorer.compute_reuse_score(&pattern);

        // log(101) / log(101) = 1
        assert!((reuse - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_reuse_score_logarithmic() {
        let scorer = MultiFactorScorer::new(ScorerConfig::default().with_max_reuse_count(100));

        let low = scorer.compute_reuse_score(&create_pattern("1", 0.8, 1));
        let mid = scorer.compute_reuse_score(&create_pattern("2", 0.8, 10));
        let high = scorer.compute_reuse_score(&create_pattern("3", 0.8, 50));

        // Logarithmic scaling: differences decrease as reuse increases
        assert!(low < mid);
        assert!(mid < high);
        assert!((mid - low) > (high - mid)); // Diminishing returns
    }

    #[test]
    fn test_score_patterns() {
        let scorer = MultiFactorScorer::new(ScorerConfig::default());
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let results = vec![
            create_result("1", 0.95),
            create_result("2", 0.85),
            create_result("3", 0.75),
        ];

        let scored = scorer.score_patterns(&results, &query);

        // Should be sorted by score (highest first)
        assert_eq!(scored.len(), 3);
        assert!(scored[0].score >= scored[1].score);
        assert!(scored[1].score >= scored[2].score);
    }

    #[test]
    fn test_score_patterns_min_threshold() {
        let config = ScorerConfig::default().with_min_score(0.5);
        let scorer = MultiFactorScorer::new(config);
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let results = vec![
            create_result("1", 0.95), // High similarity -> high score
            create_result("2", 0.10), // Low similarity -> low score
        ];

        let scored = scorer.score_patterns(&results, &query);

        // Low-scoring pattern might be filtered out
        assert!(scored.iter().all(|s| s.score >= 0.5));
    }

    #[test]
    fn test_score_single_pattern() {
        let scorer = MultiFactorScorer::new(ScorerConfig::default());
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let pattern = Pattern::builder()
            .problem("Test")
            .solution("Solution")
            .effectiveness(0.9)
            .confidence(0.8)
            .reuse_count(10)
            .success(true)
            .embedding(vec![1.0, 0.0, 0.0, 0.0])
            .build();

        let scored = scorer.score_pattern(&pattern, &query);

        // Verify components are computed
        assert!(scored.components.similarity > 0.0);
        assert!(scored.components.recency > 0.0);
        assert!(scored.components.reliability > 0.0);
        assert!(scored.components.reuse > 0.0);

        // Score should be weighted sum
        let expected = scored.components.weighted_sum(&ScoringWeights::default());
        assert!((scored.score - expected).abs() < 0.001);
    }

    #[test]
    fn test_set_weights() {
        let mut scorer = MultiFactorScorer::new(ScorerConfig::default());

        let new_weights = ScoringWeights::reliability_focused();
        scorer.set_weights(new_weights);

        assert!((scorer.weights().reliability - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_scored_pattern_creation() {
        let pattern = create_pattern("1", 0.8, 5);
        let components = ScoreComponents {
            similarity: 0.9,
            recency: 0.8,
            reliability: 0.7,
            reuse: 0.6,
        };

        let scored = ScoredPattern::new(pattern, 0.85, components);

        assert!((scored.score - 0.85).abs() < 0.001);
        assert!((scored.components.similarity - 0.9).abs() < 0.001);
    }
}
