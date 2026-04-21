//! Probability Calculator - Bayesian probability calculation from historical patterns.
//!
//! This module implements probability calculation using pattern success rates
//! as evidence, with support for:
//! - Weighted pattern contributions based on similarity and recency
//! - Bayesian updating with prior predictions
//! - Confidence estimation based on evidence quantity and quality

use serde::{Deserialize, Serialize};

use super::PredictionResult;

/// Configuration for probability calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityConfig {
    /// Base prior probability when no patterns are available
    pub base_prior: f64,
    /// Weight for similarity in pattern contribution
    pub similarity_weight: f64,
    /// Weight for confidence in pattern contribution
    pub confidence_weight: f64,
    /// Weight for recency in pattern contribution
    pub recency_weight: f64,
    /// Minimum number of patterns for high confidence
    pub high_confidence_threshold: usize,
    /// Smoothing factor for probability estimates (avoids extreme values)
    pub smoothing_factor: f64,
    /// Whether to apply Bayesian updating with prior predictions
    pub use_bayesian_update: bool,
    /// Weight given to prior predictions in Bayesian update
    pub prior_weight: f64,
}

impl Default for ProbabilityConfig {
    fn default() -> Self {
        Self {
            base_prior: 0.5,
            similarity_weight: 0.4,
            confidence_weight: 0.3,
            recency_weight: 0.3,
            high_confidence_threshold: 10,
            smoothing_factor: 0.1,
            use_bayesian_update: true,
            prior_weight: 0.3,
        }
    }
}

impl ProbabilityConfig {
    /// Create a new probability configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base prior probability.
    pub fn with_base_prior(mut self, prior: f64) -> Self {
        self.base_prior = prior.clamp(0.0, 1.0);
        self
    }

    /// Set the similarity weight.
    pub fn with_similarity_weight(mut self, weight: f64) -> Self {
        self.similarity_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the confidence weight.
    pub fn with_confidence_weight(mut self, weight: f64) -> Self {
        self.confidence_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the recency weight.
    pub fn with_recency_weight(mut self, weight: f64) -> Self {
        self.recency_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set whether to use Bayesian updating.
    pub fn with_bayesian_update(mut self, use_bayesian: bool) -> Self {
        self.use_bayesian_update = use_bayesian;
        self
    }
}

/// A weighted pattern for probability calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedPattern {
    /// Success rate of the pattern (0.0-1.0)
    pub success_rate: f64,
    /// Confidence in the pattern (0.0-1.0)
    pub confidence: f64,
    /// Similarity to current context (0.0-1.0)
    pub similarity: f64,
    /// Recency weight (1.0 for recent, decaying for older)
    pub recency_weight: f64,
}

impl WeightedPattern {
    /// Create a new weighted pattern.
    pub fn new(success_rate: f64, confidence: f64, similarity: f64) -> Self {
        Self {
            success_rate: success_rate.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            similarity: similarity.clamp(0.0, 1.0),
            recency_weight: 1.0,
        }
    }

    /// Set the recency weight.
    pub fn with_recency_weight(mut self, weight: f64) -> Self {
        self.recency_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Calculate the total weight for this pattern.
    pub fn total_weight(&self, config: &ProbabilityConfig) -> f64 {
        self.similarity * config.similarity_weight
            + self.confidence * config.confidence_weight
            + self.recency_weight * config.recency_weight
    }
}

/// A prior prediction for Bayesian updating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorPrediction {
    /// The prior probability estimate
    pub probability: f64,
    /// Confidence in the prior
    pub confidence: f64,
    /// Whether the prior was resolved and its outcome
    pub outcome: Option<bool>,
}

impl PriorPrediction {
    /// Create a new prior prediction.
    pub fn new(probability: f64, confidence: f64) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            outcome: None,
        }
    }

    /// Create from a resolved prediction.
    pub fn resolved(probability: f64, confidence: f64, outcome: bool) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            outcome: Some(outcome),
        }
    }

    /// Get the effective probability considering resolution.
    pub fn effective_probability(&self) -> f64 {
        match self.outcome {
            Some(true) => {
                // If outcome was true, boost probability based on how confident we were
                self.probability + (1.0 - self.probability) * self.confidence * 0.5
            }
            Some(false) => {
                // If outcome was false, reduce probability based on how confident we were
                self.probability * (1.0 - self.confidence * 0.5)
            }
            None => self.probability,
        }
    }
}

/// Result of probability calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityResult {
    /// Calculated probability (0.0-1.0)
    pub probability: f64,
    /// Confidence in the probability estimate (0.0-1.0)
    pub confidence: f64,
    /// Lower bound of probability interval
    pub lower_bound: f64,
    /// Upper bound of probability interval
    pub upper_bound: f64,
    /// Number of patterns used
    pub patterns_used: usize,
    /// Total weight of evidence
    pub total_evidence_weight: f64,
    /// Whether Bayesian update was applied
    pub bayesian_updated: bool,
    /// Breakdown of probability sources
    pub breakdown: ProbabilityBreakdown,
}

/// Breakdown of probability sources.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbabilityBreakdown {
    /// Contribution from pattern evidence
    pub pattern_contribution: f64,
    /// Contribution from prior predictions
    pub prior_contribution: f64,
    /// Contribution from base prior
    pub base_prior_contribution: f64,
    /// Effect of smoothing
    pub smoothing_effect: f64,
}

/// Probability calculator that computes probability from patterns.
#[derive(Debug, Clone)]
pub struct ProbabilityCalculator {
    /// Configuration
    pub config: ProbabilityConfig,
}

impl ProbabilityCalculator {
    /// Create a new probability calculator with default configuration.
    pub fn new() -> Self {
        Self {
            config: ProbabilityConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: ProbabilityConfig) -> Self {
        Self { config }
    }

    /// Calculate probability from weighted patterns.
    pub fn calculate(
        &self,
        patterns: &[WeightedPattern],
        prior: Option<&PriorPrediction>,
    ) -> PredictionResult<ProbabilityResult> {
        calculate_probability(patterns, prior, &self.config)
    }
}

impl Default for ProbabilityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate probability from weighted patterns using Bayesian-style updating.
///
/// The calculation follows these steps:
/// 1. Calculate weighted average of pattern success rates
/// 2. Apply smoothing to avoid extreme probabilities
/// 3. Optionally apply Bayesian update with prior predictions
/// 4. Calculate confidence based on evidence quantity and quality
///
/// # Arguments
///
/// * `patterns` - Weighted patterns as evidence
/// * `prior` - Optional prior prediction for Bayesian updating
/// * `config` - Configuration for the calculation
///
/// # Returns
///
/// A `ProbabilityResult` containing the calculated probability and metadata.
pub fn calculate_probability(
    patterns: &[WeightedPattern],
    prior: Option<&PriorPrediction>,
    config: &ProbabilityConfig,
) -> PredictionResult<ProbabilityResult> {
    if patterns.is_empty() && prior.is_none() {
        // No evidence at all, return base prior with low confidence
        return Ok(ProbabilityResult {
            probability: config.base_prior,
            confidence: 0.1,
            lower_bound: 0.0,
            upper_bound: 1.0,
            patterns_used: 0,
            total_evidence_weight: 0.0,
            bayesian_updated: false,
            breakdown: ProbabilityBreakdown {
                base_prior_contribution: config.base_prior,
                ..Default::default()
            },
        });
    }

    // Calculate weighted sum of pattern success rates
    let mut total_weight = 0.0;
    let mut weighted_sum = 0.0;

    for pattern in patterns {
        let weight = pattern.total_weight(config);
        weighted_sum += pattern.success_rate * weight;
        total_weight += weight;
    }

    // Calculate pattern-based probability
    let pattern_probability = if total_weight > 0.0 {
        weighted_sum / total_weight
    } else {
        config.base_prior
    };

    // Apply smoothing to avoid extreme probabilities
    let smoothed_probability = apply_smoothing(pattern_probability, config.smoothing_factor);

    // Initialize breakdown
    let mut breakdown = ProbabilityBreakdown {
        pattern_contribution: smoothed_probability * (1.0 - config.prior_weight),
        prior_contribution: 0.0,
        base_prior_contribution: config.base_prior * config.smoothing_factor,
        smoothing_effect: (smoothed_probability - pattern_probability).abs(),
    };

    // Apply Bayesian update if prior exists and config allows
    let (final_probability, bayesian_updated) =
        if config.use_bayesian_update && prior.is_some() {
            let prior = prior.unwrap();
            let updated =
                bayesian_update(smoothed_probability, prior.effective_probability(), config);
            breakdown.prior_contribution = prior.effective_probability() * config.prior_weight;
            breakdown.pattern_contribution =
                smoothed_probability * (1.0 - config.prior_weight);
            (updated, true)
        } else {
            (smoothed_probability, false)
        };

    // Calculate confidence based on evidence quantity and quality
    let confidence = calculate_confidence(patterns, total_weight, config);

    // Calculate probability interval based on confidence
    let half_width = (1.0 - confidence) * 0.5;
    let lower_bound = (final_probability - half_width).max(0.0);
    let upper_bound = (final_probability + half_width).min(1.0);

    Ok(ProbabilityResult {
        probability: final_probability,
        confidence,
        lower_bound,
        upper_bound,
        patterns_used: patterns.len(),
        total_evidence_weight: total_weight,
        bayesian_updated,
        breakdown,
    })
}

/// Apply smoothing to a probability to avoid extreme values.
fn apply_smoothing(probability: f64, smoothing_factor: f64) -> f64 {
    // Move probability towards 0.5 by the smoothing factor
    probability * (1.0 - smoothing_factor) + 0.5 * smoothing_factor
}

/// Perform Bayesian-style update combining pattern evidence with prior.
fn bayesian_update(pattern_prob: f64, prior_prob: f64, config: &ProbabilityConfig) -> f64 {
    // Weighted combination of pattern probability and prior
    let prior_weight = config.prior_weight;
    pattern_prob * (1.0 - prior_weight) + prior_prob * prior_weight
}

/// Calculate confidence based on evidence quantity and quality.
fn calculate_confidence(
    patterns: &[WeightedPattern],
    total_weight: f64,
    config: &ProbabilityConfig,
) -> f64 {
    if patterns.is_empty() {
        return 0.1;
    }

    // Base confidence from number of patterns
    let quantity_confidence = (patterns.len() as f64
        / config.high_confidence_threshold as f64)
        .min(1.0);

    // Quality confidence from average pattern confidence
    let avg_confidence: f64 =
        patterns.iter().map(|p| p.confidence).sum::<f64>() / patterns.len() as f64;

    // Weight confidence from total evidence weight
    let weight_confidence = (total_weight / (patterns.len() as f64)).min(1.0);

    // Agreement confidence - how consistent are the success rates?
    let agreement_confidence = calculate_agreement(patterns);

    // Combine confidence factors
    let combined = quantity_confidence * 0.3
        + avg_confidence * 0.3
        + weight_confidence * 0.2
        + agreement_confidence * 0.2;

    combined.clamp(0.1, 0.95) // Never completely certain or completely uncertain
}

/// Calculate agreement among patterns (how consistent are their success rates).
fn calculate_agreement(patterns: &[WeightedPattern]) -> f64 {
    if patterns.len() < 2 {
        return 0.5; // Can't measure agreement with single pattern
    }

    // Calculate variance in success rates
    let mean: f64 = patterns.iter().map(|p| p.success_rate).sum::<f64>() / patterns.len() as f64;

    let variance: f64 = patterns
        .iter()
        .map(|p| (p.success_rate - mean).powi(2))
        .sum::<f64>()
        / patterns.len() as f64;

    // Convert variance to agreement score (low variance = high agreement)
    // Max variance for [0,1] range is 0.25 (when half are 0 and half are 1)
    let normalized_variance = (variance / 0.25).min(1.0);
    1.0 - normalized_variance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability_config_defaults() {
        let config = ProbabilityConfig::default();
        assert!((config.base_prior - 0.5).abs() < 0.001);
        assert!(config.use_bayesian_update);
    }

    #[test]
    fn test_weighted_pattern() {
        let pattern = WeightedPattern::new(0.8, 0.9, 0.85).with_recency_weight(0.95);

        assert!((pattern.success_rate - 0.8).abs() < 0.001);
        assert!((pattern.confidence - 0.9).abs() < 0.001);
        assert!((pattern.similarity - 0.85).abs() < 0.001);
        assert!((pattern.recency_weight - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_weighted_pattern_total_weight() {
        let config = ProbabilityConfig::default();
        let pattern = WeightedPattern::new(0.8, 0.9, 0.7);

        let weight = pattern.total_weight(&config);
        assert!(weight > 0.0);
        assert!(weight <= 1.0);
    }

    #[test]
    fn test_prior_prediction() {
        let prior = PriorPrediction::new(0.7, 0.8);
        assert!((prior.probability - 0.7).abs() < 0.001);
        assert!((prior.effective_probability() - 0.7).abs() < 0.001);

        let resolved_true = PriorPrediction::resolved(0.7, 0.8, true);
        assert!(resolved_true.effective_probability() > 0.7);

        let resolved_false = PriorPrediction::resolved(0.7, 0.8, false);
        assert!(resolved_false.effective_probability() < 0.7);
    }

    #[test]
    fn test_calculate_probability_no_patterns() {
        let config = ProbabilityConfig::default();
        let result = calculate_probability(&[], None, &config).unwrap();

        assert!((result.probability - config.base_prior).abs() < 0.001);
        assert!(result.confidence < 0.2); // Low confidence with no evidence
        assert_eq!(result.patterns_used, 0);
    }

    #[test]
    fn test_calculate_probability_single_pattern() {
        let config = ProbabilityConfig::default();
        let patterns = vec![WeightedPattern::new(0.8, 0.9, 0.95)];

        let result = calculate_probability(&patterns, None, &config).unwrap();

        // Should be close to pattern success rate but smoothed towards 0.5
        assert!(result.probability > 0.5);
        assert!(result.probability < 0.8); // Smoothing pulls towards 0.5
        assert_eq!(result.patterns_used, 1);
    }

    #[test]
    fn test_calculate_probability_multiple_patterns() {
        let config = ProbabilityConfig::default();
        let patterns = vec![
            WeightedPattern::new(0.9, 0.9, 0.9),
            WeightedPattern::new(0.8, 0.8, 0.8),
            WeightedPattern::new(0.7, 0.7, 0.7),
        ];

        let result = calculate_probability(&patterns, None, &config).unwrap();

        // Should be weighted average of success rates
        assert!(result.probability > 0.6);
        assert!(result.probability < 0.9);
        assert_eq!(result.patterns_used, 3);
        assert!(result.confidence > 0.3); // More patterns = more confidence
    }

    #[test]
    fn test_calculate_probability_with_prior() {
        let config = ProbabilityConfig::default();
        let patterns = vec![WeightedPattern::new(0.8, 0.9, 0.9)];
        let prior = PriorPrediction::new(0.3, 0.8);

        let result_without_prior =
            calculate_probability(&patterns, None, &config).unwrap();
        let result_with_prior =
            calculate_probability(&patterns, Some(&prior), &config).unwrap();

        // Prior should pull probability towards its value
        assert!(result_with_prior.probability < result_without_prior.probability);
        assert!(result_with_prior.bayesian_updated);
    }

    #[test]
    fn test_calculate_probability_high_agreement() {
        let config = ProbabilityConfig::default();
        // All patterns agree on success rate
        let patterns = vec![
            WeightedPattern::new(0.8, 0.9, 0.9),
            WeightedPattern::new(0.8, 0.8, 0.8),
            WeightedPattern::new(0.8, 0.7, 0.7),
        ];

        let result = calculate_probability(&patterns, None, &config).unwrap();

        // High agreement should increase confidence
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_calculate_probability_low_agreement() {
        let config = ProbabilityConfig::default();
        // Patterns disagree significantly
        let patterns = vec![
            WeightedPattern::new(0.9, 0.9, 0.9),
            WeightedPattern::new(0.1, 0.8, 0.8),
            WeightedPattern::new(0.5, 0.7, 0.7),
        ];

        let result_low_agreement =
            calculate_probability(&patterns, None, &config).unwrap();

        // Create high agreement patterns
        let high_agreement_patterns = vec![
            WeightedPattern::new(0.5, 0.9, 0.9),
            WeightedPattern::new(0.5, 0.8, 0.8),
            WeightedPattern::new(0.5, 0.7, 0.7),
        ];

        let result_high_agreement =
            calculate_probability(&high_agreement_patterns, None, &config).unwrap();

        // Low agreement should result in lower confidence
        assert!(result_low_agreement.confidence < result_high_agreement.confidence);
    }

    #[test]
    fn test_smoothing() {
        let smoothed = apply_smoothing(0.9, 0.1);
        assert!(smoothed < 0.9); // Pulled towards 0.5
        assert!(smoothed > 0.8);

        let smoothed_low = apply_smoothing(0.1, 0.1);
        assert!(smoothed_low > 0.1); // Pulled towards 0.5
    }

    #[test]
    fn test_probability_interval() {
        let config = ProbabilityConfig::default();
        let patterns = vec![
            WeightedPattern::new(0.7, 0.9, 0.9),
            WeightedPattern::new(0.7, 0.8, 0.8),
        ];

        let result = calculate_probability(&patterns, None, &config).unwrap();

        // Interval should contain the probability
        assert!(result.lower_bound <= result.probability);
        assert!(result.upper_bound >= result.probability);
        assert!(result.lower_bound >= 0.0);
        assert!(result.upper_bound <= 1.0);
    }

    #[test]
    fn test_probability_calculator() {
        let calculator = ProbabilityCalculator::with_config(
            ProbabilityConfig::default().with_base_prior(0.6),
        );

        let patterns = vec![WeightedPattern::new(0.8, 0.9, 0.9)];
        let result = calculator.calculate(&patterns, None).unwrap();

        assert!(result.probability > 0.5);
    }

    #[test]
    fn test_agreement_calculation() {
        // Perfect agreement
        let perfect = vec![
            WeightedPattern::new(0.5, 0.9, 0.9),
            WeightedPattern::new(0.5, 0.9, 0.9),
        ];
        let perfect_agreement = calculate_agreement(&perfect);
        assert!((perfect_agreement - 1.0).abs() < 0.001);

        // Complete disagreement
        let complete_disagreement = vec![
            WeightedPattern::new(0.0, 0.9, 0.9),
            WeightedPattern::new(1.0, 0.9, 0.9),
        ];
        let zero_agreement = calculate_agreement(&complete_disagreement);
        assert!(zero_agreement < 0.1);
    }
}
