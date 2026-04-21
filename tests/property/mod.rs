//! Property-based tests for Nagual.
//!
//! This module provides property-based testing using proptest to verify
//! invariants and properties of core functions across a wide range of inputs.
//!
//! # Test Modules
//!
//! - `reward_test`: Properties for `calculate_reward()` function
//! - `prediction_test`: Properties for prediction/calibration functions
//! - `attention_test`: Properties for attention mechanisms and embeddings
//!
//! # Custom Strategies
//!
//! Custom proptest strategies are provided for domain types:
//! - `arb_outcome()`: Generate arbitrary Outcome values
//! - `arb_reward_modifiers()`: Generate RewardModifiers with valid ranges
//! - `arb_embedding()`: Generate embedding vectors of specified dimension
//! - `arb_probability()`: Generate valid probability values [0.0, 1.0]

pub mod attention_test;
pub mod prediction_test;
pub mod reward_test;

use proptest::prelude::*;

// ============================================================================
// Custom Strategies for Domain Types
// ============================================================================

/// Strategy for generating arbitrary Outcome values.
pub fn arb_outcome() -> impl Strategy<Value = u8> {
    // 0 = Success, 1 = PartialSuccess, 2 = Neutral, 3 = Failure
    0u8..4u8
}

/// Convert a u8 to an Outcome (used with arb_outcome).
pub fn u8_to_outcome(v: u8) -> Outcome {
    match v {
        0 => Outcome::Success,
        1 => Outcome::PartialSuccess,
        2 => Outcome::Neutral,
        _ => Outcome::Failure,
    }
}

/// Outcome enum mirrored for tests (to avoid importing from main crate in tests).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Outcome {
    Success,
    PartialSuccess,
    Neutral,
    Failure,
}

impl Outcome {
    pub fn base_reward(&self) -> f32 {
        match self {
            Outcome::Success => 0.9,
            Outcome::PartialSuccess => 0.7,
            Outcome::Neutral => 0.5,
            Outcome::Failure => 0.2,
        }
    }
}

/// Strategy for generating reward modifier components.
///
/// All values are clamped to [0.0, 1.0] as per the domain constraints.
pub fn arb_modifier_value() -> impl Strategy<Value = f32> {
    0.0f32..=1.0f32
}

/// Strategy for generating optional modifier values.
pub fn arb_optional_modifier() -> impl Strategy<Value = Option<f32>> {
    prop_oneof![
        Just(None),
        (0.0f32..=1.0f32).prop_map(Some),
    ]
}

/// Strategy for generating complete RewardModifiers.
#[derive(Debug, Clone)]
pub struct ArbitraryRewardModifiers {
    pub confidence: f32,
    pub context_relevance: f32,
    pub speed_factor: Option<f32>,
    pub user_satisfaction: Option<f32>,
    pub verified: bool,
}

impl ArbitraryRewardModifiers {
    /// Calculate the combined modifier effect.
    pub fn combined_modifier(&self) -> f32 {
        let mut modifier = 1.0;

        // Confidence affects how much we trust this outcome
        modifier *= 0.7 + (self.confidence * 0.4);

        // Context relevance affects how applicable this outcome is
        modifier *= 0.8 + (self.context_relevance * 0.2);

        // Speed factor gives a small bonus/penalty
        if let Some(speed) = self.speed_factor {
            modifier *= 0.95 + (speed * 0.1);
        }

        // User satisfaction is highly weighted if provided
        if let Some(satisfaction) = self.user_satisfaction {
            modifier *= 0.8 + (satisfaction * 0.4);
        }

        // Verification bonus
        if self.verified {
            modifier *= 1.05;
        }

        // Clamp to reasonable range
        modifier.clamp(0.5, 1.5)
    }
}

pub fn arb_reward_modifiers() -> impl Strategy<Value = ArbitraryRewardModifiers> {
    (
        arb_modifier_value(),      // confidence
        arb_modifier_value(),      // context_relevance
        arb_optional_modifier(),   // speed_factor
        arb_optional_modifier(),   // user_satisfaction
        any::<bool>(),             // verified
    )
        .prop_map(|(confidence, context_relevance, speed_factor, user_satisfaction, verified)| {
            ArbitraryRewardModifiers {
                confidence,
                context_relevance,
                speed_factor,
                user_satisfaction,
                verified,
            }
        })
}

/// Strategy for generating probability values in [0.0, 1.0].
pub fn arb_probability() -> impl Strategy<Value = f64> {
    0.0f64..=1.0f64
}

/// Strategy for generating f32 probability values in [0.0, 1.0].
pub fn arb_probability_f32() -> impl Strategy<Value = f32> {
    0.0f32..=1.0f32
}

/// Strategy for generating embedding vectors of a specific dimension.
///
/// By default, generates vectors with values in [-1.0, 1.0].
pub fn arb_embedding(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-1.0f32..=1.0f32, dim)
}

/// Strategy for generating non-zero embedding vectors.
///
/// Ensures at least one component is non-zero to avoid division by zero.
pub fn arb_nonzero_embedding(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-1.0f32..=1.0f32, dim).prop_filter(
        "vector must be non-zero",
        |v| v.iter().any(|&x| x.abs() > f32::EPSILON),
    )
}

/// Strategy for generating normalized embedding vectors (L2 norm = 1).
pub fn arb_normalized_embedding(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    arb_nonzero_embedding(dim).prop_map(|v| {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            v.iter().map(|x| x / norm).collect()
        } else {
            // Fallback to unit vector in first dimension
            let mut result = vec![0.0; dim];
            if !result.is_empty() {
                result[0] = 1.0;
            }
            result
        }
    })
}

/// Strategy for generating positive integers for bucket indices.
pub fn arb_bucket_index() -> impl Strategy<Value = usize> {
    0usize..10usize
}

/// Strategy for generating boolean outcomes for predictions.
pub fn arb_boolean_outcome() -> impl Strategy<Value = bool> {
    any::<bool>()
}

/// Strategy for generating calibration bucket counts.
pub fn arb_bucket_counts() -> impl Strategy<Value = (u32, u32)> {
    (0u32..1000u32).prop_flat_map(|total| {
        (Just(total), 0u32..=total)
    }).prop_map(|(total, positive)| (positive, total))
}

/// Strategy for generating Brier scores (which are always in [0.0, 1.0]).
pub fn arb_brier_score() -> impl Strategy<Value = f64> {
    0.0f64..=1.0f64
}

/// Strategy for generating timeline days.
pub fn arb_timeline_days() -> impl Strategy<Value = (u32, u32)> {
    (1u32..365u32).prop_flat_map(|min_days| {
        (Just(min_days), min_days..=365u32)
    })
}

/// Strategy for generating confidence values.
pub fn arb_confidence() -> impl Strategy<Value = f64> {
    0.0f64..=1.0f64
}

// ============================================================================
// Helper Functions for Testing
// ============================================================================

/// Calculate L2 norm of a vector.
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Check if a vector is normalized (L2 norm approximately 1.0).
pub fn is_normalized(v: &[f32], tolerance: f32) -> bool {
    let norm = l2_norm(v);
    (norm - 1.0).abs() < tolerance
}

/// Normalize a vector using L2 normalization.
pub fn normalize_l2(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm > f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Calculate cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    if norm_a > f32::EPSILON && norm_b > f32::EPSILON {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Calculate Brier score for a prediction.
pub fn calculate_brier_score(probability: f64, outcome: bool) -> f64 {
    let outcome_val = if outcome { 1.0 } else { 0.0 };
    (probability - outcome_val).powi(2)
}

/// Calculate reward given outcome and modifiers.
pub fn calculate_reward(outcome: Outcome, modifiers: Option<&ArbitraryRewardModifiers>) -> f32 {
    let base = outcome.base_reward();
    let modifier = modifiers.map(|m| m.combined_modifier()).unwrap_or(1.0);
    (base * modifier).clamp(0.0, 1.0)
}

/// Get the bucket index for a probability value.
pub fn bucket_index_for_probability(probability: f64) -> usize {
    let idx = (probability * 10.0).floor() as usize;
    idx.min(9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outcome_base_rewards() {
        assert_eq!(Outcome::Success.base_reward(), 0.9);
        assert_eq!(Outcome::PartialSuccess.base_reward(), 0.7);
        assert_eq!(Outcome::Neutral.base_reward(), 0.5);
        assert_eq!(Outcome::Failure.base_reward(), 0.2);
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        assert!((l2_norm(&v) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_l2() {
        let v = vec![3.0, 4.0];
        let normalized = normalize_l2(&v);
        assert!((l2_norm(&normalized) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_brier_score_perfect() {
        assert!((calculate_brier_score(1.0, true) - 0.0).abs() < 1e-10);
        assert!((calculate_brier_score(0.0, false) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_brier_score_worst() {
        assert!((calculate_brier_score(1.0, false) - 1.0).abs() < 1e-10);
        assert!((calculate_brier_score(0.0, true) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bucket_index() {
        assert_eq!(bucket_index_for_probability(0.0), 0);
        assert_eq!(bucket_index_for_probability(0.5), 5);
        assert_eq!(bucket_index_for_probability(0.99), 9);
        assert_eq!(bucket_index_for_probability(1.0), 9);
    }
}
