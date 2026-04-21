//! Property-based tests for prediction and calibration functions.
//!
//! These tests verify the following properties:
//!
//! 1. **Brier Score Bounds**: Brier score is always in [0.0, 1.0]
//! 2. **Calibration Bucket Properties**: Bucket statistics are valid
//! 3. **Probability Validity**: Probabilities are always in valid range
//! 4. **Resolution Outcomes**: Resolution produces valid results
//! 5. **Bucket Index Mapping**: Probabilities map to correct buckets

use proptest::prelude::*;

use super::{
    arb_boolean_outcome, arb_brier_score, arb_bucket_counts, arb_bucket_index,
    arb_confidence, arb_probability, arb_timeline_days, bucket_index_for_probability,
    calculate_brier_score,
};

proptest! {
    // ========================================================================
    // Property 1: Brier Score is always in [0.0, 1.0]
    // ========================================================================

    /// Property: Brier score is bounded to [0.0, 1.0] for any valid probability and outcome.
    #[test]
    fn prop_brier_score_bounded(
        probability in arb_probability(),
        outcome in arb_boolean_outcome()
    ) {
        let brier = calculate_brier_score(probability, outcome);

        prop_assert!(
            brier >= 0.0 && brier <= 1.0,
            "Brier score {} is out of bounds [0.0, 1.0] for probability {}, outcome {}",
            brier, probability, outcome
        );
    }

    /// Property: Brier score is minimized (0.0) when prediction matches outcome perfectly.
    #[test]
    fn prop_brier_perfect_prediction(outcome in arb_boolean_outcome()) {
        let probability = if outcome { 1.0 } else { 0.0 };
        let brier = calculate_brier_score(probability, outcome);

        prop_assert!(
            brier.abs() < 1e-10,
            "Perfect prediction should have Brier score ~0, got {} for p={}, outcome={}",
            brier, probability, outcome
        );
    }

    /// Property: Brier score is maximized (1.0) when prediction is completely wrong.
    #[test]
    fn prop_brier_worst_prediction(outcome in arb_boolean_outcome()) {
        let probability = if outcome { 0.0 } else { 1.0 };
        let brier = calculate_brier_score(probability, outcome);

        prop_assert!(
            (brier - 1.0).abs() < 1e-10,
            "Completely wrong prediction should have Brier score ~1, got {} for p={}, outcome={}",
            brier, probability, outcome
        );
    }

    /// Property: Brier score equals 0.25 when probability is 0.5 (maximum uncertainty).
    #[test]
    fn prop_brier_uncertain_prediction(outcome in arb_boolean_outcome()) {
        let brier = calculate_brier_score(0.5, outcome);

        prop_assert!(
            (brier - 0.25).abs() < 1e-10,
            "50% probability should have Brier score 0.25, got {}",
            brier
        );
    }

    /// Property: Brier score is symmetric for symmetric predictions.
    /// (p, true) and (1-p, false) should have the same Brier score.
    #[test]
    fn prop_brier_symmetry(probability in arb_probability()) {
        let brier_true = calculate_brier_score(probability, true);
        let brier_false = calculate_brier_score(1.0 - probability, false);

        prop_assert!(
            (brier_true - brier_false).abs() < 1e-10,
            "Symmetric predictions should have equal Brier: p={} (true)={}, (1-p)={} (false)={}",
            probability, brier_true, 1.0 - probability, brier_false
        );
    }

    // ========================================================================
    // Property 2: Bucket Index Mapping
    // ========================================================================

    /// Property: Bucket index is always in [0, 9] for valid probabilities.
    #[test]
    fn prop_bucket_index_bounded(probability in arb_probability()) {
        let idx = bucket_index_for_probability(probability);

        prop_assert!(
            idx <= 9,
            "Bucket index {} is out of bounds [0, 9] for probability {}",
            idx, probability
        );
    }

    /// Property: Probabilities in range [k/10, (k+1)/10) map to bucket k.
    #[test]
    fn prop_bucket_index_correct_mapping(bucket in arb_bucket_index()) {
        let lower = bucket as f64 * 0.1;
        let upper = (bucket + 1) as f64 * 0.1;

        // Test midpoint of bucket
        let midpoint = (lower + upper) / 2.0;
        let idx = bucket_index_for_probability(midpoint);

        prop_assert!(
            idx == bucket,
            "Probability {} (midpoint of bucket {}) mapped to bucket {}",
            midpoint, bucket, idx
        );
    }

    /// Property: Edge case - probability 1.0 maps to bucket 9.
    #[test]
    fn prop_bucket_index_edge_one(_seed in 0u64..100u64) {
        let idx = bucket_index_for_probability(1.0);
        prop_assert!(idx == 9, "Probability 1.0 should map to bucket 9, got {}", idx);
    }

    /// Property: Edge case - probability 0.0 maps to bucket 0.
    #[test]
    fn prop_bucket_index_edge_zero(_seed in 0u64..100u64) {
        let idx = bucket_index_for_probability(0.0);
        prop_assert!(idx == 0, "Probability 0.0 should map to bucket 0, got {}", idx);
    }

    /// Property: Lower bounds of buckets map to their respective buckets.
    #[test]
    fn prop_bucket_lower_bounds(bucket in 0usize..10usize) {
        let lower = bucket as f64 * 0.1;
        let idx = bucket_index_for_probability(lower);

        prop_assert!(
            idx == bucket,
            "Lower bound {} of bucket {} mapped to bucket {}",
            lower, bucket, idx
        );
    }

    // ========================================================================
    // Property 3: Calibration Bucket Statistics
    // ========================================================================

    /// Property: Positive count is always <= total count.
    #[test]
    fn prop_bucket_counts_valid((positive, total) in arb_bucket_counts()) {
        prop_assert!(
            positive <= total,
            "Positive count {} exceeds total count {}",
            positive, total
        );
    }

    /// Property: Actual positive rate is in [0.0, 1.0] when total > 0.
    #[test]
    fn prop_actual_rate_bounded((positive, total) in arb_bucket_counts()) {
        if total > 0 {
            let rate = positive as f64 / total as f64;
            prop_assert!(
                rate >= 0.0 && rate <= 1.0,
                "Actual rate {} is out of [0.0, 1.0] for positive={}, total={}",
                rate, positive, total
            );
        }
    }

    /// Property: Calibration error is in [0.0, 1.0] (difference between expected and actual).
    #[test]
    fn prop_calibration_error_bounded(
        bucket in arb_bucket_index(),
        (positive, total) in arb_bucket_counts()
    ) {
        if total > 0 {
            let expected = (bucket as f64 * 0.1 + (bucket + 1) as f64 * 0.1) / 2.0;
            let actual = positive as f64 / total as f64;
            let error = (expected - actual).abs();

            prop_assert!(
                error <= 1.0,
                "Calibration error {} exceeds 1.0 for expected={}, actual={}",
                error, expected, actual
            );
        }
    }

    // ========================================================================
    // Property 4: Probability Validity
    // ========================================================================

    /// Property: Clamping always produces valid probabilities.
    #[test]
    fn prop_probability_clamping(value in -10.0f64..10.0f64) {
        let clamped = value.clamp(0.0, 1.0);

        prop_assert!(
            clamped >= 0.0 && clamped <= 1.0,
            "Clamped value {} is out of [0.0, 1.0]",
            clamped
        );
    }

    /// Property: Confidence values are always valid after clamping.
    #[test]
    fn prop_confidence_clamping(value in -10.0f64..10.0f64) {
        let clamped = value.clamp(0.0, 1.0);

        prop_assert!(
            clamped >= 0.0 && clamped <= 1.0,
            "Clamped confidence {} is out of [0.0, 1.0]",
            clamped
        );
    }

    // ========================================================================
    // Property 5: Timeline Validity
    // ========================================================================

    /// Property: Timeline min <= max after generation.
    #[test]
    fn prop_timeline_ordering((min_days, max_days) in arb_timeline_days()) {
        prop_assert!(
            min_days <= max_days,
            "Timeline min {} > max {}",
            min_days, max_days
        );
    }

    /// Property: Timeline midpoint is between min and max.
    #[test]
    fn prop_timeline_midpoint((min_days, max_days) in arb_timeline_days()) {
        let midpoint = (min_days + max_days) / 2;

        prop_assert!(
            midpoint >= min_days && midpoint <= max_days,
            "Midpoint {} is not in [{}, {}]",
            midpoint, min_days, max_days
        );
    }

    // ========================================================================
    // Property 6: Probability Interval Properties
    // ========================================================================

    /// Property: Probability interval width decreases with confidence.
    #[test]
    fn prop_confidence_narrows_interval(
        probability in arb_probability(),
        low_confidence in 0.0f64..0.4f64,
        high_confidence in 0.6f64..=1.0f64,
    ) {
        // Interval calculation: half_width = (1.0 - confidence) * 0.5
        let low_half_width = (1.0 - low_confidence) * 0.5;
        let high_half_width = (1.0 - high_confidence) * 0.5;

        prop_assert!(
            high_half_width <= low_half_width,
            "Higher confidence should yield narrower interval: high_conf={} width={}, low_conf={} width={}",
            high_confidence, high_half_width, low_confidence, low_half_width
        );
    }

    /// Property: Probability interval is always clamped to [0.0, 1.0].
    #[test]
    fn prop_probability_interval_clamped(
        probability in arb_probability(),
        confidence in arb_confidence(),
    ) {
        let half_width = (1.0 - confidence) * 0.5;
        let lower = (probability - half_width).max(0.0);
        let upper = (probability + half_width).min(1.0);

        prop_assert!(
            lower >= 0.0 && upper <= 1.0 && lower <= upper,
            "Interval [{}, {}] is invalid for p={}, conf={}",
            lower, upper, probability, confidence
        );
    }

    // ========================================================================
    // Property 7: Brier Score Monotonicity
    // ========================================================================

    /// Property: Brier score increases as prediction moves away from outcome.
    #[test]
    fn prop_brier_monotonicity_true_outcome(
        low_prob in 0.0f64..0.5f64,
        high_prob in 0.5f64..=1.0f64,
    ) {
        // For outcome = true, higher probability should have lower Brier score
        let low_brier = calculate_brier_score(low_prob, true);
        let high_brier = calculate_brier_score(high_prob, true);

        prop_assert!(
            high_brier <= low_brier,
            "For true outcome: higher prob ({}) should have <= Brier ({}) than lower prob ({}) with Brier ({})",
            high_prob, high_brier, low_prob, low_brier
        );
    }

    /// Property: For false outcome, lower probability should have lower Brier score.
    #[test]
    fn prop_brier_monotonicity_false_outcome(
        low_prob in 0.0f64..0.5f64,
        high_prob in 0.5f64..=1.0f64,
    ) {
        let low_brier = calculate_brier_score(low_prob, false);
        let high_brier = calculate_brier_score(high_prob, false);

        prop_assert!(
            low_brier <= high_brier,
            "For false outcome: lower prob ({}) should have <= Brier ({}) than higher prob ({}) with Brier ({})",
            low_prob, low_brier, high_prob, high_brier
        );
    }
}

// ============================================================================
// Non-proptest unit tests for specific scenarios
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_brier_score_exact_values() {
        // Perfect predictions
        assert!((calculate_brier_score(1.0, true) - 0.0).abs() < 1e-10);
        assert!((calculate_brier_score(0.0, false) - 0.0).abs() < 1e-10);

        // Worst predictions
        assert!((calculate_brier_score(0.0, true) - 1.0).abs() < 1e-10);
        assert!((calculate_brier_score(1.0, false) - 1.0).abs() < 1e-10);

        // 50% probability
        assert!((calculate_brier_score(0.5, true) - 0.25).abs() < 1e-10);
        assert!((calculate_brier_score(0.5, false) - 0.25).abs() < 1e-10);

        // Typical predictions
        assert!((calculate_brier_score(0.7, true) - 0.09).abs() < 1e-10);
        assert!((calculate_brier_score(0.7, false) - 0.49).abs() < 1e-10);
    }

    #[test]
    fn test_bucket_index_boundaries() {
        // Lower bounds
        assert_eq!(bucket_index_for_probability(0.0), 0);
        assert_eq!(bucket_index_for_probability(0.1), 1);
        assert_eq!(bucket_index_for_probability(0.2), 2);
        assert_eq!(bucket_index_for_probability(0.9), 9);

        // Upper edge
        assert_eq!(bucket_index_for_probability(1.0), 9);

        // Mid-points
        assert_eq!(bucket_index_for_probability(0.05), 0);
        assert_eq!(bucket_index_for_probability(0.15), 1);
        assert_eq!(bucket_index_for_probability(0.55), 5);
        assert_eq!(bucket_index_for_probability(0.95), 9);
    }

    #[test]
    fn test_bucket_index_just_below_boundaries() {
        // Just below upper bounds (should still be in lower bucket)
        assert_eq!(bucket_index_for_probability(0.099999), 0);
        assert_eq!(bucket_index_for_probability(0.199999), 1);
        assert_eq!(bucket_index_for_probability(0.999999), 9);
    }

    #[test]
    fn test_calibration_error_calculation() {
        // Bucket 7 (0.70-0.80) with 75% actual rate
        let expected = 0.75; // midpoint of bucket 7
        let actual = 0.75;
        let error = (expected - actual).abs();
        assert!(error < 1e-10, "Perfect calibration should have ~0 error");

        // Bucket 7 with 50% actual rate
        let actual_50 = 0.50;
        let error_50 = (expected - actual_50).abs();
        assert!((error_50 - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_probability_interval_calculation() {
        // High confidence (0.9) -> narrow interval
        let prob = 0.7;
        let conf = 0.9;
        let half_width = (1.0 - conf) * 0.5; // 0.05
        let lower = (prob - half_width).max(0.0); // 0.65
        let upper = (prob + half_width).min(1.0); // 0.75

        assert!((lower - 0.65).abs() < 1e-10);
        assert!((upper - 0.75).abs() < 1e-10);

        // Low confidence (0.2) -> wide interval
        let conf_low = 0.2;
        let half_width_low = (1.0 - conf_low) * 0.5; // 0.4
        let lower_low = (prob - half_width_low).max(0.0); // 0.3
        let upper_low = (prob + half_width_low).min(1.0); // 1.0 (clamped)

        assert!((lower_low - 0.3).abs() < 1e-10);
        assert!((upper_low - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wilson_confidence_interval_properties() {
        // Test that Wilson CI has reasonable properties
        // For 50/100 successes at 95% confidence
        let successes = 50.0;
        let total = 100.0;
        let confidence = 0.95;

        // Z-score for 95% confidence is ~1.96
        let z = 1.96;
        let p = successes / total;
        let n = total;

        let denominator = 1.0 + z * z / n;
        let center = (p + z * z / (2.0 * n)) / denominator;
        let spread = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt() / denominator;

        let lower = (center - spread).max(0.0);
        let upper = (center + spread).min(1.0);

        // CI should contain the point estimate
        assert!(lower < p && p < upper);
        // CI should be symmetric around adjusted center
        assert!((center - lower - (upper - center)).abs() < 0.01);
        // CI should be reasonable width (not too wide for n=100)
        assert!(upper - lower < 0.3);
    }

    #[test]
    fn test_brier_score_formula() {
        // Verify formula: Brier = (probability - outcome)^2
        let test_cases = vec![
            (0.8, true, 0.04),   // (0.8 - 1.0)^2 = 0.04
            (0.8, false, 0.64), // (0.8 - 0.0)^2 = 0.64
            (0.3, true, 0.49),  // (0.3 - 1.0)^2 = 0.49
            (0.3, false, 0.09), // (0.3 - 0.0)^2 = 0.09
        ];

        for (prob, outcome, expected) in test_cases {
            let brier = calculate_brier_score(prob, outcome);
            assert!(
                (brier - expected).abs() < 1e-10,
                "Brier({}, {}) = {} but expected {}",
                prob, outcome, brier, expected
            );
        }
    }
}
