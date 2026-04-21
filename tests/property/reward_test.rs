//! Property-based tests for the `calculate_reward()` function.
//!
//! These tests verify the following properties:
//!
//! 1. **Bounded Output**: Reward is always in [0.0, 1.0]
//! 2. **Outcome Ordering**: Success > PartialSuccess > Neutral > Failure (base rewards)
//! 3. **Modifier Bounds**: Combined modifier is always in [0.5, 1.5]
//! 4. **No Panics**: Function never panics on any valid input
//! 5. **Determinism**: Same inputs always produce same outputs

use proptest::prelude::*;

use super::{
    arb_outcome, arb_probability_f32, arb_reward_modifiers, calculate_reward,
    ArbitraryRewardModifiers, Outcome, u8_to_outcome,
};

proptest! {
    // ========================================================================
    // Property 1: Reward is always in [0.0, 1.0]
    // ========================================================================

    /// Property: For any outcome and any modifiers, the reward is bounded to [0.0, 1.0].
    ///
    /// This is critical because rewards are used as quality signals and must be
    /// within the valid probability range.
    #[test]
    fn prop_reward_always_bounded(
        outcome_idx in arb_outcome(),
        modifiers in arb_reward_modifiers()
    ) {
        let outcome = u8_to_outcome(outcome_idx);
        let reward = calculate_reward(outcome, Some(&modifiers));

        prop_assert!(
            reward >= 0.0 && reward <= 1.0,
            "Reward {} is out of bounds [0.0, 1.0] for outcome {:?} with modifiers {:?}",
            reward, outcome, modifiers
        );
    }

    /// Property: Reward without modifiers is also bounded.
    #[test]
    fn prop_reward_bounded_without_modifiers(outcome_idx in arb_outcome()) {
        let outcome = u8_to_outcome(outcome_idx);
        let reward = calculate_reward(outcome, None);

        prop_assert!(
            reward >= 0.0 && reward <= 1.0,
            "Reward {} is out of bounds for outcome {:?} without modifiers",
            reward, outcome
        );
    }

    // ========================================================================
    // Property 2: Success > Neutral > Failure (base ordering)
    // ========================================================================

    /// Property: Without modifiers, Success always yields higher reward than Failure.
    #[test]
    fn prop_success_greater_than_failure(_seed in 0u64..1000u64) {
        let success_reward = calculate_reward(Outcome::Success, None);
        let failure_reward = calculate_reward(Outcome::Failure, None);

        prop_assert!(
            success_reward > failure_reward,
            "Success reward ({}) should be > Failure reward ({})",
            success_reward, failure_reward
        );
    }

    /// Property: Without modifiers, PartialSuccess yields higher reward than Neutral.
    #[test]
    fn prop_partial_success_greater_than_neutral(_seed in 0u64..1000u64) {
        let partial_reward = calculate_reward(Outcome::PartialSuccess, None);
        let neutral_reward = calculate_reward(Outcome::Neutral, None);

        prop_assert!(
            partial_reward > neutral_reward,
            "PartialSuccess reward ({}) should be > Neutral reward ({})",
            partial_reward, neutral_reward
        );
    }

    /// Property: Without modifiers, Neutral yields higher reward than Failure.
    #[test]
    fn prop_neutral_greater_than_failure(_seed in 0u64..1000u64) {
        let neutral_reward = calculate_reward(Outcome::Neutral, None);
        let failure_reward = calculate_reward(Outcome::Failure, None);

        prop_assert!(
            neutral_reward > failure_reward,
            "Neutral reward ({}) should be > Failure reward ({})",
            neutral_reward, failure_reward
        );
    }

    /// Property: Success > PartialSuccess > Neutral > Failure (complete ordering)
    #[test]
    fn prop_outcome_ordering(_seed in 0u64..1000u64) {
        let success = calculate_reward(Outcome::Success, None);
        let partial = calculate_reward(Outcome::PartialSuccess, None);
        let neutral = calculate_reward(Outcome::Neutral, None);
        let failure = calculate_reward(Outcome::Failure, None);

        prop_assert!(
            success > partial && partial > neutral && neutral > failure,
            "Ordering violated: Success({}) > PartialSuccess({}) > Neutral({}) > Failure({})",
            success, partial, neutral, failure
        );
    }

    // ========================================================================
    // Property 3: Modifier bounds are respected
    // ========================================================================

    /// Property: Combined modifier is always in [0.5, 1.5].
    #[test]
    fn prop_modifier_bounded(modifiers in arb_reward_modifiers()) {
        let combined = modifiers.combined_modifier();

        prop_assert!(
            combined >= 0.5 && combined <= 1.5,
            "Combined modifier {} is out of bounds [0.5, 1.5] for {:?}",
            combined, modifiers
        );
    }

    /// Property: High confidence with high context relevance gives modifier >= 1.0.
    #[test]
    fn prop_high_modifiers_boost(
        speed in proptest::option::of(0.8f32..=1.0f32),
        satisfaction in proptest::option::of(0.8f32..=1.0f32),
    ) {
        let modifiers = ArbitraryRewardModifiers {
            confidence: 0.95,
            context_relevance: 0.95,
            speed_factor: speed,
            user_satisfaction: satisfaction,
            verified: true,
        };

        let combined = modifiers.combined_modifier();

        prop_assert!(
            combined >= 1.0,
            "High modifiers should yield combined >= 1.0, got {}",
            combined
        );
    }

    /// Property: Low confidence with low context relevance gives modifier <= 1.0.
    #[test]
    fn prop_low_modifiers_reduce(
        speed in proptest::option::of(0.0f32..=0.2f32),
        satisfaction in proptest::option::of(0.0f32..=0.2f32),
    ) {
        let modifiers = ArbitraryRewardModifiers {
            confidence: 0.1,
            context_relevance: 0.1,
            speed_factor: speed,
            user_satisfaction: satisfaction,
            verified: false,
        };

        let combined = modifiers.combined_modifier();

        prop_assert!(
            combined <= 1.0,
            "Low modifiers should yield combined <= 1.0, got {}",
            combined
        );
    }

    // ========================================================================
    // Property 4: No panics on arbitrary inputs
    // ========================================================================

    /// Property: calculate_reward never panics with arbitrary valid inputs.
    #[test]
    fn prop_no_panic_with_arbitrary_modifiers(
        outcome_idx in arb_outcome(),
        confidence in arb_probability_f32(),
        context_relevance in arb_probability_f32(),
        speed_factor in proptest::option::of(arb_probability_f32()),
        user_satisfaction in proptest::option::of(arb_probability_f32()),
        verified in any::<bool>(),
    ) {
        let outcome = u8_to_outcome(outcome_idx);
        let modifiers = ArbitraryRewardModifiers {
            confidence,
            context_relevance,
            speed_factor,
            user_satisfaction,
            verified,
        };

        // Should not panic
        let _ = calculate_reward(outcome, Some(&modifiers));
    }

    /// Property: Function handles edge case values correctly.
    #[test]
    fn prop_edge_values_handled(
        outcome_idx in arb_outcome(),
        // Test with extreme boundary values
        confidence in prop_oneof![Just(0.0f32), Just(1.0f32), Just(0.5f32)],
        context_relevance in prop_oneof![Just(0.0f32), Just(1.0f32), Just(0.5f32)],
    ) {
        let outcome = u8_to_outcome(outcome_idx);
        let modifiers = ArbitraryRewardModifiers {
            confidence,
            context_relevance,
            speed_factor: Some(confidence), // reuse for variety
            user_satisfaction: Some(context_relevance),
            verified: confidence > 0.5,
        };

        let reward = calculate_reward(outcome, Some(&modifiers));

        prop_assert!(
            reward >= 0.0 && reward <= 1.0,
            "Edge case values produced out-of-bounds reward: {}",
            reward
        );
    }

    // ========================================================================
    // Property 5: Determinism
    // ========================================================================

    /// Property: Same inputs always produce the same outputs.
    #[test]
    fn prop_deterministic(
        outcome_idx in arb_outcome(),
        modifiers in arb_reward_modifiers(),
    ) {
        let outcome = u8_to_outcome(outcome_idx);

        let reward1 = calculate_reward(outcome, Some(&modifiers));
        let reward2 = calculate_reward(outcome, Some(&modifiers));

        prop_assert!(
            (reward1 - reward2).abs() < f32::EPSILON,
            "Non-deterministic: reward1={}, reward2={}",
            reward1, reward2
        );
    }

    // ========================================================================
    // Property 6: Monotonicity of individual modifiers
    // ========================================================================

    /// Property: Higher confidence (all else equal) yields higher or equal modifier.
    #[test]
    fn prop_confidence_monotonic(
        low_confidence in 0.0f32..0.5f32,
        high_confidence in 0.5f32..=1.0f32,
        context_relevance in arb_probability_f32(),
    ) {
        let low_mods = ArbitraryRewardModifiers {
            confidence: low_confidence,
            context_relevance,
            speed_factor: None,
            user_satisfaction: None,
            verified: false,
        };

        let high_mods = ArbitraryRewardModifiers {
            confidence: high_confidence,
            context_relevance,
            speed_factor: None,
            user_satisfaction: None,
            verified: false,
        };

        let low_combined = low_mods.combined_modifier();
        let high_combined = high_mods.combined_modifier();

        prop_assert!(
            high_combined >= low_combined,
            "Higher confidence ({}) should yield >= modifier than lower ({}): {} vs {}",
            high_confidence, low_confidence, high_combined, low_combined
        );
    }

    /// Property: Higher context relevance (all else equal) yields higher or equal modifier.
    #[test]
    fn prop_context_relevance_monotonic(
        confidence in arb_probability_f32(),
        low_relevance in 0.0f32..0.5f32,
        high_relevance in 0.5f32..=1.0f32,
    ) {
        let low_mods = ArbitraryRewardModifiers {
            confidence,
            context_relevance: low_relevance,
            speed_factor: None,
            user_satisfaction: None,
            verified: false,
        };

        let high_mods = ArbitraryRewardModifiers {
            confidence,
            context_relevance: high_relevance,
            speed_factor: None,
            user_satisfaction: None,
            verified: false,
        };

        let low_combined = low_mods.combined_modifier();
        let high_combined = high_mods.combined_modifier();

        prop_assert!(
            high_combined >= low_combined,
            "Higher relevance ({}) should yield >= modifier than lower ({}): {} vs {}",
            high_relevance, low_relevance, high_combined, low_combined
        );
    }

    /// Property: Verified flag always provides a bonus (or no change).
    #[test]
    fn prop_verified_bonus(
        confidence in arb_probability_f32(),
        context_relevance in arb_probability_f32(),
    ) {
        let unverified = ArbitraryRewardModifiers {
            confidence,
            context_relevance,
            speed_factor: None,
            user_satisfaction: None,
            verified: false,
        };

        let verified = ArbitraryRewardModifiers {
            confidence,
            context_relevance,
            speed_factor: None,
            user_satisfaction: None,
            verified: true,
        };

        let unverified_combined = unverified.combined_modifier();
        let verified_combined = verified.combined_modifier();

        prop_assert!(
            verified_combined >= unverified_combined,
            "Verified should yield >= modifier than unverified: {} vs {}",
            verified_combined, unverified_combined
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
    fn test_base_rewards_exact_values() {
        assert_eq!(Outcome::Success.base_reward(), 0.9);
        assert_eq!(Outcome::PartialSuccess.base_reward(), 0.7);
        assert_eq!(Outcome::Neutral.base_reward(), 0.5);
        assert_eq!(Outcome::Failure.base_reward(), 0.2);
    }

    #[test]
    fn test_calculate_reward_without_modifiers() {
        assert_eq!(calculate_reward(Outcome::Success, None), 0.9);
        assert_eq!(calculate_reward(Outcome::PartialSuccess, None), 0.7);
        assert_eq!(calculate_reward(Outcome::Neutral, None), 0.5);
        assert_eq!(calculate_reward(Outcome::Failure, None), 0.2);
    }

    #[test]
    fn test_max_modifiers_clamped() {
        // Even with maximum positive modifiers, reward should not exceed 1.0
        let max_mods = ArbitraryRewardModifiers {
            confidence: 1.0,
            context_relevance: 1.0,
            speed_factor: Some(1.0),
            user_satisfaction: Some(1.0),
            verified: true,
        };

        let reward = calculate_reward(Outcome::Success, Some(&max_mods));
        assert!(reward <= 1.0, "Reward {} exceeds 1.0", reward);
    }

    #[test]
    fn test_min_modifiers_clamped() {
        // Even with minimum modifiers, reward should not go below 0.0
        let min_mods = ArbitraryRewardModifiers {
            confidence: 0.0,
            context_relevance: 0.0,
            speed_factor: Some(0.0),
            user_satisfaction: Some(0.0),
            verified: false,
        };

        let reward = calculate_reward(Outcome::Failure, Some(&min_mods));
        assert!(reward >= 0.0, "Reward {} is below 0.0", reward);
    }

    #[test]
    fn test_neutral_modifiers() {
        // With neutral/default-like modifiers, reward should be close to base
        let neutral_mods = ArbitraryRewardModifiers {
            confidence: 0.8, // Default confidence
            context_relevance: 1.0, // Default context relevance
            speed_factor: None,
            user_satisfaction: None,
            verified: false,
        };

        let reward = calculate_reward(Outcome::Neutral, Some(&neutral_mods));
        // Should be close to 0.5 (Neutral base reward)
        assert!((reward - 0.5).abs() < 0.2, "Reward {} too far from base 0.5", reward);
    }

    #[test]
    fn test_u8_to_outcome_mapping() {
        assert_eq!(u8_to_outcome(0), Outcome::Success);
        assert_eq!(u8_to_outcome(1), Outcome::PartialSuccess);
        assert_eq!(u8_to_outcome(2), Outcome::Neutral);
        assert_eq!(u8_to_outcome(3), Outcome::Failure);
        // Out of range defaults to Failure
        assert_eq!(u8_to_outcome(4), Outcome::Failure);
        assert_eq!(u8_to_outcome(255), Outcome::Failure);
    }

    #[test]
    fn test_modifier_combined_range() {
        // Test various combinations ensure combined is in [0.5, 1.5]
        let test_cases = vec![
            (0.0, 0.0, None, None, false),
            (1.0, 1.0, Some(1.0), Some(1.0), true),
            (0.5, 0.5, Some(0.5), Some(0.5), false),
            (0.0, 1.0, None, Some(0.0), true),
            (1.0, 0.0, Some(0.0), None, false),
        ];

        for (conf, ctx, speed, sat, verified) in test_cases {
            let mods = ArbitraryRewardModifiers {
                confidence: conf,
                context_relevance: ctx,
                speed_factor: speed,
                user_satisfaction: sat,
                verified,
            };

            let combined = mods.combined_modifier();
            assert!(
                combined >= 0.5 && combined <= 1.5,
                "Combined modifier {} out of [0.5, 1.5] for conf={}, ctx={}, speed={:?}, sat={:?}, verified={}",
                combined, conf, ctx, speed, sat, verified
            );
        }
    }
}
