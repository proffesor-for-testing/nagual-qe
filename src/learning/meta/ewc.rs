//! Elastic Weight Consolidation (EWC++) Engine
//!
//! Prevents catastrophic forgetting by protecting important patterns
//! from being overwritten during learning updates.
//!
//! # Theory
//!
//! EWC adds a penalty to the loss function when updating parameters:
//! L_total = L_task + (λ/2) * Σ F_i * (θ_i - θ*_i)²
//!
//! Where:
//! - F_i is the Fisher information (importance) of parameter i
//! - θ*_i is the optimal value from previous tasks
//! - λ controls the strength of the penalty
//!
//! For patterns, we adapt this: the "parameter" is the pattern's reward,
//! and Fisher information is estimated from outcome variance.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use parking_lot::RwLock;
use tracing::{debug, info};

use super::types::{EwcConfig, MetaLearningStats, PatternImportance};

/// Elastic Weight Consolidation engine for preventing catastrophic forgetting
pub struct EwcEngine {
    /// Configuration
    config: EwcConfig,
    /// Importance cache (pattern_id -> importance)
    importance_cache: Arc<RwLock<HashMap<String, PatternImportance>>>,
    /// Statistics
    stats: Arc<RwLock<MetaLearningStats>>,
}

impl EwcEngine {
    /// Create a new EWC engine with the given configuration
    pub fn new(config: EwcConfig) -> Self {
        Self {
            config,
            importance_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(MetaLearningStats::default())),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(EwcConfig::default())
    }

    /// Get the current configuration
    pub fn config(&self) -> &EwcConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: EwcConfig) {
        self.config = config;
    }

    /// Calculate Fisher information for a pattern based on outcome history
    ///
    /// Fisher information measures how much information the outcomes carry
    /// about the pattern's effectiveness. Higher variance = more information.
    pub fn calculate_fisher_info(&self, outcomes: &[bool]) -> f64 {
        if outcomes.is_empty() {
            return 0.0;
        }

        let n = outcomes.len() as f64;
        let successes = outcomes.iter().filter(|&&o| o).count() as f64;
        let p = successes / n;

        // Fisher info for Bernoulli: 1 / (p * (1-p))
        // Bounded to avoid infinity near 0 and 1
        let fisher = if p > 0.05 && p < 0.95 {
            1.0 / (p * (1.0 - p))
        } else if p <= 0.05 || p >= 0.95 {
            // Near certainty = low information (we already know the answer)
            0.5
        } else {
            1.0
        };

        // Normalize to 0-1 range (Fisher can be up to 25 at p=0.5)
        (fisher / 25.0).min(1.0)
    }

    /// Calculate importance weight for a pattern
    ///
    /// Importance combines:
    /// - Fisher information (how informative the outcomes are)
    /// - Success rate (patterns that work are more important)
    /// - Usage frequency (frequently used patterns are more important)
    pub fn calculate_importance(
        &self,
        pattern_id: &str,
        success_count: u32,
        total_count: u32,
        outcomes: &[bool],
    ) -> f64 {
        if total_count == 0 {
            return 0.0;
        }

        let fisher = self.calculate_fisher_info(outcomes);
        let success_rate = success_count as f64 / total_count as f64;

        // Usage factor: more usage = more important (logarithmic scale)
        let usage_factor = (total_count as f64).ln().max(1.0) / 10.0;
        let usage_factor = usage_factor.min(1.0);

        // Combine factors
        // - High Fisher info = pattern outcomes are informative
        // - High success rate = pattern is valuable
        // - High usage = pattern is frequently needed
        let importance = (fisher * 0.3 + success_rate * 0.5 + usage_factor * 0.2).min(1.0);

        debug!(
            pattern_id = %pattern_id,
            fisher = fisher,
            success_rate = success_rate,
            usage_factor = usage_factor,
            importance = importance,
            "Calculated pattern importance"
        );

        importance
    }

    /// Update the importance cache for a pattern
    pub fn update_importance(
        &self,
        pattern_id: &str,
        success_count: u32,
        total_count: u32,
        outcomes: &[bool],
    ) -> PatternImportance {
        let importance = self.calculate_importance(pattern_id, success_count, total_count, outcomes);
        let fisher = self.calculate_fisher_info(outcomes);

        let mut cache = self.importance_cache.write();

        let entry = cache
            .entry(pattern_id.to_string())
            .or_insert_with(|| PatternImportance::new(pattern_id));

        entry.importance = importance;
        entry.fisher_info = fisher;
        entry.success_count = success_count;
        entry.total_count = total_count;
        entry.updated_at = Utc::now();

        // Clone entry before releasing the mutable borrow
        let result = entry.clone();

        // Update stats - now safe to iterate since entry borrow is dropped
        if importance >= self.config.importance_threshold {
            let mut stats = self.stats.write();
            stats.protected_patterns = cache
                .values()
                .filter(|p| p.importance >= self.config.importance_threshold)
                .count() as u32;
        }

        result
    }

    /// Get the importance for a pattern (from cache)
    pub fn get_importance(&self, pattern_id: &str) -> Option<PatternImportance> {
        self.importance_cache.read().get(pattern_id).cloned()
    }

    /// Calculate EWC penalty for updating a pattern's reward
    ///
    /// Returns a dampening factor (0.0-1.0):
    /// - 1.0 = no dampening, apply full update
    /// - 0.0 = full dampening, no update allowed
    pub fn ewc_penalty(&self, pattern_id: &str, proposed_change: f64) -> f64 {
        let cache = self.importance_cache.read();

        if let Some(importance) = cache.get(pattern_id) {
            if importance.importance >= self.config.importance_threshold {
                // EWC penalty formula: λ * F * (Δθ)²
                let penalty = self.config.lambda
                    * importance.fisher_info
                    * proposed_change.powi(2);

                // Convert to dampening factor
                let dampening = 1.0 / (1.0 + penalty);

                debug!(
                    pattern_id = %pattern_id,
                    importance = importance.importance,
                    fisher = importance.fisher_info,
                    proposed_change = proposed_change,
                    penalty = penalty,
                    dampening = dampening,
                    "Calculated EWC penalty"
                );

                if dampening < 0.5 {
                    // This would significantly dampen the update
                    let mut stats = self.stats.write();
                    stats.forgetting_prevented += 1;
                    info!(
                        pattern_id = %pattern_id,
                        dampening = dampening,
                        "EWC preventing potential catastrophic forgetting"
                    );
                }

                dampening
            } else {
                1.0 // Not important enough to protect
            }
        } else {
            1.0 // Unknown pattern, no protection
        }
    }

    /// Apply EWC-protected reward update
    ///
    /// Returns the new reward value after applying EWC dampening.
    pub fn protected_reward_update(
        &self,
        pattern_id: &str,
        current_reward: f64,
        proposed_new_reward: f64,
    ) -> f64 {
        let change = proposed_new_reward - current_reward;

        if change.abs() < 0.001 {
            return proposed_new_reward; // No significant change
        }

        let dampening = self.ewc_penalty(pattern_id, change);
        let actual_change = change * dampening;
        let new_reward = current_reward + actual_change;

        debug!(
            pattern_id = %pattern_id,
            current = current_reward,
            proposed = proposed_new_reward,
            actual = new_reward,
            dampening = dampening,
            "Applied EWC-protected reward update"
        );

        new_reward
    }

    /// Check if a pattern is protected (importance above threshold)
    pub fn is_protected(&self, pattern_id: &str) -> bool {
        self.importance_cache
            .read()
            .get(pattern_id)
            .map(|p| p.importance >= self.config.importance_threshold)
            .unwrap_or(false)
    }

    /// Get all protected patterns
    pub fn protected_patterns(&self) -> Vec<PatternImportance> {
        self.importance_cache
            .read()
            .values()
            .filter(|p| p.importance >= self.config.importance_threshold)
            .cloned()
            .collect()
    }

    /// Decay old Fisher information (call periodically)
    ///
    /// This allows patterns to become less protected over time if they're
    /// not being used, making room for new learning.
    pub fn decay_fisher_info(&self) {
        let mut cache = self.importance_cache.write();

        for importance in cache.values_mut() {
            importance.fisher_info *= self.config.fisher_decay;
            importance.importance *= self.config.fisher_decay;
        }

        debug!(
            decay_factor = self.config.fisher_decay,
            patterns_affected = cache.len(),
            "Applied Fisher information decay"
        );
    }

    /// Get statistics
    pub fn stats(&self) -> MetaLearningStats {
        self.stats.read().clone()
    }

    /// Clear the importance cache (use with caution)
    pub fn clear_cache(&self) {
        self.importance_cache.write().clear();
        info!("Cleared EWC importance cache");
    }

    /// Load importance data from storage
    pub fn load_importance(&self, data: Vec<PatternImportance>) {
        let mut cache = self.importance_cache.write();
        for importance in data {
            cache.insert(importance.pattern_id.clone(), importance);
        }
        info!(loaded = cache.len(), "Loaded pattern importance data");
    }

    /// Export importance data for persistence
    pub fn export_importance(&self) -> Vec<PatternImportance> {
        self.importance_cache.read().values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_info_calculation() {
        let engine = EwcEngine::default();

        // 50/50 outcomes = maximum Fisher info
        let outcomes_50_50: Vec<bool> = vec![true, false, true, false, true, false, true, false];
        let fisher_50 = engine.calculate_fisher_info(&outcomes_50_50);
        assert!(fisher_50 > 0.1, "50/50 should have high Fisher info");

        // All successes = low Fisher info (we already know)
        let outcomes_all_true: Vec<bool> = vec![true; 20];
        let fisher_all = engine.calculate_fisher_info(&outcomes_all_true);
        assert!(fisher_all < 0.1, "All true should have low Fisher info");

        // Empty outcomes
        let fisher_empty = engine.calculate_fisher_info(&[]);
        assert_eq!(fisher_empty, 0.0);
    }

    #[test]
    fn test_importance_calculation() {
        let engine = EwcEngine::default();

        // High usage, high success = high importance
        let importance_high = engine.calculate_importance(
            "test-1",
            90,
            100,
            &vec![true; 90].into_iter().chain(vec![false; 10]).collect::<Vec<_>>(),
        );
        assert!(importance_high > 0.5, "High success should be important");

        // Low usage = lower importance
        let importance_low = engine.calculate_importance("test-2", 1, 1, &[true]);
        assert!(importance_low < importance_high, "Low usage = less important");
    }

    #[test]
    fn test_ewc_penalty() {
        let engine = EwcEngine::new(EwcConfig {
            lambda: 1000.0,
            importance_threshold: 0.3,
            ..Default::default()
        });

        // Add an important pattern
        engine.update_importance(
            "important-pattern",
            80,
            100,
            &vec![true; 80].into_iter().chain(vec![false; 20]).collect::<Vec<_>>(),
        );

        // Large change to important pattern should be dampened
        let dampening = engine.ewc_penalty("important-pattern", 0.5);
        assert!(dampening < 1.0, "Should dampen large changes");
        assert!(dampening > 0.0, "Should not fully block");

        // Unknown pattern should not be dampened
        let dampening_unknown = engine.ewc_penalty("unknown-pattern", 0.5);
        assert_eq!(dampening_unknown, 1.0, "Unknown patterns not protected");
    }

    #[test]
    fn test_protected_reward_update() {
        let engine = EwcEngine::new(EwcConfig {
            lambda: 1000.0,
            importance_threshold: 0.3,
            ..Default::default()
        });

        // Add important pattern
        engine.update_importance(
            "protected",
            90,
            100,
            &vec![true; 90].into_iter().chain(vec![false; 10]).collect::<Vec<_>>(),
        );

        // Try to make a large change
        let new_reward = engine.protected_reward_update("protected", 0.8, 0.2);

        // Should be dampened (not drop all the way to 0.2)
        assert!(new_reward > 0.2, "Change should be dampened");
        assert!(new_reward < 0.8, "Some change should occur");
    }
}
