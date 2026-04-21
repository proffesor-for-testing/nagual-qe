//! Cognitive Core - Active Working Set
//!
//! The cognitive core represents the active working set of patterns at the
//! center of the light cone - the "now" point where past and future meet.
//!
//! This is inspired by the concept of working memory in cognitive science
//! and the attention mechanism in neural networks. The cognitive core
//! maintains the currently relevant patterns with attention weights that
//! determine their influence on reasoning.
//!
//! # Key Concepts
//!
//! - **Active Patterns**: Patterns currently in the working set
//! - **Attention Weights**: How much focus each pattern receives
//! - **Context Window**: Maximum number of patterns to track
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::profdag::cognitive_core::{CognitiveCore, CognitiveCoreConfig};
//!
//! let config = CognitiveCoreConfig::default().with_context_window(10);
//! let mut core = CognitiveCore::new(config);
//!
//! // Add patterns with attention weights
//! core.add_active_pattern("pattern-1", 0.9);
//! core.add_active_pattern("pattern-2", 0.7);
//!
//! // Get current context
//! let active = core.active_patterns();
//!
//! // Update attention
//! core.update_attention("pattern-1", 0.5);
//! ```

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::light_cone::PatternId;

/// Configuration for the cognitive core.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveCoreConfig {
    /// Maximum number of patterns in the context window.
    pub context_window: usize,

    /// Minimum attention threshold to keep a pattern active.
    pub min_attention_threshold: f32,

    /// Decay rate for attention over time (per tick).
    pub attention_decay_rate: f32,

    /// Whether to use soft attention (values sum to 1).
    pub use_soft_attention: bool,

    /// Temperature for softmax attention calculation.
    pub attention_temperature: f32,

    /// Maximum attention weight.
    pub max_attention: f32,
}

impl Default for CognitiveCoreConfig {
    fn default() -> Self {
        Self {
            context_window: 20,
            min_attention_threshold: 0.1,
            attention_decay_rate: 0.05,
            use_soft_attention: false,
            attention_temperature: 1.0,
            max_attention: 1.0,
        }
    }
}

impl CognitiveCoreConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the context window size.
    pub fn with_context_window(mut self, size: usize) -> Self {
        self.context_window = size.max(1);
        self
    }

    /// Set the minimum attention threshold.
    pub fn with_min_attention_threshold(mut self, threshold: f32) -> Self {
        self.min_attention_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the attention decay rate.
    pub fn with_attention_decay_rate(mut self, rate: f32) -> Self {
        self.attention_decay_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable soft attention.
    pub fn with_soft_attention(mut self, enable: bool) -> Self {
        self.use_soft_attention = enable;
        self
    }

    /// Set the attention temperature.
    pub fn with_attention_temperature(mut self, temperature: f32) -> Self {
        self.attention_temperature = temperature.max(0.01);
        self
    }
}

/// An active pattern in the cognitive core.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivePattern {
    /// Pattern ID.
    pub pattern_id: PatternId,

    /// Current attention weight (0.0 - 1.0).
    pub attention: f32,

    /// When this pattern was added to the core.
    pub added_at: DateTime<Utc>,

    /// When the attention was last updated.
    pub last_updated: DateTime<Utc>,

    /// Number of times this pattern was activated.
    pub activation_count: u32,

    /// Cumulative attention received.
    pub cumulative_attention: f32,

    /// Tags for categorization.
    pub tags: Vec<String>,

    /// Additional metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ActivePattern {
    /// Create a new active pattern.
    pub fn new(pattern_id: impl Into<PatternId>, attention: f32) -> Self {
        let now = Utc::now();
        Self {
            pattern_id: pattern_id.into(),
            attention: attention.clamp(0.0, 1.0),
            added_at: now,
            last_updated: now,
            activation_count: 1,
            cumulative_attention: attention.clamp(0.0, 1.0),
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Update the attention weight.
    pub fn update_attention(&mut self, attention: f32) {
        self.attention = attention.clamp(0.0, 1.0);
        self.last_updated = Utc::now();
        self.activation_count += 1;
        self.cumulative_attention += attention;
    }

    /// Apply attention decay.
    pub fn decay(&mut self, decay_rate: f32) {
        self.attention *= 1.0 - decay_rate;
        self.last_updated = Utc::now();
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Get the average attention over all activations.
    pub fn average_attention(&self) -> f32 {
        if self.activation_count == 0 {
            0.0
        } else {
            self.cumulative_attention / self.activation_count as f32
        }
    }

    /// Get the age of this pattern in the core (seconds).
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.added_at).num_seconds()
    }

    /// Check if this pattern is above the attention threshold.
    pub fn is_above_threshold(&self, threshold: f32) -> bool {
        self.attention >= threshold
    }
}

/// The Cognitive Core containing the active working set.
#[derive(Debug)]
pub struct CognitiveCore {
    /// Active patterns indexed by ID.
    patterns: HashMap<PatternId, ActivePattern>,

    /// Patterns sorted by attention (descending).
    sorted_patterns: Vec<PatternId>,

    /// Configuration.
    config: CognitiveCoreConfig,

    /// Total attention tick count (for decay tracking).
    tick_count: u64,
}

impl CognitiveCore {
    /// Create a new cognitive core.
    pub fn new(config: CognitiveCoreConfig) -> Self {
        Self {
            patterns: HashMap::new(),
            sorted_patterns: Vec::new(),
            config,
            tick_count: 0,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &CognitiveCoreConfig {
        &self.config
    }

    /// Add a pattern to the active set.
    ///
    /// If the pattern already exists, its attention is updated.
    pub fn add_active_pattern(&mut self, pattern_id: impl Into<PatternId>, attention: f32) {
        let id = pattern_id.into();
        let clamped_attention = attention.clamp(0.0, self.config.max_attention);

        if let Some(existing) = self.patterns.get_mut(&id) {
            // Update existing pattern
            existing.update_attention(clamped_attention);
        } else {
            // Add new pattern
            let pattern = ActivePattern::new(id.clone(), clamped_attention);
            self.patterns.insert(id.clone(), pattern);
        }

        // Re-sort patterns
        self.resort_patterns();

        // Apply soft attention if enabled
        if self.config.use_soft_attention {
            self.apply_soft_attention();
        }
    }

    /// Update attention for an existing pattern.
    pub fn update_attention(&mut self, pattern_id: &str, attention: f32) {
        if let Some(pattern) = self.patterns.get_mut(pattern_id) {
            pattern.update_attention(attention.clamp(0.0, self.config.max_attention));
            self.resort_patterns();

            if self.config.use_soft_attention {
                self.apply_soft_attention();
            }
        }
    }

    /// Remove a pattern from the active set.
    pub fn remove_pattern(&mut self, pattern_id: &str) -> bool {
        if self.patterns.remove(pattern_id).is_some() {
            self.sorted_patterns.retain(|id| id != pattern_id);
            true
        } else {
            false
        }
    }

    /// Get an active pattern by ID.
    pub fn get_pattern(&self, pattern_id: &str) -> Option<&ActivePattern> {
        self.patterns.get(pattern_id)
    }

    /// Get all active patterns.
    pub fn patterns(&self) -> impl Iterator<Item = &ActivePattern> {
        self.patterns.values()
    }

    /// Get active pattern IDs.
    pub fn active_patterns(&self) -> Vec<PatternId> {
        self.sorted_patterns.clone()
    }

    /// Get active patterns with their attention weights.
    pub fn active_patterns_with_weights(&self) -> Vec<(PatternId, f32)> {
        self.sorted_patterns
            .iter()
            .filter_map(|id| {
                self.patterns.get(id).map(|p| (id.clone(), p.attention))
            })
            .collect()
    }

    /// Get the number of active patterns.
    pub fn active_count(&self) -> usize {
        self.patterns.len()
    }

    /// Check if the core is empty.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Check if the core is at capacity.
    pub fn is_at_capacity(&self) -> bool {
        self.patterns.len() >= self.config.context_window
    }

    /// Get the total attention weight.
    pub fn total_attention(&self) -> f32 {
        self.patterns.values().map(|p| p.attention).sum()
    }

    /// Get the average attention weight.
    pub fn average_attention(&self) -> f32 {
        if self.patterns.is_empty() {
            return 0.0;
        }
        self.total_attention() / self.patterns.len() as f32
    }

    /// Get the pattern with highest attention.
    pub fn focus_pattern(&self) -> Option<&ActivePattern> {
        self.sorted_patterns
            .first()
            .and_then(|id| self.patterns.get(id))
    }

    /// Apply attention decay to all patterns.
    pub fn tick(&mut self) {
        self.tick_count += 1;

        for pattern in self.patterns.values_mut() {
            pattern.decay(self.config.attention_decay_rate);
        }

        // Remove patterns below threshold
        self.prune_below_threshold();

        // Re-sort after decay
        self.resort_patterns();
    }

    /// Prune patterns below the attention threshold.
    pub fn prune_below_threshold(&mut self) {
        let threshold = self.config.min_attention_threshold;
        let to_remove: Vec<PatternId> = self
            .patterns
            .iter()
            .filter(|(_, p)| p.attention < threshold)
            .map(|(id, _)| id.clone())
            .collect();

        for id in to_remove {
            self.patterns.remove(&id);
            self.sorted_patterns.retain(|pid| pid != &id);
        }
    }

    /// Prune to the context window size.
    ///
    /// Keeps the top N patterns by attention weight.
    pub fn prune_to_window(&mut self, window_size: usize) {
        if self.patterns.len() <= window_size {
            return;
        }

        // Keep only top window_size patterns
        let to_keep: Vec<PatternId> = self
            .sorted_patterns
            .iter()
            .take(window_size)
            .cloned()
            .collect();

        self.patterns.retain(|id, _| to_keep.contains(id));
        self.sorted_patterns.truncate(window_size);
    }

    /// Apply soft attention (softmax normalization).
    fn apply_soft_attention(&mut self) {
        if self.patterns.is_empty() {
            return;
        }

        // Calculate softmax
        let temp = self.config.attention_temperature;
        let max_attention = self
            .patterns
            .values()
            .map(|p| p.attention)
            .fold(f32::NEG_INFINITY, f32::max);

        let exp_sum: f32 = self
            .patterns
            .values()
            .map(|p| ((p.attention - max_attention) / temp).exp())
            .sum();

        if exp_sum > 0.0 {
            for pattern in self.patterns.values_mut() {
                let exp_val = ((pattern.attention - max_attention) / temp).exp();
                pattern.attention = exp_val / exp_sum;
            }
        }
    }

    /// Re-sort patterns by attention weight.
    fn resort_patterns(&mut self) {
        self.sorted_patterns = self.patterns.keys().cloned().collect();
        self.sorted_patterns.sort_by(|a, b| {
            let att_a = self.patterns.get(a).map(|p| p.attention).unwrap_or(0.0);
            let att_b = self.patterns.get(b).map(|p| p.attention).unwrap_or(0.0);
            att_b
                .partial_cmp(&att_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Boost attention for a pattern (multiplicative increase).
    pub fn boost(&mut self, pattern_id: &str, factor: f32) {
        if let Some(pattern) = self.patterns.get_mut(pattern_id) {
            let new_attention = (pattern.attention * factor).clamp(0.0, self.config.max_attention);
            pattern.update_attention(new_attention);
            self.resort_patterns();
        }
    }

    /// Inhibit attention for a pattern (multiplicative decrease).
    pub fn inhibit(&mut self, pattern_id: &str, factor: f32) {
        if let Some(pattern) = self.patterns.get_mut(pattern_id) {
            let new_attention = pattern.attention * (1.0 - factor).max(0.0);
            pattern.update_attention(new_attention);
            self.resort_patterns();
        }
    }

    /// Clear all patterns from the core.
    pub fn clear(&mut self) {
        self.patterns.clear();
        self.sorted_patterns.clear();
    }

    /// Get attention distribution statistics.
    pub fn attention_stats(&self) -> AttentionStats {
        if self.patterns.is_empty() {
            return AttentionStats::default();
        }

        let attentions: Vec<f32> = self.patterns.values().map(|p| p.attention).collect();

        let total: f32 = attentions.iter().sum();
        let mean = total / attentions.len() as f32;

        let variance = attentions
            .iter()
            .map(|a| (a - mean).powi(2))
            .sum::<f32>()
            / attentions.len() as f32;

        let mut sorted = attentions.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Calculate entropy (measure of attention spread)
        let entropy: f32 = if total > 0.0 {
            attentions
                .iter()
                .filter(|&&a| a > 0.0)
                .map(|&a| {
                    let p = a / total;
                    -p * p.ln()
                })
                .sum()
        } else {
            0.0
        };

        AttentionStats {
            count: attentions.len(),
            total,
            mean,
            median,
            std_dev: variance.sqrt(),
            min: sorted.first().copied().unwrap_or(0.0),
            max: sorted.last().copied().unwrap_or(0.0),
            entropy,
        }
    }

    /// Get patterns by tag.
    pub fn patterns_by_tag(&self, tag: &str) -> Vec<&ActivePattern> {
        self.patterns
            .values()
            .filter(|p| p.tags.contains(&tag.to_string()))
            .collect()
    }
}

/// Statistics about attention distribution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttentionStats {
    /// Number of active patterns.
    pub count: usize,
    /// Total attention weight.
    pub total: f32,
    /// Mean attention weight.
    pub mean: f32,
    /// Median attention weight.
    pub median: f32,
    /// Standard deviation.
    pub std_dev: f32,
    /// Minimum attention.
    pub min: f32,
    /// Maximum attention.
    pub max: f32,
    /// Entropy (higher = more spread out attention).
    pub entropy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_core_config_default() {
        let config = CognitiveCoreConfig::default();

        assert_eq!(config.context_window, 20);
        assert!((config.min_attention_threshold - 0.1).abs() < 0.001);
        assert!((config.attention_decay_rate - 0.05).abs() < 0.001);
        assert!(!config.use_soft_attention);
    }

    #[test]
    fn test_cognitive_core_config_builder() {
        let config = CognitiveCoreConfig::new()
            .with_context_window(10)
            .with_min_attention_threshold(0.2)
            .with_soft_attention(true);

        assert_eq!(config.context_window, 10);
        assert!((config.min_attention_threshold - 0.2).abs() < 0.001);
        assert!(config.use_soft_attention);
    }

    #[test]
    fn test_active_pattern_creation() {
        let pattern = ActivePattern::new("pattern-1", 0.8).with_tag("important");

        assert_eq!(pattern.pattern_id, "pattern-1");
        assert!((pattern.attention - 0.8).abs() < 0.001);
        assert_eq!(pattern.activation_count, 1);
        assert_eq!(pattern.tags.len(), 1);
    }

    #[test]
    fn test_active_pattern_update() {
        let mut pattern = ActivePattern::new("pattern-1", 0.5);
        pattern.update_attention(0.9);

        assert!((pattern.attention - 0.9).abs() < 0.001);
        assert_eq!(pattern.activation_count, 2);
        assert!((pattern.cumulative_attention - 1.4).abs() < 0.001);
    }

    #[test]
    fn test_active_pattern_decay() {
        let mut pattern = ActivePattern::new("pattern-1", 1.0);
        pattern.decay(0.1);

        assert!((pattern.attention - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_active_pattern_average_attention() {
        let mut pattern = ActivePattern::new("pattern-1", 0.6);
        pattern.update_attention(0.8);
        pattern.update_attention(0.5);

        // (0.6 + 0.8 + 0.5) / 3 = 0.633...
        let avg = pattern.average_attention();
        assert!((avg - 0.633).abs() < 0.01);
    }

    #[test]
    fn test_cognitive_core_basic() {
        let config = CognitiveCoreConfig::default();
        let mut core = CognitiveCore::new(config);

        core.add_active_pattern("p1", 0.8);
        core.add_active_pattern("p2", 0.6);

        assert_eq!(core.active_count(), 2);
        assert!(core.get_pattern("p1").is_some());
        assert!(core.get_pattern("p2").is_some());
    }

    #[test]
    fn test_cognitive_core_update_existing() {
        let config = CognitiveCoreConfig::default();
        let mut core = CognitiveCore::new(config);

        core.add_active_pattern("p1", 0.5);
        core.add_active_pattern("p1", 0.9);

        // Should update, not add
        assert_eq!(core.active_count(), 1);

        let pattern = core.get_pattern("p1").unwrap();
        assert!((pattern.attention - 0.9).abs() < 0.001);
        assert_eq!(pattern.activation_count, 2);
    }

    #[test]
    fn test_cognitive_core_sorted_patterns() {
        let config = CognitiveCoreConfig::default();
        let mut core = CognitiveCore::new(config);

        core.add_active_pattern("low", 0.3);
        core.add_active_pattern("high", 0.9);
        core.add_active_pattern("mid", 0.6);

        let sorted = core.active_patterns();

        assert_eq!(sorted[0], "high");
        assert_eq!(sorted[1], "mid");
        assert_eq!(sorted[2], "low");
    }

    #[test]
    fn test_cognitive_core_focus_pattern() {
        let config = CognitiveCoreConfig::default();
        let mut core = CognitiveCore::new(config);

        core.add_active_pattern("a", 0.5);
        core.add_active_pattern("b", 0.9);

        let focus = core.focus_pattern().unwrap();
        assert_eq!(focus.pattern_id, "b");
    }

    #[test]
    fn test_cognitive_core_tick_decay() {
        let config = CognitiveCoreConfig::new().with_attention_decay_rate(0.1);
        let mut core = CognitiveCore::new(config);

        core.add_active_pattern("p1", 1.0);
        core.tick();

        let pattern = core.get_pattern("p1").unwrap();
        assert!((pattern.attention - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_cognitive_core_prune_threshold() {
        let config = CognitiveCoreConfig::new().with_min_attention_threshold(0.5);
        let mut core = CognitiveCore::new(config);

        core.add_active_pattern("low", 0.3);
        core.add_active_pattern("high", 0.8);

        core.prune_below_threshold();

        assert_eq!(core.active_count(), 1);
        assert!(core.get_pattern("high").is_some());
        assert!(core.get_pattern("low").is_none());
    }

    #[test]
    fn test_cognitive_core_prune_to_window() {
        let config = CognitiveCoreConfig::new().with_context_window(2);
        let mut core = CognitiveCore::new(config);

        core.add_active_pattern("a", 0.9);
        core.add_active_pattern("b", 0.7);
        core.add_active_pattern("c", 0.5);
        core.add_active_pattern("d", 0.3);

        core.prune_to_window(2);

        assert_eq!(core.active_count(), 2);
        assert!(core.get_pattern("a").is_some());
        assert!(core.get_pattern("b").is_some());
        assert!(core.get_pattern("c").is_none());
        assert!(core.get_pattern("d").is_none());
    }

    #[test]
    fn test_cognitive_core_boost_inhibit() {
        let config = CognitiveCoreConfig::default();
        let mut core = CognitiveCore::new(config);

        core.add_active_pattern("p1", 0.5);

        core.boost("p1", 1.5);
        let pattern = core.get_pattern("p1").unwrap();
        assert!((pattern.attention - 0.75).abs() < 0.001);

        core.inhibit("p1", 0.5);
        let pattern = core.get_pattern("p1").unwrap();
        assert!((pattern.attention - 0.375).abs() < 0.001);
    }

    #[test]
    fn test_cognitive_core_total_average_attention() {
        let config = CognitiveCoreConfig::default();
        let mut core = CognitiveCore::new(config);

        core.add_active_pattern("a", 0.4);
        core.add_active_pattern("b", 0.6);

        assert!((core.total_attention() - 1.0).abs() < 0.001);
        assert!((core.average_attention() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_cognitive_core_soft_attention() {
        let config = CognitiveCoreConfig::new()
            .with_soft_attention(true)
            .with_attention_temperature(1.0);
        let mut core = CognitiveCore::new(config);

        core.add_active_pattern("a", 0.5);
        core.add_active_pattern("b", 0.5);

        // With soft attention, values should sum to ~1
        let total = core.total_attention();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cognitive_core_remove_pattern() {
        let config = CognitiveCoreConfig::default();
        let mut core = CognitiveCore::new(config);

        core.add_active_pattern("p1", 0.8);
        core.add_active_pattern("p2", 0.6);

        let removed = core.remove_pattern("p1");
        assert!(removed);
        assert_eq!(core.active_count(), 1);
        assert!(core.get_pattern("p1").is_none());
    }

    #[test]
    fn test_cognitive_core_clear() {
        let config = CognitiveCoreConfig::default();
        let mut core = CognitiveCore::new(config);

        core.add_active_pattern("p1", 0.8);
        core.add_active_pattern("p2", 0.6);
        core.clear();

        assert!(core.is_empty());
        assert_eq!(core.active_count(), 0);
    }

    #[test]
    fn test_cognitive_core_attention_stats() {
        let config = CognitiveCoreConfig::default();
        let mut core = CognitiveCore::new(config);

        core.add_active_pattern("a", 0.4);
        core.add_active_pattern("b", 0.6);
        core.add_active_pattern("c", 0.8);

        let stats = core.attention_stats();

        assert_eq!(stats.count, 3);
        assert!((stats.total - 1.8).abs() < 0.001);
        assert!((stats.mean - 0.6).abs() < 0.001);
        assert!((stats.median - 0.6).abs() < 0.001);
        assert!((stats.min - 0.4).abs() < 0.001);
        assert!((stats.max - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_cognitive_core_is_at_capacity() {
        let config = CognitiveCoreConfig::new().with_context_window(2);
        let mut core = CognitiveCore::new(config);

        assert!(!core.is_at_capacity());

        core.add_active_pattern("a", 0.5);
        assert!(!core.is_at_capacity());

        core.add_active_pattern("b", 0.5);
        assert!(core.is_at_capacity());
    }

    #[test]
    fn test_active_patterns_with_weights() {
        let config = CognitiveCoreConfig::default();
        let mut core = CognitiveCore::new(config);

        core.add_active_pattern("high", 0.9);
        core.add_active_pattern("low", 0.3);

        let weights = core.active_patterns_with_weights();

        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].0, "high");
        assert!((weights[0].1 - 0.9).abs() < 0.001);
        assert_eq!(weights[1].0, "low");
        assert!((weights[1].1 - 0.3).abs() < 0.001);
    }
}
