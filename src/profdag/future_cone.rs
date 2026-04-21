//! Future Cone - Probabilistic Predictions
//!
//! The future cone represents all possible future outcomes that could be
//! influenced by the current state (the center of the light cone).
//!
//! In physics, the future light cone contains all events that could be
//! reached by a signal from the observer. Similarly, the future cone in
//! ProfDAG contains probabilistic predictions of what might happen next.
//!
//! # Key Concepts
//!
//! - **Predicted Outcomes**: Future states with probability estimates
//! - **Probability Threshold**: Minimum probability to consider a prediction
//! - **Pattern Likelihood**: Based on historical pattern success rates
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::profdag::future_cone::{FutureCone, FutureConeConfig, PredictedOutcome};
//!
//! let config = FutureConeConfig::default().with_probability_threshold(0.5);
//! let mut cone = FutureCone::new("center_node", config);
//!
//! // Add a prediction
//! let prediction = PredictedOutcome::new("future_node", "Deployment will succeed")
//!     .with_probability(0.85)
//!     .with_confidence(0.9);
//! cone.add_prediction(prediction);
//!
//! // Query outcomes
//! let outcomes = cone.predict_outcomes("current_node")?;
//!
//! // Get likely patterns
//! let patterns = cone.likely_next_patterns(0.7);
//! ```

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::light_cone::{NodeId, PatternId};
use super::ProfDAGResult;

/// Configuration for the future cone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutureConeConfig {
    /// Minimum probability threshold to include a prediction.
    pub probability_threshold: f32,

    /// Maximum number of predictions to track.
    pub max_predictions: usize,

    /// Weight factor for recency in probability calculation.
    pub recency_weight: f32,

    /// Weight factor for pattern success rate.
    pub success_rate_weight: f32,

    /// Weight factor for pattern confidence.
    pub confidence_weight: f32,

    /// Default timeline (min days) for predictions without explicit timeline.
    pub default_timeline_min_days: u32,

    /// Default timeline (max days) for predictions without explicit timeline.
    pub default_timeline_max_days: u32,
}

impl Default for FutureConeConfig {
    fn default() -> Self {
        Self {
            probability_threshold: 0.3,
            max_predictions: 100,
            recency_weight: 0.2,
            success_rate_weight: 0.5,
            confidence_weight: 0.3,
            default_timeline_min_days: 1,
            default_timeline_max_days: 30,
        }
    }
}

impl FutureConeConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the probability threshold.
    pub fn with_probability_threshold(mut self, threshold: f32) -> Self {
        self.probability_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum number of predictions.
    pub fn with_max_predictions(mut self, max: usize) -> Self {
        self.max_predictions = max;
        self
    }

    /// Set weight factors.
    pub fn with_weights(
        mut self,
        recency: f32,
        success_rate: f32,
        confidence: f32,
    ) -> Self {
        // Normalize weights
        let total = recency + success_rate + confidence;
        if total > 0.0 {
            self.recency_weight = recency / total;
            self.success_rate_weight = success_rate / total;
            self.confidence_weight = confidence / total;
        }
        self
    }

    /// Set default timeline.
    pub fn with_default_timeline(mut self, min_days: u32, max_days: u32) -> Self {
        self.default_timeline_min_days = min_days;
        self.default_timeline_max_days = max_days.max(min_days);
        self
    }
}

/// A predicted future outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedOutcome {
    /// The node ID of the predicted outcome.
    pub node_id: NodeId,

    /// Description of the predicted outcome.
    pub description: String,

    /// Probability of this outcome (0.0 - 1.0).
    pub probability: f32,

    /// Confidence in the probability estimate (0.0 - 1.0).
    pub confidence: f32,

    /// Minimum expected days until this outcome.
    pub timeline_min_days: u32,

    /// Maximum expected days until this outcome.
    pub timeline_max_days: u32,

    /// Source pattern(s) that led to this prediction.
    pub source_patterns: Vec<PatternId>,

    /// When this prediction was made.
    pub predicted_at: DateTime<Utc>,

    /// Tags for categorization.
    pub tags: Vec<String>,

    /// Additional metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl PredictedOutcome {
    /// Create a new predicted outcome.
    pub fn new(node_id: impl Into<NodeId>, description: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            description: description.into(),
            probability: 0.5,
            confidence: 0.5,
            timeline_min_days: 1,
            timeline_max_days: 30,
            source_patterns: Vec::new(),
            predicted_at: Utc::now(),
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the probability.
    pub fn with_probability(mut self, probability: f32) -> Self {
        self.probability = probability.clamp(0.0, 1.0);
        self
    }

    /// Set the confidence.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set the timeline.
    pub fn with_timeline(mut self, min_days: u32, max_days: u32) -> Self {
        self.timeline_min_days = min_days;
        self.timeline_max_days = max_days.max(min_days);
        self
    }

    /// Add a source pattern.
    pub fn with_source_pattern(mut self, pattern_id: impl Into<PatternId>) -> Self {
        self.source_patterns.push(pattern_id.into());
        self
    }

    /// Set source patterns.
    pub fn with_source_patterns(mut self, patterns: Vec<PatternId>) -> Self {
        self.source_patterns = patterns;
        self
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

    /// Get the effective probability (probability * confidence).
    pub fn effective_probability(&self) -> f32 {
        self.probability * self.confidence
    }

    /// Get the timeline midpoint.
    pub fn timeline_midpoint(&self) -> u32 {
        (self.timeline_min_days + self.timeline_max_days) / 2
    }

    /// Get the probability interval based on confidence.
    pub fn probability_interval(&self) -> (f32, f32) {
        let half_width = (1.0 - self.confidence) * 0.5;
        let lower = (self.probability - half_width).max(0.0);
        let upper = (self.probability + half_width).min(1.0);
        (lower, upper)
    }

    /// Check if this is a high-confidence prediction.
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.7
    }

    /// Check if this is a likely outcome (probability > 0.5).
    pub fn is_likely(&self) -> bool {
        self.probability > 0.5
    }
}

/// A group of related predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionCluster {
    /// Cluster identifier.
    pub id: String,

    /// Common theme or pattern.
    pub theme: String,

    /// Predictions in this cluster.
    pub predictions: Vec<PredictedOutcome>,

    /// Combined probability (weighted average).
    pub combined_probability: f32,
}

impl PredictionCluster {
    /// Create a new prediction cluster.
    pub fn new(id: impl Into<String>, theme: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            theme: theme.into(),
            predictions: Vec::new(),
            combined_probability: 0.0,
        }
    }

    /// Add a prediction to the cluster.
    pub fn add_prediction(&mut self, prediction: PredictedOutcome) {
        self.predictions.push(prediction);
        self.recalculate_probability();
    }

    /// Recalculate the combined probability.
    fn recalculate_probability(&mut self) {
        if self.predictions.is_empty() {
            self.combined_probability = 0.0;
            return;
        }

        // Weighted average by confidence
        let total_weight: f32 = self.predictions.iter().map(|p| p.confidence).sum();

        if total_weight > 0.0 {
            self.combined_probability = self
                .predictions
                .iter()
                .map(|p| p.probability * p.confidence)
                .sum::<f32>()
                / total_weight;
        }
    }

    /// Get the number of predictions.
    pub fn len(&self) -> usize {
        self.predictions.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.predictions.is_empty()
    }
}

/// The Future Cone containing probabilistic predictions.
#[derive(Debug)]
pub struct FutureCone {
    /// The center node (present moment).
    center: NodeId,

    /// All predictions, indexed by node ID.
    predictions: HashMap<NodeId, PredictedOutcome>,

    /// Predictions sorted by probability (descending).
    sorted_predictions: Vec<NodeId>,

    /// Prediction clusters.
    clusters: Vec<PredictionCluster>,

    /// Configuration.
    config: FutureConeConfig,
}

impl FutureCone {
    /// Create a new future cone centered on a node.
    pub fn new(center: impl Into<NodeId>, config: FutureConeConfig) -> Self {
        Self {
            center: center.into(),
            predictions: HashMap::new(),
            sorted_predictions: Vec::new(),
            clusters: Vec::new(),
            config,
        }
    }

    /// Get the center node ID.
    pub fn center(&self) -> &NodeId {
        &self.center
    }

    /// Get the configuration.
    pub fn config(&self) -> &FutureConeConfig {
        &self.config
    }

    /// Add a prediction to the future cone.
    ///
    /// Predictions below the probability threshold are ignored.
    pub fn add_prediction(&mut self, prediction: PredictedOutcome) {
        if prediction.probability < self.config.probability_threshold {
            return;
        }

        // Check max predictions limit
        if self.predictions.len() >= self.config.max_predictions {
            // Remove lowest probability prediction
            if let Some(lowest) = self.sorted_predictions.last() {
                if let Some(existing) = self.predictions.get(lowest) {
                    if existing.probability < prediction.probability {
                        let lowest_id = lowest.clone();
                        self.predictions.remove(&lowest_id);
                        self.sorted_predictions
                            .retain(|id| id != &lowest_id);
                    } else {
                        return; // New prediction is lower than all existing
                    }
                }
            }
        }

        let node_id = prediction.node_id.clone();
        self.predictions.insert(node_id.clone(), prediction);

        // Insert into sorted list
        let idx = self
            .sorted_predictions
            .iter()
            .position(|id| {
                self.predictions
                    .get(id)
                    .map(|p| p.probability)
                    .unwrap_or(0.0)
                    < self.predictions.get(&node_id).map(|p| p.probability).unwrap_or(0.0)
            })
            .unwrap_or(self.sorted_predictions.len());

        self.sorted_predictions.insert(idx, node_id);
    }

    /// Get a prediction by node ID.
    pub fn get_prediction(&self, node_id: &str) -> Option<&PredictedOutcome> {
        self.predictions.get(node_id)
    }

    /// Get all predictions.
    pub fn predictions(&self) -> impl Iterator<Item = &PredictedOutcome> {
        self.predictions.values()
    }

    /// Get predictions sorted by probability (highest first).
    pub fn predictions_sorted(&self) -> Vec<&PredictedOutcome> {
        self.sorted_predictions
            .iter()
            .filter_map(|id| self.predictions.get(id))
            .collect()
    }

    /// Get the number of predictions.
    pub fn prediction_count(&self) -> usize {
        self.predictions.len()
    }

    /// Get the average probability of all predictions.
    pub fn avg_probability(&self) -> f32 {
        if self.predictions.is_empty() {
            return 0.0;
        }

        self.predictions.values().map(|p| p.probability).sum::<f32>()
            / self.predictions.len() as f32
    }

    /// Predict outcomes for a given node.
    ///
    /// This returns predictions that are connected to or influenced by
    /// the specified node.
    pub fn predict_outcomes(&self, node_id: &str) -> ProfDAGResult<Vec<PredictedOutcome>> {
        // Find predictions that have this node as a source pattern
        let mut results: Vec<PredictedOutcome> = self
            .predictions
            .values()
            .filter(|p| {
                p.source_patterns.contains(&node_id.to_string()) || node_id == self.center
            })
            .cloned()
            .collect();

        // Sort by probability
        results.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Get likely next patterns based on confidence threshold.
    ///
    /// Returns pattern IDs that have predictions above the confidence threshold.
    pub fn likely_next_patterns(&self, confidence: f32) -> Vec<PatternId> {
        self.predictions
            .values()
            .filter(|p| p.confidence >= confidence && p.probability > 0.5)
            .map(|p| p.node_id.clone())
            .collect()
    }

    /// Get high-confidence predictions.
    pub fn high_confidence_predictions(&self) -> Vec<&PredictedOutcome> {
        self.predictions
            .values()
            .filter(|p| p.is_high_confidence())
            .collect()
    }

    /// Get likely predictions (probability > 0.5).
    pub fn likely_predictions(&self) -> Vec<&PredictedOutcome> {
        self.predictions.values().filter(|p| p.is_likely()).collect()
    }

    /// Add a prediction cluster.
    pub fn add_cluster(&mut self, cluster: PredictionCluster) {
        self.clusters.push(cluster);
    }

    /// Get all clusters.
    pub fn clusters(&self) -> &[PredictionCluster] {
        &self.clusters
    }

    /// Get predictions within a timeline range.
    pub fn predictions_in_timeline(
        &self,
        min_days: u32,
        max_days: u32,
    ) -> Vec<&PredictedOutcome> {
        self.predictions
            .values()
            .filter(|p| p.timeline_min_days <= max_days && p.timeline_max_days >= min_days)
            .collect()
    }

    /// Calculate the expected value (probability-weighted outcome).
    ///
    /// For numeric predictions, this returns the weighted average.
    /// For categorical predictions, this returns the most likely outcome.
    pub fn most_likely_outcome(&self) -> Option<&PredictedOutcome> {
        self.sorted_predictions
            .first()
            .and_then(|id| self.predictions.get(id))
    }

    /// Get the probability distribution summary.
    pub fn probability_summary(&self) -> ProbabilitySummary {
        let probabilities: Vec<f32> = self.predictions.values().map(|p| p.probability).collect();

        if probabilities.is_empty() {
            return ProbabilitySummary::default();
        }

        let mean = probabilities.iter().sum::<f32>() / probabilities.len() as f32;

        let variance = probabilities
            .iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f32>()
            / probabilities.len() as f32;

        let std_dev = variance.sqrt();

        let mut sorted = probabilities.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        ProbabilitySummary {
            count: probabilities.len(),
            mean,
            median,
            std_dev,
            min: sorted.first().copied().unwrap_or(0.0),
            max: sorted.last().copied().unwrap_or(0.0),
        }
    }
}

/// Summary statistics for prediction probabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbabilitySummary {
    /// Number of predictions.
    pub count: usize,
    /// Mean probability.
    pub mean: f32,
    /// Median probability.
    pub median: f32,
    /// Standard deviation.
    pub std_dev: f32,
    /// Minimum probability.
    pub min: f32,
    /// Maximum probability.
    pub max: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_future_cone_config_default() {
        let config = FutureConeConfig::default();

        assert!((config.probability_threshold - 0.3).abs() < 0.001);
        assert_eq!(config.max_predictions, 100);
        assert!((config.recency_weight - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_future_cone_config_builder() {
        let config = FutureConeConfig::new()
            .with_probability_threshold(0.5)
            .with_max_predictions(50)
            .with_default_timeline(7, 14);

        assert!((config.probability_threshold - 0.5).abs() < 0.001);
        assert_eq!(config.max_predictions, 50);
        assert_eq!(config.default_timeline_min_days, 7);
        assert_eq!(config.default_timeline_max_days, 14);
    }

    #[test]
    fn test_predicted_outcome_creation() {
        let outcome = PredictedOutcome::new("node-1", "Test outcome")
            .with_probability(0.8)
            .with_confidence(0.9)
            .with_timeline(7, 14)
            .with_source_pattern("pattern-1")
            .with_tag("important");

        assert_eq!(outcome.node_id, "node-1");
        assert!((outcome.probability - 0.8).abs() < 0.001);
        assert!((outcome.confidence - 0.9).abs() < 0.001);
        assert_eq!(outcome.timeline_min_days, 7);
        assert_eq!(outcome.timeline_max_days, 14);
        assert_eq!(outcome.source_patterns.len(), 1);
        assert_eq!(outcome.tags.len(), 1);
    }

    #[test]
    fn test_predicted_outcome_effective_probability() {
        let outcome = PredictedOutcome::new("node", "test")
            .with_probability(0.8)
            .with_confidence(0.5);

        assert!((outcome.effective_probability() - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_predicted_outcome_probability_interval() {
        let outcome = PredictedOutcome::new("node", "test")
            .with_probability(0.7)
            .with_confidence(0.8); // narrow interval

        let (lower, upper) = outcome.probability_interval();

        assert!(lower < 0.7);
        assert!(upper > 0.7);
        assert!(lower >= 0.0);
        assert!(upper <= 1.0);
    }

    #[test]
    fn test_predicted_outcome_timeline_midpoint() {
        let outcome = PredictedOutcome::new("node", "test").with_timeline(7, 21);

        assert_eq!(outcome.timeline_midpoint(), 14);
    }

    #[test]
    fn test_predicted_outcome_classifications() {
        let high_conf = PredictedOutcome::new("a", "test")
            .with_probability(0.8)
            .with_confidence(0.85);

        assert!(high_conf.is_high_confidence());
        assert!(high_conf.is_likely());

        let low = PredictedOutcome::new("b", "test")
            .with_probability(0.3)
            .with_confidence(0.4);

        assert!(!low.is_high_confidence());
        assert!(!low.is_likely());
    }

    #[test]
    fn test_prediction_cluster() {
        let mut cluster = PredictionCluster::new("cluster-1", "Deployment outcomes");

        let p1 = PredictedOutcome::new("success", "Success")
            .with_probability(0.8)
            .with_confidence(0.9);

        let p2 = PredictedOutcome::new("partial", "Partial success")
            .with_probability(0.6)
            .with_confidence(0.7);

        cluster.add_prediction(p1);
        cluster.add_prediction(p2);

        assert_eq!(cluster.len(), 2);
        assert!(cluster.combined_probability > 0.0);
    }

    #[test]
    fn test_future_cone_basic() {
        let config = FutureConeConfig::default();
        let mut cone = FutureCone::new("center", config);

        let p1 = PredictedOutcome::new("node-1", "Outcome 1").with_probability(0.8);
        let p2 = PredictedOutcome::new("node-2", "Outcome 2").with_probability(0.6);

        cone.add_prediction(p1);
        cone.add_prediction(p2);

        assert_eq!(cone.prediction_count(), 2);
        assert!(cone.get_prediction("node-1").is_some());
        assert!(cone.get_prediction("node-2").is_some());
    }

    #[test]
    fn test_future_cone_threshold_filtering() {
        let config = FutureConeConfig::new().with_probability_threshold(0.5);
        let mut cone = FutureCone::new("center", config);

        let low = PredictedOutcome::new("low", "Low prob").with_probability(0.3);
        let high = PredictedOutcome::new("high", "High prob").with_probability(0.7);

        cone.add_prediction(low);
        cone.add_prediction(high);

        // Only high probability prediction should be included
        assert_eq!(cone.prediction_count(), 1);
        assert!(cone.get_prediction("high").is_some());
        assert!(cone.get_prediction("low").is_none());
    }

    #[test]
    fn test_future_cone_sorted_predictions() {
        let config = FutureConeConfig::default();
        let mut cone = FutureCone::new("center", config);

        cone.add_prediction(PredictedOutcome::new("low", "Low").with_probability(0.4));
        cone.add_prediction(PredictedOutcome::new("high", "High").with_probability(0.9));
        cone.add_prediction(PredictedOutcome::new("mid", "Mid").with_probability(0.6));

        let sorted = cone.predictions_sorted();

        assert_eq!(sorted[0].node_id, "high");
        assert_eq!(sorted[1].node_id, "mid");
        assert_eq!(sorted[2].node_id, "low");
    }

    #[test]
    fn test_future_cone_avg_probability() {
        let config = FutureConeConfig::default();
        let mut cone = FutureCone::new("center", config);

        cone.add_prediction(PredictedOutcome::new("a", "A").with_probability(0.6));
        cone.add_prediction(PredictedOutcome::new("b", "B").with_probability(0.8));

        assert!((cone.avg_probability() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_future_cone_predict_outcomes() {
        let config = FutureConeConfig::default();
        let mut cone = FutureCone::new("center", config);

        let p1 = PredictedOutcome::new("outcome-1", "Test")
            .with_probability(0.8)
            .with_source_pattern("center");

        let p2 = PredictedOutcome::new("outcome-2", "Test")
            .with_probability(0.6)
            .with_source_pattern("other");

        cone.add_prediction(p1);
        cone.add_prediction(p2);

        let outcomes = cone.predict_outcomes("center").unwrap();

        // Should return predictions with center as source or all if querying center
        assert!(!outcomes.is_empty());
    }

    #[test]
    fn test_future_cone_likely_next_patterns() {
        let config = FutureConeConfig::default();
        let mut cone = FutureCone::new("center", config);

        cone.add_prediction(
            PredictedOutcome::new("a", "A")
                .with_probability(0.8)
                .with_confidence(0.9),
        );
        cone.add_prediction(
            PredictedOutcome::new("b", "B")
                .with_probability(0.7)
                .with_confidence(0.5),
        );
        cone.add_prediction(
            PredictedOutcome::new("c", "C")
                .with_probability(0.3)
                .with_confidence(0.9),
        );

        let patterns = cone.likely_next_patterns(0.7);

        // Only 'a' has high probability and high confidence
        assert_eq!(patterns.len(), 1);
        assert!(patterns.contains(&"a".to_string()));
    }

    #[test]
    fn test_future_cone_most_likely_outcome() {
        let config = FutureConeConfig::default();
        let mut cone = FutureCone::new("center", config);

        cone.add_prediction(PredictedOutcome::new("a", "A").with_probability(0.6));
        cone.add_prediction(PredictedOutcome::new("b", "B").with_probability(0.9));

        let most_likely = cone.most_likely_outcome().unwrap();
        assert_eq!(most_likely.node_id, "b");
    }

    #[test]
    fn test_future_cone_probability_summary() {
        let config = FutureConeConfig::default();
        let mut cone = FutureCone::new("center", config);

        cone.add_prediction(PredictedOutcome::new("a", "A").with_probability(0.4));
        cone.add_prediction(PredictedOutcome::new("b", "B").with_probability(0.6));
        cone.add_prediction(PredictedOutcome::new("c", "C").with_probability(0.8));

        let summary = cone.probability_summary();

        assert_eq!(summary.count, 3);
        assert!((summary.mean - 0.6).abs() < 0.001);
        assert!((summary.median - 0.6).abs() < 0.001);
        assert!((summary.min - 0.4).abs() < 0.001);
        assert!((summary.max - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_future_cone_predictions_in_timeline() {
        let config = FutureConeConfig::default();
        let mut cone = FutureCone::new("center", config);

        cone.add_prediction(
            PredictedOutcome::new("short", "Short")
                .with_probability(0.5)
                .with_timeline(1, 7),
        );
        cone.add_prediction(
            PredictedOutcome::new("long", "Long")
                .with_probability(0.5)
                .with_timeline(30, 60),
        );

        let short_term = cone.predictions_in_timeline(1, 10);
        assert_eq!(short_term.len(), 1);
        assert_eq!(short_term[0].node_id, "short");

        let long_term = cone.predictions_in_timeline(30, 90);
        assert_eq!(long_term.len(), 1);
        assert_eq!(long_term[0].node_id, "long");
    }
}
