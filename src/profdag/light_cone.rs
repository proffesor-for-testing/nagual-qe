//! Light Cone Temporal Reasoning Model
//!
//! This module implements a Light Cone model for temporal reasoning over ProfDAG,
//! inspired by FoxFlow's cognitive globe concept. The light cone provides a
//! relativistic-inspired framework for understanding causality in knowledge graphs.
//!
//! # Conceptual Model
//!
//! ```text
//!                    Future Cone
//!                   /           \
//!                  /  Predicted  \
//!                 /    Outcomes   \
//!                /        |        \
//!               +---------+---------+
//!               |     NOW (center)  |
//!               +---------+---------+
//!                \        |        /
//!                 \  Causal Past  /
//!                  \   Events    /
//!                   \           /
//!                    History Cone
//! ```
//!
//! The light cone divides events into three regions:
//! - **History Cone**: All causally connected past events (can influence now)
//! - **Future Cone**: All potentially influenced future events (predictions)
//! - **Cognitive Core**: The active working set at the center (current context)
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::profdag::light_cone::{LightCone, LightConeConfig};
//!
//! // Create a light cone centered on a node
//! let light_cone = LightCone::new(center_node_id, config);
//!
//! // Query the past: "What led to this?"
//! let causes = light_cone.history_cone.trace_back(&node_id, 5).await?;
//!
//! // Query the future: "What might follow?"
//! let predictions = light_cone.future_cone.predict_outcomes(&node_id).await?;
//!
//! // Query current context: "What's relevant now?"
//! let active = light_cone.cognitive_core.active_patterns();
//! ```

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

use super::cognitive_core::{CognitiveCore, CognitiveCoreConfig};
use super::future_cone::{FutureCone, FutureConeConfig, PredictedOutcome};
use super::history_cone::{CausalChain, HistoryCone, HistoryConeConfig, TemporalNode};
use super::{EdgeType, NodeType, ProfDAGError, ProfDAGNode, ProfDAGResult, ProfDAGStorage};

/// Unique identifier for a node in the light cone context.
pub type NodeId = String;

/// Unique identifier for a pattern.
pub type PatternId = String;

/// Configuration for the light cone model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightConeConfig {
    /// Configuration for the history cone.
    pub history: HistoryConeConfig,

    /// Configuration for the future cone.
    pub future: FutureConeConfig,

    /// Configuration for the cognitive core.
    pub cognitive: CognitiveCoreConfig,

    /// Maximum temporal distance to consider (in hours).
    pub max_temporal_distance_hours: i32,

    /// Minimum edge weight to follow during traversal.
    pub min_edge_weight: f64,

    /// Whether to cache light cone computations.
    pub enable_caching: bool,

    /// Cache TTL in seconds.
    pub cache_ttl_secs: u64,
}

impl Default for LightConeConfig {
    fn default() -> Self {
        Self {
            history: HistoryConeConfig::default(),
            future: FutureConeConfig::default(),
            cognitive: CognitiveCoreConfig::default(),
            max_temporal_distance_hours: 24 * 30, // 30 days
            min_edge_weight: 0.3,
            enable_caching: true,
            cache_ttl_secs: 300, // 5 minutes
        }
    }
}

impl LightConeConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set history cone configuration.
    pub fn with_history_config(mut self, config: HistoryConeConfig) -> Self {
        self.history = config;
        self
    }

    /// Set future cone configuration.
    pub fn with_future_config(mut self, config: FutureConeConfig) -> Self {
        self.future = config;
        self
    }

    /// Set cognitive core configuration.
    pub fn with_cognitive_config(mut self, config: CognitiveCoreConfig) -> Self {
        self.cognitive = config;
        self
    }

    /// Set maximum temporal distance.
    pub fn with_max_temporal_distance(mut self, hours: i32) -> Self {
        self.max_temporal_distance_hours = hours;
        self
    }

    /// Set minimum edge weight.
    pub fn with_min_edge_weight(mut self, weight: f64) -> Self {
        self.min_edge_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable caching.
    pub fn with_caching(mut self, enable: bool) -> Self {
        self.enable_caching = enable;
        self
    }
}

/// The Light Cone model for temporal reasoning.
///
/// A light cone provides a unified view of causality centered on a specific
/// point in the knowledge graph. It enables three types of temporal queries:
///
/// 1. **Historical queries** (history cone): "What led to this?"
/// 2. **Predictive queries** (future cone): "What might follow?"
/// 3. **Contextual queries** (cognitive core): "What's relevant now?"
pub struct LightCone {
    /// The center point ("now") of the light cone.
    pub center: NodeId,

    /// The history cone containing causal past events.
    pub history_cone: HistoryCone,

    /// The future cone containing probabilistic predictions.
    pub future_cone: FutureCone,

    /// The cognitive core containing the active working set.
    pub cognitive_core: CognitiveCore,

    /// Reference to the ProfDAG storage.
    storage: Arc<ProfDAGStorage>,

    /// Configuration for the light cone.
    config: LightConeConfig,

    /// When this light cone was created.
    created_at: DateTime<Utc>,

    /// When the light cone was last updated.
    updated_at: DateTime<Utc>,
}

impl LightCone {
    /// Create a new light cone centered on a node.
    ///
    /// This initializes the light cone but does not populate the cones.
    /// Use `build()` to fully construct the light cone.
    pub fn new(center: impl Into<NodeId>, storage: Arc<ProfDAGStorage>) -> Self {
        Self::with_config(center, storage, LightConeConfig::default())
    }

    /// Create a new light cone with custom configuration.
    pub fn with_config(
        center: impl Into<NodeId>,
        storage: Arc<ProfDAGStorage>,
        config: LightConeConfig,
    ) -> Self {
        let center_id = center.into();
        let now = Utc::now();

        Self {
            center: center_id.clone(),
            history_cone: HistoryCone::new(center_id.clone(), config.history.clone()),
            future_cone: FutureCone::new(center_id.clone(), config.future.clone()),
            cognitive_core: CognitiveCore::new(config.cognitive.clone()),
            storage,
            config,
            created_at: now,
            updated_at: now,
        }
    }

    /// Build the light cone by populating all three components.
    ///
    /// This performs graph traversal to populate:
    /// - History cone with causal ancestors
    /// - Future cone with predicted outcomes
    /// - Cognitive core with active patterns
    #[instrument(skip(self), fields(center = %self.center))]
    pub async fn build(&mut self) -> ProfDAGResult<()> {
        info!(center = %self.center, "Building light cone");

        // Build history cone by traversing backward
        self.build_history_cone().await?;

        // Build future cone by analyzing patterns and predictions
        self.build_future_cone().await?;

        // Build cognitive core from active patterns
        self.build_cognitive_core().await?;

        self.updated_at = Utc::now();

        debug!(
            history_nodes = self.history_cone.node_count(),
            future_predictions = self.future_cone.prediction_count(),
            active_patterns = self.cognitive_core.active_count(),
            "Light cone built"
        );

        Ok(())
    }

    /// Build the history cone by traversing causal ancestors.
    async fn build_history_cone(&mut self) -> ProfDAGResult<()> {
        // Get the center node
        let center_node = self
            .storage
            .get_node(&self.center)
            .await?
            .ok_or_else(|| ProfDAGError::NodeNotFound {
                id: self.center.clone(),
            })?;

        // Add center node to history
        let center_temporal = TemporalNode::from_profdag_node(&center_node, 0);
        self.history_cone.add_node(center_temporal);

        // Recursively trace back through causal edges
        self.trace_history(&self.center.clone(), 1).await?;

        Ok(())
    }

    /// Recursively trace history through the graph.
    async fn trace_history(&mut self, node_id: &str, depth: usize) -> ProfDAGResult<()> {
        if depth > self.config.history.max_depth {
            return Ok(());
        }

        // Get incoming causal edges (leads_to, derived_from)
        let query = super::NeighborQuery::incoming()
            .with_edge_type(EdgeType::LeadsTo)
            .with_min_weight(self.config.min_edge_weight)
            .with_limit(self.config.history.max_ancestors_per_node);

        let neighbors = self.storage.get_neighbors(node_id, &query).await?;

        for neighbor_result in neighbors {
            let node = neighbor_result.node;
            let edge = neighbor_result.edge;

            // Check if already visited
            if self.history_cone.contains_node(&node.id) {
                continue;
            }

            // Check temporal distance
            if let Some(distance) = edge.temporal_distance_hours {
                if distance.abs() > self.config.max_temporal_distance_hours {
                    continue;
                }
            }

            // Add node to history cone
            let temporal_node = TemporalNode::from_profdag_node(&node, depth);
            self.history_cone.add_node(temporal_node);

            // Add causal chain segment
            let chain = CausalChain::new(node.id.clone(), node_id.to_string())
                .with_edge_type(edge.edge_type)
                .with_weight(edge.weight);
            self.history_cone.add_causal_chain(chain);

            // Recurse deeper
            Box::pin(self.trace_history(&node.id, depth + 1)).await?;
        }

        // Also trace derived_from edges
        let derived_query = super::NeighborQuery::incoming()
            .with_edge_type(EdgeType::DerivedFrom)
            .with_min_weight(self.config.min_edge_weight)
            .with_limit(self.config.history.max_ancestors_per_node);

        let derived_neighbors = self.storage.get_neighbors(node_id, &derived_query).await?;

        for neighbor_result in derived_neighbors {
            let node = neighbor_result.node;
            let edge = neighbor_result.edge;

            if self.history_cone.contains_node(&node.id) {
                continue;
            }

            let temporal_node = TemporalNode::from_profdag_node(&node, depth);
            self.history_cone.add_node(temporal_node);

            let chain = CausalChain::new(node.id.clone(), node_id.to_string())
                .with_edge_type(edge.edge_type)
                .with_weight(edge.weight);
            self.history_cone.add_causal_chain(chain);

            Box::pin(self.trace_history(&node.id, depth + 1)).await?;
        }

        Ok(())
    }

    /// Build the future cone by analyzing patterns and predictions.
    async fn build_future_cone(&mut self) -> ProfDAGResult<()> {
        // Get outgoing edges from center (leads_to)
        let query = super::NeighborQuery::outgoing()
            .with_edge_type(EdgeType::LeadsTo)
            .with_min_weight(self.config.min_edge_weight);

        let neighbors = self.storage.get_neighbors(&self.center, &query).await?;

        for neighbor_result in neighbors {
            let node = neighbor_result.node;
            let edge = neighbor_result.edge;

            // Create prediction based on edge weight and node confidence
            let probability = edge.weight * node.confidence as f64;

            let prediction = PredictedOutcome::new(node.id.clone(), node.content.clone())
                .with_probability(probability as f32)
                .with_confidence(node.confidence)
                .with_source_pattern(self.center.clone());

            if probability >= self.config.future.probability_threshold as f64 {
                self.future_cone.add_prediction(prediction);
            }
        }

        // Also look at prediction nodes
        let prediction_nodes = self
            .storage
            .get_nodes_by_type(NodeType::Prediction, 50)
            .await?;

        for node in prediction_nodes {
            // Check if this prediction is related to our center
            if self.is_prediction_relevant(&node).await? {
                let prediction =
                    PredictedOutcome::new(node.id.clone(), node.content.clone())
                        .with_probability(node.confidence)
                        .with_confidence(node.confidence)
                        .with_source_pattern(self.center.clone());

                if node.confidence >= self.config.future.probability_threshold {
                    self.future_cone.add_prediction(prediction);
                }
            }
        }

        Ok(())
    }

    /// Check if a prediction node is relevant to the current center.
    async fn is_prediction_relevant(&self, prediction_node: &ProfDAGNode) -> ProfDAGResult<bool> {
        // Check if there's a path from center to this prediction
        let query = super::NeighborQuery::outgoing()
            .with_edge_type(EdgeType::LeadsTo)
            .with_limit(100);

        let neighbors = self.storage.get_neighbors(&self.center, &query).await?;

        for neighbor in &neighbors {
            if neighbor.node.id == prediction_node.id {
                return Ok(true);
            }
        }

        // Check for wormhole connections
        let wormhole_query = super::NeighborQuery::both()
            .with_edge_type(EdgeType::Wormhole)
            .with_min_weight(0.5);

        let wormhole_neighbors = self
            .storage
            .get_neighbors(&self.center, &wormhole_query)
            .await?;

        for neighbor in &wormhole_neighbors {
            if neighbor.node.id == prediction_node.id {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Build the cognitive core from active patterns.
    async fn build_cognitive_core(&mut self) -> ProfDAGResult<()> {
        // Get center node
        let center_node = self.storage.get_node(&self.center).await?;

        if let Some(node) = center_node {
            // Add center as primary active pattern
            self.cognitive_core.add_active_pattern(
                node.id.clone(),
                1.0, // Highest attention weight for center
            );
        }

        // Add patterns from history cone with decaying attention
        for temporal_node in self.history_cone.nodes() {
            let depth = temporal_node.depth;
            let attention = self.calculate_attention_weight(depth);

            if attention >= self.config.cognitive.min_attention_threshold {
                self.cognitive_core
                    .add_active_pattern(temporal_node.id.clone(), attention);
            }
        }

        // Add patterns from future cone based on probability
        for prediction in self.future_cone.predictions() {
            let attention = prediction.probability * prediction.confidence;

            if attention >= self.config.cognitive.min_attention_threshold {
                self.cognitive_core
                    .add_active_pattern(prediction.node_id.clone(), attention);
            }
        }

        // Limit to context window size
        self.cognitive_core
            .prune_to_window(self.config.cognitive.context_window);

        Ok(())
    }

    /// Calculate attention weight based on depth from center.
    fn calculate_attention_weight(&self, depth: usize) -> f32 {
        // Exponential decay: attention = base_weight * decay^depth
        let base_weight = 1.0_f32;
        let decay_factor = 0.7_f32;

        base_weight * decay_factor.powi(depth as i32)
    }

    // ========================================================================
    // Query Methods
    // ========================================================================

    /// Query: "What led to X?" - Trace the causal history.
    ///
    /// Returns temporal nodes representing causally connected past events.
    #[instrument(skip(self), fields(node_id = %node_id, depth = depth))]
    pub async fn what_led_to(&self, node_id: &str, depth: usize) -> ProfDAGResult<Vec<TemporalNode>> {
        self.history_cone.trace_back(node_id, depth)
    }

    /// Query: "What might follow Y?" - Get predicted outcomes.
    ///
    /// Returns predicted outcomes with probabilities.
    #[instrument(skip(self), fields(node_id = %node_id))]
    pub async fn what_might_follow(
        &self,
        node_id: &str,
    ) -> ProfDAGResult<Vec<PredictedOutcome>> {
        self.future_cone.predict_outcomes(node_id)
    }

    /// Query: "What's currently relevant?" - Get active patterns.
    ///
    /// Returns the currently active patterns with attention weights.
    pub fn whats_relevant(&self) -> Vec<(PatternId, f32)> {
        self.cognitive_core.active_patterns_with_weights()
    }

    /// Find root causes for a given node.
    ///
    /// Traces back through the history cone to find original causative events.
    pub fn find_root_causes(&self, node_id: &str) -> Vec<NodeId> {
        self.history_cone.find_root_causes(node_id)
    }

    /// Get the causal path between two nodes.
    ///
    /// Returns the chain of causally connected events from source to target.
    pub fn get_causal_path(&self, from: &str, to: &str) -> Option<CausalChain> {
        self.history_cone.get_causal_path(from, to)
    }

    /// Get likely next patterns based on current context.
    ///
    /// Returns patterns that are likely to be relevant next, based on the
    /// future cone predictions filtered by confidence.
    pub fn likely_next_patterns(&self, confidence: f32) -> Vec<PatternId> {
        self.future_cone.likely_next_patterns(confidence)
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get the center node ID.
    pub fn center(&self) -> &NodeId {
        &self.center
    }

    /// Get the configuration.
    pub fn config(&self) -> &LightConeConfig {
        &self.config
    }

    /// Get when the light cone was created.
    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// Get when the light cone was last updated.
    pub fn updated_at(&self) -> DateTime<Utc> {
        self.updated_at
    }

    /// Get statistics about the light cone.
    pub fn stats(&self) -> LightConeStats {
        LightConeStats {
            center: self.center.clone(),
            history_node_count: self.history_cone.node_count(),
            history_chain_count: self.history_cone.chain_count(),
            future_prediction_count: self.future_cone.prediction_count(),
            active_pattern_count: self.cognitive_core.active_count(),
            max_history_depth: self.history_cone.max_depth(),
            avg_prediction_probability: self.future_cone.avg_probability(),
            created_at: self.created_at,
            updated_at: self.updated_at,
        }
    }

    /// Recenter the light cone on a different node.
    ///
    /// This rebuilds the entire light cone around the new center.
    #[instrument(skip(self, new_center))]
    pub async fn recenter(&mut self, new_center: impl Into<NodeId>) -> ProfDAGResult<()> {
        self.center = new_center.into();
        self.history_cone = HistoryCone::new(self.center.clone(), self.config.history.clone());
        self.future_cone = FutureCone::new(self.center.clone(), self.config.future.clone());
        self.cognitive_core = CognitiveCore::new(self.config.cognitive.clone());

        self.build().await
    }
}

/// Statistics about a light cone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightConeStats {
    /// The center node ID.
    pub center: NodeId,

    /// Number of nodes in the history cone.
    pub history_node_count: usize,

    /// Number of causal chains in the history cone.
    pub history_chain_count: usize,

    /// Number of predictions in the future cone.
    pub future_prediction_count: usize,

    /// Number of active patterns in the cognitive core.
    pub active_pattern_count: usize,

    /// Maximum depth reached in the history cone.
    pub max_history_depth: usize,

    /// Average probability of future predictions.
    pub avg_prediction_probability: f32,

    /// When the light cone was created.
    pub created_at: DateTime<Utc>,

    /// When the light cone was last updated.
    pub updated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_light_cone_config_default() {
        let config = LightConeConfig::default();

        assert_eq!(config.max_temporal_distance_hours, 24 * 30);
        assert!((config.min_edge_weight - 0.3).abs() < 0.001);
        assert!(config.enable_caching);
    }

    #[test]
    fn test_light_cone_config_builder() {
        let config = LightConeConfig::new()
            .with_max_temporal_distance(48)
            .with_min_edge_weight(0.5)
            .with_caching(false);

        assert_eq!(config.max_temporal_distance_hours, 48);
        assert!((config.min_edge_weight - 0.5).abs() < 0.001);
        assert!(!config.enable_caching);
    }

    #[test]
    fn test_attention_weight_calculation() {
        // We can't easily test without a full LightCone instance,
        // but we can verify the formula
        let base = 1.0_f32;
        let decay = 0.7_f32;

        let depth_0 = base * decay.powi(0);
        let depth_1 = base * decay.powi(1);
        let depth_2 = base * decay.powi(2);
        let depth_5 = base * decay.powi(5);

        assert!((depth_0 - 1.0).abs() < 0.001);
        assert!((depth_1 - 0.7).abs() < 0.001);
        assert!((depth_2 - 0.49).abs() < 0.001);
        assert!(depth_5 < depth_2);
    }

    #[test]
    fn test_light_cone_stats_serialization() {
        let stats = LightConeStats {
            center: "node-123".to_string(),
            history_node_count: 10,
            history_chain_count: 8,
            future_prediction_count: 5,
            active_pattern_count: 7,
            max_history_depth: 3,
            avg_prediction_probability: 0.75,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: LightConeStats = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.center, stats.center);
        assert_eq!(deserialized.history_node_count, stats.history_node_count);
        assert_eq!(
            deserialized.future_prediction_count,
            stats.future_prediction_count
        );
    }
}
