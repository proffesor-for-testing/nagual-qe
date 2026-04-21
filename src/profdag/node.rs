//! ProfDAG Node types and structures.
//!
//! Defines the node types and data structures for the ProfDAG graph.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Types of nodes in the ProfDAG.
///
/// Each node type represents a different kind of knowledge entity
/// in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeType {
    /// A learned problem-solution pair from ReasoningBank
    Pattern,

    /// A sequence of actions/decisions in a session
    Trajectory,

    /// A probabilistic forecast from the prediction engine
    Prediction,

    /// A choice point with tracked outcomes
    Decision,
}

impl NodeType {
    /// Returns all node types as a slice.
    pub fn all() -> &'static [NodeType] {
        &[
            NodeType::Pattern,
            NodeType::Trajectory,
            NodeType::Prediction,
            NodeType::Decision,
        ]
    }

    /// Parse node type from string.
    pub fn from_str(s: &str) -> Option<NodeType> {
        match s.to_lowercase().as_str() {
            "pattern" => Some(NodeType::Pattern),
            "trajectory" => Some(NodeType::Trajectory),
            "prediction" => Some(NodeType::Prediction),
            "decision" => Some(NodeType::Decision),
            _ => None,
        }
    }

    /// Convert to database string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeType::Pattern => "pattern",
            NodeType::Trajectory => "trajectory",
            NodeType::Prediction => "prediction",
            NodeType::Decision => "decision",
        }
    }
}

impl fmt::Display for NodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A node in the ProfDAG representing a knowledge entity.
///
/// Nodes contain content, optional embeddings for semantic search,
/// and metadata for tracking provenance and quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfDAGNode {
    /// Unique identifier (UUID)
    pub id: String,

    /// Type of this node
    pub node_type: NodeType,

    /// The actual content/description
    pub content: String,

    /// Vector embedding for semantic search (128-dimensional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,

    /// Flexible metadata as JSON
    #[serde(default)]
    pub metadata: serde_json::Value,

    /// Reference to source entity ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,

    /// Type of source entity (reasoning_patterns, predictions, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_type: Option<String>,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,

    /// Importance score (0.0 - 1.0)
    pub importance: f32,

    /// Agent that created this node
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,

    /// Session context
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,

    /// When the node was created
    pub created_at: DateTime<Utc>,

    /// When the node was last updated
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
}

impl ProfDAGNode {
    /// Create a new node with the given type and content.
    pub fn new(node_type: NodeType, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            node_type,
            content: content.into(),
            embedding: None,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            source_id: None,
            source_type: None,
            confidence: 0.5,
            importance: 0.5,
            agent_id: None,
            session_id: None,
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    /// Create a pattern node.
    pub fn pattern(content: impl Into<String>) -> Self {
        Self::new(NodeType::Pattern, content)
    }

    /// Create a trajectory node.
    pub fn trajectory(content: impl Into<String>) -> Self {
        Self::new(NodeType::Trajectory, content)
    }

    /// Create a prediction node.
    pub fn prediction(content: impl Into<String>) -> Self {
        Self::new(NodeType::Prediction, content)
    }

    /// Create a decision node.
    pub fn decision(content: impl Into<String>) -> Self {
        Self::new(NodeType::Decision, content)
    }

    /// Set a specific ID (useful for testing or migration).
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Set the embedding vector.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set metadata.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set source reference.
    pub fn with_source(mut self, source_type: impl Into<String>, source_id: impl Into<String>) -> Self {
        self.source_type = Some(source_type.into());
        self.source_id = Some(source_id.into());
        self
    }

    /// Set confidence score.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set importance score.
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Set agent ID.
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Set session ID.
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set created_at timestamp.
    pub fn with_created_at(mut self, created_at: DateTime<Utc>) -> Self {
        self.created_at = created_at;
        self
    }

    /// Update the node (sets updated_at timestamp).
    pub fn touch(&mut self) {
        self.updated_at = Some(Utc::now());
    }

    /// Check if this node has an embedding.
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    /// Get the embedding dimension (or 0 if no embedding).
    pub fn embedding_dim(&self) -> usize {
        self.embedding.as_ref().map(|e| e.len()).unwrap_or(0)
    }

    /// Get the quality score (average of confidence and importance).
    pub fn quality_score(&self) -> f32 {
        (self.confidence + self.importance) / 2.0
    }
}

impl Default for ProfDAGNode {
    fn default() -> Self {
        Self::new(NodeType::Pattern, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_type_all() {
        let types = NodeType::all();
        assert_eq!(types.len(), 4);
        assert!(types.contains(&NodeType::Pattern));
        assert!(types.contains(&NodeType::Trajectory));
        assert!(types.contains(&NodeType::Prediction));
        assert!(types.contains(&NodeType::Decision));
    }

    #[test]
    fn test_node_type_from_str() {
        assert_eq!(NodeType::from_str("pattern"), Some(NodeType::Pattern));
        assert_eq!(NodeType::from_str("TRAJECTORY"), Some(NodeType::Trajectory));
        assert_eq!(NodeType::from_str("Prediction"), Some(NodeType::Prediction));
        assert_eq!(NodeType::from_str("decision"), Some(NodeType::Decision));
        assert_eq!(NodeType::from_str("unknown"), None);
    }

    #[test]
    fn test_node_type_as_str() {
        assert_eq!(NodeType::Pattern.as_str(), "pattern");
        assert_eq!(NodeType::Trajectory.as_str(), "trajectory");
        assert_eq!(NodeType::Prediction.as_str(), "prediction");
        assert_eq!(NodeType::Decision.as_str(), "decision");
    }

    #[test]
    fn test_node_creation() {
        let node = ProfDAGNode::pattern("Test pattern content");

        assert!(!node.id.is_empty());
        assert_eq!(node.node_type, NodeType::Pattern);
        assert_eq!(node.content, "Test pattern content");
        assert!(node.embedding.is_none());
        assert_eq!(node.confidence, 0.5);
        assert_eq!(node.importance, 0.5);
    }

    #[test]
    fn test_node_builder_pattern() {
        let embedding = vec![0.1, 0.2, 0.3];

        let node = ProfDAGNode::pattern("How to handle errors")
            .with_embedding(embedding.clone())
            .with_confidence(0.9)
            .with_importance(0.8)
            .with_source("reasoning_patterns", "pat_123")
            .with_agent_id("agent-1")
            .with_session_id("session-abc");

        assert_eq!(node.node_type, NodeType::Pattern);
        assert_eq!(node.content, "How to handle errors");
        assert_eq!(node.embedding, Some(embedding));
        assert_eq!(node.confidence, 0.9);
        assert_eq!(node.importance, 0.8);
        assert_eq!(node.source_type, Some("reasoning_patterns".to_string()));
        assert_eq!(node.source_id, Some("pat_123".to_string()));
        assert_eq!(node.agent_id, Some("agent-1".to_string()));
        assert_eq!(node.session_id, Some("session-abc".to_string()));
    }

    #[test]
    fn test_node_confidence_clamping() {
        let node = ProfDAGNode::pattern("test").with_confidence(1.5);
        assert_eq!(node.confidence, 1.0);

        let node = ProfDAGNode::pattern("test").with_confidence(-0.5);
        assert_eq!(node.confidence, 0.0);
    }

    #[test]
    fn test_node_quality_score() {
        let node = ProfDAGNode::pattern("test")
            .with_confidence(0.8)
            .with_importance(0.6);

        assert!((node.quality_score() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_node_touch() {
        let mut node = ProfDAGNode::pattern("test");
        assert!(node.updated_at.is_none());

        node.touch();
        assert!(node.updated_at.is_some());
    }

    #[test]
    fn test_node_has_embedding() {
        let node_without = ProfDAGNode::pattern("test");
        assert!(!node_without.has_embedding());
        assert_eq!(node_without.embedding_dim(), 0);

        let node_with = ProfDAGNode::pattern("test").with_embedding(vec![0.1; 128]);
        assert!(node_with.has_embedding());
        assert_eq!(node_with.embedding_dim(), 128);
    }

    #[test]
    fn test_node_factory_methods() {
        assert_eq!(ProfDAGNode::pattern("p").node_type, NodeType::Pattern);
        assert_eq!(ProfDAGNode::trajectory("t").node_type, NodeType::Trajectory);
        assert_eq!(ProfDAGNode::prediction("pr").node_type, NodeType::Prediction);
        assert_eq!(ProfDAGNode::decision("d").node_type, NodeType::Decision);
    }
}
