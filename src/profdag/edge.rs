//! ProfDAG Edge types and structures.
//!
//! Defines the edge types and data structures for the ProfDAG graph.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Types of edges in the ProfDAG.
///
/// Each edge type represents a different kind of relationship
/// between nodes in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    /// Causal/temporal progression (A leads to B)
    LeadsTo,

    /// Semantic similarity based on embeddings
    SimilarTo,

    /// Derivation relationship (B was derived from A)
    DerivedFrom,

    /// Non-local connection bridging distant graph regions
    Wormhole,

    /// Time-based relationship with direction
    TemporalLink,
}

impl EdgeType {
    /// Returns all edge types as a slice.
    pub fn all() -> &'static [EdgeType] {
        &[
            EdgeType::LeadsTo,
            EdgeType::SimilarTo,
            EdgeType::DerivedFrom,
            EdgeType::Wormhole,
            EdgeType::TemporalLink,
        ]
    }

    /// Parse edge type from string.
    pub fn from_str(s: &str) -> Option<EdgeType> {
        match s.to_lowercase().as_str() {
            "leads_to" | "leadsto" => Some(EdgeType::LeadsTo),
            "similar_to" | "similarto" => Some(EdgeType::SimilarTo),
            "derived_from" | "derivedfrom" => Some(EdgeType::DerivedFrom),
            "wormhole" => Some(EdgeType::Wormhole),
            "temporal_link" | "temporallink" => Some(EdgeType::TemporalLink),
            _ => None,
        }
    }

    /// Convert to database string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeType::LeadsTo => "leads_to",
            EdgeType::SimilarTo => "similar_to",
            EdgeType::DerivedFrom => "derived_from",
            EdgeType::Wormhole => "wormhole",
            EdgeType::TemporalLink => "temporal_link",
        }
    }

    /// Returns whether this edge type is symmetric.
    ///
    /// Symmetric edges have the same meaning in both directions.
    pub fn is_symmetric(&self) -> bool {
        matches!(self, EdgeType::SimilarTo)
    }

    /// Returns the inverse edge type for directed relationships.
    ///
    /// Some edge types have natural inverses, while symmetric relationships
    /// return themselves.
    pub fn inverse(&self) -> EdgeType {
        match self {
            EdgeType::LeadsTo => EdgeType::DerivedFrom, // Inverse: was led to by
            EdgeType::SimilarTo => EdgeType::SimilarTo, // Symmetric
            EdgeType::DerivedFrom => EdgeType::LeadsTo, // Inverse: leads to
            EdgeType::Wormhole => EdgeType::Wormhole,   // Symmetric (bidirectional)
            EdgeType::TemporalLink => EdgeType::TemporalLink, // Direction is in metadata
        }
    }
}

impl fmt::Display for EdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Direction of temporal relationships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalDirection {
    /// Source occurred before target
    Forward,

    /// Source occurred after target
    Backward,

    /// Source and target occurred at approximately the same time
    Concurrent,
}

impl TemporalDirection {
    /// Parse temporal direction from string.
    pub fn from_str(s: &str) -> Option<TemporalDirection> {
        match s.to_lowercase().as_str() {
            "forward" => Some(TemporalDirection::Forward),
            "backward" => Some(TemporalDirection::Backward),
            "concurrent" => Some(TemporalDirection::Concurrent),
            _ => None,
        }
    }

    /// Convert to database string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            TemporalDirection::Forward => "forward",
            TemporalDirection::Backward => "backward",
            TemporalDirection::Concurrent => "concurrent",
        }
    }
}

impl fmt::Display for TemporalDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// An edge in the ProfDAG connecting two nodes.
///
/// Edges are directional with a weight indicating relationship strength.
/// Different edge types have specific metadata fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfDAGEdge {
    /// Unique identifier (UUID)
    pub id: String,

    /// Source node ID
    pub source_id: String,

    /// Target node ID
    pub target_id: String,

    /// Type of relationship
    pub edge_type: EdgeType,

    /// Relationship weight/strength (0.0 to 1.0)
    ///
    /// Higher values indicate stronger relationships:
    /// - 0.0-0.3: Weak relationship
    /// - 0.3-0.7: Moderate relationship
    /// - 0.7-1.0: Strong relationship
    pub weight: f64,

    /// Optional metadata as JSON
    #[serde(default)]
    pub metadata: serde_json::Value,

    // ========== Temporal link metadata ==========
    /// For temporal_link: time difference in hours between nodes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temporal_distance_hours: Option<i32>,

    /// For temporal_link: direction of time flow
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temporal_direction: Option<TemporalDirection>,

    // ========== Similarity metadata ==========
    /// For similar_to: cosine similarity between embeddings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_score: Option<f64>,

    // ========== Wormhole metadata ==========
    /// For wormhole: strength of the non-local connection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wormhole_strength: Option<f64>,

    /// For wormhole: explanation of why nodes are connected
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wormhole_reason: Option<String>,

    // ========== Timestamps ==========
    /// When the edge was created
    pub created_at: DateTime<Utc>,

    /// When the edge was last updated
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
}

impl ProfDAGEdge {
    /// Create a new edge with the given properties.
    pub fn new(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        edge_type: EdgeType,
        weight: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_id: source_id.into(),
            target_id: target_id.into(),
            edge_type,
            weight: weight.clamp(0.0, 1.0),
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            temporal_distance_hours: None,
            temporal_direction: None,
            similarity_score: None,
            wormhole_strength: None,
            wormhole_reason: None,
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    /// Create a leads_to edge.
    pub fn leads_to(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        weight: f64,
    ) -> Self {
        Self::new(source_id, target_id, EdgeType::LeadsTo, weight)
    }

    /// Create a similar_to edge.
    pub fn similar_to(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        similarity: f64,
    ) -> Self {
        let mut edge = Self::new(source_id, target_id, EdgeType::SimilarTo, similarity);
        edge.similarity_score = Some(similarity);
        edge
    }

    /// Create a derived_from edge.
    pub fn derived_from(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        weight: f64,
    ) -> Self {
        Self::new(source_id, target_id, EdgeType::DerivedFrom, weight)
    }

    /// Create a wormhole edge.
    pub fn wormhole(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        strength: f64,
        reason: impl Into<String>,
    ) -> Self {
        let mut edge = Self::new(source_id, target_id, EdgeType::Wormhole, strength);
        edge.wormhole_strength = Some(strength);
        edge.wormhole_reason = Some(reason.into());
        edge
    }

    /// Create a temporal_link edge.
    pub fn temporal_link(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        distance_hours: i32,
        direction: TemporalDirection,
    ) -> Self {
        let weight = 1.0 / (1.0 + (distance_hours.abs() as f64 / 24.0)); // Decay by day
        let mut edge = Self::new(source_id, target_id, EdgeType::TemporalLink, weight);
        edge.temporal_distance_hours = Some(distance_hours);
        edge.temporal_direction = Some(direction);
        edge
    }

    /// Set a specific ID (useful for testing or migration).
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Set metadata.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Check if this edge is a self-loop.
    pub fn is_self_loop(&self) -> bool {
        self.source_id == self.target_id
    }

    /// Get the other node ID given one node ID.
    pub fn other_node(&self, node_id: &str) -> Option<&str> {
        if self.source_id == node_id {
            Some(&self.target_id)
        } else if self.target_id == node_id {
            Some(&self.source_id)
        } else {
            None
        }
    }

    /// Update the edge (sets updated_at timestamp).
    pub fn touch(&mut self) {
        self.updated_at = Some(Utc::now());
    }

    /// Get whether this edge represents a strong relationship (weight >= 0.7).
    pub fn is_strong(&self) -> bool {
        self.weight >= 0.7
    }

    /// Get whether this edge represents a weak relationship (weight < 0.3).
    pub fn is_weak(&self) -> bool {
        self.weight < 0.3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_type_all() {
        let types = EdgeType::all();
        assert_eq!(types.len(), 5);
        assert!(types.contains(&EdgeType::LeadsTo));
        assert!(types.contains(&EdgeType::SimilarTo));
        assert!(types.contains(&EdgeType::DerivedFrom));
        assert!(types.contains(&EdgeType::Wormhole));
        assert!(types.contains(&EdgeType::TemporalLink));
    }

    #[test]
    fn test_edge_type_from_str() {
        assert_eq!(EdgeType::from_str("leads_to"), Some(EdgeType::LeadsTo));
        assert_eq!(EdgeType::from_str("SIMILAR_TO"), Some(EdgeType::SimilarTo));
        assert_eq!(EdgeType::from_str("derived_from"), Some(EdgeType::DerivedFrom));
        assert_eq!(EdgeType::from_str("wormhole"), Some(EdgeType::Wormhole));
        assert_eq!(EdgeType::from_str("temporal_link"), Some(EdgeType::TemporalLink));
        assert_eq!(EdgeType::from_str("unknown"), None);
    }

    #[test]
    fn test_edge_type_symmetric() {
        assert!(!EdgeType::LeadsTo.is_symmetric());
        assert!(EdgeType::SimilarTo.is_symmetric());
        assert!(!EdgeType::DerivedFrom.is_symmetric());
    }

    #[test]
    fn test_edge_type_inverse() {
        assert_eq!(EdgeType::LeadsTo.inverse(), EdgeType::DerivedFrom);
        assert_eq!(EdgeType::DerivedFrom.inverse(), EdgeType::LeadsTo);
        assert_eq!(EdgeType::SimilarTo.inverse(), EdgeType::SimilarTo);
    }

    #[test]
    fn test_temporal_direction_from_str() {
        assert_eq!(TemporalDirection::from_str("forward"), Some(TemporalDirection::Forward));
        assert_eq!(TemporalDirection::from_str("BACKWARD"), Some(TemporalDirection::Backward));
        assert_eq!(TemporalDirection::from_str("concurrent"), Some(TemporalDirection::Concurrent));
        assert_eq!(TemporalDirection::from_str("unknown"), None);
    }

    #[test]
    fn test_edge_creation() {
        let edge = ProfDAGEdge::new("node1", "node2", EdgeType::LeadsTo, 0.8);

        assert!(!edge.id.is_empty());
        assert_eq!(edge.source_id, "node1");
        assert_eq!(edge.target_id, "node2");
        assert_eq!(edge.edge_type, EdgeType::LeadsTo);
        assert!((edge.weight - 0.8).abs() < f64::EPSILON);
        assert!(!edge.is_self_loop());
    }

    #[test]
    fn test_edge_weight_clamping() {
        let edge_high = ProfDAGEdge::new("a", "b", EdgeType::LeadsTo, 1.5);
        assert!((edge_high.weight - 1.0).abs() < f64::EPSILON);

        let edge_low = ProfDAGEdge::new("a", "b", EdgeType::LeadsTo, -0.5);
        assert!(edge_low.weight.abs() < f64::EPSILON);
    }

    #[test]
    fn test_edge_factory_similar_to() {
        let edge = ProfDAGEdge::similar_to("a", "b", 0.95);

        assert_eq!(edge.edge_type, EdgeType::SimilarTo);
        assert_eq!(edge.similarity_score, Some(0.95));
        assert!((edge.weight - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_edge_factory_wormhole() {
        let edge = ProfDAGEdge::wormhole("a", "b", 0.7, "High semantic overlap despite distance");

        assert_eq!(edge.edge_type, EdgeType::Wormhole);
        assert_eq!(edge.wormhole_strength, Some(0.7));
        assert_eq!(edge.wormhole_reason, Some("High semantic overlap despite distance".to_string()));
    }

    #[test]
    fn test_edge_factory_temporal_link() {
        let edge = ProfDAGEdge::temporal_link("a", "b", 48, TemporalDirection::Forward);

        assert_eq!(edge.edge_type, EdgeType::TemporalLink);
        assert_eq!(edge.temporal_distance_hours, Some(48));
        assert_eq!(edge.temporal_direction, Some(TemporalDirection::Forward));
        // Weight should decay with distance
        assert!(edge.weight < 1.0);
        assert!(edge.weight > 0.0);
    }

    #[test]
    fn test_edge_other_node() {
        let edge = ProfDAGEdge::new("a", "b", EdgeType::LeadsTo, 0.5);

        assert_eq!(edge.other_node("a"), Some("b"));
        assert_eq!(edge.other_node("b"), Some("a"));
        assert_eq!(edge.other_node("c"), None);
    }

    #[test]
    fn test_edge_strength_classification() {
        let strong = ProfDAGEdge::new("a", "b", EdgeType::LeadsTo, 0.8);
        assert!(strong.is_strong());
        assert!(!strong.is_weak());

        let weak = ProfDAGEdge::new("a", "b", EdgeType::LeadsTo, 0.2);
        assert!(!weak.is_strong());
        assert!(weak.is_weak());

        let moderate = ProfDAGEdge::new("a", "b", EdgeType::LeadsTo, 0.5);
        assert!(!moderate.is_strong());
        assert!(!moderate.is_weak());
    }

    #[test]
    fn test_edge_touch() {
        let mut edge = ProfDAGEdge::new("a", "b", EdgeType::LeadsTo, 0.5);
        assert!(edge.updated_at.is_none());

        edge.touch();
        assert!(edge.updated_at.is_some());
    }
}
