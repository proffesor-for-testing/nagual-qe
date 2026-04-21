//! Context Graph Model for Nagual
//!
//! Implements a graph-based knowledge representation system for tracking
//! relationships between patterns, learnings, and predictions. The graph
//! enables:
//!
//! - Relationship discovery between knowledge entities
//! - Path-based reasoning across connected concepts
//! - Knowledge graph queries for context retrieval
//! - Co-retrieval pattern tracking for improved recommendations
//! - PageRank-style pressure propagation for influence analysis
//! - **Auto Edge Creation**: Automatic similar_to and co_retrieved edges (Phase 2.E)
//! - **Edge Maintenance**: Periodic pruning of weak edges (Phase 2.E)
//!
//! # Architecture
//!
//! The context graph uses an adjacency list representation stored in SQLite,
//! with indexes optimized for neighbor lookups and path traversal.
//!
//! # Modules
//!
//! - [`auto_edges`]: Automatic edge creation based on similarity and co-retrieval
//! - [`edges`]: Edge creation and management with upsert semantics
//! - [`maintenance`]: Edge pruning and scheduled maintenance jobs
//! - [`neighbors`]: Neighbor discovery with filtering and direction support
//! - [`pathfinding`]: BFS-based path finding with weighted paths
//! - [`pressure`]: PageRank-style pressure propagation algorithm
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::graph::{GraphStorage, EdgeType, Direction, PathQuery};
//!
//! // Create storage
//! let storage = GraphStorage::open("nagual.db")?;
//!
//! // Create an edge
//! storage.create_edge("pat_1", "pat_2", EdgeType::SimilarTo, 0.85, None)?;
//!
//! // Get neighbors
//! let neighbors = storage.get_neighbors("pat_1", Direction::Outgoing, None)?;
//!
//! // Find path
//! let paths = storage.find_paths("pat_1", "pat_3", 3)?;
//!
//! // Auto edge creation (Phase 2.E)
//! use nagual::graph::{AutoEdgeCreator, start_maintenance_scheduler};
//! let creator = AutoEdgeCreator::new(pg_pool.clone());
//! let result = creator.create_similar_edges(&pattern_id, &embedding).await?;
//! ```

pub mod auto_edges;
pub mod edges;
pub mod maintenance;
#[cfg(feature = "mincut")]
pub mod mincut;
pub mod neighbors;
pub mod pathfinding;
pub mod pressure;

pub use auto_edges::{
    AutoEdgeConfig, AutoEdgeCreator, AutoEdgeResult, CoRetrievalCandidate,
    CoRetrievalRecord, EdgeCreationReason, PatternEdge, PatternEdgeType,
};
pub use edges::{EdgeCreateResult, GraphStats, GraphStorage, GraphStorageConfig};
pub use maintenance::{
    EdgeMaintenanceConfig, EdgeMaintenanceJob, EdgeMaintenanceResult,
    MaintenanceSchedulerHandle, PruneResult, start_maintenance_scheduler,
};
pub use neighbors::{Direction, NeighborQuery, NeighborResult};
pub use pathfinding::{GraphPath, PathFinder, PathQuery};
pub use pressure::{
    propagate_pressure, GraphProvider, InMemoryGraph, PressureConfig,
    PressureError, PressureResult, PropagationStats,
};
#[cfg(feature = "mincut")]
pub use mincut::{Cluster, MinCutGraph};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Types of edges that can exist between graph nodes.
///
/// Each edge type represents a semantic relationship between two entities
/// in the knowledge graph (patterns, learnings, predictions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    /// Generic relationship - entities are related but relationship type is unknown
    RelatedTo,

    /// Derivation - target was derived/generated from source
    DerivedFrom,

    /// Supersession - target supersedes/replaces source
    Supersedes,

    /// Conflict - entities have conflicting information
    ConflictsWith,

    /// Support - target supports/reinforces source
    Supports,

    /// Contradiction - target contradicts source (stronger than conflict)
    Contradicts,

    /// Co-retrieval - entities are frequently retrieved together
    CoRetrieved,

    /// Similarity - entities are semantically similar
    SimilarTo,
}

impl EdgeType {
    /// Returns all edge types as a slice.
    pub fn all() -> &'static [EdgeType] {
        &[
            EdgeType::RelatedTo,
            EdgeType::DerivedFrom,
            EdgeType::Supersedes,
            EdgeType::ConflictsWith,
            EdgeType::Supports,
            EdgeType::Contradicts,
            EdgeType::CoRetrieved,
            EdgeType::SimilarTo,
        ]
    }

    /// Returns the inverse edge type for bidirectional relationships.
    ///
    /// Some edge types have natural inverses (e.g., DerivedFrom/Supersedes),
    /// while symmetric relationships (e.g., RelatedTo) return themselves.
    pub fn inverse(&self) -> EdgeType {
        match self {
            EdgeType::RelatedTo => EdgeType::RelatedTo,
            EdgeType::DerivedFrom => EdgeType::Supersedes,
            EdgeType::Supersedes => EdgeType::DerivedFrom,
            EdgeType::ConflictsWith => EdgeType::ConflictsWith,
            EdgeType::Supports => EdgeType::Supports,
            EdgeType::Contradicts => EdgeType::Contradicts,
            EdgeType::CoRetrieved => EdgeType::CoRetrieved,
            EdgeType::SimilarTo => EdgeType::SimilarTo,
        }
    }

    /// Returns whether this edge type is symmetric.
    pub fn is_symmetric(&self) -> bool {
        matches!(
            self,
            EdgeType::RelatedTo
                | EdgeType::ConflictsWith
                | EdgeType::CoRetrieved
                | EdgeType::SimilarTo
        )
    }

    /// Parse edge type from string.
    pub fn from_str(s: &str) -> Option<EdgeType> {
        match s.to_lowercase().as_str() {
            "related_to" | "relatedto" => Some(EdgeType::RelatedTo),
            "derived_from" | "derivedfrom" => Some(EdgeType::DerivedFrom),
            "supersedes" => Some(EdgeType::Supersedes),
            "conflicts_with" | "conflictswith" => Some(EdgeType::ConflictsWith),
            "supports" => Some(EdgeType::Supports),
            "contradicts" => Some(EdgeType::Contradicts),
            "co_retrieved" | "coretrieved" => Some(EdgeType::CoRetrieved),
            "similar_to" | "similarto" => Some(EdgeType::SimilarTo),
            _ => None,
        }
    }

    /// Convert to database string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeType::RelatedTo => "related_to",
            EdgeType::DerivedFrom => "derived_from",
            EdgeType::Supersedes => "supersedes",
            EdgeType::ConflictsWith => "conflicts_with",
            EdgeType::Supports => "supports",
            EdgeType::Contradicts => "contradicts",
            EdgeType::CoRetrieved => "co_retrieved",
            EdgeType::SimilarTo => "similar_to",
        }
    }
}

impl fmt::Display for EdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A node in the context graph.
///
/// Nodes represent references to entities in other tables (patterns, learnings,
/// predictions). The node itself is lightweight - full entity data should be
/// retrieved from the source table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique identifier for the node (matches entity ID in source table)
    pub id: String,

    /// Type of entity this node represents
    pub entity_type: EntityType,

    /// Optional label for display purposes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl GraphNode {
    /// Create a new graph node.
    pub fn new(id: impl Into<String>, entity_type: EntityType) -> Self {
        Self {
            id: id.into(),
            entity_type,
            label: None,
        }
    }

    /// Create a new graph node with a label.
    pub fn with_label(
        id: impl Into<String>,
        entity_type: EntityType,
        label: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            entity_type,
            label: Some(label.into()),
        }
    }

    /// Create a pattern node.
    pub fn pattern(id: impl Into<String>) -> Self {
        Self::new(id, EntityType::Pattern)
    }

    /// Create a learning node.
    pub fn learning(id: impl Into<String>) -> Self {
        Self::new(id, EntityType::Learning)
    }

    /// Create a prediction node.
    pub fn prediction(id: impl Into<String>) -> Self {
        Self::new(id, EntityType::Prediction)
    }
}

/// Types of entities that can be nodes in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    /// A pattern from the ReasoningBank
    Pattern,
    /// A learning/knowledge item
    Learning,
    /// A prediction
    Prediction,
}

impl EntityType {
    /// Parse entity type from string.
    pub fn from_str(s: &str) -> Option<EntityType> {
        match s.to_lowercase().as_str() {
            "pattern" => Some(EntityType::Pattern),
            "learning" => Some(EntityType::Learning),
            "prediction" => Some(EntityType::Prediction),
            _ => None,
        }
    }

    /// Convert to database string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityType::Pattern => "pattern",
            EntityType::Learning => "learning",
            EntityType::Prediction => "prediction",
        }
    }
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// An edge in the context graph connecting two nodes.
///
/// Edges are directional with a strength value indicating the relationship
/// strength. Metadata can store additional information about the relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Unique identifier for the edge
    pub id: String,

    /// Source node ID
    pub source_id: String,

    /// Target node ID
    pub target_id: String,

    /// Type of relationship
    pub edge_type: EdgeType,

    /// Relationship strength (0.0 to 1.0)
    ///
    /// Higher values indicate stronger relationships:
    /// - 0.0-0.3: Weak relationship
    /// - 0.3-0.7: Moderate relationship
    /// - 0.7-1.0: Strong relationship
    pub strength: f64,

    /// Optional metadata as JSON
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,

    /// When the edge was created
    pub created_at: DateTime<Utc>,

    /// When the edge was last updated
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
}

impl GraphEdge {
    /// Create a new edge with the given properties.
    pub fn new(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        edge_type: EdgeType,
        strength: f64,
    ) -> Self {
        let now = Utc::now();
        let source = source_id.into();
        let target = target_id.into();
        let id = format!(
            "edge_{}_{}_{}_{}",
            source,
            target,
            edge_type.as_str(),
            now.timestamp_millis()
        );

        Self {
            id,
            source_id: source,
            target_id: target,
            edge_type,
            strength: strength.clamp(0.0, 1.0),
            metadata: None,
            created_at: now,
            updated_at: None,
        }
    }

    /// Create a new edge with metadata.
    pub fn with_metadata(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        edge_type: EdgeType,
        strength: f64,
        metadata: serde_json::Value,
    ) -> Self {
        let mut edge = Self::new(source_id, target_id, edge_type, strength);
        edge.metadata = Some(metadata);
        edge
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
}

/// Error types for graph operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum GraphError {
    /// Edge strength must be between 0.0 and 1.0
    #[error("Invalid edge strength {0}: must be between 0.0 and 1.0")]
    InvalidStrength(f64),

    /// Self-loops are not allowed
    #[error("Self-loops are not allowed: source and target are both '{0}'")]
    SelfLoop(String),

    /// Node not found
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// Edge not found
    #[error("Edge not found: {0}")]
    EdgeNotFound(String),

    /// Path not found between nodes
    #[error("No path found from '{0}' to '{1}'")]
    PathNotFound(String, String),

    /// Maximum depth exceeded
    #[error("Maximum depth {0} exceeded during traversal")]
    MaxDepthExceeded(usize),

    /// Database error
    #[error("Database error: {0}")]
    Database(String),

    /// Invalid edge type
    #[error("Invalid edge type: {0}")]
    InvalidEdgeType(String),

    /// Invalid entity type
    #[error("Invalid entity type: {0}")]
    InvalidEntityType(String),
}

impl From<rusqlite::Error> for GraphError {
    fn from(err: rusqlite::Error) -> Self {
        GraphError::Database(err.to_string())
    }
}

/// SQL for creating the context_graph table in SQLite.
pub const SQLITE_CONTEXT_GRAPH_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS context_graph (
    -- Primary key: Unique edge identifier
    id TEXT PRIMARY KEY,

    -- Source node ID
    source_id TEXT NOT NULL,

    -- Target node ID
    target_id TEXT NOT NULL,

    -- Edge type (one of the EdgeType enum values)
    edge_type TEXT NOT NULL,

    -- Relationship strength (0.0 to 1.0)
    strength REAL NOT NULL CHECK (strength >= 0.0 AND strength <= 1.0),

    -- Optional metadata as JSON
    metadata TEXT,

    -- Timestamps
    created_at TEXT NOT NULL,
    updated_at TEXT,

    -- Prevent duplicate edges between same nodes with same type
    UNIQUE (source_id, target_id, edge_type)
);

-- Index for outgoing edges from a node (source_id, edge_type)
CREATE INDEX IF NOT EXISTS idx_context_graph_source_type
    ON context_graph (source_id, edge_type);

-- Index for incoming edges to a node (target_id, edge_type)
CREATE INDEX IF NOT EXISTS idx_context_graph_target_type
    ON context_graph (target_id, edge_type);

-- Index for strength-based pruning queries
CREATE INDEX IF NOT EXISTS idx_context_graph_strength
    ON context_graph (strength DESC);

-- Index for finding all edges of a specific type
CREATE INDEX IF NOT EXISTS idx_context_graph_edge_type
    ON context_graph (edge_type);

-- Index for finding all edges involving a specific node
CREATE INDEX IF NOT EXISTS idx_context_graph_source
    ON context_graph (source_id);

CREATE INDEX IF NOT EXISTS idx_context_graph_target
    ON context_graph (target_id);
"#;

/// SQL for creating the context_graph table in PostgreSQL.
pub const POSTGRES_CONTEXT_GRAPH_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS context_graph (
    -- Primary key: Unique edge identifier
    id TEXT PRIMARY KEY,

    -- Source node ID
    source_id TEXT NOT NULL,

    -- Target node ID
    target_id TEXT NOT NULL,

    -- Edge type (one of the EdgeType enum values)
    edge_type TEXT NOT NULL,

    -- Relationship strength (0.0 to 1.0)
    strength FLOAT NOT NULL CHECK (strength >= 0.0 AND strength <= 1.0),

    -- Optional metadata as JSONB
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,

    -- Prevent duplicate edges between same nodes with same type
    UNIQUE (source_id, target_id, edge_type)
);

-- Index for outgoing edges from a node (source_id, edge_type)
CREATE INDEX IF NOT EXISTS idx_context_graph_source_type
    ON context_graph (source_id, edge_type);

-- Index for incoming edges to a node (target_id, edge_type)
CREATE INDEX IF NOT EXISTS idx_context_graph_target_type
    ON context_graph (target_id, edge_type);

-- Index for strength-based pruning queries
CREATE INDEX IF NOT EXISTS idx_context_graph_strength
    ON context_graph (strength DESC);

-- Index for finding all edges of a specific type
CREATE INDEX IF NOT EXISTS idx_context_graph_edge_type
    ON context_graph (edge_type);

-- Index for finding all edges involving a specific node
CREATE INDEX IF NOT EXISTS idx_context_graph_source
    ON context_graph (source_id);

CREATE INDEX IF NOT EXISTS idx_context_graph_target
    ON context_graph (target_id);

-- Table comment
COMMENT ON TABLE context_graph IS 'Context graph for tracking relationships between patterns, learnings, and predictions';

-- Column comments
COMMENT ON COLUMN context_graph.source_id IS 'Source node ID (references patterns.id, learnings.id, or predictions.id)';
COMMENT ON COLUMN context_graph.target_id IS 'Target node ID (references patterns.id, learnings.id, or predictions.id)';
COMMENT ON COLUMN context_graph.edge_type IS 'Type of relationship: related_to, derived_from, supersedes, conflicts_with, supports, contradicts, co_retrieved, similar_to';
COMMENT ON COLUMN context_graph.strength IS 'Relationship strength from 0.0 (weak) to 1.0 (strong)';
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_type_inverse() {
        assert_eq!(EdgeType::DerivedFrom.inverse(), EdgeType::Supersedes);
        assert_eq!(EdgeType::Supersedes.inverse(), EdgeType::DerivedFrom);
        assert_eq!(EdgeType::RelatedTo.inverse(), EdgeType::RelatedTo);
        assert_eq!(EdgeType::ConflictsWith.inverse(), EdgeType::ConflictsWith);
    }

    #[test]
    fn test_edge_type_symmetric() {
        assert!(EdgeType::RelatedTo.is_symmetric());
        assert!(EdgeType::ConflictsWith.is_symmetric());
        assert!(EdgeType::CoRetrieved.is_symmetric());
        assert!(EdgeType::SimilarTo.is_symmetric());
        assert!(!EdgeType::DerivedFrom.is_symmetric());
        assert!(!EdgeType::Supersedes.is_symmetric());
    }

    #[test]
    fn test_edge_type_from_str() {
        assert_eq!(EdgeType::from_str("related_to"), Some(EdgeType::RelatedTo));
        assert_eq!(
            EdgeType::from_str("DERIVED_FROM"),
            Some(EdgeType::DerivedFrom)
        );
        assert_eq!(EdgeType::from_str("unknown"), None);
    }

    #[test]
    fn test_graph_node_creation() {
        let node = GraphNode::pattern("pat_123");
        assert_eq!(node.id, "pat_123");
        assert_eq!(node.entity_type, EntityType::Pattern);
        assert!(node.label.is_none());

        let labeled = GraphNode::with_label("learn_456", EntityType::Learning, "My Learning");
        assert_eq!(labeled.label, Some("My Learning".to_string()));
    }

    #[test]
    fn test_graph_edge_creation() {
        let edge = GraphEdge::new("a", "b", EdgeType::RelatedTo, 0.8);
        assert_eq!(edge.source_id, "a");
        assert_eq!(edge.target_id, "b");
        assert_eq!(edge.edge_type, EdgeType::RelatedTo);
        assert!((edge.strength - 0.8).abs() < f64::EPSILON);
        assert!(!edge.is_self_loop());
    }

    #[test]
    fn test_graph_edge_strength_clamping() {
        let edge_high = GraphEdge::new("a", "b", EdgeType::RelatedTo, 1.5);
        assert!((edge_high.strength - 1.0).abs() < f64::EPSILON);

        let edge_low = GraphEdge::new("a", "b", EdgeType::RelatedTo, -0.5);
        assert!(edge_low.strength.abs() < f64::EPSILON);
    }

    #[test]
    fn test_graph_edge_self_loop() {
        let edge = GraphEdge::new("a", "a", EdgeType::RelatedTo, 0.5);
        assert!(edge.is_self_loop());
    }

    #[test]
    fn test_graph_edge_other_node() {
        let edge = GraphEdge::new("a", "b", EdgeType::RelatedTo, 0.5);
        assert_eq!(edge.other_node("a"), Some("b"));
        assert_eq!(edge.other_node("b"), Some("a"));
        assert_eq!(edge.other_node("c"), None);
    }

    #[test]
    fn test_entity_type_from_str() {
        assert_eq!(EntityType::from_str("pattern"), Some(EntityType::Pattern));
        assert_eq!(
            EntityType::from_str("LEARNING"),
            Some(EntityType::Learning)
        );
        assert_eq!(
            EntityType::from_str("prediction"),
            Some(EntityType::Prediction)
        );
        assert_eq!(EntityType::from_str("unknown"), None);
    }
}
