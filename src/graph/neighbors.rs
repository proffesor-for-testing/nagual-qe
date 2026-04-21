//! Neighbor discovery for the context graph.
//!
//! Provides efficient neighbor lookups with filtering and direction support.

use serde::{Deserialize, Serialize};
use super::{EdgeType, GraphEdge};

/// Direction for neighbor queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// Outgoing edges (source -> target)
    Outgoing,
    /// Incoming edges (target -> source)
    Incoming,
    /// Both directions
    Both,
}

impl Default for Direction {
    fn default() -> Self {
        Direction::Both
    }
}

/// Query parameters for neighbor lookups.
#[derive(Debug, Clone)]
pub struct NeighborQuery {
    /// Node ID to find neighbors for
    pub node_id: String,
    /// Direction of edges to follow
    pub direction: Direction,
    /// Filter by edge type
    pub edge_type: Option<EdgeType>,
    /// Minimum edge strength
    pub min_strength: Option<f64>,
    /// Maximum results to return
    pub limit: Option<usize>,
}

impl NeighborQuery {
    /// Create a new neighbor query for the given node.
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            direction: Direction::Both,
            edge_type: None,
            min_strength: None,
            limit: None,
        }
    }

    /// Set the direction.
    pub fn direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    /// Filter by edge type.
    pub fn edge_type(mut self, edge_type: EdgeType) -> Self {
        self.edge_type = Some(edge_type);
        self
    }

    /// Set minimum strength.
    pub fn min_strength(mut self, strength: f64) -> Self {
        self.min_strength = Some(strength);
        self
    }

    /// Set result limit.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Result of a neighbor query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborResult {
    /// The neighbor node ID
    pub neighbor_id: String,
    /// The connecting edge
    pub edge: GraphEdge,
    /// Whether this is an outgoing edge
    pub is_outgoing: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neighbor_query_builder() {
        let query = NeighborQuery::new("node_1")
            .direction(Direction::Outgoing)
            .edge_type(EdgeType::SimilarTo)
            .min_strength(0.5)
            .limit(10);

        assert_eq!(query.node_id, "node_1");
        assert_eq!(query.direction, Direction::Outgoing);
        assert_eq!(query.edge_type, Some(EdgeType::SimilarTo));
        assert_eq!(query.min_strength, Some(0.5));
        assert_eq!(query.limit, Some(10));
    }
}
