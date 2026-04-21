//! Graph namespace API for knowledge graph operations.
//!
//! The graph API provides methods for manipulating and querying the
//! knowledge graph that connects patterns, learnings, and predictions.
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::graph::{EdgeType, Direction};
//!
//! // Link two nodes
//! nagual.graph.link(&node_a, &node_b, EdgeType::RelatedTo, 0.8).await?;
//!
//! // Query neighbors
//! let result = nagual.graph.query(&node_a, Direction::Outgoing).await?;
//! for neighbor in result.neighbors {
//!     println!("{}: strength {}", neighbor.id, neighbor.strength);
//! }
//!
//! // Run pressure propagation
//! let pressure = nagual.graph.pressure(&start_node).await?;
//! for (node, score) in pressure.top_n(5) {
//!     println!("{}: {:.4}", node, score);
//! }
//! ```


use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

use super::NagualState;
use crate::error::{NagualError, Result};
use crate::graph::{
    propagate_pressure, Direction, EdgeType, GraphPath, InMemoryGraph,
    PressureConfig, GraphProvider,
};

/// Result of a graph query operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQueryResult {
    /// The source node ID
    pub source: String,

    /// Direction of the query
    pub direction: String,

    /// Neighboring nodes found
    pub neighbors: Vec<NeighborInfo>,

    /// Total neighbor count
    pub total_count: usize,

    /// Query execution time in milliseconds
    pub query_time_ms: u64,
}

/// Information about a neighboring node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborInfo {
    /// Node ID
    pub id: String,

    /// Edge type connecting to this neighbor
    pub edge_type: String,

    /// Edge strength (0.0 - 1.0)
    pub strength: f64,

    /// Edge metadata (if any)
    pub metadata: Option<serde_json::Value>,
}

/// Options for graph queries.
#[derive(Debug, Clone, Default)]
pub struct GraphQueryOptions {
    /// Filter by edge type
    pub edge_type: Option<EdgeType>,

    /// Minimum edge strength
    pub min_strength: Option<f64>,

    /// Maximum results to return
    pub limit: Option<usize>,
}

impl GraphQueryOptions {
    /// Create new options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by edge type.
    pub fn edge_type(mut self, edge_type: EdgeType) -> Self {
        self.edge_type = Some(edge_type);
        self
    }

    /// Set minimum strength.
    pub fn min_strength(mut self, strength: f64) -> Self {
        self.min_strength = Some(strength.clamp(0.0, 1.0));
        self
    }

    /// Set result limit.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Options for pressure propagation.
#[derive(Debug, Clone)]
pub struct PressureOptions {
    /// Damping factor (0.0 - 1.0)
    pub damping: f64,

    /// Maximum iterations
    pub max_iterations: usize,

    /// Whether to normalize results
    pub normalize: bool,
}

impl Default for PressureOptions {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 3,
            normalize: false,
        }
    }
}

impl PressureOptions {
    /// Create new options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set damping factor.
    pub fn damping(mut self, damping: f64) -> Self {
        self.damping = damping.clamp(0.0, 1.0);
        self
    }

    /// Set maximum iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations.max(1);
        self
    }

    /// Enable normalization.
    pub fn normalized(mut self) -> Self {
        self.normalize = true;
        self
    }

    /// Convert to PressureConfig.
    fn to_config(&self) -> PressureConfig {
        PressureConfig {
            damping_factor: self.damping,
            max_iterations: self.max_iterations,
            normalize: self.normalize,
            ..PressureConfig::default()
        }
    }
}

/// Pressure propagation result wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureScore {
    /// Node ID
    pub node_id: String,

    /// Pressure score
    pub score: f64,
}

/// Result of pressure propagation.
#[derive(Debug, Clone)]
pub struct GraphPressureResult {
    /// Source node
    pub source: String,

    /// Pressure scores by node
    pub scores: Vec<PressureScore>,

    /// Number of iterations used
    pub iterations: usize,

    /// Whether algorithm converged
    pub converged: bool,

    /// Execution time in microseconds
    pub execution_time_us: u64,
}

impl GraphPressureResult {
    /// Get top N nodes by pressure.
    pub fn top_n(&self, n: usize) -> Vec<(&str, f64)> {
        self.scores
            .iter()
            .take(n)
            .map(|s| (s.node_id.as_str(), s.score))
            .collect()
    }

    /// Get pressure for a specific node.
    pub fn get(&self, node_id: &str) -> Option<f64> {
        self.scores
            .iter()
            .find(|s| s.node_id == node_id)
            .map(|s| s.score)
    }
}

/// Knowledge graph operations API.
///
/// This API provides methods for manipulating and querying the knowledge
/// graph that connects patterns, learnings, and predictions.
#[derive(Clone)]
pub struct GraphApi {
    state: NagualState,
}

impl GraphApi {
    /// Create a new GraphApi instance.
    pub(crate) fn new(state: NagualState) -> Self {
        Self { state }
    }

    /// Create a link between two nodes.
    ///
    /// # Arguments
    ///
    /// * `source` - Source node ID
    /// * `target` - Target node ID
    /// * `edge_type` - Type of relationship
    /// * `strength` - Edge strength (0.0 - 1.0)
    ///
    /// # Returns
    ///
    /// The edge ID of the created link.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let edge_id = nagual.graph.link(
    ///     &pattern_a,
    ///     &pattern_b,
    ///     EdgeType::RelatedTo,
    ///     0.85
    /// ).await?;
    /// ```
    #[instrument(skip(self), fields(source = %source, target = %target, edge_type = %edge_type))]
    pub async fn link(
        &self,
        source: &str,
        target: &str,
        edge_type: EdgeType,
        strength: f64,
    ) -> Result<String> {
        self.link_with_metadata(source, target, edge_type, strength, None)
            .await
    }

    /// Create a link with metadata.
    ///
    /// # Arguments
    ///
    /// * `source` - Source node ID
    /// * `target` - Target node ID
    /// * `edge_type` - Type of relationship
    /// * `strength` - Edge strength (0.0 - 1.0)
    /// * `metadata` - Optional metadata to attach
    ///
    /// # Returns
    ///
    /// The edge ID of the created link.
    #[instrument(skip(self, metadata), fields(source = %source, target = %target))]
    pub async fn link_with_metadata(
        &self,
        source: &str,
        target: &str,
        edge_type: EdgeType,
        strength: f64,
        metadata: Option<serde_json::Value>,
    ) -> Result<String> {
        let result = self
            .state
            .graph_storage
            .create_edge(source, target, edge_type, strength, metadata)
            .await
            .map_err(|e| NagualError::internal(e.to_string()))?;

        info!(
            source = %source,
            target = %target,
            edge_type = %edge_type,
            strength = strength,
            "Graph edge created"
        );

        Ok(result.edge_id)
    }

    /// Query neighbors of a node.
    ///
    /// Note: This currently uses an in-memory graph built from edges.
    /// For large graphs, consider using direct database queries.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Node to query
    /// * `direction` - Direction to follow (Outgoing, Incoming, or Both)
    ///
    /// # Returns
    ///
    /// A `GraphQueryResult` containing neighbors.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = nagual.graph.query(&node_id, Direction::Outgoing).await?;
    /// for neighbor in result.neighbors {
    ///     println!("{}: {}", neighbor.id, neighbor.strength);
    /// }
    /// ```
    #[instrument(skip(self))]
    pub async fn query(&self, node_id: &str, direction: Direction) -> Result<GraphQueryResult> {
        self.query_with_options(node_id, direction, GraphQueryOptions::default())
            .await
    }

    /// Query neighbors with options.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Node to query
    /// * `direction` - Direction to follow
    /// * `options` - Query options
    ///
    /// # Returns
    ///
    /// A `GraphQueryResult` containing neighbors.
    #[instrument(skip(self, options))]
    pub async fn query_with_options(
        &self,
        node_id: &str,
        direction: Direction,
        options: GraphQueryOptions,
    ) -> Result<GraphQueryResult> {
        let start = std::time::Instant::now();

        // Build in-memory graph from edges and query it
        // In a full implementation, this would query the database directly
        let graph = self.build_graph_for_pressure(node_id, 1)?;

        let mut neighbors: Vec<NeighborInfo> = Vec::new();

        // Get neighbors from the in-memory graph
        if matches!(direction, Direction::Outgoing | Direction::Both) {
            for (neighbor_id, strength) in graph.get_neighbors(node_id) {
                if let Some(min_str) = options.min_strength {
                    if strength < min_str {
                        continue;
                    }
                }
                neighbors.push(NeighborInfo {
                    id: neighbor_id,
                    edge_type: "unknown".to_string(),
                    strength,
                    metadata: None,
                });
            }
        }

        // Apply limit
        if let Some(limit) = options.limit {
            neighbors.truncate(limit);
        }

        let total_count = neighbors.len();
        let query_time_ms = start.elapsed().as_millis() as u64;

        debug!(
            node_id = %node_id,
            direction = ?direction,
            neighbors = total_count,
            time_ms = query_time_ms,
            "Graph query completed"
        );

        Ok(GraphQueryResult {
            source: node_id.to_string(),
            direction: format!("{:?}", direction),
            neighbors,
            total_count,
            query_time_ms,
        })
    }

    /// Run pressure propagation from a starting node.
    ///
    /// Propagates "pressure" through the graph using a PageRank-style algorithm.
    ///
    /// # Arguments
    ///
    /// * `start_node` - Node to start propagation from
    ///
    /// # Returns
    ///
    /// A `GraphPressureResult` containing pressure scores.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = nagual.graph.pressure(&start_node).await?;
    /// for (node, score) in result.top_n(5) {
    ///     println!("{}: {:.4}", node, score);
    /// }
    /// ```
    #[instrument(skip(self))]
    pub async fn pressure(&self, start_node: &str) -> Result<GraphPressureResult> {
        self.pressure_with_options(start_node, PressureOptions::default())
            .await
    }

    /// Run pressure propagation with options.
    ///
    /// # Arguments
    ///
    /// * `start_node` - Node to start propagation from
    /// * `options` - Pressure propagation options
    ///
    /// # Returns
    ///
    /// A `GraphPressureResult` containing pressure scores.
    #[instrument(skip(self, options))]
    pub async fn pressure_with_options(
        &self,
        start_node: &str,
        options: PressureOptions,
    ) -> Result<GraphPressureResult> {
        // Build an in-memory graph from the storage
        // This is needed because pressure propagation uses the GraphProvider trait
        let graph = self.build_graph_for_pressure(start_node, options.max_iterations * 2)?;

        let config = options.to_config();

        let result = propagate_pressure(&graph, start_node, &config)
            .map_err(|e| NagualError::internal(e.to_string()))?;

        // Convert to sorted scores
        let mut scores: Vec<PressureScore> = result
            .pressure_scores
            .into_iter()
            .map(|(id, score)| PressureScore { node_id: id, score })
            .collect();

        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        info!(
            start_node = %start_node,
            iterations = result.stats.iterations_used,
            nodes_reached = result.stats.nodes_reached,
            converged = result.stats.converged,
            "Pressure propagation completed"
        );

        Ok(GraphPressureResult {
            source: start_node.to_string(),
            scores,
            iterations: result.stats.iterations_used,
            converged: result.stats.converged,
            execution_time_us: result.stats.execution_time_us,
        })
    }

    /// Build an in-memory graph for pressure propagation.
    fn build_graph_for_pressure(
        &self,
        _start_node: &str,
        _max_depth: usize,
    ) -> Result<InMemoryGraph> {
        // For now, return an empty graph
        // In a full implementation, this would query the database
        // and build a local subgraph for the algorithm
        Ok(InMemoryGraph::new())
    }

    /// Find paths between two nodes.
    ///
    /// Note: This uses an in-memory PathFinder. For large graphs, ensure
    /// edges are loaded first.
    ///
    /// # Arguments
    ///
    /// * `source` - Source node ID
    /// * `target` - Target node ID
    /// * `max_depth` - Maximum path length
    ///
    /// # Returns
    ///
    /// Vector of paths found.
    #[instrument(skip(self))]
    pub async fn find_paths(
        &self,
        source: &str,
        target: &str,
        max_depth: usize,
    ) -> Result<Vec<GraphPath>> {
        use crate::graph::PathFinder;

        // Build a PathFinder from edges
        // In a full implementation, edges would be loaded from the database
        let finder = PathFinder::new();

        let paths = finder.find_paths(source, target, max_depth)
            .map_err(|e| NagualError::internal(e.to_string()))?;

        debug!(
            source = %source,
            target = %target,
            paths_found = paths.len(),
            "Path search completed"
        );

        Ok(paths)
    }

    /// Delete an edge from the graph.
    ///
    /// # Arguments
    ///
    /// * `edge_id` - Edge ID to delete
    #[instrument(skip(self))]
    pub async fn unlink(&self, edge_id: &str) -> Result<bool> {
        let deleted = self.state.graph_storage.delete_edge(edge_id)
            .await
            .map_err(|e| NagualError::internal(e.to_string()))?;

        if deleted {
            info!(edge_id = %edge_id, "Graph edge deleted");
        }

        Ok(deleted)
    }

    /// Get graph statistics.
    ///
    /// # Returns
    ///
    /// Graph statistics including node and edge counts.
    pub async fn stats(&self) -> Result<GraphStats> {
        let storage_stats = self.state.graph_storage.stats()
            .await
            .map_err(|e| NagualError::internal(e.to_string()))?;

        Ok(GraphStats {
            node_count: storage_stats.node_count,
            edge_count: storage_stats.edge_count,
            edge_type_counts: storage_stats.edges_by_type.into_iter().collect(),
        })
    }

    /// Update edge strength by creating a new edge with the same type.
    ///
    /// # Arguments
    ///
    /// * `source` - Source node ID
    /// * `target` - Target node ID
    /// * `edge_type` - Edge type
    /// * `new_strength` - New strength value
    #[instrument(skip(self))]
    pub async fn update_strength(
        &self,
        source: &str,
        target: &str,
        edge_type: EdgeType,
        new_strength: f64,
    ) -> Result<()> {
        // Update by re-creating (upsert behavior)
        self.state
            .graph_storage
            .create_edge(source, target, edge_type, new_strength, None)
            .await
            .map_err(|e| NagualError::internal(e.to_string()))?;

        debug!(
            source = %source,
            target = %target,
            new_strength = new_strength,
            "Edge strength updated"
        );

        Ok(())
    }
}

/// Graph statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    /// Total node count
    pub node_count: usize,

    /// Total edge count
    pub edge_count: usize,

    /// Edge counts by type
    pub edge_type_counts: std::collections::HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_query_options_builder() {
        let options = GraphQueryOptions::new()
            .edge_type(EdgeType::SimilarTo)
            .min_strength(0.7)
            .limit(10);

        assert_eq!(options.edge_type, Some(EdgeType::SimilarTo));
        assert_eq!(options.min_strength, Some(0.7));
        assert_eq!(options.limit, Some(10));
    }

    #[test]
    fn test_pressure_options_builder() {
        let options = PressureOptions::new()
            .damping(0.9)
            .iterations(5)
            .normalized();

        assert_eq!(options.damping, 0.9);
        assert_eq!(options.max_iterations, 5);
        assert!(options.normalize);
    }

    #[test]
    fn test_pressure_options_clamping() {
        let options = PressureOptions::new()
            .damping(1.5) // Should clamp to 1.0
            .iterations(0); // Should clamp to 1

        assert_eq!(options.damping, 1.0);
        assert_eq!(options.max_iterations, 1);
    }

    #[test]
    fn test_pressure_result_top_n() {
        let result = GraphPressureResult {
            source: "A".to_string(),
            scores: vec![
                PressureScore { node_id: "A".to_string(), score: 0.5 },
                PressureScore { node_id: "B".to_string(), score: 0.3 },
                PressureScore { node_id: "C".to_string(), score: 0.2 },
            ],
            iterations: 3,
            converged: true,
            execution_time_us: 100,
        };

        let top = result.top_n(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "A");
        assert_eq!(top[1].0, "B");
    }

    #[test]
    fn test_pressure_result_get() {
        let result = GraphPressureResult {
            source: "A".to_string(),
            scores: vec![
                PressureScore { node_id: "A".to_string(), score: 0.5 },
            ],
            iterations: 1,
            converged: true,
            execution_time_us: 50,
        };

        assert_eq!(result.get("A"), Some(0.5));
        assert_eq!(result.get("B"), None);
    }
}
