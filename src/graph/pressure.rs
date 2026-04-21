//! PageRank-style pressure propagation algorithm for GNN influence analysis.
//!
//! Implements a pressure propagation algorithm that spreads influence from a starting
//! node through the graph, similar to PageRank but for directed influence measurement.
//!
//! # Algorithm
//!
//! The algorithm works by iteratively distributing "pressure" from each node to its
//! neighbors based on edge weights and a damping factor:
//!
//! ```text
//! pressure[start] = 1.0
//! for iteration in 1..=max_iterations:
//!     new_pressure = {}
//!     for node, score in pressure:
//!         for neighbor, edge_strength in get_neighbors(node):
//!             transfer = score * damping * edge_strength
//!             new_pressure[neighbor] += transfer
//!         new_pressure[node] += score * (1 - damping)
//!     if converged(pressure, new_pressure):
//!         break
//!     pressure = new_pressure
//! return pressure
//! ```
//!
//! # Example
//!
//! ```rust
//! use nagual::graph::pressure::{propagate_pressure, PressureConfig, GraphProvider};
//! use hashbrown::HashMap;
//!
//! // Create a simple in-memory graph
//! struct SimpleGraph {
//!     edges: HashMap<String, Vec<(String, f64)>>,
//! }
//!
//! impl GraphProvider for SimpleGraph {
//!     fn get_neighbors(&self, node_id: &str) -> Vec<(String, f64)> {
//!         self.edges.get(node_id).cloned().unwrap_or_default()
//!     }
//!
//!     fn node_exists(&self, node_id: &str) -> bool {
//!         self.edges.contains_key(node_id)
//!     }
//! }
//!
//! let mut edges = HashMap::new();
//! edges.insert("A".to_string(), vec![("B".to_string(), 0.8), ("C".to_string(), 0.5)]);
//! edges.insert("B".to_string(), vec![("C".to_string(), 0.9)]);
//! edges.insert("C".to_string(), vec![]);
//!
//! let graph = SimpleGraph { edges };
//! let config = PressureConfig::default();
//! let result = propagate_pressure(&graph, "A", &config).unwrap();
//!
//! assert!(result.pressure_scores.contains_key("A"));
//! assert!(result.pressure_scores.contains_key("B"));
//! ```

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

/// Configuration for pressure propagation algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureConfig {
    /// Damping factor (0.0 to 1.0).
    /// Controls how much pressure transfers to neighbors.
    /// Default: 0.85 (similar to PageRank).
    pub damping_factor: f64,

    /// Maximum number of iterations.
    /// Default: 3 for shallow propagation.
    pub max_iterations: usize,

    /// Convergence threshold (epsilon).
    /// Algorithm stops early if max change is below this value.
    /// Default: 1e-6.
    pub epsilon: f64,

    /// Minimum pressure threshold to include in results.
    /// Nodes with pressure below this are excluded from output.
    /// Default: 1e-9.
    pub min_pressure_threshold: f64,

    /// Maximum number of nodes to process (prevents explosion in large graphs).
    /// Default: 10000.
    pub max_nodes: usize,

    /// Whether to normalize the final pressure scores.
    /// Default: false (keeps absolute influence values).
    pub normalize: bool,
}

impl Default for PressureConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 3,
            epsilon: 1e-6,
            min_pressure_threshold: 1e-9,
            max_nodes: 10000,
            normalize: false,
        }
    }
}

impl PressureConfig {
    /// Create a new configuration with custom damping factor.
    pub fn with_damping(damping_factor: f64) -> Self {
        Self {
            damping_factor: damping_factor.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Set the damping factor.
    pub fn damping(mut self, factor: f64) -> Self {
        self.damping_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum iterations.
    pub fn iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set the convergence epsilon.
    pub fn epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps.max(0.0);
        self
    }

    /// Enable normalization of output scores.
    pub fn normalized(mut self) -> Self {
        self.normalize = true;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), PressureError> {
        if self.damping_factor < 0.0 || self.damping_factor > 1.0 {
            return Err(PressureError::InvalidConfig {
                message: format!(
                    "Damping factor must be between 0.0 and 1.0, got {}",
                    self.damping_factor
                ),
            });
        }
        if self.max_iterations == 0 {
            return Err(PressureError::InvalidConfig {
                message: "Max iterations must be at least 1".to_string(),
            });
        }
        if self.epsilon < 0.0 {
            return Err(PressureError::InvalidConfig {
                message: "Epsilon must be non-negative".to_string(),
            });
        }
        Ok(())
    }
}

/// Statistics from pressure propagation execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationStats {
    /// Number of iterations performed.
    pub iterations_used: usize,

    /// Whether the algorithm converged before max iterations.
    pub converged: bool,

    /// Maximum change in the final iteration.
    pub final_delta: f64,

    /// Number of nodes with non-zero pressure.
    pub nodes_reached: usize,

    /// Total pressure distributed (should be close to 1.0 if normalized).
    pub total_pressure: f64,

    /// Execution time in microseconds.
    pub execution_time_us: u64,
}

/// Result of pressure propagation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureResult {
    /// Map of node_id -> pressure_score.
    pub pressure_scores: HashMap<String, f64>,

    /// Starting node for the propagation.
    pub source_node: String,

    /// Execution statistics.
    pub stats: PropagationStats,

    /// Configuration used for this propagation.
    pub config: PressureConfig,
}

impl PressureResult {
    /// Get the top N nodes by pressure score.
    pub fn top_n(&self, n: usize) -> Vec<(&String, &f64)> {
        let mut sorted: Vec<_> = self.pressure_scores.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(n).collect()
    }

    /// Get pressure score for a specific node.
    pub fn get_pressure(&self, node_id: &str) -> Option<f64> {
        self.pressure_scores.get(node_id).copied()
    }

    /// Check if a node was influenced.
    pub fn is_influenced(&self, node_id: &str) -> bool {
        self.pressure_scores.contains_key(node_id)
    }
}

/// Error types for pressure propagation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum PressureError {
    /// Invalid configuration.
    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },

    /// Node not found in graph.
    #[error("Node not found: {node_id}")]
    NodeNotFound { node_id: String },

    /// Graph limit exceeded.
    #[error("Max nodes limit exceeded: {count} > {limit}")]
    LimitExceeded { count: usize, limit: usize },
}

/// Trait for providing graph connectivity information.
///
/// Implement this trait to connect the pressure propagation algorithm
/// to your graph storage backend.
pub trait GraphProvider {
    /// Get the neighbors of a node with their edge weights.
    ///
    /// Returns a vector of (neighbor_id, edge_weight) tuples.
    /// Edge weights should be in range [0.0, 1.0] for best results.
    fn get_neighbors(&self, node_id: &str) -> Vec<(String, f64)>;

    /// Check if a node exists in the graph.
    fn node_exists(&self, node_id: &str) -> bool;
}

/// Propagate pressure from a starting node through the graph.
///
/// This implements a PageRank-style algorithm where "pressure" flows from
/// the source node to its neighbors based on edge weights and damping factor.
///
/// # Arguments
///
/// * `graph` - Graph provider implementing node/edge access
/// * `start_node` - The source node to propagate from
/// * `config` - Configuration for the propagation algorithm
///
/// # Returns
///
/// A `PressureResult` containing pressure scores for all influenced nodes.
///
/// # Algorithm Details
///
/// 1. Initialize pressure[start_node] = 1.0
/// 2. For each iteration:
///    - For each node with pressure:
///      - Transfer (score * damping * edge_weight) to each neighbor
///      - Keep (score * (1 - damping)) at current node
/// 3. Stop when converged (max delta < epsilon) or max iterations reached
/// 4. Optionally normalize scores to sum to 1.0
pub fn propagate_pressure<G: GraphProvider + ?Sized>(
    graph: &G,
    start_node: &str,
    config: &PressureConfig,
) -> Result<PressureResult, PressureError> {
    // Validate configuration
    config.validate()?;

    // Check if start node exists
    if !graph.node_exists(start_node) {
        return Err(PressureError::NodeNotFound {
            node_id: start_node.to_string(),
        });
    }

    let start_time = std::time::Instant::now();

    // Initialize pressure map
    let mut pressure: HashMap<String, f64> = HashMap::with_capacity(128);
    pressure.insert(start_node.to_string(), 1.0);

    let damping = config.damping_factor;
    let retention = 1.0 - damping;
    let mut converged = false;
    let mut final_delta = 0.0;
    let mut iterations_used = 0;

    // Iterative propagation
    for iteration in 1..=config.max_iterations {
        iterations_used = iteration;
        let mut new_pressure: HashMap<String, f64> = HashMap::with_capacity(pressure.len() * 2);
        let mut max_delta: f64 = 0.0;

        // Process each node with pressure
        for (node_id, score) in &pressure {
            // Check node limit
            if new_pressure.len() >= config.max_nodes {
                return Err(PressureError::LimitExceeded {
                    count: new_pressure.len(),
                    limit: config.max_nodes,
                });
            }

            // Get neighbors and their edge weights
            let neighbors = graph.get_neighbors(node_id);

            // Transfer pressure to neighbors
            for (neighbor_id, edge_weight) in neighbors {
                // Clamp edge weight to [0, 1]
                let weight = edge_weight.clamp(0.0, 1.0);
                let transfer = score * damping * weight;

                if transfer > config.min_pressure_threshold {
                    *new_pressure.entry(neighbor_id).or_insert(0.0) += transfer;
                }
            }

            // Retain pressure at current node
            let retained = score * retention;
            if retained > config.min_pressure_threshold {
                *new_pressure.entry(node_id.clone()).or_insert(0.0) += retained;
            }
        }

        // Calculate convergence (max change)
        for (node_id, new_score) in &new_pressure {
            let old_score = pressure.get(node_id).unwrap_or(&0.0);
            let delta = (new_score - old_score).abs();
            max_delta = max_delta.max(delta);
        }

        // Check for nodes that disappeared
        for (node_id, old_score) in &pressure {
            if !new_pressure.contains_key(node_id) {
                max_delta = max_delta.max(*old_score);
            }
        }

        final_delta = max_delta;

        // Check convergence
        if max_delta < config.epsilon {
            converged = true;
            pressure = new_pressure;
            break;
        }

        pressure = new_pressure;
    }

    // Calculate total pressure
    let total_pressure: f64 = pressure.values().sum();

    // Normalize if requested
    if config.normalize && total_pressure > 0.0 {
        for score in pressure.values_mut() {
            *score /= total_pressure;
        }
    }

    // Filter out nodes below threshold
    pressure.retain(|_, score| *score >= config.min_pressure_threshold);

    let execution_time_us = start_time.elapsed().as_micros() as u64;

    Ok(PressureResult {
        pressure_scores: pressure.clone(),
        source_node: start_node.to_string(),
        stats: PropagationStats {
            iterations_used,
            converged,
            final_delta,
            nodes_reached: pressure.len(),
            total_pressure: if config.normalize {
                1.0
            } else {
                pressure.values().sum()
            },
            execution_time_us,
        },
        config: config.clone(),
    })
}

/// Simple in-memory graph for testing and examples.
#[derive(Debug, Clone, Default)]
pub struct InMemoryGraph {
    /// Adjacency list: node_id -> [(neighbor_id, edge_weight)]
    edges: HashMap<String, Vec<(String, f64)>>,
}

impl InMemoryGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node_id: impl Into<String>) {
        self.edges.entry(node_id.into()).or_default();
    }

    /// Add a directed edge with weight.
    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>, weight: f64) {
        let from = from.into();
        let to = to.into();
        self.edges.entry(from.clone()).or_default();
        self.edges.entry(to.clone()).or_default();
        self.edges
            .get_mut(&from)
            .unwrap()
            .push((to, weight.clamp(0.0, 1.0)));
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.edges.len()
    }

    /// Get the number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum()
    }

    /// Iterate over all edges in the graph.
    ///
    /// Yields `(source_node_id, edges)` pairs where `edges` is a slice of
    /// `(target_node_id, weight)` tuples. This provides read-only access to
    /// the internal adjacency structure for conversion to other graph formats.
    pub fn edges_iter(&self) -> impl Iterator<Item = (&str, &[(String, f64)])> {
        self.edges
            .iter()
            .map(|(node, neighbors)| (node.as_str(), neighbors.as_slice()))
    }

    /// Get incoming edges (reverse neighbors) for a node.
    ///
    /// Returns all nodes that have edges pointing TO the given node.
    pub fn get_reverse_neighbors(&self, node_id: &str) -> Vec<(String, f64)> {
        let mut reverse = Vec::new();
        for (source, edges) in &self.edges {
            for (target, weight) in edges {
                if target == node_id {
                    reverse.push((source.clone(), *weight));
                }
            }
        }
        reverse
    }
}

impl GraphProvider for InMemoryGraph {
    fn get_neighbors(&self, node_id: &str) -> Vec<(String, f64)> {
        self.edges.get(node_id).cloned().unwrap_or_default()
    }

    fn node_exists(&self, node_id: &str) -> bool {
        self.edges.contains_key(node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> InMemoryGraph {
        let mut graph = InMemoryGraph::new();
        // Create a simple test graph:
        //     A --0.8--> B --0.9--> D
        //     |          |
        //     0.5        0.7
        //     v          v
        //     C <--0.6-- B
        graph.add_edge("A", "B", 0.8);
        graph.add_edge("A", "C", 0.5);
        graph.add_edge("B", "C", 0.6);
        graph.add_edge("B", "D", 0.9);
        graph.add_node("D"); // D has no outgoing edges
        graph
    }

    #[test]
    fn test_default_config() {
        let config = PressureConfig::default();
        assert_eq!(config.damping_factor, 0.85);
        assert_eq!(config.max_iterations, 3);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let invalid_damping = PressureConfig {
            damping_factor: 1.5,
            ..Default::default()
        };
        assert!(invalid_damping.validate().is_err());

        let invalid_iterations = PressureConfig {
            max_iterations: 0,
            ..Default::default()
        };
        assert!(invalid_iterations.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = PressureConfig::default()
            .damping(0.9)
            .iterations(5)
            .epsilon(1e-8)
            .normalized();

        assert_eq!(config.damping_factor, 0.9);
        assert_eq!(config.max_iterations, 5);
        assert_eq!(config.epsilon, 1e-8);
        assert!(config.normalize);
    }

    #[test]
    fn test_basic_propagation() {
        let graph = create_test_graph();
        let config = PressureConfig::default();

        let result = propagate_pressure(&graph, "A", &config).unwrap();

        // All reachable nodes should have pressure
        assert!(result.pressure_scores.contains_key("A"));
        assert!(result.pressure_scores.contains_key("B"));
        assert!(result.pressure_scores.contains_key("C"));

        // D should receive pressure from B (through multi-hop)
        // With 3 iterations (default), D should have some pressure
        assert!(result.stats.nodes_reached >= 3);

        // Total pressure should be positive (some may be lost to threshold filtering)
        assert!(result.stats.total_pressure > 0.0);

        // The algorithm should complete within max iterations
        assert!(result.stats.iterations_used <= config.max_iterations);
    }

    #[test]
    fn test_single_iteration() {
        let graph = create_test_graph();
        let config = PressureConfig::default().iterations(1);

        let result = propagate_pressure(&graph, "A", &config).unwrap();

        assert_eq!(result.stats.iterations_used, 1);

        // After 1 iteration:
        // A: 1.0 * (1 - 0.85) = 0.15 retained
        // B: 1.0 * 0.85 * 0.8 = 0.68
        // C: 1.0 * 0.85 * 0.5 = 0.425
        let a_pressure = result.get_pressure("A").unwrap();
        let b_pressure = result.get_pressure("B").unwrap();
        let c_pressure = result.get_pressure("C").unwrap();

        assert!((a_pressure - 0.15).abs() < 0.001);
        assert!((b_pressure - 0.68).abs() < 0.001);
        assert!((c_pressure - 0.425).abs() < 0.001);
    }

    #[test]
    fn test_damping_effect() {
        let graph = create_test_graph();

        // Use single iteration for predictable behavior
        // High damping = more spread to neighbors
        let high_damping = propagate_pressure(
            &graph,
            "A",
            &PressureConfig::default().damping(0.95).iterations(1),
        )
        .unwrap();

        // Low damping = more retained at source
        let low_damping = propagate_pressure(
            &graph,
            "A",
            &PressureConfig::default().damping(0.5).iterations(1),
        )
        .unwrap();

        // With low damping, A retains more pressure
        assert!(
            low_damping.get_pressure("A").unwrap()
                > high_damping.get_pressure("A").unwrap()
        );

        // With high damping, first-hop neighbors (B) get more pressure in single iteration
        assert!(
            high_damping.get_pressure("B").unwrap()
                > low_damping.get_pressure("B").unwrap()
        );
    }

    #[test]
    fn test_convergence() {
        let graph = create_test_graph();
        let config = PressureConfig::default()
            .iterations(100)
            .epsilon(1e-10);

        let result = propagate_pressure(&graph, "A", &config).unwrap();

        // Should converge before 100 iterations
        assert!(result.stats.iterations_used < 100);
        assert!(result.stats.converged || result.stats.final_delta < 1e-6);
    }

    #[test]
    fn test_normalization() {
        let graph = create_test_graph();
        let config = PressureConfig::default().normalized();

        let result = propagate_pressure(&graph, "A", &config).unwrap();

        // Normalized scores should sum to approximately 1.0
        let total: f64 = result.pressure_scores.values().sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_node_not_found() {
        let graph = create_test_graph();
        let config = PressureConfig::default();

        let result = propagate_pressure(&graph, "NonExistent", &config);
        assert!(matches!(result, Err(PressureError::NodeNotFound { .. })));
    }

    #[test]
    fn test_isolated_node() {
        let mut graph = InMemoryGraph::new();
        graph.add_node("Isolated");

        let config = PressureConfig::default();
        let result = propagate_pressure(&graph, "Isolated", &config).unwrap();

        // Isolated node should retain all pressure (with retention factor)
        assert!(result.pressure_scores.contains_key("Isolated"));
        assert_eq!(result.stats.nodes_reached, 1);
    }

    #[test]
    fn test_top_n() {
        let graph = create_test_graph();
        let config = PressureConfig::default();

        let result = propagate_pressure(&graph, "A", &config).unwrap();
        let top_2 = result.top_n(2);

        assert_eq!(top_2.len(), 2);
        // First should have highest score
        assert!(top_2[0].1 >= top_2[1].1);
    }

    #[test]
    fn test_execution_stats() {
        let graph = create_test_graph();
        let config = PressureConfig::default();

        let result = propagate_pressure(&graph, "A", &config).unwrap();

        assert!(result.stats.iterations_used > 0);
        assert!(result.stats.iterations_used <= config.max_iterations);
        assert!(result.stats.nodes_reached > 0);
        assert!(result.stats.total_pressure > 0.0);
    }

    #[test]
    fn test_zero_damping() {
        let graph = create_test_graph();
        let config = PressureConfig::default().damping(0.0);

        let result = propagate_pressure(&graph, "A", &config).unwrap();

        // With zero damping, all pressure stays at source
        assert_eq!(result.get_pressure("A").unwrap(), 1.0);
        // No pressure should transfer to neighbors
        assert!(result.get_pressure("B").unwrap_or(0.0) < 1e-9);
    }

    #[test]
    fn test_full_damping() {
        let graph = create_test_graph();
        let config = PressureConfig::default().damping(1.0).iterations(1);

        let result = propagate_pressure(&graph, "A", &config).unwrap();

        // With full damping, nothing is retained at source after 1 iteration
        assert!(result.get_pressure("A").unwrap_or(0.0) < 1e-9);
        // All pressure should transfer to neighbors
        assert!(result.get_pressure("B").unwrap() > 0.0);
        assert!(result.get_pressure("C").unwrap() > 0.0);
    }

    #[test]
    fn test_in_memory_graph() {
        let mut graph = InMemoryGraph::new();
        graph.add_edge("X", "Y", 0.5);
        graph.add_edge("Y", "Z", 0.3);

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert!(graph.node_exists("X"));
        assert!(graph.node_exists("Y"));
        assert!(graph.node_exists("Z"));
        assert!(!graph.node_exists("W"));
    }

    #[test]
    fn test_edges_iter() {
        let mut graph = InMemoryGraph::new();
        graph.add_edge("A", "B", 0.8);
        graph.add_edge("A", "C", 0.5);
        graph.add_edge("B", "C", 0.3);
        graph.add_node("D"); // isolated node, has entry but no edges

        let mut all_edges: Vec<(String, Vec<(String, f64)>)> = graph
            .edges_iter()
            .map(|(src, neighbors)| {
                (
                    src.to_string(),
                    neighbors.iter().map(|(t, w)| (t.clone(), *w)).collect(),
                )
            })
            .collect();
        all_edges.sort_by(|a, b| a.0.cmp(&b.0));

        // All 4 nodes should appear (A, B, C, D)
        assert_eq!(all_edges.len(), 4);

        // A -> [(B, 0.8), (C, 0.5)]
        let a_entry = all_edges.iter().find(|(n, _)| n == "A").unwrap();
        assert_eq!(a_entry.1.len(), 2);

        // D has no outgoing edges
        let d_entry = all_edges.iter().find(|(n, _)| n == "D").unwrap();
        assert!(d_entry.1.is_empty());
    }
}
