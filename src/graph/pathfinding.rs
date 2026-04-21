//! BFS-based path finding for the context graph.
//!
//! Provides algorithms for finding paths between nodes in the knowledge graph,
//! with support for weighted paths and configurable maximum depth.
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::graph::{PathFinder, PathQuery};
//!
//! // Create a path finder from edges
//! let finder = PathFinder::from_edges(edges);
//!
//! // Find all paths up to depth 3
//! let paths = finder.find_paths("node_a", "node_z", 3)?;
//!
//! // Get the strongest path
//! if let Some(best) = paths.first() {
//!     println!("Best path: {:?} (strength: {})", best.nodes, best.total_strength);
//! }
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use super::{EdgeType, GraphEdge, GraphError};

/// Query parameters for path finding.
#[derive(Debug, Clone)]
pub struct PathQuery {
    /// Starting node ID.
    pub from: String,
    /// Target node ID.
    pub to: String,
    /// Maximum path length (number of edges).
    pub max_depth: usize,
    /// Filter by edge types (None = all types).
    pub edge_types: Option<Vec<EdgeType>>,
    /// Minimum edge strength to traverse.
    pub min_strength: Option<f64>,
    /// Maximum number of paths to return.
    pub max_paths: Option<usize>,
}

impl PathQuery {
    /// Create a new path query.
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            max_depth: 3,
            edge_types: None,
            min_strength: None,
            max_paths: None,
        }
    }

    /// Set the maximum depth.
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Filter by edge types.
    pub fn edge_types(mut self, types: Vec<EdgeType>) -> Self {
        self.edge_types = Some(types);
        self
    }

    /// Set minimum edge strength.
    pub fn min_strength(mut self, strength: f64) -> Self {
        self.min_strength = Some(strength.clamp(0.0, 1.0));
        self
    }

    /// Limit the number of paths returned.
    pub fn max_paths(mut self, max: usize) -> Self {
        self.max_paths = Some(max);
        self
    }
}

/// A path through the graph from source to target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    /// Node IDs in the path (ordered from source to target).
    pub nodes: Vec<String>,
    /// Edges connecting the nodes.
    pub edges: Vec<GraphEdge>,
    /// Total path strength (product of edge strengths).
    pub total_strength: f64,
}

impl GraphPath {
    /// Create a new empty path.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            total_strength: 1.0,
        }
    }

    /// Create a path from nodes and edges.
    pub fn from_parts(nodes: Vec<String>, edges: Vec<GraphEdge>) -> Self {
        let total_strength = if edges.is_empty() {
            1.0
        } else {
            edges.iter().map(|e| e.strength).product()
        };
        Self {
            nodes,
            edges,
            total_strength,
        }
    }

    /// Create a trivial path (source == target).
    pub fn trivial(node: String) -> Self {
        Self {
            nodes: vec![node],
            edges: Vec::new(),
            total_strength: 1.0,
        }
    }

    /// Get the path length (number of edges/hops).
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Check if the path is empty (no nodes).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the source node (first in path).
    pub fn source(&self) -> Option<&str> {
        self.nodes.first().map(|s| s.as_str())
    }

    /// Get the target node (last in path).
    pub fn target(&self) -> Option<&str> {
        self.nodes.last().map(|s| s.as_str())
    }

    /// Check if this is a direct (single-hop) path.
    pub fn is_direct(&self) -> bool {
        self.edges.len() == 1
    }

    /// Get the minimum edge strength along the path.
    pub fn min_edge_strength(&self) -> Option<f64> {
        self.edges.iter().map(|e| e.strength).reduce(f64::min)
    }

    /// Get the edge types along the path.
    pub fn edge_types(&self) -> Vec<EdgeType> {
        self.edges.iter().map(|e| e.edge_type).collect()
    }
}

impl Default for GraphPath {
    fn default() -> Self {
        Self::new()
    }
}

/// BFS state for path finding.
#[derive(Debug, Clone)]
struct BfsState {
    node: String,
    path_nodes: Vec<String>,
    path_edges: Vec<GraphEdge>,
}

impl BfsState {
    fn new(start: String) -> Self {
        Self {
            node: start.clone(),
            path_nodes: vec![start],
            path_edges: Vec::new(),
        }
    }

    fn extend(&self, next_node: String, edge: GraphEdge) -> Self {
        let mut path_nodes = self.path_nodes.clone();
        path_nodes.push(next_node.clone());

        let mut path_edges = self.path_edges.clone();
        path_edges.push(edge);

        Self {
            node: next_node,
            path_nodes,
            path_edges,
        }
    }

    fn depth(&self) -> usize {
        self.path_edges.len()
    }

    fn to_path(self) -> GraphPath {
        GraphPath::from_parts(self.path_nodes, self.path_edges)
    }
}

/// Path finder using BFS algorithm.
///
/// Supports finding all paths between two nodes up to a maximum depth,
/// with optional filtering by edge type and minimum strength.
pub struct PathFinder {
    /// Adjacency list: node_id -> [(neighbor_id, edge)]
    adjacency: HashMap<String, Vec<(String, GraphEdge)>>,
}

impl PathFinder {
    /// Create a new empty path finder.
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
        }
    }

    /// Create a path finder from a list of edges.
    pub fn from_edges(edges: Vec<GraphEdge>) -> Self {
        let mut finder = Self::new();
        for edge in edges {
            finder.add_edge(edge);
        }
        finder
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, edge: GraphEdge) {
        // Add outgoing edge
        self.adjacency
            .entry(edge.source_id.clone())
            .or_default()
            .push((edge.target_id.clone(), edge.clone()));

        // For symmetric edge types, also add reverse direction
        if edge.edge_type.is_symmetric() {
            let reverse = GraphEdge {
                id: format!("{}_rev", edge.id),
                source_id: edge.target_id.clone(),
                target_id: edge.source_id.clone(),
                edge_type: edge.edge_type,
                strength: edge.strength,
                metadata: edge.metadata.clone(),
                created_at: edge.created_at,
                updated_at: edge.updated_at,
            };
            self.adjacency
                .entry(edge.target_id.clone())
                .or_default()
                .push((edge.source_id.clone(), reverse));
        }

        // Ensure target node exists in adjacency
        self.adjacency.entry(edge.target_id.clone()).or_default();
    }

    /// Get neighbors of a node with their connecting edges.
    fn neighbors(&self, node: &str) -> Vec<(String, GraphEdge)> {
        self.adjacency.get(node).cloned().unwrap_or_default()
    }

    /// Find all paths from source to target using BFS.
    ///
    /// # Arguments
    ///
    /// * `from` - Source node ID
    /// * `to` - Target node ID
    /// * `max_depth` - Maximum number of edges to traverse
    ///
    /// # Returns
    ///
    /// Vector of paths sorted by total strength (descending).
    pub fn find_paths(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
    ) -> Result<Vec<GraphPath>, GraphError> {
        self.find_paths_with_query(&PathQuery::new(from, to).max_depth(max_depth))
    }

    /// Find paths using a query object for more control.
    pub fn find_paths_with_query(&self, query: &PathQuery) -> Result<Vec<GraphPath>, GraphError> {
        // Handle same source and target
        if query.from == query.to {
            return Ok(vec![GraphPath::trivial(query.from.clone())]);
        }

        let max_depth = query.max_depth.min(10); // Safety limit
        let mut paths: Vec<GraphPath> = Vec::new();
        let mut queue: VecDeque<BfsState> = VecDeque::new();

        queue.push_back(BfsState::new(query.from.clone()));

        while let Some(state) = queue.pop_front() {
            // Found target
            if state.node == query.to {
                paths.push(state.to_path());

                if let Some(max) = query.max_paths {
                    if paths.len() >= max {
                        break;
                    }
                }
                continue;
            }

            // Don't expand beyond max depth
            if state.depth() >= max_depth {
                continue;
            }

            // Expand to neighbors
            for (neighbor, edge) in self.neighbors(&state.node) {
                // Avoid cycles
                if state.path_nodes.contains(&neighbor) {
                    continue;
                }

                // Apply edge type filter
                if let Some(ref types) = query.edge_types {
                    if !types.contains(&edge.edge_type) {
                        continue;
                    }
                }

                // Apply strength filter
                if let Some(min) = query.min_strength {
                    if edge.strength < min {
                        continue;
                    }
                }

                queue.push_back(state.extend(neighbor, edge));
            }
        }

        // Sort by total strength (descending)
        paths.sort_by(|a, b| {
            b.total_strength
                .partial_cmp(&a.total_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(paths)
    }

    /// Find the shortest path (fewest hops).
    pub fn find_shortest_path(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
    ) -> Result<Option<GraphPath>, GraphError> {
        let paths = self.find_paths(from, to, max_depth)?;
        Ok(paths.into_iter().min_by_key(|p| p.len()))
    }

    /// Find the strongest path (highest total strength).
    pub fn find_strongest_path(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
    ) -> Result<Option<GraphPath>, GraphError> {
        let paths = self.find_paths(from, to, max_depth)?;
        Ok(paths.into_iter().next()) // Already sorted by strength
    }

    /// Check if a path exists between two nodes.
    pub fn path_exists(&self, from: &str, to: &str, max_depth: usize) -> bool {
        if from == to {
            return true;
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        queue.push_back((from.to_string(), 0));
        visited.insert(from.to_string());

        while let Some((node, depth)) = queue.pop_front() {
            if node == to {
                return true;
            }

            if depth >= max_depth {
                continue;
            }

            for (neighbor, _) in self.neighbors(&node) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor.clone());
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }

        false
    }

    /// Get all nodes reachable within max_depth hops.
    pub fn reachable(&self, from: &str, max_depth: usize) -> HashSet<String> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        queue.push_back((from.to_string(), 0));
        visited.insert(from.to_string());

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            for (neighbor, _) in self.neighbors(&node) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor.clone());
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }

        visited
    }

    /// Check if a node exists in the graph.
    pub fn node_exists(&self, node: &str) -> bool {
        self.adjacency.contains_key(node)
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.adjacency.len()
    }

    /// Get the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.adjacency.values().map(|v| v.len()).sum()
    }
}

impl Default for PathFinder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_edges() -> Vec<GraphEdge> {
        vec![
            GraphEdge::new("A", "B", EdgeType::RelatedTo, 0.9),
            GraphEdge::new("B", "D", EdgeType::RelatedTo, 0.8),
            GraphEdge::new("A", "C", EdgeType::SimilarTo, 0.7),
            GraphEdge::new("C", "D", EdgeType::SimilarTo, 0.6),
            GraphEdge::new("A", "E", EdgeType::RelatedTo, 0.5),
        ]
    }

    #[test]
    fn test_path_query() {
        let query = PathQuery::new("a", "b").max_depth(3);
        assert_eq!(query.from, "a");
        assert_eq!(query.to, "b");
        assert_eq!(query.max_depth, 3);
    }

    #[test]
    fn test_graph_path() {
        let path = GraphPath::new();
        assert!(path.is_empty());
        assert_eq!(path.len(), 0);
    }

    #[test]
    fn test_path_from_parts() {
        let edges = vec![
            GraphEdge::new("A", "B", EdgeType::RelatedTo, 0.8),
            GraphEdge::new("B", "C", EdgeType::RelatedTo, 0.5),
        ];
        let nodes = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let path = GraphPath::from_parts(nodes, edges);

        assert_eq!(path.len(), 2);
        assert!((path.total_strength - 0.4).abs() < 0.001);
        assert_eq!(path.source(), Some("A"));
        assert_eq!(path.target(), Some("C"));
    }

    #[test]
    fn test_find_paths() {
        let finder = PathFinder::from_edges(test_edges());
        let paths = finder.find_paths("A", "D", 3).unwrap();

        // Should find 2 paths: A->B->D and A->C->D
        assert_eq!(paths.len(), 2);

        // First should be strongest (0.9 * 0.8 = 0.72)
        assert!((paths[0].total_strength - 0.72).abs() < 0.001);

        // Second (0.7 * 0.6 = 0.42)
        assert!((paths[1].total_strength - 0.42).abs() < 0.001);
    }

    #[test]
    fn test_find_paths_same_node() {
        let finder = PathFinder::from_edges(test_edges());
        let paths = finder.find_paths("A", "A", 3).unwrap();

        assert_eq!(paths.len(), 1);
        assert!(paths[0].edges.is_empty());
    }

    #[test]
    fn test_find_paths_no_path() {
        let finder = PathFinder::from_edges(test_edges());
        let paths = finder.find_paths("A", "Z", 3).unwrap();

        assert!(paths.is_empty());
    }

    #[test]
    fn test_find_paths_depth_limit() {
        let finder = PathFinder::from_edges(test_edges());

        // Depth 1: no direct path A->D
        let paths = finder.find_paths("A", "D", 1).unwrap();
        assert!(paths.is_empty());

        // Depth 2: both paths exist
        let paths = finder.find_paths("A", "D", 2).unwrap();
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_find_paths_edge_type_filter() {
        let finder = PathFinder::from_edges(test_edges());

        let query = PathQuery::new("A", "D")
            .max_depth(3)
            .edge_types(vec![EdgeType::SimilarTo]);

        let paths = finder.find_paths_with_query(&query).unwrap();

        // Only A->C->D uses SimilarTo
        assert_eq!(paths.len(), 1);
        assert!((paths[0].total_strength - 0.42).abs() < 0.001);
    }

    #[test]
    fn test_find_paths_strength_filter() {
        let finder = PathFinder::from_edges(test_edges());

        let query = PathQuery::new("A", "D").max_depth(3).min_strength(0.75);

        let paths = finder.find_paths_with_query(&query).unwrap();

        // Only A->B->D has both edges >= 0.75
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_path_exists() {
        let finder = PathFinder::from_edges(test_edges());

        assert!(finder.path_exists("A", "D", 3));
        assert!(finder.path_exists("A", "A", 0));
        assert!(!finder.path_exists("A", "Z", 3));
    }

    #[test]
    fn test_reachable() {
        let finder = PathFinder::from_edges(test_edges());

        let r1 = finder.reachable("A", 1);
        assert!(r1.contains("A"));
        assert!(r1.contains("B"));
        assert!(r1.contains("C"));
        assert!(!r1.contains("D"));

        let r2 = finder.reachable("A", 2);
        assert!(r2.contains("D"));
    }

    #[test]
    fn test_symmetric_edges() {
        let edges = vec![GraphEdge::new("A", "B", EdgeType::SimilarTo, 0.8)];
        let finder = PathFinder::from_edges(edges);

        // Forward
        assert!(finder.path_exists("A", "B", 1));
        // Reverse (symmetric)
        assert!(finder.path_exists("B", "A", 1));
    }

    #[test]
    fn test_avoid_cycles() {
        let edges = vec![
            GraphEdge::new("A", "B", EdgeType::RelatedTo, 0.9),
            GraphEdge::new("B", "C", EdgeType::RelatedTo, 0.8),
            GraphEdge::new("C", "A", EdgeType::RelatedTo, 0.7),
            GraphEdge::new("C", "D", EdgeType::RelatedTo, 0.6),
        ];

        let finder = PathFinder::from_edges(edges);
        let paths = finder.find_paths("A", "D", 5).unwrap();

        // Should find path A->B->C->D
        // The path should not contain cycles
        assert!(!paths.is_empty());

        // Verify the main path exists
        let main_path = paths.iter().find(|p| p.nodes == vec!["A", "B", "C", "D"]);
        assert!(main_path.is_some(), "Expected path A->B->C->D not found");

        // Verify no path contains duplicate nodes (no cycles)
        for path in &paths {
            let mut seen = std::collections::HashSet::new();
            for node in &path.nodes {
                assert!(seen.insert(node.clone()), "Path contains cycle: {:?}", path.nodes);
            }
        }
    }
}
