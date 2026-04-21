//! Stoer-Wagner minimum cut algorithm for graph clustering.
//!
//! Implements the Stoer-Wagner algorithm for finding the global minimum cut
//! of an undirected weighted graph, and a recursive clustering algorithm
//! that discovers cohesive communities by repeatedly applying minimum cuts.
//!
//! # Algorithm
//!
//! The Stoer-Wagner algorithm works in O(VE + V^2 log V) time:
//!
//! 1. Start with all nodes in one set
//! 2. Repeatedly find the "minimum cut of the phase":
//!    a. Pick an arbitrary start node
//!    b. Grow a set A by adding the most tightly connected node
//!    c. The last two nodes added (s, t) define a "cut of the phase"
//!    d. Merge s and t, combining their edges
//! 3. The minimum over all phases is the global minimum cut
//!
//! # Cluster Discovery
//!
//! `discover_clusters` recursively applies mincut to partition the graph.
//! If the minimum cut weight is below a threshold, the graph is split
//! into two groups and each is recursed upon. Otherwise, the group is
//! a cohesive cluster.
//!
//! # Example
//!
//! ```rust
//! use nagual::graph::mincut::{MinCutGraph, Cluster};
//!
//! let mut graph = MinCutGraph::new();
//! let a = graph.add_node("A".into());
//! let b = graph.add_node("B".into());
//! let c = graph.add_node("C".into());
//!
//! graph.add_edge(a, b, 5.0);
//! graph.add_edge(b, c, 1.0);
//! graph.add_edge(a, c, 5.0);
//!
//! if let Some((cut_weight, side_a, side_b)) = graph.minimum_cut() {
//!     println!("Min cut weight: {cut_weight}");
//! }
//!
//! let clusters = graph.discover_clusters(2.0);
//! println!("Found {} clusters", clusters.len());
//! ```

use std::collections::{HashMap, HashSet};

/// A cluster of related nodes discovered by recursive mincut.
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Unique cluster identifier (assigned during discovery).
    pub id: usize,
    /// Node IDs belonging to this cluster.
    pub node_ids: Vec<String>,
    /// Sum of internal edge weights within this cluster.
    pub internal_weight: f64,
}

/// Weighted undirected graph for minimum cut computation.
///
/// Nodes are identified by string IDs externally and by integer indices
/// internally for efficient adjacency operations.
pub struct MinCutGraph {
    /// Node names, indexed by node index.
    nodes: Vec<String>,
    /// Adjacency list: node_index -> Vec<(neighbor_index, weight)>.
    adjacency: Vec<Vec<(usize, f64)>>,
    /// Reverse lookup: node name -> node index.
    node_to_index: HashMap<String, usize>,
}

impl MinCutGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            adjacency: Vec::new(),
            node_to_index: HashMap::new(),
        }
    }

    /// Add a node to the graph, returning its index.
    ///
    /// If the node already exists, returns its existing index.
    pub fn add_node(&mut self, id: String) -> usize {
        if let Some(&idx) = self.node_to_index.get(&id) {
            return idx;
        }
        let idx = self.nodes.len();
        self.node_to_index.insert(id.clone(), idx);
        self.nodes.push(id);
        self.adjacency.push(Vec::new());
        idx
    }

    /// Add an undirected edge between two nodes with the given weight.
    ///
    /// If an edge already exists between the nodes, the weight is added
    /// to the existing edge weight (multi-edge accumulation).
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        if from == to || from >= self.nodes.len() || to >= self.nodes.len() {
            return;
        }
        // Check if edge already exists and accumulate weight
        if let Some(entry) = self.adjacency[from].iter_mut().find(|(n, _)| *n == to) {
            entry.1 += weight;
        } else {
            self.adjacency[from].push((to, weight));
        }
        if let Some(entry) = self.adjacency[to].iter_mut().find(|(n, _)| *n == from) {
            entry.1 += weight;
        } else {
            self.adjacency[to].push((from, weight));
        }
    }

    /// Return the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the node name for a given index.
    pub fn node_name(&self, idx: usize) -> Option<&str> {
        self.nodes.get(idx).map(|s| s.as_str())
    }

    /// Run the Stoer-Wagner algorithm to find the global minimum cut.
    ///
    /// Returns `Some((min_cut_weight, partition_a, partition_b))` where
    /// `partition_a` and `partition_b` are sets of node indices forming
    /// the two sides of the minimum cut.
    ///
    /// Returns `None` if the graph has fewer than 2 nodes.
    pub fn minimum_cut(&self) -> Option<(f64, HashSet<usize>, HashSet<usize>)> {
        let n = self.nodes.len();
        if n < 2 {
            return None;
        }

        // Build a working adjacency matrix (dense) for Stoer-Wagner.
        // We use a flat Vec for O(1) edge weight lookup.
        let mut w = vec![vec![0.0f64; n]; n];
        for (u, neighbors) in self.adjacency.iter().enumerate() {
            for &(v, weight) in neighbors {
                w[u][v] = weight;
            }
        }

        // merged[i] tracks which original node index i has been merged into.
        // Initially each node is its own representative.
        // When we merge t into s, we set merged[t] = s.
        // We also track which original nodes belong to each merged super-node.
        let mut groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut active: Vec<bool> = vec![true; n];
        let mut best_cut = f64::INFINITY;
        let mut best_partition: HashSet<usize> = HashSet::new();

        for _ in 0..n - 1 {
            // Count active nodes
            let active_nodes: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
            if active_nodes.len() < 2 {
                break;
            }

            // Stoer-Wagner phase: maximum adjacency ordering
            let (s, t, cut_of_phase) =
                self.minimum_cut_phase(&w, &active_nodes);

            // Check if this phase found a better cut
            if cut_of_phase < best_cut {
                best_cut = cut_of_phase;
                best_partition = groups[t].iter().copied().collect();
            }

            // Merge t into s
            let t_group = groups[t].clone();
            groups[s].extend(t_group);
            groups[t].clear();
            active[t] = false;

            // Merge adjacency: combine edges of t into s
            for i in 0..n {
                if i != s && i != t {
                    w[s][i] += w[t][i];
                    w[i][s] += w[i][t];
                }
            }
            // Clear t's edges
            for i in 0..n {
                w[t][i] = 0.0;
                w[i][t] = 0.0;
            }
        }

        // Build the two partitions
        let all_nodes: HashSet<usize> = (0..n).collect();
        let partition_b: HashSet<usize> = all_nodes.difference(&best_partition).copied().collect();

        Some((best_cut, best_partition, partition_b))
    }

    /// Execute one phase of the Stoer-Wagner algorithm.
    ///
    /// Returns (s, t, cut_of_phase) where s and t are the last two nodes
    /// added to the maximum adjacency ordering, and cut_of_phase is the
    /// weight of edges connecting t to all other nodes in the ordering.
    fn minimum_cut_phase(
        &self,
        w: &[Vec<f64>],
        active_nodes: &[usize],
    ) -> (usize, usize, f64) {
        let n = active_nodes.len();
        debug_assert!(n >= 2);

        // key[v] = total weight of edges from v to the set A
        let mut key: HashMap<usize, f64> = HashMap::with_capacity(n);
        let mut in_a: HashSet<usize> = HashSet::with_capacity(n);

        for &node in active_nodes {
            key.insert(node, 0.0);
        }

        let mut s = active_nodes[0];
        let mut t = active_nodes[0];

        for _ in 0..n {
            // Find the node not in A with the maximum key value
            let mut max_key = f64::NEG_INFINITY;
            let mut max_node = active_nodes[0];
            for &node in active_nodes {
                if !in_a.contains(&node) {
                    let k = key.get(&node).copied().unwrap_or(0.0);
                    if k > max_key {
                        max_key = k;
                        max_node = node;
                    }
                }
            }

            s = t;
            t = max_node;
            in_a.insert(max_node);

            // Update keys for remaining nodes
            for &node in active_nodes {
                if !in_a.contains(&node) {
                    *key.entry(node).or_insert(0.0) += w[max_node][node];
                }
            }
        }

        // The cut of the phase is the key value of t when it was added
        // which equals the total weight of edges from t to all other active nodes
        let cut_weight: f64 = active_nodes
            .iter()
            .filter(|&&node| node != t)
            .map(|&node| w[t][node])
            .sum();

        (s, t, cut_weight)
    }

    /// Discover clusters by recursively applying minimum cuts.
    ///
    /// The algorithm splits the graph whenever the minimum cut weight
    /// is below `min_cut_threshold`, indicating a weak connection between
    /// two groups. Recursion stops when:
    /// - The group has fewer than 2 nodes
    /// - The minimum cut weight exceeds the threshold (cohesive group)
    ///
    /// # Arguments
    ///
    /// * `min_cut_threshold` - Minimum cut weight below which the graph
    ///   is split. Higher values produce more, smaller clusters.
    pub fn discover_clusters(&self, min_cut_threshold: f64) -> Vec<Cluster> {
        let all_nodes: Vec<usize> = (0..self.nodes.len()).collect();
        let mut clusters = Vec::new();
        let mut cluster_id = 0;
        self.recursive_cluster(
            &all_nodes,
            min_cut_threshold,
            &mut clusters,
            &mut cluster_id,
        );
        clusters
    }

    /// Recursively partition a subset of nodes into clusters.
    fn recursive_cluster(
        &self,
        node_indices: &[usize],
        threshold: f64,
        clusters: &mut Vec<Cluster>,
        next_id: &mut usize,
    ) {
        if node_indices.len() < 2 {
            // Single node or empty: it's its own cluster
            if !node_indices.is_empty() {
                let internal = self.compute_internal_weight(node_indices);
                clusters.push(Cluster {
                    id: *next_id,
                    node_ids: node_indices
                        .iter()
                        .map(|&i| self.nodes[i].clone())
                        .collect(),
                    internal_weight: internal,
                });
                *next_id += 1;
            }
            return;
        }

        // Build a subgraph for this subset
        let subgraph = self.subgraph(node_indices);
        match subgraph.minimum_cut() {
            Some((cut_weight, part_a, part_b)) if cut_weight < threshold => {
                // The cut is weak enough -- split and recurse
                // Map subgraph indices back to original indices
                let original_a: Vec<usize> = part_a
                    .iter()
                    .map(|&si| node_indices[si])
                    .collect();
                let original_b: Vec<usize> = part_b
                    .iter()
                    .map(|&si| node_indices[si])
                    .collect();

                self.recursive_cluster(&original_a, threshold, clusters, next_id);
                self.recursive_cluster(&original_b, threshold, clusters, next_id);
            }
            _ => {
                // Cohesive cluster: keep as-is
                let internal = self.compute_internal_weight(node_indices);
                clusters.push(Cluster {
                    id: *next_id,
                    node_ids: node_indices
                        .iter()
                        .map(|&i| self.nodes[i].clone())
                        .collect(),
                    internal_weight: internal,
                });
                *next_id += 1;
            }
        }
    }

    /// Build a subgraph containing only the specified node indices.
    fn subgraph(&self, node_indices: &[usize]) -> MinCutGraph {
        let mut sub = MinCutGraph::new();
        let mut old_to_new: HashMap<usize, usize> = HashMap::new();

        for &old_idx in node_indices {
            let new_idx = sub.add_node(self.nodes[old_idx].clone());
            old_to_new.insert(old_idx, new_idx);
        }

        // Add edges that exist between nodes in the subset
        for &old_idx in node_indices {
            for &(neighbor, weight) in &self.adjacency[old_idx] {
                if let Some(&new_neighbor) = old_to_new.get(&neighbor) {
                    let new_from = old_to_new[&old_idx];
                    // Only add each undirected edge once (from < to)
                    if new_from < new_neighbor {
                        sub.add_edge(new_from, new_neighbor, weight);
                    }
                }
            }
        }

        sub
    }

    /// Compute the sum of internal edge weights for a set of nodes.
    fn compute_internal_weight(&self, node_indices: &[usize]) -> f64 {
        let node_set: HashSet<usize> = node_indices.iter().copied().collect();
        let mut total = 0.0;
        for &idx in node_indices {
            for &(neighbor, weight) in &self.adjacency[idx] {
                if node_set.contains(&neighbor) && idx < neighbor {
                    total += weight;
                }
            }
        }
        total
    }
}

impl Default for MinCutGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a [`MinCutGraph`] from an [`InMemoryGraph`].
///
/// Since `InMemoryGraph` stores directed edges, this function treats
/// each directed edge as an undirected edge. If both (A->B) and (B->A)
/// exist, their weights are combined into one undirected edge via
/// [`MinCutGraph::add_edge`]'s weight accumulation.
///
/// Note that `InMemoryGraph` clamps edge weights to `[0.0, 1.0]` on
/// insertion, so the resulting `MinCutGraph` will have individual edge
/// contributions within that range.
///
/// # Example
///
/// ```rust
/// use nagual::graph::pressure::InMemoryGraph;
/// use nagual::graph::mincut::from_in_memory_graph;
///
/// let mut g = InMemoryGraph::new();
/// g.add_edge("A", "B", 0.8);
/// g.add_edge("B", "C", 0.5);
///
/// let mcg = from_in_memory_graph(&g);
/// assert_eq!(mcg.node_count(), 3);
/// ```
pub fn from_in_memory_graph(graph: &super::pressure::InMemoryGraph) -> MinCutGraph {
    let mut mcg = MinCutGraph::new();

    // First pass: register all nodes and collect directed edges.
    // edges_iter yields (source, &[(target, weight)]) for every node.
    let mut directed_edges: Vec<(String, String, f64)> = Vec::new();

    for (source, neighbors) in graph.edges_iter() {
        mcg.add_node(source.to_string());
        for (target, weight) in neighbors {
            directed_edges.push((source.to_string(), target.clone(), *weight));
        }
    }

    // Second pass: add each directed edge as an undirected edge.
    // MinCutGraph::add_edge accumulates weights, so if both A->B and B->A
    // exist they will be combined automatically.
    for (source, target, weight) in directed_edges {
        let src_idx = mcg.add_node(source);
        let tgt_idx = mcg.add_node(target);
        mcg.add_edge(src_idx, tgt_idx, weight);
    }

    mcg
}

/// Build a [`MinCutGraph`] from a list of undirected edges.
///
/// Each tuple is `(node_a_id, node_b_id, weight)`. Nodes are created
/// automatically. Duplicate edges accumulate their weights.
///
/// # Example
///
/// ```rust
/// use nagual::graph::mincut::from_edges;
///
/// let edges = vec![
///     ("A".to_string(), "B".to_string(), 1.0),
///     ("B".to_string(), "C".to_string(), 2.0),
/// ];
/// let graph = from_edges(&edges);
/// assert_eq!(graph.node_count(), 3);
/// ```
pub fn from_edges(edges: &[(String, String, f64)]) -> MinCutGraph {
    let mut graph = MinCutGraph::new();
    for (a, b, weight) in edges {
        let idx_a = graph.add_node(a.clone());
        let idx_b = graph.add_node(b.clone());
        graph.add_edge(idx_a, idx_b, *weight);
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple 4-node graph with a known minimum cut.
    ///
    /// ```text
    ///   A ---5--- B
    ///   |         |
    ///   5         5
    ///   |         |
    ///   C ---1--- D
    /// ```
    ///
    /// The minimum cut should separate {C,D} from {A,B} by cutting
    /// the edge C-D (weight 1) -- but actually the min cut is the
    /// weakest bridge. With this topology the min cut is 1.0 (edge C-D).
    /// Wait -- A-C has weight 5 and B-D has weight 5. The min cut is
    /// to isolate D: cut B-D(5) + C-D(1) = 6, or isolate C: cut A-C(5) + C-D(1) = 6.
    /// Actually the min cut of this graph is across A-B|C-D: cut A-C(5)+B-D(5)=10.
    /// Or across A-C|B-D: cut A-B(5)+C-D(1)=6. The minimum is 6.
    ///
    /// Let's use a clearer example: two triangles connected by a single weak edge.
    #[test]
    fn test_two_triangles_mincut() {
        // Two tightly connected triangles joined by a weak bridge:
        //
        //   A --10-- B --10-- C      D --10-- E --10-- F
        //   |                 |      |                 |
        //   +------10--------+      +------10--------+
        //                    C --1-- D
        //
        let mut graph = MinCutGraph::new();
        let a = graph.add_node("A".into());
        let b = graph.add_node("B".into());
        let c = graph.add_node("C".into());
        let d = graph.add_node("D".into());
        let e = graph.add_node("E".into());
        let f = graph.add_node("F".into());

        // Triangle 1: A-B-C
        graph.add_edge(a, b, 10.0);
        graph.add_edge(b, c, 10.0);
        graph.add_edge(a, c, 10.0);

        // Triangle 2: D-E-F
        graph.add_edge(d, e, 10.0);
        graph.add_edge(e, f, 10.0);
        graph.add_edge(d, f, 10.0);

        // Weak bridge
        graph.add_edge(c, d, 1.0);

        let result = graph.minimum_cut();
        assert!(result.is_some());

        let (cut_weight, part_a, part_b) = result.unwrap();
        assert!(
            (cut_weight - 1.0).abs() < 1e-9,
            "Expected min cut weight 1.0, got {cut_weight}"
        );

        // One partition should be {A,B,C} and the other {D,E,F}
        let (small, large) = if part_a.len() <= part_b.len() {
            (&part_a, &part_b)
        } else {
            (&part_b, &part_a)
        };
        assert_eq!(small.len(), 3);
        assert_eq!(large.len(), 3);
    }

    #[test]
    fn test_single_node_no_cut() {
        let mut graph = MinCutGraph::new();
        graph.add_node("lonely".into());

        assert!(graph.minimum_cut().is_none());
    }

    #[test]
    fn test_two_disconnected_components() {
        let mut graph = MinCutGraph::new();
        let a = graph.add_node("A".into());
        let b = graph.add_node("B".into());
        let c = graph.add_node("C".into());
        let d = graph.add_node("D".into());

        // Component 1: A-B
        graph.add_edge(a, b, 5.0);
        // Component 2: C-D
        graph.add_edge(c, d, 5.0);

        let result = graph.minimum_cut();
        assert!(result.is_some());

        let (cut_weight, _, _) = result.unwrap();
        assert!(
            cut_weight.abs() < 1e-9,
            "Disconnected components should have min cut 0.0, got {cut_weight}"
        );
    }

    #[test]
    fn test_cluster_discovery_with_threshold() {
        // Two tight clusters connected by a weak edge
        let mut graph = MinCutGraph::new();
        let a = graph.add_node("A".into());
        let b = graph.add_node("B".into());
        let c = graph.add_node("C".into());
        let d = graph.add_node("D".into());

        // Cluster 1: A-B strongly connected
        graph.add_edge(a, b, 10.0);
        // Cluster 2: C-D strongly connected
        graph.add_edge(c, d, 10.0);
        // Weak bridge
        graph.add_edge(b, c, 0.5);

        // With threshold 1.0, should split into 2 clusters
        let clusters = graph.discover_clusters(1.0);
        assert_eq!(
            clusters.len(),
            2,
            "Expected 2 clusters, got {}",
            clusters.len()
        );

        // Each cluster should have 2 nodes
        for cluster in &clusters {
            assert_eq!(cluster.node_ids.len(), 2);
        }
    }

    #[test]
    fn test_cluster_discovery_high_threshold() {
        // With a very high threshold, everything splits into individual nodes
        let mut graph = MinCutGraph::new();
        let a = graph.add_node("A".into());
        let b = graph.add_node("B".into());
        let c = graph.add_node("C".into());

        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);

        // Threshold higher than any cut -- should split maximally
        let clusters = graph.discover_clusters(100.0);
        assert_eq!(clusters.len(), 3, "High threshold should split into 3 individual nodes");
    }

    #[test]
    fn test_cluster_discovery_low_threshold() {
        // With a very low threshold, everything stays as one cluster
        let mut graph = MinCutGraph::new();
        let a = graph.add_node("A".into());
        let b = graph.add_node("B".into());
        let c = graph.add_node("C".into());

        graph.add_edge(a, b, 5.0);
        graph.add_edge(b, c, 5.0);
        graph.add_edge(a, c, 5.0);

        // Threshold lower than min cut -- should stay as one cluster
        let clusters = graph.discover_clusters(0.1);
        assert_eq!(clusters.len(), 1, "Low threshold should keep as 1 cluster");
        assert_eq!(clusters[0].node_ids.len(), 3);
    }

    #[test]
    fn test_empty_graph() {
        let graph = MinCutGraph::new();
        assert!(graph.minimum_cut().is_none());
        let clusters = graph.discover_clusters(1.0);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_duplicate_node_add() {
        let mut graph = MinCutGraph::new();
        let idx1 = graph.add_node("A".into());
        let idx2 = graph.add_node("A".into());
        assert_eq!(idx1, idx2);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_edge_weight_accumulation() {
        let mut graph = MinCutGraph::new();
        let a = graph.add_node("A".into());
        let b = graph.add_node("B".into());
        let c = graph.add_node("C".into());

        // Add edges with accumulated weights
        graph.add_edge(a, b, 3.0);
        graph.add_edge(a, b, 2.0); // Should accumulate to 5.0
        graph.add_edge(a, c, 1.0);

        let result = graph.minimum_cut();
        assert!(result.is_some());
        let (cut_weight, _, _) = result.unwrap();
        // Min cut should isolate C: cut A-C(1.0) = 1.0
        assert!(
            (cut_weight - 1.0).abs() < 1e-9,
            "Expected min cut 1.0 after weight accumulation, got {cut_weight}"
        );
    }

    #[test]
    fn test_cluster_internal_weight() {
        let mut graph = MinCutGraph::new();
        let a = graph.add_node("A".into());
        let b = graph.add_node("B".into());
        let c = graph.add_node("C".into());

        graph.add_edge(a, b, 3.0);
        graph.add_edge(b, c, 2.0);
        graph.add_edge(a, c, 4.0);

        // One cluster with all nodes
        let clusters = graph.discover_clusters(0.01);
        assert_eq!(clusters.len(), 1);
        // Internal weight: 3+2+4 = 9
        assert!(
            (clusters[0].internal_weight - 9.0).abs() < 1e-9,
            "Expected internal weight 9.0, got {}",
            clusters[0].internal_weight
        );
    }

    #[test]
    fn test_self_loop_ignored() {
        let mut graph = MinCutGraph::new();
        let a = graph.add_node("A".into());
        let b = graph.add_node("B".into());

        graph.add_edge(a, a, 100.0); // Self-loop, should be ignored
        graph.add_edge(a, b, 1.0);

        assert_eq!(graph.node_count(), 2);
        let result = graph.minimum_cut();
        assert!(result.is_some());
        let (cut_weight, _, _) = result.unwrap();
        assert!(
            (cut_weight - 1.0).abs() < 1e-9,
            "Self-loop should not affect min cut, got {cut_weight}"
        );
    }

    #[test]
    fn test_from_in_memory_graph_basic() {
        use crate::graph::pressure::InMemoryGraph;

        let mut g = InMemoryGraph::new();
        g.add_edge("A", "B", 0.8);
        g.add_edge("B", "C", 0.5);

        let mcg = super::from_in_memory_graph(&g);

        // All three nodes should be present
        assert_eq!(mcg.node_count(), 3);
        // Undirected edges should exist
        assert!(mcg.minimum_cut().is_some());
    }

    #[test]
    fn test_from_in_memory_graph_empty() {
        use crate::graph::pressure::InMemoryGraph;

        let g = InMemoryGraph::new();
        let mcg = super::from_in_memory_graph(&g);
        assert_eq!(mcg.node_count(), 0);
        assert!(mcg.minimum_cut().is_none());
    }

    #[test]
    fn test_from_in_memory_graph_isolated_node() {
        use crate::graph::pressure::InMemoryGraph;

        let mut g = InMemoryGraph::new();
        g.add_node("lonely");

        let mcg = super::from_in_memory_graph(&g);
        assert_eq!(mcg.node_count(), 1);
        assert!(mcg.minimum_cut().is_none());
    }

    #[test]
    fn test_from_in_memory_graph_bidirectional_accumulates() {
        use crate::graph::pressure::InMemoryGraph;

        // Both A->B (0.8) and B->A (0.6) become one undirected edge
        // with accumulated weight 0.8 + 0.6 = 1.4
        let mut g = InMemoryGraph::new();
        g.add_edge("A", "B", 0.8);
        g.add_edge("B", "A", 0.6);
        g.add_edge("A", "C", 0.3);

        let mcg = super::from_in_memory_graph(&g);
        assert_eq!(mcg.node_count(), 3);

        // Min cut should isolate C: cut A-C(0.3) = 0.3
        let (cut_weight, _, _) = mcg.minimum_cut().unwrap();
        assert!(
            (cut_weight - 0.3).abs() < 1e-9,
            "Expected min cut 0.3, got {cut_weight}"
        );
    }
}
