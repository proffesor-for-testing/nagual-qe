//! History Cone - Causal Past Reasoning
//!
//! The history cone represents all causally connected past events that could
//! have influenced the current state (the center of the light cone).
//!
//! In physics, the past light cone contains all events that could have sent
//! a signal reaching the observer. Similarly, the history cone in ProfDAG
//! contains all nodes that could have causally influenced the center node.
//!
//! # Key Concepts
//!
//! - **Temporal Nodes**: Past events with temporal distance from the center
//! - **Causal Chains**: Sequences of causally connected events
//! - **Root Causes**: Original events that started causal chains
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::profdag::history_cone::{HistoryCone, HistoryConeConfig};
//!
//! let config = HistoryConeConfig::default().with_max_depth(5);
//! let history = HistoryCone::new("center_node", config);
//!
//! // Trace back through causal history
//! let ancestors = history.trace_back("current_node", 3)?;
//!
//! // Find root causes
//! let roots = history.find_root_causes("problem_node");
//!
//! // Get the causal path
//! if let Some(path) = history.get_causal_path("cause", "effect") {
//!     println!("Path: {:?}", path.nodes);
//! }
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::light_cone::NodeId;
use super::{EdgeType, ProfDAGNode, ProfDAGResult};

/// Configuration for the history cone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryConeConfig {
    /// Maximum depth to traverse backward.
    pub max_depth: usize,

    /// Maximum ancestors to fetch per node.
    pub max_ancestors_per_node: usize,

    /// Whether to include similar_to edges in causal analysis.
    pub include_similarity_edges: bool,

    /// Whether to include wormhole edges.
    pub include_wormhole_edges: bool,

    /// Minimum edge weight to consider causal.
    pub min_causal_weight: f64,

    /// Weight decay factor per depth level.
    pub depth_decay_factor: f64,
}

impl Default for HistoryConeConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            max_ancestors_per_node: 20,
            include_similarity_edges: false,
            include_wormhole_edges: true,
            min_causal_weight: 0.3,
            depth_decay_factor: 0.8,
        }
    }
}

impl HistoryConeConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum traversal depth.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set maximum ancestors per node.
    pub fn with_max_ancestors(mut self, max: usize) -> Self {
        self.max_ancestors_per_node = max;
        self
    }

    /// Enable or disable similarity edges.
    pub fn with_similarity_edges(mut self, include: bool) -> Self {
        self.include_similarity_edges = include;
        self
    }

    /// Enable or disable wormhole edges.
    pub fn with_wormhole_edges(mut self, include: bool) -> Self {
        self.include_wormhole_edges = include;
        self
    }

    /// Set minimum causal weight.
    pub fn with_min_causal_weight(mut self, weight: f64) -> Self {
        self.min_causal_weight = weight.clamp(0.0, 1.0);
        self
    }
}

/// A node in the temporal context with depth information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalNode {
    /// Node ID.
    pub id: NodeId,

    /// Node type (pattern, trajectory, prediction, decision).
    pub node_type: String,

    /// Content/description.
    pub content: String,

    /// Depth from the center (0 = center).
    pub depth: usize,

    /// Confidence score.
    pub confidence: f32,

    /// Importance score.
    pub importance: f32,

    /// When the original event occurred.
    pub created_at: DateTime<Utc>,

    /// Temporal distance from center in hours (negative = past).
    pub temporal_distance_hours: Option<i32>,

    /// Causal weight (how strongly this contributed to the center).
    pub causal_weight: f64,

    /// Source type if this node references another entity.
    pub source_type: Option<String>,

    /// Source ID if this node references another entity.
    pub source_id: Option<String>,
}

impl TemporalNode {
    /// Create a new temporal node.
    pub fn new(id: impl Into<NodeId>, content: impl Into<String>, depth: usize) -> Self {
        Self {
            id: id.into(),
            node_type: "pattern".to_string(),
            content: content.into(),
            depth,
            confidence: 0.5,
            importance: 0.5,
            created_at: Utc::now(),
            temporal_distance_hours: None,
            causal_weight: 1.0,
            source_type: None,
            source_id: None,
        }
    }

    /// Create from a ProfDAGNode.
    pub fn from_profdag_node(node: &ProfDAGNode, depth: usize) -> Self {
        Self {
            id: node.id.clone(),
            node_type: node.node_type.as_str().to_string(),
            content: node.content.clone(),
            depth,
            confidence: node.confidence,
            importance: node.importance,
            created_at: node.created_at,
            temporal_distance_hours: None,
            causal_weight: 1.0,
            source_type: node.source_type.clone(),
            source_id: node.source_id.clone(),
        }
    }

    /// Set the temporal distance.
    pub fn with_temporal_distance(mut self, hours: i32) -> Self {
        self.temporal_distance_hours = Some(hours);
        self
    }

    /// Set the causal weight.
    pub fn with_causal_weight(mut self, weight: f64) -> Self {
        self.causal_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Get the quality score (average of confidence and importance).
    pub fn quality_score(&self) -> f32 {
        (self.confidence + self.importance) / 2.0
    }
}

/// A causal chain representing a sequence of causally connected events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalChain {
    /// ID of this causal chain.
    pub id: String,

    /// Source node ID (cause).
    pub source_id: NodeId,

    /// Target node ID (effect).
    pub target_id: NodeId,

    /// Edge type that connects them.
    pub edge_type: EdgeType,

    /// Weight of the causal connection.
    pub weight: f64,

    /// Sequence of node IDs in the chain (if multi-hop).
    pub path: Vec<NodeId>,

    /// Total weight along the path (product of edge weights).
    pub total_weight: f64,

    /// Metadata about this causal relationship.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl CausalChain {
    /// Create a new causal chain between two nodes.
    pub fn new(source_id: impl Into<NodeId>, target_id: impl Into<NodeId>) -> Self {
        let source = source_id.into();
        let target = target_id.into();

        Self {
            id: format!("chain_{}_{}", &source[..8.min(source.len())], &target[..8.min(target.len())]),
            source_id: source.clone(),
            target_id: target.clone(),
            edge_type: EdgeType::LeadsTo,
            weight: 1.0,
            path: vec![source, target],
            total_weight: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Set the edge type.
    pub fn with_edge_type(mut self, edge_type: EdgeType) -> Self {
        self.edge_type = edge_type;
        self
    }

    /// Set the weight.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self.total_weight = weight;
        self
    }

    /// Set the full path.
    pub fn with_path(mut self, path: Vec<NodeId>) -> Self {
        self.path = path;
        self
    }

    /// Extend the chain with another node.
    pub fn extend(&mut self, node_id: impl Into<NodeId>, edge_weight: f64) {
        self.path.push(node_id.into());
        self.total_weight *= edge_weight;
        self.target_id = self.path.last().cloned().unwrap_or_default();
    }

    /// Get the chain length (number of hops).
    pub fn len(&self) -> usize {
        if self.path.len() <= 1 {
            0
        } else {
            self.path.len() - 1
        }
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }

    /// Check if a node is in this chain.
    pub fn contains(&self, node_id: &str) -> bool {
        self.path.iter().any(|id| id == node_id)
    }
}

/// The History Cone containing all causally connected past events.
#[derive(Debug)]
pub struct HistoryCone {
    /// The center node (present moment).
    center: NodeId,

    /// All nodes in the history cone, indexed by ID.
    nodes: HashMap<NodeId, TemporalNode>,

    /// All causal chains discovered.
    causal_chains: Vec<CausalChain>,

    /// Index: target_id -> chains ending at target.
    chains_by_target: HashMap<NodeId, Vec<usize>>,

    /// Index: source_id -> chains starting from source.
    chains_by_source: HashMap<NodeId, Vec<usize>>,

    /// Configuration.
    config: HistoryConeConfig,
}

impl HistoryCone {
    /// Create a new history cone centered on a node.
    pub fn new(center: impl Into<NodeId>, config: HistoryConeConfig) -> Self {
        Self {
            center: center.into(),
            nodes: HashMap::new(),
            causal_chains: Vec::new(),
            chains_by_target: HashMap::new(),
            chains_by_source: HashMap::new(),
            config,
        }
    }

    /// Get the center node ID.
    pub fn center(&self) -> &NodeId {
        &self.center
    }

    /// Add a node to the history cone.
    pub fn add_node(&mut self, node: TemporalNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Add a causal chain.
    pub fn add_causal_chain(&mut self, chain: CausalChain) {
        let index = self.causal_chains.len();

        // Update indexes
        self.chains_by_target
            .entry(chain.target_id.clone())
            .or_default()
            .push(index);

        self.chains_by_source
            .entry(chain.source_id.clone())
            .or_default()
            .push(index);

        self.causal_chains.push(chain);
    }

    /// Check if a node exists in the history cone.
    pub fn contains_node(&self, node_id: &str) -> bool {
        self.nodes.contains_key(node_id)
    }

    /// Get a node by ID.
    pub fn get_node(&self, node_id: &str) -> Option<&TemporalNode> {
        self.nodes.get(node_id)
    }

    /// Get all nodes in the history cone.
    pub fn nodes(&self) -> impl Iterator<Item = &TemporalNode> {
        self.nodes.values()
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of causal chains.
    pub fn chain_count(&self) -> usize {
        self.causal_chains.len()
    }

    /// Get the maximum depth in the history cone.
    pub fn max_depth(&self) -> usize {
        self.nodes.values().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Trace back from a node to find its causal ancestors.
    ///
    /// Returns nodes ordered by depth (closest first).
    pub fn trace_back(&self, node_id: &str, max_depth: usize) -> ProfDAGResult<Vec<TemporalNode>> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start from the given node
        if let Some(start_node) = self.nodes.get(node_id) {
            queue.push_back((start_node.id.clone(), 0_usize));
            visited.insert(start_node.id.clone());
        }

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth > max_depth {
                continue;
            }

            // Find chains ending at this node (incoming causal edges)
            if let Some(chain_indices) = self.chains_by_target.get(&current_id) {
                for &idx in chain_indices {
                    let chain = &self.causal_chains[idx];

                    // The source of this chain is an ancestor
                    if !visited.contains(&chain.source_id) {
                        visited.insert(chain.source_id.clone());

                        if let Some(ancestor) = self.nodes.get(&chain.source_id) {
                            result.push(ancestor.clone());
                            queue.push_back((ancestor.id.clone(), depth + 1));
                        }
                    }
                }
            }
        }

        // Sort by depth (closest first)
        result.sort_by_key(|n| n.depth);

        Ok(result)
    }

    /// Find the root causes (nodes with no incoming causal edges).
    ///
    /// These are the original events that started causal chains leading
    /// to the given node.
    pub fn find_root_causes(&self, node_id: &str) -> Vec<NodeId> {
        let mut roots = Vec::new();

        // Find all ancestors
        if let Ok(ancestors) = self.trace_back(node_id, self.config.max_depth) {
            for ancestor in ancestors {
                // Check if this ancestor has no incoming edges in our cone
                let has_incoming = self
                    .chains_by_target
                    .get(&ancestor.id)
                    .map(|chains| !chains.is_empty())
                    .unwrap_or(false);

                if !has_incoming {
                    roots.push(ancestor.id);
                }
            }
        }

        // If no roots found through ancestors, use nodes at max depth
        if roots.is_empty() {
            for node in self.nodes.values() {
                if node.depth == self.max_depth() {
                    roots.push(node.id.clone());
                }
            }
        }

        roots
    }

    /// Get the causal path between two nodes.
    ///
    /// Returns the chain connecting source to target, if one exists
    /// within the history cone.
    pub fn get_causal_path(&self, from: &str, to: &str) -> Option<CausalChain> {
        // BFS to find path
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent_map: HashMap<NodeId, (NodeId, f64)> = HashMap::new();

        queue.push_back(from.to_string());
        visited.insert(from.to_string());

        while let Some(current) = queue.pop_front() {
            if current == to {
                // Reconstruct path
                let mut path = vec![to.to_string()];
                let mut total_weight = 1.0;
                let mut current_node = to.to_string();

                while let Some((parent, weight)) = parent_map.get(&current_node) {
                    path.push(parent.clone());
                    total_weight *= weight;
                    current_node = parent.clone();

                    if current_node == from {
                        break;
                    }
                }

                path.reverse();

                let chain = CausalChain {
                    id: format!("path_{}_{}", from, to),
                    source_id: from.to_string(),
                    target_id: to.to_string(),
                    edge_type: EdgeType::LeadsTo,
                    weight: total_weight,
                    path,
                    total_weight,
                    metadata: HashMap::new(),
                };

                return Some(chain);
            }

            // Explore outgoing edges (forward direction)
            if let Some(chain_indices) = self.chains_by_source.get(&current) {
                for &idx in chain_indices {
                    let chain = &self.causal_chains[idx];

                    if !visited.contains(&chain.target_id) {
                        visited.insert(chain.target_id.clone());
                        parent_map.insert(chain.target_id.clone(), (current.clone(), chain.weight));
                        queue.push_back(chain.target_id.clone());
                    }
                }
            }
        }

        None
    }

    /// Get all causal chains.
    pub fn chains(&self) -> &[CausalChain] {
        &self.causal_chains
    }

    /// Get chains by target node.
    pub fn get_chains_to(&self, target_id: &str) -> Vec<&CausalChain> {
        self.chains_by_target
            .get(target_id)
            .map(|indices| indices.iter().map(|&i| &self.causal_chains[i]).collect())
            .unwrap_or_default()
    }

    /// Get chains by source node.
    pub fn get_chains_from(&self, source_id: &str) -> Vec<&CausalChain> {
        self.chains_by_source
            .get(source_id)
            .map(|indices| indices.iter().map(|&i| &self.causal_chains[i]).collect())
            .unwrap_or_default()
    }

    /// Calculate the causal strength from a node to the center.
    ///
    /// This is the product of edge weights along the strongest path.
    pub fn causal_strength_to_center(&self, node_id: &str) -> f64 {
        if node_id == self.center {
            return 1.0;
        }

        if let Some(chain) = self.get_causal_path(node_id, &self.center) {
            chain.total_weight
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_history_cone_config_default() {
        let config = HistoryConeConfig::default();

        assert_eq!(config.max_depth, 5);
        assert_eq!(config.max_ancestors_per_node, 20);
        assert!(!config.include_similarity_edges);
        assert!(config.include_wormhole_edges);
    }

    #[test]
    fn test_history_cone_config_builder() {
        let config = HistoryConeConfig::new()
            .with_max_depth(10)
            .with_max_ancestors(50)
            .with_similarity_edges(true);

        assert_eq!(config.max_depth, 10);
        assert_eq!(config.max_ancestors_per_node, 50);
        assert!(config.include_similarity_edges);
    }

    #[test]
    fn test_temporal_node_creation() {
        let node = TemporalNode::new("node-1", "Test content", 2)
            .with_temporal_distance(-24)
            .with_causal_weight(0.8);

        assert_eq!(node.id, "node-1");
        assert_eq!(node.depth, 2);
        assert_eq!(node.temporal_distance_hours, Some(-24));
        assert!((node.causal_weight - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_temporal_node_quality_score() {
        let mut node = TemporalNode::new("node-1", "Test", 0);
        node.confidence = 0.8;
        node.importance = 0.6;

        assert!((node.quality_score() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_causal_chain_creation() {
        let chain = CausalChain::new("cause", "effect")
            .with_edge_type(EdgeType::LeadsTo)
            .with_weight(0.9);

        assert_eq!(chain.source_id, "cause");
        assert_eq!(chain.target_id, "effect");
        assert_eq!(chain.edge_type, EdgeType::LeadsTo);
        assert!((chain.weight - 0.9).abs() < 0.001);
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn test_causal_chain_extend() {
        let mut chain = CausalChain::new("a", "b").with_weight(0.8);
        chain.extend("c", 0.7);

        assert_eq!(chain.path, vec!["a", "b", "c"]);
        assert_eq!(chain.target_id, "c");
        assert!((chain.total_weight - 0.56).abs() < 0.001); // 0.8 * 0.7
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_history_cone_basic() {
        let config = HistoryConeConfig::default();
        let mut cone = HistoryCone::new("center", config);

        let node1 = TemporalNode::new("node-1", "Content 1", 1);
        let node2 = TemporalNode::new("node-2", "Content 2", 2);

        cone.add_node(node1);
        cone.add_node(node2);

        assert_eq!(cone.node_count(), 2);
        assert!(cone.contains_node("node-1"));
        assert!(cone.contains_node("node-2"));
        assert!(!cone.contains_node("node-3"));
    }

    #[test]
    fn test_history_cone_chains() {
        let config = HistoryConeConfig::default();
        let mut cone = HistoryCone::new("center", config);

        let node1 = TemporalNode::new("node-1", "Content 1", 1);
        let node2 = TemporalNode::new("node-2", "Content 2", 2);

        cone.add_node(node1);
        cone.add_node(node2);

        let chain = CausalChain::new("node-2", "node-1").with_weight(0.8);
        cone.add_causal_chain(chain);

        assert_eq!(cone.chain_count(), 1);

        let chains_to_1 = cone.get_chains_to("node-1");
        assert_eq!(chains_to_1.len(), 1);
        assert_eq!(chains_to_1[0].source_id, "node-2");
    }

    #[test]
    fn test_history_cone_max_depth() {
        let config = HistoryConeConfig::default();
        let mut cone = HistoryCone::new("center", config);

        cone.add_node(TemporalNode::new("n0", "0", 0));
        cone.add_node(TemporalNode::new("n1", "1", 1));
        cone.add_node(TemporalNode::new("n2", "2", 3));
        cone.add_node(TemporalNode::new("n3", "3", 5));

        assert_eq!(cone.max_depth(), 5);
    }

    #[test]
    fn test_trace_back() {
        let config = HistoryConeConfig::default();
        let mut cone = HistoryCone::new("center", config);

        // Add nodes at different depths
        let center = TemporalNode::new("center", "Center", 0);
        let parent = TemporalNode::new("parent", "Parent", 1);
        let grandparent = TemporalNode::new("grandparent", "Grandparent", 2);

        cone.add_node(center);
        cone.add_node(parent);
        cone.add_node(grandparent);

        // Add causal chains
        cone.add_causal_chain(CausalChain::new("parent", "center"));
        cone.add_causal_chain(CausalChain::new("grandparent", "parent"));

        let ancestors = cone.trace_back("center", 3).unwrap();
        assert_eq!(ancestors.len(), 2);
        assert_eq!(ancestors[0].id, "parent"); // depth 1
        assert_eq!(ancestors[1].id, "grandparent"); // depth 2
    }

    #[test]
    fn test_find_root_causes() {
        let config = HistoryConeConfig::default();
        let mut cone = HistoryCone::new("effect", config);

        // Build a simple causal chain: root -> middle -> effect
        cone.add_node(TemporalNode::new("root", "Root cause", 2));
        cone.add_node(TemporalNode::new("middle", "Middle", 1));
        cone.add_node(TemporalNode::new("effect", "Effect", 0));

        cone.add_causal_chain(CausalChain::new("root", "middle"));
        cone.add_causal_chain(CausalChain::new("middle", "effect"));

        let roots = cone.find_root_causes("effect");

        // root has no incoming edges in our cone
        assert!(roots.contains(&"root".to_string()));
    }

    #[test]
    fn test_get_causal_path() {
        let config = HistoryConeConfig::default();
        let mut cone = HistoryCone::new("c", config);

        cone.add_node(TemporalNode::new("a", "A", 2));
        cone.add_node(TemporalNode::new("b", "B", 1));
        cone.add_node(TemporalNode::new("c", "C", 0));

        cone.add_causal_chain(CausalChain::new("a", "b").with_weight(0.8));
        cone.add_causal_chain(CausalChain::new("b", "c").with_weight(0.9));

        let path = cone.get_causal_path("a", "c");
        assert!(path.is_some());

        let chain = path.unwrap();
        assert_eq!(chain.source_id, "a");
        assert_eq!(chain.target_id, "c");
        assert_eq!(chain.path, vec!["a", "b", "c"]);
        assert!((chain.total_weight - 0.72).abs() < 0.001); // 0.8 * 0.9
    }

    #[test]
    fn test_causal_strength_to_center() {
        let config = HistoryConeConfig::default();
        let mut cone = HistoryCone::new("center", config);

        cone.add_node(TemporalNode::new("a", "A", 1));
        cone.add_node(TemporalNode::new("center", "Center", 0));

        cone.add_causal_chain(CausalChain::new("a", "center").with_weight(0.75));

        let strength = cone.causal_strength_to_center("a");
        assert!((strength - 0.75).abs() < 0.001);

        // Center to itself should be 1.0
        let self_strength = cone.causal_strength_to_center("center");
        assert!((self_strength - 1.0).abs() < 0.001);
    }
}
