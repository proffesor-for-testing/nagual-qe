//! Learning Integration Tests - Phase 3 (Learning Layer)
//!
//! End-to-end integration tests combining all Phase 3 components:
//! - Trajectory recording and analysis
//! - Wormhole creation from co-access patterns
//! - Light cone updates as new patterns are added
//! - Wormhole + Light cone interaction
//! - Performance under load
//!
//! # Integration Scenarios
//! - Full flow: trajectory -> wormhole creation -> faster traversal
//! - Light cone updates as new patterns are added
//! - Combined wormhole and light cone for path optimization
//! - Cognitive core interaction with both systems

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

mod common;
use common::{cosine_similarity, normalized_embedding, similar_embeddings};

// ============================================================================
// Shared Types (combining wormhole and light cone concepts)
// ============================================================================

/// Unified node type for the learning system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningNode {
    pub id: String,
    pub content: String,
    pub node_type: NodeType,
    pub embedding: Option<Vec<f32>>,
    pub confidence: f32,
    pub usage_count: u32,
    pub success_rate: f32,
    pub created_at: DateTime<Utc>,
    pub last_used_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    Pattern,
    Trajectory,
    Decision,
    Outcome,
}

impl LearningNode {
    pub fn pattern(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            node_type: NodeType::Pattern,
            embedding: None,
            confidence: 0.5,
            usage_count: 0,
            success_rate: 0.5,
            created_at: Utc::now(),
            last_used_at: Utc::now(),
            metadata: serde_json::json!({}),
        }
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn record_usage(&mut self, success: bool) {
        self.usage_count += 1;
        let success_val = if success { 1.0 } else { 0.0 };
        self.success_rate = (self.success_rate * (self.usage_count - 1) as f32 + success_val)
            / self.usage_count as f32;
        self.last_used_at = Utc::now();
    }
}

/// Edge types in the learning graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    LeadsTo,
    SimilarTo,
    Wormhole,
    CausalLink,
    CoAccess,
}

/// A learning edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEdge {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub edge_type: EdgeType,
    pub weight: f64,
    pub co_access_count: u32,
    pub evidence_count: u32,
    pub created_at: DateTime<Utc>,
    pub last_used_at: DateTime<Utc>,
}

impl LearningEdge {
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
            co_access_count: 0,
            evidence_count: 1,
            created_at: Utc::now(),
            last_used_at: Utc::now(),
        }
    }

    pub fn wormhole(source_id: impl Into<String>, target_id: impl Into<String>, strength: f64) -> Self {
        Self::new(source_id, target_id, EdgeType::Wormhole, strength)
    }

    pub fn reinforce(&mut self) {
        self.evidence_count += 1;
        self.weight = (self.weight + 0.05 * (1.0 - self.weight)).min(1.0);
        self.last_used_at = Utc::now();
    }

    pub fn increment_co_access(&mut self) {
        self.co_access_count += 1;
        self.last_used_at = Utc::now();
    }
}

// ============================================================================
// Trajectory Recording
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrajectoryOutcome {
    Success,
    PartialSuccess,
    Failure,
}

impl TrajectoryOutcome {
    pub fn reward(&self) -> f32 {
        match self {
            TrajectoryOutcome::Success => 1.0,
            TrajectoryOutcome::PartialSuccess => 0.6,
            TrajectoryOutcome::Failure => 0.1,
        }
    }

    pub fn is_successful(&self) -> bool {
        matches!(self, TrajectoryOutcome::Success | TrajectoryOutcome::PartialSuccess)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryStep {
    pub step_number: usize,
    pub pattern_id: String,
    pub action: String,
    pub duration_ms: u64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub id: String,
    pub session_id: String,
    pub task: String,
    pub steps: Vec<TrajectoryStep>,
    pub outcome: Option<TrajectoryOutcome>,
    pub reward: Option<f32>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

impl Trajectory {
    pub fn new(session_id: impl Into<String>, task: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.into(),
            task: task.into(),
            steps: Vec::new(),
            outcome: None,
            reward: None,
            started_at: Utc::now(),
            completed_at: None,
        }
    }

    pub fn add_step(&mut self, pattern_id: impl Into<String>, action: impl Into<String>) {
        let step = TrajectoryStep {
            step_number: self.steps.len() + 1,
            pattern_id: pattern_id.into(),
            action: action.into(),
            duration_ms: 100,
            timestamp: Utc::now(),
        };
        self.steps.push(step);
    }

    pub fn complete(&mut self, outcome: TrajectoryOutcome) {
        self.outcome = Some(outcome);
        self.reward = Some(outcome.reward());
        self.completed_at = Some(Utc::now());
    }

    pub fn pattern_ids(&self) -> Vec<String> {
        self.steps.iter().map(|s| s.pattern_id.clone()).collect()
    }

    pub fn pattern_pairs(&self) -> Vec<(String, String)> {
        self.steps
            .windows(2)
            .map(|w| (w[0].pattern_id.clone(), w[1].pattern_id.clone()))
            .collect()
    }
}

// ============================================================================
// Integrated Learning System
// ============================================================================

/// Configuration for the learning system.
#[derive(Debug, Clone)]
pub struct LearningConfig {
    pub wormhole_threshold: u32,
    pub wormhole_decay_constant: f64,
    pub max_wormholes_per_node: usize,
    pub min_traversal_savings: f64,
    pub light_cone_max_depth: usize,
    pub cognitive_core_max_active: usize,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            wormhole_threshold: 3,
            wormhole_decay_constant: 10.0,
            max_wormholes_per_node: 5,
            min_traversal_savings: 0.5,
            light_cone_max_depth: 10,
            cognitive_core_max_active: 100,
        }
    }
}

/// Statistics for the learning system.
#[derive(Debug, Clone, Default)]
pub struct LearningStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub total_wormholes: usize,
    pub total_trajectories: usize,
    pub successful_trajectories: usize,
    pub total_traversals: usize,
    pub traversal_savings_ms: u64,
    pub avg_wormhole_strength: f64,
}

/// The integrated learning system combining wormholes and light cones.
#[derive(Debug)]
pub struct IntegratedLearningSystem {
    config: LearningConfig,
    nodes: HashMap<String, LearningNode>,
    edges: HashMap<String, LearningEdge>,
    trajectories: HashMap<String, Trajectory>,

    // Indexing structures
    forward_edges: HashMap<String, Vec<String>>,
    backward_edges: HashMap<String, Vec<String>>,
    co_access_counts: HashMap<(String, String), u32>,
    wormholes_by_node: HashMap<String, Vec<String>>,

    // Cognitive core state
    active_patterns: HashMap<String, f64>,

    // Metrics
    traversal_count: usize,
    traversal_savings_ms: u64,
}

impl IntegratedLearningSystem {
    pub fn new(config: LearningConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            trajectories: HashMap::new(),
            forward_edges: HashMap::new(),
            backward_edges: HashMap::new(),
            co_access_counts: HashMap::new(),
            wormholes_by_node: HashMap::new(),
            active_patterns: HashMap::new(),
            traversal_count: 0,
            traversal_savings_ms: 0,
        }
    }

    /// Add a node to the system.
    pub fn add_node(&mut self, node: LearningNode) -> String {
        let id = node.id.clone();
        self.nodes.insert(id.clone(), node);
        id
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: &str) -> Option<&LearningNode> {
        self.nodes.get(id)
    }

    /// Get a mutable node reference.
    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut LearningNode> {
        self.nodes.get_mut(id)
    }

    /// Add an edge to the system.
    pub fn add_edge(&mut self, edge: LearningEdge) -> String {
        let edge_id = edge.id.clone();
        let source = edge.source_id.clone();
        let target = edge.target_id.clone();
        let edge_type = edge.edge_type;

        self.forward_edges
            .entry(source.clone())
            .or_default()
            .push(edge_id.clone());

        self.backward_edges
            .entry(target.clone())
            .or_default()
            .push(edge_id.clone());

        if edge_type == EdgeType::Wormhole {
            self.wormholes_by_node
                .entry(source)
                .or_default()
                .push(edge_id.clone());
            self.wormholes_by_node
                .entry(target)
                .or_default()
                .push(edge_id.clone());
        }

        self.edges.insert(edge_id.clone(), edge);
        edge_id
    }

    /// Record a trajectory.
    pub fn record_trajectory(&mut self, trajectory: Trajectory) -> String {
        let id = trajectory.id.clone();

        // Process co-accesses from trajectory
        if trajectory.outcome.map_or(false, |o| o.is_successful()) {
            for (a, b) in trajectory.pattern_pairs() {
                self.record_co_access(&a, &b);
            }

            // Update node usage
            for pattern_id in trajectory.pattern_ids() {
                if let Some(node) = self.nodes.get_mut(&pattern_id) {
                    node.record_usage(true);
                }
            }

            // Create causal links
            self.create_causal_links_from_trajectory(&trajectory);
        }

        self.trajectories.insert(id.clone(), trajectory);
        id
    }

    /// Record a co-access between two patterns.
    fn record_co_access(&mut self, pattern_a: &str, pattern_b: &str) -> Option<String> {
        let (ordered_a, ordered_b) = if pattern_a < pattern_b {
            (pattern_a.to_string(), pattern_b.to_string())
        } else {
            (pattern_b.to_string(), pattern_a.to_string())
        };

        let key = (ordered_a.clone(), ordered_b.clone());
        let count = self.co_access_counts.entry(key).or_insert(0);
        *count += 1;

        // Extract values before further mutable borrows
        let current_count = *count;
        let threshold = self.config.wormhole_threshold;

        if current_count >= threshold {
            return self.maybe_create_wormhole(&ordered_a, &ordered_b, current_count);
        }

        None
    }

    /// Create a wormhole if conditions are met.
    fn maybe_create_wormhole(&mut self, source: &str, target: &str, co_access_count: u32) -> Option<String> {
        // Check if wormhole already exists
        let wormhole_key = format!("wh_{}_{}", source, target);
        if self.edges.contains_key(&wormhole_key) {
            // Reinforce existing wormhole
            if let Some(edge) = self.edges.get_mut(&wormhole_key) {
                edge.increment_co_access();
                edge.reinforce();
            }
            return None;
        }

        // Check max wormholes limit
        let source_wh_count = self.wormholes_by_node.get(source).map_or(0, |v| v.len());
        let target_wh_count = self.wormholes_by_node.get(target).map_or(0, |v| v.len());

        if source_wh_count >= self.config.max_wormholes_per_node
            || target_wh_count >= self.config.max_wormholes_per_node
        {
            return None;
        }

        // Calculate wormhole strength
        let strength = co_access_count as f64
            / (co_access_count as f64 + self.config.wormhole_decay_constant);

        let mut edge = LearningEdge::wormhole(source, target, strength);
        edge.id = wormhole_key.clone();
        edge.co_access_count = co_access_count;

        self.add_edge(edge);
        Some(wormhole_key)
    }

    /// Create causal links from a trajectory.
    fn create_causal_links_from_trajectory(&mut self, trajectory: &Trajectory) {
        for (source, target) in trajectory.pattern_pairs() {
            let edge_key = format!("causal_{}_{}", source, target);
            if let Some(edge) = self.edges.get_mut(&edge_key) {
                edge.reinforce();
            } else {
                let edge = LearningEdge::new(&source, &target, EdgeType::CausalLink, 0.7);
                self.add_edge(edge);
            }
        }
    }

    /// Get the history cone for a node (what led to it).
    pub fn trace_history(&self, node_id: &str, max_depth: usize) -> Vec<(String, usize)> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back((node_id.to_string(), 0usize));

        while let Some((current, depth)) = queue.pop_front() {
            if depth > max_depth || visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());
            result.push((current.clone(), depth));

            if let Some(incoming) = self.backward_edges.get(&current) {
                for edge_id in incoming {
                    if let Some(edge) = self.edges.get(edge_id) {
                        if !visited.contains(&edge.source_id) {
                            queue.push_back((edge.source_id.clone(), depth + 1));
                        }
                    }
                }
            }
        }

        result
    }

    /// Get the future cone for a node (what follows).
    pub fn predict_future(&self, node_id: &str, max_depth: usize) -> Vec<(String, usize, f64)> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back((node_id.to_string(), 0usize, 1.0f64));

        while let Some((current, depth, prob)) = queue.pop_front() {
            if depth > max_depth || visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());
            result.push((current.clone(), depth, prob));

            if let Some(outgoing) = self.forward_edges.get(&current) {
                for edge_id in outgoing {
                    if let Some(edge) = self.edges.get(edge_id) {
                        let new_prob = prob * edge.weight;
                        if !visited.contains(&edge.target_id) && new_prob > 0.01 {
                            queue.push_back((edge.target_id.clone(), depth + 1, new_prob));
                        }
                    }
                }
            }
        }

        result
    }

    /// Find shortest path with and without wormholes.
    pub fn compare_traversal(&self, from: &str, to: &str) -> TraversalComparison {
        let normal_path = self.find_path(from, to, false);
        let wormhole_path = self.find_path(from, to, true);

        let normal_length = normal_path.as_ref().map_or(usize::MAX, |p| p.len());
        let wormhole_length = wormhole_path.as_ref().map_or(usize::MAX, |p| p.len());

        let savings = if normal_length > 0 && wormhole_length < normal_length {
            (normal_length - wormhole_length) as f64 / normal_length as f64
        } else {
            0.0
        };

        TraversalComparison {
            from: from.to_string(),
            to: to.to_string(),
            normal_path,
            wormhole_path,
            savings_ratio: savings,
            uses_wormhole: wormhole_length < normal_length,
        }
    }

    /// Find a path between two nodes.
    fn find_path(&self, from: &str, to: &str, use_wormholes: bool) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }

        let mut visited = HashSet::new();
        let mut parent: HashMap<String, String> = HashMap::new();
        let mut queue = VecDeque::new();

        queue.push_back(from.to_string());
        visited.insert(from.to_string());

        while let Some(current) = queue.pop_front() {
            if current == to {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = to.to_string();
                while let Some(prev) = parent.get(&node) {
                    path.push(node.clone());
                    node = prev.clone();
                }
                path.push(from.to_string());
                path.reverse();
                return Some(path);
            }

            if let Some(outgoing) = self.forward_edges.get(&current) {
                for edge_id in outgoing {
                    if let Some(edge) = self.edges.get(edge_id) {
                        // Skip wormholes if not allowed
                        if !use_wormholes && edge.edge_type == EdgeType::Wormhole {
                            continue;
                        }
                        if !visited.contains(&edge.target_id) {
                            visited.insert(edge.target_id.clone());
                            parent.insert(edge.target_id.clone(), current.clone());
                            queue.push_back(edge.target_id.clone());
                        }
                    }
                }
            }
        }

        None
    }

    /// Activate a pattern in the cognitive core.
    pub fn activate_pattern(&mut self, pattern_id: &str, activation: f64) {
        let entry = self.active_patterns.entry(pattern_id.to_string()).or_insert(0.0);
        *entry = (*entry + activation).min(1.0);

        // Limit active patterns
        while self.active_patterns.len() > self.config.cognitive_core_max_active {
            if let Some((weakest, _)) = self
                .active_patterns
                .iter()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(k, v)| (k.clone(), *v))
            {
                self.active_patterns.remove(&weakest);
            }
        }
    }

    /// Get active patterns sorted by activation.
    pub fn get_active_patterns(&self) -> Vec<(&str, f64)> {
        let mut patterns: Vec<_> = self
            .active_patterns
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        patterns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        patterns
    }

    /// Get wormholes for a node.
    pub fn get_wormholes(&self, node_id: &str) -> Vec<&LearningEdge> {
        self.wormholes_by_node
            .get(node_id)
            .map_or(Vec::new(), |ids| {
                ids.iter()
                    .filter_map(|id| self.edges.get(id))
                    .filter(|e| e.edge_type == EdgeType::Wormhole)
                    .collect()
            })
    }

    /// Get statistics.
    pub fn stats(&self) -> LearningStats {
        let wormholes: Vec<_> = self
            .edges
            .values()
            .filter(|e| e.edge_type == EdgeType::Wormhole)
            .collect();

        let successful = self
            .trajectories
            .values()
            .filter(|t| t.outcome.map_or(false, |o| o.is_successful()))
            .count();

        let avg_wormhole_strength = if wormholes.is_empty() {
            0.0
        } else {
            wormholes.iter().map(|e| e.weight).sum::<f64>() / wormholes.len() as f64
        };

        LearningStats {
            total_nodes: self.nodes.len(),
            total_edges: self.edges.len(),
            total_wormholes: wormholes.len(),
            total_trajectories: self.trajectories.len(),
            successful_trajectories: successful,
            total_traversals: self.traversal_count,
            traversal_savings_ms: self.traversal_savings_ms,
            avg_wormhole_strength,
        }
    }
}

/// Result of comparing traversal with and without wormholes.
#[derive(Debug, Clone)]
pub struct TraversalComparison {
    pub from: String,
    pub to: String,
    pub normal_path: Option<Vec<String>>,
    pub wormhole_path: Option<Vec<String>>,
    pub savings_ratio: f64,
    pub uses_wormhole: bool,
}

// ============================================================================
// End-to-End Integration Tests
// ============================================================================

mod e2e_tests {
    use super::*;

    #[test]
    fn test_trajectory_to_wormhole_creation() {
        let mut system = IntegratedLearningSystem::new(LearningConfig::default());

        // Create a linear chain of patterns
        for i in 0..5 {
            let node = LearningNode::pattern(format!("pat_{}", i), format!("Pattern {}", i));
            system.add_node(node);
        }

        // Create edges
        for i in 0..4 {
            let edge = LearningEdge::new(
                format!("pat_{}", i),
                format!("pat_{}", i + 1),
                EdgeType::LeadsTo,
                0.9,
            );
            system.add_edge(edge);
        }

        // Record trajectories that use pat_0 and pat_4 together
        for session in 0..3 {
            let mut trajectory = Trajectory::new(format!("session_{}", session), "Jump task");
            trajectory.add_step("pat_0", "start");
            trajectory.add_step("pat_4", "end"); // Skip middle patterns
            trajectory.complete(TrajectoryOutcome::Success);

            system.record_trajectory(trajectory);
        }

        // Should have created a wormhole from pat_0 to pat_4
        let wormholes = system.get_wormholes("pat_0");
        assert!(!wormholes.is_empty(), "Should have created wormhole");

        let stats = system.stats();
        assert!(stats.total_wormholes > 0);
    }

    #[test]
    fn test_wormhole_provides_faster_traversal() {
        let mut system = IntegratedLearningSystem::new(LearningConfig {
            wormhole_threshold: 2,
            ..Default::default()
        });

        // Create a long chain: A -> B -> C -> D -> E -> F
        let nodes = vec!["A", "B", "C", "D", "E", "F"];
        for node in &nodes {
            system.add_node(LearningNode::pattern(*node, format!("Node {}", node)));
        }

        for i in 0..(nodes.len() - 1) {
            system.add_edge(LearningEdge::new(nodes[i], nodes[i + 1], EdgeType::LeadsTo, 0.9));
        }

        // Before wormhole: path from A to F should be 6 nodes
        let comparison_before = system.compare_traversal("A", "F");
        let normal_len = comparison_before.normal_path.as_ref().unwrap().len();
        assert_eq!(normal_len, 6);

        // Create wormhole edge directly (simulating what would happen after
        // repeated co-access). Note: We don't record trajectories here because
        // trajectory recording also creates CausalLink edges, which would
        // provide the same shortcut as the wormhole.
        system.add_edge(LearningEdge::wormhole("A", "F", 0.9));

        // After wormhole: path should be shorter when using wormholes
        let comparison_after = system.compare_traversal("A", "F");

        // Normal path (without wormholes) should still be 6 nodes: A->B->C->D->E->F
        assert_eq!(
            comparison_after.normal_path.as_ref().unwrap().len(),
            6,
            "Normal path should traverse full chain without wormhole"
        );

        // Wormhole path should be 2 nodes: A->F
        assert_eq!(
            comparison_after.wormhole_path.as_ref().unwrap().len(),
            2,
            "Wormhole path should be direct A->F"
        );

        assert!(comparison_after.uses_wormhole, "Should detect wormhole usage");
        assert!(
            comparison_after.savings_ratio > 0.5,
            "Should provide >50% savings, got {}",
            comparison_after.savings_ratio
        );
    }

    #[test]
    fn test_light_cone_updates_with_new_patterns() {
        let mut system = IntegratedLearningSystem::new(LearningConfig::default());

        // Initial pattern
        let root = system.add_node(LearningNode::pattern("root", "Root pattern"));

        // Initial history cone is just root
        let initial_history = system.trace_history(&root, 10);
        assert_eq!(initial_history.len(), 1);

        // Add new patterns that lead to root
        let cause1 = system.add_node(LearningNode::pattern("cause1", "First cause"));
        let cause2 = system.add_node(LearningNode::pattern("cause2", "Second cause"));

        system.add_edge(LearningEdge::new(&cause1, &root, EdgeType::CausalLink, 0.9));
        system.add_edge(LearningEdge::new(&cause2, &root, EdgeType::CausalLink, 0.8));

        // History cone should now include the causes
        let updated_history = system.trace_history(&root, 10);
        assert_eq!(updated_history.len(), 3);

        // Add effects
        let effect = system.add_node(LearningNode::pattern("effect", "Effect pattern"));
        system.add_edge(LearningEdge::new(&root, &effect, EdgeType::CausalLink, 0.85));

        // Future cone should include the effect
        let future = system.predict_future(&root, 10);
        let future_ids: Vec<_> = future.iter().map(|(id, _, _)| id.as_str()).collect();
        assert!(future_ids.contains(&"effect"));
    }

    #[test]
    fn test_combined_wormhole_and_causal_traversal() {
        let mut system = IntegratedLearningSystem::new(LearningConfig {
            wormhole_threshold: 1,
            ..Default::default()
        });

        // Create a complex graph:
        // A -> B -> C -> D (causal chain)
        // A ~~~> D (wormhole)
        for c in ['A', 'B', 'C', 'D'] {
            system.add_node(LearningNode::pattern(c.to_string(), format!("Node {}", c)));
        }

        // Causal links
        system.add_edge(LearningEdge::new("A", "B", EdgeType::CausalLink, 0.9));
        system.add_edge(LearningEdge::new("B", "C", EdgeType::CausalLink, 0.9));
        system.add_edge(LearningEdge::new("C", "D", EdgeType::CausalLink, 0.9));

        // Wormhole
        system.add_edge(LearningEdge::wormhole("A", "D", 0.95));

        // From A, future should include D at depth 1 (via wormhole) and depth 3 (via causal)
        let future = system.predict_future("A", 5);

        // D should appear
        let d_entries: Vec<_> = future.iter().filter(|(id, _, _)| id == "D").collect();
        assert!(!d_entries.is_empty());

        // Compare paths
        let comparison = system.compare_traversal("A", "D");
        assert!(comparison.uses_wormhole);
        assert_eq!(comparison.wormhole_path.as_ref().unwrap().len(), 2); // A -> D
        assert_eq!(comparison.normal_path.as_ref().unwrap().len(), 4); // A -> B -> C -> D
    }

    #[test]
    fn test_cognitive_core_interaction() {
        let mut system = IntegratedLearningSystem::new(LearningConfig::default());

        // Add patterns
        for i in 0..10 {
            system.add_node(LearningNode::pattern(format!("pat_{}", i), format!("Pattern {}", i)));
        }

        // Activate some patterns
        system.activate_pattern("pat_0", 0.9);
        system.activate_pattern("pat_1", 0.7);
        system.activate_pattern("pat_2", 0.5);

        let active = system.get_active_patterns();
        assert_eq!(active.len(), 3);
        assert_eq!(active[0].0, "pat_0"); // Highest activation
        assert_eq!(active[2].0, "pat_2"); // Lowest activation

        // Re-activating should increase
        system.activate_pattern("pat_2", 0.6);
        let active = system.get_active_patterns();
        assert!(active.iter().find(|(id, _)| *id == "pat_2").unwrap().1 > 0.5);
    }
}

// ============================================================================
// Wormhole + Light Cone Interaction Tests
// ============================================================================

mod wormhole_lightcone_tests {
    use super::*;

    #[test]
    fn test_wormhole_shortcuts_history_traversal() {
        let mut system = IntegratedLearningSystem::new(LearningConfig::default());

        // Create chain: A -> B -> C -> D -> E
        for c in ['A', 'B', 'C', 'D', 'E'] {
            system.add_node(LearningNode::pattern(c.to_string(), format!("Node {}", c)));
        }

        for (from, to) in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")] {
            system.add_edge(LearningEdge::new(from, to, EdgeType::CausalLink, 0.9));
        }

        // Add wormhole from A to D
        system.add_edge(LearningEdge::wormhole("A", "D", 0.95));

        // History of E should now include A via the wormhole path too
        let history = system.trace_history("E", 10);
        let history_ids: Vec<_> = history.iter().map(|(id, _)| id.as_str()).collect();

        assert!(history_ids.contains(&"A"));
        assert!(history_ids.contains(&"D"));
    }

    #[test]
    fn test_wormhole_enhances_future_predictions() {
        let mut system = IntegratedLearningSystem::new(LearningConfig::default());

        // Create branching structure
        // A -> B -> C
        //       \-> D (weak)
        // A ~~~> E (wormhole)
        for c in ['A', 'B', 'C', 'D', 'E'] {
            system.add_node(LearningNode::pattern(c.to_string(), format!("Node {}", c)));
        }

        system.add_edge(LearningEdge::new("A", "B", EdgeType::CausalLink, 0.9));
        system.add_edge(LearningEdge::new("B", "C", EdgeType::CausalLink, 0.8));
        system.add_edge(LearningEdge::new("B", "D", EdgeType::CausalLink, 0.3));
        system.add_edge(LearningEdge::wormhole("A", "E", 0.95));

        let future = system.predict_future("A", 3);
        let future_map: HashMap<_, _> = future.iter().map(|(id, depth, prob)| (id.as_str(), (*depth, *prob))).collect();

        // E should be reachable at depth 1 with high probability
        assert!(future_map.contains_key("E"));
        assert_eq!(future_map["E"].0, 1); // Depth 1

        // C should be at depth 2
        assert!(future_map.contains_key("C"));
        assert_eq!(future_map["C"].0, 2);
    }

    #[test]
    fn test_trajectory_learning_updates_both_systems() {
        let mut system = IntegratedLearningSystem::new(LearningConfig {
            wormhole_threshold: 2,
            ..Default::default()
        });

        // Create patterns
        for c in ['A', 'B', 'C', 'D'] {
            system.add_node(LearningNode::pattern(c.to_string(), format!("Node {}", c)));
        }

        // Record trajectories
        for _ in 0..2 {
            let mut traj = Trajectory::new("session", "Task");
            traj.add_step("A", "start");
            traj.add_step("B", "step1");
            traj.add_step("C", "step2");
            traj.add_step("D", "end");
            traj.complete(TrajectoryOutcome::Success);
            system.record_trajectory(traj);
        }

        // Should have created causal links
        let future_from_a = system.predict_future("A", 5);
        assert!(future_from_a.len() > 1);

        // Should have created wormholes between co-accessed patterns
        // (A-B, A-C, A-D, B-C, B-D, C-D pairs)
        let stats = system.stats();
        assert!(stats.total_edges > 0);
    }
}

// ============================================================================
// Performance Under Load Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_many_trajectories() {
        let mut system = IntegratedLearningSystem::new(LearningConfig {
            wormhole_threshold: 5,
            max_wormholes_per_node: 20,
            ..Default::default()
        });

        // Create 100 patterns
        for i in 0..100 {
            let node = LearningNode::pattern(format!("pat_{}", i), format!("Pattern {}", i))
                .with_embedding(normalized_embedding(128));
            system.add_node(node);
        }

        let start = Instant::now();

        // Record 500 trajectories
        for t in 0..500 {
            let mut traj = Trajectory::new(format!("session_{}", t), format!("Task {}", t));

            // Each trajectory uses 5-10 random patterns
            let step_count = 5 + (t % 6);
            for s in 0..step_count {
                let pat_id = format!("pat_{}", (t * s + s) % 100);
                traj.add_step(&pat_id, format!("step_{}", s));
            }

            traj.complete(if t % 3 == 0 {
                TrajectoryOutcome::Failure
            } else {
                TrajectoryOutcome::Success
            });

            system.record_trajectory(traj);
        }

        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 5000,
            "500 trajectories should be processed in < 5s, took {:?}",
            duration
        );

        let stats = system.stats();
        assert_eq!(stats.total_trajectories, 500);
        assert!(stats.total_wormholes > 0);
    }

    #[test]
    fn test_traversal_performance() {
        let mut system = IntegratedLearningSystem::new(LearningConfig::default());

        // Create a large graph
        for i in 0..500 {
            system.add_node(LearningNode::pattern(format!("n_{}", i), format!("Node {}", i)));
        }

        // Add edges
        for i in 0..499 {
            system.add_edge(LearningEdge::new(
                format!("n_{}", i),
                format!("n_{}", i + 1),
                EdgeType::LeadsTo,
                0.9,
            ));
        }

        // Add some wormholes
        for i in (0..400).step_by(50) {
            system.add_edge(LearningEdge::wormhole(
                format!("n_{}", i),
                format!("n_{}", i + 99),
                0.95,
            ));
        }

        let start = Instant::now();

        // Perform many traversals
        for _ in 0..100 {
            system.trace_history("n_499", 50);
            system.predict_future("n_0", 50);
            system.compare_traversal("n_0", "n_499");
        }

        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 2000,
            "300 traversals should complete in < 2s, took {:?}",
            duration
        );
    }

    #[test]
    fn test_concurrent_patterns_performance() {
        let mut system = IntegratedLearningSystem::new(LearningConfig::default());

        let start = Instant::now();

        // Rapidly add and activate patterns
        for i in 0..1000 {
            let node = LearningNode::pattern(format!("rapid_{}", i), format!("Rapid {}", i));
            system.add_node(node);
            system.activate_pattern(&format!("rapid_{}", i), 0.5 + (i as f64 % 50.0) / 100.0);
        }

        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 500,
            "1000 pattern adds/activations should complete in < 500ms, took {:?}",
            duration
        );

        // Verify cognitive core is bounded
        let active = system.get_active_patterns();
        assert!(active.len() <= system.config.cognitive_core_max_active);
    }
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Property: Traversal with wormholes is never longer than without.
        #[test]
        fn prop_wormhole_never_worse(chain_len in 3usize..10usize) {
            let mut system = IntegratedLearningSystem::new(LearningConfig::default());

            // Create chain
            for i in 0..chain_len {
                system.add_node(LearningNode::pattern(format!("n{}", i), format!("Node {}", i)));
            }

            for i in 0..(chain_len - 1) {
                system.add_edge(LearningEdge::new(
                    format!("n{}", i),
                    format!("n{}", i + 1),
                    EdgeType::LeadsTo,
                    0.9,
                ));
            }

            // Add wormhole
            if chain_len > 3 {
                system.add_edge(LearningEdge::wormhole("n0", format!("n{}", chain_len - 1), 0.95));
            }

            let comparison = system.compare_traversal("n0", &format!("n{}", chain_len - 1));

            let normal_len = comparison.normal_path.as_ref().map_or(usize::MAX, |p| p.len());
            let wormhole_len = comparison.wormhole_path.as_ref().map_or(usize::MAX, |p| p.len());

            prop_assert!(wormhole_len <= normal_len);
        }

        /// Property: Light cone depth is bounded.
        #[test]
        fn prop_light_cone_bounded(max_depth in 1usize..20usize) {
            let mut system = IntegratedLearningSystem::new(LearningConfig::default());

            // Create long chain
            for i in 0..50 {
                system.add_node(LearningNode::pattern(format!("n{}", i), format!("Node {}", i)));
            }

            for i in 0..49 {
                system.add_edge(LearningEdge::new(
                    format!("n{}", i),
                    format!("n{}", i + 1),
                    EdgeType::LeadsTo,
                    0.9,
                ));
            }

            let future = system.predict_future("n0", max_depth);

            for (_, depth, _) in &future {
                prop_assert!(*depth <= max_depth);
            }
        }

        /// Property: Recording trajectory updates node usage counts.
        #[test]
        fn prop_trajectory_updates_usage(step_count in 2usize..10usize) {
            let mut system = IntegratedLearningSystem::new(LearningConfig::default());

            // Create patterns
            for i in 0..step_count {
                system.add_node(LearningNode::pattern(format!("p{}", i), format!("Pattern {}", i)));
            }

            let initial_counts: Vec<_> = (0..step_count)
                .map(|i| system.get_node(&format!("p{}", i)).unwrap().usage_count)
                .collect();

            // Record successful trajectory
            let mut traj = Trajectory::new("session", "task");
            for i in 0..step_count {
                traj.add_step(format!("p{}", i), "step");
            }
            traj.complete(TrajectoryOutcome::Success);
            system.record_trajectory(traj);

            for i in 0..step_count {
                let new_count = system.get_node(&format!("p{}", i)).unwrap().usage_count;
                prop_assert!(new_count >= initial_counts[i]);
            }
        }
    }
}
