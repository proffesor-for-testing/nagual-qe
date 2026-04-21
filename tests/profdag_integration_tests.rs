//! ProfDAG Integration Tests - Phase 1
//!
//! End-to-end integration tests for ProfDAG combining schema, search,
//! trajectory recording, graph integration, and learning integration.
//!
//! # Test Scenarios
//! - Full flow: create node -> search -> record trajectory
//! - Graph-based knowledge retrieval
//! - Learning from trajectory outcomes
//! - Cross-component integration

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

mod common;
use common::{
    cosine_similarity, normalized_embedding, similar_embeddings,
};

// ============================================================================
// Integrated Types (combining schema, search, trajectory)
// ============================================================================

/// Node types from schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    Pattern,
    Trajectory,
    Prediction,
    Decision,
}

/// Edge types from schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    LeadsTo,
    SimilarTo,
    DerivedFrom,
    Wormhole,
    TemporalLink,
}

/// Outcome types from trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Success,
    PartialSuccess,
    Failure,
    Timeout,
}

impl Outcome {
    pub fn reward(&self) -> f32 {
        match self {
            Outcome::Success => 1.0,
            Outcome::PartialSuccess => 0.6,
            Outcome::Failure => 0.1,
            Outcome::Timeout => 0.2,
        }
    }
}

/// A node in the ProfDAG system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub node_type: NodeType,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub usage_count: u32,
    pub success_rate: f32,
}

impl Node {
    pub fn new(node_type: NodeType, content: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            node_type,
            content: content.into(),
            embedding: None,
            metadata: serde_json::json!({}),
            created_at: now,
            updated_at: now,
            usage_count: 0,
            success_rate: 0.5,
        }
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn record_usage(&mut self, success: bool) {
        self.usage_count += 1;
        let success_count = (self.success_rate * (self.usage_count - 1) as f32) + if success { 1.0 } else { 0.0 };
        self.success_rate = success_count / self.usage_count as f32;
        self.updated_at = Utc::now();
    }
}

/// An edge in the ProfDAG system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub edge_type: EdgeType,
    pub weight: f64,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

impl Edge {
    pub fn new(source: impl Into<String>, target: impl Into<String>, edge_type: EdgeType, weight: f64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_id: source.into(),
            target_id: target.into(),
            edge_type,
            weight: weight.clamp(0.0, 1.0),
            metadata: serde_json::json!({}),
            created_at: Utc::now(),
        }
    }
}

/// A trajectory step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryStep {
    pub step_number: usize,
    pub action: String,
    pub node_ids_used: Vec<String>,
    pub input: String,
    pub output: Option<String>,
    pub duration_ms: u64,
}

/// A complete trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub id: String,
    pub session_id: String,
    pub task: String,
    pub steps: Vec<TrajectoryStep>,
    pub outcome: Option<Outcome>,
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

    pub fn add_step(&mut self, action: impl Into<String>, input: impl Into<String>, node_ids: Vec<String>) {
        let step = TrajectoryStep {
            step_number: self.steps.len() + 1,
            action: action.into(),
            node_ids_used: node_ids,
            input: input.into(),
            output: None,
            duration_ms: 0,
        };
        self.steps.push(step);
    }

    pub fn complete(&mut self, outcome: Outcome) {
        self.outcome = Some(outcome);
        self.reward = Some(outcome.reward());
        self.completed_at = Some(Utc::now());
    }

    pub fn all_used_nodes(&self) -> Vec<String> {
        self.steps.iter()
            .flat_map(|s| s.node_ids_used.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }
}

/// Search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub node_id: String,
    pub similarity: f32,
}

/// Integrated ProfDAG system.
#[derive(Debug)]
pub struct ProfDagSystem {
    nodes: HashMap<String, Node>,
    edges: HashMap<String, Edge>,
    trajectories: HashMap<String, Trajectory>,
    outgoing_edges: HashMap<String, Vec<String>>,
    incoming_edges: HashMap<String, Vec<String>>,
}

impl ProfDagSystem {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            trajectories: HashMap::new(),
            outgoing_edges: HashMap::new(),
            incoming_edges: HashMap::new(),
        }
    }

    // ========== Node Operations ==========

    pub fn create_node(&mut self, node: Node) -> String {
        let id = node.id.clone();
        self.outgoing_edges.insert(id.clone(), Vec::new());
        self.incoming_edges.insert(id.clone(), Vec::new());
        self.nodes.insert(id.clone(), node);
        id
    }

    pub fn get_node(&self, id: &str) -> Option<&Node> {
        self.nodes.get(id)
    }

    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut Node> {
        self.nodes.get_mut(id)
    }

    pub fn update_node_usage(&mut self, id: &str, success: bool) -> Result<(), String> {
        let node = self.nodes.get_mut(id).ok_or("Node not found")?;
        node.record_usage(success);
        Ok(())
    }

    // ========== Edge Operations ==========

    pub fn create_edge(&mut self, edge: Edge) -> Result<String, String> {
        if !self.nodes.contains_key(&edge.source_id) {
            return Err(format!("Source node not found: {}", edge.source_id));
        }
        if !self.nodes.contains_key(&edge.target_id) {
            return Err(format!("Target node not found: {}", edge.target_id));
        }

        let id = edge.id.clone();

        self.outgoing_edges
            .entry(edge.source_id.clone())
            .or_insert_with(Vec::new)
            .push(id.clone());

        self.incoming_edges
            .entry(edge.target_id.clone())
            .or_insert_with(Vec::new)
            .push(id.clone());

        self.edges.insert(id.clone(), edge);
        Ok(id)
    }

    pub fn get_neighbors(&self, node_id: &str, edge_type: Option<EdgeType>) -> Vec<&Node> {
        let edge_ids = self.outgoing_edges.get(node_id).map(|v| v.as_slice()).unwrap_or(&[]);

        edge_ids.iter()
            .filter_map(|edge_id| self.edges.get(edge_id))
            .filter(|edge| edge_type.map_or(true, |t| edge.edge_type == t))
            .filter_map(|edge| self.nodes.get(&edge.target_id))
            .collect()
    }

    pub fn get_related_nodes(&self, node_id: &str, max_depth: usize) -> Vec<(&Node, usize)> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue = vec![(node_id.to_string(), 0usize)];

        while let Some((current_id, depth)) = queue.pop() {
            if depth > max_depth || visited.contains(&current_id) {
                continue;
            }
            visited.insert(current_id.clone());

            if let Some(node) = self.nodes.get(&current_id) {
                if current_id != node_id {
                    result.push((node, depth));
                }

                if depth < max_depth {
                    for neighbor in self.get_neighbors(&current_id, None) {
                        if !visited.contains(&neighbor.id) {
                            queue.push((neighbor.id.clone(), depth + 1));
                        }
                    }
                }
            }
        }

        result
    }

    // ========== Search Operations ==========

    pub fn search(&self, query_embedding: &[f32], k: usize, node_type: Option<NodeType>) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = self.nodes
            .values()
            .filter(|node| {
                node.embedding.is_some() &&
                node_type.map_or(true, |t| node.node_type == t)
            })
            .map(|node| {
                let similarity = cosine_similarity(query_embedding, node.embedding.as_ref().unwrap());
                SearchResult {
                    node_id: node.id.clone(),
                    similarity,
                }
            })
            .collect();

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(k);
        results
    }

    pub fn search_with_graph_context(
        &self,
        query_embedding: &[f32],
        k: usize,
        graph_depth: usize,
    ) -> Vec<(SearchResult, Vec<String>)> {
        let initial_results = self.search(query_embedding, k, None);

        initial_results.into_iter().map(|result| {
            let related_ids: Vec<String> = self.get_related_nodes(&result.node_id, graph_depth)
                .into_iter()
                .map(|(node, _)| node.id.clone())
                .collect();
            (result, related_ids)
        }).collect()
    }

    // ========== Trajectory Operations ==========

    pub fn start_trajectory(&mut self, session_id: impl Into<String>, task: impl Into<String>) -> String {
        let trajectory = Trajectory::new(session_id, task);
        let id = trajectory.id.clone();
        self.trajectories.insert(id.clone(), trajectory);
        id
    }

    pub fn add_trajectory_step(
        &mut self,
        trajectory_id: &str,
        action: impl Into<String>,
        input: impl Into<String>,
        node_ids: Vec<String>,
    ) -> Result<(), String> {
        let trajectory = self.trajectories.get_mut(trajectory_id)
            .ok_or("Trajectory not found")?;
        trajectory.add_step(action, input, node_ids);
        Ok(())
    }

    pub fn complete_trajectory(&mut self, trajectory_id: &str, outcome: Outcome) -> Result<&Trajectory, String> {
        let trajectory = self.trajectories.get_mut(trajectory_id)
            .ok_or("Trajectory not found")?;
        trajectory.complete(outcome);

        // Update node usage statistics
        let used_nodes = trajectory.all_used_nodes();
        let success = outcome == Outcome::Success || outcome == Outcome::PartialSuccess;

        for node_id in &used_nodes {
            if let Some(node) = self.nodes.get_mut(node_id) {
                node.record_usage(success);
            }
        }

        Ok(self.trajectories.get(trajectory_id).unwrap())
    }

    pub fn get_trajectory(&self, id: &str) -> Option<&Trajectory> {
        self.trajectories.get(id)
    }

    // ========== Learning Operations ==========

    pub fn apply_learning(&mut self, trajectory_id: &str) -> Result<LearningResult, String> {
        // Extract data from trajectory first to avoid borrow issues
        let (outcome, reward, used_nodes, steps_data, task) = {
            let trajectory = self.trajectories.get(trajectory_id)
                .ok_or("Trajectory not found")?;

            if trajectory.outcome.is_none() {
                return Err("Trajectory not complete".to_string());
            }

            let outcome = trajectory.outcome.unwrap();
            let reward = trajectory.reward.unwrap_or(0.0);
            let used_nodes = trajectory.all_used_nodes();
            let steps_data: Vec<Vec<String>> = trajectory.steps.iter()
                .map(|s| s.node_ids_used.clone())
                .collect();
            let task = trajectory.task.clone();

            (outcome, reward, used_nodes, steps_data, task)
        };

        let mut updated_nodes = Vec::new();
        let mut created_edges = Vec::new();

        // Create edges between consecutively used patterns
        let mut prev_node_id: Option<String> = None;
        for step_nodes in &steps_data {
            for node_id in step_nodes {
                if let Some(ref prev) = prev_node_id {
                    if prev != node_id {
                        // Create leads_to edge
                        let weight = if outcome == Outcome::Success { 0.9 } else { 0.5 };
                        let edge = Edge::new(prev, node_id, EdgeType::LeadsTo, weight);
                        if let Ok(edge_id) = self.create_edge(edge) {
                            created_edges.push(edge_id);
                        }
                    }
                }
                prev_node_id = Some(node_id.clone());
                updated_nodes.push(node_id.clone());
            }
        }

        // Create trajectory node
        let traj_content = format!("Trajectory: {} ({})", task, outcome_str(outcome));
        let traj_node = Node::new(NodeType::Trajectory, traj_content);
        let traj_node_id = self.create_node(traj_node);

        // Link trajectory to used patterns
        for node_id in &used_nodes {
            let edge = Edge::new(&traj_node_id, node_id, EdgeType::DerivedFrom, reward as f64);
            if let Ok(edge_id) = self.create_edge(edge) {
                created_edges.push(edge_id);
            }
        }

        Ok(LearningResult {
            trajectory_id: trajectory_id.to_string(),
            outcome,
            reward,
            updated_nodes,
            created_edges,
            trajectory_node_id: traj_node_id,
        })
    }

    // ========== Statistics ==========

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn trajectory_count(&self) -> usize {
        self.trajectories.len()
    }

    pub fn successful_trajectory_count(&self) -> usize {
        self.trajectories.values()
            .filter(|t| t.outcome == Some(Outcome::Success) || t.outcome == Some(Outcome::PartialSuccess))
            .count()
    }
}

fn outcome_str(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Success => "success",
        Outcome::PartialSuccess => "partial",
        Outcome::Failure => "failure",
        Outcome::Timeout => "timeout",
    }
}

/// Result of applying learning from a trajectory.
#[derive(Debug)]
pub struct LearningResult {
    pub trajectory_id: String,
    pub outcome: Outcome,
    pub reward: f32,
    pub updated_nodes: Vec<String>,
    pub created_edges: Vec<String>,
    pub trajectory_node_id: String,
}

impl Default for ProfDagSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// End-to-End Flow Tests
// ============================================================================

mod e2e_flow_tests {
    use super::*;

    #[test]
    fn test_create_node_search_record_trajectory() {
        let mut system = ProfDagSystem::new();

        // Step 1: Create pattern nodes with embeddings
        let base_embedding = normalized_embedding(128);
        let node1 = Node::new(NodeType::Pattern, "How to handle database timeouts")
            .with_embedding(similar_embeddings(&base_embedding, 1, 0.1)[0].clone());
        let node1_id = system.create_node(node1);

        let node2 = Node::new(NodeType::Pattern, "Implement retry with exponential backoff")
            .with_embedding(similar_embeddings(&base_embedding, 1, 0.15)[0].clone());
        let node2_id = system.create_node(node2);

        // Step 2: Search for similar patterns
        let query = similar_embeddings(&base_embedding, 1, 0.05)[0].clone();
        let results = system.search(&query, 5, Some(NodeType::Pattern));

        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.node_id == node1_id || r.node_id == node2_id));

        // Step 3: Start trajectory using found patterns
        let traj_id = system.start_trajectory("session-1", "Fix database timeout issue");

        system.add_trajectory_step(
            &traj_id,
            "query",
            "Search for timeout solutions",
            vec![node1_id.clone()],
        ).unwrap();

        system.add_trajectory_step(
            &traj_id,
            "apply",
            "Apply retry pattern",
            vec![node2_id.clone()],
        ).unwrap();

        // Step 4: Complete trajectory
        let trajectory = system.complete_trajectory(&traj_id, Outcome::Success).unwrap();

        assert!(trajectory.outcome.is_some());
        assert_eq!(trajectory.outcome, Some(Outcome::Success));
        assert_eq!(trajectory.steps.len(), 2);

        // Verify nodes were updated
        let node1 = system.get_node(&node1_id).unwrap();
        assert!(node1.usage_count > 0);
        assert!(node1.success_rate > 0.0);
    }

    #[test]
    fn test_full_learning_cycle() {
        let mut system = ProfDagSystem::new();

        // Create initial patterns
        let base_embedding = normalized_embedding(128);
        let patterns: Vec<String> = (0..5).map(|i| {
            let node = Node::new(NodeType::Pattern, format!("Pattern {} solution", i))
                .with_embedding(similar_embeddings(&base_embedding, 1, 0.1 + i as f32 * 0.05)[0].clone());
            system.create_node(node)
        }).collect();

        // Run multiple trajectories
        for i in 0..3 {
            let traj_id = system.start_trajectory(format!("session-{}", i), format!("Task {}", i));

            // Use some patterns
            for j in 0..2 {
                system.add_trajectory_step(
                    &traj_id,
                    "use_pattern",
                    format!("Using pattern {}", j),
                    vec![patterns[j + i].clone()],
                ).unwrap();
            }

            let outcome = if i == 2 { Outcome::Failure } else { Outcome::Success };
            system.complete_trajectory(&traj_id, outcome).unwrap();

            // Apply learning
            let learning = system.apply_learning(&traj_id).unwrap();
            assert!(!learning.updated_nodes.is_empty());
            assert!(!learning.created_edges.is_empty());
        }

        // Verify graph was enriched
        assert!(system.edge_count() > 0);
        assert!(system.node_count() > 5); // Should have trajectory nodes too

        // Verify successful trajectories count
        assert_eq!(system.successful_trajectory_count(), 2);
    }

    #[test]
    fn test_search_to_trajectory_to_learning() {
        let mut system = ProfDagSystem::new();

        // Setup: Create a rich pattern database
        let embeddings: Vec<Vec<f32>> = (0..10)
            .map(|_| normalized_embedding(128))
            .collect();

        for (i, emb) in embeddings.iter().enumerate() {
            let node = Node::new(NodeType::Pattern, format!("Pattern content {}", i))
                .with_embedding(emb.clone())
                .with_metadata(serde_json::json!({"domain": format!("domain-{}", i % 3)}));
            system.create_node(node);
        }

        // Search phase
        let query = embeddings[0].clone();
        let search_results = system.search(&query, 3, Some(NodeType::Pattern));
        assert_eq!(search_results.len(), 3);

        // Record trajectory using search results
        let traj_id = system.start_trajectory("search-session", "Apply search results");

        for result in &search_results {
            system.add_trajectory_step(
                &traj_id,
                "apply_result",
                format!("Applying pattern with similarity {:.2}", result.similarity),
                vec![result.node_id.clone()],
            ).unwrap();
        }

        system.complete_trajectory(&traj_id, Outcome::Success).unwrap();

        // Learning phase
        let learning = system.apply_learning(&traj_id).unwrap();

        assert_eq!(learning.outcome, Outcome::Success);
        assert_eq!(learning.updated_nodes.len(), 3);
        assert!(!learning.created_edges.is_empty());

        // Verify trajectory node was created
        let traj_node = system.get_node(&learning.trajectory_node_id).unwrap();
        assert_eq!(traj_node.node_type, NodeType::Trajectory);
    }
}

// ============================================================================
// Graph Integration Tests
// ============================================================================

mod graph_integration_tests {
    use super::*;

    #[test]
    fn test_edge_creation_on_learning() {
        let mut system = ProfDagSystem::new();

        // Create patterns
        let node1 = Node::new(NodeType::Pattern, "Pattern A").with_embedding(normalized_embedding(128));
        let node2 = Node::new(NodeType::Pattern, "Pattern B").with_embedding(normalized_embedding(128));
        let node3 = Node::new(NodeType::Pattern, "Pattern C").with_embedding(normalized_embedding(128));

        let id1 = system.create_node(node1);
        let id2 = system.create_node(node2);
        let id3 = system.create_node(node3);

        // Create trajectory that uses patterns in sequence
        let traj_id = system.start_trajectory("session", "Sequential task");
        system.add_trajectory_step(&traj_id, "step1", "Using A", vec![id1.clone()]).unwrap();
        system.add_trajectory_step(&traj_id, "step2", "Using B", vec![id2.clone()]).unwrap();
        system.add_trajectory_step(&traj_id, "step3", "Using C", vec![id3.clone()]).unwrap();

        system.complete_trajectory(&traj_id, Outcome::Success).unwrap();

        let initial_edge_count = system.edge_count();

        // Apply learning
        system.apply_learning(&traj_id).unwrap();

        // Edges should be created: A->B, B->C, plus trajectory links
        assert!(system.edge_count() > initial_edge_count);

        // Check A leads to B
        let a_neighbors = system.get_neighbors(&id1, Some(EdgeType::LeadsTo));
        assert!(a_neighbors.iter().any(|n| n.id == id2));
    }

    #[test]
    fn test_graph_traversal_after_learning() {
        let mut system = ProfDagSystem::new();

        // Build a chain of patterns
        let mut ids = Vec::new();
        for i in 0..5 {
            let node = Node::new(NodeType::Pattern, format!("Chain pattern {}", i))
                .with_embedding(normalized_embedding(128));
            ids.push(system.create_node(node));
        }

        // Create edges forming a chain
        for i in 0..4 {
            let edge = Edge::new(&ids[i], &ids[i + 1], EdgeType::LeadsTo, 0.9);
            system.create_edge(edge).unwrap();
        }

        // Traverse from first node
        let related = system.get_related_nodes(&ids[0], 3);

        // Should find nodes at various depths
        assert!(!related.is_empty());

        // Check depth is correct
        for (node, depth) in &related {
            assert!(*depth <= 3);
            assert!(ids.contains(&node.id));
        }
    }

    #[test]
    fn test_search_with_graph_context() {
        let mut system = ProfDagSystem::new();

        // Create interconnected nodes
        let base_emb = normalized_embedding(128);

        let main_node = Node::new(NodeType::Pattern, "Main pattern")
            .with_embedding(base_emb.clone());
        let main_id = system.create_node(main_node);

        let related_nodes: Vec<String> = (0..3).map(|i| {
            let node = Node::new(NodeType::Pattern, format!("Related pattern {}", i))
                .with_embedding(similar_embeddings(&base_emb, 1, 0.2)[0].clone());
            system.create_node(node)
        }).collect();

        // Link main to related nodes
        for related_id in &related_nodes {
            let edge = Edge::new(&main_id, related_id, EdgeType::SimilarTo, 0.8);
            system.create_edge(edge).unwrap();
        }

        // Search with graph context
        let results = system.search_with_graph_context(&base_emb, 5, 1);

        // Main node should be found with its related nodes
        let main_result = results.iter().find(|(r, _)| r.node_id == main_id);
        assert!(main_result.is_some());

        let (_, related_ids) = main_result.unwrap();
        assert!(!related_ids.is_empty());
    }

    #[test]
    fn test_bidirectional_edges() {
        let mut system = ProfDagSystem::new();

        let node1 = Node::new(NodeType::Pattern, "Node 1").with_embedding(normalized_embedding(128));
        let node2 = Node::new(NodeType::Pattern, "Node 2").with_embedding(normalized_embedding(128));

        let id1 = system.create_node(node1);
        let id2 = system.create_node(node2);

        // Create bidirectional similar_to edges
        system.create_edge(Edge::new(&id1, &id2, EdgeType::SimilarTo, 0.9)).unwrap();
        system.create_edge(Edge::new(&id2, &id1, EdgeType::SimilarTo, 0.9)).unwrap();

        // Both should be neighbors of each other
        let neighbors1 = system.get_neighbors(&id1, Some(EdgeType::SimilarTo));
        let neighbors2 = system.get_neighbors(&id2, Some(EdgeType::SimilarTo));

        assert!(neighbors1.iter().any(|n| n.id == id2));
        assert!(neighbors2.iter().any(|n| n.id == id1));
    }
}

// ============================================================================
// Learning Integration Tests
// ============================================================================

mod learning_integration_tests {
    use super::*;

    #[test]
    fn test_node_usage_tracking() {
        let mut system = ProfDagSystem::new();

        let node = Node::new(NodeType::Pattern, "Tracked pattern")
            .with_embedding(normalized_embedding(128));
        let id = system.create_node(node);

        // Initial state
        let initial = system.get_node(&id).unwrap();
        assert_eq!(initial.usage_count, 0);

        // Record successful usage
        system.update_node_usage(&id, true).unwrap();
        let after_success = system.get_node(&id).unwrap();
        assert_eq!(after_success.usage_count, 1);
        assert_eq!(after_success.success_rate, 1.0);

        // Record failed usage
        system.update_node_usage(&id, false).unwrap();
        let after_failure = system.get_node(&id).unwrap();
        assert_eq!(after_failure.usage_count, 2);
        assert!((after_failure.success_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_learning_updates_success_rates() {
        let mut system = ProfDagSystem::new();

        let node = Node::new(NodeType::Pattern, "Learning pattern")
            .with_embedding(normalized_embedding(128));
        let id = system.create_node(node);

        // Successful trajectory
        let traj_id = system.start_trajectory("s1", "Success task");
        system.add_trajectory_step(&traj_id, "use", "Using pattern", vec![id.clone()]).unwrap();
        system.complete_trajectory(&traj_id, Outcome::Success).unwrap();

        let node_after = system.get_node(&id).unwrap();
        assert_eq!(node_after.success_rate, 1.0);

        // Failed trajectory
        let traj_id2 = system.start_trajectory("s2", "Failure task");
        system.add_trajectory_step(&traj_id2, "use", "Using pattern", vec![id.clone()]).unwrap();
        system.complete_trajectory(&traj_id2, Outcome::Failure).unwrap();

        let node_after2 = system.get_node(&id).unwrap();
        assert!((node_after2.success_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_learning_creates_trajectory_nodes() {
        let mut system = ProfDagSystem::new();

        let pattern = Node::new(NodeType::Pattern, "Source pattern")
            .with_embedding(normalized_embedding(128));
        let pattern_id = system.create_node(pattern);

        let initial_node_count = system.node_count();

        // Create and complete trajectory
        let traj_id = system.start_trajectory("session", "Task");
        system.add_trajectory_step(&traj_id, "use", "Using", vec![pattern_id.clone()]).unwrap();
        system.complete_trajectory(&traj_id, Outcome::Success).unwrap();

        // Apply learning
        let learning = system.apply_learning(&traj_id).unwrap();

        // New trajectory node should be created
        assert!(system.node_count() > initial_node_count);

        let traj_node = system.get_node(&learning.trajectory_node_id).unwrap();
        assert_eq!(traj_node.node_type, NodeType::Trajectory);
    }

    #[test]
    fn test_learning_incomplete_trajectory_fails() {
        let mut system = ProfDagSystem::new();

        let pattern = Node::new(NodeType::Pattern, "Pattern").with_embedding(normalized_embedding(128));
        let pattern_id = system.create_node(pattern);

        let traj_id = system.start_trajectory("session", "Incomplete task");
        system.add_trajectory_step(&traj_id, "step", "Input", vec![pattern_id]).unwrap();

        // Don't complete the trajectory
        let result = system.apply_learning(&traj_id);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not complete"));
    }

    #[test]
    fn test_reward_propagation() {
        let mut system = ProfDagSystem::new();

        // Create patterns with different initial states
        let patterns: Vec<String> = (0..3).map(|i| {
            let node = Node::new(NodeType::Pattern, format!("Pattern {}", i))
                .with_embedding(normalized_embedding(128));
            system.create_node(node)
        }).collect();

        // Successful trajectory using all patterns
        let traj_id = system.start_trajectory("session", "Full success");
        for (i, pattern_id) in patterns.iter().enumerate() {
            system.add_trajectory_step(&traj_id, format!("step-{}", i), "Input", vec![pattern_id.clone()]).unwrap();
        }
        system.complete_trajectory(&traj_id, Outcome::Success).unwrap();

        // Apply learning
        let learning = system.apply_learning(&traj_id).unwrap();

        // All patterns should be updated
        assert_eq!(learning.updated_nodes.len(), 3);
        assert_eq!(learning.reward, 1.0);

        // Verify all nodes have updated success rates
        for pattern_id in &patterns {
            let node = system.get_node(pattern_id).unwrap();
            assert_eq!(node.success_rate, 1.0);
        }
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_large_graph_search_performance() {
        let mut system = ProfDagSystem::new();
        let base_emb = normalized_embedding(128);

        // Create 1000 nodes
        for i in 0..1000 {
            let emb = similar_embeddings(&base_emb, 1, 0.3)[0].clone();
            let node = Node::new(NodeType::Pattern, format!("Pattern {}", i))
                .with_embedding(emb);
            system.create_node(node);
        }

        // Measure search time
        let start = Instant::now();
        for _ in 0..100 {
            let query = similar_embeddings(&base_emb, 1, 0.1)[0].clone();
            system.search(&query, 10, Some(NodeType::Pattern));
        }
        let duration = start.elapsed();

        let avg_ms = duration.as_millis() as f64 / 100.0;
        assert!(
            avg_ms < 50.0,
            "Average search time {} ms exceeds 50ms limit",
            avg_ms
        );
    }

    #[test]
    fn test_trajectory_recording_performance() {
        let mut system = ProfDagSystem::new();

        // Create patterns
        let patterns: Vec<String> = (0..100).map(|i| {
            let node = Node::new(NodeType::Pattern, format!("Pattern {}", i))
                .with_embedding(normalized_embedding(128));
            system.create_node(node)
        }).collect();

        let start = Instant::now();

        // Record 50 trajectories, each with 10 steps
        for i in 0..50 {
            let traj_id = system.start_trajectory(format!("session-{}", i), format!("Task {}", i));

            for j in 0..10 {
                system.add_trajectory_step(
                    &traj_id,
                    format!("step-{}", j),
                    format!("Input {}", j),
                    vec![patterns[(i * 10 + j) % 100].clone()],
                ).unwrap();
            }

            let outcome = if i % 3 == 0 { Outcome::Failure } else { Outcome::Success };
            system.complete_trajectory(&traj_id, outcome).unwrap();
        }

        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 1000,
            "Recording 50 trajectories took {:?}, expected < 1s",
            duration
        );

        assert_eq!(system.trajectory_count(), 50);
    }

    #[test]
    fn test_learning_application_performance() {
        let mut system = ProfDagSystem::new();

        // Create patterns
        let patterns: Vec<String> = (0..50).map(|i| {
            let node = Node::new(NodeType::Pattern, format!("Pattern {}", i))
                .with_embedding(normalized_embedding(128));
            system.create_node(node)
        }).collect();

        // Create trajectories
        let traj_ids: Vec<String> = (0..20).map(|i| {
            let traj_id = system.start_trajectory(format!("s-{}", i), format!("Task {}", i));

            for j in 0..5 {
                system.add_trajectory_step(
                    &traj_id,
                    "step",
                    "Input",
                    vec![patterns[(i * 5 + j) % 50].clone()],
                ).unwrap();
            }

            system.complete_trajectory(&traj_id, Outcome::Success).unwrap();
            traj_id
        }).collect();

        // Measure learning application time
        let start = Instant::now();
        for traj_id in &traj_ids {
            system.apply_learning(traj_id).unwrap();
        }
        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 500,
            "Applying learning to 20 trajectories took {:?}, expected < 500ms",
            duration
        );
    }

    #[test]
    fn test_graph_traversal_performance() {
        let mut system = ProfDagSystem::new();

        // Create a graph with 100 nodes and many edges
        let nodes: Vec<String> = (0..100).map(|i| {
            let node = Node::new(NodeType::Pattern, format!("Node {}", i))
                .with_embedding(normalized_embedding(128));
            system.create_node(node)
        }).collect();

        // Create edges (sparse graph)
        for i in 0..100 {
            for j in 0..3 {
                let target = (i + j + 1) % 100;
                if i != target {
                    let edge = Edge::new(&nodes[i], &nodes[target], EdgeType::LeadsTo, 0.8);
                    let _ = system.create_edge(edge);
                }
            }
        }

        // Measure traversal time
        let start = Instant::now();
        for node in &nodes {
            system.get_related_nodes(node, 3);
        }
        let duration = start.elapsed();

        let avg_us = duration.as_micros() as f64 / 100.0;
        assert!(
            avg_us < 1000.0,
            "Average traversal time {} us exceeds 1ms limit",
            avg_us
        );
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_trajectory() {
        let mut system = ProfDagSystem::new();

        let traj_id = system.start_trajectory("session", "Empty task");
        system.complete_trajectory(&traj_id, Outcome::Success).unwrap();

        let learning = system.apply_learning(&traj_id).unwrap();
        assert!(learning.updated_nodes.is_empty());
    }

    #[test]
    fn test_search_empty_system() {
        let system = ProfDagSystem::new();
        let results = system.search(&normalized_embedding(128), 10, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_nonexistent_trajectory() {
        let mut system = ProfDagSystem::new();
        let result = system.complete_trajectory("nonexistent", Outcome::Success);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_node_ids_in_step() {
        let mut system = ProfDagSystem::new();

        let node = Node::new(NodeType::Pattern, "Pattern").with_embedding(normalized_embedding(128));
        let id = system.create_node(node);

        let traj_id = system.start_trajectory("session", "Duplicate test");
        system.add_trajectory_step(
            &traj_id,
            "step",
            "Input",
            vec![id.clone(), id.clone(), id.clone()],
        ).unwrap();

        system.complete_trajectory(&traj_id, Outcome::Success).unwrap();

        // Node should only be counted once for usage
        let node = system.get_node(&id).unwrap();
        assert_eq!(node.usage_count, 1);
    }

    #[test]
    fn test_self_referential_edge() {
        let mut system = ProfDagSystem::new();

        let node = Node::new(NodeType::Pattern, "Self ref").with_embedding(normalized_embedding(128));
        let id = system.create_node(node);

        // Wormhole edges can be self-referential
        let edge = Edge::new(&id, &id, EdgeType::Wormhole, 0.5);
        let result = system.create_edge(edge);
        assert!(result.is_ok());
    }

    #[test]
    fn test_circular_graph() {
        let mut system = ProfDagSystem::new();

        let nodes: Vec<String> = (0..5).map(|i| {
            let node = Node::new(NodeType::Pattern, format!("Circular {}", i))
                .with_embedding(normalized_embedding(128));
            system.create_node(node)
        }).collect();

        // Create circular edges: 0 -> 1 -> 2 -> 3 -> 4 -> 0
        for i in 0..5 {
            let edge = Edge::new(&nodes[i], &nodes[(i + 1) % 5], EdgeType::LeadsTo, 0.9);
            system.create_edge(edge).unwrap();
        }

        // Traversal should handle cycles
        let related = system.get_related_nodes(&nodes[0], 10);

        // Should find all other nodes without infinite loop
        assert!(related.len() <= 4);
    }

    #[test]
    fn test_very_long_trajectory() {
        let mut system = ProfDagSystem::new();

        let node = Node::new(NodeType::Pattern, "Pattern").with_embedding(normalized_embedding(128));
        let id = system.create_node(node);

        let traj_id = system.start_trajectory("session", "Long trajectory");

        for i in 0..1000 {
            system.add_trajectory_step(&traj_id, format!("step-{}", i), "Input", vec![id.clone()]).unwrap();
        }

        system.complete_trajectory(&traj_id, Outcome::Success).unwrap();

        let trajectory = system.get_trajectory(&traj_id).unwrap();
        assert_eq!(trajectory.steps.len(), 1000);
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
        /// Property: Search always returns at most k results.
        #[test]
        fn prop_search_returns_at_most_k(
            node_count in 1usize..50usize,
            k in 1usize..20usize,
        ) {
            let mut system = ProfDagSystem::new();

            for i in 0..node_count {
                let node = Node::new(NodeType::Pattern, format!("Node {}", i))
                    .with_embedding(normalized_embedding(64));
                system.create_node(node);
            }

            let results = system.search(&normalized_embedding(64), k, None);
            prop_assert!(results.len() <= k);
        }

        /// Property: Trajectory completion always sets outcome and reward.
        #[test]
        fn prop_completion_sets_outcome(outcome_idx in 0usize..4usize) {
            let outcomes = [Outcome::Success, Outcome::PartialSuccess, Outcome::Failure, Outcome::Timeout];
            let outcome = outcomes[outcome_idx];

            let mut system = ProfDagSystem::new();
            let traj_id = system.start_trajectory("session", "task");
            let trajectory = system.complete_trajectory(&traj_id, outcome).unwrap();

            prop_assert!(trajectory.outcome.is_some());
            prop_assert!(trajectory.reward.is_some());
            prop_assert!(trajectory.completed_at.is_some());
        }

        /// Property: Node usage count always increases after trajectory completion.
        #[test]
        fn prop_usage_count_increases(usage_count in 1usize..10usize) {
            let mut system = ProfDagSystem::new();

            let node = Node::new(NodeType::Pattern, "Pattern").with_embedding(normalized_embedding(64));
            let id = system.create_node(node);

            for i in 0..usage_count {
                let traj_id = system.start_trajectory(format!("s{}", i), "task");
                system.add_trajectory_step(&traj_id, "step", "input", vec![id.clone()]).unwrap();
                system.complete_trajectory(&traj_id, Outcome::Success).unwrap();
            }

            let node = system.get_node(&id).unwrap();
            prop_assert_eq!(node.usage_count as usize, usage_count);
        }

        /// Property: Search similarity is always in [-1, 1].
        #[test]
        fn prop_similarity_bounded(node_count in 1usize..20usize) {
            let mut system = ProfDagSystem::new();

            for i in 0..node_count {
                let node = Node::new(NodeType::Pattern, format!("Node {}", i))
                    .with_embedding(normalized_embedding(64));
                system.create_node(node);
            }

            let results = system.search(&normalized_embedding(64), node_count, None);

            for result in &results {
                prop_assert!(result.similarity >= -1.0 && result.similarity <= 1.0);
            }
        }
    }
}
