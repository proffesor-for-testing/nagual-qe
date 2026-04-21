//! ProfDAG Schema Tests - Phase 1
//!
//! Comprehensive test suite for ProfDAG node and edge schema operations.
//! Tests cover CRUD operations, type validation, and metadata handling.
//!
//! # Node Types
//! - pattern: Stored problem-solution pairs
//! - trajectory: Recorded decision paths
//! - prediction: Future outcome forecasts
//! - decision: Agent action choices
//!
//! # Edge Types
//! - leads_to: Causal progression
//! - similar_to: Semantic similarity
//! - derived_from: Origin relationship
//! - wormhole: Non-local connections
//! - temporal_link: Time-based associations

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

mod common;
use common::{
    normalized_embedding, random_embedding, random_pattern_id, TestFixture,
};

// ============================================================================
// ProfDAG Node Types and Structures
// ============================================================================

/// Types of nodes in the ProfDAG graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfDagNodeType {
    /// A stored problem-solution pattern
    Pattern,
    /// A recorded decision trajectory
    Trajectory,
    /// A prediction about future outcomes
    Prediction,
    /// A specific decision point
    Decision,
}

impl ProfDagNodeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProfDagNodeType::Pattern => "pattern",
            ProfDagNodeType::Trajectory => "trajectory",
            ProfDagNodeType::Prediction => "prediction",
            ProfDagNodeType::Decision => "decision",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "pattern" => Some(ProfDagNodeType::Pattern),
            "trajectory" => Some(ProfDagNodeType::Trajectory),
            "prediction" => Some(ProfDagNodeType::Prediction),
            "decision" => Some(ProfDagNodeType::Decision),
            _ => None,
        }
    }

    pub fn all() -> &'static [ProfDagNodeType] {
        &[
            ProfDagNodeType::Pattern,
            ProfDagNodeType::Trajectory,
            ProfDagNodeType::Prediction,
            ProfDagNodeType::Decision,
        ]
    }
}

/// Types of edges in the ProfDAG graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfDagEdgeType {
    /// Causal progression from one node to another
    LeadsTo,
    /// Semantic similarity between nodes
    SimilarTo,
    /// Origin/derivation relationship
    DerivedFrom,
    /// Non-local wormhole connection
    Wormhole,
    /// Time-based association
    TemporalLink,
}

impl ProfDagEdgeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProfDagEdgeType::LeadsTo => "leads_to",
            ProfDagEdgeType::SimilarTo => "similar_to",
            ProfDagEdgeType::DerivedFrom => "derived_from",
            ProfDagEdgeType::Wormhole => "wormhole",
            ProfDagEdgeType::TemporalLink => "temporal_link",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "leads_to" => Some(ProfDagEdgeType::LeadsTo),
            "similar_to" => Some(ProfDagEdgeType::SimilarTo),
            "derived_from" => Some(ProfDagEdgeType::DerivedFrom),
            "wormhole" => Some(ProfDagEdgeType::Wormhole),
            "temporal_link" => Some(ProfDagEdgeType::TemporalLink),
            _ => None,
        }
    }

    pub fn all() -> &'static [ProfDagEdgeType] {
        &[
            ProfDagEdgeType::LeadsTo,
            ProfDagEdgeType::SimilarTo,
            ProfDagEdgeType::DerivedFrom,
            ProfDagEdgeType::Wormhole,
            ProfDagEdgeType::TemporalLink,
        ]
    }

    /// Check if this edge type is symmetric.
    pub fn is_symmetric(&self) -> bool {
        matches!(self, ProfDagEdgeType::SimilarTo | ProfDagEdgeType::TemporalLink)
    }
}

/// A node in the ProfDAG graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfDagNode {
    pub id: String,
    pub node_type: ProfDagNodeType,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl ProfDagNode {
    pub fn new(node_type: ProfDagNodeType, content: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            node_type,
            content: content.into(),
            embedding: None,
            metadata: serde_json::json!({}),
            created_at: now,
            updated_at: now,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn pattern(content: impl Into<String>) -> Self {
        Self::new(ProfDagNodeType::Pattern, content)
    }

    pub fn trajectory(content: impl Into<String>) -> Self {
        Self::new(ProfDagNodeType::Trajectory, content)
    }

    pub fn prediction(content: impl Into<String>) -> Self {
        Self::new(ProfDagNodeType::Prediction, content)
    }

    pub fn decision(content: impl Into<String>) -> Self {
        Self::new(ProfDagNodeType::Decision, content)
    }
}

/// An edge in the ProfDAG graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfDagEdge {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub edge_type: ProfDagEdgeType,
    pub weight: f64,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

impl ProfDagEdge {
    pub fn new(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        edge_type: ProfDagEdgeType,
        weight: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_id: source_id.into(),
            target_id: target_id.into(),
            edge_type,
            weight: weight.clamp(0.0, 1.0),
            metadata: serde_json::json!({}),
            created_at: Utc::now(),
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn leads_to(source: impl Into<String>, target: impl Into<String>, weight: f64) -> Self {
        Self::new(source, target, ProfDagEdgeType::LeadsTo, weight)
    }

    pub fn similar_to(source: impl Into<String>, target: impl Into<String>, weight: f64) -> Self {
        Self::new(source, target, ProfDagEdgeType::SimilarTo, weight)
    }

    pub fn derived_from(source: impl Into<String>, target: impl Into<String>, weight: f64) -> Self {
        Self::new(source, target, ProfDagEdgeType::DerivedFrom, weight)
    }
}

/// In-memory ProfDAG storage for testing.
#[derive(Debug, Default)]
pub struct TestProfDagStorage {
    nodes: HashMap<String, ProfDagNode>,
    edges: HashMap<String, ProfDagEdge>,
}

impl TestProfDagStorage {
    pub fn new() -> Self {
        Self::default()
    }

    // Node CRUD operations
    pub fn create_node(&mut self, node: ProfDagNode) -> Result<String, String> {
        if self.nodes.contains_key(&node.id) {
            return Err(format!("Node with id {} already exists", node.id));
        }
        let id = node.id.clone();
        self.nodes.insert(id.clone(), node);
        Ok(id)
    }

    pub fn get_node(&self, id: &str) -> Option<&ProfDagNode> {
        self.nodes.get(id)
    }

    pub fn update_node(&mut self, id: &str, content: Option<String>, metadata: Option<serde_json::Value>) -> Result<(), String> {
        let node = self.nodes.get_mut(id).ok_or_else(|| format!("Node not found: {}", id))?;
        if let Some(c) = content {
            node.content = c;
        }
        if let Some(m) = metadata {
            node.metadata = m;
        }
        node.updated_at = Utc::now();
        Ok(())
    }

    pub fn delete_node(&mut self, id: &str) -> Result<ProfDagNode, String> {
        // Remove associated edges first
        let edges_to_remove: Vec<String> = self.edges
            .iter()
            .filter(|(_, e)| e.source_id == id || e.target_id == id)
            .map(|(k, _)| k.clone())
            .collect();
        for edge_id in edges_to_remove {
            self.edges.remove(&edge_id);
        }
        self.nodes.remove(id).ok_or_else(|| format!("Node not found: {}", id))
    }

    pub fn list_nodes(&self, node_type: Option<ProfDagNodeType>) -> Vec<&ProfDagNode> {
        self.nodes
            .values()
            .filter(|n| node_type.map_or(true, |t| n.node_type == t))
            .collect()
    }

    // Edge CRUD operations
    pub fn create_edge(&mut self, edge: ProfDagEdge) -> Result<String, String> {
        // Validate source and target exist
        if !self.nodes.contains_key(&edge.source_id) {
            return Err(format!("Source node not found: {}", edge.source_id));
        }
        if !self.nodes.contains_key(&edge.target_id) {
            return Err(format!("Target node not found: {}", edge.target_id));
        }
        // Check for self-loop
        if edge.source_id == edge.target_id && !matches!(edge.edge_type, ProfDagEdgeType::Wormhole) {
            return Err("Self-loops are only allowed for wormhole edges".to_string());
        }
        let id = edge.id.clone();
        self.edges.insert(id.clone(), edge);
        Ok(id)
    }

    pub fn get_edge(&self, id: &str) -> Option<&ProfDagEdge> {
        self.edges.get(id)
    }

    pub fn delete_edge(&mut self, id: &str) -> Result<ProfDagEdge, String> {
        self.edges.remove(id).ok_or_else(|| format!("Edge not found: {}", id))
    }

    pub fn list_edges(&self, edge_type: Option<ProfDagEdgeType>) -> Vec<&ProfDagEdge> {
        self.edges
            .values()
            .filter(|e| edge_type.map_or(true, |t| e.edge_type == t))
            .collect()
    }

    pub fn get_outgoing_edges(&self, node_id: &str) -> Vec<&ProfDagEdge> {
        self.edges
            .values()
            .filter(|e| e.source_id == node_id)
            .collect()
    }

    pub fn get_incoming_edges(&self, node_id: &str) -> Vec<&ProfDagEdge> {
        self.edges
            .values()
            .filter(|e| e.target_id == node_id)
            .collect()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

// ============================================================================
// Node CRUD Tests
// ============================================================================

mod node_crud_tests {
    use super::*;

    #[test]
    fn test_create_node_pattern() {
        let mut storage = TestProfDagStorage::new();
        let node = ProfDagNode::pattern("How to handle database timeouts");
        let id = storage.create_node(node.clone()).unwrap();

        assert!(!id.is_empty());
        let retrieved = storage.get_node(&id).unwrap();
        assert_eq!(retrieved.content, "How to handle database timeouts");
        assert_eq!(retrieved.node_type, ProfDagNodeType::Pattern);
    }

    #[test]
    fn test_create_all_node_types() {
        let mut storage = TestProfDagStorage::new();

        for node_type in ProfDagNodeType::all() {
            let node = ProfDagNode::new(*node_type, format!("Content for {:?}", node_type));
            let id = storage.create_node(node).unwrap();
            let retrieved = storage.get_node(&id).unwrap();
            assert_eq!(retrieved.node_type, *node_type);
        }

        assert_eq!(storage.node_count(), 4);
    }

    #[test]
    fn test_create_node_with_embedding() {
        let mut storage = TestProfDagStorage::new();
        let embedding = normalized_embedding(128);
        let node = ProfDagNode::pattern("Pattern with embedding")
            .with_embedding(embedding.clone());

        let id = storage.create_node(node).unwrap();
        let retrieved = storage.get_node(&id).unwrap();

        assert!(retrieved.embedding.is_some());
        assert_eq!(retrieved.embedding.as_ref().unwrap().len(), 128);
    }

    #[test]
    fn test_create_node_with_metadata() {
        let mut storage = TestProfDagStorage::new();
        let metadata = serde_json::json!({
            "domain": "database",
            "confidence": 0.95,
            "tags": ["resilience", "timeout"]
        });
        let node = ProfDagNode::pattern("Pattern with metadata")
            .with_metadata(metadata.clone());

        let id = storage.create_node(node).unwrap();
        let retrieved = storage.get_node(&id).unwrap();

        assert_eq!(retrieved.metadata["domain"], "database");
        assert_eq!(retrieved.metadata["confidence"], 0.95);
        assert_eq!(retrieved.metadata["tags"][0], "resilience");
    }

    #[test]
    fn test_create_duplicate_node_fails() {
        let mut storage = TestProfDagStorage::new();
        let node1 = ProfDagNode::pattern("First node").with_id("unique-id");
        let node2 = ProfDagNode::pattern("Second node").with_id("unique-id");

        storage.create_node(node1).unwrap();
        let result = storage.create_node(node2);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already exists"));
    }

    #[test]
    fn test_get_nonexistent_node() {
        let storage = TestProfDagStorage::new();
        let result = storage.get_node("nonexistent-id");
        assert!(result.is_none());
    }

    #[test]
    fn test_update_node_content() {
        let mut storage = TestProfDagStorage::new();
        let node = ProfDagNode::pattern("Original content");
        let id = storage.create_node(node).unwrap();

        storage.update_node(&id, Some("Updated content".to_string()), None).unwrap();
        let retrieved = storage.get_node(&id).unwrap();

        assert_eq!(retrieved.content, "Updated content");
    }

    #[test]
    fn test_update_node_metadata() {
        let mut storage = TestProfDagStorage::new();
        let node = ProfDagNode::pattern("Content")
            .with_metadata(serde_json::json!({"version": 1}));
        let id = storage.create_node(node).unwrap();

        storage.update_node(&id, None, Some(serde_json::json!({"version": 2, "updated": true}))).unwrap();
        let retrieved = storage.get_node(&id).unwrap();

        assert_eq!(retrieved.metadata["version"], 2);
        assert_eq!(retrieved.metadata["updated"], true);
    }

    #[test]
    fn test_update_nonexistent_node_fails() {
        let mut storage = TestProfDagStorage::new();
        let result = storage.update_node("nonexistent", Some("Content".to_string()), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_node() {
        let mut storage = TestProfDagStorage::new();
        let node = ProfDagNode::pattern("To be deleted");
        let id = storage.create_node(node).unwrap();

        assert_eq!(storage.node_count(), 1);
        let deleted = storage.delete_node(&id).unwrap();
        assert_eq!(deleted.content, "To be deleted");
        assert_eq!(storage.node_count(), 0);
    }

    #[test]
    fn test_delete_node_removes_edges() {
        let mut storage = TestProfDagStorage::new();
        let node1 = ProfDagNode::pattern("Node 1");
        let node2 = ProfDagNode::pattern("Node 2");
        let id1 = storage.create_node(node1).unwrap();
        let id2 = storage.create_node(node2).unwrap();

        let edge = ProfDagEdge::leads_to(&id1, &id2, 0.8);
        storage.create_edge(edge).unwrap();

        assert_eq!(storage.edge_count(), 1);
        storage.delete_node(&id1).unwrap();
        assert_eq!(storage.edge_count(), 0);
    }

    #[test]
    fn test_delete_nonexistent_node_fails() {
        let mut storage = TestProfDagStorage::new();
        let result = storage.delete_node("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_list_nodes_all() {
        let mut storage = TestProfDagStorage::new();
        storage.create_node(ProfDagNode::pattern("Pattern 1")).unwrap();
        storage.create_node(ProfDagNode::trajectory("Trajectory 1")).unwrap();
        storage.create_node(ProfDagNode::prediction("Prediction 1")).unwrap();

        let all_nodes = storage.list_nodes(None);
        assert_eq!(all_nodes.len(), 3);
    }

    #[test]
    fn test_list_nodes_by_type() {
        let mut storage = TestProfDagStorage::new();
        storage.create_node(ProfDagNode::pattern("Pattern 1")).unwrap();
        storage.create_node(ProfDagNode::pattern("Pattern 2")).unwrap();
        storage.create_node(ProfDagNode::trajectory("Trajectory 1")).unwrap();

        let patterns = storage.list_nodes(Some(ProfDagNodeType::Pattern));
        assert_eq!(patterns.len(), 2);
        assert!(patterns.iter().all(|n| n.node_type == ProfDagNodeType::Pattern));

        let trajectories = storage.list_nodes(Some(ProfDagNodeType::Trajectory));
        assert_eq!(trajectories.len(), 1);
    }
}

// ============================================================================
// Edge CRUD Tests
// ============================================================================

mod edge_crud_tests {
    use super::*;

    #[test]
    fn test_create_edge_leads_to() {
        let mut storage = TestProfDagStorage::new();
        let id1 = storage.create_node(ProfDagNode::pattern("Source")).unwrap();
        let id2 = storage.create_node(ProfDagNode::pattern("Target")).unwrap();

        let edge = ProfDagEdge::leads_to(&id1, &id2, 0.85);
        let edge_id = storage.create_edge(edge).unwrap();

        let retrieved = storage.get_edge(&edge_id).unwrap();
        assert_eq!(retrieved.source_id, id1);
        assert_eq!(retrieved.target_id, id2);
        assert_eq!(retrieved.edge_type, ProfDagEdgeType::LeadsTo);
        assert!((retrieved.weight - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_create_all_edge_types() {
        let mut storage = TestProfDagStorage::new();
        let id1 = storage.create_node(ProfDagNode::pattern("Node 1")).unwrap();
        let id2 = storage.create_node(ProfDagNode::pattern("Node 2")).unwrap();

        for (i, edge_type) in ProfDagEdgeType::all().iter().enumerate() {
            // Skip self-loop check for regular edges
            if *edge_type == ProfDagEdgeType::Wormhole {
                // Wormhole can be self-referential
                let edge = ProfDagEdge::new(&id1, &id1, *edge_type, 0.5);
                storage.create_edge(edge).unwrap();
            } else {
                let edge = ProfDagEdge::new(&id1, &id2, *edge_type, (i as f64 + 1.0) / 10.0);
                let edge_id = storage.create_edge(edge).unwrap();
                let retrieved = storage.get_edge(&edge_id).unwrap();
                assert_eq!(retrieved.edge_type, *edge_type);
            }
        }
    }

    #[test]
    fn test_edge_weight_clamping() {
        let mut storage = TestProfDagStorage::new();
        let id1 = storage.create_node(ProfDagNode::pattern("Node 1")).unwrap();
        let id2 = storage.create_node(ProfDagNode::pattern("Node 2")).unwrap();

        // Test weight > 1.0 gets clamped
        let edge_high = ProfDagEdge::leads_to(&id1, &id2, 1.5);
        assert!((edge_high.weight - 1.0).abs() < f64::EPSILON);

        // Test weight < 0.0 gets clamped
        let edge_low = ProfDagEdge::leads_to(&id1, &id2, -0.5);
        assert!(edge_low.weight >= 0.0);
    }

    #[test]
    fn test_create_edge_with_metadata() {
        let mut storage = TestProfDagStorage::new();
        let id1 = storage.create_node(ProfDagNode::pattern("Node 1")).unwrap();
        let id2 = storage.create_node(ProfDagNode::pattern("Node 2")).unwrap();

        let metadata = serde_json::json!({
            "reason": "co-retrieval",
            "count": 42
        });
        let edge = ProfDagEdge::similar_to(&id1, &id2, 0.9).with_metadata(metadata);
        let edge_id = storage.create_edge(edge).unwrap();

        let retrieved = storage.get_edge(&edge_id).unwrap();
        assert_eq!(retrieved.metadata["reason"], "co-retrieval");
        assert_eq!(retrieved.metadata["count"], 42);
    }

    #[test]
    fn test_create_edge_missing_source_fails() {
        let mut storage = TestProfDagStorage::new();
        let id2 = storage.create_node(ProfDagNode::pattern("Target")).unwrap();

        let edge = ProfDagEdge::leads_to("nonexistent", &id2, 0.5);
        let result = storage.create_edge(edge);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Source node not found"));
    }

    #[test]
    fn test_create_edge_missing_target_fails() {
        let mut storage = TestProfDagStorage::new();
        let id1 = storage.create_node(ProfDagNode::pattern("Source")).unwrap();

        let edge = ProfDagEdge::leads_to(&id1, "nonexistent", 0.5);
        let result = storage.create_edge(edge);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Target node not found"));
    }

    #[test]
    fn test_self_loop_not_allowed_except_wormhole() {
        let mut storage = TestProfDagStorage::new();
        let id = storage.create_node(ProfDagNode::pattern("Node")).unwrap();

        // Regular edge types should not allow self-loops
        let edge = ProfDagEdge::leads_to(&id, &id, 0.5);
        let result = storage.create_edge(edge);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Self-loops"));

        // Wormhole should allow self-loops
        let wormhole = ProfDagEdge::new(&id, &id, ProfDagEdgeType::Wormhole, 0.5);
        let result = storage.create_edge(wormhole);
        assert!(result.is_ok());
    }

    #[test]
    fn test_delete_edge() {
        let mut storage = TestProfDagStorage::new();
        let id1 = storage.create_node(ProfDagNode::pattern("Node 1")).unwrap();
        let id2 = storage.create_node(ProfDagNode::pattern("Node 2")).unwrap();

        let edge = ProfDagEdge::leads_to(&id1, &id2, 0.8);
        let edge_id = storage.create_edge(edge).unwrap();

        assert_eq!(storage.edge_count(), 1);
        let deleted = storage.delete_edge(&edge_id).unwrap();
        assert_eq!(deleted.edge_type, ProfDagEdgeType::LeadsTo);
        assert_eq!(storage.edge_count(), 0);
    }

    #[test]
    fn test_delete_nonexistent_edge_fails() {
        let mut storage = TestProfDagStorage::new();
        let result = storage.delete_edge("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_list_edges_all() {
        let mut storage = TestProfDagStorage::new();
        let id1 = storage.create_node(ProfDagNode::pattern("Node 1")).unwrap();
        let id2 = storage.create_node(ProfDagNode::pattern("Node 2")).unwrap();
        let id3 = storage.create_node(ProfDagNode::pattern("Node 3")).unwrap();

        storage.create_edge(ProfDagEdge::leads_to(&id1, &id2, 0.8)).unwrap();
        storage.create_edge(ProfDagEdge::similar_to(&id2, &id3, 0.7)).unwrap();
        storage.create_edge(ProfDagEdge::derived_from(&id3, &id1, 0.6)).unwrap();

        let all_edges = storage.list_edges(None);
        assert_eq!(all_edges.len(), 3);
    }

    #[test]
    fn test_list_edges_by_type() {
        let mut storage = TestProfDagStorage::new();
        let id1 = storage.create_node(ProfDagNode::pattern("Node 1")).unwrap();
        let id2 = storage.create_node(ProfDagNode::pattern("Node 2")).unwrap();
        let id3 = storage.create_node(ProfDagNode::pattern("Node 3")).unwrap();

        storage.create_edge(ProfDagEdge::leads_to(&id1, &id2, 0.8)).unwrap();
        storage.create_edge(ProfDagEdge::leads_to(&id2, &id3, 0.7)).unwrap();
        storage.create_edge(ProfDagEdge::similar_to(&id1, &id3, 0.6)).unwrap();

        let leads_to_edges = storage.list_edges(Some(ProfDagEdgeType::LeadsTo));
        assert_eq!(leads_to_edges.len(), 2);

        let similar_edges = storage.list_edges(Some(ProfDagEdgeType::SimilarTo));
        assert_eq!(similar_edges.len(), 1);
    }

    #[test]
    fn test_get_outgoing_edges() {
        let mut storage = TestProfDagStorage::new();
        let id1 = storage.create_node(ProfDagNode::pattern("Node 1")).unwrap();
        let id2 = storage.create_node(ProfDagNode::pattern("Node 2")).unwrap();
        let id3 = storage.create_node(ProfDagNode::pattern("Node 3")).unwrap();

        storage.create_edge(ProfDagEdge::leads_to(&id1, &id2, 0.8)).unwrap();
        storage.create_edge(ProfDagEdge::leads_to(&id1, &id3, 0.7)).unwrap();
        storage.create_edge(ProfDagEdge::leads_to(&id2, &id3, 0.6)).unwrap();

        let outgoing = storage.get_outgoing_edges(&id1);
        assert_eq!(outgoing.len(), 2);
        assert!(outgoing.iter().all(|e| e.source_id == id1));
    }

    #[test]
    fn test_get_incoming_edges() {
        let mut storage = TestProfDagStorage::new();
        let id1 = storage.create_node(ProfDagNode::pattern("Node 1")).unwrap();
        let id2 = storage.create_node(ProfDagNode::pattern("Node 2")).unwrap();
        let id3 = storage.create_node(ProfDagNode::pattern("Node 3")).unwrap();

        storage.create_edge(ProfDagEdge::leads_to(&id1, &id3, 0.8)).unwrap();
        storage.create_edge(ProfDagEdge::leads_to(&id2, &id3, 0.7)).unwrap();

        let incoming = storage.get_incoming_edges(&id3);
        assert_eq!(incoming.len(), 2);
        assert!(incoming.iter().all(|e| e.target_id == id3));
    }
}

// ============================================================================
// Type Validation Tests
// ============================================================================

mod type_validation_tests {
    use super::*;

    #[test]
    fn test_node_type_from_str() {
        assert_eq!(ProfDagNodeType::from_str("pattern"), Some(ProfDagNodeType::Pattern));
        assert_eq!(ProfDagNodeType::from_str("PATTERN"), Some(ProfDagNodeType::Pattern));
        assert_eq!(ProfDagNodeType::from_str("trajectory"), Some(ProfDagNodeType::Trajectory));
        assert_eq!(ProfDagNodeType::from_str("prediction"), Some(ProfDagNodeType::Prediction));
        assert_eq!(ProfDagNodeType::from_str("decision"), Some(ProfDagNodeType::Decision));
        assert_eq!(ProfDagNodeType::from_str("invalid"), None);
    }

    #[test]
    fn test_node_type_as_str() {
        assert_eq!(ProfDagNodeType::Pattern.as_str(), "pattern");
        assert_eq!(ProfDagNodeType::Trajectory.as_str(), "trajectory");
        assert_eq!(ProfDagNodeType::Prediction.as_str(), "prediction");
        assert_eq!(ProfDagNodeType::Decision.as_str(), "decision");
    }

    #[test]
    fn test_node_type_roundtrip() {
        for node_type in ProfDagNodeType::all() {
            let str_repr = node_type.as_str();
            let parsed = ProfDagNodeType::from_str(str_repr).unwrap();
            assert_eq!(parsed, *node_type);
        }
    }

    #[test]
    fn test_edge_type_from_str() {
        assert_eq!(ProfDagEdgeType::from_str("leads_to"), Some(ProfDagEdgeType::LeadsTo));
        assert_eq!(ProfDagEdgeType::from_str("SIMILAR_TO"), Some(ProfDagEdgeType::SimilarTo));
        assert_eq!(ProfDagEdgeType::from_str("derived_from"), Some(ProfDagEdgeType::DerivedFrom));
        assert_eq!(ProfDagEdgeType::from_str("wormhole"), Some(ProfDagEdgeType::Wormhole));
        assert_eq!(ProfDagEdgeType::from_str("temporal_link"), Some(ProfDagEdgeType::TemporalLink));
        assert_eq!(ProfDagEdgeType::from_str("invalid"), None);
    }

    #[test]
    fn test_edge_type_as_str() {
        assert_eq!(ProfDagEdgeType::LeadsTo.as_str(), "leads_to");
        assert_eq!(ProfDagEdgeType::SimilarTo.as_str(), "similar_to");
        assert_eq!(ProfDagEdgeType::DerivedFrom.as_str(), "derived_from");
        assert_eq!(ProfDagEdgeType::Wormhole.as_str(), "wormhole");
        assert_eq!(ProfDagEdgeType::TemporalLink.as_str(), "temporal_link");
    }

    #[test]
    fn test_edge_type_roundtrip() {
        for edge_type in ProfDagEdgeType::all() {
            let str_repr = edge_type.as_str();
            let parsed = ProfDagEdgeType::from_str(str_repr).unwrap();
            assert_eq!(parsed, *edge_type);
        }
    }

    #[test]
    fn test_edge_type_symmetry() {
        assert!(ProfDagEdgeType::SimilarTo.is_symmetric());
        assert!(ProfDagEdgeType::TemporalLink.is_symmetric());
        assert!(!ProfDagEdgeType::LeadsTo.is_symmetric());
        assert!(!ProfDagEdgeType::DerivedFrom.is_symmetric());
        assert!(!ProfDagEdgeType::Wormhole.is_symmetric());
    }
}

// ============================================================================
// Metadata Handling Tests
// ============================================================================

mod metadata_tests {
    use super::*;

    #[test]
    fn test_node_empty_metadata() {
        let node = ProfDagNode::pattern("Content");
        assert_eq!(node.metadata, serde_json::json!({}));
    }

    #[test]
    fn test_node_complex_metadata() {
        let metadata = serde_json::json!({
            "domain": "database.postgres",
            "confidence": 0.95,
            "tags": ["resilience", "timeout", "retry"],
            "metrics": {
                "usage_count": 42,
                "success_rate": 0.87
            },
            "nested": {
                "level1": {
                    "level2": "deep value"
                }
            }
        });

        let node = ProfDagNode::pattern("Complex pattern").with_metadata(metadata.clone());

        assert_eq!(node.metadata["domain"], "database.postgres");
        assert_eq!(node.metadata["metrics"]["usage_count"], 42);
        assert_eq!(node.metadata["nested"]["level1"]["level2"], "deep value");
    }

    #[test]
    fn test_edge_empty_metadata() {
        let mut storage = TestProfDagStorage::new();
        let id1 = storage.create_node(ProfDagNode::pattern("Node 1")).unwrap();
        let id2 = storage.create_node(ProfDagNode::pattern("Node 2")).unwrap();

        let edge = ProfDagEdge::leads_to(&id1, &id2, 0.8);
        let edge_id = storage.create_edge(edge).unwrap();

        let retrieved = storage.get_edge(&edge_id).unwrap();
        assert_eq!(retrieved.metadata, serde_json::json!({}));
    }

    #[test]
    fn test_metadata_serialization() {
        let node = ProfDagNode::pattern("Test")
            .with_metadata(serde_json::json!({"key": "value"}));

        let serialized = serde_json::to_string(&node).unwrap();
        let deserialized: ProfDagNode = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.metadata["key"], "value");
    }

    #[test]
    fn test_metadata_null_values() {
        let metadata = serde_json::json!({
            "present": "value",
            "null_field": null
        });

        let node = ProfDagNode::pattern("Test").with_metadata(metadata);
        assert_eq!(node.metadata["present"], "value");
        assert!(node.metadata["null_field"].is_null());
        assert!(node.metadata["missing"].is_null()); // Non-existent key
    }

    #[test]
    fn test_metadata_array_values() {
        let metadata = serde_json::json!({
            "items": [1, 2, 3, 4, 5],
            "mixed": ["string", 42, true, null]
        });

        let node = ProfDagNode::pattern("Test").with_metadata(metadata);
        assert_eq!(node.metadata["items"].as_array().unwrap().len(), 5);
        assert_eq!(node.metadata["mixed"][0], "string");
        assert_eq!(node.metadata["mixed"][1], 42);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_content() {
        let mut storage = TestProfDagStorage::new();
        let node = ProfDagNode::pattern("");
        let id = storage.create_node(node).unwrap();
        let retrieved = storage.get_node(&id).unwrap();
        assert_eq!(retrieved.content, "");
    }

    #[test]
    fn test_unicode_content() {
        let mut storage = TestProfDagStorage::new();
        let content = "Unicode: \u{1F600} \u{1F4BB} \u{2764} \u{1F680}";
        let node = ProfDagNode::pattern(content);
        let id = storage.create_node(node).unwrap();
        let retrieved = storage.get_node(&id).unwrap();
        assert_eq!(retrieved.content, content);
    }

    #[test]
    fn test_very_long_content() {
        let mut storage = TestProfDagStorage::new();
        let content = "x".repeat(10000);
        let node = ProfDagNode::pattern(&content);
        let id = storage.create_node(node).unwrap();
        let retrieved = storage.get_node(&id).unwrap();
        assert_eq!(retrieved.content.len(), 10000);
    }

    #[test]
    fn test_special_characters_in_content() {
        let mut storage = TestProfDagStorage::new();
        let content = r#"Special chars: <>'"&\n\t\r`~!@#$%^&*()_+-=[]{}|;:,./?""#;
        let node = ProfDagNode::pattern(content);
        let id = storage.create_node(node).unwrap();
        let retrieved = storage.get_node(&id).unwrap();
        assert_eq!(retrieved.content, content);
    }

    #[test]
    fn test_zero_weight_edge() {
        let mut storage = TestProfDagStorage::new();
        let id1 = storage.create_node(ProfDagNode::pattern("Node 1")).unwrap();
        let id2 = storage.create_node(ProfDagNode::pattern("Node 2")).unwrap();

        let edge = ProfDagEdge::leads_to(&id1, &id2, 0.0);
        let edge_id = storage.create_edge(edge).unwrap();
        let retrieved = storage.get_edge(&edge_id).unwrap();
        assert!(retrieved.weight.abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_weight_edge() {
        let mut storage = TestProfDagStorage::new();
        let id1 = storage.create_node(ProfDagNode::pattern("Node 1")).unwrap();
        let id2 = storage.create_node(ProfDagNode::pattern("Node 2")).unwrap();

        let edge = ProfDagEdge::leads_to(&id1, &id2, 1.0);
        let edge_id = storage.create_edge(edge).unwrap();
        let retrieved = storage.get_edge(&edge_id).unwrap();
        assert!((retrieved.weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_embedding() {
        let node = ProfDagNode::pattern("No embedding");
        assert!(node.embedding.is_none());
    }

    #[test]
    fn test_high_dimensional_embedding() {
        let embedding = normalized_embedding(1536); // Large embedding dimension
        let node = ProfDagNode::pattern("High-dim embedding")
            .with_embedding(embedding);
        assert_eq!(node.embedding.as_ref().unwrap().len(), 1536);
    }

    #[test]
    fn test_single_node_graph() {
        let mut storage = TestProfDagStorage::new();
        let id = storage.create_node(ProfDagNode::pattern("Only node")).unwrap();

        assert_eq!(storage.node_count(), 1);
        assert_eq!(storage.edge_count(), 0);
        assert!(storage.get_outgoing_edges(&id).is_empty());
        assert!(storage.get_incoming_edges(&id).is_empty());
    }

    #[test]
    fn test_dense_graph() {
        let mut storage = TestProfDagStorage::new();
        let mut ids = Vec::new();

        // Create 10 nodes
        for i in 0..10 {
            let id = storage.create_node(ProfDagNode::pattern(format!("Node {}", i))).unwrap();
            ids.push(id);
        }

        // Create edges between all pairs (directed)
        for i in 0..ids.len() {
            for j in 0..ids.len() {
                if i != j {
                    storage.create_edge(ProfDagEdge::leads_to(&ids[i], &ids[j], 0.5)).unwrap();
                }
            }
        }

        assert_eq!(storage.node_count(), 10);
        assert_eq!(storage.edge_count(), 90); // n * (n-1) = 10 * 9 = 90
    }
}

// ============================================================================
// Performance Assertions
// ============================================================================

mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_node_creation_performance() {
        let mut storage = TestProfDagStorage::new();
        let count = 1000;

        let start = Instant::now();
        for i in 0..count {
            storage.create_node(ProfDagNode::pattern(format!("Pattern {}", i))).unwrap();
        }
        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 1000,
            "Creating {} nodes took {:?}, expected < 1s",
            count,
            duration
        );
    }

    #[test]
    fn test_edge_creation_performance() {
        let mut storage = TestProfDagStorage::new();

        // Create nodes first
        let mut ids = Vec::new();
        for i in 0..100 {
            let id = storage.create_node(ProfDagNode::pattern(format!("Node {}", i))).unwrap();
            ids.push(id);
        }

        let start = Instant::now();
        for i in 0..99 {
            storage.create_edge(ProfDagEdge::leads_to(&ids[i], &ids[i + 1], 0.5)).unwrap();
        }
        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 100,
            "Creating {} edges took {:?}, expected < 100ms",
            99,
            duration
        );
    }

    #[test]
    fn test_node_lookup_performance() {
        let mut storage = TestProfDagStorage::new();
        let mut ids = Vec::new();

        for i in 0..1000 {
            let id = storage.create_node(ProfDagNode::pattern(format!("Pattern {}", i))).unwrap();
            ids.push(id);
        }

        let start = Instant::now();
        for id in &ids {
            storage.get_node(id).unwrap();
        }
        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 100,
            "Looking up {} nodes took {:?}, expected < 100ms",
            ids.len(),
            duration
        );
    }

    #[test]
    fn test_list_nodes_performance() {
        let mut storage = TestProfDagStorage::new();

        for i in 0..5000 {
            storage.create_node(ProfDagNode::pattern(format!("Pattern {}", i))).unwrap();
        }

        let start = Instant::now();
        let nodes = storage.list_nodes(None);
        let duration = start.elapsed();

        assert_eq!(nodes.len(), 5000);
        assert!(
            duration.as_millis() < 50,
            "Listing {} nodes took {:?}, expected < 50ms",
            nodes.len(),
            duration
        );
    }
}
