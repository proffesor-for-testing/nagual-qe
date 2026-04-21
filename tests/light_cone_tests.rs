//! Light Cone Tests - Phase 3 (Learning Layer)
//!
//! Comprehensive test suite for light cone functionality.
//! Light cones provide causal/temporal analysis of patterns:
//! - History cone: What led to this pattern?
//! - Future cone: What outcomes typically follow?
//!
//! # Light Cone Mechanics
//! - History cone traversal (trace_back, find_root_causes)
//! - Future cone predictions (predict_outcomes, likely_next_patterns)
//! - Cognitive core management (active patterns, attention weights)
//! - Temporal queries ("what led to X", "what follows Y")
//! - Causal chain construction
//!
//! # Test Categories
//! - History cone traversal
//! - Future cone predictions
//! - Cognitive core management
//! - Temporal queries
//! - Causal chain construction
//! - Edge cases (circular references, missing nodes)

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

mod common;
use common::{cosine_similarity, normalized_embedding, similar_embeddings};

// ============================================================================
// Light Cone Types
// ============================================================================

/// Direction of causal traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalDirection {
    /// Backward in time - what caused this?
    Past,
    /// Forward in time - what does this cause?
    Future,
    /// Both directions
    Both,
}

/// A node in the causal graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalNode {
    pub id: String,
    pub content: String,
    pub node_type: String,
    pub timestamp: DateTime<Utc>,
    pub confidence: f32,
    pub embedding: Option<Vec<f32>>,
    pub metadata: serde_json::Value,
}

impl CausalNode {
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            node_type: "pattern".to_string(),
            timestamp: Utc::now(),
            confidence: 0.5,
            embedding: None,
            metadata: serde_json::json!({}),
        }
    }

    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_type(mut self, node_type: impl Into<String>) -> Self {
        self.node_type = node_type.into();
        self
    }
}

/// A causal link between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLink {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub strength: f64,
    pub link_type: CausalLinkType,
    pub evidence_count: u32,
    pub created_at: DateTime<Utc>,
}

/// Types of causal links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalLinkType {
    /// Direct causation
    Causes,
    /// Temporal precedence
    PrecedesTemporally,
    /// Enables/allows
    Enables,
    /// Part of the same outcome
    CoOccurs,
    /// Derived from trajectory data
    TrajectoryLink,
}

impl CausalLink {
    pub fn new(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        link_type: CausalLinkType,
        strength: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_id: source_id.into(),
            target_id: target_id.into(),
            strength: strength.clamp(0.0, 1.0),
            link_type,
            evidence_count: 1,
            created_at: Utc::now(),
        }
    }

    pub fn reinforce(&mut self) {
        self.evidence_count += 1;
        // Increase strength with diminishing returns
        self.strength = (self.strength + 0.1 * (1.0 - self.strength)).min(1.0);
    }
}

/// Result of a history cone query.
#[derive(Debug, Clone)]
pub struct HistoryCone {
    /// The focal node (what we're tracing back from).
    pub focal_node_id: String,
    /// Nodes in the history cone, ordered by distance from focal.
    pub nodes: Vec<(CausalNode, usize)>,
    /// Links in the history cone.
    pub links: Vec<CausalLink>,
    /// Identified root causes (nodes with no incoming causal links).
    pub root_causes: Vec<String>,
    /// Maximum depth reached.
    pub max_depth: usize,
}

/// Result of a future cone query.
#[derive(Debug, Clone)]
pub struct FutureCone {
    /// The focal node (what we're projecting from).
    pub focal_node_id: String,
    /// Nodes in the future cone, ordered by distance from focal.
    pub nodes: Vec<(CausalNode, usize)>,
    /// Links in the future cone.
    pub links: Vec<CausalLink>,
    /// Predicted outcomes with probabilities.
    pub predicted_outcomes: Vec<(String, f64)>,
    /// Maximum depth reached.
    pub max_depth: usize,
}

/// A causal chain from source to target.
#[derive(Debug, Clone)]
pub struct CausalChain {
    /// Ordered list of nodes in the chain.
    pub nodes: Vec<CausalNode>,
    /// Links connecting the nodes.
    pub links: Vec<CausalLink>,
    /// Overall chain strength (product of link strengths).
    pub strength: f64,
    /// Total evidence supporting this chain.
    pub total_evidence: u32,
}

impl CausalChain {
    pub fn new(nodes: Vec<CausalNode>, links: Vec<CausalLink>) -> Self {
        let strength = links.iter().map(|l| l.strength).product();
        let total_evidence = links.iter().map(|l| l.evidence_count).sum();
        Self {
            nodes,
            links,
            strength,
            total_evidence,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ============================================================================
// Cognitive Core
// ============================================================================

/// Active pattern in the cognitive core.
#[derive(Debug, Clone)]
pub struct ActivePattern {
    pub node_id: String,
    pub activation_level: f64,
    pub attention_weight: f64,
    pub activated_at: DateTime<Utc>,
    pub decay_rate: f64,
}

impl ActivePattern {
    pub fn new(node_id: impl Into<String>, initial_activation: f64) -> Self {
        Self {
            node_id: node_id.into(),
            activation_level: initial_activation.clamp(0.0, 1.0),
            attention_weight: 0.5,
            activated_at: Utc::now(),
            decay_rate: 0.1,
        }
    }

    /// Apply time-based decay to activation level.
    pub fn apply_decay(&mut self, elapsed_seconds: f64) {
        let decay = (-self.decay_rate * elapsed_seconds).exp();
        self.activation_level *= decay;
    }

    /// Boost activation (e.g., when re-accessed).
    pub fn boost(&mut self, amount: f64) {
        self.activation_level = (self.activation_level + amount).min(1.0);
        self.activated_at = Utc::now();
    }

    /// Set attention weight.
    pub fn set_attention(&mut self, weight: f64) {
        self.attention_weight = weight.clamp(0.0, 1.0);
    }

    /// Get effective activation (activation * attention).
    pub fn effective_activation(&self) -> f64 {
        self.activation_level * self.attention_weight
    }
}

/// Cognitive core managing active patterns and attention.
#[derive(Debug)]
pub struct CognitiveCore {
    active_patterns: HashMap<String, ActivePattern>,
    max_active: usize,
    activation_threshold: f64,
}

impl CognitiveCore {
    pub fn new(max_active: usize) -> Self {
        Self {
            active_patterns: HashMap::new(),
            max_active,
            activation_threshold: 0.1,
        }
    }

    /// Activate a pattern.
    pub fn activate(&mut self, node_id: &str, initial_activation: f64) {
        if let Some(pattern) = self.active_patterns.get_mut(node_id) {
            pattern.boost(initial_activation * 0.5);
        } else {
            // Check if we need to evict
            if self.active_patterns.len() >= self.max_active {
                self.evict_weakest();
            }
            self.active_patterns.insert(
                node_id.to_string(),
                ActivePattern::new(node_id, initial_activation),
            );
        }
    }

    /// Evict the weakest active pattern.
    fn evict_weakest(&mut self) {
        if let Some((weakest_id, _)) = self
            .active_patterns
            .iter()
            .min_by(|a, b| {
                a.1.effective_activation()
                    .partial_cmp(&b.1.effective_activation())
                    .unwrap()
            })
            .map(|(k, v)| (k.clone(), v.effective_activation()))
        {
            self.active_patterns.remove(&weakest_id);
        }
    }

    /// Apply decay to all active patterns.
    pub fn apply_decay(&mut self, elapsed_seconds: f64) {
        let mut to_remove = Vec::new();

        for (id, pattern) in self.active_patterns.iter_mut() {
            pattern.apply_decay(elapsed_seconds);
            if pattern.activation_level < self.activation_threshold {
                to_remove.push(id.clone());
            }
        }

        for id in to_remove {
            self.active_patterns.remove(&id);
        }
    }

    /// Update attention weights based on relevance scores.
    pub fn update_attention(&mut self, relevance_scores: &HashMap<String, f64>) {
        // Normalize scores
        let total: f64 = relevance_scores.values().sum();
        if total > 0.0 {
            for (id, pattern) in self.active_patterns.iter_mut() {
                if let Some(&score) = relevance_scores.get(id) {
                    pattern.set_attention(score / total);
                }
            }
        }
    }

    /// Get all active pattern IDs sorted by effective activation.
    pub fn get_active_patterns(&self) -> Vec<&str> {
        let mut patterns: Vec<_> = self
            .active_patterns
            .iter()
            .map(|(id, p)| (id.as_str(), p.effective_activation()))
            .collect();
        patterns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        patterns.into_iter().map(|(id, _)| id).collect()
    }

    /// Get pattern by ID.
    pub fn get_pattern(&self, id: &str) -> Option<&ActivePattern> {
        self.active_patterns.get(id)
    }

    /// Count of active patterns.
    pub fn active_count(&self) -> usize {
        self.active_patterns.len()
    }
}

// ============================================================================
// Light Cone Engine
// ============================================================================

/// Engine for computing light cones (history and future cones).
#[derive(Debug)]
pub struct LightConeEngine {
    nodes: HashMap<String, CausalNode>,
    forward_links: HashMap<String, Vec<CausalLink>>,
    backward_links: HashMap<String, Vec<CausalLink>>,
    cognitive_core: CognitiveCore,
}

impl LightConeEngine {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            forward_links: HashMap::new(),
            backward_links: HashMap::new(),
            cognitive_core: CognitiveCore::new(100),
        }
    }

    /// Add a node to the causal graph.
    pub fn add_node(&mut self, node: CausalNode) -> String {
        let id = node.id.clone();
        self.nodes.insert(id.clone(), node);
        id
    }

    /// Add a causal link.
    pub fn add_link(&mut self, link: CausalLink) {
        let link_clone = link.clone();
        self.forward_links
            .entry(link.source_id.clone())
            .or_insert_with(Vec::new)
            .push(link);
        self.backward_links
            .entry(link_clone.target_id.clone())
            .or_insert_with(Vec::new)
            .push(link_clone);
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: &str) -> Option<&CausalNode> {
        self.nodes.get(id)
    }

    /// Trace back through the history cone (what led to this pattern).
    pub fn trace_back(&self, focal_node_id: &str, max_depth: usize) -> HistoryCone {
        let mut visited = HashSet::new();
        let mut nodes = Vec::new();
        let mut links = Vec::new();
        let mut root_causes = Vec::new();

        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((focal_node_id.to_string(), 0));

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth > max_depth || visited.contains(&current_id) {
                continue;
            }
            visited.insert(current_id.clone());

            if let Some(node) = self.nodes.get(&current_id) {
                nodes.push((node.clone(), depth));

                // Get incoming links (causes)
                if let Some(incoming) = self.backward_links.get(&current_id) {
                    if incoming.is_empty() && depth > 0 {
                        root_causes.push(current_id.clone());
                    }
                    for link in incoming {
                        links.push(link.clone());
                        if !visited.contains(&link.source_id) {
                            queue.push_back((link.source_id.clone(), depth + 1));
                        }
                    }
                } else if depth > 0 {
                    // No incoming links - this is a root cause
                    root_causes.push(current_id.clone());
                }
            }
        }

        HistoryCone {
            focal_node_id: focal_node_id.to_string(),
            nodes,
            links,
            root_causes,
            max_depth,
        }
    }

    /// Find root causes (nodes with no incoming causal links in the history cone).
    pub fn find_root_causes(&self, focal_node_id: &str, max_depth: usize) -> Vec<CausalNode> {
        let cone = self.trace_back(focal_node_id, max_depth);
        cone.root_causes
            .iter()
            .filter_map(|id| self.nodes.get(id).cloned())
            .collect()
    }

    /// Predict outcomes by traversing the future cone.
    pub fn predict_outcomes(&self, focal_node_id: &str, max_depth: usize) -> FutureCone {
        let mut visited = HashSet::new();
        let mut nodes = Vec::new();
        let mut links = Vec::new();
        let mut leaf_nodes = Vec::new();

        let mut queue: VecDeque<(String, usize, f64)> = VecDeque::new();
        queue.push_back((focal_node_id.to_string(), 0, 1.0));

        while let Some((current_id, depth, path_prob)) = queue.pop_front() {
            if depth > max_depth || visited.contains(&current_id) {
                continue;
            }
            visited.insert(current_id.clone());

            if let Some(node) = self.nodes.get(&current_id) {
                nodes.push((node.clone(), depth));

                // Get outgoing links (effects)
                if let Some(outgoing) = self.forward_links.get(&current_id) {
                    if outgoing.is_empty() && depth > 0 {
                        leaf_nodes.push((current_id.clone(), path_prob));
                    }
                    for link in outgoing {
                        links.push(link.clone());
                        let new_prob = path_prob * link.strength;
                        if !visited.contains(&link.target_id) && new_prob > 0.01 {
                            queue.push_back((link.target_id.clone(), depth + 1, new_prob));
                        }
                    }
                } else if depth > 0 {
                    // No outgoing links - this is a terminal outcome
                    leaf_nodes.push((current_id.clone(), path_prob));
                }
            }
        }

        // Normalize probabilities
        let total_prob: f64 = leaf_nodes.iter().map(|(_, p)| p).sum();
        let predicted_outcomes: Vec<_> = if total_prob > 0.0 {
            leaf_nodes
                .into_iter()
                .map(|(id, p)| (id, p / total_prob))
                .collect()
        } else {
            leaf_nodes
        };

        FutureCone {
            focal_node_id: focal_node_id.to_string(),
            nodes,
            links,
            predicted_outcomes,
            max_depth,
        }
    }

    /// Get likely next patterns based on current context.
    pub fn likely_next_patterns(&self, current_id: &str, k: usize) -> Vec<(String, f64)> {
        if let Some(outgoing) = self.forward_links.get(current_id) {
            let mut candidates: Vec<_> = outgoing
                .iter()
                .map(|link| (link.target_id.clone(), link.strength))
                .collect();
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            candidates.truncate(k);
            candidates
        } else {
            Vec::new()
        }
    }

    /// Build a causal chain between two nodes.
    pub fn build_causal_chain(
        &self,
        source_id: &str,
        target_id: &str,
        max_length: usize,
    ) -> Option<CausalChain> {
        // BFS to find shortest causal path
        let mut visited = HashSet::new();
        let mut parent: HashMap<String, (String, CausalLink)> = HashMap::new();
        let mut queue = VecDeque::new();

        queue.push_back((source_id.to_string(), 0));
        visited.insert(source_id.to_string());

        while let Some((current, depth)) = queue.pop_front() {
            if current == target_id {
                // Reconstruct path
                let mut chain_nodes = Vec::new();
                let mut chain_links = Vec::new();
                let mut current = target_id.to_string();

                while let Some((prev, link)) = parent.get(&current) {
                    if let Some(node) = self.nodes.get(&current) {
                        chain_nodes.push(node.clone());
                    }
                    chain_links.push(link.clone());
                    current = prev.clone();
                }

                if let Some(node) = self.nodes.get(source_id) {
                    chain_nodes.push(node.clone());
                }

                chain_nodes.reverse();
                chain_links.reverse();

                return Some(CausalChain::new(chain_nodes, chain_links));
            }

            if depth >= max_length {
                continue;
            }

            if let Some(outgoing) = self.forward_links.get(&current) {
                for link in outgoing {
                    if !visited.contains(&link.target_id) {
                        visited.insert(link.target_id.clone());
                        parent.insert(link.target_id.clone(), (current.clone(), link.clone()));
                        queue.push_back((link.target_id.clone(), depth + 1));
                    }
                }
            }
        }

        None
    }

    /// Answer temporal query: "What led to X?"
    pub fn query_what_led_to(&self, target_id: &str, max_depth: usize) -> Vec<CausalNode> {
        let cone = self.trace_back(target_id, max_depth);
        cone.nodes
            .into_iter()
            .filter(|(_, depth)| *depth > 0)
            .map(|(node, _)| node)
            .collect()
    }

    /// Answer temporal query: "What follows Y?"
    pub fn query_what_follows(&self, source_id: &str, max_depth: usize) -> Vec<CausalNode> {
        let cone = self.predict_outcomes(source_id, max_depth);
        cone.nodes
            .into_iter()
            .filter(|(_, depth)| *depth > 0)
            .map(|(node, _)| node)
            .collect()
    }

    /// Get cognitive core reference.
    pub fn cognitive_core(&self) -> &CognitiveCore {
        &self.cognitive_core
    }

    /// Get mutable cognitive core reference.
    pub fn cognitive_core_mut(&mut self) -> &mut CognitiveCore {
        &mut self.cognitive_core
    }

    /// Node count.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Link count.
    pub fn link_count(&self) -> usize {
        self.forward_links.values().map(|v| v.len()).sum()
    }
}

impl Default for LightConeEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// History Cone Tests
// ============================================================================

mod history_cone_tests {
    use super::*;

    fn create_linear_chain(engine: &mut LightConeEngine, count: usize) -> Vec<String> {
        let base_time = Utc::now() - ChronoDuration::hours(count as i64);
        let mut ids = Vec::new();

        for i in 0..count {
            let node = CausalNode::new(format!("node_{}", i), format!("Content {}", i))
                .with_timestamp(base_time + ChronoDuration::hours(i as i64));
            let id = engine.add_node(node);
            ids.push(id);
        }

        // Add causal links
        for i in 0..(count - 1) {
            let link = CausalLink::new(&ids[i], &ids[i + 1], CausalLinkType::Causes, 0.9);
            engine.add_link(link);
        }

        ids
    }

    #[test]
    fn test_trace_back_linear() {
        let mut engine = LightConeEngine::new();
        let ids = create_linear_chain(&mut engine, 5);

        // Trace back from the last node
        let cone = engine.trace_back(&ids[4], 10);

        assert_eq!(cone.focal_node_id, ids[4]);
        assert_eq!(cone.nodes.len(), 5);
        assert_eq!(cone.root_causes.len(), 1);
        assert_eq!(cone.root_causes[0], ids[0]);
    }

    #[test]
    fn test_trace_back_depth_limit() {
        let mut engine = LightConeEngine::new();
        let ids = create_linear_chain(&mut engine, 10);

        // Trace back with limited depth
        let cone = engine.trace_back(&ids[9], 3);

        // Should only go back 3 steps + focal node = 4 nodes max
        assert!(cone.nodes.len() <= 4);
        assert_eq!(cone.max_depth, 3);
    }

    #[test]
    fn test_find_root_causes() {
        let mut engine = LightConeEngine::new();

        // Create a diamond pattern:
        //   A --> C
        //         |
        //   B --> D --> E
        let a = engine.add_node(CausalNode::new("A", "Root A"));
        let b = engine.add_node(CausalNode::new("B", "Root B"));
        let c = engine.add_node(CausalNode::new("C", "Middle C"));
        let d = engine.add_node(CausalNode::new("D", "Middle D"));
        let e = engine.add_node(CausalNode::new("E", "Effect E"));

        engine.add_link(CausalLink::new(&a, &c, CausalLinkType::Causes, 0.8));
        engine.add_link(CausalLink::new(&b, &d, CausalLinkType::Causes, 0.9));
        engine.add_link(CausalLink::new(&c, &e, CausalLinkType::Causes, 0.7));
        engine.add_link(CausalLink::new(&d, &e, CausalLinkType::Causes, 0.85));

        let root_causes = engine.find_root_causes(&e, 10);

        // A and B are root causes
        let root_ids: HashSet<_> = root_causes.iter().map(|n| n.id.as_str()).collect();
        assert!(root_ids.contains("A"));
        assert!(root_ids.contains("B"));
    }

    #[test]
    fn test_trace_back_single_node() {
        let mut engine = LightConeEngine::new();
        let node = CausalNode::new("solo", "Lonely node");
        engine.add_node(node);

        let cone = engine.trace_back("solo", 10);

        assert_eq!(cone.nodes.len(), 1);
        assert!(cone.links.is_empty());
        assert!(cone.root_causes.is_empty()); // Focal node is not counted as root cause
    }

    #[test]
    fn test_trace_back_nonexistent_node() {
        let engine = LightConeEngine::new();
        let cone = engine.trace_back("does_not_exist", 10);

        assert!(cone.nodes.is_empty());
        assert!(cone.links.is_empty());
    }

    #[test]
    fn test_history_cone_with_multiple_paths() {
        let mut engine = LightConeEngine::new();

        // Create multiple paths to the same target
        // R1 -> M1 -> T
        // R2 -> M2 -> T
        let r1 = engine.add_node(CausalNode::new("R1", "Root 1"));
        let r2 = engine.add_node(CausalNode::new("R2", "Root 2"));
        let m1 = engine.add_node(CausalNode::new("M1", "Middle 1"));
        let m2 = engine.add_node(CausalNode::new("M2", "Middle 2"));
        let t = engine.add_node(CausalNode::new("T", "Target"));

        engine.add_link(CausalLink::new(&r1, &m1, CausalLinkType::Causes, 0.9));
        engine.add_link(CausalLink::new(&r2, &m2, CausalLinkType::Causes, 0.8));
        engine.add_link(CausalLink::new(&m1, &t, CausalLinkType::Causes, 0.9));
        engine.add_link(CausalLink::new(&m2, &t, CausalLinkType::Causes, 0.85));

        let cone = engine.trace_back(&t, 10);

        // Should find all 5 nodes
        assert_eq!(cone.nodes.len(), 5);
        // Two root causes
        assert_eq!(cone.root_causes.len(), 2);
    }
}

// ============================================================================
// Future Cone Tests
// ============================================================================

mod future_cone_tests {
    use super::*;

    #[test]
    fn test_predict_outcomes_linear() {
        let mut engine = LightConeEngine::new();

        let a = engine.add_node(CausalNode::new("A", "Start"));
        let b = engine.add_node(CausalNode::new("B", "Middle"));
        let c = engine.add_node(CausalNode::new("C", "End"));

        engine.add_link(CausalLink::new(&a, &b, CausalLinkType::Causes, 0.9));
        engine.add_link(CausalLink::new(&b, &c, CausalLinkType::Causes, 0.8));

        let cone = engine.predict_outcomes(&a, 10);

        assert_eq!(cone.nodes.len(), 3);
        assert_eq!(cone.predicted_outcomes.len(), 1);
        assert_eq!(cone.predicted_outcomes[0].0, "C");
    }

    #[test]
    fn test_predict_outcomes_branching() {
        let mut engine = LightConeEngine::new();

        // A -> B -> C1
        //       \-> C2
        let a = engine.add_node(CausalNode::new("A", "Start"));
        let b = engine.add_node(CausalNode::new("B", "Branch point"));
        let c1 = engine.add_node(CausalNode::new("C1", "Outcome 1"));
        let c2 = engine.add_node(CausalNode::new("C2", "Outcome 2"));

        engine.add_link(CausalLink::new(&a, &b, CausalLinkType::Causes, 1.0));
        engine.add_link(CausalLink::new(&b, &c1, CausalLinkType::Causes, 0.7));
        engine.add_link(CausalLink::new(&b, &c2, CausalLinkType::Causes, 0.3));

        let cone = engine.predict_outcomes(&a, 10);

        assert_eq!(cone.predicted_outcomes.len(), 2);

        // C1 should have higher probability
        let c1_prob = cone.predicted_outcomes.iter().find(|(id, _)| id == "C1").map(|(_, p)| *p);
        let c2_prob = cone.predicted_outcomes.iter().find(|(id, _)| id == "C2").map(|(_, p)| *p);

        assert!(c1_prob.unwrap() > c2_prob.unwrap());
    }

    #[test]
    fn test_likely_next_patterns() {
        let mut engine = LightConeEngine::new();

        let a = engine.add_node(CausalNode::new("A", "Current"));
        let b = engine.add_node(CausalNode::new("B", "Next 1"));
        let c = engine.add_node(CausalNode::new("C", "Next 2"));
        let d = engine.add_node(CausalNode::new("D", "Next 3"));

        engine.add_link(CausalLink::new(&a, &b, CausalLinkType::Causes, 0.9));
        engine.add_link(CausalLink::new(&a, &c, CausalLinkType::Causes, 0.7));
        engine.add_link(CausalLink::new(&a, &d, CausalLinkType::Causes, 0.5));

        let next = engine.likely_next_patterns(&a, 2);

        assert_eq!(next.len(), 2);
        assert_eq!(next[0].0, "B"); // Highest strength
        assert_eq!(next[1].0, "C"); // Second highest
    }

    #[test]
    fn test_predict_outcomes_depth_limit() {
        let mut engine = LightConeEngine::new();

        // Create a long chain
        let mut prev = engine.add_node(CausalNode::new("N0", "Start"));
        for i in 1..10 {
            let curr = engine.add_node(CausalNode::new(format!("N{}", i), format!("Node {}", i)));
            engine.add_link(CausalLink::new(&prev, &curr, CausalLinkType::Causes, 0.9));
            prev = curr;
        }

        let cone = engine.predict_outcomes("N0", 3);

        // Should only reach depth 3
        assert!(cone.nodes.len() <= 4);
    }

    #[test]
    fn test_query_what_follows() {
        let mut engine = LightConeEngine::new();

        let a = engine.add_node(CausalNode::new("A", "Start"));
        let b = engine.add_node(CausalNode::new("B", "Effect 1"));
        let c = engine.add_node(CausalNode::new("C", "Effect 2"));

        engine.add_link(CausalLink::new(&a, &b, CausalLinkType::Causes, 0.9));
        engine.add_link(CausalLink::new(&a, &c, CausalLinkType::Causes, 0.8));

        let effects = engine.query_what_follows(&a, 10);

        // Should find B and C but not A itself
        assert_eq!(effects.len(), 2);
        let ids: HashSet<_> = effects.iter().map(|n| n.id.as_str()).collect();
        assert!(ids.contains("B"));
        assert!(ids.contains("C"));
        assert!(!ids.contains("A"));
    }
}

// ============================================================================
// Cognitive Core Tests
// ============================================================================

mod cognitive_core_tests {
    use super::*;

    #[test]
    fn test_activate_pattern() {
        let mut core = CognitiveCore::new(10);

        core.activate("pattern_1", 0.8);

        assert_eq!(core.active_count(), 1);
        let pattern = core.get_pattern("pattern_1").unwrap();
        assert!((pattern.activation_level - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_reactivation_boosts() {
        let mut core = CognitiveCore::new(10);

        core.activate("pattern_1", 0.5);
        let initial = core.get_pattern("pattern_1").unwrap().activation_level;

        core.activate("pattern_1", 0.5); // Reactivate
        let boosted = core.get_pattern("pattern_1").unwrap().activation_level;

        assert!(boosted > initial);
    }

    #[test]
    fn test_max_active_limit() {
        let mut core = CognitiveCore::new(3);

        core.activate("p1", 0.9);
        core.activate("p2", 0.8);
        core.activate("p3", 0.7);
        assert_eq!(core.active_count(), 3);

        // Adding 4th should evict weakest
        core.activate("p4", 0.95);
        assert_eq!(core.active_count(), 3);

        // p3 (weakest) should be evicted
        assert!(core.get_pattern("p3").is_none());
        assert!(core.get_pattern("p4").is_some());
    }

    #[test]
    fn test_decay_removes_weak_patterns() {
        let mut core = CognitiveCore::new(10);

        core.activate("p1", 0.5);
        core.activate("p2", 0.9);

        // Apply significant decay
        core.apply_decay(50.0); // 50 seconds of decay

        // Only strong patterns should survive
        assert!(core.active_count() <= 2);
    }

    #[test]
    fn test_attention_weights() {
        let mut core = CognitiveCore::new(10);

        core.activate("p1", 0.8);
        core.activate("p2", 0.8);

        let mut relevance = HashMap::new();
        relevance.insert("p1".to_string(), 0.9);
        relevance.insert("p2".to_string(), 0.1);

        core.update_attention(&relevance);

        let p1 = core.get_pattern("p1").unwrap();
        let p2 = core.get_pattern("p2").unwrap();

        assert!(p1.attention_weight > p2.attention_weight);
    }

    #[test]
    fn test_effective_activation() {
        let mut core = CognitiveCore::new(10);
        core.activate("p1", 0.8);

        let mut relevance = HashMap::new();
        relevance.insert("p1".to_string(), 1.0);
        core.update_attention(&relevance);

        let pattern = core.get_pattern("p1").unwrap();
        let effective = pattern.effective_activation();

        assert!(effective <= pattern.activation_level);
    }

    #[test]
    fn test_get_active_patterns_sorted() {
        let mut core = CognitiveCore::new(10);

        core.activate("low", 0.3);
        core.activate("high", 0.9);
        core.activate("mid", 0.6);

        let patterns = core.get_active_patterns();

        assert_eq!(patterns.len(), 3);
        assert_eq!(patterns[0], "high");
        assert_eq!(patterns[1], "mid");
        assert_eq!(patterns[2], "low");
    }
}

// ============================================================================
// Causal Chain Tests
// ============================================================================

mod causal_chain_tests {
    use super::*;

    #[test]
    fn test_build_causal_chain() {
        let mut engine = LightConeEngine::new();

        let a = engine.add_node(CausalNode::new("A", "Start"));
        let b = engine.add_node(CausalNode::new("B", "Middle"));
        let c = engine.add_node(CausalNode::new("C", "End"));

        engine.add_link(CausalLink::new(&a, &b, CausalLinkType::Causes, 0.9));
        engine.add_link(CausalLink::new(&b, &c, CausalLinkType::Causes, 0.8));

        let chain = engine.build_causal_chain(&a, &c, 10);

        assert!(chain.is_some());
        let chain = chain.unwrap();
        assert_eq!(chain.len(), 3);
        assert_eq!(chain.links.len(), 2);

        // Strength is product of link strengths
        let expected_strength = 0.9 * 0.8;
        assert!((chain.strength - expected_strength).abs() < 0.001);
    }

    #[test]
    fn test_no_chain_between_unconnected() {
        let mut engine = LightConeEngine::new();

        engine.add_node(CausalNode::new("A", "Island A"));
        engine.add_node(CausalNode::new("B", "Island B"));

        let chain = engine.build_causal_chain("A", "B", 10);
        assert!(chain.is_none());
    }

    #[test]
    fn test_chain_length_limit() {
        let mut engine = LightConeEngine::new();

        // Create long chain
        let mut prev = engine.add_node(CausalNode::new("N0", "Start"));
        for i in 1..10 {
            let curr = engine.add_node(CausalNode::new(format!("N{}", i), format!("Node {}", i)));
            engine.add_link(CausalLink::new(&prev, &curr, CausalLinkType::Causes, 0.9));
            prev = curr;
        }

        // Try to find chain with length limit
        let chain = engine.build_causal_chain("N0", "N9", 3);
        assert!(chain.is_none()); // Path is 9 links, limit is 3

        let chain = engine.build_causal_chain("N0", "N9", 10);
        assert!(chain.is_some()); // Should find it with higher limit
    }

    #[test]
    fn test_chain_evidence_count() {
        let mut engine = LightConeEngine::new();

        let a = engine.add_node(CausalNode::new("A", "Start"));
        let b = engine.add_node(CausalNode::new("B", "End"));

        let mut link = CausalLink::new(&a, &b, CausalLinkType::Causes, 0.7);
        link.reinforce();
        link.reinforce();
        engine.add_link(link);

        let chain = engine.build_causal_chain(&a, &b, 10).unwrap();
        assert_eq!(chain.total_evidence, 3); // Initial + 2 reinforcements
    }
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_circular_references_history() {
        let mut engine = LightConeEngine::new();

        // Create a cycle: A -> B -> C -> A
        let a = engine.add_node(CausalNode::new("A", "Node A"));
        let b = engine.add_node(CausalNode::new("B", "Node B"));
        let c = engine.add_node(CausalNode::new("C", "Node C"));

        engine.add_link(CausalLink::new(&a, &b, CausalLinkType::Causes, 0.9));
        engine.add_link(CausalLink::new(&b, &c, CausalLinkType::Causes, 0.9));
        engine.add_link(CausalLink::new(&c, &a, CausalLinkType::Causes, 0.9));

        // Should not infinite loop
        let cone = engine.trace_back(&a, 10);

        // Should visit each node once
        let unique_ids: HashSet<_> = cone.nodes.iter().map(|(n, _)| n.id.as_str()).collect();
        assert_eq!(unique_ids.len(), 3);
    }

    #[test]
    fn test_circular_references_future() {
        let mut engine = LightConeEngine::new();

        // Create a cycle
        let a = engine.add_node(CausalNode::new("A", "Node A"));
        let b = engine.add_node(CausalNode::new("B", "Node B"));
        let c = engine.add_node(CausalNode::new("C", "Node C"));

        engine.add_link(CausalLink::new(&a, &b, CausalLinkType::Causes, 0.9));
        engine.add_link(CausalLink::new(&b, &c, CausalLinkType::Causes, 0.9));
        engine.add_link(CausalLink::new(&c, &a, CausalLinkType::Causes, 0.9));

        // Should not infinite loop
        let cone = engine.predict_outcomes(&a, 10);

        // Should visit each node once
        let unique_ids: HashSet<_> = cone.nodes.iter().map(|(n, _)| n.id.as_str()).collect();
        assert_eq!(unique_ids.len(), 3);
    }

    #[test]
    fn test_missing_nodes_in_query() {
        let engine = LightConeEngine::new();

        let causes = engine.query_what_led_to("nonexistent", 10);
        assert!(causes.is_empty());

        let effects = engine.query_what_follows("nonexistent", 10);
        assert!(effects.is_empty());
    }

    #[test]
    fn test_self_loop() {
        let mut engine = LightConeEngine::new();

        let a = engine.add_node(CausalNode::new("A", "Self-referential"));
        engine.add_link(CausalLink::new(&a, &a, CausalLinkType::Causes, 0.5));

        let cone = engine.trace_back(&a, 10);
        assert_eq!(cone.nodes.len(), 1);
    }

    #[test]
    fn test_empty_graph() {
        let engine = LightConeEngine::new();

        assert_eq!(engine.node_count(), 0);
        assert_eq!(engine.link_count(), 0);

        let cone = engine.trace_back("any", 10);
        assert!(cone.nodes.is_empty());
    }

    #[test]
    fn test_very_deep_traversal() {
        let mut engine = LightConeEngine::new();

        // Create a very long chain with strength=1.0 to avoid probability cutoff
        let mut prev = engine.add_node(CausalNode::new("N0", "Start"));
        for i in 1..100 {
            let curr = engine.add_node(CausalNode::new(format!("N{}", i), format!("Node {}", i)));
            engine.add_link(CausalLink::new(&prev, &curr, CausalLinkType::Causes, 1.0));
            prev = curr;
        }

        // Should handle deep traversal without stack overflow
        let cone = engine.predict_outcomes("N0", 100);
        assert_eq!(cone.nodes.len(), 100);
    }

    #[test]
    fn test_wide_graph() {
        let mut engine = LightConeEngine::new();

        let root = engine.add_node(CausalNode::new("root", "Root"));

        // Add many children
        for i in 0..50 {
            let child = engine.add_node(CausalNode::new(format!("child_{}", i), format!("Child {}", i)));
            engine.add_link(CausalLink::new(&root, &child, CausalLinkType::Causes, 0.9));
        }

        let cone = engine.predict_outcomes(&root, 1);
        assert_eq!(cone.predicted_outcomes.len(), 50);
    }

    #[test]
    fn test_diamond_pattern() {
        let mut engine = LightConeEngine::new();

        //    A
        //   / \
        //  B   C
        //   \ /
        //    D
        let a = engine.add_node(CausalNode::new("A", "Top"));
        let b = engine.add_node(CausalNode::new("B", "Left"));
        let c = engine.add_node(CausalNode::new("C", "Right"));
        let d = engine.add_node(CausalNode::new("D", "Bottom"));

        engine.add_link(CausalLink::new(&a, &b, CausalLinkType::Causes, 0.9));
        engine.add_link(CausalLink::new(&a, &c, CausalLinkType::Causes, 0.9));
        engine.add_link(CausalLink::new(&b, &d, CausalLinkType::Causes, 0.9));
        engine.add_link(CausalLink::new(&c, &d, CausalLinkType::Causes, 0.9));

        // Forward from A
        let future = engine.predict_outcomes(&a, 10);
        assert_eq!(future.nodes.len(), 4);
        assert_eq!(future.predicted_outcomes.len(), 1);
        assert_eq!(future.predicted_outcomes[0].0, "D");

        // Backward from D
        let history = engine.trace_back(&d, 10);
        assert_eq!(history.nodes.len(), 4);
        assert_eq!(history.root_causes.len(), 1);
        assert_eq!(history.root_causes[0], "A");
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
        /// Property: History cone always includes the focal node.
        #[test]
        fn prop_history_cone_includes_focal(depth in 1usize..10usize) {
            let mut engine = LightConeEngine::new();
            let focal = engine.add_node(CausalNode::new("focal", "Focal node"));

            let cone = engine.trace_back(&focal, depth);

            let node_ids: Vec<_> = cone.nodes.iter().map(|(n, _)| n.id.as_str()).collect();
            prop_assert!(node_ids.contains(&"focal"));
        }

        /// Property: Future cone always includes the focal node.
        #[test]
        fn prop_future_cone_includes_focal(depth in 1usize..10usize) {
            let mut engine = LightConeEngine::new();
            let focal = engine.add_node(CausalNode::new("focal", "Focal node"));

            let cone = engine.predict_outcomes(&focal, depth);

            let node_ids: Vec<_> = cone.nodes.iter().map(|(n, _)| n.id.as_str()).collect();
            prop_assert!(node_ids.contains(&"focal"));
        }

        /// Property: Causal link strength is always in [0, 1].
        #[test]
        fn prop_link_strength_bounded(strength in -1.0f64..2.0f64) {
            let link = CausalLink::new("a", "b", CausalLinkType::Causes, strength);
            prop_assert!(link.strength >= 0.0);
            prop_assert!(link.strength <= 1.0);
        }

        /// Property: Chain strength is product of link strengths.
        #[test]
        fn prop_chain_strength_product(s1 in 0.1f64..1.0f64, s2 in 0.1f64..1.0f64) {
            let nodes = vec![
                CausalNode::new("A", "Start"),
                CausalNode::new("B", "Middle"),
                CausalNode::new("C", "End"),
            ];
            let links = vec![
                CausalLink::new("A", "B", CausalLinkType::Causes, s1),
                CausalLink::new("B", "C", CausalLinkType::Causes, s2),
            ];

            let chain = CausalChain::new(nodes, links);
            let expected = s1 * s2;

            prop_assert!((chain.strength - expected).abs() < 0.001);
        }

        /// Property: Activation level after boost is higher.
        #[test]
        fn prop_boost_increases_activation(
            initial in 0.1f64..0.5f64,
            boost in 0.1f64..0.5f64
        ) {
            let mut pattern = ActivePattern::new("test", initial);
            let before = pattern.activation_level;

            pattern.boost(boost);

            prop_assert!(pattern.activation_level >= before);
        }

        /// Property: Decay reduces activation level.
        #[test]
        fn prop_decay_reduces_activation(
            initial in 0.3f64..1.0f64,
            seconds in 1.0f64..100.0f64
        ) {
            let mut pattern = ActivePattern::new("test", initial);
            pattern.decay_rate = 0.05;

            let before = pattern.activation_level;
            pattern.apply_decay(seconds);

            prop_assert!(pattern.activation_level <= before);
        }
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_large_graph_history_traversal() {
        let mut engine = LightConeEngine::new();

        // Create a large graph with 1000 nodes
        for i in 0..1000 {
            engine.add_node(CausalNode::new(format!("node_{}", i), format!("Content {}", i)));
        }

        // Add random links
        for i in 0..999 {
            engine.add_link(CausalLink::new(
                format!("node_{}", i),
                format!("node_{}", i + 1),
                CausalLinkType::Causes,
                0.9,
            ));
        }

        let start = Instant::now();

        for _ in 0..100 {
            engine.trace_back("node_999", 50);
        }

        let duration = start.elapsed();
        assert!(
            duration.as_millis() < 1000,
            "100 history traversals should complete in < 1s, took {:?}",
            duration
        );
    }

    #[test]
    fn test_large_graph_future_traversal() {
        let mut engine = LightConeEngine::new();

        // Create a graph with branching
        engine.add_node(CausalNode::new("root", "Root"));
        for i in 0..100 {
            let node = engine.add_node(CausalNode::new(format!("child_{}", i), format!("Child {}", i)));
            engine.add_link(CausalLink::new("root", &node, CausalLinkType::Causes, 0.9));
        }

        let start = Instant::now();

        for _ in 0..100 {
            engine.predict_outcomes("root", 5);
        }

        let duration = start.elapsed();
        assert!(
            duration.as_millis() < 500,
            "100 future traversals should complete in < 500ms, took {:?}",
            duration
        );
    }

    #[test]
    fn test_cognitive_core_performance() {
        let mut core = CognitiveCore::new(1000);

        let start = Instant::now();

        // Rapid activations
        for i in 0..10000 {
            core.activate(&format!("pattern_{}", i % 500), 0.7);
        }

        let duration = start.elapsed();
        assert!(
            duration.as_millis() < 500,
            "10000 activations should complete in < 500ms, took {:?}",
            duration
        );
    }
}
