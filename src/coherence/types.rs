//! Core types for the Coherence Gate

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A belief derived from a pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Belief {
    pub id: String,
    pub pattern_id: String,
    pub statement: String,
    pub domain: String,
    pub confidence: f64,
    pub dependencies: Vec<String>,  // Belief IDs this depends on
    pub contradicts: Vec<String>,   // Belief IDs this contradicts
}

impl Belief {
    pub fn new(pattern_id: &str, statement: &str, domain: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            pattern_id: pattern_id.to_string(),
            statement: statement.to_string(),
            domain: domain.to_string(),
            confidence: 0.5,
            dependencies: Vec::new(),
            contradicts: Vec::new(),
        }
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Graph of beliefs and their relationships
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BeliefGraph {
    pub beliefs: HashMap<String, Belief>,
    pub edges: Vec<BeliefEdge>,
}

impl BeliefGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_belief(&mut self, belief: Belief) {
        self.beliefs.insert(belief.id.clone(), belief);
    }

    pub fn add_edge(&mut self, edge: BeliefEdge) {
        self.edges.push(edge);
    }

    pub fn get_belief(&self, id: &str) -> Option<&Belief> {
        self.beliefs.get(id)
    }

    pub fn beliefs_in_domain(&self, domain: &str) -> Vec<&Belief> {
        self.beliefs.values()
            .filter(|b| b.domain == domain || b.domain.starts_with(&format!("{}.", domain)))
            .collect()
    }

    pub fn get_supporting_beliefs(&self, belief_id: &str) -> Vec<&Belief> {
        self.edges.iter()
            .filter(|e| e.to == belief_id && matches!(e.relation, BeliefRelation::Supports))
            .filter_map(|e| self.beliefs.get(&e.from))
            .collect()
    }

    pub fn get_contradicting_beliefs(&self, belief_id: &str) -> Vec<&Belief> {
        self.edges.iter()
            .filter(|e| {
                (e.to == belief_id || e.from == belief_id)
                    && matches!(e.relation, BeliefRelation::Contradicts)
            })
            .filter_map(|e| {
                if e.to == belief_id {
                    self.beliefs.get(&e.from)
                } else {
                    self.beliefs.get(&e.to)
                }
            })
            .collect()
    }
}

/// Edge in the belief graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefEdge {
    pub from: String,
    pub to: String,
    pub relation: BeliefRelation,
    pub weight: f64,
}

impl BeliefEdge {
    pub fn supports(from: &str, to: &str, weight: f64) -> Self {
        Self {
            from: from.to_string(),
            to: to.to_string(),
            relation: BeliefRelation::Supports,
            weight,
        }
    }

    pub fn contradicts(from: &str, to: &str, weight: f64) -> Self {
        Self {
            from: from.to_string(),
            to: to.to_string(),
            relation: BeliefRelation::Contradicts,
            weight,
        }
    }

    pub fn depends_on(from: &str, to: &str) -> Self {
        Self {
            from: from.to_string(),
            to: to.to_string(),
            relation: BeliefRelation::DependsOn,
            weight: 1.0,
        }
    }

    pub fn refines(from: &str, to: &str) -> Self {
        Self {
            from: from.to_string(),
            to: to.to_string(),
            relation: BeliefRelation::Refines,
            weight: 1.0,
        }
    }
}

/// Relationship between beliefs
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BeliefRelation {
    Supports,      // Strengthens the other belief
    Contradicts,   // Conflicts with the other belief
    DependsOn,     // Requires the other belief
    Refines,       // More specific version of
}

impl std::fmt::Display for BeliefRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BeliefRelation::Supports => write!(f, "supports"),
            BeliefRelation::Contradicts => write!(f, "contradicts"),
            BeliefRelation::DependsOn => write!(f, "depends_on"),
            BeliefRelation::Refines => write!(f, "refines"),
        }
    }
}

/// Result of coherence check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceResult {
    pub is_coherent: bool,
    pub energy: f64,
    pub threshold: f64,
    pub conflicts: Vec<Conflict>,
    pub supporting_patterns: usize,
    pub recommendation: CoherenceAction,
}

impl CoherenceResult {
    pub fn coherent(energy: f64, threshold: f64, supporting: usize) -> Self {
        Self {
            is_coherent: true,
            energy,
            threshold,
            conflicts: Vec::new(),
            supporting_patterns: supporting,
            recommendation: CoherenceAction::Accept,
        }
    }

    pub fn has_conflicts(&self) -> bool {
        !self.conflicts.is_empty()
    }

    pub fn major_conflicts(&self) -> Vec<&Conflict> {
        self.conflicts.iter()
            .filter(|c| matches!(c.severity, ConflictSeverity::Major))
            .collect()
    }
}

/// A detected conflict between beliefs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    pub belief_a: String,
    pub belief_b: String,
    pub pattern_a_id: String,
    pub pattern_b_id: String,
    pub severity: ConflictSeverity,
    pub description: String,
    pub similarity: f64,
}

impl Conflict {
    pub fn new(
        belief_a: &Belief,
        belief_b: &Belief,
        severity: ConflictSeverity,
        description: &str,
        similarity: f64,
    ) -> Self {
        Self {
            belief_a: belief_a.id.clone(),
            belief_b: belief_b.id.clone(),
            pattern_a_id: belief_a.pattern_id.clone(),
            pattern_b_id: belief_b.pattern_id.clone(),
            severity,
            description: description.to_string(),
            similarity,
        }
    }
}

/// Severity of a conflict
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConflictSeverity {
    Minor,     // Different approaches, both valid
    Moderate,  // Conflicting recommendations
    Major,     // Logical contradiction
}

impl std::fmt::Display for ConflictSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConflictSeverity::Minor => write!(f, "minor"),
            ConflictSeverity::Moderate => write!(f, "moderate"),
            ConflictSeverity::Major => write!(f, "major"),
        }
    }
}

/// Recommended action based on coherence check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoherenceAction {
    Accept,
    AcceptWithWarning { warnings: Vec<String> },
    RequireReview { conflicts: Vec<String> },
    Reject { reason: String },
    Merge { merge_with: String },
}

impl std::fmt::Display for CoherenceAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoherenceAction::Accept => write!(f, "accept"),
            CoherenceAction::AcceptWithWarning { warnings } => {
                write!(f, "accept_with_warning ({} warnings)", warnings.len())
            }
            CoherenceAction::RequireReview { conflicts } => {
                write!(f, "require_review ({} conflicts)", conflicts.len())
            }
            CoherenceAction::Reject { reason } => {
                write!(f, "reject: {}", reason)
            }
            CoherenceAction::Merge { merge_with } => {
                write!(f, "merge_with: {}", merge_with)
            }
        }
    }
}

/// Configuration for the coherence gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceConfig {
    pub energy_threshold: f64,       // 0.4 default - minimum energy to accept
    pub similarity_threshold: f64,   // 0.85 for potential contradiction detection
    pub max_conflicts: usize,        // 3 before auto-reject
    pub check_enabled: bool,         // Enable/disable coherence checking
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            energy_threshold: 0.4,
            similarity_threshold: 0.85,
            max_conflicts: 3,
            check_enabled: true,
        }
    }
}

/// Result of storing a pattern with coherence checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StoreWithCoherenceResult {
    Stored {
        pattern_id: String,
    },
    StoredWithWarnings {
        pattern_id: String,
        warnings: Vec<String>,
    },
    PendingReview {
        pattern_id: String,
        conflicts: Vec<String>,
    },
    Rejected {
        reason: String,
    },
    MergeRequired {
        pattern_id: String,
        merge_with: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_creation() {
        let belief = Belief::new("pattern-1", "Use async for I/O operations", "rust.async")
            .with_confidence(0.9);

        assert_eq!(belief.pattern_id, "pattern-1");
        assert_eq!(belief.confidence, 0.9);
        assert_eq!(belief.domain, "rust.async");
    }

    #[test]
    fn test_belief_graph() {
        let mut graph = BeliefGraph::new();

        let belief1 = Belief::new("p1", "Use tokio for async runtime", "rust.async");
        let belief2 = Belief::new("p2", "Prefer async-std over tokio", "rust.async");

        let b1_id = belief1.id.clone();
        let b2_id = belief2.id.clone();

        graph.add_belief(belief1);
        graph.add_belief(belief2);
        graph.add_edge(BeliefEdge::contradicts(&b1_id, &b2_id, 0.8));

        assert_eq!(graph.beliefs.len(), 2);
        assert_eq!(graph.edges.len(), 1);
    }

    #[test]
    fn test_conflict_severity_ordering() {
        assert!(ConflictSeverity::Minor < ConflictSeverity::Moderate);
        assert!(ConflictSeverity::Moderate < ConflictSeverity::Major);
    }

    #[test]
    fn test_coherence_config_default() {
        let config = CoherenceConfig::default();
        assert_eq!(config.energy_threshold, 0.4);
        assert_eq!(config.similarity_threshold, 0.85);
        assert_eq!(config.max_conflicts, 3);
        assert!(config.check_enabled);
    }
}
