//! Trajectory data structures for SONA learning extension.
//!
//! This module provides trajectory recording capabilities for full reasoning path
//! capture from query to outcome. Trajectories capture:
//!
//! - Sequence of pattern retrievals
//! - Decision points with confidence scores
//! - Final outcome and reward
//! - Timing information for profiling
//!
//! # Example
//!
//! ```ignore
//! use nagual::learning::trajectory::{Trajectory, TrajectoryStep, TrajectoryBuilder};
//!
//! let trajectory = TrajectoryBuilder::new()
//!     .session_id("session-123")
//!     .add_step(TrajectoryStep::pattern_retrieval(
//!         vec!["pat_1".into(), "pat_2".into()],
//!         "search",
//!         0.85,
//!     ))
//!     .add_step(TrajectoryStep::decision(
//!         vec!["pat_1".into()],
//!         "selected_pat_1",
//!         0.92,
//!     ))
//!     .outcome(Outcome::Success, 0.9)
//!     .build();
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::reasoning_bank::pattern::PatternId;
use super::sona::Outcome;

/// Unique identifier for a trajectory.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrajectoryId(pub String);

impl TrajectoryId {
    /// Create a new random trajectory ID.
    pub fn new() -> Self {
        Self(format!("traj_{}", Uuid::new_v4()))
    }

    /// Create a trajectory ID from an existing string.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for TrajectoryId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TrajectoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for TrajectoryId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for TrajectoryId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Type of reasoning step in a trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepType {
    /// Query/search operation that retrieved patterns
    PatternRetrieval,
    /// Decision point where patterns were selected/filtered
    Decision,
    /// Pattern application to solve a problem
    PatternApplication,
    /// Learning/update operation on patterns
    Learning,
    /// Prediction made based on patterns
    Prediction,
    /// Consolidation of patterns
    Consolidation,
    /// Custom step type
    Custom,
}

impl StepType {
    /// Get string representation for database storage.
    pub fn as_str(&self) -> &'static str {
        match self {
            StepType::PatternRetrieval => "pattern_retrieval",
            StepType::Decision => "decision",
            StepType::PatternApplication => "pattern_application",
            StepType::Learning => "learning",
            StepType::Prediction => "prediction",
            StepType::Consolidation => "consolidation",
            StepType::Custom => "custom",
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "pattern_retrieval" | "retrieval" | "search" => Some(StepType::PatternRetrieval),
            "decision" | "select" | "filter" => Some(StepType::Decision),
            "pattern_application" | "application" | "apply" => Some(StepType::PatternApplication),
            "learning" | "learn" | "record" => Some(StepType::Learning),
            "prediction" | "predict" => Some(StepType::Prediction),
            "consolidation" | "consolidate" | "merge" => Some(StepType::Consolidation),
            "custom" => Some(StepType::Custom),
            _ => None,
        }
    }
}

impl std::fmt::Display for StepType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A single step in a reasoning trajectory.
///
/// Each step captures:
/// - Pattern IDs involved in this step
/// - The decision/action taken
/// - Confidence level in the decision
/// - Timing information
/// - Optional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryStep {
    /// Unique identifier for this step (auto-generated)
    pub id: String,

    /// Type of this step
    pub step_type: StepType,

    /// Pattern IDs involved in this step
    pub pattern_ids: Vec<PatternId>,

    /// Description of the decision/action taken
    pub decision: String,

    /// Confidence in this step (0.0 - 1.0)
    pub confidence: f32,

    /// When this step occurred
    pub timestamp: DateTime<Utc>,

    /// Duration of this step in milliseconds
    pub duration_ms: Option<u64>,

    /// Query that triggered this step (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,

    /// Result or output of this step
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,

    /// Additional metadata
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl TrajectoryStep {
    /// Create a new trajectory step.
    pub fn new(
        step_type: StepType,
        pattern_ids: Vec<PatternId>,
        decision: impl Into<String>,
        confidence: f32,
    ) -> Self {
        Self {
            id: format!("step_{}", Uuid::new_v4()),
            step_type,
            pattern_ids,
            decision: decision.into(),
            confidence: confidence.clamp(0.0, 1.0),
            timestamp: Utc::now(),
            duration_ms: None,
            query: None,
            result: None,
            metadata: serde_json::Value::Null,
        }
    }

    /// Create a pattern retrieval step.
    pub fn pattern_retrieval(
        pattern_ids: Vec<PatternId>,
        query: impl Into<String>,
        confidence: f32,
    ) -> Self {
        let query_str = query.into();
        let mut step = Self::new(
            StepType::PatternRetrieval,
            pattern_ids.clone(),
            format!("Retrieved {} patterns", pattern_ids.len()),
            confidence,
        );
        step.query = Some(query_str);
        step
    }

    /// Create a decision step.
    pub fn decision(
        pattern_ids: Vec<PatternId>,
        decision: impl Into<String>,
        confidence: f32,
    ) -> Self {
        Self::new(StepType::Decision, pattern_ids, decision, confidence)
    }

    /// Create a pattern application step.
    pub fn pattern_application(
        pattern_id: PatternId,
        result: impl Into<String>,
        confidence: f32,
    ) -> Self {
        let result_str = result.into();
        let mut step = Self::new(
            StepType::PatternApplication,
            vec![pattern_id],
            "Applied pattern",
            confidence,
        );
        step.result = Some(result_str);
        step
    }

    /// Create a learning step.
    pub fn learning(
        pattern_id: PatternId,
        outcome_description: impl Into<String>,
        confidence: f32,
    ) -> Self {
        Self::new(
            StepType::Learning,
            vec![pattern_id],
            outcome_description,
            confidence,
        )
    }

    /// Create a prediction step.
    pub fn prediction(
        pattern_ids: Vec<PatternId>,
        prediction: impl Into<String>,
        confidence: f32,
    ) -> Self {
        let prediction_str = prediction.into();
        let mut step = Self::new(
            StepType::Prediction,
            pattern_ids,
            "Made prediction",
            confidence,
        );
        step.result = Some(prediction_str);
        step
    }

    /// Set the duration in milliseconds.
    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    /// Set the query.
    pub fn with_query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }

    /// Set the result.
    pub fn with_result(mut self, result: impl Into<String>) -> Self {
        self.result = Some(result.into());
        self
    }

    /// Set additional metadata.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get all unique pattern IDs from this step.
    pub fn unique_pattern_ids(&self) -> Vec<&PatternId> {
        let mut seen = std::collections::HashSet::new();
        self.pattern_ids
            .iter()
            .filter(|id| seen.insert(id.as_str()))
            .collect()
    }
}

/// A complete reasoning trajectory from query to outcome.
///
/// Trajectories represent the full path of reasoning:
/// 1. Initial query/context
/// 2. Sequence of steps (retrieval, decision, application)
/// 3. Final outcome and reward
///
/// Trajectories are linked to ProfDAG nodes via `leads_to` edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    /// Unique identifier for this trajectory
    pub id: TrajectoryId,

    /// Session ID this trajectory belongs to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,

    /// Agent ID that executed this trajectory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,

    /// Original query/context that started this trajectory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,

    /// Sequence of reasoning steps
    pub steps: Vec<TrajectoryStep>,

    /// Final outcome (Success, PartialSuccess, Neutral, Failure)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<Outcome>,

    /// Total reward from this trajectory (0.0 - 1.0)
    pub total_reward: f32,

    /// When this trajectory started
    pub started_at: DateTime<Utc>,

    /// When this trajectory completed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,

    /// Total duration in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration_ms: Option<u64>,

    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,

    /// Additional metadata
    #[serde(default)]
    pub metadata: serde_json::Value,

    /// Whether this trajectory was successful (outcome is Success or PartialSuccess)
    pub success: bool,

    /// IDs of ProfDAG nodes this trajectory links to
    #[serde(default)]
    pub profdag_node_ids: Vec<String>,
}

impl Trajectory {
    /// Create a new trajectory builder.
    pub fn builder() -> TrajectoryBuilder {
        TrajectoryBuilder::new()
    }

    /// Create a new empty trajectory.
    pub fn new() -> Self {
        Self {
            id: TrajectoryId::new(),
            session_id: None,
            agent_id: None,
            query: None,
            steps: Vec::new(),
            outcome: None,
            total_reward: 0.0,
            started_at: Utc::now(),
            completed_at: None,
            total_duration_ms: None,
            tags: Vec::new(),
            metadata: serde_json::Value::Null,
            success: false,
            profdag_node_ids: Vec::new(),
        }
    }

    /// Add a step to this trajectory.
    pub fn add_step(&mut self, step: TrajectoryStep) {
        self.steps.push(step);
    }

    /// Set the outcome and calculate reward.
    pub fn set_outcome(&mut self, outcome: Outcome, reward: f32) {
        self.outcome = Some(outcome);
        self.total_reward = reward.clamp(0.0, 1.0);
        self.success = outcome.is_successful();
        self.completed_at = Some(Utc::now());
        self.total_duration_ms = self.calculate_duration_ms();
    }

    /// Mark the trajectory as complete.
    pub fn complete(&mut self) {
        if self.completed_at.is_none() {
            self.completed_at = Some(Utc::now());
            self.total_duration_ms = self.calculate_duration_ms();
        }
    }

    /// Calculate total duration in milliseconds.
    fn calculate_duration_ms(&self) -> Option<u64> {
        self.completed_at.map(|end| {
            let duration = end - self.started_at;
            duration.num_milliseconds().max(0) as u64
        })
    }

    /// Get all unique pattern IDs from all steps.
    pub fn all_pattern_ids(&self) -> Vec<PatternId> {
        let mut seen = std::collections::HashSet::new();
        self.steps
            .iter()
            .flat_map(|step| step.pattern_ids.iter())
            .filter(|id| seen.insert(id.as_str().to_string()))
            .cloned()
            .collect()
    }

    /// Get the number of steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Check if the trajectory is empty (no steps).
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Check if the trajectory is complete.
    pub fn is_complete(&self) -> bool {
        self.completed_at.is_some()
    }

    /// Get the duration as a Duration type.
    pub fn duration(&self) -> Option<Duration> {
        self.total_duration_ms.map(|ms| Duration::milliseconds(ms as i64))
    }

    /// Get the first step (if any).
    pub fn first_step(&self) -> Option<&TrajectoryStep> {
        self.steps.first()
    }

    /// Get the last step (if any).
    pub fn last_step(&self) -> Option<&TrajectoryStep> {
        self.steps.last()
    }

    /// Get steps by type.
    pub fn steps_by_type(&self, step_type: StepType) -> Vec<&TrajectoryStep> {
        self.steps
            .iter()
            .filter(|s| s.step_type == step_type)
            .collect()
    }

    /// Calculate average confidence across all steps.
    pub fn average_confidence(&self) -> f32 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.steps.iter().map(|s| s.confidence).sum();
        sum / self.steps.len() as f32
    }

    /// Add a ProfDAG node link.
    pub fn link_profdag_node(&mut self, node_id: impl Into<String>) {
        let id = node_id.into();
        if !self.profdag_node_ids.contains(&id) {
            self.profdag_node_ids.push(id);
        }
    }

    /// Convert to a compact representation for storage.
    pub fn to_compact(&self) -> CompactTrajectory {
        CompactTrajectory {
            id: self.id.clone(),
            step_count: self.steps.len(),
            pattern_ids: self.all_pattern_ids(),
            outcome: self.outcome,
            total_reward: self.total_reward,
            duration_ms: self.total_duration_ms,
            success: self.success,
        }
    }
}

impl Default for Trajectory {
    fn default() -> Self {
        Self::new()
    }
}

/// Compact trajectory representation for efficient storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactTrajectory {
    /// Trajectory ID
    pub id: TrajectoryId,
    /// Number of steps
    pub step_count: usize,
    /// All pattern IDs involved
    pub pattern_ids: Vec<PatternId>,
    /// Final outcome
    pub outcome: Option<Outcome>,
    /// Total reward
    pub total_reward: f32,
    /// Duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Success flag
    pub success: bool,
}

/// Builder for creating trajectories.
#[derive(Debug, Default)]
pub struct TrajectoryBuilder {
    id: Option<TrajectoryId>,
    session_id: Option<String>,
    agent_id: Option<String>,
    query: Option<String>,
    steps: Vec<TrajectoryStep>,
    outcome: Option<Outcome>,
    total_reward: f32,
    started_at: Option<DateTime<Utc>>,
    tags: Vec<String>,
    metadata: Option<serde_json::Value>,
}

impl TrajectoryBuilder {
    /// Create a new trajectory builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the trajectory ID.
    pub fn id(mut self, id: impl Into<TrajectoryId>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the session ID.
    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the agent ID.
    pub fn agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Set the initial query.
    pub fn query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }

    /// Add a step.
    pub fn add_step(mut self, step: TrajectoryStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Add multiple steps.
    pub fn steps(mut self, steps: Vec<TrajectoryStep>) -> Self {
        self.steps = steps;
        self
    }

    /// Set the outcome and reward.
    pub fn outcome(mut self, outcome: Outcome, reward: f32) -> Self {
        self.outcome = Some(outcome);
        self.total_reward = reward.clamp(0.0, 1.0);
        self
    }

    /// Set the start time.
    pub fn started_at(mut self, started_at: DateTime<Utc>) -> Self {
        self.started_at = Some(started_at);
        self
    }

    /// Add a tag.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set tags.
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set metadata.
    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Build the trajectory.
    pub fn build(self) -> Trajectory {
        let started_at = self.started_at.unwrap_or_else(Utc::now);
        let success = self.outcome.map(|o| o.is_successful()).unwrap_or(false);

        let mut trajectory = Trajectory {
            id: self.id.unwrap_or_else(TrajectoryId::new),
            session_id: self.session_id,
            agent_id: self.agent_id,
            query: self.query,
            steps: self.steps,
            outcome: self.outcome,
            total_reward: self.total_reward,
            started_at,
            completed_at: if self.outcome.is_some() {
                Some(Utc::now())
            } else {
                None
            },
            total_duration_ms: None,
            tags: self.tags,
            metadata: self.metadata.unwrap_or(serde_json::Value::Null),
            success,
            profdag_node_ids: Vec::new(),
        };

        if trajectory.completed_at.is_some() {
            trajectory.total_duration_ms = trajectory.calculate_duration_ms();
        }

        trajectory
    }
}

/// Statistics about trajectories.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrajectoryStats {
    /// Total trajectories recorded
    pub total_count: u64,
    /// Successful trajectories
    pub success_count: u64,
    /// Failed trajectories
    pub failure_count: u64,
    /// Average reward across all trajectories
    pub average_reward: f32,
    /// Average step count
    pub average_steps: f32,
    /// Average duration in milliseconds
    pub average_duration_ms: f64,
    /// Total patterns involved
    pub total_patterns: u64,
}

impl TrajectoryStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update stats with a new trajectory.
    pub fn record(&mut self, trajectory: &Trajectory) {
        self.total_count += 1;

        if trajectory.success {
            self.success_count += 1;
        } else if trajectory.outcome == Some(Outcome::Failure) {
            self.failure_count += 1;
        }

        // Update running averages
        let n = self.total_count as f32;
        self.average_reward = ((n - 1.0) * self.average_reward + trajectory.total_reward) / n;
        self.average_steps =
            ((n - 1.0) * self.average_steps + trajectory.step_count() as f32) / n;

        if let Some(duration) = trajectory.total_duration_ms {
            self.average_duration_ms =
                ((n - 1.0) as f64 * self.average_duration_ms + duration as f64) / n as f64;
        }

        self.total_patterns += trajectory.all_pattern_ids().len() as u64;
    }

    /// Get the success rate.
    pub fn success_rate(&self) -> f32 {
        if self.total_count == 0 {
            0.0
        } else {
            self.success_count as f32 / self.total_count as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_id_new() {
        let id1 = TrajectoryId::new();
        let id2 = TrajectoryId::new();
        assert_ne!(id1, id2);
        assert!(id1.as_str().starts_with("traj_"));
    }

    #[test]
    fn test_step_type_from_str() {
        assert_eq!(StepType::from_str("pattern_retrieval"), Some(StepType::PatternRetrieval));
        assert_eq!(StepType::from_str("search"), Some(StepType::PatternRetrieval));
        assert_eq!(StepType::from_str("decision"), Some(StepType::Decision));
        assert_eq!(StepType::from_str("unknown"), None);
    }

    #[test]
    fn test_trajectory_step_creation() {
        let step = TrajectoryStep::pattern_retrieval(
            vec![PatternId::from_string("pat_1"), PatternId::from_string("pat_2")],
            "test query",
            0.85,
        );

        assert_eq!(step.step_type, StepType::PatternRetrieval);
        assert_eq!(step.pattern_ids.len(), 2);
        assert!((step.confidence - 0.85).abs() < 0.001);
        assert_eq!(step.query, Some("test query".to_string()));
    }

    #[test]
    fn test_trajectory_builder() {
        let trajectory = Trajectory::builder()
            .session_id("session-123")
            .agent_id("agent-456")
            .query("How to optimize caching?")
            .add_step(TrajectoryStep::pattern_retrieval(
                vec![PatternId::from_string("pat_1")],
                "caching",
                0.9,
            ))
            .add_step(TrajectoryStep::decision(
                vec![PatternId::from_string("pat_1")],
                "selected_pat_1",
                0.85,
            ))
            .outcome(Outcome::Success, 0.9)
            .tag("caching")
            .build();

        assert_eq!(trajectory.session_id, Some("session-123".to_string()));
        assert_eq!(trajectory.agent_id, Some("agent-456".to_string()));
        assert_eq!(trajectory.step_count(), 2);
        assert_eq!(trajectory.outcome, Some(Outcome::Success));
        assert!(trajectory.success);
        assert!(trajectory.is_complete());
    }

    #[test]
    fn test_trajectory_all_pattern_ids() {
        let mut trajectory = Trajectory::new();
        trajectory.add_step(TrajectoryStep::pattern_retrieval(
            vec![
                PatternId::from_string("pat_1"),
                PatternId::from_string("pat_2"),
            ],
            "query",
            0.9,
        ));
        trajectory.add_step(TrajectoryStep::decision(
            vec![
                PatternId::from_string("pat_1"),
                PatternId::from_string("pat_3"),
            ],
            "decision",
            0.8,
        ));

        let all_ids = trajectory.all_pattern_ids();
        assert_eq!(all_ids.len(), 3); // pat_1, pat_2, pat_3 (unique)
    }

    #[test]
    fn test_trajectory_average_confidence() {
        let mut trajectory = Trajectory::new();
        trajectory.add_step(TrajectoryStep::new(
            StepType::Decision,
            vec![],
            "step1",
            0.8,
        ));
        trajectory.add_step(TrajectoryStep::new(
            StepType::Decision,
            vec![],
            "step2",
            0.6,
        ));

        let avg = trajectory.average_confidence();
        assert!((avg - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_trajectory_stats() {
        let mut stats = TrajectoryStats::new();

        let trajectory1 = Trajectory::builder()
            .add_step(TrajectoryStep::new(StepType::Decision, vec![], "step", 0.9))
            .outcome(Outcome::Success, 0.9)
            .build();

        let trajectory2 = Trajectory::builder()
            .add_step(TrajectoryStep::new(StepType::Decision, vec![], "step", 0.5))
            .add_step(TrajectoryStep::new(StepType::Decision, vec![], "step2", 0.5))
            .outcome(Outcome::Failure, 0.2)
            .build();

        stats.record(&trajectory1);
        stats.record(&trajectory2);

        assert_eq!(stats.total_count, 2);
        assert_eq!(stats.success_count, 1);
        assert_eq!(stats.failure_count, 1);
        assert!((stats.success_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_compact_trajectory() {
        let trajectory = Trajectory::builder()
            .add_step(TrajectoryStep::pattern_retrieval(
                vec![PatternId::from_string("pat_1")],
                "query",
                0.9,
            ))
            .outcome(Outcome::Success, 0.85)
            .build();

        let compact = trajectory.to_compact();
        assert_eq!(compact.step_count, 1);
        assert_eq!(compact.pattern_ids.len(), 1);
        assert_eq!(compact.outcome, Some(Outcome::Success));
        assert!(compact.success);
    }

    #[test]
    fn test_trajectory_serialization() {
        let trajectory = Trajectory::builder()
            .session_id("test-session")
            .add_step(TrajectoryStep::decision(
                vec![PatternId::from_string("pat_1")],
                "selected",
                0.9,
            ))
            .outcome(Outcome::Success, 0.9)
            .build();

        let json = serde_json::to_string(&trajectory).unwrap();
        let deserialized: Trajectory = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.session_id, trajectory.session_id);
        assert_eq!(deserialized.step_count(), trajectory.step_count());
        assert_eq!(deserialized.outcome, trajectory.outcome);
    }
}

// ============================================================================
// Trajectory Storage
// ============================================================================

/// SQL for creating the trajectories table in SQLite.
pub const SQLITE_TRAJECTORIES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS trajectories (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    agent_id TEXT,
    query TEXT,
    steps TEXT NOT NULL,
    outcome TEXT,
    total_reward REAL NOT NULL DEFAULT 0.0,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    total_duration_ms INTEGER,
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    success INTEGER NOT NULL DEFAULT 0,
    profdag_node_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trajectories_session ON trajectories(session_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_agent ON trajectories(agent_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_outcome ON trajectories(outcome);
CREATE INDEX IF NOT EXISTS idx_trajectories_success ON trajectories(success);
CREATE INDEX IF NOT EXISTS idx_trajectories_started ON trajectories(started_at);
CREATE INDEX IF NOT EXISTS idx_trajectories_reward ON trajectories(total_reward);
"#;

/// SQL for creating the trajectory_steps table (normalized storage for steps).
pub const SQLITE_TRAJECTORY_STEPS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS trajectory_steps (
    id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    step_type TEXT NOT NULL,
    pattern_ids TEXT NOT NULL DEFAULT '[]',
    decision TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    timestamp TEXT NOT NULL,
    duration_ms INTEGER,
    query TEXT,
    result TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    UNIQUE(trajectory_id, sequence),
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_trajectory_steps_trajectory ON trajectory_steps(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_trajectory_steps_type ON trajectory_steps(step_type);
"#;

/// SQL for creating the trajectory_pattern_links table (join table for pattern references).
pub const SQLITE_TRAJECTORY_PATTERN_LINKS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS trajectory_pattern_links (
    trajectory_id TEXT NOT NULL,
    pattern_id TEXT NOT NULL,
    step_sequence INTEGER,
    link_type TEXT NOT NULL DEFAULT 'used',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trajectory_id, pattern_id),
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tpl_trajectory ON trajectory_pattern_links(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_tpl_pattern ON trajectory_pattern_links(pattern_id);
"#;

/// Configuration for trajectory storage.
#[derive(Debug, Clone)]
pub struct TrajectoryStorageConfig {
    /// Whether to store steps in a normalized separate table.
    pub normalize_steps: bool,
    /// Whether to create pattern links for faster lookups.
    pub create_pattern_links: bool,
    /// Maximum steps to store per trajectory (0 = unlimited).
    pub max_steps_per_trajectory: usize,
    /// Whether to auto-prune old trajectories.
    pub auto_prune: bool,
    /// Maximum age in days for trajectory retention (0 = forever).
    pub retention_days: u32,
}

impl Default for TrajectoryStorageConfig {
    fn default() -> Self {
        Self {
            normalize_steps: false, // Store steps as JSON in main table by default
            create_pattern_links: true,
            max_steps_per_trajectory: 100,
            auto_prune: false,
            retention_days: 90,
        }
    }
}

/// Filter options for querying trajectories.
#[derive(Debug, Clone, Default)]
pub struct TrajectoryFilter {
    /// Filter by session ID.
    pub session_id: Option<String>,
    /// Filter by agent ID.
    pub agent_id: Option<String>,
    /// Filter by outcome.
    pub outcome: Option<Outcome>,
    /// Only successful trajectories.
    pub success_only: bool,
    /// Only failed trajectories.
    pub failure_only: bool,
    /// Minimum reward threshold.
    pub min_reward: Option<f32>,
    /// Maximum reward threshold.
    pub max_reward: Option<f32>,
    /// Filter by pattern ID (trajectory must include this pattern).
    pub pattern_id: Option<String>,
    /// Filter by tag.
    pub tag: Option<String>,
    /// Started after this timestamp.
    pub started_after: Option<DateTime<Utc>>,
    /// Started before this timestamp.
    pub started_before: Option<DateTime<Utc>>,
    /// Maximum number of results.
    pub limit: Option<usize>,
    /// Offset for pagination.
    pub offset: Option<usize>,
    /// Order by field.
    pub order_by: TrajectoryOrderBy,
    /// Descending order.
    pub descending: bool,
}

/// Order by options for trajectory queries.
#[derive(Debug, Clone, Copy, Default)]
pub enum TrajectoryOrderBy {
    #[default]
    StartedAt,
    CompletedAt,
    TotalReward,
    StepCount,
}

impl TrajectoryOrderBy {
    fn as_sql(&self) -> &'static str {
        match self {
            TrajectoryOrderBy::StartedAt => "started_at",
            TrajectoryOrderBy::CompletedAt => "completed_at",
            TrajectoryOrderBy::TotalReward => "total_reward",
            TrajectoryOrderBy::StepCount => "json_array_length(steps)",
        }
    }
}

/// Trajectory storage interface.
///
/// Provides persistence for trajectories in SQLite with support for:
/// - Storing complete trajectories with all steps
/// - Querying by session, agent, pattern, or outcome
/// - Aggregating statistics across trajectories
/// - Linking trajectories to patterns for efficient lookups
pub struct TrajectoryStorage {
    config: TrajectoryStorageConfig,
}

impl TrajectoryStorage {
    /// Create a new trajectory storage with default configuration.
    pub fn new() -> Self {
        Self {
            config: TrajectoryStorageConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: TrajectoryStorageConfig) -> Self {
        Self { config }
    }

    /// Initialize the database schema.
    pub fn init_schema(&self, conn: &rusqlite::Connection) -> Result<(), rusqlite::Error> {
        conn.execute_batch(SQLITE_TRAJECTORIES_TABLE)?;

        if self.config.normalize_steps {
            conn.execute_batch(SQLITE_TRAJECTORY_STEPS_TABLE)?;
        }

        if self.config.create_pattern_links {
            conn.execute_batch(SQLITE_TRAJECTORY_PATTERN_LINKS_TABLE)?;
        }

        Ok(())
    }

    /// Store a trajectory in the database.
    pub fn store(
        &self,
        conn: &rusqlite::Connection,
        trajectory: &Trajectory,
    ) -> Result<(), rusqlite::Error> {
        let steps_json = serde_json::to_string(&trajectory.steps).unwrap_or_else(|_| "[]".into());
        let tags_json = serde_json::to_string(&trajectory.tags).unwrap_or_else(|_| "[]".into());
        let metadata_json = serde_json::to_string(&trajectory.metadata).unwrap_or_else(|_| "{}".into());
        let profdag_json = serde_json::to_string(&trajectory.profdag_node_ids).unwrap_or_else(|_| "[]".into());
        let outcome_str = trajectory.outcome.as_ref().map(|o| o.as_str());

        conn.execute(
            r#"
            INSERT OR REPLACE INTO trajectories (
                id, session_id, agent_id, query, steps, outcome,
                total_reward, started_at, completed_at, total_duration_ms,
                tags, metadata, success, profdag_node_ids
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
            "#,
            rusqlite::params![
                trajectory.id.as_str(),
                trajectory.session_id,
                trajectory.agent_id,
                trajectory.query,
                steps_json,
                outcome_str,
                trajectory.total_reward,
                trajectory.started_at.to_rfc3339(),
                trajectory.completed_at.map(|dt| dt.to_rfc3339()),
                trajectory.total_duration_ms.map(|d| d as i64),
                tags_json,
                metadata_json,
                trajectory.success as i32,
                profdag_json,
            ],
        )?;

        // Create pattern links if enabled
        if self.config.create_pattern_links {
            self.create_pattern_links(conn, trajectory)?;
        }

        Ok(())
    }

    /// Create pattern links for a trajectory.
    fn create_pattern_links(
        &self,
        conn: &rusqlite::Connection,
        trajectory: &Trajectory,
    ) -> Result<(), rusqlite::Error> {
        // Delete existing links for this trajectory
        conn.execute(
            "DELETE FROM trajectory_pattern_links WHERE trajectory_id = ?",
            [trajectory.id.as_str()],
        )?;

        // Insert new links
        let mut stmt = conn.prepare(
            r#"
            INSERT OR IGNORE INTO trajectory_pattern_links
            (trajectory_id, pattern_id, step_sequence, link_type)
            VALUES (?, ?, ?, 'used')
            "#,
        )?;

        for (seq, step) in trajectory.steps.iter().enumerate() {
            for pattern_id in &step.pattern_ids {
                stmt.execute(rusqlite::params![
                    trajectory.id.as_str(),
                    pattern_id.as_str(),
                    seq as i32,
                ])?;
            }
        }

        Ok(())
    }

    /// Get a trajectory by ID.
    pub fn get(
        &self,
        conn: &rusqlite::Connection,
        id: &TrajectoryId,
    ) -> Result<Option<Trajectory>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            r#"
            SELECT id, session_id, agent_id, query, steps, outcome,
                   total_reward, started_at, completed_at, total_duration_ms,
                   tags, metadata, success, profdag_node_ids
            FROM trajectories WHERE id = ?
            "#,
        )?;

        let result = stmt.query_row([id.as_str()], Self::row_to_trajectory);

        match result {
            Ok(traj) => Ok(Some(traj)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Delete a trajectory by ID.
    pub fn delete(
        &self,
        conn: &rusqlite::Connection,
        id: &TrajectoryId,
    ) -> Result<bool, rusqlite::Error> {
        let rows = conn.execute("DELETE FROM trajectories WHERE id = ?", [id.as_str()])?;
        Ok(rows > 0)
    }

    /// Query trajectories with filters.
    pub fn query(
        &self,
        conn: &rusqlite::Connection,
        filter: &TrajectoryFilter,
    ) -> Result<Vec<Trajectory>, rusqlite::Error> {
        let mut sql = String::from(
            r#"
            SELECT t.id, t.session_id, t.agent_id, t.query, t.steps, t.outcome,
                   t.total_reward, t.started_at, t.completed_at, t.total_duration_ms,
                   t.tags, t.metadata, t.success, t.profdag_node_ids
            FROM trajectories t
            "#,
        );

        let mut conditions: Vec<String> = Vec::new();
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        // Handle pattern_id filter with join
        if filter.pattern_id.is_some() && self.config.create_pattern_links {
            sql.push_str(" INNER JOIN trajectory_pattern_links tpl ON t.id = tpl.trajectory_id ");
        }

        if let Some(ref session_id) = filter.session_id {
            conditions.push(format!("t.session_id = ?{}", params.len() + 1));
            params.push(Box::new(session_id.clone()));
        }

        if let Some(ref agent_id) = filter.agent_id {
            conditions.push(format!("t.agent_id = ?{}", params.len() + 1));
            params.push(Box::new(agent_id.clone()));
        }

        if let Some(ref outcome) = filter.outcome {
            conditions.push(format!("t.outcome = ?{}", params.len() + 1));
            params.push(Box::new(outcome.as_str().to_string()));
        }

        if filter.success_only {
            conditions.push("t.success = 1".to_string());
        }

        if filter.failure_only {
            conditions.push("t.success = 0".to_string());
        }

        if let Some(min_reward) = filter.min_reward {
            conditions.push(format!("t.total_reward >= ?{}", params.len() + 1));
            params.push(Box::new(min_reward as f64));
        }

        if let Some(max_reward) = filter.max_reward {
            conditions.push(format!("t.total_reward <= ?{}", params.len() + 1));
            params.push(Box::new(max_reward as f64));
        }

        if let Some(ref pattern_id) = filter.pattern_id {
            if self.config.create_pattern_links {
                conditions.push(format!("tpl.pattern_id = ?{}", params.len() + 1));
            } else {
                conditions.push(format!("t.steps LIKE ?{}", params.len() + 1));
                params.push(Box::new(format!("%{}%", pattern_id)));
                // Note: Pattern ID already added below, so we skip here
                params.pop();
            }
            params.push(Box::new(pattern_id.clone()));
        }

        if let Some(ref tag) = filter.tag {
            conditions.push(format!("t.tags LIKE ?{}", params.len() + 1));
            params.push(Box::new(format!("%\"{}\"", tag)));
        }

        if let Some(ref started_after) = filter.started_after {
            conditions.push(format!("t.started_at >= ?{}", params.len() + 1));
            params.push(Box::new(started_after.to_rfc3339()));
        }

        if let Some(ref started_before) = filter.started_before {
            conditions.push(format!("t.started_at <= ?{}", params.len() + 1));
            params.push(Box::new(started_before.to_rfc3339()));
        }

        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }

        // Add DISTINCT if using pattern links to avoid duplicates
        if filter.pattern_id.is_some() && self.config.create_pattern_links {
            sql = sql.replace("SELECT t.", "SELECT DISTINCT t.");
        }

        // Order by
        sql.push_str(&format!(
            " ORDER BY {} {}",
            filter.order_by.as_sql(),
            if filter.descending { "DESC" } else { "ASC" }
        ));

        // Limit and offset
        if let Some(limit) = filter.limit {
            sql.push_str(&format!(" LIMIT {}", limit));
        }
        if let Some(offset) = filter.offset {
            sql.push_str(&format!(" OFFSET {}", offset));
        }

        let mut stmt = conn.prepare(&sql)?;

        // Build parameter references slice
        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();

        let rows = stmt.query_map(param_refs.as_slice(), Self::row_to_trajectory)?;

        let mut trajectories = Vec::new();
        for row in rows {
            trajectories.push(row?);
        }

        Ok(trajectories)
    }

    /// Get trajectories by session ID.
    pub fn get_by_session(
        &self,
        conn: &rusqlite::Connection,
        session_id: &str,
    ) -> Result<Vec<Trajectory>, rusqlite::Error> {
        self.query(conn, &TrajectoryFilter {
            session_id: Some(session_id.to_string()),
            ..Default::default()
        })
    }

    /// Get trajectories by pattern ID.
    pub fn get_by_pattern(
        &self,
        conn: &rusqlite::Connection,
        pattern_id: &str,
    ) -> Result<Vec<Trajectory>, rusqlite::Error> {
        self.query(conn, &TrajectoryFilter {
            pattern_id: Some(pattern_id.to_string()),
            ..Default::default()
        })
    }

    /// Get successful trajectories.
    pub fn get_successful(
        &self,
        conn: &rusqlite::Connection,
        limit: Option<usize>,
    ) -> Result<Vec<Trajectory>, rusqlite::Error> {
        self.query(conn, &TrajectoryFilter {
            success_only: true,
            limit,
            descending: true,
            ..Default::default()
        })
    }

    /// Get failed trajectories.
    pub fn get_failed(
        &self,
        conn: &rusqlite::Connection,
        limit: Option<usize>,
    ) -> Result<Vec<Trajectory>, rusqlite::Error> {
        self.query(conn, &TrajectoryFilter {
            failure_only: true,
            limit,
            descending: true,
            ..Default::default()
        })
    }

    /// Get high-reward trajectories above a threshold.
    pub fn get_high_reward(
        &self,
        conn: &rusqlite::Connection,
        min_reward: f32,
        limit: Option<usize>,
    ) -> Result<Vec<Trajectory>, rusqlite::Error> {
        self.query(conn, &TrajectoryFilter {
            min_reward: Some(min_reward),
            limit,
            order_by: TrajectoryOrderBy::TotalReward,
            descending: true,
            ..Default::default()
        })
    }

    /// Count trajectories matching a filter.
    pub fn count(
        &self,
        conn: &rusqlite::Connection,
        filter: &TrajectoryFilter,
    ) -> Result<u64, rusqlite::Error> {
        let mut sql = String::from("SELECT COUNT(*) FROM trajectories t");
        let mut conditions: Vec<String> = Vec::new();
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(ref session_id) = filter.session_id {
            conditions.push(format!("t.session_id = ?{}", params.len() + 1));
            params.push(Box::new(session_id.clone()));
        }

        if filter.success_only {
            conditions.push("t.success = 1".to_string());
        }

        if filter.failure_only {
            conditions.push("t.success = 0".to_string());
        }

        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }

        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let count: i64 = conn.query_row(&sql, param_refs.as_slice(), |row| row.get(0))?;
        Ok(count as u64)
    }

    /// Get aggregate statistics for trajectories.
    pub fn get_stats(
        &self,
        conn: &rusqlite::Connection,
    ) -> Result<TrajectoryStats, rusqlite::Error> {
        let row = conn.query_row(
            r#"
            SELECT
                COUNT(*) as total,
                COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0) as success_count,
                COALESCE(SUM(CASE WHEN outcome = 'failure' THEN 1 ELSE 0 END), 0) as failure_count,
                AVG(total_reward) as avg_reward,
                AVG(json_array_length(steps)) as avg_steps,
                AVG(total_duration_ms) as avg_duration
            FROM trajectories
            "#,
            [],
            |row| {
                Ok(TrajectoryStats {
                    total_count: row.get::<_, i64>(0)? as u64,
                    success_count: row.get::<_, i64>(1)? as u64,
                    failure_count: row.get::<_, i64>(2)? as u64,
                    average_reward: row.get::<_, Option<f64>>(3)?.unwrap_or(0.0) as f32,
                    average_steps: row.get::<_, Option<f64>>(4)?.unwrap_or(0.0) as f32,
                    average_duration_ms: row.get::<_, Option<f64>>(5)?.unwrap_or(0.0),
                    total_patterns: 0, // Calculated separately if needed
                })
            },
        )?;

        Ok(row)
    }

    /// Get statistics for a specific session.
    pub fn get_session_stats(
        &self,
        conn: &rusqlite::Connection,
        session_id: &str,
    ) -> Result<TrajectoryStats, rusqlite::Error> {
        let row = conn.query_row(
            r#"
            SELECT
                COUNT(*) as total,
                COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0) as success_count,
                COALESCE(SUM(CASE WHEN outcome = 'failure' THEN 1 ELSE 0 END), 0) as failure_count,
                AVG(total_reward) as avg_reward,
                AVG(json_array_length(steps)) as avg_steps,
                AVG(total_duration_ms) as avg_duration
            FROM trajectories
            WHERE session_id = ?
            "#,
            [session_id],
            |row| {
                Ok(TrajectoryStats {
                    total_count: row.get::<_, i64>(0)? as u64,
                    success_count: row.get::<_, i64>(1)? as u64,
                    failure_count: row.get::<_, i64>(2)? as u64,
                    average_reward: row.get::<_, Option<f64>>(3)?.unwrap_or(0.0) as f32,
                    average_steps: row.get::<_, Option<f64>>(4)?.unwrap_or(0.0) as f32,
                    average_duration_ms: row.get::<_, Option<f64>>(5)?.unwrap_or(0.0),
                    total_patterns: 0,
                })
            },
        )?;

        Ok(row)
    }

    /// Prune old trajectories based on retention policy.
    pub fn prune_old(
        &self,
        conn: &rusqlite::Connection,
        days: u32,
    ) -> Result<usize, rusqlite::Error> {
        let cutoff = Utc::now() - chrono::Duration::days(days as i64);
        let rows = conn.execute(
            "DELETE FROM trajectories WHERE started_at < ?",
            [cutoff.to_rfc3339()],
        )?;
        Ok(rows)
    }

    /// Get the most recent trajectories.
    pub fn get_recent(
        &self,
        conn: &rusqlite::Connection,
        limit: usize,
    ) -> Result<Vec<Trajectory>, rusqlite::Error> {
        self.query(conn, &TrajectoryFilter {
            limit: Some(limit),
            order_by: TrajectoryOrderBy::StartedAt,
            descending: true,
            ..Default::default()
        })
    }

    /// Helper to convert a database row to a Trajectory.
    fn row_to_trajectory(row: &rusqlite::Row<'_>) -> Result<Trajectory, rusqlite::Error> {
        let id_str: String = row.get(0)?;
        let session_id: Option<String> = row.get(1)?;
        let agent_id: Option<String> = row.get(2)?;
        let query: Option<String> = row.get(3)?;
        let steps_json: String = row.get(4)?;
        let outcome_str: Option<String> = row.get(5)?;
        let total_reward: f64 = row.get(6)?;
        let started_at_str: String = row.get(7)?;
        let completed_at_str: Option<String> = row.get(8)?;
        let duration_ms: Option<i64> = row.get(9)?;
        let tags_json: String = row.get(10)?;
        let metadata_json: String = row.get(11)?;
        let success: i32 = row.get(12)?;
        let profdag_json: String = row.get(13)?;

        let steps: Vec<TrajectoryStep> = serde_json::from_str(&steps_json).unwrap_or_default();
        let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
        let metadata: serde_json::Value = serde_json::from_str(&metadata_json)
            .unwrap_or(serde_json::Value::Null);
        let profdag_node_ids: Vec<String> = serde_json::from_str(&profdag_json).unwrap_or_default();

        let outcome = outcome_str.and_then(|s| Outcome::from_str(&s));
        let started_at = DateTime::parse_from_rfc3339(&started_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());
        let completed_at = completed_at_str.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&Utc))
                .ok()
        });

        Ok(Trajectory {
            id: TrajectoryId(id_str),
            session_id,
            agent_id,
            query,
            steps,
            outcome,
            total_reward: total_reward as f32,
            started_at,
            completed_at,
            total_duration_ms: duration_ms.map(|d| d as u64),
            tags,
            metadata,
            success: success != 0,
            profdag_node_ids,
        })
    }
}

impl Default for TrajectoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Outcome Extension
// ============================================================================

impl Outcome {
    /// Convert outcome to string for database storage.
    pub fn as_str(&self) -> &'static str {
        match self {
            Outcome::Success => "success",
            Outcome::PartialSuccess => "partial_success",
            Outcome::Neutral => "neutral",
            Outcome::Failure => "failure",
        }
    }

    /// Parse outcome from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "success" => Some(Outcome::Success),
            "partial_success" | "partialsuccess" => Some(Outcome::PartialSuccess),
            "neutral" => Some(Outcome::Neutral),
            "failure" => Some(Outcome::Failure),
            _ => None,
        }
    }
}

// ============================================================================
// Additional Tests for Storage
// ============================================================================

#[cfg(test)]
mod storage_tests {
    use super::*;
    use rusqlite::Connection;

    fn setup_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        let storage = TrajectoryStorage::new();
        storage.init_schema(&conn).unwrap();
        conn
    }

    #[test]
    fn test_init_schema() {
        let conn = setup_test_db();

        // Verify tables exist
        let count: i32 = conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='trajectories'",
            [],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_store_and_get_trajectory() {
        let conn = setup_test_db();
        let storage = TrajectoryStorage::new();

        let trajectory = Trajectory::builder()
            .session_id("test-session")
            .agent_id("test-agent")
            .query("How do I optimize caching?")
            .add_step(TrajectoryStep::pattern_retrieval(
                vec![PatternId::from_string("pat_1")],
                "caching",
                0.9,
            ))
            .outcome(Outcome::Success, 0.85)
            .tag("cache")
            .build();

        storage.store(&conn, &trajectory).unwrap();

        let retrieved = storage.get(&conn, &trajectory.id).unwrap();
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.session_id, Some("test-session".to_string()));
        assert_eq!(retrieved.agent_id, Some("test-agent".to_string()));
        assert_eq!(retrieved.step_count(), 1);
        assert_eq!(retrieved.outcome, Some(Outcome::Success));
        assert!(retrieved.success);
    }

    #[test]
    fn test_query_by_session() {
        let conn = setup_test_db();
        let storage = TrajectoryStorage::new();

        // Create multiple trajectories
        for i in 0..3 {
            let trajectory = Trajectory::builder()
                .session_id("session-1")
                .query(format!("Query {}", i))
                .outcome(Outcome::Success, 0.8)
                .build();
            storage.store(&conn, &trajectory).unwrap();
        }

        for i in 0..2 {
            let trajectory = Trajectory::builder()
                .session_id("session-2")
                .query(format!("Other query {}", i))
                .outcome(Outcome::Failure, 0.2)
                .build();
            storage.store(&conn, &trajectory).unwrap();
        }

        let session1_trajs = storage.get_by_session(&conn, "session-1").unwrap();
        assert_eq!(session1_trajs.len(), 3);

        let session2_trajs = storage.get_by_session(&conn, "session-2").unwrap();
        assert_eq!(session2_trajs.len(), 2);
    }

    #[test]
    fn test_query_successful_trajectories() {
        let conn = setup_test_db();
        let storage = TrajectoryStorage::new();

        // Create mixed trajectories
        let success_traj = Trajectory::builder()
            .outcome(Outcome::Success, 0.9)
            .build();
        storage.store(&conn, &success_traj).unwrap();

        let fail_traj = Trajectory::builder()
            .outcome(Outcome::Failure, 0.1)
            .build();
        storage.store(&conn, &fail_traj).unwrap();

        let successful = storage.get_successful(&conn, None).unwrap();
        assert_eq!(successful.len(), 1);
        assert!(successful[0].success);

        let failed = storage.get_failed(&conn, None).unwrap();
        assert_eq!(failed.len(), 1);
        assert!(!failed[0].success);
    }

    #[test]
    fn test_get_stats() {
        let conn = setup_test_db();
        let storage = TrajectoryStorage::new();

        // Create test trajectories
        for reward in [0.9, 0.8, 0.3] {
            let outcome = if reward > 0.5 { Outcome::Success } else { Outcome::Failure };
            let trajectory = Trajectory::builder()
                .outcome(outcome, reward)
                .build();
            storage.store(&conn, &trajectory).unwrap();
        }

        let stats = storage.get_stats(&conn).unwrap();
        assert_eq!(stats.total_count, 3);
        assert_eq!(stats.success_count, 2);
        assert_eq!(stats.failure_count, 1);
        assert!((stats.average_reward - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_delete_trajectory() {
        let conn = setup_test_db();
        let storage = TrajectoryStorage::new();

        let trajectory = Trajectory::builder()
            .outcome(Outcome::Success, 0.9)
            .build();

        storage.store(&conn, &trajectory).unwrap();
        assert!(storage.get(&conn, &trajectory.id).unwrap().is_some());

        let deleted = storage.delete(&conn, &trajectory.id).unwrap();
        assert!(deleted);
        assert!(storage.get(&conn, &trajectory.id).unwrap().is_none());
    }

    #[test]
    fn test_pattern_links() {
        let conn = setup_test_db();
        let storage = TrajectoryStorage::with_config(TrajectoryStorageConfig {
            create_pattern_links: true,
            ..Default::default()
        });
        storage.init_schema(&conn).unwrap();

        let trajectory = Trajectory::builder()
            .add_step(TrajectoryStep::pattern_retrieval(
                vec![
                    PatternId::from_string("pat_1"),
                    PatternId::from_string("pat_2"),
                ],
                "query",
                0.9,
            ))
            .add_step(TrajectoryStep::decision(
                vec![PatternId::from_string("pat_1")],
                "selected",
                0.85,
            ))
            .outcome(Outcome::Success, 0.9)
            .build();

        storage.store(&conn, &trajectory).unwrap();

        // Check pattern links were created
        let link_count: i32 = conn.query_row(
            "SELECT COUNT(*) FROM trajectory_pattern_links WHERE trajectory_id = ?",
            [trajectory.id.as_str()],
            |row| row.get(0),
        ).unwrap();

        // Should have 3 links: pat_1 (step 0), pat_2 (step 0), pat_1 (step 1) - but pat_1 is unique
        // Actually INSERT OR IGNORE means only 2 unique pattern IDs
        assert_eq!(link_count, 2);
    }

    #[test]
    fn test_high_reward_query() {
        let conn = setup_test_db();
        let storage = TrajectoryStorage::new();

        for reward in [0.2, 0.5, 0.7, 0.9, 0.95] {
            let trajectory = Trajectory::builder()
                .outcome(Outcome::Success, reward)
                .build();
            storage.store(&conn, &trajectory).unwrap();
        }

        let high_reward = storage.get_high_reward(&conn, 0.8, None).unwrap();
        assert_eq!(high_reward.len(), 2);
        assert!(high_reward[0].total_reward >= 0.9);
    }

    #[test]
    fn test_count_trajectories() {
        let conn = setup_test_db();
        let storage = TrajectoryStorage::new();

        for i in 0..5 {
            let trajectory = Trajectory::builder()
                .session_id(if i < 3 { "session-a" } else { "session-b" })
                .outcome(Outcome::Success, 0.8)
                .build();
            storage.store(&conn, &trajectory).unwrap();
        }

        let total = storage.count(&conn, &TrajectoryFilter::default()).unwrap();
        assert_eq!(total, 5);

        let session_a_count = storage.count(&conn, &TrajectoryFilter {
            session_id: Some("session-a".to_string()),
            ..Default::default()
        }).unwrap();
        assert_eq!(session_a_count, 3);
    }
}

// ============================================================================
// Trajectory Analysis Engine (Week 2 Workstream B)
// ============================================================================

/// A chain of patterns that appear together in trajectories.
///
/// Pattern chains capture sequences of patterns that consistently lead to
/// specific outcomes (success or failure). They are used to identify
/// "compounding correctness" patterns and risky patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternChain {
    /// Ordered sequence of pattern IDs in this chain
    pub patterns: Vec<String>,
    /// Success rate when this chain is used (0.0 - 1.0)
    pub success_rate: f32,
    /// Number of times this chain has been observed
    pub occurrence_count: u32,
    /// Average reward across all trajectories containing this chain
    pub avg_reward: f32,
    /// Total reward sum for calculating averages
    pub total_reward: f32,
}

impl PatternChain {
    /// Create a new pattern chain.
    pub fn new(patterns: Vec<String>) -> Self {
        Self {
            patterns,
            success_rate: 0.0,
            occurrence_count: 0,
            avg_reward: 0.0,
            total_reward: 0.0,
        }
    }

    /// Record an observation of this chain.
    pub fn record_observation(&mut self, success: bool, reward: f32) {
        let old_count = self.occurrence_count as f32;
        self.occurrence_count += 1;
        let new_count = self.occurrence_count as f32;

        // Update success rate as running average
        let old_success_rate = self.success_rate;
        self.success_rate = (old_success_rate * old_count + if success { 1.0 } else { 0.0 }) / new_count;

        // Update average reward
        self.total_reward += reward;
        self.avg_reward = self.total_reward / new_count;
    }

    /// Get the chain length.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Get a key for this chain (for hashmap storage).
    pub fn key(&self) -> String {
        self.patterns.join("->")
    }
}

/// A transition between two patterns with probability and success metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTransition {
    /// Source pattern ID
    pub from_pattern: String,
    /// Target pattern ID
    pub to_pattern: String,
    /// Probability of this transition P(to | from)
    pub probability: f32,
    /// Success rate when this transition occurs
    pub success_rate: f32,
    /// Number of times this transition has been observed
    pub count: u32,
    /// Average reward when this transition is made
    pub avg_reward: f32,
}

impl PatternTransition {
    /// Create a new pattern transition.
    pub fn new(from: String, to: String) -> Self {
        Self {
            from_pattern: from,
            to_pattern: to,
            probability: 0.0,
            success_rate: 0.0,
            count: 0,
            avg_reward: 0.0,
        }
    }
}

/// Complete analysis of pattern chains and transitions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChainAnalysis {
    /// Chains that consistently lead to success
    pub success_chains: Vec<PatternChain>,
    /// Chains that consistently lead to failure
    pub failure_chains: Vec<PatternChain>,
    /// Patterns that compound success (using them increases subsequent success rate)
    pub high_value_patterns: Vec<String>,
    /// Patterns that often lead to failures
    pub risky_patterns: Vec<String>,
    /// Overall baseline success rate
    pub baseline_success_rate: f32,
    /// Total trajectories analyzed
    pub trajectories_analyzed: u64,
}

/// Configuration for trajectory analysis.
#[derive(Debug, Clone)]
pub struct TrajectoryAnalysisConfig {
    /// Minimum chain length to consider
    pub min_chain_length: usize,
    /// Maximum chain length to consider
    pub max_chain_length: usize,
    /// Minimum occurrences for a chain to be significant
    pub min_occurrences: u32,
    /// Minimum success rate for success chains
    pub min_success_rate: f32,
    /// Maximum success rate for failure chains (below this is failure)
    pub max_failure_rate: f32,
    /// Multiplier for compounding pattern threshold (e.g., 1.2 = 20% above baseline)
    pub compounding_threshold: f32,
    /// Multiplier for risky pattern threshold (e.g., 0.8 = 20% below baseline)
    pub risky_threshold: f32,
}

impl Default for TrajectoryAnalysisConfig {
    fn default() -> Self {
        Self {
            min_chain_length: 2,
            max_chain_length: 5,
            min_occurrences: 3,
            min_success_rate: 0.7,
            max_failure_rate: 0.3,
            compounding_threshold: 1.2,
            risky_threshold: 0.8,
        }
    }
}

/// Analyzer for finding patterns in trajectory data.
///
/// The analyzer identifies:
/// - Success chains: sequences of patterns that consistently lead to success
/// - Failure chains: sequences that consistently lead to failure
/// - Compounding patterns: patterns that increase success probability of following patterns
/// - Risky patterns: patterns that often precede failures
pub struct TrajectoryAnalyzer<'a> {
    storage: &'a TrajectoryStorage,
    config: TrajectoryAnalysisConfig,
}

impl<'a> TrajectoryAnalyzer<'a> {
    /// Create a new trajectory analyzer with default configuration.
    pub fn new(storage: &'a TrajectoryStorage) -> Self {
        Self {
            storage,
            config: TrajectoryAnalysisConfig::default(),
        }
    }

    /// Create an analyzer with custom configuration.
    pub fn with_config(storage: &'a TrajectoryStorage, config: TrajectoryAnalysisConfig) -> Self {
        Self { storage, config }
    }

    /// Find sequences of patterns that consistently lead to success.
    ///
    /// # Arguments
    /// * `conn` - Database connection
    /// * `min_length` - Minimum chain length (overrides config if provided)
    /// * `min_occurrences` - Minimum occurrences (overrides config if provided)
    ///
    /// # Returns
    /// Vector of pattern chains with high success rates
    pub fn find_success_chains(
        &self,
        conn: &rusqlite::Connection,
        min_length: Option<usize>,
        min_occurrences: Option<u32>,
    ) -> Result<Vec<PatternChain>, rusqlite::Error> {
        let min_len = min_length.unwrap_or(self.config.min_chain_length);
        let min_occ = min_occurrences.unwrap_or(self.config.min_occurrences);

        // Get successful trajectories
        let trajectories = self.storage.get_successful(conn, None)?;

        // Extract and count chains
        let mut chains = self.extract_chains(&trajectories, min_len);

        // Filter by occurrence count and success rate
        chains.retain(|chain| {
            chain.occurrence_count >= min_occ && chain.success_rate >= self.config.min_success_rate
        });

        // Sort by success rate * occurrence count (most reliable first)
        chains.sort_by(|a, b| {
            let score_a = a.success_rate * (a.occurrence_count as f32).ln().max(1.0);
            let score_b = b.success_rate * (b.occurrence_count as f32).ln().max(1.0);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(chains)
    }

    /// Find sequences that consistently lead to failure.
    ///
    /// # Arguments
    /// * `conn` - Database connection
    /// * `min_length` - Minimum chain length
    /// * `min_occurrences` - Minimum occurrences
    ///
    /// # Returns
    /// Vector of pattern chains with high failure rates
    pub fn find_failure_chains(
        &self,
        conn: &rusqlite::Connection,
        min_length: Option<usize>,
        min_occurrences: Option<u32>,
    ) -> Result<Vec<PatternChain>, rusqlite::Error> {
        let min_len = min_length.unwrap_or(self.config.min_chain_length);
        let min_occ = min_occurrences.unwrap_or(self.config.min_occurrences);

        // Get failed trajectories
        let trajectories = self.storage.get_failed(conn, None)?;

        // Extract and count chains
        let mut chains = self.extract_chains(&trajectories, min_len);

        // Filter by occurrence count and failure rate (low success rate)
        chains.retain(|chain| {
            chain.occurrence_count >= min_occ && chain.success_rate <= self.config.max_failure_rate
        });

        // Sort by failure rate * occurrence count (most reliably failing first)
        chains.sort_by(|a, b| {
            let score_a = (1.0 - a.success_rate) * (a.occurrence_count as f32).ln().max(1.0);
            let score_b = (1.0 - b.success_rate) * (b.occurrence_count as f32).ln().max(1.0);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(chains)
    }

    /// Get the transition probability from one pattern to another.
    ///
    /// P(to | from) = count(from -> to) / count(from -> *)
    ///
    /// # Returns
    /// Probability between 0.0 and 1.0
    pub fn get_transition_probability(
        &self,
        conn: &rusqlite::Connection,
        from: &str,
        to: &str,
    ) -> Result<f32, rusqlite::Error> {
        let transitions = self.build_transition_matrix(conn)?;

        let from_to_count = transitions
            .iter()
            .find(|t| t.from_pattern == from && t.to_pattern == to)
            .map(|t| t.count)
            .unwrap_or(0);

        let from_total: u32 = transitions
            .iter()
            .filter(|t| t.from_pattern == from)
            .map(|t| t.count)
            .sum();

        if from_total == 0 {
            Ok(0.0)
        } else {
            Ok(from_to_count as f32 / from_total as f32)
        }
    }

    /// Get all transitions from a pattern with probabilities.
    ///
    /// # Returns
    /// Vector of transitions sorted by probability (highest first)
    pub fn get_pattern_successors(
        &self,
        conn: &rusqlite::Connection,
        pattern_id: &str,
    ) -> Result<Vec<PatternTransition>, rusqlite::Error> {
        let all_transitions = self.build_transition_matrix(conn)?;

        // Filter to transitions from this pattern
        let mut successors: Vec<PatternTransition> = all_transitions
            .into_iter()
            .filter(|t| t.from_pattern == pattern_id)
            .collect();

        // Calculate total count for probability normalization
        let total: u32 = successors.iter().map(|t| t.count).sum();

        // Update probabilities
        for successor in &mut successors {
            if total > 0 {
                successor.probability = successor.count as f32 / total as f32;
            }
        }

        // Sort by probability descending
        successors.sort_by(|a, b| {
            b.probability.partial_cmp(&a.probability).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(successors)
    }

    /// Identify patterns that "compound correctness".
    ///
    /// These are patterns where using them increases the success probability
    /// of subsequent patterns. A pattern P is compounding if:
    ///   success_rate_after_P > baseline_success_rate * compounding_threshold
    ///
    /// # Returns
    /// Vector of pattern IDs that compound success
    pub fn find_compounding_patterns(
        &self,
        conn: &rusqlite::Connection,
    ) -> Result<Vec<String>, rusqlite::Error> {
        // Calculate baseline success rate
        let stats = self.storage.get_stats(conn)?;
        let baseline = stats.success_rate();

        if baseline <= 0.0 || stats.total_count < 5 {
            return Ok(Vec::new());
        }

        // Get all trajectories
        let all_trajectories = self.storage.get_recent(conn, 1000)?;

        // Track success rates after each pattern
        let mut pattern_follow_success: std::collections::HashMap<String, (u32, u32)> =
            std::collections::HashMap::new(); // (success_count, total_count)

        for trajectory in &all_trajectories {
            let pattern_ids = trajectory.all_pattern_ids();
            let is_success = trajectory.success;

            // For each pattern (except the last), record the trajectory outcome
            for i in 0..pattern_ids.len().saturating_sub(1) {
                let pattern_id = pattern_ids[i].as_str().to_string();
                let entry = pattern_follow_success.entry(pattern_id).or_insert((0, 0));
                entry.1 += 1;
                if is_success {
                    entry.0 += 1;
                }
            }
        }

        // Find compounding patterns
        let threshold = baseline * self.config.compounding_threshold;
        let mut compounding: Vec<(String, f32)> = pattern_follow_success
            .into_iter()
            .filter_map(|(pattern, (successes, total))| {
                if total < self.config.min_occurrences {
                    return None;
                }
                let rate = successes as f32 / total as f32;
                if rate > threshold {
                    Some((pattern, rate))
                } else {
                    None
                }
            })
            .collect();

        // Sort by success rate improvement
        compounding.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(compounding.into_iter().map(|(p, _)| p).collect())
    }

    /// Identify patterns that often precede failures.
    ///
    /// These are patterns where using them decreases the success probability
    /// of subsequent patterns.
    ///
    /// # Returns
    /// Vector of pattern IDs that are risky
    pub fn find_risky_patterns(
        &self,
        conn: &rusqlite::Connection,
    ) -> Result<Vec<String>, rusqlite::Error> {
        // Calculate baseline success rate
        let stats = self.storage.get_stats(conn)?;
        let baseline = stats.success_rate();

        if baseline <= 0.0 || stats.total_count < 5 {
            return Ok(Vec::new());
        }

        // Get all trajectories
        let all_trajectories = self.storage.get_recent(conn, 1000)?;

        // Track failure rates after each pattern
        let mut pattern_follow_failure: std::collections::HashMap<String, (u32, u32)> =
            std::collections::HashMap::new(); // (failure_count, total_count)

        for trajectory in &all_trajectories {
            let pattern_ids = trajectory.all_pattern_ids();
            let is_failure = !trajectory.success;

            // For each pattern (except the last), record the trajectory outcome
            for i in 0..pattern_ids.len().saturating_sub(1) {
                let pattern_id = pattern_ids[i].as_str().to_string();
                let entry = pattern_follow_failure.entry(pattern_id).or_insert((0, 0));
                entry.1 += 1;
                if is_failure {
                    entry.0 += 1;
                }
            }
        }

        // Find risky patterns (success rate below threshold)
        let threshold = baseline * self.config.risky_threshold;
        let mut risky: Vec<(String, f32)> = pattern_follow_failure
            .into_iter()
            .filter_map(|(pattern, (failures, total))| {
                if total < self.config.min_occurrences {
                    return None;
                }
                let success_rate = 1.0 - (failures as f32 / total as f32);
                if success_rate < threshold {
                    Some((pattern, success_rate))
                } else {
                    None
                }
            })
            .collect();

        // Sort by failure rate (lowest success rate first)
        risky.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(risky.into_iter().map(|(p, _)| p).collect())
    }

    /// Perform full chain analysis.
    ///
    /// This method combines all analysis types into a comprehensive report.
    pub fn analyze(&self, conn: &rusqlite::Connection) -> Result<ChainAnalysis, rusqlite::Error> {
        let stats = self.storage.get_stats(conn)?;

        Ok(ChainAnalysis {
            success_chains: self.find_success_chains(conn, None, None)?,
            failure_chains: self.find_failure_chains(conn, None, None)?,
            high_value_patterns: self.find_compounding_patterns(conn)?,
            risky_patterns: self.find_risky_patterns(conn)?,
            baseline_success_rate: stats.success_rate(),
            trajectories_analyzed: stats.total_count,
        })
    }

    /// Get pattern co-occurrence statistics.
    ///
    /// Finds which patterns frequently appear together in successful trajectories.
    pub fn get_pattern_cooccurrence(
        &self,
        conn: &rusqlite::Connection,
        success_only: bool,
    ) -> Result<Vec<(String, String, u32)>, rusqlite::Error> {
        let trajectories = if success_only {
            self.storage.get_successful(conn, None)?
        } else {
            self.storage.get_recent(conn, 1000)?
        };

        let mut cooccurrence: std::collections::HashMap<(String, String), u32> =
            std::collections::HashMap::new();

        for trajectory in &trajectories {
            let pattern_ids = trajectory.all_pattern_ids();
            // Count all pairs (order-independent)
            for i in 0..pattern_ids.len() {
                for j in (i + 1)..pattern_ids.len() {
                    let p1 = pattern_ids[i].as_str().to_string();
                    let p2 = pattern_ids[j].as_str().to_string();
                    // Normalize order for consistent keys
                    let key = if p1 < p2 { (p1, p2) } else { (p2, p1) };
                    *cooccurrence.entry(key).or_insert(0) += 1;
                }
            }
        }

        let mut result: Vec<(String, String, u32)> = cooccurrence
            .into_iter()
            .map(|((p1, p2), count)| (p1, p2, count))
            .collect();

        result.sort_by(|a, b| b.2.cmp(&a.2));

        Ok(result)
    }

    // ========================================================================
    // Private helper methods
    // ========================================================================

    /// Extract pattern chains from trajectories.
    fn extract_chains(
        &self,
        trajectories: &[Trajectory],
        min_length: usize,
    ) -> Vec<PatternChain> {
        let mut chain_map: std::collections::HashMap<String, PatternChain> =
            std::collections::HashMap::new();

        for trajectory in trajectories {
            let pattern_ids: Vec<String> = trajectory
                .all_pattern_ids()
                .into_iter()
                .map(|p| p.as_str().to_string())
                .collect();

            if pattern_ids.len() < min_length {
                continue;
            }

            // Extract all subsequences of valid length
            for len in min_length..=self.config.max_chain_length.min(pattern_ids.len()) {
                for start in 0..=(pattern_ids.len() - len) {
                    let chain_patterns: Vec<String> = pattern_ids[start..start + len].to_vec();
                    let key = chain_patterns.join("->");

                    let chain = chain_map.entry(key).or_insert_with(|| PatternChain::new(chain_patterns));
                    chain.record_observation(trajectory.success, trajectory.total_reward);
                }
            }
        }

        chain_map.into_values().collect()
    }

    /// Build a matrix of pattern transitions.
    fn build_transition_matrix(
        &self,
        conn: &rusqlite::Connection,
    ) -> Result<Vec<PatternTransition>, rusqlite::Error> {
        let trajectories = self.storage.get_recent(conn, 1000)?;

        let mut transitions: std::collections::HashMap<(String, String), PatternTransition> =
            std::collections::HashMap::new();

        for trajectory in &trajectories {
            let pattern_ids = trajectory.all_pattern_ids();

            // Record transitions between consecutive patterns
            for i in 0..pattern_ids.len().saturating_sub(1) {
                let from = pattern_ids[i].as_str().to_string();
                let to = pattern_ids[i + 1].as_str().to_string();
                let key = (from.clone(), to.clone());

                let transition = transitions.entry(key).or_insert_with(|| {
                    PatternTransition::new(from, to)
                });

                transition.count += 1;
                let old_count = transition.count as f32 - 1.0;
                let new_count = transition.count as f32;

                // Update running average for success rate
                let old_rate = transition.success_rate;
                transition.success_rate = (old_rate * old_count + if trajectory.success { 1.0 } else { 0.0 }) / new_count;

                // Update running average for reward
                let old_reward = transition.avg_reward;
                transition.avg_reward = (old_reward * old_count + trajectory.total_reward) / new_count;
            }
        }

        Ok(transitions.into_values().collect())
    }
}

// ============================================================================
// Analysis Tests
// ============================================================================

#[cfg(test)]
mod analysis_tests {
    use super::*;
    use rusqlite::Connection;

    fn setup_analysis_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        let storage = TrajectoryStorage::new();
        storage.init_schema(&conn).unwrap();
        conn
    }

    fn create_test_trajectory(
        storage: &TrajectoryStorage,
        conn: &Connection,
        patterns: Vec<&str>,
        success: bool,
        reward: f32,
    ) {
        let mut trajectory = Trajectory::new();
        for (i, pat) in patterns.iter().enumerate() {
            trajectory.add_step(TrajectoryStep::new(
                if i == 0 { StepType::PatternRetrieval } else { StepType::Decision },
                vec![PatternId::from_string(*pat)],
                format!("Step {}", i),
                0.8,
            ));
        }
        let outcome = if success { Outcome::Success } else { Outcome::Failure };
        trajectory.set_outcome(outcome, reward);
        storage.store(conn, &trajectory).unwrap();
    }

    #[test]
    fn test_find_success_chains() {
        let conn = setup_analysis_db();
        let storage = TrajectoryStorage::new();
        let analyzer = TrajectoryAnalyzer::with_config(&storage, TrajectoryAnalysisConfig {
            min_occurrences: 2,
            min_success_rate: 0.8,
            ..Default::default()
        });

        // Create trajectories with a common success pattern
        for _ in 0..5 {
            create_test_trajectory(&storage, &conn, vec!["A", "B", "C"], true, 0.9);
        }
        // Add some failures with different patterns
        for _ in 0..2 {
            create_test_trajectory(&storage, &conn, vec!["X", "Y", "Z"], false, 0.2);
        }

        let chains = analyzer.find_success_chains(&conn, Some(2), Some(2)).unwrap();

        assert!(!chains.is_empty(), "Should find success chains");
        // The chain A->B should appear multiple times with high success
        let ab_chain = chains.iter().find(|c| c.patterns == vec!["A", "B"]);
        assert!(ab_chain.is_some(), "Should find A->B chain");
        assert!(ab_chain.unwrap().success_rate >= 0.8);
    }

    #[test]
    fn test_find_failure_chains() {
        let conn = setup_analysis_db();
        let storage = TrajectoryStorage::new();
        let analyzer = TrajectoryAnalyzer::with_config(&storage, TrajectoryAnalysisConfig {
            min_occurrences: 2,
            max_failure_rate: 0.3,
            ..Default::default()
        });

        // Create failure trajectories with common pattern
        for _ in 0..5 {
            create_test_trajectory(&storage, &conn, vec!["BAD1", "BAD2"], false, 0.1);
        }
        // Add some successes with different patterns
        for _ in 0..2 {
            create_test_trajectory(&storage, &conn, vec!["GOOD1", "GOOD2"], true, 0.9);
        }

        let chains = analyzer.find_failure_chains(&conn, Some(2), Some(2)).unwrap();

        assert!(!chains.is_empty(), "Should find failure chains");
        let bad_chain = chains.iter().find(|c| c.patterns == vec!["BAD1", "BAD2"]);
        assert!(bad_chain.is_some(), "Should find BAD1->BAD2 chain");
        assert!(bad_chain.unwrap().success_rate <= 0.3);
    }

    #[test]
    fn test_transition_probability() {
        let conn = setup_analysis_db();
        let storage = TrajectoryStorage::new();
        let analyzer = TrajectoryAnalyzer::new(&storage);

        // Create trajectories: A->B happens 3 times, A->C happens 1 time
        for _ in 0..3 {
            create_test_trajectory(&storage, &conn, vec!["A", "B"], true, 0.8);
        }
        create_test_trajectory(&storage, &conn, vec!["A", "C"], true, 0.7);

        let prob_ab = analyzer.get_transition_probability(&conn, "A", "B").unwrap();
        let prob_ac = analyzer.get_transition_probability(&conn, "A", "C").unwrap();

        assert!((prob_ab - 0.75).abs() < 0.01, "P(B|A) should be ~0.75, got {}", prob_ab);
        assert!((prob_ac - 0.25).abs() < 0.01, "P(C|A) should be ~0.25, got {}", prob_ac);
    }

    #[test]
    fn test_pattern_successors() {
        let conn = setup_analysis_db();
        let storage = TrajectoryStorage::new();
        let analyzer = TrajectoryAnalyzer::new(&storage);

        // Create varied transitions from pattern "START"
        for _ in 0..5 {
            create_test_trajectory(&storage, &conn, vec!["START", "COMMON"], true, 0.8);
        }
        for _ in 0..3 {
            create_test_trajectory(&storage, &conn, vec!["START", "MEDIUM"], true, 0.7);
        }
        create_test_trajectory(&storage, &conn, vec!["START", "RARE"], true, 0.6);

        let successors = analyzer.get_pattern_successors(&conn, "START").unwrap();

        assert_eq!(successors.len(), 3, "Should have 3 successors");
        assert_eq!(successors[0].to_pattern, "COMMON", "COMMON should be most probable");
        assert!(successors[0].probability > successors[1].probability);
    }

    #[test]
    fn test_compounding_patterns() {
        let conn = setup_analysis_db();
        let storage = TrajectoryStorage::new();
        let analyzer = TrajectoryAnalyzer::with_config(&storage, TrajectoryAnalysisConfig {
            min_occurrences: 2,
            compounding_threshold: 1.1, // 10% above baseline
            ..Default::default()
        });

        // Create a mix of trajectories
        // "BOOST" pattern leads to high success
        for _ in 0..6 {
            create_test_trajectory(&storage, &conn, vec!["BOOST", "FOLLOW"], true, 0.9);
        }
        for _ in 0..1 {
            create_test_trajectory(&storage, &conn, vec!["BOOST", "FOLLOW"], false, 0.3);
        }

        // "NORMAL" pattern has baseline success
        for _ in 0..3 {
            create_test_trajectory(&storage, &conn, vec!["NORMAL", "NEXT"], true, 0.7);
        }
        for _ in 0..3 {
            create_test_trajectory(&storage, &conn, vec!["NORMAL", "NEXT"], false, 0.2);
        }

        let compounding = analyzer.find_compounding_patterns(&conn).unwrap();

        // BOOST should be identified as compounding
        assert!(compounding.contains(&"BOOST".to_string()),
            "BOOST should be a compounding pattern, found: {:?}", compounding);
    }

    #[test]
    fn test_risky_patterns() {
        let conn = setup_analysis_db();
        let storage = TrajectoryStorage::new();
        let analyzer = TrajectoryAnalyzer::with_config(&storage, TrajectoryAnalysisConfig {
            min_occurrences: 2,
            risky_threshold: 0.9, // Below 90% of baseline is risky
            ..Default::default()
        });

        // Create trajectories where "RISKY" leads to failures
        for _ in 0..5 {
            create_test_trajectory(&storage, &conn, vec!["RISKY", "DOOM"], false, 0.1);
        }
        for _ in 0..1 {
            create_test_trajectory(&storage, &conn, vec!["RISKY", "DOOM"], true, 0.6);
        }

        // Create normal trajectories
        for _ in 0..5 {
            create_test_trajectory(&storage, &conn, vec!["SAFE", "OK"], true, 0.8);
        }

        let risky = analyzer.find_risky_patterns(&conn).unwrap();

        assert!(risky.contains(&"RISKY".to_string()),
            "RISKY should be identified as risky, found: {:?}", risky);
    }

    #[test]
    fn test_full_analysis() {
        let conn = setup_analysis_db();
        let storage = TrajectoryStorage::new();
        let analyzer = TrajectoryAnalyzer::with_config(&storage, TrajectoryAnalysisConfig {
            min_occurrences: 2,
            ..Default::default()
        });

        // Create diverse trajectories
        for _ in 0..5 {
            create_test_trajectory(&storage, &conn, vec!["A", "B", "C"], true, 0.9);
        }
        for _ in 0..3 {
            create_test_trajectory(&storage, &conn, vec!["X", "Y"], false, 0.2);
        }
        for _ in 0..2 {
            create_test_trajectory(&storage, &conn, vec!["A", "Z"], true, 0.7);
        }

        let analysis = analyzer.analyze(&conn).unwrap();

        assert_eq!(analysis.trajectories_analyzed, 10);
        assert!(!analysis.success_chains.is_empty() || !analysis.failure_chains.is_empty(),
            "Should find some chains");
    }

    #[test]
    fn test_pattern_cooccurrence() {
        let conn = setup_analysis_db();
        let storage = TrajectoryStorage::new();
        let analyzer = TrajectoryAnalyzer::new(&storage);

        // Create trajectories with co-occurring patterns
        for _ in 0..5 {
            create_test_trajectory(&storage, &conn, vec!["A", "B", "C"], true, 0.8);
        }
        for _ in 0..3 {
            create_test_trajectory(&storage, &conn, vec!["A", "B", "D"], true, 0.7);
        }

        let cooccurrence = analyzer.get_pattern_cooccurrence(&conn, true).unwrap();

        assert!(!cooccurrence.is_empty(), "Should find co-occurrences");
        // A-B should be most common (appears in all 8 trajectories)
        let ab = cooccurrence.iter().find(|(p1, p2, _)|
            (p1 == "A" && p2 == "B") || (p1 == "B" && p2 == "A")
        );
        assert!(ab.is_some(), "Should find A-B co-occurrence");
        assert_eq!(ab.unwrap().2, 8, "A-B should appear 8 times");
    }

    #[test]
    fn test_empty_analysis() {
        let conn = setup_analysis_db();
        let storage = TrajectoryStorage::new();
        let analyzer = TrajectoryAnalyzer::new(&storage);

        // No trajectories stored
        let analysis = analyzer.analyze(&conn).unwrap();

        assert!(analysis.success_chains.is_empty());
        assert!(analysis.failure_chains.is_empty());
        assert!(analysis.high_value_patterns.is_empty());
        assert!(analysis.risky_patterns.is_empty());
        assert_eq!(analysis.trajectories_analyzed, 0);
    }

    #[test]
    fn test_chain_key() {
        let chain = PatternChain::new(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
        assert_eq!(chain.key(), "A->B->C");
    }

    #[test]
    fn test_chain_record_observation() {
        let mut chain = PatternChain::new(vec!["A".to_string()]);

        chain.record_observation(true, 0.8);
        assert_eq!(chain.occurrence_count, 1);
        assert!((chain.success_rate - 1.0).abs() < 0.001);
        assert!((chain.avg_reward - 0.8).abs() < 0.001);

        chain.record_observation(false, 0.2);
        assert_eq!(chain.occurrence_count, 2);
        assert!((chain.success_rate - 0.5).abs() < 0.001);
        assert!((chain.avg_reward - 0.5).abs() < 0.001);
    }
}
