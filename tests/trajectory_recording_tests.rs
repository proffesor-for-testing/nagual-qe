//! ProfDAG Trajectory Recording Tests - Phase 1
//!
//! Comprehensive test suite for trajectory recording functionality.
//! Tests cover step recording, full trajectory capture, replay, and outcome linking.
//!
//! # Trajectory Components
//! - TrajectoryStep: Individual decision/action with context
//! - Trajectory: Full sequence of steps with metadata
//! - Outcome: Result of trajectory execution
//! - Replay: Re-execution capability for learning

use std::collections::HashMap;
use std::time::Instant;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

mod common;
use common::normalized_embedding;

// ============================================================================
// Trajectory Structures
// ============================================================================

/// The outcome of a trajectory execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrajectoryOutcome {
    Success,
    PartialSuccess,
    Failure,
    Timeout,
    Cancelled,
    Unknown,
}

impl TrajectoryOutcome {
    pub fn is_positive(&self) -> bool {
        matches!(self, TrajectoryOutcome::Success | TrajectoryOutcome::PartialSuccess)
    }

    pub fn reward_value(&self) -> f32 {
        match self {
            TrajectoryOutcome::Success => 1.0,
            TrajectoryOutcome::PartialSuccess => 0.6,
            TrajectoryOutcome::Failure => 0.1,
            TrajectoryOutcome::Timeout => 0.2,
            TrajectoryOutcome::Cancelled => 0.0,
            TrajectoryOutcome::Unknown => 0.3,
        }
    }
}

/// Type of action in a trajectory step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    Query,
    Retrieve,
    Generate,
    Execute,
    Validate,
    Decide,
    Custom(String),
}

/// A single step in a trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryStep {
    pub id: String,
    pub step_number: usize,
    pub action_type: ActionType,
    pub input: String,
    pub output: Option<String>,
    pub context: serde_json::Value,
    pub embedding: Option<Vec<f32>>,
    pub duration_ms: u64,
    pub timestamp: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

impl TrajectoryStep {
    pub fn new(step_number: usize, action_type: ActionType, input: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            step_number,
            action_type,
            input: input.into(),
            output: None,
            context: serde_json::json!({}),
            embedding: None,
            duration_ms: 0,
            timestamp: Utc::now(),
            metadata: serde_json::json!({}),
        }
    }

    pub fn with_output(mut self, output: impl Into<String>) -> Self {
        self.output = Some(output.into());
        self
    }

    pub fn with_context(mut self, context: serde_json::Value) -> Self {
        self.context = context;
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = duration_ms;
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// A complete trajectory representing a decision path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub id: String,
    pub session_id: String,
    pub task_description: String,
    pub steps: Vec<TrajectoryStep>,
    pub outcome: Option<TrajectoryOutcome>,
    pub reward: Option<f32>,
    pub total_duration_ms: u64,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub metadata: serde_json::Value,
    pub tags: Vec<String>,
}

impl Trajectory {
    pub fn new(session_id: impl Into<String>, task_description: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.into(),
            task_description: task_description.into(),
            steps: Vec::new(),
            outcome: None,
            reward: None,
            total_duration_ms: 0,
            started_at: Utc::now(),
            completed_at: None,
            metadata: serde_json::json!({}),
            tags: Vec::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn add_step(&mut self, step: TrajectoryStep) {
        self.steps.push(step);
        self.total_duration_ms = self.steps.iter().map(|s| s.duration_ms).sum();
    }

    pub fn complete(&mut self, outcome: TrajectoryOutcome) {
        self.outcome = Some(outcome);
        self.reward = Some(outcome.reward_value());
        self.completed_at = Some(Utc::now());
    }

    pub fn complete_with_reward(&mut self, outcome: TrajectoryOutcome, reward: f32) {
        self.outcome = Some(outcome);
        self.reward = Some(reward.clamp(0.0, 1.0));
        self.completed_at = Some(Utc::now());
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    pub fn is_complete(&self) -> bool {
        self.outcome.is_some()
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// Recorder for capturing trajectories.
pub struct TrajectoryRecorder {
    current_trajectory: Option<Trajectory>,
    completed_trajectories: Vec<Trajectory>,
    step_embedder: Option<Box<dyn Fn(&str) -> Vec<f32> + Send + Sync>>,
}

impl std::fmt::Debug for TrajectoryRecorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrajectoryRecorder")
            .field("current_trajectory", &self.current_trajectory)
            .field("completed_trajectories", &self.completed_trajectories)
            .field("step_embedder", &self.step_embedder.as_ref().map(|_| "<embedder>"))
            .finish()
    }
}

impl TrajectoryRecorder {
    pub fn new() -> Self {
        Self {
            current_trajectory: None,
            completed_trajectories: Vec::new(),
            step_embedder: None,
        }
    }

    pub fn with_embedder<F>(mut self, embedder: F) -> Self
    where
        F: Fn(&str) -> Vec<f32> + Send + Sync + 'static,
    {
        self.step_embedder = Some(Box::new(embedder));
        self
    }

    pub fn start_trajectory(&mut self, session_id: impl Into<String>, task: impl Into<String>) -> &Trajectory {
        let trajectory = Trajectory::new(session_id, task);
        self.current_trajectory = Some(trajectory);
        self.current_trajectory.as_ref().unwrap()
    }

    pub fn record_step(&mut self, action: ActionType, input: impl Into<String>) -> Result<&TrajectoryStep, String> {
        let trajectory = self.current_trajectory.as_mut()
            .ok_or_else(|| "No active trajectory".to_string())?;

        let step_number = trajectory.step_count() + 1;
        let input_str = input.into();

        let mut step = TrajectoryStep::new(step_number, action, &input_str);

        // Generate embedding if embedder is available
        if let Some(ref embedder) = self.step_embedder {
            step.embedding = Some(embedder(&input_str));
        }

        trajectory.add_step(step);
        Ok(trajectory.steps.last().unwrap())
    }

    pub fn record_step_with_output(
        &mut self,
        action: ActionType,
        input: impl Into<String>,
        output: impl Into<String>,
        duration_ms: u64,
    ) -> Result<&TrajectoryStep, String> {
        let trajectory = self.current_trajectory.as_mut()
            .ok_or_else(|| "No active trajectory".to_string())?;

        let step_number = trajectory.step_count() + 1;
        let input_str = input.into();

        let mut step = TrajectoryStep::new(step_number, action, &input_str)
            .with_output(output)
            .with_duration(duration_ms);

        if let Some(ref embedder) = self.step_embedder {
            step.embedding = Some(embedder(&input_str));
        }

        trajectory.add_step(step);
        Ok(trajectory.steps.last().unwrap())
    }

    pub fn complete_trajectory(&mut self, outcome: TrajectoryOutcome) -> Result<Trajectory, String> {
        let mut trajectory = self.current_trajectory.take()
            .ok_or_else(|| "No active trajectory".to_string())?;

        trajectory.complete(outcome);
        self.completed_trajectories.push(trajectory.clone());
        Ok(trajectory)
    }

    pub fn complete_trajectory_with_reward(
        &mut self,
        outcome: TrajectoryOutcome,
        reward: f32,
    ) -> Result<Trajectory, String> {
        let mut trajectory = self.current_trajectory.take()
            .ok_or_else(|| "No active trajectory".to_string())?;

        trajectory.complete_with_reward(outcome, reward);
        self.completed_trajectories.push(trajectory.clone());
        Ok(trajectory)
    }

    pub fn cancel_trajectory(&mut self) -> Option<Trajectory> {
        if let Some(mut trajectory) = self.current_trajectory.take() {
            trajectory.complete(TrajectoryOutcome::Cancelled);
            self.completed_trajectories.push(trajectory.clone());
            Some(trajectory)
        } else {
            None
        }
    }

    pub fn current(&self) -> Option<&Trajectory> {
        self.current_trajectory.as_ref()
    }

    pub fn completed(&self) -> &[Trajectory] {
        &self.completed_trajectories
    }

    pub fn completed_count(&self) -> usize {
        self.completed_trajectories.len()
    }
}

impl Default for TrajectoryRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Trajectory storage for persistence.
#[derive(Debug, Default)]
pub struct TrajectoryStorage {
    trajectories: HashMap<String, Trajectory>,
}

impl TrajectoryStorage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn store(&mut self, trajectory: Trajectory) -> String {
        let id = trajectory.id.clone();
        self.trajectories.insert(id.clone(), trajectory);
        id
    }

    pub fn get(&self, id: &str) -> Option<&Trajectory> {
        self.trajectories.get(id)
    }

    pub fn get_by_session(&self, session_id: &str) -> Vec<&Trajectory> {
        self.trajectories
            .values()
            .filter(|t| t.session_id == session_id)
            .collect()
    }

    pub fn get_successful(&self) -> Vec<&Trajectory> {
        self.trajectories
            .values()
            .filter(|t| t.outcome.map_or(false, |o| o.is_positive()))
            .collect()
    }

    pub fn get_by_tag(&self, tag: &str) -> Vec<&Trajectory> {
        self.trajectories
            .values()
            .filter(|t| t.tags.contains(&tag.to_string()))
            .collect()
    }

    pub fn delete(&mut self, id: &str) -> Option<Trajectory> {
        self.trajectories.remove(id)
    }

    pub fn count(&self) -> usize {
        self.trajectories.len()
    }
}

/// Trajectory replayer for learning from past executions.
#[derive(Debug)]
pub struct TrajectoryReplayer {
    trajectory: Trajectory,
    current_step: usize,
}

impl TrajectoryReplayer {
    pub fn new(trajectory: Trajectory) -> Self {
        Self {
            trajectory,
            current_step: 0,
        }
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    pub fn next_step(&mut self) -> Option<&TrajectoryStep> {
        if self.current_step < self.trajectory.steps.len() {
            let step = &self.trajectory.steps[self.current_step];
            self.current_step += 1;
            Some(step)
        } else {
            None
        }
    }

    pub fn peek(&self) -> Option<&TrajectoryStep> {
        self.trajectory.steps.get(self.current_step)
    }

    pub fn is_complete(&self) -> bool {
        self.current_step >= self.trajectory.steps.len()
    }

    pub fn remaining_steps(&self) -> usize {
        self.trajectory.steps.len().saturating_sub(self.current_step)
    }

    pub fn progress(&self) -> f32 {
        if self.trajectory.steps.is_empty() {
            1.0
        } else {
            self.current_step as f32 / self.trajectory.steps.len() as f32
        }
    }

    pub fn trajectory(&self) -> &Trajectory {
        &self.trajectory
    }

    pub fn get_all_steps(&self) -> &[TrajectoryStep] {
        &self.trajectory.steps
    }
}

// ============================================================================
// Step Recording Tests
// ============================================================================

mod step_recording_tests {
    use super::*;

    #[test]
    fn test_record_single_step() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.start_trajectory("session-1", "Test task");

        let step = recorder.record_step(ActionType::Query, "What is X?").unwrap();

        assert_eq!(step.step_number, 1);
        assert_eq!(step.action_type, ActionType::Query);
        assert_eq!(step.input, "What is X?");
        assert!(step.output.is_none());
    }

    #[test]
    fn test_record_step_with_output() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.start_trajectory("session-1", "Test task");

        let step = recorder.record_step_with_output(
            ActionType::Generate,
            "Generate code",
            "fn main() {}",
            150,
        ).unwrap();

        assert_eq!(step.step_number, 1);
        assert_eq!(step.action_type, ActionType::Generate);
        assert_eq!(step.output.as_ref().unwrap(), "fn main() {}");
        assert_eq!(step.duration_ms, 150);
    }

    #[test]
    fn test_record_multiple_steps() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.start_trajectory("session-1", "Multi-step task");

        recorder.record_step(ActionType::Query, "Step 1").unwrap();
        recorder.record_step(ActionType::Retrieve, "Step 2").unwrap();
        recorder.record_step(ActionType::Generate, "Step 3").unwrap();

        let trajectory = recorder.current().unwrap();
        assert_eq!(trajectory.step_count(), 3);
        assert_eq!(trajectory.steps[0].step_number, 1);
        assert_eq!(trajectory.steps[1].step_number, 2);
        assert_eq!(trajectory.steps[2].step_number, 3);
    }

    #[test]
    fn test_record_step_without_trajectory_fails() {
        let mut recorder = TrajectoryRecorder::new();
        let result = recorder.record_step(ActionType::Query, "Orphan step");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No active trajectory"));
    }

    #[test]
    fn test_step_with_context() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.start_trajectory("session-1", "Context test");

        let context = serde_json::json!({
            "model": "gpt-4",
            "temperature": 0.7,
            "previous_results": ["result1", "result2"]
        });

        let step = TrajectoryStep::new(1, ActionType::Generate, "Generate response")
            .with_context(context.clone());

        assert_eq!(step.context["model"], "gpt-4");
        assert_eq!(step.context["temperature"], 0.7);
    }

    #[test]
    fn test_step_with_embedding() {
        let embedding = normalized_embedding(128);
        let step = TrajectoryStep::new(1, ActionType::Query, "Test query")
            .with_embedding(embedding.clone());

        assert!(step.embedding.is_some());
        assert_eq!(step.embedding.as_ref().unwrap().len(), 128);
    }

    #[test]
    fn test_step_with_metadata() {
        let metadata = serde_json::json!({
            "tool": "code_search",
            "matches": 42
        });

        let step = TrajectoryStep::new(1, ActionType::Retrieve, "Search for patterns")
            .with_metadata(metadata);

        assert_eq!(step.metadata["tool"], "code_search");
        assert_eq!(step.metadata["matches"], 42);
    }

    #[test]
    fn test_all_action_types() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.start_trajectory("session-1", "Action types test");

        let action_types = vec![
            ActionType::Query,
            ActionType::Retrieve,
            ActionType::Generate,
            ActionType::Execute,
            ActionType::Validate,
            ActionType::Decide,
            ActionType::Custom("my_action".to_string()),
        ];

        for action in action_types {
            recorder.record_step(action.clone(), format!("Action: {:?}", action)).unwrap();
        }

        assert_eq!(recorder.current().unwrap().step_count(), 7);
    }

    #[test]
    fn test_step_timestamps_increase() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.start_trajectory("session-1", "Timestamp test");

        recorder.record_step(ActionType::Query, "Step 1").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        recorder.record_step(ActionType::Query, "Step 2").unwrap();

        let trajectory = recorder.current().unwrap();
        assert!(trajectory.steps[1].timestamp >= trajectory.steps[0].timestamp);
    }
}

// ============================================================================
// Full Trajectory Capture Tests
// ============================================================================

mod trajectory_capture_tests {
    use super::*;

    #[test]
    fn test_start_and_complete_trajectory() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.start_trajectory("session-1", "Complete task");

        recorder.record_step(ActionType::Query, "Step 1").unwrap();
        recorder.record_step(ActionType::Generate, "Step 2").unwrap();

        let trajectory = recorder.complete_trajectory(TrajectoryOutcome::Success).unwrap();

        assert!(trajectory.is_complete());
        assert_eq!(trajectory.outcome, Some(TrajectoryOutcome::Success));
        assert!(trajectory.completed_at.is_some());
    }

    #[test]
    fn test_trajectory_reward_calculation() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.start_trajectory("session-1", "Reward test");

        recorder.record_step(ActionType::Query, "Step").unwrap();
        let trajectory = recorder.complete_trajectory(TrajectoryOutcome::Success).unwrap();

        assert_eq!(trajectory.reward, Some(1.0));

        // Test other outcomes
        recorder.start_trajectory("session-2", "Partial");
        recorder.record_step(ActionType::Query, "Step").unwrap();
        let partial = recorder.complete_trajectory(TrajectoryOutcome::PartialSuccess).unwrap();
        assert_eq!(partial.reward, Some(0.6));

        recorder.start_trajectory("session-3", "Failure");
        recorder.record_step(ActionType::Query, "Step").unwrap();
        let failure = recorder.complete_trajectory(TrajectoryOutcome::Failure).unwrap();
        assert_eq!(failure.reward, Some(0.1));
    }

    #[test]
    fn test_trajectory_custom_reward() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.start_trajectory("session-1", "Custom reward");

        recorder.record_step(ActionType::Query, "Step").unwrap();
        let trajectory = recorder.complete_trajectory_with_reward(
            TrajectoryOutcome::PartialSuccess,
            0.85,
        ).unwrap();

        assert_eq!(trajectory.reward, Some(0.85));
    }

    #[test]
    fn test_trajectory_reward_clamping() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.start_trajectory("session-1", "Clamp test");
        recorder.record_step(ActionType::Query, "Step").unwrap();

        let trajectory = recorder.complete_trajectory_with_reward(
            TrajectoryOutcome::Success,
            1.5, // Should be clamped to 1.0
        ).unwrap();

        assert_eq!(trajectory.reward, Some(1.0));
    }

    #[test]
    fn test_trajectory_total_duration() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.start_trajectory("session-1", "Duration test");

        recorder.record_step_with_output(ActionType::Query, "Step 1", "Out 1", 100).unwrap();
        recorder.record_step_with_output(ActionType::Generate, "Step 2", "Out 2", 200).unwrap();
        recorder.record_step_with_output(ActionType::Execute, "Step 3", "Out 3", 150).unwrap();

        let trajectory = recorder.current().unwrap();
        assert_eq!(trajectory.total_duration_ms, 450);
    }

    #[test]
    fn test_cancel_trajectory() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.start_trajectory("session-1", "Cancel test");
        recorder.record_step(ActionType::Query, "Step 1").unwrap();

        let trajectory = recorder.cancel_trajectory().unwrap();

        assert_eq!(trajectory.outcome, Some(TrajectoryOutcome::Cancelled));
        assert_eq!(trajectory.reward, Some(0.0));
        assert!(recorder.current().is_none());
    }

    #[test]
    fn test_trajectory_with_tags() {
        let trajectory = Trajectory::new("session-1", "Tagged task")
            .with_tags(vec!["important".to_string(), "database".to_string(), "optimization".to_string()]);

        assert_eq!(trajectory.tags.len(), 3);
        assert!(trajectory.tags.contains(&"important".to_string()));
    }

    #[test]
    fn test_complete_without_active_trajectory_fails() {
        let mut recorder = TrajectoryRecorder::new();
        let result = recorder.complete_trajectory(TrajectoryOutcome::Success);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_trajectories() {
        let mut recorder = TrajectoryRecorder::new();

        // First trajectory
        recorder.start_trajectory("session-1", "Task 1");
        recorder.record_step(ActionType::Query, "Step 1").unwrap();
        recorder.complete_trajectory(TrajectoryOutcome::Success).unwrap();

        // Second trajectory
        recorder.start_trajectory("session-1", "Task 2");
        recorder.record_step(ActionType::Generate, "Step 1").unwrap();
        recorder.complete_trajectory(TrajectoryOutcome::Failure).unwrap();

        assert_eq!(recorder.completed_count(), 2);
        assert_eq!(recorder.completed()[0].task_description, "Task 1");
        assert_eq!(recorder.completed()[1].task_description, "Task 2");
    }

    #[test]
    fn test_trajectory_with_embedder() {
        let embedder = |_text: &str| -> Vec<f32> {
            normalized_embedding(64)
        };

        let mut recorder = TrajectoryRecorder::new().with_embedder(embedder);
        recorder.start_trajectory("session-1", "Embedder test");

        recorder.record_step(ActionType::Query, "Test input").unwrap();

        let trajectory = recorder.current().unwrap();
        assert!(trajectory.steps[0].embedding.is_some());
        assert_eq!(trajectory.steps[0].embedding.as_ref().unwrap().len(), 64);
    }
}

// ============================================================================
// Replay Functionality Tests
// ============================================================================

mod replay_tests {
    use super::*;

    fn create_sample_trajectory() -> Trajectory {
        let mut trajectory = Trajectory::new("session-1", "Sample task");

        trajectory.add_step(TrajectoryStep::new(1, ActionType::Query, "Query input")
            .with_output("Query result")
            .with_duration(100));

        trajectory.add_step(TrajectoryStep::new(2, ActionType::Retrieve, "Retrieve patterns")
            .with_output("Pattern list")
            .with_duration(50));

        trajectory.add_step(TrajectoryStep::new(3, ActionType::Generate, "Generate code")
            .with_output("fn main() {}")
            .with_duration(200));

        trajectory.complete(TrajectoryOutcome::Success);
        trajectory
    }

    #[test]
    fn test_replay_basic() {
        let trajectory = create_sample_trajectory();
        let mut replayer = TrajectoryReplayer::new(trajectory);

        assert!(!replayer.is_complete());
        assert_eq!(replayer.remaining_steps(), 3);

        let step1 = replayer.next_step().unwrap();
        assert_eq!(step1.step_number, 1);
        assert_eq!(step1.action_type, ActionType::Query);

        let step2 = replayer.next_step().unwrap();
        assert_eq!(step2.step_number, 2);

        let step3 = replayer.next_step().unwrap();
        assert_eq!(step3.step_number, 3);

        assert!(replayer.next_step().is_none());
        assert!(replayer.is_complete());
    }

    #[test]
    fn test_replay_peek() {
        let trajectory = create_sample_trajectory();
        let replayer = TrajectoryReplayer::new(trajectory);

        let peeked = replayer.peek().unwrap();
        assert_eq!(peeked.step_number, 1);

        // Peek doesn't advance
        let peeked_again = replayer.peek().unwrap();
        assert_eq!(peeked_again.step_number, 1);
    }

    #[test]
    fn test_replay_reset() {
        let trajectory = create_sample_trajectory();
        let mut replayer = TrajectoryReplayer::new(trajectory);

        replayer.next_step();
        replayer.next_step();
        assert_eq!(replayer.remaining_steps(), 1);

        replayer.reset();
        assert_eq!(replayer.remaining_steps(), 3);
        assert!(!replayer.is_complete());
    }

    #[test]
    fn test_replay_progress() {
        let trajectory = create_sample_trajectory();
        let mut replayer = TrajectoryReplayer::new(trajectory);

        assert!((replayer.progress() - 0.0).abs() < 0.001);

        replayer.next_step();
        assert!((replayer.progress() - 0.333).abs() < 0.01);

        replayer.next_step();
        assert!((replayer.progress() - 0.666).abs() < 0.01);

        replayer.next_step();
        assert!((replayer.progress() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_replay_empty_trajectory() {
        let trajectory = Trajectory::new("session-1", "Empty task");
        let mut replayer = TrajectoryReplayer::new(trajectory);

        assert!(replayer.is_complete());
        assert!(replayer.next_step().is_none());
        assert!((replayer.progress() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_replay_get_all_steps() {
        let trajectory = create_sample_trajectory();
        let replayer = TrajectoryReplayer::new(trajectory);

        let all_steps = replayer.get_all_steps();
        assert_eq!(all_steps.len(), 3);
    }

    #[test]
    fn test_replay_trajectory_access() {
        let trajectory = create_sample_trajectory();
        let replayer = TrajectoryReplayer::new(trajectory.clone());

        assert_eq!(replayer.trajectory().task_description, "Sample task");
        assert_eq!(replayer.trajectory().outcome, Some(TrajectoryOutcome::Success));
    }
}

// ============================================================================
// Outcome Linking Tests
// ============================================================================

mod outcome_linking_tests {
    use super::*;

    #[test]
    fn test_outcome_is_positive() {
        assert!(TrajectoryOutcome::Success.is_positive());
        assert!(TrajectoryOutcome::PartialSuccess.is_positive());
        assert!(!TrajectoryOutcome::Failure.is_positive());
        assert!(!TrajectoryOutcome::Timeout.is_positive());
        assert!(!TrajectoryOutcome::Cancelled.is_positive());
        assert!(!TrajectoryOutcome::Unknown.is_positive());
    }

    #[test]
    fn test_outcome_reward_values() {
        assert_eq!(TrajectoryOutcome::Success.reward_value(), 1.0);
        assert_eq!(TrajectoryOutcome::PartialSuccess.reward_value(), 0.6);
        assert_eq!(TrajectoryOutcome::Failure.reward_value(), 0.1);
        assert_eq!(TrajectoryOutcome::Timeout.reward_value(), 0.2);
        assert_eq!(TrajectoryOutcome::Cancelled.reward_value(), 0.0);
        assert_eq!(TrajectoryOutcome::Unknown.reward_value(), 0.3);
    }

    #[test]
    fn test_trajectory_outcome_linked_to_steps() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.start_trajectory("session-1", "Linked outcome test");

        recorder.record_step(ActionType::Query, "Query").unwrap();
        recorder.record_step(ActionType::Generate, "Generate").unwrap();

        let trajectory = recorder.complete_trajectory(TrajectoryOutcome::Success).unwrap();

        // Outcome should be linked to the entire trajectory
        assert_eq!(trajectory.outcome, Some(TrajectoryOutcome::Success));
        assert_eq!(trajectory.step_count(), 2);

        // All steps should be accessible with the outcome context
        for step in &trajectory.steps {
            assert!(step.step_number > 0);
        }
    }

    #[test]
    fn test_storage_filter_by_outcome() {
        let mut storage = TrajectoryStorage::new();

        // Create trajectories with different outcomes
        let mut success_traj = Trajectory::new("s1", "Success task");
        success_traj.complete(TrajectoryOutcome::Success);
        storage.store(success_traj);

        let mut partial_traj = Trajectory::new("s2", "Partial task");
        partial_traj.complete(TrajectoryOutcome::PartialSuccess);
        storage.store(partial_traj);

        let mut failure_traj = Trajectory::new("s3", "Failure task");
        failure_traj.complete(TrajectoryOutcome::Failure);
        storage.store(failure_traj);

        let successful = storage.get_successful();
        assert_eq!(successful.len(), 2);
        assert!(successful.iter().all(|t| t.outcome.map_or(false, |o| o.is_positive())));
    }

    #[test]
    fn test_storage_filter_by_session() {
        let mut storage = TrajectoryStorage::new();

        let mut t1 = Trajectory::new("session-1", "Task 1");
        t1.complete(TrajectoryOutcome::Success);
        storage.store(t1);

        let mut t2 = Trajectory::new("session-1", "Task 2");
        t2.complete(TrajectoryOutcome::Success);
        storage.store(t2);

        let mut t3 = Trajectory::new("session-2", "Task 3");
        t3.complete(TrajectoryOutcome::Success);
        storage.store(t3);

        let session1_trajs = storage.get_by_session("session-1");
        assert_eq!(session1_trajs.len(), 2);
    }

    #[test]
    fn test_storage_filter_by_tag() {
        let mut storage = TrajectoryStorage::new();

        let mut t1 = Trajectory::new("s1", "Task 1")
            .with_tags(vec!["database".to_string(), "optimization".to_string()]);
        t1.complete(TrajectoryOutcome::Success);
        storage.store(t1);

        let mut t2 = Trajectory::new("s2", "Task 2")
            .with_tags(vec!["api".to_string()]);
        t2.complete(TrajectoryOutcome::Success);
        storage.store(t2);

        let db_trajs = storage.get_by_tag("database");
        assert_eq!(db_trajs.len(), 1);

        let api_trajs = storage.get_by_tag("api");
        assert_eq!(api_trajs.len(), 1);

        let none_trajs = storage.get_by_tag("nonexistent");
        assert_eq!(none_trajs.len(), 0);
    }
}

// ============================================================================
// Storage Tests
// ============================================================================

mod storage_tests {
    use super::*;

    #[test]
    fn test_store_and_retrieve() {
        let mut storage = TrajectoryStorage::new();

        let mut trajectory = Trajectory::new("session-1", "Test task");
        trajectory.add_step(TrajectoryStep::new(1, ActionType::Query, "Query"));
        trajectory.complete(TrajectoryOutcome::Success);

        let id = storage.store(trajectory.clone());

        let retrieved = storage.get(&id).unwrap();
        assert_eq!(retrieved.task_description, "Test task");
        assert_eq!(retrieved.step_count(), 1);
    }

    #[test]
    fn test_store_multiple() {
        let mut storage = TrajectoryStorage::new();

        for i in 0..10 {
            let mut trajectory = Trajectory::new(format!("session-{}", i), format!("Task {}", i));
            trajectory.complete(TrajectoryOutcome::Success);
            storage.store(trajectory);
        }

        assert_eq!(storage.count(), 10);
    }

    #[test]
    fn test_delete_trajectory() {
        let mut storage = TrajectoryStorage::new();

        let mut trajectory = Trajectory::new("session-1", "To delete");
        trajectory.complete(TrajectoryOutcome::Success);
        let id = storage.store(trajectory);

        assert_eq!(storage.count(), 1);

        let deleted = storage.delete(&id);
        assert!(deleted.is_some());
        assert_eq!(storage.count(), 0);
        assert!(storage.get(&id).is_none());
    }

    #[test]
    fn test_get_nonexistent() {
        let storage = TrajectoryStorage::new();
        assert!(storage.get("nonexistent").is_none());
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut storage = TrajectoryStorage::new();
        assert!(storage.delete("nonexistent").is_none());
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_recording_many_steps() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.start_trajectory("session-1", "Many steps");

        let start = Instant::now();
        for i in 0..1000 {
            recorder.record_step_with_output(
                ActionType::Query,
                format!("Step {}", i),
                format!("Output {}", i),
                10,
            ).unwrap();
        }
        let duration = start.elapsed();

        assert_eq!(recorder.current().unwrap().step_count(), 1000);
        assert!(
            duration.as_millis() < 500,
            "Recording 1000 steps took {:?}, expected < 500ms",
            duration
        );
    }

    #[test]
    fn test_replay_many_steps() {
        let mut trajectory = Trajectory::new("session-1", "Large trajectory");
        for i in 0..1000 {
            trajectory.add_step(TrajectoryStep::new(i + 1, ActionType::Query, format!("Step {}", i)));
        }

        let mut replayer = TrajectoryReplayer::new(trajectory);

        let start = Instant::now();
        while replayer.next_step().is_some() {}
        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 100,
            "Replaying 1000 steps took {:?}, expected < 100ms",
            duration
        );
    }

    #[test]
    fn test_storage_many_trajectories() {
        let mut storage = TrajectoryStorage::new();

        let start = Instant::now();
        for i in 0..1000 {
            let mut trajectory = Trajectory::new(format!("session-{}", i), format!("Task {}", i));
            trajectory.complete(TrajectoryOutcome::Success);
            storage.store(trajectory);
        }
        let duration = start.elapsed();

        assert_eq!(storage.count(), 1000);
        assert!(
            duration.as_millis() < 500,
            "Storing 1000 trajectories took {:?}, expected < 500ms",
            duration
        );
    }

    #[test]
    fn test_filter_performance() {
        let mut storage = TrajectoryStorage::new();

        for i in 0..1000 {
            let outcome = if i % 2 == 0 { TrajectoryOutcome::Success } else { TrajectoryOutcome::Failure };
            let mut trajectory = Trajectory::new(format!("session-{}", i % 10), format!("Task {}", i))
                .with_tags(vec![format!("tag-{}", i % 5)]);
            trajectory.complete(outcome);
            storage.store(trajectory);
        }

        let start = Instant::now();
        let successful = storage.get_successful();
        let session_0 = storage.get_by_session("session-0");
        let tag_0 = storage.get_by_tag("tag-0");
        let duration = start.elapsed();

        assert_eq!(successful.len(), 500);
        assert_eq!(session_0.len(), 100);
        assert_eq!(tag_0.len(), 200);
        assert!(
            duration.as_millis() < 100,
            "Filtering took {:?}, expected < 100ms",
            duration
        );
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
        /// Property: Step numbers are always sequential starting from 1.
        #[test]
        fn prop_step_numbers_sequential(step_count in 1usize..50usize) {
            let mut recorder = TrajectoryRecorder::new();
            recorder.start_trajectory("session", "task");

            for _ in 0..step_count {
                recorder.record_step(ActionType::Query, "input").unwrap();
            }

            let trajectory = recorder.current().unwrap();
            for (i, step) in trajectory.steps.iter().enumerate() {
                prop_assert_eq!(step.step_number, i + 1);
            }
        }

        /// Property: Total duration equals sum of step durations.
        #[test]
        fn prop_total_duration_is_sum(durations in proptest::collection::vec(0u64..1000u64, 1..20)) {
            let mut recorder = TrajectoryRecorder::new();
            recorder.start_trajectory("session", "task");

            let expected_total: u64 = durations.iter().sum();

            for (i, &duration) in durations.iter().enumerate() {
                recorder.record_step_with_output(
                    ActionType::Query,
                    format!("Step {}", i),
                    format!("Output {}", i),
                    duration,
                ).unwrap();
            }

            let trajectory = recorder.current().unwrap();
            prop_assert_eq!(trajectory.total_duration_ms, expected_total);
        }

        /// Property: Completed trajectory always has outcome and completed_at.
        #[test]
        fn prop_completed_has_outcome(outcome_idx in 0usize..6usize) {
            let outcomes = [
                TrajectoryOutcome::Success,
                TrajectoryOutcome::PartialSuccess,
                TrajectoryOutcome::Failure,
                TrajectoryOutcome::Timeout,
                TrajectoryOutcome::Cancelled,
                TrajectoryOutcome::Unknown,
            ];

            let mut recorder = TrajectoryRecorder::new();
            recorder.start_trajectory("session", "task");
            recorder.record_step(ActionType::Query, "step").unwrap();

            let trajectory = recorder.complete_trajectory(outcomes[outcome_idx]).unwrap();

            prop_assert!(trajectory.outcome.is_some());
            prop_assert!(trajectory.completed_at.is_some());
            prop_assert!(trajectory.reward.is_some());
        }

        /// Property: Replay progress is always in [0, 1].
        #[test]
        fn prop_replay_progress_bounded(step_count in 0usize..50usize) {
            let mut trajectory = Trajectory::new("session", "task");
            for i in 0..step_count {
                trajectory.add_step(TrajectoryStep::new(i + 1, ActionType::Query, format!("Step {}", i)));
            }

            let mut replayer = TrajectoryReplayer::new(trajectory);

            let initial_progress = replayer.progress();
            prop_assert!(initial_progress >= 0.0 && initial_progress <= 1.0);

            while !replayer.is_complete() {
                replayer.next_step();
                let progress = replayer.progress();
                prop_assert!(progress >= 0.0 && progress <= 1.0);
            }
        }

        /// Property: Reward is always clamped to [0, 1].
        #[test]
        fn prop_reward_clamped(reward in -10.0f32..10.0f32) {
            let mut recorder = TrajectoryRecorder::new();
            recorder.start_trajectory("session", "task");
            recorder.record_step(ActionType::Query, "step").unwrap();

            let trajectory = recorder.complete_trajectory_with_reward(
                TrajectoryOutcome::Success,
                reward,
            ).unwrap();

            let actual_reward = trajectory.reward.unwrap();
            prop_assert!(actual_reward >= 0.0 && actual_reward <= 1.0);
        }
    }
}
