//! Core types for GOAP planning

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// World state is a set of propositions with values
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorldState {
    pub propositions: HashMap<String, StateValue>,
}

impl WorldState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: impl Into<String>, value: StateValue) {
        self.propositions.insert(key.into(), value);
    }

    pub fn set_bool(&mut self, key: impl Into<String>, value: bool) {
        self.propositions.insert(key.into(), StateValue::Bool(value));
    }

    pub fn set_number(&mut self, key: impl Into<String>, value: f64) {
        self.propositions.insert(key.into(), StateValue::Number(value));
    }

    pub fn set_text(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.propositions.insert(key.into(), StateValue::Text(value.into()));
    }

    pub fn get(&self, key: &str) -> Option<&StateValue> {
        self.propositions.get(key)
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.propositions.get(key) {
            Some(StateValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_number(&self, key: &str) -> Option<f64> {
        match self.propositions.get(key) {
            Some(StateValue::Number(v)) => Some(*v),
            _ => None,
        }
    }
}

/// Value types for world state propositions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StateValue {
    Bool(bool),
    Number(f64),
    Text(String),
}

impl std::fmt::Display for StateValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateValue::Bool(b) => write!(f, "{}", b),
            StateValue::Number(n) => write!(f, "{}", n),
            StateValue::Text(t) => write!(f, "{}", t),
        }
    }
}

/// A goal defines desired world state conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: String,
    pub name: String,
    pub description: String,
    pub conditions: Vec<Condition>,
    pub priority: u8, // 1-10, higher = more important
}

impl Goal {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            description: description.into(),
            conditions: Vec::new(),
            priority: 5,
        }
    }

    pub fn with_condition(mut self, condition: Condition) -> Self {
        self.conditions.push(condition);
        self
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.min(10).max(1);
        self
    }

    /// Create a goal from a natural language description
    pub fn from_description(description: impl Into<String>) -> Self {
        let desc = description.into();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: desc.clone(),
            description: desc,
            conditions: Vec::new(),
            priority: 5,
        }
    }
}

/// A condition that must be true
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub proposition: String,
    pub operator: ConditionOp,
    pub value: StateValue,
}

impl Condition {
    pub fn equals(proposition: impl Into<String>, value: StateValue) -> Self {
        Self {
            proposition: proposition.into(),
            operator: ConditionOp::Equals,
            value,
        }
    }

    pub fn is_true(proposition: impl Into<String>) -> Self {
        Self::equals(proposition, StateValue::Bool(true))
    }

    pub fn is_false(proposition: impl Into<String>) -> Self {
        Self::equals(proposition, StateValue::Bool(false))
    }

    pub fn greater_than(proposition: impl Into<String>, value: f64) -> Self {
        Self {
            proposition: proposition.into(),
            operator: ConditionOp::GreaterThan,
            value: StateValue::Number(value),
        }
    }

    pub fn less_than(proposition: impl Into<String>, value: f64) -> Self {
        Self {
            proposition: proposition.into(),
            operator: ConditionOp::LessThan,
            value: StateValue::Number(value),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConditionOp {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Contains,
}

/// An action that can change world state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub id: String,
    pub name: String,
    pub description: String,
    pub preconditions: Vec<Condition>,
    pub effects: Vec<Effect>,
    pub cost: f64, // Lower = preferred
    pub duration_estimate_seconds: Option<u64>,
    pub category: ActionCategory,
}

impl Action {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            preconditions: Vec::new(),
            effects: Vec::new(),
            cost: 1.0,
            duration_estimate_seconds: None,
            category: ActionCategory::General,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn with_precondition(mut self, condition: Condition) -> Self {
        self.preconditions.push(condition);
        self
    }

    pub fn with_effect(mut self, effect: Effect) -> Self {
        self.effects.push(effect);
        self
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    pub fn with_duration(mut self, seconds: u64) -> Self {
        self.duration_estimate_seconds = Some(seconds);
        self
    }

    pub fn with_category(mut self, category: ActionCategory) -> Self {
        self.category = category;
        self
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ActionCategory {
    General,
    Research,
    Knowledge,
    Analysis,
    Testing,
    Improvement,
    Consolidation,
}

/// An effect that changes world state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Effect {
    pub proposition: String,
    pub operation: EffectOp,
    pub value: StateValue,
}

impl Effect {
    pub fn set(proposition: impl Into<String>, value: StateValue) -> Self {
        Self {
            proposition: proposition.into(),
            operation: EffectOp::Set,
            value,
        }
    }

    pub fn set_true(proposition: impl Into<String>) -> Self {
        Self::set(proposition, StateValue::Bool(true))
    }

    pub fn set_false(proposition: impl Into<String>) -> Self {
        Self::set(proposition, StateValue::Bool(false))
    }

    pub fn increment(proposition: impl Into<String>, amount: f64) -> Self {
        Self {
            proposition: proposition.into(),
            operation: EffectOp::Increment,
            value: StateValue::Number(amount),
        }
    }

    pub fn decrement(proposition: impl Into<String>, amount: f64) -> Self {
        Self {
            proposition: proposition.into(),
            operation: EffectOp::Decrement,
            value: StateValue::Number(amount),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EffectOp {
    Set,
    Increment,
    Decrement,
    Append,
}

/// A plan is a sequence of actions achieving a goal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub id: String,
    pub goal: Goal,
    pub actions: Vec<PlannedAction>,
    pub total_cost: f64,
    pub estimated_duration_seconds: Option<u64>,
    pub created_at: DateTime<Utc>,
    pub status: PlanStatus,
    pub current_step: usize,
}

impl Plan {
    pub fn new(goal: Goal, actions: Vec<PlannedAction>) -> Self {
        let total_cost = actions.iter().map(|a| a.action.cost).sum();
        let estimated_duration = actions
            .iter()
            .filter_map(|a| a.action.duration_estimate_seconds)
            .sum::<u64>();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            goal,
            total_cost,
            estimated_duration_seconds: if estimated_duration > 0 {
                Some(estimated_duration)
            } else {
                None
            },
            actions,
            created_at: Utc::now(),
            status: PlanStatus::Ready,
            current_step: 0,
        }
    }

    pub fn step_count(&self) -> usize {
        self.actions.len()
    }

    pub fn is_complete(&self) -> bool {
        matches!(self.status, PlanStatus::Completed)
    }

    pub fn current_action(&self) -> Option<&PlannedAction> {
        self.actions.get(self.current_step)
    }

    pub fn completed_steps(&self) -> usize {
        self.actions.iter().filter(|a| matches!(a.status, ActionStatus::Completed)).count()
    }

    pub fn progress_percentage(&self) -> f64 {
        if self.actions.is_empty() {
            100.0
        } else {
            (self.completed_steps() as f64 / self.actions.len() as f64) * 100.0
        }
    }
}

/// A planned action within a plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedAction {
    pub action: Action,
    pub step: usize,
    pub status: ActionStatus,
    pub result: Option<ActionResult>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

impl PlannedAction {
    pub fn new(action: Action, step: usize) -> Self {
        Self {
            action,
            step,
            status: ActionStatus::Pending,
            result: None,
            started_at: None,
            completed_at: None,
        }
    }

    pub fn start(&mut self) {
        self.status = ActionStatus::InProgress;
        self.started_at = Some(Utc::now());
    }

    pub fn complete(&mut self, result: ActionResult) {
        self.status = if result.success {
            ActionStatus::Completed
        } else {
            ActionStatus::Failed
        };
        self.result = Some(result);
        self.completed_at = Some(Utc::now());
    }

    pub fn skip(&mut self, reason: impl Into<String>) {
        self.status = ActionStatus::Skipped;
        self.result = Some(ActionResult {
            success: false,
            output: None,
            error: Some(reason.into()),
            duration_ms: 0,
        });
        self.completed_at = Some(Utc::now());
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PlanStatus {
    Planning,
    Ready,
    Executing,
    Paused,
    Completed,
    Failed,
    Cancelled,
    Replanning,
}

impl std::fmt::Display for PlanStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanStatus::Planning => write!(f, "planning"),
            PlanStatus::Ready => write!(f, "ready"),
            PlanStatus::Executing => write!(f, "executing"),
            PlanStatus::Paused => write!(f, "paused"),
            PlanStatus::Completed => write!(f, "completed"),
            PlanStatus::Failed => write!(f, "failed"),
            PlanStatus::Cancelled => write!(f, "cancelled"),
            PlanStatus::Replanning => write!(f, "replanning"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl std::fmt::Display for ActionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActionStatus::Pending => write!(f, "pending"),
            ActionStatus::InProgress => write!(f, "in_progress"),
            ActionStatus::Completed => write!(f, "completed"),
            ActionStatus::Failed => write!(f, "failed"),
            ActionStatus::Skipped => write!(f, "skipped"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    pub success: bool,
    pub output: Option<String>,
    pub error: Option<String>,
    pub duration_ms: u64,
}

impl ActionResult {
    pub fn success(output: Option<String>, duration_ms: u64) -> Self {
        Self {
            success: true,
            output,
            error: None,
            duration_ms,
        }
    }

    pub fn failure(error: impl Into<String>, duration_ms: u64) -> Self {
        Self {
            success: false,
            output: None,
            error: Some(error.into()),
            duration_ms,
        }
    }
}

/// Configuration for GOAP planning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningConfig {
    pub max_iterations: usize,
    pub max_plan_length: usize,
    pub enable_action_costs: bool,
    pub enable_duration_estimates: bool,
}

impl Default for PlanningConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            max_plan_length: 20,
            enable_action_costs: true,
            enable_duration_estimates: true,
        }
    }
}
