//! A* Planning Algorithm for GOAP

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use thiserror::Error;
use tracing::{debug, info, instrument, warn};

use super::types::*;

/// Errors that can occur during planning
#[derive(Debug, Error)]
pub enum PlanningError {
    #[error("No plan found to achieve goal")]
    NoPlanFound,

    #[error("Maximum iterations ({0}) exceeded")]
    MaxIterationsExceeded(usize),

    #[error("Goal has no conditions defined")]
    EmptyGoal,

    #[error("No actions available")]
    NoActions,

    #[error("Goal parsing failed: {0}")]
    GoalParseError(String),

    #[error("Action not found: {0}")]
    ActionNotFound(String),

    #[error("Planning error: {0}")]
    Other(String),
}

/// A node in the A* search tree
#[derive(Debug, Clone)]
struct PlanNode {
    state: WorldState,
    actions: Vec<Action>,
    g_cost: f64, // Cost so far
    h_cost: f64, // Heuristic (estimated remaining cost)
}

impl PlanNode {
    fn f_cost(&self) -> f64 {
        self.g_cost + self.h_cost
    }
}

impl PartialEq for PlanNode {
    fn eq(&self, other: &Self) -> bool {
        self.f_cost() == other.f_cost()
    }
}

impl Eq for PlanNode {}

impl Ord for PlanNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (lower f_cost = higher priority)
        other.f_cost().partial_cmp(&self.f_cost())
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for PlanNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Goal-Oriented Action Planner using A* search
pub struct GOAPPlanner {
    actions: Vec<Action>,
    config: PlanningConfig,
}

impl GOAPPlanner {
    /// Create a new planner with the given actions
    pub fn new(actions: Vec<Action>) -> Self {
        Self {
            actions,
            config: PlanningConfig::default(),
        }
    }

    /// Create a planner with custom configuration
    pub fn with_config(actions: Vec<Action>, config: PlanningConfig) -> Self {
        Self { actions, config }
    }

    /// Add an action to the planner
    pub fn add_action(&mut self, action: Action) {
        self.actions.push(action);
    }

    /// Get all available actions
    pub fn actions(&self) -> &[Action] {
        &self.actions
    }

    /// Plan a sequence of actions to achieve the goal from current state
    #[instrument(skip(self, current), fields(goal_name = %goal.name))]
    pub fn plan(&self, current: &WorldState, goal: &Goal) -> Result<Plan, PlanningError> {
        if goal.conditions.is_empty() {
            return Err(PlanningError::EmptyGoal);
        }

        if self.actions.is_empty() {
            return Err(PlanningError::NoActions);
        }

        info!("Starting GOAP planning for goal: {}", goal.name);
        debug!("Goal conditions: {:?}", goal.conditions);

        // Check if goal is already satisfied
        if self.goal_satisfied(current, goal) {
            info!("Goal already satisfied, returning empty plan");
            return Ok(Plan::new(goal.clone(), vec![]));
        }

        let mut open_set = BinaryHeap::new();
        let mut visited = HashSet::new();

        let initial_node = PlanNode {
            state: current.clone(),
            actions: vec![],
            g_cost: 0.0,
            h_cost: self.heuristic(current, goal),
        };

        open_set.push(initial_node);

        let mut iterations = 0;

        while let Some(node) = open_set.pop() {
            iterations += 1;

            if iterations > self.config.max_iterations {
                warn!("Max iterations exceeded: {}", self.config.max_iterations);
                return Err(PlanningError::MaxIterationsExceeded(self.config.max_iterations));
            }

            // Check if goal is satisfied
            if self.goal_satisfied(&node.state, goal) {
                info!(
                    "Plan found with {} actions after {} iterations",
                    node.actions.len(),
                    iterations
                );
                return Ok(self.build_plan(goal.clone(), node.actions));
            }

            // Generate state hash for visited check
            let state_hash = self.hash_state(&node.state);
            if visited.contains(&state_hash) {
                continue;
            }
            visited.insert(state_hash);

            // Check plan length limit
            if node.actions.len() >= self.config.max_plan_length {
                debug!("Max plan length reached, skipping expansion");
                continue;
            }

            // Expand: try each applicable action
            for action in &self.actions {
                if self.preconditions_met(&node.state, action) {
                    let new_state = self.apply_effects(&node.state, action);
                    let new_actions = {
                        let mut a = node.actions.clone();
                        a.push(action.clone());
                        a
                    };

                    let action_cost = if self.config.enable_action_costs {
                        action.cost
                    } else {
                        1.0
                    };

                    let new_node = PlanNode {
                        state: new_state.clone(),
                        actions: new_actions,
                        g_cost: node.g_cost + action_cost,
                        h_cost: self.heuristic(&new_state, goal),
                    };

                    open_set.push(new_node);
                }
            }
        }

        warn!("No plan found after {} iterations", iterations);
        Err(PlanningError::NoPlanFound)
    }

    /// Heuristic: count unsatisfied goal conditions
    fn heuristic(&self, state: &WorldState, goal: &Goal) -> f64 {
        let unsatisfied = goal.conditions.iter()
            .filter(|c| !self.condition_met(state, c))
            .count();
        unsatisfied as f64
    }

    /// Check if all goal conditions are satisfied
    fn goal_satisfied(&self, state: &WorldState, goal: &Goal) -> bool {
        goal.conditions.iter().all(|c| self.condition_met(state, c))
    }

    /// Check if all preconditions of an action are met
    fn preconditions_met(&self, state: &WorldState, action: &Action) -> bool {
        action.preconditions.iter().all(|c| self.condition_met(state, c))
    }

    /// Check if a single condition is met
    fn condition_met(&self, state: &WorldState, condition: &Condition) -> bool {
        match state.propositions.get(&condition.proposition) {
            Some(value) => {
                match (&condition.operator, value, &condition.value) {
                    (ConditionOp::Equals, a, b) => a == b,
                    (ConditionOp::NotEquals, a, b) => a != b,
                    (ConditionOp::GreaterThan, StateValue::Number(a), StateValue::Number(b)) => a > b,
                    (ConditionOp::LessThan, StateValue::Number(a), StateValue::Number(b)) => a < b,
                    (ConditionOp::GreaterThanOrEqual, StateValue::Number(a), StateValue::Number(b)) => a >= b,
                    (ConditionOp::LessThanOrEqual, StateValue::Number(a), StateValue::Number(b)) => a <= b,
                    (ConditionOp::Contains, StateValue::Text(a), StateValue::Text(b)) => a.contains(b.as_str()),
                    _ => false,
                }
            }
            None => {
                // Proposition not in state - check if condition is for "false" or default
                matches!((&condition.operator, &condition.value),
                    (ConditionOp::Equals, StateValue::Bool(false)) |
                    (ConditionOp::NotEquals, StateValue::Bool(true)))
            }
        }
    }

    /// Apply action effects to create a new state
    fn apply_effects(&self, state: &WorldState, action: &Action) -> WorldState {
        let mut new_state = state.clone();

        for effect in &action.effects {
            match &effect.operation {
                EffectOp::Set => {
                    new_state.propositions.insert(
                        effect.proposition.clone(),
                        effect.value.clone(),
                    );
                }
                EffectOp::Increment => {
                    if let StateValue::Number(delta) = &effect.value {
                        let current = new_state.get_number(&effect.proposition).unwrap_or(0.0);
                        new_state.propositions.insert(
                            effect.proposition.clone(),
                            StateValue::Number(current + delta),
                        );
                    }
                }
                EffectOp::Decrement => {
                    if let StateValue::Number(delta) = &effect.value {
                        let current = new_state.get_number(&effect.proposition).unwrap_or(0.0);
                        new_state.propositions.insert(
                            effect.proposition.clone(),
                            StateValue::Number(current - delta),
                        );
                    }
                }
                EffectOp::Append => {
                    if let StateValue::Text(suffix) = &effect.value {
                        let current = match new_state.get(&effect.proposition) {
                            Some(StateValue::Text(t)) => t.clone(),
                            _ => String::new(),
                        };
                        new_state.propositions.insert(
                            effect.proposition.clone(),
                            StateValue::Text(format!("{}{}", current, suffix)),
                        );
                    }
                }
            }
        }

        new_state
    }

    /// Hash the world state for visited set
    fn hash_state(&self, state: &WorldState) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Sort keys for deterministic hashing
        let mut keys: Vec<_> = state.propositions.keys().collect();
        keys.sort();

        for key in keys {
            key.hash(&mut hasher);
            // Hash the value representation
            if let Some(value) = state.propositions.get(key) {
                format!("{:?}", value).hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Build a Plan from the action sequence
    fn build_plan(&self, goal: Goal, actions: Vec<Action>) -> Plan {
        let planned_actions: Vec<PlannedAction> = actions
            .into_iter()
            .enumerate()
            .map(|(i, action)| PlannedAction::new(action, i + 1))
            .collect();

        Plan::new(goal, planned_actions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_actions() -> Vec<Action> {
        vec![
            Action::new("identify_gaps", "Identify Knowledge Gaps")
                .with_effect(Effect::set_true("gaps_identified"))
                .with_effect(Effect::set_true("research_target_defined"))
                .with_cost(1.0),
            Action::new("research_topic", "Research Topic")
                .with_precondition(Condition::is_true("research_target_defined"))
                .with_effect(Effect::set_true("has_research_results"))
                .with_cost(2.0),
            Action::new("store_pattern", "Store Pattern")
                .with_precondition(Condition::is_true("has_research_results"))
                .with_effect(Effect::set_true("pattern_stored"))
                .with_effect(Effect::increment("pattern_count", 1.0))
                .with_cost(1.0),
        ]
    }

    #[test]
    fn test_simple_plan() {
        let planner = GOAPPlanner::new(test_actions());
        let current = WorldState::new();

        let goal = Goal::new("Store a pattern", "Store knowledge")
            .with_condition(Condition::is_true("pattern_stored"));

        let plan = planner.plan(&current, &goal).expect("Should find a plan");

        assert_eq!(plan.step_count(), 3);
        assert_eq!(plan.actions[0].action.id, "identify_gaps");
        assert_eq!(plan.actions[1].action.id, "research_topic");
        assert_eq!(plan.actions[2].action.id, "store_pattern");
    }

    #[test]
    fn test_partial_state() {
        let planner = GOAPPlanner::new(test_actions());

        let mut current = WorldState::new();
        current.set_bool("research_target_defined", true);

        let goal = Goal::new("Store a pattern", "")
            .with_condition(Condition::is_true("pattern_stored"));

        let plan = planner.plan(&current, &goal).expect("Should find a plan");

        // Should skip identify_gaps since research_target_defined is already true
        assert_eq!(plan.step_count(), 2);
        assert_eq!(plan.actions[0].action.id, "research_topic");
        assert_eq!(plan.actions[1].action.id, "store_pattern");
    }

    #[test]
    fn test_goal_already_satisfied() {
        let planner = GOAPPlanner::new(test_actions());

        let mut current = WorldState::new();
        current.set_bool("pattern_stored", true);

        let goal = Goal::new("Already done", "")
            .with_condition(Condition::is_true("pattern_stored"));

        let plan = planner.plan(&current, &goal).expect("Should succeed");

        assert_eq!(plan.step_count(), 0); // Empty plan
    }

    #[test]
    fn test_no_plan_possible() {
        let planner = GOAPPlanner::new(test_actions());
        let current = WorldState::new();

        let goal = Goal::new("Impossible", "")
            .with_condition(Condition::is_true("impossible_condition"));

        let result = planner.plan(&current, &goal);

        assert!(matches!(result, Err(PlanningError::NoPlanFound)));
    }

    #[test]
    fn test_empty_goal() {
        let planner = GOAPPlanner::new(test_actions());
        let current = WorldState::new();
        let goal = Goal::new("Empty", "");

        let result = planner.plan(&current, &goal);

        assert!(matches!(result, Err(PlanningError::EmptyGoal)));
    }
}
