//! GOAP Planning Module
//!
//! Goal-Oriented Action Planning (GOAP) for nagual.
//! Transforms natural language goals into executable action sequences.
//!
//! # Overview
//!
//! GOAP uses A* search to find optimal paths through action space:
//! - World State: Current beliefs about knowledge/codebase
//! - Goals: Desired end states
//! - Actions: Operations with preconditions and effects
//! - Plans: Sequences of actions achieving goals
//!
//! # Usage
//!
//! ```rust,ignore
//! use nagual::planning::{GOAPPlanner, Goal, WorldState, default_actions};
//!
//! let planner = GOAPPlanner::new(default_actions());
//! let current = WorldState::default();
//! let goal = Goal::from_description("improve test coverage");
//! let plan = planner.plan(&current, &goal)?;
//!
//! for step in &plan.actions {
//!     println!("Step {}: {}", step.step, step.action.name);
//! }
//! ```

mod types;
mod planner;
mod actions;
mod executor;
mod storage;
mod parser;
pub mod workflow;

pub use types::*;
pub use planner::{GOAPPlanner, PlanningError};
pub use actions::default_actions;
pub use executor::{PlanExecutor, ExecutionContext, ExecutionError, ReplanConfig, ReplanResult};
pub use storage::PlanStorage;
pub use parser::GoalParser;
