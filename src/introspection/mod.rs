//! Strange Loop Introspection
//!
//! Self-referential system awareness for nagual, inspired by Hofstadter's
//! concept and agentic-qe's SwarmSelfModel.
//!
//! Provides:
//! - Self-model of pattern health, domain coverage, and trends
//! - Vulnerability detection (decay, gaps, degradation)
//! - Actionable recommendations for self-improvement
//! - GOAP integration for automated improvement cycles

mod self_model;
mod engine;

pub use self_model::*;
pub use engine::*;
