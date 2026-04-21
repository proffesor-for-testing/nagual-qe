//! Coherence Gate
//!
//! Verifies belief consistency before allowing pattern storage.
//! Inspired by energy-based belief systems and agentic-qe's coherence gates.
//!
//! Provides:
//! - Belief extraction from patterns
//! - Contradiction detection via similarity + semantic analysis
//! - Coherence energy calculation
//! - Configurable conflict resolution recommendations

mod types;
mod engine;

pub mod scoring;

pub use types::*;
pub use engine::*;
pub use scoring::*;
