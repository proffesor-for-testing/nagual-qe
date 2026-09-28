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

mod engine;
mod types;

pub mod scoring;

pub use engine::*;
pub use scoring::*;
pub use types::*;
