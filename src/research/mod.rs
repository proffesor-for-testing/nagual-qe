//! Research Swarm module for autonomous knowledge acquisition
//!
//! This module implements ADR-030: Research Swarm, providing:
//! - Ephemeral research agents (WebSearch, DocFetch, CodeAnalysis, KnowledgeBase)
//! - MaTTS (Memory-aware Test-Time Scaling) for quality control
//! - Attention-weighted consensus aggregation
//! - Automatic pattern synthesis and storage
//!
//! # Example
//!
//! ```ignore
//! use nagual::research::{ResearchCoordinator, ResearchRequest, ResearchDepth};
//!
//! let coordinator = ResearchCoordinator::with_defaults(db);
//!
//! let request = ResearchRequest::new("Rust async error handling")
//!     .with_depth(ResearchDepth::Medium)
//!     .with_domain("rust.async");
//!
//! let result = coordinator.research(request).await?;
//! println!("Created {} patterns", result.patterns_created.len());
//! ```

pub mod agents;
pub mod coordinator;
pub mod matts;
pub mod types;

pub use agents::{AgentFactory, ResearchAgent};
pub use coordinator::{CoordinatorConfig, ResearchCoordinator, ResearchPlan};
pub use matts::{MaTTS, MaTTSConfig};
pub use types::{
    AgentType, ConsensusResult, PatternSummary, ResearchAction, ResearchBudget, ResearchDepth,
    ResearchFinding, ResearchRequest, ResearchResult, ResearchStep, ResearchStrategy,
    ResearchTrajectory,
};
