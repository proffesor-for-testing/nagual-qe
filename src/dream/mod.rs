//! Dream Cycle Module
//!
//! Background maintenance system that runs during idle periods.
//! Inspired by sleep-based memory consolidation, this module:
//!
//! - **Consolidates** similar patterns (merge, dedupe, archive)
//! - **Refreshes** stale patterns via targeted research
//! - **Calibrates** prediction scores for better forecasting
//! - **Activates** spreading activation through the pattern graph
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                   Dream Cycle Scheduler                  │
//! │                                                          │
//! │  - Monitors idle state                                   │
//! │  - Triggers dream phases                                 │
//! │  - Respects resource budgets                             │
//! │  - Tracks metrics                                        │
//! └──────────────────────────┬──────────────────────────────┘
//!                            │
//!            ┌───────────────┼───────────────┐
//!            │               │               │
//!     ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
//!     │  Consolidate │ │   Refresh   │ │  Calibrate  │
//!     │    Phase     │ │    Phase    │ │    Phase    │
//!     └──────────────┘ └─────────────┘ └─────────────┘
//!            │               │               │
//!            └───────────────┼───────────────┘
//!                            │
//!                     ┌──────▼──────┐
//!                     │   Activate   │
//!                     │    Phase     │
//!                     └──────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use nagual::dream::{DreamCycle, DreamConfig};
//!
//! // Create dream cycle engine
//! let mut dream = DreamCycle::new(db, DreamConfig::default());
//!
//! // Check if ready to run
//! if dream.is_idle() {
//!     let result = dream.run_cycle().await?;
//!     println!("Processed {} items", result.total_items_processed());
//! }
//!
//! // Get status
//! let status = dream.status();
//! println!("State: {}, Total cycles: {}", status.state, status.total_cycles);
//! ```
//!
//! # CLI Commands
//!
//! ```bash
//! # Trigger dream cycle manually
//! nagual dream
//!
//! # Show status
//! nagual dream status
//!
//! # Configure
//! nagual dream config --idle-threshold 600 --max-duration 60
//!
//! # View history
//! nagual dream history --limit 10
//! ```
//!
//! # References
//!
//! - [Sleep and Memory Consolidation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3768102/)
//! - [Spreading Activation](https://en.wikipedia.org/wiki/Spreading_activation)
//! - ADR-033: Dream Cycle

pub mod types;
pub mod engine;

pub use types::*;
pub use engine::DreamCycle;
