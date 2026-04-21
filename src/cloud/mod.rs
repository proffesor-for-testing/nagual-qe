//! Cloud sync module for bidirectional pattern synchronization.
//!
//! Provides incremental push/pull between local SQLite and a remote
//! nagual serve instance, using `updated_at` timestamps for change detection.
//!
//! # Usage
//!
//! ```bash
//! nagual cloud push --token <token>
//! nagual cloud pull --token <token>
//! nagual cloud status --token <token>
//! ```

pub mod client;
pub mod pull;
pub mod push;
pub mod sync_state;
pub mod types;
