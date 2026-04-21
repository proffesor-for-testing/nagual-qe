//! OSpipe - Rust-native activity ingestion pipeline.
//!
//! OSpipe replaces direct Screenpipe SQLite access with a configurable pipeline
//! that includes:
//!
//! - **PII Safety Gate**: Detect and redact/reject sensitive data before storage
//! - **Sliding Window Deduplication**: Cosine similarity-based dedup within time windows
//! - **Multi-Dimension Embeddings**: Support for 128-dim (nagual native) and 384-dim
//! - **Query Router**: Route queries to semantic, keyword, temporal, or hybrid search
//!
//! # Architecture
//!
//! ```text
//! Screenpipe (capture)     OSpipe Pipeline (Rust-native)
//! localhost:3030           ┌─────────────────────────────┐
//!      │                   │ 1. PII Gate (redact/reject) │
//!      │ raw_sql()         │ 2. Sliding Window Dedup     │
//!      └──────────────────▶│ 3. 384-dim Embedding        │
//!                          │ 4. Query Router             │
//!                          └──────────┬──────────────────┘
//!                                     │
//!                          ┌──────────┴──────────┐
//!                          ▼                     ▼
//!                     SQLite              PostgreSQL
//!                    (nagual.db)          (ruvector)
//! ```
//!
//! # Usage
//!
//! ```bash
//! # Legacy ingest (unchanged)
//! nagual activity ingest --since 4h
//!
//! # OSpipe ingest with PII protection + dedup
//! nagual activity ingest --ospipe --since 4h
//!
//! # OSpipe with custom policy
//! nagual activity ingest --ospipe --pii-policy warn --dedup-threshold 0.85
//! ```

mod config;
mod dedup;
mod pii_gate;
mod pipeline;
mod query_router;

#[allow(unused_imports)]
pub use config::{EmbeddingDim, IngestConfig, OSpipeConfig};
#[allow(unused_imports)]
pub use dedup::{DedupResult, DedupStats, SlidingWindowDedup};
#[allow(unused_imports)]
pub use pii_gate::{PiiGate, PiiGateResult, PiiPolicy};
#[allow(unused_imports)]
pub use pipeline::{IngestResult, OSpipeClient, OSpipePipeline};
#[allow(unused_imports)]
pub use query_router::{QueryMode, QueryRouter, SearchResult};

#[cfg(test)]
mod tests;
