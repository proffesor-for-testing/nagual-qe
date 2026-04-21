//! Nagual - Self-Learning Agentic System
//!
//! A dual-write persistence system with SQLite (local) and PostgreSQL (cloud),
//! featuring ReasoningBank pattern storage, SONA learning, and Brier-calibrated predictions.
//!
//! # Quick Start
//!
//! The easiest way to use Nagual is through the unified API:
//!
//! ```rust,ignore
//! use nagual::api::{Nagual, NagualConfig};
//! use nagual::learning::Outcome;
//!
//! // Initialize Nagual
//! let nagual = Nagual::new(NagualConfig::default()).await?;
//!
//! // Store knowledge
//! let id = nagual.knowledge.store(
//!     "How to handle database timeouts",
//!     "Implement retry with exponential backoff",
//!     "database.resilience"
//! ).await?;
//!
//! // Search for patterns
//! let results = nagual.patterns.search("database optimization").await?;
//!
//! // Record learning outcomes
//! nagual.learning.record_outcome(&id, Outcome::Success, None).await?;
//!
//! // Query the knowledge graph
//! let neighbors = nagual.graph.query(&id, nagual::graph::Direction::Outgoing).await?;
//!
//! // Run a backup
//! let backup = nagual.sync.backup().await?;
//! ```
//!
//! # Modules
//!
//! ## High-Level API
//!
//! - [`api`] - Unified API for all Nagual functionality (recommended entry point)
//!
//! ## Core Components
//!
//! - [`db`] - Database abstraction layer (SQLite + PostgreSQL)
//! - [`error`] - Error types and resilience patterns
//! - [`injection`] - E_nagual attention bias injection for vendor LLMs
//! - [`reasoning_bank`] - Pattern storage and retrieval for self-learning
//! - [`learning`] - SONA learning loop and A/B testing infrastructure
//! - [`graph`] - Graph algorithms (GNN pressure propagation)
//! - [`ml`] - Machine learning and embedding operations
//! - [`prediction`] - Prediction engine with Brier-calibrated forecasts
//! - [`profdag`] - ProfDAG: Probabilistic Forecasting DAG for knowledge graphs
//! - [`router`] - FastGRNN-based vendor router for LLM selection
//! - [`sync`] - GCloud Storage sync with incremental backups and retention
//!
//! ## Infrastructure
//!
//! - [`cli`] - Command-line interface handlers
//! - [`events`] - Event bus for publish-subscribe messaging
//! - [`health`] - Health checking and degradation
//! - [`mcp`] - MCP (Model Context Protocol) tool integration
//! - [`migration`] - Schema migration system
//! - [`security`] - Security, privacy, and audit logging
//!
//! # Configuration
//!
//! Use [`api::NagualConfigBuilder`] for fine-grained configuration:
//!
//! ```rust,ignore
//! let config = NagualConfig::builder()
//!     .sqlite_path("my-database.db")
//!     .postgres_url("postgres://localhost/nagual")
//!     .embedding_dim(128)
//!     .gcloud("my-bucket", "my-project")
//!     .retention_days(30)
//!     .build()?;
//!
//! let nagual = Nagual::new(config).await?;
//! ```
//!
//! # Event Bus
//!
//! The event bus enables components to communicate through domain events:
//!
//! ```ignore
//! use nagual::events::{EventBus, NagualEvent};
//!
//! let bus = EventBus::new();
//! let mut receiver = bus.subscribe();
//!
//! // Publish events from any component
//! bus.publish(NagualEvent::pattern_stored("id", "domain")).await?;
//! ```
//!
//! # MCP Integration
//!
//! MCP tools allow LLMs to interact with Nagual:
//!
//! ```ignore
//! use nagual::mcp::{McpRegistry, NagualContext};
//!
//! let registry = McpRegistry::with_all_tools();
//! let context = NagualContext::new(storage, learner, event_bus);
//!
//! let result = registry.execute("nagual_store_pattern", &input, &context).await?;
//! ```

// High-level API (recommended entry point)
pub mod api;

// Core modules
pub mod cli;
pub mod cloud;
pub mod coherence;
pub mod constitution;
pub mod db;
pub mod drift;
pub mod dream;
pub mod error;
pub mod events;
pub mod graph;
pub mod health;
pub mod injection;
pub mod introspection;
pub mod learning;
pub mod mcp;
pub mod migration;
pub mod ml;
pub mod observability;
pub mod planning;
pub mod prediction;
pub mod research;
pub mod profdag;
pub mod reasoning_bank;
pub mod router;
pub mod security;
pub mod sync;

pub mod lineage;
pub mod witness;
pub mod delta;
pub mod epoch;
pub mod tiering;
pub mod agent_views;

#[cfg(feature = "serve")]
pub mod serve;

// Re-exports for convenience
pub use error::{NagualError, Result};

// Re-export API types for easy access
pub use api::{
    HealthStatus, Nagual, NagualConfig, NagualConfigBuilder,
    // Knowledge API
    KnowledgeApi, KnowledgeItem, KnowledgeSearchResult,
    // Learning API
    LearningApi, ConsolidationResult, ImprovementResult, InsightsResult,
    // Sync API
    SyncApi, BackupStatus, RestoreStatus, SyncStatus,
    // Graph API
    GraphApi, GraphQueryResult,
    // Patterns API
    PatternsApi, PatternSearchResult, PatternStatsResult,
    // ProfDAG API
    ProfDAGApi,
    // Router API
    RouterApi,
    // Injection API
    InjectionApi,
};

// Re-export KOS API
pub use api::KosApi;

// Re-export event types
pub use events::{EventBus, EventReceiver, NagualEvent, PatternChanges, SyncType};

// Re-export MCP types
pub use mcp::{McpRegistry, NagualContext, ToolDefinition, ToolResult};

// Re-export injection types for E_nagual
pub use injection::{
    ENagual, ENagualBuilder, ENagualConfig, ContextBuilder, ContextConfig, Provider,
    InjectionContext, format_for_anthropic, format_for_openai, format_for_local,
    // Attention surgery for open-weight models
    AttentionBias, AttentionSurgery, AttentionSurgeryConfig, BiasMethod, ModelConfig,
    AttentionState, ENagualHook, HookRegistry, ModelHook,
};

// Re-export router types for vendor selection
pub use router::{
    ComplexityEstimator, ComplexityFeatures, ComplexityScore, EstimatorConfig,
    FastGRNN, FastGRNNConfig, FastGRNNWeights, GRNNCell,
    FallbackChain, RoutingDecision, RoutingMetrics, Vendor, VendorConfig, VendorRouter,
    VendorSelector, VendorStatus, RouterConfig, RouterError, RouterResult,
};
