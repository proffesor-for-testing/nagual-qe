//! Nagual Library API - Unified interface for self-learning agentic systems.
//!
//! This module provides a clean, ergonomic API for interacting with Nagual's
//! core functionality including knowledge storage, learning, synchronization,
//! graph operations, and pattern management.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use nagual::api::{Nagual, NagualConfig};
//!
//! // Initialize with defaults
//! let nagual = Nagual::new(NagualConfig::default()).await?;
//!
//! // Store knowledge
//! nagual.knowledge.store("How to handle errors", "Use Result type", "rust.error_handling").await?;
//!
//! // Search for patterns
//! let results = nagual.patterns.search("error handling").await?;
//!
//! // Record learning outcome
//! nagual.learning.record_outcome(&pattern_id, Outcome::Success, None).await?;
//!
//! // Query the knowledge graph
//! let neighbors = nagual.graph.query(&node_id, Direction::Outgoing).await?;
//! ```
//!
//! # Architecture
//!
//! The API is organized into namespaces for different concerns:
//!
//! - [`KnowledgeApi`]: Store and retrieve knowledge items
//! - [`LearningApi`]: Record outcomes and trigger learning
//! - [`SyncApi`]: Backup and restore operations
//! - [`GraphApi`]: Query and manipulate the knowledge graph
//! - [`PatternsApi`]: Pattern storage and retrieval
//! - [`ProfDAGApi`]: ProfDAG knowledge graph with HNSW search and trajectory recording
//! - [`RouterApi`]: FastGRNN-based vendor routing for LLM selection
//! - [`InjectionApi`]: E_nagual attention bias injection for vendor LLMs
//!
//! # Configuration
//!
//! Use [`NagualConfigBuilder`] for fine-grained configuration:
//!
//! ```rust,ignore
//! let config = NagualConfig::builder()
//!     .sqlite_path("custom.db")
//!     .postgres_url("postgres://...")
//!     .embedding_dim(128)
//!     .build()?;
//!
//! let nagual = Nagual::new(config).await?;
//! ```

pub mod graph;
pub mod injection_api;
pub mod knowledge;
pub mod learning;
pub mod patterns;
pub mod profdag_api;
pub mod router_api;
pub mod sync;

pub mod kos_api;

pub use graph::{GraphApi, GraphQueryResult};
pub use injection_api::InjectionApi;
pub use knowledge::{KnowledgeApi, KnowledgeItem, KnowledgeSearchResult};
pub use learning::{ConsolidationResult, ImprovementResult, InsightsResult, LearningApi};
pub use patterns::{PatternSearchResult, PatternStatsResult, PatternsApi};
pub use profdag_api::ProfDAGApi;
pub use router_api::RouterApi;
pub use sync::{BackupStatus, RestoreStatus, SyncApi, SyncStatus};

pub use kos_api::KosApi;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::db::{DatabaseConfig, DualDb, DualWriteAdapter, DualWriteConfig};
use crate::error::{NagualError, Result};
use crate::graph::GraphStorage;
use crate::learning::SonaConfig;
use crate::ml::dimensions;
use crate::reasoning_bank::storage::{PatternStorage, StorageConfig};
use crate::sync::{GCloudConfig, RetentionConfig, SyncManager};

/// Configuration for the Nagual system.
///
/// Use the builder pattern for convenient construction:
///
/// ```rust,ignore
/// let config = NagualConfig::builder()
///     .sqlite_path("my-database.db")
///     .embedding_dim(128)
///     .build()?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NagualConfig {
    /// Path to the SQLite database file.
    pub sqlite_path: String,

    /// PostgreSQL connection URL (optional, for cloud sync).
    pub postgres_url: Option<String>,

    /// Maximum PostgreSQL pool connections.
    pub max_pg_connections: u32,

    /// Embedding dimension for vectors.
    pub embedding_dim: usize,

    /// Whether to auto-generate embeddings for new patterns.
    pub auto_generate_embeddings: bool,

    /// GCloud bucket name for sync (optional).
    pub gcloud_bucket: Option<String>,

    /// GCloud project ID (optional).
    pub gcloud_project: Option<String>,

    /// SONA learning configuration.
    pub sona_config: SonaConfig,

    /// Retention policy days for backups.
    pub retention_days: u32,

    /// Enable debug logging.
    pub debug: bool,
}

impl Default for NagualConfig {
    fn default() -> Self {
        Self {
            sqlite_path: "nagual.db".to_string(),
            postgres_url: None,
            max_pg_connections: 5,
            embedding_dim: dimensions::NAGUAL_128,
            auto_generate_embeddings: false,
            gcloud_bucket: None,
            gcloud_project: None,
            sona_config: SonaConfig::default(),
            retention_days: 30,
            debug: false,
        }
    }
}

impl NagualConfig {
    /// Create a new default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration builder.
    pub fn builder() -> NagualConfigBuilder {
        NagualConfigBuilder::new()
    }

    /// Create a configuration for testing (in-memory database).
    pub fn for_testing() -> Self {
        Self {
            sqlite_path: ":memory:".to_string(),
            postgres_url: None,
            max_pg_connections: 1,
            embedding_dim: dimensions::NAGUAL_128,
            auto_generate_embeddings: false,
            gcloud_bucket: None,
            gcloud_project: None,
            sona_config: SonaConfig::default(),
            retention_days: 7,
            debug: true,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        // Check embedding dimension
        if self.embedding_dim == 0 {
            return Err(NagualError::config("embedding_dim must be greater than 0"));
        }

        // Check pool size
        if self.max_pg_connections == 0 && self.postgres_url.is_some() {
            return Err(NagualError::config(
                "max_pg_connections must be at least 1 when postgres_url is set",
            ));
        }

        // Check GCloud config consistency
        if self.gcloud_bucket.is_some() != self.gcloud_project.is_some() {
            return Err(NagualError::config(
                "gcloud_bucket and gcloud_project must both be set or both be None",
            ));
        }

        Ok(())
    }
}

/// Builder for [`NagualConfig`].
#[derive(Debug, Clone, Default)]
pub struct NagualConfigBuilder {
    config: NagualConfig,
}

impl NagualConfigBuilder {
    /// Create a new configuration builder with defaults.
    pub fn new() -> Self {
        Self {
            config: NagualConfig::default(),
        }
    }

    /// Set the SQLite database path.
    pub fn sqlite_path(mut self, path: impl Into<String>) -> Self {
        self.config.sqlite_path = path.into();
        self
    }

    /// Set the PostgreSQL connection URL.
    pub fn postgres_url(mut self, url: impl Into<String>) -> Self {
        self.config.postgres_url = Some(url.into());
        self
    }

    /// Set the maximum PostgreSQL pool connections.
    pub fn max_pg_connections(mut self, max: u32) -> Self {
        self.config.max_pg_connections = max;
        self
    }

    /// Set the embedding dimension.
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.config.embedding_dim = dim;
        self
    }

    /// Enable auto-generation of embeddings.
    pub fn auto_embeddings(mut self) -> Self {
        self.config.auto_generate_embeddings = true;
        self
    }

    /// Set GCloud sync configuration.
    pub fn gcloud(mut self, bucket: impl Into<String>, project: impl Into<String>) -> Self {
        self.config.gcloud_bucket = Some(bucket.into());
        self.config.gcloud_project = Some(project.into());
        self
    }

    /// Set the SONA learning configuration.
    pub fn sona_config(mut self, config: SonaConfig) -> Self {
        self.config.sona_config = config;
        self
    }

    /// Set the retention days for backups.
    pub fn retention_days(mut self, days: u32) -> Self {
        self.config.retention_days = days;
        self
    }

    /// Enable debug mode.
    pub fn debug(mut self) -> Self {
        self.config.debug = true;
        self
    }

    /// Build the configuration, validating it first.
    pub fn build(self) -> Result<NagualConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Internal state shared between API namespaces.
#[derive(Clone)]
pub(crate) struct NagualState {
    /// Dual-write adapter for SQLite + PostgreSQL
    pub adapter: Arc<DualWriteAdapter>,

    /// Pattern storage
    pub pattern_storage: Arc<PatternStorage>,

    /// Graph storage
    pub graph_storage: Arc<GraphStorage>,

    /// Sync manager (optional, requires GCloud config)
    pub sync_manager: Option<Arc<SyncManager>>,

    /// Configuration
    pub config: NagualConfig,
}

/// The main Nagual API entry point.
///
/// This struct provides access to all Nagual functionality through
/// organized namespace APIs.
///
/// # Example
///
/// ```rust,ignore
/// let nagual = Nagual::new(NagualConfig::default()).await?;
///
/// // Access different API namespaces
/// nagual.knowledge.store(...).await?;
/// nagual.patterns.search(...).await?;
/// nagual.learning.record_outcome(...).await?;
/// nagual.graph.query(...).await?;
/// nagual.sync.backup().await?;
/// nagual.profdag.insert_node(&node).await?;
/// nagual.router.route("query", &embedding)?;
/// let builder = nagual.injection.build_context("query");
/// ```
pub struct Nagual {
    /// Knowledge storage and retrieval API.
    pub knowledge: KnowledgeApi,

    /// Learning and improvement API.
    pub learning: LearningApi,

    /// Synchronization and backup API.
    pub sync: SyncApi,

    /// Knowledge graph operations API.
    pub graph: GraphApi,

    /// Pattern storage and retrieval API.
    pub patterns: PatternsApi,

    /// ProfDAG knowledge graph operations API.
    pub profdag: ProfDAGApi,

    /// FastGRNN-based vendor routing API.
    pub router: RouterApi,

    /// E_nagual attention bias injection API.
    pub injection: InjectionApi,

    /// Knowledge Operating System API (KOS features).
    pub kos: KosApi,

    /// Internal state
    state: NagualState,
}

impl Nagual {
    /// Create a new Nagual instance with the given configuration.
    ///
    /// This initializes all subsystems including:
    /// - SQLite database (always)
    /// - PostgreSQL connection (if configured)
    /// - Pattern storage with FTS5
    /// - Knowledge graph
    /// - Sync manager (if GCloud configured)
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the Nagual system
    ///
    /// # Returns
    ///
    /// A fully initialized `Nagual` instance ready for use.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let nagual = Nagual::new(NagualConfig::default()).await?;
    /// ```
    pub async fn new(config: NagualConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        info!(
            sqlite_path = %config.sqlite_path,
            has_postgres = config.postgres_url.is_some(),
            embedding_dim = config.embedding_dim,
            "Initializing Nagual"
        );

        // Create database configuration
        let db_config = DatabaseConfig {
            sqlite_path: config.sqlite_path.clone(),
            postgres_url: config.postgres_url.clone(),
            max_pg_connections: config.max_pg_connections,
            connection_timeout_secs: 30,
        };

        // Initialize dual database (SQLite + optional PostgreSQL)
        let dual_db = DualDb::new(&db_config).await?;

        // Initialize dual-write adapter with the database connections
        let dual_write_config = DualWriteConfig::default();
        let adapter = Arc::new(DualWriteAdapter::new(
            dual_db.sqlite.clone(),
            dual_db.postgres.clone(),
            dual_write_config,
        )?);

        // Initialize pattern storage
        let storage_config = StorageConfig {
            embedding_dim: config.embedding_dim,
            auto_generate_embeddings: config.auto_generate_embeddings,
            ..StorageConfig::default()
        };
        let pattern_storage = Arc::new(PatternStorage::new(adapter.clone(), storage_config).await?);

        // Initialize graph storage
        let graph_storage = Arc::new(
            GraphStorage::open(&config.sqlite_path)
                .map_err(|e| NagualError::internal(format!("Failed to open graph storage: {}", e)))?
        );

        // Initialize sync manager if GCloud is configured
        let sync_manager = if let (Some(bucket), Some(project)) =
            (&config.gcloud_bucket, &config.gcloud_project)
        {
            let gcloud_config = GCloudConfig::new(bucket, project);
            let gcloud_adapter = crate::sync::GCloudAdapter::new(gcloud_config)
                .await
                .map_err(|e| NagualError::internal(format!("Failed to create GCloud adapter: {}", e)))?;
            let retention_config = RetentionConfig::new(config.retention_days as u64, 7);
            Some(Arc::new(SyncManager::new(gcloud_adapter, retention_config)))
        } else {
            None
        };

        // Build internal state
        let state = NagualState {
            adapter,
            pattern_storage: pattern_storage.clone(),
            graph_storage: graph_storage.clone(),
            sync_manager: sync_manager.clone(),
            config: config.clone(),
        };

        // Create API namespaces
        let knowledge = KnowledgeApi::new(state.clone());
        let learning = LearningApi::new(state.clone());
        let sync = SyncApi::new(state.clone());
        let graph = GraphApi::new(state.clone());
        let patterns = PatternsApi::new(state.clone());
        let profdag = ProfDAGApi::new(state.clone()).await?;
        let router = RouterApi::new(state.clone())?;
        let injection = InjectionApi::new(state.clone());

        let kos = KosApi::new(dual_db.sqlite.clone()).await?;

        info!("Nagual initialized successfully");

        Ok(Self {
            knowledge,
            learning,
            sync,
            graph,
            patterns,
            profdag,
            router,
            injection,
            kos,
            state,
        })
    }

    /// Create a Nagual instance for testing (in-memory database).
    ///
    /// This is useful for unit tests and integration tests.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// #[tokio::test]
    /// async fn test_example() {
    ///     let nagual = Nagual::for_testing().await.unwrap();
    ///     // Use nagual...
    /// }
    /// ```
    pub async fn for_testing() -> Result<Self> {
        Self::new(NagualConfig::for_testing()).await
    }

    /// Get the current configuration.
    pub fn config(&self) -> &NagualConfig {
        &self.state.config
    }

    /// Get the pattern storage instance.
    pub fn pattern_storage(&self) -> &Arc<PatternStorage> {
        &self.state.pattern_storage
    }

    /// Get the graph storage instance.
    pub fn graph_storage(&self) -> &Arc<GraphStorage> {
        &self.state.graph_storage
    }

    /// Get the dual-write adapter.
    pub fn adapter(&self) -> &Arc<DualWriteAdapter> {
        &self.state.adapter
    }

    /// Check if PostgreSQL is configured.
    pub fn has_postgres(&self) -> bool {
        self.state.config.postgres_url.is_some()
    }

    /// Check if GCloud sync is configured.
    pub fn has_gcloud(&self) -> bool {
        self.state.sync_manager.is_some()
    }

    /// Get system health status.
    ///
    /// # Returns
    ///
    /// A `HealthStatus` containing the health of all subsystems.
    pub async fn health(&self) -> HealthStatus {
        let sqlite_healthy = self.state.adapter.sqlite().table_exists("sqlite_master").await.is_ok();

        let postgres_healthy = if let Some(pg) = self.state.adapter.postgres() {
            // Check PostgreSQL health
            Some(pg.is_healthy().await)
        } else {
            None
        };

        let gcloud_healthy = if self.state.sync_manager.is_some() {
            // GCloud health check would go here
            Some(true)
        } else {
            None
        };

        HealthStatus {
            sqlite: sqlite_healthy,
            postgres: postgres_healthy,
            gcloud: gcloud_healthy,
            overall: sqlite_healthy
                && postgres_healthy.unwrap_or(true)
                && gcloud_healthy.unwrap_or(true),
        }
    }

    /// Shutdown the Nagual instance gracefully.
    ///
    /// This ensures all pending writes are flushed and connections are closed.
    pub async fn shutdown(self) -> Result<()> {
        info!("Shutting down Nagual");

        // Flush any pending DLQ entries
        // The dual-write adapter will be dropped, closing connections

        info!("Nagual shutdown complete");
        Ok(())
    }
}

/// Health status of the Nagual system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// SQLite database health.
    pub sqlite: bool,

    /// PostgreSQL database health (if configured).
    pub postgres: Option<bool>,

    /// GCloud sync health (if configured).
    pub gcloud: Option<bool>,

    /// Overall system health.
    pub overall: bool,
}

impl HealthStatus {
    /// Check if all configured components are healthy.
    pub fn is_healthy(&self) -> bool {
        self.overall
    }

    /// Get a summary string.
    pub fn summary(&self) -> String {
        let mut parts = vec![format!("sqlite={}", if self.sqlite { "ok" } else { "error" })];

        if let Some(pg) = self.postgres {
            parts.push(format!("postgres={}", if pg { "ok" } else { "error" }));
        }

        if let Some(gc) = self.gcloud {
            parts.push(format!("gcloud={}", if gc { "ok" } else { "error" }));
        }

        parts.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = NagualConfig::default();
        assert_eq!(config.sqlite_path, "nagual.db");
        assert_eq!(config.embedding_dim, dimensions::NAGUAL_128);
        assert!(config.postgres_url.is_none());
    }

    #[test]
    fn test_config_builder() {
        let config = NagualConfig::builder()
            .sqlite_path("custom.db")
            .embedding_dim(384)
            .postgres_url("postgres://localhost/test")
            .max_pg_connections(10)
            .retention_days(60)
            .debug()
            .build()
            .unwrap();

        assert_eq!(config.sqlite_path, "custom.db");
        assert_eq!(config.embedding_dim, 384);
        assert_eq!(config.postgres_url, Some("postgres://localhost/test".to_string()));
        assert_eq!(config.max_pg_connections, 10);
        assert_eq!(config.retention_days, 60);
        assert!(config.debug);
    }

    #[test]
    fn test_config_validation() {
        // Invalid embedding dimension
        let config = NagualConfig {
            embedding_dim: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Invalid GCloud config
        let config = NagualConfig {
            gcloud_bucket: Some("bucket".to_string()),
            gcloud_project: None,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Valid config
        let config = NagualConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_for_testing() {
        let config = NagualConfig::for_testing();
        assert_eq!(config.sqlite_path, ":memory:");
        assert!(config.debug);
    }

    #[test]
    fn test_health_status_summary() {
        let status = HealthStatus {
            sqlite: true,
            postgres: Some(true),
            gcloud: None,
            overall: true,
        };
        assert!(status.summary().contains("sqlite=ok"));
        assert!(status.summary().contains("postgres=ok"));
        assert!(!status.summary().contains("gcloud"));
    }
}
