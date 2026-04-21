//! Meta-Learning Module
//!
//! Provides higher-order learning capabilities:
//! - **EWC++**: Prevents catastrophic forgetting by protecting important patterns
//! - **Transfer Learning**: Enables knowledge transfer between related domains
//! - **Learning Rate Adaptation**: Dynamically adjusts learning based on performance
//! - **Pattern Generalization**: Abstracts similar patterns into templates
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Meta-Learning System                      │
//! │                                                              │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
//! │  │  EWC Engine  │  │   Transfer   │  │  Learning Rate   │  │
//! │  │              │  │    Engine    │  │     Adapter      │  │
//! │  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
//! │         │                 │                   │            │
//! │         └─────────────────┼───────────────────┘            │
//! │                           │                                │
//! │               ┌───────────▼───────────┐                    │
//! │               │   MetaLearningEngine  │                    │
//! │               │   (orchestrator)      │                    │
//! │               └───────────────────────┘                    │
//! └─────────────────────────────────────────────────────────────┘
//! ```

mod ewc;
mod transfer;
pub mod types;

pub use ewc::EwcEngine;
pub use transfer::TransferEngine;
pub use types::*;

use std::sync::Arc;

use chrono::Utc;
use parking_lot::RwLock;
use tracing::{debug, info};

use crate::db::SqliteDb;
use crate::error::Result;

/// Main orchestrator for the meta-learning system
pub struct MetaLearningEngine {
    /// EWC engine for catastrophic forgetting prevention
    pub ewc: EwcEngine,
    /// Transfer learning engine
    pub transfer: TransferEngine,
    /// Learning rate configurations per domain
    learning_rates: Arc<RwLock<std::collections::HashMap<String, LearningRateConfig>>>,
    /// Configuration
    config: MetaLearningConfig,
    /// Database for persistence
    db: Option<Arc<SqliteDb>>,
    /// Combined statistics
    stats: Arc<RwLock<MetaLearningStats>>,
}

impl MetaLearningEngine {
    /// Create a new meta-learning engine
    pub fn new(config: MetaLearningConfig) -> Self {
        Self {
            ewc: EwcEngine::new(config.ewc.clone()),
            transfer: TransferEngine::new(),
            learning_rates: Arc::new(RwLock::new(std::collections::HashMap::new())),
            config,
            db: None,
            stats: Arc::new(RwLock::new(MetaLearningStats::default())),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(MetaLearningConfig::default())
    }

    /// Create with database connection
    pub fn with_db(config: MetaLearningConfig, db: Arc<SqliteDb>) -> Self {
        let mut engine = Self::new(config);
        engine.db = Some(db);
        engine
    }

    /// Initialize the meta-learning system
    pub async fn initialize(&self) -> Result<()> {
        // Initialize transfer relationships
        self.transfer.initialize_common_transfers();

        // Load persisted data if database available
        if let Some(ref db) = self.db {
            self.load_from_db(db).await?;
        }

        info!("Meta-learning engine initialized");
        Ok(())
    }

    /// Get or create learning rate config for a domain
    pub fn get_learning_rate(&self, domain: &str) -> f64 {
        self.learning_rates
            .read()
            .get(domain)
            .map(|c| c.current_rate)
            .unwrap_or(0.1) // Default
    }

    /// Record performance and adapt learning rate
    pub fn record_domain_performance(&self, domain: &str, success_rate: f64) {
        let mut rates = self.learning_rates.write();

        let config = rates
            .entry(domain.to_string())
            .or_insert_with(|| LearningRateConfig::new(domain));

        config.record_performance(success_rate);

        let mut stats = self.stats.write();
        stats.rate_adjustments += 1;

        debug!(
            domain = %domain,
            success_rate = success_rate,
            new_rate = config.current_rate,
            "Adapted learning rate"
        );
    }

    /// Apply EWC-protected reward update
    pub fn protected_update(
        &self,
        pattern_id: &str,
        current_reward: f64,
        proposed_reward: f64,
        domain: &str,
    ) -> f64 {
        // Get domain-specific learning rate
        let learning_rate = self.get_learning_rate(domain);

        // Scale the proposed change by learning rate
        let change = proposed_reward - current_reward;
        let scaled_change = change * learning_rate;
        let scaled_proposed = current_reward + scaled_change;

        // Apply EWC protection
        self.ewc.protected_reward_update(pattern_id, current_reward, scaled_proposed)
    }

    /// Update pattern importance after an outcome
    pub fn record_outcome(
        &self,
        pattern_id: &str,
        domain: &str,
        _success: bool,
        success_count: u32,
        total_count: u32,
    ) {
        // Build outcomes array for Fisher calculation
        let outcomes: Vec<bool> = (0..total_count)
            .map(|i| i < success_count)
            .collect();

        self.ewc.update_importance(pattern_id, success_count, total_count, &outcomes);

        // Update domain performance
        let success_rate = success_count as f64 / total_count.max(1) as f64;
        self.record_domain_performance(domain, success_rate);
    }

    /// Find patterns to transfer from related domains
    pub fn suggest_transfers(&self, target_domain: &str) -> Vec<(String, f64)> {
        if !self.config.transfer_enabled {
            return Vec::new();
        }

        self.transfer.suggest_transfer_candidates(target_domain, 0.4)
    }

    /// Record a transfer attempt
    pub fn record_transfer(&self, source_domain: &str, target_domain: &str, success: bool) {
        self.transfer.record_transfer(
            source_domain,
            target_domain,
            success,
            None,
            None,
            None,
        );
    }

    /// Run optimization cycle (called during dream cycle)
    pub async fn optimize(&self) -> Result<OptimizationResult> {
        info!("Running meta-learning optimization");

        let mut result = OptimizationResult::default();

        // 1. Decay old Fisher information
        self.ewc.decay_fisher_info();
        result.fisher_decayed = true;

        // 2. Update statistics
        let ewc_stats = self.ewc.stats();
        let transfer_stats = self.transfer.stats();

        let mut stats = self.stats.write();
        stats.protected_patterns = ewc_stats.protected_patterns;
        stats.forgetting_prevented = ewc_stats.forgetting_prevented;
        stats.successful_transfers = transfer_stats.successful_transfers;
        stats.failed_transfers = transfer_stats.failed_transfers;
        stats.last_optimization = Some(Utc::now());

        result.stats = stats.clone();

        // 3. Persist if database available
        if let Some(ref db) = self.db {
            self.save_to_db(db).await?;
            result.persisted = true;
        }

        info!(
            protected = result.stats.protected_patterns,
            forgetting_prevented = result.stats.forgetting_prevented,
            "Meta-learning optimization complete"
        );

        Ok(result)
    }

    /// Get current statistics
    pub fn stats(&self) -> MetaLearningStats {
        self.stats.read().clone()
    }

    /// Get configuration
    pub fn config(&self) -> &MetaLearningConfig {
        &self.config
    }

    /// Load data from database
    async fn load_from_db(&self, db: &SqliteDb) -> Result<()> {
        // Create tables if they don't exist
        db.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS meta_learning_importance (
                pattern_id TEXT PRIMARY KEY,
                importance REAL NOT NULL,
                fisher_info REAL NOT NULL,
                success_count INTEGER NOT NULL,
                total_count INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meta_learning_transfers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_domain TEXT NOT NULL,
                target_domain TEXT NOT NULL,
                transfer_coefficient REAL NOT NULL,
                successful_transfers INTEGER NOT NULL,
                failed_transfers INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(source_domain, target_domain)
            );

            CREATE TABLE IF NOT EXISTS meta_learning_rates (
                domain TEXT PRIMARY KEY,
                base_rate REAL NOT NULL,
                current_rate REAL NOT NULL,
                prior_alpha REAL NOT NULL,
                prior_beta REAL NOT NULL
            );
            "#,
        )
        .await?;

        // Load importance data
        let importance_rows: Vec<(String, f64, f64, u32, u32, String)> = db
            .query(
                "SELECT pattern_id, importance, fisher_info, success_count, total_count, updated_at
                 FROM meta_learning_importance",
                &[],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .await?;

        let importance_data: Vec<PatternImportance> = importance_rows
            .into_iter()
            .filter_map(|(id, imp, fisher, succ, total, updated)| {
                chrono::DateTime::parse_from_rfc3339(&updated)
                    .ok()
                    .map(|dt| PatternImportance {
                        pattern_id: id,
                        importance: imp,
                        fisher_info: fisher,
                        success_count: succ,
                        total_count: total,
                        updated_at: dt.with_timezone(&Utc),
                    })
            })
            .collect();

        self.ewc.load_importance(importance_data);

        // Load transfer data
        let transfer_rows: Vec<(String, String, f64, u32, u32, String)> = db
            .query(
                "SELECT source_domain, target_domain, transfer_coefficient,
                        successful_transfers, failed_transfers, updated_at
                 FROM meta_learning_transfers",
                &[],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .await?;

        let transfer_data: Vec<DomainTransfer> = transfer_rows
            .into_iter()
            .filter_map(|(src, tgt, coef, succ, fail, updated)| {
                chrono::DateTime::parse_from_rfc3339(&updated)
                    .ok()
                    .map(|dt| DomainTransfer {
                        source_domain: src,
                        target_domain: tgt,
                        transfer_coefficient: coef,
                        successful_transfers: succ,
                        failed_transfers: fail,
                        pattern_mappings: Vec::new(),
                        updated_at: dt.with_timezone(&Utc),
                    })
            })
            .collect();

        self.transfer.load_transfers(transfer_data);

        info!("Loaded meta-learning data from database");
        Ok(())
    }

    /// Save data to database
    async fn save_to_db(&self, db: &SqliteDb) -> Result<()> {
        // Save importance data
        for importance in self.ewc.export_importance() {
            db.execute(
                r#"INSERT OR REPLACE INTO meta_learning_importance
                   (pattern_id, importance, fisher_info, success_count, total_count, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)"#,
                &[
                    &importance.pattern_id,
                    &importance.importance.to_string(),
                    &importance.fisher_info.to_string(),
                    &importance.success_count.to_string(),
                    &importance.total_count.to_string(),
                    &importance.updated_at.to_rfc3339(),
                ],
            )
            .await?;
        }

        // Save transfer data
        for transfer in self.transfer.export_transfers() {
            db.execute(
                r#"INSERT OR REPLACE INTO meta_learning_transfers
                   (source_domain, target_domain, transfer_coefficient,
                    successful_transfers, failed_transfers, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)"#,
                &[
                    &transfer.source_domain,
                    &transfer.target_domain,
                    &transfer.transfer_coefficient.to_string(),
                    &transfer.successful_transfers.to_string(),
                    &transfer.failed_transfers.to_string(),
                    &transfer.updated_at.to_rfc3339(),
                ],
            )
            .await?;
        }

        info!("Saved meta-learning data to database");
        Ok(())
    }
}

/// Result of an optimization cycle
#[derive(Debug, Clone, Default)]
pub struct OptimizationResult {
    /// Whether Fisher decay was applied
    pub fisher_decayed: bool,
    /// Whether data was persisted
    pub persisted: bool,
    /// Current statistics
    pub stats: MetaLearningStats,
}
