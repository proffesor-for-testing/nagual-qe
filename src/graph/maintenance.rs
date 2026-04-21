//! Edge Maintenance and Scheduled Jobs.
//!
//! Implements periodic maintenance of pattern edges including:
//! - **Edge Pruning**: Remove weak edges (strength < 0.1 AND age > 90 days)
//! - **Co-Retrieval Edge Creation**: Periodic check for new co-retrieval edges
//! - **Scheduled Maintenance**: Daily jobs at 3 AM using tokio-cron-scheduler
//!
//! # Design
//!
//! The `EdgeMaintenanceJob` runs periodic tasks:
//! - Prunes weak, old edges to prevent graph bloat
//! - Creates co-retrieval edges for frequently co-retrieved patterns
//! - Logs all operations to audit trail
//!
//! # Pruning Criteria
//!
//! Edges are pruned if ALL of these conditions are true:
//! - Strength < 0.1
//! - Age > 90 days
//! - Auto-created (manual edges are preserved)

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPool;
use sqlx::Row;
use tokio::sync::mpsc;
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::auto_edges::{AutoEdgeCreator, AutoEdgeResult};
use crate::error::{NagualError, Result};

/// Configuration for edge maintenance.
#[derive(Debug, Clone)]
pub struct EdgeMaintenanceConfig {
    /// Minimum strength for an edge to be kept (below this = weak).
    pub weak_edge_threshold: f64,
    /// Minimum age in days for an edge to be prunable.
    pub min_age_days: i64,
    /// Maximum number of edges to prune per batch.
    pub prune_batch_size: usize,
    /// Maximum number of co-retrieval edges to create per run.
    pub coretrieval_batch_size: usize,
    /// Whether to prune only auto-created edges.
    pub prune_only_auto_created: bool,
    /// Cron expression for scheduling (default: "0 0 3 * * *" = 3 AM daily).
    pub cron_expression: String,
}

impl Default for EdgeMaintenanceConfig {
    fn default() -> Self {
        Self {
            weak_edge_threshold: 0.1,
            min_age_days: 90,
            prune_batch_size: 1000,
            coretrieval_batch_size: 100,
            prune_only_auto_created: true,
            cron_expression: "0 0 3 * * *".to_string(), // 3 AM daily
        }
    }
}

impl EdgeMaintenanceConfig {
    /// Create config with custom weak edge threshold.
    pub fn with_weak_threshold(mut self, threshold: f64) -> Self {
        self.weak_edge_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Create config with custom minimum age.
    pub fn with_min_age_days(mut self, days: i64) -> Self {
        self.min_age_days = days.max(0);
        self
    }

    /// Create config with custom cron expression.
    pub fn with_cron(mut self, cron: impl Into<String>) -> Self {
        self.cron_expression = cron.into();
        self
    }

    /// Create config for testing (immediate execution, smaller batches).
    pub fn for_testing() -> Self {
        Self {
            weak_edge_threshold: 0.1,
            min_age_days: 0, // Prune immediately for testing
            prune_batch_size: 100,
            coretrieval_batch_size: 10,
            prune_only_auto_created: true,
            cron_expression: "*/10 * * * * *".to_string(), // Every 10 seconds
        }
    }
}

/// Result of edge pruning operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruneResult {
    /// Number of edges pruned.
    pub edges_pruned: usize,
    /// Number of edges that matched criteria but couldn't be pruned.
    pub errors: usize,
    /// IDs of pruned edges.
    pub pruned_edge_ids: Vec<Uuid>,
    /// Time taken in milliseconds.
    pub duration_ms: u64,
    /// Job ID for this maintenance run.
    pub job_id: String,
}

impl PruneResult {
    /// Create an empty result.
    pub fn empty(job_id: &str) -> Self {
        Self {
            edges_pruned: 0,
            errors: 0,
            pruned_edge_ids: Vec::new(),
            duration_ms: 0,
            job_id: job_id.to_string(),
        }
    }

    /// Check if any edges were pruned.
    pub fn has_changes(&self) -> bool {
        self.edges_pruned > 0
    }
}

/// Result of a complete edge maintenance run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMaintenanceResult {
    /// Job identifier for this run.
    pub job_id: String,
    /// Start time of maintenance.
    pub started_at: DateTime<Utc>,
    /// End time of maintenance.
    pub completed_at: DateTime<Utc>,
    /// Total duration in milliseconds.
    pub total_duration_ms: u64,
    /// Edge pruning result.
    pub prune_result: PruneResult,
    /// Co-retrieval edge creation result.
    pub coretrieval_result: AutoEdgeResult,
    /// Whether the job completed successfully.
    pub success: bool,
    /// Error message if failed.
    pub error_message: Option<String>,
}

/// Edge pruning information for audit logging.
#[derive(Debug, Clone)]
struct PrunedEdgeInfo {
    id: Uuid,
    source_pattern: String,
    target_pattern: String,
    edge_type: String,
    strength: f64,
    created_at: DateTime<Utc>,
}

/// Edge maintenance job for periodic cleanup.
pub struct EdgeMaintenanceJob {
    pool: PgPool,
    config: EdgeMaintenanceConfig,
    auto_edge_creator: AutoEdgeCreator,
}

impl EdgeMaintenanceJob {
    /// Create a new edge maintenance job with default configuration.
    pub fn new(pool: PgPool) -> Self {
        let auto_edge_creator = AutoEdgeCreator::new(pool.clone());
        Self {
            pool,
            config: EdgeMaintenanceConfig::default(),
            auto_edge_creator,
        }
    }

    /// Create a new edge maintenance job with custom configuration.
    pub fn with_config(pool: PgPool, config: EdgeMaintenanceConfig) -> Self {
        let auto_edge_creator = AutoEdgeCreator::new(pool.clone());
        Self {
            pool,
            config,
            auto_edge_creator,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &EdgeMaintenanceConfig {
        &self.config
    }

    /// Run a complete maintenance cycle.
    ///
    /// This performs:
    /// 1. Edge pruning (weak + old edges)
    /// 2. Co-retrieval edge creation
    /// 3. Audit logging
    pub async fn run(&self) -> Result<EdgeMaintenanceResult> {
        let job_id = format!("maint_{}", Utc::now().timestamp_millis());
        let started_at = Utc::now();

        info!(job_id = job_id, "Starting edge maintenance job");

        // Run edge pruning
        let prune_result = self.prune_weak_edges(&job_id).await?;

        // Run co-retrieval edge creation
        let coretrieval_result = self
            .auto_edge_creator
            .check_and_create_coretrieval_edges(self.config.coretrieval_batch_size)
            .await?;

        let completed_at = Utc::now();
        let total_duration_ms = (completed_at - started_at).num_milliseconds() as u64;

        let result = EdgeMaintenanceResult {
            job_id: job_id.clone(),
            started_at,
            completed_at,
            total_duration_ms,
            prune_result,
            coretrieval_result,
            success: true,
            error_message: None,
        };

        info!(
            job_id = job_id,
            edges_pruned = result.prune_result.edges_pruned,
            edges_created = result.coretrieval_result.edges_created,
            duration_ms = total_duration_ms,
            "Edge maintenance job completed"
        );

        Ok(result)
    }

    /// Prune weak edges that meet pruning criteria.
    ///
    /// # Pruning Criteria
    ///
    /// Edges are pruned if ALL conditions are true:
    /// - Strength < weak_edge_threshold (default: 0.1)
    /// - Age > min_age_days (default: 90 days)
    /// - Auto-created (if prune_only_auto_created is true)
    ///
    /// # Arguments
    ///
    /// * `job_id` - Unique identifier for this maintenance job
    ///
    /// # Returns
    ///
    /// Result containing count of pruned edges and their IDs.
    pub async fn prune_weak_edges(&self, job_id: &str) -> Result<PruneResult> {
        let start = std::time::Instant::now();

        // Calculate cutoff date
        let cutoff_date = Utc::now() - chrono::Duration::days(self.config.min_age_days);

        // Find edges to prune
        let auto_created_filter = if self.config.prune_only_auto_created {
            "AND auto_created = true"
        } else {
            ""
        };

        let query = format!(
            r#"
            SELECT
                id, source_pattern, target_pattern, edge_type::text,
                strength, created_at
            FROM pattern_edges
            WHERE strength < $1
                AND created_at < $2
                {}
            ORDER BY created_at ASC
            LIMIT $3
            "#,
            auto_created_filter
        );

        let edges_to_prune: Vec<PrunedEdgeInfo> = sqlx::query(&query)
            .bind(self.config.weak_edge_threshold)
            .bind(cutoff_date)
            .bind(self.config.prune_batch_size as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| NagualError::internal(format!("Failed to find edges to prune: {}", e)))?
            .into_iter()
            .map(|row| PrunedEdgeInfo {
                id: row.get("id"),
                source_pattern: row.get("source_pattern"),
                target_pattern: row.get("target_pattern"),
                edge_type: row.get("edge_type"),
                strength: row.get("strength"),
                created_at: row.get("created_at"),
            })
            .collect();

        if edges_to_prune.is_empty() {
            debug!(job_id = job_id, "No weak edges found to prune");
            return Ok(PruneResult {
                duration_ms: start.elapsed().as_millis() as u64,
                ..PruneResult::empty(job_id)
            });
        }

        let mut result = PruneResult::empty(job_id);

        // Prune edges in batches with audit logging
        for edge in edges_to_prune {
            match self.prune_single_edge(&edge, job_id).await {
                Ok(()) => {
                    result.edges_pruned += 1;
                    result.pruned_edge_ids.push(edge.id);
                }
                Err(e) => {
                    warn!(
                        job_id = job_id,
                        edge_id = %edge.id,
                        error = %e,
                        "Failed to prune edge"
                    );
                    result.errors += 1;
                }
            }
        }

        result.duration_ms = start.elapsed().as_millis() as u64;

        if result.edges_pruned > 0 {
            info!(
                job_id = job_id,
                edges_pruned = result.edges_pruned,
                errors = result.errors,
                duration_ms = result.duration_ms,
                "Pruned weak edges"
            );
        }

        Ok(result)
    }

    /// Prune a single edge with audit logging.
    async fn prune_single_edge(&self, edge: &PrunedEdgeInfo, job_id: &str) -> Result<()> {
        // Start a transaction
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| NagualError::internal(format!("Failed to start transaction: {}", e)))?;

        // Log to audit trail first
        sqlx::query(
            r#"
            INSERT INTO edge_audit_log (
                edge_id, source_pattern, target_pattern, edge_type,
                operation, old_strength, new_strength, reason, job_id
            ) VALUES ($1, $2, $3, $4::edge_type, 'pruned', $5, NULL, $6, $7)
            "#,
        )
        .bind(edge.id)
        .bind(&edge.source_pattern)
        .bind(&edge.target_pattern)
        .bind(&edge.edge_type)
        .bind(edge.strength)
        .bind(format!(
            "Weak edge pruned: strength {:.4} < {} and age > {} days",
            edge.strength,
            self.config.weak_edge_threshold,
            self.config.min_age_days
        ))
        .bind(job_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| NagualError::internal(format!("Failed to log edge pruning: {}", e)))?;

        // Delete the edge
        sqlx::query("DELETE FROM pattern_edges WHERE id = $1")
            .bind(edge.id)
            .execute(&mut *tx)
            .await
            .map_err(|e| NagualError::internal(format!("Failed to delete edge: {}", e)))?;

        // Commit transaction
        tx.commit()
            .await
            .map_err(|e| NagualError::internal(format!("Failed to commit transaction: {}", e)))?;

        debug!(
            edge_id = %edge.id,
            source = edge.source_pattern,
            target = edge.target_pattern,
            strength = edge.strength,
            job_id = job_id,
            "Pruned weak edge"
        );

        Ok(())
    }

    /// Get statistics about pruneable edges.
    pub async fn get_prune_stats(&self) -> Result<PruneStats> {
        let cutoff_date = Utc::now() - chrono::Duration::days(self.config.min_age_days);

        let row = sqlx::query(
            r#"
            SELECT
                COUNT(*) as total_weak,
                COUNT(*) FILTER (WHERE auto_created = true) as auto_weak,
                AVG(strength) FILTER (WHERE strength < $1) as avg_weak_strength
            FROM pattern_edges
            WHERE strength < $1
                AND created_at < $2
            "#,
        )
        .bind(self.config.weak_edge_threshold)
        .bind(cutoff_date)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| NagualError::internal(format!("Failed to get prune stats: {}", e)))?;

        Ok(PruneStats {
            total_weak_edges: row.get::<i64, _>("total_weak") as usize,
            auto_created_weak_edges: row.get::<i64, _>("auto_weak") as usize,
            avg_weak_strength: row.get::<Option<f64>, _>("avg_weak_strength").unwrap_or(0.0),
        })
    }
}

/// Statistics about pruneable edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruneStats {
    /// Total number of weak edges meeting age criteria.
    pub total_weak_edges: usize,
    /// Number of auto-created weak edges.
    pub auto_created_weak_edges: usize,
    /// Average strength of weak edges.
    pub avg_weak_strength: f64,
}

/// Handle for managing the maintenance scheduler.
pub struct MaintenanceSchedulerHandle {
    scheduler: JobScheduler,
    shutdown_tx: mpsc::Sender<()>,
    job_id: Uuid,
}

impl MaintenanceSchedulerHandle {
    /// Stop the maintenance scheduler.
    pub async fn stop(mut self) -> Result<()> {
        let _ = self.shutdown_tx.send(()).await;
        self.scheduler
            .shutdown()
            .await
            .map_err(|e| NagualError::internal(format!("Failed to shutdown scheduler: {}", e)))?;
        info!("Maintenance scheduler stopped");
        Ok(())
    }

    /// Get the job ID.
    pub fn job_id(&self) -> Uuid {
        self.job_id
    }
}

/// Start the edge maintenance scheduler.
///
/// Schedules daily maintenance job at 3 AM (configurable via config).
///
/// # Arguments
///
/// * `pool` - PostgreSQL connection pool
///
/// # Returns
///
/// Handle to control the scheduler.
///
/// # Example
///
/// ```rust,ignore
/// let handle = start_maintenance_scheduler(pool).await?;
///
/// // Later, to stop:
/// handle.stop().await?;
/// ```
pub async fn start_maintenance_scheduler(pool: PgPool) -> Result<MaintenanceSchedulerHandle> {
    start_maintenance_scheduler_with_config(pool, EdgeMaintenanceConfig::default()).await
}

/// Start the edge maintenance scheduler with custom configuration.
pub async fn start_maintenance_scheduler_with_config(
    pool: PgPool,
    config: EdgeMaintenanceConfig,
) -> Result<MaintenanceSchedulerHandle> {
    let scheduler = JobScheduler::new()
        .await
        .map_err(|e| NagualError::internal(format!("Failed to create scheduler: {}", e)))?;

    let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

    let cron_expr = config.cron_expression.clone();
    let maintenance_job = EdgeMaintenanceJob::with_config(pool.clone(), config);
    let maintenance_job = Arc::new(maintenance_job);

    let job = Job::new_async(cron_expr.as_str(), move |uuid, _lock| {
        let job = maintenance_job.clone();
        Box::pin(async move {
            info!(job_uuid = %uuid, "Scheduled maintenance job starting");
            match job.run().await {
                Ok(result) => {
                    info!(
                        job_uuid = %uuid,
                        job_id = result.job_id,
                        edges_pruned = result.prune_result.edges_pruned,
                        edges_created = result.coretrieval_result.edges_created,
                        "Scheduled maintenance completed"
                    );
                }
                Err(e) => {
                    error!(
                        job_uuid = %uuid,
                        error = %e,
                        "Scheduled maintenance failed"
                    );
                }
            }
        })
    })
    .map_err(|e| NagualError::internal(format!("Failed to create job: {}", e)))?;

    let job_id = job.guid();

    scheduler
        .add(job)
        .await
        .map_err(|e| NagualError::internal(format!("Failed to add job: {}", e)))?;

    scheduler
        .start()
        .await
        .map_err(|e| NagualError::internal(format!("Failed to start scheduler: {}", e)))?;

    info!(
        job_id = %job_id,
        cron = cron_expr,
        "Edge maintenance scheduler started"
    );

    // Spawn shutdown listener
    let mut scheduler_clone = scheduler.clone();
    tokio::spawn(async move {
        shutdown_rx.recv().await;
        let _ = scheduler_clone.shutdown().await;
    });

    Ok(MaintenanceSchedulerHandle {
        scheduler,
        shutdown_tx,
        job_id,
    })
}

/// Run maintenance immediately (for testing or manual trigger).
pub async fn run_maintenance_now(pool: PgPool) -> Result<EdgeMaintenanceResult> {
    let job = EdgeMaintenanceJob::new(pool);
    job.run().await
}

/// Run maintenance with custom configuration.
pub async fn run_maintenance_with_config(
    pool: PgPool,
    config: EdgeMaintenanceConfig,
) -> Result<EdgeMaintenanceResult> {
    let job = EdgeMaintenanceJob::with_config(pool, config);
    job.run().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_maintenance_config_defaults() {
        let config = EdgeMaintenanceConfig::default();
        assert_eq!(config.weak_edge_threshold, 0.1);
        assert_eq!(config.min_age_days, 90);
        assert_eq!(config.prune_batch_size, 1000);
        assert!(config.prune_only_auto_created);
        assert_eq!(config.cron_expression, "0 0 3 * * *");
    }

    #[test]
    fn test_edge_maintenance_config_builder() {
        let config = EdgeMaintenanceConfig::default()
            .with_weak_threshold(0.2)
            .with_min_age_days(30)
            .with_cron("0 0 4 * * *");

        assert_eq!(config.weak_edge_threshold, 0.2);
        assert_eq!(config.min_age_days, 30);
        assert_eq!(config.cron_expression, "0 0 4 * * *");
    }

    #[test]
    fn test_edge_maintenance_config_clamping() {
        let config = EdgeMaintenanceConfig::default().with_weak_threshold(1.5);
        assert_eq!(config.weak_edge_threshold, 1.0);

        let config = EdgeMaintenanceConfig::default().with_weak_threshold(-0.5);
        assert_eq!(config.weak_edge_threshold, 0.0);

        let config = EdgeMaintenanceConfig::default().with_min_age_days(-10);
        assert_eq!(config.min_age_days, 0);
    }

    #[test]
    fn test_prune_result_empty() {
        let result = PruneResult::empty("test_job");
        assert_eq!(result.edges_pruned, 0);
        assert!(!result.has_changes());
        assert_eq!(result.job_id, "test_job");
    }

    #[test]
    fn test_testing_config() {
        let config = EdgeMaintenanceConfig::for_testing();
        assert_eq!(config.min_age_days, 0); // Immediate for testing
        assert_eq!(config.prune_batch_size, 100);
        assert!(config.cron_expression.contains("*/10")); // Every 10 seconds
    }
}
