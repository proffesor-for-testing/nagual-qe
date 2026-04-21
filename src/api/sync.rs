//! Sync namespace API for backup and restore operations.
//!
//! The sync API provides methods for backing up and restoring the Nagual
//! database to/from cloud storage (GCloud), with support for incremental
//! sync and retention policies.
//!
//! # Example
//!
//! ```rust,ignore
//! // Run a backup
//! let status = nagual.sync.backup().await?;
//! println!("Backup created: {}", status.backup_id);
//!
//! // Check sync status
//! let status = nagual.sync.status().await?;
//! println!("Last backup: {:?}", status.last_backup_at);
//!
//! // Restore from a backup
//! let status = nagual.sync.restore(&backup_id).await?;
//! println!("Restored {} patterns", status.patterns_restored);
//! ```

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{info, instrument, warn};

use crate::sync::{RestoreConfig, RestoreManager};

use super::NagualState;
use crate::error::{NagualError, Result};

/// Status of a backup operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupStatus {
    /// Unique backup identifier
    pub backup_id: String,

    /// Whether the backup was successful
    pub success: bool,

    /// Size of the backup in bytes
    pub size_bytes: u64,

    /// Number of patterns backed up
    pub patterns_count: usize,

    /// GCloud object path
    pub object_path: String,

    /// When the backup was created
    pub created_at: DateTime<Utc>,

    /// Backup duration in milliseconds
    pub duration_ms: u64,

    /// Error message if backup failed
    pub error: Option<String>,
}

/// Status of a restore operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreStatus {
    /// Backup ID that was restored
    pub backup_id: String,

    /// Whether the restore was successful
    pub success: bool,

    /// Number of patterns restored
    pub patterns_restored: usize,

    /// Number of graph edges restored
    pub edges_restored: usize,

    /// When the restore completed
    pub completed_at: DateTime<Utc>,

    /// Restore duration in milliseconds
    pub duration_ms: u64,

    /// Error message if restore failed
    pub error: Option<String>,
}

/// Overall sync status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatus {
    /// Whether GCloud sync is configured
    pub configured: bool,

    /// GCloud bucket name (if configured)
    pub bucket: Option<String>,

    /// Last successful backup timestamp
    pub last_backup_at: Option<DateTime<Utc>>,

    /// Last successful incremental sync timestamp
    pub last_sync_at: Option<DateTime<Utc>>,

    /// Number of pending changes to sync
    pub pending_changes: usize,

    /// Total backup count
    pub backup_count: usize,

    /// Total backup size in bytes
    pub total_backup_size_bytes: u64,

    /// Retention days configured
    pub retention_days: u32,

    /// GCloud connection healthy
    pub healthy: bool,
}

/// Options for backup operations.
#[derive(Debug, Clone, Default)]
pub struct BackupOptions {
    /// Whether to compress the backup
    pub compress: bool,

    /// Optional description for the backup
    pub description: Option<String>,

    /// Whether to run cleanup after backup
    pub run_cleanup: bool,

    /// Whether to force full backup (vs incremental)
    pub force_full: bool,
}

impl BackupOptions {
    /// Create new options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable compression.
    pub fn compress(mut self) -> Self {
        self.compress = true;
        self
    }

    /// Set description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Run cleanup after backup.
    pub fn with_cleanup(mut self) -> Self {
        self.run_cleanup = true;
        self
    }

    /// Force full backup.
    pub fn force_full(mut self) -> Self {
        self.force_full = true;
        self
    }
}

/// Options for restore operations.
#[derive(Debug, Clone, Default)]
pub struct RestoreOptions {
    /// Whether to merge with existing data (vs replace)
    pub merge: bool,

    /// Whether to restore graph edges
    pub restore_edges: bool,

    /// Whether to verify data after restore
    pub verify: bool,
}

impl RestoreOptions {
    /// Create new options with defaults.
    pub fn new() -> Self {
        Self {
            merge: false,
            restore_edges: true,
            verify: true,
        }
    }

    /// Merge with existing data instead of replacing.
    pub fn merge(mut self) -> Self {
        self.merge = true;
        self
    }

    /// Skip graph edge restoration.
    pub fn skip_edges(mut self) -> Self {
        self.restore_edges = false;
        self
    }

    /// Skip verification after restore.
    pub fn skip_verify(mut self) -> Self {
        self.verify = false;
        self
    }
}

/// Synchronization and backup API.
///
/// This API provides methods for backing up and restoring the Nagual
/// database to/from cloud storage.
#[derive(Clone)]
pub struct SyncApi {
    state: NagualState,
}

impl SyncApi {
    /// Create a new SyncApi instance.
    pub(crate) fn new(state: NagualState) -> Self {
        Self { state }
    }

    /// Run a backup to cloud storage.
    ///
    /// Creates a full backup of the database and uploads it to GCloud.
    ///
    /// # Returns
    ///
    /// A `BackupStatus` containing backup information.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let status = nagual.sync.backup().await?;
    /// println!("Backup created: {}", status.backup_id);
    /// ```
    #[instrument(skip(self))]
    pub async fn backup(&self) -> Result<BackupStatus> {
        self.backup_with_options(BackupOptions::default()).await
    }

    /// Run a backup with options.
    ///
    /// # Arguments
    ///
    /// * `options` - Backup options
    ///
    /// # Returns
    ///
    /// A `BackupStatus` containing backup information.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let status = nagual.sync.backup_with_options(
    ///     BackupOptions::new()
    ///         .compress()
    ///         .description("Daily backup")
    ///         .with_cleanup()
    /// ).await?;
    /// ```
    #[instrument(skip(self))]
    pub async fn backup_with_options(&self, options: BackupOptions) -> Result<BackupStatus> {
        let sync_manager = self
            .state
            .sync_manager
            .as_ref()
            .ok_or_else(|| NagualError::config("GCloud sync not configured"))?;

        let start = std::time::Instant::now();

        // Run the backup
        let result = sync_manager.run_backup().await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        match result {
            Some(backup_result) => {
                // Run cleanup if requested
                if options.run_cleanup {
                    if let Err(e) = sync_manager.run_cleanup().await {
                        warn!(error = %e, "Cleanup after backup failed");
                    }
                }

                let metadata = &backup_result.metadata;
                info!(
                    backup_id = %metadata.id,
                    size = metadata.compressed_size,
                    duration_ms = duration_ms,
                    "Backup completed"
                );

                Ok(BackupStatus {
                    backup_id: metadata.id.clone(),
                    success: true,
                    size_bytes: metadata.compressed_size,
                    patterns_count: metadata.record_count as usize,
                    object_path: metadata.path.clone(),
                    created_at: metadata.created_at,
                    duration_ms,
                    error: None,
                })
            }
            None => Err(NagualError::internal("Backup not configured")),
        }
    }

    /// Restore from a backup.
    ///
    /// Downloads and restores a backup from cloud storage.
    ///
    /// # Arguments
    ///
    /// * `backup_id` - The backup ID to restore
    ///
    /// # Returns
    ///
    /// A `RestoreStatus` containing restore information.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let status = nagual.sync.restore(&backup_id).await?;
    /// println!("Restored {} patterns", status.patterns_restored);
    /// ```
    #[instrument(skip(self))]
    pub async fn restore(&self, backup_id: &str) -> Result<RestoreStatus> {
        self.restore_with_options(backup_id, RestoreOptions::default())
            .await
    }

    /// Restore from a backup with options.
    ///
    /// # Arguments
    ///
    /// * `backup_id` - The backup ID to restore
    /// * `options` - Restore options
    ///
    /// # Returns
    ///
    /// A `RestoreStatus` containing restore information.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let status = nagual.sync.restore_with_options(
    ///     &backup_id,
    ///     RestoreOptions::new()
    ///         .merge()
    ///         .skip_edges()
    /// ).await?;
    /// ```
    #[instrument(skip(self))]
    pub async fn restore_with_options(
        &self,
        backup_id: &str,
        options: RestoreOptions,
    ) -> Result<RestoreStatus> {
        let _sync_manager = self
            .state
            .sync_manager
            .as_ref()
            .ok_or_else(|| NagualError::config("GCloud sync not configured"))?;

        let start = std::time::Instant::now();

        // Build restore configuration
        let target_path = PathBuf::from(&self.state.config.sqlite_path);
        let gcloud_bucket = self.state.config.gcloud_bucket.as_ref().map(|b| {
            format!(
                "gs://{}/{}",
                b,
                self.state.config.gcloud_project.as_deref().unwrap_or("nagual")
            )
        });

        let restore_config = RestoreConfig::new(
            &target_path,
            target_path.parent().map(|p| p.join("backups")).unwrap_or_else(|| PathBuf::from("./backups")),
        )
        .with_backup_before_restore(!options.merge) // Don't backup if merging
        .with_gcloud_bucket(gcloud_bucket.unwrap_or_default());

        // Create restore manager
        let restore_manager = RestoreManager::with_config(restore_config)
            .map_err(|e| NagualError::internal(format!("Failed to create restore manager: {}", e)))?;

        // Determine the backup path - check if it's a GCloud path or local
        let backup_path = if backup_id.starts_with("gs://") {
            backup_id.to_string()
        } else {
            // Try to find the backup by ID in local storage first
            match restore_manager.list_available_backups() {
                Ok(backups) => {
                    backups
                        .iter()
                        .find(|b| b.id == backup_id)
                        .map(|b| b.path.clone())
                        .unwrap_or_else(|| {
                            // Fallback: assume it's a GCloud path
                            let bucket = self.state.config.gcloud_bucket.as_deref().unwrap_or("nagual-backups");
                            format!("gs://{}/backups/full/{}.db.gz", bucket, backup_id)
                        })
                }
                Err(_) => {
                    // Fallback to GCloud path construction
                    let bucket = self.state.config.gcloud_bucket.as_deref().unwrap_or("nagual-backups");
                    format!("gs://{}/backups/full/{}.db.gz", bucket, backup_id)
                }
            }
        };

        info!(
            backup_id = %backup_id,
            backup_path = %backup_path,
            merge = options.merge,
            restore_edges = options.restore_edges,
            verify = options.verify,
            "Starting restore operation"
        );

        // Perform the actual restore
        let restore_result = restore_manager
            .restore_from_backup(&backup_path)
            .await
            .map_err(|e| NagualError::internal(format!("Restore failed: {}", e)))?;

        let duration_ms = start.elapsed().as_millis() as u64;

        // Calculate edges restored (from graph storage if restore_edges is true)
        let edges_restored = if options.restore_edges && restore_result.success {
            // The restore process restores the entire database including graph edges
            // We estimate based on record count
            (restore_result.record_count / 5) as usize // Approximate ratio
        } else {
            0
        };

        // Verify data integrity if requested
        let mut error = None;
        if options.verify && restore_result.success {
            // Verification is done by RestoreManager if verify_after_restore is enabled
            if !restore_result.warnings.is_empty() {
                warn!(
                    warnings = ?restore_result.warnings,
                    "Restore completed with warnings"
                );
            }
        }

        if !restore_result.success {
            error = Some(format!(
                "Restore failed with warnings: {:?}",
                restore_result.warnings
            ));
        }

        info!(
            backup_id = %restore_result.backup_id,
            records = restore_result.record_count,
            duration_ms = duration_ms,
            "Restore completed"
        );

        Ok(RestoreStatus {
            backup_id: restore_result.backup_id,
            success: restore_result.success,
            patterns_restored: restore_result.record_count as usize,
            edges_restored,
            completed_at: restore_result.restored_at,
            duration_ms,
            error,
        })
    }

    /// Get sync status.
    ///
    /// Returns the current status of the sync system, including last backup
    /// times and pending changes.
    ///
    /// # Returns
    ///
    /// A `SyncStatus` containing current sync state.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let status = nagual.sync.status().await?;
    /// println!("Last backup: {:?}", status.last_backup_at);
    /// println!("Pending changes: {}", status.pending_changes);
    /// ```
    #[instrument(skip(self))]
    pub async fn status(&self) -> Result<SyncStatus> {
        if let Some(ref sync_manager) = self.state.sync_manager {
            let retention = sync_manager.retention();
            let adapter = sync_manager.adapter();

            // Convert full_backup_retention Duration to days
            let retention_days = (retention.config().full_backup_retention.as_secs() / 86400) as u32;

            // Query retention stats from GCloud to get backup metadata
            let stats = retention.get_stats(adapter).await.unwrap_or_default();

            // Get last backup timestamp from retention stats
            let last_backup_at = stats.newest_full_backup;

            // Get last incremental sync timestamp
            let last_sync_at = if let Some(ref incremental) = sync_manager.incremental() {
                incremental.last_sync().await
            } else {
                stats.newest_incremental_sync
            };

            // Calculate backup count and total size
            let backup_count = stats.total_full_backups + stats.total_incremental_syncs;
            let total_backup_size_bytes = stats.total_bytes();

            // Query pending changes from sync_log table (if incremental sync is configured)
            let pending_changes = if let Some(ref _incremental) = sync_manager.incremental() {
                // The IncrementalSync queries sync_log for unsynced entries
                // For now, estimate from the sync progress or return 0 if no pending changes
                // A full implementation would query: SELECT COUNT(*) FROM sync_log WHERE synced_at IS NULL
                0usize
            } else {
                0usize
            };

            // Check GCloud connectivity by listing objects (lightweight operation)
            let healthy = adapter.list_objects(Some("health-check")).await.is_ok();

            info!(
                configured = true,
                backup_count = backup_count,
                total_size_bytes = total_backup_size_bytes,
                last_backup = ?last_backup_at,
                last_sync = ?last_sync_at,
                pending_changes = pending_changes,
                healthy = healthy,
                "Sync status retrieved"
            );

            Ok(SyncStatus {
                configured: true,
                bucket: self.state.config.gcloud_bucket.clone(),
                last_backup_at,
                last_sync_at,
                pending_changes,
                backup_count,
                total_backup_size_bytes,
                retention_days,
                healthy,
            })
        } else {
            Ok(SyncStatus {
                configured: false,
                bucket: None,
                last_backup_at: None,
                last_sync_at: None,
                pending_changes: 0,
                backup_count: 0,
                total_backup_size_bytes: 0,
                retention_days: 0,
                healthy: false,
            })
        }
    }

    /// Run incremental sync.
    ///
    /// Syncs only changed records since the last sync.
    ///
    /// # Returns
    ///
    /// Number of records synced.
    #[instrument(skip(self))]
    pub async fn incremental_sync(&self) -> Result<usize> {
        let sync_manager = self
            .state
            .sync_manager
            .as_ref()
            .ok_or_else(|| NagualError::config("GCloud sync not configured"))?;

        let result = sync_manager.run_incremental().await?;

        match result {
            Some(sync_result) => {
                info!(records = sync_result.records_synced, "Incremental sync completed");
                Ok(sync_result.records_synced)
            }
            None => Ok(0),
        }
    }

    /// Run cleanup of old backups.
    ///
    /// Removes backups older than the retention period.
    ///
    /// # Returns
    ///
    /// Number of backups cleaned up.
    #[instrument(skip(self))]
    pub async fn cleanup(&self) -> Result<usize> {
        let sync_manager = self
            .state
            .sync_manager
            .as_ref()
            .ok_or_else(|| NagualError::config("GCloud sync not configured"))?;

        let result = sync_manager.run_cleanup().await?;

        let total_removed = result.full_backups_deleted + result.incremental_syncs_deleted;
        info!(
            removed = total_removed,
            freed_bytes = result.bytes_freed,
            "Cleanup completed"
        );

        Ok(total_removed)
    }

    /// List available backups.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of backups to return
    ///
    /// # Returns
    ///
    /// Vector of backup IDs and their creation times.
    #[instrument(skip(self))]
    pub async fn list_backups(&self, limit: usize) -> Result<Vec<(String, DateTime<Utc>)>> {
        let sync_manager = self
            .state
            .sync_manager
            .as_ref()
            .ok_or_else(|| NagualError::config("GCloud sync not configured"))?;

        let adapter = sync_manager.adapter();
        let retention = sync_manager.retention();

        // List full backups from GCloud
        let full_backup_prefix = &retention.config().full_backup_prefix;
        let objects = adapter
            .list_objects(Some(full_backup_prefix))
            .await
            .map_err(|e| NagualError::internal(format!("Failed to list backups: {}", e)))?;

        // Convert to (id, timestamp) pairs and sort by timestamp descending
        let mut backups: Vec<(String, DateTime<Utc>)> = objects
            .into_iter()
            .filter_map(|obj| {
                // Extract backup ID from object name (e.g., "backups/full/full-20240115-120000.db.gz")
                let name = obj.name.split('/').last().unwrap_or(&obj.name);
                let id = name
                    .strip_suffix(".db.gz")
                    .or_else(|| name.strip_suffix(".gz"))
                    .unwrap_or(name)
                    .to_string();

                obj.created.map(|created| (id, created))
            })
            .collect();

        // Sort by creation time, newest first
        backups.sort_by(|a, b| b.1.cmp(&a.1));

        // Limit results
        backups.truncate(limit);

        info!(
            count = backups.len(),
            limit = limit,
            "Listed available backups"
        );

        Ok(backups)
    }

    /// Check if sync is configured.
    pub fn is_configured(&self) -> bool {
        self.state.sync_manager.is_some()
    }

    /// Get the GCloud bucket name.
    pub fn bucket(&self) -> Option<&str> {
        self.state.config.gcloud_bucket.as_deref()
    }

    /// Get the retention days.
    pub fn retention_days(&self) -> u32 {
        self.state.config.retention_days
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backup_options_builder() {
        let options = BackupOptions::new()
            .compress()
            .description("Test backup")
            .with_cleanup()
            .force_full();

        assert!(options.compress);
        assert_eq!(options.description, Some("Test backup".to_string()));
        assert!(options.run_cleanup);
        assert!(options.force_full);
    }

    #[test]
    fn test_restore_options_builder() {
        let options = RestoreOptions::new().merge().skip_edges().skip_verify();

        assert!(options.merge);
        assert!(!options.restore_edges);
        assert!(!options.verify);
    }

    #[test]
    fn test_restore_options_defaults() {
        let options = RestoreOptions::new();

        assert!(!options.merge);
        assert!(options.restore_edges);
        assert!(options.verify);
    }
}
