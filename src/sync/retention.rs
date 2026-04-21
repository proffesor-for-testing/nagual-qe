//! Backup retention policy management.
//!
//! Implements automatic cleanup of old backups based on configurable retention policies:
//! - Full backups: keep for 30 days (default)
//! - Incremental syncs: keep for 7 days (default)
//!
//! # Usage
//!
//! ```rust,ignore
//! use nagual::sync::{RetentionPolicy, RetentionConfig};
//!
//! let config = RetentionConfig::default();
//! let policy = RetentionPolicy::new(config);
//!
//! // Run cleanup
//! let result = policy.cleanup(&adapter).await?;
//! println!("Deleted {} full backups, {} incremental syncs",
//!     result.full_backups_deleted, result.incremental_syncs_deleted);
//! ```
//!
//! # Scheduling
//!
//! The cleanup should be scheduled to run after each sync operation
//! to prevent unbounded storage growth.

use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use super::gcloud::{GCloudAdapter, GCloudResult, ObjectInfo};
use crate::error::Result;

/// Configuration for retention policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionConfig {
    /// How long to keep full backups (default: 30 days).
    pub full_backup_retention: Duration,

    /// How long to keep incremental syncs (default: 7 days).
    pub incremental_retention: Duration,

    /// Prefix for full backups in GCS.
    pub full_backup_prefix: String,

    /// Prefix for incremental syncs in GCS.
    pub incremental_prefix: String,

    /// Minimum number of full backups to keep (regardless of age).
    pub min_full_backups: usize,

    /// Minimum number of incremental syncs to keep (regardless of age).
    pub min_incremental_syncs: usize,

    /// Dry run mode (log what would be deleted without deleting).
    pub dry_run: bool,

    /// Enable detailed logging.
    pub verbose: bool,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            full_backup_retention: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            incremental_retention: Duration::from_secs(7 * 24 * 60 * 60),  // 7 days
            full_backup_prefix: "backups/full".to_string(),
            incremental_prefix: "incremental".to_string(),
            min_full_backups: 3,
            min_incremental_syncs: 10,
            dry_run: false,
            verbose: true,
        }
    }
}

impl RetentionConfig {
    /// Create a new config with custom retention periods.
    pub fn new(full_days: u64, incremental_days: u64) -> Self {
        Self {
            full_backup_retention: Duration::from_secs(full_days * 24 * 60 * 60),
            incremental_retention: Duration::from_secs(incremental_days * 24 * 60 * 60),
            ..Default::default()
        }
    }

    /// Set full backup retention period.
    pub fn with_full_backup_retention(mut self, days: u64) -> Self {
        self.full_backup_retention = Duration::from_secs(days * 24 * 60 * 60);
        self
    }

    /// Set incremental sync retention period.
    pub fn with_incremental_retention(mut self, days: u64) -> Self {
        self.incremental_retention = Duration::from_secs(days * 24 * 60 * 60);
        self
    }

    /// Set full backup prefix.
    pub fn with_full_backup_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.full_backup_prefix = prefix.into();
        self
    }

    /// Set incremental sync prefix.
    pub fn with_incremental_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.incremental_prefix = prefix.into();
        self
    }

    /// Enable dry run mode.
    pub fn with_dry_run(mut self) -> Self {
        self.dry_run = true;
        self
    }

    /// Set minimum full backups to keep.
    pub fn with_min_full_backups(mut self, count: usize) -> Self {
        self.min_full_backups = count;
        self
    }

    /// Set minimum incremental syncs to keep.
    pub fn with_min_incremental_syncs(mut self, count: usize) -> Self {
        self.min_incremental_syncs = count;
        self
    }

    /// Get cutoff time for full backups.
    pub fn full_backup_cutoff(&self) -> DateTime<Utc> {
        Utc::now() - chrono::Duration::from_std(self.full_backup_retention).unwrap()
    }

    /// Get cutoff time for incremental syncs.
    pub fn incremental_cutoff(&self) -> DateTime<Utc> {
        Utc::now() - chrono::Duration::from_std(self.incremental_retention).unwrap()
    }
}

/// Statistics about stored backups.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetentionStats {
    /// Total full backups.
    pub total_full_backups: usize,

    /// Total incremental syncs.
    pub total_incremental_syncs: usize,

    /// Full backups that are expired (past retention).
    pub expired_full_backups: usize,

    /// Incremental syncs that are expired.
    pub expired_incremental_syncs: usize,

    /// Total storage used by full backups (bytes).
    pub full_backup_bytes: u64,

    /// Total storage used by incremental syncs (bytes).
    pub incremental_bytes: u64,

    /// Oldest full backup timestamp.
    pub oldest_full_backup: Option<DateTime<Utc>>,

    /// Newest full backup timestamp.
    pub newest_full_backup: Option<DateTime<Utc>>,

    /// Oldest incremental sync timestamp.
    pub oldest_incremental_sync: Option<DateTime<Utc>>,

    /// Newest incremental sync timestamp.
    pub newest_incremental_sync: Option<DateTime<Utc>>,
}

impl RetentionStats {
    /// Calculate total storage used.
    pub fn total_bytes(&self) -> u64 {
        self.full_backup_bytes + self.incremental_bytes
    }

    /// Format total storage as human-readable.
    pub fn total_storage_formatted(&self) -> String {
        format_bytes(self.total_bytes())
    }

    /// Get count of backups that would be deleted.
    pub fn pending_deletion(&self) -> usize {
        self.expired_full_backups + self.expired_incremental_syncs
    }
}

/// Result of a cleanup operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupResult {
    /// Whether cleanup was successful.
    pub success: bool,

    /// Number of full backups deleted.
    pub full_backups_deleted: usize,

    /// Number of incremental syncs deleted.
    pub incremental_syncs_deleted: usize,

    /// Bytes freed.
    pub bytes_freed: u64,

    /// Objects that failed to delete.
    pub failed_deletions: Vec<String>,

    /// Duration of cleanup operation.
    pub duration_ms: u64,

    /// Whether this was a dry run.
    pub dry_run: bool,

    /// Stats before cleanup.
    pub stats_before: RetentionStats,

    /// Stats after cleanup (same as before if dry run).
    pub stats_after: RetentionStats,
}

impl CleanupResult {
    /// Get total objects deleted.
    pub fn total_deleted(&self) -> usize {
        self.full_backups_deleted + self.incremental_syncs_deleted
    }

    /// Format bytes freed as human-readable.
    pub fn bytes_freed_formatted(&self) -> String {
        format_bytes(self.bytes_freed)
    }
}

/// Retention policy manager.
///
/// Handles automatic cleanup of old backups based on configured retention periods.
pub struct RetentionPolicy {
    config: RetentionConfig,
}

impl RetentionPolicy {
    /// Create a new retention policy with the given configuration.
    pub fn new(config: RetentionConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_policy() -> Self {
        Self::new(RetentionConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &RetentionConfig {
        &self.config
    }

    /// Get current backup statistics.
    pub async fn get_stats(&self, adapter: &GCloudAdapter) -> GCloudResult<RetentionStats> {
        let mut stats = RetentionStats::default();

        let full_cutoff = self.config.full_backup_cutoff();
        let incremental_cutoff = self.config.incremental_cutoff();

        // Get full backups
        let full_objects = adapter
            .list_objects(Some(&self.config.full_backup_prefix))
            .await?;

        for obj in &full_objects {
            stats.total_full_backups += 1;
            stats.full_backup_bytes += obj.size;

            if let Some(created) = obj.created {
                if stats.oldest_full_backup.is_none() || Some(created) < stats.oldest_full_backup {
                    stats.oldest_full_backup = Some(created);
                }
                if stats.newest_full_backup.is_none() || Some(created) > stats.newest_full_backup {
                    stats.newest_full_backup = Some(created);
                }
                if created < full_cutoff {
                    stats.expired_full_backups += 1;
                }
            }
        }

        // Get incremental syncs
        let incremental_objects = adapter
            .list_objects(Some(&self.config.incremental_prefix))
            .await?;

        for obj in &incremental_objects {
            stats.total_incremental_syncs += 1;
            stats.incremental_bytes += obj.size;

            if let Some(created) = obj.created {
                if stats.oldest_incremental_sync.is_none()
                    || Some(created) < stats.oldest_incremental_sync
                {
                    stats.oldest_incremental_sync = Some(created);
                }
                if stats.newest_incremental_sync.is_none()
                    || Some(created) > stats.newest_incremental_sync
                {
                    stats.newest_incremental_sync = Some(created);
                }
                if created < incremental_cutoff {
                    stats.expired_incremental_syncs += 1;
                }
            }
        }

        Ok(stats)
    }

    /// Run cleanup operation.
    ///
    /// Deletes backups that are older than the configured retention periods,
    /// while ensuring minimum backup counts are maintained.
    pub async fn cleanup(&self, adapter: &GCloudAdapter) -> Result<CleanupResult> {
        let start = std::time::Instant::now();

        info!(
            full_backup_retention_days = self.config.full_backup_retention.as_secs() / 86400,
            incremental_retention_days = self.config.incremental_retention.as_secs() / 86400,
            dry_run = self.config.dry_run,
            "Starting retention cleanup"
        );

        // Get stats before cleanup
        let stats_before = self.get_stats(adapter).await.map_err(|e| {
            crate::error::NagualError::internal(format!("Failed to get stats: {}", e))
        })?;

        if self.config.verbose {
            debug!(
                total_full = stats_before.total_full_backups,
                expired_full = stats_before.expired_full_backups,
                total_incremental = stats_before.total_incremental_syncs,
                expired_incremental = stats_before.expired_incremental_syncs,
                "Pre-cleanup stats"
            );
        }

        let mut full_deleted = 0usize;
        let mut incremental_deleted = 0usize;
        let mut bytes_freed = 0u64;
        let mut failed_deletions = Vec::new();

        // Cleanup full backups
        let full_result = self
            .cleanup_prefix(
                adapter,
                &self.config.full_backup_prefix,
                self.config.full_backup_cutoff(),
                self.config.min_full_backups,
            )
            .await;

        match full_result {
            Ok((deleted, bytes, failures)) => {
                full_deleted = deleted;
                bytes_freed += bytes;
                failed_deletions.extend(failures);
            }
            Err(e) => {
                error!(error = %e, "Failed to cleanup full backups");
                failed_deletions.push(format!("Full backup cleanup error: {}", e));
            }
        }

        // Cleanup incremental syncs
        let incremental_result = self
            .cleanup_prefix(
                adapter,
                &self.config.incremental_prefix,
                self.config.incremental_cutoff(),
                self.config.min_incremental_syncs,
            )
            .await;

        match incremental_result {
            Ok((deleted, bytes, failures)) => {
                incremental_deleted = deleted;
                bytes_freed += bytes;
                failed_deletions.extend(failures);
            }
            Err(e) => {
                error!(error = %e, "Failed to cleanup incremental syncs");
                failed_deletions.push(format!("Incremental sync cleanup error: {}", e));
            }
        }

        // Get stats after cleanup
        let stats_after = if self.config.dry_run {
            stats_before.clone()
        } else {
            self.get_stats(adapter).await.unwrap_or_default()
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        let result = CleanupResult {
            success: failed_deletions.is_empty(),
            full_backups_deleted: full_deleted,
            incremental_syncs_deleted: incremental_deleted,
            bytes_freed,
            failed_deletions,
            duration_ms,
            dry_run: self.config.dry_run,
            stats_before,
            stats_after,
        };

        if result.success {
            info!(
                full_deleted = result.full_backups_deleted,
                incremental_deleted = result.incremental_syncs_deleted,
                bytes_freed = result.bytes_freed_formatted(),
                duration_ms,
                dry_run = self.config.dry_run,
                "Retention cleanup completed"
            );
        } else {
            warn!(
                failures = result.failed_deletions.len(),
                "Retention cleanup completed with errors"
            );
        }

        Ok(result)
    }

    /// Cleanup objects under a specific prefix.
    async fn cleanup_prefix(
        &self,
        adapter: &GCloudAdapter,
        prefix: &str,
        cutoff: DateTime<Utc>,
        min_keep: usize,
    ) -> GCloudResult<(usize, u64, Vec<String>)> {
        let objects = adapter.list_objects(Some(prefix)).await?;

        // Sort by creation time (newest first)
        let mut sorted_objects: Vec<(ObjectInfo, Option<DateTime<Utc>>)> = objects
            .into_iter()
            .map(|obj| {
                let created = obj.created;
                (obj, created)
            })
            .collect();

        sorted_objects.sort_by(|a, b| b.1.cmp(&a.1));

        // Determine which objects to delete
        let mut to_delete = Vec::new();

        for (i, (obj, created)) in sorted_objects.iter().enumerate() {
            // Keep minimum number regardless of age
            if i < min_keep {
                continue;
            }

            // Delete if past cutoff
            if let Some(created_at) = created {
                if *created_at < cutoff {
                    to_delete.push(obj.clone());
                }
            }
        }

        if to_delete.is_empty() {
            return Ok((0, 0, Vec::new()));
        }

        if self.config.verbose {
            debug!(
                prefix = %prefix,
                total = sorted_objects.len(),
                to_delete = to_delete.len(),
                min_keep,
                "Cleaning up expired objects"
            );
        }

        // Delete objects
        let mut deleted = 0;
        let mut bytes_freed = 0u64;
        let mut failures = Vec::new();

        for obj in to_delete {
            if self.config.dry_run {
                debug!(
                    object = %obj.name,
                    size = obj.size,
                    created = ?obj.created,
                    "[DRY RUN] Would delete object"
                );
                deleted += 1;
                bytes_freed += obj.size;
            } else {
                match adapter.delete_object(&obj.name).await {
                    Ok(_) => {
                        debug!(object = %obj.name, "Deleted object");
                        deleted += 1;
                        bytes_freed += obj.size;
                    }
                    Err(e) => {
                        warn!(object = %obj.name, error = %e, "Failed to delete object");
                        failures.push(format!("{}: {}", obj.name, e));
                    }
                }
            }
        }

        Ok((deleted, bytes_freed, failures))
    }

    /// Get objects that would be deleted (for preview).
    pub async fn get_pending_deletions(
        &self,
        adapter: &GCloudAdapter,
    ) -> GCloudResult<Vec<ObjectInfo>> {
        let mut pending = Vec::new();

        // Check full backups
        let full_objects = adapter
            .list_objects(Some(&self.config.full_backup_prefix))
            .await?;
        let full_cutoff = self.config.full_backup_cutoff();

        let mut sorted_full: Vec<_> = full_objects
            .into_iter()
            .map(|obj| (obj.created, obj))
            .collect();
        sorted_full.sort_by(|a, b| b.0.cmp(&a.0));

        for (i, (created, obj)) in sorted_full.into_iter().enumerate() {
            if i >= self.config.min_full_backups {
                if let Some(created_at) = created {
                    if created_at < full_cutoff {
                        pending.push(obj);
                    }
                }
            }
        }

        // Check incremental syncs
        let incremental_objects = adapter
            .list_objects(Some(&self.config.incremental_prefix))
            .await?;
        let incremental_cutoff = self.config.incremental_cutoff();

        let mut sorted_inc: Vec<_> = incremental_objects
            .into_iter()
            .map(|obj| (obj.created, obj))
            .collect();
        sorted_inc.sort_by(|a, b| b.0.cmp(&a.0));

        for (i, (created, obj)) in sorted_inc.into_iter().enumerate() {
            if i >= self.config.min_incremental_syncs {
                if let Some(created_at) = created {
                    if created_at < incremental_cutoff {
                        pending.push(obj);
                    }
                }
            }
        }

        Ok(pending)
    }
}

impl Clone for RetentionPolicy {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
        }
    }
}

impl std::fmt::Debug for RetentionPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetentionPolicy")
            .field("config", &self.config)
            .finish()
    }
}

/// Format bytes as human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retention_config_defaults() {
        let config = RetentionConfig::default();

        assert_eq!(
            config.full_backup_retention,
            Duration::from_secs(30 * 24 * 60 * 60)
        );
        assert_eq!(
            config.incremental_retention,
            Duration::from_secs(7 * 24 * 60 * 60)
        );
        assert_eq!(config.min_full_backups, 3);
        assert_eq!(config.min_incremental_syncs, 10);
        assert!(!config.dry_run);
    }

    #[test]
    fn test_retention_config_custom() {
        let config = RetentionConfig::new(14, 3)
            .with_min_full_backups(5)
            .with_dry_run();

        assert_eq!(
            config.full_backup_retention,
            Duration::from_secs(14 * 24 * 60 * 60)
        );
        assert_eq!(
            config.incremental_retention,
            Duration::from_secs(3 * 24 * 60 * 60)
        );
        assert_eq!(config.min_full_backups, 5);
        assert!(config.dry_run);
    }

    #[test]
    fn test_retention_cutoffs() {
        let config = RetentionConfig::new(30, 7);
        let now = Utc::now();

        let full_cutoff = config.full_backup_cutoff();
        let inc_cutoff = config.incremental_cutoff();

        // Full cutoff should be ~30 days ago
        let expected_full = now - chrono::Duration::days(30);
        assert!((full_cutoff - expected_full).num_seconds().abs() < 5);

        // Incremental cutoff should be ~7 days ago
        let expected_inc = now - chrono::Duration::days(7);
        assert!((inc_cutoff - expected_inc).num_seconds().abs() < 5);
    }

    #[test]
    fn test_retention_stats() {
        let stats = RetentionStats {
            total_full_backups: 10,
            expired_full_backups: 3,
            total_incremental_syncs: 50,
            expired_incremental_syncs: 20,
            full_backup_bytes: 1024 * 1024 * 100, // 100 MB
            incremental_bytes: 1024 * 1024 * 50,  // 50 MB
            ..Default::default()
        };

        assert_eq!(stats.total_bytes(), 1024 * 1024 * 150);
        assert_eq!(stats.pending_deletion(), 23);
        assert_eq!(stats.total_storage_formatted(), "150.00 MB");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_bytes(1024 * 1024 * 1024 * 1024), "1.00 TB");
    }

    #[test]
    fn test_cleanup_result() {
        let result = CleanupResult {
            success: true,
            full_backups_deleted: 5,
            incremental_syncs_deleted: 15,
            bytes_freed: 1024 * 1024 * 200, // 200 MB
            failed_deletions: Vec::new(),
            duration_ms: 1500,
            dry_run: false,
            stats_before: RetentionStats::default(),
            stats_after: RetentionStats::default(),
        };

        assert_eq!(result.total_deleted(), 20);
        assert_eq!(result.bytes_freed_formatted(), "200.00 MB");
    }

    #[test]
    fn test_retention_policy_creation() {
        let config = RetentionConfig::default();
        let policy = RetentionPolicy::new(config.clone());

        assert_eq!(
            policy.config().full_backup_retention,
            config.full_backup_retention
        );
    }
}
