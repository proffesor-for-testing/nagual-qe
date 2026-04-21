//! GCloud Sync Module for Nagual
//!
//! Provides comprehensive backup, restore, and synchronization capabilities:
//!
//! - `gcloud`: GCloud Storage adapter with upload/download operations
//! - `backup`: Full and incremental database backup functionality
//! - `restore`: Database restoration and point-in-time recovery
//! - `scheduler`: Cron-based scheduling for automated sync operations
//! - `drill`: Monthly restore drill automation for data integrity verification
//! - `retention`: Backup retention policy management
//!
//! # Architecture
//!
//! The sync module follows a tiered backup strategy:
//! - Full backups every 6 hours
//! - Incremental backups every 30 minutes
//! - Automatic retention cleanup (daily at 2 AM)
//! - Monthly restore drills (first Sunday of each month)
//!
//! # Security
//!
//! - All backups are compressed with gzip (flate2)
//! - CMEK encryption is supported for at-rest encryption in GCS
//! - Credentials are loaded via Google Application Default Credentials
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::sync::{RestoreManager, SyncScheduler, RestoreDrill};
//!
//! // Initialize restore manager
//! let restore_manager = RestoreManager::new(
//!     "/path/to/nagual.db",
//!     Some("gs://my-bucket/backups".to_string()),
//! )?;
//!
//! // Restore from a specific backup
//! restore_manager.restore_from_backup("gs://my-bucket/backups/full-2024-01-15.gz").await?;
//!
//! // Point-in-time recovery
//! let target = chrono::Utc::now() - chrono::Duration::hours(2);
//! restore_manager.point_in_time_recovery(target).await?;
//!
//! // Start the sync scheduler
//! let mut scheduler = SyncScheduler::new(Default::default())?;
//! scheduler.start().await?;
//! ```

mod backup;
pub mod brain;
pub mod drill;
mod gcloud;
mod incremental;
pub mod pii;
pub mod restore;
mod retention;
pub mod scheduler;

pub use backup::{
    BackupConfig, BackupManager, BackupMetadata, BackupResult, BackupType, FullBackup,
};
pub use drill::{DrillReport, DrillResult, RestoreDrill, RestoreDrillConfig};
pub use gcloud::{
    EncryptionConfig, GCloudAdapter, GCloudConfig, GCloudError, GCloudResult,
    KeyNameComponents, ObjectInfo,
};
pub use incremental::{
    IncrementalSync, IncrementalSyncConfig, SyncBatch, SyncLogEntry, SyncProgress,
    SyncResult,
};
pub use restore::{RecoveryPlan, RestoreConfig, RestoreManager, RestoreResult};
pub use retention::{CleanupResult, RetentionConfig, RetentionPolicy, RetentionStats};
pub use scheduler::{
    ScheduledTask, SchedulerEvent, SchedulerState, SyncHealth, SyncScheduler,
    SyncSchedulerConfig, SyncStatus, SyncStatusReport,
};

use crate::error::Result;

/// Sync manager that coordinates all sync operations.
pub struct SyncManager {
    /// GCloud adapter for storage operations
    adapter: GCloudAdapter,
    /// Incremental sync handler
    incremental: Option<IncrementalSync>,
    /// Full backup handler
    backup: Option<FullBackup>,
    /// Retention policy
    retention: RetentionPolicy,
}

impl SyncManager {
    /// Create a new sync manager with the given configuration.
    pub fn new(adapter: GCloudAdapter, retention_config: RetentionConfig) -> Self {
        Self {
            adapter,
            incremental: None,
            backup: None,
            retention: RetentionPolicy::new(retention_config),
        }
    }

    /// Configure incremental sync.
    pub fn with_incremental(mut self, config: IncrementalSyncConfig) -> Self {
        self.incremental = Some(IncrementalSync::new(self.adapter.clone(), config));
        self
    }

    /// Configure full backup.
    pub fn with_backup(mut self, config: BackupConfig) -> Self {
        self.backup = Some(FullBackup::new(self.adapter.clone(), config));
        self
    }

    /// Run incremental sync if configured.
    pub async fn run_incremental(&self) -> Result<Option<SyncResult>> {
        match &self.incremental {
            Some(sync) => Ok(Some(sync.sync().await?)),
            None => Ok(None),
        }
    }

    /// Run full backup if configured.
    pub async fn run_backup(&self) -> Result<Option<BackupResult>> {
        match &self.backup {
            Some(backup) => Ok(Some(backup.run().await?)),
            None => Ok(None),
        }
    }

    /// Run retention cleanup.
    pub async fn run_cleanup(&self) -> Result<CleanupResult> {
        self.retention.cleanup(&self.adapter).await
    }

    /// Get the GCloud adapter.
    pub fn adapter(&self) -> &GCloudAdapter {
        &self.adapter
    }

    /// Get retention policy.
    pub fn retention(&self) -> &RetentionPolicy {
        &self.retention
    }

    /// Get incremental sync handler.
    pub fn incremental(&self) -> Option<&IncrementalSync> {
        self.incremental.as_ref()
    }

    /// Get backup handler.
    pub fn backup(&self) -> Option<&FullBackup> {
        self.backup.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_manager_creation() {
        // Just verify the types compile correctly
        let _config = GCloudConfig::new("test-bucket", "test-project");
        let _retention = RetentionConfig::default();
    }

    #[tokio::test]
    async fn test_gcloud_config() {
        let config = GCloudConfig::new("my-bucket", "my-project")
            .with_prefix("nagual/sync");

        assert_eq!(config.bucket, "my-bucket");
        assert_eq!(config.project_id, "my-project");
        assert_eq!(config.prefix, "nagual/sync");
    }
}
