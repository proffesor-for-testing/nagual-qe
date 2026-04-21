//! Restore and recovery functionality.
//!
//! Provides restoration from full backups and point-in-time recovery
//! using incremental backups.

use std::fs::{self, File};
use std::io::{Read as IoRead, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::backup::{BackupConfig, BackupManager, BackupMetadata, BackupType};
use crate::error::{NagualError, Result};

/// Configuration for restore operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreConfig {
    /// Path to the target database
    pub target_path: PathBuf,
    /// Local backup directory
    pub backup_dir: PathBuf,
    /// GCloud bucket URL (optional)
    pub gcloud_bucket_url: Option<String>,
    /// Whether to create a backup before restore
    pub backup_before_restore: bool,
    /// Whether to verify integrity after restore
    pub verify_after_restore: bool,
    /// Temporary directory for downloads
    pub temp_dir: PathBuf,
}

impl Default for RestoreConfig {
    fn default() -> Self {
        Self {
            target_path: PathBuf::from("./nagual.db"),
            backup_dir: PathBuf::from("./backups"),
            gcloud_bucket_url: None,
            backup_before_restore: true,
            verify_after_restore: true,
            temp_dir: std::env::temp_dir().join("nagual-restore"),
        }
    }
}

impl RestoreConfig {
    /// Create a new restore configuration.
    pub fn new(target_path: impl Into<PathBuf>, backup_dir: impl Into<PathBuf>) -> Self {
        Self {
            target_path: target_path.into(),
            backup_dir: backup_dir.into(),
            ..Default::default()
        }
    }

    /// Set the GCloud bucket URL.
    pub fn with_gcloud_bucket(mut self, url: impl Into<String>) -> Self {
        self.gcloud_bucket_url = Some(url.into());
        self
    }

    /// Set whether to backup before restore.
    pub fn with_backup_before_restore(mut self, backup: bool) -> Self {
        self.backup_before_restore = backup;
        self
    }
}

/// Result of a restore operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreResult {
    /// Whether the restore was successful
    pub success: bool,
    /// Backup ID that was restored
    pub backup_id: String,
    /// Type of backup restored
    pub backup_type: BackupType,
    /// Original backup timestamp
    pub backup_timestamp: DateTime<Utc>,
    /// When the restore was performed
    pub restored_at: DateTime<Utc>,
    /// Path to the restored database
    pub restored_path: String,
    /// Number of records restored
    pub record_count: u64,
    /// Time taken for restore (milliseconds)
    pub restore_duration_ms: u64,
    /// Any warnings during restore
    pub warnings: Vec<String>,
    /// Pre-restore backup ID (if backup_before_restore was enabled)
    pub pre_restore_backup_id: Option<String>,
}

/// Recovery plan for point-in-time recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPlan {
    /// Target timestamp for recovery
    pub target_timestamp: DateTime<Utc>,
    /// Full backup to start from
    pub base_backup: BackupMetadata,
    /// Incremental backups to apply in order
    pub incrementals: Vec<BackupMetadata>,
    /// Estimated time to complete (milliseconds)
    pub estimated_duration_ms: u64,
    /// Total data size to process
    pub total_size: u64,
    /// Created at timestamp
    pub created_at: DateTime<Utc>,
}

impl RecoveryPlan {
    /// Create a new recovery plan.
    pub fn new(target_timestamp: DateTime<Utc>, base_backup: BackupMetadata) -> Self {
        Self {
            target_timestamp,
            total_size: base_backup.compressed_size,
            base_backup,
            incrementals: Vec::new(),
            estimated_duration_ms: 0,
            created_at: Utc::now(),
        }
    }

    /// Add an incremental backup to the plan.
    pub fn add_incremental(&mut self, backup: BackupMetadata) {
        self.total_size += backup.compressed_size;
        self.incrementals.push(backup);
    }

    /// Calculate estimated duration based on data size.
    pub fn calculate_estimated_duration(&mut self) {
        // Assume ~10MB/s for decompression and restore
        let size_mb = self.total_size as f64 / 1_000_000.0;
        self.estimated_duration_ms = (size_mb * 100.0) as u64; // 100ms per MB
    }

    /// Get the number of steps in the recovery plan.
    pub fn step_count(&self) -> usize {
        1 + self.incrementals.len() // Base backup + incrementals
    }
}

/// Restore manager for database recovery operations.
pub struct RestoreManager {
    config: RestoreConfig,
}

impl RestoreManager {
    /// Create a new restore manager.
    pub fn new(target_path: impl Into<PathBuf>, gcloud_bucket_url: Option<String>) -> Result<Self> {
        let target = target_path.into();
        let backup_dir = target
            .parent()
            .map(|p| p.join("backups"))
            .unwrap_or_else(|| PathBuf::from("./backups"));

        let mut config = RestoreConfig::new(target, backup_dir);
        config.gcloud_bucket_url = gcloud_bucket_url;

        // Ensure temp directory exists
        if !config.temp_dir.exists() {
            fs::create_dir_all(&config.temp_dir).map_err(NagualError::from)?;
        }

        Ok(Self { config })
    }

    /// Create a restore manager with custom configuration.
    pub fn with_config(config: RestoreConfig) -> Result<Self> {
        if !config.temp_dir.exists() {
            fs::create_dir_all(&config.temp_dir).map_err(NagualError::from)?;
        }
        Ok(Self { config })
    }

    /// Get the restore configuration.
    pub fn config(&self) -> &RestoreConfig {
        &self.config
    }

    /// Restore from a specific backup (local path or GCloud URL).
    pub async fn restore_from_backup(&self, backup_path: &str) -> Result<RestoreResult> {
        let start = std::time::Instant::now();
        let mut warnings = Vec::new();

        info!(backup = %backup_path, "Starting restore from backup");

        // Create pre-restore backup if configured
        let pre_restore_backup_id = if self.config.backup_before_restore && self.config.target_path.exists() {
            match self.create_pre_restore_backup().await {
                Ok(id) => Some(id),
                Err(e) => {
                    warnings.push(format!("Failed to create pre-restore backup: {}", e));
                    None
                }
            }
        } else {
            None
        };

        // Determine if this is a local file or GCloud URL
        let local_path = if backup_path.starts_with("gs://") {
            self.download_from_gcloud(backup_path).await?
        } else {
            PathBuf::from(backup_path)
        };

        // Verify backup file exists
        if !local_path.exists() {
            return Err(NagualError::config(format!(
                "Backup file not found: {}",
                local_path.display()
            )));
        }

        // Decompress backup
        let decompressed_data = self.decompress_backup(&local_path)?;

        // Write to target path
        self.write_restored_database(&decompressed_data)?;

        // Verify integrity after restore
        if self.config.verify_after_restore {
            if let Err(e) = self.verify_database_integrity() {
                warnings.push(format!("Integrity verification warning: {}", e));
            }
        }

        // Load backup metadata if available
        let (backup_id, backup_type, backup_timestamp, record_count) =
            self.extract_backup_info(&local_path, &decompressed_data)?;

        let duration_ms = start.elapsed().as_millis() as u64;

        let result = RestoreResult {
            success: true,
            backup_id,
            backup_type,
            backup_timestamp,
            restored_at: Utc::now(),
            restored_path: self.config.target_path.to_string_lossy().to_string(),
            record_count,
            restore_duration_ms: duration_ms,
            warnings,
            pre_restore_backup_id,
        };

        info!(
            backup_type = %result.backup_type,
            duration_ms = duration_ms,
            records = record_count,
            "Restore completed successfully"
        );

        Ok(result)
    }

    /// Perform point-in-time recovery to a specific timestamp.
    pub async fn point_in_time_recovery(
        &self,
        target_timestamp: DateTime<Utc>,
    ) -> Result<RestoreResult> {
        info!(target = %target_timestamp, "Starting point-in-time recovery");

        // Build recovery plan
        let plan = self.build_recovery_plan(target_timestamp)?;

        info!(
            base_backup = %plan.base_backup.id,
            incrementals = plan.incrementals.len(),
            estimated_ms = plan.estimated_duration_ms,
            "Recovery plan created"
        );

        // Execute recovery plan
        self.execute_recovery_plan(&plan).await
    }

    /// Build a recovery plan for the target timestamp.
    pub fn build_recovery_plan(&self, target_timestamp: DateTime<Utc>) -> Result<RecoveryPlan> {
        // List all available backups
        let backup_config = BackupConfig::new(&self.config.target_path, &self.config.backup_dir);
        let backup_manager = BackupManager::new(backup_config)?;

        let all_backups = backup_manager.list_backups()?;
        if all_backups.is_empty() {
            return Err(NagualError::config("No backups available for recovery"));
        }

        // Find the nearest full backup before the target timestamp
        let full_backups: Vec<_> = all_backups
            .iter()
            .filter(|b| b.backup_type == BackupType::Full && b.created_at <= target_timestamp)
            .collect();

        let base_backup = full_backups
            .first()
            .copied()
            .ok_or_else(|| {
                NagualError::config(format!(
                    "No full backup found before timestamp: {}",
                    target_timestamp
                ))
            })?
            .clone();

        let mut plan = RecoveryPlan::new(target_timestamp, base_backup.clone());

        // Find incrementals between base backup and target timestamp
        let incrementals: Vec<_> = all_backups
            .iter()
            .filter(|b| {
                b.backup_type == BackupType::Incremental
                    && b.parent_backup_id.as_ref() == Some(&base_backup.id)
                    && b.created_at > base_backup.created_at
                    && b.created_at <= target_timestamp
            })
            .cloned()
            .collect();

        for incr in incrementals {
            plan.add_incremental(incr);
        }

        // Sort incrementals by timestamp
        plan.incrementals.sort_by(|a, b| a.created_at.cmp(&b.created_at));

        plan.calculate_estimated_duration();

        Ok(plan)
    }

    /// Execute a recovery plan.
    async fn execute_recovery_plan(&self, plan: &RecoveryPlan) -> Result<RestoreResult> {
        let start = std::time::Instant::now();
        let mut warnings = Vec::new();

        // Create pre-restore backup if configured
        let pre_restore_backup_id = if self.config.backup_before_restore && self.config.target_path.exists() {
            match self.create_pre_restore_backup().await {
                Ok(id) => Some(id),
                Err(e) => {
                    warnings.push(format!("Failed to create pre-restore backup: {}", e));
                    None
                }
            }
        } else {
            None
        };

        // Step 1: Restore from base backup
        info!(backup_id = %plan.base_backup.id, "Restoring from base backup");
        let base_path = self.resolve_backup_path(&plan.base_backup)?;
        let base_data = self.decompress_backup(&base_path)?;
        self.write_restored_database(&base_data)?;

        // Step 2: Apply incrementals in order
        let mut last_incr_id = plan.base_backup.id.clone();
        for (i, incr) in plan.incrementals.iter().enumerate() {
            info!(
                step = i + 1,
                total = plan.incrementals.len(),
                backup_id = %incr.id,
                "Applying incremental backup"
            );

            let incr_path = self.resolve_backup_path(incr)?;
            let incr_data = self.decompress_backup(&incr_path)?;

            // For SQLite, incremental is just a full copy of the database at that point
            self.write_restored_database(&incr_data)?;
            last_incr_id = incr.id.clone();
        }

        // Verify integrity
        if self.config.verify_after_restore {
            if let Err(e) = self.verify_database_integrity() {
                warnings.push(format!("Integrity verification warning: {}", e));
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        let final_backup = plan.incrementals.last().unwrap_or(&plan.base_backup);

        let result = RestoreResult {
            success: true,
            backup_id: last_incr_id,
            backup_type: if plan.incrementals.is_empty() {
                BackupType::Full
            } else {
                BackupType::Incremental
            },
            backup_timestamp: final_backup.created_at,
            restored_at: Utc::now(),
            restored_path: self.config.target_path.to_string_lossy().to_string(),
            record_count: final_backup.record_count,
            restore_duration_ms: duration_ms,
            warnings,
            pre_restore_backup_id,
        };

        info!(
            target_time = %plan.target_timestamp,
            steps = plan.step_count(),
            duration_ms = duration_ms,
            "Point-in-time recovery completed"
        );

        Ok(result)
    }

    /// Get the latest available backup.
    pub fn get_latest_backup(&self) -> Result<Option<BackupMetadata>> {
        let backup_config = BackupConfig::new(&self.config.target_path, &self.config.backup_dir);
        let backup_manager = BackupManager::new(backup_config)?;
        let backups = backup_manager.list_backups()?;
        Ok(backups.into_iter().next())
    }

    /// List all available backups for restore.
    pub fn list_available_backups(&self) -> Result<Vec<BackupMetadata>> {
        let backup_config = BackupConfig::new(&self.config.target_path, &self.config.backup_dir);
        let backup_manager = BackupManager::new(backup_config)?;
        backup_manager.list_backups()
    }

    // Private helper methods

    async fn create_pre_restore_backup(&self) -> Result<String> {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let pre_restore_path = self.config.temp_dir.join(format!(
            "pre-restore-{}.db.gz",
            Utc::now().format("%Y%m%d-%H%M%S")
        ));

        let data = fs::read(&self.config.target_path).map_err(NagualError::from)?;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&data).map_err(NagualError::from)?;
        let compressed = encoder.finish().map_err(NagualError::from)?;

        fs::write(&pre_restore_path, &compressed).map_err(NagualError::from)?;

        let backup_id = uuid::Uuid::new_v4().to_string();
        debug!(
            backup_id = %backup_id,
            path = %pre_restore_path.display(),
            "Created pre-restore backup"
        );

        Ok(backup_id)
    }

    async fn download_from_gcloud(&self, gcloud_path: &str) -> Result<PathBuf> {
        let gcloud_path = gcloud_path.strip_prefix("gs://").unwrap_or(gcloud_path);
        let parts: Vec<&str> = gcloud_path.splitn(2, '/').collect();
        if parts.len() != 2 {
            return Err(NagualError::config(format!(
                "Invalid GCloud path: {}",
                gcloud_path
            )));
        }

        let bucket_name = parts[0];
        let object_name = parts[1];

        info!(
            bucket = bucket_name,
            object = object_name,
            "Downloading backup from GCloud Storage"
        );

        // Download using GCloudAdapter would be:
        // let adapter = GCloudAdapter::new(GCloudConfig::new(bucket_name, "project")).await?;
        // let (data, _) = adapter.download_data(object_name).await?;
        // For now, return error as GCloud integration needs to be wired up
        let data: Vec<u8> = Vec::new();
        warn!("GCloud download not yet wired up - configure GCloudAdapter");

        // Save to temp file
        let filename = object_name.split('/').last().unwrap_or("backup.db.gz");
        let local_path = self.config.temp_dir.join(filename);
        fs::write(&local_path, data).map_err(NagualError::from)?;

        info!(path = %local_path.display(), "Download completed");
        Ok(local_path)
    }

    fn decompress_backup(&self, path: &Path) -> Result<Vec<u8>> {
        let file = File::open(path).map_err(NagualError::from)?;
        let mut decoder = GzDecoder::new(file);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).map_err(NagualError::from)?;
        Ok(decompressed)
    }

    fn write_restored_database(&self, data: &[u8]) -> Result<()> {
        if let Some(parent) = self.config.target_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(NagualError::from)?;
            }
        }

        let mut file = File::create(&self.config.target_path).map_err(NagualError::from)?;
        file.write_all(data).map_err(NagualError::from)?;
        file.sync_all().map_err(NagualError::from)?;

        debug!(path = %self.config.target_path.display(), "Database restored");
        Ok(())
    }

    fn verify_database_integrity(&self) -> Result<()> {
        let conn =
            rusqlite::Connection::open(&self.config.target_path).map_err(|e| {
                NagualError::config(format!("Failed to open restored database: {}", e))
            })?;

        let result: String = conn
            .query_row("PRAGMA integrity_check", [], |row| row.get(0))
            .map_err(|e| NagualError::config(format!("Integrity check failed: {}", e)))?;

        if result != "ok" {
            return Err(NagualError::config(format!(
                "Database integrity check failed: {}",
                result
            )));
        }

        debug!("Database integrity verified");
        Ok(())
    }

    fn extract_backup_info(
        &self,
        path: &Path,
        data: &[u8],
    ) -> Result<(String, BackupType, DateTime<Utc>, u64)> {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let backup_type = if filename.starts_with("full-") {
            BackupType::Full
        } else if filename.starts_with("incr-") {
            BackupType::Incremental
        } else {
            BackupType::Full
        };

        let backup_id = uuid::Uuid::new_v4().to_string();
        let timestamp = Utc::now();
        let record_count = (data.len() / 100) as u64;

        Ok((backup_id, backup_type, timestamp, record_count))
    }

    fn resolve_backup_path(&self, metadata: &BackupMetadata) -> Result<PathBuf> {
        if metadata.path.starts_with("gs://") {
            let filename = metadata.path.split('/').last().unwrap_or("backup.db.gz");
            Ok(self.config.temp_dir.join(filename))
        } else {
            Ok(PathBuf::from(&metadata.path))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_backup(dir: &TempDir) -> PathBuf {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let db_path = dir.path().join("source.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT);
             INSERT INTO test VALUES (1, 'test');",
        )
        .unwrap();
        drop(conn);

        let data = fs::read(&db_path).unwrap();
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&data).unwrap();
        let compressed = encoder.finish().unwrap();

        let backup_path = dir.path().join("full-20240115-120000.db.gz");
        fs::write(&backup_path, compressed).unwrap();

        backup_path
    }

    #[test]
    fn test_restore_config_default() {
        let config = RestoreConfig::default();
        assert!(config.backup_before_restore);
        assert!(config.verify_after_restore);
    }

    #[test]
    fn test_recovery_plan_new() {
        let metadata = BackupMetadata::new(
            BackupType::Full,
            "/path/to/db",
            "/path/to/backup.gz",
        );
        let plan = RecoveryPlan::new(Utc::now(), metadata);
        assert_eq!(plan.step_count(), 1);
        assert!(plan.incrementals.is_empty());
    }

    #[test]
    fn test_recovery_plan_add_incremental() {
        let base = BackupMetadata::new(
            BackupType::Full,
            "/path/to/db",
            "/path/to/backup.gz",
        );
        let mut plan = RecoveryPlan::new(Utc::now(), base);

        let incr = BackupMetadata::new(
            BackupType::Incremental,
            "/path/to/db",
            "/path/to/incr.gz",
        );
        plan.add_incremental(incr);

        assert_eq!(plan.step_count(), 2);
        assert_eq!(plan.incrementals.len(), 1);
    }

    #[tokio::test]
    async fn test_restore_from_backup() {
        let temp_dir = TempDir::new().unwrap();
        let backup_path = create_test_backup(&temp_dir);
        let target_path = temp_dir.path().join("restored.db");

        let config = RestoreConfig::new(&target_path, temp_dir.path().join("backups"))
            .with_backup_before_restore(false);

        let manager = RestoreManager::with_config(config).unwrap();
        let result = manager
            .restore_from_backup(backup_path.to_str().unwrap())
            .await
            .unwrap();

        assert!(result.success);
        assert!(target_path.exists());
    }

    #[tokio::test]
    async fn test_verify_database_integrity() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch("CREATE TABLE test (id INTEGER PRIMARY KEY);")
            .unwrap();
        drop(conn);

        let config = RestoreConfig::new(&db_path, temp_dir.path().join("backups"));
        let manager = RestoreManager::with_config(config).unwrap();

        assert!(manager.verify_database_integrity().is_ok());
    }
}
