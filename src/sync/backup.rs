//! Backup functionality for SQLite and PostgreSQL databases.
//!
//! Provides full and incremental backup capabilities with compression
//! and optional upload to GCloud Storage.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::gcloud::GCloudAdapter;
use crate::error::{NagualError, Result};

/// Type of database being backed up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatabaseType {
    /// SQLite database (file copy)
    Sqlite,
    /// PostgreSQL database (pg_dump)
    Postgres,
}

impl DatabaseType {
    /// Get the file extension for backups.
    pub fn extension(&self) -> &str {
        match self {
            DatabaseType::Sqlite => "db.gz",
            DatabaseType::Postgres => "sql.gz",
        }
    }
}

/// Type of backup being performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackupType {
    /// Full database backup
    Full,
    /// Incremental backup (changes since last backup)
    Incremental,
}

impl std::fmt::Display for BackupType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackupType::Full => write!(f, "full"),
            BackupType::Incremental => write!(f, "incremental"),
        }
    }
}

/// Metadata for a backup file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// Unique backup identifier
    pub id: String,
    /// Type of backup
    pub backup_type: BackupType,
    /// When the backup was created
    pub created_at: DateTime<Utc>,
    /// Size of the compressed backup in bytes
    pub compressed_size: u64,
    /// Size of the original data in bytes
    pub original_size: u64,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// SHA-256 checksum of the backup file
    pub checksum: String,
    /// Path or URL to the backup file
    pub path: String,
    /// Number of records included
    pub record_count: u64,
    /// Source database path
    pub source_path: String,
    /// For incremental backups: ID of the parent full backup
    pub parent_backup_id: Option<String>,
    /// For incremental backups: timestamp of last change included
    pub last_change_timestamp: Option<DateTime<Utc>>,
}

impl BackupMetadata {
    /// Create new backup metadata.
    pub fn new(
        backup_type: BackupType,
        source_path: impl Into<String>,
        target_path: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            backup_type,
            created_at: Utc::now(),
            compressed_size: 0,
            original_size: 0,
            compression_ratio: 0.0,
            checksum: String::new(),
            path: target_path.into(),
            record_count: 0,
            source_path: source_path.into(),
            parent_backup_id: None,
            last_change_timestamp: None,
        }
    }

    /// Calculate and set the compression ratio.
    pub fn calculate_compression_ratio(&mut self) {
        if self.original_size > 0 {
            self.compression_ratio =
                1.0 - (self.compressed_size as f64 / self.original_size as f64);
        }
    }
}

/// Configuration for backup operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Local directory for storing backups
    pub local_backup_dir: PathBuf,
    /// GCloud Storage bucket URL (optional)
    pub gcloud_bucket_url: Option<String>,
    /// Compression level (0-9, default: 6)
    pub compression_level: u32,
    /// Whether to upload to GCloud after local backup
    pub upload_to_gcloud: bool,
    /// Number of full backups to retain
    pub retain_full_backups: usize,
    /// Number of incremental backups to retain per full backup
    pub retain_incremental_backups: usize,
    /// Whether to verify checksum after backup
    pub verify_after_backup: bool,
    /// Source database path
    pub source_path: PathBuf,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            local_backup_dir: PathBuf::from("./backups"),
            gcloud_bucket_url: None,
            compression_level: 6,
            upload_to_gcloud: false,
            retain_full_backups: 7,
            retain_incremental_backups: 48,
            verify_after_backup: true,
            source_path: PathBuf::from("./nagual.db"),
        }
    }
}

impl BackupConfig {
    /// Create a new backup configuration with the specified directory.
    pub fn new(source_path: impl Into<PathBuf>, backup_dir: impl Into<PathBuf>) -> Self {
        Self {
            source_path: source_path.into(),
            local_backup_dir: backup_dir.into(),
            ..Default::default()
        }
    }

    /// Set the GCloud bucket URL.
    pub fn with_gcloud_bucket(mut self, url: impl Into<String>) -> Self {
        self.gcloud_bucket_url = Some(url.into());
        self.upload_to_gcloud = true;
        self
    }

    /// Set the compression level.
    pub fn with_compression_level(mut self, level: u32) -> Self {
        self.compression_level = level.min(9);
        self
    }

    /// Set retention policies.
    pub fn with_retention(mut self, full_backups: usize, incremental_backups: usize) -> Self {
        self.retain_full_backups = full_backups;
        self.retain_incremental_backups = incremental_backups;
        self
    }
}

/// Backup manager for creating and managing database backups.
pub struct BackupManager {
    config: BackupConfig,
    last_full_backup_id: Option<String>,
    last_backup_timestamp: Option<DateTime<Utc>>,
}

impl BackupManager {
    /// Create a new backup manager.
    pub fn new(config: BackupConfig) -> Result<Self> {
        // Ensure backup directory exists
        if !config.local_backup_dir.exists() {
            fs::create_dir_all(&config.local_backup_dir).map_err(NagualError::from)?;
        }

        Ok(Self {
            config,
            last_full_backup_id: None,
            last_backup_timestamp: None,
        })
    }

    /// Get the backup configuration.
    pub fn config(&self) -> &BackupConfig {
        &self.config
    }

    /// Get the source database path.
    pub fn source_path(&self) -> &Path {
        &self.config.source_path
    }

    /// Create a full backup of the database.
    pub async fn create_full_backup(&mut self) -> Result<BackupMetadata> {
        info!(source = %self.config.source_path.display(), "Creating full backup");

        if !self.config.source_path.exists() {
            return Err(NagualError::config(format!(
                "Source database not found: {}",
                self.config.source_path.display()
            )));
        }

        let timestamp = Utc::now();
        let backup_filename = format!(
            "full-{}.db.gz",
            timestamp.format("%Y%m%d-%H%M%S")
        );
        let backup_path = self.config.local_backup_dir.join(&backup_filename);

        let mut metadata = BackupMetadata::new(
            BackupType::Full,
            self.config.source_path.to_string_lossy(),
            backup_path.to_string_lossy(),
        );

        // Read source file
        let source_data = fs::read(&self.config.source_path).map_err(NagualError::from)?;
        metadata.original_size = source_data.len() as u64;

        // Compress and write backup
        let compressed_data = self.compress_data(&source_data)?;
        metadata.compressed_size = compressed_data.len() as u64;
        metadata.calculate_compression_ratio();

        // Calculate checksum
        metadata.checksum = self.calculate_checksum(&compressed_data);

        // Write to file
        let mut file = File::create(&backup_path).map_err(NagualError::from)?;
        file.write_all(&compressed_data).map_err(NagualError::from)?;
        file.sync_all().map_err(NagualError::from)?;

        // Count records (estimate from file size)
        metadata.record_count = self.estimate_record_count(&source_data);

        // Verify if configured
        if self.config.verify_after_backup {
            self.verify_backup(&backup_path, &metadata.checksum)?;
        }

        // Update tracking
        self.last_full_backup_id = Some(metadata.id.clone());
        self.last_backup_timestamp = Some(timestamp);

        // Save metadata
        self.save_metadata(&metadata)?;

        info!(
            backup_id = %metadata.id,
            size = metadata.compressed_size,
            ratio = format!("{:.1}%", metadata.compression_ratio * 100.0),
            "Full backup created successfully"
        );

        Ok(metadata)
    }

    /// Create an incremental backup (changes since last backup).
    pub async fn create_incremental_backup(&mut self) -> Result<BackupMetadata> {
        let parent_id = self.last_full_backup_id.clone().ok_or_else(|| {
            NagualError::config("No full backup exists. Create a full backup first.")
        })?;

        info!(
            source = %self.config.source_path.display(),
            parent = %parent_id,
            "Creating incremental backup"
        );

        if !self.config.source_path.exists() {
            return Err(NagualError::config(format!(
                "Source database not found: {}",
                self.config.source_path.display()
            )));
        }

        let timestamp = Utc::now();
        let backup_filename = format!(
            "incr-{}.db.gz",
            timestamp.format("%Y%m%d-%H%M%S")
        );
        let backup_path = self.config.local_backup_dir.join(&backup_filename);

        let mut metadata = BackupMetadata::new(
            BackupType::Incremental,
            self.config.source_path.to_string_lossy(),
            backup_path.to_string_lossy(),
        );
        metadata.parent_backup_id = Some(parent_id);
        metadata.last_change_timestamp = self.last_backup_timestamp;

        // For SQLite, we backup the entire database (simpler approach)
        let source_data = fs::read(&self.config.source_path).map_err(NagualError::from)?;
        metadata.original_size = source_data.len() as u64;

        // Compress and write backup
        let compressed_data = self.compress_data(&source_data)?;
        metadata.compressed_size = compressed_data.len() as u64;
        metadata.calculate_compression_ratio();

        // Calculate checksum
        metadata.checksum = self.calculate_checksum(&compressed_data);

        // Write to file
        let mut file = File::create(&backup_path).map_err(NagualError::from)?;
        file.write_all(&compressed_data).map_err(NagualError::from)?;
        file.sync_all().map_err(NagualError::from)?;

        // Count records
        metadata.record_count = self.estimate_record_count(&source_data);

        // Verify if configured
        if self.config.verify_after_backup {
            self.verify_backup(&backup_path, &metadata.checksum)?;
        }

        // Update tracking
        self.last_backup_timestamp = Some(timestamp);

        // Save metadata
        self.save_metadata(&metadata)?;

        info!(
            backup_id = %metadata.id,
            size = metadata.compressed_size,
            "Incremental backup created successfully"
        );

        Ok(metadata)
    }

    /// List all available backups.
    pub fn list_backups(&self) -> Result<Vec<BackupMetadata>> {
        let metadata_dir = self.config.local_backup_dir.join("metadata");
        if !metadata_dir.exists() {
            return Ok(Vec::new());
        }

        let mut backups = Vec::new();

        for entry in fs::read_dir(&metadata_dir).map_err(NagualError::from)? {
            let entry = entry.map_err(NagualError::from)?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false) {
                let content = fs::read_to_string(&path).map_err(NagualError::from)?;
                if let Ok(metadata) = serde_json::from_str::<BackupMetadata>(&content) {
                    backups.push(metadata);
                }
            }
        }

        // Sort by creation time, newest first
        backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(backups)
    }

    /// List backups of a specific type.
    pub fn list_backups_by_type(&self, backup_type: BackupType) -> Result<Vec<BackupMetadata>> {
        let all = self.list_backups()?;
        Ok(all
            .into_iter()
            .filter(|b| b.backup_type == backup_type)
            .collect())
    }

    /// Get a backup by ID.
    pub fn get_backup(&self, backup_id: &str) -> Result<Option<BackupMetadata>> {
        let metadata_path = self
            .config
            .local_backup_dir
            .join("metadata")
            .join(format!("{}.json", backup_id));

        if !metadata_path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&metadata_path).map_err(NagualError::from)?;
        let metadata: BackupMetadata = serde_json::from_str(&content)?;
        Ok(Some(metadata))
    }

    /// Apply retention policy and clean up old backups.
    pub async fn apply_retention_policy(&self) -> Result<usize> {
        let mut deleted_count = 0;

        // Get all full backups
        let full_backups = self.list_backups_by_type(BackupType::Full)?;

        // Keep only the configured number of full backups
        if full_backups.len() > self.config.retain_full_backups {
            let to_delete = &full_backups[self.config.retain_full_backups..];
            for backup in to_delete {
                if let Err(e) = self.delete_backup(&backup.id).await {
                    warn!(backup_id = %backup.id, error = %e, "Failed to delete old backup");
                } else {
                    deleted_count += 1;
                }
            }
        }

        // Clean up orphaned incremental backups
        let incremental_backups = self.list_backups_by_type(BackupType::Incremental)?;
        let valid_parent_ids: std::collections::HashSet<_> = full_backups
            .iter()
            .take(self.config.retain_full_backups)
            .map(|b| b.id.clone())
            .collect();

        for backup in incremental_backups {
            if let Some(ref parent_id) = backup.parent_backup_id {
                if !valid_parent_ids.contains(parent_id) {
                    if let Err(e) = self.delete_backup(&backup.id).await {
                        warn!(backup_id = %backup.id, error = %e, "Failed to delete orphaned backup");
                    } else {
                        deleted_count += 1;
                    }
                }
            }
        }

        info!(deleted = deleted_count, "Retention policy applied");
        Ok(deleted_count)
    }

    /// Delete a backup by ID.
    pub async fn delete_backup(&self, backup_id: &str) -> Result<()> {
        let metadata = self.get_backup(backup_id)?.ok_or_else(|| {
            NagualError::config(format!("Backup not found: {}", backup_id))
        })?;

        // Delete local file
        let local_path = PathBuf::from(&metadata.path);
        if local_path.exists() {
            fs::remove_file(&local_path).map_err(NagualError::from)?;
        }

        // Delete metadata
        let metadata_path = self
            .config
            .local_backup_dir
            .join("metadata")
            .join(format!("{}.json", backup_id));
        if metadata_path.exists() {
            fs::remove_file(&metadata_path).map_err(NagualError::from)?;
        }

        debug!(backup_id = %backup_id, "Backup deleted");
        Ok(())
    }

    // Private helper methods

    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = GzEncoder::new(
            Vec::new(),
            Compression::new(self.config.compression_level),
        );
        encoder.write_all(data).map_err(NagualError::from)?;
        encoder.finish().map_err(NagualError::from)
    }

    fn calculate_checksum(&self, data: &[u8]) -> String {
        use ring::digest::{Context, SHA256};
        let mut context = Context::new(&SHA256);
        context.update(data);
        let digest = context.finish();
        hex::encode(digest.as_ref())
    }

    fn verify_backup(&self, path: &Path, expected_checksum: &str) -> Result<()> {
        let data = fs::read(path).map_err(NagualError::from)?;
        let actual_checksum = self.calculate_checksum(&data);

        if actual_checksum != expected_checksum {
            return Err(NagualError::config(format!(
                "Backup verification failed: checksum mismatch (expected {}, got {})",
                expected_checksum, actual_checksum
            )));
        }

        debug!(path = %path.display(), "Backup verification passed");
        Ok(())
    }

    fn estimate_record_count(&self, data: &[u8]) -> u64 {
        // Simple heuristic: estimate based on SQLite page size
        (data.len() / 100) as u64
    }

    fn save_metadata(&self, metadata: &BackupMetadata) -> Result<()> {
        let metadata_dir = self.config.local_backup_dir.join("metadata");
        if !metadata_dir.exists() {
            fs::create_dir_all(&metadata_dir).map_err(NagualError::from)?;
        }

        let path = metadata_dir.join(format!("{}.json", metadata.id));
        let json = serde_json::to_string_pretty(metadata)?;
        fs::write(&path, json).map_err(NagualError::from)?;

        Ok(())
    }
}

/// Full backup handler that works with GCloud.
pub struct FullBackup {
    adapter: GCloudAdapter,
    config: BackupConfig,
}

impl FullBackup {
    /// Create a new full backup handler.
    pub fn new(adapter: GCloudAdapter, config: BackupConfig) -> Self {
        Self { adapter, config }
    }

    /// Run a full backup.
    pub async fn run(&self) -> Result<BackupResult> {
        let mut manager = BackupManager::new(self.config.clone())?;
        let metadata = manager.create_full_backup().await?;

        // Upload to GCloud if configured
        if self.config.upload_to_gcloud {
            let local_path = PathBuf::from(&metadata.path);
            let object_name = format!("full/{}", local_path.file_name().unwrap().to_string_lossy());
            self.adapter.upload_file(&local_path, &object_name).await
                .map_err(|e| NagualError::internal(format!("Upload failed: {}", e)))?;
        }

        Ok(BackupResult {
            metadata,
            uploaded_to_gcloud: self.config.upload_to_gcloud,
        })
    }
}

/// Result of a backup operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupResult {
    /// Backup metadata
    pub metadata: BackupMetadata,
    /// Whether the backup was uploaded to GCloud
    pub uploaded_to_gcloud: bool,
}

/// Scheduler for periodic backups.
pub struct BackupScheduler {
    backup: FullBackup,
    running: std::sync::Arc<tokio::sync::RwLock<bool>>,
    interval: std::time::Duration,
}

impl BackupScheduler {
    /// Create a new backup scheduler.
    pub fn new(backup: FullBackup, interval: std::time::Duration) -> Self {
        Self {
            backup,
            running: std::sync::Arc::new(tokio::sync::RwLock::new(false)),
            interval,
        }
    }

    /// Create with default 6-hour interval.
    pub fn with_default_interval(backup: FullBackup) -> Self {
        Self::new(backup, std::time::Duration::from_secs(6 * 60 * 60))
    }

    /// Start the scheduler.
    pub async fn start(&self) {
        let mut running = self.running.write().await;
        if *running {
            warn!("Backup scheduler already running");
            return;
        }
        *running = true;
        drop(running);

        let backup = self.backup.clone();
        let running = std::sync::Arc::clone(&self.running);
        let interval = self.interval;

        tokio::spawn(async move {
            info!(
                interval_secs = interval.as_secs(),
                "Started backup scheduler"
            );

            loop {
                if !*running.read().await {
                    info!("Backup scheduler stopped");
                    break;
                }

                tokio::time::sleep(interval).await;

                match backup.run().await {
                    Ok(result) => {
                        info!(
                            backup_id = %result.metadata.id,
                            size = result.metadata.compressed_size,
                            "Scheduled backup completed"
                        );
                    }
                    Err(e) => {
                        warn!(error = %e, "Scheduled backup failed");
                    }
                }
            }
        });
    }

    /// Stop the scheduler.
    pub async fn stop(&self) {
        *self.running.write().await = false;
    }

    /// Check if the scheduler is running.
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }
}

impl Clone for FullBackup {
    fn clone(&self) -> Self {
        Self {
            adapter: self.adapter.clone(),
            config: self.config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_db(dir: &TempDir) -> PathBuf {
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT);
             INSERT INTO test VALUES (1, 'test');",
        )
        .unwrap();
        db_path
    }

    #[test]
    fn test_backup_config_default() {
        let config = BackupConfig::default();
        assert_eq!(config.compression_level, 6);
        assert_eq!(config.retain_full_backups, 7);
        assert!(!config.upload_to_gcloud);
    }

    #[test]
    fn test_backup_metadata_new() {
        let metadata = BackupMetadata::new(
            BackupType::Full,
            "/path/to/db",
            "/path/to/backup.gz",
        );
        assert_eq!(metadata.backup_type, BackupType::Full);
        assert!(!metadata.id.is_empty());
    }

    #[test]
    fn test_compression_ratio() {
        let mut metadata = BackupMetadata::new(BackupType::Full, "src", "dst");
        metadata.original_size = 1000;
        metadata.compressed_size = 300;
        metadata.calculate_compression_ratio();
        assert!((metadata.compression_ratio - 0.7).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_create_full_backup() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = create_test_db(&temp_dir);
        let backup_dir = temp_dir.path().join("backups");

        let config = BackupConfig::new(&db_path, &backup_dir)
            .with_compression_level(1);

        let mut manager = BackupManager::new(config).unwrap();
        let metadata = manager.create_full_backup().await.unwrap();

        assert_eq!(metadata.backup_type, BackupType::Full);
        assert!(metadata.compressed_size > 0);
        assert!(!metadata.checksum.is_empty());
    }

    #[tokio::test]
    async fn test_list_backups() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = create_test_db(&temp_dir);
        let backup_dir = temp_dir.path().join("backups");

        let config = BackupConfig::new(&db_path, &backup_dir);
        let mut manager = BackupManager::new(config).unwrap();

        manager.create_full_backup().await.unwrap();

        let backups = manager.list_backups().unwrap();
        assert_eq!(backups.len(), 1);
        assert_eq!(backups[0].backup_type, BackupType::Full);
    }

    #[tokio::test]
    async fn test_incremental_backup_requires_full() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = create_test_db(&temp_dir);
        let backup_dir = temp_dir.path().join("backups");

        let config = BackupConfig::new(&db_path, &backup_dir);
        let mut manager = BackupManager::new(config).unwrap();

        let result = manager.create_incremental_backup().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_incremental_backup_after_full() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = create_test_db(&temp_dir);
        let backup_dir = temp_dir.path().join("backups");

        let config = BackupConfig::new(&db_path, &backup_dir);
        let mut manager = BackupManager::new(config).unwrap();

        let full = manager.create_full_backup().await.unwrap();
        let incr = manager.create_incremental_backup().await.unwrap();

        assert_eq!(incr.backup_type, BackupType::Incremental);
        assert_eq!(incr.parent_backup_id, Some(full.id));
    }
}
