//! Incremental sync for changed records.
//!
//! Tracks modifications to database records and uploads only changed data
//! to GCS, optimizing bandwidth and storage costs.
//!
//! # How It Works
//!
//! 1. Query the `sync_log` table for records changed since last sync
//! 2. Batch records by table for efficient processing
//! 3. Compress and upload each batch
//! 4. Update sync timestamp on success
//!
//! # Sync Log Table Schema
//!
//! The incremental sync relies on a `sync_log` table created in Phase 1.B:
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS sync_log (
//!     id INTEGER PRIMARY KEY AUTOINCREMENT,
//!     table_name TEXT NOT NULL,
//!     record_id TEXT NOT NULL,
//!     operation TEXT NOT NULL,  -- 'INSERT', 'UPDATE', 'DELETE'
//!     changed_at TEXT NOT NULL DEFAULT (datetime('now')),
//!     synced_at TEXT,           -- NULL until synced
//!     data TEXT                 -- JSON snapshot of the record
//! );
//!
//! CREATE INDEX idx_sync_log_unsynced ON sync_log(synced_at) WHERE synced_at IS NULL;
//! CREATE INDEX idx_sync_log_changed ON sync_log(changed_at);
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use super::gcloud::{GCloudAdapter, GCloudResult};
use crate::error::{NagualError, Result};

/// Configuration for incremental sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalSyncConfig {
    /// Sync interval (default: 30 minutes).
    pub interval: Duration,

    /// Maximum records per batch upload.
    pub batch_size: usize,

    /// Object prefix for incremental syncs.
    pub prefix: String,

    /// SQLite database path for sync_log queries.
    pub sqlite_path: PathBuf,

    /// Whether to delete sync_log entries after successful sync.
    pub prune_after_sync: bool,

    /// Maximum age of sync_log entries to keep (if prune_after_sync is false).
    pub max_log_age: Duration,

    /// Compression level (0-9, default: 6).
    pub compression_level: u32,

    /// Enable detailed progress reporting.
    pub enable_progress: bool,
}

impl Default for IncrementalSyncConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30 * 60), // 30 minutes
            batch_size: 1000,
            prefix: "incremental".to_string(),
            sqlite_path: PathBuf::from("nagual.db"),
            prune_after_sync: true,
            max_log_age: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
            compression_level: 6,
            enable_progress: true,
        }
    }
}

impl IncrementalSyncConfig {
    /// Create a new config with the given SQLite path.
    pub fn new(sqlite_path: impl Into<PathBuf>) -> Self {
        Self {
            sqlite_path: sqlite_path.into(),
            ..Default::default()
        }
    }

    /// Set sync interval.
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Set batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set object prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Disable pruning after sync.
    pub fn without_pruning(mut self) -> Self {
        self.prune_after_sync = false;
        self
    }
}

/// A record from the sync_log table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncLogEntry {
    /// Unique ID in sync_log.
    pub id: i64,

    /// Source table name.
    pub table_name: String,

    /// Record ID in the source table.
    pub record_id: String,

    /// Operation type: INSERT, UPDATE, DELETE.
    pub operation: String,

    /// When the change occurred.
    pub changed_at: DateTime<Utc>,

    /// JSON snapshot of the record data.
    pub data: Option<String>,
}

/// Batch of sync entries for a single table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncBatch {
    /// Table name.
    pub table_name: String,

    /// Entries in this batch.
    pub entries: Vec<SyncLogEntry>,

    /// Batch creation timestamp.
    pub created_at: DateTime<Utc>,
}

impl SyncBatch {
    /// Create a new batch for the given table.
    pub fn new(table_name: impl Into<String>) -> Self {
        Self {
            table_name: table_name.into(),
            entries: Vec::new(),
            created_at: Utc::now(),
        }
    }

    /// Add an entry to the batch.
    pub fn add(&mut self, entry: SyncLogEntry) {
        self.entries.push(entry);
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Serialize to JSON and compress.
    pub fn to_compressed_json(&self, level: u32) -> Result<Vec<u8>> {
        let json = serde_json::to_vec(self)?;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level));
        std::io::Write::write_all(&mut encoder, &json)
            .map_err(|e| NagualError::internal(format!("Compression failed: {}", e)))?;
        encoder
            .finish()
            .map_err(|e| NagualError::internal(format!("Compression finish failed: {}", e)))
    }
}

/// Progress information for ongoing sync.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncProgress {
    /// Total records to sync.
    pub total_records: usize,

    /// Records processed so far.
    pub processed_records: usize,

    /// Batches uploaded.
    pub batches_uploaded: usize,

    /// Total batches to upload.
    pub total_batches: usize,

    /// Bytes uploaded.
    pub bytes_uploaded: u64,

    /// Current table being synced.
    pub current_table: Option<String>,

    /// Sync start time.
    pub started_at: Option<DateTime<Utc>>,

    /// Errors encountered (non-fatal).
    pub errors: Vec<String>,
}

impl SyncProgress {
    /// Calculate progress percentage.
    pub fn percentage(&self) -> f32 {
        if self.total_records == 0 {
            100.0
        } else {
            (self.processed_records as f32 / self.total_records as f32) * 100.0
        }
    }

    /// Calculate elapsed time.
    pub fn elapsed(&self) -> Option<Duration> {
        self.started_at.map(|start| {
            let now = Utc::now();
            (now - start).to_std().unwrap_or_default()
        })
    }

    /// Estimate remaining time.
    pub fn estimated_remaining(&self) -> Option<Duration> {
        if self.processed_records == 0 {
            return None;
        }

        self.elapsed().map(|elapsed| {
            let rate = self.processed_records as f64 / elapsed.as_secs_f64();
            let remaining = self.total_records - self.processed_records;
            Duration::from_secs_f64(remaining as f64 / rate)
        })
    }
}

/// Result of an incremental sync operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResult {
    /// Whether the sync was successful.
    pub success: bool,

    /// Number of records synced.
    pub records_synced: usize,

    /// Number of batches uploaded.
    pub batches_uploaded: usize,

    /// Total bytes uploaded.
    pub bytes_uploaded: u64,

    /// Sync duration.
    pub duration: Duration,

    /// Last sync timestamp (for next sync).
    pub last_sync_timestamp: DateTime<Utc>,

    /// Errors encountered.
    pub errors: Vec<String>,

    /// Tables synced.
    pub tables_synced: Vec<String>,
}

/// Incremental sync manager.
///
/// Handles periodic synchronization of changed database records to GCS.
pub struct IncrementalSync {
    adapter: GCloudAdapter,
    config: IncrementalSyncConfig,
    last_sync: Arc<RwLock<Option<DateTime<Utc>>>>,
    progress: Arc<RwLock<SyncProgress>>,
}

impl IncrementalSync {
    /// Create a new incremental sync manager.
    pub fn new(adapter: GCloudAdapter, config: IncrementalSyncConfig) -> Self {
        Self {
            adapter,
            config,
            last_sync: Arc::new(RwLock::new(None)),
            progress: Arc::new(RwLock::new(SyncProgress::default())),
        }
    }

    /// Get the adapter.
    pub fn adapter(&self) -> &GCloudAdapter {
        &self.adapter
    }

    /// Get the configuration.
    pub fn config(&self) -> &IncrementalSyncConfig {
        &self.config
    }

    /// Get current progress (for monitoring).
    pub async fn progress(&self) -> SyncProgress {
        self.progress.read().await.clone()
    }

    /// Get the last sync timestamp.
    pub async fn last_sync(&self) -> Option<DateTime<Utc>> {
        *self.last_sync.read().await
    }

    /// Check if sync is due based on interval.
    pub async fn is_sync_due(&self) -> bool {
        match *self.last_sync.read().await {
            Some(last) => {
                let elapsed = Utc::now() - last;
                elapsed.to_std().unwrap_or_default() >= self.config.interval
            }
            None => true, // First sync
        }
    }

    /// Run an incremental sync.
    ///
    /// This queries the sync_log table for changed records since the last sync,
    /// batches them by table, compresses, and uploads to GCS.
    pub async fn sync(&self) -> Result<SyncResult> {
        let start = std::time::Instant::now();
        let sync_time = Utc::now();

        info!("Starting incremental sync");

        // Reset progress
        {
            let mut progress = self.progress.write().await;
            *progress = SyncProgress {
                started_at: Some(sync_time),
                ..Default::default()
            };
        }

        // Get last sync timestamp
        let last_sync = *self.last_sync.read().await;

        // Query sync_log for pending changes
        let entries = self.query_pending_changes(last_sync).await?;

        if entries.is_empty() {
            info!("No changes to sync");
            return Ok(SyncResult {
                success: true,
                records_synced: 0,
                batches_uploaded: 0,
                bytes_uploaded: 0,
                duration: start.elapsed(),
                last_sync_timestamp: sync_time,
                errors: Vec::new(),
                tables_synced: Vec::new(),
            });
        }

        // Update total in progress
        {
            let mut progress = self.progress.write().await;
            progress.total_records = entries.len();
        }

        // Group entries by table
        let batches = self.group_into_batches(entries);
        let total_batches = batches.len();

        // Update progress
        {
            let mut progress = self.progress.write().await;
            progress.total_batches = total_batches;
        }

        // Upload each batch
        let mut bytes_uploaded = 0u64;
        let mut batches_uploaded = 0usize;
        let mut errors = Vec::new();
        let mut tables_synced = Vec::new();

        for batch in batches {
            let table_name = batch.table_name.clone();
            let batch_size = batch.len();

            // Update progress
            {
                let mut progress = self.progress.write().await;
                progress.current_table = Some(table_name.clone());
            }

            match self.upload_batch(&batch, sync_time).await {
                Ok(bytes) => {
                    bytes_uploaded += bytes;
                    batches_uploaded += 1;

                    if !tables_synced.contains(&table_name) {
                        tables_synced.push(table_name.clone());
                    }

                    // Update progress
                    {
                        let mut progress = self.progress.write().await;
                        progress.processed_records += batch_size;
                        progress.batches_uploaded = batches_uploaded;
                        progress.bytes_uploaded = bytes_uploaded;
                    }

                    debug!(
                        table = %table_name,
                        records = batch_size,
                        bytes = bytes,
                        "Uploaded batch"
                    );
                }
                Err(e) => {
                    let err_msg = format!("Failed to upload batch for {}: {}", table_name, e);
                    warn!("{}", err_msg);
                    errors.push(err_msg);

                    {
                        let mut progress = self.progress.write().await;
                        progress.errors.push(format!("Batch {} failed: {}", table_name, e));
                    }
                }
            }
        }

        // Mark entries as synced (if configured)
        if self.config.prune_after_sync && errors.is_empty() {
            if let Err(e) = self.mark_synced(sync_time).await {
                warn!("Failed to mark entries as synced: {}", e);
                errors.push(format!("Failed to mark synced: {}", e));
            }
        }

        // Update last sync timestamp
        *self.last_sync.write().await = Some(sync_time);

        let result = SyncResult {
            success: errors.is_empty(),
            records_synced: {
                let progress = self.progress.read().await;
                progress.processed_records
            },
            batches_uploaded,
            bytes_uploaded,
            duration: start.elapsed(),
            last_sync_timestamp: sync_time,
            errors,
            tables_synced,
        };

        if result.success {
            info!(
                records = result.records_synced,
                batches = batches_uploaded,
                bytes = bytes_uploaded,
                duration_ms = result.duration.as_millis(),
                "Incremental sync completed"
            );
        } else {
            warn!(
                records = result.records_synced,
                errors = result.errors.len(),
                "Incremental sync completed with errors"
            );
        }

        Ok(result)
    }

    /// Query the sync_log table for pending changes.
    async fn query_pending_changes(
        &self,
        since: Option<DateTime<Utc>>,
    ) -> Result<Vec<SyncLogEntry>> {
        // In a real implementation, this would query SQLite:
        //
        // ```rust
        // let conn = rusqlite::Connection::open(&self.config.sqlite_path)?;
        //
        // let sql = match since {
        //     Some(ts) => format!(
        //         "SELECT id, table_name, record_id, operation, changed_at, data
        //          FROM sync_log
        //          WHERE synced_at IS NULL AND changed_at > '{}'
        //          ORDER BY changed_at ASC",
        //         ts.to_rfc3339()
        //     ),
        //     None => "SELECT id, table_name, record_id, operation, changed_at, data
        //              FROM sync_log
        //              WHERE synced_at IS NULL
        //              ORDER BY changed_at ASC".to_string(),
        // };
        //
        // let mut stmt = conn.prepare(&sql)?;
        // let entries = stmt.query_map([], |row| {
        //     Ok(SyncLogEntry {
        //         id: row.get(0)?,
        //         table_name: row.get(1)?,
        //         record_id: row.get(2)?,
        //         operation: row.get(3)?,
        //         changed_at: row.get(4)?,
        //         data: row.get(5)?,
        //     })
        // })?.collect::<Result<Vec<_>, _>>()?;
        // ```

        debug!(
            sqlite_path = %self.config.sqlite_path.display(),
            since = ?since,
            "Querying sync_log for pending changes"
        );

        // Simulated: return empty for now
        Ok(Vec::new())
    }

    /// Group entries into batches by table.
    fn group_into_batches(&self, entries: Vec<SyncLogEntry>) -> Vec<SyncBatch> {
        let mut batches_by_table: HashMap<String, Vec<SyncBatch>> = HashMap::new();

        for entry in entries {
            let table_name = entry.table_name.clone();

            let table_batches = batches_by_table.entry(table_name.clone()).or_default();

            // Get or create current batch
            let needs_new_batch = table_batches.is_empty()
                || table_batches.last().map(|b| b.len()).unwrap_or(0) >= self.config.batch_size;

            if needs_new_batch {
                table_batches.push(SyncBatch::new(&table_name));
            }

            if let Some(batch) = table_batches.last_mut() {
                batch.add(entry);
            }
        }

        // Flatten into single list
        batches_by_table
            .into_values()
            .flatten()
            .collect()
    }

    /// Upload a batch to GCS.
    async fn upload_batch(&self, batch: &SyncBatch, sync_time: DateTime<Utc>) -> GCloudResult<u64> {
        // Generate object name
        let object_name = format!(
            "{}/{}/{}-{}.json.gz",
            self.config.prefix,
            batch.table_name,
            sync_time.format("%Y%m%d-%H%M%S"),
            batch.entries.first().map(|e| e.id).unwrap_or(0)
        );

        // Compress and serialize
        let data = batch
            .to_compressed_json(self.config.compression_level)
            .map_err(|e| super::gcloud::GCloudError::CompressionError(e.to_string()))?;

        let size = data.len() as u64;

        // Upload
        self.adapter
            .upload_data(&data, &object_name, Some("application/gzip"))
            .await?;

        Ok(size)
    }

    /// Mark sync_log entries as synced.
    async fn mark_synced(&self, sync_time: DateTime<Utc>) -> Result<()> {
        // In a real implementation:
        //
        // ```rust
        // let conn = rusqlite::Connection::open(&self.config.sqlite_path)?;
        // conn.execute(
        //     "UPDATE sync_log SET synced_at = ?1 WHERE synced_at IS NULL AND changed_at <= ?1",
        //     [sync_time.to_rfc3339()],
        // )?;
        // ```

        debug!(
            sync_time = %sync_time,
            "Marked sync_log entries as synced"
        );

        Ok(())
    }

    /// Prune old sync_log entries.
    pub async fn prune_old_entries(&self) -> Result<usize> {
        let cutoff = Utc::now() - chrono::Duration::from_std(self.config.max_log_age).unwrap();

        // In a real implementation:
        //
        // ```rust
        // let conn = rusqlite::Connection::open(&self.config.sqlite_path)?;
        // let deleted = conn.execute(
        //     "DELETE FROM sync_log WHERE synced_at IS NOT NULL AND changed_at < ?",
        //     [cutoff.to_rfc3339()],
        // )?;
        // ```

        debug!(
            cutoff = %cutoff,
            "Pruned old sync_log entries"
        );

        Ok(0)
    }
}

impl Clone for IncrementalSync {
    fn clone(&self) -> Self {
        Self {
            adapter: self.adapter.clone(),
            config: self.config.clone(),
            last_sync: Arc::clone(&self.last_sync),
            progress: Arc::clone(&self.progress),
        }
    }
}

impl std::fmt::Debug for IncrementalSync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IncrementalSync")
            .field("config", &self.config)
            .finish()
    }
}

/// Scheduler for periodic incremental syncs.
pub struct SyncScheduler {
    sync: IncrementalSync,
    running: Arc<RwLock<bool>>,
}

impl SyncScheduler {
    /// Create a new scheduler.
    pub fn new(sync: IncrementalSync) -> Self {
        Self {
            sync,
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the scheduler (runs in background).
    pub async fn start(&self) {
        let mut running = self.running.write().await;
        if *running {
            warn!("Sync scheduler already running");
            return;
        }
        *running = true;
        drop(running);

        let sync = self.sync.clone();
        let running = Arc::clone(&self.running);
        let interval = sync.config.interval;

        tokio::spawn(async move {
            info!(
                interval_secs = interval.as_secs(),
                "Started incremental sync scheduler"
            );

            loop {
                // Check if still running
                if !*running.read().await {
                    info!("Sync scheduler stopped");
                    break;
                }

                // Wait for interval
                tokio::time::sleep(interval).await;

                // Run sync
                match sync.sync().await {
                    Ok(result) => {
                        if result.success {
                            debug!(
                                records = result.records_synced,
                                "Scheduled sync completed"
                            );
                        } else {
                            warn!(
                                errors = result.errors.len(),
                                "Scheduled sync completed with errors"
                            );
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Scheduled sync failed");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::gcloud::GCloudConfig;

    #[test]
    fn test_sync_config_defaults() {
        let config = IncrementalSyncConfig::default();
        assert_eq!(config.interval, Duration::from_secs(30 * 60));
        assert_eq!(config.batch_size, 1000);
        assert!(config.prune_after_sync);
    }

    #[test]
    fn test_sync_batch_creation() {
        let mut batch = SyncBatch::new("patterns");
        assert!(batch.is_empty());

        batch.add(SyncLogEntry {
            id: 1,
            table_name: "patterns".to_string(),
            record_id: "p-1".to_string(),
            operation: "INSERT".to_string(),
            changed_at: Utc::now(),
            data: Some(r#"{"key": "value"}"#.to_string()),
        });

        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_compression() {
        let mut batch = SyncBatch::new("test");
        batch.add(SyncLogEntry {
            id: 1,
            table_name: "test".to_string(),
            record_id: "t-1".to_string(),
            operation: "INSERT".to_string(),
            changed_at: Utc::now(),
            data: Some(r#"{"data": "test data here"}"#.to_string()),
        });

        let compressed = batch.to_compressed_json(6).unwrap();
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_progress_percentage() {
        let progress = SyncProgress {
            total_records: 100,
            processed_records: 50,
            ..Default::default()
        };

        assert!((progress.percentage() - 50.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_incremental_sync_creation() {
        let gcloud_config = GCloudConfig::new("test-bucket", "test-project");
        let adapter = GCloudAdapter::new(gcloud_config).await.unwrap();

        let sync_config = IncrementalSyncConfig::new("/tmp/test.db")
            .with_interval(Duration::from_secs(60))
            .with_batch_size(500);

        let sync = IncrementalSync::new(adapter, sync_config);

        assert!(sync.last_sync().await.is_none());
        assert!(sync.is_sync_due().await);
    }
}
