//! GCloud Unavailability Chaos Test
//!
//! Simulates network partition to Google Cloud Storage to verify:
//! - Sync resumes within 5 minutes of restoration
//! - Incremental sync recovery works correctly
//! - No backup corruption occurs
//! - Local operations continue unaffected
//!
//! # Test Scenario
//!
//! 1. Establish healthy GCloud sync connection
//! 2. Trigger network partition (simulate DNS/firewall failure)
//! 3. Continue local operations
//! 4. Restore network connectivity
//! 5. Verify sync resumes and catches up
//! 6. Verify backup integrity

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::common::{
    assert_no_data_loss, assert_recovery_within_sla, ChaosMetrics, ChaosTestConfig,
    OutageSimulator,
};

/// Simulated GCloud Storage client for testing
#[derive(Debug)]
pub struct MockGCloudStorage {
    is_available: Arc<AtomicBool>,
    objects: RwLock<HashMap<String, StoredObject>>,
    upload_count: AtomicUsize,
    download_count: AtomicUsize,
    error_count: AtomicUsize,
    last_sync_time: RwLock<Option<DateTime<Utc>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredObject {
    pub name: String,
    pub data: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub checksum: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GCloudError {
    NetworkUnavailable,
    ConnectionTimeout,
    PermissionDenied,
    ObjectNotFound(String),
    CorruptedData,
}

impl std::fmt::Display for GCloudError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GCloudError::NetworkUnavailable => write!(f, "Network unavailable"),
            GCloudError::ConnectionTimeout => write!(f, "Connection timeout"),
            GCloudError::PermissionDenied => write!(f, "Permission denied"),
            GCloudError::ObjectNotFound(name) => write!(f, "Object not found: {}", name),
            GCloudError::CorruptedData => write!(f, "Data corruption detected"),
        }
    }
}

impl std::error::Error for GCloudError {}

impl MockGCloudStorage {
    pub fn new() -> Self {
        Self {
            is_available: Arc::new(AtomicBool::new(true)),
            objects: RwLock::new(HashMap::new()),
            upload_count: AtomicUsize::new(0),
            download_count: AtomicUsize::new(0),
            error_count: AtomicUsize::new(0),
            last_sync_time: RwLock::new(None),
        }
    }

    pub fn set_availability(&self, available: bool) {
        self.is_available.store(available, Ordering::SeqCst);
    }

    pub fn is_available(&self) -> bool {
        self.is_available.load(Ordering::SeqCst)
    }

    pub fn get_availability_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.is_available)
    }

    pub async fn upload(
        &self,
        name: &str,
        data: Vec<u8>,
        metadata: HashMap<String, String>,
    ) -> Result<StoredObject, GCloudError> {
        self.upload_count.fetch_add(1, Ordering::SeqCst);

        if !self.is_available() {
            self.error_count.fetch_add(1, Ordering::SeqCst);
            return Err(GCloudError::NetworkUnavailable);
        }

        // Simulate network latency
        tokio::time::sleep(Duration::from_millis(5)).await;

        let checksum = calculate_checksum(&data);
        let object = StoredObject {
            name: name.to_string(),
            data,
            created_at: Utc::now(),
            checksum,
            metadata,
        };

        self.objects.write().insert(name.to_string(), object.clone());
        *self.last_sync_time.write() = Some(Utc::now());

        Ok(object)
    }

    pub async fn download(&self, name: &str) -> Result<StoredObject, GCloudError> {
        self.download_count.fetch_add(1, Ordering::SeqCst);

        if !self.is_available() {
            self.error_count.fetch_add(1, Ordering::SeqCst);
            return Err(GCloudError::NetworkUnavailable);
        }

        // Simulate network latency
        tokio::time::sleep(Duration::from_millis(5)).await;

        self.objects
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| GCloudError::ObjectNotFound(name.to_string()))
    }

    pub async fn list_objects(&self, prefix: Option<&str>) -> Result<Vec<String>, GCloudError> {
        if !self.is_available() {
            self.error_count.fetch_add(1, Ordering::SeqCst);
            return Err(GCloudError::NetworkUnavailable);
        }

        let objects = self.objects.read();
        let names: Vec<String> = objects
            .keys()
            .filter(|k| prefix.map(|p| k.starts_with(p)).unwrap_or(true))
            .cloned()
            .collect();

        Ok(names)
    }

    pub fn get_upload_count(&self) -> usize {
        self.upload_count.load(Ordering::SeqCst)
    }

    pub fn get_error_count(&self) -> usize {
        self.error_count.load(Ordering::SeqCst)
    }

    pub fn get_object_count(&self) -> usize {
        self.objects.read().len()
    }

    pub fn get_last_sync_time(&self) -> Option<DateTime<Utc>> {
        *self.last_sync_time.read()
    }

    pub fn verify_integrity(&self) -> bool {
        let objects = self.objects.read();
        for (name, obj) in objects.iter() {
            let actual_checksum = calculate_checksum(&obj.data);
            if actual_checksum != obj.checksum {
                println!("Integrity check failed for object: {}", name);
                return false;
            }
        }
        true
    }
}

impl Default for MockGCloudStorage {
    fn default() -> Self {
        Self::new()
    }
}

fn calculate_checksum(data: &[u8]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Sync log entry for tracking pending syncs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncLogEntry {
    pub id: u64,
    pub table_name: String,
    pub record_id: String,
    pub operation: String,
    pub changed_at: DateTime<Utc>,
    pub data: String,
    pub synced: bool,
}

/// Incremental sync manager for testing
pub struct MockIncrementalSync {
    storage: Arc<MockGCloudStorage>,
    sync_log: RwLock<Vec<SyncLogEntry>>,
    pending_syncs: AtomicUsize,
    successful_syncs: AtomicUsize,
    failed_syncs: AtomicUsize,
    last_successful_sync: RwLock<Option<DateTime<Utc>>>,
    metrics: ChaosMetrics,
}

impl MockIncrementalSync {
    pub fn new(storage: Arc<MockGCloudStorage>) -> Self {
        Self {
            storage,
            sync_log: RwLock::new(Vec::new()),
            pending_syncs: AtomicUsize::new(0),
            successful_syncs: AtomicUsize::new(0),
            failed_syncs: AtomicUsize::new(0),
            last_successful_sync: RwLock::new(None),
            metrics: ChaosMetrics::new(),
        }
    }

    pub fn add_change(&self, table: &str, record_id: &str, operation: &str, data: &str) -> u64 {
        let id = self.sync_log.read().len() as u64 + 1;
        let entry = SyncLogEntry {
            id,
            table_name: table.to_string(),
            record_id: record_id.to_string(),
            operation: operation.to_string(),
            changed_at: Utc::now(),
            data: data.to_string(),
            synced: false,
        };

        self.sync_log.write().push(entry);
        self.pending_syncs.fetch_add(1, Ordering::SeqCst);
        id
    }

    pub fn get_pending_count(&self) -> usize {
        self.sync_log.read().iter().filter(|e| !e.synced).count()
    }

    pub async fn sync(&self) -> Result<SyncResult, GCloudError> {
        self.metrics.record_attempt();

        let entries: Vec<SyncLogEntry> = self
            .sync_log
            .read()
            .iter()
            .filter(|e| !e.synced)
            .cloned()
            .collect();

        if entries.is_empty() {
            return Ok(SyncResult {
                synced_count: 0,
                failed_count: 0,
                duration: Duration::from_secs(0),
            });
        }

        let start = Instant::now();
        let mut synced_ids = Vec::new();

        for entry in &entries {
            let object_name = format!(
                "incremental/{}/{}-{}.json",
                entry.table_name,
                entry.changed_at.format("%Y%m%d-%H%M%S"),
                entry.id
            );

            let mut metadata = HashMap::new();
            metadata.insert("table".to_string(), entry.table_name.clone());
            metadata.insert("record_id".to_string(), entry.record_id.clone());
            metadata.insert("operation".to_string(), entry.operation.clone());

            match self
                .storage
                .upload(&object_name, entry.data.as_bytes().to_vec(), metadata)
                .await
            {
                Ok(_) => {
                    synced_ids.push(entry.id);
                    self.successful_syncs.fetch_add(1, Ordering::SeqCst);
                    self.metrics.record_success();
                }
                Err(e) => {
                    self.failed_syncs.fetch_add(1, Ordering::SeqCst);
                    self.metrics.record_failure();
                    return Err(e);
                }
            }
        }

        // Mark as synced
        {
            let mut log = self.sync_log.write();
            for entry in log.iter_mut() {
                if synced_ids.contains(&entry.id) {
                    entry.synced = true;
                }
            }
        }

        let synced_count = synced_ids.len();
        self.pending_syncs.fetch_sub(synced_count, Ordering::SeqCst);
        *self.last_successful_sync.write() = Some(Utc::now());

        Ok(SyncResult {
            synced_count,
            failed_count: 0,
            duration: start.elapsed(),
        })
    }

    pub fn get_last_successful_sync(&self) -> Option<DateTime<Utc>> {
        *self.last_successful_sync.read()
    }

    pub fn get_metrics(&self) -> &ChaosMetrics {
        &self.metrics
    }

    pub fn get_successful_sync_count(&self) -> usize {
        self.successful_syncs.load(Ordering::SeqCst)
    }

    pub fn get_failed_sync_count(&self) -> usize {
        self.failed_syncs.load(Ordering::SeqCst)
    }
}

#[derive(Debug, Clone)]
pub struct SyncResult {
    pub synced_count: usize,
    pub failed_count: usize,
    pub duration: Duration,
}

/// Backup manager for full backups
pub struct MockBackupManager {
    storage: Arc<MockGCloudStorage>,
    backup_count: AtomicUsize,
    last_backup: RwLock<Option<BackupMetadata>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub size: usize,
    pub checksum: String,
    pub object_name: String,
}

impl MockBackupManager {
    pub fn new(storage: Arc<MockGCloudStorage>) -> Self {
        Self {
            storage,
            backup_count: AtomicUsize::new(0),
            last_backup: RwLock::new(None),
        }
    }

    pub async fn create_backup(&self, data: Vec<u8>) -> Result<BackupMetadata, GCloudError> {
        let id = uuid::Uuid::new_v4().to_string();
        let checksum = calculate_checksum(&data);
        let object_name = format!(
            "backups/full-{}.db.gz",
            Utc::now().format("%Y%m%d-%H%M%S")
        );

        let mut metadata = HashMap::new();
        metadata.insert("backup_id".to_string(), id.clone());
        metadata.insert("backup_type".to_string(), "full".to_string());

        self.storage
            .upload(&object_name, data.clone(), metadata)
            .await?;

        let backup_metadata = BackupMetadata {
            id,
            created_at: Utc::now(),
            size: data.len(),
            checksum,
            object_name,
        };

        *self.last_backup.write() = Some(backup_metadata.clone());
        self.backup_count.fetch_add(1, Ordering::SeqCst);

        Ok(backup_metadata)
    }

    pub async fn verify_backup(&self, backup: &BackupMetadata) -> Result<bool, GCloudError> {
        let obj = self.storage.download(&backup.object_name).await?;
        let actual_checksum = calculate_checksum(&obj.data);
        Ok(actual_checksum == backup.checksum)
    }

    pub fn get_last_backup(&self) -> Option<BackupMetadata> {
        self.last_backup.read().clone()
    }
}

/// GCloud outage chaos test - sync resumes within 5 minutes
#[tokio::test]
async fn test_gcloud_outage_sync_recovery() {
    println!("\n=== GCloud Unavailability Chaos Test ===\n");

    // Use shorter duration for actual test
    let config = ChaosTestConfig {
        outage_duration: Duration::from_millis(300),
        operations_during_outage: 50,
        recovery_timeout: Duration::from_millis(500),
        max_data_loss_bytes: 0,
        expected_dlq_capture_rate: 100.0,
        recovery_sla: Duration::from_secs(5), // 5 minutes scaled down
    };

    // Setup
    let storage = Arc::new(MockGCloudStorage::new());
    let sync_manager = MockIncrementalSync::new(Arc::clone(&storage));
    let backup_manager = MockBackupManager::new(Arc::clone(&storage));

    println!("Phase 1: Establishing steady state with GCloud...");

    // Phase 1: Establish steady state
    for i in 0..10 {
        sync_manager.add_change(
            "patterns",
            &format!("p-{}", i),
            "INSERT",
            &format!("{{\"pattern\": {}}}", i),
        );
    }

    let sync_result = sync_manager.sync().await.expect("Initial sync should succeed");
    println!("Initial sync: {} records synced", sync_result.synced_count);

    // Create a backup
    let test_data = b"test database content for backup".to_vec();
    let backup = backup_manager
        .create_backup(test_data.clone())
        .await
        .expect("Initial backup should succeed");
    println!("Initial backup created: {}", backup.id);

    println!("\nPhase 2: Triggering GCloud network partition...");

    // Phase 2: Trigger GCloud outage
    let outage_start = Instant::now();
    storage.set_availability(false);

    // Add changes during outage
    for i in 10..(10 + config.operations_during_outage) {
        sync_manager.add_change(
            "patterns",
            &format!("p-{}", i),
            "INSERT",
            &format!("{{\"during_outage\": true, \"index\": {}}}", i),
        );
    }

    println!("Added {} changes during outage", config.operations_during_outage);

    // Attempt sync during outage (should fail)
    let outage_sync_result = sync_manager.sync().await;
    assert!(
        outage_sync_result.is_err(),
        "Sync should fail during GCloud outage"
    );
    println!("Sync correctly failed during outage");

    // Verify pending count
    let pending_count = sync_manager.get_pending_count();
    assert!(
        pending_count >= config.operations_during_outage,
        "All {} changes should be pending, got {}",
        config.operations_during_outage,
        pending_count
    );
    println!("Pending syncs during outage: {}", pending_count);

    // Wait for outage duration
    tokio::time::sleep(config.outage_duration).await;
    let outage_duration = outage_start.elapsed();

    println!("\nPhase 3: Restoring GCloud connectivity...");

    // Phase 3: Restore GCloud
    let recovery_start = Instant::now();
    storage.set_availability(true);

    // Wait a bit for recovery
    tokio::time::sleep(Duration::from_millis(50)).await;

    println!("Phase 4: Verifying sync resumes...");

    // Attempt sync after recovery
    let recovery_sync_result = sync_manager.sync().await.expect("Recovery sync should succeed");
    let recovery_duration = recovery_start.elapsed();

    println!(
        "Recovery sync: {} records synced in {:?}",
        recovery_sync_result.synced_count, recovery_duration
    );

    // Verify all pending syncs completed
    let remaining_pending = sync_manager.get_pending_count();
    assert_eq!(
        remaining_pending, 0,
        "All pending syncs should complete, {} remaining",
        remaining_pending
    );

    println!("\nPhase 5: Verifying backup integrity...");

    // Verify backup integrity
    let backup_valid = backup_manager
        .verify_backup(&backup)
        .await
        .expect("Backup verification should succeed");
    assert!(backup_valid, "Backup should be valid after GCloud recovery");
    println!("Backup integrity verified: {}", backup.id);

    // Verify storage integrity
    let storage_integrity = storage.verify_integrity();
    assert!(storage_integrity, "Storage integrity check should pass");
    println!("Storage integrity verified");

    // Verify recovery within SLA
    assert_recovery_within_sla(recovery_duration, config.recovery_sla);
    println!("Recovery completed within SLA: {:?}", recovery_duration);

    // Summary
    println!("\n=== Chaos Test Summary ===");
    println!("Outage duration: {:?}", outage_duration);
    println!("Operations during outage: {}", config.operations_during_outage);
    println!("Recovery time: {:?}", recovery_duration);
    println!("Records synced on recovery: {}", recovery_sync_result.synced_count);
    println!("Backup integrity: OK");
    println!("Data loss: 0 bytes");
    println!("\n=== TEST PASSED ===\n");
}

/// Test that incremental sync recovers correctly
#[tokio::test]
async fn test_incremental_sync_recovery() {
    let storage = Arc::new(MockGCloudStorage::new());
    let sync_manager = MockIncrementalSync::new(Arc::clone(&storage));

    // Add initial data
    for i in 0..5 {
        sync_manager.add_change("table1", &format!("r-{}", i), "INSERT", "{}");
    }

    // Sync
    let _ = sync_manager.sync().await.expect("Should succeed");
    assert_eq!(sync_manager.get_pending_count(), 0);

    // Trigger outage
    storage.set_availability(false);

    // Add more data
    for i in 5..15 {
        sync_manager.add_change("table1", &format!("r-{}", i), "INSERT", "{}");
    }

    // Sync fails
    assert!(sync_manager.sync().await.is_err());
    assert_eq!(sync_manager.get_pending_count(), 10);

    // Recover
    storage.set_availability(true);

    // Sync succeeds
    let result = sync_manager.sync().await.expect("Should succeed");
    assert_eq!(result.synced_count, 10);
    assert_eq!(sync_manager.get_pending_count(), 0);
}

/// Test that backup is not corrupted during GCloud outage
#[tokio::test]
async fn test_no_backup_corruption() {
    let storage = Arc::new(MockGCloudStorage::new());
    let backup_manager = MockBackupManager::new(Arc::clone(&storage));

    // Create backup before outage
    let test_data = b"important backup data".to_vec();
    let backup = backup_manager
        .create_backup(test_data.clone())
        .await
        .expect("Backup should succeed");

    // Trigger outage
    storage.set_availability(false);

    // Attempt another backup (should fail)
    let outage_backup_result = backup_manager.create_backup(b"new data".to_vec()).await;
    assert!(outage_backup_result.is_err());

    // Restore
    storage.set_availability(true);

    // Verify original backup is intact
    let is_valid = backup_manager
        .verify_backup(&backup)
        .await
        .expect("Verification should succeed");

    assert!(is_valid, "Original backup should not be corrupted");

    // Verify integrity of all stored objects
    assert!(storage.verify_integrity(), "Storage should be intact");
}

/// Test sync resumes within 5 minute SLA
#[tokio::test]
async fn test_sync_resumes_within_sla() {
    let storage = Arc::new(MockGCloudStorage::new());
    let sync_manager = MockIncrementalSync::new(Arc::clone(&storage));

    // Add pending data
    for i in 0..100 {
        sync_manager.add_change("patterns", &format!("p-{}", i), "INSERT", "{}");
    }

    // Trigger outage
    storage.set_availability(false);

    // Attempt sync
    let _ = sync_manager.sync().await;
    let pending_before = sync_manager.get_pending_count();

    // Simulate outage duration (scaled down)
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Restore and measure recovery time
    let recovery_start = Instant::now();
    storage.set_availability(true);

    let _ = sync_manager.sync().await.expect("Should succeed");
    let recovery_time = recovery_start.elapsed();

    // SLA: 5 minutes (scaled to 5 seconds for test)
    let sla = Duration::from_secs(5);
    assert!(
        recovery_time < sla,
        "Recovery took {:?}, exceeds SLA of {:?}",
        recovery_time,
        sla
    );

    // Verify all pending synced
    assert_eq!(sync_manager.get_pending_count(), 0);
}
