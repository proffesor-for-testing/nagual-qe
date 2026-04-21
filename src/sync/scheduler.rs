//! Sync scheduler with cron expressions.
//!
//! Provides automated scheduling for sync operations using tokio-cron-scheduler.
//! Supports scheduling for:
//! - Incremental backups (default: every 30 minutes)
//! - Full backups (default: every 6 hours)
//! - Retention cleanup (default: daily at 2 AM)
//! - Restore drills (default: first Sunday of each month)

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{debug, error, info, warn};

use crate::error::{NagualError, Result};

/// Type of scheduled task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduledTask {
    /// Incremental backup sync
    IncrementalBackup,
    /// Full database backup
    FullBackup,
    /// Retention policy cleanup
    RetentionCleanup,
    /// Monthly restore drill
    RestoreDrill,
}

impl std::fmt::Display for ScheduledTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScheduledTask::IncrementalBackup => write!(f, "incremental_backup"),
            ScheduledTask::FullBackup => write!(f, "full_backup"),
            ScheduledTask::RetentionCleanup => write!(f, "retention_cleanup"),
            ScheduledTask::RestoreDrill => write!(f, "restore_drill"),
        }
    }
}

/// Configuration for the sync scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncSchedulerConfig {
    /// Cron expression for incremental backups (default: every 30 min)
    pub incremental_cron: String,
    /// Cron expression for full backups (default: every 6 hours)
    pub full_backup_cron: String,
    /// Cron expression for retention cleanup (default: daily at 2 AM)
    pub retention_cron: String,
    /// Cron expression for restore drills (default: first Sunday of month at 3 AM)
    pub drill_cron: String,
    /// Whether to enable incremental backups
    pub enable_incremental: bool,
    /// Whether to enable full backups
    pub enable_full_backup: bool,
    /// Whether to enable retention cleanup
    pub enable_retention: bool,
    /// Whether to enable restore drills
    pub enable_drill: bool,
}

impl Default for SyncSchedulerConfig {
    fn default() -> Self {
        Self {
            // Every 30 minutes
            incremental_cron: "0 */30 * * * *".to_string(),
            // Every 6 hours at minute 5
            full_backup_cron: "0 5 */6 * * *".to_string(),
            // Daily at 2:00 AM
            retention_cron: "0 0 2 * * *".to_string(),
            // First Sunday of each month at 3:00 AM
            drill_cron: "0 0 3 1-7 * 0".to_string(),
            enable_incremental: true,
            enable_full_backup: true,
            enable_retention: true,
            enable_drill: true,
        }
    }
}

impl SyncSchedulerConfig {
    /// Create a new scheduler configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the incremental backup schedule.
    pub fn with_incremental_cron(mut self, cron: impl Into<String>) -> Self {
        self.incremental_cron = cron.into();
        self
    }

    /// Set the full backup schedule.
    pub fn with_full_backup_cron(mut self, cron: impl Into<String>) -> Self {
        self.full_backup_cron = cron.into();
        self
    }

    /// Set the retention cleanup schedule.
    pub fn with_retention_cron(mut self, cron: impl Into<String>) -> Self {
        self.retention_cron = cron.into();
        self
    }

    /// Set the drill schedule.
    pub fn with_drill_cron(mut self, cron: impl Into<String>) -> Self {
        self.drill_cron = cron.into();
        self
    }

    /// Enable or disable incremental backups.
    pub fn enable_incremental(mut self, enable: bool) -> Self {
        self.enable_incremental = enable;
        self
    }

    /// Enable or disable full backups.
    pub fn enable_full_backup(mut self, enable: bool) -> Self {
        self.enable_full_backup = enable;
        self
    }

    /// Enable or disable retention cleanup.
    pub fn enable_retention(mut self, enable: bool) -> Self {
        self.enable_retention = enable;
        self
    }

    /// Enable or disable restore drills.
    pub fn enable_drill(mut self, enable: bool) -> Self {
        self.enable_drill = enable;
        self
    }
}

/// Sync status information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatus {
    /// Last sync time for each task type
    pub last_sync_times: HashMap<ScheduledTask, DateTime<Utc>>,
    /// Number of pending records to sync
    pub pending_records: u64,
    /// Overall sync health (healthy, degraded, unhealthy)
    pub sync_health: SyncHealth,
    /// Number of consecutive failures
    pub consecutive_failures: u32,
    /// Last error message if any
    pub last_error: Option<String>,
}

impl Default for SyncStatus {
    fn default() -> Self {
        Self {
            last_sync_times: HashMap::new(),
            pending_records: 0,
            sync_health: SyncHealth::Healthy,
            consecutive_failures: 0,
            last_error: None,
        }
    }
}

/// Overall sync health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncHealth {
    /// All sync operations are working normally
    Healthy,
    /// Some issues but sync is still operational
    Degraded,
    /// Sync is not working properly
    Unhealthy,
}

impl std::fmt::Display for SyncHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SyncHealth::Healthy => write!(f, "healthy"),
            SyncHealth::Degraded => write!(f, "degraded"),
            SyncHealth::Unhealthy => write!(f, "unhealthy"),
        }
    }
}

/// Sync status report with history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatusReport {
    /// Current status
    pub current: SyncStatus,
    /// Recent sync history (last 10 entries)
    pub history: Vec<SyncHistoryEntry>,
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
}

/// Entry in the sync history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncHistoryEntry {
    /// Task type
    pub task: ScheduledTask,
    /// When it was executed
    pub executed_at: DateTime<Utc>,
    /// Whether it succeeded
    pub success: bool,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Number of records synced
    pub records_synced: u64,
    /// Error message if failed
    pub error: Option<String>,
}

/// Event emitted by the scheduler.
#[derive(Debug, Clone)]
pub enum SchedulerEvent {
    /// A task started
    TaskStarted(ScheduledTask),
    /// A task completed successfully
    TaskCompleted(ScheduledTask, u64), // duration_ms
    /// A task failed
    TaskFailed(ScheduledTask, String),
    /// Scheduler started
    Started,
    /// Scheduler stopped
    Stopped,
}

/// State of the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerState {
    /// Scheduler has not been started
    Stopped,
    /// Scheduler is running
    Running,
    /// Scheduler is paused
    Paused,
    /// Scheduler is shutting down
    ShuttingDown,
}

/// Sync scheduler for automated sync operations.
pub struct SyncScheduler {
    config: SyncSchedulerConfig,
    state: Arc<RwLock<SchedulerState>>,
    status: Arc<RwLock<SyncStatus>>,
    history: Arc<RwLock<Vec<SyncHistoryEntry>>>,
    event_tx: broadcast::Sender<SchedulerEvent>,
    shutdown_tx: Option<mpsc::Sender<()>>,
    scheduler_handle: Arc<RwLock<Option<JobScheduler>>>,
    started_at: Arc<RwLock<Option<DateTime<Utc>>>>,
}

impl SyncScheduler {
    /// Create a new sync scheduler.
    pub fn new(config: SyncSchedulerConfig) -> Result<Self> {
        let (event_tx, _) = broadcast::channel(100);

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(SchedulerState::Stopped)),
            status: Arc::new(RwLock::new(SyncStatus::default())),
            history: Arc::new(RwLock::new(Vec::with_capacity(100))),
            event_tx,
            shutdown_tx: None,
            scheduler_handle: Arc::new(RwLock::new(None)),
            started_at: Arc::new(RwLock::new(None)),
        })
    }

    /// Create a scheduler with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(SyncSchedulerConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &SyncSchedulerConfig {
        &self.config
    }

    /// Subscribe to scheduler events.
    pub fn subscribe(&self) -> broadcast::Receiver<SchedulerEvent> {
        self.event_tx.subscribe()
    }

    /// Get the current state.
    pub async fn state(&self) -> SchedulerState {
        *self.state.read().await
    }

    /// Get when the scheduler was started.
    pub async fn started_at(&self) -> Option<DateTime<Utc>> {
        *self.started_at.read().await
    }

    /// Get the current sync status.
    pub async fn status(&self) -> SyncStatus {
        self.status.read().await.clone()
    }

    /// Get a full status report.
    pub async fn status_report(&self) -> SyncStatusReport {
        let current = self.status.read().await.clone();
        let history = self.history.read().await.clone();

        // Get last 10 entries
        let history = history.into_iter().rev().take(10).collect();

        SyncStatusReport {
            current,
            history,
            generated_at: Utc::now(),
        }
    }

    /// Start the scheduler.
    pub async fn start(&mut self) -> Result<()> {
        let current_state = *self.state.read().await;
        if current_state == SchedulerState::Running {
            return Ok(());
        }

        info!("Starting sync scheduler");

        // Create shutdown channel
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        self.shutdown_tx = Some(shutdown_tx);

        // Create the job scheduler
        let scheduler = JobScheduler::new()
            .await
            .map_err(|e| NagualError::internal(format!("Failed to create scheduler: {}", e)))?;

        // Add incremental backup job
        if self.config.enable_incremental {
            let job = self.create_task_job(
                ScheduledTask::IncrementalBackup,
                &self.config.incremental_cron,
            )?;
            scheduler.add(job).await.map_err(|e| {
                NagualError::internal(format!("Failed to add incremental job: {}", e))
            })?;
            debug!("Added incremental backup job");
        }

        // Add full backup job
        if self.config.enable_full_backup {
            let job = self.create_task_job(
                ScheduledTask::FullBackup,
                &self.config.full_backup_cron,
            )?;
            scheduler.add(job).await.map_err(|e| {
                NagualError::internal(format!("Failed to add full backup job: {}", e))
            })?;
            debug!("Added full backup job");
        }

        // Add retention cleanup job
        if self.config.enable_retention {
            let job = self.create_task_job(
                ScheduledTask::RetentionCleanup,
                &self.config.retention_cron,
            )?;
            scheduler.add(job).await.map_err(|e| {
                NagualError::internal(format!("Failed to add retention job: {}", e))
            })?;
            debug!("Added retention cleanup job");
        }

        // Add drill job
        if self.config.enable_drill {
            let job = self.create_task_job(
                ScheduledTask::RestoreDrill,
                &self.config.drill_cron,
            )?;
            scheduler.add(job).await.map_err(|e| {
                NagualError::internal(format!("Failed to add drill job: {}", e))
            })?;
            debug!("Added restore drill job");
        }

        // Start the scheduler
        scheduler.start().await.map_err(|e| {
            NagualError::internal(format!("Failed to start scheduler: {}", e))
        })?;

        // Store the scheduler handle
        {
            let mut handle = self.scheduler_handle.write().await;
            *handle = Some(scheduler);
        }

        // Update state
        {
            let mut state = self.state.write().await;
            *state = SchedulerState::Running;
        }

        {
            let mut started = self.started_at.write().await;
            *started = Some(Utc::now());
        }

        // Emit started event
        let _ = self.event_tx.send(SchedulerEvent::Started);

        info!("Sync scheduler started");

        // Spawn shutdown listener
        let state = self.state.clone();
        let scheduler_handle = self.scheduler_handle.clone();
        let event_tx = self.event_tx.clone();

        tokio::spawn(async move {
            shutdown_rx.recv().await;
            debug!("Received scheduler shutdown signal");

            {
                let mut state_guard = state.write().await;
                *state_guard = SchedulerState::ShuttingDown;
            }

            if let Some(mut scheduler) = scheduler_handle.write().await.take() {
                if let Err(e) = scheduler.shutdown().await {
                    error!("Error shutting down scheduler: {}", e);
                }
            }

            {
                let mut state_guard = state.write().await;
                *state_guard = SchedulerState::Stopped;
            }

            let _ = event_tx.send(SchedulerEvent::Stopped);
        });

        Ok(())
    }

    /// Stop the scheduler.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping sync scheduler");

        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }

        // Wait for shutdown to complete
        loop {
            let state = *self.state.read().await;
            if state == SchedulerState::Stopped {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        info!("Sync scheduler stopped");
        Ok(())
    }

    /// Record a task execution.
    pub async fn record_execution(
        &self,
        task: ScheduledTask,
        success: bool,
        duration_ms: u64,
        records_synced: u64,
        error: Option<String>,
    ) {
        let entry = SyncHistoryEntry {
            task,
            executed_at: Utc::now(),
            success,
            duration_ms,
            records_synced,
            error: error.clone(),
        };

        // Add to history
        {
            let mut history = self.history.write().await;
            history.push(entry);
            // Keep only last 100 entries
            if history.len() > 100 {
                history.remove(0);
            }
        }

        // Update status
        {
            let mut status = self.status.write().await;
            status.last_sync_times.insert(task, Utc::now());

            if success {
                status.consecutive_failures = 0;
                status.last_error = None;
                status.sync_health = SyncHealth::Healthy;
            } else {
                status.consecutive_failures += 1;
                status.last_error = error;
                status.sync_health = if status.consecutive_failures >= 3 {
                    SyncHealth::Unhealthy
                } else {
                    SyncHealth::Degraded
                };
            }
        }

        // Emit event
        if success {
            let _ = self.event_tx.send(SchedulerEvent::TaskCompleted(task, duration_ms));
        } else {
            let _ = self.event_tx.send(SchedulerEvent::TaskFailed(
                task,
                self.status.read().await.last_error.clone().unwrap_or_default(),
            ));
        }
    }

    /// Create a job for a specific task.
    fn create_task_job(&self, task: ScheduledTask, cron: &str) -> Result<Job> {
        let event_tx = self.event_tx.clone();
        let state = self.state.clone();
        let status = self.status.clone();
        let history = self.history.clone();

        Job::new_async(cron, move |_uuid, _lock| {
            let event_tx = event_tx.clone();
            let state = state.clone();
            let status = status.clone();
            let history = history.clone();
            let task = task;

            Box::pin(async move {
                let current_state = *state.read().await;
                if current_state != SchedulerState::Running {
                    return;
                }

                let start = std::time::Instant::now();
                let _ = event_tx.send(SchedulerEvent::TaskStarted(task));

                info!(task = %task, "Executing scheduled task");

                // Simulate task execution
                // In a real implementation, this would call the actual backup/restore logic
                let (success, records, error) = match task {
                    ScheduledTask::IncrementalBackup => {
                        // Would call backup_manager.create_incremental_backup()
                        (true, 100, None)
                    }
                    ScheduledTask::FullBackup => {
                        // Would call backup_manager.create_full_backup()
                        (true, 1000, None)
                    }
                    ScheduledTask::RetentionCleanup => {
                        // Would call backup_manager.apply_retention_policy()
                        (true, 0, None)
                    }
                    ScheduledTask::RestoreDrill => {
                        // Would call drill.run_drill()
                        (true, 0, None)
                    }
                };

                let duration_ms = start.elapsed().as_millis() as u64;

                // Record execution
                let entry = SyncHistoryEntry {
                    task,
                    executed_at: Utc::now(),
                    success,
                    duration_ms,
                    records_synced: records,
                    error: error.clone(),
                };

                {
                    let mut hist = history.write().await;
                    hist.push(entry);
                    if hist.len() > 100 {
                        hist.remove(0);
                    }
                }

                {
                    let mut stat = status.write().await;
                    stat.last_sync_times.insert(task, Utc::now());
                    if success {
                        stat.consecutive_failures = 0;
                        stat.sync_health = SyncHealth::Healthy;
                    } else {
                        stat.consecutive_failures += 1;
                        stat.last_error = error.clone();
                        stat.sync_health = if stat.consecutive_failures >= 3 {
                            SyncHealth::Unhealthy
                        } else {
                            SyncHealth::Degraded
                        };
                    }
                }

                if success {
                    info!(task = %task, duration_ms, "Task completed successfully");
                    let _ = event_tx.send(SchedulerEvent::TaskCompleted(task, duration_ms));
                } else {
                    warn!(task = %task, error = ?error, "Task failed");
                    let _ = event_tx.send(SchedulerEvent::TaskFailed(task, error.unwrap_or_default()));
                }
            })
        })
        .map_err(|e| NagualError::internal(format!("Failed to create job: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_config_default() {
        let config = SyncSchedulerConfig::default();
        assert!(config.enable_incremental);
        assert!(config.enable_full_backup);
        assert!(config.enable_retention);
        assert!(config.enable_drill);
    }

    #[test]
    fn test_scheduler_config_custom() {
        let config = SyncSchedulerConfig::new()
            .with_incremental_cron("0 */15 * * * *")
            .enable_drill(false);

        assert_eq!(config.incremental_cron, "0 */15 * * * *");
        assert!(!config.enable_drill);
    }

    #[test]
    fn test_sync_status_default() {
        let status = SyncStatus::default();
        assert!(status.last_sync_times.is_empty());
        assert_eq!(status.sync_health, SyncHealth::Healthy);
        assert_eq!(status.consecutive_failures, 0);
    }

    #[tokio::test]
    async fn test_scheduler_creation() {
        let scheduler = SyncScheduler::new(SyncSchedulerConfig::default()).unwrap();
        assert_eq!(scheduler.state().await, SchedulerState::Stopped);
    }

    #[tokio::test]
    async fn test_status_report() {
        let scheduler = SyncScheduler::new(SyncSchedulerConfig::default()).unwrap();
        let report = scheduler.status_report().await;

        assert!(report.history.is_empty());
        assert_eq!(report.current.sync_health, SyncHealth::Healthy);
    }

    #[tokio::test]
    async fn test_record_execution() {
        let scheduler = SyncScheduler::new(SyncSchedulerConfig::default()).unwrap();

        scheduler.record_execution(
            ScheduledTask::IncrementalBackup,
            true,
            100,
            50,
            None,
        ).await;

        let status = scheduler.status().await;
        assert!(status.last_sync_times.contains_key(&ScheduledTask::IncrementalBackup));
        assert_eq!(status.consecutive_failures, 0);
    }

    #[tokio::test]
    async fn test_record_execution_failure() {
        let scheduler = SyncScheduler::new(SyncSchedulerConfig::default()).unwrap();

        scheduler.record_execution(
            ScheduledTask::FullBackup,
            false,
            100,
            0,
            Some("Connection failed".to_string()),
        ).await;

        let status = scheduler.status().await;
        assert_eq!(status.consecutive_failures, 1);
        assert_eq!(status.sync_health, SyncHealth::Degraded);
    }

    #[tokio::test]
    async fn test_multiple_failures_unhealthy() {
        let scheduler = SyncScheduler::new(SyncSchedulerConfig::default()).unwrap();

        for _ in 0..3 {
            scheduler.record_execution(
                ScheduledTask::FullBackup,
                false,
                100,
                0,
                Some("Failed".to_string()),
            ).await;
        }

        let status = scheduler.status().await;
        assert_eq!(status.consecutive_failures, 3);
        assert_eq!(status.sync_health, SyncHealth::Unhealthy);
    }

    #[test]
    fn test_scheduled_task_display() {
        assert_eq!(ScheduledTask::IncrementalBackup.to_string(), "incremental_backup");
        assert_eq!(ScheduledTask::FullBackup.to_string(), "full_backup");
        assert_eq!(ScheduledTask::RetentionCleanup.to_string(), "retention_cleanup");
        assert_eq!(ScheduledTask::RestoreDrill.to_string(), "restore_drill");
    }
}
