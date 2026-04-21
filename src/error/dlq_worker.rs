//! Background DLQ worker for processing failed operations.
//!
//! Provides a background worker that:
//! - Continuously processes DLQ entries
//! - Uses increasing retry intervals (1min, 5min, 15min, 1hr, 6hr)
//! - Abandons entries after 10 attempts
//! - Supports graceful shutdown
//! - Handles stuck entries from crashed workers
//!
//! Note: Since rusqlite::Connection is not Send+Sync, this worker runs
//! the DLQ operations in a dedicated thread using spawn_blocking.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use super::dlq::{BatchProcessResult, DeadLetterQueue, DlqEntry, DlqStats};
use super::DlqError;

/// Configuration for the DLQ worker.
#[derive(Debug, Clone)]
pub struct DlqWorkerConfig {
    /// How often to check for ready entries.
    pub poll_interval: Duration,
    /// Maximum entries to process per batch.
    pub batch_size: usize,
    /// Maximum number of retry attempts before abandoning.
    pub max_attempts: u32,
    /// Threshold for considering an entry "stuck" (processing for too long).
    pub stuck_threshold: Duration,
    /// How often to check for and requeue stuck entries.
    pub stuck_check_interval: Duration,
    /// How often to clean up abandoned entries.
    pub cleanup_interval: Duration,
    /// Age threshold for cleaning up abandoned entries.
    pub cleanup_age: Duration,
}

impl Default for DlqWorkerConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(10),
            batch_size: 10,
            max_attempts: 10,
            stuck_threshold: Duration::from_secs(300), // 5 minutes
            stuck_check_interval: Duration::from_secs(60),
            cleanup_interval: Duration::from_secs(3600), // 1 hour
            cleanup_age: Duration::from_secs(86400 * 7), // 7 days
        }
    }
}

/// Message type for controlling the worker.
#[derive(Debug)]
enum WorkerCommand {
    /// Shutdown the worker gracefully.
    Shutdown,
    /// Process entries immediately (don't wait for poll interval).
    ProcessNow,
    /// Get current statistics.
    GetStats(oneshot::Sender<DlqStats>),
}

/// Handle for controlling a running DLQ worker.
#[derive(Clone)]
pub struct DlqWorkerHandle {
    command_tx: mpsc::Sender<WorkerCommand>,
    is_running: Arc<AtomicBool>,
}

impl DlqWorkerHandle {
    /// Request graceful shutdown of the worker.
    pub async fn shutdown(&self) -> Result<(), DlqError> {
        if self.is_running.load(Ordering::Relaxed) {
            self.command_tx
                .send(WorkerCommand::Shutdown)
                .await
                .map_err(|e| DlqError::EnqueueFailed(format!("Failed to send shutdown: {}", e)))?;
        }
        Ok(())
    }

    /// Request immediate processing of ready entries.
    pub async fn process_now(&self) -> Result<(), DlqError> {
        self.command_tx
            .send(WorkerCommand::ProcessNow)
            .await
            .map_err(|e| DlqError::EnqueueFailed(format!("Failed to send process_now: {}", e)))
    }

    /// Get current DLQ statistics.
    pub async fn stats(&self) -> Result<DlqStats, DlqError> {
        let (tx, rx) = oneshot::channel();
        self.command_tx
            .send(WorkerCommand::GetStats(tx))
            .await
            .map_err(|e| DlqError::EnqueueFailed(format!("Failed to send stats request: {}", e)))?;
        rx.await
            .map_err(|e| DlqError::DequeueFailed(format!("Failed to receive stats: {}", e)))
    }

    /// Check if the worker is still running.
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }
}

/// Handler function type for processing DLQ entries.
pub type HandlerFn = Arc<dyn Fn(&DlqEntry) -> Result<(), String> + Send + Sync + 'static>;

/// Router for handling different operation types.
pub struct OperationRouter {
    handlers: hashbrown::HashMap<String, Arc<dyn Fn(&DlqEntry) -> Result<(), String> + Send + Sync>>,
    default_handler: Option<Arc<dyn Fn(&DlqEntry) -> Result<(), String> + Send + Sync>>,
}

impl Default for OperationRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl OperationRouter {
    pub fn new() -> Self {
        Self {
            handlers: hashbrown::HashMap::new(),
            default_handler: None,
        }
    }

    /// Register a handler for a specific operation type.
    pub fn register<F>(mut self, operation: impl Into<String>, handler: F) -> Self
    where
        F: Fn(&DlqEntry) -> Result<(), String> + Send + Sync + 'static,
    {
        self.handlers.insert(operation.into(), Arc::new(handler));
        self
    }

    /// Set a default handler for unregistered operations.
    pub fn with_default<F>(mut self, handler: F) -> Self
    where
        F: Fn(&DlqEntry) -> Result<(), String> + Send + Sync + 'static,
    {
        self.default_handler = Some(Arc::new(handler));
        self
    }

    /// Handle an entry by routing to the appropriate handler.
    pub fn handle(&self, entry: &DlqEntry) -> Result<(), String> {
        if let Some(handler) = self.handlers.get(&entry.operation) {
            handler(entry)
        } else if let Some(default) = &self.default_handler {
            default(entry)
        } else {
            Err(format!("No handler registered for operation: {}", entry.operation))
        }
    }

    /// Convert to a HandlerFn.
    pub fn into_handler(self) -> HandlerFn {
        Arc::new(move |entry: &DlqEntry| self.handle(entry))
    }
}

/// Storage configuration for the DLQ.
#[derive(Clone)]
pub enum DlqStorage {
    /// In-memory storage (for testing).
    InMemory,
    /// File-backed SQLite storage.
    File(PathBuf),
}

/// Background DLQ worker.
pub struct DlqWorker {
    config: DlqWorkerConfig,
    storage: DlqStorage,
    is_running: Arc<AtomicBool>,
}

impl DlqWorker {
    /// Create a new DLQ worker with an in-memory queue.
    pub fn in_memory(config: DlqWorkerConfig) -> Self {
        Self {
            config,
            storage: DlqStorage::InMemory,
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Create a new DLQ worker with a file-backed queue.
    pub fn with_file(path: impl AsRef<Path>, config: DlqWorkerConfig) -> Self {
        Self {
            config,
            storage: DlqStorage::File(path.as_ref().to_path_buf()),
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the worker in the background.
    ///
    /// Returns a handle that can be used to control the worker.
    pub fn spawn(self, handler: HandlerFn) -> DlqWorkerHandle {
        let (command_tx, command_rx) = mpsc::channel::<WorkerCommand>(32);
        let is_running = self.is_running.clone();
        is_running.store(true, Ordering::Relaxed);

        let handle = DlqWorkerHandle {
            command_tx,
            is_running: is_running.clone(),
        };

        tokio::spawn(async move {
            self.run(command_rx, handler).await;
        });

        handle
    }

    /// Start the worker with an operation router.
    pub fn spawn_with_router(self, router: OperationRouter) -> DlqWorkerHandle {
        self.spawn(router.into_handler())
    }

    /// Run the worker loop.
    async fn run(self, mut command_rx: mpsc::Receiver<WorkerCommand>, handler: HandlerFn) {
        info!(
            poll_interval_ms = self.config.poll_interval.as_millis(),
            batch_size = self.config.batch_size,
            max_attempts = self.config.max_attempts,
            "DLQ worker started"
        );

        let mut poll_interval = interval(self.config.poll_interval);
        let mut stuck_check_interval = interval(self.config.stuck_check_interval);
        let mut cleanup_interval = interval(self.config.cleanup_interval);
        let mut total_processed: u64 = 0;
        let mut total_succeeded: u64 = 0;
        let mut total_failed: u64 = 0;
        let mut total_abandoned: u64 = 0;

        // Create the DLQ in a blocking task
        let storage = self.storage.clone();
        let config = self.config.clone();

        loop {
            tokio::select! {
                // Handle commands
                Some(cmd) = command_rx.recv() => {
                    match cmd {
                        WorkerCommand::Shutdown => {
                            info!(
                                total_processed = total_processed,
                                total_succeeded = total_succeeded,
                                total_failed = total_failed,
                                total_abandoned = total_abandoned,
                                "DLQ worker shutting down"
                            );
                            self.is_running.store(false, Ordering::Relaxed);
                            break;
                        }
                        WorkerCommand::ProcessNow => {
                            debug!("Processing DLQ entries on demand");
                            let result = Self::process_batch_blocking(
                                storage.clone(),
                                config.batch_size,
                                handler.clone(),
                            ).await;
                            match result {
                                Ok(r) => {
                                    total_processed += r.total() as u64;
                                    total_succeeded += r.succeeded as u64;
                                    total_failed += r.failed as u64;
                                    total_abandoned += r.abandoned as u64;
                                }
                                Err(e) => error!(error = %e, "Failed to process batch"),
                            }
                        }
                        WorkerCommand::GetStats(tx) => {
                            let storage = storage.clone();
                            let stats = tokio::task::spawn_blocking(move || {
                                let dlq = match &storage {
                                    DlqStorage::InMemory => DeadLetterQueue::in_memory(),
                                    DlqStorage::File(path) => DeadLetterQueue::new(path),
                                };
                                dlq.and_then(|d| d.stats())
                            }).await;

                            if let Ok(Ok(s)) = stats {
                                let _ = tx.send(s);
                            }
                        }
                    }
                }

                // Regular polling
                _ = poll_interval.tick() => {
                    let result = Self::process_batch_blocking(
                        storage.clone(),
                        config.batch_size,
                        handler.clone(),
                    ).await;
                    match result {
                        Ok(r) if r.total() > 0 => {
                            total_processed += r.total() as u64;
                            total_succeeded += r.succeeded as u64;
                            total_failed += r.failed as u64;
                            total_abandoned += r.abandoned as u64;
                        }
                        Err(e) => error!(error = %e, "Failed to process batch"),
                        _ => {}
                    }
                }

                // Check for stuck entries
                _ = stuck_check_interval.tick() => {
                    let storage = storage.clone();
                    let threshold = config.stuck_threshold;
                    let _ = tokio::task::spawn_blocking(move || {
                        let dlq = match &storage {
                            DlqStorage::InMemory => DeadLetterQueue::in_memory(),
                            DlqStorage::File(path) => DeadLetterQueue::new(path),
                        };
                        if let Ok(dlq) = dlq {
                            if let Ok(requeued) = dlq.requeue_stuck(threshold) {
                                if requeued > 0 {
                                    warn!(requeued = requeued, "Requeued stuck DLQ entries");
                                }
                            }
                        }
                    }).await;
                }

                // Cleanup abandoned entries
                _ = cleanup_interval.tick() => {
                    let storage = storage.clone();
                    let age = config.cleanup_age;
                    let _ = tokio::task::spawn_blocking(move || {
                        let dlq = match &storage {
                            DlqStorage::InMemory => DeadLetterQueue::in_memory(),
                            DlqStorage::File(path) => DeadLetterQueue::new(path),
                        };
                        if let Ok(dlq) = dlq {
                            if let Ok(cleaned) = dlq.cleanup_abandoned(age) {
                                if cleaned > 0 {
                                    info!(cleaned = cleaned, "Cleaned up abandoned DLQ entries");
                                }
                            }
                        }
                    }).await;
                }
            }
        }
    }

    /// Process a batch of entries in a blocking task.
    async fn process_batch_blocking(
        storage: DlqStorage,
        batch_size: usize,
        handler: HandlerFn,
    ) -> Result<BatchProcessResult, DlqError> {
        tokio::task::spawn_blocking(move || {
            let dlq = match &storage {
                DlqStorage::InMemory => DeadLetterQueue::in_memory()?,
                DlqStorage::File(path) => DeadLetterQueue::new(path)?,
            };

            // Use process_batch which handles the processing status internally
            let result = dlq.process_batch(batch_size, |entry| handler(entry))?;

            if result.total() > 0 {
                debug!(
                    succeeded = result.succeeded,
                    failed = result.failed,
                    abandoned = result.abandoned,
                    "DLQ batch processing complete"
                );
            }

            Ok(result)
        })
        .await
        .map_err(|e| DlqError::EnqueueFailed(format!("Spawn blocking failed: {}", e)))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    fn create_test_config() -> DlqWorkerConfig {
        DlqWorkerConfig {
            poll_interval: Duration::from_millis(10),
            batch_size: 10,
            stuck_threshold: Duration::from_secs(5),
            stuck_check_interval: Duration::from_secs(60),
            cleanup_interval: Duration::from_secs(3600),
            cleanup_age: Duration::from_secs(86400),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_worker_creation() {
        let config = create_test_config();
        let worker = DlqWorker::in_memory(config);
        assert!(!worker.is_running.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_worker_spawn_and_shutdown() {
        let config = create_test_config();
        let worker = DlqWorker::in_memory(config);

        let handler: HandlerFn = Arc::new(|_| Ok(()));
        let handle = worker.spawn(handler);

        assert!(handle.is_running());

        // Give it time to start
        sleep(Duration::from_millis(50)).await;

        handle.shutdown().await.unwrap();

        // Give it time to shutdown
        sleep(Duration::from_millis(50)).await;

        assert!(!handle.is_running());
    }

    #[test]
    fn test_operation_router() {
        let router = OperationRouter::new()
            .register("op_a", |_| Ok(()))
            .register("op_b", |_| Err("op_b error".to_string()))
            .with_default(|entry| Err(format!("unknown: {}", entry.operation)));

        let entry_a = DlqEntry::new("op_a", "{}", "");
        assert!(router.handle(&entry_a).is_ok());

        let entry_b = DlqEntry::new("op_b", "{}", "");
        assert!(router.handle(&entry_b).is_err());

        let entry_c = DlqEntry::new("op_c", "{}", "");
        let result = router.handle(&entry_c);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown"));
    }
}
