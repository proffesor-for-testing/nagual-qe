//! Health Check Scheduler
//!
//! Provides periodic health check scheduling using tokio-cron-scheduler.
//! Supports configurable intervals, event emission on status changes,
//! and graceful start/stop mechanisms.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{debug, error, info, warn};

use super::{HealthChangeEvent, HealthRegistry, HealthReport, HealthStatus};
use crate::error::{NagualError, Result};

/// Configuration for the health check scheduler
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Interval between health checks (default: 30 seconds)
    pub check_interval: Duration,
    /// Whether to run an initial check on start
    pub check_on_start: bool,
    /// Channel capacity for health events
    pub event_channel_capacity: usize,
    /// Whether to emit events only on status changes
    pub emit_only_on_change: bool,
    /// Cron expression for custom scheduling (overrides check_interval)
    pub cron_expression: Option<String>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            check_on_start: true,
            event_channel_capacity: 100,
            emit_only_on_change: true,
            cron_expression: None,
        }
    }
}

impl SchedulerConfig {
    /// Create a new configuration with the given interval
    pub fn with_interval(interval: Duration) -> Self {
        Self {
            check_interval: interval,
            ..Default::default()
        }
    }

    /// Create a configuration using a cron expression
    pub fn with_cron(expression: impl Into<String>) -> Self {
        Self {
            cron_expression: Some(expression.into()),
            ..Default::default()
        }
    }

    /// Set whether to check on start
    pub fn check_on_start(mut self, check: bool) -> Self {
        self.check_on_start = check;
        self
    }

    /// Set whether to emit events only on change
    pub fn emit_only_on_change(mut self, only_on_change: bool) -> Self {
        self.emit_only_on_change = only_on_change;
        self
    }
}

/// State of the scheduler
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

/// Event emitted by the scheduler
#[derive(Debug, Clone)]
pub enum SchedulerEvent {
    /// A health check completed
    CheckCompleted(HealthReport),
    /// Health status changed for a component
    StatusChanged(HealthChangeEvent),
    /// Scheduler started
    Started,
    /// Scheduler stopped
    Stopped,
    /// Error occurred during check
    Error(String),
}

/// Health check scheduler
///
/// Runs periodic health checks and emits events on status changes.
pub struct HealthScheduler {
    registry: Arc<HealthRegistry>,
    config: SchedulerConfig,
    state: Arc<RwLock<SchedulerState>>,
    last_statuses: Arc<RwLock<std::collections::HashMap<String, HealthStatus>>>,
    event_tx: broadcast::Sender<SchedulerEvent>,
    shutdown_tx: Option<mpsc::Sender<()>>,
    scheduler_handle: Arc<RwLock<Option<JobScheduler>>>,
    started_at: Arc<RwLock<Option<DateTime<Utc>>>>,
}

impl HealthScheduler {
    /// Create a new health scheduler
    pub fn new(registry: Arc<HealthRegistry>, config: SchedulerConfig) -> Self {
        let (event_tx, _) = broadcast::channel(config.event_channel_capacity);

        Self {
            registry,
            config,
            state: Arc::new(RwLock::new(SchedulerState::Stopped)),
            last_statuses: Arc::new(RwLock::new(std::collections::HashMap::new())),
            event_tx,
            shutdown_tx: None,
            scheduler_handle: Arc::new(RwLock::new(None)),
            started_at: Arc::new(RwLock::new(None)),
        }
    }

    /// Create a scheduler with default configuration
    pub fn with_defaults(registry: Arc<HealthRegistry>) -> Self {
        Self::new(registry, SchedulerConfig::default())
    }

    /// Subscribe to scheduler events
    pub fn subscribe(&self) -> broadcast::Receiver<SchedulerEvent> {
        self.event_tx.subscribe()
    }

    /// Get the current state
    pub async fn state(&self) -> SchedulerState {
        *self.state.read().await
    }

    /// Get when the scheduler was started
    pub async fn started_at(&self) -> Option<DateTime<Utc>> {
        *self.started_at.read().await
    }

    /// Start the scheduler
    pub async fn start(&mut self) -> Result<()> {
        let current_state = *self.state.read().await;
        if current_state == SchedulerState::Running {
            return Ok(());
        }

        info!("Starting health check scheduler");

        // Create shutdown channel
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        self.shutdown_tx = Some(shutdown_tx);

        // Run initial check if configured
        if self.config.check_on_start {
            self.run_check().await;
        }

        // Create the job scheduler
        let scheduler = JobScheduler::new()
            .await
            .map_err(|e| NagualError::Internal {
                message: format!("Failed to create job scheduler: {}", e),
            })?;

        // Clone data for the job closure
        let registry = self.registry.clone();
        let event_tx = self.event_tx.clone();
        let last_statuses = self.last_statuses.clone();
        let emit_only_on_change = self.config.emit_only_on_change;
        let state = self.state.clone();

        // Create the job
        let job = if let Some(ref cron_expr) = self.config.cron_expression {
            Job::new_async(cron_expr.as_str(), move |_uuid, _lock| {
                let registry = registry.clone();
                let event_tx = event_tx.clone();
                let last_statuses = last_statuses.clone();
                let state = state.clone();

                Box::pin(async move {
                    let current_state = *state.read().await;
                    if current_state != SchedulerState::Running {
                        return;
                    }

                    run_health_check(
                        &registry,
                        &event_tx,
                        &last_statuses,
                        emit_only_on_change,
                    )
                    .await;
                })
            })
        } else {
            let interval_secs = self.config.check_interval.as_secs();
            Job::new_repeated_async(
                std::time::Duration::from_secs(interval_secs),
                move |_uuid, _lock| {
                    let registry = registry.clone();
                    let event_tx = event_tx.clone();
                    let last_statuses = last_statuses.clone();
                    let state = state.clone();

                    Box::pin(async move {
                        let current_state = *state.read().await;
                        if current_state != SchedulerState::Running {
                            return;
                        }

                        run_health_check(
                            &registry,
                            &event_tx,
                            &last_statuses,
                            emit_only_on_change,
                        )
                        .await;
                    })
                },
            )
        }
        .map_err(|e| NagualError::Internal {
            message: format!("Failed to create health check job: {}", e),
        })?;

        scheduler.add(job).await.map_err(|e| NagualError::Internal {
            message: format!("Failed to add health check job: {}", e),
        })?;

        // Start the scheduler
        scheduler.start().await.map_err(|e| NagualError::Internal {
            message: format!("Failed to start scheduler: {}", e),
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

        info!("Health check scheduler started");

        // Spawn shutdown listener
        let state = self.state.clone();
        let scheduler_handle = self.scheduler_handle.clone();
        let event_tx = self.event_tx.clone();

        tokio::spawn(async move {
            shutdown_rx.recv().await;
            debug!("Received shutdown signal");

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

    /// Stop the scheduler
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping health check scheduler");

        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }

        // Wait for shutdown to complete
        loop {
            let state = *self.state.read().await;
            if state == SchedulerState::Stopped {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        info!("Health check scheduler stopped");
        Ok(())
    }

    /// Pause the scheduler (stops running checks but keeps scheduler alive)
    pub async fn pause(&self) {
        let mut state = self.state.write().await;
        if *state == SchedulerState::Running {
            *state = SchedulerState::Paused;
            info!("Health check scheduler paused");
        }
    }

    /// Resume a paused scheduler
    pub async fn resume(&self) {
        let mut state = self.state.write().await;
        if *state == SchedulerState::Paused {
            *state = SchedulerState::Running;
            info!("Health check scheduler resumed");
        }
    }

    /// Run a single health check immediately
    pub async fn run_check(&self) -> HealthReport {
        run_health_check(
            &self.registry,
            &self.event_tx,
            &self.last_statuses,
            self.config.emit_only_on_change,
        )
        .await
    }

    /// Get the last known statuses
    pub async fn last_statuses(&self) -> std::collections::HashMap<String, HealthStatus> {
        self.last_statuses.read().await.clone()
    }

    /// Check if a specific component is healthy
    pub async fn is_component_healthy(&self, component: &str) -> Option<bool> {
        let statuses = self.last_statuses.read().await;
        statuses
            .get(component)
            .map(|s| *s == HealthStatus::Healthy)
    }

    /// Get the overall system status
    pub async fn overall_status(&self) -> HealthStatus {
        let statuses = self.last_statuses.read().await;
        statuses
            .values()
            .fold(HealthStatus::Healthy, |acc, s| acc.combine(*s))
    }
}

/// Run a health check and emit events
async fn run_health_check(
    registry: &Arc<HealthRegistry>,
    event_tx: &broadcast::Sender<SchedulerEvent>,
    last_statuses: &Arc<RwLock<std::collections::HashMap<String, HealthStatus>>>,
    emit_only_on_change: bool,
) -> HealthReport {
    debug!("Running scheduled health check");

    let report = registry.check_all().await;

    // Check for status changes and emit events
    let mut statuses = last_statuses.write().await;

    for (component, result) in &report.components {
        let previous_status = statuses.get(component).copied();
        let new_status = result.status;

        if let Some(prev) = previous_status {
            if prev != new_status {
                let event = HealthChangeEvent::new(
                    component.clone(),
                    prev,
                    new_status,
                    result.message.clone(),
                );

                if event.is_degradation() {
                    warn!(
                        component = %component,
                        previous = %prev,
                        new = %new_status,
                        "Health status degraded"
                    );
                } else {
                    info!(
                        component = %component,
                        previous = %prev,
                        new = %new_status,
                        "Health status improved"
                    );
                }

                let _ = event_tx.send(SchedulerEvent::StatusChanged(event));
            }
        }

        statuses.insert(component.clone(), new_status);
    }

    // Emit check completed event
    if !emit_only_on_change || report.unhealthy_count > 0 || report.degraded_count > 0 {
        let _ = event_tx.send(SchedulerEvent::CheckCompleted(report.clone()));
    }

    report
}

/// Builder for creating a health scheduler with custom settings
pub struct HealthSchedulerBuilder {
    registry: Option<Arc<HealthRegistry>>,
    config: SchedulerConfig,
}

impl HealthSchedulerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            registry: None,
            config: SchedulerConfig::default(),
        }
    }

    /// Set the health registry
    pub fn registry(mut self, registry: Arc<HealthRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Set the check interval
    pub fn interval(mut self, interval: Duration) -> Self {
        self.config.check_interval = interval;
        self
    }

    /// Set whether to check on start
    pub fn check_on_start(mut self, check: bool) -> Self {
        self.config.check_on_start = check;
        self
    }

    /// Set whether to emit only on change
    pub fn emit_only_on_change(mut self, only_on_change: bool) -> Self {
        self.config.emit_only_on_change = only_on_change;
        self
    }

    /// Set a cron expression for scheduling
    pub fn cron(mut self, expression: impl Into<String>) -> Self {
        self.config.cron_expression = Some(expression.into());
        self
    }

    /// Build the scheduler
    pub fn build(self) -> Result<HealthScheduler> {
        let registry = self.registry.ok_or_else(|| NagualError::Internal {
            message: "Health registry is required".to_string(),
        })?;

        Ok(HealthScheduler::new(registry, self.config))
    }
}

impl Default for HealthSchedulerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::{HealthCheck, HealthCheckResult};

    struct MockHealthCheck {
        name: String,
        status: HealthStatus,
    }

    #[async_trait::async_trait]
    impl HealthCheck for MockHealthCheck {
        fn name(&self) -> &str {
            &self.name
        }

        async fn check(&self) -> HealthCheckResult {
            match self.status {
                HealthStatus::Healthy => HealthCheckResult::healthy(&self.name, "OK"),
                HealthStatus::Degraded => HealthCheckResult::degraded(&self.name, "Slow"),
                HealthStatus::Unhealthy => HealthCheckResult::unhealthy(&self.name, "Failed"),
                HealthStatus::Unknown => HealthCheckResult::unknown(&self.name, "Unknown"),
            }
        }
    }

    #[tokio::test]
    async fn test_scheduler_creation() {
        let registry = Arc::new(HealthRegistry::new());
        let scheduler = HealthScheduler::with_defaults(registry);
        assert_eq!(scheduler.state().await, SchedulerState::Stopped);
    }

    #[tokio::test]
    async fn test_scheduler_config() {
        let config = SchedulerConfig::with_interval(Duration::from_secs(60))
            .check_on_start(false)
            .emit_only_on_change(false);

        assert_eq!(config.check_interval, Duration::from_secs(60));
        assert!(!config.check_on_start);
        assert!(!config.emit_only_on_change);
    }

    #[tokio::test]
    async fn test_run_check() {
        let registry = Arc::new(HealthRegistry::new());
        registry
            .register(
                "test",
                Arc::new(MockHealthCheck {
                    name: "test".to_string(),
                    status: HealthStatus::Healthy,
                }),
            )
            .await;

        let config = SchedulerConfig::default().check_on_start(false);
        let scheduler = HealthScheduler::new(registry, config);

        let report = scheduler.run_check().await;
        assert_eq!(report.status, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_scheduler_builder() {
        let registry = Arc::new(HealthRegistry::new());

        let scheduler = HealthSchedulerBuilder::new()
            .registry(registry)
            .interval(Duration::from_secs(10))
            .check_on_start(false)
            .build()
            .unwrap();

        assert_eq!(scheduler.state().await, SchedulerState::Stopped);
    }

    #[tokio::test]
    async fn test_scheduler_builder_without_registry() {
        let result = HealthSchedulerBuilder::new()
            .interval(Duration::from_secs(10))
            .build();

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let registry = Arc::new(HealthRegistry::new());
        registry
            .register(
                "test",
                Arc::new(MockHealthCheck {
                    name: "test".to_string(),
                    status: HealthStatus::Healthy,
                }),
            )
            .await;

        let config = SchedulerConfig::default()
            .check_on_start(false)
            .emit_only_on_change(false);
        let scheduler = HealthScheduler::new(registry, config);

        let mut rx = scheduler.subscribe();

        // Run a check
        scheduler.run_check().await;

        // Should receive the check completed event
        let event = tokio::time::timeout(Duration::from_millis(100), rx.recv()).await;
        assert!(event.is_ok());
    }
}
