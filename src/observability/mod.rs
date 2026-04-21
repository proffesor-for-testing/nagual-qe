//! Observability module for Nagual.
//!
//! Provides comprehensive observability features including:
//!
//! - **Metrics**: System metrics collection with SQLite persistence and 30-day retention
//! - **Tracing**: Integrated tracing spans with automatic metric recording
//! - **Logging**: Structured JSON logging with daily rotation and env-filter support
//! - **Redaction**: Privacy-preserving log redaction using PII detection
//! - **Alerts**: Local alerting system with desktop notifications and rate limiting
//!
//! # Example Usage
//!
//! ```rust,ignore
//! use nagual::observability::{
//!     metrics::{MetricsCollector, MetricsConfig, SystemMetric},
//!     logging::{init_logging, LoggingConfig, LogLevel},
//!     alerts::{AlertManager, Alert, AlertLevel},
//!     redaction::{LogRedactor, RedactionConfig},
//!     tracing::{MetricSpan, timed},
//! };
//!
//! // Initialize logging
//! let _log_handle = init_logging(
//!     LoggingConfig::new()
//!         .with_level(LogLevel::Info)
//!         .with_json(true)
//! )?;
//!
//! // Create metrics collector
//! let metrics = MetricsCollector::new(MetricsConfig::default());
//!
//! // Record metrics
//! metrics.record_metric("requests_total", 1.0);
//! metrics.timer("request_latency", 150.5);
//! metrics.gauge("active_connections", 42.0);
//!
//! // Create alert manager
//! let alerts = AlertManager::new();
//!
//! // Send alerts
//! alerts.info("System Started", "Nagual system initialized successfully")?;
//! alerts.warning("High Memory", "Memory usage exceeds 80%")?;
//!
//! // Use metric spans for automatic timing
//! {
//!     let _span = MetricSpan::new("database_query");
//!     // ... perform operation ...
//! } // Duration automatically recorded on drop
//!
//! // Redact PII from logs
//! let redactor = LogRedactor::new();
//! let safe_log = redactor.redact("User email: john@example.com");
//! ```
//!
//! # Features
//!
//! ## Metrics Collection
//!
//! The metrics system provides:
//! - Counter, gauge, histogram, and timer metric types
//! - SQLite persistence with configurable retention (default: 30 days)
//! - In-memory buffering with automatic flushing
//! - Time-range queries and aggregation statistics
//!
//! ## Structured Logging
//!
//! The logging system provides:
//! - JSON-formatted output for structured logging
//! - Daily log file rotation
//! - Configurable log levels per module
//! - Integration with tracing-subscriber
//!
//! ## Privacy-Preserving Redaction
//!
//! The redaction system provides:
//! - Automatic PII detection using regex patterns
//! - Configurable redaction levels (Low, Medium, High, Critical)
//! - Field-name based redaction for known sensitive fields
//! - Integration with tracing layers
//!
//! ## Local Alerting
//!
//! The alerting system provides:
//! - Desktop notifications via notify-rust
//! - Alert levels: Info, Warning, Critical
//! - Rate limiting to prevent notification spam
//! - Configurable alert rules with cooldown periods
//! - Alert history tracking

pub mod alerts;
pub mod logging;
pub mod metrics;
pub mod redaction;
pub mod tracing;

// Re-exports for convenience
pub use alerts::{Alert, AlertConfig, AlertLevel, AlertManager, AlertRule, AlertCondition, AlertStats};
pub use logging::{
    init_logging, init_default_logging, init_dev_logging, init_prod_logging,
    LoggingConfig, LogLevel, LoggingHandle, LogEntry,
};
pub use metrics::{MetricsCollector, MetricsConfig, MetricStats, MetricType, SystemMetric};
pub use redaction::{LogRedactor, RedactionConfig, RedactionAnalysis, redact, redact_with_config};
pub use self::tracing::{
    MetricSpan, InstrumentedOperation, Instrumented, MetricsLayer,
    DbOperationMetrics, timed, timed_async, timed_result, timed_result_async,
};

/// Default observability configuration.
#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    /// Metrics configuration.
    pub metrics: MetricsConfig,
    /// Logging configuration.
    pub logging: LoggingConfig,
    /// Alert configuration.
    pub alerts: AlertConfig,
    /// Redaction configuration.
    pub redaction: RedactionConfig,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            metrics: MetricsConfig::default(),
            logging: LoggingConfig::default(),
            alerts: AlertConfig::default(),
            redaction: RedactionConfig::default(),
        }
    }
}

impl ObservabilityConfig {
    /// Create a new configuration with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set metrics configuration.
    pub fn with_metrics(mut self, config: MetricsConfig) -> Self {
        self.metrics = config;
        self
    }

    /// Set logging configuration.
    pub fn with_logging(mut self, config: LoggingConfig) -> Self {
        self.logging = config;
        self
    }

    /// Set alerts configuration.
    pub fn with_alerts(mut self, config: AlertConfig) -> Self {
        self.alerts = config;
        self
    }

    /// Set redaction configuration.
    pub fn with_redaction(mut self, config: RedactionConfig) -> Self {
        self.redaction = config;
        self
    }
}

/// Initialize all observability components.
///
/// This is a convenience function that sets up logging, metrics, alerts,
/// and returns a handle to manage them.
pub fn init_observability(config: ObservabilityConfig) -> crate::error::Result<ObservabilityHandle> {
    // Initialize logging
    let logging_handle = init_logging(config.logging)?;

    // Create metrics collector (without persistence for now)
    let metrics = MetricsCollector::new(config.metrics);

    // Create alert manager
    let alerts = AlertManager::with_config(config.alerts);

    // Create redactor
    let redactor = LogRedactor::with_config(config.redaction);

    ::tracing::info!("Observability initialized");

    Ok(ObservabilityHandle {
        logging: logging_handle,
        metrics: std::sync::Arc::new(metrics),
        alerts: std::sync::Arc::new(alerts),
        redactor: std::sync::Arc::new(redactor),
    })
}

/// Handle for managing observability components.
pub struct ObservabilityHandle {
    /// Logging handle.
    pub logging: LoggingHandle,
    /// Metrics collector.
    pub metrics: std::sync::Arc<MetricsCollector>,
    /// Alert manager.
    pub alerts: std::sync::Arc<AlertManager>,
    /// Log redactor.
    pub redactor: std::sync::Arc<LogRedactor>,
}

impl ObservabilityHandle {
    /// Get the metrics collector.
    pub fn metrics(&self) -> &std::sync::Arc<MetricsCollector> {
        &self.metrics
    }

    /// Get the alert manager.
    pub fn alerts(&self) -> &std::sync::Arc<AlertManager> {
        &self.alerts
    }

    /// Get the log redactor.
    pub fn redactor(&self) -> &std::sync::Arc<LogRedactor> {
        &self.redactor
    }

    /// Record a metric.
    pub fn record_metric(&self, name: &str, value: f64) {
        self.metrics.record_metric(name, value);
    }

    /// Send an info alert.
    pub fn info(&self, title: &str, body: &str) -> crate::error::Result<bool> {
        self.alerts.info(title, body)
    }

    /// Send a warning alert.
    pub fn warning(&self, title: &str, body: &str) -> crate::error::Result<bool> {
        self.alerts.warning(title, body)
    }

    /// Send a critical alert.
    pub fn critical(&self, title: &str, body: &str) -> crate::error::Result<bool> {
        self.alerts.critical(title, body)
    }

    /// Redact PII from text.
    pub fn redact(&self, text: &str) -> String {
        self.redactor.redact(text)
    }

    /// Flush metrics to storage.
    pub fn flush_metrics(&self) -> crate::error::Result<usize> {
        self.metrics.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observability_config_defaults() {
        let config = ObservabilityConfig::default();
        assert_eq!(config.metrics.retention_days, 30);
        assert!(config.logging.json_format);
        assert!(config.alerts.desktop_notifications);
    }

    #[test]
    fn test_observability_config_builder() {
        let config = ObservabilityConfig::new()
            .with_metrics(MetricsConfig {
                retention_days: 7,
                ..Default::default()
            })
            .with_logging(LoggingConfig::new().with_level(LogLevel::Debug))
            .with_alerts(AlertConfig::new().with_rate_limit(5));

        assert_eq!(config.metrics.retention_days, 7);
        assert_eq!(config.logging.level, LogLevel::Debug);
        assert_eq!(config.alerts.rate_limit_per_minute, 5);
    }
}
