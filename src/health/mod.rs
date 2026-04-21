//! Health Check Framework for Nagual
//!
//! Provides a comprehensive health check system with:
//! - `HealthStatus` enum for representing component health states
//! - `HealthCheck` trait for implementing custom health checks
//! - `HealthRegistry` for managing and coordinating multiple health checks
//! - `HealthReport` for aggregating health check results
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::health::{HealthRegistry, HealthStatus, HealthReport};
//! use nagual::health::checks::SqliteHealthCheck;
//!
//! let mut registry = HealthRegistry::new();
//! registry.register("sqlite", Box::new(SqliteHealthCheck::new("./data.db")));
//!
//! let report = registry.check_all().await;
//! println!("System status: {:?}", report.overall_status());
//! ```

pub mod checks;
pub mod degradation;
pub mod scheduler;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::error::{NagualError, Result};

/// Health status of a component
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// Component is fully operational
    Healthy,
    /// Component is operational but with reduced capacity or warnings
    Degraded,
    /// Component is not operational
    Unhealthy,
    /// Component status is unknown (check failed or not yet performed)
    Unknown,
}

impl HealthStatus {
    /// Returns true if the status indicates the component is operational
    pub fn is_operational(&self) -> bool {
        matches!(self, HealthStatus::Healthy | HealthStatus::Degraded)
    }

    /// Returns the severity level (0 = healthy, 3 = unknown)
    pub fn severity(&self) -> u8 {
        match self {
            HealthStatus::Healthy => 0,
            HealthStatus::Degraded => 1,
            HealthStatus::Unhealthy => 2,
            HealthStatus::Unknown => 3,
        }
    }

    /// Combine two statuses, returning the worst one
    pub fn combine(self, other: HealthStatus) -> HealthStatus {
        if self.severity() >= other.severity() {
            self
        } else {
            other
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            HealthStatus::Healthy => "All systems operational",
            HealthStatus::Degraded => "Some systems degraded",
            HealthStatus::Unhealthy => "System failure detected",
            HealthStatus::Unknown => "Status unknown",
        }
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
            HealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

/// Result of a single health check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Name of the component
    pub component: String,
    /// Health status
    pub status: HealthStatus,
    /// Human-readable message
    pub message: String,
    /// Time the check was performed
    pub checked_at: DateTime<Utc>,
    /// Duration of the check
    #[serde(with = "duration_millis")]
    pub duration: Duration,
    /// Additional metadata
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

mod duration_millis {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

impl HealthCheckResult {
    /// Create a new healthy result
    pub fn healthy(component: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            component: component.into(),
            status: HealthStatus::Healthy,
            message: message.into(),
            checked_at: Utc::now(),
            duration: Duration::ZERO,
            metadata: HashMap::new(),
        }
    }

    /// Create a new degraded result
    pub fn degraded(component: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            component: component.into(),
            status: HealthStatus::Degraded,
            message: message.into(),
            checked_at: Utc::now(),
            duration: Duration::ZERO,
            metadata: HashMap::new(),
        }
    }

    /// Create a new unhealthy result
    pub fn unhealthy(component: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            component: component.into(),
            status: HealthStatus::Unhealthy,
            message: message.into(),
            checked_at: Utc::now(),
            duration: Duration::ZERO,
            metadata: HashMap::new(),
        }
    }

    /// Create an unknown result (check failed)
    pub fn unknown(component: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            component: component.into(),
            status: HealthStatus::Unknown,
            message: message.into(),
            checked_at: Utc::now(),
            duration: Duration::ZERO,
            metadata: HashMap::new(),
        }
    }

    /// Set the check duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Trait for implementing health checks
#[async_trait::async_trait]
pub trait HealthCheck: Send + Sync {
    /// Get the name of this health check
    fn name(&self) -> &str;

    /// Perform the health check
    async fn check(&self) -> HealthCheckResult;

    /// Get the timeout for this check (default: 10 seconds)
    fn timeout(&self) -> Duration {
        Duration::from_secs(10)
    }

    /// Whether this check is critical (affects overall system health)
    fn is_critical(&self) -> bool {
        true
    }
}

/// Aggregated health report for multiple components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Overall system status
    pub status: HealthStatus,
    /// Individual component results
    pub components: HashMap<String, HealthCheckResult>,
    /// Time the report was generated
    pub generated_at: DateTime<Utc>,
    /// Total number of checks
    pub total_checks: usize,
    /// Number of healthy checks
    pub healthy_count: usize,
    /// Number of degraded checks
    pub degraded_count: usize,
    /// Number of unhealthy checks
    pub unhealthy_count: usize,
    /// System uptime
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uptime: Option<Duration>,
}

impl HealthReport {
    /// Create a new empty report
    pub fn new() -> Self {
        Self {
            status: HealthStatus::Unknown,
            components: HashMap::new(),
            generated_at: Utc::now(),
            total_checks: 0,
            healthy_count: 0,
            degraded_count: 0,
            unhealthy_count: 0,
            uptime: None,
        }
    }

    /// Create a report from component results
    pub fn from_results(results: Vec<HealthCheckResult>) -> Self {
        let mut report = Self::new();

        for result in results {
            report.add_result(result);
        }

        report.calculate_overall_status();
        report
    }

    /// Add a health check result
    pub fn add_result(&mut self, result: HealthCheckResult) {
        match result.status {
            HealthStatus::Healthy => self.healthy_count += 1,
            HealthStatus::Degraded => self.degraded_count += 1,
            HealthStatus::Unhealthy => self.unhealthy_count += 1,
            HealthStatus::Unknown => {}
        }
        self.total_checks += 1;
        self.components.insert(result.component.clone(), result);
    }

    /// Calculate the overall status from component statuses
    pub fn calculate_overall_status(&mut self) {
        self.status = if self.unhealthy_count > 0 {
            HealthStatus::Unhealthy
        } else if self.degraded_count > 0 {
            HealthStatus::Degraded
        } else if self.healthy_count > 0 {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unknown
        };
    }

    /// Get the overall status
    pub fn overall_status(&self) -> HealthStatus {
        self.status
    }

    /// Check if the system is operational
    pub fn is_operational(&self) -> bool {
        self.status.is_operational()
    }

    /// Get results for unhealthy components
    pub fn unhealthy_components(&self) -> Vec<&HealthCheckResult> {
        self.components
            .values()
            .filter(|r| r.status == HealthStatus::Unhealthy)
            .collect()
    }

    /// Get results for degraded components
    pub fn degraded_components(&self) -> Vec<&HealthCheckResult> {
        self.components
            .values()
            .filter(|r| r.status == HealthStatus::Degraded)
            .collect()
    }

    /// Set uptime
    pub fn with_uptime(mut self, uptime: Duration) -> Self {
        self.uptime = Some(uptime);
        self
    }

    /// Format as human-readable text
    pub fn to_text(&self) -> String {
        let mut output = String::new();

        // Overall status
        let status_icon = match self.status {
            HealthStatus::Healthy => "[OK]",
            HealthStatus::Degraded => "[WARN]",
            HealthStatus::Unhealthy => "[FAIL]",
            HealthStatus::Unknown => "[?]",
        };

        output.push_str(&format!(
            "{} Overall Status: {}\n",
            status_icon,
            self.status.description()
        ));

        if let Some(uptime) = self.uptime {
            output.push_str(&format!("    Uptime: {:?}\n", uptime));
        }

        output.push_str(&format!(
            "    Checks: {} total, {} healthy, {} degraded, {} unhealthy\n\n",
            self.total_checks, self.healthy_count, self.degraded_count, self.unhealthy_count
        ));

        // Component details
        output.push_str("Components:\n");

        let mut components: Vec<_> = self.components.iter().collect();
        components.sort_by_key(|(name, _)| name.as_str());

        for (name, result) in components {
            let icon = match result.status {
                HealthStatus::Healthy => "[OK]",
                HealthStatus::Degraded => "[WARN]",
                HealthStatus::Unhealthy => "[FAIL]",
                HealthStatus::Unknown => "[?]",
            };

            output.push_str(&format!(
                "  {} {}: {}\n",
                icon, name, result.message
            ));

            if result.duration.as_millis() > 0 {
                output.push_str(&format!("      Response time: {:?}\n", result.duration));
            }

            for (key, value) in &result.metadata {
                output.push_str(&format!("      {}: {}\n", key, value));
            }
        }

        output
    }
}

impl Default for HealthReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry for managing health checks
pub struct HealthRegistry {
    checks: Arc<RwLock<HashMap<String, Arc<dyn HealthCheck>>>>,
    last_report: Arc<RwLock<Option<HealthReport>>>,
    start_time: std::time::Instant,
}

impl HealthRegistry {
    /// Create a new health registry
    pub fn new() -> Self {
        Self {
            checks: Arc::new(RwLock::new(HashMap::new())),
            last_report: Arc::new(RwLock::new(None)),
            start_time: std::time::Instant::now(),
        }
    }

    /// Register a health check
    pub async fn register(&self, name: impl Into<String>, check: Arc<dyn HealthCheck>) {
        let mut checks = self.checks.write().await;
        checks.insert(name.into(), check);
    }

    /// Unregister a health check
    pub async fn unregister(&self, name: &str) -> Option<Arc<dyn HealthCheck>> {
        let mut checks = self.checks.write().await;
        checks.remove(name)
    }

    /// Get all registered check names
    pub async fn list_checks(&self) -> Vec<String> {
        let checks = self.checks.read().await;
        checks.keys().cloned().collect()
    }

    /// Run a single health check by name
    pub async fn check(&self, name: &str) -> Result<HealthCheckResult> {
        let checks = self.checks.read().await;
        let check = checks
            .get(name)
            .ok_or_else(|| NagualError::Internal {
                message: format!("Health check '{}' not found", name),
            })?
            .clone();
        drop(checks);

        let timeout = check.timeout();
        let start = std::time::Instant::now();

        let result = tokio::time::timeout(timeout, check.check())
            .await
            .map(|mut result| {
                result.duration = start.elapsed();
                result
            })
            .unwrap_or_else(|_| {
                HealthCheckResult::unhealthy(
                    name,
                    format!("Health check timed out after {:?}", timeout),
                )
                .with_duration(start.elapsed())
            });

        Ok(result)
    }

    /// Run all health checks
    pub async fn check_all(&self) -> HealthReport {
        let checks = self.checks.read().await;
        let check_list: Vec<_> = checks.iter().map(|(n, c)| (n.clone(), c.clone())).collect();
        drop(checks);

        let mut results = Vec::new();

        for (name, check) in check_list {
            let timeout = check.timeout();
            let start = std::time::Instant::now();

            let result = tokio::time::timeout(timeout, check.check())
                .await
                .map(|mut result| {
                    result.duration = start.elapsed();
                    result
                })
                .unwrap_or_else(|_| {
                    HealthCheckResult::unhealthy(
                        &name,
                        format!("Health check timed out after {:?}", timeout),
                    )
                    .with_duration(start.elapsed())
                });

            results.push(result);
        }

        let report = HealthReport::from_results(results)
            .with_uptime(self.start_time.elapsed());

        // Store the last report
        let mut last_report = self.last_report.write().await;
        *last_report = Some(report.clone());

        report
    }

    /// Run only critical health checks
    pub async fn check_critical(&self) -> HealthReport {
        let checks = self.checks.read().await;
        let check_list: Vec<_> = checks
            .iter()
            .filter(|(_, c)| c.is_critical())
            .map(|(n, c)| (n.clone(), c.clone()))
            .collect();
        drop(checks);

        let mut results = Vec::new();

        for (name, check) in check_list {
            let timeout = check.timeout();
            let start = std::time::Instant::now();

            let result = tokio::time::timeout(timeout, check.check())
                .await
                .map(|mut result| {
                    result.duration = start.elapsed();
                    result
                })
                .unwrap_or_else(|_| {
                    HealthCheckResult::unhealthy(
                        &name,
                        format!("Health check timed out after {:?}", timeout),
                    )
                    .with_duration(start.elapsed())
                });

            results.push(result);
        }

        HealthReport::from_results(results).with_uptime(self.start_time.elapsed())
    }

    /// Get the last health report
    pub async fn last_report(&self) -> Option<HealthReport> {
        let report = self.last_report.read().await;
        report.clone()
    }

    /// Get system uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Default for HealthRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Event emitted when health status changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthChangeEvent {
    /// Component that changed
    pub component: String,
    /// Previous status
    pub previous_status: HealthStatus,
    /// New status
    pub new_status: HealthStatus,
    /// Time of change
    pub changed_at: DateTime<Utc>,
    /// Message describing the change
    pub message: String,
}

impl HealthChangeEvent {
    /// Create a new health change event
    pub fn new(
        component: impl Into<String>,
        previous: HealthStatus,
        new: HealthStatus,
        message: impl Into<String>,
    ) -> Self {
        Self {
            component: component.into(),
            previous_status: previous,
            new_status: new,
            changed_at: Utc::now(),
            message: message.into(),
        }
    }

    /// Check if this represents a degradation
    pub fn is_degradation(&self) -> bool {
        self.new_status.severity() > self.previous_status.severity()
    }

    /// Check if this represents a recovery
    pub fn is_recovery(&self) -> bool {
        self.new_status.severity() < self.previous_status.severity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_combine() {
        assert_eq!(
            HealthStatus::Healthy.combine(HealthStatus::Degraded),
            HealthStatus::Degraded
        );
        assert_eq!(
            HealthStatus::Unhealthy.combine(HealthStatus::Healthy),
            HealthStatus::Unhealthy
        );
    }

    #[test]
    fn test_health_status_severity() {
        assert!(HealthStatus::Healthy.severity() < HealthStatus::Degraded.severity());
        assert!(HealthStatus::Degraded.severity() < HealthStatus::Unhealthy.severity());
    }

    #[test]
    fn test_health_check_result_creation() {
        let result = HealthCheckResult::healthy("test", "All good")
            .with_metadata("version", serde_json::json!("1.0.0"));

        assert_eq!(result.status, HealthStatus::Healthy);
        assert_eq!(result.component, "test");
        assert!(result.metadata.contains_key("version"));
    }

    #[test]
    fn test_health_report_from_results() {
        let results = vec![
            HealthCheckResult::healthy("db", "Connected"),
            HealthCheckResult::degraded("cache", "Slow"),
            HealthCheckResult::healthy("api", "Ready"),
        ];

        let report = HealthReport::from_results(results);
        assert_eq!(report.status, HealthStatus::Degraded);
        assert_eq!(report.healthy_count, 2);
        assert_eq!(report.degraded_count, 1);
    }

    #[test]
    fn test_health_change_event() {
        let event = HealthChangeEvent::new(
            "database",
            HealthStatus::Healthy,
            HealthStatus::Unhealthy,
            "Connection lost",
        );

        assert!(event.is_degradation());
        assert!(!event.is_recovery());
    }

    #[tokio::test]
    async fn test_health_registry() {
        struct MockCheck;

        #[async_trait::async_trait]
        impl HealthCheck for MockCheck {
            fn name(&self) -> &str {
                "mock"
            }

            async fn check(&self) -> HealthCheckResult {
                HealthCheckResult::healthy("mock", "OK")
            }
        }

        let registry = HealthRegistry::new();
        registry.register("mock", Arc::new(MockCheck)).await;

        let checks = registry.list_checks().await;
        assert!(checks.contains(&"mock".to_string()));

        let result = registry.check("mock").await.unwrap();
        assert_eq!(result.status, HealthStatus::Healthy);
    }
}
