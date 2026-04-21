//! Local alerting with desktop notifications.
//!
//! Provides an `AlertManager` for sending system alerts via desktop notifications
//! using notify-rust, with configurable alert rules and rate limiting.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use notify_rust::{Notification, Timeout};
#[cfg(target_os = "linux")]
use notify_rust::Urgency;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::error::Result;

/// Alert severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AlertLevel {
    /// Informational alert.
    Info,
    /// Warning alert.
    Warning,
    /// Critical alert requiring immediate attention.
    Critical,
}

impl AlertLevel {
    /// Convert to notify-rust Urgency (Linux only).
    #[cfg(target_os = "linux")]
    pub fn to_urgency(self) -> Urgency {
        match self {
            AlertLevel::Info => Urgency::Low,
            AlertLevel::Warning => Urgency::Normal,
            AlertLevel::Critical => Urgency::Critical,
        }
    }

    /// Get the icon name for the notification.
    pub fn icon_name(self) -> &'static str {
        match self {
            AlertLevel::Info => "dialog-information",
            AlertLevel::Warning => "dialog-warning",
            AlertLevel::Critical => "dialog-error",
        }
    }

    /// Get the notification timeout.
    pub fn timeout(self) -> Timeout {
        match self {
            AlertLevel::Info => Timeout::Milliseconds(5000),
            AlertLevel::Warning => Timeout::Milliseconds(10000),
            AlertLevel::Critical => Timeout::Never,
        }
    }

    /// Get a color for the alert (for terminal output).
    pub fn color(self) -> &'static str {
        match self {
            AlertLevel::Info => "\x1b[34m",    // Blue
            AlertLevel::Warning => "\x1b[33m", // Yellow
            AlertLevel::Critical => "\x1b[31m", // Red
        }
    }

    /// Get the string representation.
    pub fn as_str(self) -> &'static str {
        match self {
            AlertLevel::Info => "INFO",
            AlertLevel::Warning => "WARNING",
            AlertLevel::Critical => "CRITICAL",
        }
    }
}

impl std::fmt::Display for AlertLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// An alert to be sent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Unique identifier for this alert.
    pub id: String,
    /// Alert title.
    pub title: String,
    /// Alert message body.
    pub body: String,
    /// Alert level.
    pub level: AlertLevel,
    /// When the alert was created.
    pub created_at: DateTime<Utc>,
    /// Source of the alert (component/module).
    pub source: Option<String>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl Alert {
    /// Create a new alert.
    pub fn new(level: AlertLevel, title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            title: title.into(),
            body: body.into(),
            level,
            created_at: Utc::now(),
            source: None,
            metadata: HashMap::new(),
        }
    }

    /// Create an info alert.
    pub fn info(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self::new(AlertLevel::Info, title, body)
    }

    /// Create a warning alert.
    pub fn warning(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self::new(AlertLevel::Warning, title, body)
    }

    /// Create a critical alert.
    pub fn critical(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self::new(AlertLevel::Critical, title, body)
    }

    /// Set the source.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A rule for triggering alerts.
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule name.
    pub name: String,
    /// Rule description.
    pub description: String,
    /// Alert level to use when triggered.
    pub level: AlertLevel,
    /// Condition function that returns true when alert should trigger.
    pub condition: AlertCondition,
    /// Cooldown period between alerts (prevents spam).
    pub cooldown: Duration,
    /// Whether the rule is enabled.
    pub enabled: bool,
}

impl AlertRule {
    /// Create a new alert rule.
    pub fn new(
        name: impl Into<String>,
        level: AlertLevel,
        condition: AlertCondition,
    ) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            level,
            condition,
            cooldown: Duration::from_secs(300), // 5 minutes default
            enabled: true,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the cooldown period.
    pub fn with_cooldown(mut self, cooldown: Duration) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Enable or disable the rule.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// Condition types for alert rules.
#[derive(Debug, Clone)]
pub enum AlertCondition {
    /// Trigger when a metric exceeds a threshold.
    MetricAbove {
        metric_name: String,
        threshold: f64,
    },
    /// Trigger when a metric falls below a threshold.
    MetricBelow {
        metric_name: String,
        threshold: f64,
    },
    /// Trigger when error rate exceeds threshold.
    ErrorRateAbove {
        threshold: f64,
    },
    /// Trigger on any error.
    OnError,
    /// Custom condition with a name.
    Custom {
        name: String,
    },
}

impl AlertCondition {
    /// Get a description of the condition.
    pub fn description(&self) -> String {
        match self {
            AlertCondition::MetricAbove { metric_name, threshold } => {
                format!("{} > {}", metric_name, threshold)
            }
            AlertCondition::MetricBelow { metric_name, threshold } => {
                format!("{} < {}", metric_name, threshold)
            }
            AlertCondition::ErrorRateAbove { threshold } => {
                format!("error_rate > {}", threshold)
            }
            AlertCondition::OnError => "on_error".to_string(),
            AlertCondition::Custom { name } => name.clone(),
        }
    }
}

/// Configuration for the alert manager.
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Application name for notifications.
    pub app_name: String,
    /// Enable desktop notifications.
    pub desktop_notifications: bool,
    /// Enable console output for alerts.
    pub console_output: bool,
    /// Global rate limit (max alerts per minute).
    pub rate_limit_per_minute: u32,
    /// Minimum alert level to send.
    pub min_level: AlertLevel,
    /// Maximum alerts to keep in history.
    pub history_size: usize,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            app_name: "Nagual".to_string(),
            desktop_notifications: true,
            console_output: true,
            rate_limit_per_minute: 10,
            min_level: AlertLevel::Info,
            history_size: 1000,
        }
    }
}

impl AlertConfig {
    /// Create a new config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the application name.
    pub fn with_app_name(mut self, name: impl Into<String>) -> Self {
        self.app_name = name.into();
        self
    }

    /// Enable or disable desktop notifications.
    pub fn with_desktop_notifications(mut self, enabled: bool) -> Self {
        self.desktop_notifications = enabled;
        self
    }

    /// Enable or disable console output.
    pub fn with_console_output(mut self, enabled: bool) -> Self {
        self.console_output = enabled;
        self
    }

    /// Set the rate limit.
    pub fn with_rate_limit(mut self, per_minute: u32) -> Self {
        self.rate_limit_per_minute = per_minute;
        self
    }

    /// Set the minimum alert level.
    pub fn with_min_level(mut self, level: AlertLevel) -> Self {
        self.min_level = level;
        self
    }
}

/// Rate limiter for alerts.
struct RateLimiter {
    timestamps: RwLock<Vec<Instant>>,
    max_per_minute: u32,
}

impl RateLimiter {
    fn new(max_per_minute: u32) -> Self {
        Self {
            timestamps: RwLock::new(Vec::new()),
            max_per_minute,
        }
    }

    fn check(&self) -> bool {
        let now = Instant::now();
        let minute_ago = now - Duration::from_secs(60);

        let mut timestamps = self.timestamps.write();

        // Remove old timestamps
        timestamps.retain(|&t| t > minute_ago);

        if timestamps.len() < self.max_per_minute as usize {
            timestamps.push(now);
            true
        } else {
            false
        }
    }
}

/// Alert manager for sending and tracking alerts.
pub struct AlertManager {
    config: AlertConfig,
    rules: RwLock<Vec<AlertRule>>,
    rule_last_triggered: RwLock<HashMap<String, Instant>>,
    history: RwLock<Vec<Alert>>,
    rate_limiter: RateLimiter,
    /// Total alerts sent.
    alerts_sent: RwLock<u64>,
    /// Alerts suppressed by rate limiting.
    alerts_suppressed: RwLock<u64>,
}

impl AlertManager {
    /// Create a new alert manager with default configuration.
    pub fn new() -> Self {
        Self::with_config(AlertConfig::default())
    }

    /// Create an alert manager with custom configuration.
    pub fn with_config(config: AlertConfig) -> Self {
        Self {
            rate_limiter: RateLimiter::new(config.rate_limit_per_minute),
            config,
            rules: RwLock::new(Vec::new()),
            rule_last_triggered: RwLock::new(HashMap::new()),
            history: RwLock::new(Vec::new()),
            alerts_sent: RwLock::new(0),
            alerts_suppressed: RwLock::new(0),
        }
    }

    /// Send an alert.
    pub fn send_alert(&self, alert: Alert) -> Result<bool> {
        // Check minimum level
        if alert.level < self.config.min_level {
            debug!(
                alert_id = %alert.id,
                level = %alert.level,
                min_level = %self.config.min_level,
                "Alert below minimum level, skipping"
            );
            return Ok(false);
        }

        // Check rate limit
        if !self.rate_limiter.check() {
            *self.alerts_suppressed.write() += 1;
            warn!(
                alert_id = %alert.id,
                "Alert suppressed by rate limiter"
            );
            return Ok(false);
        }

        // Add to history
        {
            let mut history = self.history.write();
            history.push(alert.clone());

            // Trim history if needed
            if history.len() > self.config.history_size {
                history.remove(0);
            }
        }

        // Send desktop notification
        if self.config.desktop_notifications {
            self.send_desktop_notification(&alert)?;
        }

        // Console output
        if self.config.console_output {
            self.print_console_alert(&alert);
        }

        *self.alerts_sent.write() += 1;

        info!(
            alert_id = %alert.id,
            level = %alert.level,
            title = %alert.title,
            "Alert sent"
        );

        Ok(true)
    }

    /// Send a desktop notification.
    fn send_desktop_notification(&self, alert: &Alert) -> Result<()> {
        let mut notification = Notification::new();
        notification
            .summary(&alert.title)
            .body(&alert.body)
            .appname(&self.config.app_name)
            .icon(alert.level.icon_name())
            .timeout(alert.level.timeout());
        #[cfg(target_os = "linux")]
        notification.urgency(alert.level.to_urgency());
        let result = notification.show();

        match result {
            Ok(_) => {
                debug!(alert_id = %alert.id, "Desktop notification sent");
                Ok(())
            }
            Err(e) => {
                warn!(
                    alert_id = %alert.id,
                    error = %e,
                    "Failed to send desktop notification"
                );
                // Don't fail the whole alert, just log the error
                Ok(())
            }
        }
    }

    /// Print alert to console.
    fn print_console_alert(&self, alert: &Alert) {
        let reset = "\x1b[0m";
        let color = alert.level.color();

        eprintln!(
            "{}[{}]{} {}: {}",
            color,
            alert.level,
            reset,
            alert.title,
            alert.body
        );

        if let Some(ref source) = alert.source {
            eprintln!("  Source: {}", source);
        }

        for (key, value) in &alert.metadata {
            eprintln!("  {}: {}", key, value);
        }
    }

    /// Register an alert rule.
    pub fn register_rule(&self, rule: AlertRule) {
        info!(rule_name = %rule.name, "Registering alert rule");
        self.rules.write().push(rule);
    }

    /// Check rules against current metrics.
    pub fn check_rules(&self, metrics: &HashMap<String, f64>) -> Vec<Alert> {
        let rules = self.rules.read();
        let mut triggered = Vec::new();
        let now = Instant::now();

        for rule in rules.iter() {
            if !rule.enabled {
                continue;
            }

            // Check cooldown
            {
                let last_triggered = self.rule_last_triggered.read();
                if let Some(last) = last_triggered.get(&rule.name) {
                    if now.duration_since(*last) < rule.cooldown {
                        continue;
                    }
                }
            }

            let should_trigger = match &rule.condition {
                AlertCondition::MetricAbove { metric_name, threshold } => {
                    metrics.get(metric_name).map(|v| *v > *threshold).unwrap_or(false)
                }
                AlertCondition::MetricBelow { metric_name, threshold } => {
                    metrics.get(metric_name).map(|v| *v < *threshold).unwrap_or(false)
                }
                AlertCondition::ErrorRateAbove { threshold } => {
                    metrics.get("error_rate").map(|v| *v > *threshold).unwrap_or(false)
                }
                AlertCondition::OnError => {
                    metrics.get("error_count").map(|v| *v > 0.0).unwrap_or(false)
                }
                AlertCondition::Custom { .. } => false, // Custom conditions need explicit triggering
            };

            if should_trigger {
                let alert = Alert::new(
                    rule.level,
                    &rule.name,
                    format!("{}: {}", rule.description, rule.condition.description()),
                )
                .with_source("AlertManager")
                .with_metadata("rule", &rule.name);

                triggered.push(alert);

                // Update last triggered
                self.rule_last_triggered.write().insert(rule.name.clone(), now);
            }
        }

        triggered
    }

    /// Trigger a custom rule by name.
    pub fn trigger_rule(&self, rule_name: &str, message: &str) -> Option<Alert> {
        let rules = self.rules.read();

        if let Some(rule) = rules.iter().find(|r| r.name == rule_name && r.enabled) {
            // Check cooldown
            let now = Instant::now();
            {
                let last_triggered = self.rule_last_triggered.read();
                if let Some(last) = last_triggered.get(&rule.name) {
                    if now.duration_since(*last) < rule.cooldown {
                        return None;
                    }
                }
            }

            let alert = Alert::new(rule.level, &rule.name, message)
                .with_source("AlertManager")
                .with_metadata("rule", &rule.name);

            self.rule_last_triggered.write().insert(rule.name.clone(), now);

            Some(alert)
        } else {
            None
        }
    }

    /// Get alert history.
    pub fn history(&self) -> Vec<Alert> {
        self.history.read().clone()
    }

    /// Get recent alerts (last N).
    pub fn recent_alerts(&self, count: usize) -> Vec<Alert> {
        let history = self.history.read();
        history.iter().rev().take(count).cloned().collect()
    }

    /// Get alerts by level.
    pub fn alerts_by_level(&self, level: AlertLevel) -> Vec<Alert> {
        let history = self.history.read();
        history.iter().filter(|a| a.level == level).cloned().collect()
    }

    /// Get alert statistics.
    pub fn stats(&self) -> AlertStats {
        let history = self.history.read();

        AlertStats {
            total_sent: *self.alerts_sent.read(),
            total_suppressed: *self.alerts_suppressed.read(),
            history_size: history.len(),
            info_count: history.iter().filter(|a| a.level == AlertLevel::Info).count(),
            warning_count: history.iter().filter(|a| a.level == AlertLevel::Warning).count(),
            critical_count: history.iter().filter(|a| a.level == AlertLevel::Critical).count(),
            rules_count: self.rules.read().len(),
        }
    }

    /// Clear alert history.
    pub fn clear_history(&self) {
        self.history.write().clear();
    }

    /// Convenience method to send an info alert.
    pub fn info(&self, title: impl Into<String>, body: impl Into<String>) -> Result<bool> {
        self.send_alert(Alert::info(title, body))
    }

    /// Convenience method to send a warning alert.
    pub fn warning(&self, title: impl Into<String>, body: impl Into<String>) -> Result<bool> {
        self.send_alert(Alert::warning(title, body))
    }

    /// Convenience method to send a critical alert.
    pub fn critical(&self, title: impl Into<String>, body: impl Into<String>) -> Result<bool> {
        self.send_alert(Alert::critical(title, body))
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Alert statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertStats {
    /// Total alerts sent.
    pub total_sent: u64,
    /// Total alerts suppressed by rate limiting.
    pub total_suppressed: u64,
    /// Current history size.
    pub history_size: usize,
    /// Info alerts in history.
    pub info_count: usize,
    /// Warning alerts in history.
    pub warning_count: usize,
    /// Critical alerts in history.
    pub critical_count: usize,
    /// Number of registered rules.
    pub rules_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AlertConfig {
        AlertConfig::new()
            .with_desktop_notifications(false) // Don't send real notifications in tests
            .with_console_output(false)
    }

    #[test]
    fn test_alert_creation() {
        let alert = Alert::new(AlertLevel::Warning, "Test Alert", "This is a test")
            .with_source("test")
            .with_metadata("key", "value");

        assert_eq!(alert.level, AlertLevel::Warning);
        assert_eq!(alert.title, "Test Alert");
        assert_eq!(alert.body, "This is a test");
        assert_eq!(alert.source, Some("test".to_string()));
        assert_eq!(alert.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_alert_convenience_constructors() {
        let info = Alert::info("Info", "Info message");
        assert_eq!(info.level, AlertLevel::Info);

        let warning = Alert::warning("Warning", "Warning message");
        assert_eq!(warning.level, AlertLevel::Warning);

        let critical = Alert::critical("Critical", "Critical message");
        assert_eq!(critical.level, AlertLevel::Critical);
    }

    #[test]
    fn test_alert_level_ordering() {
        assert!(AlertLevel::Info < AlertLevel::Warning);
        assert!(AlertLevel::Warning < AlertLevel::Critical);
    }

    #[test]
    fn test_alert_manager_send() {
        let manager = AlertManager::with_config(test_config());

        let result = manager.send_alert(Alert::info("Test", "Test message"));
        assert!(result.is_ok());
        assert!(result.unwrap());

        let stats = manager.stats();
        assert_eq!(stats.total_sent, 1);
        assert_eq!(stats.history_size, 1);
    }

    #[test]
    fn test_alert_min_level_filter() {
        let config = test_config().with_min_level(AlertLevel::Warning);
        let manager = AlertManager::with_config(config);

        // Info should be filtered
        let result = manager.send_alert(Alert::info("Info", "Should be filtered"));
        assert!(!result.unwrap());

        // Warning should pass
        let result = manager.send_alert(Alert::warning("Warning", "Should pass"));
        assert!(result.unwrap());

        let stats = manager.stats();
        assert_eq!(stats.total_sent, 1);
    }

    #[test]
    fn test_alert_rule_registration() {
        let manager = AlertManager::with_config(test_config());

        let rule = AlertRule::new(
            "high_memory",
            AlertLevel::Warning,
            AlertCondition::MetricAbove {
                metric_name: "memory_usage".to_string(),
                threshold: 90.0,
            },
        )
        .with_description("Memory usage exceeds 90%")
        .with_cooldown(Duration::from_secs(60));

        manager.register_rule(rule);

        let stats = manager.stats();
        assert_eq!(stats.rules_count, 1);
    }

    #[test]
    fn test_alert_rule_check() {
        let manager = AlertManager::with_config(test_config());

        manager.register_rule(AlertRule::new(
            "high_cpu",
            AlertLevel::Warning,
            AlertCondition::MetricAbove {
                metric_name: "cpu_usage".to_string(),
                threshold: 80.0,
            },
        ).with_cooldown(Duration::from_millis(0))); // No cooldown for test

        // Should not trigger
        let mut metrics = HashMap::new();
        metrics.insert("cpu_usage".to_string(), 50.0);

        let triggered = manager.check_rules(&metrics);
        assert!(triggered.is_empty());

        // Should trigger
        metrics.insert("cpu_usage".to_string(), 90.0);

        let triggered = manager.check_rules(&metrics);
        assert_eq!(triggered.len(), 1);
        assert_eq!(triggered[0].level, AlertLevel::Warning);
    }

    #[test]
    fn test_alert_rule_cooldown() {
        let manager = AlertManager::with_config(test_config());

        manager.register_rule(AlertRule::new(
            "test_rule",
            AlertLevel::Info,
            AlertCondition::MetricAbove {
                metric_name: "test".to_string(),
                threshold: 0.0,
            },
        ).with_cooldown(Duration::from_secs(60)));

        let mut metrics = HashMap::new();
        metrics.insert("test".to_string(), 1.0);

        // First check should trigger
        let triggered = manager.check_rules(&metrics);
        assert_eq!(triggered.len(), 1);

        // Second check should not trigger (cooldown)
        let triggered = manager.check_rules(&metrics);
        assert!(triggered.is_empty());
    }

    #[test]
    fn test_alert_history() {
        let manager = AlertManager::with_config(test_config());

        manager.send_alert(Alert::info("Alert 1", "Body 1")).unwrap();
        manager.send_alert(Alert::warning("Alert 2", "Body 2")).unwrap();
        manager.send_alert(Alert::critical("Alert 3", "Body 3")).unwrap();

        let history = manager.history();
        assert_eq!(history.len(), 3);

        let recent = manager.recent_alerts(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].title, "Alert 3"); // Most recent first

        let warnings = manager.alerts_by_level(AlertLevel::Warning);
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn test_rate_limiting() {
        let config = test_config().with_rate_limit(2);
        let manager = AlertManager::with_config(config);

        // First two should succeed
        assert!(manager.send_alert(Alert::info("1", "")).unwrap());
        assert!(manager.send_alert(Alert::info("2", "")).unwrap());

        // Third should be rate limited
        assert!(!manager.send_alert(Alert::info("3", "")).unwrap());

        let stats = manager.stats();
        assert_eq!(stats.total_sent, 2);
        assert_eq!(stats.total_suppressed, 1);
    }

    #[test]
    fn test_convenience_methods() {
        let manager = AlertManager::with_config(test_config());

        manager.info("Info", "Info message").unwrap();
        manager.warning("Warning", "Warning message").unwrap();
        manager.critical("Critical", "Critical message").unwrap();

        let stats = manager.stats();
        assert_eq!(stats.total_sent, 3);
        assert_eq!(stats.info_count, 1);
        assert_eq!(stats.warning_count, 1);
        assert_eq!(stats.critical_count, 1);
    }

    #[test]
    fn test_clear_history() {
        let manager = AlertManager::with_config(test_config());

        manager.send_alert(Alert::info("Test", "")).unwrap();
        assert_eq!(manager.history().len(), 1);

        manager.clear_history();
        assert!(manager.history().is_empty());
    }

    #[test]
    fn test_alert_condition_description() {
        let above = AlertCondition::MetricAbove {
            metric_name: "cpu".to_string(),
            threshold: 80.0,
        };
        assert_eq!(above.description(), "cpu > 80");

        let below = AlertCondition::MetricBelow {
            metric_name: "memory".to_string(),
            threshold: 10.0,
        };
        assert_eq!(below.description(), "memory < 10");
    }
}
