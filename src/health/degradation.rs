//! Graceful Degradation Manager
//!
//! Provides intelligent degradation management that allows the system to
//! continue operating with reduced functionality when components fail.
//!
//! Features:
//! - Feature flags for degraded mode
//! - Fallback strategies per component
//! - Automatic recovery when health is restored
//! - Circuit breaker pattern support

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::{HealthChangeEvent, HealthRegistry, HealthStatus};

/// Feature flag representing a system capability
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FeatureFlag {
    /// Unique identifier for this feature
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Whether this feature is currently enabled
    pub enabled: bool,
    /// Components this feature depends on
    pub dependencies: Vec<String>,
    /// Whether this feature can be manually overridden
    pub can_override: bool,
}

impl FeatureFlag {
    /// Create a new feature flag
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            enabled: true,
            dependencies: Vec::new(),
            can_override: true,
        }
    }

    /// Add a dependency
    pub fn depends_on(mut self, component: impl Into<String>) -> Self {
        self.dependencies.push(component.into());
        self
    }

    /// Set whether this feature can be overridden
    pub fn can_override(mut self, can: bool) -> Self {
        self.can_override = can;
        self
    }

    /// Check if all dependencies are satisfied
    pub fn check_dependencies(&self, healthy_components: &[String]) -> bool {
        self.dependencies
            .iter()
            .all(|dep| healthy_components.contains(dep))
    }
}

/// Fallback strategy for a component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackStrategy {
    /// No fallback - feature is simply disabled
    Disable,
    /// Use cached data if available
    UseCache {
        /// Maximum age of cached data to use
        max_age: Duration,
    },
    /// Use a default/static value
    UseDefault {
        /// Description of the default behavior
        description: String,
    },
    /// Redirect to an alternative component
    Redirect {
        /// Name of the alternative component
        target: String,
    },
    /// Retry with exponential backoff
    Retry {
        /// Maximum number of retries
        max_retries: u32,
        /// Base delay between retries
        base_delay: Duration,
    },
    /// Queue operations for later processing
    Queue {
        /// Maximum queue size
        max_size: usize,
    },
    /// Custom fallback with a handler name
    Custom {
        /// Name of the custom handler
        handler: String,
    },
}

/// State of a component in the degradation system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentState {
    /// Component name
    pub name: String,
    /// Current health status
    pub status: HealthStatus,
    /// Active fallback strategy (if any)
    pub fallback: Option<FallbackStrategy>,
    /// Number of consecutive failures
    pub failure_count: u32,
    /// When the component last changed status
    pub last_status_change: DateTime<Utc>,
    /// Whether the component is in a circuit breaker "open" state
    pub circuit_open: bool,
    /// When the circuit breaker will attempt to close
    pub circuit_reset_at: Option<DateTime<Utc>>,
}

impl ComponentState {
    /// Create a new healthy component state
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Healthy,
            fallback: None,
            failure_count: 0,
            last_status_change: Utc::now(),
            circuit_open: false,
            circuit_reset_at: None,
        }
    }

    /// Record a failure
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
    }

    /// Reset failure count
    pub fn reset_failures(&mut self) {
        self.failure_count = 0;
    }

    /// Check if the component is operational
    pub fn is_operational(&self) -> bool {
        !self.circuit_open && self.status.is_operational()
    }
}

/// Configuration for the degradation manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationConfig {
    /// Enable automatic degradation management
    pub enabled: bool,
    /// Number of failures before triggering circuit breaker
    pub circuit_breaker_threshold: u32,
    /// Duration to keep circuit open
    pub circuit_breaker_duration: Duration,
    /// Whether to auto-recover when health is restored
    pub auto_recover: bool,
    /// Minimum time a component must be healthy before recovery
    pub recovery_threshold: Duration,
}

impl Default for DegradationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            circuit_breaker_threshold: 3,
            circuit_breaker_duration: Duration::from_secs(30),
            auto_recover: true,
            recovery_threshold: Duration::from_secs(10),
        }
    }
}

/// Event emitted by the degradation manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DegradationEvent {
    /// Component entered degraded mode
    Degraded {
        component: String,
        fallback: FallbackStrategy,
        reason: String,
    },
    /// Component recovered
    Recovered {
        component: String,
        downtime: Duration,
    },
    /// Circuit breaker opened
    CircuitOpened {
        component: String,
        failure_count: u32,
    },
    /// Circuit breaker closed
    CircuitClosed {
        component: String,
    },
    /// Feature was disabled
    FeatureDisabled {
        feature: String,
        reason: String,
    },
    /// Feature was re-enabled
    FeatureEnabled {
        feature: String,
    },
}

/// Graceful Degradation Manager
///
/// Manages system degradation, feature flags, and fallback strategies.
pub struct DegradationManager {
    config: DegradationConfig,
    registry: Option<Arc<HealthRegistry>>,
    components: Arc<RwLock<HashMap<String, ComponentState>>>,
    features: Arc<RwLock<HashMap<String, FeatureFlag>>>,
    fallbacks: Arc<RwLock<HashMap<String, FallbackStrategy>>>,
    event_handlers: Arc<RwLock<Vec<Box<dyn Fn(DegradationEvent) + Send + Sync>>>>,
    recovery_trackers: Arc<RwLock<HashMap<String, Instant>>>,
}

impl DegradationManager {
    /// Create a new degradation manager
    pub fn new(config: DegradationConfig) -> Self {
        Self {
            config,
            registry: None,
            components: Arc::new(RwLock::new(HashMap::new())),
            features: Arc::new(RwLock::new(HashMap::new())),
            fallbacks: Arc::new(RwLock::new(HashMap::new())),
            event_handlers: Arc::new(RwLock::new(Vec::new())),
            recovery_trackers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(DegradationConfig::default())
    }

    /// Set the health registry for automatic status tracking
    pub fn with_registry(mut self, registry: Arc<HealthRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Register a component with its fallback strategy
    pub async fn register_component(
        &self,
        name: impl Into<String>,
        fallback: FallbackStrategy,
    ) {
        let name = name.into();

        let mut components = self.components.write().await;
        components.insert(name.clone(), ComponentState::new(&name));

        let mut fallbacks = self.fallbacks.write().await;
        fallbacks.insert(name, fallback);
    }

    /// Register a feature flag
    pub async fn register_feature(&self, feature: FeatureFlag) {
        let mut features = self.features.write().await;
        features.insert(feature.name.clone(), feature);
    }

    /// Register an event handler
    pub async fn on_event<F>(&self, handler: F)
    where
        F: Fn(DegradationEvent) + Send + Sync + 'static,
    {
        let mut handlers = self.event_handlers.write().await;
        handlers.push(Box::new(handler));
    }

    /// Handle a health status change
    pub async fn handle_health_change(&self, event: HealthChangeEvent) {
        if !self.config.enabled {
            return;
        }

        let component = &event.component;

        match event.new_status {
            HealthStatus::Unhealthy => {
                self.handle_component_failure(component, &event.message).await;
            }
            HealthStatus::Degraded => {
                self.handle_component_degradation(component, &event.message).await;
            }
            HealthStatus::Healthy if event.is_recovery() => {
                self.handle_component_recovery(component).await;
            }
            _ => {}
        }
    }

    /// Handle component failure
    async fn handle_component_failure(&self, component: &str, reason: &str) {
        let mut components = self.components.write().await;

        let state = components
            .entry(component.to_string())
            .or_insert_with(|| ComponentState::new(component));

        state.status = HealthStatus::Unhealthy;
        state.record_failure();
        state.last_status_change = Utc::now();

        // Check circuit breaker threshold
        if state.failure_count >= self.config.circuit_breaker_threshold && !state.circuit_open {
            state.circuit_open = true;
            state.circuit_reset_at = Some(
                Utc::now() + chrono::Duration::from_std(self.config.circuit_breaker_duration).unwrap_or_default()
            );

            warn!(
                component = %component,
                failures = state.failure_count,
                "Circuit breaker opened"
            );

            self.emit_event(DegradationEvent::CircuitOpened {
                component: component.to_string(),
                failure_count: state.failure_count,
            })
            .await;
        }

        // Apply fallback strategy
        let fallbacks = self.fallbacks.read().await;
        if let Some(fallback) = fallbacks.get(component) {
            state.fallback = Some(fallback.clone());

            info!(
                component = %component,
                fallback = ?fallback,
                "Applying fallback strategy"
            );

            self.emit_event(DegradationEvent::Degraded {
                component: component.to_string(),
                fallback: fallback.clone(),
                reason: reason.to_string(),
            })
            .await;
        }

        // Update dependent features
        drop(components);
        drop(fallbacks);
        self.update_features().await;
    }

    /// Handle component degradation (partial failure)
    async fn handle_component_degradation(&self, component: &str, reason: &str) {
        let mut components = self.components.write().await;

        let state = components
            .entry(component.to_string())
            .or_insert_with(|| ComponentState::new(component));

        state.status = HealthStatus::Degraded;
        state.last_status_change = Utc::now();

        // Apply fallback for degraded state
        let fallbacks = self.fallbacks.read().await;
        if let Some(fallback) = fallbacks.get(component) {
            state.fallback = Some(fallback.clone());

            debug!(
                component = %component,
                "Component degraded, applying fallback"
            );

            self.emit_event(DegradationEvent::Degraded {
                component: component.to_string(),
                fallback: fallback.clone(),
                reason: reason.to_string(),
            })
            .await;
        }
    }

    /// Handle component recovery
    async fn handle_component_recovery(&self, component: &str) {
        if !self.config.auto_recover {
            return;
        }

        // Track recovery time
        {
            let mut trackers = self.recovery_trackers.write().await;
            trackers
                .entry(component.to_string())
                .or_insert_with(Instant::now);
        }

        // Check if component has been healthy long enough
        let recovery_started = {
            let trackers = self.recovery_trackers.read().await;
            trackers.get(component).copied()
        };

        if let Some(started) = recovery_started {
            if started.elapsed() < self.config.recovery_threshold {
                debug!(
                    component = %component,
                    "Component recovering, waiting for threshold"
                );
                return;
            }
        }

        // Recover the component
        let mut components = self.components.write().await;

        if let Some(state) = components.get_mut(component) {
            let downtime = Utc::now()
                .signed_duration_since(state.last_status_change)
                .to_std()
                .unwrap_or_default();

            state.status = HealthStatus::Healthy;
            state.fallback = None;
            state.reset_failures();
            state.last_status_change = Utc::now();

            let was_circuit_open = state.circuit_open;
            state.circuit_open = false;
            state.circuit_reset_at = None;

            info!(
                component = %component,
                downtime = ?downtime,
                "Component recovered"
            );

            self.emit_event(DegradationEvent::Recovered {
                component: component.to_string(),
                downtime,
            })
            .await;

            if was_circuit_open {
                self.emit_event(DegradationEvent::CircuitClosed {
                    component: component.to_string(),
                })
                .await;
            }
        }

        // Clear recovery tracker
        {
            let mut trackers = self.recovery_trackers.write().await;
            trackers.remove(component);
        }

        // Update dependent features
        drop(components);
        self.update_features().await;
    }

    /// Update feature flags based on component health
    async fn update_features(&self) {
        let components = self.components.read().await;
        let healthy_components: Vec<String> = components
            .iter()
            .filter(|(_, state)| state.is_operational())
            .map(|(name, _)| name.clone())
            .collect();
        drop(components);

        let mut features = self.features.write().await;

        for (name, feature) in features.iter_mut() {
            let was_enabled = feature.enabled;
            let should_enable = feature.check_dependencies(&healthy_components);

            if was_enabled && !should_enable {
                feature.enabled = false;
                info!(feature = %name, "Feature disabled due to unhealthy dependencies");

                // Emit event (cannot borrow self mutably here, so we collect events)
            } else if !was_enabled && should_enable {
                feature.enabled = true;
                info!(feature = %name, "Feature re-enabled");
            }
        }
    }

    /// Manually enable a feature
    pub async fn enable_feature(&self, name: &str) -> bool {
        let mut features = self.features.write().await;

        if let Some(feature) = features.get_mut(name) {
            if feature.can_override {
                feature.enabled = true;
                info!(feature = %name, "Feature manually enabled");

                self.emit_event(DegradationEvent::FeatureEnabled {
                    feature: name.to_string(),
                })
                .await;

                return true;
            }
        }
        false
    }

    /// Manually disable a feature
    pub async fn disable_feature(&self, name: &str, reason: &str) -> bool {
        let mut features = self.features.write().await;

        if let Some(feature) = features.get_mut(name) {
            if feature.can_override {
                feature.enabled = false;
                info!(feature = %name, reason = %reason, "Feature manually disabled");

                self.emit_event(DegradationEvent::FeatureDisabled {
                    feature: name.to_string(),
                    reason: reason.to_string(),
                })
                .await;

                return true;
            }
        }
        false
    }

    /// Check if a feature is enabled
    pub async fn is_feature_enabled(&self, name: &str) -> bool {
        let features = self.features.read().await;
        features.get(name).map(|f| f.enabled).unwrap_or(false)
    }

    /// Get the current fallback strategy for a component
    pub async fn get_fallback(&self, component: &str) -> Option<FallbackStrategy> {
        let components = self.components.read().await;
        components.get(component).and_then(|s| s.fallback.clone())
    }

    /// Check if a component's circuit breaker is open
    pub async fn is_circuit_open(&self, component: &str) -> bool {
        let components = self.components.read().await;
        components
            .get(component)
            .map(|s| s.circuit_open)
            .unwrap_or(false)
    }

    /// Get the state of all components
    pub async fn component_states(&self) -> HashMap<String, ComponentState> {
        self.components.read().await.clone()
    }

    /// Get all feature flags
    pub async fn feature_flags(&self) -> HashMap<String, FeatureFlag> {
        self.features.read().await.clone()
    }

    /// Get a summary of the degradation state
    pub async fn summary(&self) -> DegradationSummary {
        let components = self.components.read().await;
        let features = self.features.read().await;

        let total_components = components.len();
        let healthy_count = components
            .values()
            .filter(|s| s.status == HealthStatus::Healthy)
            .count();
        let degraded_count = components
            .values()
            .filter(|s| s.status == HealthStatus::Degraded)
            .count();
        let unhealthy_count = components
            .values()
            .filter(|s| s.status == HealthStatus::Unhealthy)
            .count();
        let circuits_open = components.values().filter(|s| s.circuit_open).count();

        let total_features = features.len();
        let enabled_features = features.values().filter(|f| f.enabled).count();

        DegradationSummary {
            total_components,
            healthy_count,
            degraded_count,
            unhealthy_count,
            circuits_open,
            total_features,
            enabled_features,
            is_degraded: degraded_count > 0 || unhealthy_count > 0,
        }
    }

    /// Emit a degradation event
    async fn emit_event(&self, event: DegradationEvent) {
        let handlers = self.event_handlers.read().await;
        for handler in handlers.iter() {
            handler(event.clone());
        }
    }

    /// Attempt to reset a circuit breaker
    pub async fn reset_circuit(&self, component: &str) -> bool {
        let mut components = self.components.write().await;

        if let Some(state) = components.get_mut(component) {
            if state.circuit_open {
                state.circuit_open = false;
                state.circuit_reset_at = None;
                state.reset_failures();

                info!(component = %component, "Circuit breaker manually reset");

                self.emit_event(DegradationEvent::CircuitClosed {
                    component: component.to_string(),
                })
                .await;

                return true;
            }
        }
        false
    }
}

/// Summary of degradation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationSummary {
    pub total_components: usize,
    pub healthy_count: usize,
    pub degraded_count: usize,
    pub unhealthy_count: usize,
    pub circuits_open: usize,
    pub total_features: usize,
    pub enabled_features: usize,
    pub is_degraded: bool,
}

impl DegradationSummary {
    /// Format as human-readable text
    pub fn to_text(&self) -> String {
        format!(
            "Degradation Status:\n\
             Components: {}/{} healthy, {} degraded, {} unhealthy\n\
             Circuit breakers: {} open\n\
             Features: {}/{} enabled\n\
             System state: {}",
            self.healthy_count,
            self.total_components,
            self.degraded_count,
            self.unhealthy_count,
            self.circuits_open,
            self.enabled_features,
            self.total_features,
            if self.is_degraded { "DEGRADED" } else { "HEALTHY" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_flag_creation() {
        let feature = FeatureFlag::new("sync", "Cloud synchronization")
            .depends_on("postgres")
            .depends_on("network")
            .can_override(false);

        assert_eq!(feature.name, "sync");
        assert_eq!(feature.dependencies.len(), 2);
        assert!(!feature.can_override);
    }

    #[test]
    fn test_feature_dependency_check() {
        let feature = FeatureFlag::new("sync", "Cloud sync")
            .depends_on("postgres")
            .depends_on("network");

        let healthy = vec!["postgres".to_string(), "network".to_string(), "disk".to_string()];
        assert!(feature.check_dependencies(&healthy));

        let partial = vec!["postgres".to_string()];
        assert!(!feature.check_dependencies(&partial));
    }

    #[tokio::test]
    async fn test_component_registration() {
        let manager = DegradationManager::with_defaults();

        manager
            .register_component("database", FallbackStrategy::UseCache {
                max_age: Duration::from_secs(300),
            })
            .await;

        let states = manager.component_states().await;
        assert!(states.contains_key("database"));
    }

    #[tokio::test]
    async fn test_feature_registration() {
        let manager = DegradationManager::with_defaults();

        let feature = FeatureFlag::new("cloud_sync", "Cloud synchronization")
            .depends_on("postgres");

        manager.register_feature(feature).await;

        assert!(manager.is_feature_enabled("cloud_sync").await);
    }

    #[tokio::test]
    async fn test_manual_feature_control() {
        let manager = DegradationManager::with_defaults();

        let feature = FeatureFlag::new("beta", "Beta features");
        manager.register_feature(feature).await;

        assert!(manager.disable_feature("beta", "Testing").await);
        assert!(!manager.is_feature_enabled("beta").await);

        assert!(manager.enable_feature("beta").await);
        assert!(manager.is_feature_enabled("beta").await);
    }

    #[tokio::test]
    async fn test_health_change_handling() {
        let manager = DegradationManager::with_defaults();

        manager
            .register_component("database", FallbackStrategy::UseCache {
                max_age: Duration::from_secs(300),
            })
            .await;

        let event = HealthChangeEvent::new(
            "database",
            HealthStatus::Healthy,
            HealthStatus::Unhealthy,
            "Connection lost",
        );

        manager.handle_health_change(event).await;

        let states = manager.component_states().await;
        let state = states.get("database").unwrap();
        assert_eq!(state.status, HealthStatus::Unhealthy);
        assert!(state.fallback.is_some());
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let config = DegradationConfig {
            circuit_breaker_threshold: 2,
            ..Default::default()
        };
        let manager = DegradationManager::new(config);

        manager
            .register_component("api", FallbackStrategy::Disable)
            .await;

        // First failure
        let event1 = HealthChangeEvent::new(
            "api",
            HealthStatus::Healthy,
            HealthStatus::Unhealthy,
            "Error 1",
        );
        manager.handle_health_change(event1).await;
        assert!(!manager.is_circuit_open("api").await);

        // Second failure - should trigger circuit breaker
        let event2 = HealthChangeEvent::new(
            "api",
            HealthStatus::Unhealthy,
            HealthStatus::Unhealthy,
            "Error 2",
        );
        manager.handle_health_change(event2).await;
        assert!(manager.is_circuit_open("api").await);
    }

    #[tokio::test]
    async fn test_summary() {
        let manager = DegradationManager::with_defaults();

        manager
            .register_component("db", FallbackStrategy::Disable)
            .await;

        let feature = FeatureFlag::new("feature1", "Test feature");
        manager.register_feature(feature).await;

        let summary = manager.summary().await;
        assert_eq!(summary.total_components, 1);
        assert_eq!(summary.total_features, 1);
        assert!(!summary.is_degraded);
    }

    #[test]
    fn test_fallback_strategy_serialization() {
        let strategy = FallbackStrategy::UseCache {
            max_age: Duration::from_secs(300),
        };

        let json = serde_json::to_string(&strategy).unwrap();
        let parsed: FallbackStrategy = serde_json::from_str(&json).unwrap();

        match parsed {
            FallbackStrategy::UseCache { max_age } => {
                assert_eq!(max_age, Duration::from_secs(300));
            }
            _ => panic!("Wrong variant"),
        }
    }
}
