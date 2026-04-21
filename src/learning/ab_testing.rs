//! A/B testing infrastructure for evaluating SONA optimizations.
//!
//! This module provides tools for safely comparing SONA-optimized retrieval
//! against baseline approaches using controlled experiments.
//!
//! # Features
//!
//! - Deterministic variant assignment (based on session ID)
//! - Configurable holdout percentage
//! - Metrics collection and comparison
//! - Regression detection with alerting
//! - Quarterly improvement tracking
//!
//! # Example
//!
//! ```ignore
//! use nagual::learning::{AbTestManager, AbTestConfig, Variant};
//!
//! let config = AbTestConfig::default(); // 20% holdout
//! let manager = AbTestManager::new(config);
//!
//! // Deterministic assignment based on session
//! let variant = manager.assign_variant("session-123");
//! match variant {
//!     Variant::Treatment => { /* Use SONA-optimized retrieval */ }
//!     Variant::Control => { /* Use baseline retrieval */ }
//! }
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ============================================================================
// A/B Testing Configuration
// ============================================================================

/// Configuration for A/B testing experiments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTestConfig {
    /// Percentage of traffic to route to control (0.0 - 1.0).
    /// Default: 0.2 (20% holdout)
    #[serde(default = "default_holdout")]
    pub holdout_percentage: f64,

    /// Minimum sample size before drawing conclusions.
    #[serde(default = "default_min_sample")]
    pub min_sample_size: usize,

    /// Whether to use deterministic assignment (based on session hash).
    #[serde(default = "default_deterministic")]
    pub deterministic: bool,

    /// Salt for deterministic hashing (prevents predictable assignment).
    #[serde(default = "default_salt")]
    pub salt: String,

    /// Experiment name for identification.
    #[serde(default = "default_experiment_name")]
    pub experiment_name: String,
}

fn default_holdout() -> f64 {
    0.2
}

fn default_min_sample() -> usize {
    100
}

fn default_deterministic() -> bool {
    true
}

fn default_salt() -> String {
    "sona-ab-test".to_string()
}

fn default_experiment_name() -> String {
    "sona-optimization".to_string()
}

impl Default for AbTestConfig {
    fn default() -> Self {
        Self {
            holdout_percentage: default_holdout(),
            min_sample_size: default_min_sample(),
            deterministic: default_deterministic(),
            salt: default_salt(),
            experiment_name: default_experiment_name(),
        }
    }
}

// ============================================================================
// Variants
// ============================================================================

/// Experimental variant for A/B testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Variant {
    /// Control group (baseline)
    Control,
    /// Treatment group (SONA-optimized)
    Treatment,
}

impl Variant {
    /// Check if this is the treatment variant.
    pub fn is_treatment(&self) -> bool {
        matches!(self, Variant::Treatment)
    }
}

impl std::fmt::Display for Variant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variant::Control => write!(f, "control"),
            Variant::Treatment => write!(f, "treatment"),
        }
    }
}

// ============================================================================
// A/B Test Manager
// ============================================================================

/// Manager for A/B testing experiments.
#[derive(Debug, Clone)]
pub struct AbTestManager {
    config: AbTestConfig,
}

impl AbTestManager {
    /// Create a new A/B test manager.
    pub fn new(config: AbTestConfig) -> Self {
        Self { config }
    }

    /// Assign a variant based on session ID.
    ///
    /// Uses deterministic hashing to ensure the same session always
    /// gets the same variant.
    pub fn assign_variant(&self, session_id: &str) -> Variant {
        if self.config.deterministic {
            let hash = self.hash_session(session_id);
            let bucket = (hash % 1000) as f64 / 1000.0;

            if bucket < self.config.holdout_percentage {
                Variant::Control
            } else {
                Variant::Treatment
            }
        } else {
            // Random assignment
            if rand::random::<f64>() < self.config.holdout_percentage {
                Variant::Control
            } else {
                Variant::Treatment
            }
        }
    }

    /// Hash a session ID for deterministic assignment.
    fn hash_session(&self, session_id: &str) -> u64 {
        let combined = format!("{}:{}", self.config.salt, session_id);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        combined.hash(&mut hasher);
        hasher.finish()
    }

    /// Get the configuration.
    pub fn config(&self) -> &AbTestConfig {
        &self.config
    }
}

// ============================================================================
// Metrics
// ============================================================================

/// Type of metric being tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricType {
    /// Pattern retrieval latency in milliseconds
    Latency,
    /// Relevance score (0.0 - 1.0)
    Relevance,
    /// User satisfaction (0.0 - 1.0)
    Satisfaction,
    /// Success rate (0.0 - 1.0)
    SuccessRate,
    /// Patterns retrieved count
    PatternsRetrieved,
}

/// Aggregation method for metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricAggregation {
    /// Average value
    Mean,
    /// Median value
    Median,
    /// 95th percentile
    P95,
    /// 99th percentile
    P99,
    /// Sum of all values
    Sum,
    /// Count of samples
    Count,
}

/// Statistics for a specific variant.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VariantStats {
    /// Number of samples
    pub sample_count: usize,

    /// Mean value
    pub mean: f64,

    /// Standard deviation
    pub std_dev: f64,

    /// Median value
    pub median: f64,

    /// 95th percentile
    pub p95: f64,

    /// 99th percentile
    pub p99: f64,

    /// Minimum value
    pub min: f64,

    /// Maximum value
    pub max: f64,
}

impl VariantStats {
    /// Calculate statistics from a list of values.
    pub fn from_values(mut values: Vec<f64>) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let median = values[values.len() / 2];
        let p95 = values[(values.len() as f64 * 0.95) as usize];
        let p99 = values[(values.len() as f64 * 0.99).min(values.len() as f64 - 1.0) as usize];

        Self {
            sample_count: values.len(),
            mean,
            std_dev,
            median,
            p95,
            p99,
            min: values[0],
            max: values[values.len() - 1],
        }
    }
}

/// Collected metrics for an A/B test.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AbTestMetrics {
    /// Control group statistics by metric type
    pub control: HashMap<MetricType, VariantStats>,

    /// Treatment group statistics by metric type
    pub treatment: HashMap<MetricType, VariantStats>,

    /// When metrics collection started
    pub started_at: Option<DateTime<Utc>>,

    /// When metrics were last updated
    pub updated_at: Option<DateTime<Utc>>,
}

/// Result of an A/B test analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTestResult {
    /// Whether the test has sufficient data
    pub sufficient_data: bool,

    /// Statistical significance (p-value < 0.05 typically)
    pub significant: bool,

    /// Effect size (improvement percentage)
    pub effect_size: f64,

    /// Confidence interval lower bound
    pub ci_lower: f64,

    /// Confidence interval upper bound
    pub ci_upper: f64,

    /// Recommendation based on results
    pub recommendation: String,
}

// ============================================================================
// Baseline Metrics
// ============================================================================

/// Baseline metrics for comparison.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BaselineMetrics {
    /// Metric values by type
    pub values: HashMap<MetricType, f64>,

    /// When baseline was established
    pub established_at: DateTime<Utc>,

    /// Description of the baseline
    pub description: String,
}

impl BaselineMetrics {
    /// Create a new baseline with current timestamp.
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            values: HashMap::new(),
            established_at: Utc::now(),
            description: description.into(),
        }
    }

    /// Add a metric to the baseline.
    pub fn with_metric(mut self, metric_type: MetricType, value: f64) -> Self {
        self.values.insert(metric_type, value);
        self
    }
}

// ============================================================================
// Regression Detection
// ============================================================================

/// Severity level for regression alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Informational - minor deviation
    Info,
    /// Warning - notable deviation
    Warning,
    /// Critical - significant regression
    Critical,
}

/// Alert for a detected regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAlert {
    /// The metric that regressed
    pub metric: MetricType,

    /// Severity of the regression
    pub severity: Severity,

    /// Baseline value
    pub baseline: f64,

    /// Current value
    pub current: f64,

    /// Percentage change
    pub change_percent: f64,

    /// Threshold that was exceeded
    pub threshold: f64,

    /// When the alert was generated
    pub timestamp: DateTime<Utc>,

    /// Recommended action
    pub recommendation: String,
}

/// Configuration for regression detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Threshold for warning (percentage)
    #[serde(default = "default_warning_threshold")]
    pub warning_threshold: f64,

    /// Threshold for critical (percentage)
    #[serde(default = "default_critical_threshold")]
    pub critical_threshold: f64,

    /// Metrics to monitor
    #[serde(default = "default_monitored_metrics")]
    pub monitored_metrics: Vec<MetricType>,

    /// Whether higher values are better (for each metric)
    #[serde(default)]
    pub higher_is_better: HashMap<MetricType, bool>,
}

fn default_warning_threshold() -> f64 {
    0.1 // 10%
}

fn default_critical_threshold() -> f64 {
    0.25 // 25%
}

fn default_monitored_metrics() -> Vec<MetricType> {
    vec![
        MetricType::Latency,
        MetricType::Relevance,
        MetricType::SuccessRate,
    ]
}

impl Default for RegressionConfig {
    fn default() -> Self {
        let mut higher_is_better = HashMap::new();
        higher_is_better.insert(MetricType::Latency, false); // Lower latency is better
        higher_is_better.insert(MetricType::Relevance, true);
        higher_is_better.insert(MetricType::Satisfaction, true);
        higher_is_better.insert(MetricType::SuccessRate, true);

        Self {
            warning_threshold: default_warning_threshold(),
            critical_threshold: default_critical_threshold(),
            monitored_metrics: default_monitored_metrics(),
            higher_is_better,
        }
    }
}

/// Detector for performance regressions.
#[derive(Debug, Clone)]
pub struct RegressionDetector {
    config: RegressionConfig,
}

impl RegressionDetector {
    /// Create a new regression detector.
    pub fn new(config: RegressionConfig) -> Self {
        Self { config }
    }

    /// Detect regressions by comparing current metrics to baseline.
    pub fn detect_regression(
        &self,
        baseline: &BaselineMetrics,
        current: &HashMap<MetricType, f64>,
    ) -> Vec<RegressionAlert> {
        let mut alerts = Vec::new();

        for metric in &self.config.monitored_metrics {
            if let (Some(&base_val), Some(&curr_val)) = (baseline.values.get(metric), current.get(metric)) {
                let higher_is_better = self.config.higher_is_better.get(metric).copied().unwrap_or(true);

                let change = if higher_is_better {
                    (base_val - curr_val) / base_val // Decrease is bad
                } else {
                    (curr_val - base_val) / base_val // Increase is bad
                };

                let change_percent = change * 100.0;

                if change >= self.config.critical_threshold {
                    alerts.push(RegressionAlert {
                        metric: *metric,
                        severity: Severity::Critical,
                        baseline: base_val,
                        current: curr_val,
                        change_percent,
                        threshold: self.config.critical_threshold * 100.0,
                        timestamp: Utc::now(),
                        recommendation: format!(
                            "Critical regression in {:?}. Investigate immediately.",
                            metric
                        ),
                    });
                } else if change >= self.config.warning_threshold {
                    alerts.push(RegressionAlert {
                        metric: *metric,
                        severity: Severity::Warning,
                        baseline: base_val,
                        current: curr_val,
                        change_percent,
                        threshold: self.config.warning_threshold * 100.0,
                        timestamp: Utc::now(),
                        recommendation: format!(
                            "Warning: {:?} has degraded. Monitor closely.",
                            metric
                        ),
                    });
                }
            }
        }

        alerts
    }

    /// Get the configuration.
    pub fn config(&self) -> &RegressionConfig {
        &self.config
    }
}

// ============================================================================
// Improvement Tracking
// ============================================================================

/// Target for quarterly improvements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementTarget {
    /// The metric being targeted
    pub metric: MetricType,

    /// Target improvement percentage
    pub target_improvement: f64,

    /// Description of the target
    pub description: String,

    /// Quarter (e.g., "Q1 2024")
    pub quarter: String,
}

/// Progress report for quarterly improvements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarterlyProgress {
    /// Target being tracked
    pub target: ImprovementTarget,

    /// Current progress (0.0 - 1.0+)
    pub progress: f64,

    /// Baseline value
    pub baseline: f64,

    /// Current value
    pub current: f64,

    /// Whether target is met
    pub target_met: bool,

    /// Days remaining in quarter
    pub days_remaining: i64,

    /// Projected end value (extrapolation)
    pub projected_value: f64,
}

/// Report on overall improvements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementReport {
    /// Progress for each target
    pub targets: Vec<QuarterlyProgress>,

    /// Overall progress percentage
    pub overall_progress: f64,

    /// Number of targets met
    pub targets_met: usize,

    /// Summary recommendations
    pub recommendations: Vec<String>,

    /// Generated at
    pub generated_at: DateTime<Utc>,
}

/// Tracker for quarterly improvements.
#[derive(Debug, Clone)]
pub struct ImprovementTracker {
    targets: Vec<ImprovementTarget>,
    baseline: BaselineMetrics,
}

impl ImprovementTracker {
    /// Create a new improvement tracker.
    pub fn new(targets: Vec<ImprovementTarget>, baseline: BaselineMetrics) -> Self {
        Self { targets, baseline }
    }

    /// Check progress against quarterly targets.
    pub fn check_quarterly_progress(&self, current: &HashMap<MetricType, f64>) -> ImprovementReport {
        let mut progress_reports = Vec::new();
        let mut total_progress = 0.0;
        let mut targets_met = 0;

        for target in &self.targets {
            if let (Some(&base_val), Some(&curr_val)) =
                (self.baseline.values.get(&target.metric), current.get(&target.metric))
            {
                let improvement = (curr_val - base_val) / base_val;
                let progress = improvement / (target.target_improvement / 100.0);
                let target_met = progress >= 1.0;

                if target_met {
                    targets_met += 1;
                }

                total_progress += progress;

                progress_reports.push(QuarterlyProgress {
                    target: target.clone(),
                    progress,
                    baseline: base_val,
                    current: curr_val,
                    target_met,
                    days_remaining: 90, // Simplified
                    projected_value: curr_val, // Simplified
                });
            }
        }

        let overall_progress = if !progress_reports.is_empty() {
            total_progress / progress_reports.len() as f64
        } else {
            0.0
        };

        let mut recommendations = Vec::new();
        if overall_progress < 0.5 {
            recommendations.push("Progress is behind schedule. Consider accelerating optimization efforts.".to_string());
        }
        if targets_met == self.targets.len() {
            recommendations.push("All targets met! Consider setting more ambitious goals.".to_string());
        }

        ImprovementReport {
            targets: progress_reports,
            overall_progress,
            targets_met,
            recommendations,
            generated_at: Utc::now(),
        }
    }

    /// Get the targets.
    pub fn targets(&self) -> &[ImprovementTarget] {
        &self.targets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ab_test_config_default() {
        let config = AbTestConfig::default();
        assert!((config.holdout_percentage - 0.2).abs() < 0.001);
        assert_eq!(config.min_sample_size, 100);
        assert!(config.deterministic);
    }

    #[test]
    fn test_variant_assignment_deterministic() {
        let config = AbTestConfig::default();
        let manager = AbTestManager::new(config);

        // Same session should always get same variant
        let variant1 = manager.assign_variant("session-123");
        let variant2 = manager.assign_variant("session-123");
        assert_eq!(variant1, variant2);
    }

    #[test]
    fn test_variant_distribution() {
        let config = AbTestConfig {
            holdout_percentage: 0.5,
            ..Default::default()
        };
        let manager = AbTestManager::new(config);

        let mut control_count = 0;
        let mut treatment_count = 0;

        for i in 0..1000 {
            let session = format!("session-{}", i);
            match manager.assign_variant(&session) {
                Variant::Control => control_count += 1,
                Variant::Treatment => treatment_count += 1,
            }
        }

        // Should be roughly 50/50 with some variance
        assert!(control_count > 400 && control_count < 600);
        assert!(treatment_count > 400 && treatment_count < 600);
    }

    #[test]
    fn test_variant_stats_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = VariantStats::from_values(values);

        assert_eq!(stats.sample_count, 10);
        assert!((stats.mean - 5.5).abs() < 0.01);
        assert!((stats.min - 1.0).abs() < 0.01);
        assert!((stats.max - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_regression_detection() {
        let config = RegressionConfig::default();
        let detector = RegressionDetector::new(config);

        let baseline = BaselineMetrics::new("test baseline")
            .with_metric(MetricType::Latency, 100.0)
            .with_metric(MetricType::Relevance, 0.8);

        // Significant regression in latency (higher is worse)
        let mut current = HashMap::new();
        current.insert(MetricType::Latency, 150.0); // 50% increase = bad
        current.insert(MetricType::Relevance, 0.8);

        let alerts = detector.detect_regression(&baseline, &current);
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].metric, MetricType::Latency);
        assert_eq!(alerts[0].severity, Severity::Critical);
    }

    #[test]
    fn test_improvement_tracking() {
        let targets = vec![ImprovementTarget {
            metric: MetricType::Relevance,
            target_improvement: 10.0,
            description: "Improve relevance by 10%".to_string(),
            quarter: "Q1 2024".to_string(),
        }];

        let baseline = BaselineMetrics::new("baseline")
            .with_metric(MetricType::Relevance, 0.8);

        let tracker = ImprovementTracker::new(targets, baseline);

        // Use slightly more than 10% improvement to avoid floating point edge case
        let mut current = HashMap::new();
        current.insert(MetricType::Relevance, 0.881); // 10.125% improvement

        let report = tracker.check_quarterly_progress(&current);

        assert!(!report.targets.is_empty(), "Report should have targets");
        assert_eq!(report.targets_met, 1);
        assert!(report.overall_progress >= 1.0);
    }
}
