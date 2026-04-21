//! Timeline Estimator - Estimates when predictions may resolve.
//!
//! This module provides timeline estimation based on historical pattern
//! resolution times using percentile-based analysis.

use serde::{Deserialize, Serialize};

use super::PredictionResult;

/// Configuration for timeline estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineConfig {
    /// Percentile for minimum timeline (e.g., 10 = 10th percentile)
    pub min_percentile: f64,
    /// Percentile for maximum timeline (e.g., 90 = 90th percentile)
    pub max_percentile: f64,
    /// Default minimum days when no data available
    pub default_min_days: u32,
    /// Default maximum days when no data available
    pub default_max_days: u32,
    /// Minimum allowed timeline in days
    pub floor_min_days: u32,
    /// Maximum allowed timeline in days
    pub ceiling_max_days: u32,
    /// Whether to apply outlier removal
    pub remove_outliers: bool,
    /// Multiplier for IQR in outlier detection
    pub outlier_iqr_multiplier: f64,
}

impl Default for TimelineConfig {
    fn default() -> Self {
        Self {
            min_percentile: 10.0,
            max_percentile: 90.0,
            default_min_days: 1,
            default_max_days: 30,
            floor_min_days: 1,
            ceiling_max_days: 365,
            remove_outliers: true,
            outlier_iqr_multiplier: 1.5,
        }
    }
}

impl TimelineConfig {
    /// Create a new timeline configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the min/max percentiles.
    pub fn with_percentiles(mut self, min: f64, max: f64) -> Self {
        self.min_percentile = min.clamp(0.0, 100.0);
        self.max_percentile = max.clamp(0.0, 100.0);
        self
    }

    /// Set default values when no data available.
    pub fn with_defaults(mut self, min_days: u32, max_days: u32) -> Self {
        self.default_min_days = min_days;
        self.default_max_days = max_days;
        self
    }

    /// Set floor and ceiling constraints.
    pub fn with_constraints(mut self, floor: u32, ceiling: u32) -> Self {
        self.floor_min_days = floor;
        self.ceiling_max_days = ceiling;
        self
    }

    /// Enable or disable outlier removal.
    pub fn with_outlier_removal(mut self, remove: bool) -> Self {
        self.remove_outliers = remove;
        self
    }
}

/// Result of timeline estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEstimate {
    /// Minimum expected days until resolution
    pub min_days: u32,
    /// Maximum expected days until resolution
    pub max_days: u32,
    /// Median expected days (50th percentile)
    pub median_days: u32,
    /// Mean expected days
    pub mean_days: f64,
    /// Standard deviation of resolution times
    pub std_dev_days: f64,
    /// Number of data points used
    pub data_points: usize,
    /// Number of outliers removed
    pub outliers_removed: usize,
    /// Analysis details
    pub analysis: ResolutionTimeAnalysis,
}

impl Default for TimelineEstimate {
    fn default() -> Self {
        Self {
            min_days: 1,
            max_days: 30,
            median_days: 15,
            mean_days: 15.0,
            std_dev_days: 10.0,
            data_points: 0,
            outliers_removed: 0,
            analysis: ResolutionTimeAnalysis::default(),
        }
    }
}

/// Detailed analysis of resolution times.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResolutionTimeAnalysis {
    /// Minimum observed resolution time
    pub observed_min: Option<u32>,
    /// Maximum observed resolution time
    pub observed_max: Option<u32>,
    /// 25th percentile (Q1)
    pub q1: Option<u32>,
    /// 75th percentile (Q3)
    pub q3: Option<u32>,
    /// Interquartile range (Q3 - Q1)
    pub iqr: Option<u32>,
    /// Lower fence for outlier detection
    pub lower_fence: Option<f64>,
    /// Upper fence for outlier detection
    pub upper_fence: Option<f64>,
    /// Confidence in the estimate (0.0-1.0)
    pub confidence: f64,
}

/// Timeline estimator that calculates expected resolution times.
#[derive(Debug, Clone)]
pub struct TimelineEstimator {
    /// Configuration
    pub config: TimelineConfig,
}

impl TimelineEstimator {
    /// Create a new timeline estimator with default configuration.
    pub fn new() -> Self {
        Self {
            config: TimelineConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: TimelineConfig) -> Self {
        Self { config }
    }

    /// Estimate timeline from resolution times.
    pub fn estimate(&self, resolution_times: &[u32]) -> PredictionResult<TimelineEstimate> {
        estimate_timeline(resolution_times, &self.config)
    }
}

impl Default for TimelineEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate timeline from historical resolution times using percentile-based analysis.
///
/// The calculation:
/// 1. Optionally removes outliers using IQR method
/// 2. Calculates percentiles for min/max bounds
/// 3. Computes statistics (mean, median, std dev)
/// 4. Applies floor/ceiling constraints
///
/// # Arguments
///
/// * `resolution_times` - Historical resolution times in days
/// * `config` - Configuration for the estimation
///
/// # Returns
///
/// A `TimelineEstimate` with min/max bounds and statistics.
pub fn estimate_timeline(
    resolution_times: &[u32],
    config: &TimelineConfig,
) -> PredictionResult<TimelineEstimate> {
    // Handle empty input
    if resolution_times.is_empty() {
        return Ok(TimelineEstimate {
            min_days: config.default_min_days,
            max_days: config.default_max_days,
            median_days: (config.default_min_days + config.default_max_days) / 2,
            mean_days: (config.default_min_days + config.default_max_days) as f64 / 2.0,
            std_dev_days: 0.0,
            data_points: 0,
            outliers_removed: 0,
            analysis: ResolutionTimeAnalysis {
                confidence: 0.0,
                ..Default::default()
            },
        });
    }

    // Convert to f64 for calculations
    let mut times: Vec<f64> = resolution_times.iter().map(|&t| t as f64).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate quartiles for outlier detection
    let q1 = percentile(&times, 25.0);
    let q3 = percentile(&times, 75.0);
    let iqr = q3 - q1;

    // Calculate fences for outlier detection
    let lower_fence = q1 - config.outlier_iqr_multiplier * iqr;
    let upper_fence = q3 + config.outlier_iqr_multiplier * iqr;

    // Remove outliers if configured
    let (filtered_times, outliers_removed) = if config.remove_outliers && times.len() > 3 {
        let filtered: Vec<f64> = times
            .iter()
            .filter(|&&t| t >= lower_fence && t <= upper_fence)
            .copied()
            .collect();

        // If filtering removed too many points, keep original
        if filtered.len() >= 2 {
            let removed = times.len() - filtered.len();
            (filtered, removed)
        } else {
            (times.clone(), 0)
        }
    } else {
        (times.clone(), 0)
    };

    // Calculate statistics on filtered data
    let n = filtered_times.len();
    let mean = filtered_times.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        filtered_times
            .iter()
            .map(|&t| (t - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    // Calculate percentiles for timeline bounds
    let min_days_raw = percentile(&filtered_times, config.min_percentile);
    let max_days_raw = percentile(&filtered_times, config.max_percentile);
    let median = percentile(&filtered_times, 50.0);

    // Apply constraints
    let min_days = (min_days_raw.round() as u32)
        .max(config.floor_min_days)
        .min(config.ceiling_max_days);

    let max_days = (max_days_raw.round() as u32)
        .max(min_days) // Ensure max >= min
        .min(config.ceiling_max_days);

    let median_days = (median.round() as u32).clamp(min_days, max_days);

    // Calculate confidence based on data quantity and quality
    let confidence = calculate_timeline_confidence(n, std_dev, mean);

    // Build analysis
    let analysis = ResolutionTimeAnalysis {
        observed_min: filtered_times.first().map(|&t| t.round() as u32),
        observed_max: filtered_times.last().map(|&t| t.round() as u32),
        q1: Some(q1.round() as u32),
        q3: Some(q3.round() as u32),
        iqr: Some(iqr.round() as u32),
        lower_fence: Some(lower_fence),
        upper_fence: Some(upper_fence),
        confidence,
    };

    Ok(TimelineEstimate {
        min_days,
        max_days,
        median_days,
        mean_days: mean,
        std_dev_days: std_dev,
        data_points: n,
        outliers_removed,
        analysis,
    })
}

/// Calculate a specific percentile from sorted data.
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }

    if sorted_data.len() == 1 {
        return sorted_data[0];
    }

    let p = p.clamp(0.0, 100.0) / 100.0;
    let n = sorted_data.len();
    let index = p * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted_data[lower]
    } else {
        let fraction = index - lower as f64;
        sorted_data[lower] * (1.0 - fraction) + sorted_data[upper] * fraction
    }
}

/// Calculate confidence in the timeline estimate.
fn calculate_timeline_confidence(n: usize, std_dev: f64, mean: f64) -> f64 {
    // Base confidence from sample size (more data = more confidence)
    let size_confidence = (n as f64 / 20.0).min(1.0);

    // Coefficient of variation penalty (high variability = less confidence)
    let cv = if mean > 0.0 { std_dev / mean } else { 1.0 };
    let cv_confidence = (1.0 - cv.min(1.0)).max(0.0);

    // Combine factors
    let combined = size_confidence * 0.6 + cv_confidence * 0.4;

    combined.clamp(0.1, 0.95)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline_config_defaults() {
        let config = TimelineConfig::default();
        assert!((config.min_percentile - 10.0).abs() < 0.001);
        assert!((config.max_percentile - 90.0).abs() < 0.001);
        assert_eq!(config.default_min_days, 1);
        assert_eq!(config.default_max_days, 30);
    }

    #[test]
    fn test_estimate_timeline_empty() {
        let config = TimelineConfig::default();
        let result = estimate_timeline(&[], &config).unwrap();

        assert_eq!(result.min_days, config.default_min_days);
        assert_eq!(result.max_days, config.default_max_days);
        assert_eq!(result.data_points, 0);
        assert!((result.analysis.confidence - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_estimate_timeline_single_point() {
        let config = TimelineConfig::default();
        let result = estimate_timeline(&[14], &config).unwrap();

        assert_eq!(result.min_days, 14);
        assert_eq!(result.max_days, 14);
        assert_eq!(result.median_days, 14);
        assert_eq!(result.data_points, 1);
    }

    #[test]
    fn test_estimate_timeline_basic() {
        let config = TimelineConfig::default();
        let times = vec![5, 7, 10, 12, 14, 15, 18, 20, 25, 30];

        let result = estimate_timeline(&times, &config).unwrap();

        // 10th percentile should be around 5-7
        assert!(result.min_days >= 5);
        assert!(result.min_days <= 10);

        // 90th percentile should be around 25-30
        assert!(result.max_days >= 20);
        assert!(result.max_days <= 30);

        assert_eq!(result.data_points, 10);
    }

    #[test]
    fn test_estimate_timeline_with_outliers() {
        let config = TimelineConfig::default().with_outlier_removal(true);

        // Include extreme outliers
        let times = vec![7, 10, 12, 14, 15, 18, 20, 100, 200];

        let result = estimate_timeline(&times, &config).unwrap();

        // Outliers should be removed
        assert!(result.outliers_removed > 0);
        // Max should be reasonable (not influenced by 100, 200)
        assert!(result.max_days < 50);
    }

    #[test]
    fn test_estimate_timeline_constraints() {
        let config = TimelineConfig::default()
            .with_constraints(7, 60); // Floor of 7, ceiling of 60

        let times = vec![1, 2, 3, 100, 150, 200];

        let result = estimate_timeline(&times, &config).unwrap();

        // Should respect floor
        assert!(result.min_days >= 7);
        // Should respect ceiling
        assert!(result.max_days <= 60);
    }

    #[test]
    fn test_percentile_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // 0th percentile should be min
        assert!((percentile(&data, 0.0) - 1.0).abs() < 0.001);

        // 100th percentile should be max
        assert!((percentile(&data, 100.0) - 10.0).abs() < 0.001);

        // 50th percentile should be median
        let median = percentile(&data, 50.0);
        assert!(median >= 5.0 && median <= 6.0);
    }

    #[test]
    fn test_timeline_estimate_default() {
        let estimate = TimelineEstimate::default();

        assert_eq!(estimate.min_days, 1);
        assert_eq!(estimate.max_days, 30);
        assert_eq!(estimate.median_days, 15);
    }

    #[test]
    fn test_timeline_estimator() {
        let estimator = TimelineEstimator::with_config(
            TimelineConfig::default().with_percentiles(5.0, 95.0),
        );

        let times = vec![7, 14, 21, 28, 35];
        let result = estimator.estimate(&times).unwrap();

        assert!(result.min_days <= result.median_days);
        assert!(result.median_days <= result.max_days);
    }

    #[test]
    fn test_timeline_confidence() {
        let config = TimelineConfig::default();

        // Small dataset should have lower confidence
        let small_result = estimate_timeline(&[7, 14], &config).unwrap();

        // Large dataset should have higher confidence
        let large_data: Vec<u32> = (1..=30).collect();
        let large_result = estimate_timeline(&large_data, &config).unwrap();

        assert!(large_result.analysis.confidence > small_result.analysis.confidence);
    }

    #[test]
    fn test_iqr_calculation() {
        let config = TimelineConfig::default();
        let times = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        let result = estimate_timeline(&times, &config).unwrap();

        // Q1 should be around 3-4
        assert!(result.analysis.q1.unwrap() >= 3);
        assert!(result.analysis.q1.unwrap() <= 4);

        // Q3 should be around 9-10
        assert!(result.analysis.q3.unwrap() >= 9);
        assert!(result.analysis.q3.unwrap() <= 10);
    }

    #[test]
    fn test_min_max_ordering() {
        let config = TimelineConfig::default();

        // Various distributions
        let test_cases = vec![
            vec![1, 1, 1, 1, 1],       // All same
            vec![1, 10, 100],          // Wide spread
            vec![30, 31, 32, 33, 34],  // Tight cluster
            vec![5],                    // Single value
        ];

        for times in test_cases {
            let result = estimate_timeline(&times, &config).unwrap();
            assert!(result.min_days <= result.max_days);
            assert!(result.min_days <= result.median_days);
            assert!(result.median_days <= result.max_days);
        }
    }
}
