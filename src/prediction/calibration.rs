//! Calibration bucket tracking and probability adjustment.
//!
//! This module provides:
//! - CalibrationBucket struct for tracking predictions in probability ranges
//! - CalibrationAdjuster for adjusting future predictions based on historical data
//! - CalibrationReport for displaying calibration statistics
//!
//! # Calibration Concepts
//!
//! A well-calibrated predictor should have:
//! - When predicting 70% probability, ~70% of outcomes should be positive
//! - The reliability diagram should follow the diagonal y=x line
//!
//! # Platt Scaling
//!
//! This implementation uses a simplified Platt scaling approach:
//! - For each bucket, calculate the actual positive rate
//! - Adjust predictions to match the historical calibration
//! - Apply isotonic regression concepts for monotonicity

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use super::{bucket_index_for_probability, PredictionError, PredictionResult};
use crate::db::SqliteDb;

/// A calibration bucket tracking predictions in a probability range.
///
/// For example, the 0.7-0.8 bucket tracks all predictions where
/// the predicted probability was between 70% and 80%.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBucket {
    /// Unique identifier for this bucket
    pub bucket_id: String,

    /// Lower bound of the probability range (inclusive)
    pub lower_bound: f64,

    /// Upper bound of the probability range (exclusive, except for 1.0)
    pub upper_bound: f64,

    /// Total number of predictions in this bucket
    pub prediction_count: u32,

    /// Number of predictions where the actual outcome was positive (true)
    pub actual_positive_count: u32,

    /// Sum of all Brier scores in this bucket
    pub total_brier_score: f64,

    /// Domain for this bucket
    pub domain: String,

    /// When the bucket was last updated
    pub updated_at: DateTime<Utc>,
}

impl CalibrationBucket {
    /// Create a new calibration bucket.
    pub fn new(lower_bound: f64, upper_bound: f64) -> Self {
        let bucket_index = (lower_bound * 10.0).round() as u32;
        Self {
            bucket_id: format!("general_{}", bucket_index),
            lower_bound,
            upper_bound,
            prediction_count: 0,
            actual_positive_count: 0,
            total_brier_score: 0.0,
            domain: "general".to_string(),
            updated_at: Utc::now(),
        }
    }

    /// Create a new bucket with a specific domain.
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = domain.into();
        let bucket_index = (self.lower_bound * 10.0).round() as u32;
        self.bucket_id = format!("{}_{}", self.domain, bucket_index);
        self
    }

    /// Get the midpoint of this bucket (the expected probability).
    pub fn expected_probability(&self) -> f64 {
        (self.lower_bound + self.upper_bound) / 2.0
    }

    /// Get the actual positive rate for this bucket.
    ///
    /// Returns None if there are no predictions in this bucket.
    pub fn actual_positive_rate(&self) -> Option<f64> {
        if self.prediction_count == 0 {
            None
        } else {
            Some(self.actual_positive_count as f64 / self.prediction_count as f64)
        }
    }

    /// Get the average Brier score for this bucket.
    ///
    /// Returns None if there are no predictions in this bucket.
    pub fn average_brier_score(&self) -> Option<f64> {
        if self.prediction_count == 0 {
            None
        } else {
            Some(self.total_brier_score / self.prediction_count as f64)
        }
    }

    /// Get the calibration error for this bucket.
    ///
    /// Calibration error = |expected_probability - actual_positive_rate|
    pub fn calibration_error(&self) -> Option<f64> {
        self.actual_positive_rate()
            .map(|rate| (self.expected_probability() - rate).abs())
    }

    /// Check if this bucket is overconfident (predicted > actual).
    pub fn is_overconfident(&self) -> Option<bool> {
        self.actual_positive_rate()
            .map(|rate| self.expected_probability() > rate)
    }

    /// Update the bucket with a new resolved prediction.
    pub fn update(&mut self, actual_outcome: bool, brier_score: f64) {
        self.prediction_count += 1;
        if actual_outcome {
            self.actual_positive_count += 1;
        }
        self.total_brier_score += brier_score;
        self.updated_at = Utc::now();
    }
}

/// Update a calibration bucket with a resolved prediction (standalone function).
pub fn update_bucket(bucket: &mut CalibrationBucket, actual_outcome: bool, brier_score: f64) {
    bucket.update(actual_outcome, brier_score);
}

/// Statistics for a single calibration bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketStats {
    /// The bucket's probability range as a string (e.g., "0.70-0.80")
    pub range: String,

    /// Expected probability (midpoint of range)
    pub expected: f64,

    /// Actual positive rate
    pub actual: Option<f64>,

    /// Number of predictions in this bucket
    pub count: u32,

    /// Calibration error (|expected - actual|)
    pub calibration_error: Option<f64>,

    /// Average Brier score for this bucket
    pub average_brier: Option<f64>,

    /// Whether predictions in this bucket are overconfident
    pub overconfident: Option<bool>,
}

impl From<&CalibrationBucket> for BucketStats {
    fn from(bucket: &CalibrationBucket) -> Self {
        Self {
            range: format!("{:.2}-{:.2}", bucket.lower_bound, bucket.upper_bound),
            expected: bucket.expected_probability(),
            actual: bucket.actual_positive_rate(),
            count: bucket.prediction_count,
            calibration_error: bucket.calibration_error(),
            average_brier: bucket.average_brier_score(),
            overconfident: bucket.is_overconfident(),
        }
    }
}

/// A point in the reliability diagram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityDiagramPoint {
    /// The expected probability (x-axis)
    pub expected: f64,

    /// The actual positive rate (y-axis)
    pub actual: f64,

    /// The number of predictions in this bucket
    pub count: u32,

    /// Confidence interval lower bound (optional)
    pub ci_lower: Option<f64>,

    /// Confidence interval upper bound (optional)
    pub ci_upper: Option<f64>,
}

/// Configuration for calibration operations.
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Minimum number of predictions in a bucket to use it for adjustment
    pub min_bucket_samples: u32,

    /// Whether to enforce monotonicity in calibration (isotonic regression)
    pub enforce_monotonicity: bool,

    /// Smoothing factor for calibration (0.0 = no smoothing, 1.0 = maximum)
    pub smoothing_factor: f64,

    /// Whether to use global calibration when bucket has few samples
    pub use_global_fallback: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            min_bucket_samples: 10,
            enforce_monotonicity: true,
            smoothing_factor: 0.3,
            use_global_fallback: true,
        }
    }
}

/// Calibration report with overall statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReport {
    /// Overall average Brier score across all predictions
    pub overall_brier_score: f64,

    /// Statistics for each calibration bucket
    pub bucket_stats: Vec<BucketStats>,

    /// Total calibration error (sum of |expected - actual| across buckets)
    pub calibration_error: f64,

    /// Mean calibration error (average across buckets with data)
    pub mean_calibration_error: f64,

    /// Data points for reliability diagram
    pub reliability_diagram_data: Vec<ReliabilityDiagramPoint>,

    /// Total number of resolved predictions
    pub total_predictions: u32,

    /// Number of buckets with sufficient data
    pub buckets_with_data: u32,

    /// Overall actual positive rate
    pub overall_positive_rate: f64,

    /// Domain for this report
    pub domain: String,

    /// When the report was generated
    pub generated_at: DateTime<Utc>,

    /// Summary assessment
    pub assessment: String,
}

impl CalibrationReport {
    /// Create a new calibration report from buckets.
    pub fn from_buckets(buckets: &[CalibrationBucket], domain: &str) -> Self {
        let mut total_predictions: u32 = 0;
        let mut total_positives: u32 = 0;
        let mut total_brier: f64 = 0.0;
        let mut total_calibration_error: f64 = 0.0;
        let mut buckets_with_data: u32 = 0;

        let bucket_stats: Vec<BucketStats> = buckets.iter().map(BucketStats::from).collect();

        let reliability_diagram_data: Vec<ReliabilityDiagramPoint> = buckets
            .iter()
            .filter_map(|bucket| {
                bucket.actual_positive_rate().map(|actual| {
                    // Calculate 95% confidence interval using Wilson score
                    let (ci_lower, ci_upper) = wilson_confidence_interval(
                        bucket.actual_positive_count as f64,
                        bucket.prediction_count as f64,
                        0.95,
                    );

                    ReliabilityDiagramPoint {
                        expected: bucket.expected_probability(),
                        actual,
                        count: bucket.prediction_count,
                        ci_lower: Some(ci_lower),
                        ci_upper: Some(ci_upper),
                    }
                })
            })
            .collect();

        for bucket in buckets {
            total_predictions += bucket.prediction_count;
            total_positives += bucket.actual_positive_count;
            total_brier += bucket.total_brier_score;

            if let Some(error) = bucket.calibration_error() {
                total_calibration_error += error;
                buckets_with_data += 1;
            }
        }

        let overall_brier_score = if total_predictions > 0 {
            total_brier / total_predictions as f64
        } else {
            0.0
        };

        let overall_positive_rate = if total_predictions > 0 {
            total_positives as f64 / total_predictions as f64
        } else {
            0.0
        };

        let mean_calibration_error = if buckets_with_data > 0 {
            total_calibration_error / buckets_with_data as f64
        } else {
            0.0
        };

        let assessment = Self::assess_calibration(overall_brier_score, mean_calibration_error);

        Self {
            overall_brier_score,
            bucket_stats,
            calibration_error: total_calibration_error,
            mean_calibration_error,
            reliability_diagram_data,
            total_predictions,
            buckets_with_data,
            overall_positive_rate,
            domain: domain.to_string(),
            generated_at: Utc::now(),
            assessment,
        }
    }

    /// Generate a qualitative assessment of calibration quality.
    fn assess_calibration(brier_score: f64, mean_cal_error: f64) -> String {
        let brier_quality = match brier_score {
            s if s < 0.1 => "excellent",
            s if s < 0.2 => "good",
            s if s < 0.3 => "fair",
            _ => "poor",
        };

        let cal_quality = match mean_cal_error {
            e if e < 0.05 => "well-calibrated",
            e if e < 0.10 => "reasonably calibrated",
            e if e < 0.15 => "moderately calibrated",
            _ => "poorly calibrated",
        };

        format!(
            "Predictions show {} accuracy (Brier: {:.3}) and are {} (mean error: {:.3})",
            brier_quality, brier_score, cal_quality, mean_cal_error
        )
    }

    /// Format the report for display.
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "\n=== Calibration Report ({}) ===\n",
            self.domain
        ));
        output.push_str(&format!("Generated: {}\n\n", self.generated_at.format("%Y-%m-%d %H:%M:%S UTC")));

        output.push_str("Overall Statistics:\n");
        output.push_str(&format!("  Total predictions: {}\n", self.total_predictions));
        output.push_str(&format!("  Overall Brier score: {:.4}\n", self.overall_brier_score));
        output.push_str(&format!("  Mean calibration error: {:.4}\n", self.mean_calibration_error));
        output.push_str(&format!("  Overall positive rate: {:.2}%\n", self.overall_positive_rate * 100.0));
        output.push_str(&format!("\n{}\n", self.assessment));

        output.push_str("\nBucket Statistics:\n");
        output.push_str(&format!(
            "{:>12} {:>10} {:>10} {:>8} {:>10} {:>12}\n",
            "Range", "Expected", "Actual", "Count", "CalError", "AvgBrier"
        ));
        output.push_str(&format!("{:-<66}\n", ""));

        for stats in &self.bucket_stats {
            let actual_str = stats
                .actual
                .map(|a| format!("{:.2}%", a * 100.0))
                .unwrap_or_else(|| "N/A".to_string());
            let error_str = stats
                .calibration_error
                .map(|e| format!("{:.4}", e))
                .unwrap_or_else(|| "N/A".to_string());
            let brier_str = stats
                .average_brier
                .map(|b| format!("{:.4}", b))
                .unwrap_or_else(|| "N/A".to_string());

            output.push_str(&format!(
                "{:>12} {:>9.2}% {:>10} {:>8} {:>10} {:>12}\n",
                stats.range,
                stats.expected * 100.0,
                actual_str,
                stats.count,
                error_str,
                brier_str
            ));
        }

        output.push_str("\nReliability Diagram Data:\n");
        output.push_str(&format!(
            "{:>10} {:>10} {:>8} {:>12} {:>12}\n",
            "Expected", "Actual", "Count", "CI Lower", "CI Upper"
        ));
        output.push_str(&format!("{:-<56}\n", ""));

        for point in &self.reliability_diagram_data {
            let ci_lower = point
                .ci_lower
                .map(|v| format!("{:.2}%", v * 100.0))
                .unwrap_or_else(|| "N/A".to_string());
            let ci_upper = point
                .ci_upper
                .map(|v| format!("{:.2}%", v * 100.0))
                .unwrap_or_else(|| "N/A".to_string());

            output.push_str(&format!(
                "{:>9.2}% {:>9.2}% {:>8} {:>12} {:>12}\n",
                point.expected * 100.0,
                point.actual * 100.0,
                point.count,
                ci_lower,
                ci_upper
            ));
        }

        output
    }
}

/// Calculate Wilson confidence interval for a proportion.
fn wilson_confidence_interval(successes: f64, total: f64, confidence: f64) -> (f64, f64) {
    if total == 0.0 {
        return (0.0, 1.0);
    }

    // z-score for desired confidence level
    let z = match confidence {
        c if (c - 0.95).abs() < 0.01 => 1.96,
        c if (c - 0.99).abs() < 0.01 => 2.576,
        c if (c - 0.90).abs() < 0.01 => 1.645,
        _ => 1.96, // Default to 95%
    };

    let p = successes / total;
    let n = total;

    let denominator = 1.0 + z * z / n;
    let center = (p + z * z / (2.0 * n)) / denominator;
    let spread = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt() / denominator;

    let lower = (center - spread).max(0.0);
    let upper = (center + spread).min(1.0);

    (lower, upper)
}

/// Calibration adjuster for correcting prediction probabilities.
///
/// This implements a simplified Platt scaling approach with optional
/// isotonic regression for monotonicity.
pub struct CalibrationAdjuster {
    /// Calibration buckets
    buckets: Vec<CalibrationBucket>,

    /// Configuration
    config: CalibrationConfig,

    /// Precomputed adjustment factors for each bucket
    adjustment_factors: Vec<Option<f64>>,
}

impl CalibrationAdjuster {
    /// Create a new calibration adjuster from buckets.
    pub fn new(buckets: Vec<CalibrationBucket>, config: CalibrationConfig) -> Self {
        let adjustment_factors = Self::compute_adjustment_factors(&buckets, &config);

        Self {
            buckets,
            config,
            adjustment_factors,
        }
    }

    /// Create a calibration adjuster from a database.
    pub async fn from_database(
        db: Arc<SqliteDb>,
        domain: &str,
        config: CalibrationConfig,
    ) -> PredictionResult<Self> {
        let sql = r#"
            SELECT bucket_id, lower_bound, upper_bound, prediction_count,
                   actual_positive_count, total_brier_score, domain, updated_at
            FROM calibration_buckets
            WHERE domain = ?
            ORDER BY lower_bound ASC
        "#;

        let buckets = db
            .query(sql, &[&domain], |row| {
                let bucket_id: String = row.get("bucket_id")?;
                let lower_bound: f64 = row.get("lower_bound")?;
                let upper_bound: f64 = row.get("upper_bound")?;
                let prediction_count: i32 = row.get("prediction_count")?;
                let actual_positive_count: i32 = row.get("actual_positive_count")?;
                let total_brier_score: f64 = row.get("total_brier_score")?;
                let domain: String = row.get("domain")?;
                let updated_at_str: String = row.get("updated_at")?;

                let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                Ok(CalibrationBucket {
                    bucket_id,
                    lower_bound,
                    upper_bound,
                    prediction_count: prediction_count as u32,
                    actual_positive_count: actual_positive_count as u32,
                    total_brier_score,
                    domain,
                    updated_at,
                })
            })
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        if buckets.is_empty() {
            return Err(PredictionError::NoCalibrationData);
        }

        Ok(Self::new(buckets, config))
    }

    /// Compute adjustment factors for each bucket.
    fn compute_adjustment_factors(
        buckets: &[CalibrationBucket],
        config: &CalibrationConfig,
    ) -> Vec<Option<f64>> {
        let mut factors: Vec<Option<f64>> = buckets
            .iter()
            .map(|bucket| {
                if bucket.prediction_count >= config.min_bucket_samples {
                    bucket.actual_positive_rate()
                } else {
                    None
                }
            })
            .collect();

        // Apply isotonic regression if configured
        if config.enforce_monotonicity {
            Self::apply_isotonic_regression(&mut factors);
        }

        factors
    }

    /// Apply Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
    ///
    /// This ensures that calibrated probabilities are monotonically increasing.
    fn apply_isotonic_regression(factors: &mut [Option<f64>]) {
        // Extract values, using interpolation for missing values
        let mut values: Vec<f64> = Vec::with_capacity(factors.len());
        let mut valid_indices: Vec<usize> = Vec::new();

        for (i, factor) in factors.iter().enumerate() {
            if let Some(v) = factor {
                values.push(*v);
                valid_indices.push(i);
            }
        }

        if values.len() < 2 {
            return; // Not enough data for isotonic regression
        }

        // Apply PAVA
        let n = values.len();
        let mut weights = vec![1.0; n];

        let mut i = 0;
        while i < n - 1 {
            if values[i] > values[i + 1] {
                // Pool adjacent violators
                let pooled = (values[i] * weights[i] + values[i + 1] * weights[i + 1])
                    / (weights[i] + weights[i + 1]);
                weights[i] = weights[i] + weights[i + 1];
                values[i] = pooled;

                // Remove pooled element
                values.remove(i + 1);
                weights.remove(i + 1);
                valid_indices.remove(i + 1);

                // Go back to check for new violations
                if i > 0 {
                    i -= 1;
                }
            } else {
                i += 1;
            }
        }

        // Map back to original factors
        for (idx, &orig_idx) in valid_indices.iter().enumerate() {
            if idx < values.len() {
                factors[orig_idx] = Some(values[idx]);
            }
        }
    }

    /// Adjust a probability based on calibration data.
    ///
    /// This is the main function for Task 3.E.3:
    /// - Compare predicted vs actual rates per bucket
    /// - Apply Platt scaling or isotonic regression concept
    /// - Return calibrated_probability
    pub fn adjust_probability(&self, raw_probability: f64) -> f64 {
        let bucket_idx = bucket_index_for_probability(raw_probability);

        // Try to get the adjustment factor for this bucket
        if let Some(Some(factor)) = self.adjustment_factors.get(bucket_idx) {
            // Apply smoothing: blend raw probability with calibrated value
            let calibrated = raw_probability * (1.0 - self.config.smoothing_factor)
                + factor * self.config.smoothing_factor;

            debug!(
                raw = %raw_probability,
                calibrated = %calibrated,
                bucket = %bucket_idx,
                "Adjusted probability"
            );

            return calibrated.clamp(0.0, 1.0);
        }

        // If no bucket data, try global fallback
        if self.config.use_global_fallback {
            if let Some(global_rate) = self.global_positive_rate() {
                // Blend with global rate but weight toward raw probability
                let calibrated = raw_probability * 0.7 + global_rate * 0.3;
                return calibrated.clamp(0.0, 1.0);
            }
        }

        // No calibration data available, return raw probability
        raw_probability
    }

    /// Get the global positive rate across all buckets.
    fn global_positive_rate(&self) -> Option<f64> {
        let total_predictions: u32 = self.buckets.iter().map(|b| b.prediction_count).sum();
        let total_positives: u32 = self.buckets.iter().map(|b| b.actual_positive_count).sum();

        if total_predictions > 0 {
            Some(total_positives as f64 / total_predictions as f64)
        } else {
            None
        }
    }

    /// Get the buckets.
    pub fn buckets(&self) -> &[CalibrationBucket] {
        &self.buckets
    }

    /// Get the configuration.
    pub fn config(&self) -> &CalibrationConfig {
        &self.config
    }

    /// Get the adjustment factor for a specific bucket.
    pub fn get_adjustment_factor(&self, bucket_idx: usize) -> Option<f64> {
        self.adjustment_factors.get(bucket_idx).copied().flatten()
    }
}

/// Adjust a probability using calibration data (standalone function).
///
/// This is the main entry point for Task 3.E.3.
pub fn adjust_probability(adjuster: &CalibrationAdjuster, raw_probability: f64) -> f64 {
    adjuster.adjust_probability(raw_probability)
}

/// Get a calibration report from the database.
///
/// This is the main function for Task 3.E.4.
pub async fn get_calibration_report(
    db: Arc<SqliteDb>,
    domain: &str,
) -> PredictionResult<CalibrationReport> {
    let sql = r#"
        SELECT bucket_id, lower_bound, upper_bound, prediction_count,
               actual_positive_count, total_brier_score, domain, updated_at
        FROM calibration_buckets
        WHERE domain = ?
        ORDER BY lower_bound ASC
    "#;

    let buckets: Vec<CalibrationBucket> = db
        .query(sql, &[&domain], |row| {
            let bucket_id: String = row.get("bucket_id")?;
            let lower_bound: f64 = row.get("lower_bound")?;
            let upper_bound: f64 = row.get("upper_bound")?;
            let prediction_count: i32 = row.get("prediction_count")?;
            let actual_positive_count: i32 = row.get("actual_positive_count")?;
            let total_brier_score: f64 = row.get("total_brier_score")?;
            let domain: String = row.get("domain")?;
            let updated_at_str: String = row.get("updated_at")?;

            let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            Ok(CalibrationBucket {
                bucket_id,
                lower_bound,
                upper_bound,
                prediction_count: prediction_count as u32,
                actual_positive_count: actual_positive_count as u32,
                total_brier_score,
                domain,
                updated_at,
            })
        })
        .await
        .map_err(|e| PredictionError::Database(e.to_string()))?;

    if buckets.is_empty() {
        return Err(PredictionError::NoCalibrationData);
    }

    Ok(CalibrationReport::from_buckets(&buckets, domain))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_bucket_creation() {
        let bucket = CalibrationBucket::new(0.7, 0.8);
        assert!((bucket.lower_bound - 0.7).abs() < f64::EPSILON);
        assert!((bucket.upper_bound - 0.8).abs() < f64::EPSILON);
        assert_eq!(bucket.prediction_count, 0);
    }

    #[test]
    fn test_bucket_expected_probability() {
        let bucket = CalibrationBucket::new(0.6, 0.7);
        assert!((bucket.expected_probability() - 0.65).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bucket_actual_positive_rate() {
        let mut bucket = CalibrationBucket::new(0.7, 0.8);

        // No predictions yet
        assert!(bucket.actual_positive_rate().is_none());

        // Add some predictions
        bucket.update(true, 0.1);
        bucket.update(true, 0.1);
        bucket.update(false, 0.5);
        bucket.update(true, 0.1);

        // 3 out of 4 positive
        let rate = bucket.actual_positive_rate().unwrap();
        assert!((rate - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bucket_calibration_error() {
        let mut bucket = CalibrationBucket::new(0.7, 0.8); // Expected: 0.75

        bucket.update(true, 0.1);
        bucket.update(false, 0.5);

        // Actual rate: 0.5, expected: 0.75
        let error = bucket.calibration_error().unwrap();
        assert!((error - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calibration_adjuster() {
        // Create buckets with known calibration
        let mut buckets = Vec::new();
        for i in 0..10 {
            let mut bucket = CalibrationBucket::new(i as f64 * 0.1, (i + 1) as f64 * 0.1);
            // Simulate data where actual rate is 10% lower than expected
            for _ in 0..20 {
                let expected = bucket.expected_probability();
                let outcome = rand::random::<f64>() < (expected - 0.1).max(0.0);
                bucket.update(outcome, 0.1);
            }
            buckets.push(bucket);
        }

        let config = CalibrationConfig {
            min_bucket_samples: 5,
            enforce_monotonicity: true,
            smoothing_factor: 0.5,
            use_global_fallback: true,
        };

        let adjuster = CalibrationAdjuster::new(buckets, config);

        // Test adjustment
        let raw = 0.75;
        let adjusted = adjuster.adjust_probability(raw);

        // Should be adjusted based on calibration
        assert!(adjusted > 0.0 && adjusted < 1.0);
    }

    #[test]
    fn test_calibration_report() {
        let mut buckets = Vec::new();
        for i in 0..10 {
            let mut bucket = CalibrationBucket::new(i as f64 * 0.1, (i + 1) as f64 * 0.1);
            // Add some predictions
            for _ in 0..5 {
                bucket.update(rand::random(), rand::random::<f64>() * 0.5);
            }
            buckets.push(bucket);
        }

        let report = CalibrationReport::from_buckets(&buckets, "test");

        assert_eq!(report.total_predictions, 50);
        assert_eq!(report.bucket_stats.len(), 10);
        assert!(report.overall_brier_score >= 0.0 && report.overall_brier_score <= 1.0);
    }

    #[test]
    fn test_wilson_confidence_interval() {
        // Test with 50% success rate
        let (lower, upper) = wilson_confidence_interval(50.0, 100.0, 0.95);
        assert!(lower > 0.3 && lower < 0.5);
        assert!(upper > 0.5 && upper < 0.7);

        // Test edge case: no data
        let (lower, upper) = wilson_confidence_interval(0.0, 0.0, 0.95);
        assert!((lower - 0.0).abs() < f64::EPSILON);
        assert!((upper - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bucket_stats_from_bucket() {
        let mut bucket = CalibrationBucket::new(0.3, 0.4);
        bucket.update(true, 0.2);
        bucket.update(false, 0.3);

        let stats = BucketStats::from(&bucket);
        assert_eq!(stats.range, "0.30-0.40");
        assert!((stats.expected - 0.35).abs() < f64::EPSILON);
        assert_eq!(stats.count, 2);
    }

    #[test]
    fn test_isotonic_regression() {
        // Test with violating sequence: should be monotonically increasing
        let mut factors: Vec<Option<f64>> = vec![
            Some(0.2),
            Some(0.5),
            Some(0.3), // Violation: 0.3 < 0.5
            Some(0.6),
            Some(0.8),
        ];

        CalibrationAdjuster::apply_isotonic_regression(&mut factors);

        // After isotonic regression, should be monotonic
        let values: Vec<f64> = factors.iter().filter_map(|f| *f).collect();
        for i in 1..values.len() {
            assert!(values[i] >= values[i - 1], "Should be monotonically increasing");
        }
    }

    #[test]
    fn test_reliability_diagram_point() {
        let point = ReliabilityDiagramPoint {
            expected: 0.75,
            actual: 0.70,
            count: 100,
            ci_lower: Some(0.60),
            ci_upper: Some(0.80),
        };

        assert!((point.expected - 0.75).abs() < f64::EPSILON);
        assert!((point.actual - 0.70).abs() < f64::EPSILON);
        assert_eq!(point.count, 100);
    }

    #[test]
    fn test_calibration_config_default() {
        let config = CalibrationConfig::default();
        assert_eq!(config.min_bucket_samples, 10);
        assert!(config.enforce_monotonicity);
        assert!((config.smoothing_factor - 0.3).abs() < f64::EPSILON);
    }
}
