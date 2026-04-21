//! A/B Testing Framework for Retrieval Methods
//!
//! This module provides a comprehensive test harness for comparing different
//! retrieval methods in the learning system. It tracks metrics including
//! precision, recall, latency, and performs statistical significance testing.
//!
//! # Test Scenarios
//!
//! - Compare baseline retrieval vs SONA-optimized retrieval
//! - Compare different MMR configurations
//! - Compare scoring weight configurations
//! - Validate statistical significance of improvements

use std::collections::HashMap;
use std::time::{Duration, Instant};

use chrono::Utc;
use ndarray::Array1;

/// Metrics collected during A/B test execution.
#[derive(Debug, Clone, Default)]
pub struct RetrievalMetrics {
    /// Precision at K (percentage of retrieved results that are relevant)
    pub precision_at_k: f64,

    /// Recall at K (percentage of relevant results that were retrieved)
    pub recall_at_k: f64,

    /// Mean reciprocal rank
    pub mrr: f64,

    /// Average latency in milliseconds
    pub avg_latency_ms: f64,

    /// P95 latency in milliseconds
    pub p95_latency_ms: f64,

    /// P99 latency in milliseconds
    pub p99_latency_ms: f64,

    /// Total queries executed
    pub query_count: usize,

    /// Average number of results returned
    pub avg_results_returned: f64,
}

impl RetrievalMetrics {
    /// Calculate metrics from raw measurements.
    pub fn from_measurements(
        latencies: &[f64],
        precisions: &[f64],
        recalls: &[f64],
        mrrs: &[f64],
        results_counts: &[usize],
    ) -> Self {
        let query_count = latencies.len();
        if query_count == 0 {
            return Self::default();
        }

        let avg_latency_ms = latencies.iter().sum::<f64>() / query_count as f64;
        let precision_at_k = precisions.iter().sum::<f64>() / query_count as f64;
        let recall_at_k = recalls.iter().sum::<f64>() / query_count as f64;
        let mrr = mrrs.iter().sum::<f64>() / query_count as f64;
        let avg_results_returned = results_counts.iter().sum::<usize>() as f64 / query_count as f64;

        // Calculate percentiles
        let mut sorted_latencies = latencies.to_vec();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p95_idx = (query_count as f64 * 0.95) as usize;
        let p99_idx = (query_count as f64 * 0.99) as usize;

        let p95_latency_ms = sorted_latencies.get(p95_idx.min(query_count - 1)).copied().unwrap_or(0.0);
        let p99_latency_ms = sorted_latencies.get(p99_idx.min(query_count - 1)).copied().unwrap_or(0.0);

        Self {
            precision_at_k,
            recall_at_k,
            mrr,
            avg_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            query_count,
            avg_results_returned,
        }
    }
}

/// Variant being tested in the A/B test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TestVariant {
    /// Control group (baseline)
    Control,
    /// Treatment group (new method)
    Treatment,
}

/// Configuration for the A/B test harness.
#[derive(Debug, Clone)]
pub struct AbTestConfig {
    /// Number of test queries to execute
    pub num_queries: usize,

    /// K value for precision/recall@K
    pub k: usize,

    /// Minimum sample size for significance testing
    pub min_sample_size: usize,

    /// Confidence level for statistical tests (e.g., 0.95)
    pub confidence_level: f64,

    /// Whether to enable warmup queries
    pub warmup_queries: usize,

    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for AbTestConfig {
    fn default() -> Self {
        Self {
            num_queries: 100,
            k: 10,
            min_sample_size: 30,
            confidence_level: 0.95,
            warmup_queries: 10,
            seed: 42,
        }
    }
}

/// Result of statistical significance testing.
#[derive(Debug, Clone)]
pub struct SignificanceResult {
    /// T-statistic value
    pub t_statistic: f64,

    /// P-value (probability of observing results under null hypothesis)
    pub p_value: f64,

    /// Whether the result is statistically significant
    pub is_significant: bool,

    /// Effect size (Cohen's d)
    pub effect_size: f64,

    /// 95% confidence interval lower bound
    pub ci_lower: f64,

    /// 95% confidence interval upper bound
    pub ci_upper: f64,
}

impl SignificanceResult {
    /// Check if the treatment shows significant improvement.
    pub fn treatment_improves(&self) -> bool {
        self.is_significant && self.effect_size > 0.0
    }
}

/// Comparison report between control and treatment.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// Control group metrics
    pub control_metrics: RetrievalMetrics,

    /// Treatment group metrics
    pub treatment_metrics: RetrievalMetrics,

    /// Precision improvement percentage
    pub precision_improvement_pct: f64,

    /// Recall improvement percentage
    pub recall_improvement_pct: f64,

    /// Latency improvement percentage (negative = faster)
    pub latency_improvement_pct: f64,

    /// Statistical significance for precision
    pub precision_significance: SignificanceResult,

    /// Statistical significance for recall
    pub recall_significance: SignificanceResult,

    /// Statistical significance for latency
    pub latency_significance: SignificanceResult,

    /// Overall recommendation
    pub recommendation: String,

    /// Report generation timestamp
    pub generated_at: chrono::DateTime<Utc>,
}

impl ComparisonReport {
    /// Generate a markdown report.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# A/B Test Comparison Report\n\n");
        md.push_str(&format!("Generated: {}\n\n", self.generated_at.format("%Y-%m-%d %H:%M:%S UTC")));

        md.push_str("## Summary\n\n");
        md.push_str("| Metric | Control | Treatment | Change |\n");
        md.push_str("|--------|---------|-----------|--------|\n");
        md.push_str(&format!(
            "| Precision@K | {:.4} | {:.4} | {:+.2}% |\n",
            self.control_metrics.precision_at_k,
            self.treatment_metrics.precision_at_k,
            self.precision_improvement_pct
        ));
        md.push_str(&format!(
            "| Recall@K | {:.4} | {:.4} | {:+.2}% |\n",
            self.control_metrics.recall_at_k,
            self.treatment_metrics.recall_at_k,
            self.recall_improvement_pct
        ));
        md.push_str(&format!(
            "| MRR | {:.4} | {:.4} | {:+.2}% |\n",
            self.control_metrics.mrr,
            self.treatment_metrics.mrr,
            if self.control_metrics.mrr > 0.0 {
                (self.treatment_metrics.mrr - self.control_metrics.mrr) / self.control_metrics.mrr * 100.0
            } else {
                0.0
            }
        ));
        md.push_str(&format!(
            "| Avg Latency (ms) | {:.2} | {:.2} | {:+.2}% |\n",
            self.control_metrics.avg_latency_ms,
            self.treatment_metrics.avg_latency_ms,
            self.latency_improvement_pct
        ));

        md.push_str("\n## Statistical Significance\n\n");
        md.push_str("| Metric | T-stat | P-value | Significant | Effect Size |\n");
        md.push_str("|--------|--------|---------|-------------|-------------|\n");
        md.push_str(&format!(
            "| Precision | {:.4} | {:.4} | {} | {:.4} |\n",
            self.precision_significance.t_statistic,
            self.precision_significance.p_value,
            if self.precision_significance.is_significant { "Yes" } else { "No" },
            self.precision_significance.effect_size
        ));
        md.push_str(&format!(
            "| Recall | {:.4} | {:.4} | {} | {:.4} |\n",
            self.recall_significance.t_statistic,
            self.recall_significance.p_value,
            if self.recall_significance.is_significant { "Yes" } else { "No" },
            self.recall_significance.effect_size
        ));
        md.push_str(&format!(
            "| Latency | {:.4} | {:.4} | {} | {:.4} |\n",
            self.latency_significance.t_statistic,
            self.latency_significance.p_value,
            if self.latency_significance.is_significant { "Yes" } else { "No" },
            self.latency_significance.effect_size
        ));

        md.push_str(&format!("\n## Recommendation\n\n{}\n", self.recommendation));

        md
    }
}

/// Test harness for A/B testing retrieval methods.
pub struct AbTestHarness {
    config: AbTestConfig,
    control_latencies: Vec<f64>,
    control_precisions: Vec<f64>,
    control_recalls: Vec<f64>,
    control_mrrs: Vec<f64>,
    control_result_counts: Vec<usize>,
    treatment_latencies: Vec<f64>,
    treatment_precisions: Vec<f64>,
    treatment_recalls: Vec<f64>,
    treatment_mrrs: Vec<f64>,
    treatment_result_counts: Vec<usize>,
}

impl AbTestHarness {
    /// Create a new test harness with the given configuration.
    pub fn new(config: AbTestConfig) -> Self {
        Self {
            config,
            control_latencies: Vec::new(),
            control_precisions: Vec::new(),
            control_recalls: Vec::new(),
            control_mrrs: Vec::new(),
            control_result_counts: Vec::new(),
            treatment_latencies: Vec::new(),
            treatment_precisions: Vec::new(),
            treatment_recalls: Vec::new(),
            treatment_mrrs: Vec::new(),
            treatment_result_counts: Vec::new(),
        }
    }

    /// Record a measurement for the control group.
    pub fn record_control(
        &mut self,
        latency_ms: f64,
        precision: f64,
        recall: f64,
        mrr: f64,
        result_count: usize,
    ) {
        self.control_latencies.push(latency_ms);
        self.control_precisions.push(precision);
        self.control_recalls.push(recall);
        self.control_mrrs.push(mrr);
        self.control_result_counts.push(result_count);
    }

    /// Record a measurement for the treatment group.
    pub fn record_treatment(
        &mut self,
        latency_ms: f64,
        precision: f64,
        recall: f64,
        mrr: f64,
        result_count: usize,
    ) {
        self.treatment_latencies.push(latency_ms);
        self.treatment_precisions.push(precision);
        self.treatment_recalls.push(recall);
        self.treatment_mrrs.push(mrr);
        self.treatment_result_counts.push(result_count);
    }

    /// Calculate statistical significance using Welch's t-test.
    fn calculate_significance(&self, control: &[f64], treatment: &[f64]) -> SignificanceResult {
        let n1 = control.len() as f64;
        let n2 = treatment.len() as f64;

        if n1 < self.config.min_sample_size as f64 || n2 < self.config.min_sample_size as f64 {
            return SignificanceResult {
                t_statistic: 0.0,
                p_value: 1.0,
                is_significant: false,
                effect_size: 0.0,
                ci_lower: 0.0,
                ci_upper: 0.0,
            };
        }

        // Calculate means
        let mean1 = control.iter().sum::<f64>() / n1;
        let mean2 = treatment.iter().sum::<f64>() / n2;

        // Calculate variances
        let var1 = control.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
        let var2 = treatment.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

        // Welch's t-test
        let se = ((var1 / n1) + (var2 / n2)).sqrt();
        let t_statistic = if se > 0.0 { (mean2 - mean1) / se } else { 0.0 };

        // Degrees of freedom (Welch-Satterthwaite)
        let df = if var1 > 0.0 && var2 > 0.0 {
            let num = (var1 / n1 + var2 / n2).powi(2);
            let den = (var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0);
            num / den
        } else {
            n1 + n2 - 2.0
        };

        // Approximate p-value using normal approximation for large df
        let p_value = if df > 30.0 {
            // Use normal approximation
            2.0 * (1.0 - normal_cdf(t_statistic.abs()))
        } else {
            // Simplified t-distribution approximation
            2.0 * (1.0 - t_cdf(t_statistic.abs(), df))
        };

        // Effect size (Cohen's d with pooled standard deviation)
        let pooled_std = ((((n1 - 1.0) * var1) + ((n2 - 1.0) * var2)) / (n1 + n2 - 2.0)).sqrt();
        let effect_size = if pooled_std > 0.0 {
            (mean2 - mean1) / pooled_std
        } else {
            0.0
        };

        // 95% confidence interval
        let t_critical = 1.96; // Approximate for large samples
        let ci_lower = (mean2 - mean1) - t_critical * se;
        let ci_upper = (mean2 - mean1) + t_critical * se;

        let alpha = 1.0 - self.config.confidence_level;
        let is_significant = p_value < alpha;

        SignificanceResult {
            t_statistic,
            p_value,
            is_significant,
            effect_size,
            ci_lower,
            ci_upper,
        }
    }

    /// Generate a comparison report.
    pub fn generate_report(&self) -> ComparisonReport {
        let control_metrics = RetrievalMetrics::from_measurements(
            &self.control_latencies,
            &self.control_precisions,
            &self.control_recalls,
            &self.control_mrrs,
            &self.control_result_counts,
        );

        let treatment_metrics = RetrievalMetrics::from_measurements(
            &self.treatment_latencies,
            &self.treatment_precisions,
            &self.treatment_recalls,
            &self.treatment_mrrs,
            &self.treatment_result_counts,
        );

        // Calculate improvement percentages
        let precision_improvement_pct = if control_metrics.precision_at_k > 0.0 {
            (treatment_metrics.precision_at_k - control_metrics.precision_at_k)
                / control_metrics.precision_at_k * 100.0
        } else {
            0.0
        };

        let recall_improvement_pct = if control_metrics.recall_at_k > 0.0 {
            (treatment_metrics.recall_at_k - control_metrics.recall_at_k)
                / control_metrics.recall_at_k * 100.0
        } else {
            0.0
        };

        let latency_improvement_pct = if control_metrics.avg_latency_ms > 0.0 {
            (control_metrics.avg_latency_ms - treatment_metrics.avg_latency_ms)
                / control_metrics.avg_latency_ms * 100.0
        } else {
            0.0
        };

        // Statistical significance tests
        let precision_significance = self.calculate_significance(
            &self.control_precisions,
            &self.treatment_precisions,
        );
        let recall_significance = self.calculate_significance(
            &self.control_recalls,
            &self.treatment_recalls,
        );
        let latency_significance = self.calculate_significance(
            &self.control_latencies,
            &self.treatment_latencies,
        );

        // Generate recommendation
        let recommendation = generate_recommendation(
            &precision_significance,
            &recall_significance,
            &latency_significance,
            precision_improvement_pct,
            recall_improvement_pct,
            latency_improvement_pct,
        );

        ComparisonReport {
            control_metrics,
            treatment_metrics,
            precision_improvement_pct,
            recall_improvement_pct,
            latency_improvement_pct,
            precision_significance,
            recall_significance,
            latency_significance,
            recommendation,
            generated_at: Utc::now(),
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &AbTestConfig {
        &self.config
    }

    /// Get the number of control samples collected.
    pub fn control_sample_count(&self) -> usize {
        self.control_latencies.len()
    }

    /// Get the number of treatment samples collected.
    pub fn treatment_sample_count(&self) -> usize {
        self.treatment_latencies.len()
    }
}

/// Generate a recommendation based on statistical results.
fn generate_recommendation(
    precision_sig: &SignificanceResult,
    recall_sig: &SignificanceResult,
    latency_sig: &SignificanceResult,
    precision_improvement: f64,
    recall_improvement: f64,
    latency_improvement: f64,
) -> String {
    let mut improvements = Vec::new();
    let mut regressions = Vec::new();
    let mut neutral = Vec::new();

    // Check precision
    if precision_sig.is_significant {
        if precision_improvement > 0.0 {
            improvements.push(format!("precision (+{:.1}%)", precision_improvement));
        } else {
            regressions.push(format!("precision ({:.1}%)", precision_improvement));
        }
    } else {
        neutral.push("precision");
    }

    // Check recall
    if recall_sig.is_significant {
        if recall_improvement > 0.0 {
            improvements.push(format!("recall (+{:.1}%)", recall_improvement));
        } else {
            regressions.push(format!("recall ({:.1}%)", recall_improvement));
        }
    } else {
        neutral.push("recall");
    }

    // Check latency (note: positive improvement = faster)
    if latency_sig.is_significant {
        if latency_improvement > 0.0 {
            improvements.push(format!("latency ({:.1}% faster)", latency_improvement));
        } else {
            regressions.push(format!("latency ({:.1}% slower)", -latency_improvement));
        }
    } else {
        neutral.push("latency");
    }

    // Generate recommendation text
    if !regressions.is_empty() {
        format!(
            "NOT RECOMMENDED: Treatment shows significant regression in {}. \
             Consider investigating root causes before deployment.",
            regressions.join(", ")
        )
    } else if !improvements.is_empty() && neutral.len() <= 1 {
        format!(
            "RECOMMENDED: Treatment shows significant improvement in {}. \
             Safe to proceed with gradual rollout.",
            improvements.join(", ")
        )
    } else if !improvements.is_empty() {
        format!(
            "CAUTIOUSLY RECOMMENDED: Treatment improves {} but {} show no significant change. \
             Consider extended testing.",
            improvements.join(", "),
            neutral.join(", ")
        )
    } else {
        "INCONCLUSIVE: No significant differences detected. Consider increasing sample size or \
         test duration."
            .to_string()
    }
}

/// Approximate normal CDF using error function approximation.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun).
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Approximate t-distribution CDF.
fn t_cdf(t: f64, df: f64) -> f64 {
    // Use normal approximation for simplicity
    // More accurate implementations would use the incomplete beta function
    let x = df / (df + t * t);
    0.5 + 0.5 * (1.0 - incomplete_beta(0.5 * df, 0.5, x).min(1.0).max(0.0)) * t.signum()
}

/// Incomplete beta function approximation.
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    // Simplified approximation
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    // Use continued fraction expansion approximation
    // This is a simplified version - production would use a library
    let mut bt = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp()
    };

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Continued fraction for incomplete beta function.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 100;
    let eps = 1e-10;

    let mut h = 1.0_f64;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < eps { d = eps; }
    let mut c = 1.0;
    d = 1.0 / d;
    h = d;

    for m in 1..=max_iter {
        let m = m as f64;

        // Even step
        let an = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + an * d;
        if d.abs() < eps { d = eps; }
        c = 1.0 + an / c;
        if c.abs() < eps { c = eps; }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let an = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + an * d;
        if d.abs() < eps { d = eps; }
        c = 1.0 + an / c;
        if c.abs() < eps { c = eps; }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    h
}

/// Log gamma function approximation (Stirling's approximation).
fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation coefficients
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = c[0];
        for i in 1..9 {
            a += c[i] / (x + i as f64);
        }
        let t = x + g + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (t - 0.5) * t.ln() - t + a.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_metrics_from_measurements() {
        let latencies = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let precisions = vec![0.8, 0.9, 0.7, 0.85, 0.75];
        let recalls = vec![0.6, 0.7, 0.65, 0.75, 0.7];
        let mrrs = vec![0.5, 0.6, 0.55, 0.65, 0.6];
        let result_counts = vec![10, 8, 9, 10, 7];

        let metrics = RetrievalMetrics::from_measurements(
            &latencies,
            &precisions,
            &recalls,
            &mrrs,
            &result_counts,
        );

        assert_eq!(metrics.query_count, 5);
        assert!((metrics.avg_latency_ms - 30.0).abs() < 0.001);
        assert!((metrics.precision_at_k - 0.8).abs() < 0.001);
        assert!((metrics.recall_at_k - 0.68).abs() < 0.001);
    }

    #[test]
    fn test_ab_test_harness_record() {
        let mut harness = AbTestHarness::new(AbTestConfig::default());

        // Record control measurements
        for i in 0..50 {
            harness.record_control(
                10.0 + (i as f64 * 0.1),
                0.75 + (i as f64 * 0.001),
                0.65 + (i as f64 * 0.001),
                0.55 + (i as f64 * 0.001),
                10,
            );
        }

        // Record treatment measurements (slightly better)
        for i in 0..50 {
            harness.record_treatment(
                9.0 + (i as f64 * 0.1),
                0.80 + (i as f64 * 0.001),
                0.70 + (i as f64 * 0.001),
                0.60 + (i as f64 * 0.001),
                10,
            );
        }

        assert_eq!(harness.control_sample_count(), 50);
        assert_eq!(harness.treatment_sample_count(), 50);
    }

    #[test]
    fn test_generate_report_with_improvement() {
        let mut harness = AbTestHarness::new(AbTestConfig {
            min_sample_size: 10,
            ..Default::default()
        });

        // Record control measurements
        for _ in 0..50 {
            harness.record_control(20.0, 0.70, 0.60, 0.50, 10);
        }

        // Record treatment measurements (better precision and recall)
        for _ in 0..50 {
            harness.record_treatment(18.0, 0.85, 0.75, 0.65, 10);
        }

        let report = harness.generate_report();

        assert!(report.precision_improvement_pct > 0.0);
        assert!(report.recall_improvement_pct > 0.0);
        assert!(report.latency_improvement_pct > 0.0);
    }

    #[test]
    fn test_significance_result_treatment_improves() {
        let result = SignificanceResult {
            t_statistic: 2.5,
            p_value: 0.01,
            is_significant: true,
            effect_size: 0.5,
            ci_lower: 0.1,
            ci_upper: 0.9,
        };

        assert!(result.treatment_improves());

        let no_improvement = SignificanceResult {
            t_statistic: 0.5,
            p_value: 0.5,
            is_significant: false,
            effect_size: 0.1,
            ci_lower: -0.1,
            ci_upper: 0.3,
        };

        assert!(!no_improvement.treatment_improves());
    }

    #[test]
    fn test_report_to_markdown() {
        let mut harness = AbTestHarness::new(AbTestConfig {
            min_sample_size: 10,
            ..Default::default()
        });

        for _ in 0..30 {
            harness.record_control(20.0, 0.70, 0.60, 0.50, 10);
            harness.record_treatment(18.0, 0.80, 0.70, 0.60, 10);
        }

        let report = harness.generate_report();
        let markdown = report.to_markdown();

        assert!(markdown.contains("# A/B Test Comparison Report"));
        assert!(markdown.contains("Precision@K"));
        assert!(markdown.contains("Recall@K"));
        assert!(markdown.contains("Statistical Significance"));
        assert!(markdown.contains("Recommendation"));
    }

    #[test]
    fn test_normal_cdf() {
        // Test standard normal values
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.001);
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.01);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.01);
    }

    #[test]
    fn test_ab_test_config_default() {
        let config = AbTestConfig::default();
        assert_eq!(config.num_queries, 100);
        assert_eq!(config.k, 10);
        assert_eq!(config.min_sample_size, 30);
        assert!((config.confidence_level - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_generate_recommendation_improvement() {
        let good_sig = SignificanceResult {
            t_statistic: 3.0,
            p_value: 0.001,
            is_significant: true,
            effect_size: 0.6,
            ci_lower: 0.2,
            ci_upper: 1.0,
        };

        let recommendation = generate_recommendation(
            &good_sig,
            &good_sig,
            &good_sig,
            10.0,
            8.0,
            5.0,
        );

        assert!(recommendation.contains("RECOMMENDED"));
    }

    #[test]
    fn test_generate_recommendation_regression() {
        let bad_sig = SignificanceResult {
            t_statistic: -3.0,
            p_value: 0.001,
            is_significant: true,
            effect_size: -0.6,
            ci_lower: -1.0,
            ci_upper: -0.2,
        };

        let recommendation = generate_recommendation(
            &bad_sig,
            &bad_sig,
            &bad_sig,
            -10.0,
            -8.0,
            -5.0,
        );

        assert!(recommendation.contains("NOT RECOMMENDED"));
    }

    #[test]
    fn test_empty_metrics() {
        let metrics = RetrievalMetrics::from_measurements(
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        assert_eq!(metrics.query_count, 0);
        assert_eq!(metrics.avg_latency_ms, 0.0);
    }

    #[test]
    fn test_single_sample_metrics() {
        let metrics = RetrievalMetrics::from_measurements(
            &[15.0],
            &[0.8],
            &[0.7],
            &[0.6],
            &[10],
        );

        assert_eq!(metrics.query_count, 1);
        assert!((metrics.avg_latency_ms - 15.0).abs() < 0.001);
        assert!((metrics.precision_at_k - 0.8).abs() < 0.001);
    }
}
