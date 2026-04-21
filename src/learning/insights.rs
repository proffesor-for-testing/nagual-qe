//! Domain insights aggregation for pattern analysis.
//!
//! This module provides time-windowed analysis of pattern performance by domain,
//! including trend detection (improving/stable/declining) and top pattern tracking.
//!
//! # Features
//!
//! - Success rate by domain
//! - Average reward by domain
//! - Top patterns per domain
//! - Trend analysis (7, 30, 90 day windows)
//!
//! # Example
//!
//! ```ignore
//! use nagual::learning::{aggregate_insights, InsightsConfig, TimeWindow};
//!
//! let config = InsightsConfig::default()
//!     .with_time_windows(vec![TimeWindow::Days7, TimeWindow::Days30]);
//!
//! let insights = aggregate_insights(&patterns, "rust.async", &config)?;
//!
//! println!("Domain: {}", insights.domain);
//! println!("Success rate: {:.1}%", insights.success_rate * 100.0);
//! println!("Trend: {:?}", insights.trend);
//!
//! for top in &insights.top_patterns {
//!     println!("Top pattern: {} (reward: {:.2})", top.id, top.reward);
//! }
//! ```

use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::reasoning_bank::pattern::{Pattern, PatternId};

/// Time window for analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeWindow {
    /// Last 24 hours.
    Hours24,

    /// Last 7 days.
    Days7,

    /// Last 30 days.
    Days30,

    /// Last 90 days.
    Days90,

    /// Last 365 days.
    Days365,

    /// All time (no limit).
    AllTime,

    /// Custom duration in hours.
    Custom(u64),
}

impl TimeWindow {
    /// Convert to a Duration.
    pub fn to_duration(&self) -> Option<Duration> {
        match self {
            TimeWindow::Hours24 => Some(Duration::hours(24)),
            TimeWindow::Days7 => Some(Duration::days(7)),
            TimeWindow::Days30 => Some(Duration::days(30)),
            TimeWindow::Days90 => Some(Duration::days(90)),
            TimeWindow::Days365 => Some(Duration::days(365)),
            TimeWindow::AllTime => None,
            TimeWindow::Custom(hours) => Some(Duration::hours(*hours as i64)),
        }
    }

    /// Get the cutoff datetime for this window.
    pub fn cutoff(&self) -> Option<DateTime<Utc>> {
        self.to_duration().map(|d| Utc::now() - d)
    }

    /// Get the label for this window.
    pub fn label(&self) -> String {
        match self {
            TimeWindow::Hours24 => "24h".to_string(),
            TimeWindow::Days7 => "7d".to_string(),
            TimeWindow::Days30 => "30d".to_string(),
            TimeWindow::Days90 => "90d".to_string(),
            TimeWindow::Days365 => "365d".to_string(),
            TimeWindow::AllTime => "all".to_string(),
            TimeWindow::Custom(h) => format!("{}h", h),
        }
    }
}

impl Default for TimeWindow {
    fn default() -> Self {
        TimeWindow::Days30
    }
}

/// Overall trend direction for a domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Trend {
    /// Performance is improving.
    Improving,

    /// Performance is stable.
    Stable,

    /// Performance is declining.
    Declining,

    /// Not enough data to determine trend.
    Unknown,
}

impl std::fmt::Display for Trend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Trend::Improving => write!(f, "improving"),
            Trend::Stable => write!(f, "stable"),
            Trend::Declining => write!(f, "declining"),
            Trend::Unknown => write!(f, "unknown"),
        }
    }
}

/// Trend analysis for a specific time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// The time window analyzed.
    pub window: TimeWindow,

    /// Average reward in this window.
    pub avg_reward: f32,

    /// Average success rate in this window.
    pub avg_success_rate: f32,

    /// Number of patterns in this window.
    pub pattern_count: usize,

    /// Number of patterns created in this window.
    pub new_patterns: usize,

    /// Number of patterns updated in this window.
    pub updated_patterns: usize,

    /// Total usage in this window.
    pub total_usage: u64,
}

impl TrendAnalysis {
    /// Create empty trend analysis for a window.
    pub fn empty(window: TimeWindow) -> Self {
        Self {
            window,
            avg_reward: 0.0,
            avg_success_rate: 0.0,
            pattern_count: 0,
            new_patterns: 0,
            updated_patterns: 0,
            total_usage: 0,
        }
    }
}

/// Information about a top-performing pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopPatternInfo {
    /// Pattern ID.
    pub id: PatternId,

    /// Problem summary (truncated).
    pub problem_summary: String,

    /// Reward score.
    pub reward: f32,

    /// Usage count.
    pub usage_count: u32,

    /// Success rate.
    pub success_rate: f32,

    /// Effectiveness score.
    pub effectiveness: f32,

    /// Quality score (combined).
    pub quality_score: f32,

    /// When last updated.
    pub updated_at: DateTime<Utc>,
}

impl TopPatternInfo {
    /// Create from a Pattern.
    pub fn from_pattern(pattern: &Pattern) -> Self {
        Self {
            id: pattern.id().clone(),
            problem_summary: truncate(&pattern.problem(), 80),
            reward: pattern.reward(),
            usage_count: pattern.reuse_count(),
            success_rate: if pattern.success() { 1.0 } else { 0.0 },
            effectiveness: pattern.effectiveness(),
            quality_score: pattern.quality_score(),
            updated_at: pattern.updated_at(),
        }
    }
}

/// Trend over multiple time periods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTrend {
    /// Reward change percentage (positive = improving).
    pub reward_change_pct: f32,

    /// Success rate change percentage.
    pub success_rate_change_pct: f32,

    /// Usage growth percentage.
    pub usage_growth_pct: f32,

    /// New patterns added.
    pub new_patterns: usize,

    /// Comparison window start.
    pub comparison_window: TimeWindow,
}

impl PatternTrend {
    /// Calculate the overall trend direction.
    pub fn direction(&self) -> Trend {
        // Weight the metrics
        let weighted_change = self.reward_change_pct * 0.4
            + self.success_rate_change_pct * 0.3
            + self.usage_growth_pct.min(50.0) * 0.3 / 50.0; // Cap usage growth contribution

        if weighted_change > 5.0 {
            Trend::Improving
        } else if weighted_change < -5.0 {
            Trend::Declining
        } else {
            Trend::Stable
        }
    }
}

/// Configuration for insights aggregation.
#[derive(Debug, Clone)]
pub struct InsightsConfig {
    /// Time windows to analyze.
    pub time_windows: Vec<TimeWindow>,

    /// Number of top patterns to include.
    pub top_patterns_count: usize,

    /// Minimum patterns for reliable insights.
    pub min_patterns: usize,

    /// Minimum reward to consider successful.
    pub success_threshold: f32,

    /// Whether to include child domain patterns.
    pub include_child_domains: bool,

    /// Threshold for trend detection (percentage change).
    pub trend_threshold_pct: f32,
}

impl Default for InsightsConfig {
    fn default() -> Self {
        Self {
            time_windows: vec![
                TimeWindow::Days7,
                TimeWindow::Days30,
                TimeWindow::Days90,
            ],
            top_patterns_count: 10,
            min_patterns: 3,
            success_threshold: 0.6,
            include_child_domains: true,
            trend_threshold_pct: 5.0,
        }
    }
}

impl InsightsConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set time windows.
    pub fn with_time_windows(mut self, windows: Vec<TimeWindow>) -> Self {
        self.time_windows = windows;
        self
    }

    /// Set top patterns count.
    pub fn with_top_patterns_count(mut self, count: usize) -> Self {
        self.top_patterns_count = count;
        self
    }

    /// Set minimum patterns for reliable insights.
    pub fn with_min_patterns(mut self, min: usize) -> Self {
        self.min_patterns = min;
        self
    }

    /// Set whether to include child domains.
    pub fn with_child_domains(mut self, include: bool) -> Self {
        self.include_child_domains = include;
        self
    }
}

/// Aggregated insights for a domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainInsights {
    /// The domain these insights are for.
    pub domain: String,

    /// Overall success rate (0.0 - 1.0).
    pub success_rate: f32,

    /// Overall average reward.
    pub avg_reward: f32,

    /// Overall average effectiveness.
    pub avg_effectiveness: f32,

    /// Overall average confidence.
    pub avg_confidence: f32,

    /// Total patterns in this domain.
    pub total_patterns: usize,

    /// Patterns with reliable metrics.
    pub patterns_with_reliable_metrics: usize,

    /// Total usage across all patterns.
    pub total_usage: u64,

    /// Top performing patterns.
    pub top_patterns: Vec<TopPatternInfo>,

    /// Analysis by time window.
    pub window_analysis: HashMap<String, TrendAnalysis>,

    /// Overall trend direction.
    pub trend: Trend,

    /// Detailed trend analysis.
    pub trend_details: Option<PatternTrend>,

    /// When these insights were generated.
    pub generated_at: DateTime<Utc>,

    /// Child domain breakdown (if any).
    pub child_domains: Vec<ChildDomainSummary>,
}

/// Summary for a child domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChildDomainSummary {
    /// Child domain name.
    pub domain: String,

    /// Pattern count.
    pub pattern_count: usize,

    /// Average reward.
    pub avg_reward: f32,

    /// Overall trend.
    pub trend: Trend,
}

/// Aggregate insights for patterns in a given domain.
///
/// This function analyzes patterns belonging to the specified domain and returns
/// comprehensive insights including success rates, averages, trends, and top patterns.
///
/// # Arguments
///
/// * `patterns` - All patterns to analyze
/// * `domain` - Domain to filter by (use empty string for all domains)
/// * `config` - Configuration for insights aggregation
///
/// # Returns
///
/// `DomainInsights` containing aggregated statistics and trends.
///
/// # Example
///
/// ```ignore
/// let insights = aggregate_insights(&patterns, "rust.async", &InsightsConfig::default())?;
/// println!("Avg reward: {:.2}", insights.avg_reward);
/// ```
pub fn aggregate_insights(
    patterns: &[Pattern],
    domain: &str,
    config: &InsightsConfig,
) -> DomainInsights {
    // Filter patterns by domain
    let filtered_patterns: Vec<&Pattern> = if domain.is_empty() {
        patterns.iter().collect()
    } else if config.include_child_domains {
        patterns
            .iter()
            .filter(|p| {
                let cat = p.category().to_string();
                cat == domain || cat.starts_with(&format!("{}.", domain))
            })
            .collect()
    } else {
        patterns
            .iter()
            .filter(|p| p.category().to_string() == domain)
            .collect()
    };

    let total_patterns = filtered_patterns.len();

    // Handle empty case
    if total_patterns == 0 {
        return DomainInsights {
            domain: domain.to_string(),
            success_rate: 0.0,
            avg_reward: 0.0,
            avg_effectiveness: 0.0,
            avg_confidence: 0.0,
            total_patterns: 0,
            patterns_with_reliable_metrics: 0,
            total_usage: 0,
            top_patterns: Vec::new(),
            window_analysis: HashMap::new(),
            trend: Trend::Unknown,
            trend_details: None,
            generated_at: Utc::now(),
            child_domains: Vec::new(),
        };
    }

    // Calculate overall metrics
    let total_reward: f32 = filtered_patterns.iter().map(|p| p.reward()).sum();
    let total_effectiveness: f32 = filtered_patterns.iter().map(|p| p.effectiveness()).sum();
    let total_confidence: f32 = filtered_patterns.iter().map(|p| p.confidence()).sum();
    let total_usage: u64 = filtered_patterns.iter().map(|p| p.reuse_count() as u64).sum();

    let avg_reward = total_reward / total_patterns as f32;
    let avg_effectiveness = total_effectiveness / total_patterns as f32;
    let avg_confidence = total_confidence / total_patterns as f32;

    // Calculate success rate
    let successful = filtered_patterns
        .iter()
        .filter(|p| p.reward() >= config.success_threshold)
        .count();
    let success_rate = successful as f32 / total_patterns as f32;

    // Count patterns with reliable metrics (used at least once)
    let patterns_with_reliable_metrics = filtered_patterns
        .iter()
        .filter(|p| p.reuse_count() >= 1)
        .count();

    // Get top patterns
    let mut sorted_patterns: Vec<_> = filtered_patterns.iter().collect();
    sorted_patterns.sort_by(|a, b| {
        b.quality_score()
            .partial_cmp(&a.quality_score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let top_patterns: Vec<TopPatternInfo> = sorted_patterns
        .iter()
        .take(config.top_patterns_count)
        .map(|p| TopPatternInfo::from_pattern(p))
        .collect();

    // Analyze by time window
    let mut window_analysis = HashMap::new();
    for window in &config.time_windows {
        let analysis = analyze_window(&filtered_patterns, *window, config);
        window_analysis.insert(window.label(), analysis);
    }

    // Calculate trend
    let (trend, trend_details) = calculate_trend(&window_analysis, config);

    // Calculate child domain summaries
    let child_domains = calculate_child_domains(&filtered_patterns, domain, config);

    DomainInsights {
        domain: domain.to_string(),
        success_rate,
        avg_reward,
        avg_effectiveness,
        avg_confidence,
        total_patterns,
        patterns_with_reliable_metrics,
        total_usage,
        top_patterns,
        window_analysis,
        trend,
        trend_details,
        generated_at: Utc::now(),
        child_domains,
    }
}

/// Analyze patterns within a specific time window.
fn analyze_window(
    patterns: &[&Pattern],
    window: TimeWindow,
    config: &InsightsConfig,
) -> TrendAnalysis {
    let cutoff = window.cutoff();

    // Filter patterns by update time
    let window_patterns: Vec<_> = if let Some(cut) = cutoff {
        patterns
            .iter()
            .filter(|p| p.updated_at() >= cut)
            .cloned()
            .collect()
    } else {
        patterns.to_vec()
    };

    if window_patterns.is_empty() {
        return TrendAnalysis::empty(window);
    }

    let pattern_count = window_patterns.len();

    // Calculate averages
    let total_reward: f32 = window_patterns.iter().map(|p| p.reward()).sum();
    let avg_reward = total_reward / pattern_count as f32;

    let successful = window_patterns
        .iter()
        .filter(|p| p.reward() >= config.success_threshold)
        .count();
    let avg_success_rate = successful as f32 / pattern_count as f32;

    // Count new vs updated
    let new_patterns = if let Some(cut) = cutoff {
        window_patterns
            .iter()
            .filter(|p| p.timestamp() >= cut)
            .count()
    } else {
        pattern_count
    };

    let updated_patterns = pattern_count - new_patterns;

    let total_usage: u64 = window_patterns
        .iter()
        .map(|p| p.reuse_count() as u64)
        .sum();

    TrendAnalysis {
        window,
        avg_reward,
        avg_success_rate,
        pattern_count,
        new_patterns,
        updated_patterns,
        total_usage,
    }
}

/// Calculate trend from window analysis.
fn calculate_trend(
    window_analysis: &HashMap<String, TrendAnalysis>,
    config: &InsightsConfig,
) -> (Trend, Option<PatternTrend>) {
    // Need at least two windows to calculate trend
    let short_window = window_analysis.get("7d");
    let long_window = window_analysis.get("30d");

    match (short_window, long_window) {
        (Some(short), Some(long)) if short.pattern_count >= config.min_patterns && long.pattern_count >= config.min_patterns => {
            // Calculate changes
            let reward_change = if long.avg_reward > 0.0 {
                ((short.avg_reward - long.avg_reward) / long.avg_reward) * 100.0
            } else {
                0.0
            };

            let success_change = if long.avg_success_rate > 0.0 {
                ((short.avg_success_rate - long.avg_success_rate) / long.avg_success_rate) * 100.0
            } else {
                0.0
            };

            let usage_growth = if long.total_usage > 0 {
                ((short.total_usage as f32 - long.total_usage as f32) / long.total_usage as f32) * 100.0
            } else {
                0.0
            };

            let trend_details = PatternTrend {
                reward_change_pct: reward_change,
                success_rate_change_pct: success_change,
                usage_growth_pct: usage_growth,
                new_patterns: short.new_patterns,
                comparison_window: TimeWindow::Days7,
            };

            let trend = trend_details.direction();

            (trend, Some(trend_details))
        }
        _ => (Trend::Unknown, None),
    }
}

/// Calculate child domain summaries.
fn calculate_child_domains(
    patterns: &[&Pattern],
    parent_domain: &str,
    config: &InsightsConfig,
) -> Vec<ChildDomainSummary> {
    if parent_domain.is_empty() || !config.include_child_domains {
        return Vec::new();
    }

    // Group by immediate child domains
    let mut child_groups: HashMap<String, Vec<&Pattern>> = HashMap::new();
    let prefix = format!("{}.", parent_domain);

    for pattern in patterns {
        let cat = pattern.category().to_string();
        if cat.starts_with(&prefix) {
            // Extract immediate child
            let remainder = &cat[prefix.len()..];
            let child = if let Some(pos) = remainder.find('.') {
                format!("{}{}", prefix, &remainder[..pos])
            } else {
                cat.clone()
            };

            child_groups.entry(child).or_default().push(*pattern);
        }
    }

    // Calculate summary for each child
    let mut summaries: Vec<_> = child_groups
        .into_iter()
        .filter(|(_, patterns)| patterns.len() >= config.min_patterns)
        .map(|(domain, patterns)| {
            let avg_reward = patterns.iter().map(|p| p.reward()).sum::<f32>() / patterns.len() as f32;

            // Simple trend based on recency
            let recent_avg: f32 = patterns
                .iter()
                .filter(|p| p.updated_at() >= Utc::now() - Duration::days(7))
                .map(|p| p.reward())
                .collect::<Vec<_>>()
                .iter()
                .copied()
                .sum::<f32>()
                / patterns
                    .iter()
                    .filter(|p| p.updated_at() >= Utc::now() - Duration::days(7))
                    .count()
                    .max(1) as f32;

            let trend = if recent_avg > avg_reward * 1.05 {
                Trend::Improving
            } else if recent_avg < avg_reward * 0.95 {
                Trend::Declining
            } else {
                Trend::Stable
            };

            ChildDomainSummary {
                domain,
                pattern_count: patterns.len(),
                avg_reward,
                trend,
            }
        })
        .collect();

    // Sort by pattern count descending
    summaries.sort_by(|a, b| b.pattern_count.cmp(&a.pattern_count));

    summaries
}

/// Truncate a string to a maximum length.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reasoning_bank::pattern::PatternCategory;

    fn create_test_pattern(
        problem: &str,
        category: PatternCategory,
        reward: f32,
        reuse_count: u32,
    ) -> Pattern {
        Pattern::builder()
            .problem(problem)
            .solution("Test solution")
            .category(category)
            .reward(reward)
            .reuse_count(reuse_count)
            .effectiveness(reward)
            .confidence(reward)
            .build()
    }

    #[test]
    fn test_time_window_cutoff() {
        let window = TimeWindow::Days7;
        let cutoff = window.cutoff().unwrap();
        let now = Utc::now();

        assert!(now > cutoff);
        assert!(now - cutoff <= Duration::days(8)); // Allow for test execution time
    }

    #[test]
    fn test_time_window_label() {
        assert_eq!(TimeWindow::Hours24.label(), "24h");
        assert_eq!(TimeWindow::Days7.label(), "7d");
        assert_eq!(TimeWindow::Days30.label(), "30d");
        assert_eq!(TimeWindow::AllTime.label(), "all");
        assert_eq!(TimeWindow::Custom(48).label(), "48h");
    }

    #[test]
    fn test_trend_display() {
        assert_eq!(Trend::Improving.to_string(), "improving");
        assert_eq!(Trend::Stable.to_string(), "stable");
        assert_eq!(Trend::Declining.to_string(), "declining");
        assert_eq!(Trend::Unknown.to_string(), "unknown");
    }

    #[test]
    fn test_pattern_trend_direction() {
        let improving = PatternTrend {
            reward_change_pct: 10.0,
            success_rate_change_pct: 8.0,
            usage_growth_pct: 20.0,
            new_patterns: 5,
            comparison_window: TimeWindow::Days7,
        };
        assert_eq!(improving.direction(), Trend::Improving);

        let declining = PatternTrend {
            reward_change_pct: -10.0,
            success_rate_change_pct: -8.0,
            usage_growth_pct: -5.0,
            new_patterns: 0,
            comparison_window: TimeWindow::Days7,
        };
        assert_eq!(declining.direction(), Trend::Declining);

        let stable = PatternTrend {
            reward_change_pct: 1.0,
            success_rate_change_pct: -1.0,
            usage_growth_pct: 2.0,
            new_patterns: 1,
            comparison_window: TimeWindow::Days7,
        };
        assert_eq!(stable.direction(), Trend::Stable);
    }

    #[test]
    fn test_insights_config_default() {
        let config = InsightsConfig::default();
        assert_eq!(config.time_windows.len(), 3);
        assert_eq!(config.top_patterns_count, 10);
        assert!(config.include_child_domains);
    }

    #[test]
    fn test_insights_config_builder() {
        let config = InsightsConfig::new()
            .with_time_windows(vec![TimeWindow::Days7])
            .with_top_patterns_count(5)
            .with_child_domains(false);

        assert_eq!(config.time_windows.len(), 1);
        assert_eq!(config.top_patterns_count, 5);
        assert!(!config.include_child_domains);
    }

    #[test]
    fn test_aggregate_insights_empty() {
        let patterns: Vec<Pattern> = Vec::new();
        let insights = aggregate_insights(&patterns, "test", &InsightsConfig::default());

        assert_eq!(insights.total_patterns, 0);
        assert_eq!(insights.avg_reward, 0.0);
        assert_eq!(insights.trend, Trend::Unknown);
    }

    #[test]
    fn test_aggregate_insights_basic() {
        let patterns = vec![
            create_test_pattern("P1", PatternCategory::Testing, 0.8, 5),
            create_test_pattern("P2", PatternCategory::Testing, 0.6, 3),
            create_test_pattern("P3", PatternCategory::Testing, 0.9, 10),
        ];

        let insights = aggregate_insights(&patterns, "testing", &InsightsConfig::default());

        assert_eq!(insights.total_patterns, 3);
        assert!(insights.avg_reward > 0.7);
        assert!(!insights.top_patterns.is_empty());
    }

    #[test]
    fn test_aggregate_insights_filters_by_domain() {
        let patterns = vec![
            create_test_pattern("P1", PatternCategory::Testing, 0.8, 5),
            create_test_pattern("P2", PatternCategory::Performance, 0.6, 3),
            create_test_pattern("P3", PatternCategory::Security, 0.9, 10),
        ];

        let insights = aggregate_insights(&patterns, "testing", &InsightsConfig::default());

        assert_eq!(insights.total_patterns, 1);
    }

    #[test]
    fn test_aggregate_insights_all_domains() {
        let patterns = vec![
            create_test_pattern("P1", PatternCategory::Testing, 0.8, 5),
            create_test_pattern("P2", PatternCategory::Performance, 0.6, 3),
        ];

        let insights = aggregate_insights(&patterns, "", &InsightsConfig::default());

        assert_eq!(insights.total_patterns, 2);
    }

    #[test]
    fn test_top_pattern_info() {
        let pattern = create_test_pattern("Test problem description", PatternCategory::Testing, 0.85, 10);
        let info = TopPatternInfo::from_pattern(&pattern);

        assert_eq!(info.reward, 0.85);
        assert_eq!(info.usage_count, 10);
        assert!(!info.problem_summary.is_empty());
    }

    #[test]
    fn test_trend_analysis_empty() {
        let analysis = TrendAnalysis::empty(TimeWindow::Days7);

        assert_eq!(analysis.pattern_count, 0);
        assert_eq!(analysis.avg_reward, 0.0);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("Hello", 10), "Hello");
        assert_eq!(truncate("Hello World", 8), "Hello...");
        assert_eq!(truncate("Short", 5), "Short");
    }

    #[test]
    fn test_insights_success_rate() {
        let patterns = vec![
            create_test_pattern("P1", PatternCategory::Testing, 0.9, 5),  // Success
            create_test_pattern("P2", PatternCategory::Testing, 0.7, 3),  // Success
            create_test_pattern("P3", PatternCategory::Testing, 0.3, 10), // Failure
        ];

        let config = InsightsConfig::default(); // success_threshold = 0.6
        let insights = aggregate_insights(&patterns, "testing", &config);

        // 2 out of 3 are successful
        assert!((insights.success_rate - 0.666).abs() < 0.01);
    }
}
