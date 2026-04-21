//! Pattern statistics and analytics.
//!
//! This module provides comprehensive statistics about stored patterns:
//! - Total counts and breakdowns by domain
//! - Success rates and average rewards
//! - Top performing patterns
//! - Reuse count distribution
//!
//! # Example
//!
//! ```ignore
//! use nagual::reasoning_bank::{get_pattern_stats, StatsConfig};
//!
//! let stats = get_pattern_stats(&patterns, &StatsConfig::default())?;
//! println!("Total patterns: {}", stats.total_patterns);
//! println!("Average reward: {:.2}", stats.average_reward);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{domain, Pattern, ReasoningBankResult};

/// Configuration for statistics calculation.
#[derive(Debug, Clone)]
pub struct StatsConfig {
    /// Number of top patterns to include.
    pub top_patterns_count: usize,

    /// Whether to compute domain hierarchy stats.
    pub include_domain_hierarchy: bool,

    /// Number of buckets for reuse distribution.
    pub reuse_distribution_buckets: usize,

    /// Minimum patterns for a domain to be included in stats.
    pub min_domain_patterns: usize,
}

impl Default for StatsConfig {
    fn default() -> Self {
        Self {
            top_patterns_count: 10,
            include_domain_hierarchy: true,
            reuse_distribution_buckets: 10,
            min_domain_patterns: 1,
        }
    }
}

impl StatsConfig {
    /// Set the number of top patterns to include.
    pub fn with_top_patterns(mut self, count: usize) -> Self {
        self.top_patterns_count = count;
        self
    }

    /// Disable domain hierarchy computation.
    pub fn without_domain_hierarchy(mut self) -> Self {
        self.include_domain_hierarchy = false;
        self
    }
}

/// Comprehensive statistics about patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStats {
    /// Total number of patterns.
    pub total_patterns: usize,

    /// Number of patterns with embeddings.
    pub patterns_with_embeddings: usize,

    /// Average reward score across all patterns.
    pub average_reward: f32,

    /// Average confidence score.
    pub average_confidence: f32,

    /// Overall success rate (patterns with success_rate > 0.5).
    pub success_rate: f32,

    /// Number of successful patterns.
    pub successful_patterns: usize,

    /// Total usage count across all patterns.
    pub total_usage_count: u64,

    /// Average usage count per pattern.
    pub average_usage_count: f32,

    /// Statistics by domain.
    pub domains: Vec<DomainStats>,

    /// Top performing patterns by reward.
    pub top_by_reward: Vec<TopPattern>,

    /// Top performing patterns by usage count.
    pub top_by_usage: Vec<TopPattern>,

    /// Top performing patterns by success rate.
    pub top_by_success_rate: Vec<TopPattern>,

    /// Reuse count distribution.
    pub reuse_distribution: ReuseDistribution,

    /// Reward score distribution (10 buckets from 0.0 to 1.0).
    pub reward_distribution: Vec<u32>,

    /// Timestamp when stats were computed.
    pub computed_at: chrono::DateTime<chrono::Utc>,
}

/// Statistics for a specific domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainStats {
    /// Domain name.
    pub domain: String,

    /// Number of patterns in this domain.
    pub pattern_count: usize,

    /// Average reward for this domain.
    pub average_reward: f32,

    /// Average confidence for this domain.
    pub average_confidence: f32,

    /// Success rate for this domain.
    pub success_rate: f32,

    /// Total usage count for this domain.
    pub total_usage: u64,

    /// Child domains (for hierarchy).
    pub children: Vec<DomainStats>,
}

impl DomainStats {
    /// Create a new domain stats instance.
    fn new(domain: String) -> Self {
        Self {
            domain,
            pattern_count: 0,
            average_reward: 0.0,
            average_confidence: 0.0,
            success_rate: 0.0,
            total_usage: 0,
            children: Vec::new(),
        }
    }
}

/// A top-performing pattern summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopPattern {
    /// Pattern ID.
    pub id: String,

    /// Brief problem description (truncated).
    pub problem_summary: String,

    /// Domain.
    pub domain: String,

    /// The metric value that makes this pattern top.
    pub metric_value: f32,

    /// Reward score.
    pub reward: f32,

    /// Usage count.
    pub usage_count: u32,

    /// Success rate.
    pub success_rate: f32,
}

impl TopPattern {
    fn from_pattern(pattern: &Pattern, metric_value: f32) -> Self {
        Self {
            id: pattern.id.clone(),
            problem_summary: truncate_string(&pattern.problem, 50),
            domain: pattern.domain.clone(),
            metric_value,
            reward: pattern.reward,
            usage_count: pattern.usage_count,
            success_rate: pattern.success_rate,
        }
    }
}

/// Distribution of reuse counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReuseDistribution {
    /// Bucket labels (e.g., "0", "1-5", "6-10", etc.).
    pub labels: Vec<String>,

    /// Count of patterns in each bucket.
    pub counts: Vec<u32>,

    /// Percentage of patterns in each bucket.
    pub percentages: Vec<f32>,

    /// Minimum reuse count.
    pub min: u32,

    /// Maximum reuse count.
    pub max: u32,

    /// Median reuse count.
    pub median: u32,

    /// Mean reuse count.
    pub mean: f32,
}

/// Calculate comprehensive statistics for a set of patterns.
pub fn get_pattern_stats(
    patterns: &[Pattern],
    config: &StatsConfig,
) -> ReasoningBankResult<PatternStats> {
    let total_patterns = patterns.len();

    if total_patterns == 0 {
        return Ok(PatternStats {
            total_patterns: 0,
            patterns_with_embeddings: 0,
            average_reward: 0.0,
            average_confidence: 0.0,
            success_rate: 0.0,
            successful_patterns: 0,
            total_usage_count: 0,
            average_usage_count: 0.0,
            domains: Vec::new(),
            top_by_reward: Vec::new(),
            top_by_usage: Vec::new(),
            top_by_success_rate: Vec::new(),
            reuse_distribution: ReuseDistribution {
                labels: Vec::new(),
                counts: Vec::new(),
                percentages: Vec::new(),
                min: 0,
                max: 0,
                median: 0,
                mean: 0.0,
            },
            reward_distribution: vec![0; 10],
            computed_at: chrono::Utc::now(),
        });
    }

    // Basic aggregations
    let patterns_with_embeddings = patterns.iter().filter(|p| p.embedding.is_some()).count();
    let total_reward: f32 = patterns.iter().map(|p| p.reward).sum();
    let total_confidence: f32 = patterns.iter().map(|p| p.confidence).sum();
    let successful_patterns = patterns
        .iter()
        .filter(|p| p.success_rate > 0.5)
        .count();
    let total_usage_count: u64 = patterns.iter().map(|p| p.usage_count as u64).sum();

    let average_reward = total_reward / total_patterns as f32;
    let average_confidence = total_confidence / total_patterns as f32;
    let success_rate = successful_patterns as f32 / total_patterns as f32;
    let average_usage_count = total_usage_count as f32 / total_patterns as f32;

    // Domain statistics
    let domains = compute_domain_stats(patterns, config);

    // Top patterns
    let top_by_reward = compute_top_patterns(patterns, config.top_patterns_count, |p| p.reward);
    let top_by_usage =
        compute_top_patterns(patterns, config.top_patterns_count, |p| p.usage_count as f32);
    let top_by_success_rate =
        compute_top_patterns(patterns, config.top_patterns_count, |p| p.success_rate);

    // Reuse distribution
    let reuse_distribution = compute_reuse_distribution(patterns, config.reuse_distribution_buckets);

    // Reward distribution (10 buckets from 0.0 to 1.0)
    let reward_distribution = compute_reward_distribution(patterns);

    Ok(PatternStats {
        total_patterns,
        patterns_with_embeddings,
        average_reward,
        average_confidence,
        success_rate,
        successful_patterns,
        total_usage_count,
        average_usage_count,
        domains,
        top_by_reward,
        top_by_usage,
        top_by_success_rate,
        reuse_distribution,
        reward_distribution,
        computed_at: chrono::Utc::now(),
    })
}

/// Compute statistics grouped by domain.
fn compute_domain_stats(patterns: &[Pattern], config: &StatsConfig) -> Vec<DomainStats> {
    // Group patterns by domain
    let mut domain_patterns: HashMap<String, Vec<&Pattern>> = HashMap::new();

    for pattern in patterns {
        // Add to exact domain
        domain_patterns
            .entry(pattern.domain.clone())
            .or_default()
            .push(pattern);

        // Add to parent domains if hierarchy is enabled
        if config.include_domain_hierarchy {
            let hierarchy = domain::parse_hierarchy(&pattern.domain);
            for ancestor in &hierarchy[..hierarchy.len().saturating_sub(1)] {
                domain_patterns
                    .entry(ancestor.clone())
                    .or_default()
                    .push(pattern);
            }
        }
    }

    // Compute stats for each domain
    let mut stats: Vec<DomainStats> = domain_patterns
        .into_iter()
        .filter(|(_, patterns)| patterns.len() >= config.min_domain_patterns)
        .map(|(domain, patterns)| {
            let count = patterns.len();
            let total_reward: f32 = patterns.iter().map(|p| p.reward).sum();
            let total_confidence: f32 = patterns.iter().map(|p| p.confidence).sum();
            let successful = patterns.iter().filter(|p| p.success_rate > 0.5).count();
            let total_usage: u64 = patterns.iter().map(|p| p.usage_count as u64).sum();

            DomainStats {
                domain,
                pattern_count: count,
                average_reward: total_reward / count as f32,
                average_confidence: total_confidence / count as f32,
                success_rate: successful as f32 / count as f32,
                total_usage,
                children: Vec::new(),
            }
        })
        .collect();

    // Sort by pattern count descending
    stats.sort_by(|a, b| b.pattern_count.cmp(&a.pattern_count));

    stats
}

/// Compute top patterns by a metric.
fn compute_top_patterns<F>(patterns: &[Pattern], count: usize, metric: F) -> Vec<TopPattern>
where
    F: Fn(&Pattern) -> f32,
{
    let mut scored: Vec<(&Pattern, f32)> = patterns.iter().map(|p| (p, metric(p))).collect();

    // Sort by metric descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored
        .into_iter()
        .take(count)
        .map(|(p, m)| TopPattern::from_pattern(p, m))
        .collect()
}

/// Compute the reuse count distribution.
fn compute_reuse_distribution(patterns: &[Pattern], bucket_count: usize) -> ReuseDistribution {
    if patterns.is_empty() {
        return ReuseDistribution {
            labels: Vec::new(),
            counts: Vec::new(),
            percentages: Vec::new(),
            min: 0,
            max: 0,
            median: 0,
            mean: 0.0,
        };
    }

    let mut counts: Vec<u32> = patterns.iter().map(|p| p.usage_count).collect();
    counts.sort();

    let min = *counts.first().unwrap_or(&0);
    let max = *counts.last().unwrap_or(&0);
    let median = counts[counts.len() / 2];
    let mean = counts.iter().sum::<u32>() as f32 / counts.len() as f32;

    // Create buckets
    let bucket_size = ((max - min) as f32 / bucket_count as f32).ceil().max(1.0) as u32;
    let mut bucket_counts = vec![0u32; bucket_count];
    let mut labels = Vec::with_capacity(bucket_count);

    for i in 0..bucket_count {
        let start = min + (i as u32) * bucket_size;
        let end = if i == bucket_count - 1 {
            max
        } else {
            start + bucket_size - 1
        };

        labels.push(if start == end {
            format!("{}", start)
        } else {
            format!("{}-{}", start, end)
        });
    }

    // Count patterns in each bucket
    for &count in &counts {
        let bucket_idx = if max == min {
            0
        } else {
            (((count - min) as f32 / bucket_size as f32).floor() as usize).min(bucket_count - 1)
        };
        bucket_counts[bucket_idx] += 1;
    }

    // Compute percentages
    let total = counts.len() as f32;
    let percentages: Vec<f32> = bucket_counts
        .iter()
        .map(|&c| (c as f32 / total) * 100.0)
        .collect();

    ReuseDistribution {
        labels,
        counts: bucket_counts,
        percentages,
        min,
        max,
        median,
        mean,
    }
}

/// Compute reward score distribution (10 buckets from 0.0 to 1.0).
fn compute_reward_distribution(patterns: &[Pattern]) -> Vec<u32> {
    let mut distribution = vec![0u32; 10];

    for pattern in patterns {
        let bucket = ((pattern.reward * 10.0).floor() as usize).min(9);
        distribution[bucket] += 1;
    }

    distribution
}

/// Truncate a string to a maximum length.
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_patterns() -> Vec<Pattern> {
        vec![
            Pattern::new("Error handling", "Use Result", "rust.error")
                .with_reward(0.9)
                .with_confidence(0.85),
            Pattern::new("Async programming", "Use tokio", "rust.async")
                .with_reward(0.8)
                .with_confidence(0.9),
            Pattern::new("Database connection", "Use pool", "database.postgres")
                .with_reward(0.7)
                .with_confidence(0.75),
            Pattern::new("API design", "Use REST", "api.rest")
                .with_reward(0.6)
                .with_confidence(0.8),
            Pattern::new("Testing", "Use unit tests", "testing")
                .with_reward(0.5)
                .with_confidence(0.7),
        ]
    }

    #[test]
    fn test_get_pattern_stats_basic() {
        let patterns = create_test_patterns();
        let config = StatsConfig::default();

        let stats = get_pattern_stats(&patterns, &config).unwrap();

        assert_eq!(stats.total_patterns, 5);
        assert!((stats.average_reward - 0.7).abs() < 0.01);
        assert!((stats.average_confidence - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_get_pattern_stats_empty() {
        let patterns: Vec<Pattern> = Vec::new();
        let config = StatsConfig::default();

        let stats = get_pattern_stats(&patterns, &config).unwrap();

        assert_eq!(stats.total_patterns, 0);
        assert_eq!(stats.average_reward, 0.0);
    }

    #[test]
    fn test_domain_stats() {
        let patterns = create_test_patterns();
        let config = StatsConfig::default();

        let stats = get_pattern_stats(&patterns, &config).unwrap();

        // Should have domains
        assert!(!stats.domains.is_empty());

        // Should have rust domain stats
        let rust_domains: Vec<_> = stats
            .domains
            .iter()
            .filter(|d| d.domain.starts_with("rust"))
            .collect();
        assert!(!rust_domains.is_empty());
    }

    #[test]
    fn test_domain_hierarchy_stats() {
        let patterns = vec![
            Pattern::new("A", "B", "rust.async.tokio").with_reward(0.9),
            Pattern::new("C", "D", "rust.async.async-std").with_reward(0.8),
            Pattern::new("E", "F", "rust.error").with_reward(0.7),
        ];

        let config = StatsConfig::default();
        let stats = get_pattern_stats(&patterns, &config).unwrap();

        // Should have "rust" domain aggregating all rust.* patterns
        let rust_domain = stats.domains.iter().find(|d| d.domain == "rust");
        assert!(rust_domain.is_some());
        assert_eq!(rust_domain.unwrap().pattern_count, 3);

        // Should have "rust.async" aggregating both tokio and async-std
        let rust_async = stats.domains.iter().find(|d| d.domain == "rust.async");
        assert!(rust_async.is_some());
        assert_eq!(rust_async.unwrap().pattern_count, 2);
    }

    #[test]
    fn test_top_patterns_by_reward() {
        let patterns = create_test_patterns();
        let config = StatsConfig::default().with_top_patterns(3);

        let stats = get_pattern_stats(&patterns, &config).unwrap();

        assert_eq!(stats.top_by_reward.len(), 3);
        // First should be highest reward
        assert_eq!(stats.top_by_reward[0].reward, 0.9);
    }

    #[test]
    fn test_top_patterns_by_usage() {
        let mut patterns = create_test_patterns();
        patterns[2].usage_count = 100;
        patterns[0].usage_count = 50;

        let config = StatsConfig::default().with_top_patterns(3);
        let stats = get_pattern_stats(&patterns, &config).unwrap();

        // First should be highest usage
        assert_eq!(stats.top_by_usage[0].usage_count, 100);
    }

    #[test]
    fn test_reuse_distribution() {
        let mut patterns = create_test_patterns();
        patterns[0].usage_count = 0;
        patterns[1].usage_count = 5;
        patterns[2].usage_count = 10;
        patterns[3].usage_count = 50;
        patterns[4].usage_count = 100;

        let config = StatsConfig::default();
        let stats = get_pattern_stats(&patterns, &config).unwrap();

        assert_eq!(stats.reuse_distribution.min, 0);
        assert_eq!(stats.reuse_distribution.max, 100);
        assert!(!stats.reuse_distribution.labels.is_empty());
    }

    #[test]
    fn test_reward_distribution() {
        let patterns = create_test_patterns();
        let config = StatsConfig::default();

        let stats = get_pattern_stats(&patterns, &config).unwrap();

        // Should have 10 buckets
        assert_eq!(stats.reward_distribution.len(), 10);
        // Sum should equal total patterns
        let sum: u32 = stats.reward_distribution.iter().sum();
        assert_eq!(sum, 5);
    }

    #[test]
    fn test_success_rate() {
        let mut patterns = create_test_patterns();
        patterns[0].success_rate = 0.9; // Successful
        patterns[1].success_rate = 0.6; // Successful
        patterns[2].success_rate = 0.4; // Not successful
        patterns[3].success_rate = 0.3; // Not successful
        patterns[4].success_rate = 0.8; // Successful

        let config = StatsConfig::default();
        let stats = get_pattern_stats(&patterns, &config).unwrap();

        assert_eq!(stats.successful_patterns, 3);
        assert!((stats.success_rate - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_stats_config_builder() {
        let config = StatsConfig::default()
            .with_top_patterns(5)
            .without_domain_hierarchy();

        assert_eq!(config.top_patterns_count, 5);
        assert!(!config.include_domain_hierarchy);
    }

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("Hello", 10), "Hello");
        assert_eq!(truncate_string("Hello World", 8), "Hello...");
    }

    #[test]
    fn test_patterns_with_embeddings() {
        let mut patterns = create_test_patterns();
        patterns[0].embedding = Some(ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0]));
        patterns[1].embedding = Some(ndarray::Array1::from_vec(vec![4.0, 5.0, 6.0]));

        let config = StatsConfig::default();
        let stats = get_pattern_stats(&patterns, &config).unwrap();

        assert_eq!(stats.patterns_with_embeddings, 2);
    }

    #[test]
    fn test_total_usage_count() {
        let mut patterns = create_test_patterns();
        patterns[0].usage_count = 10;
        patterns[1].usage_count = 20;
        patterns[2].usage_count = 30;

        let config = StatsConfig::default();
        let stats = get_pattern_stats(&patterns, &config).unwrap();

        assert_eq!(stats.total_usage_count, 60);
        assert!((stats.average_usage_count - 12.0).abs() < 0.01);
    }
}
