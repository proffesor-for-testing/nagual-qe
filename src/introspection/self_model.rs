//! Self-model types for Strange Loop introspection

use std::collections::HashMap;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Self-model of nagual's current state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModel {
    pub snapshot_at: DateTime<Utc>,
    pub pattern_health: PatternHealth,
    pub domain_coverage: HashMap<String, DomainMetrics>,
    pub temporal_trends: TemporalTrends,
    pub vulnerabilities: Vec<Vulnerability>,
    pub recommendations: Vec<Recommendation>,
}

/// Pattern health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternHealth {
    pub total_patterns: usize,
    pub high_reward_count: usize,      // reward >= 0.8
    pub medium_reward_count: usize,    // 0.4 <= reward < 0.8
    pub low_reward_count: usize,       // reward < 0.4
    pub average_reward: f64,
    pub average_effectiveness: f64,
    pub average_age_days: f64,
    pub stale_count: usize,            // >30 days, low usage
    pub orphan_count: usize,           // No graph connections
    pub with_embeddings: usize,        // Have vector embeddings
    pub total_reuse_count: usize,      // Total reuses across all patterns
}

impl Default for PatternHealth {
    fn default() -> Self {
        Self {
            total_patterns: 0,
            high_reward_count: 0,
            medium_reward_count: 0,
            low_reward_count: 0,
            average_reward: 0.0,
            average_effectiveness: 0.0,
            average_age_days: 0.0,
            stale_count: 0,
            orphan_count: 0,
            with_embeddings: 0,
            total_reuse_count: 0,
        }
    }
}

/// Domain-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainMetrics {
    pub domain: String,
    pub pattern_count: usize,
    pub coverage_score: f64,           // 0-1, estimated completeness
    pub avg_reward: f64,
    pub avg_effectiveness: f64,
    pub last_activity: Option<DateTime<Utc>>,
    pub gap_areas: Vec<String>,
    pub total_reuse_count: usize,
}

impl DomainMetrics {
    pub fn new(domain: String) -> Self {
        Self {
            domain,
            pattern_count: 0,
            coverage_score: 0.0,
            avg_reward: 0.0,
            avg_effectiveness: 0.0,
            last_activity: None,
            gap_areas: Vec::new(),
            total_reuse_count: 0,
        }
    }
}

/// Temporal trends analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalTrends {
    pub reward_trend_7d: TrendDirection,
    pub reward_trend_30d: TrendDirection,
    pub pattern_growth_rate: f64,      // patterns/week
    pub decay_rate: f64,               // patterns becoming stale/week
    pub patterns_created_7d: usize,
    pub patterns_created_30d: usize,
    pub avg_reward_7d: f64,
    pub avg_reward_30d: f64,
    pub estimated_time_to_stale: Option<Duration>, // when avg quality drops below threshold
}

impl Default for TemporalTrends {
    fn default() -> Self {
        Self {
            reward_trend_7d: TrendDirection::Stable,
            reward_trend_30d: TrendDirection::Stable,
            pattern_growth_rate: 0.0,
            decay_rate: 0.0,
            patterns_created_7d: 0,
            patterns_created_30d: 0,
            avg_reward_7d: 0.0,
            avg_reward_30d: 0.0,
            estimated_time_to_stale: None,
        }
    }
}

/// Direction of a trend
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
}

impl std::fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrendDirection::Improving => write!(f, "improving"),
            TrendDirection::Stable => write!(f, "stable"),
            TrendDirection::Declining => write!(f, "declining"),
        }
    }
}

/// A detected vulnerability in the knowledge system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    pub id: String,
    pub category: VulnerabilityCategory,
    pub severity: Severity,
    pub description: String,
    pub affected_domains: Vec<String>,
    pub estimated_impact: f64,         // 0-1
}

/// Categories of vulnerabilities
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum VulnerabilityCategory {
    KnowledgeDecay,      // Patterns becoming stale
    CoverageGap,         // Missing knowledge areas
    QualityDegradation,  // Declining rewards
    Fragmentation,       // Disconnected patterns
    Overspecialization,  // Too narrow domain focus
    LowEmbeddingCoverage, // Many patterns without embeddings
}

impl std::fmt::Display for VulnerabilityCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulnerabilityCategory::KnowledgeDecay => write!(f, "knowledge_decay"),
            VulnerabilityCategory::CoverageGap => write!(f, "coverage_gap"),
            VulnerabilityCategory::QualityDegradation => write!(f, "quality_degradation"),
            VulnerabilityCategory::Fragmentation => write!(f, "fragmentation"),
            VulnerabilityCategory::Overspecialization => write!(f, "overspecialization"),
            VulnerabilityCategory::LowEmbeddingCoverage => write!(f, "low_embedding_coverage"),
        }
    }
}

/// Severity levels for vulnerabilities
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Low => write!(f, "low"),
            Severity::Medium => write!(f, "medium"),
            Severity::High => write!(f, "high"),
            Severity::Critical => write!(f, "critical"),
        }
    }
}

/// A recommendation for self-improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub id: String,
    pub action: RecommendedAction,
    pub priority: u8,                  // 1-10
    pub reason: String,
    pub estimated_benefit: f64,        // 0-1
    pub goap_goal: Option<String>,     // Goal for GOAP planner
}

/// Types of recommended actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendedAction {
    RefreshStalePatterns { domain: String, count: usize },
    FillCoverageGap { domain: String, topic: String },
    ConsolidatePatterns { domain: String, threshold: f64 },
    ArchiveLowQuality { count: usize },
    ResearchTopic { topic: String },
    ReviewConflicts { pattern_ids: Vec<String> },
    GenerateEmbeddings { count: usize },
    ImproveTestCoverage { domain: String },
}

impl std::fmt::Display for RecommendedAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecommendedAction::RefreshStalePatterns { domain, count } => {
                write!(f, "Refresh {} stale patterns in '{}'", count, domain)
            }
            RecommendedAction::FillCoverageGap { domain, topic } => {
                write!(f, "Fill coverage gap in '{}' for topic '{}'", domain, topic)
            }
            RecommendedAction::ConsolidatePatterns { domain, threshold } => {
                write!(f, "Consolidate patterns in '{}' (threshold: {:.2})", domain, threshold)
            }
            RecommendedAction::ArchiveLowQuality { count } => {
                write!(f, "Archive {} low-quality patterns", count)
            }
            RecommendedAction::ResearchTopic { topic } => {
                write!(f, "Research topic: {}", topic)
            }
            RecommendedAction::ReviewConflicts { pattern_ids } => {
                write!(f, "Review {} conflicting patterns", pattern_ids.len())
            }
            RecommendedAction::GenerateEmbeddings { count } => {
                write!(f, "Generate embeddings for {} patterns", count)
            }
            RecommendedAction::ImproveTestCoverage { domain } => {
                write!(f, "Improve test coverage for '{}'", domain)
            }
        }
    }
}

/// Configuration for introspection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrospectionConfig {
    pub stale_threshold_days: u64,
    pub low_reward_threshold: f64,
    pub high_reward_threshold: f64,
    pub min_coverage_score: f64,
    pub min_embedding_coverage: f64,
    pub trend_window_days: u64,
}

impl Default for IntrospectionConfig {
    fn default() -> Self {
        Self {
            stale_threshold_days: 30,
            low_reward_threshold: 0.4,
            high_reward_threshold: 0.8,
            min_coverage_score: 0.6,
            min_embedding_coverage: 0.8,
            trend_window_days: 7,
        }
    }
}

/// Summary health status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Degraded,
    Critical,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Warning => write!(f, "warning"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Critical => write!(f, "critical"),
        }
    }
}

impl SelfModel {
    /// Calculate overall health status from vulnerabilities
    pub fn health_status(&self) -> HealthStatus {
        let critical_count = self.vulnerabilities.iter()
            .filter(|v| matches!(v.severity, Severity::Critical))
            .count();
        let high_count = self.vulnerabilities.iter()
            .filter(|v| matches!(v.severity, Severity::High))
            .count();
        let medium_count = self.vulnerabilities.iter()
            .filter(|v| matches!(v.severity, Severity::Medium))
            .count();

        if critical_count > 0 {
            HealthStatus::Critical
        } else if high_count > 1 {
            HealthStatus::Degraded
        } else if high_count > 0 || medium_count > 2 {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        }
    }

    /// Get top N recommendations by priority
    pub fn top_recommendations(&self, n: usize) -> Vec<&Recommendation> {
        let mut recs: Vec<_> = self.recommendations.iter().collect();
        recs.sort_by(|a, b| b.priority.cmp(&a.priority));
        recs.into_iter().take(n).collect()
    }

    /// Get vulnerabilities by severity
    pub fn vulnerabilities_by_severity(&self, severity: Severity) -> Vec<&Vulnerability> {
        self.vulnerabilities.iter()
            .filter(|v| v.severity == severity)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_healthy() {
        let model = SelfModel {
            snapshot_at: Utc::now(),
            pattern_health: PatternHealth::default(),
            domain_coverage: HashMap::new(),
            temporal_trends: TemporalTrends::default(),
            vulnerabilities: vec![],
            recommendations: vec![],
        };
        assert_eq!(model.health_status(), HealthStatus::Healthy);
    }

    #[test]
    fn test_health_status_critical() {
        let model = SelfModel {
            snapshot_at: Utc::now(),
            pattern_health: PatternHealth::default(),
            domain_coverage: HashMap::new(),
            temporal_trends: TemporalTrends::default(),
            vulnerabilities: vec![Vulnerability {
                id: "1".into(),
                category: VulnerabilityCategory::KnowledgeDecay,
                severity: Severity::Critical,
                description: "Test".into(),
                affected_domains: vec![],
                estimated_impact: 0.9,
            }],
            recommendations: vec![],
        };
        assert_eq!(model.health_status(), HealthStatus::Critical);
    }

    #[test]
    fn test_trend_direction_display() {
        assert_eq!(format!("{}", TrendDirection::Improving), "improving");
        assert_eq!(format!("{}", TrendDirection::Stable), "stable");
        assert_eq!(format!("{}", TrendDirection::Declining), "declining");
    }

    #[test]
    fn test_config_default() {
        let config = IntrospectionConfig::default();
        assert_eq!(config.stale_threshold_days, 30);
        assert_eq!(config.low_reward_threshold, 0.4);
        assert_eq!(config.high_reward_threshold, 0.8);
    }
}
